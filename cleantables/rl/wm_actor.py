"""The world model as the planner: RL post-training of a trainable copy of the world model in physics.

The actor is a copy of the frozen world model (seed 0: trunk + output tokens + its own T2 flow decoder), built from the env state in
robot mode (3-D tokens, no video) with the two grippers in its hand tokens (their live palm positions). Heads on the output tokens:
  type        take / wait / move / done per gripper: head(hand token g, cls) + a 'done' prior tied to the frozen WM's STOP read-out
  object      log p_now (the actor's own read-out) + bilinear(hand token g, object tokens), zero-initialised so it starts as the WM's
              choice; masked by the gripper's IK reach, the other gripper's object, met / static objects
  grasp yaw   head(chosen object token, hand token) + a prior on the default short-axis yaw; release u, retreat: Gaussian heads
  move        head(hand token): a free target (x, y, z, yaw) in the workspace box
  route       head(hand token, target object token | cls for a free move): a 5-point spline edit (xyz, tanh * R_EDIT) of the hand's
              approach / free-move route (the zero edit = the over-route polyline / straight line)
  path        the actor's own T2 flow, made stochastic (Flow-GRPO, Liu et al., 2025; ReinFlow, Zhang et al., 2025):
              x_{i+1} = x_i + v(x_i, t_i) dt + sigma sqrt(dt) eps, with noise only in the chosen object's channel, giving an exact
              Gaussian log-prob per Euler step; the path = watch_path (the T2 shape around the straight line from the object to its target)
Trainable: the new heads, the T2 velocity decoder (full) and LoRA (r 8) on every trunk layer's MLP; the rest is frozen. Trained with GRPO
(group-relative advantages over rollouts of the same start table). The judge is a separate frozen 3-seed ensemble
(`cleantables.rl.chain_env.FrozenJudge`); the KL reference is the actor at init (the original WM + zero heads), closed form for the flow.
"""
from __future__ import annotations

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cleantables.rl.chain_env import GYAWS, N_RCTRL

TYPES = ("take", "wait", "move", "done")
STEPS = int(os.environ.get("FPW_STEPS", "12"))
EP_S = float(os.environ.get("FPR_EP_S", "90"))                  # episode time limit (s)
SIGMA = float(os.environ.get("FPW_SIGMA", "0.2"))          # exploration noise in the flow (PS units; PS = 0.25 m, so ~5 cm)
NPATH = 32
SPLIT_RATIO = os.environ.get("FPW_SPLIT_RATIO", "0") == "1"   # a ratio + clip per component (discrete / continuous / flow)
NO_GRASP = os.environ.get("FPW_NO_GRASP_HEAD", "0") == "1"   # grasp-yaw head unused: default yaw, not in the loss / entropy
GL_W = 0.0 if NO_GRASP else 1.0
ENT_W = float(os.environ.get("FPW_ENT_W", "0.0"))              # entropy bonus on the type / pointer / grasp heads
EP_TOK = os.environ.get("FPW_EP_TOK", "0") == "1"              # the episode token
OBS_NOISE = os.environ.get("FPW_OBS_NOISE", "0") == "1"        # pose noise on the actor's inputs
ROBOT_TOK = os.environ.get("FPW_ROBOT_TOK", "0") == "1"       # robot tokens (arm bases, joints, gripper pose, links, state, target)
EMB_T = None                                                   # instruction sampling: (T, 2, D) task embeddings, own / other; set by cleantables.rl.train
WARM_I0 = int(os.environ.get("FPW_WARM_I0", "0"))              # the flow starts at t0 = WARM_I0 / STEPS from global_plan's route (0 = from noise)
WM_CACHE = os.environ.get("FPW_WM_CACHE", "0") == "1"       # reuse the frozen WM's read-out while a table's state is unchanged
FAST2 = os.environ.get("FP_FAST2", "1") == "1"                   # exact speed-ups (identical outputs); 0 = the reference implementation
KL_HEADS = os.environ.get("FPW_KL_HEADS", "0") == "1"       # the early stop's approx KL on the heads' choices only
CLIP_GROUPS = os.environ.get("FPW_CLIP_GROUPS", "0") == "1"  # separate grad clips for the WM (T2 + LoRA) and the heads
R_EDIT = float(os.environ.get("FPW_R_EDIT", "0.15"))       # route edit range (m), as the carry edit's (policy.R_EDIT)
ROT_EDIT = os.environ.get("FPW_ROT_EDIT", "0") == "1"       # learned roll / pitch / yaw along routes / carries (motion only)
R_ROT = math.radians(float(os.environ.get("FPW_R_ROT_DEG", "30")))
PATH_REPAIR = os.environ.get("FPW_PATH_REPAIR", "0") == "1"    # route_opt only repairs the WM's own carry path (interior points, clear of objects)
REPAIR_STEPS = int(os.environ.get("FPW_REPAIR_STEPS", "8")); REPAIR_LR = float(os.environ.get("FPW_REPAIR_LR", "0.006"))
REPAIR_FAST = os.environ.get("FPW_REPAIR_FAST", "0") == "1"   # the fast SDF push-out instead of route_opt
REACH_REPAIR = os.environ.get("FPW_REACH_REPAIR", "0") == "1"
NO_EARLY_STOP = os.environ.get("FPW_NO_EARLY_STOP", "0") == "1"   # no per-arm DONE while objects remain: the arm waits
NO_FREE_MOVE = os.environ.get("FPW_NO_FREE_MOVE", "0") == "1"     # no planner free move: pick up or wait
# Self-imitation: the T2 flow decoder also learns (flow matching) the executed carry paths of successful carries (the controller's
# offsets / slow-downs and the SDF and reach repairs folded in), conditioned on the decision's own logged inputs; the same success test
# for every object (sil_select). Off: no code path reached, no RNG drawn.
SIL = os.environ.get("FPW_SIL", "0") == "1"
SIL_W = float(os.environ.get("FPW_SIL_W", "0.5"))             # loss weight (a separate Adam: it only matters through the grad clip)
SIL_N = int(os.environ.get("FPW_SIL_N", "4096"))              # at most this many successful carries per iteration (random subset)
SIL_LR = float(os.environ.get("FPW_SIL_LR", "3e-5"))          # the SIL Adam's lr over the T2 decoder (GRPO uses its own WM lr); on CPU,
                                                              # 40 steps at 3e-5 reduce a smooth 5 cm offset from 3.3 to 1.2 cm on the trained decoder
SIL_ACC = int(os.environ.get("FPW_SIL_ACC", "4"))             # micro-batches (FPW_BS each) per SIL optimiser step
END_EDIT = os.environ.get("FPW_END_EDIT", "0") == "1"          # a learned carry end offset (dx, dy, dz): the release point is learned
R_END = float(os.environ.get("FPW_R_END", "0.10"))
END_LS0 = float(os.environ.get("FPW_END_LS0", "-1.0"))         # the end offset's initial log-sd (pre-tanh)
ROT_LS0 = float(os.environ["FPW_ROT_LS0"]) if os.environ.get("FPW_ROT_LS0") else None   # the rotation edits' initial log-sd
ROUTE_EDIT = os.environ.get("FPW_ROUTE_EDIT", "1") == "1"    # learned approach / free-move routes (0 = the fixed polyline / line)
FPW_PALM_Z0 = float(os.environ.get("FPW_PALM_Z0", "0.09"))   # free-move palm height floor: the fingertips (8.9 cm below the palm centre) stay above the table


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base; self.A = nn.Linear(base.in_features, r, bias=False); self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / r); nn.init.zeros_(self.B.weight); self.s = alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.s


class WMActor(nn.Module):
    def __init__(self, wm, r=8):
        super().__init__()
        self.wm = copy.deepcopy(wm)
        for p in self.wm.parameters():
            p.requires_grad_(False)
        for m in self.wm.modules():                                       # no dropout; train mode is kept (the fused eval path would skip LoRA)
            if isinstance(m, nn.Dropout):
                m.p = 0.0
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = 0.0
        for lay in self.wm.trunk.layers:
            lay.linear1 = LoRA(lay.linear1, r); lay.linear2 = LoRA(lay.linear2, r)
        for nm in ("g_in", "g_step", "g_obj", "g_t", "g_dec", "g_out"):    # the T2 velocity decoder: fully trainable
            mod = getattr(self.wm, nm)
            for p in (mod.parameters() if isinstance(mod, nn.Module) else [mod]):
                p.requires_grad_(True)
        d = self.wm.cls.shape[-1] if hasattr(self.wm, "cls") else 256
        self.type_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(TYPES)))
        self.ptr_q = nn.Linear(d, d, bias=False); nn.init.zeros_(self.ptr_q.weight)
        self.grasp_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(GYAWS)))
        self.cont_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))          # u, retreat (pre-squash)
        self.move_h = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.log_std = nn.Parameter(torch.full((6,), -1.0))
        self.rob_in = nn.Sequential(nn.Linear(F_ROB, d), nn.GELU(), nn.Linear(d, d))
        self.rob_type = nn.Parameter(torch.zeros(d)); self.rob_attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.0)
        self.rob_out = nn.Linear(d, d); nn.init.zeros_(self.rob_out.weight); nn.init.zeros_(self.rob_out.bias)
        self.ep_in = nn.Sequential(nn.Linear(F_EP, d), nn.GELU(), nn.Linear(d, d)); self.ep_type = nn.Parameter(torch.zeros(d))
        self.route_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3 * N_RCTRL))        # route edit means (pre-tanh)
        self.route_ls = nn.Parameter(torch.full((3 * N_RCTRL,), -1.0))
        self.rot_route_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3 * N_RCTRL))   # roll / pitch / yaw along a route
        self.rot_carry_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3 * N_RCTRL))   # ... and along a carry
        self.rot_route_ls = nn.Parameter(torch.full((3 * N_RCTRL,), -1.0)); self.rot_carry_ls = nn.Parameter(torch.full((3 * N_RCTRL,), -1.0))
        self.end_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3)); self.end_ls = nn.Parameter(torch.full((3,), END_LS0))   # carry end offset
        if ROT_LS0 is not None:                                          # the rotation edits start with small exploration (any-angle range)
            with torch.no_grad():
                self.rot_route_ls.fill_(ROT_LS0); self.rot_carry_ls.fill_(ROT_LS0)
        self.stop_w = nn.Parameter(torch.tensor([6.0])); self.grasp_prior = nn.Parameter(torch.tensor([3.0]))
        for h in (self.type_h, self.grasp_h, self.cont_h, self.move_h, self.route_h, self.rot_route_h, self.rot_carry_h, self.end_h):
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)
        with torch.no_grad():
            self.type_h[-1].bias.copy_(torch.tensor([0.0, -3.5, -3.5, -3.0]))             # take; wait / move rare at the start; done from STOP
            self.cont_h[-1].bias.copy_(torch.tensor([3.0, 0.0]))                           # u ~ 0.95 (-> 1), retreat ~ 0.1 m
        self.train()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---------------------------------------------------------------------------------------------------------------------------------
    def encode(self, bc, rob=None, ept=None):
        mode = torch.ones(bc["obj_geo"].shape[0], dtype=torch.long, device=bc["obj_geo"].device)
        E = self.wm.encode(bc, mode); E["kind"] = bc["obj_kind"]
        toks = []
        if ROBOT_TOK and rob is not None:                                # robot tokens, one per arm,
            toks.append(self.rob_in(rob.float()) + self.rob_type)         # read by every WM output token through one cross-attention block whose
        if EP_TOK and ept is not None:                                   # output projection is zero-initialised: at init the WM is unchanged;
            toks.append((self.ep_in(ept.float()) + self.ep_type)[:, None])   # + the episode token, the same block
        if toks:
            R = torch.cat(toks, 1)
            att, _ = self.rob_attn(E["mem"], R, R, need_weights=False)
            y = E["mem"] + self.rob_out(att)
            E["mem"], E["cls"], E["obj"] = y, y[:, :4], y[:, 5:29]
        return E

    def heads(self, E, slot, mask, stop, gdef):
        """slot (n,) the gripper's hand slot (0 / 1); mask (n, MAXO) takeable objects; stop (n,) the frozen WM's STOP; gdef (n, MAXO) default yaw idx
        -> type logits (n, 4), object logits (n, MAXO), hand token (n, d), object tokens (n, MAXO, d)."""
        n = len(slot); ar = torch.arange(n, device=slot.device)
        mem = E["mem"]; ot = E["obj"]; nobj = ot.shape[1]
        ht = mem[ar, 5 + nobj + slot]                                                       # this gripper's output hand token
        cls = E["cls"][:, 0]
        tl = self.type_h(torch.cat([ht, cls], -1))
        tl = tl + torch.stack([torch.zeros_like(stop), torch.zeros_like(stop), torch.zeros_like(stop), self.stop_w * stop.float()], -1)
        tl = tl.masked_fill(~mask.any(1)[:, None] & (torch.arange(4, device=slot.device) == 0)[None], -1e4)
        if NO_EARLY_STOP:                                                # DONE is never chosen (sampling and GRPO log-probs both use this); the
            tl = tl.masked_fill((torch.arange(4, device=slot.device) == 3)[None], -1e4)   # episode still ends when every object is judged placed / at 45 s
        if NO_FREE_MOVE:                                                 # the planner's free move (an untrained fixed target) is never chosen
            tl = tl.masked_fill((torch.arange(4, device=slot.device) == 2)[None], -1e4)
        pn, _ = self.wm.next_scores(E)
        pl = torch.log(pn.float().clamp(min=1e-6)) + (self.ptr_q(ht)[:, None] * ot).sum(-1) / math.sqrt(ot.shape[-1])
        pl = pl.masked_fill(~mask, -1e4)
        return tl, pl, ht, ot

    def grasp_logits(self, ht, ok_, gdef_k):
        gl = self.grasp_h(torch.cat([ok_, ht], -1))
        return gl + self.grasp_prior * F.one_hot(gdef_k, len(GYAWS)).float()

    # ---------------------------------------------------------------------------------------------------------------------------------
    def flow_sample(self, E, j, x0, noise=True, sigma=SIGMA, steps=STEPS, i0=0):
        """the actor's T2 flow from x0 (n, nobj, H, 3) PS units, stochastic in object j's channel (n,) -> (x_final, xc (n, steps+1, H, 3) the
        chosen channel's states, means (n, steps, H, 3) of each step's Gaussian)."""
        n = x0.shape[0]; ar = torch.arange(n, device=x0.device); dt = 1.0 / steps
        x = x0.clone(); xc = [x[ar, j]]; mus = []; lp = torch.zeros(n, device=x0.device); s2 = sigma ** 2 * dt
        for i in range(i0, steps):                                       # i0 > 0: a warm start at t = i0 / steps
            v = self.wm.velocity_all(E, x, torch.full((n,), i * dt, device=x.device)).float()
            mu = x + v * dt
            xn = mu.clone()
            if noise:
                eps = torch.randn_like(mu[ar, j]); xn[ar, j] = mu[ar, j] + sigma * math.sqrt(dt) * eps
                lp = lp + (-0.5 * eps ** 2 - 0.5 * math.log(2 * math.pi * s2)).sum((-1, -2))   # = log N(x_{i+1}; mu, s2) exactly
            mus.append(mu[ar, j]); x = xn; xc.append(x[ar, j])
        self._last_lp = lp
        return x, torch.stack(xc, 1), torch.stack(mus, 1)

    def flow_logp(self, E, j, x0, xc, sigma=SIGMA, steps=STEPS, i0=0):
        """the exact log-prob of the recorded chosen-channel path xc under this actor (the other channels replayed deterministically from x0),
        and the step means (for the closed-form KL)."""
        n = x0.shape[0]; ar = torch.arange(n, device=x0.device); dt = 1.0 / steps
        x = x0.clone(); lp = torch.zeros(n, device=x0.device); mus = []
        for i in range(i0, steps):
            v = self.wm.velocity_all(E, x, torch.full((n,), i * dt, device=x.device)).float()
            mu = x + v * dt
            s2 = sigma ** 2 * dt
            lp = lp + (-0.5 * ((xc[:, i - i0 + 1] - mu[ar, j]) ** 2) / s2 - 0.5 * math.log(2 * math.pi * s2)).sum((-1, -2))
            mus.append(mu[ar, j])
            x = mu.clone(); x[ar, j] = xc[:, i - i0 + 1]
        return lp, torch.stack(mus, 1)


# ======================================================================================================================================
F_ROB = 46
F_EP = 3 + 24 + 72 + 24                                          # time left, carries left, unmet share; per slot: met flag, goal offset (3), speed


def robot_feats(CS, envs):
    """(n, 2, F_ROB) one token per arm, table frame: base (xyz, sin / cos yaw); the 7 joint angles (sin, cos); the gripper pose (TCP xyz +
    6D rotation); the elbow / wrist / hand link centres (franka_kinematics FK); the gripper's state (idle, approach, carry, release, free
    move, retract); its current target (the route / carry path end, else the TCP itself)."""
    from cleantables.robots import franka_kinematics as K
    A_ = CS.arms; e = torch.as_tensor(np.asarray(envs), device=CS.dev).long(); n = len(e)
    if A_ is None:
        return np.zeros((n, 2, F_ROB), np.float32)
    B = A_.base[e]; q = A_.q[e]                                                    # (n, 2, 4, 4), (n, 2, 7)
    Fk = B[:, :, None] @ K.fk(q)                                                   # (n, 2, frames, 4, 4) table frame
    tcp = Fk[:, :, K.TCP]; R = tcp[..., :3, :3]
    pts = torch.stack([Fk[:, :, K.F_IDX["link4"], :3, 3], Fk[:, :, K.F_IDX["link6"], :3, 3], Fk[:, :, K.F_IDX["hand"], :3, 3]], 2).reshape(n, 2, 9)
    yaw = torch.atan2(B[..., 1, 0], B[..., 0, 0])
    ph = CS.phase[e]; af = CS.after[e]; ret = A_.retract[e]
    st = torch.stack([(ph == 0) | (ph == 3) | (ph == 5), (ph == 4) & (af == 1) & ~ret, ph == 1, ph == 2, (ph == 4) & (af != 1) & ~ret, ret], -1).float()
    org = CS.fp.org[e][:, None]
    tgt = torch.where((ph == 4)[..., None], CS.mpath[e][:, :, -1] - org, torch.where((ph == 1)[..., None], CS.path[e][:, :, -1], tcp[..., :3, 3]))
    f = torch.cat([B[..., :3, 3], torch.sin(yaw)[..., None], torch.cos(yaw)[..., None], torch.sin(q), torch.cos(q), tcp[..., :3, 3],
                   R[..., :, 0], R[..., :, 1], pts, st, tgt], -1)
    return torch.nan_to_num(f).float().cpu().numpy()


def load_actor(A, sd):
    """Load actor weights; a checkpoint without the route head leaves it at its zero init (= the fixed routes)."""
    miss, unexp = A.load_state_dict(sd, strict=False)
    assert not unexp and all(k.startswith(("route_", "rob_", "ep_", "rot_", "end_")) for k in miss), (miss, unexp)
    return miss


def build_batch(LP, tpl_ids, xyz, yaw, palms=None):
    """the WM's robot-mode input for rows (n,): LivePlanner's cached constants + obj_geo + the two hands (palm xyz (n, 2, 3) table frame, or
    None = no hands). Hands are put in the training order (slots sorted by x) -> (batch, slot_of_gripper (n, 2))."""
    from cleantables.worldmodel.schema import MAXO
    TBL = LP._const_tables(); dev = LP.dev
    tp = torch.as_tensor(np.asarray(tpl_ids), device=dev).long(); n = len(tp)
    X = torch.zeros(n, MAXO, 3, device=dev); Yw = torch.zeros(n, MAXO, device=dev); Ii = min(np.asarray(xyz).shape[1], MAXO)
    X[:, :Ii] = torch.as_tensor(np.asarray(xyz)[:, :Ii], dtype=torch.float32, device=dev)
    Yw[:, :Ii] = torch.as_tensor(np.asarray(yaw)[:, :Ii], dtype=torch.float32, device=dev)
    gc = TBL["_gc"][tp]; I = TBL["_I"][tp]; slot = torch.arange(MAXO, device=dev)[None]
    L, Wd, H, yl = gc[..., 0], gc[..., 1], gc[..., 2], gc[..., 3] + Yw
    top = X[..., 2] + H
    geo = torch.stack([X[..., 0], X[..., 1], top, L, Wd, top, torch.sin(2 * yl), torch.cos(2 * yl)], -1) * (slot < I[:, None])[..., None]
    geo = torch.where((slot == I[:, None])[..., None], TBL["_tg"][tp][:, None].expand(-1, MAXO, -1), geo)
    bc = {k: v[tp] for k, v in TBL.items() if not k.startswith("_")}
    bc["obj_geo"] = geo
    sog = torch.zeros(n, 2, dtype=torch.long, device=dev); sog[:, 1] = 1
    if palms is not None:
        p = torch.as_tensor(np.asarray(palms, np.float32), device=dev)
        o = p[..., 0].argsort(-1)                                                          # training slots sorted by x
        hg = torch.zeros(n, 2, 13, device=dev)
        hx = torch.stack([torch.as_tensor(np.asarray(LP.B[int(t)].Z["hand_ext"], np.float32)) for t in np.asarray(tpl_ids)]).to(dev)
        hg[..., 0] = 1; hg[..., 1:4] = torch.gather(p, 1, o[..., None].expand(-1, -1, 3)); hg[..., 4:7] = hx[:, None]
        hg[..., 7:10] = torch.tensor([0.0, 1.0, 0.0], device=dev); hg[..., 10:13] = torch.tensor([1.0, 0.0, 0.0], device=dev)
        bc["hand_geo"] = hg
        sog = torch.argsort(o, -1)                                                         # gripper g sits in slot sog[:, g]
    return bc, sog, H, yl


def watch_path_root(start_root, goal_root, traj, Hh):
    """the WM's path (watch_path, top centre) for one object from its root start to its root goal -> (n, 33, 3) root points (+ the goal)."""
    from cleantables.worldmodel.bridge import watch_path
    up = torch.stack([torch.zeros_like(Hh), torch.zeros_like(Hh), Hh], -1)
    p = watch_path(start_root + up, goal_root + up, traj, NP=NPATH) - up[:, None]
    p[..., 2] = p[..., 2].clamp(min=0.0)
    return torch.cat([p, goal_root[:, None]], 1)


# ======================================================================================================================================
def _sub(E, rows):
    """the rows of an encoding dict (every tensor with the batch as its first dim)."""
    B = E["obj"].shape[0]
    return {k: (v[rows] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B else v) for k, v in E.items()}


def gauss_lp(x, mu, ls):
    ls = ls.clamp(-2.5, 0.5)
    return (-0.5 * ((x - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)


class WMActorDecider:
    """mode 'sample' (RL rollouts), 'greedy' (the mode of every head; flow without noise) or 'init' (= greedy at the actor's init: the WM's
    own choice, its raw T2 path to the frozen target, the default grasp, u = 1)."""

    def __init__(self, actor, LP, J, CS, tpl_env, gtab, mode="sample", ws_box=None, sigma=SIGMA):
        self.A, self.LP, self.J, self.CS, self.tpl_env, self.G, self.mode = actor, LP, J, CS, np.asarray(tpl_env), gtab, mode
        self.box, self.sigma, self.log, self.ctx = ws_box, sigma, [], None
        self.tm = dict(wm_frozen=0.0, judge=0.0, reach=0.0, encode_heads=0.0, flow=0.0, rest=0.0)

    def _wm_cached(self, tp, xyz, yaw):
        """The frozen WM's read-out per row, reused while that table's state is unchanged (every object within 1 mm / 0.01 rad of a
        state already evaluated, e.g. a decision again after an aborted take)."""
        if not hasattr(self, "_wmc"):
            self._wmc = {}
        keys = [(int(t), np.round(np.asarray(x) * 1000).astype(np.int32).tobytes(), np.round(np.asarray(y) * 100).astype(np.int32).tobytes())
                for t, x, y in zip(tp, xyz, yaw)]
        miss = [i for i, k in enumerate(keys) if k not in self._wmc]
        if miss:
            out = self.LP._wm_fast(tp[miss], xyz[miss], yaw[miss])
            for j, i in enumerate(miss):
                self._wmc[keys[i]] = {k: v[j] for k, v in out.items()}
            if len(self._wmc) > 200000:
                self._wmc.clear()
        self.n_wm_hit = getattr(self, "n_wm_hit", 0) + len(keys) - len(miss); self.n_wm_q = getattr(self, "n_wm_q", 0) + len(keys)
        rows = [self._wmc[k] for k in keys]
        return {k: np.stack([r[k] for r in rows]) for k in rows[0]}

    def _gp_init(self, tp, xyz, yaw, obj, tg, valid, K, H, wmo=None):
        """Warm start. X (n, MAXO, H, 3) PS units: in the chosen object's channel global_plan's collision-free route for it (start -> its
        goal), as the T2 representation watch_path inverts (the top-centre displacement from the start at the H steps); the other channels 0
        (they stay put during this carry). A row the planner cannot route keeps 0 (the straight line)."""
        from cleantables.worldmodel.schema import MAXO
        from cleantables.worldmodel.base_model import PS
        import time as _t
        t_ = _t.time()
        n = len(tp); dev = self.CS.fp.dev; on = obj.cpu().numpy()
        placed = ~valid.copy()
        gy = np.nan_to_num(self.G[tp][:, :K], nan=np.nan)
        out = self.LP.plan(tp, xyz, yaw, placed, top_k=1, want=on, gyaw=gy, goal_xy=tg[..., :2], wmo=wmo)
        X = torch.zeros(n, MAXO, H, 3, device=dev)
        ext = self.CS.fp.ext.cpu().numpy()
        pth = out["path"][:, 0]; pth = pth.cpu().numpy() if torch.is_tensor(pth) else np.asarray(pth)
        ok = (out["obj"][:, 0] == on) & np.isfinite(pth).all((-1, -2))
        self.n_gp_ok = getattr(self, "n_gp_ok", 0) + int(ok.sum()); self.n_gp_q = getattr(self, "n_gp_q", 0) + n
        if ok.any():
            r = np.flatnonzero(ok)
            env_ = np.asarray(self._cur_e)[r]
            hk = ext[env_, on[r], 2][:, None]
            top = pth[r] + np.concatenate([np.zeros((len(r), 1, 2)), hk[..., None].repeat(1, 1)], -1)        # root -> top centre
            idx = np.linspace(0, top.shape[1] - 1, H + 1)[1:]                                    # index-uniform, as watch_path interpolates
            lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, top.shape[1] - 1); fr = (idx - lo)[None, :, None]
            samp = top[:, lo] * (1 - fr) + top[:, hi] * fr
            X[torch.as_tensor(r, device=dev), torch.as_tensor(on[r], device=dev)] = torch.as_tensor((samp - top[:, :1]) / PS, dtype=torch.float32, device=dev)
        self.tm["gp_init"] = self.tm.get("gp_init", 0.0) + _t.time() - t_
        return X

    @torch.no_grad()
    def _rec_plans(self, CS, e, g, tp, xyz, yaw, xf, tg, wm, ext, K, typn, objn, gin, path_exec):
        """Plan visualisation (only while a Recorder is active): every mover's carry path from this decision's T2 flow sample xf (n, MAXO,
        H, 3), with the same post-processing as the chosen object's (watch_path_root from its root start to its target, then the path
        repair: SDF push-out + the arm's reach) -> CS.rec.plans. No effect on behaviour: the repair's counters / timers / reach stats are
        restored afterwards. chosen_err_cm = max |recorded chosen path - the executed planned path| (cm) checks the conversion (END_EDIT,
        when on, is not applied to the non-chosen objects, so then the chosen one differs by its end offset)."""
        from cleantables.worldmodel.base_model import PS
        dev = xf.device; n = xf.shape[0]; J = self.J
        mv = np.asarray(J.mov)[e][:, :K] & ~np.asarray(J.static)[e][:, :K]
        gz = wm["G"][:, :K, 2] if wm["G"].shape[1] >= K else np.pad(wm["G"][..., 2], ((0, 0), (0, K - wm["G"].shape[1])))
        goal_t = torch.as_tensor(np.stack([tg[:, :, 0], tg[:, :, 1], gz], -1), dtype=torch.float32, device=dev)
        snap = (getattr(self, "n_rep_chk", None), getattr(self, "n_rep", None), getattr(self, "rep_cm", None), self.tm.get("repair"),
                copy.deepcopy(getattr(CS.arms, "rr", None)) if CS.arms is not None else None)
        ar = torch.arange(n, device=dev)
        plans = [dict() for _ in range(n)]; raw = [dict() for _ in range(n)]
        for k in range(K):
            rows = mv[:, k] & np.isfinite(xyz[:, k]).all(-1)
            if not rows.any():
                continue
            kk = torch.full((n,), k, dtype=torch.long, device=dev)
            traj = xf[ar, kk] * PS
            st = torch.as_tensor(xyz, device=dev)[ar, kk]
            Hk = torch.as_tensor(np.nan_to_num(ext[..., 2]), device=dev)[ar, kk]
            pk = watch_path_root(st, goal_t[ar, kk], traj, Hk).cpu().numpy()
            pr = self._repair(tp, xyz, yaw, np.full(n, k), np.where(rows, 0, 1), gin, pk.copy(), np.asarray(wm["dy"])) if PATH_REPAIR else pk
            for r in np.flatnonzero(rows):
                plans[r][int(k)] = np.asarray(pr[r], np.float32); raw[r][int(k)] = np.asarray(pk[r], np.float32)
        for nm_, v_ in (("n_rep_chk", snap[0]), ("n_rep", snap[1]), ("rep_cm", snap[2])):     # restore (or remove what the extra calls created)
            if v_ is None:
                self.__dict__.pop(nm_, None)
            else:
                setattr(self, nm_, v_)
        if snap[3] is None:
            self.tm.pop("repair", None)
        else:
            self.tm["repair"] = snap[3]
        if CS.arms is not None:
            if snap[4] is None:
                CS.arms.__dict__.pop("rr", None)
            else:
                CS.arms.rr = snap[4]
        for r in range(n):
            ch = int(objn[r]) if int(typn[r]) == 0 else None
            err = float(np.abs(plans[r][ch] - np.asarray(path_exec[r])).max() * 100) if (ch is not None and ch in plans[r]) else None
            CS.rec.plans.append(dict(step=int(CS.step_i), env=int(e[r]), g=int(g), chosen=ch, paths=plans[r], raw=raw[r], chosen_err_cm=err))

    def _repair(self, tp, xyz, yaw, on, typn, gin, path, dyn):
        """Repair the WM's own carry paths of the take rows so they clear other objects with some margin: route_opt.optimise moves only
        the interior points (start / end fixed), by gradient steps on the clearance cost (every other object's mesh SDF, the hand's points,
        the band, the set-down rule). No candidates, no over / around routes. -> the repaired paths; logs the mean point displacement."""
        from cleantables.worldmodel.physics import route_opt as RO
        import time as _t
        t_ = _t.time(); rows = [r for r in range(len(tp)) if typn[r] == 0]
        if not rows:
            return path
        if REPAIR_FAST:                                                  # the fast SDF push-out (ChainSim.fast_repair), all rows batched
            ee = np.asarray(self._cur_e)[rows]; aa = np.asarray(on)[rows]
            dd = np.array([float(dyn[r, int(on[r])]) if int(on[r]) < dyn.shape[1] else 0.0 for r in rows], np.float32)
            pr, mv_ = self.CS.fast_repair(ee, aa, path[rows], dd)
            if REACH_REPAIR and getattr(self.CS, "arms", None) is not None:   # then the carrying arm's reach; the clearance re-checked once
                pr, rm_ = self.CS.reach_repair_carry(ee, int(getattr(self, "_cur_g", 0)), aa, pr)
                if rm_.any():
                    pr = pr.copy(); pr[rm_] = self.CS.fast_repair(ee[rm_], aa[rm_], pr[rm_], dd[rm_], iters=1)[0]
            out = path.copy(); out[rows] = pr
            self.n_rep_chk = getattr(self, "n_rep_chk", 0) + len(rows); self.n_rep = getattr(self, "n_rep", 0) + int((mv_ > 1e-4).sum())
            self.rep_cm = getattr(self, "rep_cm", 0.0) + 100 * float(mv_[mv_ > 1e-4].sum())
            self.tm["repair"] = self.tm.get("repair", 0.0) + _t.time() - t_
            return out
        items = []
        for r in rows:
            B = self.LP.B[int(tp[r])]; I = B.I; a = int(on[r])
            items.append(dict(S=B.S, xyz=np.asarray(xyz[r, :I], np.float64), yaw=np.asarray(yaw[r, :I], np.float64), a=a, plans=path[r][None].astype(np.float32),
                              dyaws=np.array([float(dyn[r, a]) if a < dyn.shape[1] else 0.0], np.float32), gyaw=float(GYAWS[gin[r]])))
        ts = self.LP.ts; out = path.copy()
        B0 = RO.build_batch(ts, items)                                   # one exact check of every path first: only paths that touch something
        if B0 is not None:                                               # (hit or the leeway band) are repaired, so clear paths stay bit for bit
            with torch.no_grad():
                P0 = RO.path_cost(ts, B0, B0["plans"])
            need = ((P0["hit"] > 0) | (P0["band"] > 0)).cpu().numpy()
            sel = [B0["keep"][q] for q in np.flatnonzero(need)]
            self.n_rep_chk = getattr(self, "n_rep_chk", 0) + len(B0["keep"])
            if sel:
                with torch.enable_grad():
                    res, _ = RO.optimise(ts, [items[i] for i in sel], steps=REPAIR_STEPS, lr=REPAIR_LR)
                moved = []
                for i, p_ in zip(sel, res):
                    if p_ is not None:
                        r = rows[i]; out[r] = p_[0]; moved.append(float(np.linalg.norm(p_[0] - path[r], axis=-1).mean()))
                self.n_rep = getattr(self, "n_rep", 0) + len(moved); self.rep_cm = getattr(self, "rep_cm", 0.0) + 100 * sum(moved)
        self.tm["repair"] = self.tm.get("repair", 0.0) + _t.time() - t_
        return out

    def set_ctx(self, n_car, max_carries, ep_start):
        self.ctx = (n_car, max_carries, ep_start)

    @torch.no_grad()
    def decide(self, envs, Pn, yw, busy, g=0):
        from cleantables.worldmodel.schema import MAXO
        from cleantables.worldmodel.base_model import PS
        A, CS, J = self.A, self.CS, self.J; fp = CS.fp; dev = fp.dev
        e = np.asarray(envs); n = len(e); tp = self.tpl_env[e]; K = Pn.shape[1]; self._cur_e = e; self._cur_g = g
        xyz = np.nan_to_num(Pn[e, :, :3]); yaw = np.nan_to_num(yw[e])
        if OBS_NOISE:                                                    # simulated tracker error on what the actor sees (not the judge / sim):
            xyz = xyz + np.random.normal(0, 1, xyz.shape).astype(np.float32) * np.array([0.01, 0.01, 0.005], np.float32)   # 1 cm xy, 0.5 cm z,
            yaw = yaw + np.random.normal(0, np.radians(3.0), yaw.shape).astype(np.float32)                                # 3 deg yaw
        import time as _t
        t_ = _t.time()
        palms = (CS.hP[torch.as_tensor(e, device=dev)] - fp.org[torch.as_tensor(e, device=dev)][:, None]).cpu().numpy()
        inst = np.asarray(CS.inst)[e].astype(np.int64) if (EMB_T is not None and hasattr(CS, "inst")) else None
        te = None if inst is None else EMB_T[torch.as_tensor(tp, device=EMB_T.device), torch.as_tensor(inst, device=EMB_T.device)]   # instruction sampling
        wm = (self._wm_cached(tp, xyz, yaw) if (WM_CACHE and te is None) else self.LP._wm_fast(tp, xyz, yaw, temb=te))       # the frozen WM read-out
        wm_own = None                                                    # exact: global_plan's WM read-out under the scene's own instruction,
        if not FAST2:
            pass
        elif te is not None and WARM_I0 > 0:                             # reused from this one for own-instruction rows, recomputed only for the rest
            r1 = np.flatnonzero(inst != 0)
            wm_own = {k_: np.array(v_, copy=True) for k_, v_ in wm.items()}
            if len(r1):
                o1 = self.LP._wm_fast(np.asarray(tp)[r1], np.asarray(xyz)[r1], np.asarray(yaw)[r1])
                for k_ in wm_own:
                    wm_own[k_][r1] = o1[k_]
        elif te is None and not WM_CACHE:
            wm_own = wm
        self.tm["wm_frozen"] += _t.time() - t_; t_ = _t.time()
        sc = J.score(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw)); tg = J.targets(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw))[e]
        valid = fp.valid.cpu().numpy()[e] & np.isfinite(Pn[e, :, 0])
        ext = fp.ext.cpu().numpy()[e]; top = xyz.copy(); top[..., 2] += np.nan_to_num(ext[..., 2])
        mask = valid & ~busy[e] & ~(sc["met"][e] & ~J.static[e]) & J.mov[e] & ~J.static[e] & (xyz[..., 2] > -0.05)   # off the table: not takeable
        if os.environ.get("FPW_DEBUG"):
            m0 = mask.copy()
        self.tm["judge"] += _t.time() - t_; t_ = _t.time()
        if CS.arms is not None and os.environ.get("FPW_REACH_MASK") == "1":   # off by default: the actor learns its reach itself
            rs_, rg_ = CS.arms.reach(e, g, top, kind="start"), CS.arms.reach(e, g, tg, kind="target")
            mask &= rs_ & rg_
            if os.environ.get("FPW_DEBUG"):
                print(f"DEBUG mask g{g}: rows {n}, before reach any {int(m0.any(1).sum())}, start-reach any {int((m0 & rs_).any(1).sum())}, "
                      f"target-reach any {int((m0 & rg_).any(1).sum())}, both {int(mask.any(1).sum())}; met {int(sc['met'][e].sum())} static {int(J.static[e].sum())} "
                      f"mov {int(J.mov[e].sum())} valid {int(valid.sum())}", flush=True)
        m24 = np.zeros((n, MAXO), bool); m24[:, :K] = mask
        if getattr(self, "_GD", None) is None:                                                   # the default grasp's index, per template slot (once)
            Gt = np.nan_to_num(self.G, nan=0.0); d_ = np.abs(np.remainder(Gt[..., None] - np.array(GYAWS)[None, None] + math.pi / 2, math.pi) - math.pi / 2)
            self._GD = np.zeros((Gt.shape[0], MAXO), np.int64); self._GD[:, :Gt.shape[1]] = d_.argmin(-1)
        gd = self._GD[tp]
        self.tm["reach"] += _t.time() - t_; t_ = _t.time()
        bc, sog, Hh, _ = build_batch(self.LP, tp, xyz, yaw, palms)
        if te is not None:
            bc["task_emb"] = te
        rob = robot_feats(CS, e) if ROBOT_TOK else None
        ept = None
        if EP_TOK:                                                       # the episode token: time left, carries left, unmet share, met flags
            n_car, mc, ep_st = self.ctx
            from cleantables.worldmodel.schema import MAXO as _MX
            mf = np.zeros((n, _MX), np.float32); mm = (sc["met"][e] & J.mov[e] & ~J.static[e]); mf[:, :mm.shape[1]] = mm
            mov_ = (J.mov[e] & ~J.static[e]).sum(1).clip(min=1)
            # + per slot its offset to the goal (the target the reward and the path end use) and its speed
            off_ = np.zeros((n, _MX, 3), np.float32); off_[:, :K] = np.where((J.mov[e] & ~J.static[e])[..., None], tg - top, 0.0)
            ve_ = CS.fp.wp.to_torch(CS.fp.OV.get_velocities()).reshape(CS.fp.OV.count, 6)[CS.fp.IDX[torch.as_tensor(e, device=dev)].clamp(min=0)][..., :3]
            sp_ = np.zeros((n, _MX), np.float32); sp_[:, :K] = ve_.norm(dim=-1).cpu().numpy()[:, :K]
            ept = np.concatenate([(1 - (CS.step_i - np.asarray(ep_st)[e]) * CS.dt / EP_S)[:, None], (1 - (np.asarray(n_car)[e] - CS.n_ab_env[e]) / mc)[:, None],
                                  (1 - mm.sum(1) / mov_)[:, None], mf, off_.reshape(n, -1), sp_], 1).astype(np.float32)
        E = A.encode(bc, None if rob is None else torch.as_tensor(rob, device=dev), None if ept is None else torch.as_tensor(ept, device=dev))
        slot = sog[:, g]; m24t = torch.as_tensor(m24, device=dev); stop = torch.as_tensor(wm["stop"], device=dev)
        tl, pl, ht, ot = A.heads(E, slot, m24t, stop, None)
        greedy = self.mode in ("greedy", "init")
        typ = tl.argmax(1) if greedy else torch.distributions.Categorical(logits=tl).sample()
        obj = pl.argmax(1) if greedy else torch.distributions.Categorical(logits=pl).sample()
        obj = torch.where(m24t.any(1), obj, torch.zeros_like(obj)).clamp(max=K - 1)          # no takeable object: a placeholder (type != take)
        ar = torch.arange(n, device=dev); gdk = torch.as_tensor(gd, device=dev)[ar, obj]
        gl = A.grasp_logits(ht, ot[ar, obj], gdk)
        gi = gl.argmax(1) if greedy else torch.distributions.Categorical(logits=gl).sample()
        if NO_GRASP:                                                    # the grasp is not learned: the default yaw (closing across
            gi = gdk.clone()                                            # the object's short horizontal axis, every object the same); auto-attach on arrival
        cm = A.cont_h(torch.cat([ot[ar, obj], ht], -1)); ls = A.log_std
        zc = cm if greedy else cm + ls[:2].clamp(-2.5, 0.5).exp() * torch.randn_like(cm)
        mm = A.move_h(ht); zm = mm if greedy else mm + ls[2:].clamp(-2.5, 0.5).exp() * torch.randn_like(mm)
        take = typ == 0
        rm = A.route_h(torch.cat([ht, torch.where(take[:, None], ot[ar, obj], E["cls"][:, 0])], -1))       # approach (take) / free-move route
        zr = rm if greedy else rm + A.route_ls.clamp(-2.5, 0.5).exp() * torch.randn_like(rm)
        if self.mode == "init" or not ROUTE_EDIT:
            zr = torch.zeros_like(rm)
        if ROT_EDIT:                                                     # roll / pitch / yaw edits: along the route (take / move) and the carry
            rin = torch.cat([ht, torch.where(take[:, None], ot[ar, obj], E["cls"][:, 0])], -1)
            rrm = A.rot_route_h(rin); crm = A.rot_carry_h(torch.cat([ot[ar, obj], ht], -1))
            zrr = rrm if greedy else rrm + A.rot_route_ls.clamp(-2.5, 0.5).exp() * torch.randn_like(rrm)
            zcr = crm if greedy else crm + A.rot_carry_ls.clamp(-2.5, 0.5).exp() * torch.randn_like(crm)
            if self.mode == "init":
                zrr, zcr = torch.zeros_like(rrm), torch.zeros_like(crm)
        if END_EDIT:                                                     # the carry's end offset (the release point), zero-init head
            em = A.end_h(torch.cat([ot[ar, obj], ht], -1))
            ze = em if greedy else em + A.end_ls.clamp(-2.5, 0.5).exp() * torch.randn_like(em)
            if self.mode == "init":
                ze = torch.zeros_like(em)
        lp = torch.log_softmax(tl, -1)[ar, typ]
        lpd = lp + torch.where(take, torch.log_softmax(pl, -1)[ar, obj] + torch.log_softmax(gl, -1)[ar, gi] * GL_W, torch.zeros_like(lp))   # discrete heads
        lp = lp + torch.where(take, torch.log_softmax(pl, -1)[ar, obj] + torch.log_softmax(gl, -1)[ar, gi] * GL_W + gauss_lp(zc, cm, ls[:2]), torch.zeros_like(lp))
        lp = lp + torch.where(typ == 2, gauss_lp(zm, mm, ls[2:]), torch.zeros_like(lp))
        if ROUTE_EDIT:
            lp = lp + torch.where(take | (typ == 2), gauss_lp(zr, rm, A.route_ls), torch.zeros_like(lp))
        if ROT_EDIT:
            lp = lp + torch.where(take | (typ == 2), gauss_lp(zrr, rrm, A.rot_route_ls), torch.zeros_like(lp)) + torch.where(take, gauss_lp(zcr, crm, A.rot_carry_ls), torch.zeros_like(lp))
        if END_EDIT:
            lp = lp + torch.where(take, gauss_lp(ze, em, A.end_ls), torch.zeros_like(lp))
        if self.mode == "init":                                                                 # the WM as it is: default grasp, release at the goal
            gi = gdk.clone(); zc = torch.tensor([3.0, 0.0], device=dev).expand(n, 2).clone()
        # ---- the path: the actor's own T2 flow (noise only in the chosen object's channel), to the frozen target
        torch.cuda.synchronize(); self.tm["encode_heads"] += _t.time() - t_; t_ = _t.time(); gp0_ = self.tm.get("gp_init", 0.0)
        H = A.wm.H
        x0 = torch.zeros(n, MAXO, H, 3, device=dev) if greedy else torch.randn(n, MAXO, H, 3, device=dev)   # greedy / init: a deterministic path
        if WARM_I0 > 0:                                                  # the flow starts from global_plan's collision-free
            X = self._gp_init(tp, xyz, yaw, obj, tg, valid, K, H, wmo=wm_own)   # carry route: x_t0 = (1 - t0) eps + t0 X, t0 = WARM_I0 / STEPS
            t0 = WARM_I0 / STEPS
            x0 = (1 - t0) * x0 + t0 * X
        xf, xc, _ = A.flow_sample(E, obj, x0, noise=not greedy, sigma=self.sigma, i0=WARM_I0)
        lpf = A._last_lp if not greedy else torch.zeros(n, device=dev)
        lph = lp.clone()                                                                        # the heads' choices only (no flow)
        lp = lp + torch.where(take, lpf, torch.zeros_like(lpf))
        torch.cuda.synchronize(); self.tm["flow"] += _t.time() - t_ - (self.tm.get("gp_init", 0.0) - gp0_); t_ = _t.time()   # gp_init counted separately
        traj = xf[ar, obj] * PS
        st = torch.as_tensor(xyz, device=dev)[ar, obj]
        goal = torch.as_tensor(np.stack([tg[:, :, 0], tg[:, :, 1], wm["G"][:, :K, 2] if wm["G"].shape[1] >= K else np.pad(wm["G"][..., 2], ((0, 0), (0, K - wm["G"].shape[1])))], -1),
                               dtype=torch.float32, device=dev)[ar, obj]
        Hk = torch.as_tensor(np.nan_to_num(ext[..., 2]), device=dev)[ar, obj]
        path = watch_path_root(st, goal, traj, Hk).cpu().numpy()
        path_w = path.copy() if SIL else None                            # the WM's own carry path (before any edit / repair): the SIL reference
        if END_EDIT:                                                     # the end offset blended along the path's arc length (0 at the start)
            zen = ze.cpu().numpy(); off_e = np.tanh(zen) * R_END
            sl_ = np.concatenate([np.zeros((n, 1)), np.linalg.norm(np.diff(path, axis=1), axis=-1).cumsum(1)], 1)
            path = path + (sl_ / np.maximum(sl_[:, -1:], 1e-9))[..., None] * off_e[:, None, :]
            path[..., 2] = np.maximum(path[..., 2], 0.0)                 # the table is solid: the end cannot go below its top
        if PATH_REPAIR:                                                  # route_opt repairs the WM's own path: interior points pushed clear
            path = self._repair(tp, xyz, yaw, obj.cpu().numpy(), typ.cpu().numpy(), gi.cpu().numpy(), path, np.asarray(wm["dy"]))
        zcn = zc.cpu().numpy(); zmn = torch.tanh(zm).cpu().numpy()
        u_rel = 1 / (1 + np.exp(-zcn[:, 0])); u_rel = np.where(u_rel > 0.9, 1.0, u_rel); ret = 0.2 / (1 + np.exp(-zcn[:, 1]))
        res = [None] * n; typn, objn, gin = typ.cpu().numpy(), obj.cpu().numpy(), gi.cpu().numpy(); lpn = lp.cpu().numpy()
        lphn = lph.cpu().numpy(); lpdn = lpd.cpu().numpy()
        x0n, xcn, zmr, gdkn = x0.cpu().numpy(), xc.cpu().numpy(), zm.cpu().numpy(), gdk.cpu().numpy()
        zrn = zr.cpu().numpy(); rres = (np.tanh(zrn) * R_EDIT).reshape(n, N_RCTRL, 3)
        if ROT_EDIT:
            zrrn, zcrn = zrr.cpu().numpy(), zcr.cpu().numpy()
            rrot = (np.tanh(zrrn) * R_ROT).reshape(n, N_RCTRL, 3); crot = (np.tanh(zcrn) * R_ROT).reshape(n, N_RCTRL, 3)
        dyn = np.asarray(wm["dy"])
        if getattr(CS, "rec", None) is not None:                         # recording only: the WM's plan for every mover at this decision
            self._rec_plans(CS, e, g, tp, xyz, yaw, xf, tg, wm, ext, K, typn, objn, gin, path)
        for r in range(n):
            if typn[r] == 0:
                a = int(objn[r])
                res[r] = dict(type="take", carried=a, path=path[r], dyaw=float(dyn[r, a]) if a < dyn.shape[1] else 0.0, gyaw=float(GYAWS[gin[r]]),
                              rel_u=float(u_rel[r]), retreat=float(ret[r]), **(dict(app_resid=rres[r]) if ROUTE_EDIT else {}),
                              **(dict(app_rot=rrot[r], carry_rot=crot[r]) if ROT_EDIT else {}))
            elif typn[r] == 2:
                b = self.box[tp[r]] if self.box is not None else np.array([-0.6, 0.6, -0.4, 0.4])
                res[r] = dict(type="move", P=np.array([b[0] + (zmn[r, 0] + 1) / 2 * (b[1] - b[0]), b[2] + (zmn[r, 1] + 1) / 2 * (b[3] - b[2]),
                                                       FPW_PALM_Z0 + (zmn[r, 2] + 1) / 2 * (0.60 - FPW_PALM_Z0)], np.float32), Y=float(zmn[r, 3] * math.pi),
                              **(dict(resid=rres[r]) if ROUTE_EDIT else {}), **(dict(rot=rrot[r]) if ROT_EDIT else {}))
            elif typn[r] == 1:
                res[r] = dict(type="wait")
            self.log.append(dict(env=int(e[r]), g=int(g), step=int(CS.step_i), tp=int(tp[r]), xyz=xyz[r], yaw=yaw[r], palms=palms[r], mask=m24[r],
                                 stop=bool(wm["stop"][r]), gdk=int(gdkn[r]), typ=int(typn[r]), obj=int(objn[r]), gi=int(gin[r]), zc=zcn[r], zm=zmr[r], zr=zrn[r], **(dict(zrr=zrrn[r], zcr=zcrn[r]) if ROT_EDIT else {}), **(dict(ze=zen[r]) if END_EDIT else {}),
                                 x0=x0n[r] if typn[r] == 0 else None, xc=xcn[r] if typn[r] == 0 else None, logp=float(lpn[r]),
                                 logp_h=float(lphn[r]), logp_d=float(lpdn[r]), inst=None if inst is None else int(inst[r]), rob=None if rob is None else rob[r], ept=None if ept is None else ept[r],
                                 **(dict(wpath=path_w[r].astype(np.float32)) if SIL and typn[r] == 0 else {})))
        self.tm["rest"] += _t.time() - t_
        return res


def logp_of(A, LP, B, sigma=SIGMA, with_means=False):
    """log-probs of logged decisions B under actor A (with grad when enabled); with_means: also the flow step means of the take rows."""
    dev = LP.dev
    bc, sog, _, _ = build_batch(LP, np.array([b["tp"] for b in B]), np.stack([b["xyz"] for b in B]), np.stack([b["yaw"] for b in B]),
                                np.stack([b["palms"] for b in B]))
    if EMB_T is not None and B[0].get("inst") is not None:              # instruction sampling: each decision's own instruction
        bc["task_emb"] = EMB_T[torch.as_tensor([b["tp"] for b in B], device=EMB_T.device), torch.as_tensor([b["inst"] for b in B], device=EMB_T.device)]
    rob = torch.as_tensor(np.stack([b["rob"] for b in B]), device=dev) if ROBOT_TOK and B[0].get("rob") is not None else None
    ept = torch.as_tensor(np.stack([b["ept"] for b in B]), device=dev) if EP_TOK and B[0].get("ept") is not None else None
    E = A.encode(bc, rob, ept); n = len(B); ar = torch.arange(n, device=dev)
    g = torch.tensor([b["g"] for b in B], device=dev); slot = sog[ar, g]
    mask = torch.as_tensor(np.stack([b["mask"] for b in B]), device=dev); stop = torch.tensor([b["stop"] for b in B], device=dev)
    tl, pl, ht, ot = A.heads(E, slot, mask, stop, None)
    typ = torch.tensor([b["typ"] for b in B], device=dev); obj = torch.tensor([b["obj"] for b in B], device=dev); gi = torch.tensor([b["gi"] for b in B], device=dev)
    gdk = torch.tensor([b["gdk"] for b in B], device=dev)
    gl = A.grasp_logits(ht, ot[ar, obj], gdk)
    cm = A.cont_h(torch.cat([ot[ar, obj], ht], -1)); mm = A.move_h(ht); ls = A.log_std
    zc = torch.as_tensor(np.stack([b["zc"] for b in B]), device=dev); zm = torch.as_tensor(np.stack([b["zm"] for b in B]), device=dev)
    take = typ == 0
    ent_ = lambda l: -(torch.softmax(l, -1) * torch.log_softmax(l, -1)).sum(-1)                                       # noqa: E731
    A._last_ent = dict(type=ent_(tl).mean(), ptr=(ent_(pl) * take).sum() / take.sum().clamp(min=1), grasp=(ent_(gl) * take).sum() / take.sum().clamp(min=1))
    lp = torch.log_softmax(tl, -1)[ar, typ]
    lpd = lp + torch.where(take, torch.log_softmax(pl, -1)[ar, obj] + torch.log_softmax(gl, -1)[ar, gi] * GL_W, torch.zeros_like(lp))
    lp = lp + torch.where(take, torch.log_softmax(pl, -1)[ar, obj] + torch.log_softmax(gl, -1)[ar, gi] * GL_W + gauss_lp(zc, cm, ls[:2]), torch.zeros_like(lp))
    lp = lp + torch.where(typ == 2, gauss_lp(zm, mm, ls[2:]), torch.zeros_like(lp))
    if ROUTE_EDIT and "zr" in B[0]:
        rm = A.route_h(torch.cat([ht, torch.where(take[:, None], ot[ar, obj], E["cls"][:, 0])], -1))
        zr = torch.as_tensor(np.stack([b["zr"] for b in B]), device=dev)
        lp = lp + torch.where(take | (typ == 2), gauss_lp(zr, rm, A.route_ls), torch.zeros_like(lp))
    if ROT_EDIT and "zrr" in B[0]:                                       # the orientation edits
        rrm = A.rot_route_h(torch.cat([ht, torch.where(take[:, None], ot[ar, obj], E["cls"][:, 0])], -1)); crm = A.rot_carry_h(torch.cat([ot[ar, obj], ht], -1))
        zrr = torch.as_tensor(np.stack([b["zrr"] for b in B]), device=dev); zcr = torch.as_tensor(np.stack([b["zcr"] for b in B]), device=dev)
        lp = lp + torch.where(take | (typ == 2), gauss_lp(zrr, rrm, A.rot_route_ls), torch.zeros_like(lp)) + torch.where(take, gauss_lp(zcr, crm, A.rot_carry_ls), torch.zeros_like(lp))
    if END_EDIT and "ze" in B[0]:                                        # the carry end offset
        em = A.end_h(torch.cat([ot[ar, obj], ht], -1)); ze = torch.as_tensor(np.stack([b["ze"] for b in B]), device=dev)
        lp = lp + torch.where(take, gauss_lp(ze, em, A.end_ls), torch.zeros_like(lp))
    rows = [i for i, b in enumerate(B) if b["typ"] == 0 and b["x0"] is not None]
    mus = None
    if rows:
        rt = torch.as_tensor(rows, device=dev)
        x0 = torch.as_tensor(np.stack([B[i]["x0"] for i in rows]), device=dev); xc = torch.as_tensor(np.stack([B[i]["xc"] for i in rows]), device=dev)
        lpf, mus = A.flow_logp(_sub(E, rt), obj[rt], x0, xc, sigma=sigma, i0=WARM_I0)
    lph = lp
    if rows:
        lp = lp.index_add(0, rt, lpf)
    A._last_lpd = lpd
    return (lp, mus, rows, lph) if with_means else lp


def grpo_update(A, ref, LP, opt, B, adv, epochs=int(os.environ.get("FPW_EPOCHS", "2")), bs=int(os.environ.get("FPW_BS", "64")), clip=0.2, beta_kl=0.05, sigma=SIGMA, log=print, weights=None):
    """PPO-clipped GRPO update of the WM actor on logged decisions with advantages adv; KL to the init actor: k3 on the sampled choices +
    the closed-form Gaussian KL of the flow's step means (both actors share sigma)."""
    dev = LP.dev
    old = torch.tensor([b["logp"] for b in B], device=dev)
    oldh = torch.tensor([b.get("logp_h", 0.0) for b in B], device=dev); oldd = torch.tensor([b.get("logp_d", 0.0) for b in B], device=dev)
    Wt = torch.as_tensor(weights if weights is not None else np.ones(len(B)), dtype=torch.float32, device=dev); Adv = torch.as_tensor(adv, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(0); st = {}
    acc = int(os.environ.get("FPW_ACC", "1"))                    # micro-batches per optimiser step (effective batch bs * acc; fewer, larger steps)
    kl_stop = float(os.environ.get("FPW_KL_STOP", "0"))          # early stop: approx KL to the rollout policy above this ends the update (0 = off)
    bf = float(os.environ.get("FPW_BETA_FLOW", "1"))             # the flow's KL-to-init weight = bf * beta_kl
    groups = {"ptr": ["ptr_q."], "type": ["type_h.", "stop_w"], "grasp": ["grasp_h.", "grasp_prior"], "cont": ["cont_h."], "move": ["move_h."],
              "route": ["route_h.", "route_ls"], "end": ["end_h.", "end_ls"], "log_std": ["log_std"], "lora": ["wm.trunk."], "t2": ["wm.g_"]}
    named = [(n_, p_) for n_, p_ in A.named_parameters() if p_.requires_grad]
    n_steps, stopped, akl_sum, cf_sum, nb = 0, False, 0.0, 0.0, 0
    for ep in range(epochs):
        if stopped:
            break
        mbs = np.array_split(rng.permutation(len(B)), max(1, len(B) // bs))
        for mi, idx in enumerate(mbs):
            Bb = [B[i] for i in idx]; it = torch.as_tensor(idx, device=dev)
            lp, mus, rows, lph = logp_of(A, LP, Bb, sigma, with_means=True)
            with torch.no_grad():
                lr_, mr, _, lrh_ = logp_of(ref, LP, Bb, sigma, with_means=True); lrd_ = ref._last_lpd
            ratio = (lp - old[it]).exp()
            if ep == 0 and not st and os.environ.get("FPW_DEBUG"):
                ty = np.array([b["typ"] for b in Bb]); dd = (lp - old[it]).detach().cpu().numpy(); dr = (lr_ - lp).detach().cpu().numpy()
                log("DEBUG lp-old by type: " + str({TYPES[t]: (int((ty == t).sum()), float(np.abs(dd[ty == t]).mean()) if (ty == t).any() else None,
                                                                   float(np.abs(dr[ty == t]).mean()) if (ty == t).any() else None) for t in range(4)}))
            if SPLIT_RATIO and "logp_d" in Bb[0]:                       # one ratio + clip per component: discrete heads, continuous heads,
                lpd = A._last_lpd                                         # the flow, so flow drift cannot clip the take / pointer learning
                comps = [(lpd, oldd[it]), (lph - lpd, oldh[it] - oldd[it]), (lp - lph, old[it] - oldh[it])]
                pg = 0.0
                for lc, oc in comps:
                    rc_ = (lc - oc).exp()
                    pg = pg - (Wt[it] * torch.minimum(rc_ * Adv[it], rc_.clamp(1 - clip, 1 + clip) * Adv[it])).sum() / Wt[it].sum()
            else:
                pg = -(Wt[it] * torch.minimum(ratio * Adv[it], ratio.clamp(1 - clip, 1 + clip) * Adv[it])).sum() / Wt[it].sum()
            d = lr_ - lp; kl = (d.exp() - 1 - d).mean()
            klf = ((mus - mr) ** 2).sum((-1, -2, -3)).mean() / (2 * sigma ** 2 / STEPS) / max(len(Bb), 1) * len(rows) if mus is not None else torch.zeros((), device=dev)
            with torch.no_grad():                                         # KL to the reference per component (logged), k3
                k3 = lambda dd: float(((dd.exp() - 1) - dd).mean())                                              # noqa: E731
                kl_d_ = k3(lrd_ - A._last_lpd); kl_c_ = k3((lrh_ - lrd_) - (lph - A._last_lpd))
            ent = A._last_ent
            loss = pg + beta_kl * (kl + bf * klf) - ENT_W * (ent["type"] + ent["ptr"] + GL_W * ent["grasp"]); kl = kl + klf
            with torch.no_grad():
                if KL_HEADS and "logp_h" in Bb[0]:                     # the heads' choices only: the flow is constrained
                    lr_o = lph - oldh[it]                              # by its own KL to init and its own grad clip
                else:
                    lr_o = lp - old[it]
                akl = float(((lr_o.exp() - 1) - lr_o).mean())          # approx KL(rollout || current), k3
            akl_sum += akl; cf_sum += float(((ratio - 1).abs() > clip).float().mean()); nb += 1
            if mi % acc == 0:
                opt.zero_grad()
            (loss / acc).backward()
            if mi % acc == acc - 1 or mi == len(mbs) - 1:
                with torch.no_grad():
                    hg = {k: math.sqrt(sum(float((p_.grad ** 2).sum()) for n_, p_ in named if p_.grad is not None and any(n_.startswith(q) for q in v)))
                          for k, v in groups.items()}
                if CLIP_GROUPS:                                       # the WM (T2 + LoRA) and the heads clipped separately (the flow's grad norm
                    wm_ = [p_ for n_, p_ in named if n_.startswith("wm.")]                  # is much larger and would shrink the heads' share of one shared clip)
                    hd_ = [p_ for n_, p_ in named if not n_.startswith("wm.")]
                    gn = nn.utils.clip_grad_norm_(wm_, 1.0) + nn.utils.clip_grad_norm_(hd_, 1.0)
                else:
                    gn = nn.utils.clip_grad_norm_(A.trainable(), 1.0)
                opt.step(); n_steps += 1
            else:
                continue
            st = dict(pg=float(pg), kl=float(kl), kl_flow=float(klf), ratio=float(ratio.mean()), clipfrac=cf_sum / max(nb, 1), approx_kl=akl_sum / max(nb, 1),
                      grad_norm=float(gn), steps=n_steps, epoch=ep, kl_ref_discrete=kl_d_, kl_ref_continuous=kl_c_, kl_ref_flow=float(klf), ent_type=float(ent["type"]), ent_ptr=float(ent["ptr"]), ent_grasp=float(ent["grasp"]), **{f"gn_{k}": round(v, 4) for k, v in hg.items()})
            akl_sum, cf_sum, nb = 0.0, 0.0, 0
            if kl_stop > 0 and st["approx_kl"] > kl_stop:
                stopped = True; st["early_stop"] = 1
                break
    log(f"WM-actor GRPO update: {st}")
    return st


# ======================================================================================================================================
# Self-imitation of successful executed carries (FPW_SIL=1)
_SIL_M = {}


def _interp_mat(H, NP):
    """F.interpolate(mode="linear", align_corners=True) from H + 1 knots to NP points as a matrix (NP, H + 1), and the pseudo-inverse of its
    columns 1 .. H - 1 (knot 0 is the zero displacement, knot H the gauge)."""
    if (H, NP) not in _SIL_M:
        src = np.arange(NP) * H / (NP - 1); i0 = np.minimum(np.floor(src).astype(int), H); w = src - i0
        M = np.zeros((NP, H + 1)); M[np.arange(NP), i0] += 1 - w
        nz = i0 < H; M[np.arange(NP)[nz], i0[nz] + 1] += w[nz]
        _SIL_M[(H, NP)] = (M, np.linalg.pinv(M[:, 1:H]))
    return _SIL_M[(H, NP)]


def _arclen01(P):
    """polylines P (n, M, 3) -> normalised cumulative arc length (n, M), total length (n,)."""
    s_ = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(np.linalg.norm(np.diff(P, axis=1), axis=-1), 1)], 1)
    return s_ / np.maximum(s_[:, -1:], 1e-12), s_[:, -1]


def sil_targets(wpath, trajs, a_end, H):
    """the flow targets of executed carries. wpath (n, >= NPATH, 3) the WM's own (pre-edit) carry paths (root, table frame; point NPATH - 1
    = the goal); trajs: n executed root trajectories (T_i, 3) (carry start .. release, table frame); a_end (n, 3) each decision's own T2 end
    displacement (m; watch_path uses only the T2 shape, so this is the gauge, kept). The executed curve is read at the WM path's own
    normalised arc-length positions (its parametrisation kept), its endpoints blended onto the WM path's start / goal (only the shape is
    representable: watch_path pins both ends), and its residual about the start-goal line fitted by the T2 knots (least squares; exact for a
    path watch_path made). -> x1 (n, H, 3) PS units, dist_cm (n,) mean |target - WM path| (cm), ok (n,) bool."""
    from cleantables.worldmodel.base_model import PS
    NP = NPATH; n = len(trajs)
    W = np.asarray(wpath, np.float64)[:, :NP]; sW, totW = _arclen01(W)
    M, Mp = _interp_mat(H, NP); u = np.linspace(0.0, 1.0, NP)[:, None]
    x1 = np.zeros((n, H, 3), np.float32); dist = np.zeros(n, np.float32); ok = np.zeros(n, bool)
    for i in range(n):
        T = np.asarray(trajs[i], np.float64)
        if T.ndim != 2 or len(T) < 3 or not np.isfinite(T).all() or not np.isfinite(W[i]).all() or totW[i] < 0.01:
            continue
        sT, totT = _arclen01(T[None]); sT = sT[0]
        if totT[0] < 0.01:
            continue
        keep = np.concatenate([[True], np.diff(sT) > 1e-9]); sT, T_ = sT[keep], T[keep]
        tgt = np.stack([np.interp(sW[i], sT, T_[:, k]) for k in range(3)], -1)                  # the executed curve at the WM path's positions
        tgt = tgt + (1 - sW[i])[:, None] * (W[i, 0] - T[0]) + sW[i][:, None] * (W[i, -1] - T[-1])   # ends onto the WM path's start / goal
        r = tgt - (W[i, 0] + u * (W[i, -1] - W[i, 0]))                                      # the residual about the start-goal line
        a = np.asarray(a_end[i], np.float64)
        y = Mp @ (r + u * a[None] - M[:, H:H + 1] * a[None])                                # knots 1 .. H - 1 (knot H = the gauge a)
        x1[i] = (np.concatenate([y, a[None]], 0) / PS).astype(np.float32)
        dist[i] = 100 * np.linalg.norm(tgt - W[i], axis=-1).mean(); ok[i] = True
    return x1, dist, ok


def sil_select(D, takes, carries, ev, met_cm):
    """Mark each take decision whose carry succeeded (the same test for every object): its take not aborted, the carry released and not
    disturbed, its end within met_cm (xy) of the carry's planned end, nothing else pushed > 1 cm while held, no hand-hand overlap during it,
    no object put off the table by the take; its executed trajectory recorded (FPW_SIL in chain_env). Sets d["sil_traj"] (else None).
    takes {(env, g, step): take record}; carries {ev index: carry_metrics row}; ev = CS.ev. -> (carries completed, successful)."""
    n_c = n_ok = 0
    for d in D:
        d["sil_traj"] = None
        if d.get("typ") != 0 or d.get("x0") is None or d.get("wpath") is None:
            continue
        t_ = takes.get((d["env"], d["g"], d["step"]))
        if t_ is None or t_.get("ev") is None:
            continue
        c = carries.get(t_["ev"])
        if c is None:
            continue
        n_c += 1
        e_ = ev[t_["ev"]]
        if (t_.get("abort") is None and not c["disturbed"] and c["end_err_cm"] <= met_cm and c["push_held_cm"] <= 1.0 and c["hand_hand_s"] <= 0
                and not t_.get("lost") and e_.get("traj") is not None and not e_.get("traj_over", False) and len(e_["traj"]) >= 3):
            d["sil_traj"] = e_["traj"]; n_ok += 1
    return n_c, n_ok


def _encode_logged(A, LP, B):
    """the actor's encoding of logged decisions B (as logp_of builds it)."""
    dev = LP.dev
    bc, _, _, _ = build_batch(LP, np.array([b["tp"] for b in B]), np.stack([b["xyz"] for b in B]), np.stack([b["yaw"] for b in B]),
                              np.stack([b["palms"] for b in B]))
    if EMB_T is not None and B[0].get("inst") is not None:
        bc["task_emb"] = EMB_T[torch.as_tensor([b["tp"] for b in B], device=EMB_T.device), torch.as_tensor([b["inst"] for b in B], device=EMB_T.device)]
    rob = torch.as_tensor(np.stack([b["rob"] for b in B]), device=dev) if ROBOT_TOK and B[0].get("rob") is not None else None
    ept = torch.as_tensor(np.stack([b["ept"] for b in B]), device=dev) if EP_TOK and B[0].get("ept") is not None else None
    return A.encode(bc, rob, ept)


def sil_fm(A, E, obj, x0, x1, grad=False, scale=1.0, i0=None, steps=STEPS):
    """flow matching of the chosen channel toward x1 along the straight interpolant from its logged start x0 (the decision's own noise /
    warm start), the other channels replayed by the actor's own flow from x0 (as conditioning: detached, no backprop through time).
    E (n, ...) encoding (no grad); obj (n,); x0 (n, nobj, H, 3); x1 (n, H, 3) PS units. grad: backward per Euler step (scaled by
    scale / steps) into the T2 decoder. -> per-sample loss (n,) = mean over steps / knots / xyz of |v - (x1 - x0_j) / (1 - t0)|^2."""
    i0 = WARM_I0 if i0 is None else i0
    n = x0.shape[0]; ar = torch.arange(n, device=x0.device); dt = 1.0 / steps; t0 = i0 / steps
    xs = x0[ar, obj]; S = steps - i0
    x = x0.clone(); out = torch.zeros(n, device=x0.device)
    for i in range(i0, steps):
        fr = (i + 1 - i0) / S
        tgt = xs + fr * (x1 - xs)                                                           # the interpolant's state at t_{i+1}
        with torch.set_grad_enabled(grad):
            v = A.wm.velocity_all(E, x, torch.full((n,), i * dt, device=x0.device)).float()
            mu = x + v * dt
            li = (((tgt - mu[ar, obj]) / dt) ** 2).mean((-1, -2))
            if grad:
                (li.mean() * scale / S).backward()
        out = out + li.detach() / S
        x = mu.detach().clone(); x[ar, obj] = tgt
    return out


def sil_update(A, LP, D, opt_sil, log=print, seed=0, n_carries=0, bs=int(os.environ.get("FPW_BS", "64"))):
    """one self-imitation pass per iteration on the successful carries of D (sil_select): at most SIL_N (random subset), micro-batches of bs,
    one Adam step (opt_sil: the T2 decoder only) per SIL_ACC micro-batches, grad-norm clip 1.0. -> stats (also logged as one SIL line)."""
    import time as _t
    from cleantables.worldmodel.base_model import PS
    t_ = _t.time(); dev = LP.dev
    C = [d for d in D if d.get("sil_traj") is not None and d.get("x0") is not None and d.get("wpath") is not None]
    n_ok = len(C); rng = np.random.default_rng(10_000 + int(seed))
    if len(C) > SIL_N:
        C = [C[i] for i in rng.choice(len(C), SIL_N, replace=False)]
    st = dict(sil_carries=int(n_carries), sil_successful=n_ok, sil_share=n_ok / max(int(n_carries), 1), sil_used=0)
    if not C:
        log(f"SIL: carries {n_carries}, successful {n_ok} ({st['sil_share']:.3f}), used 0"); return st
    H = int(np.asarray(C[0]["xc"]).shape[-2])
    a_end = np.stack([np.asarray(d["xc"])[-1, H - 1] * PS for d in C])
    x1, dist, ok = sil_targets(np.stack([d["wpath"] for d in C]), [d["sil_traj"] for d in C], a_end, H)
    C = [c for c, o in zip(C, ok) if o]; x1 = x1[ok]; dist = dist[ok]
    if not C:
        log(f"SIL: carries {n_carries}, successful {n_ok} ({st['sil_share']:.3f}), used 0 (no valid target)"); return st
    T2 = [p_ for n_, p_ in A.named_parameters() if p_.requires_grad and n_.startswith("wm.g_")]

    def run(idx, grad, scale=1.0):
        Bb = [C[i] for i in idx]
        with torch.no_grad():
            E = _encode_logged(A, LP, Bb)
        obj = torch.tensor([b["obj"] for b in Bb], device=dev)
        x0 = torch.as_tensor(np.stack([b["x0"] for b in Bb]), device=dev); X1 = torch.as_tensor(x1[idx], device=dev)
        return sil_fm(A, E, obj, x0, X1, grad=grad, scale=scale)

    ev_idx = np.arange(min(256, len(C)))
    with torch.no_grad():
        l0 = float(torch.cat([run(ev_idx[k:k + bs], False) for k in range(0, len(ev_idx), bs)]).mean())
    mbs = np.array_split(rng.permutation(len(C)), max(1, len(C) // bs)); n_steps = 0; ltr = []
    for mi, idx in enumerate(mbs):
        if mi % SIL_ACC == 0:
            opt_sil.zero_grad()
        ltr.append(float(run(idx, True, scale=SIL_W / min(SIL_ACC, len(mbs))).mean()))
        if mi % SIL_ACC == SIL_ACC - 1 or mi == len(mbs) - 1:
            gn = float(nn.utils.clip_grad_norm_(T2, 1.0)); opt_sil.step(); n_steps += 1
    with torch.no_grad():
        l1 = float(torch.cat([run(ev_idx[k:k + bs], False) for k in range(0, len(ev_idx), bs)]).mean())
    st.update(sil_used=len(C), sil_loss_before=l0, sil_loss_after=l1, sil_loss_train=float(np.mean(ltr)), sil_dist_cm=float(dist.mean()),
              sil_steps=n_steps, sil_grad_norm=gn, sil_s=_t.time() - t_)
    log(f"SIL: carries {n_carries}, successful {n_ok} ({st['sil_share']:.3f}), used {len(C)}, loss before {l0:.4f} / after {l1:.4f} "
        f"(train {st['sil_loss_train']:.4f}), mean executed-vs-planned path distance {st['sil_dist_cm']:.2f} cm; {n_steps} steps, {st['sil_s']:.1f} s")
    return st
