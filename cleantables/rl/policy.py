"""Small action policy on top of the frozen world model, acting in the `cleantables.rl.chain_env` environment.

Per decision (env, gripper), one forward pass over the table's objects:
  type     take / wait / move / done
  take     object (pointer over the free objects), grasp yaw (4 top-down yaws), route (one of global_plan's 9 scored candidates to the
           fixed target), path edit (5 spline control points, xyz residuals in [-R_EDIT, R_EDIT]), release u in (0, 1], retreat dz in
           [0, 0.2] m
  move     a free hand target (x, y, z, yaw) inside the workspace box
The goal of a take is the frozen judge's target (xy) at the world model's own goal height, so goals come only from the frozen model.
PolicyDecider samples (or takes the mode of) the policy and records every decision (obs, action, log-prob) for the update; its
baseline mode emits the non-learned baseline's action in the same format as behaviour-cloning data (the WM's object, default grasp,
cheapest route, no edit, u = 1, default retreat). bc_train is the imitation init; grpo_update applies group-relative policy updates
(groups = episodes from the same start state) with a KL penalty to the frozen BC policy. Rewards: episode_reward_v2().
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cleantables.rl.chain_env import GYAWS, edit_path

N_CTRL, NC = 5, 9                       # spline control points, route candidates (GP_SMALL)
R_EDIT = float(os.environ.get("FPP_R_EDIT", "0.15"))     # path edit range (m)
EP_S = float(os.environ.get("FPR_EP_S", "90"))                  # episode time limit (s)
TYPES = ("take", "wait", "move", "done")
FO, FG_, FC = 25, 18, 7                 # per-object, global and per-candidate feature sizes


# ======================================================================================================================================
def obs_batch(CS, J, LP, tpl_env, envs, g, Pn, yw, busy, n_car, max_carries, ep_start, wm=None):
    """Observation of gripper g in envs (n,): per object (n, K, FO), global (n, FG_), object mask (n, K) (free to take). wm: the live WM
    read-out for these rows (LivePlanner._wm_fast); its p_now is a feature (the WM as the prior)."""
    fp = CS.fp; e = np.asarray(envs); n = len(e); K = fp.K
    org = fp.org[torch.as_tensor(e, device=fp.dev)].cpu().numpy()
    P = Pn[e]; Y = yw[e]; ext = fp.ext.cpu().numpy()[e]; valid = fp.valid.cpu().numpy()[e] & np.isfinite(P[..., 0])
    tg = J.targets(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw))[e]; sc = J.score(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw))
    hP = CS.hP.cpu().numpy()[e] - org[:, None]; ph = CS.phase.cpu().numpy()[e]
    own, oth = hP[:, g], hP[:, 1 - g]
    xyz = np.nan_to_num(P[..., :3]); yb = J._geo_consts()[1][e] if hasattr(J, "_geo_consts") else 0.0
    yl = np.nan_to_num(Y) + yb
    pn = np.zeros((n, K), np.float32)
    if wm is not None:
        for r in range(n):
            I = int(wm["I"][r]); pn[r, :I] = np.maximum(np.nan_to_num(wm["p_now"][r, :I]), 0)
    top = xyz.copy(); top[..., 2] += np.nan_to_num(ext[..., 2])
    d_t = (tg[..., :2] - top[..., :2])
    fo = np.concatenate([valid[..., None], xyz, np.sin(2 * yl)[..., None], np.cos(2 * yl)[..., None], np.nan_to_num(ext), d_t,
                         sc["dist_cm"][e][..., None] / 10, sc["met"][e][..., None], J.static[e][..., None], busy[e][..., None], pn[..., None],
                         xyz - own[:, None], (xyz - oth[:, None])[..., :2]], -1).astype(np.float32)
    oph = np.eye(6, dtype=np.float32)[np.clip(ph[:, 1 - g], 0, 5)]
    t_ep = (CS.step_i - np.asarray(ep_start)[e]) * CS.dt / 60.0
    fg = np.concatenate([own, oth, oph, t_ep[:, None], (np.asarray(n_car)[e] / max(max_carries, 1))[:, None], np.full((n, 1), float(g)),
                         np.full((n, 1), CS.dt * 0)], -1).astype(np.float32)[:, :FG_]
    if fg.shape[1] < FG_:
        fg = np.concatenate([fg, np.zeros((n, FG_ - fg.shape[1]), np.float32)], 1)
    # reach (franka_kinematics IK, top-down, yaw free): own arm / other arm, at the object now and at its target
    if CS.arms is not None:
        rs = [CS.arms.reach(e, gg, top, kind="start") for gg in (g, 1 - g)]; rg = [CS.arms.reach(e, gg, tg, kind="target") for gg in (g, 1 - g)]
        reach = np.stack([rs[0], rg[0], rs[1], rg[1]], -1).astype(np.float32)
    else:
        reach = np.ones((n, K, 4), np.float32)
    fo = np.concatenate([fo, reach], -1)
    assert fo.shape[-1] == FO, fo.shape
    take_ok = valid & ~busy[e] & ~(sc["met"][e] & ~J.static[e]) & J.mov[e] & ~J.static[e] & (reach[..., 0] > 0) & (reach[..., 1] > 0)
    return dict(fo=fo, fg=fg, mask=take_ok, valid=valid, tg=tg, top=top)


class Policy(nn.Module):
    def __init__(self, d=128, L=2, heads=4):
        super().__init__()
        self.obj = nn.Sequential(nn.Linear(FO, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(FG_, d), nn.GELU(), nn.Linear(d, d))
        self.trunk = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, batch_first=True, norm_first=True), L)
        self.type_h = nn.Linear(d, len(TYPES)); self.ptr_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.grasp_h = nn.Linear(d, len(GYAWS))
        self.cand = nn.Sequential(nn.Linear(FC, d), nn.GELU(), nn.Linear(d, d)); self.route_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.edit_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3 * N_CTRL + 2))     # residual means, u, retreat (pre-squash)
        self.move_h = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.log_std = nn.Parameter(torch.full((3 * N_CTRL + 2 + 4,), -1.0))
        nn.init.zeros_(self.edit_h[-1].weight); nn.init.zeros_(self.edit_h[-1].bias)              # starts at: no edit
        with torch.no_grad():
            self.edit_h[-1].bias[3 * N_CTRL] = 3.0; self.edit_h[-1].bias[3 * N_CTRL + 1] = 0.0     # u ~ sigmoid(3) = 0.95, retreat 0.1 m

    def encode(self, fo, fg, valid):
        x = torch.cat([self.glob(fg)[:, None], self.obj(fo)], 1)
        pad = torch.cat([torch.zeros(len(fo), 1, dtype=torch.bool, device=fo.device), ~valid], 1)
        y = self.trunk(x, src_key_padding_mask=pad)
        return y[:, 0], y[:, 1:]

    def stage1(self, fo, fg, valid, mask):
        gt, ot = self.encode(fo, fg, valid)
        tl = self.type_h(gt)
        tl = tl.masked_fill(~mask.any(1)[:, None] & (torch.arange(len(TYPES), device=fo.device) == 0)[None], -1e4)   # no take without a free object
        pl = self.ptr_h(torch.cat([ot, gt[:, None].expand_as(ot)], -1)).squeeze(-1).masked_fill(~mask, -1e4)
        return gt, ot, tl, pl

    def grasp_logits(self, ot_k):
        return self.grasp_h(ot_k)

    def stage2(self, gt, ot_k, cf):
        """ot_k (n, d) the chosen object's token; cf (n, NC, FC) candidate features -> route logits (n, NC), edit params (n, 3 N_CTRL + 2)."""
        ct = self.cand(cf)
        rl = self.route_h(torch.cat([ct, ot_k[:, None].expand_as(ct)], -1)).squeeze(-1)
        ed = self.edit_h(torch.cat([ot_k, gt], -1))
        return rl, ed

    def move_mean(self, gt):
        return self.move_h(gt)


def gauss_logp(x, mu, log_std):
    log_std = log_std.clamp(-2.5, 0.5)                                                            # bounded spread (no collapse on BC data)
    return (-0.5 * ((x - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)


def cand_feats(cands, costs, parts, start):
    """(n, NC, FC): exact cost, hit, band, econ, table, lift (m), length (m) per candidate."""
    out = np.zeros((len(cands), NC, FC), np.float32)
    for r, (c, co, pa) in enumerate(zip(cands, costs, parts)):
        m = min(len(c), NC)
        lift = c[:m, :, 2].max(1) - c[:m, 0, 2]; ln = np.linalg.norm(np.diff(c[:m], axis=1), axis=-1).sum(1)
        out[r, :m] = np.stack([np.minimum(co[:m], 50), pa.get("hit", np.zeros(m))[:m], pa.get("band", np.zeros(m))[:m], pa.get("econ", np.zeros(m))[:m],
                               pa.get("table", np.zeros(m))[:m], lift, ln], -1)
    return out


# ======================================================================================================================================
class PolicyDecider:
    """mode 'policy' (sample), 'greedy' (the mode) or 'baseline' (the non-learned baseline's action, recorded in the policy's format)."""

    def __init__(self, pol, LP, J, CS, tpl_env, gtab, mode="policy", retreat0=0.1, ws_box=None, dev="cuda"):
        from cleantables.worldmodel.physics import global_plan as GP
        from cleantables.worldmodel.physics import route_opt as RO
        from cleantables.worldmodel.physics.geo_cost import MeshCost
        self.pol, self.LP, self.J, self.CS, self.tpl_env, self.G, self.mode = pol, LP, J, CS, np.asarray(tpl_env), gtab, mode
        self.GP, self.RO = GP, RO
        RO.HAND, RO.PLACE, RO.BAND = 1, 1, 0.02
        self.ts = RO.TorchSDF(MeshCost(w_drag=0.0), torch.device(dev)); self.dev = dev
        self.retreat0 = retreat0; self.log = []; self.ctx = None
        self.ws_box = ws_box

    def set_ctx(self, n_car, max_carries, ep_start):
        self.ctx = (n_car, max_carries, ep_start)

    @torch.no_grad()
    def decide(self, envs, Pn, yw, busy, g=0):
        e = np.asarray(envs); n = len(e); tp = self.tpl_env[e]; K = Pn.shape[1]
        n_car, mc, ep_start = self.ctx
        wm = self.LP._wm_fast(tp, np.nan_to_num(Pn[e, :, :3]), np.nan_to_num(yw[e]))
        ob = obs_batch(self.CS, self.J, self.LP, self.tpl_env, e, g, Pn, yw, busy, n_car, mc, ep_start, wm=wm)
        dev = self.dev
        fo, fg = torch.as_tensor(ob["fo"], device=dev), torch.as_tensor(ob["fg"], device=dev)
        valid, mask = torch.as_tensor(ob["valid"], device=dev), torch.as_tensor(ob["mask"], device=dev)
        gt, ot, tl, pl = self.pol.stage1(fo, fg, valid, mask)
        # ---- type and object
        if self.mode == "baseline":                                       # the WM's choice among the free objects; STOP -> done
            pn = np.where(ob["mask"], np.pad(wm["p_now"], ((0, 0), (0, max(0, K - wm["p_now"].shape[1]))))[:, :K], -1)
            obj = pn.argmax(1); typ = np.where(wm["stop"] | (pn.max(1) <= 1e-6), 3, 0)
        elif self.mode == "greedy":
            typ = tl.argmax(1).cpu().numpy(); obj = pl.argmax(1).cpu().numpy()
        else:
            typ = torch.distributions.Categorical(logits=tl).sample().cpu().numpy(); obj = torch.distributions.Categorical(logits=pl).sample().cpu().numpy()
        ar = torch.arange(n, device=dev); objt = torch.as_tensor(obj, device=dev)
        lp = torch.log_softmax(tl, -1)[ar, torch.as_tensor(typ, device=dev)]
        take = typ == 0
        lp = lp + torch.where(torch.as_tensor(take, device=dev), torch.log_softmax(pl, -1)[ar, objt], torch.zeros_like(lp))
        ot_k = ot[ar, objt]
        gl = self.pol.grasp_logits(ot_k)
        if self.mode == "baseline":
            gi = np.zeros(n, int)
            for r in range(n):                                            # the default short-axis grasp's index in GYAWS
                g0 = self.G[tp[r], obj[r]] if np.isfinite(self.G[tp[r], obj[r]]) else 0.0
                gi[r] = int(np.argmin([abs(math.remainder(g0 - x, math.pi)) for x in GYAWS]))
        elif self.mode == "greedy":
            gi = gl.argmax(1).cpu().numpy()
        else:
            gi = torch.distributions.Categorical(logits=gl).sample().cpu().numpy()
        lp = lp + torch.where(torch.as_tensor(take, device=dev), torch.log_softmax(gl, -1)[ar, torch.as_tensor(gi, device=dev)], torch.zeros_like(lp))
        # ---- candidates for the takes (batched across rows): start = the object now, end = the frozen target xy at the WM's goal height
        items, rows = [], np.flatnonzero(take)
        for r in rows:
            t, a = tp[r], obj[r]; B = self.LP.B[t]; I = B.I
            st = np.nan_to_num(Pn[e[r], a, :3]).astype(np.float64); gz = float(wm["G"][r, a, 2])
            gl_ = np.array([ob["tg"][r, a, 0], ob["tg"][r, a, 1], gz]); u = np.linspace(0, 1, 33)[:, None]
            items.append(dict(S=B.S, xyz=np.nan_to_num(Pn[e[r], :I, :3]).astype(np.float64), yaw=np.nan_to_num(yw[e[r], :I]).astype(np.float64), a=int(a),
                              plans=(st[None] + u * (gl_ - st)[None])[None].astype(np.float32), dyaws=np.array([wm["dy"][r, a]], np.float32),
                              gyaw=float(GYAWS[gi[r]])))
        res = [None] * n; rec = []
        if items:
            with torch.enable_grad():
                cands, costs, parts = self.GP.candidates_scored(self.ts, items)
            cf = torch.as_tensor(cand_feats(cands, costs, parts, None), device=dev)
            rt = torch.as_tensor(rows, device=dev)
            rl, ed = self.pol.stage2(gt[rt], ot_k[rt], cf)
            rl = rl.masked_fill(~torch.isfinite(torch.as_tensor(np.stack([np.pad(c, (0, NC - len(c)), constant_values=np.inf)[:NC] for c in costs]), device=dev)), -1e4)
            ls = self.pol.log_std[:3 * N_CTRL + 2]
            if self.mode == "baseline":
                ri = np.array([int(np.argmin(c)) for c in costs]); z = torch.zeros_like(ed)
                z[:, 3 * N_CTRL] = 3.0; z[:, 3 * N_CTRL + 1] = 0.0                      # u ~ 0.95 -> 1 below, retreat 0.1 m
            elif self.mode == "greedy":
                ri = rl.argmax(1).cpu().numpy(); z = ed
            else:
                ri = torch.distributions.Categorical(logits=rl).sample().cpu().numpy(); z = ed + ls.exp() * torch.randn_like(ed)
            lpr = torch.log_softmax(rl, -1)[torch.arange(len(rows), device=dev), torch.as_tensor(ri, device=dev)] + gauss_logp(z, ed, ls)
            lp[rt] = lp[rt] + lpr
            zc = z.cpu().numpy()
            resid = np.tanh(zc[:, :3 * N_CTRL]).reshape(-1, N_CTRL, 3) * R_EDIT
            u_rel = 1.0 / (1.0 + np.exp(-zc[:, 3 * N_CTRL])); u_rel = np.where(u_rel > 0.9, 1.0, u_rel)                # > 0.9: at the goal
            ret = 0.2 / (1.0 + np.exp(-zc[:, 3 * N_CTRL + 1]))
            if self.mode == "baseline":
                resid[:] = 0.0; u_rel[:] = 1.0; ret[:] = self.retreat0
            base = np.stack([cands[q][ri[q]] for q in range(len(rows))])
            paths = edit_path(base, resid)
            for q, r in enumerate(rows):
                a = obj[r]
                res[r] = dict(type="take", carried=int(a), path=paths[q], dyaw=float(wm["dy"][r, a]), gyaw=float(GYAWS[gi[r]]), rel_u=float(u_rel[q]),
                              retreat=float(ret[q]), prior=cands[q][int(np.argmin(costs[q]))])
                rec.append((r, dict(route=int(ri[q]), z=zc[q], cf=cf[q].cpu().numpy())))
        mv = np.flatnonzero(typ == 2)
        mz = None
        if len(mv):
            mm = self.pol.move_mean(gt[torch.as_tensor(mv, device=dev)]); ls2 = self.pol.log_std[3 * N_CTRL + 2:]
            mz = mm if self.mode == "greedy" else mm + ls2.exp() * torch.randn_like(mm)
            lp[torch.as_tensor(mv, device=dev)] += gauss_logp(mz, mm, ls2)
            box = self.ws_box[tp[mv]] if self.ws_box is not None else np.tile([[-0.6, 0.6, -0.4, 0.4]], (len(mv), 1))
            zz = torch.tanh(mz).cpu().numpy()
            for q, r in enumerate(mv):
                x = box[q, 0] + (zz[q, 0] + 1) / 2 * (box[q, 1] - box[q, 0]); y = box[q, 2] + (zz[q, 1] + 1) / 2 * (box[q, 3] - box[q, 2])
                res[r] = dict(type="move", P=np.array([x, y, 0.05 + (zz[q, 2] + 1) / 2 * 0.55], np.float32), Y=float(zz[q, 3] * math.pi))
        for r in np.flatnonzero(typ == 1):
            res[r] = dict(type="wait")
        lpc = lp.cpu().numpy()
        recd = dict(rec)
        for r in range(n):                                                # every decision, for the update (baseline: the BC targets)
            self.log.append(dict(env=int(e[r]), g=int(g), step=int(self.CS.step_i), fo=ob["fo"][r], fg=ob["fg"][r], valid=ob["valid"][r], mask=ob["mask"][r],
                                 typ=int(typ[r]), obj=int(obj[r]), gi=int(gi[r]), logp=float(lpc[r]), mz=None if mz is None or r not in mv else
                                 mz[list(mv).index(r)].cpu().numpy(), **recd.get(r, {})))
        return res


# ======================================================================================================================================
def logp_of(pol, B, dev="cuda"):
    """the log-prob of logged decisions B (list of dicts) under pol (with grad)."""
    fo = torch.as_tensor(np.stack([b["fo"] for b in B]), device=dev); fg = torch.as_tensor(np.stack([b["fg"] for b in B]), device=dev)
    valid = torch.as_tensor(np.stack([b["valid"] for b in B]), device=dev); mask = torch.as_tensor(np.stack([b["mask"] for b in B]), device=dev)
    gt, ot, tl, pl = pol.stage1(fo, fg, valid, mask)
    n = len(B); ar = torch.arange(n, device=dev)
    typ = torch.tensor([b["typ"] for b in B], device=dev); obj = torch.tensor([b["obj"] for b in B], device=dev); gi = torch.tensor([b["gi"] for b in B], device=dev)
    take = typ == 0
    lp = torch.log_softmax(tl, -1)[ar, typ]
    ot_k = ot[ar, obj]
    lp = lp + torch.where(take, torch.log_softmax(pl, -1)[ar, obj] + torch.log_softmax(pol.grasp_logits(ot_k), -1)[ar, gi], torch.zeros_like(lp))
    rows = [i for i, b in enumerate(B) if b["typ"] == 0 and "route" in b]
    if rows:
        rt = torch.as_tensor(rows, device=dev)
        cf = torch.as_tensor(np.stack([B[i]["cf"] for i in rows]), device=dev)
        rl, ed = pol.stage2(gt[rt], ot_k[rt], cf)
        ri = torch.tensor([B[i]["route"] for i in rows], device=dev); z = torch.as_tensor(np.stack([B[i]["z"] for i in rows]), device=dev)
        lp = lp.index_add(0, rt, torch.log_softmax(rl, -1)[torch.arange(len(rows), device=dev), ri] + gauss_logp(z, ed, pol.log_std[:3 * N_CTRL + 2]))
    mrows = [i for i, b in enumerate(B) if b["typ"] == 2 and b.get("mz") is not None]
    if mrows:
        rt = torch.as_tensor(mrows, device=dev); mz = torch.as_tensor(np.stack([B[i]["mz"] for i in mrows]), device=dev)
        lp = lp.index_add(0, rt, gauss_logp(mz, pol.move_mean(gt[rt]), pol.log_std[3 * N_CTRL + 2:]))
    return lp


def bc_train(pol, B, steps=2000, lr=3e-4, bs=512, dev="cuda", log=print, explore=0.03):
    """imitation init: maximise the log-prob of the baseline's logged actions (the edit / u / retreat heads keep their no-edit init). The
    baseline never waits or moves freely: a small floor (explore: the target share of wait / move) keeps them explorable for RL."""
    opt = torch.optim.Adam(pol.parameters(), lr=lr); rng = np.random.default_rng(0)
    for it in range(steps):
        idx = rng.integers(0, len(B), min(bs, len(B))); Bb = [B[i] for i in idx]
        loss = -logp_of(pol, Bb, dev).mean()
        if explore > 0:
            fo = torch.as_tensor(np.stack([b["fo"] for b in Bb]), device=dev); fg = torch.as_tensor(np.stack([b["fg"] for b in Bb]), device=dev)
            va = torch.as_tensor(np.stack([b["valid"] for b in Bb]), device=dev); ma = torch.as_tensor(np.stack([b["mask"] for b in Bb]), device=dev)
            _, _, tl, _ = pol.stage1(fo, fg, va, ma)
            loss = loss - explore * torch.log_softmax(tl, -1)[:, 1:3].mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if it % 200 == 0:
            log(f"BC {it}: nll {float(loss):.3f}")
    return pol


def episode_reward_v2(ep, carries, moves, w):
    """Task-dominated episode reward: per mover 2 x progress (1 - d / d0, uncapped below) + 1 per mover at met_3d + 0.5 if all met_3d.
    Costs on the same order: per second (arm contacts, hand-hand, hand-object overlap, fingers in the table) 0.05, IK failure 0.02, time
    0.01 (a robot-caused termination is charged the whole 90 s horizon), per push held 0.5, per approach / free-move push 0.1, per carry
    attempt 0.02, per hand-object penetration abort 0.5, per robot-lost object 2, per cm of static motion 0.1, per unmet mover at a stop
    or a robot-caused termination 0.5."""
    term = ep.get("stop") in ("stop", "lost_robot")
    w_aa = float(os.environ.get("FPR_W_ARMARM", "0.05"))                 # arm-arm contact per second
    w_tc = float(os.environ.get("FPR_W_TOUCH", "0.0"))                   # arm-arm overlap per s (arm_touch_s; separate from the shield's stops)
    w_ao = float(os.environ.get("FPR_W_ARMOBJ", "0.05"))                 # arm-object / arm-table contact per s
    w_mp = float(os.environ.get("FPR_W_MPUSH", "0.1"))                   # approach / free-move push
    w_st = float(os.environ.get("FPR_W_STATIC_V2", "0.1"))               # static-object motion per cm
    w_met = float(os.environ.get("FPR_W_MET", "1.0")); w_all = float(os.environ.get("FPR_W_ALLMET", "0.5"))   # placing reward weights
    return (2.0 * ep.get("progress", 0.0) + w_met * ep.get("n_met3", 0) + w_all * float(ep.get("all_met3", ep["all_met"]))
            - w_aa * ep.get("arm_arm_s", 0.0) - w_tc * ep.get("arm_touch_s", 0.0)
            - float(os.environ.get("FPR_W_NEAR_NOW", os.environ.get("FPR_W_NEAR", "0"))) * ep.get("near_s", 0.0)   # fading closeness cost
            - w_ao * (ep.get("arm_obj_s", 0.0) + ep.get("arm_table_s", 0.0))
            - 0.05 * (ep.get("hand_hand_s", 0.0) + ep.get("hand_obj_overlap_s", 0.0) + ep.get("finger_table_s", 0.0)) - 0.02 * ep.get("ik_fail_s", 0.0)
            - 0.01 * (EP_S if ep.get("stop") == "lost_robot" else ep["makespan_s"])
            - 0.5 * sum(c["push_held_cm"] > 1 for c in carries) - w_mp * sum(m["push_cm"] > 1 for m in moves) - 0.02 * ep["carries"]
            - 0.5 * ep.get("hand_obj_aborts", 0) - 2.0 * ep.get("n_lost_robot", 0) - w_st * ep.get("static_moved_cm", 0.0)
            - float(os.environ.get("FPR_W_UNMET", "0.5")) * ep.get("n_unmet", 0) * float(term))   # stopping with objects left unplaced


def episode_reward(ep, carries, moves, w):
    """Earlier uniform episode reward with weights w: - mean goal error (capped, /10 cm) - pushes while held (> 1 cm) - hand-move pushes
    - static objects moved - hand overlap (s) - time (s) - carries + all-met bonus."""
    return (-ep["goal_err_cm"] / 10.0 - w["push"] * sum(c["push_held_cm"] > 1 for c in carries) - w["mpush"] * sum(m["push_cm"] > 1 for m in moves)
            - w["static"] * ep["static_moved_gt1"] - w["hh"] * ep["hh_steps"] * w["dt"] - w["time"] * ep["makespan_s"] - w["carry"] * ep["carries"]
            - w.get("arm", 0.0) * (ep.get("arm_obj_s", 0.0) + ep.get("arm_arm_s", 0.0) + ep.get("arm_table_s", 0.0)) - w.get("ik", 0.0) * ep.get("ik_fail_s", 0.0)
            - w.get("ft", 0.0) * ep.get("finger_table_s", 0.0) - w.get("lost", 0.0) * ep.get("n_lost_robot", 0)
            - w.get("hpen", 0.0) * ep.get("hand_obj_aborts", 0) - w.get("hov", 0.0) * ep.get("hand_obj_overlap_s", 0.0)
            - w.get("static_cm", 0.0) * ep.get("static_moved_cm", 0.0)
            - w.get("early", 0.0) * ep.get("n_unmet", 0) * float(ep.get("stop") == "stop")
            + w.get("met3", 0.0) * ep.get("met3", 0.0)
            + w["all_met"] * float(ep["all_met3"] if (w.get("met3", 0.0) > 0 and "all_met3" in ep) else ep["all_met"]))


def grpo_update(pol, ref, opt, B, adv, epochs=4, bs=512, clip=0.2, beta_kl=0.05, dev="cuda", log=print):
    """PPO-clipped policy update on logged decisions B with group-relative advantages adv (per decision), KL leash to ref (k3 estimator)."""
    old = torch.tensor([b["logp"] for b in B], device=dev); A = torch.as_tensor(adv, dtype=torch.float32, device=dev)
    with torch.no_grad():
        lref = torch.cat([logp_of(ref, B[i:i + 2048], dev) for i in range(0, len(B), 2048)])
    rng = np.random.default_rng(0); st = {}
    for ep in range(epochs):
        for idx in np.array_split(rng.permutation(len(B)), max(1, len(B) // bs)):
            lp = logp_of(pol, [B[i] for i in idx], dev); it = torch.as_tensor(idx, device=dev)
            ratio = (lp - old[it]).exp()
            pg = -torch.minimum(ratio * A[it], ratio.clamp(1 - clip, 1 + clip) * A[it]).mean()
            lr_ = lref[it] - lp; kl = (lr_.exp() - 1 - lr_).mean()
            loss = pg + beta_kl * kl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            st = dict(pg=float(pg), kl=float(kl), ratio=float(ratio.mean()))
    log(f"GRPO update: {st}")
    return st
