"""Shared two-arm low-level controller under the world-model planner (enabled with FPL_ON=1).

One network drives both arms every decision tick (0.1 s), batched over all envs. Per arm it outputs a path offset (3-D, tanh x
FPL_R_OFF, default 25 cm, room to dodge the other arm), a gripper yaw turn (tanh x 90 deg) and a speed (0..1.5 x the plan's; 0 = pause).
`cleantables.rl.chain_env` applies the offset / turn through sin(pi u) of the move's progress (zero at the start and end of every
approach / free move / carry), so pick-ups and set-downs stay where the world model's plan puts them, and advances the move's clock by
the speed. The arm shield stays active. The output layer starts at zero: offset 0, turn 0, speed 1 = the plan.

Reward per tick (task-agnostic, the same rule for every object): progress = the drop in the movers' summed xy distance to their fixed
goals (0.1 / cm); closeness -0.3 x (8 cm - arm gap)+ / 8 cm per s; -0.1 per shield yield; -0.5 per aborted take; per s -0.05 arm-object
contact, -0.5 hand-object overlap, -0.05 fingers in the table; -0.01 / s time. At an episode end: +1 per mover placed, +0.5 if all,
-FPR_W_UNMET per unplaced mover at a stop or robot-caused end, -2 per object the robot put off the table. Trained with PPO and a value
network (critic), GAE (gamma 0.995, lambda 0.95). The observation includes the episode time left (time-outs end episodes).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn

TICK_S = 0.1  # set in LowPol.__init__ to chain_env.TICK x dt (0.1 s with FPC_TICK=12 at 120 Hz)
R_OFF, R_YAW, SPD_MAX = float(os.environ.get("FPL_R_OFF", "0.25")), math.pi / 2, 1.5  # 10 cm leaves too little room to dodge
SPD_B0 = math.log(2.0)                       # sigmoid(0 + ln 2) * 1.5 = 1.0: speed 1 at zero output
A_PER_ARM = 5                                # offset xyz, yaw, speed
W_PROG, W_NEAR, W_YIELD, W_ABORT, W_TIME = float(os.environ.get("FPL_W_PROG") or "0.1"), float(os.environ.get("FPL_W_NEAR") or "0.3"), 0.1, 0.5, 0.01  # FPL_W_NEAR: closeness cost per s within NEAR_M
W_ARMOBJ, W_HOV, W_FT = float(os.environ.get("FPL_W_ARMOBJ", "0.05")), 0.5, 0.05  # per s: arm-object contact, hand-object overlap, fingers in the table
# Other objects pushed: per cm an object not held moves in a tick beyond 0.2 cm; not in the 1 s after it leaves a hand, not within
# 3 s of a test knock / drop in its env, not while the table settles at the start. 0.15 / cm > progress's 0.1 / cm, so pushing a
# mover toward its goal never pays. Also charged: the two grippers overlapping (per s) and the two arms overlapping (per s; exact
# link padding, separate from the shield's stops, which have their own cost).
W_MET_L = float(os.environ.get("FPL_W_MET") or "1.0")  # end-of-episode placing weight for the controller (x n_met3; all-met
                                                       # gets half of it); the planner's weight is FPR_W_MET
W_DEV = float(os.environ.get("FPL_W_DEV") or "0")  # per m of offset from the WM path, per s, on steered arms: the path
                                                   # as a soft prior (0 = no cost)
W_PUSH, PUSH_MIN_CM, W_HH, W_TOUCH = float(os.environ.get("FPL_W_PUSH", "0.15")), 0.2, 0.5, float(os.environ.get("FPL_W_TOUCH") or "1.0")
# The arm touching the table costs like the arm touching any object (W_ARMOBJ per s); the table enters the obs as one more
# object-like slot (centre, half-extents, valid).
W_ARMTAB = W_ARMOBJ
PUSH_CONTACT = os.environ.get("FPC_PUSH_CONTACT", "0") == "1"  # charge a push only for objects the robot touched in the last PUSH_RECENT_S
ZERO_OFF = os.environ.get("FPL_ZERO_OFF", "0") == "1"  # evaluation only: the controller's path offsets forced to 0
JOINTS = os.environ.get("FPL_ACTION", "offset") == "joints"  # joint-space controller (cleantables.rl.joint_control); unset = path offsets
if JOINTS:
    from cleantables.rl import joint_control as JI                                # noqa: E402
PUSH_RECENT_S = float(os.environ.get("FPC_PUSH_RECENT_S", "0.5"))
GAMMA, LAM = float(os.environ.get("FPL_GAMMA", "0.995")), float(os.environ.get("FPL_LAM", "0.95"))
EPOCHS, MB = int(os.environ.get("FPL_EPOCHS", "4")), int(os.environ.get("FPL_MB", "8192"))
LR, CLIP, ENT_W, VF_W = float(os.environ.get("FPL_LR", "3e-4")), 0.2, float(os.environ.get("FPL_ENT", "0.001")), 0.5
KL_STOP = float(os.environ.get("FPL_KL_STOP", "0"))  # stop a PPO update once its approx KL exceeds this (0 = off)
UPDATE_TICKS = int(os.environ.get("FPL_UPDATE_TICKS", "0"))  # a PPO update every this many ticks during the training rollout
                                                             # (100 = 10 s of sim), bootstrapped from the value of the newest
                                                             # state; 0 = once per iteration
LS0_OFF, LS0_SPD = float(os.environ.get("FPL_LS0", "-2.5")), float(os.environ.get("FPL_LS0_SPD", "-2.0"))  # small initial exploration noise
# Smooth exploration: the sampling noise is an AR(1) process with time constant NOISE_TAU s (each step's marginal stays N(0, 1), so
# the Gaussian log-prob PPO uses is unchanged; cf. gSDE, Raffin et al., CoRL 2021): an offset / slow-down persists long enough to
# change what happens. 0 = white noise. FPL_LS_RESET=1 sets log_std to FPL_LS0 / FPL_LS0_SPD when resuming.
NOISE_TAU = float(os.environ.get("FPL_NOISE_TAU", "0"))
LS_RESET = os.environ.get("FPL_LS_RESET", "0") == "1"
LS_SET = os.environ.get("FPL_LS_SET", "")  # "i:v,i:v": set those log-sd entries once when loading ("" = none)


class LowNet(nn.Module):
    def __init__(self, d_obs, d_act=2 * A_PER_ARM, h=256):
        super().__init__()
        self.pi = nn.Sequential(nn.LayerNorm(d_obs), nn.Linear(d_obs, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, d_act))
        self.v = nn.Sequential(nn.LayerNorm(d_obs), nn.Linear(d_obs, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.pi[-1].weight); nn.init.zeros_(self.pi[-1].bias)   # zero output: offset 0, turn 0, speed 1 (the plan)
        ls0 = torch.tensor(([LS0_OFF] * 4 + [LS0_SPD]) * 2)                              # per arm: offset xyz, yaw, speed
        self.log_std = nn.Parameter(ls0.clone())

    def forward(self, o):
        return self.pi(o), self.v(o).squeeze(-1)


def squash(a):
    """raw actions (..., 10) -> offset (..., 2, 3) m, yaw (..., 2) rad, speed (..., 2)."""
    a = a.reshape(*a.shape[:-1], 2, A_PER_ARM)
    spd = torch.round(torch.sigmoid(a[..., 4] + SPD_B0) * SPD_MAX * 1e4) / 1e4  # 0.0001 steps: zero output gives exactly 1.0, so the plan is untouched
    return torch.tanh(a[..., :3]) * R_OFF, torch.tanh(a[..., 3]) * R_YAW, spd


def gauss_logp(a, mu, ls):
    return (-0.5 * ((a - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)


class LowPol:
    """The controller, its rollout buffer and PPO. Attached to ChainSim as CS.lowpol; run_continuous calls tick() at the end of every
    decision tick; cleantables.rl.train calls begin() / update() / state_dict()."""

    def __init__(self, CS, judge, dev):
        self.CS, self.J, self.dev = CS, judge, dev
        fp = CS.fp; self.N, self.K = fp.N, fp.K
        assert CS.arms is not None, "the low-level controller needs the arms (no --no_arms)"
        from cleantables.rl import chain_env as FC_
        global TICK_S
        TICK_S = FC_.TICK * CS.dt
        self.ep_s = float(os.environ.get("FPR_EP_S", "90"))
        self.K_obs = max(self.K, int(os.environ.get("FPL_K_OBS", "10")))  # fixed number of object slots (train K = 10, held-out K = 7
        self.d_obs = 2 * 46 + 2 + 11 * (self.K_obs + 1)  # padded with empty slots) + one table slot
        if JOINTS:  # + per arm sin q, cos q, last dq_RL, dq_prior
            self.d_obs += JI.D_EXTRA
        self.net = LowNet(self.d_obs).to(dev)
        if JOINTS:  # 2 x (7 joint residuals + speed)
            self.net = JI.joints_net(self.d_obs).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.sample = True
        self.learn_inline = False; self.log_fn = None  # set by the training loop in cleantables.rl.train
        self.begin()

    # ------------------------------------------------------------------------------------------------------------------ rollout
    def begin(self, sample=True):
        self.sample = sample
        self.buf = dict(obs=[], act=[], logp=[], val=[], valid=[], rew=[], done=[])
        self.prev_d = None; self.prev_m = None; self.prev_pos = None; self.prev_cnt = None; self.n_tick = 0
        self.prev_xy = None; self.prev_held = None; self.excuse = torch.zeros(self.N, self.K, dtype=torch.long, device=self.dev)
        self.noise = None  # AR(1) exploration state
        self.stats = dict(prog=0.0, near=0.0, yld=0.0, abort=0.0, contact=0.0, push=0.0, hh=0.0, touch=0.0, term=0.0, n_end=0,
                          off_cm=0.0, yaw_deg=0.0, spd=0.0, slow=0.0, n_act=0, mu_off_cm=0.0, mu_yaw_deg=0.0, mu_spd=0.0, mu_slow=0.0, n_samp=0, n_upd=0,
                          apr_n=0, apr_z_cm=0.0, apr_muz_cm=0.0, apr_dn3=0.0, car_n=0, car_z_cm=0.0, car_muz_cm=0.0, car_dn3=0.0)

    def _movers(self, lost):
        J = self.J; K = self.K
        m = np.asarray(J.mov, bool)[:, :K] & ~np.asarray(J.static, bool)[:, :K]
        if lost is not None:
            m = m & ~np.asarray(lost, bool)[:, :K]
        return m

    def _counters(self):
        """Per-env cumulative counters (copies): closeness, shield yields, aborted takes (all kinds), arm-object steps, hand-object
        overlap steps, finger-in-table steps."""
        CS = self.CS; A = CS.arms; z = torch.zeros(self.N, device=self.dev)
        near = A.n_near.clone().float() if hasattr(A, "n_near") else z.clone()
        yld = A.yield_g.sum(1).clone() if hasattr(A, "yield_g") else z.clone()
        ab = torch.as_tensor(np.asarray(CS.n_ab_env), dtype=torch.float32, device=self.dev)
        aobj = A.n_obj.clone().float()
        hov = CS.hov_ep.clone().float() if hasattr(CS, "hov_ep") else z.clone()
        ft = CS.ft_ep.clone().float()
        hh = CS.hh_steps.clone().float()                                                                 # the two grippers overlapping (steps)
        touch = A.n_touch.clone().float()                                                                # the two arms overlapping (steps)
        atab = A.n_tab.clone().float()  # arm links in the table (steps)
        return near, yld, ab, aobj, hov, ft, hh, touch, atab

    def _held(self):
        CS = self.CS; N, K = self.N, self.K; ar = torch.arange(N, device=self.dev)
        h = torch.zeros(N, K, dtype=torch.bool, device=self.dev)
        for g in (0, 1):
            hm = CS.phase[:, g] == 1
            h[ar[hm], CS.car[hm, g].clamp(0, K - 1)] = True
        return h

    def _sample_path(self, path, sl, u, frac):
        """points of paths (n, M, 3) with arc lengths sl (n, M) at progress fractions min(u + frac, 1) -> (n, F, 3)."""
        uq = (u[:, None] + frac[None]).clamp(max=1.0)
        s_ = uq * sl[:, -1:]
        j = (torch.searchsorted(sl.contiguous(), s_.contiguous(), right=True) - 1).clamp(0, sl.shape[1] - 2)
        a0, a1 = sl.gather(1, j), sl.gather(1, j + 1)
        fr = ((s_ - a0) / (a1 - a0).clamp(min=1e-12)).clamp(0, 1)[..., None]
        p0 = path.gather(1, j[..., None].expand(*j.shape, 3)); p1 = path.gather(1, (j + 1)[..., None].expand(*j.shape, 3))
        return p0 + fr * (p1 - p0)

    @torch.no_grad()
    def _obs(self, P, tleft):
        """(N, d_obs) privileged state, table frame."""
        CS = self.CS; fp = CS.fp; dev = self.dev; N, K = self.N, self.K; A = CS.arms; Kin = A.K
        org = fp.org[:, None]
        hp = CS.hP - org                                                                                     # (N, 2, 3)
        vel = torch.zeros_like(hp) if self.prev_pos is None else (hp - self.prev_pos) / TICK_S
        self.prev_pos = hp.clone()
        ph = CS.phase.clamp(0, 5)
        oh = torch.nn.functional.one_hot(ph, 6).float()                                                     # (N, 2, 6)
        dt = CS.dt
        u4 = (CS.t.float() * dt / CS.mdur.clamp(min=1e-6)).clamp(0, 1); u1 = (CS.t.float() * dt / CS.dur.clamp(min=1e-6)).clamp(0, 1)
        u = torch.where(CS.phase == 4, u4, torch.where(CS.phase == 1, u1, torch.zeros_like(u4)))
        hold = (CS.phase == 1).float()
        ar = torch.arange(N, device=dev)
        car = CS.car.clamp(0, K - 1)
        cext = torch.stack([fp.ext[ar, car[:, g]] for g in (0, 1)], 1) * hold[..., None]                 # (N, 2, 3)
        frac = torch.tensor([0.1, 0.25, 0.5, 0.75, 1.0], device=dev)
        plan = torch.zeros(N, 2, 5, 3, device=dev)
        for g in (0, 1):
            m4 = CS.phase[:, g] == 4; m1 = CS.phase[:, g] == 1
            if bool(m4.any()):
                e_ = torch.nonzero(m4).flatten()
                plan[e_, g] = self._sample_path(CS.mpath[e_, g], CS.msl[e_, g], u[e_, g], frac) - org[e_] - hp[e_, g][:, None]
            if bool(m1.any()):
                e_ = torch.nonzero(m1).flatten()
                plan[e_, g] = self._sample_path(CS.path[e_, g], CS.sl[e_, g], u[e_, g], frac) - hp[e_, g][:, None]
        blk = A.blocked_n.float() if hasattr(A, "blocked_n") else torch.zeros(N, 2, device=dev)
        Fk = Kin.fk(A.q.reshape(-1, 7)).reshape(N, 2, -1, 4, 4)
        Wk = A.base[:, :, None] @ Fk                                                                         # (N, 2, 13, 4, 4) table frame
        elb = Wk[:, :, Kin.F_IDX["link4"], :3, 3]; wri = Wk[:, :, Kin.F_IDX["link6"], :3, 3]
        arm_ = torch.cat([hp, vel, torch.sin(CS.hY)[..., None], torch.cos(CS.hY)[..., None], oh, u[..., None], hold[..., None], cext,
                          plan.reshape(N, 2, 15), CS.lo_off, CS.lo_yaw[..., None], CS.lo_spd[..., None], blk[..., None], elb, wri], -1)   # (N, 2, 46)
        if JOINTS:  # the realised hand offset / turn
            arm_ = JI.arm_feats(arm_, CS)
        gap = A._clear(A.q)[:, None].clamp(-0.2, 0.5)
        tl = tleft[:, None]
        Pt = P[..., :3].clone(); Pt = torch.nan_to_num(Pt)
        ext = torch.nan_to_num(fp.ext)                                                                   # (unused slots hold NaN extents)
        end0 = torch.nan_to_num(torch.as_tensor(np.asarray(self.J.end0, np.float32)[:, :K], device=dev))
        top = Pt.clone(); top[..., 2] += ext[..., 2]
        mv = torch.as_tensor(self._movers(None), device=dev)
        goff = torch.where(mv[..., None], end0 - top, torch.zeros_like(top))
        held = torch.zeros(N, K, device=dev)
        for g in (0, 1):
            hm = CS.phase[:, g] == 1
            held[ar[hm], car[hm, g]] = 1.0
        val = fp.valid.float()
        obj = torch.cat([Pt * val[..., None], ext * val[..., None], goff, held[..., None], val[..., None]], -1).reshape(N, -1)   # (N, 11 K)
        bx = A.box  # the table: one more object-like slot
        tab = torch.stack([(bx[:, 0] + bx[:, 1]) / 2, (bx[:, 2] + bx[:, 3]) / 2, torch.full_like(bx[:, 0], -0.02),
                           (bx[:, 1] - bx[:, 0]) / 2, (bx[:, 3] - bx[:, 2]) / 2, torch.full_like(bx[:, 0], 0.02)], -1)   # centre (top at z 0), half-extents
        if self.K_obs > K:  # empty slots: all zeros (valid 0)
            obj = torch.cat([obj, torch.zeros(N, 11 * (self.K_obs - K), device=dev)], -1)
        obj = torch.cat([obj, tab, torch.zeros(N, 4, device=dev), torch.ones(N, 1, device=dev)], -1)        # goal offset 0, held 0, valid 1
        o = torch.cat([arm_.reshape(N, -1), gap, tl, obj], -1)
        if JOINTS:  # + 56 joint features
            o = torch.cat([o, JI.obs_extra(CS)], -1)
        return torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)  # no NaN reaches the network

    @torch.no_grad()
    def tick(self, CS, judge, sc, lost, ended_eps, retired, ep_start):
        """end of a decision tick: score the last interval, then act for the next one."""
        dev = self.dev; N, K = self.N, self.K
        m = torch.as_tensor(self._movers(lost), device=dev)
        d_obj = torch.nan_to_num(torch.as_tensor(np.asarray(sc["dist_raw_cm"], np.float32)[:, :K], device=dev))
        cnt = self._counters()
        ended = torch.zeros(N, dtype=torch.bool, device=dev); term = torch.zeros(N, device=dev)
        w_un = float(os.environ.get("FPR_W_UNMET", "0.5"))
        for ep in ended_eps:
            e = int(ep["env"]); ended[e] = True
            tr = W_MET_L * float(ep.get("n_met3", 0)) + 0.5 * W_MET_L * float(ep.get("all_met3", ep.get("all_met", False))) - 2.0 * float(ep.get("n_lost_robot", 0))
            if ep.get("stop") in ("stop", "lost_robot"):
                tr -= w_un * float(ep.get("n_unmet", 0))
            term[e] = tr
        P = CS.poses()
        xy = torch.nan_to_num(P[..., :2].float()); held = self._held()
        if self.prev_held is not None:                                                                   # an object leaving a hand settles 1 s
            self.excuse[self.prev_held & ~held] = int(round(1.0 / TICK_S))
        push = torch.zeros(N, device=dev)
        if self.prev_xy is not None:
            dist_step = getattr(CS, "dist_step", None)
            calm = torch.ones(N, dtype=torch.bool, device=dev) if dist_step is None else \
                torch.as_tensor((CS.step_i - np.asarray(dist_step)) * CS.dt > 3.0, device=dev)
            calm = calm & ~(CS.phase == 5).any(1) & ~ended
            lost_ = torch.as_tensor(np.asarray(lost, bool)[:, :K], device=dev) if lost is not None else torch.zeros_like(held)
            okk = CS.fp.valid[:, :K].bool() & ~held & ~self.prev_held & (self.excuse == 0) & ~lost_ & calm[:, None]
            if PUSH_CONTACT and getattr(CS.arms, "obj_touch_age", None) is not None:  # only motion the robot caused
                okk = okk & (CS.arms.obj_touch_age[:, :K] <= int(round(PUSH_RECENT_S / CS.dt)))
            mv_cm = (xy - self.prev_xy).norm(dim=-1) * 100.0
            push = ((mv_cm - PUSH_MIN_CM).clamp(min=0) * okk).sum(1)
        self.excuse = (self.excuse - 1).clamp(min=0)
        self.prev_xy = xy; self.prev_held = held
        if self.prev_d is not None:                                                                      # the transition that ends now
            dd = [(c - p_).clamp(min=0) for c, p_ in zip(cnt, self.prev_cnt)]                          # (an ended env's counters were reset)
            d_near, d_yld, d_ab, d_aobj, d_hov, d_ft, d_hh, d_touch, d_atab = dd
            both = m & self.prev_m  # objects tracked at both ticks
            prog = torch.where(ended, torch.zeros(N, device=dev), ((self.prev_d - d_obj) * both).sum(1))
            dt = CS.dt
            contact = W_ARMOBJ * d_aobj * dt + W_HOV * d_hov * dt + W_FT * d_ft * dt + W_ARMTAB * d_atab * dt
            c_push, c_hh, c_touch = W_PUSH * push, W_HH * d_hh * dt, W_TOUCH * d_touch * dt
            r = W_PROG * prog - W_NEAR * d_near * dt - W_YIELD * d_yld - W_ABORT * d_ab - contact - c_push - c_hh - c_touch - W_TIME * TICK_S + term
            if W_DEV > 0 and not JOINTS:  # deviation from the WM path
                st_ = ((CS.phase == 1) | (CS.phase == 4)).float()                                         # the offset applied in the interval
                c_dev = W_DEV * (CS.lo_off.norm(dim=-1) * st_).sum(1) * TICK_S; r = r - c_dev
                self.stats["dev"] = self.stats.get("dev", 0.0) + float((c_dev * self.buf["valid"][-1].float()).sum())
            if JOINTS:  # deviation / tilt / limit costs of the joint-space interface
                r = JI.add_costs(self, CS, r)
            self.buf["rew"].append(r); self.buf["done"].append(ended.clone())
            v_ = self.buf["valid"][-1].float()
            self.stats["prog"] += float((W_PROG * prog * v_).sum()); self.stats["near"] += float((W_NEAR * d_near * dt * v_).sum())
            self.stats["yld"] += float((W_YIELD * d_yld * v_).sum()); self.stats["abort"] += float((W_ABORT * d_ab * v_).sum())
            self.stats["contact"] += float((contact * v_).sum()); self.stats["term"] += float(term.sum()); self.stats["n_end"] += int(ended.sum())
            self.stats["push"] += float((c_push * v_).sum()); self.stats["hh"] += float((c_hh * v_).sum()); self.stats["touch"] += float((c_touch * v_).sum())
        # the next interval starts (an ended env: a fresh episode at its start distances)
        d0 = torch.nan_to_num(torch.as_tensor(np.asarray(judge.d0_cm, np.float32)[:, :K], device=dev))
        m0 = torch.as_tensor(self._movers(None), device=dev)
        self.prev_d = torch.where(ended[:, None], d0, d_obj); self.prev_m = torch.where(ended[:, None], m0, m)
        self.prev_cnt = self._counters()
        if self.prev_pos is not None:
            self.prev_pos[ended] = (CS.hP - CS.fp.org[:, None])[ended]
        tleft = torch.as_tensor(1.0 - (CS.step_i - np.asarray(ep_start, np.float64)) * CS.dt / self.ep_s, dtype=torch.float32, device=dev).clamp(0, 1)
        o = self._obs(P, tleft).half().float()  # stored fp16 obs equals the obs acted on
        mu, v = self.net(o)
        ls = self.net.log_std.clamp(-4.0, 0.5)
        if self.sample and NOISE_TAU > 0:  # smooth exploration noise
            rho = math.exp(-TICK_S / NOISE_TAU)
            xi = torch.randn_like(mu)
            if self.noise is None:
                self.noise = xi
            else:
                self.noise = rho * self.noise + math.sqrt(1.0 - rho * rho) * xi
                self.noise[ended] = xi[ended]                                                            # a fresh episode: a fresh draw
            a = mu + ls.exp() * self.noise
        else:
            a = mu + ls.exp() * torch.randn_like(mu) if self.sample else mu
        lp = gauss_logp(a, mu, ls)
        valid = ~torch.as_tensor(np.asarray(retired), device=dev)
        self.buf["obs"].append(o.half()); self.buf["act"].append(a); self.buf["logp"].append(lp); self.buf["val"].append(v); self.buf["valid"].append(valid)
        off, yaw, spd = squash(a) if not JOINTS else JI.act_view(a)
        if ZERO_OFF:  # evaluation only: no offsets; turn / speed as learned
            off = torch.zeros_like(off)
        act_ = ((CS.phase == 1) | (CS.phase == 4)) & valid[:, None]  # logging: arms the controller steers now
        if bool(act_.any()):
            self.stats["off_cm"] += float(off.norm(dim=-1)[act_].sum()) * 100; self.stats["yaw_deg"] += float(yaw.abs()[act_].sum()) * 180 / math.pi
            self.stats["spd"] += float(spd[act_].sum()); self.stats["slow"] += float((spd[act_] < 0.5).sum()); self.stats["n_act"] += int(act_.sum())
            off_m, yaw_m, spd_m = squash(mu) if not JOINTS else JI.act_view(mu)  # logging only: the noise-free action on the same states
            self.stats["mu_off_cm"] += float(off_m.norm(dim=-1)[act_].sum()) * 100; self.stats["mu_yaw_deg"] += float(yaw_m.abs()[act_].sum()) * 180 / math.pi
            self.stats["mu_spd"] += float(spd_m[act_].sum()); self.stats["mu_slow"] += float((spd_m[act_] < 0.5).sum())
            apr_ = (CS.phase == 4) & (CS.after == 1) & valid[:, None]  # logging only: vertical (table z) part of the offset
            if getattr(CS, "arms", None) is not None:  # on approaches and carries: signed means and share below -3 cm
                apr_ = apr_ & ~CS.arms.retract
            for nm_, m_ in (("apr", apr_), ("car", (CS.phase == 1) & valid[:, None])):
                if bool(m_.any()):
                    self.stats[f"{nm_}_n"] += int(m_.sum()); self.stats[f"{nm_}_z_cm"] += float(off[..., 2][m_].sum()) * 100
                    self.stats[f"{nm_}_muz_cm"] += float(off_m[..., 2][m_].sum()) * 100; self.stats[f"{nm_}_dn3"] += float((off[..., 2][m_] < -0.03).sum())
        CS.lo_off[:] = torch.nan_to_num(off); CS.lo_yaw[:] = torch.nan_to_num(yaw); CS.lo_spd[:] = torch.nan_to_num(spd, nan=1.0)  # guard against NaN
        if JOINTS:  # joint residuals -> hand displacement
            JI.apply_action(CS, a)
        self.n_tick += 1
        if self.sample and self.learn_inline and UPDATE_TICKS > 0 and len(self.buf["rew"]) >= UPDATE_TICKS:  # learn every UPDATE_TICKS
            with torch.enable_grad():
                st = self._ppo(len(self.buf["rew"]))
            if st and self.log_fn is not None:
                self.log_fn(f"LOWPOL inline update {self.stats['n_upd']} at tick {self.n_tick}: " + ", ".join(f"{k} {v:.4g}" for k, v in st.items()))

    # ------------------------------------------------------------------------------------------------------------------ learning
    def _ppo(self, T):
        """PPO on the first T stored steps (each has its reward), bootstrapped from the value of step T; then those T steps leave the buffer
        (step T, still waiting for its reward, stays). -> the last minibatch's stats (+ samples, ticks, ret_mean)."""
        B = self.buf
        if T < 2:
            return {}
        obs = torch.stack(B["obs"][:T]); act = torch.stack(B["act"][:T]); lp0 = torch.stack(B["logp"][:T])
        val = torch.stack(B["val"][:T + 1]); valid = torch.stack(B["valid"][:T]).float(); rew = torch.stack(B["rew"]); done = torch.stack(B["done"]).float()
        adv = torch.zeros_like(rew); gae = torch.zeros(self.N, device=self.dev)
        for t in reversed(range(T)):
            nv = val[t + 1] * (1 - done[t])
            delta = rew[t] + GAMMA * nv - val[t]
            gae = delta + GAMMA * LAM * (1 - done[t]) * gae
            adv[t] = gae
        ret = adv + val[:T]
        m = valid.reshape(-1) > 0
        O, Aa, L0, Ad, R = obs.reshape(-1, obs.shape[-1])[m], act.reshape(-1, act.shape[-1])[m], lp0.reshape(-1)[m], adv.reshape(-1)[m], ret.reshape(-1)[m]
        Ad = (Ad - Ad.mean()) / (Ad.std() + 1e-8) if len(Ad) > 1 else Ad
        n = len(O); st = {}
        if n < 2:
            for k_ in B:  # nothing to learn from: the steps are still consumed
                B[k_] = B[k_][T:]
            return {}
        stop_ = False; self.kl_stops = getattr(self, "kl_stops", 0)
        for ep in range(EPOCHS):
            if stop_:
                break
            perm = torch.randperm(n, device=self.dev); kls = []
            for i in range(0, n, MB):
                ix = perm[i:i + MB]
                mu, v = self.net(O[ix].float()); ls = self.net.log_std.clamp(-4.0, 0.5)
                lp = gauss_logp(Aa[ix], mu, ls); ratio = (lp - L0[ix]).exp()
                pg = -torch.minimum(ratio * Ad[ix], ratio.clamp(1 - CLIP, 1 + CLIP) * Ad[ix]).mean()
                vl = ((v - R[ix]) ** 2).mean(); ent = (ls + 0.5 * math.log(2 * math.pi * math.e)).sum()
                loss = pg + VF_W * vl - ENT_W * ent
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 0.5); self.opt.step()
                with torch.no_grad():
                    lr_ = lp.detach() - L0[ix]; kls.append(float(((lr_.exp() - 1) - lr_).mean()))  # logging: approx KL old -> new
                st = dict(pg=float(pg), vl=float(vl), ratio=float(ratio.mean()), clipfrac=float(((ratio - 1).abs() > CLIP).float().mean()),
                          approx_kl=float(np.mean(kls)))
                if KL_STOP > 0 and st["approx_kl"] > KL_STOP:  # early stop
                    stop_ = True; self.kl_stops += 1; st["kl_stop_epoch"] = ep; break
        for k_ in B:  # the used steps leave the buffer
            B[k_] = B[k_][T:]
        self.stats["n_samp"] += int(n); self.stats["n_upd"] += 1
        return dict(st, samples=int(n), ticks=T, ret_mean=float(R.mean()))

    def update(self, log=print):
        """PPO on the iteration's remaining ticks (all of them when there are no inline updates). The last stored step has no reward yet: dropped."""
        st = self._ppo(len(self.buf["rew"]))
        if not st and self.stats["n_upd"] == 0:
            return {}
        n = self.stats["n_samp"]
        s = dict(st, samples_iter=int(n), updates_iter=self.stats["n_upd"], log_std=float(self.net.log_std.mean()),
                 **{f"r_{k}": v / max(n, 1) for k, v in self.stats.items() if k in ("prog", "near", "yld", "abort", "contact", "push", "hh", "touch", "term", "dev")},
                 episodes_ended=self.stats["n_end"], **{f"act_{k}": self.stats[k] / max(self.stats["n_act"], 1) for k in ("off_cm", "yaw_deg", "spd", "slow", "mu_off_cm", "mu_yaw_deg", "mu_spd", "mu_slow")})
        s.update({f"{nm_}_{k_}": self.stats[f"{nm_}_{k_}"] / max(self.stats[f"{nm_}_n"], 1) for nm_ in ("apr", "car") for k_ in ("z_cm", "muz_cm", "dn3")})
        log(f"LOWPOL update: {s}")
        return s

    def state_dict(self):
        return dict(net=self.net.state_dict(), opt=self.opt.state_dict(), r_off=R_OFF)  # the offset cap the weights were trained with

    def load_state_dict(self, sd):
        net_sd = dict(sd["net"]); grown = []
        for k, v in self.net.state_dict().items():  # new obs inputs (e.g. the table slot):
            if k in net_sd and net_sd[k].shape != v.shape:  # old weights are kept, new input
                old = net_sd[k]; new = v.clone()  # columns start at zero (LayerNorm: gain 1, bias 0)
                if old.dim() == 1:
                    new[:old.shape[0]] = old; new[old.shape[0]:] = 1.0 if k.endswith("0.weight") else 0.0
                else:
                    new.zero_(); new[:, :old.shape[1]] = old
                net_sd[k] = new; grown.append(k)
        self.net.load_state_dict(net_sd); self.opt.load_state_dict(sd["opt"])
        if grown:
            for p_ in self.net.parameters():                                                              # Adam moments of a grown tensor restart
                st_ = self.opt.state.get(p_)
                if st_ and "exp_avg" in st_ and st_["exp_avg"].shape != p_.shape:
                    del self.opt.state[p_]
            print(f"LOWPOL: obs grew to {self.d_obs} (new inputs zero-weighted): {grown}", flush=True)
        k_ = R_OFF / float(sd.get("r_off", 0.25))  # the checkpoint's offset cap (0.25 if absent)
        if abs(k_ - 1.0) > 1e-6 and not JOINTS:  # cap changed: rescale once
            idx = [0, 1, 2, 5, 6, 7]  # keep the learned offsets and the noise in metres:
            with torch.no_grad():  # offset rows of the output layer / k, their
                self.net.pi[-1].weight[idx] /= k_; self.net.pi[-1].bias[idx] /= k_  # log-sd - ln k (tanh is ~linear there)
                self.net.log_std[idx] -= math.log(k_)
            print(f"LOWPOL: bend cap now {R_OFF * 100:.0f} cm; offset outputs / {k_:.3f}, offset log-sd now {[round(float(x), 3) for x in self.net.log_std[idx]]}", flush=True)
        if LS_RESET and JOINTS:  # joint-space log-sd layout
            JI.ls_reset(self)
        if LS_RESET and not JOINTS:  # reset exploration noise
            with torch.no_grad():
                self.net.log_std.copy_(torch.tensor(([LS0_OFF] * 4 + [LS0_SPD]) * 2, device=self.net.log_std.device))
            print(f"LOWPOL: log_std reset to {self.net.log_std.tolist()} (noise tau {NOISE_TAU} s)", flush=True)
        if LS_SET:  # exact entries, once
            with torch.no_grad():
                for kv_ in LS_SET.split(","):
                    i_, v_ = kv_.split(":"); self.net.log_std[int(i_)] = float(v_)
            print(f"LOWPOL: log_std entries set ({LS_SET}) -> {self.net.log_std.tolist()}", flush=True)
