"""Joint-space interface of the low-level controller (FPL_ACTION=joints).

The controller (`cleantables.rl.controller`) outputs, per arm, 7 joint residuals (tanh x J_MAX) and a speed instead of a path offset;
its network is LowNet(271, 16) with a null-space avoidance prior, the layout of the separate joint-space training gym. `chain_env` and
`controller` call these functions from `if JOINTS:` blocks; with the flag unset they are not used. Requires FPC_ARM_EVERY = FPC_TICK =
12 (the arm is posed at every controller tick; forced, with a log line).

Per tick, ArmTracker.check (right before LowPol.tick):
  the planned hand of every arm (planned_hands: the move's path point / the carry's planned object root + its grasp offset, bounce shove
  included, at the hand's current clock; idle / settling / pre hands: where they are) -> IK of its TCP, top-down at the planned yaw,
  warm-started from the last plain IK pose -> the null-space prior (prior_update, from the arms' current q) -> q = clamp(q_IK + env(u)
  (dq_prior + dq_RL), Franka limits), env = the flat bend_env (waits: the ramp; retracts: 0) -> shield, bounce, clearances and counters
  on this q (ArmTracker.q; the elbow / wrist / gap obs read it).
  Then jt_after_check computes the realised offset / turn FK(q) - FK(q_IK) (obs slots lo_off / lo_yaw, the W_DEV cost), the hand's tilt
  (carry cost) and the hand drive: FK(q_IK + dq) vs FK(q_IK) as palm offset lo_off, yaw turn lo_yaw and roll / pitch lo_rp (gripper
  frame: x = finger axis, z up; orientation = qyaw(Y) q_tilt(roll, pitch), the ROT_EDIT convention of chain_env). chain_env's per-step
  hand target applies these scaled by env(u), so mid-path (env 1) the gripper is where FK(q) puts it (up to the IK residual) and at the
  path ends exactly on the plan, top-down. During carries the held object's target moves by env lo_off / lo_yaw and turns with the
  hand's roll / pitch about the grasp point (carry_rotate); the hand riding it takes the commanded tilt (carry_hand). The end zones are
  chain_env's FPC_BEND_START_M / FPC_BEND_END_M.
LowPol obs = the 215 offset-mode features with lo_off / lo_yaw replaced by the realised offset / turn (arm_feats), plus per arm sin q,
cos q, the last dq_RL and dq_prior (obs_extra). Rewards = the controller's plus deviation (W_DEV on the realised offset), tilt
(FPL_W_TILT) and joint-limit (FPL_W_LIM) costs.

Approximations relative to the gym: between ticks the displacement / turn / tilt found at the tick is scaled by env(u) along the path
(exact mid-path and at the ends, linearised in the ramps); the carried object's tilt is commanded and physics follows; a waiting hand
keeps chain_env's palm floor (FLUID_WAIT); carried objects lag their target, so the arm's q at a check is that of the planned hand.
FPC_ARM_FAST is not supported (warned).

Typical flags: FPL_ACTION=joints FPC_ARM_EVERY=12 FPC_FLAT_BEND=1 FPC_BEND_START_M=0.02 FPC_BEND_END_M=0.03 FPL_W_DEV=0.3.
"""
from __future__ import annotations

import math
import os

import torch

from cleantables.robots import franka_kinematics as KIN
from cleantables.rl import chain_env as FC                                         # circular with chain_env: attributes used at call time only

J_MAX = float(os.environ.get("FPL_J_MAX", "0.6"))                    # rad, residual cap per joint
W_TILT = float(os.environ.get("FPL_W_TILT", "0.1"))                  # per rad of hand tilt from top-down per s while carrying
W_LIM = float(os.environ.get("FPL_W_LIM", "0.1"))                    # per s per joint, 0 -> 1 over the outer 10 % of its range
PRIOR = os.environ.get("FPL_PRIOR", "nullspace")                     # redundancy-resolution avoidance prior ("none" = off)
PRIOR_D0 = float(os.environ.get("FPL_PRIOR_D0", "0.15"))
PRIOR_K = float(os.environ.get("FPL_PRIOR_K", "0.3"))
PRIOR_QD = float(os.environ.get("FPL_PRIOR_QD", "1.0"))
PRIOR_TAU = float(os.environ.get("FPL_PRIOR_TAU", "0.5"))
SOLID_TABLE = os.environ.get("FPL_SOLID_TABLE", "0") == "1"           # the residual never drives the fingertips below the table
JVEL = os.environ.get("FPL_JVEL", "0") == "1"                         # realised joint motion per tick <= the Franka's v_max
JACC = os.environ.get("FPL_JACC", "0") == "1"                         # with FPL_JVEL: the step's change <= a_max dt^2
JPEN = float(os.environ.get("FPL_W_ACTION_RATE") or 0) > 0 or float(os.environ.get("FPL_W_JOINT_VEL") or 0) > 0   # Isaac Lab smoothness costs
ACT_LPF = float(os.environ.get("FPL_ACT_LPF") or 0)                  # dq = a dq_prev + (1 - a) dq_new (0 = off)
LPF = ACT_LPF > 0
GATE_CLOCK = float(os.environ.get("FPL_GATE_CLOCK") or 0)            # m; a move's / carry's plan clock advances only
GATE = GATE_CLOCK > 0                                                # while the hand (FK of q) is within GATE_CLOCK of its planned point (0 = off)
GATE_BUDGET = float(os.environ.get("FPL_GATE_BUDGET") or 6.0)        # after this x the planned time (wall) the gate opens: the move ends on its clock
REACH_DONE = float(os.environ.get("FPL_REACH_DONE") or 0)            # m; a take's grasp / a carry's
REACH = REACH_DONE > 0                                               # planned release fires only with the hand within REACH_DONE of the point (0 = off)
UNPIN = os.environ.get("FPL_UNPIN", "0") == "1"                       # the residual acts in every phase (unpin_*)
LS_JOINT = os.environ.get("FPL_LS_JOINT")                            # initial log-sd of the joint residuals (default FPL_LS0)
J_PER_ARM = 8
D_EXTRA = 2 * 28


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _interp(path, sl, s):
    M = sl.shape[1]
    j = (torch.searchsorted(sl.contiguous(), s[:, None].contiguous(), right=True)[:, 0] - 1).clamp(0, M - 2)
    a0, a1 = sl.gather(1, j[:, None])[:, 0], sl.gather(1, (j + 1)[:, None])[:, 0]
    fr = ((s - a0) / (a1 - a0).clamp(min=1e-12)).clamp(0, 1)
    ar = torch.arange(len(s), device=s.device)
    return path[ar, j] + fr[:, None] * (path[ar, j + 1] - path[ar, j])


# ----------------------------------------------------------------------------------------------------------------------------- the net
def squash_joints(a, j_max=None):
    """raw actions (..., 16) -> joint residuals (..., 2, 7) rad (tanh x J_MAX), speed (..., 2) (the rule of controller.squash)."""
    from cleantables.rl import controller as LP
    a = a.reshape(*a.shape[:-1], 2, J_PER_ARM)
    spd = torch.round(torch.sigmoid(a[..., 7] + LP.SPD_B0) * LP.SPD_MAX * 1e4) / 1e4
    return torch.tanh(a[..., :7]) * (J_MAX if j_max is None else j_max), spd


def act_view(a):
    """Zero offset, zero turn and the speed of joint-space actions a (the controller's logging / offset fields)."""
    _, spd = squash_joints(a)
    return torch.zeros(*spd.shape, 3, device=a.device), torch.zeros_like(spd), spd


def joints_net(d_obs):
    """controller.LowNet(d_obs, 16) with the joint layout's log-sd (7 x FPL_LS_JOINT / FPL_LS0, FPL_LS0_SPD per arm)."""
    from cleantables.rl import controller as LP
    net = LP.LowNet(d_obs, d_act=2 * J_PER_ARM)
    ls = LP.LS0_OFF if LS_JOINT is None else float(LS_JOINT)
    net.log_std = torch.nn.Parameter(torch.tensor(([ls] * 7 + [LP.LS0_SPD]) * 2))
    return net


def ls_reset(lp):
    from cleantables.rl import controller as LP
    with torch.no_grad():
        ls = LP.LS0_OFF if LS_JOINT is None else float(LS_JOINT)
        lp.net.log_std.copy_(torch.tensor(([ls] * 7 + [LP.LS0_SPD]) * 2, device=lp.net.log_std.device))
    print(f"LOWPOL (joints): log_std reset to {lp.net.log_std.tolist()}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------- the tracker
def tracker_init(A):
    CS = A.CS; N = A.q.shape[0]; dev = A.q.device
    z = lambda *s: torch.zeros(*s, device=dev)                       # noqa: E731
    A.q_ik = A.q.clone(); A.dq_prior = z(N, 2, 7)
    A.jP = z(N, 2, 3); A.jY = z(N, 2); A.jW = z(N, 2)
    A.lo_real = z(N, 2, 3); A.lo_yaw_real = z(N, 2); A.tilt = z(N, 2); A.jpal_ik = z(N, 2, 3); A.jyaw_ik = z(N, 2)
    A.qlo = torch.tensor(KIN.Q_LO, device=dev); A.qhi = torch.tensor(KIN.Q_HI, device=dev)
    A.jst = dict(arm_ticks=0.0, active=0.0, dq_abs=0.0)
    if JACC and not JVEL:
        print("JOINTS WARNING: FPL_JACC applies inside the joint-speed limit: set FPL_JVEL=1 too (ignored without it)", flush=True)
    if A.every != FC.TICK:
        print(f"JOINTS: the arm is posed at every controller tick: ArmTracker.every {A.every} -> {FC.TICK} (set FPC_ARM_EVERY={FC.TICK})", flush=True)
        A.every = FC.TICK
    if FC.ARM_FAST:
        print("JOINTS WARNING: FPC_ARM_FAST is not supported with FPL_ACTION=joints (its IK cache ignores the planned targets)", flush=True)
    if not hasattr(CS, "lo_dq"):
        CS.lo_dq = z(N, 2, 7)
    if not hasattr(CS, "lo_rp"):
        CS.lo_rp = z(N, 2, 2)
    if JACC or JPEN:                                                  # the last realised joint step, the pose before a check, the last action
        A.jdq_last = z(N, 2, 7); A.jq_before = A.q.clone(); A.jact_prev = z(N, 2, 7)
    A.jYd_ik = z(N, 2); A.jrp_ik = z(N, 2, 2)
    if GATE:                                                          # gate open until the first check; wall steps of the current move / carry
        A.jgate = torch.ones(N, 2, dtype=torch.bool, device=dev); CS.jtw = z(N, 2)
        print(f"JOINTS: pure-pursuit gate on the plan clock: {GATE_CLOCK} m (the gym's --gate_clock), budget {GATE_BUDGET} x the planned time", flush=True)
    if REACH:                                                         # hand (FK of q) at the last check; move / carry wall steps
        A.jpal_w = _fk_palm(A.q, A.base)[0] + CS.fp.org[:, None]; CS.jdrop = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        if getattr(CS, "jtw", None) is None:
            CS.jtw = z(N, 2)
        print(f"JOINTS: reach done: a take's grasp / a carry's release at its path end fire with the hand within {REACH_DONE} m of the point "
              f"(fallback after {GATE_BUDGET} x the planned time)", flush=True)
    if LPF:
        print(f"JOINTS: low-pass on the commanded joint offsets: dq = {ACT_LPF} dq_prev + {1 - ACT_LPF:.3f} dq_new (the gym's --act_lpf)", flush=True)
    print(f"JOINTS: joint-space controller (J_MAX {J_MAX} rad, prior {PRIOR}: d0 {PRIOR_D0} m, k {PRIOR_K} m/s, qd {PRIOR_QD} rad/s, "
          f"tau {PRIOR_TAU} s; W_TILT {W_TILT}, W_LIM {W_LIM})", flush=True)


def tracker_reset(A, e):
    A.q_ik[e] = A.q0; A.dq_prior[e] = 0.0; A.lo_real[e] = 0.0; A.lo_yaw_real[e] = 0.0; A.tilt[e] = 0.0
    A.jpal_ik[e] = 0.0; A.jyaw_ik[e] = 0.0; A.jW[e] = 0.0
    if hasattr(A.CS, "lo_dq"):
        A.CS.lo_dq[e] = 0.0
    if hasattr(A.CS, "lo_rp"):
        A.CS.lo_rp[e] = 0.0
    A.jYd_ik[e] = 0.0; A.jrp_ik[e] = 0.0
    if JACC or JPEN:
        A.jdq_last[e] = 0.0; A.jact_prev[e] = 0.0
    if GATE:
        A.jgate[e] = True; A.CS.jtw[e] = 0.0
    if REACH:
        A.CS.jtw[e] = 0.0; A.CS.jdrop[e] = False


def planned_hands(CS):
    """(N, 2) every hand's planned pose (world palm, yaw) and the residual's envelope weight, at its current clock: moves (phase 4) the
    path point (+ the bounce shove) and the path yaw (chain_env._hand_now without the controller's offset), weight bend_env (0 on retracts,
    the ramp on waits); carries (phase 1) the planned object root (+ the start offset's fade, + the shove) with the grasp's palm offset at
    the planned object yaw (chain_env.step's target, the hand riding the object), weight bend_env; other phases: the hand where it is, 0."""
    A = CS.arms; N = CS.N; dt = CS.dt; dev = CS.dev
    ph = CS.phase; t = CS.t.float(); org = CS.fp.org[:, None]
    MP = CS.mpath.shape[2]; NP = CS.path.shape[2]
    bp = CS.b_push if FC.PUSH_BOUNCE else torch.zeros(N, 2, 3, device=dev)
    ur = torch.clamp(t * dt / CS.mdur.clamp(min=1e-6), max=1.0); u = FC.minjerk(ur)
    Lm = CS.msl[..., -1]
    Pm = _interp(CS.mpath.reshape(-1, MP, 3), CS.msl.reshape(-1, MP), (u * Lm).reshape(-1)).reshape(N, 2, 3) + bp
    Ym = CS.mY0 + u * (CS.mY1 - CS.mY0)
    w = FC.bend_env(u, Lm)
    if FC.NO_HOME_BEND:
        w = w * (~A.retract).float()
    if FC.FLUID_WAIT:
        w = torch.where(CS.is_wait, u.clamp(0, 1), w)
    uc = FC.minjerk(torch.clamp(t * dt / CS.dur.clamp(min=1e-6), max=1.0)); Lc = CS.L
    X = _interp(CS.path.reshape(-1, NP, 3), CS.sl.reshape(-1, NP), (uc * Lc).reshape(-1)).reshape(N, 2, 3) + (1 - uc)[..., None] * CS.off + bp
    ar = torch.arange(N, device=dev); car = CS.car.clamp(0, CS.K - 1)
    yb = torch.stack([CS.fp.yb[ar, car[:, g]] for g in (0, 1)], 1)
    yo = CS.yaw0 + uc * CS.dyaw + yb
    HC0 = CS.HC[:, :, 0]; cz, sz = torch.cos(yo), torch.sin(yo)
    Pc = X + org + torch.stack([cz * HC0[..., 0] - sz * HC0[..., 1], sz * HC0[..., 0] + cz * HC0[..., 1], HC0[..., 2]], -1)
    Yc = yo + CS.gyaw
    wc = FC.bend_env(uc, Lc)
    P = torch.where((ph == 4)[..., None], Pm, torch.where((ph == 1)[..., None], Pc, CS.hP))
    Y = torch.where(ph == 4, Ym, torch.where(ph == 1, Yc, CS.hY))
    W = torch.where(ph == 4, w, torch.where(ph == 1, wc, torch.zeros_like(w)))
    if UNPIN:                                                        # every phase: see unpin_hands
        P, Y, W = unpin_planned(CS, P, Y, W)
    return P, Y, W


# ----------------------------------------------------------------------------------------------------------------------------- unpin
# FPL_UNPIN=1: the joint residual's envelope W is 1 in every phase and along the whole path, except the final grasp / place descent:
# a take's approach keeps the end ramp over the last FPC_BEND_END_M, a carry keeps its bend_env (lift-off from the grasp, place
# descent; the hand is exactly on the object at both ends). The PhysX hand follows: idle / done / pre hands sit at their plan pose +
# W lo_off, a settling hand's retreat gets it too, and a new move starts from the plan pose with the displacement carried over (no
# jump). Bookkeeping: CS.jdisp / jdispY = the displacement / turn applied to each hand at the last step (so the plan pose = hP - jdisp);
# CS.jw = the W applied; CS.jw0 = W when the current move / settle began, from which the envelope blends to 1 over the start zone
# (moves) / the 0.4 s retreat (settles): 1 -> 1 normally (an idle hand already displaced), 0 -> 1 after a grasp / place (or an
# approach aborted inside its grasp zone).
def _smooth(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def _unpin_state(CS):
    if getattr(CS, "jdisp", None) is None:
        N = CS.phase.shape[0]; dev = CS.phase.device
        CS.jdisp = torch.zeros(N, 2, 3, device=dev); CS.jdispY = torch.zeros(N, 2, device=dev); CS.jdisp0 = torch.zeros(N, 2, 3, device=dev)
        CS.jw = torch.ones(N, 2, device=dev); CS.jw0 = torch.ones(N, 2, device=dev); CS.jph = CS.phase.clone()


def unpin_w(CS, ee, gg, u, L):
    """moves (n,): W blends from jw0 to 1 over the first FPC_BEND_START_M; a take's approach (after 1) ramps to 0 over the last FPC_BEND_END_M."""
    _unpin_state(CS)
    w0 = CS.jw0[ee, gg]
    w = w0 + (1 - w0) * _smooth(u * L / FC.BEND_START_M)
    return torch.where(CS.after[ee, gg] == 1, w * _smooth((1 - u.clamp(0, 1)) * L / FC.BEND_END_M), w)


def _settle_w(CS):
    us = FC.minjerk(torch.clamp(CS.t.float() * CS.dt / 0.4, max=1.0))
    return CS.jw0 + (1 - CS.jw0) * us


def _move_plan(CS):
    """(N, 2) every hand's planned move point (+ the bounce shove) and yaw, its u and path length (chain_env._hand_now before the offset)."""
    N = CS.phase.shape[0]; MP = CS.mpath.shape[2]
    bp = CS.b_push if FC.PUSH_BOUNCE else torch.zeros(N, 2, 3, device=CS.phase.device)
    u = FC.minjerk(torch.clamp(CS.t.float() * CS.dt / CS.mdur.clamp(min=1e-6), max=1.0)); Lm = CS.msl[..., -1]
    Pm = _interp(CS.mpath.reshape(-1, MP, 3), CS.msl.reshape(-1, MP), (u * Lm).reshape(-1)).reshape(N, 2, 3) + bp
    return Pm, CS.mY0 + u * (CS.mY1 - CS.mY0), u, Lm


def unpin_start(CS, e, g):
    """chain_env._start_move: the new move starts from the hand's plan pose (hP - its displacement); its envelope blends in from the W now."""
    _unpin_state(CS)
    CS.jw0[e, g] = CS.jw[e, g]
    return (CS.hP[e, g] - CS.jdisp[e, g]).clone()


def unpin_yaw(CS, e, g, Y1):
    Y0 = CS.hY[e, g] - CS.jdispY[e, g]
    CS.mY0[e, g] = Y0; CS.mY1[e, g] = Y0 + _wrap(torch.as_tensor(Y1, dtype=Y0.dtype, device=Y0.device) - Y0)


def unpin_hands(CS, P, Y, RP):
    """End of chain_env._hand_now: idle / done / pre hands at their plan pose + lo_off (W 1), settling hands' retreat + W lo_off (W
    blending in), the displacement applied to every hand recorded (moves: what _hand_now applied; carries: bend_env lo_off)."""
    _unpin_state(CS)
    ph = CS.phase
    if bool(CS.tele.any()):                                          # reset: home, no displacement, no stale drive
        tl = CS.tele
        CS.jdisp[tl] = 0.0; CS.jdispY[tl] = 0.0; CS.jw0[tl] = 1.0; CS.jw[tl] = 1.0
        CS.lo_off[tl] = 0.0; CS.lo_yaw[tl] = 0.0
        if hasattr(CS, "lo_rp"):
            CS.lo_rp[tl] = 0.0
    setl = ph == 2
    entry = setl & (CS.jph != 2)
    CS.jw0 = torch.where(entry, CS.jw, CS.jw0); CS.jdisp0 = torch.where(entry[..., None], CS.jdisp, CS.jdisp0)
    sel = setl | (ph == 0) | (ph == 3) | (ph == 5)
    Wp = torch.where(setl, _settle_w(CS), torch.ones_like(CS.jw))
    st = setl & (CS.ret > 0)
    plan = torch.where(st[..., None], P - CS.jdisp0, CS.hP - CS.jdisp)
    Yp = CS.hY - CS.jdispY
    lrp = CS.lo_rp if hasattr(CS, "lo_rp") else torch.zeros_like(RP)
    P = torch.where(sel[..., None], plan + Wp[..., None] * CS.lo_off, P)
    Y = torch.where(sel, Yp + Wp * CS.lo_yaw, Y)
    RP = torch.where(sel[..., None], Wp[..., None] * lrp, RP)
    Pm, Ym, _, _ = _move_plan(CS)
    uc = FC.minjerk(torch.clamp(CS.t.float() * CS.dt / CS.dur.clamp(min=1e-6), max=1.0)); wc = FC.bend_env(uc, CS.L)
    mv, cr = ph == 4, ph == 1
    Wm = torch.where(mv, torch.zeros_like(Wp), Wp)
    if bool(mv.any()):
        ee, gg = torch.nonzero(mv, as_tuple=True)
        Wm[ee, gg] = unpin_w(CS, ee, gg, FC.minjerk(torch.clamp(CS.t[ee, gg].float() * CS.dt / CS.mdur[ee, gg].clamp(min=1e-6), max=1.0)),
                             CS.msl[ee, gg, -1])
    CS.jdisp = torch.where(mv[..., None], P - Pm, torch.where(cr[..., None], wc[..., None] * CS.lo_off, Wp[..., None] * CS.lo_off))
    CS.jdispY = torch.where(mv, Y - Ym, torch.where(cr, wc * CS.lo_yaw, Wp * CS.lo_yaw))
    CS.jw = torch.where(mv, Wm, torch.where(cr, wc, Wp)); CS.jph = ph.clone()
    return P, Y, RP


def unpin_planned(CS, P, Y, W):
    """planned_hands under FPL_UNPIN: moves W = unpin_w; carries unchanged; idle / done / pre / settling hands: the plan pose hP - jdisp
    (the pose chain_env drove minus the displacement it applied), W 1 (settles: blending in)."""
    _unpin_state(CS)
    ph = CS.phase
    mv = ph == 4
    if bool(mv.any()):
        ee, gg = torch.nonzero(mv, as_tuple=True)
        W = W.clone()
        W[ee, gg] = unpin_w(CS, ee, gg, FC.minjerk(torch.clamp(CS.t[ee, gg].float() * CS.dt / CS.mdur[ee, gg].clamp(min=1e-6), max=1.0)),
                            CS.msl[ee, gg, -1])
    sel = (ph == 0) | (ph == 2) | (ph == 3) | (ph == 5)
    Wp = torch.where(ph == 2, _settle_w(CS), torch.ones_like(W))
    P = torch.where(sel[..., None], CS.hP - CS.jdisp, P)
    Y = torch.where(sel, CS.hY - CS.jdispY, Y)
    W = torch.where(sel, Wp, W)
    return P, Y, W


def jt_target(A):
    """ArmTracker.check's IK target in joints mode: the planned hands' TCP (table frame), top-down at the planned yaw."""
    CS = A.CS
    P, Y, W = planned_hands(CS)
    if SOLID_TABLE:                                                  # the table also stops the planned hand (a bounce shove)
        from cleantables.rl.table import HAND_LOW
        P = torch.cat([P[..., :2], torch.maximum(P[..., 2:], CS.fp.org[:, None, 2:] + HAND_LOW)], -1)
    A.jP, A.jY, A.jW = P, Y, W
    tcp = P - CS.fp.org[:, None] - torch.tensor([0, 0, A.TCP_DZ], device=P.device)
    return tcp, KIN.top_down(tcp, Y)


def jt_compose(A, q_ik):
    """the plain IK pose -> the prior's update (from the arms' current q) -> q = clamp(q_IK + env (dq_prior + dq_RL))."""
    A.q_ik = q_ik.clone()
    if JACC or JPEN:
        A.jq_before = A.q.clone()
    prior_update(A)
    q = q_ik + A.jW[..., None] * (A.dq_prior + A.CS.lo_dq)
    if JVEL:                                                         # the arm pose moves at most v_max x the tick per joint from the last one
        from cleantables.rl.table import rate_limit, table_clamp
        qc = torch.minimum(torch.maximum(q, A.qlo), A.qhi)
        if SOLID_TABLE:
            qc = table_clamp(q_ik, A.jW[..., None] * (A.dq_prior + A.CS.lo_dq), A.base, A.qlo, A.qhi)[0]
        if JACC:                                                     # acceleration limit first
            from cleantables.rl.table import A_MAX, accel_limit
            qc = accel_limit(A.q, qc, A.jdq_last, torch.tensor(A_MAX, device=q.device) * (FC.TICK * A.CS.dt) ** 2)[0]
        qc = rate_limit(A.q, qc, torch.tensor(KIN.V_MAX, device=q.device) * (FC.TICK * A.CS.dt))[0]
        if SOLID_TABLE:                                              # the table still holds after the limit
            qc = table_clamp(q_ik, qc - q_ik, A.base, A.qlo, A.qhi)[0]
        return qc
    if SOLID_TABLE:                                                  # the arm pose's residual shrunk to keep the fingertips up
        from cleantables.rl.table import table_clamp
        return table_clamp(q_ik, A.jW[..., None] * (A.dq_prior + A.CS.lo_dq), A.base, A.qlo, A.qhi)[0]
    return torch.minimum(torch.maximum(q, A.qlo), A.qhi)


def _fk_palm(q, base):
    """Palm centre (table frame), finger-axis yaw and tilt from top-down of FK(q)."""
    N = q.shape[0]
    F = KIN.fk(q.reshape(-1, 7)).reshape(N, 2, -1, 4, 4)
    Tt = (base[:, :, None] @ F)[:, :, KIN.TCP]
    z = Tt[..., :3, 2]
    return Tt[..., :3, 3] - FC.ArmTracker.TCP_DZ * z, torch.atan2(Tt[..., 1, 1], Tt[..., 0, 1]), torch.acos((-z[..., 2]).clamp(-1.0, 1.0))


def _hand_pose(q, base):
    """FK(q)'s gripper pose in chain_env's hand convention: palm (table frame), yaw Y, roll r, pitch p with orientation = Rz(Y) Rx(r) Ry(p)
    (= qyaw(Y) q_tilt((r, p)); the gripper's x = the finger axis = the TCP's y, its z = - the TCP's z)."""
    N = q.shape[0]
    F = KIN.fk(q.reshape(-1, 7)).reshape(N, 2, -1, 4, 4)
    Tt = (base[:, :, None] @ F)[:, :, KIN.TCP]
    R = Tt[..., :3, :3]
    palm = Tt[..., :3, 3] - FC.ArmTracker.TCP_DZ * R[..., :, 2]
    yb = R[..., :, 0]; zb = -R[..., :, 2]                                                      # the gripper's y and z axes
    Y = torch.atan2(-yb[..., 0], yb[..., 1])
    c, s = torch.cos(Y), torch.sin(Y)
    zy = torch.stack([c * zb[..., 0] + s * zb[..., 1], -s * zb[..., 0] + c * zb[..., 1], zb[..., 2]], -1)   # Rz(-Y) z
    p = torch.asin(zy[..., 0].clamp(-1.0, 1.0)); r = torch.atan2(-zy[..., 1], zy[..., 2])
    return palm, Y, torch.stack([r, p], -1)


def jt_after_check(A):
    """after the shield: the realised offset / turn FK(q) - FK(q_IK) (obs, W_DEV), the tilt (carry cost), then the hand drive."""
    pal, yaw, tilt = _fk_palm(A.q, A.base)
    pik, yik, _ = _fk_palm(A.q_ik, A.base)
    A.jpal_ik, A.jyaw_ik = pik, yik
    _, A.jYd_ik, A.jrp_ik = _hand_pose(A.q_ik, A.base)
    A.lo_real = pal - pik; A.lo_yaw_real = _wrap(yaw - yik); A.tilt = tilt
    if GATE:                                                          # hand (FK of q) within GATE_CLOCK of the planned point (|hP - pP|)
        A.jgate = (pal + A.CS.fp.org[:, None] - A.jP).norm(dim=-1) <= GATE_CLOCK
    if REACH:                                                         # hand where FK(q) puts it (world), as the gate measures it
        A.jpal_w = pal + A.CS.fp.org[:, None]
    if JACC or JPEN:                                                  # the realised joint step of this check
        A.jdq_last = A.q - A.jq_before
    drive(A)


def drive(A):
    """Per-step hand displacement chain_env applies through env(u): FK(q_IK + dq) vs FK(q_IK), dq = dq_prior + dq_RL at full weight:
    lo_off the palm offset, lo_yaw the yaw turn, lo_rp the roll / pitch (so at env 1 the gripper is where FK(q) puts it; all 0
    when dq is 0)."""
    CS = A.CS
    qf = torch.minimum(torch.maximum(A.q_ik + A.dq_prior + CS.lo_dq, A.qlo), A.qhi)
    if SOLID_TABLE:                                                  # ... and the hand drive (the PhysX hand never enters the table)
        from cleantables.rl.table import table_clamp
        qf = table_clamp(A.q_ik, A.dq_prior + CS.lo_dq, A.base, A.qlo, A.qhi)[0]
    if JVEL:                                                         # ... and the hand drive: at most v_max x the tick from the arm's pose now
        from cleantables.rl.table import rate_limit, table_clamp
        if JACC:
            from cleantables.rl.table import A_MAX, accel_limit
            qf = accel_limit(A.q, qf, A.jdq_last, torch.tensor(A_MAX, device=qf.device) * (FC.TICK * CS.dt) ** 2)[0]
        qf = rate_limit(A.q, qf, torch.tensor(KIN.V_MAX, device=qf.device) * (FC.TICK * CS.dt))[0]
        if SOLID_TABLE:
            qf = table_clamp(A.q_ik, qf - A.q_ik, A.base, A.qlo, A.qhi)[0]
        A.jq_drive = qf
    pal, Yf, rpf = _hand_pose(qf, A.base)
    CS.lo_off[:] = torch.nan_to_num(pal - A.jpal_ik); CS.lo_yaw[:] = torch.nan_to_num(_wrap(Yf - A.jYd_ik))
    if hasattr(CS, "lo_rp"):
        CS.lo_rp[:] = torch.nan_to_num(rpf - A.jrp_ik)


def carry_rotate(CS, ee, gg, tt, X0, Y0, Q):
    """chain_env.step's carry target in joints mode: the held object (root target X0 (n, 3) table frame, orientation Q) turns with the
    hand's roll / pitch env(u) lo_rp about the grasp point (the ROT_EDIT carry rule, grasp point at the object's full yaw Y0 + its baked
    yaw). -> X0, Q, the hand's RP (n, 2), the edit (n, 3) for carry_hand."""
    from cleantables.rl import sim_env as FE
    dur = CS.dur[ee, gg]
    uc = FC.minjerk(torch.clamp(tt / dur.clamp(min=1e-6), max=1.0))
    w = FC.bend_env(uc, CS.L[ee, gg])
    rp = w[:, None] * CS.lo_rp[ee, gg]
    ec = torch.cat([rp, torch.zeros_like(rp[:, :1])], -1)
    Yh = Y0 + CS.gyaw[ee, gg]
    D = FE.qmul(FE.qmul(FE.qyaw(Yh), FC.q_tilt(rp)), FE.qyaw(-Yh))                               # the hand's extra rotation (world)
    yo = Y0 + CS.fp.yb[ee, CS.car[ee, gg]]
    HC0 = CS.HC[ee, gg, 0]; cy, sy = torch.cos(yo), torch.sin(yo)
    hpt = X0 + torch.stack([cy * HC0[:, 0] - sy * HC0[:, 1], sy * HC0[:, 0] + cy * HC0[:, 1], HC0[:, 2]], -1)   # the grasp point
    ed = (rp.abs().amax(-1) > 0)[:, None]
    X0 = torch.where(ed, hpt + (FE.qmat(D) @ (X0 - hpt)[..., None])[..., 0], X0)
    Q = torch.where(ed, FE.qmul(D, Q), Q)
    return X0, Q, rp, ec


def carry_hand(CS, ee, gg, x_, yw_, ec, P):
    """The hand riding a held object (root x_, yaw yw_): its palm P (= x_ + the yawed grasp offset) turned about the root by the
    commanded tilt ec (the ROT_EDIT rule), so the palm stays on the grasp point of the tilted object."""
    from cleantables.rl import sim_env as FE
    ywh = yw_ + CS.gyaw[ee, gg]
    Dt = FE.qmat(FE.qmul(FE.qmul(FE.qyaw(ywh), FC.q_tilt(ec[:, :2])), FE.qyaw(-ywh)))
    return torch.where((ec[:, :2].abs().amax(-1) > 0)[:, None], x_ + (Dt @ (P - x_)[..., None])[..., 0], P)


def apply_action(CS, a):
    """LowPol.tick's acting half in joints mode: the residuals and speed, then the hand re-driven with them at once."""
    dq, spd = squash_joints(a)
    if LPF:                                                           # low-pass filter: lo_dq (the obs' last dq_RL) is the filtered
        dq = ACT_LPF * CS.lo_dq + (1.0 - ACT_LPF) * torch.nan_to_num(dq)   # residual, 0 at an episode start (tracker_reset)
    if JPEN:                                                          # action rate: the residual commanded in the interval that starts now
        CS.arms.jact_prev = CS.lo_dq.clone()
    CS.lo_dq[:] = torch.nan_to_num(dq); CS.lo_spd[:] = torch.nan_to_num(spd, nan=1.0)
    drive(CS.arms)


# ----------------------------------------------------------------------------------------------------------------------------- the gate
def gate_restart(CS, e, g):
    """(FPL_GATE_CLOCK) a new move / carry of hands (e, g): its wall-time budget starts."""
    if getattr(CS, "jtw", None) is None:
        CS.jtw = torch.zeros(CS.phase.shape, device=CS.phase.device)
    CS.jtw[e, g] = 0.0


def gate_open(CS):
    """(FPL_GATE_CLOCK; once per physics step, on chain_env.step's clock) (N, 2): may this hand's plan clock advance? Yes while its hand
    was within GATE_CLOCK of its planned point at the last arm check (|hP - pP| <= gate_clock, evaluated per tick), or once the move /
    carry has used GATE_BUDGET x its planned time (moves mdur, carries the planned release time) of wall time: then it runs on its clock
    again and ends as chain_env ends it (approach -> grasp, carry -> release at the path end, free move -> idle)."""
    A = CS.arms
    if getattr(CS, "jtw", None) is None:
        CS.jtw = torch.zeros(CS.phase.shape, device=CS.phase.device)
    mv = (CS.phase == 1) | (CS.phase == 4)
    CS.jtw = torch.where(mv, CS.jtw + 1, CS.jtw)
    T = torch.where(CS.phase == 4, CS.mdur, CS.rel.float() * CS.dt)
    over = CS.jtw * CS.dt >= GATE_BUDGET * T
    gate = getattr(A, "jgate", None)
    if gate is None:
        return torch.ones_like(mv)
    return gate | over


# ----------------------------------------------------------------------------------------------------------------------------- reach done
# FPL_REACH_DONE=m covers the two events otherwise ended on the clock, the grasp and the release. Take: an approach's clock may finish,
# but the grasp fires only once the hand (FK of the arm's actual q at the last check, as for the gate) is within m of the approach's
# end (the planned grasp point); until then the approach holds at its end. Release: a carry's planned release at the path end (rel_u
# = 1; the hold / at-rest rule included) fires only once the hand is within m of the planned release point (the planned hand at the
# path end); until then the release is postponed step by step (the carry drive keeps holding the object at the goal). Drops
# (disturbances, aborted carries) and rel_u < 1 releases are not held. Fallback: past GATE_BUDGET x the planned time (moves mdur,
# carries dur + FPC_HOLD_MAX) of wall time (the gate's per-move / per-carry counter) it fires as without the flag.
def reach_restart(CS, e, g):
    """a new move / carry of hands (e, g): its wall counter starts, not a drop."""
    if getattr(CS, "jtw", None) is None:
        CS.jtw = torch.zeros(CS.phase.shape, device=CS.phase.device)
    CS.jtw[e, g] = 0.0
    if getattr(CS, "jdrop", None) is not None:
        CS.jdrop[e, g] = False


def reach_drop(CS, e, g):
    """chain_env.drop: this release is a drop (a disturbance / an aborted carry): never held."""
    if getattr(CS, "jdrop", None) is None:
        CS.jdrop = torch.zeros(CS.phase.shape, dtype=torch.bool, device=CS.phase.device)
    CS.jdrop[e, g] = True


def _reach_hand(CS):
    A = CS.arms
    if getattr(A, "jpal_w", None) is None:
        A.jpal_w = _fk_palm(A.q, A.base)[0] + CS.fp.org[:, None]
    return A.jpal_w


def reach_take_ok(CS):
    """(N, 2) False only for an approach that ends in a take whose hand is > REACH_DONE from the path end, within its wall-time budget."""
    take = (CS.phase == 4) & (CS.after == 1)
    d = (_reach_hand(CS) - CS.mpath[:, :, -1]).norm(dim=-1)
    over = CS.jtw * CS.dt >= GATE_BUDGET * CS.mdur
    return ~take | (d <= REACH_DONE) | over


def reach_release_point(CS):
    """(N, 2, 3) the planned hand at the carry path's end (world): the path end + the grasp's palm offset at the planned final object yaw
    (planned_hands' carry hand at u = 1, without the fading bounce shove)."""
    N = CS.phase.shape[0]; ar = torch.arange(N, device=CS.phase.device); car = CS.car.clamp(0, CS.K - 1)
    yb = torch.stack([CS.fp.yb[ar, car[:, g]] for g in (0, 1)], 1)
    yo = CS.yaw0 + CS.dyaw + yb
    HC0 = CS.HC[:, :, 0]; cz, sz = torch.cos(yo), torch.sin(yo)
    return CS.path[:, :, -1] + CS.fp.org[:, None] + torch.stack([cz * HC0[..., 0] - sz * HC0[..., 1], sz * HC0[..., 0] + cz * HC0[..., 1], HC0[..., 2]], -1)


def reach_hold_release(CS):
    """Right before chain_env's release check: a due planned release (t >= rel, rel_u = 1, not a drop) whose hand is > REACH_DONE from the
    release point, within its budget, is postponed one step (rel = t + 1, end = rel + n_dec), so the carry keeps holding at the goal."""
    if getattr(CS, "jtw", None) is None:
        CS.jtw = torch.zeros(CS.phase.shape, device=CS.phase.device)
    if getattr(CS, "jdrop", None) is None:
        CS.jdrop = torch.zeros(CS.phase.shape, dtype=torch.bool, device=CS.phase.device)
    if not GATE:                                                      # the gate's wall counter, when the gate is off
        CS.jtw = torch.where((CS.phase == 1) | (CS.phase == 4), CS.jtw + 1, CS.jtw)
    due = (CS.phase == 1) & (CS.t >= CS.rel) & (CS.relu >= 0.999) & ~CS.jdrop
    if not bool(due.any()):
        return
    d = (_reach_hand(CS) - reach_release_point(CS)).norm(dim=-1)
    over = CS.jtw * CS.dt >= GATE_BUDGET * (CS.dur + FC.HOLD_MAX_S)
    hold = due & (d > REACH_DONE) & ~over
    CS.rel = torch.where(hold, CS.t + 1, CS.rel); CS.end = torch.where(hold, CS.rel + CS.n_dec, CS.end)


# ----------------------------------------------------------------------------------------------------------------------------- the prior
def prior_update(A, q=None, phase=None):
    """FPL_PRIOR=nullspace, from the arms' current q: every link sphere (link0 / link1 left out) within PRIOR_D0 of the other arm asks
    for v = k (d0 - d) / d0 n; dq = N_h sum J_o+ v (damped pseudo-inverses; N_h the hand's 6-D task null space), capped at PRIOR_QD
    (max-abs), integrated into the persistent dq_prior (|.| <= J_MAX), which decays with PRIOR_TAU while no sphere is within d0."""
    if PRIOR != "nullspace":
        return
    q = A.q if q is None else q
    CS = A.CS; dev = q.device; N = q.shape[0]
    tick_s = FC.TICK * CS.dt
    Wf = A.base[:, :, None] @ KIN.fk(q.reshape(-1, 7)).reshape(N, 2, -1, 4, 4)
    Pall, r = KIN.spheres(q.reshape(-1, 7), A.base.reshape(-1, 4, 4))
    Pall = Pall.reshape(N, 2, -1, 3)
    ip = torch.nonzero(KIN._mask(q, KIN.TABLE_SKIP)).flatten()
    Pn = Pall[:, :, ip]; rr = (r - FC.ARM_PAD_SHRINK).clamp(min=0.005)[ip]; RR = rr[:, None] + rr[None, :]
    da, ja, db, jb = [], [], [], []
    for i in range(0, N, 4096):
        D = torch.cdist(Pn[i:i + 4096, 0], Pn[i:i + 4096, 1]) - RR
        a_, ia = D.min(2); b_, ib = D.min(1)
        da.append(a_); ja.append(ia); db.append(b_); jb.append(ib)
    d = torch.stack([torch.cat(da), torch.cat(db)], 1); j = torch.stack([torch.cat(ja), torch.cat(jb)], 1)
    act = d < PRIOR_D0
    anyact = act.any(-1)
    A.dq_prior = torch.where(anyact[..., None], A.dq_prior, A.dq_prior * math.exp(-tick_s / PRIOR_TAU))
    if bool(anyact.any()):
        e, g, i = torch.nonzero(act, as_tuple=True)
        o = 1 - g
        p_i = Pn[e, g, i]; p_j = Pn[e, o, j[e, g, i]]
        nv = (p_i - p_j) / (p_i - p_j).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        v = PRIOR_K * ((PRIOR_D0 - d[e, g, i]) / PRIOR_D0)[:, None] * nv
        We = Wf[e, g]
        fi = torch.as_tensor(KIN.S_FRAME, device=dev)[ip[i]]
        zk, ok_ = We[:, 1:8, :3, 2], We[:, 1:8, :3, 3]
        msk = (torch.arange(1, 8, device=dev)[None] <= fi.clamp(max=7)[:, None]).float()
        Jo = (torch.linalg.cross(zk, p_i[:, None] - ok_, dim=-1) * msk[..., None]).transpose(1, 2)
        x = torch.linalg.solve(Jo @ Jo.transpose(1, 2) + 0.05 ** 2 * torch.eye(3, device=dev), v[..., None])
        jv = (Jo.transpose(1, 2) @ x)[..., 0]
        sumv = torch.zeros(2 * N, 7, device=dev).index_add_(0, e * 2 + g, jv).reshape(N, 2, 7)
        ea, ga = torch.nonzero(anyact, as_tuple=True)
        Jh = KIN.jacobian(Wf[ea, ga])
        Jp = torch.linalg.solve(Jh @ Jh.transpose(1, 2) + 0.02 ** 2 * torch.eye(6, device=dev), Jh).transpose(1, 2)
        Np = torch.eye(7, device=dev) - Jp @ Jh
        qd = (Np @ sumv[ea, ga][..., None])[..., 0]
        qd = qd * (PRIOR_QD / qd.abs().amax(-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        A.dq_prior[ea, ga] = A.dq_prior[ea, ga] + qd * tick_s
    A.dq_prior = A.dq_prior.clamp(-J_MAX, J_MAX)
    ph = CS.phase if phase is None else phase
    live = (ph == 1) | (ph == 4)
    A.jst["arm_ticks"] += float(live.sum()); A.jst["active"] += float((anyact & live).sum())
    A.jst["dq_abs"] += float((A.dq_prior.abs().mean(-1) * live).sum())


# ----------------------------------------------------------------------------------------------------------------------------- obs / costs
def arm_feats(arm_, CS):
    """The 46 per-arm features with lo_off / lo_yaw = the realised hand offset / turn."""
    A = CS.arms
    arm_ = arm_.clone(); arm_[..., 34:37] = A.lo_real; arm_[..., 37] = A.lo_yaw_real
    return arm_


def obs_extra(CS):
    """Per arm sin q, cos q, the last dq_RL, dq_prior (28 each; arm 0 then arm 1)."""
    A = CS.arms; N = A.q.shape[0]
    return torch.cat([torch.sin(A.q), torch.cos(A.q), CS.lo_dq, A.dq_prior], -1).reshape(N, -1)


def add_costs(lp, CS, r):
    """Joint-mode costs on the state at this tick: W_DEV x the realised hand offset (m, per s, moving hands), W_TILT x the
    hand's tilt while carrying, W_LIM x the joints in the outer 10 % of their range."""
    from cleantables.rl import controller as LP
    A = CS.arms
    st = ((CS.phase == 1) | (CS.phase == 4)).float()
    c_dev = LP.W_DEV * (A.lo_real.norm(dim=-1) * st).sum(1) * LP.TICK_S
    c_tilt = W_TILT * (A.tilt * (CS.phase == 1).float()).sum(1) * LP.TICK_S
    un = ((A.q - (A.qlo + A.qhi) / 2) / ((A.qhi - A.qlo) / 2)).abs()
    live = (~((CS.phase == 5).all(1))).float()
    c_lim = W_LIM * ((un - 0.9).clamp(min=0) / 0.1).sum(-1).sum(-1) * live * LP.TICK_S
    v_ = lp.buf["valid"][-1].float()
    if JPEN:                                                          # Isaac Lab action_rate_l2 / joint_vel_l2: weight x value x dt
        w_ar = float(os.environ.get("FPL_W_ACTION_RATE") or 0); w_jv = float(os.environ.get("FPL_W_JOINT_VEL") or 0)
        c_ar = w_ar * ((CS.lo_dq - A.jact_prev) ** 2).sum((1, 2)) * LP.TICK_S
        c_jv = w_jv * ((A.jdq_last / LP.TICK_S) ** 2).sum((1, 2)) * LP.TICK_S
        lp.stats["action_rate"] = lp.stats.get("action_rate", 0.0) + float((c_ar * v_).sum())
        lp.stats["joint_vel"] = lp.stats.get("joint_vel", 0.0) + float((c_jv * v_).sum())
        r = r - c_ar - c_jv
    for k_, c_ in (("dev", c_dev), ("tilt", c_tilt), ("lim", c_lim)):
        lp.stats[k_] = lp.stats.get(k_, 0.0) + float((c_ * v_).sum())
    return r - c_dev - c_tilt - c_lim
