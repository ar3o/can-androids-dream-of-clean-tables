"""Stage A environment: chains of carries on the batched tables of cleantables.rl.sim_env (FP_HANDS=2).

Two asynchronous grippers, the grasp pose as a parameter (top-down + yaw; the hand's shape at that yaw, the same in global_plan's
cost), the release point, decisions from the actual state at every release, random disturbances (training), and the frozen
start-frame judge. One rule for every object; goals only from the frozen WM; grasp / route candidates from generic geometry.
  ChainSim(fp)            the stepper: reset(layout) / start(envs, g, carried, path, dyaw, gyaw, rel_u) / step() / poses(); per gripper:
                          idle -> carrying (force-limited carry of sim_env: arc-length minimum jerk, start offset faded out, yaw from the
                          current yaw, duration = length / 0.25 m/s in [1, 4] s, hold 0.15 s at u = 1) -> released at rel_u -> settle ->
                          idle (= a decision is due). The hand (gripper.boxes_g at the grasp yaw) rides the object; kinematic, so the two
                          hands never touch each other physically: their overlap is measured geometrically (hand_hand) and is a penalty.
  FrozenJudge(LP, ...)    at the start frame: the ensemble's T1 per object, reference-relative (the most likely reference j and the members'
                          mean offset in j's frame); later targets follow j's actual pose (j = the table or the object itself: fixed).
                          dist_cm (capped 10), met (< MET_CM), static (the start target within STATIC_CM of the start: moving it is a penalty).
  BaselineDecider(LP, J)  the non-learned policy (reference for headroom and the BC target): the live WM (LivePlanner, the actual state) chooses
                          the object (the frozen judge's met objects, the other gripper's object and top-down-ungraspable objects excluded),
                          the default short-axis grasp yaw (or the narrowest feasible one), global_plan's path for that grasp, release
                          at u = 1; 'done' on the WM's STOP or when nothing is left.
  run_episodes(...)       both grippers on every env until both are done (or max carries / max time / every frozen goal met), with the
                          per-carry log and the episode metrics.
  $SIM_PY -m cleantables.rl.chain_env --headless --split heldout --copies 8 [--disturb 0] [--tag m1_baseline]"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("FP_HANDS", "2")
from cleantables.rl import sim_env as FE                                          # noqa: E402

MET_CM, STATIC_CM, CAP_CM = 3.0, 3.0, 10.0
MAX_OPEN = 0.08                                                     # the Franka hand's largest finger opening (m)
GYAWS = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)            # top-down grasp yaw candidates (baked frame; 180 deg symmetric)
DECIDE_SETTLE_S = float(os.environ.get("FPC_SETTLE", "1.0"))                                               # physics after a release before that gripper decides again (the reference settle time)
HOLD_MAX_S, HOLD_TOL_M, HOLD_VTOL = float(os.environ.get("FPC_HOLD_MAX", "1.0")), 0.005, 0.02               # at the goal (rel_u = 1), after the 0.15 s hold: release once the object is
                                                                    # within 5 mm of the goal and slower than 2 cm/s, at the latest after 1 s
TICK = int(os.environ.get("FPC_TICK", "6"))                                                            # decisions are checked every TICK physics steps
W = Path(os.environ.get("WM_EVAL_OUT", CT_DATA + "/wm_eval")) / "wm_v2_world"


KNOCK_V = tuple(float(v) for v in os.environ.get("FPC_KNOCK_V", "0.3,0.8").split(","))   # knock speed range (m/s)
KNOCK_INWARD = os.environ.get("FPC_KNOCK_INWARD", "0") == "1"
ARM_FAST = os.environ.get("FPC_ARM_FAST", "0") == "1"                     # smaller IK budgets for the arm check
COLD_IT, COLD_SEEDS = (15, 2) if ARM_FAST else (40, 4)
TILT_IT, TILT_SEEDS = (15, 1) if ARM_FAST else (30, 2)
CONT_EVAL = os.environ.get("FPC_CONT_EVAL", "0") == "1"               # evaluation only. Per episode: do the task objects
CONT_FRUIT = ("banana", "apple", "orange", "lemon", "lime", "pear", "peach", "plum", "grape", "strawberry", "kiwi", "mango", "avocado", "tomato")  # end inside
CONT_TOOLS = ("wrench", "plier", "screwdriver", "hammer", "knife", "scissor", "tape", "spanner", "saw", "drill", "file", "chisel", "clamp")  # the container
                                                                        # (names used only here, never by the robot, reward or judge)
INSTR_P = float(os.environ.get("FPC_INSTR_P", "0.0"))                     # P(the other instruction) per episode (instruction sampling)
STOP_ABORT = os.environ.get("FPC_STOP_ABORT", "0") == "1"                   # an aborted carry halts before it opens (released at rest)
REFLEX = os.environ.get("FPC_REFLEX", "0") == "1"                         # collision reflex (stand-in for the arm's force limits)
REFLEX_V, REFLEX_GAP = float(os.environ.get("FPC_REFLEX_V", "0.3")), float(os.environ.get("FPC_REFLEX_GAP", "0.03"))
HAND_F_CAP = float(os.environ.get("FPC_HAND_F_CAP", "20.0"))              # max push force of a whole hand (N), from the libfranka example thresholds
HAND_T_CAP = float(os.environ.get("FPC_HAND_T_CAP", "2.0"))               # its max torque (N m)
HAND_KP, HAND_KD = (2 * math.pi * 8.0) ** 2, 2 * (2 * math.pi * 8.0)      # the PD: an 8 Hz critically damped tracker (per unit mass / inertia)
MOVE_ROUTE = os.environ.get("FPC_MOVE_ROUTE", "0") == "1"                 # a free move's zero-edit route = the collision-free over-route
HAND_PEN = os.environ.get("FPC_HAND_PEN", "0") == "1"                     # hand-object penetration: abort above HAND_PEN_ABORT, cost below
PEN_SDF = os.environ.get("FPC_PEN_SDF", "1") == "1"                       # penetration from the objects' mesh SDFs (default), boxes only as fallback
HAND_PEN_ABORT = float(os.environ.get("FPC_HAND_PEN_ABORT", "0.01"))
FAST2 = os.environ.get("FP_FAST2", "1") == "1"
PROF = os.environ.get("FP_PROF", "0") == "1"                                # CUDA-synced per-step split timers (ChainSim.prof)
PEN_TGT_ALL = os.environ.get("FPC_PEN_TGT_ALL", "0") == "1"                # the take's target object is excluded from the penetration test over the
                                                                          # whole approach, not only its last 20 %
TERM_LOST = os.environ.get("FPC_TERM_LOST", "0") == "1"                   # robot-caused off-table object -> the episode ends at once
REF_START = os.environ.get("FPC_REF_START", "0") == "1"                   # references hold their start pose for the goals
MET3 = os.environ.get("FPC_MET3", "0") == "1"                             # met_3d recorded per episode (and trained on by cleantables.rl.train)
MET_NEW = os.environ.get("FPC_MET_NEW", "0") == "1"                       # met = xy within the WM's own sd of the start-frame goal
LOWPOL = os.environ.get("FPL_ON", "0") == "1"                              # the shared two-arm low-level controller (cleantables.rl.controller)
JOINTS = os.environ.get("FPL_ACTION", "offset") == "joints"               # the controller acts in joint space (cleantables.rl.joint_control);
if JOINTS:                                                                # unset / "offset": the hand-offset controller
    from cleantables.rl import joint_control as JI                                      # noqa: E402
UNPIN = JOINTS and os.environ.get("FPL_UNPIN", "0") == "1"                # joints only: the residual acts in every phase
GATE = JOINTS and float(os.environ.get("FPL_GATE_CLOCK") or 0) > 0        # joints only: joint_control.gate_open, the pure-pursuit clock
REACH = JOINTS and float(os.environ.get("FPL_REACH_DONE") or 0) > 0       # joints only: joint_control.reach_*, grasp / release wait for the hand
NEAR_M = float(os.environ.get("FPC_NEAR_M", "0"))                         # clearance (m) below
                                                                          # which the arms' closeness is costed, (NEAR_M - gap) / NEAR_M per s (0 = off)
ARM_ACCURATE = os.environ.get("FPC_ARM_ACCURATE", "0") == "1"               # arm-arm contact measured with the accurate padding, no shield
ARM_SHIELD = os.environ.get("FPC_ARM_SHIELD", "0") == "1"                   # projection shield for arm-arm contact
ARM_PAD_SHRINK = float(os.environ.get("FPC_ARM_PAD_SHRINK", "0.008"))       # the shield's sphere radii minus this (m): ~0.5 cm padding left
SH_ITERS, SH_STEP = int(os.environ.get("FPC_SH_ITERS", "4")), float(os.environ.get("FPC_SH_STEP", "0.15"))
ARM_SOLID = os.environ.get("FPC_ARM_SOLID", "0") == "1"                     # the arms block each other
ARM_BLOCK_S = float(os.environ.get("FPC_ARM_BLOCK_S", "1.0"))               # blocked this long (s) -> the take aborts ("arm_block")
ARM_ABORT = float(os.environ.get("FPC_ARM_ABORT", "0"))                     # arm-arm overlap deeper than this (m) aborts a take (0 = off)
ARM_BOUNCE = os.environ.get("FPC_ARM_BOUNCE", "0") == "1"                  # arms that touch bounce:
BOUNCE_S = float(os.environ.get("FPC_BOUNCE_S", "0.5"))                       # every moving arm of that env plays its move backward for BOUNCE_S
BOUNCE_COOL_S = float(os.environ.get("FPC_BOUNCE_COOL_S", "1.0"))             # (smoothly, along its own clear path); no new bounce for BOUNCE_COOL_S
BOUNCE_MODE = os.environ.get("FPC_BOUNCE_MODE", "rewind")                     # "rewind" (as above) or "push":
BOUNCE_PUSH_M = float(os.environ.get("FPC_BOUNCE_PUSH_M", "0.05"))            # the touching hand's target is shoved BOUNCE_PUSH_M away from the
BOUNCE_TAU_S = float(os.environ.get("FPC_BOUNCE_TAU_S", "0.3"))               # other arm (along the closest sphere pair), fading with BOUNCE_TAU_S; the move
BOUNCE_STUCK_N = int(os.environ.get("FPC_BOUNCE_STUCK_N", "3"))               # keeps progressing; >= BOUNCE_STUCK_N bounces within ARM_BLOCK_S abort the
PUSH_BOUNCE = ARM_BOUNCE and BOUNCE_MODE == "push"                            # take ("arm_bounce") so the planner re-decides. Needs FPL_ON (the bend block).
PUSH_CONTACT = os.environ.get("FPC_PUSH_CONTACT", "0") == "1"             # an object's motion is charged as a push only if the
PUSH_CONTACT_M = float(os.environ.get("FPC_PUSH_CONTACT_M", "0.01"))         # robot (arm / hand / fingers, or an object it carries) touched it
PUSH_RECENT_S = float(os.environ.get("FPC_PUSH_RECENT_S", "0.5"))            # (within PUSH_CONTACT_M) within the last PUSH_RECENT_S
                                                                              # (a knock and its slide); other motion is physics settling
MET_NOREQ = os.environ.get("FPC_MET_NOREQ", "0") == "1"                   # met_new without the judge re-query: xy within the WM's sd of the start-frame goal and at rest
                                                                          # (with MET_NEW alone: also the frozen judge re-queried on the current scene says it stays;
                                                                          # the re-query stays logged as stay_movers; met_new replaces met_3d in the reward, the old one is logged as met3_old)
GRASP_FILT = os.environ.get("FPC_GRASP_FILT", "0") == "1"                 # a take whose grasp opening exceeds the hand aborts at once
OBJ_SPHERES = os.environ.get("FPC_OBJ_SPHERES", "0") == "1"               # 3 spheres per object for the arm checks
IK_YAW = os.environ.get("FPC_IK_YAW", "0") == "1"                         # arm IK at the hand's actual yaw (+180 deg) + self-collision
EXCUSE_WIN = os.environ.get("FPC_EXCUSE_WIN", "0") == "1"                 # a disturbed object's fall is excused only soon after it
EXCUSE_S = 3.0
REL_STOP = os.environ.get("FPC_REL_STOP", "0") == "1"                     # an early release (u < 0.9) releases at rest
DIST_SEEDED = os.environ.get("FPC_DIST_SEEDED", "0") == "1"               # the same disturbance schedule for the copies of a table
DR_EP = os.environ.get("FPC_DR_EP", "0") == "1"                           # per-episode mass resampling (x0.5-2, log-uniform)
FREEZE_REF = os.environ.get("FPC_FREEZE_REF", "0") == "1"                 # a reference off the table freezes its dependents' goals
OFF_Z = 0.05                                                              # an object this far below the table top has left the table
ROT_EDIT = os.environ.get("FPW_ROT_EDIT", "0") == "1"                     # learned hand roll / pitch / yaw along routes and carries
# FPW_REACH_REPAIR: route / carry points the executing arm cannot reach are pulled
# horizontally toward that arm's base (same rule for every object; start / end fixed). See reach_pull().
REACH_REPAIR = os.environ.get("FPW_REACH_REPAIR", "0") == "1"
# FPW_SIL, self-imitation: each carry's executed root trajectory (table frame) is recorded once per decision tick (TICK steps) from its
# start to its release and attached to its ev record ("traj"); used by wm_actor.sil_select / sil_update.
SIL = os.environ.get("FPW_SIL", "0") == "1"
SIL_T = 512                                                                 # samples per carry kept (51 s at TICK 12 / 120 Hz); longer ones are flagged
RR_J7, RR_A7, RR_D1 = 0.2104, 0.088, 0.333                                  # Franka: TCP -> joint 7 (top-down), joint 7 -> 6, base -> shoulder (m)
RR_R, RR_RT, RR_MAX, RR_NEAR, RR_SMOOTH = 0.715, 0.74, 0.30, 0.32, 2        # wrist reach top-down / mid-route tilt (1 cm margin), max pull, floor
BEND_REACH = os.environ.get("FPC_BEND_REACH", "0") == "1"                # the controller's bend kept inside the arm's reach
NO_HOME_BEND = os.environ.get("FPC_NO_HOME_BEND", "0") == "1"           # no controller bend / turn on go-home retreats
FLAT_BEND = os.environ.get("FPC_FLAT_BEND", "0") == "1"                 # the WM path as a hint: the
BEND_START_M = float(os.environ.get("FPC_BEND_START_M", "0.05"))       # controller's offset applies in full along the whole move, faded in over the
BEND_END_M = float(os.environ.get("FPC_BEND_END_M", "0.10"))   # first BEND_START_M and out over the last BEND_END_M of path (pick / place stay exact)
BEND_BLEND = 1.0                                                        # weight of the flat envelope vs sin(pi u); the trainer ramps it (FPC_BEND_BLEND_ITS)
                                                                        # flat bend off: the sin(pi u) bump (zero at both ends)
FLUID_WAIT = os.environ.get("FPC_FLUID_WAIT", "0") == "1"               # the controller moves a waiting
WAIT_MOVE_M = float(os.environ.get("FPC_WAIT_MOVE_M", "0.10"))            # arm: its bend ramps 0 -> 1 over the wait (the hand ends displaced), at most
WAIT_R_OFF = float(os.environ.get("FPL_R_OFF", "0.25"))                   # WAIT_MOVE_M per wait per axis (bend x WAIT_MOVE_M / the controller's cap);
WAIT_PALM_Z0 = float(os.environ.get("FPW_PALM_Z0", "0.09"))               # never below the free-move palm floor (table frame) unless it started lower
ABORT_FREE = os.environ.get("FPC_ABORT_FREE", "1") == "1"            # aborted takes do not count toward the carry cap
MP = 33                                                                   # points of a hand-move route (approach / free move)
N_RCTRL = 5                                                               # route-edit spline control points (the carry edit's count)


def edit_path_t(route, resid):
    """torch twin of edit_path: route (n, M, 3), resid (n, n_ctrl, 3) (m) -> the route plus a Catmull-Rom displacement through
    (0, resid..., 0) over the arc length (endpoints fixed)."""
    import torch
    n, M, _ = route.shape; nc = resid.shape[1]
    s = torch.cat([torch.zeros(n, 1, device=route.device), (route[:, 1:] - route[:, :-1]).norm(dim=-1).cumsum(1)], 1)
    s = s / s[:, -1:].clamp(min=1e-9)
    z0 = torch.zeros(n, 1, 3, device=route.device)
    vals = torch.cat([z0, resid, z0], 1); vp = torch.cat([vals[:, :1], vals, vals[:, -1:]], 1)
    h = 1.0 / (nc + 1)
    k = (s / h).long().clamp(0, nc); u = ((s - k.float() * h) / h)[..., None]
    gat = lambda o: vp.gather(1, (k + o)[..., None].expand(n, M, 3))                                   # noqa: E731
    p0, p1, p2, p3 = gat(0), gat(1), gat(2), gat(3)
    disp = 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    out = route + disp
    out[:, 0], out[:, -1] = route[:, 0], route[:, -1]
    return out


def carried_contact(objs, phase, car):
    """(N, K) clearance (m) from every object to the object(s) the hands carry (phase 1), from the objects' contact spheres objs
    (N, K, S, 4) [x, y, z, r]; 1e3 where no hand carries or for the carried object itself."""
    import torch
    N, K = objs.shape[:2]; ar = torch.arange(N, device=objs.device)
    out = torch.full((N, K), 1e3, device=objs.device)
    for g in (0, 1):
        hm = phase[:, g] == 1
        if not bool(hm.any()):
            continue
        sh = objs[ar, car[:, g].clamp(0, K - 1)]                                                       # (N, S, 4)
        dd = (objs[:, :, :, None, :3] - sh[:, None, None, :, :3]).norm(dim=-1) - objs[:, :, :, None, 3] - sh[:, None, None, :, 3]
        dd = dd.amin(-1).amin(-1)                                                                        # (N, K)
        dd[ar, car[:, g].clamp(0, K - 1)] = 1e3
        out = torch.where(hm[:, None], torch.minimum(out, dd), out)
    return out


def bounce_trigger(touch_env, phase, t, rel, cool, b_left, n_b, n_c):
    """envs whose two arms really touch (touch_env (N,) bool): each of their moving arms (approach / free move / carry, not at a due
    release / abort-drop, not in its cool-down) starts a bounce of n_b steps and a cool-down of n_c steps. -> b_left, cool, started (N, 2)."""
    import torch
    mv = (phase == 1) | (phase == 4)
    due = (phase == 1) & (rel <= t + 1)
    st = touch_env[:, None] & mv & ~due & (cool <= 0)
    return torch.where(st, torch.full_like(b_left, n_b), b_left), torch.where(st, torch.full_like(cool, n_c), cool), st


def bounce_advance(phase, t, rel, b_left, cool):
    """one sim step: a bouncing arm (b_left > 0, still moving, no due release) goes back one step along its move instead of forward.
    -> back (N, 2) bool (their forward advance is skipped), t, b_left, cool."""
    import torch
    mv = (phase == 1) | (phase == 4)
    due = (phase == 1) & (rel <= t + 1)
    back = (b_left > 0) & mv & ~due
    t = torch.where(back, (t - 1).clamp(min=0), t)
    return back, t, (b_left - 1).clamp(min=0), (cool - 1).clamp(min=0)


def push_dirs(pa, pb, ra, rb):
    """unit push directions (N, 2, 3) from the closest pair of arm spheres: arm 0's points from arm 1's sphere centre to its own, arm 1's
    the opposite. pa (N, Sa, 3), pb (N, Sb, 3) sphere centres (table frame), ra (Sa,), rb (Sb,) radii."""
    import torch
    d = torch.cdist(pa, pb) - (ra[:, None] + rb[None, :])                                          # (N, Sa, Sb)
    Sb = d.shape[2]; k = d.flatten(1).argmin(1); i = k // Sb; j = k % Sb
    ar = torch.arange(d.shape[0], device=pa.device)
    v = pa[ar, i] - pb[ar, j]
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.stack([v, -v], 1)


def bounce_stuck(times, started, step, win, need):
    """times (N, 2, need): the steps of each hand's last `need` bounces (-1e9 = none); started (N, 2) bool: bouncing now ->
    times, stuck (N, 2) (this bounce is the need-th within win steps)."""
    import torch
    new = torch.cat([times[..., 1:], torch.full_like(times[..., :1], int(step))], -1)
    times = torch.where(started[..., None], new, times)
    return times, started & ((int(step) - times[..., 0]) <= win)


def cont_check(meta, t, other, P_e, L_e):
    """Evaluation only. For template t under its own (other=False) or the other instruction: the task objects (fruit -> the fruit bowl, tools ->
    the toolbox, by object name) and the container; -> dict(cont_found, cont_n, cont_in (end pose P_e), cont_in0 (start layout L_e)); footprint
    test in xy on the container's yawed box (its extents)."""
    inst = (meta["inst_other"] if other else meta["inst_own"])[t] or ""; fruit = "fruit" in inst.lower()
    ids = meta["ids"][t]; ext = meta["ext"][t]
    cont = [i for i, n in enumerate(ids) if ("bowl" in n.lower() if fruit else "toolbox" in n.lower()) and "lid" not in n.lower()]   # (never a lid)
    task = [i for i, n in enumerate(ids) if i not in cont and any(k in n.lower() for k in (CONT_FRUIT if fruit else CONT_TOOLS))]
    if not cont or not task:
        return dict(cont_found=False, cont_n=0, cont_in=0, cont_in0=0)
    def _in1(P, i, c):
        x, y, z, w = [float(v) for v in P[c, 3:7]]; a_ = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dx, dy = float(P[i, 0] - P[c, 0]), float(P[i, 1] - P[c, 1]); rx = math.cos(-a_) * dx - math.sin(-a_) * dy; ry = math.sin(-a_) * dx + math.cos(-a_) * dy
        return abs(rx) <= ext[c][0] / 2 and abs(ry) <= ext[c][1] / 2
    def _in(P, i):                                                       # inside any container of the task's kind (some tables have two toolboxes)
        return any(_in1(P, i, c) for c in cont)
    return dict(cont_found=True, cont_n=len(task), cont_in=int(sum(_in(P_e, i) for i in task)), cont_in0=int(sum(_in(L_e, i) for i in task)))


def bend_env(u, L):
    """the weight of the controller's offset / turn along a move at progress u (0..1) of a path of length L (m): sin(pi u) unless
    FLAT_BEND; then 1 in the middle, smoothstep in over the first BEND_START_M and out over the last BEND_END_M of path."""
    import torch
    if not FLAT_BEND:
        return torch.sin(math.pi * u.clamp(0, 1))
    uc = u.clamp(0, 1); a_ = (uc * L / BEND_START_M).clamp(0, 1); b_ = ((1 - uc) * L / BEND_END_M).clamp(0, 1)
    flat = (a_ * a_ * (3 - 2 * a_)) * (b_ * b_ * (3 - 2 * b_))
    if BEND_BLEND >= 1.0:
        return flat
    return (1.0 - BEND_BLEND) * torch.sin(math.pi * uc) + BEND_BLEND * flat     # blended in over iterations (set by the trainer)


def bend_scale(base, tcp, bend, tilt, iters=6):
    """(FPC_BEND_REACH) the largest s in [0, 1] such that the TCP target tcp + s * bend (n, 3, table frame) stays inside the closed-form
    reach of the arm at base (n, 4, 4, table frame) — the reach_pull geometry (the wrist within RR_R of the shoulder, RR_RT where a mid-route
    tilt is allowed: tilt (n,) bool); s = 1 where the full bend is reachable (also when it brings an out-of-reach plan point back into reach),
    s = 0 where neither the plan point nor the full bend is reachable. The reachable set
    (points within R of the shoulder-height disk of radius RR_A7) is convex, so bisection on the segment is exact up to 2^-iters. -> s (n,)"""
    import torch
    R = torch.where(tilt, torch.full_like(tcp[:, 0], RR_RT), torch.full_like(tcp[:, 0], RR_R))
    bx, bz = base[:, :2, 3], base[:, 2, 3] + RR_D1

    def ok(x):
        hd = (x[:, :2] - bx).norm(dim=-1); dz = x[:, 2] + RR_J7 - bz
        return ((hd - RR_A7).clamp(min=0) ** 2 + dz ** 2) < R ** 2
    ok0, ok1 = ok(tcp), ok(tcp + bend)
    lo = torch.zeros_like(R); hi = torch.ones_like(R)
    for _ in range(iters):
        mid = 0.5 * (lo + hi); om = ok(tcp + mid[:, None] * bend)
        lo = torch.where(om, mid, lo); hi = torch.where(om, hi, mid)
    return torch.where(ok1, torch.ones_like(lo), torch.where(ok0, lo, torch.zeros_like(lo)))


def reach_pull(base, tcp, tilt, fixed):
    """(FPW_REACH_REPAIR) TCP targets tcp (n, M, 3) (table frame) of one arm per row (base (n, 4, 4), table frame, yaw-only mount).
    Reach in closed form (no IK): a top-down TCP is reachable iff the wrist (joint 6: RR_J7 above the TCP, then RR_A7 toward the shoulder; the
    hand yaw is free to joint 7) lies within RR_R of the shoulder (RR_D1 above the base); mid-route (tilt (n, M) True: position + a tilt up to
    45 deg, as the arm check allows there) within RR_RT. Calibrated on 12k random targets on the humanoid bases against the full Franka IK (40 its,
    4 seeds): 100.0 % agreement at 0.725 (top-down, yaw free and at a fixed yaw + 180), 99.6 % at 0.75 (tilt); 1 cm margin here. A point out
    of reach is pulled horizontally toward the base by the exact amount that brings it to RR_R (at most RR_MAX, never nearer than RR_NEAR);
    the pull is spread to the neighbours (every point keeps its own required pull, then RR_SMOOTH plain passes round the onset); fixed (n, M)
    points never move; heights never change. -> new tcp (n, M, 3), stats dict."""
    import torch
    n, M, _ = tcp.shape
    rel = tcp[..., :2] - base[:, None, :2, 3]; hd = rel.norm(dim=-1)                              # (n, M) horizontal distance to the shoulder
    dirn = -rel / hd[..., None].clamp(min=1e-6)                                                   # toward the base
    dz = tcp[..., 2] + RR_J7 - (base[:, None, 2, 3] + RR_D1)
    R = torch.where(tilt, torch.full_like(hd, RR_RT), torch.full_like(hd, RR_R))
    bad = (((hd - RR_A7).clamp(min=0) ** 2 + dz ** 2) >= R ** 2) & ~fixed & (hd > RR_NEAR)
    need = hd - (RR_A7 + torch.sqrt((R ** 2 - dz ** 2).clamp(min=0)))                             # the pull that reaches R (inf if dz >= R)
    need = torch.where(dz.abs() < R, need, torch.full_like(need, 1e3))
    cap = torch.minimum(torch.full_like(hd, RR_MAX), (hd - RR_NEAR).clamp(min=0))
    pull = torch.where(bad, torch.minimum(need.clamp(min=0), cap), torch.zeros_like(hd))
    st = dict(pts=n * M, chk=int((~fixed).sum()), fail=int(bad.sum()), pulled=0, cm_sum=0.0, cm_max=0.0, unreach=int((bad & (need > cap)).sum()))
    if st["fail"] == 0:
        return tcp, st
    ps = pull.clone()
    for _ in range(6):                                                                            # spread: every point keeps its required pull
        sm = ps.clone(); sm[:, 1:-1] = 0.25 * ps[:, :-2] + 0.5 * ps[:, 1:-1] + 0.25 * ps[:, 2:]
        ps = torch.maximum(pull, sm)
    for _ in range(RR_SMOOTH):                                                                    # round the onset corners
        sm = ps.clone(); sm[:, 1:-1] = 0.25 * ps[:, :-2] + 0.5 * ps[:, 1:-1] + 0.25 * ps[:, 2:]; ps = sm
    ps = torch.where(fixed, torch.zeros_like(ps), torch.minimum(ps, cap))
    out = tcp.clone(); out[..., :2] = tcp[..., :2] + ps[..., None] * dirn
    st["pulled"] = int((ps > 1e-6).sum()); st["cm_sum"] = 100 * float(ps.sum()); st["cm_max"] = 100 * float(ps.max())
    return out, st


def resample_t(pts, M=MP):
    """polyline pts (n, m, 3) -> M points evenly spaced along its arc length."""
    import torch
    n, m, _ = pts.shape
    sl = torch.cat([torch.zeros(n, 1, device=pts.device), (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).cumsum(1)], 1)
    q = torch.linspace(0, 1, M, device=pts.device)[None] * sl[:, -1:]
    j = (torch.searchsorted(sl.contiguous(), q.contiguous(), right=True) - 1).clamp(0, m - 2)
    a0, a1 = sl.gather(1, j), sl.gather(1, j + 1)
    fr = ((q - a0) / (a1 - a0).clamp(min=1e-12)).clamp(0, 1)[..., None]
    P0 = pts.gather(1, j[..., None].expand(n, M, 3)); P1 = pts.gather(1, (j + 1)[..., None].expand(n, M, 3))
    return P0 + fr * (P1 - P0)


def rot_spline(res, u):
    """a zero-ended Catmull-Rom spline of orientation residuals: res (n, n_ctrl, 3) (roll, pitch, yaw; rad) at the control points
    placed at 1/(n_ctrl+1) ... n_ctrl/(n_ctrl+1), u (n,) in [0, 1] -> (n, 3); 0 at u = 0 and u = 1 (the endpoints keep the base pose)."""
    import torch
    n, nc, _ = res.shape
    z0 = torch.zeros(n, 1, 3, device=res.device)
    vals = torch.cat([z0, res, z0], 1); vp = torch.cat([vals[:, :1], vals, vals[:, -1:]], 1)
    h = 1.0 / (nc + 1); k = (u.clamp(0, 1) / h).long().clamp(0, nc); t = ((u.clamp(0, 1) - k.float() * h) / h)[:, None]
    ar = torch.arange(n, device=res.device)
    p0, p1, p2, p3 = vp[ar, k], vp[ar, k + 1], vp[ar, k + 2], vp[ar, k + 3]
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def q_tilt(rp):
    """roll about the hand x then pitch about the hand y: rp (..., 2) -> xyzw quaternion (..., 4)."""
    import torch
    r, p = rp[..., 0] / 2, rp[..., 1] / 2
    qx = torch.stack([torch.sin(r), torch.zeros_like(r), torch.zeros_like(r), torch.cos(r)], -1)
    qy = torch.stack([torch.zeros_like(p), torch.sin(p), torch.zeros_like(p), torch.cos(p)], -1)
    return FE.qmul(qx, qy)


def minjerk(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class ChainSim:
    """The Stage A stepper. Per gripper g (2, asynchronous): phase 0 idle (a decision is due), 1 carrying (attached), 2 settling (released;
    the hand retreats by the take's retreat dz), 3 done (said STOP), 4 moving (hand only: the approach before a take, or a free move).
    Every hand has a persistent pose (palm centre P, yaw Y, part offsets O in the hand frame); it is an obstacle wherever it is."""

    def __init__(self, fp, decide_settle_s=DECIDE_SETTLE_S):
        import torch
        assert FE.HANDS == 2, "FP_HANDS=2"
        self.fp, self.dev, self.N, self.K, self.dt = fp, fp.dev, fp.N, fp.K, fp.dt
        self.prof = {}
        N, dev = self.N, self.dev
        self.n_dec = int(round(decide_settle_s / self.dt))
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)      # noqa: E731
        self.z = z
        self.phase = z(N, 2, dt=torch.long)
        self.car = z(N, 2, dt=torch.long); self.t = z(N, 2, dt=torch.long); self.rel = z(N, 2, dt=torch.long); self.end = z(N, 2, dt=torch.long)
        self.rel_min = z(N, 2, dt=torch.long)
        self.path = z(N, 2, FE.NP_PATH, 3); self.sl = z(N, 2, FE.NP_PATH); self.L = z(N, 2); self.dur = z(N, 2); self.off = z(N, 2, 3)
        self.yaw0 = z(N, 2); self.dyaw = z(N, 2); self.gyaw = z(N, 2); self.relu = z(N, 2); self.ret = z(N, 2)
        self.HC = z(N, 2, 3, 3)                                       # hand part centres in the carried object's baked frame
        # the hand: palm centre (world), yaw (world), part offsets (hand frame, relative to the palm centre)
        self.hP = z(N, 2, 3); self.hY = z(N, 2); self.hO = z(N, 2, fp.NPART, 3); self.hPrel = z(N, 2, 3)
        # a hand move (approach / free): an MP-point route, its arc lengths, duration, start / end yaw and offsets, what follows
        self.hRP = z(N, 2, 2); self.mrot = z(N, 2, N_RCTRL, 3); self.crot = z(N, 2, N_RCTRL, 3)      # hand tilt; route / carry orientation edits
        self.mpath = z(N, 2, MP, 3); self.msl = z(N, 2, MP); self.mdur = z(N, 2); self.mY0 = z(N, 2); self.mY1 = z(N, 2)
        self.mO0 = z(N, 2, fp.NPART, 3); self.mO1 = z(N, 2, fp.NPART, 3); self.after = z(N, 2, dt=torch.long)
        self.hh_max = z(N); self.hh_steps = z(N, dt=torch.long)
        if SIL:                                                          # the executed carry trajectories, one buffer row per gripper
            self.sil_buf = z(N, 2, SIL_T, 3); self.sil_n = z(N, 2, dt=torch.long)
        # fingertips vs the table top (the hand is kinematic: nothing stops it sinking into the table): steps per env, a flag per take
        self.ft_steps = z(N, dt=torch.long); self.ft_ep = z(N, dt=torch.long); self.ft_take = z(N, 2, dt=torch.bool); self.n_take_ft = 0
        self.ft_abort = os.environ.get("FPC_FINGER_ABORT") == "1"
        from cleantables.rl.gripper import FL, PH
        self.part_half_z = torch.tensor([PH / 2] + [FL / 2] * (fp.NPART - 1), device=dev)[:fp.NPART]
        self.ev, self.ev_open, self.mev, self.mev_open = [], {}, [], {}
        self.takes, self.tk_open = [], {}                              # per take decision: its approach (mev), carry (ev), abort reason
        self.n_ab_env = np.zeros(N, int)                              # aborted takes in each env's current episode (not counted toward the carry cap)
        self.pending = {}                                             # (env, g) -> carry parameters waiting for the approach to end
        self.n_take = 0; self.n_take_abort = 0; self.n_take_ft = 0; self.ft_steps.zero_(); self.ft_take.zero_(); self.n_reflex = 0
        self.n_hand_obj_abort = 0; self.hpen_ep = torch.zeros_like(self.ft_ep); self.hov_ep = torch.zeros_like(self.ft_ep)
        self.step_i = 0
        self.HROW = fp.HIDX.reshape(N, 2, fp.NPART)
        self.arms = None                                              # ArmTracker when attached
        self.prev_hp = None
        self.GUP = torch.tensor([0.0, 0.0, FE.G], device=dev)
        self.tele = z(N, dt=torch.bool)                              # envs whose hands were teleported (reset): no kinematic velocity this step
        # the low-level controller's per-hand commands: path offset (m, table frame), yaw turn (rad), speed (x the plan's); the offset and turn
        # act through sin(pi u) of the move's progress (zero at its start and end); t advances by the speed (accumulated) in approaches / carries
        self.lo_off = torch.zeros(N, 2, 3, device=self.dev); self.lo_yaw = torch.zeros(N, 2, device=self.dev)
        self.lo_spd = torch.ones(N, 2, device=self.dev); self.tacc = torch.zeros(N, 2, device=self.dev)
        if JOINTS:                                                       # the controller's last joint residuals per hand
            self.lo_dq = torch.zeros(N, 2, 7, device=self.dev)
            self.lo_rp = torch.zeros(N, 2, 2, device=self.dev)       # the hand's roll / pitch (q_tilt) from FK(q)
        if FLUID_WAIT:                                                   # the hand's current move is a planner WAIT (set by run_continuous)
            self.is_wait = torch.zeros(N, 2, dtype=torch.bool, device=self.dev)
        if ARM_BOUNCE:                                                   # steps of bounce left / of cool-down left per hand
            self.b_left = torch.zeros(N, 2, dtype=torch.long, device=self.dev); self.b_cool = torch.zeros(N, 2, dtype=torch.long, device=self.dev)
        if PUSH_BOUNCE:                                                  # the fading push offset per hand + its last bounce steps
            self.b_push = torch.zeros(N, 2, 3, device=self.dev); self.b_times = torch.full((N, 2, BOUNCE_STUCK_N), -10 ** 9, dtype=torch.long, device=self.dev)
        self.n_pre = int(round(FE.PRE_S / self.dt))

    # ---------------------------------------------------------------------------------------------------------------------------------
    def home(self):
        """(N, 2, 3) the hands' start poses (world): 35 cm above the two ends of each table's long axis."""
        import torch
        if getattr(self, "_home", None) is not None:
            return self._home
        fp = self.fp
        if self.arms is not None:                                     # the Franka arms' home TCP poses
            P, Y = self.arms.home_palms(); self._home = P + fp.org[:, None]; self._home_yaw = Y
            return self._home
        T = fp.tpl["templates"]; out = torch.zeros(self.N, 2, 3, device=self.dev)
        for e in range(self.N):
            tb = T[fp.tpl_of_env[e]]["table"]; xs, ys = tb["x"], tb["y"]; cx, cy = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
            if xs[1] - xs[0] >= ys[1] - ys[0]:
                pts = [(xs[0] + 0.05, cy), (xs[1] - 0.05, cy)]
            else:
                pts = [(cx, ys[0] + 0.05), (cx, ys[1] - 0.05)]
            for g in (0, 1):
                out[e, g] = torch.tensor([pts[g][0], pts[g][1], 0.35], device=self.dev)
        self._home = out + fp.org[:, None]
        return self._home

    def reset(self, layout, pre_s=FE.PRE_S):
        import torch
        from cleantables.rl.gripper import boxes_g
        fp = self.fp
        layout = torch.as_tensor(layout, dtype=torch.float32, device=self.dev).clone()
        self.park = fp._reset_poses(layout)
        self.phase.zero_(); self.t.zero_(); self.ev, self.ev_open, self.mev, self.mev_open, self.pending = [], {}, [], {}, {}
        self.takes, self.tk_open = [], {}; self.n_ab_env[:] = 0
        self.n_take = 0; self.n_take_abort = 0; self.n_take_ft = 0; self.ft_steps.zero_(); self.ft_take.zero_(); self.n_reflex = 0
        self.n_hand_obj_abort = 0; self.hpen_ep = torch.zeros_like(self.ft_ep); self.hov_ep = torch.zeros_like(self.ft_ep)
        self.step_i = 0; self.hh_max.zero_(); self.hh_steps.zero_()
        bx = boxes_g(np.array([0.06, 0.06, 0.06]), 0.0)                   # an open hand's default part layout (a 6 cm object's grasp)
        O = torch.tensor(np.stack([b[0] - bx[0][0] for b in bx]), dtype=torch.float32, device=self.dev)
        self.hP[:] = self.home(); self.hY[:] = getattr(self, "_home_yaw", 0.0); self.hO[:] = O; self.hO_default = O
        if self.arms is not None:
            self.arms.reset(torch.arange(self.N, device=self.dev))
        self.prev_hp = None
        P0_, Y0_, O0_, RP0_ = self._hand_now(); self._write_hands(P0_, Y0_, O0_, RP0_, still=True)
        fp._step(int(round(pre_s / self.dt)))
        self.pose0 = self.poses()
        return self.pose0

    def poses(self):
        return self.fp._poses()

    def reset_envs(self, envs, layout):
        """(continuous runs) re-start envs (n,) at layout (n, K, 7) while every other env keeps running: objects re-posed (zero velocity),
        hands home, grippers in phase 5 (the start layout settles for PRE_S, then they decide)."""
        import torch
        fp, dev = self.fp, self.dev
        e = torch.as_tensor(envs, device=dev).long()
        if len(e) == 0:
            return
        if DR_EP:
            self._dr_mass(e)
        lay = torch.as_tensor(layout, dtype=torch.float32, device=dev)
        vm = fp.valid[e] & torch.isfinite(lay).all(-1)
        for k in range(self.K):
            ids = torch.nonzero(vm[:, k]).flatten()
            if len(ids):
                a_ = fp.env.scene[f"obj{k}"]; ee = e[ids]
                a_.write_root_pose_to_sim_index(root_pose=torch.cat([lay[ids, k, :3] + fp.org[ee], lay[ids, k, 3:]], -1), env_ids=ee)
                a_.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=dev), env_ids=ee)
        fp._inval()
        H = self.home()
        self.hP[e] = H[e]; self.hY[e] = self._home_yaw[e] if getattr(self, "_home_yaw", None) is not None else 0.0
        if FLUID_WAIT:
            self.is_wait[e] = False
        if ARM_BOUNCE:                                                   # a re-started env has no bounce in progress
            self.b_left[e] = 0; self.b_cool[e] = 0
        if PUSH_BOUNCE:
            self.b_push[e] = 0.0; self.b_times[e] = -10 ** 9
        self.hO[e] = self.hO_default; self.tele[e] = True
        if self.arms is not None:
            self.arms.reset(e)
        self.phase[e] = 5; self.t[e] = 0
        el = set(e.tolist())
        for key in [k_ for k_ in self.pending if k_[0] in el]:
            self.pending.pop(key)
        for key in [k_ for k_ in self.ev_open if k_[0] in el]:
            self.ev_open.pop(key)
        for key in [k_ for k_ in self.mev_open if k_[0] in el]:
            self.mev_open.pop(key)
        for key in [k_ for k_ in self.tk_open if k_[0] in el]:
            self.tk_open.pop(key)
        self.n_ab_env[e.cpu().numpy()] = 0

    def _dr_mass(self, e):
        """per-episode mass: every object of envs e gets its scene mass x a log-uniform factor in [0.5, 2] (inertia not rescaled,
        as the scene DR); the held-object drive's cap follows (fp.MV)."""
        import torch
        fp, wp = self.fp, self.fp.wp
        if not hasattr(self, "_mv0"):
            self._mv0 = fp.MV.clone(); self._ms0 = wp.to_torch(fp.OV.get_masses()).clone()
        rows = fp.IDX[e].reshape(-1); rows = rows[(rows >= 0)]
        fac = torch.exp(torch.empty(len(rows)).uniform_(math.log(0.5), math.log(2.0)))
        ms = self._ms0.clone().reshape(fp.OV.count, -1)
        ms[rows.cpu()] = ms[rows.cpu()] * fac[:, None]
        fp.OV.set_masses(wp.from_torch(ms.reshape(self._ms0.shape).contiguous()), wp.from_torch(rows.cpu().to(torch.int32).contiguous()))
        fp.MV[rows] = self._mv0[rows] * fac.to(fp.MV.device)

    def yaw_off(self, P):
        """(N, K) each object's yaw offset (the twins' convention: on top of the baked yaw) from poses P (N, K, 7)."""
        return FE.yaw_zyx(FE.qmul(P[..., 3:], FE.qyaw(-self.fp.yb)))

    # ---------------------------------------------------------------------------------------------------------------------------------
    def _grasp_hand(self, e, c, gy, P=None):
        """the hand pose for grasping object c (n,) of envs e at grasp yaw gy: palm centre (world), hand yaw, part offsets, HC (baked)."""
        import torch
        from cleantables.rl.gripper import boxes_g
        fp = self.fp
        P = self.poses() if P is None else P
        x = P[e, c, :3] + fp.org[e]; yw = FE.yaw_zyx(P[e, c, 3:])
        ext = fp.ext[e, c].detach().cpu().numpy(); gyc = torch.as_tensor(gy).detach().cpu().numpy()
        HC = torch.tensor(np.stack([[b[0] for b in boxes_g(xx, float(q))] for xx, q in zip(ext, gyc)]), dtype=torch.float32, device=self.dev)
        cz, sz = torch.cos(yw), torch.sin(yw)
        rot = lambda v, c_, s_: torch.stack([c_ * v[..., 0] - s_ * v[..., 1], s_ * v[..., 0] + c_ * v[..., 1], v[..., 2]], -1)   # noqa: E731
        Pp = x + rot(HC[:, 0], cz, sz)
        gyt = torch.as_tensor(gy, dtype=torch.float32, device=self.dev)
        O = rot(HC - HC[:, :1], torch.cos(-gyt)[:, None], torch.sin(-gyt)[:, None])
        return Pp, yw + gyt, O, HC

    def _start_move(self, e, g, P1, Y1, O1, after, route=True, resid=None, rot=None):
        """hand move of envs e (n,) gripper g to palm P1 (n, 3) world, yaw Y1, offsets O1: over-route (rise to clear every object near the
        straight line, cross, descend: generic geometry) if route else a straight line; min-jerk on arc length at the carry speed.
        resid (n, N_RCTRL, 3) (m): the actor's spline edit of that route (the polyline / line = the zero edit)."""
        import torch
        fp, dev = self.fp, self.dev
        if GATE:                                                         # the move's wall-time budget starts
            JI.gate_restart(self, e, g)
        if REACH:                                                        # the move's wall counter starts
            JI.reach_restart(self, e, g)
        if UNPIN:                                                        # the hand's displacement carries over into the new move
            lo_keep = (self.lo_off[e, g].clone(), self.lo_yaw[e, g].clone())
        if LOWPOL:                                                       # a new move starts with no low-level offset / turn
            self.lo_off[e, g] = 0.0; self.lo_yaw[e, g] = 0.0; self.tacc[e, g] = 0.0
        if UNPIN:
            self.lo_off[e, g] = lo_keep[0]; self.lo_yaw[e, g] = lo_keep[1]
        if ARM_BOUNCE:                                                   # a new move starts without a bounce
            self.b_left[e, g] = 0
        if PUSH_BOUNCE:
            self.b_push[e, g] = 0.0; self.b_times[e, g] = -10 ** 9
        if FLUID_WAIT:                                                   # any new move is not a wait (run_continuous marks waits after)
            self.is_wait[e, g] = False
        P0 = self.hP[e, g].clone()
        if UNPIN:                                                        # from the hand's plan pose (its displacement is the controller's)
            P0 = JI.unpin_start(self, e, g)
        if route:
            Pq = self.poses()                                                                        # object tops near the xy segment
            top = Pq[e, :, 2] + fp.ext[e, :, 2]; xy = Pq[e, :, :2] + fp.org[e, None, :2]
            a_, b_ = P0[:, None, :2], P1[:, None, :2]; ab = b_ - a_
            tt = (((xy - a_) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0, 1)
            d = (xy - (a_ + tt[..., None] * ab)).norm(dim=-1)
            near = (d < 0.12 + 0.5 * fp.ext[e, :, :2].norm(dim=-1)) & fp.valid[e] & torch.isfinite(top)
            ztop = torch.where(near, top + fp.org[e, None, 2], torch.full_like(top, -1e9)).amax(1)
            low = -torch.minimum(self.hO[e, g, :, 2].amin(-1), O1[..., 2].amin(-1)) + 0.03            # palm centre to the hand's lowest point
            zc = torch.maximum(torch.maximum(P0[:, 2], P1[:, 2]), ztop + low + 0.03)
            pts = torch.stack([P0, torch.cat([P0[:, :2], zc[:, None]], 1), torch.cat([P1[:, :2], zc[:, None]], 1), P1], 1)
        else:
            pts = torch.stack([P0, P0 + (P1 - P0) / 3, P0 + 2 * (P1 - P0) / 3, P1], 1)
        pts = resample_t(pts); pts0 = pts
        if resid is not None:
            pts = edit_path_t(pts, torch.as_tensor(np.asarray(resid), dtype=torch.float32, device=dev).reshape(len(e), -1, 3))
        if REACH_REPAIR and self.arms is not None:                       # route points this arm cannot reach: pulled toward its base
            pts = self._reach_route(e, g, pts, after)
        if getattr(self, "rec", None) is not None:
            o_ = fp.org[e][:, None]
            for i, ei in enumerate(e.tolist()):
                self.rec.routes.append(dict(step=self.step_i, env=ei, g=g, after=int(after), route=(pts[i] - o_[i]).cpu().numpy(), route0=(pts0[i] - o_[i]).cpu().numpy()))
        seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1); sl = torch.cat([torch.zeros(len(e), 1, device=dev), seg.cumsum(1)], 1)
        self.mrot[e, g] = 0.0 if rot is None else torch.as_tensor(np.asarray(rot), dtype=torch.float32, device=dev).reshape(len(e), -1, 3)
        self.mpath[e, g] = pts; self.msl[e, g] = sl; self.mdur[e, g] = (sl[:, -1] / FE.CARRY_SPEED).clamp(0.3, 3.0)
        d_ = torch.remainder(Y1 - self.hY[e, g] + math.pi, 2 * math.pi) - math.pi                    # the shorter turn
        self.mY0[e, g] = self.hY[e, g]; self.mY1[e, g] = self.hY[e, g] + d_
        if UNPIN:
            JI.unpin_yaw(self, e, g, Y1)
        self.mO0[e, g] = self.hO[e, g]; self.mO1[e, g] = O1; self.after[e, g] = after; self.t[e, g] = 0; self.phase[e, g] = 4
        Pn = self.poses()
        for ei in e.tolist():
            self.mev_open[(ei, g)] = len(self.mev)
            self.mev.append(dict(env=ei, g=g, kind="approach" if after == 1 else "move", s0=self.step_i, s1=None, P0=Pn[ei].clone(), P1=None))

    def _reach_route(self, e, g, pts, after):
        """a hand route pts (n, M, 3) (palm centres, world) of gripper g: its TCP points judged as the arm check judges them (mid-route,
        before route_u of the move's time, or a free move: position + a tilt; near the grasp: top-down), out-of-reach ones pulled toward the
        arm's base (reach_pull); start and end fixed."""
        import torch
        A = self.arms; n, M, _ = pts.shape
        dz = torch.tensor([0.0, 0.0, A.TCP_DZ], device=pts.device)
        org = self.fp.org[e][:, None]
        tcp = pts - org - dz
        sl = torch.cat([torch.zeros(n, 1, device=pts.device), (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).cumsum(1)], 1)
        s = sl / sl[:, -1:].clamp(min=1e-9); ru = A.route_u; s_r = 10 * ru ** 3 - 15 * ru ** 4 + 6 * ru ** 5    # minjerk(route_u): the arc share
        tilt = (s < s_r) | (int(after) == 0)
        fixed = torch.zeros(n, M, dtype=torch.bool, device=pts.device); fixed[:, 0] = True; fixed[:, -1] = True
        out = A.reach_pull(e, g, tcp, tilt, fixed, "route")
        res = pts.clone(); res[..., :2] = pts[..., :2] + (out[..., :2] - tcp[..., :2])          # only the xy pull (heights and fixed points unchanged)
        return res

    def _carry_tcp_dz(self, ee, c):
        """(n, 3) offset from a carried object's root to the TCP the arm needs (the palm above the object as gripper.py places it)."""
        import torch
        from cleantables.rl.gripper import FB, FC, FL, PH
        h = torch.nan_to_num(self.fp.ext[ee, c, 2])
        dz = torch.zeros(len(ee), 3, device=self.dev); dz[:, 2] = (h - FC).clamp(min=FB) + FL + PH / 2 - self.arms.TCP_DZ
        return dz

    def _wait_stat(self, bend_, wt_, s_b, u):
        """aggregate fluid-wait stats; one log line a minute."""
        b = self.__dict__.setdefault("wstat", dict(t=time.time(), n=0, ends=0, dsum=0.0, cl=0))
        b["n"] += int(wt_.sum()); end_ = wt_ & (u >= 0.999)
        b["ends"] += int(end_.sum()); b["dsum"] += 100 * float(bend_[end_].norm(dim=-1).sum())
        if s_b is not None:
            b["cl"] += int((wt_ & (s_b < 1)).sum())
        if time.time() - b["t"] >= 60.0:
            print(f"{time.strftime('%H:%M:%S')} FLUID WAIT (last {time.time() - b['t']:.0f} s): waiting hand-steps {b['n']}, wait ends {b['ends']}, "
                  f"mean displacement per wait {b['dsum'] / max(b['ends'], 1):.1f} cm, reach-clamped {100 * b['cl'] / max(b['n'], 1):.1f}%", flush=True)
            self.wstat = dict(t=time.time(), n=0, ends=0, dsum=0.0, cl=0)

    def _bend_stat(self, s_b, w_):
        """aggregate bend-clamp stats; one log line a minute."""
        act = w_ > 1e-6
        b = self.__dict__.setdefault("bstat", dict(t=time.time(), n=0, cl=0, z=0, ssum=0.0))
        b["n"] += int(act.sum()); clm = act & (s_b < 1); b["cl"] += int(clm.sum()); b["z"] += int((act & (s_b <= 0)).sum()); b["ssum"] += float(s_b[clm].sum())
        if time.time() - b["t"] >= 60.0:
            print(f"{time.strftime('%H:%M:%S')} BEND REACH (last {time.time() - b['t']:.0f} s): bent hand-steps {b['n']}, clamped {b['cl']} "
                  f"({100 * b['cl'] / max(b['n'], 1):.1f}%), mean scale when clamped {b['ssum'] / max(b['cl'], 1):.2f}, bend dropped (0) {b['z']}", flush=True)
            self.bstat = dict(t=time.time(), n=0, cl=0, z=0, ssum=0.0)

    def reach_repair_carry(self, e, g, a, paths):
        """(FPW_REACH_REPAIR) carry root paths (n, M, 3) (table frame) of objects a (n,) carried by gripper g of envs e: the TCP the carrying
        arm needs at each point (the palm above the object as gripper.py places it, TCP_DZ below the palm centre; top-down) pulled into reach
        (reach_pull, start / end fixed). -> paths (n, M, 3) numpy, changed (n,) bool numpy."""
        import torch
        from cleantables.rl.gripper import FB, FC, FL, PH
        A = self.arms; dev = self.dev
        e_t = torch.as_tensor(np.asarray(e), device=dev).long(); a_t = torch.as_tensor(np.asarray(a), device=dev).long()
        X = torch.as_tensor(np.asarray(paths), dtype=torch.float32, device=dev); n, M, _ = X.shape
        h = torch.nan_to_num(self.fp.ext[e_t, a_t, 2])
        dz = torch.zeros(n, 1, 3, device=dev); dz[:, 0, 2] = (h - FC).clamp(min=FB) + FL + PH / 2 - A.TCP_DZ
        fixed = torch.zeros(n, M, dtype=torch.bool, device=dev); fixed[:, 0] = True; fixed[:, -1] = True
        tcp = X + dz; out_t = A.reach_pull(e_t, int(g), tcp, torch.zeros_like(fixed), fixed, "carry")
        out = X.clone(); out[..., :2] = X[..., :2] + (out_t[..., :2] - tcp[..., :2])          # only the xy pull (heights and fixed points unchanged)
        return out.cpu().numpy(), ((out - X).abs().amax(-1).amax(-1) > 0).cpu().numpy()

    # ---------------------------------------------------------------------------------------------------------------------------------
    def take(self, envs, g, carried, path, dyaw, gyaw, rel_u, retreat=0.0, approach=True, app_resid=None, app_rot=None, carry_rot=None):
        """gripper g of envs (n,) takes object carried (n,): approach (hand move to the grasp pose) then the carry along the root path
        (n, 33, 3) (the offset to the object's actual position fades out), turning by dyaw, grasp yaw gyaw (baked frame), release at the
        min-jerk parameter rel_u in (0, 1] (1 = at the goal, settled hold), then the hand rises by retreat (m) during the settle."""
        import torch
        dev = self.dev
        e = torch.as_tensor(envs, device=dev).long(); n = len(e)
        if n == 0:
            return
        c = torch.as_tensor(carried, device=dev).long(); gy = torch.as_tensor(gyaw, dtype=torch.float32, device=dev)
        self.n_take += n
        if GRASP_FILT:                                                    # the hand opens 8 cm: a wider grasp fails at once (R_FAIL)
            from cleantables.rl.gripper import opening
            ext_ = self.fp.ext[e, c].cpu().numpy(); gyn = gy.cpu().numpy()
            wide = np.array([opening(ext_[i], float(gyn[i])) > MAX_OPEN for i in range(n)])
            if wide.any():
                for i in np.flatnonzero(wide):
                    ei = int(e[i]); self.n_take_abort += 1; self.n_ab_env[ei] += 1
                    self.takes.append(dict(env=ei, g=g, c=int(c[i]), step=self.step_i, mev=None, ev=None, abort="grasp_width", abort_step=self.step_i))
                keep = torch.as_tensor(~wide, device=dev)
                e, c, gy = e[keep], c[keep], gy[keep]
                keep_np = ~wide
                path = np.asarray(path)[keep_np]; dyaw = np.asarray(dyaw, np.float32).reshape(-1)[keep_np] if np.asarray(dyaw).size == len(keep_np) else dyaw
                rel_u = np.asarray(rel_u, np.float32).reshape(-1)[keep_np] if np.asarray(rel_u).size == len(keep_np) else rel_u
                if torch.is_tensor(retreat) and retreat.numel() == len(keep_np):
                    retreat = retreat[keep]
                if app_resid is not None:
                    app_resid = np.asarray(app_resid)[keep_np]
                n = len(e)
                if n == 0:
                    return
        self.n_take_ft += int(self.ft_take[e, g].sum()); self.ft_take[e, g] = False          # close the previous take's flag
        rt = torch.as_tensor(retreat, dtype=torch.float32, device=dev).expand(n) if not torch.is_tensor(retreat) else retreat.to(dev)
        pth = np.asarray(path, np.float32); dyl = np.asarray(dyaw, np.float32).reshape(-1); rul = np.asarray(rel_u, np.float32).reshape(-1)
        for i, ei in enumerate(e.tolist()):
            arm0 = None if self.arms is None else (float(self.arms.n_obj[ei] + self.arms.n_arm[ei] + self.arms.n_tab[ei]) * self.dt, float(self.arms.n_ik[ei]) * self.dt,
                                                   float(self.arms.yield_g[ei, g]) * self.arms.every * self.dt)   # this gripper's shield yields (s)
            self.tk_open[(ei, g)] = len(self.takes); self.takes.append(dict(env=ei, g=g, c=int(c[i]), step=self.step_i, mev=None, ev=None, abort=None, arm0=arm0))
            self.pending[(ei, g)] = dict(c=int(c[i]), path=pth[i], dyaw=float(dyl[i % len(dyl)]), gyaw=float(gy[i]), relu=float(rul[i % len(rul)]),
                                          ret=float(rt[i]), crot=None if carry_rot is None else np.asarray(carry_rot, np.float32)[i])
        if approach:
            P1, Y1, O1, _ = self._grasp_hand(e, c, gy)
            self._start_move(e, g, P1, Y1, O1, after=1, resid=app_resid, rot=app_rot)
            for ei in e.tolist():
                self.takes[self.tk_open[(ei, g)]]["mev"] = self.mev_open.get((ei, g))
        else:
            self._begin_carry(e, g)

    def move(self, envs, g, P1, Y1, route=False, resid=None, rot=None):
        """A free move of gripper g's hand (no object) to palm P1 (n, 3) (table frame) and yaw Y1 (n,) (world); the fingers keep their
        current opening. The policy's own target: no parking spots."""
        import torch
        e = torch.as_tensor(envs, device=self.dev).long()
        if len(e) == 0:
            return
        P1 = torch.as_tensor(P1, dtype=torch.float32, device=self.dev) + self.fp.org[e]
        self._start_move(e, g, P1, torch.as_tensor(Y1, dtype=torch.float32, device=self.dev), self.hO[e, g].clone(), after=0, route=route, resid=resid, rot=rot)

    def _begin_carry(self, e, g):
        import torch
        fp, dev, dt = self.fp, self.dev, self.dt
        rows = [self.pending.pop((ei, g)) for ei in e.tolist()]
        self.crot[e, g] = torch.as_tensor(np.stack([r_["crot"] if r_.get("crot") is not None else np.zeros((N_RCTRL, 3), np.float32) for r_ in rows]),
                                          dtype=torch.float32, device=self.dev)
        n = len(rows)
        c = torch.tensor([r["c"] for r in rows], device=dev); pth = torch.as_tensor(np.stack([r["path"] for r in rows]), device=dev)
        dy = torch.tensor([r["dyaw"] for r in rows], device=dev); gy = torch.tensor([r["gyaw"] for r in rows], device=dev)
        ru = torch.tensor([r["relu"] for r in rows], device=dev).clamp(0.05, 1.0); rt = torch.tensor([r["ret"] for r in rows], device=dev)
        P = self.poses()
        p_now = P[e, c, :3]; q_now = P[e, c, 3:]
        if SIL:                                                          # the executed trajectory starts here
            self.sil_buf[e, g, 0] = p_now; self.sil_n[e, g] = 1
        seg = (pth[:, 1:] - pth[:, :-1]).norm(dim=-1); sl = torch.cat([torch.zeros(n, 1, device=dev), seg.cumsum(1)], 1)
        L = sl[:, -1]; dur = torch.clamp(L / FE.CARRY_SPEED, FE.CARRY_MIN_S, FE.CARRY_MAX_S)
        tg = torch.linspace(0, 1, 2001, device=dev); mj = minjerk(tg)
        tau = tg[torch.searchsorted(mj.contiguous(), ru.contiguous()).clamp(max=2000)] * dur
        rel = torch.where(ru >= 0.999, torch.ceil((dur + HOLD_MAX_S) / dt - 1e-6), torch.ceil(tau / dt - 1e-6)).long().clamp(min=1)
        self.rel_min[e, g] = torch.where(ru >= 0.999, torch.ceil((dur + FE.HOLD_S) / dt - 1e-6).long(), rel)
        self.car[e, g] = c; self.t[e, g] = 0; self.rel[e, g] = rel; self.end[e, g] = rel + self.n_dec
        self.path[e, g] = pth; self.sl[e, g] = sl; self.L[e, g] = L; self.dur[e, g] = dur; self.off[e, g] = p_now - pth[:, 0]
        self.yaw0[e, g] = FE.yaw_zyx(FE.qmul(q_now, FE.qyaw(-fp.yb[e, c]))); self.dyaw[e, g] = dy; self.gyaw[e, g] = gy; self.relu[e, g] = ru; self.ret[e, g] = rt
        _, _, O, HC = self._grasp_hand(e, c, gy, P)
        self.HC[e, g] = HC; self.hO[e, g] = O
        self.phase[e, g] = 1
        if GATE:                                                         # the carry's wall-time budget starts
            JI.gate_restart(self, e, g)
        if REACH:                                                        # the carry's wall counter starts, not a drop
            JI.reach_restart(self, e, g)
        if LOWPOL:                                                       # the carry starts with no low-level offset / turn
            self.lo_off[e, g] = 0.0; self.lo_yaw[e, g] = 0.0; self.tacc[e, g] = 0.0
        if ARM_BOUNCE:                                                   # the carry starts without a bounce
            self.b_left[e, g] = 0
        if PUSH_BOUNCE:
            self.b_push[e, g] = 0.0; self.b_times[e, g] = -10 ** 9
        for i, (ei, ci) in enumerate(zip(e.tolist(), c.tolist())):
            k_ = self.tk_open.get((ei, g))
            if k_ is not None:
                self.takes[k_]["ev"] = len(self.ev)
            self.ev_open[(ei, g)] = len(self.ev)
            self.ev.append(dict(env=ei, g=g, carried=ci, s0=self.step_i, rel=None, end=None, P0=P[ei].clone(), Prel=None, Pend=None,
                                goal=pth[i, -1].clone(), path=pth[i].clone(), relu=float(ru[i]), gyaw=float(gy[i]), hh0=self.hh_steps[ei].clone(), hh1=None))

    def _sil_record(self, ee, gg, xr):
        """append root positions xr (n, 3) (table frame) to the executed trajectories of grippers (ee, gg) (unique pairs)."""
        ix = self.sil_n[ee, gg].clamp(max=SIL_T - 1)
        self.sil_buf[ee, gg, ix] = xr; self.sil_n[ee, gg] += 1

    def _sil_close(self, ei, g_, P, k_):
        """at a release: the carry's executed trajectory (start, every tick, the release pose) onto its ev record."""
        import torch
        n_ = int(self.sil_n[ei, g_]); m_ = max(1, min(n_, SIL_T))
        tr = torch.cat([self.sil_buf[ei, g_, :m_], P[ei, self.car[ei, g_], :3][None]], 0)
        self.ev[k_]["traj"] = tr.cpu().numpy(); self.ev[k_]["traj_over"] = n_ > SIL_T

    def drop(self, envs, g, stop=False):
        """Gripper g lets go now (a disturbance: the carry dropped mid-way, keeping its momentum). stop=True (an aborted carry):
        the hand halts before it opens, i.e. the held object is released at rest."""
        import torch
        e = torch.as_tensor(envs, device=self.dev).long()
        e = e[self.phase[e, g] == 1]
        self.rel[e, g] = self.t[e, g] + 1; self.end[e, g] = self.rel[e, g] + self.n_dec
        if REACH:                                                        # a drop is never held
            JI.reach_drop(self, e, g)
        if stop and len(e) and STOP_ABORT:
            self._zero_vel(e, self.car[e, g])

    def _zero_vel(self, e, obj):
        import torch
        fp = self.fp; wp = fp.wp
        rows = fp.IDX[e, obj]
        ve = wp.to_torch(fp.OV.get_velocities()).reshape(fp.OV.count, 6).clone()
        fp.OVel[rows] = 0.0
        fp.OV.set_velocities(wp.from_torch(fp.OVel), indices=wp.from_torch(rows.to(torch.int32).contiguous()))

    def knock(self, envs, obj, vel):
        """disturbance: object obj (n,) of envs (n,) gets the horizontal velocity vel (n, 2) (m/s), e.g. knocked by something."""
        import torch
        fp = self.fp; wp = fp.wp
        e = torch.as_tensor(envs, device=self.dev).long(); o = torch.as_tensor(obj, device=self.dev).long()
        rows = fp.IDX[e, o]
        ve = wp.to_torch(fp.OV.get_velocities()).reshape(fp.OV.count, 6).clone()
        ve[rows, :2] = torch.as_tensor(vel, dtype=torch.float32, device=self.dev); ve[rows, 2] = 0.5
        fp.OVel[rows] = ve[rows]
        fp.OV.set_velocities(wp.from_torch(fp.OVel), indices=wp.from_torch(rows.to(torch.int32).contiguous()))

    # ---------------------------------------------------------------------------------------------------------------------------------
    def _hand_now(self):
        """(P (N, 2, 3) palm centre world, Y (N, 2), O (N, 2, parts, 3)) for every hand that is not carrying (moving / retreating / still)."""
        import torch
        dt = self.dt
        P, Y, O = self.hP.clone(), self.hY.clone(), self.hO.clone(); RP = torch.zeros_like(self.hRP)
        mv = self.phase == 4
        if bool(mv.any()):
            ee, gg = torch.nonzero(mv, as_tuple=True)
            u = minjerk(torch.clamp(self.t[ee, gg].float() * dt / self.mdur[ee, gg], max=1.0))
            sl = self.msl[ee, gg]; s_ = u * sl[:, -1]
            j = (torch.searchsorted(sl.contiguous(), s_[:, None].contiguous(), right=True)[:, 0] - 1).clamp(0, MP - 2)
            a0, a1 = sl.gather(1, j[:, None])[:, 0], sl.gather(1, (j + 1)[:, None])[:, 0]
            fr = ((s_ - a0) / (a1 - a0).clamp(min=1e-12)).clamp(0, 1); ar = torch.arange(len(ee), device=self.dev)
            pt = self.mpath[ee, gg]
            P[ee, gg] = pt[ar, j] + fr[:, None] * (pt[ar, j + 1] - pt[ar, j])
            Y[ee, gg] = self.mY0[ee, gg] + u * (self.mY1[ee, gg] - self.mY0[ee, gg])
            O[ee, gg] = self.mO0[ee, gg] + u[:, None, None] * (self.mO1[ee, gg] - self.mO0[ee, gg])
            if ROT_EDIT:                                                  # the learned roll / pitch / yaw along the route
                er = rot_spline(self.mrot[ee, gg], u); Y[ee, gg] = Y[ee, gg] + er[:, 2]; RP[ee, gg] = er[:, :2]
            if LOWPOL:                                                    # the low level's offset / turn, zero at the move's start and end
                w_ = bend_env(u, sl[:, -1])                                # (flat when FPC_FLAT_BEND, else sin(pi u))
                if NO_HOME_BEND and self.arms is not None:                # a hand retreating home follows the plain path (no flailing)
                    w_ = w_ * (~self.arms.retract[ee, gg]).float()
                bend_ = w_[:, None] * self.lo_off[ee, gg]; wy_ = w_
                wt_ = self.is_wait[ee, gg] if FLUID_WAIT else None
                if FLUID_WAIT and bool(wt_.any()):                        # a waiting hand: min-jerk ramp 0 -> 1, so it ends displaced and stays
                    ramp_ = u.clamp(0, 1)                                 # there; at most WAIT_MOVE_M per axis per wait; the turn ramps x 0.5
                    bend_ = torch.where(wt_[:, None], ramp_[:, None] * self.lo_off[ee, gg] * (WAIT_MOVE_M / WAIT_R_OFF), bend_)
                    wy_ = torch.where(wt_, 0.5 * ramp_, w_)
                    if JOINTS:                                            # the joint residual's ramp: full displacement / turn
                        bend_ = torch.where(wt_[:, None], ramp_[:, None] * self.lo_off[ee, gg], bend_); wy_ = torch.where(wt_, ramp_, w_)
                if UNPIN:                                                 # 1 along the whole move (blending in from the hand's W at its start;
                    w_ = JI.unpin_w(self, ee, gg, u, sl[:, -1]); bend_ = w_[:, None] * self.lo_off[ee, gg]; wy_ = w_   # retracts, waits), the take's grasp descent excepted
                if PUSH_BOUNCE:                                           # the push-apart bounce, kept inside the reach clamp below
                    bend_ = bend_ + self.b_push[ee, gg]
                if BEND_REACH and self.arms is not None and not JOINTS:   # the bend kept inside the arm's reach (tilt rule as the arm check)
                    A_ = self.arms; ur_ = torch.clamp(self.t[ee, gg].float() * dt / self.mdur[ee, gg].clamp(min=1e-6), max=1.0)
                    tl_ = (self.after[ee, gg] == 0) | (ur_ < A_.route_u)
                    tcp_ = P[ee, gg] - self.fp.org[ee] - torch.tensor([0.0, 0.0, A_.TCP_DZ], device=self.dev)
                    s_b = bend_scale(A_.base[ee, gg], tcp_, bend_, tl_); bend_ = bend_ * s_b[:, None]; self._bend_stat(s_b, w_)
                else:
                    s_b = None
                if FLUID_WAIT and bool(wt_.any()):                        # a waiting palm never goes below min(its start height, the free-move floor)
                    oz_ = self.fp.org[ee, 2]; z0_ = P[ee, gg][:, 2]
                    fl_ = oz_ + torch.minimum(z0_ - oz_, torch.full_like(z0_, WAIT_PALM_Z0))
                    bz_ = torch.where(wt_, torch.maximum(bend_[:, 2], fl_ - z0_), bend_[:, 2])
                    bend_ = torch.cat([bend_[:, :2], bz_[:, None]], 1)
                    self._wait_stat(bend_, wt_, s_b, u)
                P[ee, gg] = P[ee, gg] + bend_; Y[ee, gg] = Y[ee, gg] + wy_ * self.lo_yaw[ee, gg]
                if JOINTS:                                                # the hand's roll / pitch from FK(q), env(u)-scaled as the offset
                    RP[ee, gg] = wy_[:, None] * self.lo_rp[ee, gg]
        st = (self.phase == 2) & (self.ret > 0)
        if bool(st.any()):
            ee, gg = torch.nonzero(st, as_tuple=True)
            u = minjerk(torch.clamp(self.t[ee, gg].float() * dt / 0.4, max=1.0))
            P[ee, gg] = self.hPrel[ee, gg] + torch.stack([torch.zeros_like(u), torch.zeros_like(u), u * self.ret[ee, gg]], -1)
        if UNPIN:                                                        # idle / done / pre / settling hands: plan pose + W lo_off
            P, Y, RP = JI.unpin_hands(self, P, Y, RP)
        return P, Y, O, RP

    def _write_hands(self, P, Y, O, RP=None, still=False):
        import torch
        fp, wp = self.fp, self.fp.wp
        if RP is None:
            RP = torch.zeros(self.N, 2, 2, device=self.dev)
        self.hRP = RP
        qf = FE.qmul(FE.qyaw(Y), q_tilt(RP))                                                     # (N, 2, 4) the hand's full orientation
        cz, sz = torch.cos(Y)[..., None], torch.sin(Y)[..., None]
        hp = P[:, :, None] + torch.stack([cz * O[..., 0] - sz * O[..., 1], sz * O[..., 0] + cz * O[..., 1], O[..., 2]], -1)   # (N, 2, parts, 3)
        tl_ = RP.abs().amax(-1) > 0                                                              # tilted hands only: the full rotation
        if (ROT_EDIT or JOINTS) and bool(tl_.any()):                                             # (untilted rows keep the yaw-only formula)
            Rf = FE.qmat(qf.reshape(-1, 4)).reshape(self.N, 2, 3, 3)
            hp = torch.where(tl_[..., None, None], P[:, :, None] + (Rf[:, :, None] @ O[..., None])[..., 0], hp)
        hq = qf[:, :, None].expand(-1, -1, fp.NPART, -1)
        hv = torch.zeros_like(hp) if (still or self.prev_hp is None) else (hp - self.prev_hp) / self.dt
        hv = torch.where(self.tele[:, None, None, None], torch.zeros_like(hv), hv); tele = self.tele.clone(); self.tele.zero_()
        first = self.prev_hp is None
        self.prev_hp = hp
        rows = self.HROW.reshape(-1)
        if FE.HAND_DYN:
            self._drive_hands(hp, hq, hv, tele | first | still, rows)
        else:
            fp.HT[rows, :3] = hp.reshape(-1, 3); fp.HT[rows, 3:] = hq.reshape(-1, 4); fp.HVel[rows, :3] = hv.reshape(-1, 3); fp.HVel[rows, 3:] = 0.0
            ra = wp.from_torch(rows.to(torch.int32).contiguous())
            fp.HV.set_transforms(wp.from_torch(fp.HT), indices=ra); fp.HV.set_velocities(wp.from_torch(fp.HVel), indices=ra)
        fp._inval()
        # the two hands' overlap (geometric; kinematic bodies never touch in PhysX): palm centres closer than two palm radii
        from cleantables.rl.gripper import PL, PW, PH
        r = 0.5 * math.sqrt(PL ** 2 + PW ** 2 + PH ** 2) * 0.6
        hh = (2 * r - (P[:, 0] - P[:, 1]).norm(dim=-1)).clamp(min=0)
        self.hh_max = torch.maximum(self.hh_max, hh); self.hh_steps += (hh > 0).long()
        low = (hp[..., 2] - self.part_half_z).amin(-1) - self.fp.org[:, None, 2]                 # (N, 2) the hand's lowest point above the table top
        under = low < -0.002
        self.ft_steps += under.any(1).long(); self.ft_ep += under.any(1).long()
        in_take = (self.phase == 1) | ((self.phase == 4) & (self.after == 1))
        self.ft_take |= under & in_take
        if self.ft_abort and bool((under & in_take).any()):                  # fingers in the table fail the take, as an IK failure does
            for ei, g_ in torch.nonzero(under & in_take).tolist():
                self.n_take_abort += 1; self.n_ab_env[ei] += 1
                k_ = self.tk_open.get((ei, g_))
                if k_ is not None and self.takes[k_]["abort"] is None:
                    self.takes[k_]["abort"] = "finger_carry" if int(self.phase[ei, g_]) == 1 else "finger_approach"; self.takes[k_]["abort_step"] = self.step_i
                if int(self.phase[ei, g_]) == 1:
                    self.drop([ei], g_, stop=True)
                else:                                                    # the approach is cancelled and the hand leaves the table (home), as
                    self.pending.pop((ei, g_), None); self.mev_open.pop((ei, g_), None)     # after an IK abort (not left inside the table)
                    if self.arms is not None:
                        self.arms._retract([ei], g_)
                    else:
                        self.phase[ei, g_] = 0
        self._mk("hand_write_finger")
        if REFLEX:
            self._reflex(hp)
        if HAND_PEN:
            self._hand_pen(hp, Y)
        return hh

    def _mk(self, name):
        """FP_PROF=1: CUDA-synced split timer: the time since the last mark is charged to `name` (None = only set the mark)."""
        if not PROF:
            return
        import torch
        torch.cuda.synchronize(); t = time.perf_counter()
        if name is not None and getattr(self, "_mk_t", None) is not None:
            self.prof[name] = self.prof.get(name, 0.0) + t - self._mk_t
            if name == "physx_step":
                self.prof["_steps"] = self.prof.get("_steps", 0) + 1
        self._mk_t = t

    def _pending_idx(self, g):
        """the pending approaches of hand g as index tensors (envs, target objects), or None."""
        import torch
        it = [(ei, pd["c"]) for (ei, g_), pd in self.pending.items() if g_ == g]
        if not it:
            return None
        e_, c_ = zip(*it)
        return torch.as_tensor(e_, dtype=torch.long, device=self.dev), torch.as_tensor(c_, dtype=torch.long, device=self.dev)

    def _hand_pen(self, hp, Y):
        """Hand-object penetration. With FP_HAND=ghost the hand boxes do not collide with objects (a kinematic hand cannot fling
        anything), so penetration is measured geometrically every step: hand part box vs object box (oriented, xy SAT + z interval). Excluded: the held object (carry /
        release), objects riding on it, and the approach's target in the approach's final 20% (the grasp itself). Depth > HAND_PEN_ABORT
        (1 cm) aborts like an IK / finger-table abort (carry: stop, release in place; approach / free move: cancelled, the arm takes the hand
        home over the objects) and counts a penalised event; any smaller overlap costs per step."""
        import torch
        from cleantables.rl.gripper import FL, FT, FW, PH, PL, PW
        fp = self.fp; N = self.N; dev = self.dev
        ret = self.arms.retract if self.arms is not None else torch.zeros_like(self.phase, dtype=torch.bool)
        active = ((self.phase == 1) | (self.phase == 2) | (self.phase == 4)) & ~ret
        if not bool(active.any()):
            return
        P = self.poses(); K = P.shape[1]
        oc = P[..., :2] + fp.org[:, None, :2]; oz0 = P[..., 2] + fp.org[:, None, 2]; ext = fp.ext
        if PEN_SDF and FAST2 and self._sdf_covers():                     # every valid slot has a mesh field: the box depth below is
            dep = self._pen_sdf(hp, Y, P, None)                           # never used (invalid slots are excluded further down)
        else:
            dep = self._pen_box(hp, Y, P, oc, oz0, ext)
            if PEN_SDF:                                                   # the objects' own mesh signed distance fields (the
                dep = self._pen_sdf(hp, Y, P, dep)                        # planner's geo_cost fields: hollow containers are hollow), boxes only where
                                                                          # an object has no mesh field
        ok_o = fp.valid & torch.isfinite(P).all(-1) & (P[..., 2] > -OFF_Z)
        excl = ~ok_o[:, None].expand(N, 2, K).clone()
        ar = torch.arange(N, device=dev)
        for g in (0, 1):
            held = (self.phase[:, g] == 1) | (self.phase[:, g] == 2); c = self.car[:, g]
            excl[ar, g, c] |= held
            hc = oc[ar, c]; ride = ((oc - hc[:, None]).norm(dim=-1) < 0.5 * ext[ar, c, :2].amax(-1)[:, None]) & (oz0 > oz0[ar, c][:, None] + 0.005)
            excl[:, g] |= held[:, None] & ride
            u_ = (self.t[:, g].float() * self.dt / self.mdur[:, g].clamp(min=1e-6))
            if FAST2:
                pt_ = self._pending_idx(g)                                # (one indexed write, no per-approach host sync)
                if pt_ is not None:
                    pe_, pc_ = pt_
                    if not PEN_TGT_ALL:                                   # the take's target: the whole approach
                        m_ = u_[pe_] >= 0.8; pe_, pc_ = pe_[m_], pc_[m_]
                    excl[pe_, g, pc_] = True
            else:
                for (ei, g_), pd in self.pending.items():
                    if g_ == g and (PEN_TGT_ALL or float(u_[ei]) >= 0.8):
                        excl[ei, g, pd["c"]] = True
        dep = torch.where(excl | ~active[..., None], torch.full_like(dep, -1.0), dep)
        dmax = dep.amax(-1)                                                                                    # (N, 2)
        self.hov_ep += ((dmax > 0) & (dmax <= HAND_PEN_ABORT)).any(1).long()
        ab = (dmax > HAND_PEN_ABORT) & ((self.phase == 1) | (self.phase == 4))   # aborts only while carrying / moving (a release only costs)
        self.hov_ep += ((dmax > HAND_PEN_ABORT) & (self.phase == 2)).any(1).long()
        if not bool(ab.any()):
            return
        for ei, g in torch.nonzero(ab).tolist():
            ph_ = int(self.phase[ei, g]); self.n_hand_obj_abort += 1; self.hpen_ep[ei] += 1
            if os.environ.get("FPC_PEN_DIAG") == "1":
                k_ = int(dep[ei, g].argmax()); pd_ = self.pending.get((ei, g))
                self.__dict__.setdefault("pen_diag", []).append(dict(env=ei, g=g, phase=ph_, after=int(self.after[ei, g]), k=k_,
                    target=(pd_ is not None and pd_["c"] == k_) or (ph_ == 1 and int(self.car[ei, g]) == k_), depth_cm=float(dep[ei, g, k_]) * 100,
                    u=float(self.t[ei, g]) * self.dt / max(float(self.mdur[ei, g]), 1e-6) if ph_ == 4 else None,
                    palm_z=float(hp[ei, g, 0, 2] - fp.org[ei, 2]), obj_z=float(P[ei, k_, 2]), held=int(self.car[ei, g]) if ph_ == 1 else -1,
                    part=int(self._pen_arg[ei, g, k_]) // 15 if hasattr(self, "_pen_arg") else -1,
                    obj_top=float(P[ei, k_, 2] + fp.ext[ei, k_, 2]), tcp_z=float(hp[ei, g, 0, 2] - fp.org[ei, 2]) - float(os.environ.get("FPW_PALM_Z0", "0.09")),
                    xy_to_obj=float((hp[ei, g, 0, :2] - fp.org[ei, :2] - P[ei, k_, :2]).norm()),
                    tgt=(pd_["c"] if pd_ is not None else (int(self.car[ei, g]) if ph_ == 1 else -1)),
                    xy_to_tgt=(float((hp[ei, g, 0, :2] - fp.org[ei, :2] - P[ei, pd_["c"], :2]).norm()) if pd_ is not None else None),
                    obj_name=str(fp.tpl["templates"][int(fp.tpl_of_env[ei])]["objects"][k_].get("id", k_)),
                    tgt_name=(str(fp.tpl["templates"][int(fp.tpl_of_env[ei])]["objects"][pd_["c"]].get("id", "")) if pd_ is not None else
                              (str(fp.tpl["templates"][int(fp.tpl_of_env[ei])]["objects"][int(self.car[ei, g])].get("id", "")) if ph_ == 1 else ""))))
            k_ = self.tk_open.get((ei, g))
            if ph_ == 1 or (ph_ == 4 and int(self.after[ei, g]) == 1):
                self.n_take_abort += 1; self.n_ab_env[ei] += 1
                if k_ is not None and self.takes[k_]["abort"] is None:
                    self.takes[k_]["abort"] = "hand_obj_carry" if ph_ == 1 else "hand_obj_approach"; self.takes[k_]["abort_step"] = self.step_i
            if ph_ == 1:
                self.drop([ei], g, stop=True)
            elif ph_ == 4:
                self.pending.pop((ei, g), None); self.mev_open.pop((ei, g), None)
                if self.arms is not None:
                    self.arms._retract([ei], g)
                else:
                    self.phase[ei, g] = 0

    def _sdf_covers(self):
        """True when every valid slot of every env has a mesh SDF field (then the box fallback is never needed)."""
        import torch
        if not hasattr(self, "_sdf_cov"):
            if not hasattr(self, "_sdf_buf"):
                self._build_sdf()
            u = self._sdf_u[torch.as_tensor(self.fp.tpl_of_env, device=self.dev)][:, :self.fp.K]
            self._sdf_cov = bool(((u >= 0) | ~self.fp.valid[:, :u.shape[1]]).all()) and u.shape[1] == self.fp.K
        return self._sdf_cov

    def _pen_box(self, hp, Y, P, oc, oz0, ext):
        """hand part box vs object box penetration depth (N, 2, K): oriented xy SAT + z interval, the deepest part."""
        import torch
        from cleantables.rl.gripper import FL, FT, FW, PH, PL, PW
        dev = self.dev
        oyaw = FE.yaw_zyx(P[..., 3:]); ou = torch.stack([torch.cos(oyaw), torch.sin(oyaw)], -1); ov = torch.stack([-ou[..., 1], ou[..., 0]], -1)
        half = torch.tensor([[PL, PW, PH], [FT, FW, FL], [FT, FW, FL]], device=dev)[:hp.shape[2]] / 2
        hu = torch.stack([torch.cos(Y), torch.sin(Y)], -1); hv_ = torch.stack([-hu[..., 1], hu[..., 0]], -1)          # (N, 2, 2)
        d_ = oc[:, None, None] - hp[:, :, :, None, :2]                                                     # (N, 2, parts, K, 2)
        U1 = hu[:, :, None, None].expand_as(d_); V1 = hv_[:, :, None, None].expand_as(d_)
        U2 = ou[:, None, None].expand_as(d_); V2 = ov[:, None, None].expand_as(d_)
        h1 = half[None, None, :, None]; h2 = (ext[:, None, None, :, :2] / 2)
        dep = None
        for a in (U1, V1, U2, V2):
            r1 = h1[..., 0] * (U1 * a).sum(-1).abs() + h1[..., 1] * (V1 * a).sum(-1).abs()
            r2 = h2[..., 0] * (U2 * a).sum(-1).abs() + h2[..., 1] * (V2 * a).sum(-1).abs()
            ov_ = r1 + r2 - (d_ * a).sum(-1).abs()
            dep = ov_ if dep is None else torch.minimum(dep, ov_)
        zt = hp[..., 2][..., None]; hz = half[None, None, :, None, 2]
        oz = torch.minimum(zt + hz, (oz0 + ext[..., 2])[:, None, None]) - torch.maximum(zt - hz, oz0[:, None, None])
        return torch.minimum(dep, oz).amax(2)                                                          # (N, 2, K) deepest part (boxes)

    def _drive_hands(self, hp, hq, hv, tele, rows):
        """Force-limited hands: every hand part is a dynamic body (gravity off) pulled towards its
        commanded pose by a PD force / torque, capped so the whole hand pushes with at most HAND_F_CAP (20 N: libfranka's example collision
        thresholds, setCollisionBehavior Cartesian force 20 N); a blocked hand stops instead of flinging what it touches. Teleports (resets)
        still set the pose directly."""
        import torch
        fp, wp = self.fp, self.fp.wp
        T = wp.to_torch(fp.HV.get_transforms()).reshape(fp.HV.count, 7)[rows]; V = wp.to_torch(fp.HV.get_velocities()).reshape(fp.HV.count, 6)[rows]
        x, q, v, w = T[:, :3], T[:, 3:], V[:, :3], V[:, 3:]
        xs, qs, vs = hp.reshape(-1, 3), hq.reshape(-1, 4), hv.reshape(-1, 3)
        npart = hp.shape[2]
        if not hasattr(self, "_hm"):
            from cleantables.rl.gripper import box_sizes
            m = torch.tensor(FE.HAND_MASS[:npart], device=self.dev); sz = torch.tensor(box_sizes()[:npart], device=self.dev, dtype=torch.float32)
            I = m[:, None] / 12 * torch.stack([sz[:, 1] ** 2 + sz[:, 2] ** 2, sz[:, 0] ** 2 + sz[:, 2] ** 2, sz[:, 0] ** 2 + sz[:, 1] ** 2], -1)
            share = m / m.sum()                                                                     # the cap shared by mass: palm 15 N, fingers 2.5 N
            self._hm = m.repeat(self.N * 2); self._hI = I.repeat(self.N * 2, 1); self._hcap = (HAND_F_CAP * share).repeat(self.N * 2)
            self._htcap = (HAND_T_CAP * share).repeat(self.N * 2)
            self._HF = torch.zeros(fp.HV.count, 3, device=self.dev); self._HTq = torch.zeros(fp.HV.count, 3, device=self.dev)
        F = self._hm[:, None] * (HAND_KP * (xs - x) + HAND_KD * (vs - v))
        F = F * (self._hcap / F.norm(dim=-1).clamp(min=1e-9)).clamp(max=1.0)[:, None]
        qinv = torch.cat([-q[:, :3], q[:, 3:]], -1)
        rv = FE.qrotvec(FE.qmul(qs, qinv))
        Tq = self._hI * (HAND_KP * rv - HAND_KD * w)
        Tq = Tq * (self._htcap / Tq.norm(dim=-1).clamp(min=1e-9)).clamp(max=1.0)[:, None]
        self._HF[rows] = F; self._HTq[rows] = Tq
        fp.HV.apply_forces_and_torques_at_position(force_data=wp.from_torch(self._HF.reshape(-1)), torque_data=wp.from_torch(self._HTq.reshape(-1)),
                                                    position_data=None, indices=wp.from_torch(rows.to(torch.int32).contiguous()), is_global=True)
        self.hand_err = (xs - x).norm(dim=-1).reshape(self.N, 2, npart)                             # tracking error (m): a blocked hand lags
        st_ = self.__dict__.setdefault("herr_st", {})                       # palm tracking error while carrying (1) / moving (4)
        for ph_ in (1, 4):
            m_ = (self.phase == ph_)
            if bool(m_.any()):
                pe = self.hand_err[..., 0][m_]
                d_ = st_.setdefault(ph_, dict(n=0, s=0.0, gt1=0, gt3=0, mx=0.0))
                d_["n"] += pe.numel(); d_["s"] += float(pe.sum()); d_["gt1"] += int((pe > 0.01).sum()); d_["gt3"] += int((pe > 0.03).sum())
                d_["mx"] = max(d_["mx"], float(pe.max()))
        self.hp_act = x.reshape(self.N, 2, npart, 3)
        if bool(tele.any()):                                                                        # resets: teleport those hands
            tr = tele[:, None, None].expand(self.N, 2, npart).reshape(-1)
            r_ = rows[tr]
            fp.HT[r_, :3] = xs[tr]; fp.HT[r_, 3:] = qs[tr]; fp.HVel[r_] = 0.0
            ra = wp.from_torch(r_.to(torch.int32).contiguous())
            fp.HV.set_transforms(wp.from_torch(fp.HT), indices=ra); fp.HV.set_velocities(wp.from_torch(fp.HVel), indices=ra)

    def _pen_sdf(self, hp, Y, P, dep_box):
        """penetration depth (N, 2, K) of each hand into each object from the object's mesh SDF (geo_cost.MeshCost.field, WM_SDF_H grid):
        45 points per hand (each part's 8 corners, 6 face centres, centre), posed into the object's frame (root + full world yaw); depth =
        -min SDF over the points. Objects without a field keep the box estimate."""
        import torch
        fp = self.fp; dev = self.dev; N = self.N
        if not hasattr(self, "_sdf_buf"):
            self._build_sdf()
        from cleantables.rl.gripper import FL, FT, FW, PH, PL, PW
        if not hasattr(self, "_hpts"):
            half = torch.tensor([[PL, PW, PH], [FT, FW, FL], [FT, FW, FL]], device=dev)[:hp.shape[2]] / 2
            pat = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)] + [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]],
                               dtype=torch.float32, device=dev)
            self._hpts = pat[None] * half[:, None]                                                           # (parts, 15, 3) hand frame
        c, s_ = torch.cos(Y)[..., None, None], torch.sin(Y)[..., None, None]
        loc = self._hpts[None, None]                                                                         # (1, 1, parts, 15, 3)
        pw = torch.stack([c * loc[..., 0] - s_ * loc[..., 1], s_ * loc[..., 0] + c * loc[..., 1], loc[..., 2].expand_as(c * loc[..., 0])], -1)
        if ROT_EDIT or JOINTS:                                            # tilted hands: the parts' points at the full orientation
            tl_ = self.hRP.abs().amax(-1) > 0
            if bool(tl_.any()):
                Rf = FE.qmat(FE.qmul(FE.qyaw(Y), q_tilt(self.hRP)).reshape(-1, 4)).reshape(N, 2, 3, 3)
                pw = torch.where(tl_[..., None, None, None], (Rf[:, :, None, None] @ loc.expand(N, 2, -1, -1, -1)[..., None])[..., 0], pw)
        pw = (pw + hp[..., None, :]).reshape(N, 2, -1, 3)                                                   # (N, 2, M, 3) world
        root = P[..., :3] + fp.org[:, None]; oy = FE.yaw_zyx(P[..., 3:])                                     # (N, K)
        d = pw[:, :, :, None] - root[:, None, None]                                                          # (N, 2, M, K, 3)
        co, so = torch.cos(-oy)[:, None, None], torch.sin(-oy)[:, None, None]
        q = torch.stack([co * d[..., 0] - so * d[..., 1], so * d[..., 0] + co * d[..., 1], d[..., 2]], -1)
        u = self._sdf_u[torch.as_tensor(fp.tpl_of_env, device=dev)][:, None, None, :P.shape[1]]            # (N, 1, 1, K) field index (-1 none)
        uu = u.clamp(min=0)
        lo, dims, off = self._sdf_lo[uu], self._sdf_dims[uu], self._sdf_off[uu]
        ix = torch.floor((q - lo) / self._sdf_h).long()
        inb = (ix >= 0).all(-1) & (ix < dims).all(-1) & (u >= 0)
        ixc = torch.minimum(ix.clamp(min=0), dims - 1)
        flat = off + (ixc[..., 0] * dims[..., 1] + ixc[..., 1]) * dims[..., 2] + ixc[..., 2]
        v = torch.where(inb, self._sdf_buf[flat].float(), torch.full_like(flat, 1e3, dtype=torch.float32))
        dep_sdf = (-v).amax(2)                                                                               # (N, 2, K)
        if os.environ.get("FPC_PEN_DIAG") == "1":                         # diagnostic: the deepest hand point per object; its part = index // 15
            self._pen_arg = (-v).argmax(2)
        has = (self._sdf_u[torch.as_tensor(fp.tpl_of_env, device=dev)][:, :P.shape[1]] >= 0)[:, None]
        return dep_sdf if dep_box is None else torch.where(has, dep_sdf, dep_box)

    def fast_repair(self, e, a, paths, dyaw, margin=0.02, iters=3):
        """Push the WM's carry paths (n, M, 3) root points, table frame, out of every other object by the objects' mesh SDFs (the same
        fields as the hand penetration check), batched: the carried object = its box corners + centre at mid height + the palm above it, posed
        along the path (yaw from its current yaw + u * dyaw); a point closer than `margin` to an object is moved out along that object's SDF
        gradient by the shortfall; iters passes + a light smoothing; the start and end stay fixed. Exempt as in worldmodel.physics.route_opt: the start segment
        while still near (it rests among its neighbours) and the set-down segment (within one own height of the goal height and one own
        length of it in xy: touching the support is allowed). The same rule for every object. -> repaired paths (n, M, 3), mean move (m)."""
        import torch
        fp = self.fp; dev = self.dev
        if not hasattr(self, "_sdf_buf"):
            self._build_sdf()
        e = torch.as_tensor(np.asarray(e), device=dev).long(); a = torch.as_tensor(np.asarray(a), device=dev).long()
        X = torch.as_tensor(np.asarray(paths), dtype=torch.float32, device=dev).clone(); n, M, _ = X.shape
        P = self.poses()[e]                                                                                  # (n, K, 7) table frame
        K = P.shape[1]; ext = fp.ext[e]                                                                      # (n, K, 3)
        ea = ext[torch.arange(n, device=dev), a]                                                             # (n, 3) carried extents
        hx, hy, hz = ea[:, 0] / 2, ea[:, 1] / 2, ea[:, 2]
        sgn = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float32, device=dev)
        loc = torch.cat([torch.stack([sgn[None, :, 0] * hx[:, None], sgn[None, :, 1] * hy[:, None], torch.zeros(n, 4, device=dev) + 0.1 * hz[:, None]], -1),
                         torch.stack([sgn[None, :, 0] * hx[:, None], sgn[None, :, 1] * hy[:, None], torch.zeros(n, 4, device=dev) + 0.9 * hz[:, None]], -1),
                         torch.stack([torch.zeros(n), torch.zeros(n), 0.5 * hz.cpu()], -1).to(dev)[:, None],
                         torch.stack([torch.zeros(n), torch.zeros(n), (hz + 0.04).cpu()], -1).to(dev)[:, None]], 1)   # (n, 10, 3) object frame (root at bottom)
        y0 = FE.yaw_zyx(P[torch.arange(n, device=dev), a, 3:]); dy = torch.as_tensor(np.asarray(dyaw, np.float32), device=dev).reshape(-1)
        dy = dy if dy.numel() == n else torch.zeros(n, device=dev)
        root = P[..., :3]; oy = FE.yaw_zyx(P[..., 3:])                                                     # other objects (n, K)
        u_idx = self._sdf_u[torch.as_tensor(np.asarray(fp.tpl_of_env), device=dev)[e]][:, :K]                # (n, K) field index
        other = (u_idx >= 0) & fp.valid[e][:, :K]; other[torch.arange(n, device=dev), a] = False
        uu = u_idx.clamp(min=0); lo, dims, off = self._sdf_lo[uu], self._sdf_dims[uu], self._sdf_off[uu]   # (n, K, 3)
        h = self._sdf_h; goal = X[:, -1]; start = X[:, 0]

        def sdf(q):                                                                                          # q (n, M, Q, K, 3) object frames -> (n, M, Q, K)
            ix = torch.floor((q - lo[:, None, None]) / h).long()
            inb = (ix >= 0).all(-1) & (ix < dims[:, None, None]).all(-1)
            ixc = torch.minimum(ix.clamp(min=0), dims[:, None, None] - 1)
            flat = off[:, None, None] + (ixc[..., 0] * dims[:, None, None, :, 1] + ixc[..., 1]) * dims[:, None, None, :, 2] + ixc[..., 2]
            return torch.where(inb, self._sdf_buf[flat].float(), torch.full_like(flat, 1e3, dtype=torch.float32))

        moved = torch.zeros(n, device=dev)
        for _ in range(iters):
            sl = torch.cat([torch.zeros(n, 1, device=dev), (X[:, 1:] - X[:, :-1]).norm(dim=-1).cumsum(1)], 1)
            u = sl / sl[:, -1:].clamp(min=1e-6)                                                              # (n, M) arc fraction
            yw = y0[:, None] + u * dy[:, None]
            c, s_ = torch.cos(yw)[..., None], torch.sin(yw)[..., None]
            pw = torch.stack([c * loc[:, None, :, 0] - s_ * loc[:, None, :, 1], s_ * loc[:, None, :, 0] + c * loc[:, None, :, 1], loc[:, None, :, 2].expand(n, M, -1)], -1) + X[:, :, None]   # (n, M, 10, 3)
            d = pw[:, :, :, None] - root[:, None, None]                                                      # (n, M, 10, K, 3)
            co, so = torch.cos(-oy)[:, None, None], torch.sin(-oy)[:, None, None]
            q = torch.stack([co * d[..., 0] - so * d[..., 1], so * d[..., 0] + co * d[..., 1], d[..., 2]], -1)
            v = sdf(q); v = torch.where(other[:, None, None], v, torch.full_like(v, 1e3))
            gq = torch.stack([(sdf(q + torch.tensor(dd, device=dev) * h) - v) / h for dd in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], -1)   # forward diff, object frame
            co2, so2 = torch.cos(oy)[:, None, None], torch.sin(oy)[:, None, None]
            gw = torch.stack([co2 * gq[..., 0] - so2 * gq[..., 1], so2 * gq[..., 0] + co2 * gq[..., 1], gq[..., 2]], -1)   # world (table) frame
            gw = gw / gw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            short = (margin - v).clamp(min=0) * (v < 1e2)                                                    # (n, M, 10, K)
            with torch.no_grad():                                                                            # exemptions (as route_opt)
                near_s = (v < margin).any(-1).any(-1)                                                        # (n, M)
                ex_s = torch.cummin(near_s.int(), dim=1).values.bool()
                fin = ((X[..., 2] - goal[:, None, 2]).abs() <= hz[:, None]) & ((X[..., :2] - goal[:, None, :2]).norm(dim=-1) <= 2 * torch.maximum(hx, hy)[:, None])
                ex_f = torch.cummin(fin.int().flip(1), dim=1).values.bool().flip(1)
            push = (short[..., None] * gw).sum(3).amax(2) + (short[..., None] * gw).sum(3).amin(2)          # (n, M, 3): the largest push per axis
            push = torch.where((ex_s | ex_f)[..., None], torch.zeros_like(push), push)
            push[:, 0] = 0; push[:, -1] = 0
            X = X + push
            X[..., 2] = torch.maximum(X[..., 2], torch.zeros_like(X[..., 2]))                                 # the table is solid
            Xs = X.clone(); Xs[:, 1:-1] = 0.25 * X[:, :-2] + 0.5 * X[:, 1:-1] + 0.25 * X[:, 2:]
            moved = moved + push.norm(dim=-1).mean(1); X = Xs
        X[:, 0] = start; X[:, -1] = goal
        return X.cpu().numpy(), moved.cpu().numpy()

    def _build_sdf(self):
        import torch
        from cleantables.worldmodel.physics.geo_cost import SDF_H, MeshCost
        fp = self.fp; dev = self.dev; T = fp.tpl["templates"]; K = max(t["n"] for t in T)
        mc = MeshCost(); key2u, bufs, lo, dims, off = {}, [], [], [], []; n_vox = 0
        U = np.full((len(T), K), -1, np.int64)
        for ti, t in enumerate(T):
            for k, o in enumerate(t["objects"]):
                key = (str(o["usd"]), float(o.get("scale") or 1.0))
                if key not in key2u:
                    u_ = str(o["usd"]); root_ = str(Path(__file__).resolve().parents[2]) + "/"
                    o2 = dict(o, usd=u_[len(root_):] if u_.startswith(root_) else u_)    # the occupancy cache is keyed by the scene's relative path
                    F = mc.field(o2)
                    if F is None:
                        key2u[key] = -1; continue
                    sdf, _, l, _, _ = F
                    key2u[key] = len(bufs); bufs.append(sdf.reshape(-1)); lo.append(l); dims.append(sdf.shape); off.append(n_vox); n_vox += sdf.size
                U[ti, k] = key2u[key]
        self._sdf_buf = torch.as_tensor(np.concatenate(bufs), dtype=torch.float16, device=dev)
        self._sdf_lo = torch.as_tensor(np.stack(lo), dtype=torch.float32, device=dev)
        self._sdf_dims = torch.as_tensor(np.array(dims), dtype=torch.long, device=dev)
        self._sdf_off = torch.as_tensor(np.array(off), dtype=torch.long, device=dev)
        self._sdf_u = torch.as_tensor(U, device=dev); self._sdf_h = float(SDF_H)
        print(f"[hand_pen] mesh SDFs: {len(bufs)} unique objects, {n_vox / 1e6:.1f} M voxels ({SDF_H * 100:.1f} cm); "
              f"{int((U < 0).sum() - sum(K - t['n'] for t in T))} slots without a field (box fallback)", flush=True)

    def _reflex(self, hp):
        """The real Franka's collision reflex, as a stand-in for its force limits (the fast sim's hands are kinematic: they follow
        the commanded path whatever they hit, and the sim has no contact sensors): a non-target object within REFLEX_GAP of a hand part or of the
        held object, moving faster than REFLEX_V (and not moving with the held object), stops that hand at once: a carry is aborted as with STOP_ABORT
        (released at rest), an approach / free move is cancelled; the arm then takes the hand home over the objects. Logged as abort 'reflex_*'."""
        import torch
        fp = self.fp; N = self.N
        ret = self.arms.retract if self.arms is not None else torch.zeros_like(self.phase, dtype=torch.bool)
        active = (self.phase == 1) | ((self.phase == 4) & ~ret)
        if not bool(active.any()):
            return
        ve = fp.wp.to_torch(fp.OV.get_velocities()).reshape(fp.OV.count, 6)[fp.IDX.clamp(min=0)]     # (N, K, 6)
        v = ve[..., :3]; spd = v.norm(dim=-1)
        P = self.poses()
        ctr = P[..., :3] + fp.org[:, None]
        ctr = torch.cat([ctr[..., :2], (ctr[..., 2] + 0.5 * fp.ext[..., 2])[..., None]], -1)
        r_o = 0.5 * fp.ext[..., :2].norm(dim=-1)
        fast = (spd > REFLEX_V) & fp.valid & torch.isfinite(ctr).all(-1) & (P[..., 2] > -OFF_Z)
        if not bool(fast.any()):
            return
        r_p = torch.tensor([0.106, 0.03, 0.03], device=self.dev)[:hp.shape[2]]
        ar = torch.arange(N, device=self.dev)
        for g in (0, 1):
            act = active[:, g]
            if not bool(act.any()):
                continue
            d_h = ((torch.nan_to_num(ctr, nan=1e3)[:, :, None] - hp[:, g, None]).norm(dim=-1) - r_o[..., None] - r_p).amin(-1)   # (N, K)
            held = self.phase[:, g] == 1; c = self.car[:, g]
            ch = ctr[ar, c]; d_c = (torch.nan_to_num(ctr, nan=1e3) - ch[:, None]).norm(dim=-1) - r_o - r_o[ar, c][:, None]
            near = (d_h < REFLEX_GAP) | (held[:, None] & (d_c < REFLEX_GAP))
            excl = torch.zeros_like(near)
            excl[ar, c] = held                                                                            # the held object itself
            excl |= held[:, None] & ((v - v[ar, c][:, None]).norm(dim=-1) < 0.1)                           # riding on / moving with it
            oc = self.car[:, 1 - g]; oh = (self.phase[:, 1 - g] == 1) | (self.phase[:, 1 - g] == 2)
            excl[ar, oc] |= oh                                                                             # the other gripper's object
            for (ei, g_), pd in self.pending.items():                                                    # the object this hand approaches
                if g_ == g:
                    excl[ei, pd["c"]] = True
            trig = act & (fast & near & ~excl).any(1)
            if not bool(trig.any()):
                continue
            for ei in torch.nonzero(trig).flatten().tolist():
                ph_ = int(self.phase[ei, g]); self.n_reflex += 1
                k_ = self.tk_open.get((ei, g))
                if ph_ == 1 or (ph_ == 4 and int(self.after[ei, g]) == 1):
                    self.n_take_abort += 1; self.n_ab_env[ei] += 1
                    if k_ is not None and self.takes[k_]["abort"] is None:
                        self.takes[k_]["abort"] = "reflex_carry" if ph_ == 1 else "reflex_approach"; self.takes[k_]["abort_step"] = self.step_i
                if ph_ == 1:
                    self.drop([ei], g, stop=True)
                else:
                    self.pending.pop((ei, g), None); self.mev_open.pop((ei, g), None)
                    if self.arms is not None:
                        self.arms._retract([ei], g)
                    else:
                        self.phase[ei, g] = 0

    def step(self):
        """one physics step: carrying grippers drive their objects (sim_env's capped implicit drive) with the hand attached; moving hands
        follow their move; releases, retreats, settles and move ends advance; poses are logged at release / settle end / move end."""
        import torch
        fp, dev, dt, wp, N = self.fp, self.dev, self.dt, self.fp.wp, self.N
        act = self.phase == 1
        self._mk("between_steps")                                          # FP_PROF: the step's event tail + the run loop (ticks, decisions)
        P_, Y_, O_, RP_ = self._hand_now()
        self._mk("hand_now")
        holding = act & (self.t < self.rel)
        if bool(holding.any()):
            ee, gg = torch.nonzero(holding, as_tuple=True)
            c = self.car[ee, gg]; ci = fp.IDX[ee, c]
            tr_ = wp.to_torch(fp.OV.get_transforms()).reshape(fp.OV.count, 7); ve_ = wp.to_torch(fp.OV.get_velocities()).reshape(fp.OV.count, 6)
            x_, q_, v_, w_ = tr_[ci, :3], tr_[ci, 3:7], ve_[ci, :3], ve_[ci, 3:6]
            if SIL and self.step_i % TICK == 0:                          # the held objects' root positions, once per decision tick
                self._sil_record(ee, gg, x_ - fp.org[ee])
            tt = self.t[ee, gg].float() * dt; dur = self.dur[ee, gg]
            def tgt(tq):
                u = minjerk(torch.clamp(tq / dur, max=1.0)); s_ = u * self.L[ee, gg]
                sl = self.sl[ee, gg]; j = (torch.searchsorted(sl.contiguous(), s_[:, None].contiguous(), right=True)[:, 0] - 1).clamp(0, FE.NP_PATH - 2)
                a0, a1 = sl.gather(1, j[:, None])[:, 0], sl.gather(1, (j + 1)[:, None])[:, 0]
                fr = ((s_ - a0) / (a1 - a0).clamp(min=1e-12)).clamp(0, 1)
                pth = self.path[ee, gg]; ar = torch.arange(len(ee), device=dev)
                X = pth[ar, j] + fr[:, None] * (pth[ar, j + 1] - pth[ar, j]) + (1 - u)[:, None] * self.off[ee, gg]
                if LOWPOL:                                                # the low level's offset / turn of the carried object, zero at both ends
                    w_ = bend_env(u, self.L[ee, gg])                      # (flat when FPC_FLAT_BEND, else sin(pi u))
                    bend_ = w_[:, None] * self.lo_off[ee, gg]
                    if PUSH_BOUNCE:                                       # the push-apart bounce of the carrying hand
                        bend_ = bend_ + self.b_push[ee, gg]
                    if BEND_REACH and self.arms is not None and not JOINTS:   # carry: the palm-above-object TCP (reach_repair_carry), top-down
                        s_b = bend_scale(self.arms.base[ee, gg], X + self._carry_tcp_dz(ee, c), bend_, torch.zeros_like(u, dtype=torch.bool))
                        bend_ = bend_ * s_b[:, None]; self._bend_stat(s_b, w_)
                    return X + bend_, self.yaw0[ee, gg] + u * self.dyaw[ee, gg] + w_ * self.lo_yaw[ee, gg]
                return X, self.yaw0[ee, gg] + u * self.dyaw[ee, gg]
            X0, Y0 = tgt(tt); X1, Y1 = tgt(tt + dt)
            Vt = (X1 - X0) / dt; Wt = (Y1 - Y0) / dt
            if LOWPOL:                                                    # the drive's feed-forward follows the speed command
                Vt = Vt * self.lo_spd[ee, gg][:, None]; Wt = Wt * self.lo_spd[ee, gg]
            yb = fp.yb[ee, c]; Q = FE.qmul(FE.qyaw(Y0), FE.qyaw(yb))
            if ROT_EDIT:                                                  # the learned roll / pitch / yaw along the carry: the held object
                uc = minjerk(torch.clamp(tt / dur, max=1.0)); ec = rot_spline(self.crot[ee, gg], uc)   # turns with the hand about the grasp point
                Yh = Y0 + self.gyaw[ee, gg]
                D = FE.qmul(FE.qmul(FE.qyaw(Yh + ec[:, 2]), q_tilt(ec[:, :2])), FE.qyaw(-Yh))     # the hand's extra rotation (world)
                HC0b = self.HC[ee, gg, 0]; cy, sy = torch.cos(Y0), torch.sin(Y0)
                hpt = X0 + torch.stack([cy * HC0b[:, 0] - sy * HC0b[:, 1], sy * HC0b[:, 0] + cy * HC0b[:, 1], HC0b[:, 2]], -1)   # the grasp point
                ed_ = (ec.abs().amax(-1) > 0)[:, None]                    # rows with a non-zero edit (zero rows keep the plain pose)
                X0 = torch.where(ed_, hpt + (FE.qmat(D) @ (X0 - hpt)[..., None])[..., 0], X0)
                Q = torch.where(ed_, FE.qmul(D, Q), Q)
                RP_[ee, gg] = ec[:, :2]
            if JOINTS:                                                    # the held object turns with the hand's roll / pitch (grasp point)
                X0, Q, RP_[ee, gg], ecj = JI.carry_rotate(self, ee, gg, tt, X0, Y0, Q)
            X0w = X0 + fp.org[ee]
            MVs, IVs = fp.MV[ci], fp.IV[ci]
            LINg = dict(FE.LIN_D, effort=torch.clamp(FE.EFFORT_C * MVs * FE.G, min=FE.EFFORT_FLOOR)[:, None])
            GUP = self.GUP

            def drive(cur_v, tgt_x, tgt_v, m_, gains, up=None):
                m_ = m_[:, None]
                vn = (m_ * cur_v + dt * (gains["stiffness"] * tgt_x + gains["damping"] * tgt_v)) / (m_ + dt * gains["damping"] + dt * dt * gains["stiffness"])
                sp = vn.norm(dim=-1, keepdim=True); vn = vn * torch.clamp(gains["vel"] / sp.clamp(min=1e-9), max=1.0)
                Fd = m_ * (vn - cur_v) / dt + (0.0 if up is None else m_ * up)
                fn = Fd.norm(dim=-1, keepdim=True); Fd = Fd * torch.clamp(gains["effort"] / fn.clamp(min=1e-9), max=1.0)
                return cur_v + dt * Fd / m_
            vlin = drive(v_, X0w - x_, Vt, MVs, LINg, up=GUP)
            qinv = torch.cat([-q_[:, :3], q_[:, 3:]], -1)
            Vang = torch.zeros(len(ee), 3, device=dev); Vang[:, 2] = Wt
            wang = drive(w_, FE.qrotvec(FE.qmul(Q, qinv)), Vang, IVs, FE.ANG_D)
            fp.OVel[ci, :3] = vlin; fp.OVel[ci, 3:] = wang
            gl = self.path[ee, gg, -1] + fp.org[ee]                       # settled at the goal (after the minimum hold): release now
            ok_ = (self.t[ee, gg] >= self.rel_min[ee, gg]) & ((x_ - gl).norm(dim=-1) < HOLD_TOL_M) & (v_.norm(dim=-1) < HOLD_VTOL) & (self.relu[ee, gg] >= 0.999)
            if bool(ok_.any()):
                self.rel[ee[ok_], gg[ok_]] = self.t[ee[ok_], gg[ok_]] + 1; self.end[ee[ok_], gg[ok_]] = self.rel[ee[ok_], gg[ok_]] + self.n_dec
            fp.OV.set_velocities(wp.from_torch(fp.OVel), indices=wp.from_torch(ci.to(torch.int32).contiguous()))
            yw_ = FE.yaw_zyx(q_); cz, sz = torch.cos(yw_), torch.sin(yw_)
            HC0 = self.HC[ee, gg, 0]
            P_[ee, gg] = x_ + torch.stack([cz * HC0[:, 0] - sz * HC0[:, 1], sz * HC0[:, 0] + cz * HC0[:, 1], HC0[:, 2]], -1)
            if ROT_EDIT:                                                  # the grasp-point offset turns with the commanded tilt only
                ywh = yw_ + self.gyaw[ee, gg]                             # (using the object's full actual quaternion would also apply its baked / physics
                Dt = FE.qmat(FE.qmul(FE.qmul(FE.qyaw(ywh), q_tilt(ec[:, :2])), FE.qyaw(-ywh)))   # tilt, so zero edits would not be a no-op)
                P_[ee, gg] = torch.where((ec[:, :2].abs().amax(-1) > 0)[:, None], x_ + (Dt @ (P_[ee, gg] - x_)[..., None])[..., 0], P_[ee, gg])
            if JOINTS:                                                    # the hand on the tilted object: the commanded tilt about its root
                P_[ee, gg] = JI.carry_hand(self, ee, gg, x_, yw_, ecj, P_[ee, gg])
            Y_[ee, gg] = yw_ + self.gyaw[ee, gg]
        self._mk("carry_drive")
        self._write_hands(P_, Y_, O_, RP_)                                                            # (hand-hand overlap: per env, hh_steps)
        self._mk("hand_pen")
        self.hP, self.hY, self.hO = P_, Y_, O_
        fp._step()
        self._mk("physx_step")
        self.step_i += 1
        if self.arms is not None and self.step_i % self.arms.every == 0:
            self.arms.check()
        self._mk("arm_check")
        live = (self.phase == 1) | (self.phase == 2) | (self.phase == 4) | (self.phase == 5)
        mvg_ = (self.phase == 1) | (self.phase == 4)
        if ARM_BOUNCE and not PUSH_BOUNCE:                               # a bouncing hand goes back one step along its move instead
            back_, self.t, self.b_left, self.b_cool = bounce_advance(self.phase, self.t, self.rel, self.b_left, self.b_cool)
        else:
            back_ = torch.zeros_like(mvg_)
            if PUSH_BOUNCE:                                              # no rewind: the cool-down counts down, the push fades
                self.b_cool = (self.b_cool - 1).clamp(min=0); self.b_push = self.b_push * math.exp(-self.dt / BOUNCE_TAU_S)
        if LOWPOL:                                                       # approaches / free moves / carries advance at the low level's speed
            sp_ = mvg_ & ~back_
            if GATE:                                                     # a hand > FPL_GATE_CLOCK from its plan point: its clock holds (no speed
                tacc0_ = self.tacc; open_ = JI.gate_open(self)           # accumulated); a due release still fires on time
            self.tacc = torch.where(sp_, self.tacc + self.lo_spd, self.tacc)
            if GATE:
                self.tacc = torch.where(sp_ & ~open_, tacc0_, self.tacc)
            inc = torch.floor(self.tacc + 1e-4); self.tacc = self.tacc - inc            # (+1e-4: speed 1 computed as 0.99999994 still advances 1 / step)
            inc = torch.where((self.phase == 1) & (self.rel <= self.t + 1), inc.clamp(min=1), inc)   # a due release / abort-drop fires on time
            self.t[sp_] += inc[sp_].long(); oth = live & ~mvg_; self.t[oth] += 1
        else:
            self.t[live & ~back_] += 1
        pre_done = (self.phase == 5) & (self.t >= self.n_pre)                                     # auto-reset envs: the start layout has settled
        self.phase[pre_done] = 0
        if REACH:                                                        # a planned release waits for the hand (within FPL_REACH_DONE)
            JI.reach_hold_release(self)
        rel_now = act & (self.t >= self.rel)
        end_now = (self.phase == 2) & (self.t >= self.end)
        mv_end = (self.phase == 4) & (self.t.float() * dt >= self.mdur)
        if REACH:                                                        # a take's grasp waits for the hand (within FPL_REACH_DONE)
            mv_end = mv_end & JI.reach_take_ok(self)
        if bool((rel_now | end_now | mv_end).any()):                                              # one host sync per step
            P = self.poses()
            for ei, g_ in torch.nonzero(rel_now).tolist():
                k_ = self.ev_open.get((ei, g_))
                if k_ is not None:
                    self.ev[k_]["rel"] = self.step_i; self.ev[k_]["Prel"] = P[ei].clone(); self.ev[k_]["hh1"] = self.hh_steps[ei].clone()
                    if SIL:
                        self._sil_close(ei, g_, P, k_)
            for ei, g_ in torch.nonzero(end_now).tolist():
                k_ = self.ev_open.pop((ei, g_), None)
                if k_ is not None:
                    self.ev[k_]["end"] = self.step_i; self.ev[k_]["Pend"] = P[ei].clone()
            if bool(rel_now.any()):
                if REL_STOP:                                              # a release before the goal (u < 0.9) halts first: no toss
                    early = rel_now & (self.relu < 0.9)
                    for ei, g_ in torch.nonzero(early).tolist():
                        self._zero_vel(torch.tensor([ei], device=self.dev), self.car[ei:ei + 1, g_])
                self.hPrel[rel_now] = self.hP[rel_now]; self.t[rel_now] = self.t[rel_now]   # the retreat starts from the release pose
            self.phase[rel_now] = 2; self.phase[end_now] = 0
            if bool(mv_end.any()):
                for ei, g_ in torch.nonzero(mv_end).tolist():
                    k_ = self.mev_open.pop((ei, g_), None)
                    if k_ is not None:
                        self.mev[k_]["s1"] = self.step_i; self.mev[k_]["P1"] = P[ei].clone()
                carry_next = mv_end & (self.after == 1)
                self.phase[mv_end & (self.after == 0)] = 0
                self.phase[mv_end & (self.after == 3)] = 3                   # a stopped gripper's retraction ends stopped
                for g_ in (0, 1):
                    e_ = torch.nonzero(carry_next[:, g_]).flatten()
                    if len(e_):
                        self._begin_carry(e_, g_)


def edit_path(route, resid, n_ctrl=5):
    """route (n, 33, 3) root path, resid (n, n_ctrl, 3) displacements (m) at the control points placed
    at arc-length fractions 1/(n_ctrl+1) ... n_ctrl/(n_ctrl+1); the endpoints stay fixed; the displacement field is a Catmull-Rom spline
    through (0, resid..., 0) over the arc length, added to the route -> (n, 33, 3)."""
    route = np.asarray(route, np.float64); resid = np.asarray(resid, np.float64)
    n, NP_, _ = route.shape
    seg = np.linalg.norm(np.diff(route, axis=1), axis=-1); s = np.concatenate([np.zeros((n, 1)), seg.cumsum(1)], 1)
    s = s / np.maximum(s[:, -1:], 1e-9)
    knots = np.linspace(0, 1, n_ctrl + 2)
    vals = np.concatenate([np.zeros((n, 1, 3)), resid, np.zeros((n, 1, 3))], 1)                     # (n, n_ctrl + 2, 3)
    vp = np.concatenate([vals[:, :1], vals, vals[:, -1:]], 1)                                         # padded for Catmull-Rom
    h = knots[1] - knots[0]
    k = np.clip((s / h).astype(int), 0, n_ctrl)                                                       # segment index (n, 33)
    u = (s - knots[k]) / h
    ar = np.arange(n)[:, None]
    p0, p1, p2, p3 = vp[ar, k], vp[ar, k + 1], vp[ar, k + 2], vp[ar, k + 3]
    u_ = u[..., None]
    disp = 0.5 * (2 * p1 + (-p0 + p2) * u_ + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u_ ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u_ ** 3)
    out = route + disp
    out[:, 0], out[:, -1] = route[:, 0], route[:, -1]
    return out.astype(np.float32)


def path_shape(path, prior):
    """per path: lift (cm, max z - start z), flat (lift < 2 cm), mean deviation from the prior (cm)."""
    path = np.asarray(path); prior = np.asarray(prior)
    lift = 100 * (path[..., 2].max(-1) - path[..., 0, 2])
    return dict(lift_cm=lift, flat=lift < 2, dev_cm=100 * np.linalg.norm(path - prior, axis=-1).mean(-1))


# =====================================================================================================================================
class ArmTracker:
    """the two Franka arms behind the hands, kinematic, on the env's clock: every ARM_TICK steps
    a warm-started batched IK of each hand's TCP (top-down, yaw free), then sphere clearances: arm vs objects (the object the gripper holds
    or is approaching left out), arm vs the other arm, arm vs the table top (fingers exempt). Per env: steps with an IK failure / any
    overlap (penalties; arm contact never pushes anything). Bases: fastpath/franka_bases_clean.json (twin table frame)."""

    TCP_DZ = 0.08                                                     # palm centre (gripper.py) -> Franka TCP between the fingertips

    def __init__(self, CS, names, every=12, path=None):
        import torch
        from cleantables.robots import franka_kinematics as K
        self.K, self.CS, self.every = K, CS, int(every)
        fp = CS.fp; dev = fp.dev; N = fp.N
        path = path or os.environ.get("FPC_BASES") or (W / "fastpath" / "franka_bases_clean.json")    # FPC_BASES: alternative arm bases file
        self.bases_file = str(path)
        Bj = json.load(open(path))["scenes"]
        bs = np.array([Bj[names[t]]["bases"] for t in fp.tpl_of_env], np.float32)              # (N, 2, 4) x y z yaw
        self.base = K.base_T(torch.as_tensor(bs[..., 0]), torch.as_tensor(bs[..., 1]), torch.as_tensor(bs[..., 2]), torch.as_tensor(bs[..., 3]),
                             device=dev)                                                         # (N, 2, 4, 4) table frame
        self.binv = torch.linalg.inv(self.base)
        self.home_tcp = torch.as_tensor(np.array([Bj[names[t]]["home_tcp"] for t in fp.tpl_of_env], np.float32), device=dev)   # (N, 2, 7)
        self.q0 = torch.tensor(K.Q_HOME, device=dev)
        self.q = self.q0.expand(N, 2, 7).clone()
        T = fp.tpl["templates"]
        self.box = torch.tensor([[T[t]["table"]["x"][0], T[t]["table"]["x"][1], T[t]["table"]["y"][0], T[t]["table"]["y"][1]] for t in fp.tpl_of_env], device=dev)
        z = lambda: torch.zeros(N, dtype=torch.long, device=dev)                                 # noqa: E731
        self.n_ik, self.n_obj, self.n_arm, self.n_tab = z(), z(), z(), z()
        self.n_near = torch.zeros_like(self.n_arm, dtype=torch.float32)  # the closeness integral (steps x 0..1)
        self.n_touch = z()                                               # steps the two arms actually overlap (not the shield's stops)
        if PUSH_CONTACT:                                                 # steps since the robot last touched each object (large = never)
            self.obj_touch_age = torch.full((N, fp.K), 10 ** 6, dtype=torch.long, device=dev)
        self.yield_g = torch.zeros(self.n_arm.shape[0], 2, device=self.n_arm.device)  # shield yields per gripper (a leader pause counts 0.5)
        self.fail_streak = torch.zeros(N, 2, dtype=torch.long, device=dev)
        self.retract = torch.zeros(N, 2, dtype=torch.bool, device=dev)   # a hand going back to its arm's home after an unreachable pose
        self.n_retract = 0; self.n_move_abort = 0
        # the arm must hold the hand top-down only near the grasp / during the carry; mid-route
        # (an approach before ROUTE_U of its length, every free move) a tilted TCP up to ROUTE_TILT is fine
        self.route_u = float(os.environ.get("FPC_ROUTE_U", "0.8")); self.route_tilt = math.radians(float(os.environ.get("FPC_ROUTE_TILT", "45")))
        self.abort_after = int(os.environ.get("FPC_IK_ABORT", "2"))      # consecutive IK failures (checks, 0.1 s each) that abort a take
        self.t_s = 0.0
        if JOINTS:                                                       # the joint-space controller's per-arm state
            JI.tracker_init(self)

    def reach_pull(self, e, g, tcp, tilt, fixed, kind):
        """reach_pull() for arm g (int or (n,)) of envs e (n,) with this tracker's bases (closed form: no IK, no random numbers);
        an aggregated log line every 60 s."""
        import torch
        t0 = time.time()
        e = torch.as_tensor(e, device=tcp.device).long()
        gg = g.long() if torch.is_tensor(g) else torch.full_like(e, int(g))
        out, st = reach_pull(self.base[e, gg], tcp, tilt, fixed)
        rr = self.__dict__.setdefault("rr", dict(t_last=time.time(), carry=0, route=0, pts=0, chk=0, fail=0, pulled=0, cm_sum=0.0, cm_max=0.0, unreach=0, sec=0.0))
        rr[kind] += len(e); rr["sec"] += time.time() - t0
        for k_ in ("pts", "chk", "fail", "pulled", "cm_sum", "unreach"):
            rr[k_] += st[k_]
        rr["cm_max"] = max(rr["cm_max"], st["cm_max"])
        if time.time() - rr["t_last"] >= 60.0:
            print(f"{time.strftime('%H:%M:%S')} REACH REPAIR (last {time.time() - rr['t_last']:.0f} s): carry paths {rr['carry']}, routes {rr['route']}; "
                  f"points {rr['pts']}, IK-checked {rr['chk']}, out of reach {rr['fail']} ({100 * rr['fail'] / max(rr['pts'], 1):.1f}%), pulled {rr['pulled']} "
                  f"(mean {rr['cm_sum'] / max(rr['pulled'], 1):.1f} cm, max {rr['cm_max']:.0f} cm), still out of reach at the cap {rr['unreach']}; {rr['sec']:.1f} s", flush=True)
            self.rr = dict(t_last=time.time(), carry=0, route=0, pts=0, chk=0, fail=0, pulled=0, cm_sum=0.0, cm_max=0.0, unreach=0, sec=0.0)
        return out

    def reach(self, envs, g, xyz, iters=40, seeds=4, tol=0.01, kind="pt"):
        """(n, K) bool: can arm g of envs (n,) put its TCP top-down (yaw free) 2 cm above each point xyz (n, K, 3) (table frame)? Batched IK
        from the home posture (4 seeds). Cached per (env, arm, slot): only points that moved > tol (m) since their last check are
        solved again (the same answer for the same point)."""
        import torch
        K = self.K; dev = self.CS.fp.dev; fp = self.CS.fp
        e_np = np.asarray(envs); x_np = np.asarray(xyz, np.float32); n, k = x_np.shape[:2]
        if not hasattr(self, "_rc"):
            self._rc = {}
        key = (g, k, kind)
        if key not in self._rc:
            self._rc[key] = (np.full((fp.N, k, 3), np.inf, np.float32), np.zeros((fp.N, k), bool))
        cx, cok = self._rc[key]
        if key not in getattr(self, "_rq", {}):
            self.__dict__.setdefault("_rq", {})[key] = (torch.zeros(fp.N, k, 7, device=dev), np.zeros((fp.N, k), bool))
        cq, hasq = self._rq[key]
        stale = ~(np.abs(cx[e_np] - x_np) <= tol).all(-1)
        if stale.any():
            rr, kk = np.nonzero(stale)
            er, xr = e_np[rr], x_np[rr, kk]
            x = torch.as_tensor(xr, device=dev); e = torch.as_tensor(er, device=dev).long()
            T = K.top_down(x + torch.tensor([0, 0, 0.02], device=dev), 0.0)
            warm = hasq[er, kk]
            q_init = torch.where(torch.as_tensor(warm, device=dev)[:, None], cq[e, torch.as_tensor(kk, device=dev)], self.q0.expand(len(rr), 7))
            # warm rows (a solved point that moved a little): fewer iterations / seeds; cold rows: the full solve
            it_ = 15 if warm.all() else iters; sd_ = 2 if warm.all() else seeds
            q, ok, _, _ = K.ik(self.binv[e, g] @ T, q_init, yaw_free=True, iters=it_, n_seeds=sd_)
            cx[er, kk] = xr; okn = ok.cpu().numpy(); cok[er, kk] = okn
            cq[e, torch.as_tensor(kk, device=dev)] = q; hasq[er, kk] = okn
        return cok[e_np].copy()

    def prewarm(self, J, layout):
        """Reach of every (env, arm, slot) at the start layout and at the frozen targets, in one batched solve per (arm, kind), before a
        rollout; afterwards only moved points are re-solved (warm-started)."""
        import torch
        fp = self.CS.fp; t0 = time.time()
        lay = layout.cpu().numpy() if torch.is_tensor(layout) else np.asarray(layout)
        xyz = np.nan_to_num(lay[..., :3]) - np.array([0, 0, 0.002], np.float32)
        top = xyz.copy(); top[..., 2] += np.nan_to_num(fp.ext.cpu().numpy()[..., 2])
        yaw = self.CS.yaw_off(torch.as_tensor(lay, device=fp.dev)).cpu().numpy()
        tg = J.targets(xyz, np.nan_to_num(yaw))
        env = np.arange(fp.N)
        for g in (0, 1):
            self.reach(env, g, top, kind="start"); self.reach(env, g, tg, kind="target")
        self.t_prewarm = time.time() - t0

    def home_palms(self):
        """(N, 2, 3) palm centres (table frame) and (N, 2) yaws of the hands at the arms' home TCP poses."""
        import torch
        tcp = self.home_tcp; R = self.K.quat_to_R(tcp[..., 3:])
        yaw = torch.atan2(R[..., 1, 1], R[..., 0, 1])                                           # the finger axis (hand y) direction
        return tcp[..., :3] + torch.tensor([0, 0, self.TCP_DZ], device=tcp.device), yaw

    def reset(self, e):
        self.q[e] = self.q0; self.retract[e] = False; self.fail_streak[e] = 0
        if getattr(self, "_last", None) is not None:                     # a reset hand is re-solved at the next check
            self._last[0][e] = 1e3
        for c in (self.n_ik, self.n_obj, self.n_arm, self.n_tab, self.n_near, self.yield_g, self.n_touch):
            c[e] = 0
        if PUSH_CONTACT:
            self.obj_touch_age[e] = 10 ** 6
        if JOINTS:
            JI.tracker_reset(self, e)

    def _retract(self, envs, g):
        """move gripper g's hand of envs back to its arm's home pose (straight line; exempt from the abort while it goes)."""
        import torch
        CS = self.CS; e = torch.as_tensor(envs, device=CS.dev).long()
        H = CS.home()
        CS._start_move(e, g, H[e, g], CS._home_yaw[e, g], CS.hO[e, g].clone(), after=0, route=REFLEX or HAND_PEN)   # over the objects, not through
        self.retract[e, g] = True; self.n_retract += len(e)

    def _clear(self, q):
        """min arm-arm surface distance (N,) with the Franka sphere model shrunk by ARM_PAD_SHRINK (their padding is 1-2.5 cm)."""
        import torch
        K = self.K
        pa, r = K.spheres(q[:, 0], self.base[:, 0]); pb, _ = K.spheres(q[:, 1], self.base[:, 1])
        rr = (r - ARM_PAD_SHRINK).clamp(min=0.005)
        return (torch.cdist(pa, pb) - (rr[:, None] + rr[None, :])).flatten(1).amin(1)

    def _shield(self, q):
        """Projection shield for arm-arm contact: a safety shield that projects the motion to a safe one and penalises its
        interventions (Thumm & Althoff, 2023), with collision-aware IK as in CollisionIK. At each arm check the IK's new pair q (N, 2, 7) is tested (accurate padding). Where it would
        touch: (1) projection: each arm steps along the null space of its TCP Jacobian (the elbow swing that keeps the hand's pose) by
        +-SH_STEP, the side that increases the clearance, up to SH_ITERS rounds; a step that moves the TCP > 5 mm / 3 deg is undone; (2) only
        if the pair still touches (deeper than the last accepted pair), the arm whose move started more recently keeps its last pose and its
        hand steps back one check (blocked ARM_BLOCK_S in a row -> its take aborts, "arm_block"). Penalties: a projection is charged as 1/4
        check of arm-arm contact, a hold as a full check (FPR_W_ARMARM / s). -> the accepted q."""
        import torch
        K, CS = self.K, self.CS
        q_old = self.q
        if not hasattr(self, "blocked_n"):
            self.blocked_n = torch.zeros_like(CS.phase, dtype=torch.long); self.n_block = 0; self.n_block_abort = 0; self.n_proj = 0
        d = self._clear(q); bad = d < 0
        if bool(bad.any()):
            ix = torch.nonzero(bad).flatten(); qb = q[ix].clone(); b_ = self.base[ix]
            T0 = [K.fk_tcp(qb[:, g]) for g in (0, 1)]
            lo = torch.as_tensor(K.Q_LO, device=q.device); hi = torch.as_tensor(K.Q_HI, device=q.device)

            def clr(qq):
                pa, r = K.spheres(qq[:, 0], b_[:, 0]); pb, _ = K.spheres(qq[:, 1], b_[:, 1]); rr = (r - ARM_PAD_SHRINK).clamp(min=0.005)
                return (torch.cdist(pa, pb) - (rr[:, None] + rr[None, :])).flatten(1).amin(1)
            cur = clr(qb)
            for _ in range(SH_ITERS):
                act_ = cur < 0
                if not bool(act_.any()):
                    break
                for g in (0, 1):
                    n_ = torch.linalg.svd(K.jacobian(K.fk(qb[:, g]))).Vh[:, -1, :]                    # (m, 7) the null direction
                    best, bq = cur.clone(), qb[:, g].clone()
                    for sg in (1.0, -1.0):
                        qt = qb.clone(); qt[:, g] = torch.minimum(torch.maximum(qb[:, g] + sg * SH_STEP * n_, lo), hi)
                        Tt = K.fk_tcp(qt[:, g]); pe = (Tt[:, :3, 3] - T0[g][:, :3, 3]).norm(dim=-1)
                        ce = ((Tt[:, :3, :3] * T0[g][:, :3, :3]).sum((-1, -2)) - 1).div(2).clamp(-1, 1).acos()
                        okp = (pe < 0.005) & (ce < math.radians(3.0)) & act_
                        dt_ = clr(qt); imp = okp & (dt_ > best)
                        best = torch.where(imp, dt_, best); bq = torch.where(imp[:, None], qt[:, g], bq)
                    qb[:, g] = bq; cur = best
            moved = (qb != q[ix]).any(-1).any(-1)
            self.n_proj += int(moved.sum())
            self.n_arm[ix[moved]] += max(1, self.every // 4)                 # the intervention is penalised
            q = q.clone(); q[ix] = qb
        # (deadlock-free) leader / follower per env, re-decided every check: the leader = the moving arm that has been in its current move
        # longest (an idle / done arm is never the leader). Where the new pair would touch (or touch deeper): the follower yields by moving
        # away (it keeps its last pose and its hand's progress steps back one check along its own path; an idle / done follower is sent home);
        # only if the pair would still touch does the leader pause for this check. One arm always progresses or moves away, so a jam clears;
        # a follower that has yielded ARM_BLOCK_S aborts its take and retreats home (that retreat is always accepted: it moves away).
        d_old = self._clear(q_old); d_new = self._clear(q)
        moving = (CS.phase == 1) | (CS.phase == 4) | (CS.phase == 2)
        bad = (d_new < 0) & (d_new < d_old - 1e-4) & moving.any(1)
        if bool(bad.any()):                                              # (batched over envs: no Python loop per table)
            ar = torch.arange(q.shape[0], device=q.device)
            t_ = torch.where(moving, CS.t.float(), torch.full_like(CS.t.float(), -1.0))
            lead = (t_[:, 1] > t_[:, 0]).long()                            # the moving arm longest in its current move (ties: arm 0)
            fol = 1 - lead
            fmov = moving[ar, fol]; fret = self.retract[ar, fol]
            y = bad & fmov & ~fret                                          # a moving follower yields: last pose, progress stepped back two checks
            if bool(y.any()):
                e_ = ar[y]; g_ = fol[y]
                q[e_, g_] = q_old[e_, g_]
                CS.t[e_, g_] = (CS.t[e_, g_] - 2 * self.every).clamp(min=0)
                self.blocked_n[e_, g_] += 1; self.n_block += int(y.sum()); self.n_arm[e_] += self.every
                self.yield_g[e_, g_] += 1.0                                # charged to this gripper's own take
            h = bad & ~fmov & ~fret                                         # an idle / done follower in the way goes home
            for g in (0, 1):
                eh = torch.nonzero(h & (fol == g)).flatten()
                if len(eh):
                    self._retract(eh.tolist(), g); self.n_idle_home = getattr(self, "n_idle_home", 0) + len(eh)
            d2 = self._clear(q)
            p_ = bad & (d2 < 0) & (d2 < d_old - 1e-4)                      # still touching: the leader pauses this one check
            if bool(p_.any()):
                e_ = ar[p_]; g_ = lead[p_]
                q[e_, g_] = q_old[e_, g_]
                CS.t[e_, g_] = (CS.t[e_, g_] - self.every).clamp(min=0)
                self.n_pause = getattr(self, "n_pause", 0) + int(p_.sum()); self.n_arm[e_] += max(1, self.every // 2)
                self.yield_g[e_, g_] += 0.5
        unb = ~(q == q_old).all(-1)
        self.blocked_n[unb] = 0
        lim = max(1, int(round(ARM_BLOCK_S / (CS.dt * self.every))))
        for ei, g_ in torch.nonzero(self.blocked_n >= lim).tolist():
            if int(CS.phase[ei, g_]) in (1, 4):
                self._abort(ei, g_, "arm_block"); self.n_block_abort += 1
            self.blocked_n[ei, g_] = 0
        return q

    def _solid(self, q):
        """the arms are solid to each other (a kinematic stand-in for physical arms): at each check the IK's new poses q (N, 2, 7) are
        tested with the Franka sphere model (which pads each arm 1-2.5 cm beyond its mesh). Where the new pair would touch (or touch deeper
        than the last accepted pair), the arm whose move started more recently keeps its last pose, and its hand's progress steps back one
        check (the pose it had then); blocked for ARM_BLOCK_S in a row, its take aborts ("arm_block"). Then the other arm is re-tested
        against the kept pose the same way. No waiting or turn-taking rule: an arm simply cannot pass through the other; the policy learns to
        avoid it (time, aborts). -> the accepted q."""
        import torch
        K, CS = self.K, self.CS
        q_old = self.q
        d_old = K.arm_arm_min_dist(q_old[:, 0], self.base[:, 0], q_old[:, 1], self.base[:, 1])
        if not hasattr(self, "blocked_n"):
            self.blocked_n = torch.zeros_like(CS.phase, dtype=torch.long); self.n_block = 0; self.n_block_abort = 0
        q = q.clone()
        moving = (CS.phase == 1) | (CS.phase == 4) | (CS.phase == 2)
        for _ in range(2):                                                # at most one arm reverted per pass (the more recent mover first)
            d_new = K.arm_arm_min_dist(q[:, 0], self.base[:, 0], q[:, 1], self.base[:, 1])
            bad = (d_new < 0) & (d_new < d_old - 1e-4)
            if not bool(bad.any()):
                break
            for ei in torch.nonzero(bad).flatten().tolist():
                cand = [g for g in (0, 1) if bool((q[ei, g] != q_old[ei, g]).any())]
                if not cand:
                    continue
                mv = [g for g in cand if bool(moving[ei, g])] or cand
                g_ = min(mv, key=lambda g: int(CS.t[ei, g]))
                q[ei, g_] = q_old[ei, g_]
                if bool(moving[ei, g_]):
                    CS.t[ei, g_] = (CS.t[ei, g_] - self.every).clamp(min=0)
                    self.blocked_n[ei, g_] += 1; self.n_block += 1
                self.n_arm[ei] += self.every                               # the attempt is charged as arm-arm contact (one check's time)
        unb = (q == q_old).all(-1) == False                              # noqa: E712  arms that moved freely this check
        self.blocked_n[unb] = 0
        lim = max(1, int(round(ARM_BLOCK_S / (CS.dt * self.every))))
        for ei, g_ in torch.nonzero(self.blocked_n >= lim).tolist():
            if int(CS.phase[ei, g_]) in (1, 4):
                self._abort(ei, g_, "arm_block"); self.n_block_abort += 1
            self.blocked_n[ei, g_] = 0
        return q

    def _abort(self, ei, g_, why):
        """abort gripper g_'s current move in env ei: why "ik" (its arm can't follow: ik_carry / ik_approach) or "arm_arm" (the arms
        overlap). A carry stops, then releases in place; an approach / free move is cancelled and the arm takes the hand home."""
        CS = self.CS
        CS.n_take_abort += 1; CS.n_ab_env[ei] += 1
        k_ = CS.tk_open.get((ei, g_))
        ph_ = int(CS.phase[ei, g_])
        if ph_ == 4 and int(CS.after[ei, g_]) == 0:                      # a free move: not a take
            CS.n_take_abort -= 1; CS.n_ab_env[ei] -= 1; self.n_move_abort += 1; k_ = None
        if k_ is not None and CS.takes[k_]["abort"] is None:
            CS.takes[k_]["abort"] = (why + "_carry" if ph_ == 1 else why + "_approach"); CS.takes[k_]["abort_step"] = CS.step_i
        if ph_ == 1:
            CS.drop([ei], g_, stop=True)
            k_ = CS.ev_open.get((ei, g_))
            if k_ is not None:
                CS.ev[k_]["ik_abort"] = True
        else:                                                            # the approach / free move is cancelled: the arm takes the hand home
            CS.pending.pop((ei, g_), None); CS.mev_open.pop((ei, g_), None); self._retract([ei], g_)
        self.fail_streak[ei, g_] = 0

    def check(self):
        import torch
        K, CS = self.K, self.CS; fp = CS.fp
        t0 = time.time()
        tcp = CS.hP - fp.org[:, None] - torch.tensor([0, 0, self.TCP_DZ], device=fp.dev)          # (N, 2, 3) table frame
        T = K.top_down(tcp, CS.hY)
        if JOINTS:                                                       # IK of the planned hand (the path point, top-down at the planned yaw)
            tcp, T = JI.jt_target(self)
        if ROT_EDIT and not JOINTS and bool((CS.hRP.abs().amax(-1) > 0).any()):   # tilted hands: the TCP at the full commanded orientation
            tl_ = CS.hRP.abs().amax(-1) > 0                                # (untilted rows keep the top-down target)
            qf = FE.qmul(FE.qyaw(CS.hY), q_tilt(CS.hRP)); Rf = FE.qmat(qf.reshape(-1, 4)).reshape(*qf.shape[:-1], 3, 3)
            tcp_t = CS.hP - fp.org[:, None] - (Rf @ torch.tensor([0, 0, self.TCP_DZ], device=fp.dev))
            D = FE.qmat(FE.qmul(qf, FE.qyaw(-CS.hY)).reshape(-1, 4)).reshape(*qf.shape[:-1], 3, 3)
            T_t = K.top_down(tcp_t, CS.hY).clone(); T_t[..., :3, :3] = D @ T_t[..., :3, :3]
            tcp = torch.where(tl_[..., None], tcp_t, tcp); T = torch.where(tl_[..., None, None], T_t, T)
        loc = self.binv @ T
        if ARM_FAST:                                                     # only hands that moved since the last check are re-solved
            last = getattr(self, "_last", None)                           # (an unmoved hand keeps its solution and verdict); 6 iterations for those,
            mv = torch.ones_like(CS.phase, dtype=torch.bool) if last is None else (   # 6 more for the rows not yet converged (the same
                ((tcp - last[0]).norm(dim=-1) > 1e-3) | ((CS.hY - last[1]).abs() > 0.01))            # 12-iteration sequence for them)
            q = self.q.clone(); ok = last[2].clone() if last is not None else torch.zeros_like(mv)
            pe = last[3].clone() if last is not None else torch.zeros(mv.shape, device=mv.device); re = last[4].clone() if last is not None else torch.zeros_like(pe)
            if bool(mv.any()):
                fe, fg = torch.nonzero(mv, as_tuple=True)
                q1, ok1, pe1, re1 = K.ik(loc[fe, fg], self.q[fe, fg], yaw_free=not IK_YAW, iters=6)
                if bool((~ok1).any()):
                    nf = torch.nonzero(~ok1).flatten()
                    q2, ok2, pe2, re2 = K.ik(loc[fe[nf], fg[nf]], q1[nf], yaw_free=not IK_YAW, iters=6)
                    q1[nf] = q2; ok1[nf] = ok2; pe1[nf] = pe2; re1[nf] = re2
                q[fe, fg] = q1; ok[fe, fg] = ok1; pe[fe, fg] = pe1; re[fe, fg] = re1
        else:
            q, ok, pe, re = K.ik(loc, self.q_ik if JOINTS else self.q, yaw_free=not IK_YAW, iters=12)
        mvm = mv if ARM_FAST else torch.ones_like(ok)                    # unmoved hands keep their verdict: no repeated rescue solves
        if bool((~ok & mvm).any()):                                      # a failed warm-started solve is re-solved cold (several seeds) before it
            fe, fg = torch.nonzero(~ok & mvm, as_tuple=True)              # counts: tracking lag is not a reach failure
            qc, okc, pec, rec = K.ik(loc[fe, fg], self.q0.expand(len(fe), 7), yaw_free=not IK_YAW, iters=COLD_IT, n_seeds=COLD_SEEDS)
            if IK_YAW and bool((~okc).any()):                              # the gripper is symmetric: the yaw + 180 deg target too
                nf = torch.nonzero(~okc).flatten()
                Rz = torch.tensor([[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], device=loc.device)
                qf, okf, pef, ref_ = K.ik(loc[fe[nf], fg[nf]] @ Rz, self.q0.expand(len(nf), 7), yaw_free=False, iters=COLD_IT, n_seeds=COLD_SEEDS)
                qc[nf] = torch.where(okf[:, None], qf, qc[nf]); okc[nf] = okf; pec[nf] = pef; rec[nf] = ref_
            q[fe, fg] = torch.where(okc[:, None], qc, q[fe, fg]); ok[fe, fg] = okc; pe[fe, fg] = pec; re[fe, fg] = rec
        u_ = (CS.t.float() * CS.dt / CS.mdur.clamp(min=1e-6)).clamp(max=1.0)
        route = (CS.phase == 4) & ((CS.after == 0) | (u_ < self.route_u))
        if bool((~ok & route & mvm).any()):                              # mid-route: position + a tilt up to route_tilt is enough
            fe, fg = torch.nonzero(~ok & route & mvm, as_tuple=True)
            qt, okt, pet, ret_ = K.ik(loc[fe, fg], q[fe, fg], yaw_free=True, iters=TILT_IT, w_rot=0.1, tol_rot=self.route_tilt, n_seeds=TILT_SEEDS)
            q[fe, fg] = torch.where(okt[:, None], qt, q[fe, fg]); ok[fe, fg] = okt; pe[fe, fg] = pet; re[fe, fg] = ret_
        if JOINTS:                                                       # q = IK + env(u) (prior + the controller's residual), clamped
            q = JI.jt_compose(self, q)
        if ARM_SHIELD:                                                   # the projection shield (null-space push-apart, hold only as a fallback)
            q = self._shield(q)
        elif ARM_SOLID:                                                  # solid arms: a new pose is accepted only if it creates or deepens no
            q = self._solid(q)                                             # arm-arm contact; else that arm keeps its last pose and its hand steps back
        self.q = q
        if JOINTS:                                                       # the hand displacement / obs / costs from FK(this q)
            JI.jt_after_check(self)
        if IK_YAW:                                                         # each arm's self-collision (non-adjacent bodies of the kinematic model)
            d_self = K.self_min_dist(q)                                    # (N, 2) m; < 0 = overlap -> counted with the arm contacts
            self.n_tab += (((d_self < 0).any(1)) & ~((CS.phase == 5).all(1))).long() * self.every
        if ARM_FAST:
            self._last = (tcp.clone(), CS.hY.clone(), ok.clone(), pe.clone(), re.clone())
        P = CS.poses(); top = P[..., 2] + fp.ext[..., 2]
        r = 0.5 * torch.minimum(fp.ext[..., :2].amin(-1), fp.ext[..., 2])                       # one inscribed-ish sphere per object
        ctr = torch.stack([P[..., 0], P[..., 1], P[..., 2] + 0.5 * fp.ext[..., 2]], -1)
        if OBJ_SPHERES:                                                    # 3 spheres along each object's long axis (knives, bananas, rims)
            yo = FE.yaw_zyx(P[..., 3:]); lx = fp.ext[..., 0] >= fp.ext[..., 1]
            ax = torch.where(lx[..., None], torch.stack([torch.cos(yo), torch.sin(yo)], -1), torch.stack([-torch.sin(yo), torch.cos(yo)], -1))
            half = (0.5 * fp.ext[..., :2].amax(-1) - r).clamp(min=0.0)
            sh = torch.stack([-half, torch.zeros_like(half), half], -1)                           # (N, K, 3)
            c3 = ctr[..., None, :] + torch.cat([ax[..., None, :] * sh[..., None], torch.zeros_like(sh)[..., None]], -1)
            objs = torch.cat([c3, r[..., None, None].expand(*r.shape, 3, 1)], -1)                 # (N, K, 3, 4)
        else:
            objs = torch.cat([ctr, r[..., None]], -1)[..., None, :]                               # (N, K, 1, 4) table frame
        objs = torch.where((fp.valid & torch.isfinite(objs).all(-1).all(-1))[..., None, None], objs, torch.tensor([0, 0, -10.0, 0.0], device=fp.dev))
        dmin = []; dko_ = []
        far = torch.tensor([0, 0, -10.0, 0.0], device=fp.dev)
        for g in (0, 1):
            o = objs.clone()
            held = (CS.phase[:, g] == 1) | (CS.phase[:, g] == 2) | ((CS.phase[:, g] == 4) & (CS.after[:, g] == 1))
            ar = torch.arange(fp.N, device=fp.dev)
            o[ar, CS.car[:, g]] = torch.where(held[:, None, None], far, o[ar, CS.car[:, g]])
            if FAST2:
                pt_ = CS._pending_idx(g)                                                         # the object being approached (one write)
                if pt_ is not None:
                    o[pt_[0], pt_[1]] = far
            else:
                for (ei, g_), pd in CS.pending.items():
                    if g_ == g:
                        o[ei, pd["c"]] = far
            dk_ = K.arm_objects_min_dist(q[:, g], self.base[:, g], o.reshape(fp.N, -1, 4), skip=K.TABLE_SKIP)
            dmin.append(dk_.amin(-1))
            if PUSH_CONTACT:
                dko_.append(dk_.reshape(fp.N, o.shape[1], -1).amin(-1))                            # per object
        d_obj = torch.stack(dmin, 1).amin(1)
        if PUSH_CONTACT:                                                 # which objects the robot (arm / hand / fingers / a carried object) touches now
            dk2 = torch.minimum(torch.stack(dko_, 0).amin(0), carried_contact(objs, CS.phase, CS.car))
            self.obj_touch_age = torch.where(dk2 <= PUSH_CONTACT_M, torch.zeros_like(self.obj_touch_age), self.obj_touch_age + self.every)
        d_arm = self._clear(q) if (ARM_SHIELD or ARM_ACCURATE) else K.arm_arm_min_dist(q[:, 0], self.base[:, 0], q[:, 1], self.base[:, 1])
        d_tab = torch.stack([K.arm_plane_min_dist(q[:, g], self.base[:, g], 0.0, skip=K.CONTACT_SKIP) for g in (0, 1)], 1).amin(1)
        live = ~((CS.phase == 5).all(1))
        if os.environ.get("FPC_ARM_LOG") == "1":                            # FPC_ARM_LOG=1: every check's contacts, for choosing the check interval
            self.__dict__.setdefault("clog", []).append((CS.step_i, ((d_arm < 0) & live).cpu().numpy(), ((d_obj < 0) & live).cpu().numpy()))
        self.n_ik += ((~ok).any(1) & live).long() * self.every; self.n_obj += ((d_obj < 0) & live).long() * self.every
        # a take whose arm cannot follow the hand (IK fails abort_after checks in a row) fails: the approach is cancelled / the object dropped
        self.retract &= CS.phase == 4                                    # a finished retraction
        busy_ = ((CS.phase == 1) | (CS.phase == 4)) & ~self.retract       # takes and free moves are judged (a free move to an unreachable pose aborts)
        if os.environ.get("FPC_IK_DIAG") == "1" and bool((~ok & busy_).any()):   # diagnostic: every failed check of a take, where and why
            fe, fg = torch.nonzero(~ok & busy_, as_tuple=True)
            qc, okc, pec, rec = K.ik(loc[fe, fg], self.q0.expand(len(fe), 7), yaw_free=True, iters=80, n_seeds=8)     # cold, many seeds
            u = (CS.t[fe, fg].float() * CS.dt / CS.mdur[fe, fg].clamp(min=1e-6)).clamp(max=1.0)
            bxy = self.base[fe, fg][:, :2, 3]
            for i in range(len(fe)):
                ph_ = int(CS.phase[fe[i], fg[i]])
                self.__dict__.setdefault("diag", []).append(dict(
                    step=int(CS.step_i), env=int(fe[i]), g=int(fg[i]), phase=ph_, u=float(u[i]) if ph_ == 4 else None,
                    tcp=[round(float(v), 4) for v in tcp[fe[i], fg[i]]], reach_xy=float((tcp[fe[i], fg[i], :2] - bxy[i]).norm()),
                    pos_err=float(pe[fe[i], fg[i]]), rot_err_deg=float(re[fe[i], fg[i]]) * 57.2958,
                    cold_ok=bool(okc[i]), cold_pos_err=float(pec[i]), cold_rot_err_deg=float(rec[i]) * 57.2958, streak=int(self.fail_streak[fe[i], fg[i]])))
        self.fail_streak = torch.where(~ok & busy_, self.fail_streak + 1, torch.zeros_like(self.fail_streak))
        ab = self.fail_streak >= self.abort_after
        if bool(ab.any()):
            for ei, g_ in torch.nonzero(ab).tolist():
                self._abort(ei, g_, "ik")
        if ARM_ABORT > 0:                                                # an arm-arm overlap
            act_ = ((CS.phase == 1) | (CS.phase == 4)) & ~self.retract     # deeper than ARM_ABORT aborts the more recently started active move of that env
            hit_ = (d_arm < -ARM_ABORT) & live & act_.any(1)               # (a carry stops, then releases in place; an approach / free move is cancelled)
            for ei in torch.nonzero(hit_).flatten().tolist():
                gs_ = [g for g in (0, 1) if bool(act_[ei, g])]
                g_ = min(gs_, key=lambda g: int(CS.t[ei, g])) if len(gs_) > 1 else gs_[0]
                self._abort(ei, g_, "arm_arm"); self.n_arm_abort = getattr(self, "n_arm_abort", 0) + 1
        # an idle hand where its arm cannot be (e.g. after a dropped carry's settle) goes home too: the hand is never where its arm can't hold it
        idle = ((CS.phase == 0) | (CS.phase == 3)) & ~ok & ~self.retract
        if bool(idle.any()):
            for ei, g_ in torch.nonzero(idle).tolist():
                ph0 = int(CS.phase[ei, g_]); self._retract([ei], g_)
                if ph0 == 3:
                    CS.phase[ei, g_] = 4; CS.after[ei, g_] = 3            # a stopped gripper stays stopped after its retraction
        self.n_arm += ((d_arm < 0) & live).long() * self.every; self.n_tab += ((d_tab < -0.005) & live).long() * self.every
        self.n_touch += ((d_arm < 0) & live).long() * self.every
        if ARM_BOUNCE:                                                   # arms that really touch bounce (both moving arms of the env)
            CS.b_left, CS.b_cool, st_ = bounce_trigger((d_arm < 0) & live, CS.phase, CS.t, CS.rel, CS.b_cool, CS.b_left,
                                                       int(round(BOUNCE_S / CS.dt)), int(round(BOUNCE_COOL_S / CS.dt)))
            self.n_bounce = getattr(self, "n_bounce", 0) + int(st_.sum())
            if PUSH_BOUNCE and bool(st_.any()):                          # shove each bouncing hand away from the other arm; repeated -> stuck
                pa_, r_ = K.spheres(q[:, 0], self.base[:, 0]); pb_, _ = K.spheres(q[:, 1], self.base[:, 1]); rr_ = (r_ - ARM_PAD_SHRINK).clamp(min=0.005)
                CS.b_push = torch.where(st_[..., None], push_dirs(pa_, pb_, rr_, rr_) * BOUNCE_PUSH_M, CS.b_push)
                CS.b_times, stuck_ = bounce_stuck(CS.b_times, st_, CS.step_i, int(round(ARM_BLOCK_S / CS.dt)), BOUNCE_STUCK_N)
                for ei, g_ in torch.nonzero(stuck_).tolist():
                    if int(CS.phase[ei, g_]) in (1, 4):
                        self._abort(ei, g_, "arm_bounce"); self.n_bounce_abort = getattr(self, "n_bounce_abort", 0) + 1
                    CS.b_times[ei, g_] = -10 ** 9; CS.b_push[ei, g_] = 0.0
        if NEAR_M > 0:                                                   # every arm / gripper part closer than NEAR_M to the other arm costs
            self.n_near += ((NEAR_M - d_arm).clamp(min=0.0, max=NEAR_M) / NEAR_M * live.float()) * self.every
        self.t_s += time.time() - t0


# =====================================================================================================================================
class FrozenJudge:
    """the start frame's T1 (ensemble), reference-relative, fixed for the episode (not re-queried later, so the policy cannot change its goals)."""

    def __init__(self, LP, tpl_ids, xyz, yaw, instruction=None):
        import torch
        import torch.nn.functional as F
        from cleantables.worldmodel.model import ref_frame, to_frame
        self.LP, self.dev = LP, LP.dev; self.instruction = instruction
        tpl_ids = np.asarray(tpl_ids).astype(int); xyz = np.asarray(xyz, np.float32); yaw = np.asarray(yaw, np.float32)
        n, Imax = xyz.shape[:2]; self.tpl = tpl_ids; self.Imax = Imax
        self.j = np.full((n, Imax), -1, np.int64); self.off = np.zeros((n, Imax, 3), np.float32); self.szj = np.ones((n, Imax, 3), np.float32)
        self.fixed = np.zeros((n, Imax, 3), np.float32); self.is_fixed = np.zeros((n, Imax), bool); self.mov = np.zeros((n, Imax), bool)
        self.sd = np.zeros((n, Imax, 2), np.float32); self.sdz = np.zeros((n, Imax), np.float32)
        with torch.no_grad():
            for t in np.unique(tpl_ids):
                rows = np.flatnonzero(tpl_ids == t); B = LP.B[t]; I = B.I
                b = B._batch(torch.as_tensor(xyz[rows, :I], device=self.dev), torch.as_tensor(yaw[rows, :I], device=self.dev), None)
                ins_ = instruction(t) if callable(instruction) else instruction     # (instruction sampling) a per-template instruction override
                temb = B._task_emb(ins_, len(rows))
                if temb is not None:
                    b["task_emb"] = temb
                mode = torch.ones(len(rows), dtype=torch.long, device=self.dev)
                A, OFF, MS = [], [], []
                for m in LP.models:
                    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                    M = m.t1_mix(E); A.append(F.softmax(M["logit"], -1)); OFF.append(M["off"]); MS.append(M)
                a = torch.stack(A).mean(0); j = a.argmax(-1)                                         # (r, slots)
                # the tolerance: the ensemble's spread of the chosen component (members' mean variance + variance of the means) x the
                # global sd_scale, as the WM read-out's STOP rule (an object is done within one predicted sd of its T1 end)
                mu_s, sd_s = zip(*[m.t1_comp(M_, j) for m, M_ in zip(LP.models, MS)])
                mu_s, sd_s = torch.stack(mu_s), torch.stack(sd_s)
                sdT = torch.sqrt((sd_s ** 2).mean(0) + mu_s.var(0, unbiased=False)) * float(getattr(LP.models[0], "sd_scale", 1.0))
                off = torch.stack(OFF).mean(0)                                                       # (r, i, j, 3) members' mean offset
                geo = b["obj_geo"].float(); p, yw, sz = ref_frame(geo)
                g3 = lambda T_: torch.gather(T_, 2, j[..., None, None].expand(*j.shape, 1, 3)).squeeze(2)   # noqa: E731
                oj = g3(off); szj = torch.gather(sz, 1, j[..., None].expand(*j.shape, 3))
                from cleantables.worldmodel.model import from_frame
                pj = torch.gather(p, 1, j[..., None].expand(*j.shape, 3)); ywj = torch.gather(yw, 1, j)
                end0 = from_frame(oj, pj, ywj, szj)                                                  # (r, slots, 3) top centre at the start
                kind = b["obj_kind"]
                for q, r in enumerate(rows):
                    self.j[r, :I] = j[q, :I].cpu().numpy(); self.off[r, :I] = oj[q, :I].cpu().numpy(); self.szj[r, :I] = szj[q, :I].cpu().numpy()
                    fx = (j[q, :I] == torch.arange(I, device=self.dev)) | (kind[q, j[q, :I]] != 0)            # self / table reference: fixed
                    self.is_fixed[r, :I] = fx.cpu().numpy(); self.fixed[r, :I] = end0[q, :I].cpu().numpy()
                    self.mov[r, :I] = (kind[q, :I] == 0).cpu().numpy(); self.sd[r, :I] = sdT[q, :I, :2].cpu().numpy()
                    self.sdz[r, :I] = sdT[q, :I, 2].cpu().numpy() if sdT.shape[-1] > 2 else 0.0
        self.end0 = self.targets(xyz, yaw)
        top0 = self._top(xyz)
        self.static = (np.linalg.norm((self.end0 - top0)[..., :2], axis=-1) * 100 < STATIC_CM) | ~self.mov
        self.start_xyz = xyz.copy(); self.start_yaw = np.asarray(yaw, np.float32).copy()
        self.d0_cm = np.linalg.norm((self.end0 - top0)[..., :2], axis=-1) * 100                     # each object's start distance to its target

    def _top(self, xyz):
        H = np.zeros(xyz.shape[:2], np.float32)
        for r, t in enumerate(self.tpl):
            I = self.LP.B[t].I; H[r, :I] = np.asarray(self.LP.B[t].Z["extents"], np.float32)[:, 2]
        out = np.asarray(xyz, np.float32).copy(); out[..., 2] += H
        return out

    def _geo_consts(self):
        """per env (n, I): height H, baked yaw + long-side turn (the WM's box_feat yaw), for the vectorised frames below."""
        if not hasattr(self, "_H"):
            n, I = self.j.shape; H = np.zeros((n, I), np.float32); yb = np.zeros((n, I), np.float32); ok = np.zeros((n, I), bool)
            for r, t in enumerate(self.tpl):
                Z = self.LP.B[t].Z; Ii = self.LP.B[t].I; ex = np.asarray(Z["extents"], np.float32)
                H[r, :Ii] = ex[:, 2]; yb[r, :Ii] = np.asarray(Z["yaw_baked"], np.float32) + np.where(ex[:, 1] > ex[:, 0], np.pi / 2, 0.0); ok[r, :Ii] = True
            self._H, self._yb, self._ok = H, yb, ok
        return self._H, self._yb, self._ok

    def targets(self, xyz, yaw):
        """(n, I, 3) every object's frozen target (top centre) given the current poses (references follow their actual pose). Vectorised
        over envs: the reference frame of j = (x, y, bottom z + H) and the long side's yaw, as worldmodel.bridge._batch / ref_frame."""
        H, yb, ok = self._geo_consts()
        xyz = np.asarray(xyz, np.float32)[:, :self.Imax]; yaw = np.asarray(yaw, np.float32)[:, :self.Imax]
        n, I = self.j.shape; ar = np.arange(n)[:, None]; jj = np.clip(self.j, 0, I - 1)
        if REF_START and getattr(self, "start_yaw", None) is not None and xyz.shape[0] == n:   # references hold their start pose
            xyz, yaw = self.start_xyz[:, :self.Imax], self.start_yaw[:, :self.Imax]          # for every goal: nudging a container moves nothing
        elif FREEZE_REF and xyz.shape[0] == n:                            # a reference that left the table keeps its last on-table pose for
            if getattr(self, "_on_xyz", None) is None or self._on_xyz.shape != xyz.shape:          # its dependents' goals
                self._on_xyz, self._on_yaw = xyz.copy(), yaw.copy()
            on = xyz[..., 2] > -OFF_Z
            self._on_xyz = np.where(on[..., None], xyz, self._on_xyz); self._on_yaw = np.where(on, yaw, self._on_yaw)
            xyz, yaw = self._on_xyz, self._on_yaw
        pj = xyz[ar, jj].copy(); pj[..., 2] += H[ar, jj]
        ywj = 0.5 * np.arctan2(np.sin(2 * (yaw[ar, jj] + yb[ar, jj])), np.cos(2 * (yaw[ar, jj] + yb[ar, jj])))   # 180 deg symmetric, as sin2 / cos2
        u = self.off * self.szj; c, s_ = np.cos(ywj), np.sin(ywj)
        e = pj + np.stack([c * u[..., 0] - s_ * u[..., 1], s_ * u[..., 0] + c * u[..., 1], u[..., 2]], -1)
        out = np.where(self.is_fixed[..., None] | (self.j < 0)[..., None] | (self.j >= I)[..., None], self.fixed, e)
        return np.where(ok[..., None], out, self.fixed)

    def met3(self, xyz, yaw, ext, vel=None, dy0=None):
        """met_3d (n, I): the object's 3-D pose
        matches the frozen WM's own predicted end pose: |xy - goal xy| within the per-axis tolerance of score (the WM's sd, floor MET_CM,
        capped at half the start distance), |top z - goal z| within the WM's own z sd (floor MET_CM), at rest (speed < 2 cm/s, when vel is
        given), and, where the WM turns the object by > 15 deg (dy0), its yaw within 20 deg of start + dy (mod 180). The same rule for every
        object; objects the WM keeps static: as score."""
        sc = self.score(xyz, yaw); tg = self.targets(xyz, yaw)
        xyz = np.asarray(xyz, np.float32)[:, :self.Imax]; yaw = np.asarray(yaw, np.float32)[:, :self.Imax]
        top = self._top(xyz)
        zok = np.abs(top[..., 2] - tg[..., 2]) * 100 <= np.maximum(self.sdz[:, :self.Imax] * 100, MET_CM)
        rest = np.ones_like(zok) if vel is None else (np.linalg.norm(np.asarray(vel)[:, :self.Imax, :3], axis=-1) < 0.02)
        yok = np.ones_like(zok)
        if dy0 is not None:
            dy0 = np.asarray(dy0, np.float32)[:, :self.Imax]
            err = np.abs((yaw - self.start_yaw[:, :self.Imax] - dy0 + np.pi / 2) % np.pi - np.pi / 2)
            yok = (np.abs(dy0) <= np.radians(15)) | (err <= np.radians(20))
        mv = self.mov & ~self.static
        return np.where(mv, sc["met"] & zok & rest & yok, sc["met"])

    def targets_ref(self, xyz, yaw):
        """(n, I, 3) every object's frozen target (top centre) given the current poses (references follow their actual pose)."""
        import torch
        from cleantables.worldmodel.model import from_frame, ref_frame
        xyz = np.asarray(xyz, np.float32); yaw = np.asarray(yaw, np.float32); out = self.fixed.copy()
        with torch.no_grad():
            for t in np.unique(self.tpl):
                rows = np.flatnonzero(self.tpl == t); B = self.LP.B[t]; I = B.I
                b = B._batch(torch.as_tensor(xyz[rows, :I], device=self.dev), torch.as_tensor(yaw[rows, :I], device=self.dev), None)
                p, yw, _ = ref_frame(b["obj_geo"].float())
                j = torch.as_tensor(self.j[rows, :I], device=self.dev).clamp(min=0)
                pj = torch.gather(p, 1, j[..., None].expand(*j.shape, 3)); ywj = torch.gather(yw, 1, j)
                e = from_frame(torch.as_tensor(self.off[rows, :I], device=self.dev), pj, ywj, torch.as_tensor(self.szj[rows, :I], device=self.dev)).cpu().numpy()
                fx = self.is_fixed[rows, :I]
                out[rows, :I] = np.where(fx[..., None], self.fixed[rows, :I], e)
        return out

    def score(self, xyz, yaw):
        """-> dict per env: dist_cm (n, I) capped, met (n, I), static_moved_cm (n, I) (moved T1-static objects; 0 elsewhere)."""
        tg = self.targets(xyz, yaw); top = self._top(xyz)
        # xy only: the target's height is the WM's top-centre guess, while the actual height is set by what the object rests on (a fruit
        # dropped into a bowl sits lower than the predicted top): the same rule for every object
        d = np.minimum(np.linalg.norm((tg - top)[..., :2], axis=-1) * 100, CAP_CM)
        sm = np.where(self.static & self.mov, np.linalg.norm(np.asarray(xyz)[..., :2] - self.start_xyz[..., :2], axis=-1) * 100, 0.0)
        d = np.where(self.mov & ~self.static, d, 0.0)
        # per axis (cm): the WM's own sd, floored at MET_CM and capped at half the object's start distance to its target (with sd > the start
        # distance, an untouched object would count as met and a table could be 'done' at its first frame)
        cap = np.maximum(MET_CM, 0.5 * self.d0_cm)[..., None]
        tol = np.minimum(np.maximum(self.sd * 100, MET_CM), cap)
        within = (np.abs((tg - top)[..., :2]) * 100 <= tol).all(-1)
        draw = np.where(self.mov & ~self.static, np.linalg.norm((tg - top)[..., :2], axis=-1) * 100, 0.0)       # uncapped (the progress reward)
        return dict(dist_cm=d, met=within | self.static | ~self.mov, static_moved_cm=sm, dist_raw_cm=draw)


def met_new(judge, CS, xyz, yaw, vel, fe):
    """(FPC_MET_NEW) the episode-end met per object, for the finished envs fe (others False). A mover is met when
    (a) its top centre's xy is within the WM's own per-axis sd (floor MET_CM; no cap) of its start-frame frozen goal, (b) the frozen judge
    ensemble (the same seeds and instruction), re-queried on the current scene, predicts it stays (its predicted move < STATIC_CM: the judge's
    own 'static' read-out), and (c) it is at rest (< 2 cm/s). No z / yaw thresholds. Non-movers: as score. -> met, stays, predicted move (cm)."""
    Js = judge.J if hasattr(judge, "J") else (judge,)
    inst = np.asarray(CS.inst).reshape(-1).astype(int) if hasattr(judge, "J") else np.zeros(len(xyz), int)
    n, K = xyz.shape[:2]
    out = np.zeros((n, K), bool); stay = np.zeros((n, K), bool); disp = np.full((n, K), np.nan, np.float32)
    for q, J_ in enumerate(Js):
        rows = np.array([e for e in np.asarray(fe).tolist() if inst[e] == q], int)
        if not len(rows):
            continue
        I = J_.Imax
        Jc = FrozenJudge(J_.LP, J_.tpl[rows], xyz[rows, :I], yaw[rows, :I], instruction=J_.instruction)
        top = Jc._top(xyz[rows, :I])                                     # (Jc holds exactly these rows)
        tol = np.maximum(J_.sd[rows] * 100, MET_CM)
        a_ = (np.abs((J_.end0[rows] - top)[..., :2]) * 100 <= tol).all(-1)
        rest = np.linalg.norm(np.asarray(vel)[rows, :I, :3], axis=-1) < 0.02
        mv = J_.mov[rows] & ~J_.static[rows]
        sc = J_.score(xyz, yaw)["met"][rows]
        out[rows, :I] = np.where(mv, a_ & (True if MET_NOREQ else Jc.static) & rest, sc); stay[rows, :I] = Jc.static; disp[rows, :I] = Jc.d0_cm
    return out, stay, disp


# =====================================================================================================================================
class JudgeMux:
    """Instruction sampling: two frozen judges, one per instruction (the scene's own / the other one), chosen per env by
    CS.inst (resampled every episode): score / targets / met3 and the per-object arrays are taken row-wise from the env's judge."""

    def __init__(self, J0, J1, CS):
        self.J = (J0, J1); self.CS = CS

    def _w(self, a0, a1):
        m = np.asarray(self.CS.inst, bool).reshape((-1,) + (1,) * (np.ndim(a0) - 1))
        return np.where(m, a1, a0)

    def score(self, xyz, yaw):
        s0, s1 = self.J[0].score(xyz, yaw), self.J[1].score(xyz, yaw)
        return {k: self._w(s0[k], s1[k]) for k in s0}

    def targets(self, xyz, yaw):
        return self._w(self.J[0].targets(xyz, yaw), self.J[1].targets(xyz, yaw))

    def met3(self, xyz, yaw, ext, vel=None, dy0=None):
        return self._w(self.J[0].met3(xyz, yaw, ext, vel, getattr(self.J[0], "dy0", None)), self.J[1].met3(xyz, yaw, ext, vel, getattr(self.J[1], "dy0", None)))

    def __getattr__(self, k):
        if k in ("J", "CS"):
            raise AttributeError(k)
        a0, a1 = getattr(self.J[0], k), getattr(self.J[1], k)
        return self._w(a0, a1) if isinstance(a0, np.ndarray) else a0


def grasp_table(tpl, filt=False):
    """per template slot: the default grasp yaw if its opening fits the Franka hand, else the narrowest feasible yaw of GYAWS; NaN =
    no top-down grasp fits (generic geometry: the object's width along the grasp direction)."""
    from cleantables.rl.gripper import default_gyaw, opening
    T, K = len(tpl["templates"]), tpl["k_max"]
    G = np.full((T, K), np.nan, np.float32)
    for t, tt in enumerate(tpl["templates"]):
        for k, o in enumerate(tt["objects"]):
            ex = np.asarray(o["extents"], np.float64)                    # scene.json extents: the object as spawned (scale included)
            g0 = default_gyaw(ex)
            if not filt:                                                  # default: no width filter (box extents of curved /
                G[t, k] = g0; continue                                    # round objects overstate the grip width)
            if opening(ex, g0) <= MAX_OPEN:
                G[t, k] = g0
            else:
                best = min(GYAWS, key=lambda g: opening(ex, g))
                if opening(ex, best) <= MAX_OPEN:
                    G[t, k] = best
    return G


class BaselineDecider:
    """the non-learned policy: live WM object choice on the actual state + global_plan path for the chosen grasp, release at the goal."""

    def __init__(self, LP, judge, gtab, tpl_of_env, ik=None, goal="frozen", CS=None):
        self.LP, self.J, self.G, self.tpl_env, self.ik, self.goal = LP, judge, gtab, np.asarray(tpl_of_env), ik, goal
        self.CS = CS                                                    # with CS.arms: the IK filter (start and goal reachable)

    def decide(self, envs, P, yaw, busy, g=0):
        """envs (n,) needing a decision; P (N, K, 7) poses; yaw (N, K) offsets; busy (N, K) bool (carried by the other gripper)
        -> list per env of None ('done') or dict(carried, path (33, 3), dyaw, gyaw, rel_u)."""
        e = np.asarray(envs); tp = self.tpl_env[e]
        xyz = P[e, :, :3].copy(); yw = yaw[e].copy()
        I = xyz.shape[1]
        sc = self.J.score(np.nan_to_num(P[:, :, :3]), np.nan_to_num(yaw))
        met = sc["met"][e]
        gy = self.G[tp][:, :I]
        excl = met | busy[e] | ~np.isfinite(gy) | ~np.isfinite(xyz).all(-1) | (np.nan_to_num(xyz[..., 2]) < -OFF_Z)   # off the table: not takeable
        gxy = None
        tg_all = self.J.targets(np.nan_to_num(P[:, :, :3]), np.nan_to_num(yaw))[e]
        if self.goal == "frozen":                                        # the frozen start-frame target (xy), following its reference
            gxy = tg_all[..., :2]
        if self.CS is not None and self.CS.arms is not None:             # IK filter: this gripper's arm must reach the object and its goal
            top = np.nan_to_num(xyz).copy(); top[..., 2] += np.nan_to_num(self.CS.fp.ext.cpu().numpy()[e][..., 2])
            ok = self.CS.arms.reach(e, g, top, kind="start") & self.CS.arms.reach(e, g, tg_all, kind="target")
            excl = excl | ~ok
        out = self.LP.plan(tp, np.nan_to_num(xyz), np.nan_to_num(yw), excl, top_k=1, gyaw=gy, goal_xy=gxy)
        res = []
        for q in range(len(e)):
            a = int(out["obj"][q, 0])
            if bool(out["stop"][q]) or a < 0 or not np.isfinite(out["path"][q, 0].numpy()).all():
                res.append(None); continue
            res.append(dict(carried=a, path=out["path"][q, 0].numpy(), dyaw=float(out["dyaw"][q, 0]), gyaw=float(gy[q, a]), rel_u=1.0))
        return res


def run_episodes(fp, CS, decider, judge, layout, max_carries, max_s=90.0, disturb=0.0, rng=None, log=print, max_active=2, retreat=0.1, approach=True):
    """every env one episode from layout; both grippers until both are done, or max carries per env, or max_s, or every frozen goal met
    (early end). disturb: per carry, the chance of a random disturbance (a random object knocked, or the carry dropped mid-way)."""
    import torch
    rng = rng or np.random.default_rng(0)
    N, dev = fp.N, fp.dev
    CS.reset(layout)
    n_car = np.zeros(N, int); done_env = np.zeros(N, bool); t0 = time.time(); stop_kind = np.full(N, "", object)
    max_steps = int(max_s / CS.dt); n_dec_calls = 0; dec_s = 0.0; dist_ev = []; done_at = np.full((N, 2), -1); done_step = np.full(N, -1)
    t_step = t_tick = 0.0
    while CS.step_i < max_steps:
        if CS.step_i % TICK == 0:
            tk0 = time.time()
            ph = CS.phase.cpu().numpy()
            P = CS.poses(); Pn = P.cpu().numpy(); yw = CS.yaw_off(P).cpu().numpy()
            sc = judge.score(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw))
            allmet = sc["met"].all(1)
            idle_all = ((ph != 1) & (ph != 2) & (ph != 4)).all(1)
            fin = ~done_env & ((allmet & idle_all) | ((ph == 3).all(1)) | ((n_car >= max_carries) & idle_all))
            for ei in np.flatnonzero(fin):
                stop_kind[ei] = "all_met" if allmet[ei] else ("stop" if (ph[ei] == 3).all() else "max_carries"); done_step[ei] = CS.step_i
            done_env |= fin
            if done_env.all():
                break
            for g in (0, 1):
                need = ~done_env & (ph[:, g] == 0) & (n_car < max_carries)
                if max_active == 1:                                      # one gripper at a time: the other one waits until this one is idle
                    need &= (ph[:, 1 - g] != 1) & (ph[:, 1 - g] != 2) & (ph[:, 1 - g] != 4)
                need = np.flatnonzero(need)
                if not len(need):
                    continue
                busy = np.zeros((N, fp.K), bool)
                o_ = 1 - g; carrying = (ph[:, o_] == 1) | (ph[:, o_] == 2)
                busy[np.arange(N)[carrying], CS.car[:, o_].cpu().numpy()[carrying]] = True
                for (ei_, g_), pd_ in CS.pending.items():                 # the other gripper's object while it approaches it
                    if g_ == o_:
                        busy[ei_, pd_["c"]] = True
                td = time.time(); res = decider.decide(need, Pn, yw, busy); dec_s += time.time() - td; n_dec_calls += 1
                go = [(ei, r) for ei, r in zip(need, res) if r is not None]
                for ei, r in zip(need, res):
                    if r is None:
                        CS.phase[ei, g] = 3; done_at[ei, g] = CS.step_i
                if go:
                    e_ = np.array([x[0] for x in go])
                    CS.take(e_, g, [x[1]["carried"] for x in go], np.stack([x[1]["path"] for x in go]), [x[1]["dyaw"] for x in go],
                            [x[1]["gyaw"] for x in go], [x[1]["rel_u"] for x in go], retreat=float(retreat), approach=approach)
                    n_car[e_] += 1
                    ph[e_, g] = 1
                    if disturb > 0:
                        for ei in e_:
                            if rng.random() < disturb:
                                dist_ev.append((CS.step_i + int(rng.integers(20, 240)), int(ei), g, "drop" if rng.random() < 0.5 else "knock"))
                # a gripper that said done is asked again when the other gripper releases (the state changed)
            # a gripper that said done is asked again after any carry in its env has ended since (the state changed)
            ends = np.zeros(N); [ends.__setitem__(d_["env"], max(ends[d_["env"]], d_["end"])) for d_ in CS.ev if d_["end"] is not None]
            ph = CS.phase.cpu().numpy()
            re_ask = (ph == 3) & (ends[:, None] > done_at) & ~retired[:, None] & ~done_env[:, None]
            if re_ask.any():
                CS.phase[torch.as_tensor(re_ask, device=dev)] = 0
            t_tick += time.time() - tk0
        for d_ in [d for d in dist_ev if d[0] == CS.step_i]:
            _, ei, g, kind = d_
            if kind == "drop":
                CS.drop([ei], g)
            else:
                k = int(rng.integers(0, int(fp.valid[ei].sum())))
                ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(0.3, 0.8)
                CS.knock([ei], [k], [[sp * math.cos(ang), sp * math.sin(ang)]])
        ts0 = time.time(); CS.step(); t_step += time.time() - ts0
    wall = time.time() - t0
    for ei in np.flatnonzero(~done_env):
        stop_kind[ei] = "time"; done_step[ei] = CS.step_i
    return dict(n_carries=n_car, stop_kind=stop_kind, wall_s=wall, steps=CS.step_i, dec_calls=n_dec_calls, dec_s=dec_s, n_disturb=len(dist_ev),
                step_s=t_step, tick_s=t_tick,
                makespan_s=done_step * CS.dt)


class Recorder:
    """Every `every` steps of a small continuous run: object poses, hands (palm + parts), the arms' joint angles
    (an IK of each hand pose at the frame: top-down, else the mid-route tilt), phases, the frozen goals (they follow references) and met;
    plus every decision, hand route (learned edit and zero edit), carry, take outcome and disturbance. Dumped as one pickle for
    cleantables.rl.render (a kinematic replay; collisions are recomputed there from the recorded arm poses)."""

    def __init__(self, CS, judge, names, out, every=12):
        self.CS, self.J, self.names, self.out, self.every = CS, judge, names, out, every
        self.frames, self.decisions, self.dist, self.routes, self.q = [], [], [], [], None
        self.plans = []                                                  # plan visualisation: the WM's plan for every mover at each decision (wm_actor)

    def frame(self):
        import torch
        CS = self.CS; fp = CS.fp
        P = CS.poses(); Pn = P.cpu().numpy()
        xyz = np.nan_to_num(Pn[..., :3]); yw = np.nan_to_num(CS.yaw_off(P).cpu().numpy())
        sc = self.J.score(xyz, yw); tg = self.J.targets(xyz, yw)
        q = ok = None
        if CS.arms is not None:
            A = CS.arms; K = A.K
            tcp = CS.hP - fp.org[:, None] - torch.tensor([0, 0, A.TCP_DZ], device=fp.dev)
            loc = A.binv @ K.top_down(tcp, CS.hY)
            q0 = A.q if self.q is None else self.q
            q, ok, _, _ = K.ik(loc, q0, yaw_free=True, iters=30, n_seeds=2)
            if bool((~ok).any()):
                fe, fg = torch.nonzero(~ok, as_tuple=True)
                qt, okt, _, _ = K.ik(loc[fe, fg], q[fe, fg], yaw_free=True, iters=30, w_rot=0.1, tol_rot=A.route_tilt, n_seeds=2)
                q[fe, fg] = torch.where(okt[:, None], qt, q[fe, fg]); ok[fe, fg] = okt
            self.q = q; q = q.cpu().numpy(); ok = ok.cpu().numpy()
        hpa = getattr(CS, "hp_act", None) if FE.HAND_DYN else None                                 # (force-limited hands) the actual part poses
        parts = ((hpa if hpa is not None else CS.prev_hp) - fp.org[:, None, None]).cpu().numpy() if CS.prev_hp is not None else None
        excl = np.full((CS.N, 2), -1)
        ph = CS.phase.cpu().numpy(); car = CS.car.cpu().numpy()
        for g in (0, 1):
            m_ = (ph[:, g] == 1) | (ph[:, g] == 2); excl[m_, g] = car[m_, g]
        for (ei, g_), pd in CS.pending.items():
            excl[ei, g_] = pd["c"]
        self.frames.append(dict(step=CS.step_i, P=Pn.astype(np.float32), palm=(CS.hP - fp.org[:, None]).cpu().numpy(), yaw=CS.hY.cpu().numpy(), rp=CS.hRP.cpu().numpy(),
                                parts=parts, phase=ph, after=CS.after.cpu().numpy(), retract=(CS.arms.retract.cpu().numpy() if CS.arms is not None else None),
                                q=q, ik_ok=ok, met=sc["met"], dist_cm=sc["dist_cm"], static_moved_cm=sc["static_moved_cm"], goal=tg, excl=excl))

    def dump(self, eps):
        import pickle
        import torch
        CS = self.CS; fp = CS.fp
        cv = lambda x: x.cpu().numpy() if torch.is_tensor(x) else x                                     # noqa: E731
        ev = [{k: cv(v) for k, v in d.items() if k not in ("P0", "Prel", "Pend", "hh0", "hh1")} for d in CS.ev]
        mev = [{k: v for k, v in d.items() if k not in ("P0", "P1")} for d in CS.mev]
        T = fp.tpl["templates"]
        env = [dict(name=self.names[t], table=T[t]["table"], ids=[o["id"] for o in T[t]["objects"]], ext=np.asarray(T[t]["extents"], np.float32),
                    bases=(CS.arms.base[e].cpu().numpy() if CS.arms is not None else None)) for e, t in enumerate(fp.tpl_of_env)]
        pickle.dump(dict(dt=CS.dt, every=self.every, env=env, frames=self.frames, decisions=self.decisions, dist=self.dist, routes=self.routes, plans=self.plans,
                         ev=ev, mev=mev, takes=CS.takes, eps=eps, mov=self.J.mov, static=self.J.static, valid=fp.valid.cpu().numpy(),
                         bases_file=getattr(CS.arms, "bases_file", None)), open(self.out, "wb"))


def run_continuous(fp, CS, decider, judge, layout, max_carries, n_episodes, max_ep_s=90.0, disturb=0.0, rng=None, log=print, max_active=1,
                   retreat=0.1, approach=True, warm_eps=0, quota=0):
    """Continuous episodes, so no env waits for the slowest one: every env runs an episode; when it ends (every frozen goal met,
    both grippers done, the carry cap, or max_ep_s) its record is taken (judge score at that moment) and the env restarts at its start
    layout while the others keep going. Stops after n_episodes finished episodes. -> dict(episodes (list of per-episode dicts),
    wall_s, steady eps/s after warm_eps episodes, timing)."""
    import torch
    rng = rng or np.random.default_rng(0)
    N, dev = fp.N, fp.dev
    lay = torch.as_tensor(layout, dtype=torch.float32, device=dev)
    LRd = getattr(CS, "layout_rand", None)                                  # (FPC_LAYOUT_RAND) per-episode random layouts (training rollouts)
    if LRd is not None:
        lay = lay.clone()                                                   # (the caller's original layouts stay untouched)

    def _lr_next(envs_, epi_):
        """the layout of envs_' next episode (index epi_): random with p per (table, episode) in a training rollout, else the original;
        the frozen judge re-derived from the new start frame (or restored), in place."""
        if LRd is None or len(envs_) == 0 or (not LRd.active and not LRd.cur_rand[np.asarray(envs_, int)].any()):
            return                                                          # (e.g. quota 1)
        from cleantables.rl.layouts import SOLVABLE, rejudge_rows, solvable_draw
        envs_ = np.asarray(envs_, int)
        if LRd.active and SOLVABLE:                                          # one-arm-solvable random layouts: drawn, judged, checked,
            ln, rn, dyw = solvable_draw(LRd, judge, envs_, np.stack([np.full(len(envs_), LRd.ep_seed), np.asarray(epi_, np.int64)], 1))
            lay[torch.as_tensor(envs_, device=dev)] = torch.as_tensor(ln, device=dev)   # redrawn (the judge rows updated in place)
            return
        if LRd.active:
            ln, rn, dyw = LRd.draw(envs_, np.stack([np.full(len(envs_), LRd.ep_seed), np.asarray(epi_, np.int64)], 1))
        else:
            ln, rn, dyw = LRd.lay0[envs_].copy(), np.zeros(len(envs_), bool), np.zeros((len(envs_), LRd.lay0.shape[1]), np.float32)
            LRd.cur_rand[envs_] = False
        lay[torch.as_tensor(envs_, device=dev)] = torch.as_tensor(ln, device=dev)
        LRd.t_judge += rejudge_rows(judge, LRd.snap, envs_, rn, np.nan_to_num(ln[..., :3] - np.array([0, 0, 0.002], np.float32)), LRd.yaw0[envs_] + dyw,
                                    LRd.LPa, LRd.temb_T)
    _lr_next(np.arange(N), np.zeros(N, np.int64))
    CS.reset(lay)
    n_car = np.zeros(N, int); done_at = np.full((N, 2), -1); ep_start = np.zeros(N, int); t0 = time.time()
    eps, dist_ev, n_dec_calls, dec_s, t_step, t_tick = [], [], 0, 0.0, 0.0, 0.0
    t_warm = None; max_steps_ep = int(max_ep_s / CS.dt)
    n_done = np.zeros(N, int); retired = np.zeros(N, bool)                 # quota > 0 (unbiased evaluation): an env runs exactly quota episodes
    CS.dist_step = np.full(N, -10 ** 9, np.int64)                           # each env's last test knock / drop step (push cost)
    dobj = [set() for _ in range(N)]; dist_at = {}; n_dist_done = 0         # disturbed objects of each env's current episode (recovery rate)
    dist_base = int(rng.integers(1 << 30))                                  # the seeded schedule's base for this run
    blamed = set()                                                          # robot-lost objects already blamed on a take
    dstep, excused = {}, set()                                              # each disturbed object's disturbance step; excused falls
    while len(eps) < n_episodes and not retired.all():
        if CS.step_i % TICK == 0:
            tk0 = time.time(); n_eps0 = len(eps)
            ph = CS.phase.cpu().numpy()
            P = CS.poses(); Pn = P.cpu().numpy(); yw = CS.yaw_off(P).cpu().numpy()
            sc = judge.score(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw))
            # an object a disturbance threw off the table (below the top by > OFF_Z) is lost: not scored, not takeable
            lost = np.zeros(sc["met"].shape, bool)
            for ei_ in np.flatnonzero([len(x) > 0 for x in dobj]):
                for k_ in dobj[ei_]:
                    if k_ < lost.shape[1] and Pn[ei_, k_, 2] < -OFF_Z:
                        # excused only if it left the table within EXCUSE_S of its disturbance (sticky once excused)
                        if not EXCUSE_WIN or (ei_, k_) in excused or CS.step_i - dstep.get((ei_, k_), -10 ** 9) <= int(EXCUSE_S / CS.dt):
                            lost[ei_, k_] = True; excused.add((ei_, k_))
            allmet = (sc["met"] | lost).all(1)
            idle_all = ((ph != 1) & (ph != 2) & (ph != 4)).all(1) & (ph != 5).all(1)
            over = (CS.step_i - ep_start) >= max_steps_ep
            n_cap = n_car - (CS.n_ab_env if ABORT_FREE else 0)              # aborted takes don't use up the carry cap (the 90 s limit bounds them)
            fin = (idle_all & (allmet | (ph == 3).all(1) | (n_cap >= max_carries)) | over) & ~retired
            if TERM_LOST:                                                     # an object the robot put off the table ends the episode
                offr = (np.nan_to_num(Pn[:, :lost.shape[1], 2], nan=0.0) < -OFF_Z) & fp.valid.cpu().numpy()[:, :lost.shape[1]] & ~lost
                term_lost = offr.any(1) & ~retired
                for ei_, k_ in zip(*np.nonzero(offr)):                        # blame: the takes active when it left the table
                    key_ = (int(ei_), int(k_), int(ep_start[ei_]))
                    if key_ in blamed:
                        continue
                    blamed.add(key_)
                    _ph2 = [int(ph[ei_, g2_]) for g2_ in (0, 1)]; _car2 = [int(CS.car[ei_, g2_]) for g2_ in (0, 1)]   # (logging only)
                    kd_ = ("held" if any(p2_ == 1 and c2_ == int(k_) for p2_, c2_ in zip(_ph2, _car2)) else            # how it left the table:
                           "release" if any(p2_ == 2 and c2_ == int(k_) for p2_, c2_ in zip(_ph2, _car2)) else         # the carried object itself,
                           "knock_carry" if 1 in _ph2 else "knock_reach" if 4 in _ph2 else "other")                    # or another one knocked
                    CS.__dict__.setdefault("lost_kinds", {})[kd_] = CS.__dict__.setdefault("lost_kinds", {}).get(kd_, 0) + 1
                    for g_ in (0, 1):
                        t_ = CS.tk_open.get((int(ei_), g_))
                        if t_ is not None and int(ph[ei_, g_]) in (1, 2, 4):
                            CS.takes[t_]["lost"] = CS.takes[t_].get("lost", 0) + 1
                fin = fin | term_lost
            else:
                term_lost = np.zeros(N, bool)
            if fin.any():
                # robot-caused losses (not excused): a moving object the WM moves that the robot put off the
                # table, or out of every arm's top-down reach (and >= 5 cm from where it started), and that no disturbance touched
                lost_r = np.zeros(sc["met"].shape, bool)
                fe_ = np.flatnonzero(fin)
                movm = judge.mov[fe_] & ~judge.static[fe_]
                off_ = Pn[fe_, :lost_r.shape[1], 2] < -OFF_Z
                cand = movm & ~sc["met"][fe_]
                if CS.arms is not None and cand.any():
                    top_ = np.nan_to_num(Pn[fe_, :, :3]).copy(); top_[..., 2] += np.nan_to_num(fp.ext.cpu().numpy()[fe_][..., 2])
                    rch = CS.arms.reach(fe_, 0, top_, kind="end") | CS.arms.reach(fe_, 1, top_, kind="end")
                    moved_ = np.linalg.norm(Pn[fe_, :, :2] - lay[fe_, :, :2].cpu().numpy(), axis=-1) > 0.05
                    unreach = ~rch[:, :lost_r.shape[1]] & moved_[:, :lost_r.shape[1]]
                else:
                    unreach = np.zeros_like(off_)
                vld_ = fp.valid.cpu().numpy()[fe_][:, :lost_r.shape[1]]
                for r_, ei_ in enumerate(fe_):
                    # any valid object off the table (a moved one, a T1-static one or a reference, see FREEZE_REF) or a not-met mover out of reach
                    for k_ in np.flatnonzero((vld_[r_] & off_[r_]) | (cand[r_] & unreach[r_])):
                        if k_ not in dobj[ei_]:
                            lost_r[ei_, k_] = True
                if MET3:                                                      # met_3d at the episode end
                    ve3 = fp.wp.to_torch(fp.OV.get_velocities()).reshape(fp.OV.count, 6)[fp.IDX.clamp(min=0)][..., :3].cpu().numpy()
                    m3 = judge.met3(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw), fp.ext.cpu().numpy(), vel=ve3, dy0=getattr(judge, "dy0", None))
                    if MET_NEW:                                                   # the new met replaces met_3d; the old one is logged
                        m3_old = m3
                        m3, mstay, mdisp = met_new(judge, CS, np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw), ve3, np.flatnonzero(fin))
                if MET3 and os.environ.get("FPC_PEN_DIAG") == "1":            # diagnostic: why met_xy objects fail met_3d (z / rest / yaw), per mover
                    xyz_ = np.nan_to_num(Pn[:, :, :3]); yw_ = np.nan_to_num(yw)
                    for ei in np.flatnonzero(fin):
                        J_ = judge.J[int(np.asarray(CS.inst).reshape(-1)[ei])] if hasattr(judge, "J") else judge
                        I_ = J_.Imax; sc_ = J_.score(xyz_, yw_); tg_ = J_.targets(xyz_, yw_); top_ = J_._top(xyz_[:, :I_]); H_ = J_._geo_consts()[0]
                        nm_ = fp.tpl["templates"][int(fp.tpl_of_env[ei])]["objects"]
                        for k_ in np.flatnonzero(J_.mov[ei] & ~J_.static[ei] & ~lost[ei][:I_]):
                            jr = int(J_.j[ei, k_]); sp_ = float(np.linalg.norm(ve3[ei, k_]))
                            dy_ = getattr(J_, "dy0", None)
                            err_y = None if dy_ is None else float(np.degrees(abs((yw_[ei, k_] - J_.start_yaw[ei, k_] - dy_[ei, k_] + np.pi / 2) % np.pi - np.pi / 2)))
                            self_ = CS.__dict__.setdefault("met3_diag", [])
                            self_.append(dict(env=int(ei), k=int(k_), obj=str(nm_[k_].get("id", k_)), ref=(str(nm_[jr].get("id", jr)) if 0 <= jr < len(nm_) else None),
                                              fixed=bool(J_.is_fixed[ei, k_]), met_xy=bool(sc_["met"][ei, k_]), met3=bool(m3[ei, k_]),
                                              **(dict(met3_old=bool(m3_old[ei, k_]), stays=bool(mstay[ei, k_]), pred_move_cm=float(mdisp[ei, k_]),
                                                      xy_err_cm=float(np.linalg.norm((J_.end0[ei, k_] - top_[ei, k_])[:2]) * 100),
                                                      tol_cm=float(np.maximum(J_.sd[ei, k_] * 100, MET_CM).max())) if MET_NEW else {}),
                                              top_z=float(top_[ei, k_, 2]), goal_z=float(tg_[ei, k_, 2]), dz_cm=float((top_[ei, k_, 2] - tg_[ei, k_, 2]) * 100),
                                              sdz_cm=float(J_.sdz[ei, k_] * 100), H_cm=float(H_[ei, k_] * 100),
                                              ref_top_start=(float(J_.start_xyz[ei, jr, 2] + H_[ei, jr]) if 0 <= jr < I_ else None),
                                              ref_H_cm=(float(H_[ei, jr] * 100) if 0 <= jr < I_ else None), off_z=float(J_.off[ei, k_, 2] * J_.szj[ei, k_, 2] * 100),
                                              speed=sp_, dy0_deg=(None if dy_ is None else float(np.degrees(dy_[ei, k_]))), yaw_err_deg=err_y,
                                              bottom_z=float(xyz_[ei, k_, 2])))
                for ei in np.flatnonzero(fin):
                    kind = "lost_robot" if term_lost[ei] else "time" if over[ei] and not idle_all[ei] else ("all_met" if allmet[ei] else ("stop" if (ph[ei] == 3).all() else "max_carries"))
                    m_ = judge.mov[ei] & ~judge.static[ei] & ~lost[ei]
                    _cf = (cont_check(CS.cont_meta, int(decider.tpl_env[ei]), bool(np.asarray(CS.inst).reshape(-1)[ei]) if hasattr(CS, "inst") else False,
                                      Pn[ei], lay[ei].cpu().numpy()) if (CONT_EVAL and getattr(CS, "cont_meta", None)) else {})   # (evaluation only)
                    eps.append(dict(**_cf, env=int(ei), tpl=int(decider.tpl_env[ei]), s0=int(ep_start[ei]), s1=int(CS.step_i), stop=kind, n_lost=int(lost[ei].sum()), n_lost_robot=int(lost_r[ei].sum()),
                                    n_unmet=int((~sc["met"][ei] & judge.mov[ei] & ~judge.static[ei] & ~lost[ei]).sum()),
                                    static_moved_cm=float(sc["static_moved_cm"][ei].sum()),
                                    # per-mover progress 1 - d / d0 (uncapped: moving away is negative), lost (excused) movers left out
                                    progress=float((1 - sc["dist_raw_cm"][ei] / np.maximum(judge.d0_cm[ei], 1.0))[m_].sum()), n_movers=int(m_.sum()),
                                    hand_hand_s=float(CS.hh_steps[ei]) * CS.dt,
                                    **(dict(met3=float(m3[ei][m_].mean()) if m_.any() else 1.0, n_met3=int(m3[ei][m_].sum()), all_met3=bool((m3[ei] | lost[ei]).all())) if MET3 else {}), carries=int(n_car[ei]), carries_ok=int(n_car[ei] - CS.n_ab_env[ei]),
                                    **(dict(met3_old=float(m3_old[ei][m_].mean()) if m_.any() else 1.0, all_met3_old=bool((m3_old[ei] | lost[ei]).all()),
                                            stay_movers=float(mstay[ei][m_].mean()) if m_.any() else 1.0) if (MET3 and MET_NEW) else {}),
                                    makespan_s=(CS.step_i - ep_start[ei]) * CS.dt, goal_err_cm=float(sc["dist_cm"][ei][m_].mean()) if m_.any() else 0.0,
                                    met=float(sc["met"][ei][m_].mean()) if m_.any() else 1.0, all_met=bool(allmet[ei]),
                                    static_moved_gt1=int((sc["static_moved_cm"][ei] > 1).sum()), hh_steps=int(CS.hh_steps[ei]),
                                    finger_table_s=float(CS.ft_ep[ei]) * CS.dt,
                                    hand_obj_aborts=int(CS.hpen_ep[ei]) if hasattr(CS, "hpen_ep") else 0,
                                    hand_obj_overlap_s=float(CS.hov_ep[ei]) * CS.dt if hasattr(CS, "hov_ep") else 0.0,
                                    # recovery: a disturbed object is back at its frozen goal at the episode end (judge met; a T1-static one
                                    # back within MET_CM of its start)
                                    n_dist=len(dobj[ei]), n_dist_rec=int(sum(bool(sc["met"][ei][k]) and sc["static_moved_cm"][ei][k] <= MET_CM
                                                                             for k in dobj[ei])),
                                    **({} if CS.arms is None else dict(ik_fail_s=float(CS.arms.n_ik[ei]) * CS.dt, arm_obj_s=float(CS.arms.n_obj[ei]) * CS.dt,
                                                                      arm_arm_s=float(CS.arms.n_arm[ei]) * CS.dt, arm_table_s=float(CS.arms.n_tab[ei]) * CS.dt,
                                                                      near_s=float(CS.arms.n_near[ei]) * CS.dt, arm_touch_s=float(CS.arms.n_touch[ei]) * CS.dt))))
                    if t_warm is None and len(eps) >= warm_eps:
                        t_warm = (time.time(), len(eps))
                n_done[fin] += 1
                if quota > 0:
                    newly = fin & (n_done >= quota); retired |= newly; fin = fin & ~newly
                    if newly.any():
                        CS.phase[torch.as_tensor(np.flatnonzero(newly), device=dev)] = 3
                for ei in np.flatnonzero(fin | retired):
                    dobj[ei].clear()
                    for kk in [x for x in excused if x[0] == ei]:
                        excused.discard(kk)
                fe = np.flatnonzero(fin)
                if INSTR_P > 0 and hasattr(CS, "inst"):                       # (instruction sampling) the next episode's instruction, per env:
                    CS.inst[fe] = rng.random(len(fe)) < INSTR_P               # the other instruction with probability INSTR_P
                _lr_next(fe, n_done[fe])                                     # the next episode's layout (+ judge) when random layouts are on
                CS.reset_envs(fe, lay[torch.as_tensor(fe, device=dev)]); CS.hh_steps[torch.as_tensor(fe, device=dev)] = 0
                CS.ft_ep[torch.as_tensor(fe, device=dev)] = 0
                if hasattr(CS, "hpen_ep"):
                    CS.hpen_ep[torch.as_tensor(fe, device=dev)] = 0; CS.hov_ep[torch.as_tensor(fe, device=dev)] = 0
                n_car[fe] = 0; done_at[fe] = -1; ep_start[fe] = CS.step_i
                ph = CS.phase.cpu().numpy()
            for g in (0, 1):
                need = (ph[:, g] == 0) & ((n_car - (CS.n_ab_env if ABORT_FREE else 0)) < max_carries) & ~retired
                if max_active == 1:
                    need &= (ph[:, 1 - g] != 1) & (ph[:, 1 - g] != 2) & (ph[:, 1 - g] != 4)
                need = np.flatnonzero(need)
                if not len(need):
                    continue
                busy = np.zeros((N, fp.K), bool)
                o_ = 1 - g; carrying = (ph[:, o_] == 1) | (ph[:, o_] == 2)
                busy[np.arange(N)[carrying], CS.car[:, o_].cpu().numpy()[carrying]] = True
                for (ei_, g_), pd_ in CS.pending.items():
                    if g_ == o_:
                        busy[ei_, pd_["c"]] = True
                td = time.time()
                if hasattr(decider, "set_ctx"):
                    decider.set_ctx(n_car, max_carries, ep_start)
                res = decider.decide(need, Pn, yw, busy, g=g); dec_s += time.time() - td; n_dec_calls += 1
                if getattr(CS, "rec", None) is not None:
                    for ei, r in zip(need, res):
                        CS.rec.decisions.append(dict(step=CS.step_i, env=int(ei), g=g, type="done" if r is None else r.get("type", "take"),
                                                     obj=None if r is None else r.get("carried"), gyaw=None if r is None else r.get("gyaw"),
                                                     P=None if r is None else r.get("P"), rel_u=None if r is None else r.get("rel_u")))
                go = [(ei, r) for ei, r in zip(need, res) if r is not None and r.get("type", "take") == "take"]
                for ei, r in zip(need, res):
                    if r is None:
                        CS.phase[ei, g] = 3; done_at[ei, g] = CS.step_i
                mv_ = [(ei, r) for ei, r in zip(need, res) if r is not None and r.get("type") == "move"]
                wt_ = [ei for ei, r in zip(need, res) if r is not None and r.get("type") == "wait"]
                if mv_:
                    CS.move(np.array([x[0] for x in mv_]), g, np.stack([x[1]["P"] for x in mv_]), np.array([x[1]["Y"] for x in mv_], np.float32),
                            resid=np.stack([x[1]["resid"] for x in mv_]) if "resid" in mv_[0][1] else None, route=MOVE_ROUTE,
                            rot=np.stack([x[1]["rot"] for x in mv_]) if "rot" in mv_[0][1] else None)
                if wt_:                                                   # wait = hold the hand where it is (0.3 s), then decide again
                    w_ = np.array(wt_); cur = (CS.hP[torch.as_tensor(w_, device=dev), g] - fp.org[torch.as_tensor(w_, device=dev)]).cpu().numpy()
                    CS.move(w_, g, cur, CS.hY[torch.as_tensor(w_, device=dev), g].cpu().numpy())
                    if FLUID_WAIT:                                        # the controller may move this waiting hand (fluid waiting)
                        CS.is_wait[torch.as_tensor(w_, device=dev), g] = True
                if go:
                    e_ = np.array([x[0] for x in go])
                    CS.take(e_, g, [x[1]["carried"] for x in go], np.stack([x[1]["path"] for x in go]), [x[1]["dyaw"] for x in go],
                            [x[1]["gyaw"] for x in go], [x[1]["rel_u"] for x in go],
                            retreat=torch.tensor([float(x[1].get("retreat", retreat)) for x in go], device=dev), approach=approach,
                            app_resid=np.stack([x[1]["app_resid"] for x in go]) if "app_resid" in go[0][1] else None,
                            app_rot=np.stack([x[1]["app_rot"] for x in go]) if "app_rot" in go[0][1] else None,
                            carry_rot=np.stack([x[1]["carry_rot"] for x in go]) if "carry_rot" in go[0][1] else None)
                    n_car[e_] += 1; ph[e_, g] = 4 if approach else 1
                    if disturb > 0:
                        for ei in e_:
                            # DIST_SEEDED: the draw is keyed by (template, the env's episode number, its take number), so the
                            # copies of one table in a GRPO group share the disturbance schedule (take for take)
                            rr = (np.random.default_rng([dist_base, int(decider.tpl_env[ei]), int(n_done[ei]), int(n_car[ei])]) if DIST_SEEDED else rng)
                            u_, dl_, kd_, kk_, an_, sp_ = rr.random(), int(rr.integers(20, 240)), rr.random(), rr.random(), rr.uniform(0, 2 * math.pi), rr.uniform(*KNOCK_V)
                            if u_ < disturb:
                                d_ = (CS.step_i + dl_, int(ei), g, "drop" if kd_ < 0.5 else "knock", int(ep_start[ei]), kk_, an_, sp_)
                                dist_ev.append(d_); dist_at.setdefault(d_[0], []).append(d_)
            ends = np.zeros(N); [ends.__setitem__(d_["env"], max(ends[d_["env"]], d_["end"])) for d_ in CS.ev[-4 * N:] if d_["end"] is not None]
            ph = CS.phase.cpu().numpy()
            re_ask = (ph == 3) & (ends[:, None] > done_at) & ~retired[:, None]
            if re_ask.any():
                CS.phase[torch.as_tensor(re_ask, device=dev)] = 0
            if getattr(CS, "lowpol", None) is not None:                     # the shared two-arm controller: score the last 0.1 s, act for the next
                CS.lowpol.tick(CS, judge, sc, lost, eps[n_eps0:], retired, ep_start)
            t_tick += time.time() - tk0
        for d_ in dist_at.pop(CS.step_i, []):
            _, ei, g, kind, es, kk_, an_, sp_ = d_
            if es != ep_start[ei] or retired[ei]:                           # its episode already ended: no disturbance
                continue
            CS.dist_step[ei] = CS.step_i                                    # the low level is not charged for a test knock / drop
            if kind == "drop":
                if int(CS.phase[ei, g]) != 1:                                # only a carry still held can be dropped
                    continue
                k_ = CS.ev_open.get((ei, g))
                if k_ is not None:
                    CS.ev[k_]["disturbed"] = True                            # its carry result is not the policy's doing
                dobj[ei].add(int(CS.car[ei, g])); dstep[(int(ei), int(CS.car[ei, g]))] = CS.step_i; CS.drop([ei], g)
                if getattr(CS, "rec", None) is not None:
                    CS.rec.dist.append(dict(step=CS.step_i, env=int(ei), kind="drop", obj=int(CS.car[ei, g]), g=g))
            else:
                k = min(int(kk_ * int(fp.valid[ei].sum())), int(fp.valid[ei].sum()) - 1)
                ang = an_; sp = sp_
                if KNOCK_INWARD:                                         # KNOCK_INWARD: a knock never points off the table, within +-90 deg of the
                    tb_ = fp.tpl["templates"][fp.tpl_of_env[ei]]["table"]              # direction to the table centre (every object the same)
                    c_ = np.array([np.mean(tb_["x"]), np.mean(tb_["y"])]) - Pn[ei, k, :2]
                    a_c = math.atan2(c_[1], c_[0]); ang = a_c + rng.uniform(-math.pi / 2, math.pi / 2)
                dobj[ei].add(k); dstep[(int(ei), int(k))] = CS.step_i; CS.knock([ei], [k], [[sp * math.cos(ang), sp * math.sin(ang)]])
                if getattr(CS, "rec", None) is not None:
                    CS.rec.dist.append(dict(step=CS.step_i, env=int(ei), kind="knock", obj=k, vel=[sp * math.cos(ang), sp * math.sin(ang)]))
            n_dist_done += 1
        if getattr(CS, "rec", None) is not None and CS.step_i % CS.rec.every == 0:
            CS.rec.frame()
        ts0 = time.time(); CS.step(); t_step += time.time() - ts0
    wall = time.time() - t0
    if getattr(CS, "rec", None) is not None:
        CS.rec.dump(eps)
    steady = (len(eps) - t_warm[1]) / max(time.time() - t_warm[0], 1e-9) if t_warm is not None and len(eps) > t_warm[1] else len(eps) / wall
    return dict(episodes=eps, wall_s=wall, eps_per_s=len(eps) / wall, eps_per_s_steady=steady, steps=CS.step_i, dec_calls=n_dec_calls, dec_s=dec_s,
                step_s=t_step, tick_s=t_tick, n_disturb=n_dist_done, n_disturb_sched=len(dist_ev))


def move_pushes(CS, fp):
    """per finished hand move (approach / free): the largest move of any object (not carried by the other gripper) during it (cm)."""
    valid = fp.valid.cpu().numpy(); out = []
    for mi, m in enumerate(CS.mev):
        if m["P1"] is None:
            continue
        e = m["env"]; P0, P1 = m["P0"].cpu().numpy(), m["P1"].cpu().numpy()
        busy = {d["carried"] for d in CS.ev if d["env"] == e and d["s0"] < m["s1"] and (d["end"] or 1e9) > m["s0"]}
        ks = [k for k in range(fp.K) if valid[e, k] and k not in busy and np.isfinite(P0[k]).all() and np.isfinite(P1[k]).all()]
        out.append(dict(mev_i=mi, env=e, g=m["g"], kind=m["kind"], push_cm=100 * max([float(np.linalg.norm(P1[k, :3] - P0[k, :3])) for k in ks], default=0.0),
                        dur_s=(m["s1"] - m["s0"]) * fp.dt, s0=m["s0"]))
    return out


def carry_metrics(CS, fp, ev):
    """per finished carry: push while held (cm: the largest move of any other object, not the other gripper's object, not riding on the
    carried one, from the carry start to the release), the carried object's end error vs its planned goal (cm, xy), the hand overlap."""
    ext = fp.ext.cpu().numpy(); valid = fp.valid.cpu().numpy(); out = []
    for di, d in enumerate(ev):
        if d["Prel"] is None:
            continue
        e, a = d["env"], d["carried"]; P0 = d["P0"].cpu().numpy(); P1 = d["Prel"].cpu().numpy()
        oth = [k for k in range(fp.K) if valid[e, k] and k != a]
        ride = [k for k in oth if np.linalg.norm(P0[k, :2] - P0[a, :2]) < 0.5 * ext[e, a, :2].max() and P0[k, 2] > P0[a, 2] + 0.005]
        others_moving = {x["carried"] for x in ev if x["env"] == e and x is not d and x["s0"] < (d["rel"] or 1e9) and (x["end"] or 1e9) > d["s0"]}
        cnt = [k for k in oth if k not in ride and k not in others_moving]
        push = 100 * max([float(np.linalg.norm(P1[k, :3] - P0[k, :3])) for k in cnt if np.isfinite(P0[k]).all() and np.isfinite(P1[k]).all()], default=0.0)
        Pe = d["Pend"].cpu().numpy() if d["Pend"] is not None else P1
        err = 100 * float(np.linalg.norm(Pe[a, :2] - d["goal"].cpu().numpy()[:2]))
        from cleantables.rl.gripper import opening
        pth_ = d["path"].cpu().numpy() if "path" in d else None
        hhs = int(d["hh1"] - d["hh0"]) if d.get("hh1") is not None else 0                          # steps the two hands overlapped during it
        out.append(dict(ev_i=di, env=e, g=d["g"], carried=a, s0=d["s0"], push_held_cm=push, end_err_cm=err, disturbed=bool(d.get("disturbed", False)), hand_hand_cm=float(hhs > 0), hand_hand_s=hhs * fp.dt,
                        open_cm=100 * opening(ext[e, a], d["gyaw"]), relu=d["relu"],
                        lift_cm=float(100 * (pth_[:, 2].max() - pth_[0, 2])) if pth_ is not None else float("nan"),
                        dur_s=((d["rel"] or d["s0"]) - d["s0"]) * fp.dt))
    return out


# =====================================================================================================================================
def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="heldout"); ap.add_argument("--copies", type=int, default=8); ap.add_argument("--dt", type=int, default=120)
    ap.add_argument("--disturb", type=float, default=0.0); ap.add_argument("--max_carries", type=int, default=0); ap.add_argument("--max_s", type=float, default=90.0)
    ap.add_argument("--tag", default="m1_baseline"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grasp_filter", action="store_true", help="exclude objects whose box width along every top-down grasp yaw exceeds 8 cm")
    ap.add_argument("--goal", default="frozen", choices=["frozen", "live"], help="baseline goal: the frozen judge's target (xy) or the live WM's")
    ap.add_argument("--max_active", type=int, default=2, help="grippers carrying at the same time (1 = take turns)")
    ap.add_argument("--retreat", type=float, default=0.1, help="baseline: the hand rises this far (m) after each release")
    ap.add_argument("--continuous", type=int, default=0, help="continuous episodes (auto-reset per env): stop after this many episodes")
    ap.add_argument("--arms", action="store_true", help="franka_kin arms behind the hands (home poses, IK, sphere clearances)")
    ap.add_argument("--no_approach", action="store_true", help="no hand approach move before a take (the old timing)")
    add_launcher_args(ap)
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    rd = W / "fastpath" / f"chain_{a.tag}"; rd.mkdir(parents=True, exist_ok=True)
    lf = rd / "log.txt"

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"; print(line, flush=True); open(lf, "a").write(line + "\n")
    twr = os.environ["WM_TWIN_ROOT"]
    names = json.load(open(W / "fastpath" / "scope_clean.json"))[a.split]
    names = [n for n in names if (Path(twr) / n / "wm_tokens.npz").exists()]
    if os.environ.get("FPC_SCENES"):                                     # (videos) only these scenes
        names = [n for n in os.environ["FPC_SCENES"].split(",") if n in names]
    if a.limit:
        names = names[:a.limit]
    if os.environ.get("FPC_LAYOUT_OVERRIDE"):
        _ex = set(json.load(open(os.environ["FPC_LAYOUT_OVERRIDE"])).get("excluded", []))
        names = [n for n in names if n not in _ex]
    tpl = FE.build_templates(names, twin_root=twr, out=rd / "templates.pt", meta=dict(split={n: a.split for n in names}))
    from cleantables.rl.layout_override import apply as _lo_apply
    _lo_apply(tpl, os.environ.get("FPC_LAYOUT_OVERRIDE"), log=print)
    N = len(names) * a.copies
    cfg = FE.make_cfg(tpl, N, 1 / a.dt)
    with launch_simulation(cfg, a):
        fp = FE.FastPath(cfg)
        from cleantables.worldmodel.physics.live_plan import LivePlanner
        LP = LivePlanner(names, device=str(fp.dev))
        tenv = np.asarray(fp.tpl_of_env)
        lay = fp.lay0.clone()
        xyz0 = (lay[..., :3] - torch.tensor([0, 0, 0.002], device=fp.dev)).cpu().numpy()
        yaw0 = np.zeros((N, fp.K), np.float32)
        for e in range(N):
            tt = tpl["templates"][tenv[e]]
            yaw0[e, :tt["n"]] = [o["yaw"] for o in tt["objects"]]
        t0 = time.time(); J = FrozenJudge(LP, tenv, np.nan_to_num(xyz0), yaw0); log(f"frozen judge: {time.time() - t0:.1f} s; "
            f"movable {int(J.mov.sum())}, T1-static among them {int((J.static & J.mov).sum())}, fixed references {int((J.is_fixed & J.mov).sum())}; "
            f"tolerance sd (cm, xy, non-static) median {np.median(100 * J.sd[J.mov & ~J.static], 0).round(1).tolist()} p90 {np.percentile(100 * J.sd[J.mov & ~J.static], 90, 0).round(1).tolist()}")
        gtab = grasp_table(tpl, filt=a.grasp_filter)
        gw = grasp_table(tpl, filt=True)
        log(f"grasp width filter {'ON' if a.grasp_filter else 'off (reported only)'}: box width fits {MAX_OPEN * 100:.0f} cm for {int(np.isfinite(gw).sum())} of "
            f"{sum(t['n'] for t in tpl['templates'])} objects")
        CS = ChainSim(fp)
        if a.arms:
            CS.arms = ArmTracker(CS, names, every=int(os.environ.get("FPC_ARM_EVERY", "12")))
        dec = BaselineDecider(LP, J, gtab, tenv, goal=a.goal, CS=CS)
        mc = a.max_carries or int(max(t["n"] for t in tpl["templates"]) + 2)
        if a.continuous:
            if os.environ.get("FPC_REC_OUT"):
                CS.rec = Recorder(CS, J, names, os.environ["FPC_REC_OUT"], every=int(os.environ.get("FPC_REC_EVERY", "12")))
            rc = run_continuous(fp, CS, dec, J, lay, mc, a.continuous, a.max_s, a.disturb, log=log, max_active=a.max_active, retreat=a.retreat,
                                approach=not a.no_approach, warm_eps=N // 2, quota=int(os.environ.get("FPC_QUOTA", "0")))
            E = rc.pop("episodes"); cm = carry_metrics(CS, fp, CS.ev); mvp = move_pushes(CS, fp)
            fam = np.array([names[e["tpl"]].split("_")[0] for e in E])
            def summ_c(sel):
                E_ = [e for e, s_ in zip(E, sel) if s_]
                return dict(episodes=len(E_), goal_err_cm_mean=float(np.mean([e["goal_err_cm"] for e in E_])), goals_met_share=float(np.mean([e["met"] for e in E_])),
                            all_met_share=float(np.mean([e["all_met"] for e in E_])), carries_per_ep=float(np.mean([e["carries"] for e in E_])),
                            makespan_s_median=float(np.median([e["makespan_s"] for e in E_])), static_moved_gt1_per_ep=float(np.mean([e["static_moved_gt1"] for e in E_])),
                            hand_hand_ep_share=float(np.mean([e["hh_steps"] > 0 for e in E_])),
                            stop_kinds={k: int(sum(e["stop"] == k for e in E_)) for k in ("all_met", "stop", "max_carries", "time")},
                            wrong_stop_share=float(np.mean([e["stop"] == "stop" and not e["all_met"] for e in E_])),
                            **({k: float(np.mean([e[k] > 0 for e in E_])) for k in ("ik_fail_s", "arm_obj_s", "arm_arm_s", "arm_table_s") if E_ and k in E_[0]}))
            res = dict(tag=a.tag, mode="continuous", N=N, templates=len(names), copies=a.copies, goal=a.goal, max_active=a.max_active, retreat=a.retreat,
                       approach=not a.no_approach, disturb=a.disturb, **rc, lp_wm_s=float(sum(x["wm_s"] for x in LP.timing)),
                       lp_planner_s=float(sum(x["planner_s"] for x in LP.timing)),
                       carries=dict(n=len(cm), push_held_gt1cm_share=float(np.mean([c["push_held_cm"] > 1 for c in cm])) if cm else None,
                                    end_err_cm_median=float(np.median([c["end_err_cm"] for c in cm])) if cm else None,
                                    lift_cm_median=float(np.median([c["lift_cm"] for c in cm])) if cm else None,
                                    hand_move_push_gt1cm_share=float(np.mean([m["push_cm"] > 1 for m in mvp])) if mvp else None),
                       all=summ_c(np.ones(len(E), bool)), by_family={f: summ_c(fam == f) for f in sorted(set(fam))})
            json.dump(res, open(rd / "baseline.json", "w"), indent=1, default=float); json.dump(E, open(rd / "episodes.json", "w"), default=float)
            if CS.arms is not None:                                       # take outcomes + (FPC_IK_DIAG) every failed IK check
                ab = {}
                for t_ in CS.takes:
                    ab[str(t_["abort"])] = ab.get(str(t_["abort"]), 0) + 1
                log(f"TAKES {len(CS.takes)}: {ab}; reflex stops {CS.n_reflex}; IK diag rows {len(getattr(CS.arms, 'diag', []))}")
                if getattr(CS, "pen_diag", None):
                    import collections as _co
                    pdg = CS.pen_diag
                    log("PEN DIAG: " + str(dict(n=len(pdg), by_phase=_co.Counter((x["phase"], x["after"]) for x in pdg), target=sum(x["target"] for x in pdg),
                                                 depth_med=float(np.median([x["depth_cm"] for x in pdg])), u=np.histogram([x["u"] for x in pdg if x["u"] is not None], bins=[0, .2, .4, .6, .8, 1.01])[0].tolist(),
                                                 palm_z_med=float(np.median([x["palm_z"] for x in pdg])))))
                    json.dump(pdg, open(rd / "pen_diag.json", "w"))
                json.dump(dict(takes=ab, diag=getattr(CS.arms, "diag", [])), open(rd / "ik_diag.json", "w"), default=float)
            log("CONTINUOUS " + json.dumps({k: v for k, v in res.items() if k != "by_family"}, default=float))
            return
        r = run_episodes(fp, CS, dec, J, lay, mc, a.max_s, a.disturb, log=log, max_active=a.max_active, retreat=a.retreat, approach=not a.no_approach)
        mvp = move_pushes(CS, fp)
        P = CS.poses(); Pn = P.cpu().numpy(); yw = CS.yaw_off(P).cpu().numpy()
        sc = J.score(np.nan_to_num(Pn[:, :, :3]), np.nan_to_num(yw)); s0 = J.score(np.nan_to_num(xyz0), yaw0)
        cm = carry_metrics(CS, fp, CS.ev)
        mov_ns = J.mov & ~J.static
        fam = np.array([names[t].split("_")[0] for t in tenv])
        def summ(sel):
            ev_ = [c for c in cm if sel[c["env"]]]
            m_ = mov_ns & sel[:, None]
            return dict(envs=int(sel.sum()), goal_err_cm_mean=float(sc["dist_cm"][m_].mean()), goal_err_cm_start=float(s0["dist_cm"][m_].mean()),
                        goals_met_share=float(sc["met"][m_].mean()), all_met_share=float(sc["met"][sel].all(1).mean()),
                        static_moved_gt1cm=float((sc["static_moved_cm"][sel] > 1).sum() / max(int(sel.sum()), 1)),
                        carries_per_env=float(r["n_carries"][sel].mean()), carries=len(ev_),
                        push_held_gt1cm_share=float(np.mean([c["push_held_cm"] > 1 for c in ev_])) if ev_ else None,
                        grasp_box_width_gt8cm_share=float(np.mean([c["open_cm"] > 100 * MAX_OPEN for c in ev_])) if ev_ else None,
                        end_err_gt3cm_share=float(np.mean([c["end_err_cm"] > 3 for c in ev_])) if ev_ else None,
                        hand_hand_share=float(np.mean([c["hand_hand_cm"] > 0 for c in ev_])) if ev_ else None,
                        end_err_cm_median=float(np.median([c["end_err_cm"] for c in ev_])) if ev_ else None,
                        stop_kinds={k: int((r["stop_kind"][sel] == k).sum()) for k in ("all_met", "stop", "max_carries", "time")},
                        wrong_stop_share=float(((r["stop_kind"][sel] == "stop") & ~sc["met"][sel].all(1)).mean()),
                        makespan_s_median=float(np.median(r["makespan_s"][sel])), hand_hand_env_share=float((CS.hh_max.cpu().numpy()[sel] > 0).mean()),
                        hand_move_push_gt1cm_share=float(np.mean([m["push_cm"] > 1 for m in mvp if sel[m["env"]]])) if any(sel[m["env"]] for m in mvp) else None,
                        lift_cm_median=float(np.median([c["lift_cm"] for c in ev_])) if ev_ else None,
                        flat_share=float(np.mean([c["lift_cm"] < 2 for c in ev_])) if ev_ else None)
        res = dict(tag=a.tag, split=a.split, N=N, templates=len(names), copies=a.copies, dt=1 / a.dt, disturb=a.disturb, max_carries=mc,
                   goal=a.goal, max_active=a.max_active, retreat=a.retreat, approach=not a.no_approach,
                   wall_s=r["wall_s"], sim_s=r["steps"] * fp.dt, episodes_per_s=N / r["wall_s"], decision_calls=r["dec_calls"], decision_s=r["dec_s"],
                   step_s=r["step_s"], tick_s=r["tick_s"], lp_wm_s=float(sum(x["wm_s"] for x in LP.timing)), lp_planner_s=float(sum(x["planner_s"] for x in LP.timing)),
                   lp_rows=int(sum(x["n"] for x in LP.timing)), lp_carries=int(sum(x["carries"] for x in LP.timing)),
                   n_disturb=r["n_disturb"], all=summ(np.ones(N, bool)), by_family={f: summ(fam == f) for f in sorted(set(fam))})
        json.dump(res, open(rd / "baseline.json", "w"), indent=1, default=float)
        json.dump(cm, open(rd / "carries.json", "w"), default=float)
        log("BASELINE " + json.dumps(res, default=float))


if __name__ == "__main__":
    main()
