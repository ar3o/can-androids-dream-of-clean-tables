"""Solid table for the joint-space controller.

The controller's residual is shrunk so the fingertips never pass through the table top; the hand may touch the top and slide
on it. Also contains per-tick joint speed and acceleration limits. Enabled with FPL_SOLID_TABLE=1 on the joint-space path."""
from __future__ import annotations

import torch

from cleantables.robots import franka_kinematics as KIN
from cleantables.rl.gripper import FL as H_FL, PH as H_PH

HAND_LOW = H_FL + H_PH / 2                       # fingertips below the palm centre (0.089 m)
TCP_DZ = 0.08                                    # palm centre -> TCP (as ArmTracker.TCP_DZ in chain_env)


def _palm_z(q, base):
    """Palm centre height (table frame) of FK(q); q (m, 7), base (m, 4, 4)."""
    Tt = base @ KIN.fk(q)[:, KIN.TCP]
    return Tt[:, 2, 3] - TCP_DZ * Tt[:, 2, 2]


def table_clamp(q_ik, dq, base, qlo, qhi, low=HAND_LOW, iters=6):
    """Return q = clamp(q_IK + s dq, Franka limits) per hand with the largest s in [0, 1] (bisection, `iters` steps, batched over the offending
    hands only) that keeps the fingertips (palm z - low) at or above the table top (z 0, table frame): s = 1 (q unchanged) where they
    already are; s = 0 (the plain IK pose) where even that is below. q_ik, dq (N, 2, 7), base (N, 2, 4, 4). -> q, s (N, 2), clamped (N, 2)."""
    N = q_ik.shape[0]
    q = torch.minimum(torch.maximum(q_ik + dq, qlo), qhi)
    s = torch.ones(N, 2, device=q_ik.device)
    h1 = _palm_z(q.reshape(-1, 7), base.reshape(-1, 4, 4)).reshape(N, 2) - low
    bad = h1 < 0
    if not bool(bad.any()):
        return q, s, bad
    e, g = torch.nonzero(bad, as_tuple=True)
    qi, d, b = q_ik[e, g], dq[e, g], base[e, g]
    h = lambda s_: _palm_z(torch.minimum(torch.maximum(qi + s_[:, None] * d, qlo), qhi), b) - low   # noqa: E731
    ok0 = h(torch.zeros(len(e), device=q_ik.device)) >= 0
    lo = torch.zeros(len(e), device=q_ik.device); hi = torch.ones(len(e), device=q_ik.device)
    for _ in range(iters):
        mid = 0.5 * (lo + hi); okm = h(mid) >= 0
        lo = torch.where(okm, mid, lo); hi = torch.where(okm, hi, mid)
    ss = torch.where(ok0, lo, torch.zeros_like(lo))
    q = q.clone(); q[e, g] = torch.minimum(torch.maximum(qi + ss[:, None] * d, qlo), qhi); s[e, g] = ss
    return q, s, bad


def rate_limit(q_prev, q_cmd, step):
    """Joint-speed limit: the realised joint motion per controller tick: q = q_prev + clamp(q_cmd - q_prev, +-step) per joint (step (7,) =
    the Franka's published v_max (franka_kinematics.V_MAX) x the tick); joints within the limit get q_cmd exactly. -> q, hit (..., 7) bool."""
    d = q_cmd - q_prev
    hit = d.abs() > step
    return torch.where(hit, q_prev + torch.maximum(torch.minimum(d, step), -step), q_cmd), hit


# Franka's published joint acceleration limits (rad/s^2, Franka control parameters, joints 1..7)
A_MAX = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]


def accel_limit(q_prev, q_cmd, dq_prev, step):
    """Joint-acceleration limit: the realised joint step per tick may change by at most a_max dt^2 from the last tick's step dq_prev
    (step (7,) = A_MAX x tick^2): q = q_prev + clamp(q_cmd - q_prev, dq_prev +- step); joints within it get q_cmd exactly. -> q, hit."""
    d = q_cmd - q_prev
    lo, hi = dq_prev - step, dq_prev + step
    hit = (d < lo) | (d > hi)
    return torch.where(hit, q_prev + torch.minimum(torch.maximum(d, lo), hi), q_cmd), hit
