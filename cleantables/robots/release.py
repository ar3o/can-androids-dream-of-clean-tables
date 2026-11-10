"""Where the hand lets go: the world model's contact head, read from WMBridge2.plan_all's `contact` (N, I, 2, 32): P(a hand holds
the object) at each of the 32 steps of the world model's own imagined path. Training and evaluation both read the release from this
head; the release point is the last step with hold >= 0.5 (the head's own read-out, as plan_all's off_step). The robot never passes a
destination to the head.
"""
from __future__ import annotations

import torch


REL_PEAK_MIN = 0.1        # a path whose absolute peak P(hold) is below this carries no contact signal: the path-end fallback


def rel_hold(hold32):
    """(n,32) the head's absolute P(hold) -> hold_rel = P / its own peak along that path. The head is trained on a rare event (contact
    on ~8 % of tokens), so its absolute values are low and a fixed 0.5 cut-off does not fit; the shape decides, scale-free.
    The release point (hold_rel >= 0.5), the observed contact command and the reach term's h all use hold_rel. A path whose absolute
    peak is < REL_PEAK_MIN (no signal), or NaN (no head): NaN = the path-end fallback (the callers turn it into 1)."""
    pk = hold32.amax(1, keepdim=True)
    return torch.where(pk >= REL_PEAK_MIN, hold32 / pk.clamp(min=1e-6), torch.full_like(hold32, float("nan")))


def on_fallback(hold32):
    """(n,) bool: the path-end fallback applies (no head, or its absolute peak < REL_PEAK_MIN)"""
    return ~(hold32.amax(1) >= REL_PEAK_MIN)


EARLY_FRAC = 0.5          # a release before half of the path


def release_frac(hold32):
    """(n,32) -> (n,) where the robot lets go, as a fraction of the path: the last step with hold_rel >= 0.5 over 31; 1 on the fallback"""
    on = rel_hold(hold32) >= 0.5                                         # NaN (fallback) -> False
    last = torch.where(on, torch.arange(hold32.shape[1], device=hold32.device)[None], torch.full_like(on, -1, dtype=torch.long)).amax(1)
    return torch.where(last >= 0, last.float() / (hold32.shape[1] - 1), torch.ones_like(last, dtype=torch.float32))


def early_to_end(hold32):
    """Early-release rule (applied only when > 10 % of a WM's training slots release before half the path): a path whose release point
    lies before half of it is released at the path end instead (letting go mid-carry would drop the object in the air: a physical rule,
    not a task rule). Those rows -> NaN = the path-end fallback; the rest unchanged. Training and evaluation use this same function."""
    early = (release_frac(hold32) < EARLY_FRAC) & ~on_fallback(hold32)
    return torch.where(early[:, None], torch.full_like(hold32, float("nan")), hold32), early


def hold_points(hold32):
    """(n,32) P(hold) of the path steps, either hand (NaN: no contact head) -> (n,33) hold_rel per point of plan_all's 33-point root path;
    the fallback (NaN) holds to the end; the last point is the WM's end pose, a resting pose: 0 there."""
    return torch.cat([torch.nan_to_num(rel_hold(hold32), nan=1.0), torch.zeros_like(hold32[:, :1])], 1)
