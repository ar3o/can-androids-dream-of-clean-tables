"""Per-episode random table layouts for Stage A (FPC_LAYOUT_RAND=<p>; 0 = off).

At an episode start, with probability p per (table template, episode key), every valid object of that table gets a fresh random pose on the
table: xy uniform, yaw uniform (a world-z turn on top of its template orientation; roll, pitch and baked yaw kept), resting height from the
template (objects stacked or nested in the template, root above SURF_Z_MAX, are put on the table surface). The same rule applies to every
object. Constraints, checked per candidate: the object's footprint (oriented box of its extents + MARGIN / 2 per side) lies on the table,
does not overlap an already placed footprint, and the TCP 2 cm above its top is within the closed-form top-down reach of at least one arm
(cleantables.rl.chain_env reach test). Objects are placed largest footprint first, N_CAND candidates each; a table where any object finds
no valid spot keeps its original layout for the episode. The draw is seeded per (seed, template, episode key), so the copies of a table
(one GRPO group) get the same layout in the same episode.

The frozen judge is then re-derived for those envs from the new start frame (FrozenJudge on the new poses: reference, offset, tolerance,
mover / static, end0, d0, start pose; dy0 for met_3d; both judges of the instruction mux), and restored from a snapshot for envs on their
original layout. Training rollouts only (mode 'sample'); evaluations keep the original layouts.
"""
from __future__ import annotations

import math
import os

import numpy as np

LAYOUT_RAND = float(os.environ.get("FPC_LAYOUT_RAND", "0"))
# FPC_LAYOUT_SOLVABLE=1: a random draw is kept only if it is one-arm solvable (every mover's judged goal, under both instruction judges,
# reachable by an arm that also reaches its start); otherwise redrawn, up to LAYOUT_TRIES tries in all, then the original layout
SOLVABLE = os.environ.get("FPC_LAYOUT_SOLVABLE", "0") == "1"
LAYOUT_TRIES = int(os.environ.get("FPC_LAYOUT_TRIES", "4"))
MARGIN = 0.02            # m: gap between two footprints (each box grows by MARGIN / 2 per side)
TABLE_MARGIN = 0.01      # m: footprint corners at least this far inside the table edge
SURF_Z_MAX = 0.01        # m: a template root above this rests on something (stacked / nested): put on the table surface
SURF_Z = 0.002           # m: the template's resting root height (bottom + 2 mm)
N_CAND = 128             # candidates per object
SOBOL = os.environ.get("FPC_LAYOUT_SOBOL", "0") == "1"   # scrambled-Sobol candidates instead of i.i.d. uniform
TCP_UP = 0.02            # m: the reach point is the TCP this far above the object's top (as ArmTracker.reach)

JUDGE_ROW_ATTRS = ("j", "off", "szj", "fixed", "is_fixed", "mov", "sd", "sdz", "end0", "static", "start_xyz", "start_yaw", "d0_cm")


# ---- quaternions (xyzw), as in sim_env
def _qyaw(y):
    y = np.asarray(y, np.float64)
    return np.stack([np.zeros_like(y), np.zeros_like(y), np.sin(y / 2), np.cos(y / 2)], -1)


def _qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0); bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def _yaw_zyx(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---- geometry
def reachable(tcp, bases):
    """(..., 3) TCP points (table frame) vs arm bases (..., A, 3) broadcastable -> (...) bool: within the closed-form top-down reach of >= 1 arm
    (as chain_env.reach_pull: the wrist within RR_R of the shoulder, never nearer than RR_NEAR to the base)."""
    from cleantables.rl.chain_env import RR_A7, RR_D1, RR_J7, RR_NEAR, RR_R
    rel = tcp[..., None, :2] - bases[..., :2]; hd = np.linalg.norm(rel, axis=-1)
    dz = tcp[..., None, 2] + RR_J7 - (bases[..., 2] + RR_D1)
    ok = ((np.clip(hd - RR_A7, 0, None) ** 2 + dz ** 2) < RR_R ** 2) & (hd > RR_NEAR)
    return ok.any(-1)


def _corners(c, h, a):
    """Oriented boxes: centres c (..., 2), half sizes h (..., 2), angles a (...) -> corners (..., 4, 2)."""
    ca, sa = np.cos(a)[..., None], np.sin(a)[..., None]
    sx = np.array([1, -1, -1, 1], np.float64); sy = np.array([1, 1, -1, -1], np.float64)
    lx, ly = h[..., 0:1] * sx, h[..., 1:2] * sy
    return np.stack([c[..., 0:1] + ca * lx - sa * ly, c[..., 1:2] + sa * lx + ca * ly], -1)


def obb_overlap(c1, h1, a1, c2, h2, a2):
    """Separating-axis test for 2-D oriented boxes, broadcast over leading dims -> bool (True = overlap)."""
    k1, k2 = _corners(c1, h1, a1), _corners(c2, h2, a2)
    sep = np.zeros(np.broadcast_shapes(k1.shape[:-2], k2.shape[:-2]), bool)
    for ang in (a1, a2):
        for d in (0.0, math.pi / 2):
            ax = np.stack([np.cos(ang + d), np.sin(ang + d)], -1)[..., None, :]          # (..., 1, 2)
            p1 = (k1 * ax).sum(-1); p2 = (k2 * ax).sum(-1)
            sep |= (p1.max(-1) < p2.min(-1)) | (p2.max(-1) < p1.min(-1))
    return ~sep


# ---- the sampler
def sample_table(rng, lay0, valid, ext, box, bases, n_cand=N_CAND):
    """Sample one table: original layout lay0 (K, 7) (root xyz bottom centre + quat xyzw), valid (K,), extents ext (K, 3) (full sizes in the object's
    rotated frame), table box (x0, x1, y0, y1), arm bases (A, 3). -> (layout (K, 7), dyaw (K,)) or None (no valid spot for some object)."""
    K = lay0.shape[0]; idx = np.flatnonzero(valid)
    out = lay0.astype(np.float64).copy(); dyaw = np.zeros(K)
    if not len(idx):
        return out.astype(np.float32), dyaw.astype(np.float32)
    a0 = _yaw_zyx(lay0[:, 3:].astype(np.float64))                         # the footprint's angle (extents are along it)
    half = np.abs(ext[:, :2].astype(np.float64)) / 2 + MARGIN / 2
    z = np.where(lay0[:, 2] > SURF_Z_MAX, SURF_Z, lay0[:, 2]).astype(np.float64)
    order = idx[np.argsort(-(half[idx, 0] * half[idx, 1]), kind="stable")]  # largest footprint first
    if SOBOL:                                                               # each object's candidates: a scrambled Sobol set over (x, y, yaw), seeded
        from scipy.stats import qmc                                         # from this table's stream (even coverage instead of random clumps)
        U = np.stack([qmc.Sobol(d=3, scramble=True, seed=int(rng.integers(2 ** 31))).random(n_cand) for _ in range(len(order))])
    else:
        U = rng.random((len(order), n_cand, 3))                             # every random number drawn up front (a fixed stream per table)
    pc, ph, pa = [], [], []
    x0, x1, y0, y1 = [float(v) for v in box]
    for s, k in enumerate(order):
        cx = x0 + U[s, :, 0] * (x1 - x0); cy = y0 + U[s, :, 1] * (y1 - y0)
        dy = (U[s, :, 2] * 2 - 1) * math.pi; ang = a0[k] + dy
        c = np.stack([cx, cy], -1); h = np.broadcast_to(half[k], (n_cand, 2))
        kc = _corners(c, h - MARGIN / 2, ang)                               # the true footprint on the table
        ok = ((kc[..., 0] >= x0 + TABLE_MARGIN) & (kc[..., 0] <= x1 - TABLE_MARGIN) & (kc[..., 1] >= y0 + TABLE_MARGIN) & (kc[..., 1] <= y1 - TABLE_MARGIN)).all(-1)
        tcp = np.stack([cx, cy, np.full(n_cand, z[k] + abs(float(ext[k, 2])) + TCP_UP)], -1)
        ok &= reachable(tcp, np.asarray(bases, np.float64)[None])
        if pc:
            P_c, P_h, P_a = np.stack(pc), np.stack(ph), np.asarray(pa)
            ok &= ~obb_overlap(c[:, None], h[:, None], ang[:, None], P_c[None], P_h[None], P_a[None]).any(-1)
        hit = np.flatnonzero(ok)
        if not len(hit):
            return None
        i = hit[0]
        pc.append(c[i]); ph.append(h[i]); pa.append(ang[i])
        out[k, 0], out[k, 1], out[k, 2] = c[i, 0], c[i, 1], z[k]
        out[k, 3:] = _qmul(_qyaw(dy[i]), lay0[k, 3:].astype(np.float64)); dyaw[k] = dy[i]
    return out.astype(np.float32), dyaw.astype(np.float32)


class LayoutRand:
    """Per-episode random layouts for a fixed set of envs. lay0 (N, K, 7) the original layouts, valid (N, K), ext (N, K, 3), tpl (N,),
    box (N, 4), bases (N, A, 3) (table frame). draw(envs, keys) -> (layout (n, K, 7), random (n,) bool, dyaw (n, K))."""

    def __init__(self, p, lay0, valid, ext, tpl, box, bases, seed=0):
        self.p, self.seed = float(p), int(seed)
        self.lay0, self.valid, self.ext = np.asarray(lay0, np.float32), np.asarray(valid, bool), np.asarray(ext, np.float32)
        self.tpl, self.box, self.bases = np.asarray(tpl).astype(int), np.asarray(box, np.float32), np.asarray(bases, np.float32)
        self.st = dict(draws=0, random=0, fallback=0)
        self.cur_rand = np.zeros(len(self.tpl), bool)                         # each env's current episode: random layout?
        self.active, self.ep_seed, self.t_judge = False, 0, 0.0              # set per rollout by begin()
        self.snap = self.yaw0 = self.LPa = self.temb_T = None                # set by the trainer: judge snapshot, start yaws, dy0 planner

    def begin(self, active, ep_seed):
        """Start a rollout: random layouts on (training rollouts) or off (evaluations: every env back on its original layout)."""
        self.active, self.ep_seed = bool(active), int(ep_seed); self.st = dict(draws=0, random=0, fallback=0); self.t_judge = 0.0
        self.sv = dict(rej=[], redraw=0, redraw_fail=0, fallback=0)                # one-arm-solvable filter stats

    def summary(self):
        """One-line log of this rollout's draws."""
        sv = "" if not (SOLVABLE and getattr(self, "sv", None)) else (
            f"; one-arm solvable: tables rejected per try {self.sv['rej']}, redraws {self.sv['redraw']} (no valid spot {self.sv['redraw_fail']}), "
            f"kept the original after {LAYOUT_TRIES} tries {self.sv['fallback']}")
        return (f"LAYOUT RAND: tables drawn {self.st['draws']}, random {self.st['random']}, fallback to the original {self.st['fallback']}; "
                f"envs on a random layout now {int(self.cur_rand.sum())} of {len(self.cur_rand)}; judge re-derived in {self.t_judge:.1f} s" + sv)

    def draw(self, envs, keys, always=False):
        """Draw layouts for envs with episode keys; always=True skips the probability p (a redraw)."""
        envs = np.asarray(envs, int)
        if len(envs) == 0:                                                  # nothing to draw
            return self.lay0[envs].copy(), np.zeros(0, bool), np.zeros((0, self.lay0.shape[1]), np.float32)
        keys = np.asarray(keys, np.int64).reshape(len(envs), -1)
        lay = self.lay0[envs].copy(); rnd = np.zeros(len(envs), bool); dyaw = np.zeros((len(envs), self.lay0.shape[1]), np.float32)
        memo = {}
        for i, e in enumerate(envs):
            t = int(self.tpl[e]); key = (t,) + tuple(int(x) for x in keys[i])
            if key not in memo:
                rng = np.random.default_rng([self.seed, *key])
                res = None
                if always or rng.random() < self.p:                         # always: a redraw of a table already chosen random
                    if always:
                        self.sv["redraw"] += 1
                    else:
                        self.st["random"] += 1
                    res = sample_table(rng, self.lay0[e], self.valid[e], self.ext[e], self.box[e], self.bases[e])
                    if res is None:
                        if always:
                            self.sv["redraw_fail"] += 1
                        else:
                            self.st["fallback"] += 1
                if not always:
                    self.st["draws"] += 1
                memo[key] = res
            res = memo[key]
            if res is not None:
                lay[i], dyaw[i] = res; rnd[i] = True
        self.cur_rand[envs] = rnd
        return lay, rnd, dyaw


# ---- one-arm solvability
def solvable_rows(judge, envs, xyz, valid, bases):
    """envs (n,) whose judge rows hold their current start frame xyz (n, K, 3) (the judge's convention: bottom centre - 2 mm): True where every
    mover (mov & ~static, valid) of every judge part (the instruction mux) has its goal (end0) top-down reachable by an arm that also reaches
    its start top (TCP TCP_UP above; the closed-form reach per arm). bases (n, A, 3). -> (n,) bool."""
    envs = np.asarray(envs, int); ok = np.ones(len(envs), bool); up = np.array([0, 0, TCP_UP], np.float32)
    bases = np.asarray(bases, np.float32)
    for J in judge_parts(judge):
        I = J.Imax; H = np.zeros((len(envs), I), np.float32)
        for r, e in enumerate(envs):                                         # the start top (as FrozenJudge._top)
            t = J.tpl[e]; Ii = J.LP.B[t].I; H[r, :Ii] = np.asarray(J.LP.B[t].Z["extents"], np.float32)[:Ii, 2]
        top = np.asarray(xyz[:, :I], np.float32).copy(); top[..., 2] += H
        goal = np.asarray(J.end0[envs], np.float32)
        m = (np.asarray(J.mov[envs], bool) & ~np.asarray(J.static[envs], bool) & np.asarray(valid, bool)[:, :I]
             & np.isfinite(top).all(-1) & np.isfinite(goal).all(-1))
        rs = np.stack([reachable(top + up, bases[:, None, a:a + 1, :]) for a in range(bases.shape[1])], -1)
        rg = np.stack([reachable(np.nan_to_num(goal) + up, bases[:, None, a:a + 1, :]) for a in range(bases.shape[1])], -1)
        ok &= (~m | (rs & rg).any(-1)).all(-1)
    return ok


def solvable_draw(LR, judge, envs, keys, tries=None):
    """LR.draw (try 0: the plain draw), the judge re-derived in place (rejudge_rows), then every random table checked
    (solvable_rows; the copies of a table, same template + key, decide together: all must pass); failing tables redrawn with the try in the
    key (the same new layout for every copy) and re-judged; after `tries` tries in all, a still failing table keeps its original layout (judge
    rows restored). -> layout (n, K, 7), random (n,), dyaw (n, K); LR.t_judge / LR.sv updated."""
    tries = LAYOUT_TRIES if tries is None else int(tries)
    envs = np.asarray(envs, int); keys = np.asarray(keys, np.int64).reshape(len(envs), -1)
    xf = lambda l_: np.nan_to_num(l_[..., :3] - np.array([0, 0, 0.002], np.float32))   # noqa: E731  (the judge's convention)
    ln, rn, dyw = LR.draw(envs, keys)
    LR.t_judge += rejudge_rows(judge, LR.snap, envs, rn, xf(ln), LR.yaw0[envs] + dyw, LR.LPa, LR.temb_T)
    gk = [(int(LR.tpl[e]),) + tuple(int(x) for x in keys[i]) for i, e in enumerate(envs)]
    for tr in range(1, tries + 1):
        cand = np.flatnonzero(rn)
        if not len(cand):
            break
        ok = solvable_rows(judge, envs[cand], xf(ln[cand]), LR.valid[envs[cand]], LR.bases[envs[cand]])
        gok = {}
        for j, i in enumerate(cand):
            gok[gk[i]] = gok.get(gk[i], True) and bool(ok[j])
        bad = np.array([i for i in cand if not gok[gk[i]]], int)
        LR.sv["rej"].append(len({gk[i] for i in bad}))
        if not len(bad):
            break
        if tr == tries:                                                       # out of tries: the original layout
            ln[bad] = LR.lay0[envs[bad]]; rn[bad] = False; dyw[bad] = 0.0; LR.cur_rand[envs[bad]] = False
            LR.sv["fallback"] += len({gk[i] for i in bad})
            LR.t_judge += rejudge_rows(judge, LR.snap, envs[bad], rn[bad], xf(ln[bad]), LR.yaw0[envs[bad]] + dyw[bad], LR.LPa, LR.temb_T)
            break
        kt = np.concatenate([keys[bad], np.full((len(bad), 1), tr, np.int64)], 1)
        l2, r2, d2 = LR.draw(envs[bad], kt, always=True)
        ln[bad] = l2; rn[bad] = r2; dyw[bad] = d2
        LR.t_judge += rejudge_rows(judge, LR.snap, envs[bad], r2, xf(l2), LR.yaw0[envs[bad]] + d2, LR.LPa, LR.temb_T)
    LR.cur_rand[envs] = rn
    return ln, rn, dyw


# ---- the frozen judge, row-wise
def judge_parts(judge):
    """The FrozenJudge objects behind judge (a FrozenJudge or a JudgeMux)."""
    return tuple(judge.J) if hasattr(judge, "J") and isinstance(getattr(judge, "J"), (tuple, list)) else (judge,)


def judge_snapshot(judge):
    """Copies of every judge part's per-env rows (JUDGE_ROW_ATTRS and dy0), for restoring envs on their original layout."""
    snap = []
    for J in judge_parts(judge):
        d = {a: np.array(getattr(J, a), copy=True) for a in JUDGE_ROW_ATTRS}
        if getattr(J, "dy0", None) is not None:
            d["dy0"] = np.array(J.dy0, copy=True)
        snap.append(d)
    return snap


def rejudge_rows(judge, snap, envs, rnd, xyz, yaw, LPa=None, temb_T=None):
    """Re-judge restarting envs (n,): rows with rnd re-derived by the frozen judge from their new start frame xyz (n, K, 3) (bottom centre, table frame)
    / yaw (n, K) (the judge's yaw convention); rows without rnd restored from snap (their original layout). In place, every judge part (the
    instruction mux: part q under its own instruction; dy0 for met_3d with the actor planner LPa and, for part 1, the other instruction's task
    embedding temb_T (T, 2, D)). -> seconds."""
    import time

    import torch

    from cleantables.rl.chain_env import FrozenJudge
    t0 = time.time()
    envs = np.asarray(envs, int); rnd = np.asarray(rnd, bool)
    for q, (J, S) in enumerate(zip(judge_parts(judge), snap)):
        er = envs[~rnd]
        if len(er):
            for a in JUDGE_ROW_ATTRS:
                getattr(J, a)[er] = S[a][er]
            if "dy0" in S:
                J.dy0[er] = S["dy0"][er]
        ep = envs[rnd]
        if len(ep):
            I = J.Imax
            Jn = FrozenJudge(J.LP, J.tpl[ep], xyz[rnd][:, :I], yaw[rnd][:, :I], instruction=J.instruction)
            for a in JUDGE_ROW_ATTRS:
                getattr(J, a)[ep] = getattr(Jn, a)
            if "dy0" in S and LPa is not None:
                kw = {} if (q == 0 or temb_T is None) else dict(temb=temb_T[torch.as_tensor(J.tpl[ep], device=temb_T.device), 1])
                J.dy0[ep] = np.asarray(LPa._wm_fast(J.tpl[ep], xyz[rnd][:, :I], yaw[rnd][:, :I], **kw)["dy"], np.float32)[:, :J.dy0.shape[1]]
        if getattr(J, "_on_xyz", None) is not None:                          # frozen references start from the new start pose
            J._on_xyz[envs] = np.where(rnd[:, None, None], xyz[:, :J.Imax], S["start_xyz"][envs]); J._on_yaw[envs] = np.where(rnd[:, None], yaw[:, :J.Imax], S["start_yaw"][envs])
    return time.time() - t0
