"""Global path planner: physics-checked geometric planning so that carries go over or around things rather than into container
walls. Task-agnostic, one rule for every object, from shapes and positions only. The world model gives the object and its T1 goal;
this module gives the path.
  candidates  per carry, whole routes from the object's start (root) to its goal (root), 33 points:
                the world model's own path; the straight line with a small arc;
                over routes: rise straight up to a cruise height, cross, come straight down onto the goal. The cruise height clears
                the top of every other object whose footprint comes within reach of the start -> goal corridor (the carried object's
                and the hand's footprint + margin), plus a margin m from MARGINS (a height map from positions and extents only);
                around routes: the same, but the crossing bends through a side waypoint (SIDE offsets from the corridor's middle)
                at the lowest cruise height that clears that bent corridor.
  choose      every candidate is scored with route_opt's exact cost (own-shape SDF, the hand's points (WM_ROUTE_HAND), the set-down
              rule (WM_ROUTE_PLACE), the band), batched on the GPU; the cheapest is polished by route_opt's gradient steps, and the
              polished path is kept only if its exact cost is not worse.
Use: plan_paths(ts, items, polish_steps) with route_opt items (S, xyz (I, 3), yaw (I,), a, plans (1, 33, 3) = the world-model path,
dyaws (1,)) -> list of (33, 3) float32 paths (the world-model path where the carried object has no shape)."""
from __future__ import annotations

import os
import numpy as np
import torch

from cleantables.worldmodel.physics import route_opt as RO

NP = 33
MARGINS = (0.02, 0.04, 0.08, 0.12)
SIDE = (-0.30, -0.20, -0.10, 0.10, 0.20, 0.30)
SIDE_OVER_M = (0.04, 0.08)                                       # around routes are also tried at these higher cruise margins


def _poly(pts, n=NP):
    """Polyline corners (m, 3) -> n points spaced by arc length (corners kept within one spacing)."""
    pts = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = np.concatenate([[0.0], np.cumsum(seg)])
    if L[-1] < 1e-6:
        return np.repeat(pts[:1], n, 0).astype(np.float32)
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, pts[:, d]) for d in range(3)], -1).astype(np.float32)


def _corridor_height(objs, xyz, a, pts_xy, reach):
    """Highest top (root z + height) of any other object whose footprint comes within `reach` of the polyline pts_xy (k, 2)
    (vectorised over objects and segments)."""
    n = min(len(objs), len(xyz))
    idx = [j for j in range(n) if j != a]
    if not idx:
        return 0.0
    ex = np.array([objs[j]["extents"] for j in idx], np.float64); c = np.asarray(xyz, np.float64)[idx, :2]
    r = 0.5 * np.hypot(ex[:, 0], ex[:, 1])
    P = np.asarray(pts_xy, np.float64); A, Bp = P[:-1], P[1:]; AB = Bp - A                                 # (S, 2)
    t = np.clip(((c[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12)[None], 0, 1)   # (J, S)
    d = np.linalg.norm(c[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1).min(1)
    hit = d < r + reach
    return float((np.asarray(xyz, np.float64)[idx, 2] + ex[:, 2])[hit].max()) if hit.any() else 0.0


def _seg_dist(p, a, b):
    ab = b - a; t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def _sets():
    """Candidate grid. GP_SMALL=1 (for speed in live training) uses margins 2 / 4 / 8 cm and sides +-10 / 20 cm at 2 cm, 9 routes
    instead of 24; on held-out scenes its side-hit rate is close to the full grid's (4.5 % vs 4.3 %)."""
    import os
    if os.environ.get("GP_SMALL") == "1":
        return (0.02, 0.04, 0.08), (-0.20, -0.10, 0.10, 0.20), ()
    return MARGINS, SIDE, SIDE_OVER_M


def candidates(it):
    """Candidate routes of one carry (n, 33, 3) float32 (root points)."""
    margins, side, side_over = _sets()
    objs = it["S"]["objects"]; a = it["a"]; xyz = np.asarray(it["xyz"], np.float64)
    st = xyz[a].copy(); wm = np.asarray(it["plans"][0], np.float64); gl = wm[-1].copy()
    ex = np.asarray(objs[a]["extents"], np.float64)
    reach = 0.5 * float(np.hypot(ex[0], ex[1]))
    if RO.HAND:                                                  # the hand's footprint (fp_hand points) widens the corridor
        from cleantables.rl.gripper import points as hand_pts
        hp = hand_pts(ex, gyaw=it.get("gyaw")); reach = max(reach, float(np.hypot(hp[:, 0], hp[:, 1]).max()))
    C = [wm.astype(np.float32)]
    mid = 0.5 * (st + gl); C.append(_poly([st, mid + np.array([0, 0, 0.03 + 0.1 * np.linalg.norm(gl[:2] - st[:2])]), gl]))
    d_xy = gl[:2] - st[:2]; L = float(np.linalg.norm(d_xy)); nrm = np.array([-d_xy[1], d_xy[0]]) / max(L, 1e-6)
    for m in margins:
        zc = max(_corridor_height(objs, xyz, a, [st[:2], gl[:2]], reach + m) + m, st[2] + m, gl[2] + m)
        C.append(_poly([st, [st[0], st[1], zc], [gl[0], gl[1], zc], gl]))
    for off in side:
        w = 0.5 * (st[:2] + gl[:2]) + off * nrm
        for m in (0.02,) + tuple(side_over):
            zc = max(_corridor_height(objs, xyz, a, [st[:2], w, gl[:2]], reach + m) + m, st[2] + m, gl[2] + m)
            C.append(_poly([st, [st[0], st[1], zc], [w[0], w[1], zc], [gl[0], gl[1], zc], gl]))
    return np.stack(C).astype(np.float32)


@torch.no_grad()
def _score(ts, items):
    B = RO.build_batch(ts, items)
    if B is None:
        return None, None
    P = RO.path_cost(ts, B, B["plans"], soft=False)
    return B["keep"], P


def plan_paths(ts, items, polish_steps=100, chunk=128):
    """-> (paths [(33, 3)], info dict). Every candidate is scored exactly, the best is polished by route_opt, and the polish is kept only if not worse."""
    out = [np.asarray(it["plans"][0], np.float32) for it in items]
    chosen = [-1] * len(items); best_c = [np.inf] * len(items)
    for c0 in range(0, len(items), chunk):
        sub = items[c0:c0 + chunk]
        cand = [candidates(it) for it in sub]
        its = [dict(it, plans=cd, dyaws=np.repeat(np.asarray(it["dyaws"][:1], np.float32), len(cd))) for it, cd in zip(sub, cand)]
        keep, P = _score(ts, its)
        if keep is None:
            continue
        cost = P["cost"].cpu().numpy(); q = 0
        for k in keep:
            n = len(cand[k]); c = cost[q:q + n]; q += n
            i = int(np.argmin(c)); out[c0 + k] = cand[k][i]; chosen[c0 + k] = i; best_c[c0 + k] = float(c[i])
    info = dict(n=len(items), chosen=chosen)
    if polish_steps > 0:
        pol = [dict(it, plans=out[i][None]) for i, it in enumerate(items)]
        res, oi = RO.optimise(ts, pol, steps=polish_steps)
        k2 = [i for i, r in enumerate(res) if r is not None]
        if k2:
            keep, P = _score(ts, [dict(pol[i], plans=np.asarray(res[i], np.float32)) for i in k2])
            cost = P["cost"].cpu().numpy() if keep is not None else []
            for q, kk in enumerate(keep or []):
                i = k2[kk]
                if cost[q] <= best_c[i] + 1e-6:
                    out[i] = np.asarray(res[i][0], np.float32); best_c[i] = float(cost[q])
        info["polish_seconds"] = oi.get("seconds")
    info["cost_mean"] = float(np.mean([c for c in best_c if np.isfinite(c)])) if any(np.isfinite(best_c)) else None
    return out, info


# ---------------------------------------------------------------------------------------------- hook for the robot environment's plan pool
_TS = {}


def plan_paths_bridge(B, xyz, yaw, placed, r, polish_steps=None, hand="gripper", band=0.02):
    """plan_pool hook of the robot environment (called right after each B.plan_all): every object's path comes from this planner
    instead of the world model.
    B: the scene's WMBridge2; xyz (I, 3) object roots (twin frame); yaw (I,) yaw offsets; placed (I,) already placed (they stay obstacles);
    r: B.plan_all's output (end_xyz (1, I, 3) root goals, path (1, I, 33, 3), end_yaw_off (1, I)). CPU torch.
    -> (path (1, I, 33, 3) float32: path[:, i, 0] = xyz[i], path[:, i, -1] = r["end_xyz"][0, i]; contact (1, I, 2, 32): held (1) on every
    segment but the last, released (0) on the last, i.e. the object is set down at its goal, then let go).
    The hand kept clear is the Franka gripper (fp_hand) unless hand="ball"; band 2 cm; the set-down rule is on."""
    import os
    os.environ.setdefault("FP_HAND", hand)
    RO.HAND, RO.PLACE, RO.BAND = 1, 1, float(band)
    steps = int(os.environ.get("GP_POLISH", "50")) if polish_steps is None else int(polish_steps)
    if "ts" not in _TS:
        from cleantables.worldmodel.physics.geo_cost import MeshCost
        _TS["ts"] = RO.TorchSDF(MeshCost(w_drag=0.0), torch.device("cpu"))
    ts = _TS["ts"]
    xyz = np.asarray(xyz.detach().cpu() if torch.is_tensor(xyz) else xyz, np.float64)
    yaw = np.asarray(yaw.detach().cpu() if torch.is_tensor(yaw) else yaw, np.float64) if yaw is not None else np.zeros(len(xyz))
    end = r["end_xyz"][0].detach().cpu().numpy() if torch.is_tensor(r["end_xyz"]) else np.asarray(r["end_xyz"][0])
    wm = r["path"][0].detach().cpu().numpy() if torch.is_tensor(r["path"]) else np.asarray(r["path"][0])
    dy = r["end_yaw_off"][0].detach().cpu().numpy() if "end_yaw_off" in r else np.zeros(len(xyz))
    I = len(xyz); out = np.array(wm, np.float32, copy=True)
    idx = [i for i in range(I) if np.isfinite(end[i]).all() and np.isfinite(wm[i]).all()]
    items = []
    for i in idx:
        w = np.array(wm[i], np.float64); w[0] = xyz[i]; w[-1] = end[i]
        items.append(dict(S=B.S, xyz=xyz, yaw=yaw, a=i, plans=w[None].astype(np.float32), dyaws=np.array([np.nan_to_num(dy[i])], np.float32)))
    if items:
        with torch.enable_grad():                                          # (callers may run under no_grad)
            pp, _ = plan_paths(ts, items, polish_steps=steps)
        for i, p in zip(idx, pp):
            p = np.array(p, np.float32); p[0] = xyz[i]; p[-1] = end[i]; out[i] = p
    contact = np.ones((1, I, 2, NP - 1), np.float32); contact[..., -1] = 0.0
    return torch.as_tensor(out[None]), torch.as_tensor(contact)


def candidates_scored(ts, items, chunk=128):
    """Every item's candidate routes and their exact route_opt costs, for a policy that chooses the route:
    -> (cands list of (C, 33, 3) float32, costs list of (C,) float64 (inf where unscored), parts list of dicts of (C,) arrays: hit, band,
    econ, table). C = 9 with GP_SMALL=1 (world-model path / straight line, the arc, 3 over, 4 around)."""
    cands = [candidates(it) for it in items]
    costs = [np.full(len(c), np.inf) for c in cands]; parts = [dict() for _ in cands]
    for c0 in range(0, len(items), chunk):
        sub = items[c0:c0 + chunk]
        its = [dict(it, plans=cd, dyaws=np.repeat(np.asarray(it["dyaws"][:1], np.float32), len(cd))) for it, cd in zip(sub, cands[c0:c0 + chunk])]
        keep, P = _score(ts, its)
        if keep is None:
            continue
        cost = P["cost"].cpu().numpy(); pp = {k: P[k].cpu().numpy() for k in ("hit", "band", "econ", "table")}; q = 0
        for k in keep:
            n = len(cands[c0 + k]); costs[c0 + k] = cost[q:q + n].astype(np.float64)
            parts[c0 + k] = {kk: v[q:q + n] for kk, v in pp.items()}; q += n
    return cands, costs, parts
