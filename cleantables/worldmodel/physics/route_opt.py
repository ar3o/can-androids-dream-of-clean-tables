"""Route optimisation, path only. Reweighting (RWR) can only pick routes the model already samples, and the watch model's T2
rarely lifts (the human training carries are mostly flat), so each candidate path is improved by gradient steps on its own points
before the path generator is trained on it. The same rule for every object (no category, role or name, no hand-made lift template):
  start         the model's sampled path (watch_path: the T2-shaped residual + the straight line to the frozen T1 goal), 33 points
  variables     the 31 interior points (the start and the T1 goal stay fixed)
  hit           geo_cost.MeshCost made differentiable: the carried object's own 1 cm shell (M_PTS points, a fixed farthest-point
                subset) posed along the path as the PhysX chain replay poses it ((1 - u) x start offset, yaw u x dyaw); each other
                object's signed distance field is read with torch grid_sample (trilinear, its own 1 cm grid, one call per object type
                batched over every candidate, reading and point that meets it); separation = soft-min over the points - half a
                voxel; graded like the replay's hit (0 at >= 1 cm, 1 touching, 2 at >= 1 cm inside), max over objects, start-near
                exemption, mean over the readings at 5-95% of the length
  econ3d        3D length / max(straight, 2 cm) - 1 (weight 0.5, as MeshCost)
  smooth        mean squared second difference of the points / (1 cm)^2 (weight W_SMOOTH)
  table         penetration only: the carried object's lowest point more than 5 mm below the table top z = 0 (weight W_TABLE;
                not a drag term: resting on or sliding along the table costs nothing)
  band          surrogate only (WM_ROUTE_BAND = eps, 0.04 m when used): CHOMP-style clearance cost chomp_n(sep, eps) (1 at contact,
                smooth to 0 at eps, linear inside), max over objects; objects within eps of the start or the end pose are exempt
                while they stay within eps of it; mean over 5-95%; weight W_BAND = 10. The exact cost used for scoring and
                selection keeps the 1 cm band.
  cost          10 x hit + W_BAND x band + 0.5 x econ3d + W_SMOOTH x smooth + W_TABLE x table;  Adam, STEPS steps, lr LR (m)
Reported hit / econ before and after are always the exact CPU cost (MeshCost.score, full shell, 19 readings), never this
differentiable surrogate.
  WM_TWIN_ROOT=<dr_root> WM_PF_DIR=<pf> python -m cleantables.worldmodel.physics.route_opt --proto <out_dir> [--physx]
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.physics.geo_cost import HIT_ZONE, SDF_H, W_ECON, W_HIT, MeshCost

M_PTS = 256
W_SMOOTH, W_TABLE = 1.0, 10.0
STEPS, LR, TAU_MIN = 150, 0.002, 0.003
BAND = float(os.environ.get("WM_ROUTE_BAND", "0"))                                # surrogate-only clearance band eps (m); 0 = off
W_BAND = W_HIT
# Set-down rule (WM_ROUTE_PLACE, 0 = off; the same for every object): in the path's final segment (the suffix where the carried object
# stays within one own height of its goal height and within one own length (largest xy extent) of the goal in xy; the xy condition
# keeps a table-to-table carry from being 'final' all along) only penetration is graded (the same ramp shifted by HIT_ZONE: touching = 0,
# HIT_ZONE deep = 1) and the band is off, for every other object alike (no support or container relation: resting on something is
# touching, not penetrating). Elsewhere hit and band are unchanged. Only the carried object's own geometry defines the segment.
PLACE = int(os.environ.get("WM_ROUTE_PLACE", "0"))
HAND = int(os.environ.get("WM_ROUTE_HAND", "0"))              # add the gripper hand's points (fp_hand) to the carried object's shell, so the
                                                               # planned path keeps the hand clear too (same rule for its points)


def chomp_n(sep, eps):
    """CHOMP-style obstacle cost (Ratliff et al. 2009), normalised to 1 at contact: (eps - d)^2 / eps^2 for 0 <= d <= eps, 1 - 2 d / eps
    inside (d < 0; the same slope at 0), 0 beyond eps. Smooth, so a path anywhere within eps is pushed out along the SDF gradient."""
    return torch.where(sep < 0, 1 - 2 * sep / eps, ((eps - sep).clamp(min=0) / eps) ** 2)
MAX_CAND = 256                                                                    # candidates per optimisation batch (GPU memory)


def fps(P, n, seed=0):
    """Farthest-point subset of P (k, 3) of size n, deterministic; fewer than n points -> padded with a point 100 m up (it reads the
    far border of every grid: no hit, a negligible soft-min weight)."""
    if len(P) <= n:
        return np.concatenate([P, np.tile(np.array([[0, 0, 100.0]], np.float32), (n - len(P), 1))]).astype(np.float32)
    rng = np.random.default_rng(seed); ix = [int(rng.integers(len(P)))]; d = np.linalg.norm(P - P[ix[0]], axis=1)
    for _ in range(n - 1):
        ix.append(int(d.argmax())); d = np.minimum(d, np.linalg.norm(P - P[ix[-1]], axis=1))
    return P[ix]


class TorchSDF:
    """Every object type's MeshCost field as a torch grid on `dev`; path_cost reads each type once per step for every query that meets it."""

    def __init__(self, geo: MeshCost, dev):
        self.geo, self.dev, self._k = geo, dev, {}
        self.grids, self.lo, self.size, self.shell, self.bot = [], [], [], [], []

    def idx(self, o):
        k = (str(o["usd"]), float(o.get("scale") or 1.0))
        if k not in self._k:
            Fd = self.geo.field(o)
            if Fd is None:
                self._k[k] = None
            else:
                sdf, _, lo, shell, bottom = Fd
                self._k[k] = len(self.grids)
                self.grids.append(sdf); self.lo.append(lo); self.size.append(np.array(sdf.shape, np.float32))
                self.shell.append(fps(shell, M_PTS)); self.bot.append(float(bottom[:, 2].min()))
                self._stack = None
        return self._k[k]

    def tensors(self):
        """Per object type u: its SDF as (1, 1, Z, Y, X) on dev, lo (U, 3), size (U, 3) [voxels, x y z]."""
        if getattr(self, "_stack", None) is None or len(self._stack[0]) != len(self.grids):
            self._stack = ([torch.as_tensor(g, device=self.dev).permute(2, 1, 0)[None, None].contiguous() for g in self.grids],
                           torch.as_tensor(np.stack(self.lo), device=self.dev), torch.as_tensor(np.stack(self.size), device=self.dev))
        return self._stack


def yaw_rot(P, yaw):
    """Rotate points P (..., 3) about z by yaw (...,) (broadcast)."""
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    xr, yr = c * P[..., :1] - s * P[..., 1:2], s * P[..., :1] + c * P[..., 1:2]
    return torch.cat([xr, yr, P[..., 2:].expand_as(xr)], -1)


def zgrade_t(sep):
    return ((HIT_ZONE - sep) / HIT_ZONE).clamp(0, 1) + (-sep / HIT_ZONE).clamp(0, 1)


def build_batch(ts: TorchSDF, items):
    """items: list of dict(S, xyz (I, 3), yaw (I,), a, plans (n, 33, 3), dyaws (n,)) -> the flat tensors of one batch (every candidate
    of every item; every (candidate, other object) pair). Objects without a field -> the item is skipped (cost NaN, as MeshCost)."""
    dev = ts.dev; C, keep = [], []
    pr_c, pr_u, pr_xyz, pr_yaw = [], [], [], []
    sh_u, start, off_y0, ha, ext_a, gy_a = [], [], [], [], [], []
    for it, d in enumerate(items):
        objs = d["S"]["objects"]; a = d["a"]; ua = ts.idx(objs[a])
        uo = {j: ts.idx(objs[j]) for j in range(len(d["xyz"])) if j != a}
        if ua is None or any(v is None for v in uo.values()):
            continue
        keep.append(it)
        y0 = float(objs[a].get("yaw_baked", 0.0)) + float(d["yaw"][a])
        for c in range(len(d["plans"])):
            q = len(C); C.append(np.asarray(d["plans"][c], np.float32))
            sh_u.append(ua); start.append(np.asarray(d["xyz"][a], np.float32)); off_y0.append((y0, float(d["dyaws"][c])))
            ha.append((float(objs[a]["extents"][2]), float(max(objs[a]["extents"][:2])))); ext_a.append(np.asarray(objs[a]["extents"], np.float64))
            gy_a.append(d.get("gyaw"))                                    # grasp yaw (None = the default short-axis grasp)
            for j, u in uo.items():
                pr_c.append(q); pr_u.append(u); pr_xyz.append(np.asarray(d["xyz"][j], np.float32))
                pr_yaw.append(float(objs[j].get("yaw_baked", 0.0)) + float(d["yaw"][j]))
    if not C:
        return None
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)   # noqa: E731
    if HAND:
        from cleantables.rl.gripper import points as hand_pts
        shs = [np.concatenate([ts.shell[u], hand_pts(e, gyaw=g)], 0) for u, e, g in zip(sh_u, ext_a, gy_a)]
    else:
        shs = [ts.shell[u] for u in sh_u]
    return dict(keep=keep, plans=t(np.stack(C)), sh=t(np.stack(shs)), bot=t([ts.bot[u] for u in sh_u]),
                start=t(np.stack(start)), y0=t([v[0] for v in off_y0]), dy=t([v[1] for v in off_y0]), ha=t([v[0] for v in ha]), hl=t([v[1] for v in ha]),
                pc=t(pr_c, torch.long), pu=t(pr_u, torch.long), pxyz=t(np.stack(pr_xyz)) if pr_c else None, pyaw=t(pr_yaw))


def path_cost(ts: TorchSDF, B, pl, soft=True):
    """Differentiable cost of paths pl (N, 33, 3) of batch B -> dict of (N,) tensors."""
    G, lo, Ssz = ts.tensors()
    N, NP, _ = pl.shape
    seg = (pl[:, 1:] - pl[:, :-1]).norm(dim=-1)
    L = seg.sum(1)
    s = torch.cat([torch.zeros_like(L)[:, None], seg.cumsum(1)], 1) / L.clamp(min=1e-6)[:, None]       # arc-length fraction (N, 33)
    off = B["start"] - pl[:, 0]
    x = pl + (1 - s)[..., None] * off[:, None]                                                          # as the chain replay poses it
    ya = B["y0"][:, None] + s.detach() * B["dy"][:, None]
    Q = yaw_rot(B["sh"][:, None], ya[..., None]) + x[:, :, None]                                       # (N, 33, M, 3) world
    sep = torch.full((N, NP, 1), float("inf"), device=pl.device)
    if B["pc"].numel():
        Qp = Q[B["pc"]] - B["pxyz"][:, None, None]                                                     # (P, 33, M, 3)
        Qp = yaw_rot(Qp, -B["pyaw"][:, None, None])
        nrm = 2 * (Qp - lo[B["pu"]][:, None, None]) / (SDF_H * Ssz[B["pu"]][:, None, None]) - 1         # grid_sample coords (x y z)
        P_, M = Qp.shape[0], Qp.shape[2]
        order = torch.argsort(B["pu"]); us, cnt = torch.unique_consecutive(B["pu"][order], return_counts=True)
        segs, c0 = [], 0
        for u, n_ in zip(us.tolist(), cnt.tolist()):                                                   # one call per object type
            q_ = nrm[order[c0:c0 + n_]].reshape(1, -1, 1, 1, 3)
            segs.append(F.grid_sample(G[u], q_, mode="bilinear", padding_mode="border", align_corners=False).reshape(n_, NP, M)); c0 += n_
        v = torch.cat(segs)[torch.argsort(order)]                                                      # (P, 33, M), pair order
        sp = (-TAU_MIN * torch.logsumexp(-v / TAU_MIN, -1) if soft else v.amin(-1)) - SDF_H / 2        # (P, 33)
        J = torch.zeros(N, dtype=torch.long, device=pl.device).index_add_(0, B["pc"], torch.ones_like(B["pc"]))
        rank = torch.arange(len(B["pc"]), device=pl.device) - torch.repeat_interleave(torch.cumsum(J, 0) - J, J)
        sep = torch.full((N, int(J.max()), NP), float("inf"), device=pl.device)
        sep = sep.index_put((B["pc"], rank), sp).transpose(1, 2)                                       # (N, 33, Jmax)
    with torch.no_grad():                                                                               # start-near exemption
        close = sep < HIT_ZONE
        ex = torch.cummin(close.int(), dim=1).values.bool()
    sep_e = torch.where(ex, torch.full_like(sep, float("inf")), sep)
    fin = torch.zeros(N, NP, dtype=torch.bool, device=pl.device)
    if PLACE:
        with torch.no_grad():                                                  # the final segment (see PLACE)
            near = ((x[..., 2] - pl[:, -1:, 2]) <= B["ha"][:, None]) & \
                   ((x[..., :2] - pl[:, -1:, :2]).norm(dim=-1) <= B["hl"][:, None])   # and within one own length of it (xy): a suffix
            fin = torch.cummin(near.int().flip(1), dim=1).values.bool().flip(1)
        z = torch.where(fin[..., None], zgrade_t(sep_e + HIT_ZONE), zgrade_t(sep_e)).amax(-1)            # final: penetration only
    else:
        z = zgrade_t(sep_e).amax(-1)                                                                    # (N, 33)
    rd = ((s >= 0.05) & (s <= 0.95)).float().detach()
    hit = (z * rd).sum(1) / rd.sum(1).clamp(min=1)
    band = torch.zeros_like(hit)
    if BAND > 0:                                                                   # surrogate only: objects within eps of the start or the
        with torch.no_grad():                                                      # end pose are exempt while they stay within eps of it
            cb = sep < BAND                                                        # (the endpoints are fixed: no penalty can move them)
            exb = torch.cummin(cb.int(), dim=1).values.bool() | torch.cummin(cb.int().flip(1), dim=1).values.bool().flip(1)
        zb = chomp_n(torch.where(exb | fin[..., None], torch.full_like(sep, float("inf")), sep), BAND).amax(-1)
        band = (zb * rd).sum(1) / rd.sum(1).clamp(min=1)
    straight = (pl[:, -1] - B["start"]).norm(dim=-1).clamp(min=0.02)
    econ = F.relu(L / straight - 1)
    smooth = ((pl[:, 2:] - 2 * pl[:, 1:-1] + pl[:, :-2]) / 0.01).pow(2).sum(-1).mean(1)
    table = (F.relu(-(x[..., 2] + B["bot"][:, None]) - 0.005) / 0.01).mean(1)
    return dict(hit=hit, band=band, econ=econ, smooth=smooth, table=table, final_frac=fin.float().mean(1),
                cost=W_HIT * hit + W_BAND * band + W_ECON * econ + W_SMOOTH * smooth + W_TABLE * table)


def optimise(ts: TorchSDF, items, steps=STEPS, lr=LR, max_cand=MAX_CAND):
    """items -> (list of optimised plans (n, 33, 3) per item [None if skipped], log dict). The start and the last point stay fixed.
    Items are optimised in batches of <= max_cand candidates (independent problems; the batching only bounds GPU memory)."""
    res, hist, n_c, dt = [None] * len(items), [], 0, 0.0
    chunks, cur, nc = [], [], 0
    for it, d in enumerate(items):
        if cur and nc + len(d["plans"]) > max_cand:
            chunks.append(cur); cur, nc = [], 0
        cur.append(it); nc += len(d["plans"])
    if cur:
        chunks.append(cur)
    for ch in chunks:
        B = build_batch(ts, [items[i] for i in ch])
        if B is None:
            continue
        pl0 = B["plans"]
        inner = pl0[:, 1:-1].clone().requires_grad_(True)
        opt = torch.optim.Adam([inner], lr=lr)
        if ts.dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); h_ = []
        for st in range(steps):
            pl = torch.cat([pl0[:, :1], inner, pl0[:, -1:]], 1)
            P = path_cost(ts, B, pl)
            opt.zero_grad(set_to_none=True); P["cost"].sum().backward(); opt.step()
            if st in (0, steps - 1):
                h_.append({k: float(v.sum()) for k, v in P.items()})
        if ts.dev.type == "cuda":
            torch.cuda.synchronize()
        dt += time.time() - t0; n_c += len(pl0); hist.append(h_)
        out = torch.cat([pl0[:, :1], inner.detach(), pl0[:, -1:]], 1).cpu().numpy()
        q = 0
        for j in B["keep"]:
            it = ch[j]; n = len(items[it]["plans"]); res[it] = out[q:q + n]; q += n
    agg = lambda i: {k: sum(h[i][k] for h in hist) / max(n_c, 1) for k in hist[0][0]} if hist else {}   # noqa: E731
    return res, dict(seconds=dt, n_cand=n_c, steps=steps, batches=len(chunks), hist=[agg(0), agg(-1)],
                     peak_mib=float(torch.cuda.max_memory_allocated() / 2**20) if ts.dev.type == "cuda" else None)


def optimise_rows(states, rows, groups, geo, steps, any_bridge, lf=None):
    """For physics.stage --route_opt: every path candidate row's plan -> its optimised plan (row["plan_raw"] keeps the model's own
    sample); choice rows are unchanged. The caller re-scores with the exact MeshCost."""
    from cleantables.worldmodel.common import log
    dev = torch.device(os.environ.get("WM_PHYS_DEV", "cpu"))
    by = {}
    for G in groups:
        for k in G["path"]:
            by.setdefault((rows[k]["si"], rows[k]["a"]), []).append(k)
    items = []
    for (si, a), ks in sorted(by.items()):
        s = states[si]; B_ = any_bridge(s["ep"])
        items.append(dict(S=B_.S, xyz=s["P"], yaw=s["Y"], a=a, plans=np.stack([rows[k]["plan"] for k in ks]),
                          dyaws=np.array([rows[k]["dyaw"] for k in ks]), ks=ks))
    res, info = optimise(TorchSDF(geo, dev), items, steps=steps)
    n = 0
    for d, r in zip(items, res):
        if r is None:
            continue
        for k, pl in zip(d["ks"], r):
            rows[k]["plan_raw"] = rows[k]["plan"]; rows[k]["plan"] = pl.astype(np.float32); n += 1
    if lf is not None:
        log(f"   route-opt: {n} path candidates x {steps} steps on {dev} ({info.get('seconds', 0):.1f} s, {info.get('batches')} batches); "
            f"surrogate {info.get('hist', [{}])[0].get('cost', float('nan')):.2f} -> {info.get('hist', [{}])[-1].get('cost', float('nan')):.2f}", lf)
    return info


# ------------------------------------------------------------------ prototype on one round's candidates (no training)
def lift_stats(plans, start_z=None):
    """Peak lift (max z - start z, the root = the object's bottom) and flat (within 1 cm of its start height > 50% of the path)."""
    P = np.asarray(plans)
    z = P[..., 2] - (P[..., :1, 2] if start_z is None else start_z[:, None])
    return z.max(-1), (np.abs(z) < 0.01).mean(-1) > 0.5


def proto(out, physx=False, n_states=48, steps=STEPS, dev="cuda"):
    from cleantables.worldmodel.physics import stage as PS_
    from cleantables.worldmodel.physics.screen import ensure_occ
    from cleantables.worldmodel.bridge import load_models
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    W = Path(os.environ.get("WM_EVAL_OUT", CT_DATA + "/wm_eval")) / "wm_v2_world"
    PS_.PATH_ONLY = True
    cf = out / "cand.pt"
    models = [load_models(W / "runs_final" / "watch_notab", seeds=(s,))[0].eval() for s in (0, 1, 2)]
    if cf.exists():
        C_ = torch.load(cf, weights_only=False); states, rows, groups = C_["states"], C_["rows"], C_["groups"]
        for s in states:
            PS_.bridge(s["ep"], models[:1])
    else:                                                                         # candidates as in round 1 of a path-only stage run (k_end 0)
        pool = torch.load(W / "pf" / "watch_notab_phys" / "pool_r0.pt", weights_only=False)
        rng = np.random.default_rng(0 * 1000 + 1)
        states = PS_.pick_states(pool, n_states, rng)
        gens = [torch.Generator().manual_seed(0 * 1000 + 97 * 1 + k) for k in range(3)]
        torch.manual_seed(0 * 1000 + 1)
        a_ = argparse.Namespace(cand=3, temp=1.5, k_path=8, k_end=0, diverse=True)
        t0 = time.time(); rows, groups = PS_.propose(models, states, a_, rng, gens)
        print(f"propose: {len(rows)} rows, {len(groups)} groups ({time.time() - t0:.0f} s)", flush=True)
        torch.save(dict(states=states, rows=rows, groups=groups), cf)
    ensure_occ({str(PS_.any_bridge(s["ep"]).S["objects"][k]["usd"]) for s in states for k in range(len(s["P"]))})
    geo = MeshCost(w_drag=0.0)
    ts = TorchSDF(geo, torch.device(dev))
    by = {}
    for G in groups:
        for k in G["path"]:
            by.setdefault((rows[k]["si"], rows[k]["a"]), []).append(k)
    keys = sorted(by)
    items = []
    for si, a in keys:
        s = states[si]; B_ = PS_.any_bridge(s["ep"]); ks = by[(si, a)]
        items.append(dict(S=B_.S, xyz=s["P"], yaw=s["Y"], a=a, plans=np.stack([rows[k]["plan"] for k in ks]),
                          dyaws=np.array([rows[k]["dyaw"] for k in ks]), ks=ks, si=si))

    def exact(plans_of):
        t0 = time.time(); R = {}
        for it, d in enumerate(items):
            if plans_of[it] is None:
                continue
            r = geo.score(("x", it), d["S"], d["xyz"], d["yaw"], d["a"], plans_of[it], dyaws=d["dyaws"])
            for q, k in enumerate(d["ks"]):
                if r["ok"][q]:
                    R[k] = dict(hit=float(r["geo_hit"][q]), econ=float(r["econ3d"][q]), cost=float(W_HIT * r["geo_hit"][q] + W_ECON * r["econ3d"][q]))
        return R, time.time() - t0
    before, t_cpu = exact([d["plans"] for d in items])
    print(f"exact geo v3 (CPU), before: {len(before)} candidates ({t_cpu:.1f} s)", flush=True)
    for d in items:                                                                # build the fields once, outside the timing
        [ts.idx(d["S"]["objects"][j]) for j in range(len(d["xyz"]))]
    ts.tensors()
    if ts.dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt_plans, info = optimise(ts, items, steps=steps)
    print(f"optimised {info.get('n_cand')} candidates x {steps} steps on {dev}: {info.get('seconds', 0):.1f} s, peak "
          f"{info.get('peak_mib')} MiB; surrogate {info.get('hist', [{}])[0]} -> {info.get('hist', [{}])[-1]}", flush=True)
    after, _ = exact(opt_plans)
    # surrogate vs exact cost before optimisation (surrogate fidelity)
    with torch.no_grad():
        Bb = build_batch(ts, items); Ps = path_cost(ts, Bb, Bb["plans"], soft=False)
    ks_flat = [k for it in Bb["keep"] for k in items[it]["ks"]]
    sur = {k: float(h) for k, h in zip(ks_flat, Ps["hit"].cpu().numpy())}
    cm = [k for k in ks_flat if k in before]
    from scipy.stats import spearmanr
    rep = dict(n_states=len(states), n_groups=len(groups), n_path_candidates=len(cm), steps=steps, lr=LR, m_pts=M_PTS, band_eps=BAND, w_smooth=W_SMOOTH,
               w_table=W_TABLE, gpu=dict(seconds=info.get("seconds"), cand_per_min=60 * info.get("n_cand", 0) / max(info.get("seconds", 1), 1e-6),
                                         peak_mib=info.get("peak_mib")), cpu_exact_seconds=t_cpu,
               surrogate_vs_exact_hit_spearman=float(spearmanr([sur[k] for k in cm], [before[k]["hit"] for k in cm])[0]))

    def summ(R, plans_of, sel):
        ks = [k for k in sel if k in R]
        P_ = {k: pl for it, d in enumerate(items) if plans_of[it] is not None for k, pl in zip(d["ks"], plans_of[it])}
        lf, fl = lift_stats(np.stack([P_[k] for k in ks]))
        h = np.array([R[k]["hit"] for k in ks]); e = np.array([R[k]["econ"] for k in ks])
        return dict(n=len(ks), hit_mean=float(h.mean()), hit_ge_05=float((h >= 0.5).mean()), econ_mean=float(e.mean()),
                    lift_cm_median=100 * float(np.median(lf)), lift_cm_p90=100 * float(np.percentile(lf, 90)), flat_share=float(fl.mean()))
    orig = [d["plans"] for d in items]
    best_b = [min((k for k in G["path"] if k in before), key=lambda k: before[k]["cost"], default=None) for G in groups]
    best_a = [min((k for k in G["path"] if k in after), key=lambda k: after[k]["cost"], default=None) for G in groups]
    rep["all_candidates"] = dict(before=summ(before, orig, cm), after=summ(after, opt_plans, cm))
    rep["model_plan_k0"] = dict(before=summ(before, orig, [G["path"][0] for G in groups]), after=summ(after, opt_plans, [G["path"][0] for G in groups]))
    rep["best_per_group"] = dict(before=summ(before, orig, [k for k in best_b if k is not None]),
                                 after=summ(after, opt_plans, [k for k in best_a if k is not None]))
    hum = W / "logs" / "lift_report.json"
    rep["human_contact_moves"] = json.load(open(hum)).get("human_contact_moves") if hum.exists() else None
    P_after = {k: pl for it, d in enumerate(items) if opt_plans[it] is not None for k, pl in zip(d["ks"], opt_plans[it])}
    torch.save(dict(after=after, before=before, opt=P_after, best_b=best_b, best_a=best_a), out / "opt.pt")
    if physx:                                                                     # PhysX check: each group's best path before vs after
        from cleantables.worldmodel.physics.chains import run_chain_physx
        rb = [dict(rows[k], si=rows[k]["si"]) for k in best_b if k is not None]
        ra = [dict(rows[k], plan=P_after[k].astype(np.float32)) for k in best_a if k is not None and k in P_after]
        both = rb + ra
        n = PS_.write_jobs(states, both, out / "carries.npz", json.load(open(W / "pf" / "watch_notab_geo2" / "cfg.json"))["traj_t"], audit=True)
        t0 = time.time(); S = run_chain_physx(out / "carries.npz", out / "carry_scores.npz", 16)
        hg = np.nan_to_num(S["hit_g"][:, 0])
        hb, ha = hg[[x["j"] for x in rb]], hg[[x["j"] for x in ra]]
        gb = np.array([before[k]["hit"] for k in best_b if k is not None]); ga = np.array([after[k]["hit"] for k in best_a if k is not None and k in P_after])
        rep["physx_check"] = dict(n_before=len(rb), n_after=len(ra), seconds=time.time() - t0,
                                  physx_hit_before=float(hb.mean()), physx_hit_after=float(ha.mean()),
                                  physx_hit_ge_05_before=float((hb >= 0.5).mean()), physx_hit_ge_05_after=float((ha >= 0.5).mean()),
                                  geo_hit_before=float(gb.mean()), geo_hit_after=float(ga.mean()),
                                  geo_vs_physx_spearman_after=float(spearmanr(ga, ha)[0]))
        ps = S["pose_start"]; pe = S["pose_end"]
        rep["physx_check"]["others_moved_cm_mean"] = {}
        for nm, rr in (("before", rb), ("after", ra)):
            dd = []
            for x in rr:
                I = len(states[x["si"]]["P"]); o = [j for j in range(I) if j != x["a"]]
                dd.append(100 * float(np.nan_to_num(np.linalg.norm(pe[x["j"], o, :3] - ps[x["j"], o, :3], axis=1)).max()) if o else 0.0)
            rep["physx_check"]["others_moved_cm_mean"][nm] = float(np.mean(dd))
    json.dump(rep, open(out / "route_opt_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


def from_saved(out, physx=False):
    """Prototype report from its saved cand.pt + opt.pt (no re-optimisation); physx: the deferred PhysX check (each group's best
    exact-cost path before vs after), run under the GPU memory guard."""
    import re
    from scipy.stats import spearmanr
    from cleantables.worldmodel.physics import stage as PS_
    out = Path(out); W = Path(os.environ.get("WM_EVAL_OUT", CT_DATA + "/wm_eval")) / "wm_v2_world"
    C_ = torch.load(out / "cand.pt", weights_only=False); states, rows, groups = C_["states"], C_["rows"], C_["groups"]
    O_ = torch.load(out / "opt.pt", weights_only=False); before, after, P_after = O_["before"], O_["after"], O_["opt"]
    best_b, best_a = O_["best_b"], O_["best_a"]
    ln = [x for x in open(out / "log.txt") if x.startswith("optimised ")]
    m_ = re.match(r"optimised (\d+) candidates x (\d+) steps on (\S+): ([\d.]+) s, peak ([\d.]+|None) MiB", ln[-1]) if ln else None

    def summ(R, plans, sel):
        ks = [k for k in sel if k in R and k in plans]
        lf, fl = lift_stats(np.stack([plans[k] for k in ks]))
        h = np.array([R[k]["hit"] for k in ks]); e = np.array([R[k]["econ"] for k in ks])
        return dict(n=len(ks), hit_mean=float(h.mean()), hit_ge_05=float((h >= 0.5).mean()), hit_zero=float((h == 0).mean()),
                    econ_mean=float(e.mean()), lift_cm_median=100 * float(np.median(lf)), lift_cm_p90=100 * float(np.percentile(lf, 90)),
                    flat_share=float(fl.mean()))
    P_b = {k: rows[k]["plan"] for k in before}
    cm = [k for k in before if k in after]
    k0 = [G["path"][0] for G in groups]
    rep = dict(n_states=len(states), n_groups=len(groups), n_path_candidates=len(cm), steps=STEPS, lr=LR, m_pts=M_PTS, w_smooth=W_SMOOTH, w_table=W_TABLE,
               band_eps=BAND, w_band=W_BAND)
    if m_:
        n_, st_, dev_, sec_, pk_ = m_.groups()
        rep["gpu"] = dict(device=dev_, n_cand=int(n_), steps=int(st_), seconds=float(sec_), peak_mib=None if pk_ == "None" else float(pk_),
                          cand_per_min_150_steps=60 * int(n_) / float(sec_), ms_per_cand_step=1000 * float(sec_) / int(n_) / int(st_),
                          batch=MAX_CAND, s_per_batch=float(sec_) / -(-int(n_) // MAX_CAND))
    rep["all_candidates"] = dict(before=summ(before, P_b, cm), after=summ(after, P_after, cm))
    rep["model_plan_k0"] = dict(before=summ(before, P_b, k0), after=summ(after, P_after, k0))
    rep["best_per_group"] = dict(before=summ(before, P_b, [k for k in best_b if k is not None]),
                                 after=summ(after, P_after, [k for k in best_a if k is not None]))
    d_lift = [100 * (lift_stats(P_after[k][None])[0][0] - lift_stats(P_b[k][None])[0][0]) for k in cm]
    rep["lift_change_cm"] = dict(median=float(np.median(d_lift)), p10=float(np.percentile(d_lift, 10)), p90=float(np.percentile(d_lift, 90)),
                                 share_raised_ge_1cm=float((np.array(d_lift) >= 1).mean()))
    hb = np.array([before[k]["hit"] for k in cm]); ha = np.array([after[k]["hit"] for k in cm])
    rep["hit_transitions"] = dict(hit_before_ge_05=int((hb >= 0.5).sum()), of_those_after_lt_05=int(((hb >= 0.5) & (ha < 0.5)).sum()),
                                  clean_before_lt_05_now_ge_05=int(((hb < 0.5) & (ha >= 0.5)).sum()))
    hum = W / "logs" / "lift_report.json"
    rep["human_contact_moves"] = json.load(open(hum)).get("human_contact_moves") if hum.exists() else None
    if physx:
        from cleantables.worldmodel.physics.chains import run_chain_physx
        rb = [dict(rows[k]) for k in best_b if k is not None]
        ra = [dict(rows[k], plan=P_after[k].astype(np.float32)) for k in best_a if k is not None and k in P_after]
        PS_.write_jobs(states, rb + ra, out / "carries.npz", json.load(open(W / "pf" / "watch_notab_geo2" / "cfg.json"))["traj_t"], audit=True)
        t0 = time.time(); S = run_chain_physx(out / "carries.npz", out / "carry_scores.npz", 16)
        hg = np.nan_to_num(S["hit_g"][:, 0]); hB, hA = hg[[x["j"] for x in rb]], hg[[x["j"] for x in ra]]
        ga = np.array([after[k]["hit"] for k in best_a if k is not None and k in P_after])
        rep["physx_check"] = dict(n_before=len(rb), n_after=len(ra), seconds=time.time() - t0, physx_hit_before=float(hB.mean()),
                                  physx_hit_after=float(hA.mean()), physx_hit_ge_05_before=float((hB >= 0.5).mean()),
                                  physx_hit_ge_05_after=float((hA >= 0.5).mean()), geo_vs_physx_spearman_after=float(spearmanr(ga, hA)[0]))
    json.dump(rep, open(out / ("route_opt_report_physx.json" if physx else "route_opt_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True, help="output folder"); ap.add_argument("--physx", action="store_true")
    ap.add_argument("--states", type=int, default=48); ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from_saved", action="store_true", help="report from the saved cand.pt + opt.pt (with --physx: the held check)")
    a = ap.parse_args()
    if a.from_saved:
        return from_saved(a.proto, physx=a.physx)
    proto(a.proto, physx=a.physx, n_states=a.states, steps=a.steps, dev=a.device)


if __name__ == "__main__":
    main()
