"""Physical plausibility of the world model's imagined carries in the per-episode twins, scored over the full workspace.

FullScorer reuses the occupancy data and the imagine() sampler of cleantables.worldmodel.physics.feedback_base and differs from its
Scorer in three ways:
  * the grid covers the whole workspace: table +-15 cm, from 2 cm below the table top up to Z_TOP (1.2 m). A carry voxel outside
    that box counts as "out of workspace" (reported separately; the truncated grid treats it as free space);
  * the truncated score (0.5 m grid, out-of-grid = free) is also computed, since it is the proxy that feedback_base optimises; the
    two together give the exploitability gap;
  * clearance (cm) between the moving object and the static scene during the carry (path steps 4..27 of 32; pick-up and set-down
    contacts are intended), peak height of the object's bottom above the table, lift above the higher of start / end, and path
    length vs the straight start -> end distance. Human reference ranges come from the human fruit / tools carries in tasks3c.npz.

  python -m cleantables.worldmodel.physics.twin_check --runs RUN@heldout RUN [RUN ...]
      -> OUT/twin_phys/<run>.npz (per-sample rows) + OUT/twin_phys/summary.json. Resumable per run.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.common import HELD_TWINS, OUT, ROOT, RUNS, SPLIT_NOTE, jdump, log, twin_fold

PITCH = 0.005
Z_TOP = 1.2
OLD_TOP = 0.5
MID = slice(4, 28)                         # carry part of the 32-step path (clearance only)


class FullScorer:
    def __init__(self, ep, dev="cuda"):
        self.dev = dev
        self.S = json.load(open(ROOT / "data/twin_table/ep" / ep / "scene.json"))
        Z = np.load(ROOT / "data/twin_table/ep" / ep / "occ.npz")
        self.occ = {k: torch.as_tensor(Z[k], device=dev) for k in Z.files}
        T = self.S["table"]
        self.lo = torch.tensor([T["x"][0] - 0.15, T["y"][0] - 0.15, -0.02], device=dev)
        self.shape = (int((T["x"][1] - T["x"][0] + 0.3) / PITCH) + 1, int((T["y"][1] - T["y"][0] + 0.3) / PITCH) + 1, int((Z_TOP + 0.02) / PITCH))
        self.nz_old = int((OLD_TOP + 0.0) / PITCH)                        # truncated grid: int(0.5 / PITCH) cells from z = -0.02

    def grid(self, ids, pos, skip):
        G = torch.zeros(self.shape, dtype=torch.bool, device=self.dev)
        for j, oid in enumerate(ids):
            if j == skip or oid not in self.occ:
                continue
            q = self.occ[oid] + torch.as_tensor(pos[j], device=self.dev)
            ix = ((q - self.lo) / PITCH).round().long()
            ok = (ix >= 0).all(1) & (ix[:, 0] < self.shape[0]) & (ix[:, 1] < self.shape[1]) & (ix[:, 2] < self.shape[2])
            ix = ix[ok]; G[ix[:, 0], ix[:, 1], ix[:, 2]] = True
        return G

    def clearance_field(self, G, cap=0.10):
        """Distance (m) from every cell to the nearest occupied cell or the table top, on a 1 cm grid up to 0.5 m (capped)."""
        from scipy.ndimage import distance_transform_edt
        g = G[:, :, : self.nz_old].float()[None, None]
        g2 = F.max_pool3d(g, 2)[0, 0].bool().cpu().numpy()
        g2[:, :, 0] = True                                                     # the table (z < 0) counts as an obstacle
        D = distance_transform_edt(~g2) * 2 * PITCH
        return torch.as_tensor(np.minimum(D, cap), device=self.dev, dtype=torch.float32)

    def score(self, G, D, oid, paths_bottom, erode=1):
        """paths_bottom (K,T,3). Returns dict of (K,) tensors."""
        P = self.occ[oid]
        c = P.mean(0)
        d = (P - c).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        Pe = P - (P - c) / d * (erode * PITCH)
        out = {k: [] for k in ("pen_full", "pen_old", "out_ws", "above_old", "clear_cm")}
        for k0 in range(0, paths_bottom.shape[0], 8):                          # chunk over samples (memory)
            pb = paths_bottom[k0:k0 + 8]
            Q = Pe[None, None] + pb[:, :, None, :]
            ix = ((Q - self.lo) / PITCH).round().long()
            inb = (ix >= 0).all(-1) & (ix[..., 0] < self.shape[0]) & (ix[..., 1] < self.shape[1]) & (ix[..., 2] < self.shape[2])
            ixc = torch.stack([ix[..., a].clamp(0, self.shape[a] - 1) for a in range(3)], -1)
            hit = G[ixc[..., 0], ixc[..., 1], ixc[..., 2]] & inb
            below = Q[..., 2] < -PITCH
            in_old = inb & (ix[..., 2] < self.nz_old)
            out["pen_full"].append((hit | below).float().sum(-1).mean(-1))
            out["pen_old"].append(((hit & in_old) | below).float().sum(-1).mean(-1))
            out["out_ws"].append((~inb & ~below).float().mean((-1, -2)))
            out["above_old"].append((Q[..., 2] > OLD_TOP - 0.02).float().mean((-1, -2)))
            # clearance of the (un-eroded) object during the carry
            Qf = P[None, None] + pb[:, MID, None, :]
            j = ((Qf - self.lo) / (2 * PITCH)).floor().long()
            okj = (j >= 0).all(-1) & (j[..., 0] < D.shape[0]) & (j[..., 1] < D.shape[1]) & (j[..., 2] < D.shape[2])
            jc = torch.stack([j[..., a].clamp(0, D.shape[a] - 1) for a in range(3)], -1)
            dv = torch.where(okj, D[jc[..., 0], jc[..., 1], jc[..., 2]], torch.full_like(Qf[..., 0], 0.10))
            dv = torch.where(Qf[..., 2] < 0, torch.zeros_like(dv), dv)
            out["clear_cm"].append(100 * dv.amin(-1).amin(-1))
        return {k: torch.cat(v) for k, v in out.items()}


def geometry(paths_top, H, org_top):
    """Per sample: peak bottom height (cm above the table), lift above the higher of the object's true start and the path end (cm;
    using the path's own first point would hide a path that starts in the air), start gap = distance of the path's first point from
    where the object really is (cm; objects cannot teleport), length / straight distance."""
    z = paths_top[..., 2] - H
    peak = 100 * z.amax(-1)
    lift = 100 * (paths_top[..., 2].amax(-1) - torch.maximum(org_top[2].expand(paths_top.shape[0]), paths_top[:, -1, 2]))
    gap = 100 * (paths_top[:, 0] - org_top).norm(dim=-1)
    seg = (paths_top[:, 1:] - paths_top[:, :-1]).norm(dim=-1).sum(-1)
    straight = (paths_top[:, -1] - paths_top[:, 0]).norm(dim=-1).clamp(min=0.02)
    return dict(peak_cm=peak, lift_cm=lift, start_gap_cm=gap, len_ratio=seg / straight, len_cm=100 * seg)


def human_reference():
    """Lift and length ratio of the human carries (fruit + tools moves in tasks3c, tracked object top centre)."""
    z = np.load(ROOT / "data/uwm/tasks3c.npz")
    ok = (z["mover"] >= 0) & np.isin(z["dataset"], ["fruit", "tools"]) & np.isfinite(z["path_obj"]).all((1, 2))
    P = z["path_obj"][ok].astype(np.float64)
    # one row per move event (snapshots repeat the same move); key on episode + event id
    key = np.char.add(z["group"][ok].astype(str), z["ev_id"][ok].astype(str))
    _, first = np.unique(key, return_index=True); P = P[first]
    lift = 100 * (P[..., 2].max(-1) - np.maximum(P[:, 0, 2], P[:, -1, 2]))
    L = np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(-1); S = np.maximum(np.linalg.norm(P[:, -1] - P[:, 0], axis=-1), 0.02)
    r = L / S
    q = lambda x: {f"p{p}": float(np.percentile(x, p)) for p in (50, 90, 95, 99)} | {"max": float(x.max())}
    return dict(n_moves=int(len(P)), lift_cm=q(lift), len_ratio=q(r), len_cm=q(100 * L))


def run_one(spec, n_layouts=8, K=8, seed=1, dev="cuda"):
    """spec = 'RUN' (final fold -1 members) or 'RUN@heldout' (the fold model that never saw the episode; base only); a '#hashed' suffix
    feeds the hashed instruction vector the models were trained on instead of the bridge's MiniLM vector."""
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel.physics.feedback_base import imagine
    from cleantables.worldmodel.common import hashed
    spec, _, instr = spec.partition("#")
    run, _, mode = spec.partition("@")
    rd = RUNS / run if (RUNS / run).exists() else OUT / run
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("ep", "obj", "layout", "is_median", "pen_full", "pen_old", "out_ws", "above_old", "clear_cm", "peak_cm", "lift_cm",
                            "start_gap_cm", "len_ratio", "len_cm")}
    for ep in HELD_TWINS:
        f = twin_fold(ep) if mode == "heldout" else -1
        B = W.WMBridge(ep, ckpts=[rd / f"fold{f}_seed{s}.pt" for s in range(3) if (rd / f"fold{f}_seed{s}.pt").exists()], device=dev)
        if instr == "hashed":
            B.Z = dict(B.Z); B.Z["task_emb"] = hashed(str(B.Z["instruction"]))
        sc = FullScorer(ep, dev)
        Z = B.Z; ids = [str(x) for x in Z["ids"]]; H = torch.as_tensor(Z["extents"][:, 2], device=dev)
        task = [i for i, r in enumerate(Z["role"]) if str(r) == "task"]
        for L in range(n_layouts):
            P0 = Z["start_pos"].copy() if L == 0 else W.shuffle_layout(Z, B.S, rng, task)
            if P0 is None:
                continue
            for i in task:
                torch.manual_seed(1000 * L + i)
                paths, end, d, _, _ = imagine(B, P0, i, K)                     # (M*K,32,3) top centre, re-anchored to the world-model end (teacher logic)
                med = paths.median(0).values[None]
                allp = torch.cat([paths, med], 0)
                G = sc.grid(ids, P0, i); D = sc.clearance_field(G)
                pb = allp.clone(); pb[..., 2] -= H[i]
                s = sc.score(G, D, ids[i], pb)
                org = torch.as_tensor(P0[i], device=dev, dtype=torch.float32).clone(); org[2] += H[i]
                gm = geometry(allp, H[i], org)
                n = allp.shape[0]
                for k, v in {**s, **gm}.items():
                    rows[k].append(v.float().cpu().numpy())
                rows["ep"].append(np.array([ep] * n)); rows["obj"].append(np.array([ids[i]] * n)); rows["layout"].append(np.full(n, L))
                rows["is_median"].append(np.r_[np.zeros(n - 1, bool), True])
        del B, sc; torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in rows.items()}


def summarise(R, HR):
    out = {}
    for who, sel in (("samples", ~R["is_median"]), ("followed_median_path", R["is_median"])):
        x = {k: R[k][sel] for k in R}
        cls = np.where(np.char.startswith(x["ep"].astype(str), "IMG_2516"), "fruit", "tools")
        o = {}
        for c in ("all", "fruit", "tools"):
            s = np.ones(len(cls), bool) if c == "all" else cls == c
            pen_ok_old = x["pen_old"][s] <= 0.5
            plaus = (x["pen_full"][s] <= 0.5) & (x["out_ws"][s] == 0) & (x["lift_cm"][s] <= HR["lift_cm"]["p99"]) & (x["len_ratio"][s] <= HR["len_ratio"]["p99"])
            o[c] = dict(n=int(s.sum()),
                        collide_old_grid=float((~pen_ok_old).mean()), collide_full=float((x["pen_full"][s] > 0.5).mean()),
                        pen_full_mean_vox=float(x["pen_full"][s].mean()), out_of_workspace_share=float((x["out_ws"][s] > 0).mean()),
                        above_old_ceiling_share=float((x["above_old"][s] > 0).mean()),
                        clearance_cm_median=float(np.median(x["clear_cm"][s])), grazing_lt1cm_share=float((x["clear_cm"][s] < 1.0).mean()),
                        peak_cm=dict(p50=float(np.median(x["peak_cm"][s])), p90=float(np.percentile(x["peak_cm"][s], 90)), max=float(x["peak_cm"][s].max())),
                        lift_cm=dict(p50=float(np.median(x["lift_cm"][s])), p90=float(np.percentile(x["lift_cm"][s], 90)), max=float(x["lift_cm"][s].max())),
                        share_peak_gt_30cm=float((x["peak_cm"][s] > 30).mean()), share_peak_gt_48cm=float((x["peak_cm"][s] > 48).mean()),
                        share_peak_gt_65cm=float((x["peak_cm"][s] > 65).mean()),
                        lift_above_human_p99_share=float((x["lift_cm"][s] > HR["lift_cm"]["p99"]).mean()),
                        start_gap_cm_p50=float(np.median(x["start_gap_cm"][s])), start_gap_gt5cm_share=float((x["start_gap_cm"][s] > 5).mean()),
                        len_ratio_p50=float(np.median(x["len_ratio"][s])), len_ratio_above_human_p99_share=float((x["len_ratio"][s] > HR["len_ratio"]["p99"]).mean()),
                        proxy_ok=float(pen_ok_old.mean()), plausible=float(plaus.mean()),
                        plausible_and_starts_on_object=float((plaus & (x["start_gap_cm"][s] <= 5)).mean()))
        out[who] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["tasks3_v4@heldout", "tasks3_v4", "tasks3_v4_pf", "tasks3_v4_pf2", "tasks3_v4_pf3"])
    ap.add_argument("--layouts", type=int, default=8); ap.add_argument("--K", type=int, default=8)
    a = ap.parse_args()
    od = OUT / "twin_phys"; od.mkdir(parents=True, exist_ok=True); lf = od / "log.txt"
    HR = human_reference()
    summ = json.load(open(od / "summary.json")) if (od / "summary.json").exists() else {}
    summ["split"] = SPLIT_NOTE; summ["human_reference"] = HR
    for spec in a.runs:
        f = od / f"{spec.replace('@', '_').replace('#', '_')}.npz"
        if f.exists():
            R = dict(np.load(f)); log(f"{spec}: cached", lf)
        else:
            t0 = time.time(); R = run_one(spec, a.layouts, a.K)
            np.savez_compressed(f, **R); log(f"{spec}: {len(R['ep'])} rows, {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", lf)
        summ[spec] = summarise(R, HR)
        jdump(summ, od / "summary.json")
    # exploitability gap vs the base final model: change in the optimised proxy vs change in independent plausibility
    base = summ.get("tasks3_v4")
    if base:
        gap = {}
        for spec in summ:
            if spec in ("split", "human_reference", "xgap", "tasks3_v4") or not isinstance(summ[spec], dict) or "samples" not in summ[spec]:
                continue
            g = {}
            for who in ("followed_median_path", "samples"):
                for c in ("all", "fruit", "tools"):
                    b, r = base[who][c], summ[spec][who][c]
                    dp, di = r["proxy_ok"] - b["proxy_ok"], r["plausible"] - b["plausible"]
                    g.setdefault(who, {})[c] = dict(proxy_gain_pts=100 * dp, plausible_gain_pts=100 * di, xgap_pts=100 * (dp - di))
            gap[spec] = g
        summ["xgap"] = gap
        jdump(summ, od / "summary.json")
    log("done", lf)


if __name__ == "__main__":
    main()
