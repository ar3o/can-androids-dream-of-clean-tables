"""Checks whether the world model's carry paths are physically followable (CPU only).

Rebuilds each carry path the way the robot environment does (path carry, K=4 samples, 3D-only):
  goal G (object root) = destination top + top mode of the target-relative end mixture (ensemble mean; for the table,
  start + end displacement), z = top - H clamped at 0; path = median over the 3 x 4 anchored samples (+ object top centre),
  5-point moving average, minus H (root / bottom centre), re-anchored linearly so it ends at G, then G appended (33 points).
Every point is scored against the twin's 5 mm filled occupancy (hollow containers stay hollow, touching = 1 voxel erosion),
and each hit is labelled as the chosen destination, another object or the table. Workspace = table +-15 cm, up to 1.2 m.

  CUDA_VISIBLE_DEVICES= python -m cleantables.worldmodel.physics.carry_check --runs RUN [RUN ...]   -> OUT/carry_check.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.common import OUT, ROOT, RUNS, jdump
from cleantables.worldmodel.physics.twin_check import FullScorer, PITCH

EVAL_TWINS = ["IMG_2516_t11", "IMG_2516_t14", "IMG_2546_t01", "IMG_2547_t01"]


@torch.no_grad()
def eval_path(B, P0, i, K=4):
    from cleantables.worldmodel.base_model import PS
    I = B.I; dev = B.dev
    b = B._batch(torch.as_tensor(P0[None], device=dev), torch.zeros(1, I, device=dev), None)
    mode = torch.ones(1, dtype=torch.long, device=dev); mv = torch.tensor([i], device=dev)
    if getattr(B.models[0], "watch", False):                                # watch model: plan_all path of object i (ends at its T1 goal)
        o = B.plan_all(torch.as_tensor(P0[None], dtype=torch.float32), None, K=K, contact=False)
        return o["path"][0, i].to(dev), int(o["dest"][0, i])
    if getattr(B.models[0], "endref", False):
        # end-referenced model: no destination pointer and no warp; the goal is the ensemble's single end point and the path is
        # built to end there (flow residuals conditioned on the goal plus the straight line to it), so goal and path end coincide
        from cleantables.worldmodel.bridge import dest_from_ref, endref_goal, endref_paths
        Es = []
        for m in B.models:
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]; Es.append(E)
        goal, j, _, _ = endref_goal(B.models, Es, mv)
        H = float(B.Z["extents"][i, 2])
        goal = goal.clone(); goal[:, 2] = goal[:, 2].clamp(min=H)                   # not below the table (numeric guard), before the path
        tp, _ = endref_paths(B.models, Es, mv, goal, K=K)
        G = goal[0].clone(); G[2] = G[2] - H
        path = tp[0].clone(); path[:, 2] -= H
        return torch.cat([path, G[None]], 0), int(dest_from_ref(j, mv, b["obj_kind"])[0])
    Es, dp = [], []
    for m in B.models:
        E = m.encode(b, mode); E["kind"] = b["obj_kind"]; _, tl = m.pointers(E, mv)
        Es.append(E); dp.append(F.softmax(torch.cat([tl[:, :I], tl[:, 24:25]], -1), -1))
    d = int(torch.stack(dp).mean(0).argmax(-1)); tg = torch.tensor([d if d < I else 24], device=dev)
    ee, ps = [], []
    for m, E in zip(B.models, Es):
        c, _, mdn = m.heads_from(E, mv, tg); z = m.end_disp(c)
        off = 0.1 * mdn["mu"][0, int(mdn["logit"].argmax(-1))]
        ee.append(b["obj_geo"][0, d, :3] + off if d < I else b["obj_geo"][0, i, :3] + z["mu"][0, int(z["logit"].argmax(-1))] * PS)
        ps.append(m.sample_paths_anchored(E, c, K=K)[..., :3])
    H = float(B.Z["extents"][i, 2])
    G = torch.stack(ee).mean(0).clone(); G[2] = (G[2] - H).clamp(min=0.0)
    path = torch.cat(ps, 1).median(1).values + b["obj_geo"][:, i, :3][:, None]
    path = F.avg_pool1d(path.transpose(1, 2), 5, 1, 2, count_include_pad=False).transpose(1, 2)
    path[..., 2] -= H
    lin = torch.linspace(0, 1, path.shape[1], device=dev)
    path = path + lin[None, :, None] * (G - path[:, -1])[:, None]
    return torch.cat([path[0], G[None]], 0), d                         # (33,3) root positions


def label_grid(sc, ids, pos, skip):
    L = torch.zeros(sc.shape, dtype=torch.int16)
    for j, oid in enumerate(ids):
        if j == skip or oid not in sc.occ:
            continue
        q = sc.occ[oid] + torch.as_tensor(pos[j], dtype=torch.float32)
        ix = ((q - sc.lo) / PITCH).round().long()
        ok = (ix >= 0).all(1) & (ix[:, 0] < sc.shape[0]) & (ix[:, 1] < sc.shape[1]) & (ix[:, 2] < sc.shape[2])
        ix = ix[ok]; L[ix[:, 0], ix[:, 1], ix[:, 2]] = j + 1
    return L


def hits_per_point(sc, L, oid, path):
    """Per path point (33): object voxels inside each scene object (dict j -> (33,)), below the table, and outside the grid."""
    P = sc.occ[oid]; c = P.mean(0); dd = (P - c).norm(dim=-1, keepdim=True).clamp(min=1e-6); Pe = P - (P - c) / dd * PITCH
    Q = Pe[None] + path[:, None]
    ix = ((Q - sc.lo) / PITCH).round().long()
    inb = (ix >= 0).all(-1) & (ix[..., 0] < sc.shape[0]) & (ix[..., 1] < sc.shape[1]) & (ix[..., 2] < sc.shape[2])
    ixc = torch.stack([ix[..., a].clamp(0, sc.shape[a] - 1) for a in range(3)], -1)
    lab = torch.where(inb, L[ixc[..., 0], ixc[..., 1], ixc[..., 2]].long(), torch.zeros_like(ixc[..., 0]))
    below = Q[..., 2] < -PITCH
    out = {int(j) - 1: (lab == j).sum(-1).numpy() for j in torch.unique(lab) if j > 0}
    return out, below.sum(-1).numpy(), (~inb & ~below).sum(-1).numpy()


def make_bridge(run, ep):
    """Base-model runs (RUNS/<run> or OUT/<run>) load as WMBridge; runs under OUT/wm_v2/runs/<run> load as WMBridge2."""
    from cleantables.worldmodel import base_bridge as W
    v2 = OUT / "wm_v2" / "runs" / run
    if v2.exists() and any(v2.glob("fold-1_seed*.json")) and "cfg" in __import__("json").load(open(sorted(v2.glob("fold-1_seed*.json"))[0])) \
            and run != "B0":
        from cleantables.worldmodel.bridge import WMBridge2
        return WMBridge2(ep, v2, device="cpu")
    rd = v2 if run == "B0" else (RUNS / run if (RUNS / run).exists() else OUT / run)
    return W.WMBridge(ep, ckpts=[p for p in (rd / f"fold-1_seed{s}.pt" for s in range(3)) if p.exists()], device="cpu")


def main(runs=None, layouts=16, twins=None, out_json=None):
    if runs is None:
        ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", default=["tasks3_v4", "tasks3_v4_pf2", "regpf_a01_t2"])
        ap.add_argument("--layouts", type=int, default=16)
        a = ap.parse_args(); runs, layouts = a.runs, a.layouts
    torch.set_grad_enabled(False)
    from cleantables.worldmodel import base_bridge as W
    twins = twins or EVAL_TWINS
    res = {"note": "CPU; paths rebuilt as joint_rl_env_franka.py (v2 skill eval flags); measured layout + random within-table layouts",
           "twins": twins}

    class _A:
        pass
    a = _A(); a.runs, a.layouts = runs, layouts
    for run in a.runs:
        rng = np.random.default_rng(0); rows = []
        for ep in twins:
            B = make_bridge(run, ep)
            sc = FullScorer(ep, "cpu")
            Z = B.Z; ids = [str(x) for x in Z["ids"]]
            task = [i for i, r in enumerate(Z["role"]) if str(r) == "task"]
            T = sc.S["table"]
            for Lk in range(a.layouts):
                P0 = Z["start_pos"].copy() if Lk == 0 else W.shuffle_layout(Z, B.S, rng, task)
                if P0 is None:
                    continue
                for i in task:
                    torch.manual_seed(1000 * Lk + i)
                    path, d = eval_path(B, P0, i)
                    Lg = label_grid(sc, ids, P0, i)
                    hit, below, outside = hits_per_point(sc, Lg, ids[i], path)
                    dest_hits = hit.get(d, np.zeros(33)) if d < B.I else np.zeros(33)
                    other = sum((v for j, v in hit.items() if j != d), np.zeros(33))
                    G = path[-1].numpy(); H = float(Z["extents"][i, 2])
                    mid = slice(3, 30)
                    seg = np.linalg.norm(np.diff(path.numpy(), axis=0), axis=1)
                    tot = dest_hits + other + below
                    end_other = [ids[j] for j, v in hit.items() if j != d and v[-3:].max() > 0]
                    rows.append(dict(ep=ep, obj=ids[i], dest=ids[d] if d < B.I else "table", end_other_ids=",".join(end_other),
                                     carry_max_vox=float(tot[mid].max()), carry_hit_step_frac=float((tot[mid] > 0).mean()),
                                     carry_deep=bool((tot[mid] >= 50).any()),
                                     dest_ok=(str(Z["dest_id"][i]) != "" and d < B.I and str(Z["dest_id"][i]) in ids[d]),
                                     carry_hits_dest=bool(dest_hits[mid].max() > 0), carry_hits_other=bool(other[mid].max() > 0),
                                     carry_below_table=bool(below[mid].max() > 0),
                                     end_in_dest_wall=bool(dest_hits[-3:].max() > 0), end_in_other=bool(other[-3:].max() > 0),
                                     start_in_something=bool((dest_hits[:3] + other[:3]).max() > 0),
                                     end_out_ws=bool(outside[-1] > 0),
                                     end_off_table=bool(not (T["x"][0] <= G[0] <= T["x"][1] and T["y"][0] <= G[1] <= T["y"][1])),
                                     len_cm=float(100 * seg.sum()), straight_cm=float(100 * np.linalg.norm(G - P0[i])),
                                     peak_cm=float(100 * path[:, 2].max()), end_z_cm=float(100 * G[2]),
                                     lift_cm=float(100 * (path[:, 2].max() - max(P0[i][2], G[2])))))
            print(run, ep, len(rows), flush=True)
        R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        cls = np.where(np.char.startswith(R["ep"].astype(str), "IMG_2516"), "fruit", "tools")
        out = {}
        for c in ("all", "fruit", "tools"):
            s = np.ones(len(cls), bool) if c == "all" else cls == c
            q = lambda x: [round(float(np.percentile(x[s], p)), 1) for p in (10, 50, 90)] + [round(float(x[s].max()), 1)]
            coll = R["carry_hits_dest"] | R["carry_hits_other"] | R["carry_below_table"]
            endbad = R["end_in_dest_wall"] | R["end_in_other"] | R["end_out_ws"]
            out[c] = dict(n=int(s.sum()), dest_correct=float(R["dest_ok"][s].mean()),
                          carry_passes_through_anything=float(coll[s].mean()), carry_through_destination_rim_or_wall=float(R["carry_hits_dest"][s].mean()),
                          carry_through_other_object=float(R["carry_hits_other"][s].mean()), carry_below_table=float(R["carry_below_table"][s].mean()),
                          end_inside_destination_wall=float(R["end_in_dest_wall"][s].mean()), end_inside_other_object=float(R["end_in_other"][s].mean()),
                          end_outside_workspace=float(R["end_out_ws"][s].mean()), end_off_table_area=float(R["end_off_table"][s].mean()),
                          end_bad_any=float(endbad[s].mean()), start_overlaps_something=float(R["start_in_something"][s].mean()),
                          clean_path=float((~coll & ~endbad)[s].mean()),
                          carry_deep_ge50_voxels=float(R["carry_deep"][s].mean()),
                          carry_hit_steps_frac_p50_given_hit=float(np.median(R["carry_hit_step_frac"][s & coll])) if (s & coll).any() else 0.0,
                          carry_max_voxels_p50_given_hit=float(np.median(R["carry_max_vox"][s & coll])) if (s & coll).any() else 0.0,
                          end_inside_other_which={k: int(v) for k, v in zip(*np.unique([x for x in R["end_other_ids"][s] if x], return_counts=True))}
                          if any(R["end_other_ids"][s]) else {},
                          len_cm_p10_50_90_max=q(R["len_cm"]), len_over_straight_p50=round(float(np.median((R["len_cm"] / np.maximum(R["straight_cm"], 2))[s])), 2),
                          peak_cm_p10_50_90_max=q(R["peak_cm"]), lift_cm_p10_50_90_max=q(R["lift_cm"]), end_height_cm_p10_50_90_max=q(R["end_z_cm"]))
        res[run] = out
        np.savez_compressed((out_json.parent if out_json else OUT) / f"carry_check_{run}.npz", **R)
        print(run, json.dumps(out["all"]), flush=True)
        jdump(res, out_json or (OUT / "carry_check.json"))
    return res


if __name__ == "__main__":
    main()
