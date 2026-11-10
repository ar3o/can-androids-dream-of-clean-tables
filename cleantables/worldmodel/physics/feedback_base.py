"""Physics feedback on the base world model's imagined carries, run after imitation of the human videos.

For random layouts of the per-episode twins (training episodes only) the world model imagines K carry paths per in-scope object,
with the goal logic the RL teacher uses (ensemble destination + target-relative end, path re-anchored to that end). Each path is
scored on the twins' mesh geometry: the moving object's filled 5 mm voxel occupancy is swept along the path and compared with every
other object's filled occupancy at its pose and with the table top. Overlap deeper than one voxel counts as penetration (touching
or resting = 0). Hollow containers stay hollow (only enclosed space is filled), so ending inside a bowl is not a collision but
passing through its rim is. There are no path-fraction windows and no per-task rules. The flow-matching path head is then
fine-tuned towards the low-penetration samples (reward-weighted flow matching / filtered self-imitation) while the imitation
losses on the human data keep running.

  (sim env: torch + trimesh)  $SIM_PY -m cleantables.worldmodel.physics.feedback_base occ            -> data/twin_table/ep/<ep>/occ.npz (all tools/fruit twins)
                              $SIM_PY -m cleantables.worldmodel.physics.feedback_base score RUN      -> penetration stats of RUN's imagined carries (held-out folds)
                              $SIM_PY -m cleantables.worldmodel.physics.feedback_base finetune RUN OUT [fold ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
EP = ROOT / "data/twin_table/ep"
PITCH = 0.005


def episodes(kind="train"):
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    held = {"IMG_2546_t01", "IMG_2547_t01", "IMG_2548_t01", "IMG_2516_t11", "IMG_2516_t12", "IMG_2516_t13", "IMG_2516_t14"}
    eps = [t["id"] for t in T if t["task"] in ("tools", "fruit")]
    return [e for e in eps if (e in held) == (kind == "heldout")] if kind in ("train", "heldout") else eps


# ------------------------------------------------------------------------------------------------ occupancy (object frame, filled)
def occ_points(mesh_path, pitch=PITCH):
    import trimesh
    m = trimesh.load(mesh_path, force="mesh")
    v = m.voxelized(pitch)
    try:
        v = v.fill()
    except Exception:
        pass
    return np.asarray(v.points, np.float32)                            # (n,3) voxel centres, object frame (bottom at z=0)


def build_occ(ep):
    S = json.load(open(EP / ep / "scene.json"))
    out = {}
    for o in S["objects"]:
        p = EP / ep / "usd" / f"{o['id']}.obj"
        if p.exists():
            out[o["id"]] = occ_points(p)
    np.savez_compressed(EP / ep / "occ.npz", **out)
    return {k: len(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ scorer
class Scorer:
    """World occupancy grid of the static objects of one scene and a sweep test of one moving object along paths (torch, GPU)."""
    def __init__(self, ep, dev="cuda"):
        self.dev = dev
        self.S = json.load(open(EP / ep / "scene.json"))
        Z = np.load(EP / ep / "occ.npz")
        self.occ = {k: torch.as_tensor(Z[k], device=dev) for k in Z.files}
        T = self.S["table"]
        self.lo = torch.tensor([T["x"][0] - 0.15, T["y"][0] - 0.15, -0.02], device=dev)
        self.shape = (int((T["x"][1] - T["x"][0] + 0.3) / PITCH) + 1, int((T["y"][1] - T["y"][0] + 0.3) / PITCH) + 1, int(0.5 / PITCH))

    def grid(self, ids, pos, skip):
        """Bool grid of every object in ids at bottom-centre pos (table frame, yaw baked into the mesh), except index skip."""
        G = torch.zeros(self.shape, dtype=torch.bool, device=self.dev)
        for j, oid in enumerate(ids):
            if j == skip or oid not in self.occ:
                continue
            q = self.occ[oid] + torch.as_tensor(pos[j], device=self.dev)
            ix = ((q - self.lo) / PITCH).round().long()
            ok = (ix >= 0).all(1) & (ix[:, 0] < self.shape[0]) & (ix[:, 1] < self.shape[1]) & (ix[:, 2] < self.shape[2])
            ix = ix[ok]; G[ix[:, 0], ix[:, 1], ix[:, 2]] = True
        return G

    def penetration(self, G, oid, paths_bottom, erode=1):
        """paths_bottom (K,T,3) bottom-centre positions of the moving object -> (K,) mean over path steps of the number of the object's
        voxels lying inside other objects (deeper than `erode` voxels: the object's own points are shrunk towards its centroid by erode*PITCH)
        or below the table top (z < -PITCH)."""
        P = self.occ[oid]
        c = P.mean(0)
        d = (P - c).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        Pe = P - (P - c) / d * (erode * PITCH)                        # shrink: touching contacts do not count
        K, T_, _ = paths_bottom.shape
        Q = Pe[None, None] + paths_bottom[:, :, None, :]              # (K,T,n,3)
        ix = ((Q - self.lo) / PITCH).round().long()
        inb = (ix >= 0).all(-1) & (ix[..., 0] < self.shape[0]) & (ix[..., 1] < self.shape[1]) & (ix[..., 2] < self.shape[2])
        ixc = ix.clamp(min=0)
        ixc[..., 0] = ixc[..., 0].clamp(max=self.shape[0] - 1); ixc[..., 1] = ixc[..., 1].clamp(max=self.shape[1] - 1); ixc[..., 2] = ixc[..., 2].clamp(max=self.shape[2] - 1)
        hit = G[ixc[..., 0], ixc[..., 1], ixc[..., 2]] & inb
        below = Q[..., 2] < -PITCH
        return (hit | below).float().sum(-1).mean(-1)                 # voxels in collision, averaged over the path


# ------------------------------------------------------------------------------------------------ imagined carries (teacher logic)
@torch.no_grad()
def imagine(B, pos, i, K=8):
    """For object i: ensemble destination, target-relative end and K imagined paths per member (anchored samples), each re-anchored
    to the world-model end -> (paths_top (M,32,3) table frame, end (3,), dest index or -1, context for training)."""
    from cleantables.worldmodel.base_model import PS
    I = B.I
    b = B._batch(torch.tensor(pos[None], device=B.dev), torch.zeros(1, I, device=B.dev), None)
    mode = torch.ones(1, dtype=torch.long, device=B.dev)
    mv = torch.tensor([i], device=B.dev)
    Es, dp = [], []
    for m in B.models:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]
            _, tl = m.pointers(E, mv)
        Es.append(E); tlf = tl.float(); dp.append(F.softmax(torch.cat([tlf[:, :I], tlf[:, 24:25]], -1), -1))
    d = int(torch.stack(dp).mean(0).argmax(-1)); tg = torch.tensor([d if d < I else 24], device=B.dev)
    ends, ps = [], []
    for m, E in zip(B.models, Es):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c, _, mdn = m.heads_from(E, mv, tg); z = m.end_disp(c)
            ps.append(m.sample_paths_anchored(E, c, K=K).float()[0, :, :, :3])
        tm = int(mdn["logit"].float().argmax(-1))
        ends.append(b["obj_geo"][0, d, :3] + 0.1 * mdn["mu"].float()[0, tm] if d < I else
                    b["obj_geo"][0, i, :3] + z["mu"].float()[0, int(z["logit"].float().argmax(-1))] * PS)
    end = torch.stack(ends).mean(0)
    org = b["obj_geo"][0, i, :3]
    P = torch.cat(ps, 0) + org                                        # (M,32,3) top centre
    lin = torch.linspace(0, 1, P.shape[1], device=B.dev)[None, :, None]
    P = P + lin * (end - P[:, -1:])                                   # re-anchored to the world-model end, as the RL teacher uses it
    return P, end, (d if d < I else -1), b, tg


def score_run(run, n_layouts=6, K=8, seed=0, kind="heldout"):
    """Penetration of RUN's imagined carries (held-out fold per episode) on measured and random layouts."""
    os.environ["WM_RUN"] = run
    import importlib
    from cleantables.worldmodel import base_bridge as W
    importlib.reload(W)
    from cleantables.worldmodel.tasks3_train import NF
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(t["id"] for t in T if t["video"] == "IMG_2516")
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    rng = np.random.default_rng(seed); res = {}
    for t in T:
        if t["task"] not in ("tools", "fruit"):
            continue
        ep = t["id"]; f = fold_of[ep] if ep.startswith("IMG_2516") else fold_of[t["video"]]
        B = W.WMBridge(ep, ckpts=[W.RUN / f"fold{f}_seed{s}.pt" for s in range(3)])
        sc = Scorer(ep)
        Z = B.Z; ids = [str(x) for x in Z["ids"]]; H = torch.as_tensor(Z["extents"][:, 2], device=B.dev)
        task = [i for i, r in enumerate(Z["role"]) if str(r) == "task"]
        for L in range(n_layouts):
            P0 = Z["start_pos"].copy() if L == 0 else W.shuffle_layout(Z, B.S, rng, task)
            if P0 is None:
                continue
            for i in task:
                paths, end, d, _, _ = imagine(B, P0, i, K)
                G = sc.grid(ids, P0, i)
                pb = paths.clone(); pb[..., 2] -= H[i]
                pen = sc.penetration(G, ids[i], pb)                     # (M,)
                med = paths.median(0).values; pbm = med.clone()[None]; pbm[..., 2] -= H[i]
                penm = float(sc.penetration(G, ids[i], pbm)[0])
                r = res.setdefault(t["task"], dict(n=0, coll_med=0, coll_any=0, frac_samples=[]))
                r["n"] += 1; r["coll_med"] += int(penm > 0.5); r["frac_samples"].append(float((pen > 0.5).float().mean()))
        print(ep, "done", flush=True)
    for task, r in res.items():
        print(f"== {run} {task:5s}: {r['n']} carries; the carry the robot follows (median path) passes THROUGH an object {r['coll_med'] / r['n']:.2f};"
              f" share of individual imagined samples that collide {np.mean(r['frac_samples']):.2f}")
    json.dump({k: dict(n=v["n"], coll_med=v["coll_med"], frac_samples=float(np.mean(v["frac_samples"]))) for k, v in res.items()},
              open(ROOT / "runs/uwm" / run / "phys_score.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ fine-tune (reward-weighted flow matching)
def finetune(run_in, run_out, seeds=(0, 1, 2), steps=600, B_sc=12, K=16, keep=0.25, lr=1e-4, w_phys=1.0, seed=0, member=None):
    """Fine-tune the final world-model members (fold-1_seed*) of run_in with physics feedback on the training twins (held-out twins
    are not used). Per step: B_sc random (twin, random layout, in-scope object); the member imagines K carries (its own sampler), the voxel
    scorer scores them, and the flow head is trained (flow-matching) towards the `keep` fraction with the least penetration; the ordinary
    imitation losses on the human data (tasks3 dataset, same cfg as training) run every step as well."""
    from cleantables.worldmodel.base_model import UWM, NP, PS
    from cleantables.worldmodel.base_train import compute_loss, get_batch, augment, prep_store, to_dev, DEF
    from cleantables.worldmodel import base_bridge as W
    cfg = {**DEF, "prog_loss": "bce", "w_prog_abs": 3.0, "path_resid": True, "mover_set": True}
    dev = "cuda"
    eps = episodes("train")
    if os.environ.get("PF_ONLY") == "tools":                       # physics rounds focused on the tools twins (imitation still on all data)
        eps = [e for e in eps if not e.startswith("IMG_2516")]
    S_ = prep_store([os.environ.get("TASKS3_DATA", "tasks3c")]); G = to_dev(S_, dev)
    vids = {v: i for i, v in enumerate(sorted(set(S_.d["group"])))}
    G["video_id"] = torch.tensor([vids[v] for v in S_.d["group"]], device=dev)
    out = ROOT / "runs/uwm" / run_out; out.mkdir(parents=True, exist_ok=True)
    for sd in ([member] if member is not None else seeds):
        rng = np.random.default_rng(seed + 17 * sd); gen = torch.Generator(device=dev); gen.manual_seed(seed + sd)
        m = UWM(path_resid=True).to(dev); m.load_state_dict(torch.load(ROOT / "runs/uwm" / run_in / f"fold-1_seed{sd}.pt", map_location=dev))
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        bridges, scorers = {}, {}
        log = open(out / f"finetune_seed{sd}.log", "w")
        t0 = time.time()
        for it in range(steps):
            m.eval()
            items = []
            for _ in range(B_sc):                                   # ---- collect: imagine K carries per scene, score with geometry
                ep = eps[rng.integers(len(eps))]
                if ep not in bridges:
                    bridges[ep] = W.WMBridge(ep, ckpts=[ROOT / "runs/uwm" / run_in / f"fold-1_seed{sd}.pt"]); scorers[ep] = Scorer(ep)
                Bb = bridges[ep]; Bb.models = [m]
                Z = Bb.Z; ids = [str(x) for x in Z["ids"]]
                task = [i for i, r in enumerate(Z["role"]) if str(r) == "task"]
                P0 = Z["start_pos"].copy() if rng.random() < 0.2 else W.shuffle_layout(Z, Bb.S, rng, task)
                if P0 is None:
                    continue
                i = int(task[rng.integers(len(task))])
                paths, end, d, b, tg = imagine(Bb, P0, i, K)
                Hh = float(Z["extents"][i, 2])
                pb = paths.clone(); pb[..., 2] -= Hh
                pen = scorers[ep].penetration(scorers[ep].grid(ids, P0, i), ids[i], pb)
                n_keep = max(1, int(keep * len(pen)))
                sel = pen.argsort()[:n_keep]
                items.append((b, i, tg, paths[sel], pen.mean().item(), pen[sel].mean().item()))
            if not items:
                continue
            m.train()                                               # ---- update: flow matching towards the low-penetration samples
            lphys = 0.0
            for b, i, tg, tgt, _, _ in items:
                n = tgt.shape[0]
                bb = {k: (v.expand(n, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in b.items()}
                mode = torch.ones(n, dtype=torch.long, device=dev)
                mv = torch.full((n,), i, dtype=torch.long, device=dev); tgn = tg.expand(n)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    E = m.encode(bb, mode); E["kind"] = bb["obj_kind"]
                    c, _, _ = m.heads_from(E, mv, tgn)
                org = bb["obj_geo"][:, i, :3]
                rel = (tgt - org[:, None]) / PS                            # (n,NP,3)
                u = torch.linspace(0, 1, NP, device=dev)[None, :, None]
                x1 = torch.cat([rel - u * rel[:, -1:], torch.zeros_like(rel)], -1)
                x0 = torch.randn(x1.shape, device=dev, generator=gen); t = torch.rand(n, device=dev, generator=gen)
                xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = m.velocity(E, c, xt, t)
                lphys = lphys + ((v.float() - (x1 - x0))[..., :3] ** 2).mean() / len(items)
            idx = torch.as_tensor(np.random.default_rng(it).integers(0, S_.n, cfg["bs"]), device=dev)
            bt = augment(get_batch(G, idx, dev), cfg, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Ls = compute_loss(m, bt, cfg, gen)
            loss = w_phys * lphys + sum(Ls.values())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 3.0); opt.step()
            if it % 25 == 0 or it == steps - 1:
                pm = np.mean([x[4] for x in items]); ps_ = np.mean([x[5] for x in items])
                msg = f"[seed{sd}] it {it} phys_flow {float(lphys):.3f} imitation {float(sum(Ls.values())):.3f} | mean penetration of its samples {pm:.1f} vox (kept {ps_:.1f}) | {time.time() - t0:.0f}s"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
        torch.save(m.state_dict(), out / f"fold-1_seed{sd}.pt")


def score_final(run, n_layouts=8, K=8, seed=1, kind="heldout"):
    """Penetration of the final members of RUN on the held-out twins (random and measured layouts)."""
    from cleantables.worldmodel import base_bridge as W
    rng = np.random.default_rng(seed); res = {}
    for ep in episodes(kind):
        B = W.WMBridge(ep, ckpts=[ROOT / "runs/uwm" / run / f"fold-1_seed{s}.pt" for s in range(3)]); sc = Scorer(ep)
        Z = B.Z; ids = [str(x) for x in Z["ids"]]; H = torch.as_tensor(Z["extents"][:, 2], device=B.dev)
        task = [i for i, r in enumerate(Z["role"]) if str(r) == "task"]
        cls = "fruit" if ep.startswith("IMG_2516") else "tools"
        for L in range(n_layouts):
            P0 = Z["start_pos"].copy() if L == 0 else W.shuffle_layout(Z, B.S, rng, task)
            if P0 is None:
                continue
            for i in task:
                paths, end, d, _, _ = imagine(B, P0, i, K)
                G = sc.grid(ids, P0, i)
                med = paths.median(0).values[None].clone(); med[..., 2] -= H[i]
                pm = float(sc.penetration(G, ids[i], med)[0])
                pb = paths.clone(); pb[..., 2] -= H[i]
                ps = sc.penetration(G, ids[i], pb)
                lift = float(paths.median(0).values[:, 2].max() - max(paths[:, 0, 2].median(), paths[:, -1, 2].median()))
                r = res.setdefault(cls, dict(n=0, coll=0, pen=[], samp=[], lift=[]))
                r["n"] += 1; r["coll"] += int(pm > 0.5); r["pen"].append(pm); r["samp"].append(float((ps > 0.5).float().mean())); r["lift"].append(lift)
    for c, r in res.items():
        print(f"== {run} FINAL on HELD-OUT twins {c:5s}: {r['n']} carries; followed (median) path collides {r['coll'] / r['n']:.2f}; mean penetration "
              f"{np.mean(r['pen']):.1f} voxels/step; samples colliding {np.mean(r['samp']):.2f}; lift median {100 * np.median(r['lift']):.1f} cm", flush=True)
    json.dump({c: dict(n=r["n"], coll=r["coll"], pen=float(np.mean(r["pen"])), samp=float(np.mean(r["samp"])), lift=float(np.median(r["lift"])))
               for c, r in res.items()}, open(ROOT / "runs/uwm" / run / "phys_score_final.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "occ":
        for ep in episodes("all"):
            t0 = time.time(); n = build_occ(ep); print(ep, n, f"{time.time() - t0:.1f}s", flush=True)
    elif sys.argv[1] == "score":
        score_run(sys.argv[2])
    elif sys.argv[1] == "score_final":
        score_final(sys.argv[2])
    elif sys.argv[1] == "finetune":
        finetune(sys.argv[2], sys.argv[3], steps=int(os.environ.get("PF_STEPS", 600)), member=int(sys.argv[4]) if len(sys.argv) > 4 else None)
