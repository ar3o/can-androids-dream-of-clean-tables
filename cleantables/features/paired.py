"""Features of the paired renders, in the format of the `grids` stage of cleantables.features.video.

For every row (video, k) of the world model's rows.json whose episode has paired renders: a V-JEPA 2 ViT-L clip (layer 24, last
tubelet, 448 x 256) and DINOv2-L of render frame k, plain and horizontally flipped, as memmaps aligned with the rows of vj_grid.npy /
dn_grid.npy. One set per render variant.
The clip is held the way the robot's run-time clip is (a new still every 1 s, the latest one held): slot j frames before now
(j = 0..31) shows frame k - 10*ceil(j/10), clamped at the episode's first render (k0 - 31), i.e. k, then k-10 x10, k-20 x10,
k-30 x10, k-40 x1 -> vj_hold_v<n>.npy. DINOv2 of frame k -> dn_v<n>.npy.
The `gap` stage then measures the real-vs-render distance per row against the stored real grids (DINOv2 is the clean same-frame
comparison; the held V-JEPA clip also differs from the real 10 fps clip in its motion).

  $VLA_PY -m cleantables.features.paired encode [--variants 0 1 2 3]   (GPU, resumable, <= 3.5 GB)
  $VLA_PY -m cleantables.features.paired gap                           (CPU)  -> paired_feats/gap.csv, gap_summary.json
  $VLA_PY -m cleantables.features.paired uv [--variants ...]           (CPU)  -> paired_feats/uv_v<n>.npy (rows, J tracks, 2),
        uv_ok_v<n>.npy: each tracked object's uv in that render (v >= 4 move the camera and rescale objects, so the real frame's
        uv no longer fits); index j = the track index of masks.npz (scene.json objects[].track); convex hull of the twin mesh, as at
        robot run time
Out: $CLEANTABLES_DATA/vjepa_wm/paired_feats/{vj,dn}_v<n>.npy (rows, 2 flips, h, w, 1024) fp16 + .done.npy
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WM = Path(CT_DATA + "/wm_eval/wm_v2/feats")
PR = Path(CT_DATA + "/vjepa_wm/paired_renders")
OUT = Path(CT_DATA + "/vjepa_wm/paired_feats")
HW = {"vj": (16, 28), "dn": (18, 32)}
T = 32
TEST = ("IMG_2516_t11", "IMG_2516_t12", "IMG_2516_t13", "IMG_2516_t14", "IMG_2546", "IMG_2547", "IMG_2548")


def rows():
    return json.load(open(WM / "rows.json"))["rows"]


@lru_cache(maxsize=None)
def ep_ranges():
    """Video -> [(ep, k_lo, k_hi)] of the rendered episodes (k range of their renders)."""
    out = {}
    for d in sorted(PR.iterdir()):
        m = d / "v0" / "meta.json"
        if m.exists():
            meta = json.load(open(m))
            out.setdefault(meta["video"], []).append((d.name, min(meta["k"]), max(meta["k"])))
    return out


def row_ep(v, k):
    """The rendered episode whose render range covers frame k (else None). Near the episode start the held clip repeats the first
    render (held_frames clamps at lo, as the robot at the start of a run); k one past the last render uses the last render (render_k)."""
    for ep, lo, hi in ep_ranges().get(v, []):
        if lo <= k <= hi + 1:
            return ep
    return None


def render_k(ep, k):
    """The rendered frame shown for row frame k (= k, or the episode's last render when k is one past it: 0.1 s earlier)."""
    return min(k, {e: hi for vid in ep_ranges() for e, _, hi in ep_ranges()[vid]}[ep])


def held_frames(k, lo):
    """The 32 frame indices of the robot-style held clip ending at k (oldest first)."""
    return [max(lo, k - 10 * int(np.ceil(j / 10))) for j in range(T - 1, -1, -1)]


def loader(variant):
    cache = {}

    def load(ep, k):
        if (ep, k) not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[(ep, k)] = cv2.cvtColor(cv2.imread(str(PR / ep / f"v{variant}" / f"{k:05d}.jpg")), cv2.COLOR_BGR2RGB)
        return cache[(ep, k)]
    return load


def encode(variants, dev="cuda"):
    import torch
    from cleantables.features.encoders import Encoder
    R = rows()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(3.5 / 23.5)
    encs = {"vj": Encoder("vj2l", layers=(24,), dev=dev), "dn": Encoder("dino2l", layers=(24,), dev=dev)}
    for v in variants:
        if not all((PR / ep / f"v{v}" / "meta.json").exists() for vid in ep_ranges() for ep, _, _ in ep_ranges()[vid]):
            pass                                                          # partial render sets are allowed: rows without renders are filled on a later run
        if not any((PR / ep / f"v{v}" / "meta.json").exists() for vid in ep_ranges() for ep, _, _ in ep_ranges()[vid]):
            print(f"v{v}: no renders, skipped", flush=True)
            continue
        load = loader(v)
        todo_all = [(i, row_ep(*r)) for i, r in enumerate(R)]
        lo_of = {ep: lo for vid in ep_ranges() for ep, lo, _ in ep_ranges()[vid]}
        for name in ("vj", "dn"):
            fn = "vj_hold" if name == "vj" else "dn"
            f, df = OUT / f"{fn}_v{v}.npy", OUT / f"{fn}_v{v}.done.npy"
            G = np.lib.format.open_memmap(f, mode="r+" if f.exists() else "w+", dtype=np.float16, shape=(len(R), 2) + HW[name] + (1024,))
            done = np.load(df) if df.exists() else np.zeros(len(R), bool)
            todo = [(i, ep) for i, ep in todo_all if ep and not done[i] and (PR / ep / f"v{v}" / "meta.json").exists()]
            t0 = time.time()
            for n, (i, ep) in enumerate(todo):
                k = render_k(ep, R[i][1])
                X = np.stack([load(ep, kk) for kk in held_frames(k, lo_of[ep])])[None] if name == "vj" else load(ep, k)[None, None]
                for fl in (0, 1):
                    Xi = X if fl == 0 else np.ascontiguousarray(X[:, :, :, ::-1])
                    G[i, fl] = encs[name](Xi)[24][0][0].float().cpu().numpy().astype(np.float16)
                done[i] = True
                if n % 200 == 0:
                    G.flush(); np.save(df, done)
                    print(f"v{v} {name}: {n + 1}/{len(todo)} {(time.time() - t0) / (n + 1):.2f} s/row", flush=True)
            G.flush(); np.save(df, done)
            print(f"v{v} {name}: done {int(done.sum())}/{len(R)} rows", flush=True)


def uv(variants):
    from cleantables.features.randomize import hull_uv
    R = rows()
    J = int(np.load(WM / "masks.npz")["cov_vj"].shape[1])
    SF = ROOT / "data/twin_table/ep/_scalefix"
    eps = {}
    for v in variants:
        U, OK = np.zeros((len(R), J, 2), np.float32), np.zeros((len(R), J), bool)
        n = 0
        for i, (vid, k) in enumerate(R):
            ep = row_ep(vid, k)
            m = PR / str(ep) / f"v{v}" / "meta.json"
            if ep is None or not m.exists():
                continue
            k = render_k(ep, k)
            if ep not in eps:
                S = json.load(open(SF / ep / "scene.json"))
                Z = np.load(json.load(open(PR / ep / "v0" / "meta.json"))["pose_npz"], allow_pickle=True)
                verts = [np.array([l.split()[1:4] for l in open(ROOT / o["usd"].replace(".usda", ".obj")) if l.startswith("v ")], np.float64)
                         for o in S["objects"]]
                eps[ep] = (S, {int(kk): j for j, kk in enumerate(Z["k"])}, [str(x) for x in Z["ids"]], Z["pose"].astype(np.float64), verts)
            S, kidx, ids, P, verts = eps[ep]
            dr = json.load(open(m))["dr"]
            cam = S["cameras"]["main"]
            K, T = (dr["cam_K"], dr["cam_T"]) if dr.get("full") else (cam["K"], cam["T_cam_to_table"])
            poses = []
            for o in S["objects"]:                                        # as paired_render.state(): untracked / NaN -> start pose
                p = np.r_[o["pos"], 0.0]
                if o["id"] in ids and int(k) in kidx:
                    q = P[kidx[int(k)], ids.index(o["id"])]
                    if np.isfinite(q).all():
                        p = q
                poses.append(p)
            sz = [dr["obj_size"][o["id"]] for o in S["objects"]] if dr.get("full") else None
            u, ok = hull_uv(verts, poses, K, T, sz)
            for o, uu, oo in zip(S["objects"], u, ok):
                j = o.get("track")
                if j is not None and 0 <= j < J:
                    U[i, j], OK[i, j] = uu, oo
            n += 1
        if n == 0:
            print(f"uv v{v}: no renders, skipped", flush=True)
            continue
        for nm, X in ((f"uv_v{v}", U), (f"uv_ok_v{v}", OK)):            # atomic replace: a loader may read these files while this runs
            np.save(OUT / f"{nm}.tmp.npy", X); (OUT / f"{nm}.tmp.npy").replace(OUT / f"{nm}.npy")
        print(f"uv v{v}: {n} rows, {int(OK.sum())} object uvs", flush=True)


def gap():
    import pandas as pd
    R = rows()
    J = json.load(open(WM / "rows.json"))
    VJ = np.load(WM / "vj_grid.npy", mmap_mode="r"); DN = np.load(WM / "dn_grid.npy", mmap_mode="r")

    def cosd(a, b):
        a = a.astype(np.float32).reshape(-1, a.shape[-1]); b = b.astype(np.float32).reshape(-1, b.shape[-1])
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-6; b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-6
        return float(1 - (a * b).sum(1).mean())

    def globd(a, b):
        a = a.astype(np.float32).reshape(-1, a.shape[-1]).mean(0); b = b.astype(np.float32).reshape(-1, b.shape[-1]).mean(0)
        return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    vs = sorted(int(p.stem.split("_v")[1]) for p in OUT.glob("dn_v*.npy") if not p.stem.endswith("done") and np.load(OUT / f"{p.stem}.done.npy").any())
    fn = {"vj": "vj_hold", "dn": "dn"}
    G = {(nm, v): np.load(OUT / f"{fn[nm]}_v{v}.npy", mmap_mode="r") for nm in ("vj", "dn") for v in vs}
    D = {(nm, v): np.load(OUT / f"{fn[nm]}_v{v}.done.npy") for nm in ("vj", "dn") for v in vs}
    ep_goal = {e: r for e, r in J["goal_row"].items()}; ep_start = {e: r for e, r in J["start_row"].items()}
    out = []
    for i, (vid, k) in enumerate(R):
        ep = row_ep(vid, k)
        if ep is None or ep not in ep_goal:
            continue
        g, s = ep_goal[ep], ep_start[ep]
        row = dict(row=i, ep=ep, video=vid, k=k, split="test" if ep.startswith(TEST) else "train")
        for nm, REAL in (("vj", VJ), ("dn", DN)):
            row[f"{nm}_task"] = cosd(REAL[s, 0], REAL[g, 0])                # how far the task moves the real scene (start -> goal)
            for v in vs:
                if D[(nm, v)][i]:
                    row[f"{nm}_v{v}"] = cosd(REAL[i, 0], G[(nm, v)][i, 0])      # real vs render, same moment, patch cosine
                    row[f"{nm}_v{v}_glob"] = globd(REAL[i, 0], G[(nm, v)][i, 0])
            if all(D[(nm, v)][i] for v in vs) and len(vs) > 1:
                row[f"{nm}_var_spread"] = float(np.mean([cosd(G[(nm, a)][i, 0], G[(nm, b)][i, 0]) for a in vs for b in vs if a < b]))
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(OUT / "gap.csv", index=False)
    S = {}
    for nm in ("vj", "dn"):
        for v in vs:
            c = f"{nm}_v{v}"
            if c in df:
                q = df.dropna(subset=[c])
                S[c] = dict(rows=len(q), patch_dist=round(float(q[c].mean()), 4), ratio_to_task=round(float((q[c] / q[f"{nm}_task"]).median()), 3),
                            glob_dist=round(float(q[c + "_glob"].mean()), 4))
        if f"{nm}_var_spread" in df:
            S[f"{nm}_variant_spread"] = round(float(df[f"{nm}_var_spread"].mean()), 4)
    json.dump(S, open(OUT / "gap_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "gap", "uv"])
    ap.add_argument("--variants", type=int, nargs="*", default=[0, 1, 2, 3])
    a = ap.parse_args()
    encode(a.variants) if a.stage == "encode" else uv(a.variants) if a.stage == "uv" else gap()
