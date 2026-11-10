"""Dense object point tracks for the table videos: CoTracker3 (offline model, frozen) seeded inside the SAM 3 object masks.

Per video, 10 fps work frames (960x540) are cut into windows of WIN frames (overlap OV). In each window every SAM track that is seen there
gets up to NQ query points sampled inside its (eroded) mask at up to 2 of its sightings; CoTracker3 tracks them forwards and backwards
through the whole window. Output per window: point tracks (T, P, 2) in work px, visibility (T, P), owner track id per point.

  $VLA_PY -m cleantables.perception.flow.cotrack IMG_2513 [t0 t1]   -> data/twin_table/flow/<video>_tracks.npz
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cleantables.perception.scene.common import meta, read_frame
from cleantables.perception.flow import cached_tracks as build_tracks

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/twin_table/flow"
WIN, OV, NQ = 80, 16, 24
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))


def seeds(d, tracks, k0, k1, rng):
    """Query points (t_local, x, y) in work px and their owner track ids, for frames k0..k1-1."""
    q, own = [], []
    ker = np.ones((5, 5), np.uint8)
    for ti, t in enumerate(tracks):
        obs = [(k, i) for k, i in t["obs"] if k0 <= k < k1]
        if not obs:
            continue
        pick = [obs[0]] if len(obs) == 1 else [obs[0], obs[len(obs) // 2]]
        for k, i in pick:
            m = cv2.erode(d["masks"][i].astype(np.uint8), ker).astype(bool)
            if m.sum() < 8:
                m = d["masks"][i]
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            sel = rng.choice(len(xs), size=min(NQ // len(pick), len(xs)), replace=False)
            for s in sel:
                q.append((k - k0, xs[s] * 2 + 1.0, ys[s] * 2 + 1.0)); own.append(ti)
    return np.array(q, np.float32).reshape(-1, 3), np.array(own, np.int32)


def run(video, t0=None, t1=None, dev="cuda"):
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    d, tracks, table, kept = build_tracks(video)
    n = meta(video)["n"]
    k_lo = 0 if t0 is None else int(t0 * 10); k_hi = n if t1 is None else min(n, int(t1 * 10))
    rng = np.random.default_rng(0)
    W_out = []
    tic = time.time()
    for k0 in range(k_lo, k_hi - OV, WIN - OV):
        k1 = min(k0 + WIN, k_hi)
        q, own = seeds(d, tracks, k0, k1, rng)
        if len(q) == 0:
            continue
        fr = np.stack([read_frame(video, k)[..., ::-1] for k in range(k0, k1)])
        vid = torch.from_numpy(fr).permute(0, 3, 1, 2)[None].float().to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            tr, vis = model(vid, queries=torch.from_numpy(q)[None].to(dev))
        W_out.append(dict(k0=k0, k1=k1, tr=tr[0].float().cpu().numpy().astype(np.float16), vis=vis[0].cpu().numpy() > 0.5 if vis.dtype == torch.bool else vis[0].float().cpu().numpy() > 0.5,
                          own=own, q=q))
        print(video, f"{k0 / 10:.0f}-{k1 / 10:.0f}s", len(q), "pts", f"{time.time() - tic:.0f}s", f"{torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        del vid
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if t0 is None else f"_{int(t0)}_{int(t1)}"
    np.savez_compressed(OUT / f"{video}{tag}_tracks.npz",
                        k0=np.array([w["k0"] for w in W_out]), k1=np.array([w["k1"] for w in W_out]),
                        tr=np.array([w["tr"] for w in W_out], dtype=object), vis=np.array([w["vis"] for w in W_out], dtype=object),
                        own=np.array([w["own"] for w in W_out], dtype=object), q=np.array([w["q"] for w in W_out], dtype=object),
                        track_cat=np.array([t["category"] for t in tracks]), track_prompt=np.array([t["prompt"] for t in tracks]))
    return W_out, tracks


def overlay(video, W_out, tracks, path):
    """Check video: every point with a short trail, coloured by its object."""
    fdir = Path(str(path) + "_f"); fdir.mkdir(parents=True, exist_ok=True)
    cols = {ti: tuple(int(c) for c in cv2.applyColorMap(np.uint8([[(ti * 47) % 255]]), cv2.COLORMAP_HSV)[0, 0]) for ti in range(len(tracks))}
    nf = 0
    for wi, w in enumerate(W_out):
        a = 0 if wi == 0 else OV // 2
        b = (w["k1"] - w["k0"]) if wi == len(W_out) - 1 else (w["k1"] - w["k0"]) - OV // 2
        for t in range(a, b):
            img = read_frame(video, w["k0"] + t)
            for p in range(w["tr"].shape[1]):
                if not w["vis"][t, p]:
                    continue
                c = cols[int(w["own"][p])]
                for s in range(max(0, t - 8), t):
                    if w["vis"][s, p] and w["vis"][s + 1, p]:
                        cv2.line(img, tuple(w["tr"][s, p].astype(int)), tuple(w["tr"][s + 1, p].astype(int)), c, 1, cv2.LINE_AA)
                cv2.circle(img, tuple(w["tr"][t, p].astype(int)), 2, c, -1, cv2.LINE_AA)
            cv2.putText(img, f"{video}  t={(w['k0'] + t) / 10:.1f}s  CoTracker3 points seeded in SAM 3 masks", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(fdir / f"{nf:05d}.jpg"), img); nf += 1
    os.system(f"ffmpeg -v error -y -framerate 10 -i {fdir}/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 24 {path}")
    for f in fdir.glob("*.jpg"):
        f.unlink()
    fdir.rmdir()


if __name__ == "__main__":
    v = sys.argv[1]
    t0, t1 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (None, None)
    W_out, tracks = run(v, t0, t1)
    if t0 is not None:
        overlay(v, W_out, tracks, ROOT / f"videos/twin_table/flow/{v}_{int(t0)}_{int(t1)}_points.mp4")
