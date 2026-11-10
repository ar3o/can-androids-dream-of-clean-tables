"""Task-agnostic frozen V-JEPA 2 clip tokens for any video of the generic pipeline (facebook/vjepa2-vitl-fpc64-256, 16 frames,
squashed to 256x256, ImageNet normalisation, fp16).

Input: data/table_clean/_work/<V>/frames/<k>.jpg (10 fps) + frames_meta.json. Keyframes: k = 0,2,4,.. (5 per second).
Clip window: the encoder expects 16 frames at ~15 Hz (frames k-15..k at 15 fps, 1.0 s between first and last frame). At `fps` frames/s we take 16 sample positions linearly spaced over the last 1.0 s ending at k:
  idx_j = round(k - fps * (15-j)/15), j=0..15, clamped to >= 0.   (10 fps -> k-10..k with repeated frames; 15 fps -> exactly k-15..k)
Output data/generic/<V>/vjepa.npz: k (n,), tok (n,32,1024) fp16, glob (n,1024), tok_f, glob_f (horizontal flip). Skips if present.

  $VLA_PY -m cleantables.features.vjepa_tokens IMG_2533 IMG_2534 ...
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
WORK = ROOT / "data/table_clean/_work"
OUT = ROOT / "data/generic"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NF, RES, KSTEP = 16, 256, 2
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _prep(path):
    im = cv2.imread(str(path))[..., ::-1].astype(np.float32) / 255
    return torch.from_numpy((cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA) - MEAN) / STD).permute(2, 0, 1).half()


def window_idx(ks, fps):
    off = fps * (NF - 1 - np.arange(NF)) / (NF - 1)
    return np.clip(np.round(np.asarray(ks)[:, None] - off[None]).astype(int), 0, None)


def _pool(x):
    return F.avg_pool2d(x.permute(0, 3, 1, 2), 4).flatten(2).transpose(1, 2)


@torch.no_grad()
def encode_clips(model, X, idx, dev, flip=False, bs=8):
    """X (n,3,H,W) fp16 frames on cpu; idx (m,16) -> tok (m,32,1024), glob (m,1024) fp16."""
    toks, globs = [], []
    for i in range(0, len(idx), bs):
        v = X[torch.as_tensor(idx[i:i + bs])].to(dev)
        if flip:
            v = v.flip(-1)
        g = model(pixel_values_videos=v, skip_predictor=True).last_hidden_state.float().reshape(len(v), 8, 16, 16, -1)
        globs.append(g.mean((1, 2, 3)).cpu())
        toks.append(torch.cat([_pool(g.mean(1)), _pool(g[:, -1])], 1).cpu())
    return torch.cat(toks).numpy().astype(np.float16), torch.cat(globs).numpy().astype(np.float16)


def get_model(dev="cuda"):
    from transformers import AutoModel
    return AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(dev).eval()


def compute(model, frame_paths, fps, dev="cuda", ks=None):
    """frame_paths: list of image paths (all frames). Frames are decoded once (each ~0.4 MB fp16 at 256^2 -> whole video in RAM)."""
    X = torch.stack([_prep(p) for p in frame_paths])
    n = len(X)
    ks = np.arange(0, n, KSTEP) if ks is None else np.asarray(ks)
    idx = window_idx(ks, fps)
    tok, glob = encode_clips(model, X, idx, dev)
    tok_f, glob_f = encode_clips(model, X, idx, dev, flip=True)
    return dict(k=ks.astype(np.int32), tok=tok, glob=glob, tok_f=tok_f, glob_f=glob_f)


def run(videos, dev="cuda"):
    model = None
    times = {}
    for V in videos:
        f = OUT / V / "vjepa.npz"
        if f.exists():
            print(f"[vjepa] {V} exists, skip", flush=True)
            continue
        if model is None:
            model = get_model(dev)
        t0 = time.time()
        n = int(json.load(open(WORK / V / "frames_meta.json"))["n"])
        d = compute(model, [WORK / V / "frames" / f"{k}.jpg" for k in range(n)], 10.0, dev)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name("vjepa.tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(f)
        times[V] = time.time() - t0
        print(f"[vjepa] {V} {n} frames {len(d['k'])} keyframes {times[V]:.1f}s", flush=True)
    return times


def load(V):
    z = np.load(OUT / V / "vjepa.npz")
    return {k: z[k] for k in z.files}


def nearest_keyframe(V, k, d=None):
    """Index into the arrays of load(V) of the keyframe nearest to (at or before preferred on ties) frame k."""
    d = d or load(V)
    ks = d["k"]
    return int(np.argmin(np.abs(ks - k) + 1e-3 * (ks > k)))


if __name__ == "__main__":
    from cleantables.perception.common import wait_gpu
    wait_gpu(3000, "vjepa_tokens")
    torch.cuda.set_per_process_memory_fraction(min(0.99, 3.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    t0 = time.time()
    tt = run(sys.argv[1:])
    print(f"[vjepa] total {time.time() - t0:.0f}s, {len(tt)} videos, mean {np.mean(list(tt.values())) if tt else 0:.1f}s/video")
