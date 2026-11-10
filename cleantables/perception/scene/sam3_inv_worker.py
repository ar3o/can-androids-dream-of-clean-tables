"""SAM 3 text-prompted inventory on the table_clean work frames (every EVERY-th 10 fps frame -> 2 fps by default).

Reuses cleantables.perception.recon.sam3_worker.load() (official facebook/sam3 checkpoint, fp32 weights + bf16 autocast, text encoder on the CPU).
Run in the SAM 3 env:
  source scripts/env.sh
  PYTHONPATH=$SAM3_REPO:. HF_HUB_OFFLINE=1 \
    $SAM3_PY -m cleantables.perception.scene.sam3_inv_worker VIDEO [--every 5] [--only k0,k1]
Out data/table_clean/_work/<video>/sam3_inv[_<tag>].npz, flat list of detections (p >= MIN_P):
  frame (m,) work-frame index, prompt (m,) index into `prompts`, prob (m,), box (m,4) xyxy work px (960x540),
  masks packed bool (m, 270, 480) (half resolution), shape, prompts, frames_done.
Checkpointed: partial results are flushed every 50 frames to <out>.part.npz and resumed.
"""
from __future__ import annotations

import os
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cleantables.perception.recon.sam3_worker import load
from cleantables.perception.scene.common import WORK, meta, read_frame
from cleantables.perception.scene.vocab import PROMPTS

MIN_P = 0.3
MH, MW = 270, 480


def load_any(dev):
    """recon.sam3_worker's SAM 3 loader; on the CPU the position-encoding caches that SAM 3 precomputes with device="cuda" are made on the CPU."""
    if dev == "cuda":
        return load(dev)
    torch.Tensor.pin_memory = lambda self, *a, **k: self       # no accelerator: pinning is meaningless and raises
    origs = {n: getattr(torch, n) for n in ("zeros", "arange", "ones", "tensor", "empty")}

    def wrap(fn):
        def f(*a, **k):
            if str(k.get("device", "")).startswith("cuda"):
                k["device"] = "cpu"
            return fn(*a, **k)
        return f
    for n, fn in origs.items():
        setattr(torch, n, wrap(fn))
    try:
        return load(dev)
    finally:
        for n, fn in origs.items():
            setattr(torch, n, fn)


def run(video, frames, out, phrases, dev="cuda"):
    proc = load_any(dev)
    proc.confidence_threshold = MIN_P
    part = Path(str(out) + ".part.npz")
    rec = {k: [] for k in ("frame", "prompt", "prob", "box", "mask")}
    done = set()
    if part.exists():
        z = np.load(part)
        m = len(z["frame"])
        rec["frame"] = list(z["frame"]); rec["prompt"] = list(z["prompt"]); rec["prob"] = list(z["prob"]); rec["box"] = list(z["box"])
        rec["mask"] = list(np.unpackbits(z["masks"])[:m * MH * MW].reshape(m, MH, MW).astype(bool))
        done = set(z["frames_done"].tolist())
    t0 = time.time()
    todo = [k for k in frames if k not in done]
    print(f"{video}: {len(todo)} frames to do ({len(done)} done), {len(phrases)} prompts, dev {dev}",
          flush=True)

    def flush(path):
        m = len(rec["frame"])
        np.savez_compressed(str(path) + ".tmp.npz", frame=np.asarray(rec["frame"], np.int32), prompt=np.asarray(rec["prompt"], np.int16),
                            prob=np.asarray(rec["prob"], np.float32), box=np.asarray(rec["box"], np.float32).reshape(m, 4),
                            masks=np.packbits(np.asarray(rec["mask"], bool).reshape(m, MH, MW)) if m else np.zeros(0, np.uint8),
                            shape=np.array([m, MH, MW]), prompts=np.array(phrases), frames_done=np.array(sorted(done), np.int32))
        Path(str(path) + ".tmp.npz").replace(path)

    for i, k in enumerate(todo):
        bgr = read_frame(video, k)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            st = proc.set_image(torch.from_numpy(rgb).permute(2, 0, 1))
            for p, ph in enumerate(phrases):
                proc.reset_all_prompts(st)
                st = proc.set_text_prompt(ph, st)
                pr = st["scores"].float().cpu().numpy()
                if len(pr) == 0:
                    continue
                ms = torch.nn.functional.interpolate(st["masks"].float(), (MH, MW), mode="area")[:, 0] > 0.5
                ms = ms.cpu().numpy()
                bx = st["boxes"].float().cpu().numpy()
                for j in np.argsort(-pr)[:12]:
                    rec["frame"].append(k); rec["prompt"].append(p); rec["prob"].append(pr[j]); rec["box"].append(bx[j]); rec["mask"].append(ms[j])
        del st
        done.add(k)
        if (i + 1) % 50 == 0:
            flush(part)
            el = time.time() - t0
            print(f"{video}: {i + 1}/{len(todo)} {el:.0f}s ({el / (i + 1):.2f} s/frame) ",
                  flush=True)
    flush(out)
    if part.exists():
        part.unlink()
    print(f"{video}: done {len(todo)} frames {time.time() - t0:.0f}s, {len(rec['frame'])} detections", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import os
    if a.device == "cpu":
        torch.set_num_threads(int(os.environ.get("SAM3_THREADS", "6")))
    n = meta(a.video)["n"]
    frames = [int(x) for x in a.only.split(",")] if a.only else list(range(a.offset, n, a.every))
    out = WORK / a.video / (f"sam3_inv_{a.tag}.npz" if a.tag else "sam3_inv.npz")
    if out.exists():
        print(f"{out} exists, skip")
        return
    run(a.video, frames, out, [p for p, _ in PROMPTS], a.device)


if __name__ == "__main__":
    main()
