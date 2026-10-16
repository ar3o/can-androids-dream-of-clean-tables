"""Monocular depth with MoGe-3 (Wang et al. 2025) on the work frames -> depth10.npy (+ depth10.json with K and the FOV).

    PYTHONPATH=<MoGe-3 paths>:<repo> python -m cleantables.perception.depth VIDEO [VIDEO ...]

One model load for all videos and one field of view per video (median of per-frame estimates on a few frames),
so all frames share K. Depth is computed on every `stride`-th frame (the frames SAM 3 tracks); the others copy it.
Usually started by run.py, which sets the MoGe-3 PYTHONPATH from the config.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from cleantables import paths

from .config import DepthConfig, load_config
from .files import H_LO, W_LO, n_frames, read_frame, write_json

PREFETCH = 4


def cached_weights(repo_id: str) -> Path:
    """model.pt of a Hugging Face repo in the local cache (the MoGe-3 environment ships no hub client)."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    found = sorted(hub.glob(f"models--{repo_id.replace('/', '--')}/snapshots/*/model.pt"))
    if not found:
        raise FileNotFoundError(f"{repo_id}/model.pt is not in {hub}")
    return found[0]


class MoGe:
    def __init__(self, repo_id: str, device: str = "cuda"):
        import torch
        from moge.model.v3 import MoGeModel

        self.torch = torch
        self.device = device
        self.model = MoGeModel.from_pretrained(str(cached_weights(repo_id))).to(device).eval()

    def infer(self, bgr: np.ndarray, cfg: DepthConfig, fov_x=None) -> dict:
        image = (
            self.torch.tensor(bgr[..., ::-1].copy(), dtype=self.torch.float32, device=self.device).permute(2, 0, 1)
            / 255.0
        )
        with self.torch.no_grad():
            return self.model.infer(
                image, resolution_level=cfg.resolution_level, fov_x=fov_x, use_fp16=True, refine_steps=cfg.refine_steps
            )


def video_fov(model: MoGe, video: str, ks: list[int], cfg: DepthConfig) -> float:
    picks = [ks[i] for i in np.linspace(0, len(ks) - 1, min(cfg.fov_frames, len(ks))).astype(int)]
    fovs = []
    for k in picks:
        fx = model.infer(read_frame(video, k), cfg)["intrinsics"].float().cpu().numpy()[0, 0]
        fovs.append(float(np.degrees(2 * np.arctan(0.5 / fx))))
    return float(np.median(fovs))


def run_video(model: MoGe, video: str, cfg: DepthConfig, pool: ThreadPoolExecutor) -> None:
    out = paths.perception_dir(video) / "depth10.npy"
    if out.exists():
        return
    n = n_frames(video)
    ks = list(range(0, n, cfg.stride))
    t0 = time.time()
    fov = video_fov(model, video, ks, cfg)
    part = Path(str(out) + ".part.npy")
    D = np.lib.format.open_memmap(str(part), mode="w+", dtype=np.float16, shape=(n, H_LO, W_LO))
    pending = [pool.submit(read_frame, video, k) for k in ks[:PREFETCH]]
    for i, k in enumerate(ks):
        bgr = pending[i].result()
        if i + PREFETCH < len(ks):
            pending.append(pool.submit(read_frame, video, ks[i + PREFETCH]))
        o = model.infer(bgr, cfg, fov_x=fov)
        d = o["depth"].float().cpu().numpy()
        valid = o["mask"].cpu().numpy().astype(bool) & np.isfinite(d)
        d_lo = cv2.resize(np.where(valid, d, 0).astype(np.float32), (W_LO, H_LO), interpolation=cv2.INTER_NEAREST)
        D[k : min(n, k + cfg.stride)] = d_lo.astype(np.float16)
        intrinsics = o["intrinsics"].float().cpu().numpy()
    h, w = bgr.shape[:2]
    K = np.array(
        [[intrinsics[0, 0] * w, 0, intrinsics[0, 2] * w], [0, intrinsics[1, 1] * h, intrinsics[1, 2] * h], [0, 0, 1.0]]
    )
    D.flush()
    del D
    part.rename(out)
    wall = time.time() - t0
    write_json(
        dict(
            video=video,
            n=n,
            computed_every=cfg.stride,
            refine_steps=cfg.refine_steps,
            n_computed=len(ks),
            fov_x=fov,
            K=K.tolist(),
            model=cfg.model,
            wall_s=round(wall, 1),
        ),
        paths.perception_dir(video) / "depth10.json",
    )
    print(f"{video}: depth on {len(ks)}/{n} frames in {wall:.0f} s", flush=True)


def intrinsics(video: str) -> np.ndarray:
    return np.array(json.loads((paths.perception_dir(video) / "depth10.json").read_text())["K"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default=None, help="perception config (default configs/perception.yaml)")
    args = ap.parse_args()
    cfg = (load_config(args.config) if args.config else load_config()).depth
    model = MoGe(cfg.model)
    with ThreadPoolExecutor(PREFETCH) as pool:
        for video in args.videos:
            run_video(model, video, cfg, pool)


if __name__ == "__main__":
    main()
