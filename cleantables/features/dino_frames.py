"""Frozen DINOv2 ViT-S/14 patch features for every frame of every rollout episode (resumable disk cache).

    python -m cleantables.features.dino_frames --config configs/wm_features.yaml [--smoke] [--device cuda|cpu]

For frame i of `ep_<k>_<cam>.mp4`: resize to 224x224 (patch 14 -> 16x16 = 256 patches), ImageNet-normalise, run
`facebook/dinov2-small` (Apache-2.0, HF) and store the last-layer patch tokens (CLS dropped), fp16, as
    runs/wm_cache/<run_id>/ep_<k>_<cam>.npy        shape (T, 256, 384) float16
    runs/wm_cache/<run_id>/cache_meta.json          model id/revision, resolution, fit mode, per-file frame counts
Resumable: a file is skipped when it already has the right frame count. Batched; GPU or CPU.
Features are raw (un-normalised); the world model standardises per channel with train statistics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.rl.config_utils import load_yaml, merge_smoke, pick_device, resolve, std_parser, write_receipt
from cleantables.rl.rollout_io import list_episodes, load_episode, read_video

MODEL_ID = "facebook/dinov2-small"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RES = 224
GRID = RES // 14  # 16
DIM = 384


def to_square_tensor(frames: np.ndarray, fit: str = "crop") -> torch.Tensor:
    """uint8 (B,H,W,3) -> float (B,3,224,224), ImageNet-normalised.
    fit: crop (centre square crop, default) | pad (letterbox with edge-replicate) | squash (distorts aspect)."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    B, _, H, W = x.shape
    if H != W:
        if fit == "crop":
            s = min(H, W)
            y0, x0 = (H - s) // 2, (W - s) // 2
            x = x[:, :, y0:y0 + s, x0:x0 + s]
        elif fit == "pad":
            s = max(H, W)
            ph, pw = s - H, s - W
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="replicate")
        elif fit != "squash":
            raise ValueError(fit)
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class DinoExtractor:
    def __init__(self, device: str = "cpu", fp16: bool | None = None):
        from transformers import AutoModel

        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self.half = (self.device.type == "cuda") if fp16 is None else fp16
        if self.half:
            self.model.half()

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, batch_size: int = 32, fit: str = "crop") -> np.ndarray:
        """uint8 (T,H,W,3) -> float16 (T,256,384)."""
        outs = []
        for i in range(0, len(frames), batch_size):
            x = to_square_tensor(frames[i:i + batch_size], fit).to(self.device)
            if self.half:
                x = x.half()
            h = self.model(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
            assert h.shape[1] == GRID * GRID and h.shape[2] == DIM, h.shape
            outs.append(h.float().cpu().numpy().astype(np.float16))
        return np.concatenate(outs)


def cache_dir_for(run_dir: Path, cache_root: Path) -> Path:
    return cache_root / run_dir.name


def build_cache(run_dirs: list[Path], cache_root: Path, cams: list[str], device: str, batch_size: int = 32,
                fit: str = "crop", max_episodes: int | None = None, verbose: bool = True) -> dict:
    ext = None
    stats = {"computed": 0, "skipped": 0, "frames": 0}
    t0 = time.time()
    for rd in run_dirs:
        out = cache_dir_for(rd, cache_root)
        out.mkdir(parents=True, exist_ok=True)
        meta_p = out / "cache_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"files": {}}
        meta.update({"model": MODEL_ID, "resolution": RES, "grid": GRID, "dim": DIM, "fit": fit, "dtype": "float16",
                     "source_run": str(rd)})
        for k, p in enumerate(list_episodes(rd)):
            if max_episodes is not None and k >= max_episodes:
                break
            ep_name = p.stem
            for cam in cams:
                f = out / f"{ep_name}_{cam}.npy"
                vid = p.with_name(f"{ep_name}_{cam}.mp4")
                n_expect = int(np.load(p, allow_pickle=False)["action"].shape[0])
                st = vid.stat()
                sig = {"frames": n_expect, "video_bytes": st.st_size, "video_mtime_ns": st.st_mtime_ns}
                if f.exists() and meta["files"].get(f.name) == sig:  # up to date w.r.t. the source video
                    try:
                        if np.load(f, mmap_mode="r").shape[0] == n_expect:
                            stats["skipped"] += 1
                            continue
                    except Exception:  # noqa: BLE001  (truncated file from an interrupted run: recompute)
                        pass
                if ext is None:
                    ext = DinoExtractor(device)
                frames = read_video(vid)
                assert len(frames) == n_expect, f"{vid}: {len(frames)} frames vs {n_expect} steps"
                feats = ext(frames, batch_size, fit)
                tmp = f.with_suffix(".tmp.npy")
                np.save(tmp, feats)
                tmp.rename(f)  # atomic rename, so an interrupted run leaves no partial file
                meta["files"][f.name] = sig
                meta_p.write_text(json.dumps(meta, indent=2))  # after every file, so a restart resumes where it stopped
                stats["computed"] += 1
                stats["frames"] += int(feats.shape[0])
                if verbose:
                    print(f"[features] {rd.name}/{f.name} {feats.shape} ({time.time() - t0:.0f}s)", flush=True)
        meta_p.write_text(json.dumps(meta, indent=2))
    stats["wall_s"] = round(time.time() - t0, 1)
    return stats


def expand_runs(patterns: list[str]) -> list[Path]:
    out = []
    for pat in patterns:
        p = resolve(pat)
        if p.is_dir() and (p / "manifest.json").exists():
            out.append(p)
        else:
            out += sorted(q for q in p.parent.glob(p.name) if q.is_dir())
    return list(dict.fromkeys(out))


def main(argv=None):
    ap = std_parser(__doc__.split("\n")[0])
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = merge_smoke(load_yaml(args.config), args.smoke)
    device = pick_device(args.device or cfg.get("device", "auto"), need_gb=0.8, who="features")
    runs = expand_runs(cfg["rollouts"])
    if not runs:
        raise FileNotFoundError(f"no rollout runs for {cfg['rollouts']}")
    cache_root = resolve(cfg.get("cache_root", "runs/wm_cache"))
    stats = build_cache(runs, cache_root, list(cfg.get("cameras", ["front"])), device, int(cfg.get("batch_size", 32)),
                        cfg.get("fit", "crop"), cfg.get("max_episodes"))
    print(json.dumps(stats))
    write_receipt(cache_root, "wm_features", cfg, {"stats": stats, "runs": [str(r) for r in runs]}, t0)
    return stats


if __name__ == "__main__":
    main()
