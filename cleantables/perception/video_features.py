"""Frozen V-JEPA 2 clip tokens (Assran et al. 2025, facebook/vjepa2-vitl-fpc64-256) every 0.2 s of video -> vjepa.npz.

    python -m cleantables.perception.video_features VIDEO [VIDEO ...]

The clip for keyframe k is 16 frames spread over the second ending at k (at 10 fps some frames repeat), squashed to
256x256 with ImageNet normalisation, fp16. Encoder output (8 x 16 x 16 x 1024) is pooled to 32 tokens: 16 from the
time-averaged grid and 16 from the last time step (4x4 average pooling each), plus one global mean token.
The same is stored for the horizontally flipped clip.
Output: k (n,), tok (n, 32, 1024), glob (n, 1024), tok_f, glob_f; all fp16.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from cleantables import paths

from .config import VideoFeaturesConfig, load_config
from .files import WORK_FPS, frame_path, n_frames, wait_for_gpu

N_CLIP_FRAMES = 16
RESOLUTION = 256
KEYFRAME_STEP = 2  # 5 keyframes per second at 10 fps
BATCH = 8
MEMORY_FRACTION_GIB = 3.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def clip_indices(ks, fps: float) -> np.ndarray:
    """(n, 16) frame indices of the clip of every keyframe: 16 positions over the last second, clamped at 0."""
    back = fps * (N_CLIP_FRAMES - 1 - np.arange(N_CLIP_FRAMES)) / (N_CLIP_FRAMES - 1)
    return np.clip(np.round(np.asarray(ks)[:, None] - back[None]).astype(int), 0, None)


def nearest_keyframe(ks: np.ndarray, k: int) -> int:
    """Index of the keyframe nearest to frame k (the earlier one on ties)."""
    return int(np.argmin(np.abs(ks - k) + 1e-3 * (ks > k)))


def load(video: str) -> dict:
    z = np.load(paths.perception_dir(video) / "vjepa.npz")
    return {k: z[k] for k in z.files}


class Encoder:
    def __init__(self, cfg: VideoFeaturesConfig, device: str = "cuda"):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(cfg.model, torch_dtype=torch.float16).to(device).eval()

    def prepare(self, path):
        im = cv2.imread(str(path))[..., ::-1].astype(np.float32) / 255
        im = (cv2.resize(im, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA) - MEAN) / STD
        return self.torch.from_numpy(im).permute(2, 0, 1).half()

    def pool(self, x):
        return self.torch.nn.functional.avg_pool2d(x.permute(0, 3, 1, 2), 4).flatten(2).transpose(1, 2)

    def encode(self, X, idx: np.ndarray, flip: bool = False):
        """X (n, 3, 256, 256) fp16 frames on the CPU; idx (m, 16) -> tokens (m, 32, 1024), global (m, 1024)."""
        torch = self.torch
        tokens, globals_ = [], []
        with torch.no_grad():
            for i in range(0, len(idx), BATCH):
                clips = X[torch.as_tensor(idx[i : i + BATCH])].to(self.device)
                if flip:
                    clips = clips.flip(-1)
                h = self.model(pixel_values_videos=clips, skip_predictor=True).last_hidden_state.float()
                h = h.reshape(len(clips), 8, 16, 16, -1)
                globals_.append(h.mean((1, 2, 3)).cpu())
                tokens.append(torch.cat([self.pool(h.mean(1)), self.pool(h[:, -1])], 1).cpu())
        return torch.cat(tokens).numpy().astype(np.float16), torch.cat(globals_).numpy().astype(np.float16)

    def video(self, video: str) -> dict:
        n = n_frames(video)
        X = self.torch.stack([self.prepare(frame_path(video, k)) for k in range(n)])
        ks = np.arange(0, n, KEYFRAME_STEP)
        idx = clip_indices(ks, WORK_FPS)
        tok, glob = self.encode(X, idx)
        tok_f, glob_f = self.encode(X, idx, flip=True)
        return dict(k=ks.astype(np.int32), tok=tok, glob=glob, tok_f=tok_f, glob_f=glob_f)


def run(videos: list[str], cfg: VideoFeaturesConfig) -> None:
    todo = [v for v in videos if not (paths.perception_dir(v) / "vjepa.npz").exists()]
    if not todo:
        return
    import torch

    wait_for_gpu(cfg.min_free_mib, "v-jepa 2")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.99, MEMORY_FRACTION_GIB * 2**30 / total))
    encoder = Encoder(cfg)
    for video in todo:
        t0 = time.time()
        out = paths.perception_dir(video) / "vjepa.npz"
        tmp = out.with_name("vjepa.tmp.npz")
        np.savez(tmp, **encoder.video(video))
        tmp.replace(out)
        print(f"{video}: v-jepa 2 tokens, {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default=None, help="perception config (default configs/perception.yaml)")
    args = ap.parse_args()
    run(args.videos, (load_config(args.config) if args.config else load_config()).video_features)


if __name__ == "__main__":
    main()
