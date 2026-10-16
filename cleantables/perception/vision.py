"""Frozen DINOv2 features for the world model's object tokens -> vision.npz.

    python -m cleantables.perception.vision VIDEO [VIDEO ...]

object crop  every tracked detection's box (+15 % context), pixels outside its mask grey, 224x224 -> CLS (768)
scene crop   the table region of the tracked frame (+80 px above), 224x392 -> CLS (768)
Output: det (detection ids), emb (n, 768) fp16, keyframes, scene (n_frames, 768) fp16, scene_box.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from cleantables import paths

from .config import VisionConfig, load_config
from .files import gpu_free_mib, load_detections, read_frame, tracks, video_dir
from .objects import table_crop_box

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GREY = 124
CROP_PAD = 0.15
BATCH = 32


class Encoder:
    def __init__(self, cfg: VisionConfig):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = "cuda" if gpu_free_mib() >= cfg.gpu_min_free_mib else "cpu"
        self.model = AutoModel.from_pretrained(cfg.model).eval().to(self.device)

    def __call__(self, images) -> np.ndarray:
        x = (np.stack(images).astype(np.float32) / 255 - MEAN) / STD
        x = self.torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            h = self.model(pixel_values=x).last_hidden_state[:, 0]
        return self.torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy().astype(np.float16)


def object_crop(img_bgr: np.ndarray, mask_lo: np.ndarray, pad: float = CROP_PAD) -> np.ndarray:
    """Square RGB crop around the mask (224x224), pixels outside the mask grey."""
    m = cv2.resize(
        mask_lo.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.full((224, 224, 3), GREY, np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    half = max(x1 - x0, y1 - y0) * (1 + 2 * pad) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b, c, d = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    im = img_bgr[..., ::-1].copy()
    im[~m] = GREY
    out = np.full((d - b, c - a, 3), GREY, np.uint8)
    ya, yb, xa, xb = max(0, b), min(im.shape[0], d), max(0, a), min(im.shape[1], c)
    out[ya - b : yb - b, xa - a : xb - a] = im[ya:yb, xa:xb]
    return cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)


def scene_crop(img_bgr: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return cv2.resize(img_bgr[y0:y1, x0:x1][..., ::-1], (392, 224), interpolation=cv2.INTER_AREA)


def run(video: str, encoder: Encoder) -> None:
    out = paths.perception_dir(video) / "vision.npz"
    if out.exists():
        return
    R = tracks(video)
    dets = load_detections(paths.perception_dir(video) / "sam3_vinv.npz")
    box = table_crop_box(video, margin=24, up=80)
    ids, embs, frames, scene = [], [], [], []
    batch, batch_ids = [], []
    for k, kept in sorted(R["kept"].items()):
        img = read_frame(video, k)
        frames.append(k)
        scene.append(encoder([scene_crop(img, box)])[0])
        for i in kept:
            batch.append(object_crop(img, dets["masks"][i]))
            batch_ids.append(i)
        if len(batch) >= BATCH:
            embs.append(encoder(batch))
            ids += batch_ids
            batch, batch_ids = [], []
    if batch:
        embs.append(encoder(batch))
        ids += batch_ids
    np.savez_compressed(
        video_dir(video) / "vision.npz",
        det=np.array(ids, np.int32),
        emb=np.concatenate(embs) if embs else np.zeros((0, 768), np.float16),
        keyframes=np.array(frames, np.int32),
        scene=np.stack(scene),
        scene_box=np.array(box),
    )
    print(f"{video}: {len(ids)} object crops, {len(frames)} scene crops ({encoder.device})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default=None, help="perception config (default configs/perception.yaml)")
    args = ap.parse_args()
    encoder = Encoder((load_config(args.config) if args.config else load_config()).vision)
    for video in args.videos:
        run(video, encoder)


if __name__ == "__main__":
    main()
