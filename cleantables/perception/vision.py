"""Frozen visual features for the world model's tokens (DINOv2-base, facebook/dinov2-base, Apache-2.0, cached locally):
  * object crop  : each kept SAM 3 detection's box crop (+ 15 % context), pixels outside its mask set to grey, 224x224 -> CLS (768)
  * scene token  : the table-region crop of the keyframe (table pixels from det3d + room above for edges / walls), 224x392 -> CLS (768)
Everything comes from one camera frame + its SAM 3 masks, so the same function serves a sim render at inference (render -> SAM 3 ->
features), see frame_model.tokens_from_detections.
  $VLA_PY -m cleantables.perception.vision VIDEOS...   -> data/generic/<V>/vision.npz (det ids, crop emb, keyframes, scene emb)
"""
from __future__ import annotations

import pickle
import sys

import cv2
import numpy as np
import torch

from cleantables.perception.common import frame, gpu_free, log, names, table_crop_box, vdir
from cleantables.perception.tracking import load_inventory

MODEL = "facebook/dinov2-base"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_M = {}


def model():
    from transformers import AutoModel
    if "m" not in _M:
        dev = "cuda" if gpu_free() >= 1500 else "cpu"
        _M["m"] = AutoModel.from_pretrained(MODEL).eval().to(dev)
        _M["dev"] = dev
    return _M["m"], _M["dev"]


def _enc(ims):
    m, dev = model()
    x = torch.from_numpy(((np.stack(ims).astype(np.float32) / 255 - MEAN) / STD)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        h = m(pixel_values=x).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy().astype(np.float16)


def crop_obj(img_bgr, mask_lo, pad=0.15):
    """img (540,960,3) BGR, mask (270,480) -> 224x224 RGB with non-mask pixels grey."""
    m = cv2.resize(mask_lo.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.full((224, 224, 3), 124, np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    s = max(x1 - x0, y1 - y0) * (1 + 2 * pad) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b, c, d = int(cx - s), int(cy - s), int(cx + s), int(cy + s)
    im = img_bgr[..., ::-1].copy()
    im[~m] = 124
    out = np.full((d - b, c - a, 3), 124, np.uint8)
    ya, yb, xa, xb = max(0, b), min(im.shape[0], d), max(0, a), min(im.shape[1], c)
    out[ya - b:yb - b, xa - a:xb - a] = im[ya:yb, xa:xb]
    return cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)


def scene_crop(img_bgr, box):
    x0, y0, x1, y1 = box
    return cv2.resize(img_bgr[y0:y1, x0:x1][..., ::-1], (392, 224), interpolation=cv2.INTER_AREA)


def run(v):
    p = vdir(v) / "vision.npz"
    if p.exists():
        return
    R = pickle.load(open(vdir(v) / "tracks.pkl", "rb"))
    d = load_inventory(v)
    box = table_crop_box(v, margin=24, up=80)
    ids, embs, kfs, sc = [], [], [], []
    batch, bid = [], []
    for k, dets in sorted(R["kept"].items()):
        img = frame(v, k)
        kfs.append(k); sc.append(_enc([scene_crop(img, box)])[0])
        for i in dets:
            batch.append(crop_obj(img, d["masks"][i])); bid.append(i)
        if len(batch) >= 32:
            embs.append(_enc(batch)); ids += bid; batch, bid = [], []
    if batch:
        embs.append(_enc(batch)); ids += bid
    np.savez_compressed(p, det=np.array(ids, np.int32), emb=np.concatenate(embs) if embs else np.zeros((0, 768), np.float16),
                        keyframes=np.array(kfs, np.int32), scene=np.stack(sc), scene_box=np.array(box))
    log(f"{v} vision: {len(ids)} object crops + {len(kfs)} scene crops ({_M['dev']})")


if __name__ == "__main__":
    for v in names(sys.argv[1:]):
        run(v)
