"""Readers for the perception outputs the world model consumes (data/perception/<video>/), plus the frozen text encoder.

Frame convention: table frame, z up, table top at z = 0, metres. Depth maps are 270x480; intrinsics K are for 960x540.
"""

from __future__ import annotations

import json
import pickle

import numpy as np

from cleantables.paths import perception_dir

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384
FPS = 10.0  # work-frame rate of every perception output


def load_geometry(video: str) -> dict:
    """Camera-to-table transforms (per frame), intrinsics, per-frame depth scale and the table pixel mask."""
    z = np.load(perception_dir(video) / "det3d.npz")
    return dict(T=z["T"], T_all=z["T_all"], K=z["K"], scale=z["scale"].astype(np.float32), table=z["table"])


def load_depth(video: str) -> np.ndarray:
    return np.load(perception_dir(video) / "depth10.npy", mmap_mode="r")


def load_events(video: str) -> dict:
    return json.load(open(perception_dir(video) / "events.json"))


def load_tracks(video: str) -> dict:
    with open(perception_dir(video) / "tracks.pkl", "rb") as f:
        return pickle.load(f)


def load_crop_embeddings(video: str, dtype=np.float16) -> dict:
    """DINOv2 crop embedding of every detection id."""
    z = np.load(perception_dir(video) / "vision.npz")
    return dict(zip(z["det"].tolist(), z["emb"].astype(dtype), strict=False))


def load_vjepa(video: str) -> dict:
    """V-JEPA 2 tokens per keyframe: k (n,), tok (n,32,1024), glob (n,1024), tok_f / glob_f (horizontal flip)."""
    z = np.load(perception_dir(video) / "vjepa.npz")
    return {k: z[k] for k in z.files}


def nearest_keyframe(vj: dict, k: int) -> int:
    """Index of the V-JEPA keyframe nearest to frame k (the earlier one on ties)."""
    ks = vj["k"]
    return int(np.argmin(np.abs(ks - k) + 1e-3 * (ks > k)))


def box_features(box: dict) -> np.ndarray:
    """[x, y, top, long side, short side, height, sin 2yaw, cos 2yaw] of an oriented box.

    Height equals top because every box rests on the table.
    """
    return np.array(
        [
            box["x"],
            box["y"],
            box["top"],
            max(box["L"], box["W"]),
            min(box["L"], box["W"]),
            box["top"],
            np.sin(2 * box["yaw"]),
            np.cos(2 * box["yaw"]),
        ],
        np.float32,
    )


def to_table(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def backproject(depth_lo: np.ndarray, K: np.ndarray, mask_lo: np.ndarray | None = None):
    """Camera-frame points (n,3) of the (masked) pixels of a 270x480 depth map, and their pixel indices."""
    ys, xs = np.nonzero((depth_lo > 0) & (mask_lo if mask_lo is not None else True))
    z = depth_lo[ys, xs]
    u, v = xs * 2 + 0.5, ys * 2 + 0.5
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1), (ys, xs)


def table_extent(T: np.ndarray, K: np.ndarray, table_mask: np.ndarray) -> list:
    """Axis-aligned [[x0, x1], [y0, y1]] of the table pixels intersected with the table plane (2-98 percentile)."""
    normal, offset = T[2, :3], T[2, 3]
    ys, xs = np.nonzero(table_mask[::3, ::3])
    ys, xs = ys * 3, xs * 3
    rays = np.stack([(xs * 2 + 0.5 - K[0, 2]) / K[0, 0], (ys * 2 + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1)
    t = -offset / (rays @ normal)
    P = to_table(T, rays[t > 0] * t[t > 0, None])
    lo, hi = np.percentile(P[:, :2], 2, 0), np.percentile(P[:, :2], 98, 0)
    return [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]


def table_token_geometry(x_range, y_range) -> np.ndarray:
    """Box features of the table token: centre, extent, zero height, yaw 0."""
    return np.array(
        [
            (x_range[0] + x_range[1]) / 2,
            (y_range[0] + y_range[1]) / 2,
            0,
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            0,
            0,
            1,
        ],
        np.float32,
    )


_TEXT_ENCODER = {}


def embed_texts(texts) -> np.ndarray:
    """MiniLM sentence embeddings (mean-pooled, L2-normalised), (n, 384) float32."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if "model" not in _TEXT_ENCODER:
        _TEXT_ENCODER["tok"] = AutoTokenizer.from_pretrained(TEXT_MODEL)
        _TEXT_ENCODER["model"] = AutoModel.from_pretrained(TEXT_MODEL).eval()
    texts = list(texts)
    if not texts:
        return np.zeros((0, TEXT_DIM), np.float32)
    out = []
    for i in range(0, len(texts), 64):
        b = _TEXT_ENCODER["tok"](texts[i : i + 64], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = _TEXT_ENCODER["model"](**b).last_hidden_state
        e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)
