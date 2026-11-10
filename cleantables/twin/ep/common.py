"""Shared paths and loaders for the per-episode twins (cleantables.twin.ep). Inputs are read only:
  frames  data/table_clean/_work/<V>/frames/<k>.jpg (10 fps, 960x540)
  geometry data/twin_table/boxstack/<V>/{det3d.npz, depth10.npy}   (static phone: T, K, per-frame MoGe depth scale, table mask)
  objects data/generic/<V>/{tracks.pkl, events.json, sam3_vinv.npz, hand_video.npz}
Outputs: data/twin_table/ep/ (library/, _video/<V>/, <episode_id>/).
"""
from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/table_clean/_work"
GEOM = ROOT / "data/twin_table/boxstack"
GEN = ROOT / "data/generic"
OUT = ROOT / "data/twin_table/ep"
LIB = OUT / "library"
H, W = 270, 480
FPS = 10.0


def episodes(task=None):
    # TWIN_TASKS_JSON: another label file in the same format (the held-out demo clip, kept out of tasks_train.json so no world-model training sees it)
    E = json.load(open(os.environ.get("TWIN_TASKS_JSON") or ROOT / "data/labels/tasks_train.json"))["tasks"]
    return [e for e in E if task is None or e["task"] == task]


def frame(v, k):
    import cv2
    return cv2.imread(str(WORK / v / "frames" / f"{int(k)}.jpg"))


def n_frames(v):
    return json.load(open(WORK / v / "frames_meta.json"))["n"]


def geom(v):
    z = np.load(GEOM / v / "det3d.npz")
    return dict(T=z["T"], K=z["K"], scale=z["scale"].astype(np.float32), table=z["table"])


def depth(v):
    return np.load(GEOM / v / "depth10.npy", mmap_mode="r")


def tracks(v):
    return pickle.load(open(GEN / v / "tracks.pkl", "rb"))


def events(v):
    return json.load(open(GEN / v / "events.json"))


class Dets:
    """Per-detection masks of the generic pipeline (sam3_vinv.npz: det id i -> frame, phrase, mask 270x480)."""

    def __init__(self, v):
        z = np.load(GEN / v / "sam3_vinv.npz")
        self.frame, self.prompt, self.prob = z["frame"], z["prompt"], z["prob"]
        self.prompts = [str(p) for p in z["prompts"]]
        self.bits = z["masks"]
        self.nb = H * W // 8

    def mask(self, i):
        return np.unpackbits(self.bits[i * self.nb:(i + 1) * self.nb])[:H * W].reshape(H, W).astype(bool)


def hand_masks(v):
    """-> dict k -> (270,480) bool SAM 3 'hand' video masks."""
    z = np.load(GEN / v / "hand_video.npz")
    shp = tuple(int(s) for s in z["shape"])
    M = np.unpackbits(z["masks"])[:int(np.prod(shp))].reshape(shp).astype(bool)
    return {int(k): M[t] for t, k in enumerate(z["frames"].tolist())}


def to_table(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def rays_lo(K):
    """Camera-frame rays (z=1) of the 270x480 depth grid (pixel centres in 960x540 coordinates)."""
    vv, uu = np.mgrid[0:H, 0:W]
    u, v = uu * 2 + 0.5, vv * 2 + 0.5
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], -1)


def project(P, K, T):
    """Table-frame points -> pixels (960x540) and camera depth. T = cam -> table."""
    Ti = np.linalg.inv(T)
    pc = P @ Ti[:3, :3].T + Ti[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def plane_hit(px, K, T, z=0.0):
    """Pixels (N,2) at 960x540 -> points on the plane z (table frame)."""
    rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T
    d = rays @ T[:3, :3].T
    o = T[:3, 3]
    s = (z - o[2]) / d[:, 2]
    return o[None] + s[:, None] * d


def table_extent(T, K, table_mask):
    from cleantables.perception.boxes.scene_export import table_extent as te
    return te(T, K, table_mask)


def jdump(obj, path):
    import os
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)


def lab(rgb):
    import cv2
    a = np.uint8([[np.clip(rgb, 0, 255)]])
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[0, 0].astype(float)


def yaw_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
