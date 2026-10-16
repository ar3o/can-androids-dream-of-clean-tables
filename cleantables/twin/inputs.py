"""Read-only access to the perception outputs of one video and to the twin output folders.

Frames are the 10 fps work frames at 960x540; depth, detection masks and the table mask live on a 270x480 grid.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

from cleantables.paths import EPISODES, TWINS, perception_dir, twin_dir

LIBRARY = TWINS / "library"  # <key>/: one mesh per physical object, shared by every episode
PLATES = TWINS / "plates"  # <video>/: static background colour and depth of one video
HUNYUAN = TWINS / "hy3d"  # Hunyuan3D-2.1 inputs and outputs used to make library meshes

H, W = 270, 480  # depth / mask grid
IMG_W, IMG_H = 960, 540  # frame size
FPS = 10.0


def episodes(task: str | None = None) -> list[dict]:
    eps = json.loads(EPISODES.read_text())["tasks"]
    return [e for e in eps if task is None or e["task"] == task]


def episode(episode_id: str) -> dict:
    for e in episodes():
        if e["id"] == episode_id:
            return e
    raise KeyError(f"no episode {episode_id} in {EPISODES}")


def plate_dir(video: str) -> Path:
    return PLATES / video


def frames_dir(video: str) -> Path:
    return perception_dir(video) / "frames"


def frame(video: str, k: int) -> np.ndarray:
    """BGR work frame k (10 fps, 960x540)."""
    img = cv2.imread(str(frames_dir(video) / f"{int(k)}.jpg"))
    if img is None:
        raise FileNotFoundError(f"missing frame {k} of {video} in {frames_dir(video)}")
    return img


def camera(video: str) -> dict:
    """static phone camera: T (camera -> table), K at 960x540, per-frame depth scale, table mask (270x480)."""
    z = np.load(perception_dir(video) / "det3d.npz")
    return dict(T=z["T"], K=z["K"], scale=z["scale"].astype(np.float32), table=z["table"])


def depth(video: str) -> np.ndarray:
    """per-frame MoGe depth (n, 270, 480), unscaled; multiply by camera()['scale'][k] for metres."""
    return np.load(perception_dir(video) / "depth10.npy", mmap_mode="r")


def tracks(video: str) -> dict:
    with open(perception_dir(video) / "tracks.pkl", "rb") as f:
        return pickle.load(f)


def events(video: str) -> dict:
    return json.loads((perception_dir(video) / "events.json").read_text())


class Detections:
    """SAM 3 detection masks of one video (id -> frame, phrase, 270x480 mask), unpacked on demand."""

    def __init__(self, video: str):
        z = np.load(perception_dir(video) / "sam3_vinv.npz")
        self.frame, self.prompt, self.prob = z["frame"], z["prompt"], z["prob"]
        self.prompts = [str(p) for p in z["prompts"]]
        self.bits = z["masks"]
        self.n_bytes = H * W // 8

    def mask(self, i: int) -> np.ndarray:
        return np.unpackbits(self.bits[i * self.n_bytes : (i + 1) * self.n_bytes])[: H * W].reshape(H, W).astype(bool)


def hand_masks(video: str) -> dict[int, np.ndarray]:
    """frame k -> (270, 480) SAM 3 'hand' video mask."""
    z = np.load(perception_dir(video) / "hand_video.npz")
    shape = tuple(int(s) for s in z["shape"])
    masks = np.unpackbits(z["masks"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    return {int(k): masks[t] for t, k in enumerate(z["frames"].tolist())}


def upsample_mask(m: np.ndarray) -> np.ndarray:
    """270x480 mask -> 960x540."""
    return cv2.resize(m.astype(np.uint8), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST).astype(bool)


def write_json(obj, path: Path | str) -> None:
    """atomic JSON write; numpy values become lists / floats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)


__all__ = ["twin_dir"]
