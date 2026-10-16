"""Per-video perception files under data/perception/<video>/ and their loaders (also used by the twin and world model).

frames/<k>.jpg, frames_meta.json   work frames (10 fps, 960x540)
depth10.npy, depth10.json          MoGe-3 depth of every work frame at 270x480 (camera z, model scale) + intrinsics
det3d.npz, table.json              table frame: T / T_all (camera -> table), K, per-frame depth scale, table pixel mask
sam3_video.npz                     SAM 3 video masklets per phrase
tracks.pkl, sam3_vinv.npz          object identities with per-frame 3D boxes, and their per-frame masks
hand_video.npz                     SAM 3 'hand' masks of every tracked frame
vision.npz, vjepa.npz              DINOv2 crop features and V-JEPA 2 clip tokens
events.json                        rests, moves and end relations of every object
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from cleantables import paths

H_LO, W_LO = 270, 480  # depth and mask resolution: half the work frame
WORK_FPS = 10.0
AGENT_PHRASES = ["hand"]  # the person: prompted and tracked, never an object


def video_dir(video: str) -> Path:
    d = paths.perception_dir(video)
    d.mkdir(parents=True, exist_ok=True)
    return d


def frame_path(video: str, k: int) -> Path:
    return paths.perception_dir(video) / "frames" / f"{int(k)}.jpg"


def read_frame(video: str, k: int) -> np.ndarray:
    return cv2.imread(str(frame_path(video, k)))


def frame_meta(video: str) -> dict:
    return json.loads((paths.perception_dir(video) / "frames_meta.json").read_text())


def n_frames(video: str) -> int:
    return int(frame_meta(video)["n"])


def depth(video: str) -> np.ndarray:
    """(n, 270, 480) float16 camera depth in model scale; multiply by geometry()['scale'][k] for metres."""
    return np.load(paths.perception_dir(video) / "depth10.npy", mmap_mode="r")


def geometry(video: str) -> dict:
    """det3d.npz -> dict(T (4,4), T_all (n,4,4), K (3,3) at 960x540, scale (n,), table (270,480) bool)."""
    z = np.load(paths.perception_dir(video) / "det3d.npz")
    return dict(T=z["T"], T_all=z["T_all"], K=z["K"], scale=z["scale"].astype(np.float32), table=z["table"])


def tracks(video: str) -> dict:
    with open(paths.perception_dir(video) / "tracks.pkl", "rb") as f:
        return pickle.load(f)


def events(video: str) -> dict:
    return json.loads((paths.perception_dir(video) / "events.json").read_text())


def unpack_masks(bits: np.ndarray, shape) -> np.ndarray:
    n = int(np.prod(shape))
    return np.unpackbits(bits)[:n].reshape(tuple(int(s) for s in shape)).astype(bool)


def load_detections(path: Path) -> dict:
    """Detections (keyframe_dets.npz, sam3_vinv.npz): frame, prompt, prob, box (xyxy at 960x540), masks (m,270,480)."""
    z = np.load(path)
    m = int(z["shape"][0])
    masks = unpack_masks(z["masks"], (m, H_LO, W_LO)) if m else np.zeros((0, H_LO, W_LO), bool)
    return dict(
        frame=z["frame"],
        prompt=z["prompt"],
        prob=z["prob"],
        box=z["box"],
        masks=masks,
        prompts=[str(x) for x in z["prompts"]],
        frames=[int(x) for x in z["frames_done"]],
    )


def masklets(video: str):
    """sam3_video.npz -> (phrases, tracked frames, masklets [dict(phrase, prob (n_t,), masks (n_t,270,480))])."""
    z = np.load(paths.perception_dir(video) / "sam3_video.npz")
    phrases = [str(p) for p in z["phrases"]]
    out = []
    for pi, phrase in enumerate(phrases):
        shape = z[f"p{pi}_shape"]
        n_obj = int(shape[0])
        if n_obj == 0:
            continue
        masks = unpack_masks(z[f"p{pi}_masks"], shape)
        for j in range(n_obj):
            out.append(dict(phrase=phrase, prob=z[f"p{pi}_probs"][j], masks=masks[j]))
    return phrases, z["frames"].astype(int), out


def hand_masks(video: str) -> dict[int, np.ndarray]:
    """frame -> (270,480) bool SAM 3 'hand' mask."""
    z = np.load(paths.perception_dir(video) / "hand_video.npz")
    masks = unpack_masks(z["masks"], z["shape"])
    return {int(k): masks[t] for t, k in enumerate(z["frames"].tolist())}


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def gpu_free_mib() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 0


def wait_for_gpu(need_mib: int, what: str, poll_s: float = 60.0) -> None:
    """Block until the minimum free GPU memory over ~15 s reaches need_mib; other jobs on the GPU are never touched."""
    announced = False
    while True:
        free = []
        for _ in range(5):
            free.append(gpu_free_mib())
            time.sleep(3)
        if min(free) >= need_mib:
            return
        if not announced:
            print(f"{what}: waiting for {need_mib} MiB of free GPU memory ({min(free)} MiB free)", flush=True)
            announced = True
        time.sleep(poll_s)
