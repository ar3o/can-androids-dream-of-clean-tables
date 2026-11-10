"""Shared paths and helpers for the table-scene videos (no heavy imports; GPU workers in other envs use it too).

Frames: every STRIDE-th frame of data/raw_clips/table_clean/<video>.MOV (29.97 fps -> 9.99 fps at STRIDE 3), long side WORK_LONG (960x540),
JPG q95 in data/table_clean/_work/<video>/frames/<k>.jpg. Decoded once with the vla env's OpenCV (same decoder and tone curve as cleantables.perception.recon).
Frame k <-> video frame k*STRIDE <-> time k*STRIDE/fps_video.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw_clips/table_clean"
OUT = ROOT / "data/table_clean"
WORK = OUT / "_work"
PROOF = ROOT / "proof/table_clean"
LOG = ROOT / "runs/table_clean_ingest/PROGRESS.md"
STRIDE = 3
WORK_LONG = 960
EPISODE_VIDEOS = ["IMG_2513", "IMG_2516", "IMG_2514", "IMG_2518"]   # IMG_2511 is a handheld walk-around for the splat, without episodes


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"\n## {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def extract_frames(video, stride=STRIDE, long=WORK_LONG):
    import cv2
    d = WORK / video / "frames"
    meta = WORK / video / "frames_meta.json"
    if meta.exists():
        return json.load(open(meta))
    cap = cv2.VideoCapture(str(RAW / f"{video}.MOV"))
    fps = cap.get(5)
    k = i = 0
    d.mkdir(parents=True, exist_ok=True)
    size = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            s = min(1.0, long / max(f.shape[:2]))
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else f
            cv2.imwrite(str(d / f"{k}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            size = [f.shape[1], f.shape[0]]
            k += 1
        i += 1
    cap.release()
    m = dict(n=k, fps_video=fps, n_video_frames=i, stride=stride, size_wh=size, fps=fps / stride)
    json.dump(m, open(meta, "w"))
    return m


def meta(video):
    return json.load(open(WORK / video / "frames_meta.json"))


def frame_path(video, k):
    return WORK / video / "frames" / f"{k}.jpg"


def read_frame(video, k):
    import cv2
    return cv2.imread(str(frame_path(video, k)))


def unpack(bits, shape):
    n = int(np.prod(shape))
    return np.unpackbits(bits)[:n].reshape(shape).astype(bool)


if __name__ == "__main__":
    import sys
    for v in sys.argv[1:]:
        print(v, extract_frames(v), flush=True)
