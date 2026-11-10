"""Shared paths and helpers of the per-clip reconstruction (data/processed/recon_v3). No heavy imports: the GPU workers in other envs
use it too.

Frames: every STRIDE-th frame of data/episodes_v2/<clip>.mp4, resized so the long side is WORK_LONG (the same pixel space as the
grounding frames). Recon frame k <-> video frame k*STRIDE <-> time k*STRIDE/fps_video.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EPS = ROOT / "data/episodes_v2"
OUT = ROOT / "data/processed/recon_v3"
WORK = OUT / "_work"
ALERT = ROOT / "runs/overnight2/ALERT.md"
STRIDE = 2
WORK_LONG = 960

# Task prompts: the only per-clip input besides the video. Phrase per object name = cleantables.perception.ground PROMPT (same wording).
CONTAINER_PROMPT = "a white box"


def alert(msg):
    ALERT.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT, "a") as f:
        f.write(f"## {datetime.datetime.now():%a %d %b %H:%M:%S %Y} recon_v3: {msg}\n")


def all_clips():
    return sorted(p.stem for p in EPS.glob("*.mp4"))


def prompts():
    """{clip: {"object": phrase, "container": phrase}} from data/processed/recon_v3/prompts.json (written once by make_prompts)."""
    return json.load(open(OUT / "prompts.json"))


def make_prompts():
    """Write prompts.json: the task text per clip. Only the object name of the clip is read from object_sizes.json['clip_objects'] (the
    task description of what was manipulated) and turned into the grounding phrase; no size is read."""
    from cleantables.perception.ground import BOX, PROMPT
    names = json.load(open(ROOT / "data/object_sizes.json"))["clip_objects"]
    pr = {c: {"object": PROMPT.get(names.get(c, ""), "a flat object"), "container": BOX, "object_name": names.get(c, "")} for c in all_clips()}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(pr, open(OUT / "prompts.json", "w"), indent=1)
    return pr


def extract_frames(clip, stride=STRIDE, long=WORK_LONG):
    """Decode the clip once (OpenCV of the vla env) and cache the work frames as JPG q95 in _work/<clip>/frames/<k>.jpg + meta.json.
    Every stage (MoGe, SAM 3 in its own env, fitting) reads these, so all see identical pixels (the system ffmpeg decodes these HDR
    phone videos with a different tone curve: mean grey 162 vs 187)."""
    import cv2
    d = WORK / clip / "frames"
    meta = WORK / clip / "frames_meta.json"
    if meta.exists():
        return json.load(open(meta))
    path = EPS / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(path))
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
    if k == 0:
        raise RuntimeError(f"no frames decoded from {path} (OpenCV in this env has no video I/O? run extraction in the vla env)")
    m = dict(n=k, fps_video=fps, n_video_frames=i, stride=stride, size_wh=size, fps=fps / stride)
    json.dump(m, open(meta, "w"))
    return m


def read_work_frames(clip):
    """-> (frames list BGR, meta). Frames must have been extracted (extract_frames, vla env)."""
    import cv2
    meta = WORK / clip / "frames_meta.json"
    if not meta.exists():
        raise RuntimeError(f"{clip}: frames not extracted (extract the frames first)")
    m = json.load(open(meta))
    fr = [cv2.imread(str(WORK / clip / "frames" / f"{k}.jpg")) for k in range(m["n"])]
    if any(f is None for f in fr):
        raise RuntimeError(f"{clip}: missing frame jpg")
    return fr, m


def unpack(bits, shape):
    n = int(np.prod(shape))
    return np.unpackbits(bits)[:n].reshape(shape).astype(bool)
