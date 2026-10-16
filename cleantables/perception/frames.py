"""Work frames: every video decoded at about 10 fps, landscape, 960x540 JPEG -> data/perception/<video>/frames/.

python -m cleantables.perception.frames VIDEO [VIDEO ...]
"""

from __future__ import annotations

import argparse
import json
import math

import cv2

from cleantables import paths

from .config import FramesConfig, load_config


def frame_stride(fps: float, work_fps: float) -> int:
    return max(1, int(round(fps / work_fps)))


def work_frame_count(video: str, work_fps: float = 10.0) -> int:
    """Number of work frames of a video: read from frames_meta.json when extracted, else from the container."""
    meta = paths.perception_dir(video) / "frames_meta.json"
    if meta.exists():
        return int(json.loads(meta.read_text())["n"])
    cap = cv2.VideoCapture(str(paths.video_path(video)))
    count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return math.ceil(count / frame_stride(fps, work_fps))


def extract(video: str, cfg: FramesConfig) -> dict:
    """Decode every stride-th frame; portrait video is rotated 90 degrees clockwise."""
    out_dir = paths.perception_dir(video) / "frames"
    meta_path = paths.perception_dir(video) / "frames_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    src = paths.video_path(video)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = frame_stride(fps, cfg.fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = i = 0
    rotated = False
    size = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if f.shape[0] > f.shape[1]:
                f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
                rotated = True
            if f.shape[:2] != (cfg.height, cfg.width):
                f = cv2.resize(f, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{k}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            size = [f.shape[1], f.shape[0]]
            k += 1
        i += 1
    cap.release()
    if k == 0:
        raise RuntimeError(f"no frames decoded from {src}")
    meta = dict(
        n=k,
        fps_video=fps,
        n_video_frames=i,
        stride=stride,
        size_wh=size,
        fps=fps / stride,
        src=src.name,
        rotated_portrait=rotated,
    )
    meta_path.write_text(json.dumps(meta))
    print(f"{video}: {k} frames at {fps / stride:.2f} fps" + (" (portrait, rotated)" if rotated else ""), flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default=None, help="perception config (default configs/perception.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    for video in args.videos:
        extract(video, cfg.frames)


if __name__ == "__main__":
    main()
