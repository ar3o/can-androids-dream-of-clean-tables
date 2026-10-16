"""Check video: every object's 3D box (from events.json) drawn on the real frames, with its phrase and the current
move's end relation -> data/perception/<video>/objects3d.mp4.

    python -m cleantables.perception.visualize VIDEO [VIDEO ...]
"""

from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np

from cleantables import paths

from .files import WORK_FPS, events, geometry, n_frames, read_frame
from .geometry import BOX_EDGES, box_corners3d, project
from .objects import vocabulary

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)


def colour(i: int):
    c = cv2.applyColorMap(np.uint8([[(i * 67 + 20) % 255]]), cv2.COLORMAP_HSV)[0, 0]
    return tuple(int(x) for x in c)


def banner(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    h = 28 + 24 * (len(lines) - 1)
    out[:h] = (out[:h] * 0.25).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (10, 20 + 24 * i),
            FONT,
            0.58,
            WHITE if i == 0 else (215, 215, 215),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )
    return out


def draw_box(img, T, K, pose: dict, size, c, label: str) -> None:
    L, W, H = size
    yaw = pose["yaw"] if np.isfinite(pose["yaw"]) else 0.0
    uv, z = project(box_corners3d(pose["x"], pose["y"], max(0.0, pose["top"] - H), L, W, H, yaw), K, T)
    if (z <= 0).any() or np.abs(uv).max() > 5000:
        return
    uv = uv.astype(int)
    for a, b in BOX_EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), c, 2, cv2.LINE_AA)
    x, y = int(uv[4:, 0].mean()) - 4 * len(label), int(uv[4:, 1].min()) - 6
    cv2.putText(img, label, (x, y), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (x, y), FONT, 0.45, c, 1, cv2.LINE_AA)


def relation_text(e: dict) -> str:
    if e["relation"] == "table":
        return "onto the table"
    if e["relation"] == "leaves":
        return "out of view"
    return f"{e['relation'].replace('_', ' ')} '{e['target_phrase']}'"


def pose_at(o: dict, moves: list[dict], f: int):
    """Pose of object o at frame f: on the carry path while moving, else from its rests; None when hidden or carried."""
    for m in moves:
        if o["id"] in m["members"] and m["f_start"] <= f <= m["f_end"] and m["path"]:
            if m["obj"] != o["id"]:
                return None
            path = np.array(m["path"])
            s = (f - m["f_start"]) / max(1, m["f_end"] - m["f_start"])
            h = s * (len(path) - 1)
            lo = int(h)
            p = path[lo] * (1 - (h - lo)) + path[min(lo + 1, len(path) - 1)] * (h - lo)
            y0 = m["start"]["yaw"]
            y1 = m["end"]["yaw"] if m["end"] else y0
            yaw = y0 if not np.isfinite(y1) else y0 + s * (((y1 - y0) + np.pi / 2) % np.pi - np.pi / 2)
            return dict(x=p[0], y=p[1], top=p[2], yaw=yaw)
    current = [r for r in o["rests"] if r["f0"] - 2 <= f <= r["f1"] + 2]
    if current:
        return current[0]["pose"]
    before = [r for r in o["rests"] if r["f1"] < f]
    if before and f <= o["last"] + 5:
        return before[-1]["pose"]
    return None


def objects3d(video: str) -> None:
    E = events(video)
    g = geometry(video)
    n = n_frames(video)
    out = paths.perception_dir(video) / "objects3d.mp4"
    h, w = read_frame(video, 0).shape[:2]
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(WORK_FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(out),
    ]
    names = ", ".join(vocabulary(video))
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ffmpeg:
        for f in range(n):
            img = read_frame(video, f)
            for o in E["objects"]:
                p = pose_at(o, E["events"], f)
                if p is not None and np.isfinite(p["x"]):
                    draw_box(img, g["T_all"][f], g["K"], p, o["size"], colour(o["id"]), f"{o['id']}:{o['phrase']}")
            lines = [
                f"{video}  t={f / WORK_FPS:.1f}s   3D object boxes (SAM 3 mask + depth)",
                f"phrases: {names}"[:120],
            ]
            for e in [e for e in E["events"] if e["f_start"] <= f <= e["f_end"] + 10][:2]:
                who = f"'{e['phrase']}'" + (f" + {len(e['members']) - 1} carried with it" if e["group"] else "")
                lines.append(f"move: {who} -> {relation_text(e)}")
            ffmpeg.stdin.write(banner(img, lines).tobytes())
        ffmpeg.stdin.close()
    print(f"{video}: {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    for video in ap.parse_args().videos:
        objects3d(video)


if __name__ == "__main__":
    main()
