"""Split each recorded video into tasks with Gemini Flash; writes data/episodes_raw.json.

    python -m cleantables.episodes.segment [VIDEO ...] [--config configs/episodes.yaml] [--redo]

Needs GEMINI_API_KEY. Videos already in the output file are skipped unless --redo is given.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from cleantables import paths

CONFIG = paths.REPO / "configs" / "episodes.yaml"

PROMPT = (
    "You are the task-segmentation step of an automatic robot-learning data pipeline. Input: a phone video of one"
    " person doing table-top tasks for data collection (fixed camera). Sample it at about 5 frames per second. "
    "Split the video into TASKS. A task is one complete goal the person achieves on the table (moving a set of "
    "objects to where they belong). You decide what the tasks are, which objects each involves and where they go."
    " This is repetitive data collection: between tasks the person often resets the scene (undoes the task, "
    "spreads objects out, moves things back). Resetting is NOT a task. Do not output it. Only output the tasks. "
    "For each task give: start_s: the moment the person starts moving the FIRST object towards its goal. end_s: "
    "about 1 second AFTER the last object has come to rest at its goal and the hands have moved away (so the "
    "finished state is clearly visible), and before any reset starts. instruction: a short natural-language "
    "instruction a user could give to get this task done. objects: the objects moved in this task (your own short"
    " names). destination: where they ended up (your own words). complete: true if every object of the task "
    'reached its goal, false otherwise. note: anything unclear (one sentence, or ""). Times are in seconds of the'
    " original video. Be precise to about 0.2 s.\n\n"
    'Answer with one JSON object for this video only: {"tasks": [{"start_s": ..., "end_s": ..., "instruction": '
    '..., "objects": [...], "destination": ..., "complete": ..., "note": ...}, ...]} in time order, or {"tasks": '
    "[]} if there is none."
)

TASK_FIELDS = ("instruction", "start_s", "end_s", "objects", "destination", "complete", "note")


@dataclass
class SegmentConfig:
    model: str = "gemini-2.5-flash"
    sample_fps: int = 5
    poll_s: float = 5.0


def load_config(path: Path = CONFIG) -> SegmentConfig:
    with open(path) as f:
        return SegmentConfig(**(yaml.safe_load(f).get("segment") or {}))


def make_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")
    return genai.Client(api_key=key)


def ask_gemini(client, video_file: Path, cfg: SegmentConfig) -> str:
    """Upload one video, wait until it is processed and return the model's raw JSON answer."""
    from google.genai import types

    uploaded = client.files.upload(file=str(video_file))
    try:
        while uploaded.state.name == "PROCESSING":
            time.sleep(cfg.poll_s)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"{video_file.name}: upload ended in state {uploaded.state.name}")
        video = types.Part(
            file_data=types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
            video_metadata=types.VideoMetadata(fps=cfg.sample_fps),
        )
        response = client.models.generate_content(
            model=cfg.model,
            contents=[video, PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
        )
        return response.text
    finally:
        client.files.delete(name=uploaded.name)


def parse_answer(video: str, text: str) -> list[dict]:
    """Model answer -> task list with ids <video>_tNN, sorted by start time."""
    answer = json.loads(text)
    tasks = answer["tasks"] if isinstance(answer, dict) else answer
    tasks = sorted(tasks, key=lambda t: float(t["start_s"]))
    out = []
    for i, task in enumerate(tasks, start=1):
        row = {"id": f"{video}_t{i:02d}"}
        for key in TASK_FIELDS:
            row[key] = task.get(key, "" if key in ("destination", "note", "instruction") else None)
        row["start_s"] = round(float(row["start_s"]), 1)
        row["end_s"] = round(float(row["end_s"]), 1)
        row["objects"] = list(row["objects"] or [])
        row["complete"] = bool(row["complete"])
        out.append(row)
    return out


def segment_videos(videos: list[str], cfg: SegmentConfig, out_path: Path, redo: bool = False, client=None) -> dict:
    data = json.loads(out_path.read_text()) if out_path.exists() else {"videos": []}
    by_video = {row["video"]: row for row in data["videos"]}
    for video in videos:
        if video in by_video and not redo:
            continue
        client = client or make_client()
        t0 = time.time()
        tasks = parse_answer(video, ask_gemini(client, paths.video_path(video), cfg))
        by_video[video] = {"video": video, "fps_used": cfg.sample_fps, "tasks": tasks}
        data["videos"] = [by_video[v] for v in sorted(by_video)]
        write_json(data, out_path)
        print(f"{video}: {len(tasks)} tasks, {time.time() - t0:.0f} s")
    return data


def write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def all_videos() -> list[str]:
    return sorted(p.stem for p in paths.VIDEOS.iterdir() if p.suffix.lower() in (".mov", ".mp4"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="video names (default: every video in data/videos)")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, default=paths.EPISODES_RAW)
    ap.add_argument("--redo", action="store_true", help="segment videos that are already in the output again")
    args = ap.parse_args()
    segment_videos(args.videos or all_videos(), load_config(args.config), args.out, args.redo)


if __name__ == "__main__":
    main()
