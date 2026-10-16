"""Turn the raw segments into the episode list every later stage reads (data/episodes.json).

    python -m cleantables.episodes.prepare [--config configs/episodes.yaml]

Each kept task gets a margin before and after (clipped to the video), keeps its labelled times as core_start_s /
core_end_s and gets a task type from its instruction. Which videos are used is listed in the config.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cleantables import paths

CONFIG = paths.REPO / "configs" / "episodes.yaml"
SOURCE = (
    "episodes_raw.json (video-language model segmentation) + {margin:g} s margin before/after; excluded: {excluded}"
)


@dataclass
class PrepareConfig:
    margin_s: float = 1.0
    work_fps: float = 10.0
    videos: list[str] = field(default_factory=list)
    task_keywords: dict[str, str] = field(default_factory=dict)
    excluded_videos: dict[str, str] = field(default_factory=dict)


def load_config(path: Path = CONFIG) -> PrepareConfig:
    with open(path) as f:
        return PrepareConfig(**(yaml.safe_load(f).get("prepare") or {}))


def task_type(instruction: str, keywords: dict[str, str]) -> str:
    text = instruction.lower()
    for name, word in keywords.items():
        if word in text:
            return name
    raise ValueError(f"no task type matches the instruction {instruction!r}; add a keyword to the config")


def last_frame_time(video: str, work_fps: float) -> float:
    from cleantables.perception.frames import work_frame_count

    return round((work_frame_count(video) - 1) / work_fps, 1)


def make_episode(video: str, task: dict, task_name: str, margin_s: float, video_end_s: float) -> dict:
    return {
        "id": task["id"],
        "video": video,
        "task": task_name,
        "instruction": task["instruction"],
        "start_s": max(0.0, round(task["start_s"] - margin_s, 1)),
        "end_s": min(round(task["end_s"] + margin_s, 1), video_end_s),
        "core_start_s": task["start_s"],
        "core_end_s": task["end_s"],
        "objects": task["objects"],
        "destination": task["destination"],
        "complete": task["complete"],
        "note": task["note"],
    }


def prepare(raw: dict, cfg: PrepareConfig, video_end_s=None) -> dict:
    """raw: the content of episodes_raw.json. video_end_s: video -> last work-frame time (looked up when None)."""
    rows = {row["video"]: row for row in raw["videos"]}
    episodes = []
    for video in cfg.videos:
        end_s = video_end_s[video] if video_end_s else last_frame_time(video, cfg.work_fps)
        for task in rows[video]["tasks"]:
            name = task_type(task["instruction"], cfg.task_keywords)
            episodes.append(make_episode(video, task, name, cfg.margin_s, end_s))
    excluded = ", ".join(f"{video} ({why})" for video, why in cfg.excluded_videos.items())
    return {"source": SOURCE.format(margin=cfg.margin_s, excluded=excluded), "tasks": episodes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--raw", type=Path, default=paths.EPISODES_RAW)
    ap.add_argument("--out", type=Path, default=paths.EPISODES)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = prepare(json.loads(args.raw.read_text()), cfg)
    args.out.write_text(json.dumps(out, indent=1))
    counts = {}
    for e in out["tasks"]:
        counts[e["task"]] = counts.get(e["task"], 0) + 1
    print(f"{len(out['tasks'])} episodes from {len(cfg.videos)} videos: {counts}")


if __name__ == "__main__":
    main()
