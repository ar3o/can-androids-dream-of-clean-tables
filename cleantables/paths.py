"""Filesystem layout shared by every stage.

All inputs and outputs live under three roots, each overridable with an environment variable:

    data/         CLEANTABLES_DATA         videos, episode labels, perception outputs, twins, datasets
    runs/         CLEANTABLES_RUNS         trained models, logs and evaluation results
    checkpoints/  CLEANTABLES_CHECKPOINTS  third-party weights
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CLEANTABLES_DATA", REPO / "data"))
RUNS = Path(os.environ.get("CLEANTABLES_RUNS", REPO / "runs"))
CHECKPOINTS = Path(os.environ.get("CLEANTABLES_CHECKPOINTS", REPO / "checkpoints"))

VIDEOS = DATA / "videos"  # <video>.MOV, as recorded
EPISODES_RAW = DATA / "episodes_raw.json"  # segments returned by the video-language model
EPISODES = DATA / "episodes.json"  # the same segments with a margin, used by every later stage
PERCEPTION = DATA / "perception"  # <video>/: depth, table frame, object tracks, video features
TWINS = DATA / "twins"  # <episode>/: one Isaac Sim scene per episode; library/: shared meshes
WM_DATA = DATA / "worldmodel"  # world-model training snapshots

WM_RUNS = RUNS / "worldmodel"
RL_RUNS = RUNS / "rl"


def video_path(video: str) -> Path:
    for ext in (".MOV", ".mov", ".mp4"):
        p = VIDEOS / f"{video}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no video named {video} in {VIDEOS}")


def perception_dir(video: str) -> Path:
    return PERCEPTION / video


def twin_dir(episode: str) -> Path:
    return TWINS / episode
