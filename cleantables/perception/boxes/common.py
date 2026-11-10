"""Shared paths and constants for the box-stacking pipeline. Frames use the table_clean layout
(data/table_clean/_work/<VIDEO>/frames/<k>.jpg, 10 fps, 960x540) so the depth and CoTracker helpers apply unchanged; box-specific
outputs go to data/twin_table/boxstack/<VIDEO>/."""
from __future__ import annotations

import os
import sys
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/twin_table/boxstack"
VIDS = ROOT / "videos/twin_table/boxstack"
LOG = ROOT / "runs/twin_table/boxstack/PROGRESS.md"
SAM_PY = os.environ.get("SAM3_PY", sys.executable)
SAM_PATH = os.environ.get("SAM3_REPO", "third_party/sam3")
VLA_PY = str(ROOT / "envs/vla/bin/python")
MOGE_PATH = "third_party/MoGe_latest:cache/pylibs_moge3:cache/pylibs:."

# SAM 3 phrases for boxes. 'hand' is prompted too: its masks are used to drop box detections that are mostly hand and to tell
# when a hand is near a box.
BOX_PHRASES = ["cardboard box", "box"]
HAND_PHRASES = ["hand"]
TAPE_PHRASES = ["masking tape"]
# Kitting: small parts / tools packed into the kit box before stacking (separate SAM 3 pass -> sam3_tools.npz).
TOOL_PHRASES = ["screwdriver", "wrench", "pliers", "roll of tape", "glue bottle", "spray can", "battery pack", "box of screws", "marker pen", "utility knife", "torch", "hammer", "tool", "small can"]


def vdir(video):
    d = OUT / video
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"\n## {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    print(msg, flush=True)


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
