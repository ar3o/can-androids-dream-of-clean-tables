"""Contact sheets of all twins: the real | twin side-by-side still of every episode, one sheet per task, for a visual
check.

  python -m cleantables.twin.contact_sheet
Out data/twins/contact_<task>.jpg
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from cleantables.paths import TWINS, twin_dir
from cleantables.twin.inputs import episodes

TASKS = ("tools", "fruit", "stack_boxes")
TILE_W, TILE_H = 800, 225
COLUMNS = 2


def tile(episode_id: str) -> np.ndarray:
    d = twin_dir(episode_id)
    im = cv2.imread(str(d / "side_by_side.jpg"))
    if im is None:
        im = np.zeros((540, 1920, 3), np.uint8)
        cv2.putText(im, "not rendered", (700, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    t = cv2.resize(im, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    S = json.loads((d / "scene.json").read_text()) if (d / "scene.json").exists() else {}
    arms = S.get("arms", {})
    label = (
        f"{episode_id}  {len(S.get('objects', []))} obj  flags {len(S.get('flags', []))}  arms {arms.get('side', '?')} "
        f"reach {arms.get('tasks_one_arm', '?')}"
    )
    cv2.rectangle(t, (0, TILE_H - 22), (TILE_W, TILE_H), (0, 0, 0), -1)
    cv2.putText(t, label, (6, TILE_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main() -> None:
    argparse.ArgumentParser(description="One contact sheet of real | twin stills per task.").parse_args()
    blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    for task in TASKS:
        tiles = [tile(e["id"]) for e in episodes(task)]
        if not tiles:
            continue
        rows = [
            np.hstack(tiles[i : i + COLUMNS] + [blank] * (COLUMNS - len(tiles[i : i + COLUMNS])))
            for i in range(0, len(tiles), COLUMNS)
        ]
        out = TWINS / f"contact_{task}.jpg"
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{task}: {len(tiles)} episodes -> {out}", flush=True)


if __name__ == "__main__":
    main()
