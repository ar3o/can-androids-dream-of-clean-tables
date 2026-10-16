"""Checks of the world model as a goal-setter in the twins, each episode scored by the fold models that never saw it.

    python -m cleantables.worldmodel.teacher_checks goals        # predicted goal vs the person's real end
    python -m cleantables.worldmodel.teacher_checks containers   # goal inside the labelled container

Results go to runs/worldmodel/<run>/teacher_goals.json and teacher_containers.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.paths import EPISODES, WM_RUNS

from .dataset import episode_folds, holdout_key
from .perception_io import FPS, load_events
from .physics_feedback import TWIN_TASKS, task_objects
from .teacher import WorldModelTeacher, load_scene, shuffle_layout

log = logging.getLogger(__name__)
MOVED_THRESHOLD = 0.05  # an object counts as moved by the person when its end is more than 5 cm from its start
CONTAINER_WALL = 0.01  # inside = closer to the container centre than half its short side minus this
GOAL_SAMPLES = 8


def heldout_teachers(run_dir: Path):
    """(task entry, teacher with the held-out fold ensemble) for every tools and fruit episode."""
    tasks = json.load(open(EPISODES))["tasks"]
    fold_of = episode_folds(holdout_key(t["video"], t["id"]) for t in tasks)
    for t in tasks:
        if t["task"] not in TWIN_TASKS:
            continue
        f = fold_of[holdout_key(t["video"], t["id"])]
        yield (
            t,
            WorldModelTeacher(
                t["id"], checkpoints=[run_dir / f"fold{f}_seed{s}.pt" for s in range(3)], samples=GOAL_SAMPLES
            ),
        )


def real_ends(episode: str) -> dict:
    """Twin object id -> top-centre end of its last tracked move in the labelled window (None if it did not move)."""
    episode, scene = load_scene(episode)
    events = load_events(scene["video"])
    task = {t["id"]: t for t in json.load(open(EPISODES))["tasks"]}[episode]
    k0, k1 = int(task["core_start_s"] * FPS) - 10, int(task["core_end_s"] * FPS) + 10
    src_to_id = {o["src"]: o["id"] for o in events["objects"]}
    out = {}
    for o in scene["objects"]:
        if o.get("track") is None or o["track"] < 0:
            continue
        oid = src_to_id.get(o["track"])
        moves = sorted(
            [
                e
                for e in events["events"]
                if e.get("kind", "move") == "move" and e.get("obj") == oid and e.get("end") and k0 <= e["f_start"] <= k1
            ],
            key=lambda e: e["f_start"],
        )
        out[o["id"]] = (
            np.array([moves[-1]["end"]["x"], moves[-1]["end"]["y"], moves[-1]["end"]["top"]]) if moves else None
        )
    return out


@torch.no_grad()
def predicted_ends(teacher: WorldModelTeacher, xyz: np.ndarray):
    """Per object, as if it were moved next: destination, end = mean end point of the anchored carries (top centre),
    and P(next) at this layout."""
    I = teacher.I
    b = teacher.batch(
        torch.as_tensor(xyz, dtype=torch.float32, device=teacher.dev),
        torch.zeros(xyz.shape[:2], device=teacher.dev),
        None,
    )
    Es = teacher.encode(b)
    p_next = []
    for m, E in zip(teacher.models, Es, strict=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mover_logits, _ = m.pointers(E, torch.zeros(1, dtype=torch.long, device=teacher.dev))
        p_next.append(F.softmax(mover_logits.float(), -1)[0, :I].cpu().numpy())
    ends, dest = np.zeros((I, 3)), np.zeros(I, int)
    for i in range(I):
        mover = torch.tensor([i], device=teacher.dev)
        d = int(teacher.destination(Es, mover))
        dest[i] = d
        target = torch.tensor([d if d < I else 24], device=teacher.dev)
        member_ends = []
        for m, E in zip(teacher.models, Es, strict=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, _, _ = m.heads_from(E, mover, target)
                member_ends.append(m.sample_paths_anchored(E, c, K=GOAL_SAMPLES).float()[0, :, -1, :3].mean(0))
        ends[i] = (torch.stack(member_ends).mean(0) + b["obj_geo"][0, i, :3]).cpu().numpy()
    return ends, dest, np.mean(p_next, 0)


def goal_check(run_dir: Path, seed: int = 0) -> list:
    torch.manual_seed(seed)
    rows = []
    for t, teacher in heldout_teachers(run_dir):
        Z = teacher.Z
        xyz = Z["start_pos"][None]
        ends, dest, p_next = predicted_ends(teacher, xyz)
        real = real_ends(t["id"])
        height = {o["id"]: o["extents"][2] for o in teacher.scene["objects"]}
        top0 = xyz[0].copy()
        top0[:, 2] += np.array([height[i] for i in Z["ids"]])
        for i, oid in enumerate(Z["ids"]):
            re = real.get(str(oid))
            moved = re is not None and np.linalg.norm(re[:2] - top0[i, :2]) > MOVED_THRESHOLD
            rows.append(
                dict(
                    ep=t["id"],
                    task=t["task"],
                    obj=str(oid),
                    role=str(Z["role"][i]),
                    moved=bool(moved),
                    p_next=float(p_next[i]),
                    wm_dest=str(Z["ids"][dest[i]]) if dest[i] < len(Z["ids"]) else "table",
                    wm_disp=float(np.linalg.norm(ends[i, :2] - top0[i, :2])),
                    err_xy=float(np.linalg.norm(ends[i, :2] - re[:2])) if moved else None,
                    stay_err=float(np.linalg.norm(top0[i, :2] - re[:2])) if moved else None,
                )
            )
        log.info("episode %s done", t["id"])
    json.dump(rows, open(run_dir / "teacher_goals.json", "w"), indent=1)
    for task in TWIN_TASKS:
        moved = [r for r in rows if r["task"] == task and r["moved"]]
        if moved:
            err = np.array([r["err_xy"] for r in moved])
            log.info(
                "%s: %d moved objects, goal vs real end median %.1f cm (left in place %.1f cm); destinations %s",
                task,
                len(moved),
                100 * np.median(err),
                100 * np.median([r["stay_err"] for r in moved]),
                dict(Counter(r["wm_dest"] for r in moved).most_common(5)),
            )
    return rows


@torch.no_grad()
def container_check(run_dir: Path, n_layouts: int = 20, seed: int = 0) -> list:
    """Every in-scope object's goal at the start state, all at once (as the robot teacher sets them at reset)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, teacher in heldout_teachers(run_dir):
        Z, I = teacher.Z, teacher.I
        ids = [str(x) for x in Z["ids"]]
        task = task_objects(Z)
        objects = {str(o["id"]): o for o in teacher.scene["objects"]}
        for layout in range(n_layouts):
            P0 = Z["start_pos"].copy() if layout == 0 else shuffle_layout(Z, teacher.scene, rng, task)
            if P0 is None:
                continue
            b = teacher.batch(torch.tensor(P0[None], device=teacher.dev), torch.zeros(1, I, device=teacher.dev), None)
            Es = teacher.encode(b)
            ok, dists = [], []
            for i in task:
                mover = torch.tensor([i], device=teacher.dev)
                end = teacher.end_point(b, Es, mover, teacher.destination(Es, mover))[0].cpu().numpy()
                container = objects.get(str(Z["dest_id"][i]))
                cp = P0[ids.index(str(Z["dest_id"][i]))] if str(Z["dest_id"][i]) in ids else None
                r_in = 0.5 * min(container["extents"][0], container["extents"][1]) - CONTAINER_WALL if container else 0
                dist = float(np.hypot(*(end[:2] - cp[:2]))) if cp is not None else 9.0
                ok.append(dist < r_in)
                dists.append(dist)
            rows.append(
                dict(
                    ep=t["id"],
                    task=t["task"],
                    layout="measured" if layout == 0 else "random",
                    n=len(task),
                    ok=int(sum(ok)),
                    all_ok=bool(all(ok)),
                    d=dists,
                )
            )
        log.info("episode %s done", t["id"])
    json.dump(rows, open(run_dir / "teacher_containers.json", "w"), indent=1)
    for task in TWIN_TASKS:
        for kind in ("measured", "random"):
            R = [r for r in rows if r["task"] == task and r["layout"] == kind]
            if R:
                log.info(
                    "%s %s layouts %d: goals inside the correct container %.2f, every object correct %.2f, "
                    "goal to container centre median %.1f cm",
                    task,
                    kind,
                    len(R),
                    sum(r["ok"] for r in R) / sum(r["n"] for r in R),
                    np.mean([r["all_ok"] for r in R]),
                    100 * np.median(np.concatenate([r["d"] for r in R])),
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Goal-setter checks of the world model on held-out episodes and twins.")
    ap.add_argument("check", choices=["goals", "containers"])
    ap.add_argument("--run", default="imitation", help="run whose fold checkpoints are used")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    (goal_check if a.check == "goals" else container_check)(WM_RUNS / a.run)


if __name__ == "__main__":
    main()
