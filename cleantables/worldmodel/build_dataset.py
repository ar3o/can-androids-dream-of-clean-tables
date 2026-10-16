"""Build the world-model dataset from the perception outputs of every episode in data/episodes.json.

Every tracked keyframe (5 fps) inside an episode's buffered window becomes a snapshot:
  - progress = moves of the labelled task finished by then / all of its tracked moves (time through the labelled core
    when the task has no tracked move); mover_set = objects that still move later in the episode;
  - when nothing is moving and a tracked move starts at least 0.3 s later inside the window, the labels of that
    next move (mover, destination, relation, end pose, object and hand path, future V-JEPA latent);
    otherwise mover = -1.

    python -m cleantables.worldmodel.build_dataset [--videos IMG_2516 IMG_2533] [--out data/worldmodel/dataset.npz]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from cleantables.paths import EPISODES, perception_dir

from .dataset import DATASET, KINDS, LEAVES, MAX_OBJECTS, RELATIONS, TABLE, blank_sample, resample_path, save_dataset
from .perception_io import (
    FPS,
    backproject,
    box_features,
    embed_texts,
    load_crop_embeddings,
    load_depth,
    load_events,
    load_geometry,
    load_tracks,
    load_vjepa,
    nearest_keyframe,
    table_extent,
    table_token_geometry,
    to_table,
)

log = logging.getLogger(__name__)
MIN_HAND_PIXELS = 60  # connected hand-mask component at 270x480
MIN_HAND_POINTS = 30
LEAD_FRAMES = 3  # a next move must start at least 0.3 s after the snapshot
CORE_MARGIN_FRAMES = 5  # moves within 0.5 s of the labelled core count towards the task's progress


def hand_tokens(geometry: dict, depth: np.ndarray, hand_video: dict) -> dict:
    """Frame -> (2,13) hand tokens from the SAM 3 hand masks and depth: up to two components sorted by x, each
    [valid, median centroid, 5-95 % extent, first two principal axes]."""
    out = {}
    frames, shape = hand_video["frames"], hand_video["shape"]
    masks = np.unpackbits(hand_video["masks"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    for t, k in enumerate(frames.tolist()):
        g = np.zeros((2, 13), np.float32)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(masks[t].astype(np.uint8), connectivity=8)
        comps = [
            i for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1 if stats[i, cv2.CC_STAT_AREA] >= MIN_HAND_PIXELS
        ][:2]
        hands = []
        depth_k = np.asarray(depth[k], np.float32) * geometry["scale"][k]
        for i in comps:
            P, _ = backproject(depth_k, geometry["K"], labels == i)
            if len(P) < MIN_HAND_POINTS:
                continue
            Q = to_table(geometry["T_all"][k], P)
            c = np.median(Q, 0)
            lo, hi = np.percentile(Q, 5, 0), np.percentile(Q, 95, 0)
            _, V = np.linalg.eigh(np.cov((Q - c).T))
            hands.append(np.r_[1.0, c, hi - lo, V[:, 2], V[:, 1]].astype(np.float32))
        hands.sort(key=lambda h: h[1])
        for j, h in enumerate(hands[:2]):
            g[j] = h
        out[k] = g
    return out


def hand_path(event: dict, path: np.ndarray, keyframes: list, hands: dict):
    """Path of the hand nearest the object during a move, resampled to 32 steps, and the steps it covers."""
    ks = [k for k in keyframes if event["f_start"] <= k <= event["f_end"] and k in hands]
    hp = np.zeros((32, 3), np.float32)
    ok = np.zeros(32, np.uint8)
    if len(ks) < 2:
        return hp, ok
    rp = resample_path(path)
    tt, pts = [], []
    for k in ks:
        h = hands[k]
        if h[:, 0].any():
            u = (k - event["f_start"]) / max(1, event["f_end"] - event["f_start"])
            po = np.array([np.interp(u, np.linspace(0, 1, 32), rp[:, i]) for i in range(3)])
            j = int(np.argmin([np.linalg.norm(h[i, 1:4] - po) if h[i, 0] else 1e9 for i in range(2)]))
            tt.append(u)
            pts.append(h[j, 1:4])
    if len(tt) >= 2:
        u = np.linspace(0, 1, 32)
        pts = np.array(pts)
        hp = np.stack([np.interp(u, tt, pts[:, i]) for i in range(3)], 1).astype(np.float32)
        ok = ((u >= min(tt)) & (u <= max(tt))).astype(np.uint8)
    return hp, ok


def _set_move_labels(s: dict, e: dict, dets: list, tok_obj: list, k: int, objdets: dict, hp, hok, vj) -> None:
    path = np.asarray(e["path"], np.float32)
    s["mover"][...] = dets.index(objdets[e["obj"]][k])
    if e["relation"] == "leaves":
        tg = LEAVES
    elif e["target"] is None:
        tg = TABLE
    else:
        tg = tok_obj.index(e["target"]) if e["target"] in tok_obj else -1
    s["target"][...] = tg
    s["relation"][...] = RELATIONS.get(e["relation"], -1)
    end = e["end"]
    if end and np.isfinite(end.get("yaw", np.nan)):
        s["end_pose"][:] = [end["x"], end["y"], end["top"], end["yaw"]]
        s["end_ok"][...] = 1
        off = e.get("offset")
        s["end_off"][:] = (
            off
            if off is not None
            else np.array([end["x"] - e["start"]["x"], end["y"] - e["start"]["y"], end["top"] - e["start"]["top"]])
        )
    s["path_obj"][:] = resample_path(path)
    s["path_hand"][:] = hp
    s["path_hand_ok"][:] = hok
    s["path_dt"][...] = (e["f_end"] - e["f_start"]) / FPS / 31
    s["prog_steps"][:] = np.linspace(0, 1, 32)
    s["time_to_move"][...] = (e["f_start"] - k) / FPS
    s["ev_id"][...] = e["id"]
    if vj is not None:
        b = nearest_keyframe(vj, e["f_end"])
        s["vj_future_glob"][:] = vj["glob"][b]
        s["vj_future_ok"][...] = 1


def _video_samples(video: str, tasks: list, stats: dict, phrases: set) -> list:
    ev = load_events(video)
    R = load_tracks(video)
    crops = load_crop_embeddings(video)
    geometry = load_geometry(video)
    depth = load_depth(video)
    ext = table_extent(geometry["T"], geometry["K"], geometry["table"])
    table_geo = table_token_geometry(ext[0], ext[1])
    try:
        vj = load_vjepa(video)
    except FileNotFoundError:
        vj = None
    try:
        hands = hand_tokens(geometry, depth, dict(np.load(perception_dir(video) / "hand_video.npz")))
    except FileNotFoundError:
        hands = {}
    src = {o["id"]: o["src"] for o in ev["objects"]}
    objdets = {
        oid: dict(zip(R["objects"][s]["sight_k"].tolist(), R["objects"][s]["det"], strict=False))
        for oid, s in src.items()
    }
    keyframes = sorted(R["kept"])
    events = ev["events"]
    if any("kind" in e for e in events):
        events = [e for e in events if e["kind"] == "move"]
    events = sorted(events, key=lambda e: e["f_start"])
    hand_paths = {}
    samples = []
    for t in tasks:
        k0, k1 = int(round(t["start_s"] * FPS)), int(round(t["end_s"] * FPS))
        c0, c1 = t["core_start_s"] * FPS, t["core_end_s"] * FPS
        st = stats.setdefault(t["id"], dict(task=t["task"], snaps=0, move_snaps=0, moves=0, no_tokens=0))
        ev_in = [e for e in events if k0 <= e["f_start"] <= k1]
        ev_core = [e for e in events if c0 - CORE_MARGIN_FRAMES <= e["f_start"] <= c1 + CORE_MARGIN_FRAMES]
        st["moves"] = len(ev_in)
        st["core_moves"] = len(ev_core)
        for k in keyframes:
            if k < k0 or k > k1:
                continue
            dets = [i for i in R["kept"][k] if R["det_box"].get(i) and i in crops]
            dets = sorted(dets, key=lambda i: -R["det_box"][i]["prob"])[: MAX_OBJECTS - 1]
            if not dets:
                st["no_tokens"] += 1
                continue
            owner = {dd[k]: oid for oid, dd in objdets.items() if k in dd}
            tok_obj = [owner.get(i, -1) for i in dets]
            s = blank_sample()
            n = len(dets)
            for j, i in enumerate(dets):
                box = R["det_box"][i]
                s["obj_geo"][j] = box_features(box)
                s["obj_crop"][j] = crops[i]
                s["obj_crop_ok"][j] = 1
                s["obj_mask"][j] = 1
                s["obj_kind"][j] = KINDS["movable"]
                s["obj_id"][j] = tok_obj[j]
                phrases.add(box["phrase"])
            s["_phrases"] = [R["det_box"][i]["phrase"] for i in dets] + ["table"]
            s["obj_geo"][n] = table_geo
            s["obj_mask"][n] = 1
            s["obj_kind"][n] = KINDS["table"]
            s["table_idx"][...] = n
            time_progress = float(np.clip((k - c0) / max(1.0, c1 - c0), 0, 1))
            s["prog_time"][...] = time_progress
            if ev_core:
                s["prog_video"][...] = sum(e["f_end"] <= k for e in ev_core) / len(ev_core)
            else:
                s["prog_video"][...] = time_progress
            s["prog_video_ok"][...] = 1
            remaining = {e["obj"] for e in ev_in if e["f_start"] >= k + LEAD_FRAMES}
            for j in range(n):
                s["mover_set"][j] = int(tok_obj[j] in remaining and tok_obj[j] != -1)
            s["time_to_move"][...] = np.nan
            if k in hands:
                s["hand_geo"][:] = hands[k]
            if vj is not None:
                a = nearest_keyframe(vj, k)
                s["vj_tok"][:] = vj["tok"][a]
                s["vj_tok_flip"][:] = vj["tok_f"][a]
                s["vj_ok"][...] = 1
            moving = any(e["f_start"] <= k <= e["f_end"] for e in ev_in)
            e = next((e for e in ev_in if e["f_start"] >= k + LEAD_FRAMES), None)
            if (
                not moving
                and e is not None
                and e.get("path")
                and e["obj"] in objdets
                and k in objdets[e["obj"]]
                and objdets[e["obj"]][k] in dets
            ):
                if e["id"] not in hand_paths:
                    hand_paths[e["id"]] = hand_path(e, np.asarray(e["path"], np.float32), keyframes, hands)
                _set_move_labels(s, e, dets, tok_obj, k, objdets, *hand_paths[e["id"]], vj)
                st["move_snaps"] += 1
            s.update(dataset=t["task"], video=video, group=t["id"], split="train", task_text=t["instruction"])
            s["snap_k"][...] = k
            samples.append(s)
            st["snaps"] += 1
    return samples


def build(out: Path = DATASET, videos: list | None = None) -> dict:
    tasks = json.load(open(EPISODES))["tasks"]
    samples, stats, phrases = [], {}, {"table"}
    for v in sorted({t["video"] for t in tasks}):
        if videos and v not in videos:
            continue
        if not (perception_dir(v) / "events.json").exists():
            log.info("video %s: no perception outputs, skipped", v)
            continue
        video_samples = _video_samples(v, [t for t in tasks if t["video"] == v], stats, phrases)
        samples += video_samples
        log.info("video %s: %d snapshots", v, len(video_samples))
    E = dict(zip(sorted(phrases), embed_texts(sorted(phrases)), strict=False))
    for s in samples:
        for j, p in enumerate(s.pop("_phrases")):
            s["obj_txt"][j] = E[p]
    meta = save_dataset(samples, out, dict(episodes=stats, source=str(EPISODES), events="events.json"))
    log.info("%d snapshots -> %s", meta["n"], out)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the world-model dataset from the perception outputs.")
    ap.add_argument("--videos", nargs="*", default=None, help="only these videos (default: all in data/episodes.json)")
    ap.add_argument("--out", type=Path, default=DATASET)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(a.out, a.videos)


if __name__ == "__main__":
    main()
