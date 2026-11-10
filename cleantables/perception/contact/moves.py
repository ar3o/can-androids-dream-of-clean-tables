"""Moves from hand contact: a move is one HOIST-Former contact interval (grasp -> release; published model at its own 0.5 score,
read from events_contact.json). These replace the threshold-based moves of cleantables.perception.events.
The object's path and end come from the tracker over that interval: path = its tracked boxes from grasp to release, end = its tracked
box at the first keyframe at / after the release. No relation and no destination label: the world model learns where things end from
the end position alone.
Also the self-supervised targets: every object's tracked displacement at +0.5 / 1 / 2 / 4 s.
  intervals(v)      -> [event-like dicts]: id, obj (events.json object id), f_start, f_end (keyframe units, 0.1 s), path, start, end, hand
  fut_disp(v, ...)  -> per token (24, 4, 3) displacement (m) + ok (24, 4)
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

from cleantables.perception.common import vdir

from cleantables.worldmodel.model import FUT_S

_CACHE = {}


def _tracks(v):
    if v not in _CACHE:
        R = pickle.load(open(vdir(v) / "tracks.pkl", "rb")); Ev = json.load(open(vdir(v) / "events.json"))
        src = {int(o["id"]): int(o["src"]) for o in Ev["objects"]}
        per = {}
        for oid, s in src.items():
            ob = R["objects"][s]
            kb = [(int(k), b) for k, b in zip(ob["sight_k"], ob["boxes"]) if b is not None]           # sightings with a box
            per[oid] = (np.asarray([k for k, _ in kb], int), [(float(b["x"]), float(b["y"]), float(b["top"]), float(b["yaw"]), float(b["L"]),
                                                              float(b["W"])) for _, b in kb])
        _CACHE[v] = (R, per, sorted(R["kept"]))
    return _CACHE[v]


def pose_at(v, oid, k, side="after", max_gap=10):
    """The object's tracked box at keyframe k (side 'after': the first sighting at / after k; 'before': the last at / before k), within
    max_gap keyframe units (1 s), else None. -> (x, y, top, yaw, L, W)."""
    _, per, _ = _tracks(v)
    if oid not in per:
        return None
    sk, bx = per[oid]
    if not len(sk):
        return None
    i = np.searchsorted(sk, k, side="left") if side == "after" else np.searchsorted(sk, k, side="right") - 1
    if i < 0 or i >= len(sk) or abs(int(sk[i]) - k) > max_gap:
        return None
    return bx[i]


def _read(v):
    """Read data/generic/<V>/events_contact.json:
    obj = the tracks.pkl index; kind move | touch | undecided | leaves (only 'move' is a move); f_start / f_end = the first / last
    HOIST-held keyframe; start / end = the last / first pose outside the contact (FoundationPose top centre where tracked, else the
    tracker), null at a coverage edge; path = 32 points from start to end; hoist_coverage = the frames HOIST scored per episode."""
    f = vdir(v) / "events_contact.json"
    return json.load(open(f)) if f.exists() else None


def intervals(v):
    """-> (moves: event-like dicts sorted by grasp, with obj = the events.json object id the tokens use; cover: [(f_start, f_end)] of
    every move / leaves / undecided contact (k inside one = "something is held": no next-move label there); coverage {episode: [k0, k1]})
    or None if this video has no events_contact.json."""
    J = _read(v)
    if J is None:
        return None
    Ev = json.load(open(vdir(v) / "events.json"))
    ev_id = {int(o["src"]): int(o["id"]) for o in Ev["objects"]}                # tracks.pkl index -> events.json id (token obj_id)
    moves, cover = [], []
    for e in sorted(J["events"], key=lambda x: (int(x["f_start"]), int(x["id"]))):
        if e["kind"] in ("move", "leaves", "undecided"):
            cover.append((int(e["f_start"]), int(e["f_end"])))
        if e["kind"] != "move" or int(e["obj"]) not in ev_id:
            continue
        oid = ev_id[int(e["obj"])]
        st, en = e.get("start"), e.get("end")
        pe = pose_at(v, oid, int(en["k"]), "after") if en else None                  # end yaw from the tracker's box (positions from FoundationPose / tracker)
        dct = lambda p, q: None if p is None else dict(x=p["x"], y=p["y"], top=p["top"], yaw=(q[3] if q else float("nan")), L=(q[4] if q else 0.0),
                                                        W=(q[5] if q else 0.0))      # noqa: E731
        moves.append(dict(id=int(e["id"]), obj=oid, f_start=int(e["f_start"]), f_end=int(e["f_end"]), t_start=e["t_start"], t_end=e["t_end"],
                          path=(e["path"] if (st is not None and en is not None and e.get("path")) else None), start=dct(st, None), end=dct(en, pe),
                          offset=None, relation="none", target=None, episode=e.get("episode"), hand=e.get("hand"), source="contact"))
    return moves, cover, J.get("hoist_coverage", {})


def fut_disp(v, k, oid_row, mask_row):
    """Displacement targets for one example at keyframe k: each token's tracked displacement (m) at k + FUT_S (the nearest sighting within
    0.3 s of each horizon), relative to its own tracked position at k. -> (disp (24, 4, 3) float32, ok (24, 4) bool)."""
    n = len(oid_row)
    D = np.zeros((n, len(FUT_S), 3), np.float32); ok = np.zeros((n, len(FUT_S)), bool)
    for j in range(n):
        if not mask_row[j] or oid_row[j] < 0:
            continue
        p0 = pose_at(v, int(oid_row[j]), k, "after", max_gap=3)
        if p0 is None:
            continue
        for h, s in enumerate(FUT_S):
            p1 = pose_at(v, int(oid_row[j]), k + int(round(10 * s)), "after", max_gap=3)
            if p1 is not None:
                D[j, h] = np.subtract(p1[:3], p0[:3]); ok[j, h] = True
    return D, ok
