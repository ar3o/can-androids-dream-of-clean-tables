"""Rests, moves and end relations of every tracked object -> events.json. No task knowledge.

    python -m cleantables.perception.events VIDEO [VIDEO ...]

rest      >= 3 sightings spanning >= 0.8 s whose 3D box stays within 4.5 cm (xy and top) of their median; one outlier is
          skipped and unseen gaps do not break a rest; rests at the same place are merged
size      footprint = 75th percentile over sightings with little hand overlap (occlusion only ever shrinks it); height =
          top height in rests where nothing lies under the object, else its lowest top
move      two consecutive rests more than 8 cm apart in xy or 5 cm in top height; an object that disappears for good
          after its last rest has left the view ('leaves')
group     moves that overlap in time with the same displacement are one move of a stack; the lowest object carries it
relation  of the carrier's end pose against every other object at that moment:
            inside B     >= 60 % of A's footprint within B, A's top >= 2 cm below B's top, B's footprint >= 1.25x A's
            on_top_of B  footprint overlap >= 25 % of A and A's top >= B's top + 0.4 A's height (the highest such B)
            next_to B    rectangle gap < 6 cm (the nearest)
            table        otherwise
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from scipy.ndimage import median_filter

from cleantables import paths

from .files import WORK_FPS, n_frames, tracks, write_json
from .geometry import poly_area, poly_clip, rect_corners

T_REST = 8  # frames
D_MOVE, DZ_MOVE = 0.08, 0.05
REST_TOL = 0.045
HAND_OK = 0.15
REST_MAX_HAND = 0.3
NPATH = 16
LEAVE_MARGIN = 30  # frames before the video end / after the last sighting


def fill(p: np.ndarray):
    """Linearly interpolate the NaN rows of a (n, 3) trajectory; returns (filled, observed mask)."""
    ok = ~np.isnan(p[:, 0])
    f = np.arange(len(p))
    q = p.copy()
    if ok.sum() >= 2:
        for c in range(3):
            q[:, c] = np.interp(f, f[ok], p[ok, c])
    return q, ok


def close(b: dict, med) -> bool:
    return np.hypot(b["x"] - med[0], b["y"] - med[1]) < REST_TOL and abs(b["top"] - med[2]) < REST_TOL


def sight_rests(o: dict) -> list[tuple[int, int]]:
    sightings = [
        (int(k), b) for k, b in zip(o["sight_k"], o["boxes"], strict=True) if b and b["hand_frac"] < REST_MAX_HAND
    ]
    runs, cur, i = [], [], 0
    while i < len(sightings):
        k, b = sightings[i]
        if not cur:
            cur = [(k, b)]
            i += 1
            continue
        med = np.median([[c["x"], c["y"], c["top"]] for _, c in cur], 0)
        if close(b, med):
            cur.append((k, b))
            i += 1
        elif i + 1 < len(sightings) and close(sightings[i + 1][1], med):
            i += 1  # a single outlier (bad mask or swap)
        else:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if len(r) < 3 or r[-1][0] - r[0][0] < T_REST:
            continue
        med = np.median([[c["x"], c["y"], c["top"]] for _, c in r], 0)
        if out and np.hypot(*(out[-1][2][:2] - med[:2])) < D_MOVE / 2 and abs(out[-1][2][2] - med[2]) < DZ_MOVE:
            out[-1][1] = r[-1][0]
        else:
            out.append([r[0][0], r[-1][0], med])
    return [(a, b) for a, b, _ in out]


def clean_path(path: np.ndarray, p0: dict, p1: dict) -> np.ndarray:
    """Drop carry-path samples that cannot be the object (on the arm, off or below the table) and re-interpolate."""
    path = path.copy()
    zmax = max(p0["top"], p1["top"]) + 0.45
    bad = ~np.isfinite(path[:, 0]) | (path[:, 2] > zmax) | (path[:, 2] < -0.05)
    lo = np.minimum([p0["x"], p0["y"]], [p1["x"], p1["y"]]) - 0.25
    hi = np.maximum([p0["x"], p0["y"]], [p1["x"], p1["y"]]) + 0.25
    bad |= (path[:, :2] < lo).any(1) | (path[:, :2] > hi).any(1)
    bad[0] = bad[-1] = False
    path[0] = [p0["x"], p0["y"], p0["top"]]
    path[-1] = [p1["x"], p1["y"], p1["top"]]
    f = np.arange(len(path))
    for c in range(3):
        path[bad, c] = np.interp(f[bad], f[~bad], path[~bad, c])
    return median_filter(path, size=(3, 1), mode="nearest")


def circular_mean_pi(a) -> float:
    """Mean of angles defined modulo pi (rectangle yaw)."""
    a = np.asarray(a)
    return float((np.arctan2(np.sin(2 * a).mean(), np.cos(2 * a).mean()) / 2) % np.pi)


def pose_in(o: dict, f0: int, f1: int) -> dict:
    """Median 3D box of the object's sightings in frames f0..f1, sightings with little hand overlap preferred."""
    boxes = [b for k, b in zip(o["sight_k"], o["boxes"], strict=True) if b and f0 - 2 <= k <= f1 + 2]
    good = [b for b in boxes if b["hand_frac"] < HAND_OK] or boxes
    if not good:
        return dict(x=np.nan, y=np.nan, top=np.nan, yaw=np.nan, L=np.nan, W=np.nan, src="trajectory")
    return dict(
        x=float(np.median([b["x"] for b in good])),
        y=float(np.median([b["y"] for b in good])),
        top=float(np.median([b["top"] for b in good])),
        yaw=circular_mean_pi([b["yaw"] for b in good]),
        L=float(np.median([max(b["L"], b["W"]) for b in good])),
        W=float(np.median([min(b["L"], b["W"]) for b in good])),
        src=f"{len(good)} sightings",
    )


def rect(p: dict, size) -> np.ndarray:
    return rect_corners(p["x"], p["y"], size[0], size[1], p["yaw"] if np.isfinite(p["yaw"]) else 0.0)


def overlap(pa, sa, pb, sb) -> float:
    """area(A and B) / area(A)."""
    A, B = rect(pa, sa), rect(pb, sb)
    return poly_area(poly_clip(A, B)) / max(1e-9, poly_area(A))


def gap(pa, sa, pb, sb) -> float:
    A, B = rect(pa, sa), rect(pb, sb)
    if poly_area(poly_clip(A, B)) > 0:
        return 0.0
    d = min(np.min(np.linalg.norm(A[:, None] - B[None], axis=-1)), 9)
    for P, Q in ((A, B), (B, A)):
        for p in P:
            for i in range(4):
                a, b = Q[i], Q[(i + 1) % 4]
                t = np.clip((p - a) @ (b - a) / max(1e-12, (b - a) @ (b - a)), 0, 1)
                d = min(d, np.linalg.norm(p - (a + t * (b - a))))
    return float(d)


def relation(A: dict, size_a, others) -> tuple[str, int | None]:
    """A's end pose and size (L, W, H) against others [(id, pose, size)] -> (relation, target id or None)."""
    best = None
    for j, pB, sB in others:
        covered = overlap(A, size_a, pB, sB)
        if covered >= 0.6 and A["top"] <= pB["top"] - 0.02 and sB[0] * sB[1] > 1.25 * size_a[0] * size_a[1]:
            cand = ("inside", j, 3, -abs(A["top"] - pB["top"]))
        elif covered >= 0.25 and A["top"] >= pB["top"] + 0.4 * size_a[2]:
            cand = ("on_top_of", j, 2, pB["top"])
        else:
            continue
        if best is None or cand[2:] > best[2:]:
            best = cand
    if best:
        return best[0], best[1]
    gaps = [(gap(A, size_a, pB, sB), j) for j, pB, sB in others]
    if gaps and min(gaps)[0] < 0.06:
        return "next_to", min(gaps)[1]
    return "table", None


def object_sizes(O: list[dict]) -> None:
    for A in O:
        boxes = [b for b in A["o"]["boxes"] if b and b["hand_frac"] < HAND_OK] or [b for b in A["o"]["boxes"] if b]
        L = float(np.percentile([max(b["L"], b["W"]) for b in boxes], 75)) if boxes else 0.05
        W = float(np.percentile([min(b["L"], b["W"]) for b in boxes], 75)) if boxes else 0.05
        A["size"] = [L, W, 0.0]
    for A in O:
        free_tops = []
        for r in A["rests"]:
            p = r["pose"]
            if not np.isfinite(p["yaw"]):
                continue
            under = [
                B
                for B in O
                if B is not A
                and any(
                    rb["f0"] <= r["f1"]
                    and rb["f1"] >= r["f0"]
                    and np.isfinite(rb["pose"]["yaw"])
                    and overlap(p, A["size"], rb["pose"], B["size"]) > 0.2
                    and rb["pose"]["top"] < p["top"] - 0.01
                    for rb in B["rests"]
                )
            ]
            if not under:
                free_tops.append(p["top"])
        all_tops = [r["pose"]["top"] for r in A["rests"]] or [0.02]
        A["size"][2] = float(np.median(free_tops)) if free_tops else float(min(all_tops))
        A["size"] = [round(max(x, 0.005), 4) for x in A["size"]]


def pose_at(B: dict, f: int, n: int):
    """B's pose at frame f: the rest containing f, else the last one before (unless B has left), else its first rest."""
    rs = B["rests"]
    if not rs or f < B["first"] - 5:
        return None
    current = [r for r in rs if r["f0"] <= f <= r["f1"]]
    if current:
        return current[0]["pose"]
    before = [r for r in rs if r["f1"] < f]
    if before:
        if (
            B["last"] < n - LEAVE_MARGIN
            and f > B["last"] + LEAVE_MARGIN
            and before[-1] is rs[-1]
            and rs[-1]["f1"] < B["last"] - 3
        ):
            return None
        return before[-1]["pose"]
    return rs[0]["pose"]


def object_moves(A: dict, n: int) -> list[dict]:
    moves = []
    rs = A["rests"]
    for r0, r1 in zip(rs[:-1], rs[1:], strict=True):
        p0, p1 = r0["pose"], r1["pose"]
        if np.hypot(p0["x"] - p1["x"], p0["y"] - p1["y"]) < D_MOVE and abs(p0["top"] - p1["top"]) < DZ_MOVE:
            continue
        fs, fe = r0["f1"], r1["f0"]
        path = A["o"]["q"][fs : fe + 1].astype(float)
        if len(path) < 3:
            path = np.array([[p0["x"], p0["y"], p0["top"]], [p1["x"], p1["y"], p1["top"]]])
        else:
            # the box-centre trajectory, shifted to the start and end rest poses (linear blend), then cleaned
            s = np.linspace(0, 1, len(path))[:, None]
            off0 = np.array([p0["x"], p0["y"], p0["top"]]) - path[0]
            off1 = np.array([p1["x"], p1["y"], p1["top"]]) - path[-1]
            if np.all(np.isfinite(off0)) and np.all(np.isfinite(off1)):
                path = path + (1 - s) * off0 + s * off1
            path = clean_path(path, p0, p1)
        u = np.linspace(0, len(path) - 1, NPATH)
        resampled = np.stack([np.interp(u, np.arange(len(path)), path[:, c]) for c in range(3)], 1)
        moves.append(
            dict(obj=A["id"], f_start=int(fs), f_end=int(fe), start=p0, end=p1, path=resampled.tolist(), kind="move")
        )
    if rs and A["last"] < n - LEAVE_MARGIN and rs[-1]["f1"] < A["last"] - 3:
        moves.append(
            dict(
                obj=A["id"],
                f_start=int(rs[-1]["f1"]),
                f_end=int(A["last"]),
                start=rs[-1]["pose"],
                end=None,
                path=[],
                kind="leaves",
            )
        )
    return moves


def group_moves(moves: list[dict]) -> list[list[int]]:
    """Moves of different objects that overlap in time, start close together and share their displacement."""
    used, groups = set(), []
    for a, ma in enumerate(moves):
        if a in used:
            continue
        g = [a]
        for b in range(a + 1, len(moves)):
            mb = moves[b]
            if b in used or mb["obj"] == ma["obj"]:
                continue
            shared = min(ma["f_end"], mb["f_end"]) - max(ma["f_start"], mb["f_start"])
            if shared < 0.5 * min(ma["f_end"] - ma["f_start"], mb["f_end"] - mb["f_start"]) - 3:
                continue
            if ma["kind"] != mb["kind"]:
                continue
            if ma["kind"] == "move":
                da = np.array([ma["end"]["x"] - ma["start"]["x"], ma["end"]["y"] - ma["start"]["y"]])
                db = np.array([mb["end"]["x"] - mb["start"]["x"], mb["end"]["y"] - mb["start"]["y"]])
                if np.linalg.norm(da - db) > 0.10:
                    continue
            if np.hypot(ma["start"]["x"] - mb["start"]["x"], ma["start"]["y"] - mb["start"]["y"]) > 0.25:
                continue
            g.append(b)
        used |= set(g)
        groups.append(g)
    return groups


def make_event(eid: int, members: list[dict], O: list[dict], n: int) -> dict:
    carrier = min(members, key=lambda m: m["start"]["top"] - O[m["obj"]]["size"][2])
    A = O[carrier["obj"]]
    fs, fe = min(m["f_start"] for m in members), max(m["f_end"] for m in members)
    member_ids = {m["obj"] for m in members}
    others = []
    for B in O:
        if B["id"] in member_ids:
            continue
        pb = pose_at(B, fe + 5, n)
        if pb is not None and np.isfinite(pb["yaw"]):
            others.append((B["id"], pb, B["size"]))
    if carrier["kind"] == "leaves":
        rel, target = "leaves", None
    else:
        rel, target = relation(carrier["end"], A["size"], others)
    e = dict(
        id=eid,
        obj=A["id"],
        phrase=A["phrase"],
        members=sorted(member_ids),
        group=len(member_ids) > 1,
        kind=carrier["kind"],
        f_start=fs,
        f_end=fe,
        t_start=round(fs / WORK_FPS, 1),
        t_end=round(fe / WORK_FPS, 1),
        start=carrier["start"],
        end=carrier["end"],
        path=carrier["path"],
        relation=rel,
        target=target,
        target_phrase=O[target]["phrase"] if target is not None else ("leaves" if rel == "leaves" else "table"),
    )
    if target is not None:
        tp = next(p for j, p, _ in others if j == target)
        e["target_pose"] = tp
        e["offset"] = [carrier["end"]["x"] - tp["x"], carrier["end"]["y"] - tp["y"], carrier["end"]["top"] - tp["top"]]
    e["scene"] = {}  # every object's pose just before the move
    for B in O:
        pb = pose_at(B, max(0, fs - 3), n)
        if pb is not None and np.isfinite(pb["x"]):
            e["scene"][B["id"]] = pb
    return e


def run(video: str) -> dict:
    R = tracks(video)
    n = n_frames(video)
    O = []
    for oi, o in enumerate(R["objects"]):
        rests = [dict(f0=f0, f1=f1, pose=pose_in(o, f0, f1)) for f0, f1 in sight_rests(o)]
        rests = [r for r in rests if np.isfinite(r["pose"]["x"])]
        if not rests and len(o["sight_k"]) < 3:
            continue
        o["q"], o["ok"] = fill(o["P10"])
        O.append(
            dict(
                id=len(O),
                src=oi,
                phrase=o["phrase"],
                rgb=o["rgb"],
                rests=rests,
                last=int(o["sight_k"].max()),
                first=int(o["sight_k"].min()),
                o=o,
            )
        )
    object_sizes(O)
    moves = sorted([m for A in O for m in object_moves(A, n)], key=lambda m: m["f_start"])
    events = [make_event(i, [moves[j] for j in g], O, n) for i, g in enumerate(group_moves(moves))]
    objects = [
        dict(
            id=A["id"],
            src=A["src"],
            phrase=A["phrase"],
            rgb=A["rgb"],
            size=A["size"],
            first=A["first"],
            last=A["last"],
            rests=[dict(f0=r["f0"], f1=r["f1"], pose=r["pose"]) for r in A["rests"]],
            votes=A["o"]["votes"],
        )
        for A in O
    ]
    out = dict(
        video=video, n=n, objects=objects, events=events, params=dict(T_REST=T_REST, D_MOVE=D_MOVE, DZ_MOVE=DZ_MOVE)
    )
    write_json(out, paths.perception_dir(video) / "events.json")
    print(
        f"{video}: {len(O)} objects, {len(events)} moves ({sum(e['group'] for e in events)} group moves), relations "
        f"{dict(Counter(e['relation'] for e in events))}",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    for video in ap.parse_args().videos:
        run(video)


if __name__ == "__main__":
    main()
