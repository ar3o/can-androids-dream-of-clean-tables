"""Generic events from the object tracks: rest / move segmentation, end relations from geometry, co-moving groups. No step types.

Per object:
  * rests     : from the 2 fps 3D box sightings (SAM 3 mask + depth): >= 3 consecutive sightings (>= 0.8 s) within 4.5 cm of their
                median (one outlier skipped, occlusion gaps allowed); rests at the same place are merged. Rest pose = median box of the
                sightings in it (low hand overlap first). The 10 fps CoTracker point trajectory only gives the carry path.
  * size      : footprint L x W = median over rest sightings with little hand overlap; height H = top height in the rests where the
                object stands on the table (no other footprint under it), else its lowest top.
  * moves     : consecutive rests > D_MOVE apart (xy) or > DZ_MOVE in top height. A track that disappears for good > 3 s before the
                video ends right after moving 'leaves' (left the view).
Co-moving: moves of several objects that overlap in time with the same displacement form one group move (carrier = lowest member).
End relation of a move, computed for any object A against every other object B (their poses at that moment):
  inside B     : A's footprint >= 60 % within B's, A's top >= 2 cm below B's top (sunk into B), B's footprint >= 1.25x A's
  on-top-of B  : footprints overlap >= 25 % of A and A's top >= B's top + 0.4 H_A (A sits above B); the highest such B
  next-to B    : rectangle gap < 6 cm (nearest)
  table        : otherwise;   leaves: out of view.
Each move also carries the VLM subtask label (cleantables.episodes.labels) at its mid time and its MiniLM embedding.
  $VLA_PY -m cleantables.perception.events VIDEOS...   -> data/generic/<V>/events.json
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
from scipy.ndimage import median_filter

from cleantables.perception.boxes.geometry import poly_area, poly_clip, rect_corners

from cleantables.perception.common import jdump, log, meta, names, vdir

V_REST, T_REST, D_MOVE, DZ_MOVE = 0.06, 8, 0.08, 0.05
HAND_OK = 0.15
NPATH = 16


def fill(p):
    ok = ~np.isnan(p[:, 0]); f = np.arange(len(p))
    q = p.copy()
    if ok.sum() >= 2:
        for c in range(3):
            q[:, c] = np.interp(f, f[ok], p[ok, c])
    return q, ok


def rests(p, ok):
    a, b = np.nonzero(ok)[0][[0, -1]]
    q = median_filter(p[a:b + 1], size=(5, 1), mode="nearest")
    sp = np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 10]
    sp = median_filter(sp, size=5)
    still = sp < V_REST
    out, i = [], 0
    while i < len(q):
        if still[i]:
            j = i
            while j < len(q) and still[j]:
                j += 1
            if j - i >= T_REST:
                out.append([a + i, a + j - 1, np.median(q[i:j], 0)])
            i = j
        else:
            i += 1
    merged = []
    for r in out:
        if merged and np.linalg.norm(merged[-1][2][:2] - r[2][:2]) < D_MOVE / 2 and abs(merged[-1][2][2] - r[2][2]) < DZ_MOVE:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged, q, a


def close(b, med, tol_xy=0.045, tol_z=0.045):
    return np.hypot(b["x"] - med[0], b["y"] - med[1]) < tol_xy and abs(b["top"] - med[2]) < tol_z


def sight_rests(o):
    """A rest is >= 3 consecutive 2 fps sightings (>= T_REST frames apart end to end) whose 3D box stays within 4.5 cm (xy and top) of the
    run's median; one outlier sighting is skipped if the next one is back in place; unseen gaps (occlusion) do not break a rest."""
    S = [(int(k), b) for k, b in zip(o["sight_k"], o["boxes"]) if b and b["hand_frac"] < 0.3]
    runs, cur, i = [], [], 0
    while i < len(S):
        k, b = S[i]
        if not cur:
            cur = [(k, b)]; i += 1; continue
        med = np.median([[c["x"], c["y"], c["top"]] for _, c in cur], 0)
        if close(b, med):
            cur.append((k, b)); i += 1
        elif i + 1 < len(S) and close(S[i + 1][1], med):
            i += 1                                           # single outlier (bad mask / swap)
        else:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if len(r) < 3 or r[-1][0] - r[0][0] < T_REST:
            continue
        med = np.median([[c["x"], c["y"], c["top"]] for _, c in r], 0)
        if out and np.hypot(*(out[-1][2][:2] - med[:2])) < D_MOVE / 2 and abs(out[-1][2][2] - med[2]) < DZ_MOVE:
            out[-1][1] = r[-1][0]                            # same place again: one rest
        else:
            out.append([r[0][0], r[-1][0], med])
    return [(a, b) for a, b, _ in out]


def clean_path(path, p0, p1):
    """Drop carry-path samples that cannot be the object (depth on the arm / off the table / below it) and re-interpolate."""
    path = path.copy()
    zmax = max(p0["top"], p1["top"]) + 0.45
    bad = ~np.isfinite(path[:, 0]) | (path[:, 2] > zmax) | (path[:, 2] < -0.05)
    lo = np.minimum([p0["x"], p0["y"]], [p1["x"], p1["y"]]) - 0.25; hi = np.maximum([p0["x"], p0["y"]], [p1["x"], p1["y"]]) + 0.25
    bad |= (path[:, :2] < lo).any(1) | (path[:, :2] > hi).any(1)
    bad[0] = bad[-1] = False
    path[0] = [p0["x"], p0["y"], p0["top"]]; path[-1] = [p1["x"], p1["y"], p1["top"]]
    f = np.arange(len(path))
    for c in range(3):
        path[bad, c] = np.interp(f[bad], f[~bad], path[~bad, c])
    return median_filter(path, size=(3, 1), mode="nearest")


def circ_mean_pi(a):
    a = np.asarray(a)
    return float((np.arctan2(np.sin(2 * a).mean(), np.cos(2 * a).mean()) / 2) % np.pi)


def pose_in(o, f0, f1, fallback):
    """Median 3D box of the object's sightings in frames f0..f1 (low hand overlap preferred)."""
    bs = [b for k, b in zip(o["sight_k"], o["boxes"]) if b and f0 - 2 <= k <= f1 + 2]
    good = [b for b in bs if b["hand_frac"] < HAND_OK] or bs
    if not good:
        return dict(x=float(fallback[0]), y=float(fallback[1]), top=float(max(fallback[2], 0.005)), yaw=np.nan, L=np.nan, W=np.nan,
                    src="trajectory")
    return dict(x=float(np.median([b["x"] for b in good])), y=float(np.median([b["y"] for b in good])),
                top=float(np.median([b["top"] for b in good])), yaw=circ_mean_pi([b["yaw"] for b in good]),
                L=float(np.median([max(b["L"], b["W"]) for b in good])), W=float(np.median([min(b["L"], b["W"]) for b in good])),
                src=f"{len(good)} sightings")


def rect(p, size):
    return rect_corners(p["x"], p["y"], size[0], size[1], p["yaw"] if np.isfinite(p["yaw"]) else 0.0)


def overlap(pa, sa, pb, sb):
    """area(A ∩ B) / area(A)."""
    A, B = rect(pa, sa), rect(pb, sb)
    return poly_area(poly_clip(A, B)) / max(1e-9, poly_area(A))


def gap(pa, sa, pb, sb):
    A, B = rect(pa, sa), rect(pb, sb)
    if poly_area(poly_clip(A, B)) > 0:
        return 0.0
    d = min(np.min(np.linalg.norm(A[:, None] - B[None], axis=-1)), 9)
    for P, Q in ((A, B), (B, A)):                      # point-to-segment distances
        for p in P:
            for i in range(4):
                a, b = Q[i], Q[(i + 1) % 4]
                t = np.clip((p - a) @ (b - a) / max(1e-12, (b - a) @ (b - a)), 0, 1)
                d = min(d, np.linalg.norm(p - (a + t * (b - a))))
    return float(d)


def relation(A, sA, others):
    """A pose + size (L, W, H); others: list of (id, pose, size). -> (relation, target id or None, details)."""
    best = None
    for j, pB, sB in others:
        ov_in = overlap(A, sA, pB, sB)
        if ov_in >= 0.6 and A["top"] <= pB["top"] - 0.02 and sB[0] * sB[1] > 1.25 * sA[0] * sA[1]:
            cand = ("inside", j, 3, -abs(A["top"] - pB["top"]))
        elif ov_in >= 0.25 and A["top"] >= pB["top"] + 0.4 * sA[2]:
            cand = ("on_top_of", j, 2, pB["top"])
        else:
            continue
        if best is None or cand[2:] > best[2:]:
            best = cand
    if best:
        return best[0], best[1]
    gs = [(gap(A, sA, pB, sB), j) for j, pB, sB in others]
    if gs and min(gs)[0] < 0.06:
        return "next_to", min(gs)[1]
    return "table", None


def run(v):
    R = pickle.load(open(vdir(v) / "tracks.pkl", "rb"))
    n = meta(v)["n"]
    objs = R["objects"]
    lab = None
    if (vdir(v) / "vlm_labels.json").exists():
        from cleantables.episodes.labels import label_at, labels
        rows, E = labels(v)
        lab = (rows, label_at)
    # ---- rests + poses per object (from the 2 fps 3D box sightings; the 10 fps point trajectory only shapes the carry path)
    O = []
    for oi, o in enumerate(objs):
        Rs = [dict(f0=f0, f1=f1, pose=pose_in(o, f0, f1, [np.nan] * 3)) for f0, f1 in sight_rests(o)]
        Rs = [r for r in Rs if np.isfinite(r["pose"]["x"])]
        if not Rs and len(o["sight_k"]) < 3:
            continue
        p, ok = fill(o["P10"])
        o["q"], o["ok"] = p, ok
        last = int(o["sight_k"].max())
        O.append(dict(id=len(O), src=oi, phrase=o["phrase"], rgb=o["rgb"], rests=Rs, last=last, first=int(o["sight_k"].min()), o=o))
    # ---- sizes
    for A in O:
        Ls = [b for b in A["o"]["boxes"] if b and b["hand_frac"] < HAND_OK] or [b for b in A["o"]["boxes"] if b]
        # 75th percentile rather than the median: occlusion (a box on top, the arm, the frame edge) only shrinks the visible footprint
        L = float(np.percentile([max(b["L"], b["W"]) for b in Ls], 75)) if Ls else 0.05
        Wd = float(np.percentile([min(b["L"], b["W"]) for b in Ls], 75)) if Ls else 0.05
        A["size"] = [L, Wd, 0.0]
    for A in O:
        tops_free = []
        for r in A["rests"]:
            p = r["pose"]
            if not np.isfinite(p["yaw"]):
                continue
            under = [B for B in O if B is not A and any(rb["f0"] <= r["f1"] and rb["f1"] >= r["f0"] and np.isfinite(rb["pose"]["yaw"])
                                                        and overlap(p, A["size"], rb["pose"], B["size"]) > 0.2 and rb["pose"]["top"] < p["top"] - 0.01
                                                        for rb in B["rests"])]
            if not under:
                tops_free.append(p["top"])
        alltops = [r["pose"]["top"] for r in A["rests"]] or [0.02]
        A["size"][2] = float(np.median(tops_free)) if tops_free else float(min(alltops))
        A["size"] = [round(max(x, 0.005), 4) for x in A["size"]]

    def pose_at(B, f):
        """B's pose at frame f: the rest containing f, else the last rest before it, else the first rest."""
        rs = B["rests"]
        if not rs or f < B["first"] - 5:
            return None
        cur = [r for r in rs if r["f0"] <= f <= r["f1"]]
        if cur:
            return cur[0]["pose"]
        before = [r for r in rs if r["f1"] < f]
        if before:
            if B["last"] < n - 30 and f > B["last"] + 30 and before[-1] is rs[-1] and rs[-1]["f1"] < B["last"] - 3:
                return None                                  # it has left
            return before[-1]["pose"]
        return rs[0]["pose"]

    # ---- moves
    moves = []
    for A in O:
        rs = A["rests"]
        for r0, r1 in zip(rs[:-1], rs[1:]):
            p0, p1 = r0["pose"], r1["pose"]
            if np.hypot(p0["x"] - p1["x"], p0["y"] - p1["y"]) < D_MOVE and abs(p0["top"] - p1["top"]) < DZ_MOVE:
                continue
            fs, fe = r0["f1"], r1["f0"]
            q = A["o"]["q"]
            path = q[fs:fe + 1].astype(float)
            if len(path) < 3:
                path = np.array([[p0["x"], p0["y"], p0["top"]], [p1["x"], p1["y"], p1["top"]]])
            else:
                # tracked point -> box top centre: shift by the start / end offsets (linear blend), then drop impossible samples
                s_ = np.linspace(0, 1, len(path))[:, None]
                off0 = np.array([p0["x"], p0["y"], p0["top"]]) - path[0]; off1 = np.array([p1["x"], p1["y"], p1["top"]]) - path[-1]
                if np.all(np.isfinite(off0)) and np.all(np.isfinite(off1)):
                    path = path + (1 - s_) * off0 + s_ * off1
                path = clean_path(path, p0, p1)
            pr = np.linspace(0, len(path) - 1, NPATH)
            path_r = np.stack([np.interp(pr, np.arange(len(path)), path[:, c]) for c in range(3)], 1)
            moves.append(dict(obj=A["id"], f_start=int(fs), f_end=int(fe), start=p0, end=p1, path=path_r.tolist(), kind="move"))
        if rs and A["last"] < n - 30 and rs[-1]["f1"] < A["last"] - 3:
            p0 = rs[-1]["pose"]
            moves.append(dict(obj=A["id"], f_start=int(rs[-1]["f1"]), f_end=int(A["last"]), start=p0, end=None, path=[], kind="leaves"))
    # ---- co-moving groups
    moves.sort(key=lambda m: m["f_start"])
    used, groups = set(), []
    for a, ma in enumerate(moves):
        if a in used:
            continue
        g = [a]
        for b in range(a + 1, len(moves)):
            mb = moves[b]
            if b in used or mb["obj"] == ma["obj"]:
                continue
            ov = min(ma["f_end"], mb["f_end"]) - max(ma["f_start"], mb["f_start"])
            if ov < 0.5 * min(ma["f_end"] - ma["f_start"], mb["f_end"] - mb["f_start"]) - 3:
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
    ev = []
    for g in groups:
        mem = [moves[i] for i in g]
        car = min(mem, key=lambda m: m["start"]["top"] - O[m["obj"]]["size"][2])        # lowest bottom = the carrier
        A = O[car["obj"]]
        fs, fe = min(m["f_start"] for m in mem), max(m["f_end"] for m in mem)
        others = []
        member_ids = {m["obj"] for m in mem}
        for B in O:
            if B["id"] in member_ids:
                continue
            pb = pose_at(B, fe + 5)
            if pb is not None and np.isfinite(pb["yaw"]):
                others.append((B["id"], pb, B["size"]))
        if car["kind"] == "leaves":
            rel, tgt = "leaves", None
        else:
            rel, tgt = relation(car["end"], A["size"], others)
        e = dict(id=len(ev), obj=A["id"], phrase=A["phrase"], members=sorted(member_ids), group=len(member_ids) > 1, kind=car["kind"],
                 f_start=fs, f_end=fe, t_start=round(fs / 10, 1), t_end=round(fe / 10, 1),
                 start=car["start"], end=car["end"], path=car["path"], relation=rel, target=tgt,
                 target_phrase=O[tgt]["phrase"] if tgt is not None else ("leaves" if rel == "leaves" else "table"))
        if tgt is not None:
            tp = next(p for j, p, _ in others if j == tgt)
            e["target_pose"] = tp
            e["offset"] = [car["end"]["x"] - tp["x"], car["end"]["y"] - tp["y"], car["end"]["top"] - tp["top"]]
        # scene just before the move: every object's pose (for the world model)
        e["scene"] = {}
        for B in O:
            pb = pose_at(B, max(0, fs - 3))
            if pb is not None and np.isfinite(pb["x"]):
                e["scene"][B["id"]] = pb
        if lab:
            e["subtask"] = lab[1](lab[0], (fs + fe) / 20, "subtask")
            e["activity"] = lab[1](lab[0], fs / 10, "activity")
        ev.append(e)
    objects = [dict(id=A["id"], src=A["src"], phrase=A["phrase"], rgb=A["rgb"], size=A["size"], first=A["first"], last=A["last"],
                    rests=[dict(f0=r["f0"], f1=r["f1"], pose=r["pose"]) for r in A["rests"]], votes=A["o"]["votes"]) for A in O]
    out = dict(video=v, n=n, objects=objects, events=ev,
               params=dict(V_REST=V_REST, T_REST=T_REST, D_MOVE=D_MOVE, DZ_MOVE=DZ_MOVE))
    jdump(out, vdir(v) / "events.json")
    from collections import Counter
    log(f"{v} events: {len(O)} objects, {len(ev)} moves ({sum(e['group'] for e in ev)} group), relations "
        f"{dict(Counter(e['relation'] for e in ev))}")
    return out


if __name__ == "__main__":
    for v in names(sys.argv[1:]):
        run(v)
