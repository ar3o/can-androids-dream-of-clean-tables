"""Snapshot episodes: a stacking run read from two unoccluded moments instead of tracking every move.

The box videos are filmed from a tripod behind the person, whose back hides most of the build. The first seconds show every box on the
table and the last seconds show the finished stack from the side, close to the camera.
  1. Inventory (first INV_S s): det3d rows clustered over keyframes by xy (6 cm) and top (3 cm) -> one item per box with pose,
     footprint L x W, top height and median CIELab colour.
  2. Final stack (last FIN_S s): box masks near the tallest column, clustered into levels by mid-height (gap >= 2.5 cm).
  3. Match levels to items (Hungarian): cost = chroma distance (a, b) / 8 + 0.7 x |brightness rank difference|; exposure shifts near
     the camera but the brightness order holds. No height term (see match()).
  4. Episode: placements bottom -> top, each supported by the level below; pick / place times from the last sighting at the start pose
     and the first box top at the level's height within 25 cm of the build site. Carry = final stack > 15 cm from the build site.
The tracker (cleantables.perception.boxes.events) still provides the other moves as statistics.
  -> data/twin_table/boxstack/<V>/snapshot.json (episode in the common format + diagnostics)
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from cleantables.perception.scene.common import meta

from cleantables.perception.boxes import geometry as G
from cleantables.perception.boxes.tracking import circ_mean_pi
from cleantables.perception.boxes.common import jdump, log, vdir

INV_S, FIN_S = 3.0, 3.0
C = {c: i for i, c in enumerate(["k", "det", "prob", "x", "y", "top", "L", "W", "yaw", "ok", "touched", "n_top",
                                   "zlo", "zhi", "xa", "ya", "colL", "cola", "colb"])}


def cluster_items(R, min_frac=0.4):
    """Rows of several keyframes -> items (same box: xy within 6 cm and top within 3 cm)."""
    ks = sorted(set(R[:, 0].astype(int)))
    items = []
    for r in R[np.argsort(-R[:, C["prob"]])]:
        for it in items:
            if np.hypot(r[C["x"]] - it["x"], r[C["y"]] - it["y"]) < 0.06 and abs(r[C["top"]] - it["top"]) < 0.03:
                it["rows"].append(r)
                break
        else:
            items.append(dict(x=r[C["x"]], y=r[C["y"]], top=r[C["top"]], rows=[r]))
    out = []
    for it in items:
        A = np.array(it["rows"])
        nk = len(set(A[:, 0].astype(int)))
        if nk < max(2, min_frac * len(ks)):
            continue
        out.append(dict(x=float(np.median(A[:, C["x"]])), y=float(np.median(A[:, C["y"]])), top=float(np.median(A[:, C["top"]])),
                        L=float(np.median(A[:, C["L"]])), W=float(np.median(A[:, C["W"]])), yaw=circ_mean_pi(A[:, C["yaw"]]),
                        zlo=float(np.median(A[:, C["zlo"]])), col=np.median(A[:, C["colL"]:C["colb"] + 1], 0).tolist(), n_kf=nk))
    return out


def levels_of(R):
    """Final-stack rows -> levels bottom-to-top."""
    if not len(R):
        return []
    # the stack column: rows within 20 cm (xy) of the tallest row
    top = R[np.argmax(R[:, C["zhi"]])]
    R = R[np.hypot(R[:, C["xa"]] - top[C["xa"]], R[:, C["ya"]] - top[C["ya"]]) < 0.2]
    h = R[:, C["zhi"]] - R[:, C["zlo"]]
    R = R[(h > 0.02) & (h < 0.35)]
    if not len(R):
        return []
    zc = (R[:, C["zlo"]] + R[:, C["zhi"]]) / 2
    o = np.argsort(zc)
    groups, cur = [], [o[0]]
    for a, b in zip(o[:-1], o[1:]):
        if zc[b] - zc[a] > 0.025:
            groups.append(cur); cur = [b]
        else:
            cur.append(b)
    groups.append(cur)
    nk = len(set(R[:, 0].astype(int)))
    out = []
    for g in groups:
        A = R[g]
        if len(set(A[:, 0].astype(int))) < max(2, 0.4 * nk):
            continue
        out.append(dict(zlo=float(np.median(A[:, C["zlo"]])), zhi=float(np.median(A[:, C["zhi"]])), x=float(np.median(A[:, C["xa"]])),
                        y=float(np.median(A[:, C["ya"]])), tx=float(np.median(A[:, C["x"]])), ty=float(np.median(A[:, C["y"]])),
                        L=float(np.median(A[:, C["L"]])), W=float(np.median(A[:, C["W"]])), yaw=circ_mean_pi(A[:, C["yaw"]]),
                        col=np.median(A[:, C["colL"]:C["colb"] + 1], 0).tolist(), n=len(A)))
    # drop a 'level' that spans two others (a mask of two boxes together)
    keep = []
    for lv in out:
        inner = [q for q in out if q is not lv and q["zlo"] >= lv["zlo"] - 0.01 and q["zhi"] <= lv["zhi"] + 0.01]
        if len(inner) < 2:
            keep.append(lv)
    return keep


def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def match(items, levels):
    if not items or not levels:
        return {}
    ri = rank([it["col"][0] for it in items]); rl = rank([lv["col"][0] for lv in levels])
    ri = ri / max(1, len(items) - 1); rl = rl / max(1, len(levels) - 1)
    Cm = np.zeros((len(levels), len(items)))
    for a, lv in enumerate(levels):
        for b, it in enumerate(items):
            chroma = np.hypot(lv["col"][1] - it["col"][1], lv["col"][2] - it["col"][2])
            hl = lv["zhi"] - lv["zlo"]
            hi = it["top"] - max(0.0, it["zlo"] - 0.02)
            # Colour only: side-face height ranges are compressed (top faces dominate some masks) and the inventory H is unreliable
            # for boxes that start on others. Height sets the order instead (levels are sorted by z).
            Cm[a, b] = chroma / 8 + 0.7 * abs(rl[a] - ri[b]) * max(1, len(items) - 1) / 2
    ra, cb = linear_sum_assignment(Cm)
    return {int(a): (int(b), float(Cm[a, b])) for a, b in zip(ra, cb)}


def build(video, unseen=False):
    Z = np.load(vdir(video) / "det3d.npz")
    R = Z["rows"]
    if R.shape[1] < 19:
        raise RuntimeError("det3d.npz without appearance columns: re-run the geom stage")
    m = meta(video)
    fps, n = m["fps"], m["n"]
    R = R[(R[:, C["ok"]] > 0) & (R[:, C["top"]] >= 0.03) & (R[:, C["L"]] >= 0.08)]
    k_inv = int(INV_S * fps)
    Ri = R[R[:, 0] <= k_inv]
    items = cluster_items(Ri) if len(Ri) else []
    if len(items) < 3 and len(R):                            # arm in the way early: widen the window
        Ri = R[R[:, 0] <= 2 * k_inv]; items = cluster_items(Ri, 0.3)
    Rall = Z["rows"]
    Rf = Rall[(Rall[:, 0] >= n - 1 - FIN_S * fps)]
    levels = levels_of(Rf)
    mt = match(items, levels)
    diag = dict(video=video, n_items=len(items), n_levels=len(levels), match_cost=[round(mt[a][1], 2) for a in sorted(mt)],
                items=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in it.items()} for it in items],
                levels=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in lv.items()} for lv in levels])
    order = [(a, mt[a][0]) for a in range(len(levels)) if a in mt]
    ep = None
    if len(order) >= 2:
        boxes = {}
        for b, it in enumerate(items):
            lv = next((levels[a] for a, bb in order if bb == b), None)
            Hinv = it["top"] - max(0.0, it["zlo"] - 0.02) if it["zlo"] > 0.04 else it["top"]
            H = max(Hinv, (lv["zhi"] - lv["zlo"]) if lv else 0.0)
            boxes[f"b{b}"] = dict(L=it["L"], W=it["W"], H=float(H), init=dict(x=it["x"], y=it["y"], yaw=it["yaw"],
                                  z0=float(max(0.0, it["top"] - H))), color_lab=[round(c, 1) for c in it["col"]])
        bottom_item = items[order[0][1]]
        site = np.array([bottom_item["x"], bottom_item["y"]])
        sq = dict(x=float(site[0]), y=float(site[1]), yaw=bottom_item["yaw"], size=bottom_item["L"], src="inferred: bottom box start pose")
        placements = []
        # carry start = the last keyframe the bottom box is still seen at the build site
        Hb = boxes[f"b{order[0][1]}"]["H"]
        atb = R[(np.hypot(R[:, C["x"]] - site[0], R[:, C["y"]] - site[1]) < 0.08) & (np.abs(R[:, C["top"]] - Hb) < 0.03)]
        k_carry = int(atb[:, 0].max()) if len(atb) else n - 1
        z_acc = 0.0
        prev = None
        Rok = R
        for i, (a, b) in enumerate(order):
            lv = levels[a]; bid = f"b{b}"; B = boxes[bid]
            # pose at the build site: final-stack geometry moved back by the carry (bottom level -> build site)
            bl = levels[order[0][0]]
            dx, dy = site[0] - bl["x"], site[1] - bl["y"]
            top_lvl = i == len(order) - 1
            x = lv["x"] + dx                                  # the same measure (median of the visible points) for every level
            y = lv["y"] + dy
            if i == 0:
                x, y = site
            yaw = lv["yaw"] if i > 0 else bottom_item["yaw"]
            z0 = z_acc
            # times: pick = last keyframe the item is seen at its start pose; place = first keyframe a top at z0 + H shows at the site
            it = items[b]
            near0 = R[(np.hypot(R[:, C["x"]] - it["x"], R[:, C["y"]] - it["y"]) < 0.06) & (np.abs(R[:, C["top"]] - it["top"]) < 0.03)]
            k_pick = int(near0[:, 0].max()) if len(near0) else 0
            at = Rok[(np.hypot(Rok[:, C["x"]] - site[0], Rok[:, C["y"]] - site[1]) < 0.25) & (np.abs(Rok[:, C["top"]] - (z0 + B["H"])) < 0.035)
                     & (Rok[:, 0] >= k_pick)]
            k_place = int(at[:, 0].min()) if len(at) else k_pick + int(3 * fps)
            k_place = min(k_place, k_carry) if k_carry > k_pick else k_place
            if i == 0 and np.hypot(it["x"] - site[0], it["y"] - site[1]) < 0.05:
                k_pick = k_place = 0                          # the bottom box never moved: it already sat where the stack was built
            if prev is None:
                sup, off, rel = "square", [0.0, 0.0], 0.0
            else:
                ps = placements[-1]["pose"]
                c_, s_ = np.cos(ps["yaw"]), np.sin(ps["yaw"])
                ddx, ddy = x - ps["x"], y - ps["y"]
                sup, off, rel = prev, [float(c_ * ddx + s_ * ddy), float(-s_ * ddx + c_ * ddy)], G.wrap_pi2(yaw - ps["yaw"])
            placements.append(dict(box=bid, support=sup, kind="stack", k_pick=k_pick, k_place=max(k_place, k_pick), t_pick=round(k_pick / fps, 2),
                                   t_place=round(max(k_place, k_pick) / fps, 2), pose=dict(x=float(x), y=float(y), yaw=float(yaw), z0=float(z0)),
                                   offset=off, rel_yaw=float(rel), path=None, path_src="none (snapshot)", level=lv,
                                   match_cost=round(mt[a][1], 2)))
            z_acc += B["H"]
            prev = bid
        placements.sort(key=lambda p: (p["pose"]["z0"]))
        fl = levels[order[0][0]]
        carry_d = float(np.hypot(fl["x"] - site[0], fl["y"] - site[1]))
        ep = dict(id=f"{video}_r01", video=video, source="real", split="unseen" if unseen else "seen", reader="snapshot",
                  t_start=0.0, t_end=round(n / fps, 2), k_start=0, boxes=boxes, square=sq, spot_b=dict(x=fl["x"], y=fl["y"], yaw=fl["yaw"], src="final stack"),
                  placements=placements, final_order=[p["box"] for p in placements],
                  carry=dict(distance_m=round(carry_d, 3), t0=round(k_carry / fps, 2)) if carry_d > 0.15 else None, usable=len(placements) >= 2,
                  n_unmatched_levels=len(levels) - len(order), n_unstacked_items=len(items) - len(order))
    out = dict(diag=diag, episode=ep)
    jdump(out, vdir(video) / "snapshot.json")
    log(f"{video} snapshot: {len(items)} inventory boxes, {len(levels)} final-stack levels, order "
        f"{ep['final_order'] if ep else None}, carry {ep['carry'] if ep else None}")
    return out
