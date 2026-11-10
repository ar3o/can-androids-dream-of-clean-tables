"""Identity of generic-pipeline tracks -> physical library objects (one mesh per physical object, shared across episodes and tasks).

A track (data/generic/<V>/events.json object = tracks.pkl object) is mapped by its SAM 3 phrase, colour and measured footprint. The rules
below are written for the objects that occur in the recorded episodes (listed in LIBRARY); every decision records its reason, and ambiguous
cases are returned with `uncertain` set instead of being guessed silently. The two pliers are told apart later by shape (render-and-compare
of both library silhouettes against the episode mask, cleantables.twin.ep.build), not here.
"""
from __future__ import annotations

import numpy as np

from cleantables.twin.ep.common import lab

# key: class, category, kind, physical description (mass kg, friction, thickness for flat objects), destination container
LIBRARY = {
    # tools task (IMG_2533-2548)
    "pliers_a": dict(cls="tool", category="tool/pliers", kind="flat", thick=0.014, mass=0.16, mu=(0.6, 0.5), dest="toolbox"),
    "pliers_b": dict(cls="tool", category="tool/pliers", kind="flat", thick=0.014, mass=0.16, mu=(0.6, 0.5), dest="toolbox"),
    "wrench": dict(cls="tool", category="tool/wrench", kind="flat", thick=0.010, mass=0.12, mu=(0.6, 0.5), dest="toolbox"),
    "utility_knife": dict(cls="tool", category="tool/utility_knife", kind="flat", thick=0.018, mass=0.10, mu=(0.6, 0.5), dest="toolbox"),
    "screwdriver": dict(cls="tool", category="tool/screwdriver", kind="flat", thick=0.020, mass=0.06, mu=(0.6, 0.5), dest="toolbox"),
    "toolbox": dict(cls="container", category="container/toolbox", kind="tray", mass=0.35, mu=(0.7, 0.6)),
    "toolbox_lid": dict(cls="obstacle", category="box/toolbox_lid", kind="cuboid", mass=0.08, mu=(0.7, 0.6)),
    # boxes (tools scenes: distractors; stacking task: the items)
    "amazon_box": dict(cls="box", category="box/amazon", kind="cuboid", mass=0.45, mu=(0.7, 0.6)),
    "cipriata_box": dict(cls="box", category="box/cipriata", kind="cuboid", mass=0.35, mu=(0.7, 0.6)),
    "lotus_box": dict(cls="box", category="box/lotus", kind="cuboid", mass=0.30, mu=(0.7, 0.6)),
    "black_box": dict(cls="box", category="box/black", kind="cuboid", mass=0.20, mu=(0.7, 0.6)),
    "orange_box": dict(cls="box", category="box/orange", kind="cuboid", mass=0.12, mu=(0.7, 0.6)),
    "yellow_box": dict(cls="box", category="box/yellow", kind="cuboid", mass=0.15, mu=(0.7, 0.6)),
    # fruit task (IMG_2516)
    "fruit_bowl": dict(cls="container", category="container/fruit_bowl", kind="hy3d", mass=0.6, mu=(0.6, 0.5)),
    "banana": dict(cls="fruit", category="fruit/banana", kind="hy3d", mass=0.12, mu=(0.7, 0.6), dest="fruit_bowl"),
    "banana_a": dict(cls="fruit", category="fruit/banana", kind="hy3d", mass=0.12, mu=(0.7, 0.6), dest="fruit_bowl"),
    "banana_b": dict(cls="fruit", category="fruit/banana", kind="hy3d", mass=0.12, mu=(0.7, 0.6), dest="fruit_bowl"),
    "green_apple": dict(cls="fruit", category="fruit/apple", kind="apple", mass=0.15, mu=(0.7, 0.6), dest="fruit_bowl"),
    "ginger": dict(cls="fruit", category="fruit/ginger", kind="pillow", thick=0.022, mass=0.05, mu=(0.7, 0.6), dest="fruit_bowl"),
}

TOOL_PHRASES = {"plier": "pliers", "wrench": "wrench", "yellow tool": "utility_knife", "utility knife": "utility_knife", "knife": "utility_knife",
                "screwdriver": "screwdriver"}


def classify(o, others=()):
    """o: events.json object (phrase, rgb, size [L, W, top]). -> (key or None, reason, uncertain)."""
    ph = o["phrase"].lower()
    L, Wd, top = o["size"]
    r, g, b = o["rgb"]
    Ll, a, bb = lab(o["rgb"])
    for k, v in TOOL_PHRASES.items():
        if k in ph:
            return v, f"phrase '{o['phrase']}'", False
    if "banana" in ph:
        return "banana", "phrase", False
    if "apple" in ph:
        return "green_apple", "phrase", False
    if "ginger" in ph:
        return "ginger", "phrase", False
    if "bowl" in ph:
        return "fruit_bowl", "phrase", False
    if any(k in ph for k in ("tape", "label")):
        return None, f"part / junk phrase '{o['phrase']}'", False
    if Wd < 0.06 or max(L, Wd) < 0.12:
        return None, f"too thin / small for a box ({L:.2f} x {Wd:.2f})", False
    sat = max(r, g, b) - min(r, g, b)
    if min(r, g, b) >= 235 and sat <= 25:
        return "white", "white (toolbox tray or lid; resolved per video by height)", False
    if r - b >= 90 and r - g >= 45 and max(L, Wd) < 0.26:
        return "orange_box", f"orange rgb {o['rgb']}", False
    if b - r >= 15 and b >= g:
        return "lotus_box", f"navy rgb {o['rgb']}", False
    if L >= 0.345 and Wd >= 0.24:
        return "amazon_box", f"largest footprint {L:.2f} x {Wd:.2f}", False
    if r >= 240 and g >= 195 and b <= 120:
        return "yellow_box", f"yellow rgb {o['rgb']}", L > 0.30           # 2555: footprint measured 0.35 (probably merged with a neighbour)
    if max(r, g, b) <= 80 and sat <= 25:
        return "black_box", f"black rgb {o['rgb']}", False
    if max(L, Wd) < 0.24 and r - b >= 25 and r >= g:
        return "orange_box", f"small reddish box rgb {o['rgb']} (over-exposed orange)", True
    if Ll < 60:
        return "black_box", f"dark L*={Ll:.0f}", True
    return "cipriata_box", f"remaining grey-brown medium box rgb {o['rgb']} {L:.2f} x {Wd:.2f}", not (0.24 <= L <= 0.36)


def white_saturation(v, o, n=7):
    """Median over a few sightings of the 90th-percentile HSV saturation inside the track's mask: the open toolbox tray shows its gold
    outer walls (>= 100), the closed box / the lid are white (< 100)."""
    import cv2
    from cleantables.twin.ep.common import Dets, frame, tracks
    D, R = Dets(v), tracks(v)
    T = R["objects"][o["src"]]
    pairs = list(zip(T["sight_k"].tolist(), T["det"]))
    vals = []
    for k, d in pairs[::max(1, len(pairs) // n)]:
        m = cv2.resize(D.mask(d).astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.sum() < 50:
            continue
        vals.append(float(np.percentile(cv2.cvtColor(frame(v, k), cv2.COLOR_BGR2HSV)[..., 1][m], 90)))
    return float(np.median(vals)) if vals else 0.0


def resolve_video(objs, v=None):
    """Classify every track of a video. White toolbox-footprint tracks: saturated (gold walls visible) = the open toolbox tray ('toolbox');
    white = the closed toolbox or its lid ('white_box', resolved per episode: the tray is placed there when no open tray is present, else it
    is the lid). Without `v` (no frames) every toolbox-like white track is 'white_box'."""
    res = []
    for o in objs:
        k, why, unc = classify(o)
        res.append(dict(track=o["id"], key=k, why=why, uncertain=unc))
    byid = {o["id"]: o for o in objs}
    dd = lambda r: abs(byid[r["track"]]["size"][0] - 0.26) + abs(byid[r["track"]]["size"][1] - 0.155)   # noqa: E731  toolbox footprint
    for r in res:
        if r["key"] != "white":
            continue
        if dd(r) >= 0.07:
            r["key"], r["why"], r["uncertain"] = None, r["why"] + f" -> white but not toolbox-like ({byid[r['track']]['size']}): dropped", True
            continue
        sat = white_saturation(v, byid[r["track"]]) if v else 0.0
        r["sat"] = sat
        if sat >= 100:
            r["key"], r["why"] = "toolbox", r["why"] + f" -> gold walls visible (sat {sat:.0f}) = open toolbox tray"
        else:
            r["key"], r["why"] = "white_box", r["why"] + f" -> white (sat {sat:.0f}) = closed toolbox or lid"
    return res
