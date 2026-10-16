"""Which physical object a perception track is: SAM 3 phrase, colour and measured footprint -> library key.

The catalogue lists the objects of the recordings. Ambiguous cases come back with `uncertain` set instead of a silent
guess. The two pliers are told apart later, by fitting both library silhouettes to the episode mask (scene.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from cleantables.twin.inputs import Detections, frame, tracks, upsample_mask
from cleantables.twin.projection import lab

# key: class, category, kind of mesh, real mass (kg), friction (static, dynamic), thickness prior of flat objects (m),
# destination container of task objects
CATALOGUE = {
    # tools task
    "pliers_a": dict(
        cls="tool", category="tool/pliers", kind="flat", thick=0.014, mass=0.16, mu=(0.6, 0.5), dest="toolbox"
    ),
    "pliers_b": dict(
        cls="tool", category="tool/pliers", kind="flat", thick=0.014, mass=0.16, mu=(0.6, 0.5), dest="toolbox"
    ),
    "wrench": dict(
        cls="tool", category="tool/wrench", kind="flat", thick=0.010, mass=0.12, mu=(0.6, 0.5), dest="toolbox"
    ),
    "utility_knife": dict(
        cls="tool", category="tool/utility_knife", kind="flat", thick=0.018, mass=0.10, mu=(0.6, 0.5), dest="toolbox"
    ),
    "screwdriver": dict(
        cls="tool", category="tool/screwdriver", kind="flat", thick=0.020, mass=0.06, mu=(0.6, 0.5), dest="toolbox"
    ),
    "toolbox": dict(cls="container", category="container/toolbox", kind="tray", mass=0.35, mu=(0.7, 0.6)),
    "toolbox_lid": dict(cls="obstacle", category="box/toolbox_lid", kind="cuboid", mass=0.08, mu=(0.7, 0.6)),
    # boxes: distractors in the tools scenes, the items of the stacking task
    "amazon_box": dict(cls="box", category="box/amazon", kind="cuboid", mass=0.45, mu=(0.7, 0.6)),
    "cipriata_box": dict(cls="box", category="box/cipriata", kind="cuboid", mass=0.35, mu=(0.7, 0.6)),
    "lotus_box": dict(cls="box", category="box/lotus", kind="cuboid", mass=0.30, mu=(0.7, 0.6)),
    "black_box": dict(cls="box", category="box/black", kind="cuboid", mass=0.20, mu=(0.7, 0.6)),
    "orange_box": dict(cls="box", category="box/orange", kind="cuboid", mass=0.12, mu=(0.7, 0.6)),
    "yellow_box": dict(cls="box", category="box/yellow", kind="cuboid", mass=0.15, mu=(0.7, 0.6)),
    # fruit task
    "fruit_bowl": dict(cls="container", category="container/fruit_bowl", kind="hy3d", mass=0.6, mu=(0.6, 0.5)),
    "banana": dict(cls="fruit", category="fruit/banana", kind="hy3d", mass=0.12, mu=(0.7, 0.6), dest="fruit_bowl"),
    "banana_a": dict(cls="fruit", category="fruit/banana", kind="hy3d", mass=0.12, mu=(0.7, 0.6), dest="fruit_bowl"),
    "banana_b": dict(cls="fruit", category="fruit/banana", kind="hy3d", mass=0.12, mu=(0.7, 0.6), dest="fruit_bowl"),
    "green_apple": dict(cls="fruit", category="fruit/apple", kind="apple", mass=0.15, mu=(0.7, 0.6), dest="fruit_bowl"),
    "ginger": dict(
        cls="fruit", category="fruit/ginger", kind="pillow", thick=0.022, mass=0.05, mu=(0.7, 0.6), dest="fruit_bowl"
    ),
}

TOOL_PHRASES = {
    "plier": "pliers",
    "wrench": "wrench",
    "yellow tool": "utility_knife",
    "utility knife": "utility_knife",
    "knife": "utility_knife",
    "screwdriver": "screwdriver",
}
TOOLBOX_FOOTPRINT = (0.26, 0.155)  # m, closed toolbox / tray / lid
OPEN_TRAY_SATURATION = 100  # 90th-percentile HSV saturation: the open tray shows its gold walls


def classify(obj: dict) -> tuple[str | None, str, bool]:
    """events.json object (phrase, rgb, size [L, W, top]) -> (key or None, reason, uncertain)."""
    ph = obj["phrase"].lower()
    L, Wd, _ = obj["size"]
    r, g, b = obj["rgb"]
    lightness = lab(obj["rgb"])[0]
    for k, v in TOOL_PHRASES.items():
        if k in ph:
            return v, f"phrase '{obj['phrase']}'", False
    for word, key in (("banana", "banana"), ("apple", "green_apple"), ("ginger", "ginger"), ("bowl", "fruit_bowl")):
        if word in ph:
            return key, "phrase", False
    if any(k in ph for k in ("tape", "label")):
        return None, f"part / junk phrase '{obj['phrase']}'", False
    if Wd < 0.06 or max(L, Wd) < 0.12:
        return None, f"too thin / small for a box ({L:.2f} x {Wd:.2f})", False
    sat = max(r, g, b) - min(r, g, b)
    if min(r, g, b) >= 235 and sat <= 25:
        return "white", "white (toolbox tray or lid; resolved per video by height)", False
    if r - b >= 90 and r - g >= 45 and max(L, Wd) < 0.26:
        return "orange_box", f"orange rgb {obj['rgb']}", False
    if b - r >= 15 and b >= g:
        return "lotus_box", f"navy rgb {obj['rgb']}", False
    if L >= 0.345 and Wd >= 0.24:
        return "amazon_box", f"largest footprint {L:.2f} x {Wd:.2f}", False
    if r >= 240 and g >= 195 and b <= 120:
        # one recording measures its footprint at 0.35 m, probably merged with a neighbour
        return "yellow_box", f"yellow rgb {obj['rgb']}", L > 0.30
    if max(r, g, b) <= 80 and sat <= 25:
        return "black_box", f"black rgb {obj['rgb']}", False
    if max(L, Wd) < 0.24 and r - b >= 25 and r >= g:
        return "orange_box", f"small reddish box rgb {obj['rgb']} (over-exposed orange)", True
    if lightness < 60:
        return "black_box", f"dark L*={lightness:.0f}", True
    return (
        "cipriata_box",
        f"remaining grey-brown medium box rgb {obj['rgb']} {L:.2f} x {Wd:.2f}",
        not (0.24 <= L <= 0.36),
    )


def white_saturation(video: str, obj: dict, n: int = 7) -> float:
    """median over a few sightings of the 90th-percentile HSV saturation inside the track's mask."""
    dets, R = Detections(video), tracks(video)
    track = R["objects"][obj["src"]]
    pairs = list(zip(track["sight_k"].tolist(), track["det"], strict=False))
    vals = []
    for k, d in pairs[:: max(1, len(pairs) // n)]:
        m = upsample_mask(dets.mask(d))
        if m.sum() < 50:
            continue
        vals.append(float(np.percentile(cv2.cvtColor(frame(video, k), cv2.COLOR_BGR2HSV)[..., 1][m], 90)))
    return float(np.median(vals)) if vals else 0.0


def resolve_video(objs: list[dict], video: str | None = None) -> list[dict]:
    """classify every track of a video.

    A white track with the toolbox footprint is the open tray when its gold walls show ('toolbox'), else the closed
    toolbox or its lid ('white_box', resolved per episode). Without `video` (no frames) every toolbox-like white track
    is 'white_box'.
    """
    res = []
    for o in objs:
        k, why, unc = classify(o)
        res.append(dict(track=o["id"], key=k, why=why, uncertain=unc))
    by_id = {o["id"]: o for o in objs}

    def footprint_error(r):
        size = by_id[r["track"]]["size"]
        return abs(size[0] - TOOLBOX_FOOTPRINT[0]) + abs(size[1] - TOOLBOX_FOOTPRINT[1])

    for r in res:
        if r["key"] != "white":
            continue
        if footprint_error(r) >= 0.07:
            size = by_id[r["track"]]["size"]
            r["key"], r["why"], r["uncertain"] = (
                None,
                r["why"] + f" -> white but not toolbox-like ({size}): dropped",
                True,
            )
            continue
        sat = white_saturation(video, by_id[r["track"]]) if video else 0.0
        r["sat"] = sat
        if sat >= OPEN_TRAY_SATURATION:
            r["key"], r["why"] = "toolbox", r["why"] + f" -> gold walls visible (sat {sat:.0f}) = open toolbox tray"
        else:
            r["key"], r["why"] = "white_box", r["why"] + f" -> white (sat {sat:.0f}) = closed toolbox or lid"
    return res
