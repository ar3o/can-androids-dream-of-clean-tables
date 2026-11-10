"""Proof images for one box video: per episode a top view (initial scene, placements in order, final stack) and a strip of keyframes with
every box's estimated 3D wireframe (projected with K, T), for checking an ingest by eye.
  -> videos/twin_table/boxstack/<V>_ep<i>_top.png, <V>_ep<i>_wire.jpg, <V>_boxes.jpg (every box identity with its size)"""
from __future__ import annotations

import json

import cv2
import numpy as np

from cleantables.perception.scene.common import read_frame

from cleantables.perception.boxes import geometry as G
from cleantables.perception.boxes.common import VIDS, vdir

COLS = [(0, 140, 255), (60, 200, 60), (255, 120, 0), (200, 60, 200), (0, 220, 220), (60, 60, 255), (255, 255, 0), (150, 150, 255),
        (100, 255, 160), (255, 160, 200)]


def draw_box(img, K, T, x, y, z0, L, W, H, yaw, col, th=2, label=None):
    C = G.box_corners3d(x, y, z0, L, W, H, yaw)
    uv, zc = G.project(C, K, T)
    if (zc <= 0).any():
        return
    for a, b in G.BOX_EDGES:
        cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), col, th, cv2.LINE_AA)
    if label:
        p = uv[4:].mean(0).astype(int)
        cv2.putText(img, label, (int(p[0]) - 10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (int(p[0]) - 10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def topview(ep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ids = sorted(ep["boxes"])
    for i, b in enumerate(ids):
        B = ep["boxes"][b]
        r = G.rect_corners(B["init"]["x"], B["init"]["y"], B["L"], B["W"], B["init"]["yaw"])
        ax[0].fill(r[:, 0], r[:, 1], alpha=0.35, color=np.array(COLS[i % 10][::-1]) / 255)
        ax[0].text(B["init"]["x"], B["init"]["y"], f"{b}\n{B['L']*100:.0f}x{B['W']*100:.0f}x{B['H']*100:.0f}", ha="center", fontsize=7)
    q = ep["square"]
    r = G.rect_corners(q["x"], q["y"], q.get("size", 0.4), q.get("size", 0.4), q["yaw"])
    ax[0].plot(*np.r_[r, r[:1]].T, "k--", lw=1); ax[0].set_title(f"{ep['id']} start (square: {q['src']})", fontsize=9)
    for n, pl in enumerate(ep["placements"]):
        if pl.get("path"):
            P = np.array(pl["path"]); ax[0].plot(P[:, 0], P[:, 1], "-", lw=0.8, color="gray")
        B0 = ep["boxes"][pl["box"]]["init"]
        ax[0].annotate(str(n + 1), (B0["x"], B0["y"] - 0.03), fontsize=9, color="red")
    ax[0].set_aspect("equal")
    zc = 0
    for n, pl in enumerate(ep["placements"]):
        B = ep["boxes"][pl["box"]]
        z0 = pl["pose"]["z0"]
        i = ids.index(pl["box"])
        ax[1].add_patch(plt.Rectangle((pl["pose"]["x"] - B["L"] / 2, z0), B["L"], B["H"], alpha=0.4, color=np.array(COLS[i % 10][::-1]) / 255))
        ax[1].text(pl["pose"]["x"], z0 + B["H"] / 2, f"{n + 1}:{pl['box']} on {pl['support']}", ha="center", fontsize=7)
        zc = max(zc, z0 + B["H"])
    ax[1].set_xlim(ep["square"]["x"] - 0.4, ep["square"]["x"] + 0.4); ax[1].set_ylim(0, max(0.3, zc + 0.05)); ax[1].set_aspect("equal")
    ax[1].set_title("stack (side view along table x; z0 measured)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def video_sheets(video):
    VIDS.mkdir(parents=True, exist_ok=True)
    ev = json.load(open(vdir(video) / "events.json"))
    tb = json.load(open(vdir(video) / "table.json"))
    K, T0 = np.array(tb["K"]), np.array(tb["T_cam_to_table"])
    z_ = np.load(vdir(video) / "det3d.npz")
    T_all = z_["T_all"] if "T_all" in z_.files else None
    Tk = lambda k: T_all[min(k, len(T_all) - 1)] if T_all is not None else T0
    segs = {s["id"]: s for s in json.load(open(vdir(video) / "segments.json"))["segments"]}
    # every box identity at its first segment
    tiles = []
    for i, (b, B) in enumerate(sorted(ev["boxes"].items())[:12]):
        if B["chain"][0] not in segs:
            continue
        s = segs[B["chain"][0]]
        img = read_frame(video, s["k0"])
        draw_box(img, K, Tk(s["k0"]), s["x"], s["y"], max(0.0, s["top"] - B["H"]), B["L"], B["W"], B["H"], s["yaw"], COLS[i % 10], 2,
                 f"{b} {B['L']*100:.0f}x{B['W']*100:.0f}x{B['H']*100:.0f}cm")
        tiles.append(cv2.resize(img, (480, 270)))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(VIDS / f"{video}_boxes.jpg"), np.vstack([np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)]))
    eps = ev["episodes"]
    sp = vdir(video) / "snapshot.json"
    if sp.exists():
        sn = json.load(open(sp)).get("episode")
        if sn and sn.get("usable"):
            eps = [sn]
    for e_i, ep in enumerate(eps):
        topview(ep, VIDS / f"{ep['id']}_top.png")
        ids = sorted(ep["boxes"])
        tiles = []
        for pl in ep["placements"]:
            k = min(pl["k_place"] + 10, int(json.load(open(vdir(video).parents[2] / "table_clean/_work" / video / "frames_meta.json"))["n"]) - 1)
            img = read_frame(video, k)
            B = ep["boxes"][pl["box"]]
            draw_box(img, K, Tk(k), pl["pose"]["x"], pl["pose"]["y"], pl["pose"]["z0"], B["L"], B["W"], B["H"], pl["pose"]["yaw"],
                     COLS[ids.index(pl["box"]) % 10], 2, f"{pl['box']} on {pl['support']}")
            tiles.append(cv2.resize(img, (480, 270)))
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(VIDS / f"{ep['id']}_wire.jpg"), np.vstack([np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)]))
