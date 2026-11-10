"""Kitting stage: small parts / tools that the person packs into the kit box before stacking the boxes.

Input: sam3_tools.npz (SAM 3 with TOOL_PHRASES, sam3_box --tools), the box stages' det3d.npz (table frame T, K, per-frame depth
scale), depth10.npy, sam3_box.npz (hand masks), events.json (episodes).
  1. every tool detection (prob >= 0.4, not mostly hand, < 3 % of the image) -> 3D centroid in the table frame (median depth inside
     the eroded mask x the frame's depth scale);
  2. tracks: greedy linking across keyframes by 3D distance (< 10 cm, gap <= 6 s), label = probability-weighted majority phrase;
  3. per episode, a tool is packed when it is last seen inside a box footprint (box detections of det3d at the nearest keyframe,
     +3 cm margin) and not seen on the table again in that episode; the kit box = the episode box whose size matches that footprint.
Out: data/twin_table/boxstack/<video>/tools.json  {tracks: [...], episodes: {ep_id: {packs: [{tool, label, t, k, kit_box, xy}], kit_box}}}
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from cleantables.perception.scene.common import WORK, meta, unpack

from cleantables.perception.boxes.common import jdump, log, vdir


def load_npz_dets(path):
    z = np.load(path)
    m = int(z["shape"][0])
    return dict(frame=z["frame"], prompt=z["prompt"], prob=z["prob"], box=z["box"],
                masks=unpack(z["masks"], (m, 270, 480)) if m else np.zeros((0, 270, 480), bool),
                prompts=[str(x) for x in z["prompts"]], frames=[int(x) for x in z["frames_done"]])


def stage_tools(video, p_min=0.4):
    tp = WORK / video / "sam3_tools.npz"
    if not tp.exists():
        log(f"{video} tools: no sam3_tools.npz, skipped")
        return None
    d = load_npz_dets(tp)
    Z = np.load(vdir(video) / "det3d.npz")
    T_all = Z["T_all"] if "T_all" in Z.files else None
    T, K, scale = Z["T"], Z["K"], Z["scale"]
    R, cols = Z["rows"], [str(c) for c in Z["cols"]]
    ci = {c: i for i, c in enumerate(cols)}
    D10 = np.load(vdir(video) / "depth10.npy", mmap_mode="r")
    hands = {}
    bp = WORK / video / "sam3_box.npz"
    if bp.exists():
        db = load_npz_dets(bp)
        for i in range(len(db["frame"])):
            if db["prompts"][db["prompt"][i]] == "hand" and db["prob"][i] > 0.4:
                k = int(db["frame"][i]); hands[k] = hands.get(k, np.zeros((270, 480), bool)) | db["masks"][i]
    Ti = np.asarray(T)
    ker = np.ones((3, 3), np.uint8)
    dets = []
    for i in range(len(d["frame"])):
        if d["prob"][i] < p_min:
            continue
        k = int(d["frame"][i]); m = d["masks"][i]
        a = m.sum()
        if a < 15 or a > 0.03 * 270 * 480:
            continue
        if k in hands and (m & hands[k]).sum() > 0.6 * a:
            continue
        me = cv2.erode(m.astype(np.uint8), ker).astype(bool)
        me = me if me.sum() >= 5 else m
        if k >= len(D10):
            continue
        z = np.median(np.asarray(D10[k], np.float32)[me]) * scale[k]
        if not z > 0:
            continue
        ys, xs = np.nonzero(m)
        u, v = xs.mean() * 2 + 1, ys.mean() * 2 + 1
        pc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        Tk = T_all[k] if T_all is not None else Ti                  # per-frame T (piecewise after a tripod bump)
        p = Tk[:3, :3] @ pc + Tk[:3, 3]
        dets.append(dict(k=k, label=d["prompts"][d["prompt"][i]], prob=float(d["prob"][i]), p=p))
    fps = meta(video)["fps"]
    # ---- tracks
    dets.sort(key=lambda x: x["k"])
    tracks = []
    for dt in dets:
        best = None
        for t in tracks:
            if 0 < dt["k"] - t["k"][-1] <= 6 * fps and np.linalg.norm(t["p"][-1][:2] - dt["p"][:2]) < 0.10:
                dd = np.linalg.norm(t["p"][-1][:2] - dt["p"][:2])
                if best is None or dd < best[0]:
                    best = (dd, t)
        if best is None:
            tracks.append(dict(k=[dt["k"]], p=[dt["p"]], votes={dt["label"]: dt["prob"]}))
        else:
            t = best[1]
            if t["k"][-1] == dt["k"]:
                continue
            t["k"].append(dt["k"]); t["p"].append(dt["p"]); t["votes"][dt["label"]] = t["votes"].get(dt["label"], 0) + dt["prob"]
    tracks = [t for t in tracks if len(t["k"]) >= 3]
    for j, t in enumerate(tracks):
        t["id"] = f"tool{j:02d}"; t["label"] = max(t["votes"], key=t["votes"].get)
    # ---- box footprints per keyframe (det3d rows)
    kb = R[:, ci["k"]].astype(int)

    def footprints(k):
        ks = np.unique(kb)
        kk = ks[np.argmin(np.abs(ks - k))]
        rows = R[(kb == kk) & (R[:, ci["ok"]] > 0)]
        return [dict(x=r[ci["x"]], y=r[ci["y"]], L=r[ci["L"]], W=r[ci["W"]], yaw=r[ci["yaw"]], top=r[ci["top"]]) for r in rows]

    def inside(p, f, margin=0.03):
        c, s = np.cos(f["yaw"]), np.sin(f["yaw"])
        dx, dy = p[0] - f["x"], p[1] - f["y"]
        return abs(c * dx + s * dy) <= f["L"] / 2 + margin and abs(-s * dx + c * dy) <= f["W"] / 2 + margin
    ev = json.load(open(vdir(video) / "events.json")) if (vdir(video) / "events.json").exists() else dict(episodes=[])
    out_eps = {}
    for ep in ev.get("episodes", []):
        k0 = int(round((ep["t_start"]) * fps)); k1 = int(round(ep["t_end"] * fps))
        packs = []
        for t in tracks:
            ks = np.array(t["k"]); inn = (ks >= k0) & (ks <= k1)
            if not inn.any():
                continue
            last = int(np.nonzero(inn)[0][-1]); kl = int(ks[last]); pl = t["p"][last]
            fp = [f for f in footprints(kl) if inside(pl, f)]
            if not fp:
                continue
            f = min(fp, key=lambda f: f["L"] * f["W"])
            kit = None
            if ep.get("boxes"):
                kit = min(ep["boxes"], key=lambda b: abs(ep["boxes"][b]["L"] * ep["boxes"][b]["W"] - f["L"] * f["W"]))
            packs.append(dict(tool=t["id"], label=t["label"], k=kl, t=round(kl / fps, 2), kit_box=kit, xy=[float(pl[0]), float(pl[1])]))
        packs.sort(key=lambda x: x["k"])
        kits = [p["kit_box"] for p in packs if p["kit_box"]]
        out_eps[ep["id"]] = dict(packs=packs, kit_box=max(set(kits), key=kits.count) if kits else None)
    out = dict(video=video, tracks=[dict(id=t["id"], label=t["label"], k=[int(x) for x in t["k"]], p=[[float(c) for c in p] for p in t["p"]])
                                    for t in tracks], episodes=out_eps)
    jdump(out, vdir(video) / "tools.json")
    log(f"{video} tools: {len(dets)} detections, {len(tracks)} tracks, packs per episode "
        f"{ {e: len(v['packs']) for e, v in out_eps.items()} }")
    return out
