"""Video -> 3D sim scene: data/twin_table/boxstack/<V>/scene_<run>.json, one per run, spawnable in the box-stacking sim (cuboids;
tools as primitives).

Contents (all lengths in m, angles in rad):
  table   : plane (z = 0 in the table frame; normal + offset in the camera frame of the video), extent (xy box of the table pixels lifted to
            the plane, 2-98 %), camera pose T_cam_to_table and K (to re-render / check).
  boxes   : id, size [L, W, H] (L >= W, H incl. the lid for the kit), start pose (x, y, z_centre, yaw) in the table frame, colour (median RGB of
            its SAM 3 masks in the first seconds), role ('box' | 'kit'), stack order index (from the run), mass_kg (not measured: 0.3 kg
            default for empty cardboard, flagged).
  kit     : which box is the kit, its lid (measured if the tracker found the lid as a separate thin box, else assumed 2 cm with the kit's
            footprint, flagged 'assumed'), tools packed (from tools.json).
  tools   : label, start pose (first sighting, table frame), rough size from depth (extent of the lifted SAM 3 mask: L, W, H), primitive
            'box', mesh_path '' (no meshes).
  stack_square, haul_spot : xy + yaw (from the run: the build site and the final-stack position).
  sim     : the same scene in the sim's world frame (table top z = 0; the stack square moved to STACK_XY, rotated so that the haul spot
            lies along -y from it as HAUL_XY does): sizes (K,3), pos (K,3) box centres, yaw (K) - ready for build_cfg(sizes=...) /
            place_boxes(pos, yaw). Boxes that would fall off the sim table are flagged ('outside_sim_table').
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from cleantables.perception.scene.common import WORK, meta

from cleantables.perception.boxes import geometry as G
from cleantables.perception.boxes.common import jdump, log, vdir

STACK_XY, HAUL_XY = (0.0, -0.05), (0.0, -0.25)          # stack square and haul spot in the sim world frame
SIM_TABLE_X, SIM_TABLE_Y = (-0.55, 0.55), (-0.55, 0.45)
MASS_DEFAULT = 0.3


def _inside(p, B, m=0.05):
    q = B["init"]; c, s = np.cos(q["yaw"]), np.sin(q["yaw"])
    dx, dy = p[0] - q["x"], p[1] - q["y"]
    return abs(c * dx + s * dy) <= B["L"] / 2 + m and abs(-s * dx + c * dy) <= B["W"] / 2 + m


def lab2rgb(lab):
    px = np.uint8([[np.clip(lab, 0, 255)]])
    return [int(c) for c in cv2.cvtColor(px, cv2.COLOR_LAB2RGB)[0, 0]]


def table_extent(T, K, table_mask):
    nrm, off = T[2, :3], T[2, 3]
    ys, xs = np.nonzero(table_mask[::3, ::3]); ys, xs = ys * 3, xs * 3
    rays = np.stack([(xs * 2 + 0.5 - K[0, 2]) / K[0, 0], (ys * 2 + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1)
    t = -off / (rays @ nrm)
    P = G.to_table(T, rays[t > 0] * t[t > 0, None])
    lo, hi = np.percentile(P[:, :2], 2, 0), np.percentile(P[:, :2], 98, 0)
    return [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]


def tool_sizes(video, tracks, T_all, K, scale, D10):
    """Rough size of each tool track: extent of its SAM 3 mask lifted with depth, at its first 3 sightings (median)."""
    p = WORK / video / "sam3_tools.npz"
    if not p.exists():
        return {}
    from cleantables.perception.boxes.tools import load_npz_dets
    d = load_npz_dets(p)
    out = {}
    for t in tracks:
        ext = []
        for k, pos in list(zip(t["k"], t["p"]))[:3]:
            cand = [i for i in np.nonzero(d["frame"] == k)[0] if d["prompts"][d["prompt"][i]] == t["label"]]
            best, bd = None, 9
            for i in cand:
                Q = G.lift_mask(np.asarray(D10[k], np.float32) * scale[k], K, T_all[k], d["masks"][i])
                if len(Q) < 10:
                    continue
                dd = np.linalg.norm(np.median(Q, 0)[:2] - np.array(pos[:2]))
                if dd < bd:
                    best, bd = Q, dd
            if best is not None and bd < 0.05:
                x, y, L, W, yaw = G.rect_extent(best[:, :2])
                ext.append([L, W, float(np.percentile(best[:, 2], 95) - min(0.0, np.percentile(best[:, 2], 5))), yaw])
        if ext:
            e = np.median(np.array(ext), 0)
            out[t["id"]] = dict(size=[round(float(max(e[0], e[1])), 3), round(float(min(e[0], e[1])), 3), round(float(max(0.01, e[2])), 3)],
                                yaw=float(e[3]))
    return out


def to_sim(scene):
    """Rigid 2D transform table frame -> sim world: square -> STACK_XY, haul direction -> HAUL_XY - STACK_XY (= -y)."""
    sq, hs = scene["stack_square"], scene["haul_spot"]
    v = np.array([hs["x"] - sq["x"], hs["y"] - sq["y"]]) if hs else np.array([0.0, -1.0])
    if np.linalg.norm(v) < 0.05:
        v = np.array([0.0, -1.0])
    ang = np.arctan2(-1.0, 0.0) - np.arctan2(v[1], v[0])
    c, s = np.cos(ang), np.sin(ang)
    def f(x, y):
        dx, dy = x - sq["x"], y - sq["y"]
        return STACK_XY[0] + c * dx - s * dy, STACK_XY[1] + s * dx + c * dy
    sizes, pos, yaw, ids, out = [], [], [], [], []
    for b in scene["boxes"]:
        x, y = f(b["pose"]["x"], b["pose"]["y"])
        sizes.append(b["size"]); pos.append([x, y, b["pose"]["z"]]); yaw.append(b["pose"]["yaw"] + ang); ids.append(b["id"])
        L, W = b["size"][0], b["size"][1]
        r = G.rect_corners(x, y, L, W, b["pose"]["yaw"] + ang)
        if r[:, 0].min() < SIM_TABLE_X[0] or r[:, 0].max() > SIM_TABLE_X[1] or r[:, 1].min() < SIM_TABLE_Y[0] or r[:, 1].max() > SIM_TABLE_Y[1]:
            out.append(b["id"])
    tools = []
    for t in scene["tools"]:
        x, y = f(t["pose"]["x"], t["pose"]["y"])
        tools.append(dict(id=t["id"], pos=[x, y, t["pose"]["z"]], yaw=t["pose"]["yaw"] + ang, size=t["size"]))
    hx, hy = f(hs["x"], hs["y"]) if hs else HAUL_XY
    return dict(frame="sim_env world (table top z=0, square at STACK_XY, haul spot along -y)", box_ids=ids, sizes=sizes, pos=pos, yaw=yaw,
                masses=[b["mass_kg"] for b in scene["boxes"]], stack_xy=list(STACK_XY), haul_xy=[float(hx), float(hy)], rot_rad=float(ang),
                tools=tools, outside_sim_table=out)


def export(video):
    snp = vdir(video) / "snapshot.json"
    ep = json.load(open(snp)).get("episode") if snp.exists() else None
    if not ep:
        evp = vdir(video) / "events.json"
        eps = json.load(open(evp))["episodes"] if evp.exists() else []
        ep = eps[0] if eps else None
    if not ep:
        log(f"{video} scene: no episode, skipped")
        return None
    Z = np.load(vdir(video) / "det3d.npz")
    T_all = Z["T_all"] if "T_all" in Z.files else np.repeat(Z["T"][None], meta(video)["n"], 0)
    K, scale = Z["K"], Z["scale"]
    T = T_all[0]
    D10 = np.load(vdir(video) / "depth10.npy", mmap_mode="r")
    ev = json.load(open(vdir(video) / "events.json")) if (vdir(video) / "events.json").exists() else {}
    tl = json.load(open(vdir(video) / "tools.json")) if (vdir(video) / "tools.json").exists() else None
    order = {b: i for i, b in enumerate(ep["final_order"])}
    # kit box: the box whose start footprint holds most of the tool-pack positions (packs are recorded at the moment of packing)
    kit = None
    packs = [p for e in (tl or {}).get("episodes", {}).values() for p in e.get("packs", [])]
    if packs:
        votes = {}
        for p in packs:
            for b, B in ep["boxes"].items():
                q = B["init"]; c, s = np.cos(q["yaw"]), np.sin(q["yaw"])
                dx, dy = p["xy"][0] - q["x"], p["xy"][1] - q["y"]
                if abs(c * dx + s * dy) <= B["L"] / 2 + 0.05 and abs(-s * dx + c * dy) <= B["W"] / 2 + 0.05:
                    votes[b] = votes.get(b, 0) + 1
        kit = max(votes, key=votes.get) if votes else None
    kit_src = "tool packs (tools.json)" if kit else None
    voters = {}
    if kit is None and tl:
        # No explicit packs: tools whose track ends (vanishes into a box) well before the video does vote for the box footprint
        # (start pose + 5 cm) holding their last sighting. Boxes move before packing, so the box is found among the SAM box
        # detections at that moment and matched to the run's boxes by footprint and colour.
        n_ = meta(video)["n"]
        Rr = Z["rows"]
        kf = np.unique(Rr[:, 0].astype(int))
        votes = {}
        for t in tl["tracks"]:
            if t["k"][-1] > n_ - 30:
                continue
            p = t["p"][-1]
            kk = kf[np.argmin(np.abs(kf - t["k"][-1]))]
            rows = Rr[(Rr[:, 0] == kk) & (Rr[:, 9] > 0)]
            hold = [r for r in rows if _inside(p, dict(L=r[6], W=r[7], init=dict(x=r[3], y=r[4], yaw=r[8])), 0.03)]
            if not hold:
                continue
            r = min(hold, key=lambda r: r[6] * r[7])
            def cost(B):
                cl = B.get("color_lab") or [0, 128, 128]
                cc = (np.hypot(cl[1] - r[17], cl[2] - r[18]) / 8 + abs(cl[0] - r[16]) / 20) if len(r) > 18 else 0
                return abs(max(B["L"], B["W"]) - max(r[6], r[7])) / 0.06 + abs(min(B["L"], B["W"]) - min(r[6], r[7])) / 0.06 + cc
            b = min(ep["boxes"], key=lambda b: cost(ep["boxes"][b]))
            votes[b] = votes.get(b, 0) + 1
            voters.setdefault(b, []).append(t["id"])
        if votes and max(votes.values()) >= 2:
            kit = max(votes, key=votes.get); kit_src = f"tools vanishing into it ({votes[kit]} tracks)"
    boxes = []
    for b, B in ep["boxes"].items():
        col = B.get("color_lab")
        q = B["init"]
        boxes.append(dict(id=b, size=[round(max(B["L"], B["W"]), 3), round(min(B["L"], B["W"]), 3), round(B["H"], 3)],
                          pose=dict(x=round(q["x"], 3), y=round(q["y"], 3), z=round(q.get("z0", 0.0) + B["H"] / 2, 3), yaw=round(q["yaw"], 3)),
                          color_rgb=lab2rgb(col) if col else None, role="kit" if b == kit else "box", stack_index=order.get(b),
                          mass_kg=MASS_DEFAULT, mass_src="default (not measured)"))
    lid = None
    if kit:
        lids = ev.get("lids", {})
        if lids:
            lb = list(lids)[0]; LB = ev["boxes"].get(lb, {})
            lid = dict(size=[round(LB.get("L", 0), 3), round(LB.get("W", 0), 3), round(LB.get("H", 0.02), 3)], on=kit, src="measured (tracker: thin box on the kit)")
        else:
            kb = ep["boxes"][kit]
            lid = dict(size=[round(max(kb["L"], kb["W"]) + 0.01, 3), round(min(kb["L"], kb["W"]) + 0.01, 3), 0.02], on=kit,
                       src="assumed (lid not seen as a separate box): kit footprint + 1 cm, 2 cm thick; the kit's H includes it")
    tools = []
    if tl:
        sz = tool_sizes(video, tl["tracks"], T_all, K, scale, D10)
        for t in tl["tracks"]:
            p = t["p"][0]
            s_ = sz.get(t["id"], dict(size=[0.1, 0.03, 0.03], yaw=0.0))
            tools.append(dict(id=t["id"], label=t["label"], pose=dict(x=round(p[0], 3), y=round(p[1], 3), z=round(max(p[2], s_["size"][2] / 2), 3),
                                                                     yaw=round(s_["yaw"], 3)),
                              size=s_["size"], size_src="depth" if t["id"] in sz else "default", primitive="box", mesh_path="",
                              t_first=round(t["k"][0] / meta(video)["fps"], 2),
                              packed_into_kit=any(pk["tool"] == t["id"] for pk in packs) or t["id"] in voters.get(kit, [])))
    Ti = np.linalg.inv(T)
    scene = dict(video=video, run=ep["id"], split=ep.get("split", "seen"), source="real video (automatic)", reader=ep.get("reader", "tracker"),
                 table=dict(plane_in_table_frame=dict(normal=[0, 0, 1], offset=0.0),
                            plane_in_camera_frame=dict(normal=[float(x) for x in T[2, :3]], offset=float(T[2, 3])),
                            extent_xy=table_extent(T, K, Z["table"]), T_cam_to_table=T.tolist(), K=K.tolist(),
                            camera_bumps=json.load(open(vdir(video) / "table.json")).get("bump_frames", [])),
                 boxes=boxes, kit=dict(box=kit, src=kit_src, lid=lid, n_tools_packed=sum(t["packed_into_kit"] for t in tools)) if kit else None,
                 tools=tools,
                 stack_square=dict(x=ep["square"]["x"], y=ep["square"]["y"], yaw=ep["square"]["yaw"], src=ep["square"]["src"]),
                 haul_spot=dict(x=ep["spot_b"]["x"], y=ep["spot_b"]["y"], yaw=ep["spot_b"]["yaw"], src=ep["spot_b"]["src"]) if ep.get("spot_b") else None,
                 stack_order=ep["final_order"])
    scene["sim"] = to_sim(scene)
    out = vdir(video) / f"scene_{ep['id']}.json"
    jdump(scene, out)
    log(f"{video} scene: {len(boxes)} boxes, kit {kit}, {len(tools)} tools -> {out.name}"
        + (f" (outside sim table: {scene['sim']['outside_sim_table']})" if scene["sim"]["outside_sim_table"] else ""))
    return scene


def load_sim_arrays(path):
    """Scene json -> (sizes (K,3), pos (K,3), yaw (K,)) numpy arrays, for build_cfg(sizes[None]) / env.place_boxes(...)."""
    s = json.load(open(path))["sim"]
    return np.array(s["sizes"]), np.array(s["pos"]), np.array(s["yaw"])
