"""Convert a video processed by cleantables.perception (events.json / tracks.pkl / vision.npz / vjepa.npz / hand_video.npz) into the base world-model schema.

  $VLA_PY -m cleantables.worldmodel.data_generic box          # the 19 box videos
  from cleantables.worldmodel.data_generic import build_from_generic
  build_from_generic(["V1", "V2"], "sort the fruit ...", {"V1": ("train", "V1"), "V2": ("unseen", "V2")}, "fruit_tools")

Snapshots are all tracked keyframes (5 fps) from the end of the previous move (or the video start) to 0.3 s before the next move starts;
each is labelled with that next move. Snapshots where the mover has no valid 3D box at that keyframe are skipped (counted in the meta).
Frame: the generic table frame (z up, table top 0). Table = one 'table' token (axis-aligned extent rectangle of the table pixels).
Hand tokens: SAM 3 'hand' video masks of that keyframe + MoGe depth -> up to 2 connected components (>= 60 px at 270x480), sorted by x:
[valid, centroid xyz (median), extent xyz (5-95 pct), first two PCA axes (6D)].
"""
from __future__ import annotations

import json
import pickle
import sys

import cv2
import numpy as np

from cleantables.perception.common import WORK, depth, embed, geom, meta, vdir
from cleantables.perception.frame_model import box_feat
from cleantables.perception.scene.recon3d import backproject, to_table
from cleantables.perception.boxes.scene_export import table_extent

from cleantables.worldmodel.schema import KIND, MAXO, REL, TABLE, LEAVES, blank, resample, save

MIN_HAND_PX = 60


def hand_tokens(v, G, D10, hv):
    """Return {keyframe: (2, 12) hand_geo} for every keyframe of hand_video.npz."""
    out = {}
    fr, shp = hv["frames"], hv["shape"]
    M = np.unpackbits(hv["masks"])[:int(np.prod(shp))].reshape(shp).astype(bool)
    for t, k in enumerate(fr.tolist()):
        g = np.zeros((2, 13), np.float32)
        n, lab, st, _ = cv2.connectedComponentsWithStats(M[t].astype(np.uint8), connectivity=8)
        comps = [i for i in np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1 if st[i, cv2.CC_STAT_AREA] >= MIN_HAND_PX][:2]
        hs = []
        Dk = np.asarray(D10[k], np.float32) * G["scale"][k]
        for i in comps:
            P, _ = backproject(Dk, G["K"], lab == i)
            if len(P) < 30:
                continue
            Q = to_table(G["T_all"][k], P)
            c = np.median(Q, 0)
            lo, hi = np.percentile(Q, 5, 0), np.percentile(Q, 95, 0)
            w, V = np.linalg.eigh(np.cov((Q - c).T))
            a1 = V[:, 2]; a2 = V[:, 1]
            hs.append(np.r_[1.0, c, hi - lo, a1, a2].astype(np.float32))
        hs.sort(key=lambda h: h[1])
        for j, h in enumerate(hs[:2]):
            g[j] = h
        out[k] = g
    return out


def build_from_generic(video_list, task_text, split_map, out_name, dataset=None, prog_video_ok=None, save_out=True):
    """split_map: video -> (split, group); prog_video_ok: video -> 0/1 (default 0). Returns (samples, meta)."""
    from cleantables.features import vjepa_tokens as VJ
    dataset = dataset or out_name
    prog_video_ok = prog_video_ok or {}
    S, skipped, miss_vj, miss_hand = [], {}, [], []
    phr_all = {"table"}
    for v in video_list:
        if not (vdir(v) / "events.json").exists():
            continue
        Ev = json.load(open(vdir(v) / "events.json"))
        R = pickle.load(open(vdir(v) / "tracks.pkl", "rb"))
        Z = np.load(vdir(v) / "vision.npz")
        cemb = dict(zip(Z["det"].tolist(), Z["emb"].astype(np.float16)))
        G = geom(v); D10 = depth(v)
        ext = table_extent(G["T"], G["K"], G["table"])
        table_geo = np.array([(ext[0][0] + ext[0][1]) / 2, (ext[1][0] + ext[1][1]) / 2, 0, ext[0][1] - ext[0][0], ext[1][1] - ext[1][0], 0, 0, 1], np.float32)
        try:
            vj = VJ.load(v)
        except Exception:
            vj = None; miss_vj.append(v)
        try:
            hg = hand_tokens(v, G, D10, np.load(vdir(v) / "hand_video.npz"))
        except Exception:
            hg = {}; miss_hand.append(v)
        src = {o["id"]: o["src"] for o in Ev["objects"]}
        objdets = {oid: dict(zip(R["objects"][s]["sight_k"].tolist(), R["objects"][s]["det"])) for oid, s in src.items()}
        kfs = sorted(R["kept"])
        n_fr = meta(v)["n"]
        dur = n_fr / 10.0
        split, group = split_map[v]
        evs = sorted(Ev["events"], key=lambda e: e["f_start"])
        prev_end = 0
        for e in evs:
            mover = e["obj"]
            rel = REL.get(e["relation"], -1)
            path = np.asarray(e["path"], np.float32) if e.get("path") else None
            if path is None or mover not in objdets:
                skipped[v] = skipped.get(v, 0) + 1; prev_end = max(prev_end, e["f_end"]); continue
            dt = (e["f_end"] - e["f_start"]) / 10.0 / 31
            # hand path: hand centroid (nearest valid hand to the mover's path point) at keyframes in the move
            ks_m = [k for k in kfs if e["f_start"] <= k <= e["f_end"] and k in hg]
            hp = np.zeros((32, 3), np.float32); hok = np.zeros(32, np.uint8)
            if len(ks_m) >= 2:
                tt, pts, okk = [], [], []
                for k in ks_m:
                    h = hg[k]
                    if h[:, 0].any():
                        po = np.array([np.interp((k - e["f_start"]) / max(1, e["f_end"] - e["f_start"]), np.linspace(0, 1, 32), resample(path)[:, i]) for i in range(3)])
                        j = int(np.argmin([np.linalg.norm(h[i, 1:4] - po) if h[i, 0] else 1e9 for i in range(2)]))
                        tt.append((k - e["f_start"]) / max(1, e["f_end"] - e["f_start"])); pts.append(h[j, 1:4])
                if len(tt) >= 2:
                    u = np.linspace(0, 1, 32); pts = np.array(pts)
                    hp = np.stack([np.interp(u, tt, pts[:, i]) for i in range(3)], 1).astype(np.float32)
                    hok = ((u >= min(tt)) & (u <= max(tt))).astype(np.uint8)
            end = e["end"]
            for k in kfs:
                if k < prev_end or k > e["f_start"] - 3:
                    continue
                dets = [i for i in R["kept"][k] if R["det_box"].get(i) and i in cemb]
                dets = sorted(dets, key=lambda i: -R["det_box"][i]["prob"])[:MAXO - 1]
                if mover not in objdets or k not in objdets[mover] or objdets[mover][k] not in dets:
                    skipped[v] = skipped.get(v, 0) + 1; continue
                owner = {dd[k]: oid for oid, dd in objdets.items() if k in dd}
                tok_obj = [owner.get(i, -1) for i in dets]
                s = blank()
                n = len(dets)
                for j, i in enumerate(dets):
                    b = R["det_box"][i]
                    s["obj_geo"][j] = box_feat(b); s["obj_crop"][j] = cemb[i]; s["obj_crop_ok"][j] = 1
                    s["obj_mask"][j] = 1; s["obj_kind"][j] = KIND["movable"]; s["obj_id"][j] = tok_obj[j]
                    phr_all.add(b["phrase"])
                s["_phr"] = [R["det_box"][i]["phrase"] for i in dets] + ["table"]
                s["obj_geo"][n] = table_geo; s["obj_mask"][n] = 1; s["obj_kind"][n] = KIND["table"]; s["table_idx"][...] = n
                s["mover"][...] = dets.index(objdets[mover][k])
                if e["relation"] == "leaves":
                    tg = LEAVES
                elif e["target"] is None:
                    tg = TABLE
                else:
                    tg = tok_obj.index(e["target"]) if e["target"] in tok_obj else -1
                s["target"][...] = tg; s["relation"][...] = rel
                if end and np.isfinite(end.get("yaw", np.nan)):
                    s["end_pose"][:] = [end["x"], end["y"], end["top"], end["yaw"]]; s["end_ok"][...] = 1
                    off = e.get("offset")
                    s["end_off"][:] = off if off is not None else np.array([end["x"] - e["start"]["x"], end["y"] - e["start"]["y"], end["top"] - e["start"]["top"]])
                s["path_obj"][:] = resample(path); s["path_hand"][:] = hp; s["path_hand_ok"][:] = hok
                s["path_dt"][...] = dt; s["prog_steps"][:] = np.linspace(0, 1, 32)
                s["prog_video"][...] = k / n_fr; s["prog_video_ok"][...] = prog_video_ok.get(v, 0)
                s["time_to_move"][...] = (e["f_start"] - k) / 10.0
                if k in hg:
                    s["hand_geo"][:] = hg[k]
                if vj is not None:
                    a = VJ.nearest_keyframe(v, k, vj); b = VJ.nearest_keyframe(v, e["f_end"], vj)
                    s["vj_tok"][:] = vj["tok"][a]; s["vj_tok_flip"][:] = vj["tok_f"][a]; s["vj_ok"][...] = 1
                    s["vj_future_glob"][:] = vj["glob"][b]; s["vj_future_ok"][...] = 1
                s.update(dataset=dataset, video=v, group=group, split=split, task_text=task_text)
                s["snap_k"][...] = k; s["ev_id"][...] = e["id"]
                S.append(s)
            prev_end = max(prev_end, e["f_end"])
    # text embeddings (MiniLM)
    E = dict(zip(sorted(phr_all), embed(sorted(phr_all))))
    for s in S:
        for j, p in enumerate(s.pop("_phr")):
            s["obj_txt"][j] = E[p]
    meta_extra = dict(skipped_snapshots_mover_not_visible=skipped, videos_missing_vjepa=miss_vj, videos_missing_hand=miss_hand,
                      table_idx_note="the table token index is in `table_idx`; relation 'table' moves use target = 24 (TABLE) per SCHEMA")
    if save_out:
        return S, save(S, out_name, meta_extra)
    return S, meta_extra


SEEN = [f"IMG_{i}" for i in (2533, 2534, 2535, 2536, 2537, 2538, 2539, 2542, 2544, 2546, 2547, 2548)]
UNSEEN = [f"IMG_{i}" for i in range(2549, 2556)]

if __name__ == "__main__":
    if sys.argv[1] == "box":
        sm = {v: ("train", v) for v in SEEN} | {v: ("unseen", v) for v in UNSEEN}
        _, M = build_from_generic(SEEN + UNSEEN, "stack the boxes", sm, "box")
        print(json.dumps({k: v for k, v in M.items() if k != "keys"}, indent=1))
