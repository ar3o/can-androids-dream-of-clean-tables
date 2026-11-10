"""World-model data without move cuts: the model learns by predicting every object's future; hand contact is an input and a
future-contact target, never a segmentation cue. One sample per tracked keyframe of every video, covering episodes and the reset /
setup stretches between them (data/labels/episodes_all.json 'segments'). Poses come from the SAM 3 tracker (FoundationPose only as the
alternative end pose end_xyz_fp), contact from HOIST-Former (track-level linked) and Hands23, hands from SAM 3 'hand' masks (+ WiLoR).

Per sample:
  segment   group = seg id, seg_kind (0 episode, 1 between), task_text (Gemini's instruction or ""), seg_t0 / seg_t1, t_in_seg
  heights   every 'top' (tokens, futures, end) is the tracker's where it has a sighting within +-2 frames, x, y from FP when src = 1
  tokens    obj_geo / obj_txt / obj_crop / obj_mask / obj_kind / obj_id / table_idx as tasks3 (x, y from FP where it has the object
            within +-2 frames, else the tracker; heights / L / W / yaw from the tracker), obj_pose_src (0 tracker, 1 FP)
  hands     hand_geo (SAM 3 hand tokens, every frame); wilor_J (2, 21, 3) table frame + wilor_ok (2,) where WiLoR exists
  contact   obj_contact (MAXO,) HOIST-Former score where it holds the object (track-level link), contact_ok (frame covered by HOIST);
            hands23_contact (MAXO,) + hands23_ok
  futures   fut_xyz (MAXO, F, 3) top-centre at k + FUT_DT, fut_ok (MAXO, F), fut_src (MAXO, F) 0 tracker / 1 FP (mask entries whose
            source differs from obj_pose_src at k: a source switch would look like motion); fut_contact (MAXO, F), fut_contact_ok (F,)
  end       end_xyz (MAXO, 3) last estimate inside the segment, end_ok, end_src, end_yaw (tracker, last sighting in the segment) + end_yaw_ok,
            end_present (SAM 3 sees it in the segment's last 1 s)
  ids       video = the real video (frames); track_set = the tracks.pkl it came from (IMG_2516s for IMG_2516's stretches);
            obj_id = tracks.pkl index of that track set; events_obj_id = events.json object id (-1 if none: new videos / IMG_2516s have no events.json)
  V-JEPA    vj_tok / vj_tok_flip / vj_ok as tasks3
  $VLA_PY -m cleantables.perception.tracks [VIDEOS...]   -> WM_EVAL_OUT/wm_v3_tracks/tracks_all.npz + tracks_all_meta.json
"""
from __future__ import annotations

from cleantables import CT_DATA
import json
import pickle
import sys
from collections import Counter

import numpy as np

from cleantables.perception.common import embed, geom, depth, vdir
from cleantables.perception.frame_model import box_feat
from cleantables.perception.boxes.scene_export import table_extent

from cleantables.worldmodel.schema import KIND, MAXO, ROOT
from cleantables.worldmodel.data_generic import hand_tokens

EPISODES = ROOT / "data/labels/episodes_all_v2.json"      # reviewed cuts + reset episodes (cleantables.episodes.refine)
EPISODES_V1 = ROOT / "data/labels/episodes_all.json"      # its segment ids name the HOIST / WiLoR / FP files
TRACK_SETS = ROOT / "data/generic/_track_sets.json"


def track_sets(SEG):
    """Segment id -> track set. A video's frames may be split over several SAM 3 track sets (IMG_2516: episode windows + IMG_2516s for the
    stretches; IMG_2513 / 2514: one set per v1 segment +-3 s, to bound memory). Each segment takes the set covering most of its frames."""
    reg = json.load(open(TRACK_SETS)) if TRACK_SETS.exists() else {}
    out = {}
    for sg in SEG:
        v = sg["video"]; a, b = int(round(sg["start_s"] * 10)), int(round(sg["end_s"] * 10))
        best, bo = v, -1
        for st in reg.get(v, []):
            wins = st.get("windows") or [[st["k0"], st["k1"]]]
            ov = sum(max(0, min(b, w1) - max(a, w0) + 1) for w0, w1 in wins)
            if ov > bo:
                best, bo = st["set"], ov
        out[sg["id"]] = best
    return out, reg
PV2 = __import__("pathlib").Path(CT_DATA + "/perception_v2")
HGT = PV2 / "contact_moves" / "hoist_generic_trk"
FUT_DT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]        # future horizon T2 = 4 s in 8 steps
NF = len(FUT_DT)


def pose_timeline(v, R, eps):
    """track -> (frames (n,), xyz (n, 3), src (n,) 0 tracker / 1 FP). FP samples from every episode that has an FP track for it (first
    episode wins on overlapping frames); tracker samples where no FP track covers the frame (within the FP track's first..last frame)."""
    from cleantables.perception import pose_6d as F
    fp_s, cover = {}, {}
    for e in eps:
        T = F.fp(e) or {}
        for tr, d in T.items():
            if F.sf_height(e, d["id"]) is None:
                continue
            fr = d["frames"]
            for j, k in enumerate(fr):
                fp_s.setdefault(tr, {}).setdefault(int(k), F.topc(e, d, j)[0])
            cover.setdefault(tr, []).append((int(fr.min()), int(fr.max())))
    out = {}
    for i, o in enumerate(R["objects"]):
        # FP gives x, y; the height stays the tracker's (nearest sighting within +-2 frames), as for the tokens (the FP centre height is
        # affected by rotation flips, see pose_6d.build_pos); FP's own height only where the tracker has no sighting
        tz = {int(k): b["top"] for k, b in zip(o["sight_k"], o["boxes"]) if b}
        tk = np.array(sorted(tz), int)
        def ztr(k, p):
            if len(tk):
                j = int(np.argmin(np.abs(tk - k)))
                if abs(tk[j] - k) <= 2:
                    return np.array([p[0], p[1], tz[int(tk[j])]])
            return p
        rows = {k: (ztr(k, p), 1) for k, p in fp_s.get(i, {}).items()}
        cv = cover.get(i, [])
        for k, b in zip(o["sight_k"], o["boxes"]):
            k = int(k)
            if b and not any(a <= k <= z for a, z in cv):
                rows[k] = (np.array([b["x"], b["y"], b["top"]]), 0)
        ks = np.array(sorted(rows), int)
        out[i] = (ks, np.array([rows[k][0] for k in ks], float).reshape(-1, 3), np.array([rows[k][1] for k in ks], np.int8))
    return out


def at(tl, k, tol=2):
    ks, X, S = tl
    if not len(ks):
        return None, None
    j = int(np.argmin(np.abs(ks - k)))
    return (X[j], int(S[j])) if abs(ks[j] - k) <= tol else (None, None)


def contact_tables(v, eps, n_tr):
    """-> (held (n_frames_video, n_tracks) float32 score, covered (n_frames_video,) bool) from the track-level HOIST files; each 10 fps frame
    takes the nearest HOIST (6 fps) frame within +-1."""
    n = json.load(open(ROOT / f"data/table_clean/_work/{v}/frames_meta.json"))["n"]
    H = np.zeros((n, n_tr), np.float32); C = np.zeros(n, bool)
    for e in eps:
        p = HGT / f"{e}.npz"
        if not p.exists():
            continue
        Z = np.load(p); wf, held = Z["work_frames"].astype(int), Z["held"]
        for k in range(wf[0], wf[-1] + 1):
            j = int(np.argmin(np.abs(wf - k)))
            if abs(wf[j] - k) <= 1:
                H[k, :held.shape[1]] = np.maximum(H[k, :held.shape[1]], held[j]); C[k] = True
    return H, C


def hands23_tables(base, vset, R, kfs, n):
    """Second contact channel: Hands23 (published, demo defaults; run on every frame of the video) -> per keyframe and track the
    contact score of a hand in 'object_contact' whose first-object mask overlaps the track's SAM 3 mask with IoU >= 0.1 (the paper's
    association threshold, applied to masks); max over hands. ok = Hands23 processed that frame."""
    import glob
    from cleantables.perception.episode_io import sam3_masks
    H = np.zeros((n, len(R["objects"])), np.float32); C = np.zeros(n, bool)
    f = glob.glob(str(PV2 / "contact_moves" / "hands23_full" / f"{base}_0_*.npz"))
    f = [x for x in f if not x.endswith("_objmasks.npz")]
    if not f or not __import__("os").path.exists(f[0][:-4] + "_objmasks.npz"):
        return H, C
    z = np.load(f[0], allow_pickle=True); om = np.load(f[0][:-4] + "_objmasks.npz")
    OM = np.unpackbits(om["masks"])[:int(np.prod(om["shape"]))].reshape(om["shape"]).astype(bool)
    pos = {int(i): r for r, i in enumerate(om["inst_index"])}
    inst = {(int(fr), int(rw)): i for i, (fr, rw) in enumerate(zip(z["inst_frame"], z["inst_row"]))}
    C[np.asarray(z["frames"], int)[np.asarray(z["frames"], int) < n]] = True
    M = sam3_masks(vset)[0]
    mk = [{int(k): d for k, d in zip(o["sight_k"], o["det"]) if d is not None} for o in R["objects"]]
    kset = set(int(k) for k in kfs)
    for h in range(len(z["hand_frame"])):
        k = int(z["hand_frame"][h])
        if k not in kset or int(z["contact_state"][h]) != 3 or int(z["obj_row"][h]) < 0:
            continue
        i = inst.get((k, int(z["obj_row"][h])))
        if i is None or i not in pos:
            continue
        m = OM[pos[i]]
        best, bi = 0.0, None
        for j, d in enumerate(mk):
            near = next((kk for kk in (k, k - 1, k + 1, k - 2, k + 2) if kk in d), None)
            if near is None:
                continue
            s_ = M[d[near]]; u = (m | s_).sum()
            v_ = (m & s_).sum() / u if u else 0.0
            if v_ > best:
                best, bi = v_, j
        if bi is not None and best >= 0.1:
            H[k, bi] = max(H[k, bi], float(z["contact_score"][h]))
    return H, C


def wilor(v, eps, T_all):
    """Frame -> (2, 21, 3) joints in the table frame + ok (2,), from perception_v2/hands/<ep>/hands_raw.npz (camera frame)."""
    out = {}
    for e in eps:
        p = PV2 / "hands" / e / "hands_raw.npz"
        if not p.exists():
            continue
        z = np.load(p); k0 = int(z["k0"])
        for hi, s in enumerate(("left", "right")):
            for i in range(len(z[f"{s}_ok"])):
                if not z[f"{s}_ok"][i]:
                    continue
                k = k0 + i
                J = z[f"{s}_J"][i].astype(np.float64); T = T_all[min(k, len(T_all) - 1)]
                Jt = J @ T[:3, :3].T + T[:3, 3]
                d = out.setdefault(k, [np.zeros((2, 21, 3), np.float32), np.zeros(2, np.uint8)])
                d[0][hi] = Jt; d[1][hi] = 1
    return out


def build(videos=None, out_name="tracks_all"):
    from cleantables.features import vjepa_tokens as VJ
    from cleantables.worldmodel.common import OUT
    D = json.load(open(EPISODES))
    SEG = D["segments"]
    V1 = json.load(open(EPISODES_V1))["segments"]
    SET, REG = track_sets(SEG)
    vids = sorted(set(SET.values()))
    if videos:
        vids = [x for x in vids if x in videos or any(x.startswith(v_) for v_ in videos)]
    S, stats, phr_all = [], {}, {"table"}
    for v in vids:
        if not (vdir(v) / "tracks.pkl").exists() or not (vdir(v) / "vision.npz").exists():
            print("MISSING tracks / vision for", v, flush=True); stats[v] = dict(missing=True); continue
        R = pickle.load(open(vdir(v) / "tracks.pkl", "rb"))
        Z = np.load(vdir(v) / "vision.npz")
        cemb = dict(zip(Z["det"].tolist(), Z["emb"].astype(np.float16)))
        G = geom(v); D10 = depth(v)
        ext = table_extent(G["T"], G["K"], G["table"])
        table_geo = np.array([(ext[0][0] + ext[0][1]) / 2, (ext[1][0] + ext[1][1]) / 2, 0, ext[0][1] - ext[0][0], ext[1][1] - ext[1][0], 0, 0, 1], np.float32)
        try:
            vj = VJ.load(v)
        except Exception:
            vj = None
        try:
            hg = hand_tokens(v, G, D10, np.load(vdir(v) / "hand_video.npz"))
        except Exception:
            hg = {}
        segs = [s for s in SEG if SET[s["id"]] == v]
        base = segs[0]["video"]
        v1ids = [s["id"] for s in V1 if s["video"] == base]
        st_ = next((x for x in REG.get(base, []) if x["set"] == v), None)
        hoist_ids = st_["hoist_ids"] if st_ else v1ids                 # HOIST files linked to this set's track ids
        fp_ids = v1ids if v == base else []                            # FP track ids refer to the base video's tracks.pkl only
        # FoundationPose is jumpier than the tracker during carries (p90 14 vs 9 cm) and its tracking mode can lose objects (IoU 0.04
        # in carries), so tokens, futures (T2) and end (T1) come from the SAM 3 tracker; FP only as the alternative end pose end_xyz_fp
        TL = pose_timeline(v, R, [])
        TLF = pose_timeline(v, R, fp_ids) if fp_ids else None
        HC, CC = contact_tables(base, hoist_ids, len(R["objects"]))
        WL = wilor(base, v1ids, G["T_all"])
        H23, C23 = hands23_tables(base, v, R, sorted(R["kept"]), len(HC))
        evp = vdir(v) / "events.json"
        ev_id = {int(o["src"]): int(o["id"]) for o in json.load(open(evp))["objects"]} if evp.exists() else {}
        objdets = {i: {int(k): d for k, d in zip(o["sight_k"], o["det"]) if d is not None} for i, o in enumerate(R["objects"])}
        sight = {i: np.array(sorted(objdets[i]), int) for i in objdets}
        kfs = sorted(R["kept"]); n = len(HC)
        st = stats.setdefault(v, Counter())
        for sg in segs:
            k0, k1 = int(round(sg["start_s"] * 10)), min(int(round(sg["end_s"] * 10)), n - 1)
            # end of segment: each track's last estimate inside the segment; present = SAM 3 sighting in its last 1 s
            endp = {}
            for i, (ks, X, SR) in TL.items():
                j = np.flatnonzero((ks >= k0) & (ks <= k1))
                pres = bool(((sight[i] >= k1 - 10) & (sight[i] <= k1)).any())
                yb = [(int(kk), b["yaw"]) for kk, b in zip(R["objects"][i]["sight_k"], R["objects"][i]["boxes"]) if b and k0 <= kk <= k1]
                endp[i] = (X[j[-1]] if len(j) else None, pres, int(SR[j[-1]]) if len(j) else -1, float(yb[-1][1]) if yb else None)
            endf = {}
            if TLF is not None:
                for i, (ks, X, SR) in TLF.items():
                    j = np.flatnonzero((ks >= k0) & (ks <= k1) & (SR == 1))
                    endf[i] = X[j[-1]] if len(j) else None
            for k in kfs:
                if k < k0 or k > k1:
                    continue
                dets = [i for i in R["kept"][k] if R["det_box"].get(i) and i in cemb]
                dets = sorted(dets, key=lambda i: -R["det_box"][i]["prob"])[:MAXO - 1]
                if not dets:
                    st["no_tokens"] += 1; continue
                owner = {dd[k]: oid for oid, dd in objdets.items() if k in dd}
                tok = [owner.get(i, -1) for i in dets]
                s = dict(obj_geo=np.zeros((MAXO, 8), np.float32), obj_txt=np.zeros((MAXO, 384), np.float16),
                         obj_crop=np.zeros((MAXO, 768), np.float16), obj_crop_ok=np.zeros(MAXO, np.uint8), obj_mask=np.zeros(MAXO, np.uint8),
                         obj_kind=np.full(MAXO, -1, np.int8), obj_id=np.full(MAXO, -1, np.int32), obj_pose_src=np.zeros(MAXO, np.int8),
                         hand_geo=np.zeros((2, 13), np.float32), wilor_J=np.zeros((2, 21, 3), np.float32), wilor_ok=np.zeros(2, np.uint8),
                         obj_contact=np.zeros(MAXO, np.float32), contact_ok=np.uint8(0),
                         hands23_contact=np.zeros(MAXO, np.float32), hands23_ok=np.uint8(0),
                         fut_xyz=np.zeros((MAXO, NF, 3), np.float32), fut_ok=np.zeros((MAXO, NF), np.uint8),
                         fut_src=np.full((MAXO, NF), -1, np.int8), end_src=np.full(MAXO, -1, np.int8), end_yaw=np.zeros(MAXO, np.float32),
                         end_yaw_ok=np.zeros(MAXO, np.uint8), events_obj_id=np.full(MAXO, -1, np.int32),
                         end_xyz_fp=np.zeros((MAXO, 3), np.float32), end_fp_ok=np.zeros(MAXO, np.uint8),
                         fut_contact=np.zeros((MAXO, NF), np.float32), fut_contact_ok=np.zeros(NF, np.uint8),
                         end_xyz=np.zeros((MAXO, 3), np.float32), end_ok=np.zeros(MAXO, np.uint8), end_present=np.zeros(MAXO, np.uint8),
                         vj_tok=np.zeros((32, 1024), np.float16), vj_tok_flip=np.zeros((32, 1024), np.float16), vj_ok=np.uint8(0),
                         table_idx=np.int16(-1), snap_k=np.int32(k), seg_kind=np.int8(0 if sg["kind"] == "episode" else 1),
                         seg_t0=np.float32(sg["start_s"]), seg_t1=np.float32(sg["end_s"]),
                         t_in_seg=np.float32((k - k0) / max(1, k1 - k0)))
                for j, i in enumerate(dets):
                    b = R["det_box"][i]
                    s["obj_geo"][j] = box_feat(b); s["obj_crop"][j] = cemb[i]; s["obj_crop_ok"][j] = 1
                    s["obj_mask"][j] = 1; s["obj_kind"][j] = KIND["movable"]; s["obj_id"][j] = tok[j]
                    phr_all.add(b["phrase"])
                    t = tok[j]
                    if t < 0:
                        continue
                    p, src = at(TL[t], k)
                    if p is not None and src == 1:
                        s["obj_geo"][j, 0], s["obj_geo"][j, 1] = p[0], p[1]; s["obj_pose_src"][j] = 1
                    s["obj_contact"][j] = HC[k, t]
                    s["hands23_contact"][j] = H23[k, t]
                    for f, dt in enumerate(FUT_DT):
                        kf = k + int(round(dt * 10))
                        q, qs = at(TL[t], kf)
                        if q is not None:
                            s["fut_xyz"][j, f] = q; s["fut_ok"][j, f] = 1; s["fut_src"][j, f] = qs
                        if kf < n and CC[kf]:
                            s["fut_contact"][j, f] = HC[kf, t]
                    e_, pres, esrc, eyaw = endp[t]
                    if e_ is not None:
                        s["end_xyz"][j] = e_; s["end_ok"][j] = 1; s["end_src"][j] = esrc
                    if eyaw is not None:
                        s["end_yaw"][j] = eyaw; s["end_yaw_ok"][j] = 1
                    s["events_obj_id"][j] = ev_id.get(t, -1)
                    if endf.get(t) is not None:
                        s["end_xyz_fp"][j] = endf[t]; s["end_fp_ok"][j] = 1
                    s["end_present"][j] = int(pres)
                nn = len(dets)
                s["_phr"] = [R["det_box"][i]["phrase"] for i in dets] + ["table"]
                s["obj_geo"][nn] = table_geo; s["obj_mask"][nn] = 1; s["obj_kind"][nn] = KIND["table"]; s["table_idx"] = np.int16(nn)
                s["contact_ok"] = np.uint8(CC[k]); s["hands23_ok"] = np.uint8(C23[k])
                for f, dt in enumerate(FUT_DT):
                    kf = k + int(round(dt * 10)); s["fut_contact_ok"][f] = int(kf < n and CC[kf])
                if k in hg:
                    s["hand_geo"][:] = hg[k]
                if k in WL:
                    s["wilor_J"][:], s["wilor_ok"][:] = WL[k]
                if vj is not None:
                    a = VJ.nearest_keyframe(v, k, vj)
                    s["vj_tok"][:] = vj["tok"][a]; s["vj_tok_flip"][:] = vj["tok_f"][a]; s["vj_ok"] = np.uint8(1)
                s.update(video=sg["video"], track_set=v, group=sg["id"], task_text=sg["instruction"])
                S.append(s)
                st["samples"] += 1; st[f"samples_{sg['kind']}"] += 1
                st["contact_ok"] += int(CC[k]); st["fp_tokens"] += int(s["obj_pose_src"].sum()); st["tokens"] += nn
                st["wilor_ok"] += int(s["wilor_ok"].any())
        print(v, dict(st), flush=True)
    E = dict(zip(sorted(phr_all), embed(sorted(phr_all))))
    for s in S:
        for j, p in enumerate(s.pop("_phr")):
            s["obj_txt"][j] = E[p]
    keys = [k for k in S[0] if k not in ("video", "track_set", "group", "task_text")]
    out = {k: np.stack([np.asarray(s[k]) for s in S]) for k in keys}
    for k in ("video", "track_set", "group", "task_text"):
        out[k] = np.array([s[k] for s in S])
    out["fut_dt"] = np.array(FUT_DT, np.float32)
    d = OUT / "wm_v3_tracks"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{out_name}.npz", **out)
    tot = Counter()
    for c in stats.values():
        if isinstance(c, Counter):
            tot.update(c)
    rows = OUT / "wm_v2" / "feats" / "rows.json"
    have = {(a, int(b)) for a, b in json.load(open(rows))["rows"]} if rows.exists() else set()
    new_kf = Counter(str(v_) for v_, k_ in zip(out["video"], out["snap_k"]) if (str(v_), int(k_)) not in have)
    meta = dict(n=len(S), keyframes_not_in_wm_v2_feats=dict(new_kf), keys={k: [list(v.shape), str(v.dtype)] for k, v in out.items()}, per_video={v: dict(c) for v, c in stats.items()},
                totals=dict(tot), fut_dt=FUT_DT, source=str(EPISODES),
                notes=["no move / mover / target / relation / path keys: the WM predicts every object's future",
                       "obj_contact is unknown (not 'free') where contact_ok == 0",
                       "x, y from FoundationPose where obj_pose_src == 1 (role objects inside episode windows only), else the SAM 3 tracker",
                       "wilor_J: WiLoR joints (camera frame from hands_raw.npz) moved to the table frame with T_all; only where WiLoR ran"])
    json.dump(meta, open(d / f"{out_name}_meta.json", "w"), indent=1)
    print(json.dumps(dict(n=len(S), totals=dict(tot)), indent=1))


if __name__ == "__main__":
    build([a for a in sys.argv[1:] if a.startswith("IMG_")] or None)
