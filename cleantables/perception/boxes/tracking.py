"""Per-video box state estimation: SAM 3 box masks + MoGe depth -> table frame -> per-keyframe 3D box detections -> REST SEGMENTS (a box
sitting still at one pose) -> CoTracker follow of every segment end (did the box move, and where to) -> box identities.

stage geom     : table plane / frame (geometry.fit_table on frames with every SAM mask removed), table mask, per-frame depth scale, and
                 for every kept SAM 3 box detection the 3D box pose (geometry.box_from_points: top-face centre, top height, L, W, yaw) and
                 whether a hand ('hand' SAM 3 masks) touches it.                                   -> <OUT>/<v>/det3d.npz, table.json
stage segments : rest segments = Hungarian tracking of detections over keyframes with tight gates (3.5 cm in xy, 2.5 cm in top height,
                 6/5 cm in L/W) and up to GAP_S unseen (hand / arm occlusion). >= 3 sightings over >= 1 s; a carried box (moving) never
                 forms one. Pose = median over untouched sightings (circular mean for yaw).       -> <OUT>/<v>/segments.json
stage follow   : for every segment that ends before the video does, ~40 CoTracker3 points seeded on its top face (mask pixels whose 3D height
                 is within 2 cm of its top) at its last good sighting are followed forward for up to FOLLOW_S (chained 60-frame windows,
                 re-seeded with the points still tracked). The median point, lifted with the every-frame depth, is the box's 3D path.
                 Classified 'static' (never moved > 4 cm: the box got covered / SAM lost it) or 'moved' (+ motion start / end frames).
                                                                                                   -> <OUT>/<v>/follow.npz
Identity (cleantables.perception.boxes.events) = segments chained by follow results ('moved' -> the next segment holding the followed points; 'static' -> the next
segment at the same place), with a size-matching fallback when CoTracker loses the box.
"""
from __future__ import annotations

import os
import json
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cleantables.perception.scene.common import WORK, meta, read_frame, unpack
from cleantables.perception.scene.tracks import mask_iou

from cleantables.perception.boxes import geometry as G
from cleantables.perception.boxes.common import BOX_PHRASES, HAND_PHRASES, TAPE_PHRASES, jdump, log, vdir

P_KEEP, P_HAND, P_TAPE, NMS_IOU = 0.40, 0.40, 0.35, 0.6
GAP_S = 8.0
FOLLOW_S, FWIN = 15.0, 24      # 24-frame windows at half resolution: CoTracker3 peak ~1.5 GB (60 frames at full res need 4.6 GB)
MIN_H = 0.03
MIN_W = 0.05


# ------------------------------------------------------------------ loading
def load_sam(video):
    z = np.load(WORK / video / "sam3_box.npz")
    m = int(z["shape"][0])
    return dict(frame=z["frame"], prompt=z["prompt"], prob=z["prob"], box=z["box"],
                masks=unpack(z["masks"], (m, 270, 480)) if m else np.zeros((0, 270, 480), bool),
                prompts=[str(x) for x in z["prompts"]], frames=[int(x) for x in z["frames_done"]])


def load_depth10(video):
    return np.load(vdir(video) / "depth10.npy", mmap_mode="r")


def load_K(video):
    return np.load(WORK / video / "depth.npz")["K"]


# ------------------------------------------------------------------ stage geom
def stage_geom(video):
    t0 = time.time()
    d = load_sam(video)
    D10 = load_depth10(video)
    K = load_K(video)
    P = d["prompts"]
    is_box = np.array([P[p] in BOX_PHRASES or P[p] not in HAND_PHRASES + TAPE_PHRASES for p in d["prompt"]], bool)
    is_hand = np.array([P[p] in HAND_PHRASES for p in d["prompt"]], bool)
    is_tape = np.array([P[p] in TAPE_PHRASES for p in d["prompt"]], bool)
    frames = d["frames"]
    by_f = {}
    for i, k in enumerate(d["frame"]):
        by_f.setdefault(int(k), []).append(i)
    # ---- table plane from 15 keyframes with every detection (dilated) removed
    sel = [frames[i] for i in np.linspace(0, len(frames) - 1, min(15, len(frames))).astype(int)]
    ker = np.ones((9, 9), np.uint8)
    excl, deps = [], []
    for k in sel:
        e = np.zeros((270, 480), np.uint8)
        for i in by_f.get(k, []):
            if d["prob"][i] >= 0.3:
                e |= d["masks"][i].astype(np.uint8)
        excl.append(cv2.dilate(e, ker) > 0)
        deps.append(D10[k].astype(np.float32))
    up = (1.0, 0, 0) if meta(video).get("rotated_portrait") else (0, -1.0, 0)
    T, n, c, info = G.fit_table(deps, K, excl, up=up)
    # ---- bump check (the tripod may be knocked): re-fit the plane in 6 time windows; a window whose plane tilts > 2.5 deg from its
    # predecessor starts a new camera segment with its own T, re-fit on that segment's frames. Heights are not compared because the
    # MoGe scale varies per frame.
    nfr = meta(video)["n"]
    table0 = G.table_mask(deps, K, T, frac=0.12)                    # window fits use only the global table region, so the person does not bias them
    W_ = 6
    wins = np.array_split(np.array(frames), W_)
    wfit = []
    for wk in wins:
        ks_ = [int(x) for x in wk[np.linspace(0, len(wk) - 1, min(4, len(wk))).astype(int)]] if len(wk) else []
        if not ks_:
            wfit.append(None); continue
        ex_ = []
        for k in ks_:
            e = np.zeros((270, 480), np.uint8)
            for i in by_f.get(k, []):
                if d["prob"][i] >= 0.3:
                    e |= d["masks"][i].astype(np.uint8)
            ex_.append((cv2.dilate(e, ker) > 0) | ~table0)
        try:
            Tw, nw_, cw_, iw = G.fit_table([D10[k].astype(np.float32) for k in ks_], K, ex_, up=up)
            wfit.append((Tw, ks_, ex_))
        except RuntimeError:
            wfit.append(None)
    def dT(a, b):
        ang = float(np.degrees(np.arccos(np.clip(a[2, :3] @ b[2, :3], -1, 1))))
        return ang, abs(float(a[2, 3] - b[2, 3]))
    seg_id, cur, bumps, wdiff = [], 0, [], []
    for wi in range(W_):
        if wi > 0 and wfit[wi] is not None and wfit[wi - 1] is not None:
            ang, dh = dT(wfit[wi][0], wfit[wi - 1][0])
            wdiff.append([round(ang, 2), round(dh * 100, 2)])
            if ang > 2.5:                                   # MoGe per-frame scale makes the height unusable here
                cur += 1; bumps.append(int(wins[wi][0]))
        seg_id.append(cur)
    T_all = np.repeat(T[None], nfr, 0)
    if bumps:                                              # piecewise T: re-fit each camera segment on its own frames
        for sid in sorted(set(seg_id)):
            ws = [wi for wi in range(W_) if seg_id[wi] == sid and wfit[wi] is not None]
            ks_ = sum([wfit[wi][1] for wi in ws], []); ex_ = sum([wfit[wi][2] for wi in ws], [])
            Ts, _, _, _ = G.fit_table([D10[k].astype(np.float32) for k in ks_], K, ex_, up=up)
            lo = int(wins[min(wi for wi in range(W_) if seg_id[wi] == sid)][0])
            hi = nfr if sid == max(seg_id) else int(wins[min(wi for wi in range(W_) if seg_id[wi] == sid + 1)][0])
            if sid == 0:
                lo = 0
            T_all[lo:hi] = Ts
        T = T_all[0]
    info.update(bump_frames=bumps, window_plane_diff_deg_cm=wdiff)
    table = G.table_mask(deps, K, T, frac=0.12)        # low: the person hides parts of the table in many frames
    zp = G.plane_z_map(T, K)
    zps = {}
    f = K[0, 0]
    rows, tape = [], []
    scale = np.ones(meta(video)["n"], np.float32)
    for k in frames:
        ids = by_f.get(k, [])
        T = T_all[k]
        if bumps:
            key = int(np.argmax([np.allclose(T, T_all[b_]) for b_ in [0] + bumps]))
            if key not in zps:
                zps[key] = G.plane_z_map(T, K)
            zp = zps[key]
        D = D10[k].astype(np.float32)
        allm = np.zeros((270, 480), bool)
        hand = np.zeros((270, 480), bool)
        for i in ids:
            allm |= d["masks"][i]
            if is_hand[i] and d["prob"][i] >= P_HAND:
                hand |= d["masks"][i]
        s = G.depth_scale(D, zp, table & ~cv2.dilate(allm.astype(np.uint8), ker).astype(bool))
        scale[k] = s
        D = D * s
        handd = cv2.dilate(hand.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        # boxes: on-table, not mostly hand, cross-phrase NMS
        cand = []
        for i in ids:
            if not is_box[i] or d["prob"][i] < P_KEEP:
                continue
            mk = d["masks"][i]
            a = mk.sum()
            if a < 30 or a > 0.25 * mk.size:
                continue
            ys, xs = np.nonzero(mk)
            if not table[int(ys.mean()), int(xs.mean())]:
                continue
            if (mk & hand).sum() > 0.5 * a:
                continue
            cand.append(i)
        cand.sort(key=lambda i: -d["prob"][i])
        keep = []
        for i in cand:
            if all(mask_iou(d["masks"][i], d["masks"][j]) < NMS_IOU for j in keep):
                keep.append(i)
        dets = []
        lab = cv2.cvtColor(cv2.resize(read_frame(video, k), (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32) if keep else None
        for i in keep:
            mk = d["masks"][i]
            Q = G.lift_mask(D, K, T, mk)
            if len(Q) < 25:
                continue
            zc = float(np.median(D[mk & (D > 0)])) if (mk & (D > 0)).any() else 1.0
            b = G.box_from_points(Q, zc, f)
            if b is None:
                continue
            # appearance + full 3D extent of the mask (side faces included), used to read the final stack level by level (snapshot)
            me = cv2.erode(mk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            me = me if me.sum() >= 10 else mk
            col = np.median(lab[me], 0)
            b["extra"] = [float(np.percentile(Q[:, 2], 5)), float(np.percentile(Q[:, 2], 95)), float(np.median(Q[:, 0])),
                          float(np.median(Q[:, 1])), float(col[0]), float(col[1]), float(col[2])]
            touched = bool((handd & cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).sum() > 15)
            dets.append((i, b, touched))
        # 3D de-duplication: a whole-stack mask and the top box's own mask give the same top-face pose -> keep the more probable one
        out = []
        for i, b, t in sorted(dets, key=lambda x: -d["prob"][x[0]]):
            if b["ok"] and any(o[1]["ok"] and np.hypot(o[1]["x"] - b["x"], o[1]["y"] - b["y"]) < 0.03 and abs(o[1]["top"] - b["top"]) < 0.02
                               for o in out):
                continue
            out.append((i, b, t))
        for i, b, t in out:
            rows.append([k, i, d["prob"][i], b["x"], b["y"], b["top"], b["L"], b["W"], b["yaw"], float(b["ok"]), float(t), b["n_top"]] + b["extra"])
        # tape (the stack square / carry spot): flat detections
        for i in ids:
            if is_tape[i] and d["prob"][i] >= P_TAPE:
                Q = G.lift_mask(D, K, T, d["masks"][i], erode=0)
                if len(Q) >= 20 and np.percentile(Q[:, 2], 90) < 0.02:
                    x, y, L, W, yaw = G.rect_extent(Q[:, :2])
                    tape.append([k, i, d["prob"][i], x, y, L, W, yaw])
    # every-frame depth scale (for the follow stage): nearest keyframe's free-table pixels
    kf = np.array(frames)
    for k in range(len(scale)):
        if k in by_f:
            continue
        kn = int(kf[np.argmin(np.abs(kf - k))])
        free = table.copy()
        for i in by_f.get(kn, []):
            free &= ~d["masks"][i]
        scale[k] = G.depth_scale(D10[k].astype(np.float32), G.plane_z_map(T_all[k], K) if bumps else zp, free)
    R = np.array(rows, np.float64).reshape(-1, 19)
    np.savez_compressed(vdir(video) / "det3d.npz", rows=R, cols=np.array(["k", "det", "prob", "x", "y", "top", "L", "W", "yaw", "ok", "touched", "n_top",
                                                                         "zlo", "zhi", "xa", "ya", "colL", "cola", "colb"]),
                        tape=np.array(tape, np.float64).reshape(-1, 8), T=T_all[0], T_all=T_all, K=K, table=table, scale=scale)
    tb = dict(video=video, T_cam_to_table=T_all[0].tolist(), K=K.tolist(), **info, n_keyframes=len(frames), n_box_dets=int(len(R)),
              n_box_dets_ok=int(R[:, 9].sum()) if len(R) else 0, n_tape_dets=len(tape), wall_s=round(time.time() - t0, 1))
    jdump(tb, vdir(video) / "table.json")
    log(f"{video} geom: camera {info['camera_height_m']} m above the table, tilt {info['tilt_deg']} deg; {len(R)} box detections "
        f"({tb['n_box_dets_ok']} with a 3D box), {len(tape)} tape detections, {time.time() - t0:.0f}s")
    return tb


# ------------------------------------------------------------------ stage segments
def circ_mean_pi(a):
    a = np.asarray(a)
    return float(np.arctan2(np.sin(2 * a).mean(), np.cos(2 * a).mean()) / 2 % np.pi)


def stage_segments(video):
    Z = np.load(vdir(video) / "det3d.npz")
    R = Z["rows"]
    fps = meta(video)["fps"]
    R = R[(R[:, 9] > 0) & (R[:, 5] >= MIN_H) & (R[:, 7] >= MIN_W)]     # thin top faces = side faces of a covered box, not a box
    frames = sorted(set(R[:, 0].astype(int).tolist()))
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 5
    gap = int(round(GAP_S * fps))
    segs, active = [], []
    for k in frames:
        idx = np.nonzero(R[:, 0] == k)[0]
        cand = [s for s in active if k - s["last"] <= gap]
        used = set()
        if cand and len(idx):
            C = np.full((len(cand), len(idx)), 9.0)
            for a, s in enumerate(cand):
                o = np.array(s["obs"][-5:])
                sx, sy, st, sL, sW = np.median(o[:, 3]), np.median(o[:, 4]), np.median(o[:, 5]), np.median(o[:, 6]), np.median(o[:, 7])
                for b, r in enumerate(idx):
                    x = R[r]
                    dxy, dt, dL, dW = np.hypot(x[3] - sx, x[4] - sy), abs(x[5] - st), abs(x[6] - sL), abs(x[7] - sW)
                    if dxy < 0.035 and dt < 0.025 and dL < 0.06 and dW < 0.05:
                        C[a, b] = dxy / 0.035 + dt / 0.025 + 0.3 * (dL + dW) / 0.05
            ra, cb = linear_sum_assignment(C)
            for a, b in zip(ra, cb):
                if C[a, b] < 9:
                    cand[a]["obs"].append(R[idx[b]].tolist()); cand[a]["last"] = k; used.add(b)
        for b, r in enumerate(idx):
            if b not in used:
                s = dict(obs=[R[r].tolist()], last=k)
                segs.append(s); active.append(s)
        active = [s for s in active if k - s["last"] <= gap]
    # duplicates: two tracks of the same box at the same time (two SAM masks that survived NMS, e.g. a box + its open flaps) -> merge
    segs = [s for s in segs if len(s["obs"]) >= 3]
    segs.sort(key=lambda s: -len(s["obs"]))
    kept = []
    for s in segs:
        o = np.array(s["obs"])
        dup = None
        for t in kept:
            q = np.array(t["obs"])
            ov = min(o[-1, 0], q[-1, 0]) - max(o[0, 0], q[0, 0])
            if ov > 0 and np.hypot(np.median(o[:, 3]) - np.median(q[:, 3]), np.median(o[:, 4]) - np.median(q[:, 4])) < 0.05 \
                    and abs(np.median(o[:, 5]) - np.median(q[:, 5])) < 0.03:
                dup = t
                break
        if dup is None:
            kept.append(s)
        else:
            have = {r[0] for r in dup["obs"]}
            dup["obs"] = sorted(dup["obs"] + [r for r in s["obs"] if r[0] not in have], key=lambda r: r[0])
    segs = sorted(kept, key=lambda s: s["obs"][0][0])
    out = []
    for s in segs:
        o = np.array(s["obs"])
        if len(o) < 3 or o[-1, 0] - o[0, 0] < fps:
            continue
        u = o[o[:, 10] == 0] if (o[:, 10] == 0).sum() >= 2 else o
        out.append(dict(id=len(out), k0=int(o[0, 0]), k1=int(o[-1, 0]), n=int(len(o)), n_untouched=int((o[:, 10] == 0).sum()),
                        x=float(np.median(u[:, 3])), y=float(np.median(u[:, 4])), top=float(np.median(u[:, 5])), L=float(np.median(u[:, 6])),
                        W=float(np.median(u[:, 7])), yaw=circ_mean_pi(u[:, 8]), sd_xy_cm=round(float(np.std(u[:, 3]) + np.std(u[:, 4])) * 100, 2),
                        sd_L_cm=round(float(np.std(u[:, 6])) * 100, 2), dets=[[int(a), int(b)] for a, b in zip(o[:, 0], o[:, 1])],
                        seed=[int(u[-1, 0]), int(u[-1, 1])]))
    jdump(dict(video=video, keyframe_step=step, segments=out), vdir(video) / "segments.json")
    log(f"{video} segments: {len(out)} rest segments (from {len(segs)} raw tracks)")
    return out


# ------------------------------------------------------------------ stage follow (CoTracker3)
_COT = {}


def cotracker(dev="cuda"):
    import os

    import torch
    from cleantables.perception.flow.cotrack import ROOT
    os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))
    if "m" not in _COT:
        _COT["m"] = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    return _COT["m"]


def follow_points(video, k0, k1, q_xy, dev="cuda"):
    """Track points (P,2 work px) visible at frame k0 forward to k1 (chained FWIN-frame windows). -> uv (k1-k0+1, P, 2), vis (.., P)"""
    import torch
    m = cotracker(dev)
    n = k1 - k0 + 1
    UV = np.full((n, len(q_xy), 2), np.nan, np.float32); V = np.zeros((n, len(q_xy)), bool)
    cur = np.asarray(q_xy, np.float32); UV[0] = cur; V[0] = True
    a = k0
    while a < k1:
        b = min(k1, a + FWIN - 1)
        fr = np.stack([cv2.resize(read_frame(video, k), (480, 270), interpolation=cv2.INTER_AREA)[..., ::-1] for k in range(a, b + 1)])
        vid = torch.from_numpy(fr).permute(0, 3, 1, 2)[None].float().to(dev)
        q = torch.from_numpy(np.c_[np.zeros(len(cur)), cur / 2].astype(np.float32))[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            tr, vis = m(vid, queries=q)
        tr = tr[0].float().cpu().numpy() * 2; vis = (vis[0].float().cpu().numpy() > 0.5)
        UV[a - k0:b - k0 + 1] = tr; V[a - k0:b - k0 + 1] = vis
        cur = tr[-1]
        del vid
        if b == k1:
            break
        a = b
    return UV, V


def top_face_seeds(video, k, det, T, K, scale, D10, d, nq=40, rng=None):
    mk = cv2.erode(d["masks"][det].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if mk.sum() < 10:
        mk = d["masks"][det]
    D = D10[k].astype(np.float32) * scale[k]
    P, (ys, xs) = G.backproject(D, K, mk)
    if not len(P):
        return np.zeros((0, 2), np.float32)
    z = G.to_table(T[k] if T.ndim == 3 else T, P)[:, 2]
    top = np.percentile(z, 90)
    sel = np.abs(z - top) < 0.02
    if sel.sum() < 8:
        sel = np.ones(len(z), bool)
    ys, xs = ys[sel], xs[sel]
    rng = rng or np.random.default_rng(0)
    j = rng.choice(len(xs), min(nq, len(xs)), replace=False)
    return np.c_[xs[j] * 2 + 1.0, ys[j] * 2 + 1.0].astype(np.float32)


def lift_uv(uv, k, T, K, D10, scale):
    u, v = uv
    if not (0 <= u < 960 and 0 <= v < 540):
        return np.full(3, np.nan)
    D = D10[k]
    z = np.median(D[max(0, int(v / 2) - 2):int(v / 2) + 3, max(0, int(u / 2) - 2):int(u / 2) + 3].astype(np.float32)) * scale[k]
    if not z > 0:
        return np.full(3, np.nan)
    pc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    return G.to_table(T[k] if T.ndim == 3 else T, pc[None])[0]


def stage_follow(video, dev="cuda"):
    if dev == "cpu":
        import torch
        torch.set_num_threads(8)
    segs = json.load(open(vdir(video) / "segments.json"))["segments"]
    Z = np.load(vdir(video) / "det3d.npz")
    T, K, scale = (Z["T_all"] if "T_all" in Z else Z["T"]), Z["K"], Z["scale"]
    D10 = load_depth10(video)
    d = load_sam(video)
    n = meta(video)["n"]
    fps = meta(video)["fps"]
    rng = np.random.default_rng(0)
    res = {}
    t0 = time.time()
    last_k = max(s["k1"] for s in segs) if segs else 0
    for s in segs:
        if s["k1"] >= last_k - 5:                         # still there at the end of the video
            continue
        k, det = s["seed"]
        q = top_face_seeds(video, k, det, T, K, scale, D10, d, rng=rng)
        if len(q) < 5:
            continue
        k1 = min(n - 1, k + int(FOLLOW_S * fps))
        UV, V = follow_points(video, k, k1, q, dev)
        med = np.nanmedian(UV, 1)                         # (t, 2): CoTracker also estimates occluded points (box inside the hands)
        P3 = np.array([lift_uv(med[t], k + t, T, K, D10, scale) for t in range(len(med))])
        p0 = np.array([s["x"], s["y"], s["top"]])
        disp = np.linalg.norm(P3[:, :2] - p0[:2], axis=1)
        moved = bool(np.nanmax(np.r_[disp, 0]) > 0.06)
        res[s["id"]] = dict(k=k, uv=UV.astype(np.float16), vis=V, med=med, P3=P3, moved=moved, vis_frac=float(V.mean()))
    np.savez_compressed(vdir(video) / "follow.npz", ids=np.array(list(res), np.int32),
                        k=np.array([r["k"] for r in res.values()], np.int32),
                        uv=np.array([r["uv"] for r in res.values()] + [None], dtype=object)[:-1],
                        vis=np.array([r["vis"] for r in res.values()] + [None], dtype=object)[:-1],
                        med=np.array([r["med"] for r in res.values()] + [None], dtype=object)[:-1],
                        P3=np.array([r["P3"] for r in res.values()] + [None], dtype=object)[:-1],
                        moved=np.array([r["moved"] for r in res.values()], bool), vis_frac=np.array([r["vis_frac"] for r in res.values()]))
    _COT.clear()                                          # free the GPU for the other stages / jobs
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    log(f"{video} follow: {len(res)} segment ends tracked with CoTracker3, {sum(r['moved'] for r in res.values())} moved, {time.time() - t0:.0f}s")
    return res


def load_follow(video):
    p = vdir(video) / "follow.npz"
    if not p.exists():
        return {}
    Z = np.load(p, allow_pickle=True)
    return {int(i): dict(k=int(k), med=m, P3=P, moved=bool(mv), vis_frac=float(vf), uv=uv, vis=vs)
            for i, k, m, P, mv, vf, uv, vs in zip(Z["ids"], Z["k"], Z["med"], Z["P3"], Z["moved"], Z["vis_frac"], Z["uv"], Z["vis"])}
