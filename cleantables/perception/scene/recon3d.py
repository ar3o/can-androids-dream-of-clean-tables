"""Per-episode 3D: table plane, metric scale, 3D hand joints (10 fps) and object centroid tracks (SAM 3 keyframes).

Depth: MoGe-3 (data/table_clean/_work/<video>/depth.npz) when present, else depth_cpu.npz (MoGe-2 ViT-S on the CPU, keyframes).
RANSAC table plane on table pixels with hands and objects masked; per-frame depth scale aligned to the reference plane.
Hands: MediaPipe 21 joints (same joint order as MANO/OpenPose). 3D = MediaPipe's metric, camera-axis-aligned world landmarks x one
per-video hand scale s, translated (linear least squares, solve_t) so they project onto the 2D landmarks with our K; s = median over
keyframes of (MoGe palm depth + 1 cm skin offset) / (palm depth at s = 1). The rotation is not re-estimated: cv2.solvePnP with a free
rotation put hands below the table in 31 % of frames on one video.
Objects: per keyframe, the eroded SAM 3 mask's pixels lifted with the aligned depth; centroid (median) in the table frame, height of
the top surface (p95) above the table, footprint L x W (trimmed oriented rectangle of the points projected on the table).
Table frame: z = plane normal (up), origin = where the principal ray meets the plane, x = image-right projected on the plane.
Out: data/table_clean/3d/<episode>.npz + <episode>.json (summary), proof/table_clean/3d_<video>.jpg (top view).
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from cleantables.perception.scene.common import OUT, WORK, meta
from cleantables.perception.scene.tracks import H, W, build_tracks

PALM = [0, 5, 9, 13, 17]


def load_depth(video):
    p = WORK / video / "depth.npz"
    if p.exists():                       # MoGe-3 (GPU) depth when present
        z = np.load(p)
        return dict(k=z["k"], depth=z["depth"].astype(np.float32), K=z["K"], model=str(z["model"]))
    z = np.load(WORK / video / "depth_cpu.npz")
    return dict(k=z["k"], depth=z["depth"].astype(np.float32), K=z["K"], model=str(z["model"]))


def backproject(depth_lo, K, mask_lo=None):
    """Points (n,3) camera frame for (masked) pixels of a 270x480 depth map; K is at 960x540."""
    ys, xs = np.nonzero((depth_lo > 0) & (mask_lo if mask_lo is not None else True))
    z = depth_lo[ys, xs]
    u, v = xs * 2 + 0.5, ys * 2 + 0.5
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1), (ys, xs)


def solve_t(J, uv, K):
    """Translation t (camera frame) so that the camera-axis-aligned metric joints J (21,3) + t project onto uv (21,2) with K.
    Linear least squares: fx (X + tx) - (u - cx)(Z + tz) = 0, fy (Y + ty) - (v - cy)(Z + tz) = 0."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    A = np.zeros((2 * len(J), 3))
    b = np.zeros(2 * len(J))
    A[0::2, 0] = fx; A[0::2, 2] = -du; b[0::2] = du * J[:, 2] - fx * J[:, 0]
    A[1::2, 1] = fy; A[1::2, 2] = -dv; b[1::2] = dv * J[:, 2] - fy * J[:, 1]
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


def mano_hands(video, T, K, dz, scale, n):
    """WiLoR MANO hands (hand_wilor.npz): per frame and handedness the detector-consistent regression with the highest confidence;
    MANO metric joints (WiLoR orientation kept) translated by solve_t onto WiLoR's own 2D joints with our K.
    Depth: as in cleantables.perception.hands.lift, one per-video scale s along the camera rays (s = median over depth keyframes of MoGe palm depth + 1 cm /
    MANO palm depth; scaling about the camera centre keeps the 2D projection), then a per-frame depth offset from the 2 fps depth
    keyframes (gated, interpolated, used within 1 s). Depth from MANO size alone is 6-12 cm too close (median).
    -> (J (n,2,21,3) table frame, slot 0 = left, 1 = right; conf (n,2); dict(s, residual after s at keyframes))"""
    p = WORK / video / "hand_wilor.npz"
    if not p.exists():
        return None, None, {}
    J = np.full((n, 2, 21, 3), np.nan, np.float32)
    conf = np.zeros((n, 2), np.float32)
    z = np.load(p)
    kd = {int(k): i for i, k in enumerate(dz["k"])}
    best = {}
    for r in np.where(z["det_is_right"] == z["is_right"])[0]:
        k, h = int(z["frame"][r]), int(z["is_right"][r])
        if k < n and z["det_conf"][r] > best.get((k, h), (0, None))[0]:
            best[(k, h)] = (float(z["det_conf"][r]), r)
    Pcs, ratios, keyz, keyoff, keyidx = {}, [], [], {}, set()
    for (k, h), (c, r) in best.items():
        Jm = z["joints_mano"][r].astype(np.float64)
        uv = z["kp2d"][r].astype(np.float64)
        Pc = Jm + solve_t(Jm, uv, K)
        Pcs[(k, h)] = (Pc, c)
        if k in kd:
            D = dz["depth"][kd[k]] * scale[k]
            zs = [D[int(v / 2), int(u / 2)] for u, v in uv[PALM] if 0 <= u < 960 and 0 <= v < 540 and D[int(v / 2), int(u / 2)] > 0]
            if len(zs) >= 3 and Pc[PALM, 2].mean() > 0:
                ratios.append((np.median(zs) + 0.01) / Pc[PALM, 2].mean())
                keyz.append((np.median(zs) + 0.01, Pc[PALM, 2].mean()))
                keyoff[(k, h)] = keyz[-1]
    s = float(np.median(ratios)) if ratios else 1.0
    # per-frame depth offset (as hands.lift): at depth keyframes dz = MoGe palm z - s * MANO palm z, gated (> 8 cm from the 5-keyframe
    # median = glitch), interpolated over time per hand (only within 1 s of a keyframe), applied by scaling the hand along the camera rays
    from scipy.ndimage import median_filter
    off = {0: {}, 1: {}}
    for (k, h), (a, b) in keyoff.items():
        off[h][k] = a - s * b
    for h in (0, 1):
        if len(off[h]) < 3:
            continue
        ks_ = np.array(sorted(off[h]))
        v = np.array([off[h][k] for k in ks_])
        med = median_filter(v, 5, mode="nearest")
        good = np.abs(v - med) <= 0.08
        ks_, v = ks_[good], median_filter(v[good], 3, mode="nearest")
        off[h] = (ks_, v)
    for (k, h), (Pc, c) in Pcs.items():
        P = Pc * s
        if isinstance(off[h], tuple):
            ks_, v = off[h]
            j = np.searchsorted(ks_, k)
            near_ = min([abs(ks_[i] - k) for i in (j - 1, j) if 0 <= i < len(ks_)] + [999])
            if near_ <= 10:
                dzk = float(np.interp(k, ks_, v))
                zc = P[PALM, 2].mean()
                P = P * (zc + dzk) / zc
        J[k, h] = to_table(T, P)
        conf[k, h] = c
    res = np.array([a - s * b for a, b in keyz]) if keyz else np.zeros(0)
    info = dict(s=round(s, 3), n_keyframes=len(keyz),
                raw_offset_median_m=round(float(np.median([a - b for a, b in keyz])), 3) if keyz else None,
                residual_after_s_m=dict(median=round(float(np.median(res)), 3), p10=round(float(np.percentile(res, 10)), 3),
                                        p90=round(float(np.percentile(res, 90)), 3)) if len(res) else None)
    return J, conf, info


def ransac_plane(P, it=400, thr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(it):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        cnt = (np.abs((P - a) @ n) < thr).sum()
        if cnt > best[0]:
            best = (cnt, (n, a))
    n, a = best[1]
    inl = np.abs((P - a) @ n) < thr
    c = P[inl].mean(0)
    n = np.linalg.svd(P[inl] - c)[2][2]
    if n @ c > 0:            # normal pointing toward the camera (camera at origin above the table)
        n = -n
    return n, c, float(inl.mean())


def table_frame(n, c, K):
    """4x4 camera->table transform. z = n (up toward the camera side), origin = principal ray x plane."""
    ray = np.array([0, 0, 1.0])
    s = (n @ c) / (n @ ray)
    o = s * ray
    x = np.array([1.0, 0, 0]) - n * n[0]
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n])            # rows: table axes in camera coords
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ o
    return T


def to_table(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def oriented_footprint(xy):
    if len(xy) < 10:
        return np.nan, np.nan
    rect = cv2.minAreaRect(xy.astype(np.float32))
    # shrink to the 5-95 percentile extent along the rectangle axes (outlier points)
    ang = np.radians(rect[2])
    Rm = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    q = (xy - xy.mean(0)) @ Rm.T
    ext = np.percentile(q, 95, 0) - np.percentile(q, 5, 0)
    return float(max(ext)), float(min(ext))


def run(video):
    m = meta(video)
    dz = load_depth(video)
    K = dz["K"]
    d, tracks, table, kept = build_tracks(video)
    mp = np.load(WORK / video / "mp.npz")
    kd = {int(k): n for n, k in enumerate(dz["k"])}
    # ---- reference plane: median over up to 15 keyframes, table hull minus every detection minus hands
    planes = []
    for k in list(kd)[::max(1, len(kd) // 15)]:
        D = dz["depth"][kd[k]]
        free = table.copy()
        for i in kept.get(k, []):
            free &= ~cv2.dilate(d["masks"][i].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        from cleantables.perception.scene.tracks import hand_hulls
        free &= ~cv2.dilate(hand_hulls(video, k).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        P, _ = backproject(D, K, free)
        if len(P) > 500:
            planes.append(ransac_plane(P[::3]))
    N = np.array([p[0] for p in planes])
    n_ref = np.median(N, 0); n_ref /= np.linalg.norm(n_ref)
    c_ref = np.median(np.array([p[1] for p in planes]), 0)
    T = table_frame(n_ref, c_ref, K)
    cam_h = float(-(T[:3, :3] @ np.zeros(3) + T[:3, 3])[2] * -1)
    cam_height = float((T[:3, :3] @ np.zeros(3) + T[:3, 3])[2])
    # ---- per-keyframe depth scale to the reference plane (least squares on free table pixels)
    scale = {}
    for k, n in kd.items():
        D = dz["depth"][n]
        P, _ = backproject(D, K, table)
        if len(P) < 200:
            scale[k] = 1.0; continue
        rays = P / P[:, 2:3]
        z_plane = (n_ref @ c_ref) / (rays @ n_ref)
        r = z_plane / np.maximum(P[:, 2], 1e-6)
        scale[k] = float(np.median(r[(r > 0.8) & (r < 1.25)])) if np.any((r > 0.8) & (r < 1.25)) else 1.0
    # ---- hands: every landmark stays on its own pixel ray. Palm depth = median aligned depth at the 5 palm landmarks
    # + 1 cm skin offset at depth keyframes; at the other 10 fps frames, z = z(nearest depth keyframe of the same hand, <= 3 s) x
    # palm_px(keyframe) / palm_px(frame) (apparent-size ratio). Relative joint depth = MediaPipe world z x the frame's metric scale
    # (z_palm x palm_px / (f x palm_world)). Hand identity = MediaPipe handedness label.
    n = m["n"]
    lab = mp["hand_label"]
    Lm = mp["hand_lm"]
    Wm = mp["hand_w"]

    def palm_px(L):
        return max(np.linalg.norm(L[0, :2] - L[5, :2]), np.linalg.norm(L[0, :2] - L[17, :2]), np.linalg.norm(L[5, :2] - L[17, :2]))

    def palm_w(Wl):
        return max(np.linalg.norm(Wl[0] - Wl[5]), np.linalg.norm(Wl[0] - Wl[17]), np.linalg.norm(Wl[5] - Wl[17]))
    zkey = {0: {}, 1: {}}
    for k, nk in kd.items():
        D = dz["depth"][nk] * scale[k]
        for j in range(2):
            if lab[k, j] < 0:
                continue
            L = Lm[k, j]
            zs = []
            for i in PALM:
                u, v = int(L[i, 0] / 2), int(L[i, 1] / 2)
                if 0 <= u < W and 0 <= v < H:
                    p_ = D[max(0, v - 1):v + 2, max(0, u - 1):u + 2]
                    p_ = p_[p_ > 0]
                    if len(p_):
                        zs.append(np.median(p_))
            if len(zs) >= 3:
                zkey[int(lab[k, j])][k] = (float(np.median(zs)) + 0.01, palm_px(L))
    J = np.full((n, 2, 21, 3), np.nan, np.float32)
    zsrc = np.zeros((n, 2), np.int8)
    f = K[0, 0]
    for k in range(n):
        for j in range(2):
            h = int(lab[k, j])
            if h < 0:
                continue
            L = Lm[k, j]
            cand = [(abs(kk - k), kk) for kk in zkey[h] if abs(kk - k) <= 30]
            if not cand:
                continue
            kk = min(cand)[1]
            z0, px0 = zkey[h][kk]
            z = z0 if kk == k else z0 * px0 / max(palm_px(L), 1.0)
            sc = z * palm_px(L) / (f * max(palm_w(Wm[k, j]), 1e-3))
            zj = z + sc * (Wm[k, j][:, 2] - Wm[k, j][PALM, 2].mean())
            Pc = np.stack([(L[:, 0] - K[0, 2]) * zj / K[0, 0], (L[:, 1] - K[1, 2]) * zj / K[1, 1], zj], 1)
            J[k, h] = to_table(T, Pc)
            zsrc[k, h] = 1 if kk == k else 2
    s_hand = float(np.nanmedian([z * px / (f * 1) for hk in zkey.values() for z, px in hk.values()])) if any(zkey.values()) else np.nan
    ratios = [1] * sum(len(v) for v in zkey.values())
    # ---- objects per keyframe
    obj = []
    for ti, t in enumerate(tracks):
        for (k, i) in t["obs"]:
            if k not in kd:
                continue
            mk = cv2.erode(d["masks"][i].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            P, _ = backproject(dz["depth"][kd[k]] * scale[k], K, mk)
            if len(P) < 8:
                continue
            Q = to_table(T, P)
            Lx, Wy = oriented_footprint(Q[:, :2])
            obj.append((ti, k, *np.median(Q, 0), float(np.percentile(Q[:, 2], 95)), Lx, Wy, len(P)))
    obj = np.array(obj, np.float64) if obj else np.zeros((0, 10))
    Jm, cm, chk = mano_hands(video, T, K, dz, scale, n)
    return dict(T=T, n=n_ref, c=c_ref, K=K, cam_height=cam_height, s_hand=s_hand, n_ratio=len(ratios), J=J, zsrc=zsrc, obj=obj, tracks=tracks,
                Jm=Jm, cm=cm, mano_depth_check=chk,
                model=dz["model"], plane_inlier=[p[2] for p in planes], scale=scale)


def main(videos=None):
    eps = json.load(open(OUT / "episodes.json"))["episodes"]
    (OUT / "3d").mkdir(parents=True, exist_ok=True)
    summary = {}
    for video in videos or sorted({e["video"] for e in eps}):
        try:
            r = run(video)
        except FileNotFoundError as ex:
            print(video, "missing input", ex)
            continue
        m = meta(video)
        for e in [e for e in eps if e["video"] == video]:
            k0, k1 = e["start_frame"] // m["stride"], e["end_frame"] // m["stride"]
            objs = json.load(open(OUT / "objects" / f"{e['id']}.json"))["objects"]
            idmap = {o["track_index"]: o["id"] for o in objs}
            sel = (r["obj"][:, 1] >= k0) & (r["obj"][:, 1] <= k1) if len(r["obj"]) else np.zeros(0, bool)
            O = r["obj"][sel]
            O = O[np.isin(O[:, 0], list(idmap))] if len(O) else O
            np.savez_compressed(OUT / "3d" / f"{e['id']}.npz", T_cam_to_table=r["T"], K=r["K"], hand_joints_table=r["J"][k0:k1 + 1], hand_depth_src=r["zsrc"][k0:k1 + 1],
                                hand_slots=np.array(["MediaPipe Left", "MediaPipe Right"]),
                                mano_joints_table=(r["Jm"][k0:k1 + 1] if r["Jm"] is not None else np.zeros(0)), mano_conf=(r["cm"][k0:k1 + 1] if r["cm"] is not None else np.zeros(0)),
                                mano_slots=np.array(["WiLoR left", "WiLoR right"]),
                                hand_t=np.arange(k0, k1 + 1) * m["stride"] / m["fps_video"],
                                obj_track=O[:, 0].astype(int) if len(O) else np.zeros(0, int), obj_t=O[:, 1] * m["stride"] / m["fps_video"] if len(O) else np.zeros(0),
                                obj_xyz_table=O[:, 2:5] if len(O) else np.zeros((0, 3)), obj_top_height=O[:, 5] if len(O) else np.zeros(0),
                                obj_footprint_LW=O[:, 6:8] if len(O) else np.zeros((0, 2)),
                                obj_ids=np.array([idmap[int(x)] for x in O[:, 0]]) if len(O) else np.zeros(0, str))
            sizes = {}
            for o in objs:
                rows = O[O[:, 0] == o["track_index"]] if len(O) else O
                if len(rows):
                    sizes[o["id"]] = dict(category=o["category"], n=int(len(rows)), L_m=round(float(np.nanmedian(rows[:, 6])), 3),
                                          W_m=round(float(np.nanmedian(rows[:, 7])), 3), top_height_m=round(float(np.nanmedian(rows[:, 5])), 3))
            valid = np.isfinite(r["J"][k0:k1 + 1, :, 0, 0])
            summ = dict(episode=e["id"], depth_model=r["model"], plane_normal_cam=np.round(r["n"], 4).tolist(), camera_height_m=round(r["cam_height"], 3),
                        plane_inlier_frac=[round(x, 2) for x in r["plane_inlier"]], hand_scale=round(r["s_hand"], 3), hand_scale_n=r["n_ratio"],
                        hand_frames_with_joints=int(valid.any(1).sum()), n_hand_frames=int(len(valid)),
                        hand_joint_min_height_m_p5=round(float(np.nanpercentile(r["J"][k0:k1 + 1, :, :, 2], 5)), 3) if valid.any() else None,
                        n_object_points=int(len(O)), object_sizes=sizes,
                        mano_frames=int((r["cm"][k0:k1 + 1] > 0).any(1).sum()) if r["cm"] is not None else 0,
                        mano_depth_alignment=r["mano_depth_check"])
            json.dump(summ, open(OUT / "3d" / f"{e['id']}.json", "w"), indent=1)
            summary[e["id"]] = summ
            print(e["id"], "cam h", summ["camera_height_m"], "hand s", summ["hand_scale"], "objs", len(sizes), "hand p5 z", summ["hand_joint_min_height_m_p5"])
    return summary


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
