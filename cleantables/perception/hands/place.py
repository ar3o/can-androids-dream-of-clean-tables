"""Two-hand 3D hands from WiLoR per episode (camera frame, OpenCV, metres).

Stage 1 `raw` (needs WiLoR's MANO layer: vla env + third_party/WiLoR on PYTHONPATH):
  Per frame and per hand class (WiLoR detector: left / right), the detection regressed with its own class (each box is in the file twice,
  once per class) and the highest detector confidence, preferring the box nearest the previous frame's box of that hand (within 1.5 box
  diagonals). A left and a right box covering the same hand (IoU > 0.5) keep only the more confident one.
  Camera rotation = perspective-crop correction (R_ray, as in hands.lift). Translation = least squares so the metric MANO joints
  project onto WiLoR's own 2D joints with our K (depth from hand size only).
  -> OUT/hands/<ep>/hands_raw.npz  per hand h in (left, right): <h>_ok (n,), <h>_conf, <h>_kp2d (n,21,2) work px, <h>_J (n,21,3),
     <h>_V (n,778,3) camera frame from size only; faces (MANO, right-hand winding), k0.
Stage 2 `place` (numpy only): metric placement along the camera ray from a depth map (as hands.lift step 5, per hand):
  one scale s per hand and episode = median over frames of median(depth at visible vertices / vertex z) (hand size x focal consistency),
  then per frame an offset dz along the root ray = median(depth - s z); dz glitches (> 8 cm from the +-4 frame median) interpolated, then a
  5-frame temporal median. Object-mask pixels, depth edges and pixels nearer than 2 cm in front of the hand surface are excluded.
  -> OUT/hands/<ep>/hands_<depth variant>.npz (<h>_J, <h>_V, <h>_ok, <h>_s, <h>_dz, <h>_npx)
  $VLA_PY -m cleantables.perception.hands.place raw [EP...]     $FP_PY -m cleantables.perception.hands.place place [EP...]
"""
from __future__ import annotations

import sys

import numpy as np

from cleantables.perception.episode_io import OUT, TRIAL, H, W, episode, geom, jdump

HANDS = ("left", "right")
TIPS = [4, 8, 12, 16, 20]          # OpenPose joint order: thumb, index, middle, ring, little tips
GATE = 0.08


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def select(w, ks):
    """-> {hand: (n,) row index or -1}"""
    fr = w["frame"]
    own = w["is_right"] == w["det_is_right"]
    sel = {h: np.full(len(ks), -1) for h in HANDS}
    prev = {h: None for h in HANDS}
    for j, k in enumerate(ks):
        rows = np.flatnonzero((fr == k) & own)
        if not len(rows):
            continue
        # one physical hand detected as both classes: keep the more confident box
        keep = []
        for r in sorted(rows, key=lambda r: -w["det_conf"][r]):
            if all(_iou(w["box"][r], w["box"][q]) < 0.5 for q in keep):
                keep.append(r)
        for h, cls in (("left", 0.0), ("right", 1.0)):
            cand = np.array([r for r in keep if w["det_is_right"][r] == cls])
            if not len(cand):
                continue
            if prev[h] is not None:
                c = (w["box"][cand, :2] + w["box"][cand, 2:]) / 2
                pb = w["box"][prev[h]]
                near = np.linalg.norm(c - (pb[:2] + pb[2:]) / 2, axis=1) <= 1.5 * np.linalg.norm(pb[2:] - pb[:2])
                if near.any():
                    cand = cand[near]
            sel[h][j] = cand[np.argmax(w["det_conf"][cand])]
            prev[h] = sel[h][j]
    return sel


def raw(ep):
    import torch
    from cleantables.perception.hands.lift import mano_layer, mano_verts, rot_z_to, solve_t
    layer, _ = mano_layer()
    e = episode(ep)
    K = geom(e["video"])["K"]
    w = dict(np.load(OUT / "hands" / ep / "hand_wilor.npz"))
    ks = np.arange(e["w0"], e["w1"] + 1)
    sel = select(w, ks)
    n = len(ks)
    res = dict(k0=np.int32(ks[0]), faces=np.asarray(layer.faces, np.int32))
    info = {}
    for h in HANDS:
        s = sel[h]
        ok = s >= 0
        idx = s[ok]
        J = np.full((n, 21, 3), np.nan, np.float32); V = np.full((n, 778, 3), np.nan, np.float32)
        kp = np.full((n, 21, 2), np.nan, np.float32); conf = np.zeros(n, np.float32); box = np.full((n, 4), np.nan, np.float32)
        if len(idx):
            Vm, _ = mano_verts(layer, torch, w["global_orient"][idx], w["hand_pose"][idx], w["betas"][idx])
            if h == "left":
                Vm[..., 0] *= -1                                      # WiLoR: left hands are the mirrored right-hand model
            Jm = w["joints_mano"][idx]
            for a, j in enumerate(np.flatnonzero(ok)):
                i = s[j]
                b = w["box"][i]
                Rr = rot_z_to(np.linalg.solve(K, np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])))
                root = Jm[a, 0]
                Jc = (Jm[a] - root) @ Rr.T
                Vc = (Vm[a] - root) @ Rr.T
                t = solve_t(Jc, w["kp2d"][i].astype(np.float64), K)
                if not ((Jc + t)[:, 2] > 0.05).all():
                    ok[j] = False
                    continue
                J[j], V[j] = Jc + t, Vc + t
                kp[j], conf[j], box[j] = w["kp2d"][i], w["det_conf"][i], b
        res.update({f"{h}_ok": ok, f"{h}_J": J, f"{h}_V": V, f"{h}_kp2d": kp, f"{h}_conf": conf, f"{h}_box": box})
        info[h] = dict(frames=int(ok.sum()), of=n)
    np.savez_compressed(OUT / "hands" / ep / "hands_raw.npz", **res)
    print(ep, info, flush=True)


def load_hands(ep):
    """Stage-1 hands for depth_lock: {hand: dict(ok, kp2d, z_size (n,21), k0)} or None."""
    p = OUT / "hands" / ep / "hands_raw.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {h: dict(ok=z[f"{h}_ok"], kp2d=z[f"{h}_kp2d"], z_size=z[f"{h}_J"][..., 2], k0=int(z["k0"])) for h in HANDS}


def _visible(Vc, K_lo, cell=3):
    u = Vc[:, 0] / Vc[:, 2] * K_lo[0, 0] + K_lo[0, 2]
    v = Vc[:, 1] / Vc[:, 2] * K_lo[1, 1] + K_lo[1, 2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ci = (v // cell).astype(int) * (W // cell + 1) + (u // cell).astype(int)
    zb = np.full((H // cell + 1) * (W // cell + 1), np.inf)
    np.minimum.at(zb, ci[ok], Vc[ok, 2])
    vis = ok & (Vc[:, 2] <= zb[np.clip(ci, 0, len(zb) - 1)] + 0.012)
    return vis, np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)


def place(ep, variant):
    """Metric placement of the stage-1 hands with depth `variant` ('current' = det3d table scale, 'lock_sb' = locked)."""
    from cleantables.perception.episode_io import depth_full, sam3_masks
    e = episode(ep)
    G = geom(e["video"])
    z = dict(np.load(OUT / "hands" / ep / "hands_raw.npz"))
    ks = np.arange(e["w0"], e["w1"] + 1)
    if variant == "lock_sb":
        L = np.load(OUT / "depth" / ep / "locked.npz")
        D = L["depth"].astype(np.float32)
    else:
        _, D = depth_full(ep)
        mz = np.load(OUT / "depth" / ep / "maps.npz")
        D = D * mz["a_cur"][:, None, None]
    M, det_frame, _, _, _ = sam3_masks(e["video"])
    objm = {}
    for i, k in enumerate(det_frame):
        if ks[0] - 3 <= k <= ks[-1] + 3:
            objm[int(k)] = objm.get(int(k), np.zeros((H, W), bool)) | M[i]
    gx = np.abs(np.diff(D, axis=2, prepend=D[:, :, :1])); gy = np.abs(np.diff(D, axis=1, prepend=D[:, :1]))
    edge = np.maximum(gx, gy) > 0.03 * np.maximum(D, 1e-3)
    out = dict(k0=z["k0"], faces=z["faces"])
    info = {}
    for h in HANDS:
        ok, J, V = z[f"{h}_ok"].copy(), z[f"{h}_J"], z[f"{h}_V"]
        n = len(ok)
        ratio = np.full(n, np.nan); cache = {}; npx = np.zeros(n, int)
        for j in np.flatnonzero(ok):
            k = ks[j]
            vis, u, v = _visible(V[j], G["K_lo"])
            om = next((objm[kk] for kk in (k, k - 1, k + 1, k - 2, k + 2) if kk in objm), np.zeros((H, W), bool))
            g = vis & ~om[v, u] & ~edge[j][v, u] & (D[j][v, u] > 0.2)
            npx[j] = int(g.sum())
            if npx[j] >= 15:
                zp, zv = D[j][v[g], u[g]].astype(np.float64), V[j][g, 2].astype(np.float64)
                cache[j] = (zp, zv)
                ratio[j] = np.median(zp / zv)
        good = np.isfinite(ratio)
        s = float(np.median(ratio[good])) if good.sum() >= 3 else 1.0
        dz = np.full(n, np.nan)
        for j, (zp, zv) in cache.items():
            r = zp - s * zv
            r = r[r > -0.02]                     # depth samples well in front of the hand surface are occluders, not the hand
            if len(r) >= 10:
                dz[j] = np.median(r)
        dzg = dz.copy()
        for j in range(n):
            win = dz[max(0, j - 4):j + 5]; win = win[np.isfinite(win)]
            if np.isfinite(dz[j]) and len(win) >= 3 and abs(dz[j] - np.median(win)) > GATE:
                dzg[j] = np.nan
        fin = np.isfinite(dzg)
        if fin.sum() >= 2:
            dzg = np.where(fin, dzg, np.interp(np.arange(n), np.flatnonzero(fin), dzg[fin]))
        dzg = np.nan_to_num(dzg)
        dzs = dzg.copy()
        for j in range(n):
            win = dzg[max(0, j - 2):j + 3][ok[max(0, j - 2):j + 3]]
            dzs[j] = np.median(win) if len(win) else dzg[j]
        Jp = np.full_like(J, np.nan); Vp = np.full_like(V, np.nan)
        for j in np.flatnonzero(ok):
            dvec = J[j, 0] / J[j, 0, 2]
            Jp[j] = J[j] * s + dzs[j] * dvec
            Vp[j] = V[j] * s + dzs[j] * dvec
        out.update({f"{h}_ok": ok, f"{h}_J": Jp.astype(np.float32), f"{h}_V": Vp.astype(np.float32), f"{h}_s": np.float32(s),
                    f"{h}_dz": dzs.astype(np.float32), f"{h}_dz_raw": dz.astype(np.float32), f"{h}_npx": npx.astype(np.int32),
                    f"{h}_conf": z[f"{h}_conf"], f"{h}_kp2d": z[f"{h}_kp2d"]})
        st = np.abs(np.diff(Jp[:, 0, 2]))
        info[h] = dict(frames=int(ok.sum()), scale=round(s, 3), depth_measured=int(np.isfinite(dz).sum()),
                       dz_med_cm=round(float(np.nanmedian(dz) * 100), 1) if np.isfinite(dz).any() else None,
                       wrist_step_gt5cm=int(np.sum(st[np.isfinite(st)] > 0.05)))
    np.savez_compressed(OUT / "hands" / ep / f"hands_{variant}.npz", **out)
    jdump(info, OUT / "hands" / ep / f"hands_{variant}.json")
    print(ep, variant, info, flush=True)
    return info


if __name__ == "__main__":
    stage = sys.argv[1]
    for ep in sys.argv[2:] or TRIAL:
        if stage == "raw":
            raw(ep)
        else:
            for var in ("current", "lock_sb"):
                place(ep, var)
