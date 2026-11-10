"""Lift the WiLoR MANO hands (cleantables.perception.hands.wilor_worker) to the metric world frame W of the recon clips.

  source scripts/env.sh
  PYTHONPATH=third_party/WiLoR:cache/pylibs_wilor:. CUDA_VISIBLE_DEVICES= $VLA_PY -m cleantables.perception.hands.lift [CLIP ...]

Per clip (all inputs read-only from data/processed/recon_v3; output data/processed/recon_v3c/_work/<clip>/hand_mano_lift.npz):
1. Handedness: one per clip = the detector class with the larger summed confidence (one hand is used per demo). Every box was
   regressed both as a right and as a left hand (worker), so frames where the detector's class disagrees use the clip handedness's
   regression (counted in `n_class_conflict`).
2. Track: per frame the box with the highest detector confidence among boxes whose centre is within 1.5 box diagonals of the previous
   chosen box (if any; else the highest overall).
3. Camera-frame rotation: WiLoR predicts the hand orientation for a crop seen (near-)orthographically along the optical axis (its own
   focal is 5000/256 * 960 px). Our camera is an ultra-wide (fx ~450 px), so a hand off the image centre is seen along an oblique ray.
   R_cam = R_ray @ R_wilor, R_ray = the minimal rotation taking the optical axis to the ray through the box centre (a priori choice, the
   standard perspective-crop correction; not tuned on any metric).
4. Translation (MANO metric hand size): least squares so the rotated MANO joints project onto WiLoR's own 2D joints with our K.
5. Metric depth from MoGe-3 (never from object sizes): visible MANO vertices (z-buffer over 6 px cells) are compared with the aligned
   depth map at their pixels (object-mask pixels, depth edges and invalid depth excluded).
   * per-clip hand scale s = median over frames of median(D_pix / z_vertex)  (one number per clip: hand size x focal consistency)
   * per-frame depth offset dz = median(D_pix - s z_vertex) moves the hand along the root ray; glitches (> 8 cm from the +-4-frame median)
     are interpolated, then a temporal median of 5.
   Frames with < 15 valid depth samples use dz interpolated from neighbours (`depth_src` 2) or 0 (pure 2D-size depth, `depth_src` 0).
6. W = recon T_world_from_cam.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from cleantables.perception.recon.common import ROOT, WORK, unpack
from cleantables.perception.recon.geometry import project

OUT = ROOT / "data/processed/recon_v3c/_work"
RECON = ROOT / "data/processed/recon_v3"
MANO_DIR = ROOT / "cache/mano/mano_v1_2/models"
GATE = 0.08
MIN_PIX = 15


def mano_layer():
    import torch
    from wilor.models.mano_wrapper import MANO      # WiLoR's own MANO layer (smplx.MANOLayer + OpenPose joint order)
    m = MANO(data_dir=str(MANO_DIR), model_path=str(MANO_DIR), gender="neutral", num_hand_joints=15,
             mean_params=str(ROOT / "third_party/WiLoR/mano_data/mano_mean_params.npz"), create_body_pose=False)
    m.eval()
    return m, torch


def mano_verts(layer, torch, go, hp, betas):
    """Right-hand MANO vertices + 16 joints for rotation-matrix params (B,3,3), (B,15,3,3), (B,10). Same call as WiLoR (pose2rot=False)."""
    with torch.no_grad():
        o = layer(global_orient=torch.tensor(go[:, None], dtype=torch.float32), hand_pose=torch.tensor(hp, dtype=torch.float32),
                  betas=torch.tensor(betas, dtype=torch.float32), pose2rot=False)
    return o.vertices.numpy(), o.joints.numpy()


def rot_z_to(d):
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    s, c = np.linalg.norm(v), float(z @ d)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def solve_t(P, uv, K):
    """t minimising the reprojection of P + t onto uv (linearised, exact for zero noise)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a = uv[:, 0] - cx
    b = uv[:, 1] - cy
    A = np.zeros((2 * len(P), 3))
    y = np.zeros(2 * len(P))
    A[0::2, 0] = fx; A[0::2, 2] = -a; y[0::2] = a * P[:, 2] - fx * P[:, 0]
    A[1::2, 1] = fy; A[1::2, 2] = -b; y[1::2] = b * P[:, 2] - fy * P[:, 1]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def visible_vertices(V, K, hw, cell=6):
    uv = project(V, K)
    h, w = hw
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (V[:, 2] > 0.05)
    ci = (uv[:, 1] // cell).astype(int) * (w // cell + 1) + (uv[:, 0] // cell).astype(int)
    zb = np.full((h // cell + 1) * (w // cell + 1), np.inf)
    np.minimum.at(zb, ci[ok], V[ok, 2])
    vis = ok & (V[:, 2] <= zb[np.clip(ci, 0, len(zb) - 1)] + 0.012)
    return vis, uv


def depth_edges(D, rel=0.03):
    gx = np.abs(np.diff(D, axis=1, prepend=D[:, :1]))
    gy = np.abs(np.diff(D, axis=0, prepend=D[:1]))
    return np.maximum(gx, gy) > rel * np.maximum(D, 1e-3)


def choose_track(w, hand_right):
    fr = w["frame"]
    rows = np.flatnonzero(w["is_right"] == hand_right)
    n = int(w["n_frames"])
    sel = np.full(n, -1)
    prev = None
    for k in range(n):
        cand = rows[fr[rows] == k]
        if not len(cand):
            continue
        if prev is not None:
            c = (w["box"][cand, :2] + w["box"][cand, 2:]) / 2
            diag = np.linalg.norm(w["box"][prev, 2:] - w["box"][prev, :2])
            pc = (w["box"][prev, :2] + w["box"][prev, 2:]) / 2
            near = np.linalg.norm(c - pc, axis=1) <= 1.5 * diag
            if near.any():
                cand = cand[near]
        sel[k] = cand[np.argmax(w["det_conf"][cand])]
        prev = sel[k]
    return sel


def lift_clip(clip, layer, torch):
    w = dict(np.load(OUT / clip / "hand_wilor.npz"))
    r = np.load(RECON / f"{clip}.npz", allow_pickle=True)
    K = r["K"].astype(np.float64)
    T_wc = r["T_world_from_cam"]
    dz_ = np.load(WORK / clip / "depth.npz")
    D = dz_["depth"]
    n, h, wd = D.shape
    sz = np.load(WORK / clip / "sam3.npz")
    om = unpack(sz["p0_masks"], tuple(sz["shape"]))                              # (n, 4, h, w) object candidates
    objm = (om & (sz["p0_probs"][:, :, None, None] >= 0.3)).any(1)
    # handedness: summed detector confidence per class (each box appears twice in the file: count it once)
    first = np.unique(w["det_id"], return_index=True)[1]
    cr = w["det_is_right"][first]; cc = w["det_conf"][first]
    hand_right = float(cc[cr == 1].sum() > cc[cr == 0].sum())
    sel = choose_track(w, hand_right)
    ok = sel >= 0
    idx = sel[ok]
    V, Jchk = mano_verts(layer, torch, w["global_orient"][idx], w["hand_pose"][idx], w["betas"][idx])
    if not hand_right:
        Jchk[..., 0] *= -1
    assert len(idx) == 0 or np.abs(Jchk - w["joints_mano"][idx]).max() < 2e-3, "regenerated MANO joints differ from WiLoR's"
    Vall = np.full((n, 778, 3), np.nan, np.float32)
    Vall[ok] = V
    if not hand_right:
        Vall[..., 0] *= -1
    Jm = np.full((n, 21, 3), np.nan); Jm[ok] = w["joints_mano"][idx]
    # sanity: the regenerated MANO joints == WiLoR's (16 kinematic joints subset is implicit: compare wrist)
    Jc = np.full((n, 21, 3), np.nan); Vc = np.full((n, 778, 3), np.nan, np.float32); Rcam = np.full((n, 3, 3), np.nan)
    ratio = np.full(n, np.nan); npx = np.zeros(n, int); zv_med = np.full(n, np.nan); dmed = np.full(n, np.nan)
    pix_cache = {}
    for k in np.flatnonzero(ok):
        i = sel[k]
        b = w["box"][i]
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])
        Rr = rot_z_to(np.linalg.solve(K, c))
        root = Jm[k, 0]
        J = (Jm[k] - root) @ Rr.T
        Vk = (Vall[k] - root) @ Rr.T
        t = solve_t(J, w["kp2d"][i].astype(np.float64), K)
        J, Vk = J + t, Vk + t
        if not (J[:, 2] > 0.05).all():
            ok[k] = False
            continue
        go = w["global_orient"][i].astype(np.float64)
        if not hand_right:
            M = np.diag([-1.0, 1, 1])
            go = M @ go @ M
        Rcam[k] = Rr @ go
        Jc[k], Vc[k] = J, Vk
        vis, uv = visible_vertices(Vk, K, (h, wd))
        u = np.clip(np.round(uv[vis, 0]).astype(int), 0, wd - 1); v = np.clip(np.round(uv[vis, 1]).astype(int), 0, h - 1)
        Dk = D[k].astype(np.float32) * float(r["depth_scale"][k]) + float(r["depth_shift"][k])
        Dk[D[k] <= 0] = 0
        bad = objm[k] | depth_edges(Dk) | (Dk <= 0)
        g = ~bad[v, u]
        npx[k] = int(g.sum())
        if npx[k] >= MIN_PIX:
            zp, zv = Dk[v[g], u[g]].astype(np.float64), Vk[vis][g, 2].astype(np.float64)
            ratio[k] = np.median(zp / zv)
            pix_cache[k] = (zp, zv)
    good = np.isfinite(ratio)
    s = float(np.median(ratio[good])) if good.sum() >= 3 else 1.0
    dz = np.full(n, np.nan)
    for k, (zp, zv) in pix_cache.items():
        dz[k] = np.median(zp - s * zv)
    # glitch gate + interpolation + temporal median
    dzg = dz.copy()
    for k in range(n):
        win = dz[max(0, k - 4):k + 5]; win = win[np.isfinite(win)]
        if np.isfinite(dz[k]) and len(win) >= 3 and abs(dz[k] - np.median(win)) > GATE:
            dzg[k] = np.nan
    src = np.where(np.isfinite(dzg), 1, 0)
    fin = np.isfinite(dzg)
    if fin.sum() >= 2:
        ii = np.arange(n)
        fill = np.interp(ii, ii[fin], dzg[fin])
        dzg = np.where(fin, dzg, fill)
        src = np.where(fin, 1, 2)
    dzg = np.where(np.isfinite(dzg), dzg, 0.0)
    dzs = dzg.copy()
    for k in range(n):
        win = dzg[max(0, k - 2):k + 3][ok[max(0, k - 2):k + 3]]
        dzs[k] = np.median(win) if len(win) else dzg[k]
    Jcf = np.full((n, 21, 3), np.nan); Vcf = np.full((n, 778, 3), np.nan, np.float32)
    for k in np.flatnonzero(ok):
        J, Vk = Jc[k] * s, Vc[k] * s
        dvec = J[0] / J[0, 2]
        Jcf[k] = J + dzs[k] * dvec
        Vcf[k] = Vk + dzs[k] * dvec
    src = np.where(ok, src, -1)
    toW = lambda P: P @ T_wc[:3, :3].T + T_wc[:3, 3]
    Jw = np.where(ok[:, None, None], toW(np.nan_to_num(Jcf)), np.nan)
    Vw = np.where(ok[:, None, None], toW(np.nan_to_num(Vcf)), np.nan).astype(np.float32)
    Rw = np.where(ok[:, None, None], np.einsum("ij,njk->nik", T_wc[:3, :3], np.nan_to_num(Rcam)), np.nan)
    conf = np.where(ok, w["det_conf"][np.maximum(sel, 0)], 0.0)
    cls_conflict = np.where(ok, w["det_is_right"][np.maximum(sel, 0)] != hand_right, False)
    # 2D residual of our projection vs WiLoR's own 2D joints (should be ~0 up to the depth-offset move along the ray)
    rp = [np.median(np.linalg.norm(project(Jcf[k], K) - w["kp2d"][sel[k]], axis=1)) for k in np.flatnonzero(ok)]
    res = dict(joints_w=Jw.astype(np.float32), joints_cam=Jcf.astype(np.float32), verts_w=Vw.astype(np.float16), valid=ok,
               R_mano_cam=Rcam.astype(np.float32), R_mano_w=Rw.astype(np.float32), conf=conf.astype(np.float32),
               det_row=sel.astype(np.int32), class_conflict=cls_conflict, hand_right=np.float32(hand_right),
               hand_scale=np.float32(s), depth_offset=dzs.astype(np.float32), depth_offset_raw=dz.astype(np.float32), depth_src=src.astype(np.int8),
               n_depth_px=npx.astype(np.int32), betas=np.where(ok[:, None], w["betas"][np.maximum(sel, 0)], np.nan).astype(np.float32),
               hand_pose=np.where(ok[:, None, None, None], w["hand_pose"][np.maximum(sel, 0)], np.nan).astype(np.float32),
               global_orient_wilor=np.where(ok[:, None, None], w["global_orient"][np.maximum(sel, 0)], np.nan).astype(np.float32))
    info = dict(n=n, n_valid=int(ok.sum()), hand_right=bool(hand_right), hand_scale=s, n_scale_frames=int(good.sum()),
                frac_depth_measured=float(np.mean(src[ok] == 1)) if ok.any() else 0.0, n_class_conflict=int(cls_conflict.sum()),
                reproj_px_median=float(np.median(rp)) if rp else None, dz_median=float(np.nanmedian(dz)) if np.isfinite(dz).any() else None)
    return res, info


def main():
    clips = sys.argv[1:] or sorted(p.parent.name for p in OUT.glob("*/hand_wilor.npz"))
    layer, torch = mano_layer()
    summ = {}
    for c in clips:
        t0 = time.time()
        res, info = lift_clip(c, layer, torch)
        np.savez_compressed(OUT / c / "hand_mano_lift.npz", **res)
        info["wall_s"] = round(time.time() - t0, 1)
        summ[c] = info
        print(c, info, flush=True)
    f = OUT / "lift_summary.json"
    old = json.load(open(f)) if f.exists() else {}
    old.update(summ)
    json.dump(old, open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
