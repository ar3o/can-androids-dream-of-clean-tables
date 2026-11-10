"""Full domain randomisation for render variants >= 4 of paired_render.py and composite_render.py, applied on top of the v1-v3
randomisation (lights, table / background textures, exposure). Generic ranges, the same for every scene, seeded by
(episode, variant); every draw is logged in the render's meta.json (dr):
  camera   the phone camera moved about its calibrated pose, in its own axes: +-3 cm sideways and up-down, +-5 cm along the view;
           tilt and pan +-3 deg, roll +-2 deg; focal length x0.95-1.05 (zoom)
  sizes    every object scaled x0.92-1.08 about its bottom centre (so it stays on the table)
  objects  every object's texture: hue +-6 deg, saturation x0.85-1.15, value x0.85-1.15
  lights   key light direction +-40 deg (v1-v3: +-25); dome / key tint and their ratio as v1-v3, but the total (dome + key) is
           x0.6-1.3 the robot runner's (600 + 700), so highlights do not blow out
  sensor   auto-exposure (as a phone: the frame's mean grey set to 150-195; the real videos' frames average 164-189), white balance
           x0.94-1.06 per channel, blur sigma 0.2-0.8 px (half the variants), noise sigma 0-3 grey levels (fresh every frame),
           JPEG quality 70-95
Not randomised: object positions in the paired renders (they must show the real frame's layout; new layouts come from the
composites) and physics (the renders follow tracked or given poses; physics is randomised in robot training).
"""
from __future__ import annotations

import os
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

RANGES = dict(cam_shift_m=(0.03, 0.03, 0.05), cam_rot_deg=(3, 3, 2), focal=(0.95, 1.05), size=(0.92, 1.08), obj_hue_deg=6,
              obj_sat=(0.85, 1.15), obj_val=(0.85, 1.15), key_rot_deg=40, light_total=(0.6 * 1300, 1.3 * 1300), mean_grey=(150, 195), wb=(0.94, 1.06), blur=(0.2, 0.8), noise=(0, 3), jpeg=(70, 95))


def jitter_png(src, dst, rng, hue=4, sat=(0.8, 1.2), val=(0.85, 1.15), con=(0.85, 1.15)):
    """Hue in OpenCV units (deg / 2)."""
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    alpha = im[..., 3:] if im.ndim == 3 and im.shape[-1] == 4 else None
    hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue)) % 180
    hsv[..., 1] *= rng.uniform(*sat)
    hsv[..., 2] *= rng.uniform(*val)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if con:
        m = out.mean((0, 1)); out = (out - m) * rng.uniform(*con) + m
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, out if alpha is None else np.concatenate([out, alpha], -1))


def full(S, rng, vdir, root, DR0):
    """-> (S with a moved / zoomed main camera and per-variant object USDs, DR dict). Object sizes go on the spawn cfg (apply_sizes)."""
    G = RANGES
    c = dict(S["cameras"]["main"])
    T0 = np.array(c["T_cam_to_table"], np.float64)
    K0 = np.array(c["K"], np.float64)
    T, K = T0.reshape(4, 4), K0.reshape(3, 3).copy()
    shift = rng.uniform(-1, 1, 3) * G["cam_shift_m"]
    rot = rng.uniform(-1, 1, 3) * G["cam_rot_deg"]                        # about the camera's x (tilt), y (pan), z (roll)
    J = np.eye(4); J[:3, :3] = R.from_euler("xyz", rot, degrees=True).as_matrix(); J[:3, 3] = shift
    T = T @ J
    f = float(rng.uniform(*G["focal"])); K[0, 0] *= f; K[1, 1] *= f
    c.update(T_cam_to_table=T.reshape(T0.shape).tolist(), K=K.reshape(K0.shape).tolist())
    DR = dict(full=True, ranges=G, cam_shift_m=shift.tolist(), cam_rot_deg=rot.tolist(), focal=f, cam_T=T.tolist(), cam_K=K.tolist())
    objs, size = [], {}
    for o in S["objects"]:
        size[o["id"]] = float(rng.uniform(*G["size"]))
        src = os.path.join(root, o["usd"])
        txt = open(src).read()
        for tex in sorted(set(re.findall(r"@([^@]+\.(?:png|jpg))@", txt))):
            p = tex if os.path.isabs(tex) else os.path.normpath(os.path.join(os.path.dirname(src), tex))
            dst = os.path.join(vdir, f"obj_{o['id']}_{os.path.basename(p)}")
            jitter_png(p, dst, rng, hue=G["obj_hue_deg"] / 2, sat=G["obj_sat"], val=G["obj_val"], con=None)
            txt = txt.replace(f"@{tex}@", f"@{dst}@")
        u = os.path.join(vdir, f"obj_{o['id']}.usda")
        open(u, "w").write(txt)
        objs.append(dict(o, usd=u))                                        # absolute: os.path.join(ROOT, u) == u
    DR["obj_size"] = size
    DR.update(key_rot_deg=(rng.uniform(-1, 1, 2) * G["key_rot_deg"]).tolist(), wb=rng.uniform(*G["wb"], 3).tolist(),
              blur=float(rng.uniform(*G["blur"])) if rng.random() < 0.5 else 0.0, noise=float(rng.uniform(*G["noise"])),
              jpeg=int(rng.integers(G["jpeg"][0], G["jpeg"][1] + 1)), noise_seed=int(rng.integers(1 << 31)))
    tot = float(rng.uniform(*G["light_total"])); f = tot / (DR0["dome"] + DR0["key"])
    DR.update(dome=DR0["dome"] * f, key=DR0["key"] * f, mean_grey=float(rng.uniform(*G["mean_grey"])))
    S = dict(S, objects=objs, cameras=dict(S["cameras"], main=c))
    return S, DR


def apply_sizes(cfg_scene, S, DR):
    for k, o in enumerate(S["objects"]):
        s = DR["obj_size"][o["id"]]
        getattr(cfg_scene, f"obj_{k}").spawn.scale = (s, s, s)


def post(im, DR, frame):
    """Sensor effects on a BGR uint8 render (in place of the v1-v3 gain; gamma is kept); noise differs per frame (seeded by noise_seed + frame)."""
    x = im.astype(np.float32)
    x = x * (DR["mean_grey"] / max(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean(), 1.0))                      # auto-exposure
    x = x * np.array(DR["wb"], np.float32)
    if DR["blur"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), DR["blur"])
    if DR["noise"] > 0:
        x = x + np.random.default_rng(DR["noise_seed"] + int(frame)).normal(0, DR["noise"], x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, DR["jpeg"]])[1], cv2.IMREAD_COLOR)


def hull_uv(verts, poses, K, T_cam_to_table, sizes=None, hw=(16, 28), cov_min=0.25, wh=(960, 540)):
    """uv (n, 2) / ok (n,) as at robot run time: coverage-weighted centroid on the hw grid of each object's projected convex hull.
    verts: list of (v, 3) mesh vertices in the object frame (root = bottom centre); poses: list of (x, y, z_bottom, yaw_off)."""
    Tinv = np.linalg.inv(np.asarray(T_cam_to_table, np.float64).reshape(4, 4))
    K = np.asarray(K, np.float64).reshape(3, 3)
    hh, ww = hw
    yy, xx = np.meshgrid((np.arange(hh) + 0.5) / hh, (np.arange(ww) + 0.5) / ww, indexing="ij")
    W, H = wh[0] // 2, wh[1] // 2
    uv, ok = np.zeros((len(verts), 2), np.float32), np.zeros(len(verts), bool)
    for j, (V, p) in enumerate(zip(verts, poses)):
        if V is None or p is None or not np.isfinite(p).all():
            continue
        s = 1.0 if sizes is None else sizes[j]
        X = (s * V) @ R.from_euler("z", p[3]).as_matrix().T + p[:3]
        C = X @ Tinv[:3, :3].T + Tinv[:3, 3]
        if (C[:, 2] < 0.05).any():
            continue
        q = (C @ K.T)[:, :2] / C[:, 2:3] / 2.0
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(q).astype(np.int32)), 1)
        c = cv2.resize(m.astype(np.float32), (ww, hh), interpolation=cv2.INTER_AREA)
        w = c.sum()
        if w > cov_min:
            uv[j] = [(c * xx).sum() / w, (c * yy).sum() / w]; ok[j] = True
    return uv, ok
