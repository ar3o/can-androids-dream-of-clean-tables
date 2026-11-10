"""Box geometry from MoGe depth + SAM 3 masks: table plane / table frame, per-detection 3D box pose (top-face centre, top height,
footprint L x W, yaw), rectangle polygons and projection helpers.

Table frame (as in cleantables.perception.scene.recon3d.table_frame): z = table normal (up), origin = where the principal ray meets
the table, x = image-right projected on the table. All lengths in metres.

box_from_points lifts the eroded mask pixels with the per-frame depth (rescaled so free table pixels lie on the fitted plane). The top
is the 90th percentile of z (insensitive to depth bleeding at the mask border), refined to the median of points within 2 cm of it.
Points within TOP_BAND of the top form the top face; its footprint is a cv2.minAreaRect with 3-97 percentile extents plus the erosion
margin. yaw is the direction of the long side, in [0, pi). When a stack is segmented as one mask, the top-face slice keeps only the
top box, so the pose is that box's rather than a blend.
"""
from __future__ import annotations

import cv2
import numpy as np

from cleantables.perception.scene.recon3d import backproject, ransac_plane, table_frame, to_table  # noqa: F401 (re-exported)

TOP_BAND = 0.02
H_LO, W_LO = 270, 480


# ------------------------------------------------------------------ table plane
def fit_table(depths, K, masks_excl=None, seed=0, up=(0, -1.0, 0)):
    """depths: list of (270,480) depth maps (camera z, metres); masks_excl: list of (270,480) bool to exclude (boxes, hands).
    Per frame, RANSAC planes are found greedily (up to 3); the table plane is the one with the most inliers whose normal points up in the
    image (camera high and to the side, looking down: n_cam . up > 0.25 after orienting n towards the camera; up = (0,-1,0) for a
    landscape frame, (1,0,0) for a portrait video rotated clockwise) or that faces the camera head-on (|n_z| > 0.9, near top-down view).
    Returns (T_cam_to_table 4x4, n, c, info)."""
    rng = np.random.default_rng(seed)
    planes = []
    for i, D in enumerate(depths):
        ok = D > 0
        if masks_excl is not None:
            ok &= ~masks_excl[i]
        P, _ = backproject(D, K, ok)
        if len(P) < 2000:
            continue
        P = P[rng.choice(len(P), min(8000, len(P)), replace=False)]
        rest = P
        cands = []
        for _ in range(3):
            if len(rest) < 500:
                break
            n, c, frac = ransac_plane(rest, it=200, thr=0.012, seed=int(rng.integers(1e9)))
            inl = np.abs((rest - c) @ n) < 0.012
            cands.append((int(inl.sum()), n, c))
            rest = rest[~inl]
        up_ = [x for x in cands if x[1] @ np.asarray(up, float) > 0.25 or abs(x[1][2]) > 0.9]
        if up_:
            planes.append(max(up_, key=lambda x: x[0]))
    if not planes:
        raise RuntimeError("no table plane found")
    N = np.array([p[1] for p in planes])
    n = np.median(N, 0); n /= np.linalg.norm(n)
    c = np.median(np.array([p[2] for p in planes]), 0)
    T = table_frame(n, c, K)
    cam_h = float(T[2, 3])
    return T, n, c, dict(n_frames=len(planes), camera_height_m=round(cam_h, 3),
                         tilt_deg=round(float(np.degrees(np.arccos(abs(n @ np.array([0, 0, 1.0]))))), 1))


def plane_z_map(T, K):
    """Camera depth (z) of the table plane at every half-res pixel (270,480), NaN where the ray misses it."""
    nrm, off = T[2, :3], T[2, 3]
    ys, xs = np.mgrid[0:H_LO, 0:W_LO]
    rays = np.stack([(xs * 2 + 0.5 - K[0, 2]) / K[0, 0], (ys * 2 + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    den = rays @ nrm
    with np.errstate(all="ignore"):
        z = -off / den
    z[(z <= 0) | ~np.isfinite(z)] = np.nan
    return z.astype(np.float32)


def table_mask(depths, K, T, thr=0.02, frac=0.3):
    """Pixels on the table plane (|height| < thr) in >= frac of the given frames (after per-frame scale), filled to its convex hull."""
    zp = plane_z_map(T, K)
    acc = np.zeros((H_LO, W_LO))
    for D in depths:
        s = depth_scale(D, zp)
        P, (ys, xs) = backproject(D * s, K)
        h = to_table(T, P)[:, 2]
        on = np.abs(h) < thr
        acc[ys[on], xs[on]] += 1
    m = (acc / max(1, len(depths)) > frac).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones((H_LO, W_LO), bool)
    big = max(cnts, key=cv2.contourArea)
    hm = np.zeros((H_LO, W_LO), np.uint8)
    cv2.fillPoly(hm, [cv2.convexHull(big)], 1)
    return cv2.dilate(hm, np.ones((15, 15), np.uint8)) > 0


def depth_scale(D, zp, free=None):
    """Per-frame MoGe scale so that free table pixels lie on the reference plane (median depth ratio)."""
    ok = np.isfinite(zp) & (D > 0)
    if free is not None:
        ok &= free
    r = zp[ok] / D[ok]
    r = r[(r > 0.75) & (r < 1.33)]
    return float(np.median(r)) if len(r) > 200 else 1.0


# ------------------------------------------------------------------ one detection -> box pose
def lift_mask(D, K, T, mask_lo, erode=1):
    m = mask_lo.astype(np.uint8)
    if erode:
        m2 = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        if m2.sum() >= 15:
            m = m2
    P, _ = backproject(D, K, m.astype(bool))
    return to_table(T, P) if len(P) else np.zeros((0, 3))


def rect_extent(xy):
    """Percentile-based oriented rectangle: (cx, cy, L, W, yaw in [0,pi) of the long side)."""
    rect = cv2.minAreaRect(xy.astype(np.float32))
    ang = np.radians(rect[2])
    ax = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    q = (xy - xy.mean(0)) @ ax.T
    lo, hi = np.percentile(q, 3, 0), np.percentile(q, 97, 0)
    ext = hi - lo
    mid = xy.mean(0) + ((lo + hi) / 2) @ ax
    if ext[0] >= ext[1]:
        L, W, d = ext[0], ext[1], ax[0]
    else:
        L, W, d = ext[1], ext[0], ax[1]
    yaw = float(np.arctan2(d[1], d[0]) % np.pi)
    return float(mid[0]), float(mid[1]), float(L), float(W), yaw


def box_from_points(Q, depth_cam=None, f=None, erode=1):
    """Q (n,3) table-frame points of one box mask -> dict(x, y, top, L, W, yaw, n_top, ok)."""
    if len(Q) < 25:
        return None
    top = float(np.percentile(Q[:, 2], 90))
    if top < 0.015:                                     # flat on the table (a flattened sheet, the tape): not a box
        return dict(x=float(np.median(Q[:, 0])), y=float(np.median(Q[:, 1])), top=top, L=np.nan, W=np.nan, yaw=np.nan, n_top=0, ok=False,
                    why="flat")
    tf = Q[np.abs(Q[:, 2] - top) < max(TOP_BAND, 0.08 * top)]
    if len(tf) >= 15:                                   # p90 is biased up by depth noise: re-centre on the top face's median
        top = float(np.median(tf[:, 2]))
    if len(tf) < 15:
        return dict(x=float(np.median(Q[:, 0])), y=float(np.median(Q[:, 1])), top=top, L=np.nan, W=np.nan, yaw=np.nan, n_top=len(tf),
                    ok=False, why="few top-face points")
    x, y, L, W, yaw = rect_extent(tf[:, :2])
    if depth_cam is not None and f is not None:          # add back the erosion margin (half-res px each side)
        px = 2.0 * depth_cam / f
        L += 2 * erode * px; W += 2 * erode * px
    return dict(x=x, y=y, top=top, L=L, W=W, yaw=yaw, n_top=int(len(tf)), ok=True, why="")


# ------------------------------------------------------------------ rectangles / projection
def rect_corners(x, y, L, W, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = np.array([c, s]) * L / 2, np.array([-s, c]) * W / 2
    return np.array([[x, y]]) + np.array([u + v, -u + v, -u - v, u - v])


def box_corners3d(x, y, z0, L, W, H, yaw):
    """8 corners (bottom 4 then top 4) of a box whose bottom face is at height z0."""
    r = rect_corners(x, y, L, W, yaw)
    return np.concatenate([np.c_[r, np.full(4, z0)], np.c_[r, np.full(4, z0 + H)]])


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def project(P, K, T):
    """Table-frame points (n,3) -> work px (n,2) and camera depth (n,)."""
    Ti = np.linalg.inv(T)
    pc = np.atleast_2d(P) @ Ti[:3, :3].T + Ti[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6), pc[:, 2]


def wrap_pi2(a):
    """Relative yaw of two rectangles, folded to (-pi/2, pi/2]."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def poly_clip(subject, clip):
    """Sutherland-Hodgman: intersection of two convex polygons (counter-clockwise or clockwise, consistent) -> (m,2)."""
    def ccw(p):
        a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        return p if a >= 0 else p[::-1]
    out = list(ccw(np.asarray(subject, float)))
    C = ccw(np.asarray(clip, float))
    for i in range(len(C)):
        a, b = C[i], C[(i + 1) % len(C)]
        inp, out = out, []
        if not inp:
            break
        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12
        def inter(p, q):
            d1, d2 = p - q, a - b
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-15:
                return q
            t = ((p[0] - a[0]) * d2[1] - (p[1] - a[1]) * d2[0]) / den
            return p - t * d1
        for j in range(len(inp)):
            p, q = inp[j], inp[j - 1]
            if inside(p):
                if not inside(q):
                    out.append(inter(q, p))
                out.append(p)
            elif inside(q):
                out.append(inter(q, p))
    return np.array(out).reshape(-1, 2)


def poly_area(p):
    if len(p) < 3:
        return 0.0
    return float(abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))


def point_in_poly_margin(pt, poly):
    """Signed distance from pt to the boundary of a convex polygon: > 0 inside, < 0 outside."""
    P = np.asarray(poly, float)
    a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    if a < 0:
        P = P[::-1]
    ds = []
    for i in range(len(P)):
        a_, b_ = P[i], P[(i + 1) % len(P)]
        e = b_ - a_
        nrm = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-12)   # inward normal for a ccw polygon
        ds.append((np.asarray(pt) - a_) @ nrm)
    return float(min(ds))
