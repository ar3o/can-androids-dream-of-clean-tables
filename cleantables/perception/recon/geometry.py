"""Geometry helpers for the per-clip reconstruction: back-projection, RANSAC planes, oriented rectangles, rotations (numpy / cv2 only)."""
from __future__ import annotations

import cv2
import numpy as np


def backproject(depth, K, mask=None, stride=1):
    """Depth (h,w) metres -> points (N,3) camera frame (+ pixel coords (N,2) u,v) for mask & depth>0."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    ok = d > 0
    if mask is not None:
        ok &= mask[::stride, ::stride]
    u, v, d = u[ok].astype(np.float64), v[ok].astype(np.float64), d[ok].astype(np.float64)
    X = (u - K[0, 2]) / K[0, 0] * d
    Y = (v - K[1, 2]) / K[1, 1] * d
    return np.stack([X, Y, d], 1), np.stack([u, v], 1)


def project(P, K):
    P = np.asarray(P, float)
    z = np.where(np.abs(P[..., 2]) < 1e-9, 1e-9, P[..., 2])
    return np.stack([K[0, 0] * P[..., 0] / z + K[0, 2], K[1, 1] * P[..., 1] / z + K[1, 2]], -1)


def fit_plane_lsq(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    return n, -n @ c


def fit_plane_ransac(P, thresh, iters=300, rng=None):
    """-> (n, d, inliers) with n.p + d = 0, refined by least squares on the inliers. n oriented towards the camera (d > 0 <=> camera above)."""
    rng = rng or np.random.default_rng(0)
    best, bi = None, None
    N = len(P)
    if N < 3:
        return None, None, np.zeros(N, bool)
    for _ in range(iters):
        s = P[rng.choice(N, 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d = -n @ s[0]
        inl = np.abs(P @ n + d) < thresh
        if bi is None or inl.sum() > bi.sum():
            best, bi = (n, d), inl
    n, d = fit_plane_lsq(P[bi])
    inl = np.abs(P @ n + d) < thresh
    n, d = fit_plane_lsq(P[inl])
    if d < 0:               # camera origin on the positive side: n points towards the camera
        n, d = -n, -d
    return n, d, np.abs(P @ n + d) < thresh


def plane_basis(n, xhint=np.array([1.0, 0, 0])):
    """Right-handed basis (ex, ey, n) with ex = xhint projected on the plane."""
    ex = xhint - (xhint @ n) * n
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([0, 1.0, 0]) - n[1] * n
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    return ex, ey


def min_area_rect(xy, trim=0.002, outlier=0.01):
    """Trimmed oriented rectangle of 2D points: drop the `outlier` fraction farthest from the median, cv2.minAreaRect, then set each side to
    the [trim, 1-trim] quantiles along the rect axes. -> centre (2,), (L, W) with L >= W, yaw of the L axis (rad, in (-pi/2, pi/2]).
    trim is small on purpose: in perspective the point density along the object falls off with distance, so a 2 % quantile cut the far end
    (laptop 0.31 -> 0.27 m; checked on IMG_2480_ep06)."""
    xy = np.asarray(xy, np.float64)
    if len(xy) < 5:
        return None
    r = np.linalg.norm(xy - np.median(xy, 0), axis=1)
    keep = r <= np.quantile(r, 1 - outlier)
    (cx, cy), (a, b), ang = cv2.minAreaRect(xy[keep].astype(np.float32))
    th = np.radians(ang)
    ax = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    q = (xy[keep] - [cx, cy]) @ ax.T
    lo, hi = np.quantile(q, trim, 0), np.quantile(q, 1 - trim, 0)
    ext = hi - lo
    c = np.array([cx, cy]) + ((lo + hi) / 2) @ ax
    if ext[0] >= ext[1]:
        L, W, yaw = ext[0], ext[1], th
    else:
        L, W, yaw = ext[1], ext[0], th + np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return c, (float(L), float(W)), float(yaw)


def quat_from_R(R):
    """(w, x, y, z)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def R_from_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_corners(pos, R, size):
    s = np.asarray(size) / 2
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
    return c @ np.asarray(R).T + pos


def point_box_signed_distance(P, pos, R, size):
    """Signed distance (m) of points P (N,3) to an oriented box (negative inside)."""
    q = (np.asarray(P) - pos) @ np.asarray(R)        # into box frame
    h = np.asarray(size) / 2
    d = np.abs(q) - h
    out = np.linalg.norm(np.maximum(d, 0), axis=-1)
    inside = np.minimum(d.max(-1), 0)
    return out + inside
