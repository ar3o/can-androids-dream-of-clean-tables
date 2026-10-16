"""Geometry helpers: back-projection, the table plane and frame, rectangles and polygons in the table plane.

Table frame: z is the table normal (up), the origin is where the principal ray meets the table, x is image-right
projected onto the table; metres. Depth maps and masks are 270x480 (half the 960x540 work frame); K is at 960x540.
"""

from __future__ import annotations

import cv2
import numpy as np

from .files import H_LO, W_LO

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
PLANE_THRESHOLD_M = 0.012
SCALE_RANGE = (0.75, 1.33)
MIN_SCALE_PIXELS = 200


def backproject(depth_lo: np.ndarray, K: np.ndarray, mask_lo=None):
    """Camera-frame points (n,3) of the valid (masked) pixels of a 270x480 depth map, and their pixel indices."""
    ys, xs = np.nonzero((depth_lo > 0) & (mask_lo if mask_lo is not None else True))
    z = depth_lo[ys, xs]
    u, v = xs * 2 + 0.5, ys * 2 + 0.5
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1), (ys, xs)


def to_table(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def project(P: np.ndarray, K: np.ndarray, T: np.ndarray):
    """Table-frame points (n,3) -> pixels (n,2) at 960x540 and camera depth (n,)."""
    Ti = np.linalg.inv(T)
    pc = np.atleast_2d(P) @ Ti[:3, :3].T + Ti[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6), pc[:, 2]


def ransac_plane(P: np.ndarray, it: int = 400, thr: float = 0.01, seed: int = 0):
    """(normal towards the camera, inlier centroid, inlier fraction); the normal is refined by SVD on the inliers."""
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(it):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = (np.abs((P - a) @ n) < thr).sum()
        if count > best[0]:
            best = (count, (n, a))
    n, a = best[1]
    inliers = np.abs((P - a) @ n) < thr
    c = P[inliers].mean(0)
    n = np.linalg.svd(P[inliers] - c)[2][2]
    if n @ c > 0:
        n = -n
    return n, c, float(inliers.mean())


def table_frame(n: np.ndarray, c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """4x4 camera -> table transform for the plane (n, c)."""
    ray = np.array([0, 0, 1.0])
    origin = (n @ c) / (n @ ray) * ray
    x = np.array([1.0, 0, 0]) - n * n[0]
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    return T


def fit_table(depths, K, masks_excl=None, seed: int = 0, up=(0, -1.0, 0)):
    """Table plane from several depth maps, ignoring the excluded pixels (objects, hands).

    Per frame up to three RANSAC planes are peeled off; the table is the largest one whose normal points up in the image
    (n . up > 0.25; up = (0,-1,0) for landscape video, (1,0,0) for portrait video rotated clockwise) or faces the camera
    (|n_z| > 0.9, near top-down views). Normals and centres are pooled by the median.
    Returns (T camera -> table, n, c, info).
    """
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
        candidates = []
        for _ in range(3):
            if len(rest) < 500:
                break
            n, c, _ = ransac_plane(rest, it=200, thr=PLANE_THRESHOLD_M, seed=int(rng.integers(1e9)))
            inliers = np.abs((rest - c) @ n) < PLANE_THRESHOLD_M
            candidates.append((int(inliers.sum()), n, c))
            rest = rest[~inliers]
        upward = [x for x in candidates if x[1] @ np.asarray(up, float) > 0.25 or abs(x[1][2]) > 0.9]
        if upward:
            planes.append(max(upward, key=lambda x: x[0]))
    if not planes:
        raise RuntimeError("no table plane found")
    n = np.median(np.array([p[1] for p in planes]), 0)
    n /= np.linalg.norm(n)
    c = np.median(np.array([p[2] for p in planes]), 0)
    T = table_frame(n, c, K)
    info = dict(
        n_frames=len(planes),
        camera_height_m=round(float(T[2, 3]), 3),
        tilt_deg=round(float(np.degrees(np.arccos(abs(n @ np.array([0, 0, 1.0]))))), 1),
    )
    return T, n, c, info


def plane_z_map(T: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera depth of the table plane at every 270x480 pixel, NaN where the ray misses it."""
    normal, offset = T[2, :3], T[2, 3]
    ys, xs = np.mgrid[0:H_LO, 0:W_LO]
    rays = np.stack(
        [(xs * 2 + 0.5 - K[0, 2]) / K[0, 0], (ys * 2 + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1
    )
    den = rays @ normal
    with np.errstate(all="ignore"):
        z = -offset / den
    z[(z <= 0) | ~np.isfinite(z)] = np.nan
    return z.astype(np.float32)


def depth_scale(D: np.ndarray, zp: np.ndarray, free=None) -> float:
    """Per-frame scale that puts the free table pixels on the reference plane (median ratio)."""
    ok = np.isfinite(zp) & (D > 0)
    if free is not None:
        ok &= free
    r = zp[ok] / D[ok]
    r = r[(r > SCALE_RANGE[0]) & (r < SCALE_RANGE[1])]
    return float(np.median(r)) if len(r) > MIN_SCALE_PIXELS else 1.0


def table_mask(depths, K, T, thr: float = 0.02, frac: float = 0.3) -> np.ndarray:
    """Pixels on the table plane (|height| < thr) in more than frac of the frames, filled to their convex hull."""
    zp = plane_z_map(T, K)
    acc = np.zeros((H_LO, W_LO))
    for D in depths:
        s = depth_scale(D, zp)
        P, (ys, xs) = backproject(D * s, K)
        on = np.abs(to_table(T, P)[:, 2]) < thr
        acc[ys[on], xs[on]] += 1
    m = (acc / max(1, len(depths)) > frac).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((H_LO, W_LO), bool)
    hull = np.zeros((H_LO, W_LO), np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(max(contours, key=cv2.contourArea))], 1)
    return cv2.dilate(hull, np.ones((15, 15), np.uint8)) > 0


def table_extent(T: np.ndarray, K: np.ndarray, table: np.ndarray) -> list[list[float]]:
    """[[x_lo, x_hi], [y_lo, y_hi]] of the table pixels on the plane (2nd-98th percentile), table frame."""
    normal, offset = T[2, :3], T[2, 3]
    ys, xs = np.nonzero(table[::3, ::3])
    ys, xs = ys * 3, xs * 3
    rays = np.stack([(xs * 2 + 0.5 - K[0, 2]) / K[0, 0], (ys * 2 + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1)
    t = -offset / (rays @ normal)
    P = to_table(T, rays[t > 0] * t[t > 0, None])
    lo, hi = np.percentile(P[:, :2], 2, 0), np.percentile(P[:, :2], 98, 0)
    return [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]


def rect_extent(xy: np.ndarray):
    """Oriented rectangle of 2D points, 3rd-97th percentile extent: (cx, cy, L, W, yaw of the long side in [0, pi))."""
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


def rect_corners(x, y, L, W, yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = np.array([c, s]) * L / 2, np.array([-s, c]) * W / 2
    return np.array([[x, y]]) + np.array([u + v, -u + v, -u - v, u - v])


def box_corners3d(x, y, z0, L, W, H, yaw) -> np.ndarray:
    """8 corners (bottom four, then top four) of a box whose bottom face is at height z0."""
    r = rect_corners(x, y, L, W, yaw)
    return np.concatenate([np.c_[r, np.full(4, z0)], np.c_[r, np.full(4, z0 + H)]])


def _counter_clockwise(p: np.ndarray) -> np.ndarray:
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if area >= 0 else p[::-1]


def poly_clip(subject, clip) -> np.ndarray:
    """Intersection of two convex polygons (Sutherland-Hodgman)."""
    out = list(_counter_clockwise(np.asarray(subject, float)))
    C = _counter_clockwise(np.asarray(clip, float))
    for i in range(len(C)):
        a, b = C[i], C[(i + 1) % len(C)]
        points, out = out, []
        if not points:
            break

        def inside(p, a=a, b=b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

        def intersect(p, q, a=a, b=b):
            d1, d2 = p - q, a - b
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-15:
                return q
            t = ((p[0] - a[0]) * d2[1] - (p[1] - a[1]) * d2[0]) / den
            return p - t * d1

        for j in range(len(points)):
            p, q = points[j], points[j - 1]
            if inside(p):
                if not inside(q):
                    out.append(intersect(q, p))
                out.append(p)
            elif inside(q):
                out.append(intersect(q, p))
    return np.array(out).reshape(-1, 2)


def poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    return float(abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))
