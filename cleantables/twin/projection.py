"""Pinhole projection between the phone image (960x540) and the table frame (z up, table top z = 0, metres).

T is the camera -> table transform of det3d.npz, K the intrinsics at 960x540.
"""

from __future__ import annotations

import cv2
import numpy as np

from cleantables.twin.inputs import IMG_H, IMG_W, H, W


def rays_lo(K: np.ndarray) -> np.ndarray:
    """camera-frame rays (z = 1) of the 270x480 depth grid (pixel centres in 960x540 coordinates)."""
    vv, uu = np.mgrid[0:H, 0:W]
    u, v = uu * 2 + 0.5, vv * 2 + 0.5
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], -1)


def project(P: np.ndarray, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """table-frame points -> pixels (960x540) and camera depth."""
    Ti = np.linalg.inv(T)
    pc = P @ Ti[:3, :3].T + Ti[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def plane_hit(px: np.ndarray, K: np.ndarray, T: np.ndarray, z: float = 0.0) -> np.ndarray:
    """pixels (N, 2) at 960x540 -> points on the horizontal plane at height z (table frame)."""
    rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T
    d = rays @ T[:3, :3].T
    o = T[:3, 3]
    s = (z - o[2]) / d[:, 2]
    return o[None] + s[:, None] * d


def table_extent(T: np.ndarray, K: np.ndarray, table_mask: np.ndarray) -> list[list[float]]:
    """[[x0, x1], [y0, y1]] of the table top: 2nd-98th percentile of the table-mask pixels hit on the plane z = 0."""
    nrm, off = T[2, :3], T[2, 3]
    ys, xs = np.nonzero(table_mask[::3, ::3])
    ys, xs = ys * 3, xs * 3
    rays = np.stack([(xs * 2 + 0.5 - K[0, 2]) / K[0, 0], (ys * 2 + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1)
    t = -off / (rays @ nrm)
    pc = rays[t > 0] * t[t > 0, None]
    P = pc @ T[:3, :3].T + T[:3, 3]
    lo, hi = np.percentile(P[:, :2], 2, 0), np.percentile(P[:, :2], 98, 0)
    return [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]


def ortho(
    img: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z: float = 0.0,
    res: float = 0.002,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """orthophoto of the plane at height z over [x0, x1] x [y0, y1] (image rows: +y at the top), plus the pixel maps."""
    gx, gy = np.meshgrid(np.arange(x0, x1, res), np.arange(y1, y0, -res))
    G = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z)]
    uv, _ = project(G, K, T)
    mx, my = uv[:, 0].reshape(gx.shape).astype(np.float32), uv[:, 1].reshape(gx.shape).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), (mx, my)


def lab(rgb) -> np.ndarray:
    """CIE L*a*b* of one RGB colour (OpenCV 8-bit scaling)."""
    a = np.uint8([[np.clip(rgb, 0, 255)]])
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[0, 0].astype(float)


__all__ = ["IMG_H", "IMG_W"]
