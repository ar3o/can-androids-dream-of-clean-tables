"""Pose fitting through the real phone camera: a mesh or cuboid is moved until its projected silhouette matches a SAM
3 mask.

Masks are 960x540 bool; `care` marks the pixels that count (occluders such as other objects or the hand are excluded).
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

from cleantables.twin.inputs import IMG_H, IMG_W
from cleantables.twin.projection import project

IMG_SHAPE = (IMG_H, IMG_W)
DIM_TOLERANCE = 0.22  # free-size cuboid fits may leave the depth-measured size by about this log-ratio for free


def box_corners(x: float, y: float, yaw: float, L: float, W: float, H: float, z0: float = 0.0) -> np.ndarray:
    """8 corners: bottom (-,-) (+,-) (+,+) (-,+), then the same at the top."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dz in (0.0, H):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            px, py = sx * L / 2, sy * W / 2
            out.append([x + c * px - s * py, y + s * px + c * py, z0 + dz])
    return np.array(out)


def poly_mask(P: np.ndarray, K: np.ndarray, T: np.ndarray, shape=IMG_SHAPE) -> np.ndarray:
    """filled convex hull of the projected points (empty if any point is behind the camera)."""
    uv, z = project(P, K, T)
    m = np.zeros(shape, np.uint8)
    if (z <= 0.01).any():
        return m.astype(bool)
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def fit_cuboid(mask, K, T, p0, fix_dims=None, z0=0.0, care=None, free_scale=False, scale_bounds=(0.7, 1.4)) -> dict:
    """refine (x, y, yaw, L, W, H) so the projected cuboid matches `mask`.

    fix_dims: (L, W, H) kept fixed (pose only); free_scale: one isotropic factor on fix_dims within scale_bounds.
    Nelder-Mead from p0 and from p0 turned by 90 degrees (L and W swapped).
    """
    care = np.ones_like(mask) if care is None else care
    target = mask & care

    def loss(q):
        x, y, yw = q[:3]
        if free_scale:
            if not scale_bounds[0] <= q[3] <= scale_bounds[1]:
                return 2.0
            L, Wd, Hh = np.asarray(fix_dims) * q[3]
        else:
            L, Wd, Hh = fix_dims if fix_dims is not None else np.abs(q[3:6])
        if min(L, Wd, Hh) < 0.01:
            return 2.0
        penalty = 0.0
        if fix_dims is None:
            # a silhouette alone cannot tell size from distance: keep the dims near the depth-measured box
            r = np.abs(np.log(np.array([L, Wd, Hh]) / np.maximum(np.array([p0[3], p0[4], p0[5]]), 1e-3)))
            penalty = 2.0 * float(np.clip(r - DIM_TOLERANCE, 0, None).sum())
        pm = poly_mask(box_corners(x, y, yw, L, Wd, Hh, z0), K, T) & care
        return 1.0 - iou(pm, target) + penalty

    best = None
    x, y, yw, L, Wd, Hh = p0
    for q0 in ([x, y, yw, L, Wd, Hh], [x, y, yw + math.pi / 2, Wd, L, Hh]):
        q0 = np.array((q0[:3] + [1.0]) if free_scale else (q0[:3] if fix_dims is not None else q0), float)
        step = np.array([0.02, 0.02, 0.15, 0.05] if free_scale else [0.02, 0.02, 0.15, 0.02, 0.02, 0.02])[: len(q0)]
        simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * step[i] for i in range(len(q0))])
        r = minimize(
            loss, q0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxiter=400, xatol=1e-4, fatol=1e-4)
        )
        if best is None or r.fun < best.fun:
            best = r
    q = best.x
    sc = float(q[3]) if free_scale else 1.0
    dims = np.asarray(fix_dims) * sc if free_scale else (fix_dims if fix_dims is not None else np.abs(q[3:6]))
    return dict(
        x=float(q[0]),
        y=float(q[1]),
        yaw=float(q[2]),
        L=float(dims[0]),
        W=float(dims[1]),
        H=float(dims[2]),
        iou=1 - float(best.fun),
        scale=sc,
    )


_CUBE_ROTATIONS: list[np.ndarray] = []


def cube_rotations() -> list[np.ndarray]:
    """the 24 proper rotations of a cube (signed permutation matrices with det +1)."""
    if not _CUBE_ROTATIONS:
        for p in itertools.permutations(range(3)):
            for sg in itertools.product((1, -1), repeat=3):
                M = np.zeros((3, 3))
                for i in range(3):
                    M[i, p[i]] = sg[i]
                if np.linalg.det(M) > 0:
                    _CUBE_ROTATIONS.append(M)
    return _CUBE_ROTATIONS


def footprint_mask(mesh: trimesh.Trimesh, x, y, yaw, K, T, shape=IMG_SHAPE) -> np.ndarray:
    """projected silhouette of the mesh's upward-facing triangles (enough for objects seen from above)."""
    c, s = math.cos(yaw), math.sin(yaw)
    V = mesh.vertices @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + np.array([x, y, 0])
    uv, _ = project(V, K, T)
    m = np.zeros(shape, np.uint8)
    tri = uv[mesh.faces].astype(np.int32)
    keep = mesh.face_normals[:, 2] > 0.5
    for t in tri[keep]:
        cv2.fillConvexPoly(m, t, 1)
    return m.astype(bool)


def scaled(mesh: trimesh.Trimesh, s: float, flat: bool = True) -> trimesh.Trimesh:
    """copy scaled by s in x / y, and in z too unless flat (flat objects keep their thickness prior)."""
    m = mesh.copy()
    m.vertices = m.vertices * np.array([s, s, 1.0 if flat else s])
    return m


def flip_over(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """turn upside down about x, bottom back at z = 0 (in place, returned for chaining)."""
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0)))
    mesh.apply_translation((0, 0, -mesh.bounds[0][2]))
    return mesh


def fit_flat(
    mesh, mask, K, T, x, y, yaw, flips=(False, True), care=None, free_scale=False, scale_bounds=(0.7, 1.4)
) -> dict:
    """pose (x, y, yaw, flipped) of a lying mesh: 10-deg yaw sweep x flip, then Nelder-Mead on (x, y, yaw[, scale])."""
    care = np.ones_like(mask) if care is None else care
    target = mask & care
    best = (-1, None)
    for fl in flips:
        m = mesh.copy()
        if fl:
            flip_over(m)
        for yw in np.radians(np.arange(0, 360, 10)):
            v = iou(footprint_mask(m, x, y, yw, K, T) & care, target)
            if v > best[0]:
                best = (v, (fl, yw, m))
    fl, yw0, m = best[1]

    def loss(q):
        mm = scaled(m, q[3]) if free_scale else m
        if free_scale and not scale_bounds[0] <= q[3] <= scale_bounds[1]:
            return 2.0
        return 1 - iou(footprint_mask(mm, q[0], q[1], q[2], K, T) & care, target)

    q0 = np.array([x, y, yw0] + ([1.0] if free_scale else []))
    simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * [0.01, 0.01, 0.1, 0.05][i] for i in range(len(q0))])
    r = minimize(
        loss, q0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxiter=300, xatol=1e-4, fatol=1e-4)
    )
    return dict(
        x=float(r.x[0]),
        y=float(r.x[1]),
        yaw=float(r.x[2]),
        flipped=bool(fl),
        iou=1 - float(r.fun),
        scale=float(r.x[3]) if free_scale else 1.0,
    )
