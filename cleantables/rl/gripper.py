"""Gripper-shaped hand proxy, sized after a Franka hand.

One rule for every carried object, from its own size only:
  grasp axis  the object's short horizontal axis (in its baked frame: x if extents x <= y, else y); the fingers close along it
  fingers     two boxes FT (along the grasp axis) x FW x FL, one each side, FG outside the object's half-width on that axis; they cover
              the object's top FC, or reach down to FB above its bottom if it is shorter (fingertip z = max(bottom + FB, top - FC))
  palm        a box PL (along the grasp axis) x PW x PH sitting on top of the fingers (palm bottom = fingertip z + FL); with the numbers
              below it is always >= 1.4 cm above the object's top
Sizes after a Franka hand (housing ~20 x 6.5 x 7 cm, fingers ~2 cm wide, ~5.4 cm long). Frame: the carried object's baked local frame
(the frame its extents are given in), origin = its root (bottom centre), z up. The hand turns with the object's yaw (about z only).
The simulator (kinematic cuboids), the route planner (extra shell points of the carried object) and the reward (the same points)
all use this module, so they see the same hand."""
from __future__ import annotations

import os
import numpy as np

PL, PW, PH = 0.20, 0.065, 0.07          # palm: along the grasp axis, across, height
FT, FW, FL = 0.012, 0.02, 0.054         # finger: thickness (along the grasp axis), width, length
FG, FC, FB = 0.005, 0.04, 0.01          # finger gap to the object's side, coverage of the top, clearance above the bottom


def default_gyaw(ext):
    """Default grasp yaw (baked frame): the fingers close along the object's short horizontal axis."""
    return 0.0 if float(ext[0]) <= float(ext[1]) else float(np.pi / 2)


def half_width(ext, gyaw):
    """The object's half-width along the grasp direction (cos g, sin g) of its baked frame (box support function)."""
    return 0.5 * (abs(float(ext[0]) * np.cos(gyaw)) + abs(float(ext[1]) * np.sin(gyaw)))


_MW = {}


def mesh_opening(o, ext, gyaw):
    """Finger opening (m) a top-down grasp at yaw gyaw (baked frame) needs, measured on the mesh rather than the box (for the 8 cm
    opening check): the object's occupancy points (physics screen cache: object frame, scaled; rotated by yaw_baked into the baked frame, origin = the root =
    where the hand centres) inside the fingers' slab (|across| <= FW / 2, z from the fingertip height to the top), 2 x their largest distance
    from the grasp centre along the closing direction, + both gaps. -> opening (m), or inf when the slab holds no object point (the fingers
    would close on air)."""
    from cleantables.worldmodel.physics import screen as PSc
    from pathlib import Path
    u = str(o["usd"]); root = str(Path(__file__).resolve().parents[2]) + "/"
    key = (u[len(root):] if u.startswith(root) else u, float(o.get("scale") or 1.0), float(o.get("yaw_baked", 0.0)))
    if key not in _MW:
        scr = _MW.setdefault("_scr", PSc.Screen())
        P = scr._obj(dict(o, usd=key[0]))
        _MW[key] = None if P is None else PSc.Screen._pose(np.asarray(P, np.float64), key[2])
    P = _MW[key]
    if P is None:
        return opening(ext, gyaw)
    g = float(gyaw); d = np.array([np.cos(g), np.sin(g)]); n = np.array([-d[1], d[0]])
    h = float(ext[2]); tip = max(FB, h - FC)
    m = (np.abs(P[:, :2] @ n) <= FW / 2) & (P[:, 2] >= tip - 1e-3)
    if not m.any():
        return float("inf")
    return 2 * (float(np.abs(P[m, :2] @ d).max()) + FG)


def boxes_g(ext, gyaw=None):
    """Hand boxes for a top-down grasp at grasp yaw gyaw (rad, baked frame; None = default_gyaw): list of (centre (3,) in the
    object's baked root frame, half size (3,) in the hand frame (x = the grasp direction), hand yaw = gyaw): [palm, finger -, finger +]."""
    g = default_gyaw(ext) if gyaw is None else float(gyaw)
    h = float(ext[2]); hw = half_width(ext, g)
    tip = max(FB, h - FC); pb = tip + FL
    d = np.array([np.cos(g), np.sin(g), 0.0])
    out = [(np.array([0.0, 0.0, pb + PH / 2]), np.array([PL / 2, PW / 2, PH / 2]), g)]
    for sgn in (-1.0, 1.0):
        c = sgn * (hw + FG + FT / 2) * d; c[2] = tip + FL / 2
        out.append((c, np.array([FT / 2, FW / 2, FL / 2]), g))
    return out


def opening(ext, gyaw=None):
    """Finger opening the box grasp needs (m): the object's width along the grasp direction + both gaps."""
    g = default_gyaw(ext) if gyaw is None else float(gyaw)
    return 2 * (half_width(ext, g) + FG)


def boxes(ext):
    """ext (3,) the object's extents (baked frame) -> list of (centre (3,), half size (3,)) in the object's baked root frame:
    [palm, finger -, finger +]. The grasp axis is the short horizontal one."""
    ex, ey, h = float(ext[0]), float(ext[1]), float(ext[2])
    ax = 0 if ex <= ey else 1                                   # grasp axis index
    half_w = (ex if ax == 0 else ey) / 2
    tip = max(FB, h - FC)                                        # fingertip z (root frame: bottom = 0)
    pb = tip + FL                                                # palm bottom
    out = []
    pc = np.zeros(3); pc[2] = pb + PH / 2
    ph = np.array([PL / 2, PW / 2, PH / 2]) if ax == 0 else np.array([PW / 2, PL / 2, PH / 2])
    out.append((pc, ph))
    for sgn in (-1.0, 1.0):
        c = np.zeros(3); c[ax] = sgn * (half_w + FG + FT / 2); c[2] = tip + FL / 2
        hs = np.array([FT / 2, FW / 2, FL / 2]) if ax == 0 else np.array([FW / 2, FT / 2, FL / 2])
        out.append((c, hs))
    return out


BALL_R, BALL_GAP = 0.045, 0.01         # ball proxy (FP_HAND=ball): sphere r 4.5 cm, 1 cm above the object's top


_CACHE = {}


def points(ext, n_palm=64, n_finger=24, seed=0, mode=None, gyaw=None):
    """Hand surface points, cached per (size, settings); deterministic for a given object size. gyaw: the grasp yaw
    (None = the default short-axis grasp)."""
    import os
    key = (tuple(np.round(np.asarray(ext, np.float64), 6)), n_palm, n_finger, seed, mode or os.environ.get("FP_HAND", "gripper"),
           None if gyaw is None else round(float(gyaw), 5))
    if key not in _CACHE:
        _CACHE[key] = _points(ext, n_palm, n_finger, seed, mode) if gyaw is None else _points_g(ext, n_palm, n_finger, seed, gyaw)
    return _CACHE[key]


def _points_g(ext, n_palm, n_finger, seed, gyaw):
    """As _points for the yawed grasp boxes of boxes_g."""
    rng = np.random.default_rng(seed); P = []
    for (c, hs, g), n in zip(boxes_g(ext, gyaw), (n_palm, n_finger, n_finger)):
        f = rng.integers(0, 6, n); u = rng.uniform(-1, 1, (n, 3))
        ax = f // 2; sg = np.where(f % 2 == 0, -1.0, 1.0)
        u[np.arange(n), ax] = sg
        q = u * hs; cg, sg_ = np.cos(g), np.sin(g)
        q = np.stack([cg * q[:, 0] - sg_ * q[:, 1], sg_ * q[:, 0] + cg * q[:, 1], q[:, 2]], 1)
        P.append(c + q)
    return np.concatenate(P).astype(np.float32)


def _points(ext, n_palm=64, n_finger=24, seed=0, mode=None):
    """Surface points of the hand (n_palm + 2 n_finger, 3) in the object's baked root frame (uniform on each box's faces). mode "ball"
    (default from FP_HAND): 112 points on the ball proxy's sphere instead."""
    import os
    rng = np.random.default_rng(seed)
    if (mode or os.environ.get("FP_HAND", "gripper")) == "ball":
        v = rng.normal(size=(n_palm + 2 * n_finger, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        return (v * BALL_R + np.array([0.0, 0.0, float(ext[2]) + BALL_GAP + BALL_R])).astype(np.float32)
    P = []
    for (c, hs), n in zip(boxes(ext), (n_palm, n_finger, n_finger)):
        f = rng.integers(0, 6, n); u = rng.uniform(-1, 1, (n, 3))
        ax = f // 2; sg = np.where(f % 2 == 0, -1.0, 1.0)
        u[np.arange(n), ax] = sg
        P.append(c + u * hs)
    return np.concatenate(P).astype(np.float32)


def box_sizes():
    """The three kinematic cuboids' full sizes (fixed at build) for the grasp axis along local x: palm, finger, finger."""
    return [(PL, PW, PH), (FT, FW, FL), (FT, FW, FL)]
