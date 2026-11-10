"""Object-frame footprints for the world-model bridge tokens.

Twin meshes are stored with their rotation baked in, so a twin's `extents` (scene.json / wm_tokens.npz) are the axis-aligned
box of the rotated mesh. WMBridge._batch reads them as the object's own length and width and applies its long-side rule on
top of yaw_baked, so thin objects at an angle come out about 1.6x too wide with the long side often 90 deg off. The world
model was trained on each object's own box and long-side yaw.
Here, for each twin mesh, the minimum-area footprint rectangle (length >= width, in the table plane) and the yaw of its long
side are computed from the vertices; the tokens get extents[:, :2] = (length, width) and yaw_baked = that yaw, so the bridge's
formulas give the object's own box. Heights are unchanged. scene.json, the USD meshes and the physics keep the baked yaw;
only the world-model inputs change.

  self-check: $VLA_PY -m cleantables.worldmodel.bridge_obb IMG_2536_t01
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cleantables.worldmodel.common import ROOT

_CACHE = {}


def obj_vertices(path):
    V = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
    return np.asarray(V, np.float64)


def footprint(V, step_deg=0.5):
    """Return (length, width, long-side yaw in [0, pi)) of the minimum-area rectangle around the vertices' x, y."""
    xy = V[:, :2] - V[:, :2].mean(0)
    best = None
    for a in np.radians(np.arange(0.0, 90.0, step_deg)):
        c, s = np.cos(a), np.sin(a)
        q = xy @ np.array([[c, -s], [s, c]])                                         # coordinates along (cos a, sin a) and its normal
        e = q.max(0) - q.min(0)
        if best is None or e[0] * e[1] < best[0]:
            best = (e[0] * e[1], a, e)
    _, a, e = best
    L, W = float(max(e)), float(min(e))
    yl = a if e[0] >= e[1] else a + np.pi / 2
    return L, W, float(np.mod(yl, np.pi))


def obb_Z(ep, Z, S):
    """Copy of the bridge tokens Z with object-frame footprints (see module docstring); S = scene.json."""
    key = (ep, len(Z["ids"]))
    if key not in _CACHE:
        rows = []
        for k, o in enumerate(S["objects"][:len(Z["ids"])]):
            mesh = ROOT / str(o["usd"]).replace(".usda", ".obj")
            sc = float(o.get("scale") or 1.0)
            if not mesh.exists():
                rows.append(None); continue
            L, W, yl = footprint(obj_vertices(mesh) * sc)
            rows.append((L, W, yl))
        _CACHE[key] = rows
    Z = dict(Z)
    ex = np.array(Z["extents"], np.float32).copy(); yb = np.array(Z["yaw_baked"], np.float32).copy()
    for k, r in enumerate(_CACHE[key]):
        if r is None:
            continue
        ex[k, 0], ex[k, 1], yb[k] = r[0], r[1], r[2]
    Z["extents"], Z["yaw_baked"] = ex, yb
    return Z


if __name__ == "__main__":
    import json
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel import bridge as v2_bridge  # noqa: F401
    ep = sys.argv[1] if len(sys.argv) > 1 else "IMG_2536_t01"
    S = json.load(open(W.EP / ep / "scene.json")); z = np.load(W.EP / ep / "wm_tokens.npz"); Z = {k: z[k] for k in z.files}
    Z2 = obb_Z(ep, Z, S)
    for k, nm in enumerate(Z["ids"]):
        e0, e1 = Z["extents"][k][:2]; y0 = Z["yaw_baked"][k] + (np.pi / 2 if e1 > e0 else 0)
        print(f"{str(nm):14s} old {max(e0, e1):.3f} x {min(e0, e1):.3f} long-yaw {np.degrees(y0) % 180:5.1f} | "
              f"new {Z2['extents'][k][0]:.3f} x {Z2['extents'][k][1]:.3f} long-yaw {np.degrees(Z2['yaw_baked'][k]) % 180:5.1f}")
