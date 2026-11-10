"""Geometric path cost for the physics stage: path collisions scored without simulation, with one rule for every object
(no per-category or per-object choice).

GeoCost
  carried object  its rotation-invariant bounds: a vertical cylinder, radius = max horizontal distance from its centre to its own
                  filled occupancy (+ half a voxel), height = its occupancy height (+ half a voxel)
  every other     its twin mesh's filled 5 mm voxel occupancy (cache of physics.screen: trimesh voxelize + fill, so hollow objects
                  stay hollow because their meshes are), posed as the PhysX chain replay poses the bodies
  per reading     separation of the cylinder from each other object: min over its voxels of max(radial distance - R, below the
                  base, above the top) - half a voxel; graded like the replay's hit: 0 at >= 1 cm away, 1 touching, 2 at >= 1 cm
                  inside (z = clamp((1 cm - sep) / 1 cm, 0, 1) + clamp(-sep / 1 cm, 0, 1)); max over the other objects; objects
                  within 1 cm at the start reading are exempt only while they stay within 1 cm; the table is not a counted body
  geo_hit         mean grade over 19 readings at 5-95% of the path length (the replay's u; the path shifted to start where the
                  object is, (1 - u) x offset) -> [0, 2], deterministic, CPU
  econ3d          3D path length (vertical included) / max(straight 3D start -> end distance, 2 cm) - 1, so lifting is not free
  econ="xy"       economy in the table plane only (xy length / max(straight xy, 2 cm) - 1): lifting is free
  w_drag > 0      drag: per reading, the cylinder base's clearance above whatever lies beneath its footprint (table top z = 0, or
                  the highest voxel of any other object within radius R below the base), graded like the hit; no start exemption,
                  so sliding is never free; mean over the same 19 readings -> geo_drag
  cost            W_HIT x geo_hit + w_drag x geo_drag + W_ECON x econ
MeshCost replaces the cylinder with the carried object's own shape against signed distance fields (see its docstring).
"""
from __future__ import annotations

import numpy as np

from cleantables.worldmodel.physics.screen import N_READ, PITCH, Screen

HIT_ZONE = 0.01
W_HIT, W_ECON = 10.0, 0.5


def zgrade(sep):
    return np.clip((HIT_ZONE - sep) / HIT_ZONE, 0, 1) + np.clip(-sep / HIT_ZONE, 0, 1)


def econ3d(path, start, dims=3):
    p = np.asarray(path, np.float64)[:, :dims]
    L = np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()
    return max(0.0, L / max(float(np.linalg.norm(p[-1] - np.asarray(start, np.float64)[:dims])), 0.02) - 1.0)


class GeoCost(Screen):
    """Posed occupancies of every object (cache and posing of screen.Screen) and the carried object's swept cylinder."""

    def __init__(self, econ="3d", w_drag=0.0):
        super().__init__(frac=1.0)
        self._cyl, self._posed = {}, {}
        self.econ_dims, self.w_drag = (2 if econ == "xy" else 3), float(w_drag)

    def cylinder(self, o):
        k = (str(o["usd"]), float(o.get("scale") or 1.0))
        if k not in self._cyl:
            P = self._obj(o)
            self._cyl[k] = None if P is None else (float(np.linalg.norm(P[:, :2], axis=1).max()) + PITCH / 2, float(P[:, 2].max()) + PITCH / 2)
        return self._cyl[k]

    def posed(self, key, S, xyz, yaw):
        """Every object's posed occupancy points (world) at the state, cached per state key."""
        if key not in self._posed:
            out = []
            for j in range(len(xyz)):
                o = S["objects"][j]; P = self._obj(o)
                out.append(None if P is None else (self._pose(P, float(o.get("yaw_baked", 0.0)) + float(yaw[j])) + np.asarray(xyz[j])).astype(np.float32))
            self._posed = {key: out}                                                   # one state at a time (memory)
        return self._posed[key]

    def score(self, key, S, xyz, yaw, a, plans, starts=None, dyaws=None):
        """plans (n, 33, 3) root points of carried object a from state (xyz (I, 3), yaw (I,)) -> dict of arrays (n,): geo_hit [0, 2],
        econ3d, cost, ok (False = a mesh occupancy is missing, no score). dyaws is unused: the cylinder is yaw-invariant."""
        n = len(plans)
        res = dict(geo_hit=np.full(n, np.nan, np.float32), econ3d=np.zeros(n, np.float32), cost=np.full(n, np.nan, np.float32),
                   ok=np.zeros(n, bool), max_grade=np.full(n, np.nan, np.float32), geo_drag=np.zeros(n, np.float32))
        cyl = self.cylinder(S["objects"][a]); occ = self.posed(key, S, xyz, yaw)
        if cyl is None or any(o is None for j, o in enumerate(occ) if j != a):
            return res
        R, Hc = cyl
        others = [j for j in range(len(xyz)) if j != a]
        bb = {j: (occ[j][:, :2].min(0), occ[j][:, :2].max(0), float(occ[j][:, 2].min()), float(occ[j][:, 2].max())) for j in others}
        s_r = np.r_[0.0, np.linspace(0.05, 0.95, N_READ)]
        for c in range(n):
            pth = np.asarray(plans[c], np.float64)
            sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pth, axis=0), axis=1))]
            off = np.asarray(xyz[a], np.float64) - pth[0]
            ex = None; grades = []; drags = []
            for r, s_ in enumerate(s_r):
                x = (np.array([np.interp(s_ * sl[-1], sl, pth[:, d]) for d in range(3)]) if sl[-1] > 0 else pth[0]) + (1 - s_) * off
                sep = np.full(len(xyz), np.inf)
                bt = 0.0                                                               # highest surface under the footprint (table z = 0)
                for j in others:
                    lo, hi, zlo, zhi = bb[j]                                           # cheap reject: the object's box is beyond the zone
                    gx = max(lo[0] - x[0], 0.0, x[0] - hi[0]); gy = max(lo[1] - x[1], 0.0, x[1] - hi[1])
                    if np.hypot(gx, gy) - R > HIT_ZONE + PITCH or zlo - (x[2] + Hc) > HIT_ZONE + PITCH or x[2] - zhi > HIT_ZONE + PITCH:
                        continue
                    q = occ[j]
                    rad = np.hypot(q[:, 0] - x[0], q[:, 1] - x[1]) - R
                    below, above = x[2] - q[:, 2], q[:, 2] - (x[2] + Hc)
                    out = np.maximum(rad, np.maximum(below, above))                    # > 0 outside the cylinder (a lower bound); < 0 inside
                    sep[j] = float(out.min()) - PITCH / 2
                    if self.w_drag > 0:                                                # voxels under the footprint, not above the base
                        under = (rad <= 0) & (q[:, 2] <= x[2] + PITCH)
                        if under.any():
                            bt = max(bt, float(q[under, 2].max()) + PITCH / 2)
                close = sep < HIT_ZONE
                if self.w_drag > 0 and r > 0:                                         # drag: clearance above the highest surface beneath
                    drags.append(float(zgrade(np.array([x[2] - bt]))[0]))
                if r == 0:
                    ex = close.copy(); continue
                ex &= close
                z = zgrade(np.where(ex, np.inf, sep))
                grades.append(float(z.max()) if len(z) else 0.0)
            res["geo_hit"][c] = float(np.mean(grades)); res["max_grade"][c] = float(np.max(grades))
            res["econ3d"][c] = econ3d(pth, xyz[a] if starts is None else starts[c], self.econ_dims)
            res["geo_drag"][c] = float(np.mean(drags)) if drags else 0.0
            res["cost"][c] = W_HIT * res["geo_hit"][c] + self.w_drag * res["geo_drag"][c] + W_ECON * res["econ3d"][c]; res["ok"][c] = True
        return res


# ------------------------------------------------------------------ mesh cost: each object's own shape
SDF_H = float(__import__("os").environ.get("WM_SDF_H", "0.01"))                        # one grid resolution for every object (default 1 cm)


class MeshCost(GeoCost):
    """The carried object's own shape swept along the path and checked against every other object's signed distance field, with
    the same construction for every object:
      per object   its filled occupancy (cache of screen.Screen; hollow objects stay hollow) on one 1 cm grid ->
                   a signed distance field (scipy distance transform: outside > 0, inside < 0) + a top-surface height map
      carried      its own 1 cm surface (shell) voxels, posed along the plan as the chain replay poses it: the path shifted to start at the
                   object ((1 - u) x offset), yaw interpolated from its start yaw to its end yaw (u x dyaw)
      per reading  hit: separation = min over its points of each other object's SDF - half a voxel, graded like the replay's hit
                   (0 at >= 1 cm, 1 touching, 2 at >= 1 cm into), max over objects, start-near exemption while still within 1 cm
                   drag: its bottom points' clearance above whatever lies beneath (the table top z = 0, or another object's top surface
                   under them), graded the same way, no exemption
      cost         10 x hit + w_drag x drag + 0.5 x econ3d (3D length / straight - 1: lifting over a wall is clearly cheaper than a hit
                   or a drag, yet not free)"""

    def __init__(self, w_drag=10.0):
        super().__init__(econ="3d", w_drag=w_drag)
        self._sdf = {}

    def field(self, o):
        """(sdf (X, Y, Z) float32, top (X, Y) float32 [-inf = empty column], lo (3,), shell points (n, 3), bottom points (m, 3)) of o."""
        from scipy.ndimage import distance_transform_edt
        k = (str(o["usd"]), float(o.get("scale") or 1.0))
        if k not in self._sdf:
            P = self._obj(o)
            if P is None:
                self._sdf[k] = None
            else:
                lo = P.min(0) - 3 * SDF_H
                ix = np.floor((P - lo) / SDF_H).astype(np.int64)
                G = np.zeros(ix.max(0) + 4, bool); G[ix[:, 0], ix[:, 1], ix[:, 2]] = True
                sdf = (distance_transform_edt(~G) - distance_transform_edt(G)).astype(np.float32) * SDF_H
                zc = lo[2] + (np.arange(G.shape[2]) + 0.5) * SDF_H
                top = np.where(G.any(2), (G * zc[None, None]).max(2) + SDF_H / 2, -np.inf).astype(np.float32)
                occ = np.argwhere(G); pts = lo + (occ + 0.5) * SDF_H
                inner = np.ones(len(occ), bool)
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nb = occ + d; okn = (nb >= 0).all(1) & (nb < np.array(G.shape)).all(1)
                    inner &= okn & G[np.clip(nb[:, 0], 0, G.shape[0] - 1), np.clip(nb[:, 1], 0, G.shape[1] - 1), np.clip(nb[:, 2], 0, G.shape[2] - 1)]
                shell = pts[~inner].astype(np.float32)
                bottom = shell[shell[:, 2] <= shell[:, 2].min() + SDF_H].astype(np.float32)
                self._sdf[k] = (sdf, top, lo.astype(np.float32), shell, bottom)
        return self._sdf[k]

    @staticmethod
    def _local(p, xyz, yaw):
        c, s = np.cos(-yaw), np.sin(-yaw); d = p - xyz
        return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], -1)

    def _sdf_at(self, F, q):
        sdf, _, lo, _, _ = F
        ix = np.floor((q - lo) / SDF_H).astype(np.int64)
        ok = (ix >= 0).all(1) & (ix < np.array(sdf.shape)).all(1)
        v = np.full(len(q), np.inf, np.float32)
        v[ok] = sdf[ix[ok, 0], ix[ok, 1], ix[ok, 2]]
        return v

    def score(self, key, S, xyz, yaw, a, plans, starts=None, dyaws=None):
        n = len(plans)
        res = dict(geo_hit=np.full(n, np.nan, np.float32), econ3d=np.zeros(n, np.float32), cost=np.full(n, np.nan, np.float32),
                   ok=np.zeros(n, bool), max_grade=np.full(n, np.nan, np.float32), geo_drag=np.zeros(n, np.float32))
        objs = S["objects"]; Fa = self.field(objs[a])
        others = [j for j in range(len(xyz)) if j != a]
        Fo = {j: self.field(objs[j]) for j in others}
        if Fa is None or any(v is None for v in Fo.values()):
            return res
        yo = {j: float(objs[j].get("yaw_baked", 0.0)) + float(yaw[j]) for j in others}
        R_ = {j: float(np.linalg.norm(Fo[j][3][:, :2], axis=1).max()) for j in others}          # reach of each other object (cheap reject)
        Ra = float(np.linalg.norm(Fa[3][:, :2], axis=1).max())
        y0 = float(objs[a].get("yaw_baked", 0.0)) + float(yaw[a])
        s_r = np.r_[0.0, np.linspace(0.05, 0.95, N_READ)]
        for c in range(n):
            pth = np.asarray(plans[c], np.float64); dy = 0.0 if dyaws is None else float(dyaws[c])
            sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pth, axis=0), axis=1))]
            off = np.asarray(xyz[a], np.float64) - pth[0]
            ex = None; grades, drags = [], []
            for r, s_ in enumerate(s_r):
                x = (np.array([np.interp(s_ * sl[-1], sl, pth[:, d]) for d in range(3)]) if sl[-1] > 0 else pth[0]) + (1 - s_) * off
                ya = y0 + s_ * dy
                Q = self._pose(Fa[3], ya) + x; Qb = self._pose(Fa[4], ya) + x
                sep = np.full(len(xyz), np.inf); bt_clear = float(Qb[:, 2].min())               # table top z = 0
                for j in others:
                    if np.hypot(*(np.asarray(xyz[j][:2]) - x[:2])) > R_[j] + Ra + HIT_ZONE + SDF_H:
                        continue
                    ql = self._local(Q, np.asarray(xyz[j]), yo[j])
                    sep[j] = float(self._sdf_at(Fo[j], ql).min()) - SDF_H / 2
                    if self.w_drag > 0:                                                  # the surface under its bottom points
                        bl = self._local(Qb, np.asarray(xyz[j]), yo[j]); sdf, top, lo, _, _ = Fo[j]
                        ix = np.floor((bl[:, :2] - lo[:2]) / SDF_H).astype(np.int64)
                        ok = (ix >= 0).all(1) & (ix < np.array(top.shape)).all(1)
                        if ok.any():
                            tz = top[ix[ok, 0], ix[ok, 1]]; hz = bl[ok, 2]
                            m = np.isfinite(tz) & (hz >= tz - SDF_H)
                            if m.any():
                                bt_clear = min(bt_clear, float((hz[m] - tz[m]).min()))
                close = sep < HIT_ZONE
                if r == 0:
                    ex = close.copy(); continue
                ex &= close
                z = zgrade(np.where(ex, np.inf, sep))
                grades.append(float(z.max()) if len(z) else 0.0)
                if self.w_drag > 0:
                    drags.append(float(zgrade(np.array([bt_clear]))[0]))
            res["geo_hit"][c] = float(np.mean(grades)); res["max_grade"][c] = float(np.max(grades))
            res["econ3d"][c] = econ3d(pth, xyz[a] if starts is None else starts[c], 3)
            res["geo_drag"][c] = float(np.mean(drags)) if drags else 0.0
            res["cost"][c] = W_HIT * res["geo_hit"][c] + self.w_drag * res["geo_drag"][c] + W_ECON * res["econ3d"][c]; res["ok"][c] = True
        return res
