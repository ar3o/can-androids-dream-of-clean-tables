"""CPU clearance pre-screen for the physics stage (cleantables.worldmodel.physics.stage): candidate carries that clearly run into
another object are not sent to PhysX.

Geometry: every twin mesh's filled 5 mm voxel occupancy (cleantables.worldmodel.physics.feedback_base.occ_points: trimesh voxelize
+ fill, so hollow containers stay hollow; computed once per mesh file in the `sim` env, CPU only, cached as .npy), posed as the PhysX
chain replay poses the bodies (root = bottom centre; quaternion = yaw offset x the baked yaw; the domain-randomised isotropic scale).
The carried object is tested with the shell of its occupancy eroded by one voxel (touching = 0, as carry_check.hits_per_point),
swept along the plan at 19 readings over 5-95% of its path length (the replay's u = share of the path length, yaw ramped by u x dyaw).
  reading hit   a shell voxel lies inside another object's occupancy (the table is not a counted body)
  exemption     objects it already overlaps at the start reading stay exempt while that start overlap continues (as in the replay)
  hit_lb        share of the 19 readings with a hit: a lower bound of the replay's graded hit (every contact reading counts >= 1)
  screened      hit_lb >= FRAC (default 0.5: inside another object for at least half of the carry), so replaying it adds nothing
No names and no roles. Conservative by construction (voxels, one-voxel erosion, a whole-carry share), so it only skips carries
that PhysX would score as hits anyway.

  occupancy cache (sim env, CPU):  $SIM_PY -m cleantables.worldmodel.physics.screen --occ LIST.json    (ensure_occ runs it)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PITCH = 0.005
N_READ = 19
FRAC = 0.5
CACHE = Path(os.environ.get("WM_OCC_CACHE", str(Path(os.environ.get("WM_PF_DIR", str(ROOT / "outputs"))) / "occ_cache")))
SIM_PY = os.environ.get("SIM_PY", str(ROOT / "envs/sim/bin/python"))


def _key(usd):
    return CACHE / (re.sub(r"[^A-Za-z0-9_.-]", "_", str(usd).replace(".usda", "")) + ".npy")


def ensure_occ(usds, timeout=3600):
    """Filled occupancy of every mesh in usds (scene.json 'usd' paths, relative to ROOT), computed where missing (sim env,
    trimesh, CPU). -> True if all exist afterwards."""
    miss = sorted({str(u) for u in usds if not _key(u).exists()})
    if miss:
        CACHE.mkdir(parents=True, exist_ok=True)
        lst = CACHE / f"todo_{os.getpid()}.json"
        json.dump(miss, open(lst, "w"))
        env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="")
        subprocess.run([SIM_PY, "-m", "cleantables.worldmodel.physics.screen", "--occ", str(lst)], cwd=str(ROOT), env=env, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        lst.unlink(missing_ok=True)
    return all(_key(u).exists() for u in usds)


def _lattice(P):
    ix = np.round(P / PITCH).astype(np.int64)
    return ix - ix.min(0), ix.min(0)


def _shell(ix):
    """Voxels of the set ix (n, 3) with at least one empty 6-neighbour."""
    G = np.zeros(ix.max(0) + 3, bool); G[ix[:, 0] + 1, ix[:, 1] + 1, ix[:, 2] + 1] = True
    inner = G[1:-1, 1:-1, 1:-1] & G[:-2, 1:-1, 1:-1] & G[2:, 1:-1, 1:-1] & G[1:-1, :-2, 1:-1] & G[1:-1, 2:, 1:-1] & G[1:-1, 1:-1, :-2] & G[1:-1, 1:-1, 2:]
    keep = ~inner[ix[:, 0], ix[:, 1], ix[:, 2]]
    return keep, inner


class Screen:
    """Per twin: posed occupancies; per state: one label grid of every object; per candidate: the swept test (module docstring)."""

    def __init__(self, frac=FRAC):
        self.frac, self._occ, self._shell, self._grid = frac, {}, {}, {}
        self.ok = True

    def _obj(self, o):
        u = str(o["usd"]); s = float(o.get("scale") or 1.0)
        if u not in self._occ:
            f = _key(u)
            self._occ[u] = np.load(f).astype(np.float32) if f.exists() else None
        P = self._occ[u]
        return None if P is None else P * s

    def _carried(self, o):
        """Carried object's test points: the shell of its occupancy eroded by one voxel (thin objects: its plain shell)."""
        k = (str(o["usd"]), float(o.get("scale") or 1.0))
        if k not in self._shell:
            P = self._obj(o)
            if P is None:
                self._shell[k] = None
            else:
                ix, lo = _lattice(P / (k[1] if k[1] > 0 else 1.0))
                sh, inner = _shell(ix)
                core = ~sh
                if core.sum() > 0:
                    sh2, _ = _shell(ix[core]); pts = P[core][sh2]
                else:
                    pts = P[sh]
                self._shell[k] = pts.astype(np.float32)
        return self._shell[k]

    @staticmethod
    def _pose(P, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.stack([c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1], P[:, 2]], -1)

    def grid(self, key, S, xyz, yaw):
        """Label grid (int16, object index + 1) of every object at (xyz, yaw offset), cached per state key."""
        if key in self._grid:
            return self._grid[key]
        tb = S["table"]
        lo = np.array([tb["x"][0] - 0.15, tb["y"][0] - 0.15, -0.02], np.float32)
        shape = (int((tb["x"][1] - tb["x"][0] + 0.3) / PITCH) + 1, int((tb["y"][1] - tb["y"][0] + 0.3) / PITCH) + 1, int(0.62 / PITCH))
        L = np.zeros(shape, np.int16)
        for j in range(len(xyz)):
            o = S["objects"][j]; P = self._obj(o)
            if P is None:
                self.ok = False; continue
            q = self._pose(P, float(o.get("yaw_baked", 0.0)) + float(yaw[j])) + xyz[j]
            ix = np.round((q - lo) / PITCH).astype(np.int64)
            m = (ix >= 0).all(1) & (ix < np.array(shape)).all(1)
            L[ix[m, 0], ix[m, 1], ix[m, 2]] = j + 1
        self._grid = {key: (L, lo)}                                                # one state at a time (memory)
        return L, lo

    def check(self, key, S, xyz, yaw, a, plans, dyaws):
        """plans (n, 33, 3) root points, dyaws (n,) for carried object a from state (xyz (I, 3), yaw offsets (I,)) -> hit_lb (n,),
        screened (n,) bool. Missing meshes: nothing is screened."""
        n = len(plans)
        pts = self._carried(S["objects"][a])
        if pts is None:
            return np.zeros(n, np.float32), np.zeros(n, bool)
        L, lo = self.grid(key, S, xyz, yaw)
        if not self.ok:
            return np.zeros(n, np.float32), np.zeros(n, bool)
        shape = np.array(L.shape)
        y0 = float(S["objects"][a].get("yaw_baked", 0.0)) + float(yaw[a])
        s_r = np.r_[0.0, np.linspace(0.05, 0.95, N_READ)]
        hit_lb = np.zeros(n, np.float32)
        for c in range(n):
            pth = np.asarray(plans[c], np.float64)
            sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pth, axis=0), axis=1))]
            off = np.asarray(xyz[a], np.float64) - pth[0]                          # as the replay: the path is shifted to start where the object
            ex = np.ones(len(xyz) + 1, bool)                                       # is, (1 - u) x offset; exempt while in start overlap
            nh = 0
            for r, s_ in enumerate(s_r):
                x = (np.array([np.interp(s_ * sl[-1], sl, pth[:, d]) for d in range(3)]) if sl[-1] > 0 else pth[0]) + (1 - s_) * off
                q = self._pose(pts, y0 + s_ * float(dyaws[c])) + x
                ix = np.round((q - lo) / PITCH).astype(np.int64)
                m = (ix >= 0).all(1) & (ix < shape).all(1)
                lab = np.unique(L[ix[m, 0], ix[m, 1], ix[m, 2]])
                lab = lab[(lab > 0) & (lab != a + 1)]
                hitm = np.zeros(len(xyz) + 1, bool); hitm[lab] = True
                if r == 0:
                    ex = hitm.copy()
                    continue
                ex &= hitm
                nh += int((hitm & ~ex).any())
            hit_lb[c] = nh / N_READ
        return hit_lb, hit_lb >= self.frac


def _occ_main(lst):
    """Sim env: filled occupancy (voxel centres, object frame) of every listed mesh -> CACHE/<key>.npy."""
    from cleantables.worldmodel.physics.feedback_base import occ_points
    for u in json.load(open(lst)):
        f = ROOT / str(u).replace(".usda", ".obj")
        try:
            np.save(_key(u), occ_points(f, PITCH).astype(np.float32))
        except Exception as e:                                                     # noqa: BLE001  (a mesh trimesh cannot read: not screened)
            print(f"occ FAILED {u}: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--occ", required=True)
    _occ_main(ap.parse_args().occ)
