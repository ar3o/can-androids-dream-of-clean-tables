"""Filled voxel occupancy of the twin meshes and the penetration test of a carried object swept along paths.

Containers stay hollow (only enclosed space is filled): ending inside a bowl is free, passing through its rim is not.

    python -m cleantables.worldmodel.occupancy [episode ...]    # -> data/twins/<episode>/occ.npz (needs trimesh)
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch

from cleantables.paths import twin_dir

log = logging.getLogger(__name__)
PITCH = 0.005  # voxel size, metres
TABLE_MARGIN = 0.15  # grid margin around the table, metres
GRID_HEIGHT = 0.5  # metres above the table covered by the grid
OCC_FILE = "occ.npz"


def occupancy_points(mesh_path) -> np.ndarray:
    """Voxel centres (n,3) of the filled mesh in its own frame (bottom at z = 0)."""
    import trimesh

    v = trimesh.load(mesh_path, force="mesh").voxelized(PITCH)
    try:
        v = v.fill()
    except Exception:  # open meshes cannot be filled; their surface voxels are used as they are
        pass
    return np.asarray(v.points, np.float32)


def build_occupancy(episode: str) -> dict:
    scene = json.load(open(twin_dir(episode) / "scene.json"))
    out = {}
    for o in scene["objects"]:
        p = twin_dir(episode) / "usd" / f"{o['id']}.obj"
        if p.exists():
            out[o["id"]] = occupancy_points(p)
    np.savez_compressed(twin_dir(episode) / OCC_FILE, **out)
    return {k: len(v) for k, v in out.items()}


class PenetrationScorer:
    """Occupancy grid of the static objects of one twin and the sweep test of one moving object (torch, GPU)."""

    def __init__(self, episode: str, device: str = "cuda"):
        self.dev = device
        scene = json.load(open(twin_dir(episode) / "scene.json"))
        z = np.load(twin_dir(episode) / OCC_FILE)
        self.occ = {k: torch.as_tensor(z[k], device=device) for k in z.files}
        T = scene["table"]
        self.lo = torch.tensor([T["x"][0] - TABLE_MARGIN, T["y"][0] - TABLE_MARGIN, -0.02], device=device)
        self.shape = (
            int((T["x"][1] - T["x"][0] + 2 * TABLE_MARGIN) / PITCH) + 1,
            int((T["y"][1] - T["y"][0] + 2 * TABLE_MARGIN) / PITCH) + 1,
            int(GRID_HEIGHT / PITCH),
        )

    def grid(self, ids: list, pos: np.ndarray, skip: int) -> torch.Tensor:
        """Occupancy of every object ids[j] at bottom-centre pos[j] (yaw baked into the meshes), except index skip."""
        G = torch.zeros(self.shape, dtype=torch.bool, device=self.dev)
        for j, oid in enumerate(ids):
            if j == skip or oid not in self.occ:
                continue
            ix = ((self.occ[oid] + torch.as_tensor(pos[j], device=self.dev) - self.lo) / PITCH).round().long()
            inside = (
                (ix >= 0).all(1) & (ix[:, 0] < self.shape[0]) & (ix[:, 1] < self.shape[1]) & (ix[:, 2] < self.shape[2])
            )
            ix = ix[inside]
            G[ix[:, 0], ix[:, 1], ix[:, 2]] = True
        return G

    def penetration(self, G: torch.Tensor, oid: str, paths_bottom: torch.Tensor, erode: int = 1) -> torch.Tensor:
        """Per path (K,), the mean over steps of the moving object's voxels inside other objects or below the table.
        paths_bottom (K,T,3) are bottom-centre positions. The object's points are shrunk towards their centroid by
        `erode` voxels first, so resting contact does not count."""
        P = self.occ[oid]
        c = P.mean(0)
        d = (P - c).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        Pe = P - (P - c) / d * (erode * PITCH)
        Q = Pe[None, None] + paths_bottom[:, :, None, :]
        ix = ((Q - self.lo) / PITCH).round().long()
        in_grid = (
            (ix >= 0).all(-1)
            & (ix[..., 0] < self.shape[0])
            & (ix[..., 1] < self.shape[1])
            & (ix[..., 2] < self.shape[2])
        )
        ixc = ix.clamp(min=0)
        for a in range(3):
            ixc[..., a] = ixc[..., a].clamp(max=self.shape[a] - 1)
        hit = G[ixc[..., 0], ixc[..., 1], ixc[..., 2]] & in_grid
        below = Q[..., 2] < -PITCH
        return (hit | below).float().sum(-1).mean(-1)


def main() -> None:
    from .physics_feedback import twin_episodes

    ap = argparse.ArgumentParser(description="Voxelise the twin meshes of the tools and fruit episodes.")
    ap.add_argument("episodes", nargs="*", help="episode ids (default: every tools and fruit episode)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for ep in a.episodes or twin_episodes("all"):
        t0 = time.time()
        counts = build_occupancy(ep)
        log.info("episode %s: %s voxels, %.1f s", ep, counts, time.time() - t0)


if __name__ == "__main__":
    main()
