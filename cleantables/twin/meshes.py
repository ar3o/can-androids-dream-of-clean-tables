"""Textured meshes built from the video: cuboids with face textures, open trays, slabs extruded from a top-down
silhouette.

Meshes are trimesh objects with per-vertex UVs; footprint centred at the origin, bottom at z = 0, L along x.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt

from cleantables.twin.fitting import box_corners
from cleantables.twin.projection import project

TEX_RES = 0.002  # m per texel of slab textures and orthophotos
FACE_ORDER = ["top", "bottom", "front", "back", "right", "left"]
# face: (corner indices into box_corners order, outward normal); corners ordered so each texture is upright
FACES = {
    "top": ([7, 6, 5, 4], (0, 0, 1)),  # image top = +y side
    "bottom": ([0, 1, 2, 3], (0, 0, -1)),
    "front": ([4, 5, 1, 0], (0, -1, 0)),  # -y face, image top = top edge
    "back": ([6, 7, 3, 2], (0, 1, 0)),
    "right": ([5, 6, 2, 1], (1, 0, 0)),
    "left": ([7, 4, 0, 3], (-1, 0, 0)),
}
OPPOSITE = {"top": "bottom", "bottom": "top", "front": "back", "back": "front", "right": "left", "left": "right"}


def face_textures(img, own_mask, K, T, pose, min_vis=0.75, tile=512) -> dict:
    """face -> (tile x tile BGR texture, visibility score) for faces seen front-on, >= min_vis covered by own_mask."""
    P = box_corners(pose["x"], pose["y"], pose["yaw"], pose["L"], pose["W"], pose["H"], pose.get("z0", 0.0))
    cam = T[:3, 3]
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    out = {}
    for name, (idx, normal) in FACES.items():
        Q = P[idx]
        nw = Rz @ np.array(normal, float)
        view = (cam - Q.mean(0)) @ nw / np.linalg.norm(cam - Q.mean(0))
        if view < 0.12:  # back-facing or grazing
            continue
        uv, _ = project(Q, K, T)
        fm = np.zeros(own_mask.shape, np.uint8)
        cv2.fillConvexPoly(fm, uv.astype(np.int32), 1)
        fm = fm.astype(bool)
        vis = float((fm & own_mask).sum() / max(1, fm.sum()))
        if vis < min_vis:
            continue
        dst = np.float32([[0, 0], [tile - 1, 0], [tile - 1, tile - 1], [0, tile - 1]])
        Hm = cv2.getPerspectiveTransform(dst, uv.astype(np.float32))
        tex = cv2.warpPerspective(
            img, Hm, (tile, tile), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )
        out[name] = (tex, vis * view)
    return out


def cuboid_mesh(L, W, H, faces: dict, tile=512):
    """textured cuboid with a 3x2 face atlas -> (mesh, atlas, texture source per face).

    Missing faces take the opposite face's texture, else the mean colour of the seen faces.
    """
    mean = np.mean([f.reshape(-1, 3).mean(0) for f in faces.values()], 0) if faces else np.array([128, 128, 128])
    atlas = np.zeros((2 * tile, 3 * tile, 3), np.uint8)
    src = {}
    for i, name in enumerate(FACE_ORDER):
        if name in faces:
            t, src[name] = faces[name], "observed"
        elif OPPOSITE[name] in faces:
            t, src[name] = faces[OPPOSITE[name]], f"copy of {OPPOSITE[name]}"
        else:
            t, src[name] = np.full((tile, tile, 3), mean, np.uint8), "mean colour"
        r, c = divmod(i, 3)
        atlas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = cv2.resize(t, (tile, tile))
    P = box_corners(0, 0, 0, L, W, H)
    V, F, UV = [], [], []
    for i, name in enumerate(FACE_ORDER):
        idx, _ = FACES[name]
        r, c = divmod(i, 3)
        e = 2.0 / tile  # half-texel inset
        u0, u1 = c / 3 + e / 3, (c + 1) / 3 - e / 3
        v1, v0 = 1 - r / 2 - e / 2, 1 - (r + 1) / 2 + e / 2  # uv origin bottom-left; atlas row r from the top
        b = len(V)
        V += [P[j] for j in idx]
        UV += [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]  # image TL, TR, BR, BL
        F += [[b, b + 2, b + 1], [b, b + 3, b + 2]]
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    inward = ((m.triangles_center - np.array([0, 0, H / 2])) * m.face_normals).sum(1) < 0
    Fa = np.array(F)
    Fa[inward] = Fa[inward][:, ::-1]
    m = trimesh.Trimesh(np.array(V), Fa, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.array(UV))
    return m, atlas, src


def tray_mesh(fit: dict, atlas: np.ndarray, interior, t: float = 0.006):
    """open box: outer walls (cuboid atlas tiles), inner walls and floor in the interior colour, rim."""
    L, W, H = fit["L"], fit["W"], fit["H"]
    tile = atlas.shape[0] // 2
    atlas2 = np.concatenate([atlas, np.zeros((tile, atlas.shape[1], 3), np.uint8)], 0)
    atlas2[2 * tile :] = np.array(interior, np.uint8)
    outer, _, _ = cuboid_mesh(L, W, H, {})
    top = FACE_ORDER.index("top")
    oV, oF, oUV = outer.vertices, outer.faces, outer.visual.uv.copy()
    oUV[:, 1] = oUV[:, 1] * 2 / 3 + 1 / 3  # the 2-row atlas becomes the top 2/3 of the 3-row one
    V, UV = list(oV), list(oUV)
    F = [oF[i] for i in range(len(oF)) if i // 2 != top]
    inner, _, _ = cuboid_mesh(L - 2 * t, W - 2 * t, H - t, {})
    interior_uv = [0.5, 1 / 6]
    b = len(V)
    V += list(inner.vertices + np.array([0, 0, t]))
    UV += [interior_uv] * len(inner.vertices)
    F += [inner.faces[i][::-1] + b for i in range(len(inner.faces)) if i // 2 != top]
    oc = box_corners(0, 0, 0, L, W, H)[4:]
    ic = box_corners(0, 0, 0, L - 2 * t, W - 2 * t, H)[4:]
    b = len(V)
    V += list(oc) + list(ic)
    UV += [interior_uv] * 8
    for i in range(4):
        j = (i + 1) % 4
        F += [[b + i, b + j, b + 4 + j], [b + i, b + 4 + j, b + 4 + i]]
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.array(UV))
    return m, atlas2


def slab_mesh(sil: np.ndarray, thick: float, res: float = TEX_RES, pillow: bool = False) -> trimesh.Trimesh:
    """watertight slab from a top-down silhouette (rows = +y at the top): top and bottom grids of the mask cells,
    walls on the boundary. pillow: top height = thick x sqrt(distance / max distance), for rounded objects. UV = (col,
    row) in the silhouette."""
    Hs, Ws = sil.shape
    if pillow:
        dt = distance_transform_edt(np.pad(sil, 1))[1:-1, 1:-1]
        ht = thick * np.sqrt(np.clip(dt / max(dt.max(), 1), 0.15, 1.0))
    vid = {}
    V, UV, F = [], [], []

    def vert(r, c, top):
        key = (r, c, top)
        if key not in vid:
            if top and pillow:
                rr, cc = min(max(r, 0), Hs - 1), min(max(c, 0), Ws - 1)
                nb = [ht[a, b] for a in (rr - 1, rr) for b in (cc - 1, cc) if 0 <= a < Hs and 0 <= b < Ws and sil[a, b]]
                z = float(np.mean(nb)) if nb else thick * 0.4
            else:
                z = thick if top else 0.0
            vid[key] = len(V)
            V.append((c * res, -r * res, z))
            UV.append((c / Ws, 1 - r / Hs))
        return vid[key]

    ys, xs = np.nonzero(sil)
    for r, c in zip(ys, xs, strict=False):
        a, b, cc, d = vert(r, c, 1), vert(r, c + 1, 1), vert(r + 1, c + 1, 1), vert(r + 1, c, 1)
        F += [[a, d, cc], [a, cc, b]]
        a, b, cc, d = vert(r, c, 0), vert(r, c + 1, 0), vert(r + 1, c + 1, 0), vert(r + 1, c, 0)
        F += [[a, cc, d], [a, b, cc]]
        for (dr, dc), (p, q) in (
            ((-1, 0), ((r, c), (r, c + 1))),
            ((1, 0), ((r + 1, c + 1), (r + 1, c))),
            ((0, -1), ((r + 1, c), (r, c))),
            ((0, 1), ((r, c + 1), (r + 1, c + 1))),
        ):
            rr, c2 = r + dr, c + dc
            if 0 <= rr < Hs and 0 <= c2 < Ws and sil[rr, c2]:
                continue
            p0, p1 = vert(*p, 1), vert(*q, 1)
            q0, q1 = vert(*p, 0), vert(*q, 0)
            F += [[p0, q0, q1], [p0, q1, p1]]
    V = np.array(V)
    m = trimesh.Trimesh(V, np.array(F), process=False)
    if m.volume < 0:
        m.invert()
    lo, hi = V.min(0), V.max(0)
    m.vertices = V - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
    m.visual = trimesh.visual.TextureVisuals(uv=np.array(UV))
    return m


def convex_parts(mesh: trimesh.Trimesh, threshold: float = 0.05, max_parts: int = 12) -> list[trimesh.Trimesh]:
    """CoACD approximate convex decomposition (collision shapes of non-convex objects)."""
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=threshold,
        max_convex_hull=max_parts,
        preprocess_mode="auto",
    )
    return [trimesh.Trimesh(np.asarray(v), np.asarray(f)) for v, f in parts]


def bowl_profile(V: np.ndarray) -> dict:
    """centre, bottom / top radius and height of a bowl mesh (for its floor-disc and wall-slab collider)."""
    Vb = V - np.r_[V[:, :2].mean(0), 0]
    zz = Vb[:, 2]
    rr = np.hypot(Vb[:, 0], Vb[:, 1])
    Hb = float(zz.max())
    r_bot = float(np.percentile(rr[zz < 0.15 * Hb], 90)) if np.any(zz < 0.15 * Hb) else 0.4 * rr.max()
    r_top = float(np.percentile(rr[zz > 0.85 * Hb], 95)) if np.any(zz > 0.85 * Hb) else rr.max()
    cx, cy = [float(v) for v in V[:, :2].mean(0)]
    return dict(cx=cx, cy=cy, r_bot=min(r_bot, r_top), r_top=r_top, H=Hb)


def tray_colliders(L: float, W: float, H: float, t: float = 0.006) -> list[tuple]:
    """open box (floor + 4 walls) as 5 thin boxes: [(size, centre)]."""
    return [
        ((L, W, t), (0, 0, t / 2)),
        ((t, W, H), (L / 2 - t / 2, 0, H / 2)),
        ((t, W, H), (-L / 2 + t / 2, 0, H / 2)),
        ((L, t, H), (0, W / 2 - t / 2, H / 2)),
        ((L, t, H), (0, -W / 2 + t / 2, H / 2)),
    ]
