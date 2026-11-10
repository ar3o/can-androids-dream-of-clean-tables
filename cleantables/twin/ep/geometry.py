"""Geometry for the per-episode twins: cuboid / flat-object fitting to SAM 3 masks through the real phone camera, face textures by
homography, textured meshes (trimesh with UVs), USD export (physics-ready rigid bodies, explicit colliders).

Conventions: table frame (z up, table top z = 0, metres); K at 960x540; T = camera -> table (det3d.npz).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh

from cleantables.twin.ep.common import project

TEX_RES = 0.002          # 2 mm / texel for face textures and flat-object orthophotos


# ------------------------------------------------------------------ cuboids
def box_corners(x, y, yaw, L, W, H, z0=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dz in (0.0, H):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            px, py = sx * L / 2, sy * W / 2
            out.append([x + c * px - s * py, y + s * px + c * py, z0 + dz])
    return np.array(out)


def poly_mask(P, K, T, shape=(540, 960)):
    uv, z = project(P, K, T)
    m = np.zeros(shape, np.uint8)
    if (z <= 0.01).any():
        return m.astype(bool)
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def fit_cuboid(mask, K, T, p0, fix_dims=None, z0=0.0, free_dims=True, care=None, free_scale=False, scale_bounds=(0.7, 1.4)):
    """Refine (x, y, yaw, L, W, H) so the projected cuboid silhouette matches `mask` (960x540 bool). care: pixels that count
    (e.g. ~occluders). fix_dims: (L, W, H) kept fixed (only pose fitted). Nelder-Mead from p0 and p0 with yaw + 90 deg (L<->W)."""
    from scipy.optimize import minimize
    care = np.ones_like(mask) if care is None else care
    mk = mask & care

    def loss(q):
        x, y, yw = q[:3]
        if free_scale:
            if not scale_bounds[0] <= q[3] <= scale_bounds[1]:
                return 2.0
            L, Wd, Hh = np.asarray(fix_dims) * q[3]
        else:
            L, Wd, Hh = (fix_dims if fix_dims is not None else np.abs(q[3:6]))
        if min(L, Wd, Hh) < 0.01:
            return 2.0
        pen = 0.0
        if fix_dims is None:                          # dims stay within +-25 % of the depth-measured box (silhouette alone is ambiguous)
            r = np.abs(np.log(np.array([L, Wd, Hh]) / np.maximum(np.array([p0[3], p0[4], p0[5]]), 1e-3)))
            pen = 2.0 * float(np.clip(r - 0.22, 0, None).sum())
        pm = poly_mask(box_corners(x, y, yw, L, Wd, Hh, z0), K, T) & care
        return 1.0 - iou(pm, mk) + pen
    best = None
    x, y, yw, L, Wd, Hh = p0
    for q0 in ([x, y, yw, L, Wd, Hh], [x, y, yw + math.pi / 2, Wd, L, Hh]):
        q0 = np.array((q0[:3] + [1.0]) if free_scale else (q0[:3] if fix_dims is not None else q0), float)
        step = np.array([0.02, 0.02, 0.15, 0.05] if free_scale else [0.02, 0.02, 0.15, 0.02, 0.02, 0.02])[:len(q0)]
        sim = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * step[i] for i in range(len(q0))])
        r = minimize(loss, q0, method="Nelder-Mead", options=dict(initial_simplex=sim, maxiter=400, xatol=1e-4, fatol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    q = best.x
    sc = float(q[3]) if free_scale else 1.0
    dims = np.asarray(fix_dims) * sc if free_scale else (fix_dims if fix_dims is not None else np.abs(q[3:6]))
    return dict(x=float(q[0]), y=float(q[1]), yaw=float(q[2]), L=float(dims[0]), W=float(dims[1]), H=float(dims[2]), iou=1 - float(best.fun),
                scale=sc)


FACES = {   # name: (corner indices into box_corners order, outward normal in the box frame)  corners ordered so the texture is upright
    "top": ([7, 6, 5, 4], (0, 0, 1)),        # (-,+,H) (+,+,H) (+,-,H) (-,-,H): image top = +y side
    "bottom": ([0, 1, 2, 3], (0, 0, -1)),
    "front": ([4, 5, 1, 0], (0, -1, 0)),     # -y face, image top = top edge
    "back": ([6, 7, 3, 2], (0, 1, 0)),
    "right": ([5, 6, 2, 1], (1, 0, 0)),
    "left": ([7, 4, 0, 3], (-1, 0, 0)),
}
OPP = {"top": "bottom", "bottom": "top", "front": "back", "back": "front", "right": "left", "left": "right"}


def face_textures(img, own_mask, K, T, pose, min_vis=0.75, tile=512):
    """-> dict face -> (tile x tile x 3 BGR texture, visibility score). Faces seen front-on in `img` and >= min_vis covered by own_mask."""
    P = box_corners(pose["x"], pose["y"], pose["yaw"], pose["L"], pose["W"], pose["H"], pose.get("z0", 0.0))
    cam = T[:3, 3]
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    out = {}
    for nm, (idx, n) in FACES.items():
        Q = P[idx]
        nw = Rz @ np.array(n, float)
        view = (cam - Q.mean(0)) @ nw / np.linalg.norm(cam - Q.mean(0))
        if view < 0.12:                                   # back-facing or grazing
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
        tex = cv2.warpPerspective(img, Hm, (tile, tile), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        out[nm] = (tex, vis * view)
    return out


def cuboid_mesh(L, W, H, faces, tile=512):
    """Textured cuboid (centre of the footprint at the origin, bottom z=0, L along x, W along y) with a 3x2 face atlas.
    faces: dict face -> BGR tile (missing faces: the opposite face's texture, else the mean colour of the seen faces)."""
    names = ["top", "bottom", "front", "back", "right", "left"]
    mean = np.mean([f.reshape(-1, 3).mean(0) for f in faces.values()], 0) if faces else np.array([128, 128, 128])
    atlas = np.zeros((2 * tile, 3 * tile, 3), np.uint8)
    src = {}
    for i, nm in enumerate(names):
        if nm in faces:
            t, src[nm] = faces[nm], "observed"
        elif OPP[nm] in faces:
            t, src[nm] = faces[OPP[nm]], f"copy of {OPP[nm]}"
        else:
            t, src[nm] = np.full((tile, tile, 3), mean, np.uint8), "mean colour"
        r, c = divmod(i, 3)
        atlas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = cv2.resize(t, (tile, tile))
    P = box_corners(0, 0, 0, L, W, H)
    V, F, UV = [], [], []
    for i, nm in enumerate(names):
        idx, _ = FACES[nm]
        r, c = divmod(i, 3)
        e = 2.0 / tile            # half-texel inset
        u0, u1 = c / 3 + e / 3, (c + 1) / 3 - e / 3
        v1, v0 = 1 - r / 2 - e / 2, 1 - (r + 1) / 2 + e / 2      # uv origin bottom-left; tile row r from the top
        uvs = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]          # corner order = image TL, TR, BR, BL
        b = len(V)
        V += [P[j] for j in idx]; UV += uvs
        F += [[b, b + 2, b + 1], [b, b + 3, b + 2]]
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    m.fix_normals() if False else None
    # outward winding: flip faces whose normal points inward
    cen = np.array([0, 0, H / 2])
    fn = m.face_normals
    fc = m.triangles_center
    flip = ((fc - cen) * fn).sum(1) < 0
    Fa = np.array(F)
    Fa[flip] = Fa[flip][:, ::-1]
    m = trimesh.Trimesh(np.array(V), Fa, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.array(UV))
    return m, atlas, src


CUBE_ROTS = None


def cube_rotations():
    """The 24 proper rotations of a cube (3x3 signed permutation matrices, det +1)."""
    global CUBE_ROTS
    if CUBE_ROTS is None:
        R = []
        for p in itertools.permutations(range(3)):
            for sg in itertools.product((1, -1), repeat=3):
                M = np.zeros((3, 3))
                for i in range(3):
                    M[i, p[i]] = sg[i]
                if np.linalg.det(M) > 0:
                    R.append(M)
        CUBE_ROTS = R
    return CUBE_ROTS


# ------------------------------------------------------------------ flat objects (tools, ginger): textured slab from a top-down silhouette
def ortho(img, K, T, x0, x1, y0, y1, z=0.0, res=TEX_RES):
    """Orthophoto of the plane z over [x0,x1]x[y0,y1] (rows: +y at the top) from the 960x540 frame."""
    gx, gy = np.meshgrid(np.arange(x0, x1, res), np.arange(y1, y0, -res))
    G = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z)]
    uv, _ = project(G, K, T)
    mx, my = uv[:, 0].reshape(gx.shape).astype(np.float32), uv[:, 1].reshape(gx.shape).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), (mx, my)


def slab_mesh(sil, thick, res=TEX_RES, pillow=False):
    """Watertight mesh from a top-down silhouette (bool HxW, rows = +y at the top): top + bottom grid of the mask cells and side walls on
    the boundary. pillow: top height = thick x sqrt(distance / max distance) (rounded objects such as ginger). Origin = silhouette bbox centre,
    bottom z = 0. UV of every vertex = its (col, row) in the silhouette image (the top texture); sides / bottom get the same UVs."""
    Hs, Ws = sil.shape
    from scipy.ndimage import distance_transform_edt
    if pillow:
        dt = distance_transform_edt(np.pad(sil, 1))[1:-1, 1:-1]
        ht = thick * np.sqrt(np.clip(dt / max(dt.max(), 1), 0.15, 1.0))
    vid = {}
    V, UV, F = [], [], []

    def vert(r, c, top):
        key = (r, c, top)
        if key not in vid:
            if top:
                if pillow:
                    rr, cc = min(max(r, 0), Hs - 1), min(max(c, 0), Ws - 1)
                    nb = [ht[a, b] for a in (rr - 1, rr) for b in (cc - 1, cc) if 0 <= a < Hs and 0 <= b < Ws and sil[a, b]]
                    z = float(np.mean(nb)) if nb else thick * 0.4
                else:
                    z = thick
            else:
                z = 0.0
            vid[key] = len(V)
            V.append((c * res, -r * res, z))
            UV.append((c / Ws, 1 - r / Hs))
        return vid[key]
    ys, xs = np.nonzero(sil)
    for r, c in zip(ys, xs):
        a, b, cc, d = vert(r, c, 1), vert(r, c + 1, 1), vert(r + 1, c + 1, 1), vert(r + 1, c, 1)
        F += [[a, d, cc], [a, cc, b]]                           # top (+z)
        a, b, cc, d = vert(r, c, 0), vert(r, c + 1, 0), vert(r + 1, c + 1, 0), vert(r + 1, c, 0)
        F += [[a, cc, d], [a, b, cc]]                           # bottom (-z)
        for (dr, dc), (p, q) in (((-1, 0), ((r, c), (r, c + 1))), ((1, 0), ((r + 1, c + 1), (r + 1, c))),
                                 ((0, -1), ((r + 1, c), (r, c))), ((0, 1), ((r, c + 1), (r + 1, c + 1)))):
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


def footprint_mask(mesh, x, y, yaw, K, T, z_top=None, shape=(540, 960)):
    """Projected silhouette of a mesh (top-down footprint polygon at the top height, rasterised triangle by triangle)."""
    c, s = math.cos(yaw), math.sin(yaw)
    V = mesh.vertices @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + np.array([x, y, 0])
    uv, z = project(V, K, T)
    m = np.zeros(shape, np.uint8)
    tri = uv[mesh.faces].astype(np.int32)
    keep = mesh.face_normals[:, 2] > 0.5                  # top faces only (enough for flat objects seen from above)
    for t in tri[keep]:
        cv2.fillConvexPoly(m, t, 1)
    return m.astype(bool)


def scaled(mesh, s, flat=True):
    m = mesh.copy()
    m.vertices = m.vertices * np.array([s, s, 1.0 if flat else s])
    return m


def fit_flat(mesh, mask, K, T, x, y, yaw, flips=(False, True), care=None, free_scale=False, scale_bounds=(0.7, 1.4)):
    """Pose (x, y, yaw, flipped) of a flat mesh whose projected footprint best matches `mask`: coarse yaw sweep (10 deg) x flip, then
    Nelder-Mead on (x, y, yaw)."""
    from scipy.optimize import minimize
    care = np.ones_like(mask) if care is None else care
    mk = mask & care
    best = (-1, None)
    for fl in flips:
        m = mesh.copy()
        if fl:
            m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0)))
            m.apply_translation((0, 0, -m.bounds[0][2]))
        for yw in np.radians(np.arange(0, 360, 10)):
            v = iou(footprint_mask(m, x, y, yw, K, T) & care, mk)
            if v > best[0]:
                best = (v, (fl, yw, m))
    fl, yw0, m = best[1]

    def loss(q):
        mm = scaled(m, q[3]) if free_scale else m
        if free_scale and not scale_bounds[0] <= q[3] <= scale_bounds[1]:
            return 2.0
        return 1 - iou(footprint_mask(mm, q[0], q[1], q[2], K, T) & care, mk)
    q0 = np.array([x, y, yw0] + ([1.0] if free_scale else []))
    sim = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * [0.01, 0.01, 0.1, 0.05][i] for i in range(len(q0))])
    r = minimize(loss, q0, method="Nelder-Mead", options=dict(initial_simplex=sim, maxiter=300, xatol=1e-4, fatol=1e-4))
    return dict(x=float(r.x[0]), y=float(r.x[1]), yaw=float(r.x[2]), flipped=bool(fl), iou=1 - float(r.fun),
                scale=float(r.x[3]) if free_scale else 1.0)


# ------------------------------------------------------------------ USD
def convex_parts(mesh, threshold=0.05, max_parts=12):
    import coacd
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), threshold=threshold, max_convex_hull=max_parts,
                            preprocess_mode="auto")
    return [trimesh.Trimesh(np.asarray(v), np.asarray(f)) for v, f in parts]


def write_usd(path, mesh, tex_path, category, mass, mu, damping, collider, emissive=False, extra_attrs=None, emissive_scale=1.0):
    """One rigid-body USD (root /Root = rigid body, /Root/visual textured mesh, colliders under /Root):
    collider: ("hull",) | ("parts", [trimesh...]) | ("boxes", [(size3, pos3)]) | ("bowl", dict) | ("none",) ; tex_path relative to the usd."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    path = Path(path)
    if path.exists():
        path.unlink()
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root"); rp = root.GetPrim(); st.SetDefaultPrim(rp)
    rp.CreateAttribute("twin:category", Sdf.ValueTypeNames.String).Set(category)
    for k, v in (extra_attrs or {}).items():
        rp.CreateAttribute(f"twin:{k}", Sdf.ValueTypeNames.String).Set(str(v))
    if collider[0] != "none":
        UsdPhysics.RigidBodyAPI.Apply(rp)
        UsdPhysics.MassAPI.Apply(rp).CreateMassAttr(float(mass))
    mat = UsdShade.Material.Define(st, "/Root/Looks/mat")
    sh = UsdShade.Shader.Define(st, "/Root/Looks/mat/pbr"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    gm = UsdGeom.Mesh.Define(st, "/Root/visual")
    gm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
    gm.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    gm.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32)))
    gm.CreateSubdivisionSchemeAttr("none")
    if emissive:
        gm.CreateDoubleSidedAttr(True)
    gm.CreateExtentAttr([Gf.Vec3f(*map(float, V.min(0))), Gf.Vec3f(*map(float, V.max(0)))])
    uv = getattr(mesh.visual, "uv", None)
    if tex_path is not None and uv is not None and len(uv) == len(V):
        pv = UsdGeom.PrimvarsAPI(gm).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        rd = UsdShade.Shader.Define(st, "/Root/Looks/mat/st"); rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(st, "/Root/Looks/mat/tex"); tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex_path))
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp"); tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        if emissive:     # unlit: the photo's colours unchanged (background); diffuse black
            tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(emissive_scale, emissive_scale, emissive_scale, 1.0))
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.7, 0.7))
    if emissive and tex_path is not None:           # RTX ignores UsdPreviewSurface emission here: OmniPBR (MDL) emission-only material
        em = UsdShade.Material.Define(st, "/Root/Looks/emis")
        es = UsdShade.Shader.Define(st, "/Root/Looks/emis/Shader")
        es.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        es.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        es.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        es.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
        es.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
        es.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        es.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(tex_path)))
        es.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(emissive_scale))
        es.CreateOutput("out", Sdf.ValueTypeNames.Token)
        em.CreateSurfaceOutput("mdl").ConnectToSource(es.ConnectableAPI(), "out")
        em.CreateDisplacementOutput("mdl").ConnectToSource(es.ConnectableAPI(), "out")
        em.CreateVolumeOutput("mdl").ConnectToSource(es.ConnectableAPI(), "out")
        mat = em
    UsdShade.MaterialBindingAPI.Apply(gm.GetPrim()).Bind(mat)
    kind = collider[0]
    if kind == "hull":
        UsdPhysics.CollisionAPI.Apply(gm.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(gm.GetPrim()).CreateApproximationAttr("convexHull")
    elif kind == "parts":
        for i, p in enumerate(collider[1]):
            cm = UsdGeom.Mesh.Define(st, f"/Root/col_{i:02d}")
            pv_, pf_ = np.asarray(p.vertices, np.float32), np.asarray(p.faces, np.int32)
            cm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pv_))
            cm.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(pf_)))
            cm.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(pf_.reshape(-1)))
            cm.CreatePurposeAttr("guide")
            UsdPhysics.CollisionAPI.Apply(cm.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(cm.GetPrim()).CreateApproximationAttr("convexHull")
    elif kind == "boxes":
        for i, (sz, pos) in enumerate(collider[1]):
            c = UsdGeom.Cube.Define(st, f"/Root/col_{i:02d}"); c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos))); c.AddScaleOp().Set(Gf.Vec3f(*map(float, sz)))
            c.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    elif kind == "bowl":
        d = collider[1]
        cx, cy, r_bot, r_top, Hb = d["cx"], d["cy"], d["r_bot"], d["r_top"], d["H"]
        t_wall, floor_z, n_seg = 0.006, max(0.006, 0.08 * Hb), 32
        cyl = UsdGeom.Cylinder.Define(st, "/Root/col_floor"); cyl.CreateRadiusAttr(r_bot); cyl.CreateHeightAttr(floor_z); cyl.CreateAxisAttr("Z")
        cyl.AddTranslateOp().Set(Gf.Vec3d(cx, cy, floor_z / 2)); cyl.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        alpha = math.atan2(r_top - r_bot, Hb); Ls = math.hypot(r_top - r_bot, Hb)
        r_mid = (r_bot + r_top) / 2 - t_wall / 2 * math.cos(alpha)
        wdt = 2 * math.pi * ((r_bot + r_top) / 2) / n_seg * 1.15
        for k in range(n_seg):
            phi = 2 * math.pi * k / n_seg
            c = UsdGeom.Cube.Define(st, f"/Root/col_wall_{k:02d}"); c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(cx + r_mid * math.cos(phi), cy + r_mid * math.sin(phi), Hb / 2))
            q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(alpha)) * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(phi))).GetQuat()
            c.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *map(float, q.GetImaginary())))
            c.AddScaleOp().Set(Gf.Vec3f(t_wall, wdt, Ls)); c.CreatePurposeAttr("guide")
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if kind != "none":
        pm = UsdShade.Material.Define(st, "/Root/Looks/physics")
        pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        pma.CreateStaticFrictionAttr(float(mu[0])); pma.CreateDynamicFrictionAttr(float(mu[1])); pma.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(rp).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    st.GetRootLayer().Save()


def bowl_profile(V):
    Vb = V - np.r_[V[:, :2].mean(0), 0]
    zz = Vb[:, 2]; rr = np.hypot(Vb[:, 0], Vb[:, 1]); Hb = float(zz.max())
    r_bot = float(np.percentile(rr[zz < 0.15 * Hb], 90)) if np.any(zz < 0.15 * Hb) else 0.4 * rr.max()
    r_top = float(np.percentile(rr[zz > 0.85 * Hb], 95)) if np.any(zz > 0.85 * Hb) else rr.max()
    cx, cy = [float(v) for v in V[:, :2].mean(0)]
    return dict(cx=cx, cy=cy, r_bot=min(r_bot, r_top), r_top=r_top, H=Hb)


def tray_colliders(L, W, H, t=0.006):
    """Open box (floor + 4 walls) as 5 thin boxes: [(size, pos)]."""
    return [((L, W, t), (0, 0, t / 2)), ((t, W, H), (L / 2 - t / 2, 0, H / 2)), ((t, W, H), (-L / 2 + t / 2, 0, H / 2)),
            ((L, t, H), (0, W / 2 - t / 2, H / 2)), ((L, t, H), (0, -W / 2 + t / 2, H / 2))]
