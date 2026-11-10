"""Messy-table twin from one table_clean episode start: every object on the table at the start of a clean-up, placed where the video
measured it, with a mesh, a pose, a colour, a size, a mass and physics properties; plus the real table top as a texture.

  source scripts/env.sh; $SIM_PY -I -m cleantables.twin.table_scene IMG_2513_ep01 [--t_offset 0.3]

Inputs (from the episode ingest, read only): data/table_clean/{episodes.json, objects/<ep>.json, 3d/<ep>.npz} and
data/table_clean/_work/<video>/{frames/, depth.npz, sam3_inv.npz}; meshes from assets/table_clean/meshes (Hunyuan3D-2.1 of the recorded objects,
YCB) or simple procedural shapes for the categories without a usable Hunyuan mesh (tin can, glass, recycling box).
Output: data/twin_table/<ep>/scene.json (the episode's table frame: z up, origin where the camera's principal ray hits the table, metres),
usd/<obj>.usda (one per object, physics-ready), table_tex.png (orthophoto of the real table with objects inpainted out), start.jpg and
proof sheet topdown.png. CPU only (pxr + trimesh + OpenCV).

Per object:
  * present at the start: SAM 3 track first seen within START_WIN s of the episode start (main tracks only), duplicates of the same
    category within DEDUP_M merged (largest track wins), box flaps (cardboard inside the recycling-box footprint) dropped;
  * position: median 3D centroid over the first ~1.5 s of its track; footprint and yaw: PCA of its SAM 3 mask back-projected onto the table
    plane (max with the track's footprint, which under-measures round objects);
  * mesh pose: the trimesh stable resting pose whose height best matches the measured height (so a jug stands, a banana lies);
    scale: isotropic fit to the measured footprint/height, clamped to the category's plausible size (flat things: per-axis);
  * colour: median colour of its mask in the real frame;
  * physics: category mass, friction 0.5-0.8, no bounce; inertia computed by PhysX from the collision shape (no added inertia);
    light floppy things (bags, wrappers, cardboard) get air-drag-like damping so a touch does not launch them.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
TC = ROOT / "data/table_clean"
OUTROOT = ROOT / "data/twin_table"
MESHES = ROOT / "assets/table_clean/meshes"
START_WIN = 2.0
DEDUP_M = 0.09
H, W = 270, 480                       # SAM 3 / depth work resolution
FPS = 29.97
STRIDE = 3

# category -> asset recipe. mesh: registry mesh ids (instance chosen by index) or "proc:<shape>";
# size: plausible largest dimension [m] (clamp); mass kg; mu (static, dynamic); collider; damping (linear, angular); flat: per-axis fit
CAT = {
    "fruit/banana":            dict(mesh=["real_hy3d_banana_1", "real_hy3d_banana_2"], size=(0.15, 0.22), mass=0.12, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.3)),
    "fruit/apple":             dict(mesh=["ycb_013_apple"], size=(0.065, 0.08), mass=0.18, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.6)),
    "dish/mug":                dict(mesh=["real_hy3d_mug_0", "real_hy3d_mug_2"], size=(0.10, 0.13), mass=0.30, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.05, 0.1)),
    "dish/bowl":               dict(mesh=["real_hy3d_bowl_1", "real_hy3d_bowl_2"], size=(0.13, 0.17), mass=0.30, mu=(0.6, 0.5), col="bowl", damp=(0.05, 0.1)),
    "container/fruit_bowl":    dict(mesh=["real_hy3d_fruit_bowl_0", "real_hy3d_fruit_bowl_1"], size=(0.22, 0.28), mass=0.60, mu=(0.6, 0.5), col="bowl", damp=(0.05, 0.1)),
    "dish/glass":              dict(mesh=["proc:glass"], size=(0.09, 0.12), mass=0.20, mu=(0.5, 0.4), col="convexDecomposition", damp=(0.05, 0.1)),
    "recycling/tin_can":       dict(mesh=["proc:tin"], size=(0.10, 0.12), mass=0.06, mu=(0.5, 0.4), col="convexHull", damp=(0.05, 0.3)),
    "recycling/milk_jug":      dict(mesh=["real_hy3d_milk_jug_0", "real_hy3d_milk_jug_1"], size=(0.20, 0.27), mass=0.07, mu=(0.6, 0.5), col="convexHull", damp=(0.3, 0.3)),
    "recycling/cardboard":     dict(mesh=["real_hy3d_cardboard_0", "real_hy3d_cardboard_1", "real_hy3d_cardboard_2"], size=(0.12, 0.45), mass=0.05, mu=(0.7, 0.6), col="convexHull", damp=(1.0, 1.0), flat=True),
    "trash/plastic_bag":       dict(mesh=["real_hy3d_plastic_bag_1"], size=(0.10, 0.35), mass=0.01, mu=(0.8, 0.7), col="convexHull", damp=(3.0, 3.0), flat=True),
    "trash/wrapper":           dict(mesh=["real_hy3d_wrapper_0"], size=(0.07, 0.18), mass=0.01, mu=(0.8, 0.7), col="convexHull", damp=(3.0, 3.0), flat=True),
    "tool/dustpan":            dict(mesh=["real_hy3d_dustpan_1", "real_hy3d_dustpan_2"], size=(0.25, 0.36), mass=0.15, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.1, 0.1)),
    "container/recycling_box": dict(mesh=["proc:box"], size=(0.30, 0.50), mass=0.40, mu=(0.7, 0.6), col="boxes", damp=(0.1, 0.1)),
}
SKIP = {"container/bin"}              # the bin stands off the table (on the shelf / floor) in these videos


# ---------------------------------------------------------------- geometry helpers
def proc_mesh(kind):
    if kind == "tin":                                   # 7.4 cm x 11 cm food tin
        return trimesh.creation.cylinder(radius=0.037, height=0.11, sections=48)
    if kind == "glass":                                 # ribbed tumbler, open top, 3 mm wall
        prof = np.array([[0.0, 0.0], [0.031, 0.0], [0.036, 0.10], [0.033, 0.10], [0.028, 0.006], [0.0, 0.006]])
        return trimesh.creation.revolve(prof, sections=48)
    if kind == "box":                                   # open cardboard box, 1 x 1 x 1 m, 1 % walls (scaled per axis later)
        t = 0.02
        parts = [trimesh.creation.box((1, 1, t), trimesh.transformations.translation_matrix((0, 0, t / 2 - 0.5)))]
        for sx, sy, px, py in ((t, 1, 0.5 - t / 2, 0), (t, 1, -0.5 + t / 2, 0), (1, t, 0, 0.5 - t / 2), (1, t, 0, -0.5 + t / 2)):
            parts.append(trimesh.creation.box((sx, sy, 1), trimesh.transformations.translation_matrix((px, py, 0))))
        return trimesh.util.concatenate(parts)
    raise ValueError(kind)


PAINT = TC / "asset_meshes"
NO_TEXTURE = {"trash/plastic_bag"}       # clear plastic: Hunyuan paints it opaque brown; keep the translucent flat material
YUP2ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def paint_dir(name, category):
    """Hunyuan3D-2.1 Paint output for a registry mesh real_hy3d_<x>_<n>, or None."""
    if not name.startswith("real_hy3d_"):
        return None
    n = name.rsplit("_", 1)[1]
    d = PAINT / category.replace("/", "__") / f"{n}_paint"
    return d if (d / "textured_mesh.obj").exists() else None


def load_template(name, category=None):
    """-> (mesh, texture image path or None). Textured Hunyuan meshes are y-up in model units; scale/pose are refit below anyway."""
    if name.startswith("proc:"):
        return proc_mesh(name[5:]), None
    d = paint_dir(name, category) if category and category not in NO_TEXTURE else None
    if d is not None:
        m = trimesh.load(d / "textured_mesh.obj", force="mesh", process=False)
        m.apply_transform(YUP2ZUP)
        ref = trimesh.load(MESHES / f"{name}.obj", force="mesh", process=False)   # metres (the registry's measured size)
        m.apply_scale(float(ref.extents.max() / m.extents.max()))
        return m, d / "textured_mesh.jpg"
    return trimesh.load(MESHES / f"{name}.obj", force="mesh", process=True), None


def rest_pose(m, target_h, flat):
    """Stable resting pose (trimesh) whose height best matches the measured height; flat things: most probable pose."""
    T, p = trimesh.poses.compute_stable_poses(m, n_samples=1)
    cands = [(T[i], p[i]) for i in range(min(10, len(T))) if p[i] >= 0.01] or [(T[0], p[0])]
    if flat or not np.isfinite(target_h) or target_h < 0.02:
        Tb = cands[0][0]
    else:
        def cost(tp):
            mm = m.copy(); mm.apply_transform(tp[0])
            return abs(math.log(max(mm.extents[2], 1e-3) / target_h)) - 0.3 * tp[1]
        Tb = min(cands, key=cost)[0]
    m2 = m.copy(); m2.apply_transform(Tb)
    # major horizontal axis -> +x (PCA of the footprint), centre the footprint at the origin, bottom at z = 0
    xy = m2.vertices[:, :2] - m2.vertices[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    ax = v[:, np.argmax(w)]
    a = -math.atan2(ax[1], ax[0])
    m2.apply_transform(trimesh.transformations.rotation_matrix(a, (0, 0, 1)))
    lo, hi = m2.bounds
    m2.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    return m2


# ---------------------------------------------------------------- video helpers
def load_frame(video, k):
    return cv2.imread(str(TC / "_work" / video / "frames" / f"{k}.jpg"))


class Dets:
    """Lazy SAM 3 detections of one video (masks unpacked one at a time)."""

    def __init__(self, video):
        z = np.load(TC / "_work" / video / "sam3_inv.npz")
        self.frame, self.prompt, self.prob, self.box = z["frame"], z["prompt"], z["prob"], z["box"]
        self.prompts = [str(p) for p in z["prompts"]]
        self.bits = z["masks"]
        self.nb = H * W // 8
        self.frames = np.array(sorted(set(int(f) for f in z["frames_done"])))

    def mask(self, i):
        return np.unpackbits(self.bits[i * self.nb:(i + 1) * self.nb])[:H * W].reshape(H, W).astype(bool)

    def nearest_kf(self, k):
        return int(self.frames[np.argmin(np.abs(self.frames - k))])

    def find(self, k, prompt, cxy):
        """Detection at keyframe k with this prompt whose mask centroid (480x270 px) is closest to cxy."""
        pid = self.prompts.index(prompt)
        idx = np.nonzero((self.frame == k) & (self.prompt == pid) & (self.prob >= 0.45))[0]
        best, bd = None, 1e9
        for i in idx:
            m = self.mask(i)
            ys, xs = np.nonzero(m)
            if len(xs) < 15:
                continue
            d = math.hypot(xs.mean() - cxy[0], ys.mean() - cxy[1])
            if d < bd:
                best, bd = (i, m), d
        return (best, bd) if best else (None, 1e9)


def backproject_to_plane(px, K, T):
    """Pixels (N,2) in 960x540 coordinates -> points on the table plane z = 0 (table frame)."""
    Tinv = np.linalg.inv(T)
    cam_o = T[:3, 3]
    rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T            # camera frame
    d = rays @ T[:3, :3].T                                               # table frame directions
    s = -cam_o[2] / d[:, 2]
    return cam_o[None] + s[:, None] * d


def project(P, K, T):
    Tinv = np.linalg.inv(T)
    pc = P @ Tinv[:3, :3].T + Tinv[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


# ---------------------------------------------------------------- main
def build(ep_id, t_offset=0.3):
    eps = {e["id"]: e for e in json.load(open(TC / "episodes.json"))["episodes"]}
    ep = eps[ep_id]
    video = ep["video"]
    inv = json.load(open(TC / "objects" / f"{ep_id}.json"))
    z = np.load(TC / "3d" / f"{ep_id}.npz", allow_pickle=True)
    K, T = z["K"], z["T_cam_to_table"]
    out = OUTROOT / ep_id
    (out / "usd").mkdir(parents=True, exist_ok=True)
    dets = Dets(video)
    t0 = ep["start"] + t_offset
    k0 = dets.nearest_kf(int(round(t0 * FPS / STRIDE)))
    img = load_frame(video, k0)
    cv2.imwrite(str(out / "start.jpg"), img)
    Kh = K.copy(); Kh[:2] *= 0.5                                       # 480x270

    ids, ot, oxyz = z["obj_ids"], z["obj_t"], z["obj_xyz_table"]
    objs = []
    for o in inv["objects"]:
        if not o["main"] or o["first_seen"] > ep["start"] + START_WIN or o["category"] in SKIP or o["category"] not in CAT:
            continue
        sel = (ids == o["id"]) & (ot <= o["first_seen"] + 1.5)
        if sel.sum() == 0:
            continue
        xyz = np.median(oxyz[sel], 0)
        LW = np.nanmedian(z["obj_footprint_LW"][sel], 0)
        top = float(np.nanmedian(z["obj_top_height"][sel]))
        prompt = max(o["sam3_prompt_votes"], key=o["sam3_prompt_votes"].get)
        kk = dets.nearest_kf(int(round(o["first_seen"] * FPS / STRIDE)))
        (hit, dist) = dets.find(kk, prompt, np.array(o["first_centroid_px"]) / 4.0)
        yaw, LWm, colour, mask_kf = None, (np.nan, np.nan), None, None
        if hit is not None and dist < 25:
            i, m = hit
            mask_kf = kk
            ys, xs = np.nonzero(m)
            P = backproject_to_plane(np.c_[xs, ys] * 2.0, K, T)[:, :2]
            # tall objects smear away from the camera when projected onto the plane: use the near half (bottom of the mask) for yaw/size
            c = P - np.median(P, 0)
            lo_, hi_ = np.percentile(c, 2, 0), np.percentile(c, 98, 0)
            c = c[np.all((c >= lo_) & (c <= hi_), 1)]
            (_, _), (rw, rh), ang = cv2.minAreaRect(c.astype(np.float32))     # min-area rectangle: stable yaw for boxy shapes too
            yaw = math.radians(ang) if rw >= rh else math.radians(ang + 90)
            LWm = (float(max(rw, rh)), float(min(rw, rh)))
            fr = load_frame(video, kk)
            mm = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
            mm = cv2.resize(mm, (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mm.sum() > 20:
                colour = (np.median(fr[mm][:, ::-1], 0) / 255.0).tolist()   # RGB
        objs.append(dict(id=o["id"], category=o["category"], n_kf=o["n_keyframes"], xyz=xyz.tolist(), LW_track=LW.tolist(), LW_mask=list(LWm),
                         top=top, yaw=yaw, colour=colour, mask_kf=mask_kf))

    # ---- dedupe: same category close together -> keep the longest track; box flaps
    objs.sort(key=lambda o: -o["n_kf"])
    keep = []
    for o in objs:
        dup = [k for k in keep if k["category"] == o["category"] and math.dist(k["xyz"][:2], o["xyz"][:2]) < DEDUP_M]
        if dup:
            continue
        keep.append(o)
    boxes = [o for o in keep if o["category"] == "container/recycling_box"]
    if boxes:                                      # the biggest "cardboard box" track is the box; others are its flaps / misfires
        bx = max(boxes, key=lambda o: np.nanmax(o["LW_track"]) * max(o["top"], 0.01))
        keep = [o for o in keep if o["category"] != "container/recycling_box" or o is bx]
        r = 0.5 * np.nanmax(bx["LW_track"])
        keep = [o for o in keep if not (o["category"] == "recycling/cardboard" and math.dist(o["xyz"][:2], bx["xyz"][:2]) < r)]
        if bx["top"] < 0.08:                       # a flat "box" = a flattened sheet
            bx["category"] = "recycling/cardboard"
    print(f"{ep_id}: {len(objs)} start tracks -> {len(keep)} objects", flush=True)

    # ---- per-object asset
    rp_ = out / "refine.json"
    REFINE = json.load(open(rp_)) if rp_.exists() else {}
    used = {}
    placed = []
    for o in keep:
        rec = CAT[o["category"]]
        n = used.get(o["category"], 0); used[o["category"]] = n + 1
        tname = rec["mesh"][n % len(rec["mesh"])]
        m0, tex = load_template(tname, o["category"])
        flat = rec.get("flat", False)
        L_meas = np.nanmax([o["LW_track"][0], o["LW_mask"][0]])
        W_meas = np.nanmax([o["LW_track"][1], o["LW_mask"][1]])
        Hm = o["top"] if o["top"] > 0.015 else np.nan
        if tname == "proc:box":
            L = float(np.clip(L_meas, *rec["size"])); Wd = float(np.clip(W_meas, 0.25, 0.4)); Hb = float(np.clip(o["top"], 0.15, 0.3))
            m = m0.copy(); m.apply_scale((L, Wd, Hb)); m.apply_translation((0, 0, Hb / 2))
        else:
            m = rest_pose(m0, Hm, flat)
            e = m.extents
            if flat:                           # per-axis: footprint from the video, thickness from the mesh (clamped)
                L = float(np.clip(L_meas, rec["size"][0], rec["size"][1]))
                Wd = float(np.clip(W_meas, 0.3 * L, L))
                s = np.array([L / e[0], Wd / e[1], 1.0])
                s[2] = float(np.clip(np.sqrt(s[0] * s[1]), 0.3, 1.5))
                m.apply_scale(s)
            else:
                ratios = [L_meas / e[0], W_meas / e[1]] + ([Hm / e[2]] if np.isfinite(Hm) else [])
                ratios = [r for r in ratios if np.isfinite(r) and r > 0]
                s = float(np.median(ratios)) if ratios else 1.0
                big = e.max() * s
                s *= float(np.clip(big, *rec["size"])) / big
                m.apply_scale(s)
        # place: footprint centre at the measured centroid's xy; bottom on the table, or on top of what it was measured resting on
        yaw = o["yaw"] if o["yaw"] is not None else 0.0
        ext0 = m.extents.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
        rf = REFINE.get(o["id"])
        if rf:                                       # render-and-compare refinement: pose fitted to the SAM 3 mask
            c = m.vertices[:, :2].mean(0)
            m.apply_translation((-c[0], -c[1], 0)); m.apply_scale(rf["scale"])
            m.apply_transform(trimesh.transformations.rotation_matrix(rf["dyaw"], (0, 0, 1)))
            m.apply_translation((c[0], c[1], 0))
            o["xyz"] = [o["xyz"][0] + rf["dx"], o["xyz"][1] + rf["dy"], o["xyz"][2]]
            o["yaw"] = (o["yaw"] or 0.0) + rf["dyaw"]
        h = float(m.extents[2])
        z_bot = 0.0
        if np.isfinite(Hm) and Hm > h + 0.025:      # e.g. a banana lying in the fruit bowl
            z_bot = Hm - h
        objs_entry = dict(o, template=tname, texture=str(tex) if tex else None, extents=m.extents.round(4).tolist(), ext0=ext0.tolist(), z_bottom=round(z_bot, 4), rec={k: v for k, v in rec.items() if k != "mesh"})
        placed.append((objs_entry, m))

    # ---- simple stacking resolution: lift anything whose box overlaps a placed non-container below it (containers may hold things)
    placed.sort(key=lambda p: p[0]["z_bottom"])
    boxes3 = []
    hollow = {"container/fruit_bowl", "dish/bowl", "container/recycling_box"}
    for e, m in placed:
        x, y = e["xyz"][0], e["xyz"][1]
        lo = m.bounds[0] + (x, y, 0); hi = m.bounds[1] + (x, y, 0)
        zb = e["z_bottom"]
        for (lo2, hi2, cat2, zt2, zf2) in boxes3:
            ov = min(hi[0], hi2[0]) - max(lo[0], lo2[0]), min(hi[1], hi2[1]) - max(lo[1], lo2[1])
            if ov[0] > 0.01 and ov[1] > 0.01:
                zb = max(zb, (zf2 if cat2 in hollow else zt2) + 0.004)
        e["z_bottom"] = round(zb, 4)
        boxes3.append((lo, hi, e["category"], zb + m.extents[2], zb + 0.012))

    # ---- write USDs
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    scene_objs = []
    for e, m in placed:
        rec = e["rec"]
        oid = e["id"]
        usd = out / "usd" / f"{oid}.usda"
        if usd.exists():
            usd.unlink()
        st = Usd.Stage.CreateNew(str(usd))
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
        root = UsdGeom.Xform.Define(st, "/Root"); rp = root.GetPrim(); st.SetDefaultPrim(rp)
        rp.CreateAttribute("twin:category", Sdf.ValueTypeNames.String).Set(e["category"])
        UsdPhysics.RigidBodyAPI.Apply(rp)
        UsdPhysics.MassAPI.Apply(rp).CreateMassAttr(float(rec["mass"]))
        mat = UsdShade.Material.Define(st, "/Root/Looks/mat")
        sh = UsdShade.Shader.Define(st, "/Root/Looks/mat/pbr"); sh.CreateIdAttr("UsdPreviewSurface")
        col = e["colour"] or [0.7, 0.7, 0.7]
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, col)))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25 if e["category"] in ("dish/glass", "trash/plastic_bag") else 0.6)
        if e["category"] == "dish/glass":
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
        if e["category"] == "trash/plastic_bag":
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        V, F = np.asarray(m.vertices), np.asarray(m.faces)
        mesh = UsdGeom.Mesh.Define(st, "/Root/visual")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in V]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in F.reshape(-1)]))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr([Gf.Vec3f(*map(float, V.min(0))), Gf.Vec3f(*map(float, V.max(0)))])
        uv = getattr(m.visual, "uv", None) if e.get("texture") else None
        if uv is not None and len(uv) == len(V):
            import shutil
            (out / "usd" / "textures").mkdir(exist_ok=True)
            shutil.copy(e["texture"], out / "usd" / "textures" / f"{oid}.jpg")
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in uv]))
            rd = UsdShade.Shader.Define(st, "/Root/Looks/mat/st"); rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(st, "/Root/Looks/mat/tex"); tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{oid}.jpg")
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.GetInput("diffuseColor").ConnectToSource(tx.ConnectableAPI(), "rgb")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        if rec["col"] == "boxes":                  # open box: 5 thin box colliders (convex, exact)
            ext = e["ext0"]
            t = 0.006
            for nm, sz, pos in (("floor", (ext[0], ext[1], t), (0, 0, t / 2)),
                                ("wx0", (t, ext[1], ext[2]), (ext[0] / 2 - t / 2, 0, ext[2] / 2)), ("wx1", (t, ext[1], ext[2]), (-ext[0] / 2 + t / 2, 0, ext[2] / 2)),
                                ("wy0", (ext[0], t, ext[2]), (0, ext[1] / 2 - t / 2, ext[2] / 2)), ("wy1", (ext[0], t, ext[2]), (0, -ext[1] / 2 + t / 2, ext[2] / 2))):
                c = UsdGeom.Cube.Define(st, f"/Root/col_{nm}")
                c.CreateSizeAttr(1.0)
                # undo the yaw baked into the visual: colliders are built in the yawed frame
                R = trimesh.transformations.rotation_matrix(e["yaw"] or 0.0, (0, 0, 1))
                p = R[:3, :3] @ np.array(pos)
                c.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
                q = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(e["yaw"] or 0.0)).GetQuat()
                c.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *map(float, q.GetImaginary())))
                c.AddScaleOp().Set(Gf.Vec3f(*map(float, sz)))
                c.CreatePurposeAttr("guide")
                UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        elif rec["col"] == "bowl":                 # floor disc + 32 wall slabs following the mesh's outer profile (convex, exact, no filled hull)
            Vb = np.asarray(m.vertices) - np.r_[np.asarray(m.vertices)[:, :2].mean(0), 0]
            zz = Vb[:, 2]; rr = np.hypot(Vb[:, 0], Vb[:, 1]); Hb = float(zz.max())
            r_bot = float(np.percentile(rr[zz < 0.15 * Hb], 90)) if np.any(zz < 0.15 * Hb) else 0.4 * rr.max()
            r_top = float(np.percentile(rr[zz > 0.85 * Hb], 95)) if np.any(zz > 0.85 * Hb) else rr.max()
            r_bot = min(r_bot, r_top)
            t_wall, floor_z, n_seg = 0.006, max(0.006, 0.08 * Hb), 32
            cx, cy = [float(v) for v in np.asarray(m.vertices)[:, :2].mean(0)]
            cyl = UsdGeom.Cylinder.Define(st, "/Root/col_floor"); cyl.CreateRadiusAttr(r_bot); cyl.CreateHeightAttr(floor_z); cyl.CreateAxisAttr("Z")
            cyl.AddTranslateOp().Set(Gf.Vec3d(cx, cy, floor_z / 2)); cyl.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
            alpha = math.atan2(r_top - r_bot, Hb); L = math.hypot(r_top - r_bot, Hb)
            r_mid = (r_bot + r_top) / 2 - t_wall / 2 * math.cos(alpha)
            wdt = 2 * math.pi * ((r_bot + r_top) / 2) / n_seg * 1.15
            for k in range(n_seg):
                phi = 2 * math.pi * k / n_seg
                c = UsdGeom.Cube.Define(st, f"/Root/col_wall_{k:02d}"); c.CreateSizeAttr(1.0)
                c.AddTranslateOp().Set(Gf.Vec3d(cx + r_mid * math.cos(phi), cy + r_mid * math.sin(phi), Hb / 2))
                q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(alpha)) * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(phi))).GetQuat()
                c.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *map(float, q.GetImaginary())))
                c.AddScaleOp().Set(Gf.Vec3f(t_wall, wdt, L)); c.CreatePurposeAttr("guide")
                UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        else:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(rec["col"])
        pm = UsdShade.Material.Define(st, "/Root/Looks/physics")
        pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        pma.CreateStaticFrictionAttr(float(rec["mu"][0])); pma.CreateDynamicFrictionAttr(float(rec["mu"][1])); pma.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(rp).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
        st.GetRootLayer().Save()
        m.export(out / "usd" / f"{oid}.obj")
        scene_objs.append(dict(id=oid, category=e["category"], template=e["template"], textured=bool(e.get("texture")), usd=str(usd.relative_to(ROOT)),
                               pos=[e["xyz"][0], e["xyz"][1], e["z_bottom"]], extents=e["extents"], yaw=e["yaw"], colour=e["colour"],
                               mass=rec["mass"], mu=rec["mu"], damping=rec["damp"], collider=rec["col"],
                               measured=dict(LW_track=e["LW_track"], LW_mask=e["LW_mask"], top=e["top"], n_keyframes=e["n_kf"])))

    # ---- table texture: orthophoto of the real table (2 mm/px), non-table pixels (objects, hands, person) inpainted
    dz = np.load(TC / "_work" / video / "depth.npz")
    k_tex = dets.nearest_kf(int(round((ep["end"] - 0.5) * FPS / STRIDE)))      # clean-up ends with a clear table: best texture source
    img_tex = load_frame(video, k_tex)
    kd = int(np.argmin(np.abs(dz["k"] - k_tex)))
    D = dz["depth"][kd].astype(np.float32)
    vv, uu = np.mgrid[0:H, 0:W]
    rays = np.c_[uu.ravel() * 2.0, vv.ravel() * 2.0, np.ones(H * W)] @ np.linalg.inv(K).T
    # per-keyframe depth scale to the reference plane: ratio of the ray's plane-hit depth to MoGe depth on table-like pixels
    dt = rays @ T[:3, :3].T
    s_plane = -T[2, 3] / dt[:, 2]
    r = s_plane / np.maximum(D.ravel(), 1e-3)
    ok = (r > 0.8) & (r < 1.25)
    D = D * (float(np.median(r[ok])) if ok.any() else 1.0)
    Pc = rays * D.ravel()[:, None]
    Pt = Pc @ T[:3, :3].T + T[:3, 3]
    is_table = (np.abs(Pt[:, 2]) < 0.012).reshape(H, W)
    # other surfaces at the same height (kitchen worktop) are also "on the plane": keep the connected region that holds the objects
    closed = cv2.morphologyEx(is_table.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    for (e, m) in placed:                                     # objects sit on the table: fill their footprints so they join it
        uvp, _ = project(np.c_[m.vertices[:, :2] + e["xyz"][:2], np.zeros(len(m.vertices))], K, T)
        hull = cv2.convexHull((uvp / 2).astype(np.int32))
        cv2.fillConvexPoly(closed, hull, 1)
    n, lab = cv2.connectedComponents(closed)
    cuv, _ = project(np.array([[e["xyz"][0], e["xyz"][1], 0.0] for e, _ in placed]), K, T)
    cuv = np.clip((cuv / 2).astype(int), 0, [W - 1, H - 1])
    votes = np.bincount(lab[cuv[:, 1], cuv[:, 0]], minlength=n)
    votes[0] = 0
    is_table = (lab == int(np.argmax(votes))) & is_table
    tab_xy = Pt[is_table.ravel(), :2]
    x0, x1 = np.percentile(tab_xy[:, 0], [1, 99]); y0, y1 = np.percentile(tab_xy[:, 1], [1, 99])
    ox = np.array([e["xyz"][:2] for e, _ in placed])
    x0, x1 = min(x0, ox[:, 0].min() - 0.15), max(x1, ox[:, 0].max() + 0.15)
    y0, y1 = min(y0, ox[:, 1].min() - 0.15), max(y1, ox[:, 1].max() + 0.15)
    # the table continues toward the camera below the image: extend 0.25 m
    y0 -= 0.25
    res = 0.002
    gx, gy = np.meshgrid(np.arange(x0, x1, res), np.arange(y1, y0, -res))      # image rows: +y (far) at the top
    G = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    uv, depth = project(G, K, T)
    mx, my = uv[:, 0].reshape(gx.shape).astype(np.float32), uv[:, 1].reshape(gx.shape).astype(np.float32)
    ortho = cv2.remap(img_tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    objmask = np.zeros((H, W), bool)                       # every SAM 3 detection in this keyframe (flat objects pass the plane test)
    for i in np.nonzero((dets.frame == k_tex) & (dets.prob >= 0.3))[0]:
        objmask |= dets.mask(i)
    objmask = cv2.dilate(objmask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    tm = cv2.resize((is_table & ~objmask).astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST)
    tm = cv2.erode(tm, np.ones((7, 7), np.uint8))
    valid = cv2.remap(tm, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    # small holes (objects, hands) -> inpaint; big holes (outside the photo, under the person) -> tiled patch of real table
    inview = (mx >= 0) & (mx < 959) & (my >= 0) & (my < 539)
    integral = cv2.integral(valid.astype(np.uint8))
    ps = 150                                                        # 30 cm patch
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    sat = cv2.cvtColor(ortho, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    best, bp = -1e9, (0, 0)
    for r in range(0, valid.shape[0] - ps, 10):
        for c in range(0, valid.shape[1] - ps, 10):
            v = integral[r + ps, c + ps] - integral[r, c + ps] - integral[r + ps, c] + integral[r, c]
            if v < 0.97 * ps * ps:
                continue
            sc = -lap[r:r + ps, c:c + ps].mean() - 0.05 * np.abs(sat[r:r + ps, c:c + ps] - np.median(sat[valid])).mean()
            if sc > best:
                best, bp = sc, (r, c)
    patch = ortho[bp[0]:bp[0] + ps, bp[1]:bp[1] + ps]
    patch = np.concatenate([patch, patch[:, ::-1]], 1); patch = np.concatenate([patch, patch[::-1]], 0)   # mirror-tile: no seams
    reps = (valid.shape[0] // patch.shape[0] + 1, valid.shape[1] // patch.shape[1] + 1, 1)
    tiled = np.tile(patch, reps)[:valid.shape[0], :valid.shape[1]]
    hole = (~valid).astype(np.uint8)
    big = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8)).astype(bool) | ~inview
    base = np.where(big[..., None], tiled, ortho)
    small_hole = (hole.astype(bool) & ~big).astype(np.uint8) * 255
    filled = cv2.inpaint(base, small_hole, 5, cv2.INPAINT_TELEA)
    tex = np.where(valid[..., None], ortho, filled)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    cv2.imwrite(str(out / "table_tex.png"), tex)
    table = dict(x=[float(x0), float(x1)], y=[float(y0), float(y1)], top_z=0.0, texture="table_tex.png", tex_res_m=res, tex_keyframe=k_tex,
                 tex_valid_frac=float(valid.mean()))

    scene = dict(episode=ep_id, video=video, keyframe=k0, t_start=t0, K=K.tolist(), T_cam_to_table=T.tolist(), image_wh=[960, 540],
                 frame="recon3d table frame: z up (table normal), x = image right on the table, origin = principal ray x table, metres",
                 table=table, objects=scene_objs,
                 notes=["sizes from MoGe-3 metric depth (not tape-measured); round objects are under-measured, clamped to category size ranges",
                        "colours: median colour of the SAM 3 mask; meshes are untextured Hunyuan3D-2.1 / YCB / procedural",
                        "physics: real masses (category priors), PhysX-computed inertia (no +0.01 kg m^2), restitution 0"])
    json.dump(scene, open(out / "scene.json", "w"), indent=1)

    # ---- proof: top-down + footprints over the orthophoto
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(img[..., ::-1]); ax[0].set_title(f"{ep_id} start frame (k={k0})"); ax[0].axis("off")
    ax[1].imshow(tex[..., ::-1], extent=[x0, x1, y0, y1])
    for (e, m) in placed:
        xy = m.vertices[:, :2] + e["xyz"][:2]
        h2 = cv2.convexHull(xy.astype(np.float32))[:, 0]
        ax[1].fill(h2[:, 0], h2[:, 1], alpha=0.45, color=e["colour"] or (0.7, 0.7, 0.7), ec="k")
        ax[1].text(e["xyz"][0], e["xyz"][1], e["id"], fontsize=7, ha="center")
        # also project the footprint into the photo
        P3 = np.c_[xy, np.full(len(xy), e["z_bottom"])]
        uvp, _ = project(P3[::max(1, len(P3) // 400)], K, T)
        ax[0].scatter(uvp[:, 0], uvp[:, 1], s=1, color=e["colour"] or "w")
    ax[1].set_title("twin layout over the real table orthophoto (objects inpainted)"); ax[1].set_aspect("equal")
    plt.tight_layout(); plt.savefig(out / "topdown.png", dpi=80); plt.close()
    print("wrote", out, flush=True)
    return scene


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--t_offset", type=float, default=0.3)
    a = ap.parse_args()
    for e in a.episodes:
        build(e, a.t_offset)
