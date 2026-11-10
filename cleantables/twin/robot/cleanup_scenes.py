"""Randomised table-cleanup scenes for the Isaac twin: fruit -> fruit bowl, tools -> open toolbox, cans -> recycling box, plus out-of-scope
obstacles that must not be touched.

  source scripts/env.sh
  $SIM_PY -m cleantables.twin.robot.cleanup_scenes --build_assets          # once: object USDs in data/twin_table/cleanup_assets/
  $SIM_PY -m cleantables.twin.robot.cleanup_scenes --name cl_seen_0 --split seen --seed 0

A "scene" is one composition (a fixed list of objects: containers, in-scope items, obstacles) saved as data/twin_table/<name>/scene.json in
the same format as the video twins (so cleantables.twin.robot.env_kuka / env_franka load it unchanged).  Positions inside the scene.json
are only nominal: the env draws a fresh random layout (position + yaw, no overlaps, reachable by an arm) for every parallel episode with `sample_layouts`.

Object library:
  fruit    banana x2 (Hunyuan3D of a recorded banana), apple (YCB 013)             SEEN   | pear, orange, lemon, plum (YCB)                 HELD-OUT
  tools    wrench (YCB 042), hammer (YCB 048), phillips screwdriver (YCB 043)    "seen" | scissors (YCB 037), power drill (YCB 035), padlock (YCB 038)  HELD-OUT
  cans     procedural tin 7.4 x 11 cm (the tin used in the video twins)           SEEN   | tomato-soup (YCB 005), tuna (YCB 007), master chef (YCB 002)   HELD-OUT
  obstacles mug, milk jug (Hunyuan3D), cracker box (YCB 003), wood block (YCB 036), plate (YCB 029)
  containers fruit bowl (Hunyuan3D bowl + exact wall colliders), open toolbox and recycling box (procedural open boxes)
"Seen" = types that occur in the video twins or are designated training types; no tool occurs in these video twins, so the tool split
is a designation, not a measured data property.  There is no pliers or tape-roll mesh; YCB objects come from the YCB S3 bucket (CC BY 4.0),
stored in assets/twin_cleanup/raw.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "data/twin_table/cleanup_assets"
RAW = ROOT / "assets/twin_cleanup/raw"
RAW_TC = ROOT / "assets/table_clean/raw"

# key: class, split, kind, source, mass kg, mu (static, dyn), collider, damping (lin, ang), pose ("flat" most stable | "upright" tallest), size clamp (largest dim, m)
LIB = {
    "banana_1": dict(cls="fruit", split="seen", cat="fruit/banana", src=("hy3d", "real_hy3d_banana_1"), mass=0.12, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.3), clamp=(0.15, 0.20)),
    "banana_2": dict(cls="fruit", split="seen", cat="fruit/banana", src=("hy3d", "real_hy3d_banana_2"), mass=0.12, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.3), clamp=(0.15, 0.20)),
    "apple": dict(cls="fruit", split="seen", cat="fruit/apple", src=("ycb", RAW_TC / "ycb_013_apple/013_apple/google_16k"), mass=0.18, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.6)),
    "pear": dict(cls="fruit", split="heldout", cat="fruit/pear", src=("ycb", RAW_TC / "ycb_016_pear/016_pear/google_16k"), mass=0.20, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.6)),
    "orange": dict(cls="fruit", split="heldout", cat="fruit/orange", src=("ycb", RAW_TC / "ycb_017_orange/017_orange/google_16k"), mass=0.20, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.6)),
    "lemon": dict(cls="fruit", split="heldout", cat="fruit/lemon", src=("ycb", RAW_TC / "ycb_014_lemon/014_lemon/google_16k"), mass=0.10, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.6)),
    "plum": dict(cls="fruit", split="heldout", cat="fruit/plum", src=("ycb", RAW_TC / "ycb_018_plum/018_plum/google_16k"), mass=0.05, mu=(0.7, 0.6), col="convexHull", damp=(0.05, 0.6)),
    "wrench": dict(cls="tool", split="seen", cat="tool/wrench", src=("ycb", RAW / "042_adjustable_wrench/042_adjustable_wrench/google_16k"), mass=0.20, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.05, 0.2)),
    "hammer": dict(cls="tool", split="seen", cat="tool/hammer", src=("ycb", RAW / "048_hammer/048_hammer/google_16k"), mass=0.30, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.05, 0.2)),
    "screwdriver": dict(cls="tool", split="seen", cat="tool/screwdriver", src=("ycb", RAW / "043_phillips_screwdriver/043_phillips_screwdriver/google_16k"), mass=0.10, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.05, 0.2)),
    "scissors": dict(cls="tool", split="heldout", cat="tool/scissors", src=("ycb", RAW / "037_scissors/037_scissors/google_16k"), mass=0.08, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.05, 0.2)),
    "drill": dict(cls="tool", split="heldout", cat="tool/power_drill", src=("ycb", RAW / "035_power_drill/035_power_drill/google_16k"), mass=0.40, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.05, 0.2)),
    "padlock": dict(cls="tool", split="heldout", cat="tool/padlock", src=("ycb", RAW / "038_padlock/038_padlock/google_16k"), mass=0.10, mu=(0.6, 0.5), col="convexHull", damp=(0.05, 0.2)),
    "tin": dict(cls="can", split="seen", cat="recycling/tin_can", src=("proc", "tin"), mass=0.06, mu=(0.5, 0.4), col="convexHull", damp=(0.05, 0.3)),
    "soup_can": dict(cls="can", split="heldout", cat="recycling/soup_can", src=("ycb", RAW / "005_tomato_soup_can/005_tomato_soup_can/google_16k"), mass=0.10, mu=(0.5, 0.4), col="convexHull", damp=(0.05, 0.3), pose="upright"),
    "tuna_can": dict(cls="can", split="heldout", cat="recycling/tuna_can", src=("ycb", RAW / "007_tuna_fish_can/007_tuna_fish_can/google_16k"), mass=0.06, mu=(0.5, 0.4), col="convexHull", damp=(0.05, 0.3)),
    "chef_can": dict(cls="can", split="heldout", cat="recycling/chef_can", src=("ycb", RAW / "002_master_chef_can/002_master_chef_can/google_16k"), mass=0.12, mu=(0.5, 0.4), col="convexHull", damp=(0.05, 0.3), pose="upright"),
    "mug": dict(cls="obstacle", split="-", cat="dish/mug", src=("hy3d", "real_hy3d_mug_0"), mass=0.30, mu=(0.6, 0.5), col="convexDecomposition", damp=(0.05, 0.1), clamp=(0.10, 0.13)),
    "milk_jug": dict(cls="obstacle", split="-", cat="recycling/milk_jug", src=("hy3d", "real_hy3d_milk_jug_0"), mass=0.15, mu=(0.6, 0.5), col="convexHull", damp=(0.3, 0.3), clamp=(0.20, 0.24), pose="upright"),
    "cracker_box": dict(cls="obstacle", split="-", cat="food/cracker_box", src=("ycb", RAW / "003_cracker_box/003_cracker_box/google_16k"), mass=0.15, mu=(0.6, 0.5), col="convexHull", damp=(0.1, 0.1), pose="upright"),
    "wood_block": dict(cls="obstacle", split="-", cat="misc/wood_block", src=("ycb", RAW / "036_wood_block/036_wood_block/google_16k"), mass=0.30, mu=(0.6, 0.5), col="convexHull", damp=(0.1, 0.1)),
    "plate": dict(cls="obstacle", split="-", cat="dish/plate", src=("ycb", RAW / "029_plate/029_plate/google_16k"), mass=0.25, mu=(0.6, 0.5), col="convexHull", damp=(0.1, 0.1)),
    # containers (kinematic): fruit bowl = Hunyuan3D bowl with exact wall colliders; open boxes procedural
    "fruit_bowl": dict(cls="container", split="-", cat="container/fruit_bowl", src=("hy3d", "real_hy3d_fruit_bowl_0"), mass=0.6, mu=(0.6, 0.5), col="bowl", damp=(0.05, 0.1), clamp=(0.22, 0.24)),
    "toolbox": dict(cls="container", split="-", cat="container/toolbox", src=("box", (0.32, 0.20, 0.09)), mass=0.5, mu=(0.7, 0.6), col="boxes", damp=(0.1, 0.1), colour=(0.78, 0.15, 0.12)),
    "recycling_box": dict(cls="container", split="-", cat="container/recycling_box", src=("box", (0.30, 0.24, 0.14)), mass=0.4, mu=(0.7, 0.6), col="boxes", damp=(0.1, 0.1), colour=(0.15, 0.35, 0.75)),
}
DEST = {"fruit": "fruit_bowl", "tool": "toolbox", "can": "recycling_box"}
SPLITS = {s: {c: [k for k, v in LIB.items() if v["cls"] == c and v["split"] == s] for c in ("fruit", "tool", "can")} for s in ("seen", "heldout")}
OBSTACLES = [k for k, v in LIB.items() if v["cls"] == "obstacle"]
TABLE_SRC = ["IMG_2513_ep01", "IMG_2514_ep01", "IMG_2516_ep10"]


def _mesh_and_tex(key):
    import trimesh
    from cleantables.twin import table_scene as B
    rec = LIB[key]
    kind, src = rec["src"]
    if kind == "hy3d":
        return B.load_template(src, rec["cat"])
    if kind == "ycb":
        d = Path(src)
        m = trimesh.load(d / "textured.obj", force="mesh", process=False)
        return m, d / "texture_map.png"
    if kind == "proc":
        return B.proc_mesh(src), None
    raise ValueError(kind)


def build_asset(key):
    """-> dict with the library entry's physical description; writes data/twin_table/cleanup_assets/<key>.{usda,obj}."""
    import trimesh
    from cleantables.twin import table_scene as B
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    rec = LIB[key]
    ASSETS.mkdir(parents=True, exist_ok=True)
    tex = None
    if rec["src"][0] == "box":
        ex = rec["src"][1]
        m = B.proc_mesh("box").copy(); m.apply_scale(ex); m.apply_translation((0, 0, ex[2] / 2))
    else:
        m0, tex = _mesh_and_tex(key)
        if rec.get("pose") == "upright":              # tallest stable pose (a can stands on its base)
            T, p = trimesh.poses.compute_stable_poses(m0, n_samples=1)
            hs = []
            for Ti in T[:10]:
                mm = m0.copy(); mm.apply_transform(Ti); hs.append(mm.extents[2])
            Tb = T[int(np.argmax(hs))]
            m = m0.copy(); m.apply_transform(Tb)
            lo, hi = m.bounds
            m.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
        else:
            m = B.rest_pose(m0, float("nan"), True)
        if "clamp" in rec:
            big = float(m.extents.max()); s = float(np.clip(big, *rec["clamp"])) / big
            m.apply_scale(s)
    ext = m.extents.copy()
    usd = ASSETS / f"{key}.usda"
    if usd.exists():
        usd.unlink()
    st = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root"); rp = root.GetPrim(); st.SetDefaultPrim(rp)
    rp.CreateAttribute("twin:category", Sdf.ValueTypeNames.String).Set(rec["cat"])
    UsdPhysics.RigidBodyAPI.Apply(rp)
    UsdPhysics.MassAPI.Apply(rp).CreateMassAttr(float(rec["mass"]))
    mat = UsdShade.Material.Define(st, "/Root/Looks/mat")
    sh = UsdShade.Shader.Define(st, "/Root/Looks/mat/pbr"); sh.CreateIdAttr("UsdPreviewSurface")
    col = rec.get("colour", (0.72, 0.72, 0.75) if key == "tin" else (0.7, 0.7, 0.7))
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, col)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45 if key == "tin" else 0.6)
    if key == "tin":
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    mesh = UsdGeom.Mesh.Define(st, "/Root/visual")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in V]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in F.reshape(-1)]))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, V.min(0))), Gf.Vec3f(*map(float, V.max(0)))])
    uv = getattr(m.visual, "uv", None) if tex is not None else None
    if uv is not None and len(uv) == len(V):
        (ASSETS / "textures").mkdir(exist_ok=True)
        shutil.copy(tex, ASSETS / "textures" / f"{key}{Path(tex).suffix}")
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in uv]))
        rd = UsdShade.Shader.Define(st, "/Root/Looks/mat/st"); rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(st, "/Root/Looks/mat/tex"); tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{key}{Path(tex).suffix}")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.GetInput("diffuseColor").ConnectToSource(tx.ConnectableAPI(), "rgb")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    if rec["col"] == "boxes":                                  # open box: floor + 4 thin walls (convex, exact)
        t = 0.006
        for nm, sz, pos in (("floor", (ext[0], ext[1], t), (0, 0, t / 2)),
                            ("wx0", (t, ext[1], ext[2]), (ext[0] / 2 - t / 2, 0, ext[2] / 2)), ("wx1", (t, ext[1], ext[2]), (-ext[0] / 2 + t / 2, 0, ext[2] / 2)),
                            ("wy0", (ext[0], t, ext[2]), (0, ext[1] / 2 - t / 2, ext[2] / 2)), ("wy1", (ext[0], t, ext[2]), (0, -ext[1] / 2 + t / 2, ext[2] / 2))):
            c = UsdGeom.Cube.Define(st, f"/Root/col_{nm}"); c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos))); c.AddScaleOp().Set(Gf.Vec3f(*map(float, sz)))
            c.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    elif rec["col"] == "bowl":                                 # floor disc + 32 wall slabs along the outer profile (as table_scene.py)
        Vb = V - np.r_[V[:, :2].mean(0), 0]
        zz = Vb[:, 2]; rr = np.hypot(Vb[:, 0], Vb[:, 1]); Hb = float(zz.max())
        r_bot = float(np.percentile(rr[zz < 0.15 * Hb], 90)) if np.any(zz < 0.15 * Hb) else 0.4 * rr.max()
        r_top = float(np.percentile(rr[zz > 0.85 * Hb], 95)) if np.any(zz > 0.85 * Hb) else rr.max()
        r_bot = min(r_bot, r_top)
        t_wall, floor_z, n_seg = 0.006, max(0.006, 0.08 * Hb), 32
        cx, cy = [float(v) for v in V[:, :2].mean(0)]
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
    m.export(ASSETS / f"{key}.obj")
    return dict(extents=[round(float(x), 4) for x in ext])


def build_assets(keys=None):
    out = json.load(open(ASSETS / "assets.json")) if (ASSETS / "assets.json").exists() else {}
    for k in (keys or LIB):
        try:
            out[k] = build_asset(k)
            print("asset", k, out[k]["extents"], flush=True)
        except Exception as e:                                      # reported, not hidden
            print("ASSET FAILED", k, repr(e), flush=True)
    json.dump(out, open(ASSETS / "assets.json", "w"), indent=1)
    return out


# ------------------------------------------------------------------------------------------------ scene composition
def reach_poses(table, sep=0.52):
    """Two arm bases side by side behind the far edge (+y), centred on the table (same convention as env.arm_poses)."""
    xc = 0.5 * (table["x"][0] + table["x"][1])
    y_b = table["y"][1] + 0.15
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    return {"a": ((xc + sep / 2, y_b, -0.255), q), "b": ((xc - sep / 2, y_b, -0.255), q)}


def make_scene(name, split="seen", seed=0, n_obstacles=None, types=None, n_per_class=1, spare=False, counts=None, sep=0.52):
    """Compose one scene: 3 containers + n_per_class items of each class (types drawn from the split) + 1-3 obstacles."""
    rng = np.random.default_rng(seed)
    assets = json.load(open(ASSETS / "assets.json"))
    if types is None:
        cnt = dict(zip(("fruit", "tool", "can"), counts)) if counts else {c: n_per_class for c in ("fruit", "tool", "can")}
        types = {c: [str(rng.choice(SPLITS[split][c])) for _ in range(cnt[c])] for c in ("fruit", "tool", "can")}
    n_obs = int(rng.integers(1, 4)) if n_obstacles is None else n_obstacles
    obstacles = [str(x) for x in rng.choice(OBSTACLES, n_obs, replace=False)]
    src_ep = TABLE_SRC[int(rng.integers(len(TABLE_SRC)))]                 # table texture randomisation: the three real tables
    sdir = ROOT / "data/twin_table" / name
    sdir.mkdir(parents=True, exist_ok=True)
    s0 = json.load(open(ROOT / "data/twin_table" / src_ep / "scene.json"))
    tex = (ROOT / "data/twin_table" / src_ep / "table_tex.png")
    shutil.copy(tex, sdir / "table_tex.png")
    txt = open(ROOT / "data/twin_table" / src_ep / "table.usda").read()
    txt = txt.replace(f"data/twin_table/{src_ep}/table_tex.png", f"data/twin_table/{name}/table_tex.png")
    open(sdir / "table.usda", "w").write(txt)
    table = s0["table"]
    objs = []

    def add(key, oid, role, cls):
        rec = LIB[key]
        objs.append(dict(id=oid, key=key, category=rec["cat"], cls=cls, role=role, split=rec["split"], usd=str((ASSETS / f"{key}.usda").relative_to(ROOT)),
                         pos=[0.0, 0.0, 0.0], extents=assets[key]["extents"], mass=rec["mass"], mu=list(rec["mu"]), damping=list(rec["damp"]),
                         collider=rec["col"], dest=(f"{DEST[cls]}_01" if role == "task" else None), yaw=0.0))
    for cname in ("fruit_bowl", "toolbox", "recycling_box"):
        add(cname, f"{cname}_01", "container", "container")
    for c in ("fruit", "tool", "can"):
        for i, key in enumerate(types[c]):
            add(key, f"{c}_{i}_{key}", "task", c)
    for i, key in enumerate(obstacles):
        add(key, f"obstacle_{i}_{key}", "obstacle", "obstacle")
    if spare:                                                  # an extra object that is not on the table at the start (perturbation: obstacle appears)
        add("wood_block", "spare_wood_block", "spare", "obstacle")
    poses = reach_poses(table, sep)
    scene = dict(episode=name, video=None, keyframe=0, t_start=0.0, K=s0["K"], T_cam_to_table=s0["T_cam_to_table"], image_wh=s0["image_wh"],
                 frame=s0["frame"], table=table, objects=objs, poses={a: [list(p[0]), list(p[1])] for a, p in poses.items()},
                 cleanup=dict(split=split, seed=seed, types=types, obstacles=obstacles, table_source=src_ep,
                              instruction="clean up the table: put the fruit in the fruit bowl, the tools in the toolbox and the cans in the recycling box; leave everything else"),
                 notes=["cleanup scene (cleanup_scenes.py): positions are nominal, the runner randomises layouts per episode"])
    json.dump(scene, open(sdir / "scene.json", "w"), indent=1)
    return scene


# ------------------------------------------------------------------------------------------------ layouts
def sample_layouts(scene, n, rng, reach_b, min_gap=0.012, max_tries=400, pboth_c=0.0, pboth_i=0.0):
    """n random layouts: containers (reachable by an arm), each in-scope item where one arm reaches both it and its container, obstacles anywhere
    inside the reach region; random yaw (containers: 0/90 deg), footprint circles with a gap.  Returns xyz (n,K,3), yaw (n,K)."""
    objs = scene["objects"]; tb = scene["table"]; K = len(objs)
    ids = [o["id"] for o in objs]
    poses = {a: (np.array(p[0][:2]), p[1]) for a, p in scene["poses"].items()}
    base = {a: v[0] for a, v in poses.items()}
    ext = np.array([o["extents"] for o in objs])
    rad = 0.5 * np.hypot(ext[:, 0], ext[:, 1]) * 0.92
    order = [k for k, o in enumerate(objs) if o["role"] == "container"] + [k for k, o in enumerate(objs) if o["role"] == "task"] + \
            [k for k, o in enumerate(objs) if o["role"] == "obstacle"]
    xlo, xhi = tb["x"][0] + 0.06, tb["x"][1] - 0.06
    yhi = tb["y"][1] - 0.06
    ylo = tb["y"][1] - 0.85
    xyz = np.zeros((n, K, 3)); yaw = np.zeros((n, K)); fails = 0; failed = []
    for e in range(n):
        for attempt in range(30):
            placed, ok_all = {}, True
            for k in order:
                o = objs[k]
                ok = False
                for _ in range(max_tries):
                    xy = rng.uniform([xlo, ylo], [xhi, yhi])
                    ra = {a: bool(reach_b(base[a], xy)) for a in base}
                    both = (rng.random() < (pboth_c if o["role"] == "container" else pboth_i)) if (pboth_c > 0 or pboth_i > 0) and o["role"] != "obstacle" else False
                    if both and not all(ra.values()):
                        continue                                       # overlap layouts: put this object where both arms reach it
                    if o["role"] == "container":
                        # containers: reachable with a 7 cm margin in depth (the hand needs room to lift over the rim)
                        good = any(bool(reach_b(base[a], xy + np.array([0.0, 0.07]))) and bool(reach_b(base[a], xy - np.array([0.0, 0.07]))) for a in base)
                    elif o["role"] == "task":
                        c = placed[ids.index(o["dest"])][0]
                        good = any(ra[a] and bool(reach_b(base[a], c)) for a in base)
                    else:
                        good = any(ra.values())
                    if good and all(np.linalg.norm(xy - q) > rad[k] + rq + min_gap for q, rq in placed.values()):
                        ok = True; break
                if not ok:
                    ok_all = False; break
                placed[k] = (xy, rad[k])
            if ok_all:
                break
            fails += 1
        if not ok_all:
            failed.append(e)                                       # layout could not be completed: the episode is invalid (some object left at the origin)
        for k, (xy, _) in placed.items():
            xyz[e, k, :2] = xy
            xyz[e, k, 2] = 0.003 if objs[k]["role"] != "container" else 0.0
            yaw[e, k] = (rng.choice([0.0, np.pi / 2]) if objs[k]["role"] == "container" else rng.uniform(-np.pi, np.pi))
    return xyz, yaw, dict(layout_redraws=fails, failed_layouts=failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_assets", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="seen", choices=["seen", "heldout"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_class", type=int, default=1)
    ap.add_argument("--spare", action="store_true")
    ap.add_argument("--counts", default=None, help="fruit,tool,can counts e.g. 2,1,1")
    ap.add_argument("--sep", type=float, default=0.52)
    a = ap.parse_args()
    if a.build_assets:
        build_assets(a.only)
    if a.name:
        s = make_scene(a.name, a.split, a.seed, n_per_class=a.n_per_class, spare=a.spare, counts=[int(x) for x in a.counts.split(",")] if a.counts else None, sep=a.sep)
        print(json.dumps(s["cleanup"], indent=1))
