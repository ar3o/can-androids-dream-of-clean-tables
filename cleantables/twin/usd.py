"""USD writers for the twins: physics-ready rigid bodies, the textured table, the emissive background plate.

Stages are z-up, 1 unit = 1 m. Texture paths are stored relative to the USD file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh


def _new_stage(path: Path):
    from pxr import Usd, UsdGeom

    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root")
    st.SetDefaultPrim(root.GetPrim())
    return st, root.GetPrim()


def _textured_preview(st, mat_path: str, sh, tex_path: str, clamp: bool = True):
    """UsdUVTexture reading primvar st -> returns the texture shader."""
    from pxr import Sdf, UsdShade

    rd = UsdShade.Shader.Define(st, f"{mat_path}/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(st, f"{mat_path}/tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex_path))
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    if clamp:
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    return tx


def _emissive_mdl(st, tex_path: str, scale: float):
    """OmniPBR emission-only material: RTX ignores UsdPreviewSurface emission, so the background photo needs MDL."""
    from pxr import Gf, Sdf, UsdShade

    em = UsdShade.Material.Define(st, "/Root/Looks/emis")
    es = UsdShade.Shader.Define(st, "/Root/Looks/emis/Shader")
    es.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    es.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    es.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    es.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
    es.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
    es.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    es.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(tex_path)))
    es.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(scale))
    es.CreateOutput("out", Sdf.ValueTypeNames.Token)
    em.CreateSurfaceOutput("mdl").ConnectToSource(es.ConnectableAPI(), "out")
    em.CreateDisplacementOutput("mdl").ConnectToSource(es.ConnectableAPI(), "out")
    em.CreateVolumeOutput("mdl").ConnectToSource(es.ConnectableAPI(), "out")
    return em


def _bowl_collider(st, d: dict, wall: float = 0.006, n_seg: int = 32) -> None:
    """floor disc + n_seg tilted wall slabs following the bowl's cone."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cx, cy, r_bot, r_top, Hb = d["cx"], d["cy"], d["r_bot"], d["r_top"], d["H"]
    floor_z = max(0.006, 0.08 * Hb)
    cyl = UsdGeom.Cylinder.Define(st, "/Root/col_floor")
    cyl.CreateRadiusAttr(r_bot)
    cyl.CreateHeightAttr(floor_z)
    cyl.CreateAxisAttr("Z")
    cyl.AddTranslateOp().Set(Gf.Vec3d(cx, cy, floor_z / 2))
    cyl.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    alpha = math.atan2(r_top - r_bot, Hb)
    slant = math.hypot(r_top - r_bot, Hb)
    r_mid = (r_bot + r_top) / 2 - wall / 2 * math.cos(alpha)
    width = 2 * math.pi * ((r_bot + r_top) / 2) / n_seg * 1.15  # 15 % overlap closes the gaps between slabs
    for k in range(n_seg):
        phi = 2 * math.pi * k / n_seg
        c = UsdGeom.Cube.Define(st, f"/Root/col_wall_{k:02d}")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(cx + r_mid * math.cos(phi), cy + r_mid * math.sin(phi), Hb / 2))
        q = (
            Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(alpha)) * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(phi))
        ).GetQuat()
        c.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *map(float, q.GetImaginary())))
        c.AddScaleOp().Set(Gf.Vec3f(wall, width, slant))
        c.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())


def write_object_usd(
    path,
    mesh: trimesh.Trimesh,
    tex_path: str | None,
    category: str,
    mass: float,
    mu,
    collider: tuple,
    emissive: bool = False,
    extra_attrs: dict | None = None,
    emissive_scale: float = 1.0,
) -> None:
    """one rigid body: /Root (rigid body, mass), /Root/visual (textured mesh), colliders under /Root.

    collider: ("hull",) | ("parts", [trimesh, ...]) | ("bowl", bowl_profile dict) | ("none",) for a visual-only prop.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    path = Path(path)
    if path.exists():
        path.unlink()
    st, rp = _new_stage(path)
    rp.CreateAttribute("twin:category", Sdf.ValueTypeNames.String).Set(category)
    for k, v in (extra_attrs or {}).items():
        rp.CreateAttribute(f"twin:{k}", Sdf.ValueTypeNames.String).Set(str(v))
    if collider[0] != "none":
        UsdPhysics.RigidBodyAPI.Apply(rp)
        UsdPhysics.MassAPI.Apply(rp).CreateMassAttr(float(mass))
    mat = UsdShade.Material.Define(st, "/Root/Looks/mat")
    sh = UsdShade.Shader.Define(st, "/Root/Looks/mat/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
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
        tx = _textured_preview(st, "/Root/Looks/mat", sh, tex_path)
        if emissive:
            tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(emissive_scale, emissive_scale, emissive_scale, 1.0)
            )
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.7, 0.7))
    if emissive and tex_path is not None:
        mat = _emissive_mdl(st, tex_path, emissive_scale)
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
    elif kind == "bowl":
        _bowl_collider(st, collider[1])
    if kind != "none":
        pm = UsdShade.Material.Define(st, "/Root/Looks/physics")
        pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        pma.CreateStaticFrictionAttr(float(mu[0]))
        pma.CreateDynamicFrictionAttr(float(mu[1]))
        pma.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(rp).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    st.GetRootLayer().Save()


def write_table_usd(path, tex_path: str, extent, thick: float = 0.02) -> None:
    """visual table: textured top quad + plain wooden slab edge (the collider is a cuboid in the simulation scene)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    (x0, x1), (y0, y1) = extent
    st, _ = _new_stage(Path(path))
    mat = UsdShade.Material.Define(st, "/Root/Looks/wood")
    sh = UsdShade.Shader.Define(st, "/Root/Looks/wood/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    tx = _textured_preview(st, "/Root/Looks/wood", sh, tex_path, clamp=False)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    top = UsdGeom.Mesh.Define(st, "/Root/top")
    z = 0.0005
    top.CreatePointsAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)])
    top.CreateFaceVertexCountsAttr([4])
    top.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    top.CreateExtentAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y1, z)])
    pv = UsdGeom.PrimvarsAPI(top).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(mat)
    em = UsdShade.Material.Define(st, "/Root/Looks/edge")
    es = UsdShade.Shader.Define(st, "/Root/Looks/edge/pbr")
    es.CreateIdAttr("UsdPreviewSurface")
    es.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.78, 0.6, 0.4))
    em.CreateSurfaceOutput().ConnectToSource(es.ConnectableAPI(), "surface")
    slab = UsdGeom.Cube.Define(st, "/Root/slab")
    slab.CreateSizeAttr(1.0)
    slab.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, -thick / 2))
    slab.AddScaleOp().Set(Gf.Vec3f(x1 - x0, y1 - y0, thick - 0.001))
    UsdShade.MaterialBindingAPI.Apply(slab.GetPrim()).Bind(em)
    st.GetRootLayer().Save()
