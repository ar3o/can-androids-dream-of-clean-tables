"""Per-episode physics twin, fully automatic: tasks_train.json episode in -> data/twin_table/ep/<episode_id>/ out.

  source scripts/env.sh; $SIM_PY -m cleantables.twin.ep.build [--task tools] [--ids IMG_2533_t01 ...]      (CPU)

Per episode (video V, k0 = round(start_s x 10) in the 10 fps work frames):
  1. static background of V (cleantables.twin.ep.plate; built once per video): background.usda = the plate back-projected with its static MoGe depth
     (metric, det3d scale) into a textured triangle mesh in the table frame (visual only, emissive = the photo's colours, depth-discontinuity
     triangles dropped, table-top triangles dropped: the table is its own sim collider + textured quad). Rendered from the phone camera it
     reproduces the real (person-free) view; from other cameras it stretches behind occlusion edges.
  2. objects present at the start: every generic-pipeline track of V (data/generic/V) with a library identity (identity.py: phrase + colour +
     footprint) that is sighted in [k0, k0 + 1.5 s]; pose = median of its tracked 3D boxes over that window, then refined by fitting the
     library mesh's projected silhouette to the SAM 3 mask through the real camera (cuboids: 24 face-up choices x yaw, flat tools: yaw x flip;
     both pliers meshes are tried on both pliers tracks and assigned by fit = shape-based identity). Tracks not sighted in the window but
     present before and after (hidden by the person's body) use the nearest sighting and are flagged; so are uncertain identities, duplicate
     identities, weak fits (IoU < 0.6) and the toolbox when it was closed at the start (twin: open tray, lid omitted).
  3. USDs per object (mesh from the library, yaw / face-up rotation baked into the vertices so the rigid-body root carries only the
     position, as the loader of cleantables.twin.robot.env_kuka expects), textures referenced from the library; colliders: cuboid hull, open-tray 5 slabs,
     flat tools CoACD convex parts.
  4. table: textured quad (orthophoto of the static plate, 2 mm/px, holes filled) + 2 cm slab edge; collider = the env's TableCol cuboid.
  5. two Kuka-Allegro arms across the table from the phone (far +y edge, env_kuka convention: base 0.255 m below the top, 0.15 m behind
     the edge, facing the camera), x-centre / separation chosen to cover the tasks (object and destination in one arm's Lift workspace).
  6. cameras: main = the phone (K, T of det3d), overview (high, behind the phone), wrist cameras on each palm (see rtx.py).
Out: data/twin_table/ep/<id>/{scene.json, usd/, table.usda, table_tex.png, background.usda, start_real.jpg, layout_check.jpg}.
twin_render.jpg / side_by_side.jpg come from the Isaac RTX renderer.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from cleantables.twin.ep import geometry as geo
from cleantables.twin.ep import plate as PL
from cleantables.twin.ep.common import LIB, OUT, ROOT, Dets, H, W, episodes, events, frame, geom, jdump, project, rays_lo, table_extent, tracks
from cleantables.twin.ep.identity import LIBRARY, resolve_video
from cleantables.twin.ep.library import load_entry

EMIS = 600.0                     # emissive gain of the background plate (RTX exposure; calibrated against the real frame)
WIN = 15                        # 1.5 s at 10 fps
TABLE_DROP, BASE_GAP = 0.255, 0.15
REACH_XB, REACH_YB = (-0.80, -0.25), 0.34
DAMP = {"tool": (0.05, 0.2), "fruit": (0.05, 0.3), "box": (0.1, 0.1), "container": (0.1, 0.1), "obstacle": (0.1, 0.1)}


YAW_CAM = -math.pi / 2            # camera-side bases face +y (across the table, away from the phone), like the person's arms
Q_CAM = [0.0, 0.0, math.sin(YAW_CAM / 2), math.cos(YAW_CAM / 2)]       # xyzw


def phys_policy(category, lib):
    """Fixed for physics stability: boxes / the toolbox lid (category box/*) 3.0 kg, mu (0.9, 0.8); containers (toolbox tray, fruit
    bowl) 2.5 kg; tools / fruit keep their real masses. Real boxes are lighter (lib['mass'] is kept in scene.json as mass_real)."""
    if category.startswith("box/"):
        return 3.0, (0.9, 0.8)
    if category.startswith("container/"):
        return 2.5, tuple(lib["mu"])
    return lib["mass"], tuple(lib["mu"])


def reach_b(base_xy, xy, yaw=YAW_CAM):
    """Object xy inside the published Lift policy's workspace of an arm based at base_xy with yaw (base-frame box, as env_kuka.reach_b,
    which is the special case yaw = +90 deg): xb in [-0.80, -0.25] (forward = base -x), |yb| <= 0.34."""
    d = np.asarray(xy, float) - np.asarray(base_xy, float)
    c, s_ = math.cos(yaw), math.sin(yaw)
    xb = c * d[..., 0] + s_ * d[..., 1]
    yb = -s_ * d[..., 0] + c * d[..., 1]
    return (xb >= REACH_XB[0]) & (xb <= REACH_XB[1]) & (np.abs(yb) <= REACH_YB)


def shoulder_poses(v, table, objs, tasks, flags, cam_xyz=None):
    """Robot bases where the person's shoulders are: on the phone side of the table (near edge, -y), reaching across it (+y).
    Shoulder centre x = median x of the top 15 cm band of the body points (plate.person_cloud: everything nearer than the static background
    by > 10 cm that is not a tracked object, pooled over the video), restricted to the camera side of the table edge. Base y = near table
    edge - BASE_GAP, height = the Lift convention (table top 0.255 m above the base). Shoulder width is not measurable from behind (the body
    fills the frame edge): the separation 0.40-0.60 m (biacromial ~0.40 m and wider) and a +-10 cm shift of the centre are chosen to maximise
    the tasks one arm can do alone (object and destination in its Lift workspace), then sharing between the arms."""
    from cleantables.twin.ep import plate as PL
    pp = OUT / "_video" / v / "person_pts.npy"
    P = np.load(pp) if pp.exists() else PL.person_cloud(v)
    y0, y1 = table["y"]
    up_ = (P[:, 2] > 0.05) & (P[:, 2] < 1.2) if len(P) else np.zeros(0, bool)
    near = P[up_ & (P[:, 1] < y0 + 0.05)] if len(P) else P
    far = P[up_ & (P[:, 1] > y1 - 0.05)] if len(P) else P
    side = "camera" if len(near) >= len(far) else "far"            # which side of the table the person stands on (most body points)
    body = near if side == "camera" else far
    if len(body) >= 300:
        ztop = float(np.percentile(body[:, 2], 95))
        band = body[body[:, 2] > ztop - 0.15]
        xc_m = float(np.median(band[:, 0])); est = dict(x=xc_m, top_z=ztop, y=float(np.median(band[:, 1])), n_pts=int(len(body)))
    else:
        xc_m = 0.5 * (table["x"][0] + table["x"][1]); est = dict(x=xc_m, n_pts=int(len(body)), fallback="table centre (too few body points)")
        flags.append("shoulders: too few body points, table centre used")
    est.update(side=side, n_near=int(len(near)), n_far=int(len(far)))
    yaw = YAW_CAM if side == "camera" else -YAW_CAM
    Q = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    y_b = y0 - BASE_GAP if side == "camera" else y1 + BASE_GAP
    best, arg = -1e9, (xc_m, 0.48)
    clear = lambda xc, sp: 0.0 if (side != "camera" or cam_xyz is None) else \
        max(0.0, 0.28 - min(abs(xc + sp / 2 - cam_xyz[0]), abs(xc - sp / 2 - cam_xyz[0])))   # noqa: E731
    for xc in np.arange(xc_m - 0.10, xc_m + 0.1001, 0.02):
        for sp in np.arange(0.40, 0.641, 0.04):
            ok, na, nb = 0, 0, 0
            for k, kc in tasks:
                po, pd = objs[k]["pos"][:2], objs[kc]["pos"][:2]
                ra = bool(reach_b((xc + sp / 2, y_b), po, yaw) and reach_b((xc + sp / 2, y_b), pd, yaw))
                rb = bool(reach_b((xc - sp / 2, y_b), po, yaw) and reach_b((xc - sp / 2, y_b), pd, yaw))
                ok += ra or rb; na += ra; nb += rb
            # phone on the person's side: its lens is just behind the bases -> keep each base >= 28 cm sideways from it (else an elbow fills the view)
            score = ok + 0.3 * min(na, nb) - 0.5 * abs(xc - xc_m) - 8.0 * clear(xc, sp)
            if score > best:
                best, arg = score, (float(xc), float(sp))
    xc, sp = arg
    poses = {"a": [[xc + sp / 2, y_b, -TABLE_DROP], Q], "b": [[xc - sp / 2, y_b, -TABLE_DROP], Q]}
    est.update(chosen_centre_x=xc, separation=sp, base_y=y_b, yaw_deg=round(math.degrees(yaw), 1))
    return poses, est


def up(m):
    return cv2.resize(m.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)


# ------------------------------------------------------------------ objects at the start
def start_objects(ep, flags):
    v = ep["video"]
    k0 = int(round(ep["start_s"] * 10))
    E, R = events(v), tracks(v)
    res = {r["track"]: r for r in resolve_video(E["objects"], v)}
    out = []
    for o in E["objects"]:
        r = res[o["id"]]
        if r["key"] is None:
            continue
        T_ = R["objects"][o["src"]]
        ks = T_["sight_k"]
        sel = [i for i, k in enumerate(ks.tolist()) if k0 <= k <= k0 + WIN and R["det_box"].get(T_["det"][i])]
        note = None
        if not sel:
            if not (o["first"] <= k0 + WIN and o["last"] >= k0):
                continue                                              # not on the table at the start
            near = sorted(range(len(ks)), key=lambda i: abs(int(ks[i]) - k0))
            near = [i for i in near if R["det_box"].get(T_["det"][i]) and abs(int(ks[i]) - k0) <= 50][:3]
            if not near:
                flags.append(f"{r['key']} (track {o['id']}): present but not sighted within 5 s of the start -> omitted")
                continue
            sel = near
            note = f"hidden at the start (the person's body / hand): pose from the nearest sightings k={[int(ks[i]) for i in near]}"
            flags.append(f"{r['key']} (track {o['id']}): {note}")
        B = [R["det_box"][T_["det"][i]] for i in sel]
        hf = float(np.median([b.get("hand_frac", 0.0) for b in B]))
        resting = any(q["f0"] <= k0 + WIN and q["f1"] >= k0 for q in o["rests"])
        if note is None and not resting and hf > 0.02:
            # carried at the start: containers are being set down for the task -> their next resting pose; anything else -> where it lay
            # before it was picked up (previous rest), else the next one. Sightings of that rest replace the window.
            cont = r["key"] in ("toolbox", "white_box", "fruit_bowl")
            prev = [q for q in o["rests"] if q["f1"] < k0]
            nxt = [q for q in o["rests"] if q["f0"] > k0]
            if cont:
                q = nxt[0] if nxt else (prev[-1] if prev else None)
            else:                                           # just picked up (rest ended <= 3 s before k0): where it lay; else not on the table
                q = prev[-1] if (prev and k0 - prev[-1]["f1"] <= 30) else None
            if q is None:
                flags.append(f"{r['key']} (track {o['id']}): carried at the start, no resting pose -> omitted")
                continue
            sel2 = [i for i, k in enumerate(ks.tolist()) if q["f0"] <= k <= q["f1"] and R["det_box"].get(T_["det"][i])]
            if not sel2:
                flags.append(f"{r['key']} (track {o['id']}): carried at the start, resting pose not sighted -> omitted")
                continue
            sel = sel2[:15] if q in nxt else sel2[-15:]
            B = [R["det_box"][T_["det"][i]] for i in sel]
            note = f"carried at the start (hand overlap {hf:.2f}): placed at its {'next' if q in nxt else 'previous'} resting pose (k {q['f0']}-{q['f1']})"
            flags.append(f"{r['key']} (track {o['id']}): {note}")
        xyz = np.median([[b["x"], b["y"], b["top"]] for b in B], 0)
        # mask for the fit: least hand-covered, largest sighting in the window
        i_best = max(sel, key=lambda i: (-(R["det_box"][T_["det"][i]].get("hand_frac", 0) > 0.02), R["det_box"][T_["det"][i]]["n"]))
        b = R["det_box"][T_["det"][i_best]]
        out.append(dict(track=o["id"], key=r["key"], why=r["why"], uncertain=r["uncertain"], x=float(xyz[0]), y=float(xyz[1]), top=float(xyz[2]),
                        yaw=float(b["yaw"]), L=float(b["L"]), W=float(b["W"]), k_mask=int(ks[i_best]), det=int(T_["det"][i_best]),
                        n_sight=len(sel), note=note, hand_frac=float(b.get("hand_frac", 0))))
        if r["uncertain"]:
            flags.append(f"{r['key']} (track {o['id']}): uncertain identity ({r['why']})")
        if len(sel) < 3 and note is None:
            flags.append(f"{r['key']} (track {o['id']}): only {len(sel)} sightings in the first 1.5 s")
    # toolbox: an open tray if present; a white box = the tray (closed at the start) if no open tray, else the lid
    trays = [o for o in out if o["key"] == "toolbox"]
    whites = [o for o in out if o["key"] == "white_box"]
    for w in whites:
        if trays and min(math.hypot(w["x"] - t["x"], w["y"] - t["y"]) for t in trays) < 0.15:
            w["key"] = None                                  # the lid lies on / under the tray (or is the tray's own top): not a separate object
            flags.append(f"white box track {w['track']} overlaps the open tray -> lid on / under the tray, omitted")
        elif trays:
            w["key"] = "toolbox_lid"
        else:
            w["key"] = "toolbox"; trays.append(w)
            flags.append("toolbox closed at the start (lid on): the twin shows the open tray, lid omitted")
    out = [o for o in out if o["key"] is not None]
    # duplicates of one physical object (fruit can repeat: several bananas / apples are different physical objects of one type)
    multi = {"banana", "green_apple"}
    keep = []
    for o in sorted(out, key=lambda o: -o["n_sight"]):
        if o["key"] not in multi and o["key"] != "pliers" and any(k["key"] == o["key"] for k in keep):
            flags.append(f"{o['key']}: second track {o['track']} at the start (duplicate identity) -> dropped")
            continue
        keep.append(o)
    return keep, k0


# ------------------------------------------------------------------ fitting
def care_mask(dets, R, k, own_det, hand):
    """Pixels where the fit counts: everything except other objects' masks and the hand (they may occlude this object)."""
    other = np.zeros((H, W), bool)
    for d in R["kept"].get(k, []):
        if d != own_det:
            other |= dets.mask(d)
    if hand is not None:
        other |= hand
    return ~up(cv2.dilate(other.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))


def rot_dims(dims, Rm):
    return np.abs(Rm @ np.asarray(dims))


def fit_cuboid_obj(o, lib_mesh, meta, mask, care, K, T, img, scale=None):
    """Choose the face-up rotation (24) whose (L, W, H) best matches the measured footprint / height, try yaw / yaw+180 (appearance), fit."""
    dims = np.array(meta["extents"]) * (scale or 1.0)
    lib_mesh = geo.scaled(lib_mesh, scale or 1.0, flat=False)
    meas = np.array([max(o["L"], o["W"]), min(o["L"], o["W"]), max(o["top"], 0.01)])
    cands = []
    for Rm in geo.cube_rotations():
        d = rot_dims(dims, Rm)
        if d[0] + 1e-6 < d[1]:
            continue                                                     # keep L >= W in the rotated frame (yaw covers the rest)
        cost = float(np.abs(np.log(d[:2] / np.maximum(meas[:2], 0.02))).sum())
        if o["top"] - d[2] > 0.03:                                       # resting on something: the height cannot be compared
            pass
        else:
            cost += float(abs(math.log(d[2] / meas[2])))
        cands.append((cost, Rm, d))
    cands.sort(key=lambda c: c[0])
    best = None
    cmin = cands[0][0]
    for cost, Rm, d in [c for c in cands if c[0] <= cmin + 0.15][:8]:
        z0 = max(0.0, o["top"] - d[2]) if o["top"] - d[2] > 0.03 else 0.0
        yaw0 = o["yaw"] if o["L"] >= o["W"] else o["yaw"] + math.pi / 2
        # final fit: the video scale times a per-object factor in [0.9, 1.1] (box sizes from another video's MoGe depth differ by a few %)
        f = geo.fit_cuboid(mask, K, T, (o["x"], o["y"], yaw0, d[0], d[1], d[2]), fix_dims=tuple(d), z0=z0, care=care, free_scale=True,
                           scale_bounds=(0.7, 1.4) if scale is None else (0.9, 1.1))
        if scale is None:
            return f, None                                               # scale probe: best rotation only
        for dy in (0.0, math.pi):
            m = geo.scaled(lib_mesh, f["scale"], flat=False)
            M = np.eye(4); M[:3, :3] = Rm
            m.apply_transform(M)
            lo, hi = m.bounds
            m.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
            yaw = f["yaw"] + dy
            app = appearance(m, meta, yaw, f["x"], f["y"], z0, K, T, img, mask & care)
            score = f["iou"] - 0.002 * app - 0.05 * cost
            if best is None or score > best[0]:
                best = (score, dict(f, yaw=yaw, z0=z0, R=Rm.tolist(), dims=(d * f["scale"]).tolist(), app_err=app, dim_cost=cost,
                                    obj_scale=f["scale"]), m)
    return best[1], best[2]


def appearance(m, meta, yaw, x, y, z0, K, T, img, mask):
    """Mean abs colour error (0-255) between the textured mesh's top-facing triangles projected into the frame and the real pixels."""
    tex = cv2.imread(str(LIB / meta["key"] / "texture.png"))
    c, s = math.cos(yaw), math.sin(yaw)
    V = m.vertices @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + np.array([x, y, z0])
    uv_img, _ = project(V, K, T)
    UV = m.visual.uv
    errs = []
    nz = m.face_normals @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    for f in np.nonzero(nz[:, 2] > 0.7)[0][:40]:
        tri = uv_img[m.faces[f]]
        cen = tri.mean(0)
        for w in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
            p = (np.array(w)[:, None] * tri).sum(0)
            if not (0 <= p[0] < 959 and 0 <= p[1] < 539) or not mask[int(p[1]), int(p[0])]:
                continue
            t = (np.array(w)[:, None] * UV[m.faces[f]]).sum(0)
            tc = tex[int((1 - t[1]) * (tex.shape[0] - 1)), int(t[0] * (tex.shape[1] - 1))].astype(float)
            errs.append(np.abs(tc - img[int(p[1]), int(p[0])].astype(float)).mean())
        del cen
    return float(np.mean(errs)) if errs else 60.0


# ------------------------------------------------------------------ table + background
def table_texture(v, K, T, ext, out):
    pl = cv2.imread(str(OUT / "_video" / v / "plate.png"))
    valid = cv2.imread(str(OUT / "_video" / v / "plate_valid.png"), 0) > 127
    (x0, x1), (y0, y1) = ext
    res = 0.002
    orth, (mx, my) = geo.ortho(pl, K, T, x0, x1, y0, y1, z=0.0, res=res)
    inview = (mx >= 0) & (mx < 959) & (my >= 0) & (my < 539)
    vv = cv2.remap(valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & inview
    integral = cv2.integral(vv.astype(np.uint8))
    ps = 120
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(orth, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    best, bp = -1e9, (0, 0)
    for r in range(0, vv.shape[0] - ps, 10):
        for c in range(0, vv.shape[1] - ps, 10):
            s = integral[r + ps, c + ps] - integral[r, c + ps] - integral[r + ps, c] + integral[r, c]
            if s < 0.98 * ps * ps:
                continue
            sc = -lap[r:r + ps, c:c + ps].mean()
            if sc > best:
                best, bp = sc, (r, c)
    patch = orth[bp[0]:bp[0] + ps, bp[1]:bp[1] + ps]
    patch = np.concatenate([patch, patch[:, ::-1]], 1); patch = np.concatenate([patch, patch[::-1]], 0)
    tiled = np.tile(patch, (vv.shape[0] // patch.shape[0] + 1, vv.shape[1] // patch.shape[1] + 1, 1))[:vv.shape[0], :vv.shape[1]]
    hole = (~vv).astype(np.uint8)
    big = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8)).astype(bool)
    base = np.where(big[..., None], tiled, orth)
    small = (hole.astype(bool) & ~big).astype(np.uint8) * 255
    tex = cv2.inpaint(base, small, 5, cv2.INPAINT_TELEA)
    tex = np.where(vv[..., None], orth, tex)
    cv2.imwrite(str(out / "table_tex.png"), tex)
    return float(vv.mean())


def write_table_usd(path, tex_abs, ext, thick=0.02):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    (x0, x1), (y0, y1) = ext
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root"); st.SetDefaultPrim(root.GetPrim())
    mat = UsdShade.Material.Define(st, "/Root/Looks/wood")
    sh = UsdShade.Shader.Define(st, "/Root/Looks/wood/pbr"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    rd = UsdShade.Shader.Define(st, "/Root/Looks/wood/st"); rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(st, "/Root/Looks/wood/tex"); tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex_abs))
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    top = UsdGeom.Mesh.Define(st, "/Root/top")
    z = 0.0005
    top.CreatePointsAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)])
    top.CreateFaceVertexCountsAttr([4]); top.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    top.CreateExtentAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y1, z)])
    pv = UsdGeom.PrimvarsAPI(top).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(mat)
    em = UsdShade.Material.Define(st, "/Root/Looks/edge")
    es = UsdShade.Shader.Define(st, "/Root/Looks/edge/pbr"); es.CreateIdAttr("UsdPreviewSurface")
    es.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.78, 0.6, 0.4))
    em.CreateSurfaceOutput().ConnectToSource(es.ConnectableAPI(), "surface")
    slab = UsdGeom.Cube.Define(st, "/Root/slab"); slab.CreateSizeAttr(1.0)
    slab.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, -thick / 2))
    slab.AddScaleOp().Set(Gf.Vec3f(x1 - x0, y1 - y0, thick - 0.001))
    UsdShade.MaterialBindingAPI.Apply(slab.GetPrim()).Bind(em)
    st.GetRootLayer().Save()


def background_usd(v, K, T, ext, path, step=1, rel_jump=0.04, emissive_scale=None):
    """Static plate + static depth -> textured triangle mesh in the table frame (emissive, visual only)."""
    D = np.load(OUT / "_video" / v / "plate_depth.npy")
    rays = rays_lo(K)
    P = (rays * D[..., None]).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    P = P.reshape(H, W, 3)
    (x0, x1), (y0, y1) = ext
    on_top = (np.abs(P[..., 2]) < 0.02) & (P[..., 0] > x0 + 0.01) & (P[..., 0] < x1 - 0.01) & (P[..., 1] > y0 + 0.01) & (P[..., 1] < y1 - 0.01)
    ys, xs = np.mgrid[0:H - 1:step, 0:W - 1:step]
    idx = lambda r, c: r * W + c   # noqa: E731
    F = []
    for (a, b, c) in (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))):
        r0, c0 = ys + a[0] * step, xs + a[1] * step
        r1, c1 = ys + b[0] * step, xs + b[1] * step
        r2, c2 = ys + c[0] * step, xs + c[1] * step
        d = np.stack([D[r0, c0], D[r1, c1], D[r2, c2]], -1)
        ok = (d.max(-1) - d.min(-1)) < rel_jump * d.min(-1)
        ok &= ~(on_top[r0, c0] & on_top[r1, c1] & on_top[r2, c2])
        ok &= np.isfinite(d).all(-1) & (d.min(-1) > 0.2)
        tri = np.stack([idx(r0, c0), idx(r1, c1), idx(r2, c2)], -1)[ok]      # counter-clockwise as seen from the phone (front face)
        F.append(tri)
    F = np.concatenate(F)
    used = np.unique(F)
    remap = -np.ones(H * W, np.int64); remap[used] = np.arange(len(used))
    V = P.reshape(-1, 3)[used]
    vv, uu = np.divmod(used, W)
    UV = np.c_[(uu * 2 + 1) / 960.0, 1 - (vv * 2 + 1) / 540.0]
    # far backdrop: the whole plate on a quad 8 m in front of the phone (fills the dropped discontinuity slivers seamlessly from the phone)
    zf = 8.0
    cr = np.array([[0, 0], [960, 0], [960, 540], [0, 540]], float)
    Pc = np.c_[(cr[:, 0] - K[0, 2]) / K[0, 0] * zf, (cr[:, 1] - K[1, 2]) / K[1, 1] * zf, np.full(4, zf)]
    Pb = Pc @ T[:3, :3].T + T[:3, 3]
    nb = len(V)
    V = np.r_[V, Pb]
    UV = np.r_[UV, [[0, 1], [1, 1], [1, 0], [0, 0]]]
    Fall = np.r_[remap[F], [[nb, nb + 3, nb + 2], [nb, nb + 2, nb + 1]]]
    m = trimesh.Trimesh(V, Fall, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=UV)
    geo.write_usd(path, m, str(OUT / "_video" / v / "plate.png"), "background/static_plate", 0, (0, 0), (0, 0), ("none",), emissive=True,
                  emissive_scale=EMIS if emissive_scale is None else emissive_scale)
    return len(F)


# ------------------------------------------------------------------ main
# Variant switches (environment); unset = default behaviour:
#   TWIN_SCALE_MODE=per_object  no shared video factor: every object gets its own free-scale fit (the same fit and bounds used for the
#                               video-scale probes), i.e. s_vid = 1 and per-object bounds (0.7, 1.4) instead of (0.9, 1.1)
#   TWIN_OUT=<dir under data/>  write the episode twin there instead of data/twin_table/ep/<id> (existing twins are left as they are)
SCALE_MODE = os.environ.get("TWIN_SCALE_MODE", "video")


def build(ep, force=False):
    t0 = time.time()
    eid, v = ep["id"], ep["video"]
    out = (Path(os.environ["TWIN_OUT"]) if os.environ.get("TWIN_OUT") else OUT) / eid
    if (out / "scene.json").exists() and not force:
        return None
    if out.exists():
        shutil.rmtree(out / "usd", ignore_errors=True)
    (out / "usd").mkdir(parents=True, exist_ok=True)
    PL.build(v)
    G = geom(v); K, T = G["K"], G["T"]
    flags = []
    objs0, k0 = start_objects(ep, flags)
    img0 = frame(v, k0)
    cv2.imwrite(str(out / "start_real.jpg"), img0)
    dets, R = Dets(v), tracks(v)
    from cleantables.twin.ep.common import hand_masks
    hm = hand_masks(v)
    placed = []
    # ---- per-object masks / care regions (other objects, hand and the person's body = pixels nearer than the static background)
    pd = np.load(OUT / "_video" / v / "plate_depth.npy")
    from cleantables.twin.ep.common import depth
    D10 = depth(v)
    ctx = {}
    for o in objs0:
        k = o["k_mask"]
        allm = np.zeros((H, W), bool)
        for d in R["kept"].get(k, []):
            allm |= dets.mask(d)
        dk = np.asarray(D10[k], np.float32) * G["scale"][k]
        body = (dk < pd - 0.08) & ~cv2.dilate(allm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        body = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        care = care_mask(dets, R, k, o["det"], hm.get(k)) & ~up(body)
        ctx[o["track"]] = (up(dets.mask(o["det"])), care, frame(v, k))

    def keys_of(o):
        return {"pliers": ["pliers_a", "pliers_b"], "banana": ["banana_a", "banana_b"]}.get(o["key"], [o["key"]])

    def fit_one(o, key, scale):
        mask, care, img = ctx[o["track"]]
        mesh, tex, meta = load_entry(key)
        kind = LIBRARY[key]["kind"]
        if kind in ("flat", "pillow", "hy3d", "apple"):
            flat_ = kind in ("flat", "pillow")
            f = geo.fit_flat(mesh if scale is None else geo.scaled(mesh, scale, flat_), mask, K, T, o["x"], o["y"], o["yaw"], care=care,
                             free_scale=True, scale_bounds=(0.7, 1.4) if (scale is None or SCALE_MODE == "per_object") else (0.9, 1.1),
                             flips=(False, True) if flat_ else (False,))
            if scale is None:
                return f, None, meta
            f["obj_scale"] = f["scale"]
            m = geo.scaled(mesh, scale * f["obj_scale"], flat_)
            if f["flipped"]:
                m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0)))
                m.apply_translation((0, 0, -m.bounds[0][2]))
            f["z0"] = max(0.0, o["top"] - m.extents[2]) if o["top"] - m.extents[2] > 0.02 else 0.0
        else:
            f, m = fit_cuboid_obj(o, mesh, meta, mask, care, K, T, img, scale)
        if scale is not None:
            f["scale"] = scale * f.get("obj_scale", 1.0)
        return f, m, meta
    # ---- video metric scale: library sizes come from other videos whose MoGe depth scale differs by a few %; one isotropic factor per
    # video (median of free-scale fits of the well-seen, unoccluded objects) keeps every physical object's mesh identical up to that factor
    probes = []
    for o in objs0:
        mask, care, _ = ctx[o["track"]]
        if o["key"] == "pliers" or (mask & ~care).sum() > 0.05 * mask.sum() or o["note"]:
            continue
        key = keys_of(o)[0]
        if key not in LIBRARY or not (LIB / key / "mesh.npz").exists():
            continue
        f, _, _ = fit_one(o, key, None)
        if f["iou"] >= 0.75:
            probes.append((key, f["scale"], f["iou"]))
    s_vid = float(np.median([p_[1] for p_ in probes])) if len(probes) >= 2 else 1.0
    if SCALE_MODE == "per_object":
        s_vid = 1.0
    if not probes or len(probes) < 2:
        flags.append("video scale: < 2 clean objects to estimate it, 1.0 used")
    fits = {}
    for o in objs0:
        for key in keys_of(o):
            if key not in LIBRARY or not (LIB / key / "mesh.npz").exists():
                flags.append(f"{key}: no library mesh -> omitted")
                continue
            fits[(o["track"], key)] = fit_one(o, key, s_vid)
    pl = [o for o in objs0 if o["key"] == "pliers"]
    used = set()
    if len(pl) == 2:
        s1 = fits[(pl[0]["track"], "pliers_a")][0]["iou"] + fits[(pl[1]["track"], "pliers_b")][0]["iou"]
        s2 = fits[(pl[0]["track"], "pliers_b")][0]["iou"] + fits[(pl[1]["track"], "pliers_a")][0]["iou"]
        assign = {pl[0]["track"]: "pliers_a" if s1 >= s2 else "pliers_b", pl[1]["track"]: "pliers_b" if s1 >= s2 else "pliers_a"}
        if abs(s1 - s2) < 0.05:
            flags.append(f"pliers identity by shape is weak (IoU sums {s1:.2f} vs {s2:.2f})")
    elif len(pl) == 1:
        o = pl[0]
        a, b = fits[(o["track"], "pliers_a")][0]["iou"], fits[(o["track"], "pliers_b")][0]["iou"]
        assign = {o["track"]: "pliers_a" if a >= b else "pliers_b"}
        if abs(a - b) < 0.05:
            flags.append(f"single pliers: identity by shape weak ({a:.2f} vs {b:.2f})")
    else:
        assign = {}
        if len(pl) > 2:
            flags.append(f"{len(pl)} pliers tracks at the start (expected <= 2): extra ones dropped")
    inst = {}
    for o in objs0:
        if o["key"] == "pliers":
            key = assign.get(o["track"])
        elif o["key"] == "banana":                 # bananas are not individually identifiable: the better-fitting of the two banana meshes
            cand_ = [(fits[(o["track"], kk)][0]["iou"], kk) for kk in ("banana_a", "banana_b") if (o["track"], kk) in fits]
            key = max(cand_)[1] if cand_ else None
        else:
            key = o["key"]
        if key is None or (o["track"], key) not in fits:
            continue
        if key in used and key not in ("banana_a", "banana_b", "green_apple"):
            continue
        used.add(key)
        f, m, meta = fits[(o["track"], key)]
        if f["iou"] < 0.3 and LIBRARY[key]["kind"] in ("flat", "pillow", "hy3d", "apple"):
            # no silhouette fit (mask occluded / merged): keep the mesh, use the tracked 3D box pose (median over the window)
            xy_ = np.asarray(m.vertices)[:, :2] - np.asarray(m.vertices)[:, :2].mean(0)
            w_, V_ = np.linalg.eigh(np.cov(xy_.T)); ax_ = V_[:, np.argmax(w_)]
            f = dict(f, x=o["x"], y=o["y"], yaw=o["yaw"] - math.atan2(ax_[1], ax_[0]), fallback="tracked 3D box pose (fit IoU %.2f)" % f["iou"])
            m = m.copy(); c_ = m.bounds.mean(0); m.apply_translation((-c_[0], -c_[1], 0))
            flags.append(f"{key}: silhouette fit failed (IoU {fits[(o['track'], key)][0]['iou']:.2f}) -> tracked 3D box pose used")
        elif f["iou"] < 0.6:
            flags.append(f"{key}: weak silhouette fit IoU {f['iou']:.2f} (occluded or wrong size)")
        inst[key] = inst.get(key, 0) + 1
        placed.append(dict(o=o, key=key, fit=f, mesh=m, meta=meta,
                           id=f"{key}_{inst[key]:02d}" if key in ("banana_a", "banana_b", "green_apple") else key))
    # ---- USDs
    scene_objs = []
    lib_rel = os.path.relpath(LIB, out / "usd")           # "../../library" for the default output folder
    for p in placed:
        key, f, m, meta = p["key"], p["fit"], p["mesh"].copy(), p["meta"]
        L_ = LIBRARY[key]
        c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
        Rz = np.eye(4); Rz[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        m.apply_transform(Rz)
        kind = L_["kind"]
        if kind == "tray":
            d = f["dims"]
            col = ("boxes", [(sz, tuple(np.array(Rz[:3, :3]) @ np.array(pos))) for sz, pos in geo.tray_colliders(d[0], d[1], d[2])])
            # slabs must follow the yaw: rotate each slab as its own oriented box -> emit as convex mesh parts instead
            parts = []
            for sz, pos in geo.tray_colliders(d[0], d[1], d[2]):
                bx = trimesh.creation.box(sz, trimesh.transformations.translation_matrix(pos))
                bx.apply_transform(Rz); parts.append(bx)
            col = ("parts", parts)
        elif kind == "cuboid" or kind == "apple":
            col = ("hull",)
        elif key == "fruit_bowl":
            col = ("bowl", geo.bowl_profile(np.asarray(m.vertices)))
        else:
            cp = LIB / key / "coacd.npz"
            if not cp.exists():
                parts0 = geo.convex_parts(load_entry(key)[0])
                np.savez(cp, **{f"v{i}": q.vertices for i, q in enumerate(parts0)}, **{f"f{i}": q.faces for i, q in enumerate(parts0)}, n=len(parts0))
            z = np.load(cp)
            sc_ = float(f.get("scale") or 1.0)               # the video metric scale applies to the collision parts too (x / y, as the visual)
            zs_ = 1.0 if kind in ("flat", "pillow") else sc_
            parts = [trimesh.Trimesh(z[f"v{i}"] * np.array([sc_, sc_, zs_]), z[f"f{i}"]) for i in range(int(z["n"]))]
            if f.get("flipped"):
                for q in parts:
                    q.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0)))
                lo = min(q.bounds[0][2] for q in parts)
                for q in parts:
                    q.apply_translation((0, 0, -lo))
            for q in parts:
                q.apply_transform(Rz)
            col = ("parts", parts)
        usd = out / "usd" / f"{p['id']}.usda"
        cat_ = "box/toolbox" if (ep["task"] == "stack_boxes" and key == "toolbox") else L_["category"]   # stacking: the toolbox is a movable box
        mass_, mu_ = phys_policy(cat_, L_)
        geo.write_usd(usd, m, f"{lib_rel}/{key}/texture.png", cat_, mass_, mu_, DAMP[L_["cls"]], col,
                      extra_attrs=dict(library_key=key))
        m.export(out / "usd" / f"{p['id']}.obj")
        cls = L_["cls"]
        if ep["task"] == "tools":
            role = "task" if cls == "tool" else ("container" if key == "toolbox" else "obstacle")
        elif ep["task"] == "fruit":
            role = "task" if cls == "fruit" else ("container" if key == "fruit_bowl" else "obstacle")
        elif ep["task"] == "stack_boxes":
            role = "stack" if (cls == "box" or key == "toolbox") else "obstacle"
        else:   # any other task (e.g. a mixed table): no task-specific rule, the library's destination decides
            role = "task" if L_.get("dest") else ("container" if cls == "container" else "obstacle")
        cls_env = cls if cls in ("tool", "fruit", "container") else "obstacle"
        dest = L_.get("dest") if role == "task" else None
        scene_objs.append(dict(id=p["id"], key=key, category=cat_, cls=cls_env, role=role, split="-", usd=str(usd.relative_to(ROOT)),
                               pos=[f["x"], f["y"], float(f.get("z0", 0.0))], extents=[round(float(x), 4) for x in m.extents], yaw=0.0,
                               yaw_baked=round(float(f["yaw"]), 4), mass=mass_, mu=list(mu_), mass_real=L_["mass"], damping=list(DAMP[L_["cls"]]),
                               collider=col[0], dest=dest, library=key, track=p["o"]["track"], fit_iou=round(float(f["iou"]), 3),
                               n_sightings_start=p["o"]["n_sight"], flipped=bool(f.get("flipped", False)), note=p["o"]["note"]))
    depenetrate(scene_objs, out, flags)
    drop(scene_objs, out, flags)
    ids = {o["id"] for o in scene_objs}
    for o in scene_objs:
        if o["dest"] is not None and o["dest"] not in ids:
            flags.append(f"{o['id']}: destination {o['dest']} not on the table at the start -> role set to obstacle")
            o["role"], o["dest"] = "obstacle", None
    # ---- table, background, arms
    ext = table_extent(T, K, G["table"])
    ox = np.array([o["pos"][:2] for o in scene_objs]) if scene_objs else np.zeros((1, 2))
    ext = [[min(ext[0][0], ox[:, 0].min() - 0.1), max(ext[0][1], ox[:, 0].max() + 0.1)], [min(ext[1][0], ox[:, 1].min() - 0.1), max(ext[1][1], ox[:, 1].max() + 0.1)]]
    vfrac = table_texture(v, K, T, ext, out)
    write_table_usd(out / "table.usda", (out / "table_tex.png").resolve(), ext)
    nbg = background_usd(v, K, T, ext, out / "background.usda")
    table = dict(x=ext[0], y=ext[1], top_z=0.0, texture="table_tex.png", tex_res_m=0.002, tex_source="static plate (ep/plate.py)",
                 tex_observed_frac=round(vfrac, 3))
    idx = {o["id"]: i for i, o in enumerate(scene_objs)}
    tasks = [(idx[o["id"]], idx[o["dest"]]) for o in scene_objs if o["role"] == "task"]
    if ep["task"] == "stack_boxes":
        st = [idx[o["id"]] for o in scene_objs if o["role"] == "stack"]
        tasks = [(a, b) for a in st for b in st if a != b]
    poses, shoulder = shoulder_poses(v, table, scene_objs, tasks, flags, cam_xyz=T[:3, 3])
    n_reach = sum(1 for k, kc in tasks if any(reach_b(poses[a][0][:2], scene_objs[k]["pos"][:2]) and reach_b(poses[a][0][:2], scene_objs[kc]["pos"][:2]) for a in "ab"))
    yaw_ = math.radians(shoulder["yaw_deg"])
    n_reach = sum(1 for k, kc in tasks if any(reach_b(poses[a][0][:2], scene_objs[k]["pos"][:2], yaw_) and reach_b(poses[a][0][:2], scene_objs[kc]["pos"][:2], yaw_) for a in "ab"))
    reach = {o["id"]: {a: bool(reach_b(poses[a][0][:2], o["pos"][:2], yaw_)) for a in "ab"} for o in scene_objs}
    for o in scene_objs:
        if o["role"] in ("task", "container", "stack") and not any(reach[o["id"]].values()):
            flags.append(f"{o['id']}: outside both arms' Lift workspace")
    if tasks and n_reach < len(tasks):
        flags.append(f"arms: {n_reach}/{len(tasks)} tasks have object AND destination inside one arm's Lift workspace")
    cam = T[:3, 3]
    overview = overview_pose(table)
    scene = dict(episode=f"ep/{eid}", video=v, keyframe=k0, t_start=ep["start_s"], t_end=ep["end_s"], task=ep["task"], instruction=ep["instruction"],
                 labelled_objects=ep.get("objects"), K=K.tolist(), T_cam_to_table=T.tolist(), image_wh=[960, 540],
                 frame="generic table frame (det3d): z up (table normal), table top z = 0, metres",
                 table=table, objects=scene_objs, poses=poses, layout="measured",
                 arms=dict(side=shoulder["side"], yaw_deg=shoulder["yaw_deg"], shoulder_estimate=shoulder, reach=reach,
                           tasks_one_arm=f"{n_reach}/{len(tasks)}",
                           note="bases where the person's shoulders are (their side of the table, from their body points), reaching across the table; Lift height "
                                "convention. "
                                "robot/env.py reach_b / joint_env._in_reach / cleanup_scenes.sample_layouts assume yaw +90 deg (far side): correct only for "
                                "side='far'; for side='camera' use build.reach_b(base, xy, yaw)"),
                 background=dict(usd="background.usda", source=f"data/twin_table/ep/_video/{v}", triangles=int(nbg),
                                 note="visual only, emissive static plate on its static MoGe depth; stretches behind objects from non-phone cameras"),
                 cameras=dict(main=dict(K=K.tolist(), T_cam_to_table=T.tolist(), wh=[960, 540], note="the real phone camera"),
                              overview=overview,
                              wrist=dict(mount="ee_link/palm_link", note="see render.py WRIST (pose in the palm frame, verified by rendering)")),
                 flags=flags + ["physics policy (deliberate, PI): boxes / lid 3.0 kg mu 0.9/0.8, containers 2.5 kg; tools / fruit real masses "
                                "(real boxes are lighter, mass_real per object)"], build_s=None, video_scale=dict(value=round(s_vid, 4), probes=[(a, round(b, 3), round(c, 3)) for a, b, c in probes],
                 note="isotropic factor on library meshes (flat objects: x/y only) for this video's MoGe metric scale"),
                 notes=["object poses: tracked 3D boxes (MoGe metric depth, not tape-measured) refined by silhouette fit through the phone camera",
                        "yaw (and the face-up rotation of boxes) is baked into each USD's vertices; pos = footprint centre, bottom height",
                        "joint_env.py loads this file unchanged (its runner re-samples layouts); a measured-layout loader uses pos as is (layout='measured')",
                        "flat tools: real top-down silhouette + texture, thickness = category prior; boxes: textured cuboids from the video",
                        "camera height %.2f m above the table" % cam[2]])
    scene["build_s"] = round(time.time() - t0, 1)
    jdump(scene, out / "scene.json")
    layout_check(scene, img0, K, T, out)
    print(eid, f"{len(scene_objs)} objects, {len(flags)} flags, {scene['build_s']} s", flush=True)
    return scene


HOLLOW = {"fruit_bowl", "toolbox"}


def depenetrate(objs, out, flags, res=0.002, step=0.002, max_it=60, max_push=0.06):
    """Silhouette fits are independent per object, so neighbours can interpenetrate by a few mm-cm (the physics solver then ejects them):
    rasterise every footprint (2 mm), and while two objects whose height ranges overlap share cells, move the one with the weaker
    fit (never a kinematic container) 2 mm away from the overlap. Reported per object."""
    meshes = [trimesh.load(out / "usd" / f"{o['id']}.obj", force="mesh", process=False) for o in objs]
    allv = np.concatenate([m.vertices[:, :2] + np.array(o["pos"][:2]) for m, o in zip(meshes, objs)])
    lo = allv.min(0) - 0.2
    shp = tuple(((allv.max(0) + 0.2 - lo) / res).astype(int)[::-1] + 1)

    def raster(i):
        m, o = meshes[i], objs[i]
        img = np.zeros(shp, np.uint8)
        P = ((m.vertices[:, :2] + np.array(o["pos"][:2]) - lo) / res).astype(np.int32)
        up_ = m.face_normals[:, 2] > 0.3
        cv2.fillPoly(img, [P[f] for f in m.faces[up_]], 1) if up_.any() else None
        return img.astype(bool)
    zr = [(o["pos"][2], o["pos"][2] + m.extents[2]) for m, o in zip(meshes, objs)]
    moved, raised = {}, {}
    for _ in range(max_it):
        R_ = [raster(i) for i in range(len(objs))]
        R_ = [cv2.dilate(r.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) for r in R_]     # 2 mm clearance (drop() is conservative)
        any_ov = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if min(zr[i][1], zr[j][1]) - max(zr[i][0], zr[j][0]) <= 0.003:
                    continue
                if objs[i]["key"] in HOLLOW or objs[j]["key"] in HOLLOW:      # things may lie inside the bowl / tray
                    continue
                if min(zr[i][1], zr[j][1]) - max(zr[i][0], zr[j][0]) <= 0.003:   # (re-check: an earlier raise may have separated them)
                    continue
                ov = R_[i] & R_[j]
                if ov.sum() < 2:
                    continue
                any_ov = True
                cand = [k for k in (i, j) if objs[k]["role"] != "container"]
                if not cand:
                    continue
                k = min(cand, key=lambda k: objs[k]["fit_iou"])
                o_ = j if k == i else i
                if ov.sum() > 0.20 * min(R_[i].sum(), R_[j].sum()) or moved.get(objs[k]["id"], 0.0) >= max_push:
                    # large overlap (they really lie on each other) or pushed max_push already: left to drop() (rests on top of it)
                    continue
                ys, xs = np.nonzero(ov)
                c_ov = lo + res * np.array([xs.mean(), ys.mean()])
                d = np.array(objs[k]["pos"][:2]) - c_ov
                n = np.linalg.norm(d)
                d = d / n if n > 1e-6 else np.array([1.0, 0.0])
                objs[k]["pos"][0] += float(d[0] * step); objs[k]["pos"][1] += float(d[1] * step)
                moved[objs[k]["id"]] = moved.get(objs[k]["id"], 0.0) + step
        if not any_ov:
            break
    for k, v in moved.items():
        flags.append(f"{k}: moved {v * 100:.1f} cm to remove interpenetration with a neighbour (independent silhouette fits)")
    for k, v in raised.items():
        flags.append(f"{k}: footprint largely overlaps a neighbour -> placed on top of it (bottom z {v * 100:.1f} cm)")


def drop(objs, out, flags, res=0.002, margin=0.002):
    """No object may start interpenetrating anything (the physics solver would eject it): every object is lowered vertically from above onto
    the table and the objects placed before it (order: measured bottom height, then larger footprint first), i.e. its bottom z =
    max over its vertices of (support height under the vertex - vertex height above its own bottom). Support = a 2 mm height map of the
    placed objects' visual meshes (each triangle rasterised with its max z: conservative) -> a banana measured 'in' the bowl rests on the
    bowl's inner surface, a tool on a box lies on the box. Reported when it changes the measured height by > 1 cm."""
    meshes = [trimesh.load(out / "usd" / f"{o['id']}.obj", force="mesh", process=False) for o in objs]
    allv = np.concatenate([m.vertices[:, :2] + np.array(o["pos"][:2]) for m, o in zip(meshes, objs)])
    lo = allv.min(0) - 0.05
    shp = tuple(((allv.max(0) + 0.05 - lo) / res).astype(int)[::-1] + 1)
    Hm = np.zeros(shp, np.float32)                       # table top z = 0
    order = sorted(range(len(objs)), key=lambda k: (round(objs[k]["pos"][2], 2), -objs[k]["extents"][0] * objs[k]["extents"][1]))
    for k in order:
        o, m = objs[k], meshes[k]
        V = m.vertices
        ij = ((V[:, :2] + np.array(o["pos"][:2]) - lo) / res).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, shp[1] - 1); ij[:, 1] = np.clip(ij[:, 1], 0, shp[0] - 1)
        rel = V[:, 2] - V[:, 2].min()
        # the 3x3 neighbourhood max under each vertex (triangle interiors between vertices)
        Hd = cv2.dilate(Hm, np.ones((3, 3), np.uint8))
        z0 = float(max(0.0, (Hd[ij[:, 1], ij[:, 0]] - rel).max())) + (margin if Hm.max() > 0 else 0.0)
        z0 = 0.0 if z0 <= margin + 1e-6 and (Hd[ij[:, 1], ij[:, 0]] - rel).max() <= 0 else z0
        if abs(z0 - o["pos"][2]) > 0.01:
            flags.append(f"{o['id']}: bottom height {o['pos'][2] * 100:.1f} -> {z0 * 100:.1f} cm (rests on what is under it, no interpenetration)")
        o["pos"][2] = round(z0, 4)
        P = ((V[:, :2] + np.array(o["pos"][:2]) - lo) / res).astype(np.int32)
        Z = V[:, 2] - V[:, 2].min() + z0
        if len(m.faces) > 3000:                          # dense mesh: splat vertices (max z), close 1-cell gaps
            sub = np.zeros(shp, np.float32)
            np.maximum.at(sub, (np.clip(P[:, 1], 0, shp[0] - 1), np.clip(P[:, 0], 0, shp[1] - 1)), Z.astype(np.float32))
            np.maximum(Hm, cv2.dilate(sub, np.ones((3, 3), np.uint8)), out=Hm)
            continue
        for f in m.faces:
            zf = float(Z[f].max())
            tri = P[f]
            x0_, y0_ = tri.min(0); x1_, y1_ = tri.max(0) + 1
            if x1_ - x0_ > 400 or y1_ - y0_ > 400:
                continue
            sub = np.zeros((y1_ - y0_, x1_ - x0_), np.uint8)
            cv2.fillConvexPoly(sub, tri - [x0_, y0_], 1)
            reg = Hm[y0_:y1_, x0_:x1_]
            if reg.shape != sub.shape:
                continue
            np.maximum(reg, np.where(sub.astype(bool), zf, 0.0).astype(np.float32), out=reg)


def overview_pose(table):
    """High camera behind the phone side (-y), looking down at the table centre at ~55 deg."""
    cx, cy = (table["x"][0] + table["x"][1]) / 2, (table["y"][0] + table["y"][1]) / 2
    eye = np.array([cx, table["y"][0] - 0.55, 1.35]); tgt = np.array([cx, cy, 0.0])
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    Rc = np.stack([r, -u, f], 1)                       # OpenCV camera axes (x right, y down, z forward) in the table frame
    Tc = np.eye(4); Tc[:3, :3] = Rc; Tc[:3, 3] = eye
    K = [[600.0, 0, 640], [0, 600.0, 360], [0, 0, 1]]
    return dict(K=K, T_cam_to_table=Tc.tolist(), wh=[1280, 720], note="high overview behind the phone")


def layout_check(scene, img, K, T, out):
    im = img.copy()
    for o in scene["objects"]:
        m = trimesh.load(ROOT / o["usd"].replace(".usda", ".obj"), force="mesh", process=False)
        V = m.vertices + np.array(o["pos"])
        uv, z = project(V, K, T)
        if (z <= 0).any():
            continue
        e = m.edges_unique
        col = (0, 255, 0) if o["role"] in ("task", "stack") else ((255, 128, 0) if o["role"] == "container" else (0, 200, 255))
        if len(e) > 3000:
            hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
            cv2.polylines(im, [hull], True, col, 1, cv2.LINE_AA)
        else:
            for a, b in e:
                cv2.line(im, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), col, 1, cv2.LINE_AA)
        c = uv.mean(0).astype(int)
        cv2.putText(im, f"{o['id']} {o['fit_iou']:.2f}", (int(c[0]) - 30, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "layout_check.jpg"), im)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    eps = [e for e in episodes(a.task) if not a.ids or e["id"] in a.ids]
    t0 = time.time(); n = 0
    for e in eps:
        r = build(e, a.force)
        n += r is not None
    print(f"built {n} episodes in {time.time() - t0:.0f} s ({(time.time() - t0) / max(n, 1):.1f} s / episode)", flush=True)
