"""Physics twin of one episode at its start frame: objects at their measured poses, textured table, static background,
two arms.

Per episode (video V, start frame k0 = round(start_s x 10)):
  1. objects on the table at the start: every track with a library identity sighted in the first 1.5 s; pose = median
     of its tracked 3D boxes, refined by fitting the library mesh's silhouette to the SAM 3 mask through the phone
     camera (cuboids: 24 face-up choices x yaw, lying objects: yaw x flip; both pliers meshes are tried on both pliers
     tracks and assigned by fit). Hidden, carried, uncertain, duplicate or badly fitting objects are flagged;
  2. one USD per object (yaw baked into the vertices, so the rigid-body root carries only the position), library
     textures, colliders: cuboid hull, open tray as 5 slabs, bowl as floor + wall slabs, other shapes as CoACD parts;
  3. independent fits can interpenetrate: footprints are pushed apart, then every object is dropped onto what is under
     it;
  4. table (orthophoto of the static plate) and background (the plate on its static depth, emissive, visual only);
  5. two Kuka-Allegro arms where the person's shoulders were, reaching across the table (Lift task base height and
     workspace).

  python -m cleantables.twin.scene [--task tools] [--ids IMG_2533_t01 ...] [--force]        (CPU, 2-30 s per episode)
Out data/twins/<episode>/: scene.json, usd/<object>.{usda,obj}, table.usda, table_tex.png, background.usda,
start_real.jpg, layout_check.jpg.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from cleantables.twin import fitting, meshes
from cleantables.twin.config import SceneConfig, load_config
from cleantables.twin.identity import CATALOGUE, resolve_video
from cleantables.twin.inputs import (
    IMG_H,
    IMG_W,
    LIBRARY,
    Detections,
    H,
    W,
    camera,
    depth,
    episodes,
    events,
    frame,
    hand_masks,
    plate_dir,
    tracks,
    twin_dir,
    upsample_mask,
    write_json,
)
from cleantables.twin.library import load_entry
from cleantables.twin.projection import ortho, project, rays_lo, table_extent
from cleantables.twin.usd import write_object_usd, write_table_usd

REACH_XB, REACH_YB = (-0.80, -0.25), 0.34  # Lift policy workspace in the arm base frame (forward = base -x)
DAMPING = {
    "tool": (0.05, 0.2),
    "fruit": (0.05, 0.3),
    "box": (0.1, 0.1),
    "container": (0.1, 0.1),
    "obstacle": (0.1, 0.1),
}
YAW_CAMERA_SIDE = -math.pi / 2  # bases on the phone side face +y, across the table
HOLLOW = {"fruit_bowl", "toolbox"}  # things legitimately lie inside these
LYING_KINDS = ("flat", "pillow", "hy3d", "apple")
MULTI_INSTANCE = ("banana_a", "banana_b", "green_apple")  # several physical objects share one library mesh
CONTAINER_KEYS = ("toolbox", "white_box", "fruit_bowl")
LIBRARY_FROM_USD = "../../library"  # library folder seen from twins/<episode>/usd/
TABLE_TEX_RES = 0.002
BACKDROP_DISTANCE = 8.0  # m in front of the phone: fills the background's dropped slivers


# ------------------------------------------------------------------ physics and arms
def physics_policy(category: str, entry: dict, cfg: SceneConfig) -> tuple[float, tuple]:
    """boxes and the lid are heavy with high friction and containers heavy, so stacks and containers stay put under
    contact; tools and fruit keep their real masses (the real box masses stay in scene.json as mass_real)."""
    if category.startswith("box/"):
        return cfg.box_mass_kg, tuple(cfg.box_friction)
    if category.startswith("container/"):
        return cfg.container_mass_kg, tuple(entry["mu"])
    return entry["mass"], tuple(entry["mu"])


def reach_b(base_xy, xy, yaw: float = YAW_CAMERA_SIDE):
    """True where xy lies in the Lift policy's workspace of an arm based at base_xy with the given yaw."""
    d = np.asarray(xy, float) - np.asarray(base_xy, float)
    c, s_ = math.cos(yaw), math.sin(yaw)
    xb = c * d[..., 0] + s_ * d[..., 1]
    yb = -s_ * d[..., 0] + c * d[..., 1]
    return (xb >= REACH_XB[0]) & (xb <= REACH_XB[1]) & (np.abs(yb) <= REACH_YB)


def shoulder_poses(video: str, table: dict, objs: list, tasks: list, flags: list, cfg: SceneConfig, cam_xyz=None):
    """robot bases where the person's shoulders were.

    The side of the table with most body points (plate.person_cloud) is the person's side; shoulder centre x = median
    x of the top 15 cm of the body there. Shoulder width cannot be measured from behind, so the separation (0.40-0.64
    m) and a +-10 cm shift of the centre are chosen to maximise the tasks one arm can do alone (object and destination
    in its workspace), then the sharing between the arms.
    """
    P = np.load(plate_dir(video) / "person_pts.npy")
    y0, y1 = table["y"]
    upright = (P[:, 2] > 0.05) & (P[:, 2] < 1.2) if len(P) else np.zeros(0, bool)
    near = P[upright & (P[:, 1] < y0 + 0.05)] if len(P) else P
    far = P[upright & (P[:, 1] > y1 - 0.05)] if len(P) else P
    side = "camera" if len(near) >= len(far) else "far"
    body = near if side == "camera" else far
    if len(body) >= 300:
        ztop = float(np.percentile(body[:, 2], 95))
        band = body[body[:, 2] > ztop - 0.15]
        xc_m = float(np.median(band[:, 0]))
        est = dict(x=xc_m, top_z=ztop, y=float(np.median(band[:, 1])), n_pts=int(len(body)))
    else:
        xc_m = 0.5 * (table["x"][0] + table["x"][1])
        est = dict(x=xc_m, n_pts=int(len(body)), fallback="table centre (too few body points)")
        flags.append("shoulders: too few body points, table centre used")
    est.update(side=side, n_near=int(len(near)), n_far=int(len(far)))
    yaw = YAW_CAMERA_SIDE if side == "camera" else -YAW_CAMERA_SIDE
    quat = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    y_b = y0 - cfg.base_gap_m if side == "camera" else y1 + cfg.base_gap_m

    def camera_clearance_penalty(xc, sp):
        # phone on the person's side: its lens is just behind the bases; closer than 28 cm sideways an elbow fills the
        # view
        if side != "camera" or cam_xyz is None:
            return 0.0
        return max(0.0, 0.28 - min(abs(xc + sp / 2 - cam_xyz[0]), abs(xc - sp / 2 - cam_xyz[0])))

    best, arg = -1e9, (xc_m, 0.48)
    for xc in np.arange(xc_m - 0.10, xc_m + 0.1001, 0.02):
        for sp in np.arange(0.40, 0.641, 0.04):
            ok, na, nb = 0, 0, 0
            for k, kc in tasks:
                po, pd = objs[k]["pos"][:2], objs[kc]["pos"][:2]
                ra = bool(reach_b((xc + sp / 2, y_b), po, yaw) and reach_b((xc + sp / 2, y_b), pd, yaw))
                rb = bool(reach_b((xc - sp / 2, y_b), po, yaw) and reach_b((xc - sp / 2, y_b), pd, yaw))
                ok += ra or rb
                na += ra
                nb += rb
            score = ok + 0.3 * min(na, nb) - 0.5 * abs(xc - xc_m) - 8.0 * camera_clearance_penalty(xc, sp)
            if score > best:
                best, arg = score, (float(xc), float(sp))
    xc, sp = arg
    poses = {"a": [[xc + sp / 2, y_b, -cfg.table_drop_m], quat], "b": [[xc - sp / 2, y_b, -cfg.table_drop_m], quat]}
    est.update(chosen_centre_x=xc, separation=sp, base_y=y_b, yaw_deg=round(math.degrees(yaw), 1))
    return poses, est


# ------------------------------------------------------------------ objects at the start
def _resting_pose_sightings(o, r, k0, ks, R, track, hand_frac, flags):
    """object carried at the start: containers being set down take their next resting pose, anything just picked up
    (rest ended <= 3 s before k0) the previous one. -> (sighting indices, note) or None if it is omitted."""
    prev = [q for q in o["rests"] if q["f1"] < k0]
    nxt = [q for q in o["rests"] if q["f0"] > k0]
    if r["key"] in CONTAINER_KEYS:
        q = nxt[0] if nxt else (prev[-1] if prev else None)
    else:
        q = prev[-1] if (prev and k0 - prev[-1]["f1"] <= 30) else None
    if q is None:
        flags.append(f"{r['key']} (track {o['id']}): carried at the start, no resting pose -> omitted")
        return None
    sel = [i for i, k in enumerate(ks.tolist()) if q["f0"] <= k <= q["f1"] and R["det_box"].get(track["det"][i])]
    if not sel:
        flags.append(f"{r['key']} (track {o['id']}): carried at the start, resting pose not sighted -> omitted")
        return None
    sel = sel[:15] if q in nxt else sel[-15:]
    note = (
        f"carried at the start (hand overlap {hand_frac:.2f}): placed at its {'next' if q in nxt else 'previous'} "
        f"resting pose (k {q['f0']}-{q['f1']})"
    )
    flags.append(f"{r['key']} (track {o['id']}): {note}")
    return sel, note


def start_objects(ep: dict, flags: list, cfg: SceneConfig) -> tuple[list[dict], int]:
    v = ep["video"]
    k0 = int(round(ep["start_s"] * 10))
    win = cfg.start_window_frames
    E, R = events(v), tracks(v)
    res = {r["track"]: r for r in resolve_video(E["objects"], v)}
    out = []
    for o in E["objects"]:
        r = res[o["id"]]
        if r["key"] is None:
            continue
        track = R["objects"][o["src"]]
        ks = track["sight_k"]
        sel = [i for i, k in enumerate(ks.tolist()) if k0 <= k <= k0 + win and R["det_box"].get(track["det"][i])]
        note = None
        if not sel:
            if not (o["first"] <= k0 + win and o["last"] >= k0):
                continue  # not on the table at the start
            near = sorted(range(len(ks)), key=lambda i: abs(int(ks[i]) - k0))
            near = [i for i in near if R["det_box"].get(track["det"][i]) and abs(int(ks[i]) - k0) <= 50][:3]
            if not near:
                flags.append(
                    f"{r['key']} (track {o['id']}): present but not sighted within 5 s of the start -> omitted"
                )
                continue
            sel = near
            near_k = [int(ks[i]) for i in near]
            note = f"hidden at the start (person's body / hand): pose from the nearest sightings k={near_k}"
            flags.append(f"{r['key']} (track {o['id']}): {note}")
        B = [R["det_box"][track["det"][i]] for i in sel]
        hand_frac = float(np.median([b.get("hand_frac", 0.0) for b in B]))
        resting = any(q["f0"] <= k0 + win and q["f1"] >= k0 for q in o["rests"])
        if note is None and not resting and hand_frac > 0.02:
            found = _resting_pose_sightings(o, r, k0, ks, R, track, hand_frac, flags)
            if found is None:
                continue
            sel, note = found
            B = [R["det_box"][track["det"][i]] for i in sel]
        xyz = np.median([[b["x"], b["y"], b["top"]] for b in B], 0)
        # mask for the fit: the least hand-covered, largest sighting
        i_best = max(
            sel,
            key=lambda i: (
                -(R["det_box"][track["det"][i]].get("hand_frac", 0) > 0.02),
                R["det_box"][track["det"][i]]["n"],
            ),
        )
        b = R["det_box"][track["det"][i_best]]
        out.append(
            dict(
                track=o["id"],
                key=r["key"],
                why=r["why"],
                uncertain=r["uncertain"],
                x=float(xyz[0]),
                y=float(xyz[1]),
                top=float(xyz[2]),
                yaw=float(b["yaw"]),
                L=float(b["L"]),
                W=float(b["W"]),
                k_mask=int(ks[i_best]),
                det=int(track["det"][i_best]),
                n_sight=len(sel),
                note=note,
                hand_frac=float(b.get("hand_frac", 0)),
            )
        )
        if r["uncertain"]:
            flags.append(f"{r['key']} (track {o['id']}): uncertain identity ({r['why']})")
        if len(sel) < 3 and note is None:
            flags.append(f"{r['key']} (track {o['id']}): only {len(sel)} sightings in the first 1.5 s")
    _resolve_toolbox(out, flags)
    out = [o for o in out if o["key"] is not None]
    keep = []
    for o in sorted(out, key=lambda o: -o["n_sight"]):
        if (
            o["key"] not in {"banana", "green_apple"}
            and o["key"] != "pliers"
            and any(k["key"] == o["key"] for k in keep)
        ):
            flags.append(f"{o['key']}: second track {o['track']} at the start (duplicate identity) -> dropped")
            continue
        keep.append(o)
    return keep, k0


def _resolve_toolbox(objs: list, flags: list) -> None:
    """an open tray wins; a white box is then the lid (or lies on / under the tray), else it is the closed toolbox."""
    trays = [o for o in objs if o["key"] == "toolbox"]
    for w in [o for o in objs if o["key"] == "white_box"]:
        if trays and min(math.hypot(w["x"] - t["x"], w["y"] - t["y"]) for t in trays) < 0.15:
            w["key"] = None
            flags.append(f"white box track {w['track']} overlaps the open tray -> lid on / under the tray, omitted")
        elif trays:
            w["key"] = "toolbox_lid"
        else:
            w["key"] = "toolbox"
            trays.append(w)
            flags.append("toolbox closed at the start (lid on): the twin shows the open tray, lid omitted")


# ------------------------------------------------------------------ fitting the library meshes
def care_mask(dets: Detections, R: dict, k: int, own_det: int, hand) -> np.ndarray:
    """pixels where the fit counts: all but other objects' masks and the hand (they may occlude this object)."""
    other = np.zeros((H, W), bool)
    for d in R["kept"].get(k, []):
        if d != own_det:
            other |= dets.mask(d)
    if hand is not None:
        other |= hand
    return ~upsample_mask(cv2.dilate(other.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))


def appearance_error(m, key, yaw, x, y, z0, K, T, img, mask) -> float:
    """mean abs colour error (0-255) between the mesh's textured top-facing triangles and the real pixels under them."""
    tex = cv2.imread(str(LIBRARY / key / "texture.png"))
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    uv_img, _ = project(m.vertices @ Rz + np.array([x, y, z0]), K, T)
    UV = m.visual.uv
    errs = []
    nz = m.face_normals @ Rz
    for f in np.nonzero(nz[:, 2] > 0.7)[0][:40]:
        tri = uv_img[m.faces[f]]
        for w in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
            p = (np.array(w)[:, None] * tri).sum(0)
            if not (0 <= p[0] < IMG_W - 1 and 0 <= p[1] < IMG_H - 1) or not mask[int(p[1]), int(p[0])]:
                continue
            t = (np.array(w)[:, None] * UV[m.faces[f]]).sum(0)
            tc = tex[int((1 - t[1]) * (tex.shape[0] - 1)), int(t[0] * (tex.shape[1] - 1))].astype(float)
            errs.append(np.abs(tc - img[int(p[1]), int(p[0])].astype(float)).mean())
    return float(np.mean(errs)) if errs else 60.0


def fit_cuboid_object(o, lib_mesh, meta, mask, care, K, T, img, scale=None):
    """face-up rotation (of 24) whose (L, W, H) best matches the measured footprint and height, both yaw and yaw + 180
    deg
    (told apart by appearance). scale=None: probe fit with a free scale (0.7-1.4), returns (fit, None)."""
    dims = np.array(meta["extents"]) * (scale or 1.0)
    lib_mesh = fitting.scaled(lib_mesh, scale or 1.0, flat=False)
    meas = np.array([max(o["L"], o["W"]), min(o["L"], o["W"]), max(o["top"], 0.01)])
    cands = []
    for Rm in fitting.cube_rotations():
        d = np.abs(Rm @ dims)
        if d[0] + 1e-6 < d[1]:
            continue  # L >= W in the rotated frame; yaw covers the rest
        cost = float(np.abs(np.log(d[:2] / np.maximum(meas[:2], 0.02))).sum())
        if o["top"] - d[2] <= 0.03:  # resting on something: the height says nothing
            cost += float(abs(math.log(d[2] / meas[2])))
        cands.append((cost, Rm, d))
    cands.sort(key=lambda c: c[0])
    best = None
    cmin = cands[0][0]
    for cost, Rm, d in [c for c in cands if c[0] <= cmin + 0.15][:8]:
        z0 = max(0.0, o["top"] - d[2]) if o["top"] - d[2] > 0.03 else 0.0
        yaw0 = o["yaw"] if o["L"] >= o["W"] else o["yaw"] + math.pi / 2
        # the video scale times a per-object factor in [0.9, 1.1]: box sizes measured in another video's depth differ
        # by a few %
        f = fitting.fit_cuboid(
            mask,
            K,
            T,
            (o["x"], o["y"], yaw0, d[0], d[1], d[2]),
            fix_dims=tuple(d),
            z0=z0,
            care=care,
            free_scale=True,
            scale_bounds=(0.7, 1.4) if scale is None else (0.9, 1.1),
        )
        if scale is None:
            return f, None
        for dyaw in (0.0, math.pi):
            m = fitting.scaled(lib_mesh, f["scale"], flat=False)
            M = np.eye(4)
            M[:3, :3] = Rm
            m.apply_transform(M)
            lo, hi = m.bounds
            m.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
            yaw = f["yaw"] + dyaw
            app = appearance_error(m, meta["key"], yaw, f["x"], f["y"], z0, K, T, img, mask & care)
            score = f["iou"] - 0.002 * app - 0.05 * cost
            if best is None or score > best[0]:
                best = (
                    score,
                    dict(
                        f,
                        yaw=yaw,
                        z0=z0,
                        R=Rm.tolist(),
                        dims=(d * f["scale"]).tolist(),
                        app_err=app,
                        dim_cost=cost,
                        obj_scale=f["scale"],
                    ),
                    m,
                )
    return best[1], best[2]


def fit_library_object(o, key, scale, ctx, K, T):
    """fit library entry `key` to object o. scale=None: probe with a free scale; else video scale x 0.9-1.1."""
    mask, care, img = ctx[o["track"]]
    mesh, _, meta = load_entry(key)
    kind = CATALOGUE[key]["kind"]
    if kind not in LYING_KINDS:
        f, m = fit_cuboid_object(o, mesh, meta, mask, care, K, T, img, scale)
        if scale is not None:
            f["scale"] = scale * f.get("obj_scale", 1.0)
        return f, m, meta
    flat = kind in ("flat", "pillow")
    f = fitting.fit_flat(
        mesh if scale is None else fitting.scaled(mesh, scale, flat),
        mask,
        K,
        T,
        o["x"],
        o["y"],
        o["yaw"],
        care=care,
        free_scale=True,
        scale_bounds=(0.7, 1.4) if scale is None else (0.9, 1.1),
        flips=(False, True) if flat else (False,),
    )
    if scale is None:
        return f, None, meta
    f["obj_scale"] = f["scale"]
    m = fitting.scaled(mesh, scale * f["obj_scale"], flat)
    if f["flipped"]:
        fitting.flip_over(m)
    f["z0"] = max(0.0, o["top"] - m.extents[2]) if o["top"] - m.extents[2] > 0.02 else 0.0
    f["scale"] = scale * f.get("obj_scale", 1.0)
    return f, m, meta


def library_keys(o: dict) -> list[str]:
    """library meshes to try for a perception identity (pliers and bananas are not individually identifiable)."""
    return {"pliers": ["pliers_a", "pliers_b"], "banana": ["banana_a", "banana_b"]}.get(o["key"], [o["key"]])


def has_entry(key: str) -> bool:
    return key in CATALOGUE and (LIBRARY / key / "mesh.npz").exists()


def fit_contexts(v, objs, G, dets, R, hands) -> dict:
    """track -> (mask, care, frame) at the fit sighting; care excludes other objects, the hand and the person's body
    (pixels nearer than the static background)."""
    plate_depth = np.load(plate_dir(v) / "plate_depth.npy")
    D10 = depth(v)
    ctx = {}
    for o in objs:
        k = o["k_mask"]
        allm = np.zeros((H, W), bool)
        for d in R["kept"].get(k, []):
            allm |= dets.mask(d)
        dk = np.asarray(D10[k], np.float32) * G["scale"][k]
        body = (dk < plate_depth - 0.08) & ~cv2.dilate(allm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        body = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        care = care_mask(dets, R, k, o["det"], hands.get(k)) & ~upsample_mask(body)
        ctx[o["track"]] = (upsample_mask(dets.mask(o["det"])), care, frame(v, k))
    return ctx


def video_scale(objs, ctx, K, T, flags) -> tuple[float, list]:
    """library sizes come from other videos whose depth scale differs by a few %: one isotropic factor per video = the
    median free-scale fit of the well-seen, unoccluded objects."""
    probes = []
    for o in objs:
        mask, care, _ = ctx[o["track"]]
        if o["key"] == "pliers" or (mask & ~care).sum() > 0.05 * mask.sum() or o["note"]:
            continue
        key = library_keys(o)[0]
        if not has_entry(key):
            continue
        f, _, _ = fit_library_object(o, key, None, ctx, K, T)
        if f["iou"] >= 0.75:
            probes.append((key, f["scale"], f["iou"]))
    if len(probes) < 2:
        flags.append("video scale: < 2 clean objects to estimate it, 1.0 used")
        return 1.0, probes
    return float(np.median([p[1] for p in probes])), probes


def assign_pliers(pliers: list, fits: dict, flags: list) -> dict:
    """track -> pliers_a / pliers_b by the better total silhouette fit."""
    if len(pliers) == 2:
        s1 = fits[(pliers[0]["track"], "pliers_a")][0]["iou"] + fits[(pliers[1]["track"], "pliers_b")][0]["iou"]
        s2 = fits[(pliers[0]["track"], "pliers_b")][0]["iou"] + fits[(pliers[1]["track"], "pliers_a")][0]["iou"]
        if abs(s1 - s2) < 0.05:
            flags.append(f"pliers identity by shape is weak (IoU sums {s1:.2f} vs {s2:.2f})")
        return {
            pliers[0]["track"]: "pliers_a" if s1 >= s2 else "pliers_b",
            pliers[1]["track"]: "pliers_b" if s1 >= s2 else "pliers_a",
        }
    if len(pliers) == 1:
        o = pliers[0]
        a, b = fits[(o["track"], "pliers_a")][0]["iou"], fits[(o["track"], "pliers_b")][0]["iou"]
        if abs(a - b) < 0.05:
            flags.append(f"single pliers: identity by shape weak ({a:.2f} vs {b:.2f})")
        return {o["track"]: "pliers_a" if a >= b else "pliers_b"}
    if len(pliers) > 2:
        flags.append(f"{len(pliers)} pliers tracks at the start (expected <= 2): extra ones dropped")
    return {}


def place_objects(objs, ctx, K, T, flags) -> tuple[list[dict], float, list]:
    """fit every start object -> placed objects (library key, fit, posed mesh), video scale and its probes."""
    s_vid, probes = video_scale(objs, ctx, K, T, flags)
    fits = {}
    for o in objs:
        for key in library_keys(o):
            if not has_entry(key):
                flags.append(f"{key}: no library mesh -> omitted")
                continue
            fits[(o["track"], key)] = fit_library_object(o, key, s_vid, ctx, K, T)
    assign = assign_pliers([o for o in objs if o["key"] == "pliers"], fits, flags)
    placed, used, count = [], set(), {}
    for o in objs:
        if o["key"] == "pliers":
            key = assign.get(o["track"])
        elif o["key"] == "banana":  # the better-fitting of the two banana meshes
            cands = [
                (fits[(o["track"], kk)][0]["iou"], kk) for kk in ("banana_a", "banana_b") if (o["track"], kk) in fits
            ]
            key = max(cands)[1] if cands else None
        else:
            key = o["key"]
        if key is None or (o["track"], key) not in fits:
            continue
        if key in used and key not in MULTI_INSTANCE:
            continue
        used.add(key)
        f, m, meta = fits[(o["track"], key)]
        if f["iou"] < 0.3 and CATALOGUE[key]["kind"] in LYING_KINDS:
            # the silhouette fit failed (mask occluded or merged): keep the mesh at the tracked 3D box pose
            xy = np.asarray(m.vertices)[:, :2] - np.asarray(m.vertices)[:, :2].mean(0)
            w, V = np.linalg.eigh(np.cov(xy.T))
            ax = V[:, np.argmax(w)]
            f = dict(
                f,
                x=o["x"],
                y=o["y"],
                yaw=o["yaw"] - math.atan2(ax[1], ax[0]),
                fallback=f"tracked 3D box pose (fit IoU {f['iou']:.2f})",
            )
            m = m.copy()
            c = m.bounds.mean(0)
            m.apply_translation((-c[0], -c[1], 0))
            flags.append(
                f"{key}: silhouette fit failed (IoU {fits[(o['track'], key)][0]['iou']:.2f}) "
                "-> tracked 3D box pose used"
            )
        elif f["iou"] < 0.6:
            flags.append(f"{key}: weak silhouette fit IoU {f['iou']:.2f} (occluded or wrong size)")
        count[key] = count.get(key, 0) + 1
        placed.append(
            dict(o=o, key=key, fit=f, mesh=m, meta=meta, id=f"{key}_{count[key]:02d}" if key in MULTI_INSTANCE else key)
        )
    return placed, s_vid, probes


# ------------------------------------------------------------------ USD per object
def collider_for(key: str, f: dict, m: trimesh.Trimesh, Rz: np.ndarray) -> tuple:
    kind = CATALOGUE[key]["kind"]
    if kind == "tray":
        parts = []
        for size, pos in meshes.tray_colliders(*f["dims"]):
            bx = trimesh.creation.box(size, trimesh.transformations.translation_matrix(pos))
            bx.apply_transform(Rz)
            parts.append(bx)
        return ("parts", parts)
    if kind in ("cuboid", "apple"):
        return ("hull",)
    if key == "fruit_bowl":
        return ("bowl", meshes.bowl_profile(np.asarray(m.vertices)))
    cache = LIBRARY / key / "coacd.npz"
    if not cache.exists():
        parts0 = meshes.convex_parts(load_entry(key)[0])
        np.savez(
            cache,
            **{f"v{i}": q.vertices for i, q in enumerate(parts0)},
            **{f"f{i}": q.faces for i, q in enumerate(parts0)},
            n=len(parts0),
        )
    z = np.load(cache)
    sc = float(f.get("scale") or 1.0)  # the video scale applies to the collision parts as to the visual
    sz = 1.0 if kind in ("flat", "pillow") else sc
    parts = [trimesh.Trimesh(z[f"v{i}"] * np.array([sc, sc, sz]), z[f"f{i}"]) for i in range(int(z["n"]))]
    if f.get("flipped"):
        for q in parts:
            q.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0)))
        lo = min(q.bounds[0][2] for q in parts)
        for q in parts:
            q.apply_translation((0, 0, -lo))
    for q in parts:
        q.apply_transform(Rz)
    return ("parts", parts)


def object_role(task: str, key: str, cls: str) -> str:
    if task == "tools":
        return "task" if cls == "tool" else ("container" if key == "toolbox" else "obstacle")
    if task == "fruit":
        return "task" if cls == "fruit" else ("container" if key == "fruit_bowl" else "obstacle")
    return "stack" if (cls == "box" or key == "toolbox") else "obstacle"


def write_objects(ep: dict, placed: list, out: Path, cfg: SceneConfig) -> list[dict]:
    """object USDs + .obj (yaw baked in) -> scene.json object records."""
    scene_objs = []
    for p in placed:
        key, f, m = p["key"], p["fit"], p["mesh"].copy()
        entry = CATALOGUE[key]
        c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
        Rz = np.eye(4)
        Rz[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        m.apply_transform(Rz)
        col = collider_for(key, f, m, Rz)
        usd = out / "usd" / f"{p['id']}.usda"
        # in the stacking task the toolbox is one of the boxes to stack, so it must be movable
        category = "box/toolbox" if (ep["task"] == "stack_boxes" and key == "toolbox") else entry["category"]
        mass, mu = physics_policy(category, entry, cfg)
        write_object_usd(
            usd, m, f"{LIBRARY_FROM_USD}/{key}/texture.png", category, mass, mu, col, extra_attrs=dict(library_key=key)
        )
        m.export(out / "usd" / f"{p['id']}.obj")
        cls = entry["cls"]
        role = object_role(ep["task"], key, cls)
        scene_objs.append(
            dict(
                id=p["id"],
                key=key,
                category=category,
                cls=cls if cls in ("tool", "fruit", "container") else "obstacle",
                role=role,
                split="-",
                usd=f"usd/{p['id']}.usda",
                pos=[f["x"], f["y"], float(f.get("z0", 0.0))],
                extents=[round(float(x), 4) for x in m.extents],
                yaw=0.0,
                yaw_baked=round(float(f["yaw"]), 4),
                mass=mass,
                mu=list(mu),
                mass_real=entry["mass"],
                damping=list(DAMPING[cls]),
                collider=col[0],
                dest=entry.get("dest") if role == "task" else None,
                library=key,
                track=p["o"]["track"],
                fit_iou=round(float(f["iou"]), 3),
                n_sightings_start=p["o"]["n_sight"],
                flipped=bool(f.get("flipped", False)),
                note=p["o"]["note"],
            )
        )
    return scene_objs


# ------------------------------------------------------------------ contact clean-up
def _load_obj(out: Path, o: dict) -> trimesh.Trimesh:
    return trimesh.load(out / o["usd"].replace(".usda", ".obj"), force="mesh", process=False)


def depenetrate(objs: list, out: Path, flags: list, res=0.002, step=0.002, max_it=60, max_push=0.06) -> None:
    """separate interpenetrating neighbours: rasterise every footprint (2 mm) and, while two objects with overlapping
    height ranges share cells, move the one with the weaker fit (never a container) 2 mm away from the overlap. Large
    overlaps (one lies on the other) are left to drop()."""
    if not objs:
        return
    meshes_ = [_load_obj(out, o) for o in objs]
    allv = np.concatenate([m.vertices[:, :2] + np.array(o["pos"][:2]) for m, o in zip(meshes_, objs, strict=False)])
    lo = allv.min(0) - 0.2
    shp = tuple(((allv.max(0) + 0.2 - lo) / res).astype(int)[::-1] + 1)

    def raster(i):
        m, o = meshes_[i], objs[i]
        img = np.zeros(shp, np.uint8)
        P = ((m.vertices[:, :2] + np.array(o["pos"][:2]) - lo) / res).astype(np.int32)
        up = m.face_normals[:, 2] > 0.3
        if up.any():
            cv2.fillPoly(img, [P[f] for f in m.faces[up]], 1)
        return img.astype(bool)

    zr = [(o["pos"][2], o["pos"][2] + m.extents[2]) for m, o in zip(meshes_, objs, strict=False)]
    moved = {}
    for _ in range(max_it):
        # 2 mm clearance (drop() rests objects conservatively)
        rasters = [
            cv2.dilate(raster(i).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) for i in range(len(objs))
        ]
        any_overlap = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if min(zr[i][1], zr[j][1]) - max(zr[i][0], zr[j][0]) <= 0.003:
                    continue
                if objs[i]["key"] in HOLLOW or objs[j]["key"] in HOLLOW:
                    continue
                ov = rasters[i] & rasters[j]
                if ov.sum() < 2:
                    continue
                any_overlap = True
                movable = [k for k in (i, j) if objs[k]["role"] != "container"]
                if not movable:
                    continue
                k = min(movable, key=lambda k: objs[k]["fit_iou"])
                if (
                    ov.sum() > 0.20 * min(rasters[i].sum(), rasters[j].sum())
                    or moved.get(objs[k]["id"], 0.0) >= max_push
                ):
                    continue
                ys, xs = np.nonzero(ov)
                d = np.array(objs[k]["pos"][:2]) - (lo + res * np.array([xs.mean(), ys.mean()]))
                n = np.linalg.norm(d)
                d = d / n if n > 1e-6 else np.array([1.0, 0.0])
                objs[k]["pos"][0] += float(d[0] * step)
                objs[k]["pos"][1] += float(d[1] * step)
                moved[objs[k]["id"]] = moved.get(objs[k]["id"], 0.0) + step
        if not any_overlap:
            break
    for k, v in moved.items():
        flags.append(
            f"{k}: moved {v * 100:.1f} cm to remove interpenetration with a neighbour (independent silhouette fits)"
        )


def drop(objs: list, out: Path, flags: list, res=0.002, margin=0.002) -> None:
    """lower every object from above onto the table and the objects placed before it (by measured bottom height, larger
    footprint first): bottom z = max over its vertices of (support height - vertex height above its bottom). Support =
    2 mm height map of the placed visual meshes (triangles at their max z, conservative), so a banana measured in the
    bowl rests on the bowl's inner surface. Flagged when the height changes by more than 1 cm."""
    if not objs:
        return
    meshes_ = [_load_obj(out, o) for o in objs]
    allv = np.concatenate([m.vertices[:, :2] + np.array(o["pos"][:2]) for m, o in zip(meshes_, objs, strict=False)])
    lo = allv.min(0) - 0.05
    shp = tuple(((allv.max(0) + 0.05 - lo) / res).astype(int)[::-1] + 1)
    Hm = np.zeros(shp, np.float32)  # table top z = 0
    order = sorted(
        range(len(objs)), key=lambda k: (round(objs[k]["pos"][2], 2), -objs[k]["extents"][0] * objs[k]["extents"][1])
    )
    for k in order:
        o, m = objs[k], meshes_[k]
        V = m.vertices
        ij = ((V[:, :2] + np.array(o["pos"][:2]) - lo) / res).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, shp[1] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, shp[0] - 1)
        rel = V[:, 2] - V[:, 2].min()
        Hd = cv2.dilate(Hm, np.ones((3, 3), np.uint8))  # 3x3 max: triangle interiors between vertices
        gap = (Hd[ij[:, 1], ij[:, 0]] - rel).max()
        z0 = float(max(0.0, gap)) + (margin if Hm.max() > 0 else 0.0)
        z0 = 0.0 if z0 <= margin + 1e-6 and gap <= 0 else z0
        if abs(z0 - o["pos"][2]) > 0.01:
            flags.append(
                f"{o['id']}: bottom height {o['pos'][2] * 100:.1f} -> {z0 * 100:.1f} cm "
                "(rests on what is under it, no interpenetration)"
            )
        o["pos"][2] = round(z0, 4)
        P = ((V[:, :2] + np.array(o["pos"][:2]) - lo) / res).astype(np.int32)
        Z = V[:, 2] - V[:, 2].min() + z0
        if len(m.faces) > 3000:  # dense mesh: splat vertices (max z), close 1-cell gaps
            sub = np.zeros(shp, np.float32)
            np.maximum.at(sub, (np.clip(P[:, 1], 0, shp[0] - 1), np.clip(P[:, 0], 0, shp[1] - 1)), Z.astype(np.float32))
            np.maximum(Hm, cv2.dilate(sub, np.ones((3, 3), np.uint8)), out=Hm)
            continue
        for f in m.faces:
            zf = float(Z[f].max())
            tri = P[f]
            x0, y0 = tri.min(0)
            x1, y1 = tri.max(0) + 1
            if x1 - x0 > 400 or y1 - y0 > 400:
                continue
            sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(sub, tri - [x0, y0], 1)
            reg = Hm[y0:y1, x0:x1]
            if reg.shape != sub.shape:
                continue
            np.maximum(reg, np.where(sub.astype(bool), zf, 0.0).astype(np.float32), out=reg)


# ------------------------------------------------------------------ table and background
def table_texture(v: str, K, T, ext, out: Path) -> float:
    """orthophoto of the plate over the table (2 mm/px); large unseen areas tiled with the flattest fully seen 24 cm
    patch
    (mirrored), small holes inpainted. Returns the observed fraction."""
    plate = cv2.imread(str(plate_dir(v) / "plate.png"))
    valid = cv2.imread(str(plate_dir(v) / "plate_valid.png"), 0) > 127
    (x0, x1), (y0, y1) = ext
    orth, (mx, my) = ortho(plate, K, T, x0, x1, y0, y1, z=0.0, res=TABLE_TEX_RES)
    in_view = (mx >= 0) & (mx < IMG_W - 1) & (my >= 0) & (my < IMG_H - 1)
    seen = (
        cv2.remap(
            valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)
        & in_view
    )
    integral = cv2.integral(seen.astype(np.uint8))
    ps = 120
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(orth, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    best, bp = -1e9, (0, 0)
    for r in range(0, seen.shape[0] - ps, 10):
        for c in range(0, seen.shape[1] - ps, 10):
            s = integral[r + ps, c + ps] - integral[r, c + ps] - integral[r + ps, c] + integral[r, c]
            if s < 0.98 * ps * ps:
                continue
            sc = -lap[r : r + ps, c : c + ps].mean()
            if sc > best:
                best, bp = sc, (r, c)
    patch = orth[bp[0] : bp[0] + ps, bp[1] : bp[1] + ps]
    patch = np.concatenate([patch, patch[:, ::-1]], 1)
    patch = np.concatenate([patch, patch[::-1]], 0)
    tiled = np.tile(patch, (seen.shape[0] // patch.shape[0] + 1, seen.shape[1] // patch.shape[1] + 1, 1))
    tiled = tiled[: seen.shape[0], : seen.shape[1]]
    hole = (~seen).astype(np.uint8)
    big = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8)).astype(bool)
    base = np.where(big[..., None], tiled, orth)
    small = (hole.astype(bool) & ~big).astype(np.uint8) * 255
    tex = cv2.inpaint(base, small, 5, cv2.INPAINT_TELEA)
    tex = np.where(seen[..., None], orth, tex)
    cv2.imwrite(str(out / "table_tex.png"), tex)
    return float(seen.mean())


def write_background(v: str, K, T, ext, path: Path, emissive: float, rel_jump: float = 0.04) -> int:
    """static plate on its static depth as a triangle mesh in the table frame (emissive, visual only). Triangles
    across depth jumps and on the table top (the table has its own textured quad) are dropped; a far backdrop quad
    fills the gaps."""
    D = np.load(plate_dir(v) / "plate_depth.npy")
    P = ((rays_lo(K) * D[..., None]).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(H, W, 3)
    (x0, x1), (y0, y1) = ext
    on_top = (
        (np.abs(P[..., 2]) < 0.02)
        & (P[..., 0] > x0 + 0.01)
        & (P[..., 0] < x1 - 0.01)
        & (P[..., 1] > y0 + 0.01)
        & (P[..., 1] < y1 - 0.01)
    )
    ys, xs = np.mgrid[0 : H - 1, 0 : W - 1]
    F = []
    for a, b, c in (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))):
        r0, c0 = ys + a[0], xs + a[1]
        r1, c1 = ys + b[0], xs + b[1]
        r2, c2 = ys + c[0], xs + c[1]
        d = np.stack([D[r0, c0], D[r1, c1], D[r2, c2]], -1)
        ok = (d.max(-1) - d.min(-1)) < rel_jump * d.min(-1)
        ok &= ~(on_top[r0, c0] & on_top[r1, c1] & on_top[r2, c2])
        ok &= np.isfinite(d).all(-1) & (d.min(-1) > 0.2)
        F.append(np.stack([r0 * W + c0, r1 * W + c1, r2 * W + c2], -1)[ok])  # counter-clockwise seen from the phone
    F = np.concatenate(F)
    used = np.unique(F)
    remap = -np.ones(H * W, np.int64)
    remap[used] = np.arange(len(used))
    V = P.reshape(-1, 3)[used]
    vv, uu = np.divmod(used, W)
    UV = np.c_[(uu * 2 + 1) / float(IMG_W), 1 - (vv * 2 + 1) / float(IMG_H)]
    zf = BACKDROP_DISTANCE
    corners = np.array([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]], float)
    Pc = np.c_[(corners[:, 0] - K[0, 2]) / K[0, 0] * zf, (corners[:, 1] - K[1, 2]) / K[1, 1] * zf, np.full(4, zf)]
    nb = len(V)
    V = np.r_[V, Pc @ T[:3, :3].T + T[:3, 3]]
    UV = np.r_[UV, [[0, 1], [1, 1], [1, 0], [0, 0]]]
    faces = np.r_[remap[F], [[nb, nb + 3, nb + 2], [nb, nb + 2, nb + 1]]]
    m = trimesh.Trimesh(V, faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=UV)
    write_object_usd(
        path,
        m,
        f"../plates/{v}/plate.png",
        "background/static_plate",
        0,
        (0, 0),
        ("none",),
        emissive=True,
        emissive_scale=emissive,
    )
    return len(F)


def overview_pose(table: dict) -> dict:
    """high camera behind the phone side (-y), looking down at the table centre at about 55 deg."""
    cx, cy = (table["x"][0] + table["x"][1]) / 2, (table["y"][0] + table["y"][1]) / 2
    eye = np.array([cx, table["y"][0] - 0.55, 1.35])
    f = np.array([cx, cy, 0.0]) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    Tc = np.eye(4)
    Tc[:3, :3] = np.stack([r, -u, f], 1)  # OpenCV camera axes (x right, y down, z forward)
    Tc[:3, 3] = eye
    K = [[600.0, 0, 640], [0, 600.0, 360], [0, 0, 1]]
    return dict(K=K, T_cam_to_table=Tc.tolist(), wh=[1280, 720], note="high overview behind the phone")


def layout_check(scene: dict, img: np.ndarray, K, T, out: Path) -> None:
    """fitted meshes drawn over the real start frame, with their fit IoU."""
    im = img.copy()
    for o in scene["objects"]:
        m = _load_obj(out, o)
        uv, z = project(m.vertices + np.array(o["pos"]), K, T)
        if (z <= 0).any():
            continue
        col = (
            (0, 255, 0)
            if o["role"] in ("task", "stack")
            else ((255, 128, 0) if o["role"] == "container" else (0, 200, 255))
        )
        e = m.edges_unique
        if len(e) > 3000:
            cv2.polylines(im, [cv2.convexHull(uv.astype(np.float32)).astype(np.int32)], True, col, 1, cv2.LINE_AA)
        else:
            for a, b in e:
                cv2.line(im, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), col, 1, cv2.LINE_AA)
        c = uv.mean(0).astype(int)
        cv2.putText(
            im,
            f"{o['id']} {o['fit_iou']:.2f}",
            (int(c[0]) - 30, int(c[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out / "layout_check.jpg"), im)


# ------------------------------------------------------------------ one episode
def build_episode(ep: dict, cfg: SceneConfig, force: bool = False) -> dict | None:
    t0 = time.time()
    eid, v = ep["id"], ep["video"]
    out = twin_dir(eid)
    if (out / "scene.json").exists() and not force:
        return None
    if not (plate_dir(v) / "plate.png").exists():
        raise FileNotFoundError(f"no static plate for {v}: run cleantables.twin.plate first")
    shutil.rmtree(out / "usd", ignore_errors=True)
    (out / "usd").mkdir(parents=True, exist_ok=True)
    G = camera(v)
    K, T = G["K"], G["T"]
    flags = []
    objs0, k0 = start_objects(ep, flags, cfg)
    img0 = frame(v, k0)
    cv2.imwrite(str(out / "start_real.jpg"), img0)
    ctx = fit_contexts(v, objs0, G, Detections(v), tracks(v), hand_masks(v))
    placed, s_vid, probes = place_objects(objs0, ctx, K, T, flags)
    scene_objs = write_objects(ep, placed, out, cfg)
    depenetrate(scene_objs, out, flags)
    drop(scene_objs, out, flags)
    ids = {o["id"] for o in scene_objs}
    for o in scene_objs:
        if o["dest"] is not None and o["dest"] not in ids:
            flags.append(f"{o['id']}: destination {o['dest']} not on the table at the start -> role set to obstacle")
            o["role"], o["dest"] = "obstacle", None

    ext = table_extent(T, K, G["table"])
    ox = np.array([o["pos"][:2] for o in scene_objs]) if scene_objs else np.zeros((1, 2))
    ext = [
        [min(ext[0][0], ox[:, 0].min() - 0.1), max(ext[0][1], ox[:, 0].max() + 0.1)],
        [min(ext[1][0], ox[:, 1].min() - 0.1), max(ext[1][1], ox[:, 1].max() + 0.1)],
    ]
    seen_frac = table_texture(v, K, T, ext, out)
    write_table_usd(out / "table.usda", "./table_tex.png", ext)
    n_bg = write_background(v, K, T, ext, out / "background.usda", cfg.background_emissive)
    table = dict(
        x=ext[0],
        y=ext[1],
        top_z=0.0,
        texture="table_tex.png",
        tex_res_m=TABLE_TEX_RES,
        tex_source="static plate (plate.py)",
        tex_observed_frac=round(seen_frac, 3),
    )

    idx = {o["id"]: i for i, o in enumerate(scene_objs)}
    tasks = [(idx[o["id"]], idx[o["dest"]]) for o in scene_objs if o["role"] == "task"]
    if ep["task"] == "stack_boxes":
        stack = [idx[o["id"]] for o in scene_objs if o["role"] == "stack"]
        tasks = [(a, b) for a in stack for b in stack if a != b]
    poses, shoulder = shoulder_poses(v, table, scene_objs, tasks, flags, cfg, cam_xyz=T[:3, 3])
    yaw = math.radians(shoulder["yaw_deg"])

    def reachable(a, xy):
        return bool(reach_b(poses[a][0][:2], xy, yaw))

    n_reach = sum(
        1
        for k, kc in tasks
        if any(reachable(a, scene_objs[k]["pos"][:2]) and reachable(a, scene_objs[kc]["pos"][:2]) for a in "ab")
    )
    reach = {o["id"]: {a: reachable(a, o["pos"][:2]) for a in "ab"} for o in scene_objs}
    for o in scene_objs:
        if o["role"] in ("task", "container", "stack") and not any(reach[o["id"]].values()):
            flags.append(f"{o['id']}: outside both arms' Lift workspace")
    if tasks and n_reach < len(tasks):
        flags.append(f"arms: {n_reach}/{len(tasks)} tasks have object AND destination inside one arm's Lift workspace")

    scene = dict(
        episode=eid,
        video=v,
        keyframe=k0,
        t_start=ep["start_s"],
        t_end=ep["end_s"],
        task=ep["task"],
        instruction=ep["instruction"],
        labelled_objects=ep.get("objects"),
        K=K.tolist(),
        T_cam_to_table=T.tolist(),
        image_wh=[IMG_W, IMG_H],
        frame="generic table frame (det3d): z up (table normal), table top z = 0, metres",
        table=table,
        objects=scene_objs,
        poses=poses,
        layout="measured",
        arms=dict(
            side=shoulder["side"],
            yaw_deg=shoulder["yaw_deg"],
            shoulder_estimate=shoulder,
            reach=reach,
            tasks_one_arm=f"{n_reach}/{len(tasks)}",
            note="bases where the person's shoulders are (their side of the table, from their body points), "
            "reaching across the table; Lift height convention. Workspace test for any side: "
            "scene.reach_b(base, xy, yaw)",
        ),
        background=dict(
            usd="background.usda",
            source=f"plates/{v}",
            triangles=int(n_bg),
            note="visual only, emissive static plate on its static MoGe depth; stretches behind objects from "
            "non-phone cameras",
        ),
        cameras=dict(
            main=dict(K=K.tolist(), T_cam_to_table=T.tolist(), wh=[IMG_W, IMG_H], note="the real phone camera"),
            overview=overview_pose(table),
            wrist=dict(
                mount="ee_link/palm_link", note="see cameras.py WRIST (pose in the palm frame, verified by rendering)"
            ),
        ),
        flags=flags
        + [
            "physics policy: boxes / lid 3.0 kg mu 0.9/0.8, containers 2.5 kg; tools / fruit real masses "
            "(real boxes are lighter, mass_real per object)"
        ],
        build_s=None,
        video_scale=dict(
            value=round(s_vid, 4),
            probes=[(a, round(b, 3), round(c, 3)) for a, b, c in probes],
            note="isotropic factor on library meshes (flat objects: x/y only) for this video's MoGe metric scale",
        ),
        notes=[
            "object poses: tracked 3D boxes (MoGe metric depth, not tape-measured) refined by silhouette fit "
            "through the phone camera",
            "yaw (and the face-up rotation of boxes) is baked into each USD's vertices; "
            "pos = footprint centre, bottom height",
            "usd paths are relative to this folder; layout='measured' means pos is the measured start layout",
            "tools: Hunyuan3D-2.1 meshes sized and carved by their real top-down silhouette; "
            "boxes: textured cuboids from the video",
            f"camera height {T[2, 3]:.2f} m above the table",
        ],
    )
    scene["build_s"] = round(time.time() - t0, 1)
    write_json(scene, out / "scene.json")
    layout_check(scene, img0, K, T, out)
    print(f"episode {eid}: {len(scene_objs)} objects, {len(flags)} flags, {scene['build_s']} s", flush=True)
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the physics twin of each episode (data/twins/<episode>/).")
    ap.add_argument("--task", default=None, help="only episodes of this task (tools, fruit, stack_boxes)")
    ap.add_argument("--ids", nargs="*", help="only these episode ids")
    ap.add_argument("--force", action="store_true", help="rebuild twins that already exist")
    ap.add_argument("--config", default=None, help="config file (default configs/twin.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config).scene
    eps = [e for e in episodes(args.task) if not args.ids or e["id"] in args.ids]
    t0, n = time.time(), 0
    for e in eps:
        n += build_episode(e, cfg, args.force) is not None
    print(f"built {n} episodes in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
