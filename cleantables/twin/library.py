"""Object library: one textured, physics-ready mesh per physical object, shared by every episode.

Entries built from the video through the phone camera (det3d K, T) and the video's static plate:
  cuboid  boxes, toolbox lid: pose and size fitted to the mask silhouette; faces seen front-on get the real pixels by
          homography, unseen faces copy the opposite face (listed in meta.json);
  tray    open toolbox: the same fit, outer walls from the video, inside = the measured interior colour;
  flat    tools: SAM 3 mask refined against the plate, rectified onto the table plane and extruded to the category
          thickness prior (the phone cannot measure 1-2 cm); texture = the orthophoto. hunyuan.py later replaces the
          tool meshes with Hunyuan3D-2.1 shapes, carved and sized by these silhouettes;
  pillow  ginger: the same silhouette as a rounded slab;
  apple   revolved apple profile sized to the measured apple, texture = rectified top view (Hunyuan's apple shapes
          failed);
  hy3d    fruit bowl, bananas: Hunyuan3D-2.1 painted meshes of the same objects (data/twins/hy3d/<key>/paint/).
The source sighting of each entry is picked automatically: the largest complete, unoccluded, sharp SAM 3 mask while
the object rests.

  python -m cleantables.twin.library [--only key,key]       (CPU, a few minutes)
Out data/twins/library/<key>/: mesh.npz (V, F, UV), texture.png, meta.json, source.jpg; library/sheet.jpg.
"""

from __future__ import annotations

import argparse
import json
import math

import cv2
import numpy as np
import trimesh

from cleantables.twin import fitting, meshes
from cleantables.twin.identity import CATALOGUE, resolve_video
from cleantables.twin.inputs import (
    HUNYUAN,
    IMG_H,
    IMG_W,
    LIBRARY,
    Detections,
    camera,
    depth,
    episodes,
    events,
    frame,
    plate_dir,
    tracks,
    upsample_mask,
    write_json,
)
from cleantables.twin.projection import lab, ortho, plane_hit

FRUIT_VIDEO = "IMG_2516"
BOX_KEYS = ["amazon_box", "cipriata_box", "lotus_box", "black_box", "orange_box", "yellow_box"]
FLAT_TOOL_KEYS = ["wrench", "utility_knife", "screwdriver"]
# painted Hunyuan3D-2.1 meshes of the fruit-task objects: key -> longest side in metres, measured from the video (MoGe)
HUNYUAN_FRUIT = {"fruit_bowl": 0.1992428, "banana_a": 0.14026962, "banana_b": 0.14026962}
YUP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def task_videos(task: str) -> list[str]:
    return sorted({e["video"] for e in episodes(task)})


# ------------------------------------------------------------------ choosing the source sighting
def candidates(videos, key_filter, first_rest=False, max_top=9.0) -> list[dict]:
    """sightings (video, track, detection, frame, ...) of tracks whose identity passes key_filter(key), resting,
    hand-free and away from the image border."""
    out = []
    for v in videos:
        E, R = events(v), tracks(v)
        res = {r["track"]: r for r in resolve_video(E["objects"], v)}
        for o in E["objects"]:
            r = res[o["id"]]
            if not key_filter(r["key"]):
                continue
            track = R["objects"][o["src"]]
            rests = [(q["f0"], q["f1"]) for q in o["rests"] if q["pose"]["top"] <= max_top]
            if first_rest:  # boxes: the initial layout of the video, nothing stacked yet
                rests = [(q["f0"], q["f1"]) for q in o["rests"][:1] if q["f0"] <= 10 and q["pose"]["top"] <= max_top]
            for k, d in zip(track["sight_k"].tolist(), track["det"], strict=False):
                b = R["det_box"].get(d)
                if b is None or b.get("hand_frac", 1) > 0.0 or not any(f0 <= k <= f1 for f0, f1 in rests):
                    continue
                x0, y0, x1, y1 = b["bbox"]
                if x0 <= 4 or y0 <= 4 or x1 >= IMG_W - 4 or y1 >= IMG_H - 4:
                    continue
                out.append(
                    dict(
                        v=v,
                        track=o["id"],
                        src=o["src"],
                        det=d,
                        k=int(k),
                        box=b,
                        area=float(b["n"]) * (0.5 if r["uncertain"] else 1.0),
                        key=r["key"],
                        rgb=o["rgb"],
                        uncertain=r["uncertain"],
                    )
                )
    return out


def occlusion(c: dict, cache: dict, pad: int = 6) -> tuple[float, np.ndarray]:
    """fraction of the mask covered by other detections within `pad` px that are at least 2 cm nearer to the camera
    (neighbours behind or beside the object do not count), and the mask."""
    v = c["v"]
    if v not in cache:
        cache[v] = (Detections(v), tracks(v), depth(v), camera(v)["scale"])
    D, R, D10, sc = cache[v]
    m = D.mask(c["det"])
    near_ring = cv2.dilate(m.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool)
    other = np.zeros_like(m)
    for d in R["kept"].get(c["k"], []):
        if d != c["det"]:
            other |= D.mask(d)
    near = near_ring & other & ~m
    if not near.any():
        return 0.0, m
    dk = np.asarray(D10[c["k"]], np.float32) * sc[c["k"]]
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    d_obj = float(np.median(dk[edge])) if edge.any() else float(np.median(dk[m]))
    occ = near & (dk < d_obj - 0.02)
    return float(occ.sum() / max(1, m.sum())), m


def sharpness(c: dict, m: np.ndarray) -> float:
    img = cv2.cvtColor(frame(c["v"], c["k"]), cv2.COLOR_BGR2GRAY)
    mm = cv2.dilate(upsample_mask(m).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return float(cv2.Laplacian(img, cv2.CV_32F)[mm].var())


def complete(c: dict, cands: list[dict]) -> bool:
    """not cut by occlusion or the frame: area >= 85 % of the 75th-percentile area of the same track."""
    a = [x["area"] for x in cands if x["v"] == c["v"] and x["track"] == c["track"]]
    return c["area"] >= 0.85 * np.percentile(a, 75)


def pick(cands: list[dict], n: int = 1, cache: dict | None = None) -> list[dict]:
    """best sightings: complete and unoccluded, ranked by area x sqrt(sharpness) (close, sharp views)."""
    cache = {} if cache is None else cache
    pool = sorted([c for c in cands if complete(c, cands)], key=lambda c: -c["area"])[:80]
    ok = []
    for c in pool:
        occ, m = occlusion(c, cache)
        if occ < 0.01:
            c["mask"] = m
            c["sharp"] = sharpness(c, m)
            ok.append(c)
    ok.sort(key=lambda c: -(c["area"] * math.sqrt(c["sharp"])))
    return ok[:n]


# ------------------------------------------------------------------ entries on disk
def save_entry(key: str, mesh: trimesh.Trimesh, tex: np.ndarray, meta: dict, src_img: np.ndarray | None = None) -> None:
    d = LIBRARY / key
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "mesh.npz",
        V=np.asarray(mesh.vertices, np.float32),
        F=np.asarray(mesh.faces, np.int32),
        UV=np.asarray(mesh.visual.uv, np.float32),
    )
    cv2.imwrite(str(d / "texture.png"), tex)
    if src_img is not None:
        cv2.imwrite(str(d / "source.jpg"), src_img)
    meta.update(
        extents=[round(float(x), 4) for x in mesh.extents],
        key=key,
        **{k: v for k, v in CATALOGUE[key].items() if k not in meta},
    )
    write_json(meta, d / "meta.json")
    print(f"library {key}: extents {meta['extents']} {meta.get('source', '')}", flush=True)


def load_entry(key: str) -> tuple[trimesh.Trimesh, object, dict]:
    """(mesh with UVs, texture path, meta)."""
    d = LIBRARY / key
    z = np.load(d / "mesh.npz")
    m = trimesh.Trimesh(z["V"], z["F"], process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=z["UV"])
    return m, d / "texture.png", json.loads((d / "meta.json").read_text())


def source_crop(img: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, xs.min() - 40), min(IMG_W, xs.max() + 40)
    y0, y1 = max(0, ys.min() - 40), min(IMG_H, ys.max() + 40)
    im = img.copy()
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnt, -1, (0, 255, 0), 1)
    c = im[y0:y1, x0:x1]
    s = 300 / max(c.shape[:2])
    c = cv2.resize(c, None, fx=s, fy=s)
    cv2.putText(c, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return c


# ------------------------------------------------------------------ cuboids and the tray
def build_cuboid(key: str, c: dict, tray: bool = False, interior=None) -> None:
    v, k = c["v"], c["k"]
    G = camera(v)
    K, T = G["K"], G["T"]
    img = frame(v, k)
    m = upsample_mask(c["mask"])
    b = c["box"]
    H0 = max(0.02, float(b["top"]))
    if key == "toolbox_lid":  # a silhouette cannot measure 2 cm: the lid thickness is a prior
        H0 = 0.02
    fit = fitting.fit_cuboid(m, K, T, (b["x"], b["y"], b["yaw"], b["L"], b["W"], H0))
    if key == "toolbox_lid":
        fit = fitting.fit_cuboid(
            m, K, T, (fit["x"], fit["y"], fit["yaw"], fit["L"], fit["W"], 0.02), fix_dims=(fit["L"], fit["W"], 0.02)
        )
        fit.update(H=0.02)
    if fit["L"] < fit["W"]:  # canonical: L >= W, x = long side
        fit["L"], fit["W"], fit["yaw"] = fit["W"], fit["L"], fit["yaw"] + math.pi / 2
    own = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    faces = {f: t for f, (t, _) in meshes.face_textures(img, own, K, T, fit).items()}
    if tray:
        faces.pop("top", None)
    mesh, atlas, src = meshes.cuboid_mesh(fit["L"], fit["W"], fit["H"], faces)
    meta = dict(
        source=f"{v} k={k} track={c['track']} det={c['det']}",
        fit=fit,
        faces=src,
        fit_iou=round(fit["iou"], 3),
        note="textured cuboid from the video (faces by homography from the source frame)",
    )
    if tray:
        mesh, atlas = meshes.tray_mesh(fit, atlas, interior)
        meta["note"] = (
            "open tray: outside walls from the video, inside = measured interior colour; lid separate (toolbox_lid)"
        )
        meta["interior_bgr"] = [int(x) for x in interior]
    save_entry(key, mesh, atlas, meta, source_crop(img, m, f"{key}: {v} k{k} IoU {fit['iou']:.2f}"))


def build_boxes(keys: list[str]) -> None:
    videos = sorted(set(task_videos("tools") + task_videos("stack_boxes")))
    cache = {}
    for key in keys:
        s = pick(candidates(videos, lambda k, key=key: k == key, first_rest=True, max_top=0.2), 1, cache)
        if not s:  # seen in one video only (yellow box): any resting phase on the table
            s = pick(candidates(videos, lambda k, key=key: k == key, max_top=0.2), 1, cache)
        if s:
            build_cuboid(key, s[0])
        else:
            print(f"library {key}: no clean sighting", flush=True)


def build_toolbox() -> None:
    """tray: best sighting of the open tray, interior colour = median of its eroded mask.
    lid: a white-box sighting at a keyframe where the open tray is visible elsewhere (so it is the lid, not the closed
    box)."""
    videos = task_videos("tools")
    cache = {}
    c = pick(candidates(videos, lambda k: k == "toolbox", max_top=0.09), 1, cache)[0]
    img = frame(c["v"], c["k"])
    inner = cv2.erode(upsample_mask(c["mask"]).astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    interior = np.median(img[inner], 0) if inner.sum() > 100 else np.array([235, 240, 242])
    build_cuboid("toolbox", c, tray=True, interior=interior)
    trays = {(x["v"], x["k"]) for x in candidates(videos, lambda k: k == "toolbox")}
    lids = [x for x in candidates(videos, lambda k: k == "white_box", max_top=0.05) if (x["v"], x["k"]) in trays]
    s = pick(lids, 1, cache)
    if s:
        build_cuboid("toolbox_lid", s[0])
    else:
        print("library toolbox_lid: no clean sighting", flush=True)


# ------------------------------------------------------------------ flat objects
def refine_mask(img: np.ndarray, plate: np.ndarray, m_lo: np.ndarray, thr: int = 40) -> np.ndarray:
    """SAM 3 mask (270x480) -> 960x540 silhouette by GrabCut on a crop: sure foreground = eroded mask, probable
    foreground = the mask plus its ring where the frame differs from the static plate, probable background = the rest
    of the ring."""
    m = upsample_mask(m_lo)
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = max(0, xs.min() - 30), min(IMG_W, xs.max() + 31), max(0, ys.min() - 30), min(IMG_H, ys.max() + 31)
    crop, mc = img[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1].astype(np.uint8)
    gc = np.full(mc.shape, cv2.GC_BGD, np.uint8)
    ring = cv2.dilate(mc, np.ones((13, 13), np.uint8)).astype(bool)
    diff = np.abs(crop.astype(np.int16) - plate[y0:y1, x0:x1].astype(np.int16)).max(2) > thr
    gc[ring] = cv2.GC_PR_BGD
    gc[ring & diff] = cv2.GC_PR_FGD
    gc[mc.astype(bool)] = cv2.GC_PR_FGD
    gc[cv2.erode(mc, np.ones((5, 5), np.uint8)).astype(bool)] = cv2.GC_FGD
    cv2.grabCut(crop, gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    s = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    s = largest_component(s)
    out = np.zeros((IMG_H, IMG_W), bool)
    out[y0:y1, x0:x1] = s.astype(bool)
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab_, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n > 1:
        return (lab_ == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(mask.dtype)
    return mask


def build_flat(key: str, c: dict, pillow: bool = False) -> None:
    v, k = c["v"], c["k"]
    plate = cv2.imread(str(plate_dir(v) / "plate.png"))
    G = camera(v)
    K, T = G["K"], G["T"]
    img = frame(v, k)
    thick = CATALOGUE[key]["thick"]
    sil = refine_mask(img, plate, c["mask"])
    ys, xs = np.nonzero(sil)
    P = plane_hit(np.c_[xs, ys].astype(float), K, T, z=thick)
    x0, y0 = P[:, :2].min(0) - 0.02
    x1, y1 = P[:, :2].max(0) + 0.02
    orth, (mx, my) = ortho(img, K, T, x0, x1, y0, y1, z=thick)
    so = cv2.remap(sil.astype(np.uint8) * 255, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 127
    so = cv2.morphologyEx(so.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(so.astype(np.uint8))
    so = lab_ == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    rs, cs = np.nonzero(so)
    so = so[rs.min() : rs.max() + 1, cs.min() : cs.max() + 1]
    orth = orth[rs.min() : rs.max() + 1, cs.min() : cs.max() + 1].copy()
    core = cv2.erode(so.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    mean = np.median(orth[core if core.sum() > 20 else so], 0)
    tex = orth.copy()
    tex[~so] = mean.astype(np.uint8)
    tex = cv2.resize(tex, (tex.shape[1] * 2, tex.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    mesh = meshes.slab_mesh(so, thick, pillow=pillow)
    meta = dict(
        source=f"{v} k={k} track={c['track']} det={c['det']}",
        thickness_prior=thick,
        silhouette_px=int(so.sum()),
        note="flat object: real top-down silhouette (SAM 3 mask refined by plate difference, rectified to the table "
        "plane) extruded to the thickness prior; texture = rectified real pixels",
        mean_bgr=[int(x) for x in mean],
        redness=float(lab(mean[::-1])[1]),
    )
    save_entry(key, mesh, tex, meta, source_crop(img, sil, f"{key}: {v} k{k}"))


def build_tools() -> None:
    """pliers: one frame where both pliers are isolated (same lighting); the redder one (a*) is pliers_a (red grips)."""
    videos = task_videos("tools")
    cache = {}
    pc = candidates(videos, lambda k: k == "pliers")
    by_frame: dict[tuple, list] = {}
    for c in pc:
        by_frame.setdefault((c["v"], c["k"]), []).append(c)
    best = None
    for _, cs in sorted(by_frame.items(), key=lambda kv: -sum(c["area"] for c in kv[1])):
        if len(cs) != 2 or not all(complete(c, pc) for c in cs):
            continue
        occ = [occlusion(c, cache) for c in cs]
        if all(o < 0.01 for o, _ in occ):
            for c, (_, m) in zip(cs, occ, strict=False):
                c["mask"] = m
            best = cs
            break
    for c in best:
        mm = upsample_mask(c["mask"])
        c["red"] = float(cv2.cvtColor(frame(c["v"], c["k"]), cv2.COLOR_BGR2LAB)[mm][:, 1].astype(float).mean())
    best.sort(key=lambda c: -c["red"])
    build_flat("pliers_a", best[0])
    build_flat("pliers_b", best[1])
    for key in FLAT_TOOL_KEYS:
        s = pick(candidates(videos, lambda k, key=key: k == key), 1, cache)
        if s:
            build_flat(key, s[0])
        else:
            print(f"library {key}: no clean sighting", flush=True)


# ------------------------------------------------------------------ fruit
def rest_pose(m: trimesh.Trimesh, target_h: float, flat: bool) -> trimesh.Trimesh:
    """stable resting pose whose height best matches target_h (flat objects: the most probable pose); long horizontal
    axis -> +x, footprint centred, bottom at z = 0."""
    T, p = trimesh.poses.compute_stable_poses(m, n_samples=1)
    cands = [(T[i], p[i]) for i in range(min(10, len(T))) if p[i] >= 0.01] or [(T[0], p[0])]
    if flat or not np.isfinite(target_h) or target_h < 0.02:
        Tb = cands[0][0]
    else:

        def cost(tp):
            mm = m.copy()
            mm.apply_transform(tp[0])
            return abs(math.log(max(mm.extents[2], 1e-3) / target_h)) - 0.3 * tp[1]

        Tb = min(cands, key=cost)[0]
    m2 = m.copy()
    m2.apply_transform(Tb)
    xy = m2.vertices[:, :2] - m2.vertices[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    ax = v[:, np.argmax(w)]
    m2.apply_transform(trimesh.transformations.rotation_matrix(-math.atan2(ax[1], ax[0]), (0, 0, 1)))
    lo, hi = m2.bounds
    m2.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    return m2


def build_hunyuan_fruit(key: str) -> None:
    """painted Hunyuan3D-2.1 mesh (y-up, model units) scaled to the measured longest side, at rest."""
    d = HUNYUAN / key / "paint"
    m = trimesh.load(d / "textured_mesh.obj", force="mesh", process=False)
    m.apply_transform(YUP_TO_ZUP)
    m.apply_scale(float(HUNYUAN_FRUIT[key] / m.extents.max()))
    m = rest_pose(m, float("nan"), True)
    img = cv2.imread(str(d / "textured_mesh.jpg"))
    note = "Hunyuan3D-2.1 textured mesh of the same physical object; size = longest side measured from the video (MoGe)"
    if key == "fruit_bowl":  # the paint step coloured the inside teal; the real bowl is plain white
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 0.1
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.05 + 10, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        note += "; paint hue removed (teal interior is a Hunyuan paint artefact, the real bowl is white)"
    save_entry(key, m, img, dict(source=f"Hunyuan3D-2.1 shape + paint: {d}", fit=None, note=note))


def build_apple() -> None:
    """revolved apple profile with a stalk dimple, sized to the tracked 3D box; texture = rectified top view."""
    c = pick(candidates([FRUIT_VIDEO], lambda k: k == "green_apple"), 1)[0]
    v, k = c["v"], c["k"]
    G = camera(v)
    K, T = G["K"], G["T"]
    img = frame(v, k)
    sil = refine_mask(img, cv2.imread(str(plate_dir(v) / "plate.png")), c["mask"])
    ys, xs = np.nonzero(sil)
    b = c["box"]
    Hm = float(np.clip(max(b["top"], 0.85 * 0.5 * (b["L"] + b["W"])), 0.045, 0.075))
    P = plane_hit(np.c_[xs, ys].astype(float), K, T, z=Hm / 2)
    D = float(
        np.clip(0.5 * (b["L"] + b["W"]), 0.05, 0.08)
    )  # the plane-projected blob over-measures; the 3D box does not
    x0, y0 = P[:, :2].mean(0) - D * 0.6
    x1, y1 = P[:, :2].mean(0) + D * 0.6
    orth, (mx, my) = ortho(img, K, T, x0, x1, y0, y1, z=Hm * 0.8)
    so = cv2.remap(sil.astype(np.uint8) * 255, mx, my, cv2.INTER_LINEAR) > 127
    mean = np.median(orth[so], 0) if so.any() else np.array([60, 160, 150])
    tex = orth.copy()
    tex[~so] = mean.astype(np.uint8)
    tex = cv2.resize(tex, (256, 256), interpolation=cv2.INTER_CUBIC)
    t = np.linspace(0, math.pi, 25)
    r = 0.5 * D * np.sin(t) * (1 + 0.08 * np.sin(t))
    z = 0.5 * Hm * (1 - np.cos(t))
    z[-3:] -= np.array([0.0, 0.004, 0.008])
    profile = np.c_[np.r_[0, r[1:-1], 0], np.r_[0, z[1:-1], z[-1] - 0.006]]
    m = trimesh.creation.revolve(profile, sections=40)
    m.apply_translation((0, 0, -m.bounds[0][2]))
    V = m.vertices
    m.visual = trimesh.visual.TextureVisuals(uv=np.c_[(V[:, 0] / (1.2 * D)) + 0.5, (V[:, 1] / (1.2 * D)) + 0.5])
    meta = dict(
        source=f"{v} k={k} track={c['track']} det={c['det']}",
        diameter=D,
        height=Hm,
        note="procedural apple shape (Hunyuan apple shapes failed) sized to the measured apple; "
        "texture = rectified real top view",
    )
    save_entry("green_apple", m, tex, meta)


def build_fruit() -> None:
    for key in HUNYUAN_FRUIT:
        build_hunyuan_fruit(key)
    build_apple()
    s = pick(candidates([FRUIT_VIDEO], lambda k: k == "ginger"), 1)
    if s:
        build_flat("ginger", s[0], pillow=True)
    else:
        print("library ginger: no clean sighting", flush=True)


def contact_sheet() -> None:
    """source crop over texture of every entry -> library/sheet.jpg."""
    tiles = []
    for d in sorted(p for p in LIBRARY.iterdir() if p.is_dir()):
        if not (d / "source.jpg").exists():
            continue
        a, t = cv2.imread(str(d / "source.jpg")), cv2.imread(str(d / "texture.png"))
        a = cv2.resize(a, (300, int(300 * a.shape[0] / a.shape[1])))
        t = cv2.resize(t, (300, int(300 * t.shape[0] / t.shape[1])))
        col = np.full((440, 300, 3), 255, np.uint8)
        col[: min(220, a.shape[0])] = a[:220]
        col[220 : 220 + min(220, t.shape[0])] = t[:220]
        cv2.putText(col, d.name, (4, 434), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(col)
    blank = np.full((440, 300, 3), 255, np.uint8)
    rows = [np.hstack(tiles[i : i + 5] + [blank] * (5 - len(tiles[i : i + 5]))) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(LIBRARY / "sheet.jpg"), np.vstack(rows))


STEPS = {"tools": build_tools, "toolbox": build_toolbox, "boxes": lambda: build_boxes(BOX_KEYS), "fruit": build_fruit}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the object library (one mesh per physical object) from the videos.")
    ap.add_argument("--only", default="", help=f"comma-separated steps out of {','.join(STEPS)} (default: all)")
    args = ap.parse_args()
    for name in args.only.split(",") if args.only else STEPS:
        STEPS[name]()
    contact_sheet()


if __name__ == "__main__":
    main()
