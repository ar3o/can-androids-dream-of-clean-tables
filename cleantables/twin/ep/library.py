"""Object library: one textured, physics-ready mesh per physical object of the recorded episodes, built from the video itself.

  source scripts/env.sh; $SIM_PY -m cleantables.twin.ep.library --task tools      (CPU, ~1 min)

Per library key (ident.LIBRARY) the source sighting is chosen automatically over all videos of the task: the SAM 3 detection (generic
pipeline, sam3_vinv.npz) with the largest mask that has no hand on it, does not touch another detection or the image border, and is in a
resting phase (events.json rest). Then, through the real phone camera (det3d K, T) and the video's static plate (cleantables.twin.ep.plate):
  cuboid  (boxes, toolbox lid): pose + size fitted to the mask silhouette (fit_cuboid: projected cuboid vs mask IoU, Nelder-Mead); each face
          seen front-on gets the real pixels by homography (2 mm/texel); unseen faces = copy of the opposite face (stated in meta.json);
  tray    (toolbox, open): the same fit; outside walls textured from the video, inside = the interior colour measured where the lid is off;
          visual open box, collision = floor + 4 wall slabs;
  flat    (pliers, wrench, utility knife, screwdriver): silhouette = SAM 3 mask refined at 960x540 by the difference to the static plate,
          rectified onto the table plane (orthophoto, 2 mm/px); mesh = that silhouette extruded to the category thickness prior (the thickness
          is a prior: the phone cannot measure 1-2 cm reliably); texture = the orthophoto (real top-down appearance); collision = CoACD parts.
Fruit (banana, bowl, apple, ginger) are not built here; they are meshed with Hunyuan3D-2.1.
Out: data/twin_table/ep/library/<key>/{mesh.npz (V, F, UV), texture.png, meta.json, source.jpg} + library/sheet.jpg.
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import trimesh

from cleantables.twin.ep import geometry as geo
from cleantables.twin.ep.common import LIB, OUT, Dets, episodes, events, frame, geom, jdump, lab, tracks
from cleantables.twin.ep.identity import LIBRARY, resolve_video

TASK_VIDEOS = {
    "tools": sorted({e["video"] for e in episodes("tools")}),
    "stack_boxes": sorted({e["video"] for e in episodes("stack_boxes")}),
}


def up(m):
    return cv2.resize(m.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)


def candidates(videos, key_filter, first_rest=False, max_top=9.0):
    """-> list of sightings dict(v, track, det, k, area, ...) for tracks whose identity passes key_filter(key)."""
    out = []
    for v in videos:
        E = events(v)
        R = tracks(v)
        res = {r["track"]: r for r in resolve_video(E["objects"], v)}
        dets = None
        for o in E["objects"]:
            r = res[o["id"]]
            if not key_filter(r["key"]):
                continue
            T_ = R["objects"][o["src"]]
            rests = [(q["f0"], q["f1"]) for q in o["rests"] if q["pose"]["top"] <= max_top]
            if first_rest:                  # boxes: the video's initial layout (every box on the table, nothing stacked yet)
                rests = [(q["f0"], q["f1"]) for q in o["rests"][:1] if q["f0"] <= 10 and q["pose"]["top"] <= max_top]
            for j, (k, d) in enumerate(zip(T_["sight_k"].tolist(), T_["det"])):
                b = R["det_box"].get(d)
                if b is None or b.get("hand_frac", 1) > 0.0 or not any(f0 <= k <= f1 for f0, f1 in rests):
                    continue
                x0, y0, x1, y1 = b["bbox"]
                if x0 <= 4 or y0 <= 4 or x1 >= 956 or y1 >= 536:
                    continue
                out.append(dict(v=v, track=o["id"], src=o["src"], det=d, k=int(k), box=b, area=float(b["n"]) * (0.5 if r["uncertain"] else 1.0),
                                key=r["key"], rgb=o["rgb"], uncertain=r["uncertain"]))
        del dets
    return out


def isolated(c, dets_cache, pad=6):
    """Occlusion score: fraction of the object's mask size covered by other detections of that keyframe that lie within `pad` px of it and
    are nearer to the camera there (MoGe depth 2 cm in front of the object's own boundary depth); adjacent objects behind / beside it do not
    count. 0 = unoccluded."""
    v = c["v"]
    if v not in dets_cache:
        from cleantables.twin.ep.common import depth
        dets_cache[v] = (Dets(v), tracks(v), depth(v), geom(v)["scale"])
    D, R, D10, sc = dets_cache[v]
    m = D.mask(c["det"])
    md = cv2.dilate(m.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool)
    other = np.zeros_like(m)
    for d in R["kept"].get(c["k"], []):
        if d != c["det"]:
            other |= D.mask(d)
    near = md & other & ~m
    if not near.any():
        return 0.0, m
    dk = np.asarray(D10[c["k"]], np.float32) * sc[c["k"]]
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    d_obj = float(np.median(dk[edge])) if edge.any() else float(np.median(dk[m]))
    occ = near & (dk < d_obj - 0.02)
    return float(occ.sum() / max(1, m.sum())), m


def sharpness(c, m):
    img = cv2.cvtColor(frame(c["v"], c["k"]), cv2.COLOR_BGR2GRAY)
    mm = cv2.dilate(up(m).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return float(cv2.Laplacian(img, cv2.CV_32F)[mm].var())


def complete(c, cands):
    """The mask is not cut (occluded / partly out of view): area >= 85 % of the 75th percentile area of the same track."""
    a = [x["area"] for x in cands if x["v"] == c["v"] and x["track"] == c["track"]]
    return c["area"] >= 0.85 * np.percentile(a, 75)


def pick(cands, n=1, dets_cache=None):
    """Best sightings: complete, isolated, then the highest area x sqrt(sharpness) (sharp, close views)."""
    dets_cache = {} if dets_cache is None else dets_cache
    pool = sorted([c for c in cands if complete(c, cands)], key=lambda c: -c["area"])[:80]
    ok = []
    for c in pool:
        ov, m = isolated(c, dets_cache)
        if ov < 0.01:
            c["mask"] = m
            c["sharp"] = sharpness(c, m)
            ok.append(c)
    ok.sort(key=lambda c: -(c["area"] * math.sqrt(c["sharp"])))
    return ok[:n]


def save_entry(key, mesh, tex, meta, src_img=None):
    d = LIB / key
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "mesh.npz", V=np.asarray(mesh.vertices, np.float32), F=np.asarray(mesh.faces, np.int32),
                        UV=np.asarray(mesh.visual.uv, np.float32))
    cv2.imwrite(str(d / "texture.png"), tex)
    if src_img is not None:
        cv2.imwrite(str(d / "source.jpg"), src_img)
    meta.update(extents=[round(float(x), 4) for x in mesh.extents], key=key, **{k: v for k, v in LIBRARY[key].items() if k not in meta})
    jdump(meta, d / "meta.json")
    print("library", key, meta["extents"], meta.get("source", ""), flush=True)


def load_entry(key):
    d = LIB / key
    z = np.load(d / "mesh.npz")
    m = trimesh.Trimesh(z["V"], z["F"], process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=z["UV"])
    import json
    return m, d / "texture.png", json.load(open(d / "meta.json"))


def src_crop(img, mask, label):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = max(0, xs.min() - 40), min(960, xs.max() + 40), max(0, ys.min() - 40), min(540, ys.max() + 40)
    im = img.copy()
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnt, -1, (0, 255, 0), 1)
    c = im[y0:y1, x0:x1]
    s = 300 / max(c.shape[:2])
    c = cv2.resize(c, None, fx=s, fy=s)
    cv2.putText(c, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return c


# ------------------------------------------------------------------ cuboids
def build_cuboid(key, c, tray=False, interior=None):
    v, k = c["v"], c["k"]
    G = geom(v); K, T = G["K"], G["T"]
    img = frame(v, k)
    m = up(c["mask"])
    b = c["box"]
    H0 = max(0.02, float(b["top"]))
    if key == "toolbox_lid":                                  # flat: the silhouette cannot measure 2 cm; lid thickness = 2 cm prior
        H0 = 0.02
    fit = geo.fit_cuboid(m, K, T, (b["x"], b["y"], b["yaw"], b["L"], b["W"], H0))
    if key == "toolbox_lid":
        fit = geo.fit_cuboid(m, K, T, (fit["x"], fit["y"], fit["yaw"], fit["L"], fit["W"], 0.02), fix_dims=(fit["L"], fit["W"], 0.02))
        fit.update(L=fit["L"], W=fit["W"], H=0.02)
    if fit["L"] < fit["W"]:                                   # canonical: L >= W (x = long side)
        fit["L"], fit["W"], fit["yaw"] = fit["W"], fit["L"], fit["yaw"] + math.pi / 2
    faces = geo.face_textures(img, cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool), K, T, fit)
    tex_faces = {f: t for f, (t, s) in faces.items()}
    if tray:
        tex_faces.pop("top", None)
    mesh, atlas, src = geo.cuboid_mesh(fit["L"], fit["W"], fit["H"], tex_faces)
    meta = dict(source=f"{v} k={k} track={c['track']} det={c['det']}", fit=fit, faces=src, fit_iou=round(fit["iou"], 3),
                note="textured cuboid from the video (faces by homography from the source frame)")
    if tray:
        mesh, atlas = tray_mesh(fit, atlas, interior)
        meta["note"] = "open tray: outside walls from the video, inside = measured interior colour; lid separate (toolbox_lid)"
        meta["interior_bgr"] = [int(x) for x in interior]
    save_entry(key, mesh, atlas, meta, src_crop(img, m, f"{key}: {v} k{k} IoU {fit['iou']:.2f}"))


def tray_mesh(fit, atlas, interior, t=0.006):
    """Open box: outer walls (atlas tiles of the cuboid) + inner walls / floor (interior colour) + rim. The atlas gets a 7th tile."""
    L, W, H = fit["L"], fit["W"], fit["H"]
    tile = atlas.shape[0] // 2
    atlas2 = np.concatenate([atlas, np.zeros((tile, atlas.shape[1], 3), np.uint8)], 0)
    atlas2[2 * tile:] = np.array(interior, np.uint8)
    outer, _, _ = geo.cuboid_mesh(L, W, H, {})                     # geometry only; reuse UVs of the original atlas layout
    names = ["top", "bottom", "front", "back", "right", "left"]
    V, F, UV = [], [], []
    oV, oF, oUV = outer.vertices, outer.faces, outer.visual.uv.copy()
    oUV[:, 1] = oUV[:, 1] * 2 / 3 + 1 / 3                          # atlas rows: old (2 tiles) -> top 2/3 of the new 3-tile-high atlas
    keep = [i for i in range(len(oF)) if i // 2 != names.index("top")]
    V += list(oV); UV += list(oUV); F += [oF[i] for i in keep]
    inner, _, _ = geo.cuboid_mesh(L - 2 * t, W - 2 * t, H - t, {})
    iV = inner.vertices + np.array([0, 0, t])
    iUV = np.tile([[0.5, 1 / 6]], (len(iV), 1))                    # interior tile
    b = len(V)
    keep = [i for i in range(len(inner.faces)) if i // 2 != names.index("top")]
    V += list(iV); UV += list(iUV); F += [inner.faces[i][::-1] + b for i in keep]
    # rim: quads between outer and inner top edges
    oc = geo.box_corners(0, 0, 0, L, W, H)[4:]
    ic = geo.box_corners(0, 0, 0, L - 2 * t, W - 2 * t, H)[4:]
    b = len(V)
    V += list(oc) + list(ic); UV += [[0.5, 1 / 6]] * 8
    for i in range(4):
        j = (i + 1) % 4
        F += [[b + i, b + j, b + 4 + j], [b + i, b + 4 + j, b + 4 + i]]
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.array(UV))
    return m, atlas2


# ------------------------------------------------------------------ flat tools
def refine_mask(img, plate, m_lo, thr=40):
    """SAM 3 mask (270x480) -> 960x540 silhouette by GrabCut on a crop: sure foreground = eroded SAM mask, probable foreground = the SAM
    mask, probable background = its 6 px ring where the frame differs from the static plate (shadow / blur), sure background = outside."""
    m = up(m_lo)
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = max(0, xs.min() - 30), min(960, xs.max() + 31), max(0, ys.min() - 30), min(540, ys.max() + 31)
    crop, mc = img[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1].astype(np.uint8)
    gc = np.full(mc.shape, cv2.GC_BGD, np.uint8)
    ring = cv2.dilate(mc, np.ones((13, 13), np.uint8)).astype(bool)
    diff = np.abs(crop.astype(np.int16) - plate[y0:y1, x0:x1].astype(np.int16)).max(2) > thr
    gc[ring] = cv2.GC_PR_BGD
    gc[ring & diff] = cv2.GC_PR_FGD
    gc[mc.astype(bool)] = cv2.GC_PR_FGD
    gc[cv2.erode(mc, np.ones((5, 5), np.uint8)).astype(bool)] = cv2.GC_FGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, gc, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    s = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(s)
    if n > 1:
        s = (lab_ == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    out = np.zeros((540, 960), bool)
    out[y0:y1, x0:x1] = s.astype(bool)
    return out


def build_flat(key, c, pillow=False):
    from cleantables.twin.ep import plate as PL
    v, k = c["v"], c["k"]
    PL.build(v)
    plate = cv2.imread(str(OUT / "_video" / v / "plate.png"))
    G = geom(v); K, T = G["K"], G["T"]
    img = frame(v, k)
    thick = LIBRARY[key]["thick"]
    sil = refine_mask(img, plate, c["mask"])
    ys, xs = np.nonzero(sil)
    from cleantables.twin.ep.common import plane_hit
    P = plane_hit(np.c_[xs, ys].astype(float), K, T, z=thick)
    x0, y0 = P[:, :2].min(0) - 0.02
    x1, y1 = P[:, :2].max(0) + 0.02
    orth, (mx, my) = geo.ortho(img, K, T, x0, x1, y0, y1, z=thick)
    so = cv2.remap(sil.astype(np.uint8) * 255, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 127
    so = cv2.morphologyEx(so.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(so.astype(np.uint8))
    so = lab_ == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    rs, cs = np.nonzero(so)
    so = so[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
    orth = orth[rs.min():rs.max() + 1, cs.min():cs.max() + 1].copy()
    core = cv2.erode(so.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    mean = np.median(orth[core if core.sum() > 20 else so], 0)
    tex = orth.copy(); tex[~so] = mean.astype(np.uint8)
    tex = cv2.resize(tex, (tex.shape[1] * 2, tex.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    mesh = geo.slab_mesh(so, thick, pillow=pillow)
    meta = dict(source=f"{v} k={k} track={c['track']} det={c['det']}", thickness_prior=thick, silhouette_px=int(so.sum()),
                note="flat object: real top-down silhouette (SAM 3 mask refined by plate difference, rectified to the table plane) extruded "
                     "to the thickness prior; texture = rectified real pixels",
                mean_bgr=[int(x) for x in mean], redness=float(lab(mean[::-1])[1]))
    save_entry(key, mesh, tex, meta, src_crop(img, sil, f"{key}: {v} k{k}"))


def build_tools():
    vids = TASK_VIDEOS["tools"]
    cache = {}
    # pliers: one video/frame where both pliers are isolated (same lighting); higher redness (a*) = pliers_a (needle-nose, red grips)
    pc = candidates(vids, lambda k: k == "pliers")
    best = None
    by = {}
    for c in pc:
        by.setdefault((c["v"], c["k"]), []).append(c)
    for (v, k), cs in sorted(by.items(), key=lambda kv: -sum(c["area"] for c in kv[1])):
        if len(cs) != 2 or not all(complete(c, pc) for c in cs):
            continue
        ok = [isolated(c, cache) for c in cs]
        if all(o < 0.01 for o, _ in ok):
            for c, (_, m) in zip(cs, ok):
                c["mask"] = m
            best = cs
            break
    for c in best:
        img = frame(c["v"], c["k"])
        mm = up(c["mask"])
        c["red"] = float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[mm][:, 1].astype(float).mean())
    best.sort(key=lambda c: -c["red"])
    build_flat("pliers_a", best[0]); build_flat("pliers_b", best[1])
    for key in ("wrench", "utility_knife", "screwdriver"):
        s = pick(candidates(vids, lambda k, key=key: k == key), 1, cache)
        if s:
            build_flat(key, s[0])
        else:
            print("NO clean sighting for", key, flush=True)


def build_boxes(task):
    vids = TASK_VIDEOS["tools"] + TASK_VIDEOS["stack_boxes"]
    vids = sorted(set(vids))
    cache = {}
    keys = ["amazon_box", "cipriata_box", "lotus_box", "black_box", "orange_box", "yellow_box"] if task == "boxes" else []
    import sys
    if "--only" in sys.argv:
        keys = [k for k in keys if k in sys.argv[sys.argv.index("--only") + 1].split(",")]
    for key in keys:
        s = pick(candidates(vids, lambda k, key=key: k == key, first_rest=True, max_top=0.2), 1, cache)
        if not s:                   # e.g. the yellow box (one video): any resting phase on the table
            s = pick(candidates(vids, lambda k, key=key: k == key, max_top=0.2), 1, cache)
        if s:
            build_cuboid(key, s[0])
        else:
            print("NO clean sighting for", key, flush=True)


def build_toolbox():
    """Tray: best sighting of an open tray (gold walls); interior colour = median of its eroded mask. lid: a white_box sighting at a
    keyframe where an open tray is visible elsewhere (so it is the lid, not the closed box)."""
    vids = TASK_VIDEOS["tools"]
    cache = {}
    s = pick(candidates(vids, lambda k: k == "toolbox", max_top=0.09), 1, cache)
    c = s[0]
    img = frame(c["v"], c["k"])
    m = cv2.erode(up(c["mask"]).astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    interior = np.median(img[m], 0) if m.sum() > 100 else np.array([235, 240, 242])
    build_cuboid("toolbox", c, tray=True, interior=interior)
    trays = {(x["v"], x["k"]) for x in candidates(vids, lambda k: k == "toolbox")}
    lids = [x for x in candidates(vids, lambda k: k == "white_box", max_top=0.05) if (x["v"], x["k"]) in trays]
    s = pick(lids, 1, cache)
    if s:
        build_cuboid("toolbox_lid", s[0])
    else:
        print("NO clean lid sighting", flush=True)


def sheet():
    tiles = []
    for d in sorted(LIB.iterdir()):
        if (d / "source.jpg").exists():
            a = cv2.imread(str(d / "source.jpg")); t = cv2.imread(str(d / "texture.png"))
            a = cv2.resize(a, (300, int(300 * a.shape[0] / a.shape[1])))
            t = cv2.resize(t, (300, int(300 * t.shape[0] / t.shape[1])))
            col = np.vstack([a[:220], t[:220]]) if a.shape[0] >= 220 and t.shape[0] >= 220 else None
            if col is None:
                c = np.full((440, 300, 3), 255, np.uint8)
                c[:min(220, a.shape[0])] = a[:220]; c[220:220 + min(220, t.shape[0])] = t[:220]
                col = c
            cv2.putText(col, d.name, (4, 434), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            tiles.append(col)
    rows = [np.hstack(tiles[i:i + 5] + [np.full((440, 300, 3), 255, np.uint8)] * (5 - len(tiles[i:i + 5]))) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(LIB / "sheet.jpg"), np.vstack(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tools", choices=["tools", "boxes", "toolbox", "all"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.task in ("tools", "all"):
        build_tools()
    if a.task in ("toolbox", "all"):
        build_toolbox()
    if a.task in ("boxes", "all"):
        build_boxes("boxes")
    sheet()
