"""Library entries from Hunyuan3D-2.1 (cleantables.twin.ep.mesh_run), in place of the extruded-silhouette / cuboid meshes of
cleantables.twin.ep.library.

  source scripts/env.sh; $SIM_PY -m cleantables.twin.ep.library_hy3d [keys]

Per key: the Hunyuan paint textured mesh if it exists (textured_mesh.obj + .jpg). Otherwise the Hunyuan shape mesh + texture
projected from the input crop (orthographic, along the mesh axis whose silhouette matches the crop alpha best; surfaces facing away get
the colour of the front surface through the object) -- stated in meta.json as texture='projected crop (paint failed: ...)'.
Pose: y-up -> z-up, the stable resting pose closest to the measured height (lying flat for tools), long horizontal axis -> +x, footprint
centred, bottom at z = 0. Scale: isotropic, so the long footprint side equals the size measured from the video (the silhouette /
cuboid fit of the same object in cleantables.twin.ep.library, MoGe metric): the thickness is Hunyuan's own (not a prior). The
previous entries are kept as <key>/old_*.
"""
from __future__ import annotations

import json
import math
import shutil
import sys

import cv2
import numpy as np
import trimesh

from cleantables.twin import table_scene as B
from cleantables.twin.ep.common import LIB, ROOT
from cleantables.twin.ep.identity import LIBRARY
from cleantables.twin.ep.library import save_entry

HY = ROOT / "data/twin_table/ep/hy3d"
YUP2ZUP = B.YUP2ZUP


def measured(key):
    """(long footprint side, height or nan) of the object as measured from the video by the silhouette / cuboid library entry."""
    d = LIB / key
    src = d / "old_meta.json" if (d / "old_meta.json").exists() else d / "meta.json"
    if not src.exists():                       # no silhouette entry (e.g. screwdriver): the tracked 3D box of the Hunyuan input sighting
        from cleantables.twin.ep.common import geom, plane_hit
        c = json.load(open(HY / "inputs" / "sightings.json"))[key][0]
        mk = np.load(HY / "inputs" / f"mask_{key}_0.npy")
        ys, xs = np.nonzero(mk)
        G = geom(c["v"])
        P = plane_hit(np.c_[xs * 2 + 1.0, ys * 2 + 1.0], G["K"], G["T"], z=0.01)[:, :2]
        P -= P.mean(0)
        w, V = np.linalg.eigh(np.cov(P.T))
        pr = P @ V[:, np.argmax(w)]
        return float(np.percentile(pr, 99) - np.percentile(pr, 1)), float("nan")
    m = json.load(open(src))
    if m.get("fit"):
        return float(max(m["fit"]["L"], m["fit"]["W"])), float(m["fit"]["H"])
    if "diameter" in m:
        return float(m["diameter"]), float(m["height"])
    z = np.load(d / ("old_mesh.npz" if (d / "old_mesh.npz").exists() else "mesh.npz"))
    xy = z["V"][:, :2] - z["V"][:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov(xy.T))
    p = xy @ V[:, np.argmax(w)]
    return float(np.ptp(p)), float("nan")


def project_texture(m, crop_rgba):
    """UVs + texture from the crop: choose the orthographic view (6 axis directions x 4 in-plane rotations) whose silhouette matches the
    crop alpha (IoU), then every vertex takes the crop pixel it projects to."""
    a = crop_rgba[..., 3] > 127
    ys, xs = np.nonzero(a)
    ca = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rgb = crop_rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1, :3].copy()
    mean = np.median(rgb[ca], 0)
    rgb[~ca] = mean.astype(np.uint8)
    rgb = cv2.inpaint(rgb, (~ca).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    H_, W_ = ca.shape
    V = m.vertices
    best = (-1, None)
    for ax in range(3):
        for sgn in (1, -1):
            o = [i for i in range(3) if i != ax]
            for rot in range(4):
                u, v = V[:, o[0]].copy(), V[:, o[1]].copy() * sgn
                for _ in range(rot):
                    u, v = v, -u
                un = (u - u.min()) / max(np.ptp(u), 1e-9); vn = (v - v.min()) / max(np.ptp(v), 1e-9)
                if abs(math.log((np.ptp(u) / max(np.ptp(v), 1e-9)) / (W_ / H_))) > 0.5:
                    continue
                sil = np.zeros((H_, W_), np.uint8)
                P = np.c_[un * (W_ - 1), (1 - vn) * (H_ - 1)].astype(np.int32)
                for f in m.faces[::max(1, len(m.faces) // 6000)]:
                    cv2.fillConvexPoly(sil, P[f], 1)
                iou = (sil.astype(bool) & ca).sum() / max(1, (sil.astype(bool) | ca).sum())
                if iou > best[0]:
                    best = (iou, (un, vn))
    if best[1] is None:
        un = (V[:, 0] - V[:, 0].min()) / np.ptp(V[:, 0]); vn = (V[:, 1] - V[:, 1].min()) / np.ptp(V[:, 1]); best = (0.0, (un, vn))
    un, vn = best[1]
    m.visual = trimesh.visual.TextureVisuals(uv=np.c_[un, vn])
    return rgb, float(best[0])


def lie_flat(m):
    """Flat tools lie on their broadest side: principal axes of the vertices -> the thinnest direction becomes z, the longest x
    (stable-pose sampling can put a thin open shape such as the cutter on its edge); centred, bottom z = 0."""
    X = m.vertices - m.vertices.mean(0)
    w, V = np.linalg.eigh(np.cov(X.T))                 # ascending
    R = np.stack([V[:, 2], V[:, 1], V[:, 0]], 0)       # rows: long -> x, mid -> y, thin -> z
    if np.linalg.det(R) < 0:
        R[1] *= -1
    T = np.eye(4); T[:3, :3] = R
    bm = m.copy(); bm.apply_transform(T)
    lo, hi = bm.bounds
    bm.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    return bm


def footprint(V, F, res, lo, shape):
    img = np.zeros(shape, np.uint8)
    P = ((V[:, :2] - lo) / res).astype(np.int32)
    for f in F:
        cv2.fillConvexPoly(img, P[f], 1)
    return img.astype(bool)


def carve(m, key, res=0.001):
    """Remove Hunyuan 'webbing' (membranes between pliers handles, flaps): faces whose top-view centroid falls outside the real top-down
    silhouette of the object (silhouette library entry: SAM 3 mask rectified to the table plane). The two are registered by the long axis
    (both PCA-aligned), the 4 flips/half-turns and a +-1 cm shift (best footprint IoU)."""
    d = LIB / key
    op = d / ("old_mesh.npz" if (d / "old_mesh.npz").exists() else "mesh.npz")
    if not op.exists() or json.load(open(d / ("old_meta.json" if (d / "old_meta.json").exists() else "meta.json"))).get("fit"):
        return m, "no real silhouette (not carved)"
    z = np.load(op)
    Vo, Fo = z["V"].astype(float), z["F"]
    xy = Vo[:, :2] - Vo[:, :2].mean(0)
    w, Vv = np.linalg.eigh(np.cov(xy.T)); ax = Vv[:, np.argmax(w)]
    c, s_ = math.cos(-math.atan2(ax[1], ax[0])), math.sin(-math.atan2(ax[1], ax[0]))
    Vo2 = np.c_[xy @ np.array([[c, s_], [-s_, c]]), Vo[:, 2]]
    Vo2[:, :2] -= (Vo2[:, :2].min(0) + Vo2[:, :2].max(0)) / 2
    top = Fo[(np.cross(Vo[Fo[:, 1]] - Vo[Fo[:, 0]], Vo[Fo[:, 2]] - Vo[Fo[:, 0]])[:, 2]) > 0]
    ext = max(np.abs(Vo2[:, :2]).max(), np.abs(m.vertices[:, :2]).max()) + 0.03
    lo = np.array([-ext, -ext]); shape = (int(2 * ext / res) + 1,) * 2
    sil = footprint(Vo2, top, res, lo, shape)
    sil = cv2.dilate(sil.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    Vm, Fm = m.vertices.copy(), m.faces
    best = (-1, None)
    Fs = Fm[::max(1, len(Fm) // 3000)]
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for rot in np.radians(np.arange(-30, 31, 6)):
            c_, s2 = math.cos(rot), math.sin(rot)
            V2 = (Vm * np.array([sx, sy, 1.0])) @ np.array([[c_, s2, 0], [-s2, c_, 0], [0, 0, 1]])
            for dx in np.arange(-0.02, 0.0201, 0.005):
                for dy in np.arange(-0.02, 0.0201, 0.005):
                    fp = footprint(V2 + np.array([dx, dy, 0]), Fs, res, lo, shape)
                    iou = (fp & sil).sum() / max(1, (fp | sil).sum())
                    if iou > best[0]:
                        best = (iou, (sx, sy, rot, dx, dy))
    sx, sy, rot, dx, dy = best[1]
    c_, s2 = math.cos(rot), math.sin(rot)
    Rr = np.array([[c_, s2, 0], [-s2, c_, 0], [0, 0, 1]])
    best_iou_note = "(mirror-image registration: the Hunyuan mesh is the mirror of the real silhouette)" if sx * sy < 0 else ""
    V3 = (Vm * np.array([sx, sy, 1.0])) @ Rr + np.array([dx, dy, 0])
    cen = V3[Fm].mean(1)
    ij = ((cen[:, :2] - lo) / res).astype(int)
    ij = np.clip(ij, 0, shape[0] - 1)
    keep = sil[ij[:, 1], ij[:, 0]]
    m2 = m.copy()
    m2.update_faces(keep)
    m2.remove_unreferenced_vertices()
    m2.vertices = (m2.vertices * np.array([sx, sy, 1.0])) @ Rr + np.array([dx, dy, 0])
    if sx * sy < 0:
        m2.invert()                                    # reflection flips the winding: restore outward faces
    lo2, hi2 = m2.bounds
    m2.apply_translation((-(lo2[0] + hi2[0]) / 2, -(lo2[1] + hi2[1]) / 2, -lo2[2]))
    return m2, f"carved by the real top-down silhouette: kept {keep.mean() * 100:.0f} % of faces, registration IoU {best[0]:.2f} {best_iou_note}"


def build(key):
    d = HY / key
    jobs = json.load(open(HY / "jobs.json"))["jobs"]
    if not (d / "shape.glb").exists():
        print("NO Hunyuan shape for", key, "(", (d / "shape.fail.json").read_text() if (d / "shape.fail.json").exists() else "not run", ")")
        return None
    L_long, H_meas = measured(key)
    paint_ok = (d / "paint" / "textured_mesh.obj").exists()
    if paint_ok:
        m = trimesh.load(d / "paint" / "textured_mesh.obj", force="mesh", process=False)
        tex = cv2.imread(str(d / "paint" / "textured_mesh.jpg"))
        how = "Hunyuan3D-2.1 PAINT texture"
    else:
        m = trimesh.load(d / "shape.glb", force="mesh", process=False)
        crop = cv2.imread(str(ROOT / jobs[key]), cv2.IMREAD_UNCHANGED)
        tex, piou = None, None
        why = (d / "paint" / "paint.fail.json").read_text() if (d / "paint" / "paint.fail.json").exists() else "paint not run (GPU memory)"
        how = f"projected crop texture (Hunyuan paint unavailable: {why.strip()[:120]})"
    m.apply_transform(YUP2ZUP)
    flat = LIBRARY[key]["kind"] in ("flat", "pillow")
    m.apply_scale(L_long / max(m.extents[:2].max(), 1e-6))                    # provisional, for the rest-pose height match
    if flat:
        m = lie_flat(m)
    else:
        m = B.rest_pose(m, H_meas if np.isfinite(H_meas) else float("nan"), flat)
    e = m.extents
    kind = LIBRARY[key]["kind"]
    old = json.load(open(LIB / key / ("old_meta.json" if (LIB / key / "old_meta.json").exists() else "meta.json"))) if (LIB / key / "meta.json").exists() else {}
    if kind in ("cuboid", "tray") and old.get("fit"):
        f = old["fit"]                                 # boxes: per-axis to the measured box (Hunyuan's box proportions are not metric)
        if e[0] < e[1]:
            m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 0, 1)))
            e = m.extents
        m.apply_scale([max(f["L"], f["W"]) / e[0], min(f["L"], f["W"]) / e[1], f["H"] / e[2]])
        lo_, hi_ = m.bounds
        m.apply_translation((-(lo_[0] + hi_[0]) / 2, -(lo_[1] + hi_[1]) / 2, -lo_[2]))
    else:
        s = L_long / max(e[0], e[1])
        m.apply_scale(s)
    carve_note = ""
    if flat:
        m, carve_note = carve(m, key)
    if not paint_ok:
        tex, piou = project_texture(m, crop)
        how += f", view match IoU {piou:.2f}"
    d2 = LIB / key
    if (d2 / "mesh.npz").exists() and not (d2 / "old_mesh.npz").exists():
        for f in ("mesh.npz", "texture.png", "meta.json"):
            shutil.copy(d2 / f, d2 / f"old_{f}")
        (d2 / "coacd.npz").unlink(missing_ok=True)
    (d2 / "coacd.npz").unlink(missing_ok=True)
    crop = cv2.imread(str(ROOT / jobs[key]), cv2.IMREAD_UNCHANGED)
    a = crop[..., 3:] / 255.0
    src = (crop[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    meta = dict(source=f"Hunyuan3D-2.1 shape of {jobs[key]}", texture=how, measured_long_side=L_long, measured_height=H_meas,
                hunyuan_height=float(m.extents[2]), fit=None, carve=carve_note,
                note="Hunyuan3D-2.1 mesh from the video crop; isotropic scale to the measured long side; thickness = Hunyuan's")
    save_entry(key, m, tex, meta, cv2.resize(src, (300, 300)))
    return meta


if __name__ == "__main__":
    keys = sys.argv[1:] or list(json.load(open(HY / "jobs.json"))["jobs"])
    for k in keys:
        build(k)
