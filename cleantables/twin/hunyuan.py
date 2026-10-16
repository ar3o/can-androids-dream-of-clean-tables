"""Hunyuan3D-2.1 meshes for the flat tools: inputs, and library entries from the generated shapes.

Steps (library.py must have built the silhouette entries first; they give the size and the carving silhouette):
  python -m cleantables.twin.hunyuan select     best unoccluded sightings + masks          -> data/twins/hy3d/inputs/
  python -m cleantables.twin.hunyuan rect       top-down RGBA crops from the full-res video ->
  inputs/<key>/rect<i>_<video>_k<k>.png
  (check the crops, list the chosen one per key in data/twins/hy3d/jobs.json: {"jobs": {key: crop path relative to
  data/}})
  bash scripts/build_twins_hunyuan.sh shape     Hunyuan shape (and optionally paint) in its own environment
  python -m cleantables.twin.hunyuan build      library entries from hy3d/<key>/shape.glb (texture: paint, else the
  crop projected)
A top-down crop is the view in which Hunyuan reconstructs a lying tool's outline; oblique crops gave webbing and flaps.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil

import cv2
import numpy as np
import trimesh

from cleantables.paths import DATA, video_path
from cleantables.twin import library as lib
from cleantables.twin.identity import CATALOGUE
from cleantables.twin.inputs import HUNYUAN, LIBRARY, Detections, camera, frames_dir, plate_dir
from cleantables.twin.projection import plane_hit

INPUTS = HUNYUAN / "inputs"
FLAT_TOOLS = ["pliers_a", "pliers_b", "wrench", "utility_knife", "screwdriver"]
N_VIEWS = 3
RECT_RES = 0.0006  # m per pixel of the top-down crops
RECT_HEIGHT = 0.008  # plane height of the rectification (about half a tool's thickness)
VIDEO_PREFIX = (
    "video_"  # library files of the video-built entry kept next to the Hunyuan one (mesh.npz -> video_mesh.npz)
)


# ------------------------------------------------------------------ inputs
def select() -> None:
    """best unoccluded sightings per flat tool (different tracks first); pliers keep their library sighting."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    cache, out = {}, {}
    videos = lib.task_videos("tools")
    for key in FLAT_TOOLS:
        if key.startswith("pliers"):
            src = json.loads((LIBRARY / key / f"{_video_name('meta.json', key)}").read_text())["source"]
            v, k = src.split()[0], int(src.split("k=")[1].split()[0])
            track, det = int(src.split("track=")[1].split()[0]), int(src.split("det=")[1])
            out[key] = [dict(v=v, k=k, det=det, track=track, area=0)]
            np.save(INPUTS / f"mask_{key}_0.npy", Detections(v).mask(det))
            continue
        chosen, seen = [], set()
        for c in lib.pick(lib.candidates(videos, lambda k, key=key: k == key), 12, cache):
            if (c["v"], c["track"]) in seen:
                continue
            chosen.append(c)
            seen.add((c["v"], c["track"]))
            if len(chosen) >= N_VIEWS:
                break
        out[key] = [dict(v=c["v"], k=c["k"], det=c["det"], track=c["track"], area=c["area"]) for c in chosen]
        for i, c in enumerate(chosen):
            np.save(INPUTS / f"mask_{key}_{i}.npy", c["mask"])
    (INPUTS / "sightings.json").write_text(json.dumps(out, indent=1))
    print({k: [(x["v"], x["k"]) for x in v] for k, v in out.items()}, flush=True)


def _read_full_frame(caps: dict, video: str, k: int):
    """frame k of the 10 fps work frames, decoded at full resolution from the original video."""
    meta = json.loads((frames_dir(video).parent / "frames_meta.json").read_text())
    if video not in caps:
        caps[video] = cv2.VideoCapture(str(video_path(video)))
    caps[video].set(cv2.CAP_PROP_POS_FRAMES, k * int(meta.get("stride", 3)))
    ok, img = caps[video].read()
    return img if ok else None


def _keep_large_parts(a: np.ndarray, min_frac: float = 0.05) -> np.ndarray:
    """drop specks: keep every component >= min_frac of the largest (a handle can be cut off by the colour key)."""
    n, lab_, st, _ = cv2.connectedComponentsWithStats((a > 127 if a.max() > 1 else a > 0).astype(np.uint8))
    if n <= 1:
        return a
    big = 1 + np.nonzero(st[1:, cv2.CC_STAT_AREA] >= min_frac * st[1:, cv2.CC_STAT_AREA].max())[0]
    return np.where(np.isin(lab_, big), a, 0).astype(np.uint8)


def rect() -> None:
    """flat tools: each sighting rectified to a top-down view of the table plane (full-res frame, background removed
    by GrabCut; for the pliers also the table seen between the handles, by plate colour and the wood's hue)."""
    S = json.loads((INPUTS / "sightings.json").read_text())
    caps = {}
    for key in FLAT_TOOLS:
        for i, c in enumerate(S[key][:2]):
            v, k = c["v"], c["k"]
            img = _read_full_frame(caps, v, k)
            if img is None:
                continue
            Hf, Wf = img.shape[:2]
            G = camera(v)
            K = G["K"].copy()
            K[:2] *= Wf / 960.0
            T = G["T"]
            mu = cv2.resize(
                np.load(INPUTS / f"mask_{key}_{i}.npy").astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST
            )
            gc = np.full(mu.shape, cv2.GC_BGD, np.uint8)
            gc[cv2.dilate(mu, np.ones((25, 25), np.uint8)).astype(bool)] = cv2.GC_PR_BGD
            gc[mu.astype(bool)] = cv2.GC_PR_FGD
            gc[cv2.erode(mu, np.ones((9, 9), np.uint8)).astype(bool)] = cv2.GC_FGD
            ys, xs = np.nonzero(cv2.dilate(mu, np.ones((61, 61), np.uint8)))
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sub = gc[y0:y1, x0:x1].copy()
            cv2.grabCut(img[y0:y1, x0:x1], sub, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
            a = np.zeros(mu.shape, np.uint8)
            a[y0:y1, x0:x1] = ((sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)).astype(np.uint8) * 255
            plate = cv2.resize(
                cv2.imread(str(plate_dir(v) / "plate_raw.png")), (Wf, Hf), interpolation=cv2.INTER_LINEAR
            )
            same = np.abs(img.astype(np.int16) - plate.astype(np.int16)).max(2) < 22
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # pine table in the tool's shadow (differs from the plate): orange hue, saturated, not dark.
            # Pliers only: the screwdriver handle and the knife are orange-ish themselves.
            wood = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 70) & (hsv[..., 2] > 60)
            if not key.startswith("pliers"):
                wood[:] = False
            a[(same | wood) & (a > 0)] = 0
            a = _keep_large_parts(cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
            P = plane_hit(np.c_[xs, ys].astype(float), K, T, z=RECT_HEIGHT)
            lo, hi = P[:, :2].min(0) - 0.01, P[:, :2].max(0) + 0.01
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0], RECT_RES), np.arange(hi[1], lo[1], -RECT_RES))
            grid = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, RECT_HEIGHT)]
            Ti = np.linalg.inv(T)
            uv = (grid @ Ti[:3, :3].T + Ti[:3, 3]) @ K.T
            uv = uv[:, :2] / uv[:, 2:3]
            mx, my = uv[:, 0].reshape(gx.shape).astype(np.float32), uv[:, 1].reshape(gx.shape).astype(np.float32)
            rgb = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            alpha = _keep_large_parts(cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
            rgba = np.dstack([rgb, alpha])
            ys2, xs2 = np.nonzero(alpha > 127)
            rgba = rgba[ys2.min() : ys2.max() + 1, xs2.min() : xs2.max() + 1]
            h, w = rgba.shape[:2]
            s = int(max(h, w) * 1.2)
            square = np.zeros((s, s, 4), np.uint8)
            square[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = rgba
            p = INPUTS / key / f"rect{i}_{v}_k{k}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cv2.resize(square, (768, 768), interpolation=cv2.INTER_CUBIC))
            print(f"{key}: {p.name} {w * RECT_RES:.3f} x {h * RECT_RES:.3f} m", flush=True)


# ------------------------------------------------------------------ library entries
def _video_name(name: str, key: str) -> str:
    """the video-built file if it was set aside, else the current one."""
    return VIDEO_PREFIX + name if (LIBRARY / key / (VIDEO_PREFIX + name)).exists() else name


def measured_long_side(key: str) -> float:
    """long footprint side of the video-built entry (silhouette or cuboid fit, MoGe metric)."""
    d = LIBRARY / key
    meta = json.loads((d / _video_name("meta.json", key)).read_text())
    if meta.get("fit"):
        return float(max(meta["fit"]["L"], meta["fit"]["W"]))
    z = np.load(d / _video_name("mesh.npz", key))
    xy = z["V"][:, :2] - z["V"][:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov(xy.T))
    return float(np.ptp(xy @ V[:, np.argmax(w)]))


def project_texture(m: trimesh.Trimesh, crop_rgba: np.ndarray) -> tuple[np.ndarray, float]:
    """UVs + texture from the crop: the orthographic view (6 axis directions x 4 in-plane turns) whose silhouette
    matches the crop alpha best; every vertex takes the crop pixel it projects to (faces turned away get the colour
    through the object)."""
    a = crop_rgba[..., 3] > 127
    ys, xs = np.nonzero(a)
    ca = a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    rgb = crop_rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1, :3].copy()
    rgb[~ca] = np.median(rgb[ca], 0).astype(np.uint8)
    rgb = cv2.inpaint(rgb, (~ca).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    H_, W_ = ca.shape
    V = m.vertices
    best = (-1, None)
    for ax in range(3):
        for sgn in (1, -1):
            o = [i for i in range(3) if i != ax]
            for turns in range(4):
                u, v = V[:, o[0]].copy(), V[:, o[1]].copy() * sgn
                for _ in range(turns):
                    u, v = v, -u
                un = (u - u.min()) / max(np.ptp(u), 1e-9)
                vn = (v - v.min()) / max(np.ptp(v), 1e-9)
                if abs(math.log((np.ptp(u) / max(np.ptp(v), 1e-9)) / (W_ / H_))) > 0.5:
                    continue
                sil = np.zeros((H_, W_), np.uint8)
                P = np.c_[un * (W_ - 1), (1 - vn) * (H_ - 1)].astype(np.int32)
                for f in m.faces[:: max(1, len(m.faces) // 6000)]:
                    cv2.fillConvexPoly(sil, P[f], 1)
                iou = (sil.astype(bool) & ca).sum() / max(1, (sil.astype(bool) | ca).sum())
                if iou > best[0]:
                    best = (iou, (un, vn))
    if best[1] is None:
        un = (V[:, 0] - V[:, 0].min()) / np.ptp(V[:, 0])
        vn = (V[:, 1] - V[:, 1].min()) / np.ptp(V[:, 1])
        best = (0.0, (un, vn))
    m.visual = trimesh.visual.TextureVisuals(uv=np.c_[best[1]])
    return rgb, float(best[0])


def lie_flat(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """principal axes: thinnest -> z, longest -> x (stable poses put thin open shapes on edge); centred, bottom at 0."""
    X = m.vertices - m.vertices.mean(0)
    _, V = np.linalg.eigh(np.cov(X.T))
    R = np.stack([V[:, 2], V[:, 1], V[:, 0]], 0)
    if np.linalg.det(R) < 0:
        R[1] *= -1
    T = np.eye(4)
    T[:3, :3] = R
    bm = m.copy()
    bm.apply_transform(T)
    lo, hi = bm.bounds
    bm.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    return bm


def _footprint(V, F, res, lo, shape) -> np.ndarray:
    img = np.zeros(shape, np.uint8)
    P = ((V[:, :2] - lo) / res).astype(np.int32)
    for f in F:
        cv2.fillConvexPoly(img, P[f], 1)
    return img.astype(bool)


def carve(m: trimesh.Trimesh, key: str, res: float = 0.001) -> tuple[trimesh.Trimesh, str]:
    """remove Hunyuan webbing (membranes between pliers handles, flaps): faces whose top-view centroid lies outside
    the real top-down silhouette of the video-built entry. Registered by long axis, mirror / half-turn, +-30 deg and
    +-2 cm (best IoU)."""
    d = LIBRARY / key
    z = np.load(d / _video_name("mesh.npz", key))
    Vo, Fo = z["V"].astype(float), z["F"]
    xy = Vo[:, :2] - Vo[:, :2].mean(0)
    w, Vv = np.linalg.eigh(np.cov(xy.T))
    ax = Vv[:, np.argmax(w)]
    c, s_ = math.cos(-math.atan2(ax[1], ax[0])), math.sin(-math.atan2(ax[1], ax[0]))
    Vo2 = np.c_[xy @ np.array([[c, s_], [-s_, c]]), Vo[:, 2]]
    Vo2[:, :2] -= (Vo2[:, :2].min(0) + Vo2[:, :2].max(0)) / 2
    top = Fo[(np.cross(Vo[Fo[:, 1]] - Vo[Fo[:, 0]], Vo[Fo[:, 2]] - Vo[Fo[:, 0]])[:, 2]) > 0]
    ext = max(np.abs(Vo2[:, :2]).max(), np.abs(m.vertices[:, :2]).max()) + 0.03
    lo = np.array([-ext, -ext])
    shape = (int(2 * ext / res) + 1,) * 2
    sil = cv2.dilate(_footprint(Vo2, top, res, lo, shape).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    Vm, Fm = m.vertices.copy(), m.faces
    Fs = Fm[:: max(1, len(Fm) // 3000)]
    best = (-1, None)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for rot in np.radians(np.arange(-30, 31, 6)):
            c_, s2 = math.cos(rot), math.sin(rot)
            V2 = (Vm * np.array([sx, sy, 1.0])) @ np.array([[c_, s2, 0], [-s2, c_, 0], [0, 0, 1]])
            for dx in np.arange(-0.02, 0.0201, 0.005):
                for dy in np.arange(-0.02, 0.0201, 0.005):
                    fp = _footprint(V2 + np.array([dx, dy, 0]), Fs, res, lo, shape)
                    iou = (fp & sil).sum() / max(1, (fp | sil).sum())
                    if iou > best[0]:
                        best = (iou, (sx, sy, rot, dx, dy))
    sx, sy, rot, dx, dy = best[1]
    c_, s2 = math.cos(rot), math.sin(rot)
    Rr = np.array([[c_, s2, 0], [-s2, c_, 0], [0, 0, 1]])
    centres = ((Vm * np.array([sx, sy, 1.0])) @ Rr + np.array([dx, dy, 0]))[Fm].mean(1)
    ij = np.clip(((centres[:, :2] - lo) / res).astype(int), 0, shape[0] - 1)
    keep = sil[ij[:, 1], ij[:, 0]]
    m2 = m.copy()
    m2.update_faces(keep)
    m2.remove_unreferenced_vertices()
    m2.vertices = (m2.vertices * np.array([sx, sy, 1.0])) @ Rr + np.array([dx, dy, 0])
    mirrored = sx * sy < 0
    if mirrored:
        m2.invert()  # a reflection flips the winding
    lo2, hi2 = m2.bounds
    m2.apply_translation((-(lo2[0] + hi2[0]) / 2, -(lo2[1] + hi2[1]) / 2, -lo2[2]))
    note = " (mirror-image registration: the Hunyuan mesh is the mirror of the real silhouette)" if mirrored else ""
    return (
        m2,
        f"carved by the real top-down silhouette: kept {keep.mean() * 100:.0f} % of faces, "
        f"registration IoU {best[0]:.2f}{note}",
    )


def build_entry(key: str) -> None:
    """Hunyuan shape -> z-up, lying flat, isotropic scale to the measured long side, carved; the video-built files are
    kept as video_*."""
    d = HUNYUAN / key
    jobs = json.loads((HUNYUAN / "jobs.json").read_text())["jobs"]
    if not (d / "shape.glb").exists():
        print(f"{key}: no Hunyuan shape", flush=True)
        return
    long_side = measured_long_side(key)
    painted = (d / "paint" / "textured_mesh.obj").exists()
    if painted:
        m = trimesh.load(d / "paint" / "textured_mesh.obj", force="mesh", process=False)
        tex = cv2.imread(str(d / "paint" / "textured_mesh.jpg"))
        how = "Hunyuan3D-2.1 PAINT texture"
    else:
        m = trimesh.load(d / "shape.glb", force="mesh", process=False)
        how = "projected crop texture (Hunyuan paint not run)"
    m.apply_transform(lib.YUP_TO_ZUP)
    m.apply_scale(long_side / max(m.extents[:2].max(), 1e-6))
    m = lie_flat(m)
    m.apply_scale(long_side / max(m.extents[0], m.extents[1]))
    m, carve_note = carve(m, key)
    crop = cv2.imread(str(DATA / jobs[key]), cv2.IMREAD_UNCHANGED)
    if not painted:
        tex, view_iou = project_texture(m, crop)
        how += f", view match IoU {view_iou:.2f}"
    lib_dir = LIBRARY / key
    if (lib_dir / "mesh.npz").exists() and not (lib_dir / f"{VIDEO_PREFIX}mesh.npz").exists():
        for f in ("mesh.npz", "texture.png", "meta.json"):
            shutil.copy(lib_dir / f, lib_dir / f"{VIDEO_PREFIX}{f}")
    (lib_dir / "coacd.npz").unlink(missing_ok=True)
    a = crop[..., 3:] / 255.0
    preview = (crop[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    meta = dict(
        source=f"Hunyuan3D-2.1 shape of {jobs[key]}",
        texture=how,
        measured_long_side=long_side,
        hunyuan_height=float(m.extents[2]),
        fit=None,
        carve=carve_note,
        note="Hunyuan3D-2.1 mesh from the video crop; isotropic scale to the measured long side; thickness = Hunyuan's",
    )
    lib.save_entry(key, m, tex, meta, cv2.resize(preview, (300, 300)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Hunyuan3D-2.1 inputs and library entries for the flat tools.")
    ap.add_argument("step", choices=["select", "rect", "build"])
    ap.add_argument("keys", nargs="*", help="build: library keys (default: every key in jobs.json)")
    args = ap.parse_args()
    if args.step == "select":
        select()
    elif args.step == "rect":
        rect()
    else:
        for key in args.keys or list(json.loads((HUNYUAN / "jobs.json").read_text())["jobs"]):
            if CATALOGUE[key]["kind"] == "flat":
                build_entry(key)


if __name__ == "__main__":
    main()
