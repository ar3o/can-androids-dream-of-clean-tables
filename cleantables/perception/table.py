"""Table plane, table frame, table mask and per-frame depth scale -> det3d.npz and table.json.

The camera stands on a tripod, so one plane fit serves the whole video unless the camera was knocked: the plane is
fitted in six time windows and a tilt change above BUMP_TILT_DEG starts a new camera segment with its own fit. MoGe
depth has a slightly different scale on every frame; each frame gets the scale that puts its free table pixels on the
plane.
"""

from __future__ import annotations

import cv2
import numpy as np

from cleantables import paths

from . import geometry as G
from .depth import intrinsics
from .files import H_LO, W_LO, depth, frame_meta, geometry, n_frames, read_frame, unpack_masks, video_dir, write_json

N_WINDOWS = 6
BUMP_TILT_DEG = 2.5
TABLE_FRACTION = 0.12  # low: the person hides parts of the table in many frames
EXCLUDE_DILATE = np.ones((9, 9), np.uint8)
MASK_MIN_PROB = 0.3


def key_frames(video: str, stride: int, n_key: int) -> list[int]:
    """n_key frames spread over the video, all multiples of the depth stride (their depth was computed, not copied)."""
    n = n_frames(video)
    ks = sorted({int(round(x / stride)) * stride for x in np.linspace(2, n - 3, n_key)})
    return [k for k in ks if k < n]


def image_up(video: str):
    return (1.0, 0, 0) if frame_meta(video).get("rotated_portrait") else (0, -1.0, 0)


def save(video: str, T_all, K, table, scale, info: dict, preliminary: bool) -> None:
    d = video_dir(video)
    np.savez_compressed(
        d / "det3d.npz", T=T_all[0], T_all=T_all, K=K, table=table, scale=scale, prelim=np.array(preliminary)
    )
    write_json(
        dict(video=video, T_cam_to_table=T_all[0].tolist(), K=K.tolist(), prelim=preliminary, **info), d / "table.json"
    )


def preliminary(video: str, stride: int, n_key: int) -> None:
    """Table from depth alone (largest upward plane); only used to crop the object listing to the table."""
    D10 = depth(video)
    K = intrinsics(video)
    deps = [np.asarray(D10[k], np.float32) for k in key_frames(video, stride, n_key)]
    T, _, _, info = G.fit_table(deps, K, None, up=image_up(video))
    table = G.table_mask(deps, K, T, frac=TABLE_FRACTION)
    n = n_frames(video)
    save(video, np.repeat(T[None], n, 0), K, table, np.ones(n, np.float32), info, True)


def is_preliminary(video: str) -> bool:
    return bool(np.load(paths.perception_dir(video) / "det3d.npz")["prelim"])


def fit(video: str, dets: dict) -> dict:
    """Final table geometry with the keyframe detections (objects and hands) removed from the plane fits.

    dets: dict(frames [k], frame (m,), prob (m,), masks (m,270,480)) from the SAM 3 keyframe pass.
    """
    D10 = depth(video)
    K = intrinsics(video)
    n = n_frames(video)
    frames = [int(k) for k in dets["frames"]]
    by_frame = {}
    for i, k in enumerate(dets["frame"]):
        if dets["prob"][i] >= MASK_MIN_PROB:
            by_frame.setdefault(int(k), []).append(i)

    def excluded(k):
        e = np.zeros((H_LO, W_LO), np.uint8)
        for i in by_frame.get(k, []):
            e |= dets["masks"][i].astype(np.uint8)
        return cv2.dilate(e, EXCLUDE_DILATE) > 0

    up = image_up(video)
    deps = [np.asarray(D10[k], np.float32) for k in frames]
    T, _, _, info = G.fit_table(deps, K, [excluded(k) for k in frames], up=up)
    T_all, bumps, window_diffs = camera_segments(D10, K, frames, excluded, T, deps, n, up)
    T = T_all[0]
    info.update(bump_frames=bumps, window_plane_diff_deg_cm=window_diffs, n_keyframes=len(frames))
    table = G.table_mask(deps, K, T, frac=TABLE_FRACTION)
    scale = frame_scales(D10, K, T_all, table, frames, by_frame, excluded, n)
    save(video, T_all, K, table, scale, info, False)
    qc_image(video, frames[len(frames) // 2], info)
    return info


def camera_segments(D10, K, frames, excluded, T, deps, n, up):
    """Per-frame camera -> table transforms: one fit per camera segment (a knocked camera starts a new segment)."""
    table0 = G.table_mask(deps, K, T, frac=TABLE_FRACTION)
    windows = np.array_split(np.array(frames), N_WINDOWS)
    fits = []
    for wk in windows:
        ks = [int(x) for x in wk]
        if not ks:
            fits.append(None)
            continue
        excl = [excluded(k) | ~table0 for k in ks]
        try:
            Tw, *_ = G.fit_table([np.asarray(D10[k], np.float32) for k in ks], K, excl, up=up)
            fits.append((Tw, ks, excl))
        except RuntimeError:
            fits.append(None)
    segment, current, bumps, diffs = [], 0, [], []
    for wi in range(N_WINDOWS):
        if wi > 0 and fits[wi] is not None and fits[wi - 1] is not None:
            a, b = fits[wi][0], fits[wi - 1][0]
            angle = float(np.degrees(np.arccos(np.clip(a[2, :3] @ b[2, :3], -1, 1))))
            diffs.append([round(angle, 2), round(abs(float(a[2, 3] - b[2, 3])) * 100, 2)])
            # heights are not compared: the MoGe scale varies from frame to frame
            if angle > BUMP_TILT_DEG:
                current += 1
                bumps.append(int(windows[wi][0]))
        segment.append(current)
    T_all = np.repeat(T[None], n, 0)
    if bumps:
        for sid in sorted(set(segment)):
            members = [wi for wi in range(N_WINDOWS) if segment[wi] == sid and fits[wi] is not None]
            ks = sum([fits[wi][1] for wi in members], [])
            excl = sum([fits[wi][2] for wi in members], [])
            Ts, *_ = G.fit_table([np.asarray(D10[k], np.float32) for k in ks], K, excl, up=up)
            lo = 0 if sid == 0 else int(windows[min(wi for wi in range(N_WINDOWS) if segment[wi] == sid)][0])
            hi = (
                n
                if sid == max(segment)
                else int(windows[min(wi for wi in range(N_WINDOWS) if segment[wi] == sid + 1)][0])
            )
            T_all[lo:hi] = Ts
    return T_all, bumps, diffs


def frame_scales(D10, K, T_all, table, frames, by_frame, excluded, n):
    """Depth scale of every frame; frames without detections use the masks of the nearest keyframe."""
    planes = {}

    def plane(k):
        key = T_all[k].tobytes()
        if key not in planes:
            planes[key] = G.plane_z_map(T_all[k], K)
        return planes[key]

    scale = np.ones(n, np.float32)
    for k in frames:
        scale[k] = G.depth_scale(np.asarray(D10[k], np.float32), plane(k), table & ~excluded(k))
    kf = np.array(frames)
    for k in range(n):
        if k not in by_frame and k not in frames:
            nearest = int(kf[np.argmin(np.abs(kf - k))])
            scale[k] = G.depth_scale(np.asarray(D10[k], np.float32), plane(k), table & ~excluded(nearest))
    return scale


def refine_scale(video: str) -> None:
    """Re-estimate the depth scale of every tracked frame with all tracked masks (objects and hands) removed."""
    path = paths.perception_dir(video) / "det3d.npz"
    d = dict(np.load(path))
    if "refined_scale" in d:
        return
    z = np.load(paths.perception_dir(video) / "sam3_video.npz")
    ks = z["frames"].astype(int)
    covered = np.zeros((len(ks), H_LO, W_LO), bool)
    for pi in range(len(z["phrases"])):
        shape = z[f"p{pi}_shape"]
        if int(shape[0]) == 0:
            continue
        masks = unpack_masks(z[f"p{pi}_masks"], shape)
        covered |= (masks & (z[f"p{pi}_probs"][:, :, None, None] >= MASK_MIN_PROB)).any(0)
    D10 = depth(video)
    K, T_all, table = d["K"], d["T_all"], d["table"]
    scale = d["scale"].copy()
    planes = {}
    for t, k in enumerate(ks):
        key = T_all[k].tobytes()
        if key not in planes:
            planes[key] = G.plane_z_map(T_all[k], K)
        free = table & ~(cv2.dilate(covered[t].astype(np.uint8), EXCLUDE_DILATE) > 0)
        scale[k] = G.depth_scale(np.asarray(D10[k], np.float32), planes[key], free)
    tracked = set(ks.tolist())
    for k in range(len(scale)):
        if k not in tracked:
            scale[k] = scale[int(ks[np.argmin(np.abs(ks - k))])]
    d["scale"] = scale
    d["refined_scale"] = np.array(True)
    np.savez_compressed(path, **d)


def qc_image(video: str, k: int, info: dict) -> None:
    """Table mask and table axes (20 cm, x red, y green, z blue) drawn on one frame -> table_qc.jpg."""
    g = geometry(video)
    img = read_frame(video, k)
    m = cv2.resize(g["table"].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = img.copy()
    out[m] = (0.55 * out[m] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    axes = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], float)
    uv, _ = G.project(axes, g["K"], g["T_all"][k])
    for j, colour in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
        cv2.arrowedLine(out, tuple(int(x) for x in uv[0]), tuple(int(x) for x in uv[j]), colour, 3)
    text = (
        f"{video} k={k} camera height {info.get('camera_height_m')} m, tilt {info.get('tilt_deg')} deg, "
        f"bumps {info.get('bump_frames')}, scale {g['scale'][k]:.3f}"
    )
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(paths.perception_dir(video) / "table_qc.jpg"), out)
