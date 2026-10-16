"""Static background of one video (colour + metric depth) from temporal stability, and the person's body points.

The phone does not move, so every pixel sees the same static surface whenever nothing transient (the person, a carried
object) is in front of it. Per pixel, over the tracked keyframes:
  1. metric depth d_k = depth10[k] x scale[k]; samples under a tracked object's or the hand's SAM 3 mask are dropped
     (movable objects are simulated, so they must not be baked into the background even when they sit still for
     minutes);
  2. static depth = the farthest log-depth cluster with enough support (transient things are always in front of the
     static surface; the person covers some pixels in more than half the frames, so the most frequent cluster would be
     wrong). Samples nearer than min_static_depth_m are never static, and inside the table mask only samples on the
     table plane count;
  3. colour = per-pixel median of the frames whose depth agrees with the static depth;
  4. pixels with too few agreeing samples are inpainted (colour) or filled from the table plane / nearest valid depth.

  python -m cleantables.twin.plate IMG_2533 [...] [--force]          raw plate + body points (CPU)
  python -m cleantables.twin.plate IMG_2533 [...] --apply-person      inpaint the person mask of person_mask.py
Out data/twins/plates/<video>/: plate_raw.png, plate_valid_raw.png, plate_depth.npy (270x480, camera z),
plate_meta.json, plate_proof.jpg, person_pts.npy; after --apply-person: plate.png, plate_valid.png,
plate_person_proof.jpg.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from cleantables.twin.config import PlateConfig, load_config
from cleantables.twin.inputs import (
    IMG_H,
    IMG_W,
    Detections,
    H,
    W,
    camera,
    depth,
    frame,
    hand_masks,
    plate_dir,
    tracks,
    write_json,
)
from cleantables.twin.projection import rays_lo

ROW_CHUNK = 60  # image rows per median batch (bounds memory)
BODY_MARGIN_M = 0.10  # nearer than the static depth by this much and not an object = the person's body


def _object_mask(dets: Detections, R: dict, k: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    for d in R["kept"].get(k, []):
        m |= dets.mask(d)
    return m


def _subsample(ks: list[int], n: int) -> list[int]:
    if len(ks) <= n:
        return ks
    return [ks[i] for i in np.linspace(0, len(ks) - 1, n).round().astype(int)]


def static_depth(Ds: np.ndarray, cfg: PlateConfig) -> np.ndarray:
    """(n, H, W) metric depth samples (NaN = invalid) -> (H, W) static depth (NaN where no cluster is supported)."""
    ld = np.log(np.clip(Ds, 0.05, 50.0))
    lo = np.nanmin(ld)
    b = np.floor((ld - lo) / cfg.depth_bin)
    nb = int(np.nanmax(b)) + 3
    hist = np.zeros((nb, H, W), np.int16)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(len(Ds)):
        ok = np.isfinite(b[i])
        np.add.at(hist, (b[i][ok].astype(int), yy[ok], xx[ok]), 1)
    smooth = hist.astype(np.int32)
    smooth[1:] += hist[:-1]
    smooth[:-1] += hist[1:]
    n_valid = np.isfinite(Ds).sum(0)
    supported = smooth >= np.maximum(cfg.min_support_frames, cfg.min_support_frac * n_valid)[None]
    far_bin = nb - 1 - np.argmax(supported[::-1], 0)
    centre = lo + (far_bin + 0.5) * cfg.depth_bin
    near_centre = np.abs(ld - centre[None]) <= 1.5 * cfg.depth_bin
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_static = np.nanmedian(np.where(near_centre, Ds, np.nan), 0)
    d_static[~supported.any(0)] = np.nan
    return d_static


def build_plate(video: str, cfg: PlateConfig, force: bool = False) -> None:
    out = plate_dir(video)
    if (out / "plate_meta.json").exists() and not force:
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    G = camera(video)
    K, T = G["K"], G["T"]
    D10, R, dets, hands = depth(video), tracks(video), Detections(video), hand_masks(video)
    ks = _subsample([int(k) for k in R["keyframes"]], cfg.max_frames)
    n = len(ks)
    kernel = np.ones((2 * cfg.mask_dilation_px + 1,) * 2, np.uint8)
    dirs = rays_lo(K) @ T[:3, :3].T
    s_plane = -T[2, 3] / dirs[..., 2]  # camera depth of the table plane along each ray
    table = cv2.erode(G["table"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(
        bool
    )  # interior only: edges see the floor

    Ds = np.full((n, H, W), np.nan, np.float32)
    for i, k in enumerate(ks):
        bad = _object_mask(dets, R, k)
        if k in hands:
            bad |= hands[k]
        bad = cv2.dilate(bad.astype(np.uint8), kernel).astype(bool)
        d = np.asarray(D10[k], np.float32) * G["scale"][k]
        off_plane = table & (np.abs(d - s_plane) > np.maximum(0.03, 0.03 * s_plane))
        d[bad | ~(d > cfg.min_static_depth_m) | off_plane] = np.nan
        Ds[i] = d
    d_static = static_depth(Ds, cfg)

    agree = (np.abs(Ds - d_static[None]) < np.maximum(0.02, 0.025 * d_static[None])) & np.isfinite(Ds)
    imgs = np.stack([frame(video, k) for k in ks])
    A = np.repeat(np.repeat(agree, 2, 1), 2, 2)
    plate = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    for y0 in range(0, IMG_H, ROW_CHUNK):
        chunk = imgs[:, y0 : y0 + ROW_CHUNK].astype(np.float32)
        chunk[~A[:, y0 : y0 + ROW_CHUNK]] = np.nan
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # pixels never seen: all-NaN median, inpainted below
            plate[y0 : y0 + ROW_CHUNK] = np.nan_to_num(np.nanmedian(chunk, 0), nan=0).clip(0, 255).astype(np.uint8)
    valid = A.sum(0) >= cfg.min_colour_samples
    invented = (~valid).astype(np.uint8)
    if invented.any():
        plate = cv2.inpaint(plate, cv2.dilate(invented, np.ones((3, 3), np.uint8)), 7, cv2.INPAINT_TELEA)

    valid_lo = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & np.isfinite(
        d_static
    )
    dfill = d_static.copy()
    on_table = ~valid_lo & table & (s_plane > 0)
    dfill[on_table] = s_plane[on_table]
    rest = ~np.isfinite(dfill) | (~valid_lo & ~on_table)
    if rest.any():
        _, idx = distance_transform_edt(rest, return_indices=True)
        dfill = dfill[idx[0], idx[1]]

    cv2.imwrite(str(out / "plate_raw.png"), plate)
    np.save(out / "plate_depth.npy", dfill.astype(np.float32))
    cv2.imwrite(str(out / "plate_valid_raw.png"), valid.astype(np.uint8) * 255)
    meta = dict(
        video=video,
        n_frames=n,
        keyframes=[ks[0], ks[-1]],
        bin=cfg.depth_bin,
        min_support=[cfg.min_support_frames, cfg.min_support_frac],
        min_colour_samples=cfg.min_colour_samples,
        invented_frac=float(1 - valid.mean()),
        invented_frac_table=float(1 - valid_lo[G["table"]].mean()),
        depth_invented_frac=float(1 - valid_lo.mean()),
        wall_s=time.time() - t0,
        method="temporal stability (farthest well-supported depth cluster; colour median over depth-agreeing frames); "
        "tracked objects + hand masks cut",
    )
    write_json(meta, out / "plate_meta.json")
    _proof_sheet(video, ks, plate, dfill, valid, meta)
    pts = person_cloud(video, cfg)
    print(
        f"{video}: {n} frames, invented {meta['invented_frac'] * 100:.2f} % "
        f"(table {meta['invented_frac_table'] * 100:.2f} %), "
        f"{len(pts)} body points, {meta['wall_s']:.0f} s",
        flush=True,
    )


def _label(img: np.ndarray, text: str, org=(12, 32), scale: float = 0.9) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _proof_sheet(video, ks, plate, dfill, valid, meta) -> None:
    """raw frame | static colour | static depth | invented pixels."""
    k_raw = ks[len(ks) // 3]
    raw = frame(video, k_raw)
    dv = cv2.applyColorMap((np.clip((dfill - 0.3) / 2.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dv = cv2.resize(dv, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    inv = cv2.cvtColor(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    inv[~valid] = (0, 0, 255)
    tiles = [raw, plate.copy(), dv, inv]
    labels = [
        f"raw frame k={k_raw}",
        "static colour (temporal stability)",
        "static depth (0.3-2.8 m)",
        f"invented (red): {meta['invented_frac'] * 100:.1f} % of px",
    ]
    for t, text in zip(tiles, labels, strict=False):
        _label(t, text)
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(
        str(plate_dir(video) / "plate_proof.jpg"), cv2.resize(sheet, (1440, 810)), [cv2.IMWRITE_JPEG_QUALITY, 88]
    )


def apply_person(video: str, cfg: PlateConfig) -> float:
    """inpaint the SAM 3 person / foot pixels found on the raw plate (feet standing still survive temporal stability).

    Writes plate.png and plate_valid.png (those pixels marked invented); without a person mask they equal the raw files.
    """
    d = plate_dir(video)
    raw = cv2.imread(str(d / "plate_raw.png"))
    valid = cv2.imread(str(d / "plate_valid_raw.png"), 0) > 127
    pm = d / "plate_person.png"
    M = cv2.imread(str(pm), 0) > 127 if pm.exists() else np.zeros(valid.shape, bool)
    out = cv2.inpaint(raw, M.astype(np.uint8) * 255, cfg.person_inpaint_radius, cv2.INPAINT_TELEA) if M.any() else raw
    cv2.imwrite(str(d / "plate.png"), out)
    cv2.imwrite(str(d / "plate_valid.png"), (valid & ~M).astype(np.uint8) * 255)
    shown = raw.copy()
    shown[M] = (0.5 * shown[M] + [0, 0, 127]).astype(np.uint8)
    sheet = np.hstack([shown, out])
    for text, x in (("plate before (red = SAM 3 person/foot, inpainted)", 10), ("plate after", 970)):
        cv2.putText(sheet, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(d / "plate_person_proof.jpg"), cv2.resize(sheet, (1440, 405)))
    meta = json.loads((d / "plate_meta.json").read_text())
    meta["person_inpainted_frac"] = float(M.mean())
    meta["invented_frac_total"] = float(1 - (valid & ~M).mean())
    write_json(meta, d / "plate_meta.json")
    return float(M.mean())


def person_cloud(video: str, cfg: PlateConfig, pts_per_frame: int = 800) -> np.ndarray:
    """table-frame points nearer than the static background and not on a tracked object = the person's body and arms,
    pooled over keyframes. scene.py places the robot shoulders where the person's are. -> person_pts.npy"""
    G = camera(video)
    K, T = G["K"], G["T"]
    D10, R, dets = depth(video), tracks(video), Detections(video)
    pd = np.load(plate_dir(video) / "plate_depth.npy")
    rays = rays_lo(K)
    ks = _subsample([int(k) for k in R["keyframes"]], cfg.person_cloud_frames)
    pts = []
    rng = np.random.default_rng(0)
    for k in ks:
        objm = cv2.dilate(_object_mask(dets, R, k).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        d = np.asarray(D10[k], np.float32) * G["scale"][k]
        fg = (d < pd - BODY_MARGIN_M) & (d > 0.05) & ~objm
        if fg.sum() < 50:
            continue
        P = (rays[fg] * d[fg][:, None]) @ T[:3, :3].T + T[:3, 3]
        if len(P) > pts_per_frame:
            P = P[rng.choice(len(P), pts_per_frame, replace=False)]
        pts.append(P)
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    np.save(plate_dir(video) / "person_pts.npy", P.astype(np.float32))
    return P


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Static background plate per video (temporal stability) and person inpainting."
    )
    ap.add_argument("videos", nargs="+", help="video names, e.g. IMG_2533")
    ap.add_argument(
        "--apply-person", action="store_true", help="inpaint plate_person.png (from person_mask.py) into plate.png"
    )
    ap.add_argument("--force", action="store_true", help="rebuild plates that already exist")
    ap.add_argument("--config", default=None, help="config file (default configs/twin.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config).plate
    for v in args.videos:
        if args.apply_person:
            print(f"{v}: person pixels inpainted {apply_person(v, cfg) * 100:.2f} %", flush=True)
        else:
            build_plate(v, cfg, args.force)


if __name__ == "__main__":
    main()
