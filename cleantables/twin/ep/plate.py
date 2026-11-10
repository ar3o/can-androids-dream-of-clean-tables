"""Per-video static background (colour + metric depth) from temporal stability; no person detector.

The phone is static (det3d T is one pose for the whole video), so every pixel sees the same static surface whenever nothing transient (the
person's body / arms, an object being carried) is in front of it. Per pixel, over the tracked keyframes (every 2nd 10 fps frame; long videos subsampled
to <= MAX_FR frames):
  1. metric depth d_k = depth10[k] * scale[k] (MoGe-3, per-frame scale to the table plane, det3d.npz);
     samples covered by a SAM 3 mask of a tracked object (any generic-pipeline track of this video: the movable task objects are simulated
     meshes, so they must not be baked into the background even when they sit still for minutes) or a SAM 3 'hand' mask are dropped;
  2. the log-depth histogram (bins of BIN = 3 %) is smoothed over 3 bins; the static depth is the farthest cluster supported by
     >= max(MIN_N, MIN_FRAC x valid samples) frames (transient things are always in front of the static surface; the farthest well-supported
     cluster rather than the most frequent one, because the person's body covers some pixels in more than half the frames);
     two geometric sanity rules: samples nearer than D_MIN = 0.6 m are never static (the phone stands on a tripod behind the person: their back /
     arms are the only things that close), and inside the (eroded) det3d table mask only samples on the table plane (+-3 cm) are static
     (the movable objects are simulated, the table top is a sim collider);
  3. colour = per-pixel median of the frames whose depth agrees with the static depth (|d - d_s| < max(2 cm, 2.5 % d_s)), at 960x540;
  4. pixels with < MIN_COL agreeing samples are invented: colour by OpenCV Telea inpainting, depth by the table plane (inside the table
     mask) or the nearest valid static depth; their fraction is reported and shown as a mask.
Out data/twin_table/ep/_video/<V>/: plate.png (960x540), plate_depth.npy (270x480 float32, camera z, m), plate_valid.png (960x540: 255 = observed),
plate_meta.json, plate_proof.jpg (raw frame | static colour | static depth | invented pixels).
  source scripts/env.sh; $SIM_PY -m cleantables.twin.ep.plate IMG_2533 [...]      (CPU)
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from cleantables.twin.ep.common import OUT, H, W, Dets, depth, frame, geom, hand_masks, jdump, rays_lo, tracks

BIN = 0.03
MIN_N, MIN_FRAC = 6, 0.08
MIN_COL = 3
MAX_FR = 300
DIL = 3
D_MIN = 0.6         # m: nothing static is this close to the phone (tripod behind the person); their back / arms are


def build(v, force=False):
    out = OUT / "_video" / v
    if (out / "plate_meta.json").exists() and not force:
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    G = geom(v); K, T = G["K"], G["T"]
    D10 = depth(v)
    R = tracks(v)
    dets = Dets(v)
    hm = hand_masks(v)
    ks = [int(k) for k in R["keyframes"]]
    if len(ks) > MAX_FR:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, MAX_FR).round().astype(int)]
    n = len(ks)
    ker = np.ones((2 * DIL + 1, 2 * DIL + 1), np.uint8)
    rays = rays_lo(K)
    dirs = rays @ T[:3, :3].T
    s_plane = -T[2, 3] / dirs[..., 2]                                 # camera depth (z) of the table plane z=0 along each ray
    tab = cv2.erode(G["table"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)        # table interior (edges: floor behind)
    off_plane = lambda d: tab & (np.abs(d - s_plane) > np.maximum(0.03, 0.03 * s_plane))   # noqa: E731
    Ds = np.full((n, H, W), np.nan, np.float32)
    for i, k in enumerate(ks):
        bad = np.zeros((H, W), bool)
        for d in R["kept"].get(k, []):
            bad |= dets.mask(d)
        if k in hm:
            bad |= hm[k]
        bad = cv2.dilate(bad.astype(np.uint8), ker).astype(bool)
        d = np.asarray(D10[k], np.float32) * G["scale"][k]
        d[bad | ~(d > D_MIN) | off_plane(d)] = np.nan
        Ds[i] = d
    # ---- static depth: farthest well-supported log-depth cluster
    ld = np.log(np.clip(Ds, 0.05, 50.0))
    lo = np.nanmin(ld)
    b = np.floor((ld - lo) / BIN)
    nb = int(np.nanmax(b)) + 3
    hist = np.zeros((nb, H, W), np.int16)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):
        ok = np.isfinite(b[i])
        np.add.at(hist, (b[i][ok].astype(int), yy[ok], xx[ok]), 1)
    hs = hist.astype(np.int32)
    hs[1:] += hist[:-1]; hs[:-1] += hist[1:]
    nval = np.isfinite(Ds).sum(0)
    thr = np.maximum(MIN_N, MIN_FRAC * nval)
    qual = hs >= thr[None]
    has = qual.any(0)
    far_bin = nb - 1 - np.argmax(qual[::-1], 0)                     # highest qualifying bin
    centre = lo + (far_bin + 0.5) * BIN
    sel = np.abs(ld - centre[None]) <= 1.5 * BIN
    dsel = np.where(sel, Ds, np.nan)
    with np.errstate(all="ignore"):
        d_static = np.nanmedian(dsel, 0)
    d_static[~has] = np.nan
    # ---- colour from agreeing frames (960x540)
    agree = np.abs(Ds - d_static[None]) < np.maximum(0.02, 0.025 * d_static[None])
    agree &= np.isfinite(Ds)
    imgs = np.stack([frame(v, k) for k in ks])                        # (n,540,960,3) uint8
    A = np.repeat(np.repeat(agree, 2, 1), 2, 2)                      # (n,540,960)
    plate = np.zeros((540, 960, 3), np.uint8)
    cnt = A.sum(0)
    for y0 in range(0, 540, 60):
        st = imgs[:, y0:y0 + 60].astype(np.float32)
        st[~A[:, y0:y0 + 60]] = np.nan
        with np.errstate(all="ignore"):
            plate[y0:y0 + 60] = np.nan_to_num(np.nanmedian(st, 0), nan=0).clip(0, 255).astype(np.uint8)
    valid = cnt >= MIN_COL
    inv = (~valid).astype(np.uint8)
    if inv.any():
        plate = cv2.inpaint(plate, cv2.dilate(inv, np.ones((3, 3), np.uint8)), 7, cv2.INPAINT_TELEA)
    # ---- depth fill
    valid_lo = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & np.isfinite(d_static)
    dfill = d_static.copy()
    m_tab = ~valid_lo & tab & (s_plane > 0)
    dfill[m_tab] = s_plane[m_tab]
    rest = ~np.isfinite(dfill) | (~valid_lo & ~m_tab)
    if rest.any():
        src = ~rest
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(~src, return_indices=True)
        dfill = dfill[idx[0], idx[1]]
    cv2.imwrite(str(out / "plate.png"), plate)
    np.save(out / "plate_depth.npy", dfill.astype(np.float32))
    cv2.imwrite(str(out / "plate_valid.png"), valid.astype(np.uint8) * 255)
    meta = dict(video=v, n_frames=n, keyframes=[ks[0], ks[-1]], bin=BIN, min_support=[MIN_N, MIN_FRAC], min_colour_samples=MIN_COL,
                invented_frac=float(1 - valid.mean()), invented_frac_table=float(1 - valid_lo[G["table"]].mean()),
                depth_invented_frac=float(1 - valid_lo.mean()), wall_s=time.time() - t0,
                method="temporal stability (farthest well-supported depth cluster; colour median over depth-agreeing frames); tracked objects + hand masks cut")
    jdump(meta, out / "plate_meta.json")
    # ---- proof
    raw = frame(v, ks[len(ks) // 3])
    dv = cv2.applyColorMap((np.clip((dfill - 0.3) / 2.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dv = cv2.resize(dv, (960, 540), interpolation=cv2.INTER_NEAREST)
    ivm = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY); ivm = cv2.cvtColor(ivm, cv2.COLOR_GRAY2BGR)
    ivm[~valid] = (0, 0, 255)
    tiles = [raw, plate, dv, ivm]
    labs = [f"raw frame k={ks[len(ks) // 3]}", "static colour (temporal stability)", "static depth (0.3-2.8 m)",
            f"invented (red): {meta['invented_frac'] * 100:.1f} % of px"]
    for t_, l_ in zip(tiles, labs):
        cv2.putText(t_, l_, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t_, l_, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out / "plate_proof.jpg"), cv2.resize(sheet, (1440, 810)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(v, f"{n} frames, invented {meta['invented_frac'] * 100:.2f} % (table {meta['invented_frac_table'] * 100:.2f} %), {meta['wall_s']:.0f}s", flush=True)




# ------------------------------------------------------------------ person handling
def apply_person(v):
    """Inpaint the SAM 3 person / foot pixels found on the plate (feet that stand still survive the temporal-stability test); the
    original is kept as plate_raw.png, those pixels are marked invented in plate_valid.png; proof plate_person_proof.jpg."""
    import shutil
    d = OUT / "_video" / v
    pm = d / "plate_person.png"
    if not pm.exists():
        return None
    if not (d / "plate_raw.png").exists():
        shutil.copy(d / "plate.png", d / "plate_raw.png")
        shutil.copy(d / "plate_valid.png", d / "plate_valid_raw.png")
    raw = cv2.imread(str(d / "plate_raw.png"))
    M = cv2.imread(str(pm), 0) > 127
    out = cv2.inpaint(raw, M.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA) if M.any() else raw
    cv2.imwrite(str(d / "plate.png"), out)
    val = cv2.imread(str(d / "plate_valid_raw.png"), 0) > 127
    cv2.imwrite(str(d / "plate_valid.png"), (val & ~M).astype(np.uint8) * 255)
    a = raw.copy(); a[M] = (0.5 * a[M] + [0, 0, 127]).astype(np.uint8)
    sheet = np.hstack([a, out])
    for t_, x_ in (("plate before (red = SAM 3 person/foot, inpainted)", 10), ("plate after", 970)):
        cv2.putText(sheet, t_, (x_, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(d / "plate_person_proof.jpg"), cv2.resize(sheet, (1440, 405)))
    meta = __import__("json").load(open(d / "plate_meta.json"))
    meta["person_inpainted_frac"] = float(M.mean())
    meta["invented_frac_total"] = float(1 - (val & ~M).mean())
    jdump(meta, d / "plate_meta.json")
    return float(M.mean())


def person_cloud(v, max_pts=60000):
    """3D points (table frame) of everything nearer than the static background by > 10 cm and not a tracked object (the person's body /
    arms), pooled over the keyframes; used by cleantables.twin.ep.build to place the robot shoulders where the person's are. -> _video/<V>/person_pts.npy"""
    G = geom(v); K, T = G["K"], G["T"]
    D10 = depth(v); R = tracks(v); dets = Dets(v)
    pd = np.load(OUT / "_video" / v / "plate_depth.npy")
    rays = rays_lo(K)
    ks = [int(k) for k in R["keyframes"]]
    if len(ks) > 150:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, 150).round().astype(int)]
    pts = []
    rng = np.random.default_rng(0)
    for k in ks:
        objm = np.zeros((H, W), bool)
        for d_ in R["kept"].get(k, []):
            objm |= dets.mask(d_)
        objm = cv2.dilate(objm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        d = np.asarray(D10[k], np.float32) * G["scale"][k]
        fg = (d < pd - 0.10) & (d > 0.05) & ~objm
        if fg.sum() < 50:
            continue
        P = (rays[fg] * d[fg][:, None]) @ T[:3, :3].T + T[:3, 3]
        if len(P) > 800:
            P = P[rng.choice(len(P), 800, replace=False)]
        pts.append(P)
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    np.save(OUT / "_video" / v / "person_pts.npy", P.astype(np.float32))
    return P


if __name__ == "__main__":
    force = "--force" in sys.argv
    for v in [a for a in sys.argv[1:] if a.startswith("IMG_")]:
        if "--person" in sys.argv:
            print(v, "person inpainted frac", apply_person(v), "person points", len(person_cloud(v)), flush=True)
        else:
            build(v, force)
