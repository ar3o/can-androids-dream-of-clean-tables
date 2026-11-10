"""Depth-model comparison on our own clips (5 clips, 5 recordings), used to choose the reconstruction depth backend.

  run:   PYTHONPATH=<depth_models.pythonpath(m)>:. $VLA_PY -m cleantables.perception.recon.bakeoff run MODEL [--device cpu|cuda]
  score: $VLA_PY -m cleantables.perception.recon.bakeoff score            -> data/processed/recon_v3/bakeoff/report.json + table on stdout

Frames per clip: 5 consecutive recon frames starting where the object track starts (object at rest) + 5 spread over the clip.
Masks used for scoring are the same for every model: container + object = SAM 3 (top detection of the task phrases, prob >= 0.5; container =
temporal median mask), table = a model-independent table mask, hand = MediaPipe landmark hull (dilated).
Metrics (all per clip, then median over clips):
  box_H / box_L: agreement with a hand-scale estimate, not ground truth: 0.295 x 0.169 x 0.205 m is the white container size estimated
      from the videos assuming MediaPipe hand size is metric (not a tape measurement). H = 97th pct height of container
      points above the fitted table plane (+ visible rim); L = longest side of the trimmed oriented rectangle of the container points projected
      on the table plane. Visible faces only, so L can be under-estimated when the long side faces away; H is the cleaner number.
  obj err: object length / width from its top-surface points (oriented rectangle in the object's own fitted plane) vs data/object_sizes.json,
      which is also a video-based estimate (not a tape measurement): agreement only, not ground truth.
  flat: RMS (mm) of the table points about their least-squares plane (5 % largest residuals trimmed); flat_rel = flat / median depth.
  jitter: median |d_t+1 - d_t| / d on table pixels away from hand/object, consecutive frames; raw and after per-frame scale alignment.
  s/frame: wall time (device noted).
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from cleantables.perception.recon.common import OUT, ROOT, WORK, read_work_frames
from cleantables.perception.recon.geometry import backproject, fit_plane_lsq, fit_plane_ransac, min_area_rect, plane_basis

CLIPS = ["IMG_2476_ep00", "IMG_2478_ep04", "IMG_2480_ep06", "IMG_2483_ep02", "IMG_2485_ep04"]
# self-consistency set: the same silver laptop ("a silver laptop") in all 7 recordings (+ the same white box, + the same person's hand)
LAPTOP = ["IMG_2476_ep04", "IMG_2478_ep09", "IMG_2480_ep06", "IMG_2481_ep09", "IMG_2483_ep08", "IMG_2484_ep04", "IMG_2485_ep08"]
ALL = CLIPS + [c for c in LAPTOP if c not in CLIPS]
BOX_TRUE = (0.295, 0.169, 0.205)
BO = OUT / "bakeoff"


def frames_for(clip, n):
    f = ROOT / f"data/processed/object_v2/{clip}/obj_track.npz"
    k0 = int(np.load(f)["frames"][0]) // 2 + 1 if f.exists() else 1
    k0 = min(k0, n - 5)
    cons = list(range(k0, k0 + 5))
    spread = [int(x) for x in np.linspace(0, n - 1, 7)[1:-1] if int(x) not in cons]
    return cons + spread[:5]


def run(model, dev):
    from cleantables.perception.recon.depth_models import load
    m = load(model, dev)
    (BO / model).mkdir(parents=True, exist_ok=True)
    for c in ALL:
        f = BO / model / f"{c}.npz"
        if f.exists():
            continue
        fr, meta = read_work_frames(c)
        idx = frames_for(c, len(fr))
        sel = [fr[i] for i in idx]
        t0 = time.time()
        kw = {}
        if model == "da3metric":     # needs a focal: the clip's MoGe-2 ViT-L FOV (the same K for both)
            kw["focal_px"] = float(np.load(BO / "moge2_vitl" / f"{c}.npz")["K"][0, 0])
        if model.endswith("_exif"):     # FOV forced to the phone metadata (lens focal, no zoom)
            from cleantables.perception.recon.depth_models import exif_fov_x
            kw["fov_x"] = exif_fov_x(c, sel[0].shape[1], sel[0].shape[0])[0]
        D, K, extra = m(sel, **kw)
        dt = (time.time() - t0) / len(sel)
        np.savez_compressed(f, depth=D.astype(np.float16), K=K, idx=np.array(idx), s_per_frame=dt, device=dev,
                            extra=json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in extra.items()}, default=float))
        print(f"{model} {c}: {dt:.2f} s/frame on {dev}, fx {K[0, 0]:.0f}", flush=True)


def _unpack_small(path, shape_hw, out_hw):
    bits = np.load(path)
    m = np.unpackbits(bits)[: shape_hw[0] * shape_hw[1]].reshape(shape_hw)
    return cv2.resize(m, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def masks(clip, idx, hw):
    h, w = hw
    small = (512, 288) if h > w else (288, 512)
    sc = ROOT / "data/processed/scene_v1" / clip
    table = _unpack_small(sc / "table_mask.npy", small, hw)
    box = _unpack_small(sc / "box_mask.npy", small, hw) if (sc / "box_mask.npy").exists() else np.zeros(hw, bool)
    obj = [None] * len(idx)
    src = "scene_v1 box_mask (no object mask)"
    if (WORK / clip / "sam3.npz").exists():        # SAM 3 masks of the task phrases (preferred)
        from cleantables.perception.recon.common import unpack
        sz = np.load(WORK / clip / "sam3.npz")
        shp = tuple(sz["shape"])
        m0, m1 = unpack(sz["p0_masks"], shp), unpack(sz["p1_masks"], shp)
        obj = [m0[k, 0] if sz["p0_probs"][k, 0] >= 0.5 else None for k in idx]
        bx = [m1[k, 0] for k in idx if sz["p1_probs"][k, 0] >= 0.5]
        box = np.median(np.stack(bx), 0) > 0.5 if bx else box
        src = "SAM 3"
    tr = np.load(ROOT / f"data/processed/markerless_v3/{clip}/hand_track.npz")
    hand = []
    for k in idx:
        m = np.zeros(hw, np.uint8)
        j = 2 * k
        if j < len(tr["score"]) and tr["score"][j] > 0:
            pts = tr["lm2d"][j].astype(np.float32)
            if np.isfinite(pts).all():
                cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1)
                m = cv2.dilate(m, np.ones((41, 41), np.uint8))
        hand.append(m.astype(bool))
    return table, box, obj, hand, src


def score_clip(model, clip, sizes):
    z = np.load(BO / model / f"{clip}.npz")
    D, K, idx = z["depth"].astype(np.float32), z["K"], list(z["idx"])
    hw = D.shape[1:]
    table, box, obj, hand, msrc = masks(clip, idx, hw)
    trz = np.load(ROOT / f"data/processed/markerless_v3/{clip}/hand_track.npz")
    tr_s, tr_l = trz["score"], trz["lm2d"]
    er = lambda m: cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    table_e, box_e = er(table), er(box)
    r = {}
    flats, rels, Hs, Ls, Ws, oL, oW = [], [], [], [], [], [], []
    for i in range(len(idx)):
        static_t = table_e & ~hand[i]
        if obj[i] is not None:
            static_t &= ~cv2.dilate(obj[i].astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        P, _ = backproject(D[i], K, static_t, stride=2)
        if len(P) < 500:
            continue
        n, d, inl = fit_plane_ransac(P, thresh=0.02 * np.median(P[:, 2]))
        Q = P[inl]
        n2, d2 = fit_plane_lsq(Q)
        res = np.abs(Q @ n2 + d2)
        res = res[res <= np.quantile(res, 0.95)]
        flats.append(float(np.sqrt(np.mean(res ** 2)) * 1000)); rels.append(flats[-1] / 1000 / float(np.median(P[:, 2])))
        if d2 < 0:
            n2, d2 = -n2, -d2
        if box_e.any() and i < 5:
            B, _ = backproject(D[i], K, box_e & ~hand[i], stride=1)
            if len(B) > 200:
                hgt = B @ n2 + d2
                Hs.append(float(np.quantile(hgt, 0.97)))
                ex, ey = plane_basis(n2)
                rr = min_area_rect(np.stack([B @ ex, B @ ey], 1))
                if rr:
                    Ls.append(rr[1][0]); Ws.append(rr[1][1])
        if obj[i] is not None and i < 5:
            om = er(obj[i]) & ~hand[i]
            O, _ = backproject(D[i], K, om, stride=1)
            if len(O) > 100 and om.sum() > 0.7 * obj[i].sum():
                no, do_, oin = fit_plane_ransac(O, thresh=0.01 * np.median(O[:, 2]))
                ex, ey = plane_basis(no)
                rr = min_area_rect(np.stack([O[oin] @ ex, O[oin] @ ey], 1))
                if rr:
                    oL.append(rr[1][0]); oW.append(rr[1][1])
    # jitter on consecutive frames 0..4
    J, Ja = [], []
    for i in range(4):
        m = table_e & ~hand[i] & ~hand[i + 1] & (D[i] > 0) & (D[i + 1] > 0)
        for o in (obj[i], obj[i + 1]):
            if o is not None:
                m &= ~cv2.dilate(o.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        if m.sum() < 500:
            continue
        a, b = D[i][m], D[i + 1][m]
        J.append(float(np.median(np.abs(b - a) / a)))
        s = np.median(a / b)
        Ja.append(float(np.median(np.abs(b * s - a))) * 1000)
    # palm width (index MCP - pinky MCP) of the same person: image distance x palm depth / fx; 80th pct (foreshortening only shrinks it)
    pw = []
    for i, k in enumerate(idx):
        j = 2 * k
        if j < len(tr_s) and tr_s[j] > 0 and np.isfinite(tr_l[j]).all():
            zz = [D[i][int(v), int(u)] for u, v in tr_l[j][[0, 5, 9, 13, 17]] if 0 <= int(u) < hw[1] and 0 <= int(v) < hw[0]]
            zz = [z_ for z_ in zz if z_ > 0]
            if len(zz) >= 3:
                pw.append(np.linalg.norm(tr_l[j][5] - tr_l[j][17]) * (np.median(zz) + 0.01) / K[0, 0])
    true = sizes["objects"][sizes["clip_objects"][clip]]
    r = dict(flat_mm=np.median(flats) if flats else None, flat_rel=np.median(rels) if rels else None,
             box_H=np.median(Hs) if Hs else None, box_L=np.median(Ls) if Ls else None, box_W=np.median(Ws) if Ws else None,
             obj_L=np.median(oL) if oL else None, obj_W=np.median(oW) if oW else None, obj_true=true[:2],
             jitter_rel=np.median(J) if J else None, jitter_aligned_mm=np.median(Ja) if Ja else None, s_per_frame=float(z["s_per_frame"]),
             palm_w=float(np.quantile(pw, 0.8)) if len(pw) >= 3 else None, device=str(z["device"]), mask_source=msrc, fx=float(K[0, 0]), median_depth=float(np.median(D[D > 0])))
    r["box_H_err"] = abs(r["box_H"] - BOX_TRUE[2]) / BOX_TRUE[2] if r["box_H"] else None
    r["box_L_err"] = abs(r["box_L"] - BOX_TRUE[0]) / BOX_TRUE[0] if r["box_L"] else None
    r["obj_L_err"] = abs(r["obj_L"] - true[0]) / true[0] if r["obj_L"] else None
    r["obj_W_err"] = abs(r["obj_W"] - true[1]) / true[1] if r["obj_W"] else None
    return r


def cv(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.std(v) / np.mean(v)) if len(v) >= 3 else None


def score():
    """Self-consistency scoring (no external ground truth): the same white box in every recording, the same laptop in 7 recordings, the
    same person's palm. A correct metric model gives the same size every time: CV = sd / mean across recordings (lower = better)."""
    sizes = json.load(open(ROOT / "data/object_sizes.json"))
    models = sorted(p.name for p in BO.iterdir() if p.is_dir())
    rep, tab = {}, {}
    for m in models:
        rep[m] = {c: score_clip(m, c, sizes) for c in ALL if (BO / m / f"{c}.npz").exists() and (WORK / c / "sam3.npz").exists()}
        R = rep[m]
        rec = lambda c: c.split("_")[1]
        def per_rec(key, clips):
            d = {}
            for c in clips:
                if c in R and R[c].get(key) is not None:
                    d.setdefault(rec(c), []).append(R[c][key])
            return [float(np.median(v)) for v in d.values()]
        box_H, box_L, box_W = per_rec("box_H", R), per_rec("box_L", R), per_rec("box_W", R)
        lap = [c for c in LAPTOP if c in R]
        lap_L, lap_W = per_rec("obj_L", lap), per_rec("obj_W", lap)
        palm = per_rec("palm_w", R)
        med = lambda k: float(np.median([r[k] for r in R.values() if r.get(k) is not None])) if any(r.get(k) is not None for r in R.values()) else None
        tab[m] = dict(n_clips=len(R), n_rec_box=len(box_H), cv_box_H=cv(box_H), cv_box_L=cv(box_L), cv_box_W=cv(box_W), n_rec_laptop=len(lap_L),
                      cv_laptop_L=cv(lap_L), cv_laptop_W=cv(lap_W), cv_palm=cv(palm), flat_mm=med("flat_mm"), flat_rel=med("flat_rel"),
                      jitter_rel=med("jitter_rel"), jitter_aligned_mm=med("jitter_aligned_mm"), s_per_frame=med("s_per_frame"),
                      box_HLW_median=[float(np.median(x)) if x else None for x in (box_H, box_L, box_W)],
                      laptop_LW_median=[float(np.median(x)) if x else None for x in (lap_L, lap_W)], palm_median=float(np.median(palm)) if palm else None,
                      agree_earlier_box_H=abs(np.median(box_H) - BOX_TRUE[2]) / BOX_TRUE[2] if box_H else None,
                      agree_earlier_laptop_L=abs(np.median(lap_L) - 0.304) / 0.304 if lap_L else None)
    json.dump(dict(per_clip=rep, table=tab, note="self-consistency only; 'agree_earlier_*' = agreement with the EARLIER pipeline's video/hand-scale "
                   "estimates (box 0.295x0.169x0.205, laptop 0.304 in object_sizes.json), NOT ground truth"), open(BO / "report.json", "w"), indent=1, default=float)
    keys = ["n_clips", "n_rec_box", "cv_box_H", "cv_box_L", "cv_box_W", "n_rec_laptop", "cv_laptop_L", "cv_laptop_W", "cv_palm", "flat_mm", "jitter_aligned_mm",
            "jitter_rel", "s_per_frame", "agree_earlier_box_H", "agree_earlier_laptop_L"]
    print("| model | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for m in models:
        print(f"| {m} | " + " | ".join("-" if tab[m][k] is None else (f"{tab[m][k]:.3f}" if isinstance(tab[m][k], float) else str(tab[m][k])) for k in keys) + " |")
    for m in models:
        print(m, {k: (np.round(v, 3).tolist() if v is not None else None) for k, v in tab[m].items() if k in ("box_HLW_median", "laptop_LW_median", "palm_median")})


if __name__ == "__main__":
    if sys.argv[1] == "run":
        dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
        run(sys.argv[2], dev)
    else:
        score()
