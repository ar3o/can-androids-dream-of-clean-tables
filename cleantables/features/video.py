"""Frozen video features for the world model, computed once and saved; only the small world model is trained, on these features.
Encoders and settings as in cleantables.features.encoders:
  V-JEPA 2 ViT-L: 32 frames at 10 fps ending at the snapshot frame, aspect-correct 448 x 256, last layer (24), last tubelet -> (16, 28, 1024)
  DINOv2-L (+ registers): the snapshot frame, 448 x 252, last layer -> (18, 32, 1024)
both also for the horizontally flipped input (the world model's mirror augmentation).

Stages (each resumable; a stage that is complete is skipped):
  rows    (CPU) the frames to encode: every tasks3c snapshot (video, snap_k) + each episode's start and end frame (k0, k1)
  masks   (CPU) per row and per tracked object (tracks.pkl index = obj_id): coverage of its SAM 3 mask (sam3_vinv.npz detection of that
          object nearest in time, within 2 frames) on both patch grids; plus the raw-colour grid (control)
  grids   (GPU) the V-JEPA and DINOv2 grids, plain and flipped -> fp16 memmaps
  objg    (CPU) per tasks3c sample and object slot: "does the goal spot look like the goal?": the cosine distance between
          the snapshot and the episode's goal (end) frame, both pooled inside the object's mask in the goal frame, and the same for the
          episode start (so the model can normalise); plus the DINOv2 whole-grid mean patch cosine distance now->goal and start->goal.
          Location-only control: the same pooling on a fixed position code (chance by construction).
  app     (CPU) per (episode, object): appearance tokens = median of the pooled V-JEPA / DINOv2 features inside the object's mask over the
          first 15 snapshot rows of the episode where the mask covers > 0.25 cells (the same rule as the DINOv2 crop feature);
          + the location-only control (position code pooled the same way)
  render  (GPU) robot vision path: the same encoders on the twin replay renders at the same moments (26 twins)
  aug     (GPU) reversed clips (vj_rev, dn_rev) and random crop / zoom + random clip start (vj_aug, dn_aug)

  source scripts/env.sh
  $VLA_PY -m cleantables.features.video rows masks            (CPU)
  $VLA_PY -m cleantables.features.video grids [--bs 1]        (GPU, ~3 GB)
  $VLA_PY -m cleantables.features.video objg app              (CPU)
  $VLA_PY -m cleantables.features.video render                (GPU)
  $VLA_PY -m cleantables.features.video grids --dev cpu --limit 2 --out /tmp/x   (CPU test)
Out: OUT/wm_v2/feats/
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from cleantables.worldmodel.common import OUT, ROOT, jdump, log

FE = Path(os.environ.get("WM_V2_FEATS", str(OUT / "wm_v2" / "feats")))          # override only for tests
VJ_HW, DN_HW = (16, 28), (18, 32)
MASK_HW = (270, 480)
T_CLIP = 32
APP_N = 15
COV_MIN = 0.25
RENDERS = Path(CT_DATA + "/vjepa_wm/renders")
TASKS = {t["id"]: t for t in json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]}


_LOADED = {}


def _loaded(p):
    """An npz fully in memory (indexing an NpzFile key decompresses the whole array on every access)."""
    if str(p) not in _LOADED:
        z = np.load(p, allow_pickle=True)
        _LOADED[str(p)] = {k: z[k] for k in z.files}
    return _LOADED[str(p)]


def data():
    return _loaded(ROOT / f"data/uwm/{os.environ.get('TASKS3_DATA', 'tasks3c')}.npz")


@lru_cache(maxsize=None)
def fmeta(v):
    return json.load(open(ROOT / "data/table_clean/_work" / v / "frames_meta.json"))


def ep_k(ep):
    t, fm = TASKS[ep], fmeta(TASKS[ep]["video"])
    return round(t["start_s"] * fm["fps_video"] / fm["stride"]), min(fm["n"] - 1, round(t["end_s"] * fm["fps_video"] / fm["stride"]))


# ------------------------------------------------------------------ rows
def rows(force=False):
    p = FE / "rows.json"
    if p.exists() and not force:
        return json.load(open(p))
    z = data()
    keys = set(zip(z["video"].astype(str), z["snap_k"].astype(int)))
    goal, start = {}, {}
    for ep in sorted(set(z["group"].astype(str))):
        k0, k1 = ep_k(ep)
        v = TASKS[ep]["video"]
        keys |= {(v, k0), (v, k1)}
        goal[ep], start[ep] = [v, k1], [v, k0]
    # the frame at the END of each sample's next move (target of the future-latent head: events.json f_end)
    fut = []
    evc = {}
    for v, eid, mv in zip(z["video"].astype(str), z["ev_id"].astype(int), z["mover"].astype(int)):
        if mv < 0 or eid < 0:
            fut.append(None); continue
        if v not in evc:
            evc[v] = {e["id"]: e for e in json.load(open(ROOT / "data/generic" / v / "events.json"))["events"]}
        k = min(int(evc[v][eid]["f_end"]), fmeta(v)["n"] - 1)
        fut.append((v, k)); keys.add((v, k))
    R = sorted([list(x) for x in keys])
    idx = {f"{v}:{k}": i for i, (v, k) in enumerate(R)}
    snap_row = [idx[f"{v}:{k}"] for v, k in zip(z["video"].astype(str), z["snap_k"].astype(int))]
    out = dict(rows=R, snap_row=snap_row, goal_row={e: idx[f"{v}:{k}"] for e, (v, k) in goal.items()},
               start_row={e: idx[f"{v}:{k}"] for e, (v, k) in start.items()}, n_snap=len(snap_row),
               future_row=[idx[f"{f[0]}:{f[1]}"] if f else -1 for f in fut])
    FE.mkdir(parents=True, exist_ok=True)
    jdump(out, p)
    log(f"rows: {len(R)} frames to encode ({len(snap_row)} snapshots, {len(goal)} episodes)")
    return out


# ------------------------------------------------------------------ masks
@lru_cache(maxsize=4)
def det_masks(v):
    z = np.load(ROOT / "data/generic" / v / "sam3_vinv.npz", allow_pickle=True)
    n, H, W = [int(x) for x in z["shape"]]
    return z["frame"].astype(int), np.unpackbits(z["masks"])[:n * H * W].reshape(n, H, W)


@lru_cache(maxsize=4)
def tracks(v):
    return pickle.load(open(ROOT / "data/generic" / v / "tracks.pkl", "rb"))["objects"]


def obj_mask(v, j, k, tol=2):
    """SAM 3 mask (270 x 480 bool) of tracked object j at work frame k: its detection nearest in time (|dk| <= tol), else None."""
    fr, M = det_masks(v)
    d = np.asarray(tracks(v)[j]["det"], int)
    if not len(d):
        return None
    dk = np.abs(fr[d] - k)
    i = int(np.argmin(dk))
    return M[d[i]].astype(bool) if dk[i] <= tol else None


def coverage(m, hw):
    return cv2.resize(m.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def masks(force=False):
    p = FE / "masks.npz"
    if p.exists() and not force:
        return _loaded(p)
    R = rows()["rows"]
    n = len(R)
    J = max(len(tracks(v)) for v in {r[0] for r in R})
    cv = np.zeros((n, J) + VJ_HW, np.uint8)
    cd = np.zeros((n, J) + DN_HW, np.uint8)
    pix = np.zeros((n, 2) + VJ_HW + (3,), np.float16)
    nobj = np.zeros(n, np.int16)
    for i, (v, k) in enumerate(R):
        nobj[i] = len(tracks(v))
        for j in range(nobj[i]):
            m = obj_mask(v, j, k)
            if m is not None:
                cv[i, j] = np.round(255 * coverage(m, VJ_HW)); cd[i, j] = np.round(255 * coverage(m, DN_HW))
        im = cv2.cvtColor(cv2.imread(str(ROOT / "data/table_clean/_work" / v / "frames" / f"{k}.jpg")), cv2.COLOR_BGR2RGB)
        g = cv2.resize(cv2.resize(im, (448, 256), interpolation=cv2.INTER_AREA), (VJ_HW[1], VJ_HW[0]), interpolation=cv2.INTER_AREA)
        g = (g.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        pix[i, 0], pix[i, 1] = g, g[:, ::-1]
        if i % 500 == 0:
            log(f"masks {i}/{n}")
    np.savez_compressed(p, cov_vj=cv, cov_dn=cd, pix=pix, nobj=nobj)
    log(f"masks: {n} rows, up to {J} objects")
    return _loaded(p)


# ------------------------------------------------------------------ grids (GPU)
def frame_cache():
    c = {}

    def load(v, k):
        k = max(0, min(int(k), fmeta(v)["n"] - 1))
        if (v, k) not in c:
            if len(c) > 400:
                c.clear()
            c[(v, k)] = cv2.cvtColor(cv2.imread(str(ROOT / "data/table_clean/_work" / v / "frames" / f"{k}.jpg")), cv2.COLOR_BGR2RGB)
        return c[(v, k)]
    return load


def _memmap(name, shape, mode):
    return np.lib.format.open_memmap(FE / name, mode=mode, dtype=np.float16, shape=shape)


def grids(dev="cuda", bs=1, limit=None, which=("vj", "dn")):
    import torch
    from cleantables.features.encoders import Encoder
    R = rows()["rows"]
    n = len(R) if limit is None else min(limit, len(R))
    load = frame_cache()
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get("WM_GPU_GB", "3.5")) / 23.5))
    for name, enc_name, hw, T in (("vj", "vj2l", VJ_HW, T_CLIP), ("dn", "dino2l", DN_HW, 1)):
        if name not in which:
            continue
        f, done_f = FE / f"{name}_grid.npy", FE / f"{name}_grid.done.npy"
        mode = "r+" if f.exists() else "w+"
        G = _memmap(f"{name}_grid.npy", (len(R), 2) + hw + (1024,), mode)
        done = np.load(done_f) if done_f.exists() else np.zeros(len(R), bool)
        enc = Encoder(enc_name, layers=(24,), dev=dev)
        todo = [i for i in range(n) if not done[i]]
        t0 = time.time()
        for b0 in range(0, len(todo), bs):
            ib = todo[b0:b0 + bs]
            X = np.stack([np.stack([load(R[i][0], R[i][1] - (T - 1 - t)) for t in range(T)]) for i in ib])
            for fl in (0, 1):
                Xi = X if fl == 0 else np.ascontiguousarray(X[:, :, :, ::-1])
                g = enc(Xi)[24][0]                                                   # (B, h, w, 1024) fp16
                assert tuple(g.shape[1:3]) == hw, (g.shape, hw)
                G[ib, fl] = g.float().cpu().numpy().astype(np.float16)
            done[ib] = True
            if (b0 // bs) % 50 == 0:
                G.flush(); np.save(done_f, done)
                el = time.time() - t0
                pk = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
                log(f"grids {name}: {b0 + len(ib)}/{len(todo)} ({el / max(1, b0 + len(ib)):.2f} s/frame, peak {pk:.2f} GB)")
        G.flush(); np.save(done_f, done)
        del enc
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"grids {name}: done {int(done.sum())}/{len(R)}")


N_AUG = 2


def aug_params(n, seed=0):
    """Per row and variant: (x0, y0, s, L): a crop of the 960x540 frame keeping its aspect (side fraction s in [0.70, 0.95], offset
    uniform in the remaining margin, as fractions of the frame), and the clip's visible length L in [8, 32] frames (an episode clip
    starting at a random point: the first 32 - L frames repeat its first visible frame)."""
    r = np.random.default_rng(seed)
    s = r.uniform(0.70, 0.95, (n, N_AUG))
    x0, y0 = r.uniform(0, 1, (n, N_AUG)) * (1 - s), r.uniform(0, 1, (n, N_AUG)) * (1 - s)
    L = r.integers(8, T_CLIP + 1, (n, N_AUG))
    return np.stack([x0, y0, s, L], -1).astype(np.float32)


def aug(dev="cuda", bs=1, limit=None, which=("rev", "crop"), n_crop=None):
    """Augmented encodings of every row, same encoders and settings as `grids`:
      rev    vj_rev (rows, 2 flip): the same 32 frames played backwards (k, k-1, ..., k-31; the clip ends at frame k-31) and
             dn_rev (rows, 2 flip): DINOv2 of frame k-31 (the frame the reversed clip ends on). Negatives for progress / STOP.
      crop   vj_aug / dn_aug (rows, N_AUG): random crop + zoom of the frames (aug_params, saved in aug_params.npy) and a random visible
             clip length; not flipped (mirrored samples use the plain flipped encoding)."""
    import torch
    from cleantables.features.encoders import Encoder
    R = rows()["rows"]
    n = len(R) if limit is None else min(limit, len(R))
    load = frame_cache()
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get("WM_GPU_GB", "3.5")) / 23.5))
    AP = FE / "aug_params.npy"
    if not AP.exists():
        np.save(AP, aug_params(len(R)))
    P = np.load(AP)

    def crop(img, p):
        H, W = img.shape[:2]; x0, y0, s = int(p[0] * W), int(p[1] * H), p[2]
        return img[y0:y0 + int(round(s * H)), x0:x0 + int(round(s * W))]

    jobs = []
    if "rev" in which:
        jobs += [("vj_rev", "vj2l", VJ_HW, 2), ("dn_rev", "dino2l", DN_HW, 2)]
    if "crop" in which:
        nc = N_AUG if n_crop is None else n_crop                                  # fewer crop variants (new rows, disk)
        jobs += [("vj_aug", "vj2l", VJ_HW, nc), ("dn_aug", "dino2l", DN_HW, nc)]
    for name, enc_name, hw, nv in jobs:
        f, done_f = FE / f"{name}.npy", FE / f"{name}.done.npy"
        G = _memmap(f"{name}.npy", (len(R), nv) + hw + (1024,), "r+" if f.exists() else "w+")
        done = np.load(done_f) if done_f.exists() else np.zeros(len(R), bool)
        enc = Encoder(enc_name, layers=(24,), dev=dev)
        todo = [i for i in range(n) if not done[i]]
        t0 = time.time()
        T = T_CLIP if name.startswith("vj") else 1
        for b0 in range(0, len(todo), bs):
            ib = todo[b0:b0 + bs]
            for v in range(nv):
                if name == "vj_rev":                                                      # frames k, k-1, ..., k-31
                    X = np.stack([np.stack([load(R[i][0], R[i][1] - t) for t in range(T)]) for i in ib])
                elif name == "dn_rev":
                    X = np.stack([load(R[i][0], R[i][1] - (T_CLIP - 1))[None] for i in ib])
                elif name == "vj_aug":
                    X = np.stack([np.stack([crop(load(R[i][0], R[i][1] - min(T - 1 - t, int(P[i, v, 3]) - 1)), P[i, v]) for t in range(T)]) for i in ib])
                else:
                    X = np.stack([crop(load(R[i][0], R[i][1]), P[i, v])[None] for i in ib])
                if name.endswith("_rev") and v == 1:
                    X = np.ascontiguousarray(X[:, :, :, ::-1])
                g = enc(np.ascontiguousarray(X))[24][0]
                assert tuple(g.shape[1:3]) == hw, (g.shape, hw)
                G[ib, v] = g.float().cpu().numpy().astype(np.float16)
            done[ib] = True
            if (b0 // bs) % 50 == 0:
                G.flush(); np.save(done_f, done)
                el = time.time() - t0
                pk = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
                log(f"aug {name}: {b0 + len(ib)}/{len(todo)} ({el / max(1, b0 + len(ib)):.2f} s/row, peak {pk:.2f} GB)")
        G.flush(); np.save(done_f, done)
        del enc
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"aug {name}: done {int(done.sum())}/{len(R)}")


# ------------------------------------------------------------------ objg + appearance (CPU, from the saved grids)
def _pool(G, cov):
    """G (h, w, D) float32, cov (J, h, w) coverage 0..1 -> (J, D) mean inside each mask (NaN where the mask covers <= COV_MIN cells)."""
    ws = cov.reshape(len(cov), -1).sum(1)
    P = cov.reshape(len(cov), -1) @ G.reshape(-1, G.shape[-1]) / np.maximum(ws, 1e-6)[:, None]
    P[ws <= COV_MIN] = np.nan
    return P


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def posenc(hw, dim=128, seed=0):
    """Location-only code: a fixed random Fourier code of each cell's (x, y), independent of the image."""
    yy, xx = np.meshgrid(np.linspace(0, 1, hw[0]), np.linspace(0, 1, hw[1]), indexing="ij")
    Bf = np.random.default_rng(seed).standard_normal((2, dim // 2)) * 6.0
    ang = np.stack([xx, yy], -1) @ Bf
    return np.concatenate([np.sin(ang), np.cos(ang)], -1).astype(np.float32)


def objg(force=False):
    p = FE / "objg.npz"
    if p.exists() and not force:
        return _loaded(p)
    z = data(); Rj = rows(); M = masks()
    VJ = np.load(FE / "vj_grid.npy", mmap_mode="r"); DN = np.load(FE / "dn_grid.npy", mmap_mode="r")
    assert np.load(FE / "vj_grid.done.npy").all() and np.load(FE / "dn_grid.done.npy").all(), "grids are not complete"
    PE = posenc(VJ_HW)
    N, S = len(z["group"]), z["obj_id"].shape[1]
    out = np.full((N, S, 2), np.nan, np.float32)          # [d(now, goal), d(start, goal)] in the goal-frame mask, V-JEPA
    ctl = np.full((N, S, 2), np.nan, np.float32)          # the same with the position code (location-only control)
    dino = np.full((N, 2), np.nan, np.float32)            # DINOv2 whole-grid mean patch cosine distance: now->goal, start->goal
    cache = {}
    for n in range(N):
        ep = str(z["group"][n]); r, rg, rs = Rj["snap_row"][n], Rj["goal_row"][ep], Rj["start_row"][ep]
        if ep not in cache:
            covg = M["cov_vj"][rg].astype(np.float32) / 255
            gv = VJ[rg, 0].astype(np.float32)
            sv = VJ[rs, 0].astype(np.float32)
            Pg, Ps = _unit(_pool(gv, covg)), _unit(_pool(sv, covg))
            Qg = _unit(_pool(PE, covg))
            gd = _unit(DN[rg, 0].astype(np.float32)); sd = _unit(DN[rs, 0].astype(np.float32))
            cache = {ep: (covg, Pg, Ps, Qg, gd, float(1 - (sd * gd).sum(-1).mean()))}
        covg, Pg, Ps, Qg, gd, d_start_dn = cache[ep]
        Pn = _unit(_pool(VJ[r, 0].astype(np.float32), covg))
        d_now, d_st = 1 - (Pn * Pg).sum(-1), 1 - (Ps * Pg).sum(-1)
        c_now = 1 - (Qg * Qg).sum(-1)                                      # the code at the same places is identical -> 0 (chance)
        nd = _unit(DN[r, 0].astype(np.float32))
        dino[n] = [float(1 - (nd * gd).sum(-1).mean()), d_start_dn]
        for s in range(S):
            j = int(z["obj_id"][n, s])
            if j < 0 or j >= len(d_now):
                continue
            out[n, s] = [d_now[j], d_st[j]]
            ctl[n, s] = [c_now[j], c_now[j]]
        if n % 500 == 0:
            log(f"objg {n}/{N}")
    np.savez_compressed(p, objg=out, objg_ctl=ctl, dino=dino)
    log(f"objg: share of (sample, object) with a goal-frame mask: {np.isfinite(out[..., 0]).mean():.2f}")
    return np.load(p)


def app(force=False):
    p = FE / "app.npz"
    if p.exists() and not force:
        return _loaded(p)
    z = data(); Rj = rows(); M = masks()
    VJ = np.load(FE / "vj_grid.npy", mmap_mode="r"); DN = np.load(FE / "dn_grid.npy", mmap_mode="r")
    PEv, PEd = posenc(VJ_HW), posenc(DN_HW)
    keys, av, ad, ac, avf, adf, nused = [], [], [], [], [], [], []
    for ep in sorted(set(z["group"].astype(str))):
        v = TASKS[ep]["video"]
        idx = np.where(z["group"].astype(str) == ep)[0]
        rr = sorted(set(Rj["snap_row"][i] for i in idx), key=lambda r: Rj["rows"][r][1])
        for j in range(len(tracks(v))):
            fv, fd, fc, fvf, fdf = [], [], [], [], []
            for r in rr:
                cvj = M["cov_vj"][r, j].astype(np.float32)[None] / 255
                if cvj.sum() <= COV_MIN:
                    continue
                cdn = M["cov_dn"][r, j].astype(np.float32)[None] / 255
                fv.append(_pool(VJ[r, 0].astype(np.float32), cvj)[0]); fd.append(_pool(DN[r, 0].astype(np.float32), cdn)[0])
                fc.append(_pool(PEv, cvj)[0])
                fvf.append(_pool(VJ[r, 1].astype(np.float32), cvj[:, :, ::-1])[0]); fdf.append(_pool(DN[r, 1].astype(np.float32), cdn[:, :, ::-1])[0])
                if len(fv) >= APP_N:
                    break
            keys.append(f"{ep}:{j}"); nused.append(len(fv))
            av.append(np.median(fv, 0) if fv else np.zeros(1024, np.float32))
            ad.append(np.median(fd, 0) if fd else np.zeros(1024, np.float32))
            ac.append(np.median(fc, 0) if fc else np.zeros(PEv.shape[-1], np.float32))
            avf.append(np.median(fvf, 0) if fvf else np.zeros(1024, np.float32))
            adf.append(np.median(fdf, 0) if fdf else np.zeros(1024, np.float32))
    np.savez_compressed(p, keys=np.array(keys), app_vj=np.array(av, np.float16), app_dn=np.array(ad, np.float16),
                        app_ctl=np.array(ac, np.float16), app_vj_flip=np.array(avf, np.float16), app_dn_flip=np.array(adf, np.float16),
                        n_frames=np.array(nused, np.int16))
    log(f"app: {len(keys)} (episode, object) tokens; with no visible frame: {int((np.array(nused) == 0).sum())}")
    return np.load(p)


# ------------------------------------------------------------------ render (GPU): the robot's vision path
def render(dev="cuda", bs=1, limit=None):
    import torch
    from cleantables.features.encoders import Encoder
    z = data()
    sel = []
    for n in range(len(z["group"])):
        ep = str(z["group"][n])
        mp = RENDERS / ep / "meta.json"
        if not mp.exists():
            continue
        k0, k1 = ep_k(ep)
        k = int(z["snap_k"][n])
        if k0 <= k <= k1:
            sel.append((n, ep, k - k0))
    if limit is not None:
        sel = sel[:limit]
    meta = {}
    p = FE / "render_index.json"
    jdump(dict(samples=[s[0] for s in sel], note="tasks3c sample index -> row of render_vj/render_dn; replay renders at the same moments"), p)
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get("WM_GPU_GB", "3.5")) / 23.5))
    c = {}

    def load(ep, j):
        if ep not in meta:
            meta[ep] = json.load(open(RENDERS / ep / "meta.json"))["variants"]["replay"]["files"]
        fs = meta[ep]
        j = max(0, min(j, len(fs) - 1))
        key = (ep, j)
        if key not in c:
            if len(c) > 400:
                c.clear()
            c[key] = cv2.cvtColor(cv2.imread(str(RENDERS / ep / fs[j])), cv2.COLOR_BGR2RGB)
        return c[key]
    for name, enc_name, hw, T in (("vj", "vj2l", VJ_HW, T_CLIP), ("dn", "dino2l", DN_HW, 1)):
        f, done_f = FE / f"render_{name}.npy", FE / f"render_{name}.done.npy"
        G = _memmap(f"render_{name}.npy", (len(sel),) + hw + (1024,), "r+" if f.exists() else "w+")
        done = np.load(done_f) if done_f.exists() else np.zeros(len(sel), bool)
        enc = Encoder(enc_name, layers=(24,), dev=dev)
        todo = [i for i in range(len(sel)) if not done[i]]
        for b0 in range(0, len(todo), bs):
            ib = todo[b0:b0 + bs]
            X = np.stack([np.stack([load(sel[i][1], sel[i][2] - (T - 1 - t)) for t in range(T)]) for i in ib])
            G[ib] = enc(X)[24][0].float().cpu().numpy().astype(np.float16)
            done[ib] = True
            if (b0 // bs) % 50 == 0:
                G.flush(); np.save(done_f, done); log(f"render {name}: {b0 + len(ib)}/{len(todo)}")
        G.flush(); np.save(done_f, done)
        del enc
    log(f"render: {len(sel)} snapshots from {len({s[1] for s in sel})} twins")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["rows", "masks", "grids", "objg", "app", "render", "aug"])
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--which", default="vj,dn")
    ap.add_argument("--out", default=None, help="other output folder (tests)")
    ap.add_argument("--aug_which", default="rev,crop")
    ap.add_argument("--n_crop", type=int, default=None, help="crop variants to encode (default N_AUG)")
    a = ap.parse_args()
    if a.out:
        FE = Path(a.out)
    FE.mkdir(parents=True, exist_ok=True)
    for s in a.stages:
        if s == "rows":
            rows()
        elif s == "masks":
            masks()
        elif s == "grids":
            grids(a.dev, a.bs, a.limit, tuple(a.which.split(",")))
        elif s == "objg":
            objg()
        elif s == "app":
            app()
        elif s == "render":
            render(a.dev, a.bs, a.limit)
        elif s == "aug":
            aug(a.dev, a.bs, a.limit, tuple(a.aug_which.split(",")), a.n_crop)
