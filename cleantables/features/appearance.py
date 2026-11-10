"""Appearance tokens for the watch data, keyed by each video's own track set.

Same rule as the `app` stage of cleantables.features.video, applied to every sample: per (track set, segment, object), the median
over up to APP_N of the segment's keyframes (time order) of the V-JEPA 2 / DINOv2 grid pooled inside the object's mask, plus the
location-only code and the mirrored versions. Grid rows come from the sample's feature row (base folder + feats_new, one row space);
masks from that row's mask set, so only samples whose track set is that set (mask_main) are used and get a key. CPU only; reads the
saved grids.

  key = "<track_set>|<group>:<obj_id>"  -> <feats_new>/app_watch.npz (read by cleantables.worldmodel.data when present, gated by mask_main)
  $VLA_PY -m cleantables.features.appearance [--limit_segs N]
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from cleantables.worldmodel.common import OUT, jdump, log

W = OUT / "wm_v2_world"


def app_key(track_set, group, oid):
    return f"{track_set}|{group}:{int(oid)}"


def main():
    import cleantables.features.video as VF
    from cleantables.worldmodel.data import RowCat
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(W / "data" / "tasks3_watch.npz"))
    ap.add_argument("--base", default=os.environ.get("WM_V2_FEATS", str(OUT / "wm_v2_seg" / "feats")))
    ap.add_argument("--new", default=os.environ.get("WM_V2_FEATS_EXT", str(W / "feats_new")))
    ap.add_argument("--out", default=None); ap.add_argument("--limit_segs", type=int, default=None)
    a = ap.parse_args()
    B, N2 = Path(a.base), Path(a.new); out = Path(a.out) if a.out else N2 / "app_watch.npz"
    Z = np.load(a.data, allow_pickle=True)
    ts, grp = Z["track_set"].astype(str), Z["group"].astype(str); kk = Z["snap_k"].astype(int)
    oid, om = Z["obj_id"].astype(int), Z["obj_mask"].astype(bool); fr = Z["feat_row"].astype(np.int64); mm = Z["mask_main"].astype(bool)
    VJ = RowCat(np.load(B / "vj_grid.npy", mmap_mode="r"), np.load(N2 / "vj_grid.npy", mmap_mode="r"))
    DN = RowCat(np.load(B / "dn_grid.npy", mmap_mode="r"), np.load(N2 / "dn_grid.npy", mmap_mode="r"))
    M0, M1 = VF._loaded(B / "masks.npz"), VF._loaded(N2 / "masks.npz")
    J = max(M0["cov_vj"].shape[1], M1["cov_vj"].shape[1])
    padj = lambda x: np.pad(x, [(0, 0), (0, J - x.shape[1])] + [(0, 0)] * (x.ndim - 2))   # noqa: E731
    CV = np.concatenate([padj(M0["cov_vj"]), padj(M1["cov_vj"])]); CD = np.concatenate([padj(M0["cov_dn"]), padj(M1["cov_dn"])])
    PEv = VF.posenc(VF.VJ_HW)
    segs = defaultdict(list)                                                    # (track set, group) -> sample indices
    for i in np.flatnonzero(mm):
        segs[(ts[i], grp[i])].append(i)
    keys_seg = sorted(segs)[:a.limit_segs] if a.limit_segs else sorted(segs)
    keys, av, ad, ac, avf, adf, nused = [], [], [], [], [], [], []
    for q, sk in enumerate(keys_seg):
        idx = segs[sk]
        objs = sorted({int(oid[i, s]) for i in idx for s in range(oid.shape[1]) if om[i, s] and oid[i, s] >= 0 and oid[i, s] < J})
        rk = {int(fr[i]): int(kk[i]) for i in idx}
        rows = sorted(rk, key=rk.get)                                          # the segment's keyframes in time order
        acc = {j: ([], [], [], [], []) for j in objs}
        for r in rows:
            need = [j for j in objs if len(acc[j][0]) < VF.APP_N and CV[r, j].astype(np.float32).sum() / 255 > VF.COV_MIN]
            if not need:
                if all(len(acc[j][0]) >= VF.APP_N for j in objs):
                    break
                continue
            vj, dn = VJ[r].astype(np.float32), DN[r].astype(np.float32)        # (2 flips, h, w, D)
            cvj = CV[r, need].astype(np.float32) / 255; cdn = CD[r, need].astype(np.float32) / 255
            pv, pd, pc = VF._pool(vj[0], cvj), VF._pool(dn[0], cdn), VF._pool(PEv, cvj)
            pvf, pdf = VF._pool(vj[1], cvj[:, :, ::-1]), VF._pool(dn[1], cdn[:, :, ::-1])
            for n_, j in enumerate(need):
                for L, x in zip(acc[j], (pv, pd, pc, pvf, pdf)):
                    L.append(x[n_])
        for j in objs:
            fv, fd, fc, fvf, fdf = acc[j]
            keys.append(app_key(sk[0], sk[1], j)); nused.append(len(fv))
            med = lambda L, d: np.nanmedian(L, 0) if L else np.zeros(d, np.float32)   # noqa: E731
            av.append(med(fv, 1024)); ad.append(med(fd, 1024)); ac.append(med(fc, PEv.shape[-1])); avf.append(med(fvf, 1024)); adf.append(med(fdf, 1024))
        if q % 20 == 0:
            log(f"app_watch: {q + 1}/{len(keys_seg)} segments, {len(keys)} tokens")
    np.savez_compressed(out, keys=np.array(keys), app_vj=np.array(av, np.float16), app_dn=np.array(ad, np.float16),
                        app_ctl=np.array(ac, np.float16), app_vj_flip=np.array(avf, np.float16), app_dn_flip=np.array(adf, np.float16),
                        n_frames=np.array(nused, np.int16))
    # coverage: share of object tokens (obj_kind 0, in mask) that get an appearance token, overall and per video / track-set kind
    kset = {k for k, n in zip(keys, nused) if n > 0}
    has = np.zeros(om.shape, bool)
    for i in range(len(ts)):
        for s in range(oid.shape[1]):
            if om[i, s] and oid[i, s] >= 0:
                has[i, s] = app_key(ts[i], grp[i], oid[i, s]) in kset
    obj = om & (Z["obj_kind"] == 0)
    vid = Z["video"].astype(str)
    rep = dict(segments=len(keys_seg), tokens=len(keys), with_frames=int(len(kset)), token_share=float(has[obj].mean()),
               token_share_main=float(has[obj & (ts == vid)[:, None]].mean()), token_share_other=float(has[obj & (ts != vid)[:, None]].mean()),
               per_video={v: float(has[obj & (vid == v)[:, None]].mean()) for v in np.unique(vid)})
    jdump(rep, out.with_suffix(".json"))
    log(f"app_watch: {len(keys)} (segment, object) tokens, {len(kset)} with a visible frame; object-token coverage {rep['token_share']:.3f} -> {out}")


if __name__ == "__main__":
    main()
