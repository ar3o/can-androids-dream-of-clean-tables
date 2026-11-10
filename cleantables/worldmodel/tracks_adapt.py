"""Convert the per-keyframe track dataset (WM_EVAL_OUT/wm_v3_tracks/tracks_all.npz, built by cleantables.perception.tracks)
into the format of the watch-data loader. One example per tracked keyframe (episodes and reset / setup stretches), no move labels.
  inputs  tokens, hands and V-JEPA tokens as given; HOIST-Former contact at k (score >= 0.5, the model's own threshold) where it covers k
  T1      end_xyz / end_present / end_yaw of the object's segment, episodes only, and only where the end pose comes from the same
          pose source as the token at k (end_src == obj_pose_src; a FoundationPose -> tracker switch would look like motion)
  T2      fut_xyz minus the token's position, where fut_src == obj_pose_src
  T3      the nearest keyframe of the same video at k + 4 s (within 0.3 s)
  T5      fut_contact >= 0.5 where HOIST-Former covers that step
The move fields are inert placeholders (SPEC defaults). obj_id is the tracks.pkl index (equal to the events.json id on the 20
original videos). Samples from another track set (track_set != video, e.g. IMG_2516's separately tracked reset stretches) keep
their tokens but get no image-position query and no appearance token, since the mask and appearance stores index the video's
own track set.
Feature rows: each (video, k) maps to a row of the base feature folder (WM_V2_FEATS) or of a new folder holding only the
missing rows (--feats_new; encode it on the GPU with WM_V2_FEATS=<feats_new> and cleantables.features.video grids aug
--aug_which crop). cleantables.worldmodel.data reads both (WM_V2_FEATS_EXT) through one row space: feat_row = base row,
or n_base + new row.
  $VLA_PY -m cleantables.worldmodel.tracks_adapt --tracks WM_EVAL_OUT/wm_v3_tracks/tracks_all.npz --out tasks3_watch --feats_new <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from cleantables.worldmodel.common import OUT, ROOT, jdump, log
from cleantables.worldmodel.data import HELD_TWINS

DST = Path(os.environ.get("WM_SUB_DST", str(OUT / "wm_v2_world" / "data")))
BASE_FE = Path(os.environ.get("WM_V2_FEATS", str(OUT / "wm_v2_seg" / "feats")))
EXTRA_KEYS = dict(rot_ok=((), np.uint8, 0), fp_move_ok=((), np.uint8, 0), fp_tok_n=((), np.int16, 0), path_rot=((32, 6), np.float32, np.nan),
                  end_rot=((6,), np.float32, np.nan))


def new_feats_folder(keys, out, base_fe, mask_set):
    """Create a feature folder holding only the rows (video, k) in keys: rows.json, aug_params, masks (as in
    cleantables.features.video.masks, from track set mask_set[(video, k)], since IMG_2513 / 2514 have no whole-video tracks.pkl),
    and links to the shared appearance / render stores. The grid and aug memmaps are left for cleantables.features.video to fill on the GPU."""
    import cv2
    import cleantables.features.video as VF
    out.mkdir(parents=True, exist_ok=True)
    jdump(dict(rows=[list(k) for k in keys], mask_set=[mask_set[k] for k in keys], snap_row=[], future_row=[], n_snap=0), out / "rows.json")
    np.save(out / "aug_params.npy", VF.aug_params(len(keys), seed=2))
    M0 = VF._loaded(base_fe / "masks.npz"); J = M0["cov_vj"].shape[1]
    n = len(keys)
    cv = np.zeros((n, J) + VF.VJ_HW, np.uint8); cd = np.zeros((n, J) + VF.DN_HW, np.uint8)
    pix = np.zeros((n, 2) + VF.VJ_HW + (3,), np.float16); nobj = np.zeros(n, np.int16)
    for q, (v, k) in enumerate(keys):
        ts = mask_set[(v, k)]
        nobj[q] = len(VF.tracks(ts))
        for j in range(min(J, nobj[q])):
            m = VF.obj_mask(ts, j, k)
            if m is not None:
                cv[q, j] = np.round(255 * VF.coverage(m, VF.VJ_HW)); cd[q, j] = np.round(255 * VF.coverage(m, VF.DN_HW))
        im = cv2.cvtColor(cv2.imread(str(VF.ROOT / "data/table_clean/_work" / v / "frames" / f"{k}.jpg")), cv2.COLOR_BGR2RGB)
        g = cv2.resize(cv2.resize(im, (448, 256), interpolation=cv2.INTER_AREA), (VF.VJ_HW[1], VF.VJ_HW[0]), interpolation=cv2.INTER_AREA)
        g = (g.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        pix[q, 0], pix[q, 1] = g, g[:, ::-1]
    np.savez_compressed(out / "masks.npz", cov_vj=cv, cov_dn=cd, pix=pix, nobj=nobj)
    for nm in ("app.npz", "render_vj.npy", "render_dn.npy", "render_index.json", "render_vj.done.npy", "render_dn.done.npy"):
        if (base_fe / nm).exists() and not (out / nm).exists():
            os.symlink((base_fe / nm).resolve(), out / nm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tracks", required=True); ap.add_argument("--out", default="tasks3_watch")
    ap.add_argument("--feats_new", default=str(OUT / "wm_v2_world" / "feats_new"))
    ap.add_argument("--episodes", default=str(ROOT / "data/labels/episodes_all.json"), help="for the held-out windows of the split")
    a = ap.parse_args()
    from cleantables.worldmodel.schema import SPEC, STR
    T = dict(np.load(a.tracks, allow_pickle=True)); N = len(T["snap_k"])
    vid = T["video"].astype(str); kk = T["snap_k"].astype(int); seg = T["group"].astype(str); tset = T["track_set"].astype(str)
    out = {k: np.full((N,) + sh, pad, dt) for k, (sh, dt, pad) in {**SPEC, **EXTRA_KEYS}.items()}
    for k in ("obj_geo", "obj_txt", "obj_crop", "obj_crop_ok", "obj_mask", "obj_kind", "obj_id", "hand_geo", "vj_tok", "vj_tok_flip", "vj_ok",
              "table_idx", "snap_k"):
        out[k] = T[k].astype(SPEC[k][1])
    out.update(video=vid, group=seg, task_text=T["task_text"].astype(str), dataset=np.full(N, "all"), split=np.full(N, "train"))
    episode = T["seg_kind"].astype(int) == 0
    # split: labelled episodes by id; unlabelled gaps and reset episodes (id ..._r<j>) within +-5 s of a held-out episode, or in a
    # held-out video, are test (a reset next to a test episode shows the same table)
    segs = [s for s in json.load(open(a.episodes))["segments"] if s["kind"] == "episode" and s["id"] in HELD_TWINS]
    base_ep = np.where(episode, seg, np.array([f"{v}~world" for v in vid]))
    reset = episode & np.array([re.search(r"_r\d+$", g) is not None for g in seg])
    t0, t1 = T["seg_t0"].astype(float), T["seg_t1"].astype(float)
    for i in np.flatnonzero(~episode | reset):                                # a reset episode is assigned whole (its T1 is its own end)
        lo, hi = (10 * t0[i], 10 * t1[i]) if reset[i] else (kk[i], kk[i])
        te = next((s["id"] for s in segs if s["video"] == vid[i] and 10 * s["start_s"] - 50 <= hi and lo <= 10 * s["end_s"] + 50), None)
        if te:
            base_ep[i] = te
    out.update(src_index=np.arange(N, dtype=np.int64), fut_src=np.full(N, -1, np.int64), sub_kind=np.where(episode, "whole", "world"),
               base_ep=base_ep, world_only=~episode, track_main=(tset == vid), track_set=tset)
    om = T["obj_mask"].astype(bool); obj = om & (T["obj_kind"] == 0); ps = T["obj_pose_src"].astype(int)
    out["t1_pos"] = T["end_xyz"].astype(np.float32); out["t1_yaw"] = T["end_yaw"].astype(np.float32)
    out["t1_ok"] = obj & episode[:, None] & T["end_ok"].astype(bool) & (T["end_src"] == ps)
    out["t1_present"] = T["end_present"].astype(bool); out["t1_yaw_ok"] = T["end_yaw_ok"].astype(bool) & out["t1_ok"]
    out["t2_disp"] = (T["fut_xyz"] - T["obj_geo"][..., None, :3]).astype(np.float32)
    out["t2_ok"] = T["fut_ok"].astype(bool) & (T["fut_src"] == ps[..., None]) & om[..., None]
    out["t5"] = T["fut_contact"] >= 0.5; out["t5_ok"] = T["fut_contact_ok"].astype(bool)[:, None, :] & om[..., None]
    out["obj_contact"] = T["obj_contact"] >= 0.5; out["obj_contact_ok"] = T["contact_ok"].astype(bool)[:, None] & om
    if "end_xyz_fp" in T:                                                     # FoundationPose registration at the segment end: evaluation only (27 % of tokens)
        out["t1_pos_fp"] = T["end_xyz_fp"].astype(np.float32); out["t1_fp_ok"] = T["end_fp_ok"].astype(bool) & out["t1_ok"]
    if "hands23_contact" in T:                                                # a second contact input (Hands23, demo defaults), kept separate
        out["obj_contact2"] = T["hands23_contact"] > 0; out["obj_contact2_ok"] = T["hands23_ok"].astype(bool)[:, None] & om
    t3 = np.full(N, -1, np.int64)
    for v in np.unique(vid):
        ix = np.flatnonzero(vid == v); ks = np.unique(kk[ix])
        for i in ix:
            d = np.abs(ks - (kk[i] + 40)); j = int(d.argmin())
            t3[i] = ks[j] if d[j] <= 3 else -1
    out["t3_k"] = t3; out["t1_k"] = np.full(N, -1, np.int64)
    # feature rows: the base folder's rows, else a new row in feats_new
    R0 = json.load(open(BASE_FE / "rows.json")); ri = {(str(v), int(k)): n for n, (v, k) in enumerate(R0["rows"])}
    new = sorted({(vid[i], int(kk[i])) for i in range(N) if (vid[i], int(kk[i])) not in ri})
    rn = {k: n for n, k in enumerate(new)}
    out["feat_row"] = np.array([ri[(vid[i], int(kk[i]))] if (vid[i], int(kk[i])) in ri else len(R0["rows"]) + rn[(vid[i], int(kk[i]))]
                                for i in range(N)], np.int64)
    # mask set of a row: base rows hold the video's own track set, a new row the commonest track set of its samples. uv (the image
    # position query) is valid where the sample's track set is the row's mask set; appearance only on the video's own set (track_main)
    from collections import Counter
    cnt = Counter((vid[i], int(kk[i]), tset[i]) for i in range(N))
    mset = {}
    for (v, k, t), c in sorted(cnt.items(), key=lambda x: -x[1]):
        mset.setdefault((v, k), t)
    out["mask_main"] = np.array([tset[i] == (mset[(vid[i], int(kk[i]))] if (vid[i], int(kk[i])) in rn else vid[i]) for i in range(N)])
    if new:
        new_feats_folder(new, Path(a.feats_new), BASE_FE, mset)
    DST.mkdir(parents=True, exist_ok=True)
    np.savez(DST / f"{a.out}.npz", **out)
    per = {v: dict(examples=int((vid == v).sum()), episodes=int((episode & (vid == v)).sum()), new_rows=sum(1 for x in new if x[0] == v))
           for v in np.unique(vid)}
    rep = dict(tracks=a.tracks, n_examples=N, in_episodes=int(episode.sum()), outside=int((~episode).sum()), other_track_set=int((tset != vid).sum()),
               mask_main=float(out["mask_main"].mean()), rows_two_track_sets=int(sum(1 for x in new if sum(1 for (v, k, _) in cnt if (v, k) == x) > 1)),
               base_rows=len(R0["rows"]), new_rows=len(new), feats_new=a.feats_new if new else None,
               t1_ok_of_episode_objects=float(out["t1_ok"].sum() / max(1, (obj & episode[:, None]).sum())),
               t1_source_mismatch=int((obj & episode[:, None] & T["end_ok"].astype(bool) & (T["end_src"] != ps)).sum()),
               t2_ok=float(out["t2_ok"][om].mean()), t2_source_mismatch=float((T["fut_ok"].astype(bool) & (T["fut_src"] != ps[..., None]))[om].mean()),
               t5_ok=float(out["t5_ok"][om].mean()), contact_input_ok=float(out["obj_contact_ok"][om].mean()), t3_ok=float((t3 >= 0).mean()),
               reset_examples=int(reset.sum()), reset_to_test=sorted({str(seg[i]) for i in np.flatnonzero(reset) if base_ep[i] != seg[i]}),
               t1_fp_ok=float(out["t1_fp_ok"].sum() / max(1, out["t1_ok"].sum())) if "t1_fp_ok" in out else None,
               test=int(sum((b in HELD_TWINS) or b.startswith(("IMG_2546", "IMG_2547", "IMG_2548")) for b in base_ep)), per_video=per,
               est_new_rows_gb=round(len(new) * 8.4 / 1024, 1))
    jdump(rep, DST / f"{a.out}_report.json")
    log(f"tracks_adapt: {N} examples ({rep['in_episodes']} in episodes), {len(new)} new feature rows -> {DST / (a.out + '.npz')}")


if __name__ == "__main__":
    main()
