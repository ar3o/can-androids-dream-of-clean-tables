"""The world model's video inputs from a rendered clip (cleantables.twin.dr_render --wm_input --frames_only: the phone camera,
no robot, each object's mask), for finetune.predict --video, so the model sees the randomised world's pixels. Built as the robot's
live video path (video_live.LiveVideo.features) builds them, at each held-out keyframe k of the episode:
  vj_grid  V-JEPA 2 L (layer 24, last tubelet) of the 32-slot held clip of stills at k, k-1, k-2, k-3, k-4 s (paired.held_frames;
           before the clip's first frame the first frame is held, as at a robot episode start)
  dn_grid  DINOv2 L (layer 24) of the frame at k
  uv       each object's mask centroid on the 16 x 28 grid (cell coverage weighted), uv_ok = coverage > 0.25 cells (as video_live)
Objects are keyed by tracker id: each scene object <- the tracked object nearest its start position in the first sample (as in
cleantables.twin.dr_render). Frames are at --fps from the episode's first tracked sample (the renderer's clock).
  python -m cleantables.worldmodel.finetune.input_features --dir OUT --world seed1 --out feats.npz   (GPU ~1.5 GB)"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.video_live import COV_MIN, MAXO, SLOT_STILL, VJ_HW, _gpu_prep

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--world", default="seed1"); ap.add_argument("--episode", default="IMG_2516_t11")
    ap.add_argument("--tracks", default=CT_DATA + "/wm_eval/wm_v3_tracks/tracks_all.npz")
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--out", required=True)
    ap.add_argument("--mask_dir", default="", help="label masks to use instead of the render's (e.g. wm_sim_perceive's SAM 3 masks, any size)")
    a = ap.parse_args()
    from cleantables.features.encoders import Encoder
    S = json.load(open(ROOT / "data/twin_table/ep/_scalefix" / a.episode / "scene.json"))
    z = np.load(a.tracks, allow_pickle=True)
    i = np.nonzero(z["group"] == a.episode)[0]; i = i[np.argsort(z["snap_k"][i])]
    kk = z["snap_k"][i].astype(int); k0 = kk[0]
    geo, msk, oid, kind = z["obj_geo"][i[0]], z["obj_mask"][i[0]], z["obj_id"][i[0]], z["obj_kind"][i[0]]
    cand = [(k, int(oid[k])) for k in range(len(oid)) if msk[k] and kind[k] == 0]
    d = np.array([[np.linalg.norm(np.array(o["pos"][:2]) - geo[k][:2]) for k, _ in cand] for o in S["objects"]])
    track = np.full(len(S["objects"]), -1)
    for _ in range(min(len(S["objects"]), len(cand))):
        r, c = np.unravel_index(np.nanargmin(d), d.shape); track[r] = cand[c][1]; d[r, :] = np.nan; d[:, c] = np.nan
    print("[wm-in] scene object -> track id:", {S["objects"][r]["id"]: int(track[r]) for r in range(len(track))}, flush=True)
    D = Path(a.dir); fd, md = D / f"frames_{a.episode}_{a.world}", Path(a.mask_dir) if a.mask_dir else D / f"masks_{a.episode}_{a.world}"
    nf = len(list(fd.glob("*.jpg")))
    fr = lambda k: min(nf - 1, max(0, int(round((k - k0) / 10 * a.fps))))                                    # noqa: E731
    enc = dict(vj=Encoder("vj2l", layers=(24,), dev="cuda"), dn=Encoder("dino2l", layers=(24,), dev="cuda"))
    for e_ in enc.values():
        e_._prep = _gpu_prep.__get__(e_)
    img = lambda f: cv2.cvtColor(cv2.imread(str(fd / f"{f:05d}.jpg")), cv2.COLOR_BGR2RGB)                   # noqa: E731
    from cleantables.features.video import APP_N, posenc
    PEV = torch.as_tensor(posenc(VJ_HW)); acc = [[] for _ in range(len(track))]; AV, AD, AC, AO = [], [], [], []
    VJ, DN, UV, UK = [], [], [], []
    yy = (torch.arange(VJ_HW[0]) + 0.5) / VJ_HW[0]; xx = (torch.arange(VJ_HW[1]) + 0.5) / VJ_HW[1]
    with torch.no_grad():
        for k in kk:
            st = [img(fr(max(k0, k - 10 * h))) for h in range(5)]                                            # now, -1 .. -4 s (first frame held at the start)
            clip = torch.as_tensor(np.stack([st[h] for h in SLOT_STILL]))[None].cuda()
            VJ.append(enc["vj"](clip)[24][0].half().cpu()); DN.append(enc["dn"](clip[:, -1:])[24][0].half().cpu())
            m = torch.as_tensor(cv2.imread(str(md / f"{fr(k):05d}.png"), 0).astype(np.int64))
            cov = F.adaptive_avg_pool2d(F.one_hot(m.clamp(max=MAXO), MAXO + 1).permute(2, 0, 1)[None].float(), VJ_HW)[0, 1:]   # (24, 16, 28)
            w = cov.sum((-1, -2))
            uv = torch.stack([(cov * xx[None, None]).sum((-1, -2)), (cov * yy[None, :, None]).sum((-1, -2))], -1) / w.clamp(min=1e-6)[..., None]
            UV.append(uv[:len(track)].numpy()); UK.append((w > COV_MIN)[:len(track)].numpy())
            # each object's appearance token from these frames, causal: median over its first APP_N keyframes up to k (features.video pooling)
            oh = F.one_hot(m.clamp(max=MAXO), MAXO + 1).permute(2, 0, 1)[None].float()
            cd = F.adaptive_avg_pool2d(oh, (18, 32))[0, 1:]; wd = cd.flatten(1).sum(1)
            vjg = VJ[-1].float().reshape(*VJ_HW, -1); dng = DN[-1].float().reshape(18, 32, -1)
            for r in range(len(track)):
                if float(w[r]) > COV_MIN and len(acc[r]) < APP_N:
                    acc[r].append(((cov[r].flatten() @ vjg.flatten(0, 1)) / w[r], (cd[r].flatten() @ dng.flatten(0, 1)) / wd[r].clamp(min=1e-6),
                                   (cov[r].flatten() @ PEV.flatten(0, 1)) / w[r]))
            med = lambda r, q, dim: torch.stack([x[q] for x in acc[r]]).median(0).values.numpy() if acc[r] else np.zeros(dim, np.float32)   # noqa: E731
            AV.append(np.stack([med(r, 0, 1024) for r in range(len(track))])); AD.append(np.stack([med(r, 1, 1024) for r in range(len(track))]))
            AC.append(np.stack([med(r, 2, PEV.shape[-1]) for r in range(len(track))])); AO.append(np.array([bool(acc[r]) for r in range(len(track))]))
            print(f"[wm-in] k {k}: frame {fr(k)}, objects seen {int(UK[-1].sum())}/{len(track)}", flush=True)
    np.savez(a.out, snap_k=kk, vj_grid=torch.cat(VJ).numpy(), dn_grid=torch.cat(DN).numpy(), uv=np.stack(UV), uv_ok=np.stack(UK),
             track_id=track, world=a.world, frames=str(fd), app_vj=np.stack(AV).astype(np.float16), app_dn=np.stack(AD).astype(np.float16),
             app_ctl=np.stack(AC).astype(np.float32), app_ok=np.stack(AO))
    print(f"[wm-in] {len(kk)} keyframes -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
