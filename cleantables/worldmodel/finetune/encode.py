"""Encode the domain-randomised training renders (finetune.render) into the watch model's video inputs, one row per (episode,
world, keyframe):
  vj / dn   V-JEPA 2 L of the 32-slot held clip (stills at k, k - 1 .. k - 4 s) + DINOv2 L of frame k (format of
            cleantables.worldmodel.video_live and cleantables.features.paired)
  uv        each scene object's mask centroid on the 16 x 28 grid, uv_ok = > COV_MIN cells (the renderer's own masks)
  app       each object's appearance token from this render: V-JEPA / DINOv2 / location code pooled in its mask, median over its
            first APP_N keyframes up to k (causal, as on the robot; pooling of cleantables.features.video)
Objects are keyed by tracker id (scene object <- nearest tracked object at the first sample, as in cleantables.twin.dr_render).
  python -m cleantables.worldmodel.finetune.encode --dr DR/train --out DR/feats   (GPU ~2.5 GB)"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cleantables.features.video import APP_N, posenc
from cleantables.worldmodel.video_live import COV_MIN, MAXO, SLOT_STILL, VJ_HW, _gpu_prep

ROOT = Path(__file__).resolve().parents[3]
DN_HW = (18, 32)
DEV = "cuda"


def pairing(ep, S, z):
    ii = np.nonzero(z["group"] == ep)[0]; ii = ii[np.argsort(z["snap_k"][ii])]
    g0 = {int(o): z["obj_geo"][ii[0]][s][:3] for s, o in enumerate(z["obj_id"][ii[0]]) if z["obj_mask"][ii[0]][s] and z["obj_kind"][ii[0]][s] == 0}
    cand = list(g0); dd = np.array([[np.linalg.norm(np.array(o["pos"][:2]) - g0[c][:2]) for c in cand] for o in S["objects"]])
    tr = np.full(len(S["objects"]), -1)
    for _ in range(min(len(S["objects"]), len(cand))):
        r, c = np.unravel_index(np.nanargmin(dd), dd.shape); tr[r] = cand[c]; dd[r, :] = np.nan; dd[:, c] = np.nan
    return tr, int(z["snap_k"][ii[0]])


def encode_dir(d, enc, PEV, z, fps=10):
    cam = next(d.glob("camera_*.json")); ep = json.load(open(cam))["episode"]; wn = cam.stem.split("_")[-1]
    S = json.load(open(ROOT / "data/twin_table/ep/_scalefix" / ep / "scene.json")); N = len(S["objects"])
    track, k0 = pairing(ep, S, z); keys = json.load(open(d / "keys.json"))
    fd, md = d / f"frames_{ep}_{wn}", d / f"masks_{ep}_{wn}"
    img = lambda f: cv2.cvtColor(cv2.imread(str(fd / f"{f:05d}.jpg")), cv2.COLOR_BGR2RGB)                     # noqa: E731
    yy = (torch.arange(VJ_HW[0]) + 0.5) / VJ_HW[0]; xx = (torch.arange(VJ_HW[1]) + 0.5) / VJ_HW[1]
    out = {k_: [] for k_ in ("vj", "dn", "uv", "uv_ok", "app_vj", "app_dn", "app_ctl", "app_ok")}; acc = [[] for _ in range(N)]
    for k in keys:
        f = k - k0
        st = [img(max(0, f - 10 * h)) for h in range(5)]
        clip = torch.as_tensor(np.stack([st[h] for h in SLOT_STILL]))[None].to(DEV)
        vj = enc["vj"](clip)[24][0].float().reshape(*VJ_HW, -1); dn = enc["dn"](clip[:, -1:])[24][0].float().reshape(*DN_HW, -1)
        oh = F.one_hot(torch.as_tensor(cv2.imread(str(md / f"{f:05d}.png"), 0).astype(np.int64)).clamp(max=MAXO), MAXO + 1).permute(2, 0, 1)[None].float().to(DEV)
        cv = F.adaptive_avg_pool2d(oh, VJ_HW)[0, 1:N + 1]; cd = F.adaptive_avg_pool2d(oh, DN_HW)[0, 1:N + 1]
        wv, wd = cv.flatten(1).sum(1), cd.flatten(1).sum(1)
        uv = torch.stack([(cv * xx.to(DEV)[None, None]).sum((-1, -2)), (cv * yy.to(DEV)[None, :, None]).sum((-1, -2))], -1) / wv.clamp(min=1e-6)[:, None]
        pv = cv.flatten(1) @ vj.flatten(0, 1) / wv.clamp(min=1e-6)[:, None]; pd = cd.flatten(1) @ dn.flatten(0, 1) / wd.clamp(min=1e-6)[:, None]
        pc = cv.flatten(1) @ PEV.flatten(0, 1) / wv.clamp(min=1e-6)[:, None]
        for r in range(N):
            if float(wv[r]) > COV_MIN and len(acc[r]) < APP_N:
                acc[r].append((pv[r].cpu().numpy(), pd[r].cpu().numpy(), pc[r].cpu().numpy()))
        out["vj"].append(vj.half().cpu().numpy()); out["dn"].append(dn.half().cpu().numpy())
        out["uv"].append(uv.cpu().numpy()); out["uv_ok"].append((wv > COV_MIN).cpu().numpy())
        out["app_vj"].append(np.stack([np.median([a[0] for a in acc[r]], 0) if acc[r] else np.zeros(1024) for r in range(N)]).astype(np.float16))
        out["app_dn"].append(np.stack([np.median([a[1] for a in acc[r]], 0) if acc[r] else np.zeros(1024) for r in range(N)]).astype(np.float16))
        out["app_ctl"].append(np.stack([np.median([a[2] for a in acc[r]], 0) if acc[r] else np.zeros(PEV.shape[-1]) for r in range(N)]).astype(np.float32))
        out["app_ok"].append(np.array([bool(acc[r]) for r in range(N)]))
    return dict(episode=ep, world=d.name, snap_k=np.array(keys), track_id=track, **{k_: np.stack(v_) for k_, v_ in out.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dr", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default="cuda"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    from cleantables.features.encoders import Encoder
    torch.set_grad_enabled(False)
    enc = dict(vj=Encoder("vj2l", layers=(24,), dev=a.dev), dn=Encoder("dino2l", layers=(24,), dev=a.dev))
    for e_ in enc.values():
        e_._prep = _gpu_prep.__get__(e_)
    PEV = torch.as_tensor(posenc(VJ_HW)).to(a.dev); global DEV; DEV = a.dev
    z = np.load(CT_DATA + "/wm_eval/wm_v3_tracks/tracks_all.npz", allow_pickle=True)
    dirs = sorted(p.parent for p in Path(a.dr).glob("*/*/camera_*.json")); t0 = time.time(); n = 0
    dirs = dirs[:a.limit] if a.limit else dirs
    for d in dirs:
        o = O / f"{d.parent.name}__{d.name}.npz"
        if o.exists():
            continue
        r = encode_dir(d, enc, PEV, z); np.savez(o, **r); n += len(r["snap_k"])
        print(f"[dr-encode] {d.parent.name} {d.name}: {len(r['snap_k'])} rows ({n} total, {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
