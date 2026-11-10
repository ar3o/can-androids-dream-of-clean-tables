"""Monocular / multi-view metric depth backends (used by the comparison in recon.bakeoff and by the pipeline). All return, for a list
of BGR work frames: depth (n,h,w) float32 metres (camera z; 0 = invalid), K (3,3) work pixels (one per clip), extra dict.

Run with the vla python and PYTHONPATH from `pythonpath(name)`; nothing is installed outside the project:
  moge2_vitl / moge2_vits : third_party/MoGe (MoGe-2, 2507.02546), fov fixed per clip (median of per-frame estimates)
  moge3_vitl              : third_party/MoGe_latest (MoGe-3, 2607.17967) + cache/pylibs_moge3 (FlexGEMM), fov fixed per clip
  da3metric               : Depth Anything 3 DA3METRIC-LARGE (2511.10647), per frame, metric via its focal-normalised output and our K
  da3mv                   : DA3-LARGE-1.1 any-view on all frames of the clip in one pass (shared geometry + poses); its depth is up to scale,
                            so one per-clip scale is taken from da3metric (median ratio on valid pixels), as DA3NESTED's metric branch does
"""
from __future__ import annotations

import glob
import time

import numpy as np

from cleantables.perception.recon.common import ROOT

HF = ROOT / "cache/hf/hub"


def pythonpath(name):
    if name.startswith("moge2"):
        return f"{ROOT}/third_party/MoGe:{ROOT}/cache/pylibs"
    if name.startswith("moge3"):
        return f"{ROOT}/third_party/MoGe_latest:{ROOT}/cache/pylibs_moge3:{ROOT}/cache/pylibs"
    return f"{ROOT}/third_party/Depth-Anything-3/src:{ROOT}/cache/pylibs_da3:{ROOT}/cache/pylibs_da3_stubs"


def _snap(repo):
    return sorted(glob.glob(str(HF / f"models--{repo.replace('/', '--')}/snapshots/*")))[0]


def _fovx_from_K(K, w):
    return float(np.degrees(2 * np.arctan(w / 2 / K[0, 0])))


class MoGe:
    def __init__(self, name, dev):
        import torch
        self.torch = torch
        if name.startswith("moge3"):
            from moge.model.v3 import MoGeModel
            ck = _snap("Ruicheng/moge-3-vitl") + "/model.pt"
        else:
            from moge.model.v2 import MoGeModel
            ck = _snap({"moge2_vitl": "Ruicheng/moge-2-vitl-normal", "moge2_vits": "Ruicheng/moge-2-vits-normal"}[name]) + "/model.pt"
        self.m = MoGeModel.from_pretrained(ck).to(dev).eval()
        self.dev, self.name = dev, name

    def _t(self, bgr):
        return self.torch.tensor(bgr[..., ::-1].copy(), dtype=self.torch.float32, device=self.dev).permute(2, 0, 1) / 255.0

    def __call__(self, frames, nfov=12, level=9, fov_x=None):
        torch = self.torch
        h, w = frames[0].shape[:2]
        fp16 = self.dev == "cuda"
        with torch.no_grad():
            sel = np.linspace(0, len(frames) - 1, min(nfov, len(frames))).astype(int) if fov_x is None else []
            fovs = [fov_x] if fov_x is not None else []
            for i in sel:
                o = self.m.infer(self._t(frames[i]), resolution_level=level, use_fp16=fp16)
                k = o["intrinsics"].float().cpu().numpy()
                fovs.append(float(np.degrees(2 * np.arctan(0.5 / k[0, 0]))))
            fov = float(np.median(fovs))
            D = []
            for f in frames:
                o = self.m.infer(self._t(f), resolution_level=level, fov_x=fov, use_fp16=fp16)
                d = o["depth"].float().cpu().numpy()
                m = o["mask"].cpu().numpy().astype(bool) & np.isfinite(d)
                D.append(np.where(m, d, 0).astype(np.float32))
                k = o["intrinsics"].float().cpu().numpy()
        K = np.array([[k[0, 0] * w, 0, k[0, 2] * w], [0, k[1, 1] * h, k[1, 2] * h], [0, 0, 1.0]])
        return np.stack(D), K, dict(fov_x=fov, fov_x_frames=fovs)


class DA3:
    def __init__(self, name, dev):
        import torch
        from depth_anything_3.api import DepthAnything3
        self.torch = torch
        self.name, self.dev = name, dev
        self.DA3 = DepthAnything3
        self.mv = name == "da3mv"      # models are loaded one at a time in __call__ (shared GPU)

    def _load(self, repo):
        return self.DA3.from_pretrained(_snap(repo)).to(self.dev).eval()

    def _free(self, m):
        del m
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()

    def _rgb(self, frames):
        return [np.ascontiguousarray(f[..., ::-1]) for f in frames]

    def _resize(self, d, h, w):
        import cv2
        return cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    def __call__(self, frames, process_res=504, focal_px=None):
        h, w = frames[0].shape[:2]
        rgb = self._rgb(frames)
        # DA3METRIC outputs depth normalised by focal: metric = focal * net / 300 (README). Focal: from the any-view model when available,
        # else the caller's (MoGe) focal. Prediction is at processed resolution; resized to the work frame.
        D, F = [], []
        mv_extra = {}
        if self.mv:
            mvm = self._load("depth-anything/DA3-LARGE-1.1")
            p = mvm.inference(rgb, process_res=process_res)
            self._free(mvm)
            dm = np.asarray(p.depth)
            Kp = np.asarray(p.intrinsics)                       # (n,3,3) at processed res
            sx = w / dm.shape[2]
            fx = float(np.median(Kp[:, 0, 0])) * sx
            mv_extra = dict(extrinsics=np.asarray(p.extrinsics), conf=np.asarray(p.conf).mean() if p.conf is not None else None)
            focal = fx
        else:
            focal = focal_px
        met = self._load("depth-anything/DA3METRIC-LARGE")
        for i, f in enumerate(rgb):
            p = met.inference([f], process_res=process_res)
            dn = np.asarray(p.depth)[0]
            sx = w / dn.shape[1]
            if focal is None:
                Kp = np.asarray(p.intrinsics)[0] if p.intrinsics is not None else None
                focal = float(Kp[0, 0]) * sx if Kp is not None else 0.9 * max(h, w)
            D.append(self._resize(dn, h, w) * (focal / sx) / 300.0)     # README: metric = focal(px, at the net's resolution) * out / 300
        self._free(met)
        D = np.stack(D)
        extra = dict(focal_px=focal)
        if self.mv:
            Dm = np.stack([self._resize(x, h, w) for x in dm])
            ok = (Dm > 0) & (D > 0)
            s = float(np.median(D[ok] / Dm[ok]))
            extra.update(mv_scale=s, **{k: v for k, v in mv_extra.items() if v is not None})
            D = Dm * s
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
        return D.astype(np.float32), K, extra


def load(name, dev):
    return MoGe(name.replace("_exif", ""), dev) if name.startswith("moge") else DA3(name, dev)


def exif_fov_x(clip, w, h):
    """Horizontal FOV (deg) of the work frame from the phone's QuickTime metadata camera.focal_length.35mm_equivalent (diagonal convention,
    43.27 mm). This is the lens focal (+ video crop); digital zoom is not included, so it is a lower bound on the true focal."""
    import re
    import struct
    from cleantables.perception.recon.common import ROOT
    rec = clip.split("_")[1]
    d = open(ROOT / f"data/raw_clips/IMG_{rec}.MOV", "rb").read()
    m = d[d.find(b"moov") - 4:]
    keys = [k.decode() for k in re.findall(rb"mdta(com\.apple\.quicktime\.camera\.[a-z0-9._]+)", m)]
    k = m.find(b"ilst", m.find(b"camera.focal_length"))
    blk = m[k - 4:k - 4 + struct.unpack(">I", m[k - 4:k])[0]]
    p, vals = 8, {}
    while p < len(blk):
        sz, idx = struct.unpack(">II", blk[p:p + 8])
        item = blk[p + 8:p + sz]
        dsz = struct.unpack(">I", item[:4])[0]
        vals[idx] = item[16:dsz]
        p += sz
    f35 = float(vals[keys.index("com.apple.quicktime.camera.focal_length.35mm_equivalent") + 1].decode())
    fpx = f35 / 43.27 * np.hypot(w, h)
    return float(np.degrees(2 * np.arctan(w / 2 / fpx))), f35, fpx
