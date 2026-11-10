"""Video features for the world model inside the simulated robot loop, computed on the GPU in-process.

The features match the robot-mode video the model was trained on (cleantables.features.paired): robot hidden, phone camera,
a 32-slot clip of stills held at now, -1, -2, -3, -4 s -> V-JEPA 2 L layer 24, last tubelet, 16 x 28; DINOv2 L of the
current frame, 18 x 32; each object's visible-mask centroid on the 16 x 28 grid as uv.
  frames    cleantables.twin.splat_render (twin meshes, table and background plate at the env's object poses; no robot in
            view), rendered only for the envs being replanned, from a 5-entry pose history per env (one pose per sim second)
  features  one batched V-JEPA 2 + DINOv2 pass (fp16, encoders loaded once, ~1.2 GB)
Use (templates in the order of cleantables.worldmodel.physics.live_plan.LivePlanner):
  LV = LiveVideo(scenes, n_envs, device="cuda")      # scenes = LivePlanner's template names; loads the splat cache (build: `cache` below)
  LV.reset(env_ids, tpl_ids, xyz, yaw)               # at episode start: the history = the start layout (training's clamped clip start)
  LV.record(env_ids, xyz, yaw, t_s)                  # every control step (cheap): keeps one pose per sim second
  v = LV.features(env_ids, tpl_ids, xyz, yaw)        # at a replan: dict for LivePlanner.plan(..., video=v)
xyz (n, I, 3) are twin-frame object roots (bottom centre) and yaw (n, I) yaw offsets, as for LivePlanner. Slot order follows the
template's scene.json objects (= wm_tokens rows).
  $VLA_PY -m cleantables.worldmodel.video_live cache [--scope clean]   -> fastpath/splat_cache_<scope>.pt (one-off, GPU ~1.5 GB)"""
from __future__ import annotations

from cleantables import CT_DATA
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.twin.splat_render import H, W, SplatRender

FP = Path(CT_DATA + "/wm_eval/wm_v2_world/fastpath")
TWIN_ROOT = Path(os.environ.get("WM_TWIN_ROOT", CT_DATA + "/wm_eval/wm_v2_final_model/dr_root_clean"))
NH = 5                                                       # stills held: now, -1, -2, -3, -4 s
SLOT_STILL = [4] + [3] * 10 + [2] * 10 + [1] * 10 + [0]     # the 32 slots (oldest first) -> history index (0 = now), as in paired.held_frames
VJ_HW, MAXO, COV_MIN = (16, 28), 24, 0.25


VLA_SITE = Path(__file__).resolve().parents[2] / "envs/vla/lib/python3.12/site-packages"


def _vla_fallback():
    """The Isaac sim env (envs/sim) has no transformers (needed for V-JEPA 2), so append the vla env's site-packages to sys.path;
    only modules missing here are taken from there. Both envs use Python 3.12 and torch 2.11+cu130, and the V-JEPA 2 / DINOv2
    outputs are identical in both (cos 1.0, max |diff| 0). Nothing is installed into envs/sim."""
    import importlib.util
    import sys
    if importlib.util.find_spec("transformers") is None and VLA_SITE.exists() and str(VLA_SITE) not in sys.path:
        sys.path.append(str(VLA_SITE))


def _gpu_prep(self, clips):
    """cleantables.features.encoders.Encoder._prep for a uint8 CUDA tensor (B, T, H, W, 3): same resize (bilinear, antialias) and ImageNet normalisation."""
    from cleantables.features.encoders import MEAN, STD
    x = clips.to(self.dev); B, T = x.shape[:2]
    x = x.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
    Wr, Hr = self.res
    if x.shape[-2:] != (Hr, Wr):
        x = F.interpolate(x, size=(Hr, Wr), mode="bilinear", antialias=True, align_corners=False)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    return x.to(self.dt).view(B, T, 3, Hr, Wr)


class LiveVideo:
    def __init__(self, scenes, n_envs, device="cuda", cache=None, batch=8, imax=MAXO):
        _vla_fallback()
        from cleantables.features.encoders import Encoder
        self.dev = torch.device(device); self.scenes = list(scenes); self.batch = int(batch)
        self.R = SplatRender(dev=device)
        cache = Path(cache) if cache else FP / "splat_cache_clean.pt"
        if cache.exists():
            self.R.load(cache)
        for s in self.scenes:                                # templates missing from the cache are added now (~3 s each)
            if s not in self.R.T:
                self.R.add(s, json.load(open(TWIN_ROOT / s / "scene.json")))
        self.enc = dict(vj=Encoder("vj2l", layers=(24,), dev=device), dn=Encoder("dino2l", layers=(24,), dev=device))
        for e_ in self.enc.values():                         # frames stay on the GPU (the encoders' own _prep goes through numpy)
            e_._prep = _gpu_prep.__get__(e_)
        self.hist = torch.full((n_envs, NH, imax, 4), float("nan"), device=self.dev)
        self.sec = torch.full((n_envs,), -1, dtype=torch.long, device=self.dev)
        self.timing = []

    @staticmethod
    def _pose(xyz, yaw, dev):
        xyz = torch.as_tensor(xyz, dtype=torch.float32, device=dev); yaw = torch.as_tensor(yaw, dtype=torch.float32, device=dev)
        return torch.cat([xyz, yaw[..., None]], -1)

    @torch.no_grad()
    def reset(self, env_ids, tpl_ids, xyz, yaw):
        e = torch.as_tensor(env_ids, device=self.dev).long(); p = self._pose(xyz, yaw, self.dev); I = p.shape[1]
        self.hist[e] = float("nan"); self.hist[e, :, :I] = p[:, None]; self.sec[e] = 0

    @torch.no_grad()
    def record(self, env_ids, xyz, yaw, t_s):
        """Keep one pose per sim second: when an env crosses a whole second, the history shifts by one (index 0 = the latest second)."""
        e = torch.as_tensor(env_ids, device=self.dev).long(); p = self._pose(xyz, yaw, self.dev); I = p.shape[1]
        s = torch.as_tensor(t_s, device=self.dev).float().expand(len(e)).floor().long()
        new = s > self.sec[e]
        if bool(new.any()):
            en = e[new]
            self.hist[en, 1:] = self.hist[en, :-1].clone(); self.hist[en, 0, :I] = p[new]; self.sec[en] = s[new]

    @torch.no_grad()
    def features(self, env_ids, tpl_ids, xyz=None, yaw=None):
        """Return dict(vj_grid (n, 16, 28, 1024) fp16, dn_grid (n, 18, 32, 1024) fp16, uv (n, 24, 2), uv_ok (n, 24), vid_ok (n,)) on
        the GPU, in slot order; xyz / yaw (optional) give the current pose (else the latest recorded second is used)."""
        t0 = time.time()
        e = torch.as_tensor(env_ids, device=self.dev).long(); n = len(e)
        tp = np.asarray(tpl_ids.cpu() if torch.is_tensor(tpl_ids) else tpl_ids).astype(int)
        Hs = self.hist[e].clone()                                                  # (n, 5, I, 4)
        if xyz is not None:
            p = self._pose(xyz, yaw, self.dev); Hs[:, 0] = float("nan"); Hs[:, 0, :p.shape[1]] = p
        for h in range(1, NH):                                                     # missing seconds (episode start) use the next newer still
            miss = ~torch.isfinite(Hs[:, h]).any(-1).any(-1)
            Hs[miss, h] = Hs[miss, h - 1]
        names = [self.scenes[t] for t in tp for _ in range(NH)]
        img, ids = self.R.render_batch(names, Hs.reshape(n * NH, *Hs.shape[2:]), ids=True)
        img = img.view(n, NH, H, W, 3); ids = ids.view(n, NH, H, W)
        t1 = time.time()
        vj, dn = [], []
        for b0 in range(0, n, self.batch):
            sl = slice(b0, b0 + self.batch)
            clip = img[sl][:, SLOT_STILL]                                          # (b, 32, H, W, 3)
            vj.append(self.enc["vj"](clip)[24][0]); dn.append(self.enc["dn"](img[sl, 0:1])[24][0])
        cov = F.adaptive_avg_pool2d(F.one_hot(ids[:, 0].long(), MAXO + 1).permute(0, 3, 1, 2).float(), VJ_HW)[:, 1:]   # (n, 24, 16, 28) cell fractions
        yy = (torch.arange(VJ_HW[0], device=self.dev) + 0.5) / VJ_HW[0]; xx = (torch.arange(VJ_HW[1], device=self.dev) + 0.5) / VJ_HW[1]
        w = cov.sum((-1, -2))
        uv = torch.stack([(cov * xx[None, None, None]).sum((-1, -2)), (cov * yy[None, None, :, None]).sum((-1, -2))], -1) / w.clamp(min=1e-6)[..., None]
        out = dict(vj_grid=torch.cat(vj).half(), dn_grid=torch.cat(dn).half(), uv=uv, uv_ok=w > COV_MIN, vid_ok=torch.ones(n, dtype=torch.bool, device=self.dev))
        torch.cuda.synchronize()
        self.timing.append(dict(n=n, render_s=round(t1 - t0, 4), encode_s=round(time.time() - t1, 4)))
        return out


def build_cache(scope="clean"):
    sc = json.load(open(FP / f"scope_{scope}.json"))
    names = list(sc["train"]) + list(sc["heldout"])
    R = SplatRender(); t0 = time.time()
    for i, nm in enumerate(names):
        R.add(nm, json.load(open(TWIN_ROOT / nm / "scene.json")))
        if i % 25 == 0:
            print(f"{i + 1}/{len(names)} {time.time() - t0:.0f} s, GPU {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
    p = FP / f"splat_cache_{scope}.pt"; R.save(p)
    print(f"-> {p} ({p.stat().st_size / 2**20:.0f} MB), {len(R.T)} templates, {len(R.OC)} meshes, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cache"]); ap.add_argument("--scope", default="clean")
    a = ap.parse_args()
    if a.cmd == "cache":
        torch.cuda.set_per_process_memory_fraction(3.0 / 23.5)
        build_cache(a.scope)
