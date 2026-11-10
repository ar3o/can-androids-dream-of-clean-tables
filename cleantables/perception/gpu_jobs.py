"""Per-episode GPU jobs (each well under an 8 GB GPU budget).

  depth : MoGe-3 ViT-L (same model and settings as cleantables.perception.recon.depth_models, resolution level 9, the video's fixed
          FOV from depth10.json) on every 10 fps frame of the episode window +- 2 s. Some stored depth10.npy files are computed only
          on every 3rd frame (the others are copies), so frame-to-frame flicker can only be measured on a full-rate run.
          The box videos already have full-rate MoGe-3 depth with the same settings and are skipped.
          -> OUT/depth/<ep>/moge3_full.npz (k, depth (n,270,480) f16 camera z in MoGe units, fov_x)
  wilor : WiLoR (cleantables.perception.hands.wilor_worker.run_clip, unchanged) on every frame of the window.
          -> OUT/hands/<ep>/hand_wilor.npz (same fields as data/table_clean/_work/<V>/hand_wilor.npz; `frame` = video frame index)
Resumable: an episode whose output exists is skipped.
  source scripts/env.sh
  PYTHONPATH=<MoGe_latest paths>:. $VLA_PY -m cleantables.perception.gpu_jobs depth EP...
  PYTHONPATH=third_party/WiLoR:cache/pylibs_wilor:. $VLA_PY -m cleantables.perception.gpu_jobs wilor EP...
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from cleantables.perception.episode_io import OUT, TRIAL, depth_stride, env_dirs, episode, fov_x, frame

env_dirs()


class LazyFrames:
    def __init__(self, v, ks):
        self.v, self.ks = v, list(ks)

    def __len__(self):
        return len(self.ks)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [frame(self.v, k) for k in self.ks[i]]
        return frame(self.v, self.ks[i])


def depth(eps):
    import torch
    from cleantables.perception.recon.depth_models import load
    mdl = load("moge3_vitl", "cuda")
    for ep in eps:
        e = episode(ep)
        out = OUT / "depth" / ep / "moge3_full.npz"
        if depth_stride(e["video"]) == 1:
            print(ep, "depth10 is already full-rate MoGe-3 (box pipeline), nothing to compute", flush=True)
            continue
        if out.exists():
            print(ep, "depth exists, skipped", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        ks = list(range(e["w0"], e["w1"] + 1))
        fr = LazyFrames(e["video"], ks)
        fov = fov_x(e["video"])
        t0 = time.time()
        D = np.zeros((len(ks), 270, 480), np.float16)
        for c0 in range(0, len(ks), 16):
            Dc, K, _ = mdl(fr[c0:c0 + 16], fov_x=fov)
            for j, dd in enumerate(Dc):
                D[c0 + j] = cv2.resize(dd, (480, 270), interpolation=cv2.INTER_NEAREST).astype(np.float16)
        tmp = str(out) + ".tmp.npz"
        np.savez_compressed(tmp, k=np.array(ks), depth=D, K=K, fov_x=fov, model="moge3_vitl", wall_s=time.time() - t0,
                            peak_mib=torch.cuda.max_memory_allocated() / 2**20)
        import os
        os.replace(tmp, out)
        print(ep, "depth", len(ks), f"frames {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)


def wilor(eps):
    from cleantables.perception.hands import wilor_worker as ww
    model, cfg, det = ww.load()
    for ep in eps:
        e = episode(ep)
        out = OUT / "hands" / ep / "hand_wilor.npz"
        if out.exists():
            print(ep, "wilor exists, skipped", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        ks = np.arange(e["w0"], e["w1"] + 1)
        ww.read_work_frames = lambda clip, _v=e["video"], _ks=ks: (LazyFrames(_v, _ks), {})
        t0 = time.time()
        r = ww.run_clip(model, cfg, det, ep)
        r["frame"] = ks[r["frame"]].astype(np.int32)           # window index -> video frame index
        r["window"] = np.array([e["w0"], e["w1"]], np.int32)
        tmp = str(out) + ".tmp.npz"
        np.savez_compressed(tmp, **r)
        import os
        os.replace(tmp, out)
        print(ep, "wilor", len(ks), "frames", len(r["frame"]), f"rows {time.time() - t0:.0f}s", flush=True)


def vda(eps):
    """Metric Video Depth Anything Small (arXiv 2501.12375; github DepthAnything/Video-Depth-Anything @ 4f5ae23; checkpoint
    metric_video_depth_anything_vits.pth, Apache-2.0), the published video-consistent depth model, run as published: offline
    infer_video_depth on the window's frames (RGB, 960x540 < max_res 1280, so no resize), input_size 518, fp16 (the defaults).
    Small only: Large needs ~24 GB (VRAM table in the project README), over the 8 GB budget.
    -> OUT/depth/<ep>/vda_metric_s.npz (k, depth (n,270,480) f16 metric m, INTER_NEAREST resize as depth10)
      PYTHONPATH=<VDA>/pylibs:<VDA>/repo:. $VLA_PY -m cleantables.perception.gpu_jobs vda EP..."""
    import os
    import torch
    from video_depth_anything.video_depth import VideoDepthAnything
    ck = OUT / "third_party/vda/ckpt/metric_video_depth_anything_vits.pth"
    m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=True)
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
    m = m.to("cuda").eval()
    for ep in eps:
        e = episode(ep)
        out = OUT / "depth" / ep / "vda_metric_s.npz"
        if out.exists():
            print(ep, "vda exists, skipped", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        ks = list(range(e["w0"], e["w1"] + 1))
        frames = np.stack([frame(e["video"], k)[..., ::-1] for k in ks])
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        d, _ = m.infer_video_depth(frames, 10, input_size=518, device="cuda", fp32=False)
        D = np.stack([cv2.resize(x.astype(np.float32), (480, 270), interpolation=cv2.INTER_NEAREST) for x in d]).astype(np.float16)
        tmp = str(out) + ".tmp.npz"
        np.savez_compressed(tmp, k=np.array(ks), depth=D, model="metric_video_depth_anything_vits", wall_s=time.time() - t0,
                            peak_mib=torch.cuda.max_memory_allocated() / 2**20)
        os.replace(tmp, out)
        print(ep, "vda", len(ks), f"frames {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    {"depth": depth, "wilor": wilor, "vda": vda}[sys.argv[1]](sys.argv[2:] or TRIAL)
