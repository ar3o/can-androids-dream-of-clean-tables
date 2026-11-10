"""Frozen video and image encoders (inference only) for the V-JEPA goal / progress judge and the world-model features.

All encoders take a clip X (T, H, W, 3) uint8 RGB at the working resolution and return, for each requested layer, the
patch grid of the last tubelet (or the only frame for image encoders) as (h, w, D) plus the time-mean grid (h, w, D).

  vj2l   facebook/vjepa2-vitl-fpc64-256 (HF weights)   V-JEPA 2 ViT-L, 300M, native 256
  vj2g   facebook/vjepa2-vitg-fpc64-256 (HF weights)   V-JEPA 2 ViT-g, 1B, native 256
  vj21l  vjepa2_1_vitl_dist_vitG_384.pt (Meta)          V-JEPA 2.1 ViT-L distilled from ViT-G, native 384
  dino2l dinov2_vitl14_reg4 (torch hub, local)          DINOv2 ViT-L/14 + registers, image baseline (DINOv3 is gated on HF)

Non-square input: the HF port of V-JEPA 2 builds RoPE positions for a square grid. We patch it to do what Meta's own
code does (src/models/utils/modules.py RoPEAttention.forward): positions from the real (H, W) patch grid, then
snapped to the training grid (h *= 16/H, w *= 16/W). V-JEPA 2.1 handles non-square inputs natively (interpolate_rope).
"""
from __future__ import annotations

from cleantables import CT_DATA
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(CT_DATA)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

HF_PATHS = {
    "vj2l": ROOT / "cache/hf/hub/models--facebook--vjepa2-vitl-fpc64-256",
    "vj2g": V2 / "tmp/hf/hub/models--facebook--vjepa2-vitg-fpc64-256",
}
VJ21_PT = V2 / "vjepa_wm/weights/vjepa2_1_vitl_dist_vitG_384.pt"
VJ21_REPO = V2 / "vjepa_wm/third_party/vjepa2"
DINO_REPO = ROOT / "cache/torch/hub/facebookresearch_dinov2_main"
DINO_CKPT = ROOT / "cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"

# working resolution (W, H) per encoder family; all keep the 16:9 phone aspect (960x540 -> 1.78)
RES = {"vj2l": (448, 256), "vj2g": (448, 256), "vj21l": (672, 384), "dino2l": (448, 252), "pixel": (448, 256), "posenc": (448, 256)}
SQUASH = (256, 256)  # squashed square input, kept for comparison


def _snapshot(p: Path) -> Path:
    s = sorted((p / "snapshots").iterdir())
    return s[-1]


def _patch_hf_rope(model, hw: list):
    """Replace square-grid RoPE positions with Meta's non-square + snap-to-grid positions. hw = [H_patches, W_patches]
    is a mutable list read at call time (set before each forward)."""
    for layer in model.encoder.layer:
        att = layer.attention
        g = float(att.grid_size)

        def get_position_ids(x, masks=None, att=att, g=g):
            Hp, Wp = hw
            ids = torch.arange(x.size(1), device=x.device)
            tpf = Hp * Wp
            fr = ids // tpf
            rem = ids - tpf * fr
            hh = rem // Wp
            ww = rem - Wp * hh
            return fr.float(), hh.float() * (g / Hp), ww.float() * (g / Wp)

        orig_apply = att.apply_rotary_embeddings

        def apply_rot(qk, pos_ids, orig_apply=orig_apply):
            return orig_apply(qk.float(), pos_ids).to(qk.dtype)

        att.get_position_ids = get_position_ids
        att.apply_rotary_embeddings = apply_rot


class Encoder:
    """enc = Encoder("vj2l", layers=(8, 16, 24)); out = enc(clips) with clips uint8 (B, T, H, W, 3) numpy/torch."""

    def __init__(self, name: str, layers=None, dev="cuda", squash=False):
        self.name, self.dev, self.squash = name, dev, squash
        self.res = SQUASH if squash else RES[name]
        self.video = name.startswith("vj")
        self.dt = torch.float32 if str(dev) == "cpu" else torch.float16          # fp32 on CPU
        if name in HF_PATHS:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(_snapshot(HF_PATHS[name]), torch_dtype=self.dt, attn_implementation="sdpa")
            self.m = m.to(dev).eval()
            self.depth = len(m.encoder.layer)
            self.hw = [0, 0]
            _patch_hf_rope(m, self.hw)
            self.patch, self.tub = 16, 2
        elif name == "vj21l":
            sys.path.insert(0, str(VJ21_REPO))
            sys.path.append(str(V2 / "vjepa_wm/third_party/stubs"))
            from app.vjepa_2_1.models import vision_transformer as vit
            enc = vit.vit_large(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True,
                                use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                                img_temporal_dim_size=1, interpolate_rope=True, n_output_distillation=1)
            sd = torch.load(VJ21_PT, map_location="cpu", weights_only=False)["ema_encoder"]
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
            enc.load_state_dict(sd, strict=True)
            self.m = enc.half().to(dev).eval()
            self.depth = 24
            self.patch, self.tub = 16, 2
        elif name == "dino2l":
            m = torch.hub.load(str(DINO_REPO), "dinov2_vitl14_reg", source="local", pretrained=False)
            m.load_state_dict(torch.load(DINO_CKPT, map_location="cpu"))
            self.m = m.half().to(dev).eval()
            self.depth = 24
            self.patch, self.tub = 14, 1
        elif name in ("pixel", "posenc"):
            # controls with no learned features, same 28 x 16 grid as V-JEPA:
            #   pixel  = mean RGB (ImageNet-normalised) of each 16 x 16 cell of the last frame
            #   posenc = a fixed random Fourier code of each cell's (x, y) position, independent of the image: pooled inside
            #            an object mask it only says where the object is, so it tests how much the mask location alone explains
            self.m, self.depth, self.patch, self.tub = None, 1, 16, 1
            if name == "posenc":
                h, w = self.res[1] // 16, self.res[0] // 16
                yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
                Bf = torch.randn(2, 64, generator=torch.Generator().manual_seed(0)) * 6.0
                ang = torch.stack([xx, yy], -1) @ Bf
                self.pe = torch.cat([ang.sin(), ang.cos()], -1).to(dev)                     # (h, w, 128)
        else:
            raise ValueError(name)
        self.layers = tuple(layers) if layers else (self.depth // 3, 2 * self.depth // 3, self.depth)
        if name == "vj21l" and not layers:
            self.layers = (6, 12, 18, 24)  # 2.1 hierarchical layers 5, 11, 17, 23 (0-based)

    def grid(self):
        return self.res[1] // self.patch, self.res[0] // self.patch

    def _prep(self, clips):
        x = torch.as_tensor(np.ascontiguousarray(clips)).to(self.dev)  # (B,T,H,W,3) uint8
        B, T = x.shape[:2]
        x = x.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        W, H = self.res
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", antialias=True, align_corners=False)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        return x.to(self.dt).view(B, T, 3, H, W)

    @torch.no_grad()
    def __call__(self, clips):
        """Returns dict layer -> (last (B,h,w,D) fp16, tmean (B,h,w,D) fp16). For image encoders: last frame only."""
        x = self._prep(clips)
        B, T = x.shape[:2]
        h, w = self.grid()
        outs = {}
        if self.name in HF_PATHS:
            self.hw[0], self.hw[1] = h, w
            hs = self.m.encoder.embeddings(x)
            for i, lay in enumerate(self.m.encoder.layer, 1):
                hs = lay(hs, None)[0]
                if i in self.layers:
                    outs[i] = self.m.encoder.layernorm(hs) if i == self.depth else F.layer_norm(hs, hs.shape[-1:])
        elif self.name == "vj21l":
            self.m.out_layers = [l - 1 for l in self.layers]
            with torch.autocast("cuda", dtype=torch.float16):  # 2.1 RoPE promotes q/k to fp32
                res = self.m(x.permute(0, 2, 1, 3, 4))
            outs = {l: r for l, r in zip(self.layers, res)}
        elif self.name == "pixel":
            g = F.avg_pool2d(x[:, -1].float(), 16)                                          # (B, 3, h, w)
            outs = {1: g.permute(0, 2, 3, 1).reshape(B, h * w, 3)}
        elif self.name == "posenc":
            outs = {1: self.pe.reshape(1, h * w, -1).repeat(B, 1, 1)}
        else:  # dino: last frame of the clip
            res = self.m.get_intermediate_layers(x[:, -1], n=[l - 1 for l in self.layers], norm=True)
            outs = {l: r for l, r in zip(self.layers, res)}
        out = {}
        for l, r in outs.items():
            g = r.reshape(B, -1, h, w, r.shape[-1])
            out[l] = (g[:, -1].half(), g.float().mean(1).half())
        return out


if __name__ == "__main__":  # smoke test + memory/speed
    import time
    name = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    enc = Encoder(name)
    clip = np.random.randint(0, 255, (1, T, 540, 960, 3), np.uint8)
    torch.cuda.reset_peak_memory_stats()
    o = enc(clip)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        o = enc(clip)
    torch.cuda.synchronize()
    print(name, T, enc.res, enc.layers, {l: tuple(v[0].shape) for l, v in o.items()},
          f"{(time.time() - t0) / 3:.3f}s/clip", f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
