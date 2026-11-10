"""Reconstruction stage 'sam3': the official SAM 3 image model (facebook/sam3 checkpoint + facebookresearch/sam3 code), text-prompted
per frame with the task's two phrases (object, container). Image mode rather than the video predictor because its GPU memory does not
grow with clip length (the video predictor grows ~15 MB/frame; clips have up to 203 recon frames).

Run in the SAM 3 env:
  PYTHONPATH=<sam3 repo>:. $SAM3_PY -m cleantables.perception.recon.sam3_worker [clips...]
Memory: weights fp32, compute under bf16 autocast (the official setting), peak 4.2 GB; the 354 M text encoder runs on the CPU (each
phrase encoded once per clip). Casting the whole vision trunk to bf16 (peak 1.85 GB) returned no detections on 5/5 test frames where
fp32 found both objects; load(bf16=True) therefore casts the trunk's outputs back to fp32.
Output _work/<clip>/sam3.npz, per prompt p (0 = object, 1 = container) and frame k, up to TOPK detections with prob >= MIN_P:
  p{p}_masks packed bool (n, TOPK, H, W), p{p}_probs (n, TOPK) (0 = empty slot), p{p}_boxes (n, TOPK, 4) xyxy work pixels.
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cleantables.perception.recon.common import WORK, alert, all_clips, prompts, read_work_frames

REPO = Path(os.environ.get("SAM3_REPO", "third_party/sam3"))
CKPT = Path(os.environ.get("SAM3_CKPT", "third_party/sam3/sam3.pt"))
TOPK = 4
MIN_P = 0.2
BACKOFF_RATE = 0.10     # object phrase detected (top p >= 0.5) in < 10 % of frames -> drop adjectives one at a time and re-query


def backoff_phrases(phrase):
    """'a metal laptop stand' -> ['a laptop stand', 'a stand']: drop the leading modifier words one at a time, keep the article and head noun.
    Generic (no vocabulary, no per-clip choice)."""
    w = phrase.split()
    art = [w[0]] if w and w[0].lower() in ("a", "an", "the") else []
    body = w[len(art):]
    out = []
    for i in range(1, len(body)):
        rest = body[i:]
        if rest[0].lower() in ("of", "for", "with", "in", "on", "to", "and", "from"):
            continue
        a = (["an"] if rest[0][0].lower() in "aeiou" else ["a"]) if art else []
        out.append(" ".join(a + rest))
    return out


def load(dev="cuda", bf16=False):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(device="cpu", checkpoint_path=str(CKPT), load_from_HF=False)
    lang = model.backbone.language_backbone
    model.backbone.language_backbone = None
    if bf16:   # low-memory path (~2 GB free): only the vision trunk's weights in bf16 (checked against fp32 on the CPU);
        # its outputs are cast back to fp32 so the detector/decoder (fp32, partly run outside autocast) see the dtype they expect
        vb = model.backbone.vision_backbone
        vb.to(torch.bfloat16)
        fwd = vb.forward

        def _f32(x):
            if torch.is_tensor(x):
                return x.float() if x.is_floating_point() else x
            if isinstance(x, (list, tuple)):
                return type(x)(_f32(v) for v in x)
            if isinstance(x, dict):
                return {k: _f32(v) for k, v in x.items()}
            if hasattr(x, "tensors"):        # NestedTensor-like
                x.tensors = _f32(x.tensors)
                return x
            return x

        def fwd_bf16(*a, **k):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _f32(fwd(*a, **k))
        vb.forward = fwd_bf16
    model.to(dev)
    model.backbone.language_backbone = lang       # stays on the CPU, fp32
    orig = model.backbone.forward_text
    cache = {}

    def forward_text_cpu(captions, input_boxes=None, additional_text=None, device="cuda", **kw):
        key = tuple(captions)
        if key not in cache:
            with torch.autocast(device_type="cuda", enabled=False), torch.autocast(device_type="cpu", enabled=False):
                out = orig(captions, input_boxes=input_boxes, additional_text=additional_text, device="cpu")
            cache[key] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in out.items()}
        return dict(cache[key])
    model.backbone.forward_text = forward_text_cpu
    torch.cuda.empty_cache()
    return Sam3Processor(model, device=dev, confidence_threshold=MIN_P)


def det_rate(r, p=0):
    return float((r[f"p{p}_probs"][:, 0] >= 0.5).mean())


def run_with_backoff(proc, clip, phrases):
    """Run [object, container]; if the object phrase is (almost) never found, re-query with shorter phrases and keep the first that reaches
    BACKOFF_RATE (else the best). Records the phrase actually used and every phrase's detection rate."""
    r = run_clip(proc, clip, phrases)
    tried = {phrases[0]: det_rate(r)}
    used = phrases[0]
    if tried[used] < BACKOFF_RATE:
        best = (tried[used], None)
        for ph in backoff_phrases(phrases[0]):
            rb = run_clip(proc, clip, [ph])
            tried[ph] = det_rate(rb)
            if tried[ph] > best[0]:
                best = (tried[ph], (ph, rb))
            if tried[ph] >= BACKOFF_RATE:
                break
        if best[1] is not None:
            used, rb = best[1]
            for k in ("p0_masks", "p0_probs", "p0_boxes"):
                r[k] = rb[k]
    r["object_prompt_task"] = np.array(phrases[0])
    r["object_prompt_used"] = np.array(used)
    r["backoff_rates"] = np.array(json.dumps(tried))
    return r


def run_clip(proc, clip, phrases, only=None):
    fr, meta = read_work_frames(clip)
    if only is not None:
        fr = [fr[i] for i in only]
    n = len(fr)
    H, W = fr[0].shape[:2]
    res = {}
    for p in range(len(phrases)):
        res[f"p{p}_masks"] = np.zeros((n, TOPK, H, W), bool)
        res[f"p{p}_probs"] = np.zeros((n, TOPK), np.float32)
        res[f"p{p}_boxes"] = np.full((n, TOPK, 4), np.nan, np.float32)
    t0 = time.time()
    for k, bgr in enumerate(fr):
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        with torch.autocast(device_type=proc.device, dtype=torch.bfloat16, enabled=proc.device == "cuda"):
            st = proc.set_image(torch.from_numpy(rgb).permute(2, 0, 1))
            for p, ph in enumerate(phrases):
                proc.reset_all_prompts(st)
                st = proc.set_text_prompt(ph, st)
                pr = st["scores"].float().cpu().numpy()
                o = np.argsort(-pr)[:TOPK]
                if len(o):
                    res[f"p{p}_probs"][k, :len(o)] = pr[o]
                    res[f"p{p}_masks"][k, :len(o)] = st["masks"][o, 0].cpu().numpy()
                    res[f"p{p}_boxes"][k, :len(o)] = st["boxes"][o].float().cpu().numpy()
        del st
    for p in range(len(phrases)):
        res[f"p{p}_masks"] = np.packbits(res[f"p{p}_masks"])
    res.update(shape=np.array([n, TOPK, H, W]), prompts=np.array(phrases), wall_s=time.time() - t0)
    return res


def run_with_backoff_existing(proc, clip, z, phrase):
    shp = z["shape"]
    tried = {phrase: det_rate(z)}
    best = (tried[phrase], None)
    for ph in backoff_phrases(phrase):
        rb = run_clip(proc, clip, [ph])
        tried[ph] = det_rate(rb)
        if tried[ph] > best[0]:
            best = (tried[ph], (ph, rb))
        if tried[ph] >= BACKOFF_RATE:
            break
    used = phrase
    if best[1] is not None:
        used, rb = best[1]
        for k in ("p0_masks", "p0_probs", "p0_boxes"):
            z[k] = rb[k]
    z["object_prompt_task"] = np.array(phrase)
    z["object_prompt_used"] = np.array(used)
    z["backoff_rates"] = np.array(json.dumps(tried))
    z["prompts"] = np.array([used, str(z["prompts"][1])])
    return z


def main():
    redo_backoff = "--backoff" in sys.argv
    clips = [a for a in sys.argv[1:] if not a.startswith("--")] or all_clips()
    from cleantables.perception.recon.bakeoff import CLIPS as FIRST      # comparison clips first (their masks score the depth models)
    clips = [c for c in FIRST if c in clips] + [c for c in clips if c not in FIRST]
    pr = prompts()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    proc = load()
    print(f"SAM 3 image model loaded in {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**20:.0f} MiB, repo {commit[:10]}", flush=True)
    for c in clips:
        f = WORK / c / "sam3.npz"
        if f.exists() and redo_backoff:
            z = dict(np.load(f))
            if "object_prompt_used" in z or det_rate(z) >= BACKOFF_RATE:
                continue
            # existing result whose object phrase was (almost) never found: run only the back-off phrases, keep the container result
            try:
                r = run_with_backoff_existing(proc, c, z, pr[c]["object"])
                np.savez_compressed(str(f) + ".tmp.npz", **r)
                Path(str(f) + ".tmp.npz").replace(f)
                print(f"{c}: back-off {r['backoff_rates']} -> used '{r['object_prompt_used']}'", flush=True)
            except Exception as e:
                alert(f"sam3 back-off failed on {c} (not retried): {e!r}")
            continue
        if f.exists():
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
            r = run_with_backoff(proc, c, [pr[c]["object"], pr[c]["container"]])
            r.update(repo_commit=commit, checkpoint=str(CKPT), peak_mib=torch.cuda.max_memory_allocated() / 2**20)
            np.savez_compressed(str(f) + ".tmp.npz", **r)
            Path(str(f) + ".tmp.npz").replace(f)
            n = r["shape"][0]
            print(f"{c}: {n} frames {r['wall_s']:.0f}s ({r['wall_s'] / n:.2f} s/frame) peak {r['peak_mib']:.0f} MiB; obj found "
                  f"{(r['p0_probs'][:, 0] > 0.5).mean():.2f}, box found {(r['p1_probs'][:, 0] > 0.5).mean():.2f}", flush=True)
        except Exception as e:
            alert(f"sam3 failed on {c} (skipped, no retry): {e!r}")
            print(f"{c}: FAILED {e!r}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
