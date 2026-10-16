"""SAM 3 video tracking (official facebook/sam3 checkpoint) of every phrase of a video; runs in the SAM 3 environment.

    PYTHONPATH=<sam3 repo>:<repo> HF_HUB_OFFLINE=1 <sam3 python> -m cleantables.perception.sam3_video \\
        --jobs jobs.json --checkpoint sam3.pt --repo <sam3 repo> [--stride 3]

jobs.json = {"<video>": {"phrases": [...], "windows": [[k0, k1], ...] or null, "out": "<file>.npz"}, ...}.
SAM 3's video predictor holds one text prompt per session, so each phrase is its own session; the vision backbone output
depends on the image only and is cached across the phrases of a video. With windows, each window is a separate session
(object ids offset by 1000 per window) because the session memory grows with its length.
Output per video: phrases, frames (work-frame index of each tracked frame), stride, and per phrase i: p{i}_ids,
p{i}_probs (n_obj, n_t), p{i}_masks (packed bool n_obj x n_t x 270 x 480), p{i}_shape; plus provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cleantables import paths

from .files import H_LO, W_LO, gpu_free_mib, n_frames

WINDOW_ID_OFFSET = 1000


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


class FeatureCache:
    """Caches backbone.forward_image per input image (keyed by a coarse fingerprint of the tensor) in CPU memory."""

    def __init__(self, enabled: bool = True):
        self.store, self.hits, self.misses, self.enabled = {}, 0, 0, enabled

    def clear(self) -> None:
        self.store.clear()

    def wrap(self, backbone) -> None:
        forward_image = backbone.forward_image

        def cached_forward_image(samples, *a, **k):
            if not self.enabled or samples.shape[0] != 1:
                return forward_image(samples, *a, **k)
            s = samples[0, :, ::61, ::59].double()
            key = (round(float(s.sum()), 4), round(float((s * s).sum()), 3), tuple(samples.shape))
            if key in self.store:
                self.hits += 1
                return to_device(self.store[key], samples.device)
            out = forward_image(samples, *a, **k)
            self.misses += 1
            self.store[key] = to_device(out, "cpu")
            return out

        backbone.forward_image = cached_forward_image


def load_predictor(checkpoint: str, text_on_gpu: bool):
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    predictor = Sam3VideoPredictor(checkpoint_path=checkpoint)
    backbone = predictor.model.detector.backbone
    if not text_on_gpu:
        # the text encoder runs once per phrase; on the CPU it saves ~1.4 GB of GPU memory
        backbone.language_backbone.to("cpu")
        encode_text = backbone._forward_text_no_ack_ckpt

        def forward_text_cpu(captions, input_boxes=None, additional_text=None, device="cuda", **kw):
            with torch.autocast(device_type="cuda", enabled=False):
                out = encode_text(
                    captions=captions, input_boxes=input_boxes, additional_text=additional_text, device="cpu"
                )
            return {k: (x.to(device) if torch.is_tensor(x) else x) for k, x in out.items()}

        backbone.forward_text = forward_text_cpu
    torch.cuda.empty_cache()
    return predictor


def share_decoded_frames() -> None:
    """Decode, resize and normalise a video's frames once; every phrase session of that video reuses the tensor."""
    import sam3.model.sam3_video_inference as svi

    load_frames = svi.load_resource_as_video_frames
    cache = {}

    def cached(resource_path, *a, **k):
        key = (str(resource_path), tuple(sorted((kk, str(vv)) for kk, vv in k.items())))
        if key not in cache:
            cache.clear()
            cache[key] = load_frames(resource_path, *a, **k)
        return cache[key]

    svi.load_resource_as_video_frames = cached


def frame_folder(video: str, stride: int) -> tuple[Path, list[int]]:
    """Folder of symlinks <i>.jpg -> work frame i * stride (SAM 3 orders frame files by their integer name)."""
    ks = list(range(0, n_frames(video), stride))
    d = paths.perception_dir(video) / f"sam3_frames_s{stride}"
    d.mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(ks):
        link = d / f"{i}.jpg"
        if not link.exists():
            os.symlink(paths.perception_dir(video) / "frames" / f"{k}.jpg", link)
    return d, ks


def track_phrase(predictor, folder: Path, phrase: str, state_on_gpu: bool) -> dict:
    """frame index -> (object ids, probabilities, masks at the work-frame resolution)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    session = predictor.handle_request(
        dict(
            type="start_session",
            resource_path=str(folder),
            offload_video_to_cpu=True,
            offload_state_to_cpu=not state_on_gpu,
        )
    )["session_id"]
    per_frame = {}
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=0, text=phrase))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session)):
                o = r["outputs"]
                ids = np.asarray(o["out_obj_ids"]).astype(int)
                probs = np.asarray(o.get("out_probs", np.ones(len(ids)))).astype(np.float32)
                per_frame[r["frame_index"]] = (ids, probs, np.asarray(o["out_binary_masks"]).astype(bool))
    predictor.handle_request(dict(type="close_session", session_id=session))
    return per_frame


def track_in_windows(predictor, folder: Path, ks: list[int], windows, phrase: str, state_on_gpu: bool) -> dict:
    per_frame = {}
    for j, (k0, k1) in enumerate(windows):
        sub = [i for i, k in enumerate(ks) if k0 <= k <= k1]
        window_folder = folder.parent / f"{folder.name}_w{j}"
        window_folder.mkdir(exist_ok=True)
        for r, i in enumerate(sub):
            link = window_folder / f"{r}.jpg"
            if not link.exists():
                os.symlink(os.readlink(folder / f"{i}.jpg"), link)
        for f, (ids, probs, masks) in track_phrase(predictor, window_folder, phrase, state_on_gpu).items():
            per_frame[sub[f]] = (ids + WINDOW_ID_OFFSET * (j + 1), probs, masks)
        torch.cuda.empty_cache()
    return per_frame


def pack(per_frame: dict, n_t: int):
    ids = sorted({int(i) for o in per_frame.values() for i in o[0]})
    probs = np.zeros((len(ids), n_t), np.float32)
    masks = np.zeros((len(ids), n_t, H_LO, W_LO), bool)
    row = {i: r for r, i in enumerate(ids)}
    for f, (oid, pr, m) in per_frame.items():
        for j, i in enumerate(oid):
            probs[row[int(i)], f] = pr[j]
            masks[row[int(i)], f] = cv2.resize(
                m[j].astype(np.uint8), (W_LO, H_LO), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
    return ids, probs, masks


def run_video(predictor, cache: FeatureCache, video: str, job: dict, stride: int, state_on_gpu: bool, provenance: dict):
    folder, ks = frame_folder(video, stride)
    n_t = len(ks)
    t0 = time.time()
    cache.clear()
    result = dict(phrases=np.array(job["phrases"]), frames=np.array(ks, np.int32), stride=stride)
    seconds = []
    for pi, phrase in enumerate(job["phrases"]):
        t1 = time.time()
        if job.get("windows"):
            per_frame = track_in_windows(predictor, folder, ks, job["windows"], phrase, state_on_gpu)
        else:
            per_frame = track_phrase(predictor, folder, phrase, state_on_gpu)
        ids, probs, masks = pack(per_frame, n_t)
        result[f"p{pi}_ids"] = np.array(ids, np.int32)
        result[f"p{pi}_probs"] = probs
        result[f"p{pi}_masks"] = np.packbits(masks)
        result[f"p{pi}_shape"] = np.array([len(ids), n_t, H_LO, W_LO])
        seconds.append(time.time() - t1)
        print(
            f"{video} '{phrase}': {len(ids)} masklets over {n_t} frames, {seconds[-1]:.0f} s, "
            f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB",
            flush=True,
        )
        torch.cuda.empty_cache()
    cache.clear()
    result.update(
        provenance,
        peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        wall_s=time.time() - t0,
        phrase_s=np.array(seconds, np.float32),
        feature_cache=cache.enabled,
    )
    tmp = job["out"] + ".tmp.npz"
    np.savez_compressed(tmp, **result)
    os.replace(tmp, job["out"])
    for d in folder.parent.glob(f"{folder.name}*"):
        shutil.rmtree(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True, help="SAM 3 source checkout (its commit is recorded)")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--text-on-gpu-min-free-mib", type=int, default=8000)
    ap.add_argument("--state-on-gpu-min-free-mib", type=int, default=8000)
    ap.add_argument("--no-feature-cache", action="store_true")
    args = ap.parse_args()
    with open(args.jobs) as f:
        jobs = json.load(f)
    todo = {v: j for v, j in jobs.items() if not os.path.exists(j["out"])}
    if not todo:
        return
    t0 = time.time()
    text_on_gpu = gpu_free_mib() >= args.text_on_gpu_min_free_mib
    predictor = load_predictor(args.checkpoint, text_on_gpu)
    state_on_gpu = gpu_free_mib() >= args.state_on_gpu_min_free_mib
    share_decoded_frames()
    cache = FeatureCache(enabled=not args.no_feature_cache)
    cache.wrap(predictor.model.detector.backbone)
    torch.cuda.reset_peak_memory_stats()
    print(
        f"sam3 video model loaded in {time.time() - t0:.0f} s; text encoder on {'gpu' if text_on_gpu else 'cpu'}, "
        f"tracking state on {'gpu' if state_on_gpu else 'cpu'}",
        flush=True,
    )
    commit = subprocess.run(
        ["git", "-C", args.repo, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    provenance = dict(repo_commit=commit, checkpoint=args.checkpoint, tool="sam3_video")
    for video, job in todo.items():
        run_video(predictor, cache, video, job, args.stride, state_on_gpu, provenance)


if __name__ == "__main__":
    main()
