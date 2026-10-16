"""SAM 3 image model (official facebook/sam3 checkpoint), text prompts on chosen frames; runs in the SAM 3 environment.

    PYTHONPATH=<sam3 repo>:<repo> HF_HUB_OFFLINE=1 <sam3 python> -m cleantables.perception.sam3_image \\
        --jobs jobs.json --checkpoint sam3.pt [--min-prob 0.3]

jobs.json = {"<video>": {"frames": [k, ...], "phrases": [...], "out": "<file>.npz"}, ...}; one model load in all.
Output per video: frame, prompt, prob, box (xyxy at 960x540), masks (packed bool, m x 270 x 480), shape, prompts,
frames_done. Weights stay fp32 under bf16 autocast (a bf16 trunk lost detections); the text encoder runs on the CPU and
each phrase is encoded once.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .files import H_LO, W_LO, read_frame

TOP_K = 12


def load_model(checkpoint: str, min_prob: float, device: str = "cuda"):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device="cpu", checkpoint_path=checkpoint, load_from_HF=False)
    language = model.backbone.language_backbone
    model.backbone.language_backbone = None
    model.to(device)
    model.backbone.language_backbone = language
    encode_text = model.backbone.forward_text
    target_device = device
    cache = {}

    def forward_text_cpu(captions, input_boxes=None, additional_text=None, device="cuda", **kw):
        key = tuple(captions)
        if key not in cache:
            with torch.autocast(device_type="cuda", enabled=False), torch.autocast(device_type="cpu", enabled=False):
                out = encode_text(captions, input_boxes=input_boxes, additional_text=additional_text, device="cpu")
            cache[key] = {k: (v.to(target_device) if torch.is_tensor(v) else v) for k, v in out.items()}
        return dict(cache[key])

    model.backbone.forward_text = forward_text_cpu
    torch.cuda.empty_cache()
    return Sam3Processor(model, device=device, confidence_threshold=min_prob)


def detect(processor, video: str, frames: list[int], phrases: list[str]) -> dict:
    rec = {k: [] for k in ("frame", "prompt", "prob", "box", "mask")}
    for k in frames:
        rgb = np.ascontiguousarray(read_frame(video, k)[..., ::-1])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(torch.from_numpy(rgb).permute(2, 0, 1))
            for p, phrase in enumerate(phrases):
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(phrase, state)
                probs = state["scores"].float().cpu().numpy()
                if len(probs) == 0:
                    continue
                masks = torch.nn.functional.interpolate(state["masks"].float(), (H_LO, W_LO), mode="area")[:, 0] > 0.5
                masks = masks.cpu().numpy()
                boxes = state["boxes"].float().cpu().numpy()
                for j in np.argsort(-probs)[:TOP_K]:
                    rec["frame"].append(k)
                    rec["prompt"].append(p)
                    rec["prob"].append(probs[j])
                    rec["box"].append(boxes[j])
                    rec["mask"].append(masks[j])
        del state
    return rec


def save(rec: dict, phrases: list[str], frames: list[int], out: str) -> None:
    m = len(rec["frame"])
    masks = np.asarray(rec["mask"], bool).reshape(m, H_LO, W_LO)
    tmp = out + ".tmp.npz"
    np.savez_compressed(
        tmp,
        frame=np.asarray(rec["frame"], np.int32),
        prompt=np.asarray(rec["prompt"], np.int16),
        prob=np.asarray(rec["prob"], np.float32),
        box=np.asarray(rec["box"], np.float32).reshape(m, 4),
        masks=np.packbits(masks) if m else np.zeros(0, np.uint8),
        shape=np.array([m, H_LO, W_LO]),
        prompts=np.array(phrases),
        frames_done=np.array(sorted(frames), np.int32),
    )
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--min-prob", type=float, default=0.3)
    args = ap.parse_args()
    with open(args.jobs) as f:
        jobs = json.load(f)
    todo = {v: j for v, j in jobs.items() if not os.path.exists(j["out"])}
    if not todo:
        return
    t0 = time.time()
    processor = load_model(args.checkpoint, args.min_prob)
    print(f"sam3 image model loaded in {time.time() - t0:.0f} s", flush=True)
    for video, job in todo.items():
        t1 = time.time()
        rec = detect(processor, video, job["frames"], job["phrases"])
        save(rec, job["phrases"], job["frames"], job["out"])
        print(
            f"{video}: {len(job['frames'])} frames x {len(job['phrases'])} phrases -> {len(rec['frame'])} detections, "
            f"{time.time() - t1:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
