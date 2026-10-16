"""Person pixels that stay still (feet on the floor) survive the temporal-stability plate: find them on the raw plate
with the official SAM 3 image model (text prompts, fp32 weights under bf16 autocast) and write the mask for plate.py
--apply-person.

Runs in the SAM 3 environment (facebookresearch/sam3 code on PYTHONPATH):
  PYTHONPATH=<sam3 repo>:. HF_HUB_OFFLINE=1 <sam3 python> -m cleantables.twin.person_mask IMG_2533 [...]
Out data/twins/plates/<video>/plate_person.png (255 = person) and plate_person.json (prompts, scores, checkpoint,
commit).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from cleantables.paths import CHECKPOINTS
from cleantables.twin.config import PersonMaskConfig, load_config
from cleantables.twin.inputs import plate_dir

DEFAULT_CHECKPOINT = CHECKPOINTS / "sam3" / "sam3.pt"


def load_sam3(checkpoint: Path, target_device: str = "cuda"):
    """SAM 3 image processor; the text encoder stays on the CPU in fp32 and each phrase is encoded once."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device="cpu", checkpoint_path=str(checkpoint), load_from_HF=False)
    lang = model.backbone.language_backbone
    model.backbone.language_backbone = None
    model.to(target_device)
    model.backbone.language_backbone = lang
    encode_text = model.backbone.forward_text
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
    return Sam3Processor(model, device=target_device, confidence_threshold=0.2)


def person_mask(proc, bgr: np.ndarray, cfg: PersonMaskConfig) -> tuple[np.ndarray, list[dict]]:
    """union of every detection with prob >= min_prob over the prompts, dilated."""
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    M = np.zeros(bgr.shape[:2], bool)
    found = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = proc.set_image(torch.from_numpy(rgb).permute(2, 0, 1))
        for phrase in cfg.prompts:
            proc.reset_all_prompts(state)
            state = proc.set_text_prompt(phrase, state)
            probs = state["scores"].float().cpu().numpy()
            for j in np.nonzero(probs >= cfg.min_prob)[0]:
                m = state["masks"][j, 0].cpu().numpy().astype(bool)
                if m.shape != M.shape:
                    m = cv2.resize(
                        m.astype(np.uint8), (M.shape[1], M.shape[0]), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                M |= m
                found.append(dict(prompt=phrase, prob=float(probs[j]), px=int(m.sum())))
    k = 2 * cfg.dilation_px + 1
    return cv2.dilate(M.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool), found


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM 3 person / foot mask on each video's raw static plate.")
    ap.add_argument("videos", nargs="+", help="video names, e.g. IMG_2533")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="official facebook/sam3 checkpoint (sam3.pt)")
    ap.add_argument("--config", default=None, help="config file (default configs/twin.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config).person_mask
    import sam3

    repo = Path(sam3.__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    proc = load_sam3(Path(args.checkpoint))
    proc.confidence_threshold = cfg.min_prob
    for v in args.videos:
        d = plate_dir(v)
        src = d / "plate_raw.png"
        M, found = person_mask(proc, cv2.imread(str(src)), cfg)
        cv2.imwrite(str(d / "plate_person.png"), M.astype(np.uint8) * 255)
        info = dict(
            video=v,
            source=src.name,
            prompts=cfg.prompts,
            p_min=cfg.min_prob,
            dilate_px=cfg.dilation_px,
            detections=found,
            frac=float(M.mean()),
            model="facebook/sam3 image model (fp32 weights, bf16 autocast)",
            checkpoint=str(args.checkpoint),
            repo_commit=commit,
        )
        (d / "plate_person.json").write_text(json.dumps(info, indent=1))
        print(f"{v}: person px {M.mean() * 100:.2f} %", [(x["prompt"], round(x["prob"], 2)) for x in found], flush=True)


if __name__ == "__main__":
    main()
