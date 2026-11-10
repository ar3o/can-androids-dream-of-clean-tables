"""Prompt grounding for the real episodes: an open-vocabulary detector (OWLv2, google/owlv2-base-patch16-ensemble) finds the two things a
task prompt names - "pick up <the blue phone> and put it in <the white box>" - in every frame. Nothing else (e.g. the table edge) is labelled.
Output data/processed/ground_v1/<clip>.npz: obj_xyxy, box_xyxy (T,4) in work-resolution pixels (NaN where not found), scores.
python -m cleantables.perception.ground
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EPS = ROOT / "data/episodes_v2"
OUT = ROOT / "data/processed/ground_v1"
PROMPT = {"blue_iphone": "a blue phone", "blue_card_bookmark": "a blue card", "blue_square_wipes_packet": "a blue packet",
          "yellow_small_notebook": "a yellow notebook", "yellow_single_page": "a yellow sheet of paper", "yellow_textbook_bigger": "a yellow book",
          "white_id_card": "a white card", "macbook": "a silver laptop", "metal_laptop_holder": "a metal laptop stand", "black_square_tablet": "a black tablet"}
BOX = "a white box"


def main():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from cleantables.worldmodel.progress import read_all
    names = json.load(open(ROOT / "data/object_sizes.json"))["clip_objects"]
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    m = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval().half()
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {}
    for vid in sorted(EPS.glob("*.mp4")):
        c = vid.stem
        if (OUT / f"{c}.npz").exists():
            continue
        q = [PROMPT.get(names.get(c, ""), "a flat object"), BOX]
        fr = read_all(vid)
        H, W = fr[0].shape[:2]; s = max(H, W)
        ob, bb, osc, bsc = [], [], [], []
        for i in range(0, len(fr), 8):
            imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in fr[i:i + 8]]
            inp = proc(text=[q] * len(imgs), images=imgs, return_tensors="pt").to("cuda")
            inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
            with torch.no_grad():
                out = m(**inp)
            res = proc.post_process_grounded_object_detection(out, threshold=0.08, target_sizes=torch.tensor([[s, s]] * len(imgs)).cuda())
            for r in res:
                for lab, acc_b, acc_s, pref_big in ((0, ob, osc, False), (1, bb, bsc, True)):
                    sel = r["labels"] == lab
                    if sel.any():
                        bx = r["boxes"][sel].float().cpu().numpy(); sc = r["scores"][sel].float().cpu().numpy()
                        area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) / (W * H)
                        k = int(np.argmax(sc * (np.sqrt(area) if pref_big else 1.0)))   # prefer the larger detection for the container
                        acc_b.append(np.clip(bx[k], 0, [W, H, W, H])); acc_s.append(sc[k])
                    else:
                        acc_b.append(np.full(4, np.nan)); acc_s.append(0.0)
        np.savez(OUT / f"{c}.npz", obj_xyxy=np.array(ob, np.float32), box_xyxy=np.array(bb, np.float32), obj_score=np.array(osc, np.float32),
                 box_score=np.array(bsc, np.float32), prompts=np.array(q), size=np.array([W, H]))
        summ[c] = dict(prompt=q[0], obj_found=float(np.mean(np.array(osc) > 0.15)), box_found=float(np.mean(np.array(bsc) > 0.15)))
        if c.endswith("ep00") or c.endswith("ep04"):
            g = fr[len(fr) // 2].copy(); j = len(fr) // 2
            for b, col in ((ob[j], (0, 255, 0)), (bb[j], (0, 0, 255))):
                if np.isfinite(b).all():
                    cv2.rectangle(g, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3)
            cv2.putText(g, f"{q[0]} / {q[1]}", (8, 26), 0, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(OUT / f"{c}_check.jpg"), g)
        print(c, summ[c], flush=True)
    old = json.load(open(OUT / "summary.json")) if (OUT / "summary.json").exists() else {}
    old.update(summ)
    json.dump(old, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
