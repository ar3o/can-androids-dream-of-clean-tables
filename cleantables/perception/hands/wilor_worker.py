"""WiLoR (arXiv 2409.12259, github.com/rolpotamias/WiLoR @ fcb91131) on the cached recon work frames: hand detector (WiLoR's YOLO
detector.pt, conf >= 0.3, handedness = its class) + MANO regression per detected hand.

  source scripts/env.sh
  PYTHONPATH=third_party/WiLoR:cache/pylibs_wilor:. $VLA_PY -m cleantables.perception.hands.wilor_worker [CLIP ...]   (default: all 65)

Reads data/processed/recon_v3/_work/<clip>/frames/*.jpg (read-only). Writes data/processed/recon_v3c/_work/<clip>/hand_wilor.npz with all
detections (flat arrays, `frame` = recon frame index):
  Every detected box appears twice: regressed as a right hand and as a left hand (is_right), det_is_right = the detector's class,
  det_id = the box id (shared by its two rows).
  frame (m,), det_conf (m,), is_right (m,) 1 = regressed as a right hand (left: WiLoR runs on the flipped crop and mirrors back),
  box (m,4) xyxy work px, global_orient (m,3,3) and hand_pose (m,15,3,3) MANO rotations as predicted (right-hand model; for a left hand
  these describe the mirrored image), betas (m,10), joints_mano (m,21,3) OpenPose/MediaPipe joint order, MANO metric scale, root-relative
  to nothing (= MANO output, camera-axis aligned; x already mirrored back for left hands, as in WiLoR's demo), cam_t_wilor (m,3) WiLoR's
  full-image translation for its own focal (5000/256 * max(w,h), principal point at the image centre), kp2d (m,21,2) the WiLoR joints projected
  with that camera = the model's 2D joint estimate in work pixels.
MANO files: cache/mano/mano_v1_2/models (non-commercial licence; never copied). GPU: fp16 weights (WiLoR's --fast path without layer
skipping / compile), batch 16 crops; peak ~2 GB.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from cleantables.perception.recon.common import ROOT, WORK, read_work_frames, all_clips

OUT = ROOT / "data/processed/recon_v3c/_work"
WI = ROOT / "third_party/WiLoR"


def load():
    from wilor.configs import get_config
    from wilor.models import WiLoR
    from ultralytics import YOLO
    cfg = get_config(str(WI / "pretrained_models/model_config.yaml"), update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    cfg.MANO.DATA_DIR = str(ROOT / "cache/mano/mano_v1_2/models")
    cfg.MANO.MODEL_PATH = str(ROOT / "cache/mano/mano_v1_2/models")
    cfg.MANO.MEAN_PARAMS = str(WI / "mano_data/mano_mean_params.npz")
    cfg.freeze()
    model = WiLoR.load_from_checkpoint(str(WI / "pretrained_models/wilor_final.ckpt"), strict=False, cfg=cfg, init_renderer=False)
    model = model.half().cuda().eval()
    import os
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"   # ultralytics 8.1 pickles the model class; the file is the WiLoR authors' HF release
    det = YOLO(str(WI / "pretrained_models/detector.pt"))
    det.to("cuda")
    return model, cfg, det


@torch.no_grad()
def run_clip(model, cfg, det, clip, conf=0.3, rescale=2.0):
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full
    fr, meta = read_work_frames(clip)
    rec_id = 0
    rec = {k: [] for k in ("det_id", "det_is_right", "frame", "det_conf", "is_right", "box", "global_orient", "hand_pose", "betas", "joints_mano", "cam_t_wilor", "kp2d")}
    for k0 in range(0, len(fr), 16):
        dets = det(fr[k0:k0 + 16], conf=conf, verbose=False, half=True)
        for dk, d in enumerate(dets):
            k = k0 + dk
            if len(d.boxes) == 0:
                continue
            boxes = d.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls = d.boxes.cls.cpu().numpy().astype(np.float32)
            cf = d.boxes.conf.cpu().numpy().astype(np.float32)
            # Every box is regressed twice, as the detector's handedness and as the other one (as_right), so the lift can enforce one
            # handedness per clip without a second GPU pass (a wrong handedness makes WiLoR regress a mirrored hand).
            nb = len(boxes)
            boxes = np.concatenate([boxes, boxes]); cf = np.concatenate([cf, cf]); det_right = np.concatenate([cls, cls])
            right = np.concatenate([cls, 1 - cls])
            ds = ViTDetDataset(cfg, fr[k], boxes, right, rescale_factor=rescale, fp16=True)
            dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            i0 = 0
            for b in dl:
                b = recursive_to(b, "cuda")
                out = model(b)
                mult = 2 * b["right"] - 1
                pc = out["pred_cam"].float().clone()
                pc[:, 1] = mult * pc[:, 1]
                img_size = b["img_size"].float()
                fl = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
                ct = cam_crop_to_full(pc, b["box_center"].float(), b["box_size"].float(), img_size, fl).cpu().numpy()
                J = out["pred_keypoints_3d"].float().cpu().numpy()
                rr = b["right"].cpu().numpy()
                J[:, :, 0] *= (2 * rr - 1)[:, None]
                mp = out["pred_mano_params"]
                B = len(J)
                for n in range(B):
                    P = J[n] + ct[n]
                    W, H = img_size[n].cpu().numpy()
                    f = float(fl.cpu())
                    uv = np.stack([f * P[:, 0] / P[:, 2] + W / 2, f * P[:, 1] / P[:, 2] + H / 2], 1)
                    rec["frame"].append(k); rec["det_conf"].append(cf[i0 + n]); rec["is_right"].append(rr[n]); rec["box"].append(boxes[i0 + n])
                    rec["det_is_right"].append(det_right[i0 + n]); rec["det_id"].append(rec_id + (i0 + n) % nb)
                    rec["global_orient"].append(mp["global_orient"][n].float().cpu().numpy().reshape(3, 3))
                    rec["hand_pose"].append(mp["hand_pose"][n].float().cpu().numpy().reshape(15, 3, 3))
                    rec["betas"].append(mp["betas"][n].float().cpu().numpy().reshape(10))
                    rec["joints_mano"].append(J[n]); rec["cam_t_wilor"].append(ct[n]); rec["kp2d"].append(uv)
                i0 += B
            rec_id += nb
    r = {k: np.asarray(v, np.float32) if k not in ("frame", "det_id") else np.asarray(v, np.int32) for k, v in rec.items()}
    for k, shp in (("box", (0, 4)), ("global_orient", (0, 3, 3)), ("hand_pose", (0, 15, 3, 3)), ("betas", (0, 10)), ("joints_mano", (0, 21, 3)),
                   ("cam_t_wilor", (0, 3)), ("kp2d", (0, 21, 2))):
        if len(r[k]) == 0:
            r[k] = np.zeros(shp, np.float32)
    r["n_frames"] = np.int32(len(fr))
    r["focal_wilor"] = np.float32(cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(fr[0].shape[:2]))
    r["image_size_wh"] = np.array([fr[0].shape[1], fr[0].shape[0]], np.int32)
    return r


def main():
    clips = sys.argv[1:] or [c for c in all_clips() if (ROOT / "data/processed/recon_v3" / f"{c}.npz").exists()]
    model, cfg, det = load()
    for c in clips:
        f = OUT / c / "hand_wilor.npz"
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            r = run_clip(model, cfg, det, c)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"FAILED {c}: {ex!r}", flush=True)
            continue
        np.savez_compressed(f, **r)
        nf = int(r["n_frames"])
        fr_with = len(set(r["frame"].tolist()))
        print(f"{c}: {nf} frames, {len(r['frame'])} detections, frames with >=1 hand {fr_with}/{nf}, detector right frac {r['det_is_right'].mean() if len(r['is_right']) else float('nan'):.2f}, "
              f"{time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
