"""Real-video perception run on a rendered sim clip, so that the world model can be tested in a held-out simulated environment
with inputs made the same way as for a real video. Runs at the episode's keyframes, for the frozen watch model:
  sam     SAM 3 image model, text prompts = the real video's own vocabulary (data/generic/<video>/vocab.json, named by the VLM) + 'table'
          (perception.scene.sam3_inv_worker.run on the rendered frames)                                [SAM venv, GPU]
  depth   MoGe-3 ViT-L (resolution level 9, as perception.ingest), with the camera's horizontal FOV given   [vla env + MoGe paths, GPU]
  tokens  per keyframe: depth scale from the SAM 3 'table' mask (table points on the plane z = 0, the pipeline's per-frame table-plane
          scale); duplicate masks across prompts merged (IoU >= 0.5, highest probability kept, as perception.video_tracking);
          each object's 3D box from its mask + depth (perception.tracking.obb) -> its token geometry (frame_model.box_feat); its
          DINOv2-base crop (vision.crop_obj / _enc); identities by nearest box to the object's last position among masks of its own
          phrase (the scene objects' start layout at the first keyframe); an unseen object keeps its last box. Label masks (k + 1)
          give the image positions.
Camera pose to the table and intrinsics come from the sim camera (camera_<episode>_<world>.json); for a real video they come from the
twin's calibration. No human hands are in view (robot arms move the objects), so the model gets no hand / contact inputs.
  PYTHONPATH=<sam3>:. $SAM_PY -m cleantables.worldmodel.finetune.perceive sam --dir D --world seed8
  PYTHONPATH=<MoGe>:. python -m cleantables.worldmodel.finetune.perceive depth --dir D --world seed8
  python -m cleantables.worldmodel.finetune.perceive tokens --dir D --world seed8   -> D/perceived_<world>.npz + masks_sam_<episode>_<world>/"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TRACKS = CT_DATA + "/wm_eval/wm_v3_tracks/tracks_all.npz"
MH, MW = 270, 480


def setup(a):
    D = Path(a.dir); cam = json.load(open(D / f"camera_{a.episode}_{a.world}.json"))
    z = np.load(TRACKS, allow_pickle=True)
    i = np.nonzero(z["group"] == a.episode)[0]; i = i[np.argsort(z["snap_k"][i])]
    kk = z["snap_k"][i].astype(int)
    fr = [int(round((k - kk[0]) / 10 * cam["fps"])) for k in kk]
    S = json.load(open(ROOT / "data/twin_table/ep/_scalefix" / a.episode / "scene.json"))
    vocab = json.load(open(ROOT / "data/generic" / S["video"] / "vocab.json"))["phrases"]
    return D, cam, z, i, kk, fr, S, vocab


def img(D, a, f):
    return cv2.resize(cv2.imread(str(D / f"frames_{a.episode}_{a.world}" / f"{f:05d}.jpg")), (960, 540), interpolation=cv2.INTER_AREA)


def run_sam(a):
    from cleantables.perception.scene import sam3_inv_worker as W
    D, cam, z, i, kk, fr, S, vocab = setup(a)
    W.read_frame = lambda v, k: img(D, a, k)                                    # the rendered frames instead of a video's work frames
    W.run(f"{a.episode}_{a.world}", sorted(set(fr)), str(D / f"sam3_{a.world}.npz"), vocab + ["table"], "cuda")


def run_depth(a):
    import torch
    from cleantables.perception.recon.depth_models import load
    D, cam, z, i, kk, fr, S, vocab = setup(a)
    K = np.array(cam["K"]); fov = float(np.degrees(2 * np.arctan(cam["wh"][0] / 2 / K[0, 0])))
    mdl = load("moge3_vitl", "cuda"); out = []
    for f in fr:
        with torch.no_grad():
            o = mdl.m.infer(mdl._t(img(D, a, f)), resolution_level=9, fov_x=fov, use_fp16=True)
        d = o["depth"].float().cpu().numpy(); m = o["mask"].cpu().numpy().astype(bool) & np.isfinite(d)    # as perception.ingest: invalid -> 0
        out.append(cv2.resize(np.where(m, d, 0).astype(np.float32), (MW, MH), interpolation=cv2.INTER_NEAREST))
    np.savez(D / f"depth_{a.world}.npz", frames=np.array(fr), depth=np.stack(out).astype(np.float32), fov_x=fov)
    print(f"[perceive] MoGe-3 depth for {len(fr)} keyframes, fov {fov:.1f} deg", flush=True)


def perceive_frame(im, dets, Dk, Kf, T, last, lastg, lastc, phr, prev_scale=1.0):
    """One frame: dets = [(mask 270 x 480, phrase, prob)] from SAM 3, Dk the MoGe depth (270 x 480), Kf at 960 x 540, T camera -> table.
    Depth scale from the 'table' masks (prob >= 0.5) on the plane z = 0; duplicates across prompts merged (IoU >= 0.5, the most probable
    kept); a 3D box per mask (tracking.obb) -> token geometry (wm.box_feat) + DINOv2-base crop; each scene object <- the nearest box of its
    own phrase (within 25 cm of its last position; Hungarian). last / lastg / lastc are updated in place (an unseen object keeps them).
    -> geo (n, 8), crop (n, 768), seen (n,), label mask (k + 1), scale."""
    from scipy.optimize import linear_sum_assignment
    from cleantables.perception.objects import iou
    from cleantables.perception.tracking import obb
    from cleantables.perception.vision import _enc, crop_obj
    from cleantables.perception.frame_model import box_feat
    n = len(phr); K = Kf * 0.5; K[2, 2] = 1.0
    tab = np.zeros((MH, MW), bool)
    for m, p, pr in dets:
        if p == "table" and pr >= 0.5:
            tab |= m
    ys, xs = np.nonzero(tab & np.isfinite(Dk) & (Dk > 0))
    rays = np.linalg.solve(K, np.stack([xs + 0.5, ys + 0.5, np.ones(len(xs))]))                # (3, n) camera rays (z = 1)
    sz = T[2, :3] @ (rays * Dk[ys, xs])                                                          # table z per unit depth scale
    scale = float(np.median(-T[2, 3] / sz[sz < -1e-6])) if (sz < -1e-6).sum() > 50 else prev_scale
    ob = sorted([d for d in dets if d[1] != "table"], key=lambda d: -d[2]); keep = []
    for d in ob:                                                                                 # merge duplicates across prompts (as perception.video_tracking)
        if all(iou(d[0], q[0]) < 0.5 for q in keep):
            keep.append(d)
    boxes = [(d, obb(Dk * scale, Kf, T, d[0])) for d in keep]
    boxes = [(d, b) for d, b in boxes if b is not None]
    C = np.full((n, max(1, len(boxes))), 1e3)
    for r_ in range(n):
        for c_, (d, b) in enumerate(boxes):
            if d[1] == phr[r_]:
                C[r_, c_] = np.hypot(b["x"] - last[r_, 0], b["y"] - last[r_, 1])
    lab = np.zeros((MH, MW), np.uint8); seen = np.zeros(n, bool)
    if boxes:
        for r_, c_ in zip(*linear_sum_assignment(C)):
            if C[r_, c_] > 0.25:
                continue
            d, b = boxes[c_]
            lastg[r_] = box_feat(b); last[r_] = [b["x"], b["y"], b["top"]]; seen[r_] = True
            lastc[r_] = _enc([crop_obj(im, d[0])])[0]; lab[d[0] & (lab == 0)] = r_ + 1
    geo = np.stack([lastg[r_] if lastg[r_] is not None else np.r_[last[r_], 0.05, 0.05, last[r_, 2], 0.0, 1.0] for r_ in range(n)]).astype(np.float32)
    return geo, np.stack(lastc), seen, lab, scale


def phrases_for(S, vocab):
    """Each scene object's vocabulary phrase: the one sharing most words with its category."""
    word = lambda s: set(s.replace("_", " ").replace("/", " ").split())                                      # noqa: E731
    return [max(vocab, key=lambda p: len(word(p) & word(o["category"]))) for o in S["objects"]]


def run_sam_frames(a):
    """SAM 3 on any frames of a folder (used by rollout.plan): --frames_dir, --frames f1,f2,.., --out, the real video's vocabulary + 'table'."""
    from cleantables.perception.scene import sam3_inv_worker as W
    S = json.load(open(ROOT / "data/twin_table/ep/_scalefix" / a.episode / "scene.json"))
    vocab = json.load(open(ROOT / "data/generic" / S["video"] / "vocab.json"))["phrases"]
    W.read_frame = lambda v, k: cv2.resize(cv2.imread(str(Path(a.frames_dir) / f"{k:05d}.jpg")), (960, 540), interpolation=cv2.INTER_AREA)
    W.run("frames", [int(x) for x in a.frames.split(",")], a.out, vocab + ["table"], "cuda")


def run_tokens(a):
    from scipy.optimize import linear_sum_assignment
    from cleantables.perception.objects import iou
    from cleantables.perception.tracking import obb
    from cleantables.perception.vision import _enc, crop_obj
    from cleantables.perception.frame_model import box_feat
    D, cam, z, i, kk, fr, S, vocab = setup(a)
    sm = np.load(D / f"sam3_{a.world}.npz"); dp = np.load(D / f"depth_{a.world}.npz")
    m_all = np.unpackbits(sm["masks"])[:int(np.prod(sm["shape"]))].reshape(sm["shape"]).astype(bool)
    prompts = [str(p) for p in sm["prompts"]]
    Kf = np.array(cam["K"], float) * (960 / cam["wh"][0]); Kf[2, 2] = 1.0                                   # at 960 x 540 (backproject's convention)
    K = Kf * 0.5; K[2, 2] = 1.0                                                                             # at the 480 x 270 depth / masks
    T = np.array(cam["T_cam_to_table"], float)
    # each scene object <- its tracker id (pairing of twin.dr_render) and the vocabulary phrase of its category (by its words)
    g0 = {int(o): z["obj_geo"][i[0]][s][:3] for s, o in enumerate(z["obj_id"][i[0]]) if z["obj_mask"][i[0]][s] and z["obj_kind"][i[0]][s] == 0}
    cand = list(g0); dd = np.array([[np.linalg.norm(np.array(ob["pos"][:2]) - g0[o][:2]) for o in cand] for ob in S["objects"]])
    track = np.full(len(S["objects"]), -1)
    for _ in range(min(len(S["objects"]), len(cand))):
        r_, c_ = np.unravel_index(np.nanargmin(dd), dd.shape); track[r_] = cand[c_]; dd[r_, :] = np.nan; dd[:, c_] = np.nan
    phr = phrases_for(S, vocab)
    print("[perceive] scene object -> track id / phrase:", {o["id"]: (int(t), p) for o, t, p in zip(S["objects"], track, phr)}, flush=True)
    n = len(S["objects"]); last = np.array([o["pos"] for o in S["objects"]], float)
    geo = np.zeros((len(kk), n, 8), np.float32); crop = np.zeros((len(kk), n, 768), np.float16); seen = np.zeros((len(kk), n), bool)
    scale = np.zeros(len(kk)); lastg = [None] * n; lastc = [np.zeros(768, np.float16)] * n
    md = D / f"masks_sam_{a.episode}_{a.world}"; md.mkdir(exist_ok=True)
    for j, f in enumerate(fr):
        sel = np.flatnonzero(sm["frame"] == f)
        dets = [(m_all[q], prompts[int(sm["prompt"][q])], float(sm["prob"][q])) for q in sel]
        g_, c_, sn, lab, scale[j] = perceive_frame(img(D, a, f), dets, dp["depth"][j], Kf, T, last, lastg, lastc, phr, scale[j - 1] if j else 1.0)
        geo[j], crop[j], seen[j] = g_, c_, sn
        cv2.imwrite(str(md / f"{f:05d}.png"), lab)
        print(f"[perceive] k {kk[j]} (frame {f}): table scale {scale[j]:.3f}, seen {int(seen[j].sum())}/{n}", flush=True)
    np.savez(D / f"perceived_{a.world}.npz", snap_k=kk, frames=np.array(fr), track_id=track, geo=geo, crop=crop, seen=seen, scale=scale,
             phrases=np.array(phr))
    print(f"[perceive] -> {D / f'perceived_{a.world}.npz'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sam", "depth", "tokens", "sam_frames"]); ap.add_argument("--dir", default="")
    ap.add_argument("--frames_dir", default=""); ap.add_argument("--frames", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--world", default="seed8"); ap.add_argument("--episode", default="IMG_2516_t11")
    a = ap.parse_args()
    dict(sam=run_sam, depth=run_depth, tokens=run_tokens, sam_frames=run_sam_frames)[a.stage](a)
