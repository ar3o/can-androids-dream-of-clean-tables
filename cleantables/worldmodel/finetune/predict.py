"""Keyframe-by-keyframe predictions of the pretrained watch model on one held-out video, for a pause-predict-reveal video: pause at
a frame, show the model's outputs, then show what really happens; optionally on a randomised table, with the model seeing its pixels.

Model: watch_notab (the 3-seed ensemble, sd_scale), the held-out test examples of --episode (never in training), mode 0 (video + 3D,
human hand / contact inputs as recorded), batched as in worldmodel.evaluate. Per keyframe and object:
  path   bridge.watch_path: the T2 4-s path shape (median of K samples) bent to start at the object and end at its T1 end spot
         (as drawn in the robot demos); goal = the T1 end spot (top centre), sd = its predicted spread (x sd_scale)
  next   p_now (argmax over the movable objects = the object the model expects to move next)
  truth  the tracker's real positions at the T2 steps (t2_disp, where ok) and the episode's real end (t1_pos, where ok)
--video NPZ (twin.dr_render --wm_input frames encoded by finetune.input_features): the video inputs (V-JEPA grid, DINOv2 grid, each
object's image position) are replaced by those of the rendered (randomised) world, as the robot's live video path does
(physics.live_plan); per-object appearance / crop inputs and every 3D input stay the real video's.
  CUDA_VISIBLE_DEVICES= python -m cleantables.worldmodel.finetune.predict --episode IMG_2516_t11 --out preds_real.json [--video feats.npz]
(env: WM_DATA_ROOT, TASKS3_DATA=tasks3_watch_h23, WM_V2_FEATS(_EXT), WM_TEXT_DIR, WM_RUNS2)"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="IMG_2516_t11"); ap.add_argument("--run", default="watch_notab")
    ap.add_argument("--out", required=True); ap.add_argument("--video", default="")
    ap.add_argument("--tokens", default="", help="wm_sim_perceive.py's perceived objects: their geometry / crop replace the real video's; "
                    "objects it has not got are dropped; no hand / contact inputs (no human hands in view)")
    ap.add_argument("--no_hands", action="store_true", help="hand / contact inputs off (as with --tokens), real objects kept: the ablation")
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from cleantables.worldmodel.base_model import PS
    from cleantables.worldmodel.bridge import watch_path, watch_readout
    from cleantables.worldmodel.data import Data
    from cleantables.worldmodel.evaluate import Batcher, load_run
    D = Data(text="minilm"); Bt = Batcher(D)
    ms, kind, cfg = load_run(a.run)
    grp = D.S.d["group"].astype(str); kk = D.S.d["snap_k"].astype(int)
    idx = np.array([i for i in D.test_idx if grp[i] == a.episode], int)
    assert len(idx), f"{a.episode}: no held-out test examples"
    idx = idx[np.argsort(kk[idx])]
    V = dict(np.load(a.video, allow_pickle=True)) if a.video else None
    TK = dict(np.load(a.tokens, allow_pickle=True)) if a.tokens else None
    out = dict(episode=a.episode, run=a.run, n_members=len(ms), mode="0 (video + 3D)", video=a.video or "real", tokens=a.tokens or "real",
               steps_s=[0.5 * (h + 1) for h in range(8)], keyframes=[])
    for i in idx:
        b = Bt.get("v2", cfg, np.array([i]))
        oid = np.asarray(D.S.d["obj_id"][i]).astype(int)                       # slot -> tracker id (as in evaluate)
        if V is not None:                                                       # the rendered world's video inputs at this keyframe
            r = int(np.flatnonzero(V["snap_k"] == kk[i])[0])
            b["vj_grid"] = torch.as_tensor(V["vj_grid"][r:r + 1]).float(); b["dn_grid"] = torch.as_tensor(V["dn_grid"][r:r + 1]).float()
            uv = torch.zeros_like(b["uv"]); uk = torch.zeros_like(b["uv_ok"])
            for s, o in enumerate(oid):
                hit = np.flatnonzero(V["track_id"] == o)
                if len(hit) and V["uv_ok"][r, hit[0]]:
                    uv[0, s] = torch.as_tensor(V["uv"][r, hit[0]]); uk[0, s] = True
            b["uv"], b["uv_ok"] = uv, uk & b["obj_mask"].bool()
            b["vid_ok"] = torch.ones_like(b["vid_ok"]); b["vj_ok"] = b["vid_ok"]
            if "app_vj" in V:                                                   # the scene's own appearance tokens (causal), not the real video's
                for k_ in ("app_vj", "app_dn", "app_ctl"):
                    b[k_] = torch.zeros_like(b[k_])
                b["app_ok"] = torch.zeros_like(b["app_ok"])
                for s, o in enumerate(oid):
                    hit = np.flatnonzero(V["track_id"] == o)
                    if len(hit) and V["app_ok"][r, hit[0]] and bool(b["obj_mask"][0, s]):
                        for k_ in ("app_vj", "app_dn", "app_ctl"):
                            b[k_][0, s] = torch.as_tensor(V[k_][r, hit[0]].astype(np.float32))
                        b["app_ok"][0, s] = True
        if TK is not None:                                                      # the perceived (sim) objects instead of the real video's
            r = int(np.flatnonzero(TK["snap_k"] == kk[i])[0])
            for s, o in enumerate(oid):
                if not (b["obj_mask"][0, s] and int(b["obj_kind"][0, s]) == 0):
                    continue
                hit = np.flatnonzero(TK["track_id"] == o)
                if not len(hit):
                    b["obj_mask"][0, s] = 0; continue
                b["obj_geo"][0, s] = torch.as_tensor(TK["geo"][r, hit[0]]).to(b["obj_geo"].dtype)
                b["obj_crop"][0, s] = torch.as_tensor(TK["crop"][r, hit[0]].astype(np.float32)).to(b["obj_crop"].dtype); b["obj_crop_ok"][0, s] = 1
        if TK is not None or a.no_hands:                                        # no human hands in view: no hand / contact inputs
            b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
            b["obj_contact_ok"] = torch.zeros_like(b["obj_contact_ok"]); b["obj_contact2_ok"] = torch.zeros_like(b["obj_contact2_ok"])
        md = torch.zeros(1, dtype=torch.long)
        torch.manual_seed(a.seed)
        Es = []
        for m in ms:
            E = m.encode(b, md); E["kind"] = b["obj_kind"]; Es.append(E)
        R = watch_readout(ms, Es, paths=True, K=a.K)
        cur = b["obj_geo"][0, :, :3].float(); obj = (b["obj_mask"][0].bool() & (b["obj_kind"][0] == 0)).numpy()
        traj = R["traj"][0] * 1.0                                               # (n, H, 3) metres (sample_traj already x PS)
        path = watch_path(cur, R["goal"][0], traj)
        n = cur.shape[0]
        t2 = b["t2_disp"].view(1, n, -1, 3)[0].float(); ok2 = b["t2_ok"][0].bool()
        pn = R["p_now"][0].masked_fill(~torch.as_tensor(obj), -1)
        objs = []
        for s in np.flatnonzero(obj):
            objs.append(dict(slot=int(s), track_id=int(oid[s]), cur=cur[s].tolist(), goal=R["goal"][0, s].tolist(), sd=R["sd"][0, s].tolist(),
                             path=path[s].tolist(), t2_pred=(cur[s] + traj[s]).tolist(), p_now=float(R["p_now"][0, s]),
                             t2_true=[(cur[s] + t2[s, h]).tolist() if bool(ok2[s, h]) else None for h in range(t2.shape[1])],
                             t1_true=b["t1_pos"][0, s].tolist() if bool(b["t1_ok"][0, s]) else None))
        out["keyframes"].append(dict(snap_k=int(kk[i]), t=kk[i] / 10.0, next_track_id=int(oid[int(pn.argmax())]), stop=bool(R["stop"][0]),
                                     vid_ok=bool(b["vid_ok"][0]), objects=objs))
        print(f"[wm-pred] k {kk[i]}: next track {out['keyframes'][-1]['next_track_id']}, {len(objs)} objects", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wm-pred] {len(idx)} keyframes -> {a.out} (PS {PS})", flush=True)


if __name__ == "__main__":
    main()
