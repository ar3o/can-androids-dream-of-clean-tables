"""Closed loop: the frozen watch model's plan carried out in sim, in a new world or in the twin of the real scene, so that errors
can be compared on the same footing. Per step:
  1 perceive  the last rendered frame treated as real video: SAM 3 (the real video's own vocabulary + 'table'), MoGe-3 depth scaled
              to the table plane, a 3D box per mask (perception.tracking.obb) -> token geometry + DINOv2-base crop
              (finetune.perceive.perceive_frame); V-JEPA 2 L of the held clip (now, -1 .. -4 s) + DINOv2 L + each object's mask
              centroid (as finetune.input_features); no hand / contact inputs (no human hands in view)
  2 predict   the world model (watch_notab, 3 seeds, frozen) on the real episode's batch at the same elapsed time with those inputs;
              --mode 0 = video + 3D (as trained), --mode 1 = its 3D-only mode (V-JEPA, DINOv2 and appearance tokens off, the
              ablation); its next object (argmax p_now over the objects not yet moved whose end spot is >= 3 cm away; none: stop),
              that object's predicted path (bridge.watch_path) and end spot, and its STOP read-out
  3 execute   a robot arm (the one whose shoulder is nearer) carries that object along that path (shifted to start at the object's
              true position, with a 6 cm lift arc over its length so it clears rims) and lets go; twin.dr_render plays it in PhysX
              (synthetic track + hand: approach 1 s, carry at 0.25 m/s (1-3 s), release 0.4 s, back 0.8 s) from the previous step's
              end state
The loop stops at the model's STOP, when its chosen object's end spot is < 3 cm away, or after --max_steps. Every step's frames, the
sim's object positions and the model's outputs go to OUT/rollout.json.
  PYTHONPATH=<MoGe>:. python -m cleantables.worldmodel.rollout.plan --world seed8|twin --mode 0|1 --out OUT
(environment as for finetune.predict)"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.video_live import COV_MIN, MAXO, SLOT_STILL, VJ_HW, _gpu_prep
from cleantables.worldmodel.finetune.perceive import MH, MW, perceive_frame, phrases_for

ROOT = Path(__file__).resolve().parents[3]
RTX = Path(CT_DATA + "/wm_eval/wm_v2_seg/videos/stageA/rtx_dr")
SAM_ENV = dict(PYTHONPATH=f"{os.environ.get('SAM3_REPO', 'third_party/sam3')}:{ROOT}", HF_HUB_OFFLINE="1")
SAM_PY = os.environ.get("SAM3_PY", sys.executable)
FPS, DT = 30, 0.1
STANCE = dict(shoulder=dict(left=[-0.9, 0.26, 0.6], right=[-0.9, -0.14, 0.6]), facing_deg=0.0,
              note="the real demonstrator's stance (render_watch_dr Gripper on IMG_2516_t11's hands), fixed for the plan runs")
FOREARM = np.array([-0.95, -0.31, 0.0]) * 0.25 + [0, 0, 0.08]                    # wrist relative to the grasp point (the real forearm direction)

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="seed8"); ap.add_argument("--mode", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--episode", default="IMG_2516_t11"); ap.add_argument("--run", default="watch_notab"); ap.add_argument("--max_steps", type=int, default=6)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--progress_only", action="store_true", help="no new rollout: the WM's 'how much is left' read-out every 0.5 s through the "
                "existing rollout in --out (same perception and inputs) -> OUT/progress.json")
ap.add_argument("--progress_v2", action="store_true", help="no new rollout: the EARLIER v2 world model's trained task-progress head "
                "(runs/final_s2, p_vid, trained on the cut episodes) every 0.5 s through the existing rollout, on the same perceived inputs "
                "-> OUT/progress_v2.json")
ap.add_argument("--progress_rem", action="store_true", help="as --progress_v2, but the watch model's own label-free progress: pilot 5 "
                "runs_pilot/watch_rem (fold 0, 1 seed), progress = 1 - its learned fraction of the motion still remaining -> OUT/progress_rem.json")
ap.add_argument("--kinematic", action="store_true", help="physics off (render_watch_dr --kinematic) and no added lift: the object follows the WM's path exactly")
ap.add_argument("--carry_exact", action="store_true", help="the carried object exactly on the WM's path (collisions off, no added lift); let go, it falls "
                "and settles under PhysX (render_watch_dr --carry_exact)")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); (OUT / "all_frames").mkdir(exist_ok=True)
S = json.load(open(ROOT / "data/twin_table/ep/_scalefix" / a.episode / "scene.json"))
VOCAB = json.load(open(ROOT / "data/generic" / S["video"] / "vocab.json"))["phrases"]
PHR = phrases_for(S, VOCAB); N = len(S["objects"])
EXT = np.array([o.get("extents", [0.1, 0.1, 0.1]) for o in S["objects"]], float)
json.dump(STANCE, open(OUT / "stance.json", "w"))
WORLD_ARGS = (("--wm_cam hero --zoom 0.9 --mood flood --same_objects" if a.world != "twin" else "") + (" --kinematic" if a.kinematic else "")
              + (" --carry_exact" if a.carry_exact else ""))
LIFT = 0.0 if (a.kinematic or a.carry_exact) else 0.06                                              # m: the added lift arc (physics runs only)


def log(*x):
    print("[rollout]", *x, flush=True)


# ---- the frozen world model + the real episode's batches (the template: text, time in the episode) --------------------------------------------
torch.set_grad_enabled(False)
from cleantables.worldmodel.bridge import watch_path, watch_readout                                 # noqa: E402
from cleantables.worldmodel.data import Data                                                       # noqa: E402
from cleantables.worldmodel.evaluate import Batcher, load_run                                          # noqa: E402
D = Data(text="minilm"); Bt = Batcher(D); MS, _, CFG = load_run(a.run)
grp = D.S.d["group"].astype(str); kk = D.S.d["snap_k"].astype(int)
IDX = np.array(sorted([i for i in D.test_idx if grp[i] == a.episode], key=lambda i: kk[i]))
T0 = kk[IDX[0]] / 10.0
z = np.load(CT_DATA + "/wm_eval/wm_v3_tracks/tracks_all.npz", allow_pickle=True)
ii = np.nonzero(z["group"] == a.episode)[0]; ii = ii[np.argsort(z["snap_k"][ii])]
g0 = {int(o): z["obj_geo"][ii[0]][s][:3] for s, o in enumerate(z["obj_id"][ii[0]]) if z["obj_mask"][ii[0]][s] and z["obj_kind"][ii[0]][s] == 0}
cand = list(g0); dd = np.array([[np.linalg.norm(np.array(ob["pos"][:2]) - g0[o][:2]) for o in cand] for ob in S["objects"]])
TRACK = np.full(N, -1)
for _ in range(min(N, len(cand))):
    r_, c_ = np.unravel_index(np.nanargmin(dd), dd.shape); TRACK[r_] = cand[c_]; dd[r_, :] = np.nan; dd[:, c_] = np.nan
log("scene objects -> track ids", dict(zip([o["id"] for o in S["objects"]], TRACK.tolist())), "phrases", PHR)

# ---- encoders (loaded once; GPU) ---------------------------------------------------------------------------------------------------
import sys                                                                      # noqa: E402
for p_ in ("third_party/MoGe_latest", "cache/pylibs_moge3", "cache/pylibs"):   # MoGe's code, appended after the world model's imports (cache/pylibs has an
    sys.path.append(str(ROOT / p_))                                              # older huggingface_hub that must not shadow the env's)
from cleantables.perception.recon.depth_models import load as load_depth                    # noqa: E402
from cleantables.features.encoders import Encoder                                 # noqa: E402
ENC = dict(vj=Encoder("vj2l", layers=(24,), dev="cuda"), dn=Encoder("dino2l", layers=(24,), dev="cuda"))
for e_ in ENC.values():
    e_._prep = _gpu_prep.__get__(e_)
MOGE = load_depth("moge3_vitl", "cuda")


def frame(f):
    return cv2.imread(str(OUT / "all_frames" / f"{f:05d}.jpg"))


def perceive(fs, cam, last, lastg, lastc, prev_scale, each=None):
    """Perception tracked through frames fs (10 per second of the clip since the last decision, as a real video is tracked through
    every frame): one SAM 3 call for all, then per frame MoGe + perceive_frame, each frame's identities from the previous one's. -> the
    last frame's outputs."""
    so = OUT / f"sam_{fs[0]:05d}_{fs[-1]:05d}.npz"
    if not so.exists():
        subprocess.run([SAM_PY, "-m", "cleantables.worldmodel.finetune.perceive", "sam_frames", "--frames_dir", str(OUT / "all_frames"),
                        "--frames", ",".join(map(str, fs)), "--out", str(so)], env=dict(os.environ, **SAM_ENV), cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sm = np.load(so); m_all = np.unpackbits(sm["masks"])[:int(np.prod(sm["shape"]))].reshape(sm["shape"]).astype(bool)
    pr = [str(p) for p in sm["prompts"]]
    Kf = np.array(cam["K"], float) * (960 / cam["wh"][0]); Kf[2, 2] = 1.0; T = np.array(cam["T_cam_to_table"], float)
    fov = float(np.degrees(2 * np.arctan(480 / Kf[0, 0]))); out = None
    for f in fs:
        dets = [(m_all[q], pr[int(sm["prompt"][q])], float(sm["prob"][q])) for q in np.flatnonzero(sm["frame"] == f)]
        im = cv2.resize(frame(f), (960, 540), interpolation=cv2.INTER_AREA)
        o = MOGE.m.infer(MOGE._t(im), resolution_level=9, fov_x=fov, use_fp16=True)
        d = o["depth"].float().cpu().numpy(); mk = o["mask"].cpu().numpy().astype(bool) & np.isfinite(d)
        Dk = cv2.resize(np.where(mk, d, 0).astype(np.float32), (MW, MH), interpolation=cv2.INTER_NEAREST)
        out = perceive_frame(im, dets, Dk, Kf, T, last, lastg, lastc, PHR, out[4] if out else prev_scale)
        if each is not None:
            each(f, out)
    return out


def video_inputs(f, lab):
    st = [cv2.cvtColor(frame(max(0, f - FPS * h)), cv2.COLOR_BGR2RGB) for h in range(5)]                  # now, -1 .. -4 s (first frame held at the start)
    clip = torch.as_tensor(np.stack([st[h] for h in SLOT_STILL]))[None].cuda()
    vj = ENC["vj"](clip)[24][0].float().cpu(); dn = ENC["dn"](clip[:, -1:])[24][0].float().cpu()
    m = torch.as_tensor(lab.astype(np.int64))
    cov = F.adaptive_avg_pool2d(F.one_hot(m.clamp(max=MAXO), MAXO + 1).permute(2, 0, 1)[None].float(), VJ_HW)[0, 1:]
    yy = (torch.arange(VJ_HW[0]) + 0.5) / VJ_HW[0]; xx = (torch.arange(VJ_HW[1]) + 0.5) / VJ_HW[1]; w = cov.sum((-1, -2))
    uv = torch.stack([(cov * xx[None, None]).sum((-1, -2)), (cov * yy[None, :, None]).sum((-1, -2))], -1) / w.clamp(min=1e-6)[..., None]
    return vj, dn, uv[:N], (w > COV_MIN)[:N]


# ---- per-object appearance tokens from this scene's frames, so every input comes from the unseen scene: the recipe of
# features.video.app / features.appearance (V-JEPA / DINOv2 grid + location code pooled inside the object's mask, the median over up
# to APP_N keyframes 0.3 s apart from the start where it covers > COV_MIN cells), but causal: only the keyframes seen so far. --mode 1
# (no V-JEPA) switches them off too (app_ok = 0, as for training samples without one), so no V-JEPA reaches the model there.
from cleantables.features.video import APP_N, posenc                                              # noqa: E402
PEV = torch.as_tensor(posenc(VJ_HW)); APP = dict(frame=[], vj=[], dn=[], ctl=[], ok=[])


def app_collect(f, out):
    if APP["frame"] and f - APP["frame"][-1] < 9:                                # keyframes 0.3 s apart (as the real data's)
        return
    lab = out[3]
    st = [cv2.cvtColor(frame(max(0, f - FPS * h)), cv2.COLOR_BGR2RGB) for h in range(5)]
    clip = torch.as_tensor(np.stack([st[h] for h in SLOT_STILL]))[None].cuda()
    vj = ENC["vj"](clip)[24][0].float().cpu().reshape(*VJ_HW, -1); dn = ENC["dn"](clip[:, -1:])[24][0].float().cpu().reshape(18, 32, -1)
    oh = F.one_hot(torch.as_tensor(lab.astype(np.int64)).clamp(max=MAXO), MAXO + 1).permute(2, 0, 1)[None].float()
    cv = F.adaptive_avg_pool2d(oh, VJ_HW)[0, 1:N + 1]; cd = F.adaptive_avg_pool2d(oh, (18, 32))[0, 1:N + 1]
    wv, wd = cv.flatten(1).sum(1), cd.flatten(1).sum(1)
    APP["frame"].append(f); APP["ok"].append((wv > COV_MIN).numpy())
    APP["vj"].append((cv.flatten(1) @ vj.flatten(0, 1) / wv.clamp(min=1e-6)[:, None]).numpy())
    APP["dn"].append((cd.flatten(1) @ dn.flatten(0, 1) / wd.clamp(min=1e-6)[:, None]).numpy())
    APP["ctl"].append((cv.flatten(1) @ PEV.flatten(0, 1) / wv.clamp(min=1e-6)[:, None]).numpy())


def app_at(fmax):
    """Each object's token: the median over its first APP_N visible keyframes up to frame fmax."""
    fr_ = np.array(APP["frame"]); okm = np.array(APP["ok"]) & (fr_ <= fmax)[:, None]
    out = dict(vj=np.zeros((N, 1024), np.float32), dn=np.zeros((N, 1024), np.float32), ctl=np.zeros((N, PEV.shape[-1]), np.float32), ok=np.zeros(N, bool))
    for r in range(N):
        rows_ = np.flatnonzero(okm[:, r])[:APP_N]
        if len(rows_):
            for k_ in ("vj", "dn", "ctl"):
                out[k_][r] = np.median(np.array(APP[k_])[rows_, r], 0)
            out["ok"][r] = True
    return out


def set_app(b, slot, app):
    for k_ in ("app_vj", "app_dn", "app_ctl"):
        b[k_] = torch.zeros_like(b[k_])
    b["app_ok"] = torch.zeros_like(b["app_ok"])
    if app is None or a.mode == 1:
        return
    for r, s_ in slot.items():
        if app["ok"][r]:
            b["app_vj"][0, s_] = torch.as_tensor(app["vj"][r]); b["app_dn"][0, s_] = torch.as_tensor(app["dn"][r])
            b["app_ctl"][0, s_] = torch.as_tensor(app["ctl"][r]); b["app_ok"][0, s_] = True


def predict(tau, geo, crop, vid, moved=(), app=None):
    i = IDX[int(np.argmin(np.abs(kk[IDX] / 10.0 - (T0 + tau))))]                # the real episode's batch at the same elapsed time
    b = Bt.get("v2", CFG, np.array([i])); oid = np.asarray(D.S.d["obj_id"][i]).astype(int)
    slot = {}
    for s, o in enumerate(oid):
        if not (b["obj_mask"][0, s] and int(b["obj_kind"][0, s]) == 0):
            continue
        hit = np.flatnonzero(TRACK == o)
        if not len(hit):
            b["obj_mask"][0, s] = 0; continue
        r = int(hit[0]); slot[r] = s
        b["obj_geo"][0, s] = torch.as_tensor(geo[r]).to(b["obj_geo"].dtype)
        b["obj_crop"][0, s] = torch.as_tensor(crop[r].astype(np.float32)).to(b["obj_crop"].dtype); b["obj_crop_ok"][0, s] = 1
    b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
    b["obj_contact_ok"] = torch.zeros_like(b["obj_contact_ok"]); b["obj_contact2_ok"] = torch.zeros_like(b["obj_contact2_ok"])
    set_app(b, slot, app)
    if a.mode == 0:
        vj, dn, uv, uk = vid
        b["vj_grid"] = vj.reshape(1, *vj.shape[-3:]); b["dn_grid"] = dn.reshape(1, *dn.shape[-3:])
        U = torch.zeros_like(b["uv"]); UK = torch.zeros_like(b["uv_ok"])
        for r, s in slot.items():
            U[0, s] = uv[r]; UK[0, s] = bool(uk[r])
        b["uv"], b["uv_ok"] = U, UK & b["obj_mask"].bool(); b["vid_ok"] = torch.ones_like(b["vid_ok"]); b["vj_ok"] = b["vid_ok"]
    md = torch.full((1,), a.mode, dtype=torch.long)
    torch.manual_seed(a.seed); Es = []
    for m in MS:
        E = m.encode(b, md); E["kind"] = b["obj_kind"]; Es.append(E)
    R = watch_readout(MS, Es, paths=True, K=a.K)
    cur = b["obj_geo"][0, :, :3].float(); path = watch_path(cur, R["goal"][0], R["traj"][0])
    cur = b["obj_geo"][0, :, :3].float(); path = watch_path(cur, R["goal"][0], R["traj"][0])
    far = (R["goal"][0, :, :2] - cur[:, :2]).norm(dim=-1) >= 0.03                # objects the model says still have to move
    pn = R["p_now"][0].clone()
    pn[[s for s in range(pn.shape[0]) if s not in slot.values() or any(slot.get(r) == s for r in moved) or not bool(far[s])]] = -1
    objs = [dict(obj=r, track_id=int(TRACK[r]), cur=cur[s].tolist(), goal=R["goal"][0, s].tolist(), sd=R["sd"][0, s].tolist(),
                 path=path[s].tolist(), p_now=float(R["p_now"][0, s])) for r, s in sorted(slot.items())]
    if float(pn.max()) < 0:                                                     # nothing left that the model wants moved
        return dict(next=None, stop=True, objects=objs, template_k=int(kk[i]), p_stop=float(R["p_stop"][0]))
    inv = {s: r for r, s in slot.items()}; nxt = inv[int(pn.argmax())]
    objs = [dict(obj=r, track_id=int(TRACK[r]), cur=cur[s].tolist(), goal=R["goal"][0, s].tolist(), sd=R["sd"][0, s].tolist(),
                 path=path[s].tolist(), p_now=float(R["p_now"][0, s])) for r, s in sorted(slot.items())]
    return dict(next=nxt, stop=bool(R["stop"][0]), objects=objs, template_k=int(kk[i]), p_stop=float(R["p_stop"][0]))


def synth(step_dir, tau, top, k, P_exec, carry):
    """The step's synthetic track (all objects, 0.1 s samples) + hands for twin.dr_render: object k follows P_exec while held."""
    t_app, t_rel, t_back = 1.0, 0.4, 0.8
    dur = t_app + carry + t_rel + t_back; n = int(round(dur / DT)) + 1; tt = np.arange(n) * DT
    MAX = 24; G = np.zeros((n, MAX, 8), np.float32); held = np.zeros((n, MAX), np.float32)
    pos = np.repeat(top[None], n, 0)
    for j, t in enumerate(tt):
        u = np.clip((t - t_app) / carry, 0, 1) if k is not None else 0.0
        if k is not None:
            pos[j, k] = P_exec[min(len(P_exec) - 1, int(round(u * (len(P_exec) - 1))))]
            held[j, k] = 1.0 if t_app - 0.15 <= t <= t_app + carry + 0.1 else 0.0
    G[:, :N, :3] = pos; G[:, :N, 3:5] = EXT[None, :, :2]; G[:, :N, 5] = pos[..., 2]; G[:, :N, 7] = 1.0
    ks = int(round(10 * (T0 + tau))) + np.arange(n)
    np.savez(step_dir / "tracks.npz", group=np.array([a.episode] * n), snap_k=ks, obj_geo=G,
             obj_mask=np.r_[np.ones(N), np.zeros(MAX - N)][None].repeat(n, 0).astype(np.uint8),
             obj_id=np.r_[100 + np.arange(N), -np.ones(MAX - N)][None].repeat(n, 0).astype(np.int32),
             obj_kind=np.r_[np.zeros(N), -np.ones(MAX - N)][None].repeat(n, 0).astype(np.int8),
             contact_ok=np.ones(n, np.uint8), obj_contact=held, hands23_ok=np.zeros(n, np.uint8), hands23_contact=np.zeros((n, MAX), np.float32))
    T = np.array(S["cameras"]["main"]["T_cam_to_table"], float)
    rest = {s: np.array(STANCE["shoulder"][s]) + [0.35, 0.0, -0.35] for s in ("left", "right")}
    side = None if k is None else min(rest, key=lambda s: abs(STANCE["shoulder"][s][1] - top[k, 1]))
    H = {}
    for s in ("left", "right"):
        g = np.repeat(rest[s][None], n, 0)
        if s == side:
            grip = lambda p: p - [0, 0, 0.5 * EXT[k, 2]]                                                  # noqa: E731
            above = grip(P_exec[0]) + [0, 0, 0.15]
            for j, t in enumerate(tt):
                if t < t_app:
                    u = t / t_app; g[j] = (rest[s] + (above - rest[s]) * min(1, u / 0.6)) if u < 0.6 else above + (grip(P_exec[0]) - above) * (u - 0.6) / 0.4
                elif t < t_app + carry:
                    g[j] = grip(pos[j, k])
                elif t < t_app + carry + t_rel:
                    g[j] = grip(P_exec[-1])
                else:
                    u = (t - t_app - carry - t_rel) / t_back; up = grip(P_exec[-1]) + [0, 0, 0.15]
                    g[j] = grip(P_exec[-1]) + (up - grip(P_exec[-1])) * min(1, u / 0.4) if u < 0.4 else up + (rest[s] - up) * (u - 0.4) / 0.6
        J = np.repeat(g[:, None], 21, 1).copy(); yv = np.array([0, 1.0, 0])
        J[:, 4] = g - yv * 0.03; J[:, 8] = g + yv * 0.03; J[:, 0] = g + FOREARM; J[:, 9] = g
        H[f"{s}_J"] = ((J - T[:3, 3]) @ T[:3, :3]).astype(np.float32); H[f"{s}_ok"] = np.ones(n, bool)
    np.savez(step_dir / "hands.npz", k0=ks[0], **H)
    return dur


def render(step_dir, init_state):
    args = (f"--w 960 --h 540 --arms --wm_input --frames_only --tracks {step_dir / 'tracks.npz'} --hands {step_dir / 'hands.npz'} "
            f"--stance {OUT / 'stance.json'} " + (f"--init_state {init_state} " if init_state else "") + WORLD_ARGS)
    r = subprocess.run(["bash", str(RTX / "run_watch_dr.sh"), "full", a.world], env=dict(os.environ, RDR_OUT=str(step_dir), RDR_ARGS=args),
                       capture_output=True, text=True)
    log(r.stdout.strip().splitlines()[-2:] if r.stdout else r.stderr[-500:])
    wn = a.world if a.world == "twin" else a.world
    cam = json.load(open(step_dir / f"camera_{a.episode}_{wn}.json"))
    return cam, sorted((step_dir / f"frames_{a.episode}_{wn}").glob("*.jpg"))


def load_app():
    z_ = np.load(OUT / "app_sim.npz")
    for k_ in APP:
        APP[k_] = list(z_[k_])


if a.progress_v2 or a.progress_rem:
    from cleantables.worldmodel.bridge import load_models
    load_app()
    if a.progress_rem:
        RV2 = Path(os.environ["WM_EVAL_OUT"]) / "wm_v2_world/runs_pilot/watch_rem"
        MS2 = load_models(RV2, fold=0); CFG2 = json.load(open(RV2 / "fold0_seed0.json"))["cfg"]
        readp = lambda m, E: 1.0 - float(torch.sigmoid(m.rem_logit(E)))                                   # noqa: E731
    else:
        RV2 = Path(os.environ["WM_EVAL_OUT"]) / "wm_v2/runs/final_s2"
        MS2 = load_models(RV2); CFG2 = json.load(open(RV2 / "fold-1_seed0.json"))["cfg"]
        readp = lambda m, E: float(torch.sigmoid(m.p_vid(E["cls"][:, 0]).squeeze(-1)))                   # noqa: E731
    R0 = json.load(open(OUT / "rollout.json")); camr = dict(R0["camera"]); nfr = R0["n_frames"]
    fs = sorted(set(list(range(0, nfr, FPS // 2)) + [nfr - 1]))
    last = np.array([o["pos"] for o in S["objects"]], float); lastg = [None] * N; lastc = [np.zeros(768, np.float16)] * N; rows = []

    def each2(f, out):
        geo, crop, seen, lab, sc = out
        tau = R0["truth_t"][f]; i = IDX[int(np.argmin(np.abs(kk[IDX] / 10.0 - (T0 + tau))))]
        b = Bt.get("v2", CFG2, np.array([i])); oid = np.asarray(D.S.d["obj_id"][i]).astype(int); slot = {}
        for s_, o in enumerate(oid):
            if not (b["obj_mask"][0, s_] and int(b["obj_kind"][0, s_]) == 0):
                continue
            hit = np.flatnonzero(TRACK == o)
            if not len(hit):
                b["obj_mask"][0, s_] = 0; continue
            r = int(hit[0]); slot[r] = s_
            b["obj_geo"][0, s_] = torch.as_tensor(geo[r]).to(b["obj_geo"].dtype)
            b["obj_crop"][0, s_] = torch.as_tensor(crop[r].astype(np.float32)).to(b["obj_crop"].dtype); b["obj_crop_ok"][0, s_] = 1
        b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
        for k_ in ("obj_contact_ok", "obj_contact2_ok"):
            if k_ in b:
                b[k_] = torch.zeros_like(b[k_])
        if "app_vj" in b:
            set_app(b, slot, app_at(f))
        if a.mode == 0:
            vj, dn, uv, uk = video_inputs(f, lab)
            b["vj_grid"] = vj.reshape(1, *vj.shape[-3:]); b["dn_grid"] = dn.reshape(1, *dn.shape[-3:])
            U = torch.zeros_like(b["uv"]); UK = torch.zeros_like(b["uv_ok"])
            for r, s_ in slot.items():
                U[0, s_] = uv[r]; UK[0, s_] = bool(uk[r])
            b["uv"], b["uv_ok"] = U, UK & b["obj_mask"].bool(); b["vid_ok"] = torch.ones_like(b["vid_ok"]); b["vj_ok"] = b["vid_ok"]
        md = torch.full((1,), a.mode, dtype=torch.long)
        p = float(np.mean([readp(m, m.encode(b, md)) for m in MS2]))
        rows.append(dict(frame=f, progress=p, seen=int(seen.sum()))); log(f"v2 progress frame {f}: {p:.2f}")
    perceive(fs, camr, last, lastg, lastc, 1.0, each2)
    nm_ = "progress_rem.json" if a.progress_rem else "progress_v2.json"
    json.dump(dict(rule=("watch_rem (pilot 5, fold 0): 1 - sigmoid(rem_logit), its learned fraction of the motion still remaining" if a.progress_rem
                         else "the earlier v2 world model's trained task-progress head (runs/final_s2: sigmoid(p_vid), 3-seed mean)")
                   + ", on the rollout's perceived inputs; mode as the rollout", rows=rows), open(OUT / nm_, "w"), indent=1)
    log(f"progress -> {OUT / nm_}"); raise SystemExit(0)
if a.progress_only:
    load_app()
    # the model's own "how much is left": the summed table-plane distance from each object to its predicted end spot, read every
    # 0.5 s through the rollout with the rollout's own perception / inputs; progress = 1 - left / left at the start (the watch model has
    # no trained progress head); p_stop = its STOP read-out (every object within 1 sd of its end spot)
    R0 = json.load(open(OUT / "rollout.json")); camr = dict(R0["camera"]); nfr = R0["n_frames"]
    fs = sorted(set(list(range(0, nfr, FPS // 2)) + [nfr - 1]))
    last = np.array([o["pos"] for o in S["objects"]], float); lastg = [None] * N; lastc = [np.zeros(768, np.float16)] * N; rows = []

    def each(f, out):
        geo, crop, seen, lab, sc = out
        moved = [R0["objects"].index(st["chosen"]) for st in R0["steps"] if st.get("frames") and st["frames"][1] < f]
        Pp = predict(R0["truth_t"][f], geo, crop, video_inputs(f, lab) if a.mode == 0 else None, moved, app_at(f))
        left = sum(float(np.hypot(*(np.array(o["goal"][:2]) - np.array(o["cur"][:2])))) for o in Pp["objects"])
        rows.append(dict(frame=f, left_cm=100 * left, p_stop=Pp["p_stop"], seen=int(seen.sum())))
        log(f"progress frame {f}: left {100 * left:.1f} cm, p_stop {Pp['p_stop']:.2f}")
    perceive(fs, camr, last, lastg, lastc, 1.0, each)
    l0 = max(rows[0]["left_cm"], 1e-6)
    for r in rows:
        r["progress"] = float(np.clip(1 - r["left_cm"] / l0, 0, 1))
    json.dump(dict(rule="progress = 1 - (summed distance object -> its predicted end spot) / that at the start; the watch WM has no "
                        "progress head", rows=rows), open(OUT / "progress.json", "w"), indent=1)
    log(f"progress -> {OUT / 'progress.json'}"); raise SystemExit(0)
# ---- the loop --------------------------------------------------------------------------------------------------------------------------
log(f"world {a.world}, mode {a.mode} ({'video + 3D' if a.mode == 0 else '3D only: V-JEPA / DINOv2 off'}), out {OUT}")
steps, tau, nf, state = [], 0.0, 0, None
last = np.array([o["pos"] for o in S["objects"]], float); lastg = [None] * N; lastc = [np.zeros(768, np.float16)] * N; scale = 1.0
truth_t, truth_p = [], []


def run_step(name, k, P_exec, carry, top):
    global tau, nf, state
    sd = OUT / name; sd.mkdir(exist_ok=True)
    dur = synth(sd, tau, top, k, P_exec, carry)
    cam, fr = render(sd, state)
    for f in fr:
        shutil.copy(f, OUT / "all_frames" / f"{nf:05d}.jpg"); nf += 1
    truth_t.extend((tau + np.array(cam["t"]) - cam["t"][0]).tolist()); truth_p.extend(cam["top_centre"])
    st = OUT / f"{name}_end_state.json"; json.dump(dict(root_pos=cam["root_pos"], root_quat=cam["root_quat"]), open(st, "w")); state = str(st)
    tau += dur
    return cam, np.array(cam["top_centre"][-1]), (nf - len(fr), nf - 1)


top = np.array([[o["pos"][0], o["pos"][1], o["pos"][2] + o.get("extents", [0, 0, 0.05])[2] / 2] for o in S["objects"]], float)
cam, top, fr0 = run_step("step00_idle", None, None, 1.0, top)                    # 1 s at rest: the first frames to perceive
f_prev = 0; moved = []
for si in range(1, a.max_steps + 1):
    f = nf - 1
    fs = sorted(set(list(range(f_prev, f, FPS // 10)) + [f]))                     # tracked through the clip since the last decision (10 per s)
    geo, crop, seen, lab, scale = perceive(fs, cam, last, lastg, lastc, scale, each=app_collect); f_prev = f
    vid = video_inputs(f, lab) if a.mode == 0 else None
    P = predict(tau, geo, crop, vid, moved, app_at(f))
    np.savez(OUT / "app_sim.npz", **{k_: np.array(v_) for k_, v_ in APP.items()})
    if P["next"] is None:
        steps.append(dict(step=si, frame=f, t=tau, ended="nothing left that the WM wants moved", wm=P, chosen=None, seen=seen.tolist(),
                          perception_err_cm=None)); break
    perr = [float(np.hypot(*(geo[r, :2] - top[r, :2]))) for r in range(N) if seen[r]]
    k = P["next"]; ob = next(o for o in P["objects"] if o["obj"] == k)
    far = float(np.hypot(*(np.array(ob["goal"][:2]) - np.array(ob["cur"][:2]))))
    rec = dict(step=si, frame=f, t=tau, seen=seen.tolist(), perceived=geo[:, :3].tolist(), truth=top.tolist(), perception_err_cm=100 * float(np.median(perr)) if perr else None,
               wm=P, chosen=S["objects"][k]["id"], goal_dist_cm=100 * far)
    log(f"step {si}: perceived {int(seen.sum())}/{N}, perception err {rec['perception_err_cm']} cm; WM next {S['objects'][k]['id']} "
        f"({100 * far:.1f} cm away), stop {P['stop']}")
    if P["stop"] or far < 0.03:
        rec["ended"] = "WM STOP" if P["stop"] else "chosen object already at its end spot"; steps.append(rec); break
    path = np.array(ob["path"]); u = np.linspace(0, 1, len(path))[:, None]
    P_exec = path + (top[k] - path[0]) * (1 - u); P_exec[:, 2] += LIFT * np.sin(np.pi * u[:, 0])
    carry = float(np.clip(np.linalg.norm(np.diff(P_exec, axis=0), axis=1).sum() / 0.25, 1.0, 3.0))
    cam, top, frs = run_step(f"step{si:02d}", k, P_exec, carry, top); moved.append(k)
    rec.update(executed=P_exec.tolist(), frames=frs, carry_s=carry, result=top.tolist(),
               landed_cm=100 * float(np.hypot(*(top[k, :2] - np.array(ob["goal"][:2])))))
    log(f"  carried {S['objects'][k]['id']} {carry:.1f} s; it landed {rec['landed_cm']:.1f} cm from the WM's end spot")
    steps.append(rec)
    json.dump(dict(world=a.world, mode=a.mode, episode=a.episode, objects=[o["id"] for o in S["objects"]], camera=dict(K=cam["K"], wh=cam["wh"], T_cam_to_table=cam["T_cam_to_table"]),
                   fps=FPS, n_frames=nf, truth_t=truth_t, truth_top=truth_p, idle_frames=fr0, steps=steps), open(OUT / "rollout.json", "w"))
json.dump(dict(world=a.world, mode=a.mode, episode=a.episode, objects=[o["id"] for o in S["objects"]], camera=dict(K=cam["K"], wh=cam["wh"], T_cam_to_table=cam["T_cam_to_table"]),
               fps=FPS, n_frames=nf, truth_t=truth_t, truth_top=truth_p, idle_frames=fr0, steps=steps, final_top=top.tolist()), open(OUT / "rollout.json", "w"))
log(f"done: {len(steps)} WM decisions, {nf} frames -> {OUT / 'rollout.json'}")
