"""Bridge between the world model (UWM2 / UWMW checkpoints) and a simulated twin, extending cleantables.worldmodel.base_bridge.WMBridge.

By default it runs 3D-only, as on the robot, with the per-object inputs available at robot time: each twin object's
appearance tokens from the real video (V-JEPA 2 and DINOv2 pooled in its SAM 3 masks, cleantables.features.video.app),
linked through the twin object's `track` (its tracks.pkl index in the episode's video). No objg: the twin has no goal frame.
Live render features can be passed per call or set once (see cleantables.worldmodel.video_live).

  B = WMBridge2("IMG_2516_t11", OUT / "wm_v2/runs/full", seeds=(0, 1, 2), device="cpu")
  out = B.plan(obj_xyz, obj_yaw)                 # as WMBridge.plan, plus p_stop for models with the stop option
  yaw = B.goal_yaw(obj_xyz, obj_yaw, mover, dest) # end yaw of the object's long side (rad, 180 deg symmetric)
  r = B.release(obj_xyz, obj_yaw, mover, dest, path) # contact head: P(hand holds the mover) along the path + the release point
  out = B.plan(obj_xyz, obj_yaw, video=feats)        # live features for this call (dict as from video_live, batched over N); None = 3D-only
  B.set_video(vj, dn) / B.set_video_frames(frames)    # the same, sticky; set_video(None) = 3D-only
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel import base_bridge as W
from cleantables.worldmodel.schema import MAXO

import os
from pathlib import Path
if os.environ.get("WM_TWIN_ROOT"):            # alternative twin folder (e.g. links to rescaled twins)
    W.EP = Path(os.environ["WM_TWIN_ROOT"])

from cleantables.features import video as VF


def _app_table():
    A = VF._loaded(VF.FE / "app.npz")
    return {str(k): i for i, k in enumerate(A["keys"])}, A


def app_keys_for(ep, Z):
    """app.npz keys (episode:track) of the twin's objects, in Z order."""
    S = json.load(open(W.EP / ep / "scene.json"))
    track = {o["id"]: o.get("track") for o in S["objects"]}
    src = S.get("sources") or {}                                  # composite twins (e.g. DEMO_mixed_01): object id -> source episode
    return [f"{src.get(str(i), ep)}:{track[str(i)]}" if track.get(str(i)) is not None else "" for i in Z["ids"]]


# ---------------------------------------------------------------------------------------------- endref models
@torch.no_grad()
def endref_goal(models, Es, mv):
    """The ensemble's single goal for each (scene, mover) pair: a_j averaged over the members, j* = argmax, and the goal is the
    members' mean of their j* component means (object top centre, world metres). Es: per-member encodings batched like mv.
    Returns (goal (B, 3), j* (B,) token slot, a (B, 24), entropy of a (B,))."""
    from cleantables.worldmodel.model import from_frame
    A, C = [], []
    for m, E in zip(models, Es):
        M = m.end_mix(E, mv)
        A.append(F.softmax(M["logit"], -1)); C.append(from_frame(M["off"], M["p"], M["yaw"], M["sz"]))
    a = torch.stack(A).mean(0); j = a.argmax(-1); ar = torch.arange(len(j), device=j.device)
    goal = torch.stack([c[ar, j] for c in C]).mean(0)
    ent = -(a * torch.log(a.clamp(min=1e-9))).sum(-1)
    return goal, j, a, ent


@torch.no_grad()
def endref_paths(models, Es, mv, goal, K=4, steps=12):
    """Paths that end at the goal: per member K flow residuals conditioned on the goal (plan_paths), the median over members x K,
    smoothed with a 5-point mean, plus the straight line start -> goal. Returns (top-centre path (B, NP, 3) world, [(E, ctx)] per member)."""
    from cleantables.worldmodel.base_model import NP
    st = Es[0]["geo"][torch.arange(len(mv), device=mv.device), mv.clamp(min=0), :3].float()
    u = torch.linspace(0, 1, NP, device=mv.device)[None, None, :, None]
    line = u * (goal - st)[:, None, None]
    rs, cs = [], []
    for m, E in zip(models, Es):
        P, _ = m.plan_paths(E, mv, K=K, steps=steps, end=goal)
        rs.append(P[..., :3].float() - line); cs.append((E, m.ctx_end(E, mv, goal)))
    r = torch.cat(rs, 1).median(1).values
    r = F.avg_pool1d(r.transpose(1, 2), 5, 1, 2, count_include_pad=False).transpose(1, 2)
    uu = u[:, 0]                                                                       # (1, NP, 1)
    r = r - (1 - uu) * r[:, :1] - uu * r[:, -1:]                 # residual is 0 at both ends, so the path starts at the start and ends at the goal
    return st[:, None] + line[:, 0] + r, cs


def dest_from_ref(j, mv, kind):
    """The reference the end is most likely relative to, as an object index: an object slot other than the mover gives that
    object; the table token or the mover's own start gives -1 (the table)."""
    is_obj = (kind.gather(1, j[:, None]).squeeze(1) == 0) & (j != mv)
    return torch.where(is_obj, j, torch.full_like(j, -1))


# ---------------------------------------------------------------------------------------------- watch models (UWMW)
@torch.no_grad()
def watch_readout(models, Es, paths=True, K=4, steps=12):
    """Ensemble read-outs of a watch model (UWMW) for every object at once:
      p_now (B, n): predicted displacement magnitude over the horizon, normalised over the movable objects (next object = argmax)
      goal (B, n, 3): each object's T1 end (top centre): a averaged over the members, the argmax reference, the members' mean of its
            component mean; sd (B, n, 3) its spread; j (B, n), a (B, n, n), entropy (B, n)
      stop (B,), z (B,), p_stop (B,): every object's current position within one predicted sd of its T1 end
      yaw (B, n) the T1 end yaw; t5 (B, n, H) P(hand contact) at the T2 steps; traj (B, n, H, 3) median T2 displacement (paths=True)."""
    from cleantables.worldmodel.model import from_frame
    P, A, C, SD, Y, T5, TR, MS = [], [], [], [], [], [], [], []
    for m, E in zip(models, Es):
        P.append(m.next_scores(E)[0])
        M = m.t1_mix(E); A.append(F.softmax(M["logit"], -1)); C.append(from_frame(M["off"], M["p"], M["yaw"], M["sz"])); MS.append(M)
        SD.append(m.t1_mode(E)[3])
        y2 = 2 * m.t1_yaw_mode(E); Y.append(torch.stack([y2.cos(), y2.sin()], -1))
        T5.append(torch.sigmoid(m.t5_logit(E)))
        if paths:
            TR.append(m.sample_traj(E, K=K, steps=steps))
    a = torch.stack(A).mean(0); j = a.argmax(-1)
    goal = torch.stack([torch.gather(c, 2, j[..., None, None].expand(*j.shape, 1, 3)).squeeze(2) for c in C]).mean(0)
    sd = torch.stack(SD).mean(0)
    if getattr(models[0], "dist_head", False):                           # dist_head models: the ensemble spread of the chosen
        mu_s, sd_s = zip(*[m.t1_comp(M, j) for m, M in zip(models, MS)])  # component = members' mean variance + variance of the
        mu_s, sd_s = torch.stack(mu_s), torch.stack(sd_s)                 # members' means (total variance), times the global sd_scale
        sd = torch.sqrt((sd_s ** 2).mean(0) + mu_s.var(0, unbiased=False)) * float(getattr(models[0], "sd_scale", 1.0))
    E0 = Es[0]; cur = E0["geo"][..., :3].float(); mov = E0["om"] & (E0["kindx"] == 0)
    zz = ((goal - cur) / sd.clamp(min=1e-4)).abs().max(-1).values.masked_fill(~mov, 0.0).max(-1).values
    zy = torch.stack(Y).mean(0)
    out = dict(p_now=torch.stack(P).mean(0), goal=goal, sd=sd, j=j, a=a, entropy=-(a * torch.log(a.clamp(min=1e-9))).sum(-1),
               stop=zz < 1.0, z=zz, p_stop=torch.exp(-0.5 * zz ** 2), yaw=0.5 * torch.atan2(zy[..., 1], zy[..., 0]), t5=torch.stack(T5).mean(0))
    if paths:
        out["traj"] = torch.cat(TR, 1).median(1).values
    return out


def watch_path(start, goal, traj, NP=32):
    """One object's path (top centre, world): the shape of the T2 sample (displacement at the H steps, 0 at k), resampled to NP
    points as a residual around its own straight line and added to the line start -> T1 goal, so it starts at the start and ends
    at the goal. start, goal (..., 3); traj (..., H, 3) -> (..., NP, 3)."""
    pts = torch.cat([torch.zeros_like(traj[..., :1, :]), traj], -2)                       # (..., H+1, 3)
    sh = pts.shape
    q = F.interpolate(pts.reshape(-1, sh[-2], 3).transpose(1, 2), size=NP, mode="linear", align_corners=True).transpose(1, 2).reshape(*sh[:-2], NP, 3)
    u = torch.linspace(0, 1, NP, device=traj.device)[:, None]
    r = q - u * q[..., -1:, :]
    return start[..., None, :] + u * (goal - start)[..., None, :] + r


def load_models(run_dir, seeds=(0, 1, 2), fold=-1, device="cpu"):
    from cleantables.worldmodel.train import build
    ms = []
    for s in seeds:
        p = run_dir / f"fold{fold}_seed{s}.pt"
        if not p.exists():
            continue
        cfg = json.load(open(p.with_suffix(".json")))["cfg"]
        cfg["vid"] = tuple(cfg["vid"])
        m = build(cfg).to(device); m.load_state_dict(torch.load(p, map_location=device)); m.cfg = cfg; ms.append(m.eval())
        sc = run_dir / "sd_scale.json"                                   # global T1 sd scale fitted on validation data
        if sc.exists():
            m.sd_scale = float(json.load(open(sc))["sd_scale"])
    assert ms, f"no checkpoints in {run_dir}"
    return ms


class WMBridge2(W.WMBridge):
    def __init__(self, ep, run_dir, seeds=(0, 1, 2), fold=-1, device="cpu", K=8, Z=None, app_keys=None, models=None):
        self.ep, self.S = W._scene(ep)
        if Z is None:
            z = np.load(W.EP / self.ep / "wm_tokens.npz")
            Z = {k: z[k] for k in z.files}
        self.Z, self.dev, self.K, self.run_dir = Z, device, K, run_dir
        if os.environ.get("WM_BRIDGE_OBB") == "1" and len(self.S["objects"]) >= len(self.Z["ids"]) and \
                [str(o["id"]) for o in self.S["objects"][:len(self.Z["ids"])]] == [str(i) for i in self.Z["ids"]]:
            from cleantables.worldmodel.bridge_obb import obb_Z                                     # object-frame footprints from the meshes
            self.Z = obb_Z(self.ep, self.Z, self.S)
        self.models = models if models is not None else load_models(run_dir, seeds, fold, device)
        if getattr(self.models[0], "cfg", {}).get("text") == "hashed":            # 'no_text' ablation: feed the hashed text it was trained on
            from cleantables.worldmodel.common import hashed
            self.Z = dict(self.Z); self.Z["task_emb"] = hashed(str(self.Z["instruction"]))
        self.I = len(self.Z["ids"])
        assert self.I + 1 <= MAXO
        keys = app_keys if app_keys is not None else app_keys_for(self.ep, Z)
        idx, A = _app_table()
        ai = np.array([idx.get(k, -1) for k in keys])
        ok = (ai >= 0) & (A["n_frames"][np.maximum(ai, 0)] > 0)
        self._app = {}
        for nm in ("app_vj", "app_dn", "app_ctl"):
            a = torch.zeros(MAXO, A[nm].shape[1], device=device)
            a[:self.I][torch.as_tensor(ok)] = torch.as_tensor(A[nm][ai[ok]].astype(np.float32), device=device)
            self._app[nm] = a
        self._app_ok = torch.zeros(MAXO, dtype=torch.bool, device=device); self._app_ok[:self.I] = torch.as_tensor(ok, device=device)
        self.app_coverage = float(ok.mean())

    def _batch(self, obj_xyz, obj_yaw, palm_xyz):
        b = super()._batch(obj_xyz, obj_yaw, palm_xyz)
        N, dev = obj_xyz.shape[0], self.dev
        for nm, a in self._app.items():
            b[nm] = a[None].expand(N, -1, -1)
        b["app_ok"] = self._app_ok[None].expand(N, -1)
        b["uv"] = torch.zeros(N, MAXO, 2, device=dev); b["uv_ok"] = torch.zeros(N, MAXO, dtype=torch.bool, device=dev)
        b["objg"] = torch.zeros(N, MAXO, 2, device=dev); b["objg_ok"] = torch.zeros(N, MAXO, dtype=torch.bool, device=dev)
        b["dino_goal"] = torch.zeros(N, 2, device=dev)
        b["vid_ok"] = torch.zeros(N, dtype=torch.bool, device=dev)
        b.pop("vj_tok", None)
        v = getattr(self, "_video", None)
        if v is not None:                                                 # live video: this call runs with video and 3D
            ex = lambda t: t.expand(N, *t.shape[1:]) if t.shape[0] == 1 else t
            b["vj_grid"] = ex(v["vj"]); b["dn_grid"] = ex(v["dn"])
            b["vid_ok"] = ex(v["vid_ok"]) if "vid_ok" in v else torch.ones(N, dtype=torch.bool, device=dev)
            b["_force_video"] = True
            if "uv" in v:                                                 # given: visible-mask centroids on the grid
                b["uv"] = ex(v["uv"]).clone(); b["uv_ok"] = ex(v["uv_ok"]).clone()
            else:                                                         # else: project each object's 3D centre through the phone camera
                cam = self.S.get("cameras", {}).get("main") or dict(K=self.S["K"], T_cam_to_table=self.S["T_cam_to_table"], wh=self.S.get("image_wh", [960, 540]))
                K = torch.as_tensor(np.array(cam["K"], np.float32).reshape(3, 3), device=dev)
                Ti = torch.as_tensor(np.linalg.inv(np.array(cam["T_cam_to_table"], np.float64)).astype(np.float32), device=dev)
                wh = torch.as_tensor(np.array(cam.get("wh", [960, 540]), np.float32), device=dev)
                ctr = obj_xyz.clone(); ctr[..., 2] = ctr[..., 2] + 0.5 * torch.as_tensor(self.Z["extents"][:, 2], dtype=torch.float32, device=dev)
                X = ctr @ Ti[:3, :3].T + Ti[:3, 3]
                uvp = (X @ K.T)[..., :2] / X[..., 2:3].clamp(min=1e-3) / wh
                ok = (X[..., 2] > 0.05) & (uvp >= 0).all(-1) & (uvp <= 1).all(-1)
                b["uv"][:, :self.I] = uvp; b["uv_ok"][:, :self.I] = ok
            # objg / dino_goal stay off: they compare with the human episode's goal frame, which the robot does not have (training
            # also turned them off on render samples)
        return b

    def _live(self, video):
        """Video (dict or None) -> per-call video state. Keys (as from video_live, batched over N or unbatched):
        vj_grid (N,16,28,1024), dn_grid (N,18,32,1024) [also accepted: vj, dn], optional uv (N,MAXO,2) + uv_ok (N,MAXO), optional
        vid_ok (N,). objg / objg_ok / dino_goal are accepted and ignored. None -> 3D-only."""
        if video is None:
            return None
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x.detach().cpu() if torch.is_tensor(x) else x), device=self.dev).to(dt)
        vj = video.get("vj_grid", video.get("vj")); dn = video.get("dn_grid", video.get("dn"))
        v = dict(vj=t(vj).reshape(-1, 16, 28, 1024), dn=t(dn).reshape(-1, 18, 32, 1024))
        if video.get("uv") is not None and video.get("uv_ok") is not None:
            v["uv"] = t(video["uv"]).reshape(-1, MAXO, 2); v["uv_ok"] = t(video["uv_ok"], torch.bool).reshape(-1, MAXO)
        if video.get("vid_ok") is not None:
            v["vid_ok"] = t(video["vid_ok"], torch.bool).reshape(-1)
        for m in self.models:                                             # the model reads video only where mode != 1
            if not getattr(m, "_video_hook", False):
                enc = m.encode
                def encode(b, mode, _enc=enc):
                    return _enc(b, torch.zeros_like(mode) if b.get("_force_video") else mode)
                m.encode = encode; m._video_hook = True
        return v

    def set_video(self, vj=None, dn=None, **kw):
        """Sticky form of plan(..., video=...): later calls use these features until set_video(None)."""
        self._video = None if vj is None or dn is None else self._live(dict(vj_grid=vj, dn_grid=dn, **kw))

    def set_video_frames(self, frames, dev=None):
        """As set_video, from RGB frames: (T, H, W, 3) uint8 from the twin's phone camera (960x540), T >= 1 at 10 fps, the last
        one being now (with fewer than 32 the first is repeated, as with training's random clip start). Encodes with the frozen
        encoders (~1.2 GB GPU on 'cuda'), as cleantables.features.video does."""
        from cleantables.features.encoders import Encoder
        fr = np.asarray(frames)
        if fr.ndim == 3:
            fr = fr[None]
        fr = np.concatenate([np.repeat(fr[:1], max(0, 32 - len(fr)), 0), fr[-32:]], 0)
        dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(self, "_enc"):
            self._enc = dict(vj=Encoder("vj2l", layers=(24,), dev=dev), dn=Encoder("dino2l", layers=(24,), dev=dev))
        vj = self._enc["vj"](fr[None])[24][0][0].float().cpu().numpy()
        dn = self._enc["dn"](fr[-1:][None])[24][0][0].float().cpu().numpy()
        self.set_video(vj, dn)

    def _with_video(self, video, fn):
        if video is None:
            return fn()
        keep = getattr(self, "_video", None)
        self._video = self._live(video)
        try:
            return fn()
        finally:
            self._video = keep

    @torch.no_grad()
    def plan(self, obj_xyz, obj_yaw=None, palm_xyz=None, topk=2, video=None):
        """As WMBridge.plan; models with the stop option also return p_stop (N,), the ensemble P("no further object") from the joint
        softmax over the movable objects and stop, and p_next_joint (N, I), the objects' share of that softmax.
        video: live video features (see _live) for this call, batched like obj_xyz; None = 3D-only (unless set_video is active)."""
        return self._with_video(video, lambda: self._plan(obj_xyz, obj_yaw, palm_xyz, topk))

    def _watch_batch(self, obj_xyz, obj_yaw):
        obj_xyz = torch.as_tensor(obj_xyz, dtype=torch.float32, device=self.dev)
        N = obj_xyz.shape[0]
        obj_yaw = torch.zeros(N, self.I, device=self.dev) if obj_yaw is None else torch.as_tensor(obj_yaw, dtype=torch.float32, device=self.dev)
        b = self._batch(obj_xyz, obj_yaw, None)
        mode = torch.ones(N, dtype=torch.long, device=self.dev)
        Es = []
        for m in self.models:
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]; Es.append(E)
        return b, Es, N

    def _plan(self, obj_xyz, obj_yaw=None, palm_xyz=None, topk=2):
        if getattr(self.models[0], "watch", False):                          # watch models: read-outs only (plan_all gives paths)
            b, Es, N = self._watch_batch(obj_xyz, obj_yaw)
            R = watch_readout(self.models, Es, paths=False); I = self.I
            dest = R["j"][:, :I]; mv = torch.arange(I, device=self.dev)[None].expand(N, I)
            dest = torch.where((torch.gather(b["obj_kind"], 1, dest) == 0) & (dest != mv), dest, torch.full_like(dest, -1))
            return dict(p_next=R["p_now"][:, :I], p_next_joint=R["p_now"][:, :I], p_stop=R["p_stop"], stop=R["stop"], dest=dest,
                        p_dest=R["a"][:, :I], progress=torch.full((N,), float("nan")), spread=torch.zeros(N))
        out = super().plan(obj_xyz, obj_yaw, palm_xyz, topk=topk)
        if not getattr(self.models[0], "use_stop", False):
            return out
        obj_xyz = torch.as_tensor(obj_xyz, dtype=torch.float32, device=self.dev)
        N = obj_xyz.shape[0]
        obj_yaw = torch.zeros(N, self.I, device=self.dev) if obj_yaw is None else torch.as_tensor(obj_yaw, dtype=torch.float32, device=self.dev)
        b = self._batch(obj_xyz, obj_yaw, None if palm_xyz is None else torch.as_tensor(palm_xyz, dtype=torch.float32, device=self.dev))
        mode = torch.ones(N, dtype=torch.long, device=self.dev)
        pj = []
        for m in self.models:
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]
            ml, _ = m.pointers(E, torch.zeros(N, dtype=torch.long, device=self.dev))
            pj.append(F.softmax(torch.cat([ml.float(), m.stop_logit(E).float()[:, None]], -1), -1))
        P = torch.stack(pj).mean(0)
        out["p_stop"] = P[:, -1]; out["p_next_joint"] = P[:, :self.I]
        return out

    @torch.no_grad()
    def goal_yaw(self, obj_xyz, obj_yaw, mover, dest, video=None):
        if video is not None:
            return self._with_video(video, lambda: self.goal_yaw(obj_xyz, obj_yaw, mover, dest))
        """ensemble end yaw (rad) of the mover's long side; mover (N,), dest (N,) object index or -1 (table). NaN if no yaw head."""
        if getattr(self.models[0], "watch", False):                          # watch models: the mover's T1 end yaw (dest is not used)
            b, Es, N = self._watch_batch(obj_xyz, obj_yaw)
            y = watch_readout(self.models, Es, paths=False)["yaw"]
            return y[torch.arange(N), torch.as_tensor(mover, device=self.dev).long().reshape(N)].cpu()
        obj_xyz = torch.as_tensor(obj_xyz, dtype=torch.float32, device=self.dev)
        N = obj_xyz.shape[0]
        obj_yaw = torch.zeros(N, self.I, device=self.dev) if obj_yaw is None else torch.as_tensor(obj_yaw, dtype=torch.float32, device=self.dev)
        b = self._batch(obj_xyz, obj_yaw, None)
        mode = torch.ones(N, dtype=torch.long, device=self.dev)
        mv = torch.as_tensor(mover, device=self.dev).long()
        tg = torch.as_tensor(dest, device=self.dev).long(); tg = torch.where(tg >= 0, tg, torch.full_like(tg, 24))
        zs = []
        if getattr(self.models[0], "endref", False):                         # yaw at the single goal (dest is not used)
            Es = []
            for m in self.models:
                E = m.encode(b, mode); E["kind"] = b["obj_kind"]; Es.append(E)
            goal = endref_goal(self.models, Es, mv)[0]
        for k_m, m in enumerate(self.models):
            if not getattr(m, "use_yaw", False):
                return torch.full((N,), float("nan"))
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]
            c = m.ctx_end(E, mv, goal) if getattr(m, "endref", False) else m.heads_from(E, mv, tg)[0]
            y2 = 2 * m.yaw_mode(c.float())
            zs.append(torch.stack([y2.cos(), y2.sin()], -1))
        z = torch.stack(zs).mean(0)
        return 0.5 * torch.atan2(z[:, 1], z[:, 0]).cpu()


    @torch.no_grad()
    def release(self, obj_xyz, obj_yaw, mover, dest, path, run_dir=None, thr=0.5, video=None):
        if video is not None:
            return self._with_video(video, lambda: self.release(obj_xyz, obj_yaw, mover, dest, path, run_dir, thr))
        """CONTACT head (contact_train.py, trained on the frozen trunk, 3D-only, no hands): along the path the mover will follow
        (N, L, 3) table frame, metres (resampled to 32 steps if L != 32): P(a hand holds the mover) per step and hand, and the predicted release point = the last step
        where either hand's P >= thr (the robot's "let go" command; path end if no step reaches thr).
        -> dict(p_contact (N, 2, 32), off_step (N,) int, off_frac (N,) in 0..1, off_xyz (N, 3), has_contact (N,) bool)."""
        from cleantables.worldmodel.base_model import PS
        from cleantables.worldmodel.contact_head import NP, ContactHead
        if getattr(self.models[0], "watch", False):
            # watch models: T5 (the mover's predicted hand contact over the T2 steps), resampled to the path; dest and path are not used
            b, Es, N = self._watch_batch(obj_xyz, obj_yaw)
            mv = torch.as_tensor(mover, device=self.dev).long().reshape(N)
            path = torch.as_tensor(path, dtype=torch.float32, device=self.dev).reshape(N, -1, 3)
            t5 = watch_readout(self.models, Es, paths=False)["t5"][torch.arange(N), mv]
            L_ = path.shape[1] if path.shape[1] != NP + 1 else NP
            p = F.interpolate(torch.cat([t5[:, :1], t5], -1)[:, None], size=L_, mode="linear", align_corners=True)[:, 0]
            p = p[:, None].expand(N, 2, L_).clone()
            on = p.max(1).values >= thr; has = on.any(-1)
            last = torch.where(has, L_ - 1 - on.flip(-1).float().argmax(-1), torch.full((N,), L_ - 1, device=self.dev))
            return dict(p_contact=p.cpu(), off_step=last.cpu(), off_frac=(last.float() / (L_ - 1)).cpu(), off_xyz=path[torch.arange(N), last].cpu(),
                        has_contact=has.cpu())
        if not hasattr(self, "_contact"):
            rd = Path(run_dir or self.run_dir)
            hs = []
            for k, m in enumerate(self.models):
                f = rd / f"contact_seed{k}.pt"                                         # one contact head per ensemble member
                h = ContactHead(m.d).to(self.dev); h.load_state_dict(torch.load(f, map_location=self.dev)); hs.append(h.eval())
            self._contact = hs
        obj_xyz = torch.as_tensor(obj_xyz, dtype=torch.float32, device=self.dev)
        N = obj_xyz.shape[0]
        obj_yaw = torch.zeros(N, self.I, device=self.dev) if obj_yaw is None else torch.as_tensor(obj_yaw, dtype=torch.float32, device=self.dev)
        b = self._batch(obj_xyz, obj_yaw, None)                                         # no hand tokens, as in training
        mode = torch.ones(N, dtype=torch.long, device=self.dev)
        mv = torch.as_tensor(mover, device=self.dev).long().reshape(N)
        tg = torch.as_tensor(dest, device=self.dev).long().reshape(N); tg = torch.where(tg >= 0, tg, torch.full_like(tg, 24))
        path = torch.as_tensor(path, dtype=torch.float32, device=self.dev).reshape(N, -1, 3)
        if path.shape[1] != NP:                                                          # e.g. a followed path of 33 points: resample to 32
            path = F.interpolate(path.transpose(1, 2), size=NP, mode="linear", align_corners=True).transpose(1, 2)
        org = b["obj_geo"][torch.arange(N), mv, :3]
        rel = (path - org[:, None]) / PS
        ps = []
        for m, h in zip(self.models, self._contact):
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]
            # endref: the context of the followed path is its end (last point, top centre)
            c = m.ctx_end(E, mv, path[:, -1] + torch.stack([torch.zeros(N, device=self.dev), torch.zeros(N, device=self.dev),
                          torch.as_tensor(self.Z["extents"][:, 2], dtype=torch.float32, device=self.dev)[mv]], -1)) \
                if getattr(m, "endref", False) else m.heads_from(E, mv, tg)[0]
            ps.append(torch.sigmoid(h(E, c, rel).float()))
        p = torch.stack(ps).mean(0)                                                      # (N, 2, 32)
        on = p.max(1).values >= thr                                                      # either hand
        has = on.any(-1)
        last = torch.where(has, NP - 1 - on.flip(-1).float().argmax(-1), torch.full((N,), NP - 1, device=self.dev))
        return dict(p_contact=p.cpu(), off_step=last.cpu(), off_frac=(last.float() / (NP - 1)).cpu(),
                    off_xyz=path[torch.arange(N), last].cpu(), has_contact=has.cpu())


    # ------------------------------------------------------------------------------------------------------------ plan_all
    @torch.no_grad()
    def plan_all(self, obj_xyz, obj_yaw=None, taken=None, instruction=None, video=None, paths=True, K=4, steps=12, seed=None, contact=True,
                 top_m=None, path_members=None):
        """What could be moved now, and how, for every object from the current state. The trained heads are queried for all
        (env, object) pairs at once; call again after every placement.
          obj_xyz (N, I, 3) footprint centre x, y and bottom z (table frame); obj_yaw (N, I) yaw on top of the baked one (or None)
          taken (N, I) bool or None: objects the robot already holds / has reserved (e.g. the other arm's): p_now = 0 there
          instruction: None (the twin's own), a string, a list of N strings, or MiniLM vectors (384,) / (N, 384)
          video: live video features as plan(..., video=); paths: False = probabilities, destinations and end poses only (cheap)
          K, steps: flow samples per ensemble member and ODE steps; seed: an int = deterministic sampling (eval), None = random
          top_m: paths only for each env's top_m objects by p_now_free (the rest NaN); path_members: use only the first n ensemble
                 members for the paths (destinations / end poses always use all) -- the two big cost knobs
        -> dict (torch, CPU):
          p_now (N, I): P(this object is the next one to move) in the joint softmax over the movable objects and stop (ensemble
                        mean), so order follows from the state alone
          p_now_raw (N, I): the same before zeroing taken objects; p_stop (N,), progress (N,)
          movable (N, I) bool: the twin's metadata role is not 'container' / 'fixture' (the world model does not use roles)
          rests_on (N, I) long: from geometry, not the world model: the object j whose footprint contains i's centre and whose top is within
                        3 cm of i's bottom (i rests on / in j), -1 = on the table
          blocked (N, I) bool: something rests on / in this object (geometry)
          p_now_free (N, I): p_now with taken and blocked objects at 0 (not renormalised; compare with p_stop directly). p_now
                        alone does not reliably drop for a box with another box on it, so the robot uses p_now_free
          dest (N, I) object index or -1 (table); p_dest (N, I, I+1) (last column = table); dest_conf (N, I)
          end_xyz (N, I, 3) root (bottom-centre) goal position; end_yaw (N, I) the long side's yaw (rad, 180 deg symmetric);
          end_yaw_off (N, I) the yaw change to apply (wrap_half(end_yaw - current long-side yaw))
          path (N, I, 33, 3) [paths] root positions to follow, as physics.carry_check.eval_path builds them (median of K x members
                        samples, smoothed, ending at end_xyz); contact (N, I, 2, 32) and off_step / off_xyz (N, I) [paths + trained
                        contact head]: P(a hand holds the object) per path step and hand, and where to let go; has_contact
                        (N, I) bool: the head predicts contact somewhere (else off_step = the path end)
        """
        from cleantables.worldmodel.base_model import PS
        dev = self.dev
        obj_xyz = torch.as_tensor(obj_xyz, dtype=torch.float32, device=dev)
        N, I = obj_xyz.shape[:2]
        obj_yaw = torch.zeros(N, I, device=dev) if obj_yaw is None else torch.as_tensor(obj_yaw, dtype=torch.float32, device=dev)
        temb = self._task_emb(instruction, N)
        rests = self._rests_on(obj_xyz.cpu(), obj_yaw.cpu())
        blocked = torch.zeros(N, I, dtype=torch.bool)
        for j in range(I):
            blocked[:, j] = (rests == j).any(-1)
        tk = torch.zeros(N, I, dtype=torch.bool) if taken is None else torch.as_tensor(np.asarray(taken), dtype=torch.bool).reshape(N, I)
        out = self._with_video(video, lambda: self._plan_all(obj_xyz, obj_yaw, temb, paths, K, steps, seed, contact, PS, top_m, path_members, tk | blocked))
        out["p_now_raw"] = out["p_now"].clone()
        out["p_now"] = out["p_now"] * (~tk).float()
        out["p_now_free"] = out["p_now_raw"] * (~(tk | blocked)).float()
        role = np.array([str(r) for r in self.Z["role"]])
        out["movable"] = torch.as_tensor(~np.isin(role, ("container", "fixture")))[None].expand(N, -1).clone()
        out["rests_on"], out["blocked"] = rests, blocked
        return out

    def _rests_on(self, xyz, yaw, tol=0.03):
        """Geometry only: i rests on / in j if i's centre is inside j's footprint (yawed box) and j's top is within tol of i's bottom
        (or i's bottom lies between j's bottom and top, i.e. inside a container). -1 = the table."""
        ex = torch.as_tensor(self.Z["extents"], dtype=torch.float32)
        yb = torch.as_tensor(self.Z["yaw_baked"], dtype=torch.float32)[None] + yaw            # (N, I) the footprint's yaw
        d = xyz[:, :, None, :2] - xyz[:, None, :, :2]                                       # (N, i, j, 2): i's centre relative to j
        c, s_ = torch.cos(-yb)[:, None, :], torch.sin(-yb)[:, None, :]
        lx = c * d[..., 0] - s_ * d[..., 1]; ly = s_ * d[..., 0] + c * d[..., 1]
        inside = (lx.abs() <= 0.5 * ex[None, None, :, 0]) & (ly.abs() <= 0.5 * ex[None, None, :, 1])
        bot_i = xyz[..., 2][:, :, None]; bot_j = xyz[..., 2][:, None, :]; top_j = bot_j + ex[None, None, :, 2]
        on = inside & (((bot_i - top_j).abs() <= tol) | ((bot_i > bot_j + 0.005) & (bot_i < top_j)))
        I = xyz.shape[1]; on = on & ~torch.eye(I, dtype=torch.bool)[None]
        top_on = torch.where(on, top_j.expand_as(on), torch.full_like(on, -1e9, dtype=torch.float32))
        j = top_on.argmax(-1)
        return torch.where(on.any(-1), j, torch.full_like(j, -1))

    def _task_emb(self, instruction, N):
        if instruction is None:
            return None
        if isinstance(instruction, str):
            instruction = [instruction]
        if isinstance(instruction, (list, tuple)) and isinstance(instruction[0], str):
            from cleantables.worldmodel.text import embed
            cache = self.__dict__.setdefault("_emb_cache", {})
            new = [t for t in set(instruction) if t not in cache]
            if new:
                for t, e in zip(new, embed(new)):
                    cache[t] = e
            e = np.stack([cache[t] for t in instruction])
        else:
            e = np.asarray(instruction, np.float32).reshape(-1, 384)
        e = torch.as_tensor(e, dtype=torch.float32, device=self.dev)
        return e.expand(N, -1) if e.shape[0] == 1 else e

    def _plan_all(self, obj_xyz, obj_yaw, temb, paths, K, steps, seed, contact, PS, top_m=None, path_members=None, excl=None):
        dev, N, I = self.dev, obj_xyz.shape[0], obj_xyz.shape[1]
        b = self._batch(obj_xyz, obj_yaw, None)
        b["_obj_yaw"] = obj_yaw
        if temb is not None:
            b["task_emb"] = temb
        mode = torch.ones(N, dtype=torch.long, device=dev)
        if getattr(self.models[0], "watch", False):
            return self._plan_all_watch(b, mode, paths, K, steps, seed, contact, top_m, path_members, excl)
        NI = N * I
        mv = torch.arange(I, device=dev).repeat(N)                               # (N*I,): object index of each (env, object) pair
        rep = lambda E: {k: (v.repeat_interleave(I, 0) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == N else v) for k, v in E.items()}
        Es, pj, pdl, prog = [], [], [], []
        for m in self.models:
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]
            ml, _ = m.pointers(E, torch.zeros(N, dtype=torch.long, device=dev))
            sl = m.stop_logit(E).float()[:, None] if getattr(m, "use_stop", False) else torch.full((N, 1), -1e4, device=dev)
            pj.append(F.softmax(torch.cat([ml.float()[:, :I], sl], -1), -1))
            prog.append(torch.sigmoid(m.p_vid(E["cls"][:, 0]).float().squeeze(-1)))
            Es.append(E)
            if getattr(m, "endref", False):
                continue
            Er = rep(E)
            _, tl = m.pointers(Er, mv)
            pdl.append(F.softmax(torch.cat([tl[:, :I], tl[:, 24:25]], -1).float(), -1))
        if getattr(self.models[0], "endref", False):
            return self._plan_all_endref(b, mode, Es, pj, prog, mv, rep, paths, K, steps, seed, contact, PS, top_m, path_members, excl)
        P = torch.stack(pj).mean(0); Pd = torch.stack(pdl).mean(0)                # (N, I+1), (N*I, I+1)
        d = Pd.argmax(-1); tg = torch.where(d < I, d, torch.full_like(d, 24))
        out = dict(p_now=P[:, :I].cpu(), p_stop=P[:, I].cpu(), progress=torch.stack(prog).mean(0).cpu(),
                   p_dest=Pd.view(N, I, I + 1).cpu(), dest=torch.where(d < I, d, torch.full_like(d, -1)).view(N, I).cpu(),
                   dest_conf=Pd.max(-1).values.view(N, I).cpu())
        geo = b["obj_geo"]                                                         # (N, MAXO, 8): x, y, top, ...
        org = geo.repeat_interleave(I, 0)[torch.arange(NI, device=dev), mv, :3]      # each pair's mover (top centre)
        tgeo = geo.repeat_interleave(I, 0)[torch.arange(NI, device=dev), d.clamp(max=I - 1), :3]
        H = torch.as_tensor(self.Z["extents"][:, 2], dtype=torch.float32, device=dev)[mv]
        ee, ys, ps, cs = [], [], [], []
        gen = torch.random.fork_rng(devices=[] if dev == "cpu" else [torch.device(dev)]) if seed is not None else None
        if gen is not None:
            gen.__enter__(); torch.manual_seed(int(seed))
        try:
            # which (env, object) pairs get a path: all, or each env's top_m by p_now with excluded (taken / blocked) objects at 0
            if paths and top_m is not None and top_m < I:
                pf = P[:, :I] * (~excl.to(dev)).float() if excl is not None else P[:, :I]
                sel = torch.zeros(N, I, dtype=torch.bool, device=dev)
                sel[torch.arange(N, device=dev)[:, None], pf.topk(top_m, -1).indices] = True
                psel = sel.reshape(-1)
            else:
                psel = torch.ones(NI, dtype=torch.bool, device=dev)
            n_pm = len(self.models) if path_members is None else max(1, min(int(path_members), len(self.models)))
            for k_m, (m, E) in enumerate(zip(self.models, Es)):
                Er = rep(E)
                c, _, mdn = m.heads_from(Er, mv, tg); z = m.end_disp(c)
                k1 = mdn["logit"].float().argmax(-1); k2 = z["logit"].float().argmax(-1)
                off = 0.1 * mdn["mu"].float()[torch.arange(NI, device=dev), k1]
                disp = z["mu"].float()[torch.arange(NI, device=dev), k2] * PS
                ee.append(torch.where((d < I)[:, None], tgeo + off, org + disp))
                if getattr(m, "use_yaw", False):
                    y2 = 2 * m.yaw_mode(c.float()); ys.append(torch.stack([y2.cos(), y2.sin()], -1))
                if paths and k_m < n_pm:
                    Es_ = {k: (v[psel] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == NI else v) for k, v in Er.items()}
                    ps.append(m.sample_paths_anchored(Es_, c[psel], K=K, steps=steps)[..., :3].float())
                    cs.append((Es_, c[psel]))
        finally:
            if gen is not None:
                gen.__exit__(None, None, None)
        G = torch.stack(ee).mean(0).clone(); G[:, 2] = (G[:, 2] - H).clamp(min=0.0)     # root goal (bottom centre), as in eval_path
        out["end_xyz"] = G.view(N, I, 3).cpu()
        if ys:
            zy = torch.stack(ys).mean(0); ey = 0.5 * torch.atan2(zy[:, 1], zy[:, 0])
            ex = torch.as_tensor(self.Z["extents"], dtype=torch.float32, device=dev)
            cur = obj_yaw + torch.as_tensor(self.Z["yaw_baked"], dtype=torch.float32, device=dev)[None] + \
                torch.where(ex[:, 1] > ex[:, 0], torch.full_like(ex[:, 0], np.pi / 2), torch.zeros_like(ex[:, 0]))[None]
            out["end_yaw"] = ey.view(N, I).cpu()
            dy = (ey.view(N, I) - cur).cpu()
            out["end_yaw_off"] = (dy + np.pi / 2) % np.pi - np.pi / 2                      # wrap to [-pi/2, pi/2) (180 deg symmetric)
        if paths:
            pth_s = torch.cat(ps, 1).median(1).values + org[psel][:, None]             # (n_sel, 32, 3) top-centre path
            pth = torch.full((NI, pth_s.shape[1], 3), float("nan"), device=dev); pth[psel] = pth_s
            top_path = pth.clone()
            pth = F.avg_pool1d(pth.transpose(1, 2), 5, 1, 2, count_include_pad=False).transpose(1, 2)
            pth[..., 2] -= H[:, None]
            lin = torch.linspace(0, 1, pth.shape[1], device=dev)
            pth = pth + lin[None, :, None] * (G - pth[:, -1])[:, None]
            out["path"] = torch.cat([pth, G[:, None]], 1).view(N, I, 33, 3).cpu()
            heads = self._contact_heads() if contact else None
            if heads is not None:
                rel = (top_path[psel] - org[psel][:, None]) / PS
                pc_s = torch.stack([torch.sigmoid(h(Er, c, rel).float()) for h, (Er, c) in zip(heads, cs)]).mean(0)   # (n_sel, 2, 32)
                pc = torch.full((NI, 2, pc_s.shape[-1]), float("nan"), device=dev); pc[psel] = pc_s
                on = torch.nan_to_num(pc, nan=0.0).max(1).values >= 0.5
                last = torch.where(on.any(-1), pc.shape[-1] - 1 - on.flip(-1).float().argmax(-1), torch.full((NI,), pc.shape[-1] - 1, device=dev))
                out["contact"] = pc.view(N, I, 2, -1).cpu(); out["off_step"] = last.view(N, I).cpu()
                out["off_xyz"] = pth[torch.arange(NI, device=dev), last].view(N, I, 3).cpu()
                out["has_contact"] = on.any(-1).view(N, I).cpu()
        return out

    def _plan_all_watch(self, b, mode, paths, K, steps, seed, contact, top_m, path_members, excl):
        """plan_all for watch models: every field comes from watch_readout, with the same names and shapes as plan_all.
        p_now = the T2 read-out; end_xyz = T1 goal (root), where the path ends (watch_path); contact = T5 along the path's time (both
        hands); dest = the argmax T1 reference as an object (-1: the table or its own start); progress = NaN (no progress head)."""
        dev, I = self.dev, self.I
        N = b["obj_geo"].shape[0]
        Es = []
        for m in self.models:
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]; Es.append(E)
        n_pm = len(self.models) if path_members is None else max(1, min(int(path_members), len(self.models)))
        gen = torch.random.fork_rng(devices=[] if dev == "cpu" else [torch.device(dev)]) if seed is not None else None
        if gen is not None:
            gen.__enter__(); torch.manual_seed(int(seed))
        try:
            R = watch_readout(self.models, Es, paths=False)
            if paths:
                R["traj"] = watch_readout(self.models[:n_pm], Es[:n_pm], paths=True, K=K, steps=steps)["traj"]
        finally:
            if gen is not None:
                gen.__exit__(None, None, None)
        kind = b["obj_kind"]; ar = torch.arange(N, device=dev)
        mv = torch.arange(I, device=dev)[None].expand(N, I)
        j = R["j"][:, :I]
        is_obj = (torch.gather(kind, 1, j) == 0) & (j != mv)
        tab = (kind == 1).float().argmax(-1)
        a = R["a"][:, :I]
        out = dict(p_now=R["p_now"][:, :I].cpu(), p_stop=R["p_stop"].cpu(), stop=R["stop"].cpu(), stop_z=R["z"].cpu(),
                   progress=torch.full((N,), float("nan")),
                   p_dest=torch.cat([a[..., :I], a[ar[:, None], torch.arange(I, device=dev)[None], tab[:, None].expand(N, I)][..., None]], -1).cpu(),
                   dest=torch.where(is_obj, j, torch.full_like(j, -1)).cpu(), dest_conf=a.max(-1).values.cpu(), end_ref=j.cpu(),
                   end_ref_entropy=R["entropy"][:, :I].cpu())
        H = torch.as_tensor(self.Z["extents"][:, 2], dtype=torch.float32, device=dev)[None].expand(N, I)
        goal = R["goal"][:, :I].clone(); goal[..., 2] = torch.maximum(goal[..., 2], H)          # not below the table (numeric guard)
        G = goal.clone(); G[..., 2] = G[..., 2] - H
        out["end_xyz"] = G.cpu()
        ex = torch.as_tensor(self.Z["extents"], dtype=torch.float32, device=dev)
        cur = torch.as_tensor(self.Z["yaw_baked"], dtype=torch.float32, device=dev)[None] + \
            torch.where(ex[:, 1] > ex[:, 0], torch.full_like(ex[:, 0], np.pi / 2), torch.zeros_like(ex[:, 0]))[None]
        if b.get("_obj_yaw") is not None:
            cur = cur + b["_obj_yaw"]
        out["end_yaw"] = R["yaw"][:, :I].cpu(); out["end_yaw_off"] = ((R["yaw"][:, :I] - cur).cpu() + np.pi / 2) % np.pi - np.pi / 2
        if paths:
            st = b["obj_geo"][:, :I, :3].float()
            top = watch_path(st, goal, R["traj"][:, :I])                                       # (N, I, 32, 3)
            if top_m is not None and top_m < I:
                pf = R["p_now"][:, :I] * (~excl.to(dev)).float() if excl is not None else R["p_now"][:, :I]
                sel = torch.zeros(N, I, dtype=torch.bool, device=dev); sel[ar[:, None], pf.topk(top_m, -1).indices] = True
                top = torch.where(sel[..., None, None], top, torch.full_like(top, float("nan")))
            pth = top.clone(); pth[..., 2] -= H[..., None]
            out["path"] = torch.cat([pth, G[:, :, None]], 2).cpu()
            if contact:
                t5 = R["t5"][:, :I]
                pc = F.interpolate(torch.cat([t5[..., :1], t5], -1).reshape(N * I, 1, -1), size=pth.shape[2], mode="linear", align_corners=True)
                pc = pc.reshape(N, I, 1, -1).expand(N, I, 2, -1).clone()
                pc = torch.where(torch.isfinite(pth[..., :1, 0])[:, :, None], pc, torch.full_like(pc, float("nan")))
                on = torch.nan_to_num(pc, nan=0.0).max(2).values >= 0.5
                last = torch.where(on.any(-1), pc.shape[-1] - 1 - on.flip(-1).float().argmax(-1), torch.full((N, I), pc.shape[-1] - 1, device=dev))
                out["contact"] = pc.cpu(); out["off_step"] = last.cpu()
                out["off_xyz"] = torch.gather(pth, 2, last[..., None, None].expand(N, I, 1, 3)).squeeze(2).cpu(); out["has_contact"] = on.any(-1).cpu()
        return out

    def _plan_all_endref(self, b, mode, Es, pj, prog, mv, rep, paths, K, steps, seed, contact, PS, top_m, path_members, excl):
        """plan_all for endref models: the goal is the ensemble's single end (endref_goal) and the path ends there (endref_paths:
        flow residuals conditioned on it plus the line to it); no destination pointer. dest is the reference the end is most likely
        relative to (dest_from_ref); physical support at the end is measured in simulation, not here."""
        dev, I = self.dev, self.I
        N = b["obj_geo"].shape[0]; NI = N * I
        P = torch.stack(pj).mean(0)
        Er_all = [rep(E) for E in Es]
        goal, j, a, ent = endref_goal(self.models, Er_all, mv)
        kind = b["obj_kind"].repeat_interleave(I, 0)
        d = dest_from_ref(j, mv, kind)
        tab = (kind == 1).float().argmax(-1)
        pd_ = torch.cat([a[:, :I], a[torch.arange(NI, device=dev), tab][:, None]], -1)
        out = dict(p_now=P[:, :I].cpu(), p_stop=P[:, I].cpu(), progress=torch.stack(prog).mean(0).cpu(),
                   p_dest=pd_.view(N, I, I + 1).cpu(), dest=d.view(N, I).cpu(), dest_conf=a.max(-1).values.view(N, I).cpu(),
                   end_ref=j.view(N, I).cpu(), end_ref_entropy=ent.view(N, I).cpu())
        H = torch.as_tensor(self.Z["extents"][:, 2], dtype=torch.float32, device=dev)[mv]
        goal = goal.clone(); goal[:, 2] = torch.maximum(goal[:, 2], H)             # not below the table (numeric guard), applied before the paths
        G = goal.clone(); G[:, 2] = G[:, 2] - H                                    # root goal (bottom centre)
        out["end_xyz"] = G.view(N, I, 3).cpu()
        ys = []
        for m, Er in zip(self.models, Er_all):
            if getattr(m, "use_yaw", False):
                y2 = 2 * m.yaw_mode(m.ctx_end(Er, mv, goal).float()); ys.append(torch.stack([y2.cos(), y2.sin()], -1))
        if ys:
            zy = torch.stack(ys).mean(0); ey = 0.5 * torch.atan2(zy[:, 1], zy[:, 0])
            ex = torch.as_tensor(self.Z["extents"], dtype=torch.float32, device=dev)
            obj_yaw = b.get("_obj_yaw")
            cur = torch.as_tensor(self.Z["yaw_baked"], dtype=torch.float32, device=dev)[None] + \
                torch.where(ex[:, 1] > ex[:, 0], torch.full_like(ex[:, 0], np.pi / 2), torch.zeros_like(ex[:, 0]))[None]
            if obj_yaw is not None:
                cur = cur + obj_yaw
            out["end_yaw"] = ey.view(N, I).cpu()
            out["end_yaw_off"] = ((ey.view(N, I) - cur).cpu() + np.pi / 2) % np.pi - np.pi / 2
        if paths:
            if top_m is not None and top_m < I:
                pf = P[:, :I] * (~excl.to(dev)).float() if excl is not None else P[:, :I]
                sel = torch.zeros(N, I, dtype=torch.bool, device=dev)
                sel[torch.arange(N, device=dev)[:, None], pf.topk(top_m, -1).indices] = True
                psel = sel.reshape(-1)
            else:
                psel = torch.ones(NI, dtype=torch.bool, device=dev)
            n_pm = len(self.models) if path_members is None else max(1, min(int(path_members), len(self.models)))
            sub = lambda E: {k: (v[psel] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == NI else v) for k, v in E.items()}   # noqa: E731
            gen = torch.random.fork_rng(devices=[] if dev == "cpu" else [torch.device(dev)]) if seed is not None else None
            if gen is not None:
                gen.__enter__(); torch.manual_seed(int(seed))
            try:
                tp, cs = endref_paths(self.models[:n_pm], [sub(E) for E in Er_all[:n_pm]], mv[psel], goal[psel], K=K, steps=steps)
            finally:
                if gen is not None:
                    gen.__exit__(None, None, None)
            top_path = torch.full((NI, tp.shape[1], 3), float("nan"), device=dev); top_path[psel] = tp
            pth = top_path.clone(); pth[..., 2] -= H[:, None]
            out["path"] = torch.cat([pth, G[:, None]], 1).view(N, I, 33, 3).cpu()     # the last point is the goal
            heads = self._contact_heads() if contact else None
            if heads is not None:
                org = b["obj_geo"].repeat_interleave(I, 0)[torch.arange(NI, device=dev), mv, :3]
                rel = (top_path[psel] - org[psel][:, None]) / PS
                pc_s = torch.stack([torch.sigmoid(h(E_, c_, rel).float()) for h, (E_, c_) in zip(heads, cs)]).mean(0)
                pc = torch.full((NI, 2, pc_s.shape[-1]), float("nan"), device=dev); pc[psel] = pc_s
                on = torch.nan_to_num(pc, nan=0.0).max(1).values >= 0.5
                last = torch.where(on.any(-1), pc.shape[-1] - 1 - on.flip(-1).float().argmax(-1), torch.full((NI,), pc.shape[-1] - 1, device=dev))
                out["contact"] = pc.view(N, I, 2, -1).cpu(); out["off_step"] = last.view(N, I).cpu()
                out["off_xyz"] = pth[torch.arange(NI, device=dev), last].view(N, I, 3).cpu()
                out["has_contact"] = on.any(-1).view(N, I).cpu()
        return out

    def _contact_heads(self):
        """Contact heads stored next to the checkpoints, or None if there are none."""
        if hasattr(self, "_contact"):
            return self._contact
        from cleantables.worldmodel.contact_head import ContactHead
        rd = Path(self.run_dir) if self.run_dir is not None else None
        fs = [rd / f"contact_seed{k}.pt" for k in range(len(self.models))] if rd is not None else []
        if not fs or not all(f.exists() for f in fs):
            return None
        hs = []
        for m, f in zip(self.models, fs):
            h = ContactHead(m.d).to(self.dev); h.load_state_dict(torch.load(f, map_location=self.dev)); hs.append(h.eval())
        self._contact = hs
        return hs


def long_side_yaw(Z, i, yaw_off=0.0):
    """Yaw (rad) of object i's long side in the table frame, as in the bridge's box_feat."""
    ex = Z["extents"][i]
    return float(Z["yaw_baked"][i] + yaw_off + (np.pi / 2 if ex[1] > ex[0] else 0.0))


def wrap_half(a):
    """Wrap an angle difference to [-pi/2, pi/2): 180 deg symmetric shapes need at most a quarter turn."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)
