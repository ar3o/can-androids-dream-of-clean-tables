"""Live world model and path planner for robot training: both run inside the simulator and re-plan from the actual table state
whenever a hand lets go of an object. The trigger is the robot's own action boundary, so it is task-agnostic. One batched call per
env step for every env that needs a new plan:
  plan(tpl_ids (n,), xyz (n, I, 3) twin-frame roots, yaw (n, I) yaw offsets, placed (n, I) bool, top_k=2) ->
    obj (n, k) object index (-1 none), p_now (n, k), stop (n,), goal (n, k, 3) root, dyaw (n, k), path (n, k, 33, 3), contact (n, k, 2, 32)
  choice  the frozen world-model ensemble (default run watch_notab, 3 seeds, sd_scale): its top_k not-yet-placed objects by p_now
          and its STOP read-out (as the plan pool: plan_all K 4, steps 12, seed 12345)
  path    global_plan (over / around candidates, exact own-shape cost including the Franka gripper's points, set-down rule, band
          2 cm, route_opt polish), batched on the GPU across all n x k carries; exact start (xyz) and end (the world model's T1 goal)
  contact held on every segment but the last (set down at the goal, then release)
World-model calls are batched per template (plan_all takes N layouts of one scene); the planner batches across templates."""
from __future__ import annotations

from cleantables import CT_DATA
import os
import time
from pathlib import Path

import numpy as np
import torch

from cleantables.worldmodel.physics import global_plan as GP
from cleantables.worldmodel.physics import route_opt as RO

NP = 33


class LivePlanner:
    def __init__(self, scenes, run_dir=None, seeds=(0, 1, 2), device="cuda", K=4, steps=12, seed=12345, polish_steps=None, band=0.02, hand="gripper"):
        """scenes: list of scene names (template order); the twin root is WM_TWIN_ROOT."""
        from cleantables.worldmodel.bridge import WMBridge2, load_models
        from cleantables.worldmodel.physics.geo_cost import MeshCost
        os.environ.setdefault("FP_HAND", hand)
        RO.HAND, RO.PLACE, RO.BAND = 1, 1, float(band)
        self.dev = torch.device(device)
        rd = Path(run_dir or CT_DATA + "/wm_eval/wm_v2_world/runs_final/watch_notab")
        self.models = load_models(rd, seeds=tuple(seeds), fold=-1, device=str(self.dev))
        self.B = [WMBridge2(s, None, models=self.models, device=str(self.dev)) for s in scenes]
        self.K, self.steps, self.seed = int(K), int(steps), int(seed)
        self.polish = int(os.environ.get("GP_POLISH", "0")) if polish_steps is None else int(polish_steps)   # default for live use: no polish (speed)
        self.ts = RO.TorchSDF(MeshCost(w_drag=0.0), self.dev)
        self.wm_paths = os.environ.get("LIVE_WM_PATHS", "0") == "1"           # add the world model's own path as a candidate (costly; rarely chosen, 13 / 341)
        self.chunk = int(os.environ.get("LIVE_CHUNK", "32"))
        self.batch_wm = os.environ.get("LIVE_WM_BATCH", "1") == "1"         # one encode across templates (else per template)
        self.timing = []

    def _const_tables(self):
        """Every template's position-independent world-model inputs, built once and stacked on the GPU (for speed): per key (T, ...)
        from B._batch at a zero layout; plus the geometry constants for the vectorised obj_geo (long / short side, height, baked yaw + the
        long-side turn, the table token row, the object count)."""
        if getattr(self, "_TBL", None) is not None:
            return self._TBL
        from cleantables.worldmodel.schema import MAXO
        per, geo_c = [], []
        for B in self.B:
            I = B.I; b = B._batch(torch.zeros(1, I, 3, device=self.dev), torch.zeros(1, I, device=self.dev), None)
            temb = B._task_emb(None, 1)
            if temb is not None:
                b["task_emb"] = temb
            per.append({k: v[0] for k, v in b.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 and not k.startswith("_")})
            ex = torch.as_tensor(B.Z["extents"], dtype=torch.float32, device=self.dev)
            L, W, H = ex[:, 0].max(ex[:, 1]), ex[:, 0].min(ex[:, 1]), ex[:, 2]
            yl = torch.as_tensor(B.Z["yaw_baked"], dtype=torch.float32, device=self.dev) + torch.where(ex[:, 1] > ex[:, 0], torch.full_like(L, np.pi / 2), torch.zeros_like(L))
            g = torch.zeros(MAXO, 4, device=self.dev); g[:I, 0], g[:I, 1], g[:I, 2], g[:I, 3] = L, W, H, yl
            geo_c.append((g, torch.as_tensor(B.Z["table_geo"], dtype=torch.float32, device=self.dev), I))
        keys = [k for k in per[0] if all(k in d and d[k].shape == per[0][k].shape for d in per)]
        TBL = {k: torch.stack([d[k] for d in per]) for k in keys}
        TBL["_gc"] = torch.stack([g for g, _, _ in geo_c]); TBL["_tg"] = torch.stack([t for _, t, _ in geo_c])
        TBL["_I"] = torch.tensor([i for _, _, i in geo_c], device=self.dev)
        self._TBL = TBL
        return TBL

    @torch.no_grad()
    def _wm_fast(self, tpl_ids, xyz, yaw, chunk=512, temb=None):
        """Chunked over rows (memory: t1_mix is rows x 24 x 24 x d)."""
        tpl_ids = np.asarray(tpl_ids)
        if len(tpl_ids) <= chunk:
            return self._wm_fast1(tpl_ids, xyz, yaw, temb=temb)
        parts = [self._wm_fast1(tpl_ids[i:i + chunk], np.asarray(xyz)[i:i + chunk], np.asarray(yaw)[i:i + chunk],
                                temb=None if temb is None else temb[i:i + chunk]) for i in range(0, len(tpl_ids), chunk)]
        return {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}

    @torch.no_grad()
    def _wm_fast1(self, tpl_ids, xyz, yaw, temb=None):
        """Same read-outs as _wm_batched (3D-only), vectorised over rows: the constants gathered from _const_tables, obj_geo computed for
        every row at once, one encode per member, one host copy. -> numpy arrays: p_now (n, MAXO), stop (n,), G (n, MAXO, 3) root goals
        (height guard), dy (n, MAXO)."""
        from cleantables.worldmodel.schema import MAXO
        from cleantables.worldmodel.bridge import watch_readout
        TBL = self._const_tables()
        tp = torch.as_tensor(np.asarray(tpl_ids), device=self.dev).long(); n = len(tp)
        X = torch.zeros(n, MAXO, 3, device=self.dev); Yw = torch.zeros(n, MAXO, device=self.dev)
        Ii = min(xyz.shape[1], MAXO)
        X[:, :Ii] = torch.as_tensor(np.asarray(xyz)[:, :Ii], dtype=torch.float32, device=self.dev)
        Yw[:, :Ii] = torch.as_tensor(np.asarray(yaw)[:, :Ii], dtype=torch.float32, device=self.dev)
        gc = TBL["_gc"][tp]; I = TBL["_I"][tp]
        slot = torch.arange(MAXO, device=self.dev)[None]
        isobj = slot < I[:, None]
        L, W, H, yl = gc[..., 0], gc[..., 1], gc[..., 2], gc[..., 3] + Yw
        top = X[..., 2] + H
        geo = torch.stack([X[..., 0], X[..., 1], top, L, W, top, torch.sin(2 * yl), torch.cos(2 * yl)], -1) * isobj[..., None]
        istab = slot == I[:, None]
        geo = torch.where(istab[..., None], TBL["_tg"][tp][:, None].expand(-1, MAXO, -1), geo)
        bc = {k: v[tp] for k, v in TBL.items() if not k.startswith("_")}
        if temb is not None and "task_emb" in bc:                         # optional per-row instruction embedding
            bc["task_emb"] = torch.as_tensor(temb, dtype=bc["task_emb"].dtype, device=bc["task_emb"].device)
        bc["obj_geo"] = geo
        mode = torch.ones(n, dtype=torch.long, device=self.dev)
        Es = []
        for m in self.models:
            E = m.encode(bc, mode); E["kind"] = bc["obj_kind"]; Es.append(E)
        R = watch_readout(self.models, Es, paths=False)
        goal = R["goal"].float().clone(); goal[..., 2] = torch.maximum(goal[..., 2], H); G = goal.clone(); G[..., 2] = G[..., 2] - H
        dy = ((R["yaw"].float() - yl) + np.pi / 2) % np.pi - np.pi / 2
        return dict(p_now=R["p_now"].double().cpu().numpy(), stop=R["stop"].cpu().numpy().astype(bool), G=G.cpu().numpy(), dy=dy.cpu().numpy(),
                    I=I.cpu().numpy())

    @torch.no_grad()
    def _wm_batched(self, tpl_ids, xyz, yaw, placed, video=None):
        """World-model read-outs for every row in one encode per member, across templates (the tokens are padded to the same slot count):
        the same fields and formulas as WMBridge2.plan_all(paths=False) for watch models (p_now, stop, end_xyz root with the height
        guard, end_yaw_off). -> per row: (p_now (I,), stop, end (I, 3), dyaw (I,))."""
        from cleantables.worldmodel.bridge import watch_readout
        groups, bs = [], []
        for t in np.unique(tpl_ids):
            rows = np.flatnonzero(tpl_ids == t); B = self.B[t]; I = B.I
            P = torch.as_tensor(xyz[rows, :I], device=self.dev); Y = torch.as_tensor(yaw[rows, :I], device=self.dev)
            b = B._batch(P, Y, None)
            temb = B._task_emb(None, len(rows))
            if temb is not None:
                b["task_emb"] = temb
            groups.append((t, rows)); bs.append(b)
        keys = [k for k, v in bs[0].items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == len(groups[0][1]) and not k.startswith("_")]
        bc = {k: torch.cat([b[k].expand(len(g[1]), *b[k].shape[1:]) if b[k].shape[0] != len(g[1]) else b[k] for b, g in zip(bs, groups)], 0) for k in keys}
        n = sum(len(g[1]) for g in groups); mode = torch.ones(n, dtype=torch.long, device=self.dev)
        if video is not None:                                              # live video (video_live.LiveVideo.features), per row in plan()'s order
            od = torch.as_tensor(np.concatenate([g[1] for g in groups]), device=self.dev).long()
            ok = torch.as_tensor(video["vid_ok"], device=self.dev).bool()[od]
            bc["vj_grid"] = torch.as_tensor(video["vj_grid"], device=self.dev)[od].float()
            bc["dn_grid"] = torch.as_tensor(video["dn_grid"], device=self.dev)[od].float()
            uv = torch.zeros(n, bc["uv"].shape[1], 2, device=self.dev); uk = torch.zeros(n, bc["uv"].shape[1], dtype=torch.bool, device=self.dev)
            vu, vk = torch.as_tensor(video["uv"], device=self.dev)[od], torch.as_tensor(video["uv_ok"], device=self.dev)[od]
            m_ = min(uv.shape[1], vu.shape[1]); uv[:, :m_] = vu[:, :m_].float(); uk[:, :m_] = vk[:, :m_].bool()
            bc["uv"], bc["uv_ok"], bc["vid_ok"] = uv, uk, ok
            mode = torch.where(ok, torch.zeros_like(mode), mode)            # mode 0 = video + 3D (training's robot-mode render samples)
        Es = []
        for m in self.models:
            E = m.encode(bc, mode); E["kind"] = bc["obj_kind"]; Es.append(E)
        R = watch_readout(self.models, Es, paths=False)
        res, q = {}, 0
        for (t, rows), b in zip(groups, bs):
            B = self.B[t]; I = B.I; ex = torch.as_tensor(B.Z["extents"], dtype=torch.float32, device=self.dev)
            H = ex[:, 2]
            for r in rows:
                goal = R["goal"][q, :I].clone(); goal[:, 2] = torch.maximum(goal[:, 2], H); G = goal.clone(); G[:, 2] = G[:, 2] - H
                cur = torch.as_tensor(B.Z["yaw_baked"], dtype=torch.float32, device=self.dev) + torch.where(ex[:, 1] > ex[:, 0], torch.full_like(H, np.pi / 2), torch.zeros_like(H))
                cur = cur + torch.as_tensor(yaw[r, :I], device=self.dev)
                dy = ((R["yaw"][q, :I] - cur) + np.pi / 2) % np.pi - np.pi / 2
                res[int(r)] = (R["p_now"][q, :I].double().cpu().numpy(), bool(R["stop"][q]), G.cpu().numpy(), dy.cpu().numpy())
                q += 1
        return res

    @torch.no_grad()
    def plan(self, tpl_ids, xyz, yaw, placed, top_k=2, want=None, video=None, gyaw=None, goal_xy=None, wmo=None):
        """want (n,) optional: an object index per row to plan first (slot j = 0) whatever its rank (the robot keeps its current object
        and refreshes its path from the actual pose); -1 = the world model's best. The remaining slots are filled by rank.
        video: optional LiveVideo.features(...) for these n rows (same order); None = 3D-only.
        gyaw: optional (n, I) grasp yaw per object (baked frame, rad; NaN = the default short-axis grasp): the hand the planner keeps clear.
        wmo: optional output of _wm_fast(tpl_ids, xyz, yaw) for exactly these rows, already computed by the caller (saves a call)."""
        gy = None if gyaw is None else np.asarray(gyaw.cpu() if torch.is_tensor(gyaw) else gyaw, np.float64)
        gxy = None if goal_xy is None else np.asarray(goal_xy, np.float64)   # (n, I, 2) optional goal xy override (e.g. a judge's
                                                                         # target); the world model's goal height (height guard) is kept
        t0 = time.time()
        tpl_ids = np.asarray(tpl_ids.cpu() if torch.is_tensor(tpl_ids) else tpl_ids).astype(int)
        xyz = np.asarray(xyz.detach().cpu() if torch.is_tensor(xyz) else xyz, np.float32)
        yaw = np.asarray(yaw.detach().cpu() if torch.is_tensor(yaw) else yaw, np.float32)
        placed = np.asarray(placed.detach().cpu() if torch.is_tensor(placed) else placed, bool)
        n, Imax = xyz.shape[:2]; k = int(top_k)
        out = dict(obj=np.full((n, k), -1, np.int64), p_now=np.zeros((n, k), np.float32), stop=np.zeros(n, bool),
                   goal=np.full((n, k, 3), np.nan, np.float32), dyaw=np.zeros((n, k), np.float32),
                   path=np.full((n, k, NP, 3), np.nan, np.float32), contact=np.zeros((n, k, 2, NP - 1), np.float32))
        out["p_now_all"] = np.zeros((n, Imax), np.float32)
        items, where = [], []
        if self.batch_wm and not self.wm_paths:                            # one encode per member across every template
            if video is None and os.environ.get("LIVE_WM_FAST", "1") == "1":   # vectorised, constants cached on the GPU
                F_ = self._wm_fast(tpl_ids, xyz, yaw) if wmo is None else wmo
                W_ = {int(r): (F_["p_now"][r, :F_["I"][r]], bool(F_["stop"][r]), F_["G"][r, :F_["I"][r]], F_["dy"][r, :F_["I"][r]]) for r in range(len(tpl_ids))}
            else:
                W_ = self._wm_batched(tpl_ids, xyz, yaw, placed, video=video)
            wv = None if want is None else np.asarray(want.cpu() if torch.is_tensor(want) else want).astype(int)
            for r in range(n):
                t = tpl_ids[r]; B = self.B[t]; I = B.I; pn, stp, end_r, dy_r = W_[r]
                pn = np.where(placed[r, :I], -1.0, np.nan_to_num(pn, nan=-1.0)); out["stop"][r] = stp
                out["p_now_all"][r, :I] = np.maximum(pn, 0)
                u = np.linspace(0, 1, NP)[:, None]
                order = list(np.argsort(-pn))
                if wv is not None and 0 <= wv[r] < I and not placed[r, wv[r]]:
                    order = [wv[r]] + [x for x in order if x != wv[r]]
                for j, a in enumerate(order[:k]):
                    if (pn[a] <= 1e-6 and not (j == 0 and wv is not None and a == wv[r])) or not np.isfinite(end_r[a]).all():
                        continue
                    if gxy is not None and np.isfinite(gxy[r, a]).all():
                        end_r = end_r.copy(); end_r[a, :2] = gxy[r, a]
                    out["obj"][r, j] = a; out["p_now"][r, j] = max(pn[a], 0.0); out["goal"][r, j] = end_r[a]; out["dyaw"][r, j] = dy_r[a]
                    w = xyz[r, a][None].astype(np.float64) + u * (end_r[a] - xyz[r, a])[None]
                    items.append(dict(S=B.S, xyz=xyz[r, :I].astype(np.float64), yaw=yaw[r, :I].astype(np.float64), a=int(a),
                                      plans=w[None].astype(np.float32), dyaws=np.array([dy_r[a]], np.float32),
                                      gyaw=None if gy is None or not np.isfinite(gy[r, a]) else float(gy[r, a])))
                    where.append((r, j))
        for t in (np.unique(tpl_ids) if not (self.batch_wm and not self.wm_paths) else []):   # world model batched per template
            rows = np.flatnonzero(tpl_ids == t); B = self.B[t]; I = B.I
            P = torch.as_tensor(xyz[rows, :I]); Y = torch.as_tensor(yaw[rows, :I]); tk = torch.as_tensor(placed[rows, :I])
            o = B.plan_all(P, Y, taken=tk, paths=self.wm_paths, K=self.K, steps=self.steps, seed=self.seed, contact=False)
            pn = o["p_now"].double().numpy(); pn = np.where(placed[rows, :I], -1.0, np.nan_to_num(pn, nan=-1.0))
            stop = o["stop"].numpy().astype(bool) if "stop" in o else np.zeros(len(rows), bool)
            end = o["end_xyz"].numpy(); dy = np.nan_to_num(o["end_yaw_off"].numpy()) if "end_yaw_off" in o else np.zeros((len(rows), I))
            if "path" in o:
                wm = o["path"].numpy()
            else:                                                          # no world-model path sampled (LIVE_WM_PATHS=0): the straight line stands
                u = np.linspace(0, 1, NP)[None, None, :, None]                # in as the first candidate (the planner adds its own routes)
                wm = xyz[rows, :I][:, :, None] + u * (end - xyz[rows, :I])[:, :, None]
            for q, r in enumerate(rows):
                out["stop"][r] = stop[q]; out["p_now_all"][r, :I] = np.maximum(pn[q], 0)
                order = np.argsort(-pn[q])[:k]
                for j, a in enumerate(order):
                    if pn[q, a] <= 1e-6 or not np.isfinite(end[q, a]).all():
                        continue
                    out["obj"][r, j] = a; out["p_now"][r, j] = pn[q, a]; out["goal"][r, j] = end[q, a]; out["dyaw"][r, j] = dy[q, a]
                    w = np.array(wm[q, a], np.float64); w[0] = xyz[r, a]; w[-1] = end[q, a]
                    items.append(dict(S=B.S, xyz=xyz[r, :I].astype(np.float64), yaw=yaw[r, :I].astype(np.float64), a=int(a),
                                      plans=w[None].astype(np.float32), dyaws=np.array([dy[q, a]], np.float32)))
                    where.append((r, j))
        t1 = time.time()
        if items:                                                          # the planner, batched across every carry on the GPU
            with torch.enable_grad():                                      # route_opt's polish takes gradient steps
                pp, _ = GP.plan_paths(self.ts, items, polish_steps=self.polish, chunk=self.chunk)
            for (r, j), p, it in zip(where, pp, items):
                p = np.array(p, np.float32); p[0] = it["xyz"][it["a"]]; p[-1] = out["goal"][r, j]
                out["path"][r, j] = p; out["contact"][r, j] = 1.0; out["contact"][r, j, :, -1] = 0.0
        self.timing.append(dict(n=n, carries=len(items), wm_s=round(t1 - t0, 3), planner_s=round(time.time() - t1, 3)))
        return {kk: torch.as_tensor(v) for kk, v in out.items()}
