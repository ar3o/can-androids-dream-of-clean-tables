"""The frozen world model queried live at every reset, on a batch of (random) layouts, in the env's own process ($SIM_PY).

The world model is used exactly as the frozen judge of `cleantables.rl.chain_env.FrozenJudge` (re-derived per reset as
`cleantables.rl.layouts.rejudge_rows`): the watch_notab members with seeds 1 + 2 (FPR_JUDGE_SEEDS=1,2), the scene's own instruction
(the twin's wm_tokens task_emb), 3-D only, no hands (mode 1), at the start frame. The same read-out is used for every object of every
table.

  role  moves (J.mov & ~J.static) / static (movable, the WM keeps it within STATIC_CM) / no_target (reference: table-fixed kinds)
  goal  the judge's frozen T1 goal, top centre, twin table frame (= J.end0)
  path  the WM's own T2 path head (watch_readout(paths=True): the median over members x K flow samples of the displacement at its H
        steps), resampled to n_path points and anchored on the goal by `cleantables.worldmodel.bridge.watch_path` (the head's shape as a
        residual around its own straight line, added to the straight line start -> goal: it starts at the start top and ends exactly at
        the goal). Sampling is seeded (seed 12345, as LivePlanner / plan_all), so a given layout always gets the same path. Non-finite
        head -> the straight segment, flagged path_ok = False.
  order the WM's next-object read-out: movers sorted by p_now (watch_readout p_now, members' mean), -1 padded

Frames / conventions: xyz (n, I, 3) object roots (bottom centre) in the twin table frame = sim_env layout[..., :3] - 2 mm; yaw (n, I)
the judge's yaw convention = scene.json yaw + the random layout's extra world-z turn (layouts: yaw0 + dyaw). query_layout() converts
a sim_env layout (n, K, 7: root xyz + 2 mm, quat xyzw) to that.

  from cleantables.demos.wm_live import WMLive
  wm = WMLive(names, device="cuda")                           # names: the env's template (scene) order, as sim_env.build_templates
  out = wm.query_layout(tpl_of_env, layout)                    # or wm.query(tpl_ids, xyz, yaw)
  out["role"], out["goal"], out["path"], out["order"]          # (n, I) int8 1 moves / 2 static / 0 no_target, (n, I, 3), (n, I, K, 3), (n, I)
"""
from __future__ import annotations

from cleantables import CT_DATA
import os
import time

os.environ.setdefault("WM_TWIN_ROOT", CT_DATA + "/wm_eval/wm_v2_final_model/dr_root_clean_relink")

import numpy as np
import torch

JUDGE_SEEDS = (1, 2)
PCA_PATH = CT_DATA + "/wm_eval/wm_v2_world/onearm/wm_tok_pca.pt"          # default: concat of the members
PCA_AVG_PATH = CT_DATA + "/wm_eval/wm_v2_world/onearm/wm_tok_pca_avg.pt"  # the members' mean
ROLE_NO_TARGET, ROLE_MOVES, ROLE_STATIC = 0, 1, 2
ROLE_NAMES = {ROLE_NO_TARGET: "no_target", ROLE_MOVES: "moves", ROLE_STATIC: "static"}


class WMLive:
    def __init__(self, scenes, device="cuda", seeds=JUDGE_SEEDS, n_path=33, path_samples=4, ode_steps=12, seed=12345, chunk=128,
                 templates=None, tok_combine="concat"):
        """scenes: template names in the env's template order. templates: optional sim_env.build_templates(...) dict (for query_layout;
        built from WM_TWIN_ROOT if not given)."""
        from cleantables.worldmodel.physics.live_plan import LivePlanner
        self.scenes = list(scenes)
        self.LP = LivePlanner(self.scenes, device=str(device), seeds=tuple(seeds))
        self.dev = self.LP.dev
        self.n_path, self.K, self.steps, self.seed, self.chunk = int(n_path), int(path_samples), int(ode_steps), int(seed), int(chunk)
        self.Imax = max(B.I for B in self.LP.B)
        self.H = np.zeros((len(self.scenes), self.Imax), np.float32)
        for t, B in enumerate(self.LP.B):
            self.H[t, :B.I] = np.asarray(B.Z["extents"], np.float32)[:, 2]
        self._tpl = templates
        self.tok_combine = tok_combine           # "concat": [seed-1 token, seed-2 token] (members' spaces are not aligned); "mean": their average
        self.pca_path = PCA_PATH if tok_combine == "concat" else PCA_AVG_PATH
        self.timing = []

    # ---------------------------------------------------------------- layouts
    def templates(self):
        if self._tpl is None:
            from cleantables.rl import sim_env as FE
            self._tpl = FE.build_templates(self.scenes, twin_root=os.environ["WM_TWIN_ROOT"])
        return self._tpl

    def layout_to_xyz_yaw(self, tpl_ids, layout):
        """sim_env layout (n, K, 7) root xyz + 2 mm, quat xyzw -> the judge's xyz (n, I, 3), yaw (n, I): yaw = scene yaw + the extra world-z
        turn of the layout's quaternion relative to the template's (the random layouts of cleantables.rl.layouts turn about world z only)."""
        from cleantables.rl.layouts import _yaw_zyx
        T = self.templates()["templates"]
        lay = np.asarray(layout.detach().cpu() if torch.is_tensor(layout) else layout, np.float64)
        tpl_ids = np.asarray(tpl_ids).astype(int); n = len(tpl_ids); I = self.Imax
        xyz = np.nan_to_num(lay[:, :I, :3] - np.array([0, 0, 0.002])).astype(np.float32)
        yaw = np.zeros((n, I), np.float32)
        for r, t in enumerate(tpl_ids):
            tt = T[t]; k = tt["n"]
            l0 = np.asarray(tt["layout"], np.float64)[:k]
            d = _yaw_zyx(lay[r, :k, 3:7]) - _yaw_zyx(l0[:, 3:7])
            yaw[r, :k] = np.array([o["yaw"] for o in tt["objects"]]) + (d + np.pi) % (2 * np.pi) - np.pi
        return xyz, yaw

    def query_layout(self, tpl_ids, layout, **kw):
        xyz, yaw = self.layout_to_xyz_yaw(tpl_ids, layout)
        return self.query(tpl_ids, xyz, yaw, **kw)

    # ---------------------------------------------------------------- the WM
    @torch.no_grad()
    def _readout(self, tpl_ids, xyz, yaw, with_paths=True):
        """the WM's p_now and T2 path head (traj) for every row, vectorised across templates (inputs built as LivePlanner._wm_fast1:
        the cached per-template constants + obj_geo from the poses; mode 1 = 3D, no hands; the twin's own task_emb). Seeded sampling."""
        from cleantables.worldmodel.schema import MAXO
        from cleantables.worldmodel.bridge import watch_readout
        LP = self.LP; TBL = LP._const_tables()
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
        bc["obj_geo"] = geo
        mode = torch.ones(n, dtype=torch.long, device=self.dev)
        Es = []
        for m in LP.models:
            E = m.encode(bc, mode); E["kind"] = bc["obj_kind"]; Es.append(E)
        if self.tok_combine == "concat":                                    # (r, MAXO, members x d) last trunk layer (after its norm)
            tok = torch.cat([E["obj"].float() for E in Es], -1)
        else:                                                               # (r, MAXO, d) the members' mean
            tok = torch.stack([E["obj"].float() for E in Es]).mean(0)
        if not with_paths:
            return dict(tok=tok)
        R = watch_readout(LP.models, Es, paths=True, K=self.K, steps=self.steps)
        return dict(p_now=R["p_now"].float(), traj=R["traj"].float(), goal=R["goal"].float(), start=geo[..., :3], tok=tok)

    # ---------------------------------------------------------------- what the WM sees: per-object tokens
    def _prep(self, tpl_ids, xyz, yaw):
        tpl_ids = np.asarray(tpl_ids.cpu() if torch.is_tensor(tpl_ids) else tpl_ids).astype(int)
        xyz = np.nan_to_num(np.asarray(xyz.detach().cpu() if torch.is_tensor(xyz) else xyz, np.float32))[:, :self.Imax]
        yaw = np.nan_to_num(np.asarray(yaw.detach().cpu() if torch.is_tensor(yaw) else yaw, np.float32))[:, :self.Imax]
        n, I = len(tpl_ids), self.Imax
        if xyz.shape[1] < I:
            xyz = np.concatenate([xyz, np.zeros((n, I - xyz.shape[1], 3), np.float32)], 1)
            yaw = np.concatenate([yaw, np.zeros((n, I - yaw.shape[1]), np.float32)], 1)
        return tpl_ids, xyz, yaw

    def valid_slots(self, tpl_ids):
        return np.arange(self.Imax)[None] < np.array([self.LP.B[t].I for t in np.asarray(tpl_ids).astype(int)])[:, None]

    @torch.no_grad()
    def object_tokens(self, tpl_ids, xyz, yaw):
        """(n, I, D) float32: the frozen WM's per-object output tokens at this start frame (the last trunk layer after its norm, E["obj"],
        of the judge's members = seeds 1 + 2: concatenated, D = 512 (default), or averaged, D = 256 (tok_combine="mean")), from the same
        3D / no-hands / own-instruction forward; 0 for invalid slots.
        query() returns the same array as out["tok"] at no extra cost (it reuses the path read-out's forward)."""
        tpl_ids, xyz, yaw = self._prep(tpl_ids, xyz, yaw)
        I = self.Imax
        parts = [self._readout(tpl_ids[c:c + self.chunk], xyz[c:c + self.chunk], yaw[c:c + self.chunk], with_paths=False)["tok"][:, :I]
                 for c in range(0, len(tpl_ids), self.chunk)]
        tok = torch.cat(parts).cpu().numpy()
        return np.where(self.valid_slots(tpl_ids)[..., None], np.nan_to_num(tok), 0.0).astype(np.float32)

    def pca(self):
        if getattr(self, "_pca", None) is None:
            P = torch.load(self.pca_path, weights_only=False)
            assert P.get("meta", {}).get("combine", "mean") == self.tok_combine, f"{self.pca_path}: PCA fitted for another token combine"
            self._pca = {k: np.asarray(v, np.float32) for k, v in P.items() if k in ("mean", "components", "explained_variance", "explained_variance_ratio")}
        return self._pca

    def project_tokens(self, tok, tpl_ids, whiten=False):
        """tok (n, I, D) -> (n, I, 16) float32: the fixed PCA (wm_tok_pca.pt; _avg for "mean") ((tok - mean) @ components.T); NaN-free, 0 for
        invalid slots. whiten=True: each component divided by its sd on the fit set (unit variance)."""
        P = self.pca()
        z = (np.nan_to_num(np.asarray(tok, np.float32)) - P["mean"]) @ P["components"].T
        if whiten:
            z = z / np.sqrt(P["explained_variance"])
        return np.where(self.valid_slots(tpl_ids)[..., None], np.nan_to_num(z), 0.0).astype(np.float32)

    def tokens16(self, tpl_ids, xyz, yaw, whiten=False):
        """(n, I, 16) the projected WM object tokens (object_tokens + project_tokens), for a call without query()."""
        return self.project_tokens(self.object_tokens(tpl_ids, xyz, yaw), tpl_ids, whiten=whiten)

    @torch.no_grad()
    def query(self, tpl_ids, xyz, yaw, paths=True, tokens=True):
        """tpl_ids (n,) template index per env; xyz (n, I, 3) roots (bottom centre, twin table frame); yaw (n, I) (judge convention).
        -> dict of numpy arrays (I = the most objects of any template; slots >= the template's count are role 0 / NaN):
          role (n, I) int8, moves / static (n, I) bool, start (n, I, 3) top centre, goal (n, I, 3) top centre (NaN unless moves),
          path (n, I, n_path, 3) top centre start -> goal (NaN unless moves), path_t (n_path,), path_ok (n, I) (False = straight fallback),
          p_now (n, I), order (n, I) movers by p_now (-1 padded), sd (n, I, 3) the judge's tolerance (xy sd, z sd), ref (n, I) reference slot,
          judge: the FrozenJudge object (for its met / score rules), t_judge / t_path seconds;
          tokens=True: tok (n, I, d) the WM's per-object tokens (object_tokens; reused from the path forward, so ~free) and, once
          wm_tok_pca.pt exists, tok16 (n, I, 16) (project_tokens); tok_valid (n, I). (paths=False: p_now / order are not computed.)"""
        from cleantables.rl.chain_env import FrozenJudge
        from cleantables.worldmodel.bridge import watch_path
        tpl_ids, xyz, yaw = self._prep(tpl_ids, xyz, yaw)
        n, I = len(tpl_ids), self.Imax
        t0 = time.time()
        J = FrozenJudge(self.LP, tpl_ids, xyz, yaw)                          # the chain env's judge, exactly (seeds of self.LP, own instruction)
        t1 = time.time()
        valid = np.arange(I)[None] < np.array([self.LP.B[t].I for t in tpl_ids])[:, None]
        moves = J.mov & ~J.static & valid
        static = J.mov & J.static & valid
        role = np.where(moves, ROLE_MOVES, np.where(static, ROLE_STATIC, ROLE_NO_TARGET)).astype(np.int8)
        start = xyz.copy(); start[..., 2] += self.H[tpl_ids]
        goal = np.where(moves[..., None], J.end0, np.nan).astype(np.float32)
        out = dict(role=role, moves=moves, static=static, start=start, goal=goal, ref=J.j.copy(),
                   sd=np.concatenate([J.sd, J.sdz[..., None]], -1).astype(np.float32), judge=J, t_judge=t1 - t0)
        u = np.linspace(0, 1, self.n_path, dtype=np.float32)
        out["path_t"] = u
        path = np.full((n, I, self.n_path, 3), np.nan, np.float32); pok = np.zeros((n, I), bool); pn = np.zeros((n, I), np.float32)
        tok = None
        if paths:
            gen = torch.random.fork_rng(devices=[] if self.dev.type == "cpu" else [self.dev])
            gen.__enter__(); torch.manual_seed(self.seed)
            try:
                for c in range(0, n, self.chunk):
                    sl = slice(c, c + self.chunk)
                    R = self._readout(tpl_ids[sl], xyz[sl], yaw[sl])
                    st = torch.as_tensor(start[sl], device=self.dev)
                    g = torch.as_tensor(np.nan_to_num(J.end0[sl]), device=self.dev).float()
                    tr = R["traj"][:, :I]
                    ok = torch.isfinite(tr).flatten(-2).all(-1)
                    pth = watch_path(st, g, torch.nan_to_num(tr), NP=self.n_path)                       # (r, I, NP, 3)
                    line = st[..., None, :] + torch.as_tensor(u, device=self.dev)[:, None] * (g - st)[..., None, :]
                    pth = torch.where(ok[..., None, None], pth, line)
                    path[sl] = pth.cpu().numpy(); pok[sl] = ok.cpu().numpy(); pn[sl] = R["p_now"][:, :I].cpu().numpy()
                    if tokens:
                        tk = R["tok"][:, :I].cpu().numpy()
                        tok = np.zeros((n, I, tk.shape[-1]), np.float32) if tok is None else tok
                        tok[sl] = tk
            finally:
                gen.__exit__(None, None, None)
        path[~moves] = np.nan
        out["path"], out["path_ok"] = path, pok & moves
        pn = np.where(moves, pn, -1.0)
        order = np.argsort(-pn, 1, kind="stable"); order = np.where(np.take_along_axis(pn, order, 1) >= 0, order, -1)
        out["p_now"], out["order"] = np.maximum(pn, 0).astype(np.float32), order.astype(np.int64)
        if tokens:
            vs = self.valid_slots(tpl_ids)
            tok = self.object_tokens(tpl_ids, xyz, yaw) if tok is None else np.where(vs[..., None], np.nan_to_num(tok), 0.0).astype(np.float32)
            out["tok"], out["tok_valid"] = tok, vs
            if os.path.exists(self.pca_path):
                out["tok16"] = self.project_tokens(tok, tpl_ids)
        out["t_path"] = time.time() - t1
        self.timing.append(dict(n=n, judge_s=round(out["t_judge"], 3), path_s=round(out["t_path"], 3)))
        return out


def object_keys(templates):
    """(tpl, slot) -> the object key used by the demo data: f"{slot}:{id}" (as fastpath/frozen_goals_all.json) per template."""
    return [[f'{k}:{o["id"]}' for k, o in enumerate(t["objects"])] for t in templates["templates"]]


def grasp_key(obj):
    """Grasp-cache key of a sim_env template object (build_templates' objects[k]): its USD (absolute, as sim_env spawns it) + its scale
    (None = the USD's own; sim_env spawns scale (s, s, s)). grasps.pt is keyed by this string."""
    s = obj.get("scale")
    return f'{obj["usd"]}|{"none" if s is None else f"{float(s):.6f}"}'


def random_layouts(tpl, n, seed=0, bases_json=CT_DATA + "/wm_eval/wm_v2_world/fastpath/franka_bases_clean.json"):
    """n layouts over the templates (round robin): layouts.sample_table (xy / yaw uniform on the table, footprints apart, reachable by
    an arm of franka_bases_clean.json); a table with no valid draw keeps its template layout.
    -> tpl_ids (n,), layout (n, K, 7)."""
    import json
    from cleantables.rl.layouts import sample_table
    T = tpl["templates"]; Bj = json.load(open(bases_json))["scenes"]
    rng = np.random.default_rng(seed); ids = np.arange(n) % len(T); lay = []
    for t in ids:
        tt = T[t]; l0 = np.asarray(tt["layout"], np.float32); k = tt["n"]
        valid = np.arange(len(l0)) < k; ext = np.nan_to_num(np.asarray(tt["extents"], np.float32))
        box = np.array([tt["table"]["x"][0], tt["table"]["x"][1], tt["table"]["y"][0], tt["table"]["y"][1]], np.float32)
        bs = np.asarray(Bj[tt["name"]]["bases"], np.float32)[:, :3] if tt["name"] in Bj else None
        res = sample_table(rng, l0, valid, ext, box, bs) if bs is not None else None
        lay.append(res[0] if res is not None else l0)
    return ids, np.stack(lay)


def fit_pca(n_layouts=512, n_comp=16, seed=0, combine="concat", out=None):
    """Fixed token projection: PCA of the per-object tokens (valid slots) of n_layouts random layouts of the train pool (scope_clean)."""
    import json
    from cleantables.rl import sim_env as FE
    names = json.load(open(CT_DATA + "/wm_eval/wm_v2_world/fastpath/scope_clean.json"))["train"]
    tpl = FE.build_templates(names, twin_root=os.environ["WM_TWIN_ROOT"])
    wm = WMLive(names, templates=tpl, tok_combine=combine)
    out = out or wm.pca_path
    ids, lay = random_layouts(tpl, n_layouts, seed=seed)
    xyz, yaw = wm.layout_to_xyz_yaw(ids, lay)
    tok = wm.object_tokens(ids, xyz, yaw)
    X = tok[wm.valid_slots(ids)].astype(np.float64)
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = S ** 2 / (len(X) - 1)
    P = dict(mean=torch.as_tensor(mu, dtype=torch.float32), components=torch.as_tensor(Vt[:n_comp], dtype=torch.float32),
             explained_variance=torch.as_tensor(ev[:n_comp], dtype=torch.float32),
             explained_variance_ratio=torch.as_tensor(ev[:n_comp] / ev.sum(), dtype=torch.float32),
             meta=dict(n_layouts=n_layouts, n_tokens=len(X), d=X.shape[1], seed=seed, seeds=JUDGE_SEEDS, twin_root=os.environ["WM_TWIN_ROOT"],
                       pool="scope_clean train", combine=combine,
                       tokens="E['obj'] (last trunk layer after norm), seeds 1 + 2 " + ("concatenated [s1, s2]" if combine == "concat" else "averaged") + ", 3D, no hands, own instruction",
                       layouts="fp_layout.sample_table random layouts, franka_bases_clean.json", created=time.strftime("%Y-%m-%d %H:%M")))
    torch.save(P, out)
    return P, wm, tpl


if __name__ == "__main__":
    import sys
    if "--fit_pca" in sys.argv:
        P, wm_, _ = fit_pca(combine="mean" if "--mean" in sys.argv else "concat")
        print("saved", wm_.pca_path, P["meta"], "explained variance ratio (16):", P["explained_variance_ratio"].numpy().round(3), "sum",
              float(P["explained_variance_ratio"].sum()))
