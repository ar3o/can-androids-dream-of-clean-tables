"""Held-out evaluation of a world-model run, one command per run.

Model kinds: 'v2' runs (OUT/wm_v2/runs/<cfg>, UWM2 / UWMW), B0 (OUT/wm_v2/runs/B0, base-model class on the same split) and
'v1' base-model runs (runs/uwm/<run>; their training data included the test episodes, so they are for reference only).
Base-model-class models get the md5-hashed instruction they were trained on.
Human tests on the 10 test episodes, for 3D-only (mode 1, as on the robot) and with video (mode 0): next object
(order-free), destination (true mover given) and end point (true destination given) against simple baselines; end yaw
against "keep the start yaw" and "align with the destination's long side"; path ADE against the straight line to the true
end; move-end latent (with video) against "copy the current latent" on centred cosines; destination confidence calibration.
Watch models (UWMW) get their own tests (watch_tests).
Language: held-out paraphrases; step commands (narration of the true next move); composite tables built from test episodes
(the instruction must choose); the mixed twin tables (instr_twin).
  $VLA_PY -m cleantables.worldmodel.evaluate --run full [--run2 ...]   -> OUT/wm_v2/eval/<run>.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.common import OUT, ROOT, RUNS, boot_ci, hashed, jdump, log
from cleantables.worldmodel.bridge import wrap_half

EV = OUT / "wm_v2" / "eval"
RUNS2 = type(OUT)(__import__("os").environ.get("WM_RUNS2", str(OUT / "wm_v2" / "runs")))     # as in cleantables.worldmodel.train (watch runs: wm_v2_world/runs)


# ------------------------------------------------------------------ models
def load_run(run, dev="cpu"):
    """Return (models, kind, cfg); kind is 'v2' or 'v1' (base-model class)."""
    from cleantables.worldmodel.base_model import UWM
    from cleantables.worldmodel.bridge import load_models
    d = RUNS2 / run
    if run != "B0" and d.exists():
        try:
            ms = load_models(d, device=dev)
        except AssertionError:                                            # a validation-only run (fold checkpoints only: --val_fold)
            ms = []
        cfg = json.load(open((sorted(d.glob("fold-1_seed*.json")) or sorted(d.glob("fold*_seed*.json")))[0]))["cfg"]
        return ms, "v2", cfg
    d = d if run == "B0" else (RUNS / run if (RUNS / run).exists() else OUT / run)
    ms = []
    for s in range(3):
        p = d / f"fold-1_seed{s}.pt"
        if p.exists():
            m = UWM(path_resid=True).to(dev); m.load_state_dict(torch.load(p, map_location=dev)); ms.append(m.eval())
    assert ms, f"no models for {run}"
    return ms, "v1", {}


class Batcher:
    def __init__(self, D, dev="cpu"):
        from cleantables.worldmodel.base_train import to_dev
        self.D, self.dev = D, dev
        self.G1 = to_dev(D.S, dev)
        self.hbank = torch.from_numpy(np.stack([hashed(s) for s in D.strings]))
        if D.feats:
            mu = D.VJ[np.unique(D.snap_row[D.train_idx]), 0].astype(np.float32).mean((1, 2))
            self.glob_mean = torch.from_numpy(mu.mean(0))

    def get(self, kind, cfg, idx, text_ids=None, comp_pool=None, seed=0):
        D = self.D
        rng = np.random.default_rng(seed)
        c = dict(cfg) if cfg else {}
        c.setdefault("vid", ("vj", "dn"))
        b = D.make_batch(idx, c, rng, self.dev, text_ids=text_ids, train=False, comp_pool=comp_pool)
        if kind == "v1" or c.get("text") == "hashed":                       # base-class models and the 'no_text' ablation: hashed text, as trained
            b["task_emb"] = self.hbank[b["text_ids"]].to(self.dev)
        if kind == "v1":
            comp = b["comp"]
            vj = self.G1["vj_tok"][torch.as_tensor(idx)].float()
            b["vj_tok"] = torch.where(comp[:, None, None], torch.zeros_like(vj), vj)
            b["vj_ok"] = self.G1["vj_ok"][torch.as_tensor(idx)].bool() & ~comp
        return b


@torch.no_grad()
def forward(ms, b, mode, mv=None, tg=None, K=0):
    from cleantables.worldmodel.base_model import PS
    B = b["obj_geo"].shape[0]
    mode = torch.full((B,), mode, dtype=torch.long)
    mv = b["mover"].clamp(min=0) if mv is None else mv
    tg = b["target"].clamp(min=0) if tg is None else tg
    R = {k: [] for k in ("pm", "pt", "off", "yaw", "path", "fut", "goal", "a")}
    if getattr(ms[0], "endref", False):
        return forward_endref(ms, b, mode, mv, K, R)
    for m in ms:
        E = m.encode(b, mode); E["kind"] = b["obj_kind"]
        ml, tl = m.pointers(E, mv)
        R["pm"].append(F.softmax(ml.float(), -1)); R["pt"].append(F.softmax(tl.float(), -1))
        c, _, mdn = m.heads_from(E, mv, tg)
        tm = mdn["logit"].float().argmax(-1)
        R["off"].append(0.1 * mdn["mu"].float()[torch.arange(B), tm])
        if getattr(m, "use_yaw", False):
            R["yaw"].append(m.yaw_mode(c.float()))
        if K:
            org = b["obj_geo"][torch.arange(B), mv, :3]
            R["path"].append(m.sample_paths_anchored(E, c, K=K).float()[..., :3] + org[:, None, None])
        if getattr(m, "use_fut", False) and "glob_now" in b:
            R["fut"].append(m.fut_pred(E, b))
        # held-out end-state test: the model's own goal with no label given, i.e. its predicted destination's end
        # (TABLE / LEAVES: start + end displacement), as physics.carry_check.eval_path builds the robot's goal
        dp = tl[:, :25].float().argmax(-1); cg, _, mg = m.heads_from(E, mv, torch.where(dp < 24, dp, torch.full_like(dp, 24)))
        og = 0.1 * mg["mu"].float()[torch.arange(B), mg["logit"].float().argmax(-1)]
        zg = m.end_disp(cg); dg = zg["mu"].float()[torch.arange(B), zg["logit"].float().argmax(-1)] * PS
        st = b["obj_geo"][torch.arange(B), mv, :3].float()
        R["goal"].append(torch.where((dp < 24)[:, None], b["obj_geo"][torch.arange(B), dp.clamp(max=23), :3].float() + og, st + dg))
    return {k: (torch.stack(v) if v else None) for k, v in R.items()}


def forward_endref(ms, b, mode, mv, K, R):
    """Forward pass for endref models: no destination pointer; the goal is the ensemble's single end (bridge.endref_goal), and the
    yaw and paths are conditioned on it (the paths end there, as on the robot)."""
    from cleantables.worldmodel.bridge import endref_goal
    B = b["obj_geo"].shape[0]
    Es = []
    for m in ms:
        E = m.encode(b, mode); E["kind"] = b["obj_kind"]; Es.append(E)
        ml, _ = m.pointers(E, mv); R["pm"].append(F.softmax(ml.float(), -1))
        if getattr(m, "use_fut", False) and "glob_now" in b:
            R["fut"].append(m.fut_pred(E, b))
    goal, j, a, _ = endref_goal(ms, Es, mv)
    for m, E in zip(ms, Es):
        R["goal"].append(goal); R["a"].append(a)
        c = m.ctx_end(E, mv, goal)
        if getattr(m, "use_yaw", False):
            R["yaw"].append(m.yaw_mode(c.float()))
        if K:
            org = b["obj_geo"][torch.arange(B), mv, :3]
            R["path"].append(m.plan_paths(E, mv, K=K, end=goal)[0].float()[..., :3] + org[:, None, None])
    return {k: (torch.stack(v) if v else None) for k, v in R.items()}


def circ_mean_half(y):
    """Mean of 180 deg symmetric angles over the ensemble (dim 0)."""
    z = torch.stack([(2 * y).cos(), (2 * y).sin()], -1).mean(0)
    return 0.5 * torch.atan2(z[..., 1], z[..., 0])


def ece(conf, ok, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for a, b_ in zip(bins[:-1], bins[1:]):
        s = (conf >= a) & (conf < b_ if b_ < 1 else conf <= b_)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - ok[s].mean())
    return float(e)


def auroc(score, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos, score)) if 0 < pos.mean() < 1 else float("nan")


# ------------------------------------------------------------------ human tests
def human(run, D, Bt, ms, kind, cfg, modes=(1, 0, "1nh"), K=4):
    """Modes: 1 = 3D-only; 0 = with video; '1nh' = 3D-only without the human hand tokens (as on the robot); '0nh' = real video,
    no hands; '0r' = robot mode with video, i.e. features of robot-hidden twin renders of the same moments, no hands, on the test
    snapshots that have a render; '1r' = 3D-only, no hands, on the same snapshots (the matched comparison for '0r')."""
    res = {}
    for mode_spec in modes:
        ms_ = str(mode_spec)
        mode = 0 if ms_.startswith("0") else 1
        if kind == "v2" and mode == 0 and not cfg.get("vid") and not cfg.get("objg"):
            continue
        if kind == "v1" and ms_ in ("0r", "1r", "0nh"):
            continue
        idx, cfg_m = D.test_idx, cfg
        if ms_ in ("0r", "1r"):
            if __import__("os").environ.get("WM_RENDER_SRC", cfg.get("render_src", "replay")) == "paired":   # encoded paired v0 rows only
                V = D._video_v2(); P0 = [p_ for p_ in V["paired"] if p_["v"] == 0]
                okr = np.zeros(D.n, bool)
                if P0:
                    okr = P0[0]["done"][D.snap_row]
                idx = D.test_idx[okr[D.test_idx]]
            else:
                rr = D._video_v2()["render_row"]
                idx = D.test_idx[rr[D.test_idx] >= 0]
            cfg_m = {**cfg, "video_v2": True, "eval_video": "render",
                     "render_src": __import__("os").environ.get("WM_RENDER_SRC", cfg.get("render_src", "replay"))}
        rows = {}

        def add(k, v, ep, task):
            rows.setdefault(k, []).append((float(v), ep, task))
        for b0 in range(0, len(idx), 64):
            bi = idx[b0:b0 + 64]
            b = Bt.get(kind, cfg_m, bi)
            if ms_ in ("1nh", "0nh", "0r", "1r"):
                b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
            P = forward(ms, b, mode, K=K)
            pm = P["pm"].mean(0); pt = P["pt"].mean(0) if P["pt"] is not None else None
            ms_ = b["mover_set"].bool() & b["obj_mask"].bool()
            mov = b["obj_mask"].bool() & (b["obj_kind"] == 0)
            top = pm.argmax(-1)
            for j in range(len(bi)):
                i = bi[j]; ep, task = D.ep[i], D.task[i]
                mvj, tgt = int(b["mover"][j]), int(b["target"][j])
                if mvj < 0:
                    continue
                add("next_ok", bool(ms_[j, top[j]]), ep, task); add("next_chance", float(ms_[j].sum() / max(1, mov[j].sum())), ep, task)
                g = b["obj_geo"][j]
                po_ = b["path_obj"][j]
                if P["goal"] is not None and torch.isfinite(po_[-1]).all():         # held-out end state: where it really ended, no label
                    add("end_world_cm", 100 * float((P["goal"].mean(0)[j] - po_[-1]).norm()), ep, task)
                    add("end_stay_cm", 100 * float((g[mvj, :3] - po_[-1]).norm()), ep, task)     # baseline: the object stays where it is
                    if P["a"] is not None:
                        aj = P["a"][0, j]; add("a_entropy", float(-(aj * torch.log(aj.clamp(min=1e-9))).sum()), ep, task)
                if tgt >= 0 and pt is not None:
                    add("dest_ok", bool(pt[j].argmax() == tgt), ep, task); add("dest_conf", float(pt[j].max()), ep, task)
                    cand = mov[j].clone(); cand[mvj] = False
                    dd = (g[:, :2] - g[mvj, :2]).norm(dim=-1).masked_fill(~cand, 1e9)
                    add("dest_nearest", bool(int(dd.argmin()) == tgt), ep, task)
                    eo = b["end_off"][j]
                    if torch.isfinite(eo).all() and tgt < 24:
                        off = P["off"].mean(0)[j]
                        add("end_cm", 100 * float((off - eo).norm()), ep, task); add("end_centre_cm", 100 * float(eo.norm()), ep, task)
                if P["yaw"] is not None and bool(b["end_ok"][j]) and torch.isfinite(b["end_pose"][j, 3]) and (tgt >= 0 or pt is None):
                        yt = float(b["end_pose"][j, 3]); yp = float(circ_mean_half(P["yaw"][:, j:j + 1])[0])
                        y0 = float(0.5 * torch.atan2(g[mvj, 6], g[mvj, 7]))
                        yd = float(0.5 * torch.atan2(g[tgt, 6], g[tgt, 7])) if 0 <= tgt < 24 else y0
                        add("yaw_deg", abs(np.degrees(wrap_half(yp - yt))), ep, task); add("yaw_keep_deg", abs(np.degrees(wrap_half(y0 - yt))), ep, task)
                        add("yaw_align_deg", abs(np.degrees(wrap_half(yd - yt))), ep, task)
                po = b["path_obj"][j]
                if P["path"] is not None and torch.isfinite(po).all():
                    med = P["path"][:, j].reshape(-1, 32, 3).median(0).values
                    add("ade_cm", 100 * float((med - po).norm(dim=-1).mean()), ep, task)
                    u = torch.linspace(0, 1, 32)[:, None]
                    line = po[0] + u * (po[-1] - po[0])
                    add("line_cm", 100 * float((line - po).norm(dim=-1).mean()), ep, task)
                if P["fut"] is not None and mode != 1 and bool(b["glob_fut_ok"][j]) and bool(b["vid_ok"][j]):
                    fp = P["fut"].mean(0)[j]; ft = F.normalize(b["glob_fut"][j], dim=0); now = F.normalize(b["glob_now"][j], dim=0)
                    mu = F.normalize(Bt.glob_mean, dim=0)
                    add("fut_cos", float(F.cosine_similarity(fp, ft, dim=0)), ep, task); add("copy_cos", float(F.cosine_similarity(now, ft, dim=0)), ep, task)
                    add("fut_cos_centred", float(F.cosine_similarity(fp - mu, ft - mu, dim=0)), ep, task)
                    add("copy_cos_centred", float(F.cosine_similarity(now - mu, ft - mu, dim=0)), ep, task)
        res[f"mode{mode_spec}" + (__import__("os").environ.get("WM_EVAL_TAG", "") if ms_ in ("0r", "1r") else "")] = summarise(rows)
    return res


@torch.no_grad()
def watch_tests(D, Bt, ms, cfg, modes=("1nh", "0"), K=4):
    """Watch models (UWMW) on the keyframes of the 10 held-out episodes:
      t1_end_cm (objects present at the episode end) vs t1_stay_cm (they stay where they are), split by what the human did
        (evaluation-only split: moved >= 2 cm / stayed); t1_present AUROC
      t2_ade_cm (median of K joint samples, over the 8 steps) vs t2_const_cm (constant position); t5 AUROC
      order: the read-out next object (argmax p_now) vs the contact-derived moves (reference used only here: the first move with
        f_start > k while no contact covers k; order-free = any move still ahead)
      stop read-out (every object within 1 / 2 predicted sd of its T1 end) vs 'no move left in the episode' (contact moves)."""
    from cleantables.worldmodel.base_model import PS
    from cleantables.perception.contact.moves import intervals
    from cleantables.worldmodel.bridge import watch_readout
    vv = D.S.d["video"].astype(str); kk = D.S.d["snap_k"].astype(int); oid_all = D.S.d["obj_id"]
    segs = {s_["id"]: s_ for s_ in json.load(open(ROOT / "data/labels/episodes_all.json"))["segments"] if s_["kind"] == "episode"}
    IV = {}
    res = {}
    for ms_ in modes:
        mode = 0 if ms_.startswith("0") else 1
        rows = {}

        def add(k, v, ep):
            rows.setdefault(k, []).append((float(v), ep, "all"))
        idx = D.test_idx
        for b0 in range(0, len(idx), 64):
            bi = idx[b0:b0 + 64]
            b = Bt.get("v2", cfg, bi)
            if ms_.endswith("nh"):
                b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
                b["obj_contact_ok"] = torch.zeros_like(b["obj_contact_ok"])
                b["obj_contact2_ok"] = torch.zeros_like(b["obj_contact2_ok"])
            B = len(bi); md = torch.full((B,), mode, dtype=torch.long)
            Es = []
            for m in ms:
                E = m.encode(b, md); E["kind"] = b["obj_kind"]; Es.append(E)
            R = watch_readout(ms, Es, paths=True, K=K)
            R2 = watch_readout(ms, Es, paths=False)
            pres = torch.stack([torch.sigmoid(m.present_logit(E)) for m, E in zip(ms, Es)]).mean(0)
            g = b["obj_geo"][..., :3].float(); obj = b["obj_mask"].bool() & (b["obj_kind"] == 0)
            N_ = g.shape[1]
            t2 = b["t2_disp"].view(B, N_, -1, 3).float(); ok2 = b["t2_ok"].bool() & obj[..., None]
            for j in range(B):
                i = int(bi[j]); ep = str(D.ep_base[i])
                ok1 = b["t1_ok"][j].bool() & obj[j]
                for s_ in np.flatnonzero(ok1.numpy()):
                    add("t1_present_p", float(pres[j, s_]), ep); add("t1_present_y", float(b["t1_present"][j, s_]), ep)
                    if bool(b["t1_present"][j, s_]):
                        tgt = b["t1_pos"][j, s_].float(); dh = float((tgt - g[j, s_]).norm())
                        e_, st_ = 100 * float((R["goal"][j, s_] - tgt).norm()), 100 * dh
                        add("t1_end_cm", e_, ep); add("t1_stay_cm", st_, ep)
                        add("t1_end_cm_moved" if dh >= 0.02 else "t1_end_cm_stayed", e_, ep)
                        add("t1_stay_cm_moved" if dh >= 0.02 else "t1_stay_cm_stayed", st_, ep)
                for s_ in np.flatnonzero(obj[j].numpy()):
                    okk = ok2[j, s_]
                    if okk.any():
                        add("t2_ade_cm", 100 * float((R["traj"][j, s_] - t2[j, s_]).norm(dim=-1)[okk].mean()), ep)
                        add("t2_const_cm", 100 * float(t2[j, s_].norm(dim=-1)[okk].mean()), ep)
                    for h in np.flatnonzero((b["t5_ok"][j, s_].bool()).numpy()):
                        add("t5_p", float(R["t5"][j, s_, h]), ep); add("t5_y", float(b["t5"][j, s_, h]), ep)
                # order and stop vs the contact moves (evaluation only)
                sg = segs.get(ep)
                if sg is None:
                    continue
                v = vv[i]; k = int(kk[i])
                if v not in IV:
                    IV[v] = intervals(v) or ([], [], {})
                mv_, cover, _ = IV[v]
                mvs = [e for e in mv_ if 10 * sg["start_s"] <= e["f_start"] and e["f_end"] <= 10 * sg["end_s"]]
                left = [e for e in mvs if e["f_end"] > k]
                add("stop_true", float(not left), ep); add("stop_1sd", float(bool(R2["stop"][j])), ep); add("stop_2sd", float(float(R2["z"][j]) < 2.0), ep)
                held = any(c0 <= k <= c1 for c0, c1 in cover)
                nx = next((e for e in left if e["f_start"] > k), None) if not held else None
                if nx is not None:
                    oid = oid_all[i]; pn = R2["p_now"][j].clone(); pn[~obj[j]] = -1
                    top = int(pn.argmax()); left_ids = {int(e["obj"]) for e in left}
                    add("next_exact", float(int(oid[top]) == int(nx["obj"])), ep); add("next_ok", float(int(oid[top]) in left_ids), ep)
                    add("next_chance", len(left_ids & set(int(x) for x in oid[obj[j].numpy()])) / max(1, int(obj[j].sum())), ep)
        out = summarise(rows)["all"]
        for a_, p_, y_ in (("t1_present_auroc", "t1_present_p", "t1_present_y"), ("t5_auroc", "t5_p", "t5_y")):
            if p_ in rows:
                out[a_] = auroc(np.array([x[0] for x in rows[p_]]), np.array([x[0] for x in rows[y_]]))
                out.pop(p_, None); out.pop(y_, None)
        if "stop_true" in rows:
            st = np.array([x[0] for x in rows["stop_true"]]) > 0.5
            for nm in ("stop_1sd", "stop_2sd"):
                pr = np.array([x[0] for x in rows[nm]]) > 0.5
                out[nm + "_balanced_acc"] = float(0.5 * (pr[st].mean() + (1 - pr[~st]).mean())) if st.any() and (~st).any() else None
                out[nm + "_stops_when_done"] = float(pr[st].mean()) if st.any() else None
                out[nm + "_stops_when_not_done"] = float(pr[~st].mean()) if (~st).any() else None
        res[f"mode{ms_}"] = out
    return res


def summarise(rows):
    out = {}
    for t in ("all", "fruit", "tools", "stack_boxes"):
        o = {}
        for k, v in rows.items():
            vals = np.array([x[0] for x in v]); ep = np.array([x[1] for x in v]); task = np.array([x[2] for x in v])
            s = np.ones(len(vals), bool) if t == "all" else task == t
            if s.any():
                stat = np.median if k.endswith("_cm") or k.endswith("_deg") else np.mean
                o[k] = [round(x, 4) for x in boot_ci(vals[s], ep[s], stat=stat)]
        out[t] = o
    if "dest_conf" in rows:
        conf = np.array([x[0] for x in rows["dest_conf"]]); ok = np.array([x[0] for x in rows["dest_ok"]])
        out["dest_calibration"] = dict(ece=ece(conf, ok), auroc_low_conf_flags_error=auroc(-conf, 1 - ok), n=len(conf))
    return out


# ------------------------------------------------------------------ language tests
def language(run, D, Bt, ms, kind, cfg):
    res = {}
    idx = D.test_idx[D.S.d["mover"][D.test_idx] >= 0]
    # (a) held-out paraphrases: does the prediction stay the same, and correct?
    acc_o, acc_p, tv = [], [], []
    for j in range(4):
        sel = idx[D.test_alt[idx, j] >= 0]
        if not len(sel):
            continue
        for b0 in range(0, len(sel), 64):
            bi = sel[b0:b0 + 64]
            bo = Bt.get(kind, cfg, bi); bp = Bt.get(kind, cfg, bi, text_ids=D.test_alt[bi, j])
            Po, Pp = forward(ms, bo, 1), forward(ms, bp, 1)
            mo, mp = Po["pm"].mean(0), Pp["pm"].mean(0)
            mset = bo["mover_set"].bool()
            acc_o += mset[torch.arange(len(bi)), mo.argmax(-1)].tolist(); acc_p += mset[torch.arange(len(bi)), mp.argmax(-1)].tolist()
            tv += (0.5 * (mo - mp).abs().sum(-1)).tolist()
    res["paraphrase_heldout"] = dict(n=len(tv), next_ok_original=float(np.mean(acc_o)), next_ok_paraphrase=float(np.mean(acc_p)),
                                     tv_change_mean=float(np.mean(tv)))
    # (b) step commands: the narration of the true next move names the object
    sel = idx[D.narr[idx] >= 0]
    okc, oko = [], []
    for b0 in range(0, len(sel), 64):
        bi = sel[b0:b0 + 64]
        bo = Bt.get(kind, cfg, bi); bn = Bt.get(kind, cfg, bi, text_ids=D.narr[bi])
        Po, Pn = forward(ms, bo, 1), forward(ms, bn, 1)
        mv = bo["mover"]
        oko += (Po["pm"].mean(0).argmax(-1) == mv).tolist(); okc += (Pn["pm"].mean(0).argmax(-1) == mv).tolist()
    res["step_command"] = dict(n=len(okc), exact_mover_with_task_instruction=float(np.mean(oko)) if oko else None,
                               exact_mover_with_step_command=float(np.mean(okc)) if okc else None)
    # (c) composite tables from test episodes: objects of another test episode (different instruction) added; they never move
    pick_added, next_ok, epc = [], [], []
    for rep in range(3):
        for b0 in range(0, len(idx), 64):
            bi = idx[b0:b0 + 64]
            b = Bt.get(kind, cfg, bi, comp_pool=D.test_idx, seed=rep)
            if not b["comp"].any():
                continue
            P = forward(ms, b, 1)
            top = P["pm"].mean(0).argmax(-1)
            for j in range(len(bi)):
                if not b["comp"][j]:
                    continue
                pick_added.append(bool(b["donor"][j, top[j]])); next_ok.append(bool(b["mover_set"][j, top[j]])); epc.append(str(D.ep[bi[j]]))
    res["composite_test_tables"] = dict(n=len(pick_added), picks_an_added_object=float(np.mean(pick_added)) if pick_added else None,
                                        next_ok=float(np.mean(next_ok)) if next_ok else None,
                                        ci=dict(picks_an_added_object=boot_ci(pick_added, epc), next_ok=boot_ci(next_ok, epc)) if pick_added else None)
    # (d) the same, but both object sets laid out by one sampler and no human hands: only the instruction tells them apart
    uniq = sorted(set(D.orig.tolist()))
    pa, pa_sw, tvs, ok_, epd = [], [], [], [], []
    for rep in range(3):
        for b0 in range(0, len(idx), 64):
            bi = idx[b0:b0 + 64]
            b = Bt.get(kind, {**cfg, "eval_relay": True}, bi, comp_pool=D.test_idx, seed=10 + rep)
            sel = b["relaid"].bool()
            if not sel.any():
                continue
            b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
            sw = np.array([[u for u in uniq if u != o][0] for o in D.orig[bi]])
            b2 = dict(b); b2["task_emb"] = (Bt.hbank if (kind == "v1" or cfg.get("text") == "hashed") else D.bank)[torch.as_tensor(sw)].to(b["task_emb"].device)
            P1, P2 = forward(ms, b, 1)["pm"].mean(0), forward(ms, b2, 1)["pm"].mean(0)
            t1, t2 = P1.argmax(-1), P2.argmax(-1); ar = torch.arange(len(bi))
            pa += b["donor"][ar, t1][sel].tolist(); pa_sw += b["donor"][ar, t2][sel].tolist(); ok_ += b["mover_set"][ar, t1][sel].tolist()
            tvs += (0.5 * (P1 - P2).abs().sum(-1))[sel].tolist(); epd += [str(e) for e in D.ep[bi][sel.numpy()]]
    res["same_sampler_composites_no_hands"] = dict(n=len(pa), picks_an_added_object=float(np.mean(pa)) if pa else None,
                                                   picks_an_added_object_with_swapped_instruction=float(np.mean(pa_sw)) if pa_sw else None,
                                                   next_ok=float(np.mean(ok_)) if ok_ else None, tv_instruction_swap=float(np.mean(tvs)) if tvs else None,
                                                   ci=dict(picks_an_added_object=boot_ci(pa, epd), picks_an_added_object_with_swapped_instruction=boot_ci(pa_sw, epd),
                                                           tv_instruction_swap=boot_ci(tvs, epd)) if pa else None)
    return res


HELD_NAMES = ("apple", "wrench")


def generalisation(run, D, Bt, ms, kind, cfg, names=HELD_NAMES):
    """Held-out names: test moves of objects whose name is in `names` (absent from the training text of the 'holdout_names' run,
    seen by every other run), 3D-only, no hands. Next object / destination accuracy with the real name and with the name replaced
    by 'object' (control); step commands that name them; held-out paraphrases that mention them."""
    from cleantables.worldmodel.text import embed
    neutral = torch.from_numpy(embed(["object"])[0])
    mv = D.S.d["mover"]
    idx = np.array([i for i in D.test_idx if mv[i] >= 0 and any(h in D.phrase[i, mv[i]].lower() for h in names)])
    res = dict(names=list(names), seen_in_training=not bool(cfg.get("holdout_names")), n_moves=len(idx))
    if not len(idx):
        return res
    for variant in ("real_name", "name_replaced_by_object"):
        nx, de = [], []
        for b0 in range(0, len(idx), 64):
            bi = idx[b0:b0 + 64]
            b = Bt.get(kind, cfg, bi)
            b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
            if variant != "real_name":
                hs = torch.as_tensor(np.vectorize(lambda p: any(h in p.lower() for h in names))(D.phrase[bi]).astype(bool))
                b["obj_txt"] = b["obj_txt"].clone(); b["obj_txt"][hs] = neutral
            P = forward(ms, b, 1)
            ar = torch.arange(len(bi))
            nx += (P["pm"].mean(0).argmax(-1) == b["mover"]).tolist()
            de += (P["pt"].mean(0).argmax(-1) == b["target"]).tolist()
        res[variant] = dict(next_exact=float(np.mean(nx)), destination=float(np.mean(de)))
    sel = np.array([i for i in idx if D.narr[i] >= 0 and any(h in D.strings[D.narr[i]].lower() for h in names)])
    if len(sel):
        b = Bt.get(kind, cfg, sel, text_ids=D.narr[sel]); b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0
        P = forward(ms, b, 1)
        res["step_command_naming_it"] = dict(n=len(sel), next_exact=float((P["pm"].mean(0).argmax(-1) == b["mover"]).float().mean()))
    pt = [(i, t) for i in idx for t in D.test_alt[i] if t >= 0 and any(h in D.strings[t].lower() for h in names)]
    if pt:
        bi = np.array([p_[0] for p_ in pt]); ti = np.array([p_[1] for p_ in pt])
        b = Bt.get(kind, cfg, bi, text_ids=ti); b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0
        P = forward(ms, b, 1)
        res["heldout_paraphrase_naming_it"] = dict(n=len(pt), next_in_set=float(b["mover_set"][torch.arange(len(bi)), P["pm"].mean(0).argmax(-1)].float().mean()))
    return res


def twin_mixed(run, kind, layouts=8):
    """Mixed-table test in the twins (tools, fruit and both containers; prompt fruit or tools), as in base_bridge, with this model;
    MiniLM prompt for v2 models, the hashed prompt (as trained) for base-class models."""
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel.physics.carry_check import make_bridge
    from cleantables.worldmodel.bridge import WMBridge2, app_keys_for
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    from cleantables.worldmodel.data import HELD_TWINS
    tools = [t["id"] for t in T if t["task"] == "tools"]; fruit = [t["id"] for t in T if t["task"] == "fruit"]
    res = {}
    rng = np.random.default_rng(0)
    base_bridge = {}
    rows_all = []                                                         # (table type, prompt, base twin, picked the prompted class)
    for base_list, extra_list, bname in ((tools, fruit, "tools_base+fruit"), (fruit, tools, "fruit_base+tools")):
        for prompt_kind in ("tools", "fruit"):
            hits, n = [], 0
            hits_held, picks, base_of = [], [], []
            for b_ in base_list:
                e = extra_list[rng.integers(len(extra_list))]
                pe = b_ if (prompt_kind == "tools") == (b_ in tools) else e
                Z = W.mixed_tokens(b_, e, pe)
                if kind == "v1":                                              # v2 'no_text': WMBridge2 swaps in the hashed vector itself
                    Z["task_emb"] = hashed(str(Z["instruction"]))
                zb, ze = dict(np.load(W.EP / b_ / "wm_tokens.npz")), dict(np.load(W.EP / e / "wm_tokens.npz"))
                n_new = len(Z["ids"]) - len(zb["ids"])
                if b_ not in base_bridge:
                    base_bridge[b_] = make_bridge(run, b_)
                B0_ = base_bridge[b_]
                if kind == "v2":
                    sel = np.isin(ze["role"], ("task", "container"))
                    keys = app_keys_for(b_, zb) + list(np.array(app_keys_for(e, ze))[sel])
                    B = WMBridge2(b_, None, Z=Z, app_keys=keys, models=B0_.models)
                else:
                    B = W.WMBridge.__new__(W.WMBridge); B.__dict__.update(B0_.__dict__); B.Z = Z; B.I = len(Z["ids"])
                Ps = [p for p in (W.place_free(Z, Z["table_geo"], n_new, rng) for _ in range(layouts)) if p is not None]
                if not Ps:
                    continue
                o = B.plan(torch.as_tensor(np.stack(Ps)), topk=1)
                want = "tool" if prompt_kind == "tools" else "fruit"
                cat = np.array([str(c).split("/")[0] for c in Z["category"]])
                for j in range(len(Ps)):
                    picks.append(cat[int(o["p_next"][j].argmax())])
                    h = cat[int(o["p_next"][j].argmax())] == want
                    hits.append(h); base_of.append(b_); rows_all.append((bname, prompt_kind, b_, bool(h)))
                    if b_ in HELD_TWINS or e in HELD_TWINS:
                        hits_held.append(h)
                n += len(Ps)
            res[f"{bname} | prompt={prompt_kind}"] = dict(n=n, pick_prompted_class=float(np.mean(hits)) if hits else None,
                                                          pick_fruit=float(np.mean([c == "fruit" for c in picks])) if picks else None,
                                                          pick_tool=float(np.mean([c == "tool" for c in picks])) if picks else None,
                                                          n_with_test_twin=len(hits_held),
                                                          pick_prompted_class_with_test_twin=float(np.mean(hits_held)) if hits_held else None)
    for bname in ("tools_base+fruit", "fruit_base+tools"):                # the instruction's own effect, same table type
        f, t = res.get(f"{bname} | prompt=fruit", {}), res.get(f"{bname} | prompt=tools", {})
        if f.get("pick_fruit") is not None and t.get("pick_fruit") is not None:
            res[f"{bname} | summary"] = dict(follows_prompt_balanced=0.5 * (f["pick_prompted_class"] + t["pick_prompted_class"]),
                                             instruction_effect_fruit=f["pick_fruit"] - t["pick_fruit"])
    # 95 % bootstrap intervals of "follows the prompt" (balanced over the two prompts), resampling base twins (each table type, and
    # both table types together: the overall share is the mean of the two table types' balanced shares)
    R = np.array(rows_all, dtype=object)
    if len(R):
        def bal(rows):
            return float(np.mean([np.mean([r[3] for r in rows if r[0] == t and r[1] == p]) for t in sorted({r[0] for r in rows})
                                  for p in ("fruit", "tools") if any(r[0] == t and r[1] == p for r in rows)]))
        g = np.random.default_rng(0)
        for name, tt in (("tools_base+fruit", ("tools_base+fruit",)), ("fruit_base+tools", ("fruit_base+tools",)), ("all", ("tools_base+fruit", "fruit_base+tools"))):
            rows = [r for r in rows_all if r[0] in tt]; bases = sorted({r[2] for r in rows})
            bs = []
            for _ in range(2000):
                q = g.choice(bases, len(bases)); rr = [r for b_ in q for r in rows if r[2] == b_]
                bs.append(bal(rr))
            res[f"{name} | follows_prompt_balanced_ci"] = [bal(rows), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), len(rows)]
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", nargs="+", required=True); ap.add_argument("--no_twin_mixed", action="store_true")
    ap.add_argument("--only_twin_mixed", action="store_true", help="re-run only the twin mixed-table test (merged into the existing json)")
    ap.add_argument("--only_language", action="store_true", help="re-run only the language + twin mixed-table tests (merged; for their CIs)")
    ap.add_argument("--only_video_modes", action="store_true", help="video_v2: only the human tests in modes 0nh / 0r / 1r (merged)")
    ap.add_argument("--val_fold", type=int, default=-1, help="watch models: the VALIDATION read-out (fold F's checkpoints, tested on the "
                    "training examples of fold F, never the held-out test) -> <run>_val<F>.json; used to choose the robot's STOP sd")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "12")))
    from cleantables.worldmodel.data import Data
    D = Data(text="minilm")
    Bt = Batcher(D)
    EV.mkdir(parents=True, exist_ok=True)
    for run in a.run:
        out_p = EV / f"{run}{__import__('os').environ.get('WM_EVAL_SUFFIX', '')}.json"   # e.g. _seg: on the subtask data's whole-subtask test
        ms, kind, cfg = load_run(run)
        if a.only_twin_mixed:
            r = json.load(open(out_p)); r["twin_mixed"] = twin_mixed(run, kind); jdump(r, out_p); log(f"twin_mixed {run} -> {out_p}"); continue
        if a.only_video_modes:
            r = json.load(open(out_p)) if out_p.exists() else dict(run=run)
            r.setdefault("human", {}).update(human(run, D, Bt, ms, kind, cfg, modes=("0nh", "0r", "1r")))
            jdump(r, out_p); log(f"video modes {run} -> {out_p}"); continue
        if a.only_language:
            r = json.load(open(out_p)); r["language"] = language(run, D, Bt, ms, kind, cfg); r["twin_mixed"] = twin_mixed(run, kind)
            jdump(r, out_p); log(f"language + twin_mixed {run} -> {out_p}"); continue
        leaky = kind == "v1" and run != "B0"
        r = dict(run=run, kind=kind, n_members=len(ms), leaky_old_split=leaky,
                 split="TEST = 10 episodes (IMG_2516_t11-t14, all of IMG_2546/2547/2548); never in v2/B0 training" + (
                     "; THIS run was trained on them (old split): reference only" if leaky else ""))
        if a.val_fold >= 0:
            from cleantables.worldmodel.bridge import load_models
            ms = load_models(RUNS2 / run, fold=a.val_fold)
            keep = D.test_idx
            D.test_idx = D.train_idx[(D.fold[D.train_idx] == a.val_fold) & (D.sub_kind[D.train_idx] == "whole")]
            r = dict(run=run, kind=kind, n_members=len(ms), split=f"VALIDATION = training fold {a.val_fold} (held out of the fold-{a.val_fold} models)",
                     watch=watch_tests(D, Bt, ms, cfg))
            D.test_idx = keep
            vp = EV / f"{run}_val{a.val_fold}{__import__('os').environ.get('WM_EVAL_SUFFIX', '')}.json"   # suffix e.g. _h23, for another WM_RUNS2
            jdump(r, vp); log(f"watch validation {run} fold {a.val_fold} -> {vp}"); continue
        if getattr(ms[0], "watch", False):                                 # watch models: their own tests (no move heads)
            r = dict(run=run, kind=kind, n_members=len(ms), split="TEST = the 10 held-out episodes", watch=watch_tests(D, Bt, ms, cfg))
            jdump(r, out_p); log(f"watch tests {run} -> {out_p}"); continue
        r["human"] = human(run, D, Bt, ms, kind, cfg, modes=(1, 0, "1nh") + ((("0nh", "0r", "1r") if cfg.get("video_v2") else ("0nh",)) if kind == "v2" else ()))
        r["language"] = language(run, D, Bt, ms, kind, cfg)
        r["generalisation"] = generalisation(run, D, Bt, ms, kind, cfg)
        if not a.no_twin_mixed:
            r["twin_mixed"] = twin_mixed(run, kind)
        jdump(r, out_p)
        log(f"eval {run}: written {out_p}")


if __name__ == "__main__":
    main()
