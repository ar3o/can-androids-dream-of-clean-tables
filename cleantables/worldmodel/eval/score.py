"""Scoring of the held-out evaluation: one metric code for every method, from the cached predictions (predict.py) and the raw pool.

CPU, numpy only. Baselines: "stays put", constant velocity, kNN retrieval, a task rule using simulator labels, and the
direct-regression baselines A1 / A2 (regression_baseline.py). Baseline settings (constant-velocity horizon H, kNN k) are
chosen on the fold-0 validation split before the test is scored. CIs: 1,000 bootstrap draws over held-out episodes (base
episode), rng seed 0. The judge section scores the seed 1 + 2 ensemble as a goal judge.

  $VLA_PY -m cleantables.worldmodel.eval.score tune     -> out/wm/tune.json (validation only)
  $VLA_PY -m cleantables.worldmodel.eval.score main     -> out/wm/main.json and tables
  $VLA_PY -m cleantables.worldmodel.eval.score judge    -> out/wm/judge.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

EB = Path(__file__).resolve().parents[3] / "outputs" / "eval"; O = EB / "out" / "wm"
THR = 0.162                                       # "moved" threshold for evaluation only (m; the tracker's rest p90)
MET_CM, STATIC_CM = 3.0, 3.0                      # as in cleantables.rl.chain_env: tolerance floor for 'met' / minimum move for a mover (cm)
H_GRID, K_GRID = (0.5, 1.0, 2.0, 4.0, 8.0), (1, 5, 20)
T2_T = np.arange(1, 9) * 0.5
NB = int(os.environ.get("NB", 1000))
REF = os.environ.get("REF", "final_1nh")          # reference prediction file; a synthetic one can be set to test the code paths
OUTTAG = os.environ.get("OUTTAG", "")


def load(nm):
    d = dict(np.load(O / nm, allow_pickle=True))
    if "t2_disp" in d and d["t2_disp"].ndim == 3 and "obj_mask" in d:  # the batch stores t2_disp flattened object-major (B, n * H, 3);
        d["t2_disp"] = d["t2_disp"].reshape(d["obj_mask"].shape[0], d["obj_mask"].shape[1], -1, 3)   # equals the raw (n, H) layout
    return d


# ------------------------------------------------------------------------------------------------ non-learned baselines
def cv_vel(now, prev, dt):
    v = (now - prev) / np.where(np.isfinite(dt), dt, 1.0)[:, None, None]
    ok = np.isfinite(prev).all(-1) & (np.isfinite(dt) & (dt <= 1.0))[:, None]     # unseen for > 1 s: no velocity estimate
    return np.where(ok[..., None], v, 0.0)


def cv_pred(now, prev, dt, H):
    v = cv_vel(now, prev, dt)
    return now + v * H, v[:, :, None, :] * T2_T[None, None, :, None], np.linalg.norm(v, axis=-1) * H


def feats(P, rows):
    """Per-object kNN features: instruction embedding, the object's name embedding, its pose and size (obj_geo); groups weighted equally."""
    te = P["task_emb"][rows].astype(np.float32); tx = P["obj_txt"][rows].astype(np.float32); g = P["obj_geo"][rows].astype(np.float32)
    n = tx.shape[1]
    return np.concatenate([np.repeat(te[:, None], n, 1), tx, g], -1)


def knn_fit(P, pool_rows):
    F = feats(P, pool_rows)
    m = (P["obj_mask"][pool_rows] > 0) & (P["obj_kind"][pool_rows] == 0) & P["t1_ok"][pool_rows] & P["t1_present"][pool_rows]
    X = F[m]; now = P["obj_geo"][pool_rows][..., :3][m]
    Y1 = P["t1_pos"][pool_rows][m] - now
    Y2 = np.where(P["t2_ok"][pool_rows][m][..., None], P["t2_disp"][pool_rows][m], np.nan)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = np.concatenate([np.full(384, 1 / np.sqrt(384)), np.full(384, 1 / np.sqrt(384)), np.full(X.shape[1] - 768, 1 / np.sqrt(X.shape[1] - 768))])
    Z = (X - mu) / sd * w
    return dict(Z=Z.astype(np.float32), Y1=Y1, Y2=Y2, mu=mu, sd=sd, w=w)


def knn_pred(M, P, rows, k):
    F = feats(P, rows); B, n = F.shape[:2]
    Q = ((F.reshape(-1, F.shape[-1]) - M["mu"]) / M["sd"] * M["w"]).astype(np.float32)
    zz = (M["Z"] ** 2).sum(1)
    off = np.zeros((len(Q), 3), np.float32); t2 = np.zeros((len(Q), 8, 3), np.float32)
    for c in range(0, len(Q), 2000):
        q = Q[c:c + 2000]; d2 = (q ** 2).sum(1)[:, None] + zz[None] - 2 * q @ M["Z"].T
        nn = np.argpartition(d2, k - 1, axis=1)[:, :k] if k > 1 else d2.argmin(1)[:, None]
        off[c:c + 2000] = M["Y1"][nn].mean(1)
        t2[c:c + 2000] = np.nan_to_num(np.nanmean(M["Y2"][nn], 1))
    now = P["obj_geo"][rows][..., :3]
    off = off.reshape(B, n, 3); return now + off, t2.reshape(B, n, 8, 3), np.linalg.norm(off, axis=-1)


# ------------------------------------------------------------------------------------------------ metrics (the same for every method)
def boot(ep, f, rng):
    u = np.unique(ep); out = []
    groups = {x: np.flatnonzero(ep == x) for x in u}
    for _ in range(NB):
        pick = np.concatenate([groups[x] for x in rng.choice(u, len(u))])
        out.append(f(pick))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def t1_block(gt, goal, rows, ref_err=None, sd=None, score=None):
    """Moved / stayed end error (median cm, CI over episodes), paired difference to the reference method, AUROC and coverage."""
    rng = np.random.default_rng(0)
    now = gt["obj_geo"][rows][..., :3]; tgt = gt["t1_pos"][rows]
    ok = gt["t1_ok"][rows] & gt["t1_present"][rows] & gt["obj_mask"][rows].astype(bool) & (gt["obj_kind"][rows] == 0)
    d = np.linalg.norm(tgt - now, axis=-1); err = 100 * np.linalg.norm(goal[rows] - tgt, axis=-1)
    epo = np.repeat(gt["ep"][rows][:, None], ok.shape[1], 1)
    out = {}
    for nm, sel in (("moved", ok & (d > THR)), ("stayed", ok & (d <= THR))):
        e, ee = err[sel], epo[sel]
        if not len(e):
            continue
        r = dict(n=int(len(e)), n_ep=int(len(np.unique(ee))), err_cm=float(np.median(e)), ci=boot(ee, lambda p: np.median(e[p]), rng))
        if ref_err is not None:
            rf = ref_err[rows][sel]
            r["diff_vs_ref_cm"] = float(np.median(e) - np.median(rf)); r["diff_ci"] = boot(ee, lambda p: np.median(e[p]) - np.median(rf[p]), rng)
        if sd is not None:
            z = np.abs(goal[rows] - tgt) / np.maximum(sd[rows], 1e-4)
            r["cov1"] = float((z[sel] <= 1).mean()); r["cov2"] = float((z[sel] <= 2).mean())
        out[nm] = r
    if score is not None:
        y = (d > THR)[ok]; s = score[rows][ok]
        if 0 < y.sum() < len(y):
            out["auroc"] = float(roc_auc_score(y, s))
            eo = epo[ok]; out["auroc_ci"] = boot(eo, lambda p: roc_auc_score(y[p], s[p]) if 0 < y[p].sum() < len(p) else np.nan, rng)
    return out, err


def next_block(gt, pick, rows):
    """Next object on keyframes with a known next object (from contact moves, evaluation only): exact, order-free and chance."""
    rng = np.random.default_rng(0)
    r = rows[gt["ref_next"][rows] >= 0]
    if not len(r):
        return {}
    ex = np.array([pick[i] == gt["ref_next"][i] for i in r], float)
    okf = np.array([gt["ref_left"][i, pick[i]] if pick[i] >= 0 else False for i in r], float)
    mov = gt["obj_mask"][r].astype(bool) & (gt["obj_kind"][r] == 0)
    ch = gt["ref_left"][r].sum(1) / np.maximum(1, mov.sum(1))
    ee = gt["ep"][r]
    return dict(n=int(len(r)), n_ep=int(len(np.unique(ee))), exact=float(ex.mean()), exact_ci=boot(ee, lambda p: ex[p].mean(), rng),
                order_free=float(okf.mean()), order_free_ci=boot(ee, lambda p: okf[p].mean(), rng), chance=float(ch.mean()))


def t2_block(gt, traj, rows):
    rng = np.random.default_rng(0)
    ok = gt["t2_ok"][rows] & (gt["obj_mask"][rows].astype(bool) & (gt["obj_kind"][rows] == 0))[..., None]
    tru = gt["t2_disp"][rows]; e = 100 * np.linalg.norm(traj[rows] - tru, axis=-1)
    has = ok.any(-1)
    ade = np.where(has, (np.where(ok, e, 0).sum(-1) / np.maximum(1, ok.sum(-1))), np.nan)
    fde = np.where(ok[..., -1], e[..., -1], np.nan)
    mvd = (np.where(ok, np.linalg.norm(tru, axis=-1), 0).max(-1) > THR) & has
    epo = np.repeat(gt["ep"][rows][:, None], ok.shape[1], 1)
    out = {}
    for nm, sel in (("all", has), ("moved", mvd)):
        a, f_, ee = ade[sel], fde[sel], epo[sel]
        out[nm] = dict(n=int(sel.sum()), ade_cm=float(np.nanmedian(a)), ade_ci=boot(ee, lambda p: np.nanmedian(a[p]), rng),
                       fde4s_cm=float(np.nanmedian(f_)))
    return out


def pick_from(score, gt, rows_all):
    s = np.where(gt["obj_mask"].astype(bool) & (gt["obj_kind"] == 0), score, -np.inf)
    return s.argmax(1)


# ------------------------------------------------------------------------------------------------ commands
def tune():
    """Choose H (constant velocity) and k (kNN) on the fold-0 validation split by the moved-object median T1 error."""
    P = load("pool.npz"); tr = ~P["is_test"]
    val = np.flatnonzero(tr & (P["fold"] == 0) & (P["sub_kind"] == "whole")); pool = np.flatnonzero(tr & (P["fold"] != 0))
    gt = dict(obj_geo=P["obj_geo"], t1_pos=P["t1_pos"], t1_ok=P["t1_ok"], t1_present=P["t1_present"], obj_mask=P["obj_mask"],
              obj_kind=P["obj_kind"], ep=P["ep"])
    res = dict(H={}, k={})
    now = P["obj_geo"][..., :3]
    for H in H_GRID:
        g, _, _ = cv_pred(now, P["prev_pos"], P["dt_prev_s"], H)
        res["H"][H] = t1_block(gt, g, val)[0]["moved"]["err_cm"]
    M = knn_fit(P, pool)
    for k in K_GRID:
        g = np.zeros_like(now); g[val] = knn_pred(M, P, val, k)[0]
        res["k"][k] = t1_block(gt, g, val)[0]["moved"]["err_cm"]
    res["H_best"] = min(res["H"], key=res["H"].get); res["k_best"] = min(res["k"], key=res["k"].get)
    res["n_val"] = int(len(val)); res["n_pool_val"] = int(len(pool))
    json.dump(res, open(O / "tune.json", "w"), indent=1); print(json.dumps(res))


def baselines(gt):
    """Predictions of the non-learned baselines on the held-out test rows (gt order)."""
    P = load("pool.npz"); T = json.load(open(O / "tune.json"))
    te = np.flatnonzero(P["is_test"]); assert (P["idx"][te] == gt["idx"]).all()
    now = gt["obj_geo"][..., :3]; n = now.shape[1]
    B = {}
    B["stays put"] = dict(goal=now.copy(), traj=np.zeros(gt["t2_disp"].shape, np.float32), score=None)
    g, t2, sc = cv_pred(now, gt["prev_pos_raw"], gt["dt_prev_s"], float(T["H_best"]))
    B["constant velocity"] = dict(goal=g, traj=t2, score=sc)
    M = knn_fit(P, np.flatnonzero(~P["is_test"]))
    g, t2, sc = knn_pred(M, P, te, int(T["k_best"]))
    B["kNN retrieval"] = dict(goal=g, traj=t2, score=sc)
    # task rule with simulator labels (privileged): 'task' objects go to the 'container' object's current position
    g = now.copy(); role = gt["twin_role"]
    for r in range(len(now)):
        if gt["scope"][r] not in ("fruit in", "tools in"):
            continue
        c = np.flatnonzero(role[r] == "container")
        if not len(c):
            continue
        for s_ in np.flatnonzero(role[r] == "task"):
            g[r, s_] = now[r, c[0]]
    B["task rule (sim labels)"] = dict(goal=g, traj=None, score=np.linalg.norm(g - now, axis=-1))
    return B, T


R2 = [("A1 direct regression, plain", "A1"), ("A2 direct regression + our data tricks", "A2")]


def round2(gt):
    """Regression and VLM baselines: {label: dict(goal, p_now, rows)} on full-length arrays (NaN outside their rows)."""
    out = {}
    N, n = gt["obj_mask"].shape
    for lab, nm in R2:
        for sd_ in (0, 1, 2):
            f = O / f"pred_{nm}_s{sd_}.npz"
            if f.exists():
                p = load(f.name); g = np.full((N, n, 3), np.nan, np.float32); pn = np.full((N, n), np.nan, np.float32)
                g[p["rows"]] = p["all__goal"]; pn[p["rows"]] = p["all__p_now"]
                out[f"{lab} (seed {sd_})"] = dict(goal=g, p_now=pn, rows=p["rows"])
    if (O / "pred_VLM.npz").exists():
        p = load("pred_VLM.npz"); g = np.full((N, n, 3), np.nan, np.float32); pn = np.full((N, n), np.nan, np.float32)
        g[p["rows"]] = p["all__goal"]; pn[p["rows"]] = p["all__p_now"]
        out["VLM zero-shot (Qwen3-VL-8B, local)"] = dict(goal=g, p_now=pn, rows=p["rows"], n_unparsed=int(p["n_unparsed"]))
    return out


def model_entry(nm, sub="all"):
    p = load(f"pred_{nm}.npz")
    e = dict(goal=p[f"{sub}__goal"], sd=p[f"{sub}__sd"], p_now=p[f"{sub}__p_now"], traj=p.get(f"{sub}__traj"), e1=p.get(f"{sub}__e1"), rows=p["rows"])
    return e


def main():
    gt = load("gt.npz"); N = len(gt["idx"])
    groups = {"in scope": np.flatnonzero(gt["scope"] != ""), "fruit in": np.flatnonzero(gt["scope"] == "fruit in"),
              "fruit out": np.flatnonzero(gt["scope"] == "fruit out"), "tools in": np.flatnonzero(gt["scope"] == "tools in"),
              "all 767": np.arange(N)}
    B, T = baselines(gt)
    now = gt["obj_geo"][..., :3]
    M = {}
    for label, nm, sub in (("Ours (3-seed ensemble)", REF, "all"), ("Ours seed 0", REF, "s0"), ("Ours seed 1", REF, "s1"),
                           ("Ours seed 2", REF, "s2")):
        if (O / f"pred_{nm}.npz").exists():
            M[label] = model_entry(nm, sub)
    ref = M["Ours (3-seed ensemble)"]
    R2_ = round2(gt)
    ref_err = 100 * np.linalg.norm(ref["goal"] - gt["t1_pos"], axis=-1)
    res = dict(tune=T, groups={k: int(len(v)) for k, v in groups.items()}, rows={})
    entries = {**{k: dict(goal=v["goal"], traj=v["traj"], score=v["score"], sd=None, pick=None) for k, v in B.items()},
               **{k: dict(goal=v["goal"], traj=v["traj"], score=np.linalg.norm(v["goal"] - now, axis=-1), sd=v["sd"],
                          pick=pick_from(v["p_now"], gt, None), e1=v["e1"]) for k, v in M.items()},
               **{k: dict(goal=np.where(np.isnan(v["goal"]), now, v["goal"]), traj=None, score=np.linalg.norm(np.nan_to_num(v["goal"] - now), axis=-1),
                          sd=None, pick=pick_from(np.nan_to_num(v["p_now"], nan=-1), gt, None)) for k, v in R2_.items() if not k.startswith("VLM")}}
    for k, e in entries.items():
        if e["pick"] is None and e["score"] is not None:
            e["pick"] = pick_from(e["score"], gt, None)
        R = {}
        for gname, rows in groups.items():
            if k.startswith("task rule") and gname == "all 767":
                continue
            t1, _ = t1_block(gt, e["goal"], rows, ref_err=ref_err, sd=e.get("sd"), score=e["score"])
            R[gname] = dict(T1=t1)
            if e.get("e1") is not None:
                y_ok = gt["t1_ok"][rows] & gt["t1_present"][rows] & gt["obj_mask"][rows].astype(bool) & (gt["obj_kind"][rows] == 0)
                y = (np.linalg.norm(gt["t1_pos"][rows] - now[rows], axis=-1) > THR)[y_ok]
                R[gname]["dist_head_auroc"] = float(roc_auc_score(y, e["e1"][rows][y_ok])) if 0 < y.sum() < len(y) else None
            if e["pick"] is not None:
                R[gname]["next"] = next_block(gt, e["pick"], rows)
            if e["traj"] is not None:
                R[gname]["T2"] = t2_block(gt, e["traj"], rows)
        res["rows"][k] = R
    # chance (next object only) and final_s2 (rows with a known next object)
    if (O / "pred_final_s2.npz").exists():
        f2 = load("pred_final_s2.npz"); R = {}
        for gname, rows in groups.items():
            R[gname] = dict(next=next_block(gt, f2["next"], rows))
            r = rows[gt["ref_next"][rows] >= 0]
            ok = np.array([gt["t1_ok"][i, gt["ref_next"][i]] & gt["t1_present"][i, gt["ref_next"][i]] for i in r], bool)
            if ok.any():
                tn = np.array([gt["t1_pos"][i, gt["ref_next"][i]] for i in r[ok]]); nn_ = np.array([now[i, gt["ref_next"][i]] for i in r[ok]])
                o2 = 100 * np.linalg.norm(f2["goal_true_next"][r[ok]] - tn, axis=-1)
                ours = 100 * np.linalg.norm(np.array([ref["goal"][i, gt["ref_next"][i]] for i in r[ok]]) - tn, axis=-1)
                st = 100 * np.linalg.norm(nn_ - tn, axis=-1)
                R[gname]["true_next_end_err_cm"] = dict(final_s2=float(np.median(o2)), ours=float(np.median(ours)), stay=float(np.median(st)), n=int(ok.sum()))
        res["rows"]["final_s2 (old model; P5 rows)"] = R
    # the VLM's keyframes: every method on the same keyframes
    if any(k.startswith("VLM") for k in R2_):
        V = [v for k, v in R2_.items() if k.startswith("VLM")][0]; vr = np.sort(V["rows"])
        first = np.array(sorted({min([r for r in vr if gt["ep"][r] == gt["ep"][q] and gt["task_text"][r] == gt["task_text"][q]], key=lambda r: gt["k"][r]) for q in vr}))
        vg = {"VLM keyframes, in scope": vr, "first keyframe of each episode": first,
              "VLM keyframes, put-into": vr[np.isin(gt["scope"][vr], ["fruit in", "tools in"])]}
        allm = dict(entries)
        allm["VLM zero-shot (Qwen3-VL-8B, local)"] = dict(goal=np.where(np.isnan(V["goal"]), now, V["goal"]), traj=None,
                                                         score=np.linalg.norm(np.nan_to_num(V["goal"] - now), axis=-1), sd=None,
                                                         pick=pick_from(np.nan_to_num(V["p_now"], nan=-1), gt, None))
        VR = {}
        for k, e in allm.items():
            if k.startswith("Ours seed") or k.startswith("constant"):
                continue
            VR[k] = {gname: dict(T1=t1_block(gt, e["goal"], rows, ref_err=ref_err, score=e["score"])[0],
                                 next=next_block(gt, e["pick"], rows) if e["pick"] is not None else {}) for gname, rows in vg.items()}
        res["vlm_keyframes"] = dict(rows=VR, n=int(len(vr)), n_unparsed=V["n_unparsed"])
    json.dump(res, open(O / f"main{OUTTAG}.json", "w"), indent=1, default=float)
    print(json.dumps({k: {g: v.get("T1", {}).get("moved", {}).get("err_cm") for g, v in R.items()} for k, R in res["rows"].items()}, indent=0))


# ------------------------------------------------------------------------------------------------ ablations, inputs, text
def full(gt, p, key):
    a = np.full(gt["obj_geo"].shape[:2] + p[key].shape[2:], np.nan, np.float32); a[p["rows"]] = p[key]; return a


def ablate():
    gt = load("gt.npz"); now = gt["obj_geo"][..., :3]
    groups = {"in scope": np.flatnonzero(gt["scope"] != ""), "fruit in": np.flatnonzero(gt["scope"] == "fruit in"),
              "fruit out": np.flatnonzero(gt["scope"] == "fruit out"), "tools in": np.flatnonzero(gt["scope"] == "tools in")}
    ref = load(f"pred_{REF}.npz"); ref_goal = ref["all__goal"]; ref_err = 100 * np.linalg.norm(ref_goal - gt["t1_pos"], axis=-1)
    s0_err = 100 * np.linalg.norm(ref["s0__goal"] - gt["t1_pos"], axis=-1)
    res = dict(rows={})
    names = [("Ours final (3 seeds, full data)", REF), ("first recipe (3 seeds, full data)", "orig_full"),
             ("L0 first recipe (fold 0)", "L0_watch"), ("L1 + noise floor + distance head", "L1_pilot"), ("L2 + paired-instruction composites", "L2_pair"),
             ("L3 + rigid re-layout", "L3_rigid"), ("L4 - table reference, 9000 steps = final recipe (seed 0)", "L4_notab"),
             ("L4 final recipe seed 1 (fold 0)", "L4_s1"), ("L4 final recipe seed 2 (fold 0)", "L4_s2"),
             ("input: video + hands (mode 0)", "final_0"), ("input: video, no hands (0nh)", "final_0nh"), ("input: appearance off (1nh)", "final_noapp"),
             ("Ours final seed 0 alone (full data; the leave-one-out reference)", REF, "s0"),
             ("leave-one-out: no auxiliary targets (T1 only; seed 0; next object = farthest predicted move)", "loo_noaux"),
             ("leave-one-out: no paired-instruction composites (seed 0)", "loo_nopair"), ("leave-one-out: no rigid re-layout (seed 0)", "loo_norigid")]
    for label, nm, *sub in names:
        sub = sub[0] if sub else "all"
        if not (O / f"pred_{nm}.npz").exists():
            res["rows"][label] = "missing"; continue
        p = load(f"pred_{nm}.npz"); g = full(gt, p, f"{sub}__goal"); sd = full(gt, p, f"{sub}__sd"); pn = full(gt, p, f"{sub}__p_now")
        e1 = full(gt, p, f"{sub}__e1") if f"{sub}__e1" in p else None
        if nm == "loo_noaux":                                     # its distance / T2 heads are untrained: use the predicted move length
            pn = np.linalg.norm(g - now, axis=-1); e1 = None
        pick = pick_from(np.nan_to_num(pn, nan=-1), gt, None); R = {}
        for gname, rows in groups.items():
            t1, _ = t1_block(gt, g, rows, ref_err=ref_err, sd=sd, score=np.linalg.norm(g - now, axis=-1))
            R[gname] = dict(T1=t1, next=next_block(gt, pick, rows))
            if nm.startswith("loo_"):                                 # the leave-one-out reference is seed 0 alone
                R[gname]["T1_vs_seed0"] = t1_block(gt, g, rows, ref_err=s0_err)[0]
            if e1 is not None:
                y_ok = gt["t1_ok"][rows] & gt["t1_present"][rows] & gt["obj_mask"][rows].astype(bool) & (gt["obj_kind"][rows] == 0)
                y = (np.linalg.norm(gt["t1_pos"][rows] - now[rows], axis=-1) > THR)[y_ok]
                R[gname]["dist_head_auroc"] = float(roc_auc_score(y, e1[rows][y_ok])) if 0 < y.sum() < len(y) else None
        res["rows"][label] = R
    # shuffled instruction (the other family's text) vs own, same rows
    if (O / "pred_final_shuffle.npz").exists():
        p = load("pred_final_shuffle.npz"); rows = p["rows"]; g = full(gt, p, "all__goal"); pn = full(gt, p, "all__p_now")
        mov = gt["obj_mask"].astype(bool) & (gt["obj_kind"] == 0)
        ok = gt["t1_ok"] & gt["t1_present"] & mov; moved = ok & (np.linalg.norm(gt["t1_pos"] - now, axis=-1) > THR)
        shift = 100 * np.linalg.norm(g - ref_goal, axis=-1)
        S = {}
        for gname, rr in groups.items():
            m_all = mov[rr]; m_mv = moved[rr]
            tot_own = np.where(m_all, np.linalg.norm(ref_goal[rr] - now[rr], axis=-1), 0).sum(1) * 100
            tot_sh = np.where(m_all, np.linalg.norm(g[rr] - now[rr], axis=-1), 0).sum(1) * 100
            t1s, _ = t1_block(gt, g, rr, ref_err=ref_err)
            S[gname] = dict(mean_end_shift_cm_all=float(np.nanmean(shift[rr][m_all])), mean_end_shift_cm_moved=float(np.nanmean(shift[rr][m_mv])),
                            moved_err_own_cm=float(np.median(ref_err[rr][m_mv])), moved_err_shuffled=t1s.get("moved"),
                            total_pred_motion_cm_own=float(np.median(tot_own)), total_pred_motion_cm_shuffled=float(np.median(tot_sh)),
                            next_own=next_block(gt, pick_from(ref["all__p_now"], gt, None), rr),
                            next_shuffled=next_block(gt, pick_from(np.nan_to_num(pn, nan=-1), gt, None), rr))
        res["shuffled_instruction"] = S
    for lab, nm in R2:
        for sd_ in (0, 1, 2):
            fo, fs = O / f"pred_{nm}_s{sd_}.npz", O / f"pred_{nm}_s{sd_}_shuffle.npz"
            if not (fo.exists() and fs.exists()):
                continue
            po, ps = load(fo.name), load(fs.name); go = po["all__goal"]; gs = full(gt, ps, "all__goal")
            err_o = 100 * np.linalg.norm(go - gt["t1_pos"], axis=-1)
            mov = gt["obj_mask"].astype(bool) & (gt["obj_kind"] == 0); ok = gt["t1_ok"] & gt["t1_present"] & mov
            moved = ok & (np.linalg.norm(gt["t1_pos"] - now, axis=-1) > THR); S2 = {}
            for gname, rr in groups.items():
                tot_o = np.where(mov[rr], np.linalg.norm(go[rr] - now[rr], axis=-1), 0).sum(1) * 100
                tot_s = np.where(mov[rr], np.linalg.norm(gs[rr] - now[rr], axis=-1), 0).sum(1) * 100
                t1s, _ = t1_block(gt, gs, rr)
                S2[gname] = dict(mean_end_shift_cm_all=float(np.nanmean(100 * np.linalg.norm(gs - go, axis=-1)[rr][mov[rr]])),
                                 moved_err_own_cm=float(np.median(err_o[rr][moved[rr]])), moved_err_shuffled=t1s.get("moved"),
                                 total_pred_motion_cm_own=float(np.median(tot_o)), total_pred_motion_cm_shuffled=float(np.median(tot_s)))
            res.setdefault("shuffled_instruction_round2", {})[f"{lab} (seed {sd_})"] = S2
    # the same shuffle test for seed 0 alone and the leave-one-out runs (in-scope rows)
    for lab, own, sub, shf in (("Ours final seed 0 alone", REF, "s0", "final_s0_shuffle"), ("leave-one-out: no auxiliary targets", "loo_noaux", "all", "loo_noaux_shuffle"),
                               ("leave-one-out: no paired-instruction composites", "loo_nopair", "all", "loo_nopair_shuffle"),
                               ("leave-one-out: no rigid re-layout", "loo_norigid", "all", "loo_norigid_shuffle")):
        if not ((O / f"pred_{own}.npz").exists() and (O / f"pred_{shf}.npz").exists()):
            continue
        go = full(gt, load(f"pred_{own}.npz"), f"{sub}__goal"); gs = full(gt, load(f"pred_{shf}.npz"), "all__goal")
        err_o = 100 * np.linalg.norm(go - gt["t1_pos"], axis=-1)
        mov = gt["obj_mask"].astype(bool) & (gt["obj_kind"] == 0); ok = gt["t1_ok"] & gt["t1_present"] & mov
        moved = ok & (np.linalg.norm(gt["t1_pos"] - now, axis=-1) > THR); S2 = {}
        for gname, rr in groups.items():
            tot_o = np.where(mov[rr], np.linalg.norm(go[rr] - now[rr], axis=-1), 0).sum(1) * 100
            tot_s = np.where(mov[rr], np.linalg.norm(gs[rr] - now[rr], axis=-1), 0).sum(1) * 100
            S2[gname] = dict(mean_end_shift_cm_all=float(np.nanmean(100 * np.linalg.norm(gs - go, axis=-1)[rr][mov[rr]])),
                             moved_err_own_cm=float(np.median(err_o[rr][moved[rr]])), moved_err_shuffled=t1_block(gt, gs, rr)[0].get("moved"),
                             total_pred_motion_cm_own=float(np.median(tot_o)), total_pred_motion_cm_shuffled=float(np.median(tot_s)))
        res.setdefault("shuffled_instruction_round2", {})[lab] = S2
    # single-object prompts
    if (O / "pred_final_prompts.npz").exists():
        p = load("pred_final_prompts.npz"); rows = p["rows"]; nm_ = p["prompt_name"]
        leaf = np.vectorize(lambda c: c.split("/")[-1])(gt["twin_cat"])
        P_ = []
        for q, (r, name) in enumerate(zip(rows, nm_)):
            mov = gt["obj_mask"][r].astype(bool) & (gt["obj_kind"][r] == 0)
            named = mov & (gt["twin_role"][r] == "task") & (leaf[r] == name); other = mov & (gt["twin_role"][r] == "task") & ~named
            cont = np.flatnonzero(gt["twin_role"][r] == "container")
            dp = np.linalg.norm(p["all__goal"][q] - now[r], axis=-1) * 100; do = np.linalg.norm(ref_goal[r] - now[r], axis=-1) * 100
            pk = int(np.where(mov, p["all__p_now"][q], -1).argmax()); pko = int(np.where(mov, ref["all__p_now"][r], -1).argmax())
            row = dict(scope=str(gt["scope"][r]), ep=str(gt["ep"][r]), name=str(name), n_named=int(named.sum()), n_other_task=int(other.sum()),
                       sel_prompt=float(dp[named].sum() / max(dp[mov].sum(), 1e-6)), sel_own=float(do[named].sum() / max(do[mov].sum(), 1e-6)),
                       named_share=float(named.sum() / max(mov.sum(), 1)), top1_prompt=bool(named[pk]), top1_own=bool(named[pko]),
                       disp_named_prompt=float(np.median(dp[named])) if named.any() else np.nan,
                       disp_other_prompt=float(np.median(dp[other])) if other.any() else np.nan,
                       disp_named_own=float(np.median(do[named])) if named.any() else np.nan,
                       disp_other_own=float(np.median(do[other])) if other.any() else np.nan)
            if len(cont) and named.any():
                c = now[r, cont[0]]
                row["named_to_container_prompt_cm"] = float(np.median(100 * np.linalg.norm(p["all__goal"][q][named] - c, axis=-1)))
                row["named_to_container_own_cm"] = float(np.median(100 * np.linalg.norm(ref_goal[r][named] - c, axis=-1)))
                row["named_to_container_start_cm"] = float(np.median(100 * np.linalg.norm(now[r][named] - c, axis=-1)))
            P_.append(row)
        rng = np.random.default_rng(0); out = {}
        for sc in ("fruit in", "tools in", None):
            X = [x for x in P_ if sc is None or x["scope"] == sc]
            if not X:
                continue
            ee = np.array([x["ep"] for x in X]); f = lambda k: np.array([x.get(k, np.nan) for x in X], float)  # noqa: E731
            out[sc or "in scope"] = {k: dict(mean=float(np.nanmean(f(k))), ci=boot(ee, lambda pp, k=k: np.nanmean(f(k)[pp]), rng))
                                     for k in ("sel_prompt", "sel_own", "named_share", "top1_prompt", "top1_own", "disp_named_prompt", "disp_other_prompt",
                                               "disp_named_own", "disp_other_own", "named_to_container_prompt_cm", "named_to_container_own_cm",
                                               "named_to_container_start_cm")} | dict(n_prompt_rows=len(X), names=sorted({x["name"] for x in X}))
        res["prompts"] = out
    json.dump(res, open(O / f"ablate{OUTTAG}.json", "w"), indent=1, default=float)
    print(json.dumps({k: (v if isinstance(v, str) else v["in scope"]["T1"].get("moved", {}).get("err_cm")) for k, v in res["rows"].items()}, indent=0))


# ------------------------------------------------------------------------------------------------ the judge (seeds 1 + 2)
def segments(gt, scopes):
    S = {}
    for r in np.flatnonzero(np.isin(gt["scope"], scopes)):
        S.setdefault((gt["ep"][r], gt["task_text"][r]), []).append(r)
    return {k: sorted(v, key=lambda r: gt["k"][r]) for k, v in S.items()}


def state_at(gt, r0, r):
    """Positions at keyframe row r in row r0's slot order (matched by tracker id); NaN where unseen."""
    out = np.full((gt["obj_id"].shape[1], 3), np.nan, np.float32)
    for s_ in np.flatnonzero(gt["obj_mask"][r0]):
        q = np.flatnonzero((gt["obj_id"][r] == gt["obj_id"][r0, s_]) & gt["obj_mask"][r].astype(bool))
        if len(q):
            out[s_] = gt["obj_geo"][r, q[0], :3]
    return out


def within(state, target, tol):
    return (np.abs(state[..., :2] - target[..., :2]) * 100 <= tol).all(-1)


def judge():
    gt = load("gt.npz"); p = load(f"pred_{REF}.npz"); now = gt["obj_geo"][..., :3]; P = {}
    J, Jsd = p["judge12__goal"], p["judge12__sd"]; G0, G1, G2, GA = p["s0__goal"], p["s1__goal"], p["s2__goal"], p["all__goal"]
    tol_all = np.maximum(Jsd[..., :2] * 100, MET_CM)
    B, _ = baselines(gt)
    R2_ = round2(gt)
    rng = np.random.default_rng(0)
    put = np.flatnonzero(np.isin(gt["scope"], ["fruit in", "tools in"]))
    segs = segments(gt, ["fruit in", "tools in"]); first = np.array([v[0] for v in segs.values()])
    ok = gt["t1_ok"] & gt["t1_present"] & gt["obj_mask"].astype(bool) & (gt["obj_kind"] == 0)
    dmv = np.linalg.norm(gt["t1_pos"] - now, axis=-1)
    # goal accuracy and acceptance
    A = {}
    for rows_nm, rows in (("every put-into keyframe", put), ("first keyframe of each episode", first)):
        Ar = {}
        for label, g in (("stays put", B["stays put"]["goal"]), ("kNN retrieval", B["kNN retrieval"]["goal"]), ("task rule (sim labels)", B["task rule (sim labels)"]["goal"]),
                         ("seed 1 alone", G1), ("seed 2 alone", G2), ("JUDGE (seeds 1+2)", J), ("3-seed ensemble", GA), ("seed 0 (actor's init) alone", G0),
                         *[(k, v["goal"]) for k, v in R2_.items()]):
            sel = rows if not np.isnan(g[rows]).any() else rows[~np.isnan(g[rows]).any((1, 2))]
            if len(sel):
                Ar[label] = t1_block(gt, g, sel)[0]; Ar[label]["n_keyframes"] = int(len(sel))
        mv = ok[rows] & (dmv[rows] > THR); st = ok[rows] & (dmv[rows] <= THR); ee = np.repeat(gt["ep"][rows][:, None], mv.shape[1], 1)
        acc_h = within(gt["t1_pos"][rows], J[rows], tol_all[rows]); acc_0 = within(now[rows], J[rows], tol_all[rows])
        Ar["JUDGE acceptance"] = dict(human_end_accepted_moved=float(acc_h[mv].mean()), ci=boot(ee[mv], lambda q: acc_h[mv][q].mean(), rng),
                                      do_nothing_accepted_moved=float(acc_0[mv].mean()), ci0=boot(ee[mv], lambda q: acc_0[mv][q].mean(), rng),
                                      human_end_accepted_stayed=float(acc_h[st].mean()), n_moved=int(mv.sum()), n_stayed=int(st.sum()),
                                      tol_cm_median=float(np.median(tol_all[rows][mv])))
        A[rows_nm] = Ar
    P["4a"] = A
    # disagreement vs error (every put-into keyframe; moved and all objects)
    U = {}
    for nm, sel in (("moved objects", ok & (dmv > THR)), ("all objects", ok)):
        m = np.zeros_like(sel); m[put] = sel[put]
        d12 = 100 * np.linalg.norm(G1 - G2, axis=-1)[m]; sdm = 100 * Jsd[..., :2].mean(-1)[m]; err = 100 * np.linalg.norm(J - gt["t1_pos"], axis=-1)[m]
        ee = np.repeat(gt["ep"][:, None], m.shape[1], 1)[m]
        y = err > MET_CM; o = np.argsort(d12, kind="stable"); cs = np.arange(1, len(o) + 1)
        cover = np.linspace(0.1, 1.0, 10)
        rc = [dict(coverage=float(c), mean_err_cm=float(err[o][: max(1, int(c * len(o)))].mean()), share_err_gt3=float(y[o][: max(1, int(c * len(o)))].mean())) for c in cover]
        aurc = float((np.cumsum(err[o]) / cs).mean()); aurc_oracle = float((np.cumsum(np.sort(err)) / cs).mean())
        U[nm] = dict(n=int(len(err)), spearman_d12=float(spearmanr(d12, err)[0]), spearman_d12_ci=boot(ee, lambda q: spearmanr(d12[q], err[q])[0], rng),
                     spearman_sd=float(spearmanr(sdm, err)[0]), auroc_d12_err_gt3=float(roc_auc_score(y, d12)) if 0 < y.sum() < len(y) else None,
                     auroc_sd_err_gt3=float(roc_auc_score(y, sdm)) if 0 < y.sum() < len(y) else None,
                     risk_coverage=rc, aurc_mean_err=aurc, aurc_random=float(err.mean()), aurc_oracle=aurc_oracle)
    P["4b_4f"] = U
    # progress curves, anchors, XGap, forward ranking and backwards; per segment, then mean over segments
    def seg_curves(scopes):
        out = []
        for (e, t), rr in segments(gt, scopes).items():
            r0, rL = rr[0], rr[-1]; s0 = now[r0]; g = J[r0]; tol = tol_all[r0]; end = gt["t1_pos"][r0]
            jm = (np.linalg.norm((g - s0)[..., :2], axis=-1) * 100 >= STATIC_CM); hm = dmv[r0] > THR
            sc = ok[r0] & (jm | hm); sc_j = ok[r0] & jm
            if sc.sum() == 0:
                continue
            k0, kL = gt["k"][r0], gt["k"][rL]
            fr = np.array([(gt["k"][r] - k0) / max(1, kL - k0) for r in rr]); st = np.stack([state_at(gt, r0, r) for r in rr])
            def J_(state, target, m):
                w = within(state, target, tol); v = np.isfinite(state).all(-1) & m
                return float(w[v].mean()) if v.any() else np.nan
            grid = np.linspace(0, 1, 11); near = [int(np.argmin(np.abs(fr - x))) for x in grid]
            cur = dict(ep=e, text=t, n_obj=int(sc.sum()), n_obj_judge_movers=int(sc_j.sum()),
                       J_judge=[J_(st[i], g, sc) for i in near], J_true=[J_(st[i], end, sc) for i in near],
                       J_judge_trainset=[J_(st[i], g, sc_j) for i in near], J_true_trainset=[J_(st[i], end, sc_j) for i in near])
            anchors = {"do nothing": st[0], "human 25%": st[int(np.argmin(np.abs(fr - .25)))],
                       "human 50%": st[int(np.argmin(np.abs(fr - .5)))], "human 75%": st[int(np.argmin(np.abs(fr - .75)))], "human 100%": st[-1],
                       "judge-optimal": g, "actor-init-optimal (seed 0 goals)": G0[r0],
                       "at the 3-seed ensemble's goals": GA[r0], "at kNN retrieval's goals": B["kNN retrieval"]["goal"][r0],
                       "at the task rule's goals": B["task rule (sim labels)"]["goal"][r0],
                       **{f"at {k}'s goals": v["goal"][r0] for k, v in R2_.items() if not np.isnan(v["goal"][r0]).any()}}
            cur["anchors"] = {a: dict(J_judge=J_(s, g, sc), J_true=J_(s, end, sc), J_judge_trainset=J_(s, g, sc_j), J_true_trainset=J_(s, end, sc_j)) for a, s in anchors.items()}
            # backwards: the judge read at the last keyframe (task done)
            gL, tolL = J[rL], tol_all[rL]; okL = gt["obj_mask"][rL].astype(bool) & (gt["obj_kind"][rL] == 0)
            stayL = within(now[rL], gL, tolL)
            hmL = np.zeros_like(okL)
            for s_ in np.flatnonzero(okL):
                q = np.flatnonzero((gt["obj_id"][r0] == gt["obj_id"][rL, s_]) & ok[r0])
                hmL[s_] = bool(len(q)) and bool(hm[q[0]])
            st0 = within(now[r0], g, tol)
            rev = []
            for x in grid:                                              # x = fraction of reversed time: 0 = the end frame, 1 = the start
                stx = state_at(gt, rL, rr[int(np.argmin(np.abs(fr - (1 - x))))]); v = np.isfinite(stx).all(-1) & okL & hmL
                rev.append(float(within(stx, gL, tolL)[v].mean()) if v.any() else np.nan)
            cur["backwards"] = dict(judge_at_end_says_stay_all=float(stayL[okL].mean()), judge_at_end_says_stay_human_movers=float(stayL[hmL].mean()) if hmL.any() else np.nan,
                                    judge_at_start_says_stay_human_movers=float(st0[ok[r0] & hm].mean()) if (ok[r0] & hm).any() else np.nan,
                                    J_judge_reversed=rev)
            out.append(cur)
        return out
    def summarise(C):
        if not C:
            return {}
        ee = np.array([c["ep"] for c in C]); rng2 = np.random.default_rng(0); S = dict(n_segments=len(C), segments=[(c["ep"], c["n_obj"]) for c in C])
        for key in ("J_judge", "J_true", "J_judge_trainset", "J_true_trainset"):
            M_ = np.array([c[key] for c in C], float); S[key + "_curve"] = np.nanmean(M_, 0).tolist()
        an = list(C[0]["anchors"])
        S["anchors"] = {}
        for a in an:
            jj = np.array([c["anchors"][a]["J_judge"] for c in C]); jt = np.array([c["anchors"][a]["J_true"] for c in C])
            S["anchors"][a] = dict(J_judge=float(np.nanmean(jj)), J_true=float(np.nanmean(jt)), XGap=float(np.nanmean(jj - jt)),
                                   XGap_ci=boot(ee, lambda q: np.nanmean(jj[q] - jt[q]), rng2),
                                   XGap_trainset=float(np.nanmean([c["anchors"][a]["J_judge_trainset"] - c["anchors"][a]["J_true_trainset"] for c in C])))
        def pra(q):
            jj = np.array([[np.nanmean([C[i]["anchors"][a]["J_judge"] for i in q])] for a in an]).ravel()
            jt = np.array([[np.nanmean([C[i]["anchors"][a]["J_true"] for i in q])] for a in an]).ravel()
            agree, tot, viol = 0, 0, np.zeros(len(an))
            for i in range(len(an)):
                for j in range(len(an)):
                    if i < j and abs(jt[i] - jt[j]) > 1e-9:
                        tot += 1; agree += (jj[i] - jj[j]) * (jt[i] - jt[j]) > 0
                    if i != j and (jj[i] - jj[j]) * (jt[i] - jt[j]) < 0:
                        viol[i] = max(viol[i], abs(jt[i] - jt[j]))
            return agree / max(tot, 1), viol.mean()
        pa, mm = pra(np.arange(len(C)))
        bs = [pra(rng2.integers(0, len(C), len(C))) for _ in range(NB)]
        S["ranking"] = dict(pairwise_accuracy=float(pa), pairwise_ci=[float(np.percentile([b[0] for b in bs], 2.5)), float(np.percentile([b[0] for b in bs], 97.5))],
                            MMRV=float(mm), MMRV_ci=[float(np.percentile([b[1] for b in bs], 2.5)), float(np.percentile([b[1] for b in bs], 97.5))], anchors=an)
        S["backwards"] = {k: float(np.nanmean([c["backwards"][k] for c in C])) for k in ("judge_at_end_says_stay_all", "judge_at_end_says_stay_human_movers",
                                                                                       "judge_at_start_says_stay_human_movers")}
        S["backwards"]["J_judge_reversed_curve"] = np.nanmean(np.array([c["backwards"]["J_judge_reversed"] for c in C], float), 0).tolist()
        S["per_segment"] = C
        return S
    P["4d_put_into"] = summarise(seg_curves(["fruit in", "tools in"]))
    P["4d_fruit_out_reverse_task"] = summarise(seg_curves(["fruit out"]))
    json.dump(P, open(O / f"judge{OUTTAG}.json", "w"), indent=1, default=float)
    print(json.dumps({k: P["4a"]["first keyframe of each episode"][k].get("moved", P["4a"]["first keyframe of each episode"][k]) for k in P["4a"]["first keyframe of each episode"]}, indent=0, default=float)[:3000])


if __name__ == "__main__":
    {"tune": tune, "main": main, "ablate": ablate, "judge": judge}[sys.argv[1]]()
