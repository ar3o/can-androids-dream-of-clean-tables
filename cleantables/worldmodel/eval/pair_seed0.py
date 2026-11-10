"""Paired comparison from stored predictions: our model (seed 0 alone) against the A2 regression baseline (seed 0) on the
same keyframes and objects. (a) moved-object end error, in scope; (b) which-moves AUROC, in scope; (c) first-frame goal
error on put-into tasks. Differences are A2 minus ours, with bootstrap CIs over episodes (score.boot, 1,000 draws, rng 0).

  $VLA_PY -m cleantables.worldmodel.eval.pair_seed0      -> out/wm/pair_s0_A2.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import cleantables.worldmodel.eval.score as W  # noqa: E402

gt = W.load("gt.npz"); p = W.load(f"pred_{W.REF}.npz"); a2 = W.load("pred_A2_s0.npz")
assert (p["rows"] == np.arange(len(gt["idx"]))).all() and (a2["rows"] == np.arange(len(gt["idx"]))).all()
now = gt["obj_geo"][..., :3]; tgt = gt["t1_pos"]
GO = {"ours seed 0": p["s0__goal"], "A2 (seed 0)": a2["all__goal"]}
ok = gt["t1_ok"] & gt["t1_present"] & gt["obj_mask"].astype(bool) & (gt["obj_kind"] == 0)
dmv = np.linalg.norm(tgt - now, axis=-1)
ERR = {k: 100 * np.linalg.norm(g - tgt, axis=-1) for k, g in GO.items()}
SC = {k: np.linalg.norm(g - now, axis=-1) for k, g in GO.items()}          # which-moves score = length of the predicted move, as in score.py
EP = np.repeat(gt["ep"][:, None], ok.shape[1], 1)


def ci(ee, f):
    return W.boot(ee, f, np.random.default_rng(0))


def err_block(rows):
    m = np.zeros_like(ok); m[rows] = (ok & (dmv > W.THR))[rows]
    e0, e2, ee = ERR["ours seed 0"][m], ERR["A2 (seed 0)"][m], EP[m]
    return dict(n_moved=int(m.sum()), n_ep=int(len(np.unique(ee))),
                ours_seed0=float(np.median(e0)), ours_seed0_ci=ci(ee, lambda q: np.median(e0[q])),
                A2=float(np.median(e2)), A2_ci=ci(ee, lambda q: np.median(e2[q])),
                diff_A2_minus_ours=float(np.median(e2) - np.median(e0)), diff_ci=ci(ee, lambda q: np.median(e2[q]) - np.median(e0[q])))


def auc_block(rows):
    m = np.zeros_like(ok); m[rows] = ok[rows]
    y = (dmv > W.THR)[m]; s0, s2, ee = SC["ours seed 0"][m], SC["A2 (seed 0)"][m], EP[m]

    def d(q):
        return roc_auc_score(y[q], s2[q]) - roc_auc_score(y[q], s0[q]) if 0 < y[q].sum() < len(q) else np.nan
    return dict(n_objects=int(m.sum()), n_moved=int(y.sum()), ours_seed0=float(roc_auc_score(y, s0)), A2=float(roc_auc_score(y, s2)),
                diff_A2_minus_ours=float(roc_auc_score(y, s2) - roc_auc_score(y, s0)), diff_ci=ci(ee, d))


ins = np.flatnonzero(gt["scope"] != "")
first = np.array([v[0] for v in W.segments(gt, ["fruit in", "tools in"]).values()])
res = {"(a) moved-object end error, in scope (cm)": err_block(ins),
       "(b) which-moves AUROC, in scope": auc_block(ins),
       "(c) first-frame goal error, put-into (cm)": err_block(first)}
for fam in ("fruit in", "tools in", "fruit out"):
    res[f"(a) by task: {fam} (cm)"] = err_block(np.flatnonzero(gt["scope"] == fam))
json.dump(res, open(W.O / "pair_s0_A2.json", "w"), indent=1)
for k, v in res.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
