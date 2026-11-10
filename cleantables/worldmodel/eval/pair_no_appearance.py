"""Paired comparison from stored predictions: our model (seed 0) with appearance tokens off, against seed 0 with appearance
on and against the A2 regression baseline (seed 0), on the same objects. Reports first-frame goal error on put-into tasks
and moved-object end error in scope, with bootstrap CIs over episodes (score.boot, 1,000 draws, rng 0).

  $VLA_PY -m cleantables.worldmodel.eval.pair_no_appearance      -> out/wm/pair_noapp_s0.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cleantables.worldmodel.eval.score as W  # noqa: E402

gt = W.load("gt.npz"); p = W.load(f"pred_{W.REF}.npz"); a2 = W.load("pred_A2_s0.npz"); na = W.load("pred_final_noapp_s0.npz")
now = gt["obj_geo"][..., :3]; tgt = gt["t1_pos"]
GO = {"ours seed 0": p["s0__goal"], "A2 (seed 0)": a2["all__goal"], "ours seed 0, appearance off": W.full(gt, na, "all__goal")}
ok = gt["t1_ok"] & gt["t1_present"] & gt["obj_mask"].astype(bool) & (gt["obj_kind"] == 0)
dmv = np.linalg.norm(tgt - now, axis=-1)
ERR = {k: 100 * np.linalg.norm(g - tgt, axis=-1) for k, g in GO.items()}
EP = np.repeat(gt["ep"][:, None], ok.shape[1], 1)


def ci(ee, f):
    return W.boot(ee, f, np.random.default_rng(0))


def block(rows):
    m = np.zeros_like(ok); m[rows] = (ok & (dmv > W.THR))[rows]
    e = {k: v[m] for k, v in ERR.items()}; ee = EP[m]; off = e["ours seed 0, appearance off"]
    assert np.isfinite(off).all()
    r = dict(n_moved=int(m.sum()), n_ep=int(len(np.unique(ee))))
    for k, v in e.items():
        r[k] = float(np.median(v)); r[k + " ci"] = ci(ee, lambda q, v=v: np.median(v[q]))
    for ref in ("ours seed 0", "A2 (seed 0)"):
        rv = e[ref]
        r[f"diff off - {ref}"] = float(np.median(off) - np.median(rv)); r[f"diff off - {ref} ci"] = ci(ee, lambda q, rv=rv: np.median(off[q]) - np.median(rv[q]))
    return r


ins = np.flatnonzero(gt["scope"] != "")
first = np.array([v[0] for v in W.segments(gt, ["fruit in", "tools in"]).values()])
res = {"first-frame goal error, put-into (cm)": block(first), "moved-object end error, in scope (cm)": block(ins)}
json.dump(res, open(W.O / "pair_noapp_s0.json", "w"), indent=1)
for k, v in res.items():
    print(k); [print("   ", a, (round(b, 2) if isinstance(b, float) else [round(x, 2) for x in b] if isinstance(b, list) else b)) for a, b in v.items()]
