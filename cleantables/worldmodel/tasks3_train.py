"""One base world model (V-JEPA 2 + 3D object tokens + task prompt) for three tasks: fruit -> bowl, tools -> toolbox, stack the boxes.

Data: data/uwm/tasks3.npz. Model, losses and augmentation are those of cleantables.worldmodel.base_train.
Progress ranking pairs are taken within an episode (video_id := episode), because one box video holds two tasks.
Hold-out: 5 folds; a box video is held out as a whole (its tools and stacking episodes together), fruit episodes
(all from one video) one by one.
  $VLA_PY -m cleantables.worldmodel.tasks3_train train --fold F --seeds 0,1,2      (F = 0..4, or -1 = all data, final model)
  $VLA_PY -m cleantables.worldmodel.tasks3_train eval
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from cleantables.worldmodel.base_eval import fmt, metrics, predict
from cleantables.worldmodel.store import ROOT
from cleantables.worldmodel.base_train import prep_store, to_dev, train_model

import os
RUN = ROOT / "runs/uwm" / os.environ.get("TASKS3_RUN", "tasks3")
CFG = json.loads(os.environ.get("TASKS3_CFG", "{}"))  # extra train cfg, e.g. {"prog_loss": "bce"}
DATA = os.environ.get("TASKS3_DATA", "tasks3")       # dataset name under data/uwm
NF = 5
TASKS = ("fruit", "tools", "stack_boxes")


def load(dev="cuda"):
    S = prep_store([DATA])
    S.d["video_name"] = S.d["video"].copy()
    Z = np.load(ROOT / f"data/uwm/{DATA}.npz")
    S.d["end_ok"] = Z["end_ok"].astype(bool)
    if "prog_time" in Z.files:
        S.d["prog_time"] = Z["prog_time"]
    ep = S.d["group"]
    ids = {g: i for i, g in enumerate(sorted(set(ep)))}
    S.T["video_id"] = torch.tensor([ids[g] for g in ep])
    S.d["video"] = ep.copy()                                    # metrics() ranks progress within 'video' -> within the episode
    key = np.where(S.d["video_name"] == "IMG_2516", ep, S.d["video_name"])
    uk = sorted(set(key))
    box = [k for k in uk if not k.startswith("IMG_2516")]; fru = [k for k in uk if k.startswith("IMG_2516")]
    fold_of = {k: i % NF for i, k in enumerate(box)} | {k: i % NF for i, k in enumerate(fru)}
    S.d["fold"] = np.array([fold_of[k] for k in key])
    return S, to_dev(S, dev)


def cmd_train(fold, seeds, steps, only=""):
    S, G = load()
    tr = np.flatnonzero(S.d["fold"] != fold) if fold >= 0 else np.arange(S.n)
    if only:                                                    # single-task ablation: train on that task's episodes only
        tr = tr[S.d["dataset"][tr] == only]
    for s in seeds:
        m = train_model(S, G, tr, dict(steps=steps, **CFG), seed=1000 * (fold + 1) + s, dev="cuda", bootstrap=fold >= 0,
                        log=lambda x: print(f"[f{fold} s{s}] {x}", flush=True))
        torch.save(m.state_dict(), RUN / f"fold{fold}_seed{s}{'_' + only if only else ''}.pt")
        print(f"fold {fold} seed {s} done {m.train_seconds:.0f}s n_train {len(tr)}", flush=True)


def prog_extra(S, idx, P):
    """Progress checks around each labelled move: mean prediction in the 1 s before the start (label 0) and after the end (label 1), and MAE."""
    y = S.d["prog_video"][idx]; p = P["prog_video"]; pr = P["prog_video_rev"]
    k = S.d["snap_k"][idx] / 10.0
    out = dict(prog_mae=float(np.abs(p - y).mean()))
    pre, post = (y == 0), (y == 1)
    out["prog_mean_pred_before_start(label0)"] = float(p[pre].mean()) if pre.any() else None
    out["prog_mean_pred_after_end(label1)"] = float(p[post].mean()) if post.any() else None
    out["prog_mean_pred_core"] = float(p[(y > 0) & (y < 1)].mean())
    out["prog_fwd_above_reversed_clip_share"] = float((p > pr).mean()); out["prog_fwd_minus_reversed_mean"] = float((p - pr).mean())
    # per-episode table
    eps = {}
    from scipy.stats import spearmanr
    for g in sorted(set(S.d["group"][idx])):
        i = S.d["group"][idx] == g; o = np.argsort(k[i])
        yy, pp = y[i][o], p[i][o]
        eps[g] = dict(n=int(i.sum()), spearman=float(spearmanr(pp, yy)[0]) if np.ptp(yy) > 0 and np.ptp(pp) > 0 else None,
                      first=float(pp[:3].mean()), last=float(pp[-3:].mean()))
    out["episodes"] = eps
    return out


def cmd_eval(seeds, K=12, only=""):
    suf = "_" + only if only else ""
    S, G = load()
    res, preds = {}, {}
    for md, nm in ((0, "both"), (1, "3d_only"), (2, "vjepa_only")):
        allidx, allP = [], []
        for f in range(NF):
            ms = []
            for s in seeds:
                from cleantables.worldmodel.base_model import UWM
                m = UWM(path_resid=bool(CFG.get("path_resid"))).cuda(); m.load_state_dict(torch.load(RUN / f"fold{f}_seed{s}{suf}.pt", map_location="cuda")); ms.append(m.eval())
            te = np.flatnonzero((S.d["fold"] == f) & ((S.d["dataset"] == only) if only else True))
            P = predict(ms, G, te, "cuda", K=max(1, K // len(ms)), mode=md)
            allidx.append(te); allP.append(P)
            del ms; torch.cuda.empty_cache()
        idx = np.concatenate(allidx); P = {k: np.concatenate([p[k] for p in allP]) for k in allP[0]}
        np.savez_compressed(RUN / f"pred_heldout_{nm}{suf}.npz", idx=idx, **P)
        for t in ((only,) if only else TASKS + ("all",)):
            sel = np.ones(len(idx), bool) if t == "all" else S.d["dataset"][idx] == t
            if not sel.any():
                continue
            Pt = {k: v[sel] for k, v in P.items()}
            tr = S.d["relation"][S.d["dataset"] == t] if t != "all" else S.d["relation"]
            tr = tr[tr >= 0]; maj = int(np.bincount(tr).argmax()) if len(tr) else None
            r = fmt(metrics(S, idx[sel], Pt, maj)); r.update(prog_extra(S, idx[sel], Pt))
            res[f"{t}_{nm}"] = r
        print(nm, "done", flush=True)
    json.dump(res, open(RUN / f"results{suf}.json", "w"), indent=1)
    keys = ["mover_top1", "mover_rand", "mover_nearest_hand", "target_ok", "target_nearest_obj", "end_err_oracle_target_cm", "baseline_target_centre_cm",
            "baseline_stays_cm", "path_bestofK_cm", "path_ade_mean_cm", "baseline_line_oracle_end_cm", "prog_video_spearman_fwd", "prog_mae",
            "prog_mean_pred_before_start(label0)", "prog_mean_pred_after_end(label1)", "prog_video_fwd_clip_scores_above_reversed_clip", "jepa_cos",
            "jepa_cos_persistence_baseline", "trust_auroc_member_spread_flags_mover_error"]
    for t in TASKS:
        r = res.get(f"{t}_both")
        if r:
            print("==", t, {k: r.get(k) for k in keys}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--only", default="")
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.cmd == "train":
        cmd_train(a.fold, seeds, a.steps, a.only)
    else:
        cmd_eval(seeds, only=a.only)
