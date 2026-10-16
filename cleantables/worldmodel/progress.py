"""Task-progress diagnosis and monotone calibration of the progress head.

Every fold's ensemble predicts progress for every snapshot, so each episode is seen both by models that trained on it
and by the fold that held it out. An isotonic map (predicted progress -> label) is fitted per fold on that fold's
training episodes only and applied to its held-out episodes.

    python -m cleantables.worldmodel.progress --run imitation
        -> runs/worldmodel/<run>/progress_by_fold.npz, progress_calibration.json, progress_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from cleantables.paths import WM_RUNS

from .dataset import DATASET, N_FOLDS, TASKS, Dataset, get_batch
from .model import MODE_3D, WorldModel
from .train import checkpoint_name, load_models

log = logging.getLogger(__name__)
EDGE_SNAPSHOTS = 3  # snapshots averaged for the start and end values of an episode
MIN_EPISODE_SNAPSHOTS = 6


@torch.no_grad()
def predict_progress(models: list[WorldModel], G: dict, idx: np.ndarray, mode: int = 0, batch_size: int = 128):
    """Ensemble-mean progress of the real clip and of the same clip played backwards."""
    fwd, rev = [], []
    for s in range(0, len(idx), batch_size):
        b = get_batch(G, torch.as_tensor(idx[s : s + batch_size], device="cuda"))
        md = torch.full((len(b["mover"]),), mode, device="cuda", dtype=torch.long)
        md = torch.where(b["vj_ok"].bool(), md, torch.full_like(md, MODE_3D))
        b_rev = dict(b)
        b_rev["vj_tok"] = b["vj_tok"].flip(1)
        pf, pr = [], []
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pf.append(torch.sigmoid(m.clip_progress_logit(m.encode(b, md))).float())
                pr.append(torch.sigmoid(m.clip_progress_logit(m.encode(b_rev, md))).float())
        fwd.append(torch.stack(pf).mean(0).cpu().numpy())
        rev.append(torch.stack(pr).mean(0).cpu().numpy())
    return np.concatenate(fwd), np.concatenate(rev)


def predictions_by_fold(run_dir: Path, data: Dataset, seeds=(0, 1, 2)) -> dict:
    """pred[f, i]: progress of snapshot i by fold f's ensemble (all snapshots, trained-on and held-out)."""
    G = data.to("cuda")
    everything = np.arange(data.n)
    fwd, rev = np.zeros((N_FOLDS, data.n), np.float32), np.zeros((N_FOLDS, data.n), np.float32)
    for f in range(N_FOLDS):
        models = load_models([run_dir / checkpoint_name(f, s) for s in seeds])
        fwd[f], rev[f] = predict_progress(models, G, everything)
        log.info("fold %d predicted", f)
    return dict(fwd=fwd, rev=rev)


def episode_metrics(d: dict, idx: np.ndarray, p: np.ndarray, p_rev: np.ndarray | None = None) -> dict:
    """Means over episodes of the within-episode rank correlation, MAE, start and end values, and the overall mean
    prediction where the label is 0 (before the task) and 1 (after it)."""
    label = d["prog_video"][idx]
    rho, mae, start, end, start_label, end_label, above = [], [], [], [], [], [], []
    for ep in sorted(set(d["group"][idx])):
        i = np.flatnonzero(d["group"][idx] == ep)
        i = i[np.argsort(d["snap_k"][idx][i])]
        if len(i) < MIN_EPISODE_SNAPSHOTS or np.ptp(label[i]) == 0:
            continue
        rho.append(spearmanr(p[i], label[i])[0] if np.ptp(p[i]) > 0 else 0.0)
        mae.append(np.abs(p[i] - label[i]).mean())
        start.append(p[i][:EDGE_SNAPSHOTS].mean())
        end.append(p[i][-EDGE_SNAPSHOTS:].mean())
        start_label.append(label[i][:EDGE_SNAPSHOTS].mean())
        end_label.append(label[i][-EDGE_SNAPSHOTS:].mean())
        if p_rev is not None:
            above.append((p[i] > p_rev[i]).mean())
    out = dict(
        episodes=len(rho),
        spearman=float(np.mean(rho)),
        mae=float(np.mean(mae)),
        start=float(np.mean(start)),
        end=float(np.mean(end)),
        start_label=float(np.mean(start_label)),
        end_label=float(np.mean(end_label)),
        pred_where_label0=float(p[label == 0].mean()),
        pred_where_label1=float(p[label == 1].mean()),
    )
    if p_rev is not None:
        out["fwd_above_reversed"] = float(np.mean(above))
    return out


def label_statistics(d: dict, idx: np.ndarray) -> dict:
    """How the count label looks: share of snapshots at 0, at 1 and in between, steps per episode, and its rank
    correlation with plain time through the labelled core."""
    label, time_label = d["prog_video"][idx], d["prog_time"][idx]
    steps, rho = [], []
    for ep in sorted(set(d["group"][idx])):
        i = np.flatnonzero(d["group"][idx] == ep)
        steps.append(len(np.unique(label[i])) - 1)
        if np.ptp(label[i]) > 0 and np.ptp(time_label[i]) > 0:
            rho.append(spearmanr(label[i], time_label[i])[0])
    return dict(
        share_label0=float((label == 0).mean()),
        share_label1=float((label == 1).mean()),
        share_between=float(((label > 0) & (label < 1)).mean()),
        median_moves_per_episode=float(np.median(steps)),
        episodes_without_tracked_move=int(sum(s == 0 for s in steps)),
        spearman_count_vs_time=float(np.mean(rho)),
    )


def fit_calibrators(d: dict, fwd: np.ndarray) -> list[IsotonicRegression]:
    """One increasing map per fold, fitted on the snapshots that fold trained on."""
    out = []
    for f in range(N_FOLDS):
        train = d["fold"] != f
        out.append(
            IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
                fwd[f][train], d["prog_video"][train]
            )
        )
    return out


def heldout(fold_values: np.ndarray, d: dict) -> np.ndarray:
    """Each snapshot's value from the fold that held it out."""
    return fold_values[d["fold"], np.arange(len(d["fold"]))]


def calibrator_to_json(c: IsotonicRegression) -> dict:
    return dict(x=c.X_thresholds_.tolist(), y=c.y_thresholds_.tolist())


def apply_calibration(table: dict, p: np.ndarray) -> np.ndarray:
    """Piecewise-linear isotonic map stored by `calibrator_to_json`."""
    return np.interp(p, table["x"], table["y"])


def report(run_dir: Path, dataset: Path = DATASET) -> dict:
    data = Dataset(dataset)
    d = data.d
    cache = run_dir / "progress_by_fold.npz"
    if cache.exists():
        P = dict(np.load(cache))
    else:
        P = predictions_by_fold(run_dir, data)
        np.savez_compressed(cache, **P)
    fwd, rev = P["fwd"], P["rev"]
    calibrators = fit_calibrators(d, fwd)
    cal = np.stack([np.clip(c.predict(fwd[f]), 0, 1) for f, c in enumerate(calibrators)])
    cal_rev = np.stack([np.clip(c.predict(rev[f]), 0, 1) for f, c in enumerate(calibrators)])
    out = {}
    for task in TASKS + ("all",):
        rows = np.ones(data.n, bool) if task == "all" else d["dataset"] == task
        idx = np.flatnonzero(rows)
        r = dict(labels=label_statistics(d, idx))
        r["heldout"] = episode_metrics(d, idx, heldout(fwd, d)[idx], heldout(rev, d)[idx])
        r["heldout_calibrated"] = episode_metrics(d, idx, heldout(cal, d)[idx], heldout(cal_rev, d)[idx])
        # the same episodes scored by the folds that trained on them (bootstrap-resampled, so mostly seen)
        trained = [
            episode_metrics(d, np.flatnonzero(rows & (d["fold"] != f)), fwd[f][rows & (d["fold"] != f)])
            for f in range(N_FOLDS)
        ]
        r["trained_on"] = {k: float(np.mean([t[k] for t in trained])) for k in trained[0]}
        out[task] = r
    json.dump(out, open(run_dir / "progress_report.json", "w"), indent=1)
    json.dump([calibrator_to_json(c) for c in calibrators], open(run_dir / "progress_calibration.json", "w"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose and calibrate the progress head on held-out folds.")
    ap.add_argument("--run", default="imitation", help="run name under runs/worldmodel/ with fold checkpoints")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for task, r in report(WM_RUNS / a.run, a.dataset).items():
        for part in ("trained_on", "heldout", "heldout_calibrated"):
            m = r[part]
            log.info(
                "%-11s %-18s rank %.2f  MAE %.3f  start %.2f end %.2f  (labels %.2f / %.2f)",
                task,
                part,
                m["spearman"],
                m["mae"],
                m["start"],
                m["end"],
                m["start_label"],
                m["end_label"],
            )


if __name__ == "__main__":
    main()
