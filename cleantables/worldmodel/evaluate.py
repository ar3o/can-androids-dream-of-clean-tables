"""Held-out evaluation of the world model: each fold's ensemble predicts the samples of its held-out episodes.

    python -m cleantables.worldmodel.evaluate --run imitation

Writes runs/worldmodel/<run>/pred_heldout_{both,3d_only,vjepa_only}.npz and results.json, per task and pooled.
Values are [mean, 2.5 %, 97.5 %, n] with a bootstrap over episodes, or [mean, n] for per-episode rank correlations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from cleantables.paths import WM_RUNS

from .config import load_train_config
from .dataset import DATASET, N_FOLDS, TASKS, Dataset, get_batch
from .model import MODE_3D, N_PATH, WorldModel
from .train import END_OFFSET_UNIT, checkpoint_name, load_models

log = logging.getLogger(__name__)
MODES = ((0, "both"), (1, "3d_only"), (2, "vjepa_only"))
BOOTSTRAP_DRAWS = 2000
MIN_EPISODE_SNAPSHOTS = 6


@torch.no_grad()
def predict(
    models: list[WorldModel],
    G: dict,
    idx: np.ndarray,
    device: str = "cuda",
    K: int = 4,
    mode: int = 0,
    batch_size: int = 64,
) -> dict:
    """Ensemble predictions for rows idx. Paths: K samples per member, given the true mover and target (`paths`) or
    the ensemble's own picks (`paths_pred`), in metres relative to the mover's start."""
    out = {}
    for s in range(0, len(idx), batch_size):
        ii = torch.as_tensor(idx[s : s + batch_size], device=device)
        b = get_batch(G, ii)
        B = len(ii)
        md = torch.full((B,), mode, device=device, dtype=torch.long)
        md = torch.where(b["vj_ok"].bool(), md, torch.full_like(md, MODE_3D))
        p_mover, p_target, off_true, paths, paths_pred, prog, prog_rev = ([] for _ in range(7))
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E = m.encode(b, md)
                mover_logits, _ = m.pointers(E, b["mover"])
                mover_pick = mover_logits.argmax(-1)
                _, target_logits = m.pointers(E, mover_pick)
                target_pick = target_logits.argmax(-1)
                target_true = torch.where(b["target"] >= 0, b["target"], target_pick)
                c_true, _, mdn_true = m.heads_from(E, b["mover"], target_true)
                c_pred, _, _ = m.heads_from(E, mover_pick, target_pick)
                paths_true_m = m.sample_paths_anchored(E, c_true, K=K)
                paths_pred_m = m.sample_paths_anchored(E, c_pred, K=K)
                pv = torch.sigmoid(m.clip_progress_logit(E))
                b_rev = dict(b)
                b_rev["vj_tok"] = b["vj_tok"].flip(1)
                pv_rev = torch.sigmoid(m.clip_progress_logit(m.encode(b_rev, md)))
            p_mover.append(F.softmax(mover_logits.float(), -1))
            p_target.append(F.softmax(target_logits.float(), -1))
            top = mdn_true["logit"].float().argmax(-1)
            off_true.append(
                END_OFFSET_UNIT * torch.gather(mdn_true["mu"].float(), 1, top[:, None, None].expand(B, 1, 3)).squeeze(1)
            )
            paths.append(paths_true_m.float())
            paths_pred.append(paths_pred_m.float())
            prog.append(pv.float())
            prog_rev.append(pv_rev.float())
        stack = torch.stack
        spread = (
            stack(paths).mean(2).std(0)[..., :3].norm(dim=-1).mean(-1)
            if len(models) > 1
            else torch.zeros(B, device=device)
        )
        r = dict(
            p_mover=stack(p_mover).mean(0),
            p_target=stack(p_target).mean(0),
            off_orc=stack(off_true).mean(0),
            paths=torch.cat(paths, 1),
            paths_pred=torch.cat(paths_pred, 1),
            prog_video=stack(prog).mean(0),
            prog_video_rev=stack(prog_rev).mean(0),
            sp_path=spread,
        )
        for k, v in r.items():
            out.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def bootstrap_mean(vals, groups, n: int = BOOTSTRAP_DRAWS, seed: int = 0) -> list:
    """[mean, 2.5 %, 97.5 %, n] with episodes resampled."""
    vals = np.asarray(vals, float)
    groups = np.asarray(groups)
    ok = np.isfinite(vals)
    vals, groups = vals[ok], groups[ok]
    if len(vals) == 0:
        return [float("nan"), float("nan"), float("nan"), 0]
    ug = sorted(set(groups))
    sums = np.array([vals[groups == g].sum() for g in ug])
    cnt = np.array([(groups == g).sum() for g in ug])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        q = rng.integers(0, len(ug), len(ug))
        draws.append(sums[q].sum() / max(cnt[q].sum(), 1))
    return [float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), int(len(vals))]


def _sample_metrics(d: dict, idx: np.ndarray, P: dict, a: int) -> dict:
    """Per-sample values of the reported metrics (cm for distances)."""
    out = {}
    j = idx[a]
    geo, om, kind = d["obj_geo"][j], d["obj_mask"][j].astype(bool), d["obj_kind"][j]
    movable = om & (kind == 0)
    n_movable = movable.sum()
    mv, tg = d["mover"][j], d["target"][j]
    pick = P["p_mover"][a].argmax()
    if n_movable > 1:
        remaining = d["mover_set"][j] & movable
        if remaining.any():
            out["mover_in_remaining_set"] = float(remaining[pick])
            out["mover_in_set_rand"] = float(remaining.sum() / n_movable)
        if mv >= 0:
            out["mover_top1"] = float(pick == mv)
            out["mover_rand"] = 1.0 / n_movable
    if mv >= 0 and tg >= 0:
        eo = d["end_off"][j]
        if np.isfinite(eo).all():
            out["end_err_oracle_target_cm"] = 100 * np.linalg.norm(P["off_orc"][a][:2] - eo[:2])
            out["baseline_target_centre_cm"] = 100 * np.linalg.norm(eo[:2])
    po = d["path_obj"][j]
    if mv >= 0 and np.isfinite(po).all():
        rel = po - geo[mv, :3]
        mean_path = P["paths"][a][..., :3].mean(0)
        out["path_ade_mean_cm"] = 100 * np.linalg.norm(mean_path - rel, axis=-1).mean()
        line = rel[0] + np.linspace(0, 1, N_PATH)[:, None] * (rel[-1] - rel[0])
        out["baseline_line_oracle_end_cm"] = 100 * np.linalg.norm(line - rel, axis=-1).mean()
        out["baseline_stay_cm"] = 100 * np.linalg.norm(rel[0][None] - rel, axis=-1).mean()
    return out


def _progress_metrics(d: dict, idx: np.ndarray, P: dict) -> dict:
    """Rank correlation of the predicted progress with its label inside each episode, for the real clips, the reversed
    clips (should be negative) and a shuffled-label control, and how often the real clip beats the reversed one."""
    episode = d["video"][idx]
    ok = d["prog_video_ok"][idx].astype(bool)
    pv, pv_rev, label = P["prog_video"], P["prog_video_rev"], d["prog_video"][idx]
    rng = np.random.default_rng(0)
    fwd, rev, shuffled, above = [], [], [], []
    for ep in sorted(set(episode[ok])):
        i = np.flatnonzero((episode == ep) & ok)
        i = i[np.argsort(d["snap_k"][idx][i])]
        if len(i) < MIN_EPISODE_SNAPSHOTS or np.ptp(label[i]) == 0:
            continue
        fwd.append(spearmanr(pv[i], label[i])[0])
        rev.append(spearmanr(pv_rev[i], label[i][::-1])[0])  # a reversed video's time runs backwards
        shuffled.append(spearmanr(pv[i], rng.permutation(label[i]))[0])
        above.append(float((pv[i] > pv_rev[i]).mean()))
    out = dict(
        prog_video_spearman_fwd=[float(np.mean(fwd)), len(fwd)],
        prog_video_spearman_reversed_video=[float(np.mean(rev)), len(rev)],
        prog_video_shuffled_control_spearman=[float(np.mean(shuffled)), len(shuffled)],
        prog_video_fwd_clip_scores_above_reversed_clip=[float(np.mean(above)), len(above)],
    )
    y, p = d["prog_video"][idx], P["prog_video"]
    before, after = y == 0, y == 1
    out["prog_mean_pred_before_start(label0)"] = float(p[before].mean()) if before.any() else None
    out["prog_mean_pred_after_end(label1)"] = float(p[after].mean()) if after.any() else None
    return out


def metrics(d: dict, idx: np.ndarray, P: dict) -> dict:
    rows = [_sample_metrics(d, idx, P, a) for a in range(len(idx))]
    groups = d["group"][idx]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = {}
    for k in keys:
        have = [a for a, r in enumerate(rows) if k in r]
        out[k] = bootstrap_mean([rows[a][k] for a in have], groups[have])
    if d["prog_video_ok"][idx].any():
        out.update(_progress_metrics(d, idx, P))
    return {k: (list(np.round(v, 4)) if isinstance(v, list) else v) for k, v in out.items()}


def evaluate(
    run_dir: Path, dataset: Path = DATASET, seeds=(0, 1, 2), samples: int = 12, device: str = "cuda", seed: int = 0
) -> dict:
    torch.manual_seed(seed)
    data = Dataset(dataset)
    G = data.to(device)
    d = data.d
    results = {}
    for mode, name in MODES:
        all_idx, all_P = [], []
        for f in range(N_FOLDS):
            models = load_models([run_dir / checkpoint_name(f, s) for s in seeds], device)
            test = np.flatnonzero(d["fold"] == f)
            all_P.append(predict(models, G, test, device, K=max(1, samples // len(models)), mode=mode))
            all_idx.append(test)
            del models
            torch.cuda.empty_cache()
        idx = np.concatenate(all_idx)
        P = {k: np.concatenate([p[k] for p in all_P]) for k in all_P[0]}
        np.savez_compressed(run_dir / f"pred_heldout_{name}.npz", idx=idx, **P)
        for task in TASKS + ("all",):
            sel = np.ones(len(idx), bool) if task == "all" else d["dataset"][idx] == task
            if sel.any():
                results[f"{task}_{name}"] = metrics(d, idx[sel], {k: v[sel] for k, v in P.items()})
        log.info("mode %s done", name)
    json.dump(results, open(run_dir / "results.json", "w"), indent=1)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out evaluation of the world model (all folds, all modality modes).")
    ap.add_argument("--run", default="imitation", help="run name under runs/worldmodel/")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_train_config()
    res = evaluate(WM_RUNS / a.run, a.dataset, cfg.seeds, cfg.eval_samples)
    for task in TASKS:
        r = res.get(f"{task}_both", {})
        log.info(
            "%s: next object still to move %s (chance %s), progress rank corr %s, real above reversed %s",
            task,
            r.get("mover_in_remaining_set", [None])[0],
            r.get("mover_in_set_rand", [None])[0],
            r.get("prog_video_spearman_fwd", [None])[0],
            r.get("prog_video_fwd_clip_scores_above_reversed_clip", [None])[0],
        )


if __name__ == "__main__":
    main()
