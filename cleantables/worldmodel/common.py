"""Shared helpers for evaluating the base world-model runs (runs/uwm/...). Reads models and data only; outputs go to OUT.

Splits:
  human held-out: the 5-fold protocol of cleantables.worldmodel.tasks3_train. Each box video is held out whole (tools and
    stacking episodes together); the 14 fruit episodes (all from IMG_2516) are held out one by one, so held-out fruit shares
    the video, table and objects with training. Only the base run (tasks3_v4) has fold models; the physics-feedback runs
    (pf, pf2, pf3, regpf) exist only as fold -1 (trained on all 45 episodes) and have no held-out human numbers.
  twin held-out: the 7 held-out robot twins (IMG_2546/2547/2548_t01, IMG_2516_t11..t14) were never used for physics feedback,
    but the fold -1 model that every physics-feedback run starts from was trained on their human videos. Base fold models
    (fold k never saw episode e) give leak-free twin numbers for the base model only.
  instruction: tasks3c.npz stores no task_emb; training recomputed it with store.embed_text, which falls back to an md5-hashed
    random vector when sentence_transformers is not installed. It was not installed in the training envs, so the models were
    trained on hashed instruction vectors, and evaluation pins the same encoding (HASHED). The twin bridge (wm_tokens.npz)
    feeds MiniLM vectors, which are out of distribution; the counterfactual tests report both.
"""
from __future__ import annotations

from cleantables import CT_DATA
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("WM_EVAL_OUT", CT_DATA + "/wm_eval"))
RUNS = ROOT / "runs/uwm"
os.environ.setdefault("TASKS3_DATA", "tasks3c")
TASKS = ("fruit", "tools", "stack_boxes")
TASK_TEXT = {"fruit": "Put the fruit into the fruit bowl.", "tools": "Put the tools into the toolbox.",
             "stack_boxes": "Stack the boxes into a tower on the table."}
HELD_TWINS = ["IMG_2546_t01", "IMG_2547_t01", "IMG_2548_t01", "IMG_2516_t11", "IMG_2516_t12", "IMG_2516_t13", "IMG_2516_t14"]
NF = 5
SPLIT_NOTE = ("human held-out = 5 folds (box videos whole, fruit episodes one by one from the same video IMG_2516); only the base run has "
              "fold models. Twin held-out = 7 twins never used by physics feedback, but their human videos ARE in the fold -1 imitation "
              "model (leak); base fold models give leak-free twin numbers.")


def hashed(s, dim=384):
    """store.embed_text's hashed fallback, which the models were trained on."""
    r = np.random.default_rng(int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16))
    v = r.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _pin_hashed():
    from cleantables.worldmodel import store
    store.embed_text = lambda strings, dim=384: np.stack([hashed(s, dim) for s in strings]).astype(np.float32)


def load_store():
    """tasks3c store as built by tasks3_train.load (fold column, episode as 'video'), on the CPU, with the hashed instruction pinned."""
    _pin_hashed()
    from cleantables.worldmodel.base_train import prep_store
    from cleantables.worldmodel.store import ROOT as R
    S = prep_store([os.environ["TASKS3_DATA"]])
    S.d["video_name"] = S.d["video"].copy()
    Z = np.load(R / f"data/uwm/{os.environ['TASKS3_DATA']}.npz")
    S.d["end_ok"] = Z["end_ok"].astype(bool)
    ep = S.d["group"]
    ids = {g: i for i, g in enumerate(sorted(set(ep)))}
    S.T["video_id"] = torch.tensor([ids[g] for g in ep])
    key = np.where(S.d["video_name"] == "IMG_2516", ep, S.d["video_name"])
    uk = sorted(set(key))
    box = [k for k in uk if not k.startswith("IMG_2516")]; fru = [k for k in uk if k.startswith("IMG_2516")]
    fold_of = {k: i % NF for i, k in enumerate(box)} | {k: i % NF for i, k in enumerate(fru)}
    S.d["fold"] = np.array([fold_of[k] for k in key])
    return S


def twin_fold(ep):
    """Held-out fold of a twin episode, as in base_bridge.check and physics.feedback_base.score_run."""
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(t["id"] for t in T if t["video"] == "IMG_2516")
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    return fold_of[ep] if ep.startswith("IMG_2516") else fold_of[ep.split("_t")[0]]


def batch(S, idx, dev="cuda"):
    """As cleantables.worldmodel.base_train.get_batch, but indexes the CPU tensors first so the 0.8 GB store stays off the GPU."""
    ii = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    b = {k: v[ii].to(dev) for k, v in S.T.items()}
    for k in b:
        if b[k].dtype == torch.float16:
            b[k] = b[k].float()
        elif b[k].dtype == torch.uint8:
            b[k] = b[k].bool()
        elif b[k].dtype in (torch.int8, torch.int16, torch.int32):
            b[k] = b[k].long()
    b["obj_kind"] = b["obj_kind"].clamp(0, 2)
    return b


def ckpts(run, fold):
    d = RUNS / run if (RUNS / run).exists() else OUT / run
    return [d / f"fold{fold}_seed{s}.pt" for s in range(3)]


def load_models(paths, dev="cuda"):
    from cleantables.worldmodel.base_model import UWM
    ms = []
    for p in paths:
        m = UWM(path_resid=True).to(dev); m.load_state_dict(torch.load(p, map_location=dev)); ms.append(m.eval())
    return ms


def boot_ci(vals, groups, n=2000, seed=0, stat=np.mean):
    """Episode-level (cluster) bootstrap 95 % CI of the mean."""
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ok = np.isfinite(vals); vals, groups = vals[ok], groups[ok]
    if len(vals) == 0:
        return [float("nan")] * 3 + [0]
    ug = sorted(set(groups)); gi = [np.flatnonzero(groups == g) for g in ug]
    r = np.random.default_rng(seed); bs = []
    for _ in range(n):
        q = r.integers(0, len(ug), len(ug)); bs.append(stat(np.concatenate([vals[gi[j]] for j in q])))
    return [float(stat(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(vals))]


def jdump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(path, "w"), indent=1, default=conv)


def gpu_free_mb():
    try:
        import subprocess
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return -1


def log(msg, f=None):
    print(msg, flush=True)
    if f is not None:
        with open(f, "a") as fh:
            fh.write(msg + "\n")
