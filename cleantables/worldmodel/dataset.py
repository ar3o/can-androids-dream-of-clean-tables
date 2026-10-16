"""World-model dataset: the snapshot schema, saving, loading onto the GPU, and the held-out folds.

One sample = one tracked keyframe of an episode: object / hand / V-JEPA tokens at that moment, the progress label, and
(when a tracked move follows) the next move's mover, destination, end pose and path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cleantables.paths import WM_DATA

DATASET = WM_DATA / "dataset.npz"
MAX_OBJECTS = 24
TABLE, LEAVES = 24, 25
RELATIONS = {"inside": 0, "on_top_of": 1, "next_to": 2, "table": 3, "leaves": 4, "slide_lift": 5}
KINDS = {"movable": 0, "table": 1, "fixture": 2}
N_FOLDS = 5
# All fruit episodes come from this one recording, so they are held out one episode at a time; every other video is held
# out as a whole (its tools and stacking episodes together).
PER_EPISODE_VIDEO = "IMG_2516"
TASKS = ("fruit", "tools", "stack_boxes")

# key -> (per-sample shape, dtype, pad value)
SCHEMA = {
    "obj_geo": ((MAX_OBJECTS, 8), np.float32, 0),
    "obj_txt": ((MAX_OBJECTS, 384), np.float16, 0),
    "obj_crop": ((MAX_OBJECTS, 768), np.float16, 0),
    "obj_crop_ok": ((MAX_OBJECTS,), np.uint8, 0),
    "obj_mask": ((MAX_OBJECTS,), np.uint8, 0),
    "obj_kind": ((MAX_OBJECTS,), np.int8, -1),
    "obj_id": ((MAX_OBJECTS,), np.int32, -1),
    "hand_geo": ((2, 13), np.float32, 0),
    "hand_q16": ((2, 16), np.float32, 0),
    "hand_q_ok": ((2,), np.uint8, 0),
    "vj_tok": ((32, 1024), np.float16, 0),
    "vj_tok_flip": ((32, 1024), np.float16, 0),
    "vj_ok": ((), np.uint8, 0),
    "vj_future_glob": ((1024,), np.float16, 0),
    "vj_future_ok": ((), np.uint8, 0),
    "mover": ((), np.int16, -1),
    "target": ((), np.int16, -1),
    "relation": ((), np.int8, -1),
    "end_pose": ((4,), np.float32, 0),
    "end_ok": ((), np.uint8, 0),
    "end_off": ((3,), np.float32, 0),
    "path_obj": ((32, 3), np.float32, 0),
    "path_hand": ((32, 3), np.float32, 0),
    "path_hand_ok": ((32,), np.uint8, 0),
    "path_dt": ((), np.float32, 0),
    "prog_steps": ((32,), np.float32, 0),
    "prog_video": ((), np.float32, 0),
    "prog_video_ok": ((), np.uint8, 0),
    "time_to_move": ((), np.float32, 0),
    "table_idx": ((), np.int16, -1),
    "snap_k": ((), np.int32, -1),
    "ev_id": ((), np.int32, -1),
    "mover_set": ((MAX_OBJECTS,), np.uint8, 0),
    "prog_time": ((), np.float32, 0),
}
STRING_KEYS = ["dataset", "video", "group", "split", "task_text"]

# dtype of each array once loaded (strings stay numpy-only)
_LOAD_DTYPES = {
    "obj_geo": np.float32,
    "obj_txt": np.float32,
    "obj_crop": np.float32,
    "obj_crop_ok": bool,
    "obj_mask": bool,
    "obj_kind": np.int64,
    "hand_geo": np.float32,
    "hand_q16": np.float32,
    "hand_q_ok": bool,
    "vj_tok": np.float16,
    "vj_tok_flip": np.float16,
    "vj_ok": bool,
    "mover": np.int64,
    "target": np.int64,
    "relation": np.int64,
    "end_off": np.float32,
    "end_pose": np.float32,
    "path_obj": np.float32,
    "path_hand": np.float32,
    "path_hand_ok": bool,
    "path_dt": np.float32,
    "vj_future_glob": np.float32,
    "prog_steps": np.float32,
    "prog_video": np.float32,
    "prog_video_ok": bool,
    "vj_future_ok": bool,
    "time_to_move": np.float32,
    "ev_id": np.int64,
    "table_idx": np.int64,
    "snap_k": np.int64,
    "mover_set": bool,
}


def blank_sample() -> dict:
    s = {k: np.full(shape, pad, dt) for k, (shape, dt, pad) in SCHEMA.items()}
    s.update({k: "" for k in STRING_KEYS})
    return s


def resample_path(P, n: int = 32) -> np.ndarray:
    P = np.asarray(P, np.float32)
    t = np.linspace(0, 1, len(P))
    u = np.linspace(0, 1, n)
    return np.stack([np.interp(u, t, P[:, i]) for i in range(P.shape[1])], 1).astype(np.float32)


def save_dataset(samples: list, path: Path, extra_meta: dict | None = None) -> dict:
    """Stack the samples into one .npz and write a summary <name>_meta.json next to it."""
    out = {k: np.stack([s[k] for s in samples]) for k in SCHEMA}
    for k in STRING_KEYS:
        out[k] = np.array([s[k] for s in samples])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    meta = dict(name=path.stem, n=len(samples), keys={k: [list(v.shape), str(v.dtype)] for k, v in out.items()})
    for k in ("split", "group", "video", "task_text"):
        u, c = np.unique(out[k], return_counts=True)
        meta[f"count_{k}"] = dict(zip(u.tolist(), c.tolist(), strict=False))
    u, c = np.unique(out["relation"], return_counts=True)
    meta["relation_hist"] = dict(zip(u.tolist(), c.tolist(), strict=False))
    for k in ("vj_ok", "vj_future_ok", "prog_video_ok", "end_ok", "path_hand_ok", "hand_q_ok"):
        meta[f"share_{k}"] = float(out[k].astype(bool).mean())
    meta["share_hand_valid"] = float((out["hand_geo"][:, :, 0] > 0).any(1).mean())
    meta["nan_keys"] = [
        k
        for k in SCHEMA
        if np.issubdtype(out[k].dtype, np.floating) and not np.isfinite(out[k].astype(np.float32)).all()
    ]
    meta.update(extra_meta or {})
    json.dump(meta, open(path.with_name(f"{path.stem}_meta.json"), "w"), indent=1)
    return meta


def hashed_text_embedding(strings, dim: int = 384) -> np.ndarray:
    """Deterministic unit Gaussian vector per distinct string (seeded by its md5). Used for the task token of the
    training data: it only has to tell a handful of instructions apart, and the trained checkpoints expect it."""
    table = {}
    for s in set(strings):
        rng = np.random.default_rng(int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16))
        v = rng.standard_normal(dim)
        table[s] = v / np.linalg.norm(v)
    return np.stack([table[s] for s in strings]).astype(np.float32)


def episode_folds(keys) -> dict:
    """Fold of every hold-out key: videos and single fruit episodes are each sorted and dealt round-robin."""
    keys = sorted(set(keys))
    whole = [k for k in keys if not k.startswith(PER_EPISODE_VIDEO)]
    single = [k for k in keys if k.startswith(PER_EPISODE_VIDEO)]
    return {k: i % N_FOLDS for i, k in enumerate(whole)} | {k: i % N_FOLDS for i, k in enumerate(single)}


def holdout_key(video: str, episode: str) -> str:
    return episode if video == PER_EPISODE_VIDEO else video


class Dataset:
    """The dataset as numpy arrays (`d`) and CPU tensors (`T`).

    `d["video"]` holds the episode id (progress is ranked within an episode); the recording is `d["video_name"]`.
    """

    def __init__(self, path: Path = DATASET):
        z = np.load(path, allow_pickle=True)
        d = {k: np.asarray(z[k]).astype(dt) for k, dt in _LOAD_DTYPES.items()}
        for k in ("dataset", "group", "split"):
            d[k] = np.asarray(z[k]).astype(str)
        d["task_text"] = np.asarray(z["task_text"]).astype(str)
        d["task_emb"] = hashed_text_embedding(list(d["task_text"]))
        d["end_ok"] = z["end_ok"].astype(bool)
        d["obj_id"] = z["obj_id"].astype(np.int64)  # tracked object behind each slot (videos only)
        d["prog_time"] = z["prog_time"]
        d["video_name"] = np.asarray(z["video"]).astype(str)
        d["video"] = d["group"].copy()
        self.d = d
        self.n = len(d["mover"])
        tensor_keys = [
            k for k, v in d.items() if v.dtype.kind not in "USO" and k not in ("end_ok", "prog_time", "obj_id")
        ]
        self.T = {k: torch.from_numpy(d[k]) for k in tensor_keys}
        episode_ids = {g: i for i, g in enumerate(sorted(set(d["group"])))}
        self.T["video_id"] = torch.tensor([episode_ids[g] for g in d["group"]])
        fold_of = episode_folds(holdout_key(v, g) for v, g in zip(d["video_name"], d["group"], strict=False))
        self.d["fold"] = np.array(
            [fold_of[holdout_key(v, g)] for v, g in zip(d["video_name"], d["group"], strict=False)]
        )

    def to(self, device: str) -> dict:
        return {k: v.to(device) for k, v in self.T.items()}


def get_batch(G: dict, idx: torch.Tensor) -> dict:
    """Rows idx of the on-device tensors, with half floats widened and small ints/bytes turned into long/bool."""
    b = {k: v[idx] for k, v in G.items()}
    for k, v in b.items():
        if v.dtype == torch.float16:
            b[k] = v.float()
        elif v.dtype == torch.uint8:
            b[k] = v.bool()
        elif v.dtype in (torch.int8, torch.int16, torch.int32):
            b[k] = v.long()
    b["obj_kind"] = b["obj_kind"].clamp(0, 2)
    return b
