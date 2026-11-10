"""Array schema for the base world-model datasets (data/uwm/<name>.npz), with save and load helpers. All arrays are stacked over samples."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/uwm"
MAXO = 24
TABLE, LEAVES = 24, 25
REL = {"inside": 0, "on_top_of": 1, "next_to": 2, "table": 3, "leaves": 4, "slide_lift": 5}
KIND = {"movable": 0, "table": 1, "fixture": 2}

# key -> (per-sample shape, dtype, pad value)
SPEC = {
    "obj_geo": ((MAXO, 8), np.float32, 0), "obj_txt": ((MAXO, 384), np.float16, 0), "obj_crop": ((MAXO, 768), np.float16, 0),
    "obj_crop_ok": ((MAXO,), np.uint8, 0), "obj_mask": ((MAXO,), np.uint8, 0), "obj_kind": ((MAXO,), np.int8, -1),
    "obj_id": ((MAXO,), np.int32, -1),
    "hand_geo": ((2, 13), np.float32, 0), "hand_q16": ((2, 16), np.float32, 0), "hand_q_ok": ((2,), np.uint8, 0),
    "vj_tok": ((32, 1024), np.float16, 0), "vj_tok_flip": ((32, 1024), np.float16, 0), "vj_ok": ((), np.uint8, 0),
    "vj_future_glob": ((1024,), np.float16, 0), "vj_future_ok": ((), np.uint8, 0),
    "mover": ((), np.int16, -1), "target": ((), np.int16, -1), "relation": ((), np.int8, -1),
    "end_pose": ((4,), np.float32, 0), "end_ok": ((), np.uint8, 0), "end_off": ((3,), np.float32, 0),
    "path_obj": ((32, 3), np.float32, 0), "path_hand": ((32, 3), np.float32, 0), "path_hand_ok": ((32,), np.uint8, 0),
    "path_dt": ((), np.float32, 0), "prog_steps": ((32,), np.float32, 0), "prog_video": ((), np.float32, 0),
    "prog_video_ok": ((), np.uint8, 0), "time_to_move": ((), np.float32, 0), "table_idx": ((), np.int16, -1),
    "snap_k": ((), np.int32, -1), "ev_id": ((), np.int32, -1),
    "mover_set": ((MAXO,), np.uint8, 0), "prog_time": ((), np.float32, 0),      # objects still to move in the episode; time-progress label
}
STR = ["dataset", "video", "group", "split", "task_text"]


def blank():
    s = {k: np.full(sh, pad, dt) for k, (sh, dt, pad) in SPEC.items()}
    s.update({k: "" for k in STR})
    return s


def resample(P, n=32):
    P = np.asarray(P, np.float32)
    t = np.linspace(0, 1, len(P)); u = np.linspace(0, 1, n)
    return np.stack([np.interp(u, t, P[:, i]) for i in range(P.shape[1])], 1).astype(np.float32)


def save(samples, name, extra_meta=None):
    out = {k: np.stack([s[k] for s in samples]) for k in SPEC}
    for k in STR:
        out[k] = np.array([s[k] for s in samples])
    np.savez(OUT / f"{name}.npz", **out)
    M = dict(name=name, n=len(samples), keys={k: [list(v.shape), str(v.dtype)] for k, v in out.items()})
    for k in ("split", "group", "video", "task_text"):
        u, c = np.unique(out[k], return_counts=True)
        M[f"count_{k}"] = dict(zip(u.tolist(), c.tolist()))
    u, c = np.unique(out["relation"], return_counts=True)
    M["relation_hist"] = dict(zip(u.tolist(), c.tolist()))
    for k in ("vj_ok", "vj_future_ok", "prog_video_ok", "end_ok", "path_hand_ok", "hand_q_ok"):
        M[f"share_{k}"] = float(out[k].astype(bool).mean())
    M["share_hand_valid"] = float((out["hand_geo"][:, :, 0] > 0).any(1).mean())
    M["nan_keys"] = [k for k in SPEC if np.issubdtype(out[k].dtype, np.floating) and not np.isfinite(out[k].astype(np.float32)).all()]
    M.update(extra_meta or {})
    json.dump(M, open(OUT / f"{name}_meta.json", "w"), indent=1)
    return M
