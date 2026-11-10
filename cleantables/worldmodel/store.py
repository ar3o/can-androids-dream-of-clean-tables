"""Data store for the base world model: loads data/uwm/<name>.npz into CPU torch tensors, concatenates datasets and pads
to N=24 objects. Missing optional fields get neutral defaults and validity=False."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/uwm"
N = 24
TABLE, LEAVES = 24, 25
KIND = {"movable": 0, "table": 1, "fixture": 2}


def embed_text(strings, dim=384):
    """Frozen text embedding: MiniLM if sentence-transformers loads offline, else a deterministic hashed Gaussian
    (distinct strings give distinct fixed random vectors; enough to tell a few task phrases apart, but not semantic)."""
    out = {}
    try:
        from sentence_transformers import SentenceTransformer  # noqa
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        u = sorted(set(strings)); e = m.encode(u, normalize_embeddings=True)
        out = {s: e[i] for i, s in enumerate(u)}
    except Exception:
        for s in set(strings):
            r = np.random.default_rng(int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16))
            v = r.standard_normal(dim); out[s] = v / np.linalg.norm(v)
    return np.stack([out[s] for s in strings]).astype(np.float32)


def _pad(a, n, fill=0):
    a = np.asarray(a)
    if a.shape[1] >= n:
        return a[:, :n]
    pw = [(0, 0), (0, n - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pw, constant_values=fill)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    n = len(z["mover"])
    g = lambda k, shape, dt=np.float32, fill=0.0: (np.asarray(z[k]) if k in z.files else np.full((n,) + shape, fill, dt))
    d = {}
    d["obj_geo"] = _pad(g("obj_geo", (N, 8)), N).astype(np.float32)
    d["obj_txt"] = _pad(g("obj_txt", (N, 384)), N).astype(np.float32)
    d["obj_crop"] = _pad(g("obj_crop", (N, 768)), N).astype(np.float32)
    d["obj_crop_ok"] = _pad(g("obj_crop_ok", (N,), bool, False), N, False).astype(bool)
    d["obj_mask"] = _pad(g("obj_mask", (N,), bool, False), N, False).astype(bool)
    kd = g("obj_kind", (N,), np.int64, 0)
    if kd.dtype.kind in "OU":
        kd = np.vectorize(lambda s: KIND.get(str(s), 0))(kd)
    d["obj_kind"] = _pad(kd, N, 0).astype(np.int64)
    d["hand_geo"] = g("hand_geo", (2, 13)).astype(np.float32)
    d["hand_q16"] = g("hand_q16", (2, 16)).astype(np.float32)
    d["hand_q_ok"] = g("hand_q_ok", (2,), bool, False).astype(bool)
    d["vj_tok"] = g("vj_tok", (32, 1024), np.float16).astype(np.float16)
    d["vj_tok_flip"] = (np.asarray(z["vj_tok_flip"]) if "vj_tok_flip" in z.files else d["vj_tok"]).astype(np.float16)
    d["vj_ok"] = g("vj_ok", (), bool, True).astype(bool)
    d["mover"] = np.asarray(z["mover"]).astype(np.int64)
    d["target"] = g("target", (), np.int64, -1).astype(np.int64)
    d["relation"] = g("relation", (), np.int64, -1).astype(np.int64)
    d["end_off"] = g("end_off", (3,), np.float32, np.nan).astype(np.float32)
    d["end_pose"] = g("end_pose", (4,), np.float32, np.nan).astype(np.float32)
    d["path_obj"] = g("path_obj", (32, 3), np.float32, np.nan).astype(np.float32)
    d["path_hand"] = g("path_hand", (32, 3), np.float32, np.nan).astype(np.float32)
    d["path_hand_ok"] = g("path_hand_ok", (32,), bool, False).astype(bool)
    d["path_dt"] = g("path_dt", (), np.float32, np.nan).astype(np.float32)
    d["vj_future_glob"] = g("vj_future_glob", (1024,), np.float32, np.nan).astype(np.float32)
    d["prog_steps"] = g("prog_steps", (32,), np.float32, np.nan).astype(np.float32)
    d["prog_video"] = g("prog_video", (), np.float32, np.nan).astype(np.float32)
    d["prog_video_ok"] = g("prog_video_ok", (), bool, False).astype(bool)
    d["vj_future_ok"] = g("vj_future_ok", (), bool, True).astype(bool)
    d["time_to_move"] = g("time_to_move", (), np.float32, np.nan).astype(np.float32)
    d["ev_id"] = g("ev_id", (), np.int64, -1).astype(np.int64)
    d["table_idx"] = g("table_idx", (), np.int64, 0).astype(np.int64)
    d["snap_k"] = g("snap_k", (), np.int64, 0).astype(np.int64)
    d["mover_set"] = _pad(g("mover_set", (N,), bool, False), N, False).astype(bool)
    for k in ("dataset", "video", "group", "split"):
        d[k] = np.asarray(z[k]).astype(str) if k in z.files else np.array(["?"] * n)
    tt = np.asarray(z["task_text"]).astype(str)
    d["task_emb"] = np.asarray(z["task_emb"]).astype(np.float32) if "task_emb" in z.files else embed_text(list(tt))
    d["task_text"] = tt
    return d


class Store:
    def __init__(self, names, root=DATA):
        parts = [load_npz(Path(root) / f"{n}.npz") for n in names]
        self.d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.n = len(self.d["mover"])
        self.T = {k: torch.from_numpy(v) for k, v in self.d.items() if v.dtype.kind not in "USO"}
        self.names = names

    def where(self, **kw):
        m = np.ones(self.n, bool)
        for k, v in kw.items():
            m &= np.isin(self.d[k], v if isinstance(v, (list, tuple, set, np.ndarray)) else [v])
        return np.flatnonzero(m)

    def groups(self, dataset, split="train"):
        i = self.where(dataset=dataset, split=split)
        return sorted(set(self.d["group"][i]))
