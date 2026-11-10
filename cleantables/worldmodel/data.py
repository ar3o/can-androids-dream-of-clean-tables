"""World-model data: train / test split, text bank, composite scenes and frozen video features. Everything stays on the CPU
until make_batch moves one batch to the device.

Split: test = the robot's 7 held-out twin episodes (IMG_2516_t11..t14, IMG_2546/2547/2548_t01) plus the other episodes of
those box videos (each box video stays whole), 10 episodes; train = the other 35. Optional folds inside train follow the
base model's rule: box videos whole, fruit episodes one by one.

Text: a bank of unique strings, one vector each (MiniLM via cleantables.worldmodel.text.embed; the 'hashed' ablation uses
the base model's md5 vectors for the same strings). Per sample: the original instruction, its 8 training paraphrases and
the narration of its next move, if any. The 4 test paraphrases per episode are in the bank for evaluation only.

Composite scenes: the movable objects of a random training sample with a different instruction are added onto free table
space (clear of every object and of the mover's path, so the labels stay true); they do not move. Composites are 3D-only.
"""
from __future__ import annotations

from cleantables import CT_DATA
import json
import os
from pathlib import Path

import numpy as np
import torch

from cleantables.worldmodel.common import OUT, ROOT, hashed, log
from cleantables.features import video as VF

FE = VF.FE
N_RENDER_VARIANTS = 6            # render variants: v0 live look, v1-v3 domain-randomised, v4-v5 also randomise the camera pose
TEXT = Path(os.environ.get("WM_TEXT_DIR", str(OUT / "wm_v2" / "text")))       # the final training used the Qwen3-VL-8B text (text_8b)
HELD_TWINS = ["IMG_2516_t11", "IMG_2516_t12", "IMG_2516_t13", "IMG_2516_t14", "IMG_2546_t01", "IMG_2547_t01", "IMG_2548_t01"]
HELD_VIDEOS = ("IMG_2546", "IMG_2547", "IMG_2548")
NF = 5
WATCH_INPUT_KEYS = ("obj_geo", "obj_txt", "obj_crop", "obj_crop_ok", "obj_mask", "obj_kind", "hand_geo", "hand_q16", "hand_q_ok", "vj_ok",
                    "video_id")                                         # watch batches: inputs only, none of the move-label keys
WATCH_KEYS = ("t1_pos", "t1_yaw", "t1_ok", "t1_present", "t2_disp", "t2_ok", "t5", "t5_ok", "obj_contact", "obj_contact_ok", "t1_yaw_ok")
MODEL_KEYS = ("obj_geo", "obj_txt", "obj_crop", "obj_crop_ok", "obj_mask", "obj_kind", "hand_geo", "hand_q16", "hand_q_ok", "vj_ok",
              "mover", "target", "relation", "end_off", "end_pose", "path_obj", "path_hand", "path_hand_ok", "prog_steps", "prog_video",
              "prog_video_ok", "time_to_move", "mover_set", "video_id", "end_ok", "path_rot", "rot_ok")


class RowCat:
    """Two row-aligned memmaps read as one (rows of `a`, then rows of `b`): the base feature folder and the folder of rows it lacks
    (tracks_adapt / WM_V2_FEATS_EXT), which avoids a 44 GB copy. Supports X[rows] and X[rows, flip] with integer arrays or ints."""

    def __init__(s, a, b):
        s.a, s.b, s.n0 = a, b, a.shape[0]
        s.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:]); s.dtype = a.dtype

    def __getitem__(s, key):
        rows, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        r = np.asarray(rows)
        if r.ndim == 0:
            r_ = int(r)
            return s.a[(r_,) + rest] if r_ < s.n0 else s.b[(r_ - s.n0,) + rest]
        rest = tuple(np.broadcast_to(np.asarray(x), r.shape) if not isinstance(x, slice) else x for x in rest)
        lo = r < s.n0
        out = None
        for m_, arr, off in ((lo, s.a, 0), (~lo, s.b, s.n0)):
            if not m_.any():
                continue
            part = arr[(r[m_] - off,) + tuple(x[m_] if not isinstance(x, slice) else x for x in rest)]
            if out is None:
                out = np.empty((len(r),) + part.shape[1:], part.dtype)
            out[m_] = part
        return out


def _rowcat(fe, fe2, name):
    a = np.load(fe / name, mmap_mode="r")
    return RowCat(a, np.load(fe2 / name, mmap_mode="r")) if fe2 is not None and (fe2 / name).exists() else a


def _donecat(fe, fe2, name):
    d = np.load(fe / name)
    return np.r_[d, np.load(fe2 / name)] if fe2 is not None and (fe2 / name).exists() else d


def is_test(ep):
    return ep in HELD_TWINS or ep.startswith(HELD_VIDEOS)


def fourier_uv(uv, nf=8):
    f = (2.0 ** torch.arange(nf, dtype=torch.float32, device=uv.device)) * np.pi
    a = uv[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)                     # (..., 2 * 2 * nf)


def rot6_to_mat(r):
    a, c = r[..., :3], r[..., 3:6]
    x = torch.nn.functional.normalize(a, dim=-1)
    y = torch.nn.functional.normalize(c - (x * c).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)                 # columns


def mat_to_rot6(M):
    return torch.cat([M[..., :, 0], M[..., :, 1]], -1)


def transform_rot6(r6, mir, th):
    """Rotation paths (B, T, 6) relative to the start, table frame, under the base augmentation: mirror x -> -x (S R S) and a rotation
    by th about z (Rz R Rz^T). th is recovered exactly from the end yaw (augment adds th and mirrors the sign)."""
    th = torch.nan_to_num(th)
    M = rot6_to_mat(r6)
    S = torch.diag_embed(torch.stack([torch.where(mir, -1.0, 1.0), torch.ones_like(th), torch.ones_like(th)], -1))[:, None]
    c, s_ = th.cos(), th.sin(); z, o = torch.zeros_like(th), torch.ones_like(th)
    Rz = torch.stack([torch.stack([c, -s_, z], -1), torch.stack([s_, c, z], -1), torch.stack([z, z, o], -1)], -2)[:, None]
    M = Rz @ (S @ M @ S) @ Rz.transpose(-1, -2)
    return mat_to_rot6(M)


class Data:
    def __init__(self, text="minilm", feats=True, narr_filter=True):
        self.narr_filter = narr_filter                                         # only narrations consistent with the tracked mover (text.narration check)
        self.comp_balance = False                                              # set from the training config (comp_balance)
        self.relay_keep_inside = False                                         # set from the training config (relay_keep_inside)
        from cleantables.worldmodel import store
        from cleantables.worldmodel import text as v2_text
        store.embed_text = lambda strings, dim=384: v2_text.embed(strings)  # no hashed fallback: raises if MiniLM is missing
        from cleantables.worldmodel.base_train import prep_store
        name = os.environ.get("TASKS3_DATA", "tasks3c")
        droot = Path(os.environ.get("WM_DATA_ROOT", str(ROOT / "data/uwm")))      # the final training used OUT/wm_v2/data (tasks3_fp6d)
        if droot == ROOT / "data/uwm":
            S = prep_store([name])
        else:
            from cleantables.worldmodel.store import Store
            import cleantables.worldmodel.base_train as _tr
            _orig = _tr.Store
            _tr.Store = lambda names: Store(names, root=droot)                   # same derived fields as prep_store, other folder
            try:
                S = prep_store([name])
            finally:
                _tr.Store = _orig
        Z = np.load(droot / f"{name}.npz", allow_pickle=True)
        if "path_rot" in Z.files:                                              # 6D tracks: rotation along the carry (6D representation, relative to the start)
            S.T["path_rot"] = torch.from_numpy(np.nan_to_num(Z["path_rot"].astype(np.float32)))
            S.T["rot_ok"] = torch.from_numpy(Z["rot_ok"].astype(bool) & np.isfinite(Z["path_rot"]).all((1, 2)))
        else:
            S.T["path_rot"] = torch.zeros(S.n, 32, 6); S.T["rot_ok"] = torch.zeros(S.n, dtype=torch.bool)
        S.d["end_ok"] = Z["end_ok"].astype(bool); S.T["end_ok"] = torch.from_numpy(S.d["end_ok"])
        if "fut_disp" in Z.files:                                              # tracked future displacements
            fd = Z["fut_disp"].astype(np.float32); S.T["fut_disp"] = torch.from_numpy(fd.reshape(fd.shape[0], -1, 3))
            S.T["fut_disp_ok"] = torch.from_numpy(Z["fut_disp_ok"].astype(bool))
        self.world_only = Z["world_only"].astype(bool) if "world_only" in Z.files else np.zeros(len(Z["mover"]), bool)
        self.feat_row = Z["feat_row"].astype(np.int64) if "feat_row" in Z.files else None          # from tracks_adapt: direct feature rows
        self.track_main = Z["track_main"].astype(bool) if "track_main" in Z.files else None
        self.mask_main = Z["mask_main"].astype(bool) if "mask_main" in Z.files else self.track_main
        self.track_set = Z["track_set"].astype(str) if "track_set" in Z.files else None   # from tracks_adapt: the track set the obj_id indexes refer to
        self.watch = "t2_disp" in Z.files                                    # watch data (watch_data.py): T1 / T2 / T5 targets
        if self.watch:
            for k_ in WATCH_KEYS:
                x_ = Z[k_] if k_ in Z.files else Z["t1_ok"]                     # t1_yaw_ok defaults to t1_ok: watch_data always stores T1 yaw with T1
                x_ = x_.reshape(x_.shape[0], -1, 3) if k_ == "t2_disp" else x_
                S.T[k_] = torch.from_numpy(x_.astype(np.float32) if x_.dtype.kind == "f" else x_.astype(bool))
            self.t3_k = Z["t3_k"].astype(np.int64)
            for k_ in ("obj_contact2", "obj_contact2_ok"):                     # Hands23 contact (optional second input channel)
                S.T[k_] = torch.from_numpy(Z[k_].astype(bool)) if k_ in Z.files else torch.zeros(S.n, Z["obj_mask"].shape[1], dtype=torch.bool)
            # remaining-motion fraction: at keyframe k, the sum over objects of max(0, |T1 end - now| - n50) divided by the same sum at
            # the episode's first keyframe (same track set and episode id); continuous, from the tracked data. Defined only in
            # episodes whose first-keyframe sum is > 5 cm.
            from cleantables.worldmodel.model import N50
            t1p, t1o = Z["t1_pos"].astype(np.float32), Z["t1_ok"].astype(bool) & Z["t1_present"].astype(bool)
            om_ = Z["obj_mask"].astype(bool) & (Z["obj_kind"] == 0) & t1o
            num = (np.maximum(np.linalg.norm(t1p - Z["obj_geo"][..., :3], axis=-1) - N50, 0) * om_).sum(-1)
            ts_ = Z["track_set"].astype(str) if "track_set" in Z.files else Z["video"].astype(str)
            key_ = np.array([f"{a}|{g}" for a, g in zip(ts_, Z["group"].astype(str))]); k_s = Z["snap_k"].astype(int)
            den = np.zeros(len(num), np.float32)
            for u_ in np.unique(key_[om_.any(-1)]):
                ix_ = np.flatnonzero((key_ == u_) & om_.any(-1))
                den[key_ == u_] = num[ix_[np.argmin(k_s[ix_])]]
            ok_ = (den > 0.05) & om_.any(-1)
            S.T["rem_frac"] = torch.from_numpy(np.where(ok_, np.clip(num / np.maximum(den, 1e-6), 0, 1), 0).astype(np.float32))
            S.T["rem_ok"] = torch.from_numpy(ok_)
        S.d["obj_id"] = Z["obj_id"].astype(np.int64)
        ep = S.d["group"]
        ids = {g: i for i, g in enumerate(sorted(set(ep)))}
        S.T["video_id"] = torch.tensor([ids[g] for g in ep])                  # progress ranking within the episode (as in tasks3_train)
        self.S, self.n = S, S.n
        self.ep = ep.astype(str)
        # subtask data: each example points to the sample at the same (video, keyframe) (src_index, for sample-indexed stores), its
        # next move's source sample (fut_src, the future-latent row), its kind and its base episode (split, paraphrases, per-episode
        # object stats). Whole subtasks keep the episode id as their group.
        if "src_index" in Z.files:
            self.src = Z["src_index"].astype(np.int64); self.fut_src = Z["fut_src"].astype(np.int64)
            self.sub_kind = Z["sub_kind"].astype(str); self.ep_base = Z["base_ep"].astype(str)
        else:
            self.src = np.arange(self.n); self.fut_src = None; self.sub_kind = np.full(self.n, "whole"); self.ep_base = self.ep
        self.task = S.d["dataset"].astype(str)
        self.test = np.array([is_test(e) for e in self.ep_base])
        trainable = S.d["task_text"].astype(str) != "PENDING"                   # subtasks without an instruction
        test_sub = os.environ.get("WM_TEST_SUB", "whole")                       # evaluation uses whole subtasks (= episodes) unless 'all'
        self.train_idx = np.flatnonzero(~self.test & trainable & ~self.world_only)
        self.test_idx = np.flatnonzero(self.test & trainable & ~self.world_only & ((self.sub_kind == "whole") | (test_sub == "all")))
        # the self-supervised world losses use every training keyframe (task examples and the reset / setup ones)
        if self.watch:                                                         # watch: every keyframe trains (T1 masked outside episodes)
            self.train_idx = np.flatnonzero(~self.test)
            self.test_idx = np.flatnonzero(self.test & ((self.sub_kind == "whole") | (test_sub == "all")))
        self.world_train_idx = np.flatnonzero(~self.test) if "fut_disp" in S.T else np.zeros(0, np.int64)
        self.world_test_idx = np.flatnonzero(self.test) if "fut_disp" in S.T else np.zeros(0, np.int64)
        key = np.where(S.d["video"] == "IMG_2516", self.ep_base, S.d["video"])  # base model's fold rule, inside train only
        uk = sorted(set(key[~self.test]))
        box = [k for k in uk if not k.startswith("IMG_2516")]; fru = [k for k in uk if k.startswith("IMG_2516")]
        fold_of = {k: i % NF for i, k in enumerate(box)} | {k: i % NF for i, k in enumerate(fru)}
        self.fold = np.array([fold_of.get(k, -1) for k in key])
        self._phrases()
        self._text(text)
        self._rest_top()
        if not self.watch:                                                     # stop augmentation: labelled data only
            self._stop_aug_index()
        self.feats = feats
        if feats:
            self._feats()
        log(f"v2 data: {self.n} samples, train {len(self.train_idx)} ({len(set(self.ep[self.train_idx]))} eps), "
            f"test {len(self.test_idx)} ({len(set(self.ep[self.test_idx]))} eps), text={text}, feats={feats}")

    # -------------------------------------------------------------- object names (held-out-names generalisation test)
    def _phrases(self):
        from cleantables.features import video as VF_
        oid = self.S.d["obj_id"]; ph = np.full(oid.shape, "", dtype=object)
        for n in range(self.n):
            tr = VF_.tracks(self.S.d["video"][n] if self.track_set is None else self.track_set[n])
            for s_ in range(oid.shape[1]):
                j = int(oid[n, s_])
                if j >= 0 and self.S.d["obj_mask"][n, s_] and 0 <= j < len(tr):
                    ph[n, s_] = str(tr[j]["phrase"])
        self.phrase = ph

    def holdout(self, names):
        """Names held out of all training text (e.g. ('apple', 'wrench')): such objects get the name vector of 'object' in training
        batches, and training strings that mention them are never sampled. Evaluation always uses the real names."""
        names = tuple(names or ())
        self.held_names = names
        self.held_slot = torch.from_numpy(np.vectorize(lambda p: any(h in p.lower() for h in names))(self.phrase).astype(bool)) if names \
            else torch.zeros(self.phrase.shape, dtype=torch.bool)
        self.string_ok = np.array([not any(h in s_.lower() for h in names) for s_ in self.strings])
        if names:
            from cleantables.worldmodel.text import embed
            self.neutral_txt = torch.from_numpy(embed(["object"])[0])

    # -------------------------------------------------------------- resting height
    def _stop_aug_index(self):
        """For the stop examples (comp_stop2): per episode, the objects that are ever a mover (the instruction's objects, by label), and
        the samples just before the task's last move (mover set = that one object, and the episode reaches its complete state)."""
        d = self.S.d
        mv, oid = d["mover"], d["obj_id"]
        self.moved_ids = {}
        for n in np.flatnonzero(mv >= 0):
            self.moved_ids.setdefault(self.ep[n], set()).add(int(oid[n, mv[n]]))
        msa = d["mover_set"].astype(bool)
        done = ~msa.any(-1) & (d["prog_video"] >= 0.999)
        last_ev = {}
        for n in np.flatnonzero(mv >= 0):
            last_ev[self.ep[n]] = max(last_ev.get(self.ep[n], -1), int(d["ev_id"][n]))
        has_done = set(self.ep[done])
        self.last_move = np.array([mv[n] >= 0 and int(d["ev_id"][n]) == last_ev.get(self.ep[n]) and self.ep[n] in has_done
                                   and msa[n].sum() == 1 and bool(msa[n, mv[n]]) for n in range(len(mv))])

    def _remove_one(self, b, n, i, rng):
        """'Remove-one' stop example: an object of the instruction (a mover at some point in this episode) that now lies in / on
        another object is put back onto free table space at its resting height. It becomes a valid next mover (the target is that
        object, not stop) and the count progress is masked. Uses only labels and geometry, no names."""
        g = b["obj_geo"][n].numpy(); m = b["obj_mask"][n].numpy().astype(bool); kind = b["obj_kind"][n].numpy()
        oid = self.S.d["obj_id"][i]; mvset = b["mover_set"][n].numpy().astype(bool); mv = int(b["mover"][n])
        mov = np.flatnonzero(m & (kind == 0)); rad = 0.25 * (g[:, 3] + g[:, 4])
        moved = self.moved_ids.get(self.ep[i], set())
        cand = [s_ for s_ in mov if int(oid[s_]) in moved and s_ != mv and not mvset[s_]
                and any(c != s_ and np.linalg.norm(g[s_, :2] - g[c, :2]) < rad[c] for c in mov)]
        tab = np.flatnonzero(m & (kind == 1))
        rt = self.rest_top[i].numpy()
        cand = [s_ for s_ in cand if np.isfinite(rt[s_])]
        if not cand or not len(tab):
            return False
        s_ = cand[rng.integers(len(cand))]
        tx, ty, _, TL, TW = g[tab[0], :5]
        occ = [(g[c, :2], rad[c]) for c in mov if c != s_]
        path = self.S.d["path_obj"][i][:, :2]
        path = path[np.isfinite(path).all(-1)] if mv >= 0 else np.zeros((0, 2))
        r = rad[s_]
        for _ in range(40):
            p = np.array([rng.uniform(tx - TL / 2 + r, tx + TL / 2 - r), rng.uniform(ty - TW / 2 + r, ty + TW / 2 - r)])
            if any(np.linalg.norm(p - q) < 1.2 * (r + rq) + 0.02 for q, rq in occ):
                continue
            if len(path) and np.min(np.linalg.norm(path - p, axis=1)) < r + 0.05:
                continue
            g2 = b["obj_geo"][n].clone()
            g2[s_, 0], g2[s_, 1] = float(p[0]), float(p[1]); g2[s_, 2] = g2[s_, 5] = float(rt[s_])
            b["obj_geo"][n] = g2
            b["mover_set"][n, s_] = True
            b["prog_video_ok"][n] = False
            b["altered"][n] = True
            return True
        return False

    def _add_last(self, b, n, i):
        """'Place-the-last-one' stop example: just before the task's last move, the mover is put at its human end pose (the label's
        end position and end yaw); the instructed task is then complete, so the target is stop and the count progress is 1. The
        mover's own labels (destination, path, end) no longer apply and are dropped."""
        ep_ = b["end_pose"][n]
        if not bool(b["end_ok"][n]) or not bool(torch.isfinite(ep_).all()):
            return False
        mv = int(b["mover"][n])
        g2 = b["obj_geo"][n].clone()
        g2[mv, 0], g2[mv, 1] = ep_[0], ep_[1]; g2[mv, 2] = g2[mv, 5] = ep_[2]
        g2[mv, 6], g2[mv, 7] = torch.sin(2 * ep_[3]), torch.cos(2 * ep_[3])
        b["obj_geo"][n] = g2
        b["mover_set"][n] = False; b["mover"][n] = -1; b["target"][n] = -1; b["relation"][n] = -1
        b["prog_video"][n] = 1.0; b["prog_video_ok"][n] = True
        b["altered"][n] = True
        return True

    def _rest_top(self):
        """obj_geo has no height (index 5 repeats the top), so an object's top when resting on the table is taken as the lowest top it
        has in its episode. Used to put composite / re-laid objects on the table rather than at the top they had inside a container."""
        g = self.S.d["obj_geo"]; oid = self.S.d["obj_id"]; ok = self.S.d["obj_mask"].astype(bool) & (oid >= 0)
        low = {}
        for n, s_ in zip(*np.nonzero(ok)):
            k = (self.ep_base[n], int(oid[n, s_]))
            low[k] = min(low.get(k, 9.0), float(g[n, s_, 2]))
        rt = np.full(oid.shape, np.nan, np.float32)
        for n, s_ in zip(*np.nonzero(ok)):
            rt[n, s_] = low[(self.ep_base[n], int(oid[n, s_]))]
        self.rest_top = torch.from_numpy(rt)

    # -------------------------------------------------------------- text bank
    def _text(self, mode):
        P = json.load(open(TEXT / "paraphrases.json"))
        Nn = json.load(open(TEXT / "narration.json")) if (TEXT / "narration.json").exists() else {}
        strings, sid = [], {}

        def add(s):
            s = str(s).strip()
            if s not in sid:
                sid[s] = len(strings); strings.append(s)
            return sid[s]
        orig = np.array([add(t) for t in self.S.d["task_text"]])
        alt = np.full((self.n, 9), -1, np.int64); test_alt = np.full((self.n, 4), -1, np.int64); narr = np.full(self.n, -1, np.int64)
        missing_p = set()
        for i in range(self.n):
            e = self.ep_base[i]
            alt[i, 0] = orig[i]
            if e in P and self.sub_kind[i] == "whole":                          # paraphrases are of the episode's own instruction
                for j, s in enumerate(P[e]["train"][:8]):
                    alt[i, j + 1] = add(s)
                for j, s in enumerate(P[e]["test"][:4]):
                    test_alt[i, j] = add(s)
            elif e not in P:
                missing_p.add(e)
            if not getattr(self, "watch", False) and self.S.d["mover"][i] >= 0:   # no narrations for label-free data
                k = f"{e}#{int(self.S.d['ev_id'][i])}"
                if k in Nn and Nn[k]["text"] and (Nn[k].get("consistent", False) or not self.narr_filter):
                    narr[i] = add(Nn[k]["text"])
        if missing_p:
            log(f"WARNING: no paraphrases for {len(missing_p)} episodes (text job not finished?)")
        self.strings, self.orig, self.alt, self.test_alt, self.narr = strings, orig, alt, test_alt, narr
        self.instr_id = orig                                                   # composites: donor must have a different instruction
        if mode == "minilm":
            from cleantables.worldmodel.text import embed
            self.bank = torch.from_numpy(embed(strings))
        elif mode == "hashed":
            self.bank = torch.from_numpy(np.stack([hashed(s) for s in strings]))
        else:
            raise ValueError(mode)
        self.text_mode = mode

    def embed_new(self, strings):
        from cleantables.worldmodel.text import embed
        return torch.from_numpy(embed(strings)) if self.text_mode == "minilm" else torch.from_numpy(np.stack([hashed(s) for s in strings]))

    # -------------------------------------------------------------- frozen features
    def _feats(self):
        R = json.load(open(FE / "rows.json"))
        M = VF._loaded(FE / "masks.npz")
        FE2 = Path(os.environ["WM_V2_FEATS_EXT"]) if os.environ.get("WM_V2_FEATS_EXT") else None
        self.FE2 = FE2
        if FE2 is not None:                                                    # plus the rows the base folder lacks (tracks_adapt)
            R2 = json.load(open(FE2 / "rows.json")); M2 = VF._loaded(FE2 / "masks.npz")
            R = dict(R, rows=list(R["rows"]) + list(R2["rows"]))
            J = max(M["cov_vj"].shape[1], M2["cov_vj"].shape[1])
            padj = lambda x: np.concatenate([x, np.zeros((x.shape[0], J - x.shape[1]) + x.shape[2:], x.dtype)], 1) if x.shape[1] < J else x   # noqa: E731
            M = dict(cov_vj=np.concatenate([padj(M["cov_vj"]), padj(M2["cov_vj"])]), cov_dn=np.concatenate([padj(M["cov_dn"]), padj(M2["cov_dn"])]),
                     pix=np.concatenate([M["pix"], M2["pix"]]), nobj=np.concatenate([M["nobj"], M2["nobj"]]))
        self.snap_row = self.feat_row if self.feat_row is not None else np.array(R["snap_row"])[self.src]
        fr_ = np.array(R["future_row"])
        self.fut_row = fr_ if self.fut_src is None else np.where(self.fut_src >= 0, fr_[np.maximum(self.fut_src, 0)], -1)
        if getattr(self, "watch", False):                                      # T3: the feature row of the keyframe at +4 s (no move end)
            ri = {(str(v), int(k)): n for n, (v, k) in enumerate(R["rows"])}
            vv = self.S.d["video"].astype(str)
            self.fut_row = np.array([ri.get((vv[n], int(self.t3_k[n])), -1) if self.t3_k[n] >= 0 else -1 for n in range(self.n)], np.int64)
        self.VJ = _rowcat(FE, FE2, "vj_grid.npy"); self.DN = _rowcat(FE, FE2, "dn_grid.npy")
        for nm in ("vj", "dn"):
            d = _donecat(FE, FE2, f"{nm}_grid.done.npy")
            assert d.all() and len(d) == self.VJ.shape[0], f"{nm} grids incomplete ({int(d.sum())}/{len(d)})"
        self.pix = M["pix"]
        cov = M["cov_vj"]
        oid = self.S.d["obj_id"]
        # per sample and slot: mask centroid on the patch grid (u, v in [0, 1]) of that object at the snapshot frame
        uv = np.zeros((self.n, oid.shape[1], 2), np.float32); uv_ok = np.zeros((self.n, oid.shape[1]), bool)
        hh, ww = VF.VJ_HW
        yy, xx = np.meshgrid((np.arange(hh) + 0.5) / hh, (np.arange(ww) + 0.5) / ww, indexing="ij")
        for i in range(self.n):
            r = self.snap_row[i]
            if self.mask_main is not None and not self.mask_main[i]:         # track set differs from the row's masks (tracks_adapt)
                continue
            for s in range(oid.shape[1]):
                j = oid[i, s]
                if j < 0 or not self.S.d["obj_mask"][i, s] or j >= cov.shape[1]:
                    continue
                c = cov[r, j].astype(np.float32) / 255
                w = c.sum()
                if w > VF.COV_MIN:
                    uv[i, s] = [(c * xx).sum() / w, (c * yy).sum() / w]; uv_ok[i, s] = True
        self.uv, self.uv_ok = torch.from_numpy(uv), torch.from_numpy(uv_ok)
        # occupancy rasters (location-only control for the grids): any tracked object's mask, at the snapshot frame
        self.occ_vj = (cov.max(1).astype(np.float32) / 255)
        self.occ_dn = (M["cov_dn"].max(1).astype(np.float32) / 255)
        G = VF._loaded(FE / "objg.npz")
        if self.feat_row is not None:                                          # tracks data: objg is stored per base sample, so it is
            G = dict(objg=np.full((self.n,) + G["objg"].shape[1:], np.nan, np.float32),   # undefined here (off in the watch model;
                     dino=np.full((self.n,) + G["dino"].shape[1:], np.nan, np.float32))   # watch_objg needs a recomputed objg)
        go, gd = G["objg"][self.src], G["dino"][self.src]
        ok_g = np.isfinite(go[..., 0]) & (self.sub_kind == "whole")[:, None]    # goal-frame features are the episode's end: whole subtasks only
        self.objg = torch.from_numpy(np.nan_to_num(go, nan=0.0)); self.objg_ok = torch.from_numpy(ok_g)
        self.dino_goal = torch.from_numpy(np.nan_to_num(gd, nan=0.0))
        FE2_ = getattr(self, "FE2", None)
        aw = FE2_ is not None and self.track_set is not None and (FE2_ / "app_watch.npz").exists()
        A = VF._loaded(FE2_ / "app_watch.npz") if aw else VF._loaded(FE / "app.npz")
        akey = {str(k): i for i, k in enumerate(A["keys"])}
        aidx = np.full(oid.shape, -1, np.int64)
        grp_ = self.S.d["group"].astype(str)
        for i in range(self.n):
            if aw:                                                             # features.appearance: keyed by (track set, segment, object);
                if not self.mask_main[i]:                                      # the row's masks must be this track set
                    continue
            elif self.track_main is not None and not self.track_main[i]:
                continue
            for s in range(oid.shape[1]):
                if oid[i, s] >= 0 and self.S.d["obj_mask"][i, s]:
                    k_ = f"{self.track_set[i]}|{grp_[i]}:{int(oid[i, s])}" if aw else f"{self.ep_base[i]}:{oid[i, s]}"
                    aidx[i, s] = akey.get(k_, -1)
        okA = A["n_frames"] > 0
        self.app_idx = torch.from_numpy(np.where((aidx >= 0) & okA[np.maximum(aidx, 0)], aidx, -1))
        self.app = {k: torch.from_numpy(A[k].astype(np.float32)) for k in ("app_vj", "app_dn", "app_ctl", "app_vj_flip", "app_dn_flip")}
        self.v2v = None                                                           # extra video feature sets, loaded on first use

    def _video_v2(self):
        """Load the extra video features: reversed clips, crop / zoom and random-start variants (cleantables.features.video aug) and
        robot-hidden twin renders of the same moments (render_row[i] = the render's row, -1 = no render)."""
        if self.v2v is None:
            V = {}
            FE2 = getattr(self, "FE2", None)
            for nm in ("vj_rev", "dn_rev", "vj_aug", "dn_aug"):
                if (FE / f"{nm}.done.npy").exists() and np.load(FE / f"{nm}.done.npy").all() and \
                        (FE2 is None or ((FE2 / f"{nm}.done.npy").exists() and np.load(FE2 / f"{nm}.done.npy").all())):
                    V[nm] = _rowcat(FE, FE2, f"{nm}.npy")
            V["aug_params"] = np.load(FE / "aug_params.npy") if (FE / "aug_params.npy").exists() else None
            if FE2 is not None and V["aug_params"] is not None and (FE2 / "aug_params.npy").exists():   # the new rows' own crop parameters
                V["aug_params"] = np.concatenate([V["aug_params"], np.load(FE2 / "aug_params.npy")])
            # the new rows may hold fewer crop variants (e.g. 1 of N_AUG): rows >= aug_n0 use variant 0 .. aug_nv2 - 1
            V["aug_n0"] = V["vj_aug"].n0 if isinstance(V.get("vj_aug"), RowCat) else 1 << 62
            V["aug_nv2"] = V["vj_aug"].b.shape[1] if isinstance(V.get("vj_aug"), RowCat) else 0
            rr = np.full(self.n, -1, np.int64)
            if (FE / "render_index.json").exists() and self.feat_row is None:     # replay renders are indexed by base samples
                ri_ = json.load(open(FE / "render_index.json"))["samples"]
                rr0 = np.full(max([len(self.src) if self.fut_src is None else int(self.src.max()) + 1] + [int(i) + 1 for i in ri_]), -1, np.int64)
                for r, i in enumerate(ri_):
                    rr0[i] = r
                rr = rr0[self.src]
                V["render_vj"] = np.load(FE / "render_vj.npy", mmap_mode="r"); V["render_dn"] = np.load(FE / "render_dn.npy", mmap_mode="r")
            V["render_row"] = rr
            # paired renders (cleantables.features.paired; robot hidden; v0 = the robot's live look, v1-v3 domain-randomised), row-aligned
            # with rows.json, plain and flipped; 'hold' = the live sampling (a still every 1 s, held over the 32 slots), else true 10 fps.
            # Rows may be partly encoded, so each variant has a per-row done mask.
            PF_ = Path(os.environ.get("WM_PAIRED_FEATS", CT_DATA + "/vjepa_wm/paired_feats"))
            V["paired"] = []
            # v4-v7 also move the camera, so the real frame's mask centroids do not fit them; their own projected uv (uv_v<n>.npy
            # (rows, 68, 2) + uv_ok_v<n>.npy, by track index) is required, and a variant without it is not used
            for v in range(N_RENDER_VARIANTS):
                uvf = (PF_ / f"uv_v{v}.npy", PF_ / f"uv_ok_v{v}.npy")
                if v >= 4 and not (uvf[0].exists() and uvf[1].exists()):
                    continue
                for kind_, vjn in (("hold", f"vj_hold_v{v}"), ("10fps", f"vj_v{v}")):
                    fv, fd = PF_ / f"{vjn}.npy", PF_ / f"dn_v{v}.npy"
                    if fv.exists() and fd.exists() and (PF_ / f"{vjn}.done.npy").exists() and (PF_ / f"dn_v{v}.done.npy").exists():
                        dn_ = np.load(PF_ / f"{vjn}.done.npy") & np.load(PF_ / f"dn_v{v}.done.npy")
                        if len(dn_) < self.VJ.shape[0]:                         # subtask data has extra rows (no twin render)
                            dn_ = np.r_[dn_, np.zeros(self.VJ.shape[0] - len(dn_), bool)]
                        V["paired"].append(dict(v=v, kind=kind_, vj=np.load(fv, mmap_mode="r"), dn=np.load(fd, mmap_mode="r"), done=dn_,
                                                uv=(np.load(uvf[0], mmap_mode="r"), np.load(uvf[1], mmap_mode="r")) if v >= 4 else None))
            # rendered composites (cleantables.features.composite_render / composite): stored training rows and their render features
            CR = OUT / "wm_v2" / "composites_render"
            CF = Path(os.environ.get("WM_COMP_FEATS", CT_DATA + "/vjepa_wm/composites_feats"))
            if not CF.exists() and (CF.parent / "composite_feats").exists():
                CF = CF.parent / "composite_feats"
            V["rcomp"] = None
            if (CR / "batch.pt").exists() and self.feat_row is None:           # their rows / base samples are base-sample indices
                def cuv(v):                                                       # per-variant uv; v0-v3 may use the single uv.npy
                    if (CF / f"uv_v{v}.npy").exists() and (CF / f"uv_ok_v{v}.npy").exists():
                        return np.load(CF / f"uv_v{v}.npy"), np.load(CF / f"uv_ok_v{v}.npy")
                    if v < 4 and (CF / "uv.npy").exists():
                        return np.load(CF / "uv.npy"), np.load(CF / "uv_ok.npy")
                    return None
                vs = [v for v in range(N_RENDER_VARIANTS) if (CF / f"vj_v{v}.npy").exists() and (CF / f"dn_v{v}.npy").exists()
                      and (CF / f"vj_v{v}.done.npy").exists() and np.load(CF / f"vj_v{v}.done.npy").all() and (v < 4 or cuv(v) is not None)]
                if vs:
                    V["rcomp"] = dict(rows=torch.load(CR / "batch.pt"), index=json.load(open(CR / "index.json")),
                                      vj={v: np.load(CF / f"vj_v{v}.npy", mmap_mode="r") for v in vs}, dn={v: np.load(CF / f"dn_v{v}.npy", mmap_mode="r") for v in vs},
                                      uv={v: cuv(v) for v in vs})
                    # base_sample indexes the data the composites were made from: map it to this data's whole-subtask example at the
                    # same sample; a composite without one (outside the core) is not used
                    tr_ = np.zeros(self.n, bool); tr_[self.train_idx] = True
                    whole = {int(sx): n for n, sx in enumerate(self.src) if self.sub_kind[n] == "whole" and tr_[n]} \
                        if self.fut_src is not None else {n: n for n in range(self.n)}
                    V["rcomp"]["base_new"] = {c: whole[int(x["base_sample"])] for c, x in enumerate(V["rcomp"]["index"]) if int(x["base_sample"]) in whole}
                    V["rcomp"]["keep"] = np.array(sorted(V["rcomp"]["base_new"]), np.int64)
            log(f"video_v2b: paired feature sets {[(p_['v'], p_['kind'], int(p_['done'].sum())) for p_ in V['paired']]}; rendered composites "
                f"{0 if V['rcomp'] is None else len(V['rcomp']['index'])} (usable {0 if V['rcomp'] is None else len(V['rcomp']['keep'])})")
            self.v2v = V
            log(f"video_v2 features: {sorted(k for k in V if V[k] is not None and k not in ('render_row',))}; samples with a render "
                f"{int((rr >= 0).sum())} (train {int((rr[self.train_idx] >= 0).sum())}, test {int((rr[self.test_idx] >= 0).sum())})")
        return self.v2v

    # -------------------------------------------------------------- batch
    def make_batch(self, idx, cfg, rng, dev, text_ids=None, train=True, gen=None, comp_pool=None, force_comp=None):
        """idx (B,) numpy sample indices -> batch dict on dev (base keys plus video features), augmented if train.
        text_ids: explicit bank ids (B,) or None (training: sampled; eval: the original instruction)."""
        from cleantables.worldmodel.base_train import augment
        S, B = self.S, len(idx)
        rc_pos, rc_id = np.zeros(0, np.int64), np.zeros(0, np.int64)            # rendered composites in this batch
        if train and cfg.get("p_rcomp", 0) > 0 and self.feats:
            V = self._video_v2()
            if V.get("rcomp") is not None:
                rc_pos = np.flatnonzero(rng.random(B) < cfg["p_rcomp"])
                rc_id = V["rcomp"]["keep"][rng.integers(0, len(V["rcomp"]["keep"]), len(rc_pos))]
                idx = np.array(idx).copy(); idx[rc_pos] = [V["rcomp"]["base_new"][int(c)] for c in rc_id]
        b = {k: S.T[k][idx].clone() for k in (WATCH_INPUT_KEYS if getattr(self, "watch", False) else MODEL_KEYS)}
        if getattr(self, "watch", False):                                      # watch targets: positions / displacements move with the scene
            for k_ in WATCH_KEYS + ("obj_contact2", "obj_contact2_ok", "rem_frac", "rem_ok"):
                b[k_] = S.T[k_][idx].clone()
            cfg = {**cfg, "aug_vec_keys": ("t1_pos", "t2_disp")}
        if cfg.get("world") and "fut_disp" in S.T:                             # world-head targets (transformed by augment with the scene)
            b["fut_disp"], b["fut_disp_ok"] = S.T["fut_disp"][idx].clone(), S.T["fut_disp_ok"][idx].clone()
            cfg = {**cfg, "aug_vec_keys": ("fut_disp",)}
        b["app_idx"] = self.app_idx[idx].clone()
        b["held_slot"] = self.held_slot[idx].clone() if hasattr(self, "held_slot") else torch.zeros(len(idx), self.S.d["obj_id"].shape[1], dtype=torch.bool)
        if self.feats:
            b["uv"], b["uv_ok"] = self.uv[idx].clone(), self.uv_ok[idx].clone()
            b["objg"], b["objg_ok"] = self.objg[idx].clone(), self.objg_ok[idx].clone()
            b["dino_goal"] = self.dino_goal[idx].clone()
        # ---- text
        if text_ids is None:
            if train:
                text_ids = np.empty(B, np.int64)
                for n, i in enumerate(idx):
                    ok_s = getattr(self, "string_ok", None)
                    if self.narr[i] >= 0 and rng.random() < cfg.get("p_narr", 0.3) and (ok_s is None or ok_s[self.narr[i]]):
                        text_ids[n] = self.narr[i]
                    else:
                        a = self.alt[i][self.alt[i] >= 0] if cfg.get("paraphrase", True) else self.alt[i][:1]
                        if ok_s is not None:
                            a = a[ok_s[a]] if ok_s[a].any() else self.alt[i][:1]
                        text_ids[n] = a[rng.integers(len(a))]
            else:
                text_ids = self.orig[idx]
        b["task_emb"] = self.bank[torch.as_tensor(text_ids)].clone()
        # ---- stop examples (comp_stop2): remove-one / place-the-last-one, before the composites
        b["altered"] = torch.zeros(B, dtype=torch.bool)
        if train and (cfg.get("p_remove", 0) > 0 or cfg.get("p_addlast", 0) > 0):
            for n, i in enumerate(idx):
                if self.last_move[i] and rng.random() < cfg.get("p_addlast", 0):
                    self._add_last(b, n, i)
                elif rng.random() < cfg.get("p_remove", 0):
                    self._remove_one(b, n, i, rng)
        # ---- video share: every sample with real video trains with video at least p_video of the time. Such a sample gets no
        # augmentation that would invalidate its video (no composite, rotation, rescale or re-placement; mirroring is fine, the
        # flipped encodings exist) and runs in mode 0; the rest stay under modality dropout (including the robot's 3D-only mode)
        vkeep = np.zeros(B, bool)
        if train and cfg.get("p_video", 0) > 0 and self.feats:
            vkeep = S.T["vj_ok"][idx].numpy().astype(bool) & (rng.random(B) < cfg["p_video"])
            if force_comp is not None:
                vkeep &= ~np.asarray(force_comp, bool)
            vkeep[rc_pos] = False
        b["vid_keep"] = torch.from_numpy(vkeep)
        # ---- composite scenes
        comp = np.zeros(B, bool)
        b["relaid"] = torch.zeros(B, dtype=torch.bool)
        if train and (cfg.get("p_comp", 0) > 0 or force_comp is not None):
            for n, i in enumerate(idx):
                if n in rc_pos or vkeep[n]:
                    continue
                if rng.random() < cfg.get("p_comp", 0) or (force_comp is not None and force_comp[n]):
                    comp[n] = self._composite(b, n, i, rng, relayout=rng.random() < cfg.get("p_relay", 0.0))
        elif comp_pool is not None:                                            # evaluation composites (donors from comp_pool)
            for n, i in enumerate(idx):
                comp[n] = self._composite(b, n, i, rng, pool=comp_pool, relayout=bool(cfg.get("eval_relay", False)))
        if train and cfg.get("p_pair", 0) > 0 and getattr(self, "watch", False):   # paired-instruction composites
            rows_ = [n for n in range(B) if not comp[n] and not vkeep[n] and n not in rc_pos]
            rng.shuffle(rows_)
            want, done_ = int(round(cfg["p_pair"] * B / 2)), 0
            for q in range(0, len(rows_) - 1, 2):
                if done_ >= want:
                    break
                n, n2 = rows_[q], rows_[q + 1]
                for _ in range(int(cfg.get("pair_donors", 4))):                  # up to 4 donor groups per row pair
                    if self._pair(b, n, n2, idx[n], rng, text_ids, cfg):
                        comp[n] = comp[n2] = True; done_ += 1
                        break
        if train and cfg.get("p_rigid", 0) > 0 and getattr(self, "watch", False):   # whole-scene rigid re-layout
            st_ = getattr(self, "rigid_stats", None)
            if st_ is None:
                st_ = self.rigid_stats = dict(rows=0, eligible=0, drawn=0, ok=0)
            for n in range(B):
                st_["rows"] += 1
                if vkeep[n] or n in rc_pos:                                    # video samples: their pixels cannot move
                    continue
                st_["eligible"] += 1
                if rng.random() < cfg["p_rigid"]:
                    st_["drawn"] += 1
                    st_["ok"] += bool(self._rigid(b, n, rng))
        b["comp"] = torch.from_numpy(comp)
        if train and getattr(self, "held_names", ()):
            hs = b["held_slot"]
            b["obj_txt"][hs] = self.neutral_txt.to(b["obj_txt"].dtype)
        # ---- dtypes as cleantables.worldmodel.base_train.get_batch, then to the device
        for k in list(b):
            v = b[k]
            if v.dtype == torch.float16:
                v = v.float()
            elif v.dtype == torch.uint8:
                v = v.bool()
            elif v.dtype in (torch.int8, torch.int16, torch.int32):
                v = v.long()
            b[k] = v.to(dev)
        if len(rc_pos):                                                   # rendered composites: the stored training rows, as exported
            R = self.v2v["rcomp"]
            for n_, c in zip(rc_pos, rc_id):
                row = R["rows"][c]
                for k, v in row.items():
                    if k in b and k not in ("task_emb", "text_ids") and torch.is_tensor(v) and torch.is_tensor(b[k]) and b[k].dim() >= 1 \
                            and b[k].shape[0] == B and tuple(b[k].shape[1:]) == tuple(v.shape):
                        b[k][n_] = v.to(dev, b[k].dtype)
            b["comp"][torch.as_tensor(rc_pos, dtype=torch.long, device=dev)] = True; comp[rc_pos] = True
        b["obj_kind"] = b["obj_kind"].clamp(0, 2)
        b["vj_tok"] = torch.zeros(B, 1, 1, device=dev); b["vj_tok_flip"] = torch.ones(B, 1, 1, device=dev)   # mirror flag carrier
        yaw0 = b["end_pose"][:, 3].clone() if "end_pose" in b else None
        g2_0 = b["obj_geo"][..., 6:8].clone()
        if train:
            b = augment(b, {**cfg, "mirror": cfg.get("mirror", True)}, gen)
            if "t1_yaw" in b:                                                 # T1 yaw: the scene's own rotation / mirror (read off the tokens)
                mir_ = (b["vj_tok"][:, 0, 0] > 0.5).float() * -2 + 1
                g2_1 = b["obj_geo"][..., 6:8]
                okg = b["obj_mask"].bool() & (g2_0.norm(dim=-1) > 0.5)
                j0 = okg.float().argmax(-1); ar_ = torch.arange(B, device=dev)
                phi = torch.atan2(g2_1[ar_, j0, 0], g2_1[ar_, j0, 1]) - mir_ * torch.atan2(g2_0[ar_, j0, 0], g2_0[ar_, j0, 1])
                b["t1_yaw"] = 0.5 * (mir_[:, None] * 2 * b["t1_yaw"] + phi[:, None])
        else:
            b["aug_video_bad"] = torch.zeros(B, dtype=torch.bool, device=dev); b["aug_jepa_bad"] = torch.zeros(B, dtype=torch.bool, device=dev)
        mir = (b["vj_tok"][:, 0, 0] > 0.5)
        b.pop("vj_tok"); b.pop("vj_tok_flip")
        if train and "rot_ok" in b and bool(b["rot_ok"].any()):
            b["path_rot"] = transform_rot6(b["path_rot"], mir, b["end_pose"][:, 3] - torch.where(mir, -yaw0, yaw0))
        if train and (cfg.get("p_handdrop", 0) > 0 or bool(b["relaid"].any()) or bool(b["altered"].any())):  # the robot runs without human hand tokens
            hd = torch.as_tensor(rng.random(B) < cfg.get("p_handdrop", 0), device=dev) | b["relaid"] | b["altered"]   # re-laid / altered: hands would point at the old spots
            b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][hd, :, 0] = 0
            b["hand_q_ok"] = b["hand_q_ok"] & ~hd[:, None]
            for k_ in ("obj_contact_ok", "obj_contact2_ok"):                   # watch: the hand-contact inputs are dropped with the hands
                if k_ in b:
                    b[k_] = b[k_] & ~hd[:, None]
        rcm = torch.zeros(B, dtype=torch.bool, device=dev); rcm[torch.as_tensor(rc_pos, dtype=torch.long, device=dev)] = True
        b["rcomp"] = rcm                                                  # rendered composites: their video shows this scene
        b["aug_video_bad"] = b["aug_video_bad"] | (b["comp"] & ~rcm) | b["altered"]
        b["aug_jepa_bad"] = b["aug_video_bad"].clone()          # mirroring is fine: the future grid uses the flipped encoding too
        # ---- appearance tokens (mirrored version for mirrored samples)
        ai = b["app_idx"].cpu()
        okA = (ai >= 0)
        mi = mir.cpu()
        for nm in ("app_vj", "app_dn"):
            a0 = self.app[nm][ai.clamp(min=0)]; a1 = self.app[nm + "_flip"][ai.clamp(min=0)]
            b[nm] = torch.where(mi[:, None, None], a1, a0).to(dev) * okA[..., None].to(dev)
        b["app_ctl"] = (self.app["app_ctl"][ai.clamp(min=0)] * okA[..., None]).to(dev)
        b["app_ok"] = okA.to(dev)
        # ---- frozen grids (after augment: the mirror flag picks the flipped encoding)
        if self.feats:
            rows = self.snap_row[idx]; fl = mi.numpy().astype(np.int64)
            vg = cfg.get("vid", ("vj", "dn"))
            if "vj" in vg:
                b["vj_grid"] = torch.from_numpy(np.ascontiguousarray(self.VJ[rows, fl])).to(dev)
            if "dn" in vg:
                b["dn_grid"] = torch.from_numpy(np.ascontiguousarray(self.DN[rows, fl])).to(dev)
            ctl = cfg.get("vid_ctl", "")
            if ctl == "occ":
                o1, o2 = self.occ_vj[rows], self.occ_dn[rows]
                o1 = np.where(fl[:, None, None] == 1, o1[:, :, ::-1], o1); o2 = np.where(fl[:, None, None] == 1, o2[:, :, ::-1], o2)
                b["vj_grid"] = torch.from_numpy(np.ascontiguousarray(o1[..., None])).to(dev)
                b["dn_grid"] = torch.from_numpy(np.ascontiguousarray(o2[..., None])).to(dev)
            elif ctl == "pix":
                px = torch.from_numpy(np.ascontiguousarray(self.pix[rows, fl].astype(np.float32))).to(dev)
                b["vj_grid"] = px; b["dn_grid"] = px
            b["uv"] = torch.where(mir[:, None, None], torch.stack([1 - b["uv"][..., 0], b["uv"][..., 1]], -1), b["uv"])
            src = np.zeros(B, np.int64)          # video source: 0 real, 1 old replay render, 2 / 3 crop variant, 4 paired render, 5 rendered composite
            if cfg.get("video_v2") and not ctl:
                V = self._video_v2(); rr = V["render_row"][idx]
                P_ = V["paired"] if cfg.get("render_src", "replay") == "paired" else []
                psel = np.full(B, -1, np.int64)
                if P_:                                                           # per sample: a paired set whose row is encoded
                    hold = [q for q, ps in enumerate(P_) if ps["kind"] == "hold"]
                    for n_ in range(B):
                        oks = [q for q, ps in enumerate(P_) if ps["done"][rows[n_]]]
                        if not oks:
                            continue
                        if train:
                            oh = [q for q in oks if q in hold]
                            pool_ = oh if (oh and rng.random() < cfg.get("p_hold", 0.7)) else oks
                            psel[n_] = pool_[rng.integers(len(pool_))]
                        else:                                                    # robot mode: v0 (the live look), held sampling first
                            v0 = [q for q in oks if P_[q]["v"] == 0]
                            v0.sort(key=lambda q: P_[q]["kind"] != "hold")
                            psel[n_] = v0[0] if v0 else -1
                    has_r = psel >= 0
                else:
                    has_r = (rr >= 0) & (fl == 0)
                if train:
                    use_r = has_r & (rng.random(B) < cfg.get("p_render", 0.5))
                    use_c = ~use_r & (fl == 0) & ("vj_aug" in V) & (rng.random(B) < cfg.get("p_vcrop", 0.5))
                    src[use_r] = 4 if P_ else 1; src[use_c] = 2 + rng.integers(0, 2, int(use_c.sum()))
                    if "vj_aug" in V and V["aug_nv2"] and V["aug_nv2"] < 2:      # new rows with one crop variant
                        src[use_c & (rows >= V["aug_n0"])] = 2
                    src[rc_pos] = 5
                elif cfg.get("eval_video") == "render":                          # robot mode: the twin render where one exists, else no video
                    src[has_r] = 4 if P_ else 1
                    b["no_render"] = torch.as_tensor(~has_r, device=dev)
                if (src == 1).any():
                    j = np.flatnonzero(src == 1)
                    b["vj_grid"][j] = torch.from_numpy(np.ascontiguousarray(V["render_vj"][rr[j]])).to(dev).to(b["vj_grid"].dtype)
                    b["dn_grid"][j] = torch.from_numpy(np.ascontiguousarray(V["render_dn"][rr[j]])).to(dev).to(b["dn_grid"].dtype)
                for n_ in np.flatnonzero(src == 4):
                    ps = P_[psel[n_]]
                    b["vj_grid"][n_] = torch.from_numpy(np.ascontiguousarray(ps["vj"][rows[n_], fl[n_]])).to(dev).to(b["vj_grid"].dtype)
                    b["dn_grid"][n_] = torch.from_numpy(np.ascontiguousarray(ps["dn"][rows[n_], fl[n_]])).to(dev).to(b["dn_grid"].dtype)
                    if ps.get("uv") is not None:                                 # moved camera (v4+): that render's own projected uv
                        oid = self.S.d["obj_id"][idx[n_]]; okk = oid >= 0
                        u_ = np.zeros((len(oid), 2), np.float32); o_ = np.zeros(len(oid), bool)
                        u_[okk] = ps["uv"][0][rows[n_], oid[okk]]; o_[okk] = ps["uv"][1][rows[n_], oid[okk]]
                        if fl[n_]:
                            u_[:, 0] = 1 - u_[:, 0]
                        b["uv"][n_] = torch.as_tensor(u_, device=dev); b["uv_ok"][n_] = torch.as_tensor(o_, device=dev) & b["obj_mask"][n_].bool()
                if len(rc_pos):
                    R = V["rcomp"]
                    for n_, c in zip(rc_pos, rc_id):
                        v = list(R["vj"])[rng.integers(len(R["vj"]))]
                        b["vj_grid"][n_] = torch.from_numpy(np.ascontiguousarray(R["vj"][v][c, fl[n_]])).to(dev).to(b["vj_grid"].dtype)
                        b["dn_grid"][n_] = torch.from_numpy(np.ascontiguousarray(R["dn"][v][c, fl[n_]])).to(dev).to(b["dn_grid"].dtype)
                        if R["uv"].get(v) is not None:                            # the render's projected centroids (slot order), per variant
                            u_, o_ = torch.as_tensor(R["uv"][v][0][c], device=dev), torch.as_tensor(R["uv"][v][1][c], device=dev).bool()
                            b["uv"][n_] = torch.where(mir[n_], torch.stack([1 - u_[..., 0], u_[..., 1]], -1), u_); b["uv_ok"][n_] = o_
                        else:
                            b["uv_ok"][n_] = False
                    b["glob_fut_ok_off"] = torch.as_tensor(src == 5, device=dev) | b.get("glob_fut_ok_off", torch.zeros(B, dtype=torch.bool, device=dev))
                rnd = torch.as_tensor((src == 1) | (src >= 4), device=dev)
                b["objg_ok"][rnd] = False                                        # objg compares real frames, so it is unavailable on a render
                b["glob_fut_ok_off"] = rnd | b.get("glob_fut_ok_off", torch.zeros_like(rnd))   # the future-latent target is real video,
                                                                                 # so it is not trained from a render (same domain only)
                if ((src == 2) | (src == 3)).any():
                    j = np.flatnonzero((src == 2) | (src == 3)); v = src[j] - 2
                    b["vj_grid"][j] = torch.from_numpy(np.ascontiguousarray(V["vj_aug"][rows[j], v])).to(dev).to(b["vj_grid"].dtype)
                    b["dn_grid"][j] = torch.from_numpy(np.ascontiguousarray(V["dn_aug"][rows[j], v])).to(dev).to(b["dn_grid"].dtype)
                    pr = torch.as_tensor(V["aug_params"][rows[j], v], device=dev)            # (n, 4): x0, y0, s, L
                    jj = torch.as_tensor(j, device=dev)
                    uvc = (b["uv"][jj] - pr[:, None, :2]) / pr[:, None, 2:3]             # mask centroids in the cropped frame
                    b["uv"][jj] = uvc
                    b["uv_ok"][jj] = b["uv_ok"][jj] & (uvc >= 0).all(-1) & (uvc <= 1).all(-1)
                if train and "vj_rev" in V:                                      # reversed clips (negatives for progress / STOP)
                    # same domain only (render vs real comparisons track progress poorly): the reversed clip is plain real video,
                    # so its forward counterpart must be the plain real clip too (src 0)
                    rv = np.flatnonzero((rng.random(B) < cfg.get("p_rev", 0.25)) & (src == 0))
                    b["rev_sel"] = torch.as_tensor(rv, device=dev, dtype=torch.long)
                    if len(rv):
                        b["vj_rev"] = torch.from_numpy(np.ascontiguousarray(V["vj_rev"][rows[rv], fl[rv]])).to(dev)
                        b["dn_rev"] = torch.from_numpy(np.ascontiguousarray(V["dn_rev"][rows[rv], fl[rv]])).to(dev)
            b["vid_src"] = torch.as_tensor(src, device=dev)
            # current and future global V-JEPA latent (future-latent head; persistence baseline = current)
            fr = self.fut_row[idx]
            fok = fr >= 0
            cur = b["vj_grid"].float().mean((1, 2)).cpu().numpy() if "vj_grid" in b and b["vj_grid"].shape[-1] == 1024 else self.VJ[rows, fl].astype(np.float32).mean((1, 2))
            fut = np.zeros_like(cur)
            if fok.any():
                fut[fok] = self.VJ[fr[fok], fl[fok]].astype(np.float32).mean((1, 2))
            b["glob_now"] = torch.from_numpy(cur).to(dev); b["glob_fut"] = torch.from_numpy(fut).to(dev)
            b["glob_fut_ok"] = torch.from_numpy(fok).to(dev)
            if "glob_fut_ok_off" in b:
                b["glob_fut_ok"] = b["glob_fut_ok"] & ~b.pop("glob_fut_ok_off")
        b["donor"] = torch.as_tensor(self._last_donor_mask(b, idx), device=dev) if comp.any() else torch.zeros_like(b["obj_mask"])
        b["vid_ok"] = ((b["vj_ok"].bool() & ~b["comp"]) | b["rcomp"]) & ~b["altered"]   # altered stop examples: the video shows the original layout
        if "no_render" in b:
            b["vid_ok"] = b["vid_ok"] & ~b["no_render"]
        b["vj_ok"] = b["vid_ok"]                                  # make_mode forces 3D-only where there is no matching video
        b["text_ids"] = torch.as_tensor(text_ids)
        return b

    def _last_donor_mask(self, b, idx):
        """Slots holding added (donor) objects: valid now but not in the sample's own object set."""
        own = self.S.T["obj_mask"][idx].bool().to(b["obj_mask"].device)
        return (b["obj_mask"].bool() & ~own).cpu().numpy()

    def _composite(self, b, n, i, rng, tries=10, pool=None, relayout=False):
        """Add the movable objects of a training sample with a different instruction onto free table space of sample n (in place)."""
        S = self.S
        pool = self.train_idx if pool is None else pool
        if self.comp_balance:                                   # donor instruction drawn uniformly among the other instructions, then a sample
            others = sorted(set(self.instr_id[pool].tolist()) - {int(self.instr_id[i])})   # (otherwise box episodes dominate the donors)
            if not others:
                return False
            want = others[rng.integers(len(others))]
            cand = pool[self.instr_id[pool] == want]
            j = int(cand[rng.integers(len(cand))])
        else:
            for _ in range(tries):
                j = int(pool[rng.integers(len(pool))])
                if self.instr_id[j] != self.instr_id[i]:
                    break
            else:
                return False
        g = b["obj_geo"][n].numpy()
        mask = b["obj_mask"][n].numpy().astype(bool)
        kind = b["obj_kind"][n].numpy()
        tab = np.flatnonzero(mask & (kind == 1))
        if not len(tab):
            return False
        tx, ty, _, TL, TW = g[tab[0], :5]
        free = list(np.flatnonzero(~mask))
        dj = np.flatnonzero(S.d["obj_mask"][j].astype(bool) & (S.d["obj_kind"][j] == 0))
        rng.shuffle(dj)
        if not getattr(self, "watch", False):
            path = S.d["path_obj"][i][:, :2]
            path = path[np.isfinite(path).all(-1)] if S.d["mover"][i] >= 0 else np.zeros((0, 2))
        else:                                      # watch: keep clear of every base object's future and end spots
            t1 = b["t1_pos"][n].numpy()[b["t1_ok"][n].numpy() & b["t1_present"][n].numpy()][:, :2]
            t2 = (g[:, None, :3] + b["t2_disp"][n].numpy().reshape(len(g), -1, 3))[b["t2_ok"][n].numpy().reshape(len(g), -1)][:, :2]
            path = np.concatenate([t1, t2], 0)
        occ = [(g[s, :2], 0.25 * (g[s, 3] + g[s, 4])) for s in np.flatnonzero(mask & (kind == 0))]
        added = 0
        for d in dj:
            if not free:
                break
            gd = S.d["obj_geo"][j, d].copy()
            r = 0.25 * (gd[3] + gd[4])
            for _ in range(40):
                p = np.array([rng.uniform(tx - TL / 2 + r, tx + TL / 2 - r), rng.uniform(ty - TW / 2 + r, ty + TW / 2 - r)])
                if any(np.linalg.norm(p - q) < 1.2 * (r + rq) + 0.02 for q, rq in occ):
                    continue
                if len(path) and np.min(np.linalg.norm(path - p, axis=1)) < r + 0.05:
                    continue
                s = free.pop(0)
                rt = float(self.rest_top[j, d]) if np.isfinite(float(self.rest_top[j, d])) else float(gd[2])
                if getattr(self, "watch", False):                              # watch: the donor's own token top, not the
                    rt = float(gd[2])                                          # thresholded rest heights of events.json
                gd[0], gd[1], gd[2], gd[5] = p[0], p[1], rt, rt   # resting on the table: top = its resting top (geo[5] repeats top)
                b["obj_geo"][n, s] = torch.from_numpy(gd)
                b["obj_txt"][n, s] = S.T["obj_txt"][j, d]; b["obj_crop"][n, s] = S.T["obj_crop"][j, d]
                b["obj_crop_ok"][n, s] = S.T["obj_crop_ok"][j, d]; b["obj_mask"][n, s] = True; b["obj_kind"][n, s] = 0
                if "mover_set" in b:
                    b["mover_set"][n, s] = False
                if getattr(self, "watch", False):                              # donor objects stay put: true by construction (they
                    b["t1_pos"][n, s] = torch.from_numpy(gd[:3]); b["t1_yaw"][n, s] = float(0.5 * np.arctan2(gd[6], gd[7]))   # belong to another episode)
                    b["t1_ok"][n, s] = bool(b["t1_ok"][n].any()); b["t1_present"][n, s] = True
                    H_ = b["t2_ok"].shape[-1]
                    b["t2_disp"][n].view(len(g), H_, 3)[s] = 0.0; b["t2_ok"][n, s] = True
                    b["t5"][n, s] = False; b["t5_ok"][n, s] = True; b["obj_contact"][n, s] = False; b["obj_contact_ok"][n, s] = True
                    b["obj_contact2"][n, s] = False; b["obj_contact2_ok"][n, s] = True
                b["app_idx"][n, s] = self.app_idx[j, d]
                if hasattr(self, "held_slot"):
                    b["held_slot"][n, s] = self.held_slot[j, d]
                if "uv_ok" in b:
                    b["uv_ok"][n, s] = False; b["objg_ok"][n, s] = False
                occ.append((p, r)); added += 1
                if getattr(self, "comp_log", None) is not None:          # for composite rendering: which donor sample / slot fills (n, s)
                    self.comp_log.append((n, int(s), j, int(d)))
                break
        if added and relayout:
            self._relayout(b, n, i, rng, (tx, ty, TL, TW))
        return added > 0

    def _pair(self, b, n, n2, i, rng, text_ids, cfg, tries=80, pool=None):
        """Paired-instruction composite: base A = row n's sample; donor group B = all movable objects of an in-episode sample j with
        another instruction, placed by one table-plane rigid move (rotation about z + translation) so that B's relative layout and
        its own T1 ends / T2 paths, moved with it, stay true. Two rows, both true by construction:
          row n   instruction A: A's own targets; B's objects stay where they are
          row n2  instruction B: B's targets (its T1 ends / yaw / T2 / T5, rigidly moved); A's objects stay where they are
        Placement: every B position, B end and B path point inside A's table and clear of A's objects; A's ends / paths clear of B's
        objects. Hands and the contact inputs are blanked in both rows (they belong to A's scene). Rows -> self.pair_stats."""
        S = self.S
        st = getattr(self, "pair_stats", None)
        if st is None:
            st = self.pair_stats = dict(tries=0, ok=0, no_slots=0, no_place=0, size_ok=[], size_fail=[])
        st["tries"] += 1
        pool = self.train_idx if pool is None else pool                       # evaluation passes a validation pool
        pool = pool[(self.sub_kind[pool] == "whole") & (self.instr_id[pool] != self.instr_id[i]) & (self.ep_base[pool] != self.ep_base[i])]
        if not len(pool):
            return False
        others = sorted(set(self.instr_id[pool].tolist()))
        want = others[rng.integers(len(others))]
        cand = pool[self.instr_id[pool] == want]
        j = int(cand[rng.integers(len(cand))])
        g = b["obj_geo"][n].numpy()
        mask = b["obj_mask"][n].numpy().astype(bool); kind = b["obj_kind"][n].numpy()
        tab = np.flatnonzero(mask & (kind == 1))
        dj = np.flatnonzero(S.d["obj_mask"][j].astype(bool) & (S.d["obj_kind"][j] == 0))
        free = list(np.flatnonzero(~mask))
        if not len(tab) or not len(dj) or len(free) < len(dj):
            st["no_slots"] += 1; st["size_fail"].append(len(dj)); return False
        tx, ty, _, TL, TW = g[tab[0], :5]
        H_ = b["t2_ok"].shape[-1]
        aobj = np.flatnonzero(mask & (kind == 0))
        occA = [(g[s, :2], 0.25 * (g[s, 3] + g[s, 4])) for s in aobj]
        t1A = b["t1_pos"][n].numpy()[b["t1_ok"][n].numpy() & b["t1_present"][n].numpy()][:, :2]
        t2A = (g[:, None, :3] + b["t2_disp"][n].numpy().reshape(len(g), H_, 3))[b["t2_ok"][n].numpy().reshape(len(g), H_)][:, :2]
        pathA = np.concatenate([t1A, t2A], 0)
        gj = S.d["obj_geo"][j][dj].astype(np.float64)
        rj = 0.25 * (gj[:, 3] + gj[:, 4])
        t1j = S.T["t1_pos"][j][dj].numpy().astype(np.float64); okj1 = (S.T["t1_ok"][j][dj] & S.T["t1_present"][j][dj]).numpy().astype(bool)
        t2j = S.T["t2_disp"][j].view(-1, H_, 3)[dj].numpy().astype(np.float64); okj2 = S.T["t2_ok"][j][dj].numpy().astype(bool)
        c = gj[:, :2].mean(0)
        for _ in range(tries):
            th = rng.uniform(0, 2 * np.pi); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            cn = np.array([rng.uniform(tx - TL / 2, tx + TL / 2), rng.uniform(ty - TW / 2, ty + TW / 2)])
            mv = lambda p: (p - c) @ R.T + cn                                         # noqa: E731
            P = mv(gj[:, :2]); E1 = mv(t1j[:, :2]); E2 = mv(gj[:, None, :2] + t2j[..., :2])
            Q = np.concatenate([P, E1[okj1], E2[okj2]], 0)                           # every B position, end and path point
            RQ = np.concatenate([rj, rj[okj1], np.repeat(rj[:, None], H_, 1)[okj2]], 0)
            if not ((np.abs(Q[:, 0] - tx) <= TL / 2 - RQ) & (np.abs(Q[:, 1] - ty) <= TW / 2 - RQ)).all():
                continue
            if len(occA):
                OA = np.array([o[0] for o in occA]); RA = np.array([o[1] for o in occA])
                if (np.linalg.norm(Q[:, None] - OA[None], axis=-1) < 1.2 * (RQ[:, None] + RA[None]) + 0.02).any():
                    continue
            if len(pathA) and np.min(np.linalg.norm(pathA[:, None] - P[None], axis=-1) - rj[None]) < 0.05:
                continue
            break
        else:
            st["no_place"] += 1; st["size_fail"].append(len(dj)); return False
        sl = free[:len(dj)]
        rot2 = np.array([[np.cos(2 * th), -np.sin(2 * th)], [np.sin(2 * th), np.cos(2 * th)]])
        for q, (s, d) in enumerate(zip(sl, dj)):                                  # row n: B's objects at the moved spots, staying (as _composite)
            gd = S.d["obj_geo"][j, d].copy()
            gd[0], gd[1] = P[q]
            sc = rot2 @ np.array([gd[7], gd[6]])                                    # (cos 2yaw, sin 2yaw) rotated by 2 theta
            gd[7], gd[6] = sc[0], sc[1]
            b["obj_geo"][n, s] = torch.from_numpy(gd)
            b["obj_txt"][n, s] = S.T["obj_txt"][j, d]; b["obj_crop"][n, s] = S.T["obj_crop"][j, d]
            b["obj_crop_ok"][n, s] = S.T["obj_crop_ok"][j, d]; b["obj_mask"][n, s] = True; b["obj_kind"][n, s] = 0
            b["t1_pos"][n, s] = torch.from_numpy(gd[:3]); b["t1_yaw"][n, s] = float(0.5 * np.arctan2(gd[6], gd[7]))
            b["t1_ok"][n, s] = bool(b["t1_ok"][n, aobj].any()); b["t1_present"][n, s] = True     # (as _composite: T1 where A has one)
            if "t1_yaw_ok" in b:
                b["t1_yaw_ok"][n, s] = bool(b["t1_ok"][n, s])
            b["t2_disp"][n].view(len(g), H_, 3)[s] = 0.0; b["t2_ok"][n, s] = True
            b["t5"][n, s] = False; b["t5_ok"][n, s] = True
            b["app_idx"][n, s] = self.app_idx[j, d]
            if "uv_ok" in b:
                b["uv_ok"][n, s] = False
            if "objg_ok" in b:
                b["objg_ok"][n, s] = False
        for k_, v_ in b.items():                                                  # row n2 = a copy of row n ...
            if torch.is_tensor(v_) and v_.dim() >= 1 and v_.shape[0] == b["obj_mask"].shape[0] and k_ not in ("task_emb", "text_ids"):
                v_[n2] = v_[n]
        for row in (n, n2):                                                       # ... hands / contact belong to A's scene, so blank them in both
            b["hand_geo"][row, ..., 0] = 0
            for k_ in ("hand_q_ok", "obj_contact_ok", "obj_contact2_ok"):
                if k_ in b:
                    b[k_][row] = False if b[k_].dtype == torch.bool else 0
        # row n2: instruction B, B moves (its targets rigidly moved), A stays
        ga = b["obj_geo"][n2].numpy()
        for s in aobj:
            b["t1_pos"][n2, s] = torch.from_numpy(ga[s, :3].copy()); b["t1_yaw"][n2, s] = float(0.5 * np.arctan2(ga[s, 6], ga[s, 7]))
            b["t1_ok"][n2, s] = True; b["t1_present"][n2, s] = True
            if "t1_yaw_ok" in b:
                b["t1_yaw_ok"][n2, s] = True
            b["t2_disp"][n2].view(len(g), H_, 3)[s] = 0.0; b["t2_ok"][n2, s] = True
            b["t5"][n2, s] = False; b["t5_ok"][n2, s] = True
        t1yj = S.T["t1_yaw"][j][dj].numpy(); t1yokj = (S.T["t1_yaw_ok"][j][dj].numpy().astype(bool) if "t1_yaw_ok" in S.T else okj1)
        for q, (s, d) in enumerate(zip(sl, dj)):
            b["t1_pos"][n2, s] = torch.tensor([E1[q, 0], E1[q, 1], t1j[q, 2]], dtype=b["t1_pos"].dtype)
            b["t1_yaw"][n2, s] = float(t1yj[q] + th)
            b["t1_ok"][n2, s] = bool(S.T["t1_ok"][j, d]); b["t1_present"][n2, s] = bool(S.T["t1_present"][j, d])
            if "t1_yaw_ok" in b:
                b["t1_yaw_ok"][n2, s] = bool(t1yokj[q])
            dd = t2j[q].copy(); dd[:, :2] = dd[:, :2] @ R.T
            b["t2_disp"][n2].view(len(g), H_, 3)[s] = torch.from_numpy(dd.astype(np.float32)); b["t2_ok"][n2, s] = torch.from_numpy(okj2[q])
            b["t5"][n2, s] = S.T["t5"][j, d]; b["t5_ok"][n2, s] = S.T["t5_ok"][j, d]
        a = self.alt[j][self.alt[j] >= 0] if cfg.get("paraphrase", True) else self.alt[j][:1]
        tid = int(a[rng.integers(len(a))]) if len(a) else int(self.orig[j])
        text_ids[n2] = tid; b["task_emb"][n2] = self.bank[tid]
        if "rem_frac" in b:                                                      # row n2's remaining motion is B's own (A stays)
            b["rem_frac"][n2] = S.T["rem_frac"][j]; b["rem_ok"][n2] = S.T["rem_ok"][j]
        st["ok"] += 1; st["size_ok"].append(len(dj)); st["last"] = (j, list(sl), list(dj))
        return True

    def _rigid(self, b, n, rng, tries=40, th=None, cn=None):
        """Whole-scene rigid re-layout: every non-table token of row n and the hand tokens move by one table-plane rigid move (rotation
        about z through the objects' centroid + translation); the table stays. Applied to positions, yaw, T1 ends / end yaw, T2
        displacements (rotated) and hand positions / directions, so every relation between objects, ends and hands is unchanged.
        Draws where any object, T1 end or T2 point would leave the table (minus its footprint radius) are rejected. The row's video
        no longer matches, so it is marked 'altered' (vid_ok off). With th / cn given, that exact move is used if it fits
        (the evaluation's same-sample test). Returns True if moved."""
        g = b["obj_geo"][n].numpy().copy()
        mask = b["obj_mask"][n].numpy().astype(bool); kind = b["obj_kind"][n].numpy()
        tab = np.flatnonzero(mask & (kind == 1)); mov = np.flatnonzero(mask & (kind != 1))
        if not len(tab) or not len(mov):
            return False
        tx, ty, _, TL, TW = g[tab[0], :5]
        H_ = b["t2_ok"].shape[-1]
        r = 0.25 * (g[mov, 3] + g[mov, 4])
        t1 = b["t1_pos"][n].numpy().copy(); ok1 = (b["t1_ok"][n] & b["t1_present"][n]).numpy().astype(bool)[mov]
        t2 = b["t2_disp"][n].view(len(g), H_, 3).numpy().copy(); ok2 = b["t2_ok"][n].numpy().reshape(len(g), H_).astype(bool)[mov]
        c = g[mov, :2].mean(0)
        Q0 = np.concatenate([g[mov, :2], t1[mov, :2][ok1], (g[mov, None, :2] + t2[mov][..., :2])[ok2]], 0)   # every point that must stay on
        RQ0 = np.concatenate([r, r[ok1], np.repeat(r[:, None], H_, 1)[ok2]], 0)                         # the table, and its radius
        for q in range(1 if th is not None else tries):
            th_ = rng.uniform(0, 2 * np.pi) if th is None else th
            R = np.array([[np.cos(th_), -np.sin(th_)], [np.sin(th_), np.cos(th_)]])
            if cn is None:                                                      # the translation range that keeps every point on the table
                O = (Q0 - c) @ R.T
                lo = np.array([tx - TL / 2, ty - TW / 2]) + (RQ0[:, None] - O).max(0)
                hi = np.array([tx + TL / 2, ty + TW / 2]) - (RQ0[:, None] + O).max(0)
                if (lo > hi).any():
                    continue
                cn_ = rng.uniform(lo, hi)
            else:
                cn_ = np.asarray(cn)
            mv = lambda p: (p - c) @ R.T + cn_                                       # noqa: E731
            P = mv(g[mov, :2]); E1 = mv(t1[mov, :2]); E2 = mv(g[mov, None, :2] + t2[mov][..., :2])
            Q = np.concatenate([P, E1[ok1], E2[ok2]], 0)
            RQ = np.concatenate([r, r[ok1], np.repeat(r[:, None], H_, 1)[ok2]], 0)
            if ((np.abs(Q[:, 0] - tx) <= TL / 2 - RQ) & (np.abs(Q[:, 1] - ty) <= TW / 2 - RQ)).all():
                break
        else:
            return False
        if th is not None and not ((np.abs(Q[:, 0] - tx) <= TL / 2 - RQ) & (np.abs(Q[:, 1] - ty) <= TW / 2 - RQ)).all():
            return False
        rot2 = np.array([[np.cos(2 * th_), -np.sin(2 * th_)], [np.sin(2 * th_), np.cos(2 * th_)]])
        g[mov, :2] = P
        cs = np.stack([g[mov, 7], g[mov, 6]], -1) @ rot2.T                          # (cos 2yaw, sin 2yaw) rotated by 2 theta
        g[mov, 7], g[mov, 6] = cs[:, 0], cs[:, 1]
        b["obj_geo"][n] = torch.from_numpy(g)
        t1[mov, :2] = E1
        b["t1_pos"][n] = torch.from_numpy(t1)
        b["t1_yaw"][n, torch.as_tensor(mov)] = b["t1_yaw"][n, torch.as_tensor(mov)] + float(th_)
        t2[mov, :, :2] = t2[mov, :, :2] @ R.T
        b["t2_disp"][n] = torch.from_numpy(t2.reshape(b["t2_disp"][n].shape))
        h = b["hand_geo"][n].numpy().copy()                                          # hands: the same rigid move (present ones)
        for k in np.flatnonzero(h[:, 0] > 0):
            h[k, 1:3] = mv(h[k, 1:3][None])[0]
            h[k, 7:9] = h[k, 7:9] @ R.T; h[k, 10:12] = h[k, 10:12] @ R.T
            ex, ey = h[k, 4], h[k, 5]
            h[k, 4] = abs(np.cos(th_)) * ex + abs(np.sin(th_)) * ey; h[k, 5] = abs(np.sin(th_)) * ex + abs(np.cos(th_)) * ey
        b["hand_geo"][n] = torch.from_numpy(h)
        if "uv_ok" in b:
            b["uv_ok"][n] = False
        b["altered"][n] = True
        self._rigid_last = (float(th_), cn_.tolist(), c.tolist())
        st_ = getattr(self, "rigid_stats", None)
        if st_ is not None:                                                     # how big the moves are
            st_.setdefault("shift_cm", []).append(float(100 * np.linalg.norm(cn_ - c)))
            st_.setdefault("rot_deg", []).append(float(np.degrees(abs((th_ + np.pi) % (2 * np.pi) - np.pi))))
        return True

    def _relayout(self, b, n, i, rng, table):
        """Same-sampler composites: every movable object resting on the table (the scene's own and the added ones) is put at a new random
        free spot by one sampler, so the two sets cannot be told apart by their layout. Kept in place: the destination of the label and
        any object that supports another (something lies on / in it), so 'already placed' states stay true. Yaw unchanged."""
        tx, ty, TL, TW = table
        g = b["obj_geo"][n].numpy().copy(); mask = b["obj_mask"][n].numpy().astype(bool); kind = b["obj_kind"][n].numpy()
        rt = self.rest_top[i].numpy().copy()
        don = mask & ~self.S.d["obj_mask"][i].astype(bool)
        rt[don] = g[don, 2]                                       # donors were just put down at their resting top
        mov = np.flatnonzero(mask & (kind == 0))
        tg = int(self.S.d["target"][i])
        rad = 0.25 * (g[:, 3] + g[:, 4])
        support = np.zeros(len(g), bool)
        for a_ in mov:
            for c in mov:
                if a_ != c and np.linalg.norm(g[c, :2] - g[a_, :2]) < rad[a_] and g[c, 2] > g[a_, 2] + 0.01:
                    support[a_] = True
        resting = np.isfinite(rt) & (np.abs(g[:, 2] - rt) < 0.015)
        if self.relay_keep_inside:
            # the height test above misses objects lying flat in a container (a wrench on the toolbox floor is < 1.5 cm above its table
            # height), so also keep in place every object whose footprint centre lies inside another object's footprint, and every
            # object that holds one, so placed states stay true
            for a_ in mov:
                for c in mov:
                    if a_ != c and np.linalg.norm(g[c, :2] - g[a_, :2]) < rad[a_]:
                        support[a_] = True; support[c] = True
        if not getattr(self, "relay_keep_dest", True):                   # endref: no destination label to preserve
            tg = -1
        move = [s_ for s_ in mov if resting[s_] and not support[s_] and s_ != tg]
        keep = [s_ for s_ in mov if s_ not in move]
        occ = [(g[s_, :2], rad[s_]) for s_ in keep]
        rng.shuffle(move)
        for s_ in move:
            r = rad[s_]
            for _ in range(60):
                p = np.array([rng.uniform(tx - TL / 2 + r, tx + TL / 2 - r), rng.uniform(ty - TW / 2 + r, ty + TW / 2 - r)])
                if all(np.linalg.norm(p - q) >= 1.2 * (r + rq) + 0.02 for q, rq in occ):
                    g[s_, 0], g[s_, 1] = p; occ.append((p, r)); break
            else:
                return False                                      # no free spot: keep the ordinary composite (no overlapping layout)
        b["obj_geo"][n] = torch.from_numpy(g)
        b["relaid"][n] = True
        return True
