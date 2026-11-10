"""Contact-event head for the mover's path, with its label loader and targets. Not used by the trained models, which predict
contact with the T5 head of cleantables.worldmodel.model.UWMW.

Along the 32-step sampled path of the moving object the head predicts, for each hand, P(in contact) per step and the
grasp-on / grasp-off times (seconds from the snapshot).

Labels, one file per episode: <CT_DATA>/perception_v2/contact/<episode>/contact.npz (read by `load_contact_labels`)
    t        (T,)    float32  video time (s)
    obj_ids  (n,)    str      object ids, as in the episode's tracks and twin scene.json
    state    (2,T,n) int8     hand (0 = left, 1 = right) x frame x object: CONTACT_STATES index, -1 = unknown (hand not visible)
    conf     (2,T,n) float16  label confidence in [0, 1]
Targets: `make_targets(track, t_snap, mover_id, path_t)` maps the labels onto the 32 path steps of a move (path_t (32,) are
the sample times, from the dataset's path_dt) and returns
    contact (2,32) float  1 = grasped / touching the mover, 0 = free / released, NaN = unknown
    t_on, t_off (2,) float seconds from t_snap of the first contact / last release in the move; NaN if none
    ok (2,) bool          the hand has at least 8 known steps
Head: `ContactHead(d=256)` reads the flow decoder's context c = [glob, mover, target] (model.heads_from) and the sampled path
(32, 3, PS units, relative to the mover), and cross-attends to the trunk memory (E["mem"]), so hand and video tokens are
used when present. It outputs logits (2, 32); on / off times are the first and last steps above 0.5, and `loss` adds a
Huber term on soft on / off times.
"""
from __future__ import annotations

from cleantables import CT_DATA
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTACT_STATES = ("free", "touching", "grasped", "released")
IN_CONTACT = {1, 2}                      # touching, grasped
NP = 32
LABEL_ROOT = Path(CT_DATA + "/perception_v2/contact")


@dataclass
class ContactTrack:
    episode: str
    t: np.ndarray            # (T,)
    obj_ids: list
    state: np.ndarray        # (2,T,n) int8
    conf: np.ndarray         # (2,T,n)


def load_contact_labels(episode, root=LABEL_ROOT):
    """Load an episode's contact labels; None when the episode has no label file."""
    f = Path(root) / episode / "contact.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return ContactTrack(episode, z["t"].astype(np.float32), [str(x) for x in z["obj_ids"]], z["state"].astype(np.int8), z["conf"].astype(np.float32))


def make_targets(track: ContactTrack, t_snap: float, mover_id: str, path_t: np.ndarray, min_conf=0.5, min_known=8):
    """Labels -> per-path-step targets for one move (see module docstring)."""
    j = track.obj_ids.index(mover_id)
    idx = np.clip(np.searchsorted(track.t, path_t), 0, len(track.t) - 1)
    st = track.state[:, idx, j]; cf = track.conf[:, idx, j]
    known = (st >= 0) & (cf >= min_conf)
    contact = np.where(known, np.isin(st, list(IN_CONTACT)).astype(np.float32), np.nan)
    t_on, t_off = np.full(2, np.nan, np.float32), np.full(2, np.nan, np.float32)
    for h in range(2):
        on = np.flatnonzero(contact[h] == 1)
        if len(on):
            t_on[h], t_off[h] = path_t[on[0]] - t_snap, path_t[on[-1]] - t_snap
    return dict(contact=contact, t_on=t_on, t_off=t_off, ok=known.sum(1) >= min_known)


class ContactHead(nn.Module):
    def __init__(s, d=256, heads=8, layers=2, drop=0.1):
        super().__init__()
        s.p_in = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d)); s.pos = nn.Parameter(torch.randn(NP, d) * 0.02)
        s.c_in = nn.Linear(3 * d, d); s.hand = nn.Parameter(torch.randn(2, d) * 0.02)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu"), layers)
        s.out = nn.Linear(d, 1)

    def forward(s, E, c, path_rel):
        """E: trunk output dict (needs mem, pad); c (B,3d) from model.heads_from; path_rel (B,32,3) sampled path, PS units.
        Returns logits (B,2,32): per hand, per path step, in contact with the mover."""
        B = c.shape[0]
        q = s.p_in(path_rel) + s.pos + s.c_in(c)[:, None]                         # (B,32,d)
        q = (q[:, None] + s.hand[None, :, None]).reshape(B * 2, NP, -1)
        mem = E["mem"].repeat_interleave(2, 0); pad = E["pad"].repeat_interleave(2, 0)
        return s.out(s.dec(q, mem, memory_key_padding_mask=pad)).view(B, 2, NP)

    @staticmethod
    def events(logits, path_t, thr=0.5):
        """Read-out: (t_on, t_off) per hand in seconds from the first path time (NaN when the hand makes no contact)."""
        p = torch.sigmoid(logits) > thr
        any_ = p.any(-1)
        first = p.float().argmax(-1); last = NP - 1 - p.flip(-1).float().argmax(-1)
        tt = torch.as_tensor(path_t, device=logits.device, dtype=torch.float32)
        nan = torch.full_like(first, 0, dtype=torch.float32) * float("nan")
        return torch.where(any_, tt[first] - tt[0], nan), torch.where(any_, tt[last] - tt[0], nan)

    @staticmethod
    def loss(logits, contact, ok, path_t, w_time=0.5):
        """BCE on known steps + Huber on soft on / off times (expected first / last contact step under the predicted per-step probs)."""
        m = torch.isfinite(contact) & ok[..., None]
        bce = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(contact), reduction="none")
        l1 = (bce * m).sum() / m.sum().clamp(min=1)
        p = torch.sigmoid(logits); tt = torch.as_tensor(path_t, device=logits.device, dtype=torch.float32)
        # soft first-on: P(step k is the first contact) = p_k * prod_{j<k} (1 - p_j)
        surv = torch.cumprod(torch.cat([torch.ones_like(p[..., :1]), 1 - p[..., :-1]], -1), -1)
        on_soft = (p * surv * tt).sum(-1) / (p * surv).sum(-1).clamp(min=1e-6)
        pr = p.flip(-1); survr = torch.cumprod(torch.cat([torch.ones_like(pr[..., :1]), 1 - pr[..., :-1]], -1), -1)
        off_soft = (pr * survr * tt.flip(0)).sum(-1) / (pr * survr).sum(-1).clamp(min=1e-6)
        c01 = torch.nan_to_num(contact)
        has = (m & (c01 > 0.5)).any(-1)
        t_on = torch.where(has, tt[(m & (c01 > 0.5)).float().argmax(-1)], torch.zeros_like(on_soft))
        t_off = torch.where(has, tt[NP - 1 - (m & (c01 > 0.5)).flip(-1).float().argmax(-1)], torch.zeros_like(on_soft))
        lt = (F.huber_loss(on_soft, t_on, reduction="none") + F.huber_loss(off_soft, t_off, reduction="none")) * has.float()
        return l1 + w_time * lt.sum() / has.float().sum().clamp(min=1)


if __name__ == "__main__":                                                       # shape smoke test (CPU, random tensors)
    B, d = 4, 256
    E = dict(mem=torch.randn(B, 63, d), pad=torch.zeros(B, 63, dtype=torch.bool))
    h = ContactHead(d)
    lg = h(E, torch.randn(B, 3 * d), torch.randn(B, NP, 3))
    path_t = np.linspace(0, 3.1, NP)
    tgt = torch.zeros(B, 2, NP); tgt[:, 1, 5:25] = 1; tgt[:, 0] = float("nan")
    L = ContactHead.loss(lg, tgt, torch.tensor([[False, True]] * B), path_t)
    on, off = ContactHead.events(lg, path_t)
    print("logits", tuple(lg.shape), "loss", float(L), "t_on", on[0].tolist(), "params", sum(p.numel() for p in h.parameters()))
