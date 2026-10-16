"""Imitation training of the world model on the video snapshots: augmentation, losses and the per-fold training CLI.

    python -m cleantables.worldmodel.train --fold -1      # all data -> runs/worldmodel/imitation/final_seed{s}.pt
    python -m cleantables.worldmodel.train --fold 0       # folds 0..4 -> fold{f}_seed{s}.pt (bootstrap-resampled)

Augmentations that change the 3D scene without a matching video (rotation, scaling, re-placement) force 3D-only mode for
that sample, since the V-JEPA tokens would contradict the scene. Mirroring swaps in the tokens of the flipped clip and
masks the latent target (there is no flipped future latent).
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.paths import WM_RUNS

from .config import TrainConfig, load_train_config
from .dataset import DATASET, MAX_OBJECTS, Dataset, get_batch
from .model import MODE_3D, MODE_BOTH, N_PATH, PATH_SCALE, WorldModel

log = logging.getLogger(__name__)
RUN = "imitation"
END_OFFSET_UNIT = 0.1  # metres per unit of the end-offset mixture
CARRY_RANK_PAIRS = 48  # random step pairs per path for the progress ranking loss


def checkpoint_name(fold: int, seed: int) -> str:
    return f"final_seed{seed}.pt" if fold < 0 else f"fold{fold}_seed{seed}.pt"


def load_models(paths, device: str = "cuda") -> list:
    models = []
    for p in paths:
        m = WorldModel().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        models.append(m.eval())
    return models


def mean_w(loss: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def rotate_z(v: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    x, y = v[..., 0], v[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y, *[v[..., i] for i in range(2, v.shape[-1])]], -1)


# ---------------------------------------------------------------------------------------------- augmentation
def _mirror(b: dict, mir: torch.Tensor) -> None:
    """x -> -x, yaw -> -yaw; the clip tokens switch to those of the horizontally flipped video."""
    sgn = torch.where(mir, -1.0, 1.0)
    g, h = b["obj_geo"].clone(), b["hand_geo"].clone()
    g[..., 0] *= sgn[:, None]
    g[..., 6] *= sgn[:, None]
    h[..., 1] *= sgn[:, None]
    a1, a2 = h[..., 7:10].clone(), h[..., 10:13].clone()
    h[..., 7:10] = torch.where(mir[:, None, None], torch.stack([a1[..., 0], -a1[..., 1], -a1[..., 2]], -1), a1)
    h[..., 10:13] = torch.where(mir[:, None, None], torch.stack([-a2[..., 0], a2[..., 1], a2[..., 2]], -1), a2)
    b["obj_geo"], b["hand_geo"] = g, h
    for k in ("path_obj", "path_hand"):
        b[k] = b[k].clone()
        b[k][..., 0] *= sgn[:, None]
    ep = b["end_pose"].clone()
    ep[:, 0] *= sgn
    ep[:, 3] *= sgn
    b["end_pose"] = ep
    eo = b["end_off"].clone()
    eo[:, 0] *= sgn
    b["end_off"] = eo
    b["vj_tok"] = torch.where(mir[:, None, None], b["vj_tok_flip"], b["vj_tok"])


def _rotate(b: dict, th: torch.Tensor) -> None:
    c, s = th.cos(), th.sin()
    g = b["obj_geo"].clone()
    g[..., :3] = rotate_z(g[..., :3], c[:, None], s[:, None])
    c2, s2 = (2 * th).cos()[:, None], (2 * th).sin()[:, None]
    sn, cs = g[..., 6].clone(), g[..., 7].clone()
    g[..., 6] = sn * c2 + cs * s2
    g[..., 7] = cs * c2 - sn * s2
    b["obj_geo"] = g
    h = b["hand_geo"].clone()
    h[..., 1:4] = rotate_z(h[..., 1:4], c[:, None], s[:, None])
    h[..., 7:10] = rotate_z(h[..., 7:10], c[:, None], s[:, None])
    h[..., 10:13] = rotate_z(h[..., 10:13], c[:, None], s[:, None])
    ex, ey = h[..., 4].clone(), h[..., 5].clone()
    h[..., 4] = c.abs()[:, None] * ex + s.abs()[:, None] * ey
    h[..., 5] = s.abs()[:, None] * ex + c.abs()[:, None] * ey
    b["hand_geo"] = h
    for k in ("path_obj", "path_hand"):
        b[k] = rotate_z(b[k], c[:, None], s[:, None])
    ep = b["end_pose"].clone()
    ep[:, :3] = rotate_z(ep[:, :3], c, s)
    ep[:, 3] = ep[:, 3] + th
    b["end_pose"] = ep


def _scale(b: dict, sc: torch.Tensor) -> None:
    """Scale the whole scene about the origin."""
    b["obj_geo"][..., :6] *= sc[:, None, None]
    b["hand_geo"][..., 1:7] *= sc[:, None, None]
    b["path_obj"] = b["path_obj"] * sc[:, None, None]
    b["path_hand"] = b["path_hand"] * sc[:, None, None]
    b["end_pose"][:, :3] *= sc[:, None]
    b["end_off"] = b["end_off"] * sc[:, None]


def _replace_bystanders(b: dict, chosen: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Move half of the objects not involved in the move to random free spots that keep the label valid: at least as far
    from the mover as its target, and clear of other objects. Returns which samples changed."""
    B, dev = b["mover"].shape[0], b["mover"].device
    N = b["obj_geo"].shape[1]
    ar = torch.arange(N, device=dev)[None]
    mv, tg = b["mover"], b["target"]
    movable = b["obj_mask"].bool() & (b["obj_kind"] == 0)
    involved = (ar == mv[:, None]) | (ar == tg[:, None])
    # this draw selects nothing; it keeps the random stream identical to the one the checkpoints were trained with
    always = torch.rand(B, device=dev, generator=g)[:, None] < 2
    cand = movable & ~involved & chosen[:, None] & always
    pick = cand & (torch.rand(B, N, device=dev, generator=g) < 0.5)
    gm = b["obj_geo"]
    xy = torch.where(movable[..., None], gm[..., :2], torch.full_like(gm[..., :2], float("nan")))
    lo = torch.nan_to_num(xy.nan_to_num(1e9).min(1).values, 0)
    hi = xy.nan_to_num(-1e9).max(1).values
    mpos = gm[torch.arange(B), mv.clamp(min=0), :2]
    tpos = gm[torch.arange(B), tg.clamp(min=0, max=N - 1), :2]
    d_target = torch.where((tg >= 0) & (tg < MAX_OBJECTS), (tpos - mpos).norm(dim=-1), torch.zeros(B, device=dev))
    new_xy = lo[:, None, None] + torch.rand(B, N, 8, 2, device=dev, generator=g) * (hi - lo)[:, None, None]
    ok = (new_xy - mpos[:, None, None]).norm(dim=-1) >= d_target[:, None, None] + 0.03
    # clear of other objects: centre distance > 0.6 * (L + W) / 2 of the moved object (the table is ignored)
    blocking = torch.where(movable | (b["obj_kind"] == 2), 0.0, 9.0)[:, None, None]
    dd = (new_xy[:, :, :, None] - gm[:, None, None, :, :2]).norm(dim=-1) + blocking
    radius = 0.5 * (gm[..., 3] + gm[..., 4])[:, :, None, None] * 0.6
    ok = ok & (dd > radius).all(-1)
    first = ok.float().argmax(-1)
    nxy = torch.gather(new_xy, 2, first[..., None, None].expand(B, N, 1, 2)).squeeze(2)
    do = pick & ok.any(-1)
    gm = gm.clone()
    gm[..., :2] = torch.where(do[..., None], nxy, gm[..., :2])
    gm[..., 2] = torch.where(do, gm[..., 5], gm[..., 2])
    b["obj_geo"] = gm
    return chosen & do.any(1)


def augment(b: dict, cfg: TrainConfig, g: torch.Generator) -> dict:
    B, dev = b["mover"].shape[0], b["mover"].device

    def rand():
        return torch.rand(B, device=dev, generator=g)

    mir = rand() < cfg.p_mirror
    rot = rand() < cfg.p_rotate
    scl = rand() < cfg.p_scale
    rep = rand() < cfg.p_replace
    _mirror(b, mir)
    _rotate(b, (rand() * 2 * math.pi) * rot.float())
    _scale(b, torch.where(scl, torch.exp((rand() * 2 - 1) * math.log(cfg.max_scale)), torch.ones(B, device=dev)))
    changed = _replace_bystanders(b, rep, g)
    N = b["obj_geo"].shape[1]
    ar = torch.arange(N, device=dev)[None]
    movable = b["obj_mask"].bool() & (b["obj_kind"] == 0)
    involved = (ar == b["mover"][:, None]) | (ar == b["target"][:, None])
    drop = movable & ~involved & (torch.rand(B, N, device=dev, generator=g) < cfg.p_object_drop)
    b["obj_mask"] = b["obj_mask"].bool() & ~drop
    b["aug_video_bad"] = rot | scl | changed
    b["aug_jepa_bad"] = rot | scl | changed | mir
    return b


def sample_modes(b: dict, cfg: TrainConfig, g: torch.Generator) -> torch.Tensor:
    """Modality dropout; samples whose video no longer matches the scene (or have none) fall back to 3D only."""
    B, dev = b["mover"].shape[0], b["mover"].device
    r = torch.rand(B, device=dev, generator=g)
    mode = torch.where(r < cfg.p_both, 0, torch.where(r < cfg.p_both + cfg.p_3d, 1, 2))
    return torch.where(b["aug_video_bad"] | ~b["vj_ok"].bool(), torch.full_like(mode, MODE_3D), mode)


# ---------------------------------------------------------------------------------------------- losses
def path_targets(b: dict):
    """Object and hand paths relative to the mover's start (path units), their weights, and the start."""
    B = b["mover"].shape[0]
    mv = b["mover"].clamp(min=0)
    org = b["obj_geo"][torch.arange(B), mv, :3]
    po = (b["path_obj"] - org[:, None]) / PATH_SCALE
    ph = (b["path_hand"] - org[:, None]) / PATH_SCALE
    has_mover = (b["mover"] >= 0).float()
    wo = torch.isfinite(po).all(-1).float() * has_mover[:, None]
    hand_seen = (b["hand_geo"][:, 0, 0] > 0.5).float()[:, None]
    wh = torch.isfinite(ph).all(-1).float() * b["path_hand_ok"].float() * hand_seen * has_mover[:, None]
    x1 = torch.cat([torch.nan_to_num(po), torch.nan_to_num(ph)], -1)
    w = torch.cat([wo[..., None].expand(-1, -1, 3), wh[..., None].expand(-1, -1, 3)], -1)
    return x1, w, org


def mixture_nll(target: torch.Tensor, z: dict) -> torch.Tensor:
    sd = z["logs"].exp()
    nll = (0.5 * ((target[:, None] - z["mu"]) / sd) ** 2 + z["logs"]).sum(-1) - F.log_softmax(z["logit"], -1)
    return -torch.logsumexp(-nll, -1)


def _progress_losses(m, b, E, c, org, mode, cfg, g, L) -> None:
    dev = b["mover"].device
    B = b["mover"].shape[0]
    pv = torch.sigmoid(m.clip_progress_logit(E))
    ok_video = b["prog_video_ok"].float()
    label = torch.nan_to_num(b["prog_video"])
    # the label term gets its own head call: sharing one changes the float summation order of the backward pass, and
    # seeded runs would no longer reproduce the released checkpoints bit for bit
    logit = m.clip_progress_logit(E).float()
    L["prog_video"] = cfg.w_prog_abs * mean_w(
        F.binary_cross_entropy_with_logits(logit, label.clamp(0, 1), reduction="none"), ok_video
    )
    # later snapshots of the same episode must score higher
    same = (b["video_id"][:, None] == b["video_id"][None]) & (ok_video[:, None] > 0) & (ok_video[None] > 0)
    later = (label[:, None] - label[None]) > 0.05
    L["prog_rank"] = 0.5 * cfg.w_prog * mean_w(F.relu(0.05 - (pv[:, None] - pv[None])), (same & later).float())
    # the time-reversed clip must score lower than the real one
    b_rev = dict(b)
    b_rev["vj_tok"] = b["vj_tok"].flip(1)
    pv_rev = torch.sigmoid(m.clip_progress_logit(m.encode(b_rev, mode)))
    w_rev = ok_video * ((mode == MODE_BOTH) & b["vj_ok"].bool()).float() * (~b["aug_video_bad"]).float()
    L["prog_rev"] = 0.5 * cfg.w_prog * mean_w(F.relu(0.1 - (pv - pv_rev)), w_rev)
    ttm = b["time_to_move"]
    L["ttm"] = (
        0.3
        * cfg.w_prog
        * mean_w(
            F.huber_loss(
                m.p_ttm(E["cls"][:, 0]).squeeze(-1), torch.log1p(torch.nan_to_num(ttm).clamp(min=0)), reduction="none"
            ),
            torch.isfinite(ttm).float(),
        )
    )
    # per-step progress along the carry; the reversed path keeps its labels attached to the content
    steps = torch.nan_to_num(b["prog_steps"])
    ok_steps = (torch.isfinite(b["prog_steps"]).all(-1) & (b["mover"] >= 0)).float()
    p6 = torch.cat([b["path_obj"] - org[:, None], b["path_hand"] - org[:, None]], -1).nan_to_num()
    pf = m.path_progress(E, c, p6)
    pr = m.path_progress(E, c, p6.flip(1))
    hub = F.huber_loss(pf, steps, reduction="none", delta=0.1).mean(-1) + F.huber_loss(
        pr, steps.flip(1), reduction="none", delta=0.1
    ).mean(-1)
    i = torch.randint(0, N_PATH - 4, (B, CARRY_RANK_PAIRS), device=dev, generator=g)
    j = (i + torch.randint(3, 12, (B, CARRY_RANK_PAIRS), device=dev, generator=g)).clamp(max=N_PATH - 1)

    def rank(p, lab):
        order = torch.sign(torch.gather(lab, 1, j) - torch.gather(lab, 1, i))
        return F.relu(0.05 - (torch.gather(p, 1, j) - torch.gather(p, 1, i)) * order).mean(-1)

    L["prog_steps"] = cfg.w_prog * mean_w(hub + rank(pf, steps) + rank(pr, steps.flip(1)), ok_steps)


def compute_loss(m: WorldModel, b: dict, cfg: TrainConfig, g: torch.Generator) -> dict:
    B, dev = b["mover"].shape[0], b["mover"].device
    mode = sample_modes(b, cfg, g)
    E = m.encode(b, mode)
    mv, tg = b["mover"], b["target"]
    L = {}
    mover_logits, target_logits = m.pointers(E, mv)
    ok_target = ((tg >= 0) & (mv >= 0)).float()
    # order-free next object: probability mass on any object still to be moved in this episode
    remaining = b["mover_set"].bool() & b["obj_mask"].bool()
    logp = F.log_softmax(mover_logits.float(), -1).masked_fill(~remaining, -1e4)
    L["mover"] = mean_w(-torch.logsumexp(logp, -1), remaining.any(-1).float())
    L["target"] = mean_w(F.cross_entropy(target_logits, tg.clamp(min=0), reduction="none"), ok_target)
    c, rel, mdn = m.heads_from(E, mv, tg)
    ok_rel = ((b["relation"] >= 0) & (mv >= 0)).float()
    L["relation"] = 0.5 * mean_w(F.cross_entropy(rel, b["relation"].clamp(min=0), reduction="none"), ok_rel)
    end_off = b["end_off"] / END_OFFSET_UNIT
    ok_end = (torch.isfinite(end_off).all(-1) & (tg >= 0) & (mv >= 0)).float()
    L["end_off"] = 0.5 * mean_w(mixture_nll(torch.nan_to_num(end_off), mdn), ok_end)
    # paths are anchored at the end: a mixture over the end displacement, and flow matching on the residual
    # around the straight line to the true end
    x1, w, org = path_targets(b)
    end = x1[:, -1, :3]
    L["end_disp"] = mean_w(mixture_nll(end, m.end_disp(c)), (w[:, -1, 0] > 0).float())
    line = torch.linspace(0, 1, N_PATH, device=dev)[None, :, None] * end[:, None, :]
    x1 = torch.cat([x1[..., :3] - line, x1[..., 3:6] - line], -1)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=dev, generator=g)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v = m.velocity(E, c, xt, t)
    L["flow"] = cfg.w_flow * mean_w((v - (x1 - x0)) ** 2, w)
    _progress_losses(m, b, E, c, org, mode, cfg, g, L)
    # future V-JEPA latent
    pj = m.future_latent(E, b, mode)
    target = torch.nan_to_num(b["vj_future_glob"]).detach()
    ok_jepa = (b["vj_future_ok"].bool() & ~b["aug_jepa_bad"]).float()
    L["jepa"] = cfg.w_jepa * mean_w(1 - F.cosine_similarity(pj.float(), target, dim=-1), ok_jepa)
    # p(seen): negatives get the V-JEPA tokens of another episode (video and scene disagree)
    half = B // 2
    b_mixed = dict(b)
    vj = b["vj_tok"].clone()
    vj[:half] = vj.roll(1, 0)[:half]
    b_mixed["vj_tok"] = vj
    differs = torch.ones(B, device=dev)
    differs[:half] = (b["video_id"][:half] != b["video_id"].roll(1, 0)[:half]).float()
    E2 = m.encode(b_mixed, torch.zeros(B, dtype=torch.long, device=dev))
    label = torch.ones(B, device=dev)
    label[:half] = 0
    ok_seen = ((differs > 0) & b["vj_ok"].bool() & ~b["aug_video_bad"]).float()
    seen_logit = m.seen(E2["cls"][:, 3]).squeeze(-1)
    L["seen"] = cfg.w_seen * mean_w(F.binary_cross_entropy_with_logits(seen_logit, label, reduction="none"), ok_seen)
    return L


# ---------------------------------------------------------------------------------------------- training
def learning_rate(cfg: TrainConfig, it: int, steps: int) -> float:
    """Linear warm-up, then cosine decay."""
    return cfg.lr * min(1, (it + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * it / steps))


def train_model(
    G: dict, train_idx, cfg: TrainConfig, seed: int = 0, device: str = "cuda", bootstrap: bool = False, tag: str = ""
) -> WorldModel:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = WorldModel(d=cfg.d_model, layers=cfg.layers).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr = np.asarray(train_idx)
    if bootstrap:
        tr = rng.choice(tr, len(tr))
    t0 = time.time()
    for it in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = learning_rate(cfg, it, cfg.steps)
        idx = torch.as_tensor(tr[rng.integers(0, len(tr), cfg.batch_size)], device=device)
        b = augment(get_batch(G, idx), cfg, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = compute_loss(m, b, cfg, gen)
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
        opt.step()
        if it % 250 == 0 or it == cfg.steps - 1:
            parts = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items())
            log.info("%s it %d loss %.3f %s", tag, it, loss.item(), parts)
    m.train_seconds = time.time() - t0
    return m.eval()


def train_fold(fold: int, cfg: TrainConfig, run_dir: Path, dataset: Path = DATASET, device: str = "cuda") -> None:
    """Fold -1 trains on everything; folds 0..4 leave that fold out and bootstrap-resample the rest."""
    data = Dataset(dataset)
    G = data.to(device)
    train_idx = np.flatnonzero(data.d["fold"] != fold) if fold >= 0 else np.arange(data.n)
    run_dir.mkdir(parents=True, exist_ok=True)
    for s in cfg.seeds:
        m = train_model(
            G,
            train_idx,
            cfg,
            seed=1000 * (fold + 1) + s,
            device=device,
            bootstrap=fold >= 0,
            tag=f"fold {fold} seed {s}",
        )
        torch.save(m.state_dict(), run_dir / checkpoint_name(fold, s))
        log.info("fold %d seed %d: %d training samples, %.0f s", fold, s, len(train_idx), m.train_seconds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the world model on the video snapshots (one fold, all seeds).")
    ap.add_argument("--fold", type=int, required=True, help="held-out fold 0-4, or -1 to train on all data")
    ap.add_argument("--run", default=RUN, help="run name under runs/worldmodel/")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default from the config)")
    ap.add_argument("--steps", type=int, default=None, help="training steps (default from the config)")
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default configs/worldmodel.yaml)")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_train_config(a.config) if a.config else load_train_config()
    if a.seeds:
        cfg.seeds = [int(x) for x in a.seeds.split(",")]
    if a.steps:
        cfg.steps = a.steps
    log.info("config %s", asdict(cfg))
    train_fold(a.fold, cfg, WM_RUNS / a.run, a.dataset)


if __name__ == "__main__":
    main()
