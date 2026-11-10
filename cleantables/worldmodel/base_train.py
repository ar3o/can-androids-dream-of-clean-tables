"""Training for the base world model: batch augmentation, losses, and per-fold training of one model or an ensemble.

  $VLA_PY -m cleantables.worldmodel.base_train --data box --steps 1500 --out runs/uwm/box_final

Augmentations that change the 3D scene without a matching video (rotation, scale, counterfactual re-placement) force
3D-only mode for that sample, since the V-JEPA tokens would contradict the scene. Mirroring uses vj_tok_flip and masks
the future-latent target, for which no flipped version exists."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cleantables.worldmodel.base_model import UWM, NP, PS, MIX
from cleantables.worldmodel.store import Store, ROOT

DEF = dict(steps=1500, bs=32, lr=3e-4, wd=0.05, rot=True, mirror=True, cf=True, scale=True, objdrop=True, moddrop=True,
           p_both=0.5, p_3d=0.25, mode_fixed=-1, w_prog=1.0, w_jepa=1.0, w_seen=0.2, w_flow=2.0, d=256, L=6, p_rot=0.5, p_cf=0.25, p_scale=0.3)
MOVE = {"obj_geo", "path_obj", "path_hand", "end_pose", "hand_geo", "end_off"}
BIGKEYS = ("obj_txt", "obj_crop", "vj_tok", "vj_tok_flip")


def to_dev(S, dev):
    return {k: v.to(dev) for k, v in S.T.items()}


def get_batch(G, idx, dev):
    b = {k: v[idx] for k, v in G.items()}
    for k in b:
        if b[k].dtype == torch.float16:
            b[k] = b[k].float()
        elif b[k].dtype == torch.uint8:
            b[k] = b[k].bool()
        elif b[k].dtype == torch.int8 or b[k].dtype == torch.int16 or b[k].dtype == torch.int32:
            b[k] = b[k].long()
    b["obj_kind"] = b["obj_kind"].clamp(0, 2)
    return b


def rotz(v, c, s):          # v (...,3 or more), c/s broadcastable to v[...,0]
    x, y = v[..., 0], v[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y, *[v[..., i] for i in range(2, v.shape[-1])]], -1)


def augment(b, cfg, g):
    B, dev = b["obj_geo"].shape[0], b["obj_geo"].device
    R = lambda: torch.rand(B, device=dev, generator=g)
    mir = (R() < 0.5) if cfg["mirror"] else torch.zeros(B, dtype=torch.bool, device=dev)
    rot = (R() < cfg["p_rot"]) if cfg["rot"] else torch.zeros(B, dtype=torch.bool, device=dev)
    scl = (R() < cfg["p_scale"]) if cfg["scale"] else torch.zeros(B, dtype=torch.bool, device=dev)
    cfm = (R() < cfg["p_cf"]) if cfg["cf"] else torch.zeros(B, dtype=torch.bool, device=dev)
    if "vid_keep" in b:                                                   # samples kept for video get no augmentation that breaks the video
        kv = b["vid_keep"].bool()
        rot, scl, cfm = rot & ~kv, scl & ~kv, cfm & ~kv
    sgn = torch.where(mir, -1.0, 1.0)
    # --- mirror (x -> -x, yaw -> -yaw)
    g_, h_ = b["obj_geo"].clone(), b["hand_geo"].clone()
    g_[..., 0] *= sgn[:, None]; g_[..., 6] *= sgn[:, None]
    h_[..., 1] *= sgn[:, None]
    c1, c2 = h_[..., 7:10].clone(), h_[..., 10:13].clone()
    h_[..., 7:10] = torch.where(mir[:, None, None], torch.stack([c1[..., 0], -c1[..., 1], -c1[..., 2]], -1), c1)
    h_[..., 10:13] = torch.where(mir[:, None, None], torch.stack([-c2[..., 0], c2[..., 1], c2[..., 2]], -1), c2)
    b["obj_geo"], b["hand_geo"] = g_, h_
    vk = tuple(k for k in ("path_obj", "path_hand") + tuple(cfg.get("aug_vec_keys", ())) if k in b)   # (B, n, 3) points / vectors about the origin
    for k in vk:
        b[k] = b[k].clone(); b[k][..., 0] *= sgn[:, None]
    if "end_pose" in b:                                                   # absent in label-free batches
        ep = b["end_pose"].clone(); ep[:, 0] *= sgn; ep[:, 3] *= sgn; b["end_pose"] = ep
        eo = b["end_off"].clone(); eo[:, 0] *= sgn; b["end_off"] = eo
    flip = mir[:, None, None]
    b["vj_tok"] = torch.where(flip, b["vj_tok_flip"], b["vj_tok"])
    # --- rotation about z
    th = (R() * 2 * math.pi) * rot.float()
    c, s = th.cos(), th.sin()
    g_ = b["obj_geo"].clone()
    xy = rotz(g_[..., :3], c[:, None], s[:, None]); g_[..., :3] = xy
    c2t, s2t = (2 * th).cos()[:, None], (2 * th).sin()[:, None]
    sn, cs = g_[..., 6].clone(), g_[..., 7].clone()
    g_[..., 6] = sn * c2t + cs * s2t; g_[..., 7] = cs * c2t - sn * s2t
    b["obj_geo"] = g_
    h_ = b["hand_geo"].clone()
    h_[..., 1:4] = rotz(h_[..., 1:4], c[:, None], s[:, None])
    h_[..., 7:10] = rotz(h_[..., 7:10], c[:, None], s[:, None]); h_[..., 10:13] = rotz(h_[..., 10:13], c[:, None], s[:, None])
    ex, ey = h_[..., 4].clone(), h_[..., 5].clone()
    h_[..., 4] = c.abs()[:, None] * ex + s.abs()[:, None] * ey; h_[..., 5] = s.abs()[:, None] * ex + c.abs()[:, None] * ey
    b["hand_geo"] = h_
    for k in vk:
        b[k] = rotz(b[k], c[:, None], s[:, None])
    if "end_pose" in b:
        ep = b["end_pose"].clone(); ep[:, :3] = rotz(ep[:, :3], c, s); ep[:, 3] = ep[:, 3] + th; b["end_pose"] = ep
    # --- size scale (whole scene about the origin)
    sc = torch.where(scl, torch.exp((R() * 2 - 1) * math.log(1.25)), torch.ones(B, device=dev))
    b["obj_geo"][..., :6] *= sc[:, None, None]
    b["hand_geo"][..., 1:7] *= sc[:, None, None]
    for k in vk:
        b[k] = b[k] * sc[:, None, None]
    if "end_pose" in b:
        b["end_pose"][:, :3] *= sc[:, None]; b["end_off"] = b["end_off"] * sc[:, None]
    # --- counterfactual re-placement of non-involved movable objects (label-preserving: kept at least as far from the mover as the target)
    N = b["obj_geo"].shape[1]
    ar = torch.arange(N, device=dev)[None]
    none_ = torch.full((B,), -1, dtype=torch.long, device=dev)
    mv, tg = b.get("mover", none_), b.get("target", none_)                # label-free batches: nothing is "involved"
    movable = b["obj_mask"].bool() & (b["obj_kind"] == 0)
    inv = (ar == mv[:, None]) | (ar == tg[:, None])
    cand = movable & ~inv & cfm[:, None] & (R()[:, None] < 2)           # all non-involved objects of the chosen samples
    pick = cand & (torch.rand(B, N, device=dev, generator=g) < 0.5)
    gm = b["obj_geo"]
    big = torch.where(movable[..., None], gm[..., :2], torch.full_like(gm[..., :2], float("nan")))
    lo = torch.nan_to_num(big.nan_to_num(1e9).min(1).values, 0); hi = big.nan_to_num(-1e9).max(1).values
    mpos = gm[torch.arange(B), mv.clamp(min=0), :2]
    tpos = gm[torch.arange(B), tg.clamp(min=0, max=N - 1), :2]
    dmt = torch.where((tg >= 0) & (tg < 24), (tpos - mpos).norm(dim=-1), torch.zeros(B, device=dev))
    newxy = lo[:, None, None] + torch.rand(B, N, 8, 2, device=dev, generator=g) * (hi - lo)[:, None, None]
    ok = ((newxy - mpos[:, None, None]).norm(dim=-1) >= dmt[:, None, None] + 0.03)
    # keep clear of other objects (centre distance > 0.6 * (L+W)/2 of the new object)
    dd = (newxy[:, :, :, None] - gm[:, None, None, :, :2]).norm(dim=-1) + torch.where(movable | (b["obj_kind"] == 2), 0.0, 9.0)[:, None, None]
    rad = 0.5 * (gm[..., 3] + gm[..., 4])[:, :, None, None] * 0.6
    ok = ok & ((dd > rad).all(-1))
    first = ok.float().argmax(-1)
    anyok = ok.any(-1)
    nxy = torch.gather(newxy, 2, first[..., None, None].expand(B, N, 1, 2)).squeeze(2)
    do = pick & anyok
    gm = gm.clone(); gm[..., :2] = torch.where(do[..., None], nxy, gm[..., :2]); gm[..., 2] = torch.where(do, gm[..., 5], gm[..., 2])
    b["obj_geo"] = gm
    changed = cfm & do.any(1)
    # --- object token dropout (non-involved)
    if cfg["objdrop"]:
        drop = movable & ~inv & (torch.rand(B, N, device=dev, generator=g) < 0.15)
        b["obj_mask"] = b["obj_mask"].bool() & ~drop
    b["aug_video_bad"] = rot | scl | changed
    b["aug_jepa_bad"] = rot | scl | changed | mir
    return b


def make_mode(b, cfg, g):
    B, dev = b["obj_geo"].shape[0], b["obj_geo"].device
    if cfg["mode_fixed"] >= 0:
        mode = torch.full((B,), cfg["mode_fixed"], device=dev, dtype=torch.long)
    elif cfg["moddrop"]:
        r = torch.rand(B, device=dev, generator=g)
        mode = torch.where(r < cfg["p_both"], 0, torch.where(r < cfg["p_both"] + cfg["p_3d"], 1, 2))
    else:
        mode = torch.zeros(B, dtype=torch.long, device=dev)
    mode = torch.where(b.get("aug_video_bad", torch.zeros(B, dtype=torch.bool, device=dev)) & (mode != 2), torch.ones_like(mode), mode)
    if cfg["mode_fixed"] != 2:
        mode = torch.where(b["aug_video_bad"] & (mode == 2), torch.ones_like(mode), mode)
    mode = torch.where(b["vj_ok"].bool(), mode, torch.ones_like(mode))
    return mode


def path_targets(b):
    B = b["mover"].shape[0]
    mv = b["mover"].clamp(min=0)
    org = b["obj_geo"][torch.arange(B), mv, :3]
    po = (b["path_obj"] - org[:, None]) / PS
    ph = (b["path_hand"] - org[:, None]) / PS
    wo = torch.isfinite(po).all(-1).float() * (b["mover"] >= 0).float()[:, None]
    wh = torch.isfinite(ph).all(-1).float() * b["path_hand_ok"].float() * (b["hand_geo"][:, 0, 0] > 0.5).float()[:, None] * (b["mover"] >= 0).float()[:, None]
    po, ph = torch.nan_to_num(po), torch.nan_to_num(ph)
    x1 = torch.cat([po, ph], -1)
    w = torch.cat([wo[..., None].expand(-1, -1, 3), wh[..., None].expand(-1, -1, 3)], -1)
    return x1, w, org


def collision_loss(m, E, c, b, x1, w, org, cfg, g, steps=4, d0=0.05):
    """Collision penalty on sampled paths.

    Samples a path with a few differentiable flow steps (teacher-forced end for anchored paths) and penalises every point
    of the carry (path fraction 0.15-0.8; the final descent into the destination is exempt) where the mover's bottom is
    below the top of another object whose footprint (rotated box, soft edges) it is over. Other objects are the valid
    movable and fixture tokens except the mover. Units: metres / d0."""
    B, dev = x1.shape[0], x1.device
    x = torch.randn(B, NP, 6, device=dev, generator=g)
    for i in range(steps):
        t = torch.full((B,), i / steps, device=dev)
        x = x + m.velocity(E, c, x, t) / steps
    p = x[..., :3]
    if cfg.get("path_resid"):
        u = torch.linspace(0, 1, NP, device=dev)[None, :, None]
        p = p + u * x1[:, -1:, :3]                                      # x1 is the residual here; add back the line to the true end
    P = org[:, None] + p * PS                                           # (B,NP,3) mover top-centre, table frame
    mv = b["mover"].clamp(min=0)
    Hm = b["obj_geo"][torch.arange(B, device=dev), mv, 5]
    zb = P[..., 2] - Hm[:, None]                                         # mover bottom
    geo = b["obj_geo"]                                                   # (B,N,8) x y top L W top sin2y cos2y
    yaw = 0.5 * torch.atan2(geo[..., 6], geo[..., 7])
    dx = P[:, :, None, 0] - geo[:, None, :, 0]; dy = P[:, :, None, 1] - geo[:, None, :, 1]
    cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    lx = (cy * dx + sy * dy).abs(); ly = (-sy * dx + cy * dy).abs()
    inside = torch.sigmoid((geo[:, None, :, 3] / 2 - lx) / 0.01) * torch.sigmoid((geo[:, None, :, 4] / 2 - ly) / 0.01)
    valid = b["obj_mask"].bool() & (b["obj_kind"] != 1)
    valid = valid & (torch.arange(geo.shape[1], device=dev)[None] != mv[:, None])
    pen = torch.relu(geo[:, None, :, 2] + 0.01 - zb[:, :, None]) / d0
    tt = torch.linspace(0, 1, NP, device=dev)
    seg = ((tt >= 0.15) & (tt <= 0.8)).float()[None, :, None]
    ok = (b["mover"] >= 0).float() * (w[:, :, 0].sum(-1) > 0).float()
    lv = (pen * inside * valid[:, None].float() * seg).sum((1, 2)) / seg.sum()
    return mean_w(lv, ok)


def mean_w(l, w):
    return (l * w).sum() / w.sum().clamp(min=1.0)


def compute_loss(m, b, cfg, g):
    B, dev = b["mover"].shape[0], b["mover"].device
    mode = make_mode(b, cfg, g)
    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
    none_ = torch.full((B,), -1, dtype=torch.long, device=dev)
    mv, tg = b.get("mover", none_), b.get("target", none_)                # label-free batches: nothing is "involved"
    L = {}
    ml, tl = m.pointers(E, mv)
    okm = (mv >= 0).float(); okt = ((tg >= 0) & (mv >= 0)).float()
    if cfg.get("mover_set") and "mover_set" in b:   # order-free: probability mass on any object still to be moved in this episode
        ms = b["mover_set"].bool() & b["obj_mask"].bool()
        oks = ms.any(-1).float()
        lp = F.log_softmax(ml.float(), -1).masked_fill(~ms, -1e4)
        L["mover"] = mean_w(-torch.logsumexp(lp, -1), oks)
    else:
        L["mover"] = mean_w(F.cross_entropy(ml, mv.clamp(min=0), reduction="none"), okm)
    L["target"] = mean_w(F.cross_entropy(tl, tg.clamp(min=0), reduction="none"), okt)
    c, rel, mdn = m.heads_from(E, mv, tg)
    L["relation"] = 0.5 * mean_w(F.cross_entropy(rel, b["relation"].clamp(min=0), reduction="none"), ((b["relation"] >= 0) & (mv >= 0)).float())
    eo = b["end_off"] / 0.1; oke = (torch.isfinite(eo).all(-1) & (tg >= 0) & (mv >= 0)).float(); eo = torch.nan_to_num(eo)
    sd = mdn["logs"].exp() * 0.0 + mdn["logs"].exp()
    nll = (0.5 * ((eo[:, None] - mdn["mu"]) / sd) ** 2 + mdn["logs"]).sum(-1) - F.log_softmax(mdn["logit"], -1)
    L["end_off"] = 0.5 * mean_w(-torch.logsumexp(-nll, -1), oke)
    # flow matching
    x1, w, org = path_targets(b)
    if cfg.get("path_resid"):      # end-anchored: end displacement mixture + flow on the residual around the straight line to the true end
        dtrue = x1[:, -1, :3]
        okd = (w[:, -1, 0] > 0).float()
        zd = m.end_disp(c); sdd = zd["logs"].exp()
        nlld = (0.5 * ((dtrue[:, None] - zd["mu"]) / sdd) ** 2 + zd["logs"]).sum(-1) - F.log_softmax(zd["logit"], -1)
        L["end_disp"] = mean_w(-torch.logsumexp(-nlld, -1), okd)
        u = torch.linspace(0, 1, NP, device=dev)[None, :, None]
        line = u * dtrue[:, None, :]
        x1 = torch.cat([x1[..., :3] - line, x1[..., 3:6] - line], -1)
    if cfg.get("col_loss", 0) > 0:      # learned non-penetration: sampled carries are penalised for passing through other objects' boxes
        L["collision"] = cfg["col_loss"] * collision_loss(m, E, c, b, x1, w, org, cfg, g)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=dev, generator=g)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v = m.velocity(E, c, xt, t)
    L["flow"] = cfg["w_flow"] * mean_w(((v - (x1 - x0)) ** 2), w)
    # progress
    if cfg["w_prog"] > 0:
        pv = torch.sigmoid(m.p_vid(E["cls"][:, 0]).squeeze(-1))
        okv_ = b["prog_video_ok"].float()
        if cfg.get("prog_loss", "huber") == "bce":     # BCE option: Huber (delta 0.1) underfits here (train error ~0.2, outputs squashed to 0.4-0.6)
            pl = m.p_vid(E["cls"][:, 0]).squeeze(-1).float()
            L["prog_video"] = cfg.get("w_prog_abs", 3.0) * mean_w(F.binary_cross_entropy_with_logits(pl, torch.nan_to_num(b["prog_video"]).clamp(0, 1), reduction="none"), okv_)
        else:
            L["prog_video"] = cfg["w_prog"] * mean_w(F.huber_loss(pv, torch.nan_to_num(b["prog_video"]), reduction="none", delta=0.1), okv_)
        # forward ranking inside a video (same video, later snapshot must score higher)
        pt_ = torch.nan_to_num(b["prog_video"]); same = (b["video_id"][:, None] == b["video_id"][None]) & (okv_[:, None] > 0) & (okv_[None] > 0)
        dpair = (pt_[:, None] - pt_[None]) > 0.05
        pm_ = (same & dpair).float()
        L["prog_rank"] = 0.5 * cfg["w_prog"] * mean_w(F.relu(0.05 - (pv[:, None] - pv[None])), pm_)
        # reversed-order negative: the time-reversed V-JEPA clip should score lower than the forward clip
        b3 = dict(b); b3["vj_tok"] = b["vj_tok"].flip(1)
        E3 = m.encode(b3, mode)
        pv3 = torch.sigmoid(m.p_vid(E3["cls"][:, 0]).squeeze(-1))
        wrev = okv_ * ((mode == 0) & b["vj_ok"].bool()).float() * (~b["aug_video_bad"]).float()
        L["prog_rev"] = 0.5 * cfg["w_prog"] * mean_w(F.relu(0.1 - (pv - pv3)), wrev)
        ttm = b["time_to_move"] if "time_to_move" in b else torch.full((B,), float("nan"), device=dev)
        okt_ = torch.isfinite(ttm).float()
        L["ttm"] = 0.3 * cfg["w_prog"] * mean_w(F.huber_loss(m.p_ttm(E["cls"][:, 0]).squeeze(-1), torch.log1p(torch.nan_to_num(ttm).clamp(min=0)), reduction="none"), okt_)
        ps = torch.nan_to_num(b["prog_steps"]); okp = (torch.isfinite(b["prog_steps"]).all(-1) & (mv >= 0)).float()
        p6 = torch.cat([b["path_obj"] - org[:, None], b["path_hand"] - org[:, None]], -1).nan_to_num() if cfg.get("path_resid") else x1 * PS
        pf = m.path_prog(E, c, p6); pr = m.path_prog(E, c, p6.flip(1))          # reversed path presented: labels stay attached to the content
        hub = F.huber_loss(pf, ps, reduction="none", delta=0.1).mean(-1) + F.huber_loss(pr, ps.flip(1), reduction="none", delta=0.1).mean(-1)
        i = torch.randint(0, NP - 4, (B, 48), device=dev, generator=g); j = (i + torch.randint(3, 12, (B, 48), device=dev, generator=g)).clamp(max=NP - 1)
        rank = lambda p, lab: F.relu(0.05 - (torch.gather(p, 1, j) - torch.gather(p, 1, i)) * torch.sign(torch.gather(lab, 1, j) - torch.gather(lab, 1, i))).mean(-1)
        L["prog_steps"] = cfg["w_prog"] * mean_w(hub + rank(pf, ps) + rank(pr, ps.flip(1)), okp)
    # JEPA latent
    if cfg["w_jepa"] > 0:
        pj = m.jepa_pred(E, b, mode)
        tgt = torch.nan_to_num(b["vj_future_glob"]).detach()
        okj = (b["vj_future_ok"].bool() & ~b["aug_jepa_bad"] & (mode != 1) | (b["vj_future_ok"].bool() & ~b["aug_jepa_bad"] & (mode == 1))).float()
        L["jepa"] = cfg["w_jepa"] * mean_w(1 - F.cosine_similarity(pj.float(), tgt, dim=-1), okj)
    # p(seen): positives = clean; negatives = V-JEPA tokens of another video with this scene (modality inconsistency)
    if cfg["w_seen"] > 0:
        half = B // 2
        b2 = dict(b); vj = b["vj_tok"].clone(); vj[:half] = vj.roll(1, 0)[:half]
        b2["vj_tok"] = vj
        diff = torch.ones(B, device=dev); diff[:half] = (b["video_id"][:half] != b["video_id"].roll(1, 0)[:half]).float()
        m2 = torch.zeros(B, dtype=torch.long, device=dev)
        E2 = m.encode(b2, m2)
        lab = torch.ones(B, device=dev); lab[:half] = 0
        okv = ((diff > 0) & b["vj_ok"].bool() & ~b["aug_video_bad"]).float()
        L["seen"] = cfg["w_seen"] * mean_w(F.binary_cross_entropy_with_logits(m.seen(E2["cls"][:, 3]).squeeze(-1), lab, reduction="none"), okv)
    return L


def train_model(S, G, tr_idx, cfg, seed=0, dev="cuda", log=None, bootstrap=False):
    cfg = {**DEF, **cfg}
    torch.manual_seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = UWM(d=cfg["d"], L=cfg["L"], path_resid=bool(cfg.get("path_resid"))).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr = np.asarray(tr_idx)
    if bootstrap:
        tr = rng.choice(tr, len(tr))
    T = cfg["steps"]
    t0 = time.time()
    for it in range(T):
        lr = cfg["lr"] * min(1, (it + 1) / 100) * 0.5 * (1 + math.cos(math.pi * it / T))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.as_tensor(tr[rng.integers(0, len(tr), cfg["bs"])], device=dev)
        b = get_batch(G, idx, dev); b = augment(b, cfg, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Ls = compute_loss(m, b, cfg, gen)
        loss = sum(Ls.values())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 3.0); opt.step()
        if log is not None and (it % 250 == 0 or it == T - 1):
            log(f"  it {it} loss {loss.item():.3f} " + " ".join(f"{k}={v.item():.3f}" for k, v in Ls.items()))
    m.train_seconds = time.time() - t0
    return m.eval()


def prep_store(names):
    S = Store(names)
    # derived: integer video id for the p(seen) negatives; optional float fields
    vids = {v: i for i, v in enumerate(sorted(set(S.d["video"])))}
    S.T["video_id"] = torch.tensor([vids[v] for v in S.d["video"]])
    # box.npz ships prog_video_ok = 0; for all box samples the label is instead the normalised time through the video
    box = torch.from_numpy(S.d["dataset"] == "box")
    if box.any():
        S.T["prog_video_ok"] = S.T["prog_video_ok"].clone(); S.T["prog_video_ok"][box] = torch.isfinite(S.T["prog_video"][box]).to(S.T["prog_video_ok"].dtype)
        S.d["prog_video_ok"] = S.T["prog_video_ok"].numpy().astype(bool)
    for k, dflt in (("vj_future_ok", True), ("time_to_move", float("nan"))):
        if k not in S.T:
            S.T[k] = torch.full((S.n,), dflt)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="box"); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="runs/uwm/test"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    S = prep_store(a.data.split(",")); G = to_dev(S, dev)
    tr = S.where(split="train")
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    m = train_model(S, G, tr, dict(steps=a.steps), a.seed, dev, log=print)
    print("params", sum(p.numel() for p in m.parameters()), "secs", m.train_seconds, "mem GB", torch.cuda.max_memory_allocated() / 2**30)
    torch.save(m.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
