"""Training of the world model (cleantables.worldmodel.model) on saved frozen features.

Optimiser, schedule, augmentations and losses follow cleantables.worldmodel.base_train (tasks3_v4 config: BCE count progress,
order-free next object, end-anchored paths), plus the end-yaw and move-end-latent heads; the base p(seen) loss is dropped
(no pooled tokens). The 'watch*' configurations train UWMW with future-prediction losses only (watch_losses).

Ablations (leave-one-out from full):
  B0            the base recipe (base_train.train_model, 32 pooled tokens, hashed instruction) on the same split
  full          everything
  no_text       hashed instruction vectors (same strings)       no_para  no paraphrases     no_narr  no narration
  no_comp       no composite scenes                               no_vj    no V-JEPA grid      no_dn    no DINOv2 grid
  vid_occ       both grids -> occupancy raster (location only)    vid_pix  both grids -> raw colour
  no_objg       no objg                                           no_video no video input at all (3D + appearance only)
  app_ctl       appearance tokens -> position code               no_app   no appearance tokens
  no_yaw        no end-yaw head
  $VLA_PY -m cleantables.worldmodel.train --cfg full --seeds 0,1,2 [--fold -1] [--steps 3000] [--smoke]
Out: OUT/wm_v2/runs/<cfg>/fold<f>_seed<s>.pt (+ .json); resumable (checkpoint every 500 steps).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.common import OUT, ROOT, log

RUNS2 = Path(os.environ.get("WM_RUNS2", str(OUT / "wm_v2" / "runs")))     # e.g. OUT/wm_v2_final_candidate/runs
V1_CFG = {"prog_loss": "bce", "w_prog_abs": 3.0, "path_resid": True, "mover_set": True}          # the base model's tasks3_v4 config
NOISE_FLOOR = 0.0075    # m per axis: the tracker's rest noise, median |d_axis| / 0.6745 over 16,882 untouched training tokens outside
                        # fold 0, pooled over x / y / z = 0.60 / 0.88 / 0.79 cm (their 3D |d| median is 1.21 cm)
MOVED_EVAL = 0.162      # m: the tracker's rest p90 (3D, 4 s); moved / stayed split for validation curves only, never trained on
V2_DEF = dict(steps=3000, p_comp=0.25, p_narr=0.3, paraphrase=True, text="minilm", vid=("vj", "dn"), vid_ctl="", objg=True, app=True,
              app_ctl=False, yaw=True, fut=True, w_yaw=0.5, w_fut=1.0, p_handdrop=0.5)
CFGS = {
    "full": {}, "no_text": dict(text="hashed"), "no_para": dict(paraphrase=False), "no_narr": dict(p_narr=0.0), "no_comp": dict(p_comp=0.0),
    "no_vj": dict(vid=("dn",)), "no_dn": dict(vid=("vj",)), "vid_occ": dict(vid_ctl="occ"), "vid_pix": dict(vid_ctl="pix"),
    "no_objg": dict(objg=False), "no_video": dict(vid=(), objg=False, fut=False, mode_fixed=1), "app_ctl": dict(app_ctl=True), "no_app": dict(app=False),
    "no_yaw": dict(yaw=False),
    # composites: same-sampler layout for half of them, donors balanced over the other instructions, 50 % composites
    "comp_relay": dict(p_relay=0.5, comp_balance=True, p_comp=0.33),       # plus a quarter of each batch as instruction-balanced composites
    # + stop option and the instructed task's progress in all composites
    "comp_stop": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, stop=True, prog_relaid=True, relay_keep_inside=True),
    # + stop examples: remove-one (a placed object of the instruction back onto the table -> it is the next mover, not stop) and
    #   place-the-last-one (the last mover at its human end pose -> stop)
    "comp_stop2": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, stop=True, prog_relaid=True, relay_keep_inside=True, p_remove=0.2, p_addlast=0.5),
    "holdout_names": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, holdout_names=("apple", "wrench")),   # generalisation test
    # final (on the 6D tracks, WM_DATA_ROOT=OUT/wm_v2/data TASKS3_DATA=tasks3_fp6d): the composite recipe, with / without 6D paths
    "final": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, stop=True, prog_relaid=True, relay_keep_inside=True),
    # video_v2: final + per-sample real / render video, crops / zooms with a random clip start, reversed-clip negatives for progress /
    # stop (mirror uses the flipped encodings; video and hand dropout as before); fewer 3D rotations (a rotated sample can have no
    # video), so that more samples train with video
    "video_v2": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, stop=True, prog_relaid=True, relay_keep_inside=True,
                     video_v2=True, p_render=0.8, p_vcrop=0.5, p_rev=0.25, w_rev=1.0, p_rot=0.25),
    # video_v2b: + paired renders (robot hidden, v0 live look + domain-randomised v1-v3; held sampling as in the live loop) instead of
    # replay renders, + rendered composites (mixed scenes with render video) as 15 % of each batch
    "video_v2b": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, stop=True, prog_relaid=True, relay_keep_inside=True,
                      video_v2=True, render_src="paired", p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.25, w_rev=1.0, p_rot=0.25, p_rcomp=0.15),
    # clean: video_v2b + a single end output (no destination / relation labels), contact-interval moves and stop, the self-supervised
    # world losses (future positions, hidden objects) on every keyframe, the consequence head (trained in the physics stage); no
    # counterfactual re-placement (it relies on the destination label)
    "clean": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, stop=True, prog_relaid=True, relay_keep_inside=True,
                  video_v2=True, render_src="paired", p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.25, w_rev=1.0, p_rot=0.25, p_rcomp=0.0,
                  endref=True, world=True, cons=True, contact_stop=True, cf=False, w_end=1.0, w_world=1.0, p_hide=(0.15, 0.30),
                  p_narr=0.0, relay_keep_dest=False),           # narrations are keyed to thresholded events; rcomp rows carry destination labels
    # watch: no move labels; the targets are T1 (every object's end-of-task pose), T2 (every object's next 4 s, one joint flow sample),
    # T3 (V-JEPA latent at +4 s), T5 (future hand contact) and the hidden-object position. Data: watch_data. objg / dino_goal off (they
    # compare with the episode's end frame, which is the T1 target); watch_objg is that ablation. Composites without re-layout (donors
    # stay put); no counterfactual re-placement.
    "watch": dict(watch=True, H=8, objg=False, p_video=0.6, comp_balance=True, p_comp=0.33, p_relay=0.0, video_v2=True, render_src="paired", p_render=0.5,
                  p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.25, p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0, w_t5=0.5,
                  w_hide=0.3, p_hide=(0.15, 0.30)),
    # watch_pilot: watch + the tracker's rest noise as a floor on the T1 sd and a dither of the T2 flow targets (noise_floor, m per
    # axis), the continuous distance head (p_now from its 4 s output), and validation curves (T1 moved / stayed) every 500 steps
    "watch_pilot": dict(watch=True, H=8, objg=False, p_video=0.6, comp_balance=True, p_comp=0.33, p_relay=0.0, video_v2=True, render_src="paired",
                        p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.25, p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0,
                        w_t5=0.5, w_hide=0.3, p_hide=(0.15, 0.30), noise_floor=NOISE_FLOOR, dist_head=True, w_dist=0.5, val_log=True,
                        moved_eval=MOVED_EVAL),
    # watch_pair: watch_pilot + paired-instruction composites for a share p_pair of every batch's rows
    # (row n: instruction A, B stays; row n2: instruction B, B's own rigidly moved targets, A stays)
    "watch_pair": dict(watch=True, H=8, objg=False, p_video=0.6, comp_balance=True, p_comp=0.33, p_relay=0.0, video_v2=True, render_src="paired",
                       p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.25, p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0,
                       w_t5=0.5, w_hide=0.3, p_hide=(0.15, 0.30), noise_floor=NOISE_FLOOR, dist_head=True, w_dist=0.5, val_log=True,
                       moved_eval=MOVED_EVAL, p_pair=0.25),
    # watch_rigid: watch_pair + a share p_rigid of the eligible (non-video) rows re-laid as one rigid group relative to the table
    # (and augment's whole-scene rotation p_rot raised from 0.25 to 0.5)
    "watch_rigid": dict(watch=True, H=8, objg=False, p_video=0.6, comp_balance=True, p_comp=0.33, p_relay=0.0, video_v2=True, render_src="paired",
                        p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.5, p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0,
                        w_t5=0.5, w_hide=0.3, p_hide=(0.15, 0.30), noise_floor=NOISE_FLOOR, dist_head=True, w_dist=0.5, val_log=True,
                        moved_eval=MOVED_EVAL, p_pair=0.25, p_rigid=0.9),
    # watch_notab: watch_rigid without the table as a T1 reference; run with --steps 9000 (the cosine schedule stretches with it);
    # per-family validation curves every 1000 steps
    "watch_notab": dict(watch=True, H=8, objg=False, p_video=0.6, comp_balance=True, p_comp=0.33, p_relay=0.0, video_v2=True, render_src="paired",
                        p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.5, p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0,
                        w_t5=0.5, w_hide=0.3, p_hide=(0.15, 0.30), noise_floor=NOISE_FLOOR, dist_head=True, w_dist=0.5, val_log=True,
                        moved_eval=MOVED_EVAL, p_pair=0.25, p_rigid=0.9, t1_no_table=True, curve_every=1000, curve_per_family=True),
    # watch_rem: watch_notab + the remaining-motion fraction head (for stop); --steps 9000
    "watch_rem": dict(watch=True, H=8, objg=False, p_video=0.6, comp_balance=True, p_comp=0.33, p_relay=0.0, video_v2=True, render_src="paired",
                      p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.5, p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0,
                      w_t5=0.5, w_hide=0.3, p_hide=(0.15, 0.30), noise_floor=NOISE_FLOOR, dist_head=True, w_dist=0.5, val_log=True,
                      moved_eval=MOVED_EVAL, p_pair=0.25, p_rigid=0.9, t1_no_table=True, curve_every=1000, curve_per_family=True,
                      rem_head=True, w_rem=0.5),
    # watch_novideo: watch with video off (no grids, so no T3; 3D-only), the comparison for what video adds
    "watch_novideo": dict(watch=True, H=8, objg=False, p_video=0.0, vid=(), fut=False, mode_fixed=1, comp_balance=True, p_comp=0.33,
                          p_relay=0.0, video_v2=True, render_src="paired", p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.25,
                          p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0, w_t5=0.5, w_hide=0.3, p_hide=(0.15, 0.30)),
    "watch_objg": dict(watch=True, H=8, objg=True, p_video=0.6, comp_balance=True, p_comp=0.33, p_relay=0.0, video_v2=True, render_src="paired",
                       p_render=0.5, p_hold=0.7, p_vcrop=0.5, p_rev=0.0, p_rot=0.25, p_rcomp=0.0, p_narr=0.0, cf=False, w_t1=1.0, w_t2=1.0,
                       w_t5=0.5, w_hide=0.3, p_hide=(0.15, 0.30)),
    "final_rot6": dict(p_relay=0.5, comp_balance=True, p_comp=0.33, stop=True, prog_relaid=True, relay_keep_inside=True, rot6=True),
}


def full_cfg(name):
    from cleantables.worldmodel.base_train import DEF
    c = {**DEF, **V1_CFG, **V2_DEF, **CFGS[name]}
    c["vid"] = tuple(c["vid"])
    return c


def build(cfg):
    from cleantables.worldmodel.model import UWM2, UWMW
    if cfg.get("watch"):
        return UWMW(H=int(cfg.get("H", 8)), noise_floor=float(cfg.get("noise_floor", 0.0)), dist_head=bool(cfg.get("dist_head", False)),
                    t1_no_table=bool(cfg.get("t1_no_table", False)), rem_head=bool(cfg.get("rem_head", False)), d=cfg["d"], L=cfg["L"], vid=cfg["vid"], vid_ctl=cfg["vid_ctl"], objg=cfg["objg"], app=cfg["app"],
                    app_ctl=cfg["app_ctl"], yaw=cfg["yaw"], fut=cfg["fut"])
    return UWM2(d=cfg["d"], L=cfg["L"], vid=cfg["vid"], vid_ctl=cfg["vid_ctl"], objg=cfg["objg"], app=cfg["app"], app_ctl=cfg["app_ctl"],
                yaw=cfg["yaw"], fut=cfg["fut"], rot6=bool(cfg.get("rot6", False)),
                stop=bool(cfg.get("stop", False)), endref=bool(cfg.get("endref", False)), world=bool(cfg.get("world", False)),
                cons=bool(cfg.get("cons", False)))


def compute_loss2(m, b, cfg, g):
    from cleantables.worldmodel.base_model import NP, PS
    from cleantables.worldmodel.base_train import make_mode, mean_w, path_targets
    B, dev = b["mover"].shape[0], b["mover"].device
    mode = make_mode(b, cfg, g)
    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
    mv, tg = b["mover"], b["target"]
    L = {}
    ml, tl = m.pointers(E, mv)
    okt = ((tg >= 0) & (mv >= 0)).float()
    ms = b["mover_set"].bool() & b["obj_mask"].bool()
    if cfg.get("stop"):
        # stop: the instructed task is complete (its count progress is 1 and no further mover is labelled). In a composite the added
        # objects are never movers, so a complete base scene with another task's objects left on the table is a stop example.
        if cfg.get("contact_stop"):
            # contact_stop: stop = no contact interval left in the segment (the mover set is empty), for every task example; no
            # progress threshold
            done = ~ms.any(-1) & b["prog_video_ok"].bool()
        else:
            done = ~ms.any(-1) & b["prog_video_ok"].bool() & (torch.nan_to_num(b["prog_video"]) >= 0.999)
        lj = torch.cat([ml.float(), m.stop_logit(E).float()[:, None]], -1)
        tgt = torch.cat([ms, done[:, None]], -1)
        lp = F.log_softmax(lj, -1).masked_fill(~tgt, -1e4)
        L["mover"] = mean_w(-torch.logsumexp(lp, -1), (ms.any(-1) | done).float())
    else:
        lp = F.log_softmax(ml.float(), -1).masked_fill(~ms, -1e4)
        L["mover"] = mean_w(-torch.logsumexp(lp, -1), ms.any(-1).float())
    endref = getattr(m, "endref", False)
    if not endref:                                                       # destination / relation / end-offset heads (labelled)
        L["target"] = mean_w(F.cross_entropy(tl.float(), tg.clamp(min=0), reduction="none"), okt)
        c, rel, mdn = m.heads_from(E, mv, tg)
        L["relation"] = 0.5 * mean_w(F.cross_entropy(rel.float(), b["relation"].clamp(min=0), reduction="none"), ((b["relation"] >= 0) & (mv >= 0)).float())
        eo = b["end_off"] / 0.1; oke = (torch.isfinite(eo).all(-1) & (tg >= 0) & (mv >= 0)).float(); eo = torch.nan_to_num(eo)
        sd = mdn["logs"].float().exp()
        nll = (0.5 * ((eo[:, None] - mdn["mu"].float()) / sd) ** 2 + mdn["logs"].float()).sum(-1) - F.log_softmax(mdn["logit"].float(), -1)
        L["end_off"] = 0.5 * mean_w(-torch.logsumexp(-nll, -1), oke)
    x1, w, org = path_targets(b)
    keep = (~b["relaid"]).float() if "relaid" in b else torch.ones(B, device=dev)   # re-laid composites: the human path and progress no longer fit
    if "altered" in b:                                                   # stop examples (remove-one / place-the-last-one): the same
        keep = keep * (~b["altered"]).float()
    w = w * keep[:, None, None]
    dtrue = x1[:, -1, :3]
    okd = (w[:, -1, 0] > 0).float()
    if endref:
        # endref: the only end target is where the object ended (the mover's path end, world metres); no destination / relation
        # label. The path / yaw context is teacher-forced on that end.
        end_true = org.float() + PS * dtrue.float()
        nll_e, _ = m.end_nll(E, mv, end_true)
        L["end"] = cfg.get("w_end", 1.0) * mean_w(nll_e, okd)
        c = m.ctx_end(E, mv, end_true)
    else:
        zd = m.end_disp(c); sdd = zd["logs"].float().exp()
        nlld = (0.5 * ((dtrue[:, None] - zd["mu"].float()) / sdd) ** 2 + zd["logs"].float()).sum(-1) - F.log_softmax(zd["logit"].float(), -1)
        L["end_disp"] = mean_w(-torch.logsumexp(-nlld, -1), okd)
    u = torch.linspace(0, 1, NP, device=dev)[None, :, None]
    line = u * dtrue[:, None, :]
    x1 = torch.cat([x1[..., :3] - line, x1[..., 3:6] - line], -1)
    if cfg.get("rot6"):                                                  # 6D anchor paths: rotation residual around 'no rotation'
        rr = b["path_rot"] - b["path_rot"].new_tensor([1, 0, 0, 0, 1, 0])
        wr = (b["rot_ok"].bool() & (mv >= 0)).float()[:, None, None].expand(-1, x1.shape[1], 6) * keep[:, None, None]
        x1 = torch.cat([x1, rr], -1); w = torch.cat([w, wr], -1)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=dev, generator=g)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v = m.velocity(E, c, xt, t)
    L["flow"] = cfg["w_flow"] * mean_w(((v.float() - (x1 - x0)) ** 2), w)
    if cfg["yaw"]:
        oky = (b["end_ok"].bool() & (mv >= 0) & torch.isfinite(b["end_pose"][:, 3])).float()
        L["yaw"] = cfg["w_yaw"] * mean_w(m.yaw_nll(c.float(), torch.nan_to_num(b["end_pose"][:, 3])), oky)
    # progress (BCE on the count label, in-episode ranking, time-to-move, per-step path progress)
    # count progress of the instructed task. Re-laid composites keep every placed object where it is (data._relayout), so the label
    # stays true there; prog_relaid=True trains on them (only the hand / path based progress terms stay masked).
    okv_ = b["prog_video_ok"].float() * (torch.ones_like(keep) if cfg.get("prog_relaid") else keep)
    pl = m.p_vid(E["cls"][:, 0]).squeeze(-1).float(); pv = torch.sigmoid(pl)
    L["prog_video"] = cfg.get("w_prog_abs", 3.0) * mean_w(F.binary_cross_entropy_with_logits(pl, torch.nan_to_num(b["prog_video"]).clamp(0, 1), reduction="none"), okv_)
    pt_ = torch.nan_to_num(b["prog_video"]); same = (b["video_id"][:, None] == b["video_id"][None]) & (okv_[:, None] > 0) & (okv_[None] > 0)
    pm_ = (same & ((pt_[:, None] - pt_[None]) > 0.05)).float()
    L["prog_rank"] = 0.5 * cfg["w_prog"] * mean_w(F.relu(0.05 - (pv[:, None] - pv[None])), pm_)
    ttm = b["time_to_move"]; okt_ = torch.isfinite(ttm).float() * keep
    L["ttm"] = 0.3 * cfg["w_prog"] * mean_w(F.huber_loss(m.p_ttm(E["cls"][:, 0]).squeeze(-1).float(), torch.log1p(torch.nan_to_num(ttm).clamp(min=0)), reduction="none"), okt_)
    ps = torch.nan_to_num(b["prog_steps"]); okp = (torch.isfinite(b["prog_steps"]).all(-1) & (mv >= 0)).float() * keep
    p6 = torch.cat([b["path_obj"] - org[:, None], b["path_hand"] - org[:, None]], -1).nan_to_num()
    pf = m.path_prog(E, c, p6).float(); pr = m.path_prog(E, c, p6.flip(1)).float()
    hub = F.huber_loss(pf, ps, reduction="none", delta=0.1).mean(-1) + F.huber_loss(pr, ps.flip(1), reduction="none", delta=0.1).mean(-1)
    i = torch.randint(0, NP - 4, (B, 48), device=dev, generator=g); j = (i + torch.randint(3, 12, (B, 48), device=dev, generator=g)).clamp(max=NP - 1)
    rank = lambda p, lab: F.relu(0.05 - (torch.gather(p, 1, j) - torch.gather(p, 1, i)) * torch.sign(torch.gather(lab, 1, j) - torch.gather(lab, 1, i))).mean(-1)
    L["prog_steps"] = cfg["w_prog"] * mean_w(hub + rank(pf, ps) + rank(pr, ps.flip(1)), okp)
    if cfg["fut"] and "glob_fut" in b:
        okf = (b["glob_fut_ok"] & ~b["aug_jepa_bad"] & E["von"]).float()
        L["fut"] = cfg["w_fut"] * mean_w(1 - F.cosine_similarity(m.fut_pred(E, b), b["glob_fut"].float(), dim=-1), okf)
    if cfg.get("video_v2") and "vj_rev" in b and len(b["rev_sel"]):
        L.update(reversed_clip_loss(m, b, cfg, pv, okv_, done if cfg.get("stop") else None))
    return L


def compute_loss_w(m, b, cfg, g):
    """Losses of the watch model (UWMW): T1 end-of-task pose (per-object reference-mixture NLL, p(present), end yaw), T2 joint flow
    over every object's next H steps (plus the mean last-step displacement), T3 future V-JEPA latent, T5 future hand contact and
    hidden-object position. All targets come from the tracked data (watch_data); none is a move label."""
    from cleantables.worldmodel.base_model import PS
    from cleantables.worldmodel.base_train import make_mode, mean_w
    B, N = b["obj_mask"].shape
    dev = b["obj_mask"].device
    mode = make_mode(b, cfg, g)
    if "vid_keep" in b:                                                  # video share: kept samples with valid video run in mode 0
        mode = torch.where(b["vid_keep"].bool() & b["vid_ok"].bool(), torch.zeros_like(mode), mode)
    obj = b["obj_mask"].bool() & (b["obj_kind"] == 0)
    lo, hi = cfg.get("p_hide", (0.15, 0.30))
    frac = lo + (hi - lo) * torch.rand(B, device=dev, generator=g)
    hide = obj & (torch.rand(B, N, device=dev, generator=g) < frac[:, None])
    b["obj_hide"] = hide
    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
    L = {}
    # T1
    ok1 = b["t1_ok"].bool() & obj
    okp = ok1 & b["t1_present"].bool()
    nll = m.t1_nll(E, b["t1_pos"].float())
    L["t1"] = cfg.get("w_t1", 1.0) * mean_w(nll, okp.float())
    L["t1_present"] = 0.3 * cfg.get("w_t1", 1.0) * mean_w(F.binary_cross_entropy_with_logits(m.present_logit(E), b["t1_present"].float(), reduction="none"), ok1.float())
    oky = okp & b["t1_yaw_ok"].bool() if "t1_yaw_ok" in b else okp
    L["t1_yaw"] = 0.3 * cfg.get("w_t1", 1.0) * mean_w(m.t1_yaw_nll(E, b["t1_yaw"].float()), oky.float())
    # T2: one joint flow sample over all objects (PS units), plus the mean last-step displacement
    H = m.H
    x1 = (b["t2_disp"].view(B, N, H, 3) / PS).float()
    w2 = (b["t2_ok"].bool() & b["obj_mask"].bool()[..., None]).float()
    x1r = x1
    if cfg.get("noise_floor", 0.0) > 0:                                   # noise floor: flow targets dithered by the tracker's rest noise
        x1 = x1 + (cfg["noise_floor"] / PS) * torch.randn(x1.shape, device=dev, generator=g)   # (a delta at 0 cannot be learnt by the flow)
    x0 = torch.randn_like(x1); t = torch.rand(B, device=dev, generator=g)
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    v = m.velocity_all(E, xt, t).float()
    L["t2"] = cfg.get("w_t2", 1.0) * mean_w(((v - (x1 - x0)) ** 2).sum(-1), w2)
    L["t2_mean"] = 0.5 * cfg.get("w_t2", 1.0) * mean_w(F.huber_loss(m.mean_disp(E), x1r[:, :, -1], reduction="none", delta=0.1).sum(-1), w2[..., -1])
    if getattr(m, "rem_head", False) and "rem_frac" in b:                 # remaining-motion fraction (soft BCE)
        L["rem"] = cfg.get("w_rem", 0.5) * mean_w(F.binary_cross_entropy_with_logits(m.rem_logit(E), b["rem_frac"].float(), reduction="none"),
                                                  b["rem_ok"].float())
    if getattr(m, "dist_head", False):                                   # continuous "how far will it be from here" (no cut-off)
        from cleantables.worldmodel.model import N50
        e = m.dist_pred(E); now = b["obj_geo"][..., :3].float()
        y1 = torch.log((b["t1_pos"].float() - now).norm(dim=-1) + N50)
        y2 = torch.log((x1r[:, :, -1] * PS).norm(dim=-1) + N50)
        L["dist"] = cfg.get("w_dist", 0.5) * (mean_w(F.huber_loss(e[..., 0], y1, reduction="none", delta=0.5), okp.float())
                                               + mean_w(F.huber_loss(e[..., 1], y2, reduction="none", delta=0.5), w2[..., -1]))
    # T5: future hand contact
    w5 = (b["t5_ok"].bool() & b["obj_mask"].bool()[..., None]).float()
    L["t5"] = cfg.get("w_t5", 0.5) * mean_w(F.binary_cross_entropy_with_logits(m.t5_logit(E), b["t5"].float(), reduction="none"), w5)
    # T3: the global V-JEPA latent at +4 s (only with video, as UWM2)
    if cfg["fut"] and "glob_fut" in b:
        okf = (b["glob_fut_ok"] & ~b["aug_jepa_bad"] & E["von"]).float()
        L["t3"] = cfg["w_fut"] * mean_w(1 - F.cosine_similarity(m.fut_pred(E, b), b["glob_fut"].float(), dim=-1), okf)
    # hidden objects (object permanence)
    mu, logs = m.hidden_pos(E)
    xh = (b["obj_geo"][..., :3] / PS).float()
    L["hidden_pos"] = cfg.get("w_hide", 0.3) * mean_w((0.5 * ((xh - mu) / logs.exp()) ** 2 + logs).sum(-1), hide.float())
    return L


def world_loss(m, D, cfg, rng, gen, dev):
    """Self-supervised world losses on every training keyframe, including reset / setup stretches without an instruction:
      fut_pos     every visible object's tracked displacement at +0.5 / 1 / 2 / 4 s (positions only), Huber in PS units;
      hidden_pos  15-30% of the object tokens (not the table) lose their geometry (and image position); their position is predicted
                  from the rest (Gaussian NLL, PS units, table frame): object permanence.
    A separate batch from D.world_train_idx; no composites (their objects have no future tracks), no counterfactual re-placement."""
    from cleantables.worldmodel.base_model import PS
    from cleantables.worldmodel.base_train import make_mode, mean_w
    if not len(D.world_train_idx):
        return {}
    bi = D.world_train_idx[rng.integers(0, len(D.world_train_idx), cfg["bs"])]
    wc = {**cfg, "p_comp": 0.0, "p_rcomp": 0.0, "p_remove": 0.0, "p_addlast": 0.0, "cf": False}
    b = D.make_batch(bi, wc, rng, dev, train=True, gen=gen)
    B, N = b["obj_mask"].shape
    mode = make_mode(b, wc, gen)
    obj = b["obj_mask"].bool() & (b["obj_kind"] != 1)
    lo, hi = cfg.get("p_hide", (0.15, 0.30))
    frac = lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)
    hide = obj & (torch.rand(B, N, device=dev, generator=gen) < frac[:, None])
    b["obj_hide"] = hide
    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
    w = cfg.get("w_world", 1.0)
    fp = m.fut_pos(E)
    tgt = (b["fut_disp"].view(B, N, -1, 3) / PS).float()
    okf = (b["fut_disp_ok"].bool() & (obj & ~hide)[..., None]).float()
    L = {"fut_pos": w * mean_w(F.huber_loss(fp, tgt, reduction="none", delta=0.1).sum(-1), okf)}
    mu, logs = m.hidden_pos(E)
    x = (b["obj_geo"][..., :3] / PS).float()
    nll = (0.5 * ((x - mu) / logs.exp()) ** 2 + logs).sum(-1)
    L["hidden_pos"] = 0.3 * w * mean_w(nll, hide.float())
    return L


def reversed_clip_loss(m, b, cfg, pv, okv_, done):
    """Reversed-clip negatives: the same moment with its 32-frame clip played backwards (and the DINOv2 frame where that reversed
    clip ends), with video (mode 0) and without human hands, as on the robot. Progress should be lower than on the forward clip
    (margin 0.05, as in base_train), and on complete states stop should not fire (the video contradicts "done"), so the stop
    decision has to use the video."""
    sel = b["rev_sel"]
    ok = b["vid_ok"][sel].bool()
    if not ok.any():
        return {}
    sel = sel[ok]
    B = len(b["mover"])
    r = {k: (v[sel.to(v.device)] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B else v) for k, v in b.items() if k not in ("vj_rev", "dn_rev", "rev_sel")}
    r["vj_grid"] = b["vj_rev"][ok].to(b["vj_grid"].dtype); r["dn_grid"] = b["dn_rev"][ok].to(b["dn_grid"].dtype)
    r["hand_geo"] = r["hand_geo"].clone(); r["hand_geo"][..., 0] = 0; r["hand_q_ok"] = torch.zeros_like(r["hand_q_ok"])
    r["objg_ok"] = torch.zeros_like(r["objg_ok"])                          # objg is computed on the forward frames
    E = m.encode(r, torch.zeros(len(sel), dtype=torch.long, device=sel.device)); E["kind"] = r["obj_kind"]
    pvr = torch.sigmoid(m.p_vid(E["cls"][:, 0]).squeeze(-1).float())
    from cleantables.worldmodel.base_train import mean_w
    L = {"rev_prog": cfg.get("w_rev", 1.0) * mean_w(F.relu(0.05 - (pv[sel].detach() - pvr)), okv_[sel])}
    if done is not None and m.use_stop:
        ml, _ = m.pointers(E, r["mover"])
        lj = torch.cat([ml.float(), m.stop_logit(E).float()[:, None]], -1)
        p_stop = F.softmax(lj, -1)[:, -1]
        L["rev_stop"] = cfg.get("w_rev", 1.0) * mean_w(-torch.log1p(-p_stop.clamp(max=1 - 1e-4)), done[sel].float())
    return L


@torch.no_grad()
def val_curve(m, D, cfg, sets, dev):
    """Validation curves: per fixed sample set (train / val), mode 1nh, T1 NLL and end error (cm) of tokens that moved
    (|T1 end - now| > cfg moved_eval, the tracker's rest p90; used only here, never as a training signal) and of those that stayed,
    plus the distance head's AUROC for 'moved'."""
    from sklearn.metrics import roc_auc_score
    m.eval(); out = {}
    for name, ids in sets.items():
        acc = dict(nll_m=[], nll_s=[], err_m=[], err_s=[], e1=[], y=[], fam_m=[], fam_e1=[])
        for b0 in range(0, len(ids), 64):
            b = D.make_batch(ids[b0:b0 + 64], cfg, np.random.default_rng(0), dev, train=False)
            b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
            for k_ in ("obj_contact_ok", "obj_contact2_ok"):
                if k_ in b:
                    b[k_] = torch.zeros_like(b[k_])
            B = b["obj_mask"].shape[0]; E = m.encode(b, torch.ones(B, dtype=torch.long, device=dev))
            ok = (b["t1_ok"].bool() & b["t1_present"].bool() & b["obj_mask"].bool() & (b["obj_kind"] == 0))
            tgt = b["t1_pos"].float(); now = b["obj_geo"][..., :3].float()
            nll = m.t1_nll(E, tgt); end = m.t1_mode(E)[0]
            dh = (tgt - now).norm(dim=-1); mv = dh > cfg["moved_eval"]
            err = 100 * (end - tgt).norm(dim=-1)
            for msk, k_ in ((ok & mv, "m"), (ok & ~mv, "s")):
                acc["nll_" + k_] += nll[msk].tolist(); acc["err_" + k_] += err[msk].tolist()
            if cfg.get("curve_per_family"):
                from cleantables.worldmodel.eval_families import family
                fam_ = np.array([family(str(D.ep_base[i_])) for i_ in ids[b0:b0 + 64]])
                rr_, _ = np.nonzero((ok & mv).cpu().numpy()); acc["fam_m"] += fam_[rr_].tolist()
                rr2, _ = np.nonzero(ok.cpu().numpy()); acc["fam_e1"] += fam_[rr2].tolist()
            if getattr(m, "dist_head", False):
                acc["e1"] += m.dist_pred(E)[..., 0][ok].tolist(); acc["y"] += mv[ok].tolist()
        r = {k: float(np.median(v)) for k, v in acc.items() if k.startswith(("nll", "err")) and v}
        r.update(n_moved=len(acc["err_m"]), n_stayed=len(acc["err_s"]))
        if acc["e1"] and 0 < sum(acc["y"]) < len(acc["y"]):
            r["dist_auroc"] = float(roc_auc_score(acc["y"], acc["e1"]))
        if acc["fam_m"]:                                                    # per family: moved T1 error + distance AUROC
            fm, em = np.array(acc["fam_m"]), np.array(acc["err_m"]); fe, e1, yy = np.array(acc["fam_e1"]), np.array(acc["e1"]), np.array(acc["y"])
            r["per_family"] = {f_: dict(err_m=float(np.median(em[fm == f_])), n_m=int((fm == f_).sum()),
                                        auroc=(float(roc_auc_score(yy[fe == f_], e1[fe == f_])) if len(e1) == len(fe) and 0 < yy[fe == f_].sum() < (fe == f_).sum() else None))
                               for f_ in sorted(set(fm.tolist()))}
        out[name] = r
    m.train()
    return out


def train_one(D, idx, cfg, seed, path, dev="cuda", smoke=False, val_sets=None):
    from torch.optim import AdamW
    torch.manual_seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = build(cfg).to(dev)
    opt = AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    T = 20 if smoke else cfg["steps"]
    ck = path.with_suffix(".ckpt")
    it0 = 0
    if ck.exists() and not smoke:
        st = torch.load(ck, map_location=dev, weights_only=False)
        m.load_state_dict(st["m"]); opt.load_state_dict(st["opt"]); it0 = st["it"] + 1
        rng = np.random.default_rng(seed + 7919 * it0); gen.manual_seed(seed + it0)
        log(f"  resume {path.name} at step {it0}")
    t0 = time.time()
    for it in range(it0, T):
        lr = cfg["lr"] * min(1, (it + 1) / 100) * 0.5 * (1 + math.cos(math.pi * it / T))
        for gp in opt.param_groups:
            gp["lr"] = lr
        bi = idx[rng.integers(0, len(idx), cfg["bs"])]
        force = None
        if cfg.get("comp_balance"):        # a quarter of every batch: instruction drawn uniformly, then a sample, always a composite
            nb = cfg["bs"] // 4; ins = D.instr_id[idx]; u = np.unique(ins)
            bi[:nb] = [rng.choice(idx[ins == u[rng.integers(len(u))]]) for _ in range(nb)]
            force = np.zeros(cfg["bs"], bool); force[:nb] = True
        b = D.make_batch(bi, cfg, rng, dev, train=True, gen=gen, force_comp=force)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda" and cfg.get("amp", False)):   # fp32 by default so CPU and GPU runs are comparable
            Ls = compute_loss_w(m, b, cfg, gen) if cfg.get("watch") else compute_loss2(m, b, cfg, gen)
            if cfg.get("world") and not cfg.get("watch"):
                Ls.update(world_loss(m, D, cfg, rng, gen, dev))
        loss = sum(Ls.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {it}: " + " ".join(f"{k}={v.item():.3f}" for k, v in Ls.items()))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 3.0); opt.step()
        if it % 250 == 0 or it == T - 1:
            pk = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
            log(f"  [{path.stem}] it {it} loss {loss.item():.3f} " + " ".join(f"{k}={v.item():.3f}" for k, v in Ls.items())
                + "".join(f" {k}={v:.3f}" for k, v in getattr(m, "_diag", {}).items())          # endref: entropy of a (detects mixture collapse)
                + f" ({(time.time() - t0) / max(1, it - it0 + 1):.2f} s/it, peak {pk:.2f} GB)"
                + (" pairs {ok}/{tries} (no slots {no_slots}, no rigid place {no_place})".format(**D.pair_stats) if getattr(D, "pair_stats", None) else "")
                + (" re-laid {ok}/{rows} rows (eligible {eligible}, drawn {drawn})".format(**D.rigid_stats) if getattr(D, "rigid_stats", None) else ""))
        ce = int(cfg.get("curve_every", 500))
        if val_sets and (it % ce == ce - 1 or it == T - 1 or it == 0):
            cv = val_curve(m, D, cfg, val_sets, dev); cv["it"] = it + 1
            log(f"  [{path.stem}] curve it {it + 1}: " + json.dumps(cv))
            with open(path.with_suffix(".curve.jsonl"), "a") as f_:
                f_.write(json.dumps(cv) + "\n")
        if not smoke and (it % 500 == 499) and it < T - 1:
            torch.save(dict(m=m.state_dict(), opt=opt.state_dict(), it=it), ck)
    if not smoke:
        torch.save(m.state_dict(), path)
        json.dump(dict(cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, seed=seed, n_train=len(idx),
                       train_episodes=sorted(set(D.ep[idx])), secs=time.time() - t0), open(path.with_suffix(".json"), "w"), indent=1)
        if ck.exists():
            ck.unlink()
    return m


def train_b0(seeds, fold, steps, smoke=False):
    """The base recipe on this split (35 training episodes), with the hashed instruction the base model was trained on."""
    from cleantables.worldmodel.common import _pin_hashed
    from cleantables.worldmodel.base_train import prep_store, to_dev, train_model
    from cleantables.worldmodel import data as v2_data
    _pin_hashed()
    name = os.environ.get("TASKS3_DATA", "tasks3c")
    S = prep_store([name])
    Z = np.load(ROOT / f"data/uwm/{name}.npz", allow_pickle=True)
    S.d["end_ok"] = Z["end_ok"].astype(bool)
    ep = S.d["group"]; ids = {g: i for i, g in enumerate(sorted(set(ep)))}
    S.T["video_id"] = torch.tensor([ids[g] for g in ep])
    G = to_dev(S, "cuda")
    test = np.array([v2_data.is_test(e) for e in ep])
    tr = np.flatnonzero(~test)
    if fold >= 0:
        D = v2_data.Data(feats=False)
        tr = tr[D.fold[tr] != fold]
    out = RUNS2 / "B0"; out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        p = out / f"fold{fold}_seed{s}.pt"
        if p.exists() and not smoke:
            log(f"exists {p}"); continue
        m = train_model(S, G, tr, dict(steps=20 if smoke else steps, **V1_CFG), seed=1000 * (fold + 1) + s, dev="cuda",
                        log=lambda x: log(f"[B0 f{fold} s{s}] {x}"))
        if not smoke:
            torch.save(m.state_dict(), p)
            json.dump(dict(cfg=V1_CFG, steps=steps, n_train=len(tr), train_episodes=sorted(set(ep[tr])), secs=m.train_seconds), open(p.with_suffix(".json"), "w"), indent=1)
        log(f"B0 fold {fold} seed {s} done ({m.train_seconds:.0f} s, {len(tr)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="full"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get("WM_GPU_GB", "3.0")) / 23.5))
    if a.cfg == "B0":
        return train_b0(seeds, a.fold, a.steps, a.smoke)
    from cleantables.worldmodel.data import Data
    cfg = full_cfg(a.cfg); cfg["steps"] = a.steps
    D = Data(text=cfg["text"], feats=bool(cfg["vid"]) or cfg["objg"] or cfg["app"] or cfg["fut"])
    if cfg.get("holdout_names"):
        D.holdout(cfg["holdout_names"])
    D.comp_balance = bool(cfg.get("comp_balance", False)); D.relay_keep_inside = bool(cfg.get("relay_keep_inside", False))
    D.relay_keep_dest = bool(cfg.get("relay_keep_dest", True))
    idx = D.train_idx if a.fold < 0 else D.train_idx[D.fold[D.train_idx] != a.fold]
    out = RUNS2 / a.cfg; out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        p = out / f"fold{a.fold}_seed{s}.pt"
        if p.exists() and not a.smoke:
            log(f"exists {p}"); continue
        vs = None
        if cfg.get("val_log") and a.fold >= 0:                               # fixed sets: 384 validation (fold) / 384 training samples
            rv = np.random.default_rng(5)
            va = D.train_idx[(D.fold[D.train_idx] == a.fold) & (D.sub_kind[D.train_idx] == "whole")]
            tr_ = idx[D.sub_kind[idx] == "whole"]
            vs = dict(val=np.sort(rv.choice(va, min(384, len(va)), replace=False)), train=np.sort(rv.choice(tr_, min(384, len(tr_)), replace=False)))
            if cfg.get("curve_per_family"):                                     # up to 128 validation samples per family
                from cleantables.worldmodel.eval_families import family
                fv = np.array([family(str(D.ep_base[i_])) for i_ in va])
                vs["val"] = np.sort(np.concatenate([rv.choice(va[fv == f_], min(128, int((fv == f_).sum())), replace=False) for f_ in np.unique(fv)]))
        train_one(D, idx, cfg, 1000 * (a.fold + 1) + s, p, a.dev, a.smoke, val_sets=vs)
        log(f"{a.cfg} fold {a.fold} seed {s} done")


if __name__ == "__main__":
    main()
