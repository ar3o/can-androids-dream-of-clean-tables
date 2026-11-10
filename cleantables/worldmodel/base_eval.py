"""Evaluation for the base world model: grouped hold-out (k-fold over groups; k = number of groups gives
leave-one-group-out) plus the unseen split, cluster-bootstrap CIs that resample groups, the baselines of
cleantables.perception.frame_model.score, and an ablation runner.

  $VLA_PY -m cleantables.worldmodel.base_eval --data box --folds 6 --members 3 --steps 1200 --out runs/uwm/box"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from cleantables.worldmodel.base_model import NP, PS, MIX
from cleantables.worldmodel.base_train import prep_store, to_dev, get_batch, train_model, DEF
from cleantables.worldmodel.store import ROOT

NaN = float("nan")


def yaw_of(g):
    return 0.5 * torch.atan2(g[..., 6], g[..., 7])


@torch.no_grad()
def predict(models, G, idx, dev="cuda", K=8, mode=0, bs=64):
    out = {}
    for s in range(0, len(idx), bs):
        ii = torch.as_tensor(idx[s:s + bs], device=dev)
        b = get_batch(G, ii, dev)
        B = len(ii)
        md = torch.full((B,), mode, device=dev, dtype=torch.long) if mode >= 0 else torch.zeros(B, dtype=torch.long, device=dev)
        md = torch.where(b["vj_ok"].bool(), md, torch.ones_like(md))
        pvrev = []
        pm, pt, rel, offp, offo, offs, paths, pp_f, pp_r, pv, ttm, jc, seen, pathp, kpp = ([] for _ in range(15))
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E = m.encode(b, md); E["kind"] = b["obj_kind"]
                ml, tl = m.pointers(E, b["mover"])                              # target pointer given the true mover
                mvp = ml.argmax(-1)
                _, tlp = m.pointers(E, mvp)
                tgp = tlp.argmax(-1)
                tg_use = torch.where(b["target"] >= 0, b["target"], tgp)
                # oracle context: true mover / target
                c_o, rel_o, mdn_o = m.heads_from(E, b["mover"], tg_use)
                c_p, rel_p, mdn_p = m.heads_from(E, mvp, tgp)
                spf = m.sample_paths_anchored if getattr(m, "path_resid", False) else m.sample_paths
                ps_ = spf(E, c_o, K=K)
                ps_p = spf(E, c_p, K=K)
                x1 = torch.cat([b["path_obj"], b["path_hand"]], -1)
                org = b["obj_geo"][torch.arange(B), b["mover"].clamp(min=0), :3]
                p6 = torch.nan_to_num(torch.cat([b["path_obj"] - org[:, None], b["path_hand"] - org[:, None]], -1))
                pf = m.path_prog(E, c_o, p6); pr = m.path_prog(E, c_o, p6.flip(1))
                pvv = torch.sigmoid(m.p_vid(E["cls"][:, 0]).squeeze(-1))
                b3 = dict(b); b3["vj_tok"] = b["vj_tok"].flip(1); E3 = m.encode(b3, md)
                pvr = torch.sigmoid(m.p_vid(E3["cls"][:, 0]).squeeze(-1)); tt = m.p_ttm(E["cls"][:, 0]).squeeze(-1)
                j = m.jepa_pred(E, b, md); sn = m.seen(E["cls"][:, 3]).squeeze(-1)
            pm.append(F.softmax(ml.float(), -1)); pt.append(F.softmax(tlp.float(), -1))
            rel.append(F.softmax(rel_p.float(), -1))
            for mdn, lst in ((mdn_p, offp), (mdn_o, offo)):
                top = mdn["logit"].float().argmax(-1)
                lst.append(0.1 * torch.gather(mdn["mu"].float(), 1, top[:, None, None].expand(B, 1, 3)).squeeze(1))
            z = mdn_o
            comp = torch.multinomial(F.softmax(z["logit"].float(), -1), K, replacement=True)
            mu = torch.gather(z["mu"].float(), 1, comp[..., None].expand(B, K, 3)); sd = torch.gather(z["logs"].float(), 1, comp[..., None].expand(B, K, 3)).exp()
            offs.append(0.1 * (mu + sd * torch.randn_like(mu)))
            paths.append(ps_.float()); pathp.append(ps_p.float())
            pp_f.append(pf.float()); pp_r.append(pr.float()); pv.append(pvv.float()); pvrev.append(pvr.float()); ttm.append(tt.float())
            jc.append(F.cosine_similarity(j.float(), torch.nan_to_num(b["vj_future_glob"]), dim=-1)); seen.append(torch.sigmoid(sn.float()))
            kpp.append(F.cosine_similarity(j.float(), torch.nan_to_num(b["vj_future_glob"]), dim=-1) * 0)
        st = lambda l: torch.stack(l, 0)
        M = len(models)
        r = dict(p_mover=st(pm).mean(0), p_target=st(pt).mean(0), p_rel=st(rel).mean(0), off_pred=st(offp).mean(0), off_orc=st(offo).mean(0),
                 off_samples=torch.cat(offs, 1), paths=torch.cat(paths, 1), paths_pred=torch.cat(pathp, 1),
                 prog_fwd=st(pp_f).mean(0), prog_rev=st(pp_r).mean(0), prog_video=st(pv).mean(0), prog_video_rev=st(pvrev).mean(0), ttm=st(ttm).mean(0),
                 jepa_cos=st(jc).mean(0), p_seen=st(seen).mean(0),
                 # uncertainty: std across ensemble members of the mover probability mass, mean path and end offset
                 sp_mover=st(pm).std(0).sum(-1) if M > 1 else torch.zeros(B, device=dev),
                 sp_path=st(paths).mean(2).std(0)[..., :3].norm(dim=-1).mean(-1) if M > 1 else torch.zeros(B, device=dev),
                 sp_end=st(offo).std(0).norm(dim=-1) if M > 1 else torch.zeros(B, device=dev),
                 ent=-(st(pm).mean(0).clamp(1e-9).log() * st(pm).mean(0)).sum(-1),
                 persist_cos=F.cosine_similarity(b["vj_tok"].mean(1), torch.nan_to_num(b["vj_future_glob"]), dim=-1))
        for k, v in r.items():
            out.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def boot(vals, groups, n=2000, seed=0):
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ok = np.isfinite(vals); vals, groups = vals[ok], groups[ok]
    if len(vals) == 0:
        return [NaN, NaN, NaN, 0]
    ug = sorted(set(groups)); sums = np.array([vals[groups == g].sum() for g in ug]); cnt = np.array([(groups == g).sum() for g in ug])
    r = np.random.default_rng(seed); bs = []
    for _ in range(n):
        q = r.integers(0, len(ug), len(ug)); bs.append(sums[q].sum() / max(cnt[q].sum(), 1))
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(vals))]


def world_end(geo, tg, off, table_idx):
    """Target-frame offset -> world xy (object target: target centre + R(yaw) off; table/leaves: table token)."""
    j = tg if tg < 24 else table_idx
    g = geo[j]; y = 0.5 * np.arctan2(g[6], g[7]); c, s = np.cos(y), np.sin(y)
    return g[:2] + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])


def spearman_by(vid, x, y):
    out = []
    for v in sorted(set(vid)):
        i = vid == v
        if i.sum() >= 4 and np.ptp(y[i]) > 0 and np.ptp(x[i]) > 0:
            out.append(spearmanr(x[i], y[i])[0])
    return float(np.mean(out)) if out else NaN, len(out)


def metrics(S, idx, P, train_rel_major=None):
    d = S.d; n = len(idx); R = {}
    g = d["group"][idx]
    def add(k, v):
        R.setdefault(k, []).append(v)
    geo = d["obj_geo"][idx]; om = d["obj_mask"][idx].astype(bool); kind = d["obj_kind"][idx]
    mov = om & (kind == 0)
    mv, tg = d["mover"][idx], d["target"][idx]
    pm = P["p_mover"]
    hand = d["hand_geo"][idx]
    rows = []
    for a in range(n):
        keys = {}
        nm = mov[a].sum()
        def put(k, v):
            keys[k] = v
        if "mover_set" in d and nm > 1:
            msa = d["mover_set"][idx][a] & mov[a]
            if msa.any():
                put("mover_in_remaining_set", float(msa[pm[a].argmax()])); put("mover_in_set_rand", float(msa.sum() / nm))
        if mv[a] >= 0 and nm > 1:
            put("mover_top1", float(pm[a].argmax() == mv[a])); put("mover_rand", 1.0 / nm)
            vol = np.where(mov[a], geo[a, :, 3] * geo[a, :, 4] * geo[a, :, 5], -1)
            put("mover_largest", float(vol.argmax() == mv[a]))
            if hand[a, 0, 0] > 0.5:
                dd = np.where(mov[a], np.linalg.norm(geo[a, :, :2] - hand[a, 0, 1:3], axis=1), 1e9)
                put("mover_nearest_hand", float(dd.argmin() == mv[a]))
        if mv[a] >= 0 and tg[a] >= 0:
            pt = P["p_target"][a]
            put("target_ok", float(pt.argmax() == tg[a]))
            oth = mov[a].copy(); oth[mv[a]] = False
            if oth.any():
                dd = np.where(oth, np.linalg.norm(geo[a, :, :2] - geo[a, mv[a], :2], axis=1), 1e9)
                put("target_nearest_obj", float(dd.argmin() == tg[a]))
                put("target_table_const", float(tg[a] == 24))
            rl = d["relation"][idx][a]
            if rl >= 0:
                put("relation_ok", float(P["p_rel"][a].argmax() == rl))
                if train_rel_major is not None:
                    put("relation_majority", float(rl == train_rel_major))
            eo = d["end_off"][idx][a]
            if np.isfinite(eo).all():
                ep = d["end_pose"][idx][a]
                ti = int(d["table_idx"][idx][a]) if "table_idx" in d else 0
                put("end_err_oracle_target_cm", 100 * np.linalg.norm(P["off_orc"][a][:2] - eo[:2]))
                put("end_bestofK_cm", 100 * np.linalg.norm(P["off_samples"][a][:, :2] - eo[:2], axis=1).min())
                put("end_z_err_cm", 100 * abs(P["off_orc"][a][2] - eo[2]))
                put("baseline_target_centre_cm", 100 * np.linalg.norm(eo[:2]))
                put("baseline_stays_cm", 100 * np.linalg.norm(geo[a, mv[a], :2] - ep[:2]))
        po = d["path_obj"][idx][a]
        if mv[a] >= 0 and np.isfinite(po).all():
            org = geo[a, mv[a], :3]; rel = po - org
            ps = P["paths"][a][..., :3]
            e = np.linalg.norm(ps - rel[None], axis=-1).mean(-1)
            put("path_ade_mean_cm", 100 * np.linalg.norm(ps.mean(0) - rel, axis=-1).mean()); put("path_ade_sample_cm", 100 * e.mean()); put("path_bestofK_cm", 100 * e.min())
            line = rel[0] + np.linspace(0, 1, NP)[:, None] * (rel[-1] - rel[0])
            put("baseline_line_oracle_end_cm", 100 * np.linalg.norm(line - rel, axis=-1).mean())
            put("baseline_stay_cm", 100 * np.linalg.norm(rel[0][None] - rel, axis=-1).mean())
            put("path_endpoint_cm", 100 * np.linalg.norm(ps.mean(0)[-1] - rel[-1]))
            put("baseline_start_to_origin_end_cm", 100 * np.linalg.norm(rel[-1]))
            ph = d["path_hand"][idx][a]; ok = d["path_hand_ok"][idx][a].astype(bool) & np.isfinite(ph).all(-1)
            if ok.sum() > 3:
                relh = ph - org
                eh = np.linalg.norm(P["paths"][a][..., 3:6] - relh[None], axis=-1)[:, ok].mean(-1)
                put("hand_path_ade_cm", 100 * eh.mean()); put("hand_path_bestofK_cm", 100 * eh.min())
                put("baseline_hand_stay_cm", 100 * np.linalg.norm(relh[ok][0][None] - relh[ok], axis=-1).mean())
            ps_ = d["prog_steps"][idx][a]
            if np.isfinite(ps_).all():
                put("prog_step_spearman_fwd", spearmanr(P["prog_fwd"][a], ps_)[0]); put("prog_step_spearman_rev_path", spearmanr(P["prog_rev"][a], ps_[::-1])[0])
                put("prog_step_content_gain", float(P["prog_fwd"][a][-8:].mean() - P["prog_fwd"][a][:8].mean()))
                put("prog_step_rev_gain", float(P["prog_rev"][a][:8].mean() - P["prog_rev"][a][-8:].mean()))
        if np.isfinite(d["vj_future_glob"][idx][a]).all():
            put("jepa_cos", P["jepa_cos"][a]); put("jepa_cos_persistence_baseline", P["persist_cos"][a])
        rows.append(keys)
    for r in rows:
        for k, v in r.items():
            add(k, v)
    ks = {k: [] for k in R}
    out = {}
    for k in R:
        vals, gg = [], []
        for a, r in enumerate(rows):
            if k in r:
                vals.append(r[k]); gg.append(g[a])
        out[k] = boot(vals, gg)
    # progress rank across a video
    vid = d["video"][idx]
    okv = d["prog_video_ok"][idx].astype(bool)
    if okv.any():
        out["prog_video_spearman_within_video"] = list(spearman_by(vid[okv], P["prog_video"][okv], d["prog_video"][idx][okv]))
    if okv.any():
        pv, pr_, tt_ = P["prog_video"], P["prog_video_rev"], d["prog_video"][idx]
        rng = np.random.default_rng(0)
        fw, rv, gn, shf, pa = [], [], [], [], []
        for v in sorted(set(vid[okv])):
            i = np.flatnonzero((vid == v) & okv); i = i[np.argsort(d["snap_k"][idx][i])]
            if len(i) < 6 or np.ptp(tt_[i]) == 0:
                continue
            fw.append(spearmanr(pv[i], tt_[i])[0]); rv.append(spearmanr(pr_[i], tt_[i][::-1])[0])   # reversed video: playback time runs backwards
            q = max(1, len(i) // 5); gn.append(pv[i][-q:].mean() - pv[i][:q].mean())
            shf.append(spearmanr(pv[i], rng.permutation(tt_[i]))[0]); pa.append(float((pv[i] > pr_[i]).mean()))
        out["prog_video_spearman_fwd"] = [float(np.mean(fw)), len(fw)]
        out["prog_video_spearman_reversed_video"] = [float(np.mean(rv)), len(rv)]       # negative = the reversed video is seen as going backwards
        out["prog_video_gain_end_minus_start"] = [float(np.mean(gn)), len(gn)]
        out["prog_video_shuffled_control_spearman"] = [float(np.mean(shf)), len(shf)]
        out["prog_video_fwd_clip_scores_above_reversed_clip"] = [float(np.mean(pa)), len(pa)]
    ttm = d["time_to_move"][idx] if "time_to_move" in d else None
    if ttm is not None and np.isfinite(ttm).any():
        okt = np.isfinite(ttm)
        out["ttm_spearman_within_video"] = list(spearman_by(d["ev_id"][idx][okt].astype(str) if "ev_id" in d else vid[okt], P["ttm"][okt], ttm[okt]))
    # uncertainty: how well ensemble spread, entropy and p(seen) flag errors
    if "mover_top1" in R:
        cor = np.array([r.get("mover_top1", NaN) for r in rows]); ok = np.isfinite(cor) & (np.unique(cor[ok := np.isfinite(cor)]).size > 1)
        from sklearn.metrics import roc_auc_score
        okk = np.isfinite(cor)
        if len(np.unique(cor[okk])) > 1:
            out["trust_auroc_entropy_flags_mover_error"] = float(roc_auc_score(1 - cor[okk], P["ent"][okk]))
            out["trust_auroc_member_spread_flags_mover_error"] = float(roc_auc_score(1 - cor[okk], P["sp_mover"][okk] + 1e-9 * np.random.rand(okk.sum())))
    return out


def fmt(R):
    return {k: (list(np.round(v, 4)) if isinstance(v, list) else round(v, 4)) for k, v in R.items()}


def folds_of(S, groups, k):
    gs = list(groups); rng = np.random.default_rng(0); rng.shuffle(gs)
    return [gs[i::k] for i in range(k)]


def run_protocol(S, G, dataset, cfg, folds, members, tag, dev="cuda", log=print, modes=(0,), keep_models=False):
    """Per-sample predictions for every train-split sample (held out by group) and for the unseen split (from the ensemble trained on all train data)."""
    tr_all = S.where(dataset=dataset, split="train")
    groups = S.groups(dataset, "train")
    held = {m: {} for m in modes}; secs = []
    for fi, fg in enumerate(folds_of(S, groups, folds)):
        te = np.array([i for i in tr_all if S.d["group"][i] in fg]); tr = np.array([i for i in tr_all if S.d["group"][i] not in fg])
        ms = []
        for mi in range(members):
            t0 = time.time(); ms.append(train_model(S, G, tr, cfg, seed=100 * fi + mi, dev=dev, bootstrap=members > 1)); secs.append(ms[-1].train_seconds)
        log(f"[{tag}] fold {fi} groups {fg} train {len(tr)} test {len(te)} secs/run {secs[-1]:.0f}")
        for md in modes:
            P = predict(ms, G, te, dev, mode=md)
            held[md][fi] = (te, P)
        del ms; torch.cuda.empty_cache()
    return held, secs


def concat_held(h):
    idx = np.concatenate([v[0] for v in h.values()]); keys = next(iter(h.values()))[1].keys()
    return idx, {k: np.concatenate([v[1][k] for v in h.values()]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="box"); ap.add_argument("--folds", type=int, default=6); ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--out", default="runs/uwm/box"); ap.add_argument("--final_members", type=int, default=5)
    ap.add_argument("--skip_folds", action="store_true"); ap.add_argument("--vjepa_only_too", action="store_true")
    a = ap.parse_args()
    dev = "cuda"; out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    lf = open(out / "log.txt", "a")
    def log(s):
        print(s, flush=True); lf.write(s + "\n"); lf.flush()
    S = prep_store(a.data.split(",")); G = to_dev(S, dev)
    cfg = dict(steps=a.steps); ds = a.data.split(",")[0]
    res = {}
    tr_all = S.where(dataset=ds, split="train")
    rl = S.d["relation"][tr_all]; rl = rl[rl >= 0]; maj = int(np.bincount(rl).argmax()) if len(rl) else None
    # final ensemble on all train data -> unseen split + saved checkpoint
    t0 = time.time(); fin = []
    for mi in range(a.final_members):
        fin.append(train_model(S, G, tr_all, cfg, seed=7000 + mi, dev=dev, bootstrap=True))
        torch.save(fin[-1].state_dict(), out / f"final_member{mi}.pt"); log(f"[final] member {mi} {fin[-1].train_seconds:.0f}s")
    json.dump(dict(final_secs_per_member=fin[0].train_seconds, params=sum(p.numel() for p in fin[0].parameters()), cfg={**DEF, **cfg}), open(out / "final_info.json", "w"))
    un = S.where(dataset=ds, split="unseen")
    for md, nm in ((0, "both"), (1, "3d_only"), (2, "vjepa_only")):
        if len(un):
            Pu = predict(fin, G, un, dev, mode=md); res[f"unseen_{nm}"] = fmt(metrics(S, un, Pu, maj))
            np.savez_compressed(out / f"pred_unseen_{nm}.npz", idx=un, **Pu)
    json.dump(res, open(out / "results.json", "w"), indent=1)
    # grouped hold-out of the seen groups
    if not a.skip_folds:
        held, secs = run_protocol(S, G, ds, cfg, a.folds, a.members, "kfold", dev, log, modes=(0, 1, 2))
        for md, nm in ((0, "both"), (1, "3d_only"), (2, "vjepa_only")):
            idx, P = concat_held(held[md]); res[f"heldout_groups_{nm}"] = fmt(metrics(S, idx, P, maj))
            np.savez_compressed(out / f"pred_heldout_{nm}.npz", idx=idx, **P)
        res["secs_per_run_kfold"] = float(np.mean(secs))
    json.dump(res, open(out / "results.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
