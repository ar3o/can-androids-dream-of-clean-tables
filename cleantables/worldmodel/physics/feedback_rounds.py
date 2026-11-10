"""Physics feedback for the world model, run online in rounds:
  sample (CPU)  the current model imagines plans in training twins (20 % measured layout, else random layouts of all objects)
  score         PhysX replay (cleantables.worldmodel.physx_replay, Isaac; the only GPU step) -> generic cost (physx_jobs.cost),
                or the voxel score of cleantables.worldmodel.physics.feedback
  update (CPU)  frozen trunk; trained: the flow path decoder and the end heads (target-relative end mixture, end displacement, end yaw)
Paths: ORW-CFM-W2 (arXiv 2502.06061) as in cleantables.worldmodel.physics.feedback: reward-weighted flow matching on the model's
  own samples, w = exp(tau A) / mean (A = group-normalised reward of the K samples of one scene, clipped at W_MAX)
  + alpha * ||v - v_ref||^2.
End heads: reward-weighted regression (Peters & Schaal, ICML 2007; IJRR 27(2) 2008, Sec. 3, Alg. 2-3: E-step weights from
  transformed rewards, M-step weighted maximum likelihood of the model's own samples). RWR has no regulariser of its own; mirroring
  the W2 term of ORW-CFM-W2 we add + alpha_end * NLL of samples from the frozen imitation heads. The same weights exp(tau A) are
  used, with one fixed tau for paths and ends (RWR would adapt tau).
Plans per scene: K_PATH path samples (each to the model's own end) + K_END end samples (end offset and yaw sampled from the heads,
  carried along the scene's median path re-anchored to that end). Training twins only (feedback_base.episodes('train')).

  python -m cleantables.worldmodel.physics.feedback_rounds --name NAME --seed 0 --init INIT_RUN [--score physx|voxel] [--rounds 6] [--smoke]
Out: OUT/wm_v2/pf/<name>/seed<s>/round<r>/{scenes.pt, jobs.npz, scores.npz, model.pt}; final OUT/wm_v2/runs/<name>/fold-1_seed<s>.pt
Resumable per round and step.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.common import OUT, log
from cleantables.worldmodel.physics import physx_jobs as PJ

FLOW = ("f_in", "f_pos", "f_c", "f_t", "f_dec", "f_out")
ENDH = ("mdn", "disp", "yaw_h")
W_MAX = 10.0
PF = Path(os.environ.get("WM_PF_DIR", str(OUT / "wm_v2" / "pf")))           # override with WM_PF_DIR to keep runs apart
RUNS2 = Path(os.environ.get("WM_RUNS2", str(OUT / "wm_v2" / "runs")))


def head_params(m, end=True):
    names = FLOW + (ENDH if end else ())
    return [p for n, p in m.named_parameters() if n.split(".")[0] in names]


def mix_sample(z, n, gen, scale=1.0):
    """Samples (n, 3) from a diagonal Gaussian mixture dict(logit (1,M), mu (1,M,3), logs (1,M,3))."""
    comp = torch.multinomial(F.softmax(z["logit"][0].float(), -1), n, replacement=True, generator=gen)
    mu, sd = z["mu"][0, comp].float(), z["logs"][0, comp].float().exp()
    return mu + scale * sd * torch.randn(mu.shape, generator=gen)


def mix_nll(z, x):
    """x (n,3) -> (n,) NLL under the mixture z (batch 1, broadcast)."""
    mu, logs = z["mu"].float(), z["logs"].float()
    nll = (0.5 * ((x[:, None] - mu) / logs.exp()) ** 2 + logs).sum(-1) - F.log_softmax(z["logit"].float(), -1)
    return -torch.logsumexp(-nll, -1)


def mdn_of(m, c):
    """Target-relative end mixture of model m for context c, with gradients (heads_from would recompute the trunk context)."""
    from cleantables.worldmodel.base_model import MIX
    z = m.mdn(c).view(-1, MIX, 7)
    return dict(logit=z[..., 0], mu=z[..., 1:4], logs=z[..., 4:7].clamp(-3, 1.5))


def yaw_sample(m, c, n, gen):
    z = m.yaw_mix(c.float())
    comp = torch.multinomial(F.softmax(z["logit"][0], -1), n, replacement=True, generator=gen)
    mu, k = z["mu"][0, comp], z["logk"][0, comp].exp()
    x = torch.distributions.VonMises(mu, k).sample()                        # 2 x yaw
    return 0.5 * x


# ------------------------------------------------------------------ one scene: imagine plans
@torch.no_grad()
def scene_plans(B, m, P0, i, K_path, K_end, gen):
    from cleantables.worldmodel.base_model import PS
    from cleantables.worldmodel.bridge import long_side_yaw, wrap_half
    I, dev = B.I, B.dev
    b = B._batch(torch.as_tensor(P0[None], dtype=torch.float32), torch.zeros(1, I), None)
    mode = torch.ones(1, dtype=torch.long); mv = torch.tensor([i])
    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
    _, tl = m.pointers(E, mv)
    pd = F.softmax(torch.cat([tl[:, :I], tl[:, 24:25]], -1).float(), -1)
    d = int(pd.argmax(-1)); tg = torch.tensor([d if d < I else 24])
    c, _, mdn = m.heads_from(E, mv, tg); zd = m.end_disp(c)
    H = float(B.Z["extents"][i, 2])
    org = b["obj_geo"][0, i, :3].float()                                    # top centre
    tm = int(mdn["logit"].float().argmax(-1))
    end_top = (b["obj_geo"][0, d, :3] + 0.1 * mdn["mu"].float()[0, tm]) if d < I else org + zd["mu"].float()[0, int(zd["logit"].float().argmax(-1))] * PS
    paths = m.sample_paths_anchored(E, c, K=K_path).float()[0, :, :, :3] + org             # (K,32,3) top centre
    lin = torch.linspace(0, 1, 32)[None, :, None]
    paths = paths + lin * (end_top - paths[:, -1:])
    y0 = long_side_yaw(B.Z, i)
    yaw_mode = float(m.yaw_mode(c.float())[0]) if getattr(m, "use_yaw", False) else float("nan")
    dy_mode = wrap_half(yaw_mode - y0) if np.isfinite(yaw_mode) else 0.0
    plans, kinds = [], []

    def to_root(p_top):
        r = p_top.clone(); r[..., 2] -= H
        return r
    for k in range(K_path):
        pr = to_root(paths[k]); plans.append(torch.cat([pr, pr[-1:]], 0)); kinds.append(("path", k))
    # end samples (offset from the end head; yaw from the yaw head) along the median path re-anchored to each sampled end
    med = paths.median(0).values
    if d < I:
        e = mix_sample(mdn, K_end, gen)                                                  # MDN units (end_off / 0.1)
        ends = b["obj_geo"][0, d, :3][None] + 0.1 * e
    else:
        e = mix_sample(zd, K_end, gen)                                                   # PS units
        ends = org[None] + e * PS
    ys = yaw_sample(m, c, K_end, gen) if getattr(m, "use_yaw", False) else None
    for k in range(K_end):
        p = med + lin[0] * (ends[k] - med[-1])
        pr = to_root(p); plans.append(torch.cat([pr, pr[-1:]], 0)); kinds.append(("end", k))
    dys = [dy_mode] * K_path + ([wrap_half(float(y) - y0) for y in ys] if ys is not None else [0.0] * K_end)
    org_root = org.clone(); org_root[2] -= H
    return dict(b={k: v for k, v in b.items()}, i=i, d=d, tg=tg, paths_top=paths, end_samples=e, yaw_samples=ys, plans=torch.stack(plans),
                kinds=kinds, dyaw=np.array(dys, np.float32), org_root=org_root, table=d >= I)


def sample_round(m, n_scenes, K_path, K_end, seed, dev="cpu"):
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel.physics.feedback_base import episodes
    from cleantables.worldmodel.bridge import WMBridge2
    rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
    eps = episodes("train")
    bridges, scenes = {}, []
    for _ in range(n_scenes):
        ep = eps[rng.integers(len(eps))]
        if ep not in bridges:
            bridges[ep] = WMBridge2(ep, None, models=[m], device=dev)
        B = bridges[ep]; Z = B.Z
        objs = list(range(B.I))                                                  # every object; no roles are read
        P0 = Z["start_pos"].copy() if rng.random() < 0.2 else W.shuffle_layout(Z, B.S, rng, objs)
        if P0 is None:
            continue
        i = int(objs[rng.integers(len(objs))])
        s = scene_plans(B, m, P0, i, K_path, K_end, gen)
        s.update(ep=ep, P0=P0)
        scenes.append(s)
    return scenes


def score_round(scenes, rd, how, copies, dev="cpu"):
    """-> per scene: cost (n_plans,) (NaN = not informative) and the raw outcome arrays."""
    if how == "voxel":
        from cleantables.worldmodel.physics.twin_check import FullScorer
        from cleantables.worldmodel.physics.feedback import reward
        sc = {}
        out = []
        for s in scenes:
            if s["ep"] not in sc:
                sc[s["ep"]] = FullScorer(s["ep"], dev)
            S = sc[s["ep"]]
            zt = np.load(Path(os.environ.get("WM_TWIN_ROOT", str(PJ.ROOT / "data/twin_table/ep"))) / s["ep"] / "wm_tokens.npz")
            ids = [str(x) for x in zt["ids"]]
            P = s["plans"][:, :32].clone()
            Hm = float(zt["extents"][s["i"], 2])
            P[..., 2] += Hm                                                              # back to top centre for feedback.reward
            org = s["org_root"].clone(); org[2] += Hm
            r, _ = reward(S, S.grid(ids, s["P0"], s["i"]), ids[s["i"]], P, Hm, org)
            out.append(dict(cost=(-r).numpy()))
        return out
    plans = []
    for s in scenes:
        L = PJ.layout4(s["P0"])
        for k in range(len(s["plans"])):
            plans.append(dict(twin=s["ep"], obj=s["i"], layout=L, path=s["plans"][k].numpy(), dyaw=float(s["dyaw"][k])))
    PJ.write_jobs(plans, rd / "jobs.npz")
    S = PJ.run_physx(rd / "jobs.npz", rd / "scores.npz", copies=copies)
    allp = np.stack([p["path"] for p in plans]); org = np.concatenate([np.repeat(s["org_root"].numpy()[None], len(s["plans"]), 0) for s in scenes])
    gap, lr = PJ.geometry(allp, org)
    c = PJ.cost({k: S[k] for k in S.files}, gap, lr)
    out, j = [], 0
    for s in scenes:
        n = len(s["plans"])
        out.append(dict(cost=c[j:j + n], **{k: S[k][j:j + n] for k in ("f_obj", "f_hand", "n_disp", "settle_cm", "off_table", "pre_unstable")}))
        j += n
    return out


def weights(cost, tau):
    """exp(tau A) / mean over the finite entries (A = group-normalised reward = -cost); NaN entries get weight 0."""
    ok = np.isfinite(cost)
    w = np.zeros_like(cost, dtype=np.float32)
    if ok.sum() < 2:
        return w
    r = -cost[ok]; A = (r - r.mean()) / (r.std() + 1e-6)
    e = np.exp(tau * A); e = np.minimum(e / e.mean(), W_MAX)
    w[ok] = e
    return w


def update(m, ref, scenes, scores, cfg, steps, gen, lf, end=True):
    from cleantables.worldmodel.base_model import NP, PS
    params = head_params(m, end)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    items = []
    for s, sc in zip(scenes, scores):
        kinds = s["kinds"]
        ip = [j for j, k in enumerate(kinds) if k[0] == "path"]; ie = [j for j, k in enumerate(kinds) if k[0] == "end"]
        wp = weights(sc["cost"][ip], cfg["tau"]); we = weights(sc["cost"][ie], cfg["tau"]) if ie else None
        if wp.sum() == 0 and (we is None or we.sum() == 0):
            continue
        items.append((s, torch.as_tensor(wp), None if we is None else torch.as_tensor(we)))
    if not items:
        log("no informative scenes this round", lf); return
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    for st in range(steps):
        mb = [items[j] for j in rng.integers(0, len(items), min(cfg["B_sc"], len(items)))]
        m.eval()
        l_rw = l_w2 = l_end = l_ref = 0.0
        for s, wp, we in mb:
            b = s["b"]; i = s["i"]; tgt = s["paths_top"]; n = tgt.shape[0]
            bb = {k: (v.expand(n, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in b.items()}
            mode = torch.ones(n, dtype=torch.long); mv = torch.full((n,), i, dtype=torch.long); tgn = s["tg"].expand(n)
            with torch.no_grad():
                E = m.encode(bb, mode); E["kind"] = bb["obj_kind"]
                c, _, _ = m.heads_from(E, mv, tgn)
                Er = dict(E)
            org = bb["obj_geo"][:, i, :3]
            rel = (tgt - org[:, None]) / PS
            u = torch.linspace(0, 1, NP)[None, :, None]
            x1 = torch.cat([rel - u * rel[:, -1:], torch.zeros_like(rel)], -1)
            x0 = torch.randn(x1.shape, generator=gen); t = torch.rand(n, generator=gen)
            xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
            v = m.velocity(E, c, xt, t)
            with torch.no_grad():
                vr = ref.velocity(Er, c, xt, t)
            l_rw = l_rw + (wp * ((v - (x1 - x0))[..., :3] ** 2).mean((-1, -2))).mean() / len(mb)
            l_w2 = l_w2 + ((v - vr)[..., :3] ** 2).mean() / len(mb)
            if end and we is not None:
                c1 = c[:1]
                z = m.end_disp(c1) if s["table"] else mdn_of(m, c1)
                with torch.no_grad():
                    zr = ref.end_disp(c1) if s["table"] else mdn_of(ref, c1)
                    xr = mix_sample(zr, 16, gen)
                nll = mix_nll(z, s["end_samples"])
                le = (we * nll).mean()
                lr_ = mix_nll(z, xr).mean()
                if s["yaw_samples"] is not None:
                    yn = m.yaw_nll(c1.expand(len(s["yaw_samples"]), -1).float(), s["yaw_samples"])
                    le = le + cfg["w_yaw"] * (we * yn).mean()
                    with torch.no_grad():
                        yr = yaw_sample(ref, c1, 16, gen)
                    lr_ = lr_ + cfg["w_yaw"] * m.yaw_nll(c1.expand(16, -1).float(), yr).mean()
                l_end = l_end + le / len(mb); l_ref = l_ref + lr_ / len(mb)
        loss = l_rw + cfg["alpha"] * l_w2 + (l_end + cfg["alpha_end"] * l_ref if end else 0.0)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if st % 20 == 0 or st == steps - 1:
            log(f"   step {st} rw {float(l_rw):.3f} w2 {float(l_w2):.4f} end {float(l_end):.3f} ref {float(l_ref):.3f}", lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", default="full")
    ap.add_argument("--score", default="physx", choices=["physx", "voxel"]); ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--scenes", type=int, default=48); ap.add_argument("--k_path", type=int, default=8); ap.add_argument("--k_end", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60); ap.add_argument("--alpha", type=float, default=0.1); ap.add_argument("--alpha_end", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=2.0); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--no_end", action="store_true", help="paths only (v1's fix)"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--weights", default="", help="json of cost-weight multipliers (sensitivity), e.g. '{\"disp\": 2}'")
    a = ap.parse_args()
    torch.set_num_threads(8); torch.manual_seed(a.seed)
    if a.weights:
        for k, v in json.loads(a.weights).items():
            PJ.PW[k] *= v
    if a.smoke:
        a.rounds, a.scenes, a.k_path, a.k_end, a.steps = 1, 3, 4, 2, 2
    from cleantables.worldmodel.bridge import load_models
    cfg = dict(alpha=a.alpha, alpha_end=a.alpha_end, tau=a.tau, lr=a.lr, B_sc=12, w_yaw=0.5)
    sd = PF / a.name / f"seed{a.seed}"; sd.mkdir(parents=True, exist_ok=True); lf = sd / "log.txt"
    json.dump(dict(vars(a), cost_weights=PJ.PW, w_max=W_MAX, trained=list(FLOW) + ([] if a.no_end else list(ENDH))), open(sd / "cfg.json", "w"), indent=1)
    m = load_models(RUNS2 / a.init, seeds=(a.seed,))[0]
    ref = copy.deepcopy(m).eval()
    for p in list(ref.parameters()) + list(m.parameters()):
        p.requires_grad_(False)
    for p in head_params(m, not a.no_end):
        p.requires_grad_(True)
    gen = torch.Generator().manual_seed(a.seed)
    for r in range(a.rounds):
        rd = sd / f"round{r}"; rd.mkdir(exist_ok=True)
        if (rd / "model.pt").exists():
            m.load_state_dict(torch.load(rd / "model.pt")); continue
        t0 = time.time()
        if (rd / "scenes.pt").exists():
            scenes = torch.load(rd / "scenes.pt", weights_only=False)
        else:
            scenes = sample_round(m, a.scenes, a.k_path, a.k_end, seed=1000 * a.seed + r)
            torch.save(scenes, rd / "scenes.pt")
        log(f"[{a.name} s{a.seed}] round {r}: {len(scenes)} scenes sampled ({time.time() - t0:.0f} s)", lf)
        scores = score_round(scenes, rd, a.score, a.copies)
        C = np.concatenate([s["cost"] for s in scores])
        log(f"[{a.name} s{a.seed}] round {r}: mean cost {np.nanmean(C):.3f} (finite {np.isfinite(C).mean():.2f}) ({time.time() - t0:.0f} s)", lf)
        update(m, ref, scenes, scores, cfg, a.steps, gen, lf, end=not a.no_end)
        torch.save(m.state_dict(), rd / "model.pt")
    if not a.smoke:
        od = RUNS2 / a.name; od.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), od / f"fold-1_seed{a.seed}.pt")
        base = json.load(open(RUNS2 / a.init / f"fold-1_seed{a.seed}.json"))
        base["pf"] = json.load(open(sd / "cfg.json"))
        json.dump(base, open(od / f"fold-1_seed{a.seed}.json", "w"), indent=1)
    log(f"done {a.name} seed {a.seed}", lf)


if __name__ == "__main__":
    main()
