"""Regularised physics feedback for the world model's flow-matching path decoder, ORW-CFM-W2 style (arXiv 2502.06061).

Differences from the plain fine-tune in cleantables.worldmodel.physics.feedback_base:
  1. Only the flow decoder is trained (f_in, f_pos, f_c, f_t, f_dec, f_out). Trunk, pointers, end heads and progress heads stay
     frozen, so the goal (destination + end) cannot drift and feedback only reshapes the carry between the fixed start and end.
  2. Reward-weighted flow matching on the model's own samples with exponential, group-normalised weights
        w_k = exp(tau * A_k) / mean_k exp(tau * A_k),   A_k = (r_k - mean r) / std r over the K samples of one scene (clipped at W_MAX),
     instead of imitating the top 25 %, which is RWR with a hard threshold and collapses (ORW-CFM-W2 Thm 2 / Lemma 1).
  3. W2 regulariser towards the frozen reference (the starting model):  + alpha * || v_theta(x_t, t) - v_ref(x_t, t) ||^2  at the
     same (x_t, t) (ORW-CFM-W2 Eq. 10, an upper bound on W2(p_theta, p_ref) by their Thm 3). The induced distribution is
     exp(tau N r - beta D) q (their Thm 5): the reward can only re-weight what the reference already proposes.
  4. Generic physics score only (no lift or height term, no path windows, no task terms):
        r = -( log1p(penetration voxels / step)            the moving object's voxels inside other objects or the table (static twin)
               + 5 * share of its voxels outside the workspace (table +-15 cm, up to 1.2 m, inside the 0.5 m grid margin)
               + start gap / 5 cm                            the path must start where the object is
               + 0.5 * (path length / straight distance - 1) path economy )
     Settling and displacing other objects need real dynamics; with a frozen end settling is unchanged, and passing through
     another object is counted by the penetration term.
  Training twins only (feedback_base.episodes('train')); the 7 held-out twins are not used.

  python -m cleantables.worldmodel.physics.feedback train --out OUT_RUN --seed 0 [--alpha 1 --tau 1 --steps 800]
  -> OUT/<out>/fold-1_seed<seed>.pt (+ log). Resumable: a checkpoint every 100 steps (ckpt_seed<seed>.pt) is picked up on restart.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from cleantables.worldmodel.common import OUT, ROOT, RUNS, log
from cleantables.worldmodel.physics.twin_check import FullScorer

FLOW = ("f_in", "f_pos", "f_c", "f_t", "f_dec", "f_out")
W_MAX = 10.0
CW = dict(ws=5.0, gap_m=0.05, econ=0.5)


def flow_params(m):
    return [p for n, p in m.named_parameters() if n.split(".")[0] in FLOW]


@torch.no_grad()
def reward(sc, G, oid, paths_top, H, org_top):
    pb = paths_top.clone(); pb[..., 2] -= H
    P = sc.occ[oid]; c = P.mean(0); d = (P - c).norm(dim=-1, keepdim=True).clamp(min=1e-6); Pe = P - (P - c) / d * 0.005
    pen, ws = [], []
    for k0 in range(0, pb.shape[0], 8):
        Q = Pe[None, None] + pb[k0:k0 + 8, :, None, :]
        ix = ((Q - sc.lo) / 0.005).round().long()
        inb = (ix >= 0).all(-1) & (ix[..., 0] < sc.shape[0]) & (ix[..., 1] < sc.shape[1]) & (ix[..., 2] < sc.shape[2])
        ixc = torch.stack([ix[..., a].clamp(0, sc.shape[a] - 1) for a in range(3)], -1)
        hit = G[ixc[..., 0], ixc[..., 1], ixc[..., 2]] & inb
        below = Q[..., 2] < -0.005
        pen.append((hit | below).float().sum(-1).mean(-1)); ws.append((~inb & ~below).float().mean((-1, -2)))
    pen, ws = torch.cat(pen), torch.cat(ws)
    gap = (paths_top[:, 0] - org_top).norm(dim=-1)
    L = (paths_top[:, 1:] - paths_top[:, :-1]).norm(dim=-1).sum(-1); S = (paths_top[:, -1] - org_top).norm(dim=-1).clamp(min=0.02)
    cost = torch.log1p(pen) + CW["ws"] * ws + gap / CW["gap_m"] + CW["econ"] * (L / S - 1).clamp(min=0)
    return -cost, dict(pen=pen, ws=ws, gap=gap, econ=L / S)


def train(out, seed=0, alpha=1.0, tau=1.0, steps=800, B_sc=12, K=16, lr=1e-4, run_in="tasks3_v4", dev="cuda"):
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel.base_model import UWM, NP, PS
    from cleantables.worldmodel.physics.feedback_base import episodes, imagine
    od = OUT / out; od.mkdir(parents=True, exist_ok=True); lf = od / f"train_seed{seed}.log"
    cfg = dict(seed=seed, alpha=alpha, tau=tau, steps=steps, B_sc=B_sc, K=K, lr=lr, run_in=run_in, cost_weights=CW, w_max=W_MAX,
               trained=list(FLOW), episodes=episodes("train"))
    json.dump(cfg, open(od / f"cfg_seed{seed}.json", "w"), indent=1)
    m = UWM(path_resid=True).to(dev); m.load_state_dict(torch.load(RUNS / run_in / f"fold-1_seed{seed}.pt", map_location=dev))
    ref = copy.deepcopy(m).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    for p in m.parameters():
        p.requires_grad_(False)
    fp = flow_params(m)
    for p in fp:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(fp, lr=lr, weight_decay=0.0)
    it0 = 0
    ck = od / f"ckpt_seed{seed}.pt"
    if ck.exists():
        C = torch.load(ck, map_location=dev); m.load_state_dict(C["model"]); opt.load_state_dict(C["opt"]); it0 = C["it"] + 1
        log(f"resumed at step {it0}", lf)
    rng = np.random.default_rng(1000 * seed + it0); gen = torch.Generator(device=dev); gen.manual_seed(seed * 7919 + it0)
    eps = episodes("train")
    bridges, scorers = {}, {}
    t0 = time.time()
    for it in range(it0, steps):
        m.eval(); items = []
        for _ in range(B_sc):
            ep = eps[rng.integers(len(eps))]
            if ep not in bridges:
                bridges[ep] = W.WMBridge(ep, ckpts=[RUNS / run_in / f"fold-1_seed{seed}.pt"], device=dev); bridges[ep].models = [m]
                scorers[ep] = FullScorer(ep, dev)
            Bb = bridges[ep]; Z = Bb.Z; ids = [str(x) for x in Z["ids"]]
            task = [i for i, r in enumerate(Z["role"]) if str(r) == "task"]
            P0 = Z["start_pos"].copy() if rng.random() < 0.2 else W.shuffle_layout(Z, Bb.S, rng, task)
            if P0 is None:
                continue
            i = int(task[rng.integers(len(task))])
            paths, end, d, b, tg = imagine(Bb, P0, i, K)
            H = float(Z["extents"][i, 2])
            org = torch.as_tensor(P0[i], device=dev, dtype=torch.float32).clone(); org[2] += H
            r, parts = reward(scorers[ep], scorers[ep].grid(ids, P0, i), ids[i], paths, H, org)
            A = (r - r.mean()) / (r.std() + 1e-6)
            w = torch.exp(tau * A); w = (w / w.mean()).clamp(max=W_MAX)
            items.append((b, i, tg, paths, w, r, parts))
        if not items:
            continue
        m.eval()                                                # dropout off, so the W2 term is 0 at step 0
        l_rw, l_w2 = 0.0, 0.0
        for b, i, tg, tgt, w, _, _ in items:
            n = tgt.shape[0]
            bb = {k: (v.expand(n, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in b.items()}
            mode = torch.ones(n, dtype=torch.long, device=dev); mv = torch.full((n,), i, dtype=torch.long, device=dev); tgn = tg.expand(n)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                E = m.encode(bb, mode); E["kind"] = bb["obj_kind"]                 # frozen trunk: same features for m and ref
                c, _, _ = m.heads_from(E, mv, tgn)
            org = bb["obj_geo"][:, i, :3]
            rel = (tgt - org[:, None]) / PS
            u = torch.linspace(0, 1, NP, device=dev)[None, :, None]
            x1 = torch.cat([rel - u * rel[:, -1:], torch.zeros_like(rel)], -1)
            x0 = torch.randn(x1.shape, device=dev, generator=gen); t = torch.rand(n, device=dev, generator=gen)
            xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = m.velocity(E, c.detach(), xt, t)
                with torch.no_grad():
                    vr = ref.velocity(E, c, xt, t)
            l_rw = l_rw + (w * ((v.float() - (x1 - x0))[..., :3] ** 2).mean((-1, -2))).mean() / len(items)
            l_w2 = l_w2 + ((v.float() - vr.float())[..., :3] ** 2).mean() / len(items)
        loss = l_rw + alpha * l_w2
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(fp, 1.0); opt.step()
        if it % 25 == 0 or it == steps - 1:
            R = torch.cat([x[5] for x in items]); P = {k: torch.cat([x[6][k] for x in items]) for k in items[0][6]}
            log(f"[{out} s{seed}] it {it} loss_rw {float(l_rw):.3f} w2 {float(l_w2):.4f} | reward {R.mean():.2f} pen {P['pen'].mean():.1f} vox "
                f"ws {P['ws'].mean():.3f} gap {100 * P['gap'].mean():.1f} cm econ {P['econ'].median():.2f} | {time.time() - t0:.0f}s "
                f"| GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", lf)
        if it % 100 == 99:
            torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), it=it), ck)
    torch.save(m.state_dict(), od / f"fold-1_seed{seed}.pt")
    if ck.exists():
        ck.unlink()
    log(f"done {out} seed {seed}", lf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--out", default="regpf_a1"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--tau", type=float, default=1.0); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--w_ws", type=float, default=1.0, help="multiplier on the workspace weight (sensitivity check)")
    ap.add_argument("--w_gap", type=float, default=1.0, help="multiplier on the start-gap weight (1 / 5 cm)")
    ap.add_argument("--w_econ", type=float, default=1.0, help="multiplier on the path-economy weight")
    a = ap.parse_args()
    CW.update(ws=CW["ws"] * a.w_ws, gap_m=CW["gap_m"] / a.w_gap, econ=CW["econ"] * a.w_econ)
    if a.cmd == "train":
        train(a.out, a.seed, a.alpha, a.tau, a.steps)
