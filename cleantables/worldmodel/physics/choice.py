"""Physics-scored next-object choice: stack order (and "don't lift the bowl with the fruit in it") learned from physical
consequences, with no rule about stacks.

  scenes   training twins of all three tasks (fruit, tools, box stacking; twin root WM_TWIN_ROOT): 40% the human episode's own
           mid-task snapshot states (where real stacks and contents occur), 10% its start layout, 25% random layouts, 25% layouts
           rolled forward 1-3 steps by the model itself (its top choice placed at its predicted end pose, plan_all)
  choices  C different next objects per scene, sampled from the model's p_now (tempered Gumbel top-k), each carried in PhysX
           along the model's follow path (physx_jobs.followed, the path the robot follows)
  cost     the generic cost physx_jobs.cost / PW (includes displaced and toppled other objects; no lift term)
  update   the next-object head (mover_h) only: reward-weighted regression on the candidates (Peters & Schaal, ICML 2007: weighted
           maximum likelihood of the model's own samples, w = exp(tau A), A = within-scene advantage of -cost, capped as in
           feedback_rounds) + beta * KL(frozen imitation pointer || model) over the joint softmax (objects + STOP), so instruction
           following and STOP are kept. Trunk, destination, end, path, yaw and STOP heads are frozen.

  python -m cleantables.worldmodel.physics.choice --name NAME --seed 0 --init INIT_RUN [--rounds 5] [--smoke]
Out: WM_PF_DIR/<name>/seed<s>/round<r>/{scenes.pt, jobs.npz, scores.npz, model.pt}; final WM_RUNS2/<name>/fold-1_seed<s>.pt.
Resumable. Models with end-referenced or watch heads train with cleantables.worldmodel.physics.stage instead. Random layouts
shuffle all objects (movable_idx reads no roles).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.common import OUT, ROOT, log
from cleantables.worldmodel.physics import physx_jobs as PJ
from cleantables.worldmodel.physics.feedback_rounds import W_MAX

PF = Path(os.environ.get("WM_PF_DIR", str(OUT / "wm_v2" / "pf")))
RUNS2 = Path(os.environ.get("WM_RUNS2", str(OUT / "wm_v2" / "runs")))
TRAINED = ("mover_h",)


def train_twins():
    """Training episodes of all three tasks that have a twin in the twin root (WM_TWIN_ROOT)."""
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel import bridge as v2_bridge  # noqa: F401  (applies WM_TWIN_ROOT)
    from cleantables.worldmodel.data import is_test
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    return sorted(t["id"] for t in T if not is_test(t["id"]) and (W.EP / t["id"] / "wm_tokens.npz").exists() and (W.EP / t["id"] / "scene.json").exists())


_HUMAN = {}


def human_states(ep, Z, S):
    """The human episode's snapshot states (tasks3 data, same object positions as the training data) as twin layouts: each twin object
    with a video track takes its token's x, y and bottom (= top - its upright height) at that snapshot; the others keep their start pose."""
    if not _HUMAN:
        from cleantables.worldmodel.data import Data
        D = Data(feats=False)
        d = D.S.d
        for i in range(D.n):                                                  # nothing in a hand: >= 0.3 s before the next move, or done
            ttm = d["time_to_move"][i]
            if (np.isfinite(ttm) and ttm >= 0.3) or d["mover"][i] < 0:
                _HUMAN.setdefault(str(D.ep[i]), []).append((d["obj_id"][i].copy(), d["obj_geo"][i].copy(), d["obj_mask"][i].copy(), d["obj_kind"][i].copy()))
    out = []
    tr = [o.get("track") for o in S["objects"]][:len(Z["ids"])]
    for oid, geo, mask, kind in _HUMAN.get(ep, []):
        P = Z["start_pos"].copy()
        for j, t in enumerate(tr):
            s_ = np.flatnonzero((oid == t) & mask.astype(bool) & (kind == 0)) if t is not None else []
            if len(s_):
                g = geo[s_[0]]
                P[j] = [g[0], g[1], max(0.0, float(g[2]) - float(Z["extents"][j, 2]))]
        out.append(P.astype(np.float32))
    return out


def movable_idx(Z):
    """Indices of every object of the twin; random layouts shuffle all objects and read no roles."""
    return list(range(len(Z["ids"])))


@torch.no_grad()
def roll_forward(B, P, Y, n_steps, rng, greedy=0.7):
    """Roll the model's own chain forward kinematically: its top (or a sampled) p_now object is placed at its predicted end pose, n_steps times."""
    P, Y = P.copy(), Y.copy()
    for _ in range(n_steps):
        o = B.plan_all(torch.as_tensor(P[None], dtype=torch.float32), torch.as_tensor(Y[None], dtype=torch.float32), paths=False)
        p = o["p_now_raw"][0].numpy().astype(np.float64)
        if p.sum() <= 1e-6:
            break
        i = int(p.argmax()) if rng.random() < greedy else int(rng.choice(len(p), p=p / p.sum()))
        P[i] = o["end_xyz"][0, i].numpy(); Y[i] = Y[i] + float(o["end_yaw_off"][0, i]) if "end_yaw_off" in o else Y[i]
    return P, Y


def sample_round(m, n_scenes, n_cand, seed, temp=2.0, dr_names=(), dr_frac=0.0):
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel.bridge import WMBridge2
    rng = np.random.default_rng(seed)
    eps = train_twins()
    bridges, scenes = {}, []
    tries = 0
    while len(scenes) < n_scenes and tries < 4 * n_scenes:
        tries += 1
        use_dr = len(dr_names) > 0 and rng.random() < dr_frac                 # stage 2: domain-randomised, physics-settled twins
        ep = dr_names[rng.integers(len(dr_names))] if use_dr else eps[rng.integers(len(eps))]
        if ep not in bridges:
            bridges[ep] = WMBridge2(ep, None, models=[m], device="cpu")
        B = bridges[ep]; Z = B.Z
        u = rng.random()
        src = ("dr" if u < 0.5 else "dr_rolled") if use_dr else ("human" if u < 0.4 else ("start" if u < 0.5 else ("random" if u < 0.75 else "rolled")))
        if use_dr:                                                            # the settled layout (yaw included); half rolled forward
            P = Z["start_pos"].copy()
        elif src == "human":                                                    # the human episode's mid-task states
            H_ = human_states(ep, Z, B.S)
            if not H_:
                continue
            P = H_[rng.integers(len(H_))].copy()
        else:
            P = Z["start_pos"].copy() if src == "start" else W.shuffle_layout(Z, B.S, rng, movable_idx(Z))
        if P is None:
            continue
        Y = Z["start_yaw_off"].astype(np.float32).copy() if (use_dr and "start_yaw_off" in Z) else np.zeros(B.I, np.float32)
        n_roll = int(rng.integers(1, 4)) if src in ("rolled", "dr_rolled") else 0   # rolled forward 1-3 steps by the model itself
        if n_roll:
            P, Y = roll_forward(B, P, Y, n_roll, rng)
        o = B.plan_all(torch.as_tensor(P[None], dtype=torch.float32), torch.as_tensor(Y[None]), paths=False)
        p = o["p_now_raw"][0].numpy().astype(np.float64)
        ok = np.flatnonzero(p > 1e-6)
        if len(ok) < 2:
            continue
        g = np.log(p[ok]) / temp + rng.gumbel(size=len(ok))                   # tempered Gumbel top-k: C different candidates
        cand = ok[np.argsort(-g)[:min(n_cand, len(ok))]]
        plans = []
        for i in cand:
            path, d, dy = PJ.followed(B, P, int(i), yaw_off=Y)
            plans.append(dict(i=int(i), path=path, d=d, dyaw=float(dy)))
        b = B._batch(torch.as_tensor(P[None], dtype=torch.float32), torch.as_tensor(Y[None]), None)
        scenes.append(dict(ep=ep, src=src, P0=P, Y0=Y, n_roll=n_roll, cand=[int(c) for c in cand], plans=plans, p_now=p, I=B.I,
                           blocked=o["blocked"][0].numpy(), b={k: v for k, v in b.items()}))
    return scenes


def score_round(scenes, rd, copies):
    plans = []
    for s in scenes:
        L = PJ.layout4(s["P0"], s["Y0"])
        for q in s["plans"]:
            plans.append(dict(twin=s["ep"], obj=q["i"], layout=L, path=q["path"], dyaw=q["dyaw"]))
    PJ.write_jobs(plans, rd / "jobs.npz")
    S = PJ.run_physx(rd / "jobs.npz", rd / "scores.npz", copies=copies)
    allp = np.stack([p["path"] for p in plans])
    org = np.stack([s["P0"][q["i"]] for s in scenes for q in s["plans"]]).astype(np.float32)
    gap, lr = PJ.geometry(allp, org)
    c = PJ.cost({k: S[k] for k in S.files}, gap, lr)
    j = 0
    for s in scenes:
        n = len(s["plans"])
        s["cost"] = c[j:j + n]; s["n_disp"] = S["n_disp"][j:j + n]; s["n_topple"] = S["n_topple"][j:j + n]
        j += n
    return scenes


def weights(cost, tau):
    ok = np.isfinite(cost)
    w = np.zeros_like(cost, dtype=np.float32)
    if ok.sum() < 2:
        return w
    r = -cost[ok]; A = (r - r.mean()) / (r.std() + 1e-6)
    e = np.minimum(np.exp(tau * A) / np.exp(tau * A).mean(), W_MAX)
    w[ok] = e
    return w


def joint_logp(m, E, I):
    ml, _ = m.pointers(E, torch.zeros(E["obj"].shape[0], dtype=torch.long))
    return F.log_softmax(torch.cat([ml.float()[:, :I], m.stop_logit(E).float()[:, None]], -1), -1)


def update(m, ref, scenes, cfg, steps, rng, lf):
    params = [p for n, p in m.named_parameters() if n.split(".")[0] in TRAINED]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    items = []
    for s in scenes:
        w = weights(s["cost"], cfg["tau"])
        if w.sum() == 0:
            continue
        mode = torch.ones(1, dtype=torch.long)
        with torch.no_grad():
            E = m.encode(s["b"], mode); E["kind"] = s["b"]["obj_kind"]          # frozen trunk: computed once per scene
            lr_ = joint_logp(ref, E, s["I"])[0]
        items.append((E, s["I"], torch.as_tensor(s["cand"]), torch.as_tensor(w), lr_))
    if not items:
        log("no informative scenes this round", lf); return
    for st in range(steps):
        mb = [items[j] for j in rng.integers(0, len(items), min(cfg["B_sc"], len(items)))]
        l_rw = l_kl = 0.0
        for E, I, cand, w, lr_ in mb:
            lp = joint_logp(m, E, I)[0]
            l_rw = l_rw - (w * lp[cand]).sum() / w.sum() / len(mb)
            l_kl = l_kl + (lr_.exp() * (lr_ - lp)).sum() / len(mb)
        loss = l_rw + cfg["beta"] * l_kl
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if st % 20 == 0 or st == steps - 1:
            log(f"   step {st} rwr {float(l_rw):.3f} kl {float(l_kl):.4f}", lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", default="demo_frozen")
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--scenes", type=int, default=64); ap.add_argument("--cand", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60); ap.add_argument("--tau", type=float, default=2.0); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--copies", type=int, default=16); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dr_list", default="", help="json with the DR twin names (dr_twins.py; stage 2)"); ap.add_argument("--dr_frac", type=float, default=0.6)
    ap.add_argument("--ref", default="", help="the imitation model for the KL leash (default: --init); stage 2: the original imitation run")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8"))); torch.manual_seed(a.seed)
    if a.smoke:
        a.rounds, a.scenes, a.steps, a.copies = 1, 4, 3, 2
    from cleantables.worldmodel.bridge import load_models
    cfg = dict(tau=a.tau, beta=a.beta, lr=a.lr, B_sc=16)
    sd = PF / a.name / f"seed{a.seed}"; sd.mkdir(parents=True, exist_ok=True); lf = sd / "log.txt"
    json.dump(dict(vars(a), cost_weights=PJ.PW, w_max=W_MAX, trained=list(TRAINED), twin_root=os.environ.get("WM_TWIN_ROOT", ""),
                   train_twins=train_twins(), n_dr_twins=len(json.load(open(a.dr_list))["names"]) if a.dr_list else 0), open(sd / "cfg.json", "w"), indent=1)
    m = load_models(RUNS2 / a.init if (RUNS2 / a.init).exists() else OUT / "wm_v2" / "runs" / a.init, seeds=(a.seed,))[0]
    if a.ref:
        rsrc = RUNS2 / a.ref if (RUNS2 / a.ref).exists() else OUT / "wm_v2" / "runs" / a.ref
        ref = load_models(rsrc, seeds=(a.seed,))[0].eval()
    else:
        ref = copy.deepcopy(m).eval()
    dr_names = json.load(open(a.dr_list))["names"] if a.dr_list else []
    for p in list(ref.parameters()) + list(m.parameters()):
        p.requires_grad_(False)
    for n, p in m.named_parameters():
        if n.split(".")[0] in TRAINED:
            p.requires_grad_(True)
    rng = np.random.default_rng(a.seed)
    for r in range(a.rounds):
        rd = sd / f"round{r}"; rd.mkdir(exist_ok=True)
        if (rd / "model.pt").exists():
            m.load_state_dict(torch.load(rd / "model.pt")); continue
        t0 = time.time()
        if (rd / "scenes.pt").exists():
            scenes = torch.load(rd / "scenes.pt", weights_only=False)
        else:
            m.eval(); scenes = sample_round(m, a.scenes, a.cand, seed=1000 * a.seed + r + (500 if dr_names else 0), dr_names=dr_names, dr_frac=a.dr_frac)
            torch.save(scenes, rd / "scenes.pt")
        top_blocked = np.mean([bool(s["blocked"][int(np.argmax(s["p_now"]))]) for s in scenes])
        log(f"[{a.name} s{a.seed}] round {r}: {len(scenes)} scenes ({sum(s['n_roll'] > 0 for s in scenes)} rolled, "
            f"{sum(s['src'].startswith('dr') for s in scenes)} DR), top choice blocked "
            f"{top_blocked:.2f} ({time.time() - t0:.0f} s)", lf)
        scenes = score_round(scenes, rd, a.copies)
        C = np.concatenate([s["cost"] for s in scenes]); D = np.concatenate([s["n_disp"] for s in scenes])
        log(f"[{a.name} s{a.seed}] round {r}: mean cost {np.nanmean(C):.3f} (finite {np.isfinite(C).mean():.2f}), moves another "
            f"{np.nanmean(D > 0):.2f} ({time.time() - t0:.0f} s)", lf)
        update(m, ref, scenes, cfg, a.steps, rng, lf)
        torch.save(m.state_dict(), rd / "model.pt")
    if not a.smoke:
        od = RUNS2 / a.name; od.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), od / f"fold-1_seed{a.seed}.pt")
        src = RUNS2 / a.init if (RUNS2 / a.init).exists() else OUT / "wm_v2" / "runs" / a.init
        base = json.load(open(src / f"fold-1_seed{a.seed}.json"))
        base["pf_choice"] = json.load(open(sd / "cfg.json"))
        json.dump(base, open(od / f"fold-1_seed{a.seed}.json", "w"), indent=1)
    log(f"done {a.name} seed {a.seed}", lf)


if __name__ == "__main__":
    main()
