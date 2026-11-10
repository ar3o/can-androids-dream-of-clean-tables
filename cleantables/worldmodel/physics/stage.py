"""Physics stage for the world model, using its T1 / T2 / T5 outputs and no move labels. The model plans with its own read-outs
(bridge plan_all: next object = its largest predicted T2 displacement, goal = that object's T1 mode, path = its T2 sample anchored to
the goal); the plans are replayed in PhysX with the force-limited carry and scored with generic terms only, and the model is updated
by reward-weighted regression on what physics did: (a) T1 NLL of the carried object's executed end, (b) T2 flow loss on the executed
trajectories of all objects with the intervention token = the carried object + its commanded path (m.phys_loss). (b) is also T4:
every replay's per-object trajectories are stored as intervention data (interventions/round<r>.npz). Every object is treated alike:
no role, no destination, no end-at-goal term, no hand-made lift candidates, no human-label states.

  pool     a cached pool of physical pre-carry states shared by all seeds and rounds (pool_r<r>.pt). Round 0: from each seed state
           (training twins: start layout or a random layout of all objects; domain-randomised twins: their settled layout, dr_frac
           of them) every object is carried once along the initial ensemble's evaluated plan (the T4 pass); each replay's settled
           start (pose_start) and end (pose_end, that object now carried) join the pool if every object is on the table and
           upright (<= 25 deg; the tokens assume upright objects). Each later round adds the end state of every seed's own pick.
           A state's carried objects (its lineage) may move without cost and are not chosen again; lineages stop at MAX_STEPS.
  round r  n_states pool states, the same for every seed (distinct twins first). Per seed and state, from that seed's read-outs:
           C next-object candidates (tempered Gumbel top-k of p_now, carried objects excluded), each along its plan_all path (the
           evaluated plan, fixed sampling seed); for the first one (the seed's own pick) also k_path - 1 path re-samples
           (--diverse: single flow samples, K = 1; else evaluation-style, K = K_MED) and k_end plans re-anchored to ends sampled
           from the T1 end mixture at ESC x its spread (t1_sample). Diversity comes only from the model's own sampler. The CPU
           pre-screen (physics.screen) drops clear hits, then one PhysX chain job holds every seed's candidates (single carries;
           WM_CARRY=force, WM_HITS=1; every object's pose recorded at the T2 times --traj_t).
  cost     generic physics terms per carry (carry_cost; weights W):
              W["hit"] x graded hit of other objects within 1 cm during 5-95% of the carry (replay hit_g: start-neighbour exemption
                         only while still in start contact; the table is not counted; no destination exemption)
            + W["disp"] x others displaced > 1 cm + W["topple"] x others tilted > 30 deg (all but the carried object and the lineage)
            + W["econ"] x xy economy (table-plane length / straight - 1, floor 2 cm; lifting is free)
            + W["off"] x objects off the table + settle_cm / 5 + settle_deg / 30 (the carried object still settling after release)
            pre_unstable layouts: NaN (no information). A screened candidate gets max(W["hit"] x hit_lb, the worst replayed cost of
            its group): a lower bound of its hit term, ranked no better than anything replayed.
  update   per seed (models with phys_loss): w = exp(tau A) within each group (the C choice candidates of a state; the pick's path
            candidates), A = normalised -cost, capped at W_MAX (Peters & Schaal 2007, feedback_rounds.weights); loss = sum_k w_k
            (t1_k + choice_k) / n_group (the executed end; the executed displacements without the intervention = the next-object
            read-out d_h and the un-intervened T2) + mean_k t4_k over the same replays, unweighted (the intervened T2 =
            consequences) + a regulariser (alpha x phys_loss of the frozen reference's own outcome for the same intervention: its
            T1 mode and one T2 sample with the intervention token, ref_sample, in the spirit of ORW-CFM-W2; for models that cannot
            sample: beta_p x ||theta - theta_ref||^2 over the trained parameters). Trained: m.phys_params() if defined, else
            everything outside the trunk (TRUNK). --t4 uniform: also the t4 term of every stored replay (all rounds), unweighted.
            STOP is not trained by physics (no generic term can score not moving).
            Models without endref / phys_loss use the older update: mover_h RWR + beta KL over the joint softmax (STOP masked out
            of the RWR), and feedback_rounds.update on chains.median_plans candidates.

  python -m cleantables.worldmodel.physics.stage --name NAME --init INIT_RUN --seeds 0,1,2 --graded --diverse --k_path 8 --k_end 4 \
       --lr 3e-4 --steps 150 --dr_list <dr_list.json> --dr_frac 0.6 [--traj_t (default 0.5 s x m.H)] [--rounds 5] [--smoke]
   (env: WM_CARRY=force WM_HITS=1 WM_TWIN_ROOT=<dr_root> WM_PF_DIR=... WM_RUNS2=... CUDA_VISIBLE_DEVICES= ; PhysX takes the GPU itself)
Out: WM_PF_DIR/<name>/{cfg.json, log.txt, pool_r<r>.pt, round<r>/{cand.pt, carries.npz, carry_scores.npz, scored.pt, model_s<k>.pt},
interventions/round<r>.npz}; final WM_RUNS2/<name>/fold-1_seed<s>.pt (+ .json). Resumable per round.
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

from cleantables.worldmodel.common import OUT, log
from cleantables.worldmodel.physics import physx_jobs as PJ
from cleantables.worldmodel.physics.chains import DISP_M, MAX_STEPS, TILT_DEG, V2, _tilt, _yaw_off, run_chain_physx
from cleantables.worldmodel.physics.feedback_rounds import W_MAX, weights

PF = Path(os.environ.get("WM_PF_DIR", str(OUT / "wm_v2" / "pf")))
RUNS2 = Path(os.environ.get("WM_RUNS2", str(OUT / "wm_v2" / "runs")))
W = dict(hit=10.0, disp=1.0, topple=2.0, econ=0.5, off=5.0, settle_cm=1 / 5, settle_deg=1 / 30)   # same weights as physics.chains (V2 hit, CW)
ZONE_M = 0.01                                                                  # 0/1 + safe-zone hit (without --graded), as physics.chains V2
ESC = 2.0                                                                      # end samples at 2 x the mixture's spread (--diverse)
K_MED = 4                                                                      # flow samples per member in the evaluated plan
TILT_POOL = 25.0                                                               # pool states keep upright objects only
EVAL_SEED = 12345                                                              # plan_all's fixed sampling seed = the evaluated plan
TRUNK = ("geo", "null_geo", "txt", "crop", "kind", "app_v", "app_d", "app_c", "hgeo", "hq", "hid", "cls", "mod", "task", "reads", "uv_e",
         "no_uv", "objg_e", "dgoal_e", "typ", "trunk", "norm", "ctc")          # encode() inputs and trunk of all model classes: frozen here
OLD_TRAINED = ("mover_h", "f_in", "f_pos", "f_c", "f_t", "f_dec", "f_out", "mdn", "disp", "yaw_h")   # heads trained for models without phys_loss
_BR = {}
PATH_ONLY = False                                                              # --path_only: physics trains only the carry's shape
ROUTE_VALID = int(os.environ.get("WM_ROUTE_VALID", "0"))                    # WM_ROUTE_VALID=1: with
PUSH_LAMBDA_CM = float(os.environ.get("WM_PUSH_LAMBDA_CM", "5"))             # --route_opt every optimised path is replayed in PhysX and distilled
                                                                               # with w = 1[hit < 0.5] * exp(-push_sum_cm / PUSH_LAMBDA_CM), push_sum =
                                                                               # sum over every other object of its displacement (cm) from before the
                                                                               # carry to settled after release (no phase split, no relations)
ROUTE_OPT = 0                                                                  # --route_opt STEPS (gradient steps per path, physics.route_opt)
PATH_PARAMS = ("g_in", "g_step", "g_obj", "g_t", "g_dec", "g_out", "iv_e")    # the T2 flow decoder (the carry's shape) + the intervention token


def bridge(ep, models):
    from cleantables.worldmodel.bridge import WMBridge2
    k = (ep, id(models[0]), len(models))
    if k not in _BR:
        _BR[k] = WMBridge2(ep, None, models=list(models), device="cpu")
    return _BR[k]


def any_bridge(ep):
    """A bridge of the twin for geometry only (scene, tokens): any model will do."""
    for (e, _, _), B in _BR.items():
        if e == ep:
            return B
    raise KeyError(ep)


def is_new(m):
    """True for the watch-and-learn model (T1 / T2 / T5, no move heads) or anything exposing phys_loss."""
    return bool(getattr(m, "watch", False)) or hasattr(m, "phys_loss")


# ------------------------------------------------------------------ states
def state_of(B, pose):
    """Physical poses (n_max, 7) -> (xyz (I, 3) root, yaw offsets (I,) on top of the baked yaw)."""
    I = B.I
    return pose[:I, :3].astype(np.float32).copy(), np.array([_yaw_off(pose[k, 3:7], B.S["objects"][k].get("yaw_baked", 0.0)) for k in range(I)], np.float32)


def top_of(B, pose):
    """Root poses (..., I, 7) -> top centres (..., I, 3): root + R (0, 0, H) (tilt-correct)."""
    from scipy.spatial.transform import Rotation as R
    I = B.I; p = np.asarray(pose)[..., :I, :]
    H = np.asarray(B.Z["extents"], np.float64)[:, 2]
    sh = p.shape[:-1]
    q = p.reshape(-1, 7); ok = np.isfinite(q).all(-1)
    up = np.full((len(q), 3), np.nan); up[ok] = R.from_quat(q[ok, 3:7]).apply([0, 0, 1.0])
    return (p[..., :3] + up.reshape(sh + (3,)) * H.reshape((1,) * (len(sh) - 1) + (I, 1))).astype(np.float32)


def valid(B, pose):
    """Every object on the table and upright: the world model's tokens assume upright objects on the table."""
    from scipy.spatial.transform import Rotation as R
    I = B.I; p = pose[:I]
    if not np.isfinite(p).all():
        return False
    tb = B.S["table"]
    up = R.from_quat(p[:, 3:7]).apply([0, 0, 1.0])[:, 2]
    on = (p[:, 0] >= tb["x"][0]) & (p[:, 0] <= tb["x"][1]) & (p[:, 1] >= tb["y"][0]) & (p[:, 1] <= tb["y"][1]) & (p[:, 2] > -0.03)
    return bool(on.all() and (np.degrees(np.arccos(np.clip(up, -1, 1))) <= TILT_POOL).all())


def seed_states(models, n, rng, eps, dr_names=(), dr_frac=0.6):
    """Round-0 states: domain-randomised twins (settled layout) at dr_frac, else a training twin's start layout or a random layout of all its objects."""
    from cleantables.worldmodel import base_bridge as Wb
    out, tries = [], 0
    while len(out) < n and tries < 4 * n:
        tries += 1
        use_dr = len(dr_names) > 0 and rng.random() < dr_frac
        ep = dr_names[rng.integers(len(dr_names))] if use_dr else eps[rng.integers(len(eps))]
        B = bridge(ep, models); Z = B.Z
        src = "dr" if use_dr else ("start" if rng.random() < 0.5 else "random")
        P = Z["start_pos"].copy() if src != "random" else Wb.shuffle_layout(Z, B.S, rng, list(range(B.I)))
        if P is None:
            continue
        Y = Z["start_yaw_off"].astype(np.float32).copy() if (use_dr and "start_yaw_off" in Z) else np.zeros(B.I, np.float32)
        out.append(dict(ep=ep, P=P.astype(np.float32), Y=Y, carried=np.zeros(B.I, bool), depth=0, src=src))
    return out


def pick_states(pool, n, rng):
    """n pool states, distinct twins first (fewer PhysX batches), not exhausted (something left to carry)."""
    idx = [k for k in rng.permutation(len(pool)) if not pool[k]["carried"].all() and pool[k]["depth"] < MAX_STEPS]
    seen, a, b = set(), [], []
    for k in idx:
        (a if pool[k]["ep"] not in seen else b).append(k); seen.add(pool[k]["ep"])
    return [pool[k] for k in (a + b)[:n]]


# ------------------------------------------------------------------ plans from the model
def _batch(B, P, Y):
    P, Y = np.asarray(P, np.float32), np.asarray(Y, np.float32)
    if P.ndim == 2:
        P, Y = P[None], Y[None]
    return B._batch(torch.as_tensor(P), torch.as_tensor(Y), None)


def encode(m, b):
    with torch.no_grad():
        E = m.encode(b, torch.ones(b["obj_geo"].shape[0], dtype=torch.long)); E["kind"] = b["obj_kind"]
    return E


def read_out(B, P, Y, carried, seed, K):
    """plan_all read-outs from (P, Y) with the carried objects taken: p_now (I,), path (I, 33, 3) root, dyaw (I,), end root (I, 3)."""
    o = B.plan_all(torch.as_tensor(P[None]), torch.as_tensor(Y[None]), taken=carried[None], paths=True, K=K, seed=seed, contact=False)
    dy = o["end_yaw_off"][0].numpy() if "end_yaw_off" in o else np.zeros(B.I, np.float32)
    return o["p_now"][0].double().numpy(), o["path"][0].numpy().astype(np.float32), np.nan_to_num(dy).astype(np.float32), o["end_xyz"][0].numpy()


@torch.no_grad()
def t1_sample(m, E, a, n, gen, scale=ESC):
    """n ends (n, 3) top centre, world metres, of object a from the watch model's T1 mixture: reference j ~ a_aj, offset ~ N(o_aj,
    (scale x sigma_aj)^2) in j's frame (sizes as the model's), back to the world."""
    from cleantables.worldmodel.model import from_frame
    M = m.t1_mix(E)
    j = torch.multinomial(F.softmax(M["logit"][0, a], -1), n, replacement=True, generator=gen)
    sdf = m.t1_sdf(M) if hasattr(m, "t1_sdf") else M["logs"].exp()           # includes the noise floor where the model has one
    zz = M["off"][0, a, j] + scale * sdf[0, a, j] * torch.randn(n, 3, generator=gen)
    return from_frame(zz, M["p"][0, a, j], M["yaw"][0, a, j], M["sz"][0, a, j])


@torch.no_grad()
def end_resamples(B, m, P, Y, a, path, n, gen):
    """n plans: the evaluated path re-anchored (linearly, as the end candidates of feedback_rounds) to ends sampled from the model's T1 end mixture at
    ESC x its spread (t1_sample); None if the model has no T1 mixture."""
    if n <= 0 or not hasattr(m, "t1_mix"):
        return None
    E = encode(m, _batch(B, P, Y))
    ends = t1_sample(m, E, a, n, gen)[None]
    H = float(B.Z["extents"][a, 2]); lin = np.linspace(0, 1, path.shape[0])[:, None]
    out = []
    for k in range(n):
        G = ends[0, k].numpy().astype(np.float64); G[2] = max(G[2] - H, 0.0)
        out.append((path + lin * (G - path[-1])).astype(np.float32))
    return out


def masked_logp(m, E, I, taken, objects_only=False):
    """Models without phys_loss: log p over the I objects + STOP (no STOP option: -1e4), carried objects masked; objects_only: STOP is not a choice."""
    ml, _ = m.pointers(E, torch.zeros(E["obj"].shape[0], dtype=torch.long))
    sl = m.stop_logit(E).float()[:, None] if getattr(m, "use_stop", False) else torch.full((ml.shape[0], 1), -1e4)
    lp = torch.cat([ml.float()[:, :I], sl], -1)[0]
    mask = torch.cat([torch.as_tensor(np.asarray(taken), dtype=torch.bool), torch.tensor([bool(objects_only)])])
    return F.log_softmax(lp.masked_fill(mask, -1e4), -1)


def propose(models, states, a_, rng, gens):
    """Per seed and state: C choice candidates (each its evaluated plan) + the pick's path candidates -> (rows, groups)."""
    from cleantables.worldmodel.physics.chains import median_plans
    rows, groups = [], []
    for si, s in enumerate(states):
        for sd, m in enumerate(models):
            B = bridge(s["ep"], [m]); I = B.I
            if is_new(m):
                p, paths, dys, _ = read_out(B, s["P"], s["Y"], s["carried"], EVAL_SEED, K_MED)
            else:
                E = encode(m, _batch(B, s["P"], s["Y"]))
                with torch.no_grad():
                    p = masked_logp(m, E, I, s["carried"], objects_only=True)[:I].exp().double().numpy()
            p = np.where(s["carried"], 0.0, np.nan_to_num(p))
            ok = np.flatnonzero(p > 1e-6)
            if not len(ok):
                continue
            g_ = np.log(p[ok]) / a_.temp + rng.gumbel(size=len(ok))              # tempered Gumbel top-k: C different candidates
            cand = [int(c) for c in ok[np.argsort(-g_)[:min(a_.cand, len(ok))]]]
            pick = cand[0]
            G = dict(si=si, seed=sd, ep=s["ep"], I=I, cand=cand, choice=[], path=[], sp=None)
            add = lambda pl, dy, kind, a=pick: (rows.append(dict(si=si, seed=sd, a=a, plan=np.asarray(pl, np.float32), dyaw=float(dy), kind=kind)), len(rows) - 1)[1]   # noqa: E731
            if is_new(m):
                G["path"].append(add(paths[pick], dys[pick], ("path", 0)))
                for k in range(1, a_.k_path):                                      # model's own sampler, fresh noise
                    _, pk, dk, _ = read_out(B, s["P"], s["Y"], s["carried"], int(torch.randint(0, 2**31 - 1, (1,), generator=gens[sd])), 1 if a_.diverse else K_MED)
                    G["path"].append(add(pk[pick], dk[pick], ("path", k)))
                ends = end_resamples(B, m, s["P"], s["Y"], pick, paths[pick], a_.k_end, gens[sd]) if a_.k_end > 0 else None
                for k in range(a_.k_end):
                    if ends is not None:
                        G["path"].append(add(ends[k], dys[pick], ("end", k)))
                    else:
                        _, pk, dk, _ = read_out(B, s["P"], s["Y"], s["carried"], int(torch.randint(0, 2**31 - 1, (1,), generator=gens[sd])), 1 if a_.diverse else K_MED)
                        G["path"].append(add(pk[pick], dk[pick], ("path", a_.k_path + k)))
                G["choice"].append(G["path"][0])                                  # the pick's evaluated plan serves both groups
                for c in cand[1:]:
                    G["choice"].append(add(paths[c], dys[c], ("choice", 0), c))
            else:                                                                  # models without phys_loss: candidates as in physics.chains
                V2["diverse"] = bool(a_.diverse)
                sp = median_plans(B, m, s["P"], s["Y"], pick, a_.k_path, a_.k_end, gens[sd])
                sp.update(ep=s["ep"], P0=s["P"], Y0=s["Y"], allowed=s["carried"])
                for k in range(len(sp["plans"])):
                    G["path"].append(add(sp["plans"][k].numpy(), sp["dyaw"][k], sp["kinds"][k]))
                G["sp"] = sp
                G["choice"].append(G["path"][0])
                for c in cand[1:]:
                    pl, _, dy = PJ.followed(B, s["P"], c, yaw_off=s["Y"])
                    G["choice"].append(add(pl, dy, ("choice", 0), c))
            groups.append(G)
    return rows, groups


# ------------------------------------------------------------------ replay
def write_jobs(states, rows, path, traj_t, audit=False):
    """Single-carry PhysX chain jobs (chains.write_chain_jobs format + traj_t) for the rows not screened out (with --screen_audit
    all rows; screened ones keep their screen cost and PhysX only checks the screen)."""
    keep = [r for r in rows if audit or not r.get("screened")]
    twins = sorted({states[r["si"]]["ep"] for r in keep})
    n_max = max(len(states[r["si"]]["P"]) for r in keep)
    lay = np.zeros((len(keep), n_max, 4), np.float32); so = np.full((len(keep), 1), -1, np.int64)
    pa = np.zeros((len(keep), 1, 33, 3), np.float32); dy = np.zeros((len(keep), 1), np.float32)
    for j, r in enumerate(keep):
        s = states[r["si"]]; I = len(s["P"])
        lay[j, :I] = PJ.layout4(s["P"], s["Y"]); so[j, 0] = r["a"]; pa[j, 0] = r["plan"]; dy[j, 0] = r["dyaw"]; r["j"] = j
    np.savez_compressed(path, twins=np.array(twins), chain_twin=np.array([twins.index(states[r["si"]]["ep"]) for r in keep]), layout=lay,
                        step_obj=so, path=pa, dyaw=dy, traj_t=np.asarray(traj_t, np.float32))
    return len(keep)


def screen_rows(states, rows, scr, lf):
    """CPU pre-screen (screen.Screen) per state and mover: clear hits are not replayed."""
    by = {}
    for k, r in enumerate(rows):
        by.setdefault((r["si"], r["a"]), []).append(k)
    for (si, a), ks in sorted(by.items()):
        s = states[si]; B = any_bridge(s["ep"])
        lb, sc = scr.check(si, B.S, s["P"], s["Y"], a, np.stack([rows[k]["plan"] for k in ks]), np.array([rows[k]["dyaw"] for k in ks]))
        for k, l_, c_ in zip(ks, lb, sc):
            rows[k]["hit_lb"] = float(l_); rows[k]["screened"] = bool(c_)
    n = sum(r.get("screened", False) for r in rows)
    log(f"   pre-screen: {n} of {len(rows)} candidates obviously hit (inside another object for >= {scr.frac:.0%} of the carry): not replayed", lf)


def carry_cost(B, S, j, a, path, start, allowed, graded=True):
    """Cost of replayed single carry j (see module docstring) -> (cost or NaN, parts)."""
    I = B.I
    if not np.isfinite(S["pre_unstable"][j]) or S["pre_unstable"][j] > 0 or not np.isfinite(S["pose_end"][j][:I]).all():
        return np.nan, dict(pre_unstable=True)
    p0, p1 = S["pose_start"][j][:I], S["pose_end"][j][:I]
    ext = np.ones(I, bool); ext[a] = False; ext &= ~np.asarray(allowed, bool)[:I]
    disp = np.linalg.norm(p1[:, :3] - p0[:, :3], axis=-1); tl = np.array([_tilt(p0[k, 3:7], p1[k, 3:7]) for k in range(I)])
    n_disp = int(((disp > DISP_M) & ext).sum()); n_top = int(((tl > TILT_DEG) & ext).sum())
    if graded and "hit_g" in S.files:
        hit = float(np.nan_to_num(S["hit_g"][j, 0]))
    elif "f_any" in S.files:
        fa, cl = float(S["f_any"][j, 0]), float(S["clear_cm"][j, 0]) / 100.0
        hit = 0.0 if not np.isfinite(fa) else (1.0 if fa > 0 else float(np.clip((ZONE_M - cl) / ZONE_M, 0.0, 1.0)))
    else:
        hit = 0.0
    pth = np.asarray(path)
    L = np.linalg.norm(np.diff(pth[:, :2], axis=0), axis=-1).sum(); Sx = max(float(np.linalg.norm(pth[-1, :2] - np.asarray(start)[:2])), 0.02)
    econ = max(0.0, L / Sx - 1)
    settle = W["settle_cm"] * float(np.nan_to_num(S["settle_cm"][j, 0])) + W["settle_deg"] * float(np.nan_to_num(S["settle_deg"][j, 0]))
    tb = B.S["table"]
    off = int(sum(not (tb["x"][0] <= p1[k, 0] <= tb["x"][1] and tb["y"][0] <= p1[k, 1] <= tb["y"][1]) or p1[k, 2] < -0.05 for k in range(I)))
    c = W["hit"] * hit + W["disp"] * n_disp + W["topple"] * n_top + W["econ"] * econ + W["off"] * off + settle
    return float(c), dict(hit=hit, n_disp=n_disp, n_topple=n_top, econ=econ, off=off, settle=settle,
                          contact=bool(np.nan_to_num(S["f_any"][j, 0]) > 0) if "f_any" in S.files else False,
                          f_ext=float(np.nan_to_num(S["f_ext"][j, 0])), gap_cm=float(np.nan_to_num(S["gap_cm"][j, 0])))   # f_ext, gap: reported only


def score(states, rows, groups, S, graded):
    """Costs per row (replayed: carry_cost; screened: module docstring); per group the cost arrays."""
    for r in rows:
        if "j" not in r and not r.get("screened"):
            continue
        s = states[r["si"]]; B = any_bridge(s["ep"])
        if r.get("screened"):
            r["cost"], r["parts"] = np.nan, dict(screened=True, hit_lb=r["hit_lb"])
            if "j" in r:                                                           # --screen_audit: what PhysX says about it
                r["audit"] = carry_cost(B, S, r["j"], r["a"], r["plan"], s["P"][r["a"]], s["carried"], graded)[1]
            continue
        r["cost"], r["parts"] = carry_cost(B, S, r["j"], r["a"], r["plan"], s["P"][r["a"]], s["carried"], graded)
    for G in groups:
        for key in ("choice", "path"):
            c = np.array([rows[k].get("cost", np.nan) for k in G[key]], np.float64)
            fin = np.isfinite(c); worst = float(c[fin].max()) if fin.any() else 0.0
            for q, k in enumerate(G[key]):
                if rows[k].get("screened"):
                    c[q] = max(W["hit"] * rows[k]["hit_lb"], worst)
            G[f"cost_{key}"] = c


def interventions(states, rows, S, out_npz, rnd):
    """T4: every valid replayed carry -> one sample, written to out_npz:
      ep, I, xyz (24, 3) / yaw (24,)  the physical scene before the carry (pose_start after the 0.6 s pre-settle; root, yaw offset)
      mover, cmd_path (32, 3)         the intervention: the carried object and its commanded path (top centre, world metres)
      exec_end (3,)                   where the carried object came to rest (top centre)
      traj (24, H, 3), traj_q (24, H, 4), traj_ok (24, H)   every object's displacement (top centre, world metres, relative to its
                                      pre-carry top centre) / quaternion at the T2 times traj_t (H,); m.phys_loss's exec_traj
      moved (24,)                     displaced > 1 cm or tilted > 30 deg (1 / 0; -1 = padding)   [reports]
      cost, seed, kind, row           the carry's cost, the proposing seed (-1 = round 0's ensemble), candidate kind, row index
    -> number of samples; row["t4"] = its index."""
    from cleantables.worldmodel.schema import MAXO
    TT = S["traj_t"] if "traj_t" in S.files else np.zeros(0, np.float32); H = len(TT)
    out = {k: [] for k in ("ep", "I", "xyz", "yaw", "mover", "cmd_path", "exec_end", "traj", "traj_q", "traj_ok", "moved", "cost", "seed", "kind", "row")}
    for q_, r in enumerate(rows):
        if "j" not in r:
            continue
        j = r["j"]; s = states[r["si"]]; B = any_bridge(s["ep"]); I = B.I
        if not (np.isfinite(S["pre_unstable"][j]) and S["pre_unstable"][j] == 0) or not np.isfinite(S["pose_end"][j][:I]).all():
            continue
        p0, p1 = S["pose_start"][j][:I], S["pose_end"][j][:I]
        xyz, yaw = state_of(B, S["pose_start"][j])
        X = np.zeros((MAXO, 3), np.float32); X[:I] = xyz; Yw = np.zeros(MAXO, np.float32); Yw[:I] = yaw
        Hm = float(B.Z["extents"][r["a"], 2])
        cmd = np.asarray(r["plan"][:32], np.float32).copy(); cmd[:, 2] += Hm
        tr = np.zeros((MAXO, H, 3), np.float32); tq = np.zeros((MAXO, H, 4), np.float32); ok = np.zeros((MAXO, H), bool)
        if H and "traj" in S.files:
            T7 = S["traj"][j][:I]                                                 # (I, H, 7)
            tops = top_of(B, np.transpose(T7, (1, 0, 2))) - top_of(B, p0[None])    # (H, I, 3) displacement from the pre-carry top
            tr[:I] = np.nan_to_num(np.transpose(tops, (1, 0, 2))); tq[:I] = np.nan_to_num(T7[..., 3:7])
            ok[:I] = np.isfinite(T7).all(-1)
        mv = (np.linalg.norm(p1[:, :3] - p0[:, :3], axis=-1) > DISP_M) | (np.array([_tilt(p0[k, 3:7], p1[k, 3:7]) for k in range(I)]) > TILT_DEG)
        y = np.full(MAXO, -1, np.int8); y[:I] = mv
        for k, v in (("ep", s["ep"]), ("I", I), ("xyz", X), ("yaw", Yw), ("mover", r["a"]), ("cmd_path", cmd), ("exec_end", top_of(B, p1[None])[0, r["a"]]),
                     ("traj", tr), ("traj_q", tq), ("traj_ok", ok), ("moved", y), ("cost", float(r.get("cost", np.nan))), ("seed", r["seed"]),
                     ("kind", str(r["kind"][0])), ("row", q_)):
            out[k].append(v)
        r["t4"] = len(out["ep"]) - 1
    np.savez_compressed(out_npz, round=rnd, traj_t=np.asarray(TT, np.float32), **{k: np.array(v) for k, v in out.items()})
    return len(out["ep"])


def load_t4(d, upto):
    """Every stored T4 sample of rounds <= upto (dict of arrays + 'round')."""
    Z = [np.load(f) for f in (d / f"round{r}.npz" for r in range(upto + 1)) if f.exists()]
    Z = [z for z in Z if len(z["ep"])]
    if not Z:
        return None
    D = {k: np.concatenate([z[k] for z in Z]) for k in ("ep", "I", "xyz", "yaw", "mover", "cmd_path", "exec_end", "traj", "traj_ok", "moved", "cost")}
    D["round"] = np.concatenate([np.full(len(z["ep"]), int(z["round"])) for z in Z])
    return D


# ------------------------------------------------------------------ geometric cost (--geo)
def geo_score_rows(states, rows, geo, lf):
    """Every candidate row -> cost = geo_cost.GeoCost (swept rotation-invariant cylinder vs the others' voxel occupancy + 3D economy);
    deterministic, CPU; no replay, no screen. A missing mesh occupancy -> NaN (weight 0)."""
    by = {}
    for k, r in enumerate(rows):
        by.setdefault((r["si"], r["a"]), []).append(k)
    for (si, a), ks in sorted(by.items()):
        s = states[si]; B = any_bridge(s["ep"])
        res = geo.score(si, B.S, s["P"], s["Y"], a, np.stack([rows[k]["plan"] for k in ks]), dyaws=np.array([rows[k]["dyaw"] for k in ks]))
        for q, k in enumerate(ks):
            rows[k]["cost"] = float(res["cost"][q]) if res["ok"][q] else np.nan
            rows[k]["parts"] = dict(geo=True, hit=float(res["geo_hit"][q]), econ3d=float(res["econ3d"][q]), drag=float(res["geo_drag"][q]))
    h = [r["parts"]["hit"] for r in rows if np.isfinite(r.get("cost", np.nan))]
    dr = [r["parts"]["drag"] for r in rows if np.isfinite(r.get("cost", np.nan))]
    log(f"   geometric cost: {len(h)} of {len(rows)} candidates scored; geo hit mean {np.mean(h) if h else float('nan'):.2f}, "
        f">= 0.5 in {np.mean(np.array(h) >= 0.5) if h else float('nan'):.2f}; drag mean {np.mean(dr) if dr else float('nan'):.2f}", lf)


def validate_rows(states, rows, groups, rd, a_, lf, replay):
    """WM_ROUTE_VALID: every optimised path candidate replayed in one PhysX job (each in its own state). Per row: v_hit (graded),
    v_push_sum / v_push_max (cm) / v_push_n1 (> 1 cm) over every other object (pose_start = before the carry -> pose_end = settled after release),
    valid_w = 1[v_hit < 0.5] * exp(-v_push_sum / PUSH_LAMBDA_CM), valid = valid_w > 0. The replay index is kept as "vj" (not "j": the
    picks' replay below owns "j")."""
    ks = sorted({k for G in groups for k in G["path"]}); vr = [rows[k] for k in ks]
    write_jobs(states, vr, rd / "valid.npz", a_.traj_t)
    S = replay(rd / "valid.npz", rd / "valid_scores.npz", a_.copies)
    for r in vr:
        j = r.pop("j"); r["vj"] = j; I = len(states[r["si"]]["P"])
        p0, p1 = S["pose_start"][j][:I], S["pose_end"][j][:I]
        ok = np.isfinite(p0[:, :3]).all(1) & np.isfinite(p1[:, :3]).all(1); ok[r["a"]] = False
        d = np.linalg.norm(p1[ok, :3] - p0[ok, :3], axis=1) if ok.any() else np.zeros(0)
        r["v_hit"] = float(np.nan_to_num(S["hit_g"][j, 0])) if "hit_g" in S.files else float("nan")
        r["v_push_sum"] = 100 * float(d.sum()); r["v_push_max"] = 100 * float(d.max()) if len(d) else 0.0; r["v_push_n1"] = int((d > 0.01).sum())
        ok_end = bool(np.isfinite(p1[r["a"], :3]).all())
        r["valid_w"] = float((r["v_hit"] < 0.5) * ok_end * np.exp(-r["v_push_sum"] / PUSH_LAMBDA_CM)); r["valid"] = r["valid_w"] > 0
    log(f"   validation: {len(vr)} optimised paths replayed; hit < 0.5 {np.mean([r['v_hit'] < 0.5 for r in vr]):.2f}; push sum "
        f"median {np.median([r['v_push_sum'] for r in vr]):.1f} cm, max median {np.median([r['v_push_max'] for r in vr]):.1f} cm; "
        f"mean weight {np.mean([r['valid_w'] for r in vr]):.2f}", lf)


def group_costs(groups, rows):
    for G in groups:
        for key in ("choice", "path"):
            G[f"cost_{key}"] = np.array([rows[k].get("cost", np.nan) for k in G[key]], np.float64)


def commanded(states, rows, traj_t):
    """Commanded outcome of every candidate (the geometric stage's RWR targets; phys_loss's inputs, the T4 store's layout): the
    carried object ends at its plan's end (top centre), its displacement known at the last T2 step only; every other object stays
    (displacement 0 at every step). row["cmd_i"] = its index. Real consequences (t4) come only from PhysX replays."""
    from cleantables.worldmodel.schema import MAXO
    H = len(traj_t)
    out = {k: [] for k in ("ep", "I", "xyz", "yaw", "mover", "cmd_path", "exec_end", "traj", "traj_ok")}
    for q_, r in enumerate(rows):
        s = states[r["si"]]; B = any_bridge(s["ep"]); I = B.I; a = r["a"]
        X = np.zeros((MAXO, 3), np.float32); X[:I] = s["P"]; Yw = np.zeros(MAXO, np.float32); Yw[:I] = s["Y"]
        Hm = float(B.Z["extents"][a, 2])
        pl = np.asarray(r["plan"], np.float32)
        cmd = pl[:32].copy(); cmd[:, 2] += Hm
        tr = np.zeros((MAXO, H, 3), np.float32); ok = np.zeros((MAXO, H), bool); ok[:I] = True
        tr[a, -1] = pl[-1] - np.asarray(s["P"][a], np.float32); ok[a, :-1] = False
        if PATH_ONLY:                                                          # the candidate path as a T2 trajectory of the carried object:
            sl_ = np.linspace(0, 1, len(pl)); uh = np.arange(1, H + 1) / H        # displacement at u = h / H (watch_path's inverse: fed back
            tr[a] = np.stack([np.interp(uh, sl_, pl[:, d]) - pl[0, d] for d in range(3)], -1)   # through watch_path it is this path)
            ok[:] = False; ok[a] = True                                        # only the carried object's shape is trained
        for k, v in (("ep", s["ep"]), ("I", I), ("xyz", X), ("yaw", Yw), ("mover", a), ("cmd_path", cmd),
                     ("exec_end", pl[-1] + np.array([0, 0, Hm], np.float32)), ("traj", tr), ("traj_ok", ok)):
            out[k].append(v)
        r["cmd_i"] = q_
    return {k: np.array(v) for k, v in out.items()}


def reuse_round0(src, sd, idir, models, lf):
    """--reuse: round 0 from an earlier run `src` (its physical pool and its stored round-0 / round-1 PhysX replays as T4), no new replay."""
    import shutil
    src = Path(src); rd = sd / "round0"; rd.mkdir(exist_ok=True)
    shutil.copy(src / "pool_r0.pt", sd / "pool_r0.pt")
    Z = [np.load(f) for f in sorted((src / "interventions").glob("round*.npz"))]
    keys = [k for k in Z[0].files if k not in ("round", "traj_t")]
    np.savez_compressed(idir / "round0.npz", round=0, traj_t=Z[0]["traj_t"], **{k: np.concatenate([z[k] for z in Z]) for k in keys})
    for k, m in enumerate(models):
        torch.save(m.state_dict(), rd / f"model_s{k}.pt")
    log(f"round 0 REUSED from {src}: pool {len(torch.load(sd / 'pool_r0.pt', weights_only=False))} states, "
        f"{sum(len(z['ep']) for z in Z)} stored PhysX T4 samples (rounds {[int(z['round']) for z in Z]})", lf)


# ------------------------------------------------------------------ updates
def trained_params(m):
    """Models with phys_loss: m.phys_params() if defined, else every parameter outside the trunk (TRUNK); others: OLD_TRAINED heads."""
    if not is_new(m):
        return [p for n, p in m.named_parameters() if n.split(".")[0] in OLD_TRAINED]
    if PATH_ONLY:                                                              # physics decides only how to carry: T1 / choice frozen
        return [p for n, p in m.named_parameters() if n.split(".")[0] in PATH_PARAMS]
    if hasattr(m, "phys_params"):
        return list(m.phys_params())
    return [p for n, p in m.named_parameters() if n.split(".")[0] not in TRUNK]


def freeze(m):
    keep = {id(p) for p in trained_params(m)}
    for p in m.parameters():
        p.requires_grad_(id(p) in keep)


def t4_batch(D, idx):
    """T4 rows idx (all of one twin) -> phys_loss inputs (b is built by the caller's bridge)."""
    I = int(D["I"][idx[0]])
    return dict(xyz=D["xyz"][idx, :I], yaw=D["yaw"][idx, :I], mv=torch.as_tensor(D["mover"][idx]).long(),
                cmd=torch.as_tensor(D["cmd_path"][idx]), end=torch.as_tensor(D["exec_end"][idx]),
                traj=torch.as_tensor(D["traj"][idx]), ok=torch.as_tensor(D["traj_ok"][idx]))


def can_sample(ref):
    return hasattr(ref, "phys_sample") or (getattr(ref, "watch", False) and hasattr(ref, "sample_traj"))


@torch.no_grad()
def ref_sample(ref, b, mv, cmd):
    """Frozen reference's own outcome for the same intervention, the regulariser's targets (like the alpha_end NLL of reference samples
    in feedback_rounds and the W2 term of ORW-CFM-W2, in phys_loss units): end (B, 3) its T1 mode of the carried object, traj (B, 24, H, 3)
    one T2 sample with the intervention token (the t4 target), traj0 the same without it (the choice target), ok (B, 24, H) = the real object slots."""
    if hasattr(ref, "phys_sample"):
        e, t_ = ref.phys_sample(b, mv, cmd)
        return e, t_, t_, b["obj_mask"].bool()[..., None].expand(*t_.shape[:3])
    from cleantables.worldmodel.base_model import PS
    Bn = len(mv); ar = torch.arange(Bn, device=mv.device); mode = torch.ones(Bn, dtype=torch.long, device=mv.device)
    E0 = ref.encode(b, mode); E0["kind"] = b["obj_kind"]
    end = ref.t1_mode(E0)[0][ar, mv]
    st = b["obj_geo"][ar, mv, :3].float()
    bi = dict(b); bi["iv_obj"] = mv; bi["iv_path"] = (cmd.float() - st[:, None]) / PS
    E1 = ref.encode(bi, mode); E1["kind"] = b["obj_kind"]
    tr = ref.sample_traj(E1, K=1)[:, 0]; tr0 = ref.sample_traj(E0, K=1)[:, 0]
    return end, tr, tr0, (b["obj_mask"].bool() & (b["obj_kind"] == 0))[..., None].expand(*tr.shape[:3])


def loss_parts(m, b, mv, cmd, end, traj, ok):
    """phys_loss split into dict(t1, choice, t4) per sample (UWMW: parts=True); a phys_loss without parts -> all of it as t1."""
    try:
        P = m.phys_loss(b, mv, cmd, end, traj, ok, parts=True)
    except TypeError:
        P = None
    if not isinstance(P, dict):
        L = m.phys_loss(b, mv, cmd, end, traj, ok)
        return dict(t1=L, choice=torch.zeros_like(L), t4=torch.zeros_like(L))
    return P


def phys_update(m, ref, groups, rows, D, off, cfg, steps, rng, lf, Dc=None):
    """Update for models with phys_loss, per group of the round (a state's choice candidates; the pick's path candidates):
        sum_k w_k (t1_k + choice_k) / n_group      RWR: the executed end (T1) and the executed displacements without the intervention
                                                   (un-intervened T2 + d_h: the next-object read-out and the plan's own future)
      + mean_k t4_k over the same replays          unweighted: the intervened T2 (consequences), so the consequence model is not
                                                   biased toward good outcomes (every replayed row of the group, w = 0 included)
      [--t4 uniform: + the mean t4 of every stored replay (all rounds, round 0 included), unweighted]
      + regulariser: alpha x (t1 + t4 + choice) of the reference's own outcome (ref_sample: its T1 mode, its intervened T2 sample for t4,
        its un-intervened T2 sample for choice), else beta_p x ||theta - theta_ref||^2.
    off: the index of this round's first T4 sample in D (rows' "t4" index the round's own file).
    Dc (--geo): the RWR (t1 + choice) runs on every candidate's commanded outcome (commanded(); rows' "cmd_i"), weighted by its geometric
    cost; the unweighted t4 term uses the group's real PhysX replays only (rows with "t4": each seed's own pick)."""
    dev = torch.device(os.environ.get("WM_PHYS_DEV", "cpu"))               # device for the update only (geometry stays on the CPU)
    m.to(dev); ref.to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    items, extra = [], []
    for G in groups:
        for key in (("path",) if PATH_ONLY else ("choice", "path")):         # path-only: the same object, the same goal, other paths
            if ROUTE_OPT and ROUTE_VALID:                                      # validated route-opt: only the paths that pass the PhysX replay
                w = np.array([float(rows[k].get("valid_w", 0.0)) for k in G[key]], np.float32)
            else:
                w = np.isfinite(G[f"cost_{key}"]).astype(np.float32) if ROUTE_OPT else weights(G[f"cost_{key}"], cfg["tau"])   # route-opt: distil every optimised path
            if Dc is not None:
                t4i = off + np.array([rows[k]["t4"] for k in G[key] if "t4" in rows[k]], np.int64)
                items.append((G["ep"], np.array([rows[k]["cmd_i"] for k in G[key]]), torch.as_tensor(w), len(G[key]), t4i))
                continue
            ks = [q for q, k in enumerate(G[key]) if "t4" in rows[G[key][q]]]
            if ks:
                items.append((G["ep"], off + np.array([rows[G[key][q]]["t4"] for q in ks]), torch.as_tensor(w[ks]), len(G[key]), None))
    if cfg["t4"] == "uniform" and D is not None:
        for ep in np.unique(D["ep"]):
            ix = np.flatnonzero(D["ep"] == ep)
            for c0 in range(0, len(ix), 16):
                extra.append((str(ep), ix[c0:c0 + 16]))
    if not (items or extra) or not params:
        m.to("cpu"); ref.to("cpu")
        log("   phys: no informative carries this round", lf); return
    ref_p = {id(p): q.detach().clone() for p, q in zip(m.parameters(), ref.parameters()) if p.requires_grad}
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)

    def inputs(ep, idx, src=None):
        x = t4_batch(D if src is None else src, idx)
        b = _batch(bridge(ep, [m]), x["xyz"], x["yaw"])
        mv_ = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}   # noqa: E731
        return mv_(x), mv_(b)
    for st in range(steps):
        mb = [items[j] for j in rng.integers(0, len(items), min(cfg["B_sc"], len(items)))] if items else []
        mx = [extra[j] for j in rng.integers(0, len(extra), min(cfg["B_sc"], len(extra)))] if extra else []
        l_rw = l_t4 = l_all = l_leash = 0.0
        for ep, idx, w, n, t4i in mb:
            x, b = inputs(ep, idx, Dc)
            P = loss_parts(m, b, x["mv"], x["cmd"], x["end"], x["traj"], x["ok"])
            w = w.to(dev)
            l_rw = l_rw + (w * (P["choice"] if PATH_ONLY else P["t1"] + P["choice"])).sum() / n / len(mb)
            if Dc is None:
                l_t4 = l_t4 + P["t4"].mean() / len(mb)
            elif len(t4i):                                                         # geo: consequences only from real replays
                x4, b4 = inputs(ep, t4i)
                l_t4 = l_t4 + loss_parts(m, b4, x4["mv"], x4["cmd"], x4["end"], x4["traj"], x4["ok"])["t4"].mean() / len(mb)
            if can_sample(ref):                                                   # the reference's own outcome under the learner
                with torch.no_grad():
                    re, rt, rt0, rok = ref_sample(ref, b, x["mv"], x["cmd"])
                A_ = loss_parts(m, b, x["mv"], x["cmd"], re, rt, rok)            # t1 + t4 on the intervened sample,
                B_ = loss_parts(m, b, x["mv"], x["cmd"], re, rt0, rok) if rt0 is not rt else A_   # choice on the un-intervened one
                l_leash = l_leash + cfg["alpha"] * (A_["t1"] + A_["t4"] + B_["choice"]).mean() / len(mb)
        for ep, idx in mx:
            x, b = inputs(ep, idx)
            l_all = l_all + loss_parts(m, b, x["mv"], x["cmd"], x["end"], x["traj"], x["ok"])["t4"].mean() / len(mx)
        if not can_sample(ref):
            l_leash = cfg["beta_p"] * sum(((p - ref_p[id(p)]) ** 2).sum() for p in params)
        loss = l_rw + l_t4 + l_all + l_leash
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if st % 20 == 0 or st == steps - 1:
            f_ = lambda v: float(torch.as_tensor(v).detach())                    # noqa: E731
            log(f"   phys step {st} rwr(t1+choice) {f_(l_rw):.4f} t4 {f_(l_t4):.4f}" + (f" t4_all {f_(l_all):.4f}" if mx else "")
                + f" leash {f_(l_leash):.5f}", lf)
    m.to("cpu"); ref.to("cpu")                                                   # (read-outs / proposals stay on the CPU)


def choice_update(m, ref, states, groups, cfg, steps, rng, lf):
    """Models without phys_loss (as physics.chains): RWR on the object-only choice over a state's C candidates + beta KL(reference || model) on the
    joint softmax objects + STOP."""
    params = [p for n, p in m.named_parameters() if n.split(".")[0] == "mover_h" and p.requires_grad]
    items = []
    for G in groups:
        w = weights(G["cost_choice"], cfg["tau"])
        if w.sum() == 0:
            continue
        s = states[G["si"]]; E = encode(m, _batch(bridge(s["ep"], [m]), s["P"], s["Y"]))
        with torch.no_grad():
            lr_ = masked_logp(ref, E, G["I"], s["carried"], False)
        items.append((E, G["I"], s["carried"], torch.as_tensor(G["cand"]), torch.as_tensor(w), lr_))
    if not items or not params:
        log("   choice: no informative states this round", lf); return
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    for it in range(steps):
        mb = [items[j] for j in rng.integers(0, len(items), min(cfg["B_sc"], len(items)))]
        l_rw = l_kl = 0.0
        for E, I, taken, cand, w, lr_ in mb:
            l_rw = l_rw - (w * masked_logp(m, E, I, taken, True)[cand]).sum() / w.sum() / len(mb)
            lpj = masked_logp(m, E, I, taken, False)
            l_kl = l_kl + (lr_.exp() * (lr_ - lpj)).sum() / len(mb)
        loss = l_rw + cfg["beta"] * l_kl
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if it % 20 == 0 or it == steps - 1:
            log(f"   choice step {it} rwr {float(l_rw):.3f} kl {float(l_kl):.4f}", lf)


# ------------------------------------------------------------------ rounds
def traj_times(m):
    """T2 steps: 0.5 s apart, m.H of them (watch: 8 -> 0.5 ... 4.0 s after the carry starts)."""
    return [0.5 * (h + 1) for h in range(int(getattr(m, "H", 8)))]


def round0(models, states, rd, a_, replay, lf):
    """Pool / T4 pass: every object of every seed state carried once along the initial ensemble's evaluated plan."""
    rows = []
    for si, s in enumerate(states):
        B = bridge(s["ep"], models)
        if is_new(models[0]):
            _, paths, dys, _ = read_out(B, s["P"], s["Y"], s["carried"], EVAL_SEED, K_MED)
        for o in range(B.I):
            if is_new(models[0]):
                pl, dy = paths[o], float(dys[o])
            else:
                torch.manual_seed(1000 * si + o)
                pl, _, dy = PJ.followed(B, s["P"], o, yaw_off=s["Y"])
            if np.isfinite(pl).all():
                rows.append(dict(si=si, seed=-1, a=o, plan=np.asarray(pl, np.float32), dyaw=float(dy), kind=("pool", 0)))
    n = write_jobs(states, rows, rd / "carries.npz", a_.traj_t)
    log(f"[{a_.name}] round 0: {len(states)} seed states, {n} single carries (every object, the ensemble's evaluated plan)", lf)
    S = replay(rd / "carries.npz", rd / "carry_scores.npz", a_.copies)
    pool = []
    for si, s in enumerate(states):
        B = any_bridge(s["ep"])
        ok0 = [r for r in rows if r["si"] == si and np.isfinite(S["pre_unstable"][r["j"]]) and S["pre_unstable"][r["j"]] == 0]
        if ok0 and valid(B, S["pose_start"][ok0[0]["j"]]):
            P, Y = state_of(B, S["pose_start"][ok0[0]["j"]])
            pool.append(dict(ep=s["ep"], P=P, Y=Y, carried=np.zeros(B.I, bool), depth=0, src=s["src"]))
        for r in ok0:
            r["cost"], r["parts"] = carry_cost(B, S, r["j"], r["a"], r["plan"], s["P"][r["a"]], s["carried"], a_.graded)
            if valid(B, S["pose_end"][r["j"]]):
                P, Y = state_of(B, S["pose_end"][r["j"]]); cr = np.zeros(B.I, bool); cr[r["a"]] = True
                pool.append(dict(ep=s["ep"], P=P, Y=Y, carried=cr, depth=1, src=s["src"] + "+1"))
    return rows, S, pool


def run(a_, models, refs, sd, replay=None, eps=None, dr_names=()):
    """Round 0 (pool + T4), then rounds 1..a_.rounds: propose (every seed), screen, one replay, score, T4 store, update every seed."""
    from cleantables.worldmodel.physics.screen import Screen, ensure_occ
    from cleantables.worldmodel.physics.geo_cost import GeoCost
    replay = replay or run_chain_physx
    GEO = bool(getattr(a_, "geo", False))
    if GEO and getattr(a_, "geo_shape", "cyl") == "mesh":                       # MeshCost: each object's own shape
        from cleantables.worldmodel.physics.geo_cost import MeshCost
        geo = MeshCost(w_drag=float(getattr(a_, "geo_drag", 10.0)))
    else:
        geo = GeoCost(econ=getattr(a_, "geo_econ", "3d"), w_drag=float(getattr(a_, "geo_drag", 0.0))) if GEO else None
    lf = sd / "log.txt"; idir = sd / "interventions"; idir.mkdir(parents=True, exist_ok=True)
    pool = None
    for r in range(a_.rounds + 1):
        rd = sd / f"round{r}"; rd.mkdir(exist_ok=True)
        if all((rd / f"model_s{k}.pt").exists() for k in range(len(models))) and (sd / f"pool_r{r}.pt").exists():
            for k, m in enumerate(models):
                m.load_state_dict(torch.load(rd / f"model_s{k}.pt"))
            pool = torch.load(sd / f"pool_r{r}.pt", weights_only=False); continue
        while (sd / "PAUSE").exists():                                              # pause between rounds while a PAUSE file exists
            time.sleep(60)
        t0 = time.time()
        rng = np.random.default_rng(int(a_.base_seed) * 1000 + r)
        for m in models:
            m.eval()
        if r == 0 and GEO and getattr(a_, "reuse", ""):
            reuse_round0(a_.reuse, sd, idir, models, lf)
            pool = torch.load(sd / "pool_r0.pt", weights_only=False); continue
        Dc = None
        if r == 0:
            states = seed_states(models, a_.pool_states, rng, eps, dr_names, a_.dr_frac)
            torch.manual_seed(int(a_.base_seed))
            rows, S, pool = round0(models, states, rd, a_, replay, lf)
            groups = []
        else:
            if (rd / "cand.pt").exists():
                C_ = torch.load(rd / "cand.pt", weights_only=False); states, rows, groups = C_["states"], C_["rows"], C_["groups"]
            else:
                states = pick_states(pool, a_.states, rng)
                gens = [torch.Generator().manual_seed(int(a_.base_seed) * 1000 + 97 * r + k) for k in range(len(models))]
                torch.manual_seed(int(a_.base_seed) * 1000 + r)
                rows, groups = propose(models, states, a_, rng, gens)
                if not a_.no_screen and not GEO:
                    if ensure_occ({str(any_bridge(s["ep"]).S["objects"][k]["usd"]) for s in states for k in range(len(s["P"]))}):
                        screen_rows(states, rows, Screen(a_.screen_frac), lf)
                    else:
                        log("   pre-screen OFF this round (occupancy cache incomplete)", lf)
                torch.save(dict(states=states, rows=rows, groups=groups), rd / "cand.pt")
            if GEO:                                                                # every candidate: the geometric cost; PhysX replays
                ensure_occ({str(any_bridge(s["ep"]).S["objects"][k]["usd"]) for s in states for k in range(len(s["P"]))})
                geo_score_rows(states, rows, geo, lf)
                if ROUTE_OPT:                                                      # every path candidate improved by gradient steps on
                    from cleantables.worldmodel.physics.route_opt import optimise_rows                           # its own points (the T1 goal fixed), then re-scored exactly
                    optimise_rows(states, rows, groups, geo, ROUTE_OPT, any_bridge, lf)
                    geo_score_rows(states, rows, geo, lf)
                    if ROUTE_VALID:
                        validate_rows(states, rows, groups, rd, a_, lf, replay)
                group_costs(groups, rows)                                          # only each seed's own pick (T4 + the pool)
                Dc = commanded(states, rows, a_.traj_t)
                picks = [rows[G["path"][0]] for G in groups]
                n = write_jobs(states, picks, rd / "carries.npz", a_.traj_t, audit=True)
                log(f"[{a_.name}] round {r}: {len(states)} states x {len(models)} seeds -> {len(rows)} candidates scored geometrically, "
                    f"{n} picks replayed in ONE PhysX job ({time.time() - t0:.0f} s)", lf)
                S = replay(rd / "carries.npz", rd / "carry_scores.npz", a_.copies)
                ph = [float(np.nan_to_num(S["hit_g"][x["j"], 0])) for x in picks if "hit_g" in S.files]
                log(f"   picks: geo hit {np.nanmean([x['parts']['hit'] for x in picks]):.2f}, PhysX graded hit {np.mean(ph) if ph else float('nan'):.2f}", lf)
            if not GEO:
                aud = bool(a_.screen_audit) and r == 1
                n = write_jobs(states, rows, rd / "carries.npz", a_.traj_t, audit=aud)
                log(f"[{a_.name}] round {r}: {len(states)} states x {len(models)} seeds -> {len(rows)} candidates, {n} replayed in ONE PhysX job "
                    f"({time.time() - t0:.0f} s)", lf)
                S = replay(rd / "carries.npz", rd / "carry_scores.npz", a_.copies)
                score(states, rows, groups, S, a_.graded)
            au = [(x["hit_lb"], x["audit"]) for x in rows if "audit" in x and "hit" in x["audit"]]
            if au:
                log(f"   screen audit ({len(au)} screened carries replayed anyway): PhysX graded hit >= hit_lb in "
                    f"{np.mean([p_['hit'] >= l_ for l_, p_ in au]):.2f}, >= 0.5 in {np.mean([p_['hit'] >= 0.5 for _, p_ in au]):.2f}; "
                    f"PhysX hit {np.mean([p_['hit'] for _, p_ in au]):.2f} vs hit_lb {np.mean([l_ for l_, _ in au]):.2f}", lf)
        P_ = [x["parts"] for x in rows if "n_disp" in x.get("parts", {})]
        if P_:
            Cc = np.array([x.get("cost", np.nan) for x in rows if "j" in x and not x.get("screened")], np.float64)
            log(f"[{a_.name}] round {r}: mean cost {np.nanmean(Cc):.3f} (finite {np.isfinite(Cc).mean():.2f}); graded hit {np.mean([p['hit'] for p in P_]):.2f}, "
                f"contact {np.mean([p['contact'] for p in P_]):.2f}, others displaced {np.mean([p['n_disp'] for p in P_]):.2f} / toppled "
                f"{np.mean([p['n_topple'] for p in P_]):.2f} per carry, off {np.mean([p['off'] for p in P_]):.2f}, econ {np.mean([p['econ'] for p in P_]):.2f}", lf)
        n_t4 = interventions(states, rows, S, idir / f"round{r}.npz", r)
        torch.save(dict(states=states, rows=rows, groups=groups), rd / "scored.pt")
        D = load_t4(idir, r)
        log(f"[{a_.name}] round {r}: {n_t4} T4 samples stored ({0 if D is None else len(D['ep'])} so far)", lf)
        for k, (m, ref) in enumerate(zip(models, refs)):
            rr = np.random.default_rng(int(a_.base_seed) * 1000 + 53 * r + k); g_ = torch.Generator().manual_seed(int(a_.base_seed) * 1000 + 31 * r + k)
            cfg = dict(lr=a_.lr, tau=a_.tau, beta=a_.beta, alpha=a_.alpha, alpha_end=a_.alpha_end, w_yaw=a_.w_yaw, B_sc=a_.B_sc, beta_p=a_.beta_p, t4=a_.t4)
            mine = [G for G in groups if G["seed"] == k]
            if is_new(m):
                if mine or (a_.t4 == "uniform" and D is not None):
                    log(f"  seed {a_.seeds[k]}: {len(mine)} states", lf)
                    phys_update(m, ref, mine, rows, D, int((D["round"] < r).sum()) if D is not None else 0, cfg, a_.steps, rr, lf, Dc=Dc)
            elif mine:                                                             # models without phys_loss: older updates
                from cleantables.worldmodel.physics.feedback_rounds import update as path_update
                log(f"  seed {a_.seeds[k]}: {len(mine)} states (legacy heads)", lf)
                choice_update(m, ref, states, mine, cfg, a_.steps, rr, lf)
                path_update(m, ref, [G["sp"] for G in mine], [dict(cost=G["cost_path"]) for G in mine], cfg, a_.steps, g_, lf, end=True)
            torch.save(m.state_dict(), rd / f"model_s{k}.pt")
        if r > 0:                                                                  # the pool grows by every seed's own pick, physically
            for G in groups:
                x = rows[G["path"][0]]
                if x.get("screened") or "j" not in x or not np.isfinite(x.get("cost", np.nan)):
                    continue
                s = states[G["si"]]; B = any_bridge(s["ep"])
                if s["depth"] + 1 < MAX_STEPS and valid(B, S["pose_end"][x["j"]]):
                    P, Y = state_of(B, S["pose_end"][x["j"]]); cr = s["carried"].copy(); cr[x["a"]] = True
                    pool.append(dict(ep=s["ep"], P=P, Y=Y, carried=cr, depth=s["depth"] + 1, src=s["src"].split("+")[0] + f"+{s['depth'] + 1}"))
        torch.save(pool, sd / f"pool_r{r}.pt")
        log(f"[{a_.name}] round {r} done: pool {len(pool)} states ({time.time() - t0:.0f} s)", lf)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--init", required=True); ap.add_argument("--ref", default="")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--states", type=int, default=48); ap.add_argument("--pool_states", type=int, default=96); ap.add_argument("--cand", type=int, default=3)
    ap.add_argument("--k_path", type=int, default=8); ap.add_argument("--k_end", type=int, default=4); ap.add_argument("--temp", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.5, help="old models: KL leash of the choice")
    ap.add_argument("--alpha", type=float, default=0.1, help="W2 / reference-sample leash"); ap.add_argument("--alpha_end", type=float, default=0.1)
    ap.add_argument("--beta_p", type=float, default=1e-3, help="new models without phys_sample: L2 leash to the reference's parameters")
    ap.add_argument("--w_yaw", type=float, default=0.5); ap.add_argument("--B_sc", type=int, default=8); ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--dr_list", default=""); ap.add_argument("--dr_frac", type=float, default=0.6)
    ap.add_argument("--traj_t", default="", help="T2 / T4 times (s after the carry start); default: the model's m.H steps of 0.5 s")
    ap.add_argument("--t4", default="rwr", choices=["rwr", "uniform"], help="uniform: also the t4 (consequence) term of every stored replay, unweighted")
    ap.add_argument("--graded", action="store_true", help="graded hit (physx_chain hit_g); else 0/1 contact + 1 cm safe-zone depth")
    ap.add_argument("--diverse", action="store_true", help="path candidates: 1 evaluated plan + single flow samples + ends at 2x spread")
    ap.add_argument("--no_screen", action="store_true"); ap.add_argument("--screen_frac", type=float, default=0.5)
    ap.add_argument("--screen_audit", action=argparse.BooleanOptionalAction, default=True,
                    help="round 1: replay the screened candidates too and log the agreement (on by default)")
    ap.add_argument("--base_seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--geo", action="store_true", help="the GEOMETRIC cost for every candidate (geo_cost.py, CONFIG §25); PhysX only for picks (T4 + pool)")
    ap.add_argument("--reuse", default="", help="--geo: round 0 (pool + stored PhysX T4) from this stopped run's folder")
    ap.add_argument("--geo_econ", default="3d", choices=["3d", "xy"], help="economy: 3D length (v1) or table-plane only (v2: lifting free)")
    ap.add_argument("--geo_drag", type=float, default=0.0, help="weight of the DRAG term (v2; CONFIG §28): 0 = off")
    ap.add_argument("--geo_shape", default="cyl", choices=["cyl", "mesh"], help="carried shape: rotation-invariant cylinder (v1/v2) or its own mesh vs SDFs (v3)")
    ap.add_argument("--route_opt", type=int, default=0, help="STEPS of route optimisation of every path candidate before the path-only distillation (§29c; implies --path_only)")
    ap.add_argument("--path_only", action="store_true", help="train ONLY the T2 flow decoder (+ iv_e); T1 / choice frozen; RWR on path groups only (§29b)")
    a = ap.parse_args()
    a.seeds = [int(x) for x in a.seeds.split(",")]
    global PATH_ONLY, ROUTE_OPT
    ROUTE_OPT = int(a.route_opt)
    PATH_ONLY = bool(a.path_only) or ROUTE_OPT > 0
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8"))); torch.manual_seed(a.base_seed)
    if a.smoke:
        a.rounds, a.states, a.pool_states, a.steps, a.copies, a.k_path, a.k_end = 1, 2, 2, 3, 2, min(a.k_path, 3), min(a.k_end, 1)
    if not a.smoke and (os.environ.get("WM_CARRY") != "force" or os.environ.get("WM_HITS") != "1"):
        raise SystemExit("phys_stage: the replay must be WM_CARRY=force WM_HITS=1")
    from cleantables.worldmodel.physics.choice import train_twins
    from cleantables.worldmodel.bridge import load_models
    src = RUNS2 / a.init if (RUNS2 / a.init).exists() else OUT / "wm_v2" / "runs" / a.init
    rsrc = (RUNS2 / a.ref if (RUNS2 / a.ref).exists() else OUT / "wm_v2" / "runs" / a.ref) if a.ref else src
    models = [load_models(src, seeds=(s,))[0] for s in a.seeds]
    if getattr(models[0], "endref", False) and not is_new(models[0]):
        raise SystemExit("phys_stage: an endref model without phys_loss (the interim clean model) has no physics update here")
    refs = [load_models(rsrc, seeds=(s,))[0].eval() for s in a.seeds] if a.ref else [copy.deepcopy(m).eval() for m in models]
    a.traj_t = [float(x) for x in a.traj_t.split(",")] if a.traj_t else traj_times(models[0])
    for m, ref in zip(models, refs):
        for p in ref.parameters():
            p.requires_grad_(False)
        freeze(m)
    sd = PF / a.name; sd.mkdir(parents=True, exist_ok=True)
    dr_names = json.load(open(a.dr_list))["names"] if a.dr_list else []
    json.dump(dict(vars(a), weights=W, esc=ESC, k_med=K_MED, w_max=W_MAX, max_steps=MAX_STEPS, new_model=is_new(models[0]),
                   trained=sorted({n.split(".")[0] for n, p in models[0].named_parameters() if p.requires_grad}),
                   replay=dict(carry=os.environ.get("WM_CARRY", "pose"), hits=os.environ.get("WM_HITS", "0"), table_in_hit_zone=False),
                   twin_root=os.environ.get("WM_TWIN_ROOT", ""), n_dr_twins=len(dr_names)), open(sd / "cfg.json", "w"), indent=1)
    run(a, models, refs, sd, eps=train_twins(), dr_names=dr_names)
    if not a.smoke:
        od = RUNS2 / a.name; od.mkdir(parents=True, exist_ok=True)
        for s, m in zip(a.seeds, models):
            torch.save(m.state_dict(), od / f"fold-1_seed{s}.pt")
            base = json.load(open(src / f"fold-1_seed{s}.json")); base["phys_stage"] = json.load(open(sd / "cfg.json"))
            json.dump(base, open(od / f"fold-1_seed{s}.json", "w"), indent=1)
    log(f"done {a.name} seeds {a.seeds}", sd / "log.txt")


if __name__ == "__main__":
    main()
