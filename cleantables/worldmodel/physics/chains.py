"""Physics-scored whole chains with a DONE judge.

Learns which orders break a task (a lid closed before the tools are in, stacking onto something that is moved later, STOP
before the table is done) from the physical consequences of the model's own chains. No task rules, no hand-built order and no
human labels at training time (labels are used for evaluation only).

  scenes   as in cleantables.worldmodel.physics.choice stage 2: training twins (human mid-task states, start, random, rolled)
           plus physics-settled domain-randomised layouts with distractors (dr_frac of the scenes)
  chains   C chains per scene, each sampled from the model's joint softmax (objects + STOP, tempered, placed objects masked as
           taken); the chosen object follows the model's path (physx_jobs.followed) and is placed kinematically at its end for the
           next choice, until STOP or MAX_STEPS; the whole chain is then replayed carry by carry in PhysX
           (cleantables.worldmodel.physx_chain)
  cost     objects the chain selects may move or settle while later items go in and are not penalised. Every other object
           (distractors, the container unless the chain carries it) is penalised if displaced, toppled or pushed at any point:
           CW["ext_disp"] n_disp_ext + CW["ext_topple"] n_topple_ext + log1p(f_ext) + log1p(f_hand_ext) per carry (contact on the
           carried object / hand proxy from non-selected objects only). Per carry: settle_cm / 5 + settle_deg / 30 (drop or slide
           after release), path economy, CW["move"]. Any object off the table at the end: CW["off"]. DONE: CW["done"] x
           (1 - judge(end state)). There is no end-at-destination term; end_check() reports it for evaluation only.
           Models with end-referenced or watch heads train with cleantables.worldmodel.physics.stage instead.
  judge    the frozen reference model (the initialisation) evaluated on the chain's physical end state: "stop" (its P(STOP)) or
           "progress" (its progress head), chosen before training with judge_check() on human end states. Being frozen, the
           learner cannot make it agree with itself.
  update   mover_h and stop_h only: reward-weighted regression over the C chains of a scene (Peters & Schaal, 2007;
           w = exp(tau A), A = within-scene advantage of -cost, capped). A chain's log-likelihood is the sum over its steps of
           log p(choice | state) in the masked joint softmax, plus beta * KL(frozen reference || model) per state.

  python -m cleantables.worldmodel.physics.chains --name NAME --seed 0 --init INIT_RUN [--rounds 5] [--smoke]
Out: WM_PF_DIR/<name>/seed<s>/round<r>/{chains.pt, chains.npz, scores.npz, model.pt}; final WM_RUNS2/<name>/fold-1_seed<s>.pt.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
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
TRAINED = ("mover_h", "stop_h")
MAX_STEPS = 8
DISP_M, TILT_DEG = 0.01, 30.0                                                  # same definitions as the PhysX replay scorer
CW = dict(ext_disp=1.0, ext_topple=2.0, f_ext=1.0, f_hand_ext=1.0, settle_cm=1 / 5, settle_deg=1 / 30, econ=0.5, move=0.25, off=5.0,
          done=3.0)                                                            # hand-set weights; sensitivity: halve / double each
# --v2cost: the carried object touching or passing within zone_m of any other object during 5-95% of its path (replay with WM_HITS)
# costs `hit` (1 per carry for contact, else the safe-zone depth); economy is measured in the table plane only; an object moved
# before its own carry counts as displaced
V2 = dict(on=False, zone_m=0.01, hit=10.0, graded=False, diverse=False)    # graded: hit grade 0-2; diverse: mixed candidate plans


# ------------------------------------------------------------------ chains sampled from the model
def _masked_joint(p_now, p_stop, temp):
    q = np.r_[p_now, p_stop].astype(np.float64)
    q = np.where(q > 1e-9, q, 0.0) ** (1.0 / temp)
    return q / q.sum() if q.sum() > 0 else None


def sampled_carry(B, m, P, Y, a, gen):
    """One plan sampled from the model's heads (as feedback_rounds.scene_plans: a flow path sample, an end sample from the end mixture,
    a yaw sample), with the path re-anchored to the sampled end. This is the sample the ORW-CFM-W2 / RWR update later weights."""
    from cleantables.worldmodel.physics.feedback_rounds import scene_plans
    sp = scene_plans(B, m, P, a, 1, 1, gen)
    H = float(B.Z["extents"][a, 2])
    if sp["d"] < B.I:
        end_top = sp["b"]["obj_geo"][0, sp["d"], :3] + 0.1 * sp["end_samples"][0]
    else:
        end_top = sp["b"]["obj_geo"][0, a, :3] + sp["end_samples"][0] * __import__("cleantables.worldmodel.base_model", fromlist=["PS"]).PS
    lin = torch.linspace(0, 1, 32)[:, None]
    ptop = sp["paths_top"][0] + lin * (end_top - sp["paths_top"][0, -1])
    proot = ptop.clone(); proot[:, 2] -= H
    path = torch.cat([proot, proot[-1:]], 0).numpy().astype(np.float32)
    dy = float(sp["dyaw"][-1])                                                    # yaw of the end sample (from the yaw head)
    learn = dict(b=sp["b"], i=a, tg=sp["tg"], path_top=ptop[None].clone(), end_sample=sp["end_samples"][:1].clone(),
                 yaw_sample=None if sp["yaw_samples"] is None else sp["yaw_samples"][:1].clone(), table=sp["table"])
    return path, (sp["d"] if sp["d"] < B.I else -1), dy, learn


def sample_chains(m, n_scenes, n_chain, seed, temp=1.5, dr_names=(), dr_frac=0.6, ref=None, paths=False):
    """Sample n_chain chains per scene. With ref (judge 'none'), STOP is sampled from the frozen reference's masked joint P(STOP) and
    the object from the learner's object-only distribution, so without a DONE judge the learner cannot learn to stop early."""
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel.physics.choice import human_states, movable_idx, roll_forward, train_twins
    from cleantables.worldmodel.bridge import WMBridge2
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(seed))
    eps = train_twins()
    bridges, scenes = {}, []
    tries = 0
    while len(scenes) < n_scenes and tries < 4 * n_scenes:
        tries += 1
        use_dr = len(dr_names) > 0 and rng.random() < dr_frac
        ep = dr_names[rng.integers(len(dr_names))] if use_dr else eps[rng.integers(len(eps))]
        if ep not in bridges:
            bridges[ep] = WMBridge2(ep, None, models=[m], device="cpu")
            if ref is not None:
                bridges[ep]._ref = WMBridge2(ep, None, models=[ref], device="cpu")
        B = bridges[ep]; Z = B.Z
        u = rng.random()
        src = ("dr" if u < 0.7 else "dr_rolled") if use_dr else ("start" if u < 0.4 else ("human" if u < 0.6 else ("random" if u < 0.85 else "rolled")))
        if use_dr or src == "start":
            P = Z["start_pos"].copy()
        elif src == "human":
            H_ = human_states(ep, Z, B.S)
            if not H_:
                continue
            P = H_[rng.integers(len(H_))].copy()
        else:
            P = W.shuffle_layout(Z, B.S, rng, movable_idx(Z))
        if P is None:
            continue
        Y = Z["start_yaw_off"].astype(np.float32).copy() if (use_dr and "start_yaw_off" in Z) else np.zeros(B.I, np.float32)
        if src.endswith("rolled"):
            P, Y = roll_forward(B, P, Y, int(rng.integers(1, 3)), rng)
        chains = []
        for _ in range(n_chain):
            Pc, Yc = P.astype(np.float32).copy(), Y.astype(np.float32).copy()
            taken = np.zeros(B.I, bool)
            steps = []
            for _s in range(MAX_STEPS + 1):
                o = B.plan_all(torch.as_tensor(Pc[None]), torch.as_tensor(Yc[None]), taken=taken[None], paths=False)
                q = _masked_joint(o["p_now"][0].numpy(), float(o["p_stop"][0]), temp)
                b = B._batch(torch.as_tensor(Pc[None]), torch.as_tensor(Yc[None]), None)
                st = dict(b=b, taken=taken.copy())
                if q is None or _s == MAX_STEPS:
                    st.update(choice=B.I, forced=True); steps.append(st); break
                if ref is not None:                                                # STOP from the reference, object from the learner
                    o_r = B._ref.plan_all(torch.as_tensor(Pc[None]), torch.as_tensor(Yc[None]), taken=taken[None], paths=False)
                    q_r = _masked_joint(o_r["p_now"][0].numpy(), float(o_r["p_stop"][0]), 1.0)
                    if q_r is None or rng.random() < q_r[-1]:
                        st.update(choice=B.I, forced=True); steps.append(st); break
                    q = q[:-1] / q[:-1].sum() if q[:-1].sum() > 0 else None
                    if q is None:
                        st.update(choice=B.I, forced=True); steps.append(st); break
                a = int(rng.choice(len(q), p=q))
                if a == B.I:                                                       # STOP
                    st.update(choice=a, forced=False); steps.append(st); break
                if paths:
                    path, d, dy, lrn = sampled_carry(B, m, Pc, Yc, a, gen)
                else:
                    path, d, dy = PJ.followed(B, Pc, a, yaw_off=Yc); lrn = None
                st.update(choice=a, forced=False, path=path.astype(np.float32), dest=int(d), dyaw=float(dy), start=Pc[a].copy(), learn=lrn)
                steps.append(st)
                Pc[a] = path[-1]; Yc[a] = Yc[a] + dy; taken[a] = True
            chains.append(dict(steps=steps, P_end_plan=Pc, Y_end_plan=Yc))
        scenes.append(dict(ep=ep, src=src, P0=P.astype(np.float32), Y0=Y.astype(np.float32), I=B.I, chains=chains))
    return scenes


def write_chain_jobs(scenes, path):
    flat = [(s, c) for s in scenes for c in s["chains"]]
    twins = sorted({s["ep"] for s in scenes})
    n_max = max(s["I"] for s in scenes)
    C = len(flat)
    lay = np.zeros((C, n_max, 4), np.float32); so = np.full((C, MAX_STEPS), -1, np.int64)
    pa = np.zeros((C, MAX_STEPS, 33, 3), np.float32); dy = np.zeros((C, MAX_STEPS), np.float32)
    for j, (s, c) in enumerate(flat):
        lay[j, :s["I"]] = PJ.layout4(s["P0"], s["Y0"])
        mv = [st for st in c["steps"] if st["choice"] < s["I"]]
        for k, st in enumerate(mv[:MAX_STEPS]):
            so[j, k] = st["choice"]; pa[j, k] = st["path"]; dy[j, k] = st["dyaw"]
    np.savez_compressed(path, twins=np.array(twins), chain_twin=np.array([twins.index(s["ep"]) for s, _ in flat]), layout=lay,
                        step_obj=so, path=pa, dyaw=dy)


def run_chain_physx(jobs, out, copies=16, timeout=4 * 3600, attempts=3):
    """Run cleantables.worldmodel.physx_chain as a subprocess, with the same broken-run checks and retries as physx_jobs._run."""
    if Path(out).exists():
        return np.load(out, allow_pickle=True)
    Jc = np.load(jobs)                                                            # envs = the most chains of any one scene (idle copies
    copies = int(min(copies, max(1, np.bincount(Jc["chain_twin"]).max())))        # only cost GPU time; the physics per env is the same)
    lf = Path(str(out) + ".log")
    for k in range(attempts):
        subprocess.run([str(ROOT / "envs/vla/bin/python"), "-m", "cleantables.worldmodel.gpu_wait", "--need_mib", "3500"], check=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT)))
        env = dict(os.environ, PYTHONPATH=str(ROOT)); env.pop("CUDA_VISIBLE_DEVICES", None)
        part = Path(str(out) + ".part.npz")
        t0 = time.time()
        with open(lf, "w") as f:
            r = subprocess.run([PJ.SIM_PY, "-m", "cleantables.worldmodel.physx_chain", "--jobs", str(jobs), "--out", str(part), "--copies", str(copies)],
                               cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        txt = lf.read_text(errors="ignore")
        bad = [p for p in PJ.BROKEN_PATTERNS if p in txt]
        if r.returncode == 0 and part.exists() and not bad and "PHYSX CHAIN DONE" in txt:
            part.rename(out); log(f"PhysX chains: done in {time.time() - t0:.0f} s")
            return np.load(out, allow_pickle=True)
        log(f"PhysX chain attempt {k + 1}/{attempts} discarded (code {r.returncode}; {', '.join(bad)}); log {lf}.attempt{k + 1}")
        lf.rename(str(lf) + f".attempt{k + 1}")
        if part.exists():
            part.unlink()
        time.sleep(60)
    raise RuntimeError(f"PhysX chain replay failed {attempts} times (see {lf}.attempt*)")


# ------------------------------------------------------------------ per-carry path scoring of evaluation-style plans (--carry_pf)
@torch.no_grad()
def median_plans(B, m, P0, Y0, i, K_path, K_end, gen, K_med=4):
    """Candidates in feedback_rounds.scene_plans format, each built like the evaluated plan (carry_check.eval_path, one member):
    the per-point median of K_med flow samples (+ the object's top centre), 5-point smoothed, re-anchored to the end. K_path such plans
    (fresh samples each) to the end head's mode, plus K_end plans: the median of all those samples re-anchored to ends sampled from
    the end mixture, with a sampled yaw."""
    from cleantables.worldmodel.base_model import PS
    from cleantables.worldmodel.physics.feedback_rounds import mix_sample, yaw_sample
    from cleantables.worldmodel.bridge import long_side_yaw, wrap_half
    I = B.I
    b = B._batch(torch.as_tensor(P0[None], dtype=torch.float32), torch.zeros(1, I), None)
    mode = torch.ones(1, dtype=torch.long); mv = torch.tensor([i])
    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
    _, tl = m.pointers(E, mv)
    d = int(F.softmax(torch.cat([tl[:, :I], tl[:, 24:25]], -1).float(), -1).argmax(-1)); tg = torch.tensor([d if d < I else 24])
    c, _, mdn = m.heads_from(E, mv, tg); zd = m.end_disp(c)
    H = float(B.Z["extents"][i, 2]); org = b["obj_geo"][0, i, :3].float()
    g_top = (b["obj_geo"][0, d, :3] + 0.1 * mdn["mu"].float()[0, int(mdn["logit"].float().argmax(-1))]) if d < I else \
        org + zd["mu"].float()[0, int(zd["logit"].float().argmax(-1))] * PS
    lin = torch.linspace(0, 1, 32)[:, None]

    def plan(end_top, raw):                                                       # as carry_check.eval_path
        p = raw.median(0).values + org
        p = F.avg_pool1d(p.T[None], 5, 1, 2, count_include_pad=False)[0].T.clone()
        p[:, 2] -= H
        G = end_top.clone(); G[2] = (G[2] - H).clamp(min=0.0)
        p = p + lin * (G - p[-1])
        top = p.clone(); top[:, 2] += H
        return torch.cat([p, G[None]], 0), top
    def single(end_top, raw):                                                     # a single flow sample, unsmoothed
        p = raw[0] + org; p = p.clone(); p[:, 2] -= H
        G = end_top.clone(); G[2] = (G[2] - H).clamp(min=0.0)
        p = p + lin * (G - p[-1])
        top = p.clone(); top[:, 2] += H
        return torch.cat([p, G[None]], 0), top
    if V2["diverse"]:                                                             # 1 evaluated plan + K_path - 1 single samples
        raws = [m.sample_paths_anchored(E, c, K=K_med)[0, :, :, :3].float()] + [m.sample_paths_anchored(E, c, K=1)[0, :, :, :3].float()
                                                                                for _ in range(K_path - 1)]
    else:
        raws = [m.sample_paths_anchored(E, c, K=K_med)[0, :, :, :3].float() for _ in range(K_path)]
    plans, tops, kinds = [], [], []
    for k, r in enumerate(raws):
        pr, tp = (single if (V2["diverse"] and k > 0) else plan)(g_top, r); plans.append(pr); tops.append(tp); kinds.append(("path", k))
    esc = 2.0 if V2["diverse"] else 1.0                                           # end mixture sampled at 2x its spread
    if d < I:
        e = mix_sample(mdn, K_end, gen, scale=esc); ends = b["obj_geo"][0, d, :3][None] + 0.1 * e
    else:
        e = mix_sample(zd, K_end, gen, scale=esc); ends = org[None] + e * PS
    allr = torch.cat(raws, 0)
    for k in range(K_end):
        pr, _ = plan(ends[k].float(), allr); plans.append(pr); kinds.append(("end", k))
    y0 = long_side_yaw(B.Z, i, float(Y0[i]))
    y = float(B.goal_yaw(torch.as_tensor(P0[None], dtype=torch.float32), torch.as_tensor(Y0[None]), [i], [d if d < I else -1])[0]) \
        if hasattr(B, "goal_yaw") else float("nan")
    dy_mode = wrap_half(y - y0) if np.isfinite(y) else 0.0
    ys = yaw_sample(m, c, K_end, gen) if getattr(m, "use_yaw", False) else None
    dys = [dy_mode] * K_path + ([wrap_half(float(v) - y0) for v in ys] if ys is not None else [0.0] * K_end)
    return dict(b={k: v for k, v in b.items()}, i=i, d=d, tg=tg, paths_top=torch.stack(tops), end_samples=e, yaw_samples=ys,
                plans=torch.stack(plans), kinds=kinds, dyaw=np.array(dys, np.float32), table=d >= I)


def carry_round(scenes, S, m, ref, rd, copies, rng, gen, k_path, k_end):
    """Per scene, one executed carry (random finite chain, random step) from its physical pre-carry state (replay pose_pre):
    k_path + k_end candidate plans (median_plans), each replayed in PhysX as a single carry with the same chain cost (objects the chain
    carried earlier may move) -> (candidate dicts for feedback_rounds.update, their scores)."""
    from cleantables.worldmodel.bridge import WMBridge2
    cand, meta = [], []
    j0 = 0
    for s in scenes:
        n_c = len(s["chains"])
        opts = [(ci, k_) for ci, ch in enumerate(s["chains"]) if np.isfinite(s["cost"][ci])
                for k_ in range(sum(st["choice"] < s["I"] for st in ch["steps"]))]
        if opts:
            ci, k_ = opts[rng.integers(len(opts))]
            jj = j0 + ci
            mv = [st for st in s["chains"][ci]["steps"] if st["choice"] < s["I"]]
            pose = S["pose_pre"][jj, k_, :s["I"]]
            if np.isfinite(pose).all():
                B = WMBridge2(s["ep"], None, models=[m], device="cpu")
                xyz = pose[:, :3].astype(np.float32)
                yaw = np.array([_yaw_off(pose[q, 3:7], B.S["objects"][q].get("yaw_baked", 0.0)) for q in range(s["I"])], np.float32)
                a = mv[k_]["choice"]
                allowed = np.zeros(s["I"], bool); allowed[[st["choice"] for st in mv[:k_]]] = True
                sp = median_plans(B, m, xyz, yaw, a, k_path, k_end, gen)
                sp.update(ep=s["ep"], P0=xyz, Y0=yaw, allowed=allowed)
                cand.append(sp); meta.append(B)
        j0 += n_c
    twins = sorted({c["ep"] for c in cand}); n_max = max(c["b"]["obj_geo"].shape[1] for c in cand)
    rows = [(c, k) for c in cand for k in range(len(c["plans"]))]
    lay = np.zeros((len(rows), n_max, 4), np.float32); so = np.full((len(rows), 1), -1, np.int64)
    pa = np.zeros((len(rows), 1, 33, 3), np.float32); dy = np.zeros((len(rows), 1), np.float32)
    for r_, (c, k) in enumerate(rows):
        I = len(c["P0"]); lay[r_, :I] = PJ.layout4(c["P0"], c["Y0"]); so[r_, 0] = c["i"]; pa[r_, 0] = c["plans"][k].numpy(); dy[r_, 0] = c["dyaw"][k]
    np.savez_compressed(rd / "carries.npz", twins=np.array(twins), chain_twin=np.array([twins.index(c["ep"]) for c, _ in rows]), layout=lay,
                        step_obj=so, path=pa, dyaw=dy)
    S2 = run_chain_physx(rd / "carries.npz", rd / "carry_scores.npz", copies=copies)
    scores, r_ = [], 0
    for c, B in zip(cand, meta):
        cs, ps = [], []
        for k in range(len(c["plans"])):
            st = dict(choice=c["i"], path=c["plans"][k].numpy(), dest=c["d"] if c["d"] < B.I else -1, start=c["P0"][c["i"]], forced=False)
            cc, parts = chain_cost(B, dict(ep=c["ep"]), dict(steps=[st]), S2, r_, 0.0, allowed=c["allowed"], report_end=False)
            cs.append(cc); ps.append(parts); r_ += 1
        scores.append(dict(cost=np.array(cs, np.float64), parts=ps))
    return cand, scores


# ------------------------------------------------------------------ cost
def _yaw_off(q, yaw_baked):
    from scipy.spatial.transform import Rotation as R
    return float((R.from_quat(q) * R.from_euler("z", float(yaw_baked)).inv()).as_euler("zyx")[0])


def _tilt(qa, qb):
    from scipy.spatial.transform import Rotation as R
    ua = R.from_quat(qa).apply([0, 0, 1.0]); ub = R.from_quat(qb).apply([0, 0, 1.0])
    return float(np.degrees(np.arccos(np.clip((ua * ub).sum(), -1, 1))))


def end_state(B, pose):
    """Physical end poses (I, 7) -> (xyz (I,3), yaw_off (I,), rests_on (I,) from the bridge's geometric test)."""
    I = B.I
    xyz = pose[:I, :3].astype(np.float32)
    # physical quaternion = extra yaw x the yaw baked into the USD (scene.json), not the bridge's long-side yaw (bridge_obb may change it)
    yaw = np.array([_yaw_off(pose[k, 3:7], B.S["objects"][k].get("yaw_baked", 0.0)) for k in range(I)], np.float32)
    ro = B._rests_on(torch.as_tensor(xyz[None]), torch.as_tensor(yaw[None]))[0].numpy()
    return xyz, yaw, ro


def end_check(B, chain, S, j):
    """Evaluation only, never part of a training cost. Per carried object: does it rest on the destination the model chose (its PhysX
    support chain reaches that destination through selected objects only) and lie within its own height (vertically) and half length
    (in the plane) of its planned goal? -> number of misses."""
    I = B.I
    mv = [st for st in chain["steps"] if st["choice"] < I]
    sel = np.zeros(I, bool); sel[[st["choice"] for st in mv]] = True
    xyz, yaw, ro = end_state(B, S["pose_end"][j])
    sup = S["support_end"][j][:I] if "support_end" in S.files else None
    if sup is not None and (sup >= -1).all():
        ro = sup                                                                   # PhysX contact support (else the bridge's geometry)
    miss = []
    for st in mv:
        i, d = st["choice"], st["dest"]
        x, seen, ok_chain = i, set(), False
        while True:                                                                # follow the physical rests-on chain from i through selected objects
            s_ = int(ro[x])
            if (d < 0 and s_ < 0) or (d >= 0 and s_ == d):
                ok_chain = True; break
            if s_ < 0 or s_ in seen or not sel[s_]:
                break
            seen.add(s_); x = s_
        goal = st["path"][-1]; ex = np.asarray(B.Z["extents"][i], np.float64)
        ok_pos = abs(float(xyz[i, 2] - goal[2])) <= float(ex[2]) and float(np.linalg.norm(xyz[i, :2] - goal[:2])) <= 0.5 * float(max(ex[0], ex[1]))
        miss.append(not (ok_chain and ok_pos))
    return int(sum(miss))


def chain_cost(B, scene, chain, S, j, judge_p, allowed=None, report_end=True):
    """Chain cost (see module docstring) of flat chain j -> (cost or NaN, parts dict). allowed (I,) bool: objects besides the chain's
    own that may move without cost (for a single carry scored alone: the objects its chain carried before). report_end: also report
    end_check() as parts["end_miss"]; it is not part of the cost, and training calls pass False."""
    I = B.I
    if not np.isfinite(S["pre_unstable"][j]) or S["pre_unstable"][j] > 0:
        return np.nan, dict(pre_unstable=True)
    mv = [st for st in chain["steps"] if st["choice"] < I]
    sel = np.zeros(I, bool); sel[[st["choice"] for st in mv]] = True
    p0, p1 = S["pose_start"][j], S["pose_end"][j]
    ok_mv = np.zeros(I, bool) if allowed is None else np.asarray(allowed, bool)[:I]
    ext = ~sel & ~ok_mv
    disp = np.array([np.linalg.norm(p1[k, :3] - p0[k, :3]) for k in range(I)])
    tl = np.array([_tilt(p0[k, 3:7], p1[k, 3:7]) for k in range(I)])
    n_disp_ext = int(((disp > DISP_M) & ext).sum()); n_top_ext = int(((tl > TILT_DEG) & ext).sum())
    n = len(mv)
    n_pre_moved = 0
    if V2["on"] and "pose_pre" in S.files:                                       # moved before its own carry step = displaced
        first = {}
        for k_, st in enumerate(mv):
            first.setdefault(st["choice"], k_)
        for i, s_ in first.items():
            if ok_mv[i]:
                continue
            pp = S["pose_pre"][j, s_, i]
            if not np.isfinite(pp).all():
                continue
            dm, tm = np.linalg.norm(pp[:3] - p0[i, :3]) > DISP_M, _tilt(p0[i, 3:7], pp[3:7]) > TILT_DEG
            n_disp_ext += int(dm); n_top_ext += int(tm); n_pre_moved += int(dm or tm)
    hits = []
    if V2["on"] and V2["graded"] and "hit_g" in S.files:                         # graded hit (0-2, mean over 5-95%; other objects; start-exempt)
        hits = [float(np.nan_to_num(S["hit_g"][j, k_])) for k_ in range(n)]
    elif V2["on"] and "f_any" in S.files:                                        # 5-95% of the path: any other object
        for k_ in range(n):
            fa, cl = float(S["f_any"][j, k_]), float(S["clear_cm"][j, k_]) / 100.0
            hits.append(0.0 if not np.isfinite(fa) else (1.0 if fa > 0 else float(np.clip((V2["zone_m"] - cl) / V2["zone_m"], 0.0, 1.0))))
    fe = np.nan_to_num(S["f_ext"][j, :n]); fh = np.nan_to_num(S["f_hand_ext"][j, :n])
    scm = np.nan_to_num(S["settle_cm"][j, :n]); sdg = np.nan_to_num(S["settle_deg"][j, :n])
    econ = []
    for st in mv:
        pth = np.asarray(st["path"]); st0 = np.asarray(st["start"])
        if V2["on"]:                                                               # table plane only: lifting is free
            L = np.linalg.norm(np.diff(pth[:, :2], axis=0), axis=-1).sum(); Sx = max(np.linalg.norm(pth[-1, :2] - st0[:2]), 0.02)
        else:
            L = np.linalg.norm(np.diff(pth, axis=0), axis=-1).sum(); Sx = max(np.linalg.norm(pth[-1] - st0), 0.02)
        econ.append(max(0.0, L / Sx - 1))
    xyz = p1[:I, :3]
    tb = B.S["table"]
    off = int(sum(not (tb["x"][0] <= xyz[k, 0] <= tb["x"][1] and tb["y"][0] <= xyz[k, 1] <= tb["y"][1]) or xyz[k, 2] < -0.05 for k in range(I)))
    parts = dict(n_moves=n, n_disp_ext=n_disp_ext, n_topple_ext=n_top_ext, f_ext=float(np.log1p(fe).sum()), f_hand_ext=float(np.log1p(fh).sum()),
                 settle=float((CW["settle_cm"] * scm + CW["settle_deg"] * sdg).sum()), econ=float(np.sum(econ)), off=off,
                 judge_done=float(judge_p), stopped=not chain["steps"][-1].get("forced", False),
                 hit=float(np.sum(hits)), n_hit=int(np.sum(np.nan_to_num(S["f_any"][j, :n]) > 0)) if "f_any" in S.files else 0,
                 n_pre_moved=n_pre_moved)
    if report_end:
        parts["end_miss"] = end_check(B, chain, S, j)                              # reported for evaluation; not in the cost
    c = (V2["hit"] * parts["hit"] if V2["on"] else 0.0) + (CW["ext_disp"] * n_disp_ext + CW["ext_topple"] * n_top_ext + CW["f_ext"] * parts["f_ext"] + CW["f_hand_ext"] * parts["f_hand_ext"]
         + parts["settle"] + CW["econ"] * parts["econ"] + CW["move"] * n + CW["off"] * off + CW["done"] * (1.0 - float(judge_p)))
    return float(c), parts


@torch.no_grad()
def judge(ref, B, xyz, yaw, kind):
    """Frozen reference model on a state: 'stop' = its P(STOP) in the joint softmax (nothing taken), 'progress' = its progress head."""
    from cleantables.worldmodel.physics.choice import joint_logp
    b = B._batch(torch.as_tensor(xyz[None]), torch.as_tensor(yaw[None]), None)
    E = ref.encode(b, torch.ones(1, dtype=torch.long)); E["kind"] = b["obj_kind"]
    if kind == "progress":
        return float(torch.sigmoid(ref.p_vid(E["cls"][:, 0]).float().squeeze(-1))[0])
    return float(joint_logp(ref, E, B.I)[0, -1].exp())


def score_round(scenes, rd, copies, ref, judge_kind):
    from cleantables.worldmodel.bridge import WMBridge2
    write_chain_jobs(scenes, rd / "chains.npz")
    S = run_chain_physx(rd / "chains.npz", rd / "scores.npz", copies=copies)
    j = 0
    bridges = {}
    for s in scenes:
        if s["ep"] not in bridges:
            bridges[s["ep"]] = WMBridge2(s["ep"], None, models=[ref], device="cpu")
        B = bridges[s["ep"]]
        cs, ps = [], []
        for c in s["chains"]:
            if np.isfinite(S["pre_unstable"][j]) and S["pre_unstable"][j] == 0:
                xyz, yaw, _ = end_state(B, S["pose_end"][j])
                jp = judge(ref, B, xyz, yaw, judge_kind)
            else:
                jp = 0.0
            cc, parts = chain_cost(B, s, c, S, j, jp, report_end=False)
            cs.append(cc); ps.append(parts); j += 1
        s["cost"] = np.array(cs, np.float64); s["parts"] = ps
    return scenes


# ------------------------------------------------------------------ update
def joint_logp_masked(m, E, I, taken, objects_only=False):
    """log p over the I objects + STOP with taken objects masked; objects_only: over the objects alone (STOP is not a choice)."""
    from cleantables.worldmodel.physics.choice import joint_logp
    lp = joint_logp(m, E, I)[0]
    mask = torch.cat([torch.as_tensor(taken), torch.tensor([bool(objects_only)])])
    return F.log_softmax(lp.masked_fill(mask, -1e4), -1)


def path_losses(m, ref, lr, gen, alpha_end, w_yaw):
    """ORW-CFM-W2 (paths) and RWR (end, yaw) terms of feedback_rounds for one executed sample -> (rw, w2, end, ref) losses, to be weighted."""
    from cleantables.worldmodel.base_model import NP, PS
    from cleantables.worldmodel.physics.feedback_rounds import mdn_of, mix_nll, mix_sample, yaw_sample
    b, i = lr["b"], lr["i"]
    mode = torch.ones(1, dtype=torch.long); mv = torch.tensor([i])
    with torch.no_grad():
        E = m.encode(b, mode); E["kind"] = b["obj_kind"]
        c, _, _ = m.heads_from(E, mv, lr["tg"])
    org = b["obj_geo"][:, i, :3]
    rel = (lr["path_top"] - org[:, None]) / PS
    u = torch.linspace(0, 1, NP)[None, :, None]
    x1 = torch.cat([rel - u * rel[:, -1:], torch.zeros_like(rel)], -1)
    x0 = torch.randn(x1.shape, generator=gen); t = torch.rand(1, generator=gen)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v = m.velocity(E, c, xt, t)
    with torch.no_grad():
        vr = ref.velocity(E, c, xt, t)
    l_rw = ((v - (x1 - x0))[..., :3] ** 2).mean()
    l_w2 = ((v - vr)[..., :3] ** 2).mean()
    z = m.end_disp(c) if lr["table"] else mdn_of(m, c)
    with torch.no_grad():
        zr = ref.end_disp(c) if lr["table"] else mdn_of(ref, c)
        xr = mix_sample(zr, 16, gen)
    le = mix_nll(z, lr["end_sample"]).mean(); lref = mix_nll(z, xr).mean()
    if lr["yaw_sample"] is not None and getattr(m, "use_yaw", False):
        le = le + w_yaw * m.yaw_nll(c.float(), lr["yaw_sample"]).mean()
        with torch.no_grad():
            yr = yaw_sample(ref, c, 16, gen)
        lref = lref + w_yaw * m.yaw_nll(c.expand(16, -1).float(), yr).mean()
    return l_rw, l_w2, le, lref


def update(m, ref, scenes, cfg, steps, rng, lf):
    oo = bool(cfg.get("objects_only"))
    learn_paths = bool(cfg.get("paths"))
    gen = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
    from cleantables.worldmodel.physics.choice import weights
    params = [p for n, p in m.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    items = []
    for s in scenes:
        w = weights(s["cost"], cfg["tau"])
        if w.sum() == 0:
            continue
        ch = []
        for c, wc in zip(s["chains"], w):
            sts = []
            for st in c["steps"]:
                if st["choice"] == s["I"] and (st.get("forced") or oo):
                    continue                                                       # a forced end / the reference's STOP is not the learner's choice
                with torch.no_grad():
                    E = m.encode(st["b"], torch.ones(1, dtype=torch.long)); E["kind"] = st["b"]["obj_kind"]
                    lr_ = joint_logp_masked(ref, E, s["I"], st["taken"], oo)
                sts.append((E, st["taken"], st["choice"], lr_, st.get("learn")))
            ch.append((float(wc), sts))
        items.append((s["I"], ch, float(w.sum())))
    if not items:
        log("no informative scenes this round", lf); return
    for it in range(steps):
        mb = [items[j] for j in rng.integers(0, len(items), min(cfg["B_sc"], len(items)))]
        l_rw = l_kl = 0.0; nkl = 0
        p_rw = p_w2 = p_end = p_ref = 0.0; npth = 0
        for I, ch, wsum in mb:
            for wc, sts in ch:
                for E, taken, a, lr_, lrn in sts:
                    lp = joint_logp_masked(m, E, I, taken, oo)
                    l_rw = l_rw - wc * lp[a] / wsum / len(mb)
                    l_kl = l_kl + (lr_.exp() * (lr_ - lp)).sum(); nkl += 1
                    if learn_paths and lrn is not None:
                        a_, b_, c_, d_ = path_losses(m, ref, lrn, gen, cfg["alpha_end"], cfg["w_yaw"])
                        p_rw = p_rw + wc / wsum * a_ / len(mb); p_end = p_end + wc / wsum * c_ / len(mb)
                        p_w2 = p_w2 + b_; p_ref = p_ref + d_; npth += 1
        loss = l_rw + cfg["beta"] * l_kl / max(1, nkl)
        if learn_paths and npth:
            loss = loss + p_rw + cfg["alpha"] * p_w2 / npth + p_end + cfg["alpha_end"] * p_ref / npth
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if it % 20 == 0 or it == steps - 1:
            log(f"   step {it} rwr {float(l_rw.detach()):.3f} kl {float(l_kl.detach()) / max(1, nkl):.4f}"
                + (f" | paths rw {float(p_rw):.3f} w2 {float(p_w2) / max(1, npth):.4f} end {float(p_end):.3f} ref {float(p_ref) / max(1, npth):.3f}" if learn_paths and npth else ""), lf)


# ------------------------------------------------------------------ judge choice (made before training; human labels for evaluation only)
@torch.no_grad()
def judge_check(run, out_json):
    """Which judge separates done from not-done twin states? Per labelled episode (train and test reported separately): not-done =
    the twin's start; done = the start with every object the human moved put at its human end position (end_pose of its last move);
    half = the first half of those objects in the human's order. AUROC of P(STOP) and of progress, done vs (start + half)."""
    from cleantables.worldmodel import base_bridge as W
    from cleantables.worldmodel.bridge import WMBridge2, load_models
    from cleantables.worldmodel.data import Data, is_test
    from cleantables.worldmodel.evaluate import RUNS2 as R2, auroc
    D = Data(feats=False); d = D.S.d
    ref = load_models(R2 / run, seeds=(0, 1, 2))
    last = {}
    for n in np.flatnonzero((d["mover"] >= 0) & d["end_ok"].astype(bool)):
        ep = str(D.ep[n]); mv = int(d["mover"][n]); key = (int(d["ev_id"][n]), int(d["snap_k"][n]))
        oid = int(d["obj_id"][n, mv]); e = last.setdefault(ep, {})
        if oid not in e or key >= e[oid][0]:
            e[oid] = (key, d["end_pose"][n, :4].astype(np.float32))
    rows = []
    for ep, e in sorted(last.items()):
        if not (W.EP / ep / "scene.json").exists():
            continue
        B = WMBridge2(ep, None, models=ref, device="cpu")
        sc = json.load(open(W.EP / ep / "scene.json")); tr = {o.get("track"): j for j, o in enumerate(sc["objects"][:B.I])}
        moves = sorted([(v[0], tr[k], v[1]) for k, v in e.items() if k in tr], key=lambda x: x[0])
        if not moves:
            continue
        P0 = B.Z["start_pos"].astype(np.float32).copy(); Y0 = np.zeros(B.I, np.float32)
        states = {"start": (P0, Y0)}
        for nm, mm in (("half", moves[:max(1, len(moves) // 2)]), ("done", moves)):
            P = P0.copy()
            for _, j, pose in mm:                                                  # end_pose = (x, y, top, yaw): bottom = top - height
                P[j] = [pose[0], pose[1], max(0.0, float(pose[2]) - float(B.Z["extents"][j, 2]))]
            states[nm] = (P, Y0)
        for nm, (P, Y) in states.items():
            if nm == "half" and len(moves) < 2:
                continue
            ps = np.mean([judge(r, B, P, Y, "stop") for r in ref]); pg = np.mean([judge(r, B, P, Y, "progress") for r in ref])
            rows.append(dict(ep=ep, split="test" if is_test(ep) else "train", state=nm, p_stop=float(ps), progress=float(pg)))
    res = dict(run=run, n=len(rows))
    for sp in ("train", "test", "all"):
        R_ = [r for r in rows if sp == "all" or r["split"] == sp]
        y = np.array([r["state"] == "done" for r in R_], float)
        res[sp] = dict(n=len(R_), auroc_stop=auroc(np.array([r["p_stop"] for r in R_]), y), auroc_progress=auroc(np.array([r["progress"] for r in R_]), y),
                       mean_stop={k: float(np.mean([r["p_stop"] for r in R_ if r["state"] == k])) for k in ("start", "half", "done")},
                       mean_progress={k: float(np.mean([r["progress"] for r in R_ if r["state"] == k])) for k in ("start", "half", "done")})
    res["rows"] = rows
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"judge_check {run}: " + "; ".join(f"{sp}: AUROC stop {res[sp]['auroc_stop']:.2f} progress {res[sp]['auroc_progress']:.2f}" for sp in ("train", "test", "all")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", required=True)
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--scenes", type=int, default=48); ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60); ap.add_argument("--tau", type=float, default=2.0); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--copies", type=int, default=16); ap.add_argument("--temp", type=float, default=1.5)
    ap.add_argument("--dr_list", default=""); ap.add_argument("--dr_frac", type=float, default=0.6)
    ap.add_argument("--judge", default="stop", choices=["stop", "progress", "none"]); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--judge_check", default="", help="only run judge_check(--init) and write this json")
    ap.add_argument("--paths", action="store_true", help="also learn the path / end / yaw heads (regpf2's ORW-CFM-W2 + RWR) from the chain cost")
    ap.add_argument("--alpha", type=float, default=0.1); ap.add_argument("--alpha_end", type=float, default=0.1); ap.add_argument("--w_yaw", type=float, default=0.5)
    ap.add_argument("--v2cost", action="store_true", help="final_chainpath2: carried-object hits / safe zone, table-plane economy, moved-before-its-turn")
    ap.add_argument("--carry_pf", action="store_true", help="final_chainpath2: paths learned per carry from k_path + k_end evaluated-style plans")
    ap.add_argument("--k_path", type=int, default=8); ap.add_argument("--k_end", type=int, default=4)
    ap.add_argument("--graded", action="store_true", help="graded carried-object hit (physx_chain hit_g)")
    ap.add_argument("--diverse", action="store_true", help="candidates: 1 evaluated plan + single flow samples + ends at 2x spread")
    a = ap.parse_args()
    V2["on"] = bool(a.v2cost); V2["graded"] = bool(a.graded); V2["diverse"] = bool(a.diverse)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8"))); torch.manual_seed(a.seed)
    if a.judge_check:
        judge_check(a.init, a.judge_check); return
    if a.smoke:
        a.rounds, a.scenes, a.chains, a.steps, a.copies = 1, 3, 2, 3, 2
        a.k_path, a.k_end = min(a.k_path, 2), min(a.k_end, 1)
    if a.judge == "none":                                                         # no DONE judge: STOP stays with the frozen reference
        CW["done"] = 0.0; CW["move"] = 0.0                                     # and no per-move cost (chain length is not the learner's)
    from cleantables.worldmodel.bridge import load_models
    cfg = dict(tau=a.tau, beta=a.beta, lr=a.lr, B_sc=8, objects_only=a.judge == "none", paths=a.paths, alpha=a.alpha, alpha_end=a.alpha_end, w_yaw=a.w_yaw)
    sd = PF / a.name / f"seed{a.seed}"; sd.mkdir(parents=True, exist_ok=True); lf = sd / "log.txt"
    json.dump(dict(vars(a), chain_cost_weights=CW, v2=V2, replay=dict(carry=os.environ.get("WM_CARRY", "pose"), hits=os.environ.get("WM_HITS", "0")), end_rule="none (end_miss removed)", max_steps=MAX_STEPS, w_max=W_MAX, trained=list(TRAINED),
                   twin_root=os.environ.get("WM_TWIN_ROOT", "")), open(sd / "cfg.json", "w"), indent=1)
    src = RUNS2 / a.init if (RUNS2 / a.init).exists() else OUT / "wm_v2" / "runs" / a.init
    m = load_models(src, seeds=(a.seed,))[0]
    if getattr(m, "endref", False) or getattr(m, "watch", False):
        raise SystemExit("chains: base models only (destination / mover heads); new models train in cleantables.worldmodel.physics.stage")
    ref = copy.deepcopy(m).eval()                                                 # frozen reference: KL anchor and DONE judge
    dr_names = json.load(open(a.dr_list))["names"] if a.dr_list else []
    for p in list(ref.parameters()) + list(m.parameters()):
        p.requires_grad_(False)
    for n, p in m.named_parameters():
        from cleantables.worldmodel.physics.feedback_rounds import ENDH, FLOW
        if n.split(".")[0] in (TRAINED if a.judge != "none" else ("mover_h",)) + ((FLOW + ENDH) if (a.paths or a.carry_pf) else ()):
            p.requires_grad_(True)
    rng = np.random.default_rng(a.seed)
    for r in range(a.rounds):
        rd = sd / f"round{r}"; rd.mkdir(exist_ok=True)
        if (rd / "model.pt").exists():
            m.load_state_dict(torch.load(rd / "model.pt")); continue
        t0 = time.time()
        if (rd / "chains.pt").exists():
            scenes = torch.load(rd / "chains.pt", weights_only=False)
        else:
            m.eval(); scenes = sample_chains(m, a.scenes, a.chains, seed=1000 * a.seed + r + 700, temp=a.temp, dr_names=dr_names, dr_frac=a.dr_frac,
                                             ref=ref if a.judge == "none" else None, paths=a.paths and not a.carry_pf)
            torch.save(scenes, rd / "chains.pt")
        nch = sum(len(s["chains"]) for s in scenes)
        log(f"[{a.name} s{a.seed}] round {r}: {len(scenes)} scenes, {nch} chains, mean moves "
            f"{np.mean([sum(st['choice'] < s['I'] for st in c['steps']) for s in scenes for c in s['chains']]):.2f} ({time.time() - t0:.0f} s)", lf)
        scenes = score_round(scenes, rd, a.copies, ref, a.judge)
        C = np.concatenate([s["cost"] for s in scenes]); P_ = [p for s in scenes for p in s["parts"] if "n_moves" in p]
        log(f"[{a.name} s{a.seed}] round {r}: mean cost {np.nanmean(C):.3f} (finite {np.isfinite(C).mean():.2f}); external moved "
            f"{np.mean([p['n_disp_ext'] > 0 for p in P_]):.2f}, "
            f"judge done {np.mean([p['judge_done'] for p in P_]):.2f} ({time.time() - t0:.0f} s)", lf)
        if V2["on"]:
            log(f"[{a.name} s{a.seed}] round {r}: chain carries hitting (contact / inside {100 * V2['zone_m']:.0f} cm, 5-95%) "
                f"{np.sum([p['hit'] for p in P_]) / max(1, np.sum([p['n_moves'] for p in P_])):.2f} / carry, contact {np.sum([p['n_hit'] for p in P_]) / max(1, np.sum([p['n_moves'] for p in P_])):.2f}; "
                f"moved before their turn {np.sum([p['n_pre_moved'] for p in P_]) / max(1, len(P_)):.2f} / chain", lf)
        torch.save(scenes, rd / "chains.pt")
        if a.carry_pf:
            from cleantables.worldmodel.physics.feedback_rounds import update as path_update
            gen = torch.Generator().manual_seed(int(1000 * a.seed + r + 900))
            S_ = np.load(rd / "scores.npz", allow_pickle=True)
            t1 = time.time()
            cand, csc = carry_round(scenes, S_, m.eval(), ref, rd, a.copies, np.random.default_rng(1000 * a.seed + r + 800), gen, a.k_path, a.k_end)
            Cc = np.concatenate([x["cost"] for x in csc]); Pp = [p for x in csc for p in x["parts"] if "n_moves" in p]
            log(f"[{a.name} s{a.seed}] round {r}: {len(cand)} carries x {a.k_path}+{a.k_end} plans: mean cost {np.nanmean(Cc):.3f} (finite "
                f"{np.isfinite(Cc).mean():.2f}), hit {np.mean([p['hit'] for p in Pp]):.2f}, contact {np.mean([p['n_hit'] for p in Pp]):.2f}, "
                f"spread within carry {np.nanmean([np.nanmax(x['cost']) - np.nanmin(x['cost']) for x in csc if np.isfinite(x['cost']).sum() > 1]):.2f} ({time.time() - t1:.0f} s)", lf)
            torch.save(dict(cand=cand, scores=csc), rd / "carries.pt")
            update(m, ref, scenes, dict(cfg, paths=False), a.steps, rng, lf)     # object choice, from the chains
            path_update(m, ref, cand, csc, dict(lr=a.lr, tau=a.tau, alpha=a.alpha, alpha_end=a.alpha_end, w_yaw=a.w_yaw, B_sc=8), a.steps, gen, lf, end=True)
        else:
            update(m, ref, scenes, cfg, a.steps, rng, lf)
        torch.save(m.state_dict(), rd / "model.pt")
    if not a.smoke:
        od = RUNS2 / a.name; od.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), od / f"fold-1_seed{a.seed}.pt")
        base = json.load(open(src / f"fold-1_seed{a.seed}.json"))
        base["pf_chain"] = json.load(open(sd / "cfg.json"))
        json.dump(base, open(od / f"fold-1_seed{a.seed}.json", "w"), indent=1)
    log(f"done {a.name} seed {a.seed}", lf)


if __name__ == "__main__":
    main()
