"""Training driver for RL in physics. One process (Isaac Lab sim env) holds the chain env (`cleantables.rl.chain_env`), the
frozen world model (LivePlanner, also the judge) and the trained policy.

Stages: `bc`, `variance`, `rl` and `eval` train and test the small action policy (`cleantables.rl.policy`): behaviour-cloning data from
the non-learned baseline -> BC policy -> variance check -> group-relative RL with a KL penalty to the BC policy. `wminit`, `wmrl` and
`wmeval` run the world-model planner (`cleantables.rl.wm_actor`): its untrained init, GRPO training with group-relative advantages over
episodes from the same start table (optionally with the PPO low-level controller, `cleantables.rl.controller`), and held-out evaluation.
  $SIM_PY -m cleantables.rl.train --stage bc       --copies 4  [--episodes 2000]
  $SIM_PY -m cleantables.rl.train --stage variance --copies 8
  $SIM_PY -m cleantables.rl.train --stage wmrl     --copies 8 --iters 50 --tag T
  $SIM_PY -m cleantables.rl.train --stage wmeval   --split SPLIT --actor CKPT
Outputs go to wm_v2_world/fastpath/stageA_<tag>/ (BC data, policies, variance.json, logs, TensorBoard, checkpoints).
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("FP_HANDS", "2")
W = Path(os.environ.get("WM_EVAL_OUT", CT_DATA + "/wm_eval")) / "wm_v2_world"
EP_S = float(os.environ.get("FPR_EP_S", "90"))
W_YIELD_TAKE = float(os.environ.get("FPR_W_YIELD_TAKE", "0"))   # per-take charge per second of shield yield (0.5 = -0.1 per yield)
SD_FLOOR = float(os.environ.get("FPR_SD_FLOOR", "0.0"))            # advantage sd floor
EP_WEIGHT = os.environ.get("FPR_EP_WEIGHT", "0") == "1"            # per-episode decision weighting
ARM_RC = os.environ.get("FPR_ARM_RC", "0") == "1"                  # arm / IK seconds during a take in its per-take reward
REWARD_V2 = os.environ.get("FPR_REWARD_V2", "0") == "1"           # progress-based reward (policy.episode_reward_v2)
RC_W = float(os.environ.get("FPR_RC_W", "1.0"))                   # weight of the per-take advantage
R_FAIL = float(os.environ.get("FPR_R_FAIL", "-1.0"))              # per-take reward of a take that never completes (= a carry ending 20 cm off)
W_DEF = dict(push=0.2, mpush=0.1, static=0.2, hh=0.5, time=0.01, carry=0.02, all_met=0.5, arm=0.5, ik=0.2,
             ft=float(os.environ.get("FPR_W_FT", "0.5")),  # fingers below the table top (s), the arm-contact weight
             lost=float(os.environ.get("FPR_W_LOST", "2.0")),  # per object the robot put off the table / out of reach (= 2 x |R_FAIL|)
             hpen=float(os.environ.get("FPR_W_HPEN", "1.0")),     # per hand-object penetration abort (> 1 cm), on top of R_FAIL
             hov=float(os.environ.get("FPR_W_HOV", "0.5")),  # per second of smaller hand-object overlap (the arm-contact weight)
             met3=float(os.environ.get("FPR_W_MET3", "0.0")),     # + w x the movers' met_3d share; the all-met bonus then uses all_met3
             static_cm=float(os.environ.get("FPR_W_STATIC_CM", "0.0")),   # per cm a static object moved
             early=float(os.environ.get("FPR_W_EARLY", "0.0")))   # per unmet mover when the episode ends by a stop


def assign(E, items):
    """items with env + s0 (decisions: step) -> episode index (by env and time window), -1 if none."""
    by_env = {}
    for i, e in enumerate(E):
        by_env.setdefault(e["env"], []).append((e["s0"], e["s1"], i))
    out = []
    for it in items:
        s = it.get("s0", it.get("step")); k = -1
        for a, b, i in by_env.get(it["env"], []):
            if a <= s <= b:
                k = i; break
        out.append(k)
    return out


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bc", "variance", "rl", "eval", "wminit", "wmrl", "wmeval"]); ap.add_argument("--split", default="train")
    ap.add_argument("--copies", type=int, default=4); ap.add_argument("--episodes", type=int, default=0); ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tag", default="v1"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--policy", default="")
    ap.add_argument("--no_arms", action="store_true"); ap.add_argument("--resume", default=""); ap.add_argument("--start_it", type=int, default=0)
    ap.add_argument("--init_seed", type=int, default=0); ap.add_argument("--disturb", type=float, default=0.0); ap.add_argument("--lr_wm", type=float, default=0.0); ap.add_argument("--det_check", action="store_true"); ap.add_argument("--sigma", type=float, default=0.2); ap.add_argument("--actor", default=""); ap.add_argument("--mode", default="greedy"); ap.add_argument("--max_active", type=int, default=2); ap.add_argument("--beta_kl", type=float, default=0.05); ap.add_argument("--lr", type=float, default=1e-4)
    add_launcher_args(ap)
    a = ap.parse_args()
    import torch
    if os.environ.get("FPR_MEM_GB"):                                     # GPU sharing: cap this process's PyTorch memory
        torch.cuda.set_per_process_memory_fraction(float(os.environ["FPR_MEM_GB"]) / 23.5)
    from cleantables.rl import chain_env as FC
    from cleantables.rl import sim_env as FE
    from cleantables.rl import policy as PO
    rd = W / "fastpath" / f"stageA_{a.tag}"; rd.mkdir(parents=True, exist_ok=True); lf = rd / "log.txt"

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"; print(line, flush=True); open(lf, "a").write(line + "\n")
    twr = os.environ["WM_TWIN_ROOT"]
    names = json.load(open(W / "fastpath" / "scope_clean.json"))[a.split]
    names = [n for n in names if (Path(twr) / n / "wm_tokens.npz").exists()][: (a.limit or None)]
    if os.environ.get("FPR_EXCLUDE"):                                    # scenes left out (e.g. objects no arm reaches from the bases)
        names = [n for n in names if n not in set(os.environ["FPR_EXCLUDE"].split(","))]
    if os.environ.get("FPC_SCENES"):                                     # (videos) only these scenes
        names = [n for n in os.environ["FPC_SCENES"].split(",") if n in names]
    if os.environ.get("FPC_LAYOUT_OVERRIDE"):                            # humanoid-stance re-placed objects; excluded scenes dropped
        _ex = set(json.load(open(os.environ["FPC_LAYOUT_OVERRIDE"])).get("excluded", []))
        names = [n for n in names if n not in _ex]
    tpl = FE.build_templates(names, twin_root=twr, out=rd / f"templates_{a.split}.pt")
    from cleantables.rl.layout_override import apply as _lo_apply
    _lo_apply(tpl, os.environ.get("FPC_LAYOUT_OVERRIDE"), log=print)
    N = len(names) * a.copies
    cfg = FE.make_cfg(tpl, N, 1 / 120)
    with launch_simulation(cfg, a):
        fp = FE.FastPath(cfg); dev = fp.dev
        from cleantables.worldmodel.physics.live_plan import LivePlanner
        LP = LivePlanner(names, device=str(dev))
        tenv = np.asarray(fp.tpl_of_env); lay = fp.lay0.clone()
        xyz0 = (lay[..., :3] - torch.tensor([0, 0, 0.002], device=dev)).cpu().numpy(); yaw0 = np.zeros((N, fp.K), np.float32)
        for e in range(N):
            tt = tpl["templates"][tenv[e]]; yaw0[e, :tt["n"]] = [o["yaw"] for o in tt["objects"]]
        if os.environ.get("FPR_JUDGE_SEEDS"):                          # the judge uses world-model seeds other than the actor's (the actor uses seed 0)
            LPJ = LivePlanner(names, device=str(dev), seeds=tuple(int(x) for x in os.environ["FPR_JUDGE_SEEDS"].split(",")))
            log(f"JUDGE seeds {os.environ['FPR_JUDGE_SEEDS']} (the actor is seed 0)")
        else:
            LPJ = LP
        J = FC.FrozenJudge(LPJ, tenv, np.nan_to_num(xyz0), yaw0)
        if FC.MET3:                                                      # the frozen WM's yaw change per object at the start (met_3d)
            J.dy0 = np.asarray(LP._wm_fast(tenv, np.nan_to_num(xyz0), yaw0)["dy"], np.float32)
        if FC.INSTR_P > 0:                                               # instruction sampling: a second frozen judge under the other instruction
            from cleantables.rl import wm_actor as WA_
            ins_own = [json.load(open(Path(twr) / n_ / "scene.json")).get("instruction") for n_ in names]
            texts = sorted({x for x in ins_own if x})
            other = [next((x for x in texts if x != o_), o_) for o_ in ins_own]                    # the clean scope has exactly two instructions
            # each twin stores its own instruction's MiniLM vector (wm_tokens task_emb); the other text's vector = any twin's with that text
            vec = {}
            for t_, o_ in enumerate(ins_own):
                vec.setdefault(o_, np.asarray(LP.B[t_].Z["task_emb"], np.float32).reshape(-1))
            ovec = [vec[o_] for o_ in other]
            J1 = FC.FrozenJudge(LPJ, tenv, np.nan_to_num(xyz0), yaw0, instruction=lambda t_: ovec[t_][None])
            emb = torch.stack([torch.stack([torch.as_tensor(vec[ins_own[t_]]), torch.as_tensor(ovec[t_])]) for t_ in range(len(names))]).to(dev)   # (T, 2, D)
            WA_.EMB_T = emb
            if FC.MET3:
                J1.dy0 = np.asarray(LP._wm_fast(tenv, np.nan_to_num(xyz0), yaw0, temb=emb[torch.as_tensor(tenv), 1])["dy"], np.float32)
            log(f"INSTRUCTION SAMPLING: P(other) {FC.INSTR_P}; texts {texts}")
        else:
            J1 = None
        gtab = FC.grasp_table(tpl, filt=FC.GRASP_FILT); CS = FC.ChainSim(fp)
        if FC.CONT_EVAL:                                                 # evaluation only: object names + instructions for the container check
            _io = [json.load(open(Path(twr) / n_ / "scene.json")).get("instruction") for n_ in names]; _tx = sorted({x for x in _io if x})
            CS.cont_meta = dict(ids=[[o["id"] for o in t_["objects"]] for t_ in tpl["templates"]], ext=[np.asarray(t_["extents"], np.float32) for t_ in tpl["templates"]],
                                inst_own=_io, inst_other=[next((x for x in _tx if x != o_), o_) for o_ in _io])
            log(f"CONTAINER CHECK on (evaluation only): instructions {_tx}")
        if J1 is not None:                                               # the judge per env follows CS.inst (resampled every episode)
            CS.inst = np.zeros(N, bool); J = FC.JudgeMux(J, J1, CS)
        if not a.no_arms:
            CS.arms = FC.ArmTracker(CS, names, every=int(os.environ.get("FPC_ARM_EVERY", "12")))
        if FC.LOWPOL:                                                    # the shared two-arm low-level controller
            from cleantables.rl import controller as LPm
            CS.lowpol = LPm.LowPol(CS, J, dev)
            log(f"LOWPOL: shared two-arm controller, obs {CS.lowpol.d_obs}, params {sum(p_.numel() for p_ in CS.lowpol.net.parameters())}")
        from cleantables.rl import layouts as FLY
        if FLY.LAYOUT_RAND > 0 and CS.arms is not None:                    # per-episode random layouts
            CS.layout_rand = FLY.LayoutRand(FLY.LAYOUT_RAND, lay.cpu().numpy(), fp.valid.cpu().numpy(), fp.ext.cpu().numpy(), tenv,
                                            CS.arms.box.cpu().numpy(), CS.arms.base[:, :, :3, 3].cpu().numpy(), seed=int(os.environ.get("FPC_LAYOUT_SEED", "0")))
            CS.layout_rand.snap = FLY.judge_snapshot(J); CS.layout_rand.yaw0 = yaw0.copy(); CS.layout_rand.LPa = LP
            CS.layout_rand.temb_T = WA_.EMB_T if J1 is not None else None
            log(f"LAYOUT RAND: p {FLY.LAYOUT_RAND} per (table, episode), WM-actor training rollouts only; seed {CS.layout_rand.seed}")
        mc = int(max(t["n"] for t in tpl["templates"]) + 2)
        box = np.array([[t["table"]["x"][0] - 0.1, t["table"]["x"][1] + 0.1, t["table"]["y"][0] - 0.1, t["table"]["y"][1] + 0.1] for t in tpl["templates"]], np.float32)
        pol = PO.Policy().to(dev)
        if a.policy:
            pol.load_state_dict(torch.load(a.policy, map_location=dev))

        def run(mode, n_eps, policy=None, seed=0, quota=0):
            dec = PO.PolicyDecider(policy or pol, LP, J, CS, tenv, gtab, mode=mode, ws_box=box, dev=str(dev))
            dec.tpl_env = tenv
            if os.environ.get("FPC_REC_OUT"):                            # record the baseline too, as run_wm does
                CS.rec = FC.Recorder(CS, J, names, os.environ["FPC_REC_OUT"], every=int(os.environ.get("FPC_REC_EVERY", "12")))
            torch.manual_seed(seed)
            if getattr(CS, "lowpol", None) is not None:                  # fresh rollout buffer; greedy / init / baseline use the controller's mean
                CS.lowpol.begin(sample=mode not in ("greedy", "init", "baseline"))
            if getattr(CS, "layout_rand", None) is not None:           # random layouts only in the WM actor's training rollouts
                CS.layout_rand.begin(active=(mode == "sample" and type(dec).__name__ == "WMActorDecider"), ep_seed=seed)
            r = FC.run_continuous(fp, CS, dec, J, lay, mc, n_eps, EP_S, disturb=a.disturb, log=log, max_active=a.max_active, rng=np.random.default_rng(seed), quota=quota)
            if getattr(CS, "layout_rand", None) is not None and CS.layout_rand.active:
                log(CS.layout_rand.summary())
            E = r.pop("episodes"); cm = FC.carry_metrics(CS, fp, CS.ev); mv = FC.move_pushes(CS, fp)
            ce, me = assign(E, cm), assign(E, mv)
            for i, ep in enumerate(E):
                ep["reward"] = (PO.episode_reward_v2 if REWARD_V2 else PO.episode_reward)(ep, [c for c, k in zip(cm, ce) if k == i], [m for m, k in zip(mv, me) if k == i], dict(W_DEF, dt=CS.dt))
                ep["push_held_n"] = sum(c["push_held_cm"] > 1 for c, k in zip(cm, ce) if k == i)
            r["carry_lift_cm_median"] = float(np.median([c["lift_cm"] for c in cm])) if cm else None
            r["carry_flat_share"] = float(np.mean([c["lift_cm"] < 2 for c in cm])) if cm else None
            ez = [np.abs(np.tanh(d["z"][:3 * PO.N_CTRL])).mean() * PO.R_EDIT * 100 for d in dec.log if "z" in d]
            r["edit_cm_mean"] = float(np.mean(ez)) if ez else 0.0
            r["types"] = {t: int(sum(d["typ"] == i for d in dec.log)) for i, t in enumerate(PO.TYPES)}
            if os.environ.get("FPR_DUMP_EPS") == "1":                    # logging only, as run_wm: episodes, carries, takes, met per mover
                _df = lambda o: o.tolist() if hasattr(o, "tolist") else float(o)  # noqa: E731
                json.dump(dict(names=[names[t_] for t_ in tenv.tolist()], episodes=E, carries=cm, moves=mv, ep_of_carry=ce), open(rd / f"eps_{mode}.json", "w"), default=_df)
                json.dump(dict(names=[names[t_] for t_ in tenv.tolist()], takes=CS.takes), open(rd / f"ik_diag_{mode}.json", "w"), default=_df)
                if getattr(CS, "met3_diag", None):
                    json.dump(CS.met3_diag, open(rd / f"met3_diag_{mode}.json", "w"), default=_df); CS.met3_diag = []
            return E, dec.log, r

        def run_wm(A_, mode, n_eps, seed=0, quota=0):
            from cleantables.rl import wm_actor as WA
            dec = WA.WMActorDecider(A_, LP, J, CS, tenv, gtab, mode=mode, ws_box=box, sigma=a.sigma)
            if CS.arms is not None:
                CS.arms.n_retract = 0; CS.arms.n_move_abort = 0
            if os.environ.get("FPC_REC_OUT"):                            # record this run for cleantables.rl.render
                CS.rec = FC.Recorder(CS, J, names, os.environ["FPC_REC_OUT"], every=int(os.environ.get("FPC_REC_EVERY", "12")))
            torch.manual_seed(seed)
            if getattr(CS, "lowpol", None) is not None:                  # fresh rollout buffer; greedy / init / baseline use the controller's mean
                CS.lowpol.begin(sample=mode not in ("greedy", "init", "baseline"))
            if getattr(CS, "layout_rand", None) is not None:           # random layouts only in the WM actor's training rollouts
                CS.layout_rand.begin(active=(mode == "sample" and type(dec).__name__ == "WMActorDecider"), ep_seed=seed)
            r = FC.run_continuous(fp, CS, dec, J, lay, mc, n_eps, EP_S, disturb=a.disturb, log=log, max_active=a.max_active, rng=np.random.default_rng(seed), quota=quota)
            if getattr(CS, "layout_rand", None) is not None and CS.layout_rand.active:
                log(CS.layout_rand.summary())
            E = r.pop("episodes"); cm = FC.carry_metrics(CS, fp, CS.ev); mv = FC.move_pushes(CS, fp)
            ce, me = assign(E, cm), assign(E, mv)
            for i, ep in enumerate(E):
                ep["reward"] = (PO.episode_reward_v2 if REWARD_V2 else PO.episode_reward)(ep, [c for c, k in zip(cm, ce) if k == i], [m for m, k in zip(mv, me) if k == i], dict(W_DEF, dt=CS.dt))
                ep["push_held_n"] = sum(c["push_held_cm"] > 1 for c, k in zip(cm, ce) if k == i)
            r["carry_lift_cm_median"] = float(np.median([c["lift_cm"] for c in cm])) if cm else None
            r["carry_flat_share"] = float(np.mean([c["lift_cm"] < 2 for c in cm])) if cm else None
            r["types"] = {t: int(sum(d["typ"] == i for d in dec.log)) for i, t in enumerate(PO.TYPES)}
            r["dec_tm"] = {k: round(v, 1) for k, v in dec.tm.items()}
            if getattr(dec, "n_rep", 0):                                 # path repair: how far route_opt moved the WM's paths
                log(f"PATH REPAIR: checked {getattr(dec, 'n_rep_chk', 0)}, repaired {dec.n_rep} carry paths, mean point displacement {dec.rep_cm / max(dec.n_rep, 1):.2f} cm, {dec.tm.get('repair', 0.0):.1f} s; "
                    f"arm-arm aborts {getattr(CS.arms, 'n_arm_abort', 0)}; solid-arm blocks {getattr(CS.arms, 'n_block', 0)}, block aborts {getattr(CS.arms, 'n_block_abort', 0)}, shield projections {getattr(CS.arms, 'n_proj', 0)}, leader pauses {getattr(CS.arms, 'n_pause', 0)}, idle arms sent home {getattr(CS.arms, 'n_idle_home', 0)}, bounces {getattr(CS.arms, 'n_bounce', 0)}, bounce aborts {getattr(CS.arms, 'n_bounce_abort', 0)}")
            if getattr(CS, "lost_kinds", None):                              # logging only: robot-caused off-table objects by kind
                log(f"LOST KINDS (robot-caused off-table, this rollout): {CS.lost_kinds}"); CS.lost_kinds = {}
            r["take_ik_fail"] = CS.n_take_abort / max(CS.n_take, 1)
            r["gp_init_ok"] = getattr(dec, "n_gp_ok", 0) / max(getattr(dec, "n_gp_q", 0), 1) if getattr(dec, "n_gp_q", 0) else float("nan")
            r["reflex"] = CS.n_reflex; r["hand_obj_abort"] = getattr(CS, "n_hand_obj_abort", 0)
            if getattr(CS, "herr_st", None):                             # the force-limited hand's palm tracking error (cm)
                r["hand_err"] = {("carry" if k_ == 1 else "move"): dict(mean_cm=round(100 * v_["s"] / max(v_["n"], 1), 2), gt1cm=round(v_["gt1"] / max(v_["n"], 1), 3),
                                 gt3cm=round(v_["gt3"] / max(v_["n"], 1), 3), max_cm=round(100 * v_["mx"], 1)) for k_, v_ in CS.herr_st.items()}
                CS.herr_st = {}
            if getattr(CS, "pen_diag", None):                            # diagnostics of penetration aborts: phase, target / neighbour, depth, approach u
                import collections as _co
                pdg = CS.pen_diag
                r["pen_diag"] = dict(n=len(pdg), by_phase={f"{a_}/{b_}": v_ for (a_, b_), v_ in _co.Counter((x["phase"], x["after"]) for x in pdg).items()},
                                     target=sum(x["target"] for x in pdg), depth_cm_q=np.round(np.percentile([x["depth_cm"] for x in pdg], [10, 50, 90]), 2).tolist(),
                                     u_hist=np.histogram([x["u"] for x in pdg if x["u"] is not None], bins=[0, .2, .4, .6, .8, 1.01])[0].tolist())
                json.dump(pdg, open(rd / f"pen_diag_{mode}.json", "w")); CS.pen_diag = []
            if getattr(CS, "met3_diag", None):                           # diagnostics: met_xy vs met_3d per mover
                json.dump(CS.met3_diag, open(rd / f"met3_diag_{mode}.json", "w")); CS.met3_diag = []
            r["retracts"] = CS.arms.n_retract if CS.arms is not None else 0; r["move_ik_abort"] = CS.arms.n_move_abort if CS.arms is not None else 0
            r["take_finger_table"] = (CS.n_take_ft + int(CS.ft_take.sum())) / max(CS.n_take, 1)        # takes whose fingers went below the table top
            r["finger_table_env_share"] = float((CS.ft_steps > 0).float().mean())
            r["finger_table_ep_share"] = float(np.mean([e_.get("finger_table_s", 0) > 0 for e_ in E])) if E else float("nan")
            ez = [np.abs(np.tanh(d["zr"])).mean() * WA.R_EDIT * 100 for d in dec.log if d.get("zr") is not None and d["typ"] in (0, 2)]
            r["route_edit_cm"] = float(np.mean(ez)) if ez else 0.0                                    # mean |route edit| (cm) of approaches / free moves
            log(f"timing: wall {r['wall_s']:.0f} s, physics steps {r['step_s']:.0f} s, ticks {r['tick_s']:.0f} s (decisions {r['dec_s']:.0f} s: {r['dec_tm']}), "
                f"arms {CS.arms.t_s if CS.arms is not None else 0:.0f} s")
            if getattr(CS, "prof", None):                                # FP_PROF=1: CUDA-synced split of the physics-step time (s)
                ns_ = CS.prof.pop("_steps", 1)
                log(f"step profile: {ns_} steps; {dict((k_, round(v_, 1)) for k_, v_ in sorted(CS.prof.items(), key=lambda kv: -kv[1]))}; "
                    f"ms / step {dict((k_, round(1000 * v_ / max(ns_, 1), 2)) for k_, v_ in sorted(CS.prof.items(), key=lambda kv: -kv[1]))}"); CS.prof = {}
            # Per-take reward of each take decision, from its own take record (CS.takes: its approach, its carry, its abort). A take that never
            # completes (IK / fingers-in-table abort) gets R_FAIL, the same for every object.
            cmi = {c["ev_i"]: c for c in cm}; mvi = {m["mev_i"]: m for m in mv}
            tk = {(t_["env"], t_["g"], t_["step"]): t_ for t_ in CS.takes}
            if ARM_RC and CS.arms is not None:                           # arm-contact / IK seconds during each take, charged to it
                nxt = {}
                for t_ in sorted(CS.takes, key=lambda x: x["step"]):
                    if t_.get("arm0") is not None:
                        prev = nxt.get((t_["env"], t_["g"]))
                        if prev is not None and prev.get("arm0") is not None:
                            prev["arm_cost"] = 0.05 * max(t_["arm0"][0] - prev["arm0"][0], 0.0) + 0.02 * max(t_["arm0"][1] - prev["arm0"][1], 0.0)
                            if len(t_["arm0"]) > 2 and len(prev["arm0"]) > 2:   # each shield yield charged to the take that caused it
                                prev["arm_cost"] += W_YIELD_TAKE * max(t_["arm0"][2] - prev["arm0"][2], 0.0)
                        nxt[(t_["env"], t_["g"])] = t_
            n_ab = {}
            for d in dec.log:
                d["rc"] = None
                if d["typ"] != 0:
                    continue
                t_ = tk.get((d["env"], d["g"], d["step"]))
                if t_ is None:
                    continue
                if t_["abort"] is not None:
                    d["rc"] = R_FAIL - W_DEF["lost"] * t_.get("lost", 0); n_ab[t_["abort"]] = n_ab.get(t_["abort"], 0) + 1
                    continue
                c = cmi.get(t_["ev"]); m = mvi.get(t_["mev"])
                if c is not None and not c["disturbed"]:                   # a carry dropped by a disturbance: no per-carry credit / blame
                    d["rc"] = -W_DEF["push"] * float(c["push_held_cm"] > 1) - W_DEF["mpush"] * float(m is not None and m["push_cm"] > 1) - min(c["end_err_cm"], 20.0) / 20.0
                if t_.get("lost"):                                         # objects this take put off the table are blamed here too
                    d["rc"] = (d["rc"] or 0.0) - W_DEF["lost"] * t_["lost"]
                if ARM_RC and t_.get("arm_cost") is not None:
                    d["rc"] = (d["rc"] or 0.0) - t_["arm_cost"]
            r["abort_kinds"] = n_ab
            if WA.SIL:                                                   # self-imitation: which take decisions' carries succeeded
                r["sil_carries"], r["sil_ok"] = WA.sil_select(dec.log, tk, cmi, CS.ev, FC.MET_CM)
            if os.environ.get("FPR_DUMP_EPS") == "1":                    # logging only: every episode + its carries / moves
                json.dump(dict(names=[names[t_] for t_ in tenv.tolist()], episodes=E, carries=cm, moves=mv, ep_of_carry=assign(E, cm)),
                          open(rd / f"eps_{mode}.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
            if os.environ.get("FPC_IK_DIAG") == "1" and CS.arms is not None:     # diagnostics: every take + every failed IK check
                json.dump(dict(names=[names[t_] for t_ in tenv.tolist()], takes=CS.takes, diag=getattr(CS.arms, "diag", [])),
                          open(rd / f"ik_diag_{mode}.json", "w"), default=float)
            return E, dec.log, r

        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(rd / "tb" / (os.environ.get("FPR_TB_SUB") or a.stage)))       # one subfolder per stage / HPO trial

        def tb_log(prefix, step, d):
            for k, v in d.items():
                if isinstance(v, (int, float, np.floating, np.integer)) and v == v:
                    tb.add_scalar(f"{prefix}/{k}", float(v), step)
            tb.flush()

        _mod = set(json.load(open(os.environ["FPC_LAYOUT_OVERRIDE"])).get("override", {})) if os.environ.get("FPC_LAYOUT_OVERRIDE") else set()
        # Video-disjoint held-out scenes: the source video (scene.json "video") appears in no train scene.
        _vid = lambda n_: json.load(open(Path(twr) / n_ / "scene.json")).get("video")                                   # noqa: E731
        _train_vid = {_vid(n_) for n_ in json.load(open(W / "fastpath" / "scope_clean.json"))["train"] if (Path(twr) / n_ / "scene.json").exists()}
        disjoint_tpl = {i for i, t in enumerate(tpl["templates"]) if _vid(t["name"]) not in _train_vid}
        modified_tpl = {i for i, t in enumerate(tpl["templates"]) if t["name"] in _mod}       # re-placed layouts: marked in every eval table

        def summ(E):
            _cE = [e for e in E if e.get("cont_found")]                     # evaluation only: task objects inside their container
            _cs = dict(in_container=float(sum(e["cont_in"] for e in _cE) / max(sum(e["cont_n"] for e in _cE), 1)),
                       in_container_start=float(sum(e["cont_in0"] for e in _cE) / max(sum(e["cont_n"] for e in _cE), 1)),
                       all_in_container=float(np.mean([e["cont_in"] == e["cont_n"] for e in _cE])) if _cE else float("nan"),
                       container_eps=len(_cE)) if _cE else {}
            return dict(**_cs, episodes=len(E), reward=float(np.mean([e["reward"] for e in E])), goal_err_cm=float(np.mean([e["goal_err_cm"] for e in E])),
                        met=float(np.mean([e["met"] for e in E])), all_met=float(np.mean([e["all_met"] for e in E])),
                        carries=float(np.mean([e["carries"] for e in E])), push_held_per_ep=float(np.mean([e["push_held_n"] for e in E])),
                        makespan_s=float(np.median([e["makespan_s"] for e in E])), hh_eps=float(np.mean([e["hh_steps"] > 0 for e in E])),
                        reward_std=float(np.std([e["reward"] for e in E])),
                        stop_at_start=float(np.mean([e["stop"] == "stop" and e["carries"] == 0 for e in E])),
                        arm_arm_s_per_ep=float(np.mean([e.get("arm_arm_s", 0.0) for e in E])) if E else 0.0,       # seconds, not shares
                        near_s_per_ep=float(np.mean([e.get("near_s", 0.0) for e in E])) if E else 0.0,
                        arm_touch_s_per_ep=float(np.mean([e.get("arm_touch_s", 0.0) for e in E])) if E else 0.0,      # the arms overlapping
                        # logging only: how episodes end, and stops with movers still unmet
                        stop_kinds={k: float(np.mean([e["stop"] == k for e in E])) for k in ("all_met", "stop", "max_carries", "time", "lost_robot")} if E else {},
                        done_early_share=float(np.mean([e["stop"] == "stop" and not e.get("all_met3", e["all_met"]) for e in E])) if E else 0.0,
                        unmet_at_done=float(np.mean([e.get("n_movers", 0) - e.get("n_met3", e.get("n_movers", 0) - e.get("n_unmet", 0)) for e in E if e["stop"] == "stop"])) if any(e["stop"] == "stop" for e in E) else 0.0,
                        **({k: float(np.mean([e[k] > 0 for e in E])) for k in ("ik_fail_s", "arm_obj_s", "arm_arm_s", "arm_table_s") if E and k in E[0]}),
                        # disturbances (--disturb): disturbed objects per episode, and the share back at their goal at the episode end
                        dist_per_ep=float(np.mean([e.get("n_dist", 0) for e in E])),
                        video_disjoint_eps=float(np.mean([e["tpl"] in disjoint_tpl for e in E])) if E else 0.0,
                        met_video_disjoint=float(np.mean([e["met"] for e in E if e["tpl"] in disjoint_tpl])) if any(e["tpl"] in disjoint_tpl for e in E) else float("nan"),
                        modified_eps=float(np.mean([e["tpl"] in modified_tpl for e in E])) if E else 0.0,
                        met_unmodified=float(np.mean([e["met"] for e in E if e["tpl"] not in modified_tpl])) if any(e["tpl"] not in modified_tpl for e in E) else float("nan"),
                        met3=float(np.mean([e["met3"] for e in E])) if E and "met3" in E[0] else float("nan"),
                        all_met3=float(np.mean([e["all_met3"] for e in E])) if E and "all_met3" in E[0] else float("nan"),
                        **({k_: float(np.mean([e[k_] for e in E])) for k_ in ("met3_old", "all_met3_old", "stay_movers")} if E and "met3_old" in E[0] else {}),
                        lost_robot_per_ep=float(np.mean([e.get("n_lost_robot", 0) for e in E])), lost_dist_per_ep=float(np.mean([e.get("n_lost", 0) for e in E])),
                        recovery=float(sum(e.get("n_dist_rec", 0) for e in E) / max(sum(e.get("n_dist", 0) for e in E), 1))
                        if any(e.get("n_dist", 0) for e in E) else float("nan"))

        if a.stage in ("wminit", "wmrl", "wmeval"):
            from cleantables.rl import wm_actor as WA
            torch.manual_seed(a.init_seed); A = WA.WMActor(LP.models[0]).to(dev)   # a seeded init: the KL reference can always be rebuilt
            if a.actor:
                sd_ = torch.load(a.actor, map_location=dev, weights_only=False)
                WA.load_actor(A, sd_["actor"] if isinstance(sd_, dict) and "actor" in sd_ else sd_)        # a full checkpoint or weights only
                if a.stage == "wmeval" and getattr(CS, "lowpol", None) is not None and isinstance(sd_, dict) and "lowpol" in sd_:
                    CS.lowpol.load_state_dict(sd_["lowpol"]); log("LOWPOL: loaded from the --actor checkpoint (eval)")   # evaluate a saved controller
            if a.stage == "wminit":
                E, _, r = run_wm(A, "init", 10 ** 9, quota=1)
                if getattr(CS.arms, "clog", None):                       # arm-check interval study: contacts seen at every check vs every 2nd
                    st_ = np.array([c[0] for c in CS.arms.clog]); AA = np.stack([c[1] for c in CS.arms.clog]); OO = np.stack([c[2] for c in CS.arms.clog])
                    even = (st_ // CS.arms.every) % 2 == 0
                    for nm_, X in (("arm-arm", AA), ("arm-object", OO)):
                        ep_all = X.any(0); ep_even = X[even].any(0)
                        log(f"ARM INTERVAL {nm_}: envs with a contact (every check) {int(ep_all.sum())}, also seen at every 2nd check {int((ep_all & ep_even).sum())} "
                            f"(missed {float((ep_all & ~ep_even).sum() / max(ep_all.sum(), 1)):.3f}); contact checks {int(X.sum())}, seen-time ratio "
                            f"{float(2 * X[even].sum() / max(X.sum(), 1)):.3f}")
                tb_log(f"eval_{a.split}_init", 0, dict(summ(E), lift_cm_median=r["carry_lift_cm_median"] or float("nan"), flat_share=r["carry_flat_share"] or float("nan")))
                log(f"WM-ACTOR INIT ({a.split}): {json.dumps(summ(E))}; lift {r['carry_lift_cm_median']}; flat {r['carry_flat_share']}; types {r['types']}; eps/s {r['eps_per_s']:.2f}")
                json.dump(dict(init=summ(E), lift=r["carry_lift_cm_median"], flat=r["carry_flat_share"]), open(rd / f"wminit_{a.split}.json", "w"), indent=1)
            elif a.stage == "wmeval":
                E, _, r = run_wm(A, a.mode, 10 ** 9, quota=1)
                tb_log(f"eval_{a.split}_{a.mode}", int(os.environ.get("FPR_EVAL_STEP", "0")), dict(summ(E), lift_cm_median=r["carry_lift_cm_median"] or float("nan"),
                                                                                              flat_share=r["carry_flat_share"] or float("nan")))
                log(f"WM-ACTOR EVAL {a.mode} ({a.split}): pen_diag {r.get('pen_diag')}; hand_err {r.get('hand_err')}; take_ik_fail {r['take_ik_fail']:.3f}; aborts {r.get('abort_kinds')}; retracts {r['retracts']}; free-move aborts {r['move_ik_abort']}; reflex stops {r['reflex']}; hand_obj_abort {r['hand_obj_abort']}; gp_init_ok {r['gp_init_ok']:.3f}; take_finger_table {r['take_finger_table']:.3f}; "
                    f"finger-table envs {r['finger_table_env_share']:.3f}; finger-table eps {r['finger_table_ep_share']:.3f}; route edit {r['route_edit_cm']:.2f} cm; {json.dumps(summ(E))}; lift {r['carry_lift_cm_median']}; flat {r['carry_flat_share']}; types {r['types']}")
            else:
                # the KL reference is the original init actor (the WM + zero heads), never the loaded / trained weights
                torch.manual_seed(a.init_seed); ref = WA.WMActor(LP.models[0]).to(dev)
                for p_ in ref.parameters():
                    p_.requires_grad_(False)
                wm_p = [p_ for n_, p_ in A.named_parameters() if p_.requires_grad and n_.startswith("wm.")]
                hd_p = [p_ for n_, p_ in A.named_parameters() if p_.requires_grad and not n_.startswith("wm.")]
                wm_names = [n_ for n_, p_ in A.named_parameters() if p_.requires_grad and n_.startswith("wm.")]
                hd_names = [n_ for n_, p_ in A.named_parameters() if p_.requires_grad and not n_.startswith("wm.")]
                lr_wm = a.lr_wm or a.lr * 0.1
                opt = torch.optim.Adam([dict(params=hd_p, lr=a.lr), dict(params=wm_p, lr=lr_wm)])   # the WM (T2 flow + LoRA) learns slower (default 10x)
                opt_sil = torch.optim.Adam([p_ for n_, p_ in A.named_parameters() if p_.requires_grad and n_.startswith("wm.g_")], lr=WA.SIL_LR) if WA.SIL else None
                start = 0
                if a.resume:                                                     # full resume: actor, ref, optimiser, iteration, RNG
                    ck = torch.load(a.resume, map_location=dev, weights_only=False)
                    if isinstance(ck, dict) and "actor" in ck:
                        WA.load_actor(A, ck["actor"]); WA.load_actor(ref, ck["ref"]); start = int(ck["it"]) + 1
                        if getattr(CS, "lowpol", None) is not None and "lowpol" in ck:   # the controller resumes too (else it starts at zero)
                            CS.lowpol.load_state_dict(ck["lowpol"]); log("LOWPOL: resumed from the checkpoint")
                        if opt_sil is not None and "opt_sil" in ck:             # the self-imitation optimiser resumes too (else fresh)
                            opt_sil.load_state_dict(ck["opt_sil"]); log("SIL: optimiser resumed from the checkpoint")
                        try:
                            opt.load_state_dict(ck["opt"])
                        except ValueError:                               # new parameters since the checkpoint (zero-init heads): remap the saved Adam
                            old = set(ck["actor"].keys())                 # state by name onto the params that existed; the new ones start fresh
                            osd = ck["opt"]; nsd = opt.state_dict()
                            for gi_, (pg_new, names_) in enumerate(zip(nsd["param_groups"], (hd_names, wm_names))):
                                kept = [k for k, n_ in zip(pg_new["params"], names_) if n_ in old]
                                for k_old, k_new in zip(osd["param_groups"][gi_]["params"], kept):
                                    if k_old in osd["state"]:
                                        nsd["state"][k_new] = osd["state"][k_old]
                            opt.load_state_dict(nsd)
                            log(f"RESUME: optimiser remapped by name; new parameters start fresh: {sorted(set(hd_names + wm_names) - old)[:8]}")
                        opt.param_groups[0]["lr"] = a.lr; opt.param_groups[1]["lr"] = lr_wm          # this run's learning rates (may differ from the checkpoint's)
                        tb.add_text("lr", f"heads {a.lr}, WM {lr_wm} from iteration {start}", start)
                        tb.add_text("config", f"from iteration {start}: " + ", ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                                                                                       if k.startswith(("FPW_", "FPC_", "FPR_"))), start)
                        log("CONFIG at resume: " + ", ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith(("FPW_", "FPC_", "FPR_"))))
                        torch.set_rng_state(ck["rng"]["torch"].cpu()); torch.cuda.set_rng_state_all([s_.cpu() for s_ in ck["rng"]["cuda"]]); np.random.set_state(ck["rng"]["np"])
                        log(f"RESUME from {a.resume}: iteration {start}, args then {ck['args'].get('sigma')}, {ck['args'].get('beta_kl')}, {ck['args'].get('lr')}")
                        if os.environ.get("FPW_WAIT_BIAS_ADD") and int(os.environ.get("FPW_WAIT_BIAS_AT_IT", "-1")) == start:
                            with torch.no_grad():                                 # more exploration of "wait": its logit's bias raised once, at this iteration
                                A.type_h[-1].bias[1] += float(os.environ["FPW_WAIT_BIAS_ADD"])
                                A.type_h[-1].bias[2] += float(os.environ.get("FPW_MOVE_BIAS_ADD", "0"))   # and of the free move
                            log(f"WAIT BIAS +{os.environ['FPW_WAIT_BIAS_ADD']}, MOVE BIAS +{os.environ.get('FPW_MOVE_BIAS_ADD', '0')} applied at iteration {start} (type-head biases now {A.type_h[-1].bias.tolist()})")
                    else:                                                        # weights only: fresh optimiser
                        WA.load_actor(A, ck); start = a.start_it
                        log(f"RESUME (weights only) from {a.resume}: iteration {start}; fresh optimiser; ref = the original init")

                def save_ckpt(it):
                    ck = dict(actor=A.state_dict(), ref=ref.state_dict(), opt=opt.state_dict(), it=it, args=vars(a),
                              rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), np=np.random.get_state()),
                              **(dict(lowpol=CS.lowpol.state_dict()) if getattr(CS, "lowpol", None) is not None else {}),
                              **(dict(opt_sil=opt_sil.state_dict()) if opt_sil is not None else {}))
                    tmp = rd / "ckpt.pt.tmp"; torch.save(ck, tmp); tmp.replace(rd / "ckpt.pt")
                    if it % 5 == 4:
                        torch.save(ck, rd / f"ckpt_it{it}.pt")
                if CS.arms is not None:
                    log(f"ARM BASES: {CS.arms.bases_file}")
                    tb.add_text("arm_bases", f"{CS.arms.bases_file} from iteration {start}", start)
                log(f"WM actor: trainable {sum(p_.numel() for p_ in wm_p) / 1e6:.2f} M in the WM (T2 + LoRA), {sum(p_.numel() for p_ in hd_p) / 1e6:.2f} M in the heads; "
                    f"sigma {a.sigma}, beta_kl {a.beta_kl}, lr {a.lr} / {lr_wm}")
                if a.det_check and start == 0:                                   # determinism check: identical actions twice
                    R_ = []
                    for rep in range(2):
                        E0, _, _ = run_wm(A, "init", 10 ** 9, seed=7, quota=1); R_.append({e_["env"]: e_["reward"] for e_ in E0})
                    com = [k for k in R_[0] if k in R_[1]]; dd = np.abs([R_[0][k] - R_[1][k] for k in com])
                    by_t = {}
                    for e_ in E0:
                        by_t.setdefault(e_["tpl"], []).append(e_["reward"])
                    tb_log("check", 0, dict(determinism_reward_diff_mean=float(dd.mean()), determinism_reward_diff_max=float(dd.max()),
                                            determinism_identical_share=float((dd < 1e-6).mean()), init_within_table_sd=float(np.mean([np.std(v) for v in by_t.values()])),
                                            init_between_table_sd=float(np.std([np.mean(v) for v in by_t.values()]))))
                    log(f"DETERMINISM (same seed, deterministic actions, x2): envs {len(com)}, |reward diff| mean {dd.mean():.4f}, max {dd.max():.4f}, "
                        f"identical {float((dd < 1e-6).mean()):.3f}; init-policy spread: within-table sd {np.mean([np.std(v) for v in by_t.values()]):.3f}, "
                        f"between-table sd {np.std([np.mean(v) for v in by_t.values()]):.3f}")
                for it in range(start, a.iters):
                    t0 = time.time()
                    if os.environ.get("FPR_W_NEAR"):                         # the closeness cost fades: W0 -> floor over FPR_NEAR_FADE iterations
                        _w0 = float(os.environ["FPR_W_NEAR"]); _fl = float(os.environ.get("FPR_W_NEAR_FLOOR", "0.1")); _nf = int(os.environ.get("FPR_NEAR_FADE", "15"))
                        _i0 = int(os.environ.get("FPR_NEAR_IT0", str(start)))
                        _w = max(_fl, _w0 * (1.0 - (it - _i0) / max(_nf, 1))); os.environ["FPR_W_NEAR_NOW"] = str(_w)
                        log(f"NEAR COST it {it}: weight {_w:.3f} per s (from {_w0} at it {_i0} to {_fl} over {_nf} its)"); tb.add_scalar("opt/w_near", _w, it)
                    if FC.FLAT_BEND and os.environ.get("FPC_BEND_BLEND_ITS"):         # flat envelope blended in over N iterations from FPC_BEND_BLEND_IT0
                        _bn = max(int(os.environ["FPC_BEND_BLEND_ITS"]), 1); _b0 = int(os.environ.get("FPC_BEND_BLEND_IT0", str(start)))
                        FC.BEND_BLEND = float(min(1.0, max(0.0, (it - _b0 + 1) / _bn))); log(f"BEND BLEND it {it}: flat envelope weight {FC.BEND_BLEND:.2f} (over {_bn} its from it {_b0})")
                    if getattr(CS, "lowpol", None) is not None:                     # the controller may learn during this rollout (FPL_UPDATE_TICKS)
                        CS.lowpol.learn_inline = True; CS.lowpol.log_fn = log
                    E, D, r = run_wm(A, "sample", 10 ** 9, seed=100 + it, quota=max(1, (a.episodes or N) // N))
                    if getattr(CS, "lowpol", None) is not None:
                        CS.lowpol.learn_inline = False
                    _bt = {}
                    for e_ in E:
                        _bt.setdefault(e_["tpl"], []).append(e_["reward"])
                    log(f"WM-RL it {it} spread: within-table sd {np.mean([np.std(v) for v in _bt.values()]):.3f}, between-table sd {np.std([np.mean(v) for v in _bt.values()]):.3f}")
                    ke = assign(E, D); grp = {}
                    for ep in E:
                        grp.setdefault(ep["tpl"], []).append(ep["reward"])
                    mu = {t: np.mean(v) for t, v in grp.items()}; sd = {t: max(np.std(v), SD_FLOOR) + 1e-3 for t, v in grp.items()}   # sd floor
                    # the per-take reward, normalised within the start-table group like the episode term
                    rcg = {}
                    for d, k in zip(D, ke):
                        if k >= 0 and d.get("rc") is not None:
                            rcg.setdefault(E[k]["tpl"], []).append(d["rc"])
                    rmu = {t: np.mean(v) for t, v in rcg.items()}; rsd = {t: max(np.std(v), SD_FLOOR) + 1e-3 for t, v in rcg.items()}
                    adv, Dk, kk_ = [], [], []
                    for d, k in zip(D, ke):
                        if k < 0 or len(grp[E[k]["tpl"]]) < 2:
                            continue
                        av = (E[k]["reward"] - mu[E[k]["tpl"]]) / sd[E[k]["tpl"]]
                        if d.get("rc") is not None and len(rcg[E[k]["tpl"]]) >= 2:
                            av += RC_W * (d["rc"] - rmu[E[k]["tpl"]]) / rsd[E[k]["tpl"]]
                        adv.append(av); Dk.append(d); kk_.append(k)
                    if adv:                                                      # logging only: mean advantage per decision type
                        _ty = np.array([int(d_["typ"]) for d_ in Dk]); _av = np.array(adv, np.float64)
                        _rc = np.array([d_["rc"] for d_ in Dk if d_.get("rc") is not None], np.float64)
                        log("ADV BY TYPE: " + "; ".join(f"{nm_} n {int((_ty == i_).sum())} mean adv {(_av[_ty == i_].mean() if (_ty == i_).any() else float('nan')):+.3f}"
                                                         for i_, nm_ in enumerate(("take", "wait", "move", "done")))
                            + f"; take rc mean {(_rc.mean() if len(_rc) else float('nan')):+.3f} (n {len(_rc)})")
                    # each episode weighs the same: its decisions share weight 1 / (its decision count), normalised to mean 1
                    if EP_WEIGHT and Dk:
                        cnt_ = {}
                        for k in kk_:
                            cnt_[k] = cnt_.get(k, 0) + 1
                        wts = np.array([1.0 / cnt_[k] for k in kk_]); wts = wts / wts.mean()
                    else:
                        wts = None
                    st = WA.grpo_update(A, ref, LP, opt, Dk, np.array(adv), beta_kl=a.beta_kl, sigma=a.sigma, log=log, weights=wts) if Dk else {}
                    if WA.SIL:                                               # self-imitation of this iteration's successful carries (T2 decoder)
                        tb_log("sil", it, WA.sil_update(A, LP, D, opt_sil, log=log, seed=it, n_carries=r.get("sil_carries", 0)))
                    S_ = summ(E)
                    tb_log("train", it, dict(S_, within_table_sd=float(np.mean([np.std(v) for v in _bt.values()])), between_table_sd=float(np.std([np.mean(v) for v in _bt.values()])),
                                             lift_cm_median=r["carry_lift_cm_median"] or float("nan"), flat_share=r["carry_flat_share"] or float("nan"),
                                             eps_per_s=r["eps_per_s"], decisions=len(Dk), take_ik_fail=r["take_ik_fail"], take_finger_table=r["take_finger_table"], hand_obj_abort=r["hand_obj_abort"],
                                             finger_table_ep_share=r["finger_table_ep_share"], route_edit_cm=r["route_edit_cm"], **{f"type_{k}": v for k, v in r["types"].items()}))
                    tb_log("opt", it, dict(st, lr_heads=a.lr, lr_wm=lr_wm, sigma=a.sigma, beta_kl=a.beta_kl))
                    if getattr(CS, "lowpol", None) is not None:                  # PPO on the controller's 0.1 s ticks of this rollout
                        tb_log("lowpol", it, CS.lowpol.update(log=log))
                    log(f"WM-RL it {it}: {json.dumps(summ(E))}; decisions {len(Dk)}; take_ik_fail {r['take_ik_fail']:.3f}; "
                        f"take_finger_table {r['take_finger_table']:.3f}; finger-table eps {r['finger_table_ep_share']:.3f}; route edit {r['route_edit_cm']:.2f} cm; aborts {r['abort_kinds']}; retracts {r['retracts']}; free-move aborts {r['move_ik_abort']}; reflex stops {r['reflex']}; hand_obj_abort {r['hand_obj_abort']}; gp_init_ok {r['gp_init_ok']:.3f}; types {r['types']}; lift {r['carry_lift_cm_median']}; "
                        f"flat {r['carry_flat_share']}; eps/s {r['eps_per_s']:.2f}; {time.time() - t0:.0f} s; {st}")
                    torch.save(A.state_dict(), rd / "wm_actor_rl.pt")
                    save_ckpt(it)
                    # Tripwires: done-at-start, the done / wait type shares, met_3d, abort kinds; stop if done-at-start > 50% after iteration 3
                    nty = max(sum(r["types"].values()), 1); tw = dict(done_at_start=S_["stop_at_start"], done_share=r["types"].get("done", 0) / nty,
                                                                     wait_share=r["types"].get("wait", 0) / nty, met3=S_.get("met3", float("nan")))
                    tb_log("tripwire", it, tw)
                    log(f"TRIPWIRE it {it}: {tw}; aborts {r.get('abort_kinds')}")
                    if it >= 3 and tw["done_at_start"] > 0.5:
                        log(f"TRIPWIRE STOP: done-at-start share {tw['done_at_start']:.2f} > 0.5 after iteration {it}")
                        break
            return
        if a.stage == "bc":
            n_eps = a.episodes or N
            E, D, r = run("baseline", 10 ** 9, quota=max(1, n_eps // N))
            log(f"BASELINE (train, BC data): {json.dumps(summ(E))}; {len(D)} decisions; eps/s {r['eps_per_s']:.2f}")
            torch.save(D, rd / "bc_data.pt")
            PO.bc_train(pol, D, steps=3000, dev=str(dev), log=log)
            torch.save(pol.state_dict(), rd / "policy_bc.pt")
            E2, _, r2 = run("greedy", 10 ** 9, quota=1)
            log(f"BC POLICY greedy (train): {json.dumps(summ(E2))}; eps/s {r2['eps_per_s']:.2f}")
            json.dump(dict(baseline=summ(E), bc=summ(E2)), open(rd / "bc.json", "w"), indent=1)
        elif a.stage == "variance":
            out = {}
            for nm, mode, seeds in (("greedy_repeat", "greedy", (0, 0)), ("sampled", "policy", (1, 2))):
                R = []
                for sd in seeds:
                    E, _, _ = run(mode, 10 ** 9, seed=sd, quota=1)
                    by = {}
                    for ep in E:
                        by.setdefault(ep["tpl"], []).append(ep["reward"])
                    R.append(by)
                same = [np.mean(R[0][t]) - np.mean(R[1][t]) for t in R[0] if t in R[1]]
                within = [np.std(R[0][t]) for t in R[0] if len(R[0][t]) > 1]
                out[nm] = dict(run_to_run_diff_abs_mean=float(np.mean(np.abs(same))), within_group_std_mean=float(np.mean(within)),
                               between_group_std=float(np.std([np.mean(v) for v in R[0].values()])))
                log(f"VARIANCE {nm}: {out[nm]}")
            json.dump(out, open(rd / "variance.json", "w"), indent=1)
        elif a.stage == "eval":
            E, _, r = run(a.mode, 10 ** 9, quota=1)
            log(f"EVAL {a.mode} ({a.split}): {json.dumps(summ(E))}; eps/s {r['eps_per_s']:.2f}")
        else:
            ref = copy.deepcopy(pol).eval()
            opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
            for it in range(a.iters):
                t0 = time.time()
                E, D, r = run("policy", 10 ** 9, seed=100 + it, quota=max(1, (a.episodes or N) // N))   # every env equally (no repeat bias)
                ke = assign(E, D)
                grp = {}
                for i, ep in enumerate(E):
                    grp.setdefault(ep["tpl"], []).append(ep["reward"])
                mu = {t: np.mean(v) for t, v in grp.items()}; sd = {t: np.std(v) + 1e-3 for t, v in grp.items()}
                adv, Dk = [], []
                for d, k in zip(D, ke):
                    if k < 0 or len(grp[E[k]["tpl"]]) < 2:
                        continue
                    adv.append((E[k]["reward"] - mu[E[k]["tpl"]]) / sd[E[k]["tpl"]]); Dk.append(d)
                st = PO.grpo_update(pol, ref, opt, Dk, np.array(adv), beta_kl=a.beta_kl, dev=str(dev), log=log) if Dk else {}
                log(f"RL it {it}: {json.dumps(summ(E))}; decisions {len(Dk)}; types {r['types']}; lift {r['carry_lift_cm_median']}; flat {r['carry_flat_share']}; "
                    f"edit {r['edit_cm_mean']:.2f} cm; eps/s {r['eps_per_s']:.2f}; {time.time() - t0:.0f} s; {st}")
                torch.save(pol.state_dict(), rd / "policy_rl.pt")


if __name__ == "__main__":
    main()
