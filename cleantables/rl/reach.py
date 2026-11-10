"""Reach coverage of the clean tables.

For every scene object, takes the start (top centre) and the frozen goal (the 3-seed ensemble's T1 at the start frame,
cleantables.rl.chain_env.FrozenJudge) and tests whether each Franka (bases from fastpath/franka_bases_clean.json) reaches it with
top-down IK, yaw free, TCP 2 cm above the point (as ArmTracker.reach). Writes the goals per object, used for the base search,
and prints the coverage per family.
  $SIM_PY -m cleantables.rl.reach --out fastpath/frozen_goals_all.json"""
from cleantables import CT_DATA
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from cleantables.robots import franka_kinematics as K
from cleantables.rl.chain_env import FrozenJudge
from cleantables.worldmodel.physics.live_plan import LivePlanner

W = Path(CT_DATA + "/wm_eval/wm_v2_world")


def reach(bases, pts, dz=0.02, iters=40, seeds=4, dev="cuda"):
    """bases (n, 4) x y z yaw, pts (n, 3) table frame -> ok (n,): top-down IK reaches the point with the TCP dz above it."""
    b = torch.as_tensor(bases, dtype=torch.float32, device=dev)
    Bt = K.base_T(b[:, 0], b[:, 1], b[:, 2], b[:, 3], device=dev)
    T = K.top_down(torch.as_tensor(pts, dtype=torch.float32, device=dev) + torch.tensor([0, 0, dz], device=dev), 0.0)
    q0 = torch.tensor(K.Q_HOME, device=dev).expand(len(pts), 7)
    out = []
    for i in range(0, len(pts), 4096):
        _, ok, _, _ = K.ik(torch.linalg.inv(Bt[i:i + 4096]) @ T[i:i + 4096], q0[i:i + 4096], yaw_free=True, iters=iters, n_seeds=seeds)
        out.append(ok.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(W / "fastpath/frozen_goals_all.json")); a = ap.parse_args()
    Bj = json.load(open(W / "fastpath/franka_bases_clean.json"))["scenes"]
    res, rows = {}, []
    from cleantables.rl import sim_env as FE
    twr = os.environ["WM_TWIN_ROOT"]; scope = json.load(open(W / "fastpath" / "scope_clean.json"))
    for split in ("train", "heldout"):
        names = [n for n in scope[split] if (Path(twr) / n / "wm_tokens.npz").exists()]
        T = FE.build_templates(names, twin_root=twr)["templates"]
        LP = LivePlanner(names, device="cuda")
        Km = max(t["n"] for t in T)
        xyz = np.zeros((len(T), Km, 3), np.float32); yaw = np.zeros((len(T), Km), np.float32); ext = np.zeros((len(T), Km, 3), np.float32)
        for i, t in enumerate(T):
            lay = np.asarray(t["layout"]); n = t["n"]
            xyz[i, :n] = lay[:n, :3] - [0, 0, 0.002]; yaw[i, :n] = [o["yaw"] for o in t["objects"]]; ext[i, :n] = np.asarray(t["extents"])[:n]
        J = FrozenJudge(LP, np.arange(len(T)), xyz, yaw)
        tg = J.targets(xyz, yaw)
        top = xyz.copy(); top[..., 2] += ext[..., 2]
        for i, t in enumerate(T):
            bs = np.array(Bj[t["name"]]["bases"], np.float32)
            for k, o in enumerate(t["objects"]):
                role = "moves" if (J.mov[i, k] and not J.static[i, k]) else ("static" if J.mov[i, k] else "no_target")
                rows.append(dict(split=split, scene=t["name"], fam=t["name"].split("_")[0], k=k, id=o["id"], role=role, start=top[i, k].tolist(),
                                 goal=tg[i, k].tolist() if role == "moves" else None, bases=bs))
        del LP
    S = np.array([r["start"] for r in rows], np.float32); G = np.array([r["goal"] if r["goal"] is not None else r["start"] for r in rows], np.float32)
    Bs = np.stack([r["bases"] for r in rows])
    ok = {}
    for g in (0, 1):
        ok[("s", g)] = reach(Bs[:, g], S); ok[("g", g)] = reach(Bs[:, g], G)
    for i, r in enumerate(rows):
        r["reach_start"] = [bool(ok[("s", 0)][i]), bool(ok[("s", 1)][i])]
        r["reach_goal"] = [bool(ok[("g", 0)][i]), bool(ok[("g", 1)][i])] if r["goal"] is not None else None
    out = {}
    for r in rows:
        out.setdefault(r["scene"], {})[f'{r["k"]}:{r["id"]}'] = dict(role=r["role"], start_top=[round(v, 4) for v in r["start"]],
                                                                       goal_top=[round(v, 4) for v in r["goal"]] if r["goal"] else None,
                                                                       reach_start=r["reach_start"], reach_goal=r["reach_goal"], split=r["split"])
    json.dump(dict(meta=dict(frame="twin table frame (as franka_bases_clean.json)", point="top centre of the object (start: spawned layout; "
                             "goal: frozen 3-seed ensemble T1 at the start frame)", roles="moves = has a frozen goal; static = the WM keeps it; "
                             "no_target = reference / not moved", reach="franka_kin top-down IK, yaw free, TCP 2 cm above, current bases"),
                   scenes=out), open(a.out, "w"), indent=0)
    # coverage: an object counts when its start and (if it moves) its goal are reached by >= 1 arm; the same arm need not do both
    def cov(sel):
        rs = [r for r in rows if sel(r)]
        one = np.mean([any(r["reach_start"]) and (r["reach_goal"] is None or any(r["reach_goal"])) for r in rs])
        both = np.mean([all(r["reach_start"]) and (r["reach_goal"] is None or all(r["reach_goal"])) for r in rs])
        same = np.mean([any(r["reach_start"][g] and (r["reach_goal"] is None or r["reach_goal"][g]) for g in (0, 1)) for r in rs])
        return len(rs), one, same, both
    print("family / split: objects, start+goal reachable by >= 1 arm, by ONE arm doing both, by both arms")
    for sp in ("train", "heldout"):
        for f in sorted({r["fam"] for r in rows}):
            n, o, s_, b = cov(lambda r: r["split"] == sp and r["fam"] == f) if any(r["split"] == sp and r["fam"] == f for r in rows) else (0, 0, 0, 0)
            if n:
                print(f"  {sp:7s} {f:5s} n={n:4d}  >=1 arm {o:.3f}  one arm both {s_:.3f}  both arms {b:.3f}")
        n, o, s_, b = cov(lambda r: r["split"] == sp); print(f"  {sp:7s} ALL   n={n:4d}  >=1 arm {o:.3f}  one arm both {s_:.3f}  both arms {b:.3f}")
    mv = [r for r in rows if r["role"] == "moves"]
    print("moving objects only: start >=1 arm %.3f, goal >=1 arm %.3f" % (np.mean([any(r["reach_start"]) for r in mv]), np.mean([any(r["reach_goal"]) for r in mv])))


if __name__ == "__main__":
    main()
