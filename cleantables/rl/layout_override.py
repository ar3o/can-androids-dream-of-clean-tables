"""Re-place objects that the humanoid stance (both arms on one side of the table) cannot reach. One rule for every scene and object:
  1  covered = the object's start top and (if it moves) its frozen goal reachable top-down by >= 1 arm (Franka IK, TCP 2 cm above the point)
  2  each uncovered object's start moves (xy only; z, yaw unchanged) to the nearest point (1 cm grid) whose footprint is inside the table,
     reachable by >= 1 arm (its top), and >= 1 cm clear of every other footprint; an object resting on / in another moves with it (the
     bottom-most one is re-placed, its riders keep their offsets)
  3  the frozen goals are recomputed from the new layout (the 3-seed ensemble judge) and rule 1 re-checked; at most 2 rounds
  4  a scene with anything still uncovered (or no admissible point) is excluded
Output: {meta, override: {scene: {slot: [x, y]}}, excluded: [scenes], stats}. apply(tpl, path) patches built templates (sim_env / chain_env,
FPC_LAYOUT_OVERRIDE); the WM tokens, meshes and physics are unchanged (the fast sim settles the new layout as usual).
  $SIM_PY -m cleantables.rl.layout_override --bases fastpath/franka_bases_humanoid.json --out fastpath/layout_override_humanoid.json"""
from cleantables import CT_DATA
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

W = Path(CT_DATA + "/wm_eval/wm_v2_world")


def apply(tpl, path, log=print):
    """Patch built templates' start layouts with an override file; -> the excluded scene names."""
    if not path:
        return []
    O = json.load(open(path)); n = 0
    for t in tpl["templates"]:
        for k, xy in O.get("override", {}).get(t["name"], {}).items():
            k = int(k); t["layout"][k][0], t["layout"][k][1] = xy
            t["objects"][k]["pos"][0], t["objects"][k]["pos"][1] = xy; n += 1
    log(f"layout override {path}: {n} objects re-placed; {len(O.get('excluded', []))} scenes excluded by it")
    return list(O.get("excluded", []))


def footprint(x, y, ext, yaw, pad=0.0):
    """Shapely polygon of a box footprint centred at (x, y) with extents ext, rotated by yaw, grown by pad."""
    from shapely.geometry import Polygon
    hx, hy = ext[0] / 2 + pad, ext[1] / 2 + pad
    c, s = math.cos(yaw), math.sin(yaw)
    return Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])


def main():
    import torch
    from cleantables.rl import sim_env as FE
    from cleantables.rl.chain_env import FrozenJudge
    from cleantables.worldmodel.physics.live_plan import LivePlanner
    from cleantables.rl.reach import reach
    ap = argparse.ArgumentParser(); ap.add_argument("--bases", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default="cpu"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--scenes", default=""); a = ap.parse_args()
    twr = os.environ["WM_TWIN_ROOT"]; scope = json.load(open(W / "fastpath" / "scope_clean.json"))
    Bj = json.load(open(a.bases))["scenes"]
    override, excluded, stats = {}, [], dict(scenes=0, objects=0, uncovered0=0, moved=0, excluded_objects=0)
    for split in ("train", "heldout"):
        names = [n for n in scope[split] if (Path(twr) / n / "wm_tokens.npz").exists()]
        if a.scenes:
            names = [n for n in names if n in a.scenes.split(",")]
        names = names[: (a.limit or None)]
        if not names:
            continue
        T = FE.build_templates(names, twin_root=twr)["templates"]
        LP = LivePlanner(names, device=a.dev)
        Km = max(t["n"] for t in T)
        lay = np.zeros((len(T), Km, 3), np.float32); yaw = np.zeros((len(T), Km), np.float32); ext = np.zeros((len(T), Km, 3), np.float32)
        for i, t in enumerate(T):
            n = t["n"]; L = np.asarray(t["layout"])
            lay[i, :n] = L[:n, :3] - [0, 0, 0.002]; yaw[i, :n] = [o["yaw"] for o in t["objects"]]; ext[i, :n] = np.asarray(t["extents"])[:n]
        for rnd in range(3):
            J = FrozenJudge(LP, np.arange(len(T)), lay, yaw)
            tg = J.targets(lay, yaw)
            top = lay.copy(); top[..., 2] += ext[..., 2]
            rows = [(i, k) for i, t in enumerate(T) for k in range(t["n"]) if T[i]["name"] not in excluded]
            S = np.array([top[i, k] for i, k in rows], np.float32); G = np.array([tg[i, k] for i, k in rows], np.float32)
            mv = np.array([bool(J.mov[i, k] and not J.static[i, k]) for i, k in rows])
            okS = np.zeros(len(rows), bool); okG = np.zeros(len(rows), bool)
            for g in (0, 1):
                bs = np.array([Bj[T[i]["name"]]["bases"][g] for i, _ in rows], np.float32)
                okS |= reach(bs, S, dev=a.dev); okG |= reach(bs, G, dev=a.dev)
            cov = okS & (okG | ~mv)
            unc = {}
            for (i, k), c, s_ in zip(rows, cov, okS):
                if not c:
                    unc.setdefault(i, []).append((k, bool(s_)))
            if rnd == 0:
                stats["scenes"] += len(T); stats["objects"] += len(rows); stats["uncovered0"] += int((~cov).sum())
            print(f"[{split}] round {rnd}: {int((~cov).sum())} uncovered objects in {len(unc)} scenes", flush=True)
            if not unc:
                break
            if rnd == 2:                                                        # rule 4
                for i in unc:
                    excluded.append(T[i]["name"]); stats["excluded_objects"] += len(unc[i])
                break
            for i, lst in unc.items():
                t = T[i]; n = t["n"]; tb = t["table"]; nm = t["name"]
                bases = np.array(Bj[nm]["bases"], np.float32)
                ok_scene = True
                for k, _ in lst:
                    # the bottom-most object under k (k rides on j if its centre is inside j's footprint and it sits higher)
                    j = k
                    for _ in range(3):
                        under = [m for m in range(n) if m != j and footprint(lay[i, m, 0], lay[i, m, 1], ext[i, m], yaw[i, m]).contains(
                            __import__("shapely.geometry", fromlist=["Point"]).Point(lay[i, j, 0], lay[i, j, 1])) and lay[i, j, 2] > lay[i, m, 2] + 0.005]
                        if not under:
                            break
                        j = under[0]
                    riders = [m for m in range(n) if m != j and footprint(lay[i, j, 0], lay[i, j, 1], ext[i, j], yaw[i, j]).contains(
                        __import__("shapely.geometry", fromlist=["Point"]).Point(lay[i, m, 0], lay[i, m, 1])) and lay[i, m, 2] > lay[i, j, 2] + 0.005]
                    group = [j] + riders
                    others = [footprint(lay[i, m, 0], lay[i, m, 1], ext[i, m], yaw[i, m], pad=0.01) for m in range(n) if m not in group]
                    xs = np.arange(tb["x"][0], tb["x"][1] + 1e-6, 0.01); ys = np.arange(tb["y"][0], tb["y"][1] + 1e-6, 0.01)
                    X, Y = np.meshgrid(xs, ys); C = np.stack([X.ravel(), Y.ravel()], -1)
                    C = C[np.argsort(np.linalg.norm(C - lay[i, j, :2], axis=1))]
                    from shapely.geometry import box
                    tbl = box(tb["x"][0], tb["y"][0], tb["x"][1], tb["y"][1])
                    adm = [c for c in C if tbl.contains(footprint(c[0], c[1], ext[i, j], yaw[i, j])) and
                           not any(o.intersects(footprint(c[0], c[1], ext[i, j], yaw[i, j])) for o in others)]
                    if not adm:
                        ok_scene = False; break
                    adm = np.array(adm[:4000], np.float32)
                    ztop = max(lay[i, m, 2] + ext[i, m, 2] for m in group)
                    P = np.concatenate([adm, np.full((len(adm), 1), ztop, np.float32)], 1)
                    r_ = np.zeros(len(adm), bool)
                    for g in (0, 1):
                        r_ |= reach(np.repeat(bases[g][None], len(adm), 0), P, dev=a.dev)
                    if not r_.any():
                        ok_scene = False; break
                    c = adm[int(np.argmax(r_))]; d = c - lay[i, j, :2]
                    for m in group:
                        lay[i, m, :2] += d
                        override.setdefault(nm, {})[str(m)] = [round(float(lay[i, m, 0]), 4), round(float(lay[i, m, 1]), 4)]
                    stats["moved"] += len(group)
                if not ok_scene:
                    excluded.append(nm); stats["excluded_objects"] += len(lst)
        del LP
    json.dump(dict(meta=dict(rule=__doc__.split("\n", 1)[1].split("Output:")[0].strip(), bases=a.bases), override=override, excluded=sorted(set(excluded)),
                   stats=stats), open(a.out, "w"), indent=1)
    print("STATS", stats, "excluded", sorted(set(excluded)), flush=True)


if __name__ == "__main__":
    main()
