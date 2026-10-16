"""Held-out evaluation: the trained policy (deterministic) against the frozen lift prior with a greedy nearest-item
pointer.

    python -m cleantables.rl.evaluate --checkpoint runs/rl/NAME/ckpt/last.pt --scene IMG_2546_t01 [--num-envs 128]
        [--seed 777] [--out result.json] [--record out/prefix]

Both passes start from the same layouts: the layouts are drawn first and every one in which an object still moves while
the robots hold still is re-drawn, so a robot touching an object early can never make a layout count as unstable.
Results (put-down rates with Wilson 95% intervals, makespan, contacts) go to a JSON file written atomically.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config

SPAWN_CHECK_STEPS = 14  # control steps the robots hold still while a fresh layout is checked
SPAWN_MOVE = 0.01
SPAWN_RETRIES = 3
VIDEO_EVERY = 2


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Proportion k/n with its Wilson score interval."""
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


@torch.inference_mode()
def stable_layouts(env, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """One layout per env in which no object moves over 1 cm while the robots hold still (re-drawn up to 3 times)."""
    from .scene import ARMS

    for a in ARMS:
        env.sim.action[a] = torch.zeros_like(env.sim.action[a])
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ids = env.arange
    env.reset_envs(ids, settle=True)
    xyz, yaw = env.last_layouts
    stats = dict(checked=0, redrawn=0, accepted_unstable=0, moved_by_object={})
    for attempt in range(SPAWN_RETRIES + 1):
        p0 = env.object_state()[0].clone()
        for _ in range(SPAWN_CHECK_STEPS):
            env.sim.step(env.zero_action)
        moved = (env.object_state()[0] - p0).norm(dim=-1)[ids]
        bad_obj = (moved > SPAWN_MOVE) | ~torch.isfinite(moved)
        bad = bad_obj.any(1)
        stats["checked"] += len(ids)
        for k in torch.nonzero(bad_obj.any(0))[:, 0].tolist():
            by_obj = stats["moved_by_object"]
            by_obj[env.names[k]] = by_obj.get(env.names[k], 0) + int(bad_obj[:, k].sum())
        ids = ids[bad]
        if len(ids) == 0:
            break
        if attempt == SPAWN_RETRIES:
            stats["accepted_unstable"] = len(ids)
            break
        stats["redrawn"] += len(ids)
        env.reset_envs(ids, settle=True)
        sel = ids.cpu().numpy()
        xyz[sel], yaw[sel] = env.last_layouts
    return xyz, yaw, stats


def run_pass(
    env,
    policy,
    prior,
    frames,
    kind: str,
    seed: int,
    action_clip: float,
    layouts=None,
    video=None,
    label: str = "",
    on_step=None,
) -> dict:
    """One episode per env. kind 'policy' = trained policy, 'baseline' = frozen prior + greedy pointer. With `layouts`
    every env starts from the given layout; without, layouts are drawn from `seed`."""
    from .policy import act
    from .scene import ARMS

    n, num_items, dev = env.num_envs, env.num_items, env.device
    if layouts is not None:
        for a in ARMS:
            env.sim.action[a] = torch.zeros_like(env.sim.action[a])
        xyz, yaw = layouts
        env.layout_override = lambda ids: (xyz[ids.cpu().numpy()], yaw[ids.cpu().numpy()])
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.reset_envs(env.arange, settle=True)
    obs = env.reset()
    env.layout_override = None
    finished: dict[int, dict | None] = {}
    invalid = 0.0
    t0 = time.time()
    for s in range(2 * env.max_steps + 5):
        a = act(policy, prior, frames, obs, env, action_clip, deterministic=True, greedy=(kind == "baseline"))
        if kind == "baseline":
            a["applied"] = torch.where(a["active"][..., None], a["prior"], torch.zeros_like(a["prior"]))
            a["action"] = a["applied"].reshape(n, 46)
        env.set_applied_action(a["applied"])
        obs, rew, done, info = env.step(a["action"], a["ptr"])
        if on_step is not None:
            on_step(s, a, obs, rew, done, info)
        if video is not None and s % VIDEO_EVERY == 0 and 0 not in finished:
            video.frame(label, s)
        invalid += float(info["invalid_pointer"].sum())
        for e in info.get("nan_ids", []):
            finished.setdefault(int(e), None)
        if "final" in info:
            fin = info["final"]
            for j, e in enumerate(fin["env"]):
                if int(e) not in finished:
                    finished[int(e)] = {
                        k: (v[j].tolist() if hasattr(v[j], "tolist") else v[j]) for k, v in fin.items() if k != "env"
                    }
        if s % 150 == 0:
            print(
                f"  [{kind}] step {s}: put down {int(env.placed.sum())}/{n * num_items}, finished {len(finished)}/{n}, "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
        if len(finished) == n:
            break
    scored = [e for e, f in finished.items() if f is not None]
    sc = torch.tensor(scored, dtype=torch.long, device=dev)
    in_container = items_in_container(env)[sc]
    per_class = {}
    for cname in sorted(set(env.item_cls_names)):
        cols = [i for i, x in enumerate(env.item_cls_names) if x == cname]
        per_class[cname] = dict(
            slots=len(scored) * len(cols),
            lifted=int(env.lifted[sc][:, cols].sum()),
            placed=int(env.placed[sc][:, cols].sum()),
            in_container=int(in_container[:, cols].sum()),
        )
    diag = dict(
        item_slots=len(scored) * num_items,
        lifted=int(env.lifted[sc].sum()),
        placed_end=int(env.placed[sc].sum()),
        in_container_end=int(in_container.sum()),
        per_class=per_class,
    )
    print(f"  [{kind}] {json.dumps(diag)}", flush=True)
    eps = [finished[e] for e in scored]
    return dict(
        kind=kind,
        seed=seed,
        episodes=len(eps),
        exploded=len(finished) - len(eps),
        diag=diag,
        env_ids=scored,
        wall_s=time.time() - t0,
        placed=[f["placed"] for f in eps],
        success=[bool(f["success"]) for f in eps],
        makespan_s=[f["makespan_s"] for f in eps],
        arm_contact_steps=[f["arm_contact_steps"] for f in eps],
        knocked=[f["knocked"] for f in eps],
        nontarget_steps=[f["nontarget_steps"] for f in eps],
        ret=[f["ret"] for f in eps],
        invalid_pointer=invalid,
    )


def items_in_container(env) -> torch.Tensor:
    """(N, I) item inside the footprint of its labelled container at the end of the episode, the judge used by the
    scripted baselines (axis-aligned footprint 1 cm inside the rim, below the rim + 5 cm)."""
    pc, _ = env.object_state()
    p, c = pc[:, env.item_obj], pc[:, env.item_dst]
    e = env.extents[env.item_dst][None]
    return (
        ((p[..., 0] - c[..., 0]).abs() < e[..., 0] / 2 - 0.01)
        & ((p[..., 1] - c[..., 1]).abs() < e[..., 1] / 2 - 0.01)
        & (p[..., 2] < c[..., 2] + e[..., 2] + 0.05)
        & (p[..., 2] > -0.02)
    )


def summarise(passes: list[dict], num_items: int) -> dict:
    placed = sum((r["placed"] for r in passes), [])
    success = sum((r["success"] for r in passes), [])
    n = len(placed)

    def mean(k):
        return float(np.mean(sum((r[k] for r in passes), [])))

    return dict(
        episodes=n,
        exploded_not_scored=sum(r["exploded"] for r in passes),
        items_placed=int(sum(placed)),
        items_total=n * num_items,
        items_in_container=sum(r["diag"]["in_container_end"] for r in passes),
        items_in_container_ci=wilson(sum(r["diag"]["in_container_end"] for r in passes), n * num_items),
        items_placed_rate_ci=wilson(int(sum(placed)), n * num_items),
        placed_mean=float(np.mean(placed)),
        all_placed=int(sum(success)),
        all_placed_ci=wilson(int(sum(success)), n),
        makespan_s_mean=mean("makespan_s"),
        arm_contact_steps_mean=mean("arm_contact_steps"),
        knocked_mean=mean("knocked"),
        nontarget_steps_mean=mean("nontarget_steps"),
        ret_mean=mean("ret"),
        invalid_pointer=sum(r["invalid_pointer"] for r in passes),
    )


def write_json_atomic(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="policy checkpoint (this package's or a research checkpoint)")
    ap.add_argument("--scene", required=True, help="held-out twin (folder name under data/twins/)")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--rounds", type=int, default=1, help="repeat with seeds seed + 1000 * round")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="", help="result JSON (default: <checkpoint dir>/../eval_<scene>_s<seed>.json)")
    ap.add_argument(
        "--record", default="", help="RTX video of env 0 of both passes to <record>_policy.mp4 / <record>_baseline.mp4"
    )
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(ap)
    args = ap.parse_args()
    if args.record:
        args.enable_cameras = True
    cfg = load_config(args.config)
    cfg.env = dataclasses.replace(cfg.env, spawn_check=False)  # layouts are checked before the robots act
    out = Path(args.out or Path(args.checkpoint).resolve().parent.parent / f"eval_{args.scene}_s{args.seed}.json")

    from .checkpoint import load_checkpoint
    from .ppo import build, prepare

    scene, sim_cfg = prepare(
        args.scene, args.num_envs, args.seed, video=bool(args.record), single_arm=cfg.env.single_arm
    )
    with launch_simulation(sim_cfg, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        env, prior, policy, frames = build(cfg, scene, sim_cfg, args.seed, autoreset=False)
        ck = load_checkpoint(args.checkpoint, policy, env.device)
        policy.eval()
        print(f"loaded {args.checkpoint} (it {ck['it']})", flush=True)
        recorder = None
        if args.record:
            from .video import VideoRecorder

            recorder = VideoRecorder(env)
        res = dict(
            checkpoint=str(args.checkpoint),
            it=ck["it"],
            scene=args.scene,
            num_envs=args.num_envs,
            seed=args.seed,
            items=env.item_names,
            item_classes=env.item_cls_names,
            config=cfg.to_dict(),
        )
        labels = dict(
            policy=f"policy ({Path(args.checkpoint).name}, it {ck['it']})",
            baseline="baseline: frozen lift prior + greedy pointer",
        )
        passes = {"policy": [], "baseline": []}
        res["layouts"] = []
        for r in range(args.rounds):
            seed = args.seed + 1000 * r
            xyz, yaw, stats = stable_layouts(env, seed)
            res["layouts"].append(dict(seed=seed, **stats))
            print(f"layouts seed {seed}: {json.dumps(stats)}", flush=True)
            for kind in ("policy", "baseline"):
                passes[kind].append(
                    run_pass(
                        env,
                        policy,
                        prior,
                        frames,
                        kind,
                        seed,
                        cfg.ppo.action_clip,
                        layouts=(xyz, yaw),
                        video=recorder,
                        label=labels[kind],
                    )
                )
                if recorder is not None:
                    recorder.write(f"{args.record}_{kind}.mp4")
        for kind, rs in passes.items():
            res[kind] = summarise(rs, env.num_items)
            res[kind + "_raw"] = rs
            print(f"eval {kind} {json.dumps(res[kind])}", flush=True)
        tg = env.teacher
        res["teacher_cost"] = dict(tg.cost, ms_per_call=1000 * tg.cost["seconds"] / max(1, tg.cost["calls"]))
        write_json_atomic(res, out)
        print(f"wrote {out}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
