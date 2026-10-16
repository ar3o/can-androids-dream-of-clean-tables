"""World-model imagination replayed in a twin, without robots.

The frozen world model looks at the twin's object layout and the instruction, picks the object it expects to move
next, its destination and end point (the goal logic of the RL teacher: ensemble-mean destination, destination +
target-relative offset) and its imagined 3D path (median of the flow samples, re-anchored to that end). The object is
moved kinematically along the path, the layout is updated and the model plans again, until every task object has been
moved once.

  python -m cleantables.twin.imagine_replay --id IMG_2516_t13 [--shuffle 1]       (Isaac Sim + world model on the GPU)
Out runs/twin_imagine/<episode>_<layout>.mp4 (phone view with the plan drawn | overview) and .json (the plan).
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import add_launcher_args, launch_simulation

from cleantables.paths import RUNS, twin_dir
from cleantables.twin.config import load_config

IMAGINE_RUNS = RUNS / "twin_imagine"
SHUFFLE_TRIES = 20
SMOOTH = 5  # moving-average window over the median path


def parse_args():
    ap = argparse.ArgumentParser(description="Replay the world model's imagined plan in a twin, robots hidden.")
    ap.add_argument("--id", required=True, help="episode id, e.g. IMG_2516_t13")
    ap.add_argument(
        "--shuffle",
        type=int,
        default=-1,
        help="seed >= 0: random non-overlapping layout of the task objects (as the RL randomisation); default measured",
    )
    ap.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="world-model checkpoints (default: the teacher ensemble, runs/worldmodel/physics_feedback/final_seed*.pt)",
    )
    ap.add_argument("--config", default=None, help="config file (default configs/twin.yaml)")
    add_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    return args


def start_layout(teacher, scene: dict, shuffle: int):
    """(I, 3) bottom-centre positions: measured, or a random layout of the task objects."""
    import numpy as np

    from cleantables.worldmodel.teacher import shuffle_layout

    Z = teacher.Z
    ids = [o["id"] for o in scene["objects"]]
    pos = np.array([scene["objects"][ids.index(str(w))]["pos"] for w in Z["ids"]], np.float32)
    if shuffle >= 0:
        rng = np.random.default_rng(shuffle)
        task_idx = [i for i, r in enumerate(Z["role"]) if str(r) == "task"]
        for _ in range(SHUFFLE_TRIES):
            P = shuffle_layout(Z, scene, rng, task_idx)
            if P is not None:
                return P.astype(np.float32)
    return pos


def next_object(teacher, pos):
    """ensemble-mean P(next object) and task progress for a layout, plus what imagine() needs."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    I, dev = teacher.I, teacher.dev
    b = teacher.batch(torch.tensor(pos[None], device=dev), torch.zeros(1, I, device=dev), None)
    Es = teacher.encode(b)
    p_next, progress = [], []
    with torch.no_grad():
        for m, E in zip(teacher.models, Es, strict=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mover_logits, _ = m.pointers(E, torch.zeros(1, dtype=torch.long, device=dev))
                progress.append(float(torch.sigmoid(m.clip_progress_logit(E).float()).squeeze()))
            p_next.append(F.softmax(mover_logits.float(), -1)[0, :I])
    return b, Es, torch.stack(p_next).mean(0).cpu().numpy(), float(np.mean(progress))


def imagine(teacher, b, Es, i: int, samples: int):
    """destination (index, or I = table), its probability, end point (top centre) and imagined path of object i."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from cleantables.worldmodel.dataset import TABLE

    I, dev = teacher.I, teacher.dev
    mover = torch.tensor([i], device=dev)
    with torch.no_grad():
        probs = []
        for m, E in zip(teacher.models, Es, strict=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, tl = m.pointers(E, mover)
            tl = tl.float()
            probs.append(F.softmax(torch.cat([tl[:, :I], tl[:, TABLE : TABLE + 1]], -1), -1))
        p_dest = torch.stack(probs).mean(0)[0]
        d = int(p_dest.argmax())
        end = teacher.end_point(b, Es, mover, torch.tensor([d], device=dev))[0].cpu().numpy()
        target = torch.tensor([d if d < I else TABLE], device=dev)
        paths = []
        for m, E in zip(teacher.models, Es, strict=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, _, _ = m.heads_from(E, mover, target)
                paths.append(m.sample_paths_anchored(E, c, K=samples).float()[0, :, :, :3])
    path = torch.cat(paths, 0).median(0).values.cpu().numpy() + b["obj_geo"][0, i, :3].cpu().numpy()
    half = SMOOTH // 2
    path = np.stack(
        [np.convolve(np.pad(path[:, c], half, mode="edge"), np.ones(SMOOTH) / SMOOTH, mode="valid") for c in range(3)],
        1,
    )
    path = path + np.linspace(0, 1, len(path))[:, None] * (end - path[-1])
    return d, float(p_dest[d]), end, path


def plan_moves(teacher, pos, samples: int):
    """sequential imagined plan, updating pos in place. Yields one dict per move (log entry, path, end, top of the
    picked object, P(next) of the runner-up); the last one has only a log entry with the stop reason."""
    import numpy as np

    Z, I = teacher.Z, teacher.I
    names = [str(x) for x in Z["ids"]]
    roles = [str(r) for r in Z["role"]]
    heights = Z["extents"][:, 2]
    moved = set()
    for step in range(2 * I):
        b, Es, P, progress = next_object(teacher, pos)
        order = np.argsort(-P)
        cands = [j for j in order if roles[j] == "task" and j not in moved]
        if not cands:
            yield dict(
                log=dict(step=step, stop="every in-scope object has been moved to its WM goal", progress=progress)
            )
            return
        i = int(cands[0])
        order = [i] + [j for j in order if j != i]
        d, p_dest, end, path = imagine(teacher, b, Es, i, samples)
        entry = dict(
            step=step,
            obj=names[i],
            role=roles[i],
            p_next=float(P[i]),
            second=names[int(order[1])] if I > 1 else None,
            dest=names[d] if d < I else "table",
            p_dest=p_dest,
            end=end.round(3).tolist(),
            progress_before=progress,
        )
        yield dict(
            log=entry,
            index=i,
            path=path,
            end=end,
            top=pos[i] + np.array([0, 0, heights[i]], np.float32),
            second_p=float(P[int(order[1])]) if I > 1 else 0.0,
        )
        pos[i] = end - np.array([0, 0, heights[i]], np.float32)
        pos[i, 2] = max(pos[i, 2], 0.0)
        moved.add(i)


def main() -> None:
    args = parse_args()
    cfg_i = load_config(args.config)
    cfg_r, cfg_i = cfg_i.render, cfg_i.imagine
    folder = twin_dir(args.id)
    from cleantables.twin import cameras
    from cleantables.twin.sim_scene import TwinSceneEnv, load_scene, make_env_cfg

    S = load_scene(folder)
    cfg = make_env_cfg(S, folder, cfg_r.dome_intensity, cfg_r.key_intensity)
    cameras.add_cameras(cfg, S, folder, wrist_names=())
    with launch_simulation(cfg, args):
        import cv2
        import imageio.v2 as imageio
        import numpy as np

        from cleantables.worldmodel.teacher import WorldModelTeacher

        env = TwinSceneEnv(cfg, S)
        cameras.hide_ground()
        env.reset()
        cameras.set_static_cams(env, S, env.device)
        cameras.set_invisible([f"/World/envs/env_0/Robot{a.upper()}" for a in ("a", "b")])
        teacher = WorldModelTeacher(
            args.id, checkpoints=args.checkpoints, device=str(env.device), samples=cfg_i.samples
        )
        obj_index = [[o["id"] for o in S["objects"]].index(str(w)) for w in teacher.Z["ids"]]
        heights = teacher.Z["extents"][:, 2]
        pos = start_layout(teacher, S, args.shuffle)
        K = np.array(S["cameras"]["main"]["K"], np.float64)
        T_inv = np.linalg.inv(np.array(S["cameras"]["main"]["T_cam_to_table"], np.float64))
        img_w = S["cameras"]["main"]["wh"][0]

        def to_pixels(P, width):
            X = T_inv[:3, :3] @ np.asarray(P, np.float64).T + T_inv[:3, 3:4]
            uv = (K @ X)[:2] / np.maximum(X[2:3], 1e-6)
            return (uv.T * (width / img_w)).astype(int)

        def frame(lines, path=None, end=None, pick=None):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            for _ in range(2):
                env.sim.render()
            m, o = cameras.grab(env, "main"), cameras.grab(env, "overview")
            w = m.shape[1]
            if path is not None:
                cv2.polylines(m, [to_pixels(path, w).reshape(-1, 1, 2)], False, (0, 140, 255), 3, cv2.LINE_AA)
            if end is not None:
                cv2.circle(m, tuple(int(x) for x in to_pixels(end[None], w)[0]), 9, (60, 220, 60), 3, cv2.LINE_AA)
            if pick is not None:
                cv2.circle(m, tuple(int(x) for x in to_pixels(pick[None], w)[0]), 14, (0, 230, 255), 3, cv2.LINE_AA)
            im = np.hstack([cv2.resize(m, (800, 450)), cv2.resize(o, (800, 450))])
            lines = (list(lines) + [""] * 4)[:4]
            bar = np.zeros((26 * len(lines) + 10, im.shape[1], 3), np.uint8)
            for j, text in enumerate(lines):
                cv2.putText(
                    bar, text, (10, 24 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                )
            return np.vstack([bar, im])

        for i, k in enumerate(obj_index):
            env.place_object(k, pos[i])
        layout = f"random layout {args.shuffle}" if args.shuffle >= 0 else "measured layout"
        frames, log = [], []
        for move in plan_moves(teacher, pos, cfg_i.samples):
            entry = move["log"]
            log.append(entry)
            if "path" not in move:
                break
            i, path, end = move["index"], move["path"], move["end"]
            print(
                f"{args.id} move {entry['step'] + 1}: {entry['obj']} -> {entry['dest']} (p_next {entry['p_next']:.2f})",
                flush=True,
            )
            head = [
                f"{args.id}  '{S.get('instruction', '')}'   world-model imagination in the twin (no robot; {layout})",
                f"move {entry['step'] + 1}: picks '{entry['obj']}' (P(next) {entry['p_next']:.2f}; "
                f"2nd '{entry['second']}' {move['second_p']:.2f}) -> destination '{entry['dest']}' "
                f"(p {entry['p_dest']:.2f})   task progress "
                f"{entry['progress_before']:.2f}",
            ]
            for _ in range(int(cfg_i.pause_s * cfg_i.fps)):
                frames.append(
                    frame(
                        head
                        + ["plan: orange = imagined 3D path of its top, green = predicted end, yellow = picked object"],
                        path,
                        end,
                        move["top"],
                    )
                )
            n = int(cfg_i.move_s * cfg_i.fps)
            for t in range(n):
                q = t / max(1, n - 1) * (len(path) - 1)
                lo = int(q)
                p = path[lo] * (1 - (q - lo)) + path[min(lo + 1, len(path) - 1)] * (q - lo)
                env.place_object(obj_index[i], p - np.array([0, 0, heights[i]], np.float32))
                frames.append(
                    frame(
                        head + ["moving the object along the imagined path (kinematic, no physics, no robot)"],
                        path,
                        end,
                    )
                )
            rest = end - np.array([0, 0, heights[i]], np.float32)
            rest[2] = max(rest[2], 0.0)
            env.place_object(obj_index[i], rest)
        _, _, _, progress = next_object(teacher, pos)
        n_moved = sum(1 for e in log if "obj" in e)
        for _ in range(int(1.5 * cfg_i.fps)):
            frames.append(
                frame(
                    [
                        f"{args.id}  imagination finished: {n_moved} objects moved   "
                        f"final task progress {progress:.2f}",
                        "stop reason: " + (log[-1].get("stop", "move budget") if log else "-"),
                    ]
                )
            )
        IMAGINE_RUNS.mkdir(parents=True, exist_ok=True)
        tag = args.id + (f"_random{args.shuffle}" if args.shuffle >= 0 else "_measured")
        imageio.mimwrite(
            IMAGINE_RUNS / f"{tag}.mp4", [f[..., ::-1] for f in frames], fps=cfg_i.fps, quality=7, macro_block_size=None
        )
        checkpoints = [str(c) for c in args.checkpoints] if args.checkpoints else "teacher ensemble"
        (IMAGINE_RUNS / f"{tag}.json").write_text(
            json.dumps(dict(id=args.id, checkpoints=checkpoints, plan=log, final_progress=progress), indent=1)
        )
        print(f"{args.id}: {len(frames)} frames -> {IMAGINE_RUNS / tag}.mp4", flush=True)
        env.close()


if __name__ == "__main__":
    main()
