"""Second training stage: physics feedback on the world model's own imagined carries in the twins.

For random layouts of the training twins the model imagines carries of an in-scope object (same goal logic as the robot
teacher), each carry is scored against the twins' mesh occupancy, and the flow head is trained towards the least
penetrating quarter of its samples (reward-weighted flow matching, cf. filtered self-imitation) while the imitation
losses on the video data keep running.

    python -m cleantables.worldmodel.physics_feedback finetune      # runs/worldmodel/imitation -> physics_feedback
    python -m cleantables.worldmodel.physics_feedback score         # penetration of the final members on held-out twins
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cleantables.paths import EPISODES, WM_RUNS

from .config import PhysicsFeedbackConfig, TrainConfig, load_physics_feedback_config, load_train_config
from .dataset import DATASET, Dataset, get_batch
from .model import MODE_3D, N_PATH, PATH_SCALE
from .occupancy import PenetrationScorer
from .teacher import WorldModelTeacher, imagine_carries, shuffle_layout
from .train import augment, compute_loss, load_models

log = logging.getLogger(__name__)
# twins never used for fine-tuning; the physics scores are reported on them
HELD_OUT_TWINS = {
    "IMG_2546_t01",
    "IMG_2547_t01",
    "IMG_2548_t01",
    "IMG_2516_t11",
    "IMG_2516_t12",
    "IMG_2516_t13",
    "IMG_2516_t14",
}
TWIN_TASKS = ("tools", "fruit")
COLLISION_VOXELS = 0.5  # a carry collides when more than this many voxels per step are inside something


def twin_episodes(kind: str = "train") -> list[str]:
    """Tools and fruit episodes: 'train' (fine-tuning twins), 'heldout' or 'all'."""
    tasks = json.load(open(EPISODES))["tasks"]
    eps = [t["id"] for t in tasks if t["task"] in TWIN_TASKS]
    if kind == "all":
        return eps
    return [e for e in eps if (e in HELD_OUT_TWINS) == (kind == "heldout")]


def task_objects(Z: dict) -> list[int]:
    return [i for i, r in enumerate(Z["role"]) if str(r) == "task"]


def _flow_loss_towards(m, items: list, gen: torch.Generator, device: str) -> torch.Tensor:
    """Flow-matching loss of the residual (object part) towards the kept low-penetration carries."""
    loss = 0.0
    for b, i, target, kept, _, _ in items:
        n = kept.shape[0]
        bb = {k: (v.expand(n, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in b.items()}
        mode = torch.full((n,), MODE_3D, dtype=torch.long, device=device)
        mover = torch.full((n,), i, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            E = m.encode(bb, mode)
            c, _, _ = m.heads_from(E, mover, target.expand(n))
        rel = (kept - bb["obj_geo"][:, i, :3][:, None]) / PATH_SCALE
        u = torch.linspace(0, 1, N_PATH, device=device)[None, :, None]
        x1 = torch.cat([rel - u * rel[:, -1:], torch.zeros_like(rel)], -1)
        x0 = torch.randn(x1.shape, device=device, generator=gen)
        t = torch.rand(n, device=device, generator=gen)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = m.velocity(E, c, xt, t)
        loss = loss + ((v.float() - (x1 - x0))[..., :3] ** 2).mean() / len(items)
    return loss


def finetune_round(
    m,
    steps: int,
    member: int,
    cfg: PhysicsFeedbackConfig,
    train_cfg: TrainConfig,
    data: Dataset,
    G: dict,
    episodes: list,
    logfile,
    tag: str,
    device: str = "cuda",
) -> None:
    rng = np.random.default_rng(cfg.seed + 17 * member)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + member)
    torch.manual_seed(cfg.seed + member)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    teachers, scorers = {}, {}
    t0 = time.time()
    for it in range(steps):
        m.eval()
        items = []
        for _ in range(cfg.scenes_per_step):
            ep = episodes[rng.integers(len(episodes))]
            if ep not in teachers:
                teachers[ep] = WorldModelTeacher(ep, checkpoints=[], device=device)
                scorers[ep] = PenetrationScorer(ep, device)
            teacher = teachers[ep]
            teacher.models = [m]
            Z = teacher.Z
            ids = [str(x) for x in Z["ids"]]
            task = task_objects(Z)
            P0 = (
                Z["start_pos"].copy()
                if rng.random() < cfg.measured_layout_share
                else shuffle_layout(Z, teacher.scene, rng, task)
            )
            if P0 is None:
                continue
            i = int(task[rng.integers(len(task))])
            paths, _, _, b, target = imagine_carries(teacher, P0, i, cfg.samples_per_scene)
            bottom = paths.clone()
            bottom[..., 2] -= float(Z["extents"][i, 2])
            pen = scorers[ep].penetration(scorers[ep].grid(ids, P0, i), ids[i], bottom)
            keep = pen.argsort()[: max(1, int(cfg.keep_fraction * len(pen)))]
            items.append((b, i, target, paths[keep], pen.mean().item(), pen[keep].mean().item()))
        if not items:
            continue
        m.train()
        l_phys = _flow_loss_towards(m, items, gen, device)
        idx = torch.as_tensor(np.random.default_rng(it).integers(0, data.n, train_cfg.batch_size), device=device)
        bt = augment(get_batch(G, idx), train_cfg, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = compute_loss(m, bt, train_cfg, gen)
        imitation = sum(losses.values())
        loss = cfg.w_phys * l_phys + imitation
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), train_cfg.grad_clip)
        opt.step()
        if it % 25 == 0 or it == steps - 1:
            msg = (
                f"{tag} it {it} phys_flow {float(l_phys.detach()):.3f} imitation {float(imitation.detach()):.3f} | "
                f"penetration of its samples {np.mean([x[4] for x in items]):.1f} vox "
                f"(kept {np.mean([x[5] for x in items]):.1f}) | {time.time() - t0:.0f} s"
            )
            log.info(msg)
            logfile.write(msg + "\n")
            logfile.flush()


def finetune(cfg: PhysicsFeedbackConfig, train_cfg: TrainConfig, dataset: Path = DATASET, device: str = "cuda") -> None:
    """Fine-tune each final imitation member through the configured rounds."""
    episodes = twin_episodes("train")
    data = Dataset(dataset)
    G = data.to(device)
    out = WM_RUNS / cfg.out_run
    out.mkdir(parents=True, exist_ok=True)
    for member in cfg.seeds:
        [m] = load_models([WM_RUNS / cfg.init_run / f"final_seed{member}.pt"], device)
        with open(out / f"finetune_seed{member}.log", "w") as logfile:
            for r, steps in enumerate(cfg.rounds, 1):
                finetune_round(
                    m, steps, member, cfg, train_cfg, data, G, episodes, logfile, f"seed {member} round {r}", device
                )
                torch.save(m.state_dict(), out / f"round{r}_seed{member}.pt")
        torch.save(m.state_dict(), out / f"final_seed{member}.pt")


@torch.no_grad()
def score(run_dir: Path, cfg: PhysicsFeedbackConfig, device: str = "cuda") -> dict:
    """Imagined carries of the final members on the held-out twins (measured layout + random ones): share of followed
    (median) carries that collide, mean penetration of the median carry, share of colliding samples, median lift."""
    torch.manual_seed(cfg.score_seed)
    rng = np.random.default_rng(cfg.score_seed)
    res = {}
    for ep in twin_episodes("heldout"):
        teacher = WorldModelTeacher(ep, checkpoints=[run_dir / f"final_seed{s}.pt" for s in cfg.seeds], device=device)
        scorer = PenetrationScorer(ep, device)
        Z = teacher.Z
        ids = [str(x) for x in Z["ids"]]
        H = torch.as_tensor(Z["extents"][:, 2], device=device)
        task = task_objects(Z)
        name = "fruit" if ep.startswith("IMG_2516") else "tools"
        for layout in range(cfg.score_layouts):
            P0 = Z["start_pos"].copy() if layout == 0 else shuffle_layout(Z, teacher.scene, rng, task)
            if P0 is None:
                continue
            for i in task:
                paths, _, _, _, _ = imagine_carries(teacher, P0, i, cfg.score_samples)
                G = scorer.grid(ids, P0, i)
                median = paths.median(0).values[None].clone()
                median[..., 2] -= H[i]
                pen_median = float(scorer.penetration(G, ids[i], median)[0])
                bottom = paths.clone()
                bottom[..., 2] -= H[i]
                pen_samples = scorer.penetration(G, ids[i], bottom)
                lift = float(
                    paths.median(0).values[:, 2].max() - max(paths[:, 0, 2].median(), paths[:, -1, 2].median())
                )
                r = res.setdefault(name, dict(n=0, coll=0, pen=[], samp=[], lift=[]))
                r["n"] += 1
                r["coll"] += int(pen_median > COLLISION_VOXELS)
                r["pen"].append(pen_median)
                r["samp"].append(float((pen_samples > COLLISION_VOXELS).float().mean()))
                r["lift"].append(lift)
        log.info("episode %s scored", ep)
    summary = {
        c: dict(
            n=r["n"],
            coll=r["coll"],
            pen=float(np.mean(r["pen"])),
            samp=float(np.mean(r["samp"])),
            lift=float(np.median(r["lift"])),
        )
        for c, r in res.items()
    }
    for c, r in summary.items():
        log.info(
            "%s: %d carries, followed carry collides %d (%.2f), penetration %.1f voxels/step, colliding samples %.2f, "
            "lift %.1f cm",
            c,
            r["n"],
            r["coll"],
            r["coll"] / r["n"],
            r["pen"],
            r["samp"],
            100 * r["lift"],
        )
    json.dump(summary, open(run_dir / "phys_score.json", "w"), indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Physics-feedback fine-tuning of the world model and its penetration score."
    )
    ap.add_argument("command", choices=["finetune", "score"])
    ap.add_argument("--run", default=None, help="score: run to score (default: the config's out_run)")
    ap.add_argument("--init", default=None, help="finetune: imitation run to start from")
    ap.add_argument("--out", default=None, help="finetune: output run")
    ap.add_argument("--seeds", default=None, help="comma-separated members (default from the config)")
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="finetune: one round of this many steps instead of the configured rounds",
    )
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default configs/physics_feedback.yaml)")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_physics_feedback_config(a.config) if a.config else load_physics_feedback_config()
    if a.seeds:
        cfg.seeds = [int(x) for x in a.seeds.split(",")]
    if a.command == "score":
        score(WM_RUNS / (a.run or cfg.out_run), cfg)
        return
    cfg.init_run = a.init or cfg.init_run
    cfg.out_run = a.out or cfg.out_run
    if a.steps:
        cfg.rounds = [a.steps]
    log.info("config %s", asdict(cfg))
    finetune(cfg, load_train_config(), a.dataset)


if __name__ == "__main__":
    main()
