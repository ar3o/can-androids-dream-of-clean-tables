"""Settings of the bimanual RL stage; defaults are the final training run, configs/rl.yaml may override any field."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cleantables.paths import REPO

DEFAULT_CONFIG = REPO / "configs" / "rl.yaml"
ARMS = ("a", "b")


@dataclass
class EnvConfig:
    max_seconds: float = 30.0
    pointer_every: int = 30
    class_vocab: tuple[str, ...] = ("fruit", "tool")
    shuffle_prob: float = 0.5
    mass_dr: float = 0.15
    object_friction_dr: tuple[float, float] = (0.8, 1.2)
    robot_friction: tuple[float, float] = (0.5, 1.0)
    heavy_box_mass: tuple[float, float] = (2.5, 4.0)
    heavy_box_friction: tuple[float, float] = (0.8, 1.0)
    goal_xy: float = 0.05
    goal_above: float = 0.04
    w_time: float = 0.001
    w_arm_contact: float = 0.01
    w_nontarget: float = 0.02
    w_knock: float = 0.5
    w_proximity: float = 0.005
    spawn_check: bool = True
    single_arm: bool = False  # only arm a works; arm b is moved off the table
    success_in_destination: bool = True  # put down inside the destination the teacher chose
    teacher_run: str = "physics_feedback"
    teacher_seeds: tuple[int, ...] = (0, 1, 2)
    teacher_samples: int = 4
    replan_every: int = 90
    offpath_dist: float = 0.06
    offpath_gap: int = 30
    lookahead: float = 0.10
    path_points: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)


@dataclass
class PPOConfig:
    steps_per_env: int = 32
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4  # pointer and critic
    lr_actor: float = 3e-5  # a pretrained actor needs a small one
    lr_bounds: tuple[float, float] = (1e-5, 1e-3)
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    ent_pointer: float = 0.05
    ent_pointer_final: float = 0.01
    ent_decay_iters: int = 400
    pointer_min_entropy: float = 0.3  # nats; below it the pointer entropy is pushed back up
    pointer_min_entropy_w: float = 1.0
    init_std: float = 0.3
    std_bounds: tuple[float, float] = (0.01, 2.0)
    action_clip: float = 3.0
    shape_w: float = 10.0
    kl_prior: float = 1.0  # leash to the frozen lift policy, decaying to kl_prior_final
    kl_prior_final: float = 0.05
    kl_decay_iters: int = 300
    warmup_iters: int = 15  # critic-only iterations at the start of training
    ckpt_every: int = 20
    stagger: bool = True  # random first-episode clocks so envs do not all finish together
    iters: int = 100000


@dataclass
class RotationConfig:
    ranks: int = 2
    envs_per_rank: int = 448
    block_iters: int = 60
    train_twins: list[str] = field(default_factory=list)
    heldout_twins: list[str] = field(default_factory=list)
    eval_every: int = 1000
    eval_envs: int = 128
    eval_seed: int = 777
    max_failures: int = 20
    retry_wait_s: int = 60


@dataclass
class RLConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _fill(cls, values: dict):
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    kw = {}
    for k, v in values.items():
        default = getattr(cls(), k)
        kw[k] = tuple(v) if isinstance(default, tuple) else v
    return cls(**kw)


def load_config(path: str | Path | None = None) -> RLConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    return RLConfig(
        env=_fill(EnvConfig, raw.get("env", {})),
        ppo=_fill(PPOConfig, raw.get("ppo", {})),
        rotation=_fill(RotationConfig, raw.get("rotation", {})),
    )
