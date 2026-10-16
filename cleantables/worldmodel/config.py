"""Configuration of world-model training and of the physics-feedback stage (defaults = the released settings)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from cleantables.paths import REPO

WORLDMODEL_YAML = REPO / "configs" / "worldmodel.yaml"
PHYSICS_FEEDBACK_YAML = REPO / "configs" / "physics_feedback.yaml"


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    grad_clip: float = 3.0
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    d_model: int = 256
    layers: int = 6
    # augmentation
    p_mirror: float = 0.5
    p_rotate: float = 0.5
    p_scale: float = 0.3
    max_scale: float = 1.25
    p_replace: float = 0.25  # counterfactual re-placement of objects not involved in the move
    p_object_drop: float = 0.15
    # modality dropout: both / 3D only / the rest video only
    p_both: float = 0.5
    p_3d: float = 0.25
    # loss weights
    w_flow: float = 2.0
    w_prog: float = 1.0
    w_prog_abs: float = 3.0
    w_jepa: float = 1.0
    w_seen: float = 0.2
    # evaluation: path samples per ensemble (split evenly over the members)
    eval_samples: int = 12


@dataclass
class PhysicsFeedbackConfig:
    init_run: str = "imitation"
    out_run: str = "physics_feedback"
    # consecutive fine-tuning rounds; each restarts the optimiser and the random streams from the previous weights
    rounds: list = field(default_factory=lambda: [600, 1500])
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    scenes_per_step: int = 12
    samples_per_scene: int = 16
    keep_fraction: float = 0.25
    measured_layout_share: float = 0.2
    lr: float = 1e-4
    weight_decay: float = 0.01
    w_phys: float = 1.0
    seed: int = 0
    # scoring on the held-out twins
    score_layouts: int = 8
    score_samples: int = 8
    score_seed: int = 1


def _load(cls, path: Path):
    if not Path(path).exists():
        return cls()
    raw = yaml.safe_load(open(path)) or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"unknown keys in {path}: {sorted(unknown)}")
    return cls(**raw)


def load_train_config(path: Path = WORLDMODEL_YAML) -> TrainConfig:
    return _load(TrainConfig, path)


def load_physics_feedback_config(path: Path = PHYSICS_FEEDBACK_YAML) -> PhysicsFeedbackConfig:
    return _load(PhysicsFeedbackConfig, path)
