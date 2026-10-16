"""Settings of the twin stage, read from configs/twin.yaml (defaults below are the same values)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from cleantables.paths import REPO

DEFAULT_CONFIG = REPO / "configs" / "twin.yaml"


@dataclass
class PlateConfig:
    depth_bin: float = 0.03
    min_support_frames: int = 6
    min_support_frac: float = 0.08
    min_colour_samples: int = 3
    max_frames: int = 300
    mask_dilation_px: int = 3
    min_static_depth_m: float = 0.6
    person_cloud_frames: int = 150
    person_inpaint_radius: int = 9


@dataclass
class PersonMaskConfig:
    prompts: list[str] = field(default_factory=lambda: ["person", "foot", "shoe"])
    min_prob: float = 0.25
    dilation_px: int = 8


@dataclass
class SceneConfig:
    start_window_frames: int = 15
    background_emissive: float = 600.0
    table_drop_m: float = 0.255
    base_gap_m: float = 0.15
    box_mass_kg: float = 3.0
    box_friction: tuple[float, float] = (0.9, 0.8)
    container_mass_kg: float = 2.5


@dataclass
class RenderConfig:
    settle_steps: int = 30
    dome_intensity: float = 220.0
    key_intensity: float = 350.0


@dataclass
class ImagineConfig:
    move_s: float = 2.0
    pause_s: float = 1.0
    fps: int = 15
    samples: int = 8


@dataclass
class TwinConfig:
    plate: PlateConfig = field(default_factory=PlateConfig)
    person_mask: PersonMaskConfig = field(default_factory=PersonMaskConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    imagine: ImagineConfig = field(default_factory=ImagineConfig)


def _fill(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    obj = cls(**values)
    if cls is SceneConfig:
        obj.box_friction = tuple(obj.box_friction)
    return obj


def load_config(path: Path | str | None = None) -> TwinConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    sections = {f.name: f.type for f in fields(TwinConfig)}
    classes = {
        "plate": PlateConfig,
        "person_mask": PersonMaskConfig,
        "scene": SceneConfig,
        "render": RenderConfig,
        "imagine": ImagineConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise KeyError(f"unknown config sections: {sorted(unknown)}")
    return TwinConfig(**{name: _fill(cls, raw.get(name) or {}) for name, cls in classes.items()})
