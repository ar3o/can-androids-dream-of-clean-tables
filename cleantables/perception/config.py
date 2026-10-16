"""Settings of the perception stage, loaded from configs/perception.yaml (defaults = the release settings)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from cleantables import paths

CONFIG = paths.REPO / "configs" / "perception.yaml"


@dataclass
class EnvironmentConfig:
    sam3_python: str = "python"
    sam3_repo: str = ""
    sam3_checkpoint: str = str(paths.CHECKPOINTS / "sam3" / "sam3.pt")
    moge_pythonpath: list[str] = field(default_factory=list)


@dataclass
class FramesConfig:
    fps: int = 10
    width: int = 960
    height: int = 540
    jpeg_quality: int = 95


@dataclass
class DepthConfig:
    model: str = "Ruicheng/moge-3-vitl"
    stride: int = 3
    fov_frames: int = 8
    resolution_level: int = 9
    refine_steps: int = 1
    min_free_mib: int = 3200


@dataclass
class ObjectsConfig:
    vlm: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_gpu_min_free_mib: int = 11000
    merge_similarity: float = 0.90
    collapse_similarity: float = 0.75
    max_phrases: int = 6
    add_episode_objects: bool = True
    keyframes: int = 12
    keyframe_min_prob: float = 0.3
    keyframe_min_free_mib: int = 5000
    name_min_prob: float = 0.5
    name_keep_prob: float = 0.6
    max_vocabulary: int = 10


@dataclass
class VideoConfig:
    stride: int = 3
    min_free_mib: int = 6800
    text_on_gpu_min_free_mib: int = 8000
    state_on_gpu_min_free_mib: int = 8000
    feature_cache: bool = True
    window_min_frames: int = 600
    window_margin_s: float = 1.0


@dataclass
class VisionConfig:
    model: str = "facebook/dinov2-base"
    gpu_min_free_mib: int = 1500


@dataclass
class VideoFeaturesConfig:
    model: str = "facebook/vjepa2-vitl-fpc64-256"
    min_free_mib: int = 3000


@dataclass
class PerceptionConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    video_features: VideoFeaturesConfig = field(default_factory=VideoFeaturesConfig)


def load_config(path: Path | str = CONFIG) -> PerceptionConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    env = raw.get("environment") or {}
    for key, value in list(env.items()):
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            if "$" in expanded:  # variable not set: keep the dataclass default
                env.pop(key)
            else:
                env[key] = expanded
    sections = {}
    for f_ in fields(PerceptionConfig):
        section_type = f_.default_factory
        sections[f_.name] = section_type(**(raw.get(f_.name) or {}))
    return PerceptionConfig(**sections)
