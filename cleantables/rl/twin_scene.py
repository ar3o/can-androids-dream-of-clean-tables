"""Load a physics twin (data/twins/<episode>/scene.json) with every asset path resolved to the current data root."""

from __future__ import annotations

import json
from pathlib import Path

from cleantables.paths import twin_dir


def twin_asset(path: str) -> Path:
    """Map an asset path stored in scene.json (`.../<episode>/usd/<file>`) to the twin folder of this data root."""
    parts = Path(path).parts
    i = parts.index("usd")
    return twin_dir(parts[i - 1]) / Path(*parts[i:])


def load_twin(episode: str) -> dict:
    """scene.json of one twin. `episode` is the twin folder name (e.g. IMG_2516_t01); asset paths become absolute."""
    episode = episode.split("/")[-1]
    root = twin_dir(episode)
    scene = json.loads((root / "scene.json").read_text())
    scene["episode"] = episode
    scene["dir"] = str(root)
    for o in scene["objects"]:
        o["usd"] = str(twin_asset(o["usd"]))
        o["mesh"] = o["usd"].replace(".usda", ".obj")
    return scene
