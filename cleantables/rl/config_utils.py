"""Small shared helpers: YAML configs, receipts, device / free-VRAM handling, seeding.

Stage CLIs take `--config configs/x.yaml [--smoke]`; `--smoke` overlays the config's `smoke:` block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # project root


def load_yaml(path: str | Path | None) -> dict:
    import yaml

    if path is None:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return yaml.safe_load(f) or {}


def merge_smoke(cfg: dict, smoke: bool) -> dict:
    """If smoke, overlay cfg['smoke'] (shallow-recursive) on top of the base config."""
    cfg = dict(cfg)
    over = cfg.pop("smoke", None) or {}
    if smoke:
        _deep_update(cfg, over)
    cfg["smoke"] = bool(smoke)
    return cfg


def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def std_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", type=str, default=None, help="YAML config (relative to project root ok)")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast run; applies the config's `smoke:` overrides")
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda (default: from config, else auto)")
    return ap


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def free_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1024**3
    except Exception:  # noqa: BLE001
        return None


def pick_device(requested: str | None, need_gb: float = 0.0, who: str = "") -> str:
    """cpu/cuda/auto. Warns (never kills anything) if the GPU is shared and has too little free VRAM."""
    import torch

    # The CPU host is shared: unbounded intra-op threads from several jobs thrash (a 12 s test took 680 s at load 50).
    torch.set_num_threads(int(os.environ.get("VLA_NUM_THREADS", 8)))
    if requested in (None, "auto"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        free = free_vram_gb()
        if free is None:
            warnings.warn("cuda requested but not available; falling back to cpu")
            return "cpu"
        if free < need_gb:
            warnings.warn(
                f"[{who}] only {free:.1f} GB VRAM free (< {need_gb:.1f} GB needed). The GPU is shared; "
                f"expect OOM. Use --device cpu, smaller batch_size, bf16, or grad_accum."
            )
        else:
            print(f"[{who}] cuda: {free:.1f} GB free VRAM (need ~{need_gb:.1f} GB)")
    return requested


def code_hash(pkg_dirs=("cleantables/rl", "cleantables/worldmodel")) -> str:
    h = hashlib.sha256()
    for d in pkg_dirs:
        for p in sorted((ROOT / d).rglob("*.py")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("torch", "transformers", "lerobot", "numpy", "av"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch

        out["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return out


def write_receipt(out_dir: Path, stage: str, cfg: dict, extra: dict | None = None, t0: float | None = None) -> Path:
    """receipt.json: config, code hash (repo is not a git checkout), seeds, wall time, versions, argv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        git_state = git.stdout.strip() if git.returncode == 0 else "not-a-git-repo"
    except Exception:  # noqa: BLE001
        git_state = "unavailable"
    rec = {
        "stage": stage,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "config": cfg,
        "git": git_state,
        "code_hash": code_hash(),
        "wall_time_s": None if t0 is None else round(time.time() - t0, 2),
        "versions": versions(),
        "env": {k: os.environ.get(k) for k in ("HF_HOME", "CUDA_VISIBLE_DEVICES")},
    }
    if extra:
        rec.update(extra)
    p = out_dir / "receipt.json"
    with open(p, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    return p
