"""Bimanual RL training over all training twins: data-parallel PPO ranks with scene rotation and held-out evaluations.

    python -m cleantables.rl.train --run NAME [--init CHECKPOINT] [--config configs/rl.yaml] [--max-blocks N]

Training runs in blocks of `block_iters` iterations. In each block every rank trains on its own twin; at the end rank 0
saves ckpt/last.pt, all ranks exit, and the next block moves every rank on to the next twins of the list and resumes. A
failed block is stopped (only its own processes) and retried from ckpt/last.pt. Every `eval_every` iterations the
checkpoint is evaluated on each held-out twin in the background (runs/rl/NAME/heldout/).
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from cleantables.paths import RL_RUNS

from .config import load_config

PORT_BASE, PORT_RANGE = 29500, 400
POLL_S = 5


class Rotation:
    def __init__(self, run: str, config: str | None, init: str, max_blocks: int):
        self.cfg = load_config(config)
        self.rot = self.cfg.rotation
        self.config_arg = ["--config", config] if config else []
        self.run, self.init, self.max_blocks = run, init, max_blocks
        self.dir = RL_RUNS / run
        for sub in ("blocks", "heldout", "ckpt"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "rotate.log"
        self.block = self._read_int("block_counter", 0)
        self.last_eval = self._read_int("last_eval", 0)
        self.evals: list[threading.Thread] = []

    def _read_int(self, name: str, default: int) -> int:
        p = self.dir / name
        return int(p.read_text()) if p.exists() else default

    def _write_int(self, name: str, value: int) -> None:
        (self.dir / name).write_text(str(value))

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _resume_from(self) -> str:
        last = self.dir / "ckpt" / "last.pt"
        return str(last) if last.exists() else self.init

    def run_block(self) -> bool:
        rot, b = self.rot, self.block
        port = PORT_BASE + random.randrange(PORT_RANGE)
        resume = self._resume_from()
        self.log(f"block {b}: resume '{resume}', port {port}")
        procs = []
        for r in range(rot.ranks):
            twin = rot.train_twins[(b * rot.ranks + r) % len(rot.train_twins)]
            self.log(f"  rank {r} -> {twin}")
            cmd = [
                sys.executable,
                "-m",
                "cleantables.rl.ppo",
                "--run",
                self.run,
                "--scene",
                twin,
                "--num-envs",
                str(rot.envs_per_rank),
                "--world",
                str(rot.ranks),
                "--rank",
                str(r),
                "--port",
                str(port),
                "--seed",
                str(1000 * b + r),
                "--stop-after",
                str(rot.block_iters),
                *self.config_arg,
            ]
            if resume:
                cmd += ["--resume", resume]
            log = open(self.dir / "blocks" / f"b{b}_r{r}_{twin}.log", "a")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        while True:
            codes = [p.poll() for p, _ in procs]
            failed = any(c not in (None, 0) for c in codes)
            if failed or all(c == 0 for c in codes):
                break
            time.sleep(POLL_S)
        for p, log in procs:
            if p.poll() is None:  # a rank that failed leaves the others waiting in an all-reduce
                p.kill()
            p.wait()
            log.close()
        return not failed

    def latest_iteration(self) -> int:
        import torch

        last = self.dir / "ckpt" / "last.pt"
        return int(torch.load(last, map_location="cpu", weights_only=False)["it"]) if last.exists() else -1

    def start_heldout_eval(self, it: int) -> None:
        ck = self.dir / "heldout" / f"model_it{it}.pt"
        shutil.copy(self.dir / "ckpt" / "last.pt", ck)

        def work():
            for twin in self.rot.heldout_twins:
                out = self.dir / "heldout" / f"it{it}_{twin}"
                cmd = [
                    sys.executable,
                    "-m",
                    "cleantables.rl.evaluate",
                    "--checkpoint",
                    str(ck),
                    "--scene",
                    twin,
                    "--num-envs",
                    str(self.rot.eval_envs),
                    "--seed",
                    str(self.rot.eval_seed),
                    "--out",
                    f"{out}.json",
                    *self.config_arg,
                ]
                with open(f"{out}.log", "w") as log:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True).returncode
                self.log(f"held-out eval it {it} {twin}: exit code {rc}")

        t = threading.Thread(target=work, daemon=False)
        t.start()
        self.evals.append(t)
        self.log(f"held-out evaluation started for it {it}")

    def loop(self) -> int:
        fails = 0
        while True:
            if not self.run_block():
                fails += 1
                self.log(f"block {self.block} failed ({fails} so far)")
                if fails >= self.rot.max_failures:
                    self.log("too many failures, stopping")
                    return 1
                time.sleep(self.rot.retry_wait_s)
                continue
            self.block += 1
            self._write_int("block_counter", self.block)
            it = self.latest_iteration()
            due = (it + 1) // self.rot.eval_every
            if due > self.last_eval:
                self.last_eval = due
                self._write_int("last_eval", due)
                self.start_heldout_eval(it)
            if self.max_blocks and self.block >= self.max_blocks:
                self.log("block limit reached")
                for t in self.evals:
                    t.join()
                return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; outputs go to runs/rl/<run>/")
    ap.add_argument("--config", default=None, help="YAML config (default configs/rl.yaml)")
    ap.add_argument("--init", default="", help="checkpoint for the first block when the run has no ckpt/last.pt yet")
    ap.add_argument("--max-blocks", type=int, default=0, help="stop after this many blocks in total (0 = never)")
    args = ap.parse_args()
    if args.init:
        args.init = str(Path(args.init).resolve())
    sys.exit(Rotation(args.run, args.config, args.init, args.max_blocks).loop())


if __name__ == "__main__":
    main()
