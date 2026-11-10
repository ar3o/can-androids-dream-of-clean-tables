"""Plans -> PhysX replay jobs -> generic physics cost. Runs in the VLA env (CPU is enough).

A plan = (twin, carried object, layout of every object, 33 root points start -> goal, yaw change). The replay itself runs in Isaac
(cleantables.worldmodel.physx_replay, `sim` env) as a subprocess. The cost uses only generic outcomes (no lift or height term, no names):
    cost = log1p(f_obj) + log1p(f_hand)            contact force on the carried object from other objects / on the hand proxy (x weight)
         + n_disp + 2 n_topple                     other objects displaced > 1 cm / tipped > 30 deg
         + settle_cm / 5 + settle_deg / 30          the carried object's motion from release to rest (dropped, toppled, slid)
         + 5 off_table                              it ended off the table
         + gap_cm / 5 + 0.5 max(0, L / S - 1)       start gap; path economy (as in physics.feedback)
Plans whose layout moved before the carry (pre_unstable) get cost NaN, since they say nothing about the plan.
The weights are hand-set; halving / doubling each is the sensitivity check (sens()).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.common import OUT, ROOT, log

PW = dict(f_obj=1.0, f_hand=1.0, disp=1.0, topple=2.0, settle_cm=1 / 5, settle_deg=1 / 30, off=5.0, gap_cm=1 / 5, econ=0.5)
SIM_PY = os.environ.get("SIM_PY", str(ROOT / "envs/sim/bin/python"))


# ------------------------------------------------------------------ plans
@torch.no_grad()
def followed(B, P0, i, yaw_off=None, K=4):
    """Path the robot follows (as carry_check.eval_path and the robot environment) and the yaw change (yaw head, else 0)."""
    from cleantables.worldmodel.physics.carry_check import eval_path
    from cleantables.worldmodel.bridge import long_side_yaw, wrap_half
    path, d = eval_path(B, P0, i, K)
    dy = 0.0
    if hasattr(B, "goal_yaw"):
        I = B.I
        y = float(B.goal_yaw(torch.as_tensor(P0[None], dtype=torch.float32), None if yaw_off is None else yaw_off[None],
                             [i], [d if d < I else -1])[0])
        if np.isfinite(y):
            dy = wrap_half(y - long_side_yaw(B.Z, i, 0.0 if yaw_off is None else float(yaw_off[i])))
    return path.cpu().numpy(), (d if d < B.I else -1), dy


def layout4(P0, yaw_off=None):
    L = np.zeros((len(P0), 4), np.float32); L[:, :3] = P0
    if yaw_off is not None:
        L[:, 3] = yaw_off
    return L


def write_jobs(plans, path):
    """plans: list of dict(twin, obj, layout (I,4), path (33,3) root points, dyaw)."""
    twins = sorted({p["twin"] for p in plans})
    n_max = max(len(p["layout"]) for p in plans)
    lay = np.zeros((len(plans), n_max, 4), np.float32)
    for j, p in enumerate(plans):
        lay[j, :len(p["layout"])] = p["layout"]
    np.savez_compressed(path, twins=np.array(twins), plan_twin=np.array([twins.index(p["twin"]) for p in plans]),
                        plan_obj=np.array([p["obj"] for p in plans]), layout=lay,
                        path=np.stack([np.asarray(p["path"], np.float32) for p in plans]), dyaw=np.array([p["dyaw"] for p in plans], np.float32))


QUEUE = OUT / "wm_v2" / "queue"


def _gpu_train_running():
    out = subprocess.run(["pgrep", "-f", "bin/python -m cleantables.worldmodel.train --cfg .* --dev cuda"], capture_output=True, text=True).stdout.split()
    return [p for p in out if p.isdigit() and int(p) != os.getpid()]


def run_physx(jobs, out, copies=16, timeout=3 * 3600, logf=None, pause_queue=True):
    """Run the Isaac replay as a subprocess and wait (resumable by the caller: skipped if `out` exists).
    pause_queue: the GPU training worker is paused (PAUSE_GPU) and its current model allowed to finish first, so Isaac and training
    never share the GPU (about 3.5 GB needed)."""
    if Path(out).exists():
        return np.load(out, allow_pickle=True)
    if os.environ.get("WM_QUEUE_PAUSED") == "1":               # the caller holds the queue paused for a whole phase
        pause_queue = False
    if pause_queue:
        QUEUE.mkdir(parents=True, exist_ok=True)
        (QUEUE / "PAUSE_GPU").touch()
        t0 = time.time()
        while _gpu_train_running():
            time.sleep(15)
        if time.time() - t0 > 20:
            log(f"PhysX: waited {time.time() - t0:.0f} s for the running GPU training job")
        subprocess.run([str(ROOT / "envs/vla/bin/python"), "-m", "cleantables.worldmodel.gpu_wait", "--need_mib", "3500"], check=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT)))
    try:
        return _run(jobs, out, copies, timeout, logf)
    finally:
        if pause_queue and (QUEUE / "PAUSE_GPU").exists():
            (QUEUE / "PAUSE_GPU").unlink()


BROKEN_PATTERNS = ("PxgCudaDeviceMemoryAllocator", "failed to allocate", "cudaErrorMemoryAllocation", "out of memory", "PHYSICS BROKEN")


def _run(jobs, out, copies, timeout, logf, attempts=3):
    """An Isaac run whose log shows a failed GPU allocation or 'PHYSICS BROKEN' is discarded and retried after waiting for free GPU
    memory: its scores are unreliable even when it exits normally."""
    lf = Path(logf or (str(out) + ".log"))
    for k in range(attempts):
        subprocess.run([str(ROOT / "envs/vla/bin/python"), "-m", "cleantables.worldmodel.gpu_wait", "--need_mib", "3500"], check=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT)))
        try:
            S = _run_once(jobs, out, copies, timeout, str(lf))
        except RuntimeError as e:
            S, err = None, str(e)
        txt = lf.read_text(errors="ignore") if lf.exists() else ""
        bad = [p for p in BROKEN_PATTERNS if p in txt]
        if S is not None and not bad:
            return S
        log(f"PhysX attempt {k + 1}/{attempts} discarded ({', '.join(bad) or err}); log kept as {lf}.attempt{k + 1}")
        if lf.exists():
            lf.rename(str(lf) + f".attempt{k + 1}")
        for f in (Path(out), Path(str(out) + ".part.npz")):
            if f.exists():
                f.unlink()
        time.sleep(60)
    raise RuntimeError(f"PhysX replay failed {attempts} times (see {lf}.attempt*)")


def _run_once(jobs, out, copies, timeout, logf):
    cmd = [SIM_PY, "-m", "cleantables.worldmodel.physx_replay", "--jobs", str(jobs), "--out", str(out) + ".part.npz", "--copies", str(copies)]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("CUDA_VISIBLE_DEVICES", None)                       # the caller runs the world model on the CPU; Isaac needs the GPU
    t0 = time.time()
    with open(logf or (str(out) + ".log"), "w") as lf:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    part = Path(str(out) + ".part.npz")
    if r.returncode != 0 or not part.exists():
        raise RuntimeError(f"PhysX replay failed (code {r.returncode}); see {logf or str(out) + '.log'}")
    S = np.load(part, allow_pickle=True)
    if not np.isfinite(S["f_obj"]).all():
        raise RuntimeError("PhysX replay incomplete")
    part.rename(out)
    log(f"PhysX: {len(S['f_obj'])} plans in {time.time() - t0:.0f} s")
    return np.load(out, allow_pickle=True)


# ------------------------------------------------------------------ cost
def geometry(paths, org):
    """paths (P,33,3) root points, org (P,3) the object's true start -> gap (cm), length / straight."""
    gap = 100 * np.linalg.norm(paths[:, 0] - org, axis=-1)
    L = np.linalg.norm(np.diff(paths, axis=1), axis=-1).sum(-1)
    S = np.maximum(np.linalg.norm(paths[:, -1] - org, axis=-1), 0.02)
    return gap, L / S


def cost(S, gap_cm, len_ratio, W=None):
    W = {**PW, **(W or {})}
    c = (W["f_obj"] * np.log1p(S["f_obj"]) + W["f_hand"] * np.log1p(S["f_hand"]) + W["disp"] * S["n_disp"] + W["topple"] * S["n_topple"]
         + W["settle_cm"] * S["settle_cm"] + W["settle_deg"] * S["settle_deg"] + W["off"] * S["off_table"]
         + W["gap_cm"] * gap_cm + W["econ"] * np.maximum(0.0, len_ratio - 1))
    return np.where(S["pre_unstable"] > 0, np.nan, c)


def outcome_table(S, gap_cm):
    """Plain-language shares for reports (evaluation)."""
    ok = S["pre_unstable"] == 0
    hit = (S["f_obj"] > 0.5) | (S["f_hand"] > 0.5)
    knock = S["n_disp"] > 0
    settled = (S["settle_cm"] < 2) & (S["settle_deg"] < 15) & (S["off_table"] == 0)
    clean = ~hit & ~knock & settled & (gap_cm < 1)
    f = lambda m: float(m[ok].mean()) if ok.any() else float("nan")
    return dict(n=int(ok.sum()), n_layout_unstable=int((~ok).sum()), collides=f(hit), moves_another_object=f(knock),
                knocks_over_another=f(S["n_topple"] > 0), settles_at_goal=f(settled), off_table=f(S["off_table"] > 0), clean=f(clean),
                f_obj_p50=float(np.median(S["f_obj"][ok])) if ok.any() else None, settle_cm_p50=float(np.median(S["settle_cm"][ok])) if ok.any() else None)
