"""Renders the training data for the visual domain-randomisation test of the watch model: 12 rooms / lightings for training and
3 held out. Every (training fruit episode, training world) pair is replayed from the SAM 3 tracks in Isaac (cleantables.twin.dr_render:
the twin's scanned objects, no robot or hands, the room's hero camera, object masks from the renderer), rendering only the stills each
chosen keyframe's held clip needs (keyframes every --every-th tracked sample; stills at k, k - 1 .. k - 4 s).
WORLDS: 12 training rooms / lightings; HELDOUT: 3 rooms never used in training (a warehouse, a U-shape kitchen, an outdoor sky +
ground); --held renders the held-out episode in them (--held2: in HELDOUT2, 3 worlds unlike any training world). GPU: up to --par renders at once, each started only at <= 16,000 MiB total; a watchdog stops
the newest render above 22,016 MiB.
  $SIM_PY -m cleantables.worldmodel.finetune.render --out DR [--par 2] [--every 2] [--held]"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LED = Path(CT_DATA + "/logs/gpu_jobs.log")
TRAIN_EPS = [f"IMG_2516_t{i:02d}" for i in range(1, 11)]                      # the fruit episodes with twins, none held out
WORLDS = [("w01_wh_shelves", 101, "warehouse_multiple_shelves", "flood"), ("w02_wh_forklifts", 102, "warehouse_with_forklifts", "none"),
          ("w03_wh_small", 103, "small_warehouse", "none"), ("w04_k_island", 104, "kitchen_l_island", "lamp"),
          ("w05_k_peninsula", 105, "kitchen_peninsula", "none"), ("w06_k_lshape", 106, "kitchen_l_shape", "none"),
          ("w07_k_gshape", 107, "kitchen_g_shape", "lamp"), ("w08_office", 108, "office", "none"), ("w09_hospital", 109, "hospital", "none"),
          ("w10_simple", 110, "simple_room", "none"), ("w11_out_a", 111, "outdoor", "sun"), ("w12_out_b", 112, "outdoor", "none")]
HELDOUT2 = [("t1_torch", 301, "outdoor", "torch", ["--sky_name", "moonlit_golf"]), ("t2_neon", 302, "any", "neon", ["--no_room", "--zoom", "1.9"]),
            ("t3_topdown", 303, "outdoor", "none", ["--cam_el", "70"])]          # test worlds unlike any training world (night, coloured light, overhead view)
HELDOUT = [("h1_full_warehouse", 8, "full_warehouse", "flood"), ("h2_k_ushape", 4, "kitchen_u_shape", "lamp"), ("h3_outdoor", 1, "outdoor", "sun")]


def gpu():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def frames_for(ep, every):
    z = np.load(CT_DATA + "/wm_eval/wm_v3_tracks/tracks_all.npz", allow_pickle=True)
    kk = np.sort(z["snap_k"][z["group"] == ep].astype(int)); k0 = kk[0]
    keys = kk[::every]
    fr = sorted({int(max(0, k - k0 - 10 * h)) for k in keys for h in range(5)})
    return [int(k) for k in keys], fr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--held", action="store_true", help="the held-out test episode in the held-out worlds")
    ap.add_argument("--eps", default=""); ap.add_argument("--held2", action="store_true", help="the 3 test worlds unlike any training world")
    a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SA_SPP="2", SA_DENOISE="1", SA_AA="DLAA")
    jobs = []
    EXTRA = {}
    if a.held or a.held2:
        for row in (HELDOUT2 if a.held2 else HELDOUT):
            wn, seed, place, mood = row[:4]; EXTRA[wn] = row[4] if len(row) > 4 else []
            jobs.append(("IMG_2516_t11", wn, seed, place, mood, None))          # the whole clip (all frames) for the test
    else:
        eps = a.eps.split(",") if a.eps else TRAIN_EPS
        for wn, seed, place, mood in WORLDS:
            for ep in eps:
                keys, fr = frames_for(ep, a.every)
                d = OUT / ep / wn; d.mkdir(parents=True, exist_ok=True); json.dump(fr, open(d / "frames.json", "w")); json.dump(keys, open(d / "keys.json", "w"))
                jobs.append((ep, wn, seed, place, mood, d / "frames.json"))
    jobs = [j for j in jobs if not (OUT / j[0] / j[1] / f"camera_{j[0]}_seed{j[2]}.json").exists()]
    print(f"[dr-render] {len(jobs)} renders to do, {a.par} at a time", flush=True)
    run, t0, done, peak = [], time.time(), 0, 0
    while jobs or run:
        for p, j, ts in list(run):
            if p.poll() is not None:
                run.remove((p, j, ts)); done += 1
                with open(LED, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M')} | rtx-dr | dr_render {j[0]} {j[1]} | 0 | DONE (rc {p.returncode}, {time.time() - ts:.0f} s, peak total {peak} MiB)\n")
                print(f"[dr-render] {done} done ({j[0]} {j[1]} rc {p.returncode}, {time.time() - ts:.0f} s); {len(jobs)} left; {(time.time() - t0) / 60:.1f} min", flush=True)
        u = gpu(); peak = max(peak, u)
        if u > 22016 and run:                                                   # watchdog: stop the newest of this script's renders
            p, j, ts = run[-1]; p.kill(); print(f"[dr-render] WATCHDOG {u} MiB: stopped {j}", flush=True)
        if jobs and len(run) < a.par and u <= 16000:
            ep, wn, seed, place, mood, fl = jobs.pop(0); d = OUT / ep / wn; d.mkdir(parents=True, exist_ok=True)
            cmd = [os.environ["SIM_PY"], "-m", "cleantables.twin.dr_render", "--episode", ep, "--out", str(d), "--seeds", str(seed),
                   "--w", "960", "--h", "540", "--wm_input", "--wm_cam", "hero", "--same_objects", "--frames_only", "--zoom", "0.9", "--fps", "10",
                   "--place", place] + (["--mood", mood] if mood != "none" else []) + (["--frame_list", str(fl)] if fl else []) + EXTRA.get(wn, [])
            with open(LED, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')} | rtx-dr | dr_render {ep} {wn} | 6 | Isaac RTX: DR backbone-test render, {place} {mood}\n")
            p = subprocess.Popen(["timeout", "900"] + cmd, cwd=ROOT, env=env, stdout=open(d / "render.log", "w"), stderr=subprocess.STDOUT)
            run.append((p, (ep, wn), time.time())); time.sleep(20)               # let it allocate before the next start check
        time.sleep(2)
    print(f"[dr-render] all done in {(time.time() - t0) / 60:.1f} min, peak total {peak} MiB", flush=True)


if __name__ == "__main__":
    main()
