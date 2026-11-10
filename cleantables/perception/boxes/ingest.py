"""One command from raw phone videos of box stacking to episodes, a trained stacking world model and check videos.

  envs/vla/bin/python -m cleantables.perception.boxes.ingest data/raw/boxes/            # every .MOV/.mp4 in the folder
  envs/vla/bin/python -m cleantables.perception.boxes.ingest a.MOV b.MOV --unseen IMG_0009,IMG_0010
  options: --t0/--t1 SECONDS (trim, test only)  --name NAME (single video)  --stages frames,sam3,depth,geom,segments,follow,events,sheets
           --no-train (stop after episodes)  --sam-device cpu  --force STAGE  --epochs 300  --pretrain-synth

Per video (outputs skipped when present; --force STAGE redoes one):
  frames   : 10 fps, long side 960 -> data/table_clean/_work/<V>/frames/<k>.jpg (+ frames_meta.json with the source, stride, trim)
  sam3     : SAM 3 'cardboard box', 'box', 'hand', 'masking tape', 'tape' every 5th frame (2 fps)   [SAM 3 env, GPU ~4.2 GB]
  depth    : MoGe-3 ViT-L every frame, one FOV per video                                              [vla env, GPU ~2.5 GB]
  geom     : table plane/frame, per detection 3D box pose + size                                     [CPU]
  segments : rest segments                                                                            [CPU]
  follow   : CoTracker3 from every segment end (24-frame windows, half resolution)                     [GPU if >= 3 GB free now, else CPU]
  events   : identities, moves, supports, carries, episodes -> data/twin_table/boxstack/<V>/events.json
  sheets   : proof images (top views + wireframe overlays) -> videos/twin_table/boxstack/
Then: data/twin_table/boxstack/episodes.json (all runs; --unseen ones split='unseen') -> stacking world model evaluation
(leave-one-run-out + unseen) -> final model runs/twin_table/boxstack/stackwm_all.pt -> pause-and-imagine check videos.
Every GPU stage first waits until the shared GPU has enough free memory (polls every 2 min; other processes are left alone).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cleantables.perception.scene.common import WORK

from cleantables.perception.boxes.common import MOGE_PATH, ROOT, SAM_PATH, SAM_PY, VLA_PY, log, vdir

STAGES = ["frames", "sam3", "sam3tools", "depth", "geom", "segments", "follow", "events", "tools", "snapshot", "scene", "sheets"]
NEED_MIB = dict(sam3=4700, depth=3200, follow=3000)   # own peak + ~0.5 GB, compared with the minimum free memory over 30 s


def gpu_free():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().split()[0])
    except Exception:
        return 0


def wait_gpu(need, name, max_wait_s=None):
    """Block until the shared GPU has >= need MiB free (minimum of 10 readings over 30 s). Returns False after max_wait_s."""
    t0 = time.time()
    first = True
    while True:
        fr = [gpu_free() for _ in range(10) if time.sleep(3) is None]   # other jobs' usage swings by ~2.5 GB: use the minimum
        if min(fr) >= need:
            return True
        if first:
            log(f"{name}: waiting for {need} MiB free GPU memory (free {min(fr)} MiB); polling every 2 min")
            first = False
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            return False
        time.sleep(120)


def extract(src, name, t0=None, t1=None, long=960):
    """Frames at ~10 fps (stride = round(fps / 10)), long side 960, landscape (portrait video is rotated 90 deg clockwise)."""
    import cv2
    d = WORK / name / "frames"
    mp = WORK / name / "frames_meta.json"
    if mp.exists():
        return json.load(open(mp))
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / 10)))
    if t0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    d.mkdir(parents=True, exist_ok=True)
    k = i = 0
    rot = False
    size = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if t1 is not None and (t0 or 0) + i / fps > t1:
            break
        if i % stride == 0:
            if f.shape[0] > f.shape[1]:
                f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE); rot = True
            s = long / max(f.shape[:2])
            f = cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) if s != 1 or f.shape[:2] != (540, 960) else f
            cv2.imwrite(str(d / f"{k}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            size = [f.shape[1], f.shape[0]]
            k += 1
        i += 1
    cap.release()
    if k == 0:
        raise RuntimeError(f"no frames decoded from {src}")
    m = dict(n=k, fps_video=fps, n_video_frames=i, stride=stride, size_wh=size, fps=fps / stride, src=str(src), t0=float(t0 or 0.0),
             rotated_portrait=rot)
    json.dump(m, open(mp, "w"))
    log(f"{name}: {k} frames at {fps / stride:.2f} fps from {src}" + (" (portrait: rotated 90 deg)" if rot else ""))
    return m


def sh(cmd, env_extra, name, logf, need=None, tries=4):
    """Run a worker; on CUDA out-of-memory (the shared GPU filled up meanwhile) wait for memory again and retry."""
    for t in range(tries):
        try:
            return _sh(cmd, env_extra, name, logf)
        except RuntimeError:
            tail = open(logf).read()[-3000:]
            if "out of memory" not in tail.lower() or t == tries - 1:
                raise
            log(f"{name}: CUDA out of memory (shared GPU) -> wait and retry ({t + 1}/{tries - 1})")
            time.sleep(180)
            if need:
                wait_gpu(need, name)


def _sh(cmd, env_extra, name, logf):
    env = dict(os.environ, **env_extra)
    env.setdefault("TORCH_HOME", str(ROOT / "cache/torch")); env.setdefault("HF_HOME", str(ROOT / "cache/hf"))
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as f:
        f.write(f"\n# {time.ctime()} {' '.join(cmd)}\n"); f.flush()
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{name} failed (rc {r.returncode}), see {logf}")


def done(stage, v):
    p = {"frames": WORK / v / "frames_meta.json", "sam3": WORK / v / "sam3_box.npz", "depth": vdir(v) / "depth10.npy",
         "geom": vdir(v) / "det3d.npz", "segments": vdir(v) / "segments.json", "follow": vdir(v) / "follow.npz",
         "events": vdir(v) / "events.json", "sheets": vdir(v) / "sheets.done",
         "sam3tools": WORK / v / "sam3_tools.npz", "tools": vdir(v) / "tools.done", "snapshot": vdir(v) / "snapshot.json",
         "scene": vdir(v) / "scene.done"}[stage]
    return p.exists()


def clear(stage, v):
    p = {"sam3": WORK / v / "sam3_box.npz", "depth": vdir(v) / "depth10.npy", "geom": vdir(v) / "det3d.npz",
         "segments": vdir(v) / "segments.json", "follow": vdir(v) / "follow.npz", "events": vdir(v) / "events.json",
         "sheets": vdir(v) / "sheets.done", "sam3tools": WORK / v / "sam3_tools.npz", "tools": vdir(v) / "tools.done",
         "snapshot": vdir(v) / "snapshot.json", "scene": vdir(v) / "scene.done"}.get(stage)
    if p and p.exists():
        p.unlink()


def run_video(src, name, a, unseen):
    lg = ROOT / "runs/twin_table/boxstack" / f"{name}.log"
    todo = a.stages.split(",") if a.stages else STAGES
    force = set(a.force.split(",")) if a.force else set()
    for st in force:
        clear(st, name)
    for st in STAGES:
        if st not in todo or (done(st, name) and st not in force):
            continue
        t0 = time.time()
        if st == "frames":
            extract(src, name, a.t0, a.t1)
        elif st == "sam3":
            if a.sam_device == "cuda":
                wait_gpu(NEED_MIB["sam3"], f"{name} sam3")
            sh([SAM_PY, "-m", "cleantables.perception.boxes.sam3_box", name, "--every", str(a.every), "--device", a.sam_device],
               dict(PYTHONPATH=f"{SAM_PATH}:{ROOT}", HF_HUB_OFFLINE="1", SAM3_THREADS="8"), "sam3", lg, NEED_MIB["sam3"])
        elif st == "sam3tools":                       # kitting: tools / parts packed into the kit box
            if a.sam_device == "cuda":
                wait_gpu(NEED_MIB["sam3"], f"{name} sam3 tools")
            sh([SAM_PY, "-m", "cleantables.perception.boxes.sam3_box", name, "--tools", "--every", str(a.every), "--device", a.sam_device],
               dict(PYTHONPATH=f"{SAM_PATH}:{ROOT}", HF_HUB_OFFLINE="1", SAM3_THREADS="8"), "sam3tools", lg, NEED_MIB["sam3"])
        elif st == "tools":
            from cleantables.perception.boxes.tools import stage_tools
            stage_tools(name)
            (vdir(name) / "tools.done").write_text("ok")
        elif st == "depth":
            wait_gpu(NEED_MIB["depth"], f"{name} depth")
            sh([VLA_PY, "-m", "cleantables.perception.boxes.depth_all", name], dict(PYTHONPATH=MOGE_PATH), "depth", lg, NEED_MIB["depth"])
        elif st == "geom":
            from cleantables.perception.boxes.tracking import stage_geom
            stage_geom(name)
        elif st == "segments":
            from cleantables.perception.boxes.tracking import stage_segments
            stage_segments(name)
        elif st == "follow":
            from cleantables.perception.boxes.tracking import stage_follow
            # GPU only if >= 3 GB is free over 30 s right now (CoTracker peak ~2 GB at half resolution), otherwise the CPU at once
            # (5-80 s per followed box depending on load); this stage does not wait for memory.
            ok = False if a.follow_cpu else wait_gpu(NEED_MIB["follow"], f"{name} follow", max_wait_s=0)
            if not ok:
                log(f"{name} follow: CoTracker on the CPU")
            stage_follow(name, dev="cuda" if ok else "cpu")
        elif st == "events":
            from cleantables.perception.boxes.events import build_video
            build_video(name, unseen=name in unseen)
        elif st == "snapshot":
            from cleantables.perception.boxes.snapshot import build
            build(name, unseen=name in unseen)
        elif st == "scene":
            from cleantables.perception.boxes.scene_export import export
            export(name)
            (vdir(name) / "scene.done").write_text("ok")
        elif st == "sheets":
            from cleantables.perception.boxes.sheets import video_sheets
            video_sheets(name)
            (vdir(name) / "sheets.done").write_text("ok")
        log(f"{name}: stage {st} done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--unseen", default="", help="comma-separated video names (file stems) with the unseen boxes")
    ap.add_argument("--name", default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--stages", default="")
    ap.add_argument("--force", default="")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--sam-device", default="cuda")
    ap.add_argument("--follow-cpu", action="store_true", help="CoTracker always on the CPU")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pretrain-synth", action="store_true")
    a = ap.parse_args()
    vids = []
    for x in a.inputs:
        p = Path(x)
        if p.is_dir():
            vids += sorted(q for q in p.iterdir() if q.suffix.lower() in (".mov", ".mp4", ".m4v"))
        else:
            vids.append(p)
    if not vids:
        sys.exit("no videos found")
    unseen = {u.strip() for u in a.unseen.split(",") if u.strip()}
    uf = (vids[0].parent / "unseen.txt")
    if uf.exists():
        unseen |= {l.strip() for l in open(uf) if l.strip()}
    names = []
    for p in vids:
        name = a.name if (a.name and len(vids) == 1) else p.stem
        names.append(name)
        log(f"=== {name} ({p}) {'UNSEEN boxes' if name in unseen else ''}")
        try:
            run_video(p, name, a, unseen)
        except Exception as ex:                                  # continue with the remaining videos
            import traceback
            traceback.print_exc()
            log(f"{name}: FAILED {ex!r} - continuing with the other videos")
    from cleantables.perception.boxes.events import collect
    eps = collect([n for n in names if (vdir(n) / "events.json").exists()], unseen)
    usable = [e for e in eps if e.get("usable")]
    log(f"{len(usable)} usable episodes ({sum(e['split'] == 'unseen' for e in usable)} unseen)")
    if a.no_train or len(usable) < 2:
        if len(usable) < 2:
            log("fewer than 2 usable episodes: not training (check the sheets in videos/twin_table/boxstack/)")
        return
    lg = ROOT / "runs/twin_table/boxstack/stackwm.log"
    extra = ["--pretrain-synth"] if a.pretrain_synth else []
    sh([VLA_PY, "-m", "cleantables.perception.boxes.stackwm", "eval", "--epochs", str(a.epochs)] + extra, dict(PYTHONPATH=str(ROOT)), "stackwm eval", lg)
    sh([VLA_PY, "-m", "cleantables.perception.boxes.stackwm", "train", "--epochs", str(a.epochs)] + extra, dict(PYTHONPATH=str(ROOT)), "stackwm train", lg)
    sh([VLA_PY, "-m", "cleantables.perception.boxes.imagine_box", "real", "--epochs", str(a.epochs)], dict(PYTHONPATH=str(ROOT)), "imagine", lg)
    log("ALL DONE: data/twin_table/boxstack/episodes.json, stackwm_eval.json, runs/twin_table/boxstack/stackwm_all.pt, videos/twin_table/boxstack/")


if __name__ == "__main__":
    main()
