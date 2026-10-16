"""Run the perception steps for a list of videos; every step skips the videos whose output already exists.

    python -m cleantables.perception.run VIDEO [VIDEO ...] [--steps frames,depth,...] [--config configs/perception.yaml]

Steps (in order):
  frames          10 fps work frames                                              frames/, frames_meta.json
  depth           MoGe-3 depth (subprocess with the MoGe-3 PYTHONPATH, GPU)        depth10.npy, depth10.json
  objects         preliminary table, Qwen3-VL object listing, name collapse    vlm_objects.json, candidate_phrases.json
  keyframes       SAM 3 image model on 12 keyframes (SAM 3 environment, GPU)        keyframe_dets.npz
  table           final table frame and depth scale; SAM 3 name check              det3d.npz, table.json, vocab.json
  video           SAM 3 video tracking of the names + 'hand' (SAM 3 env, GPU)      sam3_video.npz
  tracks          depth scale from tracked masks, identities, 3D boxes, hands  tracks.pkl, sam3_vinv.npz, hand_video.npz
  vision          DINOv2 crop features                                             vision.npz
  events          rests, moves, relations                                          events.json
  video_features  V-JEPA 2 clip tokens (GPU)                                       vjepa.npz
  visualize       3D box check video                                               objects3d.mp4
The object names of the 'objects' step are collapsed over all videos given in one call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cleantables import paths

from . import events, frames, objects, table, tracks, video_features, vision, visualize
from .config import CONFIG, PerceptionConfig, load_config
from .files import AGENT_PHRASES, WORK_FPS, gpu_free_mib, load_detections, n_frames, wait_for_gpu, write_json

STEPS = [
    "frames",
    "depth",
    "objects",
    "keyframes",
    "table",
    "video",
    "tracks",
    "vision",
    "events",
    "video_features",
    "visualize",
]
N_NAME_CHECK_FRAMES = 3


def done(video: str, name: str) -> bool:
    return (paths.perception_dir(video) / name).exists()


def run_worker(cmd: list[str], pythonpath: list[str], log: Path, what: str, offline: bool = False) -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([*pythonpath, str(paths.REPO)]))
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n# {time.ctime()} {' '.join(cmd)}\n")
        f.flush()
        result = subprocess.run(cmd, cwd=paths.REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{what} failed (exit code {result.returncode}); see {log}")


def log_file(name: str) -> Path:
    return paths.PERCEPTION / "logs" / f"{name}.log"


# ---------------------------------------------------------------- steps
def step_frames(videos, cfg: PerceptionConfig, config_path) -> None:
    todo = [v for v in videos if not done(v, "frames_meta.json")]
    if todo:
        with ProcessPoolExecutor(min(6, len(todo))) as pool:
            list(pool.map(frames.extract, todo, [cfg.frames] * len(todo)))


def step_depth(videos, cfg: PerceptionConfig, config_path) -> None:
    todo = [v for v in videos if not done(v, "depth10.npy")]
    if todo:
        wait_for_gpu(cfg.depth.min_free_mib, "depth")
        run_worker(
            [sys.executable, "-m", "cleantables.perception.depth", *todo, "--config", str(config_path)],
            cfg.environment.moge_pythonpath,
            log_file("depth"),
            "depth",
            offline=True,
        )


def step_objects(videos, cfg: PerceptionConfig, config_path) -> None:
    todo = [v for v in videos if not done(v, "vlm_objects.json")]
    if todo:
        device = "cuda" if gpu_free_mib() >= cfg.objects.vlm_gpu_min_free_mib else "cpu"
        vlm = objects.VisionLanguageModel(cfg.objects.vlm, device)
        for v in todo:
            if not done(v, "det3d.npz"):
                table.preliminary(v, cfg.depth.stride, cfg.objects.keyframes)
            objects.list_objects(v, vlm)
        del vlm
    if any(not done(v, "candidate_phrases.json") for v in videos):
        objects.collapse(videos, cfg.objects)


def step_keyframes(videos, cfg: PerceptionConfig, config_path) -> None:
    jobs = {}
    for v in videos:
        if not done(v, "keyframe_dets.npz"):
            jobs[v] = dict(
                frames=table.key_frames(v, cfg.depth.stride, cfg.objects.keyframes),
                phrases=objects.candidate_phrases(v) + AGENT_PHRASES,
                out=str(paths.perception_dir(v) / "keyframe_dets.npz"),
            )
    if jobs:
        job_file = paths.PERCEPTION / "logs" / "keyframe_jobs.json"
        write_json(jobs, job_file)
        wait_for_gpu(cfg.objects.keyframe_min_free_mib, "sam3 keyframes")
        env = cfg.environment
        run_worker(
            [
                env.sam3_python,
                "-m",
                "cleantables.perception.sam3_image",
                "--jobs",
                str(job_file),
                "--checkpoint",
                env.sam3_checkpoint,
                "--min-prob",
                str(cfg.objects.keyframe_min_prob),
            ],
            [env.sam3_repo],
            log_file("sam3_image"),
            "sam3 keyframes",
            offline=True,
        )


def name_check_subset(d: dict, candidates: list[str]) -> dict:
    """The candidate phrases' detections on 3 of the keyframes (the agent phrase is left out)."""
    ks = d["frames"]
    picked = {ks[i] for i in np.linspace(1, len(ks) - 2, N_NAME_CHECK_FRAMES).astype(int)} if len(ks) >= 4 else set(ks)
    sel = [
        i
        for i in range(len(d["prob"]))
        if int(d["frame"][i]) in picked and d["prompts"][int(d["prompt"][i])] in candidates
    ]
    return dict(
        frame=d["frame"][sel],
        prob=d["prob"][sel],
        masks=d["masks"][sel],
        prompts=candidates,
        frames=sorted(picked),
        prompt=np.array([candidates.index(d["prompts"][int(d["prompt"][i])]) for i in sel], np.int16),
    )


def step_table(videos, cfg: PerceptionConfig, config_path) -> None:
    for v in videos:
        if done(v, "vocab.json") and not table.is_preliminary(v):
            continue
        d = load_detections(paths.perception_dir(v) / "keyframe_dets.npz")
        info = table.fit(v, d)
        print(
            f"{v}: table at {info['camera_height_m']} m, tilt {info['tilt_deg']} deg, bumps {info['bump_frames']}",
            flush=True,
        )
        objects.select(v, name_check_subset(d, objects.candidate_phrases(v)), cfg.objects)


def episode_windows(video: str, margin_s: float) -> list[list[int]]:
    """Work-frame windows of the video's episodes in episodes.json, widened by the margin."""
    episodes = json.loads(paths.EPISODES.read_text())["tasks"]
    margin = int(round(margin_s * WORK_FPS))
    return [
        [int(round(e["start_s"] * WORK_FPS)) - margin, int(round(e["end_s"] * WORK_FPS)) + margin]
        for e in episodes
        if e["video"] == video
    ]


def step_video(videos, cfg: PerceptionConfig, config_path) -> None:
    jobs = {}
    for v in videos:
        if done(v, "sam3_video.npz"):
            continue
        n_tracked = len(range(0, n_frames(v), cfg.video.stride))
        windows = episode_windows(v, cfg.video.window_margin_s) if n_tracked > cfg.video.window_min_frames else None
        jobs[v] = dict(
            phrases=objects.vocabulary(v) + AGENT_PHRASES,
            windows=windows,
            out=str(paths.perception_dir(v) / "sam3_video.npz"),
        )
    if jobs:
        job_file = paths.PERCEPTION / "logs" / "video_jobs.json"
        write_json(jobs, job_file)
        wait_for_gpu(cfg.video.min_free_mib, "sam3 video")
        env = cfg.environment
        cmd = [
            env.sam3_python,
            "-m",
            "cleantables.perception.sam3_video",
            "--jobs",
            str(job_file),
            "--checkpoint",
            env.sam3_checkpoint,
            "--repo",
            env.sam3_repo,
            "--stride",
            str(cfg.video.stride),
            "--text-on-gpu-min-free-mib",
            str(cfg.video.text_on_gpu_min_free_mib),
            "--state-on-gpu-min-free-mib",
            str(cfg.video.state_on_gpu_min_free_mib),
        ]
        if not cfg.video.feature_cache:
            cmd.append("--no-feature-cache")
        run_worker(cmd, [env.sam3_repo], log_file("sam3_video"), "sam3 video", offline=True)


def step_tracks(videos, cfg: PerceptionConfig, config_path) -> None:
    for v in videos:
        if not done(v, "tracks.pkl"):
            table.refine_scale(v)
            tracks.build(v)


def step_vision(videos, cfg: PerceptionConfig, config_path) -> None:
    todo = [v for v in videos if not done(v, "vision.npz")]
    if todo:
        encoder = vision.Encoder(cfg.vision)
        for v in todo:
            vision.run(v, encoder)


def step_events(videos, cfg: PerceptionConfig, config_path) -> None:
    for v in videos:
        if not done(v, "events.json"):
            events.run(v)


def step_video_features(videos, cfg: PerceptionConfig, config_path) -> None:
    video_features.run(videos, cfg.video_features)


def step_visualize(videos, cfg: PerceptionConfig, config_path) -> None:
    for v in videos:
        if not done(v, "objects3d.mp4"):
            visualize.objects3d(v)


STEP_FUNCTIONS = {
    "frames": step_frames,
    "depth": step_depth,
    "objects": step_objects,
    "keyframes": step_keyframes,
    "table": step_table,
    "video": step_video,
    "tracks": step_tracks,
    "vision": step_vision,
    "events": step_events,
    "video_features": step_video_features,
    "visualize": step_visualize,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="video names (files in data/videos without the extension)")
    ap.add_argument(
        "--steps", default=",".join(STEPS), help="comma-separated subset of the steps, run in the order above"
    )
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()
    cfg = load_config(args.config)
    wanted = args.steps.split(",")
    unknown = set(wanted) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps {sorted(unknown)}")
    for step in [s for s in STEPS if s in wanted]:
        t0 = time.time()
        STEP_FUNCTIONS[step](args.videos, cfg, args.config.resolve())
        print(f"step {step}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
