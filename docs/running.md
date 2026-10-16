# Running the pipeline

Every stage reads and writes under `data/`, `runs/` and `checkpoints/` in the repository (override with `CLEANTABLES_DATA`, `CLEANTABLES_RUNS`, `CLEANTABLES_CHECKPOINTS`). Each command skips work that is already done unless told otherwise. Settings live in `configs/<stage>.yaml`, and their defaults are the ones used for the results in the README.

Set up the environments first ([installation.md](installation.md)), then from the repository root:

```bash
export PYTHONPATH=$PWD
```

## 0. Videos

Put the phone videos in `data/videos/` (`<name>.MOV` or `.mp4`). The camera should stay still during a video.

## 1. Episodes

```bash
$VISION_PY -m cleantables.episodes.segment          # Gemini Flash -> data/episodes_raw.json (needs GEMINI_API_KEY)
$VISION_PY -m cleantables.episodes.prepare          # margins + selection -> data/episodes.json
```

Which videos and task types are used is listed in `configs/episodes.yaml`.

## 2. Perception

```bash
VISION_PY=$VISION_PY bash scripts/run_perception.sh            # every video in data/episodes.json
```

This runs, per video: work frames, MoGe-3 depth, the table frame, Qwen3-VL object listing with a SAM 3 name check, SAM 3 video tracking, 3D boxes, hands, events and V-JEPA 2 features, then renders `data/perception/<video>/objects3d.mp4`. A single step can be run with `python -m cleantables.perception.run VIDEO --steps tracks`.

## 3. Twins

```bash
bash scripts/build_twins.sh                 # plates, person masks, mesh library, one scene per episode, RTX stills
bash scripts/build_twins_hunyuan.sh inputs  # optional: Hunyuan3D-2.1 meshes for the objects
bash scripts/build_twins_hunyuan.sh shape
bash scripts/build_twins_hunyuan.sh library
FORCE=1 bash scripts/build_twins.sh         # rebuild the scenes with the new meshes
```

Each twin is in `data/twins/<episode>/`; `side_by_side.jpg` shows the phone frame next to the twin rendered from the same camera.

## 4. World model

```bash
PY=$VISION_PY bash scripts/worldmodel.sh
```

The script builds the dataset, trains the final model and the five held-out folds (three seeds each), evaluates them, runs physics feedback and its score on the held-out twins, and renders held-out videos. The separate commands:

```bash
$VISION_PY -m cleantables.worldmodel.build_dataset
$VISION_PY -m cleantables.worldmodel.train --fold -1          # final model; --fold 0..4 for the held-out folds
$VISION_PY -m cleantables.worldmodel.evaluate --run imitation
$VISION_PY -m cleantables.worldmodel.teacher prepare          # world-model tokens for every twin
$SIM_PY    -m cleantables.worldmodel.occupancy                # 5 mm voxels of every twin mesh
$VISION_PY -m cleantables.worldmodel.physics_feedback finetune
$VISION_PY -m cleantables.worldmodel.physics_feedback score
$VISION_PY -m cleantables.worldmodel.video --run imitation IMG_2516_t05
```

To watch the world model move objects in a twin without a robot:

```bash
$SIM_PY -m cleantables.twin.imagine_replay --id IMG_2516_t13 --shuffle 1
```

## 5. Robot training

```bash
$SIM_PY -m cleantables.rl.train --run main
```

This runs two data-parallel workers with 448 environments each, rotates the training twins every 60 iterations, resumes from `runs/rl/main/ckpt/last.pt` after any interruption, and evaluates on the held-out twins every 1000 iterations.

## 6. Evaluation

```bash
$SIM_PY -m cleantables.rl.evaluate --checkpoint runs/rl/main/ckpt/last.pt --scene IMG_2546_t01
$SIM_PY -m cleantables.rl.evaluate --checkpoint runs/rl/main/ckpt/last.pt --scene IMG_2516_t13 --record runs/rl/main/videos/IMG_2516_t13
```

The policy and the baseline (NVIDIA's lift policy with greedy object assignment) start from the same layouts. Results are written as JSON next to the checkpoint, and `--record` saves videos from the phone camera with each arm's imagined path drawn in.
