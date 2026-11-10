# Setup

Everything runs on one Linux machine with one NVIDIA GPU (I used an RTX 4090, 24 GB).

## Environments

Two Python 3.12 environments, pointed to by `SIM_PY` and `VLA_PY` (see `scripts/env.sh`):

| Variable | Contents | Used for |
|:--|:--|:--|
| `SIM_PY` | Isaac Sim 6.1 (pip, `pypi.nvidia.com`), Isaac Lab 3 from source, `requirements/sim.txt` | twins, randomised worlds, RL post-training, lift demo |
| `VLA_PY` | `requirements/vision.txt`, MoGe-3 from source | perception, features, world model |

A few published models run from their own checkouts, because their dependencies clash with the two environments above:

| Variable | Model |
|:--|:--|
| `SAM3_PY`, `SAM3_REPO`, `SAM3_CKPT` | SAM 3 (object masks and video tracking) |
| `FP_PY` | FoundationPose (6D object poses) |
| `H23_PY` | Hands23 (hand-object contact), checkout in `third_party/hands23` |
| `HY3D_PY`, `HUNYUAN3D_PAINT` | Hunyuan3D-2.1 shape and paint |
| `third_party/WiLoR` | WiLoR (3D hands) |

Episode mining needs a `GEMINI_API_KEY`.

## Data

The phone videos go in `data/videos/` and the episode labels the pipeline writes go in `data/labels/`. Everything
else (depth, tracks, features, twins, runs) is written under `$CLEANTABLES_DATA`, which defaults to `./workdir`
and needs a few hundred GB once every stage has run:

```
$CLEANTABLES_DATA/
  perception_v2/        depth, hands, contact, 6D poses per episode
  wm_eval/
    wm_v3_tracks/       3D object tracks of every episode
    wm_v2_seg/feats/    frozen V-JEPA 2 and DINOv2 features
    wm_v2_world/        world-model data, runs and the RL post-training runs (fastpath/)
    wm_dr/              randomised renders and the fine-tuned world model
```

`configs/rl_post_training.env` holds every setting of the RL run behind the reported numbers.
