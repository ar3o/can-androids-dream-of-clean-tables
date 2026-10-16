# Installation

The pipeline uses three Python environments, because Isaac Sim, SAM 3 and the perception models pin different library versions:

| Environment | Used for | Variable the scripts read |
|:--|:--|:--|
| `sim` | twins, physics feedback, robot training (Isaac Sim 6.1 + Isaac Lab 3) | `SIM_PY` |
| `vision` | episodes, perception, world model (PyTorch, Transformers, MoGe-3) | `VISION_PY` |
| `sam3` | object and person masks, video tracking (SAM 3) | `SAM3_PY`, `SAM3_REPO`, `SAM3_CHECKPOINT` |

Developed and run on Ubuntu 22.04 with one RTX 4090 (24 GB) and NVIDIA driver 590. Everything runs on a single GPU; the robot training uses about 9 GB with 2 x 448 environments.

## 1. This repository

```bash
git clone https://github.com/<user>/androids-dream-of-clean-tables.git
cd androids-dream-of-clean-tables
export PYTHONPATH=$PWD
```

The code is used from the checkout (`PYTHONPATH`), so the same tree serves all three environments.

## 2. Simulation environment

```bash
python3.12 -m venv envs/sim && source envs/sim/bin/activate
pip install "isaacsim[all,extscache]==6.1.0" --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
(cd third_party/IsaacLab && git checkout 9eacda9ddc919c81d0d35a974ee081b6a607b865 && ./isaaclab.sh --install none)
pip install -r requirements/sim.txt
deactivate
export SIM_PY=$PWD/envs/sim/bin/python OMNI_KIT_ACCEPT_EULA=YES
```

## 3. Vision environment

```bash
python3.12 -m venv envs/vision && source envs/vision/bin/activate
pip install -r requirements/vision.txt
git clone https://github.com/microsoft/MoGe.git third_party/MoGe && pip install -e third_party/MoGe
deactivate
export VISION_PY=$PWD/envs/vision/bin/python
```

Model weights download from Hugging Face on first use: V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`), Qwen3-VL (`Qwen/Qwen3-VL-4B-Instruct`), MoGe-3 (`Ruicheng/moge-3-vitl`), DINOv2 and MiniLM. Set `HF_HOME` to choose the cache location.

Episode segmentation calls Gemini through the `google-genai` SDK and needs `GEMINI_API_KEY`. It is only needed for new videos.

## 4. SAM 3 environment

Follow the setup in [facebookresearch/sam3](https://github.com/facebookresearch/sam3), then:

```bash
export SAM3_PY=/path/to/sam3/env/bin/python
export SAM3_REPO=/path/to/sam3          # the checkout
export SAM3_CHECKPOINT=$PWD/checkpoints/sam3/sam3.pt
```

The checkpoint is the official `facebook/sam3` release (access is granted on its Hugging Face page).

## 5. Third-party weights for the robot

The starting policy for each arm is the Kuka-Allegro lift policy that ships with Isaac Lab. Train or download it with Isaac Lab (`Isaac-Lift-KukaAllegro`, rsl_rl, PhysX) and place the checkpoint at:

```
checkpoints/kuka_allegro_lift_physx.pt
```

## 6. Object meshes (optional)

Twins can be built with the mesh library that `cleantables.twin.library` makes from the videos. The released twins replace the tool and fruit meshes with Hunyuan3D-2.1 shapes; that step runs in a separate Hunyuan3D-2.1 environment (see `scripts/build_twins_hunyuan.sh` and the [Hunyuan3D-2.1 repository](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)).

## Docker

`docker/Dockerfile.sim` and `docker/Dockerfile.vision` describe the first two environments as containers:

```bash
docker build -f docker/Dockerfile.sim -t cleantables-sim .
docker build -f docker/Dockerfile.vision -t cleantables-vision .
docker run --gpus all -it --rm -v $PWD/data:/workspace/cleantables/data -v $PWD/runs:/workspace/cleantables/runs \
    -v $PWD/checkpoints:/workspace/cleantables/checkpoints cleantables-sim
```

The simulation image needs access to NVIDIA's Isaac Sim container on nvcr.io.

## Checking the install

```bash
$VISION_PY -m pytest tests -q
```

The tests need no GPU and no simulator.
