# Source me before running any stage.
#   CLEANTABLES_DATA  data, features, twins, runs (default: ./workdir)
#   SIM_PY            Python of the Isaac Sim 6.1 / Isaac Lab 3 environment
#   VLA_PY            Python of the vision environment (MoGe, V-JEPA 2, DINOv2, the world model)
#   SAM3_PY, SAM3_REPO  Python and source checkout for SAM 3
export CT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CLEANTABLES_DATA="${CLEANTABLES_DATA:-$CT_ROOT/workdir}"
export SIM_PY="${SIM_PY:-$CT_ROOT/envs/sim/bin/python}"
export VLA_PY="${VLA_PY:-$CT_ROOT/envs/vla/bin/python}"
export PYTHONPATH="$CT_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export HF_HOME="${HF_HOME:-$CLEANTABLES_DATA/cache/hf}" TORCH_HOME="${TORCH_HOME:-$CLEANTABLES_DATA/cache/torch}"
mkdir -p "$CLEANTABLES_DATA/logs"

# world-model data, features and runs
W=$CLEANTABLES_DATA/wm_eval/wm_v2_world
export WM_EVAL_OUT=$CLEANTABLES_DATA/wm_eval WM_DATA_ROOT=$W/data TASKS3_DATA=tasks3_watch_h23 \
       WM_V2_FEATS=$CLEANTABLES_DATA/wm_eval/wm_v2_seg/feats WM_V2_FEATS_EXT=$W/feats_new \
       WM_TEXT_DIR=$CLEANTABLES_DATA/wm_eval/wm_v2/text_8b WM_RUNS2=$W/runs_final
