#!/bin/bash
# Two Franka arms with NVIDIA's pretrained Isaac-Lift-Franka policy, each carrying objects along the frozen world model's paths.
#   bash scripts/8_lift_demo.sh IMG_2516_t14
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"
export WM_TWIN_ROOT=$CLEANTABLES_DATA/wm_eval/wm_v2_final_model/dr_root_clean_relink
O=outputs/lift_demo; mkdir -p $O
$SIM_PY -m cleantables.demos.lift --scene ${1:-IMG_2516_t14} --num_envs 1 --episodes 1 --release end --seconds 60 \
        --bowl_scale 1.6,1.3 --margin 0.16 --margin_now 0.10 --sphere_r 0.06 --seed 781 --clip $O/${1:-IMG_2516_t14} --out $O/${1:-IMG_2516_t14}.json
