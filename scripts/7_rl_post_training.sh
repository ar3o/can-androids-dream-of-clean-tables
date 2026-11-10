#!/bin/bash
# RL post-training of the world model in physics: the world model plans (GRPO), a shared two-arm controller acts (PPO),
# two frozen world-model copies judge; 2,032 randomised PhysX tables. Then the held-out comparison (75 tables x 2).
#   bash scripts/7_rl_post_training.sh train | eval
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"; source configs/rl_post_training.env
export WM_TWIN_ROOT=$CLEANTABLES_DATA/wm_eval/wm_v2_final_model/dr_root_clean
FP=$CLEANTABLES_DATA/wm_eval/wm_v2_world/fastpath
case "${1:-train}" in
  train) $SIM_PY -m cleantables.rl.train --stage wmrl --split train --copies 8 --iters 200 --disturb 0.3 \
                 --lr_wm 2e-6 --lr 3e-4 --beta_kl 0 --tag W2h_v4 ;;
  eval)  export FPC_CONT_EVAL=1 FPR_MEM_GB=5
         $SIM_PY -m cleantables.rl.train --stage wmeval --mode greedy --split heldout --copies 2 --disturb 0.3 --tag before
         $SIM_PY -m cleantables.rl.train --stage wmeval --mode greedy --split heldout --copies 2 --disturb 0.3 \
                 --actor $FP/stageA_W2h_v4/ckpt_it89.pt --tag after ;;
esac
