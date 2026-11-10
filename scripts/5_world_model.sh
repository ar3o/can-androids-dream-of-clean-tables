#!/bin/bash
# Train the world model (3 seeds on all training folds + 2 extra fold-0 seeds) and evaluate it on held-out real video.
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"; set -e
for fs in "-1 0" "-1 1" "-1 2" "0 1" "0 2"; do set -- $fs
  $VLA_PY -m cleantables.worldmodel.train --cfg watch_notab --seeds $2 --fold $1 --steps 9000 --dev cuda
done
# held-out goal error against "stays put" and a direct-regression baseline
$VLA_PY -m cleantables.worldmodel.eval.regression_baseline train A2 0
CUDA_VISIBLE_DEVICES="" $VLA_PY -m cleantables.worldmodel.eval.predict run final_1nh $WM_RUNS2/watch_notab -1 0,1,2 mode=1nh
CUDA_VISIBLE_DEVICES="" $VLA_PY -m cleantables.worldmodel.eval.score main
CUDA_VISIBLE_DEVICES="" $VLA_PY -m cleantables.worldmodel.eval.tables
