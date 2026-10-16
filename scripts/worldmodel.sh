#!/usr/bin/env bash
# World model end to end: dataset, imitation training (final model + 5 held-out folds), evaluation, physics feedback,
# its score on the held-out twins, and the held-out video. Run from the repository root in the perception/world-model
# environment ($PY); the twin occupancy step needs trimesh (the simulator environment has it).
set -euo pipefail
PY=${PY:-python}
$PY -m cleantables.worldmodel.build_dataset
for fold in -1 0 1 2 3 4; do
  $PY -m cleantables.worldmodel.train --fold "$fold"
done
$PY -m cleantables.worldmodel.evaluate --run imitation
$PY -m cleantables.worldmodel.teacher prepare
${SIM_PY:-$PY} -m cleantables.worldmodel.occupancy
$PY -m cleantables.worldmodel.physics_feedback finetune
$PY -m cleantables.worldmodel.physics_feedback score
$PY -m cleantables.worldmodel.teacher_checks goals --run imitation
$PY -m cleantables.worldmodel.teacher_checks containers --run imitation
$PY -m cleantables.worldmodel.video --run imitation IMG_2516_t05 IMG_2533_t01
