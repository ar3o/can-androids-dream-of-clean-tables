#!/bin/bash
# Frozen V-JEPA 2 and DINOv2 features for every episode, and DINOv2 appearance tokens per object.
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"; set -e
$VLA_PY -m cleantables.features.video rows masks
$VLA_PY -m cleantables.features.video grids --dev cuda
WM_V2_FEATS=$WM_V2_FEATS_EXT $VLA_PY -m cleantables.features.video grids aug --aug_which crop --n_crop 1 --dev cuda
$VLA_PY -m cleantables.features.appearance
