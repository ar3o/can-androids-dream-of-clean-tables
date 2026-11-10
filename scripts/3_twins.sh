#!/bin/bash
# One Isaac Sim twin per episode (Hunyuan3D-2.1 meshes, background plate, phone camera), then the twin-vs-video check.
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"; set -e
$SIM_PY -m cleantables.twin.ep.mesh_inputs select && $VLA_PY -m cleantables.twin.ep.mesh_inputs crop
$HY3D_PY -m cleantables.twin.ep.mesh_run shape && $HY3D_PY -m cleantables.twin.ep.mesh_run paint      # HY3D_PY: Hunyuan3D-2.1 env
$SIM_PY -m cleantables.twin.ep.library_hy3d
$SIM_PY -m cleantables.twin.ep.plate $(ls data/videos | sed 's/\..*//')
TWIN_SCALE_MODE=per_object $SIM_PY -m cleantables.twin.ep.build
$SIM_PY -m cleantables.twin.evaluate && CUDA_VISIBLE_DEVICES="" $SIM_PY -m cleantables.twin.summary    # silhouette IoU, centre error
