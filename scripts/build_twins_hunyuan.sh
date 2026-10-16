#!/usr/bin/env bash
# Hunyuan3D-2.1 stage of the object library (flat tools). Run once, after `python -m cleantables.twin.library`.
#   bash scripts/build_twins_hunyuan.sh inputs     sightings + top-down crops (then list the chosen crop per tool in
#                                                  data/twins/hy3d/jobs.json: {"jobs": {"wrench": "twins/hy3d/inputs/..."}})
#   bash scripts/build_twins_hunyuan.sh shape      Hunyuan shapes (about 7 GB GPU)
#   bash scripts/build_twins_hunyuan.sh paint      optional Hunyuan paint (about 21 GB GPU; HY3D_PAINT_DIR required)
#   bash scripts/build_twins_hunyuan.sh library    library entries from the shapes
# Needs:  SIM_PY     python of the Isaac Sim environment (CPU steps)
#         VIDEO_PY   python whose OpenCV decodes the original .MOV videos (full-resolution crops)
#         HY3D_PY    python of the Hunyuan3D-2.1 environment, HY3D_REPO its checkout, HY3DGEN_MODELS its weights
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-}" in
  inputs)
    : "${SIM_PY:?set SIM_PY}" "${VIDEO_PY:?set VIDEO_PY}"
    $SIM_PY -m cleantables.twin.hunyuan select
    $VIDEO_PY -m cleantables.twin.hunyuan rect ;;
  shape|paint)
    : "${HY3D_PY:?set HY3D_PY}" "${HY3D_REPO:?set HY3D_REPO}" "${HY3DGEN_MODELS:?set HY3DGEN_MODELS}"
    export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [ "$1" = shape ]; then
      PYTHONPATH="$HY3D_REPO/hy3dshape:$HY3D_REPO:$PYTHONPATH" $HY3D_PY -m cleantables.twin.hunyuan_run shape "${@:2}"
    else
      PYTHONPATH="$HY3D_REPO/hy3dshape:$PYTHONPATH" $HY3D_PY -m cleantables.twin.hunyuan_run paint "${@:2}"
    fi ;;
  library)
    : "${SIM_PY:?set SIM_PY}"
    $SIM_PY -m cleantables.twin.hunyuan build "${@:2}" ;;
  *)
    echo "usage: $0 inputs|shape|paint|library [keys]" >&2; exit 2 ;;
esac
