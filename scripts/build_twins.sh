#!/usr/bin/env bash
# Physics twins of every episode in data/episodes.json -> data/twins/<episode>/, plus RTX stills and contact sheets.
#   bash scripts/build_twins.sh            each step skips what already exists; FORCE=1 rebuilds everything
# Needs:  SIM_PY            python of the Isaac Sim / Isaac Lab environment (also used for the CPU steps)
#         SAM3_PY           python of the SAM 3 environment
#         SAM3_REPO         checkout of facebookresearch/sam3
#         SAM3_CHECKPOINT   official facebook/sam3 checkpoint (default: checkpoints/sam3/sam3.pt)
# The object library is built once (step 4, needs the plates). The tool meshes are then replaced by Hunyuan3D-2.1 shapes
# with scripts/build_twins_hunyuan.sh, after which this script is run again (FORCE=1) to rebuild the scenes.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SIM_PY:?set SIM_PY}" "${SAM3_PY:?set SAM3_PY}" "${SAM3_REPO:?set SAM3_REPO}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
DATA="${CLEANTABLES_DATA:-$PWD/data}"
TWINS="$DATA/twins"
FORCE_FLAG=${FORCE:+--force}
mkdir -p "$TWINS/logs"

VIDEOS=$($SIM_PY -c "from cleantables.twin.inputs import episodes; print(' '.join(sorted({e['video'] for e in episodes()})))")
EPISODES=$($SIM_PY -c "from cleantables.twin.inputs import episodes; print(' '.join(e['id'] for e in episodes()))")

# 1-3: static plate per video, SAM 3 person mask on it, inpainting
$SIM_PY -m cleantables.twin.plate $VIDEOS $FORCE_FLAG
PYTHONPATH="$SAM3_REPO:$PYTHONPATH" HF_HUB_OFFLINE=1 $SAM3_PY -m cleantables.twin.person_mask $VIDEOS \
  --checkpoint "${SAM3_CHECKPOINT:-${CLEANTABLES_CHECKPOINTS:-$PWD/checkpoints}/sam3/sam3.pt}"
$SIM_PY -m cleantables.twin.plate $VIDEOS --apply-person

# 4: object library, once
if [ ! -f "$TWINS/library/amazon_box/mesh.npz" ]; then
  $SIM_PY -m cleantables.twin.library
fi

# 5: one scene per episode (CPU)
$SIM_PY -m cleantables.twin.scene $FORCE_FLAG

# 6: RTX stills and settle check, one Isaac process per episode
for e in $EPISODES; do
  d="$TWINS/$e"
  if [ -z "${FORCE:-}" ] && [ -f "$d/side_by_side.jpg" ] && [ "$d/side_by_side.jpg" -nt "$d/scene.json" ]; then
    continue
  fi
  t0=$(date +%s)
  timeout 900 $SIM_PY -m cleantables.twin.render --id "$e" > "$TWINS/logs/render_$e.log" 2>&1 || echo "render $e failed, see $TWINS/logs/render_$e.log"
  echo "render $e: $(( $(date +%s) - t0 )) s"
done

# 7: contact sheets
$SIM_PY -m cleantables.twin.contact_sheet
