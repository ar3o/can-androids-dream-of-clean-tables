#!/bin/bash
# Episodes -> metric depth, table frame, SAM 3 object tracks, 3D hands, hand-object contact, 6D poses -> world-model data.
#   bash scripts/2_perception.sh data/videos/IMG_2516.MOV [...]
# FP_PY: Python with FoundationPose; H23_PY: Python with Hands23 (third_party/hands23).
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"; set -e
$VLA_PY -m cleantables.perception.ingest "$@" --only-stages frames,depth,objects,keyframes,geom,track,vtrack
PYTHONPATH=third_party/WiLoR:$PYTHONPATH $VLA_PY -m cleantables.perception.gpu_jobs wilor     # 3D hands (WiLoR)
$VLA_PY -m cleantables.perception.hands.place raw
$FP_PY  -m cleantables.perception.hands.place place
$FP_PY  -m cleantables.perception.pose_per_frame                                             # 6D poses (FoundationPose)
$H23_PY third_party/hands23/scripts/run_hands23.py && $H23_PY cleantables/perception/contact/hands23_masks.py
$VLA_PY -m cleantables.perception.contact.generic                                            # hand-object contact intervals
$FP_PY  -m cleantables.perception.contact.final
$VLA_PY -m cleantables.perception.tracks                                                     # -> wm_v3_tracks/tracks_all.npz
$VLA_PY -m cleantables.worldmodel.tracks_adapt --tracks $WM_EVAL_OUT/wm_v3_tracks/tracks_all.npz \
        --out tasks3_watch_h23 --feats_new $WM_V2_FEATS_EXT
