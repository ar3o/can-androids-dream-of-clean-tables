#!/bin/bash
# Fine-tune the world model on domain-randomised Isaac renders (10 episodes x 12 worlds) and test it on held-out worlds.
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"; set -e
DR=$CLEANTABLES_DATA/wm_eval/wm_dr
$SIM_PY -m cleantables.twin.dr_library
$SIM_PY -m cleantables.worldmodel.finetune.render --out $DR/train
$SIM_PY -m cleantables.worldmodel.finetune.render --out $DR/held2 --par 1 --held2
$VLA_PY -m cleantables.worldmodel.finetune.encode --dr $DR/train --out $DR/feats
$VLA_PY -m cleantables.worldmodel.finetune.train --feats $DR/feats --out $DR/runs/watch_notab_dr
for w in $DR/held2/*/*/; do
  s=$(ls $w | grep -o 'camera_[A-Za-z0-9_]*_seed[0-9]*' | head -1 | sed 's/.*_\(seed[0-9]*\)/\1/'); [ -z "$s" ] && continue
  PYTHONPATH=$SAM3_REPO:$CT_ROOT HF_HUB_OFFLINE=1 ${SAM3_PY:-$VLA_PY} -m cleantables.worldmodel.finetune.perceive sam --dir $w --world $s
  $VLA_PY -m cleantables.worldmodel.finetune.perceive depth --dir $w --world $s
  $VLA_PY -m cleantables.worldmodel.finetune.perceive tokens --dir $w --world $s
  $VLA_PY -m cleantables.worldmodel.finetune.input_features --dir $w --world $s --fps 10 --out $w/feats.npz
  $VLA_PY -m cleantables.worldmodel.finetune.predict --video $w/feats.npz --tokens $w/perceived_$s.npz --out $w/preds_old.json
  WM_RUNS2=$DR/runs $VLA_PY -m cleantables.worldmodel.finetune.predict --run watch_notab_dr --video $w/feats.npz --tokens $w/perceived_$s.npz --out $w/preds_new.json
done
WM_RUNS2=$DR/runs $VLA_PY -m cleantables.worldmodel.finetune.predict --run watch_notab_dr --out $DR/preds_real_new.json
$VLA_PY -m cleantables.worldmodel.finetune.compare --dr $DR --held held2
