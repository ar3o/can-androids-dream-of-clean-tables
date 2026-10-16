#!/usr/bin/env bash
# Perception for the given videos (default: every video in data/episodes.json), all steps, skipping finished ones.
#   scripts/run_perception.sh [VIDEO ...] [-- extra arguments for cleantables.perception.run]
# VISION_PY: Python of the perception environment; HF_HOME: Hugging Face cache with the model weights.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VISION_PY:-python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO/cache/hf}"

videos=()
while [[ $# -gt 0 && "$1" != "--" ]]; do videos+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
if [[ ${#videos[@]} -eq 0 ]]; then
    mapfile -t videos < <("$PY" -c "
import json
from cleantables import paths
print('\n'.join(sorted({e['video'] for e in json.loads(paths.EPISODES.read_text())['tasks']})))")
fi
cd "$REPO"
"$PY" -m cleantables.perception.run "${videos[@]}" "$@"
