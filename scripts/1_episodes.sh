#!/bin/bash
# Phone videos (data/videos) -> tasks with instructions (Gemini) -> the episode list used by every later stage.
source "$(dirname "$0")/env.sh"; cd "$CT_ROOT"; set -e
$VLA_PY -m cleantables.episodes.segment "$@"            # needs GEMINI_API_KEY
python3 -m cleantables.episodes.catalog                  # -> data/labels/episodes_all.json
python3 -m cleantables.episodes.refine                   # -> data/labels/episodes_all_v2.json
