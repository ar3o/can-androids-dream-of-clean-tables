"""Task-agnostic episode list: every Gemini episode from data/labels/tasks_auto.json with Gemini's own cut (start_s / end_s, no
padding), no task tag and no category exclusion. Gemini's object list and destination text are kept only under eval_only_* names so
they cannot be used as training labels. Also writes the per-episode coverage of perception outputs.
  python3 -m cleantables.episodes.catalog   -> data/labels/episodes_all.json + OUT/contact_moves/coverage.json
"""
from __future__ import annotations

from cleantables import CT_DATA
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(CT_DATA + "/perception_v2")
G = ROOT / "data/generic"
HOIST = OUT / "third_party/hoistformer/work/all45_6fps_L18"


def main():
    A = json.load(open(ROOT / "data/labels/tasks_auto.json"))["videos"]
    old = {t["id"] for t in json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]}
    eps, cov = [], []
    for x in A:
        v = x["video"]
        for t in x["tasks"]:
            eps.append(dict(id=t["id"], video=v, instruction=t["instruction"], start_s=t["start_s"], end_s=t["end_s"],
                            eval_only_gemini_objects=t.get("objects"), eval_only_gemini_destination=t.get("destination"),
                            eval_only_gemini_complete=t.get("complete"), eval_only_gemini_note=t.get("note"),
                            in_old_train_list=t["id"] in old))
            have = dict(tracks=(G / v / "tracks.pkl").exists(), sam3_video=(G / v / "sam3_video.npz").exists(),
                        vision=(G / v / "vision.npz").exists(), hands=(G / v / "hand_video.npz").exists(),
                        vjepa=(G / v / "vjepa.npz").exists(), hoist_raw=(HOIST / f"{t['id']}.npz").exists(),
                        fp=(OUT / "pose" / t["id"] / "fp_perframe_scalefix.npz").exists(),
                        twin=(ROOT / "data/twin_table/ep/_scalefix" / t["id"] / "scene.json").exists())
            cov.append(dict(id=t["id"], video=v, dur_s=round(t["end_s"] - t["start_s"], 1), new=t["id"] not in old, **have,
                            missing=[k for k, ok in have.items() if not ok]))
    json.dump(dict(source="data/labels/tasks_auto.json (Gemini 2.5 Flash, core cuts, no padding); all episodes, no task tag",
                   n=len(eps), n_new=sum(not e["in_old_train_list"] for e in eps), episodes=eps),
              open(ROOT / "data/labels/episodes_all.json", "w"), indent=1)
    json.dump(cov, open(OUT / "contact_moves/coverage.json", "w"), indent=1)
    for c in cov:
        if c["missing"]:
            print(c["id"], c["dur_s"], "s", "NEW" if c["new"] else "old", "missing:", ",".join(c["missing"]))
    print(len(eps), "episodes,", sum(c["new"] for c in cov), "new")


if __name__ == "__main__":
    main()
