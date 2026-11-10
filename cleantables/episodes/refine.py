"""Refined episode list (episodes_all_v2.json) from LLM reviews of every Gemini cut, instruction and between-episode stretch.
The first list (data/labels/episodes_all.json) is left unchanged. Every change records its reason; manual spot-check corrections
(SPOT) override a review item.
  python3 -m cleantables.episodes.refine   -> data/labels/episodes_all_v2.json + OUT/episode_review/CHANGES.md
"""
from __future__ import annotations

from cleantables import CT_DATA
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = Path(CT_DATA + "/perception_v2/episode_review")
SPOT = REV / "spot_check.json"          # {"<episode or reset id>": {"cut": [s, e]?, "instruction": "..."?, "drop": bool?, "note": "..."}}


def main():
    v1 = json.load(open(ROOT / "data/labels/episodes_all.json"))
    spot = json.load(open(SPOT)) if SPOT.exists() else {}
    vids = sorted({s["video"] for s in v1["segments"]})
    eps, missing, lines = [], [], []
    for v in vids:
        p = REV / "reviews" / f"{v}.json"
        g = [e for e in v1["episodes"] if e["video"] == v]
        if not p.exists():
            missing.append(v)
            for e in g:
                eps.append(dict(id=e["id"], video=v, kind="task", source="gemini (NOT reviewed)", instruction=e["instruction"],
                                start_s=e["start_s"], end_s=e["end_s"], reviewed=False))
            continue
        r = json.load(open(p))
        lines.append(f"\n### {v}\n\n| episode | Gemini cut | new cut | instruction (old → new) | notes |\n|---|---|---|---|---|")
        for x in r["episodes"]:
            s, e_ = x["new_cut"]
            ins = x["new_instruction"] or x["old_instruction"]
            sp = spot.get(x["id"], {})
            if sp.get("drop"):
                lines.append(f"| {x['id']} | {x['old_cut']} | DROPPED (spot check) | | {sp.get('note', '')} |"); continue
            if "cut" in sp:
                s, e_ = sp["cut"]
            ins = sp.get("instruction", ins)
            eps.append(dict(id=x["id"], video=v, kind="task", source="gemini, reviewed by Claude", instruction=ins, start_s=float(s),
                            end_s=float(e_), reviewed=True, uncertain=bool(x.get("uncertain")), uncertain_note=x.get("uncertain_note", ""),
                            change=dict(old_cut=x["old_cut"], start_reason=x.get("start_reason"), end_reason=x.get("end_reason"),
                                        old_instruction=x["old_instruction"], instruction_reason=x.get("instruction_reason"),
                                        spot_check=sp or None)))
            ch = "same" if ins.strip() == x["old_instruction"].strip() else f"{x['old_instruction']} → **{ins}**"
            lines.append(f"| {x['id']} | {x['old_cut'][0]}–{x['old_cut'][1]} | {s}–{e_} | {ch} | "
                         f"{'UNCERTAIN: ' + x['uncertain_note'] if x.get('uncertain') else ''}{' SPOT: ' + sp.get('note', '') if sp else ''} |")
        for b in r.get("between", []):
            for j, t in enumerate(b.get("tasks", [])):
                rid = f"{b['id']}_r{j}"
                sp = spot.get(rid, {})
                if sp.get("drop"):
                    lines.append(f"| {rid} (reset) | – | DROPPED (spot check) | {t['instruction']} | {sp.get('note', '')} |"); continue
                s, e_ = sp.get("cut", t["cut"])
                ins = sp.get("instruction", t["instruction"])
                eps.append(dict(id=rid, video=v, kind="reset_task", source="Claude review of the between-stretch", instruction=ins,
                                start_s=float(s), end_s=float(e_), reviewed=True, uncertain=bool(t.get("uncertain")),
                                uncertain_note=t.get("uncertain_note", ""), change=dict(reason=t.get("reason"), stretch=b["id"],
                                                                                       spot_check=sp or None)))
                lines.append(f"| {rid} (NEW reset) | – | {s}–{e_} | **{ins}** | "
                             f"{'UNCERTAIN: ' + t.get('uncertain_note', '') if t.get('uncertain') else ''} |")
        if r.get("notes"):
            lines.append(f"\nNotes: {r['notes']}")
    # segments: every task / reset episode, and the uncovered stretches between them as unlabelled
    segs = []
    for v in vids:
        n = json.load(open(ROOT / f"data/table_clean/_work/{v}/frames_meta.json"))["n"]
        dur = round((n - 1) / 10, 1)
        E = sorted([e for e in eps if e["video"] == v], key=lambda e: e["start_s"])
        t, i = 0.0, 0
        for e in E:
            if e["start_s"] - t > 0.05:
                segs.append(dict(id=f"{v}_gap_{i}", video=v, kind="between", instruction="", start_s=round(t, 1), end_s=e["start_s"])); i += 1
            if e["start_s"] < t - 0.05:
                e["overlaps_previous"] = True
            segs.append(dict(id=e["id"], video=v, kind="episode", episode_kind=e["kind"], instruction=e["instruction"],
                             start_s=e["start_s"], end_s=e["end_s"]))
            t = max(t, e["end_s"])
        if dur - t > 0.05:
            segs.append(dict(id=f"{v}_gap_{i}", video=v, kind="between", instruction="", start_s=round(t, 1), end_s=dur))
    out = dict(source="v1 (Gemini 2.5 Flash core cuts) checked and fixed by Claude Sonnet reviewers from 1 fps contact sheets + 2-4 fps zooms, "
                      "spot-checked by Claude Opus (episode_review/spot_check.json); reset / setup tasks labelled from the between-stretches",
               brief=str(REV / "BRIEF.md"), n=len(eps), n_task=sum(e["kind"] == "task" for e in eps),
               n_reset_task=sum(e["kind"] == "reset_task" for e in eps), not_reviewed=missing, episodes=eps, segments=segs,
               segments_note="every video cut into episodes (kind 'episode': Gemini tasks and the new reset tasks) and the stretches left "
                             "between them (kind 'between', empty instruction, unlabelled)")
    json.dump(out, open(ROOT / "data/labels/episodes_all_v2.json", "w"), indent=1)
    (REV / "CHANGES.md").write_text("# Episode changes v1 → v2\n" + "\n".join(lines) + "\n")
    print(json.dumps({k: out[k] for k in ("n", "n_task", "n_reset_task", "not_reviewed")}))


if __name__ == "__main__":
    main()
