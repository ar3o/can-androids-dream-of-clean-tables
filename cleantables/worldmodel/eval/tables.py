"""Markdown tables from out/wm/{main,ablate,judge}.json -> out/wm/TABLES.md (formatting only, no new numbers).

  $VLA_PY -m cleantables.worldmodel.eval.tables
"""
import json
from pathlib import Path

import os
EB = Path(__file__).resolve().parents[3] / "outputs" / "eval"; O = EB / "out" / "wm"; TAG = os.environ.get("OUTTAG", "")


def f(x, d=1):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def ci(c, d=1):
    return "" if not c else f" [{f(c[0], d)}, {f(c[1], d)}]"


def pct(x):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.0f}%"


def t1row(R, g):
    t = R.get(g, {}).get("T1", {}); m = t.get("moved", {}); s = t.get("stayed", {})
    return (f"{f(m.get('err_cm'))}{ci(m.get('ci'))}", f"{f(m.get('diff_vs_ref_cm'))}{ci(m.get('diff_ci'))}" if "diff_vs_ref_cm" in m else "–",
            f(s.get("err_cm")), f"{f(t.get('auroc'), 2)}{ci(t.get('auroc_ci'), 2)}" if "auroc" in t else "–",
            f"{pct(m.get('cov1'))} / {pct(m.get('cov2'))}" if "cov1" in m else "–", m.get("n"), m.get("n_ep"))


def nxt(R, g):
    n = R.get(g, {}).get("next", {})
    return (f"{pct(n.get('exact'))}", f"{pct(n.get('order_free'))}{' [' + pct(n['order_free_ci'][0]) + ', ' + pct(n['order_free_ci'][1]) + ']' if n.get('order_free_ci') else ''}",
            pct(n.get("chance")), n.get("n"))


def main_tables(L):
    M = json.load(open(O / f"main{TAG}.json"))
    L.append(f"Baseline settings chosen on fold-0 validation: constant velocity H = {M['tune']['H_best']} s; kNN k = {M['tune']['k_best']}. Keyframes: {M['groups']}.\n")
    for g in ("in scope", "fruit in", "tools in", "fruit out", "all 767"):
        L.append(f"\n**{g}** (mode 1nh, 3D only; moved = |end − start| > 16.2 cm)\n")
        L.append("| method | moved end err cm [CI] | Δ vs ours cm [CI] | stayed err cm | which-moves AUROC [CI] | cov 1/2 sd (68/95) | n moved / eps | next exact | next order-free [CI] | chance | T2 ADE moved cm | T2 ADE all cm |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, R in M["rows"].items():
            if g not in R:
                continue
            if k.startswith("final_s2"):
                n = nxt(R, g); te = R[g].get("true_next_end_err_cm", {})
                L.append(f"| {k} | true next object: {f(te.get('final_s2'))} (ours {f(te.get('ours'))}, stay {f(te.get('stay'))}; n {te.get('n')}) | | | | | | {n[0]} | {n[1]} | {n[2]} | | |")
                continue
            a = t1row(R, g); n = nxt(R, g) if "next" in R[g] else ("–", "–", "–", None)
            t2 = R[g].get("T2", {})
            L.append(f"| {k} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {a[4]} | {a[5]} / {a[6]} | {n[0]} | {n[1]} | {n[2]} | "
                     f"{f(t2.get('moved', {}).get('ade_cm'))} | {f(t2.get('all', {}).get('ade_cm'))} |")
        if g == "in scope":
            dh = M["rows"].get("Ours (3-seed ensemble)", {}).get(g, {}).get("dist_head_auroc")
            L.append(f"\nOurs' own distance-head AUROC (the deployed read-out): {f(dh, 2)}.")
    if "vlm_keyframes" in M:
        V = M["vlm_keyframes"]
        for g in ("first keyframe of each episode", "VLM keyframes, in scope", "VLM keyframes, put-into"):
            L.append(f"\n**Every method on the VLM's keyframes: {g}** ({V['n']} VLM keyframes answered; unparseable answers {V['n_unparsed']}, counted as stays put)\n")
            L.append("| method | moved end err cm [CI] | Δ vs ours cm [CI] | which-moves AUROC | n moved / eps | next exact | next order-free | chance |")
            L.append("|---|---|---|---|---|---|---|---|")
            for k, R in V["rows"].items():
                if g not in R:
                    continue
                a = t1row(R, g); n = nxt(R, g) if R[g].get("next") else ("–", "–", "–", None)
                L.append(f"| {k} | {a[0]} | {a[1]} | {a[3]} | {a[5]} / {a[6]} | {n[0]} | {n[1]} | {n[2]} |")


def ablate_tables(L):
    A = json.load(open(O / f"ablate{TAG}.json"))
    for g in ("in scope", "fruit in", "tools in", "fruit out"):
        L.append(f"\n**Recipe and input ablations, {g}** (mode 1nh unless stated; Δ vs the final 3-seed ensemble)\n")
        L.append("| model | moved end err cm [CI] | Δ vs final cm [CI] | stayed err cm | which-moves AUROC | dist-head AUROC | cov 1/2 sd | next exact | next order-free | chance |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, R in A["rows"].items():
            if isinstance(R, str):
                L.append(f"| {k} | {R} |"); continue
            a = t1row(R, g); n = nxt(R, g)
            L.append(f"| {k} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {f(R[g].get('dist_head_auroc'), 2)} | {a[4]} | {n[0]} | {n[1]} | {n[2]} |")
    if "shuffled_instruction" in A:
        L.append("\n**Shuffled instruction** (each in-scope keyframe gets the other family's instruction)\n")
        L.append("| keyframes | mean end shift, all objects cm | mean end shift, moved cm | moved err own → shuffled cm | total predicted motion own → shuffled cm | next order-free own → shuffled |")
        L.append("|---|---|---|---|---|---|")
        for g, S in A["shuffled_instruction"].items():
            L.append(f"| {g} | {f(S['mean_end_shift_cm_all'])} | {f(S['mean_end_shift_cm_moved'])} | {f(S['moved_err_own_cm'])} → {f((S['moved_err_shuffled'] or {}).get('err_cm'))} | "
                     f"{f(S['total_pred_motion_cm_own'])} → {f(S['total_pred_motion_cm_shuffled'])} | {pct(S['next_own'].get('order_free'))} → {pct(S['next_shuffled'].get('order_free'))} |")
    if "shuffled_instruction_round2" in A:
        L.append("\n**Shuffled instruction, direct-regression baselines** (same swap as above)\n")
        L.append("| model | keyframes | mean end shift, all cm | moved err own → shuffled cm | total predicted motion own → shuffled cm |")
        L.append("|---|---|---|---|---|")
        for mdl, SS in A["shuffled_instruction_round2"].items():
            for g, S in SS.items():
                L.append(f"| {mdl} | {g} | {f(S['mean_end_shift_cm_all'])} | {f(S['moved_err_own_cm'])} → {f((S['moved_err_shuffled'] or {}).get('err_cm'))} | "
                         f"{f(S['total_pred_motion_cm_own'])} → {f(S['total_pred_motion_cm_shuffled'])} |")
    if "prompts" in A:
        L.append("\n**Single-object prompts** (\"Put the banana into the fruit bowl.\" …), vs the episode's own instruction on the same keyframes\n")
        L.append("| keyframes | prompt rows | names | share of predicted motion on the named object: prompt / own (named share of objects) | named object is the next pick: prompt / own | named object's predicted move cm: prompt / own | other task objects' move cm: prompt / own | named end → container cm: prompt / own / start |")
        L.append("|---|---|---|---|---|---|---|---|")
        for g, S in A["prompts"].items():
            m = lambda k: f(S[k]["mean"]) if k in S else "–"  # noqa: E731
            p_ = lambda k: pct(S[k]["mean"]) if k in S else "–"  # noqa: E731
            L.append(f"| {g} | {S['n_prompt_rows']} | {', '.join(S['names'])} | {p_('sel_prompt')} {ci(S['sel_prompt']['ci'], 2)} / {p_('sel_own')} ({p_('named_share')}) | "
                     f"{p_('top1_prompt')} / {p_('top1_own')} | {m('disp_named_prompt')} / {m('disp_named_own')} | {m('disp_other_prompt')} / {m('disp_other_own')} | "
                     f"{m('named_to_container_prompt_cm')} / {m('named_to_container_own_cm')} / {m('named_to_container_start_cm')} |")


def judge_tables(L):
    J = json.load(open(O / f"judge{TAG}.json"))
    for rn, A in J["4a"].items():
        L.append(f"\n**4a. Goal error vs the human's real end, {rn}** (put-into episodes)\n")
        L.append("| goal source | moved err cm [CI] | stayed err cm | n moved / eps |")
        L.append("|---|---|---|---|")
        for k, v in A.items():
            if k == "JUDGE acceptance":
                continue
            m = v.get("moved", {}); s = v.get("stayed", {})
            L.append(f"| {k} | {f(m.get('err_cm'))}{ci(m.get('ci'))} | {f(s.get('err_cm'))} | {m.get('n')} / {m.get('n_ep')} |")
        a = A["JUDGE acceptance"]
        L.append(f"\nJudge's met rule (xy within max(sd, 3 cm); median tolerance {f(a['tol_cm_median'])} cm): accepts the human's real end for {pct(a['human_end_accepted_moved'])}"
                 f" [{pct(a['ci'][0])}, {pct(a['ci'][1])}] of moved objects; accepts doing nothing for {pct(a['do_nothing_accepted_moved'])} [{pct(a['ci0'][0])}, {pct(a['ci0'][1])}];"
                 f" accepts stayed objects' real end {pct(a['human_end_accepted_stayed'])} (n moved {a['n_moved']}, stayed {a['n_stayed']}).")
    L.append("\n**4b/4f. Does seed disagreement predict the judge's error?** (every put-into keyframe)\n")
    L.append("| objects | n | Spearman (seed-1 vs seed-2 distance, error) [CI] | Spearman (judge sd, error) | AUROC for error > 3 cm: disagreement / sd | AURC mean err cm: by disagreement / random / oracle |")
    L.append("|---|---|---|---|---|---|")
    for k, v in J["4b_4f"].items():
        L.append(f"| {k} | {v['n']} | {f(v['spearman_d12'], 2)}{ci(v['spearman_d12_ci'], 2)} | {f(v['spearman_sd'], 2)} | {f(v['auroc_d12_err_gt3'], 2)} / {f(v['auroc_sd_err_gt3'], 2)} | "
                 f"{f(v['aurc_mean_err'])} / {f(v['aurc_random'])} / {f(v['aurc_oracle'])} |")
        L.append(f"| risk–coverage ({k}) | | " + "; ".join(f"{int(100 * r['coverage'])}%: {f(r['mean_err_cm'])} cm" for r in v["risk_coverage"]) + " | | | |")
    for nm in ("4d_put_into", "4d_fruit_out_reverse_task"):
        S = J.get(nm) or {}
        if not S:
            continue
        L.append(f"\n**4d. {nm}: anchors, judge score vs true score** ({S['n_segments']} episodes; objects = judge movers ∪ human movers)\n")
        L.append("| end state | J_judge | J_true | XGap = J_judge − J_true [CI] | XGap, judge movers only (as training) |")
        L.append("|---|---|---|---|---|")
        for a, v in S["anchors"].items():
            L.append(f"| {a} | {pct(v['J_judge'])} | {pct(v['J_true'])} | {f(100 * v['XGap'])} pts{' [' + f(100 * v['XGap_ci'][0]) + ', ' + f(100 * v['XGap_ci'][1]) + ']'} | {f(100 * v['XGap_trainset'])} pts |")
        r = S["ranking"]
        L.append(f"\nRanking agreement over these {len(r['anchors'])} end states: pairwise {pct(r['pairwise_accuracy'])} [{pct(r['pairwise_ci'][0])}, {pct(r['pairwise_ci'][1])}], MMRV {f(r['MMRV'], 3)} [{f(r['MMRV_ci'][0], 3)}, {f(r['MMRV_ci'][1], 3)}].")
        L.append("\nProgress curves (0, 10, …, 100% of the episode): J_judge " + ", ".join(pct(x) for x in S["J_judge_curve"]) + "; J_true " + ", ".join(pct(x) for x in S["J_true_curve"]) + ".")
        b = S["backwards"]
        L.append(f"\nBackwards: read at the LAST keyframe (task done), the judge says the human's movers stay for {pct(b['judge_at_end_says_stay_human_movers'])} (all objects {pct(b['judge_at_end_says_stay_all'])});"
                 f" read at the first keyframe it says they stay for {pct(b['judge_at_start_says_stay_human_movers'])}. Its score along the reversed video (end → start): "
                 + ", ".join(pct(x) for x in b["J_judge_reversed_curve"]) + ".")


def mixed_tables(L):
    M = json.load(open(O / f"mixed{TAG}.json"))
    L.append(f"\n**Mixed tables** (PLAN F / G; {M['n_pairs']} held-out pairs; mode 1nh; CIs over base episodes)\n")
    L.append("| keyframes | method | share of predicted motion on the named group [CI] | next object in the named group [CI] | other group's predicted move cm | named group's end error cm (stays put) | n named movers |")
    L.append("|---|---|---|---|---|---|---|")
    for k, S in M["summary"].items():
        for m in M["methods"]:
            if m == "stays put" or m not in S:
                continue
            x = S[m]
            L.append(f"| {k} (named share of objects {pct(S['named_share'])}) | {m} | {pct(x['selectivity'])} [{pct(x['selectivity_ci'][0])}, {pct(x['selectivity_ci'][1])}] | "
                     f"{pct(x['next_in_named'])} [{pct(x['next_ci'][0])}, {pct(x['next_ci'][1])}] | {f(x['other_group_disp_cm'])} | "
                     f"{f(x['named_movers_err_cm'])} ({f(x['named_movers_stay_cm'])}) | {x['n_named_movers']} |")


def pair_tables(L):
    P = json.load(open(O / f"pair_s0_A2{TAG}.json"))
    L.append("\n**Ours seed 0 alone vs A2 (seed 0), paired on the same keyframes and objects** (PLAN H; Δ = A2 − ours seed 0, CI over episodes)\n")
    L.append("| test | n | ours seed 0 [CI] | A2 [CI] | Δ A2 − ours [CI] |")
    L.append("|---|---|---|---|---|")
    for k, v in P.items():
        d = 2 if "AUROC" in k else 1
        n = f"{v['n_moved']} moved / {v['n_ep']} eps" if "n_ep" in v else f"{v['n_objects']} objects ({v['n_moved']} moved)"
        L.append(f"| {k} | {n} | {f(v['ours_seed0'], d)}{ci(v.get('ours_seed0_ci'), d)} | {f(v['A2'], d)}{ci(v.get('A2_ci'), d)} | "
                 f"{f(v['diff_A2_minus_ours'], d)}{ci(v['diff_ci'], d)} |")


def swap_tables(L):
    C = json.load(open(O / f"check_s0_swap{TAG}.json"))
    L.append(f"\n**Check: each final seed alone on the tool tables, own vs fruit instruction** (PLAN H; {C['n_keyframes']} tools-in keyframes, {C['n_episodes']} episodes)\n")
    L.append("| seed | own text via the swap path vs stored, mean / max cm | total predicted motion own → fruit cm | moved err own → fruit cm | mean end shift cm | keyframes whose motion halves |")
    L.append("|---|---|---|---|---|---|")
    for k, v in C.items():
        if k.startswith("seed"):
            L.append(f"| {k} | {f(v['path_check_mean_cm'], 2)} / {f(v['path_check_max_cm'], 2)} | {f(v['total_motion_own_cm'])} → {f(v['total_motion_fruit_cm'])} | "
                     f"{f(v['moved_err_own_cm'])} → {f(v['moved_err_fruit_cm'])} | {f(v['mean_end_shift_cm'])} | {pct(v['share_keyframes_motion_halved'])} |")


def noapp_tables(L):
    P = json.load(open(O / f"pair_noapp_s0{TAG}.json"))
    L.append("\n**Ours seed 0 with appearance off vs ours seed 0 and A2** (PLAN I; paired on the same objects, CI over episodes)\n")
    L.append("| test | n | ours seed 0 | seed 0, appearance off | A2 (seed 0) | Δ off − ours seed 0 | Δ off − A2 |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in P.items():
        L.append(f"| {k} | {v['n_moved']} moved / {v['n_ep']} eps | {f(v['ours seed 0'])}{ci(v['ours seed 0 ci'])} | {f(v['ours seed 0, appearance off'])}{ci(v['ours seed 0, appearance off ci'])} | "
                 f"{f(v['A2 (seed 0)'])}{ci(v['A2 (seed 0) ci'])} | {f(v['diff off - ours seed 0'])}{ci(v['diff off - ours seed 0 ci'])} | {f(v['diff off - A2 (seed 0)'])}{ci(v['diff off - A2 (seed 0) ci'])} |")


if __name__ == "__main__":
    L = ["# World model and judge tables (generated by scripts/wm_tables.py)\n"]
    for nm, fn in (("main", main_tables), ("ablate", ablate_tables), ("judge", judge_tables), ("mixed", mixed_tables), ("pair_s0_A2", pair_tables),
                   ("check_s0_swap", swap_tables), ("pair_noapp_s0", noapp_tables)):
        if (O / f"{nm}{TAG}.json").exists():
            L.append(f"\n## {nm}\n"); fn(L)
    (O / f"TABLES{TAG}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
