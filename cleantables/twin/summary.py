"""Summary of the twin evaluation: reads outputs/eval/twins/per_ep/*.json (cleantables.twin.evaluate) and the existing settle / build
records (no reruns), bootstraps over episodes, and writes outputs/eval/twins/twins_results.json.

  CUDA_VISIBLE_DEVICES="" $SIM_PY -m cleantables.twin.summary
"""
from __future__ import annotations

from cleantables import CT_DATA
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EPD = ROOT / "data/twin_table/ep"
OUTD = ROOT / "outputs/eval/twins"
GPU_CHECK = Path(CT_DATA + "/perception_v2/twins_v2")
IN_SCOPE = [f"IMG_2516_t{i:02d}" for i in range(1, 15)] + [f"IMG_{v}_t01" for v in (2533, 2534, 2535, 2536, 2537, 2538, 2539, 2542, 2544, 2546, 2547, 2548)]
ROWS = ("moge", "oct", "inuse")
ROW_NAME = {"moge": "MoGe box", "oct": "Oct twins", "inuse": "twins in use", "scalefix": "_scalefix (sanity only)"}
NB, SEED = 1000, 0


def load(sub=""):
    per = {}
    for f in sorted(glob.glob(str(OUTD / sub / "per_ep" / "*.json"))):
        d = json.load(open(f))
        per[d["meta"]["episode"]] = d
    return per


def med(x):
    return float(np.median(x)) if len(x) else None


def boot(per_ep_vals, idx, stat=np.median):
    """per_ep_vals: list (episodes) of lists of object values; idx: (NB, n_ep) episode draws -> (point, lo, hi)."""
    allv = [v for vs in per_ep_vals for v in vs]
    if not allv:
        return None, None, None
    pt = float(stat(np.array(allv)))
    bs = []
    for row in idx:
        v = [x for i in row for x in per_ep_vals[i]]
        if v:
            bs.append(float(stat(np.array(v))))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, float(lo), float(hi)


def boot_diff(a_vals, b_vals, idx, stat=np.median):
    """Paired over the same episode draws: stat(a) - stat(b)."""
    fa = lambda r: [x for i in r for x in a_vals[i]]  # noqa: E731
    fb = lambda r: [x for i in r for x in b_vals[i]]  # noqa: E731
    allr = range(len(a_vals))
    pt = float(stat(np.array(fa(allr))) - stat(np.array(fb(allr))))
    bs = [float(stat(np.array(fa(r))) - stat(np.array(fb(r)))) for r in idx if fa(r) and fb(r)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, float(lo), float(hi)


def r3(t):
    return None if t is None or t[0] is None else [round(x, 4) for x in t]


def summarise(per, eps):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(eps), size=(NB, len(eps)))
    out = {}
    vals = {}
    for row in ROWS:
        R = {ep: [r for r in per[ep]["rows"] if r["row"] == row] for ep in eps}
        scored = {ep: [r for r in R[ep] if r.get("iou_vis") is not None] for ep in eps}
        g = lambda key, ep: [r[key] for r in scored[ep] if r.get(key) is not None]  # noqa: E731
        V = {k: [g(k, ep) for ep in eps] for k in ("iou_vis", "iou_full", "cen_px_vis", "cen_px_full", "dL_cm", "dW_cm", "dH_cm")}
        V["lt05"] = [[float(x < 0.5) for x in g("iou_vis", ep)] for ep in eps]
        vals[row] = V
        allr = [r for ep in eps for r in R[ep]]
        s = dict(n_objects_total=len(allr), n_objects_scored=sum(len(v) for v in scored.values()),
                 n_episodes=sum(1 for ep in eps if scored[ep]),
                 status_counts={k: sum(r["status"] == k for r in allr) for k in sorted({r["status"] for r in allr})},
                 n_sam_empty=sum(r["status"] == "sam_empty" for r in allr),
                 n_sam_frame_gt1_from_k0=sum(r.get("sam_dk", 0) > 1 for r in allr),
                 max_sam_frame_offset=max([r.get("sam_dk", 0) for r in allr] or [0]),
                 n_twin_fully_hidden_by_painter=sum(r.get("twin_px_vis") == 0 and (r.get("twin_px_full") or 0) > 0 for r in allr),
                 n_twin_off_image_or_empty=sum((r.get("twin_px_full") == 0) for r in allr),
                 n_centroid_undefined=sum(r.get("iou_vis") is not None and r.get("cen_px_vis") is None for r in allr),
                 iou_vis=r3(boot(V["iou_vis"], idx)), iou_full_no_occlusion=r3(boot(V["iou_full"], idx)),
                 iou_lt05_share=r3(boot(V["lt05"], idx, np.mean)),
                 centroid_px_vis=r3(boot(V["cen_px_vis"], idx)), centroid_px_full=r3(boot(V["cen_px_full"], idx)))
        if row != "moge":
            for k in ("dL_cm", "dW_cm", "dH_cm"):
                s[k] = r3(boot(V[k], idx))
                s[f"n_{k}"] = sum(len(v) for v in V[k])
        if row == "moge":
            s["moge_src_counts"] = {k: sum(r.get("moge_src") == k for r in allr) for k in ("window", "nearest3_within50")}
            s["n_no_moge_box"] = sum(r["status"] == "no_moge_box" for r in allr)
        out[row] = s
    for row in ("moge", "oct"):
        out[row]["iou_vis_minus_inuse"] = r3(boot_diff(vals[row]["iou_vis"], vals["inuse"]["iou_vis"], idx))
        out[row]["iou_full_minus_inuse"] = r3(boot_diff(vals[row]["iou_full"], vals["inuse"]["iou_full"], idx))
        out[row]["centroid_px_vis_minus_inuse"] = r3(boot_diff(vals[row]["cen_px_vis"], vals["inuse"]["cen_px_vis"], idx))
    return out


def sanity(per, per_ss1):
    out = {}
    for row, gname in (("scalefix", "scalefix"), ("oct", "current")):
        R = [r for d in per.values() for r in d["rows"] if r["row"] == row]
        R1 = [r for d in per_ss1.values() for r in d["rows"] if r["row"] == row]
        both = [r for r in R if r.get("iou_full") is not None and r.get("gpu_iou") is not None]
        d = np.array([r["iou_full"] - r["gpu_iou"] for r in both])
        both1 = [r for r in R1 if r.get("iou_full") is not None and r.get("gpu_iou") is not None]
        d1 = np.array([r["iou_full"] - r["gpu_iou"] for r in both1])
        out[row] = dict(gpu_variant=gname, n_matched=len(both), gpu_median=round(med([r["gpu_iou"] for r in both]), 4),
                        cpu_ss5_no_occlusion_median=round(med([r["iou_full"] for r in both]), 4),
                        cpu_ss5_abs_diff_p90=round(float(np.percentile(np.abs(d), 90)), 4), cpu_ss5_abs_diff_max=round(float(np.abs(d).max()), 4),
                        cpu_ss1_plain_fillpoly_no_occlusion_median=round(med([r["iou_full"] for r in both1]), 4) if both1 else None,
                        cpu_ss1_abs_diff_p90=round(float(np.percentile(np.abs(d1), 90)), 4) if both1 else None,
                        cpu_ss1_abs_diff_max=round(float(np.abs(d1).max()), 4) if both1 else None,
                        painter_visible_median=round(med([r["iou_vis"] for r in R if r.get("iou_vis") is not None]), 4))
    return out


def settle(eps_groups):
    S = {}
    for f in glob.glob(str(ROOT / "runs/twin_ep/settle/_scalefix/*.json")):
        d = json.load(open(f))
        S[d["id"].split("/")[-1]] = d
    out = {}
    for g, eps in eps_groups.items():
        have = [ep for ep in eps if ep in S]
        passed = [ep for ep in have if (not S[ep]["nan"]) and max(S[ep]["per_object"].values()) < 0.01]
        assert all(S[ep]["ok"] == (ep in passed) for ep in have)
        out[g] = dict(n=len(have), missing=[ep for ep in eps if ep not in S], n_pass=len(passed),
                      median_max_disp_cm=round(100 * med([S[ep]["max_disp_m"] for ep in have]), 2),
                      p90_max_disp_cm=round(100 * float(np.percentile([S[ep]["max_disp_m"] for ep in have], 90)), 2),
                      n_nan=sum(S[ep]["nan"] for ep in have))
    out["note"] = ("runs/twin_ep/settle/_scalefix/*.json: Isaac PhysX 1 s, arms present; pass = every object moves < 1 cm (= the files' 'ok'). "
                   "These are the _scalefix builds; no settle file exists for the _relink8 toolbox twins (runs/twin_ep/settle has _scalefix and _v2 only).")
    return out


PATTERNS = {
    "silhouette_fit_failed_tracked_box_used": r"silhouette fit failed",
    "lid_omitted_toolbox_closed": r"toolbox closed at the start .*lid omitted",
    "lid_omitted_overlaps_open_tray": r"overlaps the open tray -> lid on / under the tray, omitted",
    "uncertain_identity": r"uncertain identity|identity by shape (is )?weak",
    "weak_silhouette_fit": r"weak silhouette fit",
    "hidden_at_start_nearest_sightings": r"hidden at the start",
    "duplicate_identity_dropped": r"duplicate identity\) -> dropped",
    "object_omitted_not_sighted_or_carried": r"(not sighted within .* omitted|carried at the start.*omitted)",
    "moved_to_remove_interpenetration": r"moved .* cm to remove interpenetration",
}


def flags(eps_groups):
    out = {}
    for g, eps in eps_groups.items():
        c = {k: 0 for k in PATTERNS}
        c_relink = {k: 0 for k in PATTERNS}
        for ep in eps:
            for base, cc in ((EPD / "_scalefix" / ep, c), (EPD / "_relink8" / ep, c_relink)):
                if not (base / "scene.json").exists():
                    continue
                S = json.load(open(base / "scene.json"))
                txt = list(S.get("flags", [])) + list(S.get("notes", []))       # per-object notes repeat flags: not counted
                for k, p in PATTERNS.items():
                    cc[k] += sum(bool(re.search(p, t)) for t in txt)
        out[g] = dict(scalefix_builder_flags=c, relink8_scene_flags=c_relink)
    out["note"] = ("counted from scene.json flags + notes (per-object notes repeat flags) of the automatic _scalefix builds (the builder's output); "
                   "the 10 _relink8 scenes carry their own shorter flag lists (counted separately). 'uncertain_identity' includes the two "
                   "'pliers identity by shape weak' flags; the plan's '3 uncertain identities' are the three 'uncertain identity (...)' flags.")
    out["strict_uncertain_identity_flags"] = {g: sum(sum(bool(re.search(r"uncertain identity", t)) for t in json.load(open(EPD / "_scalefix" / ep / "scene.json")).get("flags", []))
                                                     for ep in eps) for g, eps in eps_groups.items()}
    return out


def build_facts(eps_groups):
    log = (ROOT / "runs/twin_ep/build_all.log").read_text().splitlines()
    built = [ln.split()[0] for ln in log if re.match(r"^IMG_\d+_t\d+ \d+ objects", ln)]
    errs = [ln for ln in log if re.search(r"error|fail|traceback", ln, re.I)]
    tail = [ln for ln in log if ln.startswith("built ")]
    inuse_ok = {}
    for g, eps in eps_groups.items():
        ok = 0
        for ep in eps:
            d = (EPD / "_relink8" / ep) if (EPD / "_relink8" / ep / "scene.json").exists() else (EPD / "_scalefix" / ep)
            S = json.load(open(d / "scene.json"))
            ok += all((d / "usd" / f"{o['id']}.obj").exists() and (d / "usd" / f"{o['id']}.usda").exists() for o in S["objects"])
        inuse_ok[g] = ok
    return dict(build_all_log=dict(path="runs/twin_ep/build_all.log", mtime=os.path.getmtime(ROOT / "runs/twin_ep/build_all.log"),
                                   n_built_lines=len(built), n_unique=len(set(built)), error_lines=errs, summary=tail,
                                   per_group={g: sum(ep in built for ep in eps) for g, eps in eps_groups.items()},
                                   note="dated 15 Oct: the October builds (Oct twins). The _scalefix rebuild has no log in runs/twin_ep/."),
                inuse_scenes_complete=inuse_ok, relinked_by_hand=sorted(p.name for p in (EPD / "_relink8").iterdir() if p.is_dir()))


def main():
    per = load()
    per_ss1 = load("ss1_sanity")
    eps = sorted(per)
    assert len(eps) == 45, len(eps)
    task = {ep: per[ep]["meta"]["task"] for ep in eps}
    ins = [ep for ep in eps if ep in IN_SCOPE]
    box = [ep for ep in eps if ep not in IN_SCOPE]
    check = dict(n_in_scope=len(ins), n_box=len(box), in_scope_tasks=sorted({task[e] for e in ins}), box_tasks=sorted({task[e] for e in box}),
                 consistent=all(task[e] in ("fruit", "tools") for e in ins) and all(task[e] == "stack_boxes" for e in box))
    groups = {"in_scope": ins, "box_stacking": box}
    res = dict(
        section="2a twins vs the real start frame", date="2026-11-09",
        method=dict(
            projection="scene.json K (960x540) scaled to the 270x480 SAM grid as the GPU check (f/2, (c-0.5)/2), T_cam_to_table; mesh = usd/<id>.obj + pos",
            renderer="every triangle filled with cv2.fillPoly (one call per triangle; one call for all would fill even-odd), at 5x the mask "
                     "resolution and sampled at each mask pixel centre (nvdiffrast's pixel-centre rule); TWINS_SS=1 = plain fillPoly (sanity only)",
            occlusion="painter's order: objects drawn far-to-near by the camera depth of their (area-weighted surface) centroid; nearer overwrite",
            sam_mask="det of the same tracker id (object 'track') at the sighting nearest k0 (twins_v2.check rule)",
            hand="SAM 3 'hand' mask at that sighting's frame, dilated 9x9, excluded from both masks (twins_v2.check rule)",
            moge_box="det_box median (x, y, top, L, W; yaw mod pi around the first) over sightings in [k0, k0+15]; bottom on the table; "
                     "fallback (builder's hidden-at-start rule): the 3 nearest sightings within 50 frames",
            sizes="twin L/W = min-area rectangle of the posed mesh's xy vertices (L >= W); twin height = the posed mesh's top z above the table, "
                  "compared with MoGe 'top' (same track). Not the scene.json 'extents' field (that is the axis-aligned box of the yaw-baked mesh).",
            centroid="mask centroids of the hand-excluded visible twin mask vs the hand-excluded SAM mask, px at 960x540 (270x480 distance x 2)",
            bootstrap=f"{NB} draws over episodes (resample episodes, median over their objects), numpy default_rng({SEED}); 95% percentile CI; "
                      "the same draws for all rows of a group (paired differences)",
            inuse="data/twin_table/ep/_scalefix/<ep>, except _relink8/<ep> for IMG_2534/2535/2539/2542/2548 x t01/t02"),
        groups=dict(in_scope=ins, box_stacking=box, check=check),
        summary={g: summarise(per, e) for g, e in groups.items()},
        summary_all45=summarise(per, eps),
        sanity_vs_gpu_check=sanity(per, per_ss1),
        settle=settle(groups), build=build_facts(groups), flags=flags(groups),
        objects=[r for ep in eps for r in per[ep]["rows"] if r["row"] in ROWS + ("scalefix",)],
        episodes_meta=[per[ep]["meta"] for ep in eps])
    json.dump(res, open(OUTD / "twins_results.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    print(json.dumps({k: res[k] for k in ("groups", "summary", "sanity_vs_gpu_check", "settle", "build", "flags")}, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main()
