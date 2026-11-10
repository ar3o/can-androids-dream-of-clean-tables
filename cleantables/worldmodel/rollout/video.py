"""Video of the closed-loop plan rollouts (rollout.plan) side by side, step by step, with the same score for every setup.
Grid: rows = setups (e.g. the real scene's twin from the phone camera; a held-out world), columns = the model as trained (video + 3D)
and its 3D-only mode (V-JEPA and DINOv2 off). Each step pauses on the frame the model decided from (every object's predicted path
faint, the chosen object's path dashed with its end-spot circle (1 sd), a white ring on it, squares where perception put the
objects), then the arm carries it (its real path in sim drawn solid). A setup that stopped holds its last frame.
Score per setup (same rules for all; drawn at the end and written to --score):
  fruit in bowl     fruit whose final centre is inside the bowl's rim (its final xy within the bowl's half-size)
  vs real end       per object, the final position vs where it ended in the held-out video (median, cm; the human's result)
  landed            the carried object's final spot vs the model's end spot for it (median over steps, cm)
  perception        perceived vs the sim's exact positions of the seen objects (median over steps, cm)
  through rim       carries whose predicted path crosses the bowl's rim circle with the object's bottom below the rim's height,
                    out of the carries that cross the rim
  python -m cleantables.worldmodel.rollout.video --rollouts A B C D --labels ... --rows 2 --out V.mp4 --score S.json"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
FONT = "/usr/share/fonts/truetype/lato/Lato-{}.ttf"
PAL = [(0, 200, 255), (60, 150, 255), (40, 90, 255), (200, 200, 200), (90, 230, 120), (255, 170, 60)]   # BGR per scene object
ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", nargs="+", required=True); ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--cols", type=int, default=2); ap.add_argument("--episode", default="IMG_2516_t11")
ap.add_argument("--progress_file", default="progress_v2.json", help="the progress bar's source in each rollout folder (wm_plan_rollout "
                "--progress_v2: the earlier v2 model's trained progress head; --progress_only: the watch WM's how-much-left read-out)")
ap.add_argument("--minimal", action="store_true", help="no text but each panel's label ")
ap.add_argument("--kinematic", action="store_true", help="the rollouts ran with physics off (labels)")
ap.add_argument("--out", required=True); ap.add_argument("--score", required=True); ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
PW, PH, BAR, CAP = (960, 540, 44, 0) if "--minimal" in __import__("sys").argv else (960, 540, 56, 120)
f_ = lambda w, s: ImageFont.truetype(FONT.format(w), s)                                                        # noqa: E731
S = json.load(open(ROOT / "data/twin_table/ep/_scalefix" / a.episode / "scene.json"))
N = len(S["objects"]); EXT = np.array([o.get("extents", [0.1, 0.1, 0.1]) for o in S["objects"]], float)
NAME = [o["category"].split("/")[-1].replace("_", " ") for o in S["objects"]]
BOWL = next(i for i, o in enumerate(S["objects"]) if "bowl" in o["category"]); FRUIT = [i for i in range(N) if i != BOWL]

# the real result: each object's end in the held-out video (tracker, the episode's last sample), paired as twin.dr_render does
z = np.load(CT_DATA + "/wm_eval/wm_v3_tracks/tracks_all.npz", allow_pickle=True)
ii = np.nonzero(z["group"] == a.episode)[0]; ii = ii[np.argsort(z["snap_k"][ii])]
g0 = {int(o): z["obj_geo"][ii[0]][s][:3] for s, o in enumerate(z["obj_id"][ii[0]]) if z["obj_mask"][ii[0]][s] and z["obj_kind"][ii[0]][s] == 0}
cand = list(g0); dd = np.array([[np.linalg.norm(np.array(ob["pos"][:2]) - g0[o][:2]) for o in cand] for ob in S["objects"]])
TRACK = np.full(N, -1)
for _ in range(min(N, len(cand))):
    r_, c_ = np.unravel_index(np.nanargmin(dd), dd.shape); TRACK[r_] = cand[c_]; dd[r_, :] = np.nan; dd[:, c_] = np.nan
REAL_END = np.zeros((N, 3))
for r in range(N):
    for i in ii[::-1]:
        ids = list(z["obj_id"][i])
        if TRACK[r] in ids and z["obj_mask"][i][ids.index(TRACK[r])]:
            REAL_END[r] = z["obj_geo"][i][ids.index(TRACK[r])][:3]; break


def in_bowl(P):
    return [r for r in FRUIT if np.hypot(*(P[r, :2] - P[BOWL, :2])) < 0.5 * EXT[BOWL, :2].max()]


RO = []
for d, lab in zip(a.rollouts, a.labels):
    R = json.load(open(Path(d) / "rollout.json")); R["dir"] = Path(d); R["label"] = lab
    K = np.array(R["camera"]["K"], float) * (PW / R["camera"]["wh"][0]); K[2, 2] = 1.0; T = np.array(R["camera"]["T_cam_to_table"], float)
    R["proj"] = (lambda K, T: lambda P: ((np.asarray(P, float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3])[:, :2]
                 / np.maximum(((np.asarray(P, float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3])[:, 2:3], 1e-6) * [K[0, 0], K[1, 1]] + K[:2, 2])(K, T)
    fin = np.array(R.get("final_top") or R["truth_top"][-1])
    ex = [s for s in R["steps"] if "executed" in s]
    TT = np.array(R["truth_top"]); cross = thru = 0
    for s_ in ex:                                                               # does the predicted path cross the bowl rim below its height?
        k = R["objects"].index(s_["chosen"]); o = next(o for o in s_["wm"]["objects"] if o["obj"] == k); W = np.array(o["path"])
        r = np.linalg.norm(W[:, :2] - TT[s_["frames"][0], BOWL, :2], axis=1); j = np.flatnonzero((r[:-1] > 0.5 * EXT[BOWL, :2].max()) & (r[1:] <= 0.5 * EXT[BOWL, :2].max()))
        if len(j) and k != BOWL:
            cross += 1; thru += int(W[j[0], 2] - EXT[k, 2] < EXT[BOWL, 2])
    R["score"] = dict(setup=lab, wm_decisions=len(R["steps"]), carried=len(ex), ended=R["steps"][-1].get("ended", "step limit"),
                      fruit_in_bowl=f"{len(in_bowl(fin))}/{len(FRUIT)}",
                      vs_real_end_cm=round(100 * float(np.median([np.hypot(*(fin[r, :2] - REAL_END[r, :2])) for r in range(N)])), 1),
                      landed_cm=round(float(np.median([s["landed_cm"] for s in ex])), 1) if ex else None,
                      through_rim=f"{thru}/{cross}",
                      perception_cm=round(float(np.median([s["perception_err_cm"] for s in R["steps"] if s["perception_err_cm"] is not None])), 1),
                      chosen=[s["chosen"] for s in R["steps"]])
    RO.append(R)
START = np.array([[o["pos"][0], o["pos"][1], 0] for o in S["objects"]])
ref = dict(setup="the human (held-out video)", fruit_in_bowl=f"{len(in_bowl(REAL_END))}/{len(FRUIT)}", vs_real_end_cm=0.0,
           start_vs_real_end_cm=round(100 * float(np.median([np.hypot(*(START[r, :2] - REAL_END[r, :2])) for r in range(N)])), 1))
json.dump(dict(rollouts=[R["score"] for R in RO], reference=ref, rules=__doc__.split("Score per setup")[1]), open(a.score, "w"), indent=1)
for R in RO:
    print("[rollout-video]", R["score"], flush=True)
print("[rollout-video] reference", ref, flush=True)


def dashed(img, pts, col, th, on=14, off=10):
    acc, draw = 0.0, True
    for p, q in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(q - p))); s = 0.0
        while s < L:
            seg = min((on if draw else off) - acc, L - s); a_, b_ = p + (q - p) * s / L, p + (q - p) * (s + seg) / L
            if draw:
                cv2.line(img, tuple(np.round(a_).astype(int)), tuple(np.round(b_).astype(int)), col, th, cv2.LINE_AA)
            s += seg; acc += seg
            if acc >= (on if draw else off) - 1e-6:
                acc, draw = 0.0, not draw


def ring(proj, c, r, n=48):
    th = np.linspace(0, 2 * np.pi, n); return np.round(proj(np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th), np.full(n, c[2])], -1))).astype(np.int32)


def img(R, f):
    return cv2.resize(cv2.imread(str(R["dir"] / "all_frames" / f"{f:05d}.jpg")), (PW, PH))


def overlay_pause(im, R, st, grow):
    pr = R["proj"]; lay = im.copy(); k = R["objects"].index(st["chosen"])
    for o in st["wm"]["objects"]:
        col = PAL[o["obj"] % len(PAL)]; cur, goal = np.array(o["cur"]), np.array(o["goal"])
        if np.hypot(*(goal[:2] - cur[:2])) >= 0.03 and o["obj"] != k:
            dashed(lay, pr(np.array(o["path"])), col, 2, 6, 8)
        u = np.round(pr(cur)[0]).astype(int)
        cv2.rectangle(lay, tuple(u - 6), tuple(u + 6), (255, 255, 255), 3); cv2.rectangle(lay, tuple(u - 6), tuple(u + 6), col, 2)
    o = next(o for o in st["wm"]["objects"] if o["obj"] == k); col = PAL[k % len(PAL)]
    P = np.array(o["path"]); m = max(2, int(round(grow * len(P))))
    dashed(lay, pr(P[:m]), col, 5)
    if grow >= 0.999:
        cv2.polylines(lay, [ring(pr, np.array(o["goal"]), float(np.clip(np.hypot(*o["sd"][:2]), 0.025, 0.15)))], True, col, 3, cv2.LINE_AA)
    cv2.polylines(lay, [ring(pr, np.array(o["cur"]), 0.06)], True, (255, 255, 255), 4, cv2.LINE_AA)
    return lay


def overlay_play(im, R, st, f):
    pr = R["proj"]; k = R["objects"].index(st["chosen"]); o = next(o for o in st["wm"]["objects"] if o["obj"] == k); col = PAL[k % len(PAL)]
    lay = im.copy(); dashed(lay, pr(np.array(o["path"])), col, 3)
    cv2.polylines(lay, [ring(pr, np.array(o["goal"]), float(np.clip(np.hypot(*o["sd"][:2]), 0.025, 0.15)))], True, col, 2, cv2.LINE_AA)
    im = cv2.addWeighted(lay, 0.6, im, 0.4, 0)
    f0 = st["frames"][0]; tp = np.array(R["truth_top"])[f0:f + 1, k]
    if len(tp) >= 2:
        uv = np.round(pr(tp)).astype(np.int32)
        cv2.polylines(im, [uv], False, (20, 20, 20), 8, cv2.LINE_AA); cv2.polylines(im, [uv], False, col, 4, cv2.LINE_AA)
    return im


def tag(im, R, text, sub):
    if a.minimal:
        return im
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    d.rectangle([0, PH - 64, PW, PH], fill=(20, 20, 22)); d.text((14, PH - 60), text, font=f_("Bold", 22), fill=(250, 210, 90))
    d.text((14, PH - 30), sub, font=f_("Regular", 17), fill=(200, 200, 200))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def bar(R, w):
    im = Image.new("RGB", (w, BAR), (22, 22, 24)); d = ImageDraw.Draw(im)
    d.text((14, 6 if not a.minimal else 9), R["label"], font=f_("Bold", 22), fill=(240, 240, 240))
    if a.minimal:
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    d.text((14, 32), "the frozen WM decides; a robot arm carries out its plan; " + ("physics off: exactly on the WM's path" if a.kinematic else "PhysX"),
           font=f_("Regular", 16), fill=(160, 160, 160))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def caption(w, line):
    if a.minimal:
        return np.zeros((0, w, 3), np.uint8)
    im = Image.new("RGB", (w, CAP), (18, 18, 20)); d = ImageDraw.Draw(im)
    d.text((18, 8), line, font=f_("Bold", 26), fill=(250, 250, 250))
    d.text((18, 46), "Each step: the scene is perceived like a real video (SAM 3 + MoGe-3 boxes; V-JEPA 2 + DINOv2 pixels); the frozen world model picks "
           "the next object and its path (dashed; circle = its end spot); an arm carries it (solid = where it went).", font=f_("Regular", 16), fill=(190, 190, 190))
    d.text((18, 70), "World model pretrained only by watching real videos; this episode was held out. Left column: as trained (video + 3D). "
           "Right column: its 3D-only mode (V-JEPA and DINOv2 off). Squares: where perception put each object.", font=f_("Regular", 16), fill=(150, 160, 170))
    d.text((18, 94), ("Physics off: the object follows the WM's path exactly (through anything in the way) and stays at its end spot. " if a.kinematic else "")
           + "Not a learned robot policy: the arm just shows who moves it.",
           font=f_("Regular", 16), fill=(150, 160, 170))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


rows = (len(RO) + a.cols - 1) // a.cols; Wt, Ht = PW * a.cols, (BAR + PH) * rows + CAP
vw = __import__("subprocess").Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wt}x{Ht}", "-r", str(a.fps), "-i", "-",
                                     "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out],
                                    stdin=__import__("subprocess").PIPE)


def emit(panels, line):
    grid = []
    for r in range(rows):
        row = [np.vstack([bar(RO[c], PW), panels[c]]) if c < len(panels) else np.zeros((BAR + PH, PW, 3), np.uint8)
               for c in range(r * a.cols, (r + 1) * a.cols)]
        grid.append(np.hstack(row))
    fr = np.vstack(grid + [caption(Wt, line)]); vw.stdin.write(np.ascontiguousarray(fr).tobytes()); return fr


if a.minimal:
    # each panel on its own clock (no shared pauses): its frames play straight through; from a decision to the end of that carry the chosen
    # object's predicted path (dashed) and end spot (circle) are drawn, during the carry its real path (solid); a bar along the bottom = the
    # progress read-out (--progress_file, every 0.5 s, interpolated); a finished run holds its last frame
    for R in RO:
        pj = R["dir"] / a.progress_file
        pr_ = json.load(open(pj))["rows"] if pj.exists() else []
        R["pf"], R["pv"] = np.array([r["frame"] for r in pr_], float), np.array([r["progress"] for r in pr_], float)
    def bar_prog(im, R, f):
        if not len(R["pf"]):
            return im
        v = float(np.interp(f, R["pf"], R["pv"])); h = 14
        cv2.rectangle(im, (0, PH - h), (PW, PH), (35, 35, 38), -1); cv2.rectangle(im, (0, PH - h), (int(PW * v), PH), (120, 205, 175), -1)
        return im
    def panel_at(R, f):
        im = img(R, f)
        st = next((s_ for s_ in R["steps"] if "executed" in s_ and s_["frame"] <= f <= s_["frames"][1]), None)
        if st is not None:
            im = overlay_play(im, R, st, f) if f >= st["frames"][0] else overlay_pause(im, R, st, 1.0)
        return bar_prog(im, R, f)
    n = max(R["n_frames"] for R in RO) + 2 * a.fps; still = None
    for t in range(n):
        fr_ = emit([panel_at(R, min(t, R["n_frames"] - 1)) for R in RO], "")
        if t == min(R["idle_frames"][1] for R in RO) + 20:
            still = fr_
    vw.stdin.close(); rc = vw.wait()
    cv2.imwrite(str(Path(a.out).with_suffix(".jpg")), still, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[rollout-video] -> {a.out} rc {rc}", flush=True); raise SystemExit(0)
cur = [R["idle_frames"][1] for R in RO]
for f in range(max(R["idle_frames"][1] for R in RO) + 1):                          # the start, at rest
    emit([img(R, min(f, R["idle_frames"][1])) for R in RO], "Start: the same scene, the same objects")
still = None
for si in range(1, max(len(R["steps"]) for R in RO) + 1):
    ST = [next((s for s in R["steps"] if s["step"] == si), None) for R in RO]
    nh = int(2.5 * a.fps)
    for h in range(nh):                                                         # the pause: every setup's decision at step si
        P = []
        for R, st, c in zip(RO, ST, cur):
            im = img(R, c)
            if st is None:
                P.append(tag(im, R, "finished", R["score"]["ended"]))
            elif "executed" not in st:
                P.append(tag(overlay_pause(im, R, st, 1.0) if st.get("chosen") else im, R, f"step {si}: WM stops", st.get("ended", "")))
            else:
                P.append(tag(overlay_pause(im, R, st, min(1.0, h / (0.4 * nh))), R, f"step {si}: WM picks the {NAME[R['objects'].index(st['chosen'])]}",
                             f"its end spot is {st['goal_dist_cm']:.0f} cm away; perception error {st['perception_err_cm']:.1f} cm"))
        fr = emit(P, f"PAUSED, step {si}: what the world model proposes")
        if si == 1 and h == nh - 1:
            still = fr
    L = max((st["frames"][1] - st["frames"][0] + 1) if st and "executed" in st else 0 for st in ST)
    for j in range(L):                                                          # play: each arm carries out its plan
        P = []
        for n_, (R, st) in enumerate(zip(RO, ST)):
            if st and "executed" in st:
                f = min(st["frames"][0] + j, st["frames"][1]); cur[n_] = f
                im = overlay_play(img(R, f), R, st, f)
                P.append(tag(im, R, f"step {si}: carrying the {NAME[R['objects'].index(st['chosen'])]}",
                             f"it lands {st['landed_cm']:.1f} cm from the WM's end spot" if f == st["frames"][1] else "solid: where it really goes"))
            else:
                P.append(tag(img(R, cur[n_]), R, "finished" if st is None or "executed" not in st else "", R["score"]["ended"]))
        emit(P, f"PLAYING, step {si}: the arm carries out the plan")
for _ in range(int(4 * a.fps)):                                                 # the score, the same for every setup
    P = [tag(img(R, cur[n_]), R, f"fruit in bowl {R['score']['fruit_in_bowl']}   vs the real end {R['score']['vs_real_end_cm']} cm",
             f"{R['score']['carried']} moves; paths through the bowl rim {R['score']['through_rim']}; perception {R['score']['perception_cm']} cm")
         for n_, R in enumerate(RO)]
    emit(P, f"Score (same rules for all). The human in the real video: fruit in bowl {ref['fruit_in_bowl']}.")
vw.stdin.close(); rc = vw.wait()
cv2.imwrite(str(Path(a.out).with_suffix(".jpg")), still, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"[rollout-video] -> {a.out} rc {rc}", flush=True)
