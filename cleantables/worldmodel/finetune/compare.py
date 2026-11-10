"""Before / after comparison of the domain-randomised fine-tune: the held-out episode in the 3 held-out worlds, every input from
the scene, the frozen base model ('old') vs its fine-tuned copy ('new'). Same rules for both:
  first-frame paths   fruit (not the bowl) whose predicted end spot is >= 3 cm from where perception put it, at the first keyframe
  end-spot error      median over keyframes / objects that still move >= 2 cm before the episode ends (real tracker, preds_real.json)
                      of the distance from the predicted end spot to the real end; 'stays put' = the real position at that keyframe
-> DR/result.json + DR/result_first_frame.jpg (old | new per held-out world, the first keyframe's predicted paths)
  python -m cleantables.worldmodel.finetune.compare --dr DR"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

P = Path(CT_DATA + "/wm_eval/wm_v2_seg/videos/stageA/wm_pause")
PAL = {48: (0, 200, 255), 49: (60, 150, 255), 50: (40, 90, 255), 51: (200, 200, 200), 52: (90, 230, 120), 53: (255, 170, 60)}


def scores(pred, real):
    RK = {kf["snap_k"]: {o["track_id"]: o for o in kf["objects"]} for kf in real["keyframes"]}
    k0 = pred["keyframes"][0]
    first = sum(1 for o in k0["objects"] if o["track_id"] != 51 and np.hypot(*(np.subtract(o["goal"][:2], o["cur"][:2]))) >= 0.03)
    e, st = [], []
    for kf in pred["keyframes"]:
        for o in kf["objects"]:
            r = RK.get(kf["snap_k"], {}).get(o["track_id"])
            if r is None or r["t1_true"] is None or np.hypot(*(np.subtract(r["t1_true"][:2], r["cur"][:2]))) < 0.02:
                continue
            e.append(np.hypot(*(np.subtract(o["goal"][:2], r["t1_true"][:2])))); st.append(np.hypot(*(np.subtract(r["cur"][:2], r["t1_true"][:2]))))
    return dict(first_frame_paths=f"{first}/5", end_err_cm=round(100 * float(np.median(e)), 1) if e else None,
                stays_put_cm=round(100 * float(np.median(st)), 1) if st else None, n=len(e))


def draw(w, s, pred, title):
    cam = json.load(open(next(w.glob("camera_*.json"))))
    K = np.array(cam["K"], float) * (960 / cam["wh"][0]); K[2, 2] = 1; T = np.array(cam["T_cam_to_table"], float)
    pr = lambda X: (lambda pc: pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2])((np.asarray(X, float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3])  # noqa: E731
    im = cv2.resize(cv2.imread(str(sorted((w / f"frames_IMG_2516_t11_{s}").glob("*.jpg"))[0])), (960, 540))
    for o in pred["keyframes"][0]["objects"]:
        col = PAL.get(o["track_id"], (230, 230, 230)); cur, goal = np.array(o["cur"]), np.array(o["goal"])
        cv2.circle(im, tuple(np.round(pr(cur)[0]).astype(int)), 6, col, -1, cv2.LINE_AA)
        if o["track_id"] != 51 and np.hypot(*(goal[:2] - cur[:2])) >= 0.03:
            pts = np.round(pr(np.array(o["path"]))).astype(np.int32)
            for i in range(0, len(pts) - 1, 2):
                cv2.line(im, tuple(pts[i]), tuple(pts[i + 1]), col, 4, cv2.LINE_AA)
            th = np.linspace(0, 2 * np.pi, 48); r = float(np.clip(np.hypot(*o["sd"][:2]), 0.025, 0.15))
            cv2.polylines(im, [np.round(pr(np.stack([goal[0] + r * np.cos(th), goal[1] + r * np.sin(th), np.full(48, goal[2])], -1))).astype(np.int32)], True, col, 3, cv2.LINE_AA)
    if title:
        cv2.rectangle(im, (0, 0), (960, 44), (20, 20, 22), -1); cv2.putText(im, title, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dr", required=True); ap.add_argument("--held", default="held")
    a = ap.parse_args(); DR = Path(a.dr); sfx = "" if a.held == "held" else f"_{a.held}"
    real = json.load(open(P / "preds_real.json")); out, rows = {}, []
    for w in sorted((DR / a.held / "IMG_2516_t11").iterdir()):
        if "_v" in w.name:                                                      # skip older renders kept alongside
            continue
        cams = list(w.glob("camera_*.json"))
        if not cams or not (w / "preds_old.json").exists() or not (w / "preds_new.json").exists():
            continue
        s = cams[0].stem.split("_")[-1]; po, pn = json.load(open(w / "preds_old.json")), json.load(open(w / "preds_new.json"))
        out[w.name] = dict(old=scores(po, real), new=scores(pn, real))
        so, sn = out[w.name]["old"], out[w.name]["new"]
        rows.append(np.hstack([draw(w, s, po, f"{w.name}: before DR  ({so['first_frame_paths']} fruit get a path)"),
                               draw(w, s, pn, f"{w.name}: after DR  ({sn['first_frame_paths']} fruit get a path)")]))
        print(w.name, out[w.name], flush=True)
    json.dump(out, open(DR / f"result{sfx}.json", "w"), indent=1)
    if rows:
        cv2.imwrite(str(DR / f"result_first_frame{sfx}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
