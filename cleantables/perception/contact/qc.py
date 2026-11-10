"""Contact overlay video (CPU): the Hands23 contact labels (cleantables.perception.contact.contact, mask matching) drawn on the frames.

Per 10 fps work frame of the episode window:
  * each twin object's SAM 3 outline (nearest mask frame within +-1, as used for matching): thin grey when no hand touches it, filled
    and outlined in the colour of the hand that touches it per the labels: left = blue, right = orange, both = purple;
  * each hand's Hands23 box (the highest-scoring detection of that side, as used for the labels), in its colour, with Hands23's contact
    state and score; when its contacted object matched none of the twin objects ("unlinked"), Hands23's own object box is drawn dashed;
  * a text list of the active contacts (and 'hand not detected').
Nothing is computed here: the states are read from OUT/contact/<ep>_frames.npz, the boxes from the Hands23 detections.
Out: OUT/qc/<ep>/contact_overlay.mp4 (H.264, 10 fps)
  $FP_PY -m cleantables.perception.contact.qc [EP...]
"""
from __future__ import annotations

import glob
import subprocess
import sys

import cv2
import numpy as np

from cleantables.perception.episode_io import OUT, TRIAL, episode, frame, sam3_masks, scene, tracks

COL = {"left": (230, 140, 40), "right": (40, 140, 255), "both": (200, 60, 200)}       # BGR
SIDES = ("left", "right")


def dashed_rect(img, b, col):
    x0, y0, x1, y1 = [int(v) for v in b]
    for x in range(x0, x1, 10):
        cv2.line(img, (x, y0), (min(x + 5, x1), y0), col, 2); cv2.line(img, (x, y1), (min(x + 5, x1), y1), col, 2)
    for y in range(y0, y1, 10):
        cv2.line(img, (x0, y), (x0, min(y + 5, y1)), col, 2); cv2.line(img, (x1, y), (x1, min(y + 5, y1)), col, 2)


def run(ep, F=None, name="contact_overlay", title="Hands23 contact labels (raw, unsmoothed)", held_only=False):
    """F: per-frame states (contact.contact _frames.npz format); default = the Hands23 labels, drawn with Hands23's hand boxes. With
    another F (e.g. per-frame states converted from a video model's events) no hand boxes exist, so none are drawn. held_only: F['left_state'] means
    'held by some hand' (a model without hand side, e.g. HOIST-Former): drawn green, text 'held'."""
    e = episode(ep)
    V = e["video"]
    h23 = F is None
    if F is None:
        F = np.load(OUT / "contact" / f"{ep}_frames.npz")
    ids = [str(x) for x in F["objects"]]
    st = {s: F[f"{s}_state"] for s in SIDES}
    k2j = {int(k): j for j, k in enumerate(F["k"])}
    if h23:                             # Hands23 detections (hand boxes) exist only for the Hands23 trial windows
        p = sorted(x for x in glob.glob(str(OUT / "contact/hands23" / f"{V}_*.npz")) if not x.endswith("_objmasks.npz"))
        p = [x for x in p if int(x.rsplit("_", 2)[1]) <= e["w0"] and int(x.rsplit("_", 1)[1][:-4]) >= e["w1"]]
        z = np.load(p[0])
        cnames = [str(x) for x in z["contact_names"]]
        hf, hs, hsc = z["hand_frame"], z["hand_side"], z["hand_score"]
    S, R = scene(ep), tracks(V)
    M, _, _, _, _ = sam3_masks(V)
    objs = [o for o in S["objects"] if o.get("track") is not None and o["track"] >= 0]
    mk = {o["id"]: {int(k): d for k, d in zip(R["objects"][o["track"]]["sight_k"], R["objects"][o["track"]]["det"]) if d is not None}
          for o in objs}
    out = OUT / "qc" / ep
    out.mkdir(parents=True, exist_ok=True)
    raw = out / f"{name}_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 540 + 70))
    for k in range(e["w0"], e["w1"] + 1):
        img = frame(V, k).copy()
        j = k2j.get(k)
        ov = img.copy()
        lines = []
        for o in objs:
            near = next((kk for kk in (k, k - 1, k + 1) if kk in mk[o["id"]]), None)
            if near is None:
                continue
            m = cv2.resize(M[mk[o["id"]][near]].astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST)
            oi = ids.index(o["id"])
            on = [s for s in SIDES if j is not None and st[s][j, oi] == 2]
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if on:
                c = (60, 200, 60) if held_only else (COL["both"] if len(on) == 2 else COL[on[0]])
                ov[m > 0] = c
                cv2.drawContours(img, cnts, -1, c, 2, cv2.LINE_AA)
            else:
                cv2.drawContours(img, cnts, -1, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
        for si, s in enumerate(SIDES):
            if not h23:
                if held_only and s == "right":
                    continue
                act = [ids[i] for i in np.flatnonzero(st[s][j] == 2)] if j is not None else []
                lab = "held (hand not given)" if held_only else s
                lines.append((s, f"{lab}: " + (", ".join(act) if act else ("nothing held" if held_only else "no object contact"))))
                continue
            rows = np.flatnonzero((hf == k) & (hs == si))
            if j is None or not len(rows) or (st[s][j] == 0).all():
                lines.append((s, f"{s}: hand not detected"))
                continue
            r = rows[np.argmax(hsc[rows])]
            b = z["hand_box"][r]
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COL[s], 2)
            cv2.putText(img, f"{s[0].upper()} {cnames[int(z['contact_state'][r])]} {float(z['contact_score'][r]):.2f}", (int(b[0]), max(12, int(b[1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[s], 1, cv2.LINE_AA)
            act = [ids[i] for i in np.flatnonzero(st[s][j] == 2)]
            if "unlinked" in act and z["obj_row"][r] >= 0:
                dashed_rect(img, z["obj_box"][r], COL[s])
            lines.append((s, f"{s}: " + (", ".join(a if a != "unlinked" else "unlinked object (no twin match)" for a in act) if act else "no object contact")))
        bar = np.full((70, 960, 3), 25, np.uint8)
        cv2.putText(bar, f"{ep}  k={k}  t={(k - e['w0']) / 10:.1f}s   {title}   " + ("green=held" if held_only else "blue=left orange=right purple=both"),
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (s, t) in enumerate(lines):
            cv2.putText(bar, t, (8, 40 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL[s], 1, cv2.LINE_AA)
        if e["k0"] <= k <= e["k1"]:
            cv2.putText(bar, "in labelled episode", (800, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 120), 1, cv2.LINE_AA)
        vw.write(np.vstack([img, bar]))
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                    str(out / f"{name}.mp4")], check=False)
    if (out / f"{name}.mp4").exists():
        raw.unlink()
    print(ep, out / f"{name}.mp4", flush=True)


if __name__ == "__main__":
    for ep in sys.argv[1:] or TRIAL:
        run(ep)
