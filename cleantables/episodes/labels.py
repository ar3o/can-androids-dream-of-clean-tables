"""VLM language labels for every moment of a video, at three time scales (forward clips only):
  action   2 s clip,  6 frames, every 2 s   "what are the hands doing"
  subtask  6 s clip,  8 frames, every 2 s   "what subtask, naming the object and where it goes"
  activity 20 s clip, 8 frames, every 5 s   "overall activity"
Frames are cropped to the table region + room above it for the arms (table pixels from det3d.npz), no fixed label list.
Qwen3-VL-4B-Instruct runs on the CPU in bf16 (its 8.9 GB of weights exceed the 4 GB GPU budget of a shared GPU).
Every label gets a frozen MiniLM embedding. Subtask labels are attached to moves by time in cleantables.perception.events.
  $VLA_PY -m cleantables.episodes.labels VIDEOS...   -> data/generic/<V>/vlm_labels.json  [{t, scale, length, label}] + vlm_label_emb.npz
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from cleantables.perception.common import embed, frame, jdump, log, meta, names, table_crop_box, vdir, vlm_ask

SCALES = {          # name: (clip length s, stride s, frames, question)
    "subtask": (6.0, 2.0, 8, "In 4 to 10 words, what subtask is the person doing over these seconds, naming the object and where it goes "
                             "(e.g. 'moving the apple into the fruit bowl', 'stacking the plates', 'putting the cup on the tray')?"),
    "activity": (20.0, 5.0, 8, "In 4 to 10 words, what is the person's overall activity in this clip (e.g. 'tidying fruit off the table', "
                               "'setting up a messy table', 'clearing dishes')?"),
    "action": (2.0, 2.0, 6, "In 3 to 8 words, what are the person's hands doing in this moment (e.g. 'picking up the banana', "
                            "'holding a mug', 'idle')?"),
}
HEAD = "These frames are a clip of a person at a table, in time order. "
TAIL = " Answer with the description only."


def clip_images(v, ks, box):
    from PIL import Image
    x0, y0, x1, y1 = box
    ims = []
    for k in ks:
        im = frame(v, k)[y0:y1, x0:x1]
        s = 448 / im.shape[1]
        ims.append(Image.fromarray(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., ::-1]))
    return ims


def run(v, scales=None):
    p = vdir(v) / "vlm_labels.json"
    rows = json.load(open(p)) if p.exists() else []
    done = {(r["scale"], round(r["t"], 2)) for r in rows}
    n = meta(v)["n"]
    dur = n / 10.0
    box = table_crop_box(v, margin=24, up=120)
    t0 = time.time()
    for scale in (scales or SCALES):
        L, stride, nf, q = SCALES[scale]
        Lc = min(L, dur - 0.2)
        for t in np.arange(Lc / 2, dur - Lc / 2 + 1e-6, stride):
            if (scale, round(float(t), 2)) in done:
                continue
            ks = [int(np.clip(round((t + dt) * 10), 0, n - 1)) for dt in np.linspace(-Lc / 2, Lc / 2, nf)]
            lab = vlm_ask(clip_images(v, ks, box), HEAD + q + TAIL, 24).strip().strip(".").strip("'").lower()
            rows.append(dict(t=round(float(t), 2), scale=scale, length=float(Lc), label=lab))
            if len(rows) % 10 == 0:
                jdump(rows, p)
                print(v, scale, f"{t:.0f}/{dur:.0f}s", lab, f"({time.time() - t0:.0f}s)", flush=True)
    jdump(rows, p)
    labs = sorted({r["label"] for r in rows})
    np.savez(vdir(v) / "vlm_label_emb.npz", labels=np.array(labs), emb=embed(labs))
    log(f"{v} VLM labels: {len(rows)} ({', '.join(f'{s} {sum(r['scale'] == s for r in rows)}' for s in SCALES)}) in {time.time() - t0:.0f}s")
    return rows


def labels(v):
    rows = json.load(open(vdir(v) / "vlm_labels.json"))
    z = np.load(vdir(v) / "vlm_label_emb.npz")
    E = dict(zip([str(x) for x in z["labels"]], z["emb"]))
    return rows, E


def label_at(rows, t, scale="subtask"):
    """Label of the clip of `scale` whose centre is nearest to t (s)."""
    rs = [r for r in rows if r["scale"] == scale]
    return min(rs, key=lambda r: abs(r["t"] - t))["label"] if rs else ""


if __name__ == "__main__":
    vids = names([a for a in sys.argv[1:] if not a.startswith("--")])
    sc = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--scales=")), None)
    for v in vids:                     # subtask for every video first (what the world model uses), then the other scales
        run(v, sc or ["subtask"])
    if not sc:
        for v in vids:
            run(v, ["activity", "action"])
