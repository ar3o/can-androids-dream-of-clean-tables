"""Open-vocabulary object discovery: the VLM names the objects, SAM 3 validates the names. No hand-written object list.

1. list   : Qwen3-VL-4B looks at 3 frames per video (start / middle / end), cropped to the table region (table pixels from det3d.npz + margin),
            and lists every separate movable object as short noun phrases (neutral prompt, no example nouns).
            -> data/generic/<V>/vlm_objects.json
2. merge  : phrases from all videos are normalised (lower case, no articles, crude singular) and near-duplicates are merged by MiniLM
            cosine similarity >= MERGE_SIM (keeps the most frequent wording) -> data/generic/vocab_candidates.json
3. verify : every candidate phrase (the video's own + any said in >= 2 listings) is run through SAM 3 on 3 frames of each video (CPU or GPU, SAM 3 env).
            A detection counts if prob >= P_OK and its mask centroid is on the table (det3d table mask, dilated).
            Detections must be on the table in 3D (depth points not below the plane, centre inside its extent) and not composite
            (>= 70 % covered by >= 2 other detections, e.g. 'stack of ...').
            Instance = detections of any phrase on the same frame with mask IoU >= 0.5. Each instance is named by its highest-probability
            phrase; a phrase is kept if it names >= 1 instance with p >= P_NAME (ranked by summed probability, capped by MAX_PHRASES). -> data/generic/<V>/vocab.json (phrases + evidence)
'hand' (the agent) is always prompted in the inventory as well, but it is never an object.
  $VLA_PY -m cleantables.perception.objects list|merge|verify|select VIDEOS...
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter

import cv2
import numpy as np

from cleantables.perception.common import AGENT_PHRASES, OUT, ROOT, SAM_PATH, SAM_PY, embed, frame, geom, jdump, log, meta, names, table_crop_box, vdir, vlm_ask

# Neutral prompt: no example nouns and no request for trays, mats or containers, so the listing does not steer towards a task.
# Q_LIST_V1 is an earlier prompt with examples; each vlm_objects*.json stores the question it was asked with.
Q_LIST = ("This photo shows a table. List every separate movable object on the table, one entry per physical object "
          "(if there are several similar ones, list each with what tells it apart). Use a short noun phrase of 1 to 4 words, adding the colour "
          "or material when it helps. Do not list people, hands, arms, the table itself, the wall or the floor. Answer with the phrases only, "
          "separated by semicolons.")
Q_LIST_V1 = ("This photo shows a table. List every separate movable object lying or standing on the table, one entry per physical object "
          "(if there are several similar ones, list each with what tells it apart). Use a short noun phrase of 1 to 4 words, adding the colour "
          "or material when it helps, e.g. 'red mug', 'glass bowl', 'banana', 'white plate'. Also list any tray, mat or container that other "
          "things are put on or in. Do not list people, hands, arms, the table itself, the wall or the floor. Answer with the phrases only, "
          "separated by semicolons.")
MERGE_SIM = 0.90
P_OK = 0.5
# P_NAME = SAM 3's own detection threshold; no phrase cap (MAX_PHRASES = None), since a cap drops objects in cluttered scenes.
P_NAME = 0.5
MAX_PHRASES = None
N_VERIFY = 3
MIN_COUNT = 2         # a phrase from another video is a candidate here only if the VLM said it in >= 2 frame listings
SKIP = {"table", "wooden table", "table top", "tabletop", "person", "hand", "hands", "arm", "floor", "wall", "nothing", "none"}


def verify_frames(v):
    n = meta(v)["n"]
    return [int(x) for x in np.linspace(2, n - 3, N_VERIFY)]


def list_objects(v, name="vlm_objects.json"):
    p = vdir(v) / name
    if p.exists():
        return json.load(open(p))
    from PIL import Image
    n = meta(v)["n"]
    x0, y0, x1, y1 = table_crop_box(v)
    rows = []
    for k in (2, n // 2, n - 3):
        im = frame(v, k)[y0:y1, x0:x1]
        s = 640 / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ans = vlm_ask([Image.fromarray(im[..., ::-1])], Q_LIST, 96)
        rows.append(dict(k=k, answer=ans, phrases=[norm(x) for x in ans.split(";") if norm(x)]))
        log(f"{v} objects k={k}: {ans}")
    out = dict(video=v, crop_xyxy=[int(c) for c in (x0, y0, x1, y1)], question=Q_LIST, frames=rows)
    jdump(out, p)
    return out


def norm(s):
    s = s.strip().strip(".").strip("'\"").lower()
    s = re.sub(r"^(a|an|the|some|two|three|several)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    w = s.split()
    if not w or s in SKIP or len(w) > 5:
        return ""
    last = w[-1]                                         # crude singular of the head noun
    if len(last) > 3 and last.endswith("es") and last[-3] in "sxz":
        last = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    return " ".join(w[:-1] + [last])


def merge(videos):
    cnt = Counter()
    per_video = {}
    for v in videos:
        ph = sorted({p for r in list_objects(v)["frames"] for p in r["phrases"]})
        per_video[v] = ph
        cnt.update(ph)
    phr = sorted(cnt, key=lambda p: (-cnt[p], p))
    E = embed(phr)
    S = E @ E.T
    rep = {}
    groups = []
    for i, p in enumerate(phr):                          # most frequent first: each phrase joins the first group it is close to
        g = next((g for g in groups if S[i, g[0]] >= MERGE_SIM), None)
        if g is None:
            groups.append([i])
        else:
            g.append(i)
    for g in groups:
        for i in g:
            rep[phr[i]] = phr[g[0]]
    cands = [phr[g[0]] for g in groups]
    out = dict(candidates=cands, counts={p: cnt[p] for p in phr}, merged_into=rep, merge_sim=MERGE_SIM,
               per_video={v: sorted({rep[p] for p in per_video[v]}) for v in videos})
    jdump(out, OUT / "vocab_candidates.json")
    log(f"vocab candidates: {len(phr)} distinct phrases from the VLM -> {len(cands)} after MiniLM merge: {cands}")
    return out


def verify(v, device="cpu", threads=8):
    out = vdir(v) / "sam3_verify.npz"
    if out.exists():
        return out
    vc = json.load(open(OUT / "vocab_candidates.json"))
    own = set(vc["per_video"].get(v, []))
    cands = [p for p in vc["candidates"] if p in own or vc["counts"].get(p, 0) >= MIN_COUNT]
    pj = vdir(v) / "verify_phrases.json"
    jdump(cands, pj)
    env = dict(__import__("os").environ, PYTHONPATH=f"{SAM_PATH}:{ROOT}", HF_HUB_OFFLINE="1", SAM3_THREADS=str(threads))
    cmd = [SAM_PY, "-m", "cleantables.perception.sam3_run", v, "--phrases-json", str(pj), "--out", str(out),
           "--frames", ",".join(map(str, verify_frames(v))), "--device", device]
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)
    return out


def load_dets(path):
    z = np.load(path)
    m = int(z["shape"][0])
    masks = np.unpackbits(z["masks"])[:m * 270 * 480].reshape(m, 270, 480).astype(bool) if m else np.zeros((0, 270, 480), bool)
    return dict(frame=z["frame"], prompt=z["prompt"], prob=z["prob"], box=z["box"], masks=masks,
                prompts=[str(x) for x in z["prompts"]], frames=[int(x) for x in z["frames_done"]])


def iou(a, b):
    return np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())


_G = {}


def on_table_3d(v, k, mask):
    """Generic 3D check: the mask's depth points stand on / above the table plane and their centre is within the table extent."""
    from cleantables.perception.scene.recon3d import backproject, to_table
    from cleantables.perception.boxes.scene_export import table_extent
    from cleantables.perception.common import depth
    if v not in _G:
        g = geom(v)
        _G[v] = (g, depth(v), np.array(table_extent(g["T"], g["K"], g["table"])))
    g, D, ext = _G[v]
    P, _ = backproject(np.asarray(D[k], np.float32) * g["scale"][k], g["K"], mask)
    if len(P) < 10:
        return False
    Q = to_table(g["T_all"][k], P)
    cx, cy = np.median(Q[:, 0]), np.median(Q[:, 1])
    return bool(np.median(Q[:, 2]) > -0.03 and ext[0, 0] - 0.05 <= cx <= ext[0, 1] + 0.05 and ext[1, 0] - 0.05 <= cy <= ext[1, 1] + 0.05)


def select(v):
    d = load_dets(vdir(v) / "sam3_verify.npz")
    table = cv2.dilate(geom(v)["table"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    P = d["prompts"]
    ok = []
    for i in range(len(d["prob"])):
        if d["prob"][i] < P_OK:
            continue
        ys, xs = np.nonzero(d["masks"][i])
        if len(xs) < 20 or len(xs) > 0.3 * d["masks"][i].size or not table[int(ys.mean()), int(xs.mean())]:
            continue
        if not on_table_3d(v, int(d["frame"][i]), d["masks"][i]):
            continue                                     # floor mat, bin on the counter, ...: not on the table in 3D
        ok.append(i)
    # composite detections ('a stack of ...'): >= 70 % covered by >= 2 other confident detections -> not an object of its own
    comp = set()
    for i in ok:
        mi = d["masks"][i]; a = mi.sum()
        parts = [j for j in ok if j != i and d["frame"][j] == d["frame"][i] and iou(mi, d["masks"][j]) < 0.5
                 and (d["masks"][j] & mi).sum() > 0.6 * d["masks"][j].sum()]
        if len(parts) >= 2:
            u = np.zeros_like(mi)
            for j in parts:
                u |= d["masks"][j]
            if (u & mi).sum() >= 0.7 * a:
                comp.add(i)
    ok = [i for i in ok if i not in comp]
    # instances per frame: greedy clustering of all phrases' detections by mask IoU
    inst = {}                                            # det -> (frame, instance id)
    for k in d["frames"]:
        ids = sorted([i for i in ok if d["frame"][i] == k], key=lambda i: -d["prob"][i])
        reps = []
        for i in ids:
            j = next((r for r in reps if iou(d["masks"][i], d["masks"][r]) >= 0.5), None)
            if j is None:
                reps.append(i); inst[i] = (k, len(reps) - 1)
            else:
                inst[i] = inst[j]
    # each instance is named by the phrase that finds it with the highest probability (the most specific confident wording wins:
    # 'blue box' 0.92 beats 'cardboard box' 0.90 on the blue box). A phrase is kept if it is the best name of >= 1 instance with
    # prob >= P_NAME; phrases ranked by the summed probability of the instances they name, at most MAX_PHRASES.
    best = {}
    for i, key in inst.items():
        if key not in best or d["prob"][i] > d["prob"][best[key]]:
            best[key] = i
    score = {}
    for key, i in best.items():
        if d["prob"][i] >= P_NAME and P[d["prompt"][i]] not in AGENT_PHRASES:
            score[P[d["prompt"][i]]] = score.get(P[d["prompt"][i]], 0) + float(d["prob"][i])
    chosen = sorted(score, key=lambda p: -score[p])[:MAX_PHRASES]
    ev = {}
    for p in range(len(P)):
        ev[P[p]] = dict(n_conf=int(sum(1 for i in ok if d["prompt"][i] == p)), names_score=round(score.get(P[p], 0), 3),
                        max_prob=round(max([float(d["prob"][i]) for i in ok if d["prompt"][i] == p], default=0), 3))
    out = dict(video=v, phrases=chosen, agent_phrases=AGENT_PHRASES, verify_frames=d["frames"], p_ok=P_OK,
               evidence=ev, rule="each SAM 3 instance on the verify frames is named by its highest-probability phrase; keep phrases naming >= 1 instance with p >= P_NAME")
    jdump(out, vdir(v) / "vocab.json")
    log(f"{v} vocabulary ({len(chosen)}): {chosen}")
    return out


def vocab(v):
    return json.load(open(vdir(v) / "vocab.json"))["phrases"]


def main():
    cmd, vids = sys.argv[1], names(sys.argv[2:])
    if cmd == "list":
        for v in vids:
            list_objects(v)
    elif cmd == "merge":
        merge(vids)
    elif cmd == "verify":
        for v in vids:
            verify(v)
    elif cmd == "select":
        for v in vids:
            select(v)


if __name__ == "__main__":
    main()
