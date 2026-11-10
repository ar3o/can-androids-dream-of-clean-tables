"""Contact on/off per hand and object from the published Hands23 detector (Cheng et al., NeurIPS 2023), no extra rules.

Input: OUT/contact/hands23/<V>_<k0>_<k1>.npz (third_party/hands23/scripts/run_hands23.py: the official demo, published thresholds).
Per frame and hand side (Hands23 'hand_side'; if a side is detected twice, the higher hand score):
  state 'on'      Hands23 contact state = object_contact and its in-contact object matches a SAM 3 object of this episode:
                  match='mask' (default): the Hands23 object mask (contact.hands23_masks) vs each SAM 3 object mask,
                  highest mask IoU, IoU >= 0.1. match='box': the paper's own association (Hands23 section 5.1), box IoU >= 0.1
                  between the Hands23 object box and the SAM 3 mask's bounding box (kept for comparison: *_boxmatch.*).
                  SAM 3 masks exist every 3rd frame in some videos: the nearest mask frame within +-1 is used.
  state 'off'     the hand is detected and is not in object contact with that object (any other contact state, or another object).
  state unknown   the hand side is not detected in that frame (Hands23 notes occluded hands are often missed).
  'unlinked'      object contact whose box matches no SAM 3 object (kept as object id "unlinked").
Intervals = maximal runs of 'on' frames; unknown frames inside a run do not end it (they are counted); the first 'off' frame does.
No smoothing: a one-frame 'off' splits a run; such flicker is counted and shown in the QC plots.
Out: OUT/contact/<ep>.json, <ep>_frames.npz (format: OUT/contact/README.md); OUT/qc/<ep>/contact_timeline.png
     (match='box': the same with suffix _boxmatch)
  $FP_PY -m cleantables.perception.contact.contact [EP...]
"""
from __future__ import annotations

import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cleantables.perception.episode_io import FPS, OUT, TRIAL, episode, jdump, sam3_masks, scene, tracks  # noqa: E402

ASSOC_IOU = 0.1
SIDES = ("left", "right")


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def run(ep, match="mask"):
    e = episode(ep)
    V = e["video"]
    suf = "" if match == "mask" else "_boxmatch"
    p = sorted(x for x in glob.glob(str(OUT / "contact/hands23" / f"{V}_*.npz")) if not x.endswith("_objmasks.npz"))
    p = [x for x in p if int(x.rsplit("_", 2)[1]) <= e["w0"] and int(x.rsplit("_", 1)[1][:-4]) >= e["w1"]]
    z = np.load(p[0], allow_pickle=False)
    om = np.load(p[0][:-4] + "_objmasks.npz")
    OM = np.unpackbits(om["masks"])[:int(np.prod(om["shape"]))].reshape(tuple(om["shape"])).astype(bool)
    inst_mask = {(int(z["inst_frame"][i]), int(z["inst_row"][i])): OM[a] for a, i in enumerate(om["inst_index"])}
    meta = json.loads(str(z["meta_json"]))
    cnames = [str(x) for x in z["contact_names"]]
    tnames = [str(x) for x in z["touch_names"]]
    gnames = [str(x) for x in z["grasp_names"]]
    OBJ = cnames.index("object_contact")
    S, R = scene(ep), tracks(V)
    M, _, _, _, _ = sam3_masks(V)
    objs = [o for o in S["objects"] if o.get("track") is not None and o["track"] >= 0]
    # SAM 3 mask bounding boxes in work pixels (960x540; masks are 270x480)
    mbox, mmask = {}, {}
    for o in objs:
        ro = R["objects"][o["track"]]
        for k, d in zip(ro["sight_k"], ro["det"]):
            if d is None:
                continue
            ys, xs = np.nonzero(M[d])
            if len(xs):
                mbox.setdefault(o["id"], {})[int(k)] = [xs.min() * 2, ys.min() * 2, xs.max() * 2 + 2, ys.max() * 2 + 2]
                mmask.setdefault(o["id"], {})[int(k)] = M[d]
    ks = np.arange(e["w0"], e["w1"] + 1)
    n = len(ks)
    ids = [o["id"] for o in objs] + ["unlinked"]
    # per frame & side: chosen hand row
    hf, hs, hsc = z["hand_frame"], z["hand_side"], z["hand_score"]
    state = {s: np.zeros((n, len(ids)), np.int8) for s in SIDES}          # 0 unknown, 1 off, 2 on
    iscore = {s: np.full(n, np.nan) for s in SIDES}
    touch = {s: np.full(n, -1) for s in SIDES}
    grasp = {s: np.full(n, -1) for s in SIDES}
    linked_iou = {s: np.full(n, np.nan) for s in SIDES}
    for j, k in enumerate(ks):
        for si, s in enumerate(SIDES):
            rows = np.flatnonzero((hf == k) & (hs == si))
            if not len(rows):
                continue
            r = rows[np.argmax(hsc[rows])]
            state[s][j] = 1
            iscore[s][j] = float(z["contact_score"][r])
            grasp[s][j] = int(z["grasp"][r])
            if int(z["contact_state"][r]) != OBJ or z["obj_row"][r] < 0:
                continue
            ob = z["obj_box"][r]
            obm = inst_mask.get((int(k), int(z["obj_row"][r])))
            touch[s][j] = int(z["obj_touch"][r])
            best, bi = 0.0, None
            for oi, o in enumerate(objs):
                near = [kk for kk in (k, k - 1, k + 1) if kk in mbox.get(o["id"], {})]
                if not near:
                    continue
                if match == "box":
                    v = box_iou(ob, mbox[o["id"]][near[0]])
                elif obm is not None:
                    sm = mmask[o["id"]][near[0]]
                    u = (obm | sm).sum()
                    v = float((obm & sm).sum() / u) if u else 0.0
                else:
                    v = 0.0
                if v > best:
                    best, bi = v, oi
            if bi is not None and best >= ASSOC_IOU:
                state[s][j, bi] = 2
                linked_iou[s][j] = best
            else:
                state[s][j, len(objs)] = 2
    # intervals
    P10 = {o["id"]: np.asarray(R["objects"][o["track"]]["P10"], float) for o in objs}
    contacts, flicker = [], 0
    for s in SIDES:
        for oi, oid in enumerate(ids):
            st = state[s][:, oi]
            j = 0
            while j < n:
                if st[j] != 2:
                    j += 1
                    continue
                a = j
                b = j
                while j < n and st[j] != 1:              # unknown frames inside a run do not end it
                    if st[j] == 2:
                        b = j
                    j += 1
                if j < n and j + 1 < n and st[j] == 1 and st[j + 1] == 2:
                    flicker += 1                         # a single 'off' frame between two 'on' frames
                seg = range(a, b + 1)
                moved = None
                if oid in P10:
                    pa, pb = P10[oid][min(ks[a], len(P10[oid]) - 1)], P10[oid][min(ks[b], len(P10[oid]) - 1)]
                    if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                        moved = round(float(np.linalg.norm(pb - pa) * 100), 1)
                tt = [tnames[touch[s][q]] for q in seg if touch[s][q] >= 0 and st[q] == 2]
                gg = [gnames[grasp[s][q]] for q in seg if grasp[s][q] >= 0 and st[q] == 2]
                contacts.append(dict(hand=s, object=oid, on_k=int(ks[a]), off_k=int(ks[b]), on_s=round(ks[a] / FPS, 2),
                                     off_s=round(ks[b] / FPS, 2), n_frames=int(b - a + 1),
                                     n_frames_hand_missing=int(np.sum(st[a:b + 1] == 0)),
                                     touch_types={t: tt.count(t) for t in sorted(set(tt))},
                                     grasp_types={g: gg.count(g) for g in sorted(set(gg))},
                                     mean_interaction_score=round(float(np.nanmean(iscore[s][a:b + 1])), 3),
                                     object_moved_cm=moved))
    contacts.sort(key=lambda c: (c["on_k"], c["hand"]))
    events = sorted([dict(t_s=c["on_s"], k=c["on_k"], type="contact_on", hand=c["hand"], object=c["object"]) for c in contacts]
                    + [dict(t_s=round((c["off_k"] + 1) / FPS, 2), k=c["off_k"] + 1, type="contact_off", hand=c["hand"], object=c["object"])
                       for c in contacts], key=lambda x: (x["k"], x["type"]))
    out = dict(episode=ep, video=V, fps=FPS, window=dict(k0=e["k0"], k1=e["k1"], start_s=e["start_s"], end_s=e["end_s"]),
               frames_scored=[int(ks[0]), int(ks[-1])],
               objects=[dict(id=o["id"], library=o.get("library"), role=o["role"], sam3_track=o["track"]) for o in objs],
               hands=list(SIDES), contacts=contacts, events=events,
               qc=dict(single_frame_off_gaps=flicker,
                       frames_hand_detected={s: int(np.sum(state[s][:, 0] > 0)) for s in SIDES}, n_frames=n,
                       unlinked_object_contact_frames={s: int(np.sum(state[s][:, -1] == 2)) for s in SIDES}),
               source=dict(model="hands23", hands23_commit=meta.get("hands23_git_commit"), detectron2_commit=meta.get("detectron2_git_commit"), weights_sha256=meta.get("weights_sha256"),
                           thresholds=meta.get("thresholds"), association=f"{match} IoU >= {ASSOC_IOU}", raw_file=p[0]))
    jdump(out, OUT / "contact" / f"{ep}{suf}.json")
    np.savez_compressed(OUT / "contact" / f"{ep}{suf}_frames.npz", k=ks, objects=np.array(ids),
                        **{f"{s}_state": state[s] for s in SIDES}, **{f"{s}_interaction_score": iscore[s] for s in SIDES},
                        **{f"{s}_touch": touch[s] for s in SIDES}, **{f"{s}_grasp": grasp[s] for s in SIDES},
                        **{f"{s}_linked_iou": linked_iou[s] for s in SIDES},
                        state_codes=np.array(["unknown (hand not detected)", "off", "on"]), touch_names=np.array(tnames),
                        grasp_names=np.array(gnames))
    timeline(ep, e, ks, ids, state, match, suf)
    print(ep, len(contacts), "intervals,", flicker, "single-frame gaps,", out["qc"], flush=True)
    return out


def timeline(ep, e, ks, ids, state, match="mask", suf=""):
    INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
    COL = {"left": "#2a78d6", "right": "#eb6834"}
    rows = [(s, oi) for s in SIDES for oi in range(len(ids)) if (state[s][:, oi] == 2).any()]
    fig, ax = plt.subplots(figsize=(11, 0.42 * max(len(rows), 1) + 1.4), facecolor="#fcfcfb")
    t = (ks - e["k0"]) / FPS
    for y, (s, oi) in enumerate(rows):
        st = state[s][:, oi]
        for j in range(len(ks)):
            if st[j] == 2:
                ax.add_patch(plt.Rectangle((t[j] - 0.05, y - 0.35), 0.1, 0.7, color=COL[s], lw=0))
            elif st[j] == 0:
                ax.add_patch(plt.Rectangle((t[j] - 0.05, y - 0.12), 0.1, 0.24, color=GRID, lw=0))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{s} hand · {ids[oi]}" for s, oi in rows], fontsize=8.5, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(t[0], t[-1])
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.axvline((e["k1"] - e["k0"]) / FPS, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("time from episode start (s); dotted = labelled episode; colour = Hands23 says in contact; thin grey = hand not detected",
                  fontsize=8.5, color=MUTED)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(f"{ep}: contact per hand and object (Hands23, raw per-frame, no smoothing; object matched by {match})",
                 fontsize=10, color=INK, loc="left")
    out = OUT / "qc" / ep
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"contact_timeline{suf}.png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    for ep in sys.argv[1:] or TRIAL:
        run(ep, "mask")
        run(ep, "box")
