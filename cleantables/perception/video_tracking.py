"""Object identities from SAM 3 video masklets (sam3_video.npz) instead of per-keyframe detections + Hungarian / CoTracker linking.

SAM 3's video tracker keeps one id per object while it moves (memory-based propagation); the keyframe matcher in
cleantables.perception.tracking can give a carried object a new id. Here:
  1. per masklet and frame: keep the mask if prob >= P_KEEP (default 0: SAM 3's own threshold already applies), sane area, centroid on the (dilated) table mask, < 60 % covered by the
     'hand' masklets of that frame, and on the table in 3D (depth points not below the plane, centre within its extent);
  2. cross-phrase merge: two masklets (any phrases) whose masks coincide (IoU >= 0.5 on >= 60 % of >= 3 co-visible frames) are one
     object; it is named by the phrase with the highest summed probability;
  3. composite objects ('stack of ...'): an object whose mask is >= 70 % covered by >= 2 other objects on most of its frames is dropped;
  4. re-identification only where SAM 3 itself started a new id (object fully hidden, e.g. behind the person): never co-visible,
     starts <= REID_GAP frames after the other ends, same colour (Lab dE < 16) and footprint -> same object;
  5. per object and tracked frame (5 fps) a 3D oriented box from mask + depth (tracking.obb), hand overlap, colour.
Writes the same tracks.pkl layout as cleantables.perception.tracking (objects / kept / det_box / phrases / keyframes) + sam3_vinv.npz
(the per-frame detections in the sam3_inv format), so the downstream stages (vision, events, frame_model) read it unchanged.
  $VLA_PY -m cleantables.perception.video_tracking VIDEOS...   (needs data/generic/<V>/sam3_video.npz)
"""
from __future__ import annotations

import pickle
import sys

import cv2
import numpy as np

from cleantables.perception.scene.recon3d import backproject, to_table
from cleantables.perception.boxes.scene_export import table_extent

from cleantables.perception.common import AGENT_PHRASES, depth, frame, geom, log, meta, names, vdir
from cleantables.perception.objects import iou
from cleantables.perception.tracking import obb

P_KEEP, MERGE_IOU, REID_GAP, REID_DE = 0.45, 0.5, 80, 16.0
# VTRACK_P_KEEP: extra probability gate (default 0: the masklets already passed SAM 3's own detection threshold).
# VTRACK_RULES=off disables the identity rules (cross-phrase merge, composite drop, re-id) for an off-vs-on comparison;
# VTRACK_OUT writes to another folder instead of overwriting tracks.pkl.
import os
from pathlib import Path
P_KEEP = float(os.environ.get("VTRACK_P_KEEP", "0"))
RULES_ON = os.environ.get("VTRACK_RULES", "on") != "off"
VT_OUT = os.environ.get("VTRACK_OUT")
H, W = 270, 480


def load_video(v):
    z = np.load(vdir(v) / os.environ.get("VTRACK_SRC", "sam3_video.npz"))
    phrases = [str(p) for p in z["phrases"]]
    ks = z["frames"].astype(int)
    M = []                                                  # masklets: dict(phrase, prob (n_t,), masks (n_t, H, W))
    for pi, ph in enumerate(phrases):
        s = z[f"p{pi}_shape"]
        n_o, n_t = int(s[0]), int(s[1])
        if n_o == 0:
            continue
        mk = np.unpackbits(z[f"p{pi}_masks"])[:n_o * n_t * H * W].reshape(n_o, n_t, H, W).astype(bool)
        for j in range(n_o):
            M.append(dict(phrase=ph, pi=pi, prob=z[f"p{pi}_probs"][j], masks=mk[j]))
    return phrases, ks, M


def run(v):
    od = Path(VT_OUT) / v if VT_OUT else vdir(v)
    od.mkdir(parents=True, exist_ok=True)
    out_p = od / "tracks.pkl"
    if out_p.exists():
        return pickle.load(open(out_p, "rb"))
    n = meta(v)["n"]
    G = geom(v)
    K, T_all, scale = G["K"], G["T_all"], G["scale"]
    D10 = depth(v)
    table = cv2.dilate(G["table"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ext = np.array(table_extent(G["T"], K, G["table"]))
    phrases, ks, M = load_video(v)
    nt = len(ks)
    hand = np.zeros((nt, H, W), bool)
    for m in M:
        if m["phrase"] in AGENT_PHRASES:
            hand |= m["masks"] & (m["prob"][:, None, None] >= 0.4)
    obj = [m for m in M if m["phrase"] not in AGENT_PHRASES]
    # ---- 1. per-frame validity
    for m in obj:
        ok = np.zeros(nt, bool)
        for t in range(nt):
            mk = m["masks"][t]
            a = mk.sum()
            if m["prob"][t] < P_KEEP or a < 12 or a > 0.25 * mk.size or (mk & hand[t]).sum() > 0.6 * a:
                continue
            ys, xs = np.nonzero(mk)
            if not table[int(ys.mean()), int(xs.mean())]:
                continue
            k = ks[t]
            P, _ = backproject(np.asarray(D10[k], np.float32) * scale[k], K, mk)
            if len(P) < 10:
                continue
            Q = to_table(T_all[k], P)
            cx, cy = np.median(Q[:, 0]), np.median(Q[:, 1])
            if np.median(Q[:, 2]) < -0.03 or not (ext[0, 0] - 0.05 <= cx <= ext[0, 1] + 0.05 and ext[1, 0] - 0.05 <= cy <= ext[1, 1] + 0.05):
                continue
            ok[t] = True
        m["ok"] = ok
    obj = [m for m in obj if m["ok"].sum() >= 3]
    J = len(obj)
    par = list(range(J))

    def root(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    # ---- 2. cross-phrase (and duplicate) merge
    for a in range(J):
        for b in range(a + 1, J):
            F = np.nonzero(obj[a]["ok"] & obj[b]["ok"])[0]
            if len(F) < 3 or not RULES_ON:
                continue
            ious = np.array([iou(obj[a]["masks"][t], obj[b]["masks"][t]) for t in F])
            if (ious >= MERGE_IOU).mean() >= 0.6:
                par[root(b)] = root(a)
    groups = {}
    for j in range(J):
        groups.setdefault(root(j), []).append(j)
    O = []
    for g in groups.values():
        best = np.full(nt, -1)
        bp = np.zeros(nt, np.float32)
        for j in g:
            for t in np.nonzero(obj[j]["ok"])[0]:
                if obj[j]["prob"][t] > bp[t]:
                    bp[t] = obj[j]["prob"][t]; best[t] = j
        votes = {}
        for j in g:
            votes[obj[j]["phrase"]] = votes.get(obj[j]["phrase"], 0) + float(obj[j]["prob"][obj[j]["ok"]].sum())
        O.append(dict(members=g, best=best, prob=bp, votes=votes, phrase=max(votes, key=votes.get)))

    def mask(o, t):
        return obj[o["best"][t]]["masks"][t]
    # ---- 3. composite objects
    drop = set()
    for i, o in enumerate(O):
        T_ = np.nonzero(o["best"] >= 0)[0]
        hits = 0
        for t in T_:
            mi = mask(o, t); a = mi.sum()
            parts = [p for jj, p in enumerate(O) if jj != i and p["best"][t] >= 0 and (mask(p, t) & mi).sum() > 0.6 * mask(p, t).sum()
                     and mask(p, t).sum() < 0.8 * a]
            if len(parts) >= 2:
                u = np.zeros_like(mi)
                for p in parts:
                    u |= mask(p, t)
                hits += (u & mi).sum() >= 0.7 * a
        if len(T_) and hits >= 0.5 * len(T_):
            drop.add(i)
    if not RULES_ON:
        drop = set()
    O = [o for i, o in enumerate(O) if i not in drop]
    # ---- 5 (before re-id, it needs colour + footprint). detections + 3D boxes
    labs = {}
    det_frame, det_prompt, det_prob, det_box_xy, det_masks = [], [], [], [], []
    det_box, kept = {}, {int(k): [] for k in ks}
    for o in O:
        o["dets"] = {}
        for t in np.nonzero(o["best"] >= 0)[0]:
            k = int(ks[t]); mk = mask(o, t)
            i = len(det_frame)
            ys, xs = np.nonzero(mk)
            det_frame.append(k); det_prompt.append(phrases.index(obj[o["best"][t]]["phrase"])); det_prob.append(float(o["prob"][t]))
            det_box_xy.append([xs.min() * 2, ys.min() * 2, xs.max() * 2 + 2, ys.max() * 2 + 2]); det_masks.append(mk)
            kept[k].append(i); o["dets"][t] = i
            Dk = np.asarray(D10[k], np.float32) * scale[k]
            b = obb(Dk, K, T_all[k], mk)
            if b is not None:
                hd = cv2.dilate(hand[t].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                if k not in labs:
                    labs[k] = cv2.cvtColor(cv2.resize(frame(v, k), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
                m2 = cv2.erode(mk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~hd
                b.update(hand_frac=float((hd & mk).sum() / max(1, mk.sum())), prob=float(o["prob"][t]), phrase=o["phrase"],
                         uv=[float(xs.mean() * 2 + 1), float(ys.mean() * 2 + 1)],
                         lab=(np.median(labs[k][m2], 0) if m2.sum() > 5 else np.median(labs[k][mk], 0)).tolist(),
                         bbox=det_box_xy[-1])
            det_box[i] = b
    # ---- 4. re-identification across SAM 3 id restarts
    def stats(o):
        T_ = np.nonzero(o["best"] >= 0)[0]
        bs = [det_box[o["dets"][t]] for t in T_ if det_box.get(o["dets"][t])]
        good = [b for b in bs if b["hand_frac"] < 0.2] or bs
        if not good:
            return None
        return dict(first=T_.min(), last=T_.max(), T=set(T_.tolist()), lab=np.median([b["lab"] for b in good], 0),
                    L=np.median([max(b["L"], b["W"]) for b in good]), W=np.median([min(b["L"], b["W"]) for b in good]))
    gap_t = max(1, int(round(REID_GAP / max(1, int(np.median(np.diff(ks))) if len(ks) > 1 else 1))))
    n_reid = 0
    while RULES_ON:
        st = [stats(o) for o in O]
        best = None
        for b_, B in enumerate(st):
            for a_, A in enumerate(st):
                if a_ == b_ or A is None or B is None or not (0 < B["first"] - A["last"] <= gap_t) or A["T"] & B["T"]:
                    continue
                dE = float(np.linalg.norm((A["lab"] - B["lab"]) * np.array([0.5, 1, 1])))
                rl, rw = max(A["L"], B["L"]) / min(A["L"], B["L"]), max(A["W"], B["W"]) / min(A["W"], B["W"])
                if dE < REID_DE and rl < 1.5 and rw < 2.2:
                    cost = dE / REID_DE + (rl - 1) + (rw - 1) + (B["first"] - A["last"]) / gap_t
                    if best is None or cost < best[0]:
                        best = (cost, a_, b_)
        if best is None:
            break
        A, B = O[best[1]], O[best[2]]
        A["dets"].update(B["dets"]); A["members"] += B["members"]
        for p_, w_ in B["votes"].items():
            A["votes"][p_] = A["votes"].get(p_, 0) + w_
        A["phrase"] = max(A["votes"], key=A["votes"].get)
        A["best"] = np.where(A["best"] >= 0, A["best"], B["best"])
        O.pop(best[2]); n_reid += 1
    # ---- output in the tracking layout
    objs = []
    for o in sorted(O, key=lambda o: min(o["dets"])):
        T_ = sorted(o["dets"])
        sight_k = np.array([int(ks[t]) for t in T_])
        boxes = [det_box[o["dets"][t]] for t in T_]
        P = np.full((n, 3), np.nan, np.float32)
        for k, b in zip(sight_k, boxes):
            if b:
                P[k] = [b["x"], b["y"], b["top"]]
        cols = []
        for t in sorted(T_, key=lambda t: -o["prob"][t] if o["best"][t] >= 0 else 0)[:3]:
            mk = cv2.erode(det_masks[o["dets"][t]].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if mk.sum() > 5:
                im = cv2.resize(frame(v, int(ks[t])), (W, H), interpolation=cv2.INTER_AREA)
                cols.append(np.median(im[mk], 0)[::-1])
        objs.append(dict(members=o["members"], phrase=o["phrase"], votes=o["votes"], sight_k=sight_k, boxes=boxes,
                         det=[o["dets"][t] for t in T_], uv=None, P10=P,
                         rgb=[int(c) for c in np.median(cols, 0)] if cols else [128, 128, 128]))
    m_all = np.array(det_masks) if det_masks else np.zeros((0, H, W), bool)
    np.savez_compressed(od / "sam3_vinv.npz", frame=np.array(det_frame, np.int32), prompt=np.array(det_prompt, np.int32),
                        prob=np.array(det_prob, np.float32), box=np.array(det_box_xy, np.float32).reshape(-1, 4),
                        masks=np.packbits(m_all), shape=np.array(m_all.shape), prompts=np.array(phrases), frames_done=np.array(ks, np.int32))
    np.savez_compressed(od / "hand_video.npz", frames=np.array(ks, np.int32), masks=np.packbits(hand), shape=np.array(hand.shape))
    res = dict(video=v, n=n, source="sam3_video", objects=objs, n_masklets=len(M), kept=kept, det_box=det_box, phrases=phrases,
               n_dets=len(det_frame), n_kept=len(det_frame), keyframes=[int(k) for k in ks], n_reid=n_reid, n_composite=len(drop))
    pickle.dump(res, open(out_p, "wb"))
    log(f"{v} video tracking: {len(M)} SAM 3 masklets -> {len(objs)} objects ({len(drop)} composite dropped, {n_reid} re-identified): "
        f"{', '.join(o['phrase'] for o in objs)}")
    return res


if __name__ == "__main__":
    for v in names(sys.argv[1:]):
        run(v)
