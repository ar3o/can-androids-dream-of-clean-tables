"""Object instances from the SAM 3 inventory keyframes: cross-prompt de-duplication, on-table filter, joint tracking with stable ids,
container identification. Pure numpy/scipy/OpenCV (CPU).

Detections: sam3_inv[_cpu05].npz (frame k at 10 fps, prompt, prob, box, half-res mask 270x480).
1. keep prob >= P_KEEP; drop tiny masks and masks > 25 % of the image; drop masks that are mostly MediaPipe hand hull (> 60 %); drop masks
   whose centroid is off the table (table = convex hull of pixels that are bare wood in >= 30 % of the keyframes, dilated 25 px).
2. per frame, cross-prompt NMS on mask IoU >= 0.6 keeps the higher probability, except PREFER pairs (more specific label wins).
3. joint tracking over all keyframes of a video (labels flip between frames, e.g. mug/glass): Hungarian on cost = (1 - IoU) if IoU > 0.05,
   else 0.95 + distance / size (gated at 3 sizes + 30 px), + 0.25 if the prompts differ; a track may be unseen for up to GAP_KF keyframes
   (hand / body occlusion). Track label = probability-weighted majority of its prompts.
4. containers: the fruit bowl = the track labelled 'bowl of fruit', or the bowl track that coincides with 'bowl of fruit' detections most
   often (IMG_2516: the bowl is often empty); the bin = 'trash bin' track, else (IMG_2514: SAM 3 never says 'trash bin') the largest static
   'plastic bag' track (> 1.2 % of the image, seen in >= 50 % of keyframes) = the bin liner.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cleantables.perception.scene.common import WORK, meta, read_frame, unpack
from cleantables.perception.scene.vocab import CATEGORY

P_KEEP = 0.45
NMS_IOU = 0.6
GAP_S = 6.0          # a track survives this long unseen (hand / body occlusion)
H, W = 270, 480
PREFER = {frozenset(("bowl of fruit", "bowl")): "bowl of fruit", frozenset(("cardboard box", "flattened cardboard")): "cardboard box",
          frozenset(("trash bin", "plastic bag")): "trash bin", frozenset(("crisp packet", "plastic bag")): "crisp packet",
          frozenset(("green apple", "bowl of fruit")): None, frozenset(("banana", "bowl of fruit")): None}


COMPATIBLE = {frozenset(p) for p in (("mug", "glass"), ("bowl", "bowl of fruit"), ("plastic bag", "crisp packet"),
                                     ("flattened cardboard", "cardboard box"), ("plastic bag", "trash bin"))}


def inv_paths(video):
    """GPU 2 fps inventory if present, else the CPU keyframe passes (0.5 fps at offset 0, + offset 10 -> 1 fps), merged."""
    if (WORK / video / "sam3_inv.npz").exists():
        return [WORK / video / "sam3_inv.npz"]
    ps = [WORK / video / n for n in ("sam3_inv_cpu05.npz", "sam3_inv_cpu05b.npz") if (WORK / video / n).exists()]
    if not ps:
        raise FileNotFoundError(video)
    return ps


def inv_path(video):
    return inv_paths(video)[0]


_DETS = {}


def load_dets(video):
    if video in _DETS:
        return _DETS[video]
    parts = []
    for p in inv_paths(video):
        z = np.load(p)
        m = int(z["shape"][0])
        parts.append(dict(frame=z["frame"], prompt=z["prompt"], prob=z["prob"], box=z["box"],
                          masks=unpack(z["masks"], (m, H, W)) if m else np.zeros((0, H, W), bool), prompts=[str(x) for x in z["prompts"]],
                          frames=[int(x) for x in z["frames_done"]]))
    assert all(p["prompts"] == parts[0]["prompts"] for p in parts)
    d = dict(frame=np.concatenate([p["frame"] for p in parts]), prompt=np.concatenate([p["prompt"] for p in parts]),
             prob=np.concatenate([p["prob"] for p in parts]), box=np.concatenate([p["box"] for p in parts]),
             masks=np.concatenate([p["masks"] for p in parts]), prompts=parts[0]["prompts"],
             frames=sorted(set(sum([p["frames"] for p in parts], []))), path="+".join(x.name for x in inv_paths(video)))
    _DETS[video] = d
    return d


def table_hull(video, frames):
    cache = WORK / video / "table_hull.npy"
    if cache.exists():
        return np.load(cache)
    hm = _table_hull(video, frames)
    if (WORK / video / "depth_cpu.npz").exists() or (WORK / video / "depth.npz").exists():
        np.save(cache, hm)
    return hm


def _table_hull(video, frames):
    """Table region (270x480 bool). Preferred: pixels on the table plane (|dist| < 2.5 cm in the keyframe depth, RANSAC plane, in >= 30 %
    of the keyframes; the wood-coloured kitchen cabinets and the counter are other planes). Fallback: bare-wood colour."""
    from cleantables.perception.scene.signals import wood_mask
    dp = WORK / video / "depth_cpu.npz"
    if dp.exists():
        from cleantables.perception.scene.recon3d import backproject, ransac_plane
        z = np.load(dp)
        K = z["K"]
        acc = np.zeros((H, W), float)
        sel = np.linspace(0, len(z["k"]) - 1, min(12, len(z["k"]))).astype(int)
        for n in sel:
            D = z["depth"][n].astype(np.float32)
            wood = cv2.resize(wood_mask(read_frame(video, int(z["k"][n]))).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            P, _ = backproject(D, K, wood)
            nrm, c, _ = ransac_plane(P[::4], thr=0.015)
            Pa, (ys, xs) = backproject(D, K)
            on = np.abs((Pa - c) @ nrm) < 0.025
            acc[ys[on], xs[on]] += 1
        hull = (acc / len(sel)) > 0.3
    else:
        acc = np.zeros((H, W), float)
        for k in frames:
            acc += cv2.resize(wood_mask(read_frame(video, k)).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0
        hull = (acc / len(frames)) > 0.3
    hull = cv2.morphologyEx(hull.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hull = cv2.morphologyEx(hull, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    hm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hm, [cv2.convexHull(big)], 1)
    hm = cv2.dilate(hm, np.ones((11, 11), np.uint8))
    return hm > 0


def hand_hulls(video, k):
    hz = _mp(video)
    mk = np.zeros((H, W), np.uint8)
    for j in range(2):
        L = hz["hand_lm"][k, j]
        if np.isnan(L[0, 0]):
            continue
        pts = (L[:, :2] * (W / 960)).astype(np.int32)
        cv2.fillConvexPoly(mk, cv2.convexHull(pts), 1)
    return mk.astype(bool)


_MP = {}


def _mp(video):
    if video not in _MP:
        _MP[video] = dict(np.load(WORK / video / "mp.npz"))
    return _MP[video]


def mask_iou(a, b):
    i = np.logical_and(a, b).sum()
    return i / max(1, np.logical_or(a, b).sum())


def clean_frame(idx, d, hand, table):
    P = d["prompts"]
    keep = []
    for i in idx:
        if d["prob"][i] < P_KEEP:
            continue
        mk = d["masks"][i]
        a = mk.sum()
        if a < 12 or a > 0.25 * mk.size:
            continue
        if (mk & hand).sum() > 0.6 * a:
            continue
        ys, xs = np.nonzero(mk)
        if not table[int(ys.mean()), int(xs.mean())]:
            continue
        keep.append(i)
    keep.sort(key=lambda i: -d["prob"][i])
    out = []
    for i in keep:
        dup = None
        for j in out:
            if mask_iou(d["masks"][i], d["masks"][j]) >= NMS_IOU:
                dup = j
                break
        if dup is None:
            out.append(i)
            continue
        pi, pj = P[d["prompt"][i]], P[d["prompt"][dup]]
        pref = PREFER.get(frozenset((pi, pj)), "x")
        if pref is None and pi != pj:
            out.append(i)              # e.g. a banana inside the fruit bowl: both are real objects
        elif pref == pi:
            out[out.index(dup)] = i
    return out


def build_tracks(video):
    d = load_dets(video)
    frames = d["frames"]
    table = table_hull(video, frames[::max(1, len(frames) // 40)])
    by_f = {}
    for i, k in enumerate(d["frame"]):
        by_f.setdefault(int(k), []).append(i)
    kept = {k: clean_frame(by_f.get(k, []), d, hand_hulls(video, k), table) for k in frames}
    cent = {}
    for k, ids in kept.items():
        for i in ids:
            ys, xs = np.nonzero(d["masks"][i])
            cent[i] = (xs.mean(), ys.mean(), np.sqrt(len(xs)))
    P = d["prompts"]
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 5
    GAP_KF = max(2, int(round(GAP_S * 10 / step)))
    tracks, active = [], []
    for fi, k in enumerate(frames):
        dets = kept[k]
        cand = [t for t in active if fi - t["last_fi"] <= GAP_KF + 1]
        used = set()
        if cand and dets:
            C = np.full((len(cand), len(dets)), 9.0)
            for a, t in enumerate(cand):
                j = t["obs"][-1][1]
                for b, i in enumerate(dets):
                    iou = mask_iou(d["masks"][j], d["masks"][i])
                    cx, cy, sz = cent[i]
                    tx, ty, tsz = cent[j]
                    size = max(6.0, (sz + tsz) / 2)
                    if abs(np.log(max(sz, 1) / max(tsz, 1))) > np.log(2.2):
                        continue                                   # area ratio > ~5: not the same object
                    dist = np.hypot(cx - tx, cy - ty)
                    c = 1 - iou if iou > 0.05 else (0.95 + dist / size / 3 if dist < 3 * size + 30 else 9.0)
                    if c < 9 and d["prompt"][i] != d["prompt"][j]:
                        pair = frozenset((P[d["prompt"][i]], P[d["prompt"][j]]))
                        c = c + 0.25 if pair in COMPATIBLE else 9.0
                    C[a, b] = c
            r, c = linear_sum_assignment(C)
            for a, b in zip(r, c):
                if C[a, b] < 9:
                    t = cand[a]
                    t["obs"].append((k, dets[b])); t["last_fi"] = fi
                    used.add(b)
        for b, i in enumerate(dets):
            if b in used:
                continue
            t = dict(obs=[(k, i)], last_fi=fi)
            active.append(t)
            tracks.append(t)
        active = [t for t in active if fi - t["last_fi"] <= GAP_KF + 1]
    P = d["prompts"]
    for t in tracks:
        votes = {}
        for k, i in t["obs"]:
            votes[P[d["prompt"][i]]] = votes.get(P[d["prompt"][i]], 0) + float(d["prob"][i])
        t["prompt"] = max(votes, key=votes.get)
        t["votes"] = votes
        t["category"] = CATEGORY[t["prompt"]]
        t["frames"] = [k for k, _ in t["obs"]]
        t["cent"] = np.array([cent[i][:2] for _, i in t["obs"]]) * (960 / W)        # work px (960x540)
        t["size"] = np.array([cent[i][2] for _, i in t["obs"]]) * (960 / W)
        t["prob"] = np.array([float(d["prob"][i]) for _, i in t["obs"]])
    _containers(tracks, d, frames, kept)
    return d, tracks, table, kept


def _containers(tracks, d, frames, kept=None):
    P = d["prompts"]
    # any bowl track that holds fruit (a banana / apple detection centred inside its filled hull) in >= 30 % of its sightings
    if kept is not None:
        for t in tracks:
            if t["prompt"] != "bowl" or len(t["obs"]) < 4:
                continue
            n_in = 0
            for k, i in t["obs"]:
                cnts, _ = cv2.findContours(d["masks"][i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hull = np.zeros((H, W), np.uint8)
                cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cnts))], 1)
                for j in kept.get(k, []):
                    if P[d["prompt"][j]] in ("banana", "green apple"):
                        ys, xs = np.nonzero(d["masks"][j])
                        if hull[int(ys.mean()), int(xs.mean())]:
                            n_in += 1
                            break
            if n_in >= 0.3 * len(t["obs"]):
                t["category"], t["container_rule"] = "container/fruit_bowl", f"bowl holding fruit in {n_in}/{len(t['obs'])} sightings"
    fb = [t for t in tracks if t["prompt"] == "bowl of fruit"]
    if not fb:
        best = None
        for t in tracks:
            if t["prompt"] != "bowl":
                continue
            n = sum(1 for k, i in t["obs"] if t["votes"].get("bowl of fruit", 0) > 0)
            score = t["votes"].get("bowl of fruit", 0) / max(1e-6, sum(t["votes"].values()))
            if score > 0.15 and (best is None or score > best[0]):
                best = (score, t)
        if best:
            best[1]["category"], best[1]["container_rule"] = "container/fruit_bowl", "bowl coinciding with 'bowl of fruit'"
    if not any(t["prompt"] == "trash bin" for t in tracks):
        big = [t for t in tracks if t["prompt"] == "plastic bag" and np.median(t["size"]) ** 2 > 0.012 * 960 * 540 and len(t["frames"]) >= 0.5 * len(frames)]
        if big:
            t = max(big, key=lambda t: np.median(t["size"]))
            t["category"], t["container_rule"] = "container/bin", "largest static 'plastic bag' (bin liner)"
