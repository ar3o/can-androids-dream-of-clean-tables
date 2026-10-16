"""Object identities and per-frame 3D boxes from the SAM 3 video masklets -> tracks.pkl, sam3_vinv.npz, hand_video.npz.

    python -m cleantables.perception.tracks VIDEO [VIDEO ...]

1. per masklet and tracked frame, keep the mask if it is confident, of sane size, centred on the table, less than 60 %
   covered by the hands and on the table in 3D;
2. masklets of any phrases whose masks coincide on most co-visible frames are one object, named by the phrase with the
   highest summed probability;
3. objects whose masks are mostly covered by two or more other objects ('stack of ...') are dropped;
4. per object and frame: a 3D oriented box from mask + depth, the hand overlap and the colour;
5. where SAM 3 started a new id for a hidden object (behind the person), a track that starts soon after another ended,
   is never seen together with it and has the same colour and footprint continues it.
tracks.pkl: objects (phrase, sight_k, boxes, det, P10, rgb, ...), kept (frame -> detection ids), det_box (id -> box).
"""

from __future__ import annotations

import argparse
import pickle

import cv2
import numpy as np

from cleantables import paths

from . import geometry as G
from . import hands
from .files import AGENT_PHRASES, H_LO, W_LO, depth, geometry, masklets, n_frames, read_frame, video_dir
from .objects import iou

MIN_PROB = 0.45
MERGE_IOU = 0.5
REID_GAP_FRAMES = 80
REID_MAX_DE = 16.0
MIN_AREA_PX = 12
MAX_AREA_FRACTION = 0.25
MAX_HAND_COVER = 0.6
TABLE_DILATE = np.ones((15, 15), np.uint8)


def oriented_box(Dk, K, T, mask):
    """3D oriented box (table frame) from one mask + metric depth: dict(x, y, top, L, W, yaw, n) or None."""
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8))
    if eroded.sum() >= 15:
        m = eroded
    P, _ = G.backproject(Dk, K, m.astype(bool))
    if len(P) < 15:
        return None
    Q = G.to_table(T, P)
    # the 90th percentile is biased up by depth bleeding at the mask border: re-centre on the top face
    top = float(np.percentile(Q[:, 2], 90))
    band = Q[np.abs(Q[:, 2] - top) < max(0.015, 0.1 * top)]
    if len(band) >= 10:
        top = float(np.median(band[:, 2]))
    above = Q[Q[:, 2] > min(0.004, 0.3 * top)]
    if len(above) < 10:
        above = Q
    x, y, L, W, yaw = G.rect_extent(above[:, :2])
    pixel = 2.0 * float(np.median(P[:, 2])) / K[0, 0]  # adds back the erosion, one half-res pixel per side
    return dict(x=x, y=y, top=max(top, 0.003), L=L + 2 * pixel, W=W + 2 * pixel, yaw=yaw, n=int(len(Q)))


def valid_frames(m, hand, ks, D10, g, table, extent) -> np.ndarray:
    n_t = len(ks)
    ok = np.zeros(n_t, bool)
    for t in range(n_t):
        mk = m["masks"][t]
        area = mk.sum()
        if (
            m["prob"][t] < MIN_PROB
            or area < MIN_AREA_PX
            or area > MAX_AREA_FRACTION * mk.size
            or (mk & hand[t]).sum() > MAX_HAND_COVER * area
        ):
            continue
        ys, xs = np.nonzero(mk)
        if not table[int(ys.mean()), int(xs.mean())]:
            continue
        k = ks[t]
        P, _ = G.backproject(np.asarray(D10[k], np.float32) * g["scale"][k], g["K"], mk)
        if len(P) < 10:
            continue
        Q = G.to_table(g["T_all"][k], P)
        cx, cy = np.median(Q[:, 0]), np.median(Q[:, 1])
        if np.median(Q[:, 2]) < -0.03 or not (
            extent[0, 0] - 0.05 <= cx <= extent[0, 1] + 0.05 and extent[1, 0] - 0.05 <= cy <= extent[1, 1] + 0.05
        ):
            continue
        ok[t] = True
    return ok


def merge_masklets(obj: list[dict], n_t: int) -> list[dict]:
    """Union-find over masklets whose masks coincide (IoU >= MERGE_IOU on >= 60 % of >= 3 co-visible frames)."""
    parent = list(range(len(obj)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(obj)):
        for b in range(a + 1, len(obj)):
            shared = np.nonzero(obj[a]["ok"] & obj[b]["ok"])[0]
            if len(shared) < 3:
                continue
            ious = np.array([iou(obj[a]["masks"][t], obj[b]["masks"][t]) for t in shared])
            if (ious >= MERGE_IOU).mean() >= 0.6:
                parent[root(b)] = root(a)
    groups = {}
    for j in range(len(obj)):
        groups.setdefault(root(j), []).append(j)
    objects = []
    for members in groups.values():
        best = np.full(n_t, -1)
        best_prob = np.zeros(n_t, np.float32)
        for j in members:
            for t in np.nonzero(obj[j]["ok"])[0]:
                if obj[j]["prob"][t] > best_prob[t]:
                    best_prob[t] = obj[j]["prob"][t]
                    best[t] = j
        votes = {}
        for j in members:
            votes[obj[j]["phrase"]] = votes.get(obj[j]["phrase"], 0) + float(obj[j]["prob"][obj[j]["ok"]].sum())
        objects.append(dict(members=members, best=best, prob=best_prob, votes=votes, phrase=max(votes, key=votes.get)))
    return objects


def drop_composites(objects: list[dict], mask) -> tuple[list[dict], int]:
    """Drop objects whose mask is >= 70 % covered by >= 2 smaller objects on at least half of their frames."""
    drop = set()
    for i, o in enumerate(objects):
        frames = np.nonzero(o["best"] >= 0)[0]
        hits = 0
        for t in frames:
            mi = mask(o, t)
            area = mi.sum()
            parts = [
                p
                for j, p in enumerate(objects)
                if j != i
                and p["best"][t] >= 0
                and (mask(p, t) & mi).sum() > 0.6 * mask(p, t).sum()
                and mask(p, t).sum() < 0.8 * area
            ]
            if len(parts) >= 2:
                union = np.zeros_like(mi)
                for p in parts:
                    union |= mask(p, t)
                hits += (union & mi).sum() >= 0.7 * area
        if len(frames) and hits >= 0.5 * len(frames):
            drop.add(i)
    return [o for i, o in enumerate(objects) if i not in drop], len(drop)


def reidentify(objects: list[dict], det_box: dict, gap_t: int) -> int:
    """Join tracks that SAM 3 split while the object was hidden; returns the number of joins."""

    def stats(o):
        frames = np.nonzero(o["best"] >= 0)[0]
        boxes = [det_box[o["dets"][t]] for t in frames if det_box.get(o["dets"][t])]
        good = [b for b in boxes if b["hand_frac"] < 0.2] or boxes
        if not good:
            return None
        return dict(
            first=frames.min(),
            last=frames.max(),
            T=set(frames.tolist()),
            lab=np.median([b["lab"] for b in good], 0),
            L=np.median([max(b["L"], b["W"]) for b in good]),
            W=np.median([min(b["L"], b["W"]) for b in good]),
        )

    joins = 0
    while True:
        st = [stats(o) for o in objects]
        best = None
        for b_, B in enumerate(st):
            for a_, A in enumerate(st):
                if a_ == b_ or A is None or B is None or not (0 < B["first"] - A["last"] <= gap_t) or A["T"] & B["T"]:
                    continue
                dE = float(np.linalg.norm((A["lab"] - B["lab"]) * np.array([0.5, 1, 1])))
                rl, rw = max(A["L"], B["L"]) / min(A["L"], B["L"]), max(A["W"], B["W"]) / min(A["W"], B["W"])
                if dE < REID_MAX_DE and rl < 1.5 and rw < 2.2:
                    cost = dE / REID_MAX_DE + (rl - 1) + (rw - 1) + (B["first"] - A["last"]) / gap_t
                    if best is None or cost < best[0]:
                        best = (cost, a_, b_)
        if best is None:
            return joins
        A, B = objects[best[1]], objects[best[2]]
        A["dets"].update(B["dets"])
        A["members"] += B["members"]
        for p, w in B["votes"].items():
            A["votes"][p] = A["votes"].get(p, 0) + w
        A["phrase"] = max(A["votes"], key=A["votes"].get)
        A["best"] = np.where(A["best"] >= 0, A["best"], B["best"])
        objects.pop(best[2])
        joins += 1


def build(video: str) -> dict:
    out_path = paths.perception_dir(video) / "tracks.pkl"
    if out_path.exists():
        with open(out_path, "rb") as f:
            return pickle.load(f)
    n = n_frames(video)
    g = geometry(video)
    K, T_all, scale = g["K"], g["T_all"], g["scale"]
    D10 = depth(video)
    table = cv2.dilate(g["table"].astype(np.uint8), TABLE_DILATE).astype(bool)
    extent = np.array(G.table_extent(g["T"], K, g["table"]))
    phrases, ks, M = masklets(video)
    n_t = len(ks)
    hand = hands.hand_union(M, n_t)
    obj = [m for m in M if m["phrase"] not in AGENT_PHRASES]
    for m in obj:
        m["ok"] = valid_frames(m, hand, ks, D10, g, table, extent)
    obj = [m for m in obj if m["ok"].sum() >= 3]
    objects = merge_masklets(obj, n_t)

    def mask(o, t):
        return obj[o["best"][t]]["masks"][t]

    objects, n_composite = drop_composites(objects, mask)

    lab_frames = {}
    det_frame, det_prompt, det_prob, det_xyxy, det_masks = [], [], [], [], []
    det_box, kept = {}, {int(k): [] for k in ks}
    for o in objects:
        o["dets"] = {}
        for t in np.nonzero(o["best"] >= 0)[0]:
            k = int(ks[t])
            mk = mask(o, t)
            i = len(det_frame)
            ys, xs = np.nonzero(mk)
            det_frame.append(k)
            det_prompt.append(phrases.index(obj[o["best"][t]]["phrase"]))
            det_prob.append(float(o["prob"][t]))
            det_xyxy.append([xs.min() * 2, ys.min() * 2, xs.max() * 2 + 2, ys.max() * 2 + 2])
            det_masks.append(mk)
            kept[k].append(i)
            o["dets"][t] = i
            b = oriented_box(np.asarray(D10[k], np.float32) * scale[k], K, T_all[k], mk)
            if b is not None:
                hand_k = cv2.dilate(hand[t].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                if k not in lab_frames:
                    small = cv2.resize(read_frame(video, k), (W_LO, H_LO), interpolation=cv2.INTER_AREA)
                    lab_frames[k] = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
                inner = cv2.erode(mk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~hand_k
                colour = np.median(lab_frames[k][inner], 0) if inner.sum() > 5 else np.median(lab_frames[k][mk], 0)
                b.update(
                    hand_frac=float((hand_k & mk).sum() / max(1, mk.sum())),
                    prob=float(o["prob"][t]),
                    phrase=o["phrase"],
                    uv=[float(xs.mean() * 2 + 1), float(ys.mean() * 2 + 1)],
                    lab=colour.tolist(),
                    bbox=det_xyxy[-1],
                )
            det_box[i] = b

    step = int(np.median(np.diff(ks))) if len(ks) > 1 else 1
    gap_t = max(1, int(round(REID_GAP_FRAMES / max(1, step))))
    n_reid = reidentify(objects, det_box, gap_t)

    tracks = [track_record(video, o, ks, det_box, det_masks, n) for o in sorted(objects, key=lambda o: min(o["dets"]))]
    save_detections(video, det_frame, det_prompt, det_prob, det_xyxy, det_masks, phrases, ks)
    hands.save(video, ks, hand)
    result = dict(
        video=video,
        n=n,
        source="sam3_video",
        objects=tracks,
        n_masklets=len(M),
        kept=kept,
        det_box=det_box,
        phrases=phrases,
        n_dets=len(det_frame),
        n_kept=len(det_frame),
        keyframes=[int(k) for k in ks],
        n_reid=n_reid,
        n_composite=n_composite,
    )
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(
        f"{video}: {len(M)} masklets -> {len(tracks)} objects ({n_composite} composite, {n_reid} re-identified): "
        f"{', '.join(o['phrase'] for o in tracks)}",
        flush=True,
    )
    return result


def track_record(video, o, ks, det_box, det_masks, n) -> dict:
    frames = sorted(o["dets"])
    sight_k = np.array([int(ks[t]) for t in frames])
    boxes = [det_box[o["dets"][t]] for t in frames]
    P10 = np.full((n, 3), np.nan, np.float32)
    for k, b in zip(sight_k, boxes, strict=True):
        if b:
            P10[k] = [b["x"], b["y"], b["top"]]
    colours = []
    for t in sorted(frames, key=lambda t: -o["prob"][t] if o["best"][t] >= 0 else 0)[:3]:
        mk = cv2.erode(det_masks[o["dets"][t]].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if mk.sum() > 5:
            im = cv2.resize(read_frame(video, int(ks[t])), (W_LO, H_LO), interpolation=cv2.INTER_AREA)
            colours.append(np.median(im[mk], 0)[::-1])
    return dict(
        members=o["members"],
        phrase=o["phrase"],
        votes=o["votes"],
        sight_k=sight_k,
        boxes=boxes,
        det=[o["dets"][t] for t in frames],
        uv=None,
        P10=P10,
        rgb=[int(c) for c in np.median(colours, 0)] if colours else [128, 128, 128],
    )


def save_detections(video, frame, prompt, prob, xyxy, masks, phrases, ks) -> None:
    """Per-frame masks of every tracked object (detection id = index), in the keyframe-detection layout."""
    m_all = np.array(masks) if masks else np.zeros((0, H_LO, W_LO), bool)
    np.savez_compressed(
        video_dir(video) / "sam3_vinv.npz",
        frame=np.array(frame, np.int32),
        prompt=np.array(prompt, np.int32),
        prob=np.array(prob, np.float32),
        box=np.array(xyxy, np.float32).reshape(-1, 4),
        masks=np.packbits(m_all),
        shape=np.array(m_all.shape),
        prompts=np.array(phrases),
        frames_done=np.array(ks, np.int32),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    for video in ap.parse_args().videos:
        build(video)


if __name__ == "__main__":
    main()
