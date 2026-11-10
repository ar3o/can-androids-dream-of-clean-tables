"""Generic object tracking: SAM 3 inventory (any phrases) -> object identities -> 10 fps 3D trajectories + per-sighting 3D oriented boxes.

1. clean   : per 2 fps keyframe, detections with prob >= P_KEEP, sane area, centroid on the table (det3d table mask, dilated), not mostly
             hand ('hand' masks of the same frame), and on the table in 3D (depth points not below the plane, centre within its extent).
             Cross-phrase duplicates: mask IoU >= NMS_IOU keeps the higher probability (no preference table). Composite masks (a 'stack' / 'group' detection that is >= 70 % covered by >= 2 other kept detections) are
             dropped: the group is its members.
2. track   : Hungarian over keyframes (as in scene.tracks) on 1 - IoU / centroid distance; a label change is allowed only when
             the masks overlap well (IoU >= RELABEL_IOU) - no hand-written compatible-label table. Gap up to GAP_S unseen. A match is
             refused when the 3D box jumps (footprint area x3, or top height > 12 cm with no hand on either detection).
3. points  : CoTracker3 (offline, frozen) seeded inside each track's masks at half resolution, windows of WIN frames (flow.cotrack).
4. link    : two tracks are the same object if the points of one follow the other's position for >= 5 frames (median < 14 px) and
             they are never seen as two separate detections on the same keyframe (no label rules);
             then appearance re-identification: a track starting <= 8 s after another ended, never co-visible, same colour (Lab dE < 16)
             and footprint (long side x1.5, short side x2.2) continues it (the object was carried behind the hand / arm and put down elsewhere).
5. 3D      : 10 fps image position = median point displacement (stitch), depth = MoGe depth10 at that pixel x the per-frame table
             scale -> table frame (z up). Per sighting (2 fps) a 3D oriented box from the mask's depth points: footprint rectangle
             (x, y, L, W, yaw; boxes.geometry.rect_extent on points above the table) and top height (p90 -> top-face median).
  $VLA_PY -m cleantables.perception.tracking VIDEOS...  [--cpu]   -> data/generic/<V>/tracks.pkl  (+ cotrack.npz)
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cleantables.perception.scene.recon3d import backproject, to_table
from cleantables.perception.boxes.geometry import rect_extent

from cleantables.perception.common import AGENT_PHRASES, depth, frame, geom, gpu_free, log, meta, names, vdir
from cleantables.perception.objects import iou, load_dets

P_KEEP, NMS_IOU, RELABEL_IOU, GAP_S = 0.45, 0.6, 0.3, 6.0
WIN, OV, NQ = 32, 8, 24
REID_GAP, REID_DE = 80, 16.0
H, W = 270, 480


# ------------------------------------------------------------------ 0. load
def reusable_hand(v):
    from cleantables.perception.common import WORK
    p = WORK / v / "sam3_box.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return p if "hand" in [str(x) for x in z["prompts"]] else None


def load_inventory(v):
    """SAM 3 inventory (data/generic/<V>/sam3_inv.npz) + the 'hand' detections of the earlier pass when 'hand' was not prompted.
    When the video tracker ran (video_tracking), its per-frame detections (sam3_vinv.npz) are the inventory (det ids match tracks.pkl)."""
    if (vdir(v) / "sam3_vinv.npz").exists():
        return load_dets(vdir(v) / "sam3_vinv.npz")
    d = load_dets(vdir(v) / "sam3_inv.npz")
    if any(p in AGENT_PHRASES for p in d["prompts"]):
        return d
    hp = reusable_hand(v)
    if hp is None:
        return d
    h = load_dets(hp)
    hi = h["prompts"].index("hand")
    sel = np.nonzero((h["prompt"] == hi) & np.isin(h["frame"], d["frames"]))[0]
    d["prompts"] = d["prompts"] + ["hand"]
    d["frame"] = np.r_[d["frame"], h["frame"][sel]]; d["prompt"] = np.r_[d["prompt"], np.full(len(sel), len(d["prompts"]) - 1)]
    d["prob"] = np.r_[d["prob"], h["prob"][sel]]; d["box"] = np.r_[d["box"].reshape(-1, 4), h["box"][sel]]
    d["masks"] = np.concatenate([d["masks"], h["masks"][sel]])
    return d


# ------------------------------------------------------------------ 1. clean
def clean(d, table):
    P = d["prompts"]
    agent = [i for i, p in enumerate(P) if p in AGENT_PHRASES]
    by_f = {}
    for i, k in enumerate(d["frame"]):
        by_f.setdefault(int(k), []).append(i)
    kept = {}
    for k in d["frames"]:
        ids = by_f.get(k, [])
        hand = np.zeros((H, W), bool)
        for i in ids:
            if d["prompt"][i] in agent and d["prob"][i] >= 0.4:
                hand |= d["masks"][i]
        cand = []
        for i in ids:
            if d["prompt"][i] in agent or d["prob"][i] < P_KEEP:
                continue
            mk = d["masks"][i]
            a = mk.sum()
            if a < 12 or a > 0.25 * mk.size or (mk & hand).sum() > 0.6 * a:
                continue
            ys, xs = np.nonzero(mk)
            if not table[int(ys.mean()), int(xs.mean())]:
                continue
            cand.append(i)
        cand.sort(key=lambda i: -d["prob"][i])
        out = []
        for i in cand:
            if all(iou(d["masks"][i], d["masks"][j]) < NMS_IOU for j in out):
                out.append(i)
        comp = set()
        for i in out:                              # composite (group) masks
            mi = d["masks"][i]; a = mi.sum()
            parts = [j for j in out if j != i and (d["masks"][j] & mi).sum() > 0.6 * d["masks"][j].sum() and d["masks"][j].sum() < 0.8 * a]
            if len(parts) >= 2:
                u = np.zeros_like(mi)
                for j in parts:
                    u |= d["masks"][j]
                if (u & mi).sum() >= 0.7 * a:
                    comp.add(i)
        kept[k] = [i for i in out if i not in comp]
        kept[k] = kept[k]
        d.setdefault("hand", {})[k] = hand
    return kept


# ------------------------------------------------------------------ 2. track
def track(d, kept, det_box=None):
    frames = d["frames"]
    cent = {}
    for k, ids in kept.items():
        for i in ids:
            ys, xs = np.nonzero(d["masks"][i])
            cent[i] = (xs.mean(), ys.mean(), np.sqrt(len(xs)))
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 5
    gap = max(2, int(round(GAP_S * 10 / step)))
    tracks, active = [], []
    for fi, k in enumerate(frames):
        dets = kept[k]
        cand = [t for t in active if fi - t["last_fi"] <= gap + 1]
        used = set()
        if cand and dets:
            C = np.full((len(cand), len(dets)), 9.0)
            for a, t in enumerate(cand):
                j = t["obs"][-1][1]
                for b, i in enumerate(dets):
                    o = iou(d["masks"][j], d["masks"][i])
                    cx, cy, sz = cent[i]; tx, ty, tsz = cent[j]
                    if abs(np.log(max(sz, 1) / max(tsz, 1))) > np.log(2.2):
                        continue
                    size = max(6.0, (sz + tsz) / 2)
                    dist = np.hypot(cx - tx, cy - ty)
                    c = 1 - o if o > 0.05 else (0.95 + dist / size / 3 if dist < 3 * size + 30 else 9.0)
                    if c < 9 and det_box is not None and det_box.get(i) and det_box.get(j):
                        bi, bj = det_box[i], det_box[j]
                        ar = (bi["L"] * bi["W"]) / max(1e-6, bj["L"] * bj["W"])
                        held = max(bi["hand_frac"], bj["hand_frac"]) >= 0.1
                        if ar > 3 or ar < 1 / 3 or (not held and abs(bi["top"] - bj["top"]) > 0.12):
                            c = 9.0                                # 3D footprint / height jump without a hand: another object
                        else:                                      # 3D consistency: an object at rest keeps its box
                            w = 0.3 if held else 1.0
                            d3 = min(1.5, np.hypot(bi["x"] - bj["x"], bi["y"] - bj["y"]) / 0.10 + abs(bi["top"] - bj["top"]) / 0.06)
                            ds = abs(bi["L"] - bj["L"]) / max(bi["L"], bj["L"]) + abs(bi["W"] - bj["W"]) / max(bi["W"], bj["W"])
                            c = 0.5 * c + w * (d3 + 0.5 * ds)
                    if c < 9 and d["prompt"][i] != d["prompt"][j]:
                        c = c + 0.25 if o >= RELABEL_IOU else 9.0          # relabel only when the masks clearly coincide
                    C[a, b] = c
            r, c = linear_sum_assignment(C)
            for a, b in zip(r, c):
                if C[a, b] < 9:
                    cand[a]["obs"].append((k, dets[b])); cand[a]["last_fi"] = fi
                    used.add(b)
        for b, i in enumerate(dets):
            if b not in used:
                t = dict(obs=[(k, i)], last_fi=fi)
                active.append(t); tracks.append(t)
        active = [t for t in active if fi - t["last_fi"] <= gap + 1]
    P = d["prompts"]
    for t in tracks:
        votes = {}
        for k, i in t["obs"]:
            votes[P[d["prompt"][i]]] = votes.get(P[d["prompt"][i]], 0) + float(d["prob"][i])
        t["votes"] = votes
        t["prompt"] = max(votes, key=votes.get)
    return [t for t in tracks if len(t["obs"]) >= 2]           # one-off detections are noise


# ------------------------------------------------------------------ 3. CoTracker3 points
_COT = {}


def cotracker(dev):
    import torch
    if dev not in _COT:
        _COT[dev] = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    return _COT[dev]


def cotrack(v, d, tracks, dev):
    import torch
    n = meta(v)["n"]
    m = cotracker(dev)
    rng = np.random.default_rng(0)
    ker = np.ones((3, 3), np.uint8)
    out = []
    t0 = time.time()
    for k0 in range(0, max(1, n - OV), WIN - OV):
        k1 = min(k0 + WIN, n)
        q, own = [], []
        for ti, t in enumerate(tracks):
            obs = [(k, i) for k, i in t["obs"] if k0 <= k < k1]
            if not obs:
                continue
            pick = [obs[0]] if len(obs) == 1 else [obs[0], obs[-1]]
            for k, i in pick:
                mk = cv2.erode(d["masks"][i].astype(np.uint8), ker).astype(bool)
                if mk.sum() < 8:
                    mk = d["masks"][i]
                ys, xs = np.nonzero(mk)
                sel = rng.choice(len(xs), size=min(NQ // len(pick), len(xs)), replace=False)
                for s in sel:
                    q.append((k - k0, xs[s] + 0.5, ys[s] + 0.5)); own.append(ti)     # half-res px
        if not q:
            continue
        fr = np.stack([cv2.resize(frame(v, k), (W, H), interpolation=cv2.INTER_AREA)[..., ::-1] for k in range(k0, k1)])
        vid = torch.from_numpy(fr).permute(0, 3, 1, 2)[None].float().to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            tr, vis = m(vid, queries=torch.tensor(np.array(q, np.float32))[None].to(dev))
        out.append(dict(k0=k0, k1=k1, tr=(tr[0].float().cpu().numpy() * 2).astype(np.float16), vis=vis[0].float().cpu().numpy() > 0.5,
                        own=np.array(own, np.int32)))
        del vid
        if k1 >= n:
            break
    print(v, f"CoTracker3 {len(out)} windows on {dev} in {time.time() - t0:.0f}s", flush=True)
    return out


def stitch(Wn, J, n):
    """Per-track image position (n, J, 2) work px from overlapping windows (window overlaps split in the middle;
    position = chained median displacement of the visible points)."""
    uv = np.full((n, J, 2), np.nan, np.float32)
    nw = len(Wn)
    for wi, w in enumerate(Wn):
        k0, k1 = w["k0"], w["k1"]
        tr, vis, own = w["tr"].astype(np.float32), w["vis"], w["own"]
        a = 0 if wi == 0 else OV // 2
        b = (k1 - k0) if wi == nw - 1 else (k1 - k0) - OV // 2
        for j in np.unique(own):
            Pj = tr[:, own == j]; V = vis[:, own == j]
            cur, blind = None, 0
            for t in range(k1 - k0):
                if cur is None:
                    if V[t].sum() >= 3:
                        cur = np.median(Pj[t, V[t]], 0)
                else:
                    both = V[t] & V[t - 1]
                    if both.sum() >= 3:
                        cur = cur + np.median(Pj[t, both] - Pj[t - 1, both], 0); blind = 0
                    else:
                        cur = cur + np.median(Pj[t] - Pj[t - 1], 0); blind += 1
                if cur is not None and a <= t < b and blind <= 25:
                    uv[k0 + t, j] = cur
    return uv


# ------------------------------------------------------------------ 5. 3D boxes per sighting
def obb(Dk, K, T, mask):
    """3D oriented box from one mask + depth (table frame): dict(x, y, top, L, W, yaw, n) or None."""
    m = mask.astype(np.uint8)
    m2 = cv2.erode(m, np.ones((3, 3), np.uint8))
    if m2.sum() >= 15:
        m = m2
    P, _ = backproject(Dk, K, m.astype(bool))
    if len(P) < 15:
        return None
    Q = to_table(T, P)
    top = float(np.percentile(Q[:, 2], 90))
    band = Q[np.abs(Q[:, 2] - top) < max(0.015, 0.1 * top)]
    if len(band) >= 10:
        top = float(np.median(band[:, 2]))
    up = Q[Q[:, 2] > min(0.004, 0.3 * top)]
    if len(up) < 10:
        up = Q
    x, y, L, Wd, yaw = rect_extent(up[:, :2])
    px = 2.0 * float(np.median(P[:, 2])) / K[0, 0]                      # add back the erosion (1 half-res px per side)
    return dict(x=x, y=y, top=max(top, 0.003), L=L + 2 * px, W=Wd + 2 * px, yaw=yaw, n=int(len(Q)))


def run(v, dev=None):
    out_p = vdir(v) / "tracks.pkl"
    if out_p.exists():
        return pickle.load(open(out_p, "rb"))
    n = meta(v)["n"]
    G = geom(v)
    K, T_all, scale = G["K"], G["T_all"], G["scale"]
    D10 = depth(v)
    table = cv2.dilate(G["table"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    d = load_inventory(v)
    kept = clean(d, table)
    # 3D 'on the table' check (generic): the detection's depth points must stand on / above the table plane (median height > -3 cm)
    # and its footprint centre must lie within the table's extent (+5 cm). Removes the floor mat, the bin on the counter, etc.
    from cleantables.perception.boxes.scene_export import table_extent
    ext = np.array(table_extent(G["T"], K, G["table"]))
    n_off = 0
    for k in list(kept):
        Dk = np.asarray(D10[k], np.float32) * scale[k]
        keep = []
        for i in kept[k]:
            P, _ = backproject(Dk, K, d["masks"][i])
            if len(P) < 10:
                continue
            Q = to_table(T_all[k], P)
            cx, cy = np.median(Q[:, 0]), np.median(Q[:, 1])
            if np.median(Q[:, 2]) < -0.03 or not (ext[0, 0] - 0.05 <= cx <= ext[0, 1] + 0.05 and ext[1, 0] - 0.05 <= cy <= ext[1, 1] + 0.05):
                n_off += 1
                continue
            keep.append(i)
        kept[k] = keep
    # 3D box of every kept detection (single frame + depth: the same input the world model gets at inference)
    det_box = {}
    for k, ids in kept.items():
        Dk = np.asarray(D10[k], np.float32) * scale[k]
        hand = cv2.dilate(d["hand"][k].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        lab = cv2.cvtColor(cv2.resize(frame(v, k), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
        for i in ids:
            b = obb(Dk, K, T_all[k], d["masks"][i])
            if b is not None:
                b["hand_frac"] = float((hand & d["masks"][i]).sum() / max(1, d["masks"][i].sum()))
                b["prob"] = float(d["prob"][i])
                b["phrase"] = d["prompts"][d["prompt"][i]]
                ys, xs = np.nonzero(d["masks"][i])
                b["uv"] = [float(xs.mean() * 2 + 1), float(ys.mean() * 2 + 1)]
                mk = cv2.erode(d["masks"][i].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~hand
                b["lab"] = (np.median(lab[mk], 0) if mk.sum() > 5 else np.median(lab[d["masks"][i]], 0)).tolist()
                b["bbox"] = [int(xs.min() * 2), int(ys.min() * 2), int(xs.max() * 2 + 2), int(ys.max() * 2 + 2)]
            det_box[i] = b
    tracks = track(d, kept, det_box)
    J = len(tracks)
    for t in tracks:
        t["box"] = [det_box.get(i) for k, i in t["obs"]]
    if dev is None:
        dev = "cuda" if gpu_free() >= 2600 else "cpu"
    if dev == "cpu":
        import torch
        torch.set_num_threads(8)
    Wn = cotrack(v, d, tracks, dev)
    np.savez_compressed(vdir(v) / "cotrack.npz", k0=np.array([w["k0"] for w in Wn]), k1=np.array([w["k1"] for w in Wn]),
                        tr=np.array([w["tr"] for w in Wn], dtype=object), vis=np.array([w["vis"] for w in Wn], dtype=object),
                        own=np.array([w["own"] for w in Wn], dtype=object))
    uv = stitch(Wn, J, n)
    # ---- 4. identity linking (union-find), no label rules: CoTracker points of one track sit on another track, never co-detected
    kf = {j: {k for k, _ in tracks[j]["obs"]} for j in range(J)}
    par = list(range(J))

    def root(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    okv = ~np.isnan(uv[..., 0])
    for a in range(J):
        for b in range(a + 1, J):
            both = okv[:, a] & okv[:, b]
            if both.sum() < 5:
                continue
            if np.median(np.linalg.norm(uv[both, a] - uv[both, b], axis=1)) >= 14:
                continue
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            ma = set().union(*[kf[j] for j in range(J) if root(j) == ra]); mb = set().union(*[kf[j] for j in range(J) if root(j) == rb])
            if len(ma & mb) > 1:                       # seen as two separate detections at the same time: two objects
                continue
            par[rb] = ra
    # ---- appearance re-identification (generic): a track that starts within REID_GAP after another ends, never seen at the same time,
    # with the same colour (median Lab of its hand-free mask pixels, dE < REID_DE) and footprint (sorted L, W within x1.5) is the same
    # object put down somewhere else (the hand hid it in between; CoTracker lost it).
    def gstats(r):
        mem = [j for j in range(J) if root(j) == r]
        bs = [b for j in mem for b in tracks[j]["box"] if b and b["hand_frac"] < 0.2] or [b for j in mem for b in tracks[j]["box"] if b]
        ks = sorted(set().union(*[kf[j] for j in mem]))
        if not bs:
            return None
        return dict(first=ks[0], last=ks[-1], ks=set(ks), lab=np.median([b["lab"] for b in bs], 0),
                    L=np.median([max(b["L"], b["W"]) for b in bs]), W=np.median([min(b["L"], b["W"]) for b in bs]))
    n_reid = 0
    changed = True
    while changed:
        changed = False
        roots = sorted({root(j) for j in range(J)})
        st = {r: gstats(r) for r in roots}
        st = {r: x for r, x in st.items() if x}
        best = None
        for b_ in st:
            for a_ in st:
                A, B = st[a_], st[b_]
                if a_ == b_ or not (0 < B["first"] - A["last"] <= REID_GAP) or len(A["ks"] & B["ks"]) > 0:
                    continue
                dE = float(np.linalg.norm((A["lab"] - B["lab"]) * np.array([0.5, 1, 1])))
                rl, rw = max(A["L"], B["L"]) / min(A["L"], B["L"]), max(A["W"], B["W"]) / min(A["W"], B["W"])
                if dE < REID_DE and rl < 1.5 and rw < 2.2:      # W is often under-measured when the object is half hidden
                    cost = dE / REID_DE + (rl - 1) + (rw - 1) + (B["first"] - A["last"]) / REID_GAP
                    if best is None or cost < best[0]:
                        best = (cost, a_, b_)
        if best:
            par[root(best[2])] = root(best[1]); n_reid += 1; changed = True
    groups = {}
    for j in range(J):
        groups.setdefault(root(j), []).append(j)
    Gs = sorted(groups.values(), key=lambda g: min(min(kf[j]) for j in g))
    objs = []
    for g in Gs:
        obs = sorted([(k, j, bi) for j in g for bi, (k, _) in enumerate(tracks[j]["obs"])])
        votes = {}
        for j in g:
            for p_, w_ in tracks[j]["votes"].items():
                votes[p_] = votes.get(p_, 0) + w_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = np.nanmedian(uv[:, g], 1)
        objs.append(dict(members=g, phrase=max(votes, key=votes.get), votes=votes,
                         sight_k=np.array([k for k, _, _ in obs]), boxes=[tracks[j]["box"][bi] for _, j, bi in obs],
                         det=[tracks[j]["obs"][bi][1] for _, j, bi in obs], uv=u))
    # ---- 10 fps 3D trajectory of each object's tracked point
    for o in objs:
        P = np.full((n, 3), np.nan, np.float32)
        ks = o["sight_k"]
        f = np.nonzero(~np.isnan(o["uv"][:, 0]))[0]
        f = f[(f >= ks.min() - 30) & (f <= ks.max() + 30)] if len(ks) else f[:0]
        for ff in f:
            uu, vv = o["uv"][ff]
            if not (0 <= uu < 960 and 0 <= vv < 540):
                continue
            z = float(np.median(np.asarray(D10[ff, max(0, int(vv / 2) - 2):int(vv / 2) + 3, max(0, int(uu / 2) - 2):int(uu / 2) + 3], np.float32))) * scale[ff]
            if not z > 0:
                continue
            pc = np.array([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z])
            P[ff] = to_table(T_all[ff], pc[None])[0]
        o["P10"] = P
        # colour: median RGB of the object's masks at its 3 most confident sightings
        cols = []
        for bi in np.argsort([-(b["prob"] if b else 0) for b in o["boxes"]])[:3]:
            k, det = int(o["sight_k"][bi]), o["det"][bi]
            im = cv2.resize(frame(v, k), (W, H), interpolation=cv2.INTER_AREA)
            mk = cv2.erode(d["masks"][det].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if mk.sum() > 5:
                cols.append(np.median(im[mk], 0)[::-1])
        o["rgb"] = [int(c) for c in np.median(cols, 0)] if cols else [128, 128, 128]
    res = dict(video=v, n=n, n_off_table=n_off, objects=objs, n_sam_tracks=J, kept={int(k): [int(i) for i in ids] for k, ids in kept.items()}, det_box=det_box, phrases=d["prompts"],
               n_dets=int(len(d["prob"])), n_kept=int(sum(len(x) for x in kept.values())), keyframes=d["frames"])
    pickle.dump(res, open(out_p, "wb"))
    log(f"{v} tracking: {res['n_kept']} kept detections -> {J} SAM tracks -> {len(objs)} objects after point linking + {n_reid} re-identifications "
        f"({', '.join(o['phrase'] for o in objs)})")
    return res


if __name__ == "__main__":
    dev = "cpu" if "--cpu" in sys.argv else None
    for v in names([a for a in sys.argv[1:] if not a.startswith("--")]):
        run(v, dev)
