"""Box identities, moves, stacking placements, carries and episodes (one per run) from rest segments and CoTracker follows.

Identity: a 'moved' follow of segment A links it to the first later segment (within FOLLOW_S) that holds the followed points (median
point within 6 cm of its top-face centre, or >= 40 % of the points inside its SAM mask); a 'static' follow links to the next segment at
the same place (6 cm, 3 cm in height); unlinked ends and starts with matching footprints (|dL|, |dW| < 4 cm) are paired by Hungarian
matching. Box L, W = sighting-weighted median over its segments; H = its lowest top height (every box rests on the table at the start).
Moves: consecutive segments of one box more than 6 cm apart; the path is the followed top-face point, snapped to both rest poses.
Support of a placement: the resting box with the largest footprint overlap (>= 25 %) whose top is within 4 cm of the placed box's
bottom, else the table. offset = placed centre in the support frame; rel_yaw is folded to (-90, 90] deg.
Carry: >= 2 stacked boxes whose moves start within 3 s with the same displacement (> 15 cm, within 8 cm).
Episode: one per carry (its stack's last placements before the carry, in time order); without a carry, the tallest stack formed.
  -> <OUT>/<v>/events.json and (all videos) <OUT>/episodes.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from cleantables.perception.scene.common import meta

from cleantables.perception.boxes import geometry as G
from cleantables.perception.boxes.tracking import FOLLOW_S, load_follow, load_sam
from cleantables.perception.boxes.common import OUT, jdump, log, vdir

H_PATH = 16
D_MOVE = 0.06


def seg_rect(s, H=None):
    return G.rect_corners(s["x"], s["y"], s["L"], s["W"], s["yaw"])


def link_segments(video, segs, fol, fps):
    n = len(segs)
    nxt = {}
    byk0 = sorted(range(n), key=lambda i: segs[i]["k0"])
    d = None
    for i, s in enumerate(segs):
        f = fol.get(s["id"])
        later = [j for j in byk0 if segs[j]["k0"] > s["k1"] - 2 and segs[j]["k0"] <= s["k1"] + FOLLOW_S * fps and j != i]
        if f is not None and f["moved"]:
            best = None
            for j in later:
                b = segs[j]
                t = b["k0"] - f["k"]
                if t < 0 or t >= len(f["P3"]):
                    continue
                p = f["P3"][t]
                if not np.isfinite(p).all():
                    continue
                dist = np.hypot(p[0] - b["x"], p[1] - b["y"])
                if abs(p[2] - b["top"]) > 0.06:             # same xy, different stack level
                    continue
                # fraction of followed points inside B's SAM mask at its first sighting
                inside = 0.0
                if f.get("uv") is not None:
                    if d is None:
                        d = load_sam(video)
                    det = b["dets"][0][1]
                    uv = f["uv"][t].astype(np.float32)
                    mk = d["masks"][det]
                    inside = np.mean([mk[int(v / 2), int(u / 2)] if 0 <= u < 960 and 0 <= v < 540 else False for u, v in uv])
                moved_far = np.hypot(b["x"] - s["x"], b["y"] - s["y"]) > D_MOVE
                if (dist < 0.06 or inside >= 0.4) and moved_far:
                    sc = (b["k0"], dist - 0.1 * inside)          # earliest rest that holds the followed points
                    if best is None or sc < best[0]:
                        best = (sc, j, "cotracker")
            if best:
                nxt[i] = (best[1], best[2])
                continue
        if f is None or not f["moved"]:
            for j in later:
                b = segs[j]
                if np.hypot(b["x"] - s["x"], b["y"] - s["y"]) < 0.06 and abs(b["top"] - s["top"]) < 0.03:
                    nxt[i] = (j, "same place")
                    break
    # one predecessor per segment
    pred = {}
    for i, (j, how) in sorted(nxt.items(), key=lambda kv: segs[kv[0]]["k1"]):
        if j not in pred:
            pred[j] = (i, how)
    nxt = {i: (j, how) for j, (i, how) in pred.items()}
    # size-matching fallback for moved ends that found no destination
    ends = [i for i in range(n) if i not in nxt and fol.get(segs[i]["id"], {}).get("moved")]
    starts = [j for j in range(n) if j not in pred]
    if ends and starts:
        C = np.full((len(ends), len(starts)), 9.0)
        for a, i in enumerate(ends):
            for b, j in enumerate(starts):
                s, t = segs[i], segs[j]
                if not (s["k1"] - 2 < t["k0"] <= s["k1"] + FOLLOW_S * fps):
                    continue
                dL, dW = abs(s["L"] - t["L"]), abs(s["W"] - t["W"])
                if dL < 0.04 and dW < 0.04 and np.hypot(t["x"] - s["x"], t["y"] - s["y"]) > D_MOVE:
                    C[a, b] = (dL + dW) / 0.04 + (t["k0"] - s["k1"]) / (FOLLOW_S * fps)
        ra, cb = linear_sum_assignment(C)
        for a, b in zip(ra, cb):
            if C[a, b] < 9:
                nxt[ends[a]] = (starts[b], "size match")
                pred[starts[b]] = (ends[a], "size match")
    chains, seen = [], set()
    for i in byk0:
        if i in seen or i in pred:
            continue
        ch, cur = [], i
        while cur is not None and cur not in seen:
            ch.append(cur); seen.add(cur)
            cur = nxt.get(cur, (None,))[0]
        chains.append(ch)
    for i in byk0:                      # cycles / leftovers
        if i not in seen:
            chains.append([i]); seen.add(i)
    how = {j: h for j, (i, h) in pred.items()}
    return chains, how


def box_size(segs, ch):
    ch = [i for i in ch if not segs[i].get("virtual")] or ch
    w = np.array([segs[i]["n"] for i in ch], float) + 1e-6
    L = float(np.average([segs[i]["L"] for i in ch], weights=w))
    W = float(np.average([segs[i]["W"] for i in ch], weights=w))
    tops = [segs[i]["top"] for i in ch if segs[i]["n"] >= 3]
    H = float(min(tops)) if tops else float(min(segs[i]["top"] for i in ch))
    return L, W, H


def build_video(video, unseen=False, tape_square=True):
    segs = json.load(open(vdir(video) / "segments.json"))["segments"]
    fol = load_follow(video)
    mt = meta(video)
    fps = mt["fps"]
    t_off = mt.get("t0", 0.0)
    det = np.load(vdir(video) / "det3d.npz")
    tapes = det["tape"]
    chains, how = link_segments(video, segs, fol, fps)
    boxes = {}
    for c, ch in enumerate(chains):
        L, W, H = box_size(segs, ch)
        if H < 0.03 or H > 0.7 or len(ch) == 0:           # > 70 cm: never seen on the table, not a box track
            continue
        boxes[f"b{c}"] = dict(chain=ch, L=L, W=W, H=H, n_obs=int(sum(segs[i]["n"] for i in ch)))
    # Hovering: a short rest in the air (box held above the stack before being lowered) is not a placement. Drop it when the next
    # rest of the same box follows within 4 s at the same xy but lower.
    for b, B in boxes.items():
        ch = B["chain"]
        keep_ = [ch[0]]
        for i in range(1, len(ch)):
            a = segs[ch[i]]
            nx = segs[ch[i + 1]] if i + 1 < len(ch) else None
            if (nx is not None and a["n"] < 6 and nx["k0"] - a["k1"] < 4 * fps and np.hypot(nx["x"] - a["x"], nx["y"] - a["y"]) < D_MOVE
                    and a["top"] - nx["top"] > 0.03):
                continue
            keep_.append(ch[i])
        B["chain"] = keep_
    # Short rests inside a followed move (e.g. the last box sits on the stack for ~1-2 s before the stack is carried, too short for a
    # rest segment): split the move at every pause of the followed point (< 6 cm/s over 0.4 s for >= 0.8 s, > 6 cm from both ends).
    rows = det["rows"]
    for b, B in boxes.items():
        ch = B["chain"]
        new_ch = [ch[0]]
        for a, c in zip(ch[:-1], ch[1:]):
            s, t = segs[a], segs[c]
            f = fol.get(s["id"])
            if f is not None and f["moved"]:
                P3 = f["P3"]
                i0, i1 = max(0, s["k1"] - f["k"]), min(len(P3), t["k0"] - f["k"])
                Q = P3[i0:i1]
                if len(Q) > 12:
                    ok = np.isfinite(Q).all(1)
                    from scipy.ndimage import median_filter
                    Qs = median_filter(np.where(ok[:, None], Q, 99.0), size=(5, 1), mode="nearest")
                    sp = np.full(len(Q), 9.0)
                    sp[2:-2] = np.linalg.norm(Qs[4:, :2] - Qs[:-4, :2], axis=1) / (4 / fps)     # speed over 0.4 s to suppress jitter
                    still = (sp < 0.06) & ok
                    i = 0
                    while i < len(Q):
                        if still[i]:
                            j = i
                            while j < len(Q) and still[j]:
                                j += 1
                            if j - i >= int(0.8 * fps):
                                r = np.median(Q[i:j], 0)
                                if np.hypot(r[0] - s["x"], r[1] - s["y"]) > D_MOVE and np.hypot(r[0] - t["x"], r[1] - t["y"]) > D_MOVE:
                                    ka, kb = f["k"] + i0 + i, f["k"] + i0 + j - 1
                                    near = rows[(rows[:, 0] >= ka) & (rows[:, 0] <= kb) & (np.hypot(rows[:, 3] - r[0], rows[:, 4] - r[1]) < 0.05)
                                                & (np.abs(rows[:, 5] - r[2]) < 0.06)]
                                    from cleantables.perception.boxes.tracking import circ_mean_pi
                                    yaw = circ_mean_pi(near[:, 8]) if len(near) else s["yaw"]
                                    top = float(np.median(near[:, 5])) if len(near) else float(r[2])
                                    v = dict(id=len(segs), k0=int(ka), k1=int(kb), n=0, n_untouched=0, x=float(r[0]), y=float(r[1]), top=top,
                                             L=B["L"], W=B["W"], yaw=yaw, dets=[], seed=None, virtual=True)
                                    segs.append(v); fol[v["id"]] = f
                                    new_ch.append(v["id"]); how[v["id"]] = "cotracker pause"
                            i = j
                        else:
                            i += 1
            new_ch.append(c)
        B["chain"] = new_ch
    # moves
    moves = []
    for b, B in boxes.items():
        ch = B["chain"]
        for a, c in zip(ch[:-1], ch[1:]):
            s, t = segs[a], segs[c]
            if np.hypot(t["x"] - s["x"], t["y"] - s["y"]) < D_MOVE and abs(t["top"] - s["top"]) < 0.03:
                continue
            f = fol.get(s["id"])
            k_pick, k_place, path = s["k1"], t["k0"], None
            if f is not None and f["moved"]:
                P3 = f["P3"]; i0 = max(0, s["k1"] - f["k"])
                ref = P3[i0:i0 + 5]; ref = ref[np.isfinite(ref).all(1)]
                ref = np.median(ref, 0)[:2] if len(ref) else np.array([s["x"], s["y"]])
                with np.errstate(all="ignore"):
                    disp = np.linalg.norm(P3[:, :2] - ref, axis=1)
                on = np.nonzero((disp > 0.02) & (np.arange(len(P3)) >= i0))[0]
                if len(on):
                    k_pick = min(f["k"] + int(on[0]), t["k0"])
                seg = P3[max(0, k_pick - f["k"]):max(0, t["k0"] - f["k"]) + 1]
                ok = np.isfinite(seg).all(1)
                if ok.sum() >= 4:
                    idx = np.nonzero(ok)[0]
                    tt = np.linspace(idx[0], idx[-1], H_PATH)
                    path = np.stack([np.interp(tt, idx, seg[ok][:, i]) for i in range(3)], 1)
            p0 = np.array([s["x"], s["y"], s["top"]]); p1 = np.array([t["x"], t["y"], t["top"]])
            if path is None:
                path = p0 + np.linspace(0, 1, H_PATH)[:, None] * (p1 - p0)
                psrc = "line (no CoTracker path)"
            else:
                lam = np.linspace(0, 1, H_PATH)[:, None]
                path = path + (1 - lam) * (p0 - path[0]) + lam * (p1 - path[-1])
                psrc = "cotracker"
            moves.append(dict(box=b, seg_from=a, seg_to=c, k_pick=int(k_pick), k_place=int(t["k0"]),
                              t_pick=round(k_pick / fps + t_off, 2), t_place=round(t["k0"] / fps + t_off, 2),
                              pose=dict(x=t["x"], y=t["y"], yaw=t["yaw"], z0=t["top"] - B["H"]), top=t["top"], path=path.tolist(),
                              path_src=psrc, link=how.get(c, "?")))
    moves.sort(key=lambda m: m["k_place"])

    def state_at(k, exclude=None, since=None):
        """Pose of every box at keyframe k from its latest segment that started <= k. `since` drops boxes whose latest segment
        started before that keyframe (stale identities from an earlier run)."""
        out = {}
        for b, B in boxes.items():
            if b == exclude:
                continue
            cur = None
            for i in B["chain"]:
                if segs[i]["k0"] <= k:
                    cur = segs[i]
            if cur is None or (since is not None and cur["k0"] < since - 2 * fps):
                continue
            out[b] = dict(x=cur["x"], y=cur["y"], yaw=cur["yaw"], top=cur["top"], z0=cur["top"] - B["H"], seg=cur["id"])
        return out

    # supports
    for m in moves:
        B = boxes[m["box"]]
        st = state_at(m["k_place"], exclude=m["box"])
        rect = G.rect_corners(m["pose"]["x"], m["pose"]["y"], B["L"], B["W"], m["pose"]["yaw"])
        area = G.poly_area(rect)
        best = None
        for b, p in st.items():
            o = G.poly_area(G.poly_clip(rect, G.rect_corners(p["x"], p["y"], boxes[b]["L"], boxes[b]["W"], p["yaw"])))
            if o >= 0.25 * area and abs(p["top"] - m["pose"]["z0"]) < 0.04 and (best is None or o > best[2]):
                best = (b, p, o)                              # the candidate under most of the placed footprint
        if best is None:
            m["support"] = "table"
            m["h_check"] = dict(top_cm=round(m["top"] * 100, 1), H_cm=round(B["H"] * 100, 1))
        else:
            b, p, _ = best
            m["support"] = b
            m["support_seg"] = int(p["seg"])
            c, s_ = np.cos(p["yaw"]), np.sin(p["yaw"])
            dx, dy = m["pose"]["x"] - p["x"], m["pose"]["y"] - p["y"]
            m["offset"] = [float(c * dx + s_ * dy), float(-s_ * dx + c * dy)]
            m["rel_yaw"] = G.wrap_pi2(m["pose"]["yaw"] - p["yaw"])
            m["pose"]["z0"] = p["top"]
            m["h_check"] = dict(measured_H_cm=round((m["top"] - p["top"]) * 100, 1), H_cm=round(B["H"] * 100, 1))
    # where each move came from (table or another box), lift height, lids
    for m in moves:
        B = boxes[m["box"]]
        s0 = segs[m["seg_from"]]
        st = state_at(max(0, m["k_pick"] - 1), exclude=m["box"])
        rect = G.rect_corners(s0["x"], s0["y"], B["L"], B["W"], s0["yaw"])
        area = G.poly_area(rect)
        src = "table"
        for b, p in st.items():
            o = G.poly_area(G.poly_clip(rect, G.rect_corners(p["x"], p["y"], boxes[b]["L"], boxes[b]["W"], p["yaw"])))
            if o >= 0.25 * area and abs(p["top"] - (s0["top"] - B["H"])) < 0.04:
                src = b
        m["from_support"] = src
        P = np.array(m["path"])
        m["lift_m"] = round(float(P[:, 2].max() - max(P[0, 2], P[-1, 2])), 3)
    # lid: a thin 'box' (H < 6 cm) that at some point rests on a box of the same footprint (the kit box)
    lids = {}
    for b, B in boxes.items():
        if B["H"] >= 0.06:
            continue
        on = [m["support"] for m in moves if m["box"] == b and m["support"] in boxes] + [m["from_support"] for m in moves if m["box"] == b and m["from_support"] in boxes]
        for q in set(on):
            if abs(boxes[q]["L"] - B["L"]) < 0.05 and abs(boxes[q]["W"] - B["W"]) < 0.05:
                lids[b] = q
    for m in moves:
        lift = m["lift_m"]
        if m["box"] in lids:
            kit = lids[m["box"]]
            m["kind"] = "lid_off" if m["from_support"] == kit else ("lid_on" if m["support"] == kit else "lid_move")
        elif m["from_support"] in boxes and m["support"] == "table":
            m["kind"] = "unstack"
        elif m["support"] in boxes:
            m["kind"] = "stack"
        elif lift < 0.03 and m["from_support"] == "table":
            m["kind"] = "push"
        else:
            m["kind"] = "place_on_table"
    # carries: stacked boxes whose moves start together with the same displacement
    carries, used = [], set()
    last_end = 0
    for i, m in enumerate(moves):
        if i in used:
            continue
        d0 = np.subtract([m["pose"]["x"], m["pose"]["y"]], [segs[m["seg_from"]]["x"], segs[m["seg_from"]]["y"]])
        if np.linalg.norm(d0) < 0.15:
            continue
        grp = [i]
        for j, n_ in enumerate(moves):
            if j == i or j in used:
                continue
            d1 = np.subtract([n_["pose"]["x"], n_["pose"]["y"]], [segs[n_["seg_from"]]["x"], segs[n_["seg_from"]]["y"]])
            if (abs(n_["k_pick"] - m["k_pick"]) <= 3 * fps or abs(n_["k_place"] - m["k_place"]) <= 2 * fps) and np.linalg.norm(d1 - d0) < 0.08:
                grp.append(j)
        # stack members under the carried boxes with no move of their own (hidden at the destination)
        k = m["k_pick"] - 1
        st = state_at(k, since=last_end)
        members = {moves[g]["box"] for g in grp}
        changed = True
        while changed:
            changed = False
            for b, p in st.items():
                if b in members:
                    continue
                for q in list(members):
                    if q not in st:
                        continue
                    pq = st[q]
                    ov = G.poly_area(G.poly_clip(G.rect_corners(p["x"], p["y"], boxes[b]["L"], boxes[b]["W"], p["yaw"]),
                                                 G.rect_corners(pq["x"], pq["y"], boxes[q]["L"], boxes[q]["W"], pq["yaw"])))
                    if ov > 0.02 * 0.02 and abs(p["top"] - pq["top"]) > 0.03:
                        members.add(b); changed = True
                        break
        def member_moved(b):
            """Whether a stack member without its own move went along (rest ends at the pick, CoTracker follow moves the same way)."""
            if b not in st:
                return False
            sg = segs[st[b]["seg"]]
            f = fol.get(sg["id"])
            if f is None or not f["moved"] or abs(sg["k1"] - m["k_pick"]) > 3 * fps:
                return False
            P = f["P3"]; ok = np.isfinite(P).all(1)
            if ok.sum() < 3:
                return False
            dd = np.linalg.norm(P[:, :2] - P[np.nonzero(ok)[0][0], :2], axis=1)
            on = np.nonzero(ok & (dd > 0.03))[0]
            if not len(on) or abs(f["k"] + on[0] - m["k_pick"]) > 1.5 * fps:     # must start moving with the carried box
                return False
            Pf = P[ok]
            v = (Pf[-1] - Pf[0])[:2]
            return np.linalg.norm(v) > 0.5 * np.linalg.norm(d0) and v @ d0 / (np.linalg.norm(v) * np.linalg.norm(d0) + 1e-9) > 0.8
        others = [b for b in members if b not in {moves[g]["box"] for g in grp}]
        really = len(grp) >= 2 or any(member_moved(b) for b in others)      # lifting only the top box off is not a carry
        if len(members) >= 2 and really:
            used.update(grp)
            carries.append(dict(moves=grp, members=sorted(members, key=lambda b: st.get(b, {}).get("z0", 0)), k_pick=m["k_pick"],
                                k_place=max(moves[g]["k_place"] for g in grp), dxy=d0.tolist(), since=last_end))
            last_end = carries[-1]["k_place"]
    merged = []                                        # one carry per time window: merge groups of the same carry
    for c in sorted(carries, key=lambda c: c["k_pick"]):
        if merged and c["k_pick"] <= merged[-1]["k_place"]:
            m_ = merged[-1]
            m_["moves"] = sorted(set(m_["moves"]) | set(c["moves"]))
            m_["members"] = m_["members"] + [b for b in c["members"] if b not in m_["members"]]
            m_["k_place"] = max(m_["k_place"], c["k_place"])
        else:
            merged.append(c)
    carries = merged
    for c in carries:
        st_ = state_at(c["k_pick"] - 1, since=c["since"])
        c["members"] = sorted(c["members"], key=lambda b: st_.get(b, {}).get("z0", 0))
        for g in c["moves"]:
            moves[g]["carry"] = True
            moves[g]["kind"] = "carry"
    # episodes
    episodes = []
    stack_runs = []
    for c in carries:
        st = state_at(c["k_pick"] - 1, since=c["since"])
        stack_runs.append((c, [b for b in c["members"] if b in st]))
    if not carries:                                       # no carry: the tallest stack at the end of the video
        k_end = max(s["k1"] for s in segs) if segs else 0
        st = state_at(k_end)
        tops = sorted(st.items(), key=lambda kv: -kv[1]["top"])
        if tops:
            col = [tops[0][0]]
            for b, p in tops[1:]:
                q = st[col[-1]]
                ov = G.poly_area(G.poly_clip(G.rect_corners(p["x"], p["y"], boxes[b]["L"], boxes[b]["W"], p["yaw"]),
                                             G.rect_corners(q["x"], q["y"], boxes[col[-1]]["L"], boxes[col[-1]]["W"], q["yaw"])))
                if ov > 0 and p["top"] < q["top"] - 0.03:
                    col.append(b)
            if len(col) >= 2:
                stack_runs.append((None, col[::-1]))
    for e_i, (c, members) in enumerate(stack_runs):
        k_lim = c["k_pick"] if c else 10 ** 9
        steps_ = []
        members = [b for b in members if b not in lids]           # the lid travels with its kit box, not a stacking step
        for b in members:
            mv = [m for m in moves if m["box"] == b and m["k_place"] <= k_lim and not m.get("carry")
                  and m["k_place"] > (c["since"] if c else -1)]                 # only this run (after the previous carry)
            if mv:
                steps_.append(mv[-1])
        steps_.sort(key=lambda m: m["k_place"])
        if not steps_:
            continue
        k0 = max(0, steps_[0]["k_pick"] - int(2 * fps))
        st0 = state_at(k0, since=c["since"] if c else None)
        bottom = steps_[0]
        sq = None
        if tape_square and len(tapes):
            for tp in tapes:
                if np.hypot(tp[3] - bottom["pose"]["x"], tp[4] - bottom["pose"]["y"]) < 0.5 * max(tp[5], 0.1):
                    sq = dict(x=float(tp[3]), y=float(tp[4]), yaw=float(tp[7]), size=float(tp[5]), src="tape (SAM 3)")
                    break
        if sq is None:
            sq = dict(x=bottom["pose"]["x"], y=bottom["pose"]["y"], yaw=bottom["pose"]["yaw"], size=boxes[bottom["box"]]["L"],
                      src="inferred from the bottom box")
        ep_boxes = {}
        for b, p in st0.items():
            ep_boxes[b] = dict(L=boxes[b]["L"], W=boxes[b]["W"], H=boxes[b]["H"], n_obs=boxes[b]["n_obs"],
                               init=dict(x=p["x"], y=p["y"], yaw=p["yaw"], z0=max(0.0, p["z0"])))
        for m in steps_:
            if m["box"] not in ep_boxes:                  # hidden at the start: start where it was picked
                s0 = segs[m["seg_from"]]
                B = boxes[m["box"]]
                ep_boxes[m["box"]] = dict(L=B["L"], W=B["W"], H=B["H"], n_obs=B["n_obs"], init=dict(x=s0["x"], y=s0["y"], yaw=s0["yaw"], z0=0.0))
        placements = []
        for i, m in enumerate(steps_):
            sup = m["support"]
            if sup in lids:                                    # placed on the kit box's lid -> the support is the kit box
                sup = lids[sup]
            if i == 0 or sup == "table":
                if np.hypot(m["pose"]["x"] - sq["x"], m["pose"]["y"] - sq["y"]) < 0.25:
                    sup = "square"
            if sup == "square":
                c_, s_ = np.cos(sq["yaw"]), np.sin(sq["yaw"])
                dx, dy = m["pose"]["x"] - sq["x"], m["pose"]["y"] - sq["y"]
                off, rel = [float(c_ * dx + s_ * dy), float(-s_ * dx + c_ * dy)], G.wrap_pi2(m["pose"]["yaw"] - sq["yaw"])
            elif sup == "table":
                off, rel = [0.0, 0.0], 0.0
            else:
                off, rel = m["offset"], m["rel_yaw"]
            placements.append(dict(box=m["box"], kind=m["kind"], lift_m=m["lift_m"], support=sup, t_pick=m["t_pick"], t_place=m["t_place"], k_pick=m["k_pick"], k_place=m["k_place"],
                                   pose=m["pose"], offset=off, rel_yaw=rel, path=m["path"], path_src=m["path_src"], link=m["link"],
                                   h_check=m["h_check"], seg_from=m["seg_from"], seg_to=m["seg_to"], support_seg=m.get("support_seg")))
        carry = None
        spot_b = None
        if c:
            top = moves[c["moves"][0]]
            carry = dict(t0=round(c["k_pick"] / fps + t_off, 2), t1=round(c["k_place"] / fps + t_off, 2), dxy=c["dxy"], members=c["members"])
            botm = [moves[g] for g in c["moves"] if moves[g]["box"] == members[0]]
            ref = botm[0] if botm else top
            spot_b = dict(x=ref["pose"]["x"], y=ref["pose"]["y"], yaw=ref["pose"]["yaw"], src="carry destination")
        for b in list(ep_boxes):
            if b in lids:                                      # kit + lid form one box for the world model (H includes the lid)
                kit = lids[b]
                if kit in ep_boxes:
                    ep_boxes[kit]["H_box"] = ep_boxes[kit]["H"]
                    ep_boxes[kit]["H"] = ep_boxes[kit]["H"] + ep_boxes[b]["H"]
                del ep_boxes[b]
        k_from = c["since"] if c else -1
        k_to = c["k_place"] if c else 10 ** 9
        pre = [dict(box=m["box"], kind=m["kind"], t_pick=m["t_pick"], t_place=m["t_place"], from_support=m["from_support"],
                    support=m["support"], lift_m=m["lift_m"]) for m in moves if k_from < m["k_place"] <= k_to and m not in steps_]
        lid_ev = [p_ for p_ in pre if p_["kind"].startswith("lid")]
        episodes.append(dict(id=f"{video}_r{e_i + 1:02d}", video=video, source="real", split="unseen" if unseen else "seen",
                             other_moves=pre, n_unstack=sum(p_["kind"] == "unstack" for p_ in pre),
                             n_push=sum(p_["kind"] == "push" for p_ in pre) + sum(m["kind"] == "push" for m in steps_),
                             kit=dict(lid=[b for b in lids if lids[b] in ep_boxes], lid_events=lid_ev) if lid_ev or lids else None,
                             t_start=round(k0 / fps + t_off, 2), t_end=round((c["k_place"] if c else steps_[-1]["k_place"]) / fps + t_off + 1, 2),
                             k_start=int(k0), boxes=ep_boxes, square=sq, spot_b=spot_b, placements=placements,
                             final_order=[m["box"] for m in steps_], carry=carry,
                             usable=len(placements) >= 2))
    out = dict(video=video, lids=lids, n_segments=len(segs), boxes={b: {k: v for k, v in B.items()} for b, B in boxes.items()},
               moves=moves, carries=carries, episodes=episodes)
    jdump(out, vdir(video) / "events.json")
    log(f"{video} events: {len(boxes)} boxes, {len(moves)} moves ({sum(m['support'] not in ('table',) for m in moves)} onto a box), "
        f"{len(carries)} carries, {len(episodes)} episodes")
    return out


def collect(videos, unseen=()):
    """One episode per run: the snapshot reading (cleantables.perception.boxes.snapshot: inventory + final stack, unaffected by the
    person hiding the build) when usable, else the tracker's. Tracker move statistics and kit packing (tools.json) are attached."""
    eps = []
    for v in videos:
        p = vdir(v) / "events.json"
        tr = json.load(open(p))["episodes"] if p.exists() else []
        sp = vdir(v) / "snapshot.json"
        sn = json.load(open(sp)).get("episode") if sp.exists() else None
        tl = json.load(open(vdir(v) / "tools.json")) if (vdir(v) / "tools.json").exists() else None
        evj = json.load(open(p)) if p.exists() else {}
        stats = dict(moves_by_kind={}, n_carries=len(evj.get("carries", [])), lids=evj.get("lids", {}))
        for mv in evj.get("moves", []):
            stats["moves_by_kind"][mv.get("kind", "?")] = stats["moves_by_kind"].get(mv.get("kind", "?"), 0) + 1
        chosen = [sn] if (sn and sn.get("usable")) else tr
        for e in chosen:
            e["split"] = "unseen" if v in unseen else "seen"
            e.setdefault("reader", "tracker")
            e["tracker_stats"] = stats
            if tr:
                e["tracker_episode"] = dict(final_order=tr[0]["final_order"], n_placements=len(tr[0]["placements"]),
                                            kit=tr[0].get("kit"), carry=tr[0].get("carry"))
            if tl:
                packs = [pk for ep_ in tl.get("episodes", {}).values() for pk in ep_.get("packs", [])]
                e["tools"] = dict(n_tool_tracks=len(tl.get("tracks", [])), packs=packs)
            eps.append(e)
    jdump(dict(episodes=eps, n=len(eps), videos=list(videos), unseen=list(unseen)), OUT / "episodes.json")
    log(f"collected {len(eps)} episodes from {len(videos)} videos -> {OUT / 'episodes.json'}")
    return eps
