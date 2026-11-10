"""Training data for the world model that learns by watching: no move segmentation and no move labels.

One example at every tracked keyframe of every video that has a token sample; the targets are what the data shows next:
  T1  each object's pose at the end of the labelled episode containing k (episodes_all.json segments, core cut, no padding):
      the same object's token in the episode's end snapshot (same source as the input tokens) and whether it is still there.
      Masked outside episodes (reset / setup / idle stretches with an empty instruction).
  T2  each object's tracked displacement at k + 0.5 ... 4 s (8 steps; the tracker's own boxes, so pose-source switches do not
      appear as jumps)
  T3  the global V-JEPA latent at k + 4 s (nearest keyframe; cleantables.worldmodel.data maps it to its feature row)
  T5  each object's HOIST-Former hand contact (track-level events_contact.json, any kind) at the T2 steps; masked outside the
      scored windows. The contact at k is an input (obj_contact), dropped together with the hands.
The move fields are inert placeholders (mover / target -1, empty mover set, no progress / end / path / future-latent labels)
so the loader keeps one format.
  $VLA_PY -m cleantables.worldmodel.watch_data --source <..._source.npz> [--have_only] --out tasks3_world
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from cleantables.perception.common import vdir

from cleantables.worldmodel.common import OUT, ROOT, jdump, log
from cleantables.perception.contact.moves import pose_at
from cleantables.worldmodel.data import HELD_TWINS, HELD_VIDEOS

DST = Path(os.environ.get("WM_SUB_DST", str(OUT / "wm_v2_world" / "data")))
T2_S = tuple(0.5 * (i + 1) for i in range(8))          # T2 / T5 steps (s)
T3_S = 4.0


def contact_of(v):
    """Return ({events.json object id: [(f_start, f_end)] of every HOIST-Former contact, any kind}, scored windows [(k0, k1)])."""
    f = vdir(v) / "events_contact.json"
    if not f.exists():
        return {}, []
    J = json.load(open(f)); Ev = json.load(open(vdir(v) / "events.json"))
    ev_id = {int(o["src"]): int(o["id"]) for o in Ev["objects"]}
    c = defaultdict(list)
    for e in J["events"]:
        if int(e["obj"]) in ev_id:
            c[ev_id[int(e["obj"])]].append((int(e["f_start"]), int(e["f_end"])))
    return dict(c), [tuple(w) for w in J.get("hoist_coverage", {}).values()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True, help="token samples (+ extra frames), e.g. wm_v2_seg/data/*_source.npz")
    ap.add_argument("--episodes", default=str(ROOT / "data/labels/episodes_all.json"))
    ap.add_argument("--have_only", action="store_true", help="only keyframes that already have a sample (no new extra frames)")
    ap.add_argument("--out", default="tasks3_world"); a = ap.parse_args()
    Z = dict(np.load(a.source, allow_pickle=True)); N = len(Z["snap_k"])
    vid = Z["video"].astype(str); kk = Z["snap_k"].astype(int); grp = Z["group"].astype(str)
    J = json.load(open(a.episodes))
    segs = [s for s in J["segments"] if s["kind"] == "episode"]
    at = defaultdict(list)
    for n in range(N):
        at[(vid[n], kk[n])].append(n)
    rep = dict(source=a.source, episodes=a.episodes, videos={}, missing_extra=0)
    rows = []                                                                  # (src, overrides)
    for v in sorted(set(vid)):
        kept = sorted(pickle.load(open(vdir(v) / "tracks.pkl", "rb"))["kept"])
        ks = [k for k in kept if (v, k) in at]
        rep["missing_extra"] += len(kept) - len(ks)
        cont, cov = contact_of(v)
        sv = [s for s in segs if s["video"] == v]
        n_ep = n_w = 0
        for k in ks:
            ep = next((s for s in sv if 10 * s["start_s"] <= k <= 10 * s["end_s"]), None)
            src = next((n for n in at[(v, k)] if ep is not None and grp[n] == ep["id"]), at[(v, k)][0])
            oid = Z["obj_id"][src].astype(int); om = Z["obj_mask"][src].astype(bool)
            S = len(oid)
            o = dict(task_text=ep["instruction"] if ep else "", group=ep["id"] if ep else f"{v}~world", sub_kind="whole" if ep else "world",
                     world=ep is None)
            if ep is None:                                                     # held-out windows stay in the test split
                te = next((s["id"] for s in sv if s["id"] in HELD_TWINS and 10 * s["start_s"] - 50 <= k <= 10 * s["end_s"] + 50), None)
                o["base_ep"] = te or f"{v}~world"
            else:
                o["base_ep"] = ep["id"]
            # T1: the same objects in the episode's end snapshot
            t1p = np.zeros((S, 3), np.float32); t1y = np.zeros(S, np.float32); t1ok = np.zeros(S, bool); pres = np.zeros(S, bool)
            if ep is not None:
                ke = next((x for x in kept if x >= 10 * ep["end_s"] and (v, x) in at), None) or max([x for x in kept if (v, x) in at and x <= 10 * ep["end_s"]], default=None)
                if ke is not None:
                    se = next((n for n in at[(v, ke)] if grp[n] == ep["id"]), at[(v, ke)][0])
                    eo = Z["obj_id"][se].astype(int); em = Z["obj_mask"][se].astype(bool); eg = Z["obj_geo"][se].astype(np.float32)
                    for j in range(S):
                        if not om[j] or oid[j] < 0 or Z["obj_kind"][src][j] != 0:
                            continue
                        hit = np.flatnonzero((eo == oid[j]) & em)
                        t1ok[j] = True
                        if len(hit):
                            pres[j] = True; t1p[j] = eg[hit[0], :3]; t1y[j] = 0.5 * np.arctan2(eg[hit[0], 6], eg[hit[0], 7])
                o["t1_k"] = ke if ke is not None else -1
            # T2 (tracker displacements), T5 (future contact) and the contact input at k
            t2 = np.zeros((S, len(T2_S), 3), np.float32); t2ok = np.zeros((S, len(T2_S)), bool)
            t5 = np.zeros((S, len(T2_S)), bool); t5ok = np.zeros((S, len(T2_S)), bool)
            c0 = np.zeros(S, bool); c0ok = np.zeros(S, bool)
            in_cov = lambda f: any(w0 <= f <= w1 for w0, w1 in cov)              # noqa: E731
            held = lambda j, f: any(f0 <= f <= f1 for f0, f1 in cont.get(int(oid[j]), []))   # noqa: E731
            for j in range(S):
                if not om[j] or oid[j] < 0:
                    continue
                p0 = pose_at(v, int(oid[j]), k, "after", max_gap=3)
                for h, dt in enumerate(T2_S):
                    f = k + int(round(10 * dt))
                    if p0 is not None:
                        p1 = pose_at(v, int(oid[j]), f, "after", max_gap=3)
                        if p1 is not None:
                            t2[j, h] = np.subtract(p1[:3], p0[:3]); t2ok[j, h] = True
                    if in_cov(f):
                        t5[j, h] = held(j, f); t5ok[j, h] = True
                if in_cov(k):
                    c0[j] = held(j, k); c0ok[j] = True
            t3 = next((x for x in kept if abs(x - (k + 10 * T3_S)) <= 3), -1)
            o.update(t1_pos=t1p, t1_yaw=t1y, t1_ok=t1ok, t1_present=pres, t2_disp=t2, t2_ok=t2ok, t5=t5, t5_ok=t5ok,
                     obj_contact=c0, obj_contact_ok=c0ok, t3_k=t3)
            rows.append((src, o)); n_ep += ep is not None; n_w += ep is None
        rep["videos"][v] = dict(keyframes=len(kept), examples=len(ks), in_episodes=n_ep, outside=n_w)
    src = np.array([r[0] for r in rows]); M = len(rows)
    out = {f: x[src].copy() for f, x in Z.items()}
    for f in ("group", "task_text"):
        out[f] = np.array([r[1][f] for r in rows])
    # inert placeholders: no move labels of any kind
    out["mover"][:] = -1; out["target"][:] = -1; out["mover_set"][:] = 0; out["relation"][:] = -1
    for f in ("prog_video_ok", "end_ok", "path_hand_ok", "vj_future_ok", "fp_move_ok", "rot_ok"):
        if f in out:
            out[f][:] = 0
    out["ev_id"][:] = -1; out["time_to_move"][:] = np.nan
    for f in ("t1_pos", "t1_yaw", "t1_ok", "t1_present", "t2_disp", "t2_ok", "t5", "t5_ok", "obj_contact", "obj_contact_ok"):
        out[f] = np.stack([r[1][f] for r in rows])
    out["t1_k"] = np.array([r[1]["t1_k"] if "t1_k" in r[1] else -1 for r in rows], np.int64); out["t3_k"] = np.array([r[1]["t3_k"] for r in rows], np.int64)
    out["src_index"] = src.astype(np.int64); out["fut_src"] = np.full(M, -1, np.int64)
    out["sub_kind"] = np.array([r[1]["sub_kind"] for r in rows]); out["base_ep"] = np.array([r[1]["base_ep"] for r in rows])
    out["world_only"] = np.array([r[1]["world"] for r in rows])
    out["dataset"] = np.full(M, "all")                                        # no task tag anywhere
    DST.mkdir(parents=True, exist_ok=True)
    np.savez(DST / f"{a.out}.npz", **out)
    rep.update(n_examples=M, in_episodes=int((~out["world_only"]).sum()), outside=int(out["world_only"].sum()),
               t1_ok=float(out["t1_ok"].sum() / max(1, out["obj_mask"][~out["world_only"]].sum())), t1_present=float(out["t1_present"][out["t1_ok"]].mean()),
               t2_ok=float(out["t2_ok"].mean()), t5_ok=float(out["t5_ok"].mean()), t5_pos=float(out["t5"][out["t5_ok"]].mean()),
               contact_input_ok=float(out["obj_contact_ok"].mean()), t3_ok=float((out["t3_k"] >= 0).mean()), steps_s=T2_S, test=int(sum(
                   (b in HELD_TWINS) or b.startswith(HELD_VIDEOS) for b in out["base_ep"])))
    jdump(rep, DST / f"{a.out}_report.json")
    log(f"world_data: {M} examples ({rep['in_episodes']} in episodes, {rep['outside']} outside), keyframes without a sample "
        f"{rep['missing_extra']} -> {DST / (a.out + '.npz')}")


if __name__ == "__main__":
    main()
