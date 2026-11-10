"""Task-agnostic hand-object contact. Same model output and linking rule as cleantables.perception.contact.final, but linked to every
generic SAM 3 video track (tracks.pkl) rather than only to twin library objects, and without the WiLoR hand side (a move needs only
"some hand holds it").
  1 HOIST-Former held masks: the published model at its demo settings (score >= 0.5 is already applied in work/<ep>.npz), 6 fps.
  2 each held mask -> the generic track whose SAM 3 mask (its sighting at the nearest frame within +-1 work frame) overlaps it most
    (highest IoU, no threshold, as in contact.final); no overlap -> 'unlinked' (counted).
  3 raw intervals per track = runs of consecutive processed (6 fps) frames in which it is held. Nothing merged or smoothed here.
Output OUT/contact_moves/hoist_generic/<ep>.npz: work_frames (F,), held (F, n_tracks) float32 score (0 = not held), iou (F, n_tracks)
  + <ep>.json: intervals [{track, phrase, k_on, k_off, n_frames, score_max}], counts, frames scored.
  $VLA_PY -m cleantables.perception.contact.generic [EP...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import cv2
import numpy as np

from cleantables.perception.episode_io import OUT, ROOT, sam3_masks, tracks
from cleantables.perception.contact.final import HF, hoist_file as _hoist_file

H, W = 270, 480
DST = OUT / "contact_moves" / "hoist_generic"
# Track-level linking: each HOIST-Former track (its own temporal identity within a chunk) is linked as a whole to the generic track
# with the highest IoU summed over the HOIST track's frames (no threshold). link_check measures how often per-frame linking switches
# object within one HOIST track. Written to hoist_generic_trk/; the per-frame result stays in hoist_generic/.
DST_TRK = OUT / "contact_moves" / "hoist_generic_trk"


def hoist_file(ep):
    """HOIST-Former output for an episode: the main run, else the later runs on added episodes and stretches (same settings)."""
    p = _hoist_file(ep)
    for run in ("new7_6fps_L18", "stretch_6fps_L18"):
        q = HF / run / f"{ep}.npz"
        if p is None and q.exists():
            p = q
    return p


def run_video(v, eps, by_track=False):
    R = tracks(v)
    M, _, _, _, _ = sam3_masks(v)
    objs = R["objects"]
    mk = [{int(k): d for k, d in zip(o["sight_k"], o["det"]) if d is not None} for o in objs]
    for ep in eps:
        hp = hoist_file(ep)
        if hp is None:
            print(ep, "no HOIST-Former output", flush=True); continue
        z = np.load(hp, allow_pickle=True)
        shape = tuple(int(x) for x in z["shape"])
        Mh = np.unpackbits(z["masks_packed"], axis=-1, count=shape[-1]).reshape(shape).astype(bool)
        wf = z["work_frames"].astype(int); sc = z["scores"].astype(float)
        held = np.zeros((len(wf), len(objs)), np.float32); iou = np.zeros_like(held)
        cnt = dict(held_masks=0, unlinked=0)
        if by_track:
            for t in range(shape[0]):
                S_ = np.zeros(len(objs)); fr_ = []
                for fi, k in enumerate(wf):
                    m = Mh[t, fi]
                    if not m.any():
                        continue
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    fr_.append(fi); cnt["held_masks"] += 1
                    for i, mm in enumerate(mk):
                        near = next((kk for kk in (k, k - 1, k + 1) if kk in mm), None)
                        if near is None:
                            continue
                        s_ = M[mm[near]]; u = (m | s_).sum()
                        S_[i] += (m & s_).sum() / u if u else 0.0
                if not fr_:
                    continue
                if S_.max() <= 0:
                    cnt["unlinked"] += len(fr_); continue
                bi = int(S_.argmax())
                for fi in fr_:
                    held[fi, bi] = max(held[fi, bi], sc[t]); iou[fi, bi] = max(iou[fi, bi], S_[bi] / len(fr_))
        for fi, k in enumerate(wf if not by_track else []):
            sm = []
            for mm in mk:
                near = next((kk for kk in (k, k - 1, k + 1) if kk in mm), None)
                sm.append(M[mm[near]] if near is not None else None)
            for t in range(shape[0]):
                m = Mh[t, fi]
                if not m.any():
                    continue
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                cnt["held_masks"] += 1
                best, bi = 0.0, None
                for i, s_ in enumerate(sm):
                    if s_ is None:
                        continue
                    u = (m | s_).sum()
                    val = (m & s_).sum() / u if u else 0.0
                    if val > best:
                        best, bi = val, i
                if bi is None:
                    cnt["unlinked"] += 1; continue
                held[fi, bi] = max(held[fi, bi], sc[t]); iou[fi, bi] = max(iou[fi, bi], best)
        ints = []
        for i in range(len(objs)):
            on = held[:, i] > 0; j = 0
            while j < len(wf):
                if not on[j]:
                    j += 1; continue
                a = j
                while j < len(wf) and on[j]:
                    j += 1
                ints.append(dict(track=i, phrase=objs[i]["phrase"], k_on=int(wf[a]), k_off=int(wf[j - 1]), n_frames=j - a,
                                 score_max=round(float(held[a:j, i].max()), 3), iou_med=round(float(np.median(iou[a:j, i])), 3)))
        ints.sort(key=lambda d: d["k_on"])
        dst = DST_TRK if by_track else DST
        dst.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst / f"{ep}.npz", work_frames=wf, held=held, iou=iou)
        json.dump(dict(episode=ep, video=v, hoist_file=str(hp), frames_scored=[int(wf[0]), int(wf[-1])], n_frames=len(wf),
                       n_tracks=len(objs), counts=cnt, intervals=ints,
                       rule=("HOIST track (whole) -> generic SAM 3 track with the highest IoU summed over its frames (no threshold)"
                             if by_track else "held mask -> generic SAM 3 track with highest mask IoU (no threshold)")
                       + "; raw 6 fps runs, no merging"),
                  open(dst / f"{ep}.json", "w"), indent=1)
        print(ep, cnt, len(ints), "raw intervals", flush=True)


def main(eps, by_track=False):
    if not eps:
        eps = [e["id"] for e in json.load(open(ROOT / "data/labels/episodes_all.json"))["episodes"]]
    by = defaultdict(list)
    for ep in eps:
        by[ep.rsplit("_t", 1)[0]].append(ep)
    for v, es in sorted(by.items()):
        if not (ROOT / "data/generic" / v / "tracks.pkl").exists():
            print(v, "no generic tracks yet:", es, flush=True); continue
        run_video(v, es, by_track)




def link_check(eps=None):
    """How often does the per-frame link (held mask -> SAM 3 track with the highest IoU at the nearest mask frame) pick a different object
    than the HOIST track's own majority? HOIST-Former keeps one id per held object within a chunk (its own temporal association). Also
    counts frames where the per-frame rule finds no overlap ('unlinked') but the HOIST track is linked on other frames of the chunk.
    -> OUT/contact_moves/link_check.json"""
    from collections import Counter
    if not eps:
        eps = [e["id"] for e in json.load(open(ROOT / "data/labels/episodes_all.json"))["episodes"]]
    by = defaultdict(list)
    for ep in eps:
        by[ep.rsplit("_t", 1)[0]].append(ep)
    tot = Counter(); per = {}
    for v, es in sorted(by.items()):
        if not (ROOT / "data/generic" / v / "tracks.pkl").exists():
            continue
        R = tracks(v); M, _, _, _, _ = sam3_masks(v)
        mk = [{int(k): d for k, d in zip(o["sight_k"], o["det"]) if d is not None} for o in R["objects"]]
        for ep in es:
            hp = hoist_file(ep)
            if hp is None:
                continue
            z = np.load(hp, allow_pickle=True)
            shape = tuple(int(x) for x in z["shape"])
            Mh = np.unpackbits(z["masks_packed"], axis=-1, count=shape[-1]).reshape(shape).astype(bool)
            wf = z["work_frames"].astype(int)
            c = Counter()
            for t in range(shape[0]):
                links = []
                for fi, k in enumerate(wf):
                    m = Mh[t, fi]
                    if not m.any():
                        continue
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    best, bi = 0.0, None
                    for i, mm in enumerate(mk):
                        near = next((kk for kk in (k, k - 1, k + 1) if kk in mm), None)
                        if near is None:
                            continue
                        s_ = M[mm[near]]; u = (m | s_).sum()
                        val = (m & s_).sum() / u if u else 0.0
                        if val > best:
                            best, bi = val, i
                    links.append(bi)
                got = [b for b in links if b is not None]
                if not got:
                    c["tracks_never_linked"] += 1; c["masks_unlinked_whole_track"] += len(links); continue
                maj = Counter(got).most_common(1)[0][0]
                c["masks"] += len(links)
                c["agree_majority"] += sum(b == maj for b in links)
                c["differ_majority"] += sum(b is not None and b != maj for b in links)
                c["unlinked_but_track_linked"] += sum(b is None for b in links)
                c["hoist_tracks"] += 1; c["tracks_with_disagreement"] += int(any(b is not None and b != maj for b in links))
            per[ep] = dict(c); tot.update(c)
            print(ep, dict(c), flush=True)
    json.dump(dict(total=dict(tot), per_episode=per), open(OUT / "contact_moves" / "link_check.json", "w"), indent=1)
    print("TOTAL", dict(tot))


if __name__ == "__main__":
    if "--link-check" in sys.argv:
        link_check([a for a in sys.argv[1:] if not a.startswith("--")])
    else:
        main([a for a in sys.argv[1:] if not a.startswith("--")], by_track="--by-track" in sys.argv)
