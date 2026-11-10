"""HOIST-Former (CVPR 2024, published model and demo settings, run in third_party/hoistformer at its training rate 6 fps in ~3 s chunks
because a whole clip does not fit in GPU memory) -> 'held' per twin object (HOIST-Former gives no hand side).

Format conversion (the only steps added here, recorded as such):
  * each HOIST-Former track mask on a processed frame is assigned to the twin object whose SAM 3 mask (nearest mask frame within +-1)
    it overlaps most (highest IoU, no threshold); no overlap -> 'unlinked';
  * a work frame between processed (6 fps) frames takes the nearest processed frame.
Out: OUT/contact_hoist/<ep>_frames.npz (left_state = held, right_state unused) + overlay OUT/qc/<ep>/contact_hoistformer.mp4
  $FP_PY -m cleantables.perception.contact.hoist [EP...]
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from cleantables.perception.episode_io import OUT, TRIAL, episode, sam3_masks, scene, tracks

HF = OUT / "third_party/hoistformer/work"


def run(ep):
    from cleantables.perception.contact.qc import run as overlay
    e = episode(ep)
    V = e["video"]
    z = np.load(HF / f"{ep}.npz", allow_pickle=True)
    shape = tuple(int(x) for x in z["shape"])
    Mh = np.unpackbits(z["masks_packed"], axis=-1, count=shape[-1]).reshape(shape).astype(bool)   # (n_tracks, n_frames, 540, 960)
    wf = z["work_frames"].astype(int)
    S, R = scene(ep), tracks(V)
    M, _, _, _, _ = sam3_masks(V)
    objs = [o for o in S["objects"] if o.get("track") is not None and o["track"] >= 0]
    mk = [{int(k): d for k, d in zip(R["objects"][o["track"]]["sight_k"], R["objects"][o["track"]]["det"]) if d is not None} for o in objs]
    ids = [o["id"] for o in objs] + ["unlinked"]
    held = np.ones((len(wf), len(ids)), np.int8)
    n_m = n_unl = 0
    for fi, k in enumerate(wf):
        sm = []
        for mm in mk:
            near = next((kk for kk in (k, k - 1, k + 1) if kk in mm), None)
            sm.append(M[mm[near]] if near is not None else None)
        for t in range(shape[0]):
            m = Mh[t, fi]
            if not m.any():
                continue
            m = cv2.resize(m.astype(np.uint8), (480, 270), interpolation=cv2.INTER_NEAREST).astype(bool)
            best, bi = 0.0, None
            for i, s_ in enumerate(sm):
                if s_ is None:
                    continue
                u = (m | s_).sum()
                v = (m & s_).sum() / u if u else 0.0
                if v > best:
                    best, bi = v, i
            n_m += 1
            n_unl += bi is None
            held[fi, bi if bi is not None else len(objs)] = 2
    ks = np.arange(e["w0"], e["w1"] + 1)
    near = np.abs(ks[:, None] - wf[None]).argmin(1)
    F = dict(k=ks, objects=np.array(ids), left_state=held[near], right_state=np.ones_like(held[near]))
    d = OUT / "contact_hoist"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{ep}_frames.npz", **F)
    print(ep, "processed frames", len(wf), "track masks", n_m, "unlinked", n_unl, flush=True)
    overlay(ep, F, name="contact_hoistformer", title="HOIST-Former (6 fps, ~3 s chunks); held masks -> SAM 3 objects by IoU", held_only=True)


if __name__ == "__main__":
    for ep in sys.argv[1:] or TRIAL:
        run(ep)
