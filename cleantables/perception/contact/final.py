"""Contact labels for the world model: HOIST-Former held objects plus the hand side from WiLoR, written in the layout that
cleantables.worldmodel.contact_head.load_contact_labels reads.

Steps (nothing is smoothed):
  1 HOIST-Former, published model and demo settings, 6 fps, <= 18-frame chunks (third_party/hoistformer, work/<ep>.npz).
  2 object (format conversion): each held mask -> the twin object whose SAM 3 mask (nearest mask frame within +-1) it overlaps most
    (highest IoU, no threshold); no overlap -> dropped from the per-object labels (counted as 'unlinked' in the summary).
  3 hand side (our combining step; no published method joins HOIST-Former with a hand model):
    WiLoR's left and right hand meshes of that frame (hands_raw.npz, projected with our K) -> the hand whose silhouette is nearest to
    the held mask (smallest pixel distance; 0 when they overlap) holds it. One hand detected -> that hand. No hand detected -> unknown
    side (state -1 for both hands on that object and frame).
  4 frames (format conversion): each 10 fps work frame of the episode window takes the nearest processed (6 fps) frame.
Output OUT/contact/<ep>/contact.npz:
  t (T,) float32 video seconds (k / 10); obj_ids (n,) twin object ids (matched by SAM 3 track; ids of rebuilt twins
  via perception_v2/twins_v2/id_map_current_to_rebuilt.json); state (2,T,n) int8 contact_head.CONTACT_STATES index
  (0 free, 2 grasped; 'touching' and 'released' are not produced by HOIST-Former), -1 unknown; conf (2,T,n) float16: the HOIST-Former
  track score where grasped, 1.0 where free (no held mask above the demo's 0.5 score), 0 where unknown.
  + contact.json: per hand and object the grasp intervals (on / off seconds) and the summary counts.
  $FP_PY -m cleantables.perception.contact.final [EP...]
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from cleantables.perception.episode_io import OUT, episode, geom, jdump, sam3_masks, scene, tracks

HF = OUT / "third_party/hoistformer/work"
H, W = 270, 480


def hoist_file(ep):
    for p in (HF / "all45_6fps_L18" / f"{ep}.npz", HF / f"{ep}.npz"):
        if p.exists():
            return p
    return None


def hand_sils(ep, K):
    z = np.load(OUT / "hands" / ep / "hands_raw.npz")
    k0, F = int(z["k0"]), z["faces"]
    out = {}
    for s in ("left", "right"):
        V = z[f"{s}_V"]
        for i in range(len(V)):
            if not z[f"{s}_ok"][i]:
                continue
            zz = np.maximum(V[i][:, 2], 1e-4)
            uv = np.stack([K[0, 0] * V[i][:, 0] / zz + K[0, 2], K[1, 1] * V[i][:, 1] / zz + K[1, 2]], -1) / 2.0   # 270x480 grid
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, list(uv[F].astype(np.int32)), 1)
            out.setdefault(k0 + i, {})[s] = m.astype(bool)
    return out


def run(ep):
    e = episode(ep)
    V = e["video"]
    hp = hoist_file(ep)
    if hp is None:
        print(ep, "no HOIST-Former output yet", flush=True)
        return
    z = np.load(hp, allow_pickle=True)
    shape = tuple(int(x) for x in z["shape"])
    Mh = np.unpackbits(z["masks_packed"], axis=-1, count=shape[-1]).reshape(shape).astype(bool)
    wf = z["work_frames"].astype(int)
    sc = z["scores"].astype(float)
    G = geom(V)
    hs = hand_sils(ep, G["K"])
    S, R = scene(ep), tracks(V)
    M, _, _, _, _ = sam3_masks(V)
    objs = [o for o in S["objects"] if o.get("track") is not None and o["track"] >= 0]
    mk = [{int(k): d for k, d in zip(R["objects"][o["track"]]["sight_k"], R["objects"][o["track"]]["det"]) if d is not None} for o in objs]
    n = len(objs)
    st6 = np.zeros((2, len(wf), n), np.int8)               # 0 free
    cf6 = np.ones((2, len(wf), n), np.float32)
    cnt = dict(held_masks=0, unlinked=0, side_left=0, side_right=0, side_unknown=0)
    for fi, k in enumerate(wf):
        sm = []
        for mm in mk:
            near = next((kk for kk in (k, k - 1, k + 1) if kk in mm), None)
            sm.append(M[mm[near]] if near is not None else None)
        hk = hs.get(int(k), {})
        dist = {s: cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for s, m in hk.items()}
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
                v = (m & s_).sum() / u if u else 0.0
                if v > best:
                    best, bi = v, i
            if bi is None:
                cnt["unlinked"] += 1
                continue
            if not dist:
                st6[:, fi, bi] = -1; cf6[:, fi, bi] = 0
                cnt["side_unknown"] += 1
                continue
            d = {s: float(D[m].min()) for s, D in dist.items()}
            side = min(d, key=d.get)
            hi = 0 if side == "left" else 1
            st6[hi, fi, bi] = 2
            cf6[hi, fi, bi] = sc[t]
            cnt[f"side_{side}"] += 1
    ks = np.arange(e["w0"], e["w1"] + 1)
    near = np.abs(ks[:, None] - wf[None]).argmin(1)
    state, conf = st6[:, near], cf6[:, near]
    d = OUT / "contact" / ep
    d.mkdir(parents=True, exist_ok=True)
    ids = [o["id"] for o in objs]
    np.savez_compressed(d / "contact.npz", t=(ks / 10.0).astype(np.float32), obj_ids=np.array(ids), state=state,
                        conf=conf.astype(np.float16), k=ks)
    ints = []
    for hi, s in enumerate(("left", "right")):
        for oi, oid in enumerate(ids):
            on = state[hi, :, oi] == 2
            j = 0
            while j < len(ks):
                if not on[j]:
                    j += 1
                    continue
                a = j
                while j < len(ks) and on[j]:
                    j += 1
                ints.append(dict(hand=s, object=oid, on_s=round(ks[a] / 10, 2), off_s=round(ks[j - 1] / 10, 2), n_frames=j - a))
    jdump(dict(episode=ep, video=V, labeller="HOIST-Former (CVPR 2024, published) + WiLoR hand side (nearest hand silhouette)",
               hoist_file=str(hp), fps=10.0, window=[int(e["w0"]), int(e["w1"])], counts=cnt, grasps=ints,
               states=dict(codes="contact_head.CONTACT_STATES: 0 free, 2 grasped, -1 unknown (held, hand side unknown)",
                           not_produced=["touching", "released"])), d / "contact.json")
    print(ep, cnt, len(ints), "grasp intervals", flush=True)


def overlay(ep):
    """qc/<ep>/contact_final.mp4: the final labels (HOIST-Former + WiLoR side) drawn like the other contact overlays."""
    from cleantables.perception.contact.qc import run as draw
    z = np.load(OUT / "contact" / ep / "contact.npz", allow_pickle=True)
    conv = lambda s: np.where(s == 2, 2, np.where(s == -1, 0, 1)).astype(np.int8)          # noqa: E731  -> qc codes (2 on, 1 off, 0 unk)
    pad = lambda a: np.concatenate([a, np.ones((a.shape[0], 1), np.int8)], 1)               # noqa: E731  'unlinked' column
    F = dict(k=z["k"], objects=np.array(list(z["obj_ids"]) + ["unlinked"]), left_state=pad(conv(z["state"][0])),
             right_state=pad(conv(z["state"][1])))
    draw(ep, F, name="contact_final", title="HOIST-Former held + WiLoR hand side (nearest hand)")


if __name__ == "__main__":
    eps = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not eps:
        from cleantables.perception.episode_io import ROOT
        eps = [t["id"] for t in json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]]
    for ep in eps:
        run(ep)
        if (OUT / "contact" / ep / "contact.npz").exists() and "--overlay" in sys.argv:
            overlay(ep)
