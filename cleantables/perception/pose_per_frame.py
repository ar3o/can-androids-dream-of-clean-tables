"""FoundationPose per-frame registration on the twin meshes (the OmniHOI recipe, arXiv 2610.10855: SAM 3 mask + MoGe depth + mesh ->
FoundationPose register on every frame, 5 refinement iterations), a published alternative to FoundationPose tracking
(cleantables.perception.pose_tracking), which can lose objects while the hand covers them.

Nothing is added to the method: no tracking, no smoothing, no physics checks, no re-registration logic. Per episode and per movable
twin object (scene.json role task / container / stack), on every frame of the window where SAM 3 has a mask for the object's track:
  FoundationPose(...).register(K, rgb, depth, ob_mask, iteration=5)
  rgb   the 960x540 work frame;  depth  the stored metric MoGe-3 depth (table-plane scale per frame), with the SAM 3 'hand'
  pixels set invalid except inside the object's own mask (as in pose_tracking);  mesh  the twin's .obj with its library texture
  (pose_tracking.load_mesh), i.e. the Hunyuan3D-2.1 / library mesh at the twin's scale.
Wrapper code (model loading, mesh texture, silhouette render) is reused from cleantables.perception.pose_tracking.
Out: OUT/pose/<ep>/fp_perframe.npz  per object '<id>': frames (m,), pose_cam (m,4,4) mesh -> camera, pose_table (m,4,4) mesh -> table,
     reg_s (m,) seconds per registration; meta json. Resumable per episode.
  $FP_PY -m cleantables.perception.pose_per_frame [EP...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

from cleantables.perception.pose_tracking import EP, Tracker, torch  # noqa: E402  (also puts FoundationPose on sys.path)

from cleantables.perception.episode_io import OUT, SUF, TRIAL, mesh_path, depth_full, episode, frame, geom, jdump, sam3_masks, scene, tracks

MOVABLE = ("task", "container", "stack")


def load_mesh(ep, o):
    """pose_tracking.load_mesh (twin .obj + library texture on its UVs), but from episode_io.mesh_path so a twin variant can be used."""
    import trimesh
    from PIL import Image
    m = trimesh.load(mesh_path(ep, o["id"]), force="mesh", process=False)
    tex = EP / "library" / o["library"] / "texture.png"
    uv = getattr(m.visual, "uv", None)
    if tex.exists() and uv is not None and len(uv) == len(m.vertices):
        m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.open(tex).convert("RGB"))
    return m


def run(trk, ep):
    out = OUT / "pose" / ep / f"fp_perframe{SUF}.npz"
    if out.exists():
        print(ep, "fp_perframe exists, skipped (resume)", flush=True)
        return
    t0 = time.time()
    e = episode(ep)
    V = e["video"]
    G = geom(V)
    K, T = G["K"], G["T"]
    ks, D = depth_full(ep)
    mp = OUT / "depth" / ep / "maps.npz"
    # per-frame table-plane scale: a recomputed one where it exists (trial episodes), otherwise det3d's stored one
    # (the same quantity; they agree to 0.02-0.3 % median on the trial episodes)
    a_cur = np.load(mp)["a_cur"] if mp.exists() else np.asarray(G["scale"][ks], np.float32)
    k2j = {int(k): j for j, k in enumerate(ks)}
    M, _, _, _, hand = sam3_masks(V)
    R = tracks(V)
    S = scene(ep)
    res, meta = {}, {}
    for o in S["objects"]:
        if o["role"] not in MOVABLE or o.get("track") is None or o["track"] < 0:
            continue
        ro = R["objects"][o["track"]]
        fr = [(int(k), d) for k, d in zip(ro["sight_k"], ro["det"]) if int(k) in k2j and d is not None]
        trk.set_mesh(load_mesh(ep, o))
        P, Pt, secs, kk = [], [], [], []
        for k, d in fr:
            mk = cv2.resize(M[d].astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mk.sum() < 40:
                continue
            dep = np.repeat(np.repeat(D[k2j[k]] * a_cur[k2j[k]], 2, 0), 2, 1).astype(np.float32)
            hk = next((hand[j] for j in (k, k - 1, k + 1) if j in hand), None)
            if hk is not None:
                hm = np.repeat(np.repeat(cv2.dilate(hk.astype(np.uint8), np.ones((5, 5), np.uint8)), 2, 0), 2, 1).astype(bool) & ~mk
                dep[hm] = 0
            rgb = frame(V, k)[..., ::-1].copy()
            ts = time.time()
            for attempt in range(60):          # the GPU is shared: wait out other processes' transient peaks (as pose_tracking does)
                try:
                    p = np.asarray(trk.est.register(K=K, rgb=rgb, depth=dep, ob_mask=mk, iteration=5)).reshape(4, 4)
                    break
                except (torch.OutOfMemoryError, RuntimeError) as ex:
                    if not isinstance(ex, torch.OutOfMemoryError) and not any(w_ in str(ex) for w_ in ("allocate", "out of memory", "cudaMalloc")):
                        raise
                    torch.cuda.empty_cache(); time.sleep(5 + 2 * attempt)
            else:
                raise RuntimeError("register: out of GPU memory 60x")
            torch.cuda.empty_cache()
            secs.append(time.time() - ts)
            P.append(p); Pt.append(T @ p); kk.append(k)
        res[o["id"]] = dict(frames=np.array(kk), pose_cam=np.array(P).reshape(-1, 4, 4), pose_table=np.array(Pt).reshape(-1, 4, 4),
                            reg_s=np.array(secs))
        meta[o["id"]] = dict(n=len(kk), track=o["track"], role=o["role"], median_reg_s=float(np.median(secs)) if secs else None)
        print(ep, o["id"], len(kk), "frames", f"{np.median(secs) if secs else 0:.2f}s/frame", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp.npz"
    np.savez_compressed(tmp, **{f"{oid}__{k}": v for oid, r in res.items() for k, v in r.items()})
    os.replace(tmp, out)
    jdump(dict(episode=ep, objects=meta, wall_s=round(time.time() - t0, 1),
               peak_mib=round(torch.cuda.max_memory_allocated() / 2**20)), out.with_suffix(".json"))


if __name__ == "__main__":
    trk = Tracker()
    for ep in sys.argv[1:] or TRIAL:
        run(trk, ep)
