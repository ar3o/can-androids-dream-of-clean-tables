"""FoundationPose 6D tracking of each episode's task objects and containers, for 3D carry paths for the world model.

The SAM-mask-centroid + MoGe-depth box sightings (video_tracking / events.clean_path) give carry paths that barely lift, because the
hand occludes the object during the carry. Here the episode twin's scaled mesh is registered to the real frames with FoundationPose
(model-based, RGB + metric MoGe depth + K) and tracked at 10 fps through the episode window.

Per episode (data/labels/tasks_train.json), per twin object with role 'task' or 'container' (scene.json objects[]):
  mesh      data/twin_table/ep/<ep>/usd/<id>.obj (the twin's per-episode mesh: library mesh x video_scale, yaw baked, footprint centre at
            the origin, bottom at z = 0, metres). Its texture is re-attached from library/<key>/texture.png (the .obj files of one
            episode share one material_0.png, so the exported texture is only right for the last object written).
  window    the labelled episode window (start_s..end_s at 10 fps), extended to the end of every move that starts inside it.
  2 passes  F: register (FoundationPose register with the SAM 3 mask of the object's track, sam3_vinv.npz) at the largest of the first 4
            hand-free sightings, track forward to the window end (then back to its start); B: register at the largest of the last 4
            hand-free sightings and track backward. The depth fed to FoundationPose has the hand pixels (SAM 3 'hand', nearest
            keyframe, dilated) set invalid.
  checks    every frame with a usable SAM 3 mask of the object (hand overlap < 0.95, >= 15-30 px, <= 2.5x the anchor mask area):
            silhouette IoU of the rendered mesh vs the mask, both minus the hand pixels (270x480). IoU < IOU_REREG -> re-register from
            that mask (hand pixels removed from the depth except the mask's own), kept if IoU >= max(0.5, old + 0.1) and the mesh
            bottom is not below the table. Physics checks every frame: bottom below the table (< -3 cm), centre off the table extent
            (+10 cm), or a jump > JUMP_MPS between frames (not counted at re-registrations).
  reliable  per pass, a frame is anchored if it is physically plausible and within GAP_OK frames of a frame that pass verified against
            a mask (IoU >= IOU_OK) with no failed check on a mostly visible mask (hand overlap < 0.6) in between. Reliable = both
            passes anchored and agreeing (top-centre < FB_AGREE_M apart), or only one anchored (its pose is used), or neither
            anchored but both plausible and agreeing. Unreliable frames keep their raw pose (flag 0); nothing is interpolated here.
Output: data/generic/<V>/fp_tracks.npz with, per object key '<ep>/<obj_id>':
  frames (n,), pose_cam (n,4,4) mesh->camera (OpenCV, 960x540 K), pose_table (n,4,4) mesh->table (T_all[k] @ pose_cam),
  top (n,3) top-centre point in the table frame (bbox-centre x, y; max z of the posed mesh) = events.json 'path' convention,
  iou (n,) mask IoU (nan = no trusted mask), kind (n,) register/track/reregister, reliable / verified / phys (n,) bool,
  fb_dist (n,) F-B top-centre distance, top_F / top_B (n,3); plus meta json (mesh path, scale check vs the measured footprint, anchors).
  $FP_PY -m cleantables.perception.pose_tracking EPISODES... [--qc]     (FP_PY = envs/fpose/bin/python, from the project root)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FP = ROOT / "third_party/FoundationPose"
sys.path.insert(0, str(FP))
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, set_logging_format, set_seed  # noqa: E402
from Utils import dr, draw_xyz_axis, make_mesh_tensors, nvdiffrast_render, torch, trimesh  # noqa: E402
from PIL import Image  # noqa: E402

TASKS = ROOT / "data/labels/tasks_train.json"
EP = ROOT / "data/twin_table/ep"
GEN = ROOT / "data/generic"
GEOM = ROOT / "data/twin_table/boxstack"
WORK = ROOT / "data/table_clean/_work"
H, W = 270, 480
IOU_REREG, IOU_OK, JUMP_MPS, FB_AGREE_M, GAP_OK = 0.45, 0.35, 3.0, 0.04, 3
REG_HAND_MAX, CHK_HAND_MAX = 0.05, 0.95


def load_masks(V):
    z = np.load(GEN / V / "sam3_vinv.npz")
    shp = tuple(z["shape"])
    M = np.unpackbits(z["masks"])[:int(np.prod(shp))].reshape(shp).astype(bool)
    hz = np.load(GEN / V / "hand_video.npz")
    hs = tuple(hz["shape"])
    Hm = np.unpackbits(hz["masks"])[:int(np.prod(hs))].reshape(hs).astype(bool)
    hand = {int(k): Hm[t] for t, k in enumerate(hz["frames"])}
    return M, hand


def load_mesh(ep, o):
    m = trimesh.load(EP / ep / "usd" / f"{o['id']}.obj", force="mesh", process=False)
    tex = EP / "library" / o["library"] / "texture.png"
    uv = getattr(m.visual, "uv", None)
    if tex.exists() and uv is not None and len(uv) == len(m.vertices):
        m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.open(tex).convert("RGB"))
    return m




class Tracker:
    def __init__(self):
        set_logging_format()
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        set_seed(0)
        self.scorer, self.refiner = ScorePredictor(), PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = None

    def set_mesh(self, mesh):
        if self.est is None:
            self.est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=self.scorer,
                                      refiner=self.refiner, debug_dir=str(ROOT / "cache/fp_dbg"), debug=0, glctx=self.glctx)
        else:
            self.est.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)
        self.mt = make_mesh_tensors(mesh)
        self.tfc = self.est.get_tf_to_centered_mesh().cpu().numpy()

    def silhouette(self, pose, K_lo):
        p = torch.as_tensor(pose[None], device="cuda", dtype=torch.float)
        _, d, _ = nvdiffrast_render(K=K_lo, H=H, W=W, ob_in_cams=p, glctx=self.glctx, mesh_tensors=self.mt)
        return (d[0] > 0).cpu().numpy()

    def score(self, rgb, depth, K, pose):
        pc = pose @ np.linalg.inv(self.tfc)               # mesh-file pose -> centred-mesh pose (what the scorer expects)
        s, _ = self.scorer.predict(mesh=self.est.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=pc[None].astype(np.float32),
                                   mesh_tensors=self.est.mesh_tensors, glctx=self.glctx, mesh_diameter=self.est.diameter)
        return float(np.asarray(s.detach().cpu() if torch.is_tensor(s) else s).reshape(-1)[0])

    def set_pose(self, pose):
        self.est.pose_last = torch.as_tensor(pose @ np.linalg.inv(self.tfc), device="cuda", dtype=torch.float)


def iou_nohand(sil, mk, hand):
    a, b = sil & ~hand, mk & ~hand
    u = (a | b).sum()
    return float((a & b).sum() / u) if u > 0 else np.nan


def run_episode(trk, t, qc=False, log=print):
    ep, V = t["id"], t["video"]
    S = json.load(open(EP / ep / "scene.json"))
    z = np.load(GEOM / V / "det3d.npz")
    K = z["K"].astype(np.float64); T_all = z["T_all"]; scale = z["scale"].astype(np.float32)
    K_lo = K.copy(); K_lo[0, 0] /= 2; K_lo[1, 1] /= 2; K_lo[0, 2] = (K[0, 2] - 0.5) / 2; K_lo[1, 2] = (K[1, 2] - 0.5) / 2
    tab = z["table"]
    D = np.load(GEOM / V / "depth10.npy", mmap_mode="r")
    R = pickle.load(open(GEN / V / "tracks.pkl", "rb"))
    M, hand = load_masks(V)
    k0, k1 = int(round(t["start_s"] * 10)), int(round(t["end_s"] * 10))
    # extend the window to the end of every move that starts inside it (the world model's path labels need the whole carry)
    Ev = json.load(open(GEN / V / "events.json"))
    fe = [e["f_end"] for e in Ev["events"] if e.get("kind", "move") == "move" and k0 <= e["f_start"] <= k1]
    k1 = min(max([k1] + [f + 3 for f in fe]), len(T_all) - 1)
    # table xy extent (from the table pixel mask at k0, depth backprojected)
    ys, xs = np.nonzero(tab)
    dk = np.asarray(D[k0], np.float32)[ys, xs] * scale[k0]
    P = np.stack([(xs * 2 + 0.5 - K[0, 2]) * dk / K[0, 0], (ys * 2 + 0.5 - K[1, 2]) * dk / K[1, 1], dk], 1)
    Q = P @ T_all[k0][:3, :3].T + T_all[k0][:3, 3]
    ext = np.percentile(Q[:, :2], [1, 99], axis=0)
    rgb_c, dep_c = {}, {}

    def rgb(k):
        if k not in rgb_c:
            rgb_c[k] = cv2.imread(str(WORK / V / "frames" / f"{k}.jpg"))[..., ::-1].copy()
        return rgb_c[k]

    def dep(k):
        if k not in dep_c:
            d = np.asarray(D[k], np.float32) * scale[k]
            dep_c[k] = np.repeat(np.repeat(d, 2, 0), 2, 1)
        return dep_c[k]

    def dep_nohand(k, keep=None):
        """Metric depth with the hand pixels (nearest SAM 3 hand mask within 1 frame, dilated) set invalid (0); pixels of `keep`
        (the object's own SAM 3 mask, 540x960) are kept."""
        d = dep(k).copy()
        hk = next((hand[j] for j in (k, k - 1, k + 1) if j in hand), None)
        if hk is not None:
            hm = np.repeat(np.repeat(cv2.dilate(hk.astype(np.uint8), np.ones((5, 5), np.uint8)), 2, 0), 2, 1).astype(bool)
            if keep is not None:
                hm &= ~keep
            d[hm] = 0
        return d

    out, metas = {}, {}
    objs = [o for o in S["objects"] if o["role"] in ("task", "container")]
    n = k1 - k0 + 1
    frames = np.arange(k0, k1 + 1)
    for o in objs:
        tr = o.get("track")
        if tr is None or tr < 0 or tr >= len(R["objects"]):
            log(f"{ep} {o['id']}: no track -> skipped"); continue
        ro = R["objects"][tr]
        sights = {int(k): (d, b) for k, d, b in zip(ro["sight_k"], ro["det"], ro["boxes"]) if k0 <= k <= k1 and b is not None}
        mesh = load_mesh(ep, o)
        trk.set_mesh(mesh)
        verts = np.asarray(mesh.vertices)
        vs = verts[::max(1, len(verts) // 3000)]
        bc = (verts.min(0) + verts.max(0)) / 2
        # scale check: mesh footprint vs the measured (SAM mask + depth box) footprint of hand-free sightings in the window
        fr = [max(b["L"], b["W"]) for d, b in sights.values() if b["hand_frac"] < 0.05]
        meas = float(np.percentile(fr, 75)) if fr else float("nan")
        mesh_xy = float(max(mesh.extents[:2]))
        # anchors: largest hand-free SAM 3 mask in the first / last 3 s of the window
        cand = [(k, d) for k, (d, b) in sights.items() if b["hand_frac"] <= REG_HAND_MAX] or [(k, d) for k, (d, b) in sights.items()]
        if not cand:
            log(f"{ep} {o['id']}: no SAM 3 sighting in the window -> skipped"); continue
        cand.sort()
        kA, dA = max(cand[:4], key=lambda c: M[c[1]].sum())          # earliest hand-free sightings (before anything moves)
        kB, dB = max(cand[-4:], key=lambda c: M[c[1]].sum())         # latest ones (after the last move)

        def register(k, d):
            mk = cv2.resize(M[d].astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)
            for attempt in range(30):                     # parallel workers share the GPU: wait out transient out-of-memory peaks
                try:
                    p = trk.est.register(K=K, rgb=rgb(k), depth=dep_nohand(k, keep=mk), ob_mask=mk, iteration=5)
                    break
                except (torch.OutOfMemoryError, RuntimeError) as ex:      # torch or Warp (depth filters) allocation failures
                    if not isinstance(ex, torch.OutOfMemoryError) and not any(w_ in str(ex) for w_ in ("allocate", "out of memory", "cudaMalloc")):
                        raise
                    torch.cuda.empty_cache(); time.sleep(3 + 2 * attempt)
            else:
                raise RuntimeError("register: out of GPU memory 30x")
            torch.cuda.empty_cache()
            return np.asarray(p).reshape(4, 4)

        a_ref = float(M[dA].sum())

        def check(k, pose):
            """-> (IoU of the rendered silhouette vs the object's SAM 3 mask, hand pixels excluded; det id; trusted; clean).
            trusted = a usable mask of this object: hand overlap < 0.95, >= 15 px (>= 30 px if mostly hidden), <= 2.5x the anchor
            mask (not merged with something else). clean = trusted and hand overlap < 0.6 (a disagreement there counts against the pose)."""
            if k not in sights:
                return np.nan, None, False, False
            d, b = sights[k]
            area = M[d].sum()
            if b["hand_frac"] >= CHK_HAND_MAX or area < (15 if b["hand_frac"] < 0.6 else 30) or area > 2.5 * a_ref:
                return np.nan, None, False, False
            hk = hand.get(k, np.zeros((H, W), bool))
            return iou_nohand(trk.silhouette(pose, K_lo), M[d], hk), d, True, b["hand_frac"] < 0.6

        def bottom_z(k, pose):
            pt_ = T_all[k] @ pose
            return float((vs @ pt_[:3, :3].T + pt_[:3, 3])[:, 2].min())

        t0 = time.time()
        passes = {}
        for name, k_reg, d_reg in (("F", kA, dA), ("B", kB, dB)):
            pc = np.zeros((n, 4, 4)); kinds = np.array(["track"] * n, dtype="U10"); ious = np.full(n, np.nan)
            trust = np.zeros(n, bool); clean = np.zeros(n, bool); times = []
            p_reg = register(k_reg, d_reg)
            i_reg = k_reg - k0
            pc[i_reg] = p_reg; kinds[i_reg] = "register"
            ious[i_reg], _, trust[i_reg], clean[i_reg] = check(k_reg, p_reg)
            # F: forward from the early anchor to the end, then back to the start; B: backward from the late anchor, then forward
            order = (1, -1) if name == "F" else (-1, 1)
            for direction in order:
                trk.set_pose(p_reg)
                rng = range(k_reg + 1, k1 + 1) if direction == 1 else range(k_reg - 1, k0 - 1, -1)
                for k in rng:
                    ts = time.time()
                    for attempt in range(30):
                        try:
                            p = np.asarray(trk.est.track_one(rgb=rgb(k), depth=dep_nohand(k), K=K, iteration=2)).reshape(4, 4)
                            break
                        except (torch.OutOfMemoryError, RuntimeError) as ex:
                            if not isinstance(ex, torch.OutOfMemoryError) and not any(w_ in str(ex) for w_ in ("allocate", "out of memory", "cudaMalloc")):
                                raise
                            torch.cuda.empty_cache(); time.sleep(3 + 2 * attempt)
                    else:
                        raise RuntimeError("track_one: out of GPU memory 30x")
                    iou, d, tq, cl = check(k, p)
                    if tq and (not np.isfinite(iou) or iou < IOU_REREG):
                        p2 = register(k, d)
                        iou2 = check(k, p2)[0]
                        if np.isfinite(iou2) and iou2 >= max(0.5, (iou if np.isfinite(iou) else 0) + 0.1) and bottom_z(k, p2) > -0.03:
                            p, iou = p2, iou2; kinds[k - k0] = "reregister"
                        else:
                            trk.set_pose(p)
                            if os.environ.get("FP_DEBUG"):
                                log(f"  {o['id']} {name} k={k}: re-register rejected (iou {iou:.2f} -> {iou2:.2f}, "
                                    f"reg top {(T_all[k] @ p2)[:3, 3].round(3)}, track top {(T_all[k] @ p)[:3, 3].round(3)})")
                    pc[k - k0] = p; ious[k - k0] = iou; trust[k - k0] = tq; clean[k - k0] = cl
                    times.append(time.time() - ts)
                    if len(rgb_c) > 40:
                        rgb_c.clear(); dep_c.clear()
            pt = np.einsum("nij,njk->nik", T_all[frames], pc)
            Vt = np.einsum("nij,vj->nvi", pt[:, :3, :3], vs) + pt[:, None, :3, 3]
            ctr = np.einsum("nij,j->ni", pt[:, :3, :3], bc) + pt[:, :3, 3]
            top = np.stack([ctr[:, 0], ctr[:, 1], Vt[:, :, 2].max(1)], 1)
            bottom = Vt[:, :, 2].min(1)
            phys = (bottom > -0.03) & (ctr[:, 0] > ext[0, 0] - 0.10) & (ctr[:, 0] < ext[1, 0] + 0.10) & \
                   (ctr[:, 1] > ext[0, 1] - 0.10) & (ctr[:, 1] < ext[1, 1] + 0.10)
            sp = np.r_[0, np.linalg.norm(np.diff(ctr, axis=0), axis=1) * 10]
            jump = (sp > JUMP_MPS) & (kinds != "reregister")
            phys &= ~jump
            ver = trust & (ious >= IOU_OK)
            fail = clean & (ious < IOU_OK)
            # anchored: within GAP_OK frames of a frame this pass verified against a SAM 3 mask, with no clean failure in between
            anch = np.zeros(n, bool)
            vi, fi = np.nonzero(ver)[0], np.nonzero(fail)[0]
            for i in range(n):
                # the pose at i was propagated from i-1 (forward segments) or i+1 (backward segments): only verified frames it was
                # propagated from count (F: forward after its anchor, backward before it; B: backward before its anchor, forward after)
                fwd = (i >= i_reg) if name == "F" else (i > i_reg)
                near = vi[(vi <= i) & (vi >= i - GAP_OK)] if fwd else vi[(vi >= i) & (vi <= i + GAP_OK)]
                anch[i] = any(not ((fi > min(i, j)) & (fi <= max(i, j))).any() if j != i else True for j in near)
            passes[name] = dict(pc=pc, pt=pt, top=top, bottom=bottom, kinds=kinds, ious=ious, trust=trust, phys=phys, ver=ver,
                                anch=anch & phys, ms=float(np.median(times) * 1000) if times else None)
        F, B = passes["F"], passes["B"]
        dFB = np.linalg.norm(F["top"] - B["top"], axis=1)
        agree = dFB < FB_AGREE_M
        aF, aB = F["anch"], B["anch"]
        # reliable: both passes anchored and agreeing, or only one anchored (use it), or neither anchored but both physically plausible
        # and agreeing (two independent trackers from opposite ends land on the same pose)
        reliable = (aF & aB & agree) | (aF ^ aB) | (~aF & ~aB & agree & F["phys"] & B["phys"])
        useB = aB & ~aF
        sel = lambda f: np.where(useB.reshape((-1,) + (1,) * (F[f].ndim - 1)), B[f], F[f])
        verified = F["ver"] | B["ver"]
        key = f"{ep}/{o['id']}"
        out[key] = dict(frames=frames, pose_cam=sel("pc").astype(np.float32), pose_table=sel("pt").astype(np.float32),
                        top=sel("top").astype(np.float32), bottom=sel("bottom").astype(np.float32),
                        iou=np.where(useB, B["ious"], F["ious"]).astype(np.float32), kind=np.where(useB, B["kinds"], F["kinds"]),
                        reliable=reliable, verified=verified, phys=F["phys"] | B["phys"], anch_F=aF, anch_B=aB, fb_dist=dFB.astype(np.float32),
                        top_F=F["top"].astype(np.float32), top_B=B["top"].astype(np.float32))
        metas[key] = dict(mesh=str((EP / ep / "usd" / f"{o['id']}.obj").relative_to(ROOT)), library=o["library"], track=tr,
                          role=o["role"], k_reg_F=int(kA), k_reg_B=int(kB), iou_reg_F=float(F["ious"][kA - k0]),
                          iou_reg_B=float(B["ious"][kB - k0]), mesh_xy_m=mesh_xy, measured_xy_m=meas,
                          scale_ratio=mesh_xy / meas if meas == meas else None, n=n, n_agree=int(agree.sum()),
                          n_verified=int(verified.sum()), n_reliable=int(reliable.sum()),
                          n_rereg=int((F["kinds"] == "reregister").sum() + (B["kinds"] == "reregister").sum()),
                          n_trusted_masks=int(F["trust"].sum()), n_anch_F=int(aF.sum()), n_anch_B=int(aB.sum()), track_ms=F["ms"], wall_s=round(time.time() - t0, 1),
                          reg_vs_twin_cm=float(np.linalg.norm(F["pt"][kA - k0, :2, 3] - np.array(o["pos"][:2])) * 100) if kA - k0 < 20 else None)
        print(f"{key}: anchors {kA}/{kB} iou {metas[key]['iou_reg_F']:.2f}/{metas[key]['iou_reg_B']:.2f}, F-B agree {agree.sum()}/{n}, "
            f"trusted masks {F['trust'].sum()}, verified {verified.sum()}, rereg {metas[key]['n_rereg']}, reliable {reliable.sum()}/{n}, "
            f"scale mesh/meas {metas[key]['scale_ratio']}, {metas[key]['wall_s']} s ({F['ms']} ms/track)", flush=True)
    if qc:
        qc_video(ep, V, k0, k1, out, metas, K, rgb)
    return out, metas


def qc_video(ep, V, k0, k1, out, metas, K, rgb):
    tmp = ROOT / f"videos/generic/fp_qc/_{ep}"
    tmp.mkdir(parents=True, exist_ok=True)
    cols = [(80, 255, 80), (255, 220, 0), (255, 0, 255), (0, 255, 255), (255, 140, 0), (180, 255, 180), (255, 255, 255), (200, 120, 255)]
    meshes = {key: trimesh.load(ROOT / m["mesh"], force="mesh", process=False) for key, m in metas.items()}
    for key, m in meshes.items():
        m.vertices = m.vertices[:]
    glctx = dr.RasterizeCudaContext()
    mts = {key: make_mesh_tensors(trimesh.Trimesh(m.vertices, m.faces, process=False)) for key, m in meshes.items()}
    for i, k in enumerate(range(k0, k1 + 1)):
        im = rgb(k)[..., ::-1].copy()
        ov = im.copy()
        y = 22
        for j, (key, r) in enumerate(out.items()):
            p = torch.as_tensor(r["pose_cam"][i][None], device="cuda", dtype=torch.float)
            _, d, _ = nvdiffrast_render(K=K, H=540, W=960, ob_in_cams=p, glctx=glctx, mesh_tensors=mts[key])
            sil = (d[0] > 0).cpu().numpy()
            c = cols[j % len(cols)] if r["reliable"][i] else (255, 40, 40)
            ov[sil] = (0.45 * ov[sil] + 0.55 * np.array(c[::-1])).astype(np.uint8)
            cnt, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnt, -1, c[::-1], 1)
            nm = key.split("/")[1]
            cv2.putText(ov, f"{nm} top {r['top'][i][2]*100:4.1f}cm {r['kind'][i][:5]} iou {r['iou'][i]:.2f} {'OK' if r['reliable'][i] else 'UNREL'}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c[::-1], 1, cv2.LINE_AA)
            y += 18
        cv2.putText(ov, f"{ep} k={k} (FoundationPose; red = unreliable)", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), ov)
    mp4 = ROOT / f"videos/generic/fp_qc/{ep}_fp_qc.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", str(tmp / "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", str(mp4)], check=True)
    for f in tmp.glob("*.jpg"):
        f.unlink()
    tmp.rmdir()
    print("wrote", mp4, flush=True)


def save(V, out, metas):
    p = GEN / V / "fp_tracks.npz"
    old, old_meta = {}, {}
    if p.exists():
        z = np.load(p, allow_pickle=True)
        old_meta = json.loads(str(z["meta"]))
        for key in old_meta:
            if key not in out:
                old[key] = {f.split("|", 1)[1]: z[f] for f in z.files if f.startswith(key + "|")}
    allo = {**old, **out}; allm = {**{k: v for k, v in old_meta.items() if k in old}, **metas}
    arrs = {f"{key}|{f}": v for key, r in allo.items() for f, v in r.items()}
    tmp = str(p) + ".tmp.npz"
    np.savez_compressed(tmp, meta=json.dumps(allm), **arrs)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*")
    ap.add_argument("--qc", nargs="*", default=[], help="episodes to render a QC overlay video for")
    a = ap.parse_args()
    T = {t["id"]: t for t in json.load(open(TASKS))["tasks"]}
    eps = a.episodes or [i for i, t in T.items() if t["task"] in ("fruit", "tools")]
    trk = Tracker()
    for ep in eps:
        t = T[ep]
        out, metas = run_episode(trk, t, qc=ep in a.qc)
        # several workers may write different episodes of the same video: lock the file per video
        import fcntl
        with open(GEN / t["video"] / "fp_tracks.lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            save(t["video"], out, metas)
        print(f"saved {ep}; GPU max alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
