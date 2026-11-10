"""Fast ingest: raw phone videos -> every perception input up to and including the 'vtrack' stage (tracks.pkl).
Generic: no object / task phrases anywhere (the VLM names the objects, SAM 3 validates the names, the agent phrase is 'hand').

  envs/vla/bin/python -m cleantables.perception.ingest <video files or a folder> [--only-stages frames,depth,objects,keyframes,geom,track,vtrack]
                                                  [--stride 3] [--cap 6] [--collapse-sim 0.75] [--batch NAME] [--depth-refine 1]

Stages (each skips what is already on disk), and where they run:
  frames     10 fps / 960x540 work frames: boxes.ingest.extract(), one process per video, parallel                   [CPU]
  depth      MoGe-3 ViT-L, one model load for all videos, one FOV per video, only every `stride`-th frame is run (the others
             copy the nearest computed frame; vtrack only reads the tracked frames), refine_steps 1 instead of 3      [GPU worker, vla env]
             -> data/twin_table/boxstack/<V>/depth10.npy (+ depth10.json, table_clean/_work/<V>/depth.npz)  same format as the box pipeline
  objects    preliminary table plane (RANSAC, no detections needed) -> table crop -> Qwen3-VL-4B lists objects on 3 crops
             (CPU bf16, runs while the GPU does the depth of the next videos) -> objects.merge over the whole folder (MiniLM 0.90,
             written to data/generic/_ingest/<batch>/ so an existing vocab_candidates.json is never overwritten) -> synonym collapse
             (MiniLM `--collapse-sim`, 0.75, + word-subset rule) to one phrase per object type, at most `--cap` per video
  keyframes  one SAM 3 image-model load for all videos; 12 keyframes per video x (the video's candidate phrases + 'hand'); the 3 verify
             frames are a subset of them                                                                         [GPU worker, SAM venv]
  geom       final table plane / table mask / bump check / per-frame depth scale from those detections (the plane logic of
             boxes.tracking.stage_geom, using boxes.geometry.fit_table / table_mask / depth_scale) -> det3d.npz (same keys as stage_geom,
             empty box rows); objects.select -> vocab.json; QC image data/generic/<V>/ingest_table_qc.jpg
  track      sam3_video_fast: one model load, one process for all videos, backbone cache shared by the phrases of a video -> sam3_video.npz
  vtrack     per-frame depth scale re-estimated from the tracked masks (hand + objects excluded at every tracked frame), then
             video_tracking.run -> tracks.pkl (CPU, parallel over videos)
Timings of every stage go to data/generic/_ingest/<batch>/timings.json. Internal: `--worker depth|image` are subprocess entry points.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cleantables.perception import common
from cleantables.perception.common import GEOM, OUT, ROOT, SAM_PATH, SAM_PY, VLA_PY, WORK, jdump, log, vdir, wait_gpu

STAGES = ["frames", "depth", "objects", "keyframes", "geom", "track", "vtrack"]
N_KEY = 12
MOGE_PP = f"{ROOT}/third_party/MoGe_latest:{ROOT}/cache/pylibs_moge3:{ROOT}/cache/pylibs:{ROOT}"
VIDEO_EXT = (".mov", ".mp4", ".m4v")


# ====================================================================================== small helpers
def meta_(v):
    return json.load(open(WORK / v / "frames_meta.json"))


def key_frames(v, stride, n_key=N_KEY):
    """n_key work frames, all multiples of `stride` (frames whose depth was computed), evenly spread over the video."""
    n = meta_(v)["n"]
    ks = sorted({int(round(x / stride)) * stride for x in np.linspace(2, n - 3, n_key)})
    return [k for k in ks if k < n]


def verify_subset(ks):
    return [ks[i] for i in np.linspace(1, len(ks) - 2, 3).astype(int)] if len(ks) >= 4 else ks


def tdir(batch):
    d = OUT / "_ingest" / batch
    d.mkdir(parents=True, exist_ok=True)
    return d


class Timings:
    def __init__(self, path):
        self.p = Path(path)
        self.d = json.load(open(self.p)) if self.p.exists() else {}

    def add(self, stage, key, secs):
        self.d.setdefault(stage, {})[key] = round(float(secs), 2)
        jdump(self.d, self.p)


# ====================================================================================== stage frames
def _extract(args):
    src, name = args
    import cv2
    cv2.setNumThreads(4)
    from cleantables.perception.boxes.ingest import extract
    t0 = time.time()
    m = extract(src, name)
    return name, m["n"], time.time() - t0


def stage_frames(vids, tm, workers=6):
    todo = [(str(p), n) for n, p in vids if not (WORK / n / "frames_meta.json").exists()]
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(min(workers, len(todo))) as ex:
            for n, nf, s in ex.map(_extract, todo):
                tm.add("frames", n, s)
    tm.add("frames", "_wall_total", time.time() - t0)


# ====================================================================================== depth worker (vla env, GPU)
def _worker_depth(a):
    import cv2
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from cleantables.perception.recon.depth_models import load
    from cleantables.perception.scene.common import read_frame
    mdl = load("moge3_vitl", "cuda")
    pool = ThreadPoolExecutor(4)
    for v in a.videos:
        out = GEOM / v / "depth10.npy"
        if out.exists():
            continue
        (GEOM / v).mkdir(parents=True, exist_ok=True)
        n = meta_(v)["n"]
        s = a.stride
        ks = list(range(0, n, s))
        t0 = time.time()
        sel = [ks[i] for i in np.linspace(0, len(ks) - 1, min(8, len(ks))).astype(int)]
        fovs = []
        for k in sel:                                    # one FOV per video: median of per-frame estimates (as MoGe.__call__)
            with torch.no_grad():
                o = mdl.m.infer(mdl._t(read_frame(v, k)), resolution_level=9, use_fp16=True, refine_steps=a.refine)
            kk0 = o["intrinsics"].float().cpu().numpy()
            fovs.append(float(np.degrees(2 * np.arctan(0.5 / kk0[0, 0]))))
        fov = float(np.median(fovs))
        h, w = 540, 960
        D = np.lib.format.open_memmap(str(out) + ".part.npy", mode="w+", dtype=np.float16, shape=(n, 270, 480))
        futs = [pool.submit(read_frame, v, k) for k in ks[:4]]
        for i, k in enumerate(ks):
            bgr = futs[i].result()
            if i + 4 < len(ks):
                futs.append(pool.submit(read_frame, v, ks[i + 4]))
            with torch.no_grad():
                o = mdl.m.infer(mdl._t(bgr), resolution_level=9, fov_x=fov, use_fp16=True, refine_steps=a.refine)
            d = o["depth"].float().cpu().numpy()
            m = o["mask"].cpu().numpy().astype(bool) & np.isfinite(d)
            dd = cv2.resize(np.where(m, d, 0).astype(np.float32), (480, 270), interpolation=cv2.INTER_NEAREST).astype(np.float16)
            for j in range(k, min(n, k + s)):            # frames between computed ones copy the computed frame at or before them
                D[j] = dd
            kk = o["intrinsics"].float().cpu().numpy()
        K = np.array([[kk[0, 0] * w, 0, kk[0, 2] * w], [0, kk[1, 1] * h, kk[1, 2] * h], [0, 0, 1.0]])
        D.flush()
        wall = time.time() - t0
        np.savez_compressed(WORK / v / "depth.npz", k=np.array(ks), depth=np.asarray(D[ks]), K=K, fov_x=fov, model="moge3_vitl",
                            wall_s=wall, peak_mib=torch.cuda.max_memory_allocated() / 2**20)
        jdump(dict(video=v, n=n, computed_every=s, refine_steps=a.refine, n_computed=len(ks), wall_s=round(wall, 1),
                   s_per_computed_frame=round(wall / len(ks), 3), fov_x=fov,
                   note="frames that were not run copy the nearest earlier computed frame (vtrack only reads tracked frames = multiples of the stride)"),
              GEOM / v / "depth10.json")
        del D
        Path(str(out) + ".part.npy").rename(out)
        print(f"{v} depth {len(ks)}/{n} frames {wall:.0f}s ({wall / len(ks) * 1000:.0f} ms/frame)", flush=True)


# ====================================================================================== table geometry (boxes.tracking.stage_geom logic, generic inputs)
def _K(v):
    return np.load(WORK / v / "depth.npz")["K"]


def _up(v):
    return (1.0, 0, 0) if meta_(v).get("rotated_portrait") else (0, -1.0, 0)


def _save_det3d(v, T_all, K, table, scale, info, prelim):
    GEOM.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GEOM / v / "det3d.npz", rows=np.zeros((0, 19)), cols=np.array(["k", "det", "prob", "x", "y", "top", "L", "W", "yaw", "ok", "touched", "n_top",
                                                                                      "zlo", "zhi", "xa", "ya", "colL", "cola", "colb"]),
                        tape=np.zeros((0, 8)), T=T_all[0], T_all=T_all, K=K, table=table, scale=scale, prelim=np.array(prelim),
                        generic_ingest=np.array(True))
    tb = dict(video=v, T_cam_to_table=T_all[0].tolist(), K=K.tolist(), prelim=prelim, **info)
    jdump(tb, GEOM / v / "table.json")


def prelim_geom(v, stride):
    """Table plane + mask from depth alone (RANSAC picks the biggest up-facing plane): only needed to crop the VLM listing to the table."""
    from cleantables.perception.boxes import geometry as G
    D10 = common.depth(v)
    K = _K(v)
    kf = key_frames(v, stride)
    deps = [np.asarray(D10[k], np.float32) for k in kf]
    T, n, c, info = G.fit_table(deps, K, None, up=_up(v))
    table = G.table_mask(deps, K, T, frac=0.12)
    nfr = meta_(v)["n"]
    _save_det3d(v, np.repeat(T[None], nfr, 0), K, table, np.ones(nfr, np.float32), info, True)


def final_geom(v, stride, det):
    """det: dict(frames [k], masks (m,270,480) bool, prob, frame (m,)) from the keyframe SAM 3 pass. Same plane / bump / table / scale logic as
    boxes.tracking.stage_geom (15 keyframes there, N_KEY here, windows of >= 2 frames)."""
    import cv2
    from cleantables.perception.boxes import geometry as G
    D10 = common.depth(v)
    K = _K(v)
    nfr = meta_(v)["n"]
    frames = [int(k) for k in det["frames"]]
    by_f = {}
    for i, k in enumerate(det["frame"]):
        if det["prob"][i] >= 0.3:
            by_f.setdefault(int(k), []).append(i)
    ker = np.ones((9, 9), np.uint8)

    def excl_of(k):
        e = np.zeros((270, 480), np.uint8)
        for i in by_f.get(k, []):
            e |= det["masks"][i].astype(np.uint8)
        return e

    deps = [np.asarray(D10[k], np.float32) for k in frames]
    excl = [cv2.dilate(excl_of(k), ker) > 0 for k in frames]
    up = _up(v)
    T, n, c, info = G.fit_table(deps, K, excl, up=up)
    table0 = G.table_mask(deps, K, T, frac=0.12)
    W_ = 6
    wins = np.array_split(np.array(frames), W_)
    wfit = []
    for wk in wins:
        ks_ = [int(x) for x in wk]
        if not ks_:
            wfit.append(None); continue
        ex_ = [(cv2.dilate(excl_of(k), ker) > 0) | ~table0 for k in ks_]
        try:
            Tw, *_ = G.fit_table([np.asarray(D10[k], np.float32) for k in ks_], K, ex_, up=up)
            wfit.append((Tw, ks_, ex_))
        except RuntimeError:
            wfit.append(None)
    seg_id, cur, bumps, wdiff = [], 0, [], []
    for wi in range(W_):
        if wi > 0 and wfit[wi] is not None and wfit[wi - 1] is not None:
            a, b = wfit[wi][0], wfit[wi - 1][0]
            ang = float(np.degrees(np.arccos(np.clip(a[2, :3] @ b[2, :3], -1, 1))))
            wdiff.append([round(ang, 2), round(abs(float(a[2, 3] - b[2, 3])) * 100, 2)])
            if ang > 2.5:
                cur += 1; bumps.append(int(wins[wi][0]))
        seg_id.append(cur)
    T_all = np.repeat(T[None], nfr, 0)
    if bumps:
        for sid in sorted(set(seg_id)):
            ws = [wi for wi in range(W_) if seg_id[wi] == sid and wfit[wi] is not None]
            ks_ = sum([wfit[wi][1] for wi in ws], []); ex_ = sum([wfit[wi][2] for wi in ws], [])
            Ts, *_ = G.fit_table([np.asarray(D10[k], np.float32) for k in ks_], K, ex_, up=up)
            lo = int(wins[min(wi for wi in range(W_) if seg_id[wi] == sid)][0])
            hi = nfr if sid == max(seg_id) else int(wins[min(wi for wi in range(W_) if seg_id[wi] == sid + 1)][0])
            if sid == 0:
                lo = 0
            T_all[lo:hi] = Ts
        T = T_all[0]
    info.update(bump_frames=bumps, window_plane_diff_deg_cm=wdiff, n_keyframes=len(frames))
    table = G.table_mask(deps, K, T, frac=0.12)
    zps = {}
    scale = np.ones(nfr, np.float32)
    for k in frames:
        key = T_all[k].tobytes()
        if key not in zps:
            zps[key] = G.plane_z_map(T_all[k], K)
        scale[k] = G.depth_scale(np.asarray(D10[k], np.float32), zps[key], table & ~(cv2.dilate(excl_of(k), ker) > 0))
    kf = np.array(frames)
    for k in range(nfr):
        if k not in by_f and k not in frames:
            kn = int(kf[np.argmin(np.abs(kf - k))])
            key = T_all[k].tobytes()
            if key not in zps:
                zps[key] = G.plane_z_map(T_all[k], K)
            scale[k] = G.depth_scale(np.asarray(D10[k], np.float32), zps[key], table & ~(cv2.dilate(excl_of(kn), ker) > 0))
    _save_det3d(v, T_all, K, table, scale, info, False)
    qc_image(v, frames[len(frames) // 2], info)
    return info


def qc_image(v, k, info):
    import cv2
    from cleantables.perception.common import frame, geom
    g = geom(v)
    im = frame(v, k)
    m = cv2.resize(g["table"].astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST) > 0
    ov = im.copy()
    ov[m] = (0.55 * ov[m] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 255), 2)
    # table-frame axes: origin = where the principal ray meets the table, x right, y forward, z up (project 20 cm axes)
    T, K = g["T_all"][k], g["K"]
    Tinv = np.linalg.inv(T)
    pts = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], float)
    P = (Tinv[:3, :3] @ pts.T).T + Tinv[:3, 3]
    uv = (K @ P.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    for j, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.arrowedLine(ov, tuple(int(x) for x in uv[0]), tuple(int(x) for x in uv[j]), col, 3)
    txt = f"{v} k={k} cam height {info.get('camera_height_m')} m tilt {info.get('tilt_deg')} deg bumps {info.get('bump_frames')} scale {g['scale'][k]:.3f}"
    cv2.putText(ov, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(vdir(v) / "ingest_table_qc.jpg"), ov)


def refine_scale(v):
    """Per-frame depth scale from the tracked masks (hand + objects, every tracked frame) instead of the nearest keyframe's detections."""
    import cv2
    from cleantables.perception.boxes import geometry as G
    z = np.load(vdir(v) / "sam3_video.npz")
    ks = z["frames"].astype(int)
    nt = len(ks)
    allm = np.zeros((nt, 270, 480), bool)
    for pi in range(len(z["phrases"])):
        s = z[f"p{pi}_shape"]
        if int(s[0]) == 0:
            continue
        mk = np.unpackbits(z[f"p{pi}_masks"])[:int(np.prod(s))].reshape(*s).astype(bool)
        allm |= (mk & (z[f"p{pi}_probs"][:, :, None, None] >= 0.3)).any(0)
    d = dict(np.load(GEOM / v / "det3d.npz", allow_pickle=True))
    if "refined_scale" in d:
        return
    D10 = common.depth(v)
    K, T_all, table = d["K"], d["T_all"], d["table"]
    nfr = len(d["scale"])
    scale = d["scale"].copy()
    zps = {}
    for t, k in enumerate(ks):
        key = T_all[k].tobytes()
        if key not in zps:
            zps[key] = G.plane_z_map(T_all[k], K)
        scale[k] = G.depth_scale(np.asarray(D10[k], np.float32), zps[key], table & ~(cv2.dilate(allm[t].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0))
    for k in range(nfr):
        if k not in set(ks.tolist()):
            scale[k] = scale[int(ks[np.argmin(np.abs(ks - k))])]
    d["scale"] = scale
    d["refined_scale"] = np.array(True)
    np.savez_compressed(GEOM / v / "det3d.npz", **d)


# ====================================================================================== SAM 3 image worker (SAM venv, GPU)
def _worker_image(a):
    import torch
    from cleantables.perception.scene import sam3_inv_worker as W
    jobs = json.load(open(a.jobs))
    t0 = time.time()
    proc = W.load_any("cuda")
    print(f"SAM 3 image model loaded in {time.time() - t0:.0f}s", flush=True)
    W.load_any = lambda dev: proc
    W.MIN_P = 0.3
    for v, j in jobs.items():
        if os.path.exists(j["out"]):
            continue
        t1 = time.time()
        W.run(v, j["frames"], j["out"], j["phrases"], "cuda")
        jdump(dict(wall_s=round(time.time() - t1, 1), n_frames=len(j["frames"]), n_phrases=len(j["phrases"]),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2)), j["out"] + ".json")


# ====================================================================================== phrase collapse
def collapse(vids, sim, cap, batch):
    """objects.merge (MiniLM 0.90, written under data/generic/_ingest/<batch>) -> one phrase per object type (lower similarity + word
    subset rule), at most `cap` per video, ranked by how many of the video's 3 listings name it (ties: phrase frequency over the folder)."""
    from cleantables.perception import objects
    objects.OUT = tdir(batch)                          # objects.merge writes vocab_candidates.json to its OUT; redirect it so existing files are kept
    vc = objects.merge(vids)
    objects.OUT = OUT
    rep = vc["merged_into"]
    cnt = vc["counts"]
    ph = list(vc["candidates"])
    E = common.embed(ph)
    S = E @ E.T
    order = sorted(range(len(ph)), key=lambda i: (-cnt[ph[i]], len(ph[i]), ph[i]))
    groups = []
    for i in order:
        wi = set(ph[i].split())
        g = next((g for g in groups if S[i, g[0]] >= sim or wi <= set(ph[g[0]].split()) or set(ph[g[0]].split()) <= wi), None)
        if g is None:
            groups.append([i])
        else:
            g.append(i)
    to_type = {ph[i]: ph[g[0]] for g in groups for i in g}
    per_video, listing_count = {}, {}
    for v in vids:
        sc = {}
        for r in objects.list_objects(v)["frames"]:
            for t in {to_type[rep[p]] for p in r["phrases"] if rep.get(p) in to_type}:
                sc[t] = sc.get(t, 0) + 1
        top = sorted(sc, key=lambda t: (-sc[t], -cnt.get(t, 0), t))[:cap]
        per_video[v] = top
        listing_count[v] = sc
        jdump(dict(phrases=top, listing_counts=sc), vdir(v) / "ingest_phrases.json")
    out = dict(types={ph[g[0]]: [ph[i] for i in g] for g in groups}, per_video=per_video, collapse_sim=sim, cap=cap)
    jdump(out, tdir(batch) / "collapsed_vocab.json")
    log(f"ingest vocab: {len(ph)} merged phrases -> {len(groups)} object types; per video: {per_video}")
    return per_video


# ====================================================================================== VLM on the GPU (optional)
def use_vlm_gpu():
    """Qwen3-VL-4B bf16 needs ~9 GB on the GPU (above the 6 GB budget of the other stages): only used when >= 11 GB are free right now.
    The CPU path (common.vlm_ask) can take ~90 s per call on a heavily loaded machine."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from cleantables.perception import objects
    proc = AutoProcessor.from_pretrained(common.VLM)
    model = AutoModelForImageTextToText.from_pretrained(common.VLM, dtype=torch.bfloat16).to("cuda").eval()

    def ask(images, question, max_new_tokens=48):
        msg = [{"role": "user", "content": [*[{"type": "image", "image": im} for im in images], {"type": "text", "text": question}]}]
        inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    objects.vlm_ask = ask
    return model


def free_vlm_gpu(model):
    import torch
    del model
    torch.cuda.empty_cache()


# ====================================================================================== subprocess helper
def run_worker(cmd, env_extra, logf, name):
    env = dict(os.environ, **env_extra)
    Path(logf).parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as f:
        f.write(f"\n# {time.ctime()} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{name} failed (rc {r.returncode}); see {logf}")


# ====================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--only-stages", default="")
    ap.add_argument("--stride", type=int, default=3, help="tracking stride in 10 fps work frames (3 -> 3.33 fps); depth is computed on the same frames")
    ap.add_argument("--cap", type=int, default=6, help="max candidate phrases per video (after synonym collapse), + 'hand'")
    ap.add_argument("--collapse-sim", type=float, default=0.75)
    ap.add_argument("--depth-refine", type=int, default=1, help="MoGe-3 sparse refinement steps (3 = original, 1 = 1.5x faster, ~0.1%% depth change)")
    ap.add_argument("--vlm-cpu", action="store_true", help="never put the VLM on the GPU (default: GPU when >= 11 GB are free)")
    ap.add_argument("--batch", default="")
    ap.add_argument("--worker", default="")
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--jobs", default="")
    ap.add_argument("--refine", type=int, default=1)
    a = ap.parse_args()
    if a.worker == "depth":
        a.videos, a.refine = a.videos, a.depth_refine
        return _worker_depth(a)
    if a.worker == "image":
        return _worker_image(a)

    common.LOG = ROOT / "runs/generic/ingest_PROGRESS.md"
    src = []
    for x in a.inputs:
        p = Path(x)
        src += sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]
    if not src:
        sys.exit("no videos found")
    vids = [(p.stem, p) for p in src]
    names = [n for n, _ in vids]
    batch = a.batch or time.strftime("%Y%m%d_%H%M%S")
    todo = a.only_stages.split(",") if a.only_stages else STAGES
    tm = Timings(tdir(batch) / "timings.json")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"ingest: {len(names)} videos {names}, stages {todo}, stride {a.stride}, batch {batch}")
    T0 = time.time()
    lg = ROOT / "runs/generic/ingest" / f"{batch}.log"

    def stage_time(st, t0):
        tm.add("stage_wall_s", st, time.time() - t0)
        log(f"ingest stage {st}: {time.time() - t0:.0f}s")

    if "frames" in todo:
        t0 = time.time()
        stage_frames(vids, tm)
        stage_time("frames", t0)

    # ---- depth (GPU worker) overlapped with objects/VLM listing (CPU)
    proc = None
    vlm_model = None
    t0 = time.time()
    if "depth" in todo:
        pend = [v for v in names if not (GEOM / v / "depth10.npy").exists()]
        if pend:
            wait_gpu(3200, "ingest depth")
            proc = subprocess.Popen([VLA_PY, "-m", "cleantables.perception.ingest", "--worker", "depth", "--stride", str(a.stride), "--depth-refine",
                                     str(a.depth_refine), "--videos", *pend], cwd=ROOT, env=dict(os.environ, PYTHONPATH=MOGE_PP),
                                    stdout=open(lg, "a"), stderr=subprocess.STDOUT)
    if "objects" in todo:
        t1 = time.time()
        if not a.vlm_cpu and any(not (vdir(v) / "vlm_objects.json").exists() for v in names) and common.gpu_free() >= 11000:
            vlm_model = use_vlm_gpu()
            log("ingest: VLM listing on the GPU")
        for v in names:
            while not (GEOM / v / "depth10.npy").exists():
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(f"depth worker exited (rc {proc.returncode}); see {lg}")
                if proc is None:
                    raise RuntimeError(f"{v}: no depth10.npy and the depth stage is not selected")
                time.sleep(2)
            tv = time.time()
            if not (vdir(v) / "vlm_objects.json").exists():
                if not (GEOM / v / "det3d.npz").exists():
                    prelim_geom(v, a.stride)
                from cleantables.perception.objects import list_objects
                list_objects(v)
            tm.add("objects_list", v, time.time() - tv)
    if "objects" in todo and vlm_model is not None:
        free_vlm_gpu(vlm_model)
    if proc is not None:
        proc.wait()
        if proc.returncode:
            raise RuntimeError(f"depth worker failed (rc {proc.returncode}); see {lg}")
    if "depth" in todo:
        for v in names:
            if (GEOM / v / "depth10.json").exists():
                tm.add("depth", v, json.load(open(GEOM / v / "depth10.json"))["wall_s"])
        stage_time("depth+objects_list(overlapped)", t0)
    if "objects" in todo:
        t1 = time.time()
        collapse(names, a.collapse_sim, a.cap, batch)
        stage_time("objects_merge", t1)

    # ---- keyframe SAM 3 image pass: one model load for all videos
    def phrases_of(v):
        return json.load(open(vdir(v) / "ingest_phrases.json"))["phrases"]

    if "keyframes" in todo:
        t0 = time.time()
        jobs = {}
        for v in names:
            out = vdir(v) / "ingest_keyframes.npz"
            if not out.exists():
                jobs[v] = dict(frames=key_frames(v, a.stride), phrases=phrases_of(v) + common.AGENT_PHRASES, out=str(out))
        if jobs:
            jp = tdir(batch) / "keyframe_jobs.json"
            jdump(jobs, jp)
            wait_gpu(5000, "ingest keyframes")
            run_worker([SAM_PY, "-m", "cleantables.perception.ingest", "--worker", "image", "--jobs", str(jp)],
                       dict(PYTHONPATH=f"{SAM_PATH}:{ROOT}", HF_HUB_OFFLINE="1"), lg, "keyframes")
            for v, j in jobs.items():
                if os.path.exists(j["out"] + ".json"):
                    tm.add("keyframes", v, json.load(open(j["out"] + ".json"))["wall_s"])
        stage_time("keyframes", t0)

    if "geom" in todo:
        t0 = time.time()
        from cleantables.perception import objects
        for v in names:
            tv = time.time()
            if (vdir(v) / "vocab.json").exists() and not np.load(GEOM / v / "det3d.npz")["prelim"]:
                continue
            d = objects.load_dets(vdir(v) / "ingest_keyframes.npz")
            info = final_geom(v, a.stride, dict(frames=d["frames"], masks=d["masks"], prob=d["prob"], frame=d["frame"]))
            # verify subset: the candidate phrases only (no 'hand'), 3 of the keyframes -> sam3_verify.npz, then objects.select
            cands = phrases_of(v)
            vf = set(verify_subset([int(k) for k in d["frames"]]))
            sel = [i for i in range(len(d["prob"])) if int(d["frame"][i]) in vf and d["prompts"][int(d["prompt"][i])] in cands]
            m = len(sel)
            np.savez_compressed(vdir(v) / "sam3_verify.npz", frame=d["frame"][sel], prompt=np.array([cands.index(d["prompts"][int(d["prompt"][i])]) for i in sel], np.int16),
                                prob=d["prob"][sel], box=d["box"][sel].reshape(m, 4),
                                masks=np.packbits(d["masks"][sel]) if m else np.zeros(0, np.uint8), shape=np.array([m, 270, 480]),
                                prompts=np.array(cands), frames_done=np.array(sorted(vf), np.int32))
            objects.select(v)
            tm.add("geom_select", v, time.time() - tv)
        stage_time("geom", t0)

    # ---- SAM 3 video tracking: one process / one model load for all videos
    if "track" in todo:
        t0 = time.time()
        jobs = {v: json.load(open(vdir(v) / "vocab.json"))["phrases"] + common.AGENT_PHRASES for v in names if not (vdir(v) / "sam3_video.npz").exists()}
        if jobs:
            jp = tdir(batch) / "video_jobs.json"
            jdump(jobs, jp)
            wait_gpu(6800, "ingest SAM 3 video")
            run_worker([SAM_PY, "-m", "cleantables.perception.sam3_video_fast", "--jobs", str(jp), "--stride", str(a.stride)],
                       dict(PYTHONPATH=f"{SAM_PATH}:{ROOT}", HF_HUB_OFFLINE="1"), lg, "track")
            for v in jobs:
                tm.add("track", v, float(np.load(vdir(v) / "sam3_video.npz")["wall_s"]))
        stage_time("track", t0)

    if "vtrack" in todo:
        t0 = time.time()
        import multiprocessing as mp
        with ProcessPoolExecutor(min(4, len(names)), mp_context=mp.get_context("spawn")) as ex:
            for v, s in ex.map(_vtrack, names):
                tm.add("vtrack", v, s)
        stage_time("vtrack", t0)
    log(f"ingest done in {time.time() - T0:.0f}s; timings {tdir(batch) / 'timings.json'}")


def _vtrack(v):
    t0 = time.time()
    common.LOG = ROOT / "runs/generic/ingest_PROGRESS.md"
    from cleantables.perception.video_tracking import run
    refine_scale(v)
    run(v)
    return v, time.time() - t0


if __name__ == "__main__":
    main()
