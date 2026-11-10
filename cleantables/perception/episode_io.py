"""Shared paths and loaders for the per-episode perception stages (6D poses, hands, contact).

Inputs (read only; outputs of the earlier video stages):
  frames    data/table_clean/_work/<V>/frames/<k>.jpg          10 fps, 960x540
  depth     data/twin_table/boxstack/<V>/depth10.npy           MoGe-3 ViT-L, camera z (m), 270x480, computed every 3rd frame
  geometry  data/twin_table/boxstack/<V>/det3d.npz             K, T cam->table, per-frame table-plane depth scale, table mask
  plate     data/twin_table/ep/_video/<V>/plate_depth.npy      static background depth (person-free), 270x480
  masks     data/generic/<V>/{sam3_vinv.npz, hand_video.npz}   SAM 3 object / 'hand' masks (every 3rd frame), 270x480
  tracks    data/generic/<V>/tracks.pkl                        3D boxes per sighting (table frame)
  twins     data/twin_table/ep/<ep>/{scene.json, usd/<obj>.obj} Hunyuan / library meshes, start poses
  hands     data/table_clean/_work/<V>/hand_wilor.npz or OUT/hands/<ep>/hand_wilor.npz (computed per episode)
All outputs go under OUT = $CLEANTABLES_DATA/perception_v2/.
"""
from __future__ import annotations

from cleantables import CT_DATA
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/table_clean/_work"
GEOM = ROOT / "data/twin_table/boxstack"
GEN = ROOT / "data/generic"
EPD = ROOT / "data/twin_table/ep"
OUT = Path(CT_DATA + "/perception_v2")
TMP = Path(CT_DATA + "/tmp/perc")
LEDGER = Path(CT_DATA + "/logs/gpu_jobs.log")
H, W = 270, 480
FPS = 10.0
MARGIN = 20          # frames (2 s) of context before / after each episode window

# Three trial episodes used for the per-stage checks
TRIAL = ["IMG_2516_t11", "IMG_2536_t01", "IMG_2550_t01"]


def env_dirs():
    """Put caches and temp files on the data disk. Model weights already cached are read in place."""
    for k, sub in (("HF_HOME", "hf"), ("TORCH_HOME", "torch"), ("TMPDIR", "tmpdir")):
        os.environ[k] = str(TMP / sub)
        (TMP / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "cache/hf/hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def episode(ep):
    E = {e["id"]: e for e in json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]}
    if ep not in E:          # other Gemini episodes and the between-episode stretches (data/labels/episodes_all.json 'segments')
        E.update({s["id"]: s for s in json.load(open(ROOT / "data/labels/episodes_all.json"))["segments"] if s["id"] not in E})
    e = dict(E[ep])
    n = n_frames(e["video"])
    e["k0"], e["k1"] = int(round(e["start_s"] * FPS)), min(int(round(e["end_s"] * FPS)), n - 1)
    e["w0"], e["w1"] = max(0, e["k0"] - MARGIN), min(n - 1, e["k1"] + MARGIN)
    return e


def n_frames(v):
    return json.load(open(WORK / v / "frames_meta.json"))["n"]


def frame(v, k):
    import cv2
    return cv2.imread(str(WORK / v / "frames" / f"{int(k)}.jpg"))


def geom(v):
    z = np.load(GEOM / v / "det3d.npz")
    K = z["K"].astype(np.float64)
    K_lo = K.copy()
    K_lo[0, 0] /= 2; K_lo[1, 1] /= 2; K_lo[0, 2] = (K[0, 2] - 0.5) / 2; K_lo[1, 2] = (K[1, 2] - 0.5) / 2
    n = n_frames(v)
    T_all = z["T_all"] if "T_all" in z.files else np.repeat(z["T"][None], n, 0)
    return dict(K=K, K_lo=K_lo, T=z["T"].astype(np.float64), T_all=T_all, scale=z["scale"].astype(np.float32), table=z["table"].astype(bool))


def depth_raw(v):
    """Stored MoGe-3 depth (unscaled camera z; computed every 3rd frame, the others are copies)."""
    return np.load(GEOM / v / "depth10.npy", mmap_mode="r")


def depth_stride(v):
    """1 for the box-pipeline videos (MoGe-3 on every frame, no depth10.json), else computed_every (3)."""
    p = GEOM / v / "depth10.json"
    return int(json.load(open(p)).get("computed_every", 3)) if p.exists() else 1


def fov_x(v):
    p = GEOM / v / "depth10.json"
    if p.exists():
        return float(json.load(open(p))["fov_x"])
    K = geom(v)["K"]
    return float(np.degrees(2 * np.arctan(0.5 * 960 / K[0, 0])))


def depth_full(ep):
    """Full-rate MoGe-3 depth (raw MoGe units) over the episode window -> (ks (n,), D (n,270,480) float32).
    Box-pipeline videos already have it in depth10.npy; for stride-3 videos it is recomputed by gpu_jobs.depth."""
    e = episode(ep)
    ks = np.arange(e["w0"], e["w1"] + 1)
    if depth_stride(e["video"]) == 1:
        return ks, np.asarray(depth_raw(e["video"])[ks], np.float32)
    if not (OUT / "depth" / ep / "moge3_full.npz").exists():
        # No full-rate recompute: use the stored stride depth. Its computed frames are the multiples of the stride, which are exactly
        # the frames SAM 3 tracked (video_tracking), so every frame with an object mask has computed (not copied) depth.
        return ks, np.asarray(depth_raw(e["video"])[ks], np.float32)
    z = np.load(OUT / "depth" / ep / "moge3_full.npz")
    assert np.array_equal(z["k"], ks)
    return ks, z["depth"].astype(np.float32)


def plate(v):
    import cv2
    d = EPD / "_video" / v
    D = np.load(d / "plate_depth.npy").astype(np.float32)
    valid = cv2.resize(cv2.imread(str(d / "plate_valid.png"), 0), (W, H), interpolation=cv2.INTER_NEAREST) > 127
    return D, valid


def _unpack(bits, shape):
    return np.unpackbits(bits)[:int(np.prod(shape))].reshape(shape).astype(bool)


def sam3_masks(v):
    """-> (M (n_det,270,480) bool, det frame (n_det,), det prompt (n_det,), prompts, hand {k: (270,480) bool})."""
    z = np.load(GEN / v / "sam3_vinv.npz")
    M = _unpack(z["masks"], tuple(z["shape"]))
    hz = np.load(GEN / v / "hand_video.npz")
    Hm = _unpack(hz["masks"], tuple(hz["shape"]))
    hand = {int(k): Hm[t] for t, k in enumerate(hz["frames"])}
    return M, z["frame"].astype(int), z["prompt"].astype(int), [str(p) for p in z["prompts"]], hand


def tracks(v):
    return pickle.load(open(GEN / v / "tracks.pkl", "rb"))


# PERC_TWIN_VARIANT=<name>: read scene.json + meshes from data/twin_table/ep/_<name>/<ep>/ (a twin rebuilt into its own folder,
# e.g. twin.ep.build with TWIN_SCALE_MODE=per_object) and write pose outputs with the suffix SUF; default = data/twin_table/ep/.
VARIANT = os.environ.get("PERC_TWIN_VARIANT", "")
SUF = f"_{VARIANT}" if VARIANT else ""
TWIN = EPD / f"_{VARIANT}" if VARIANT else EPD


def scene(ep):
    return json.load(open(TWIN / ep / "scene.json"))


def mesh_path(ep, oid):
    return TWIN / ep / "usd" / f"{oid}.obj"


def to_h(T):
    return np.asarray(T, np.float64)


def backproject(D, K_lo):
    """(270,480) depth -> (270,480,3) camera points (OpenCV)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([(xs - K_lo[0, 2]) * D / K_lo[0, 0], (ys - K_lo[1, 2]) * D / K_lo[1, 1], D], -1)


def jdump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def ledger(line):
    with open(LEDGER, "a") as f:
        f.write(line.rstrip() + "\n")
