"""Real-to-sim twins against the real start frame: silhouette IoU and centre error per object (CPU only, read-only on the inputs).

Rows (same projection + scoring code for all):
  moge      the tracked MoGe box det_box (x, y, top, L, W, yaw; bottom on the table), median over sightings in [k0, k0+15]
  oct       the original builds data/twin_table/ep/<ep>
  inuse     data/twin_table/ep/_scalefix/<ep>, except the 10 re-linked toolbox twins -> data/twin_table/ep/_relink8/<ep>
  scalefix  plain _scalefix (sanity check only: the GPU renderer check scored this variant)

Scoring follows the GPU (nvdiffrast) twin check: mesh = usd/<id>.obj + pos; SAM 3 mask = the det of the same tracker id at the
sighting nearest the keyframe; hand pixels = SAM 3 'hand' mask at that frame dilated 9x9, excluded from both masks; 270x480 with
K_lo (f/2, (c-0.5)/2). The rasteriser is CPU: every projected triangle is filled with cv2.fillPoly (one call per triangle: one call
with all triangles would fill even-odd), and objects occlude each other in painter's order (far-to-near by the camera depth of the
mesh centroid). 'full' = no inter-object occlusion (as in the GPU check), 'vis' = painter's order.

  CUDA_VISIBLE_DEVICES="" $SIM_PY -m cleantables.twin.evaluate [EP...]   (writes outputs/eval/twins/per_ep/<ep>.json)
"""
from __future__ import annotations

from cleantables import CT_DATA
import json
import math
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
EPD = ROOT / "data/twin_table/ep"
GEN = ROOT / "data/generic"
WORK = ROOT / "data/table_clean/_work"
OUTD = ROOT / "outputs/eval/twins" / os.environ.get("TWINS_SUB", "")
GPU_CHECK = Path(CT_DATA + "/perception_v2/twins_v2")
H, W = 270, 480
WIN = 15
RELINK = {f"IMG_{v}_{t}" for v in (2534, 2535, 2539, 2542, 2548) for t in ("t01", "t02")}
KER = np.ones((9, 9), np.uint8)
# cv2.fillPoly includes pixels whose centre is up to ~0.5 px outside an edge (it rounds both span ends), nvdiffrast only pixels whose
# centre is inside. Fill at SS x the mask resolution and sample each mask pixel's centre: the bias drops to 0.5 / SS px (TWINS_SS=1: plain).
SS = int(os.environ.get("TWINS_SS", "5"))
OFF = (SS - 1) / 2


def hi(uv):
    """Mask-pixel coordinates (pixel centre = integer) -> SS x coordinates whose pixel centres include every mask-pixel centre."""
    return uv * SS + OFF


def lo(img):
    o = (SS - 1) // 2
    return img[o::SS, o::SS].astype(bool)


def _unpack(bits, shape):                                   # = cleantables.perception.episode_io._unpack
    return np.unpackbits(bits)[:int(np.prod(shape))].reshape(shape).astype(bool)


def sam3_masks(v):                                          # = common.sam3_masks (masks + hand dict)
    z = np.load(GEN / v / "sam3_vinv.npz")
    M = _unpack(z["masks"], tuple(z["shape"]))
    hz = np.load(GEN / v / "hand_video.npz")
    Hm = _unpack(hz["masks"], tuple(hz["shape"]))
    hand = {int(k): Hm[t] for t, k in enumerate(hz["frames"])}
    return M, hand


def k_lo(K):
    K = np.asarray(K, np.float64)
    L = K.copy()
    L[0, 0] /= 2; L[1, 1] /= 2; L[0, 2] = (K[0, 2] - 0.5) / 2; L[1, 2] = (K[1, 2] - 0.5) / 2
    return L


class Cam:
    def __init__(self, K, T_cam_to_table):
        self.K = k_lo(K)
        self.Ttc = np.linalg.inv(np.asarray(T_cam_to_table, np.float64))

    def cam(self, P):
        return P @ self.Ttc[:3, :3].T + self.Ttc[:3, 3]

    def uvz(self, P):
        Pc = self.cam(P)
        z = np.maximum(Pc[:, 2], 1e-3)
        u = self.K[0, 0] * Pc[:, 0] / z + self.K[0, 2]
        v = self.K[1, 1] * Pc[:, 1] / z + self.K[1, 2]
        return np.stack([u, v], -1), Pc[:, 2]

    def sil(self, V, F):
        """Filled projection of every triangle (pixel (i, j) centre = (u, v) = (i, j), as the nvdiffrast renderer)."""
        uv, z = self.uvz(V)
        img = np.zeros((H * SS, W * SS), np.uint8)
        tri = uv[F]
        ok = (z[F] > 0.05).all(1)                           # near plane 0.05 m, as the GPU renderer
        tri = tri[ok]
        # drop triangles entirely outside the image (speed only)
        inside = ~((tri[:, :, 0].max(1) < -1) | (tri[:, :, 0].min(1) > W) | (tri[:, :, 1].max(1) < -1) | (tri[:, :, 1].min(1) > H))
        tri = np.round(hi(tri[inside]) * 16).astype(np.int32)
        for t in tri:
            cv2.fillPoly(img, [t], 1, lineType=cv2.LINE_8, shift=4)
        return lo(img)


def box_corners(b):
    """MoGe box (x, y, top, L, W, yaw), bottom on the table -> 8 corners (table frame)."""
    h = max(b["top"], 0.005)                                # minimum box height 5 mm
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            dx, dy = sx * b["L"] / 2, sy * b["W"] / 2
            for z in (0.0, h):
                out.append([b["x"] + c * dx - s * dy, b["y"] + s * dx + c * dy, z])
    return np.array(out)


def box_sil(cam, b):
    uv, z = cam.uvz(box_corners(b))
    img = np.zeros((H * SS, W * SS), np.uint8)
    hull = cv2.convexHull(np.round(hi(uv) * 16).astype(np.int32))
    cv2.fillPoly(img, [hull], 1, lineType=cv2.LINE_8, shift=4)
    return lo(img)


def moge_box(R, track, k0):
    """Median tracked box over the sightings in [k0, k0+WIN]; else (the builder's 'hidden at the start' rule) the 3 nearest within 50 frames."""
    ro = R["objects"][track]
    ks = [int(k) for k in ro["sight_k"]]
    sel = [i for i, k in enumerate(ks) if k0 <= k <= k0 + WIN and R["det_box"].get(ro["det"][i])]
    src = "window"
    if not sel:
        near = sorted(range(len(ks)), key=lambda i: abs(ks[i] - k0))
        sel = [i for i in near if R["det_box"].get(ro["det"][i]) and abs(ks[i] - k0) <= 50][:3]
        src = "nearest3_within50"
    if not sel:
        return None
    B = [R["det_box"][ro["det"][i]] for i in sel]
    ref = B[0]["yaw"] % math.pi
    yaws = []
    for b in B:                                             # L >= W in every det_box (checked), so yaw has period pi
        y = b["yaw"] % math.pi
        y = ref + ((y - ref + math.pi / 2) % math.pi - math.pi / 2)
        yaws.append(y)
    med = lambda key: float(np.median([float(b[key]) for b in B]))  # noqa: E731
    return dict(x=med("x"), y=med("y"), top=med("top"), L=med("L"), W=med("W"), yaw=float(np.median(yaws)),
                n=len(sel), src=src, k=[ks[i] for i in sel])


def sam_for(R, M, hand, track, k0):
    """As the GPU twin check: sighting nearest k0 with a det; hand pixels at that frame dilated 9x9 -> keep."""
    ro = R["objects"][track]
    cand = [(abs(int(k) - k0), int(k), dd) for k, dd in zip(ro["sight_k"], ro["det"]) if dd is not None]
    if not cand:
        return None
    _, k, dd = min(cand)
    hk = hand.get(k)
    keep = ~cv2.dilate(hk.astype(np.uint8), KER).astype(bool) if hk is not None else np.ones((H, W), bool)
    return dict(k=k, det=int(dd), mask=M[dd], keep=keep, hand_px=int((~keep & M[dd]).sum()))


def centroid(m):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def score(sil_full, sil_vis, sam):
    keep = sam["keep"]
    bm = sam["mask"] & keep
    out = dict(sam_px=int(bm.sum()), sam_px_raw=int(sam["mask"].sum()))
    for tag, s in (("full", sil_full), ("vis", sil_vis)):
        a = s & keep
        u = (a | bm).sum()
        out[f"iou_{tag}"] = round(float((a & bm).sum() / u), 4) if u else None
        ca, cb = centroid(a), centroid(bm)
        # px at 960x540 (masks are 270x480: x2)
        out[f"cen_px_{tag}"] = round(2 * math.hypot(ca[0] - cb[0], ca[1] - cb[1]), 2) if (ca and cb) else None
        out[f"twin_px_{tag}"] = int(a.sum())
    return out


def footprint(V):
    """Oriented footprint of the posed mesh: min-area rectangle of the xy vertices -> (L >= W) m; top = max z (table frame)."""
    (_, _), (a, b), _ = cv2.minAreaRect(V[:, :2].astype(np.float32))
    return max(a, b), min(a, b), float(V[:, 2].max()), float(V[:, 2].min())


def load_twin(d):
    S = json.load(open(d / "scene.json"))
    objs = []
    for o in S["objects"]:
        m = trimesh.load(d / "usd" / f"{o['id']}.obj", force="mesh", process=False)
        m.apply_translation(o["pos"])
        objs.append(dict(o=o, V=np.asarray(m.vertices, np.float64), F=np.asarray(m.faces, np.int64), c=np.asarray(m.centroid)))
    return S, objs


def painter(cam, items):
    """Items: list of (key, full silhouette, centroid in table frame) -> {key: visible mask}; far first, nearer overwrite."""
    lab = np.full((H, W), -1, np.int32)
    depth = [float(cam.cam(c[None])[0, 2]) for _, _, c in items]
    for j in np.argsort(depth)[::-1]:
        lab[items[j][1]] = j
    return {items[j][0]: lab == j for j in range(len(items))}, {items[j][0]: depth[j] for j in range(len(items))}


def run(ep):
    t0 = __import__("time").time()
    S_sf = json.load(open(EPD / "_scalefix" / ep / "scene.json"))
    V = S_sf["video"]
    R = pickle.load(open(GEN / V / "tracks.pkl", "rb"))
    M, hand = sam3_masks(V)
    variants = {"oct": EPD / ep, "scalefix": EPD / "_scalefix" / ep,
                "inuse": (EPD / "_relink8" / ep) if ep in RELINK else (EPD / "_scalefix" / ep)}
    gpu = {}
    gp = GPU_CHECK / f"{ep}_check.json"
    if gp.exists():
        G = json.load(open(gp))["variants"]
        gpu = {"oct": {r["obj"]: r["iou"] for r in G.get("current", {}).get("objects", [])},
               "scalefix": {r["obj"]: r["iou"] for r in G.get("scalefix", {}).get("objects", [])}}
    rows, meta = [], dict(episode=ep, video=V, task=S_sf["task"], inuse_dir=str(variants["inuse"].relative_to(ROOT)))
    twins, moge_cache = {}, {}
    for name, d in variants.items():
        twins[name] = load_twin(d)
    S_use = twins["inuse"][0]
    k0 = int(S_use["keyframe"])
    meta.update(k0=k0, k0_oct=int(twins["oct"][0]["keyframe"]), k0_scalefix=int(twins["scalefix"][0]["keyframe"]))

    def mb(track):
        if track not in moge_cache:
            moge_cache[track] = moge_box(R, track, k0)
        return moge_cache[track]

    overlay = {}
    # ---- twin rows
    for name, (S, objs) in twins.items():
        cam = Cam(S["K"], S["T_cam_to_table"])
        kk0 = int(S["keyframe"])
        items = []
        for ob in objs:
            ob["sil"] = cam.sil(ob["V"], ob["F"])
            items.append((ob["o"]["id"], ob["sil"], ob["c"]))
        vis, dep = painter(cam, items)
        for ob in objs:
            o = ob["o"]
            tr = o.get("track")
            r = dict(episode=ep, row=name, obj=o["id"], key=o.get("key"), category=o.get("category"), track=tr, depth_m=round(dep[o["id"]], 3))
            L, Wd, top, zmin = footprint(ob["V"])
            r.update(twin_L=round(L, 4), twin_W=round(Wd, 4), twin_top=round(top, 4), twin_zmin=round(zmin, 4),
                     extents_field=o.get("extents"))
            if tr is None or tr < 0:
                r["status"] = "no_track"
            else:
                sam = sam_for(R, M, hand, tr, kk0)
                if sam is None:
                    r["status"] = "no_sighting"
                else:
                    r.update(sam_k=sam["k"], sam_dk=abs(sam["k"] - kk0), sam_det=sam["det"], hand_px_on_sam=sam["hand_px"])
                    r.update(score(ob["sil"], vis[o["id"]], sam))
                    r["status"] = "sam_empty" if r["sam_px"] == 0 else "ok"
                    if name == "inuse":
                        overlay[o["id"]] = (vis[o["id"]], sam)
                b = mb(tr) if name != "oct" or int(S["keyframe"]) == k0 else moge_box(R, tr, kk0)
                if b is not None:
                    r.update(moge_L=round(b["L"], 4), moge_W=round(b["W"], 4), moge_top=round(b["top"], 4),
                             dL_cm=round(100 * abs(L - b["L"]), 2), dW_cm=round(100 * abs(Wd - b["W"]), 2), dH_cm=round(100 * abs(top - b["top"]), 2))
            if name in gpu:
                r["gpu_iou"] = gpu[name].get(o["id"])
            rows.append(r)
    # ---- MoGe box row (the in-use twin's objects and track ids)
    cam = Cam(S_use["K"], S_use["T_cam_to_table"])
    items, boxes = [], {}
    for ob in twins["inuse"][1]:
        o = ob["o"]
        tr = o.get("track")
        b = mb(tr) if (tr is not None and tr >= 0) else None
        if b is None:
            continue
        boxes[o["id"]] = b
        items.append((o["id"], box_sil(cam, b), np.array([b["x"], b["y"], max(b["top"], 0.005) / 2])))
    vis, dep = painter(cam, items)
    full = {k: s for k, s, _ in items}
    for ob in twins["inuse"][1]:
        o = ob["o"]
        tr = o.get("track")
        r = dict(episode=ep, row="moge", obj=o["id"], key=o.get("key"), category=o.get("category"), track=tr)
        if tr is None or tr < 0:
            r["status"] = "no_track"
        elif o["id"] not in boxes:
            r["status"] = "no_moge_box"
        else:
            b = boxes[o["id"]]
            r.update(depth_m=round(dep[o["id"]], 3), moge_n=b["n"], moge_src=b["src"],
                     moge_L=round(b["L"], 4), moge_W=round(b["W"], 4), moge_top=round(b["top"], 4))
            sam = sam_for(R, M, hand, tr, k0)
            if sam is None:
                r["status"] = "no_sighting"
            else:
                r.update(sam_k=sam["k"], sam_dk=abs(sam["k"] - k0), sam_det=sam["det"], hand_px_on_sam=sam["hand_px"])
                r.update(score(full[o["id"]], vis[o["id"]], sam))
                r["status"] = "sam_empty" if r["sam_px"] == 0 else "ok"
                overlay.setdefault("__moge__" + o["id"], (vis[o["id"]], None))
        rows.append(r)
    # ---- overlay (in-use twin visible: yellow; SAM 3 mask: magenta; MoGe box visible: cyan; hand-excluded area: grey)
    img = cv2.imread(str(WORK / V / "frames" / f"{k0}.jpg"))
    if img is not None:
        up = lambda m: cv2.resize(m.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST)  # noqa: E731
        for key, (vm, sam) in overlay.items():
            col = (255, 255, 0) if key.startswith("__moge__") else (0, 255, 255)
            cs, _ = cv2.findContours(up(vm), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, col, 1 if key.startswith("__moge__") else 2)
            if sam is not None:
                cs, _ = cv2.findContours(up(sam["mask"] & sam["keep"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (255, 0, 255), 1)
        cv2.putText(img, f"{ep} k0={k0}  yellow: twin in use (visible)  magenta: SAM 3  cyan: MoGe box", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        (OUTD / "overlays").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTD / "overlays" / f"{ep}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    meta["seconds"] = round(__import__("time").time() - t0, 1)
    meta["supersample"] = SS
    out = dict(meta=meta, rows=rows)
    (OUTD / "per_ep").mkdir(parents=True, exist_ok=True)
    tmp = OUTD / "per_ep" / f"{ep}.json.tmp"
    json.dump(out, open(tmp, "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    os.replace(tmp, OUTD / "per_ep" / f"{ep}.json")
    ok = [r for r in rows if r.get("row") == "inuse" and r.get("iou_vis") is not None]
    print(ep, f"SS={SS}", f"{meta['seconds']}s", "inuse median iou vis", round(float(np.median([r['iou_vis'] for r in ok])), 3) if ok else None, flush=True)
    return ep


def safe(ep):
    try:
        return run(ep)
    except Exception as ex:                                 # keep going; report the failure
        import traceback
        traceback.print_exc()
        print(ep, "FAILED", repr(ex), flush=True)
        return None


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CPU only: set CUDA_VISIBLE_DEVICES=''"
    eps = sys.argv[1:] or sorted(os.listdir(EPD / "_scalefix"))
    OUTD.mkdir(parents=True, exist_ok=True)
    nproc = int(os.environ.get("TWINS_NPROC", "4"))
    if nproc <= 1 or len(eps) == 1:
        for e in eps:
            safe(e)
    else:
        with Pool(min(nproc, 4)) as p:
            done = list(p.imap_unordered(safe, eps))
        print("done", sum(d is not None for d in done), "of", len(eps))
