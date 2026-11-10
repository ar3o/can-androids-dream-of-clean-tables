"""Hunyuan3D-2.1 inputs: per library object the best unoccluded SAM 3 sightings (largest complete, isolated, sharp mask; library.pick),
re-cut at the full 1920x1080 resolution of the original video (decoded with the vla env's OpenCV; the 10 fps work frames are
960x540), background removed (SAM 3 mask upsampled + GrabCut at full res), padded to a square RGBA crop.

  stage 1 (sim env):  $SIM_PY -m cleantables.twin.ep.mesh_inputs select     -> data/twin_table/ep/hy3d/inputs/sightings.json + masks
  stage 2 (vla env):  envs/vla/bin/python -m cleantables.twin.ep.mesh_inputs crop   -> data/twin_table/ep/hy3d/inputs/<key>/<i>_<V>_k<k>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "data/twin_table/ep/hy3d/inputs"
RAW = {"IMG_2516": ROOT / "data/raw_clips/table_clean/IMG_2516.MOV"}
KEYS_TOOLS = ["pliers_a", "pliers_b", "wrench", "utility_knife", "screwdriver", "toolbox", "toolbox_lid"]
KEYS_BOXES = ["amazon_box", "cipriata_box", "lotus_box", "black_box", "orange_box", "yellow_box"]
KEYS_FRUIT = ["green_apple", "ginger"]          # bananas and bowl of IMG_2516 are meshed separately (Hunyuan3D-2.1 shape + paint)
N_VIEWS = 3


def raw_path(v):
    return RAW.get(v, ROOT / "data/raw/boxes" / f"{v}.MOV")


def select():
    from cleantables.twin.ep import library as L
    OUTD.mkdir(parents=True, exist_ok=True)
    cache, out = {}, {}
    vt, vb = L.TASK_VIDEOS["tools"], sorted(set(L.TASK_VIDEOS["tools"] + L.TASK_VIDEOS["stack_boxes"]))
    plan = [(k, vt, {}) for k in KEYS_TOOLS] + [(k, vb, dict(first_rest=True, max_top=0.2)) for k in KEYS_BOXES] + \
           [(k, ["IMG_2516"], {}) for k in KEYS_FRUIT]
    for key, vids, kw in plan:
        if key in ("pliers_a", "pliers_b"):
            continue
        kf = (lambda k: k == "white_box") if key == "toolbox_lid" else (lambda k, key=key: k == key)
        cands = L.candidates(vids, kf, **kw)
        if key == "toolbox_lid":            # the lid = a white box seen while the open tray is elsewhere (library.build_toolbox rule)
            trays = {(x["v"], x["k"]) for x in L.candidates(vids, lambda k: k == "toolbox")}
            cands = [c for c in L.candidates(vids, kf, max_top=0.05) if (c["v"], c["k"]) in trays]
        s = L.pick(cands, 12, cache)
        if not s:
            s = L.pick(L.candidates(vids, kf, max_top=0.2), 12, cache)
        # several views: prefer different videos / tracks (different viewing angles), best first
        chosen, seen = [], set()
        for c in s:
            if (c["v"], c["track"]) in seen:
                continue
            chosen.append(c); seen.add((c["v"], c["track"]))
            if len(chosen) >= N_VIEWS:
                break
        out[key] = [dict(v=c["v"], k=c["k"], det=c["det"], track=c["track"], area=c["area"]) for c in chosen]
        for i, c in enumerate(chosen):
            np.save(OUTD / f"mask_{key}_{i}.npy", c["mask"])
    # pliers: the identities of library.py (shape-assigned per episode); their sources = library meta, plus more views of 'pliers' tracks
    for key in ("pliers_a", "pliers_b"):
        m = json.load(open(L.LIB / key / "meta.json"))
        v, k, tr, det = m["source"].split()[0], int(m["source"].split("k=")[1].split()[0]), int(m["source"].split("track=")[1].split()[0]), \
            int(m["source"].split("det=")[1])
        D = L.Dets(v)
        out[key] = [dict(v=v, k=k, det=det, track=tr, area=0)]
        np.save(OUTD / f"mask_{key}_0.npy", D.mask(det))
    json.dump(out, open(OUTD / "sightings.json", "w"), indent=1)
    print({k: [(x["v"], x["k"]) for x in v] for k, v in out.items()})


def crop():
    import cv2
    S = json.load(open(OUTD / "sightings.json"))
    caps = {}
    for key, lst in S.items():
        (OUTD / key).mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(lst):
            v, k = c["v"], c["k"]
            meta = json.load(open(ROOT / "data/table_clean/_work" / v / "frames_meta.json"))
            stride = int(meta.get("stride", 3))
            if v not in caps:
                caps[v] = cv2.VideoCapture(str(raw_path(v)))
            cap = caps[v]
            cap.set(cv2.CAP_PROP_POS_FRAMES, k * stride)
            ok, img = cap.read()
            if not ok:
                print("decode failed", key, v, k, flush=True)
                continue
            if img.shape[0] > img.shape[1] and not meta.get("rotated_portrait", False):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            Hf, Wf = img.shape[:2]
            m = np.load(OUTD / f"mask_{key}_{i}.npy")
            mu = cv2.resize(m.astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(mu)
            pad = int(0.25 * max(np.ptp(xs), np.ptp(ys))) + 10
            x0, x1, y0, y1 = max(0, xs.min() - pad), min(Wf, xs.max() + pad + 1), max(0, ys.min() - pad), min(Hf, ys.max() + pad + 1)
            cr = img[y0:y1, x0:x1].copy(); mc = mu[y0:y1, x0:x1]
            gc = np.full(mc.shape, cv2.GC_BGD, np.uint8)
            ring = cv2.dilate(mc, np.ones((25, 25), np.uint8)).astype(bool)
            gc[ring] = cv2.GC_PR_BGD
            gc[mc.astype(bool)] = cv2.GC_PR_FGD
            gc[cv2.erode(mc, np.ones((9, 9), np.uint8)).astype(bool)] = cv2.GC_FGD
            bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
            cv2.grabCut(cr, gc, None, bg, fg, 6, cv2.GC_INIT_WITH_MASK)
            a = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(a)
            if n > 1:
                a = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
            a = cv2.GaussianBlur(a.astype(np.float32), (3, 3), 0)
            rgba = np.dstack([cr, (a * 255).astype(np.uint8)])
            h, w = rgba.shape[:2]
            s = int(max(h, w) * 1.2)
            sq = np.zeros((s, s, 4), np.uint8)
            sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = rgba
            if s < 512:
                sq = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_CUBIC)
            p = OUTD / key / f"{i}_{v}_k{k}.png"
            cv2.imwrite(str(p), sq)
            c["png"] = str(p.relative_to(ROOT)); c["crop_px"] = [int(w), int(h)]
            print(key, i, p.name, (w, h), flush=True)
    json.dump(S, open(OUTD / "sightings.json", "w"), indent=1)




def rect():
    """Flat tools: the same sightings rectified to a top-down view of the table plane (orthophoto at 0.6 mm/px from the full-res frame,
    K of det3d scaled to 1920x1080), background removed by full-res GrabCut before warping. A top-down 'product shot' is the view in
    which Hunyuan reconstructs a lying tool's outline; oblique crops give webbing / flaps between thin parts."""
    import cv2
    S = json.load(open(OUTD / "sightings.json"))
    caps = {}
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pliers_a", "pliers_b", "wrench", "screwdriver", "utility_knife"]
    for key in keys:
        for i, c in enumerate(S[key][:2]):
            v, k = c["v"], c["k"]
            meta = json.load(open(ROOT / "data/table_clean/_work" / v / "frames_meta.json"))
            if v not in caps:
                caps[v] = cv2.VideoCapture(str(raw_path(v)))
            caps[v].set(cv2.CAP_PROP_POS_FRAMES, k * int(meta.get("stride", 3)))
            ok, img = caps[v].read()
            if not ok:
                continue
            Hf, Wf = img.shape[:2]
            z = np.load(ROOT / "data/twin_table/boxstack" / v / "det3d.npz")
            K = z["K"].copy(); K[:2] *= Wf / 960.0
            T = z["T"]
            m = np.load(OUTD / f"mask_{key}_{i}.npy")
            mu = cv2.resize(m.astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
            gc = np.full(mu.shape, cv2.GC_BGD, np.uint8)
            gc[cv2.dilate(mu, np.ones((25, 25), np.uint8)).astype(bool)] = cv2.GC_PR_BGD
            gc[mu.astype(bool)] = cv2.GC_PR_FGD
            gc[cv2.erode(mu, np.ones((9, 9), np.uint8)).astype(bool)] = cv2.GC_FGD
            ys, xs = np.nonzero(cv2.dilate(mu, np.ones((61, 61), np.uint8)))
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sub = gc[y0:y1, x0:x1].copy()
            cv2.grabCut(img[y0:y1, x0:x1], sub, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
            a = np.zeros(mu.shape, np.uint8)
            a[y0:y1, x0:x1] = ((sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)).astype(np.uint8) * 255
            pp = ROOT / "data/twin_table/ep/_video" / v / "plate_raw.png"
            pp = pp if pp.exists() else ROOT / "data/twin_table/ep/_video" / v / "plate.png"
            if pp.exists():          # table seen through the tool (between pliers handles): same colour as the static plate -> background
                pl = cv2.resize(cv2.imread(str(pp)), (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                same = np.abs(img.astype(np.int16) - pl.astype(np.int16)).max(2) < 22
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # the pine table seen between the handles but in the tool's shadow (differs from the plate): orange hue, saturated, not dark
                wood = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 70) & (hsv[..., 2] > 60)
                if not key.startswith("pliers"):          # wood key only for the pliers (screwdriver handle / knife are orange-ish)
                    wood[:] = False
                a[(same | wood) & (a > 0)] = 0
                a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                n_, lab_, st_, _ = cv2.connectedComponentsWithStats((a > 0).astype(np.uint8))
                if n_ > 1:                                 # keep every sizeable part (a handle can be cut off by the key)
                    big = 1 + np.nonzero(st_[1:, cv2.CC_STAT_AREA] >= 0.05 * st_[1:, cv2.CC_STAT_AREA].max())[0]
                    a = np.where(np.isin(lab_, big), a, 0).astype(np.uint8)
            # table-plane points of the mask -> ortho grid
            px = np.c_[xs, ys].astype(float)
            rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T
            d = rays @ T[:3, :3].T; o = T[:3, 3]; zp = 0.008
            P = o[None] + ((zp - o[2]) / d[:, 2])[:, None] * d
            lo, hi = P[:, :2].min(0) - 0.01, P[:, :2].max(0) + 0.01
            res = 0.0006
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0], res), np.arange(hi[1], lo[1], -res))
            G = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, zp)]
            Ti = np.linalg.inv(T)
            pc = G @ Ti[:3, :3].T + Ti[:3, 3]
            uv = pc @ K.T
            uv = uv[:, :2] / uv[:, 2:3]
            mx, my = uv[:, 0].reshape(gx.shape).astype(np.float32), uv[:, 1].reshape(gx.shape).astype(np.float32)
            rgb = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            al = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            n, lab, st, _ = cv2.connectedComponentsWithStats((al > 127).astype(np.uint8))
            if n > 1:
                big = 1 + np.nonzero(st[1:, cv2.CC_STAT_AREA] >= 0.05 * st[1:, cv2.CC_STAT_AREA].max())[0]
                al = np.where(np.isin(lab, big), al, 0).astype(np.uint8)
            rgba = np.dstack([rgb, al])
            ys2, xs2 = np.nonzero(al > 127)
            rgba = rgba[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
            h, w = rgba.shape[:2]
            s_ = int(max(h, w) * 1.2)
            sq = np.zeros((s_, s_, 4), np.uint8)
            sq[(s_ - h) // 2:(s_ - h) // 2 + h, (s_ - w) // 2:(s_ - w) // 2 + w] = rgba
            sq = cv2.resize(sq, (768, 768), interpolation=cv2.INTER_CUBIC)
            p = OUTD / key / f"rect{i}_{v}_k{k}.png"
            cv2.imwrite(str(p), sq)
            print(key, p.name, (w, h), f"{w * res:.3f} x {h * res:.3f} m", flush=True)




def select_more(key, n=4, videos=("IMG_2516",)):
    """Extra sightings of one object (different tracks = different apples / angles), appended to sightings.json."""
    from cleantables.twin.ep import library as L
    S = json.load(open(OUTD / "sightings.json"))
    have = {(c["v"], c["k"]) for c in S.get(key, [])}
    s = L.pick(L.candidates(list(videos), lambda k: k == key), 40, {})
    seen = {(c["v"], c["track"]) for c in S.get(key, [])}
    for c in s:
        if (c["v"], c["track"]) in seen or (c["v"], c["k"]) in have:
            continue
        i = len(S[key])
        np.save(OUTD / f"mask_{key}_{i}.npy", c["mask"])
        S[key].append(dict(v=c["v"], k=c["k"], det=c["det"], track=c["track"], area=c["area"]))
        seen.add((c["v"], c["track"]))
        if len(S[key]) >= n:
            break
    json.dump(S, open(OUTD / "sightings.json", "w"), indent=1)
    print(key, [(c["v"], c["k"]) for c in S[key]])


if __name__ == "__main__":
    if sys.argv[1] == "more":
        select_more(sys.argv[2], int(sys.argv[3]))
    else:
        {"select": select, "crop": crop, "rect": rect}[sys.argv[1]]()
