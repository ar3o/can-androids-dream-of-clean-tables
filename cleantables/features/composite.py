"""Features of the composite renders (cleantables.features.composite_render) in the world model's grid format.

Per layout and render variant: a static 32-frame clip (the still repeated; V-JEPA 2 ViT-L, layer 24, last tubelet, 448 x 256) and
DINOv2-L of the still, plain and horizontally flipped. Plus uv per object (coverage-weighted centroid, on the 16 x 28 grid, of the
object's projected convex hull, as at robot run time), in the layout's object order (the token slot order of the layouts file).

  $VLA_PY -m cleantables.features.composite <layouts.json> [--variants 0 1 2 3]
Out: $CLEANTABLES_DATA/vjepa_wm/composites_feats/{vj,dn}_v<n>.npy (L, 2, h, w, 1024) fp16 + .done.npy,
     uv.npy (L, 24, 2), uv_ok.npy (L, 24), index.json (layout id -> row, layout object order);
     uv_v<n>.npy / uv_ok_v<n>.npy: the uv in that variant's render (v >= 4 move the camera and rescale objects,
     cleantables.features.randomize; for v0-v3 it equals uv.npy)
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
CR = Path(CT_DATA + "/vjepa_wm/composite_renders")
OUT = Path(CT_DATA + "/vjepa_wm/composites_feats")      # path read by the world-model loader
SF = ROOT / "data/twin_table/ep/_scalefix"
HW = {"vj": (16, 28), "dn": (18, 32)}
T, MAXO, COV_MIN = 32, 24, 0.25


def mesh_v(src_ep, oid):
    o = {x["id"]: x for x in json.load(open(SF / src_ep / "scene.json"))["objects"]}[oid]
    return np.array([l.split()[1:4] for l in open(ROOT / o["usd"].replace(".usda", ".obj")) if l.startswith("v ")], np.float64)


def uv_of(lay, cam, sizes=None, base_ids=()):
    K, Tinv = cam
    hh, ww = HW["vj"]
    yy, xx = np.meshgrid((np.arange(hh) + 0.5) / hh, (np.arange(ww) + 0.5) / ww, indexing="ij")
    uv, ok = np.zeros((MAXO, 2), np.float32), np.zeros(MAXO, bool)
    for s0, o in enumerate(lay["objects"]):
        s = o.get("slot") if o.get("slot") is not None else s0          # token slot in the world model (objects without a token get no uv)
        if s is None or s >= MAXO:
            continue
        V = mesh_v(o["src_ep"], o["obj_id"])
        if sizes is not None:
            V = V * sizes[o["obj_id"] if (o["src_ep"] == lay["base_ep"] and o["obj_id"] in base_ids) else f"{o['src_ep']}__{o['obj_id']}"]
        X = V @ R.from_euler("z", o["yaw_off"]).as_matrix().T + [o["x"], o["y"], o["z_bottom"]]
        C = X @ Tinv[:3, :3].T + Tinv[:3, 3]
        if (C[:, 2] < 0.05).any():
            continue
        p = (C @ K.T)[:, :2] / C[:, 2:3] / 2.0
        m = np.zeros((270, 480), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(p).astype(np.int32)), 1)
        c = cv2.resize(m.astype(np.float32), (ww, hh), interpolation=cv2.INTER_AREA)
        w = c.sum()
        if w > COV_MIN:
            uv[s] = [(c * xx).sum() / w, (c * yy).sum() / w]; ok[s] = True
    return uv, ok


def variant_uv(L, variants, out=OUT):
    """uv_v<n>.npy / uv_ok_v<n>.npy per render variant (needs every base's render meta of that variant)."""
    for v in variants:
        if (out / f"uv_v{v}.npy").exists():
            continue
        metas = {b: CR / b / f"v{v}" / "meta.json" for b in {l["base_ep"] for l in L}}
        if not all(m.exists() for m in metas.values()):
            continue
        uv, ok = np.zeros((len(L), MAXO, 2), np.float32), np.zeros((len(L), MAXO), bool)
        for i, l in enumerate(L):
            dr = json.load(open(metas[l["base_ep"]]))["dr"]
            S = json.load(open(SF / l["base_ep"] / "scene.json"))
            if dr.get("full"):
                cam = (np.array(dr["cam_K"], np.float64).reshape(3, 3), np.linalg.inv(np.array(dr["cam_T"], np.float64).reshape(4, 4)))
                uv[i], ok[i] = uv_of(l, cam, dr["obj_size"], {o["id"] for o in S["objects"]})
            else:
                cam = (np.array(S["cameras"]["main"]["K"], np.float64).reshape(3, 3),
                       np.linalg.inv(np.array(S["cameras"]["main"]["T_cam_to_table"], np.float64)))
                uv[i], ok[i] = uv_of(l, cam)
        np.save(out / f"uv_v{v}.npy", uv); np.save(out / f"uv_ok_v{v}.npy", ok)
        print(f"uv v{v}: {int(ok.sum())} object uvs", flush=True)


def main(layouts, variants):
    import torch
    from cleantables.features.encoders import Encoder
    L = json.load(open(layouts))
    OUT.mkdir(parents=True, exist_ok=True)
    index = {l["id"]: dict(row=i, base_ep=l["base_ep"], objects=[f"{o['src_ep']}:{o['obj_id']}" for o in l["objects"]]) for i, l in enumerate(L)}
    json.dump(dict(layouts=str(layouts), index=index), open(OUT / "index.json", "w"))
    if not (OUT / "uv.npy").exists():
        cams = {}
        uv, ok = np.zeros((len(L), MAXO, 2), np.float32), np.zeros((len(L), MAXO), bool)
        for i, l in enumerate(L):
            if l["base_ep"] not in cams:
                S = json.load(open(SF / l["base_ep"] / "scene.json"))
                cams[l["base_ep"]] = (np.array(S["cameras"]["main"]["K"], np.float64).reshape(3, 3),
                                      np.linalg.inv(np.array(S["cameras"]["main"]["T_cam_to_table"], np.float64)))
            uv[i], ok[i] = uv_of(l, cams[l["base_ep"]])
        np.save(OUT / "uv.npy", uv); np.save(OUT / "uv_ok.npy", ok)
    try:
        variant_uv(L, variants)
    except Exception as e:                                                # uv is optional; the encode goes on
        print('variant uv failed:', repr(e), flush=True)
    torch.cuda.set_per_process_memory_fraction(3.5 / 23.5)
    encs = {"vj": Encoder("vj2l", layers=(24,), dev="cuda"), "dn": Encoder("dino2l", layers=(24,), dev="cuda")}
    for v in variants:
        if not any((CR / b / f"v{v}" / "meta.json").exists() for b in {l["base_ep"] for l in L}):
            print(f"v{v}: no renders, skipped", flush=True)
            continue
        for name in ("vj", "dn"):
            f, df = OUT / f"{name}_v{v}.npy", OUT / f"{name}_v{v}.done.npy"
            G = np.lib.format.open_memmap(f, mode="r+" if f.exists() else "w+", dtype=np.float16, shape=(len(L), 2) + HW[name] + (1024,))
            done = np.load(df) if df.exists() else np.zeros(len(L), bool)
            for i, l in enumerate(L):
                p = CR / l["base_ep"] / f"v{v}" / f"{l['id']}.jpg"
                if done[i] or not p.exists():
                    continue
                im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                X = np.stack([im] * T)[None] if name == "vj" else im[None, None]
                for fl in (0, 1):
                    Xi = X if fl == 0 else np.ascontiguousarray(X[:, :, :, ::-1])
                    G[i, fl] = encs[name](Xi)[24][0][0].float().cpu().numpy().astype(np.float16)
                done[i] = True
                if i % 100 == 0:
                    G.flush(); np.save(df, done); print(f"v{v} {name} {i}/{len(L)}", flush=True)
            G.flush(); np.save(df, done)
            print(f"v{v} {name}: {int(done.sum())}/{len(L)} layouts", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("layouts")
    ap.add_argument("--variants", type=int, nargs="*", default=[0, 1, 2, 3])
    a = ap.parse_args()
    main(a.layouts, a.variants)
