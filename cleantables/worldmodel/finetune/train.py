"""Fine-tune the frozen watch model (watch_notab) with strong visual domain randomisation. Each batch is half the usual real training
samples (D.make_batch, train=True, with all training augmentations) and half domain-randomised rows (finetune.encode: the same training
samples' keyframes rendered in 12 rooms / lightings), whose video inputs (V-JEPA / DINOv2 grids, image positions) and per-object
appearance tokens come from the render; every 3D input and target is the real sample's (the renders replay it).
Rendered rows: hands dropped with p 0.5 (no hands are in the renders); 25% run 3D-only (appearance tokens still on, as mode 1);
T3 (the future real-video latent) is masked for them. Same losses as training (worldmodel.train.compute_loss_w). Written to a separate
run folder; the source model is not modified.
  python -m cleantables.worldmodel.finetune.train --feats DR/feats --out DR/runs/watch_notab_dr [--seeds 0 1 2] [--steps 1500]"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=CT_DATA + "/wm_eval/wm_v2_world/runs_final/watch_notab")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--dev", default="cuda")
    a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True); SRC = Path(a.src)
    from torch.optim import AdamW
    from cleantables.worldmodel.data import Data
    from cleantables.worldmodel.train import build, compute_loss_w
    dev = a.dev
    D = Data(text="minilm")
    grp = D.S.d["group"].astype(str); kk = D.S.d["snap_k"].astype(int); oid_all = np.asarray(D.S.d["obj_id"]).astype(int)
    train = set(D.train_idx.tolist()); where = {(g, k): i for i, (g, k) in enumerate(zip(grp, kk)) if i in train}
    R = []                                                                     # rendered rows: (dataset idx, file, row)
    F = {p.name: dict(np.load(p, allow_pickle=True)) for p in sorted(Path(a.feats).glob("*.npz"))}
    for nm, f in F.items():
        for j, k in enumerate(f["snap_k"]):
            i = where.get((str(f["episode"]), int(k)))
            if i is not None:
                R.append((i, nm, j))
    print(f"[dr-ft] {len(R)} DR rows from {len(F)} renders; {len(D.train_idx)} real training samples", flush=True)
    for seed in a.seeds:
        src = SRC / f"fold-1_seed{seed}.pt"; cfg = json.load(open(src.with_suffix(".json")))["cfg"]; cfg["vid"] = tuple(cfg["vid"])
        m = build(cfg).to(dev); m.load_state_dict(torch.load(src, map_location=dev)); m.train()
        if (SRC / "sd_scale.json").exists():
            m.sd_scale = json.load(open(SRC / "sd_scale.json"))["sd_scale"]
        opt = AdamW(m.parameters(), lr=a.lr, weight_decay=cfg["wd"])
        rng = np.random.default_rng(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed); t0 = time.time()
        for it in range(a.steps):
            for gp in opt.param_groups:
                gp["lr"] = a.lr * min(1, (it + 1) / 50) * 0.5 * (1 + np.cos(np.pi * it / a.steps))
            nb = a.bs // 2
            br = D.make_batch(D.train_idx[rng.integers(0, len(D.train_idx), a.bs - nb)], cfg, rng, dev, train=True, gen=gen)
            pick = [R[q] for q in rng.integers(0, len(R), nb)]
            bd = D.make_batch(np.array([p[0] for p in pick]), cfg, rng, dev, train=False, gen=gen)
            for n_, (i, nm, j) in enumerate(pick):                              # the render's inputs for this sample
                f = F[nm]; tr = f["track_id"]
                bd["vj_grid"][n_] = torch.as_tensor(f["vj"][j].astype(np.float32), device=dev).to(bd["vj_grid"].dtype)
                bd["dn_grid"][n_] = torch.as_tensor(f["dn"][j].astype(np.float32), device=dev).to(bd["dn_grid"].dtype)
                bd["uv"][n_] = 0; bd["uv_ok"][n_] = False; bd["app_ok"][n_] = False
                for k_ in ("app_vj", "app_dn", "app_ctl"):
                    bd[k_][n_] = 0
                for s, o in enumerate(oid_all[i]):
                    hit = np.flatnonzero(tr == o)
                    if o < 0 or not len(hit) or not bool(bd["obj_mask"][n_, s]):
                        continue
                    r = int(hit[0])
                    if f["uv_ok"][j, r]:
                        bd["uv"][n_, s] = torch.as_tensor(f["uv"][j, r], device=dev); bd["uv_ok"][n_, s] = True
                    if f["app_ok"][j, r]:
                        bd["app_vj"][n_, s] = torch.as_tensor(f["app_vj"][j, r].astype(np.float32), device=dev)
                        bd["app_dn"][n_, s] = torch.as_tensor(f["app_dn"][j, r].astype(np.float32), device=dev)
                        bd["app_ctl"][n_, s] = torch.as_tensor(f["app_ctl"][j, r].astype(np.float32), device=dev); bd["app_ok"][n_, s] = True
                if rng.random() < 0.5:                                          # no hands in the renders
                    bd["hand_geo"][n_, ..., 0] = 0; bd["hand_q_ok"][n_] = False
                    bd["obj_contact_ok"][n_] = False; bd["obj_contact2_ok"][n_] = False
            bd["vid_ok"][:] = True; bd["vj_ok"][:] = True
            bd["vid_keep"] = torch.as_tensor(rng.random(nb) < 0.75, device=dev)  # 25% 3D-only
            if "aug_jepa_bad" in bd:
                bd["aug_jepa_bad"][:] = True                                    # T3's target is the real video's future latent
            Ls = compute_loss_w(m, br, cfg, gen); Ld = compute_loss_w(m, bd, cfg, gen)
            loss = sum(Ls.values()) + sum(Ld.values())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 3.0); opt.step()
            if it % 250 == 0 or it == a.steps - 1:
                print(f"[dr-ft] seed {seed} it {it} real {sum(Ls.values()).item():.3f} dr {sum(Ld.values()).item():.3f} "
                      f"({(time.time() - t0) / (it + 1):.2f} s/it)", flush=True)
        m.eval(); torch.save(m.state_dict(), O / f"fold-1_seed{seed}.pt")
        json.dump(dict(cfg={**cfg, "vid": list(cfg["vid"])}, dr=dict(src=str(src), steps=a.steps, lr=a.lr, rows=len(R), renders=len(F))),
                  open(O / f"fold-1_seed{seed}.json", "w"), indent=1)
    if (SRC / "sd_scale.json").exists():
        shutil.copy(SRC / "sd_scale.json", O / "sd_scale.json")
    print(f"[dr-ft] -> {O}", flush=True)


if __name__ == "__main__":
    main()
