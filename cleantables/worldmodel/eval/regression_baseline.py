"""Direct-regression baselines on the world model's own trunk.

The training loop of cleantables.worldmodel.train (train_one: sampler, batch, lr, warm-up, cosine, clip, checkpoint / resume)
is reused as is; only two names are swapped in this process: `build` returns the same UWMW plus one regression head, and
`compute_loss_w` is a Huber regression of each object's end offset in robot mode (3D only, no hands or contact).
A1 trains without the composite / re-layout augmentations, A2 with them. Checkpoints go to out/reg/.

  $VLA_PY -m cleantables.worldmodel.eval.regression_baseline train A1|A2 <seed> [smoke] [cpu]
  $VLA_PY -m cleantables.worldmodel.eval.regression_baseline pred  A1|A2 <seed> [text=shuffle]
      -> out/wm/pred_<A1|A2>_s<seed>[_shuffle].npz (score.py format)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.worldmodel.base_model import PS
from cleantables.worldmodel.base_train import mean_w
from cleantables.worldmodel import train as VT
from cleantables.worldmodel.model import mlp

EB = Path(__file__).resolve().parents[3] / "outputs" / "eval"; OUTD = EB / "out" / "reg"; OW = EB / "out" / "wm"
GPU_PAUSE_GB, GPU_RESUME_GB = 19.0, 18.0                  # pause training while total GPU use exceeds this (shared GPU)


def cfg_for(name):
    c = VT.full_cfg("watch_notab")
    c.update(vid=(), fut=False, objg=False, p_video=0.0, mode_fixed=1, dist_head=False, val_log=False, steps=9000,   # video off, as in watch_novideo,
             p_render=0.0, p_vcrop=0.0)                   # and no render / crop video sources
    if name == "A1":
        c.update(comp_balance=False, p_comp=0.0, p_pair=0.0, p_rigid=0.0, p_relay=0.0)
    elif name != "A2":
        raise ValueError(name)
    return c


def blank(b):                                             # eval mode 1nh: no hands, no contact input
    b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
    for k_ in ("obj_contact_ok", "obj_contact2_ok"):
        if k_ in b:
            b[k_] = torch.zeros_like(b[k_])
    return b


_orig_build = VT.build


def build_reg(cfg):
    m = _orig_build(cfg)
    m.reg_h = mlp(2 * cfg["d"], 3, cfg["d"])              # same head builder, on the token the model's own heads read (_oc)
    return m


def gpu_wait(dev):
    if dev != "cuda":
        return
    free, tot = torch.cuda.mem_get_info(); used = (tot - free) / 2**30
    if used <= GPU_PAUSE_GB:
        return
    torch.cuda.empty_cache(); t0 = time.time()
    print(f"GPU total {used:.1f} GB > {GPU_PAUSE_GB}: pausing", flush=True)
    while True:
        time.sleep(10); free, tot = torch.cuda.mem_get_info(); used = (tot - free) / 2**30
        if used <= GPU_RESUME_GB:
            print(f"resumed after {time.time() - t0:.0f} s (GPU total {used:.1f} GB)", flush=True); return


def reg_loss(m, b, cfg, g):
    gpu_wait(b["obj_mask"].device.type)
    B = b["obj_mask"].shape[0]
    b = blank(b)
    E = m.encode(b, torch.ones(B, dtype=torch.long, device=b["obj_mask"].device)); E["kind"] = b["obj_kind"]
    obj = b["obj_mask"].bool() & (b["obj_kind"] == 0)
    okp = b["t1_ok"].bool() & b["t1_present"].bool() & obj
    y = (b["t1_pos"].float() - b["obj_geo"][..., :3].float()) / PS
    p = m.reg_h(m._oc(E)).float()
    return {"reg": mean_w(F.huber_loss(p, y, reduction="none", delta=0.1).sum(-1), okp.float())}


def data(cfg):
    from cleantables.worldmodel.data import Data
    D = Data(text=cfg["text"], feats=True)                # appearance tokens need the feature store
    D.comp_balance = bool(cfg.get("comp_balance", False)); D.relay_keep_inside = bool(cfg.get("relay_keep_inside", False))
    D.relay_keep_dest = bool(cfg.get("relay_keep_dest", True))
    return D


def train(name, seed, smoke, dev):
    VT.build = build_reg; VT.compute_loss_w = reg_loss
    cfg = cfg_for(name)
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get("WM_GPU_GB", "3.0")) / 23.5))
    D = data(cfg); idx = D.train_idx
    out = OUTD / name; out.mkdir(parents=True, exist_ok=True)
    p = out / (f"smoke_seed{seed}.pt" if smoke else f"fold-1_seed{seed}.pt")
    print(f"{name} seed {seed}: {len(idx)} training samples, steps {cfg['steps']}, comp {cfg['p_comp']} pair {cfg['p_pair']} rigid {cfg['p_rigid']}", flush=True)
    VT.train_one(D, idx, cfg, 1000 * 0 + seed, p, dev, smoke)
    print("DONE", name, seed, flush=True)


def pred(name, seed, opts):
    torch.set_grad_enabled(False)
    from cleantables.worldmodel.evaluate import Batcher
    cfg = cfg_for(name); m = build_reg(cfg)
    m.load_state_dict(torch.load(OUTD / name / f"fold-1_seed{seed}.pt", map_location="cpu")); m.eval()
    D = data(cfg); Bt = Batcher(D); G = np.load(OW / "gt.npz", allow_pickle=True); test = D.test_idx
    assert (G["idx"] == test).all()
    rows = np.arange(len(test)); temb = None; tag = ""
    if "text=shuffle" in opts:                             # the other family's instruction (as predict.py text=shuffle)
        swap = {"fruit": "Put the tools into the toolbox.", "tools": "Put the fruit into the fruit bowl."}
        rows = np.flatnonzero(G["scope"] != ""); temb = D.embed_new([swap["fruit"] if G["video"][r] == "IMG_2516" else swap["tools"] for r in rows]); tag = "_shuffle"
    goal, pn = [], []
    for b0 in range(0, len(rows), 64):
        q = rows[b0:b0 + 64]
        b = blank(Bt.get("v2", cfg, test[q]))
        if temb is not None:
            b["task_emb"] = temb[b0:b0 + 64].to(b["task_emb"].dtype)
        E = m.encode(b, torch.ones(len(q), dtype=torch.long)); E["kind"] = b["obj_kind"]
        off = m.reg_h(m._oc(E)).float() * PS; now = b["obj_geo"][..., :3].float()
        goal.append((now + off).numpy()); pn.append(off.norm(dim=-1).numpy())
    np.savez_compressed(OW / f"pred_{name}_s{seed}{tag}.npz", rows=rows, all__goal=np.concatenate(goal), all__p_now=np.concatenate(pn))
    print("wrote", f"pred_{name}_s{seed}{tag}", len(rows))


if __name__ == "__main__":
    cmd, name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); opts = sys.argv[4:]
    if cmd == "train":
        train(name, seed, "smoke" in opts, "cpu" if "cpu" in opts else "cuda")
    else:
        pred(name, seed, opts)
