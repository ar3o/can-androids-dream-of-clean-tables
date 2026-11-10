"""Task progress learned from video.

Episodes are the full verified clips in data/episodes_v2 (they end when the object is in the box), with hand and forearm
painted out. Progress is judged relative to the episode's first frame: the input is the current DINOv2 patch tokens and
their change from frame 0, so the model reads what has changed in the scene rather than the camera angle. All 256 patch
tokens are kept and pooled by a learned attention query. Supervision is normalised time within each successful video plus
within-video ranking; no labels, tracks or robot data.

  python -m cleantables.worldmodel.progress extract | train | sim --rollouts R --cache C
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
EPS = ROOT / "data/episodes_v2"
V3 = ROOT / "data/processed/markerless_v3"
OUT = ROOT / "data/processed/progress_v2"
STEP = 2            # ~15 fps from 30 fps video


def read_all(path, step=STEP, long=960):
    cap = cv2.VideoCapture(str(path)); out = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            s = min(1.0, long / max(f.shape[:2]))
            out.append(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        i += 1
    return out


def arm_mask(lm, shape, dil=35):
    """Hand hull plus forearm: a thick band from the wrist, away from the fingers, to the image border."""
    m = np.zeros(shape[:2], np.uint8)
    if lm is None or not np.isfinite(lm).all():
        return m > 0
    cv2.fillConvexPoly(m, cv2.convexHull(lm.astype(np.int32)), 1)
    w, mcp = lm[0], lm[9]
    d = w - mcp; d /= (np.linalg.norm(d) + 1e-6)
    width = int(max(25, 1.1 * np.linalg.norm(lm[5] - lm[17])))
    end = w + d * 2 * max(shape[:2])
    cv2.line(m, tuple(int(v) for v in w), tuple(int(v) for v in end), 1, width)
    return cv2.dilate(m, np.ones((dil, dil), np.uint8)) > 0


def extract(device="cuda"):
    from cleantables.features.dino_frames import DinoExtractor
    ex = DinoExtractor(device)
    OUT.mkdir(parents=True, exist_ok=True)
    for vid in sorted(EPS.glob("*.mp4")):
        c = vid.stem
        fo = OUT / f"{c}.npy"
        if fo.exists():
            continue
        fr = read_all(vid)
        ht = V3 / c / "hand_track.npz"
        lm = np.load(ht)["lm2d"][::STEP] if ht.exists() else [None] * len(fr)
        H, W = fr[0].shape[:2]
        ys = [l[:, 1].mean() for l in lm if l is not None and np.isfinite(l).all()]
        side = min(H, W)
        if H > W:
            cy = float(np.median(ys)) if ys else H / 2
            y0 = int(np.clip(cy - side / 2, 0, H - side)); x0 = 0
        else:
            cx = float(np.median([l[:, 0].mean() for l in lm if l is not None and np.isfinite(l).all()] or [W / 2]))
            x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = 0
        imgs = []
        for k, f in enumerate(fr):
            g = f.copy()
            m = arm_mask(lm[k] if k < len(lm) else None, g.shape)
            if m.any():
                hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
                tab = (hsv[..., 2] < 90) & ~m
                g[m] = np.median(g[tab], 0) if tab.any() else 40
            imgs.append(cv2.cvtColor(g[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB))
        imgs = np.stack(imgs)
        np.save(fo, ex(imgs))
        cv2.imwrite(str(OUT / f"{c}_strip.jpg"), cv2.cvtColor(np.hstack([cv2.resize(imgs[i], (160, 160)) for i in np.linspace(0, len(imgs) - 1, 6).astype(int)]), cv2.COLOR_RGB2BGR))
        print("feat", c, imgs.shape, flush=True)


class RelHead(nn.Module):
    """Tokens of frame t and frame 0 -> progress. Per-token features [x_t, x_t - x_0] -> attention pooling -> MLP."""
    def __init__(self, d=384, h=128):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, h), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(256, h))
        self.q = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.att = nn.MultiheadAttention(h, 4, batch_first=True, dropout=0.1)
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Dropout(0.2), nn.Linear(h, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, xt, x0):
        z = self.inp(torch.cat([xt, xt - x0], -1)) + self.pos
        p, _ = self.att(self.q.expand(len(z), -1, -1), z, z)
        return torch.sigmoid(self.out(p[:, 0])).squeeze(-1)


def flip_tokens(x):
    return x.view(-1, 16, 16, x.shape[-1]).flip(2).reshape(x.shape)


def fit(seqs, epochs=40, seed=0, device="cuda"):
    torch.manual_seed(seed)
    m = RelHead().to(device)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
    X = [torch.as_tensor(np.asarray(s, np.float32), device=device) for s in seqs]
    for ep in range(epochs):
        for i in torch.randperm(len(X)).tolist():
            x = X[i]; T = len(x)
            k = torch.randint(0, T, (32,), device=device)
            xt, x0 = x[k], x[:1].expand(len(k), -1, -1)
            if torch.rand(1) < 0.5:
                xt, x0 = flip_tokens(xt), flip_tokens(x0)
            drop = (torch.rand(len(k), 256, 1, device=device) > 0.15).float()      # token dropout so that no single patch carries the answer
            y = k.float() / (T - 1)
            p = m(xt * drop, x0)
            a, b = torch.randint(0, len(k), (64,), device=device), torch.randint(0, len(k), (64,), device=device)
            s = torch.sign(y[b] - y[a])
            loss = ((p - y) ** 2).mean() + torch.relu(0.05 - s * (p[b] - p[a])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def predict(m, f, device="cuda"):
    x = torch.as_tensor(np.asarray(f, np.float32), device=device)
    return torch.cat([m(x[i:i + 64], x[:1].expand(len(x[i:i + 64]), -1, -1)) for i in range(0, len(x), 64)]).cpu().numpy()


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def train(device="cuda"):
    files = sorted(glob.glob(str(OUT / "IMG_*.npy")))
    clips = [Path(f).stem for f in files]
    seqs = [np.load(f) for f in files]
    recs = sorted({c.split("_")[1] for c in clips})
    per = {}
    for r in recs:
        m = fit([s for s, c in zip(seqs, clips) if c.split("_")[1] != r], device=device)
        for s, c in zip(seqs, clips):
            if c.split("_")[1] == r:
                p = predict(m, s, device)
                per[c] = dict(rho=spearman(p, np.arange(len(p))), start=float(p[:3].mean()), end=float(p[-3:].mean()), curve=np.round(p, 3).tolist())
    rho = np.array([v["rho"] for v in per.values()])
    res = dict(model="progress_v2 (full episodes, arm masked, relative to frame 0, attention over patches)", n_clips=len(clips), n_recordings=len(recs),
               loro_spearman_mean=float(rho.mean()), loro_spearman_median=float(np.median(rho)), frac_clips_rho_gt_0_7=float((rho > 0.7).mean()),
               mean_start=float(np.mean([v["start"] for v in per.values()])), mean_end=float(np.mean([v["end"] for v in per.values()])))
    m = fit(seqs, device=device)
    torch.save(m.state_dict(), OUT / "head.pt")
    json.dump(dict(summary=res, per_clip=per), open(OUT / "loro.json", "w"))
    rep = json.load(open(OUT / "report.json")) if (OUT / "report.json").exists() else {}
    rep["real_loro"] = res
    json.dump(rep, open(OUT / "report.json", "w"), indent=1)
    print(json.dumps(res), flush=True)


def sim(rollouts, cache, device="cuda"):
    from sklearn.metrics import roc_auc_score
    m = RelHead().to(device); m.load_state_dict(torch.load(OUT / "head.pt")); m.eval()
    rows = []
    for p in sorted(glob.glob(f"{rollouts}/ep_*.npz"), key=lambda q: int(Path(q).stem.split("_")[1])):
        f = Path(cache) / f"{Path(p).stem}_front.npy"
        if not f.exists():
            continue
        z = np.load(p, allow_pickle=True)
        pr = predict(m, np.load(f), device)
        succ = bool(z["success"].max() > 0)
        kind = "success" if succ else {1: "fell_off_edge", 2: "dropped", 3: "pushed_away", 4: "timeout"}.get(int(z["fail_event"].max()), "other")
        rows.append(dict(ep=Path(p).stem, kind=kind, success=succ, p_start=float(pr[:5].mean()), p_end=float(pr[-5:].mean()), p_max=float(pr.max()),
                         rho_time=spearman(pr, np.arange(len(pr))) if succ else None))
    y = np.array([r["success"] for r in rows]); pe = np.array([r["p_end"] for r in rows])
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r["p_end"])
    res = dict(rollouts=str(rollouts), n=len(rows), auroc_success_vs_fail_p_end=float(roc_auc_score(y, pe)) if 0 < y.sum() < len(y) else None,
               mean_rho_time_successes=float(np.mean([r["rho_time"] for r in rows if r["rho_time"] is not None])) if y.any() else None,
               mean_p_end_by_outcome={k: round(float(np.mean(v)), 3) for k, v in by.items()},
               mean_p_start=float(np.mean([r["p_start"] for r in rows])))
    rep = json.load(open(OUT / "report.json")) if (OUT / "report.json").exists() else {}
    rep[f"sim_{Path(rollouts).name}"] = res
    json.dump(rep, open(OUT / "report.json", "w"), indent=1)
    json.dump(rows, open(OUT / f"sim_{Path(rollouts).name}.json", "w"))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "train", "sim"])
    ap.add_argument("--rollouts"); ap.add_argument("--cache"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"extract": lambda: extract(a.device), "train": lambda: train(a.device), "sim": lambda: sim(a.rollouts, a.cache, a.device)}[a.cmd]()


def robot_mask(px, shape, radius_m=0.045):
    """Robot mask from its projected link origins (u, v, depth): a disc of ~4.5 cm around every link plus a thick segment between
    links adjacent in index order (the kinematic chain), sized by depth. Sim analogue of arm_mask()."""
    m = np.zeros(shape[:2], np.uint8)
    f = shape[1] / 2 / np.tan(np.deg2rad(60.4 / 2))          # focal (px) of the 18 mm / 20.955 mm pinhole
    ok = px[:, 2] > 0.05
    r = lambda d: int(np.clip(f * radius_m / d, 2, 80))
    for i in np.flatnonzero(ok):
        cv2.circle(m, (int(px[i, 0]), int(px[i, 1])), r(px[i, 2]), 1, -1)
    for i in range(len(px) - 1):
        for j in range(i + 1, min(len(px), i + 6)):
            if ok[i] and ok[j]:
                d3 = np.linalg.norm(px[i, :2] - px[j, :2]) / f * min(px[i, 2], px[j, 2])
                if d3 < 0.22:
                    cv2.line(m, (int(px[i, 0]), int(px[i, 1])), (int(px[j, 0]), int(px[j, 1])), 1, 2 * r(min(px[i, 2], px[j, 2])))
    return m > 0


def paint_robot(frames_rgb, robot_px):
    out = []
    for f, px in zip(frames_rgb, robot_px):
        g = f.copy()
        m = robot_mask(px, g.shape)
        if m.any():
            hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
            tab = (hsv[..., 2] < 90) & ~m
            g[m] = np.median(g[tab], 0) if tab.any() else 40
        out.append(g)
    return np.stack(out)
