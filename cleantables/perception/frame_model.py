"""Task-agnostic object world model: from one camera frame (+ depth) of the scene just before a move, predict which object moves
next, where it ends (relation target + offset + yaw) and its 3D path.

Inference input = what a single frame gives (a real video frame or a sim render):
  frame -> SAM 3 with the discovered phrases (+ 'hand') -> clean (tracking.clean) -> per detection a 3D oriented box from depth
  (tracking.obb) + phrase + DINOv2 crop feature (vision.crop_obj) ; + one DINOv2 scene token of the table-region crop.
  (tokens_from_detections() below; training uses the same tokens from the SAM 3 keyframe just before each move.)
Object token = [x, y, top, L, W, top-height, sin 2yaw, cos 2yaw] (table frame, centred on the scene's mean xy) + MiniLM(phrase)
  + DINOv2(crop) [+ 'mover' flag].  Context tokens: DINOv2 scene token [vision], VLM subtask MiniLM embedding [vlm].
Outputs: next object (pointer over tokens, mover flag off), end target (pointer over tokens + TABLE + LEAVES), end offset from the
  target's top centre (or from the mover's start for TABLE), relative yaw (sin/cos 2x), 16-point 3D path (residual on the line).
Labels come from the tracked video (cleantables.perception.events): carrier object, end pose, relation target, path.
Evaluation: leave-one-run-out on the seen runs; train on all seen runs -> unseen-object-set runs. Variants: full, no_vlm, no_vision,
  none (geometry + phrase only). Baselines: next = random / nearest to the last placement / largest remaining (baseline only);
  placement = centre of the true target object (top centre; mover start for table moves). 95 % bootstrap CIs over moves.
  $VLA_PY -m cleantables.perception.frame_model VIDEOS...   -> runs/generic/wm/eval.json, models, predictions
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cleantables.perception.common import OUT, ROOT, embed, gpu_free, log, names, vdir
from cleantables.perception.events import relation

RUNS = ROOT / os.environ.get("WM_OUT", "runs/generic/wm")
REL_ONLY = [r for r in os.environ.get("WM_REL", "").split(",") if r]   # e.g. on_top_of,inside: train / test only on these moves
NP = 16
MAXO = 24
TABLE, LEAVES = MAXO, MAXO + 1
SEEN = [f"IMG_{i}" for i in (2533, 2534, 2535, 2536, 2537, 2538, 2539, 2542, 2544, 2546, 2547, 2548)]
UNSEEN = [f"IMG_{i}" for i in range(2549, 2556)]
VARIANTS = {"full": (True, True), "no_vlm": (True, False), "no_vision": (False, True), "none": (False, False)}   # (vision, vlm)


# ------------------------------------------------------------------ tokens from one frame
def box_feat(b):
    return np.array([b["x"], b["y"], b["top"], max(b["L"], b["W"]), min(b["L"], b["W"]), b["top"],
                     np.sin(2 * b["yaw"]), np.cos(2 * b["yaw"])], np.float32)


def tokens_from_detections(boxes, phrases, crop_emb, scene_emb, E):
    """boxes: list of tracking.obb dicts (one per cleaned SAM 3 detection of one frame); phrases: their SAM 3 phrases;
    crop_emb (n,768) DINOv2 crop features; scene_emb (768,). -> dict of arrays (the world model's whole input)."""
    return dict(geo=np.stack([box_feat(b) for b in boxes]), txt=np.stack([E[p] for p in phrases]),
                vis=np.asarray(crop_emb, np.float32), scene=np.asarray(scene_emb, np.float32))


def build(videos):
    """One sample per move / group move / leave event: the tokens of the SAM 3 keyframe just before it + its labels."""
    S = []
    texts = set()
    for v in videos:
        if not (vdir(v) / "events.json").exists():
            continue
        Ev = json.load(open(vdir(v) / "events.json"))
        R = pickle.load(open(vdir(v) / "tracks.pkl", "rb"))
        Z = np.load(vdir(v) / "vision.npz")
        cemb = dict(zip(Z["det"].tolist(), Z["emb"].astype(np.float32)))
        semb = dict(zip(Z["keyframes"].tolist(), Z["scene"].astype(np.float32)))
        src = {o["id"]: o["src"] for o in Ev["objects"]}
        size = {o["id"]: o["size"] for o in Ev["objects"]}
        objdets = {}
        for oid, s in src.items():
            o = R["objects"][s]
            objdets[oid] = dict(zip(o["sight_k"].tolist(), o["det"]))
        kfs = sorted(R["kept"])
        prev_end = None
        for e in sorted(Ev["events"], key=lambda e: e["f_start"]):
            if REL_ONLY and e["relation"] not in REL_ONLY:
                continue
            mover = e["obj"]
            kf = next((k for k in reversed(kfs) if k <= e["f_start"] - 3 and k >= e["f_start"] - 40 and mover in objdets
                       and k in objdets[mover] and R["det_box"].get(objdets[mover][k])), None)
            if kf is None:
                continue
            dets = [i for i in R["kept"][kf] if R["det_box"].get(i) and i in cemb]
            dets = sorted(dets, key=lambda i: -R["det_box"][i]["prob"])[:MAXO]
            mdet = objdets[mover][kf]
            if mdet not in dets:
                continue
            boxes = [R["det_box"][i] for i in dets]
            phr = [b["phrase"] for b in boxes]
            texts.update(phr)
            owner = {}
            for oid, dd in objdets.items():
                if kf in dd:
                    owner[dd[kf]] = oid
            tok_obj = [owner.get(i, -1) for i in dets]
            mi = dets.index(mdet)
            if e["relation"] == "leaves":
                dst = LEAVES
            elif e["target"] is None:
                dst = TABLE
            else:
                dst = tok_obj.index(e["target"]) if e["target"] in tok_obj else -1
            end = e["end"]
            s = dict(video=v, ev=e["id"], kf=kf, f_start=e["f_start"], f_end=e["f_end"], kind=e["kind"], relation=e["relation"],
                     group=e["group"], members=e["members"], dets=dets, tok_obj=tok_obj, mover=mi, dst=dst, geo=np.stack([box_feat(b) for b in boxes]),
                     phr=phr, vis=np.stack([cemb[i] for i in dets]), scene=semb[kf], phrase=e["phrase"],
                     target_phrase=e["target_phrase"], subtask=e.get("subtask", ""), mover_size=size[mover],
                     start=np.array([e["start"]["x"], e["start"]["y"], e["start"]["top"]], np.float32),
                     end=np.array([end["x"], end["y"], end["top"]], np.float32) if end else None,
                     end_yaw=float(end["yaw"]) if end and np.isfinite(end["yaw"]) else np.nan,
                     path=np.array(e["path"], np.float32) if e["path"] else None, prev_end=prev_end,
                     sizes={oid: size[oid] for oid in size})
            texts.add(s["subtask"])
            S.append(s)
            if end:
                prev_end = s["end"]
    texts.discard("")
    E = dict(zip(sorted(texts), embed(sorted(texts))))
    E[""] = np.zeros(384, np.float32)
    return S, E


# ------------------------------------------------------------------ model
class GenericWM(nn.Module):
    def __init__(self, d=128, vision=True, vlm=True):
        super().__init__()
        self.vision, self.vlm = vision, vlm
        self.geo = nn.Linear(8, d)
        self.txt = nn.Linear(384, 64)
        self.vis = nn.Linear(768, 64)
        self.tok = nn.Sequential(nn.Linear(d + 64 + 64 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.scene = nn.Linear(768, d)
        self.sub = nn.Linear(384, d)
        self.special = nn.Parameter(torch.randn(2, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 3)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.off = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 8))        # offset mu (3), log-std (3), yaw (2)
        self.path = nn.Sequential(nn.Linear(d + 6, 2 * d), nn.GELU(), nn.Linear(2 * d, NP * 3))
        self.nxt = nn.Linear(d, 1)

    def forward(self, bt, mover):
        geo, txt, vis, mask = bt["geo"], bt["txt"], bt["vis"], bt["mask"]
        if not self.vision:
            vis = torch.zeros_like(vis)
        x = self.tok(torch.cat([self.geo(geo), self.txt(txt), self.vis(vis), mover[..., None]], -1))
        B, N, d = x.shape
        ctx = [self.special[None].expand(B, -1, -1)]
        if self.vision:
            ctx.append(self.scene(bt["scene"])[:, None])
        if self.vlm:
            ctx.append(self.sub(bt["sub"])[:, None])
        c = torch.cat(ctx, 1)
        h = self.tr(torch.cat([x, c], 1), src_key_padding_mask=torch.cat([~mask, torch.zeros(B, c.shape[1], dtype=torch.bool, device=x.device)], 1))
        ho, hs = h[:, :N], h[:, N:N + 2]
        hm = (ho * mover[..., None]).sum(1)
        keys = torch.cat([ho, hs], 1)
        logit = (self.q(hm)[:, None] * self.k(keys)).sum(-1) / d ** 0.5
        logit = logit.masked_fill(torch.cat([~mask | (mover > 0), torch.zeros(B, 2, dtype=torch.bool, device=x.device)], 1), -1e4)
        off = self.off(torch.cat([hm[:, None].expand(-1, N + 2, -1), keys], -1))
        return dict(dst=logit, off=off, h=hm, nxt=self.nxt(ho).squeeze(-1).masked_fill(~mask, -1e4))

    def path_from(self, hm, start, end):
        res = self.path(torch.cat([hm, start, end], -1)).view(-1, NP, 3)
        s = torch.linspace(0, 1, NP, device=hm.device)[None, :, None]
        return start[:, None] + s * (end - start)[:, None] + res


def batch(S, E, idx, rng, train, dev):
    B = len(idx)
    geo = np.zeros((B, MAXO, 8), np.float32); txt = np.zeros((B, MAXO, 384), np.float32); vis = np.zeros((B, MAXO, 768), np.float32)
    mask = np.zeros((B, MAXO), bool); mover = np.zeros((B, MAXO), np.float32); scene = np.zeros((B, 768), np.float32)
    sub = np.zeros((B, 384), np.float32); dst = np.full(B, -100, np.int64); end = np.zeros((B, 3), np.float32)
    yaw = np.zeros((B, 2), np.float32); ym = np.zeros(B, bool); path = np.zeros((B, NP, 3), np.float32); pm = np.zeros(B, bool)
    anc = np.zeros((B, 3), np.float32); cen = np.zeros((B, 3), np.float32); rot = np.zeros(B, np.float32)
    for b, i in enumerate(idx):
        s = S[i]
        g = s["geo"].copy(); n = len(g)
        keep = np.ones(n, bool)
        if train:
            keep = rng.random(n) > 0.1
            keep[s["mover"]] = True
            if 0 <= s["dst"] < MAXO:
                keep[s["dst"]] = True
        c = np.r_[g[keep, :2].mean(0), 0].astype(np.float32)
        th = rng.uniform(0, 2 * np.pi) if train else 0.0
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
        g[:, :3] -= c
        g[:, :2] = g[:, :2] @ R.T
        a2 = np.arctan2(g[:, 6], g[:, 7]) + 2 * th
        g[:, 6], g[:, 7] = np.sin(a2), np.cos(a2)
        if train:
            g[:, :3] += rng.normal(0, 0.01, (n, 3)); g[:, 3:6] *= rng.uniform(0.9, 1.1, (n, 3))
        geo[b, :n] = g; mask[b, :n] = keep
        txt[b, :n] = np.stack([E[p] for p in s["phr"]]); vis[b, :n] = s["vis"]
        mover[b, s["mover"]] = 1; scene[b] = s["scene"]; sub[b] = E.get(s["subtask"], E[""])
        cen[b] = c; rot[b] = th
        dst[b] = s["dst"] if (s["dst"] >= MAXO or (s["dst"] >= 0 and keep[s["dst"]])) else -100
        if s["end"] is not None:
            e = s["end"] - c; e[:2] = R @ e[:2]; end[b] = e
            k = s["dst"]
            anc[b] = geo[b, k, :3] if 0 <= k < MAXO else geo[b, s["mover"], :3]
            if np.isfinite(s["end_yaw"]):
                ref = np.arctan2(geo[b, k, 6], geo[b, k, 7]) / 2 if 0 <= k < MAXO else 0.0
                ry = 2 * (s["end_yaw"] + th - ref)
                yaw[b] = [np.sin(ry), np.cos(ry)]; ym[b] = True
        if s["path"] is not None:
            p = s["path"] - c; p[:, :2] = p[:, :2] @ R.T; path[b] = p; pm[b] = True
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    return dict(geo=T(geo), txt=T(txt), vis=T(vis), mask=T(mask, torch.bool), mover=T(mover), scene=T(scene), sub=T(sub),
                dst=T(dst, torch.long), end=T(end), anc=T(anc), yaw=T(yaw), ym=T(ym, torch.bool), path=T(path), pm=T(pm, torch.bool),
                cen=cen, rot=rot)


def losses(m, bt):
    o = m(bt, bt["mover"])
    L = {}
    ok = bt["dst"] >= 0
    if ok.any():
        L["dst"] = nn.functional.cross_entropy(o["dst"][ok], bt["dst"][ok])
    B = len(bt["dst"]); ar = torch.arange(B, device=bt["dst"].device)
    mv = ok & (bt["dst"] != LEAVES)
    if mv.any():
        off = o["off"][ar[mv], bt["dst"][mv]]
        mu, ls = off[:, :3], off[:, 3:6].clamp(-5, 2)
        tgt = bt["end"][mv] - bt["anc"][mv]
        L["end"] = ((((tgt - mu) / ls.exp()) ** 2 / 2 + ls).sum(-1)).mean()
        ymv = bt["ym"][mv]
        if ymv.any():
            L["yaw"] = ((nn.functional.normalize(off[ymv, 6:8], dim=-1) - bt["yaw"][mv][ymv]) ** 2).sum(-1).mean()
    if bt["pm"].any():
        pth = m.path_from(o["h"][bt["pm"]], bt["path"][bt["pm"]][:, 0], bt["end"][bt["pm"]])
        L["path"] = 10 * ((pth - bt["path"][bt["pm"]]) ** 2).sum(-1).mean()
    nl = m(bt, torch.zeros_like(bt["mover"]))["nxt"]
    L["next"] = 0.5 * nn.functional.cross_entropy(nl, bt["mover"].argmax(-1))
    return L


def train(S, E, tr, variant, dev, epochs=250, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vision, vlm = VARIANTS[variant]
    m = GenericWM(vision=vision, vlm=vlm).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), 32):
            L = losses(m, batch(S, E, perm[i:i + 32], rng, True, dev))
            opt.zero_grad(); sum(L.values()).backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    return m.eval()


@torch.no_grad()
def predict(m, S, E, idx, dev, oracle_target=False):
    bt = batch(S, E, idx, np.random.default_rng(0), False, dev)
    o = m(bt, bt["mover"])
    k = o["dst"].argmax(-1)
    if oracle_target:
        k = torch.where(bt["dst"] >= 0, bt["dst"], k)
    B = len(k); ar = torch.arange(B, device=k.device)
    anc = torch.where((k < MAXO)[:, None], bt["geo"][ar, k.clamp(max=MAXO - 1), :3], bt["geo"][ar, bt["mover"].argmax(-1), :3])
    off = o["off"][ar, k]
    end = anc + off[:, :3]
    start = bt["geo"][ar, bt["mover"].argmax(-1), :3]
    path = m.path_from(o["h"], start, end)
    nl = m(bt, torch.zeros_like(bt["mover"]))["nxt"]
    out = []
    for b, i in enumerate(idx):
        c = bt["cen"][b]
        ref = (np.arctan2(bt["geo"][b, k[b], 6].item(), bt["geo"][b, k[b], 7].item()) / 2) if k[b] < MAXO else 0.0
        yo = off[b, 6:8].cpu().numpy()
        yaw = float((ref + np.arctan2(yo[0], yo[1]) / 2) % np.pi)
        out.append(dict(i=int(i), k=int(k[b]), p=o["dst"][b].softmax(-1).cpu().numpy(), end=end[b].cpu().numpy() + c, yaw=yaw,
                        path=path[b].cpu().numpy() + c, nxt=nl[b].cpu().numpy()[:len(S[i]["dets"])]))
    return out


# ------------------------------------------------------------------ evaluation
def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [np.nan, np.nan, np.nan, 0]
    r = np.random.default_rng(seed)
    bs = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(x))]


def yaw_err(a, b):
    d = abs((a - b + np.pi / 2) % np.pi - np.pi / 2)
    return float(np.degrees(d))


def score(S, preds, preds_or):
    R = {}

    def add(k, v):
        R.setdefault(k, []).append(float(v))
    for pr, po in zip(preds, preds_or):
        s = S[pr["i"]]
        n = len(s["dets"])
        rk = int((pr["nxt"] > pr["nxt"][s["mover"]]).sum())
        add("next_top1", rk == 0); add("next_rand", 1 / n)
        vol = [g[3] * g[4] * g[5] for g in s["geo"]]
        add("next_largest", int(np.argmax(vol)) == s["mover"])
        if s["prev_end"] is not None:
            dd = np.linalg.norm(s["geo"][:, :2] - s["prev_end"][:2], axis=1)
            dd[np.linalg.norm(s["geo"][:, :2] - s["prev_end"][:2], axis=1) < 0.03] = 9   # the object just placed there is not 'next'
            add("next_nearest", int(np.argmin(dd)) == s["mover"])
        else:
            add("next_nearest", 1 / n)
        if s["dst"] >= 0:
            add("target_ok", pr["k"] == s["dst"])
            nz = [j for j in range(n) if j != s["mover"]]
            if s["dst"] < MAXO and nz:
                dd = [np.linalg.norm(s["geo"][j, :2] - s["geo"][s["mover"], :2]) for j in nz]
                add("target_nearest_obj", nz[int(np.argmin(dd))] == s["dst"])
        if s["end"] is None or s["dst"] < 0:
            continue
        e = s["end"]
        add("end_err_cm", 100 * np.linalg.norm(pr["end"][:2] - e[:2]))
        add("end_err_oracle_target_cm", 100 * np.linalg.norm(po["end"][:2] - e[:2]))
        add("end_z_err_cm", 100 * abs(pr["end"][2] - e[2]))
        base = s["geo"][s["dst"], :3] if s["dst"] < MAXO else s["start"]
        add("baseline_target_centre_cm", 100 * np.linalg.norm(base[:2] - e[:2]))
        add("baseline_stays_cm", 100 * np.linalg.norm(s["start"][:2] - e[:2]))
        if s["dst"] < MAXO:
            add("end_err_cm[on object]", 100 * np.linalg.norm(pr["end"][:2] - e[:2]))
            add("end_err_oracle_target_cm[on object]", 100 * np.linalg.norm(po["end"][:2] - e[:2]))
            add("baseline_target_centre_cm[on object]", 100 * np.linalg.norm(base[:2] - e[:2]))
        if np.isfinite(s["end_yaw"]):
            add("yaw_err_deg", yaw_err(pr["yaw"], s["end_yaw"]))
            ty = np.arctan2(s["geo"][s["dst"], 6], s["geo"][s["dst"], 7]) / 2 if s["dst"] < MAXO else np.arctan2(s["geo"][s["mover"], 6], s["geo"][s["mover"], 7]) / 2
            add("baseline_yaw_align_target_deg", yaw_err(ty, s["end_yaw"]))
        # relation of the predicted end pose, computed with the same geometry as the labels
        others = []
        for j in range(n):
            if j == s["mover"]:
                continue
            g = s["geo"][j]
            others.append((j, dict(x=g[0], y=g[1], top=g[2], yaw=np.arctan2(g[6], g[7]) / 2), [g[3], g[4], g[5]]))
        sz = list(s["mover_size"])
        rel, _ = relation(dict(x=pr["end"][0], y=pr["end"][1], top=pr["end"][2], yaw=pr["yaw"]), sz, others)
        add("relation_ok", rel == s["relation"])
        if s["path"] is not None:
            add("path_err_cm", 100 * np.linalg.norm(po["path"] - s["path"], axis=1).mean())
            line = s["path"][0] + np.linspace(0, 1, NP)[:, None] * (e - s["path"][0])
            add("baseline_line_path_err_cm", 100 * np.linalg.norm(line - s["path"], axis=1).mean())
    return R


def summarise(R):
    return {k: boot(v) for k, v in R.items()}


def main(videos=None):
    videos = videos or SEEN + UNSEEN
    dev = "cuda" if gpu_free() >= 2000 else "cpu"
    if dev == "cpu":
        torch.set_num_threads(6)
    RUNS.mkdir(parents=True, exist_ok=True)
    S, E = build(videos)
    seen = [v for v in SEEN if v in videos]; unseen = [v for v in UNSEEN if v in videos]
    from collections import Counter
    log(f"WM samples {len(S)}: {dict(Counter(s['video'] for s in S))}; relations {dict(Counter(s['relation'] for s in S))}; "
        f"target known {sum(s['dst'] >= 0 for s in S)}")
    out = dict(n_samples=len(S), device=dev, seen=seen, unseen=unseen)
    allpred = {}
    for var in [v_ for v_ in VARIANTS if v_ in os.environ.get("WM_VARIANTS", ",".join(VARIANTS)).split(",")]:
        t0 = time.time()
        Rloo = {}
        for v in seen:                                   # leave-one-run-out on the seen runs
            te = [i for i, s in enumerate(S) if s["video"] == v]
            tr = [i for i, s in enumerate(S) if s["video"] in seen and s["video"] != v]
            if not te or len(tr) < 10:
                continue
            m = train(S, E, tr, var, dev)
            pr, po = predict(m, S, E, te, dev), predict(m, S, E, te, dev, oracle_target=True)
            for k, x in score(S, pr, po).items():
                Rloo.setdefault(k, []).extend(x)
            allpred.setdefault(var, {}).update({i: p for i, p in zip(te, pr)})
        tr = [i for i, s in enumerate(S) if s["video"] in seen]
        te = [i for i, s in enumerate(S) if s["video"] in unseen]
        m = train(S, E, tr, var, dev)
        torch.save(dict(model=m.state_dict(), variant=var, n=len(tr)), RUNS / f"wm_seen_{var}.pt")
        Run = {}
        if te:
            pr, po = predict(m, S, E, te, dev), predict(m, S, E, te, dev, oracle_target=True)
            Run = score(S, pr, po)
            allpred.setdefault(var, {}).update({i: p for i, p in zip(te, pr)})
        out[var] = dict(leave_one_run_out=summarise(Rloo), unseen_objects=summarise(Run))
        log(f"WM {var} ({time.time() - t0:.0f}s): LOO next {out[var]['leave_one_run_out'].get('next_top1')} target "
            f"{out[var]['leave_one_run_out'].get('target_ok')} end {out[var]['leave_one_run_out'].get('end_err_cm')}; unseen next "
            f"{out[var]['unseen_objects'].get('next_top1')} target {out[var]['unseen_objects'].get('target_ok')} end {out[var]['unseen_objects'].get('end_err_cm')}")
        json.dump(out, open(RUNS / "eval.json", "w"), indent=1)
    pickle.dump(dict(S=[{k: v for k, v in s.items() if k not in ("vis", "scene")} for s in S], pred=allpred), open(RUNS / "predictions.pkl", "wb"))
    return out


if __name__ == "__main__":
    main(names(sys.argv[1:]) or None)
