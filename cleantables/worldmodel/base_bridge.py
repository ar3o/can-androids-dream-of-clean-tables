"""Bridge between the base world model (runs/uwm/tasks3_v4) and the simulated twin of an episode (data/twin_table/ep/<id>).

The simulator has no phone video, so the model runs in its 3D-only mode (modality 1, V-JEPA tokens masked; trained on
25 % of samples plus all rotated or scaled ones). Object tokens are the simulator's 3D boxes plus each object's own name
and appearance features from the real video (its SAM 3 phrase embedded with MiniLM and the median DINOv2 embedding of
its crops); the table token comes from scene.json; the hand tokens are the two robot palms (extents set to the training
median); the task token is the MiniLM embedding of the episode's prompt.

  prep (VLA env; needs the perception outputs and MiniLM):
      $VLA_PY -m cleantables.worldmodel.base_bridge prep [episode ids | all]   -> data/twin_table/ep/<id>/wm_tokens.npz
  runtime (any env with torch): B = WMBridge("ep/IMG_2533_t01"); out = B.plan(obj_xyz, obj_yaw, palm_xyz)
      Batched over N parallel envs on the GPU. Per env: p_next (N,I) = P(object i is moved next | task); dest (N,I)
      argmax destination per object (object index, or -1 = table); p_dest (N,I,I+1); path (N,I,K,32,3) sampled paths
      (table frame, anchored to the predicted end) for the `topk` most likely next objects, NaN for the others;
      progress (N,) in [0, 1]; spread (N,) ensemble disagreement on p_next.
  offline check (held-out fold models, measured start layouts):
      $VLA_PY -m cleantables.worldmodel.base_bridge check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
EP = ROOT / "data/twin_table/ep"
import os  # noqa: E402
RUN = ROOT / "runs/uwm" / os.environ.get("WM_RUN", "tasks3_v4")      # e.g. WM_RUN=tasks3_v5_fp for the model trained on FoundationPose paths
MAXO = 24


def _scene(ep):
    ep = ep.replace("ep/", "")
    return ep, json.load(open(EP / ep / "scene.json"))


# ------------------------------------------------------------------------------------------------ prep (VLA env)
def prep(ep):
    import pickle
    from cleantables.perception.common import embed, vdir
    ep, S = _scene(ep)
    V = S["video"]
    R = pickle.load(open(vdir(V) / "tracks.pkl", "rb"))
    Z = np.load(vdir(V) / "vision.npz")
    cemb = dict(zip(Z["det"].tolist(), Z["emb"].astype(np.float32)))
    objs = [o for o in S["objects"] if o.get("track") is not None and o["track"] >= 0]
    k0 = int(S["keyframe"])
    phr, crops, ok = [], [], []
    for o in objs:
        tr = R["objects"][o["track"]]
        phr.append(tr["phrase"])
        dets = [(abs(k - k0), d) for k, d in zip(tr["sight_k"].tolist(), tr["det"]) if d in cemb]
        dets.sort(key=lambda x: x[0])
        if dets:
            crops.append(np.median(np.stack([cemb[d] for _, d in dets[:15]]), 0)); ok.append(1)
        else:
            crops.append(np.zeros(768, np.float32)); ok.append(0)
    E = dict(zip(sorted(set(phr) | {"table", S["instruction"]}), embed(sorted(set(phr) | {"table", S["instruction"]}))))
    T = S["table"]
    table_geo = np.array([(T["x"][0] + T["x"][1]) / 2, (T["y"][0] + T["y"][1]) / 2, 0, T["x"][1] - T["x"][0], T["y"][1] - T["y"][0], 0, 0, 1], np.float32)
    # hand extent / axes defaults = medians of the training hand tokens
    d = np.load(ROOT / "data/uwm/tasks3c.npz")
    hg = d["hand_geo"].reshape(-1, 13); hg = hg[hg[:, 0] > 0.5]
    out = dict(ids=np.array([o["id"] for o in objs]), phrase=np.array(phr), role=np.array([o["role"] for o in objs]),
               dest_id=np.array([o.get("dest") or "" for o in objs]), category=np.array([o["category"] for o in objs]),
               extents=np.array([o["extents"] for o in objs], np.float32), yaw_baked=np.array([o.get("yaw_baked", 0.0) for o in objs], np.float32),
               start_pos=np.array([o["pos"] for o in objs], np.float32),
               obj_txt=np.stack([E[p] for p in phr]).astype(np.float16), obj_crop=np.stack(crops).astype(np.float16), obj_crop_ok=np.array(ok, np.uint8),
               table_txt=E["table"].astype(np.float16), table_geo=table_geo, task_emb=E[S["instruction"]].astype(np.float32),
               instruction=np.array(S["instruction"]), hand_ext=np.median(hg[:, 4:7], 0).astype(np.float32))
    np.savez(EP / ep / "wm_tokens.npz", **out)
    return out


# ------------------------------------------------------------------------------------------------ runtime
class WMBridge:
    def __init__(self, ep, ckpts=None, device="cuda", K=8, Z=None):
        from cleantables.worldmodel.base_model import UWM
        self.ep, self.S = _scene(ep)
        if Z is None:
            z = np.load(EP / self.ep / "wm_tokens.npz")
            Z = {k: z[k] for k in z.files}
        self.Z = Z
        self.dev, self.K = device, K
        ckpts = ckpts or [RUN / f"fold-1_seed{s}.pt" for s in range(3)]
        self.models = []
        for c in ckpts:
            m = UWM(path_resid=True).to(device); m.load_state_dict(torch.load(c, map_location=device)); self.models.append(m.eval())
        self.I = len(self.Z["ids"])
        assert self.I + 1 <= MAXO

    def _batch(self, obj_xyz, obj_yaw, palm_xyz):
        """obj_xyz (N,I,3): footprint centre x, y and bottom z (table frame); obj_yaw (N,I): root yaw (rad) added to the baked yaw;
        palm_xyz (N,2,3): robot palm positions (table frame) or None."""
        dev = self.dev
        N, I = obj_xyz.shape[:2]
        ex = torch.as_tensor(self.Z["extents"], device=dev)
        yaw = obj_yaw + torch.as_tensor(self.Z["yaw_baked"], device=dev)[None]
        L, W, H = ex[:, 0].max(ex[:, 1]), ex[:, 0].min(ex[:, 1]), ex[:, 2]
        # box_feat: [x, y, top, max(L,W), min(L,W), top, sin2yaw, cos2yaw]; yaw of the long side: +pi/2 if extents[1] is the long one
        yl = yaw + torch.where(ex[:, 1] > ex[:, 0], torch.full_like(L, np.pi / 2), torch.zeros_like(L))[None]
        top = obj_xyz[..., 2] + H[None]
        geo = torch.zeros(N, MAXO, 8, device=dev)
        geo[:, :I] = torch.stack([obj_xyz[..., 0], obj_xyz[..., 1], top, L[None].expand(N, I), W[None].expand(N, I), top, torch.sin(2 * yl), torch.cos(2 * yl)], -1)
        geo[:, I] = torch.as_tensor(self.Z["table_geo"], device=dev)
        b = dict(obj_geo=geo)
        txt = torch.zeros(MAXO, 384, device=dev); txt[:I] = torch.as_tensor(self.Z["obj_txt"].astype(np.float32), device=dev)
        txt[I] = torch.as_tensor(self.Z["table_txt"].astype(np.float32), device=dev)
        crop = torch.zeros(MAXO, 768, device=dev); crop[:I] = torch.as_tensor(self.Z["obj_crop"].astype(np.float32), device=dev)
        cok = torch.zeros(MAXO, dtype=torch.bool, device=dev); cok[:I] = torch.as_tensor(self.Z["obj_crop_ok"] > 0, device=dev)
        om = torch.zeros(MAXO, dtype=torch.bool, device=dev); om[:I + 1] = True
        kind = torch.zeros(MAXO, dtype=torch.long, device=dev); kind[I] = 1
        b.update(obj_txt=txt[None].expand(N, -1, -1), obj_crop=crop[None].expand(N, -1, -1), obj_crop_ok=cok[None].expand(N, -1),
                 obj_mask=om[None].expand(N, -1), obj_kind=kind[None].expand(N, -1))
        hg = torch.zeros(N, 2, 13, device=dev)
        if palm_xyz is not None:
            o = palm_xyz[..., 0].argsort(-1)                                  # training slots sorted by x
            p = torch.gather(palm_xyz, 1, o[..., None].expand(-1, -1, 3))
            hg[..., 0] = 1; hg[..., 1:4] = p; hg[..., 4:7] = torch.as_tensor(self.Z["hand_ext"], device=dev)
            hg[..., 7:10] = torch.tensor([0.0, 1.0, 0.0], device=dev); hg[..., 10:13] = torch.tensor([1.0, 0.0, 0.0], device=dev)
        b.update(hand_geo=hg, hand_q16=torch.zeros(N, 2, 16, device=dev), hand_q_ok=torch.zeros(N, 2, dtype=torch.bool, device=dev),
                 vj_tok=torch.zeros(N, 32, 1024, device=dev), vj_ok=torch.zeros(N, dtype=torch.bool, device=dev),
                 task_emb=torch.as_tensor(self.Z["task_emb"], device=dev)[None].expand(N, -1))
        return b

    @torch.no_grad()
    def plan(self, obj_xyz, obj_yaw=None, palm_xyz=None, topk=2):
        obj_xyz = torch.as_tensor(obj_xyz, dtype=torch.float32, device=self.dev)
        N, I = obj_xyz.shape[:2]
        obj_yaw = torch.zeros(N, I, device=self.dev) if obj_yaw is None else torch.as_tensor(obj_yaw, dtype=torch.float32, device=self.dev)
        palm_xyz = None if palm_xyz is None else torch.as_tensor(palm_xyz, dtype=torch.float32, device=self.dev)
        b = self._batch(obj_xyz, obj_yaw, palm_xyz)
        mode = torch.ones(N, dtype=torch.long, device=self.dev)               # 3D-only
        if getattr(self.models[0], "endref", False):                          # models whose end head predicts against reference objects
            return self._plan_endref(b, mode, N, I, topk)
        pm, pd, prog, paths = [], [], [], []
        for m in self.models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                ml, _ = m.pointers(E, torch.zeros(N, dtype=torch.long, device=self.dev))
                pm.append(F.softmax(ml.float(), -1)[:, :I])
                prog.append(torch.sigmoid(m.p_vid(E["cls"][:, 0]).float().squeeze(-1)))
                dl = []
                for i in range(I):
                    mv = torch.full((N,), i, dtype=torch.long, device=self.dev)
                    _, tl = m.pointers(E, mv)
                    tl = tl.float()
                    dl.append(F.softmax(torch.cat([tl[:, :I], tl[:, 24:25]], -1), -1))   # objects + TABLE
                pd.append(torch.stack(dl, 1))
            self._E, self._m = E, m
        P = torch.stack(pm).mean(0); Pd = torch.stack(pd).mean(0)
        out = dict(p_next=P, spread=torch.stack(pm).std(0).sum(-1), p_dest=Pd, progress=torch.stack(prog).mean(0))
        dest = Pd.argmax(-1); out["dest"] = torch.where(dest == I, torch.full_like(dest, -1), dest)
        # sampled paths for the top-k next objects (K samples per ensemble member)
        path = torch.full((N, I, len(self.models) * self.K, 32, 3), float("nan"), device=self.dev)
        top = P.topk(min(topk, I), -1).indices
        for j in range(top.shape[1]):
            mv = top[:, j]
            tg = torch.where(out["dest"].gather(1, mv[:, None]).squeeze(1) >= 0, out["dest"].gather(1, mv[:, None]).squeeze(1), torch.full_like(mv, 24))
            ps = []
            for m in self.models:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                    c, _, _ = m.heads_from(E, mv, tg)
                    ps.append(m.sample_paths_anchored(E, c, K=self.K).float()[..., :3])
            org = b["obj_geo"][torch.arange(N), mv, :3]
            path[torch.arange(N), mv] = torch.cat(ps, 1) + org[:, None, None]
        out["path"] = path
        return out


    def _plan_endref(self, b, mode, N, I, topk):
        """As plan, for endref models: p_dest / dest come from the end mixture's references (no destination pointer), and the
        sampled paths end at the ensemble's single goal (see cleantables.worldmodel.bridge.endref_goal / endref_paths)."""
        from cleantables.worldmodel.bridge import dest_from_ref, endref_goal
        Es, pm, prog = [], [], []
        for m in self.models:
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]; Es.append(E)
            ml, _ = m.pointers(E, torch.zeros(N, dtype=torch.long, device=self.dev))
            pm.append(F.softmax(ml.float(), -1)[:, :I]); prog.append(torch.sigmoid(m.p_vid(E["cls"][:, 0]).float().squeeze(-1)))
        P = torch.stack(pm).mean(0)
        tab = (b["obj_kind"] == 1).float().argmax(-1)
        pd, dest, goals = [], [], []
        for i in range(I):
            mv = torch.full((N,), i, dtype=torch.long, device=self.dev)
            g, j, a, _ = endref_goal(self.models, Es, mv)
            pd.append(torch.cat([a[:, :I], a[torch.arange(N, device=self.dev), tab][:, None]], -1)); dest.append(dest_from_ref(j, mv, b["obj_kind"])); goals.append(g)
        out = dict(p_next=P, spread=torch.stack(pm).std(0).sum(-1), p_dest=torch.stack(pd, 1), progress=torch.stack(prog).mean(0),
                   dest=torch.stack(dest, 1))
        path = torch.full((N, I, len(self.models) * self.K, 32, 3), float("nan"), device=self.dev)
        top = P.topk(min(topk, I), -1).indices
        for jj in range(top.shape[1]):
            mv = top[:, jj]; g = torch.stack(goals, 1)[torch.arange(N, device=self.dev), mv]
            ps = [m.plan_paths(E, mv, K=self.K, end=g)[0].float()[..., :3] for m, E in zip(self.models, Es)]
            org = b["obj_geo"][torch.arange(N), mv, :3]
            path[torch.arange(N), mv] = torch.cat(ps, 1) + org[:, None, None]
        out["path"] = path
        return out


def mixed_tokens(base_ep, extra_ep, prompt_ep, roles=("task", "container")):
    """Mixed scene: the base twin's objects plus the extra twin's task objects and container (each with tokens from its own video);
    the prompt embedding is taken from prompt_ep. Objects of the other task get role 'obstacle'. Returns a token dict for WMBridge(Z=...)."""
    zb = dict(np.load(EP / base_ep / "wm_tokens.npz")); ze = dict(np.load(EP / extra_ep / "wm_tokens.npz")); zp = dict(np.load(EP / prompt_ep / "wm_tokens.npz"))
    sel = np.isin(ze["role"], roles)
    Z = {}
    for k in ("ids", "phrase", "role", "dest_id", "category", "extents", "yaw_baked", "start_pos", "obj_txt", "obj_crop", "obj_crop_ok"):
        Z[k] = np.concatenate([zb[k], ze[k][sel]])
    Z["ids"] = np.array([f"{i}" for i in Z["ids"]])
    for k in ("table_txt", "table_geo", "hand_ext"):
        Z[k] = zb[k]
    Z["task_emb"], Z["instruction"] = zp["task_emb"], zp["instruction"]
    prompted = set(zp["role"][zp["role"] == "task"].tolist()) and {c.split("/")[0] for c in zp["category"][zp["role"] == "task"]}
    Z["role"] = np.array([r if (r != "task" or c.split("/")[0] in prompted) else "obstacle" for r, c in zip(Z["role"], Z["category"])])
    return Z


def place_free(Z, table_geo, n_new, rng, tries=400):
    """Random non-overlapping footprint positions on the table for the last n_new objects (the others keep start_pos)."""
    P = Z["start_pos"].copy(); ex = Z["extents"]
    cx, cy, _, Lx, Ly = table_geo[:5]
    keep = list(range(len(P) - n_new))
    for i in range(len(P) - n_new, len(P)):
        r_i = 0.5 * np.hypot(ex[i, 0], ex[i, 1])
        for _ in range(tries):
            q = np.array([cx + (rng.random() - 0.5) * (Lx - 2 * r_i - 0.1), cy + (rng.random() - 0.5) * (Ly - 2 * r_i - 0.1), 0.0])
            if all(np.hypot(*(q[:2] - P[j, :2])) > r_i + 0.5 * np.hypot(ex[j, 0], ex[j, 1]) + 0.02 for j in keep):
                P[i] = q; keep.append(i); break
        else:
            return None
    return P


def mixed_check(n_layouts=8, seed=0):
    """Prompt following in mixed scenes (tools, fruit and both containers on one table), using the held-out fold of the base episode."""
    from cleantables.worldmodel.tasks3_train import NF
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    tools = [t["id"] for t in T if t["task"] == "tools"]; fruit = [t["id"] for t in T if t["task"] == "fruit"]
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(fruit)
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    rng = np.random.default_rng(seed)
    res = {}
    for base_list, extra_list, bname in ((tools, fruit, "tools_base+fruit"), (fruit, tools, "fruit_base+tools")):
        for prompt_kind in ("tools", "fruit"):
            hits, dok, n = [], [], 0
            for b in base_list:
                e = extra_list[rng.integers(len(extra_list))]
                pe = b if (prompt_kind == "tools") == (b in tools) else e
                Z = mixed_tokens(b, e, pe)
                n_new = len(Z["ids"]) - len(np.load(EP / b / "wm_tokens.npz")["ids"])
                f = fold_of[b] if b.startswith("IMG_2516") else fold_of[b.split("_t")[0]]
                B = WMBridge(b, ckpts=[RUN / f"fold{f}_seed{s}.pt" for s in range(3)], Z=Z)
                Ps = [place_free(Z, Z["table_geo"], n_new, rng) for _ in range(n_layouts)]
                Ps = [p for p in Ps if p is not None]
                if not Ps:
                    continue
                o = B.plan(torch.as_tensor(np.stack(Ps)), topk=1)
                cls_want = "tool" if prompt_kind == "tools" else "fruit"
                for j in range(len(Ps)):
                    i = int(o["p_next"][j].argmax()); cat = str(Z["category"][i]).split("/")[0]
                    hits.append(cat == cls_want)
                    if cat == cls_want:
                        dst = int(o["dest"][j, i]); dn = str(Z["ids"][dst]) if dst >= 0 else "table"
                        dok.append(Z["dest_id"][i] != "" and Z["dest_id"][i] in dn)
                n += len(Ps)
            cats = None
            res[f"{bname} | prompt={prompt_kind}"] = dict(n=n, pick_prompted_class=float(np.mean(hits)), dest_ok_given_right_class=float(np.mean(dok)) if dok else None)
            print(f"{bname:18s} prompt={prompt_kind:5s} n={n}  picks the PROMPTED class {np.mean(hits):.2f}  dest ok {np.mean(dok) if dok else float('nan'):.2f}", flush=True)
    json.dump(res, open(RUN / "bridge_mixed_check.json", "w"), indent=1)
    return res


# ------------------------------------------------------------------------------------------------ offline check
def check():
    """For every twin, run the held-out fold ensemble (the fold that never saw the episode) on the measured start layout and check
    whether the most likely next object is a task object (role 'task', or 'stack' for stacking) and its destination the right container."""
    from cleantables.worldmodel.tasks3_train import NF
    import os
    os.environ.setdefault("TASKS3_DATA", "tasks3c")
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    # fold of each episode, as in tasks3_train.load
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(t["id"] for t in T if t["video"] == "IMG_2516")
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    rows = []
    for t in T:
        ep = t["id"]
        if not (EP / ep / "wm_tokens.npz").exists():
            continue
        f = fold_of[ep] if t["video"] == "IMG_2516" else fold_of[t["video"]]
        B = WMBridge(ep, ckpts=[RUN / f"fold{f}_seed{s}.pt" for s in range(3)])
        Z = B.Z
        xyz = torch.as_tensor(Z["start_pos"])[None]
        o = B.plan(xyz, topk=1)
        p = o["p_next"][0].cpu().numpy()
        want = "stack" if t["task"] == "stack_boxes" else "task"
        roles = Z["role"]
        i = int(p.argmax())
        dst = int(o["dest"][0, i]); dname = Z["ids"][dst] if dst >= 0 else "table"
        good_dest = (Z["dest_id"][i] != "" and Z["dest_id"][i] in dname) if want == "task" else None
        rows.append(dict(ep=ep, task=t["task"], pick=str(Z["ids"][i]), pick_role=str(roles[i]), pick_is_task=bool(roles[i] == want),
                         chance=float((roles == want).mean()), p_task_mass=float(p[roles == want].sum()), dest=str(dname), dest_ok=good_dest,
                         progress=float(o["progress"][0]), path_end=np.nanmean(o["path"][0, i, :, -1].cpu().numpy(), 0).round(3).tolist()))
        print(rows[-1], flush=True)
    json.dump(rows, open(RUN / "bridge_check.json", "w"), indent=1)
    for task in ("fruit", "tools", "stack_boxes"):
        r = [x for x in rows if x["task"] == task]
        if r:
            dd = [x["dest_ok"] for x in r if x["dest_ok"] is not None]
            print(task, f"n={len(r)} pick is a task object {np.mean([x['pick_is_task'] for x in r]):.2f} (chance {np.mean([x['chance'] for x in r]):.2f}),"
                  f" task prob mass {np.mean([x['p_task_mass'] for x in r]):.2f}, dest ok {np.mean(dd) if dd else float('nan'):.2f},"
                  f" start progress {np.mean([x['progress'] for x in r]):.2f}")


# ------------------------------------------------------------------------------------------------ world model as goal-setter: goal accuracy
def real_ends(ep):
    """{twin object id: (start xyz_top, real end xyz_top or None)} from the perception events in the episode window (the human's outcome)."""
    ep, S = _scene(ep)
    from cleantables.perception.common import vdir
    Ev = json.load(open(vdir(S["video"]) / "events.json"))
    T = {t["id"]: t for t in json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]}[ep]
    k0, k1 = int(T["core_start_s"] * 10) - 10, int(T["core_end_s"] * 10) + 10
    src2id = {o["src"]: o["id"] for o in Ev["objects"]}
    out = {}
    for o in S["objects"]:
        if o.get("track") is None or o["track"] < 0:
            continue
        oid = src2id.get(o["track"])
        mv = sorted([e for e in Ev["events"] if e.get("kind", "move") == "move" and e.get("obj") == oid and e.get("end") and k0 <= e["f_start"] <= k1],
                    key=lambda e: e["f_start"])
        end = np.array([mv[-1]["end"]["x"], mv[-1]["end"]["y"], mv[-1]["end"]["top"]]) if mv else None
        out[o["id"]] = end
    return out


@torch.no_grad()
def predicted_ends(B, xyz):
    """World-model goal per object, as if it were moved next: destination = pointer argmax, end = mean of the anchored sampled
    path ends (table frame, top centre). xyz (1,I,3) are bottom-centre positions."""
    b = B._batch(torch.as_tensor(xyz, dtype=torch.float32, device=B.dev), torch.zeros(xyz.shape[:2], device=B.dev), None)
    mode = torch.ones(1, dtype=torch.long, device=B.dev)
    I = B.I; ends = np.zeros((I, 3)); dest = np.zeros(I, int); pn = []
    for m in B.models:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            E = m.encode(b, mode); E["kind"] = b["obj_kind"]
            ml, _ = m.pointers(E, torch.zeros(1, dtype=torch.long, device=B.dev))
        pn.append(F.softmax(ml.float(), -1)[0, :I].cpu().numpy())
    for i in range(I):
        mv = torch.tensor([i], device=B.dev); dl, es = [], []
        for m in B.models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                _, tl = m.pointers(E, mv)
            dl.append(F.softmax(torch.cat([tl.float()[:, :I], tl.float()[:, 24:25]], -1), -1))
        d = int(torch.stack(dl).mean(0).argmax(-1)); dest[i] = d
        tg = torch.tensor([d if d < I else 24], device=B.dev)
        for m in B.models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                c, _, _ = m.heads_from(E, mv, tg)
                es.append(m.sample_paths_anchored(E, c, K=8).float()[0, :, -1, :3].mean(0))
        ends[i] = (torch.stack(es).mean(0) + b["obj_geo"][0, i, :3]).cpu().numpy()
    return ends, dest, np.mean(pn, 0)


def goal_check():
    from cleantables.worldmodel.tasks3_train import NF
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(t["id"] for t in T if t["video"] == "IMG_2516")
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    rows = []
    for t in T:
        if t["task"] not in ("tools", "fruit"):
            continue
        ep = t["id"]; f = fold_of[ep] if ep.startswith("IMG_2516") else fold_of[t["video"]]
        B = WMBridge(ep, ckpts=[RUN / f"fold{f}_seed{s}.pt" for s in range(3)])
        Z = B.Z; xyz = Z["start_pos"][None]
        ends, dest, pn = predicted_ends(B, xyz)
        RE = real_ends(ep)
        S = B.S; ext = {o["id"]: o["extents"] for o in S["objects"]}
        top0 = xyz[0].copy(); top0[:, 2] += np.array([ext[i][2] for i in Z["ids"]])
        for i, oid in enumerate(Z["ids"]):
            re = RE.get(str(oid))
            moved = re is not None and np.linalg.norm(re[:2] - top0[i, :2]) > 0.05
            dname = str(Z["ids"][dest[i]]) if dest[i] < len(Z["ids"]) else "table"
            rows.append(dict(ep=ep, task=t["task"], obj=str(oid), role=str(Z["role"][i]), moved=bool(moved), p_next=float(pn[i]),
                             wm_dest=dname, wm_disp=float(np.linalg.norm(ends[i, :2] - top0[i, :2])),
                             err_xy=float(np.linalg.norm(ends[i, :2] - re[:2])) if moved else None,
                             stay_err=float(np.linalg.norm(top0[i, :2] - re[:2])) if moved else None))
        print(ep, "done", flush=True)
    json.dump(rows, open(RUN / "wm_goal_check.json", "w"), indent=1)
    for task in ("tools", "fruit"):
        R = [r for r in rows if r["task"] == task]
        mv = [r for r in R if r["moved"]]; st = [r for r in R if not r["moved"]]
        e = np.array([r["err_xy"] for r in mv]); s0 = np.array([r["stay_err"] for r in mv])
        print(f"== {task}: {len(mv)} objects really moved, {len(st)} stayed")
        print(f"   WM end vs REAL end (moved objects): median {100*np.median(e):.1f} cm, <5 cm {np.mean(e<0.05):.2f}, <10 cm {np.mean(e<0.10):.2f}"
              f"   (if it stayed put: median {100*np.median(s0):.1f} cm)")
        print(f"   WM predicted displacement: moved objects median {100*np.median([r['wm_disp'] for r in mv]):.1f} cm; stayed objects median {100*np.median([r['wm_disp'] for r in st]):.1f} cm")
        print(f"   P(next) mass at start: moved objects mean {np.mean([r['p_next'] for r in mv]):.2f} vs stayed {np.mean([r['p_next'] for r in st]):.2f}")
        print(f"   WM destination of moved objects: {dict(__import__('collections').Counter(r['wm_dest'] for r in mv).most_common(5))}")


# ------------------------------------------------------------------------------------------------ world-model-only rollout on randomised layouts
def shuffle_layout(Z, S, rng, idx_move, tries=300, margin=0.02):
    """Random non-overlapping positions on the table for the objects idx_move (the others keep their measured place); circular footprints."""
    P = Z["start_pos"].copy(); ex = Z["extents"]
    T = S["table"]; x0, x1, y0, y1 = T["x"][0], T["x"][1], T["y"][0], T["y"][1]
    keep = [j for j in range(len(P)) if j not in idx_move]
    for i in idx_move:
        r_i = 0.5 * np.hypot(ex[i, 0], ex[i, 1])
        for _ in range(tries):
            q = np.array([rng.uniform(x0 + r_i + 0.03, x1 - r_i - 0.03), rng.uniform(y0 + r_i + 0.03, y1 - r_i - 0.03), 0.0], np.float32)
            if all(np.hypot(*(q[:2] - P[j, :2])) > r_i + 0.5 * np.hypot(ex[j, 0], ex[j, 1]) + margin for j in keep):
                P[i] = q; keep.append(i); break
        else:
            return None
    return P


@torch.no_grad()
def wm_rollout(B, pos, max_steps=None):
    """Sequential world-model plan executed kinematically: pick argmax P(next), place it at its predicted end (destination + offset),
    re-plan. Stops when the top choice was already moved, or all 'task' objects were moved and the next choice is not a task object."""
    from cleantables.worldmodel.base_model import PS
    Z = B.Z; I = B.I; roles = [str(r) for r in Z["role"]]; H = Z["extents"][:, 2]
    n_task = sum(r == "task" for r in roles)
    pos = pos.copy(); moved = []; dests = []
    for step in range(max_steps or 2 * I):
        b = B._batch(torch.tensor(pos[None], device=B.dev), torch.zeros(1, I, device=B.dev), None)
        mode = torch.ones(1, dtype=torch.long, device=B.dev)
        Es, pn = [], []
        for m in B.models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                ml, _ = m.pointers(E, torch.zeros(1, dtype=torch.long, device=B.dev))
            Es.append(E); pn.append(F.softmax(ml.float(), -1)[0, :I])
        P = torch.stack(pn).mean(0).cpu().numpy(); i = int(P.argmax())
        if i in moved:
            return pos, moved, dests, "top choice already moved"
        if sum(roles[j] == "task" for j in moved) >= n_task and roles[i] != "task":
            return pos, moved, dests, "all task objects moved"
        mv = torch.tensor([i], device=B.dev); dp = []
        for m, E in zip(B.models, Es):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, tl = m.pointers(E, mv)
            tlf = tl.float(); dp.append(F.softmax(torch.cat([tlf[:, :I], tlf[:, 24:25]], -1), -1))
        d = int(torch.stack(dp).mean(0).argmax(-1)); tg = torch.tensor([d if d < I else 24], device=B.dev)
        ends = []
        for m, E in zip(B.models, Es):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, _, mdn = m.heads_from(E, mv, tg); z = m.end_disp(c)
            tm = int(mdn["logit"].float().argmax(-1))
            if d < I:
                ends.append((b["obj_geo"][0, d, :3] + 0.1 * mdn["mu"].float()[0, tm]).cpu().numpy())
            else:
                tz = int(z["logit"].float().argmax(-1)); ends.append((b["obj_geo"][0, i, :3] + z["mu"].float()[0, tz] * PS).cpu().numpy())
        end = np.mean(ends, 0)
        pos[i] = [end[0], end[1], max(0.0, end[2] - H[i])]
        moved.append(i); dests.append(str(Z["ids"][d]) if d < I else "table")
    return pos, moved, dests, "step budget"


def rollout_check(n_layouts=20, seed=0, final=False):
    from cleantables.worldmodel.tasks3_train import NF
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(t["id"] for t in T if t["video"] == "IMG_2516")
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    rng = np.random.default_rng(seed); rows = []
    for t in T:
        if t["task"] not in ("tools", "fruit"):
            continue
        ep = t["id"]; f = fold_of[ep] if ep.startswith("IMG_2516") else fold_of[t["video"]]
        B = WMBridge(ep, ckpts=None if final else [RUN / f"fold{f}_seed{s}.pt" for s in range(3)])
        Z = B.Z; S = B.S; roles = [str(r) for r in Z["role"]]; ids = [str(x) for x in Z["ids"]]
        task = [i for i, r in enumerate(roles) if r == "task"]
        cont = {str(o["id"]): o for o in S["objects"]}
        for L in range(n_layouts):
            P0 = Z["start_pos"].copy() if L == 0 else shuffle_layout(Z, S, rng, task)
            if P0 is None:
                continue
            Pend, moved, dests, why = wm_rollout(B, P0)
            ok = []
            for i in task:
                dn = str(Z["dest_id"][i]); c = cont.get(dn)
                if c is None:
                    ok.append(False); continue
                ci = ids.index(dn); cp = Pend[ci]; r_in = 0.5 * min(c["extents"][0], c["extents"][1]) - 0.01
                ok.append(bool(np.hypot(*(Pend[i, :2] - cp[:2])) < r_in))
            wrong = [ids[j] for j in moved if roles[j] != "task"]
            rows.append(dict(ep=ep, task=t["task"], layout="measured" if L == 0 else "random", n_task=len(task), in_correct=int(sum(ok)),
                             all_ok=bool(all(ok)), moved_non_task=wrong, n_moves=len(moved), stop=why, dests=dests))
        print(ep, "done", flush=True)
    tag = "final" if final else "heldout"
    json.dump(rows, open(RUN / f"wm_rollout_check_{tag}.json", "w"), indent=1)
    for task in ("fruit", "tools"):
        for lay in ("measured", "random"):
            R = [r for r in rows if r["task"] == task and r["layout"] == lay]
            if not R:
                continue
            obj = sum(r["in_correct"] for r in R) / max(1, sum(r["n_task"] for r in R))
            print(f"== {task:5s} {lay:8s} layouts {len(R):3d}: objects ending in the CORRECT container {obj:.2f};  whole scene correct "
                  f"{np.mean([r['all_ok'] for r in R]):.2f};  scenes where it moved a non-task object {np.mean([len(r['moved_non_task']) > 0 for r in R]):.2f};"
                  f"  stop reasons {dict(__import__('collections').Counter(r['stop'] for r in R))}")


def teacher_check(n_layouts=20, seed=0):
    """Goal check as used for RL rewards: at the start state take every in-scope object's world-model goal (as if it were moved),
    place all objects at their goals at once, and test whether each ends inside its labelled container. Held-out folds, measured
    and random layouts."""
    from cleantables.worldmodel.tasks3_train import NF
    from cleantables.worldmodel.base_model import PS
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(t["id"] for t in T if t["video"] == "IMG_2516")
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    rng = np.random.default_rng(seed); rows = []
    for t in T:
        if t["task"] not in ("tools", "fruit"):
            continue
        ep = t["id"]; f = fold_of[ep] if ep.startswith("IMG_2516") else fold_of[t["video"]]
        B = WMBridge(ep, ckpts=[RUN / f"fold{f}_seed{s}.pt" for s in range(3)])
        Z = B.Z; S = B.S; I = B.I; roles = [str(r) for r in Z["role"]]; ids = [str(x) for x in Z["ids"]]
        task = [i for i, r in enumerate(roles) if r == "task"]; cont = {str(o["id"]): o for o in S["objects"]}
        for L in range(n_layouts):
            P0 = Z["start_pos"].copy() if L == 0 else shuffle_layout(Z, S, rng, task)
            if P0 is None:
                continue
            with torch.no_grad():
                b = B._batch(torch.tensor(P0[None], device=B.dev), torch.zeros(1, I, device=B.dev), None)
                mode = torch.ones(1, dtype=torch.long, device=B.dev)
                Es = []
                for m in B.models:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                    Es.append(E)
                ok, dists = [], []
                for i in task:
                    mv = torch.tensor([i], device=B.dev); dp = []
                    for m, E in zip(B.models, Es):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, tl = m.pointers(E, mv)
                        tlf = tl.float(); dp.append(F.softmax(torch.cat([tlf[:, :I], tlf[:, 24:25]], -1), -1))
                    d = int(torch.stack(dp).mean(0).argmax(-1)); tg = torch.tensor([d if d < I else 24], device=B.dev)
                    ends = []
                    for m, E in zip(B.models, Es):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            c, _, mdn = m.heads_from(E, mv, tg); z = m.end_disp(c)
                        tm = int(mdn["logit"].float().argmax(-1))
                        ends.append((b["obj_geo"][0, d, :3] + 0.1 * mdn["mu"].float()[0, tm]).cpu().numpy() if d < I else
                                    (b["obj_geo"][0, i, :3] + z["mu"].float()[0, int(z["logit"].float().argmax(-1))] * PS).cpu().numpy())
                    end = np.mean(ends, 0)
                    dn = str(Z["dest_id"][i]); c = cont.get(dn)
                    cp = P0[ids.index(dn)] if dn in ids else None
                    r_in = 0.5 * min(c["extents"][0], c["extents"][1]) - 0.01 if c else 0
                    dd = float(np.hypot(*(end[:2] - cp[:2]))) if cp is not None else 9.0
                    ok.append(dd < r_in); dists.append(dd)
            rows.append(dict(ep=ep, task=t["task"], layout="measured" if L == 0 else "random", n=len(task), ok=int(sum(ok)), all_ok=bool(all(ok)), d=dists))
        print(ep, "done", flush=True)
    json.dump(rows, open(RUN / "wm_teacher_check.json", "w"), indent=1)
    for task in ("fruit", "tools"):
        for lay in ("measured", "random"):
            R = [r for r in rows if r["task"] == task and r["layout"] == lay]
            if R:
                dd = np.concatenate([r["d"] for r in R])
                print(f"== TEACHER {task:5s} {lay:8s} layouts {len(R):3d}: object goals INSIDE the correct container {sum(r['ok'] for r in R) / sum(r['n'] for r in R):.2f};"
                      f"  every object correct {np.mean([r['all_ok'] for r in R]):.2f};  goal-to-container-centre median {100 * np.median(dd):.1f} cm")


def path_check(n_layouts=10, seed=0):
    """Share of sampled carries that pass through another object, for every in-scope object (held-out folds, measured and random
    layouts). The path is the median of the ensemble's anchored samples, re-anchored to the world-model goal as RL uses it. A carry
    collides if, on path fraction 0.15-0.8, the object's bottom is below another object's top while over its rotated footprint.
    Also reports the median lift."""
    from cleantables.worldmodel.tasks3_train import NF
    from cleantables.worldmodel.base_model import PS
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    box = sorted({t["video"] for t in T if t["video"] != "IMG_2516"}); fru = sorted(t["id"] for t in T if t["video"] == "IMG_2516")
    fold_of = {v: i % NF for i, v in enumerate(box)} | {e: i % NF for i, e in enumerate(fru)}
    rng = np.random.default_rng(seed); res = {}
    for t in T:
        if t["task"] not in ("tools", "fruit"):
            continue
        ep = t["id"]; f = fold_of[ep] if ep.startswith("IMG_2516") else fold_of[t["video"]]
        B = WMBridge(ep, ckpts=[RUN / f"fold{f}_seed{s}.pt" for s in range(3)])
        Z = B.Z; S = B.S; I = B.I; roles = [str(r) for r in Z["role"]]; H = Z["extents"][:, 2]
        task = [i for i, r in enumerate(roles) if r == "task"]
        for L in range(n_layouts):
            P0 = Z["start_pos"].copy() if L == 0 else shuffle_layout(Z, S, rng, task)
            if P0 is None:
                continue
            with torch.no_grad():
                b = B._batch(torch.tensor(P0[None], device=B.dev), torch.zeros(1, I, device=B.dev), None)
                mode = torch.ones(1, dtype=torch.long, device=B.dev)
                Es = []
                for m in B.models:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        E = m.encode(b, mode); E["kind"] = b["obj_kind"]
                    Es.append(E)
                geo = b["obj_geo"][0].float().cpu().numpy()
                for i in task:
                    mv = torch.tensor([i], device=B.dev); dp = []
                    for m, E in zip(B.models, Es):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, tl = m.pointers(E, mv)
                        tlf = tl.float(); dp.append(F.softmax(torch.cat([tlf[:, :I], tlf[:, 24:25]], -1), -1))
                    d = int(torch.stack(dp).mean(0).argmax(-1)); tg = torch.tensor([d if d < I else 24], device=B.dev)
                    ends, paths = [], []
                    for m, E in zip(B.models, Es):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            c, _, mdn = m.heads_from(E, mv, tg)
                            ps = m.sample_paths_anchored(E, c, K=8).float()[0, :, :, :3]
                        tm = int(mdn["logit"].float().argmax(-1))
                        if d < I:
                            ends.append((b["obj_geo"][0, d, :3] + 0.1 * mdn["mu"].float()[0, tm]).cpu().numpy())
                        paths.append(ps)
                    path = torch.cat(paths, 0).median(0).values.cpu().numpy() + geo[i, :3]
                    if ends:
                        end = np.mean(ends, 0); path = path + np.linspace(0, 1, len(path))[:, None] * (end - path[-1])
                    tt = np.linspace(0, 1, len(path)); seg = (tt >= 0.15) & (tt <= 0.8)
                    zb = path[:, 2] - H[i]
                    hit = False
                    for j in range(I):
                        if j == i:
                            continue
                        y = 0.5 * np.arctan2(geo[j, 6], geo[j, 7]); dx = path[:, 0] - geo[j, 0]; dy = path[:, 1] - geo[j, 1]
                        lx = np.abs(np.cos(y) * dx + np.sin(y) * dy); ly = np.abs(-np.sin(y) * dx + np.cos(y) * dy)
                        over = (lx < geo[j, 3] / 2) & (ly < geo[j, 4] / 2)
                        if np.any(seg & over & (zb < geo[j, 2] - 0.005)):
                            hit = True; break
                    lift = float(path[:, 2].max() - max(path[0, 2], path[-1, 2]))
                    r = res.setdefault(t["task"], dict(n=0, hit=0, lift=[]))
                    r["n"] += 1; r["hit"] += int(hit); r["lift"].append(lift)
        print(ep, "done", flush=True)
    for task, r in res.items():
        print(f"== PATHS {RUN.name} {task:5s}: {r['n']} imagined carries, passing THROUGH another object {r['hit'] / r['n']:.2f};"
              f" lift above start/end median {100 * np.median(r['lift']):.1f} cm (p90 {100 * np.percentile(r['lift'], 90):.1f})")
    json.dump({k: dict(n=v["n"], hit=v["hit"], lift_med=float(np.median(v["lift"]))) for k, v in res.items()}, open(RUN / "path_check.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
        eps = [t["id"] for t in T] if len(sys.argv) < 3 or sys.argv[2] == "all" else sys.argv[2:]
        for e in eps:
            try:
                o = prep(e); print(e, len(o["ids"]), "objects", list(zip(o["ids"].tolist(), o["phrase"].tolist())), flush=True)
            except Exception as ex:
                print(e, "FAILED", repr(ex), flush=True)
    elif sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "mixed":
        mixed_check()
    elif sys.argv[1] == "goals":
        goal_check()
    elif sys.argv[1] == "paths":
        path_check()
    elif sys.argv[1] == "teacher":
        teacher_check()
    elif sys.argv[1] == "rollout":
        rollout_check(final=len(sys.argv) > 2 and sys.argv[2] == "final")



