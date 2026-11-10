"""Cache per-object predictions of stored world-model checkpoints on the held-out test set (CPU, read-only).

Nothing here scores; cleantables.worldmodel.eval.score applies the same metric code to every method.
  $VLA_PY -m cleantables.worldmodel.eval.predict gt           -> out/wm/gt.npz (targets, tokens, previous-keyframe positions, references)
  $VLA_PY -m cleantables.worldmodel.eval.predict final_s2     -> out/wm/pred_final_s2.npz (label-trained model; rows with a known next object)
  $VLA_PY -m cleantables.worldmodel.eval.predict run <name> <run_dir> <fold> <seeds> [mode=1nh|0nh|0] [noapp] [text=shuffle|prompts] [judge]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.manual_seed(0)
from cleantables.worldmodel.common import OUT, ROOT
from cleantables.perception.contact.moves import intervals
from cleantables.worldmodel.bridge import load_models, watch_readout
from cleantables.worldmodel.data import Data
from cleantables.worldmodel.evaluate import Batcher, forward

EB = Path(__file__).resolve().parents[3] / "outputs" / "eval"; O = EB / "out" / "wm"; O.mkdir(parents=True, exist_ok=True)
W = OUT / "wm_v2_world"
IN_SCOPE = {"Put the fruit into the fruit bowl.": "fruit in", "Take the fruit out of the bowl and spread it on the table.": "fruit out",
            "Put the tools into the toolbox.": "tools in"}
D = Data(text="minilm"); Bt = Batcher(D)
test = D.test_idx
ep = D.ep_base.astype(str); vid = D.S.d["video"].astype(str); kk = D.S.d["snap_k"].astype(int); TT = D.S.d["task_text"].astype(str)
TS = np.asarray(D.track_set).astype(str)
BS = 64


def blank(b):                                                               # mode 1nh: no hands, no contact input
    b["hand_geo"] = b["hand_geo"].clone(); b["hand_geo"][..., 0] = 0; b["hand_q_ok"] = torch.zeros_like(b["hand_q_ok"])
    for k_ in ("obj_contact_ok", "obj_contact2_ok"):
        if k_ in b:
            b[k_] = torch.zeros_like(b[k_])
    return b


def noapp(b):                                                               # appearance off: crop features zeroed
    b["app_idx"] = torch.full_like(b["app_idx"], -1)
    b["obj_crop"] = torch.zeros_like(b["obj_crop"]); b["obj_crop_ok"] = torch.zeros_like(b["obj_crop_ok"])
    return b


# ---------------------------------------------------------------- twin labels (privileged; used only by the task rule and the prompt ablations)
def twin_scene(base_ep):
    for root in ("_relink8", "_scalefix"):
        p = ROOT / "data/twin_table/ep" / root / base_ep / "scene.json"
        if p.exists():
            return json.load(open(p)), root
    return None, None


def twin_labels():
    """{base_ep: {track id: (role, category, twin folder)}} from each test episode's own twin (the relinked twin where it exists)."""
    out = {}
    for e in sorted(set(ep[test])):
        s, root = twin_scene(e)
        if s is not None:
            out[e] = {int(o["track"]): (o.get("role", ""), o.get("category", ""), root) for o in s["objects"] if o.get("track") is not None}
    return out


# ---------------------------------------------------------------- next-object reference (evaluation only)
EVID = np.load(OUT / "wm_v3_tracks" / "tracks_all.npz")["events_obj_id"]
segs2 = {s["id"]: s for s in json.load(open(ROOT / "data/labels/episodes_all_v2.json"))["segments"] if s["kind"] == "episode"}
IV = {}


def reference(i):
    if not D.track_main[i] or ep[i] not in segs2:
        return None
    v = vid[i]; k = int(kk[i]); sg = segs2[ep[i]]
    if v not in IV:
        try:
            IV[v] = intervals(v) or ([], [], {})
        except Exception:
            IV[v] = ([], [], {})
    mv_, cover, _ = IV[v]
    if not mv_:
        return None
    mvs = [e for e in mv_ if 10 * sg["start_s"] <= e["f_start"] and e["f_end"] <= 10 * sg["end_s"]]
    left = [e for e in mvs if e["f_end"] > k]
    held = any(c0 <= k <= c1 for c0, c1 in cover)
    msk = D.S.d["obj_mask"][i].astype(bool)
    lefts = {int(s_) for e in left for s_ in np.flatnonzero((EVID[i] == int(e["obj"])) & msk)}
    nx = next((e for e in left if e["f_start"] > k), None) if not held else None
    sl = np.flatnonzero((EVID[i] == int(nx["obj"])) & msk) if nx is not None else []
    return (int(sl[0]) if len(sl) else -1, lefts, not left)


def prev_index(i):
    """Previous tracked keyframe of the same track set (for the constant-velocity baseline); -1 if none."""
    c = np.flatnonzero((TS == TS[i]) & (vid == vid[i]) & (kk < kk[i]))
    return int(c[np.argmax(kk[c])]) if len(c) else -1


def gt():
    cfg = load_models(W / "runs_final" / "watch_notab", seeds=(0,), fold=-1)[0].cfg
    keys = ("obj_geo", "t1_pos", "t1_ok", "t1_present", "obj_mask", "obj_kind", "t2_disp", "t2_ok")
    G = {k: [] for k in keys}
    for b0 in range(0, len(test), BS):
        b = blank(Bt.get("v2", cfg, test[b0:b0 + BS]))
        for k in keys:
            G[k].append(b[k].float().numpy() if b[k].dtype.is_floating_point else b[k].numpy())
    G = {k: np.concatenate(v) for k, v in G.items()}
    raw = D.S.d["obj_geo"][test][..., :3].astype(np.float32)
    n = G["obj_geo"].shape[1]
    G["frame_check_max_abs_diff_m"] = np.array(np.abs(G["obj_geo"][..., :3] - raw[:, :n]).max())
    pi = np.array([prev_index(int(i)) for i in test])
    G["prev_idx"] = pi; G["idx"] = test
    G["dt_prev_s"] = np.where(pi >= 0, (kk[test] - kk[np.maximum(pi, 0)]) / 10.0, np.nan)
    oid = D.S.d["obj_id"]; G["obj_id"] = oid[test][:, :n]
    prevpos = np.full((len(test), n, 3), np.nan, np.float32)
    for r, (i, p) in enumerate(zip(test, pi)):
        if p < 0:
            continue
        for s_ in range(n):
            if not G["obj_mask"][r, s_]:
                continue
            q = np.flatnonzero((oid[p] == oid[i, s_]) & (D.S.d["obj_mask"][p] > 0))
            if len(q):
                prevpos[r, s_] = D.S.d["obj_geo"][p, q[0], :3]
    G["prev_pos_raw"] = prevpos; G["now_raw"] = raw[:, :n]
    G["ep"] = ep[test]; G["task_text"] = TT[test]; G["video"] = vid[test]; G["k"] = kk[test]
    G["scope"] = np.array([IN_SCOPE.get(t, "") for t in TT[test]])
    refs = [reference(int(i)) for i in test]
    G["ref_next"] = np.array([r[0] if r is not None else -2 for r in refs])
    G["ref_done"] = np.array([r[2] if r is not None else False for r in refs])
    lef = np.zeros((len(test), n), bool)
    for r, rr in enumerate(refs):
        if rr is not None:
            for s_ in rr[1]:
                if s_ < n:
                    lef[r, s_] = True
    G["ref_left"] = lef
    TL = twin_labels(); role = np.full((len(test), n), "", object); cat = np.full((len(test), n), "", object); troot = np.full(len(test), "", object)
    for r, i in enumerate(test):
        lab = TL.get(ep[i], {})
        for s_ in range(n):
            if G["obj_mask"][r, s_] and int(oid[i, s_]) in lab:
                role[r, s_], cat[r, s_], troot[r] = lab[int(oid[i, s_])]
    G["twin_role"] = role.astype(str); G["twin_cat"] = cat.astype(str); G["twin_root"] = troot.astype(str)
    np.savez_compressed(O / "gt.npz", **G)
    ins = G["scope"] != ""
    mm = G["obj_mask"][ins].astype(bool) & (G["obj_kind"][ins] == 0)
    print("gt: n", len(test), "in scope", int(ins.sum()), "frame diff (m)", float(G["frame_check_max_abs_diff_m"]),
          "prev found", float((pi >= 0).mean()), "twin label coverage (in-scope movable tokens)", float((G["twin_role"][ins][mm] != "").mean()))


def final_s2():
    G = np.load(O / "gt.npz", allow_pickle=True); cfg_new = load_models(W / "runs_final" / "watch_notab", seeds=(0,), fold=-1)[0].cfg
    old = load_models(OUT / "wm_v2" / "runs" / "final_s2", seeds=(0, 1, 2)); cfg_old = old[0].cfg
    rr = np.flatnonzero(G["ref_next"] >= 0); n = G["obj_mask"].shape[1]
    nxt = np.full(len(test), -1); goal = np.full((len(test), 3), np.nan, np.float32)
    for b0 in range(0, len(rr), BS):
        q = rr[b0:b0 + BS]
        bo = blank(Bt.get("v2", cfg_old, test[q])); mvt = torch.as_tensor(G["ref_next"][q])
        Po = forward(old, bo, 1, mv=mvt, tg=torch.zeros_like(mvt))
        pmo = Po["pm"].mean(0)[:, :n].clone(); mov = torch.as_tensor(G["obj_mask"][q].astype(bool) & (G["obj_kind"][q] == 0))
        pmo[~mov] = -1; nxt[q] = pmo.argmax(-1).numpy(); goal[q] = Po["goal"].mean(0).numpy()
    np.savez_compressed(O / "pred_final_s2.npz", next=nxt, goal_true_next=goal)
    print("final_s2 rows", len(rr), "cfg_new unused", bool(cfg_new))


def prompts_for(G):
    """Single-object prompts: the episode's own sentence with its object noun swapped for one task object's name."""
    rows = []
    for r in np.flatnonzero(np.isin(G["scope"], ["fruit in", "tools in"])):
        names = sorted({c.split("/")[-1] for c, ro in zip(G["twin_cat"][r], G["twin_role"][r]) if ro == "task"})
        for nm in names:
            t = G["task_text"][r]
            p = t.replace("the fruit into", f"the {nm.replace('_', ' ')} into") if "fruit into" in t else t.replace("the tools into", f"the {nm.replace('_', ' ')} into")
            rows.append((r, nm, p))
    return rows


def run(name, run_dir, fold, seeds, opts):
    G = np.load(O / "gt.npz", allow_pickle=True)
    ms = load_models(Path(run_dir), seeds=seeds, fold=fold); cfg = ms[0].cfg
    mode = next((o.split("=")[1] for o in opts if o.startswith("mode=")), "1nh"); text = next((o.split("=")[1] for o in opts if o.startswith("text=")), "own")
    mods = (noapp,) if "noapp" in opts else ()
    rows = np.flatnonzero(G["scope"] != "") if "scope" in opts else np.arange(len(test)); temb = None; meta = {}
    if text == "shuffle":                                                   # the other family's instruction
        swap = {"fruit": "Put the tools into the toolbox.", "tools": "Put the fruit into the fruit bowl."}
        strs = [swap["fruit"] if v == "IMG_2516" else swap["tools"] for v in G["video"]]
        rows = np.flatnonzero(G["scope"] != ""); temb = D.embed_new([strs[r] for r in rows]); meta["texts"] = np.array([strs[r] for r in rows])
    elif text == "prompts":
        P = prompts_for(G); rows = np.array([p[0] for p in P]); temb = D.embed_new([p[2] for p in P])
        meta["prompt_name"] = np.array([p[1] for p in P]); meta["prompt"] = np.array([p[2] for p in P])
    subsets = {"all": tuple(range(len(ms)))}
    if "judge" in opts:
        subsets |= {"judge12": (1, 2), "s0": (0,), "s1": (1,), "s2": (2,)}
    paths_for = {"all"} if (text == "own" and "nopaths" not in opts) else set()
    acc = {nm: {k: [] for k in ("goal", "sd", "p_now", "traj", "e1")} for nm in subsets}
    for b0 in range(0, len(rows), BS):
        q = rows[b0:b0 + BS]; B = len(q)
        b = Bt.get("v2", cfg, test[q])
        if mode in ("1nh", "0nh"):
            b = blank(b)
        for f in mods:
            b = f(b)
        if temb is not None:
            b["task_emb"] = temb[b0:b0 + BS].to(b["task_emb"].dtype)
        mc = torch.full((B,), 1 if mode == "1nh" else 0, dtype=torch.long)
        Es = []
        for m in ms:
            E = m.encode(b, mc); E["kind"] = b["obj_kind"]; Es.append(E)
        for nm, sub in subsets.items():
            R = watch_readout([ms[i] for i in sub], [Es[i] for i in sub], paths=nm in paths_for)
            a = acc[nm]; a["goal"].append(R["goal"].numpy()); a["sd"].append(R["sd"].numpy()); a["p_now"].append(R["p_now"].numpy())
            if "traj" in R:
                a["traj"].append(R["traj"].numpy())
            if getattr(ms[sub[0]], "dist_head", False):
                a["e1"].append(torch.stack([ms[i].dist_pred(Es[i])[..., 0] for i in sub]).mean(0).numpy())
        print(name, b0 + B, "/", len(rows), flush=True)
    out = dict(rows=rows, **meta)
    for nm, a in acc.items():
        for k, v in a.items():
            if v:
                out[f"{nm}__{k}"] = np.concatenate(v)
    np.savez_compressed(O / f"pred_{name}.npz", **out)
    print("wrote", name, "members", len(ms), "sd_scale", getattr(ms[0], "sd_scale", None), "dist_head", getattr(ms[0], "dist_head", False))


if __name__ == "__main__":
    if sys.argv[1] == "gt":
        gt()
    elif sys.argv[1] == "final_s2":
        final_s2()
    else:
        _, _, name, rd, fold, seeds, *opts = sys.argv
        run(name, rd, int(fold), tuple(int(s) for s in seeds.split(",")), opts)
