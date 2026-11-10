"""Language inputs for the world model. CPU only.

Encoder: all-MiniLM-L6-v2 with mean pooling and L2 normalisation, the same as cleantables.perception.common.embed, which the
robot bridge uses for object phrases and the task prompt. Loaded from the local snapshot; a missing model raises.
Paraphrases (text only, Qwen3-VL-4B-Instruct run locally): per episode of data/labels/tasks_train.json, 8 training
paraphrases (seed 0) and, in a separate call, 4 test paraphrases (seed 1), kept only if they match no training string after
normalisation. Sampling with temperature 0.8, top-p 0.95.
Narration (video): for each of the 234 moves in tasks3c (episode, ev_id) the VLM sees 4 frames (0.5 s before the start,
1/3, 2/3, 0.5 s after the end; 10 fps frames) and the scene's object phrases, and writes one short imperative sentence
of what the hands do to which object. Greedy decoding.
Prompts, model, seed and raw answers are saved; items already in the output json are skipped, so runs resume.

  source scripts/env.sh; $VLA_PY -m cleantables.worldmodel.text check       # encoder reproduces the bridge vectors
  $VLA_PY -m cleantables.worldmodel.text paraphrase [--limit N]   -> OUT/wm_v2/text/paraphrases.json
  $VLA_PY -m cleantables.worldmodel.text narrate [--limit N]      -> OUT/wm_v2/text/narration.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from cleantables.worldmodel.common import OUT, ROOT, log

TEXT = Path(os.environ.get("WM_TEXT_DIR", str(OUT / "wm_v2" / "text")))       # the final training used WM_TEXT_DIR=.../text_8b
HUB = ROOT / "cache/hf/hub"
MINILM = HUB / "models--sentence-transformers--all-MiniLM-L6-v2"
VLM = HUB / f"models--Qwen--{os.environ.get('WM_VLM', 'Qwen3-VL-4B-Instruct')}"          # the final training used WM_VLM=Qwen3-VL-8B-Instruct
TASKS_JSON = ROOT / "data/labels/tasks_train.json"
_M = {}


def _snap(p: Path) -> Path:
    if not (p / "snapshots").exists():
        raise FileNotFoundError(f"model not found: {p} (v2 has no text fallback by design)")
    return sorted((p / "snapshots").iterdir())[-1]


# ------------------------------------------------------------------ text encoder
def embed(texts):
    """MiniLM sentence vectors (n, 384) float32, identical to cleantables.perception.common.embed. Raises if the model is missing."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if "t" not in _M:
        s = _snap(MINILM)
        _M["t"] = AutoTokenizer.from_pretrained(s)
        _M["m"] = AutoModel.from_pretrained(s).eval()
    texts = [str(t) for t in texts]
    if not texts:
        return np.zeros((0, 384), np.float32)
    out = []
    for i in range(0, len(texts), 64):
        b = _M["t"](texts[i:i + 64], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = _M["m"](**b).last_hidden_state
        e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


def check():
    """Check that the encoder reproduces the vectors stored for the robot bridge (wm_tokens.npz: task_emb, obj_txt)."""
    eps = sorted((ROOT / "data/twin_table/ep").glob("*/wm_tokens.npz"))[:6]
    worst = 0.0
    for f in eps:
        z = np.load(f, allow_pickle=True)
        e = embed([str(z["instruction"])])[0]
        worst = max(worst, float(np.abs(e - z["task_emb"].astype(np.float32)).max()))
        ph = [str(p) for p in z["phrase"]]
        eo = embed(ph)
        worst = max(worst, float(np.abs(eo - z["obj_txt"].astype(np.float32)).max()))
    log(f"text encoder vs bridge vectors ({len(eps)} twins): max abs diff {worst:.2e}")
    assert worst < 2e-3, "v2 text encoder does not match the bridge's MiniLM vectors"
    return worst


# ------------------------------------------------------------------ VLM (local Qwen3-VL-4B, CPU bf16)
def vlm():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if "vm" not in _M:
        torch.set_num_threads(int(os.environ.get("VLM_THREADS", "12")))
        s = _snap(VLM)
        _M["vp"] = AutoProcessor.from_pretrained(s)
        _M["vm"] = AutoModelForImageTextToText.from_pretrained(s, dtype=torch.bfloat16).eval()
    return _M["vp"], _M["vm"]


def ask(question, images=(), max_new_tokens=64, sample=False, seed=0):
    import torch
    proc, model = vlm()
    msg = [{"role": "user", "content": [*[{"type": "image", "image": im} for im in images], {"type": "text", "text": question}]}]
    inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
    torch.manual_seed(seed)
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=max_new_tokens, **kw)
    return proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def lines(ans):
    out = []
    for l in ans.splitlines():
        l = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", l).strip().strip('"').strip()
        if 3 <= len(l.split()) <= 30:
            out.append(l)
    return out


def _load(p):
    return json.load(open(p)) if p.exists() else {}


def _save(d, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    json.dump(d, open(tmp, "w"), indent=1)
    tmp.replace(p)


# ------------------------------------------------------------------ paraphrases
PARA_PROMPT = ("A person at a table was asked: \"{instruction}\"\n"
               "The objects they moved: {objects}. Where they went: {destination}.\n"
               "Write {n} different ways someone could ask for the same thing. Vary the wording and the sentence structure; some may "
               "name the objects, some may not. One request per line, no numbering, nothing else.")


def paraphrase(limit=None):
    tasks = json.load(open(TASKS_JSON))["tasks"]
    p = TEXT / "paraphrases.json"
    R = _load(p)
    R.setdefault("_meta", dict(model=str(_snap(VLM)), prompt=PARA_PROMPT, sampling="temperature 0.8, top-p 0.95",
                               train=dict(n=8, seed=0), test=dict(n=4, seed=1, rule="kept only if it matches no training string")))
    done = 0
    for t in tasks:
        if t["id"] in R:
            continue
        if limit is not None and done >= limit:
            break
        t0 = time.time()
        kw = dict(instruction=t["instruction"], objects=", ".join(t.get("objects", [])) or "unknown", destination=t.get("destination") or "unknown")
        a_tr = ask(PARA_PROMPT.format(n=8, **kw), max_new_tokens=220, sample=True, seed=0)
        a_te = ask(PARA_PROMPT.format(n=4, **kw), max_new_tokens=140, sample=True, seed=1)
        tr = list(dict.fromkeys(lines(a_tr)))
        seen = {norm(s) for s in tr} | {norm(t["instruction"])}
        te = [s for s in dict.fromkeys(lines(a_te)) if norm(s) not in seen]
        R[t["id"]] = dict(instruction=t["instruction"], train=tr, test=te, raw_train=a_tr, raw_test=a_te, s=round(time.time() - t0, 1))
        _save(R, p); done += 1
        log(f"paraphrase {t['id']}: {len(tr)} train, {len(te)} test ({time.time() - t0:.0f}s)")
    return R


# ------------------------------------------------------------------ narration
NARR_PROMPT = ("These {k} frames, in time order, show a person's hands at a table. Objects on the table: {names}.\n"
               "In ONE short imperative sentence, say what the hands do in these frames: which object is moved and where it goes, "
               "like a command someone could give (for example \"pick up the X and put it on the Y\"). Use the object names above. "
               "Answer with the sentence only.")


def moves():
    """(episode, video, ev_id) of every move snapshot in the training data (tasks3c), with the event from events.json."""
    z = np.load(ROOT / f"data/uwm/{os.environ.get('TASKS3_DATA', 'tasks3c')}.npz", allow_pickle=True)
    m = z["mover"] >= 0
    keys = sorted(set(zip(z["group"][m].astype(str), z["video"][m].astype(str), z["ev_id"][m].astype(int))))
    out, evc = [], {}
    for ep, v, eid in keys:
        if v not in evc:
            evc[v] = {e["id"]: e for e in json.load(open(ROOT / "data/generic" / v / "events.json"))["events"]}
        out.append((ep, v, eid, evc[v][eid]))
    return out


def scene_names(v):
    import pickle
    t = pickle.load(open(ROOT / "data/generic" / v / "tracks.pkl", "rb"))
    return list(dict.fromkeys(str(o["phrase"]) for o in t["objects"]))


def narrate(limit=None):
    from PIL import Image
    p = TEXT / "narration.json"
    R = _load(p)
    R.setdefault("_meta", dict(model=str(_snap(VLM)), prompt=NARR_PROMPT, decoding="greedy",
                               frames="10 fps work frames: f_start-5, 1/3, 2/3, f_end+5; resized to 480 px wide"))
    done = 0
    names = {}
    for ep, v, eid, e in moves():
        key = f"{ep}#{eid}"
        if key in R:
            continue
        if limit is not None and done >= limit:
            break
        t0 = time.time()
        fdir = ROOT / "data/table_clean/_work" / v / "frames"
        n = len(list(fdir.glob("*.jpg"))) if v not in names else names[v][1]
        if v not in names:
            names[v] = (scene_names(v), n)
        a, b = int(e["f_start"]), int(e["f_end"])
        ks = [max(0, a - 5), int(round(a + (b - a) / 3)), int(round(a + 2 * (b - a) / 3)), min(n - 1, b + 5)]
        ims = []
        for k in ks:
            im = Image.open(fdir / f"{k}.jpg").convert("RGB")
            ims.append(im.resize((480, round(480 * im.height / im.width))))
        q = NARR_PROMPT.format(k=len(ims), names=", ".join(names[v][0]))
        ans = ask(q, ims, max_new_tokens=40)
        sent = ans.splitlines()[0].strip().strip('"').strip() if ans else ""
        R[key] = dict(episode=ep, video=v, ev_id=int(eid), frames=[int(k) for k in ks], names=names[v][0], tracked_phrase=e.get("phrase"), text=sent, raw=ans,
                      s=round(time.time() - t0, 1))
        _save(R, p); done += 1
        log(f"narrate {key} ({e.get('phrase')}): {sent!r} ({time.time() - t0:.0f}s)")
    return R


VERIFY_PROMPT = ("These {k} frames, in time order, show a person's hands at a table. Objects on the table: {names}.\n"
                 "Which ONE of these objects do the hands move in these frames? Answer with its name from the list only.")


def verify():
    """Consistency of each narration with the tracker: the VLM picks the moved object from the scene's name list (same 4 frames),
    and a narration is consistent when the pick is the tracked mover's phrase. Training uses only consistent narrations."""
    from PIL import Image
    p = TEXT / "narration.json"
    R = _load(p)
    R["_meta"]["verify"] = dict(prompt=VERIFY_PROMPT, decoding="greedy", rule="consistent = VLM's pick from the name list == tracked mover phrase")
    done = 0
    for key, e in R.items():
        if key == "_meta" or "verify" in e:
            continue
        fdir = ROOT / "data/table_clean/_work" / e["video"] / "frames"
        ims = []
        for k in e["frames"]:
            im = Image.open(fdir / f"{k}.jpg").convert("RGB")
            ims.append(im.resize((480, round(480 * im.height / im.width))))
        ans = ask(VERIFY_PROMPT.format(k=len(ims), names=", ".join(e["names"])), ims, max_new_tokens=12)
        pick = norm(ans.splitlines()[0] if ans else "")
        cand = [nm for nm in e["names"] if norm(nm) == pick] or [nm for nm in e["names"] if norm(nm) in pick or pick in norm(nm)]
        e["verify"] = dict(answer=ans, pick=cand[0] if cand else None)
        e["consistent"] = bool(cand) and norm(cand[0]) == norm(str(e.get("tracked_phrase")))
        done += 1
        if done % 20 == 0:
            _save(R, p); log(f"verify {done}: consistent so far {np.mean([x['consistent'] for k_, x in R.items() if k_ != '_meta' and 'consistent' in x]):.2f}")
    _save(R, p)
    c = [x["consistent"] for k_, x in R.items() if k_ != "_meta" and "consistent" in x]
    log(f"verify: {sum(c)}/{len(c)} narrations consistent with the tracked mover ({np.mean(c):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "paraphrase", "narrate", "verify", "all"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    if a.cmd in ("paraphrase", "all"):
        paraphrase(a.limit)
    if a.cmd in ("narrate", "all"):
        narrate(a.limit)
    if a.cmd in ("verify", "all"):
        verify()
