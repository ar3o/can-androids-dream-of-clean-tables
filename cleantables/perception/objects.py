"""Open-vocabulary object names: a VLM lists the objects on the table, SAM 3 checks which names find real objects.

1. list      Qwen3-VL looks at three table crops per video (start, middle, end) and lists movable objects and containers
             -> vlm_objects.json
2. collapse  phrases of all videos in the batch are normalised, near-duplicates merged (MiniLM cosine), then synonyms
             collapsed to one phrase per object type; each video keeps its most listed types, plus the object names of
             its episodes that none of them covers -> candidate_phrases.json
3. select    with SAM 3 keyframe detections of the candidates: each object instance is named by the phrase that finds it
             with the highest probability; phrases that name a confident on-table instance are kept -> vocab.json
"""

from __future__ import annotations

import json
import re
from collections import Counter

import cv2
import numpy as np

from cleantables import paths

from . import geometry as G
from .config import ObjectsConfig
from .files import AGENT_PHRASES, depth, geometry, n_frames, read_frame, video_dir, write_json

LIST_QUESTION = (
    "This photo shows a table. List every separate movable object lying or standing on the table, one entry per "
    "physical object (if there are several similar ones, list each with what tells it apart). Use a short noun "
    "phrase of 1 to 4 words, adding the colour or material when it helps, e.g. 'red mug', 'glass bowl', 'banana',"
    " 'white plate'. Also list any tray, mat or container that other things are put on or in. Do not list people,"
    " hands, arms, the table itself, the wall or the floor. Answer with the phrases only, separated by "
    "semicolons."
)
NOT_OBJECTS = {
    "table",
    "wooden table",
    "table top",
    "tabletop",
    "person",
    "hand",
    "hands",
    "arm",
    "floor",
    "wall",
    "nothing",
    "none",
}
CROP_LONG_SIDE = 640
LIST_MAX_TOKENS = 96
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_DILATE = np.ones((15, 15), np.uint8)
SAME_INSTANCE_IOU = 0.5
MAX_MASK_FRACTION = 0.3


# ---------------------------------------------------------------- text and vision-language models
_TEXT = {}


def embed_text(texts) -> np.ndarray:
    """Frozen MiniLM sentence embeddings (mean-pooled, unit length), (n, 384)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if "model" not in _TEXT:
        _TEXT["tok"] = AutoTokenizer.from_pretrained(TEXT_MODEL)
        _TEXT["model"] = AutoModel.from_pretrained(TEXT_MODEL).eval()
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), np.float32)
    out = []
    for i in range(0, len(texts), 64):
        batch = _TEXT["tok"](texts[i : i + 64], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = _TEXT["model"](**batch).last_hidden_state
        e = (h * batch["attention_mask"][..., None]).sum(1) / batch["attention_mask"].sum(1, keepdim=True)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


class VisionLanguageModel:
    """Qwen3-VL (greedy decoding); on the GPU when there is room, else on the CPU in bf16."""

    def __init__(self, name: str, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()

    def ask(self, images, question: str, max_new_tokens: int = 48) -> str:
        msg = [
            {
                "role": "user",
                "content": [*[{"type": "image", "image": im} for im in images], {"type": "text", "text": question}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------- 1. list
def table_crop_box(video: str, margin: int = 24, up: int = 0) -> list[int]:
    """xyxy box (960x540) around the table pixels, with a margin and `up` extra pixels above."""
    ys, xs = np.nonzero(geometry(video)["table"])
    x0, x1, y0, y1 = xs.min() * 2, xs.max() * 2 + 2, ys.min() * 2, ys.max() * 2 + 2
    return [max(0, x0 - margin), max(0, y0 - margin - up), min(960, x1 + margin), min(540, y1 + margin)]


def normalise(phrase: str) -> str:
    """Lower case, no article or count word, crude singular of the head noun; '' for non-objects."""
    s = phrase.strip().strip(".").strip("'\"").lower()
    s = re.sub(r"^(a|an|the|some|two|three|several)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    words = s.split()
    if not words or s in NOT_OBJECTS or len(words) > 5:
        return ""
    last = words[-1]
    if len(last) > 3 and last.endswith("es") and last[-3] in "sxz":
        last = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    return " ".join(words[:-1] + [last])


def list_objects(video: str, vlm: VisionLanguageModel | None) -> dict:
    path = paths.perception_dir(video) / "vlm_objects.json"
    if path.exists():
        return json.loads(path.read_text())
    from PIL import Image

    n = n_frames(video)
    x0, y0, x1, y1 = table_crop_box(video)
    rows = []
    for k in (2, n // 2, n - 3):
        crop = read_frame(video, k)[y0:y1, x0:x1]
        s = CROP_LONG_SIDE / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        answer = vlm.ask([Image.fromarray(crop[..., ::-1])], LIST_QUESTION, LIST_MAX_TOKENS)
        rows.append(dict(k=k, answer=answer, phrases=[normalise(x) for x in answer.split(";") if normalise(x)]))
        print(f"{video} objects at frame {k}: {answer}", flush=True)
    out = dict(video=video, crop_xyxy=[int(c) for c in (x0, y0, x1, y1)], question=LIST_QUESTION, frames=rows)
    write_json(out, path)
    return out


# ---------------------------------------------------------------- 2. collapse
def greedy_groups(order, joins) -> list[list[int]]:
    """Each item joins the first group whose first member it `joins`, else starts a group."""
    groups = []
    for i in order:
        g = next((g for g in groups if joins(i, g[0])), None)
        if g is None:
            groups.append([i])
        else:
            g.append(i)
    return groups


def collapse(videos: list[str], cfg: ObjectsConfig) -> dict[str, list[str]]:
    listings = {v: list_objects(v, None) for v in videos}
    counts = Counter()
    for v in videos:
        counts.update(sorted({p for r in listings[v]["frames"] for p in r["phrases"]}))
    phrases = sorted(counts, key=lambda p: (-counts[p], p))
    E = embed_text(phrases)
    S = E @ E.T
    merged_into = {}
    for g in greedy_groups(range(len(phrases)), lambda i, j: S[i, j] >= cfg.merge_similarity):
        for i in g:
            merged_into[phrases[i]] = phrases[g[0]]
    candidates = [p for p in phrases if merged_into[p] == p]
    E = embed_text(candidates)
    S = E @ E.T
    order = sorted(range(len(candidates)), key=lambda i: (-counts[candidates[i]], len(candidates[i]), candidates[i]))

    def same_type(i, j):
        wi, wj = set(candidates[i].split()), set(candidates[j].split())
        return S[i, j] >= cfg.collapse_similarity or wi <= wj or wj <= wi

    to_type = {}
    for g in greedy_groups(order, same_type):
        for i in g:
            to_type[candidates[i]] = candidates[g[0]]
    per_video = {}
    for v in videos:
        score = {}
        for r in listings[v]["frames"]:
            for t in {to_type[merged_into[p]] for p in r["phrases"] if merged_into.get(p) in to_type}:
                score[t] = score.get(t, 0) + 1
        top = sorted(score, key=lambda t: (-score[t], -counts.get(t, 0), t))[: cfg.max_phrases]
        added = missed_episode_objects(v, top, cfg.collapse_similarity) if cfg.add_episode_objects else []
        per_video[v] = top + added
        write_json(
            dict(phrases=top + added, listing_counts=score, from_episodes=added, batch=videos),
            video_dir(v) / "candidate_phrases.json",
        )
    print(
        f"object names: {len(phrases)} phrases -> {len(candidates)} after merging -> per video {per_video}", flush=True
    )
    return per_video


def missed_episode_objects(video: str, phrases: list[str], similarity: float) -> list[str]:
    """Object names of the video's episodes (episodes.json) that no listed phrase covers (shared words or MiniLM)."""
    if not paths.EPISODES.exists():
        return []
    names = []
    for episode in json.loads(paths.EPISODES.read_text())["tasks"]:
        if episode["video"] == video:
            names += [p for p in map(normalise, episode["objects"]) if p and p not in names]
    if not names or not phrases:
        return names
    S = embed_text(names) @ embed_text(phrases).T
    added = []
    for i, name in enumerate(names):
        words = set(name.split())
        covered = any(
            S[i, j] >= similarity or words <= set(p.split()) or set(p.split()) <= words for j, p in enumerate(phrases)
        )
        if not covered:
            added.append(name)
    return added


def candidate_phrases(video: str) -> list[str]:
    return json.loads((paths.perception_dir(video) / "candidate_phrases.json").read_text())["phrases"]


# ---------------------------------------------------------------- 3. select
def iou(a: np.ndarray, b: np.ndarray) -> float:
    return np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())


def on_table_3d(mask, D, scale, K, T, extent) -> bool:
    """The mask's depth points stand on or above the table plane and their centre is within the table extent (+5 cm)."""
    P, _ = G.backproject(np.asarray(D, np.float32) * scale, K, mask)
    if len(P) < 10:
        return False
    Q = G.to_table(T, P)
    cx, cy = np.median(Q[:, 0]), np.median(Q[:, 1])
    return bool(
        np.median(Q[:, 2]) > -0.03
        and extent[0, 0] - 0.05 <= cx <= extent[0, 1] + 0.05
        and extent[1, 0] - 0.05 <= cy <= extent[1, 1] + 0.05
    )


def composite(ok: list[int], d: dict) -> set[int]:
    """Detections like 'stack of boxes': >= 70 % covered by >= 2 other confident detections of the same frame."""
    out = set()
    for i in ok:
        mi = d["masks"][i]
        parts = [
            j
            for j in ok
            if j != i
            and d["frame"][j] == d["frame"][i]
            and iou(mi, d["masks"][j]) < SAME_INSTANCE_IOU
            and (d["masks"][j] & mi).sum() > 0.6 * d["masks"][j].sum()
        ]
        if len(parts) >= 2:
            union = np.zeros_like(mi)
            for j in parts:
                union |= d["masks"][j]
            if (union & mi).sum() >= 0.7 * mi.sum():
                out.add(i)
    return out


def select(video: str, d: dict, cfg: ObjectsConfig) -> dict:
    """d: keyframe detections of the candidate phrases on the name-check frames (load_detections layout)."""
    g = geometry(video)
    D10 = depth(video)
    extent = np.array(G.table_extent(g["T"], g["K"], g["table"]))
    table = cv2.dilate(g["table"].astype(np.uint8), TABLE_DILATE).astype(bool)
    P = d["prompts"]
    ok = []
    for i in range(len(d["prob"])):
        if d["prob"][i] < cfg.name_min_prob:
            continue
        ys, xs = np.nonzero(d["masks"][i])
        if (
            len(xs) < 20
            or len(xs) > MAX_MASK_FRACTION * d["masks"][i].size
            or not table[int(ys.mean()), int(xs.mean())]
        ):
            continue
        k = int(d["frame"][i])
        if on_table_3d(d["masks"][i], D10[k], g["scale"][k], g["K"], g["T_all"][k], extent):
            ok.append(i)
    dropped = composite(ok, d)
    ok = [i for i in ok if i not in dropped]
    instance = {}
    for k in d["frames"]:
        reps = []
        for i in sorted([i for i in ok if d["frame"][i] == k], key=lambda i: -d["prob"][i]):
            j = next((r for r in reps if iou(d["masks"][i], d["masks"][r]) >= SAME_INSTANCE_IOU), None)
            if j is None:
                reps.append(i)
                instance[i] = (k, len(reps) - 1)
            else:
                instance[i] = instance[j]
    # the most confident phrase names an instance ('blue box' 0.92 beats 'cardboard box' 0.90 on the blue box)
    best = {}
    for i, key in instance.items():
        if key not in best or d["prob"][i] > d["prob"][best[key]]:
            best[key] = i
    score = {}
    for i in best.values():
        name = P[d["prompt"][i]]
        if d["prob"][i] >= cfg.name_keep_prob and name not in AGENT_PHRASES:
            score[name] = score.get(name, 0) + float(d["prob"][i])
    chosen = sorted(score, key=lambda p: -score[p])[: cfg.max_vocabulary]
    evidence = {}
    for p, name in enumerate(P):
        probs = [float(d["prob"][i]) for i in ok if d["prompt"][i] == p]
        evidence[name] = dict(
            n_conf=len(probs), names_score=round(score.get(name, 0), 3), max_prob=round(max(probs, default=0), 3)
        )
    out = dict(
        video=video,
        phrases=chosen,
        agent_phrases=AGENT_PHRASES,
        verify_frames=d["frames"],
        p_ok=cfg.name_min_prob,
        evidence=evidence,
    )
    write_json(out, video_dir(video) / "vocab.json")
    print(f"{video} vocabulary ({len(chosen)}): {chosen}", flush=True)
    return out


def vocabulary(video: str) -> list[str]:
    return json.loads((paths.perception_dir(video) / "vocab.json").read_text())["phrases"]
