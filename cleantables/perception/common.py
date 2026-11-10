"""Shared paths / loaders for the task-agnostic video -> objects -> events -> world-model pipeline (cleantables.perception).

Nothing here knows what the task is: no object names, no step types. Per video it reuses the generic outputs of earlier stages:
  frames      data/table_clean/_work/<V>/frames/<k>.jpg   (10 fps, 960x540) + frames_meta.json
  depth       data/twin_table/boxstack/<V>/depth10.npy    (MoGe, every frame, half res 270x480, camera z in m)
  table fit   data/twin_table/boxstack/<V>/det3d.npz      (T / T_all cam->table, K, per-frame depth scale, table pixel mask)
Outputs go to data/generic/<V>/ and videos/generic/.
"""
from __future__ import annotations

import sys
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/table_clean/_work"
GEOM = ROOT / "data/twin_table/boxstack"          # depth10 / det3d / table.json of each video (generic geometry, reused)
RAW = ROOT / "data/raw/boxes"
OUT = ROOT / "data/generic"
VIDS = ROOT / "videos/generic"
LOG = ROOT / "runs/generic/PROGRESS.md"
SAM_PY = os.environ.get("SAM3_PY", sys.executable)
SAM_PATH = os.environ.get("SAM3_REPO", "third_party/sam3")
VLA_PY = str(ROOT / "envs/vla/bin/python")
VLM = "Qwen/Qwen3-VL-4B-Instruct"
H_LO, W_LO = 270, 480
FPS = 10.0
AGENT_PHRASES = ["hand"]          # the agent, not an object: its masks are used to ignore hand pixels / know when an object is held

os.environ.setdefault("HF_HOME", str(ROOT / "cache/hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))


def vdir(v):
    d = OUT / v
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"\n## {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    print(msg, flush=True)


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def meta(v):
    return json.load(open(WORK / v / "frames_meta.json"))


def frame(v, k):
    import cv2
    return cv2.imread(str(WORK / v / "frames" / f"{k}.jpg"))


def names(args):
    """'IMG_2533', 'data/raw/boxes/IMG_2533.MOV', 'IMG_2533.MOV' -> 'IMG_2533'"""
    return [Path(a).stem for a in args]


def geom(v):
    """det3d.npz -> dict(T (4,4), T_all (n,4,4), K, scale (n,), table (270,480) bool)."""
    z = np.load(GEOM / v / "det3d.npz")
    n = meta(v)["n"]
    T_all = z["T_all"] if "T_all" in z.files else np.repeat(z["T"][None], n, 0)
    return dict(T=z["T"], T_all=T_all, K=z["K"], scale=z["scale"].astype(np.float32), table=z["table"])


def depth(v):
    return np.load(GEOM / v / "depth10.npy", mmap_mode="r")


def table_crop_box(v, margin=24, up=0):
    """xyxy work-px (960x540) bounding box of the table pixels (+ margin, + `up` px above for the arms)."""
    t = geom(v)["table"]
    ys, xs = np.nonzero(t)
    x0, x1, y0, y1 = xs.min() * 2, xs.max() * 2 + 2, ys.min() * 2, ys.max() * 2 + 2
    return [max(0, x0 - margin), max(0, y0 - margin - up), min(960, x1 + margin), min(540, y1 + margin)]


# ------------------------------------------------------------------ frozen MiniLM text embeddings
_ENC = {}


def embed(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if "m" not in _ENC:
        _ENC["t"] = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _ENC["m"] = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), np.float32)
    out = []
    for i in range(0, len(texts), 64):
        b = _ENC["t"](texts[i:i + 64], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = _ENC["m"](**b).last_hidden_state
        e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------ VLM (Qwen3-VL-4B), GPU if it fits in the budget else CPU bf16
_VLM = {}


def gpu_free():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return 0


def wait_gpu(need_mib, name, max_wait_s=None, poll=60):
    """Block until the minimum free GPU memory over ~15 s is >= need_mib. Other processes are left alone."""
    t0, first = time.time(), True
    while True:
        fr = []
        for _ in range(5):
            fr.append(gpu_free()); time.sleep(3)
        if min(fr) >= need_mib:
            return True
        if first:
            log(f"{name}: waiting for {need_mib} MiB free GPU (free {min(fr)} MiB)")
            first = False
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            return False
        time.sleep(poll)


def vlm():
    """Qwen3-VL-4B-Instruct. Its bf16 weights (~8.9 GB) exceed the 4 GB budget on a shared GPU, so it runs on the CPU in bf16
    (AVX512-BF16, ~6-8 s per clip)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if "m" not in _VLM:
        torch.set_num_threads(int(os.environ.get("VLM_THREADS", "12")))
        _VLM["p"] = AutoProcessor.from_pretrained(VLM)
        _VLM["m"] = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.bfloat16).eval()
    return _VLM["p"], _VLM["m"]


def vlm_ask(images, question, max_new_tokens=48):
    """images: list of PIL images (in time order) -> answer string."""
    import torch
    proc, model = vlm()
    msg = [{"role": "user", "content": [*[{"type": "image", "image": im} for im in images], {"type": "text", "text": question}]}]
    inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def ffmpeg_dir(fdir, out, fps=10):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.system(f"ffmpeg -v error -y -framerate {fps} -i {fdir}/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 {out}")
    for f in Path(fdir).glob("*.jpg"):
        f.unlink()
    Path(fdir).rmdir()
