"""The world model inside a twin: builds model inputs from the live simulator state and runs the frozen ensemble.

There is no phone video in simulation, so the model runs in its 3D-only mode. Object tokens are the simulator's exact
boxes plus each object's own name and appearance features from the real video; hand tokens are the robot palms.

    python -m cleantables.worldmodel.teacher prepare [episode ...]   # -> data/twins/<episode>/wm_tokens.npz

The task token here is the MiniLM embedding of the instruction, while training uses the hashed embedding of
`dataset.hashed_text_embedding`; the published teacher results include this mismatch.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cleantables.paths import EPISODES, WM_RUNS, twin_dir

from .dataset import DATASET, MAX_OBJECTS, TABLE
from .model import END_OFFSET_SCALE, MODE_3D, N_PATH, PATH_SCALE
from .perception_io import embed_texts, load_crop_embeddings, load_tracks, table_token_geometry
from .train import load_models

log = logging.getLogger(__name__)
TEACHER_RUN = "physics_feedback"
TOKENS_FILE = "wm_tokens.npz"
CROPS_PER_OBJECT = 15  # sightings nearest the twin's keyframe whose crop embeddings are averaged


def default_checkpoints() -> list[Path]:
    return [WM_RUNS / TEACHER_RUN / f"final_seed{s}.pt" for s in range(3)]


def load_scene(episode: str) -> tuple[str, dict]:
    """Twin scene of an episode; accepts 'IMG_2516_t05' or a prefixed id such as 'ep/IMG_2516_t05'."""
    episode = Path(episode).name
    return episode, json.load(open(twin_dir(episode) / "scene.json"))


def load_tokens(episode: str) -> dict:
    z = np.load(twin_dir(episode) / TOKENS_FILE)
    return {k: z[k] for k in z.files}


class WorldModelTeacher:
    """Frozen world-model ensemble bound to one twin.

    Attributes: models (WorldModel list, eval mode), Z (the twin's token dict), scene (scene.json), I (number of
    world-model objects), K (path samples per member), dev.
    """

    def __init__(
        self,
        episode: str,
        checkpoints: list | None = None,
        device: str = "cuda",
        samples: int = 4,
        tokens: dict | None = None,
    ):
        self.episode, self.scene = load_scene(episode)
        self.Z = tokens if tokens is not None else load_tokens(self.episode)
        self.dev, self.K = device, samples
        self.models = load_models(default_checkpoints() if checkpoints is None else checkpoints, device)
        self.I = len(self.Z["ids"])
        assert self.I + 1 <= MAX_OBJECTS

    def batch(self, obj_xyz: torch.Tensor, obj_yaw: torch.Tensor, palms: torch.Tensor | None) -> dict:
        """Model inputs for N scenes. obj_xyz (N,I,3): footprint centre x, y and bottom z (table frame); obj_yaw (N,I):
        root yaw added to the baked mesh yaw; palms (N,2,3) robot palm positions (table frame) or None."""
        dev = self.dev
        N, I = obj_xyz.shape[:2]
        ex = torch.as_tensor(self.Z["extents"], device=dev)
        yaw = obj_yaw + torch.as_tensor(self.Z["yaw_baked"], device=dev)[None]
        L, W, H = ex[:, 0].max(ex[:, 1]), ex[:, 0].min(ex[:, 1]), ex[:, 2]
        # box features use the yaw of the long side: +pi/2 when extents[1] is the long one
        yaw_long = yaw + torch.where(ex[:, 1] > ex[:, 0], torch.full_like(L, np.pi / 2), torch.zeros_like(L))[None]
        top = obj_xyz[..., 2] + H[None]
        geo = torch.zeros(N, MAX_OBJECTS, 8, device=dev)
        geo[:, :I] = torch.stack(
            [
                obj_xyz[..., 0],
                obj_xyz[..., 1],
                top,
                L[None].expand(N, I),
                W[None].expand(N, I),
                top,
                torch.sin(2 * yaw_long),
                torch.cos(2 * yaw_long),
            ],
            -1,
        )
        geo[:, I] = torch.as_tensor(self.Z["table_geo"], device=dev)
        txt = torch.zeros(MAX_OBJECTS, 384, device=dev)
        txt[:I] = torch.as_tensor(self.Z["obj_txt"].astype(np.float32), device=dev)
        txt[I] = torch.as_tensor(self.Z["table_txt"].astype(np.float32), device=dev)
        crop = torch.zeros(MAX_OBJECTS, 768, device=dev)
        crop[:I] = torch.as_tensor(self.Z["obj_crop"].astype(np.float32), device=dev)
        crop_ok = torch.zeros(MAX_OBJECTS, dtype=torch.bool, device=dev)
        crop_ok[:I] = torch.as_tensor(self.Z["obj_crop_ok"] > 0, device=dev)
        mask = torch.zeros(MAX_OBJECTS, dtype=torch.bool, device=dev)
        mask[: I + 1] = True
        kind = torch.zeros(MAX_OBJECTS, dtype=torch.long, device=dev)
        kind[I] = 1
        b = dict(
            obj_geo=geo,
            obj_txt=txt[None].expand(N, -1, -1),
            obj_crop=crop[None].expand(N, -1, -1),
            obj_crop_ok=crop_ok[None].expand(N, -1),
            obj_mask=mask[None].expand(N, -1),
            obj_kind=kind[None].expand(N, -1),
        )
        hand = torch.zeros(N, 2, 13, device=dev)
        if palms is not None:
            order = palms[..., 0].argsort(-1)  # training hand slots are sorted by x
            p = torch.gather(palms, 1, order[..., None].expand(-1, -1, 3))
            hand[..., 0] = 1
            hand[..., 1:4] = p
            hand[..., 4:7] = torch.as_tensor(self.Z["hand_ext"], device=dev)
            hand[..., 7:10] = torch.tensor([0.0, 1.0, 0.0], device=dev)
            hand[..., 10:13] = torch.tensor([1.0, 0.0, 0.0], device=dev)
        b.update(
            hand_geo=hand,
            hand_q16=torch.zeros(N, 2, 16, device=dev),
            hand_q_ok=torch.zeros(N, 2, dtype=torch.bool, device=dev),
            vj_tok=torch.zeros(N, 32, 1024, device=dev),
            vj_ok=torch.zeros(N, dtype=torch.bool, device=dev),
            task_emb=torch.as_tensor(self.Z["task_emb"], device=dev)[None].expand(N, -1),
        )
        return b

    @torch.no_grad()
    def encode(self, b: dict) -> list:
        """Trunk outputs of every member in 3D-only mode."""
        mode = torch.full((b["obj_geo"].shape[0],), MODE_3D, dtype=torch.long, device=self.dev)
        Es = []
        for m in self.models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Es.append(m.encode(b, mode))
        return Es

    @torch.no_grad()
    def destination(self, Es: list, mover: torch.Tensor) -> torch.Tensor:
        """Argmax of the ensemble-mean destination over the twin's objects and the table (index I = table)."""
        probs = []
        for m, E in zip(self.models, Es, strict=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, tl = m.pointers(E, mover)
            tl = tl.float()
            probs.append(F.softmax(torch.cat([tl[:, : self.I], tl[:, TABLE : TABLE + 1]], -1), -1))
        return torch.stack(probs).mean(0).argmax(-1)

    @torch.no_grad()
    def end_point(self, b: dict, Es: list, mover: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        """Ensemble-mean end (top centre, table frame): the destination's top plus the top mode of the target-relative
        offset mixture, or for the table the mover's start plus the top mode of its displacement mixture."""
        n = mover.shape[0]
        ar = torch.arange(n, device=self.dev)
        target = torch.where(dest < self.I, dest, torch.full_like(dest, TABLE))
        ends = []
        for m, E in zip(self.models, Es, strict=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, _, mdn = m.heads_from(E, mover, target)
                z = m.end_disp(c)
            top = mdn["logit"].float().argmax(-1)
            off = END_OFFSET_SCALE * torch.gather(mdn["mu"].float(), 1, top[:, None, None].expand(n, 1, 3)).squeeze(1)
            at_target = b["obj_geo"][ar, dest.clamp(max=self.I - 1), :3] + off
            top = z["logit"].float().argmax(-1)
            on_table = (
                b["obj_geo"][ar, mover, :3]
                + torch.gather(z["mu"].float(), 1, top[:, None, None].expand(n, 1, 3)).squeeze(1) * PATH_SCALE
            )
            ends.append(torch.where((dest < self.I)[:, None], at_target, on_table))
        return torch.stack(ends).mean(0)


@torch.no_grad()
def imagine_carries(teacher: WorldModelTeacher, pos: np.ndarray, i: int, K: int = 8):
    """Carries of object i in one layout (pos: (I,3) bottom-centre positions): K anchored samples per member,
    re-anchored to the ensemble's end point as the robot teacher uses them.

    Returns paths (M*K, N_PATH, 3) of the top centre, end (3,), destination index (-1 = table), model inputs, target.
    """
    I = teacher.I
    b = teacher.batch(torch.tensor(pos[None], device=teacher.dev), torch.zeros(1, I, device=teacher.dev), None)
    mover = torch.tensor([i], device=teacher.dev)
    Es = teacher.encode(b)
    dest = teacher.destination(Es, mover)
    d = int(dest)
    target = torch.tensor([d if d < I else TABLE], device=teacher.dev)
    samples = []
    for m, E in zip(teacher.models, Es, strict=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c, _, _ = m.heads_from(E, mover, target)
            samples.append(m.sample_paths_anchored(E, c, K=K).float()[0, :, :, :3])
    end = teacher.end_point(b, Es, mover, dest)[0]
    P = torch.cat(samples, 0) + b["obj_geo"][0, i, :3]
    u = torch.linspace(0, 1, N_PATH, device=teacher.dev)[None, :, None]
    P = P + u * (end - P[:, -1:])
    return P, end, (d if d < I else -1), b, target


def shuffle_layout(Z: dict, scene: dict, rng: np.random.Generator, task_idx, tries: int = 300, margin: float = 0.02):
    """Random non-overlapping table positions (circle footprints) for the objects task_idx; the rest keep their measured
    place. Returns None when an object finds no free spot."""
    P = Z["start_pos"].copy()
    ex = Z["extents"]
    T = scene["table"]
    x0, x1, y0, y1 = T["x"][0], T["x"][1], T["y"][0], T["y"][1]
    placed = [j for j in range(len(P)) if j not in task_idx]
    for i in task_idx:
        r_i = 0.5 * np.hypot(ex[i, 0], ex[i, 1])
        for _ in range(tries):
            q = np.array(
                [rng.uniform(x0 + r_i + 0.03, x1 - r_i - 0.03), rng.uniform(y0 + r_i + 0.03, y1 - r_i - 0.03), 0.0],
                np.float32,
            )
            if all(np.hypot(*(q[:2] - P[j, :2])) > r_i + 0.5 * np.hypot(ex[j, 0], ex[j, 1]) + margin for j in placed):
                P[i] = q
                placed.append(i)
                break
        else:
            return None
    return P


def prepare_tokens(episode: str, dataset: Path = DATASET) -> dict:
    """Per-object name and appearance features of a twin, from the perception outputs of its video."""
    episode, S = load_scene(episode)
    tracks = load_tracks(S["video"])
    crops_by_det = load_crop_embeddings(S["video"], np.float32)
    objs = [o for o in S["objects"] if o.get("track") is not None and o["track"] >= 0]
    k0 = int(S["keyframe"])
    phrases, crops, crop_ok = [], [], []
    for o in objs:
        tr = tracks["objects"][o["track"]]
        phrases.append(tr["phrase"])
        dets = sorted(
            [(abs(k - k0), d) for k, d in zip(tr["sight_k"].tolist(), tr["det"], strict=False) if d in crops_by_det],
            key=lambda x: x[0],
        )
        if dets:
            crops.append(np.median(np.stack([crops_by_det[d] for _, d in dets[:CROPS_PER_OBJECT]]), 0))
            crop_ok.append(1)
        else:
            crops.append(np.zeros(768, np.float32))
            crop_ok.append(0)
    texts = sorted(set(phrases) | {"table", S["instruction"]})
    E = dict(zip(texts, embed_texts(texts), strict=False))
    T = S["table"]
    # hand extent of the robot palms = median extent of the hands in the training data
    hg = np.load(dataset)["hand_geo"].reshape(-1, 13)
    hg = hg[hg[:, 0] > 0.5]
    out = dict(
        ids=np.array([o["id"] for o in objs]),
        phrase=np.array(phrases),
        role=np.array([o["role"] for o in objs]),
        dest_id=np.array([o.get("dest") or "" for o in objs]),
        category=np.array([o["category"] for o in objs]),
        extents=np.array([o["extents"] for o in objs], np.float32),
        yaw_baked=np.array([o.get("yaw_baked", 0.0) for o in objs], np.float32),
        start_pos=np.array([o["pos"] for o in objs], np.float32),
        obj_txt=np.stack([E[p] for p in phrases]).astype(np.float16),
        obj_crop=np.stack(crops).astype(np.float16),
        obj_crop_ok=np.array(crop_ok, np.uint8),
        table_txt=E["table"].astype(np.float16),
        table_geo=table_token_geometry(T["x"], T["y"]),
        task_emb=E[S["instruction"]].astype(np.float32),
        instruction=np.array(S["instruction"]),
        hand_ext=np.median(hg[:, 4:7], 0).astype(np.float32),
    )
    np.savez(twin_dir(episode) / TOKENS_FILE, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="World-model tokens of the twins (run in the perception environment).")
    ap.add_argument("command", choices=["prepare"])
    ap.add_argument("episodes", nargs="*", help="episode ids (default: every episode in data/episodes.json)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    episodes = a.episodes or [t["id"] for t in json.load(open(EPISODES))["tasks"]]
    for ep in episodes:
        if not (twin_dir(ep) / "scene.json").exists():
            log.info("episode %s: no twin, skipped", ep)
            continue
        out = prepare_tokens(ep)
        log.info(
            "episode %s: %d objects %s",
            ep,
            len(out["ids"]),
            list(zip(out["ids"].tolist(), out["phrase"].tolist(), strict=False)),
        )


if __name__ == "__main__":
    main()
