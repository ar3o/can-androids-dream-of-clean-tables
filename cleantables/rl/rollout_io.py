"""Read rollout episodes stored as `ep_<k>.npz` plus front / wrist mp4 videos."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 15
K_FAIL = 15
EMBODIMENTS = {  # action dim, state dim
    "allegro": {"A": 22, "S": 30},
    "franka": {"A": 7, "S": 16},
}
REQUIRED_KEYS = (
    "state action ref_action reward obj_pos obj_quat overhang contact fail_event success "
    "will_fail_within_k domain instruction"
).split()


def embodiment_from_dims(A: int, S: int) -> str:
    for name, d in EMBODIMENTS.items():
        if d["A"] == A and d["S"] == S:
            return name
    raise ValueError(f"unknown embodiment dims A={A} S={S}")


@dataclass
class Episode:
    path: Path  # .../ep_<k>.npz
    run_dir: Path
    data: dict  # arrays + parsed domain + instruction
    embodiment: str

    @property
    def T(self) -> int:
        return int(self.data["action"].shape[0])

    @property
    def front_mp4(self) -> Path:
        return self.path.with_name(self.path.stem + "_front.mp4")

    @property
    def wrist_mp4(self) -> Path:
        return self.path.with_name(self.path.stem + "_wrist.mp4")

    @property
    def split(self) -> str:
        return str(self.data["domain"].get("split", "train_domains"))

    @property
    def success(self) -> bool:
        return bool(np.asarray(self.data["success"]).max() > 0)

    @property
    def name(self) -> str:
        return f"{self.run_dir.name}/{self.path.stem}"


def load_episode(path: str | Path) -> Episode:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    missing = [k for k in REQUIRED_KEYS if k not in z.files]
    if missing:
        raise KeyError(f"{path}: missing contract keys {missing}")
    d = {k: z[k] for k in z.files}
    d["domain"] = json.loads(str(d["domain"]))
    d["instruction"] = str(d["instruction"])
    A, S = d["action"].shape[1], d["state"].shape[1]
    emb = d["domain"].get("embodiment") or embodiment_from_dims(A, S)
    return Episode(path=path, run_dir=path.parent, data=d, embodiment=emb)


def list_episodes(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    return sorted(run_dir.glob("ep_*.npz"), key=lambda p: int(p.stem.split("_")[1]))


def load_manifest(run_dir: str | Path) -> dict:
    p = Path(run_dir) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def read_video(path: str | Path) -> np.ndarray:
    """Decode an mp4 to uint8 (T,H,W,3) with PyAV."""
    import av

    frames = []
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    return np.stack(frames)


def write_video(path: str | Path, frames: np.ndarray, fps: int = FPS) -> None:
    """H.264 yuv420p mp4 with PyAV. frames uint8 (T,H,W,3), H and W even."""
    import av

    T, H, W, _ = frames.shape
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "18", "preset": "veryfast"}
        for f in frames:
            fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
            for pkt in s.encode(fr):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
