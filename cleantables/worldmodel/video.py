"""Videos of the held-out world-model predictions on the real footage, made by fold models that never saw the episode.

plan (default): at the first frame the model imagines the carry of every object the task moves (each as the mover, to
       its own predicted destination), one colour per object; the episode then plays with each object's real tracked
       path drawn in the same colour, until the labelled end of the task. The next-object pick is shown among the
       task's objects still on the table.
pause: before each tracked move the video pauses and draws the carries imagined for the object about to move.
dense: an update every second with the model's own pick, destination and carries.
A strip underneath shows the predicted progress (raw and calibrated), its label and the same clip played backwards.

    python -m cleantables.worldmodel.video --run imitation IMG_2516_t05 [--style plan|pause|dense]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from cleantables.paths import EPISODES, WM_RUNS, twin_dir, video_path

from .dataset import DATASET, MAX_OBJECTS, TABLE, Dataset, get_batch
from .model import MODE_3D, N_PATH
from .perception_io import FPS, load_events, load_geometry
from .progress import apply_calibration
from .train import checkpoint_name, load_models

log = logging.getLogger(__name__)
ORANGE, LIGHT_ORANGE, GREEN, WHITE = (0, 140, 255), (120, 190, 255), (60, 220, 60), (255, 255, 255)
YELLOW, MAGENTA, CYAN = (0, 230, 255), (255, 0, 255), (255, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
WORK_SIZE = (960, 540)
PAUSE_FRAMES = 36  # frames of one pause (the imagined object travels its path in the first 28)
AFTER_MOVE_FRAMES = 12
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


class FrameReader:
    """Work frames of a recording: every round(fps / 10)-th frame, landscape, 960x540 (the perception frame grid)."""

    def __init__(self, video: str):
        self.cap = cv2.VideoCapture(str(video_path(video)))
        self.stride = max(1, int(round((self.cap.get(cv2.CAP_PROP_FPS) or 30.0) / FPS)))
        self.next_raw = 0
        self.last = (None, None)

    def __call__(self, k: int) -> np.ndarray:
        if self.last[0] == k:
            return self.last[1].copy()
        raw = k * self.stride
        if raw < self.next_raw or raw > self.next_raw + 30 * self.stride:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, raw)
            self.next_raw = raw
        while self.next_raw < raw:
            self.cap.grab()
            self.next_raw += 1
        ok, f = self.cap.read()
        self.next_raw += 1
        if not ok:
            raise IndexError(f"frame {k} not in the recording")
        if f.shape[0] > f.shape[1]:
            f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
        if f.shape[:2] != (WORK_SIZE[1], WORK_SIZE[0]):
            f = cv2.resize(f, WORK_SIZE, interpolation=cv2.INTER_AREA)
        self.last = (k, f)
        return f.copy()


def project(P: np.ndarray, K: np.ndarray, T: np.ndarray):
    """Table-frame points -> work pixels and camera depth."""
    Ti = np.linalg.inv(T)
    pc = np.atleast_2d(P) @ Ti[:3, :3].T + Ti[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6), pc[:, 2]


def draw_box(img, T, K, x, y, top, L, W, H, yaw, color, thickness=2) -> None:
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = np.array([c, s]) * L / 2, np.array([-s, c]) * W / 2
    rect = np.array([[x, y]]) + np.array([u + v, -u + v, -u - v, u - v])
    z0 = max(0.0, top - H)
    P = np.concatenate([np.c_[rect, np.full(4, z0)], np.c_[rect, np.full(4, z0 + H)]])
    uv, z = project(P, K, T)
    if (z <= 0).any() or np.abs(uv).max() > 5000:
        return
    uv = uv.astype(int)
    for a, b in BOX_EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thickness, cv2.LINE_AA)


def draw_geo_box(img, T, K, g, color, thickness) -> None:
    """Draw a box given its feature vector [x, y, top, L, W, H, sin 2yaw, cos 2yaw]."""
    draw_box(
        img,
        T,
        K,
        g[0],
        g[1],
        g[2],
        float(g[3]),
        float(g[4]),
        float(g[5]),
        float(np.arctan2(g[6], g[7]) / 2),
        color,
        thickness,
    )


def draw_path(img, P, K, T, color, thickness) -> None:
    uv, _ = project(P, K, T)
    cv2.polylines(img, [uv.astype(int)], False, color, thickness, cv2.LINE_AA)


@dataclass(frozen=True)
class Theme:
    """Colours of the parts drawn outside the footage; light suits a white page, dark a dark one."""

    name: str
    header: tuple | None  # band above the footage for the caption lines; None draws them on the footage
    text: tuple  # caption text in the band
    outline: tuple  # outline of text drawn on the footage
    strip: tuple
    strip_frame: tuple
    strip_text: tuple
    strip_axis: tuple
    cursor: tuple
    label: tuple  # the progress label line
    label_name: str
    reversed: tuple  # progress of the reversed clip (plain cyan is too faint on a light strip)


THEMES = {
    "dark": Theme(
        name="dark",
        header=None,
        text=WHITE,
        outline=(0, 0, 0),
        strip=(30, 30, 30),
        strip_frame=(90, 90, 90),
        strip_text=(230, 230, 230),
        strip_axis=(200, 200, 200),
        cursor=(255, 255, 255),
        label=WHITE,
        label_name="white",
        reversed=CYAN,
    ),
    "light": Theme(
        name="light",
        header=(246, 246, 246),
        text=(40, 40, 40),
        outline=(255, 255, 255),
        strip=(250, 250, 250),
        strip_frame=(190, 190, 190),
        strip_text=(40, 40, 40),
        strip_axis=(90, 90, 90),
        cursor=(60, 60, 60),
        label=(70, 70, 70),
        label_name="dark grey",
        reversed=(190, 150, 0),
    ),
}
HEADER_LINE = 21


def put_text(img, txt, xy, scale=0.55, color=WHITE, thickness=1, outline=(0, 0, 0)) -> None:
    cv2.putText(img, txt, xy, FONT, scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, FONT, scale, color, thickness, cv2.LINE_AA)


def progress_strip(w, h, tmax, series, cursor, title, ymax=1.0, theme: Theme = THEMES["dark"]) -> np.ndarray:
    im = np.full((h, w, 3), theme.strip, np.uint8)
    left, right, top, bottom = 50, w - 15, 28, h - 22

    def X(t):
        return int(left + t / max(tmax, 1e-6) * (right - left))

    def Y(v):
        return int(bottom - v / ymax * (bottom - top))

    cv2.rectangle(im, (left, top), (right, bottom), theme.strip_frame, 1)
    for xs, ys, color, kind in series:
        pts = np.array([[X(x), Y(np.clip(y, 0, ymax))] for x, y in zip(xs, ys, strict=False) if x <= cursor], np.int32)
        if len(pts) > 1:
            cv2.polylines(im, [pts], False, color, 1 if kind == "thin" else 2, cv2.LINE_AA)
        if kind == "dots":
            for p in pts:
                cv2.circle(im, tuple(int(v) for v in p), 2, color, -1)
    cv2.line(im, (X(cursor), top), (X(cursor), bottom), theme.cursor, 1)
    cv2.putText(im, title, (left, 18), FONT, 0.45, theme.strip_text, 1, cv2.LINE_AA)
    for v in (0, ymax / 2, ymax):
        cv2.putText(im, f"{v:.1f}", (4, Y(v) + 4), FONT, 0.38, theme.strip_axis, 1)
    return im


def smooth(p: np.ndarray, n_modes: int = 8) -> np.ndarray:
    """Display only: keep the n_modes lowest DCT modes of each sampled path (flow samples jitter step to step)."""
    if p.ndim > 2:
        return np.stack([smooth(q, n_modes) for q in p])
    t = (np.arange(p.shape[-2]) + 0.5) / p.shape[-2]
    B = np.stack([np.cos(np.pi * k * t) for k in range(n_modes)], 1)
    B /= np.linalg.norm(B, axis=0)
    return np.einsum("nk,kc->nc", B, np.einsum("nk,...nc->kc", B, p))


def write_mp4(frames: list, out: Path, fps: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


class EpisodeView:
    """What both video styles share: the episode's rows, progress series, geometry and frame reader."""

    def __init__(self, episode: str, d: dict, P: dict, rowof: dict):
        self.task = {t["id"]: t for t in json.load(open(EPISODES))["tasks"]}[episode]
        self.video = self.task["video"]
        g = load_geometry(self.video)
        self.K, self.T_all = g["K"], g["T_all"]
        self.events = {e["id"]: e for e in load_events(self.video)["events"]}
        rows = [i for i in np.flatnonzero(d["group"] == episode) if i in rowof]
        ks = np.array([d["snap_k"][i] for i in rows])
        order = np.argsort(ks)
        self.rows, self.ks = [rows[j] for j in order], ks[order]
        self.k0, self.k1 = int(round(self.task["start_s"] * FPS)), int(round(self.task["end_s"] * FPS))
        self.pv = np.array([P["prog_video"][rowof[i]] for i in self.rows])
        self.pv_rev = np.array([P["prog_video_rev"][rowof[i]] for i in self.rows])
        self.label = np.array([d["prog_video"][i] for i in self.rows])
        self.tx = (self.ks - self.k0) / FPS
        self.tmax = (self.k1 - self.k0) / FPS
        self.frame = FrameReader(self.video)

    def summary(self) -> dict:
        return dict(
            prog_spearman=float(spearmanr(self.pv, self.label)[0]) if np.ptp(self.label) > 0 else None,
            prog_first=float(self.pv[:3].mean()),
            prog_last=float(self.pv[-3:].mean()),
        )


def episode_video(episode: str, d: dict, P: dict, rowof: dict, out: Path) -> dict | None:
    """Pause-and-imagine video of one held-out episode."""
    ev = EpisodeView(episode, d, P, rowof)
    if not ev.rows:
        log.info("episode %s: no predictions", episode)
        return None
    K = ev.K
    title = f'HELD-OUT {episode}  task prompt: "{ev.task["instruction"]}"'
    by_event = {}
    for i in ev.rows:
        if d["mover"][i] >= 0:
            by_event.setdefault(int(d["ev_id"][i]), []).append(i)
    pauses = {
        ev.events[e]["f_start"]: (e, max(v, key=lambda i: d["snap_k"][i]))
        for e, v in by_event.items()
        if e in ev.events
    }
    frames, errors, picks = [], [], []

    def compose(img, lines, f, color=WHITE):
        img = cv2.resize(img, WORK_SIZE)
        y = 22
        for j, line in enumerate(lines):
            put_text(img, line, (10, y), 0.5, color if j == 1 else (235, 235, 235))
            y += 21
        j = int(np.clip(np.searchsorted(ev.ks, f), 0, len(ev.ks) - 1))
        strip = progress_strip(
            960,
            150,
            ev.tmax,
            [(ev.tx, ev.pv, ORANGE, "dots"), (ev.tx, ev.pv_rev, CYAN, "line"), (ev.tx, ev.label, WHITE, "thin")],
            (f - ev.k0) / FPS,
            f"TASK PROGRESS (learned) {ev.pv[j]:.2f}   orange = model, white = label, cyan = same clip REVERSED",
        )
        return np.vstack([img, strip])

    f = ev.k0
    while f <= ev.k1:
        if f not in pauses:
            frames.append(
                compose(ev.frame(f), [title, "real video", f"t = {(f - ev.k0) / FPS:.1f} s of {ev.tmax:.1f} s"], f)
            )
            f += 1
            continue
        e, i = pauses[f]
        r = rowof[i]
        fs, fe = ev.events[e]["f_start"], ev.events[e]["f_end"]
        mv = d["mover"][i]
        g = d["obj_geo"][i, mv]
        L, W, H = float(g[3]), float(g[4]), float(g[5])
        yaw = float(np.arctan2(g[6], g[7]) / 2)
        ps = smooth(P["paths"][r][..., :3]) + g[:3]
        mean = ps.mean(0)
        end_pose = d["end_pose"][i]
        err = 100 * float(np.linalg.norm(mean[-1, :2] - end_pose[:2])) if d["end_ok"][i] else float("nan")
        errors.append(err)
        pick = int(P["p_mover"][r].argmax())
        picks.append(pick == mv)
        pick_txt = (
            f"yellow = model's pick of the next object "
            f"({'= the real one' if pick == mv else 'WRONG: white = real one'}, "
            f"p={P['p_mover'][r][pick]:.2f})"
        )
        base, Tm = ev.frame(fs), ev.T_all[fs]
        for a in range(PAUSE_FRAMES):
            img = base.copy()
            pos = min(1.0, a / 28) * (N_PATH - 1)
            lo, frac = int(pos), pos - int(pos)
            draw_geo_box(img, Tm, K, d["obj_geo"][i, pick], YELLOW, 2)
            for q in ps:
                draw_path(img, q, K, Tm, LIGHT_ORANGE, 1)
            draw_path(img, mean[: lo + 1], K, Tm, ORANGE, 3)
            p = mean[lo] * (1 - frac) + mean[min(lo + 1, N_PATH - 1)] * frac
            draw_box(img, Tm, K, g[0], g[1], g[2], L, W, H, yaw, WHITE, 1)
            draw_box(img, Tm, K, p[0], p[1], p[2], L, W, H, yaw, ORANGE, 3)
            frames.append(
                compose(
                    img,
                    [title, "PAUSED: imagining the next move (12 samples light orange, mean bold)", pick_txt],
                    fs,
                    ORANGE,
                )
            )
        for ff in range(fs, min(fe + AFTER_MOVE_FRAMES, ev.k1 + 1)):
            img, Tm = ev.frame(ff), ev.T_all[ff]
            for q in ps:
                draw_path(img, q, K, Tm, LIGHT_ORANGE, 1)
            draw_path(img, mean, K, Tm, ORANGE, 3)
            lines = [title, "real video resumes (imagined path stays drawn)", ""]
            if ff >= fe and d["end_ok"][i]:
                draw_box(img, Tm, K, end_pose[0], end_pose[1], end_pose[2], L, W, H, float(end_pose[3]), GREEN, 2)
                lines[1] = f"green = real end; imagined end off by {err:.1f} cm"
            frames.append(compose(img, lines, ff))
        f = min(fe + AFTER_MOVE_FRAMES, ev.k1 + 1)
    write_mp4(frames, out, int(FPS))
    return dict(
        episode=episode,
        n_moves_shown=len(errors),
        mean_end_err_cm=float(np.nanmean(errors)) if errors else None,
        next_object_right=float(np.mean(picks)) if picks else None,
        path=str(out),
        **ev.summary(),
    )


def episode_video_dense(
    episode: str, d: dict, P: dict, rowof: dict, out: Path, every: int = 10, hold: int = 6
) -> dict | None:
    """An imagination update every `every` frames: the model's own next-object pick (yellow), its destination (magenta)
    and 12 carries (for the real next object when one follows, else for its own pick); paused `hold` frames per update.
    Real tracked moves are drawn in green when they end."""
    ev = EpisodeView(episode, d, P, rowof)
    if not ev.rows:
        log.info("episode %s: no predictions", episode)
        return None
    K = ev.K
    title = f'HELD-OUT {episode} (model never trained on this episode)   prompt: "{ev.task["instruction"]}"'
    real_moves = sorted(
        [e for e in ev.events.values() if e.get("path") and ev.k0 <= e["f_start"] <= ev.k1], key=lambda e: e["f_start"]
    )
    frames, picks = [], []

    def compose(img, lines, f, color=WHITE):
        img = cv2.resize(img, WORK_SIZE)
        y = 22
        for j, line in enumerate(lines):
            put_text(img, line, (10, y), 0.48, color if j == 1 else (235, 235, 235))
            y += 20
        j = int(np.clip(np.searchsorted(ev.ks, f, side="right") - 1, 0, len(ev.ks) - 1))
        strip = progress_strip(
            960,
            150,
            ev.tmax,
            [
                (ev.tx, ev.label, WHITE, "thin"),
                (ev.tx, ev.pv_rev, CYAN, "thin"),
                (ev.tx, ev.pv, ORANGE, "line"),
                (ev.tx, ev.pv, ORANGE, "dots"),
            ],
            (f - ev.k0) / FPS,
            f"TASK PROGRESS (learned) {ev.pv[j]:.2f}   orange = model, white = label "
            "(0 before start, 1 after end), cyan = same clip REVERSED",
        )
        return np.vstack([img, strip])

    cur = None
    for f in range(ev.k0, ev.k1 + 1):
        img, Tm = ev.frame(f), ev.T_all[f]
        if (f - ev.k0) % every == 0:
            j = int(np.clip(np.searchsorted(ev.ks, f, side="right") - 1, 0, len(ev.ks) - 1))
            i = ev.rows[j]
            r = rowof[i]
            pick = int(P["p_mover"][r].argmax())
            has_truth = d["mover"][i] >= 0
            if has_truth:
                mv = int(d["mover"][i])
                tg = int(d["target"][i]) if d["target"][i] >= 0 else int(P["p_target"][r].argmax())
                ps = smooth(P["paths"][r][..., :3]) + d["obj_geo"][i, mv, :3]
            else:
                mv = pick
                tg = int(P["p_target"][r].argmax())
                ps = smooth(P["paths_pred"][r][..., :3]) + d["obj_geo"][i, pick, :3]
            g = d["obj_geo"][i, mv]
            cur = dict(g=g, ps=ps, mean=ps.mean(0), right=bool(has_truth and pick == mv))
            if has_truth:
                picks.append(pick == mv)
            spread = 100 * float(P["sp_path"][r])
            for a in range(hold):
                im2 = img.copy()
                lo = int((a + 1) / hold * (N_PATH - 1))
                draw_geo_box(im2, Tm, K, d["obj_geo"][i, pick], YELLOW, 2)
                if has_truth:
                    draw_geo_box(im2, Tm, K, g, WHITE, 2)
                if tg < MAX_OBJECTS and d["obj_mask"][i, tg]:
                    draw_geo_box(im2, Tm, K, d["obj_geo"][i, tg], MAGENTA, 2)
                for q in ps:
                    draw_path(im2, q[: lo + 1], K, Tm, LIGHT_ORANGE, 1)
                draw_path(im2, cur["mean"][: lo + 1], K, Tm, ORANGE, 3)
                pick_txt = (
                    f"yellow = model's next-object pick ({'RIGHT' if cur['right'] else 'wrong'}), "
                    "white = real next object; "
                    "paths imagined for the REAL one"
                    if has_truth
                    else "yellow = model's next-object pick; paths imagined for its pick (no tracked move follows)"
                )
                frames.append(
                    compose(
                        im2,
                        [
                            title,
                            f"IMAGINING (t={(f - ev.k0) / FPS:.0f}s): 12 paths (light), mean (bold), "
                            f"magenta = destination; spread {spread:.1f} cm",
                            pick_txt,
                        ],
                        f,
                        ORANGE,
                    )
                )
        if cur is not None:
            draw_geo_box(img, Tm, K, cur["g"], YELLOW, 1)
            for q in cur["ps"]:
                draw_path(img, q, K, Tm, LIGHT_ORANGE, 1)
            draw_path(img, cur["mean"], K, Tm, ORANGE, 2)
        lines = [title, "real video (imagination from the last update stays drawn)", ""]
        for e in real_moves:
            if e["f_end"] <= f <= e["f_end"] + 15 and e.get("end"):
                draw_path(img, np.asarray(e["path"], np.float32), K, Tm, GREEN, 2)
                lines[2] = f"green = the real move that just happened ({e.get('phrase', '')})"
        frames.append(compose(img, lines, f))
    write_mp4(frames, out, int(FPS))
    return dict(
        episode=episode,
        next_object_right=float(np.mean(picks)) if picks else None,
        n_updates=(ev.k1 - ev.k0) // every + 1,
        path=str(out),
        **ev.summary(),
    )


# one colour per object in the plan view (BGR), distinct from the progress-strip colours
OBJECT_COLOURS = [
    (0, 140, 255),
    (255, 120, 40),
    (80, 220, 80),
    (230, 60, 230),
    (60, 60, 240),
    (240, 220, 60),
    (160, 100, 255),
    (40, 200, 200),
    (200, 200, 200),
    (90, 160, 255),
]
PLAN_HOLD_FRAMES = 40  # the opening plan stays on screen 4 s; the carries draw themselves in the first 28 frames
CORE_MARGIN_FRAMES = 5  # tracked moves within 0.5 s of the labelled task belong to it
PLAN_SAMPLES = 4  # imagined carries per ensemble member


def task_moves(events: dict, c0: int, c1: int) -> tuple[list, set]:
    """Tracked moves of the labelled task and the objects they move. Containers are never in the set unless the person
    moved them during the task."""
    moves = sorted(
        [
            e
            for e in events.values()
            if e.get("kind", "move") == "move"
            and e.get("path")
            and c0 - CORE_MARGIN_FRAMES <= e["f_start"] <= c1 + CORE_MARGIN_FRAMES
        ],
        key=lambda e: e["f_start"],
    )
    return moves, {e["obj"] for e in moves}


def object_names(objects: list, ids) -> dict:
    """Event object id -> display name; repeated phrases get a number."""
    phrase = {o["id"]: o["phrase"] for o in objects}
    seen, out = {}, {}
    for oid in sorted(ids):
        p = phrase.get(oid, f"object {oid}")
        seen[p] = seen.get(p, 0) + 1
        out[oid] = p
    count = dict(seen)
    seen = {}
    for oid in sorted(ids):
        p = out[oid]
        if count[p] > 1:
            seen[p] = seen.get(p, 0) + 1
            out[oid] = f"{p} {seen[p]}"
    return out


@torch.no_grad()
def imagine_every_object(models: list, b: dict, slots: list, samples: int) -> dict:
    """slot -> (destination index over objects + TABLE/LEAVES, (M*samples, N_PATH, 3) carries relative to the mover)."""
    md = torch.where(b["vj_ok"].bool(), 0, MODE_3D).long()
    Es = []
    for m in models:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Es.append(m.encode(b, md))
    out = {}
    for j in slots:
        mover = torch.tensor([j], device="cuda")
        probs = []
        for m, E in zip(models, Es, strict=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, tl = m.pointers(E, mover)
            probs.append(torch.softmax(tl.float(), -1))
        dest = torch.stack(probs).mean(0).argmax(-1)
        paths = []
        for m, E in zip(models, Es, strict=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, _, _ = m.heads_from(E, mover, dest)
                paths.append(m.sample_paths_anchored(E, c, K=samples).float()[0, :, :, :3])
        out[j] = (int(dest), torch.cat(paths).cpu().numpy())
    return out


def _lighter(colour, amount=0.55):
    return tuple(int(c + (255 - c) * amount) for c in colour)


def episode_video_plan(
    episode: str,
    d: dict,
    P: dict,
    rowof: dict,
    out: Path,
    run_dir: Path,
    T: dict,
    calibration: list | None = None,
    theme: Theme = THEMES["dark"],
) -> dict | None:
    """Every task object's imagined carry at the start, then the real episode with real paths in the same colours."""
    ev = EpisodeView(episode, d, P, rowof)
    if not ev.rows:
        log.info("episode %s: no predictions", episode)
        return None
    K = ev.K
    c0, c1 = int(round(ev.task["core_start_s"] * FPS)), int(round(ev.task["core_end_s"] * FPS))
    moves, in_scope = task_moves(ev.events, c0, c1)
    all_objects = load_events(ev.video)["objects"]
    twin = twin_dir(episode) / "scene.json"
    twin_names = {}
    if twin.exists():  # the twin's task objects are in scope whether or not a carry was tracked
        event_id = {o["src"]: o["id"] for o in all_objects}
        twin_objects = [o for o in json.load(open(twin))["objects"] if o.get("track") in event_id]
        in_scope = {event_id[o["track"]] for o in twin_objects if o["role"] == "task"}
        twin_names = {event_id[o["track"]]: o["id"].replace("_", " ") for o in twin_objects}
    names = {o["id"]: o["phrase"] for o in all_objects} | object_names(all_objects, in_scope) | twin_names
    colour = {oid: OBJECT_COLOURS[n % len(OBJECT_COLOURS)] for n, oid in enumerate(sorted(in_scope))}
    tracked = {e["obj"] for e in moves}
    first_move = {oid: min([e["f_start"] for e in moves if e["obj"] == oid], default=c1 + 1) for oid in in_scope}
    done_at = {oid: max([e["f_end"] for e in moves if e["obj"] == oid], default=np.inf) for oid in in_scope}
    # plan: each object from the first snapshot that shows it before it is first moved (normally the first one)
    fold = int(d["fold"][ev.rows[0]])
    models = load_models([run_dir / checkpoint_name(fold, s) for s in range(3)])
    plan = {}
    for oid in sorted(in_scope):
        row = next((i for i in ev.rows if d["snap_k"][i] < first_move[oid] and oid in d["obj_id"][i]), None)
        if row is None:
            continue
        slot = int(np.flatnonzero(d["obj_id"][row] == oid)[0])
        b = get_batch(
            {k: v[row : row + 1].cuda() for k, v in T.items()}, torch.zeros(1, dtype=torch.long, device="cuda")
        )
        dest, paths = imagine_every_object(models, b, [slot], PLAN_SAMPLES)[slot]
        g = d["obj_geo"][row]
        dest_name = (
            "table"
            if dest == TABLE
            else "out of view"
            if dest > TABLE
            else names.get(int(d["obj_id"][row, dest]), "an object")
            if d["obj_id"][row, dest] >= 0
            else "an object"
        )
        samples = smooth(paths) + g[slot, :3]
        plan[oid] = dict(
            samples=samples,
            mean=samples.mean(0),
            geo=g[slot],
            dest=dest,
            dest_geo=g[dest] if dest < MAX_OBJECTS else None,
            dest_name=dest_name,
        )
    del models
    torch.cuda.empty_cache()
    fold_cal = calibration[fold] if calibration else None
    pv_cal = apply_calibration(fold_cal, ev.pv) if fold_cal else None
    keep = ev.ks <= c1
    tmax = (c1 - ev.k0) / FPS
    title = f'HELD-OUT {episode} (never seen in training)   prompt: "{ev.task["instruction"]}"'
    frames, picks = [], []

    def pick_at(f):
        """The model's next-object pick among the task's objects still on the table at the latest snapshot."""
        j = int(np.clip(np.searchsorted(ev.ks, f, side="right") - 1, 0, len(ev.ks) - 1))
        i = ev.rows[j]
        p = P["p_mover"][rowof[i]][:MAX_OBJECTS]
        ok = np.array(
            [
                d["obj_id"][i, s] in in_scope and done_at[int(d["obj_id"][i, s])] > d["snap_k"][i]
                for s in range(MAX_OBJECTS)
            ]
        )
        if not ok.any():
            return None
        s = int(np.argmax(np.where(ok, p, -1)))
        return i, s, float(p[s] / p[ok].sum())

    def compose(img, lines, f):
        img = cv2.resize(img, WORK_SIZE)
        if theme.header is None:
            for n, line in enumerate(lines):
                put_text(img, line, (10, 22 + n * HEADER_LINE), 0.5, theme.text, outline=theme.outline)
            header = np.zeros((0, WORK_SIZE[0], 3), np.uint8)
        else:
            height = 12 + HEADER_LINE * len(lines)
            header = np.full((height + height % 2, WORK_SIZE[0], 3), theme.header, np.uint8)  # x264 needs even sizes
            for n, line in enumerate(lines):
                cv2.putText(header, line, (10, 22 + n * HEADER_LINE), FONT, 0.5, theme.text, 1, cv2.LINE_AA)
        y = WORK_SIZE[1] - 12 - 20 * len(plan)
        for oid, pl in plan.items():
            cv2.line(img, (12, y - 5), (40, y - 5), colour[oid], 3, cv2.LINE_AA)
            note = "" if oid in tracked else "  (no tracked carry)"
            put_text(img, f"{names[oid]} -> {pl['dest_name']}{note}", (48, y), 0.45, colour[oid], outline=theme.outline)
            y += 20
        series = [
            (ev.tx[keep], ev.label[keep], theme.label, "thin"),
            (ev.tx[keep], ev.pv_rev[keep], theme.reversed, "thin"),
            (ev.tx[keep], ev.pv[keep], ORANGE, "dots"),
        ]
        title_strip = f"TASK PROGRESS  {theme.label_name} = label, orange = model, cyan = same clip REVERSED"
        if pv_cal is not None:
            series.append((ev.tx[keep], pv_cal[keep], GREEN, "line"))
            title_strip += ", green = model calibrated"
        strip = progress_strip(960, 150, tmax, series, (f - ev.k0) / FPS, title_strip, theme=theme)
        return np.vstack([header, img, strip])

    def draw_pick(img, Tm, f):
        p = pick_at(f)
        if p is None:
            return ""
        i, s, prob = p
        draw_geo_box(img, Tm, K, d["obj_geo"][i, s], YELLOW, 2)
        picks.append(int(d["obj_id"][i, s]))
        name = names[int(d["obj_id"][i, s])]
        return f"yellow box = model's next pick among the objects still to put away: {name} (p={prob:.2f})"

    base, Tm = ev.frame(ev.k0), ev.T_all[ev.k0]
    for a in range(PLAN_HOLD_FRAMES):
        img = base.copy()
        lo = int(min(1.0, (a + 1) / 28) * (N_PATH - 1))
        for oid, pl in plan.items():
            for q in pl["samples"]:
                draw_path(img, q[: lo + 1], K, Tm, _lighter(colour[oid]), 1)
            draw_path(img, pl["mean"][: lo + 1], K, Tm, colour[oid], 3)
            if pl["dest_geo"] is not None:
                draw_geo_box(img, Tm, K, pl["dest_geo"], colour[oid], 1)
            if lo == N_PATH - 1:
                uv, z = project(pl["mean"][-1:], K, Tm)
                if z[0] > 0:
                    cv2.circle(img, tuple(int(v) for v in uv[0]), 6, colour[oid], 2, cv2.LINE_AA)
        pick_txt = draw_pick(img, Tm, ev.k0)
        frames.append(
            compose(
                img,
                [
                    title,
                    "PLAN: imagined carries (bold = mean of 12, light = samples, circle = end, box = destination)",
                    pick_txt,
                ],
                ev.k0,
            )
        )
    for f in range(ev.k0, c1 + 1):
        img, Tm = ev.frame(f), ev.T_all[f]
        for oid, pl in plan.items():
            draw_path(img, pl["mean"], K, Tm, colour[oid], 2)
        for e in moves:
            if e["obj"] in plan and f >= e["f_start"]:
                path = np.asarray(e["path"], np.float32)
                n = (
                    len(path)
                    if f >= e["f_end"]
                    else max(1, int(len(path) * (f - e["f_start"]) / max(1, e["f_end"] - e["f_start"])))
                )
                uv, z = project(path[:n], K, Tm)
                for (u, v), zz in zip(uv.astype(int), z, strict=True):
                    if zz > 0:
                        cv2.circle(img, (int(u), int(v)), 4, colour[e["obj"]], -1, cv2.LINE_AA)
        pick_txt = draw_pick(img, Tm, f)
        frames.append(
            compose(
                img,
                [
                    title,
                    f"line = imagined at the start, dots = real tracked carry   "
                    f"t = {(f - ev.k0) / FPS:.1f} s of {tmax:.1f} s",
                    pick_txt,
                ],
                f,
            )
        )
    write_mp4(frames, out, int(FPS))
    end_err = {}
    for oid, pl in plan.items():
        last = [e for e in moves if e["obj"] == oid and e.get("end")]
        if last:
            real = np.array([last[-1]["end"]["x"], last[-1]["end"]["y"]])
            end_err[names[oid]] = round(100 * float(np.linalg.norm(pl["mean"][-1, :2] - real)), 1)
    return dict(
        episode=episode,
        fold=fold,
        objects={names[o]: plan[o]["dest_name"] for o in plan},
        not_planned=[names[o] for o in in_scope if o not in plan],
        imagined_vs_real_end_cm=end_err,
        path=str(out),
        **ev.summary(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render held-out world-model predictions over the real episode footage.")
    ap.add_argument("episodes", nargs="+", help="episode ids from data/episodes.json")
    ap.add_argument("--run", default="imitation", help="run name under runs/worldmodel/ (needs pred_heldout_both.npz)")
    ap.add_argument("--style", choices=["plan", "pause", "dense"], default="plan")
    ap.add_argument(
        "--theme", choices=sorted(THEMES), default="dark", help="plan style: page colours around the footage"
    )
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out-dir", type=Path, default=None, help="default runs/worldmodel/<run>/videos")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    run_dir = WM_RUNS / a.run
    out_dir = a.out_dir or run_dir / "videos"
    data = Dataset(a.dataset)
    d = data.d
    z = np.load(run_dir / "pred_heldout_both.npz")
    P = {k: z[k] for k in z.files if k != "idx"}
    rowof = {int(i): r for r, i in enumerate(z["idx"])}
    calibration_file = run_dir / "progress_calibration.json"
    calibration = json.load(open(calibration_file)) if calibration_file.exists() else None
    results = []
    for ep in a.episodes:
        suffix = f"_{a.theme}" if a.style == "plan" else ""
        target = out_dir / f"{ep}_{a.style}{suffix}.mp4"
        if a.style == "plan":
            r = episode_video_plan(ep, d, P, rowof, target, run_dir, data.T, calibration, THEMES[a.theme])
        else:
            r = (episode_video_dense if a.style == "dense" else episode_video)(ep, d, P, rowof, target)
        log.info("%s", r)
        results.append(r)
    name = f"videos_{a.style}_{a.theme}.json" if a.style == "plan" else f"videos_{a.style}.json"
    json.dump(results, open(out_dir / name, "w"), indent=1)


if __name__ == "__main__":
    main()
