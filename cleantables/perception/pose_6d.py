"""World-model object tokens and move labels from the FoundationPose per-frame 6D tracks (pose/<ep>/fp_perframe_scalefix.npz: per
tracked object, frames (10 fps work frames, every 2nd) and the rigid pose mesh -> table of the per-object-scale twin mesh
data/twin_table/ep/_scalefix/<ep>/usd/<id>.obj).

From the posed mesh: x, y = centre of its footprint box, top = its highest point, plus L, W and the long-side yaw of the footprint
(principal axes), i.e. the same box_feat the tokens use.
  tokens  every FP-tracked object token of a snapshot (slot whose tracks.pkl index == the FP object's 'track'): box_feat at the FP frame
          nearest to the snapshot (within 2 frames); other tokens (obstacles, untracked) keep the old tracker boxes
  move    the next move's carry path (32 top-centre points, f_start .. f_end, interpolated over the FP frames), its rotation relative to the
          start (6D rotation representation: the first two columns of R(t) R(t0)^T), the end pose (x, y, top, long-side yaw), the end
          offset (= end top-centre - destination top-centre, table frame, as in cleantables.perception.events) -- only if the FP track covers the move (a frame
          within 2 of f_start and of f_end, >= 50 % of the frames in between); otherwise the old labels stay (fp_move_ok = 0)
No filtering, smoothing or rules beyond these coverage checks. Writes a new dataset; the source dataset (tasks3c) is not modified.

  $VLA_PY -m cleantables.perception.pose_6d     -> OUT/wm_v2/data/tasks3_fp6d.npz + tasks3_fp6d_report.json
  $VLA_PY -m cleantables.perception.pose_6d --pos  -> tasks3_fppos.npz: positions only (FP rotations of near-symmetric objects can flip)
  train on it:  WM_DATA_ROOT=OUT/wm_v2/data TASKS3_DATA=tasks3_fp6d  (cleantables.worldmodel.data reads path_rot / end_rot / rot_ok)
"""
from __future__ import annotations

from cleantables import CT_DATA
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from cleantables.worldmodel.common import OUT, ROOT, jdump, log

POSE = Path(CT_DATA + "/perception_v2/pose")
MESH = ROOT / "data/twin_table/ep/_scalefix"
DATA_OUT = OUT / "wm_v2" / "data"
NP = 32


@lru_cache(maxsize=None)
def mesh(ep, oid):
    p = MESH / ep / "usd" / f"{oid}.obj"
    if not p.exists():
        return None
    return np.array([l.split()[1:4] for l in open(p) if l.startswith("v ")], np.float64)


@lru_cache(maxsize=None)
def fp(ep):
    p = POSE / ep / "fp_perframe_scalefix.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    J = json.load(open(POSE / ep / "fp_perframe_scalefix.json"))
    out = {}
    for oid, o in J["objects"].items():
        if f"{oid}__frames" in z.files and mesh(ep, oid) is not None:
            out[int(o["track"])] = dict(id=oid, frames=z[f"{oid}__frames"].astype(int), T=z[f"{oid}__pose_table"].astype(np.float64))
    return out


def posed_box(V, T):
    """Posed mesh -> (x, y, top, L, W, long-side yaw)."""
    X = V @ T[:3, :3].T + T[:3, 3]
    xy = X[:, :2]
    c = xy.mean(0)
    ev, evec = np.linalg.eigh(np.cov((xy - c).T))
    ax = evec[:, 1]                                           # long axis
    yaw = float(np.arctan2(ax[1], ax[0]))
    Rz = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    loc = (xy - c) @ Rz.T
    lo, hi = loc.min(0), loc.max(0)
    ctr = c + ((lo + hi) / 2) @ Rz                            # footprint box centre (table frame)
    return np.array([ctr[0], ctr[1], X[:, 2].max(), hi[0] - lo[0], hi[1] - lo[1], yaw])


@lru_cache(maxsize=None)
def sf_height(ep, oid):
    """The object's height in its _scalefix twin start pose (upright; rotation-free)."""
    S = json.load(open(MESH / ep / "scene.json"))
    o = {x["id"]: x for x in S["objects"]}.get(oid)
    return None if o is None else float(o["extents"][2])


def centre(V, T):
    """Posed centre of the mesh's bounding box: R c + t (unaffected by symmetry / registration flips, unlike the box's top and axes)."""
    c = 0.5 * (V.min(0) + V.max(0))
    return T[:3, :3] @ c + T[:3, 3]


def topc(ep, tr, i):
    """Positions-only top centre at FP frame index i: posed centre + half the _scalefix upright height."""
    h = sf_height(ep, tr["id"])
    c = centre(mesh(ep, tr["id"]), tr["T"][i])
    return np.array([c[0], c[1], c[2] + 0.5 * h]), c[2] - 0.5 * h


def box_feat6(b):
    x, y, top, L, W, yaw = b
    return np.array([x, y, top, max(L, W), min(L, W), top, np.sin(2 * yaw), np.cos(2 * yaw)], np.float32)


def rot6(R):
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def move_labels(ep, tr, f0, f1, tgt_tr=None):
    """-> dict(path (32,3), rot6 (32,6), end_pose (4,), end_rot6 (6,), end_off (3,) or None) or None if not covered."""
    fr = tr["frames"]
    if not len(fr) or np.min(np.abs(fr - f0)) > 2 or np.min(np.abs(fr - f1)) > 2:
        return None
    sel = np.flatnonzero((fr >= f0 - 2) & (fr <= f1 + 2))
    expected = max(1, (f1 - f0) // 2 + 1)
    if len(sel) < 0.5 * expected or len(sel) < 2:
        return None
    V = mesh(ep, tr["id"])
    boxes = np.stack([posed_box(V, tr["T"][i]) for i in sel])
    t = fr[sel].astype(float)
    tq = np.linspace(f0, f1, NP)
    path = np.stack([np.interp(tq, t, boxes[:, c]) for c in (0, 1, 2)], -1)
    Rs = Rot.from_matrix(tr["T"][sel][:, :3, :3])
    Rq = Slerp(t, Rs)(np.clip(tq, t[0], t[-1]))
    R0 = Rq[0]
    rel = (Rq * R0.inv()).as_matrix()                          # R(t) R(t0)^T, table frame
    r6 = np.stack([rot6(m) for m in rel])
    ie = sel[np.argmin(np.abs(fr[sel] - f1))]
    eb = posed_box(V, tr["T"][ie])
    out = dict(path=path.astype(np.float32), rot6=r6, end_pose=np.array([eb[0], eb[1], eb[2], eb[5]], np.float32), end_rot6=r6[-1])
    if tgt_tr is not None and len(tgt_tr["frames"]):
        j = int(np.argmin(np.abs(tgt_tr["frames"] - f1)))
        if abs(tgt_tr["frames"][j] - f1) <= 4:
            tb = posed_box(mesh(ep, tgt_tr["id"]), tgt_tr["T"][j])
            out["end_off"] = np.array([eb[0] - tb[0], eb[1] - tb[1], eb[2] - tb[2]], np.float32)
    return out


def move_labels_pos(ep, tr, f0, f1, tgt_tr=None):
    """Positions only: the carry path and end position from the FP centre track (top centre = centre + half the _scalefix upright
    height); no rotation labels."""
    fr = tr["frames"]
    if not len(fr) or np.min(np.abs(fr - f0)) > 2 or np.min(np.abs(fr - f1)) > 2:
        return None
    sel = np.flatnonzero((fr >= f0 - 2) & (fr <= f1 + 2))
    expected = max(1, (f1 - f0) // 2 + 1)
    if len(sel) < 0.5 * expected or len(sel) < 2:
        return None
    tc = np.stack([topc(ep, tr, i)[0] for i in sel])
    t = fr[sel].astype(float)
    tq = np.linspace(f0, f1, NP)
    path = np.stack([np.interp(tq, t, tc[:, c]) for c in (0, 1, 2)], -1)
    ie = sel[np.argmin(np.abs(fr[sel] - f1))]
    e = topc(ep, tr, ie)[0]
    out = dict(path=path.astype(np.float32), end_xyz=e.astype(np.float32))
    if tgt_tr is not None and len(tgt_tr["frames"]) and sf_height(ep, tgt_tr["id"]) is not None:
        j = int(np.argmin(np.abs(tgt_tr["frames"] - f1)))
        if abs(tgt_tr["frames"][j] - f1) <= 4:
            out["end_off"] = (e - topc(ep, tgt_tr, j)[0]).astype(np.float32)
    return out


def build_pos(src_name=None, out_name="tasks3_fppos"):
    """Positions-only dataset: token x, y and the move path's / end position's / end offset's x, y from the FP centre track; heights,
    L, W, yaw and the end yaw stay the tracker's (rotation-dependent quantities, the centre height included, are not taken from FP
    because of rotation flips). Same coverage rules as build(). Also reports how often the FP centre height looks off: of the tokens whose upright bottom
    (centre - half height) is within 5 cm of the table, the share more than 2 cm above or below it."""
    src_name = src_name or os.environ.get("TASKS3_DATA_SRC", "tasks3c")
    Z = dict(np.load(ROOT / f"data/uwm/{src_name}.npz", allow_pickle=True))
    N = len(Z["mover"])
    Z["rot_ok"] = np.zeros(N, np.uint8); Z["fp_move_ok"] = np.zeros(N, np.uint8); Z["fp_tok_n"] = np.zeros(N, np.int16)
    Z["path_rot"] = np.full((N, NP, 6), np.nan, np.float32); Z["end_rot"] = np.full((N, 6), np.nan, np.float32)
    evc = {}
    rep = dict(mode="positions only: x, y from the FP centre track; heights, L, W, yaw and the end yaw from the old tracker (the FP centre "
                    "height is hit by the rotation flips: see below_table_2cm and share_rest_tokens_height_off_2cm); no rotation labels",
               episodes_with_fp=[], episodes_without_fp=[], move_samples=0, moves_from_fp=0, tokens_replaced=0, tokens_total=0,
               path_shift_cm=[], end_shift_cm=[], top_shift_cm=[], rest_tokens=0, rest_height_off_2cm=0, start_gap_cm=[])
    for ep in sorted(set(Z["group"].astype(str))):
        F = fp(ep)
        idx = np.flatnonzero(Z["group"].astype(str) == ep)
        (rep["episodes_with_fp"] if F else rep["episodes_without_fp"]).append(ep)
        if not F:
            continue
        F = {k: v for k, v in F.items() if sf_height(ep, v["id"]) is not None}
        v = str(Z["video"][idx[0]])
        if v not in evc:
            evc[v] = {e["id"]: e for e in json.load(open(ROOT / "data/generic" / v / "events.json"))["events"]}
        for n in idx:
            k = int(Z["snap_k"][n])
            for s in range(Z["obj_id"].shape[1]):
                j = int(Z["obj_id"][n, s])
                if j < 0 or not Z["obj_mask"][n, s] or Z["obj_kind"][n, s] != 0:
                    continue
                rep["tokens_total"] += 1
                tr = F.get(j)
                if tr is None:
                    continue
                i = int(np.argmin(np.abs(tr["frames"] - k)))
                if abs(tr["frames"][i] - k) <= 2:
                    tc, bottom = topc(ep, tr, i)
                    old_top = float(Z["obj_geo"][n, s, 2])
                    rep["top_shift_cm"].append(100 * abs(tc[2] - old_top))
                    if abs(bottom) < 0.05:                                  # bottom within 5 cm of the table: (nearly) resting on it
                        rep["rest_tokens"] += 1; rep["rest_height_off_2cm"] += int(abs(bottom) > 0.02)
                    rep["below_table_2cm"] = rep.get("below_table_2cm", 0) + int(bottom < -0.02)
                    Z["obj_geo"][n, s, 0], Z["obj_geo"][n, s, 1] = tc[0], tc[1]           # x, y only: heights stay the tracker's
                    Z["fp_tok_n"][n] += 1; rep["tokens_replaced"] += 1
            mv = int(Z["mover"][n])
            if mv < 0:
                continue
            rep["move_samples"] += 1
            e = evc[v].get(int(Z["ev_id"][n]))
            tr = F.get(int(Z["obj_id"][n, mv]))
            if e is None or tr is None:
                continue
            tg = int(Z["target"][n])
            tgt_tr = F.get(int(Z["obj_id"][n, tg])) if 0 <= tg < Z["obj_id"].shape[1] else None
            ml = move_labels_pos(ep, tr, int(e["f_start"]), int(e["f_end"]), tgt_tr)
            if ml is None:
                continue
            old = Z["path_obj"][n].copy()
            if np.isfinite(old).all():
                rep["path_shift_cm"].append(float(100 * np.linalg.norm(old - ml["path"], axis=1).mean()))
            if np.isfinite(Z["end_pose"][n]).all():
                rep["end_shift_cm"].append(float(100 * np.linalg.norm(Z["end_pose"][n][:3] - ml["end_xyz"])))
            rep["start_gap_cm"].append(float(100 * np.linalg.norm(ml["path"][0, :2] - Z["obj_geo"][n, mv, :2])))
            # x, y from FP; heights from the tracker where it has them (the FP centre height is affected by rotation flips)
            if np.isfinite(old).all():
                Z["path_obj"][n, :, :2] = ml["path"][:, :2]
            else:
                Z["path_obj"][n] = ml["path"]; rep["moves_fp_height"] = rep.get("moves_fp_height", 0) + 1
            if np.isfinite(Z["end_pose"][n, :3]).all():
                Z["end_pose"][n, :2] = ml["end_xyz"][:2]
            if "end_off" in ml and np.isfinite(Z["end_off"][n]).all():
                Z["end_off"][n, :2] = ml["end_off"][:2]
            Z["fp_move_ok"][n] = 1
            rep["moves_from_fp"] += 1
        log(f"fppos {ep}: {int(Z['fp_move_ok'][idx].sum())} of {int((Z['mover'][idx] >= 0).sum())} move samples from FP")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_OUT / f"{out_name}.npz", **Z)
    for k in ("path_shift_cm", "end_shift_cm", "top_shift_cm", "start_gap_cm"):
        x = np.array(rep.pop(k)); rep[k + "_p50_p90"] = [round(float(np.percentile(x, 50)), 2), round(float(np.percentile(x, 90)), 2)] if len(x) else None
    rep["share_moves_from_fp"] = rep["moves_from_fp"] / max(1, rep["move_samples"])
    rep["share_tokens_from_fp"] = rep["tokens_replaced"] / max(1, rep["tokens_total"])
    rep["share_rest_tokens_height_off_2cm"] = rep["rest_height_off_2cm"] / max(1, rep["rest_tokens"])
    rep["share_tokens_fp_bottom_below_table_2cm"] = rep.get("below_table_2cm", 0) / max(1, rep["tokens_replaced"])
    rep["start_gap_note"] = "horizontal distance, path start (FP) vs the token's x, y (FP) at the snapshot"
    jdump(rep, DATA_OUT / f"{out_name}_report.json")
    log(f"fppos: {json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)})}")
    return rep


def build(src_name=None, out_name="tasks3_fp6d"):
    src_name = src_name or os.environ.get("TASKS3_DATA_SRC", "tasks3c")
    Z = dict(np.load(ROOT / f"data/uwm/{src_name}.npz", allow_pickle=True))
    N = len(Z["mover"])
    Z["path_rot"] = np.full((N, NP, 6), np.nan, np.float32); Z["end_rot"] = np.full((N, 6), np.nan, np.float32)
    Z["rot_ok"] = np.zeros(N, np.uint8); Z["fp_move_ok"] = np.zeros(N, np.uint8); Z["fp_tok_n"] = np.zeros(N, np.int16)
    evc = {}
    rep = dict(episodes_with_fp=[], episodes_without_fp=[], move_samples=0, moves_from_fp=0, tokens_replaced=0, tokens_total=0,
               path_shift_cm=[], end_shift_cm=[])
    for ep in sorted(set(Z["group"].astype(str))):
        F = fp(ep)
        idx = np.flatnonzero(Z["group"].astype(str) == ep)
        (rep["episodes_with_fp"] if F else rep["episodes_without_fp"]).append(ep)
        if not F:
            continue
        v = str(Z["video"][idx[0]])
        if v not in evc:
            evc[v] = {e["id"]: e for e in json.load(open(ROOT / "data/generic" / v / "events.json"))["events"]}
        for n in idx:
            k = int(Z["snap_k"][n])
            for s in range(Z["obj_id"].shape[1]):
                j = int(Z["obj_id"][n, s])
                if j < 0 or not Z["obj_mask"][n, s] or Z["obj_kind"][n, s] != 0:
                    continue
                rep["tokens_total"] += 1
                tr = F.get(j)
                if tr is None:
                    continue
                i = int(np.argmin(np.abs(tr["frames"] - k)))
                if abs(tr["frames"][i] - k) <= 2:
                    Z["obj_geo"][n, s] = box_feat6(posed_box(mesh(ep, tr["id"]), tr["T"][i])); Z["fp_tok_n"][n] += 1; rep["tokens_replaced"] += 1
            mv = int(Z["mover"][n])
            if mv < 0:
                continue
            rep["move_samples"] += 1
            e = evc[v].get(int(Z["ev_id"][n]))
            tr = F.get(int(Z["obj_id"][n, mv]))
            if e is None or tr is None:
                continue
            tg = int(Z["target"][n])
            tgt_tr = F.get(int(Z["obj_id"][n, tg])) if 0 <= tg < Z["obj_id"].shape[1] else None
            ml = move_labels(ep, tr, int(e["f_start"]), int(e["f_end"]), tgt_tr)
            if ml is None:
                continue
            old = Z["path_obj"][n].copy()
            if np.isfinite(old).all():
                rep["path_shift_cm"].append(float(100 * np.linalg.norm(old - ml["path"], axis=1).mean()))
            if np.isfinite(Z["end_pose"][n]).all():
                rep["end_shift_cm"].append(float(100 * np.linalg.norm(Z["end_pose"][n][:3] - ml["end_pose"][:3])))
            Z["path_obj"][n] = ml["path"]; Z["end_pose"][n] = ml["end_pose"]; Z["end_ok"][n] = 1
            if "end_off" in ml:
                Z["end_off"][n] = ml["end_off"]
            Z["path_rot"][n] = ml["rot6"]; Z["end_rot"][n] = ml["end_rot6"]; Z["rot_ok"][n] = 1; Z["fp_move_ok"][n] = 1
            rep["moves_from_fp"] += 1
        log(f"fp6d {ep}: {int(Z['fp_move_ok'][idx].sum())} of {int((Z['mover'][idx] >= 0).sum())} move samples from FP")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_OUT / f"{out_name}.npz", **Z)
    for k in ("path_shift_cm", "end_shift_cm"):
        x = np.array(rep.pop(k)); rep[k + "_p50_p90"] = [round(float(np.percentile(x, 50)), 2), round(float(np.percentile(x, 90)), 2)] if len(x) else None
    rep["share_moves_from_fp"] = rep["moves_from_fp"] / max(1, rep["move_samples"])
    rep["share_tokens_from_fp"] = rep["tokens_replaced"] / max(1, rep["tokens_total"])
    jdump(rep, DATA_OUT / f"{out_name}_report.json")
    log(f"fp6d: {json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)})}")
    return rep


if __name__ == "__main__":
    import sys
    build_pos() if "--pos" in sys.argv else build()
