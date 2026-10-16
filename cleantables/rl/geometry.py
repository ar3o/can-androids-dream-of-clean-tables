"""Quaternion helpers (xyzw, the Isaac Lab convention) and polyline helpers for following an imagined path."""

from __future__ import annotations

import torch


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conj(q), v)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        -1,
    )


def yaw_of(q: torch.Tensor) -> torch.Tensor:
    """Yaw of a near-upright body from its xyzw quaternion."""
    return 2.0 * torch.atan2(q[..., 2], q[..., 3])


def to_frame(root_pos: torch.Tensor, root_quat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Points p (..., 3) into the frame given by root_pos / root_quat (broadcast over the middle dims)."""
    return quat_apply_inv(root_quat, p - root_pos)


def arc_length(path: torch.Tensor) -> torch.Tensor:
    """Cumulative arc length (N, M) along sampled paths (N, M, 3), starting at 0."""
    seg = (path[:, 1:] - path[:, :-1]).norm(dim=-1)
    return torch.cat([torch.zeros(path.shape[0], 1, device=path.device), seg.cumsum(1)], 1)


def path_progress(path: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Arc length up to the path sample nearest to p, minus the distance to that sample (N,)."""
    ar = torch.arange(path.shape[0], device=path.device)
    cum = arc_length(path)
    d = (path - p[:, None]).norm(dim=-1)
    j = d.argmin(1)
    return cum[ar, j] - d[ar, j]


def lookahead_point(path: torch.Tensor, p: torch.Tensor, ahead: float) -> torch.Tensor:
    """First path sample after the one nearest to p whose straight-line distance from p is at least `ahead`
    (the last sample if none). Unlike an arc-length look-ahead it moves on along a zig-zagging averaged path."""
    n, m = path.shape[:2]
    d = (path - p[:, None]).norm(dim=-1)
    j0 = d.argmin(1)
    idx = torch.arange(m, device=path.device)[None]
    cand = (idx > j0[:, None]) & (d >= ahead)
    first = torch.where(cand.any(1), cand.float().argmax(1), torch.full_like(j0, m - 1))
    return path[torch.arange(n, device=path.device), first]


def points_ahead(path: torch.Tensor, p: torch.Tensor, distances: tuple[float, ...]) -> torch.Tensor:
    """Points (N, len(distances), 3) on the path at the given arc lengths beyond the sample nearest to p."""
    n = path.shape[0]
    ar = torch.arange(n, device=path.device)
    cum = arc_length(path).contiguous()
    j = (path - p[:, None]).norm(dim=-1).argmin(1)
    s0 = cum[ar, j]
    pts = []
    for dist in distances:
        s = torch.minimum(s0 + dist, cum[:, -1])
        k = torch.searchsorted(cum, s[:, None].contiguous()).squeeze(1).clamp(1, path.shape[1] - 1)
        c0, c1 = cum[ar, k - 1], cum[ar, k]
        t = ((s - c0) / (c1 - c0).clamp(min=1e-6)).clamp(0, 1)[:, None]
        pts.append(path[ar, k - 1] * (1 - t) + path[ar, k] * t)
    return torch.stack(pts, 1)
