"""Random start layouts of a twin's objects for training resets.

Containers stay where they were measured. Obstacles get a small jitter. In-scope items are either jittered around their
measured pose or, with probability `shuffle_prob`, placed anywhere on the table. Every item must be reachable, together
with its container, by one arm, and footprints (mesh convex hulls) keep a margin from each other and stay on the table.
"""

from __future__ import annotations

import math

import numpy as np

JITTER_XY = 0.03  # m
JITTER_YAW = 0.26  # rad
SPAWN_MARGIN = 0.02  # m, minimum gap between any two footprints, containers included
JITTER_TRIES = 20
SHUFFLE_TRIES = 300
SPAWN_LIFT = 0.003  # m above the measured height, so nothing starts in contact with the table
SPARE_PARK_X = 0.5  # m beyond the table edge, where unused objects wait
SPARE_PARK_Z = 0.05

# Workspace of the published lift policy in its robot base frame (its training objects and goals lie inside).
REACH_X = (-0.80, -0.25)
REACH_Y = 0.34

_HULLS: dict = {}


def in_reach(base_xy, xy, yaw: float):
    """True where xy lies in the lift-policy workspace of an arm at base_xy whose base is rotated by yaw."""
    d = np.asarray(xy, float) - np.asarray(base_xy, float)
    c, s = math.cos(yaw), math.sin(yaw)
    xb, yb = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
    return (xb >= REACH_X[0]) & (xb <= REACH_X[1]) & (np.abs(yb) <= REACH_Y)


def footprint_hull(obj: dict):
    """Convex hull of the object's mesh projected on the table, in its root frame (the extents box without a mesh)."""
    from shapely.geometry import MultiPoint

    key = obj["mesh"]
    if key not in _HULLS:
        xy = []
        try:
            with open(key) as f:
                for line in f:
                    if line.startswith("v "):
                        v = line.split()
                        xy.append((float(v[1]), float(v[2])))
        except OSError:
            pass
        if len(xy) < 3:
            hx, hy = obj["extents"][0] / 2, obj["extents"][1] / 2
            xy = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        _HULLS[key] = MultiPoint(xy).convex_hull
    return _HULLS[key]


def placed_footprint(obj: dict, x: float, y: float, yaw: float, buffer: float):
    from shapely import affinity

    h = footprint_hull(obj).buffer(buffer)
    return affinity.translate(affinity.rotate(h, yaw, origin=(0, 0), use_radians=True), x, y)


def sample_layouts(scene: dict, n: int, rng: np.random.Generator, shuffle_prob: float):
    """n layouts -> xyz (n, K, 3), yaw (n, K), info. An env whose items do not all fit keeps the measured layout."""
    from shapely.geometry import box

    objs = scene["objects"]
    ids = [o["id"] for o in objs]
    k_all = len(objs)
    p0 = np.array([o["pos"] for o in objs], float)
    tb = scene["table"]
    table = box(tb["x"][0], tb["y"][0], tb["x"][1], tb["y"][1]).buffer(-0.01)
    arm_yaw = math.radians(float(scene.get("arms", {}).get("yaw_deg", 90.0)))
    bases = [np.array(scene["poses"][a][0][:2], float) for a in ("a", "b")]
    containers = [k for k, o in enumerate(objs) if o["role"] == "container"]
    obstacles = [k for k, o in enumerate(objs) if o["role"] in ("obstacle", "stack")]
    items = [k for k, o in enumerate(objs) if o["role"] == "task"]
    spares = [k for k, o in enumerate(objs) if o["role"] == "spare"]
    half_gap = SPAWN_MARGIN / 2

    xyz = np.repeat(p0[None], n, 0)
    xyz[..., 2] += SPAWN_LIFT
    yaw = np.zeros((n, k_all))
    mode = np.zeros(n, np.int64)
    fail = np.zeros(n, bool)
    redrawn = 0
    for e in range(n):
        polys = {k: placed_footprint(objs[k], p0[k, 0], p0[k, 1], 0.0, half_gap) for k in containers}
        for k in rng.permutation(obstacles) if len(obstacles) else []:
            for _ in range(JITTER_TRIES):
                x, y = p0[k, :2] + rng.uniform(-JITTER_XY, JITTER_XY, 2)
                yw = rng.uniform(-JITTER_YAW, JITTER_YAW)
                pg = placed_footprint(objs[k], x, y, yw, half_gap)
                if not any(pg.intersects(q) for q in polys.values()):
                    break
            else:
                x, y, yw = p0[k, 0], p0[k, 1], 0.0
                pg = placed_footprint(objs[k], x, y, 0.0, half_gap)
            xyz[e, k, :2] = (x, y)
            yaw[e, k] = yw
            polys[k] = pg

        shuffle = rng.random() < shuffle_prob
        mode[e] = int(shuffle)
        for k in rng.permutation(items) if len(items) else []:
            dest_xy = p0[ids.index(objs[k]["dest"]), :2] if objs[k].get("dest") in ids else None

            def fits(x, y, yw, k=k, dest_xy=dest_xy, polys=polys):
                if dest_xy is not None and not any(
                    in_reach(b, (x, y), arm_yaw) and in_reach(b, dest_xy, arm_yaw) for b in bases
                ):
                    return None
                pg = placed_footprint(objs[k], x, y, yw, half_gap)
                if not table.contains(placed_footprint(objs[k], x, y, yw, 0.0)) or any(
                    pg.intersects(q) for q in polys.values()
                ):
                    return None
                return pg

            pg = None
            if not shuffle:
                for _ in range(JITTER_TRIES):
                    x, y = p0[k, :2] + rng.uniform(-JITTER_XY, JITTER_XY, 2)
                    yw = rng.uniform(-JITTER_YAW, JITTER_YAW)
                    pg = fits(x, y, yw)
                    if pg is not None:
                        break
                if pg is None:
                    redrawn += 1
            if pg is None:
                for _ in range(SHUFFLE_TRIES):
                    x, y = rng.uniform(tb["x"][0], tb["x"][1]), rng.uniform(tb["y"][0], tb["y"][1])
                    yw = rng.uniform(-math.pi, math.pi)
                    pg = fits(x, y, yw)
                    if pg is not None:
                        break
            if pg is None:
                fail[e] = True
                break
            xyz[e, k, :2] = (x, y)
            yaw[e, k] = yw
            polys[k] = pg
        if fail[e]:
            xyz[e] = p0
            xyz[e, :, 2] += SPAWN_LIFT
            yaw[e] = 0.0
        for k in spares:
            xyz[e, k] = (tb["x"][1] + SPARE_PARK_X, 0.0, SPARE_PARK_Z)
    info = {
        "failed_layouts": [int(k) for k in np.nonzero(fail)[0]],
        "mode": mode,
        "spawn_fail": int(fail.sum()),
        "jitter_redrawn": redrawn,
    }
    return xyz, yaw, info
