"""Fast checks of the twin stage: pure geometry, identity rules and config (no data, GPU or Isaac needed)."""

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from cleantables.twin import fitting, meshes  # noqa: E402
from cleantables.twin.config import DEFAULT_CONFIG, TwinConfig, load_config  # noqa: E402
from cleantables.twin.identity import classify  # noqa: E402
from cleantables.twin.projection import plane_hit, project  # noqa: E402
from cleantables.twin.scene import reach_b  # noqa: E402

K = np.array([[700.0, 0, 480], [0, 700.0, 270], [0, 0, 1]])


def camera_above_table(height=1.0, tilt_deg=30.0):
    """camera -> table transform (OpenCV axes): camera at (0, -0.8, height) looking along +y, tilted down."""
    a = math.radians(tilt_deg)
    forward = np.array([0.0, math.cos(a), -math.sin(a)])
    right = np.array([1.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([right, np.cross(forward, right), forward], 1)
    T[:3, 3] = [0, -0.8, height]
    return T


def test_config_file_matches_defaults():
    assert load_config(DEFAULT_CONFIG) == TwinConfig()


def test_cube_rotations_are_the_24_proper_ones():
    rots = fitting.cube_rotations()
    assert len(rots) == 24
    assert all(abs(np.linalg.det(R) - 1) < 1e-9 for R in rots)
    assert len({tuple(R.ravel()) for R in rots}) == 24


def test_box_corners_footprint():
    P = fitting.box_corners(0.1, -0.2, math.pi / 2, 0.4, 0.2, 0.1)
    assert P.shape == (8, 3)
    assert np.allclose(P[:4, 2], 0) and np.allclose(P[4:, 2], 0.1)
    assert np.allclose(np.ptp(P[:, 0]), 0.2) and np.allclose(np.ptp(P[:, 1]), 0.4)


def test_plane_hit_round_trip():
    T = camera_above_table()
    px = np.array([[480.0, 300.0], [200.0, 400.0], [800.0, 350.0]])
    P = plane_hit(px, K, T, z=0.02)
    assert np.allclose(P[:, 2], 0.02)
    uv, depth = project(P, K, T)
    assert np.allclose(uv, px, atol=1e-6) and (depth > 0).all()


def test_cuboid_fit_recovers_pose():
    T = camera_above_table()
    truth = (0.05, 0.6, 0.3, 0.30, 0.20, 0.10)
    mask = fitting.poly_mask(fitting.box_corners(*truth), K, T)
    f = fitting.fit_cuboid(mask, K, T, (0.07, 0.62, 0.25, 0.30, 0.20, 0.10), fix_dims=(0.30, 0.20, 0.10))
    assert f["iou"] > 0.97
    assert abs(f["x"] - truth[0]) < 0.01 and abs(f["y"] - truth[1]) < 0.01


def test_cuboid_mesh_faces_outward_and_sized():
    m, atlas, src = meshes.cuboid_mesh(0.3, 0.2, 0.1, {})
    assert m.volume == pytest.approx(0.3 * 0.2 * 0.1)  # positive: every face points outwards
    assert np.allclose(m.extents, [0.3, 0.2, 0.1])
    assert atlas.shape == (1024, 1536, 3) and set(src.values()) == {"mean colour"}


def test_slab_mesh_volume():
    sil = np.zeros((20, 30), bool)
    sil[5:15, 5:25] = True
    m = meshes.slab_mesh(sil, 0.01, res=0.002)
    assert m.is_watertight
    assert m.volume == pytest.approx(10 * 20 * 0.002**2 * 0.01, rel=1e-6)
    assert np.allclose(m.bounds[0][:2], -m.bounds[1][:2])


def test_classify_rules():
    assert classify(dict(phrase="red pliers", rgb=[200, 30, 30], size=[0.18, 0.06, 0.02]))[0] == "pliers"
    assert classify(dict(phrase="box", rgb=[250, 250, 250], size=[0.26, 0.16, 0.08]))[0] == "white"
    assert classify(dict(phrase="box", rgb=[240, 120, 40], size=[0.20, 0.12, 0.10]))[0] == "orange_box"
    key, _, _ = classify(dict(phrase="sticky tape", rgb=[100, 100, 100], size=[0.3, 0.3, 0.1]))
    assert key is None


def test_reach_box_of_camera_side_arm():
    base = (0.0, -0.6)
    assert reach_b(base, (0.0, -0.1))  # 0.5 m straight ahead
    assert not reach_b(base, (0.0, -0.5))  # too close
    assert not reach_b(base, (0.5, -0.1))  # too far sideways
