import numpy as np
import pytest

from cleantables.perception import events, geometry, objects, run, video_features


def test_table_frame_puts_plane_points_at_zero_height():
    n = np.array([0.0, -0.8, -0.6])
    c = np.array([0.0, 0.3, 1.2])
    K = np.array([[500.0, 0, 480], [0, 500.0, 270], [0, 0, 1]])
    T = geometry.table_frame(n, c, K)
    tangent = np.cross(n, [1.0, 0, 0])
    pts = c + np.outer(np.linspace(-0.5, 0.5, 5), tangent)
    assert np.allclose(geometry.to_table(T, pts)[:, 2], 0, atol=1e-9)
    assert geometry.to_table(T, np.zeros((1, 3)))[0, 2] > 0  # the camera is above the table


def test_rect_extent_recovers_a_rotated_rectangle():
    rng = np.random.default_rng(0)
    local = rng.uniform([-0.15, -0.05], [0.15, 0.05], size=(4000, 2))
    yaw = 0.4
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    x, y, L, W, got_yaw = geometry.rect_extent(local @ R.T + [0.2, -0.1])
    assert (x, y) == pytest.approx((0.2, -0.1), abs=0.01)
    assert L == pytest.approx(0.3 * 0.94, abs=0.01) and W == pytest.approx(0.1 * 0.94, abs=0.01)
    assert got_yaw == pytest.approx(yaw, abs=0.02)


def test_polygon_overlap_and_relations():
    a = geometry.rect_corners(0, 0, 0.2, 0.2, 0)
    b = geometry.rect_corners(0.1, 0, 0.2, 0.2, 0)
    assert geometry.poly_area(geometry.poly_clip(a, b)) == pytest.approx(0.02)
    box = dict(x=0.0, y=0.0, top=0.10, yaw=0.0)
    on_box = dict(x=0.0, y=0.0, top=0.15, yaw=0.0)
    in_box = dict(x=0.0, y=0.0, top=0.05, yaw=0.0)
    beside = dict(x=0.23, y=0.0, top=0.05, yaw=0.0)
    others = [(7, box, [0.3, 0.2, 0.10])]
    assert events.relation(on_box, [0.1, 0.1, 0.05], others) == ("on_top_of", 7)
    assert events.relation(in_box, [0.1, 0.1, 0.05], others) == ("inside", 7)
    assert events.relation(beside, [0.1, 0.1, 0.05], others) == ("next_to", 7)
    assert events.relation(dict(x=1.0, y=1.0, top=0.05, yaw=0.0), [0.1, 0.1, 0.05], others) == ("table", None)


def test_rests_split_by_a_move():
    boxes = [dict(x=0.0, y=0.0, top=0.05, hand_frac=0.0)] * 6 + [dict(x=0.3, y=0.0, top=0.05, hand_frac=0.0)] * 6
    o = dict(sight_k=np.arange(0, 36, 3), boxes=boxes)
    assert events.sight_rests(o) == [(0, 15), (18, 33)]


def test_clip_indices_and_nearest_keyframe():
    idx = video_features.clip_indices([0, 20], 10.0)
    assert idx.shape == (2, 16) and idx[0].max() == 0 and idx[1, -1] == 20 and idx[1, 0] == 10
    ks = np.array([0, 2, 4])
    assert video_features.nearest_keyframe(ks, 3) == 1


def test_phrase_normalisation():
    assert objects.normalise(" The Red Mugs.") == "red mug"
    assert objects.normalise("boxes") == "box"
    assert objects.normalise("hands") == ""


def test_name_check_subset_keeps_candidates_on_three_keyframes():
    frames = list(range(0, 36, 3))
    d = dict(
        frames=frames,
        frame=np.array([frames[1], frames[1], frames[5]]),
        prob=np.array([0.9, 0.8, 0.7], np.float32),
        prompt=np.array([0, 1, 0]),
        prompts=["box", "hand"],
        masks=np.zeros((3, 270, 480), bool),
    )
    sub = run.name_check_subset(d, ["box"])
    assert sub["frames"] == [3, 15, 30] and list(sub["prompt"]) == [0, 0] and sub["prompts"] == ["box"]
