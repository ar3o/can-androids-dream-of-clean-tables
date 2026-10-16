"""Fast CPU checks of the world-model package (no GPU, no data files)."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from cleantables.worldmodel.config import (
    PhysicsFeedbackConfig,
    TrainConfig,
    load_physics_feedback_config,
    load_train_config,
)
from cleantables.worldmodel.dataset import episode_folds, hashed_text_embedding, resample_path
from cleantables.worldmodel.model import N_PATH, WorldModel
from cleantables.worldmodel.teacher import shuffle_layout


def _batch(B=3, n_obj=4):
    geo = torch.zeros(B, 24, 8)
    geo[:, :n_obj, :2] = torch.rand(B, n_obj, 2)
    geo[:, :n_obj, 2:6] = 0.05
    geo[:, :n_obj, 7] = 1
    mask = torch.zeros(B, 24, dtype=torch.bool)
    mask[:, : n_obj + 1] = True
    kind = torch.zeros(B, 24, dtype=torch.long)
    kind[:, n_obj] = 1
    return dict(
        obj_geo=geo,
        obj_txt=torch.randn(B, 24, 384),
        obj_crop=torch.randn(B, 24, 768),
        obj_crop_ok=torch.ones(B, 24, dtype=torch.bool),
        obj_mask=mask,
        obj_kind=kind,
        hand_geo=torch.zeros(B, 2, 13),
        hand_q16=torch.zeros(B, 2, 16),
        hand_q_ok=torch.zeros(B, 2, dtype=torch.bool),
        vj_tok=torch.randn(B, 32, 1024),
        vj_ok=torch.ones(B, dtype=torch.bool),
        task_emb=torch.randn(B, 384),
    )


def test_model_heads_and_path_shapes():
    torch.manual_seed(0)
    m = WorldModel().eval()
    b = _batch()
    with torch.no_grad():
        E = m.encode(b, torch.tensor([0, 1, 2]))
        mover_logits, target_logits = m.pointers(E, torch.tensor([0, 1, 2]))
        c, rel, mdn = m.heads_from(E, torch.tensor([0, 1, 2]), torch.tensor([1, 24, 2]))
        paths = m.sample_paths_anchored(E, c, K=2, steps=2)
    assert mover_logits.shape == (3, 24) and target_logits.shape == (3, 26)
    assert mover_logits[:, 4:].max() < -1e3  # the table and empty slots are never movers
    assert rel.shape == (3, 6) and mdn["mu"].shape == (3, 5, 3)
    assert paths.shape == (3, 2, N_PATH, 6)


def test_checkpoint_parameter_names():
    keys = list(WorldModel().state_dict())
    for k in (
        "geo.0.weight",
        "null_geo",
        "disp.2.bias",
        "f_dec.layers.2.linear2.weight",
        "p_vid.0.weight",
        "seen.2.bias",
    ):
        assert k in keys
    assert sum(p.numel() for p in WorldModel().parameters()) == pytest.approx(12.6e6, rel=0.05)


def test_hashed_text_embedding_is_deterministic_unit():
    e = hashed_text_embedding(["Put the tools into the toolbox.", "a", "Put the tools into the toolbox."])
    assert e.shape == (3, 384)
    assert np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    assert np.array_equal(e[0], e[2]) and not np.allclose(e[0], e[1])


def test_episode_folds_whole_videos_and_single_fruit_episodes():
    keys = ["IMG_2533", "IMG_2534", "IMG_2516_t01", "IMG_2516_t02", "IMG_2516_t03"]
    folds = episode_folds(keys)
    assert folds == {"IMG_2533": 0, "IMG_2534": 1, "IMG_2516_t01": 0, "IMG_2516_t02": 1, "IMG_2516_t03": 2}


def test_resample_path_keeps_endpoints():
    P = np.array([[0, 0, 0], [1, 2, 0], [2, 2, 1]], np.float32)
    R = resample_path(P)
    assert R.shape == (32, 3)
    assert np.allclose(R[0], P[0]) and np.allclose(R[-1], P[-1])


def test_shuffle_layout_moves_only_task_objects_without_overlap():
    Z = dict(
        start_pos=np.array([[0, 0, 0], [0.3, 0, 0], [-0.3, 0.1, 0]], np.float32),
        extents=np.array([[0.1, 0.1, 0.1], [0.08, 0.05, 0.05], [0.3, 0.3, 0.1]], np.float32),
    )
    scene = dict(table=dict(x=[-0.7, 0.7], y=[-0.4, 0.4]))
    P = shuffle_layout(Z, scene, np.random.default_rng(0), [0, 1])
    assert np.array_equal(P[2], Z["start_pos"][2])
    r = 0.5 * np.hypot(Z["extents"][:, 0], Z["extents"][:, 1])
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.hypot(*(P[i, :2] - P[j, :2])) > r[i] + r[j]


def test_config_files_match_defaults():
    assert asdict(load_train_config()) == asdict(TrainConfig())
    assert asdict(load_physics_feedback_config()) == asdict(PhysicsFeedbackConfig())


def test_object_names_number_repeated_phrases():
    from cleantables.worldmodel.video import object_names

    objects = [dict(id=3, phrase="banana"), dict(id=1, phrase="banana"), dict(id=2, phrase="bowl")]
    assert object_names(objects, [1, 2, 3]) == {1: "banana 1", 2: "bowl", 3: "banana 2"}


def test_isotonic_calibration_is_monotone_interpolation():
    from cleantables.worldmodel.progress import apply_calibration

    table = dict(x=[0.3, 0.5, 0.7], y=[0.0, 0.4, 1.0])
    out = apply_calibration(table, np.array([0.1, 0.3, 0.6, 0.9]))
    assert np.allclose(out, [0.0, 0.0, 0.7, 1.0])
