"""Fast checks of the RL stage that need neither a GPU nor Isaac Sim."""

import math

import numpy as np
import pytest
import torch

from cleantables.rl.config import load_config
from cleantables.rl.evaluate import wilson
from cleantables.rl.geometry import lookahead_point, path_progress, points_ahead, quat_apply, quat_apply_inv, quat_mul
from cleantables.rl.layouts import in_reach


def straight_path(n=33, length=1.0):
    x = torch.linspace(0, length, n)
    return torch.stack([x, torch.zeros(n), torch.zeros(n)], -1)[None]


def test_quaternion_round_trip():
    q = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    v = torch.randn(5, 3)
    assert torch.allclose(quat_apply_inv(q, quat_apply(q, v)), v, atol=1e-5)
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(quat_mul(q, ident.expand(5, -1)), q)


def test_path_progress_and_lookahead():
    path = straight_path()
    p = torch.tensor([[0.5, 0.1, 0.0]])
    assert path_progress(path, p).item() == pytest.approx(0.5 - 0.1, abs=1e-5)
    ahead = lookahead_point(path, p, 0.1)
    assert ahead[0, 0].item() > 0.5
    end = lookahead_point(path, torch.tensor([[1.0, 0.0, 0.0]]), 0.1)
    assert torch.allclose(end, path[:, -1])
    pts = points_ahead(path, p, (0.05, 0.10, 2.0))
    assert pts.shape == (1, 3, 3)
    assert pts[0, 0, 0].item() == pytest.approx(0.55, abs=1e-5)
    assert pts[0, 2, 0].item() == pytest.approx(1.0, abs=1e-5)  # clamped at the end of the path


def test_reach_box_follows_arm_yaw():
    base = np.array([0.0, 0.5])
    in_front = base + np.array([0.0, -0.5])  # yaw +90 deg: the arm reaches towards -y
    assert in_reach(base, in_front, math.pi / 2)
    assert not in_reach(base, in_front, -math.pi / 2)


def test_wilson_interval():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and 0.0 < lo < 0.5 < hi < 1.0
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_default_config_matches_yaml():
    cfg = load_config()
    assert cfg.ppo.shape_w == 10.0 and cfg.env.class_vocab == ("fruit", "tool")
    assert len(cfg.rotation.train_twins) == 19 and len(cfg.rotation.heldout_twins) == 7
    assert not set(cfg.rotation.train_twins) & set(cfg.rotation.heldout_twins)


def _fake_prior_weights():
    dims = [1595, 512, 256, 128, 23]
    w = {f"mlp.{2 * i}.weight": torch.randn(dims[i + 1], dims[i]) for i in range(4)}
    w.update({f"mlp.{2 * i}.bias": torch.randn(dims[i + 1]) for i in range(4)})
    w["obs_normalizer._mean"] = torch.randn(1, 1595)
    w["obs_normalizer._std"] = torch.rand(1, 1595) + 0.5
    return w


RESEARCH_ORDER = [
    "log_std",
    "p_none",
    "lm",
    "ls",
    "arm_emb",
    "res",
    "rel",
    "p_obs",
    "c_obs",
    "own0",
    "own1",
    "own2",
    "own3",
    "adapt",
    "grip_h",
    "p_tok",
    "p_ctx",
    "p_enc",
    "p_head",
    "c_tok",
    "c_lift",
    "c_arm",
    "c_enc",
    "c_head",
]


def test_research_checkpoint_conversion():
    from cleantables.rl.checkpoint import (
        RESEARCH_NAMES,
        RESEARCH_PREFIXES,
        _research_group,
        convert_research_checkpoint,
    )
    from cleantables.rl.policy import Policy

    policy = Policy(_fake_prior_weights(), vocab=2, side_dim=116, init_std=0.3)
    inverse = {v: k for k, v in RESEARCH_PREFIXES.items()}
    inverse_names = {v: k for k, v in RESEARCH_NAMES.items()}
    research = {}
    for k, v in policy.state_dict().items():
        if k in inverse_names:
            research[inverse_names[k]] = v.clone() + 1
            continue
        prefix = next(p for p in inverse if k.startswith(p))
        research[inverse[prefix] + k[len(prefix) :]] = v.clone() + 1
    research.update({"res.0.weight": torch.zeros(3, 3), "rel.0.weight": torch.zeros(2, 2)})
    research = {
        k: research[k] for name in RESEARCH_ORDER for k in list(research) if k == name or k.startswith(name + ".")
    }
    params = [k for k in research if k not in ("lm", "ls")]
    groups, state, i = [], {}, 0
    for g in ("res", "ptr", "critic"):
        members = [k for k in params if _research_group(k) == g]
        groups.append(dict(name=g, lr=1e-4, params=list(range(i, i + len(members)))))
        for k in members:
            state[i] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.full_like(research[k], 7.0),
                "exp_avg_sq": torch.ones_like(research[k]),
            }
            i += 1
    ck = dict(
        policy=research, opt=dict(state=state, param_groups=groups), lr={"res": 1e-5, "ptr": 2e-4, "critic": 3e-4}
    )
    out = convert_research_checkpoint(ck, policy)
    policy.load_state_dict(out["policy"])
    assert torch.allclose(policy.actor[0].weight, research["own0.weight"])
    opt = torch.optim.Adam([dict(params=v, name=k) for k, v in policy.param_groups().items()], lr=1e-3)
    opt.load_state_dict(out["opt"])
    for p in policy.parameters():
        assert opt.state[p]["exp_avg"].shape == p.shape
    assert out["lr"] == {"actor": 1e-5, "pointer": 2e-4, "critic": 3e-4}
