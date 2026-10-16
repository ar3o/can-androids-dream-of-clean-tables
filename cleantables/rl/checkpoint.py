"""Saving and loading training checkpoints, including those written by the research code this package was ported from.

Research checkpoints name the parameters differently and also hold an unused residual MLP (`res.*`) and release head
(`rel.*`); `convert_research_checkpoint` drops those and renames the rest:

    own0..own3 -> actor.0..actor.3     adapt -> actor_adapter     grip_h -> grip_head     arm_emb -> arm_embedding
    lm, ls -> lift_mean, lift_std      p_none -> pointer_none
    p_tok, p_obs, p_ctx, p_enc, p_head -> pointer_token, pointer_obstacle, pointer_context, pointer_encoder,
                                          pointer_head
    c_tok, c_obs, c_lift, c_arm, c_enc, c_head -> critic_token, critic_obstacle, critic_lift, critic_arm,
                                                  critic_encoder, critic_head

Optimizer groups res / ptr / critic become actor / pointer / critic, with each parameter's Adam state moved by name.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

RESEARCH_PREFIXES = {
    "own0.": "actor.0.",
    "own1.": "actor.1.",
    "own2.": "actor.2.",
    "own3.": "actor.3.",
    "adapt.": "actor_adapter.",
    "grip_h.": "grip_head.",
    "arm_emb.": "arm_embedding.",
    "p_tok.": "pointer_token.",
    "p_obs.": "pointer_obstacle.",
    "p_ctx.": "pointer_context.",
    "p_enc.": "pointer_encoder.",
    "p_head.": "pointer_head.",
    "c_tok.": "critic_token.",
    "c_obs.": "critic_obstacle.",
    "c_lift.": "critic_lift.",
    "c_arm.": "critic_arm.",
    "c_enc.": "critic_encoder.",
    "c_head.": "critic_head.",
}
RESEARCH_NAMES = {"lm": "lift_mean", "ls": "lift_std", "p_none": "pointer_none", "log_std": "log_std"}
RESEARCH_DROPPED = ("res.", "rel.")
RESEARCH_GROUPS = {"res": "actor", "ptr": "pointer", "critic": "critic"}
RESEARCH_BUFFERS = ("lm", "ls")


def _rename(key: str) -> str | None:
    if key.startswith(RESEARCH_DROPPED):
        return None
    if key in RESEARCH_NAMES:
        return RESEARCH_NAMES[key]
    for prefix, replacement in RESEARCH_PREFIXES.items():
        if key.startswith(prefix):
            return replacement + key[len(prefix) :]
    raise KeyError(f"unknown parameter {key} in research checkpoint")


def is_research_checkpoint(ck: dict) -> bool:
    return "own0.weight" in ck["policy"]


def convert_research_checkpoint(ck: dict, policy: torch.nn.Module) -> dict:
    """Weights, optimizer state and learning rates of a research checkpoint, in this package's names and order."""
    research_sd = ck["policy"]
    sd = {}
    for k, v in research_sd.items():
        name = _rename(k)
        if name is not None:
            sd[name] = v
    # parameter order of the research model = its state_dict order without buffers; optimizer indices follow its groups
    research_params = [k for k in research_sd if k not in RESEARCH_BUFFERS]
    research_index_name = {}
    split_new = any(g["name"] == "new" for g in ck["opt"]["param_groups"])
    for g in ck["opt"]["param_groups"]:
        members = [k for k in research_params if _research_group(k, split_new) == g["name"]]
        assert len(members) == len(g["params"]), (g["name"], len(members), len(g["params"]))
        for idx, k in zip(g["params"], members, strict=True):
            research_index_name[idx] = _rename(k)
    groups = policy.param_groups()
    name_of = {id(p): n for n, p in policy.named_parameters()}
    index_of, param_groups, i = {}, [], 0
    research_groups = {RESEARCH_GROUPS[g["name"]]: g for g in ck["opt"]["param_groups"] if g["name"] in RESEARCH_GROUPS}
    for gname, params in groups.items():
        g = {k: v for k, v in research_groups[gname].items() if k != "params"}
        g["name"] = gname
        g["params"] = list(range(i, i + len(params)))
        for p in params:
            index_of[name_of[id(p)]] = i
            i += 1
        param_groups.append(g)
    state = {}
    for idx, s in ck["opt"]["state"].items():
        name = research_index_name[idx]
        if name is not None:
            state[index_of[name]] = s
    lr = {RESEARCH_GROUPS[k]: v for k, v in ck["lr"].items() if k in RESEARCH_GROUPS}
    return dict(ck, policy=sd, opt=dict(state=state, param_groups=param_groups), lr=lr)


def _research_group(key: str, split_new: bool = False) -> str:
    if split_new and key.startswith(("adapt.", "grip_h.")):  # newly initialised layers in their own optimizer group
        return "new"
    if key.startswith(("res.", "rel.", "own", "adapt.", "grip_h.")) or key == "log_std":
        return "res"
    if key.startswith("p_") or key.startswith("arm_emb"):
        return "ptr"
    return "critic"


def load_checkpoint(path: str | Path, policy: torch.nn.Module, device=None) -> dict:
    """Load a checkpoint (either format) and return it in this package's format; policy weights are loaded."""
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    if is_research_checkpoint(ck):
        ck = convert_research_checkpoint(ck, policy)
    policy.load_state_dict(ck["policy"])
    return ck


def save_atomic(obj, path: str | Path) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
