"""One centralised policy for both arms, weights shared across arms plus a learned arm embedding.

- actor: per arm a trainable copy of the lift-policy MLP. The extra inputs (own and other palm, arm embedding, nearest
  boxes, the other arm's state, the teacher's path ahead) enter its first layer through a zero-initialised adapter,
  and a zero-initialised head sets the grip strength, so at initialisation it equals the frozen lift policy exactly.
- pointer: a transformer over [arm context, 'none', one token per item, one per box/container] that picks each arm's
  item. Items are given in each arm's own frame with mine/other columns, so the shared weights see a symmetric problem.
- critic: a transformer over all item and box tokens plus one token per arm and the episode time.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .config import ARMS
from .geometry import quat_apply_inv
from .lift_prior import ACTION_SCALE, HIDDEN, NUM_JOINTS, OBS_DIM, LiftActor

if TYPE_CHECKING:
    from .env import CleanupEnv

EMBED = 16
WIDTH = 64
POSE = 7
ARM_FRAME_TOKEN = 14  # box token + its position in the arm's frame
OBSTACLE_TOKEN = 11
LOGIT_MASK = -1e9


def encoder(d: int, layers: int = 2, heads: int = 4) -> torch.nn.TransformerEncoder:
    layer = torch.nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
    return torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def mlp(i: int, hidden: list[int], o: int) -> torch.nn.Sequential:
    layers, d = [], i
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ELU()]
        d = h
    layers.append(torch.nn.Linear(d, o))
    return torch.nn.Sequential(*layers)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, LOGIT_MASK)


class Policy(torch.nn.Module):
    def __init__(self, prior_weights: dict, vocab: int, side_dim: int, init_std: float):
        """side_dim: width of the per-arm extra input (obs['obst_near'])."""
        super().__init__()
        token_dim = 13 + vocab  # table-frame item token, see env.TokenLayout
        arm_token_dim = token_dim + 6  # + position and destination in the arm's own frame
        self.register_buffer("lift_mean", prior_weights["obs_normalizer._mean"][0].clone())
        self.register_buffer("lift_std", prior_weights["obs_normalizer._std"][0].clone() + 1e-2)
        self.arm_embedding = torch.nn.Embedding(2, EMBED)
        dims = [OBS_DIM, *HIDDEN, NUM_JOINTS]
        self.actor = torch.nn.ModuleList(torch.nn.Linear(dims[i], dims[i + 1]) for i in range(4))
        with torch.no_grad():
            for i, layer in enumerate(self.actor):
                layer.weight.copy_(prior_weights[f"mlp.{2 * i}.weight"])
                layer.bias.copy_(prior_weights[f"mlp.{2 * i}.bias"])
        self.actor_adapter = torch.nn.Linear(2 * POSE + EMBED + side_dim, HIDDEN[0])
        torch.nn.init.zeros_(self.actor_adapter.weight)
        torch.nn.init.zeros_(self.actor_adapter.bias)
        self.grip_head = torch.nn.Linear(HIDDEN[-1], 1)
        torch.nn.init.zeros_(self.grip_head.weight)
        torch.nn.init.zeros_(self.grip_head.bias)
        self.log_std = torch.nn.Parameter(torch.full((NUM_JOINTS + 1,), math.log(init_std)))

        self.pointer_token = torch.nn.Linear(arm_token_dim, WIDTH)
        self.pointer_obstacle = torch.nn.Linear(ARM_FRAME_TOKEN, WIDTH)
        self.pointer_context = torch.nn.Linear(2 * POSE + EMBED + 1, WIDTH)
        self.pointer_none = torch.nn.Parameter(torch.zeros(WIDTH))
        self.pointer_encoder = encoder(WIDTH)
        self.pointer_head = torch.nn.Linear(WIDTH, 1)

        self.critic_token = torch.nn.Linear(token_dim, WIDTH)
        self.critic_obstacle = torch.nn.Linear(OBSTACLE_TOKEN, WIDTH)
        self.critic_lift = mlp(OBS_DIM, [256], 64)
        self.critic_arm = torch.nn.Linear(64 + POSE + 1 + EMBED, WIDTH)
        self.critic_encoder = encoder(WIDTH)
        self.critic_head = mlp(WIDTH + 1, [128], 1)

    def param_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        groups = {"actor": [], "pointer": [], "critic": []}
        for name, p in self.named_parameters():
            if name.startswith(("actor", "grip_head")) or name == "log_std":
                groups["actor"].append(p)
            elif name.startswith(("pointer", "arm_embedding")):
                groups["pointer"].append(p)
            else:
                groups["critic"].append(p)
        return groups

    def _norm_lift(self, lift: torch.Tensor) -> torch.Tensor:
        return ((lift - self.lift_mean) / self.lift_std).clamp(-10, 10)

    def action_mean(self, lift, own, other, side, prior_action) -> torch.Tensor:
        """(N, 2, 24): the actor's joint action minus the frozen prior's (0 at initialisation), and the grip value."""
        n = lift.shape[0]
        emb = self.arm_embedding.weight[None].expand(n, -1, -1)
        elu = torch.nn.functional.elu
        extra = torch.cat([own, other, emb, side], -1)
        h = elu(self.actor[0]((lift - self.lift_mean) / self.lift_std) + self.actor_adapter(extra))
        h = elu(self.actor[1](h))
        h = elu(self.actor[2](h))
        return torch.cat([self.actor[3](h) - prior_action.detach(), self.grip_head(h)], -1)

    def pointer_logits(self, arm_tokens, own, other, tfrac, arm_obstacles) -> torch.Tensor:
        """arm_tokens (N, 2, I, D), arm_obstacles (N, 2, Ko, 14) -> logits (N, 2, I+1), the last one for 'none'."""
        n, _, num_items, _ = arm_tokens.shape
        emb = self.arm_embedding.weight[None].expand(n, -1, -1)
        ctx = self.pointer_context(torch.cat([own, other, emb, tfrac[:, None, None].expand(-1, 2, 1)], -1))
        parts = [ctx[:, :, None], self.pointer_none.expand(n, 2, 1, -1), self.pointer_token(arm_tokens)]
        if arm_obstacles.shape[2] > 0:
            parts.append(self.pointer_obstacle(arm_obstacles))
        seq = torch.cat(parts, 2)
        h = self.pointer_encoder(seq.reshape(n * 2, seq.shape[2], -1))
        lg = self.pointer_head(h[:, : num_items + 2]).squeeze(-1).reshape(n, 2, num_items + 2)
        return torch.cat([lg[:, :, 2:], lg[:, :, 1:2]], -1)

    def value(self, tokens, lift, own, needs, tfrac, obstacles) -> torch.Tensor:
        n = tokens.shape[0]
        emb = self.arm_embedding.weight[None].expand(n, -1, -1)
        arm = self.critic_arm(
            torch.cat([self.critic_lift(self._norm_lift(lift)), own, needs[..., None].float(), emb], -1)
        )
        parts = [arm, self.critic_token(tokens)]
        if obstacles.shape[1] > 0:
            parts.append(self.critic_obstacle(obstacles))
        h = self.critic_encoder(torch.cat(parts, 1))
        return self.critic_head(torch.cat([h.mean(1), tfrac[:, None]], -1)).squeeze(-1)


class ArmFrames:
    """Re-expresses table-frame tokens in each arm's base frame (the bases never move)."""

    def __init__(self, env: CleanupEnv):
        self.cols = env.tokens
        self.root_pos, self.root_quat = [], []
        for a in ARMS:
            r = env.sim.robots[a]
            self.root_pos.append((r.data.root_pos_w.torch[0] - env.origins[0]).clone())
            self.root_quat.append(r.data.root_quat_w.torch[0].clone())

    def item_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        """(N, I, D) -> (N, 2, I, D + 6): pos/dest in own frame | pos/dest | class | put down | taken by me / other |
        reachable by me / other | cooldown | teacher column."""
        c, out = self.cols, []
        for ai in range(2):
            o = 1 - ai
            q = self.root_quat[ai].expand(*tok.shape[:-1], 4)
            p = quat_apply_inv(q, tok[..., 0:3] - self.root_pos[ai])
            d = quat_apply_inv(q, tok[..., 3:6] - self.root_pos[ai])
            out.append(
                torch.cat(
                    [
                        p,
                        d,
                        tok[..., 0:6],
                        tok[..., 6 : c["placed"] + 1],
                        tok[..., c["ta"] + ai : c["ta"] + ai + 1],
                        tok[..., c["ta"] + o : c["ta"] + o + 1],
                        tok[..., c["ra"] + ai : c["ra"] + ai + 1],
                        tok[..., c["ra"] + o : c["ra"] + o + 1],
                        tok[..., c["cool"] : c["cool"] + 1],
                        tok[..., c["cool"] + 1 :],
                    ],
                    -1,
                )
            )
        return torch.stack(out, 1)

    def obstacle_tokens(self, obst: torch.Tensor) -> torch.Tensor:
        """(N, Ko, 11) -> (N, 2, Ko, 14): + the bottom centre in each arm's frame."""
        out = []
        for ai in range(2):
            p = quat_apply_inv(
                self.root_quat[ai].expand(obst.shape[0], obst.shape[1], 4), obst[..., 0:3] - self.root_pos[ai]
            )
            out.append(torch.cat([p, obst], -1))
        return torch.stack(out, 1)


@torch.inference_mode()
def greedy_pointer(env: CleanupEnv, obs: dict) -> torch.Tensor:
    """Baseline: each arm takes the valid reachable item nearest to its palm."""
    num_items = env.num_items
    pi = env.object_state()[0][:, env.item_obj]
    ptr = obs["assigned"].clone()
    for ai, a in enumerate(ARMS):
        d = (pi[..., :2] - env.palm_pos(a)[:, None, :2]).norm(dim=-1)
        ok = obs["ptr_mask"][:, ai, :num_items] & obs["tokens"][:, :, env.tokens["ra"] + ai].bool()
        d = torch.where(ok, d, torch.full_like(d, 9.0))
        pick = torch.where(ok.any(1), d.argmin(1), torch.full_like(ptr[:, ai], num_items))
        ptr[:, ai] = torch.where(obs["needs_pointer"][:, ai], pick, ptr[:, ai])
    return ptr


@torch.no_grad()
def act(
    policy: Policy,
    prior: LiftActor,
    frames: ArmFrames,
    obs: dict,
    env: CleanupEnv,
    action_clip: float,
    deterministic: bool = False,
    greedy: bool = False,
) -> dict:
    """Sample (or take the mode of) both arms' actions and pointers; returns the action and what PPO stores."""
    num_items, n = env.num_items, env.num_envs
    lift, own, other = obs["lift"].clone(), obs["own_ee"].clone(), obs["other_ee"].clone()
    tok, needs, asg = obs["tokens"].clone(), obs["needs_pointer"].clone(), obs["assigned"].clone()
    side, obst = obs["obst_near"].clone(), obs["obst"].clone()
    tfrac = obs["t"].float() / env.max_steps
    prior_action = prior(lift.reshape(n * 2, -1)).reshape(n, 2, NUM_JOINTS)
    mu = policy.action_mean(lift, own, other, side, prior_action)
    std = policy.log_std.exp()
    u = mu if deterministic else mu + std * torch.randn_like(mu)
    lg = policy.pointer_logits(frames.item_tokens(tok), own, other, tfrac, frames.obstacle_tokens(obst))

    # arm b chooses after arm a, with a's new choice removed; the mask actually used is stored so the PPO ratio is exact
    valid = (tok[..., env.tokens["placed"]] < 0.5) & (tok[..., env.tokens["cool"]] < 0.5)
    m0 = obs["ptr_mask"][:, 0].clone()
    greedy_ptr = greedy_pointer(env, obs).clone() if greedy else None
    d0 = torch.distributions.Categorical(logits=masked_logits(lg[:, 0], m0))
    if greedy:
        p0 = greedy_ptr[:, 0]
    else:
        p0 = masked_logits(lg[:, 0], m0).argmax(-1) if deterministic else d0.sample()
    a0 = torch.where(needs[:, 0], p0, asg[:, 0])
    v1 = valid & ~torch.nn.functional.one_hot(a0, num_items + 1)[:, :num_items].bool()
    m1 = torch.cat([v1, ~v1.any(1, keepdim=True)], -1)
    if env.cfg.single_arm:  # arm b is never given an object
        m1 = torch.zeros_like(m1)
        m1[:, -1] = True
        if greedy_ptr is not None:
            greedy_ptr[:, 1] = num_items
    d1 = torch.distributions.Categorical(logits=masked_logits(lg[:, 1], m1))
    if greedy:
        p1 = greedy_ptr[:, 1]
    else:
        p1 = masked_logits(lg[:, 1], m1).argmax(-1) if deterministic else d1.sample()
    ptr = torch.stack([p0, p1], 1)
    new_asg = torch.where(needs, ptr, asg)
    active = new_asg < num_items

    uc = u.clamp(-action_clip, action_clip)
    full = prior_action + uc[..., :NUM_JOINTS]
    # grip strength g in [0, 1] blends the finger command towards the open hand (g = 1 is the prior's own grip)
    fj = env.sim.finger_jids
    q = torch.stack([env.sim.robots[a].data.joint_pos.torch for a in ARMS], 1)
    open_action = (env.sim.finger_open[None, None] - q[..., fj]) / ACTION_SCALE
    g = (1.0 + uc[..., NUM_JOINTS:]).clamp(0.0, 1.0)
    full = full.clone()
    full[..., fj] = g * full[..., fj] + (1 - g) * open_action
    full = torch.where(active[..., None], full, torch.zeros_like(full))
    log_probs_all = torch.stack([masked_logits(lg[:, 0], m0), masked_logits(lg[:, 1], m1)], 1).log_softmax(-1)
    return dict(
        action=full.reshape(n, 46),
        applied=full,
        ptr=ptr,
        mask=torch.stack([m0, m1], 1),
        lp_ptr=torch.stack([d0.log_prob(p0), d1.log_prob(p1)], 1),
        logp_all=log_probs_all,
        ent_ptr=torch.stack([d0.entropy(), d1.entropy()], 1),
        u=u,
        mu=mu,
        lp_res=torch.distributions.Normal(mu, std).log_prob(u).sum(-1),
        active=active,
        needs=needs,
        value=policy.value(tok, lift, own, needs, tfrac, obst),
        lift=lift,
        own=own,
        oth=other,
        tok=tok,
        tfrac=tfrac,
        prior=prior_action,
        near=side,
        obst=obst,
    )
