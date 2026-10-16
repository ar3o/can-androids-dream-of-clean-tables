"""PPO for the two-arm cleanup policy: one training block on one twin, as one rank of a data-parallel run.

    python -m cleantables.rl.ppo --run NAME --scene IMG_2533_t01 --num-envs 448 [--rank 0 --world 2 --port 29533]
        [--resume runs/rl/NAME/ckpt/last.pt] [--stop-after 60]

Normally started by `python -m cleantables.rl.train`, which rotates the ranks over the training twins. Gradients and KL
estimates are averaged over ranks (gloo), so every update averages over several twins. The PPO reward is the env's team
reward plus potential-based shaping (Ng et al. 1999) with, per arm, the progress of its object along the teacher's path.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from cleantables.paths import RL_RUNS

from .checkpoint import load_checkpoint, save_atomic
from .config import RLConfig, load_config
from .geometry import path_progress

BUFFER_KEYS = (
    "lift",
    "own",
    "oth",
    "tok",
    "tfrac",
    "u",
    "mu",
    "lp_res",
    "lp_ptr",
    "value",
    "logp_all",
    "prior",
    "near",
    "obst",
)
BOOL_KEYS = ("mask", "active", "needs")
WINDOW = 200  # finished episodes in the moving statistics
LOG_PROB_CLAMP = 20.0


def path_potential(env) -> torch.Tensor:
    """(N, 2) per arm: progress of its object along the teacher's path minus its distance from it (0 without a path)."""
    pc, _ = env.object_state()
    num_items, tg = env.num_items, env.teacher
    v = torch.zeros(env.num_envs, 2, device=env.device)
    for ai in range(2):
        asg = env.assigned[:, ai]
        p = pc[env.arange, env.item_obj[asg.clamp(max=num_items - 1)]]
        val = path_progress(tg.path[:, ai], p)
        v[:, ai] = torch.where((asg < num_items) & tg.ok[:, ai], val, torch.zeros_like(val))
    return v


class Trainer:
    def __init__(
        self,
        cfg: RLConfig,
        env,
        policy,
        prior,
        frames,
        run: str,
        rank: int = 0,
        world: int = 1,
        port: int = 29533,
        seed: int = 0,
        stop_after: int = 0,
    ):
        self.cfg, self.c = cfg, cfg.ppo
        self.env, self.policy, self.prior, self.frames = env, policy, prior, frames
        self.rank, self.world, self.seed, self.stop_after = rank, world, seed, stop_after
        self.run_dir = RL_RUNS / run
        (self.run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        c = self.c
        groups = policy.param_groups()
        self.opt = torch.optim.Adam(
            [dict(params=v, lr=(c.lr_actor if k == "actor" else c.lr), name=k) for k, v in groups.items()], lr=c.lr
        )
        self.lr = {"actor": c.lr_actor, "pointer": c.lr, "critic": c.lr}
        self.start_it, self.env_steps = 0, 0
        self.port = port
        self.dist = None

    # ------------------------------------------------------------------ setup
    def resume(self, path) -> None:
        ck = load_checkpoint(path, self.policy, self.env.device)
        self.opt.load_state_dict(ck["opt"])
        self.start_it, self.env_steps, self.lr = ck["it"] + 1, ck["env_steps"], ck["lr"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        print(
            f"resumed {path} at it {ck['it']}, {self.env_steps} env steps, lr {self.lr}, "
            f"weight sum {self.weight_sum():.6f}",
            flush=True,
        )

    def init_distributed(self) -> None:
        if self.world <= 1:
            return
        import datetime

        import torch.distributed as dist

        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{self.port}",
            rank=self.rank,
            world_size=self.world,
            timeout=datetime.timedelta(minutes=30),
        )
        self.dist = dist
        with torch.no_grad():
            for p in self.policy.parameters():
                dist.broadcast(p.data, 0)
        if self.rank != 0:
            s = self.seed + 7919 * self.rank + self.start_it
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
        print(
            f"rank {self.rank}/{self.world} scene {self.env.scene_json['episode']} weight sum {self.weight_sum():.6f}",
            flush=True,
        )

    def allreduce_mean(self, t: torch.Tensor) -> torch.Tensor:
        if self.dist is not None:
            self.dist.all_reduce(t)
            t /= self.world
        return t

    def weight_sum(self) -> float:
        return sum(float(p.detach().double().sum()) for p in self.policy.parameters())

    def save(self, it: int) -> None:
        ck = dict(
            policy=self.policy.state_dict(),
            opt=self.opt.state_dict(),
            it=it,
            env_steps=self.env_steps,
            lr=self.lr,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(),
            config=self.cfg.to_dict(),
            wsum=self.weight_sum(),
        )
        path = self.run_dir / "ckpt" / f"model_{it}.pt"
        save_atomic(ck, path)
        save_atomic(ck, self.run_dir / "ckpt" / "last.pt")
        print(f"saved it {it} (weight sum {ck['wsum']:.6f}) -> {path}", flush=True)

    # ------------------------------------------------------------------ rollout
    def _buffer(self, buffers: dict, key: str, shape, dtype=torch.float32) -> torch.Tensor:
        """Rollout storage (steps_per_env, *shape), allocated on first use."""
        if key not in buffers:
            buffers[key] = torch.zeros((self.c.steps_per_env, *shape), dtype=dtype, device=self.env.device)
        return buffers[key]

    def rollout(self, obs, phi_prev, buffers, stats):
        from .policy import act

        env, c, n, num_items = self.env, self.c, self.env.num_envs, self.env.num_items
        tg = env.teacher
        for t in range(c.steps_per_env):
            t0 = time.time()
            a = act(self.policy, self.prior, self.frames, obs, env, self.cfg.ppo.action_clip)
            torch.cuda.synchronize()
            t1 = time.time()
            stats["t_pol"] += t1 - t0
            placed_prev, skip_prev = env.placed.clone(), env.skip.clone()
            asg_prev, t_plan_prev = env.assigned.clone(), tg.t_plan.clone()
            length_prev = (tg.path[:, :, 1:] - tg.path[:, :, :-1]).norm(dim=-1).sum(-1)
            env.set_applied_action(a["applied"])
            obs2, rew, done, info = env.step(a["action"], a["ptr"])
            torch.cuda.synchronize()
            stats["t_env"] += time.time() - t1
            rew, done = rew.clone(), done.clone()
            newly = env.placed & ~placed_prev
            phi_new = path_potential(env)
            # one potential per step: an arm whose object changed or whose path was re-planned contributes no shaping on
            # that step, so finishing an object or a re-plan is never charged
            same = (env.assigned == asg_prev) & (tg.t_plan == t_plan_prev)
            shaping_arm = c.shape_w * (c.gamma * phi_new - phi_prev) * same.float()
            for ai in range(2):
                item = asg_prev[:, ai].clamp(max=num_items - 1)
                # an object just put down is a goal state: its potential is the full path length
                put = (asg_prev[:, ai] < num_items) & newly[env.arange, item] & tg.ok[:, ai]
                shaping_arm[:, ai] = torch.where(
                    put, c.shape_w * (c.gamma * length_prev[:, ai] - phi_prev[:, ai]), shaping_arm[:, ai]
                )
            shaping = torch.where(done, torch.zeros_like(done, dtype=torch.float32), shaping_arm.sum(1))
            r = rew + shaping
            success = torch.zeros(n, dtype=torch.bool, device=env.device)
            if "final" in info:
                fin = info["final"]
                success[torch.as_tensor(fin["env"], device=env.device)] = torch.as_tensor(
                    fin["success"], device=env.device
                ).bool()
                for j, e in enumerate(fin["env"]):
                    if bool(skip_prev[e]):
                        continue
                    stats["fin"].append(
                        dict(
                            placed=float(fin["placed"][j]),
                            success=float(fin["success"][j]),
                            arm=float(fin["arm_contact_steps"][j]),
                            knocked=float(fin["knocked"][j]),
                            ret=float(fin["ret"][j]),
                            nt=float(fin["nontarget_steps"][j]),
                            cls=[
                                (cname, bool(env.ep_lifted[e, i]), bool(env.ep_placed[e, i]))
                                for i, cname in enumerate(env.item_cls_names)
                            ],
                        )
                    )
            timeout = done & ~success
            r = r + c.gamma * a["value"] * timeout.float()  # time-limit bootstrap with V of the last pre-terminal state
            for k in BUFFER_KEYS:
                self._buffer(buffers, k, a[k].shape)[t] = a[k]
            for k in BOOL_KEYS:
                self._buffer(buffers, k, a[k].shape, torch.bool)[t] = a[k]
            self._buffer(buffers, "ptr", a["ptr"].shape, torch.long)[t] = a["ptr"]
            self._buffer(buffers, "done", done.shape, torch.bool)[t] = done
            self._buffer(buffers, "rew", r.shape)[t] = r
            stats["raw"] += float(rew.mean())
            stats["shaped"] += float(r.mean())
            stats["shaping"] += float(shaping.mean())
            stats["invalid"] += float(info["invalid_pointer"].sum())
            stats["nan_resets"] += int(info.get("nan_reset", 0))
            if bool(a["needs"].any()):
                stats["ent"].append(float(a["ent_ptr"][a["needs"]].mean()))
                stats["decisions"] += int(a["needs"].sum())
            if bool(a["active"].any()):
                stats["mu_abs"].append(float(a["mu"][a["active"]].abs().mean()))
            phi_prev = phi_new
            obs = obs2
        return obs, phi_prev

    def advantages(self, obs, buffers):
        c, env = self.c, self.env
        with torch.no_grad():
            last_value = self.policy.value(
                obs["tokens"].clone(),
                obs["lift"].clone(),
                obs["own_ee"].clone(),
                obs["needs_pointer"].clone(),
                obs["t"].float() / env.max_steps,
                obs["obst"].clone(),
            )
            v, r, d = buffers["value"], buffers["rew"], buffers["done"].float()
            adv = torch.zeros_like(r)
            last = torch.zeros(env.num_envs, device=env.device)
            for t in reversed(range(c.steps_per_env)):
                next_v = last_value if t == c.steps_per_env - 1 else v[t + 1]
                not_done = 1.0 - d[t]
                delta = r[t] + c.gamma * next_v * not_done - v[t]
                last = delta + c.gamma * c.lam * not_done * last
                adv[t] = last
        return adv, adv + v

    def critic_only(self, it: int) -> bool:
        return it < self.c.warmup_iters

    def update(self, it: int, buffers, adv, ret):
        from .policy import masked_logits

        c, pol, frames = self.c, self.policy, self.frames
        pol.train()
        m_total = c.steps_per_env * self.env.num_envs
        flat = {k: v.reshape((m_total,) + tuple(v.shape[2:])) for k, v in buffers.items()}
        adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
        adv_n = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        std_old = pol.log_std.detach().exp().clone()
        progress = max(0.0, 1.0 - it / max(1, c.ent_decay_iters))
        ent_coef = c.ent_pointer_final + (c.ent_pointer - c.ent_pointer_final) * progress
        mb = m_total // c.minibatches
        losses = dict(pg_c=[], pg_p=[], vl=[], ent_p=[], kl=[], klp=[], clipfrac=[])
        nan_skips = 0
        for _ in range(c.epochs):
            perm = torch.randperm(m_total, device=self.env.device)
            for b in range(c.minibatches):
                idx = perm[b * mb : (b + 1) * mb]
                lift, own, oth, tok, tfrac = (
                    flat["lift"][idx],
                    flat["own"][idx],
                    flat["oth"][idx],
                    flat["tok"][idx],
                    flat["tfrac"][idx],
                )
                a_ = adv_n[idx][:, None]
                near, obst = flat["near"][idx], flat["obst"][idx]
                mu = pol.action_mean(lift, own, oth, near, flat["prior"][idx])
                std = pol.log_std.exp()
                dist = torch.distributions.Normal(mu, std)
                lp = dist.log_prob(flat["u"][idx]).sum(-1)
                actv = flat["active"][idx].float()
                n_act = actv.sum().clamp(min=1)
                ratio = (lp - flat["lp_res"][idx]).clamp(-LOG_PROB_CLAMP, LOG_PROB_CLAMP).exp()
                s = torch.minimum(ratio * a_, ratio.clamp(1 - c.clip, 1 + c.clip) * a_)
                pg_c = -(s * actv).sum() / n_act
                lg = pol.pointer_logits(frames.item_tokens(tok), own, oth, tfrac, frames.obstacle_tokens(obst))
                dp = torch.distributions.Categorical(logits=masked_logits(lg, flat["mask"][idx]))
                lpp = dp.log_prob(flat["ptr"][idx])
                ndm = flat["needs"][idx].float()
                n_dec = ndm.sum().clamp(min=1)
                rp = (lpp - flat["lp_ptr"][idx]).clamp(-LOG_PROB_CLAMP, LOG_PROB_CLAMP).exp()
                sp = torch.minimum(rp * a_, rp.clamp(1 - c.clip, 1 + c.clip) * a_)
                pg_p = -(sp * ndm).sum() / n_dec
                ent_p = (dp.entropy() * ndm).sum() / n_dec
                v = pol.value(tok, lift, own, flat["needs"][idx], tfrac, obst)
                vl = (v - ret_f[idx]).pow(2).mean()
                loss = pg_c + pg_p + c.vf_coef * vl - ent_coef * ent_p
                # Gaussian KL to the frozen lift policy (same std), decaying over kl_decay_iters
                beta = c.kl_prior_final + (c.kl_prior - c.kl_prior_final) * max(
                    0.0, 1.0 - it / max(1, c.kl_decay_iters)
                )
                leash = ((mu[..., :23] ** 2) / (2 * std[:23] ** 2)).sum(-1)
                loss = loss + beta * (leash * actv).sum() / n_act
                frozen = self.critic_only(it)
                if frozen:
                    loss = c.vf_coef * vl
                loss = loss + c.pointer_min_entropy_w * torch.relu(c.pointer_min_entropy - ent_p)
                with torch.no_grad():
                    mo = flat["mu"][idx]
                    kl = (torch.log(std / std_old) + (std_old.pow(2) + (mo - mu).pow(2)) / (2 * std.pow(2)) - 0.5).sum(
                        -1
                    )
                    kl = float((kl * actv).sum() / n_act)
                    lo = flat["logp_all"][idx]
                    klp = float(((lo.exp() * (lo - dp.logits)).sum(-1) * ndm).sum() / n_dec)
                    cf = float((((ratio - 1).abs() > c.clip).float() * actv).sum() / n_act)
                synced = self.allreduce_mean(
                    torch.tensor([kl, klp, float(not bool(torch.isfinite(loss)))], device=self.env.device)
                )
                kl, klp = float(synced[0]), float(synced[1])
                if float(synced[2]) > 0:  # a non-finite loss on any rank: every rank skips this minibatch
                    nan_skips += 1
                    continue
                if c.desired_kl > 0 and not frozen:  # rsl_rl-style adaptive lr, held while the actor is frozen (KL ~ 0)
                    for gname, k_ in (("actor", kl), ("pointer", klp)):
                        if k_ > 2 * c.desired_kl:
                            self.lr[gname] = max(c.lr_bounds[0], self.lr[gname] / 1.5)
                        elif c.desired_kl / 2 > k_ > 0:
                            self.lr[gname] = min(c.lr_bounds[1], self.lr[gname] * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr[g["name"]]
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.dist is not None:
                    params = list(pol.parameters())
                    grads = torch.cat(
                        [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]
                    )
                    self.allreduce_mean(grads)
                    o = 0
                    for p in params:
                        p.grad = grads[o : o + p.numel()].view_as(p).clone()
                        o += p.numel()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    pol.log_std.clamp_(math.log(c.std_bounds[0]), math.log(c.std_bounds[1]))
                for k_, v_ in (
                    ("pg_c", float(pg_c.detach())),
                    ("pg_p", float(pg_p.detach())),
                    ("vl", float(vl.detach())),
                    ("ent_p", float(ent_p.detach())),
                    ("kl", kl),
                    ("klp", klp),
                    ("clipfrac", cf),
                ):
                    losses[k_].append(v_)
        return losses, nan_skips, ent_coef, adv_f

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        env, c, n = self.env, self.c, self.env.num_envs
        if self.rank == 0:
            (self.run_dir / f"config_it{self.start_it}.json").write_text(json.dumps(self.cfg.to_dict(), indent=1))
        obs = env.reset()
        if c.stagger:
            with torch.inference_mode():
                env.t[:] = torch.randint(0, int(env.max_steps * 0.9), (n,), device=env.device)
                env.skip[:] = True
            obs = env.reset()
        buffers: dict = {}
        phi_prev = path_potential(env)
        window: list = []
        log_name = "log.jsonl" if self.rank == 0 else f"log_r{self.rank}.jsonl"
        print(f"train {self.run_dir.name}: {n} envs, {env.num_items} items, start it {self.start_it}", flush=True)
        with open(self.run_dir / log_name, "a") as log_f:
            for it in range(self.start_it, c.iters):
                t_it = time.time()
                stats = dict(
                    raw=0.0,
                    shaped=0.0,
                    shaping=0.0,
                    invalid=0.0,
                    ent=[],
                    fin=[],
                    decisions=0,
                    mu_abs=[],
                    t_env=0.0,
                    t_pol=0.0,
                    nan_resets=0,
                )
                self.policy.eval()
                obs, phi_prev = self.rollout(obs, phi_prev, buffers, stats)
                t_roll = time.time() - t_it
                t_upd = time.time()
                self.env_steps += n * c.steps_per_env * self.world
                adv, ret = self.advantages(obs, buffers)
                losses, nan_skips, ent_coef, adv_f = self.update(it, buffers, adv, ret)
                finite = all(bool(torch.isfinite(p).all()) for p in self.policy.parameters())
                window = (window + stats["fin"])[-WINDOW:]
                rec = self._log_record(
                    it, t_it, t_roll, t_upd, stats, window, losses, nan_skips, ent_coef, adv_f, buffers, finite
                )
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(
                    f"it {it} fps {rec['env_steps_per_s']:.0f} {rec['it_s']:.1f} s  "
                    f"raw {rec['raw_team_rew_per_step']:.4f} "
                    f"shaped {rec['shaped_rew_per_step']:.4f}  finished {rec['n_fin']} placed {rec['placed_mean']} "
                    f"w200 {rec['w200_placed_mean']} all {rec['w200_all_placed']}  H_ptr {rec['ptr_entropy']} "
                    f"std {rec['res_std']:.3f} kl {rec['kl']} klp {rec['klp']} vl {rec['vl']} "
                    f"lr {self.lr['actor']:.1e}/{self.lr['pointer']:.1e}",
                    flush=True,
                )
                if not finite:
                    print("non-finite parameters: stopping (ckpt/last.pt is the last good checkpoint)", flush=True)
                    break
                done_block = self.stop_after > 0 and it + 1 - self.start_it >= self.stop_after
                if self.rank == 0 and ((it + 1) % c.ckpt_every == 0 or done_block):
                    self.save(it)
                if done_block:
                    print(f"block of {self.stop_after} iterations done on {env.scene_json['episode']}", flush=True)
                    break

    def _log_record(
        self, it, t_it, t_roll, t_upd, st, window, losses, nan_skips, ent_coef, adv_f, buffers, finite
    ) -> dict:
        env, c = self.env, self.c

        def mean_of(xs, k):
            return float(np.mean([x[k] for x in xs])) if xs else None

        def per_class(xs):
            out = {}
            for x in xs:
                for cname, lifted, placed in x["cls"]:
                    d = out.setdefault(cname, dict(slots=0, lifted=0, placed=0))
                    d["slots"] += 1
                    d["lifted"] += int(lifted)
                    d["placed"] += int(placed)
            return out

        fin = st["fin"]
        tg = env.teacher
        return dict(
            it=it,
            env_steps=self.env_steps,
            time=time.time(),
            it_s=time.time() - t_it,
            env_steps_per_s=env.num_envs * c.steps_per_env / t_roll,
            raw_team_rew_per_step=st["raw"] / c.steps_per_env,
            shaped_rew_per_step=st["shaped"] / c.steps_per_env,
            shaping_per_step=st["shaping"] / c.steps_per_env,
            n_fin=len(fin),
            placed_mean=mean_of(fin, "placed"),
            all_placed_rate=mean_of(fin, "success"),
            arm_contact_steps=mean_of(fin, "arm"),
            knocked=mean_of(fin, "knocked"),
            nontarget_steps=mean_of(fin, "nt"),
            ep_ret=mean_of(fin, "ret"),
            w200_n=len(window),
            w200_placed_mean=mean_of(window, "placed"),
            w200_all_placed=mean_of(window, "success"),
            w200_arm_contact_steps=mean_of(window, "arm"),
            w200_knocked=mean_of(window, "knocked"),
            invalid_pointers=st["invalid"],
            ptr_decisions=st["decisions"],
            ptr_entropy=float(np.mean(st["ent"])) if st["ent"] else None,
            res_std=float(self.policy.log_std.exp().mean()),
            res_mu_abs=float(np.mean(st["mu_abs"])) if st["mu_abs"] else None,
            value_mean=float(buffers["value"].mean()),
            adv_std=float(adv_f.std()),
            lr_res=self.lr["actor"],
            lr_ptr=self.lr["pointer"],
            lr_critic=self.lr["critic"],
            t_env_s=st["t_env"],
            t_policy_s=st["t_pol"],
            t_update_s=time.time() - t_upd,
            **{k: (float(np.mean(v)) if v else None) for k, v in losses.items()},
            nan_skips=nan_skips,
            params_finite=finite,
            nan_env_resets=st["nan_resets"],
            ent_coef=ent_coef,
            scene=env.scene_json["episode"],
            w200_per_class=per_class(window),
            per_class=per_class(fin),
            spawn=dict(checked=env.spawn_stats["checked"], rejected=env.spawn_stats["rejected"]),
            wm_cost=dict(tg.cost),
            wm_dest=dict(tg.dest_stats),
        )


def prepare(scene_name: str, num_envs: int, seed: int, video: bool = False, single_arm: bool = False):
    """Twin scene and simulation config; call before `launch_simulation` (which needs the config)."""
    from .scene import make_sim_cfg
    from .twin_scene import load_twin

    scene = load_twin(scene_name)
    sim_cfg = make_sim_cfg(scene, num_envs, seed, single_arm)
    if video:
        from .video import add_video_cameras

        add_video_cameras(sim_cfg, scene)
    return scene, sim_cfg


def build(cfg: RLConfig, scene: dict, sim_cfg, seed: int, autoreset: bool = True):
    """Task env, frozen prior, policy and arm frames for one twin. Call inside `launch_simulation`."""
    from .env import CleanupEnv
    from .lift_prior import JOINT_ORDER, LiftActor, load_prior_weights
    from .policy import ArmFrames, Policy
    from .scene import TwinSim

    env = CleanupEnv(scene, cfg.env, TwinSim(sim_cfg, scene), seed=seed, autoreset=autoreset)
    names = list(env.sim.robots["a"].joint_names)
    assert names == JOINT_ORDER, f"simulator joint order {names} differs from the lift policy's"
    weights = load_prior_weights()
    prior = LiftActor(weights, env.device)
    policy = Policy(weights, len(cfg.env.class_vocab), env.side_dim(), cfg.ppo.init_std).to(env.device)
    print(
        f"{scene['episode']}: items {env.item_names} ({env.item_cls_names}), token columns {env.tokens.idx}", flush=True
    )
    return env, prior, policy, ArmFrames(env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; outputs go to runs/rl/<run>/")
    ap.add_argument("--scene", required=True, help="training twin (folder name under data/twins/)")
    ap.add_argument("--num-envs", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="YAML config (default configs/rl.yaml)")
    ap.add_argument(
        "--resume", default="", help="checkpoint to continue from (this package's or a research checkpoint)"
    )
    ap.add_argument(
        "--stop-after", type=int, default=0, help="save and exit after this many iterations (scene rotation)"
    )
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--port", type=int, default=29533)
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)

    scene, sim_cfg = prepare(args.scene, args.num_envs, args.seed, single_arm=cfg.env.single_arm)
    with launch_simulation(sim_cfg, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        env, prior, policy, frames = build(cfg, scene, sim_cfg, args.seed)
        trainer = Trainer(
            cfg,
            env,
            policy,
            prior,
            frames,
            args.run,
            rank=args.rank,
            world=args.world,
            port=args.port,
            seed=args.seed,
            stop_after=args.stop_after,
        )
        if args.resume:
            trainer.resume(args.resume)
        trainer.init_distributed()
        trainer.train()
        env.close()


if __name__ == "__main__":
    main()
