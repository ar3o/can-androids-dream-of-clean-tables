"""Joint two-arm Franka cleanup env: one centralised policy controls both Franka Panda arms (parallel grippers).

The Franka counterpart of the Kuka-Allegro joint env, built on env_franka.TwinFrankaEnv and FrankaArmObs (the 1250-D
Isaac-Lift-Franka actor input); action 9 per arm = 7 arm joints + 2 gripper fingers, (N, 18) in total.

Build (the Isaac app must be launched first):

    from isaaclab.app import add_launcher_args, launch_simulation
    from cleantables.twin.robot.joint_env_franka import JointCleanupEnv, FrozenPriorPair, greedy_pointer
    cfg, SCENE, POSES = JointCleanupEnv.make_cfg("cl_coord_0", num_envs=64)
    with launch_simulation(cfg, args):
        env = JointCleanupEnv(cfg, SCENE, POSES, overlap=0.6)
        obs = env.reset()
        pri = FrozenPriorPair(env)                       # two weight-shared copies of the NVIDIA Lift actor (frozen prior)
        while True:
            ptr = greedy_pointer(env, obs)               # (N,2) discrete choice per arm; replaced by the policy's pointer head
            obs, rew, done, info = env.step(pri.act(obs), ptr)       # action (N,18); replaced by prior + zero-initialised residual
    python -m cleantables.twin.robot.joint_env_franka --scene cl_coord_0 --num_envs 64 --policy prior_greedy   # prior + throughput

ACTION  (N, 18): [arm a joint action (9) | arm b joint action (9)], the Lift convention: joint order = panda_joint1-7 then the two
        finger joints (robot.joint_names); the applied position target is q + 0.1 * action (as the Isaac-Lift task). Arm a is the +x one.
        An arm whose pointer is 'none' gets action 0 (holds its pose).
POINTER (N, 2) long, discrete per arm in [0, I]: i < I = "take in-scope object i" (I = number of in-scope objects, order of
        env.item_names), I = none. Re-evaluated every `ptr_every` control steps (obs["needs_pointer"] marks arms that may change it;
        between those steps the previous pointer is kept) and right after an arm released its object (auto_release mode).
        obs["ptr_mask"] (N,2,I+1) = valid options: an item is valid iff not yet placed, not taken by the other arm and not in the 2 s
        release cooldown; 'none' is valid only if no valid item exists. An invalid pointer is replaced by the first valid option and
        counted in info["invalid_pointer"].
OBS (dict of tensors on the GPU):
  lift    (N, 2, 1250)  per arm: the Isaac-Lift-Franka actor input (5-step history: object pose in the arm's base frame, goal command,
                        last action, proprio, finger contact forces, 64 object surface points) for the object the arm is assigned, built
                        by cleantables.robots.lift_policy_franka.FrankaArmObs. The command (goal) = release point 5 cm above the rim of
                        the assigned object's destination container (fruit -> fruit bowl, tool -> toolbox, can -> recycling box).
                        A 'none' arm gets the first item as a dummy target (its action is masked to 0).
  tokens  (N, I, 15)    per in-scope object (table frame, metres): pos 0-2 | destination centre 3-5 | class one-hot fruit/tool/can 6-8 |
                        placed 9 | taken by a / b 10-11 | reachable by a / b 12-13 (object and destination in the arm's Lift workspace) |
                        cooldown > 0 14. Names: env.item_names (a text embedding can be appended by the caller).
  other_ee (N, 2, 7)    the other arm's hand pose (pos 3 + quat xyzw) in each arm's own base frame (cross-arm input).
  own_ee   (N, 2, 7)    each arm's own hand pose in its base frame.
  ptr_mask (N,2,I+1) bool, needs_pointer (N,2) bool, assigned (N,2) long (I = none), t (N,) control steps (30 Hz) since the episode start.
REWARD  (N,) team reward per control step: +1 for every in-scope object newly placed (inside its destination container footprint
        shrunk by 1 cm, below the rim + 5 cm, speed < 0.05 m/s for 0.5 s: the physics judge); -1 if a placed object leaves again;
        -0.001 per step; -0.05 per step with arm-arm contact (> 1 N); -0.02 per step with a finger touching a movable non-assigned
        object (> 1 N); -0.5 for each newly knocked obstacle (> 3 cm).
DONE    all in-scope objects placed or timeout (max_seconds). Auto-reset: a done env is reset to a new random layout with overlapping
        reach (the returned obs is the first obs of the new episode; info["final"] holds the finished episode's statistics at the done
        indices).
info    dict: invalid_pointer (N,), final (dict of stats for the envs that just finished: placed, I, makespan_s, ret, arm_contact_steps,
        min_arm_dist, nontarget_steps, knocked, idle_s a/b).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

import cleantables.twin.robot.env_franka as E

E.FIXED.add("container/toolbox")
from cleantables.twin.robot.env_kuka import OPEN  # noqa: E402
from cleantables.twin.robot.env_franka import ARMS, POLICY, ROOT, TwinFrankaEnv as TwinRobotEnv, TwinRobotEnvCfg, build_scene_cfg, franka_mount  # noqa: E402
from cleantables.twin.robot.env_kuka import reach_b  # noqa: E402,F401  (only passed on to sample_layouts)
from cleantables.twin.robot.cleanup_scenes import sample_layouts  # noqa: E402

CKPT = "runs/franka_prior/ckpt/Isaac-Lift-Franka_physx_none_rsl_rl.pt"
CLS = ["fruit", "tool", "can"]


class JointCleanupEnv:
    w_placed = 1.0                     # weight of the +1 / -1 placed / lost events in the reward (0 = reward from the dense path progress only)

    @staticmethod
    def make_cfg(scene, num_envs, seed=0):
        S = franka_mount(json.load(open(os.path.join(ROOT, "data/twin_table", scene, "scene.json"))))
        POSES = {a: (tuple(p[0]), tuple(p[1])) for a, p in S["poses"].items()}
        cfg = TwinRobotEnvCfg()
        cfg.action_space = 18
        cfg.scene = build_scene_cfg(num_envs, S, POSES, record_cam=False)
        cfg.seed = seed
        return cfg, S, POSES

    def __init__(self, cfg, scene, poses, overlap=0.6, max_seconds=45.0, ptr_every=30, auto_release=False, release_h=0.05, mass_dr=0.15, seed=0,
                 w_time=0.001, w_arm=0.05, w_nt=0.02, w_knock=0.5, autoreset=True):
        from cleantables.robots.lift_policy import quat_apply_inv, sample_point_cloud
        from cleantables.robots.lift_policy_franka import FrankaArmObs as ArmObs
        self._ArmObs, self._qai = ArmObs, quat_apply_inv
        self.S, self.POSES = scene, poses
        self.env = env = TwinRobotEnv(cfg, scene)
        self.dev, self.N, self.K = env.device, env.num_envs, env.K
        dev, N, K = self.dev, self.N, self.K
        self.dt = env.step_dt
        self.origins = env.scene.env_origins
        objs = scene["objects"]
        self.names = [o["id"] for o in objs]
        role = [o["role"] for o in objs]
        self.role = role
        self.item_idx = [k for k in range(K) if role[k] == "task"]
        self.I = len(self.item_idx)
        self.item_names = [self.names[k] for k in self.item_idx]
        self.item_obj = torch.tensor(self.item_idx, device=dev)
        self.item_dst = torch.tensor([self.names.index(objs[k]["dest"]) for k in self.item_idx], device=dev)
        self.item_cls = torch.nn.functional.one_hot(torch.tensor([CLS.index(objs[k]["cls"]) for k in self.item_idx], device=dev), 3).float()
        self.mov_mask = torch.tensor([r in ("task", "obstacle", "spare") for r in role], device=dev)
        self.obst_idx = [k for k in range(K) if role[k] in ("obstacle", "spare")]
        self.ext = torch.tensor([o["extents"] for o in objs], device=dev)
        self.mass0 = torch.tensor([o["mass"] for o in objs], device=dev)
        self.pts = torch.stack([torch.from_numpy(sample_point_cloud(os.path.join(ROOT, o["usd"]).replace(".usda", ".obj"))) for o in objs]).to(dev)
        self.I_real = [__import__("warp").to_torch(o.root_view.get_inertias()).clone().reshape(N, -1, 9) for o in env.objs]
        self.overlap, self.mass_dr, self.ptr_every, self.auto_release, self.release_h = overlap, mass_dr, ptr_every, auto_release, release_h
        self.max_steps = int(max_seconds / self.dt)
        self.w = dict(time=w_time, arm=w_arm, nt=w_nt, knock=w_knock)
        self.autoreset = autoreset
        self.rng = np.random.default_rng(seed)
        self.ar = torch.arange(N, device=dev)
        self.obsb = {a: ArmObs(N, dev) for a in ARMS}
        self.base = {a: torch.tensor(poses[a][0][:2], device=dev) for a in ARMS}
        for a in ARMS:
            env.set_friction(env.robots[a])
        self.zero_act = torch.zeros(N, env.cfg.action_space, device=dev)
        self._init_state()
        env.reset()
        self._reset_envs(torch.arange(N, device=dev), settle=True)

    # ------------------------------------------------------------------ state
    def _init_state(self):
        dev, N, I, K = self.dev, self.N, self.I, self.K
        self.assigned = torch.full((N, 2), I, dtype=torch.long, device=dev)
        self.placed = torch.zeros(N, I, dtype=torch.bool, device=dev)
        self.rest_cnt = torch.zeros(N, I, dtype=torch.long, device=dev)
        self.cool = torch.zeros(N, I, dtype=torch.long, device=dev)
        self.t = torch.zeros(N, dtype=torch.long, device=dev)
        self.open_cd = torch.zeros(N, 2, dtype=torch.long, device=dev)
        self.at_goal_cnt = torch.zeros(N, 2, dtype=torch.long, device=dev)
        self.needs_ptr = torch.ones(N, 2, dtype=torch.bool, device=dev)
        self.ret = torch.zeros(N, device=dev)
        self.arm_c = torch.zeros(N, device=dev); self.nt_c = torch.zeros(N, device=dev)
        self.min_d = torch.full((N,), 9.0, device=dev)
        self.knocked = torch.zeros(N, K, dtype=torch.bool, device=dev)
        self.idle = torch.zeros(N, 2, device=dev)
        self.makespan = torch.full((N,), -1.0, device=dev)
        self.p_start = torch.zeros(N, K, 3, device=dev)
        self.goal = {a: torch.zeros(N, 3, device=dev) for a in ARMS}
        self.last_obs = None

    def _set_inertia(self, k, env_ids, add):
        I = self.I_real[k].clone()
        if add:
            I[..., 0] += 0.01; I[..., 4] += 0.01; I[..., 8] += 0.01
        self.env.objs[k].set_inertias_index(inertias=I[env_ids.cpu()].to(self.dev), env_ids=env_ids)

    def _reset_envs(self, ids, settle=False):
        env, dev, K, I = self.env, self.dev, self.K, self.I
        n = len(ids)
        for ai_, a in enumerate(ARMS):                       # restore real inertia of whatever was assigned
            ob = self.item_obj[self.assigned[ids, ai_].clamp(max=I - 1)]
            m = self.assigned[ids, ai_] < I
            for kk in torch.unique(ob[m]).tolist():
                self._set_inertia(kk, ids[m & (ob == kk)], False)
        xyz, yaw, info = sample_layouts(self.S, n, self.rng, reach_b, pboth_c=self.overlap, pboth_i=self.overlap * 0.8)
        self.layout_info = info
        failed = set(info["failed_layouts"])
        for k in range(K):
            pose = torch.zeros(n, 7, device=dev)
            if self.role[k] == "spare":
                pose[:, 0] = self.S["table"]["x"][1] + 0.5; pose[:, 2] = 0.05; pose[:, 6] = 1.0
            else:
                pose[:, :3] = torch.tensor(xyz[:, k], dtype=torch.float32, device=dev)
                yw = torch.tensor(yaw[:, k], dtype=torch.float32, device=dev)
                pose[:, 5] = torch.sin(yw / 2); pose[:, 6] = torch.cos(yw / 2)
            pose[:, :3] += self.origins[ids]
            env.objs[k].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.objs[k].write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
            if self.role[k] in ("task", "obstacle") and self.mass_dr > 0:
                env.objs[k].set_masses_index(masses=(self.mass0[k] * (1 + (torch.rand(n, 1, device=dev) * 2 - 1) * self.mass_dr)), env_ids=ids)
        for a in ARMS:
            r = env.robots[a]
            q = r.data.default_joint_pos.torch.clone()[ids]
            r.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)
            env.q_hold[a][ids] = q; env.q_ik[a][ids] = q; env._prev_mode[a][ids] = POLICY; env.mode[a][ids] = POLICY
            env.off_ema[a][ids] = 0.0
            env.finger_target[a][ids] = env.finger_default
            m = torch.zeros(self.N, dtype=torch.bool, device=dev); m[ids] = True
            self.obsb[a].reset(m)
        self.assigned[ids] = I; self.placed[ids] = False; self.rest_cnt[ids] = 0; self.cool[ids] = 0; self.t[ids] = 0; self.open_cd[ids] = 0
        self.at_goal_cnt[ids] = 0; self.needs_ptr[ids] = True; self.ret[ids] = 0; self.arm_c[ids] = 0; self.nt_c[ids] = 0; self.min_d[ids] = 9.0
        self.knocked[ids] = False; self.idle[ids] = 0; self.makespan[ids] = -1.0
        if settle:
            for _ in range(45):
                env.step(self.zero_act)
        pc, _ = self._ostate()
        self.p_start[ids] = pc[ids]
        for a in ARMS:
            env.palm_target[a][ids] = env.robots[a].data.body_pos_w.torch[ids, env.palm_bid]

    def _ostate(self):
        p = torch.stack([o.data.root_pos_w.torch for o in self.env.objs], 1) - self.origins[:, None]
        q = torch.stack([o.data.root_quat_w.torch for o in self.env.objs], 1)
        return p, q

    def _in_reach(self, a, xy):
        d = xy - self.base[a]
        xb, yb = d[..., 1], -d[..., 0]
        return (xb >= E.FR_X[0]) & (xb <= E.FR_X[1]) & (yb.abs() <= E.FR_Y)

    def _inside(self, p, kc, P):
        ar = self.ar
        c = P[ar, kc]; e = self.ext[kc]
        return ((p[:, 0] - c[:, 0]).abs() < e[:, 0] / 2 - 0.01) & ((p[:, 1] - c[:, 1]).abs() < e[:, 1] / 2 - 0.01) & (p[:, 2] < c[:, 2] + e[:, 2] + 0.05) & (p[:, 2] > -0.02)

    # ------------------------------------------------------------------ pointer / masks
    def _ptr_mask(self):
        N, I, dev = self.N, self.I, self.dev
        valid = ~self.placed & (self.cool == 0)                                      # (N,I)
        mask = torch.zeros(N, 2, I + 1, dtype=torch.bool, device=dev)
        for ai, a in enumerate(ARMS):
            other = self.assigned[:, 1 - ai]
            taken = torch.nn.functional.one_hot(other, I + 1)[:, :I].bool()
            v = valid & ~taken
            mask[:, ai, :I] = v
            mask[:, ai, I] = ~v.any(1)                                                # 'none' only when nothing is left for this arm
        return mask

    def _apply_pointer(self, ptr, pc):
        N, I = self.N, self.I
        mask = self._ptr_mask()
        invalid = torch.zeros(N, device=self.dev)
        for ai, a in enumerate(ARMS):
            want = ptr[:, ai].clamp(0, I)
            ok = mask[:, ai].gather(1, want[:, None])[:, 0]
            first = mask[:, ai].float().argmax(1)
            want = torch.where(ok, want, first)
            invalid += (~ok & self.needs_ptr[:, ai]).float()
            # the second arm sees the first arm's new choice
            change = self.needs_ptr[:, ai] & (want != self.assigned[:, ai])
            if bool(change.any()):
                old_k = self.item_obj[self.assigned[:, ai].clamp(max=I - 1)]
                new_k = self.item_obj[want.clamp(max=I - 1)]
                for kk in torch.unique(old_k[change & (self.assigned[:, ai] < I)]).tolist():
                    self._set_inertia(kk, torch.nonzero(change & (old_k == kk) & (self.assigned[:, ai] < I))[:, 0], False)
                for kk in torch.unique(new_k[change & (want < I)]).tolist():
                    self._set_inertia(kk, torch.nonzero(change & (new_k == kk) & (want < I))[:, 0], True)
                self.assigned[change, ai] = want[change]
                m = torch.zeros(N, dtype=torch.bool, device=self.dev); m[change] = True
                self.obsb[a].reset(m)
                self.at_goal_cnt[change, ai] = 0
            mask = self._ptr_mask()
        self.needs_ptr[:] = False
        return invalid

    # ------------------------------------------------------------------ observation
    def _goals(self, pc):
        I, ar = self.I, self.ar
        for ai, a in enumerate(ARMS):
            it = self.assigned[:, ai].clamp(max=I - 1)
            kc = self.item_dst[it]
            c = pc[ar, kc]
            g = c.clone(); g[:, 2] = c[:, 2] + self.ext[kc, 2] + self.release_h
            self.goal[a] = g

    def _obs(self, pc, qc):
        env, ar, I, N = self.env, self.ar, self.I, self.N
        self._goals(pc)
        lift, own_ee, oth_ee = [], [], []
        from cleantables.robots.lift_policy import quat_conj, quat_mul
        palm_pos = {a: env.robots[a].data.body_pos_w.torch[:, env.palm_bid] for a in ARMS}
        palm_q = {a: env.robots[a].data.body_quat_w.torch[:, env.palm_bid] for a in ARMS}
        for ai, a in enumerate(ARMS):
            r = env.robots[a]
            it = self.assigned[:, ai].clamp(max=I - 1)
            k = self.item_obj[it]
            gb = self._qai(r.data.root_quat_w.torch, self.goal[a] + self.origins - r.data.root_pos_w.torch)
            frm = self.obsb[a].frame(r.data.root_pos_w.torch, r.data.root_quat_w.torch, r.data.joint_pos.torch, r.data.joint_vel.torch,
                                     r.data.body_pos_w.torch[:, env.hand_bids], r.data.body_quat_w.torch[:, env.hand_bids],
                                     pc[ar, k] + self.origins, qc[ar, k], self.pts[k], env.tip_forces(a, torch.where(self.assigned[:, ai] < I, k, -1)), gb)
            lift.append(self.obsb[a].push(frm))
            o = "b" if a == "a" else "a"
            rq, rp = r.data.root_quat_w.torch, r.data.root_pos_w.torch
            own_ee.append(torch.cat([self._qai(rq, palm_pos[a] - rp), quat_mul(quat_conj(rq), palm_q[a])], -1))
            oth_ee.append(torch.cat([self._qai(rq, palm_pos[o] - rp), quat_mul(quat_conj(rq), palm_q[o])], -1))
        pi = pc[:, self.item_obj]                                                    # (N,I,3)
        pd = pc[:, self.item_dst]
        asg = torch.stack([torch.nn.functional.one_hot(self.assigned[:, ai], I + 1)[:, :I].float() for ai in range(2)], -1)   # (N,I,2)
        reach = torch.stack([(self._in_reach(a, pi[..., :2]) & self._in_reach(a, pd[..., :2])).float() for a in ARMS], -1)
        tok = torch.cat([pi, pd, self.item_cls[None].expand(N, -1, -1), self.placed[..., None].float(), asg, reach, (self.cool > 0)[..., None].float()], -1)
        return dict(lift=torch.stack(lift, 1), tokens=tok, other_ee=torch.stack(oth_ee, 1), own_ee=torch.stack(own_ee, 1), ptr_mask=self._ptr_mask(),
                    needs_pointer=self.needs_ptr.clone(), assigned=self.assigned.clone(), t=self.t.clone())

    def reset(self):
        pc, qc = self._ostate()
        self.last_obs = self._obs(pc, qc)
        return self.last_obs

    # ------------------------------------------------------------------ step
    @torch.inference_mode()
    def step(self, action, pointer=None):
        env, dev, N, I, ar = self.env, self.dev, self.N, self.I, self.ar
        pc, qc = self._ostate()
        invalid = torch.zeros(N, device=dev)
        if pointer is not None:
            invalid = self._apply_pointer(pointer.to(dev), pc)
        for ai, a in enumerate(ARMS):
            aa = action[:, ai * 9:(ai + 1) * 9]
            active = (self.assigned[:, ai] < I) & (self.open_cd[:, ai] == 0)
            env.action[a] = torch.where(active[:, None], aa, torch.zeros_like(aa))
            env.mode[a] = torch.where(self.open_cd[:, ai] > 0, OPEN, POLICY)
        env.step(self.zero_act)
        self.t += 1
        pc, qc = self._ostate()
        # ---- optional scripted release assist (for the frozen-prior rollouts; the RL policy has none)
        rel_now = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        if self.auto_release:
            for ai, a in enumerate(ARMS):
                k = self.item_obj[self.assigned[:, ai].clamp(max=I - 1)]
                near = ((pc[ar, k] - (self.goal[a])).norm(dim=-1) < 0.03) & (self.assigned[:, ai] < I)
                if getattr(self, "release_rule", "point") == "above":
                    # open once the held object is anywhere over its container's opening (2 cm inside the rim, up to 25 cm above it),
                    # not only within 3 cm of one release point
                    kc = self.item_dst[self.assigned[:, ai].clamp(max=I - 1)]
                    c = pc[ar, kc]; e = self.ext[kc]; p_ = pc[ar, k]
                    over = ((p_[:, 0] - c[:, 0]).abs() < e[:, 0] / 2 - 0.02) & ((p_[:, 1] - c[:, 1]).abs() < e[:, 1] / 2 - 0.02)
                    top = c[:, 2] + e[:, 2]                         # container root is at its base (as in _inside)
                    near = near | (over & (p_[:, 2] > top) & (p_[:, 2] < top + 0.25) & (self.assigned[:, ai] < I))
                self.at_goal_cnt[:, ai] = torch.where(near, self.at_goal_cnt[:, ai] + 1, torch.zeros_like(self.at_goal_cnt[:, ai]))
                rel = (self.at_goal_cnt[:, ai] >= 3) & (self.open_cd[:, ai] == 0)
                self.open_cd[rel, ai] = int(0.6 / self.dt)
                self.cool[rel, self.assigned[rel, ai]] = int(2.0 / self.dt)
                rel_now[:, ai] = rel
                self.at_goal_cnt[rel, ai] = 0
            # after the open phase the arm is free again
        for ai in range(2):
            fin = (self.open_cd[:, ai] == 1)
            if bool(fin.any()) and self.auto_release:
                old = self.item_obj[self.assigned[:, ai].clamp(max=I - 1)]
                for kk in torch.unique(old[fin]).tolist():
                    self._set_inertia(kk, torch.nonzero(fin & (old == kk))[:, 0], False)
                self.assigned[fin, ai] = I; self.needs_ptr[fin, ai] = True
            self.open_cd[:, ai] = (self.open_cd[:, ai] - 1).clamp(min=0)
        self.cool = (self.cool - 1).clamp(min=0)
        self.last_rel = rel_now                                                      # the assist's open decisions this step (teacher labels)
        # ---- judge
        pi = pc[:, self.item_obj]
        ins = torch.stack([self._inside(pi[:, i], self.item_dst[i].expand(N), pc) for i in range(I)], 1)
        spd = torch.stack([env.objs[k].data.root_lin_vel_w.torch.norm(dim=-1) for k in self.item_idx], 1)
        self.rest_cnt = torch.where(ins & (spd < 0.05), self.rest_cnt + 1, torch.zeros_like(self.rest_cnt))
        newly = (self.rest_cnt >= 15) & ~self.placed
        lost = self.placed & ~ins
        self.placed = (self.placed | newly) & ~lost
        # ---- penalties
        cf = env.arm_contact_force()
        arm_hit = (cf > 1.0).float()
        nt = torch.zeros(N, device=dev)
        for ai, a in enumerate(ARMS):
            k = self.item_obj[self.assigned[:, ai].clamp(max=I - 1)]
            Fm = torch.stack([s_.data.normal_force_matrix_w.torch.reshape(N, -1, 3).norm(dim=-1) for s_ in env.tip_sensors[a]], 0).sum(0)
            tgt = torch.nn.functional.one_hot(k, self.K).bool() & (self.assigned[:, ai] < I)[:, None]
            nt = torch.maximum(nt, ((Fm * (self.mov_mask[None] & ~tgt)).amax(1) > 1.0).float())
        pa, pb = [env.robots[a].data.body_pos_w.torch[:, env.hand_bids] for a in ARMS]
        self.min_d = torch.minimum(self.min_d, torch.cdist(pa, pb).flatten(1).amin(1))
        newk = torch.zeros(N, device=dev)
        for k in self.obst_idx:
            kn = ((pc[:, k] - self.p_start[:, k]).norm(dim=-1) > 0.03) & ~self.knocked[:, k]
            self.knocked[:, k] |= kn; newk += kn.float()
        rew = self.w_placed * (newly.float().sum(1) - lost.float().sum(1)) - self.w["time"] - self.w["arm"] * arm_hit - self.w["nt"] * nt - self.w["knock"] * newk
        self.ret += rew
        self.arm_c += arm_hit; self.nt_c += nt
        self.idle += (self.assigned == I).float()
        alld = self.placed.all(1)
        self.makespan = torch.where(alld & (self.makespan < 0), self.t.float() * self.dt, self.makespan)
        done = alld | (self.t >= self.max_steps)
        info = dict(invalid_pointer=invalid)
        if bool(done.any()):
            d = torch.nonzero(done)[:, 0]
            info["final"] = dict(env=d.cpu().numpy(), placed=self.placed[d].sum(1).cpu().numpy(), placed_mask=self.placed[d].cpu().numpy(), I=I, success=alld[d].cpu().numpy(),
                                 makespan_s=torch.where(self.makespan[d] >= 0, self.makespan[d], self.t[d].float() * self.dt).cpu().numpy(),
                                 ret=self.ret[d].cpu().numpy(), arm_contact_steps=self.arm_c[d].cpu().numpy(), min_arm_dist=self.min_d[d].cpu().numpy(),
                                 nontarget_steps=self.nt_c[d].cpu().numpy(), knocked=self.knocked[d].sum(1).cpu().numpy(), idle_s=(self.idle[d] * self.dt).cpu().numpy())
            if self.autoreset:
                self._reset_envs(d)
                pc, qc = self._ostate()
        self.needs_ptr |= ((self.t % self.ptr_every) == 0)[:, None]
        if self.autoreset:
            self.needs_ptr[done] = True
        self.last_obs = self._obs(pc, qc)
        return self.last_obs, rew, done, info


# ---------------------------------------------------------------------------------------------- frozen prior pair + greedy pointer
class FrozenPriorPair:
    """Two weight-shared copies of the published NVIDIA Franka actor (unchanged weights): action (N,18) = [prior(lift[:,0]) | prior(lift[:,1])]."""

    def __init__(self, env, ckpt=CKPT):
        from cleantables.robots.lift_policy import LiftActor
        self.actor = LiftActor(os.path.join(ROOT, ckpt), env.dev)

    @torch.inference_mode()
    def act(self, obs):
        x = obs["lift"]
        N = x.shape[0]
        out = self.actor(x.reshape(N * 2, -1)).reshape(N, 2, 9)
        return out.reshape(N, 18)


@torch.inference_mode()
def greedy_pointer(env, obs):
    """Each arm re-picks the valid item whose object is nearest to its own palm and reachable by it (greedy nearest object); no collision handling."""
    N, I = env.N, env.I
    pc, _ = env._ostate()
    pi = pc[:, env.item_obj]
    ptr = obs["assigned"].clone()
    for ai, a in enumerate(ARMS):
        ee = env.env.robots[a].data.body_pos_w.torch[:, env.env.palm_bid] - env.origins
        d = (pi[..., :2] - ee[:, None, :2]).norm(dim=-1)
        ok = obs["ptr_mask"][:, ai, :I] & obs["tokens"][:, :, 12 + ai].bool()
        d = torch.where(ok, d, torch.full_like(d, 9.0))
        best = d.argmin(1)
        pick = torch.where(ok.any(1), best, torch.full_like(best, I))
        ptr[:, ai] = torch.where(obs["needs_pointer"][:, ai], pick, ptr[:, ai])
    return ptr


if __name__ == "__main__":
    import argparse
    import time

    from isaaclab.app import add_launcher_args, launch_simulation
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cl_coord_0")
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--policy", default="prior_greedy", choices=["prior_greedy", "prior_greedy_norelease", "zero"])
    ap.add_argument("--steps", type=int, default=1350)
    ap.add_argument("--max_seconds", type=float, default=45.0)
    ap.add_argument("--overlap", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/twin_cleanup/joint_baseline")
    add_launcher_args(ap)
    args = ap.parse_args()
    cfg, S, POSES = JointCleanupEnv.make_cfg(args.scene, args.num_envs, args.seed)
    with launch_simulation(cfg, args):
        env = JointCleanupEnv(cfg, S, POSES, overlap=args.overlap, max_seconds=args.max_seconds, auto_release=(args.policy == "prior_greedy"), seed=args.seed, autoreset=False)
        pri = FrozenPriorPair(env)
        obs = env.reset()
        os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
        fin, ret_sum = {}, 0.0
        t0 = time.time(); steps = 0
        for s in range(args.steps):
            act = pri.act(obs) if args.policy != "zero" else torch.zeros(env.N, 18, device=env.dev)
            obs, rew, done, info = env.step(act, greedy_pointer(env, obs) if args.policy != "zero" else None)
            steps += 1
            if "final" in info:
                for j, e in enumerate(info["final"]["env"]):
                    if int(e) not in fin:
                        fin[int(e)] = {k: (v[j].tolist() if hasattr(v[j], "tolist") else v[j]) for k, v in info["final"].items() if k not in ("env", "I")}
            if s % 150 == 0:
                print(f"step {s} placed {int(env.placed.sum())}/{env.N * env.I} finished {len(fin)}/{env.N}  {time.time() - t0:.0f}s", flush=True)
            if len(fin) == env.N:
                break
        wall = time.time() - t0
        P = np.array([f["placed"] for f in fin.values()]) if fin else np.zeros(0)
        res = dict(args=vars(args), N=env.N, I=env.I, steps=steps, wall_s=wall, env_steps_per_s=env.N * steps / wall, episodes_finished=len(fin),
                   success_frac=float(np.mean([f["success"] for f in fin.values()])) if fin else 0.0,
                   placed_mean=float(P.mean()) if len(P) else 0.0, still_running_placed=float(env.placed.sum(1).float().mean()),
                   makespan_s_mean=float(np.mean([f["makespan_s"] for f in fin.values()])) if fin else None,
                   arm_contact_steps_mean=float(np.mean([f["arm_contact_steps"] for f in fin.values()])) if fin else None,
                   final_state={"placed_per_env": env.placed.sum(1).cpu().tolist(), "min_arm_dist": env.min_d.cpu().tolist(), "arm_contact_steps": env.arm_c.cpu().tolist(),
                                "nontarget_steps": env.nt_c.cpu().tolist(), "knocked": env.knocked.sum(1).cpu().tolist(), "ret": env.ret.cpu().tolist(),
                                "makespan_s": env.makespan.cpu().tolist()})
        json.dump(res, open(os.path.join(ROOT, args.out, "result.json"), "w"), indent=1)
        print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("final_state", "args")}))
        env.env.close()
