"""Two-arm table-cleanup task on one physics twin, for one centralised policy that controls both Kuka-Allegro arms.

Action (N, 46): the lift-policy joint action of arm a then arm b (joint target = q + 0.1 * action); an arm without an
object holds its pose. Pointer (N, 2): per arm the in-scope item it works on, or I = none; it is re-chosen every
`pointer_every` control steps and after a reset. The world-model teacher sets every item's goal pose at reset and the
carry path the arm's lift policy is steered along. Reward per control step: +1 per item newly put down at its goal (at
rest for 0.5 s, released, within 5 cm in xy and not above it), -1 if a put-down item leaves again, and small costs for
time, arm-arm contact and proximity, fingertips on other movable objects and knocked obstacles.
Import after the Isaac app is launched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import warp as wp

from .config import EnvConfig
from .geometry import lookahead_point, points_ahead, quat_apply_inv, quat_conj, quat_mul, yaw_of
from .layouts import REACH_X, REACH_Y, SPARE_PARK_X, SPARE_PARK_Z, sample_layouts
from .lift_prior import LiftObservation, sample_point_cloud
from .scene import ARMS, TwinSim
from .teacher_goals import TeacherGoals

SETTLE_STEPS = 45
INERTIA_PAD = 0.01  # kg m^2 added to the held object's inertia, as in the lift task's training objects
REST_SPEED = 0.05  # m/s
REST_STEPS = 15  # 0.5 s at rest before an item counts as put down
CONTACT_FORCE = 1.0  # N
KNOCK_DIST = 0.03  # m an obstacle may move before it counts as knocked
LIFTED_HEIGHT = 0.05  # m above its start, for the "lifted" statistic
SPAWN_CHECK_STEP = 15  # an env is checked 0.5 s into its episode
SPAWN_MOVE = 0.01  # m any object may move during that time
SPAWN_RETRIES = 3  # after this many re-drawn layouts in a row the layout is accepted
NAN_GUARD_TRIES = 3
FAR_AWAY = 20.0  # m from the env origin counts as exploded
OBSTACLE_TOKEN = 11
NEAR_OBSTACLES = 3
ARM_LINKS = ["iiwa7_link_3", "iiwa7_link_4", "iiwa7_link_5", "iiwa7_link_6", "iiwa7_link_7", "palm_link", ".*_tip"]
LINK_RADIUS = 0.06  # m, spheres around the arm link points for the arm-arm distance
PROXIMITY_RANGE = 0.05  # m arm-arm surface distance below which the proximity cost starts
JOINT_VEL_SCALE = 0.1


@dataclass
class TokenLayout:
    """Columns of the per-item token.

    pos 0-2 | destination 3-5 | class one-hot | put down | taken by a / b | reachable by a / b |
    cooldown (always 0 in this task, kept for checkpoint compatibility) | teacher P(next)
    """

    vocab: int
    idx: dict = field(init=False)

    def __post_init__(self):
        b = 6 + self.vocab
        self.idx = dict(cls=(6, b), placed=b, ta=b + 1, tb=b + 2, ra=b + 3, rb=b + 4, cool=b + 5, wm=b + 6, dim=b + 7)

    def __getitem__(self, key):
        return self.idx[key]


class CleanupEnv:
    def __init__(self, scene: dict, cfg: EnvConfig, sim: TwinSim, seed: int = 0, autoreset: bool = True):
        self.scene_json, self.cfg, self.sim, self.autoreset = scene, cfg, sim, autoreset
        self.device, self.num_envs, self.num_objects = sim.device, sim.num_envs, sim.num_objects
        dev, n, k_all = self.device, self.num_envs, self.num_objects
        self.dt = sim.step_dt
        self.origins = sim.scene.env_origins
        objs = scene["objects"]
        self.names = [o["id"] for o in objs]
        self.role = [o["role"] for o in objs]
        self.item_idx = [k for k in range(k_all) if self.role[k] == "task"]
        self.num_items = len(self.item_idx)
        self.item_names = [self.names[k] for k in self.item_idx]
        self.item_obj = torch.tensor(self.item_idx, device=dev)
        self.item_dst = torch.tensor([self.names.index(objs[k]["dest"]) for k in self.item_idx], device=dev)
        self.item_cls_names = [objs[k]["cls"] for k in self.item_idx]
        missing = sorted(set(self.item_cls_names) - set(cfg.class_vocab))
        assert not missing, f"scene classes {missing} not in the class vocabulary {cfg.class_vocab}"
        self.item_cls = torch.nn.functional.one_hot(
            torch.tensor([cfg.class_vocab.index(c) for c in self.item_cls_names], device=dev), len(cfg.class_vocab)
        ).float()
        self.tokens = TokenLayout(len(cfg.class_vocab))
        self.movable = torch.tensor([r in ("task", "obstacle", "spare") for r in self.role], device=dev)
        self.knockable = [k for k in range(k_all) if self.role[k] in ("obstacle", "spare")]
        self.extents = torch.tensor([o["extents"] for o in objs], device=dev)
        self.mass0 = torch.tensor([o["mass"] for o in objs], device=dev)
        self.points = torch.stack([torch.from_numpy(sample_point_cloud(o["mesh"])) for o in objs]).to(dev)
        self.inertia0 = [wp.to_torch(o.root_view.get_inertias()).clone().reshape(n, -1, 9) for o in sim.objs]
        self.arm_yaw = math.radians(float(scene.get("arms", {}).get("yaw_deg", 90.0)))
        self.max_steps = int(cfg.max_seconds / self.dt)
        self.rng = np.random.default_rng(seed)
        self.arange = torch.arange(n, device=dev)
        self.lift_obs = {a: LiftObservation(n, dev) for a in ARMS}
        self.base = {a: torch.tensor(scene["poses"][a][0][:2], device=dev) for a in ARMS}
        self.link_bids = sim.robots["a"].find_bodies(ARM_LINKS)[0]
        self._init_obstacle_tokens()
        for a in ARMS:
            sim.set_friction(sim.robots[a], *cfg.robot_friction)
        self.zero_action = torch.zeros(n, sim.cfg.action_space, device=dev)
        self.teacher: TeacherGoals | None = None
        self.layout_override = None
        self.spawn_stats = dict(checked=0, rejected=0, accepted_after_retries=0, moved_by_object={})
        self.nan_resets = 0
        self._init_state()
        sim.reset()
        self.reset_envs(self.arange, settle=True)

    # ------------------------------------------------------------------ state
    def _init_state(self):
        dev, n, num_items, k_all = self.device, self.num_envs, self.num_items, self.num_objects
        self.assigned = torch.full((n, 2), num_items, dtype=torch.long, device=dev)
        self.placed = torch.zeros(n, num_items, dtype=torch.bool, device=dev)
        self.rest_count = torch.zeros(n, num_items, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.needs_pointer = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.ret = torch.zeros(n, device=dev)
        self.arm_contact_steps = torch.zeros(n, device=dev)
        self.nontarget_steps = torch.zeros(n, device=dev)
        self.min_hand_dist = torch.full((n,), 9.0, device=dev)
        self.knocked = torch.zeros(n, k_all, dtype=torch.bool, device=dev)
        self.idle = torch.zeros(n, 2, device=dev)
        self.makespan = torch.full((n,), -1.0, device=dev)
        self.p_start = torch.zeros(n, k_all, 3, device=dev)
        self.goal = {a: torch.zeros(n, 3, device=dev) for a in ARMS}
        self.lifted = torch.zeros(n, num_items, dtype=torch.bool, device=dev)
        # the same two flags for each env's last finished episode
        self.ep_lifted = torch.zeros(n, num_items, dtype=torch.bool, device=dev)
        self.ep_placed = torch.zeros(n, num_items, dtype=torch.bool, device=dev)
        self.skip = torch.zeros(n, dtype=torch.bool, device=dev)  # episode not logged (staggered start)
        self.spawn_pending = torch.ones(n, dtype=torch.bool, device=dev)
        self.spawn_p1 = torch.zeros(n, k_all, 3, device=dev)
        self.spawn_retries = torch.zeros(n, dtype=torch.long, device=dev)
        self.exploded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_obs = None

    def object_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Object root positions (env frame) and orientations, (N, K, 3) and (N, K, 4)."""
        p = torch.stack([o.data.root_pos_w.torch for o in self.sim.objs], 1) - self.origins[:, None]
        q = torch.stack([o.data.root_quat_w.torch for o in self.sim.objs], 1)
        return p, q

    def palm_pos(self, arm: str) -> torch.Tensor:
        return self.sim.robots[arm].data.body_pos_w.torch[:, self.sim.palm_bid] - self.origins

    def _set_inertia(self, k: int, env_ids: torch.Tensor, held: bool):
        inertia = self.inertia0[k].clone()
        if held:
            inertia[..., 0] += INERTIA_PAD
            inertia[..., 4] += INERTIA_PAD
            inertia[..., 8] += INERTIA_PAD
        self.sim.objs[k].set_inertias_index(inertias=inertia[env_ids.cpu()].to(self.device), env_ids=env_ids)

    def in_reach(self, arm: str, xy: torch.Tensor) -> torch.Tensor:
        d = xy - self.base[arm]
        c, s = math.cos(self.arm_yaw), math.sin(self.arm_yaw)
        xb, yb = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
        return (xb >= REACH_X[0]) & (xb <= REACH_X[1]) & (yb.abs() <= REACH_Y)

    # ------------------------------------------------------------------ resets
    @torch.inference_mode()
    def reset_envs(self, ids: torch.Tensor, settle: bool = False) -> None:
        """New random layout and domain randomisation for envs `ids`; `settle` lets the objects come to rest first."""
        sim, dev, num_items, n = self.sim, self.device, self.num_items, len(ids)
        self.ep_lifted[ids] = self.lifted[ids]
        self.ep_placed[ids] = self.placed[ids]
        for ai in range(2):  # the held object gets its own inertia back
            obj = self.item_obj[self.assigned[ids, ai].clamp(max=num_items - 1)]
            has = self.assigned[ids, ai] < num_items
            for kk in torch.unique(obj[has]).tolist():
                self._set_inertia(kk, ids[has & (obj == kk)], False)
        if self.layout_override is not None:
            xyz, yaw = self.layout_override(ids)
            self.layout_info = {}
        else:
            xyz, yaw, self.layout_info = sample_layouts(self.scene_json, n, self.rng, self.cfg.shuffle_prob)
        self.last_layouts = (np.asarray(xyz).copy(), np.asarray(yaw).copy())
        for k in range(self.num_objects):
            pose = torch.zeros(n, 7, device=dev)
            if self.role[k] == "spare":
                pose[:, 0] = self.scene_json["table"]["x"][1] + SPARE_PARK_X
                pose[:, 2] = SPARE_PARK_Z
                pose[:, 6] = 1.0
            else:
                pose[:, :3] = torch.tensor(xyz[:, k], dtype=torch.float32, device=dev)
                yw = torch.tensor(yaw[:, k], dtype=torch.float32, device=dev)
                pose[:, 5] = torch.sin(yw / 2)
                pose[:, 6] = torch.cos(yw / 2)
            pose[:, :3] += self.origins[ids]
            sim.objs[k].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            sim.objs[k].write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
            if self.role[k] in ("task", "obstacle") and self.cfg.mass_dr > 0:
                scale = 1 + (torch.rand(n, 1, device=dev) * 2 - 1) * self.cfg.mass_dr
                sim.objs[k].set_masses_index(masses=self.mass0[k] * scale, env_ids=ids)
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        mask[ids] = True
        for a in ARMS:
            r = sim.robots[a]
            q = r.data.default_joint_pos.torch.clone()[ids]
            r.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)
            self.lift_obs[a].reset(mask)
        self.assigned[ids] = num_items
        self.placed[ids] = False
        self.rest_count[ids] = 0
        self.t[ids] = 0
        self.needs_pointer[ids] = True
        self.ret[ids] = 0
        self.arm_contact_steps[ids] = 0
        self.nontarget_steps[ids] = 0
        self.min_hand_dist[ids] = 9.0
        self.knocked[ids] = False
        self.idle[ids] = 0
        self.makespan[ids] = -1.0
        if settle:
            for _ in range(SETTLE_STEPS):
                sim.step(self.zero_action)
        self.p_start[ids] = self.object_state()[0][ids]
        self.skip[ids] = False
        self.spawn_pending[ids] = True
        self._randomise_boxes_and_friction(ids)
        self.lifted[ids] = False
        if self.teacher is None:
            c = self.cfg
            self.teacher = TeacherGoals(
                self, c.teacher_run, c.teacher_seeds, c.teacher_samples, c.replan_every, c.offpath_dist, c.offpath_gap
            )
        self.teacher.set_goals(ids)

    def _randomise_boxes_and_friction(self, ids):
        """Heavy, grippy boxes (a touch does not launch them) and item friction scaled around its measured value."""
        sim, c = self.sim, self.cfg
        for k, o in enumerate(self.scene_json["objects"]):
            if o["role"] in ("obstacle", "stack"):
                m = torch.empty(len(ids), 1, device=self.device).uniform_(*c.heavy_box_mass)
                sim.objs[k].set_masses_index(masses=m, env_ids=ids)
                inertia = self.inertia0[k][ids.cpu()].clone().to(self.device) * (m / float(self.mass0[k])).reshape(
                    -1, 1, 1
                )
                sim.objs[k].set_inertias_index(inertias=inertia, env_ids=ids)
                sim.set_friction(sim.objs[k], *c.heavy_box_friction, env_ids=ids)
        for k, o in enumerate(self.scene_json["objects"]):
            if o["role"] == "task":
                mu = float(np.mean(o.get("mu", [0.6, 0.5])))
                sim.set_friction(sim.objs[k], mu * c.object_friction_dr[0], mu * c.object_friction_dr[1], env_ids=ids)

    @torch.inference_mode()
    def reset(self) -> dict:
        """Observation of the current state (after a guard against exploded envs)."""
        if self._nan_guard():
            print(f"nan guard at reset: {self.nan_resets} env resets so far", flush=True)
        pc, qc = self.object_state()
        self.last_obs = self._obs(pc, qc)
        return self.last_obs

    # ------------------------------------------------------------------ pointer
    def pointer_mask(self) -> torch.Tensor:
        """(N, 2, I+1) valid choices: items not put down and not taken by the other arm; 'none' only if none is left."""
        n, num_items = self.num_envs, self.num_items
        valid = ~self.placed
        mask = torch.zeros(n, 2, num_items + 1, dtype=torch.bool, device=self.device)
        for ai in range(2):
            taken = torch.nn.functional.one_hot(self.assigned[:, 1 - ai], num_items + 1)[:, :num_items].bool()
            v = valid & ~taken
            mask[:, ai, :num_items] = v
            mask[:, ai, num_items] = ~v.any(1)
        return mask

    def _apply_pointer(self, pointer: torch.Tensor) -> torch.Tensor:
        """Apply arm a's choice, then arm b's (b sees a's new choice). An invalid choice becomes the first valid one."""
        n, num_items = self.num_envs, self.num_items
        mask = self.pointer_mask()
        invalid = torch.zeros(n, device=self.device)
        for ai, a in enumerate(ARMS):
            want = pointer[:, ai].clamp(0, num_items)
            ok = mask[:, ai].gather(1, want[:, None])[:, 0]
            first = mask[:, ai].float().argmax(1)
            want = torch.where(ok, want, first)
            invalid += (~ok & self.needs_pointer[:, ai]).float()
            change = self.needs_pointer[:, ai] & (want != self.assigned[:, ai])
            if bool(change.any()):
                old_k = self.item_obj[self.assigned[:, ai].clamp(max=num_items - 1)]
                new_k = self.item_obj[want.clamp(max=num_items - 1)]
                had = change & (self.assigned[:, ai] < num_items)
                for kk in torch.unique(old_k[had]).tolist():
                    self._set_inertia(kk, torch.nonzero(had & (old_k == kk))[:, 0], False)
                gets = change & (want < num_items)
                for kk in torch.unique(new_k[gets]).tolist():
                    self._set_inertia(kk, torch.nonzero(gets & (new_k == kk))[:, 0], True)
                self.assigned[change, ai] = want[change]
                m = torch.zeros(n, dtype=torch.bool, device=self.device)
                m[change] = True
                self.lift_obs[a].reset(m)
            mask = self.pointer_mask()
        self.needs_pointer[:] = False
        return invalid

    # ------------------------------------------------------------------ observation
    def _init_obstacle_tokens(self):
        objs = self.scene_json["objects"]
        self.obstacle_idx = torch.tensor(
            [k for k, o in enumerate(objs) if o["role"] in ("container", "obstacle", "stack")],
            device=self.device,
            dtype=torch.long,
        )
        flags = [
            [
                float(objs[k]["role"] == "container"),
                float(objs[k]["role"] in ("obstacle", "stack")),
                float(objs[k]["category"].startswith("box")),
            ]
            for k in self.obstacle_idx.tolist()
        ]
        self.obstacle_flags = torch.tensor(flags, device=self.device).reshape(-1, 3)

    def _obstacle_obs(self, pc, qc) -> dict:
        """Box and container observations.

        obst (N, Ko, 11), table frame:
            bottom centre 3 | half extents 3 | sin, cos yaw | is container | is obstacle | is box
        obst_near (N, 2, 24), per arm the 3 boxes nearest to its palm in its base frame:
            centre offset from the palm 3 | half extents 3 | sin, cos relative yaw
        """
        k = self.obstacle_idx
        n, ko = self.num_envs, len(k)
        if ko == 0:
            return dict(
                obst=torch.zeros(n, 0, OBSTACLE_TOKEN, device=self.device),
                obst_near=torch.zeros(n, 2, 8 * NEAR_OBSTACLES, device=self.device),
            )
        p, yaw = pc[:, k], yaw_of(qc[:, k])
        half = (self.extents[k] / 2)[None].expand(n, -1, -1)
        obst = torch.cat(
            [
                p,
                half,
                torch.sin(yaw)[..., None],
                torch.cos(yaw)[..., None],
                self.obstacle_flags[None].expand(n, -1, -1),
            ],
            -1,
        )
        near = []
        for a in ARMS:
            r = self.sim.robots[a]
            rq = r.data.root_quat_w.torch
            palm = self.palm_pos(a)
            centre = p.clone()
            centre[..., 2] = centre[..., 2] + half[..., 2]
            kk = min(NEAR_OBSTACLES, ko)
            j = (centre - palm[:, None]).norm(dim=-1).topk(kk, largest=False).indices
            cj = centre.gather(1, j[..., None].expand(-1, -1, 3))
            hj = half.gather(1, j[..., None].expand(-1, -1, 3))
            rel = quat_apply_inv(rq[:, None].expand(-1, kk, -1), cj - palm[:, None])
            yj = yaw.gather(1, j) - yaw_of(rq)[:, None]
            f = torch.cat([rel, hj, torch.sin(yj)[..., None], torch.cos(yj)[..., None]], -1).reshape(n, -1)
            if kk < NEAR_OBSTACLES:
                f = torch.cat([f, torch.zeros(n, 8 * (NEAR_OBSTACLES - kk), device=self.device)], -1)
            near.append(f)
        return dict(obst=obst, obst_near=torch.stack(near, 1))

    def arm_points(self) -> torch.Tensor:
        """(N, 2, P, 3) link, palm and fingertip positions of both arms (env frame)."""
        return torch.stack(
            [self.sim.robots[a].data.body_pos_w.torch[:, self.link_bids] - self.origins[:, None] for a in ARMS], 1
        )

    def arm_surface_dist(self, pts: torch.Tensor | None = None) -> torch.Tensor:
        pts = self.arm_points() if pts is None else pts
        return torch.cdist(pts[:, 0], pts[:, 1]).flatten(1).amin(1) - 2 * LINK_RADIUS

    def _other_arm_obs(self) -> torch.Tensor:
        """Per arm (N, 2, 77): the other arm's link points in this arm's base frame, its joint positions and velocities
        (x0.1), and the arm-arm surface distance (each arm sees the other's full state, as in Bi-DexHands)."""
        pts, n = self.arm_points(), self.num_envs
        dist = self.arm_surface_dist(pts)[:, None]
        out = []
        for ai, a in enumerate(ARMS):
            other = self.sim.robots[ARMS[1 - ai]]
            r = self.sim.robots[a]
            rq, rp = r.data.root_quat_w.torch, r.data.root_pos_w.torch - self.origins
            rel = quat_apply_inv(rq[:, None].expand(-1, pts.shape[2], -1), pts[:, 1 - ai] - rp[:, None]).reshape(n, -1)
            out.append(
                torch.cat([rel, other.data.joint_pos.torch, JOINT_VEL_SCALE * other.data.joint_vel.torch, dist], -1)
            )
        return torch.stack(out, 1)

    def _path_obs(self, pc) -> torch.Tensor:
        """Per arm (N, 2, 15): the teacher's path 5-30 cm ahead of the object, relative to it, in the arm's frame."""
        n, num_items, tg = self.num_envs, self.num_items, self.teacher
        out = torch.zeros(n, 2, 3 * len(self.cfg.path_points), device=self.device)
        for ai, a in enumerate(ARMS):
            asg = self.assigned[:, ai]
            p = pc[self.arange, self.item_obj[asg.clamp(max=num_items - 1)]]
            pts = points_ahead(tg.path[:, ai], p, self.cfg.path_points)
            rq = self.sim.robots[a].data.root_quat_w.torch
            rel = quat_apply_inv(rq[:, None].expand(-1, pts.shape[1], -1), pts - p[:, None]).reshape(n, -1)
            out[:, ai] = rel * (tg.ok[:, ai] & (asg < num_items)).float()[:, None]
        return out

    def _update_goals(self, pc, qc):
        """Goal for each arm's lift policy: a point ahead on the teacher's path, or the item's goal without a path."""
        num_items = self.num_items
        tg = self.teacher
        tg.update_paths(pc, qc)
        for ai, a in enumerate(ARMS):
            asg = self.assigned[:, ai]
            item = asg.clamp(max=num_items - 1)
            p = pc[self.arange, self.item_obj[item]]
            final = tg.goal_centre[self.arange, item]
            ahead = lookahead_point(tg.path[:, ai], p, self.cfg.lookahead)
            self.goal[a] = torch.where(tg.ok[:, ai, None], ahead, final)

    def _obs(self, pc, qc) -> dict:
        sim, ar, num_items, n = self.sim, self.arange, self.num_items, self.num_envs
        with torch.inference_mode():
            self._update_goals(pc, qc)
        lift, own_ee, other_ee = [], [], []
        palm_p = {a: sim.robots[a].data.body_pos_w.torch[:, sim.palm_bid] for a in ARMS}
        palm_q = {a: sim.robots[a].data.body_quat_w.torch[:, sim.palm_bid] for a in ARMS}
        for ai, a in enumerate(ARMS):
            r = sim.robots[a]
            rp, rq = r.data.root_pos_w.torch, r.data.root_quat_w.torch
            k = self.item_obj[self.assigned[:, ai].clamp(max=num_items - 1)]
            goal_b = quat_apply_inv(rq, self.goal[a] + self.origins - rp)
            frame = self.lift_obs[a].frame(
                rp,
                rq,
                r.data.joint_pos.torch,
                r.data.joint_vel.torch,
                r.data.body_pos_w.torch[:, sim.hand_bids],
                r.data.body_quat_w.torch[:, sim.hand_bids],
                pc[ar, k] + self.origins,
                qc[ar, k],
                self.points[k],
                sim.tip_forces(a, torch.where(self.assigned[:, ai] < num_items, k, -1)),
                goal_b,
            )
            lift.append(self.lift_obs[a].push(frame))
            o = ARMS[1 - ai]
            own_ee.append(torch.cat([quat_apply_inv(rq, palm_p[a] - rp), quat_mul(quat_conj(rq), palm_q[a])], -1))
            other_ee.append(torch.cat([quat_apply_inv(rq, palm_p[o] - rp), quat_mul(quat_conj(rq), palm_q[o])], -1))
        pi, pd = pc[:, self.item_obj], pc[:, self.item_dst]
        taken = torch.stack(
            [
                torch.nn.functional.one_hot(self.assigned[:, ai], num_items + 1)[:, :num_items].float()
                for ai in range(2)
            ],
            -1,
        )
        reach = torch.stack([(self.in_reach(a, pi[..., :2]) & self.in_reach(a, pd[..., :2])).float() for a in ARMS], -1)
        cooldown = torch.zeros(n, num_items, 1, device=self.device)
        tokens = torch.cat(
            [
                pi,
                pd,
                self.item_cls[None].expand(n, -1, -1),
                self.placed[..., None].float(),
                taken,
                reach,
                cooldown,
                self.teacher.p_next[..., None],
            ],
            -1,
        )
        obs = dict(
            lift=torch.stack(lift, 1),
            tokens=tokens,
            other_ee=torch.stack(other_ee, 1),
            own_ee=torch.stack(own_ee, 1),
            ptr_mask=self.pointer_mask(),
            needs_pointer=self.needs_pointer.clone(),
            assigned=self.assigned.clone(),
            t=self.t.clone(),
        )
        obs.update(self._obstacle_obs(pc, qc))
        obs["obst_near"] = torch.cat([obs["obst_near"], self._other_arm_obs(), self._path_obs(pc)], -1)
        return obs

    # ------------------------------------------------------------------ step
    def _put_down(self, pi: torch.Tensor) -> torch.Tensor:
        """(N, I) item released, within goal_xy of its goal and not above it (gravity decides where it settles)."""
        held = (self.sim.tip_force_norms("a") + self.sim.tip_force_norms("b")) > CONTACT_FORCE
        goal = self.teacher.goal_root
        near = (pi[..., :2] - goal[..., :2]).norm(dim=-1) < self.cfg.goal_xy
        low = pi[..., 2] < goal[..., 2] + self.cfg.goal_above
        at_goal = near & low
        if self.cfg.success_in_destination:  # inside the footprint of the teacher's destination, below its top + 5 cm
            dest = self.teacher.goal_dest
            k = dest.clamp(min=0)
            pc, _ = self.object_state()
            c = torch.gather(pc, 1, k[..., None].expand(-1, -1, 3))
            e = self.extents[k]
            inside = (
                ((pi[..., 0] - c[..., 0]).abs() < e[..., 0] / 2 - 0.01)
                & ((pi[..., 1] - c[..., 1]).abs() < e[..., 1] / 2 - 0.01)
                & (pi[..., 2] < c[..., 2] + e[..., 2] + 0.05)
                & (pi[..., 2] > -0.02)
            )
            at_goal = torch.where(dest >= 0, inside, at_goal)
        return at_goal & ~held[:, self.item_obj]

    @torch.inference_mode()
    def step(self, action: torch.Tensor, pointer: torch.Tensor | None = None):
        sim, dev, n, num_items, c = self.sim, self.device, self.num_envs, self.num_items, self.cfg
        invalid = torch.zeros(n, device=dev)
        if pointer is not None:
            invalid = self._apply_pointer(pointer.to(dev))
        for ai, a in enumerate(ARMS):
            act = action[:, ai * 23 : (ai + 1) * 23]
            active = self.assigned[:, ai] < num_items
            sim.action[a] = torch.where(active[:, None], act, torch.zeros_like(act))
        sim.step(self.zero_action)
        self.t += 1
        pc, qc = self.object_state()

        pi = pc[:, self.item_obj]
        inside = self._put_down(pi)
        speed = torch.stack([sim.objs[k].data.root_lin_vel_w.torch.norm(dim=-1) for k in self.item_idx], 1)
        self.rest_count = torch.where(
            inside & (speed < REST_SPEED), self.rest_count + 1, torch.zeros_like(self.rest_count)
        )
        newly = (self.rest_count >= REST_STEPS) & ~self.placed
        lost = self.placed & ~inside
        self.placed = (self.placed | newly) & ~lost

        arm_hit = (sim.arm_contact_force() > CONTACT_FORCE).float()
        nontarget = torch.zeros(n, device=dev)
        for ai, a in enumerate(ARMS):
            k = self.item_obj[self.assigned[:, ai].clamp(max=num_items - 1)]
            target = (
                torch.nn.functional.one_hot(k, self.num_objects).bool() & (self.assigned[:, ai] < num_items)[:, None]
            )
            touch = (sim.tip_force_norms(a) * (self.movable[None] & ~target)).amax(1) > CONTACT_FORCE
            nontarget = torch.maximum(nontarget, touch.float())
        hand_a, hand_b = [sim.robots[a].data.body_pos_w.torch[:, sim.hand_bids] for a in ARMS]
        self.min_hand_dist = torch.minimum(self.min_hand_dist, torch.cdist(hand_a, hand_b).flatten(1).amin(1))
        knocked_now = torch.zeros(n, device=dev)
        for k in self.knockable:
            kn = ((pc[:, k] - self.p_start[:, k]).norm(dim=-1) > KNOCK_DIST) & ~self.knocked[:, k]
            self.knocked[:, k] |= kn
            knocked_now += kn.float()
        rew = (
            newly.float().sum(1)
            - lost.float().sum(1)
            - c.w_time
            - c.w_arm_contact * arm_hit
            - c.w_nontarget * nontarget
            - c.w_knock * knocked_now
        )
        self.ret += rew
        self.arm_contact_steps += arm_hit
        self.nontarget_steps += nontarget
        self.idle += (self.assigned == num_items).float()
        all_done = self.placed.all(1)
        self.makespan = torch.where(all_done & (self.makespan < 0), self.t.float() * self.dt, self.makespan)
        done = all_done | (self.t >= self.max_steps)
        info = dict(invalid_pointer=invalid)
        if bool(done.any()):
            d = torch.nonzero(done)[:, 0]
            info["final"] = self._episode_stats(d, all_done)
            if self.autoreset:
                self.reset_envs(d)
                pc, qc = self.object_state()
        self.needs_pointer |= ((self.t % c.pointer_every) == 0)[:, None]
        if self.autoreset:
            self.needs_pointer[done] = True
        obs = self.last_obs = self._obs(pc, qc)

        # dense cost before contact: contact-only penalties are learned slowly
        prox = torch.relu(PROXIMITY_RANGE - self.arm_surface_dist()).clamp(max=PROXIMITY_RANGE) / PROXIMITY_RANGE
        rew = rew - c.w_proximity * prox
        if c.spawn_check:
            obs, rew, done, info = self._spawn_check(obs, rew, done, info)
        pi = self.object_state()[0][:, self.item_obj]
        self.lifted |= pi[..., 2] > self.p_start[:, self.item_obj, 2] + LIFTED_HEIGHT
        return self._drop_exploded(obs, rew, done, info)

    def _episode_stats(self, d: torch.Tensor, all_done: torch.Tensor) -> dict:
        makespan = torch.where(self.makespan[d] >= 0, self.makespan[d], self.t[d].float() * self.dt)
        return dict(
            env=d.cpu().numpy(),
            placed=self.placed[d].sum(1).cpu().numpy(),
            success=all_done[d].cpu().numpy(),
            makespan_s=makespan.cpu().numpy(),
            ret=self.ret[d].cpu().numpy(),
            arm_contact_steps=self.arm_contact_steps[d].cpu().numpy(),
            min_arm_dist=self.min_hand_dist[d].cpu().numpy(),
            nontarget_steps=self.nontarget_steps[d].cpu().numpy(),
            knocked=self.knocked[d].sum(1).cpu().numpy(),
            idle_s=(self.idle[d] * self.dt).cpu().numpy(),
        )

    def _spawn_check(self, obs, rew, done, info):
        """An env in which any object moved more than 1 cm (or became non-finite) in its first 0.5 s gets a new layout;
        its transition is cut and nothing of it is logged. After three failed layouts in a row the layout is kept."""
        p, _ = self.object_state()
        first = self.t == 1
        self.spawn_p1[first] = p[first]
        check = (self.t == SPAWN_CHECK_STEP) & self.spawn_pending
        info["spawn_rejected"] = 0
        if not bool(check.any()):
            return obs, rew, done, info
        moved = (p - self.spawn_p1).norm(dim=-1)
        bad_obj = (moved > SPAWN_MOVE) | ~torch.isfinite(moved)
        bad = check & bad_obj.any(1)
        give_up = bad & (self.spawn_retries >= SPAWN_RETRIES)
        self.spawn_stats["accepted_after_retries"] += int(give_up.sum())
        bad = bad & ~give_up
        self.spawn_retries = torch.where(
            bad, self.spawn_retries + 1, torch.where(check, torch.zeros_like(self.spawn_retries), self.spawn_retries)
        )
        self.spawn_stats["checked"] += int(check.sum())
        self.spawn_stats["rejected"] += int(bad.sum())
        self.spawn_pending[check] = False
        if bool(bad.any()):
            for k in torch.nonzero(bad_obj[bad].any(0))[:, 0].tolist():
                by_obj = self.spawn_stats["moved_by_object"]
                by_obj[self.names[k]] = by_obj.get(self.names[k], 0) + int(bad_obj[bad, k].sum())
            ids = torch.nonzero(bad)[:, 0]
            info["spawn_rejected"] = len(ids)
            self.reset_envs(ids)
            self.needs_pointer[ids] = True
            rew = torch.where(bad, torch.zeros_like(rew), rew)
            done = done | bad
            pc, qc = self.object_state()
            obs = self.last_obs = self._obs(pc, qc)
        return obs, rew, done, info

    def _exploded_envs(self) -> torch.Tensor:
        p, q = self.object_state()
        bad = ~(torch.isfinite(p).all(-1).all(-1) & torch.isfinite(q).all(-1).all(-1)) | (
            p.abs().amax(-1).amax(-1) > FAR_AWAY
        )
        for a in ARMS:
            r = self.sim.robots[a]
            bad |= ~torch.isfinite(r.data.joint_pos.torch).all(-1) | ~torch.isfinite(r.data.joint_vel.torch).all(-1)
            bad |= ~torch.isfinite(r.data.body_pos_w.torch).all(-1).all(-1)
        return bad

    def _nan_guard(self) -> int:
        count = 0
        for _ in range(NAN_GUARD_TRIES):
            bad = self._exploded_envs()
            if not bool(bad.any()):
                break
            ids = torch.nonzero(bad)[:, 0]
            count += len(ids)
            self.reset_envs(ids)
        self.nan_resets += count
        return count

    def _drop_exploded(self, obs, rew, done, info):
        """Reset envs whose simulation blew up; their transition is cut and their episode is not scored."""
        bad = self._exploded_envs()
        info["nan_reset"] = 0
        info["nan_ids"] = []
        if not bool(bad.any()):
            return obs, rew, done, info
        ids = torch.nonzero(bad)[:, 0]
        info["nan_reset"] = len(ids)
        info["nan_ids"] = ids.cpu().tolist()
        self.nan_resets += len(ids)
        if "final" in info:
            keep = ~np.isin(info["final"]["env"], ids.cpu().numpy())
            info["final"] = {
                k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
                for k, v in info["final"].items()
            }
            if len(info["final"]["env"]) == 0:
                del info["final"]
        rew = torch.where(bad, torch.zeros_like(rew), torch.nan_to_num(rew))
        done = done | bad
        self.exploded[ids] = True
        self.reset_envs(ids)
        self.needs_pointer[ids] = True
        pc, qc = self.object_state()
        obs = self.last_obs = self._obs(pc, qc)
        return obs, rew, done, info

    def side_dim(self) -> int:
        """Width of obs['obst_near']: nearest boxes, the other arm's state and the path points ahead."""
        return 8 * NEAR_OBSTACLES + 3 * len(self.link_bids) + 2 * 23 + 1 + 3 * len(self.cfg.path_points)

    def set_applied_action(self, applied: torch.Tensor) -> None:
        """Write the applied action (N, 2, 23) into each arm's last-action observation term."""
        for ai, a in enumerate(ARMS):
            self.lift_obs[a].set_action(applied[:, ai])

    def close(self):
        self.sim.close()
