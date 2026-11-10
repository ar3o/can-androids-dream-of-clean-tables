"""Two Kuka-Allegro arms at a twin of the real table (cleantables.twin.table_scene), standing side by side on the side of the table
where the person stood, facing the camera. An Isaac Lab DirectRLEnv used as the simulation container; episode logic lives in the caller.

World (env) frame = the episode's table frame: z up, table top at z = 0. Each arm base is TABLE_DROP = 0.255 m below the table top and
BASE_GAP = 0.15 m behind the far table edge, yawed so its -x axis (the published Lift task's "forward") points across the table: the
table therefore sits in front of each arm exactly as in the Lift task's training scene (table top z = 0.255 in the base frame).

Objects: every scene.json object as a rigid body (twin physics: depenetration cap, real inertia, no bounce); containers (fruit bowl,
recycling box) and the dustpan are kinematic. Control modes per arm: POLICY / OPEN / IK.
"""
from __future__ import annotations

import json
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TABLE_DROP = 0.255
BASE_GAP = 0.15
ARMS = ("a", "b")
POLICY, OPEN, IK = 0, 1, 2
FINGERTIPS = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
HAND_BODIES = (["palm_link", "allegro_mount"] + [f"{f}_link_{i}" for f in ("index", "middle", "ring", "thumb") for i in range(4)]
               + [f"{f}_biotac_tip" for f in ("index", "middle", "ring", "thumb")])
FIXED = {"container/fruit_bowl", "container/recycling_box", "container/bin", "tool/dustpan"}


REACH_XB = (-0.80, -0.25)       # base-frame box where the published lift policy works (its training objects/goals lie inside)
REACH_YB = 0.34


def reach_b(base_xy, xy):
    """Yaw +90: base x = world +y offset, base y = -(world x offset)."""
    import numpy as np
    d = np.asarray(xy, float) - np.asarray(base_xy, float)
    xb, yb = d[..., 1], -d[..., 0]
    return (xb >= REACH_XB[0]) & (xb <= REACH_XB[1]) & (np.abs(yb) <= REACH_YB)


def arm_poses(scene, sep=None, tasks=None):
    """Bases side by side behind the far table edge (+y side, where the person stood), facing the camera. The x-centre and separation
    (0.44-0.60 m, about human shoulder width; at 0.32 m the arms clash) are chosen to maximise the number of tasks one arm can do alone
    (object and destination in its reach), tie-break: wider separation (fewer arm-arm clashes)."""
    import numpy as np
    y_b = scene["table"]["y"][1] + BASE_GAP
    objs = scene["objects"]
    xs = [o["pos"][0] for o in objs]
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))          # yaw +90 deg: base x -> world +y, base -x -> world -y (xyzw)
    best, arg = -1, (0.5 * (min(xs) + max(xs)), sep or 0.6)
    if tasks:
        for xc in np.arange(min(xs) - 0.2, max(xs) + 0.2, 0.02):
            for sp in ([sep] if sep else np.arange(0.44, 0.62, 0.04)):
                ok, na, nb = 0, 0, 0
                for k, kc, _ in tasks:
                    po, pd = objs[k]["pos"][:2], objs[kc]["pos"][:2]
                    ra = bool(reach_b((xc + sp / 2, y_b), po) and reach_b((xc + sp / 2, y_b), pd))
                    rb = bool(reach_b((xc - sp / 2, y_b), po) and reach_b((xc - sp / 2, y_b), pd))
                    ok += ra or rb; na += ra; nb += rb
                score = ok + 0.3 * min(na, nb) + 0.01 * sp        # cover the tasks, then share them between both arms
                if score > best:
                    best, arg = score, (float(xc), float(sp))
    xc, sp = arg
    return {"a": ((xc + sp / 2, y_b, -TABLE_DROP), q), "b": ((xc - sp / 2, y_b, -TABLE_DROP), q)}


def build_scene_cfg(num_envs, scene, poses, record_cam=False, arm_contact=True):
    tb = scene["table"]
    sdir = os.path.join(ROOT, "data/twin_table", scene["episode"])

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.55, 0.55, 0.55)),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.76)))
        dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 0.95, 0.88)))
        key = AssetBaseCfg(prim_path="/World/key", spawn=sim_utils.DistantLightCfg(intensity=700.0, angle=25.0, color=(1.0, 0.93, 0.85)))
        table_col = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/TableCol", spawn=sim_utils.CuboidCfg(
            size=(tb["x"][1] - tb["x"][0], tb["y"][1] - tb["y"][0], 0.04), visible=False,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True), collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=((tb["x"][0] + tb["x"][1]) / 2, (tb["y"][0] + tb["y"][1]) / 2, -0.02)))
        table_vis = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/TableVis", spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(sdir, "table.usda")))

    cfg = SceneCfg(num_envs=num_envs, env_spacing=3.5, replicate_physics=True)
    for arm in ARMS:
        r = replace(KUKA_ALLEGRO_CFG, prim_path="{ENV_REGEX_NS}/Robot" + arm.upper())
        r.init_state = replace(r.init_state, pos=poses[arm][0], rot=poses[arm][1])
        setattr(cfg, f"robot_{arm}", r)
        setattr(cfg, f"ped_{arm}", AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Ped" + arm.upper(), spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.22, 0.76 - TABLE_DROP - 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(poses[arm][0][0], poses[arm][0][1], -0.76 + (0.76 - TABLE_DROP - 0.02) / 2))))
    for k, o in enumerate(scene["objects"]):
        fixed = o["category"] in FIXED
        rp = [sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=fixed),
              PhysxRigidBodyCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=0.5,
                                linear_damping=o["damping"][0], angular_damping=o["damping"][1])]
        spawn = sim_utils.UsdFileCfg(usd_path=os.path.join(ROOT, o["usd"]), rigid_props=rp,
                                     collision_props=PhysxCollisionCfg(contact_offset=0.004, rest_offset=0.0))
        setattr(cfg, f"obj_{k}", RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Obj_{k}", spawn=spawn,
                                                init_state=RigidObjectCfg.InitialStateCfg(pos=(o["pos"][0], o["pos"][1], o["pos"][2] + 0.003))))
    obj_paths = [f"{{ENV_REGEX_NS}}/Obj_{k}" for k in range(len(scene["objects"]))]
    for arm in ARMS:
        R = "{ENV_REGEX_NS}/Robot" + arm.upper()
        for link in FINGERTIPS:
            setattr(cfg, f"tip_{arm}_{link}", ContactSensorCfg(prim_path=f"{R}/ee_link/{link}", filter_prim_paths_expr=list(obj_paths)))
    if arm_contact:
        b_bodies = ([f"ee_link/{n}" for n in HAND_BODIES] + [f"iiwa7_link_{i}" for i in range(2, 8)])
        filt = [f"{{ENV_REGEX_NS}}/RobotB/{b}" for b in b_bodies]
        cfg.armcontact_hand = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/RobotA/ee_link/.*", filter_prim_paths_expr=filt)
        cfg.armcontact_arm = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/RobotA/iiwa7_link_[2-7]", filter_prim_paths_expr=filt)
    if record_cam:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg
        import numpy as np
        K = np.array(scene["K"])
        cfg.rec_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/PhoneCam", update_period=0.0, data_types=["rgb", "distance_to_image_plane"],
                                width=960, height=540,
                                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=K.reshape(-1).tolist(), width=960, height=540,
                                                                                      clipping_range=(0.02, 20.0)),
                                renderer_cfg=IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
                                    antialiasing_mode="DLAA", enable_dl_denoiser=True, enable_dlssg=False, samples_per_pixel=4,
                                    enable_shadows=True, enable_ambient_occlusion=True, enable_reflections=True)))
    return cfg


@configclass
class TwinRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 1.0e5
    decimation = 8                  # 240 Hz physics, 30 Hz control (as the Lift task's 30 Hz policy)
    action_space = 46
    observation_space = 1
    state_space = 0
    num_rerenders_on_reset = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=8, physics=PhysxCfg(
        enable_stabilization=True, bounce_threshold_velocity=0.2, gpu_max_rigid_patch_count=2**19, gpu_found_lost_pairs_capacity=2**21,
        gpu_found_lost_aggregate_pairs_capacity=2**21, gpu_total_aggregate_pairs_capacity=2**21, gpu_max_rigid_contact_count=2**21,
        gpu_collision_stack_size=2**25, gpu_heap_capacity=2**25, gpu_temp_buffer_capacity=2**23))
    scene: InteractiveSceneCfg = None


class TwinRobotEnv(DirectRLEnv):
    cfg: TwinRobotEnvCfg

    def __init__(self, cfg, scene_json, **kw):
        self.twin = scene_json
        super().__init__(cfg, None, **kw)
        dev, N = self.device, self.num_envs
        self.K = len(scene_json["objects"])
        self.robots = {a: self.scene[f"robot_{a}"] for a in ARMS}
        self.objs = [self.scene[f"obj_{k}"] for k in range(self.K)]
        r = self.robots["a"]
        self.joint_names = r.joint_names
        self.arm_jids = torch.tensor(r.find_joints("iiwa7_joint_.*")[0], device=dev)
        self.finger_jids = torch.tensor([j for j in range(r.num_joints) if j not in self.arm_jids.tolist()], device=dev)
        self.hand_bids = r.find_bodies(["palm_link", ".*_tip"])[0]
        self.palm_bid = r.find_bodies("palm_link")[0][0]
        self.jac_body = self.palm_bid - 1 if r.is_fixed_base else self.palm_bid
        lim = r.data.joint_pos_limits.torch[0]
        self.q_lo, self.q_hi = lim[:, 0].clone(), lim[:, 1].clone()
        self.finger_open = torch.zeros(r.num_joints, device=dev).clamp(self.q_lo, self.q_hi)[self.finger_jids]
        self.finger_default = r.data.default_joint_pos.torch[0, self.finger_jids].clone()
        self.mode = {a: torch.zeros(N, dtype=torch.long, device=dev) for a in ARMS}
        self.action = {a: torch.zeros(N, 23, device=dev) for a in ARMS}
        self.q_hold = {a: self.robots[a].data.default_joint_pos.torch.clone() for a in ARMS}
        self.palm_target = {a: torch.zeros(N, 3, device=dev) for a in ARMS}
        self.q_ik = {a: self.robots[a].data.default_joint_pos.torch.clone() for a in ARMS}
        self.off_ema = {a: torch.zeros(N, r.num_joints, device=dev) for a in ARMS}
        self._prev_mode = {a: torch.zeros(N, dtype=torch.long, device=dev) for a in ARMS}
        self.finger_target = {a: self.finger_default.repeat(N, 1) for a in ARMS}
        self.tip_sensors = {a: [self.scene.sensors[f"tip_{a}_{l}"] for l in FINGERTIPS] for a in ARMS}
        self.arm_sensors = [self.scene.sensors[s] for s in ("armcontact_hand", "armcontact_arm") if s in self.scene.sensors]

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        for a in ARMS:
            r = self.robots[a]
            q = r.data.joint_pos.torch
            m = self.mode[a]
            ent = (m != POLICY) & (self._prev_mode[a] == POLICY)
            self.q_ik[a] = torch.where(ent[:, None], q + self.off_ema[a], self.q_ik[a])
            self.q_hold[a] = torch.where(ent[:, None], q + self.off_ema[a], self.q_hold[a])
            self._prev_mode[a] = m.clone()
            tgt = q + 0.1 * self.action[a]
            self.off_ema[a] = torch.where((m == POLICY)[:, None], 0.95 * self.off_ema[a] + 0.05 * 0.1 * self.action[a], self.off_ema[a])
            q_ik = self._ik_targets(a)
            q_ik[:, self.finger_jids] = self.finger_target[a]
            q_open = self.q_hold[a].clone()
            q_open[:, self.finger_jids] = self.finger_open
            tgt = torch.where((m == IK)[:, None], q_ik, torch.where((m == OPEN)[:, None], q_open, tgt))
            r.set_joint_position_target_index(target=tgt)

    def _ik_targets(self, a):
        r = self.robots[a]
        jac = r.data.body_link_jacobian_w.torch[:, self.jac_body, :3][:, :, self.arm_jids]
        palm = r.data.body_pos_w.torch[:, self.palm_bid]
        err = (self.palm_target[a] - palm).clamp(-0.02, 0.02)
        lam = 0.05
        JT = jac.transpose(1, 2)
        dq = (JT @ torch.linalg.solve_ex(jac @ JT + lam**2 * torch.eye(3, device=self.device), err[:, :, None])[0])[:, :, 0]
        q_t = self.q_ik[a].clone()
        q_arm = r.data.joint_pos.torch[:, self.arm_jids]
        new = q_t[:, self.arm_jids] + 0.5 * dq.clamp(-0.02, 0.02)     # gain halved per substep: tuned at 120 Hz, physics here runs at 240 Hz
        new = torch.maximum(torch.minimum(new, q_arm + 0.25), q_arm - 0.25)
        q_t[:, self.arm_jids] = new.clamp(self.q_lo[self.arm_jids], self.q_hi[self.arm_jids])
        self.q_ik[a] = q_t
        return q_t.clone()

    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z

    def set_friction(self, asset, lo=0.5, hi=1.0, env_ids=None):
        import warp as wp
        before = wp.to_torch(asset.root_view.get_material_properties()).clone()
        mats = before.clone()
        ids = torch.arange(self.num_envs, dtype=torch.int32) if env_ids is None else env_ids.cpu().int()
        f = lo + (hi - lo) * torch.rand(len(ids), mats.shape[1])
        mats[ids.long(), :, 0] = f
        mats[ids.long(), :, 1] = f
        mats[ids.long(), :, 2] = 0.0
        asset.root_view.set_material_properties(wp.from_torch(mats.contiguous(), dtype=wp.float32), wp.from_torch(ids, dtype=wp.int32))
        return before

    def tip_forces(self, a, target):
        N = self.num_envs
        f = torch.stack([s.data.normal_force_matrix_w.torch.reshape(N, -1, 3) for s in self.tip_sensors[a]], 1)
        idx = target.clamp(min=0)[:, None, None, None].expand(-1, 4, 1, 3)
        out = f.gather(2, idx)[:, :, 0]
        return torch.where((target >= 0)[:, None, None], out, torch.zeros_like(out))

    def arm_contact_force(self):
        if not self.arm_sensors:
            return torch.zeros(self.num_envs, device=self.device)
        m = [s.data.normal_force_matrix_w.torch.reshape(self.num_envs, -1, 3).norm(dim=-1).amax(1) for s in self.arm_sensors]
        return torch.stack(m, 1).amax(1)
