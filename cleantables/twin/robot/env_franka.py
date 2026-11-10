"""Two Franka Panda arms (parallel grippers) at a twin of the real table: the Franka counterpart of cleantables.twin.robot.env_kuka (Kuka-Allegro).

World (env) frame = the episode's table frame, z up, table top at z = 0.  Each Franka stands where the Kuka stood (scene.json poses, 0.255 m below the table
top, as in the Lift task's scene) but turned 180 deg about z: the published Isaac-Lift-Franka task has its robot yawed 180 deg so the workspace lies at base +x
(Kuka: base -x).  franka_mount() additionally slides each base along its own forward axis / sideways (generic: whatever maximises the number of tasks one arm
can do alone given the Franka's smaller Lift workspace, box FR_X x |y| <= FR_Y in the base frame, tie -> smallest move); nothing else of the scene changes.

Control (as the Lift task): action (9) = relative joint targets for panda_joint1-7 and the two finger joints, target = q + 0.1 * action, 30 Hz policy, 240 Hz
physics (decimation 8).  The gripper (panda_finger_joint1 driven, finger_joint2 mimics it) is part of that action: the policy opens and closes it.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass, replace
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

import cleantables.twin.robot.env_kuka as E
from cleantables.twin.robot.env_kuka import ARMS, FIXED, POLICY, ROOT, TABLE_DROP, TwinRobotEnvCfg  # noqa: F401

FR_X = (0.30, 0.78)        # m, base-frame box (x forward) in which the Franka Lift policy works (its training objects 0.35-0.75, goals 0.3-0.7)
FR_Y = 0.30
FINGERS = ["panda_rightfinger", "panda_leftfinger"]                 # contact-sensor order of the Lift task
LINKS = [f"panda_link{i}" for i in range(8)]
GEOM = "/Geometry/" + "/".join(LINKS[:8]) + "/panda_hand"          # panda_link0/.../panda_link7/panda_hand


def link_path(R, name):
    if name in LINKS:
        return R + "/Geometry/" + "/".join(LINKS[:LINKS.index(name) + 1])
    return R + GEOM + ("" if name == "panda_hand" else "/" + name)


def franka_reach(base_xy, yaw, xy, dl=0.0):
    """Franka Lift workspace test, base frame (x forward = world direction of the yaw); dl = how far the base was slid backwards."""
    d = np.asarray(xy, float) - np.asarray(base_xy, float)
    c, s = math.cos(yaw), math.sin(yaw)
    xb, yb = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
    return (xb >= FR_X[0]) & (xb <= FR_X[1]) & (np.abs(yb) <= FR_Y)


def franka_mount(scene):
    """-> deep-copied scene with Franka poses/yaw: same base places as the Kuka scene (yaw + 180 deg), then a generic search for the base shifts (forward -0.05..+0.20 m
    away from the table edge, sideways +-0.15 m) that maximise the number of tasks (object and container reachable) one arm can do alone; the two bases keep >= 0.35 m."""
    import copy
    S = copy.deepcopy(scene)
    ky = math.radians(float(S["arms"]["yaw_deg"]))
    fy = ky + math.pi
    objs = S["objects"]; ids = [o["id"] for o in objs]
    tasks = [(o["pos"][:2], objs[ids.index(o["dest"])]["pos"][:2]) for o in objs if o["role"] == "task"]
    fwd = np.array([math.cos(fy), math.sin(fy)]); side = np.array([-math.sin(fy), math.cos(fy)])      # Franka base +x, +y in the world
    base0 = {a: np.array(S["poses"][a][0][:2], float) for a in ARMS}
    opts = [(f, y) for f in np.arange(-0.05, 0.2001, 0.025) for y in np.arange(-0.15, 0.1501, 0.05)]

    def cover(a, f, y):
        b = base0[a] - f * fwd + y * side            # base slid backwards by f (the arm's reach direction is +x of the new frame -> backwards = -fwd)
        return np.array([bool(franka_reach(b, fy, po) and franka_reach(b, fy, pd)) for po, pd in tasks]), b
    cv = {a: [cover(a, f, y) for f, y in opts] for a in ARMS}
    best = None
    for i, (fa, ya) in enumerate(opts):
        for j, (fb, yb_) in enumerate(opts):
            if np.linalg.norm(cv["a"][i][1] - cv["b"][j][1]) < 0.35:
                continue
            k = int((cv["a"][i][0] | cv["b"][j][0]).sum())
            sc = (k, int(cv["a"][i][0].sum() > 0) + int(cv["b"][j][0].sum() > 0), -(abs(fa) + abs(fb) + abs(ya) + abs(yb_)))
            if best is None or sc > best[0]:
                best = (sc, i, j)
    _, i, j = best
    q = (0.0, 0.0, math.sin(fy / 2), math.cos(fy / 2))
    for a, k in (("a", i), ("b", j)):
        b = cv[a][k][1]
        S["poses"][a] = [[float(b[0]), float(b[1]), -TABLE_DROP], list(q)]
    S["arms"]["yaw_deg"] = math.degrees(fy)
    S["arms"]["franka_mount"] = dict(tasks_one_arm=f"{best[0][0]}/{len(tasks)}", shift_a=[float(x) for x in opts[i]], shift_b=[float(x) for x in opts[j]])
    return S


def build_scene_cfg(num_envs, scene, poses, record_cam=False, arm_contact=True):
    from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FRANKA_PANDA_LIFT_CFG
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
        r = replace(FRANKA_PANDA_LIFT_CFG, prim_path="{ENV_REGEX_NS}/Robot" + arm.upper())
        r.spawn = replace(r.spawn, activate_contact_sensors=True)
        r.spawn.variants = {"Physics": "physx", "Colliders": "primitives"}
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
        spawn = sim_utils.UsdFileCfg(usd_path=os.path.join(ROOT, o["usd"]), rigid_props=rp, collision_props=PhysxCollisionCfg(contact_offset=0.004, rest_offset=0.0))
        setattr(cfg, f"obj_{k}", RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Obj_{k}", spawn=spawn,
                                                init_state=RigidObjectCfg.InitialStateCfg(pos=(o["pos"][0], o["pos"][1], o["pos"][2] + 0.003))))
    obj_paths = [f"{{ENV_REGEX_NS}}/Obj_{k}" for k in range(len(scene["objects"]))]
    for arm in ARMS:
        R = "{ENV_REGEX_NS}/Robot" + arm.upper()
        for link in FINGERS:
            setattr(cfg, f"tip_{arm}_{link}", ContactSensorCfg(prim_path=link_path(R, link), filter_prim_paths_expr=list(obj_paths)))
    if arm_contact:
        b_bodies = [link_path("{ENV_REGEX_NS}/RobotB", n) for n in LINKS[2:] + ["panda_hand"] + FINGERS]
        cfg.armcontact_hand = ContactSensorCfg(prim_path=link_path("{ENV_REGEX_NS}/RobotA", "panda_hand") + "/.*finger", filter_prim_paths_expr=b_bodies)
        cfg.armcontact_arm = ContactSensorCfg(prim_path=link_path("{ENV_REGEX_NS}/RobotA", "panda_link7").rsplit("/", 1)[0] + "/panda_link[2-7]", filter_prim_paths_expr=b_bodies)
    if record_cam:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg
        K = np.array(scene["K"])
        cfg.rec_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/PhoneCam", update_period=0.0, data_types=["rgb", "distance_to_image_plane"], width=960, height=540,
                                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=K.reshape(-1).tolist(), width=960, height=540, clipping_range=(0.02, 20.0)),
                                renderer_cfg=IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
                                    antialiasing_mode="DLAA", enable_dl_denoiser=True, enable_dlssg=False, samples_per_pixel=4,
                                    enable_shadows=True, enable_ambient_occlusion=True, enable_reflections=True)))
    return cfg


class TwinFrankaEnv(DirectRLEnv):
    """Simulation container (as TwinRobotEnv).  Same attribute names (palm_bid, hand_bids, tip_sensors, mode, q_hold, ...), so the joint env runs on either robot.
    palm_bid = panda_hand; hand_bids = the two fingers in articulation order (the Lift task's hand_tips_state_b order)."""

    def __init__(self, cfg, scene_json, **kw):
        self.twin = scene_json
        super().__init__(cfg, None, **kw)
        dev, N = self.device, self.num_envs
        self.K = len(scene_json["objects"])
        self.robots = {a: self.scene[f"robot_{a}"] for a in ARMS}
        self.objs = [self.scene[f"obj_{k}"] for k in range(self.K)]
        r = self.robots["a"]
        self.joint_names = r.joint_names
        self.arm_jids = torch.tensor(r.find_joints("panda_joint.*")[0], device=dev)
        self.finger_jids = torch.tensor(r.find_joints("panda_finger_joint.*")[0], device=dev)
        self.hand_bids = r.find_bodies(FINGERS)[0]                       # sorted by body index: (leftfinger, rightfinger)
        self.palm_bid = r.find_bodies("panda_hand")[0][0]
        self.finger_default = r.data.default_joint_pos.torch[0, self.finger_jids].clone()
        self.mode = {a: torch.zeros(N, dtype=torch.long, device=dev) for a in ARMS}
        self.action = {a: torch.zeros(N, 9, device=dev) for a in ARMS}
        self.q_hold = {a: self.robots[a].data.default_joint_pos.torch.clone() for a in ARMS}
        self.palm_target = {a: torch.zeros(N, 3, device=dev) for a in ARMS}
        self.q_ik = {a: self.robots[a].data.default_joint_pos.torch.clone() for a in ARMS}
        self.off_ema = {a: torch.zeros(N, r.num_joints, device=dev) for a in ARMS}
        self._prev_mode = {a: torch.zeros(N, dtype=torch.long, device=dev) for a in ARMS}
        self.finger_target = {a: self.finger_default.repeat(N, 1) for a in ARMS}
        self.tip_sensors = {a: [self.scene.sensors[f"tip_{a}_{l}"] for l in FINGERS] for a in ARMS}
        self.arm_sensors = [self.scene.sensors[s] for s in ("armcontact_hand", "armcontact_arm") if s in self.scene.sensors]

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        for a in ARMS:
            r = self.robots[a]
            q = r.data.joint_pos.torch
            m = self.mode[a]
            ent = (m != POLICY) & (self._prev_mode[a] == POLICY)
            self.q_hold[a] = torch.where(ent[:, None], q, self.q_hold[a])             # scripted release (open phase): arm held where it was, gripper opened
            self._prev_mode[a] = m.clone()
            tgt = q + 0.1 * self.action[a]                                          # the Lift task's RelativeJointPositionAction (scale 0.1), arm and gripper
            q_open = self.q_hold[a].clone()
            q_open[:, self.finger_jids] = self.finger_default
            r.set_joint_position_target_index(target=torch.where((m != POLICY)[:, None], q_open, tgt))

    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z

    set_friction = E.TwinRobotEnv.set_friction

    def tip_forces(self, a, target):
        """(N,2,3) finger->target-object contact forces in the Lift task's contact order (rightfinger, leftfinger)."""
        N = self.num_envs
        f = torch.stack([s.data.normal_force_matrix_w.torch.reshape(N, -1, 3) for s in self.tip_sensors[a]], 1)
        idx = target.clamp(min=0)[:, None, None, None].expand(-1, f.shape[1], 1, 3)
        out = f.gather(2, idx)[:, :, 0]
        return torch.where((target >= 0)[:, None, None], out, torch.zeros_like(out))

    arm_contact_force = E.TwinRobotEnv.arm_contact_force
