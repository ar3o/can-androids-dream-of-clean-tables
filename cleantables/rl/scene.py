"""Isaac Lab scene of one twin with two Kuka-Allegro arms, and the DirectRLEnv used as the simulation container.

Env frame = the episode's table frame: z up, table top at z = 0. The arm bases come from scene.json and sit 0.255 m
below the table top, so the table is placed in front of each arm as in the lift task's training scene. Import after the
Isaac app is launched.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

from .config import ARMS
from .lift_prior import ACTION_SCALE

TABLE_DROP = 0.255
FLOOR_Z = -0.76
PEDESTAL_SIZE = 0.22
OFF_TABLE = 1.5  # m, how far arm b is moved back in single-arm mode
FINGERTIPS = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
HAND_BODIES = (
    ["palm_link", "allegro_mount"]
    + [f"{f}_link_{i}" for f in ("index", "middle", "ring", "thumb") for i in range(4)]
    + [f"{f}_biotac_tip" for f in ("index", "middle", "ring", "thumb")]
)
KINEMATIC_CATEGORIES = {
    "container/fruit_bowl",
    "container/recycling_box",
    "container/bin",
    "tool/dustpan",
    "container/toolbox",
}
TABLE_FRICTION = (0.6, 0.5)
ENV_SPACING = 3.5


def build_scene_cfg(num_envs: int, scene: dict, single_arm: bool = False) -> InteractiveSceneCfg:
    tb = scene["table"]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(color=(0.55, 0.55, 0.55)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, FLOOR_Z)),
        )
        dome = AssetBaseCfg(
            prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 0.95, 0.88))
        )
        key = AssetBaseCfg(
            prim_path="/World/key",
            spawn=sim_utils.DistantLightCfg(intensity=700.0, angle=25.0, color=(1.0, 0.93, 0.85)),
        )
        table_col = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TableCol",
            spawn=sim_utils.CuboidCfg(
                size=(tb["x"][1] - tb["x"][0], tb["y"][1] - tb["y"][0], 0.04),
                visible=False,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=TABLE_FRICTION[0], dynamic_friction=TABLE_FRICTION[1], restitution=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=((tb["x"][0] + tb["x"][1]) / 2, (tb["y"][0] + tb["y"][1]) / 2, -0.02)
            ),
        )
        table_vis = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TableVis", spawn=sim_utils.UsdFileCfg(usd_path=f"{scene['dir']}/table.usda")
        )

    cfg = SceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING, replicate_physics=True)
    poses = {a: [list(p[0]), p[1]] for a, p in scene["poses"].items()}
    if single_arm:  # arm b stands 1.5 m further back from the table so it cannot get in the way
        poses["b"][0][1] += OFF_TABLE if poses["b"][0][1] >= 0 else -OFF_TABLE
    pedestal_h = -FLOOR_Z - TABLE_DROP - 0.02
    for arm in ARMS:
        robot = replace(KUKA_ALLEGRO_CFG, prim_path="{ENV_REGEX_NS}/Robot" + arm.upper())
        robot.init_state = replace(robot.init_state, pos=tuple(poses[arm][0]), rot=tuple(poses[arm][1]))
        setattr(cfg, f"robot_{arm}", robot)
        setattr(
            cfg,
            f"ped_{arm}",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Ped" + arm.upper(),
                spawn=sim_utils.CuboidCfg(
                    size=(PEDESTAL_SIZE, PEDESTAL_SIZE, pedestal_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(poses[arm][0][0], poses[arm][0][1], FLOOR_Z + pedestal_h / 2)
                ),
            ),
        )
    for k, o in enumerate(scene["objects"]):
        rigid = [
            sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=o["category"] in KINEMATIC_CATEGORIES),
            PhysxRigidBodyCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=0.5,
                linear_damping=o["damping"][0],
                angular_damping=o["damping"][1],
            ),
        ]
        spawn = sim_utils.UsdFileCfg(
            usd_path=o["usd"],
            rigid_props=rigid,
            collision_props=PhysxCollisionCfg(contact_offset=0.004, rest_offset=0.0),
        )
        setattr(
            cfg,
            f"obj_{k}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Obj_{k}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(o["pos"][0], o["pos"][1], o["pos"][2] + 0.003)),
            ),
        )
    obj_paths = [f"{{ENV_REGEX_NS}}/Obj_{k}" for k in range(len(scene["objects"]))]
    for arm in ARMS:
        prefix = "{ENV_REGEX_NS}/Robot" + arm.upper()
        for link in FINGERTIPS:
            setattr(
                cfg,
                f"tip_{arm}_{link}",
                ContactSensorCfg(prim_path=f"{prefix}/ee_link/{link}", filter_prim_paths_expr=list(obj_paths)),
            )
    b_bodies = [f"ee_link/{n}" for n in HAND_BODIES] + [f"iiwa7_link_{i}" for i in range(2, 8)]
    other = [f"{{ENV_REGEX_NS}}/RobotB/{b}" for b in b_bodies]
    cfg.armcontact_hand = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/RobotA/ee_link/.*", filter_prim_paths_expr=other)
    cfg.armcontact_arm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotA/iiwa7_link_[2-7]", filter_prim_paths_expr=other
    )
    return cfg


@configclass
class TwinSimCfg(DirectRLEnvCfg):
    episode_length_s = 1.0e5  # episodes are ended by CleanupEnv, never by the container
    decimation = 8  # 240 Hz physics, 30 Hz control as the lift task
    action_space = 46
    observation_space = 1
    state_space = 0
    num_rerenders_on_reset = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=8,
        physics=PhysxCfg(
            enable_stabilization=True,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=2**19,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_max_rigid_contact_count=2**21,
            gpu_collision_stack_size=2**25,
            gpu_heap_capacity=2**25,
            gpu_temp_buffer_capacity=2**23,
        ),
    )
    scene: InteractiveSceneCfg = None


def make_sim_cfg(scene: dict, num_envs: int, seed: int, single_arm: bool = False) -> TwinSimCfg:
    cfg = TwinSimCfg()
    cfg.scene = build_scene_cfg(num_envs, scene, single_arm)
    cfg.seed = seed
    return cfg


class TwinSim(DirectRLEnv):
    """Two arms under joint position control (target = q + 0.1 * action); actions are set per arm in `self.action`."""

    cfg: TwinSimCfg

    def __init__(self, cfg: TwinSimCfg, scene: dict, **kw):
        self.twin = scene
        super().__init__(cfg, None, **kw)
        dev, n = self.device, self.num_envs
        self.num_objects = len(scene["objects"])
        self.robots = {a: self.scene[f"robot_{a}"] for a in ARMS}
        self.objs = [self.scene[f"obj_{k}"] for k in range(self.num_objects)]
        r = self.robots["a"]
        self.joint_names = r.joint_names
        arm_jids = r.find_joints("iiwa7_joint_.*")[0]
        self.finger_jids = torch.tensor([j for j in range(r.num_joints) if j not in arm_jids], device=dev)
        self.hand_bids = r.find_bodies(["palm_link", ".*_tip"])[0]
        self.palm_bid = r.find_bodies("palm_link")[0][0]
        lim = r.data.joint_pos_limits.torch[0]
        self.finger_open = torch.zeros(r.num_joints, device=dev).clamp(lim[:, 0], lim[:, 1])[self.finger_jids]
        self.action = {a: torch.zeros(n, r.num_joints, device=dev) for a in ARMS}
        self.tip_sensors = {a: [self.scene.sensors[f"tip_{a}_{link}"] for link in FINGERTIPS] for a in ARMS}
        self.arm_sensors = [self.scene.sensors[s] for s in ("armcontact_hand", "armcontact_arm")]

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        for a in ARMS:
            r = self.robots[a]
            r.set_joint_position_target_index(target=r.data.joint_pos.torch + ACTION_SCALE * self.action[a])

    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z

    def set_friction(self, asset, lo: float, hi: float, env_ids: torch.Tensor | None = None) -> None:
        """Static and dynamic friction of every shape of `asset` ~ U(lo, hi) per env and shape (PhysX material API)."""
        mats = wp.to_torch(asset.root_view.get_material_properties()).clone()
        ids = torch.arange(self.num_envs, dtype=torch.int32) if env_ids is None else env_ids.cpu().int()
        f = lo + (hi - lo) * torch.rand(len(ids), mats.shape[1])
        mats[ids.long(), :, 0] = f
        mats[ids.long(), :, 1] = f
        mats[ids.long(), :, 2] = 0.0
        asset.root_view.set_material_properties(
            wp.from_torch(mats.contiguous(), dtype=wp.float32), wp.from_torch(ids, dtype=wp.int32)
        )

    def tip_forces(self, arm: str, target: torch.Tensor) -> torch.Tensor:
        """(N, 4, 3) contact forces between the four fingertips of `arm` and object `target` (-1 = none)."""
        n = self.num_envs
        f = torch.stack([s.data.normal_force_matrix_w.torch.reshape(n, -1, 3) for s in self.tip_sensors[arm]], 1)
        idx = target.clamp(min=0)[:, None, None, None].expand(-1, 4, 1, 3)
        out = f.gather(2, idx)[:, :, 0]
        return torch.where((target >= 0)[:, None, None], out, torch.zeros_like(out))

    def tip_force_norms(self, arm: str) -> torch.Tensor:
        """(N, K) summed fingertip contact force magnitude of `arm` on each object."""
        n = self.num_envs
        return torch.stack(
            [s.data.normal_force_matrix_w.torch.reshape(n, -1, 3).norm(dim=-1) for s in self.tip_sensors[arm]], 0
        ).sum(0)

    def arm_contact_force(self) -> torch.Tensor:
        m = [
            s.data.normal_force_matrix_w.torch.reshape(self.num_envs, -1, 3).norm(dim=-1).amax(1)
            for s in self.arm_sensors
        ]
        return torch.stack(m, 1).amax(1)
