"""Isaac Lab scene of one twin for rendering: table, objects as rigid bodies, two Kuka-Allegro arms on pedestals,
lights.

The env is only a container for PhysX and the RTX cameras: the arms hold their joint positions and nothing is learned.
Import after the simulation app has started (it imports Isaac Lab at module level).
"""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

ARMS = ("a", "b")
FLOOR_Z = -0.76  # floor below the table top
KINEMATIC_CATEGORIES = {"container/fruit_bowl", "container/toolbox"}  # containers stay put


def load_scene(twin_folder: Path) -> dict:
    return json.loads((Path(twin_folder) / "scene.json").read_text())


def build_scene_cfg(scene: dict, twin_folder: Path, dome: float, key: float) -> InteractiveSceneCfg:
    tb = scene["table"]
    twin_folder = Path(twin_folder)
    table_drop = -scene["poses"]["a"][0][2]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(color=(0.55, 0.55, 0.55)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, FLOOR_Z)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=dome, color=(1.0, 0.95, 0.88))
        )
        key_light = AssetBaseCfg(
            prim_path="/World/key", spawn=sim_utils.DistantLightCfg(intensity=key, angle=25.0, color=(1.0, 0.93, 0.85))
        )
        table_col = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TableCol",
            spawn=sim_utils.CuboidCfg(
                size=(tb["x"][1] - tb["x"][0], tb["y"][1] - tb["y"][0], 0.04),
                visible=False,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=((tb["x"][0] + tb["x"][1]) / 2, (tb["y"][0] + tb["y"][1]) / 2, -0.02)
            ),
        )
        table_vis = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TableVis", spawn=sim_utils.UsdFileCfg(usd_path=str(twin_folder / "table.usda"))
        )

    cfg = SceneCfg(num_envs=1, env_spacing=3.5, replicate_physics=True)
    pedestal_h = -FLOOR_Z - table_drop - 0.02
    for arm in ARMS:
        pos, rot = scene["poses"][arm]
        r = replace(KUKA_ALLEGRO_CFG, prim_path="{ENV_REGEX_NS}/Robot" + arm.upper())
        r.init_state = replace(r.init_state, pos=tuple(pos), rot=tuple(rot))
        setattr(cfg, f"robot_{arm}", r)
        setattr(
            cfg,
            f"ped_{arm}",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Ped" + arm.upper(),
                spawn=sim_utils.CuboidCfg(
                    size=(0.22, 0.22, pedestal_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(pos[0], pos[1], FLOOR_Z + pedestal_h / 2)),
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
            usd_path=str(twin_folder / o["usd"]),
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
    return cfg


@configclass
class TwinSceneEnvCfg(DirectRLEnvCfg):
    episode_length_s = 1.0e5
    decimation = 8  # 240 Hz physics, 30 Hz control (the Lift policy's rate)
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


class TwinSceneEnv(DirectRLEnv):
    """arms hold their current joint positions; objects are free rigid bodies (containers kinematic)."""

    cfg: TwinSceneEnvCfg

    def __init__(self, cfg: TwinSceneEnvCfg, scene: dict, **kw):
        self.twin = scene
        super().__init__(cfg, None, **kw)
        self.robots = {a: self.scene[f"robot_{a}"] for a in ARMS}
        self.objs = [self.scene[f"obj_{k}"] for k in range(len(scene["objects"]))]
        self.palm_body = self.robots["a"].find_bodies("palm_link")[0][0]

    def hold_default_pose(self) -> None:
        """the Lift task's default arm pose (a reset alone leaves q = 0)."""
        ids = torch.arange(self.num_envs, device=self.device)
        for r in self.robots.values():
            q = r.data.default_joint_pos.torch.clone()
            r.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)

    def place_object(self, k: int, xyz, yaw: float = 0.0) -> None:
        """teleport object k (table frame, bottom centre) and zero its velocity."""
        import math

        p = torch.zeros(1, 7, device=self.device)
        p[0, :3] = torch.as_tensor(xyz, dtype=torch.float32, device=self.device) + self.scene.env_origins[0]
        p[0, 5], p[0, 6] = math.sin(yaw / 2), math.cos(yaw / 2)
        ids = torch.arange(1, device=self.device)
        self.objs[k].write_root_pose_to_sim_index(root_pose=p, env_ids=ids)
        self.objs[k].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.device), env_ids=ids)

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        for r in self.robots.values():
            r.set_joint_position_target_index(target=r.data.joint_pos.torch)

    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z


def make_env_cfg(scene: dict, twin_folder: Path, dome: float, key: float) -> TwinSceneEnvCfg:
    cfg = TwinSceneEnvCfg()
    cfg.scene = build_scene_cfg(scene, twin_folder, dome, key)
    return cfg
