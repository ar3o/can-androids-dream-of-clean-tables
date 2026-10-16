"""Isaac RTX cameras for the twins: the real phone camera, a high overview and a wrist camera on each palm.

Isaac modules are imported inside the functions, so this file can be imported before the simulation app starts.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Wrist camera rigidly mounted on palm_link, in the palm frame. Kuka-Allegro palm frame: fingers along +x (tips at x ~
# 0.13 m), palm facing +z, index finger at -y. The camera sits 8 cm behind the palm origin, 9 cm out on the index side
# and 4 cm on the back, aimed at the grasp point between the fingertips; image 'up' = the back of the hand. Picked
# from 4 rendered candidates on a grasp rollout: fingertips and the grasped object in view.
WRIST = dict(pos=[-0.08, -0.09, -0.04], look=[0.08, -0.01, 0.07], up=[0.0, 0.0, -1.0])
WRIST_PX = 320
WRIST_FOCAL = 0.42  # focal length / image width: about 100 deg field of view


def renderer_cfg():
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg

    return IsaacRtxRendererCfg(
        global_settings=IsaacRtxRendererGlobalSettingsCfg(
            antialiasing_mode="DLAA",
            enable_dl_denoiser=True,
            enable_dlssg=False,
            samples_per_pixel=4,
            enable_shadows=True,
            enable_ambient_occlusion=True,
            enable_reflections=False,
        )
    )


def add_cameras(cfg, scene: dict, twin_folder: Path, wrist_names=("a", "b")) -> None:
    """background plate + main / overview cameras (+ wrist cameras) on an Isaac Lab scene config."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import CameraCfg

    cfg.scene.background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Background",
        spawn=sim_utils.UsdFileCfg(usd_path=str(Path(twin_folder) / scene["background"]["usd"])),
    )
    for name in ("main", "overview"):
        c = scene["cameras"][name]
        w, h = c["wh"]
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=np.array(c["K"]).reshape(-1).tolist(), width=w, height=h, clipping_range=(0.02, 30.0)
        )
        setattr(
            cfg.scene,
            f"cam_{name}",
            CameraCfg(
                prim_path="{ENV_REGEX_NS}/Cam_" + name,
                update_period=0.0,
                data_types=["rgb"],
                width=w,
                height=h,
                spawn=spawn,
                renderer_cfg=renderer_cfg(),
            ),
        )
    Kw = [WRIST_PX * WRIST_FOCAL, 0, WRIST_PX / 2, 0, WRIST_PX * WRIST_FOCAL, WRIST_PX / 2, 0, 0, 1]
    for a in wrist_names:
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=Kw, width=WRIST_PX, height=WRIST_PX, clipping_range=(0.01, 10.0)
        )
        setattr(
            cfg.scene,
            f"cam_wrist_{a}",
            CameraCfg(
                prim_path="{ENV_REGEX_NS}/Cam_wrist_" + a,
                update_period=0.0,
                data_types=["rgb"],
                width=WRIST_PX,
                height=WRIST_PX,
                spawn=spawn,
                renderer_cfg=renderer_cfg(),
            ),
        )


def pose_from_matrix(Tm: np.ndarray) -> tuple[list, list]:
    """4x4 camera -> world (OpenCV axes) -> position, quaternion xyzw."""
    return Tm[:3, 3].tolist(), Rotation.from_matrix(Tm[:3, :3]).as_quat().tolist()


def wrist_pose(robot, palm_body: int, mount: dict | None = None) -> tuple[list, list]:
    """world pose (position, quaternion xyzw, OpenCV axes) of a camera rigidly mounted on palm_link (env 0)."""
    mount = mount or WRIST
    pp = robot.data.body_pos_w.torch[0, palm_body].cpu().numpy()
    Rp = Rotation.from_quat(robot.data.body_quat_w.torch[0, palm_body].cpu().numpy()).as_matrix()
    z = np.array(mount["look"], float) - np.array(mount["pos"], float)
    z /= np.linalg.norm(z)
    y = -np.array(mount["up"], float)
    y -= y.dot(z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    Rc = Rp @ np.stack([x, y, z], 1)
    return (pp + Rp @ np.array(mount["pos"])).tolist(), Rotation.from_matrix(Rc).as_quat().tolist()


def set_static_cams(env, scene: dict, device) -> None:
    import torch

    org = env.scene.env_origins
    for name in ("main", "overview"):
        p, q = pose_from_matrix(np.array(scene["cameras"][name]["T_cam_to_table"]))
        env.scene[f"cam_{name}"].set_world_poses(
            torch.tensor([p], device=device, dtype=torch.float32) + org[:1],
            torch.tensor([q], device=device, dtype=torch.float32),
            convention="ros",
        )


def set_wrist(env, name: str, robot, palm_body: int, device, mount: dict | None = None) -> None:
    import torch

    p, q = wrist_pose(robot, palm_body, mount)
    env.scene[f"cam_wrist_{name}"].set_world_poses(
        torch.tensor([p], device=device, dtype=torch.float32),
        torch.tensor([q], device=device, dtype=torch.float32),
        convention="ros",
    )


def set_invisible(prim_paths) -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for p in prim_paths:
        UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()


def hide_ground() -> None:
    """the background plate already shows the real floor."""
    set_invisible(["/World/ground"])


def grab(env, name: str) -> np.ndarray:
    """BGR image of camera `name` in env 0."""
    im = env.scene[f"cam_{name}"].data.output["rgb"][0].cpu().numpy()[..., :3]
    return cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
