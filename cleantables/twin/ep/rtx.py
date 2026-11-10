"""Shared Isaac RTX scene pieces for the episode twins: cameras, lights and the wrist camera mount. Import only inside or right before
launch_simulation; there are no Isaac imports at module level. Used by the twin renderers (cleantables.twin.dr_render,
cleantables.features.paired_render, cleantables.features.composite_render)."""
from __future__ import annotations

import os

import numpy as np

# Wrist camera rigidly mounted on palm_link (pose in the palm frame). Palm frame of the Kuka-Allegro, measured on a Newton rollout:
# fingers extend along +x (fingertips x ~ 0.13-0.14 m), the palm faces +z (thumb tip at z ~ +0.06,
# the grasped object sits at ~(0.08, -0.01, 0.07)), index finger at -y. Camera: 8 cm behind the palm origin (wrist), 9 cm out on the index
# side and 4 cm on the back side, aimed at the grasp point between the fingertips (along the approach / grasp direction); image 'up' =
# the back of the hand. Chosen from 4 rendered candidate mounts as the one with fingertips and grasped object in view.
WRIST = dict(pos=[-0.08, -0.09, -0.04], look=[0.08, -0.01, 0.07], up=[0.0, 0.0, -1.0])
WR = 320


def rend():
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg
    return IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
        antialiasing_mode="DLAA", enable_dl_denoiser=True, enable_dlssg=False, samples_per_pixel=4, enable_shadows=True,
        enable_ambient_occlusion=True, enable_reflections=False))


def add_cameras(cfg, S, root, wrist_names=("a", "b"), extra_wrist=()):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import CameraCfg
    sdir = os.path.join(root, "data/twin_table", S["episode"])
    cfg.scene.background = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Background", spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(sdir, S["background"]["usd"])))
    for name, c in (("main", S["cameras"]["main"]), ("overview", S["cameras"]["overview"])):
        Kc = np.array(c["K"]); w, h = c["wh"]
        setattr(cfg.scene, f"cam_{name}", CameraCfg(prim_path="{ENV_REGEX_NS}/Cam_" + name, update_period=0.0, data_types=["rgb"], width=w, height=h,
                                                    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=Kc.reshape(-1).tolist(), width=w, height=h,
                                                                                                          clipping_range=(0.02, 30.0)), renderer_cfg=rend()))
    Kw = [WR * 0.42, 0, WR / 2, 0, WR * 0.42, WR / 2, 0, 0, 1]          # ~100 deg FOV
    for a in list(wrist_names) + list(extra_wrist):
        setattr(cfg.scene, f"cam_wrist_{a}", CameraCfg(prim_path="{ENV_REGEX_NS}/Cam_wrist_" + a, update_period=0.0, data_types=["rgb"], width=WR, height=WR,
                                                       spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=Kw, width=WR, height=WR,
                                                                                                             clipping_range=(0.01, 10.0)), renderer_cfg=rend()))


def T_to_pose(Tm):
    from scipy.spatial.transform import Rotation as R
    return Tm[:3, 3].tolist(), R.from_matrix(Tm[:3, :3]).as_quat().tolist()        # xyzw, OpenCV ('ros') axes


def wrist_pose(robot, palm_bid, mount=None):
    """World pose (pos, quat xyzw, OpenCV axes) of a wrist camera rigidly mounted on palm_link."""
    from scipy.spatial.transform import Rotation as Rot
    mount = mount or WRIST
    pp = robot.data.body_pos_w.torch[0, palm_bid].cpu().numpy(); pq = robot.data.body_quat_w.torch[0, palm_bid].cpu().numpy()
    Rp = Rot.from_quat(pq).as_matrix()
    z = (np.array(mount["look"], float) - np.array(mount["pos"], float)) if "look" in mount else np.array(mount["z"], float)
    z /= np.linalg.norm(z)
    if "up" in mount:                         # image 'up' direction in the palm frame
        yv = -np.array(mount["up"], float); yv -= yv.dot(z) * z; y = yv / np.linalg.norm(yv)
        x = np.cross(y, z)
    else:
        x = np.array(mount["x"], float); x -= x.dot(z) * z; x /= np.linalg.norm(x)
        y = np.cross(z, x)
    Rc = Rp @ np.stack([x, y, z], 1)
    return (pp + Rp @ np.array(mount["pos"])).tolist(), Rot.from_matrix(Rc).as_quat().tolist()


def set_static_cams(env, S, dev):
    import torch
    org = env.scene.env_origins
    for name in ("main", "overview"):
        p, q = T_to_pose(np.array(S["cameras"][name]["T_cam_to_table"]))
        n = org.shape[0]                                       # every env's camera, not only env 0's (otherwise the others stay at the origin)
        env.scene[f"cam_{name}"].set_world_poses(torch.tensor([p], device=dev, dtype=torch.float32) + org,
                                                torch.tensor([q], device=dev, dtype=torch.float32).repeat(n, 1), convention="ros")
        try:
            print(f"CAMPOSE {name}: env origins {org[:3].tolist()} -> camera pos_w {env.scene[f'cam_{name}'].data.pos_w[:3].tolist()}", flush=True)
        except Exception as e:
            print("CAMPOSE print failed", e, flush=True)


def set_wrist(env, name, robot, palm_bid, dev, mount=None):
    import torch
    p, q = wrist_pose(robot, palm_bid, mount)
    env.scene[f"cam_wrist_{name}"].set_world_poses(torch.tensor([p], device=dev, dtype=torch.float32), torch.tensor([q], device=dev, dtype=torch.float32),
                                                   convention="ros")


def hide_ground():
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/ground")).MakeInvisible()


def grab(env, name):
    import cv2
    im = env.scene[f"cam_{name}"].data.output["rgb"][0].cpu().numpy()[..., :3]
    return cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
