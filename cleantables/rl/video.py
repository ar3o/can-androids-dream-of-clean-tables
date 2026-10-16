"""RTX video of env 0: phone camera and overview, with each arm's imagined carry path and current goal drawn on."""

from __future__ import annotations

import numpy as np

from .scene import ARMS

PATH_COLOURS = ((0, 140, 255), (255, 200, 0))  # BGR, arm a / arm b
PANEL = (800, 450)
FPS = 15


def _renderer():
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


def add_video_cameras(sim_cfg, scene: dict) -> None:
    """Background plate and the main / overview cameras of the twin (before the app is launched)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import CameraCfg

    sim_cfg.scene.background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Background",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{scene['dir']}/{scene['background']['usd']}"),
    )
    for name in ("main", "overview"):
        c = scene["cameras"][name]
        w, h = c["wh"]
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=np.array(c["K"]).reshape(-1).tolist(), width=w, height=h, clipping_range=(0.02, 30.0)
        )
        setattr(
            sim_cfg.scene,
            f"cam_{name}",
            CameraCfg(
                prim_path="{ENV_REGEX_NS}/Cam_" + name,
                update_period=0.0,
                data_types=["rgb"],
                width=w,
                height=h,
                spawn=spawn,
                renderer_cfg=_renderer(),
            ),
        )


class VideoRecorder:
    def __init__(self, env):
        import omni.usd
        import torch
        from pxr import UsdGeom
        from scipy.spatial.transform import Rotation

        self.env, self.frames = env, []
        scene = env.scene_json
        UsdGeom.Imageable(omni.usd.get_context().get_stage().GetPrimAtPath("/World/ground")).MakeInvisible()
        org = env.sim.scene.env_origins
        for name in ("main", "overview"):
            t = np.array(scene["cameras"][name]["T_cam_to_table"])
            q = Rotation.from_matrix(t[:3, :3]).as_quat().tolist()  # xyzw, OpenCV axes
            env.sim.scene[f"cam_{name}"].set_world_poses(
                torch.tensor([t[:3, 3].tolist()], device=env.device, dtype=torch.float32) + org[:1],
                torch.tensor([q], device=env.device, dtype=torch.float32),
                convention="ros",
            )
        main = scene["cameras"]["main"]
        self.k = np.array(main["K"], np.float64).reshape(3, 3)
        self.table_to_cam = np.linalg.inv(np.array(main["T_cam_to_table"], np.float64))
        self.width = main["wh"][0]

    def _grab(self, name: str) -> np.ndarray:
        import cv2

        im = self.env.sim.scene[f"cam_{name}"].data.output["rgb"][0].cpu().numpy()[..., :3]
        return cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _project(self, pts: np.ndarray, scale: float) -> np.ndarray:
        x = self.table_to_cam[:3, :3] @ np.asarray(pts, np.float64).T + self.table_to_cam[:3, 3:4]
        return ((self.k @ x)[:2] / np.maximum(x[2:3], 1e-6)).T * scale

    def frame(self, label: str, step: int) -> None:
        import cv2

        env, num_items = self.env, self.env.num_items
        env.sim.sim.render()
        main = self._grab("main")
        scale = main.shape[1] / self.width
        tg = env.teacher
        for ai, a in enumerate(ARMS):
            if bool(tg.ok[0, ai]) and int(env.assigned[0, ai]) < num_items:
                uv = self._project(tg.path[0, ai].cpu().numpy(), scale).astype(int)
                cv2.polylines(main, [uv.reshape(-1, 1, 2)], False, PATH_COLOURS[ai], 2, cv2.LINE_AA)
                g = self._project(env.goal[a][0:1].cpu().numpy(), scale)[0].astype(int)
                cv2.circle(main, (int(g[0]), int(g[1])), 7, PATH_COLOURS[ai], -1)
        im = np.hstack([cv2.resize(main, PANEL), cv2.resize(self._grab("overview"), PANEL)])
        asg = [env.item_names[int(x)] if int(x) < num_items else "-" for x in env.assigned[0]]
        lines = [
            f"{env.scene_json['episode']}  '{env.scene_json.get('instruction', '')}'   {label}",
            f"t={step / 30:4.1f}s   put down {int(env.placed[0].sum())}/{num_items}   "
            f"arm a -> {asg[0]}   arm b -> {asg[1]}"
            "   (left: phone camera, right: overview)",
        ]
        bar = np.zeros((26 * len(lines) + 10, im.shape[1], 3), np.uint8)
        for i, text in enumerate(lines):
            cv2.putText(bar, text, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self.frames.append(np.vstack([bar, im]))

    def write(self, path: str) -> None:
        import imageio.v2 as imageio

        if self.frames:
            imageio.mimwrite(path, [f[..., ::-1] for f in self.frames], fps=FPS, quality=7, macro_block_size=None)
            print(f"video {path}: {len(self.frames)} frames", flush=True)
        self.frames = []
