"""NVIDIA's Isaac-Lift-KukaAllegro policy (rsl_rl checkpoint, weights unchanged) and its 1595-D observation.

The observation is rebuilt here for any (arm, target object) pair, so one copy of the policy can run per arm and the
target can change during an episode. Isaac Lab ObservationManager layout, 5-step history per term, oldest first:

    policy      object quat in base frame (4) | command (7: goal in base frame + quat) | last action (23)
    proprio     joint pos (23) | joint vel (23) | palm + 4 tip poses in base frame (35, clip 2) |
                tip contact forces (12, clip 20)
    perception  64 object surface points in base frame (192, clip 2)

After a reset the history is back-filled with the first frame, as the task's circular buffers do.
"""

from __future__ import annotations

import numpy as np
import torch

from cleantables.paths import CHECKPOINTS

from .geometry import quat_apply, quat_apply_inv, quat_conj, quat_from_euler_xyz, quat_mul, to_frame

PRIOR_CHECKPOINT = CHECKPOINTS / "kuka_allegro_lift_physx.pt"
HISTORY = 5
POLICY_TERMS = (4, 7, 23)
PROPRIO_TERMS = (23, 23, 35, 12)
PERCEPTION_TERMS = (192,)
OBS_DIM = HISTORY * (sum(POLICY_TERMS) + sum(PROPRIO_TERMS) + sum(PERCEPTION_TERMS))
NUM_JOINTS = 23
HIDDEN = (512, 256, 128)
NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization
ACTION_SCALE = 0.1  # joint position target = q + ACTION_SCALE * action, as in the lift task
CLOUD_POINTS = 64
POSE_CLIP = 2.0
FORCE_CLIP = 20.0

# The checkpoint was trained in PhysX, whose articulation orders the finger joints breadth-first.
FINGERS = ("index", "middle", "ring", "thumb")
JOINT_ORDER = [f"iiwa7_joint_{i}" for i in range(1, 8)] + [f"{f}_joint_{i}" for i in range(4) for f in FINGERS]


def load_prior_weights(path=PRIOR_CHECKPOINT) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]


class LiftActor(torch.nn.Module):
    """Observation normaliser + MLP (1595-512-256-128-23, ELU); the deterministic action is the Gaussian mean."""

    def __init__(self, weights: dict, device):
        super().__init__()
        dims = [weights["mlp.0.weight"].shape[1], *HIDDEN, weights["mlp.6.weight"].shape[0]]
        layers = []
        for i in range(4):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < 3:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers)
        self.mlp.load_state_dict({k[4:]: v for k, v in weights.items() if k.startswith("mlp.")})
        self.register_buffer("mean", weights["obs_normalizer._mean"].float())
        self.register_buffer("std", weights["obs_normalizer._std"].float())
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / (self.std + NORMALIZER_EPS))


def farthest_point_sampling(pts: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    n = len(pts)
    idx = np.zeros(k, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx[0] = rng.integers(n)
    d = np.full(n, np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((pts - pts[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(d.argmax())
    return idx


def sample_point_cloud(mesh_path: str, num_points: int = CLOUD_POINTS, seed: int = 0) -> np.ndarray:
    """Area-weighted surface samples thinned by farthest-point sampling, in the object's frame (as the lift task)."""
    import trimesh
    from trimesh.sample import sample_surface

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = sample_surface(mesh, num_points * 2, seed=seed)
    return pts[farthest_point_sampling(pts, num_points, seed)].astype(np.float32)


class LiftObservation:
    """History buffers of the lift observation for one arm across all envs."""

    def __init__(self, num_envs: int, device):
        self.n, self.device = num_envs, device
        self.hist = [
            torch.zeros(num_envs, HISTORY, d, device=device) for d in POLICY_TERMS + PROPRIO_TERMS + PERCEPTION_TERMS
        ]
        self.need_fill = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.last_action = torch.zeros(num_envs, NUM_JOINTS, device=device)
        self.cmd_quat = torch.zeros(num_envs, 4, device=device)
        self.cmd_quat[:, 3] = 1.0

    def reset(self, env_mask: torch.Tensor) -> None:
        """New target or episode: back-fill on the next frame, clear the last action, redraw the command orientation."""
        self.need_fill |= env_mask
        self.last_action[env_mask] = 0.0
        n = int(env_mask.sum())
        if n:
            # the lift task samples a random goal orientation the policy learned to ignore; keep the same distribution
            roll = (torch.rand(n, device=self.device) * 2 - 1) * 3.14
            pitch = (torch.rand(n, device=self.device) * 2 - 1) * 3.14
            self.cmd_quat[env_mask] = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))

    def frame(
        self,
        root_pos,
        root_quat,
        joint_pos,
        joint_vel,
        hand_pos_w,
        hand_quat_w,
        obj_pos_w,
        obj_quat_w,
        obj_points,
        tip_force_w,
        goal_b,
    ) -> list[torch.Tensor]:
        """One time step of every term.

        Shapes: root (N,3)/(N,4), joints (N,23), hand (N,5,3)/(N,5,4), object (N,3)/(N,4), object points in its own
        frame (N,64,3), tip forces in world frame (N,4,3), goal in base frame (N,3).
        """
        n = root_pos.shape[0]
        rq_inv = quat_conj(root_quat)
        obj_quat_b = quat_mul(rq_inv, obj_quat_w)
        cmd = torch.cat([goal_b, self.cmd_quat], -1)
        hp = to_frame(root_pos[:, None], root_quat[:, None], hand_pos_w)
        hq = quat_mul(rq_inv[:, None].expand(-1, hand_quat_w.shape[1], -1), hand_quat_w)
        hand = torch.cat([hp, hq], -1).reshape(n, -1).clamp(-POSE_CLIP, POSE_CLIP)
        contact = (
            quat_apply_inv(root_quat[:, None].expand(-1, 4, -1), tip_force_w)
            .reshape(n, -1)
            .clamp(-FORCE_CLIP, FORCE_CLIP)
        )
        pts_w = quat_apply(obj_quat_w[:, None].expand(-1, obj_points.shape[1], -1), obj_points) + obj_pos_w[:, None]
        pts_b = to_frame(root_pos[:, None], root_quat[:, None], pts_w).reshape(n, -1).clamp(-POSE_CLIP, POSE_CLIP)
        return [obj_quat_b, cmd, self.last_action.clone(), joint_pos, joint_vel, hand, contact, pts_b]

    def push(self, terms: list[torch.Tensor]) -> torch.Tensor:
        """Append one frame and return the flat 1595-D observation."""
        fill = self.need_fill
        for buf, x in zip(self.hist, terms, strict=True):
            buf[:, :-1] = buf[:, 1:].clone()
            buf[:, -1] = x
            if fill.any():
                buf[fill] = x[fill, None].expand(-1, HISTORY, -1)
        self.need_fill = torch.zeros_like(self.need_fill)
        return torch.cat([b.reshape(self.n, -1) for b in self.hist], -1)

    def set_action(self, action: torch.Tensor) -> None:
        self.last_action = action.clone()
