"""The published Isaac-Lift-KukaAllegro actor (rsl_rl, unchanged weights) as a plain torch module, plus an observation builder
that reproduces the task's 1595-D actor input for any (robot, target object) pair, so one copy of the policy can run per arm and the
target object can be switched by an allocator.

Actor input (Isaac Lab ObservationManager, rsl_rl obs_groups actor = [policy, proprio, perception]; every group has history 5 and the
default "term" history order, i.e. each term's 5-step history is contiguous, oldest first):
  policy     : object_quat_b (4) | command (7: goal pos in robot base frame + quat xyzw) | last raw action (23)      -> 5 x 34  = 170
  proprio    : joint_pos (23) | joint_vel (23) | palm + 4 tip poses in base frame (5 x 7, clip +-2) | fingertip(link_3)->object
               contact forces in base frame (4 x 3, clip +-20)                                                       -> 5 x 93  = 465
  perception : 64 object surface points in base frame (clip +-2)                                                      -> 5 x 192 = 960
After a reset the history is back-filled with the first frame (CircularBuffer semantics); we do the same at every target switch.
"""
import math

import numpy as np
import torch

H = 5
POLICY_TERMS = (4, 7, 23)
PROPRIO_TERMS = (23, 23, 35, 12)
PERCEPTION_TERMS = (192,)
OBS_DIM = H * (sum(POLICY_TERMS) + sum(PROPRIO_TERMS) + sum(PERCEPTION_TERMS))
assert OBS_DIM == 1595


class LiftActor(torch.nn.Module):
    """obs_normalizer + MLP(1595-512-256-128-23, ELU); deterministic action = MLP output (the Gaussian mean)."""

    def __init__(self, ckpt_path, device):
        super().__init__()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["actor_state_dict"]
        dims = [sd["mlp.0.weight"].shape[1], 512, 256, 128, sd["mlp.6.weight"].shape[0]]
        layers = []
        for i in range(4):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < 3:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers)
        self.mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
        self.register_buffer("mean", sd["obs_normalizer._mean"].float())
        self.register_buffer("std", sd["obs_normalizer._std"].float())
        self.eps = 1e-2  # rsl_rl EmpiricalNormalization default
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, obs):
        return self.mlp((obs - self.mean) / (self.std + self.eps))


# ---------------- quaternion helpers (xyzw, Isaac Lab 3 convention) ----------------
def quat_conj(q):
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], -1)


def quat_apply(q, v):
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inv(q, v):
    return quat_apply(quat_conj(q), v)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy], -1)


def to_base(root_pos, root_quat, p_w):
    """world points (..., 3) -> base frame; root_pos/quat broadcast over the middle dims."""
    return quat_apply_inv(root_quat, p_w - root_pos)


# ---------------- point clouds (as isaaclab_tasks lift utils.sample_object_point_cloud) ----------------
def farthest_point_sampling(pts, k, seed=0):
    n = len(pts)
    idx = np.zeros(k, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx[0] = rng.integers(n)
    d = np.full(n, np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((pts - pts[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(d.argmax())
    return idx


def sample_point_cloud(mesh_path, num_points=64, seed=0):
    """2 x num_points area-weighted surface samples, FPS down to num_points, in the object root frame (mesh frame)."""
    import trimesh
    from trimesh.sample import sample_surface
    m = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = sample_surface(m, num_points * 2, seed=seed)
    return pts[farthest_point_sampling(pts, num_points, seed)].astype(np.float32)


class ArmObs:
    """History buffers + frame builder for one arm (all envs).  Feed it tensors; it does not touch the simulator."""

    def __init__(self, num_envs, device):
        self.N, self.dev = num_envs, device
        self.hist = [torch.zeros(num_envs, H, d, device=device) for d in POLICY_TERMS + PROPRIO_TERMS + PERCEPTION_TERMS]
        self.need_fill = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.last_action = torch.zeros(num_envs, 23, device=device)
        self.cmd_quat = torch.zeros(num_envs, 4, device=device)
        self.cmd_quat[:, 3] = 1.0

    def reset(self, env_mask):
        """New target / new episode: back-fill history on the next frame, zero last action, redraw the (ignored) command quat."""
        self.need_fill |= env_mask
        self.last_action[env_mask] = 0.0
        n = int(env_mask.sum())
        if n:
            r = (torch.rand(n, device=self.dev) * 2 - 1) * 3.14
            p = (torch.rand(n, device=self.dev) * 2 - 1) * 3.14
            self.cmd_quat[env_mask] = quat_from_euler_xyz(r, p, torch.zeros_like(r))

    def frame(self, root_pos, root_quat, joint_pos, joint_vel, hand_pos_w, hand_quat_w, obj_pos_w, obj_quat_w, obj_pts_local, tip_force_w, goal_b):
        """One time step of every term (no history).  Shapes: root (N,3)/(N,4); joints (N,23); hand (N,5,3)/(N,5,4);
        obj pos/quat (N,3)/(N,4); obj_pts_local (N,64,3); tip_force_w (N,4,3); goal_b (N,3)."""
        N = root_pos.shape[0]
        rq_inv = quat_conj(root_quat)
        obj_quat_b = quat_mul(rq_inv, obj_quat_w)
        cmd = torch.cat([goal_b, self.cmd_quat], -1)
        hp = to_base(root_pos[:, None], root_quat[:, None], hand_pos_w)
        hq = quat_mul(rq_inv[:, None].expand(-1, hand_quat_w.shape[1], -1), hand_quat_w)
        hand = torch.cat([hp, hq], -1).reshape(N, -1).clamp(-2.0, 2.0)
        contact = quat_apply_inv(root_quat[:, None].expand(-1, 4, -1), tip_force_w).reshape(N, -1).clamp(-20.0, 20.0)
        pts_w = quat_apply(obj_quat_w[:, None].expand(-1, obj_pts_local.shape[1], -1), obj_pts_local) + obj_pos_w[:, None]
        pts_b = to_base(root_pos[:, None], root_quat[:, None], pts_w).reshape(N, -1).clamp(-2.0, 2.0)
        return [obj_quat_b, cmd, self.last_action.clone(), joint_pos, joint_vel, hand, contact, pts_b]

    def push(self, terms):
        """Append one frame to the history (back-filling envs flagged by reset) and return the flat 1595-D actor input."""
        fill = self.need_fill
        for buf, x in zip(self.hist, terms):
            buf[:, :-1] = buf[:, 1:].clone()
            buf[:, -1] = x
            if fill.any():
                buf[fill] = x[fill, None].expand(-1, H, -1)
        self.need_fill = torch.zeros_like(self.need_fill)
        return torch.cat([b.reshape(self.N, -1) for b in self.hist], -1)

    def set_action(self, a):
        self.last_action = a.clone()
