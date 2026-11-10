"""Observation builder for the published Isaac-Lift-Franka actor (rsl_rl, 1250-D input, 9-D action = 7 arm joints + 2 finger joints).
Same structure as lift_policy.ArmObs (Kuka-Allegro, 1595-D); only the term sizes differ. Actor input = history 5 (oldest first, contiguous per term) of
  policy     : object_quat_b (4) | command (7: goal pos in base frame + quat) | last raw action (9)            -> 5 x 20 = 100
  proprio    : joint_pos (9) | joint_vel (9) | hand_tips_state_b (2 bodies x [pos 3 + quat 4], base frame, clip +-2) | finger->object contact forces (2 x 3, base frame, clip +-20)  -> 5 x 38 = 190
  perception : 64 object surface points in base frame (clip +-2)                                                   -> 5 x 192 = 960
Body order of the hand tips = articulation order (panda_leftfinger, panda_rightfinger); contact order = the sensor list (rightfinger, leftfinger).
Checked numerically against the observations of the stock Isaac Lab task env."""
import torch

from cleantables.robots.lift_policy import ArmObs, H, quat_apply, quat_apply_inv, quat_conj, quat_mul, to_base  # noqa: F401

F_POLICY_TERMS = (4, 7, 9)
F_PROPRIO_TERMS = (9, 9, 14, 6)
F_PERCEPTION_TERMS = (192,)
F_OBS_DIM = H * (sum(F_POLICY_TERMS) + sum(F_PROPRIO_TERMS) + sum(F_PERCEPTION_TERMS))
assert F_OBS_DIM == 1250
NA = 9


class FrankaArmObs(ArmObs):
    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.hist = [torch.zeros(num_envs, H, d, device=device) for d in F_POLICY_TERMS + F_PROPRIO_TERMS + F_PERCEPTION_TERMS]
        self.last_action = torch.zeros(num_envs, NA, device=device)

    def frame(self, root_pos, root_quat, joint_pos, joint_vel, hand_pos_w, hand_quat_w, obj_pos_w, obj_quat_w, obj_pts_local, tip_force_w, goal_b):
        """hand (N,2,3)/(N,2,4) in order (leftfinger, rightfinger); tip_force_w (N,2,3) in order (rightfinger, leftfinger)."""
        N = root_pos.shape[0]
        rq_inv = quat_conj(root_quat)
        obj_quat_b = quat_mul(rq_inv, obj_quat_w)
        cmd = torch.cat([goal_b, self.cmd_quat], -1)
        hp = to_base(root_pos[:, None], root_quat[:, None], hand_pos_w)
        hq = quat_mul(rq_inv[:, None].expand(-1, hand_quat_w.shape[1], -1), hand_quat_w)
        hand = torch.cat([hp, hq], -1).reshape(N, -1).clamp(-2.0, 2.0)
        contact = quat_apply_inv(root_quat[:, None].expand(-1, tip_force_w.shape[1], -1), tip_force_w).reshape(N, -1).clamp(-20.0, 20.0)
        pts_w = quat_apply(obj_quat_w[:, None].expand(-1, obj_pts_local.shape[1], -1), obj_pts_local) + obj_pos_w[:, None]
        pts_b = to_base(root_pos[:, None], root_quat[:, None], pts_w).reshape(N, -1).clamp(-2.0, 2.0)
        return [obj_quat_b, cmd, self.last_action.clone(), joint_pos, joint_vel, hand, contact, pts_b]
