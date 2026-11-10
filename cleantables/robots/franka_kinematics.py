"""Batched Franka Panda kinematics + sphere collision model in torch (everything follows q's device / dtype; no Isaac, no cuRobo).

Frames.  BASE frame = the robot's panda_link0 frame (z up, x = the arm's forward direction).  WORLD = base pose applied (the training env
frame = the twin-template frame: table top z = 0, Franka bases at z = -0.255 yawed pi so they reach towards -x; bases_from_template()
builds them).  Every function that takes a `base` maps base -> world; base=None means the world is the base frame.  Poses are (..., 4, 4)
homogeneous matrices or (..., 7) = [x, y, z, qx, qy, qz, qw] (xyzw quaternions, as in Isaac Lab and the rest of the package).
Units: m, rad, s.

Kinematics.  Standard Franka DH (Craig / modified convention, identical to franka_description's URDF joint origins), flange (panda_link8) 0.107 along
link7 z, hand = flange rotated -pi/4 about z, TCP = fingertip centre 0.1034 along the hand z (= Isaac Lab Lift's end-effector offset), fingers 0.0584
along hand z, each slid +-w (w = finger joint value, 0 closed .. 0.04 open) along hand y.  fk() frame order: FRAMES below (13 frames, link0 = identity).

Commands this module checks (open set; each produces a joint path (..., T, 7) -> time_parameterise -> times, so all share one clock in check_two_arms):
    grasp X at G     approach(q, G)            joint move to pre-grasp, straight Cartesian descent to G
    follow P         ik_path(P, q) + check_path   carry along a (spline) TCP waypoint path, warm-started, flips flagged
    release at R     ik_path / move_to to R   (opening the gripper is not modelled here: finger width is an argument of the sphere functions)
    move empty to Q  move_to(q, Q)            free-gripper repositioning / getting out of the other arm's way
    wait             join([... , wait_s, ...]) hold the last configuration for wait_s seconds (or a later start time)
Start posture: Q_HOME (the sim's FRANKA_PANDA_LIFT_CFG default), home_tcp(bases) = each arm's TCP world pose there.
"""
import math

import torch

# ---------------------------------------------------------------- constants (base frame, m / rad)
DH_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
DH_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
DH_ALPHA = [0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2]
D_FLANGE, HAND_YAW, TCP_OFF, FINGER_Z = 0.107, -math.pi / 4, 0.1034, 0.0584
Q_LO = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
Q_HI = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
V_MAX = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]                  # rad/s
Q_NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
# the sim's start posture: FRANKA_PANDA_LIFT_CFG (isaaclab_tasks core/lift/config/franka/franka_env_cfg.py) = clone(FRANKA_PANDA_CFG), whose
# init_state is FRANKA_PANDA_LEGACY_CFG's joint_pos (third_party/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py; copied, not
# imported: isaaclab needs the sim app).  Fingers 0.04 (open).  Note q6 = 3.037 is 0.10 rad from the wrist singularity at q6 = pi.
Q_HOME = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
FINGER_HOME = 0.04
DROP = 0.255                # m, base below the table top
ENV_YAW = math.pi           # every training-env base is turned 180 deg about z (init rot xyzw (0, 0, 1, 0)): the arms reach towards env -x
FRAMES = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "flange", "hand", "tcp", "lfinger", "rfinger"]
F_IDX = {n: i for i, n in enumerate(FRAMES)}
TCP = F_IDX["tcp"]

# ---------------------------------------------------------------- sphere set (link frame: x, y, z, radius)
# Provenance: not the published cuRobo franka.yml (an approximate copy of that set covered only 30-85% of the collision-mesh surface per
# link). Fitted here to franka_description's collision meshes (meshes/collision/*.stl from the Isaac Sim URDF importer extension): per link,
# k-means init on interior points, then minimise sum r^3 subject to covering 4000 surface samples + all vertices, then every radius grown until each
# sample is inside a sphere, +1 mm and rounded up to the mm.  => conservative: covers the collision mesh (to sampling density), protrudes beyond it by
# ~10-13 mm on average (p90 ~16-25 mm, max ~20-35 mm at box corners / cylinder caps; link0 ~22 mm mean).  The collision meshes are themselves slightly
# bigger than the visual meshes.  Right finger uses the left finger's spheres (its frame below includes the pi yaw of the URDF's mesh origin).
SPHERES = {
    'link0': [(0.0474, -0.0003, 0.027, 0.063), (-0.0156, 0.0603, 0.033, 0.066), (-0.1118, -0.0445, 0.0304, 0.056), (0.0268, -0.0319, 0.111, 0.056),
              (-0.0474, -0.0012, 0.1054, 0.072), (-0.0219, -0.0505, 0.0317, 0.069), (-0.1042, 0.0255, 0.0241, 0.068), (0.0218, 0.0377, 0.1103, 0.052)],
    'link1': [(0.0017, -0.0402, -0.1196, 0.058), (-0.0021, -0.0183, -0.0036, 0.065), (0.0048, -0.0695, -0.0048, 0.069), (-0.035, 0.0078, -0.0803, 0.053),
              (-0.0038, -0.0537, -0.0657, 0.066), (0.0338, 0.0042, -0.0769, 0.056), (-0.0004, 0.0009, -0.1542, 0.070)],
    'link2': [(0.001, -0.1126, 0.0407, 0.058), (0.0001, -0.0005, 0.0775, 0.061), (-0.0001, -0.1547, 0.0003, 0.071), (0.002, -0.006, 0.0266, 0.068),
              (-0.0233, -0.081, -0.0078, 0.060), (-0.0008, -0.0588, 0.0573, 0.065), (0.0428, -0.0778, -0.0064, 0.047)],
    'link3': [(0.0811, 0.0597, -0.0059, 0.064), (0.012, 0.0002, -0.028, 0.062), (0.0198, 0.0534, 0.0056, 0.032), (0.082, 0.0168, 0.0034, 0.062),
              (0.0013, 0.0002, -0.0892, 0.068), (0.0808, -0.0019, -0.0683, 0.047), (0.0364, 0.0476, -0.0621, 0.059)],
    'link4': [(0.0027, 0.0015, 0.0649, 0.059), (-0.0321, 0.0638, 0.0526, 0.060), (-0.0067, 0.0031, 0.0214, 0.071), (-0.0789, 0.0356, 0.0032, 0.062),
              (-0.0166, 0.0779, -0.0077, 0.047), (-0.0808, 0.0947, 0.0003, 0.067), (-0.0561, 0.0107, 0.0671, 0.033)],
    'link5': [(0.0012, 0.0016, -0.2271, 0.068), (0.0015, 0.0726, -0.0539, 0.066), (-0.0007, 0.084, 0.0019, 0.059), (-0.036, 0.0091, -0.076, 0.044),
              (0.0026, 0.0525, -0.1137, 0.062), (-0.0016, 0.0452, -0.179, 0.062), (0.0005, 0.0097, -0.1505, 0.068), (0.0327, 0.0025, -0.072, 0.042),
              (0.0024, 0.0345, -0.0083, 0.062)],
    'link6': [(0.0866, 0.0438, -0.0011, 0.052), (0.0025, -0.0013, 0.0074, 0.058), (0.0451, -0.0239, 0.0366, 0.037), (0.0406, 0.0336, 0.0109, 0.055),
              (0.0398, -0.0406, -0.0172, 0.031), (0.0866, -0.0146, 0.0002, 0.060)],
    'link7': [(-0.0105, -0.0289, 0.0746, 0.040), (0.0287, 0.0477, 0.0805, 0.040), (0.0298, -0.0065, 0.0785, 0.043), (0.0591, 0.0356, 0.0832, 0.029),
              (-0.0155, 0.0118, 0.0747, 0.045)],
    'hand': [(-0.0027, 0.0733, 0.0444, 0.038), (-0.0013, -0.0668, 0.0023, 0.041), (0.0008, -0.0352, 0.0478, 0.038), (-0.0019, -0.0096, 0.0067, 0.043),
             (-0.0009, 0.0783, 0.0073, 0.035), (0.0004, 0.019, 0.052, 0.037), (0.0002, 0.0374, 0.0095, 0.040), (0.002, -0.0826, 0.043, 0.033)],
    'lfinger': [(-0.0018, 0.0144, 0.0117, 0.023), (0.0027, 0.0108, 0.04, 0.023)],
}
SPHERES['rfinger'] = SPHERES['lfinger']
S_NAMES = [n for n in FRAMES if n in SPHERES]                            # body order of the flat sphere list
S_BODY = [n for n in S_NAMES for _ in SPHERES[n]]                         # (S,) body name of each sphere
S_FRAME = [F_IDX[n] for n in S_BODY]                                      # (S,) fk frame index of each sphere
S_LOCAL = [s for n in S_NAMES for s in SPHERES[n]]                        # (S, 4)
N_SPH = len(S_LOCAL)
# self-collision pairs not checked (adjacent / always-touching bodies; the cuRobo franka.yml ignore list, + finger-finger)
SELF_IGNORE = {"link0": ["link1", "link2"], "link1": ["link2", "link3", "link4"], "link2": ["link3", "link4"], "link3": ["link4", "link6"],
               "link4": ["link5", "link6", "link7"], "link5": ["link6", "link7", "hand", "lfinger", "rfinger"],
               "link6": ["link7", "hand", "lfinger", "rfinger"], "link7": ["hand", "lfinger", "rfinger"], "hand": ["lfinger", "rfinger"],
               "lfinger": ["rfinger"]}
TABLE_SKIP = ("link0", "link1")             # beside the table, below its top (base 0.255 m under it), only spin about the vertical -> not table-checked
CONTACT_SKIP = TABLE_SKIP + ("lfinger", "rfinger")                        # final grasp descent: the fingertips are meant to reach the table


def _c(x, ref):
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


# ---------------------------------------------------------------- pose helpers
def quat_to_R(qt):
    """(..., 4) xyzw -> (..., 3, 3)"""
    x, y, z, w = (qt / qt.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(qt.shape[:-1] + (3, 3))


def R_to_quat(R):
    """(..., 3, 3) -> (..., 4) xyzw, w >= 0"""
    return _R_to_wxyz(R).roll(-1, -1)


def _R_to_wxyz(R):
    """(..., 3, 3) -> (..., 4) wxyz, w >= 0 (branch on the largest of w, x, y, z: stable everywhere); internal"""
    m = R
    t = torch.stack([1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                     1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    k = t.argmax(-1, keepdim=True)
    s = (2 * t.gather(-1, k).clamp_min(1e-12).sqrt())                     # 4 * |component k|
    d21, d02, d10 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    s21, s02, s10 = m[..., 2, 1] + m[..., 1, 2], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1]
    c = torch.stack([torch.stack([t[..., 0], d21, d02, d10], -1), torch.stack([d21, t[..., 1], s10, s02], -1),
                     torch.stack([d02, s10, t[..., 2], s21], -1), torch.stack([d10, s02, s21, t[..., 3]], -1)], -2)   # row k = 4 q q_k
    qt = c.gather(-2, k[..., None].expand(k.shape[:-1] + (1, 4))).squeeze(-2) / s
    return qt * torch.where(qt[..., :1] < 0, -1.0, 1.0)


def as_T(p):
    """(..., 7) [xyz, xyzw] or (..., 4, 4) -> (..., 4, 4)"""
    if p.shape[-1] == 4 and p.dim() >= 2 and p.shape[-2] == 4:
        return p
    T = torch.zeros(p.shape[:-1] + (4, 4), dtype=p.dtype, device=p.device)
    T[..., :3, :3] = quat_to_R(p[..., 3:7]); T[..., :3, 3] = p[..., :3]; T[..., 3, 3] = 1
    return T


def T_to_pose(T):
    """(..., 4, 4) -> (..., 7) [xyz, xyzw]"""
    return torch.cat([T[..., :3, 3], R_to_quat(T[..., :3, :3])], -1)


def base_T(x, y, z, yaw, dtype=torch.float32, device="cpu"):
    """robot base pose in the world from position + yaw about z (the twin-table mounts) -> (4, 4); broadcast tensors give (..., 4, 4)"""
    x, y, z, yaw = [torch.as_tensor(v, dtype=dtype, device=device) for v in (x, y, z, yaw)]
    x, y, z, yaw = torch.broadcast_tensors(x, y, z, yaw)
    T = torch.zeros(x.shape + (4, 4), dtype=dtype, device=device)
    c, s = torch.cos(yaw), torch.sin(yaw)
    T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1] = c, -s, s, c
    T[..., 2, 2] = T[..., 3, 3] = 1
    T[..., 0, 3], T[..., 1, 3], T[..., 2, 3] = x, y, z
    return T


def top_down(xyz, yaw=0.0):
    """(..., 3) TCP positions -> (..., 4, 4) poses with the gripper z axis pointing down (-z) and the finger axis (hand y) at `yaw` about z"""
    yaw = torch.as_tensor(yaw, dtype=xyz.dtype, device=xyz.device).expand(xyz.shape[:-1])
    c, s = torch.cos(yaw), torch.sin(yaw)
    T = torch.zeros(xyz.shape[:-1] + (4, 4), dtype=xyz.dtype, device=xyz.device)
    T[..., 0, 0], T[..., 1, 0] = -s, c              # x_tool = (-s, c, 0), y_tool = (c, s, 0) -> z_tool = x cross y = (0, 0, -1)
    T[..., 0, 1], T[..., 1, 1] = c, s
    T[..., 2, 2] = -1
    T[..., :3, 3] = xyz; T[..., 3, 3] = 1
    return T


def bases_from_template(t, drop=DROP, yaw=ENV_YAW):
    """Twin template (a dict from the "templates" list, key "bases": (A, 2) env-frame base xy, arm 0 first) or the bases array itself
    (..., A, 2) -> (..., A, 4, 4) world (= training env frame: table top z = 0) base poses: z = -drop, yaw = pi (all arms face env -x),
    as the training env places the robots (init pos (x, y, -DROP), rot xyzw (0, 0, 1, 0))."""
    xy = torch.as_tensor(t["bases"] if isinstance(t, dict) else t, dtype=torch.float32)
    return base_T(xy[..., 0], xy[..., 1], -drop, yaw)



def twin_from_env(t):
    """Twin template dict -> (4, 4) rigid transform env frame -> twin TABLE frame (scene.json frame, table top z = 0 in both).  The
    template stores tf = [th, mid_x, mid_y] with env_xy = Rz(th) (twin_xy - mid), so twin = Rz(-th) env + mid, yaw_twin = yaw_env - th.
    Bases in the twin frame: twin_from_env(t) @ bases_from_template(t)."""
    th, mx, my = [float(v) for v in t["tf"]]
    T = base_T(0.0, 0.0, 0.0, -th)
    T[0, 3], T[1, 3] = mx, my
    return T

# ---------------------------------------------------------------- forward kinematics
def fk(q, finger=0.04):
    """q (..., 7) joint angles -> (..., 13, 4, 4) frames in the BASE frame, order FRAMES = link0 (identity), link1..link7, flange (panda_link8),
    hand, tcp (fingertip centre), lfinger, rfinger.  finger: finger joint value(s) in m (0 closed .. 0.04 open), float or (...,) tensor."""
    sh = q.shape[:-1]
    c, s = torch.cos(q), torch.sin(q)
    ca, sa = torch.cos(_c(DH_ALPHA, q)), torch.sin(_c(DH_ALPHA, q))
    a, d = _c(DH_A, q), _c(DH_D, q)
    # Craig: T_i = RotX(alpha) TransX(a) RotZ(q) TransZ(d)  -> closed form
    z, o = torch.zeros_like(c), torch.ones_like(c)
    E_ = lambda x: x.expand_as(c)
    L = torch.stack([c, -s, z, E_(a),
                     s * ca, c * ca, E_(-sa), E_(-sa * d),
                     s * sa, c * sa, E_(ca), E_(ca * d),
                     z, z, z, o], -1).reshape(sh + (7, 4, 4))
    out = [torch.eye(4, dtype=q.dtype, device=q.device).expand(sh + (4, 4))]
    for i in range(7):
        out.append(out[-1] @ L[..., i, :, :])
    E = torch.eye(4, dtype=q.dtype, device=q.device)
    fl = E.clone(); fl[2, 3] = D_FLANGE
    hy = E.clone(); ch, shy = math.cos(HAND_YAW), math.sin(HAND_YAW); hy[0, 0], hy[0, 1], hy[1, 0], hy[1, 1] = ch, -shy, shy, ch
    tc = E.clone(); tc[2, 3] = TCP_OFF
    flange = out[7] @ fl; hand = flange @ hy; tcp = hand @ tc
    w = torch.as_tensor(finger, dtype=q.dtype, device=q.device).expand(sh)
    Fl = E.expand(sh + (4, 4)).clone(); Fl[..., 1, 3] = w; Fl[..., 2, 3] = FINGER_Z
    Fr = E.expand(sh + (4, 4)).clone(); Fr[..., 0, 0] = Fr[..., 1, 1] = -1; Fr[..., 1, 3] = -w; Fr[..., 2, 3] = FINGER_Z   # incl. pi yaw
    out += [flange, hand, tcp, hand @ Fl, hand @ Fr]
    return torch.stack(out, -3)


def fk_tcp(q):
    """q (..., 7) -> (..., 4, 4) TCP pose in the base frame"""
    return fk(q)[..., TCP, :, :]


def home_tcp(bases, q=None):
    """bases (..., A, 4, 4 | 7) world base poses (bases_from_template) -> (..., A, 4, 4) world TCP pose of each arm at Q_HOME (or q (7,));
    T_to_pose() gives [x, y, z, qx, qy, qz, qw].  Used as the world model's initial hand positions."""
    B = as_T(bases) if torch.is_tensor(bases) else as_T(torch.as_tensor(bases, dtype=torch.float32))
    qh = _c(Q_HOME if q is None else q, B)
    return B @ fk_tcp(qh)


def jacobian(F):
    """fk frames (..., 13, 4, 4) -> (..., 6, 7) geometric TCP Jacobian in the base frame (rows: linear xyz, angular xyz); joint i turns about
    z of link i (Craig convention)"""
    zax, org = F[..., 1:8, :3, 2], F[..., 1:8, :3, 3]                      # (..., 7, 3)
    p = F[..., TCP, None, :3, 3]
    return torch.cat([torch.linalg.cross(zax, p - org, dim=-1), zax], -1).transpose(-1, -2)


# ---------------------------------------------------------------- inverse kinematics
def _rot_err(R, Rt, yaw_free):
    """-> (..., 3) rotation-vector error taking R to Rt (base frame) and (...,) angle.  yaw_free: only the tool z axis is constrained."""
    if yaw_free:
        z, zt = R[..., :, 2], Rt[..., :, 2]
        cr = torch.linalg.cross(z, zt, dim=-1); sn = cr.norm(dim=-1); cs = (z * zt).sum(-1)
        ang = torch.atan2(sn, cs)
        ax = torch.where(sn[..., None] > 1e-9, cr / sn.clamp_min(1e-9)[..., None], torch.zeros_like(cr))
        # z antiparallel (sn = 0, ang = pi): any perpendicular axis works -> tool x
        ax = torch.where(((sn < 1e-9) & (cs < 0))[..., None], R[..., :, 0], ax)
        return ax * ang[..., None], ang
    qe = _R_to_wxyz(Rt @ R.transpose(-1, -2))                             # w >= 0
    v = qe[..., 1:]; n = v.norm(dim=-1)
    ang = 2 * torch.atan2(n, qe[..., 0])
    return v * torch.where(n > 1e-9, ang / n.clamp_min(1e-9), torch.full_like(n, 2.0))[..., None], ang


def pose_err(q, target, yaw_free=False):
    """-> position error (m) and orientation error (rad) of fk(q)'s TCP vs target (..., 7 | 4, 4), base frame"""
    T, Tt = fk_tcp(q), as_T(target)
    return (Tt[..., :3, 3] - T[..., :3, 3]).norm(dim=-1), _rot_err(T[..., :3, :3], Tt[..., :3, :3], yaw_free)[1]


def _dls(Tt, q, qr, yaw_free, iters, lam, w_rot, k_null, k_lim, step_max):
    """the DLS loop of ik(): Tt (..., 4, 4), q / qr (..., 7) -> q"""
    lo, hi = _c(Q_LO, q), _c(Q_HI, q)
    q = q.clamp(lo, hi)
    pt, Rt = Tt[..., :3, 3], Tt[..., :3, :3]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    W = _c([1, 1, 1, w_rot, w_rot, w_rot], q)
    I6 = torch.eye(6, dtype=q.dtype, device=q.device); I7 = torch.eye(7, dtype=q.dtype, device=q.device)
    for it in range(iters):
        F = fk(q); T = F[..., TCP, :, :]
        ep = pt - T[..., :3, 3]
        ep = ep * (0.1 / ep.norm(dim=-1, keepdim=True).clamp_min(0.1))     # <= 10 cm per step
        er, _ = _rot_err(T[..., :3, :3], Rt, yaw_free)
        J = jacobian(F)
        if yaw_free:                                                     # spin about the tool z is free: project it out of the angular rows
            z = T[..., :3, 2]
            J = torch.cat([J[..., :3, :], J[..., 3:, :] - z[..., :, None] * (z[..., None, :] @ J[..., 3:, :])], -2)
        J = J * W[:, None]; e = torch.cat([ep, er], -1) * W
        Jp = torch.linalg.solve(J @ J.transpose(-1, -2) + lam * lam * I6, J).transpose(-1, -2)   # (..., 7, 6) damped pseudo-inverse
        u = (q - mid) / half                                             # -1 .. 1 across each joint's range
        lim = -torch.sign(u) * ((u.abs() - 0.9).clamp_min(0) / 0.1) * half * k_lim
        dq0 = k_null * (qr - q) + lim
        dq = (Jp @ e[..., None]).squeeze(-1) + ((I7 - Jp @ J) @ dq0[..., None]).squeeze(-1)
        dq = dq * (step_max / dq.abs().amax(-1, keepdim=True).clamp_min(step_max))
        q = (q + dq).clamp(lo, hi)
    return q


def ik(target, q_init, yaw_free=False, iters=80, tol_pos=0.005, tol_rot=math.radians(3.0), lam=0.02, w_rot=0.5, k_null=0.05, k_lim=0.2,
       q_rest=None, step_max=0.3, n_seeds=1, gen=None):
    """Batched damped-least-squares IK of the TCP pose.  target (..., 7 | 4, 4) BASE frame; q_init (..., 7) (broadcast).  yaw_free: only the TCP
    z axis direction is matched (top-down grasp: target from top_down(), yaw free) - position + 2 rotational dof.
    Each iteration: dq = J+ e + (I - J+ J) dq0 with J+ = J^T (J J^T + lam^2 I)^-1 (rotation rows weighted w_rot), dq0 = k_null (q_rest - q)
    (q_rest default Q_NEUTRAL) + a joint-limit push inside the outer 10% of each range; max |dq| clamped to step_max; q clamped to the limits.
    n_seeds > 1: targets that fail from q_init are retried from n_seeds-1 uniform-random starts (torch Generator gen); among those the successful
    one closest to q_init (max-abs joint diff) is kept, else the lowest-error one.  (DLS from one start lands in a local minimum for ~20% of
    random whole-workspace targets; 8 seeds -> ~99%.)
    -> q (..., 7), ok (...,) [pos < tol_pos and rot < tol_rot], pos_err (m), rot_err (rad)."""
    Tt = as_T(target)
    sh = torch.broadcast_shapes(Tt.shape[:-2], q_init.shape[:-1])
    Tt, q0 = Tt.expand(sh + (4, 4)), q_init.expand(sh + (7,))
    qr = _c(Q_NEUTRAL, q0) if q_rest is None else q_rest
    qr = qr.expand(sh + (7,))
    q = _dls(Tt, q0, qr, yaw_free, iters, lam, w_rot, k_null, k_lim, step_max)
    pe, re = pose_err(q, Tt, yaw_free)
    ok = (pe < tol_pos) & (re < tol_rot)
    if n_seeds > 1 and not bool(ok.all()):
        f = (~ok).reshape(-1).nonzero().squeeze(-1)                      # retry the failures only
        Tf, q0f, qrf = Tt.reshape(-1, 4, 4)[f], q0.reshape(-1, 7)[f], qr.reshape(-1, 7)[f]
        lo, hi = _c(Q_LO, q0), _c(Q_HI, q0)
        n, k = len(f), n_seeds - 1
        rnd = torch.rand((n, k, 7), generator=gen, dtype=q0.dtype, device="cpu" if gen is None else gen.device).to(q0.device)
        qs = _dls(Tf[:, None].expand(n, k, 4, 4), lo + (hi - lo) * rnd, qrf[:, None].expand(n, k, 7), yaw_free, iters, lam, w_rot,
                  k_null, k_lim, step_max)
        pes, res = pose_err(qs, Tf[:, None].expand(n, k, 4, 4), yaw_free)
        oks = (pes < tol_pos) & (res < tol_rot)
        score = torch.where(oks, (qs - q0f[:, None]).abs().amax(-1), 1e3 + pes / tol_pos + res / tol_rot)
        j = score.argmin(-1); ar = torch.arange(n, device=q.device)
        q, ok, pe, re = [x.reshape((-1,) + x.shape[len(sh):]).clone() for x in (q, ok, pe, re)]
        # keep the retry if it succeeded, or if it beats the q_init run's error
        better = oks[ar, j] | ((pes[ar, j] / tol_pos + res[ar, j] / tol_rot) < (pe[f] / tol_pos + re[f] / tol_rot))
        fb = f[better]; jb = j[better]; ab = ar[better]
        q[fb], ok[fb], pe[fb], re[fb] = qs[ab, jb], oks[ab, jb], pes[ab, jb], res[ab, jb]
        q, ok, pe, re = q.reshape(sh + (7,)), ok.reshape(sh), pe.reshape(sh), re.reshape(sh)
    return q, ok, pe, re


# ---------------------------------------------------------------- configuration signature + path IK
def arm_config(q):
    """q (..., 7) -> elbow (...,) m: height of the elbow (link4 origin) above the shoulder(link2)-wrist(link6) line, measured in the vertical plane
    through that line; wrist (...,): sin of the signed angle between joint-5 and joint-7 axes about the joint-6 axis (0 = wrist singularity).
    A sign change of either between neighbouring waypoints = a branch flip (elbow / wrist configuration swap)."""
    F = fk(q)
    sh_, el, wr = F[..., 2, :3, 3], F[..., 4, :3, 3], F[..., 6, :3, 3]
    d = wr - sh_
    up = torch.zeros_like(d); up[..., 2] = 1
    n = torch.linalg.cross(d, up, dim=-1)
    u = torch.linalg.cross(n, d, dim=-1); u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)   # in-plane "up" normal to the line
    elbow = ((el - sh_) * u).sum(-1)
    z5, z6, z7 = F[..., 5, :3, 2], F[..., 6, :3, 2], F[..., 7, :3, 2]
    wrist = (torch.linalg.cross(z5, z7, dim=-1) * z6).sum(-1)
    return elbow, wrist


def flips(q_path, q_start=None, jump=0.5, elbow_eps=0.02, wrist_eps=0.1):
    """q_path (..., T, 7) -> (..., T) bool: waypoint t jumped from t-1 by > jump rad in any joint, or the elbow / wrist sign flipped (both sides
    beyond elbow_eps m / wrist_eps, so a slow pass through zero is not flagged twice by noise).  Waypoint 0 is compared to q_start if given."""
    if q_start is not None:
        q_path = torch.cat([q_start[..., None, :].expand(q_path.shape[:-2] + (1, 7)), q_path], -2)
    e, w = arm_config(q_path)
    J = (q_path[..., 1:, :] - q_path[..., :-1, :]).abs().amax(-1) > jump
    fe = (e[..., 1:] * e[..., :-1] < 0) & (e[..., 1:].abs() > elbow_eps) & (e[..., :-1].abs() > elbow_eps)
    fw = (w[..., 1:] * w[..., :-1] < 0) & (w[..., 1:].abs() > wrist_eps) & (w[..., :-1].abs() > wrist_eps)
    f = J | fe | fw
    if q_start is None:
        f = torch.cat([torch.zeros_like(f[..., :1]), f], -1)
    return f


def ik_path(poses, q_start, yaw_free=False, iters_first=80, iters=30, n_seeds_first=1, jump=0.5, gen=None, **kw):
    """IK along TCP waypoints.  poses (B, T, 7 | 4, 4) BASE frame (any leading dims), q_start (B, 7).  Waypoint 0 is solved from q_start
    (iters_first, n_seeds_first, null-space pull to Q_NEUTRAL), every later one warm-started from the previous solution with the null-space pull
    toward that previous solution (minimal joint motion).  -> q (B, T, 7), ok (B, T) IK converged, flip (B, T) branch flip vs the previous waypoint
    (flips(); waypoint 0 never flagged - the move from q_start to it is the caller's job, see approach / move_to), pos_err, rot_err (B, T)."""
    T_ = as_T(poses)
    Tn = T_.shape[-3]
    qs, oks, pes, res = [], [], [], []
    q = q_start
    for t in range(Tn):
        if t == 0:
            q, ok, pe, re = ik(T_[..., 0, :, :], q, yaw_free, iters=iters_first, n_seeds=n_seeds_first, gen=gen, **kw)
        else:
            q, ok, pe, re = ik(T_[..., t, :, :], q, yaw_free, iters=iters, q_rest=q, **kw)
        qs.append(q); oks.append(ok); pes.append(pe); res.append(re)
    Q = torch.stack(qs, -2)
    return Q, torch.stack(oks, -1), flips(Q, jump=jump), torch.stack(pes, -1), torch.stack(res, -1)


# ---------------------------------------------------------------- sphere collision model
def sphere_local(ref):
    """-> (S, 3) centres in their link frames, (S,) radii, (S,) fk frame index; on ref's device / dtype"""
    L = _c(S_LOCAL, ref)
    return L[:, :3], L[:, 3], torch.as_tensor(S_FRAME, device=ref.device)


def spheres(q, base=None, finger=0.04):
    """q (..., 7), base (..., 4, 4 | 7) or None -> world sphere centres (..., S, 3), radii (S,).  Sphere s sits on body S_BODY[s]."""
    c, r, fi = sphere_local(q)
    F = fk(q, finger)[..., fi, :, :]                                      # (..., S, 4, 4)
    p = (F[..., :3, :3] @ c[:, :, None]).squeeze(-1) + F[..., :3, 3]
    if base is not None:
        B = as_T(base)[..., None, :, :]
        p = (B[..., :3, :3] @ p[..., None]).squeeze(-1) + B[..., :3, 3]
    return p, r


def _mask(ref, skip):
    return torch.as_tensor([b not in skip for b in S_BODY], device=ref.device)


def _pair_min(pa, ra, pb, rb, valid=None, chunk=2 ** 24):
    """(..., Sa, 3), (Sa,), (..., Sb, 3), (Sb,) -> (...,) min surface distance over sphere pairs [valid (Sa, Sb) bool mask]; chunked so at most
    ~chunk pair distances are alive at once"""
    sh = torch.broadcast_shapes(pa.shape[:-2], pb.shape[:-2])
    pa, pb = pa.expand(sh + pa.shape[-2:]).reshape(-1, *pa.shape[-2:]), pb.expand(sh + pb.shape[-2:]).reshape(-1, *pb.shape[-2:])
    rr = ra[:, None] + rb[None]
    n = max(1, chunk // (pa.shape[-2] * pb.shape[-2]))
    out = []
    for i in range(0, pa.shape[0], n):
        d = torch.cdist(pa[i:i + n], pb[i:i + n]) - rr
        if valid is not None:
            d = d.masked_fill(~valid, float("inf"))
        out.append(d.flatten(-2).amin(-1))
    return torch.cat(out).reshape(sh) if out else pa.new_zeros(sh)


_SELF_VALID = None


def _self_valid():
    global _SELF_VALID
    if _SELF_VALID is None:
        ign = {(a, b) for a, bs in SELF_IGNORE.items() for b in bs}
        _SELF_VALID = torch.as_tensor([[a != b and (a, b) not in ign and (b, a) not in ign for b in S_BODY] for a in S_BODY])
    return _SELF_VALID


def arm_arm_min_dist(q_a, base_a, q_b, base_b, finger_a=0.04, finger_b=0.04, skip=()):
    """min surface distance (m, < 0 = overlap) between arm A's and arm B's sphere sets, world frame; q / base broadcast over leading dims
    (e.g. (B, K, 7) over time).  skip: bodies left out on both arms (e.g. ('link0',) if the pedestals are known apart)."""
    pa, r = spheres(q_a, base_a, finger_a); pb, _ = spheres(q_b, base_b, finger_b)
    m = _mask(q_a, skip)
    return _pair_min(pa[..., m, :], r[m], pb[..., m, :], r[m])


def self_min_dist(q, finger=0.04):
    """(...,) min surface distance between non-adjacent bodies of one arm (SELF_IGNORE pairs and same-body pairs excluded)"""
    p, r = spheres(q, None, finger)
    return _pair_min(p, r, p, r, _self_valid().to(q.device))


def arm_plane_min_dist(q, base, z_table, finger=0.04, skip=TABLE_SKIP, xy_box=None):
    """(...,) min over the arm's spheres of (sphere bottom - z_table) in the world (m, < 0 = below the plane).  skip: bodies ignored (default
    link0, link1: they stand beside the table under its top).  xy_box (xmin, xmax, ymin, ymax): only spheres whose centre is over that box count."""
    p, r = spheres(q, base, finger)
    d = p[..., 2] - r - z_table
    keep = _mask(q, skip).expand_as(d)
    if xy_box is not None:
        x0, x1, y0, y1 = xy_box
        keep = keep & (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
    return d.masked_fill(~keep, float("inf")).amin(-1)


def arm_points_min_dist(q, base, points, finger=0.04, skip=(), point_r=0.0):
    """(...,) min distance (m) from the arm's sphere surfaces to points (P, 3) or (..., P, 3) (world; an object as a point cloud, inflated by point_r)"""
    p, r = spheres(q, base, finger)
    m = _mask(q, skip)
    pr = torch.full((points.shape[-2],), float(point_r), dtype=p.dtype, device=p.device)
    return _pair_min(p[..., m, :], r[m], points, pr)


def arm_objects_min_dist(q, base, obj_spheres, finger=0.04, skip=()):
    """per-object clearance: q (..., 7), obj_spheres (..., O, 4) or (O, 4) world [x, y, z, r] (one bounding sphere per object, or several rows
    per object if you min them yourself) -> (..., O) min distance from the arm's sphere surfaces to each object sphere's surface (< 0 = overlap).
    Leave the object the hand carries / is grasping out of obj_spheres (it touches the fingers by design)."""
    p, r = spheres(q, base, finger)
    m = _mask(q, skip)
    p, r = p[..., m, :], r[m]
    o = obj_spheres
    d = (p[..., :, None, :] - o[..., None, :, :3]).norm(dim=-1) - r[:, None] - o[..., None, :, 3]     # (..., S, O)
    return d.amin(-2)


# ---------------------------------------------------------------- timing + shared clock
def time_parameterise(q_path, vmax_scale=0.5, dt_min=0.0):
    """q_path (..., T, 7) -> times (..., T) s, times[..., 0] = 0: each segment lasts max_j |dq_j| / (V_MAX_j * vmax_scale) (>= dt_min)
    (piecewise-linear joint motion at a constant per-segment speed; no acceleration limits)"""
    v = _c(V_MAX, q_path) * vmax_scale
    seg = ((q_path[..., 1:, :] - q_path[..., :-1, :]).abs() / v).amax(-1).clamp_min(dt_min)
    return torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(-1)], -1)


def sample_at(q_path, times, t):
    """q_path (..., T, 7), times (..., T) non-decreasing, t (..., K) (or (K,)) local time -> (..., K, 7) linear interpolation; held at
    q_path[0] before times[0] and at q_path[-1] after times[-1]"""
    sh = q_path.shape[:-2]
    t = t.expand(sh + t.shape[-1:]).contiguous()
    T = times.shape[-1]
    i = torch.searchsorted(times.contiguous(), t, right=True).clamp(1, T - 1)          # segment [i-1, i]
    t0, t1 = times.gather(-1, i - 1), times.gather(-1, i)
    a = ((t - t0) / (t1 - t0).clamp_min(1e-9)).clamp(0, 1)
    a = torch.where(t >= times[..., -1:], torch.ones_like(a), a)
    g = lambda j: q_path.gather(-2, j[..., None].expand(j.shape + (7,)))
    return g(i - 1) + a[..., None] * (g(i) - g(i - 1))


def resample(q_path, times, dt, start=0.0, t_end=None):
    """trajectory with local times started at global time `start` (float or (...,)) -> (t_grid (K,), q (..., K, 7)) on the global grid 0, dt, ..
    t_end (default: its own end).  Before its start the arm holds q_path[0] (waiting), after its end q_path[-1]."""
    st = torch.as_tensor(start, dtype=times.dtype, device=times.device)
    if t_end is None:
        t_end = float((times[..., -1] + st).max())
    tg = torch.arange(0.0, t_end + dt * 0.5, dt, dtype=times.dtype, device=times.device)
    return tg, sample_at(q_path, times, tg - st[..., None] if st.dim() else tg - st)


def join(parts):
    """build one trajectory from consecutive pieces: (q_path (..., T, 7), times (..., T)) tuples and floats = wait seconds (hold the last q).
    Each piece starts where the previous ended (its first waypoint should equal the previous last one).  -> (q_path, times)"""
    Q, Tm = None, None
    for p in parts:
        if isinstance(p, (int, float)):
            Q = torch.cat([Q, Q[..., -1:, :]], -2); Tm = torch.cat([Tm, Tm[..., -1:] + float(p)], -1)
            continue
        q, t = p
        if Q is None:
            Q, Tm = q, t - t[..., :1]
        else:
            Q = torch.cat([Q, q], -2); Tm = torch.cat([Tm, Tm[..., -1:] + t - t[..., :1]], -1)
    return Q, Tm


def check_two_arms(traj_a, t_a, start_a, traj_b, t_b, start_b, base_a, base_b, dt=0.05, margin=0.02, finger_a=0.04, finger_b=0.04, skip=(),
                   t_end=None, z_table=None, obstacles=None):
    """two asynchronous trajectories on one clock.  traj (..., T, 7) joint paths, t (..., T) their local times (time_parameterise / join),
    start (float or (...,)) global start time of each (before it the arm waits at its first waypoint), base (4, 4 | 7) world poses (or per sample
    (..., 4, 4)), t_end: grid end (default: when the later one finishes; a batch shares one grid, finished arms hold their last q).
    -> dict t (K,) global grid, dist (..., K) arm-arm min surface distance, min_dist (...,), hit (...,) any dist < margin,
    first_hit (...,) first global time with dist < margin (nan if none), qa / qb (..., K, 7) both arms' joints on the grid (call any clearance
    function on them, e.g. arm_points_min_dist with a moving object's points at the grid times).
    z_table (float): + table_a / table_b (..., K) arm_plane_min_dist (TABLE_SKIP).  obstacles ((O, 4) or (..., O, 4) world spheres, static):
    + obst_a / obst_b (..., K, O) arm_objects_min_dist.  Carried / grasped objects must not be in obstacles (they touch the fingers)."""
    sa, sb = [torch.as_tensor(s, dtype=t_a.dtype, device=t_a.device) for s in (start_a, start_b)]
    if t_end is None:                                                     # grid runs until the later arm finishes (both then hold still)
        t_end = float(torch.maximum((t_a[..., -1] + sa).max(), (t_b[..., -1] + sb).max()))
    tg, qa = resample(traj_a, t_a, dt, sa, t_end)
    _, qb = resample(traj_b, t_b, dt, sb, t_end)
    ba, bb = as_T(base_a), as_T(base_b)
    ba = ba[..., None, :, :] if ba.dim() > 2 else ba                    # per-sample bases broadcast over the time axis
    bb = bb[..., None, :, :] if bb.dim() > 2 else bb
    d = arm_arm_min_dist(qa, ba, qb, bb, finger_a, finger_b, skip)
    hitk = d < margin
    first = torch.where(hitk.any(-1), tg[hitk.float().argmax(-1)], torch.full(hitk.shape[:-1], float("nan"), dtype=tg.dtype, device=tg.device))
    out = dict(t=tg, dist=d, min_dist=d.amin(-1), hit=hitk.any(-1), first_hit=first, qa=qa, qb=qb)
    if z_table is not None:
        out["table_a"] = arm_plane_min_dist(qa, ba, z_table, finger_a); out["table_b"] = arm_plane_min_dist(qb, bb, z_table, finger_b)
    if obstacles is not None:
        ob = obstacles[..., None, :, :] if obstacles.dim() > 2 else obstacles                       # per-sample obstacles: add the time axis
        out["obst_a"] = arm_objects_min_dist(qa, ba, ob, finger_a); out["obst_b"] = arm_objects_min_dist(qb, bb, ob, finger_b)
    return out


# ---------------------------------------------------------------- checked moves
def check_path(q_path, base=None, z_table=None, finger=0.04, table_skip=TABLE_SKIP, table_margin=0.0, self_margin=0.0, xy_box=None,
               q_start=None, jump=0.5):
    """per-waypoint checks of a joint path (..., T, 7) -> dict flip (..., T) (flips(), vs q_start for waypoint 0 if given), table (..., T) sphere
    clearance to the table plane (inf if base / z_table None), self (..., T) self-collision clearance, bad_table / bad_self (..., T) bool"""
    fl = flips(q_path, q_start, jump)
    sd = self_min_dist(q_path, finger)
    if z_table is None:
        td = torch.full_like(sd, float("inf"))
    else:
        b = None if base is None else as_T(base)
        b = b[..., None, :, :] if (b is not None and b.dim() > 2) else b
        td = arm_plane_min_dist(q_path, b, z_table, finger, table_skip, xy_box)
    return dict(flip=fl, table=td, self=sd, bad_table=td < table_margin, bad_self=sd < self_margin)


def joint_interp(q_a, q_b, n):
    """(..., 7), (..., 7) -> (..., n, 7) straight joint-space line, endpoints included"""
    s = torch.linspace(0, 1, n, dtype=q_a.dtype, device=q_a.device)[:, None]
    return q_a[..., None, :] + s * (q_b - q_a)[..., None, :]


def _reasons(codes):
    names = ["ok", "ik", "flip", "table", "self", "grasp_ik", "grasp_flip", "grasp_table", "grasp_self"]
    return [names[int(c)] for c in codes.flatten()]


def move_to(q_start, pose, base=None, z_table=None, yaw_free=False, n=20, n_seeds=8, finger=0.04, table_margin=0.01, self_margin=0.0,
            table_skip=TABLE_SKIP, xy_box=None, gen=None, **ik_kw):
    """free move (empty gripper or carrying): IK at pose (BASE frame TCP, (B, 7 | 4, 4)) from q_start (B, 7) [n_seeds starts, closest successful
    one to q_start kept], then a joint-space line q_start -> q_goal of n waypoints, checked for flips (> 0.5 rad per step or an elbow/wrist swap),
    table clearance (world plane z_table under the base pose `base`, sphere bodies table_skip ignored) and self-collision.
    -> dict q (B, n, 7), ok (B,), reason list[str] ('ok' | 'ik' | 'flip' | 'table' | 'self'), checks (check_path dict), q_goal, pos_err, rot_err.
    Time it with time_parameterise(q) and check it against the other arm with check_two_arms exactly like any other piece."""
    qg, ok_ik, pe, re = ik(pose, q_start, yaw_free, n_seeds=n_seeds, gen=gen, **ik_kw)
    Q = joint_interp(q_start, qg, n)
    ck = check_path(Q, base, z_table, finger, table_skip, table_margin, self_margin, xy_box)
    code = torch.zeros(ok_ik.shape, dtype=torch.long, device=Q.device)
    for c, bad in ((4, ck["bad_self"].any(-1)), (3, ck["bad_table"].any(-1)), (2, ck["flip"].any(-1)), (1, ~ok_ik)):   # last write = priority
        code = torch.where(bad, torch.full_like(code, c), code)
    return dict(q=Q, ok=code == 0, reason=_reasons(code), checks=ck, q_goal=qg, pos_err=pe, rot_err=re)


def approach(q_start, grasp_pose, approach_dist=0.08, base=None, z_table=None, yaw_free=False, n_joint=20, n_lin=10, n_seeds=8, finger=0.04,
             table_margin=0.01, self_margin=0.0, xy_box=None, gen=None, **ik_kw):
    """grasp approach: q_start -> pre-grasp (grasp pose moved back approach_dist along the gripper approach axis = TCP z) by a checked joint-space
    move (move_to: IK, flips, table, self) -> straight Cartesian descent pre-grasp -> grasp (n_lin waypoints, ik_path warm-started from the
    pre-grasp solution; checked for IK, flips, self and table with the fingers exempt, CONTACT_SKIP).  grasp_pose (B, 7 | 4, 4) BASE frame.
    -> dict q (B, n_joint + n_lin - 1, 7) [index n_joint-1 = pre-grasp, last = grasp], ok (B,), reason list[str] ('ok' | 'ik' | 'flip' | 'table'
    | 'self' for the joint move, 'grasp_ik' | 'grasp_flip' | 'grasp_table' | 'grasp_self' for the descent), i_pre."""
    G = as_T(grasp_pose)
    back = torch.eye(4, dtype=G.dtype, device=G.device); back[2, 3] = -approach_dist
    P = G @ back
    mv = move_to(q_start, P, base, z_table, yaw_free, n_joint, n_seeds, finger, table_margin, self_margin, TABLE_SKIP, xy_box, gen, **ik_kw)
    s = torch.linspace(0, 1, n_lin, dtype=G.dtype, device=G.device)[1:]          # waypoint 0 = pre-grasp itself (already reached)
    L = G[..., None, :, :].expand(G.shape[:-2] + (n_lin - 1, 4, 4)).clone()
    L[..., :3, 3] = P[..., None, :3, 3] + s[:, None] * (G[..., None, :3, 3] - P[..., None, :3, 3])
    ql, okl, _, _, _ = ik_path(L, mv["q_goal"], yaw_free, iters_first=40, iters=30, **ik_kw)
    ckl = check_path(ql, base, z_table, finger, CONTACT_SKIP, table_margin, self_margin, xy_box, q_start=mv["q_goal"])
    code = torch.tensor([["ok", "ik", "flip", "table", "self"].index(r) for r in mv["reason"]], device=G.device).reshape(mv["ok"].shape)
    c2 = torch.zeros_like(code)
    for c, bad in ((8, ckl["bad_self"].any(-1)), (7, ckl["bad_table"].any(-1)), (6, ckl["flip"].any(-1)), (5, ~okl.all(-1))):
        c2 = torch.where(bad, torch.full_like(c2, c), c2)
    code = torch.where(code == 0, c2, code)
    return dict(q=torch.cat([mv["q"], ql], -2), ok=code == 0, reason=_reasons(code), i_pre=n_joint - 1, move=mv, descent_checks=ckl,
                descent_ok=okl)
