"""Two Franka arms running NVIDIA's pretrained Isaac-Lift-Franka policy (frozen, zero-shot, no training), each carrying objects along
the frozen world model's own imagined path, the gripper opening where the world model says the hand lets go. No container rule,
nearest-object rule or task labels.

  WM          `cleantables.demos.wm_live.WMLive`: the watch_notab judge members (seeds 1 + 2), twins of WM_TWIN_ROOT, 3-D only, no hands.
              First frame: the frozen judge gives the movers, their order (p_now), their goal (T1 end, top centre) and its tolerance (sd);
              the path head (T2) gives each mover's path, anchored on its goal; the T5 contact head gives P(a hand holds it) along that
              path. At every new task (an arm starts its next object) the path and T5 are re-read from the actual table (same goal; no
              re-read mid-carry).
  allocation  dynamic: a free arm (idle / back from a release) takes the next object in the WM's order that it reaches (closed-form
              Franka reach, start and goal), not let go of / given up, not targeted by the other arm, and not covered by the other arm's
              predicted occupancy; if both free arms want the same object, the nearer base gets it.
  bowl        --bowl_scale (default 1.25,1.15; env NVD_BOWL_SCALE): bowl-type objects scaled in this demo's scene and in the WM's object
              extents before its first query (one twin's bowl is reconstructed 21 % smaller than the same library bowl elsewhere);
              labelled in the clip.
  prior       per arm, the NVIDIA Lift actor on its own 1250-D observation (joint_env_franka / FrankaArmObs, last action written back);
              command = a carrot 10 cm ahead of the object's progress on the WM path; the first part of the path is kept >= 12 cm above
              the start -> goal line, since the lift policy only lifts towards a goal in the air.
  release     --release wm : at the WM contact head's release point (release.release_frac: last step with P(hold) / its peak >= 0.5;
                             release.early_to_end: a release before half of the path is moved to the path end; no signal: path end)
              --release end: at the end of the WM path.
              The gripper opens (0.6 s): at the path end, once the object is over the WM's end pose within the judge's own tolerance
              (max(sd, 3 cm) per axis) and < 15 cm above it; mid-path, once the object is at (< 6 cm) or past the head's point; or after
              1 s without progress within twice that tolerance.
  two arms    prioritised velocity tuning (Kant & Zucker 1986, path-velocity decomposition): every tick each arm's occupancy for the
              next ~1 s = its link spheres now + its commanded motion ahead (path points ahead while carrying, palm -> object while
              reaching, palm -> home while homing). Predicted clearance < MARGIN: the lower-priority arm holds (carrying: its target
              frozen; otherwise its joints frozen); clearance now < MARGIN_NOW, or the arms touch: it rewinds (carrying: target 8 path
              points back and 5 cm up) or backs off to its park pose (home with the shoulder turned --park rad away from the other arm;
              also where an arm with an empty queue waits). Link spheres around the link origins (links 3-7, hand, fingers).
              Priority: an arm carrying its object (the carry started first), then by task start; ties go to arm 0; an idle / homing arm
              is lowest. Deadlock guard: an arm that has yielded > 4 s to an arm that is not carrying takes priority (once per pair of
              tasks).
  bookkeeping an object let go of is not picked again; an object whose distance to its path end has not improved by 1 cm for 10 s (or
              that fell off the table) is given up; an arm with an empty queue goes home.
  judges      (b) task-agnostic: released, at rest, not touched, root xy within max(WM sd, 3 cm) per axis of the WM goal;
              (a) container check, reported only: inside the labelled container, at rest, not touched.

  $SIM_PY -m cleantables.demos.lift --scene IMG_2516_t11 --num_envs 32 --episodes 1 --out X.json [--clip PREFIX --num_envs 1]
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import math
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="IMG_2516_t11", help="twin name in WM_TWIN_ROOT")
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--release", default="wm", choices=["wm", "end"])
ap.add_argument("--jitter", type=float, default=0.015, help="m, start xy jitter per object (env 0 of episode 0: none = the measured layout)")
ap.add_argument("--yaw_jitter", type=float, default=8.0, help="deg")
ap.add_argument("--assign", default="balanced", choices=["balanced", "nearest"], help="balanced = env2.queues (each mover to an arm that reaches its start and goal; both: the shorter queue); nearest = the nearer reaching base")
ap.add_argument("--park", type=float, default=1.1, help="rad: the park pose turns the shoulder this far away from the other arm (0 = park at home)")
ap.add_argument("--margin", type=float, default=0.22, help="m, predicted clearance conflict trigger")
ap.add_argument("--margin_now", type=float, default=0.15, help="m, current clearance before backing off/rewinding")
ap.add_argument("--sphere_r", type=float, default=0.065, help="m, collision sphere radius around arm links")
ap.add_argument("--bowl_scale", default=os.environ.get("NVD_BOWL_SCALE", "1.25,1.15"), help="xy,z scale of bowl-type objects in this demo's scene and in what the world model sees; 1,1 = off")
ap.add_argument("--only_arm", type=int, default=-1, help="-1 = both arms; 0/1 = one-arm demo: only this arm takes objects, the other stays parked")
ap.add_argument("--coord", type=int, default=1, help="two-arm priority coordination (0 = off, a comparison)")
ap.add_argument("--clip", default="", help="prefix: env 0, phone view with the WM paths, real time (15 fps of the 30 Hz control)")
ap.add_argument("--clip_scale", type=float, default=1.0)
ap.add_argument("--out", required=True)
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

os.environ.setdefault("WM_TWIN_ROOT", CT_DATA + "/wm_eval/wm_v2_final_model/dr_root_clean_relink")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cleantables.twin.robot.joint_env_franka as JEF  # noqa: E402
from cleantables.twin.robot.env_franka import ARMS, ROOT  # noqa: E402

BASES_JSON = CT_DATA + "/wm_eval/wm_v2_world/fastpath/franka_bases_humanoid.json"
TWR = os.environ["WM_TWIN_ROOT"]
LABEL = "NVIDIA lift prior + world-model path" + (" + WM release" if args.release == "wm" else " + release at path end") + \
        ("; two-arm priority coordination; no task rules" if args.coord else "; no coordination; no task rules")
BOWL = [float(x) for x in args.bowl_scale.split(",")]
if BOWL != [1.0, 1.0]:
    LABEL += f"; bowl enlarged {BOWL[0]:g}x"

# ------------------------------------------------------------------------------------------------ scene + humanoid bases
_sc = os.path.join(ROOT, "data/twin_table/ep/_scalefix", args.scene)
SCENE = f"ep/_scalefix/{args.scene}" if os.path.exists(os.path.join(_sc, "scene.json")) else os.path.join(TWR, args.scene)
BJ = json.load(open(BASES_JSON))["scenes"][args.scene]


def humanoid_mount(scene):
    """the project's humanoid-stance bases (arm 0 -> robot a, arm 1 -> robot b), table frame; base z = -TABLE_DROP as the env's convention."""
    import copy
    S = copy.deepcopy(scene)
    for a, b in zip(ARMS, BJ["bases"]):
        x, y, z, yw = b
        S["poses"][a] = [[float(x), float(y), float(z)], [0.0, 0.0, math.sin(yw / 2), math.cos(yw / 2)]]
    S["arms"] = dict(S["arms"]); S["arms"]["yaw_deg"] = math.degrees(BJ["bases"][0][3])
    return S


JEF.franka_mount = humanoid_mount
_RNG = np.random.default_rng(args.seed)
_FIRST = [True]


def measured_layouts(S, n, rng, *a, **k):
    """the measured layout + a small xy / yaw jitter per object (the very first env of the first episode: none); spares off the table."""
    K = len(S["objects"])
    xyz = np.zeros((n, K, 3)); yaw = np.zeros((n, K))
    for kk, o in enumerate(S["objects"]):
        xyz[:, kk] = np.asarray(o["pos"], float)
        xyz[:, kk, 2] += 0.003
        if o["role"] not in ("container",):
            xyz[:, kk, :2] += _RNG.uniform(-args.jitter, args.jitter, (n, 2))
            yaw[:, kk] = np.radians(_RNG.uniform(-args.yaw_jitter, args.yaw_jitter, n))
    if _FIRST[0]:
        xyz[0] = [np.asarray(o["pos"], float) + [0, 0, 0.003] for o in S["objects"]]; yaw[0] = 0.0; _FIRST[0] = False
    return xyz, yaw, {"failed_layouts": [], "mode": "measured+jitter"}


JEF.sample_layouts = measured_layouts
cfg, S, POSES = JEF.JointCleanupEnv.make_cfg(SCENE, args.num_envs, args.seed)
BOWL_IDS = [o["id"] for o in S["objects"] if "bowl" in (o["id"] + o.get("category", "")).lower()]
if BOWL != [1.0, 1.0]:                                   # this demo's scene only (the shared twin files are not touched)
    for _k, _o in enumerate(S["objects"]):
        if _o["id"] in BOWL_IDS:
            getattr(cfg.scene, f"obj_{_k}").spawn.scale = (BOWL[0], BOWL[0], BOWL[1])
            _o["extents"] = [_o["extents"][0] * BOWL[0], _o["extents"][1] * BOWL[0], _o["extents"][2] * BOWL[1]]
    print("BOWL_SCALE", BOWL, BOWL_IDS, flush=True)
if args.clip:
    from cleantables.twin.ep import rtx as _rtx
    _rtx.add_cameras(cfg, S, ROOT, wrist_names=())
    args.enable_cameras = True

with launch_simulation(cfg, args):
    from cleantables.demos.wm_live import WMLive
    from cleantables.robots.lift_policy import LiftActor
    from cleantables.twin.robot.env_kuka import OPEN, POLICY
    from cleantables.robots.release import early_to_end, on_fallback, release_frac
    from cleantables.rl.chain_env import RR_A7, RR_D1, RR_J7, RR_NEAR, RR_R
    from cleantables.worldmodel.bridge import watch_path

    # -------------------------------------------------------------------------------------------- the WM (+ its T5 contact head)
    class WMLiveT5(WMLive):
        """WMLive with the T5 contact head and the raw T2 trajectory kept from the same seeded read-out (wm_live._readout + t5)."""
        keep = None

        @torch.no_grad()
        def _readout(self, tpl_ids, xyz, yaw, with_paths=True):
            from cleantables.worldmodel.schema import MAXO
            from cleantables.worldmodel.bridge import watch_readout
            LP = self.LP; TBL = LP._const_tables()
            tp = torch.as_tensor(np.asarray(tpl_ids), device=self.dev).long(); n = len(tp)
            X = torch.zeros(n, MAXO, 3, device=self.dev); Yw = torch.zeros(n, MAXO, device=self.dev)
            Ii = min(xyz.shape[1], MAXO)
            X[:, :Ii] = torch.as_tensor(np.asarray(xyz)[:, :Ii], dtype=torch.float32, device=self.dev)
            Yw[:, :Ii] = torch.as_tensor(np.asarray(yaw)[:, :Ii], dtype=torch.float32, device=self.dev)
            gc = TBL["_gc"][tp]; I = TBL["_I"][tp]
            slot = torch.arange(MAXO, device=self.dev)[None]
            isobj = slot < I[:, None]
            L, W, H, yl = gc[..., 0], gc[..., 1], gc[..., 2], gc[..., 3] + Yw
            top = X[..., 2] + H
            geo = torch.stack([X[..., 0], X[..., 1], top, L, W, top, torch.sin(2 * yl), torch.cos(2 * yl)], -1) * isobj[..., None]
            istab = slot == I[:, None]
            geo = torch.where(istab[..., None], TBL["_tg"][tp][:, None].expand(-1, MAXO, -1), geo)
            bc = {k: v[tp] for k, v in TBL.items() if not k.startswith("_")}
            bc["obj_geo"] = geo
            mode = torch.ones(n, dtype=torch.long, device=self.dev)
            Es = []
            for m in LP.models:
                E = m.encode(bc, mode); E["kind"] = bc["obj_kind"]; Es.append(E)
            tok = torch.stack([E["obj"].float() for E in Es]).mean(0)
            if not with_paths:
                return dict(tok=tok)
            R = watch_readout(LP.models, Es, paths=True, K=self.K, steps=self.steps)
            if self.keep is not None:
                self.keep.append(dict(t5=R["t5"].float(), traj=R["traj"].float(), start=geo[..., :3]))
            return dict(p_now=R["p_now"].float(), traj=R["traj"].float(), goal=R["goal"].float(), start=geo[..., :3], tok=tok)

        @torch.no_grad()
        def reread(self, xyz, yaw):
            """seeded T2 traj + T5 for every object of these n layouts -> traj (n, I, H, 3), t5 (n, I, H), start top (n, I, 3)"""
            n = len(xyz)
            self.keep = []
            gen = torch.random.fork_rng(devices=[] if self.dev.type == "cpu" else [self.dev]); gen.__enter__(); torch.manual_seed(self.seed)
            try:
                for c in range(0, n, self.chunk):
                    self._readout(np.zeros(len(xyz[c:c + self.chunk]), int), xyz[c:c + self.chunk], yaw[c:c + self.chunk])
            finally:
                gen.__exit__(None, None, None)
            kp, self.keep = self.keep, None
            I = self.Imax
            return (torch.cat([k["traj"] for k in kp])[:, :I], torch.cat([k["t5"] for k in kp])[:, :I], torch.cat([k["start"] for k in kp])[:, :I])

    class Demo(JEF.JointCleanupEnv):
        pass

    torch.manual_seed(args.seed)
    env = Demo(cfg, S, POSES, max_seconds=args.seconds, auto_release=False, seed=args.seed, autoreset=False)
    E_, dev, N, K = env.env, env.dev, env.N, env.K
    ar = torch.arange(N, device=dev)
    wm = WMLiveT5([args.scene], device="cuda")
    wm_ids = [str(x) for x in wm.LP.B[0].Z["ids"]]
    if BOWL != [1.0, 1.0]:                               # the WM sees the bigger bowl before its first query (its constant tables are built lazily)
        _Z = wm.LP.B[0].Z
        _Z["extents"] = np.array(_Z["extents"], np.float32, copy=True)
        for _i, _id in enumerate(wm_ids):
            if _id in BOWL_IDS:
                _Z["extents"][_i] = _Z["extents"][_i] * np.array([BOWL[0], BOWL[0], BOWL[1]], np.float32)
                wm.H[0, _i] = _Z["extents"][_i][2]
        assert getattr(wm.LP, "_TBL", None) is None
    I_wm = len(wm_ids)
    missing = [i for i in wm_ids if i not in env.names]
    assert not missing, f"WM objects not in the scene: {missing}"
    wm_k = torch.tensor([env.names.index(i) for i in wm_ids], device=dev)            # WM slot -> scene object
    H_k = env.ext[:, 2]                                                                 # object height (root = bottom centre)
    prior = LiftActor(os.path.join(ROOT, JEF.CKPT), dev)
    bases = torch.tensor([b[:3] for b in BJ["bases"]], dtype=torch.float32, device=dev)  # (2,3) table frame
    LINKS = ["panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"]
    link_ids = E_.robots["a"].find_bodies(LINKS)[0]
    q_home = {a: E_.robots[a].data.default_joint_pos.torch.clone() for a in ARMS}
    # park pose (generic, geometric): home with the shoulder (joint 1) turned PARK rad away from the other arm's base; a backing-off arm,
    # or one with nothing left to do, goes there (out of the shared space) instead of the default home
    q_park = {}
    for a_, A_ in enumerate(ARMS):
        yw = BJ["bases"][a_][3]; left = torch.tensor([-math.sin(yw), math.cos(yw)], device=dev)
        sgn = 1.0 if float(((bases[a_, :2] - bases[1 - a_, :2]) * left).sum()) >= 0 else -1.0
        qp = q_home[A_].clone(); qp[:, int(E_.arm_jids[0])] += sgn * args.park
        q_park[A_] = qp
    DT = env.dt
    NP = 33
    MARGIN, MARGIN_NOW, R_SPH, YIELD_SWAP = args.margin, args.margin_now, args.sphere_r, int(4.0 / DT)
    AHEAD, FLOOR, OPEN_STEPS, GIVE_UP, STALL_REL = 0.10, 0.12, int(0.6 / DT), int(10.0 / DT), int(1.0 / DT)
    IDLE, PRIOR, OPENING, HOME = 0, 1, 2, 3
    PH = ["idle", "prior", "open", "home"]
    if args.clip:
        _rtx.hide_ground(); _rtx.set_static_cams(E_, S, dev)

    def reach(pt, a):
        """(n,3) table-frame TCP points -> bool, the closed-form top-down Franka reach of cleantables.rl.chain_env from base a"""
        b = bases[a]
        hd = (pt[:, :2] - b[:2]).norm(dim=-1)
        dz = pt[:, 2] + RR_J7 - (b[2] + RR_D1)
        return ((hd - RR_A7).clamp(min=0) ** 2 + dz ** 2 < RR_R ** 2) & (hd > RR_NEAR)

    def state():
        pc, qc = env._ostate()
        yaw = 2.0 * torch.atan2(qc[..., 2], qc[..., 3])
        return pc, qc, yaw

    def wm_inputs(pc, yaw, rows):
        xyz = pc[rows][:, wm_k].cpu().numpy() - np.array([0, 0, 0.0], np.float32)
        return xyz.astype(np.float32), yaw[rows][:, wm_k].cpu().numpy().astype(np.float32)

    def floor_path(P):
        """(n,33,3) root path -> the carrot path: its first part kept >= FLOOR above the start -> end line"""
        u = torch.linspace(0, 1, NP, device=P.device)[None]
        z0, zg = P[:, 0, 2:3], P[:, -1, 2:3]
        fl = z0 + (zg - z0) * u + FLOOR * (1 - u)
        Q = P.clone(); Q[..., 2] = torch.maximum(P[..., 2], fl)
        return Q

    def hold_idx(t5):
        """(n,H) T5 at the T2 steps -> release index on the 33-point path (32 = the path end) + whether the head decided it"""
        n = t5.shape[0]
        h32 = torch.nn.functional.interpolate(torch.cat([t5[:, :1], t5], -1)[:, None], size=32, mode="linear", align_corners=True)[:, 0]
        h2, early = early_to_end(h32)
        fr = release_frac(h2)
        fb = on_fallback(h2)
        return (fr * (NP - 1)).round().long().clamp(1, NP - 1), ~fb, early

    def tips(a):
        """(N,K) force (N) of arm a's fingertips per object, summed; and both-fingers-on (N,K)"""
        fs = [s_.data.normal_force_matrix_w.torch.reshape(N, -1, 3).norm(dim=-1) for s_ in E_.tip_sensors[a]]
        return fs[0] + fs[1], (fs[0] > 1.0) & (fs[1] > 1.0)

    def links(a):
        return E_.robots[a].data.body_pos_w.torch[:, link_ids] - env.origins[:, None]

    def palm(a):
        return E_.robots[a].data.body_pos_w.torch[:, E_.palm_bid] - env.origins

    # -------------------------------------------------------------------------------------------- one episode
    def run_episode(ep):
        env._reset_envs(ar, settle=True)
        pc, qc, yaw = state()
        xyz, yw = wm_inputs(pc, yaw, ar)
        wm.keep = []
        out = wm.query(np.zeros(N, int), xyz, yw, paths=True, tokens=False)
        kp, wm.keep = wm.keep, None
        t5_0 = torch.cat([k["t5"] for k in kp])[:, :I_wm].to(dev)
        moves = torch.as_tensor(out["moves"][:, :I_wm], device=dev)
        goal_top = torch.as_tensor(np.nan_to_num(out["goal"][:, :I_wm]), device=dev)
        sd = torch.as_tensor(out["sd"][:, :I_wm], device=dev)
        path0_top = torch.as_tensor(np.nan_to_num(out["path"][:, :I_wm]), device=dev)
        order = out["order"][:, :I_wm]
        G_root = goal_top.clone(); G_root[..., 2] = (goal_top[..., 2] - H_k[wm_k][None]).clamp(min=0.0)
        tol = torch.maximum(sd[..., :2], torch.full_like(sd[..., :2], 0.03))
        order_l = [[x for x in order[e].tolist() if x >= 0] for e in range(N)]   # the WM's movers by p_now (first frame)
        st_top = pc[:, wm_k].clone(); st_top[..., 2] += H_k[wm_k][None]
        R0 = torch.stack([torch.stack([reach(st_top[:, i], a) & reach(goal_top[:, i], a) for i in range(I_wm)], 1) for a in range(2)], -1)
        unreach = int((moves & ~R0.any(-1)).sum())
        taken = [[[], []] for _ in range(N)]
        no_work = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        last_preds = [None]
        # per-arm state
        phase = torch.full((N, 2), IDLE, dtype=torch.long, device=dev)
        tgt = torch.full((N, 2), -1, dtype=torch.long, device=dev)          # WM slot
        Pc = torch.zeros(N, 2, NP, 3, device=dev)                            # carrot path (root, floored)
        Pv = torch.zeros(N, 2, NP, 3, device=dev)                            # the WM path as imagined (top centre), drawn
        r_idx = torch.full((N, 2), NP - 1, dtype=torch.long, device=dev)
        r_head = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        kprog = torch.zeros(N, 2, dtype=torch.long, device=dev)
        near_c = torch.zeros(N, 2, dtype=torch.long, device=dev)
        best_rel = torch.full((N, 2), 9.0, device=dev); rel_stall = torch.zeros(N, 2, dtype=torch.long, device=dev)
        best_end = torch.full((N, 2), 9.0, device=dev); end_stall = torch.zeros(N, 2, dtype=torch.long, device=dev)
        open_cd = torch.zeros(N, 2, dtype=torch.long, device=dev)
        t0task = torch.full((N, 2), 10 ** 9, dtype=torch.long, device=dev)
        lift_t = torch.full((N, 2), 10 ** 9, dtype=torch.long, device=dev)
        yld = torch.zeros(N, 2, dtype=torch.long, device=dev); swapped = torch.zeros(N, dtype=torch.bool, device=dev)
        prio_over = torch.full((N,), -1, dtype=torch.long, device=dev)
        let_go = torch.zeros(N, I_wm, dtype=torch.bool, device=dev); gave_up = torch.zeros(N, I_wm, dtype=torch.bool, device=dev)
        rel_how = {"head_mid": 0, "head_end": 0, "path_end": 0, "end_no_head": 0, "early_to_end": 0, "fallback_end": 0}
        rel_frac_log = []
        st = dict(yield_onsets=0, yield_steps=0, rewinds=0, swaps=0, aa_contact_onsets=0, aa_contact_steps=0, give_up_stall=0, give_up_fell=0,
                  releases=0, releases_while_held=0, grasped=0, both_active_steps=0, any_active_steps=0, tasks=[0, 0], idle_with_work_steps=[0, 0])
        grasped_once = torch.zeros(N, I_wm, dtype=torch.bool, device=dev)
        aa_prev = torch.zeros(N, dtype=torch.bool, device=dev)
        yld_prev = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        goal_cmd = torch.zeros(N, 2, 3, device=dev)
        yielding = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        hold_q = {a: torch.zeros(N, E_.robots[a].num_joints, device=dev) for a in ARMS}
        frames = []
        for a in ARMS:
            m = torch.ones(N, dtype=torch.bool, device=dev); env.obsb[a].reset(m)

        def start_tasks(pc, yaw):
            """Dynamic allocation: a free arm (idle / homing / just let go) takes the next object in the WM's order that it reaches
            (start and goal), that is not let go of / given up, that the other arm is not targeting, and that the other arm's predicted
            occupancy does not cover; when both arms are free at once and want the same object, the nearer base gets it. Path + T5 are
            re-read from the actual table."""
            ph = phase.cpu().numpy()
            free = (ph == IDLE) | (ph == HOME)
            top = pc[:, wm_k].clone(); top[..., 2] += H_k[wm_k][None]
            rch = torch.stack([reach(top.reshape(-1, 3), a).reshape(N, I_wm) & reach(goal_top.reshape(-1, 3), a).reshape(N, I_wm) for a in range(2)], -1)
            if args.only_arm >= 0:
                rch[..., 1 - args.only_arm] = False
            avail = moves & ~let_go & ~gave_up
            busy_o = torch.from_numpy((ph == PRIOR) | (ph == OPENING)).to(dev)
            tk = torch.stack([(torch.nn.functional.one_hot(tgt[:, 1 - a].clamp(min=0), I_wm).bool() & (tgt[:, 1 - a] >= 0)[:, None] & busy_o[:, 1 - a, None])
                              for a in range(2)], -1)                                                   # (N,I,2): the other arm has it
            way = torch.zeros(N, I_wm, 2, dtype=torch.bool, device=dev)
            if last_preds[0] is not None:
                for a in range(2):
                    way[..., a] = torch.cdist(top, last_preds[0][1 - a]).amin(-1) < 0.15
            dB = torch.stack([(top[..., :2] - bases[a, :2]).norm(dim=-1) for a in range(2)], -1)
            ok = (avail[..., None] & rch & ~tk)
            no_work.copy_(~ok.any(1))
            okw = (ok & ~way).cpu().numpy(); okn = ok.cpu().numpy(); dBn = dB.cpu().numpy()
            need = []
            for e in np.nonzero(free.any(1))[0].tolist():
                ch = {}
                for a in (0, 1):
                    if free[e, a]:
                        c = [j for j in order_l[e] if okw[e, j, a]]
                        ch[a] = c
                        if not c and okn[e, :, a].any():
                            st["idle_with_work_steps"][a] += 1
                if len(ch) == 2 and ch[0] and ch[1] and ch[0][0] == ch[1][0]:
                    j = ch[0][0]; w = 0 if dBn[e, j, 0] <= dBn[e, j, 1] else 1
                    ch[1 - w] = [x for x in ch[1 - w] if x != j]
                for a, c in ch.items():
                    if c:
                        need.append((e, a, c[0])); taken[e][a].append(wm_ids[c[0]]); st["tasks"][a] += 1
            if not need:
                return
            rows = sorted({e for e, _, _ in need})
            first = int(env.t[0]) == 0
            if not first:
                xyz, yw = wm_inputs(pc, yaw, torch.tensor(rows, device=dev))
                tr, t5, s0 = wm.reread(xyz, yw)
                tr, t5, s0 = tr.to(dev), t5.to(dev), s0.to(dev)
            for e, a, j in need:
                if first:
                    pt = path0_top[e, j]; h = t5_0[e, j]
                else:
                    ri = rows.index(e)
                    pt = watch_path(s0[ri, j][None], goal_top[e, j][None], tr[ri, j][None], NP=NP)[0]
                    if not bool(torch.isfinite(pt).all()):
                        u = torch.linspace(0, 1, NP, device=dev)[:, None]; pt = s0[ri, j] + u * (goal_top[e, j] - s0[ri, j])
                    h = t5[ri, j]
                Pv[e, a] = pt
                pr = pt.clone(); pr[:, 2] -= H_k[wm_k[j]]
                pr[0] = pc[e, wm_k[j]]
                Pc[e, a] = floor_path(pr[None])[0]
                ri_, head, early = hold_idx(h[None])
                if args.release == "wm":
                    r_idx[e, a] = ri_[0]; r_head[e, a] = head[0] & ~early[0]
                    rel_frac_log.append(dict(obj=wm_ids[j], frac=round(float(ri_[0]) / (NP - 1), 3), head=bool(head[0]), early_to_end=bool(early[0])))
                    if bool(early[0]):
                        rel_how["early_to_end"] += 1
                    elif not bool(head[0]):
                        rel_how["fallback_end"] += 1
                else:
                    r_idx[e, a] = NP - 1; r_head[e, a] = False
                tgt[e, a] = j; phase[e, a] = PRIOR; kprog[e, a] = 0; near_c[e, a] = 0
                best_rel[e, a] = 9.0; rel_stall[e, a] = 0; best_end[e, a] = 9.0; end_stall[e, a] = 0
                t0task[e, a] = int(env.t[e]); yld[e, a] = 0; lift_t[e, a] = 10 ** 9
                m = torch.zeros(N, dtype=torch.bool, device=dev); m[e] = True
                env.obsb[ARMS[a]].reset(m)
            # a new pair of tasks resets the deadlock guard
            for e in rows:
                swapped[e] = False; prio_over[e] = -1

        T = int(args.seconds / DT)
        t_wall = time.time()
        for s in range(T):
            pc, qc, yaw = state()
            start_tasks(pc, yaw)
            # ---------------- progress along the carrot path, carrot, release test
            for a in range(2):
                A = ARMS[a]
                act_ = phase[:, a] == PRIOR
                j = tgt[:, a].clamp(min=0); k = wm_k[j]
                o = pc[ar, k]
                win = (kprog[:, a, None] + torch.arange(-1, 9, device=dev)[None]).clamp(0, NP - 1)          # (N,10)
                Pw = Pc[:, a].gather(1, win[..., None].expand(-1, -1, 3))
                kk = win.gather(1, (Pw - o[:, None]).norm(dim=-1).argmin(1, keepdim=True))[:, 0]
                kprog[:, a] = torch.where(act_, torch.maximum(kprog[:, a], kk), kprog[:, a])
                # carrot: the first path point >= AHEAD from the object, at or beyond the progress index
                idx = torch.arange(NP, device=dev)[None]
                dd = (Pc[:, a] - o[:, None]).norm(dim=-1)
                cand = (idx > kprog[:, a, None]) & (dd >= AHEAD)
                ci = torch.where(cand.any(1), torch.where(cand, idx, torch.full_like(idx, NP)).amin(1), torch.full_like(kprog[:, a], NP - 1))
                carrot = Pc[ar, a, ci]
                goal_cmd[:, a] = torch.where(act_[:, None], carrot, goal_cmd[:, a])
                # release point and test
                Rp = Pc[ar, a, r_idx[:, a]]
                dxy = (o[:, :2] - Rp[:, :2]).norm(dim=-1); d3 = (o - Rp).norm(dim=-1)
                tl = tol[ar, j]                                                    # the WM judge's own tolerance (max(sd, 3 cm), per axis)
                at_end = r_idx[:, a] >= NP - 2
                inxy = ((o[:, :2] - Rp[:, :2]).abs() <= tl).all(-1)
                near_end = at_end & inxy & (o[:, 2] - Rp[:, 2] < 0.15) & (kprog[:, a] >= NP // 2)        # over the WM's end pose
                near_mid = ~at_end & (((kprog[:, a] >= r_idx[:, a] - 1) & (d3 < 0.06)) | (kprog[:, a] > r_idx[:, a] + 2))   # at / past the head's point
                near = act_ & (near_end | near_mid)
                near_c[:, a] = torch.where(near, near_c[:, a] + 1, torch.zeros_like(near_c[:, a]))
                imp = d3 < best_rel[:, a] - 0.005
                best_rel[:, a] = torch.where(act_ & imp, d3, best_rel[:, a])
                cl2 = ((o[:, :2] - Rp[:, :2]).abs() <= 2 * tl).all(-1) & (o[:, 2] - Rp[:, 2] < 0.20) & (kprog[:, a] >= torch.where(at_end, torch.full_like(r_idx[:, a], NP // 2), r_idx[:, a] - 3))
                rel_stall[:, a] = torch.where(act_ & ~imp & cl2, rel_stall[:, a] + 1, torch.zeros_like(rel_stall[:, a]))
                fire = act_ & ((near_c[:, a] >= 3) | (rel_stall[:, a] >= STALL_REL))
                _, both = tips(A)
                held_now = both[ar, k]
                grasped_once[ar, j] |= act_ & held_now
                for e in torch.nonzero(fire)[:, 0].tolist():
                    phase[e, a] = OPENING; open_cd[e, a] = OPEN_STEPS; let_go[e, int(j[e])] = True
                    st["releases"] += 1; st["releases_while_held"] += int(bool(held_now[e]))
                    if args.release == "end":
                        rel_how["path_end"] += 1
                    elif bool(r_head[e, a]):
                        rel_how["head_end" if int(r_idx[e, a]) >= NP - 2 else "head_mid"] += 1
                    else:
                        rel_how["end_no_head"] += 1
                # give up: no 1 cm progress towards the path end for 10 s, or off the table
                de = (o - Pc[:, a, -1]).norm(dim=-1)
                im2 = de < best_end[:, a] - 0.01
                best_end[:, a] = torch.where(act_ & im2, de, best_end[:, a])
                end_stall[:, a] = torch.where(act_ & ~im2 & ~yielding[:, a], end_stall[:, a] + 1, torch.zeros_like(end_stall[:, a]))
                fell = act_ & (o[:, 2] < -0.05)
                gu = act_ & ~fire & ((end_stall[:, a] >= GIVE_UP) | fell)
                for e in torch.nonzero(gu)[:, 0].tolist():
                    gave_up[e, int(j[e])] = True; phase[e, a] = HOME
                    st["give_up_fell" if bool(fell[e]) else "give_up_stall"] += 1
            # ---------------- two-arm coordination (prioritised velocity tuning)
            Ls = [links(ARMS[a]) for a in range(2)]
            yielding = torch.zeros(N, 2, dtype=torch.bool, device=dev)
            rewind = torch.zeros(N, 2, dtype=torch.bool, device=dev)
            if args.coord:
                preds, carr = [], []
                for a in range(2):
                    j = tgt[:, a].clamp(min=0); o = pc[ar, wm_k[j]]
                    pl = palm(ARMS[a])
                    carrying = (phase[:, a] == PRIOR) & (o[:, 2] > env.p_start[ar, wm_k[j], 2] + 0.03)
                    ahead = (kprog[:, a, None] + torch.arange(0, 16, 3, device=dev)[None]).clamp(max=NP - 1)
                    pa = Pc[:, a].gather(1, ahead[..., None].expand(-1, -1, 3)) + torch.tensor([0, 0, 0.06], device=dev)
                    u = torch.linspace(0, 1, 6, device=dev)[None, :, None]
                    tgt_pt = torch.where((phase[:, a] == PRIOR)[:, None], o + torch.tensor([0, 0, 0.06], device=dev), pl)
                    pr = pl[:, None] + u * (tgt_pt - pl)[:, None]
                    pred = torch.where(carrying[:, None, None], pa, pr)
                    preds.append(torch.cat([Ls[a], pred], 1)); carr.append(carrying)
                    lift_t[:, a] = torch.where(carrying & (lift_t[:, a] >= 10 ** 9), env.t, lift_t[:, a])
                last_preds[0] = preds
                d_now = torch.cdist(Ls[0], Ls[1]).flatten(1).amin(1) - 2 * R_SPH
                d_pred = torch.cdist(preds[0], preds[1]).flatten(1).amin(1) - 2 * R_SPH
                conflict = d_pred < MARGIN
                busy = (phase == PRIOR)
                carr = torch.stack(carr, 1)                                         # the carry task started at the lift: carrying arms first
                key = torch.where(busy, torch.where(carr, lift_t, 10 ** 8 + t0task), torch.full_like(t0task, 2 * 10 ** 9))
                hi = torch.where(key[:, 1] < key[:, 0], torch.ones(N, dtype=torch.long, device=dev), torch.zeros(N, dtype=torch.long, device=dev))
                both_busy = busy.all(1)
                prio_over = torch.where(both_busy, prio_over, torch.full_like(prio_over, -1))   # a swap holds only while both arms work
                hi = torch.where(prio_over >= 0, prio_over, hi)
                lo = 1 - hi
                lo_ph = phase[ar, lo]
                can = conflict & ((lo_ph == PRIOR) | (lo_ph == HOME) | (lo_ph == IDLE))
                # deadlock guard: the arm that has yielded > 4 s takes priority (once)
                sw = can & (yld[ar, lo] > YIELD_SWAP) & ~swapped & (phase[ar, hi] == PRIOR) & ~carr[ar, hi]   # a carrying arm is not stuck
                if bool(sw.any()):
                    prio_over[sw] = lo[sw]; swapped |= sw; st["swaps"] += int(sw.sum())
                    hi = torch.where(sw, lo, hi); lo = 1 - hi; lo_ph = phase[ar, lo]
                    can = conflict & ((lo_ph == PRIOR) | (lo_ph == HOME) | (lo_ph == IDLE))
                yielding[ar, lo] = can
                rewind[ar, lo] = can & ((d_now < MARGIN_NOW) | aa_prev)          # too close now, or touching: back off
                yld = torch.where(yielding, yld + 1, torch.zeros_like(yld))
                st["yield_onsets"] += int((yielding & ~yld_prev).sum()); st["yield_steps"] += int(yielding.sum())
                st["rewinds"] += int((rewind & ~yld_prev).sum())
                yld_prev = yielding.clone()
            # ---------------- commands per arm
            act = torch.zeros(N, 18, device=dev)
            for a in range(2):
                A = ARMS[a]
                j = tgt[:, a].clamp(min=0); k = wm_k[j]; o = pc[ar, k]
                carrying = (phase[:, a] == PRIOR) & (o[:, 2] > env.p_start[ar, k, 2] + 0.03)
                y = yielding[:, a]; rw = rewind[:, a]
                g = goal_cmd[:, a].clone()
                g = torch.where((y & carrying)[:, None], o + torch.tensor([0, 0, 0.01], device=dev), g)               # hold: target frozen
                back = Pc[ar, a, (kprog[:, a] - 8).clamp(min=0)] + torch.tensor([0, 0, 0.05], device=dev)
                g = torch.where((rw & carrying)[:, None], back, g)                                                    # rewind
                env.goal[A] = torch.where((phase[:, a] == PRIOR)[:, None], g, o + torch.tensor([0, 0, 0.15], device=dev))
            # prior observation (the target object of each arm; a dummy target when idle) and action
            lifts = []
            for a in range(2):
                A = ARMS[a]; r = E_.robots[A]
                k = wm_k[tgt[:, a].clamp(min=0)]
                gb = env._qai(r.data.root_quat_w.torch, env.goal[A] + env.origins - r.data.root_pos_w.torch)
                frm = env.obsb[A].frame(r.data.root_pos_w.torch, r.data.root_quat_w.torch, r.data.joint_pos.torch, r.data.joint_vel.torch,
                                        r.data.body_pos_w.torch[:, E_.hand_bids], r.data.body_quat_w.torch[:, E_.hand_bids],
                                        pc[ar, k] + env.origins, qc[ar, k], env.pts[k], E_.tip_forces(A, torch.where(tgt[:, a] >= 0, k, -1)), gb)
                lifts.append(env.obsb[A].push(frm))
            with torch.inference_mode():
                a9 = prior(torch.stack(lifts, 1).reshape(N * 2, -1)).reshape(N, 2, 9)
            for a in range(2):
                A = ARMS[a]
                ph = phase[:, a]
                kk_ = wm_k[tgt[:, a].clamp(min=0)]
                carr_ = (ph == PRIOR) & (pc[ar, kk_, 2] > env.p_start[ar, kk_, 2] + 0.03)
                y_back = yielding[:, a] & ~carr_                                                  # approaching arm retreats to park to clear way
                y_hold = torch.zeros_like(y_back)
                use_prior = (ph == PRIOR) & ~y_hold & ~y_back
                aa = torch.where(use_prior[:, None], a9[:, a], torch.zeros_like(a9[:, a]))
                env.obsb[A].set_action(aa)
                E_.action[A] = aa
                frz = ~use_prior
                E_.mode[A] = torch.where(frz, torch.full_like(E_.mode[A], OPEN), torch.full_like(E_.mode[A], POLICY))
                # homing (and retreating) arms: walk the held joint target towards the home pose; frozen arms keep their target
                walk = (ph == HOME) | (ph == IDLE) | y_back                                       # homing / parking arms keep moving away
                ent = frz & (E_._prev_mode[A] == POLICY)
                qh = torch.where(ent[:, None], E_.robots[A].data.joint_pos.torch, E_.q_hold[A])
                to_park = y_back | (((ph == HOME) | (ph == IDLE)) & no_work[:, a])
                qt = torch.where(to_park[:, None], q_park[A], q_home[A])
                s_lim = torch.where(to_park[:, None], torch.tensor(0.045, device=dev), torch.tensor(0.02, device=dev))
                stp = (qt - qh).clamp(-s_lim, s_lim)
                qh = torch.where(walk[:, None], qh + stp, qh)
                E_.q_hold[A] = qh
                E_._prev_mode[A] = torch.where(ent, torch.full_like(E_._prev_mode[A], OPEN), E_._prev_mode[A])
            E_.step(env.zero_act)
            env.t += 1
            # ---------------- phase bookkeeping after the step
            for a in range(2):
                op = phase[:, a] == OPENING
                open_cd[:, a] = torch.where(op, open_cd[:, a] - 1, open_cd[:, a])
                done_open = op & (open_cd[:, a] <= 0)
                phase[:, a] = torch.where(done_open, torch.full_like(phase[:, a], HOME), phase[:, a])
                t0task[:, a] = torch.where(phase[:, a] == PRIOR, t0task[:, a], torch.full_like(t0task[:, a], 10 ** 9))
            both_act = ((phase == PRIOR) & ~yielding).all(1); any_act = (phase == PRIOR).any(1)
            st["both_active_steps"] += int(both_act.sum()); st["any_active_steps"] += int(any_act.sum())
            aa_now = E_.arm_contact_force() > 1.0
            st["aa_contact_onsets"] += int((aa_now & ~aa_prev).sum()); st["aa_contact_steps"] += int(aa_now.sum()); aa_prev = aa_now
            if args.clip and s % 2 == 0:
                frames.append(frame(s, pc, Pv, phase, tgt, yielding, r_idx, Pc, goal_top, moves))
            if s % 150 == 0:
                print(f"  ep {ep} t={s * DT:4.1f}s phases {[[PH[int(x)] for x in phase[e]] for e in range(min(N, 2))]} "
                      f"yields {st['yield_onsets']} aa {st['aa_contact_onsets']} both_act {st['both_active_steps']}/{st['any_active_steps']} "
                      f"d_now {float(d_now[0]) if args.coord else -1:.3f} d_pred {float(d_pred[0]) if args.coord else -1:.3f} {time.time() - t_wall:.0f}s", flush=True)
            if bool(((phase == IDLE) | (phase == HOME)).all()) and bool(no_work.all()) and s > 60:
                # everything tried; let things settle 1 s, then stop
                for _ in range(30):
                    E_.step(env.zero_act)
                break
        # ---------------- judges (final state)
        pc, qc, yaw = state()
        spd = torch.stack([o_.data.root_lin_vel_w.torch.norm(dim=-1) for o_ in E_.objs], 1)
        touch = tips("a")[0] + tips("b")[0]
        rest = (spd < 0.05) & (touch < 1.0)
        o_wm = pc[:, wm_k]
        ok_b = ((o_wm[..., :2] - G_root[..., :2]).abs() <= tol).all(-1) & rest[:, wm_k] & (o_wm[..., 2] > -0.02) & moves
        cont_items = [(i, env.names.index(S["objects"][int(wm_k[i])]["dest"])) for i in range(I_wm)
                      if S["objects"][int(wm_k[i])].get("dest") and S["objects"][int(wm_k[i])]["role"] == "task"]
        ok_a = torch.zeros(N, I_wm, dtype=torch.bool, device=dev); is_task = torch.zeros(I_wm, dtype=torch.bool, device=dev)
        for i, kc in cont_items:
            is_task[i] = True
            ok_a[:, i] = env._inside(o_wm[:, i], torch.full((N,), kc, device=dev), pc) & rest[:, int(wm_k[i])]
        res = []
        for e in range(N):
            mv = moves[e]
            res.append(dict(env=e, episode=ep, movers=int(mv.sum()), queued=taken[e],
                            at_wm_goal=int((ok_b[e] & mv).sum()), task_items=int(is_task.sum()), in_container=int(ok_a[e].sum()),
                            in_container_of_movers=int((ok_a[e] & mv).sum()), movers_task=int((mv & is_task).sum()),
                            grasped=int((grasped_once[e] & mv).sum()), let_go=int(let_go[e].sum()), gave_up=int(gave_up[e].sum()),
                            per_obj={wm_ids[i]: dict(mover=bool(mv[i]), at_wm_goal=bool(ok_b[e, i]), in_container=bool(ok_a[e, i]),
                                                     grasped=bool(grasped_once[e, i]), let_go=bool(let_go[e, i]), gave_up=bool(gave_up[e, i]),
                                                     d_goal_xy_cm=round(float((o_wm[e, i, :2] - G_root[e, i, :2]).norm()) * 100, 1),
                                                     tol_cm=[round(float(x) * 100, 1) for x in tol[e, i]]) for i in range(I_wm)}))
        st["unreachable_movers"] = unreach
        st["release_how"] = rel_how
        st["seconds"] = round(time.time() - t_wall, 1)
        return res, st, frames, rel_frac_log

    # -------------------------------------------------------------------------------------------- video frame
    import cv2
    CAM = S["cameras"]["main"]
    Kc = np.array(CAM.get("K", S["K"]), np.float64).reshape(3, 3)
    Tinv = np.linalg.inv(np.array(CAM["T_cam_to_table"], np.float64))
    WH = CAM.get("wh", S.get("image_wh", [960, 540]))

    def frame(s, pc, Pv, phase, tgt, yielding, r_idx, Pc, goal_top, moves):
        E_.sim.render()
        im = _rtx.grab(E_, "main")
        sc_ = im.shape[1] / WH[0]

        def pj(P):
            X = Tinv[:3, :3] @ np.asarray(P, np.float64).T + Tinv[:3, 3:4]
            return ((Kc @ X)[:2] / np.maximum(X[2:3], 1e-6)).T * sc_
        lay = np.zeros_like(im); wd = max(3, im.shape[1] // 190)
        cols = [((0, 140, 255), (40, 200, 255)), ((255, 170, 0), (255, 230, 120))]      # BGR: arm 0 orange, arm 1 blue
        for a in range(2):
            if int(phase[0, a]) != 1:
                continue
            uv = pj(Pv[0, a].cpu().numpy()).astype(int)
            c0, c1 = cols[a]
            cv2.polylines(lay, [uv.reshape(-1, 1, 2)], False, (255, 255, 255), wd + 4, cv2.LINE_AA)
            cv2.polylines(lay, [uv.reshape(-1, 1, 2)], False, c0, wd, cv2.LINE_AA)
            for k in range(0, len(uv), 4):
                cv2.circle(lay, tuple(int(x) for x in uv[k]), wd + 1, (255, 255, 255), -1, cv2.LINE_AA)
            d_ = uv[-1] - uv[max(len(uv) - 4, 0)]; n_ = float(np.hypot(*d_)) + 1e-6; d_ = d_ / n_
            tip = uv[-1]; L_ = 4 * wd
            pts = np.array([tip + d_ * L_, tip - d_ * L_ * 0.4 + np.array([-d_[1], d_[0]]) * L_ * 0.7, tip - d_ * L_ * 0.4 - np.array([-d_[1], d_[0]]) * L_ * 0.7], np.int32)
            cv2.fillConvexPoly(lay, pts, c0, cv2.LINE_AA)
            if args.release == "wm":                                                  # the WM's release point (where its contact head lets go)
                rp = Pv[0, a, int(r_idx[0, a])].cpu().numpy()
                q = pj(rp[None])[0].astype(int)
                cv2.circle(lay, tuple(int(x) for x in q), 3 * wd, (60, 60, 255), wd, cv2.LINE_AA)
            g = pj(env.goal[ARMS[a]][0:1].cpu().numpy())[0].astype(int)
            cv2.circle(lay, tuple(int(x) for x in g), wd + 3, c1, -1, cv2.LINE_AA)
        msk = lay.any(-1, keepdims=True)
        im = np.where(msk, (0.85 * lay + 0.15 * im).astype(np.uint8), im)
        if args.clip_scale != 1.0:
            im = cv2.resize(im, None, fx=args.clip_scale, fy=args.clip_scale, interpolation=cv2.INTER_AREA)
        return (im, s, [int(phase[0, a]) for a in range(2)], [int(tgt[0, a]) for a in range(2)], [bool(yielding[0, a]) for a in range(2)])

    def write_video(frames, path, final):
        import imageio.v2 as imageio
        out = []
        for im, s, ph, tg, yl in frames:
            W_ = im.shape[1]; fs = 0.5 * W_ / 960
            bar = np.zeros((int(62 * W_ / 960), W_, 3), np.uint8)
            cv2.putText(bar, LABEL, (12, int(24 * W_ / 960)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
            arm_txt = []
            for a in range(2):
                nm = wm_ids[tg[a]] if (ph[a] == 1 and tg[a] >= 0) else ["idle", "-", "letting go", "home"][ph[a]]
                arm_txt.append(f"arm {a}: {nm}{' (yields)' if yl[a] else ''}")
            cv2.putText(bar, f"t={s * DT:4.1f}s   " + "    ".join(arm_txt), (12, int(50 * W_ / 960)), cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 230, 255), 1, cv2.LINE_AA)
            out.append(np.vstack([bar, im]))
        if out:                                                     # the result on the last frame, held 2 s
            im = out[-1].copy()
            cv2.putText(im, final, (12, im.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * im.shape[1] / 960, (120, 255, 120), 2, cv2.LINE_AA)
            out += [im] * 30
        imageio.mimwrite(path, [f[..., ::-1] for f in out], fps=15, quality=8, macro_block_size=None)
        print("VIDEO", path, len(out), "frames", flush=True)

    # -------------------------------------------------------------------------------------------- run
    ALL, STATS, RELS = [], [], []
    _FIRST[0] = True                                                # env 0 of episode 0: the measured layout
    for ep in range(args.episodes):
        res, st, frames, rl = run_episode(ep)
        ALL += res; STATS.append(st); RELS += rl
        e0 = res[0]
        print(f"EPISODE {ep}: at WM goal {sum(r['at_wm_goal'] for r in res)}/{sum(r['movers'] for r in res)}  in container "
              f"{sum(r['in_container'] for r in res)}/{sum(r['task_items'] for r in res)}  stats {json.dumps(st)}", flush=True)
        if args.clip and frames:
            write_video(frames, f"{args.clip}_ep{ep}.mp4",
                        f"put away: {e0['in_container']}/{e0['task_items']} in the container, {e0['at_wm_goal']}/{e0['movers']} at the WM goal")
    mv = sum(r["movers"] for r in ALL); gb = sum(r["at_wm_goal"] for r in ALL)
    ti = sum(r["task_items"] for r in ALL); ic = sum(r["in_container"] for r in ALL)
    tot = {k: sum(s_[k] for s_ in STATS) for k in ("yield_onsets", "yield_steps", "rewinds", "swaps", "aa_contact_onsets", "aa_contact_steps",
                                                    "give_up_stall", "give_up_fell", "releases", "releases_while_held", "unreachable_movers",
                                                    "both_active_steps", "any_active_steps")}
    tot["tasks_per_arm"] = [sum(s_["tasks"][a] for s_ in STATS) for a in range(2)]
    tot["idle_with_work_steps_per_arm"] = [sum(s_["idle_with_work_steps"][a] for s_ in STATS) for a in range(2)]
    rh = {k: sum(s_["release_how"][k] for s_ in STATS) for k in STATS[0]["release_how"]}
    summary = dict(scene=args.scene, label=LABEL, release=args.release, coord=args.coord, assign="dynamic", episodes=args.episodes, envs=N, seconds=args.seconds,
                   at_wm_goal=[gb, mv], in_container=[ic, ti], grasped=[sum(r["grasped"] for r in ALL), mv],
                   totals=tot, release_how=rh, release_points=RELS[:200], bases=BJ["bases"], wm=dict(run="runs_final/watch_notab", seeds=[1, 2], twin_root=TWR),
                   rules=["an object let go of is not picked again", "give up after 10 s without 1 cm progress or off the table",
                          "no work left -> park", "dynamic allocation: a free arm takes the next WM-ordered object it reaches that the other arm is not doing and not in the way; both free + same object -> nearer base"],
                   bowl_scale=BOWL, bowl_ids=BOWL_IDS,
                   per_env=ALL)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print("RESULT", json.dumps({k: summary[k] for k in ("scene", "release", "at_wm_goal", "in_container", "grasped", "totals", "release_how")}), flush=True)
    print("DONE", flush=True)
