"""Batched carry environment: thousands of parallel Isaac Lab / PhysX tables, no robots, no cameras, headless.

Every env is one twin scene (a "template": all its objects in scene.json order, on its own table). One object per env (the `carried`
slot, chosen at run time) is carried along a commanded 33-point root path by a force-limited drive (cap max(2 N, c m g), c = 3), then
held, released and settled; every object's pose is recorded at 10 Hz. The scene is built once for T templates over N envs (Isaac Lab's
sequential map, N / T copies each; one RigidObjectCfg per slot with a MultiAssetSpawnerCfg over the templates and aligned clone worlds);
resets only re-pose. Physics settings follow the reference chain simulator used for the setE results: kinematic table mu 0.6 / 0.5,
solver 16 / 1 iterations, contact offset 0.007, DR mass and friction, minimum-jerk carry timing (path length / 0.25 m/s in [1, 4] s,
0.15 s hold, 1.0 s settle) and the implicit PD drive (LIN 2500 / 150 / cap / 2 m/s, ANG 60 / 3 / 20 N m / 8 rad/s). Differences from the
reference: dt is an argument (production 1/120), no contact sensors, one carry per env from a layout at rest, tables spawned at the env
origin with the table frame shifted (FastPath.org), friction on shared physics materials. Poses: root xyz (bottom centre, table frame) +
quat xyzw.

Use (Isaac Lab, `sim` env):
  cfg = sim_env.make_cfg(templates_pt, num_envs, dt)            # before launch_simulation
  with launch_simulation(cfg, args):
      fp = sim_env.FastPath(cfg)
      out = fp.run(layout (N,K,7), carried (N,), path (N,33,3), dyaw (N,))
CLI:
  python -m cleantables.rl.sim_env --templates OUT.pt --select v1        # CPU: write the template file (no Isaac)
  $SIM_PY -m cleantables.rl.sim_env --bench --templates T.pt --n 4096 [--dt 120] [--headless]
  $SIM_PY -m cleantables.rl.sim_env --setE [--dt 120]
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import math
import os
import random
import threading
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("WM_EVAL_OUT", CT_DATA + "/wm_eval"))
DST = OUT / "wm_v2_world" / "fastpath"
TWIN_ROOT_DEFAULT = OUT / "wm_v2_final_model" / "dr_root"
SETE = OUT / "wm_v2_seg" / "videos" / "setE" / "place"

# ---- carry settings of the reference chain simulator (force-limited carry, c = 3)
CARRY_SPEED, CARRY_MIN_S, CARRY_MAX_S, PRE_S, PRE_CHECK_S, SETTLE_S = 0.25, 1.0, 4.0, 0.6, 0.3, 1.0
HAND_R, HAND_GAP = 0.045, 0.01
HAND_MODE = os.environ.get("FP_HAND", "gripper")      # "gripper" (gripper.py: palm + two fingers, default), "ball" (sphere) or "ghost"
HAND_DYN = os.environ.get("FP_HAND_DYN", "0") == "1"   # force-limited (dynamic, PD-driven) hand parts instead of kinematic ones
HAND_MASS = (0.6, 0.1, 0.1)                             # palm, finger, finger (kg; the Franka hand is ~0.73 kg)
HANDS = int(os.environ.get("FP_HANDS", "1"))          # 2 = a second gripper per env (prims HandQ*); run() uses gripper 0 only
DISP_M = 0.01
LIN_D = dict(stiffness=2500.0, damping=150.0, effort=150.0, vel=2.0)
ANG_D = dict(stiffness=60.0, damping=3.0, effort=20.0, vel=8.0)
HOLD_S = 0.15
EFFORT_C, EFFORT_FLOOR = 3.0, 2.0
CONTACT_OFFSET = 0.007            # as the setE reference run
REC_HZ = 10.0
NP_PATH = 33
G = 9.81
DUMMY_Z = -1.5
PARK_Z = 3.0
SPACING = 3.0
RESET_VIA_ASSETS = os.environ.get("FP_RESET_ASSETS", "1") == "1"
SLEEP = os.environ.get("FP_SLEEP", "1") == "1"     # PhysX sleeping on; FP_SLEEP=0 sets sleep_threshold 0
SOLVER_POS = int(os.environ.get("FP_SOLVER_POS", "16"))   # PhysX position iterations of objects and hand parts (16; 8 is Isaac Lab's
                                                          # usual manipulation setting)


# =====================================================================================================================================
# templates (CPU, no Isaac)
# =====================================================================================================================================
def _quat_yaw(y):
    """yaw (rad) -> quaternion xyzw about z."""
    return np.array([0.0, 0.0, math.sin(y / 2), math.cos(y / 2)])


def build_templates(scenes, twin_root=None, out=None, meta=None):
    """Build the templates. scenes: explicit list of scene names (twin_root/<scene>/scene.json).  -> dict saved to `out` (torch .pt) if given:
    templates: per template name, n (objects), table (x, y ranges), objects[k] (scene.json order) with usd (absolute), scale (None = the
    USD's own), mass_usd (scene.json mass = the USD's physics:mass), mass_f (DR factor, 1 if none), mass (= mass_usd * mass_f, what the
    carry uses), mu (DR friction or None = the USD's own material), mu_usd, damping, yaw, yaw_baked, extents, pos; layout (K, 7) the
    start pose of every object (root xyz + 2 mm, quat xyzw = Rz(yaw) Rz(yaw_baked); NaN rows = unused slots);
    extents (K, 3).  K = the most objects of any listed scene (13 for the DR scenes)."""
    twin_root = Path(twin_root or os.environ.get("WM_TWIN_ROOT", TWIN_ROOT_DEFAULT))
    S = [json.load(open(twin_root / s / "scene.json")) for s in scenes]
    K = max(len(d["objects"]) for d in S)
    T = []
    for name, d in zip(scenes, S):
        objs = []
        lay = np.full((K, 7), np.nan, np.float32)
        ext = np.full((K, 3), np.nan, np.float32)
        for k, o in enumerate(d["objects"]):
            dr = o.get("dr") or {}
            mf = float(dr["mass_f"]) if dr.get("mass_f") is not None else 1.0
            usd = str(ROOT / o["usd"])
            if not os.path.exists(usd):
                raise FileNotFoundError(f"{name}: {usd}")
            sc = o.get("scale")
            objs.append(dict(id=o.get("id"), usd=usd, scale=None if sc is None else float(sc), mass_usd=float(o["mass"]), mass_f=mf,
                             mass=float(o["mass"]) * mf, mu=None if dr.get("mu") is None else float(dr["mu"]), mu_usd=o.get("mu"),
                             damping=[float(x) for x in o["damping"]], yaw=float(o["yaw"]), yaw_baked=float(o.get("yaw_baked", 0.0)),
                             extents=[float(x) for x in o["extents"]], pos=[float(x) for x in o["pos"]], collider=o.get("collider")))
            lay[k, :3] = [o["pos"][0], o["pos"][1], o["pos"][2] + 0.002]
            lay[k, 3:] = _quat_yaw(float(o["yaw"]) + float(o.get("yaw_baked", 0.0)))
            ext[k] = o["extents"]
        T.append(dict(name=name, n=len(objs), table=dict(x=list(d["table"]["x"]), y=list(d["table"]["y"])), objects=objs, layout=lay, extents=ext))
    res = dict(templates=T, k_max=K, twin_root=str(twin_root), scenes=list(scenes), meta=meta or {}, created=time.strftime("%Y-%m-%d %H:%M"))
    if out:
        import torch
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, out)
    return res


def select_v1(twin_root=None, seed=0):
    """Scene selection of the tuning env: train = all training twins + a seeded stratified
    draw of DR_train / DRC_train / DRO_train / DRCO_train up to 128; held-out (never trained on) = the 10 test twins + a seeded draw of
    DR_test / DRC_test / OBS_ up to 32.  -> (train, heldout) lists of scene names."""
    twin_root = Path(twin_root or os.environ.get("WM_TWIN_ROOT", TWIN_ROOT_DEFAULT))
    names = sorted(p.name for p in twin_root.iterdir() if (p / "scene.json").exists())
    rng = random.Random(seed)
    tw = _twins(twin_root)
    train_tw = tw["train"]
    fams = ["DR_train_", "DRC_train_", "DRO_train_", "DRCO_train_"]
    n_left = 128 - len(train_tw)
    per = [n_left // 4 + (1 if i < n_left % 4 else 0) for i in range(4)]
    train = list(train_tw)
    for f, m in zip(fams, per):
        pool = [n for n in names if n.startswith(f)]
        train += sorted(rng.sample(pool, m))
    held = list(tw["test"])
    hf = ["DR_test_", "DRC_test_", "OBS_"]
    n_left = 32 - len(held)
    per = [n_left // 3 + (1 if i < n_left % 3 else 0) for i in range(3)]
    for f, m in zip(hf, per):
        pool = [n for n in names if n.startswith(f)]
        held += sorted(rng.sample(pool, m))
    return train, held


def _twins(twin_root):
    """Training / test twins from tasks_train.json and worldmodel.data.is_test, without importing the WM stack."""
    from cleantables.worldmodel.data import is_test
    T = json.load(open(ROOT / "data/labels/tasks_train.json"))["tasks"]
    ok = lambda i: (Path(twin_root) / i / "scene.json").exists() and (Path(twin_root) / i / "wm_tokens.npz").exists()  # noqa: E731
    return dict(train=sorted(t["id"] for t in T if not is_test(t["id"]) and ok(t["id"])),
                test=sorted(t["id"] for t in T if is_test(t["id"]) and ok(t["id"])))


def synth_paths(tpl, tidx, seed=0):
    """Benchmark inputs: per env (template index tidx (N,)) a random carried object, a goal 20-40 cm away (kept on the table), a 5-10 cm
    lift arc, 33 points; dyaw in [-0.5, 0.5].  -> layout (N,K,7), carried (N,), path (N,33,3), dyaw (N,) (numpy)."""
    rng = np.random.default_rng(seed)
    T = tpl["templates"]
    N, K = len(tidx), tpl["k_max"]
    lay = np.stack([T[t]["layout"] for t in tidx]).astype(np.float32)
    car = np.array([rng.integers(T[t]["n"]) for t in tidx])
    path = np.zeros((N, NP_PATH, 3), np.float32)
    s = np.linspace(0, 1, NP_PATH)
    for e, t in enumerate(tidx):
        p0 = lay[e, car[e], :3].astype(np.float64); tb = T[t]["table"]
        for _ in range(20):
            a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0.20, 0.40)
            g = p0[:2] + r * np.array([np.cos(a), np.sin(a)])
            if tb["x"][0] + 0.08 < g[0] < tb["x"][1] - 0.08 and tb["y"][0] + 0.08 < g[1] < tb["y"][1] - 0.08:
                break
        lift = rng.uniform(0.05, 0.10)
        xy = p0[None, :2] + s[:, None] * (g - p0[:2])[None]
        z = p0[2] + s * (0.002 - p0[2]) + lift * np.sin(np.pi * s)
        path[e] = np.c_[xy, z]
    dyaw = rng.uniform(-0.5, 0.5, N).astype(np.float32)
    return lay, car, path, dyaw


# =====================================================================================================================================
# Isaac Lab scene
# =====================================================================================================================================
def template_of_env(N, T):
    """Template index of each env under Isaac Lab's default 'sequential' clone strategy with equal weights."""
    s = (np.arange(N) + 0.5) / max(N, 1)
    return np.searchsorted(np.cumsum(np.full(T, 1.0 / T)), s, side="right")


def aligned_worlds(T):
    """Clone-combination planner for T clone worlds, world t = variant t of every T-variant asset (variant 0 of the
    single-variant hand)."""
    def make(asset_names, variant_counts, clone_combinations=None, *, all_asset_names=None):
        bad = [(n, c) for n, c in zip(asset_names, variant_counts) if c not in (1, T)]
        if bad:
            raise ValueError(f"fast path scene: every multi-variant asset needs exactly T = {T} variants: {bad}")
        offsets = np.cumsum([0, *variant_counts])
        worlds = tuple(tuple(int(offsets[i] + (t if c == T else 0)) for i, c in enumerate(variant_counts)) for t in range(T))
        return worlds, np.full(T, 1.0 / T)
    return make


def make_cfg(templates, num_envs, dt=1 / 120, contact_offset=CONTACT_OFFSET, buffers="big", bufmul=1):
    """Scene config for `templates` (a build_templates dict or its .pt) over num_envs envs.  Call before launch_simulation."""
    import torch
    import isaaclab.cloner as _cl
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.cloner.cloner_cfg import InclusionSet
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

    tpl = templates if isinstance(templates, dict) else torch.load(templates, map_location="cpu", weights_only=False)
    TT, K = tpl["templates"], tpl["k_max"]
    T = len(TT)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(), init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.76)))

    sc = SceneCfg(num_envs=num_envs, env_spacing=float(os.environ.get("FP_SPACING", SPACING)), replicate_physics=True)
    # every table is spawned centred at its env origin and never moved; the env's table frame is shifted instead (FastPath.org = env
    # origin - table centre). Re-posing a kinematic table, or spawning it off centre, let objects fall through the tables in up to
    # 57 % of the envs at 4096 envs during the first one or two resets.
    sc.table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[
        sim_utils.CuboidCfg(func=_spawn_table, size=(t["table"]["x"][1] - t["table"]["x"][0], t["table"]["y"][1] - t["table"]["y"][0], 0.04),
                            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True), collision_props=sim_utils.UsdPhysicsCollisionCfg())
        for t in TT]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)))
    for g_ in range(HANDS):                                    # gripper 0: Hand*, gripper 1 (FP_HANDS=2): HandQ*
        q_ = "" if g_ == 0 else "Q"
        if HAND_MODE == "ball":                                 # ball proxy: a kinematic sphere above the object
            setattr(sc, "hand" + ("" if g_ == 0 else "_q"), RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Hand" + q_, spawn=sim_utils.SphereCfg(
                radius=HAND_R, rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PARK_Z + g_))))
        else:                                                  # gripper-shaped hand (gripper.py: palm + two fingers), kinematic cuboids
            from cleantables.rl.gripper import box_sizes
            ghost = HAND_MODE == "ghost"                          # diagnostic: the gripper's bodies with collisions off (no physical hand)
            for nm_, sz_, m_ in zip(("HandP", "HandA", "HandB"), box_sizes(), HAND_MASS):
                # HAND_DYN: dynamic hand parts (gravity off), driven by a force-capped PD in chain_env
                rp_ = ([sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
                        PhysxRigidBodyCfg(disable_gravity=True, solver_position_iteration_count=SOLVER_POS, solver_velocity_iteration_count=1,
                                          max_depenetration_velocity=0.5, **({} if SLEEP else dict(sleep_threshold=0.0)))]
                       if HAND_DYN else sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True))
                setattr(sc, f"hand_{q_.lower()}{nm_[-1].lower()}", RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Hand" + q_ + nm_[-1], spawn=sim_utils.CuboidCfg(
                    size=sz_, rigid_props=rp_, **(dict(mass_props=sim_utils.MassPropertiesCfg(mass=m_)) if HAND_DYN else {}),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=not ghost)),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PARK_Z + g_))))
    col = PhysxCollisionCfg(contact_offset=contact_offset, rest_offset=0.0)

    def dummy():
        return sim_utils.MeshCuboidCfg(size=(0.03, 0.03, 0.03), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                       collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                                       mass_props=sim_utils.MassPropertiesCfg(mass=0.05))

    def obj(o, lay, c):
        rp = [sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
              PhysxRigidBodyCfg(solver_position_iteration_count=SOLVER_POS, solver_velocity_iteration_count=1, max_depenetration_velocity=0.5,
                                linear_damping=o["damping"][0], angular_damping=o["damping"][1], **({} if SLEEP else dict(sleep_threshold=0.0)))]
        s_ = o["scale"]
        # spawned at its template layout pose (table frame shifted by the table centre c): no teleport at start-up
        return sim_utils.UsdFileCfg(func=functools.partial(_spawn_usd_at, pos=tuple(float(x) for x in (lay[:3] - c)), quat=tuple(float(x) for x in lay[3:]),
                                                           mu=o["mu"]),
                                    usd_path=o["usd"], rigid_props=rp, collision_props=col, scale=None if s_ is None else (s_, s_, s_))
    import functools
    ctr = lambda t: np.array([(t["table"]["x"][0] + t["table"]["x"][1]) / 2, (t["table"]["y"][0] + t["table"]["y"][1]) / 2, 0.0])  # noqa: E731
    for k in range(K):
        setattr(sc, f"obj{k}", RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Obj{k}", spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[obj(t["objects"][k], np.asarray(t["layout"][k], np.float64), ctr(t)) if k < t["n"] else dummy() for t in TT]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0 + 0.45 * k))))   # stacked well apart above the table (spawning
        # every slot at one point gives explosive first steps and degraded first carries at 4096 envs)
    _cl.make_valid_clone_combinations = aligned_worlds(T)          # T aligned worlds instead of the product of every slot's prototypes
    if os.environ.get("FP_COLOC") == "1":                          # noise study: every copy of a template at the same world position (one grid cell
        import isaaclab.cloner.replicate_session as _rs            # per template; inter-env collisions are filtered), so copies share coordinates
        _grid0 = getattr(_rs, "_grid0", _rs.grid_transforms); _rs._grid0 = _grid0   # and float rounding: identical inputs -> (mostly) identical runs

        def _coloc_grid(N_, spacing, up_axis="z"):
            pos, ori = _grid0(T, spacing, up_axis)
            t_of = template_of_env(N_, T)
            return pos[t_of], np.repeat(ori[:1], N_, 0)
        _rs.grid_transforms = _coloc_grid
    sc.clone_cfg.clone_combinations = [InclusionSet(assets=["obj0"])]   # non-empty: the scene then asks the (replaced) planner

    buffers = os.environ.get("FP_BUFFERS", buffers)
    if buffers == "big":                                       # large GPU buffers sized up front, x bufmul
        m = int(bufmul)
        phys = PhysxCfg(enable_stabilization=True, bounce_threshold_velocity=0.2, gpu_max_rigid_patch_count=m * 2 ** 21,
                        gpu_max_rigid_contact_count=m * 2 ** 23, gpu_found_lost_pairs_capacity=m * 2 ** 24, gpu_found_lost_aggregate_pairs_capacity=m * 2 ** 24,
                        gpu_total_aggregate_pairs_capacity=m * 2 ** 22, gpu_heap_capacity=m * 2 ** 28, gpu_temp_buffer_capacity=m * 2 ** 26,
                        gpu_collision_stack_size=m * 2 ** 27)
    else:                                                      # the reference simulator's sizes
        phys = PhysxCfg(enable_stabilization=True, bounce_threshold_velocity=0.2, gpu_max_rigid_patch_count=2 ** 20, gpu_found_lost_pairs_capacity=2 ** 22,
                        gpu_found_lost_aggregate_pairs_capacity=2 ** 22, gpu_total_aggregate_pairs_capacity=2 ** 22, gpu_max_rigid_contact_count=2 ** 22,
                        gpu_collision_stack_size=2 ** 26, gpu_heap_capacity=2 ** 26, gpu_temp_buffer_capacity=2 ** 24)

    if os.environ.get("FP_DETERMINISM") == "1":                # PhysX enhanced determinism, to measure the run-to-run spread of
        phys.enable_enhanced_determinism = True                 # identical inputs and its speed cost

    @configclass
    class FPCfg(DirectRLEnvCfg):
        episode_length_s = 1.0e5
        decimation = 1
        action_space = 1
        observation_space = 1
        state_space = 0
        sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=1000, physics=phys)
        scene: InteractiveSceneCfg = None

    cfg = FPCfg()
    cfg.scene = sc
    cfg.fp_templates = tpl          # carried along for FastPath
    cfg.fp_contact_offset = contact_offset
    return cfg


def _shared_material(stage, sf, df, rs):
    """One global physics material per (static, dynamic, restitution) value. PhysX allows 64K materials; a per-env copy of each
    object's material exceeds that at 8192 envs."""
    path = f"/World/PhysicsMaterials/m_{int(round(sf * 1e5))}_{int(round(df * 1e5))}_{int(round(rs * 1e4))}"
    if not stage.GetPrimAtPath(path).IsValid():
        import isaaclab.sim as sim_utils
        mc = sim_utils.RigidBodyMaterialCfg(static_friction=float(sf), dynamic_friction=float(df), restitution=float(rs))
        mc.func(path, mc)
    return path


def _bind_shared(prim, path):
    """Bind the shared material (physics purpose, stronger than descendants) on the body root and on every collider under it; the
    body's own material prims lose their PhysicsMaterialAPI (no per-env PhysX material is made for them)."""
    from pxr import Usd, UsdPhysics, UsdShade
    stage = prim.GetStage()
    mat = UsdShade.Material(stage.GetPrimAtPath(path))
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.MaterialAPI):
            p.RemoveAPI(UsdPhysics.MaterialAPI)
        if p == prim or p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")


def _spawn_usd_at(prim_path, cfg, translation=None, orientation=None, pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0), mu=None, **kw):
    """spawn_from_usd at the given env-local pose (xyz, quat xyzw) instead of the slot's shared init pose (no teleport at start-up),
    with a shared physics material: the USD's own (static, dynamic, restitution), or the DR friction mu -> (mu, 0.9 mu). Setting
    friction per shape at run time would make one PhysX material per shape, over the 64K limit at 8192 envs."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import Usd, UsdPhysics
    prim = spawn_from_usd(prim_path, cfg, translation=pos, orientation=quat, **kw)
    mats = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.MaterialAPI)]
    if not mats:
        raise RuntimeError(f"{cfg.usd_path}: no physics material")
    m = UsdPhysics.MaterialAPI(mats[0])
    sf, df = m.GetStaticFrictionAttr().Get(), m.GetDynamicFrictionAttr().Get()
    rs = m.GetRestitutionAttr().Get() or 0.0
    if mu is not None:
        sf, df = float(mu), 0.9 * float(mu)
    _bind_shared(prim, _shared_material(prim.GetStage(), sf, df, rs))
    return prim


def _spawn_table(prim_path, cfg, translation=None, orientation=None, **kw):
    """spawn_cuboid (no own material) with the shared table material mu 0.6 / 0.5, restitution 0."""
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
    prim = spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kw)
    _bind_shared(prim, _shared_material(prim.GetStage(), 0.6, 0.5, 0.0))
    return prim


def _env_class():
    from isaaclab.envs import DirectRLEnv
    import torch

    class FPEnv(DirectRLEnv):
        def _pre_physics_step(self, actions):
            pass

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return z, z
    return FPEnv


# ---- quaternion helpers (xyzw)
def qmul(a, b):
    import torch
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qrotvec(q):
    import torch
    q = torch.where(q[..., 3:4] < 0, -q, q)
    s = q[..., :3].norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return q[..., :3] / s * 2 * torch.atan2(s, q[..., 3:4])


def qyaw(y):
    import torch
    return torch.stack([torch.zeros_like(y), torch.zeros_like(y), torch.sin(y / 2), torch.cos(y / 2)], -1)


def qmat(q):
    """xyzw -> (n,3,3) rotation matrix."""
    import torch
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def yaw_zyx(q):
    """scipy Rotation.from_quat(q).as_euler('zyx')[0] (extrinsic z-y-x: R = Rx Ry Rz) for xyzw quaternions."""
    import torch
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z))


class _GpuWatch:
    """Poll nvidia-smi for the peak total and per-process GPU memory; stop the process when the total exceeds limit_mb."""
    def __init__(self, limit_mb=22000, every=1.0):
        self.peak_total = 0; self.peak_proc = 0; self.limit = limit_mb; self.pid = os.getpid(); self._stop = False
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); self.every = every

    def _loop(self):
        while not self._stop:
            try:
                tot = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                                         text=True, timeout=10).stdout.split()[0])
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=10).stdout
                me = sum(int(l.split(",")[1]) for l in apps.strip().splitlines() if l.strip() and int(l.split(",")[0]) == self.pid)
                self.peak_total = max(self.peak_total, tot); self.peak_proc = max(self.peak_proc, me)
                if tot > self.limit:
                    print(f"FP_ENV GPU total {tot} MiB > {self.limit} MiB: stopping (GPU rules)", flush=True)
                    os._exit(4)
            except Exception:                                     # noqa: BLE001
                pass
            time.sleep(getattr(self, "every", 1.0))

    def stop(self):
        self._stop = True


class FastPath:
    """The built scene (T templates over N envs) and its batched carry.  Build with make_cfg + launch_simulation, then FastPath(cfg)."""

    def __init__(self, cfg):
        import torch
        import warp as wp
        from isaaclab.utils.timer import Timer
        from isaaclab_physx.physics.physx_manager import PhysxManager as _PM
        import re
        self.wp, self._PM = wp, _PM
        try:                                           # one 'Failed to find rigid body' warning per prim otherwise (fills and caps the Kit log,
            import carb                                # hiding later PhysX errors)
            carb.settings.get_settings().set_string("/log/channels/omni.physx.tensors.plugin", "error")
        except Exception:                              # noqa: BLE001
            pass
        t0 = time.time()
        self.env = _env_class()(cfg)
        self.t_env = time.time() - t0
        ti = Timer.timing_info
        self.t_scene = float(ti.get("scene_creation", {}).get("last", float("nan")))
        self.t_start = float(ti.get("simulation_start", {}).get("last", float("nan")))
        env = self.env
        self.dev, self.N = env.device, env.num_envs
        self.dt = float(env.physics_dt)
        self.tpl = cfg.fp_templates
        TT = self.tpl["templates"]; self.T = len(TT); self.K = K = self.tpl["k_max"]
        dev, N = self.dev, self.N
        env.reset()
        self.org = (env.scene.env_origins.torch if hasattr(env.scene.env_origins, "torch") else env.scene.env_origins).clone()
        # ---- clone plan check: every slot of an env carries the same template (the sequential map)
        self.tpl_of_env = self._check_clone_plan()
        tidx = self.tpl_of_env
        # ---- one PhysX view over every object (and one over the hands): the carry is 4 batched calls per physics step
        try:                                           # one 'Failed to find rigid body' warning per prim otherwise (fills the Kit log)
            import carb
            carb.settings.get_settings().set_string("/log/channels/omni.physx.tensors.plugin", "error")
        except Exception:                              # noqa: BLE001
            pass
        SV = _PM.get_physics_sim_view()
        self.OV = SV.create_rigid_body_view("/World/envs/env_*/Obj*"); self.HV = SV.create_rigid_body_view("/World/envs/env_*/Hand*")
        pat = re.compile(r"/World/envs/env_(\d+)/Obj(\d+)$")
        idx = torch.full((N, K), -1, dtype=torch.long)
        for v_, p_ in enumerate(self.OV.prim_paths):
            m_ = pat.match(p_); idx[int(m_.group(1)), int(m_.group(2))] = v_
        NPART = 1 if HAND_MODE == "ball" else 3
        hidx = torch.full((N, HANDS * NPART), -1, dtype=torch.long)                         # env-major: gripper 0's parts, then gripper 1's
        for v_, p_ in enumerate(self.HV.prim_paths):
            m_ = re.match(r"/World/envs/env_(\d+)/Hand(Q?)([PAB]?)$", p_)
            hidx[int(m_.group(1)), (NPART if m_.group(2) else 0) + {"": 0, "P": 0, "A": 1, "B": 2}[m_.group(3)]] = v_
        assert (idx >= 0).all() and (hidx >= 0).all(), "fast path: a body missing from the batched views"
        self.IDX, self.HIDX = idx.to(dev), hidx.to(dev)
        tcpu = torch.as_tensor(tidx)
        self.ntpl = torch.tensor([t["n"] for t in TT])
        self.valid = (torch.arange(K)[None] < self.ntpl[tcpu][:, None]).to(dev)                 # (N,K)
        f32 = lambda x: torch.tensor(np.asarray(x, np.float32))  # noqa: E731
        ext = f32(np.stack([t["extents"] for t in TT]))                                           # (T,K,3)
        self.ext = ext[tcpu].to(dev)
        self.yb = f32([[(t["objects"][k]["yaw_baked"] if k < t["n"] else 0.0) for k in range(K)] for t in TT])[tcpu].to(dev)
        self.lay0 = f32(np.stack([t["layout"] for t in TT]))[tcpu].to(dev)
        # ---- tables at the scene's own extents (kinematic, re-posed once)
        tab = f32([[(t["table"]["x"][0] + t["table"]["x"][1]) / 2, (t["table"]["y"][0] + t["table"]["y"][1]) / 2, 0.0] for t in TT])[tcpu].to(dev)
        tp_ = wp.to_torch(env.scene["table"].root_view.get_transforms()).reshape(N, 7)[:, :3].to(dev) - self.org     # never re-posed
        self.table_err_cm = float((tp_ - torch.tensor([0, 0, -0.02], device=dev)).norm(dim=-1).max() * 100)
        print(f"FP_ENV table placement check (centred at the env origin): max error {self.table_err_cm:.3f} cm", flush=True)
        if self.table_err_cm > 0.1:
            raise RuntimeError(f"fast path scene: a table is not at its env origin ({self.table_err_cm:.2f} cm)")
        self.org = self.org - tab                      # the table frame: world = org + p (the table centre sits at the env origin)
        # ---- DR mass factor on the USD mass (set_masses, inertia not rescaled) and DR friction (mu, 0.9 mu)
        m_usd = wp.to_torch(self.OV.get_masses()).reshape(self.OV.count, -1).sum(-1).cpu()
        mf = f32([[(t["objects"][k]["mass_f"] if k < t["n"] else 1.0) for k in range(K)] for t in TT])[tcpu]
        mexp = f32([[(t["objects"][k]["mass_usd"] if k < t["n"] else float("nan")) for k in range(K)] for t in TT])[tcpu]
        mview = m_usd[idx]                                                                          # (N,K)
        vmask = self.valid.cpu()
        self.mass_mismatch = int((((mview - mexp).abs() > 1e-3 * mexp.abs().clamp(min=1e-3)) & vmask).sum())
        print(f"FP_ENV mass check: {self.mass_mismatch} of {int(vmask.sum())} bodies differ from the template's USD mass", flush=True)
        if (mf != 1).any():
            ms = wp.to_torch(self.OV.get_masses()).clone(); shp = ms.shape                       # cpu
            fac = torch.ones(self.OV.count); fac[idx[vmask]] = mf[vmask]
            ms = (ms.reshape(self.OV.count, -1) * fac[:, None]).reshape(shp).contiguous()
            ids = torch.nonzero(fac != 1).flatten().to(torch.int32).contiguous()
            self.OV.set_masses(wp.from_torch(ms), wp.from_torch(ids))
        mu = f32([[(t["objects"][k]["mu"] if (k < t["n"] and t["objects"][k]["mu"] is not None) else float("nan")) for k in range(K)] for t in TT])[tcpu]
        sel = vmask & torch.isfinite(mu)
        self.mu_mismatch = 0
        if sel.any():                                  # check only: the DR friction was set on the USD materials at spawn
            mat = wp.to_torch(self.OV.get_material_properties())                                  # (count, max_shapes, 3) cpu
            got = mat[idx[sel], 0, 0]
            self.mu_mismatch = int(((got - mu[sel]).abs() > 1e-4).sum())
            self.n_mu_unique = int(torch.unique(mu[sel]).numel())
        try:
            mat = wp.to_torch(self.OV.get_material_properties())
            self.mu_static_range = [float(mat[idx[vmask], 0, 0].min()), float(mat[idx[vmask], 0, 0].max())]
            self.table_mu = wp.to_torch(env.scene["table"].root_view.get_material_properties()).reshape(N, -1, 3)[:, 0].unique(dim=0).tolist()
        except Exception:                              # noqa: BLE001
            self.mu_static_range = None
        print(f"FP_ENV friction check: {self.mu_mismatch} of {int(sel.sum())} DR bodies differ from their template mu (first shape); "
              f"static mu range {self.mu_static_range}; table (static, dynamic, restitution) {getattr(self, 'table_mu', None)}", flush=True)
        self.MV = wp.to_torch(self.OV.get_masses()).reshape(self.OV.count, -1).sum(-1).to(dev)
        self.IV = wp.to_torch(self.OV.get_inertias()).reshape(self.OV.count, 9)[:, [0, 4, 8]].mean(-1).to(dev)
        self.IB = wp.to_torch(self.OV.get_inertias()).reshape(self.OV.count, 3, 3).to(dev)          # body-frame inertia (force mode)
        self.OVel = torch.zeros(self.OV.count, 6, device=dev); self.OT = torch.zeros(self.OV.count, 7, device=dev)
        self.OF = torch.zeros(self.OV.count, 3, device=dev); self.OTq = torch.zeros(self.OV.count, 3, device=dev)
        self.HT = torch.zeros(self.HV.count, 7, device=dev); self.HVel = torch.zeros(self.HV.count, 6, device=dev)
        self.NPART = NPART; self.HF = self.HIDX.reshape(-1)                                   # (N * parts,) env-major
        self.HT[self.HF, :3] = (self.org + torch.tensor([0, 0, PARK_Z], device=dev)).repeat_interleave(HANDS * NPART, 0); self.HT[:, 6] = 1.0
        self.nsteps_total = 0
        P = self._poses()
        sp = (P[..., :3] - self.lay0[..., :3]).norm(dim=-1)[self.valid]
        print(f"FP_ENV spawn check: bodies spawned at their layout, max error {100 * float(sp.max()):.2f} cm", flush=True)
        self._step(int(round(PRE_S / self.dt)))
        P = self._poses()
        self.init_fallen = int(((P[..., 2] < -0.3) & self.valid).sum())
        print(f"FP_ENV init: {self.init_fallen} bodies below the table after the start layouts' settle", flush=True)
        print(f"FP_ENV built: {N} envs, {self.T} templates, K = {K}, {self.OV.count} object bodies ({int(vmask.sum())} real), dt {self.dt:.5f}; "
              f"scene {self.t_scene:.1f} s, sim start {self.t_start:.1f} s, env total {self.t_env:.1f} s", flush=True)

    # ---------------------------------------------------------------------------------------------------------------------------------
    def _check_clone_plan(self):
        import isaaclab.sim as sim_utils
        from isaaclab import cloner
        N, T, K = self.N, self.T, self.K
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        exp = template_of_env(N, T)
        res = {}
        for name in ["table"] + [f"obj{k}" for k in range(K)]:
            proto = np.full(N, -1)
            for j, index in enumerate(cloner.path.get_asset_prototypes(plan, self.env.scene[name].cfg.prim_path)):
                env_ids, _ = cloner.query.get_asset_prototype_unique_world_index(plan.topology, index)
                proto[np.asarray(env_ids)] = j
            res[name] = proto
        p0 = res["obj0"]
        same = all((v == p0).all() for v in res.values())
        seq = bool((p0 == exp).all())
        self.clone_check = dict(all_slots_same_template=bool(same), sequential_map=seq, unassigned=int((p0 < 0).sum()),
                                copies_min=int(np.bincount(p0[p0 >= 0], minlength=T).min()), copies_max=int(np.bincount(p0[p0 >= 0], minlength=T).max()))
        print(f"FP_ENV clone plan: {self.clone_check}", flush=True)
        if not same or not seq or (p0 < 0).any():
            raise RuntimeError(f"fast path scene: clone plan check failed {self.clone_check}")
        return p0

    def _poses(self):
        """(N,K,7) every slot's pose, table frame."""
        tr = self.wp.to_torch(self.OV.get_transforms()).reshape(self.OV.count, 7)[self.IDX]
        return torch_cat_pose(tr, self.org)

    def _step(self, n=1):
        for _ in range(n):
            self.env.sim.step(render=False)
        self.nsteps_total += n

    def _reset_poses(self, layout):
        """Move every real object to its layout pose (table frame; NaN rows skipped), zero velocity; hands parked.  -> parked hand positions."""
        import torch
        wp, dev, N, K = self.wp, self.dev, self.N, self.K
        vm = self.valid & torch.isfinite(layout).all(-1)
        rows = self.IDX[vm]
        self.OT[rows, :3] = layout[vm][:, :3] + self.org[:, None].expand(N, K, 3)[vm]; self.OT[rows, 3:] = layout[vm][:, 3:]
        self.OVel[rows] = 0.0
        if RESET_VIA_ASSETS:                           # per slot asset (write_root_pose / velocity per asset)
            for k in range(K):
                ids = torch.nonzero(vm[:, k]).flatten()
                if len(ids):
                    a = self.env.scene[f"obj{k}"]
                    a.write_root_pose_to_sim_index(root_pose=torch.cat([layout[ids, k, :3] + self.org[ids], layout[ids, k, 3:]], -1), env_ids=ids)
                    a.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=dev), env_ids=ids)
        else:
            rw = wp.from_torch(rows.to(torch.int32).contiguous())
            self.OV.set_transforms(wp.from_torch(self.OT), indices=rw); self.OV.set_velocities(wp.from_torch(self.OVel), indices=rw)
        hall = wp.from_torch(self.HF.to(torch.int32).contiguous())
        park = self.org + torch.tensor([0, 0, PARK_Z], device=dev)
        self.HT[self.HF, :3] = park.repeat_interleave(HANDS * self.NPART, 0); self.HT[self.HF, 3:] = torch.tensor([0, 0, 0, 1.0], device=dev); self.HVel[self.HF] = 0.0
        self.HV.set_transforms(wp.from_torch(self.HT), indices=hall); self.HV.set_velocities(wp.from_torch(self.HVel), indices=hall)
        self._inval()
        return park

    def _inval(self):
        if hasattr(self._PM, "invalidate_transforms"):
            self._PM.invalidate_transforms()

    # ---------------------------------------------------------------------------------------------------------------------------------
    def run(self, layout, carried, path, dyaw, pre_s=PRE_S, tc="batch", settle_s=SETTLE_S, rec=True, effort_c=EFFORT_C, drive_mode="velocity"):
        """layout (N,K,7) root xyz + quat xyzw, table frame (NaN / unused slots ignored); carried (N,) slot index; path (N,33,3) commanded
        root path (its first point = the planned start: the start offset fades out); dyaw (N,) the yaw change along the path.
        tc: "batch" = every env released after the longest carry of the call (+ hold), as in the reference batches; "own" = each env after
        its own; or a (N,) tensor of carry durations (s).
        drive_mode: "velocity" = the reference (the force-capped implicit drive's resulting velocity is written each step); "force" = the
        same capped drive force (incl. the weight hold) and the matching torque (full world inertia tensor) applied as external wrench instead,
        so the solver integrates it with the contacts (not the reference).  -> dict (torch, cuda), keys below."""
        import torch
        assert HANDS == 1, "run() is the single-carry env; with FP_HANDS=2 use fp_chain.ChainSim"
        wp, dev, N, K, dt = self.wp, self.dev, self.N, self.K, self.dt
        tw0 = time.time(); n0 = self.nsteps_total
        layout = torch.as_tensor(layout, dtype=torch.float32, device=dev).clone()
        carried = torch.as_tensor(carried, dtype=torch.long, device=dev)
        path = torch.as_tensor(path, dtype=torch.float32, device=dev)
        dyaw = torch.as_tensor(dyaw, dtype=torch.float32, device=dev)
        assert layout.shape == (N, K, 7) and carried.shape == (N,) and path.shape == (N, NP_PATH, 3) and dyaw.shape == (N,)
        assert bool(self.valid[torch.arange(N, device=dev), carried].all()), "carried slot is an unused (dummy) slot"
        ar = torch.arange(N, device=dev)
        # ---- reset: every real object to its layout pose, zero velocity; hands parked
        park = self._reset_poses(layout)
        hall = wp.from_torch(self.HF.to(torch.int32).contiguous())
        # ---- pre-settle (free physics)
        n_pre = int(round(pre_s / dt)); n_chk = int(round(PRE_CHECK_S / dt))
        P0 = None
        for ii in range(n_pre):
            self._step()
            if ii == n_pre - n_chk:
                P0 = self._poses()
        pose_before = self._poses()
        pre_unstable = torch.full((N,), float("nan"), device=dev)
        if P0 is not None:
            d_ = (pose_before[..., :3] - P0[..., :3]).norm(dim=-1); d_[~self.valid] = 0
            pre_unstable = (d_.amax(1) > DISP_M).float()
        fell = ((pose_before[..., 2] < -0.3) & (layout[..., 2] > -0.1) & self.valid & torch.isfinite(layout).all(-1)).any(1)            # lost the table at the reset (see make_cfg)
        if bool(fell.any()):
            print(f"FP_ENV WARNING: {int(fell.sum())} envs have a body below the table after the pre-settle", flush=True)
        # ---- the commanded carry: arc-length minimum jerk, the start offset faded out, yaw from the current yaw
        ci = self.IDX[ar, carried]                                                                   # (N,) view rows of the carried bodies
        p_now = pose_before[ar, carried, :3]; q_now = pose_before[ar, carried, 3:]
        off = p_now - path[:, 0]
        seg = (path[:, 1:] - path[:, :-1]).norm(dim=-1); sl = torch.cat([torch.zeros(N, 1, device=dev), seg.cumsum(1)], 1)   # (N,33)
        L = sl[:, -1]
        dur = torch.clamp(L / CARRY_SPEED, CARRY_MIN_S, CARRY_MAX_S)
        if isinstance(tc, str):
            Tc = torch.full_like(dur, float(dur.max())) if tc == "batch" else dur.clone()
        else:
            Tc = torch.as_tensor(tc, dtype=torch.float32, device=dev).clone()
        nst = torch.ceil((Tc + HOLD_S) / dt - 1e-6).long()                                           # carry + hold steps per env
        yb = self.yb[ar, carried]
        yaw0 = yaw_zyx(qmul(q_now, qyaw(-yb)))
        nmax = int(nst.max()); n_set = int(round(settle_s / dt))
        tt = torch.arange(nmax + 1, device=dev, dtype=torch.float32) * dt                            # (S+1,)
        u = torch.clamp(tt[:, None] / dur[None], max=1.0); u = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5            # (S+1,N)
        s_ = u * L[None]
        j = torch.searchsorted(sl.contiguous(), s_.T.contiguous(), right=True).T - 1                 # (S+1,N)
        j = j.clamp(0, NP_PATH - 2)
        sl_j = sl.gather(1, j.T).T; sl_j1 = sl.gather(1, (j + 1).T).T
        fr = ((s_ - sl_j) / (sl_j1 - sl_j).clamp(min=1e-12)).clamp(0, 1)
        pj = path[ar[None], j]; pj1 = path[ar[None], j + 1]                                         # (S+1,N,3)
        X = pj + fr[..., None] * (pj1 - pj) + (1 - u)[..., None] * off[None]
        Y = yaw0[None] + u * dyaw[None]
        Q = qmul(qyaw(Y), qyaw(yb)[None].expand(nmax + 1, N, 4))
        V = torch.zeros(nmax + 1, N, 6, device=dev)
        V[:-1, :, :3] = (X[1:] - X[:-1]) / dt; V[:-1, :, 5] = (Y[1:] - Y[:-1]) / dt
        X = X + self.org[None]
        ext_c = self.ext[ar, carried]
        if HAND_MODE == "ball":
            HC = torch.zeros(N, 1, 3, device=dev); HC[:, 0, 2] = ext_c[:, 2] + HAND_R + HAND_GAP; HAX = torch.zeros(N, device=dev)
        else:                                                  # the hand's parts in the carried object's baked frame (gripper.boxes)
            from cleantables.rl.gripper import boxes
            ec = ext_c.detach().cpu().numpy()
            HC = torch.tensor(np.stack([[c for c, _ in boxes(e)] for e in ec]), dtype=torch.float32, device=dev)          # (N, 3, 3)
            HAX = torch.tensor([0.0 if e[0] <= e[1] else math.pi / 2 for e in ec], device=dev)                            # grasp axis turn
        MVs, IVs = self.MV[ci], self.IV[ci]
        IB = self.IB[ci] if drive_mode == "force" else None
        LINg = dict(LIN_D, effort=torch.clamp(effort_c * MVs * G, min=EFFORT_FLOOR)[:, None]) if effort_c > 0 else LIN_D
        GUP = torch.tensor([0.0, 0.0, G], device=dev)
        ciw = wp.from_torch(ci.to(torch.int32).contiguous())
        TE = torch.full((nmax, N), float("nan"), device=dev)
        # ---- recording plan (10 Hz from the carry start, + the release frame(s) of the batch, + the end)
        R_ = int(round(1.0 / (REC_HZ * dt)))
        total = nmax + n_set
        fr_steps = sorted(set(list(range(0, total + 1, R_)) + ([int(nst[0])] if bool((nst == nst[0]).all()) else []) + [total]))
        fset = {s: i for i, s in enumerate(fr_steps)}
        F = len(fr_steps)
        REC = torch.full((N, F, K, 7), float("nan"), device=dev) if rec else None
        if rec:
            REC[:, 0] = pose_before
        pose_rel = torch.full((N, K, 7), float("nan"), device=dev); pose_after = torch.full((N, K, 7), float("nan"), device=dev)

        def drive(cur_v, tgt_x, tgt_v, m_, gains, up=None):
            m_ = m_[:, None]
            vn = (m_ * cur_v + dt * (gains["stiffness"] * tgt_x + gains["damping"] * tgt_v)) / (m_ + dt * gains["damping"] + dt * dt * gains["stiffness"])
            sp = vn.norm(dim=-1, keepdim=True); vn = vn * torch.clamp(gains["vel"] / sp.clamp(min=1e-9), max=1.0)
            Fd = m_ * (vn - cur_v) / dt + (0.0 if up is None else m_ * up)
            fn = Fd.norm(dim=-1, keepdim=True); Fd = Fd * torch.clamp(gains["effort"] / fn.clamp(min=1e-9), max=1.0)
            return cur_v + dt * Fd / m_
        torch.cuda.synchronize(); t_carry0 = time.time()
        for i in range(total):
            if i < nmax:
                act = (i < nst)                                                                      # (N,) still carried / held
                tr_ = wp.to_torch(self.OV.get_transforms()).reshape(self.OV.count, 7); ve_ = wp.to_torch(self.OV.get_velocities()).reshape(self.OV.count, 6)
                x_, q_, v_, w_ = tr_[ci, :3], tr_[ci, 3:7], ve_[ci, :3], ve_[ci, 3:6]
                vlin = drive(v_, X[i] - x_, V[i, :, :3], MVs, LINg, up=GUP)
                TE[i] = torch.where(act, (x_ - X[i]).norm(dim=-1), torch.full_like(x_[:, 0], float("nan")))
                qinv = torch.cat([-q_[:, :3], q_[:, 3:]], -1)
                wang = drive(w_, qrotvec(qmul(Q[i], qinv)), V[i, :, 3:6], IVs, ANG_D)
                a1 = act[:, None]
                if drive_mode == "force":
                    self.OF[ci] = torch.where(a1, MVs[:, None] * (vlin - v_) / dt, torch.zeros_like(v_))
                    Rm = qmat(q_); Iw = Rm @ IB @ Rm.transpose(1, 2)                                   # world inertia (the full tensor)
                    self.OTq[ci] = torch.where(a1, (Iw @ ((wang - w_) / dt)[..., None])[..., 0], torch.zeros_like(w_))
                    self.OV.apply_forces_and_torques_at_position(force_data=wp.from_torch(self.OF.reshape(-1)), torque_data=wp.from_torch(self.OTq.reshape(-1)),
                                                                 position_data=None, indices=ciw, is_global=True)
                else:
                    self.OVel[ci, :3] = torch.where(a1, vlin, v_); self.OVel[ci, 3:] = torch.where(a1, wang, w_)
                    self.OV.set_velocities(wp.from_torch(self.OVel), indices=ciw)
                yw_ = yaw_zyx(q_)                                                                      # the object's baked-frame yaw
                cz, sz = torch.cos(yw_), torch.sin(yw_)
                hx = torch.stack([cz[:, None] * HC[..., 0] - sz[:, None] * HC[..., 1], sz[:, None] * HC[..., 0] + cz[:, None] * HC[..., 1], HC[..., 2]], -1)
                hp = torch.where(a1[:, :, None], x_[:, None] + hx, park[:, None].expand_as(hx))                      # (N, parts, 3)
                self.HT[self.HF, :3] = hp.reshape(-1, 3); self.HT[self.HF, 3:] = qyaw(yw_ + HAX).repeat_interleave(self.NPART, 0)
                self.HVel[self.HF, :3] = torch.where(a1, vlin, torch.zeros_like(vlin)).repeat_interleave(self.NPART, 0); self.HVel[self.HF, 3:] = 0.0
                self.HV.set_transforms(wp.from_torch(self.HT), indices=hall); self.HV.set_velocities(wp.from_torch(self.HVel), indices=hall)
                self._inval()
            elif i == nmax:
                self.HT[self.HF, :3] = park.repeat_interleave(self.NPART, 0); self.HVel[self.HF] = 0.0
                self.HV.set_transforms(wp.from_torch(self.HT), indices=hall); self.HV.set_velocities(wp.from_torch(self.HVel), indices=hall)
                self._inval()
            self._step()
            js = i + 1                                                                               # steps done since the carry start
            need_rel = (nst == js); need_end = (nst + n_set == js)
            if (rec and js in fset) or bool(need_rel.any()) or bool(need_end.any()):
                P = self._poses()
                if rec and js in fset:
                    REC[:, fset[js]] = P
                if bool(need_rel.any()):
                    pose_rel[need_rel] = P[need_rel]
                if bool(need_end.any()):
                    pose_after[need_end] = P[need_end]
        torch.cuda.synchronize(); t_end = time.time()
        TEc = 100 * TE.T                                                                              # (N, nmax) cm, NaN after release
        nanq = lambda q: torch.nanquantile(TEc, q, dim=1)  # noqa: E731
        inv = ~self.valid
        for P in [pose_before, pose_rel, pose_after] + ([REC] if rec else []):
            if P.dim() == 3:
                P[inv] = float("nan")
            else:
                P[inv[:, None].expand(N, P.shape[1], K)] = float("nan")
        ext_c = self.ext[ar, carried]
        out = dict(rec=REC, step_of_frame=torch.tensor(fr_steps, device=dev), t_of_frame=torch.tensor(fr_steps, device=dev) * dt,
                   rel_step=nst, end_step=nst + n_set, pose_before=pose_before, pose_release=pose_rel, pose_after=pose_after,
                   track_med_cm=nanq(0.5), track_p90_cm=nanq(0.9), track_max_cm=torch.nan_to_num(TEc, nan=-1.0).amax(1),
                   mass=MVs, h_a=ext_c[:, 2], l_a=ext_c[:, :2].amax(1), carried=carried, template=torch.as_tensor(self.tpl_of_env, device=dev),
                   valid=self.valid, gap_cm=100 * off.norm(dim=-1), pre_unstable=pre_unstable, fell_at_reset=fell, dur=dur, tc=Tc, path=path, dyaw=dyaw,
                   drive_mode=drive_mode, timing=dict(wall_s=t_end - tw0, carry_settle_wall_s=t_end - t_carry0, steps=self.nsteps_total - n0, n_pre=n_pre,
                               n_carry_max=nmax, n_settle=n_set, dt=dt))
        return out


def torch_cat_pose(tr, org):
    import torch
    return torch.cat([tr[..., :3] - org[:, None], tr[..., 3:7]], -1)


# =====================================================================================================================================
# CLI
# =====================================================================================================================================
def _ledger(what, est_gb, est_time, done=False):
    L = CT_DATA + "/logs/gpu_jobs.log"
    line = f"{time.strftime('%Y-%m-%d %H:%M')} | rl | sim_env {os.getpid()} | {est_gb} | {what} | {'DONE' if done else est_time}\n"
    with open(L, "a") as f:
        f.write(line)


def _gpu_used():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def _merge_json(path, key, val):
    path = Path(path)
    d = json.load(open(path)) if path.exists() else {}
    d[key] = val
    tmp = path.with_suffix(".tmp")
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def _summ(out, tag):
    """CPU summary of a run: tracking, other objects' displacement (pre-settle end -> settled), the carried object's end error."""
    import torch
    N = out["carried"].shape[0]; ar = torch.arange(N, device=out["carried"].device)
    v = out["valid"].clone(); v[ar, out["carried"]] = False
    d = (out["pose_after"][..., :3] - out["pose_before"][..., :3]).norm(dim=-1) * 100
    dv = d[v]
    car_end = (out["pose_after"][ar, out["carried"], :3] - out["path"][:, -1]).norm(dim=-1) * 100
    car_rel = (out["pose_release"][ar, out["carried"], :3] - out["path"][:, -1]).norm(dim=-1) * 100
    f = lambda x: dict(median=float(x.median()), p90=float(torch.quantile(x.float(), 0.9)), max=float(x.max()), mean=float(x.mean()))  # noqa: E731
    push_sum = torch.where(v, d, torch.zeros_like(d)).sum(1)
    return dict(tag=tag, track_med_cm=f(out["track_med_cm"]), track_p90_cm=f(out["track_p90_cm"]), track_max_cm=f(out["track_max_cm"]),
                others_disp_cm=f(dv), others_moved_gt1cm_share=float((dv > 1).float().mean()), push_sum_cm_per_env=f(push_sum),
                carried_end_err_cm=f(car_end), carried_release_err_cm=f(car_rel))


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--setE", action="store_true")
    ap.add_argument("--templates", default=str(DST / "templates_v1.pt"))
    ap.add_argument("--split", default="all", help="bench: all | train | heldout (rows of the templates file)")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--dt", type=int, default=120, help="physics rate (Hz)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", default="", help="bench: save the run outputs (poses, stats; no rec) to this .pt (dt comparison)")
    ap.add_argument("--bench_json", default=str(DST / "bench.json"))
    ap.add_argument("--contact_offset", type=float, default=CONTACT_OFFSET)
    ap.add_argument("--buffers", default="big")
    ap.add_argument("--setE_out", default=str(DST / "setE_fp.pt"))
    ap.add_argument("--drive_mode", default="velocity", help="velocity (= physx_chain) | force (external wrench)")
    add_launcher_args(ap)
    args = ap.parse_args()
    import torch
    if _gpu_used() > 14000:
        raise SystemExit(f"GPU used {_gpu_used()} MiB > 14000 MiB: not starting (GPU rules)")
    dt = 1.0 / args.dt
    if args.setE:
        J = np.load(SETE / "rec_chains.npz", allow_pickle=True); Sc = np.load(SETE / "physx_c3" / "scores.npz", allow_pickle=True)
        twins = [str(t) for t in J["twins"]]
        tpl = build_templates(twins, meta=dict(kind="setE", source=str(SETE)))
        T = len(twins)
        SO, PA, DY, CT = J["step_obj"], J["path"].astype(np.float64), J["dyaw"], J["chain_twin"]
        cs = [(c, s) for c in range(SO.shape[0]) for s in range(SO.shape[1]) if SO[c, s] >= 0]
        # reference carry duration: every chain was in one batch (copies = 3 = chains per twin), so step s of every chain is released
        # after the longest step-s carry of all chains (+ hold)
        def dur_of(p):
            L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum(); return min(CARRY_MAX_S, max(CARRY_MIN_S, L / CARRY_SPEED))
        Tc_s = {s: max(CARRY_MIN_S, max(dur_of(PA[c, s]) for c in range(SO.shape[0]) if SO[c, s] >= 0)) for s in range(SO.shape[1]) if (SO[:, s] >= 0).any()}
        per_tw = np.bincount([CT[c] for c, s in cs], minlength=T)
        copies = int(per_tw.max())
        N = copies * T
        cfg = make_cfg(tpl, N, dt, contact_offset=args.contact_offset, buffers=args.buffers)
        watch = _GpuWatch()
        _ledger(f"fp_env --setE ({len(cs)} carries, {N} envs, dt 1/{args.dt})", 3, "~5 min")
        with launch_simulation(cfg, args):
            fp = FastPath(cfg)
            K = fp.K; tdx = fp.tpl_of_env
            envs_of = {t: list(np.flatnonzero(tdx == t)) for t in range(T)}
            lay = np.zeros((N, K, 7), np.float32); car = np.zeros(N, np.int64); pth = np.zeros((N, 33, 3), np.float32)
            dyw = np.zeros(N, np.float32); tcs = np.zeros(N, np.float32); which = np.full((N, 2), -1)
            used = {t: 0 for t in range(T)}
            for c, s in cs:
                t = int(CT[c]); e = envs_of[t][used[t]]; used[t] += 1
                n = tpl["templates"][t]["n"]
                lay[e] = np.nan; lay[e, :n] = Sc["pose_pre"][c, s, :n]
                car[e], pth[e], dyw[e], tcs[e], which[e] = SO[c, s], PA[c, s], DY[c, s], Tc_s[s], (c, s)
            for t in range(T):                                                                  # idle copies: repeat the template's first carry
                src = next(e for e in envs_of[t] if which[e, 0] >= 0)
                for e in envs_of[t][used[t]:]:
                    lay[e], car[e], pth[e], dyw[e], tcs[e] = lay[src], car[src], pth[src], dyw[src], tcs[src]
            out = fp.run(lay, car, pth, dyw, pre_s=0.0, tc=torch.tensor(tcs), drive_mode=args.drive_mode)
            ok = torch.as_tensor(which[:, 0] >= 0)
            res = {k: (v[ok.to(v.device)].cpu() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == N else (v.cpu() if torch.is_tensor(v) else v))
                   for k, v in out.items()}
            cc, ss = which[ok.numpy(), 0], which[ok.numpy(), 1]
            nxt = np.array([Sc["pose_pre"][c, s + 1] if (s + 1 < SO.shape[1] and SO[c, s + 1] >= 0) else Sc["pose_end"][c] for c, s in zip(cc, ss)])
            res.update(chain=torch.as_tensor(cc), step=torch.as_tensor(ss), twin=[twins[int(CT[c])] for c in cc],
                       ref_pose_before=torch.as_tensor(Sc["pose_pre"][cc, ss]), ref_pose_after=torch.as_tensor(nxt),
                       ref_after_is_chain_end=torch.as_tensor([not (s + 1 < SO.shape[1] and SO[c, s + 1] >= 0) for c, s in zip(cc, ss)]),
                       ref_track_med_cm=torch.as_tensor(Sc["track_med_cm"][cc, ss]), ref_track_p90_cm=torch.as_tensor(Sc["track_p90_cm"][cc, ss]),
                       ref_track_max_cm=torch.as_tensor(Sc["track_max_cm"][cc, ss]), ref_settle_cm=torch.as_tensor(Sc["settle_cm"][cc, ss]),
                       ref_settings=str(Sc["settings"]), clone_check=fp.clone_check, mass_mismatch=fp.mass_mismatch, twins=twins,
                       settings=dict(dt=fp.dt, pre_s=0.0, drive_mode=args.drive_mode, tc="physx_chain batch rule (max dur of step s over all 30 chains)", EFFORT_C=EFFORT_C,
                                     EFFORT_FLOOR=EFFORT_FLOOR, contact_offset=args.contact_offset, LIN=LIN_D, ANG=ANG_D, HOLD_S=HOLD_S,
                                     SETTLE_S=SETTLE_S, HAND_R=HAND_R, HAND_GAP=HAND_GAP, CARRY_SPEED=CARRY_SPEED, REC_HZ=REC_HZ),
                       summary=_summ(out, "setE")); res["summary"]["n_real"] = int(ok.sum())
            DST.mkdir(parents=True, exist_ok=True)
            torch.save(res, args.setE_out)
            print("FP_ENV setE saved", args.setE_out, json.dumps(res["summary"]), flush=True)
            watch.stop()
            _ledger(f"fp_env --setE ({len(cs)} carries)", 3, "", done=True)
            fp.env.close()
        return
    # ---------------------------------------------------------------------------------------------------------------------- bench
    tpl = torch.load(args.templates, map_location="cpu", weights_only=False)
    if args.split != "all":
        keep = [i for i, t in enumerate(tpl["templates"]) if tpl["meta"]["split"][t["name"]] == args.split]
        tpl = dict(tpl, templates=[tpl["templates"][i] for i in keep])
    T = len(tpl["templates"]); N = args.n
    key = args.tag or f"N{N}_T{T}_dt{args.dt}"
    est_gb = round(2 + N * 0.0015, 1)
    _ledger(f"fp_env --bench {key}", est_gb, "~10 min")
    watch = _GpuWatch()
    cfg = make_cfg(tpl, N, dt, contact_offset=args.contact_offset, buffers=args.buffers)
    with launch_simulation(cfg, args):
        torch.cuda.reset_peak_memory_stats()
        fp = FastPath(cfg)
        lay, car, pth, dyw = synth_paths(tpl, fp.tpl_of_env, seed=args.seed)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(100):
            fp._step()
        torch.cuda.synchronize(); raw = 100 / (time.time() - t0)
        reps = []; outs = []
        for r in range(args.reps):
            o = fp.run(lay, car, pth, dyw); outs.append({k: o[k] for k in ("pose_after", "track_med_cm", "carried")})
            tm = o["timing"]
            reps.append(dict(wall_s=tm["wall_s"], steps=tm["steps"], phys_steps_per_s=tm["steps"] / tm["wall_s"],
                             env_steps_per_s=N * tm["steps"] / tm["wall_s"], carries_per_s=N / tm["wall_s"], n_carry_max=tm["n_carry_max"]))
            sm = _summ(o, f"rep{r}"); reps[-1]["summary"] = {k_: sm[k_] for k_ in ("track_med_cm", "carried_end_err_cm", "others_disp_cm", "others_moved_gt1cm_share")}
            print(f"FP_ENV bench {key} rep {r}: {reps[-1]}", flush=True)
        rep_diff = None
        if len(outs) > 1:                                                     # the same inputs twice: GPU PhysX repeatability
            ar = torch.arange(N, device=o["carried"].device)
            d = (outs[-1]["pose_after"][..., :3] - outs[0]["pose_after"][..., :3]).norm(dim=-1) * 100
            dc = d[ar, o["carried"]]; v = fp.valid.clone(); v[ar, o["carried"]] = False; do = d[v]
            q = lambda x: dict(median=float(x.median()), p90=float(torch.quantile(x, 0.9)), max=float(x.max()), share_gt1cm=float((x > 1).float().mean()))  # noqa: E731
            rep_diff = dict(carried_end_cm=q(dc), others_end_cm=q(do), track_med_cm_absdiff=q((outs[-1]["track_med_cm"] - outs[0]["track_med_cm"]).abs()))
        res = dict(N=N, T=T, dt=fp.dt, repeat_diff_first_vs_last=rep_diff, mu_mismatch=fp.mu_mismatch, init_fallen=fp.init_fallen, templates=args.templates, split=args.split, clone_check=fp.clone_check, mass_mismatch=fp.mass_mismatch,
                   n_bodies=fp.OV.count, n_real_bodies=int(fp.valid.sum()), stage_build_s=fp.t_scene, sim_start_s=fp.t_start, env_init_s=fp.t_env,
                   raw_phys_steps_per_s=raw, raw_env_steps_per_s=raw * N, reps=reps,
                   best=dict(phys_steps_per_s=max(r["phys_steps_per_s"] for r in reps), carries_per_s=max(r["carries_per_s"] for r in reps),
                             env_steps_per_s=max(r["env_steps_per_s"] for r in reps)),
                   peak_torch_alloc_mb=torch.cuda.max_memory_allocated() / 2 ** 20, peak_torch_reserved_mb=torch.cuda.max_memory_reserved() / 2 ** 20,
                   peak_nvsmi_proc_mb=watch.peak_proc, peak_nvsmi_total_mb=watch.peak_total, summary=_summ(o, key),
                   contact_offset=args.contact_offset, buffers=args.buffers, time=time.strftime("%Y-%m-%d %H:%M"))
        _merge_json(args.bench_json, key, res)
        print(f"FP_ENV bench {key}: {json.dumps(res)}", flush=True)
        if args.save:
            torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in o.items() if k != "rec"}, args.save)
        watch.stop()
        _ledger(f"fp_env --bench {key}", est_gb, "", done=True)
        fp.env.close()


if __name__ == "__main__":
    import sys
    if "--select" in sys.argv:                                          # CPU: write the template file (no Isaac)
        ap = argparse.ArgumentParser(); ap.add_argument("--select", default="v1"); ap.add_argument("--templates", required=True)
        ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args()
        tr, ho = select_v1(seed=a.seed)
        split = {**{s: "train" for s in tr}, **{s: "heldout" for s in ho}}
        r = build_templates(tr + ho, out=a.templates, meta=dict(select="v1", seed=a.seed, train=tr, heldout=ho, split=split))
        print(f"templates: {len(r['templates'])} ({len(tr)} train, {len(ho)} held-out), K = {r['k_max']} -> {a.templates}")
    else:
        main()
