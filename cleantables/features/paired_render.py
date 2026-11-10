"""Paired renders: every human episode re-rendered in its _scalefix twin from the phone camera, robot hidden, following the
tracked object motion, at the same work frames k as the real video frames the world model encodes, so each real frame has rendered
counterparts. Variant 0 is the robot's run-time look (the twin env's default lights, no randomisation); variants >= 1 use generic
domain randomisation, seeded by (episode, variant):
  lights    dome intensity 200-1000, key (distant) intensity 150-1100, key direction +-25 deg, both colours = a random warm/cool tint
  textures  table texture and background plate: hue +-8 deg, saturation x0.8-1.2, value x0.85-1.15, contrast x0.85-1.15
  exposure  after rendering: gain x0.8-1.25, gamma 0.9-1.1 (camera exposure / response)
Variants >= 4 add the full randomisation of cleantables.features.randomize: camera pose and zoom, object sizes and colours, wider light direction, light total,
auto-exposure, sensor effects. No task rule: the same ranges for every scene.

Poses: --pose_npz with k (n,), ids (m,), pose (n, m, 4) = x, y, z_bottom, yaw_off (the twins' convention: root = bottom centre, yaw on
top of the baked orientation); objects not in the file keep their start pose.

  source scripts/env.sh; $SIM_PY -m cleantables.features.paired_render --ep IMG_2516_t05 --pose_npz <ep>.npz --variant 1
Out: $CLEANTABLES_DATA/vjepa_wm/paired_renders/<ep>/v<variant>/<k:05d>.jpg + meta.json (k, video time, DR params)
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import os
import zlib

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--ep", required=True)
ap.add_argument("--pose_npz", required=True)
ap.add_argument("--variant", type=int, default=0)
ap.add_argument("--twins", default="data/twin_table/ep/_scalefix")
ap.add_argument("--out", default=CT_DATA + "/vjepa_wm/paired_renders")
ap.add_argument("--limit", type=int, default=0, help="render only the first n frames (tests)")
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
OUT = os.path.join(args.out, args.ep, f"v{args.variant}")
if os.path.exists(os.path.join(OUT, "meta.json")):
    print("exists, skip", OUT)
    raise SystemExit(0)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from cleantables.twin.ep import rtx  # noqa: E402
from cleantables.twin.robot.env_kuka import ROOT, TwinRobotEnv, TwinRobotEnvCfg, build_scene_cfg  # noqa: E402

S = json.load(open(os.path.join(ROOT, args.twins, args.ep, "scene.json")))
Z = np.load(args.pose_npz, allow_pickle=True)
K, IDS, POSE = Z["k"].astype(int), [str(x) for x in Z["ids"]], Z["pose"].astype(np.float64)
if args.limit:
    K, POSE = K[:args.limit], POSE[:args.limit]
ids = [o["id"] for o in S["objects"]]
rng = np.random.default_rng(zlib.crc32(f"{args.ep}/{args.variant}".encode()))
V = args.variant


def tint():
    t = np.array([1.0, rng.uniform(0.86, 1.0), rng.uniform(0.72, 1.0)])           # warm .. neutral
    if rng.random() < 0.4:
        t = t[::-1]                                                                    # cool
    return tuple((t / t.max()).tolist())


def jitter_png(src, dst):
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    alpha = im[..., 3:] if im.shape[-1] == 4 else None
    hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180                            # +-8 deg (OpenCV hue = deg / 2)
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] *= rng.uniform(0.85, 1.15)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    m = out.mean((0, 1)); out = np.clip((out - m) * rng.uniform(0.85, 1.15) + m, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, out if alpha is None else np.concatenate([out, alpha], -1))


DR = dict(variant=V)
if V > 0:                                                    # textures: a variant twin folder (table + background), the rest unchanged
    src = os.path.join(ROOT, "data/twin_table", S["episode"])
    vdir = os.path.join(os.path.dirname(args.out.rstrip("/")), "dr_twins", args.ep, f"v{V}")
    os.makedirs(vdir, exist_ok=True)
    for usda in ("table.usda", S["background"]["usd"]):
        txt = open(os.path.join(src, usda)).read()
        for tex in sorted(set(t.split("@")[0] for t in txt.split("asset inputs:file = @")[1:] + txt.split("emissive_color_texture = @")[1:])):
            if tex.endswith((".png", ".jpg")) and os.path.exists(tex):
                dst = os.path.join(vdir, ("table_" if usda == "table.usda" else "bg_") + os.path.basename(tex))
                if not os.path.exists(dst):
                    jitter_png(tex, dst)
                txt = txt.replace(tex, dst)
        open(os.path.join(vdir, usda), "w").write(txt)
    S = dict(S, episode=vdir)                                # build_scene_cfg / add_cameras read table + background from here
    DR.update(dome=float(rng.uniform(200, 1000)), dome_color=tint(), key=float(rng.uniform(150, 1100)), key_color=tint(),
              key_angle=float(rng.uniform(10, 40)), key_rot_deg=rng.uniform(-25, 25, 2).tolist(),
              gain=float(rng.uniform(0.8, 1.25)), gamma=float(rng.uniform(0.9, 1.1)))
if V >= 4:                                                   # full randomisation (randomize.py): camera pose + zoom, sizes, object colours, sensor
    from cleantables.features import randomize as dr
    S, DRf = dr.full(S, np.random.default_rng(zlib.crc32(f"{args.ep}/{V}/full".encode())), vdir, ROOT, DR)
    DR.update(DRf)

cfg = TwinRobotEnvCfg()
cfg.scene = build_scene_cfg(1, S, {a: (tuple(p[0]), tuple(p[1])) for a, p in S["poses"].items()}, record_cam=False, arm_contact=False)
if V > 0:
    from dataclasses import replace
    cfg.scene.dome.spawn.intensity, cfg.scene.dome.spawn.color = DR["dome"], DR["dome_color"]
    cfg.scene.key.spawn.intensity, cfg.scene.key.spawn.color, cfg.scene.key.spawn.angle = DR["key"], DR["key_color"], DR["key_angle"]
    q0 = R.from_quat(np.r_[cfg.scene.key.init_state.rot[1:], cfg.scene.key.init_state.rot[:1]])          # wxyz -> xyzw
    q = (R.from_euler("xy", DR["key_rot_deg"], degrees=True) * q0).as_quat()
    cfg.scene.key.init_state = replace(cfg.scene.key.init_state, rot=(q[3], q[0], q[1], q[2]))
if V >= 4:
    dr.apply_sizes(cfg.scene, S, DR)
rtx.add_cameras(cfg, S, ROOT, wrist_names=())


def state(row):
    st = {o["id"]: np.eye(4) for o in S["objects"]}
    for o in S["objects"]:
        st[o["id"]][:3, 3] = o["pos"]
    for j, i in enumerate(IDS):
        if i in st and np.isfinite(row[j]).all():
            M = np.eye(4); M[:3, :3] = R.from_euler("z", row[j][3]).as_matrix(); M[:3, 3] = row[j][:3]
            st[i] = M
    return st


with launch_simulation(cfg, args):
    import omni.usd
    import torch
    from pxr import UsdGeom

    env = TwinRobotEnv(cfg, S)
    dev = env.device
    rtx.hide_ground()
    env.reset()
    rtx.set_static_cams(env, S, dev)
    for prim in omni.usd.get_context().get_stage().Traverse():           # robots and pedestals hidden before the first render
        p = str(prim.GetPath())
        if p.count("/") == 4 and p.startswith("/World/envs/env_0/") and any(s in p for s in ("/Robot", "/Ped")):
            UsdGeom.Imageable(prim).MakeInvisible()
    ids1, org = torch.arange(1, device=dev), env.scene.env_origins

    def render(st, frame=0):
        for k_, i in enumerate(ids):
            M = st[i]
            p = torch.zeros(1, 7, device=dev)
            p[0, :3] = torch.tensor(M[:3, 3], device=dev, dtype=torch.float32) + org[0]
            p[0, 3:] = torch.tensor(R.from_matrix(M[:3, :3]).as_quat(), device=dev, dtype=torch.float32)
            env.objs[k_].write_root_pose_to_sim_index(root_pose=p, env_ids=ids1)
            env.objs[k_].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids1)
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)
        for _ in range(2):
            env.sim.render()
        im = rtx.grab(env, "main")
        if im.mean() < 3:
            raise SystemExit("black frame: RTX render failed (GPU memory?)")
        if V > 0:                                                          # camera exposure
            im = np.clip(255 * (np.clip(im.astype(np.float32) * DR["gain"], 0, 255) / 255) ** DR["gamma"], 0, 255).astype(np.uint8)
        if V >= 4:
            im = dr.post(im, DR, frame)
        return im

    os.makedirs(OUT, exist_ok=True)
    render(state(POSE[0]))                                                 # warm-up
    for k, row in zip(K, POSE):
        cv2.imwrite(os.path.join(OUT, f"{int(k):05d}.jpg"), render(state(row), k), [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = dict(ep=args.ep, video=S.get("video"), twin=os.path.join(args.twins, args.ep), pose_npz=args.pose_npz, k=K.tolist(),
                files=[f"{int(k):05d}.jpg" for k in K], dr=DR)
    json.dump(meta, open(os.path.join(OUT, "meta.json.tmp"), "w"), default=float)
    os.replace(os.path.join(OUT, "meta.json.tmp"), os.path.join(OUT, "meta.json"))
    print("WROTE", OUT, len(K), "frames", flush=True)
    env.close()
