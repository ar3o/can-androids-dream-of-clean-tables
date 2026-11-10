"""Renders of composite layouts: objects of other episodes added onto a base twin by the world model's training sampler.

Each layout is rendered with the robot hidden, from the base twin's phone camera, one still per layout and render variant.
v0 is the robot's run-time look; v >= 1 adds the domain randomisation of cleantables.features.paired_render (lights, table and
background textures, exposure); v >= 4 adds the full randomisation of cleantables.features.randomize (camera pose and zoom, object
sizes and colours, light, auto-exposure, sensor).

One Isaac process per (base episode, variant): the base twin's objects plus every object any of its layouts adds (spawned once, id
"<src_ep>__<obj_id>"); per layout, the objects it uses are posed (x, y, z_bottom, yaw_off: root = bottom centre, yaw on top of the baked
orientation) and every other spawned object is parked far off camera.

  source scripts/env.sh; $SIM_PY -m cleantables.features.composite_render --layouts <layouts.json> --base IMG_2516_t05 --variant 1
Out: $CLEANTABLES_DATA/vjepa_wm/composite_renders/<base>/v<n>/<layout id>.jpg + meta.json
"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import os
import zlib

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--layouts", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--variant", type=int, default=0)
ap.add_argument("--twins", default="data/twin_table/ep/_scalefix")
ap.add_argument("--out", default=CT_DATA + "/vjepa_wm/composite_renders")
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
OUT = os.path.join(args.out, args.base, f"v{args.variant}")
if os.path.exists(os.path.join(OUT, "meta.json")):
    print("exists, skip", OUT)
    raise SystemExit(0)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from cleantables.twin.ep import rtx  # noqa: E402
from cleantables.twin.robot.env_kuka import ROOT, TwinRobotEnv, TwinRobotEnvCfg, build_scene_cfg  # noqa: E402

LAY = [l for l in json.load(open(args.layouts)) if l["base_ep"] == args.base]
if not LAY:
    raise SystemExit(f"no layouts with base {args.base}")
S = json.load(open(os.path.join(ROOT, args.twins, args.base, "scene.json")))
base_ids = {o["id"] for o in S["objects"]}
extra = {}
for l in LAY:
    for o in l["objects"]:
        if o["src_ep"] != args.base or o["obj_id"] not in base_ids:
            key = f"{o['src_ep']}__{o['obj_id']}"
            if key not in extra:
                src = {x["id"]: x for x in json.load(open(os.path.join(ROOT, args.twins, o["src_ep"], "scene.json")))["objects"]}[o["obj_id"]]
                extra[key] = dict(src, id=key, pos=[50.0 + 0.5 * len(extra), 50.0, 0.0])
S = dict(S, objects=S["objects"] + list(extra.values()))
ids = [o["id"] for o in S["objects"]]
PARK = {i: np.array([60.0 + 0.6 * k, -60.0, 0.0]) for k, i in enumerate(ids)}
rng = np.random.default_rng(zlib.crc32(f"composite/{args.base}/{args.variant}".encode()))
V = args.variant


def tint():
    t = np.array([1.0, rng.uniform(0.86, 1.0), rng.uniform(0.72, 1.0)])
    if rng.random() < 0.4:
        t = t[::-1]
    return tuple((t / t.max()).tolist())


def jitter_png(src, dst):
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    alpha = im[..., 3:] if im.shape[-1] == 4 else None
    hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] *= rng.uniform(0.85, 1.15)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    m = out.mean((0, 1)); out = np.clip((out - m) * rng.uniform(0.85, 1.15) + m, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, out if alpha is None else np.concatenate([out, alpha], -1))


DR = dict(variant=V)
if V > 0:
    src = os.path.join(ROOT, "data/twin_table", S["episode"])
    vdir = os.path.join(os.path.dirname(args.out.rstrip("/")), "dr_twins", f"composite_{args.base}", f"v{V}")
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
    S = dict(S, episode=vdir)
    DR.update(dome=float(rng.uniform(200, 1000)), dome_color=tint(), key=float(rng.uniform(150, 1100)), key_color=tint(),
              key_angle=float(rng.uniform(10, 40)), key_rot_deg=rng.uniform(-25, 25, 2).tolist(),
              gain=float(rng.uniform(0.8, 1.25)), gamma=float(rng.uniform(0.9, 1.1)))
if V >= 4:                                                   # full randomisation (randomize.py): camera pose + zoom, sizes, object colours, sensor
    from cleantables.features import randomize as dr
    S, DRf = dr.full(S, np.random.default_rng(zlib.crc32(f"composite/{args.base}/{V}/full".encode())), vdir, ROOT, DR)
    DR.update(DRf)

cfg = TwinRobotEnvCfg()
cfg.scene = build_scene_cfg(1, S, {a: (tuple(p[0]), tuple(p[1])) for a, p in S["poses"].items()}, record_cam=False, arm_contact=False)
if V > 0:
    from dataclasses import replace
    cfg.scene.dome.spawn.intensity, cfg.scene.dome.spawn.color = DR["dome"], DR["dome_color"]
    cfg.scene.key.spawn.intensity, cfg.scene.key.spawn.color, cfg.scene.key.spawn.angle = DR["key"], DR["key_color"], DR["key_angle"]
    q0 = R.from_quat(np.r_[cfg.scene.key.init_state.rot[1:], cfg.scene.key.init_state.rot[:1]])
    q = (R.from_euler("xy", DR["key_rot_deg"], degrees=True) * q0).as_quat()
    cfg.scene.key.init_state = replace(cfg.scene.key.init_state, rot=(q[3], q[0], q[1], q[2]))
if V >= 4:
    dr.apply_sizes(cfg.scene, S, DR)
rtx.add_cameras(cfg, S, ROOT, wrist_names=())


def state(lay):
    st = {}
    for i in ids:
        M = np.eye(4); M[:3, 3] = PARK[i]; st[i] = M
    used = []
    for o in lay["objects"]:
        key = o["obj_id"] if (o["src_ep"] == args.base and o["obj_id"] in base_ids) else f"{o['src_ep']}__{o['obj_id']}"
        M = np.eye(4); M[:3, :3] = R.from_euler("z", o["yaw_off"]).as_matrix(); M[:3, 3] = [o["x"], o["y"], o["z_bottom"]]
        st[key] = M; used.append(key)
    return st, used


with launch_simulation(cfg, args):
    import omni.usd
    import torch
    from pxr import UsdGeom

    env = TwinRobotEnv(cfg, S)
    dev = env.device
    rtx.hide_ground()
    env.reset()
    rtx.set_static_cams(env, S, dev)
    for prim in omni.usd.get_context().get_stage().Traverse():
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
        for _ in range(3):
            env.sim.render()
        im = rtx.grab(env, "main")
        if im.mean() < 3:
            raise SystemExit("black frame: RTX render failed (GPU memory?)")
        if V > 0:
            im = np.clip(255 * (np.clip(im.astype(np.float32) * DR["gain"], 0, 255) / 255) ** DR["gamma"], 0, 255).astype(np.uint8)
        if V >= 4:
            im = dr.post(im, DR, frame)
        return im

    os.makedirs(OUT, exist_ok=True)
    render(state(LAY[0])[0])
    rows = []
    for n_, l in enumerate(LAY):
        st, used = state(l)
        cv2.imwrite(os.path.join(OUT, f"{l['id']}.jpg"), render(st, n_), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(id=l["id"], objects=used, poses={u: np.r_[st[u][:3, 3], R.from_matrix(st[u][:3, :3]).as_quat()].round(5).tolist()
                                                          for u in used}))
    json.dump(dict(base=args.base, variant=V, dr=DR, layouts=rows, extra_objects=list(extra)), open(os.path.join(OUT, "meta.json.tmp"), "w"),
              default=float)
    os.replace(os.path.join(OUT, "meta.json.tmp"), os.path.join(OUT, "meta.json"))
    print("WROTE", OUT, len(LAY), "layouts", flush=True)
    env.close()
