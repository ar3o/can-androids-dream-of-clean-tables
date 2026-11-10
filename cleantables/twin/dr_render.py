"""One real human demonstration replayed in its Isaac twin and in domain-randomised photoreal worlds (Isaac RTX), so that each real
video yields many varied views of the same object motion.
  motion   the object motion the world model watches: SAM 3 tracker positions (wm_v3_tracks/tracks_all.npz, every 0.3 s) for this
           episode, a 3-sample median for display, linear in between; z as the rise above the start; yaw from the tracker for
           elongated objects only (round ones keep their start yaw)
  physics  PhysX, gravity on: every object is its twin's rigid body (the twin's own collision hulls, mass, friction) on an invisible
           collider of the twin's table + a ground plane. While a published contact model says a hand touches it (HOIST-Former or
           Hands23, each at its own 0.5 threshold: the world model's two contact inputs; held within +-0.6 s of a touch; height = max of
           3 samples), the object is velocity-driven along its track with physical collisions (rising by 1.5 x its sideways lag when
           blocked, as a hand lifts over an obstacle; starting from the twin's own layout, the offset to the tracks fading out over its
           first held stretch); otherwise it is free: it falls, collides, settles. The hands are not drawn (see --grippers / --arms).
  twin     world "twin": the twin as built (background.usda static plate + table.usda + the object twins) from the phone camera
  rooms    (rtx_dr_ranges room_p) the world is a whole 3-D room from the library (Isaac Environments), the table placed automatically
           on free floor (Renderer.room), a table fitted under the twin's top, a ceiling-panel light, a wider camera
  hands    --grippers: two Franka grippers follow the WiLoR hands (Gripper); --arms: on two Franka arms at the demonstrator's shoulders (IK)
  mood     --mood lamp|sun|flood: replaces the drawn lights with one strongly different lighting look (MOODS): lamp = a dark room lit by
           one warm pendant lamp low over the table; sun = a bright low sun behind the table, a glossy top (specular glint); flood = the
           room's own lights + a floodlight from the side (hard shadows). Placed from the camera direction.
  seeds    automatic worlds, one per seed, drawn from the asset library (dr_library.py -> rtx_dr_library.json) and the ranges in
           rtx_dr_ranges.json, with nothing placed by hand: the background (the static plate) -> a random Isaac sky HDRI, exposure set
           from its measured median brightness; the twin's static mesh -> a random Base MDL; a random key light + maybe one more light,
           with colour temperatures; the camera around the real phone pose (angle, distance, FOV, a slow sweep); each object -> a random
           library model of the same perceived category, weighted by box-shape similarity and auto-rotated / fit to the twin's box,
           sometimes a random material (render only)
The world model is trained on the real videos and their twins at the object level, not on these renders. Output per world: an mp4 of
[real video | plain twin] beside the world, and a still.
  $SIM_PY -m cleantables.twin.dr_render --episode IMG_2516_t11 --out DIR --seeds 1,2,3 [--twin] [--stills 0.2,0.7]"""
from __future__ import annotations

from cleantables import CT_DATA
import argparse
import json
import math
import re
import os
import subprocess
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--episode", default="IMG_2516_t11"); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default=""); ap.add_argument("--twin", action="store_true", help="also render the plain twin panel (first)")
ap.add_argument("--library", default=str(Path(__file__).with_name("rtx_dr_library.json")))
ap.add_argument("--ranges", default=str(Path(__file__).with_name("rtx_dr_ranges.json")))
ap.add_argument("--tracks", default=CT_DATA + "/wm_eval/wm_v3_tracks/tracks_all.npz")
ap.add_argument("--video", default="")
ap.add_argument("--sky_cache", default=CT_DATA + "/wm_eval/wm_v2_seg/videos/stageA/rtx_dr/assets/skies2k")
ap.add_argument("--w", type=int, default=1280); ap.add_argument("--h", type=int, default=720); ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--frames_only", action="store_true", help="write each world's raw renders to frames_<episode>_<world>/ (for compose_sbs.py), no mp4")
ap.add_argument("--stills", default="", help="comma list of clip fractions: render only these frames as jpgs (a look check, no mp4)")
ap.add_argument("--grippers", action="store_true", help="two floating Franka grippers follow the WiLoR hands (Gripper below)")
ap.add_argument("--zoom", type=float, default=1.0, help="camera distance x this (< 1 = closer)")
ap.add_argument("--place", default="any", help="room worlds: limit the room draw to this kind (a word in the room's path, e.g. kitchen, warehouse) or 'outdoor'")
ap.add_argument("--wm_input", action="store_true", help="the world model's INPUT view: the real phone camera, no grippers / arms, and each "
                "object's mask per frame (masks_<episode>_<world>/NNNNN.png, value k + 1 = scene object k) for wm_input_feats.py; with --frames_only")
ap.add_argument("--wm_cam", default="phone", choices=["phone", "hero"], help="--wm_input camera: the real phone pose, or the room's drawn "
                "hero view held still (landscape; the room in view); its K / pose go to camera_<episode>_<world>.json")
ap.add_argument("--init_state", default="", help="plan mode (wm_plan_rollout.py): start the objects from this state (root_pos, root_quat "
                "xyzw per scene object, e.g. the previous step's end) instead of the twin's layout; tracks pair to it")
ap.add_argument("--stance", default="", help="fix the arms' shoulders / facing (json: shoulder{left,right}, base_yaw) instead of reading them from the hands")
ap.add_argument("--kinematic", action="store_true", help="physics off: every object is kinematic and placed exactly on its track "
                "each frame (no gravity, no collisions; a carried object follows its path through anything)")
ap.add_argument("--carry_exact", action="store_true", help="physics ON, but a HELD object is placed exactly on its track with its collisions "
                "off (it passes through anything, as the WM's path does); let go, its collisions return and it falls / settles under PhysX")
ap.add_argument("--frame_list", default="", help="a json list of frame indices (at --fps from the first tracked sample) to render; the "
                "others are only simulated (DR training renders: the stills each keyframe's held clip needs)")
ap.add_argument("--no_room", action="store_true", help="skip the room draw: the sky-plate world (HDRI background, re-materialled twin table)")
ap.add_argument("--sky_name", default="", help="use the library sky whose file name contains this (test worlds)")
ap.add_argument("--cam_el", type=float, default=None, help="orbit / hero camera elevation (deg), e.g. 70 for a near top-down view (test worlds)")
ap.add_argument("--same_objects", action="store_true", help="the twin objects as scanned: no library model swaps, no re-materialling")
ap.add_argument("--mood", default="none", choices=["none", "lamp", "sun", "flood", "night", "neon", "torch"], help="a lighting look instead of the drawn lights (MOODS)")
ap.add_argument("--arms", action="store_true", help="the grippers on two Franka arms at the demonstrator's shoulders (IK; implies --grippers)")
ap.add_argument("--hands", default=CT_DATA + "/perception_v2/hands/{ep}/hands_current.npz")
add_launcher_args(ap)
args = ap.parse_args()
args.grippers = args.grippers or args.arms
args.enable_cameras = True
args.headless = True

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")                    # some library skies are .exr
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TWIN = ROOT / "data/twin_table/ep/_scalefix" / args.episode
HAP = 20.955
CAP_H = 120
LW = 640                                                                        # the left column (real | twin) width


def main():
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass, instantiate
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        cam = CameraCfg(prim_path="{ENV_REGEX_NS}/WDCam", update_period=0.0, data_types=["rgb"] + (["semantic_segmentation"] if args.wm_input else []),
                        width=args.w, height=args.h, spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=HAP, clipping_range=(0.01, 80.0)),
                        renderer_cfg=IsaacRtxRendererCfg(colorize_semantic_segmentation=False, global_settings=IsaacRtxRendererGlobalSettingsCfg(
                            antialiasing_mode=os.environ.get("SA_AA", "DLAA"), enable_dl_denoiser=os.environ.get("SA_DENOISE", "1") == "1",
                            enable_dlssg=False, samples_per_pixel=int(os.environ.get("SA_SPP", "4")), enable_shadows=True,
                            enable_ambient_occlusion=True, enable_reflections=True)))

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, physics=PhysxCfg(), gravity=(0.0, 0.0, -9.81))
    with launch_simulation(sim_cfg, args):
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
        from isaaclab.assets import RigidObject, RigidObjectCfg
        sim = SimulationContext(sim_cfg)
        scene = instantiate(SceneCfg(num_envs=1, env_spacing=5.0))
        S = json.load(open(TWIN / "scene.json"))
        demo = Demo(S, args.tracks)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        real = real_frames(S, demo, out)
        isaac_root = ISAAC_NUCLEUS_DIR.rsplit("/Isaac", 1)[0]
        R = Renderer(sim, scene, S, demo, dict(NV=NVIDIA_NUCLEUS_DIR, ISAAC=ISAAC_NUCLEUS_DIR, ISAAC_ROOT=isaac_root))
        R.grip = Gripper(S, demo, args.hands.format(ep=args.episode)) if args.grippers else None
        if R.grip is not None and args.stance:                                  # plan mode: the stance of the real demonstrator, fixed
            st_ = json.load(open(args.stance))
            R.grip.shoulder = {k: np.array(v, float) for k, v in st_["shoulder"].items()}; R.grip.base_yaw = math.radians(st_["facing_deg"])
            R.grip.face = np.array([math.cos(R.grip.base_yaw), math.sin(R.grip.base_yaw)]); R.grip.mid = np.mean([v[:2] for v in R.grip.shoulder.values()], 0)
        worlds = ([("twin", TWIN_SPEC)] if args.twin else [])
        lib, rg = json.load(open(args.library)), json.load(open(args.ranges))
        worlds += [(f"seed{int(x)}", sample_world(int(x), lib, rg, S, isaac_root)) for x in args.seeds.split(",") if x.strip()]
        assert len(worlds) == 1, "one world per process (physics views are made once, at sim.reset)"
        wn, spec = worlds[0]
        if args.zoom != 1.0 and spec.get("cam", {}).get("orbit"):
            spec["cam"].update(dist=spec["cam"]["dist"] * args.zoom, zoom=args.zoom)
        if args.cam_el is not None and spec.get("cam", {}).get("orbit"):
            spec["cam"]["el"] = args.cam_el
        if args.mood != "none":
            apply_mood(spec, args.mood)
        if args.same_objects:
            spec["swap"], spec["obj_mat"] = {}, {}
        if args.wm_input:                                                       # the world model's input view: the phone camera (as its training video)
            spec["wm_input"] = True
            if args.wm_cam == "phone":
                spec["cam"] = dict(phone=True)
            else:                                                               # the hero view, held still (one camera pose for the masks)
                spec["cam"]["sweep"] = 0.0
            import carb
            cs_ = carb.settings.get_settings()                                  # a mood is a fixed exposure: no auto-exposure evening it out
            print(f"[watch-dr] auto-exposure (/rtx/post/histogram/enabled) was {cs_.get('/rtx/post/histogram/enabled')}; set False", flush=True)
            cs_.set("/rtx/post/histogram/enabled", False)
            cs_.set("/rtx/sceneDb/ambientLightIntensity", 0.0)                   # Kit's constant ambient fill (0.1 grey): off, only the mood's lights
        json.dump(spec, open(out / f"{args.episode}__watch_dr_{wn}.json", "w"), indent=1)        # the drawn world, for the record
        t0 = time.time()
        R.build(spec)
        R.bodies = [RigidObject(RigidObjectCfg(prim_path=f"/World/WD/Scene/O{k}", spawn=None)) for k in range(len(S["objects"]))]
        sim.reset()
        R.start()
        R.camera(spec)
        if args.mood != "none":                                                 # the render settings as the renderer left them (exposure)
            import carb
            def _flat(d, pre=""):
                for k_, v_ in (d or {}).items():
                    yield from (_flat(v_, f"{pre}/{k_}") if isinstance(v_, dict) else [(f"{pre}/{k_}", v_)])
            for k_, v_ in _flat(carb.settings.get_settings().get("/rtx")):
                if re.search("ambient|lightingMode|ViewLighting|domeLight|indirectDiffuse|sceneDb|lightRig|rig\b|defaultLight", k_, re.I):
                    print(f"[watch-dr] setting /rtx{k_} = {v_}", flush=True)
        res = R.world_clip(wn, spec, real, out)
        print(f"[watch-dr] {wn} -> {res} ({time.time() - t0:.0f} s)" + (f"; arm IK worst position error {R.ik_err * 100:.1f} cm, held (out of reach) {getattr(R, 'ik_held', 0)} arm-frames; arm-arm overlap (conservative sphere model) in {getattr(R, 'ncol', 0)} frames, "
                                                                              f"closest {getattr(R, 'dmin', 9.0) * 100:.1f} cm; grippers pushed apart in {getattr(R, 'npush', 0)} frames" if hasattr(R, "ik_err") else ""), flush=True)


# ---- the demonstration: tracked object + hand motion ---------------------------------------------------------------------------------
class Demo:
    def __init__(self, S, path):
        z = np.load(path, allow_pickle=True)
        i = np.nonzero(z["group"] == args.episode)[0]
        i = i[np.argsort(z["snap_k"][i])]
        self.t = z["snap_k"][i].astype(np.float64) / 10.0                      # seconds in the video (keyframes at 10 fps)
        geo, msk, oid, kind = z["obj_geo"][i], z["obj_mask"][i], z["obj_id"][i], z["obj_kind"][i]
        ctc = np.where(z["contact_ok"][i][:, None] > 0, z["obj_contact"][i], 0.0)        # HOIST-Former, its own 0.5 threshold
        ctc2 = np.where(z["hands23_ok"][i][:, None] > 0, z["hands23_contact"][i], 0.0)   # Hands23, its own 0.5 threshold
        n = len(S["objects"]); T = len(i)
        self.init = json.load(open(args.init_state)) if args.init_state else None
        Xi = np.array(self.init["root_pos"], float) if self.init else np.array([o["pos"] for o in S["objects"]], float)
        # each twin object <- the tracked object (kind 0) nearest its start position in the first sample (one to one, greedy)
        cand = [(k, int(oid[0][k])) for k in range(oid.shape[1]) if msk[0][k] and kind[0][k] == 0]
        d = np.array([[np.linalg.norm(Xi[a, :2] - geo[0][k][:2]) for k, _ in cand] for a in range(n)])
        pairs = {}
        for _ in range(min(n, len(cand))):
            a, b = np.unravel_index(np.nanargmin(d), d.shape); pairs[a] = cand[b][1]; d[a, :] = np.nan; d[:, b] = np.nan
        self.track_id = [pairs.get(a) for a in range(n)]
        print("[watch-dr] twin object <- track id:", {S["objects"][a]["id"]: self.track_id[a] for a in range(n)}, flush=True)
        P = np.full((T, n, 3), np.nan); Y = np.full((T, n), np.nan); Hd = np.zeros((T, n), bool)
        for j in range(T):
            ids = list(oid[j])
            for a, tid in enumerate(self.track_id):
                if tid is not None and tid in ids and msk[j][ids.index(tid)]:
                    g_ = geo[j][ids.index(tid)]; P[j, a] = g_[:3]; Y[j, a] = math.atan2(g_[7], g_[6])
                    Hd[j, a] = max(ctc[j][ids.index(tid)], ctc2[j][ids.index(tid)]) >= 0.5
        for a in range(n):                                                      # gaps: hold the last seen value (the first: the next one)
            for arr in (P[:, a], Y[:, a]):
                ok = ~np.isnan(arr.reshape(T, -1)).any(-1)
                if ok.any():
                    idx = np.where(ok, np.arange(T), 0); np.maximum.accumulate(idx, out=idx); arr[:] = arr[idx]
                    arr[:np.argmax(ok)] = arr[np.argmax(ok)]
        for a in range(n):                                                      # yaw: unwrap modulo pi (the tracker's axis has no sign)
            y = Y[:, a].copy()
            for j in range(1, T):
                y[j] = y[j - 1] + ((y[j] - y[j - 1] + np.pi / 2) % np.pi - np.pi / 2)
            Y[:, a] = y
        med = lambda X: np.stack([np.median(X[max(0, j - 1):j + 2], 0) for j in range(T)])                       # noqa: E731
        self.P, self.Y = med(P), med(Y)
        self.Pz = np.stack([P[max(0, j - 1):j + 2, :, 2].max(0) for j in range(T)])  # height: max of 3 samples (the median would flatten a short lift)
        self.P0 = self.P[0].copy(); self.Y0 = self.Y[0].copy()
        ext = np.array([o.get("extents", [0.1, 0.1, 0.1]) for o in S["objects"]], float)
        self.use_yaw = np.maximum(ext[:, 0], ext[:, 1]) / np.maximum(np.minimum(ext[:, 0], ext[:, 1]), 1e-6) > 1.3
        self.held = np.stack([Hd[max(0, j - 2):j + 3].any(0) for j in range(T)])     # touched within +-2 samples (0.6 s): the contact models stop early
        self.t0, self.t1 = float(self.t[0]), float(self.t[-1])
        # start from the twin's own (physically consistent) layout; the twin - track offset fades out over each object's first held stretch
        self.X0 = Xi.copy(); self.off = self.X0[:, :2] - self.P0[:, :2]
        self.fade = []
        for a in range(n):
            js = np.nonzero(self.held[:, a])[0]
            if len(js):
                j1 = js[0] + int(np.argmax(np.append(~self.held[js[0]:, a], True)))          # the end of the first held run
                self.fade.append((self.t[js[0]], self.t[min(j1, T - 1)]))
            else:
                self.fade.append((np.inf, np.inf))
        print("[watch-dr] held samples per object:", {S["objects"][a]["id"]: int(self.held[:, a].sum()) for a in range(n)}, flush=True)
        print(f"[watch-dr] {args.episode}: {T} tracked samples {self.t0:.1f}-{self.t1:.1f} s; yaw used for "
              f"{[S['objects'][a]['id'] for a in range(n) if self.use_yaw[a]]}", flush=True)

    def at(self, t):
        """Targets at video time t: positions (n, 3) in the twin's frame, yaw delta (n,), held (n,) bool."""
        t = min(max(t, self.t0), self.t1); j = min(int(np.searchsorted(self.t, t, side="right")) - 1, len(self.t) - 2)
        a = (t - self.t[j]) / max(self.t[j + 1] - self.t[j], 1e-6)
        P = (1 - a) * self.P[j] + a * self.P[j + 1]; Y = (1 - a) * self.Y[j] + a * self.Y[j + 1]
        P[:, 2] = (1 - a) * self.Pz[j] + a * self.Pz[j + 1]
        w = np.array([1.0 if t <= f0 else (0.0 if t >= f1 else (f1 - t) / max(f1 - f0, 1e-6)) for f0, f1 in self.fade])
        X = np.concatenate([P[:, :2] + w[:, None] * self.off, (self.X0[:, 2] + np.maximum(0.0, P[:, 2] - self.P[0, :, 2]))[:, None]], 1)
        held = self.held[j] if a < 0.5 else self.held[j + 1]
        return X, np.where(self.use_yaw, Y - self.Y0, 0.0), held


# ---- the hands -> two floating robot grippers (--grippers) ----------------------------------------------------------------------------
class Gripper:
    """Each human hand -> a floating Franka gripper (render only, no collisions), retargeted the usual way for a parallel gripper:
    the grasp point (TCP) = midpoint of the thumb and index tips, the gripper points down (a parallel gripper picking from a table) and
    turns about the vertical so its fingers close along the thumb -> index direction, the opening = the thumb-index gap (clipped to the
    Franka's 8 cm). (The hands' full 3-D rotation is not used: it flips when the fingers relax.) Hands: WiLoR MANO joints placed with
    measured depth (hands_current.npz, camera frame, 10 fps) -> the table frame with the twin's camera pose.
    Clean-up for display: a sample is dropped when its grasp point is > 5 cm below the table top (inside the solid table: a depth error)
    or > 3 robust SDs (and > 5 cm) from the median of +-3 samples; gaps up to 1 s are bridged linearly, longer ones hide the gripper (arms:
    it eases home); then a 5-sample moving average. Arms: a target the IK misses by > 8 cm (out of reach) keeps the last pose.
    Contact anchoring (the two perception streams disagree by ~5 cm): while the contact models say an object that fits between the
    fingers (its thinnest side <= FIT) is held (Demo.held), the nearest hand's gripper is pulled onto that object's simulated body (blend over the
    0.3 s sample step) and closes on its centre; a bigger object (the bowl) is not anchored: the gripper keeps following the hand (anchoring
    to it would hold the gripper on its rim). The objects' own motion is unchanged.
    Arms (--arms): where the demonstrator stands = the table edge the forearms point to (median wrist - middle-knuckle direction), the
    shoulders just behind it (BEHIND), an average adult shoulder width apart (SHW) about the hands' centre, at standing shoulder height
    above the table top (SH_Z); each Franka's shoulder joint (0.333 m above its base) at a shoulder, facing the table."""
    TCP = 0.1034                                                                # Franka hand flange -> fingertip centre (m)
    SHW, SH_Z, BEHIND = 0.40, 0.60, 0.13                                        # m: average adult shoulder width; shoulder 1.37 m - table 0.77 m
    FIT, GAP = 0.09, 0.22                                                       # m: graspable = thinnest side <= FIT; min gripper-gripper gap (a hand ~0.2 m)
    TILT = math.radians(20.0)                                                   # arms: each gripper leans 20 deg so its hand body sits outward

    def __init__(self, S, demo, path):
        z = np.load(path); T = np.array(S["cameras"]["main"]["T_cam_to_table"], float); k0 = int(z["k0"])
        self.th = (k0 + np.arange(len(z["left_ok"]))) / 10.0                      # video seconds
        self.ext = np.array([o.get("extents", [0.1, 0.1, 0.1]) for o in S["objects"]], float)
        self.h = {}
        for s in ("left", "right"):
            J = z[f"{s}_J"] @ T[:3, :3].T + T[:3, 3]; ok = z[f"{s}_ok"].copy(); n = len(ok)
            g = (J[:, 4] + J[:, 8]) / 2; y = J[:, 8] - J[:, 4]; a = g - J[:, 0]; w = np.linalg.norm(y, axis=1)
            ok &= g[:, 2] > -0.05                                               # a grasp point inside / under the solid table: a depth error
            for i in range(n):                                                  # robust outliers (Hampel)
                nb = [j for j in range(max(0, i - 3), min(n, i + 4)) if ok[j] and j != i]
                if ok[i] and len(nb) >= 2:
                    m = np.median(g[nb], 0); mad = np.median(np.linalg.norm(g[nb] - m, axis=1))
                    if np.linalg.norm(g[i] - m) > max(3 * 1.4826 * mad, 0.05):
                        ok[i] = False
            idx = np.nonzero(ok)[0]; vis = np.zeros(n, bool)
            for i in range(n):                                                  # bridge gaps <= 1 s
                lo, hi = idx[idx <= i], idx[idx >= i]
                vis[i] = len(lo) and len(hi) and hi[0] - lo[-1] <= 10
            F = np.concatenate([g, y, a, w[:, None]], 1)
            F = np.stack([np.interp(np.arange(n), idx, F[idx, c]) for c in range(F.shape[1])], 1)
            yv = F[:, 3:6]
            for i in range(1, n):                                               # the finger axis has no sign: keep it continuous
                if yv[i] @ yv[i - 1] < 0:
                    yv[i:] *= -1
            F = np.stack([F[max(0, i - 2):i + 3].mean(0) for i in range(n)])
            self.h[s] = dict(g=F[:, :3], y=F[:, 3:6], a=F[:, 6:9], w=F[:, 9], vis=vis, fa=np.median(J[ok, 0] - J[ok, 9], 0))
            print(f"[watch-dr] gripper {s}: {ok.sum()}/{n} hand samples kept, visible {vis.sum()}", flush=True)
        d = np.mean([H["fa"][:2] for H in self.h.values()], 0); d /= max(np.linalg.norm(d), 1e-9); ax = int(abs(d[1]) > abs(d[0]))
        tb = S["table"]; sg = 1.0 if d[ax] > 0 else -1.0; edge = tb["xy"[ax]][1 if sg > 0 else 0]
        face = np.zeros(2); face[ax] = -sg; left = np.array([-face[1], face[0]])
        mid = np.mean([np.median(H["g"][H["vis"], :2], 0) for H in self.h.values()], 0); mid[ax] = edge + sg * self.BEHIND
        self.shoulder = {"left": np.array([*(mid + self.SHW / 2 * left), self.SH_Z]), "right": np.array([*(mid - self.SHW / 2 * left), self.SH_Z])}
        self.base_yaw = math.atan2(face[1], face[0]); self.mid, self.face = mid, face
        print(f"[watch-dr] stance: forearms point {np.round(d, 2)} -> shoulders {({k: np.round(v, 2).tolist() for k, v in self.shoulder.items()})}, "
              f"facing {math.degrees(self.base_yaw):.0f} deg", flush=True)
        # which hand holds which object, per tracked sample: the held object nearest the hand's grasp point (box distance, < 15 cm)
        self.tk = demo.t; self.anchor = {s: np.full(len(demo.t), -1) for s in self.h}
        for j, t in enumerate(demo.t):
            for s in self.h:
                if not self.vis(s, t):
                    continue
                g = self.at(s, t)[0]; best = (0.15, -1)
                for k in np.nonzero(demo.held[j] & (self.ext.min(1) <= self.FIT))[0]:      # only what fits between the fingers (its thinnest side)
                    c, e = demo.P[j, k], self.ext[k] / 2
                    loc = _rot_z(-demo.Y[j, k]) @ (g - c); d = np.linalg.norm(loc - np.clip(loc, -e, e))
                    best = min(best, (d, k))
                self.anchor[s][j] = best[1]
        print("[watch-dr] gripper anchors (object per 0.3 s):", {s: self.anchor[s].tolist() for s in self.h}, flush=True)
        self.lock = {}                                                          # (hand, object) -> grip point in the body frame, per held run

    def vis(self, s, t):
        i = int(round((t - self.th[0]) * 10))
        return 0 <= i < len(self.th) and bool(self.h[s]["vis"][i])

    def at(self, s, t):
        """The hand-following gripper at video time t: TCP, rotation (cols: x, finger axis y, approach z), opening (m)."""
        H = self.h[s]; f = lambda X: np.stack([np.interp(t, self.th, X[:, c]) for c in range(X.shape[1])])   # noqa: E731
        g, y = f(H["g"]), f(H["y"]); w = float(np.interp(t, self.th, H["w"]))
        yaw = math.atan2(y[1], y[0]); c, s_ = math.cos(yaw), math.sin(yaw)       # top-down (franka_kinematics.top_down): x = (-s, c, 0), y = (c, s, 0), z down
        if not args.arms:
            return g, np.array([[-s_, c, 0.0], [c, s_, 0.0], [0.0, 0.0, -1.0]]), float(np.clip(w, 0.0, 0.08))
        out = np.array([-self.face[1], self.face[0], 0.0]) * (1.0 if s == "left" else -1.0)        # this arm's outward side
        z = -math.cos(self.TILT) * np.array([0.0, 0.0, 1.0]) - math.sin(self.TILT) * out          # down, leaning in: the hand body sits outward
        yv = np.array([c, s_, 0.0]); yy = yv - (yv @ z) * z; yy /= max(np.linalg.norm(yy), 1e-9)
        return g, np.stack([np.cross(yy, z), yy, z], 1), float(np.clip(w, 0.0, 0.08))

    def weights(self, s, t):
        """Per object: how strongly this hand's gripper is anchored on it at time t (linear between the 0.3 s samples)."""
        out = {}
        for k in set(self.anchor[s].tolist()) - {-1}:
            v = float(np.interp(t, self.tk, (self.anchor[s] == k).astype(float)))
            if v > 0:
                out[k] = v * v * (3 - 2 * v)
        return out


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def real_frames(S, demo, out):
    """The real video's frames over the tracked span, at --fps, 640 wide (cached)."""
    vid = args.video or str(ROOT / "data/raw_clips/table_clean" / f"{S['video']}.MOV")
    d = out / f"real_{args.episode}"; d.mkdir(exist_ok=True)
    n = int(round((demo.t1 - demo.t0) * args.fps)) + 1
    if len(list(d.glob("*.jpg"))) < n:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{demo.t0:.3f}", "-i", vid, "-t", f"{demo.t1 - demo.t0 + 0.05:.3f}", "-r", str(args.fps),
                        "-vf", f"scale={LW}:-2", "-q:v", "2", str(d / "%05d.jpg")], check=True)
    fr = sorted(d.glob("*.jpg"))
    print(f"[watch-dr] real video frames: {len(fr)} (need {n}) from {vid}", flush=True)
    return fr


TWIN_SPEC = dict(title="Isaac twin (as built)", twin=True, sky=dict(intensity=220, color=[1.0, 0.95, 0.88]),
                 lights=[dict(type="distant", intensity=350, angle=25.0, color=[1.0, 0.93, 0.85])], cam=dict(phone=True),
                 desc="the twin exactly as built: static plate background + textured static mesh + object twins, from the real phone camera")


def sky_local(url, cache):
    """A 2k copy of a library sky (downloaded once) and its log-average luminance (the usual exposure key), for the exposure."""
    import urllib.request
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    loc = cache / url.rsplit("/", 1)[-1]; mf = cache / "logavg_luminance.json"
    med = json.load(open(mf)) if mf.exists() else {}
    if not loc.exists():
        tmp = cache / (loc.name + ".part"); urllib.request.urlretrieve(url, tmp)
        im = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED); im = cv2.resize(im, (2048, 1024), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(loc), im); tmp.unlink()
    if loc.name not in med:
        im = cv2.imread(str(loc), cv2.IMREAD_UNCHANGED)[..., :3]
        L = 0.2126 * im[..., 2] + 0.7152 * im[..., 1] + 0.0722 * im[..., 0]
        med[loc.name] = float(np.exp(np.log(np.maximum(L, 0) + 1e-4).mean())); json.dump(med, open(mf, "w"), indent=1)
    return str(loc), med[loc.name]


def _shape(b):
    b = np.sort(np.asarray(b, float)[:2])[::-1].tolist() + [float(b[2])]
    return np.log(np.maximum(b, 1e-6)) - np.log(max(b[0], 1e-6))


def sample_world(seed, lib, rg, S, isaac_root, fetch=True):
    """One automatic world: every choice drawn from the library + ranges with this seed (nothing per world by hand). The sky is the first
    draw, so a seed keeps its sky when the other ranges change. fetch=False: no sky download (exposure left at 1)."""
    r = np.random.default_rng(seed); U = lambda a: float(r.uniform(*a))                                        # noqa: E731
    sky_url = r.choice(lib["skies"]).format(ISAAC_ROOT=isaac_root)
    hdri, med = sky_local(sky_url, args.sky_cache) if fetch else (sky_url, 1.0)
    sky = dict(hdri=hdri, intensity=float(np.clip(U(rg["sky_effective"]) / max(med, 1e-4), *rg["sky_intensity_clip"])), rot_z=U([0, 360]))
    if r.random() < rg["sky_temp_p"]:
        sky["temp"] = U(rg["sky_temp"])
    def col():                                                                  # a coloured light now and then (else white / blackbody)
        if r.random() < rg["color_p"]:
            import colorsys
            return list(colorsys.hsv_to_rgb(float(r.random()), U(rg["color_sat"]), 1.0))
        return None
    key = dict(type="distant", intensity=U(rg["key_intensity"]), angle=U(rg["key_angle"]), el=U(rg["key_elevation"]), az=U([0, 360]),
               temp=U(rg["key_temp"]), color=col())
    lights = [key]
    for _ in range(int(r.choice(len(rg["extra_n_p"]), p=rg["extra_n_p"]))):
        kind = r.choice(["rect", "sphere", "rim"])
        if kind == "rect":
            lights.append(dict(type="rect", intensity=U(rg["rect_intensity"]), temp=U(rg["extra_temp"]), color=col(), pos=[U([-0.4, 0.4]), U([-0.4, 0.4]), U(rg["rect_height"])],
                               size=[U(rg["rect_size"][0]), U(rg["rect_size"][1])], rot=[U([-30, 30]), U([-30, 30]), U([0, 180])]))
        elif kind == "sphere":
            o = rg["sphere_offset"]
            lights.append(dict(type="sphere", intensity=U(rg["sphere_intensity"]), temp=U(rg["extra_temp"]), color=col(), radius=U([0.02, 0.12]),
                               pos=[U([-o, o]), U([-o, o]), U(rg["sphere_height"])]))
        else:
            lights.append(dict(type="distant", intensity=U(rg["rim_intensity"]), angle=U([0.5, 4.0]), el=U(rg["rim_elevation"]), az=U([0, 360]),
                               temp=U(rg["extra_temp"]), color=col(), rim=True))
    surf = dict(mdl=r.choice(lib["surfaces"]).format(ISAAC_ROOT=isaac_root), tex=[U(rg["surface_tex"])] * 2)
    cam = dict(orbit=True, az=U(rg["cam_az"]), el=U(rg["cam_el"]), dist=U(rg["cam_dist"]), hfov=U(rg["cam_hfov"]), look_z=U(rg["cam_look_z"]), sweep=U(rg["cam_sweep"]))
    swap, mats = {}, {}
    for o in S["objects"]:
        top = o["category"].split("/")[0]
        pool = [x for x in lib["objects"] if x["category"].split("/")[0] == top]
        if pool:
            d = np.array([np.abs(_shape(x["box"]) - _shape(o["extents"])).sum() for x in pool])
            w = np.exp(-(d - d.min()) / rg["variant_tau"]); x = pool[int(r.choice(len(pool), p=w / w.sum()))]
            if x["key"] != re.sub(r"_\d+$", "", Path(o["usd"]).stem):          # another library model (not this object's own scan)
                swap[o["id"]] = dict(usd=x["usd"].format(ISAAC_ROOT=isaac_root), name=x["key"].lower().rstrip("0123456789"), source=x["source"])
        if r.random() < rg["object_material_p"]:
            mats[o["id"]] = dict(mdl=r.choice(lib["surfaces"]).format(ISAAC_ROOT=isaac_root), tex=[U(rg["object_tex"])] * 2, proj=True, world=False)
    nm = lambda u: u.rsplit("/", 1)[-1].rsplit(".", 1)[0]                                                    # noqa: E731
    lt = lambda l: ("rim" if l.get("rim") else l["type"]) + (" (coloured)" if l.get("color") else f" {l['temp']:.0f} K")              # noqa: E731
    desc = (f"seed {seed} | sky: {nm(sky_url)}{f", {sky['temp']:.0f} K" if 'temp' in sky else ''} | static mesh: {nm(surf['mdl'])} | lights: "
            + " + ".join(lt(l) for l in lights)
            + f" | camera: az {cam['az']:.0f}, el {cam['el']:.0f} deg, {cam['dist']:.2f} m, {cam['hfov']:.0f} deg FOV")
    w = dict(title=f"Automatic world, seed {seed}", seed=seed, sky_name=nm(sky_url), sky_group=sky_url.rsplit("/", 2)[-2],
             surface_name=nm(surf["mdl"]), surface_group=surf["mdl"].split("/Base/", 1)[-1].split("/")[0], sky=sky, lights=lights, static_mat=surf, cam=cam, swap=swap, obj_mat=mats, desc=desc)
    # a whole 3-D room (drawn after every other value, so a seed keeps its earlier draws)
    if lib.get("rooms") and r.random() < rg.get("room_p", 0.0) and not args.no_room:
        pool = ["outdoor"] if args.place == "outdoor" else [x for x in lib["rooms"] if args.place == "any" or args.place.lower() in x.lower()]
        room = dict(usd=r.choice(pool).format(ISAAC_ROOT=isaac_root), spot_u=float(r.random()), quarter=int(r.integers(4)))
        table = dict(usd=r.choice(lib["tables"]).format(ISAAC_ROOT=isaac_root))
        sz = rg["room_light_size"]
        lights.append(dict(type="rect", intensity=U(rg["room_light_intensity"]), temp=U(rg["room_light_temp"]), pos=[0.0, 0.0, U(rg["room_light_height"])],
                           size=[U(sz[0]), U(sz[1])], rot=[0.0, 0.0, U([0, 180])], ceiling=True))
        cam.update(dist=U(rg["room_cam_dist"]), el=U(rg["room_cam_el"]), hfov=U(rg["room_cam_hfov"]), front=rg.get("room_cam_front"))
        tops = [m for m in lib["surfaces"] if re.search(rg["room_table_top"], m)]; legs = [m for m in lib["surfaces"] if re.search(rg["room_table_leg"], m)]
        table.update(top=dict(mdl=r.choice(tops).format(ISAAC_ROOT=isaac_root), tex=[U([0.5, 1.5])] * 2), leg=dict(mdl=r.choice(legs).format(ISAAC_ROOT=isaac_root)))
        cam.update(look_z=U(rg["room_cam_look_z"]), sweep=U(rg["room_cam_sweep"]))
        for l in lights:                                                        # rooms: no coloured lights
            l["color"] = None
        room["name"], table["name"] = nm(room["usd"]), f"{nm(table['top']['mdl'])} top, {nm(table['leg']['mdl'])} legs"
        if room["usd"] == "outdoor":                                            # outdoors: an outdoor sky over a real ground, no ceiling light
            lights[:] = [l for l in lights if not l.get("ceiling")]
            su = r.choice([x for x in lib["skies"] if re.search(rg["outdoor_sky"], x)]).format(ISAAC_ROOT=isaac_root)
            h2, m2 = sky_local(su, args.sky_cache) if fetch else (su, 1.0)
            sky.update(hdri=h2, intensity=float(np.clip(U(rg["sky_effective"]) / max(m2, 1e-4), *rg["sky_intensity_clip"])))
            sky.pop("temp", None); w.update(sky_name=nm(su), sky_group=su.rsplit("/", 2)[-2])
            room.update(outdoor=True, name="outdoor", side=float(r.choice([-1.0, 1.0])),
                        ground=dict(mdl=r.choice([m for m in lib["surfaces"] if re.search(rg["outdoor_ground"], m)]).format(ISAAC_ROOT=isaac_root),
                                    tex=[U(rg["outdoor_ground_tex"])] * 2))
        w.update(room=room, table_model=table, title=f"Automatic room, seed {seed}",
                 desc=f"seed {seed} | room: {room['name'] if not room.get('outdoor') else 'outdoor, ' + nm(room['ground']['mdl'])} | table: {table['name']} | sky: {w['sky_name']} | lights: " + " + ".join(lt(l) for l in lights)
                 + f" | camera: az {cam['az']:.0f}, el {cam['el']:.0f} deg, {cam['dist']:.2f} m, {cam['hfov']:.0f} deg FOV")
    if args.sky_name:                                                           # a named sky (test worlds): exposure from its own brightness
        su = next(x for x in lib["skies"] if args.sky_name in x).format(ISAAC_ROOT=isaac_root)
        h2, m2 = sky_local(su, args.sky_cache) if fetch else (su, 1.0)
        w["sky"].update(hdri=h2, intensity=float(np.clip(U(rg["sky_effective"]) / max(m2, 1e-4), *rg["sky_intensity_clip"]))); w["sky"].pop("temp", None)
        w.update(sky_name=nm(su), sky_group=su.rsplit("/", 2)[-2])
    return w


# ---- lighting moods: a dark kitchen with one low warm lamp; bright outdoors with a reflection on the table; a warehouse with a
# floodlight from the side (some shadows). Render-only looks, one per kind of place; the light positions follow the camera direction
# (Renderer.mood). The lamp / flood intensities are normalized (independent of the light's size), the sun's is not (as the drawn key
# lights). Kit's constant ambient fill (/rtx/sceneDb/ambientLightIntensity, a flat 0.1 grey) evens every look out, so it is off for a
# mood. The flood is unshaped (a cone shaping removes most of its light), close and low for long shadows.
MOODS = {
    "lamp": dict(desc="dark, one warm pendant lamp", sky=0.08, room_lights=0.0, ceiling=0.0,
                 lamp=dict(z=1.0, r=0.15, intensity=12000.0, temp=2600.0, cone=45.0, soft=0.3)),
    "sun": dict(desc="bright low sun, glossy top", sky=0.7, ceiling=0.0, top="Stone/Granite_Dark.mdl",
                sun=dict(intensity=15000.0, angle=0.6, temp=5600.0, el=32.0, off=30.0)),
    "night": dict(desc="night, cool moonlight", sky=0.04, room_lights=0.0, ceiling=0.0,
                  sun=dict(intensity=320.0, angle=0.6, temp=9500.0, el=35.0, off=70.0)),
    "torch": dict(desc="pitch dark, one cold flashlight beam from the camera side", sky=0.0, room_lights=0.0, ceiling=0.0,
                  torch=dict(dist=1.3, z=0.75, az=12.0, intensity=110000.0, temp=7500.0, cone=16.0, soft=0.25)),
    "neon": dict(desc="dark studio, magenta + cyan lights", sky=0.03, room_lights=0.0, ceiling=0.0,
                 neon=[dict(dist=1.6, z=0.9, az=70.0, size=[0.4, 0.4], intensity=120000.0, temp=6500.0, color=[1.0, 0.1, 0.8]),
                       dict(dist=1.6, z=0.9, az=-70.0, size=[0.4, 0.4], intensity=120000.0, temp=6500.0, color=[0.1, 0.9, 1.0])]),
    "flood": dict(desc="dim room lights + a side floodlight", sky=0.15, room_lights=0.25, ceiling=0.0,
                  flood=dict(dist=1.8, z=1.0, az=100.0, size=[0.5, 0.35], intensity=110000.0, temp=5200.0, cone=0.0, soft=0.3)),
}


def apply_mood(spec, name):
    """The drawn world's lights -> the mood's: the sky (dome) scaled, the drawn key / rim / extra lights dropped, the ceiling panel scaled;
    the mood's own lights are added by Renderer.mood once the camera direction is known."""
    M = dict(MOODS[name], name=name)
    spec["sky"]["intensity"] *= M["sky"]
    spec["lights"] = [dict(l, intensity=l["intensity"] * M["ceiling"]) for l in spec["lights"] if l.get("ceiling") and M["ceiling"] > 0]
    if M.get("top") and spec.get("table_model"):
        tp = spec["table_model"]["top"]; tp["mdl"] = tp["mdl"].split("/Base/")[0] + "/Base/" + M["top"]
        spec["table_model"]["name"] = f"{M['top'].rsplit('/', 1)[-1][:-4]} top, " + spec["table_model"]["name"].split(" top, ", 1)[-1]
    spec["mood"] = M; spec["desc"] += f" | mood: {name} ({M['desc']})"
    return spec


def _R_to_wxyz(R):
    R = np.asarray(R, np.float64); tr = np.trace(R)
    if tr > 0:
        s = 2 * math.sqrt(tr + 1); return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]); return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if i == 1:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]); return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]); return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def _rot_xyz(deg):
    x, y, z = np.radians(deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Renderer:
    def __init__(self, sim, scene, S, demo, roots):
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils
        self.sim, self.scene, self.cam, self.S, self.demo, self.roots = sim, scene, scene["cam"], S, demo, roots
        self.st = sim_utils.get_current_stage(); self.su = sim_utils
        self.org = scene.env_origins[0].cpu().numpy().astype(float)
        UsdGeom.Xform.Define(self.st, "/World/WD")

    # ---- materials / lights ------------------------------------------------------------------------------------------------------
    def _url(self, s):
        s = s.format(**self.roots)
        return s if "://" in s or s.startswith("/") else str(ROOT / s)

    def _pbr(self, path, rgb, opacity=1.0, rough=0.5):
        from pxr import Gf, Sdf, UsdShade
        mat = UsdShade.Material.Define(self.st, path)
        sh = UsdShade.Shader.Define(self.st, path + "/S"); sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity)); sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _mdl(self, path, spec, project=True):
        from pxr import Gf, Sdf, UsdShade
        url = self._url(spec["mdl"])
        mat = UsdShade.Material.Define(self.st, path)
        sh = UsdShade.Shader.Define(self.st, f"{path}/Shader")
        sh.SetSourceAsset(Sdf.AssetPath(url), "mdl"); sh.SetSourceAssetSubIdentifier(url.rsplit("/", 1)[-1].removesuffix(".mdl"), "mdl")
        o = sh.CreateOutput("out", Sdf.ValueTypeNames.Token); o.SetRenderType("material")
        mat.CreateSurfaceOutput("mdl").ConnectToSource(o); mat.CreateDisplacementOutput("mdl").ConnectToSource(o); mat.CreateVolumeOutput("mdl").ConnectToSource(o)
        if spec.get("proj", project):
            sh.GetPrim().CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
            sh.GetPrim().CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(bool(spec.get("world", True)))
        if spec.get("tex"):
            sh.GetPrim().CreateAttribute("inputs:texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*spec["tex"]))
        return mat

    def _bind(self, prim, mat, strong=False):
        from pxr import UsdShade
        b = UsdShade.MaterialBindingAPI.Apply(prim)
        b.Bind(mat, UsdShade.Tokens.strongerThanDescendants) if strong else b.Bind(mat)

    def _box(self, path, lo, hi, mat):
        from pxr import Gf, UsdGeom
        c = UsdGeom.Cube.Define(self.st, path); c.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(c.GetPrim()); lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        x.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2 + self.org))); x.AddScaleOp().Set(Gf.Vec3d(*(hi - lo)))
        if mat is not None:
            self._bind(c.GetPrim(), mat)
        return c

    def lights(self, spec, ctr, ep):
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        def temp(L, t):
            if t:
                L.CreateEnableColorTemperatureAttr(True); L.CreateColorTemperatureAttr(float(t))
        sky = spec.get("sky")
        if sky:
            d = UsdLux.DomeLight.Define(self.st, f"{ep}/Dome"); d.CreateIntensityAttr(float(sky["intensity"]))
            if sky.get("hdri"):
                d.CreateTextureFileAttr(Sdf.AssetPath(self._url(sky["hdri"]))); d.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
            if sky.get("color"):
                d.CreateColorAttr(Gf.Vec3f(*sky["color"]))
            temp(d, sky.get("temp"))
            xd = UsdGeom.Xformable(d.GetPrim())
            if sky.get("rot"):
                xd.AddRotateXYZOp().Set(Gf.Vec3f(*sky["rot"]))
            if sky.get("rot_z"):
                xd.AddRotateZOp().Set(float(sky["rot_z"]))
        for i, l in enumerate(spec.get("lights", [])):
            p = f"{ep}/L{i}"
            if l["type"] == "distant":
                L = UsdLux.DistantLight.Define(self.st, p); L.CreateAngleAttr(float(l.get("angle", 1.0)))
            elif l["type"] == "rect":
                L = UsdLux.RectLight.Define(self.st, p); L.CreateWidthAttr(float(l["size"][0])); L.CreateHeightAttr(float(l["size"][1]))
            elif l["type"] == "sphere":
                L = UsdLux.SphereLight.Define(self.st, p); L.CreateRadiusAttr(float(l.get("radius", 0.05)))
            else:
                raise ValueError(l["type"])
            L.CreateIntensityAttr(float(l["intensity"])); temp(L, l.get("temp"))
            if l.get("color"):
                L.CreateColorAttr(Gf.Vec3f(*l["color"]))
            x = UsdGeom.Xformable(L.GetPrim())
            if "pos" in l:
                x.AddTranslateOp().Set(Gf.Vec3d(*(np.array([ctr[0], ctr[1], 0.0]) + l["pos"] + self.org)))
            if "el" in l:                                                       # a distant light from elevation / azimuth (deg)
                x.AddRotateXYZOp().Set(Gf.Vec3f(90.0 - l["el"], 0.0, l["az"]))
            elif l.get("rot"):
                x.AddRotateXYZOp().Set(Gf.Vec3f(*l["rot"]))

    # ---- objects ---------------------------------------------------------------------------------------------------------------------
    def _bound(self, prim):
        from pxr import Usd, UsdGeom
        bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
        r = bc.ComputeUntransformedBound(prim).ComputeAlignedRange()
        return np.array(r.GetMin(), float), np.array(r.GetMax(), float)

    def _no_physics(self, prim):
        from pxr import Sdf, Usd, UsdPhysics
        for q in Usd.PrimRange(prim):
            if q.HasAPI(UsdPhysics.RigidBodyAPI):                               # removed, not only disabled: one body per object
                q.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False); q.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if q.HasAPI(UsdPhysics.MassAPI):
                q.RemoveAPI(UsdPhysics.MassAPI)
            if q.HasAPI(UsdPhysics.CollisionAPI):
                q.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    def _swap(self, p, orig_usd, sw, k):
        """A render-only variant mesh under the object prim `p`, fit to the twin's local box (bottom centre, horizontal long axis)."""
        from pxr import Gf, UsdGeom
        tmp = self.st.DefinePrim("/World/WD/Tmp", "Xform"); tmp.GetReferences().AddReference(orig_usd)
        UsdGeom.Xformable(tmp).ClearXformOpOrder()
        lo_o, hi_o = self._bound(tmp); self.st.RemovePrim("/World/WD/Tmp")
        fit = UsdGeom.Xform.Define(self.st, f"{p.GetPath()}/Fit")
        ref = self.st.DefinePrim(f"{p.GetPath()}/Fit/Ref", "Xform"); ref.GetReferences().AddReference(self._url(sw["usd"]))
        self._no_physics(ref)
        vis = self.st.GetPrimAtPath(f"{p.GetPath()}/visual")                     # the twin's own mesh: hidden; its collision hulls stay
        if vis and vis.IsValid():
            UsdGeom.Imageable(vis).MakeInvisible()
        lo_r, hi_r = self._bound(fit.GetPrim())
        cor = np.array([[x, y, z] for x in (lo_r[0], hi_r[0]) for y in (lo_r[1], hi_r[1]) for z in (lo_r[2], hi_r[2])])

        def rbox(Rm):
            c = cor @ Rm.T; return c.min(0), c.max(0)
        so = hi_o - lo_o
        best = None                                                             # the axis-aligned rotation whose box shape is the twin's
        for rot in ([0, 0, 0], [0, 0, 90], [90, 0, 0], [0, 90, 0], [90, 0, 90], [90, 90, 0]):
            Rm_ = _rot_xyz(rot); l_, h_ = rbox(Rm_); s_ = h_ - l_
            e = np.abs(np.log(np.maximum(s_, 1e-9) / np.maximum(so, 1e-9)) - np.log(np.maximum(s_, 1e-9) / np.maximum(so, 1e-9)).mean()).sum()
            if best is None or e < best[0] - 1e-6:
                best = (e, Rm_, l_, h_)
        _, Rm, lo2, hi2 = best; s2 = hi2 - lo2
        sv = np.full(3, so[:2].max() / max(s2[:2].max(), 1e-9))
        bc_o = np.array([(lo_o[0] + hi_o[0]) / 2, (lo_o[1] + hi_o[1]) / 2, lo_o[2]]); bc_2 = np.array([(lo2[0] + hi2[0]) / 2, (lo2[1] + hi2[1]) / 2, lo2[2]])
        x = UsdGeom.Xformable(fit.GetPrim())
        x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(bc_o - sv * bc_2)))
        x.AddScaleOp().Set(Gf.Vec3f(*sv.tolist())); x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*_R_to_wxyz(Rm)))
        print(f"[watch-dr]   variant o{k} -> {sw['usd'].rsplit('/', 1)[-1]}: twin box {np.round(so, 3)}, fitted scale {np.round(sv, 4)}", flush=True)

    # ---- one world -------------------------------------------------------------------------------------------------------------------
    def build(self, spec):
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics
        sp = "/World/WD/Scene"
        if self.st.GetPrimAtPath(sp):
            self.st.RemovePrim(sp)
        UsdGeom.Xform.Define(self.st, sp)
        S = self.S
        P0 = np.array([o["pos"] for o in S["objects"]], float)
        self.lights(spec, P0[:, :2].mean(0), sp)                                # light positions relative to the objects' centre
        if spec.get("twin"):                                                    # the twin exactly as built: static plate + textured table
            for nm, f in (("Background", TWIN / S["background"]["usd"]), ("TableVis", TWIN / "table.usda")):
                q = self.st.DefinePrim(f"{sp}/{nm}", "Xform"); q.GetReferences().AddReference(str(f))
                UsdGeom.Xformable(q).AddTranslateOp().Set(Gf.Vec3d(*self.org))
        elif spec.get("room", {}).get("outdoor"):                               # outdoors: a 40 m ground under the sky; the camera a 3/4 hero view
            self._box(f"{sp}/Ground", [-20, -20, self.FLOOR_Z - 0.05], [20, 20, self.FLOOR_Z], self._mdl(f"{sp}/MatGround", dict(spec["room"]["ground"], world=True)))
            face = math.degrees(self.grip.base_yaw) if getattr(self, "grip", None) else 0.0
            spec["room"]["cam_az"] = face + 35.0 * spec["room"]["side"]; self.table_model(spec, sp)
        elif spec.get("room"):                                                  # a whole 3-D room; a library table model under the twin's top
            self.room(spec, sp); self.table_model(spec, sp)
        else:                                                                   # the background plate -> the sky; the twin's static mesh, re-materialled
            q = self.st.DefinePrim(f"{sp}/StaticMesh", "Xform"); q.GetReferences().AddReference(str(TWIN / "table.usda"))
            UsdGeom.Xformable(q).AddTranslateOp().Set(Gf.Vec3d(*self.org))
            self._bind(q, self._mdl(f"{sp}/MatStatic", spec["static_mat"]), strong=True)
        if spec.get("mood"):                                                    # after the world: it needs the room's camera direction
            self.mood(spec, sp, P0[:, :2].mean(0))
        tb = S["table"]                                                         # invisible static colliders: the twin's table + a ground plane
        for nm, lo, hi in (("TableCol", [tb["x"][0], tb["y"][0], -0.04], [tb["x"][1], tb["y"][1], 0.0]), ("GroundCol", [-20, -20, -0.78], [20, 20, -0.76])):
            c_ = self._box(f"{sp}/{nm}", lo, hi, None); UsdPhysics.CollisionAPI.Apply(c_.GetPrim()); UsdGeom.Imageable(c_.GetPrim()).MakeInvisible()
        self.objs, self.shown, self.obox = [], [], []
        for k, o in enumerate(S["objects"]):
            oid = o.get("id", f"o{k}"); sw = spec.get("swap", {}).get(oid)
            p = self.st.DefinePrim(f"{sp}/O{k}", "Xform")
            p.GetReferences().AddReference(str(ROOT / o["usd"]))                # the twin: its rigid body, mass, hulls (physics on)
            self.obox.append(np.array(self._bound(p)) * float(o.get("scale") or 1.0))   # its local box (before any variant), for the grippers
            from pxr import PhysxSchema
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateMaxDepenetrationVelocityAttr(float(os.environ.get("WD_DEPEN", "0.5")))   # gentle un-overlap
            if args.kinematic:
                UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr().Set(True)
            if sw:
                self._swap(p, str(ROOT / o["usd"]), sw, k)
            om = spec.get("obj_mat", {}).get(oid)
            if om:
                self._bind(p, self._mdl(f"{sp}/MatO{k}", om, project=False), strong=True)
            x = UsdGeom.Xformable(p); x.ClearXformOpOrder()
            t_ = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble); r_ = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            sc = float(o.get("scale") or 1.0); x.AddScaleOp().Set(Gf.Vec3f(sc, sc, sc))
            self.objs.append((t_, r_, np.array(o["pos"], float))); self.shown.append(sw["name"] if sw else None)
            if args.wm_input:
                self.su.add_labels(p, [f"o{k}"])                                # its mask (semantic segmentation) for the world model's image positions
        for k, (t_, r_, p0) in enumerate(self.objs):                            # the twin's start layout
            t_.Set(Gf.Vec3d(*(p0 + self.org + [0, 0, 0.002]))); r_.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        self.gprims = {}
        if getattr(self, "grip", None):                                         # two floating Franka grippers: hand + 2 fingers, render only
            props = f"{self.roots['ISAAC']}/Robots/FrankaRobotics/FrankaEmika/Props"
            for s_ in self.grip.h:
                G = UsdGeom.Xform.Define(self.st, f"{sp}/Grip_{s_}"); xg = UsdGeom.Xformable(G.GetPrim())
                ops = [xg.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), xg.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)]
                for nm in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
                    q = self.st.DefinePrim(f"{sp}/Grip_{s_}/{nm}", "Xform"); q.GetReferences().AddReference(f"{props}/{nm}.usd")
                    self._no_physics(q)
                    if nm != "panda_hand":
                        ops.append(UsdGeom.Xformable(q).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble))
                self.gprims[s_] = (G.GetPrim(), ops)
            self.arm_parts = {}
            if args.arms:                                                       # links 0-7 of each arm + one torso column under both bases
                for s_ in self.grip.h:
                    row = []
                    for i in range(8):
                        q = self.st.DefinePrim(f"{sp}/Arm_{s_}/panda_link{i}", "Xform"); q.GetReferences().AddReference(f"{props}/panda_link{i}.usd")
                        self._no_physics(q); xq = UsdGeom.Xformable(q)
                        row.append((xq.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), xq.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)))
                    self.arm_parts[s_] = row
                gr = self.grip; bz = gr.SH_Z - 0.333
                c = UsdGeom.Cube.Define(self.st, f"{sp}/Torso"); c.CreateSizeAttr(1.0); xc = UsdGeom.Xformable(c.GetPrim())
                xc.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(np.array([*(gr.mid - 0.06 * gr.face), (bz + self.FLOOR_Z) / 2]) + self.org)))
                xc.AddRotateZOp().Set(float(math.degrees(gr.base_yaw)))
                xc.AddScaleOp().Set(Gf.Vec3f(0.24, gr.SHW + 0.22, bz - self.FLOOR_Z))
                self._bind(c.GetPrim(), self._pbr(f"{sp}/TorsoMat", (0.16, 0.17, 0.19), rough=0.45))
                from cleantables.robots import franka_kinematics as FK
                self.FK = FK; self.q = {}; self.ncol = 0
                # elbows out: the IK's null space pulls to a rest posture with the base turned outward (q1) and the upper arm rolled (q3)
                # so the elbow swings to the arm's own side (found by an offline search: elbows ~15 cm out, gripper error < 0.3 cm)
                self.q_rest = {}
                for s_ in gr.h:
                    sd = 1.0 if s_ == "left" else -1.0; qr = torch.tensor(FK.Q_NEUTRAL, dtype=torch.float64)
                    qr[0], qr[2] = 1.0 * sd, -1.8 * sd; self.q_rest[s_] = qr[None]
                self.Bw = {s_: FK.base_T(*gr.shoulder[s_][:2], bz, gr.base_yaw, dtype=torch.float64) for s_ in gr.h}

    FLOOR_Z = -0.77                                                             # the twin's floor (GroundCol) below the table top

    def room(self, spec, sp):
        """The drawn room, placed automatically: its floor = the largest flat horizontal mesh box; obstacles = every mesh box between 3 cm
        and 1.9 m above that floor (except whole-room shells > 3 m both ways); the keep-out = the table footprint + 0.45 m all round (robot,
        person); among the floor spots (a 10 cm grid, coarser in big rooms; 4 quarter turns) where the keep-out is free, the 5 most open
        (farthest from any obstacle), spot_u picks one; then the camera direction (Renderer.camera uses it): a free line of sight, within
        +-front deg of the robot's facing, the deepest room behind the table that ends on a wall / furniture (not open space), a 3/4 view
        preferred. Only floor under a ceiling / roof (a mesh box > 1.9 m up) counts, so the table stays indoors. The room is moved so that spot is
        under the table; its own physics is switched off."""
        from pxr import Gf, Sdf, Usd, UsdGeom
        rs = spec["room"]; url = self._url(rs["usd"])
        q = self.st.DefinePrim(f"{sp}/Room", "Xform"); q.GetReferences().AddReference(url)
        lay = Sdf.Layer.FindOrOpen(url)
        mpu = float(lay.pseudoRoot.GetInfo("metersPerUnit")) if lay is not None and lay.pseudoRoot.HasInfo("metersPerUnit") else 1.0
        x = UsdGeom.Xformable(q); x.ClearXformOpOrder()
        def op(add, *v):                                                        # own op names: the room file may already author xformOp:scale etc.
            return add(UsdGeom.XformOp.PrecisionDouble, "wd")
        t_ = op(x.AddTranslateOp); rz = op(x.AddRotateZOp); sc_ = op(x.AddScaleOp)
        V3 = lambda o_, v: o_.Set((Gf.Vec3d if o_.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f)(*[float(a) for a in v]))   # noqa: E731
        V3(sc_, [mpu] * 3)
        V3(t_, [0, 0, 0]); rz.Set(0.0)
        self._no_physics(q)
        rl = spec.get("mood", {}).get("room_lights", 1.0)
        if rl != 1.0:                                                           # --mood: the room's own lights scaled (0: off)
            from pxr import UsdLux
            n_, m_ = 0, 0
            for p_ in Usd.PrimRange(q, Usd.TraverseInstanceProxies()):
                if p_.HasAPI(UsdLux.LightAPI):
                    if p_.IsInstanceProxy():
                        m_ += 1; continue
                    if rl <= 0:
                        p_.SetActive(False)
                    else:
                        a_ = UsdLux.LightAPI(p_).GetIntensityAttr(); a_.Set(float(a_.Get() or 0.0) * rl)
                    n_ += 1
            e_ = []
            if rl <= 0:                                                         # lights off: their glowing fixture materials too
                from pxr import UsdShade
                for p_ in Usd.PrimRange(q):
                    if p_.IsA(UsdShade.Shader) and "emissive" in str(p_.GetPath()).lower():
                        for i_ in UsdShade.Shader(p_).GetInputs():
                            nm_ = i_.GetBaseName()
                            if "emissi" in nm_.lower():
                                v_ = i_.Get(); e_.append(f"{p_.GetName()}.{nm_}={v_}")
                                if isinstance(v_, bool):
                                    i_.Set(False)
                                elif isinstance(v_, (int, float)):
                                    i_.Set(type(v_)(0))
                                elif v_ is not None and hasattr(v_, "__len__"):
                                    i_.Set(type(v_)(*([0.0] * len(v_))))
            print(f"[watch-dr] mood: room lights x{rl}: {n_} changed, {m_} inside instances (unchanged); emissive inputs off: {e_}", flush=True)
        bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
        B = []
        for p_ in Usd.PrimRange(q, Usd.TraverseInstanceProxies()):
            if p_.IsA(UsdGeom.Gprim):
                r_ = bc.ComputeWorldBound(p_).ComputeAlignedRange()
                if not r_.IsEmpty():
                    B.append([*r_.GetMin(), *r_.GetMax()])
        B = np.array(B, float); lo, hi = B[:, :3], B[:, 3:]; sz = hi - lo
        flat = (sz[:, 2] < 0.3) & (sz[:, 0] * sz[:, 1] > 4.0)
        if flat.any():
            f0 = np.argmax(np.where(flat, sz[:, 0] * sz[:, 1], -1)); fz = hi[f0, 2]
            fl = flat & (np.abs(hi[:, 2] - fz) < 0.05)
        else:
            fz = lo[:, 2].min(); fl = np.ones(len(B), bool)
        obst = (hi[:, 2] > fz + 0.03) & (lo[:, 2] < fz + 1.9) & ~((sz[:, 0] > 3) & (sz[:, 1] > 3))
        g = 0.05; x0, y0 = lo[fl, 0].min(), lo[fl, 1].min(); nx, ny = int((hi[fl, 0].max() - x0) / g) + 1, int((hi[fl, 1].max() - y0) / g) + 1
        free = np.zeros((nx, ny), bool)
        for b in B[fl]:
            free[int((b[0] - x0) / g):int(np.ceil((b[3] - x0) / g)), int((b[1] - y0) / g):int(np.ceil((b[4] - y0) / g))] = True
        roof = (lo[:, 2] > fz + 1.9) & (sz[:, 0] > 0.5) & (sz[:, 1] > 0.5)     # ceilings / roofs: only floor under one counts (indoors)
        if roof.any():
            cov = np.zeros_like(free)
            for b in B[roof]:
                cov[max(0, int((b[0] - x0) / g)):max(0, int(np.ceil((b[3] - x0) / g))), max(0, int((b[1] - y0) / g)):max(0, int(np.ceil((b[4] - y0) / g)))] = True
            free &= cov
        occ = np.zeros_like(free)
        for b in B[obst]:
            occ[max(0, int((b[0] - x0) / g)):max(0, int(np.ceil((b[3] - x0) / g))), max(0, int((b[1] - y0) / g)):max(0, int(np.ceil((b[4] - y0) / g)))] = True
        free &= ~occ
        from scipy.ndimage import distance_transform_edt
        clr = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * g           # m to the nearest obstacle / off-floor cell
        tb = self.S["table"]; P0 = np.array([o["pos"] for o in self.S["objects"]], float); ctr = P0[:, :2].mean(0); cs = spec["cam"]
        gx, gy = np.meshgrid(np.arange(tb["x"][0] - 0.45, tb["x"][1] + 0.45, 0.1), np.arange(tb["y"][0] - 0.45, tb["y"][1] + 0.45, 0.1))
        K_ = np.stack([gx.ravel(), gy.ravel()], 1)                              # keep-out: the table + 0.45 m all round (robot, person)
        st_ = max(0.1, math.sqrt(nx * ny * g * g / 20000))                    # <= ~20k candidate spots in a big room
        cx, cy = np.meshgrid(x0 + np.arange(0, nx * g, st_), y0 + np.arange(0, ny * g, st_)); C = np.stack([cx.ravel(), cy.ravel()], 1)
        cell = lambda P: np.floor((P - [x0, y0]) / g).astype(int)               # noqa: E731
        inside = lambda ij: (ij[..., 0] >= 0) & (ij[..., 1] >= 0) & (ij[..., 0] < nx) & (ij[..., 1] < ny)   # noqa: E731
        cands = []
        for qt in range(4):
            yaw = qt * math.pi / 2; Rz = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            for a in range(0, len(C), 2000):
                Cc = C[a:a + 2000]; ij = cell((K_ @ Rz.T)[None] + Cc[:, None]); ins = inside(ij).all(1)
                ijc = np.clip(ij, 0, [nx - 1, ny - 1]); ok = ins & free[ijc[..., 0], ijc[..., 1]].all(1)
                for c in Cc[ok]:
                    o = cell(c + Rz @ ctr); cands.append((float(clr[o[0], o[1]]), c, yaw))
        cands.sort(key=lambda z: -z[0])
        top = cands[:5]
        pick = (tuple(top[min(len(top) - 1, int(rs["spot_u"] * len(top)))][1:]) + (len(cands),)) if top else None
        if pick is None:                                                        # no free spot: the room's centre (reported)
            pick = (np.array([(x0 + hi[fl, 0].max()) / 2, (y0 + hi[fl, 1].max()) / 2]), 0.0, 0)
        # the camera: within +-front deg of the robot's facing, its line of sight free, aimed where the room behind the table is deepest
        c_, yaw_, _ = pick; Rz = np.array([[math.cos(yaw_), -math.sin(yaw_)], [math.sin(yaw_), math.cos(yaw_)]])
        face = math.degrees(self.grip.base_yaw) if getattr(self, "grip", None) else 0.0
        dh = cs["dist"] * math.cos(math.radians(cs["el"])); best = None
        for az in np.arange(face - cs.get("front", 70), face + cs.get("front", 70) + 1e-6, 5.0):
            u = np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])
            seg = cell(c_ + (np.linspace(0, dh + 0.3, 25)[:, None] * u + ctr) @ Rz.T)
            if not (inside(seg).all() and free[np.clip(seg[:, 0], 0, nx - 1), np.clip(seg[:, 1], 0, ny - 1)].all()):
                continue
            depth, wall = 0.0, False
            for d in np.arange(0.5, 25.0, 0.1):                                 # behind the table, as the camera sees it
                q_ = cell(c_ + Rz @ (ctr - d * u))
                if not (inside(q_) and free[q_[0], q_[1]]):
                    wall = bool(inside(q_) and occ[q_[0], q_[1]]); break
                depth = d
            sc = min(depth, 12.0) - (0.0 if wall else 20.0) - 2.0 * abs(abs(az - face) - 35.0) / 70.0   # a deep backdrop ending on the room, 3/4 view
            if best is None or sc > best[0]:
                best = (sc, float(az), depth)
        if best is not None:
            rs.update(cam_az=best[1], backdrop_m=round(best[2], 1), backdrop_wall=bool(best[0] > -10))
        c, yaw, nok = pick
        Rm = np.array([[math.cos(-yaw), -math.sin(-yaw), 0], [math.sin(-yaw), math.cos(-yaw), 0], [0, 0, 1]])
        V3(t_, -Rm @ np.array([c[0], c[1], fz]) + [0, 0, self.FLOOR_Z] + self.org); rz.Set(float(math.degrees(-yaw)))
        rs.update(floor_z=float(fz), spot=[float(c[0]), float(c[1])], yaw_deg=float(math.degrees(yaw)), free_spots=int(nok), meshes=len(B), mpu=mpu)
        print(f"[watch-dr] room {rs['name']}: {len(B)} meshes, floor z {fz:.2f}, {int(obst.sum())} obstacles, {nok} free spots; "
              f"table at {np.round(c, 2)} yaw {math.degrees(yaw):.0f}, clearance {top[0][0] if top else 0:.1f} m; camera az {rs.get('cam_az')}, "
              f"backdrop {rs.get('backdrop_m')} m", flush=True)

    def mood(self, spec, sp, ctr):
        """The --mood lights (MOODS), placed from the camera's direction (set by the room / outdoor placement): lamp = a sphere light in an
        open dark-metal shade low over the objects' centre (a cord up, an emissive bulb), its beam shaped downwards; sun = a sharp distant
        light from behind the table (opposite the camera, turned off-axis by `off` deg, at elevation `el`) so a glossy top shows its glint;
        flood = a rect light `dist` m out to the camera's side, `z` m up, aimed at the objects."""
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
        M = spec["mood"]; az = math.radians(spec.get("room", {}).get("cam_az") or spec["cam"].get("az", 0.0))
        c = np.array([ctr[0], ctr[1], 0.0])

        def put(L, pos, d, m):
            x = UsdGeom.Xformable(L.GetPrim()); x.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(pos, float) + self.org)))
            q = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*[float(a) for a in d])).GetQuat()
            x.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*q.GetImaginary())))
            L.CreateIntensityAttr(float(m["intensity"])); L.CreateNormalizeAttr(m is not M.get("sun"))
            L.CreateEnableColorTemperatureAttr(True); L.CreateColorTemperatureAttr(float(m["temp"]))
            if m.get("cone"):
                sh = UsdLux.ShapingAPI.Apply(L.GetPrim()); sh.CreateShapingConeAngleAttr(float(m["cone"])); sh.CreateShapingConeSoftnessAttr(float(m["soft"]))
        if "lamp" in M:
            m = M["lamp"]; zb = m["z"]                                          # the shade's open rim, above the table top
            L = UsdLux.SphereLight.Define(self.st, f"{sp}/MoodLamp"); L.CreateRadiusAttr(0.03); put(L, c + [0, 0, zb + 0.05], [0, 0, -1], m)
            N, r0, r1, h = 48, m["r"], 0.04, m["r"]
            pts = [Gf.Vec3f(r * math.cos(2 * math.pi * i / N), r * math.sin(2 * math.pi * i / N), z) for r, z in ((r0, 0.0), (r1, h)) for i in range(N)]
            ms = UsdGeom.Mesh.Define(self.st, f"{sp}/LampShade"); ms.CreatePointsAttr(pts); ms.CreateFaceVertexCountsAttr([4] * N)
            ms.CreateFaceVertexIndicesAttr(sum([[i, (i + 1) % N, N + (i + 1) % N, N + i] for i in range(N)], []))
            ms.CreateDoubleSidedAttr(True); ms.CreateSubdivisionSchemeAttr("none")
            UsdGeom.Xformable(ms).AddTranslateOp().Set(Gf.Vec3d(*(c + [0, 0, zb] + self.org)))
            self._bind(ms.GetPrim(), self._pbr(f"{sp}/LampShadeMat", (0.04, 0.045, 0.05), rough=0.35))
            cy = UsdGeom.Cylinder.Define(self.st, f"{sp}/LampCord"); cy.CreateRadiusAttr(0.006); cy.CreateHeightAttr(1.6)
            UsdGeom.Xformable(cy).AddTranslateOp().Set(Gf.Vec3d(*(c + [0, 0, zb + h + 0.8] + self.org)))
            self._bind(cy.GetPrim(), self._pbr(f"{sp}/LampCordMat", (0.02, 0.02, 0.02), rough=0.6))
            bu = UsdGeom.Sphere.Define(self.st, f"{sp}/LampBulb"); bu.CreateRadiusAttr(0.028)
            UsdGeom.Xformable(bu).AddTranslateOp().Set(Gf.Vec3d(*(c + [0, 0, zb + 0.05] + self.org)))
            UsdGeom.PrimvarsAPI(bu.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            mb = UsdShade.Material.Define(self.st, f"{sp}/LampBulbMat"); sb = UsdShade.Shader.Define(self.st, f"{sp}/LampBulbMat/S"); sb.CreateIdAttr("UsdPreviewSurface")
            sb.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.8, 0.5))
            sb.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(6.0, 4.0, 2.0))
            mb.CreateSurfaceOutput().ConnectToSource(sb.ConnectableAPI(), "surface"); self._bind(bu.GetPrim(), mb)
        if "sun" in M:
            m = M["sun"]; a = az + math.pi + math.radians(m["off"] * spec.get("room", {}).get("side", 1.0)); e = math.radians(m["el"])
            L = UsdLux.DistantLight.Define(self.st, f"{sp}/MoodSun"); L.CreateAngleAttr(float(m["angle"]))
            put(L, c, -np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]), m)
        if "torch" in M:                                                        # a narrow shaped beam from beside the camera, aimed at the objects
            m = M["torch"]; a = az + math.radians(m["az"]); pos = c + [m["dist"] * math.cos(a), m["dist"] * math.sin(a), m["z"]]
            L = UsdLux.SphereLight.Define(self.st, f"{sp}/MoodTorch"); L.CreateRadiusAttr(0.02); put(L, pos, c + [0, 0, 0.05] - pos, m)
        for q_, m in enumerate(M.get("neon", [])):                              # coloured rect lights aimed at the objects
            a = az + math.radians(m["az"]); pos = c + [m["dist"] * math.cos(a), m["dist"] * math.sin(a), m["z"]]
            L = UsdLux.RectLight.Define(self.st, f"{sp}/MoodNeon{q_}"); L.CreateWidthAttr(float(m["size"][0])); L.CreateHeightAttr(float(m["size"][1]))
            put(L, pos, c + [0, 0, 0.1] - pos, m); L.GetEnableColorTemperatureAttr().Set(False); L.CreateColorAttr(Gf.Vec3f(*m["color"]))
        if "flood" in M:
            m = M["flood"]; a = az + math.radians(m["az"])
            pos = c + [m["dist"] * math.cos(a), m["dist"] * math.sin(a), m["z"]]
            L = UsdLux.RectLight.Define(self.st, f"{sp}/MoodFlood"); L.CreateWidthAttr(float(m["size"][0])); L.CreateHeightAttr(float(m["size"][1]))
            put(L, pos, c + [0, 0, 0.1] - pos, m)
            from pxr import Usd
            M4 = UsdGeom.Xformable(L.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            print(f"[watch-dr] mood flood at {np.round(pos, 2)} (table frame), aimed at {np.round(c + [0, 0, 0.1], 2)}; want dir "
                  f"{np.round((c + [0, 0, 0.1] - pos) / np.linalg.norm(c + [0, 0, 0.1] - pos), 2)}, USD -Z -> {M4.TransformDir(Gf.Vec3d(0, 0, -1))}, org {self.org}", flush=True)
        print(f"[watch-dr] mood {M['name']}: camera az {math.degrees(az):.0f} deg", flush=True)
        from pxr import Usd
        for p_ in Usd.PrimRange(self.st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):    # every light in the stage, for the record
            if p_.HasAPI(UsdLux.LightAPI):
                print(f"[watch-dr] light {p_.GetPath()} {p_.GetTypeName()} intensity {UsdLux.LightAPI(p_).GetIntensityAttr().Get()}", flush=True)

    def table_model(self, spec, sp):
        """The table under the twin's top, built from it: the twin's footprint as a 4 cm top (top face at z = 0) in the drawn library wood /
        stone MDL, 4 square legs (inset 6 cm) down to the floor in the drawn metal MDL; render only (the objects stand on the twin's invisible
        table collider). Built rather than taken from the library: library table models stretched to the footprint look distorted."""
        tm = spec["table_model"]; tb = self.S["table"]; x0, x1, y0, y1 = tb["x"][0], tb["x"][1], tb["y"][0], tb["y"][1]
        mt = self._mdl(f"{sp}/MatTableTop", dict(tm["top"], world=True)); ml = self._mdl(f"{sp}/MatTableLeg", dict(tm["leg"], world=True))
        self._box(f"{sp}/TableTop", [x0, y0, -0.04], [x1, y1, 0.0], mt)
        for i, (lx, ly) in enumerate([(x0 + 0.06, y0 + 0.06), (x1 - 0.11, y0 + 0.06), (x0 + 0.06, y1 - 0.11), (x1 - 0.11, y1 - 0.11)]):
            self._box(f"{sp}/TableLeg{i}", [lx, ly, self.FLOOR_Z], [lx + 0.05, ly + 0.05, -0.04], ml)

    def camera(self, spec):
        S = self.S; P0 = np.array([o["pos"] for o in S["objects"]], float)
        cs = spec["cam"]; Wd, Hd = args.w, args.h
        c = S["cameras"]["main"]; fx = float(np.array(c["K"], float)[0, 0]) * Wd / c["wh"][0]                   # the real phone camera (K scaled)
        if cs.get("orbit"):
            fx = (Wd / 2) / math.tan(math.radians(cs["hfov"]) / 2)
        elif not cs.get("phone"):
            fx = Wd * float(cs["focal"]) / HAP
        focal = fx * HAP / Wd
        Kset = np.array([[fx, 0, Wd / 2], [0, fx, Hd / 2], [0, 0, 1]], np.float32)
        self.cam.set_intrinsic_matrices(torch.tensor(Kset[None], device=self.sim.device), focal_length=focal)
        self.look = np.array([P0[:, 0].mean(), P0[:, 1].mean(), 0.0]) + np.asarray(cs.get("look", [0, 0, 0.05]))
        self.dist = float(cs.get("dist", 1.0))
        if cs.get("orbit"):                                                     # any angle about the objects' centre
            self.look = np.array([P0[:, 0].mean(), P0[:, 1].mean(), cs["look_z"]]); self.dist = cs["dist"]; self.az0, self.el0 = cs["az"], cs["el"]
            if spec.get("room", {}).get("cam_az") is not None:                  # rooms: the direction Renderer.room chose
                self.az0 = spec["room"]["cam_az"]
        if cs.get("phone_orbit"):                                               # the phone camera in spherical coordinates about the objects' centre
            self.look = np.array([P0[:, 0].mean(), P0[:, 1].mean(), 0.05])
            v = np.array(S["cameras"]["main"]["T_cam_to_table"], float)[:3, 3] - self.look
            self.dist = float(np.linalg.norm(v)) * cs["dk"]
            self.az0 = math.degrees(math.atan2(v[1], v[0])) + cs["daz"]
            self.el0 = float(np.clip(math.degrees(math.asin(v[2] / np.linalg.norm(v))) + cs["dl"], *json.load(open(args.ranges))["cam_el_clip"]))
        self.set_view(0.0, spec)
        self.sim.forward()
        for _ in range(3):
            self.sim.render()
        self.cam.update(0.0, force_recompute=True)
        if args.wm_input:                                                       # the camera as an OpenCV K + camera-to-table pose (for drawing)
            from scipy.spatial.transform import Rotation as Rot
            qw = self.cam.data.quat_w_ros[0].cpu().numpy()                      # (x, y, z, w) in Isaac Lab 3, OpenCV axes
            Tc = np.eye(4); Tc[:3, :3] = Rot.from_quat(qw).as_matrix(); Tc[:3, 3] = self.cam.data.pos_w[0].cpu().numpy() - self.org
            self.cam_rec = dict(K=self.cam.data.intrinsic_matrices[0].cpu().numpy().tolist(), wh=[Wd, Hd], T_cam_to_table=Tc.tolist(), view=args.wm_cam)
        print(f"[watch-dr] camera fx {self.cam.data.intrinsic_matrices[0, 0, 0].item():.1f} (set {fx:.1f}), focal {focal:.2f} mm, "
              f"pos {np.round(self.cam.data.pos_w[0].cpu().numpy() - self.org, 3)}", flush=True)

    def set_view(self, u, spec):
        cs = spec["cam"]
        if cs.get("phone"):
            from scipy.spatial.transform import Rotation as Rot
            Tm = np.array(self.S["cameras"]["main"]["T_cam_to_table"], float)
            q = Rot.from_matrix(Tm[:3, :3]).as_quat()                           # (x, y, z, w), OpenCV ('ros') axes, as ep.rtx
            self.cam.set_world_poses(torch.tensor([Tm[:3, 3] + self.org], dtype=torch.float32, device=self.sim.device),
                                     torch.tensor([q], dtype=torch.float32, device=self.sim.device), convention="ros")
            return
        s = u * u * (3 - 2 * u)
        if cs.get("phone_orbit") or cs.get("orbit"):
            az = math.radians(self.az0 + cs["sweep"] * (s - 0.5)); el = math.radians(self.el0)
        else:
            az = math.radians(cs["az0"] + (cs["az1"] - cs["az0"]) * s); el = math.radians(cs["el0"] + (cs.get("el1", cs["el0"]) - cs["el0"]) * s)
        eye = self.look + self.dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        self.cam.set_world_poses_from_view(torch.tensor([eye + self.org], dtype=torch.float32, device=self.sim.device),
                                           torch.tensor([self.look + self.org], dtype=torch.float32, device=self.sim.device))

    def start(self):
        """The twin's start layout (rest, zero velocity), then 0.25 s of settling before the clip."""
        dev = self.sim.device; ids = torch.tensor([0], device=dev)
        for k, b in enumerate(self.bodies):
            q0 = self.demo.init["root_quat"][k] if self.demo.init else [0.0, 0.0, 0.0, 1.0]
            pose = torch.tensor([[*(self.demo.X0[k] + self.org + [0, 0, 0.002]), *q0]], dtype=torch.float32, device=dev)   # xyzw
            b.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            b.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
        self.dt = self.sim.get_physics_dt()
        for _ in range(int(0.25 / self.dt)):
            self.sim.step(render=False)
        for b in self.bodies:
            b.update(self.dt)
        P = np.stack([b.data.root_link_pos_w.torch[0].cpu().numpy() for b in self.bodies]) - self.org
        print(f"[watch-dr] physics dt {self.dt:.4f} s; settle check: max move {np.abs(P - self.demo.X0).max() * 100:.2f} cm", flush=True)

    def advance(self, t1):
        """Step PhysX up to video time t1. A held object is velocity-driven to its track (collisions stay physical); when something
        blocks it sideways, it rises by LIFT x its sideways lag (as a hand lifts over an obstacle). A free object is simulated: gravity,
        collisions, friction."""
        TAU, VMAX, WMAX, LIFT = 0.06, 1.5, 8.0, float(os.environ.get("WD_LIFT", "1.5"))
        dev = self.sim.device; ids = torch.tensor([0], device=dev)
        if args.carry_exact:                                                    # held: exactly on the track, collisions off; let go: PhysX
            from pxr import Usd, UsdPhysics
            if not hasattr(self, "coll"):
                self.coll = [[q for q in Usd.PrimRange(self.st.GetPrimAtPath(f"/World/WD/Scene/O{k}")) if q.HasAPI(UsdPhysics.CollisionAPI)]
                             for k in range(len(self.bodies))]
                self.carried, self.cq = set(), {}
            while self.t_sim < t1 - 1e-9:
                X, _, held = self.demo.at(self.t_sim + self.dt)
                for k, b in enumerate(self.bodies):
                    if held[k] and k not in self.carried:                       # grabbed: keep its orientation, collisions off
                        self.carried.add(k); self.cq[k] = b.data.root_link_quat_w.torch[0].cpu().numpy().tolist()
                        for q in self.coll[k]:
                            UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Set(False)
                    elif not held[k] and k in self.carried:                     # let go: collisions back, at rest, gravity takes over
                        self.carried.discard(k)
                        for q in self.coll[k]:
                            UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Set(True)
                        b.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
                    if k in self.carried:
                        b.write_root_pose_to_sim_index(root_pose=torch.tensor([[*(X[k] + self.org), *self.cq[k]]], dtype=torch.float32, device=dev), env_ids=ids)
                        b.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
                self.sim.step(render=False)
                for b in self.bodies:
                    b.update(self.dt)
                self.t_sim += self.dt
            return
        if args.kinematic:                                                      # physics off: every object exactly on its track
            X, _, _ = self.demo.at(t1)
            for k, b in enumerate(self.bodies):
                q0 = self.demo.init["root_quat"][k] if self.demo.init else [0.0, 0.0, 0.0, 1.0]
                b.write_root_pose_to_sim_index(root_pose=torch.tensor([[*(X[k] + self.org), *q0]], dtype=torch.float32, device=dev), env_ids=ids)
            self.sim.step(render=False)
            for b in self.bodies:
                b.update(self.dt)
            self.t_sim = t1
            return
        while self.t_sim < t1 - 1e-9:
            X, Y, held = self.demo.at(self.t_sim + self.dt)
            for k, b in enumerate(self.bodies):
                if not held[k]:
                    continue
                p = b.data.root_link_pos_w.torch[0].cpu().numpy() - self.org; q = b.data.root_link_quat_w.torch[0].cpu().numpy()   # xyzw
                tgt = X[k].copy(); tgt[2] += LIFT * float(np.linalg.norm(tgt[:2] - p[:2]))
                v = (tgt - p) / TAU; v *= min(1.0, VMAX / max(np.linalg.norm(v), 1e-9))
                if self.demo.use_yaw[k]:                                        # turn towards the tracked yaw (upright)
                    qt = np.array([0.0, 0.0, math.sin(Y[k] / 2), math.cos(Y[k] / 2)])
                    x1, y1, z1, w1 = qt; x2, y2, z2, w2 = -q[0], -q[1], -q[2], q[3]     # qt * conj(q)
                    e = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
                    if e[3] < 0:
                        e = -e
                    ang = 2 * math.acos(min(1.0, e[3])); ax = e[:3] / max(np.linalg.norm(e[:3]), 1e-9)
                    w = ax * ang / TAU; w *= min(1.0, WMAX / max(np.linalg.norm(w), 1e-9))
                else:
                    w = b.data.root_com_ang_vel_w.torch[0].cpu().numpy() * 0.8
                b.write_root_velocity_to_sim_index(root_velocity=torch.tensor([[*v, *w]], dtype=torch.float32, device=dev), env_ids=ids)
            self.sim.step(render=False)
            for b in self.bodies:
                b.update(self.dt)
            self.t_sim += self.dt
            self.n_step = getattr(self, "n_step", 0) + 1
            if os.environ.get("WD_DEBUG") and self.n_step % 60 == 0:
                P = np.stack([b.data.root_link_pos_w.torch[0].cpu().numpy() for b in self.bodies]) - self.org
                print(f"[dbg] t {self.t_sim - self.demo.t0:4.1f} " + " | ".join(f"{'H' if h else '.'} sim {np.round(P[k], 2).tolist()} tgt {np.round(X[k], 2).tolist()}"
                                                                         for k, h in enumerate(held)), flush=True)

    def place_grippers(self, t):
        """Each gripper at video time t: on its hand's path, pulled onto the object its hand holds (Gripper.weights), fingers at the
        hand's opening or closed on that object."""
        from pxr import Gf, UsdGeom
        from scipy.spatial.transform import Rotation as Rot
        gr = self.grip
        if self.arm_parts and len(self.q) == 2:                                # the previous frame's poses: did the two arms touch?
            d = float(self.FK.arm_arm_min_dist(self.q["left"][0], self.Bw["left"], self.q["right"][0], self.Bw["right"]))
            self.ncol += d < 0; self.dmin = min(getattr(self, "dmin", 9.0), d)
        T = {}
        for s_, (prim, ops) in self.gprims.items():
            if not gr.vis(s_, t):
                if self.arm_parts:                                              # no hand in view: the arm eases back to its home pose
                    self.arm_ik(s_, None, None, 0.08, ops)
                else:
                    UsdGeom.Imageable(prim).MakeInvisible()
                continue
            UsdGeom.Imageable(prim).MakeVisible()
            g, Rm, w = gr.at(s_, t); W = gr.weights(s_, t)
            for key in [kk for kk in gr.lock if kk[0] == s_ and kk[1] not in W]:
                del gr.lock[key]                                                # a new held run takes a new grip point
            tot = sum(W.values()); sc = 1.0 / max(1.0, tot); tgt = (1 - min(1.0, tot)) * g; wf = (1 - min(1.0, tot)) * w
            for k, wk in W.items():
                b = self.bodies[k]; p = b.data.root_link_pos_w.torch[0].cpu().numpy() - self.org
                Rb = Rot.from_quat(b.data.root_link_quat_w.torch[0].cpu().numpy()).as_matrix()      # xyzw
                if (s_, k) not in gr.lock:
                    lo, hi = self.obox[k]; size = hi - lo                      # (anchored objects fit between the fingers: grip the centre)
                    gr.lock[(s_, k)] = ((lo + hi) / 2, float(min(size.min(), 0.08)))
                loc, gw = gr.lock[(s_, k)]
                tgt = tgt + wk * sc * (p + Rb @ loc); wf += wk * sc * gw
            T[s_] = [tgt, Rm, wf, min(1.0, tot), ops]
        if self.arm_parts and len(T) == 2:                                      # keep the two grippers >= GAP apart (sideways): the free one gives way
            d = T["left"][0][:2] - T["right"][0][:2]; n = float(np.linalg.norm(d))
            if n < Gripper.GAP:
                u = d / n if n > 1e-3 else np.array([-gr.face[1], gr.face[0]])
                fl, fr = 1.0 - T["left"][3], 1.0 - T["right"][3]
                fl, fr = (0.5, 0.5) if fl + fr < 1e-6 else (fl / (fl + fr), fr / (fl + fr))
                T["left"][0] = T["left"][0] + np.array([*(u * (Gripper.GAP - n) * fl), 0.0])
                T["right"][0] = T["right"][0] - np.array([*(u * (Gripper.GAP - n) * fr), 0.0])
                self.npush = getattr(self, "npush", 0) + 1
        for s_, (tgt, Rm, wf, _, ops) in T.items():
            if self.arm_parts:
                self.arm_ik(s_, tgt, Rm, wf, ops)
                continue
            o = tgt - Rm[:, 2] * Gripper.TCP + self.org
            ops[0].Set(Gf.Vec3d(*o.tolist())); ops[1].Set(Gf.Quatd(*_R_to_wxyz(Rm)))
            ops[2].Set(Gf.Vec3d(0.0, wf / 2, 0.0584)); ops[3].Set(Gf.Vec3d(0.0, -wf / 2, 0.0584))

    def arm_ik(self, s_, tgt, Rm, wf, ops):
        """One arm to the gripper target (table frame TCP position + top-down rotation): damped-least-squares IK (franka_kinematics.ik) warm-started
        from the last pose; the gripper is symmetric, so both finger directions are tried and the one closer to the last pose kept.
        tgt None: the joints move towards the home pose (<= 0.06 rad per frame). Then every link from forward kinematics."""
        from pxr import Gf
        FK = self.FK; B = self.Bw[s_]
        if s_ not in self.q:
            self.q[s_] = torch.tensor(FK.Q_HOME, dtype=torch.float64)[None]
        q0 = self.q[s_]
        if tgt is None:
            qh = torch.tensor(FK.Q_HOME, dtype=torch.float64)[None]; q = q0 + (qh - q0).clamp(-0.06, 0.06)
        else:
            best = None
            for flip in (1.0, -1.0):
                T = np.eye(4); T[:3, :3] = Rm * np.array([flip, flip, 1.0]); T[:3, 3] = tgt
                tb = torch.linalg.inv(B) @ torch.tensor(T)
                first = getattr(self, "ik_n", 0) < 2
                q, ok, pe, re = FK.ik(tb[None], q0, iters=120 if first else 40, q_rest=self.q_rest[s_], k_null=0.3, n_seeds=8 if first else 1)
                cost = float(pe) + 0.05 * float(re) + 0.02 * float((q - q0).abs().max())
                if best is None or cost < best[0]:
                    best = (cost, q, float(pe))
            self.ik_n = getattr(self, "ik_n", 0) + 1
            q = best[1] if best[2] <= 0.08 else q0                             # out of reach: hold the last pose (no flailing)
            self.ik_err = max(getattr(self, "ik_err", 0.0), best[2]); self.ik_held = getattr(self, "ik_held", 0) + (best[2] > 0.08)
        self.q[s_] = q
        F = (B @ FK.fk(q[0])).numpy()
        for i, (t_, r_) in enumerate(self.arm_parts[s_]):
            t_.Set(Gf.Vec3d(*(F[i, :3, 3] + self.org))); r_.Set(Gf.Quatd(*_R_to_wxyz(F[i, :3, :3])))
        H = F[FK.F_IDX["hand"]]
        ops[0].Set(Gf.Vec3d(*(H[:3, 3] + self.org))); ops[1].Set(Gf.Quatd(*_R_to_wxyz(H[:3, :3])))
        ops[2].Set(Gf.Vec3d(0.0, wf / 2, 0.0584)); ops[3].Set(Gf.Vec3d(0.0, -wf / 2, 0.0584))

    def grab(self):
        for _ in range(int(os.environ.get("WD_PASSES", "6"))):
            self.sim.render()
        self.cam.update(0.0, force_recompute=True)
        rgb = self.cam.data.output["rgb"]; rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[0, ..., :3].cpu().numpy().astype(np.uint8)
        if args.wm_input:                                                       # object masks -> one label image (k + 1; 0 = none)
            seg = self.cam.data.output["semantic_segmentation"]; seg = (seg.torch if hasattr(seg, "torch") else seg)[0].cpu().numpy()
            seg = seg[..., 0] if seg.ndim == 3 and seg.shape[-1] == 1 else seg
            info = self.cam.data.info.get("semantic_segmentation") if isinstance(self.cam.data.info, dict) else (
                self.cam.data.info[0].get("semantic_segmentation") if isinstance(self.cam.data.info, list) else None)
            lab = np.zeros(seg.shape[:2], np.uint8)
            for kk, v in ((info or {}).get("idToLabels", {})).items():
                name = v.get("class", "") if isinstance(v, dict) else str(v)
                for t_ in [t_.strip() for t_ in name.split(",")]:
                    if re.fullmatch(r"o\d+", t_):
                        ks = str(kk).strip()
                        if ks.startswith("("):
                            col = np.array([int(float(x)) for x in ks.strip("()").split(",")])
                            rgba = seg if seg.ndim == 3 else np.ascontiguousarray(seg).view(np.uint8).reshape(*seg.shape, -1)
                            m_ = (rgba[..., :len(col)] == col).all(-1)
                        else:
                            m_ = (seg if seg.ndim == 2 else seg[..., 0]) == int(ks)
                        lab[m_] = int(t_[1:]) + 1
            self.last_mask = lab
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ---- composition -------------------------------------------------------------------------------------------------------------------
    def label(self, img, text, scale=0.55):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ov = img.copy(); cv2.rectangle(ov, (0, 0), (tw + 20, th + 16), (20, 20, 20), -1)
        img[:] = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
        cv2.putText(img, text, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)

    def caption(self, spec, t, Wt):
        img = np.full((CAP_H, Wt, 3), 24, np.uint8); F_ = cv2.FONT_HERSHEY_SIMPLEX

        def put(tx, y, sc_, c, x0=14, w=Wt - 28, th=1):
            while sc_ > 0.3 and cv2.getTextSize(tx, F_, sc_, th)[0][0] > w:
                sc_ -= 0.02
            cv2.putText(img, tx, (x0, y), F_, sc_, c, th, cv2.LINE_AA)
        put(spec["title"], 30, 0.8, (245, 245, 245), th=2)
        r = f"{args.episode}   t = {t - self.demo.t0:4.1f} s (real time)"
        (tw, _), _ = cv2.getTextSize(r, F_, 0.6, 1); cv2.putText(img, r, (Wt - tw - 14, 30), F_, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        put(spec["desc"], 58, 0.55, (185, 185, 185))
        put(("Drawn automatically from the twin + Isaac's asset libraries by a seed; nothing placed by hand. " if spec.get("seed") is not None else "")
            + "The WM was trained on the real videos + their twins (object level); visual randomisation like this was NOT used in training.",
            84, 0.5, (120, 200, 250))
        vs = sorted(set(n for n in self.shown if n)); om = [f"{len(spec.get('obj_mat', {}))} re-materialled"] if spec.get("obj_mat") else []
        l4 = ("PhysX: objects follow the SAM 3 tracks the WM watched while a contact model says a hand touches them, else fall / settle freely."
              + (f"  Render-only object variants: {', '.join(vs + om)}." if vs or om else ""))
        put(l4, 108, 0.46, (150, 150, 150))
        return img

    def world_clip(self, wn, spec, real, out):
        d = self.demo; self.t_sim = d.t0; n = int(round((d.t1 - d.t0) * args.fps)) + 1
        if args.max_frames:
            n = min(n, args.max_frames)
        twin_dir = out / f"twin_{args.episode}"
        if spec.get("twin"):
            twin_dir.mkdir(exist_ok=True)
        stem = f"{args.episode}__watch_dr_{wn}"
        frs = list(range(n)) if not args.stills else [min(n - 1, int(float(x) * (n - 1))) for x in args.stills.split(",")]
        if args.frame_list:
            frs = sorted({min(n - 1, max(0, int(x))) for x in json.load(open(args.frame_list))})
        vw = None
        fdir = out / f"frames_{args.episode}_{wn}"
        if args.frames_only:
            fdir.mkdir(exist_ok=True)
        if not args.stills and not spec.get("twin") and not args.frames_only:
            Wt, Ht = LW + args.w, args.h + CAP_H
            vw = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wt}x{Ht}", "-r", str(args.fps), "-i", "-",
                                   "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", str(out / f"{stem}.mp4")], stdin=subprocess.PIPE)
        t0 = time.time()
        want = set(frs); c = -1
        for fi in range(max(frs) + 1):
            t = d.t0 + fi / args.fps
            self.advance(t)
            if fi not in want:
                continue
            c += 1
            self.set_view(fi / max(1, n - 1), spec)
            if getattr(self, "grip", None):
                self.place_grippers(t)
            img = self.grab()
            if args.frames_only:
                cv2.imwrite(str(fdir / f"{fi:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if args.wm_input:
                    md = out / f"masks_{args.episode}_{wn}"; md.mkdir(exist_ok=True); cv2.imwrite(str(md / f"{fi:05d}.png"), self.last_mask)
                    from scipy.spatial.transform import Rotation as Rot             # the sim's truth: each object's top centre (its box)
                    tc = []
                    for k, b in enumerate(self.bodies):
                        lo, hi = self.obox[k]; Rb = Rot.from_quat(b.data.root_link_quat_w.torch[0].cpu().numpy()).as_matrix()
                        tc.append(b.data.root_link_pos_w.torch[0].cpu().numpy() - self.org + Rb @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]]))
                    self.truth = getattr(self, "truth", []); self.truth.append([fi, t, np.array(tc)])
                if fi == 0:
                    cv2.imwrite(str(out / f"{stem}_clean.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif spec.get("twin"):
                cv2.imwrite(str(twin_dir / f"{fi:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if args.stills:
                    cv2.imwrite(str(out / f"look_{stem}_{fi:05d}.jpg"), img)
                if fi == 0:
                    cv2.imwrite(str(out / f"{stem}_frame.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
            else:
                fr = self.compose(img, real, twin_dir, fi, spec, t)
                if vw is not None:
                    vw.stdin.write(np.ascontiguousarray(fr).tobytes())
                if args.stills:
                    cv2.imwrite(str(out / f"look_{stem}_{fi:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                elif fi == 0:                                                   # the still: the first frame
                    cv2.imwrite(str(out / f"{stem}_frame.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    cv2.imwrite(str(out / f"{stem}_clean.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if c % 40 == 0:
                print(f"[watch-dr] {wn}: frame {c}/{len(frs)} ({time.time() - t0:.0f} s)", flush=True)
        if args.wm_input and args.frames_only:
            json.dump(dict(self.cam_rec, episode=args.episode, world=wn, fps=args.fps, objects=[o["id"] for o in self.S["objects"]],
                           root_pos=[(b.data.root_link_pos_w.torch[0].cpu().numpy() - self.org).tolist() for b in self.bodies],
                           root_quat=[b.data.root_link_quat_w.torch[0].cpu().numpy().tolist() for b in self.bodies],
                           frames=[x[0] for x in self.truth], t=[x[1] for x in self.truth], top_centre=[x[2].tolist() for x in self.truth]),
                      open(out / f"camera_{args.episode}_{wn}.json", "w"))
        if vw is not None:
            vw.stdin.close(); rc = vw.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed (rc {rc})")
            return out / f"{stem}.mp4"
        return fdir if args.frames_only else (twin_dir if spec.get("twin") else out)

    def compose(self, img, real, twin_dir, fi, spec, t):
        lh = args.h // 2
        def load(p):
            im = cv2.imread(str(p)) if p is not None and Path(p).exists() else None
            return cv2.resize(im, (LW, lh), interpolation=cv2.INTER_AREA) if im is not None else np.zeros((lh, LW, 3), np.uint8)
        rl = load(real[min(fi, len(real) - 1)] if real else None); tw = load(twin_dir / f"{fi:05d}.jpg")
        self.label(rl, "1  real phone video (what the WM watched)", 0.5); self.label(tw, "2  Isaac twin, same camera (as built)", 0.5)
        main = img.copy(); self.label(main, f"3  {spec['title']}", 0.65)
        left = np.vstack([rl, tw])
        cv2.line(left, (0, lh), (LW, lh), (24, 24, 24), 2)
        top = np.hstack([left, main]); cv2.line(top, (LW, 0), (LW, args.h), (24, 24, 24), 2)
        return np.vstack([top, self.caption(spec, t, top.shape[1])])


if __name__ == "__main__":
    main()
