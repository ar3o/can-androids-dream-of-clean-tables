"""Isaac RTX render of recorded episodes: a Recorder pickle (`cleantables.rl.chain_env.Recorder`: PhysX object poses at 10 Hz,
both physics hands' part poses, the two Franka arms' joint angles) played back kinematically in Isaac (RTX camera; physics is never
stepped).
  scene     the twin's USD meshes + table; the two Franka Panda USD arms (Isaac Lab's FRANKA_PANDA_LIFT_CFG) at the recording's bases
            (franka_bases_reach.json), posed from the recorded joint angles; the physics hands (the palm + finger boxes PhysX collides
            with) as translucent boxes, blue = gripper 0, orange = gripper 1
  overlays  projected into the image: the active carry path (solid, the object's colour), the hand's route (dashed; dotted grey = the
            zero-edit route when the learned edit moved it > 1 cm), the goal rings (green = met, orange = not), object names
  red       colliding arm links (arm-arm / arm-object / arm-table, the sphere model ArmTracker checks), hands overlapping or with fingers
            below the table top, objects pushed > 1 cm while held, and the aborted take's object (flashing)
  caption   per gripper the last decision (type / object / grasp yaw / result) and current events; a corner legend
  $SIM_PY cleantables/rl/render_rtx.py --recs baseline=REC1.pkl,trained=REC2.pkl --out DIR [--scenes a,b]
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import subprocess
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--recs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--scenes", default="")
ap.add_argument("--w", type=int, default=880); ap.add_argument("--h", type=int, default=620); ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--max_frames", type=int, default=0)
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
args.headless = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TWIN_ROOT = Path(os.environ.get("WM_TWIN_ROOT", str(ROOT / "data/twin_table/ep")))
RED = np.array([40, 40, 235], np.float32)
GCOL_BGR = [(220, 120, 30), (40, 140, 240)]                                     # gripper 0 blue, gripper 1 orange (BGR)
BODY_USD = {**{f"link{i}": f"panda_link{i}" for i in range(8)}, "hand": "panda_hand", "lfinger": "panda_leftfinger", "rfinger": "panda_rightfinger"}


def main():
    from dataclasses import replace

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass, instantiate
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg
    from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FRANKA_PANDA_LIFT_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        cam = CameraCfg(prim_path="{ENV_REGEX_NS}/SACam", update_period=0.0, data_types=["rgb", "semantic_segmentation"], width=args.w, height=args.h,
                        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 30.0)),
                        renderer_cfg=IsaacRtxRendererCfg(colorize_semantic_segmentation=False,
                                                         global_settings=IsaacRtxRendererGlobalSettingsCfg(antialiasing_mode=os.environ.get("SA_AA", "DLAA"), enable_dl_denoiser=os.environ.get("SA_DENOISE", "1") == "1",
                                                                                                           enable_dlssg=False, samples_per_pixel=int(os.environ.get("SA_SPP", "4")))))

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, physics=PhysxCfg(), gravity=(0.0, 0.0, 0.0))
    with launch_simulation(sim_cfg, args):
        sim = SimulationContext(sim_cfg)
        scene = instantiate(SceneCfg(num_envs=1, env_spacing=5.0))
        sim.reset()
        R = Renderer(sim, scene, FRANKA_PANDA_LIFT_CFG.spawn.usd_path)
        recs = {}
        for kv in args.recs.split(","):
            lab, p = kv.split("=", 1); recs[lab] = pickle.load(open(p, "rb"))
        labs = list(recs)
        names = [e["name"] for e in recs[labs[0]]["env"]]
        if args.scenes:
            names = [n for n in names if n in args.scenes.split(",")]
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        for nm in names:
            t0 = time.time()
            mp4 = R.scene_clip(nm, recs, out)
            print(f"[stageA-rtx] {nm} -> {mp4} ({time.time() - t0:.0f} s)", flush=True)
        print("[stageA-rtx] peak GPU memory (torch, this process) MiB", torch.cuda.max_memory_allocated() / 2**20, flush=True)


def _R_to_wxyz(R):
    """3x3 rotation -> (w, x, y, z)."""
    R = np.asarray(R, np.float64); tr = np.trace(R)
    if tr > 0:
        s = 2 * math.sqrt(tr + 1); return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]); return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if i == 1:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]); return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]); return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def _ancestors(q):
    a = q.GetParent()
    while a and a.IsValid() and str(a.GetPath()) != "/":
        yield a
        a = a.GetParent()


def _modified():
    """Scenes whose start layout was changed by the human-placement override (FPC_LAYOUT_OVERRIDE): marked 'modified' in every caption."""
    p = os.environ.get("FPC_LAYOUT_OVERRIDE")
    return set(__import__("json").load(open(p)).get("override", {})) if p and os.path.exists(p) else set()


class Renderer:
    def __init__(self, sim, scene, arm_usd):
        from pxr import UsdGeom, UsdLux
        import isaaclab.sim as sim_utils
        self.sim, self.scene, self.cam = sim, scene, scene["cam"]
        self.st = sim_utils.get_current_stage(); self.su = sim_utils
        self.org = scene.env_origins[0].cpu().numpy().astype(float)
        UsdGeom.Xform.Define(self.st, "/World/SA")
        d = UsdLux.DomeLight.Define(self.st, "/World/SA/Dome"); d.CreateIntensityAttr(900.0)
        k = UsdLux.DistantLight.Define(self.st, "/World/SA/Key"); k.CreateIntensityAttr(1500.0); k.CreateAngleAttr(8.0)
        UsdGeom.Xformable(k.GetPrim()).AddRotateXYZOp().Set((35.0, 10.0, 0.0))
        # The two arms are the Franka USD referenced as plain geometry (physics disabled) whose links are posed directly from the forward
        # kinematics of the recorded joint angles. Isaac Lab articulations are not moved by joint / root writes without a physics step.
        from pxr import Sdf, UsdPhysics
        self.arm_links = []
        for g in (0, 1):
            root = self.st.DefinePrim(f"/World/SA/Arm{g}", "Xform"); root.GetReferences().AddReference(arm_usd)
            xr = UsdGeom.Xformable(root); xr.ClearXformOpOrder()
            self.arm_root = getattr(self, "arm_root", []) + [(xr.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), xr.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))]
            links = {}
            for q in [root] + list(__import__("pxr").Usd.PrimRange(root)):
                if q.HasAPI(UsdPhysics.RigidBodyAPI):
                    q.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                if q.HasAPI(UsdPhysics.ArticulationRootAPI):
                    q.CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                    q.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if q.IsA(UsdPhysics.Joint):
                    q.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                for body, usd in BODY_USD.items():
                    if q.GetName() == usd and body not in links:
                        links[body] = q
            row = {}
            inv = {v.GetPath(): k for k, v in links.items()}
            self.arm_parent = getattr(self, "arm_parent", []) + [{b: next((inv[a.GetPath()] for a in _ancestors(q) if a.GetPath() in inv), None)
                                                                   for b, q in links.items()}]
            for body, q in links.items():
                x = UsdGeom.Xformable(q)
                ex = {o.GetOpName(): o for o in x.GetOrderedXformOps()}; x.ClearXformOpOrder()            # reuse the asset's own op attributes
                to = x.AddTranslateOp(ex["xformOp:translate"].GetPrecision()) if "xformOp:translate" in ex else x.AddTranslateOp()
                ro = x.AddOrientOp(ex["xformOp:orient"].GetPrecision()) if "xformOp:orient" in ex else x.AddOrientOp()
                row[body] = (to, ro)
                self.su.add_labels(q, [f"a{g}_{body}"])
            print(f"[stageA-rtx] arm {g}: links found {sorted(links)} (parent of link1: {links['link1'].GetParent().GetName() if 'link1' in links else None})", flush=True)
            self.arm_links.append(row)
        self.hands = None

    def _material(self, path, rgb, opacity):
        from pxr import Gf, Sdf, UsdShade
        mat = UsdShade.Material.Define(self.st, path)
        sh = UsdShade.Shader.Define(self.st, path + "/S"); sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def build(self, twin, bases):
        """the twin's table + objects (USD references, rigid bodies disabled: playback only), the physics hands' boxes, the arms' bases."""
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        sp = "/World/SA/Scene"
        if self.st.GetPrimAtPath(sp):
            self.st.RemovePrim(sp)
        UsdGeom.Xform.Define(self.st, sp)
        S = __import__("json").load(open(TWIN_ROOT / twin / "scene.json")); tb = S["table"]
        cube = UsdGeom.Cube.Define(self.st, f"{sp}/Table"); cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d((tb["x"][0] + tb["x"][1]) / 2 + self.org[0], (tb["y"][0] + tb["y"][1]) / 2 + self.org[1], -0.02 + self.org[2]))
        xf.AddScaleOp().Set(Gf.Vec3d(tb["x"][1] - tb["x"][0], tb["y"][1] - tb["y"][0], 0.04))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.55, 0.47)])
        self.su.add_labels(cube.GetPrim(), ["table"])
        fl = UsdGeom.Cube.Define(self.st, f"{sp}/Floor"); fl.CreateSizeAttr(1.0)
        fx = UsdGeom.Xformable(fl.GetPrim()); fx.AddTranslateOp().Set(Gf.Vec3d(*(self.org + [0, 0, -0.77]))); fx.AddScaleOp().Set(Gf.Vec3d(8, 8, 0.02))
        fl.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
        self.objs = []
        for k, o in enumerate(S["objects"]):
            p = self.st.DefinePrim(f"{sp}/O{k}", "Xform")
            p.GetReferences().AddReference(str(ROOT / o["usd"]))
            for q in [p] + list(p.GetAllChildren()):
                if q.HasAPI(UsdPhysics.RigidBodyAPI):
                    q.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
            x = UsdGeom.Xformable(p); x.ClearXformOpOrder()
            t_ = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble); r_ = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            sc = float(o.get("scale") or 1.0); x.AddScaleOp().Set(Gf.Vec3f(sc, sc, sc))
            self.su.add_labels(p, [f"o{k}"])
            self.objs.append((t_, r_))
        # the physics hands: palm + 2 fingers per gripper, translucent
        from cleantables.rl.gripper import FL, FT, FW, PH, PL, PW
        half = [np.array([PL, PW, PH]) / 2, np.array([FT, FW, FL]) / 2, np.array([FT, FW, FL]) / 2]
        self.hands = []
        for g in (0, 1):
            mat = self._material(f"{sp}/HandMat{g}", [(0.12, 0.47, 0.86), (0.94, 0.55, 0.16)][g], 0.45)
            row = []
            for i in range(3):
                c = UsdGeom.Cube.Define(self.st, f"{sp}/H{g}_{i}"); c.CreateSizeAttr(1.0)
                UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
                x = UsdGeom.Xformable(c.GetPrim())
                t_ = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble); r_ = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                x.AddScaleOp().Set(Gf.Vec3f(*(2 * half[i]).tolist()))
                self.su.add_labels(c.GetPrim(), [f"h{g}"])
                row.append((t_, r_))
            self.hands.append(row)
        self.S, self.n = S, len(S["objects"])
        # the arms at this recording's bases
        self.bases = np.asarray(bases, np.float64)
        for g in (0, 1):                                                    # the arm roots at the bases (the links are posed in the base frame)
            B = self.bases[g]; t_, r_ = self.arm_root[g]
            t_.Set(Gf.Vec3d(*(B[:3, 3] + self.org))); r_.Set(Gf.Quatd(*_R_to_wxyz(B[:3, :3])))
        # camera: fixed 3/4 view framing the table and both arm bases
        bx = np.asarray(bases)[:, :2, 3]
        lo = np.minimum([tb["x"][0], tb["y"][0]], bx.min(0)); hi = np.maximum([tb["x"][1], tb["y"][1]], bx.max(0))
        c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.12]); span = float(max(hi - lo))
        az, el, dist = np.radians(225.0), np.radians(38.0), 0.80 * span + 0.45
        eye = c + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.cam.set_world_poses_from_view(torch.tensor([eye + self.org], dtype=torch.float32, device=self.sim.device),
                                           torch.tensor([c + self.org], dtype=torch.float32, device=self.sim.device))
        self.sim.forward()
        for _ in range(3):
            self.sim.render()
        self.cam.update(0.0, force_recompute=True)
        self.K = self.cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64)
        self.cpos = self.cam.data.pos_w[0].cpu().numpy().astype(np.float64)
        q4 = self.cam.data.quat_w_ros[0].cpu().numpy().astype(np.float64)
        best = None
        for qw, qx, qy, qz in (q4, q4[[3, 0, 1, 2]]):
            self.Rc = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            uv, ok = self.project(c[None])
            e = float(np.abs(uv[0] - self.K[:2, 2]).sum()) if ok[0] else np.inf
            if best is None or e < best[0]:
                best = (e, self.Rc.copy())
        self.Rc = best[1]
        print(f"[stageA-rtx] {twin}: camera look-at point projects {best[0]:.1f} px from the image centre", flush=True)

    def project(self, P):
        q = (np.asarray(P, np.float64) + self.org - self.cpos) @ self.Rc
        z = q[:, 2]; uv = (q[:, :2] / np.maximum(z[:, None], 1e-6)) @ self.K[:2, :2].T + self.K[:2, 2]
        return uv, z > 0.01

    def pose(self, clip, f):
        from pxr import Gf, UsdGeom
        e = clip.e
        P = f["P"][e]
        for k, (t_, r_) in enumerate(self.objs):
            if k >= len(P) or not np.isfinite(P[k]).all():
                t_.Set(Gf.Vec3d(*(self.org + [0, 0, -5.0]))); continue
            p = P[k]; t_.Set(Gf.Vec3d(*(p[:3].astype(float) + self.org))); r_.Set(Gf.Quatd(float(p[6]), float(p[3]), float(p[4]), float(p[5])))
        if f["parts"] is not None:
            for g in (0, 1):
                y = float(f["yaw"][e, g]); qz = Gf.Quatd(math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))
                if f.get("rp") is not None:                                   # the hand's tilt: roll about its x, then pitch about its y
                    r_, p_ = float(f["rp"][e, g, 0]), float(f["rp"][e, g, 1])
                    qz = qz * Gf.Quatd(math.cos(r_ / 2), math.sin(r_ / 2), 0.0, 0.0) * Gf.Quatd(math.cos(p_ / 2), 0.0, math.sin(p_ / 2), 0.0)
                for i, (t_, r_) in enumerate(self.hands[g]):
                    t_.Set(Gf.Vec3d(*(f["parts"][e, g, i].astype(float) + self.org))); r_.Set(qz)
        if f["q"] is not None:
            from cleantables.robots import franka_kinematics as K
            Fk = K.fk(torch.as_tensor(f["q"][e], dtype=torch.float64)).numpy()                     # (2, frames, 4, 4) in the base frame
            for g in (0, 1):
                for body, (t_, r_) in self.arm_links[g].items():
                    T = Fk[g, K.F_IDX[body]]
                    par = self.arm_parent[g].get(body)
                    if par is not None:                                            # nested link prims: the transform relative to the parent link
                        T = np.linalg.inv(Fk[g, K.F_IDX[par]]) @ T
                    w_, x_, y_, z_ = _R_to_wxyz(T[:3, :3])
                    dbl_t = t_.GetPrecision() == UsdGeom.XformOp.PrecisionDouble; dbl_r = r_.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
                    t_.Set((Gf.Vec3d if dbl_t else Gf.Vec3f)(*T[:3, 3].tolist())); r_.Set(Gf.Quatd(w_, x_, y_, z_) if dbl_r else Gf.Quatf(w_, x_, y_, z_))

    def grab(self):
        for _ in range(6):
            self.sim.render()
        self.cam.update(0.0, force_recompute=True)
        rgb = self.cam.data.output["rgb"]; rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[0, ..., :3].cpu().numpy().astype(np.uint8)
        seg = self.cam.data.output["semantic_segmentation"]; seg = (seg.torch if hasattr(seg, "torch") else seg)[0].cpu().numpy()
        seg = seg[..., 0] if seg.ndim == 3 and seg.shape[-1] == 1 else seg
        info = self.cam.data.info.get("semantic_segmentation") if isinstance(self.cam.data.info, dict) else (self.cam.data.info[0].get("semantic_segmentation") if isinstance(self.cam.data.info, list) else None)
        lab = (info or {}).get("idToLabels", {})
        masks = {}
        for kk, v in lab.items():
            name = v.get("class", "") if isinstance(v, dict) else str(v)
            for t in [t.strip() for t in name.split(",")]:
                if not t or t == "table":
                    continue
                masks.setdefault(t, np.zeros(seg.shape[:2], bool))
                ks = str(kk).strip()
                if ks.startswith("("):
                    col = np.array([int(float(x)) for x in ks.strip("()").split(",")])
                    rgba = seg if seg.ndim == 3 else np.ascontiguousarray(seg).view(np.uint8).reshape(*seg.shape, -1)
                    masks[t] |= (rgba[..., :len(col)] == col).all(-1)
                else:
                    masks[t] |= (seg if seg.ndim == 2 else seg[..., 0]) == int(ks)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), masks

    # ---------------------------------------------------------------------------------------------------------------------------------
    def overlay(self, img, masks, clip, f, fi):
        """red tints (collisions, pushes, aborts) + projected carry paths, routes, goal rings, names, event labels."""
        from cleantables.rl.render import COLS, short
        e = clip.e; st = f["step"]; P = f["P"][e][:clip.n]; ok_ = np.isfinite(P).all(-1)
        col = clip.collisions(f); hh, under = clip.hand_flags(f)
        pushed = clip.pushed_now(f)
        out = img.astype(np.float32)
        tint = lambda m, a=0.6: out.__setitem__(m, (1 - a) * out[m] + a * RED)                                    # noqa: E731
        from cleantables.robots import franka_kinematics as K
        for g in (0, 1):
            hit = set(col["aa"][g]) | set(col["ao"][g]) | set(col["at"][g])
            for b in {K.S_BODY[s] for s in hit}:
                if f"a{g}_{b}" in masks:
                    tint(masks[f"a{g}_{b}"])
            if (hh or under[g]) and f"h{g}" in masks:
                tint(masks[f"h{g}"], 0.7)
        win = int(1.5 / clip.dt)
        flash = set(pushed) | set(col["ao_obj"])
        for tk in clip.takes:
            if tk.get("abort") and tk.get("abort_step") is not None and tk["abort_step"] <= st < tk["abort_step"] + win and (fi // 2) % 2 == 0:
                flash.add(tk["c"])
        for k in flash:
            if f"o{k}" in masks:
                tint(masks[f"o{k}"], 0.65)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if col["sph"] is not None:                                            # the colliding collision spheres, bright red, on top
            p_, r_s = col["sph"]
            for g in (0, 1):
                hit = sorted(set(col["aa"][g]) | set(col["ao"][g]) | set(col["at"][g]))
                if not hit:
                    continue
                q = (p_[g, hit] + self.org - self.cpos) @ self.Rc; uv, okp = self.project(p_[g, hit])
                for (u, v_), z_, rr, o in zip(uv, q[:, 2], r_s[hit], okp):
                    if o:
                        cv2.circle(out, (int(u), int(v_)), max(3, int(rr * self.K[0, 0] / max(z_, 1e-3))), (0, 0, 255), -1, cv2.LINE_AA)
        flash_t = [t for t, c in (("ARM-ARM", col["aa"][0]), ("ARM-OBJECT", col["ao"][0] or col["ao"][1]), ("ARM-TABLE", col["at"][0] or col["at"][1]),
                                  ("HAND-HAND", hh)) if c]
        for i_, t in enumerate(flash_t):                                      # an on-screen flash while a collision lasts
            cv2.putText(out, t, (14, 40 + 34 * i_), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(out, t, (14, 40 + 34 * i_), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 235), 2, cv2.LINE_AA)
        hexbgr = lambda h: (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))                                      # noqa: E731

        def poly(pts, c, th=2, dash=0):
            uv, okp = self.project(pts)
            for i in range(len(uv) - 1):
                if okp[i] and okp[i + 1] and (dash == 0 or (i // dash) % 2 == 0):
                    cv2.line(out, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[i + 1]).astype(int)), c, th, cv2.LINE_AA)
        # goal rings
        for k in range(clip.n):
            if clip.mov[k] and ok_[k]:
                gl = f["goal"][e][k]; rad = 0.5 * clip.ext[k, :2].max(); th = np.linspace(0, 2 * np.pi, 33)
                ring = np.stack([gl[0] + rad * np.cos(th), gl[1] + rad * np.sin(th), np.full(33, gl[2])], -1)
                poly(ring, (60, 200, 60) if f["met"][e][k] else (30, 140, 255), 2)
        # active carries + routes
        for g in (0, 1):
            if f["phase"][e, g] in (1, 2):
                evs = [d for d in clip.ev if d["g"] == g and d["s0"] <= st and (d.get("end") is None or d["end"] >= st)]
                if evs:
                    poly(np.asarray(evs[-1]["path"]), hexbgr(COLS[int(evs[-1]["carried"]) % len(COLS)]), 3)
            if f["phase"][e, g] == 4:
                rs = [r for r in clip.routes if r["g"] == g and r["step"] <= st]
                if rs:
                    r = rs[-1]; ret = f["retract"] is not None and f["retract"][e, g]
                    if np.abs(r["route"] - r["route0"]).max() > 0.01:
                        poly(r["route0"], (150, 150, 150), 1, dash=1)
                    poly(r["route"], (40, 40, 235) if ret else GCOL_BGR[g], 2, dash=2)
        # names + event labels
        for k in range(clip.n):
            if not ok_[k]:
                continue
            uv, okp = self.project((P[k, :3] + [0, 0, clip.ext[k, 2] + 0.02])[None])
            if okp[0]:
                t = short(clip.ids[k])[:14] + (f" pushed {pushed[k]:.1f}cm" if k in pushed else "") + (" (off table)" if P[k, 2] < -0.05 else "")
                cv2.putText(out, t, (int(uv[0, 0]) - 30, int(uv[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.putText(out, t, (int(uv[0, 0]) - 30, int(uv[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 235) if k in pushed else (250, 250, 250), 1, cv2.LINE_AA)
        for g in (0, 1):
            pl = f["palm"][e, g]
            uv, okp = self.project((pl + [0, 0, 0.12])[None])
            if not okp[0]:
                continue
            lab_ = []
            if f["ik_ok"] is not None and not f["ik_ok"][e, g]:
                lab_.append("IK FAIL")
            for tk in clip.takes:
                if tk["g"] == g and tk.get("abort") and tk.get("abort_step") is not None and tk["abort_step"] <= st < tk["abort_step"] + win:
                    lab_.append("ABORT " + tk["abort"])
            if under[g]:
                lab_.append("FINGERS BELOW TABLE")
            for i_, t in enumerate(lab_):
                cv2.putText(out, t, (int(uv[0, 0]) - 40, int(uv[0, 1]) - 16 * i_), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 235), 2, cv2.LINE_AA)
        for d in clip.dist:
            if d["step"] <= st < d["step"] + 2 * win:
                k = d["obj"]
                uv, okp = self.project((P[k, :3] + [0, 0, 0.18])[None])
                if okp[0]:
                    rec = bool(clip.end_met[k]) if clip.mov[k] else (f["static_moved_cm"][e][k] <= 3)
                    cv2.putText(out, f"{d['kind'].upper()} (recovered by end: {'yes' if rec else 'no'})", (int(uv[0, 0]) - 60, int(uv[0, 1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 40, 160), 2, cv2.LINE_AA)
        return out, col, hh, under, pushed

    def caption(self, clip, f, col, hh, under, pushed, W):
        from cleantables.rl.render import TYPES, short
        e = clip.e; st = f["step"]; t = (st - clip.s0) * clip.dt
        img = np.full((112, W, 3), 32, np.uint8)
        nm = int(f["met"][e][:clip.n][clip.mov].sum()); nmv = int(clip.mov.sum())
        cv2.putText(img, f"{clip.label}  |  t = {t:5.1f} s  |  goals met {nm}/{nmv}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        for g in (0, 1):
            ds = [d for d in clip.dec if d["g"] == g and d["step"] <= st]
            s_ = f"gripper {g}: "
            if ds:
                d = ds[-1]; s_ += TYPES.get(d["type"], d["type"])
                if d["type"] == "take":
                    s_ += f" {short(clip.ids[d['obj']])[:16]}, yaw {math.degrees(d['gyaw']):.0f}, u {d['rel_u']:.2f}"
                    tk = [x for x in clip.takes if x["g"] == g and x["step"] == d["step"]]
                    if tk:
                        r_ = tk[0]
                        res = (f"ABORTED {r_['abort']}" if r_.get("abort") and r_.get("abort_step", 1e9) <= st else
                               ("carrying" if f["phase"][e, g] == 1 else "approaching" if f["phase"][e, g] == 4 else "released" if f["phase"][e, g] == 2 else "done"))
                        s_ += f" -> {res}"
            if f["retract"] is not None and f["retract"][e, g]:
                s_ += " [retracting home]"
            cv2.putText(img, s_[:95], (8, 44 + 22 * g), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GCOL_BGR[g], 1, cv2.LINE_AA)
        ev = []
        if col["aa"][0]:
            ev.append("ARM-ARM")
        if col["ao"][0] or col["ao"][1]:
            ev.append("ARM-OBJECT")
        if col["at"][0] or col["at"][1]:
            ev.append("ARM-TABLE")
        if hh:
            ev.append("HAND-HAND")
        if any(under):
            ev.append("FINGERS<TABLE")
        if pushed:
            ev.append("PUSH " + ", ".join(f"{short(clip.ids[k])[:10]} {v:.1f}cm" for k, v in pushed.items()))
        cv2.putText(img, ("events: " + " | ".join(ev)) if ev else "events: -", (8, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 245) if ev else (150, 150, 150), 1, cv2.LINE_AA)
        if clip.ep is not None and st >= clip.ep["s1"]:
            cv2.putText(img, f"END: {clip.ep['stop']}, met {clip.ep['met']:.2f}", (W - 260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1, cv2.LINE_AA)
        return img

    def scene_clip(self, name, recs, out):
        from cleantables.rl.render import Clip
        clips = []
        for lab, R in recs.items():
            e = [i for i, x in enumerate(R["env"]) if x["name"] == name][0]
            clips.append(Clip(R, e, lab))
        self.build(name, clips[0].B)
        nf = max(len(c.F) for c in clips)
        if args.max_frames:
            nf = min(nf, args.max_frames)
        W, H = args.w, args.h
        Wt, Ht = W * len(clips), 30 + H + 112
        mp4 = out / f"{'_vs_'.join(c.label for c in clips)}__{name}.mp4"
        vw = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wt}x{Ht}", "-r", str(args.fps),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(mp4)], stdin=subprocess.PIPE)
        legend = ["blue / orange boxes: the PHYSICS hands (gripper 0 / 1)", "Franka arms: posed from the recorded IK joint angles",
                  "red: collision / push / aborted object", "solid: carry path   dashed: hand route (red = retract)", "dotted grey: zero-edit route",
                  "rings: frozen goals (green = met, orange = not)"]
        for fi in range(nf):
            row = []
            for c in clips:
                f = c.F[min(fi, len(c.F) - 1)]
                self.pose(c, f)
                img, masks = self.grab()
                im2, col, hh, under, pushed = self.overlay(img, masks, c, f, fi)
                if c is clips[0]:
                    y0 = 10
                    cv2.rectangle(im2, (W - 330, 4), (W - 4, 8 + 16 * len(legend)), (255, 255, 255), -1)
                    for i_, l_ in enumerate(legend):
                        cv2.putText(im2, l_, (W - 325, y0 + 8 + 16 * i_), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (30, 30, 30), 1, cv2.LINE_AA)
                row.append(np.vstack([im2, self.caption(c, f, col, hh, under, pushed, W)]))
            fr = np.zeros((Ht, Wt, 3), np.uint8); fr[30:] = np.hstack(row)
            mod_ = name in _modified()
            cv2.putText(fr, f"{name}{' (LAYOUT MODIFIED: objects re-placed into reach)' if mod_ else ''}   |   Isaac RTX replay of recorded PhysX episodes (fast sim), {1 / (clips[0].dt * clips[0].R['every']) * 0 + args.fps / 10:.0f}x speed   |   "
                            + "  vs  ".join(c.label for c in clips), (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            vw.stdin.write(np.ascontiguousarray(fr).tobytes())
            if fi == nf // 3:
                cv2.imwrite(str(out / f"{mp4.stem}_frame.jpg"), fr)
        vw.stdin.close(); rc = vw.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed (rc {rc}) for {mp4}")
        return mp4


if __name__ == "__main__":
    main()
