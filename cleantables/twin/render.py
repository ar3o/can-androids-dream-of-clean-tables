"""RTX stills of one twin from its cameras, with a PhysX settle check (arms at the Lift default pose, 1 s).

  python -m cleantables.twin.render --id IMG_2533_t01        (Isaac Sim, about a minute, mostly start-up)
Out data/twins/<episode>/: twin_render.jpg (phone camera), side_by_side.jpg (real start frame | twin), cameras.jpg
(main | overview | wrist a | wrist b), settle.json (no NaN, every object moved < 1 cm).
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import add_launcher_args, launch_simulation

from cleantables.paths import twin_dir
from cleantables.twin.config import load_config

SETTLE_TOLERANCE_M = 0.01
LIFT_Z = 0.002  # objects start 2 mm above their resting height


def parse_args():
    ap = argparse.ArgumentParser(description="RTX stills and settle check of one episode twin.")
    ap.add_argument("--id", required=True, help="episode id, e.g. IMG_2533_t01")
    ap.add_argument("--config", default=None, help="config file (default configs/twin.yaml)")
    add_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    return args


def caption(img, text, org, scale=0.75):
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    cfg_r = load_config(args.config).render
    folder = twin_dir(args.id)
    from cleantables.twin import cameras
    from cleantables.twin.sim_scene import ARMS, TwinSceneEnv, load_scene, make_env_cfg

    S = load_scene(folder)
    cfg = make_env_cfg(S, folder, cfg_r.dome_intensity, cfg_r.key_intensity)
    cameras.add_cameras(cfg, S, folder)
    with launch_simulation(cfg, args):
        import cv2
        import numpy as np
        import torch

        env = TwinSceneEnv(cfg, S)
        dev = env.device
        cameras.hide_ground()
        env.reset()
        env.hold_default_pose()
        for k, o in enumerate(S["objects"]):
            env.place_object(k, [o["pos"][0], o["pos"][1], o["pos"][2] + LIFT_Z])
        cameras.set_static_cams(env, S, dev)
        zero = torch.zeros(1, cfg.action_space, device=dev)
        for _ in range(cfg_r.settle_steps):
            for a in ARMS:
                cameras.set_wrist(env, a, env.robots[a], env.palm_body, dev)
            env.step(zero)

        org = env.scene.env_origins[0].cpu().numpy()
        P1 = torch.stack([o.data.root_pos_w.torch[0] for o in env.objs]).cpu().numpy() - org
        P0 = np.array([o["pos"] for o in S["objects"]])
        P0[:, 2] += LIFT_Z
        disp = np.linalg.norm(P1 - P0, axis=1)
        check = dict(
            id=args.id,
            engine="PhysX (Isaac Lab, 240 Hz)",
            seconds=cfg_r.settle_steps / 30.0,
            nan=bool(np.isnan(P1).any()),
            max_disp_m=float(np.nanmax(disp)) if not np.isnan(P1).all() else None,
            per_object={o["id"]: round(float(d), 4) for o, d in zip(S["objects"], disp, strict=False)},
        )
        check["ok"] = (
            (not check["nan"]) and check["max_disp_m"] is not None and check["max_disp_m"] < SETTLE_TOLERANCE_M
        )
        (folder / "settle.json").write_text(json.dumps(check, indent=1))
        print(f"{args.id}: settle ok={check['ok']}, max displacement {check['max_disp_m']:.4f} m", flush=True)

        real = cv2.imread(str(folder / "start_real.jpg"))
        main_ = cameras.grab(env, "main")
        overview = cameras.grab(env, "overview")
        wrists = [cameras.grab(env, f"wrist_{a}") for a in ARMS]
        cv2.imwrite(str(folder / "twin_render.jpg"), main_)
        sbs = np.hstack([real, main_])
        caption(sbs, f"{args.id}: real phone frame (episode start)", (10, 30))
        caption(sbs, "twin, same camera (static plate bg + sim objects + robots)", (970, 30))
        cv2.imwrite(str(folder / "side_by_side.jpg"), sbs, [cv2.IMWRITE_JPEG_QUALITY, 92])
        wr = np.hstack([cv2.resize(w, (270, 270)) for w in wrists])
        wr = np.pad(wr, ((0, 0), (0, 960 - wr.shape[1]), (0, 0)))
        sheet = np.vstack(
            [np.hstack([main_, cv2.resize(overview, (960, 540))]), np.hstack([wr, np.zeros((270, 960, 3), np.uint8)])]
        )
        for text, xy in (
            ("main = phone camera", (10, 30)),
            ("overview", (970, 30)),
            ("wrist a (default arm pose)", (10, 570)),
            ("wrist b", (280, 570)),
        ):
            cv2.putText(sheet, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(folder / "cameras.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{args.id}: stills written to {folder}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
