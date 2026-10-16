"""Run Hunyuan3D-2.1 on the crops listed in data/twins/hy3d/jobs.json. Runs inside the Hunyuan3D-2.1 environment
(hy3dshape / hy3dpaint on PYTHONPATH), see scripts/build_twins_hunyuan.sh.

  shape: hy3d/<key>/shape.glb (+ shape.json)                       about 7 GB GPU at octree 384
  paint: hy3d/<key>/paint/textured_mesh.{obj,jpg}                   about 21 GB GPU
         (HY3D_PAINT_DIR = local hy3dpaint checkout, HY3D_PAINT_WEIGHTS = optional local paint weights folder)
Failures are written as <stage>.fail.json and reported, never replaced silently.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from cleantables.paths import DATA
from cleantables.twin.inputs import HUNYUAN

SEED = 1234


def _jobs(only) -> dict:
    jobs = json.loads((HUNYUAN / "jobs.json").read_text())["jobs"]
    return {k: p for k, p in jobs.items() if not only or k in only}


def shape(only=()) -> None:
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover
    from PIL import Image

    todo = [(k, p) for k, p in _jobs(only).items() if not (HUNYUAN / k / "shape.glb").exists()]
    print("shape todo", [k for k, _ in todo], flush=True)
    if not todo:
        return
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1", device="cuda", dtype=torch.float16)
    for k, p in todo:
        d = HUNYUAN / k
        d.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            img = Image.open(DATA / p).convert("RGBA")
            mesh = pipe(
                image=img,
                num_inference_steps=50,
                octree_resolution=384,
                guidance_scale=5.0,
                generator=torch.Generator("cuda").manual_seed(SEED),
                output_type="trimesh",
            )[0]
            mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=40000)
            mesh.export(d / "shape.glb")
            meta = dict(
                model="tencent/Hunyuan3D-2.1 hunyuan3d-dit-v2-1 (fp16)",
                input=p,
                steps=50,
                octree_resolution=384,
                guidance=5.0,
                seed=SEED,
                faces=len(mesh.faces),
                extents_model_units=list(map(float, mesh.extents)),
                wall_s=time.time() - t1,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            (d / "shape.json").write_text(json.dumps(meta, indent=1))
            print(f"shape {k}: {meta['wall_s']:.0f} s, {meta['peak_gib']:.1f} GiB", flush=True)
        except Exception as ex:  # one failed object must not stop the others
            print(f"shape {k} failed: {ex!r}", flush=True)
            (d / "shape.fail.json").write_text(json.dumps(dict(error=repr(ex))))
        torch.cuda.empty_cache()


def paint(only=()) -> None:
    local = Path(os.environ["HY3D_PAINT_DIR"])
    sys.path[:0] = [str(local), str(local / "custom_rasterizer")]
    os.chdir(local.parent)
    weights = os.environ.get("HY3D_PAINT_WEIGHTS")
    if weights:  # paint weights from a local folder instead of the hub
        import huggingface_hub

        download = huggingface_hub.snapshot_download
        huggingface_hub.snapshot_download = lambda repo_id=None, **kw: (
            weights if repo_id == "tencent/Hunyuan3D-2.1" else download(repo_id=repo_id, **kw)
        )
    from utils.torchvision_fix import apply_fix

    apply_fix()
    import torch
    from PIL import Image
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    todo = [
        (k, p)
        for k, p in _jobs(only).items()
        if (HUNYUAN / k / "shape.glb").exists() and not (HUNYUAN / k / "paint" / "textured_mesh.obj").exists()
    ]
    print("paint todo", [k for k, _ in todo], flush=True)
    if not todo:
        return
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = str(local / "ckpt/RealESRGAN_x4plus.pth")
    conf.texture_size = 2048
    conf.render_size = 2048
    pipe = Hunyuan3DPaintPipeline(conf)
    for k, p in todo:
        out = HUNYUAN / k / "paint"
        out.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            Image.open(DATA / p).convert("RGBA").save(out / "prompt.png")
            shutil.copy(HUNYUAN / k / "shape.glb", out / "shape.glb")
            pipe(
                mesh_path=str(out / "shape.glb"),
                image_path=str(out / "prompt.png"),
                output_mesh_path=str(out / "textured_mesh.obj"),
                use_remesh=True,
                save_glb=False,
            )
            meta = dict(
                key=k,
                prompt=p,
                wall_s=time.time() - t1,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                model="Hunyuan3D-2.1 paintpbr (6 views, 512, texture 2048)",
            )
            (out / "paint.json").write_text(json.dumps(meta, indent=1))
            print(f"paint {k}: {meta['wall_s']:.0f} s, {meta['peak_gib']:.1f} GiB", flush=True)
        except Exception as ex:  # one failed object must not stop the others
            print(f"paint {k} failed: {ex!r}", flush=True)
            (out / "paint.fail.json").write_text(json.dumps(dict(error=repr(ex))))
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description="Hunyuan3D-2.1 shape / paint for the library jobs.")
    ap.add_argument("stage", choices=["shape", "paint"])
    ap.add_argument("keys", nargs="*", help="only these library keys")
    args = ap.parse_args()
    {"shape": shape, "paint": paint}[args.stage](tuple(args.keys))


if __name__ == "__main__":
    main()
