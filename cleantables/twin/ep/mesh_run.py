"""Hunyuan3D-2.1 shape and texture generation for the episode-twin object library (rt2_hunyuan3d env, weights in $HUNYUAN3D_PAINT).

  shape:  bash runs/twin_ep/hy3d.sh shape   -> data/twin_table/ep/hy3d/<key>/shape.glb (+ shape.json)   (~7 GB GPU at octree 384)
  paint:  bash runs/twin_ep/hy3d.sh paint   -> data/twin_table/ep/hy3d/<key>/paint/textured_mesh.{obj,jpg}  (~21 GB GPU)
Inputs: data/twin_table/ep/hy3d/jobs.json (key -> RGBA crop from cleantables.twin.ep.mesh_inputs). Failures are written as
<stage>.fail.json and reported, never silently replaced.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HY = ROOT / "data/twin_table/ep/hy3d"


def shape(only=()):
    import torch
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover
    jobs = json.load(open(HY / "jobs.json"))["jobs"]
    todo = [(k, p) for k, p in jobs.items() if (not only or k in only) and not (HY / k / "shape.glb").exists()]
    print("todo", [k for k, _ in todo], flush=True)
    if not todo:
        return
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1", device="cuda", dtype=torch.float16)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    for k, p in todo:
        d = HY / k; d.mkdir(parents=True, exist_ok=True)
        t1 = time.time(); torch.cuda.reset_peak_memory_stats()
        try:
            img = Image.open(ROOT / p).convert("RGBA")
            mesh = pipe(image=img, num_inference_steps=50, octree_resolution=384, guidance_scale=5.0,
                        generator=torch.Generator("cuda").manual_seed(1234), output_type="trimesh")[0]
            mesh = FloaterRemover()(mesh); mesh = DegenerateFaceRemover()(mesh); mesh = FaceReducer()(mesh, max_facenum=40000)
            mesh.export(d / "shape.glb")
            meta = dict(model="tencent/Hunyuan3D-2.1 hunyuan3d-dit-v2-1 (fp16)", input=p, steps=50, octree_resolution=384, guidance=5.0, seed=1234,
                        faces=len(mesh.faces), extents_model_units=list(map(float, mesh.extents)), wall_s=time.time() - t1,
                        peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            json.dump(meta, open(d / "shape.json", "w"), indent=1)
            print("OK shape", k, round(meta["wall_s"]), "s", round(meta["peak_gib"], 1), "GiB", flush=True)
        except Exception as ex:
            print("FAIL shape", k, repr(ex), flush=True)
            json.dump(dict(error=repr(ex)), open(d / "shape.fail.json", "w"))
        torch.cuda.empty_cache()


def paint(only=()):
    LOCAL = ROOT / "cache/hy3d_paint_local/hy3dpaint"
    PAINT_W = Path(os.environ.get("HUNYUAN3D_PAINT", "third_party/hunyuan3d_paint_2_1"))
    sys.path.insert(0, str(LOCAL)); sys.path.insert(0, str(LOCAL / "custom_rasterizer"))
    os.chdir(LOCAL.parent)
    import huggingface_hub
    _orig = huggingface_hub.snapshot_download
    huggingface_hub.snapshot_download = lambda repo_id=None, **kw: str(PAINT_W) if repo_id == "tencent/Hunyuan3D-2.1" else _orig(repo_id=repo_id, **kw)
    from utils.torchvision_fix import apply_fix
    apply_fix()
    import shutil
    import torch
    from PIL import Image
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    jobs = json.load(open(HY / "jobs.json"))["jobs"]
    todo = [(k, p) for k, p in jobs.items() if (not only or k in only) and (HY / k / "shape.glb").exists()
            and not (HY / k / "paint" / "textured_mesh.obj").exists()]
    print("todo", [k for k, _ in todo], flush=True)
    if not todo:
        return
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = str(LOCAL / "ckpt/RealESRGAN_x4plus.pth")
    conf.texture_size = 2048
    conf.render_size = int(os.environ.get("PAINT_RENDER", 2048))      # 1024 halves the rasteriser buffers (shared GPU)
    t0 = time.time()
    pipe = Hunyuan3DPaintPipeline(conf)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    for k, p in todo:
        out = HY / k / "paint"; out.mkdir(parents=True, exist_ok=True)
        t1 = time.time(); torch.cuda.reset_peak_memory_stats()
        try:
            Image.open(ROOT / p).convert("RGBA").save(out / "prompt.png")
            shutil.copy(HY / k / "shape.glb", out / "shape.glb")
            pipe(mesh_path=str(out / "shape.glb"), image_path=str(out / "prompt.png"), output_mesh_path=str(out / "textured_mesh.obj"),
                 use_remesh=True, save_glb=False)
            meta = dict(key=k, prompt=p, wall_s=time.time() - t1, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                        model="Hunyuan3D-2.1 paintpbr (6 views, 512, texture 2048)")
            json.dump(meta, open(out / "paint.json", "w"), indent=1)
            print("OK paint", k, round(meta["wall_s"]), "s", round(meta["peak_gib"], 1), "GiB", flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print("FAIL paint", k, repr(ex), flush=True)
            json.dump(dict(error=repr(ex)), open(out / "paint.fail.json", "w"))
        torch.cuda.empty_cache()




def sr(paths):
    """x4 RealESRGAN (the Hunyuan paint repo's checkpoint) of small RGBA crops (RGB super-resolved, alpha bicubic) -> <name>_sr.png."""
    import numpy as np
    from PIL import Image
    LOCAL = ROOT / "cache/hy3d_paint_local/hy3dpaint"
    sys.path.insert(0, str(LOCAL))
    from utils.torchvision_fix import apply_fix
    apply_fix()
    from utils.image_super_utils import imageSuperNet

    class C:
        realesrgan_ckpt_path = str(LOCAL / "ckpt/RealESRGAN_x4plus.pth")
    net = imageSuperNet(C())
    for p in paths:
        im = Image.open(ROOT / p).convert("RGBA")
        im = im.resize((im.width // 4, im.height // 4), Image.LANCZOS)      # the crops were bicubic-upscaled to >= 512: back to ~native px
        a = np.array(im)
        rgb = Image.fromarray(a[..., :3])
        up = np.array(net(rgb))
        al = np.array(Image.fromarray(a[..., 3]).resize((up.shape[1], up.shape[0]), Image.BICUBIC))
        out = ROOT / p.replace(".png", "_sr.png")
        Image.fromarray(np.dstack([up, al])).save(out)
        print("sr", out.name, up.shape, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sr":
        sr(sys.argv[2:])
    else:
        only = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ()
        {"shape": shape, "paint": paint}[sys.argv[1]](only)
