"""Decode the Hands23 object masks (COCO RLE, stored by third_party/hands23/scripts/run_hands23.py) for every first-object instance,
downsampled to our 270x480 mask grid (nearest), so cleantables.perception.contact.contact can match the contacted object to SAM 3
objects by mask IoU (instead of the paper's box association, with the same 0.1 IoU threshold).
Run with the Hands23 venv (it has pycocotools):
  PYTHONPATH=. $H23/venv/bin/python cleantables/perception/contact/hands23_masks.py      (from the project root; standalone)
Out: OUT/contact/hands23/<window>_objmasks.npz  inst_index (n,), masks (packbits of (n,270,480) bool), shape
"""
from __future__ import annotations

from cleantables import CT_DATA
import glob
import os

import cv2
import numpy as np
from pycocotools import mask as mask_util

OUT = CT_DATA + "/perception_v2/contact/hands23"
H, W = 270, 480


def run(p):
    out = p[:-4] + "_objmasks.npz"
    if os.path.exists(out):
        print(out, "exists, skipped"); return
    z = np.load(p, allow_pickle=False)
    h, w = (int(x) for x in z["image_hw"])
    idx = np.flatnonzero(z["inst_class"] == 1)
    M = np.zeros((len(idx), H, W), bool)
    for a, i in enumerate(idx):
        m = mask_util.decode({"size": [h, w], "counts": str(z["inst_mask_rle"][i]).encode()})
        M[a] = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, inst_index=idx, masks=np.packbits(M), shape=np.array(M.shape))
    os.replace(tmp, out)
    print(out, len(idx), "object masks")


if __name__ == "__main__":
    for p in sorted(glob.glob(f"{OUT}/*_*_*.npz")):
        if not p.endswith("_objmasks.npz"):
            run(p)
