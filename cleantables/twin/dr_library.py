"""The asset library that the automatic world randomiser (cleantables.twin.dr_render, mode `seeds`) samples from. CPU only, no Isaac app.
  skies      every HDRI in Isaac's Skies library (listed from the asset server)
  surfaces   every MDL in Isaac's Base material library, except the folders that are not opaque surfaces (Emissives, Glass, Templates, Styles)
  objects    every scanned object twin (data/twin_table/ep/_scalefix/*/usd, one per library key, with the category perception gave it and its
             box) + the Isaac models of the catalogue below (category from the asset's folder / name, box from its mesh)
  $SIM_PY -m cleantables.twin.dr_library [--out cleantables/twin/rtx_dr_library.json]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
VER = "Assets/Isaac/6.2"
SKIP_MDL = ("Emissives/", "Glass/", "Templates/", "Styles/")
# the catalogue: Isaac asset folders -> the category their models get (a library index; one rule per folder, nothing per scene)
CATALOGUE = [
    ("NVIDIA/Assets/ArchVis/Residential/Food/Fruit/", r"\.usd$", "fruit"),
    ("NVIDIA/Assets/ArchVis/Residential/Food/Berries/", r"\.usd$", "fruit"),
    ("NVIDIA/Assets/ArchVis/Residential/Food/Vegetables/", r"\.usd$", "fruit"),
    ("NVIDIA/Assets/ArchVis/Residential/Decor/Kitchenware/", r"(?i)bowl[^/]*\.usd$", "container"),
    ("NVIDIA/Assets/ArchVis/Residential/Kitchen/Kitchenware/", r"(?i)bowl[^/]*\.usd$", "container"),
    ("Isaac/Props/YCB/Axis_Aligned/", r"/011_banana\.usd$", "fruit"),
    ("Isaac/Props/YCB/Axis_Aligned/", r"/024_bowl\.usd$", "container"),
    ("Isaac/Props/YCB/Axis_Aligned/", r"/(037_scissors|051_large_clamp|052_extra_large_clamp)\.usd$", "tool"),
]


def s3_list(prefix, pat):
    keys, tok = [], None
    while True:
        q = f"{S3}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
        r = urllib.request.urlopen(q, timeout=30).read().decode()
        keys += [k for k in re.findall(r"<Key>([^<]*)</Key>", r) if re.search(pat, k) and "/.thumbs/" not in k]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", r)
        if not m:
            return keys
        tok = m.group(1)


def mesh_box(path):
    """Axis-aligned box (size, in the asset's own units) of every mesh under the default prim, as referenced."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(path))
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    r = bb.ComputeWorldBound(st.GetDefaultPrim() or st.GetPseudoRoot()).ComputeAlignedRange()
    return (np.array(r.GetMax()) - np.array(r.GetMin())).tolist()


# whole 3-D rooms (every top-level environment USD; not the grid / terrain / sky / track folders) and real table models (dining / commercial)
ROOM_SKIP = ("/Grid/", "/Terrains/", "/Skies/", "/Jetracer/", "/Modular_Warehouse/")
TABLES = [("NVIDIA/Assets/ArchVis/Residential/Furniture/DiningSets/", r"_Table\.usd$"),
          ("NVIDIA/Assets/ArchVis/Residential/Furniture/Outdoors/", r"DiningTable\.usd$"),
          ("NVIDIA/Assets/ArchVis/Commercial/Tables/", r"\.usd$")]


def rooms_tables():
    rooms = [k for k in s3_list(f"{VER}/Isaac/Environments/", r"^Assets/Isaac/[^/]+/Isaac/Environments/[^/]+/[^/]+\.usda?$")
             if not any(x in k for x in ROOM_SKIP)]
    tables = [k for folder, pat in TABLES for k in s3_list(f"{VER}/{folder}", pat)]
    to = lambda k: "{ISAAC_ROOT}/" + k.split(f"{VER}/", 1)[1]               # noqa: E731
    return [to(k) for k in sorted(rooms)], [to(k) for k in sorted(tables)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(Path(__file__).with_name("rtx_dr_library.json")))
    ap.add_argument("--rooms_only", action="store_true", help="add / refresh only the rooms + tables lists of an existing library")
    a = ap.parse_args()
    if a.rooms_only:
        lib = json.load(open(a.out)); lib["rooms"], lib["tables"] = rooms_tables()
        json.dump(lib, open(a.out, "w"), indent=1)
        print(f"library: {len(lib['rooms'])} rooms, {len(lib['tables'])} tables -> {a.out}"); return
    skies = sorted(set(k for k in s3_list(f"{VER}/NVIDIA/Assets/Skies/", r"\.(hdr|exr)$") if "Sky_Elements" not in k))
    # one copy per sky name (the 4k file when both exist)
    by = {}
    for k in skies:
        n = re.sub(r"_4k$", "", Path(k).stem)
        if n not in by or "_4k" in Path(k).stem:
            by[n] = k
    mdl = [k for k in s3_list(f"{VER}/NVIDIA/Materials/Base/", r"\.mdl$") if not any(s in k for s in SKIP_MDL)]
    objs, seen = [], set()
    from pxr import Usd
    for f in sorted(glob.glob(str(ROOT / "data/twin_table/ep/_scalefix/*/usd/*.usda"))):
        st = Usd.Stage.Open(f); p = st.GetDefaultPrim()
        cat = p.GetAttribute("twin:category").Get() if p and p.GetAttribute("twin:category") else None
        key = p.GetAttribute("twin:library_key").Get() if p and p.GetAttribute("twin:library_key") else Path(f).stem
        if not cat or key in seen:
            continue
        seen.add(key); objs.append(dict(usd=os.path.relpath(f, ROOT), category=cat, key=key, box=mesh_box(f), source="twin scan"))
    tmp = Path(tempfile.mkdtemp())
    for folder, pat, top in CATALOGUE:
        for k in s3_list(f"{VER}/{folder}", pat):
            lp = tmp / Path(k).name; urllib.request.urlretrieve(f"{S3}/{k}", lp)
            try:
                box = mesh_box(lp)
            except Exception as e:                                              # an asset whose meshes sit in another layer: skipped
                print("skip", k, e); continue
            if not np.all(np.isfinite(box)) or min(box) <= 0:
                print("skip (no box)", k); continue
            objs.append(dict(usd="{ISAAC_ROOT}/" + k.split(f"{VER}/", 1)[1], category=f"{top}/" + Path(k).stem.lower().rstrip("0123456789_"),
                             key=Path(k).stem, box=box, source="Isaac YCB" if "/YCB/" in k else "Isaac ArchVis"))
    lib = dict(skies=["{ISAAC_ROOT}/" + by[n].split(f"{VER}/", 1)[1] for n in sorted(by)],
               surfaces=["{ISAAC_ROOT}/" + k.split(f"{VER}/", 1)[1] for k in sorted(mdl)], objects=objs)
    lib["rooms"], lib["tables"] = rooms_tables()
    json.dump(lib, open(a.out, "w"), indent=1)
    print(f"library: {len(lib['skies'])} skies, {len(lib['surfaces'])} surface MDLs, {len(objs)} objects "
          f"({sum(o['source'] == 'twin scan' for o in objs)} twin scans) -> {a.out}")


if __name__ == "__main__":
    main()
