"""GPU point-splat renderer of the twin tables from the phone camera, robot hidden, so the world model in the robot loop sees the scene
at the env's actual object poses. It imitates the paired v0 render (twin env defaults: dome 600 warm + key distant light 700 from
above, the table texture, the emissive background plate), the robot-view video the world model is trained on. No Isaac, no robot: dense textured surface points per object,
projected with the scene's own phone camera (K, T_cam_to_table), a z-buffer by scatter-min, Lambert shading. The table + background
plate are static per template: rendered once (cached). Only the envs being replanned are rendered, on demand, from their poses.

  R = SplatRender(dev="cuda")
  R.add(name, S, twin_dir=None)                    # template: scene.json dict; meshes from the objects' usd (+ 'scale'), plate + table
  img = R.render(names, poses)                      # poses (n, I, 4) = x, y, z_bottom, yaw_off (the twins' convention) -> (n, H, W, 3) uint8
Output 480 x 270 (the phone 960 x 540 / 2; the encoders resize to their own 448 x 256 / 252)."""
from __future__ import annotations

from cleantables import CT_DATA
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
W, H = 480, 270


def _arr(txt, key):
    m = re.search(re.escape(key) + r"\s*=\s*\[", txt)
    if not m:
        return None
    i = m.end(); depth = 1; j = i
    while depth:
        c = txt[j]; depth += (c == "[") - (c == "]"); j += 1
    return txt[i:j - 1]


def _nums(s):
    return np.array(re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s), np.float64)


def _meshes(usda: Path):
    """-> [(V (n, 3), F (m, 3) int, st (n, 2) or None, tex path or None, emissive bool)] for every Mesh prim (fan-triangulated)."""
    txt = open(usda).read()
    texs = [(m.start(), m.group(1)) for m in re.finditer(r"asset inputs:file = @([^@]+)@", txt)]
    out = []
    for m in re.finditer(r'def Mesh "([^"]+)"', txt):
        nxt = txt.find("\n    def ", m.end()); body = txt[m.end(): nxt if nxt > 0 else len(txt)]
        P = _arr(body, "point3f[] points"); FI = _arr(body, "int[] faceVertexIndices"); FC = _arr(body, "int[] faceVertexCounts")
        if P is None or FI is None:
            continue
        V = _nums(P).reshape(-1, 3); fi = _nums(FI).astype(np.int64); fc = _nums(FC).astype(np.int64) if FC else np.full(len(fi) // 3, 3)
        F, q = [], 0
        for c in fc:
            for t in range(1, c - 1):
                F.append((fi[q], fi[q + t], fi[q + t + 1]))
            q += c
        st = _arr(body, "texCoord2f[] primvars:st") or _arr(body, "float2[] primvars:st")
        st = _nums(st).reshape(-1, 2) if st else None
        if st is not None and len(st) != len(V):
            st = None
        bind = re.search(r"material:binding = <([^>]+)>", body)
        mat = bind.group(1).split("/")[-1] if bind else None
        tex = None
        if mat:
            mm = re.search(r'def Material "' + re.escape(mat) + r'"', txt)
            if mm:
                tex = next((p for s, p in texs if s > mm.start()), None)
                end = txt.find('def Material', mm.end())
                if tex is not None and end > 0 and next(s for s, p in texs if p == tex) > end:
                    tex = None
        if tex is not None and not os.path.isabs(tex):
            tex = str((usda.parent / tex).resolve())
        emis = bool(mat) and re.search(r'def Material "' + re.escape(mat) + r'".*?emissiveColor', txt, re.S) is not None and "diffuseColor = (0, 0, 0)" in txt
        out.append((V, np.array(F, np.int64), st, tex, emis))
    return out


@lru_cache(maxsize=256)
def _img(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _sample(V, F, st, tex, n, rng):
    """Area-weighted surface points: positions, colours (0-1, from the texture, sRGB), face normals."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a); ar = 0.5 * np.linalg.norm(cr, axis=1)
    if ar.sum() <= 0:
        return None
    f = rng.choice(len(F), n, p=ar / ar.sum())
    r1, r2 = rng.random(n), rng.random(n); s1 = np.sqrt(r1)
    w = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    P = (w[:, :, None] * np.stack([a[f], b[f], c[f]], 1)).sum(1)
    N = cr[f] / np.maximum(np.linalg.norm(cr[f], axis=1, keepdims=True), 1e-12)
    im = _img(tex) if tex else None
    if st is not None and im is not None:
        uv = (w[:, :, None] * np.stack([st[F[f, 0]], st[F[f, 1]], st[F[f, 2]]], 1)).sum(1)
        x = np.clip(np.round(uv[:, 0] * (im.shape[1] - 1)), 0, im.shape[1] - 1).astype(int)
        y = np.clip(np.round((1 - uv[:, 1]) * (im.shape[0] - 1)), 0, im.shape[0] - 1).astype(int)
        C = im[y, x].astype(np.float32) / 255
    else:
        C = np.full((n, 3), 0.55, np.float32)
    return P.astype(np.float32), C, N.astype(np.float32)


class SplatRender:
    # shading (fitted once on paired v0 renders, the same numbers for every scene): colour = albedo * (amb + key * max(0, n.z)) * tint
    AMB, KEY, TINT = 0.62, 0.48, (1.0, 0.95, 0.88)
    CAL_LIT = CAL_PLATE = None                     # (4, 3) colour maps fitted on paired v0 renders of training episodes (fit_colour)

    def __init__(self, dev="cuda", pts_per_m2=4.0e5, n_min=4000, n_max=60000, seed=0):
        self.dev = torch.device(dev); self.T = {}; self.rng = np.random.default_rng(seed); self.OC = {}   # OC: unscaled object clouds per usd
        self.dens, self.n_min, self.n_max = pts_per_m2, n_min, n_max

    def _cam(self, S):
        cam = S.get("cameras", {}).get("main") or dict(K=S["K"], T_cam_to_table=S["T_cam_to_table"], wh=S.get("image_wh", [960, 540]))
        K = np.array(cam["K"], np.float64).reshape(3, 3).copy(); wh = cam.get("wh", [960, 540])
        K[0] *= W / wh[0]; K[1] *= H / wh[1]
        Ti = np.linalg.inv(np.array(cam["T_cam_to_table"], np.float64).reshape(4, 4))
        return torch.tensor(K, dtype=torch.float32, device=self.dev), torch.tensor(Ti, dtype=torch.float32, device=self.dev)

    def _project(self, X, K, Ti):
        C = X @ Ti[:3, :3].T + Ti[:3, 3]
        z = C[..., 2]; q = C @ K.T
        return q[..., 0] / z.clamp(min=1e-4), q[..., 1] / z.clamp(min=1e-4), z

    def _splat(self, u, v, z, col, img, depth, r=1, lab=None, val=1):
        """z-buffered splat of points (k,) into img (H*W, 3) / depth (H*W,) (in place), each point covering a (2r+1)^2 block."""
        ok = (z > 0.05) & (u >= -r) & (u < W + r) & (v >= -r) & (v < H + r)
        u, v, z, col = u[ok], v[ok], z[ok], col[ok]
        ui, vi = u.floor().long(), v.floor().long()
        offs = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
        U = torch.cat([ui + dx for dx, _ in offs]); Vv = torch.cat([vi + dy for _, dy in offs])
        Z = z.repeat(len(offs)); idx = torch.arange(len(u), device=u.device).repeat(len(offs))
        ok = (U >= 0) & (U < W) & (Vv >= 0) & (Vv < H)
        pix = Vv[ok] * W + U[ok]; Z = Z[ok]; idx = idx[ok]
        zb = depth.clone().scatter_reduce_(0, pix, Z, "amin", include_self=True)
        win = Z <= zb[pix] + 1e-6
        depth.copy_(zb); img[pix[win]] = col[idx[win]]
        if lab is not None:
            lab[pix[win]] = val

    def add(self, name, S, twin_dir=None):
        if name in self.T:
            return
        K, Ti = self._cam(S)
        ep = str(S.get("episode", "")).replace("ep/", "")
        tdir = Path(twin_dir) if twin_dir else ROOT / "data/twin_table/ep" / ep
        if not (tdir / "table.usda").exists():
            tdir = ROOT / "data/twin_table/ep/_scalefix" / ep
        img = torch.zeros(H * W, 3, device=self.dev); depth = torch.full((H * W,), 1e9, device=self.dev)
        lab = torch.zeros(H * W, dtype=torch.uint8, device=self.dev)
        for fn, n in (("background.usda", 2_500_000), ("table.usda", 1_500_000)):
            if not (tdir / fn).exists():
                continue
            plate = re.search(r"asset inputs:file = @([^@]+)@", open(tdir / fn).read()) if fn == "background.usda" else None
            for V, F, st, tex, emis in _meshes(tdir / fn):
                if plate is not None:                                   # the plate: an emissive photo (its MDL material carries the texture)
                    tex, emis = tex or plate.group(1), True
                s = _sample(V, F, st, tex, n, self.rng)
                if s is None:
                    continue
                P, C, N = (torch.as_tensor(x, device=self.dev) for x in s)
                lit = not (emis and fn == "background.usda")
                C = self._shade(C, N) if lit else self._plate(C)
                for c0 in range(0, len(P), 400_000):                     # chunks: bounded transient memory
                    u, v, z = self._project(P[c0:c0 + 400_000], K, Ti)
                    self._splat(u, v, z, C[c0:c0 + 400_000], img, depth, r=1, lab=lab, val=int(lit))
                del P, C, N
        objs = []
        for o in S["objects"]:
            usd = str(ROOT / o["usd"]); sc = float(o.get("scale", 1.0))
            if usd not in self.OC:                                       # sampled once per mesh at its largest use (1.25 x), shared
                ms = _meshes(Path(usd)) if Path(usd).exists() else []
                s = None
                if ms:
                    V, F, st, tex, _ = ms[0]
                    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() * 0.5 * 1.25 ** 2
                    s = _sample(V, F, st, tex, int(np.clip(a * self.dens, self.n_min, self.n_max)), self.rng)
                self.OC[usd] = None if s is None else (torch.as_tensor(s[0], device=self.dev).half(), torch.as_tensor(s[1] * 255, device=self.dev).round().to(torch.uint8),
                                                       torch.as_tensor(s[2], device=self.dev).half())
            objs.append(None if self.OC[usd] is None else (usd, sc))
        self.T[name] = dict(K=K, Ti=Ti, bg=img.view(H, W, 3).half(), bgd=depth.half(), bgl=lab, objs=objs)
        del img, depth
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()

    def _shade(self, C, N):
        t = torch.tensor(self.TINT, device=C.device)
        c = (C * (self.AMB + self.KEY * N[:, 2:3].abs().clamp(0, 1)) * t).clamp(0, 1)
        return self._cal(c, self.CAL_LIT)

    def _plate(self, C):
        return self._cal(C, self.CAL_PLATE)

    def _cal(self, c, M):
        if M is None:
            return c
        M = torch.as_tensor(np.asarray(M, np.float32), device=c.device)        # (4, 3): [r g b 1] @ M
        return (torch.cat([c, torch.ones_like(c[:, :1])], 1) @ M).clamp(0, 1)

    def _packed(self, nm):
        """Per template: every object's points stacked (pts, col, nrm, obj index), cached."""
        T = self.T[nm]; PK = self.__dict__.setdefault("PK", {})
        if nm not in PK:
            P, C, N, J = [], [], [], []
            for j, ob in enumerate(T["objs"]):
                if ob is not None:
                    p, c, n_ = self.OC[ob[0]]
                    P.append(p.float() * ob[1]); C.append(c.float() / 255); N.append(n_.float())
                    J.append(torch.full((len(p),), j, dtype=torch.long, device=self.dev))
            PK[nm] = tuple(torch.cat(x) for x in (P, C, N, J)) if P else None
            while len(PK) > 48:                                          # a small LRU: packed clouds of the most recent templates
                PK.pop(next(iter(PK)))
        else:
            PK[nm] = PK.pop(nm)
        return PK[nm]

    @torch.no_grad()
    def render_batch(self, names, poses, max_pts=2_000_000, ids=False):
        """The same as render, many frames per z-buffer pass (frames offset in one flat buffer; groups of <= max_pts points):
        names (n,), poses (n, I, 4) -> (n, H, W, 3) uint8."""
        P_ = poses.to(self.dev, torch.float32) if torch.is_tensor(poses) else torch.as_tensor(np.asarray(poses, np.float32), device=self.dev); n = len(names)
        sizes = [0 if self._packed(nm) is None else len(self._packed(nm)[0]) for nm in names]
        out, g0, acc = [], 0, 0
        for i in range(n + 1):
            if i == n or (acc + sizes[i] > max_pts and i > g0):
                out.append(self._render_group(names[g0:i], P_[g0:i], ids)); g0, acc = i, 0
            if i < n:
                acc += sizes[i]
        if ids:                                                                # (images, object id + 1 per pixel, 0 = table / plate)
            return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])
        return torch.cat(out) if out else torch.empty(0, H, W, 3, dtype=torch.uint8, device=self.dev)

    def _render_group(self, names, P_, ids=False):
        n = len(names)
        img = torch.stack([self.T[nm]["bg"].reshape(-1, 3) for nm in names]).reshape(-1, 3).float()
        depth = torch.stack([self.T[nm]["bgd"] for nm in names]).reshape(-1).float()
        U, V, Z, Cc, Jj = [], [], [], [], []
        for i, nm in enumerate(names):
            pk = self._packed(nm)
            if pk is None:
                continue
            P, C, N, J = pk; T = self.T[nm]
            J = J.clamp(max=P_.shape[1] - 1); pose = P_[i, J]                      # (k, 4)
            ok = torch.isfinite(pose).all(-1)
            c, s_ = torch.cos(pose[:, 3]), torch.sin(pose[:, 3])
            X = torch.stack([c * P[:, 0] - s_ * P[:, 1] + pose[:, 0], s_ * P[:, 0] + c * P[:, 1] + pose[:, 1], P[:, 2] + pose[:, 2]], -1)
            Nr = torch.stack([c * N[:, 0] - s_ * N[:, 1], s_ * N[:, 0] + c * N[:, 1], N[:, 2]], -1)
            u, v, z = self._project(X, T["K"], T["Ti"])
            z = torch.where(ok, z, torch.full_like(z, -1.0))
            U.append(u); V.append(v + i * H); Z.append(z); Cc.append(self._shade(C, Nr)); Jj.append(J)
        if U:
            u, v, z, col = torch.cat(U), torch.cat(V), torch.cat(Z), torch.cat(Cc)
            fr = (v // H).clamp(0, n - 1).long(); vl = v - fr * H                  # keep every splat inside its own frame
            vl = torch.where((vl >= -1) & (vl < H + 1), vl, torch.full_like(vl, -10.0))
            lab = torch.zeros(n * H * W, dtype=torch.int16, device=self.dev) if ids else None
            self._splat_multi(u, vl, fr, z, col, img, depth, lab=lab, jj=torch.cat(Jj) if ids else None)
        else:
            lab = torch.zeros(n * H * W, dtype=torch.int16, device=self.dev) if ids else None
        im = (img.view(n, H, W, 3) * 255).round().to(torch.uint8)
        return (im, lab.view(n, H, W)) if ids else im

    def _splat_multi(self, u, v, fr, z, col, img, depth, r=1, lab=None, jj=None):
        ok = (z > 0.05) & (u >= -r) & (u < W + r) & (v >= -r) & (v < H + r)
        u, v, fr, z, col = u[ok], v[ok], fr[ok], z[ok], col[ok]
        if jj is not None:
            jj = jj[ok]
        ui, vi = u.floor().long(), v.floor().long()
        offs = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
        Uu = torch.cat([ui + dx for dx, _ in offs]); Vv = torch.cat([vi + dy for _, dy in offs])
        Z = z.repeat(len(offs)); idx = torch.arange(len(u), device=u.device).repeat(len(offs)); F_ = fr.repeat(len(offs))
        ok = (Uu >= 0) & (Uu < W) & (Vv >= 0) & (Vv < H)
        pix = F_[ok] * (H * W) + Vv[ok] * W + Uu[ok]; Z = Z[ok]; idx = idx[ok]
        zb = depth.clone().scatter_reduce_(0, pix, Z, "amin", include_self=True)
        win = Z <= zb[pix] + 1e-6
        depth.copy_(zb); img[pix[win]] = col[idx[win]]
        if lab is not None:
            lab[pix] = 0                                                       # pixels an object point reached but the plate / table won
            lab[pix[win]] = (jj[idx[win]] + 1).to(torch.int16)

    def save(self, path):
        cpu = lambda d: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        torch.save(dict(T={k: cpu(v) for k, v in self.T.items()}, OC={k: None if v is None else tuple(x.cpu() for x in v) for k, v in self.OC.items()}), path)

    def load(self, path):
        z = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in z["OC"].items():
            self.OC.setdefault(k, None if v is None else tuple(x.to(self.dev) for x in v))
        for k, v in z["T"].items():
            self.T.setdefault(k, {kk: (vv.to(self.dev) if torch.is_tensor(vv) else vv) for kk, vv in v.items()})
        return self

    @torch.no_grad()
    def render(self, names, poses, labels=False):
        """names: list of n template names (added); poses (n, I, 4) x, y, z_bottom, yaw_off (NaN = not drawn) -> (n, H, W, 3) uint8."""
        P_ = poses.to(self.dev, torch.float32) if torch.is_tensor(poses) else torch.as_tensor(np.asarray(poses, np.float32), device=self.dev)
        out = torch.empty(len(names), H, W, 3, dtype=torch.uint8, device=self.dev); labs = []
        for i, nm in enumerate(names):
            T = self.T[nm]; img = T["bg"].reshape(-1, 3).float(); depth = T["bgd"].float(); lab = T["bgl"].clone()
            Xs, Cs = [], []
            for j, ob in enumerate(T["objs"]):
                if ob is None or j >= P_.shape[1] or not torch.isfinite(P_[i, j]).all():
                    continue
                p_, c_, n_ = self.OC[ob[0]]; P, C, N = p_.float() * ob[1], c_.float() / 255, n_.float(); x, y, zb, yw = P_[i, j]
                c, s = torch.cos(yw), torch.sin(yw)
                Rz = torch.stack([torch.stack([c, -s, 0 * c]), torch.stack([s, c, 0 * c]), torch.stack([0 * c, 0 * c, 1 + 0 * c])])
                Xs.append(P @ Rz.T + torch.stack([x, y, zb])); Cs.append(self._shade(C, N @ Rz.T))
            if Xs:
                X = torch.cat(Xs); C = torch.cat(Cs)
                u, v, z = self._project(X, T["K"], T["Ti"])
                self._splat(u, v, z, C, img, depth, r=1, lab=lab, val=1)
            out[i] = (img.view(H, W, 3) * 255).round().to(torch.uint8)
            if labels:
                labs.append(lab.view(H, W))
        return (out, torch.stack(labs)) if labels else out


CAL_JSON = Path(CT_DATA + "/wm_eval/wm_v2_world/fastpath/splat_colour.json")
if CAL_JSON.exists():
    import json as _json
    _c = _json.load(open(CAL_JSON)); SplatRender.CAL_LIT, SplatRender.CAL_PLATE = _c["lit"], _c["plate"]


def fit_colour(n_ep=12, frames=4, seed=0):
    """One global colour map per surface class (lit table / objects; emissive plate), least squares from this renderer to the
    paired v0 renders on training episodes only (the held-out test episodes are never used) -> CAL_JSON. Same map for every scene."""
    import json
    import cv2
    from cleantables.features.paired import TEST
    SplatRender.CAL_LIT = SplatRender.CAL_PLATE = None
    PR = Path(CT_DATA + "/vjepa_wm/paired_renders"); PP = Path(CT_DATA + "/vjepa_wm/paired_poses")
    eps = [d.name for d in sorted(PR.iterdir()) if (d / "v0" / "meta.json").exists() and not d.name.startswith(TEST) and (PP / f"{d.name}.npz").exists()]
    rng = np.random.default_rng(seed); eps = list(rng.permutation(eps)[:n_ep])
    R = SplatRender(); A = {0: [], 1: []}; Bv = {0: [], 1: []}
    for ep in eps:
        S = __import__("json").load(open(ROOT / "data/twin_table/ep/_scalefix" / ep / "scene.json"))
        R.add(ep, S, ROOT / "data/twin_table/ep/_scalefix" / ep)
        P, ks = poses_from_npz(S, PP / f"{ep}.npz")
        for r in rng.choice(len(ks), min(frames, len(ks)), replace=False):
            f = PR / ep / "v0" / f"{int(ks[r]):05d}.jpg"
            if not f.exists():
                continue
            img, lab = R.render([ep], P[r:r + 1], labels=True)
            ref = cv2.resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).reshape(-1, 3) / 255.0
            mine = img[0].cpu().numpy().reshape(-1, 3) / 255.0; lb = lab[0].cpu().numpy().reshape(-1)
            sub = rng.choice(len(lb), 20000, replace=False)
            for c in (0, 1):
                m = sub[lb[sub] == c]
                A[c].append(mine[m]); Bv[c].append(ref[m])
    out = {}
    for c, nm in ((1, "lit"), (0, "plate")):
        X = np.concatenate(A[c]); Y = np.concatenate(Bv[c]); X1 = np.c_[X, np.ones(len(X))]
        M = np.linalg.lstsq(X1, Y, rcond=None)[0]
        out[nm] = M.tolist(); out[nm + "_rmse_before"] = float(np.sqrt(((X - Y) ** 2).mean())); out[nm + "_rmse_after"] = float(np.sqrt(((X1 @ M - Y) ** 2).mean()))
    out["episodes"] = eps
    json.dump(out, open(CAL_JSON, "w"), indent=1)
    SplatRender.CAL_LIT, SplatRender.CAL_PLATE = out["lit"], out["plate"]
    return out


def poses_from_npz(S, npz):
    """paired_poses (k, ids, pose (n, m, 4)) -> (n, I, 4) in scene.json object order (objects not in the file keep their start pose), k."""
    Z = np.load(npz, allow_pickle=True); ids = [str(x) for x in Z["ids"]]
    P = np.zeros((len(Z["k"]), len(S["objects"]), 4), np.float32)
    for j, o in enumerate(S["objects"]):
        P[:, j] = Z["pose"][:, ids.index(o["id"])] if o["id"] in ids else [*o["pos"][:3], o.get("yaw", 0.0)]
    return P, np.asarray(Z["k"]).astype(int)


if __name__ == "__main__":
    print(__import__("json").dumps({k: v for k, v in fit_colour().items() if "rmse" in k or k == "episodes"}))
