"""Object-centric world model, extending cleantables.worldmodel.base_model.UWM.

UWM2 replaces the 32 pooled V-JEPA tokens with cross-attention from the object, CLS and task tokens to two frozen patch
grids (V-JEPA 2 ViT-L, 32 frames, 16 x 28; DINOv2-L, 18 x 32); each object queries with the centroid of its SAM 3 mask.
It adds per-object goal-appearance distances (objg) and a DINOv2 whole-grid goal distance, per-object appearance tokens
(V-JEPA 2 and DINOv2 pooled in the object's early masks), an end-yaw head (von Mises mixture on 2 x yaw), a future-latent
head for the global V-JEPA latent at the move end, an optional stop logit, and an end head that is a mixture over
reference objects (endref).
UWMW, the model trained by cleantables.worldmodel.train, keeps the UWM2 trunk and inputs and has only future-prediction
heads: where each object ends, a flow-matching motion decoder over all objects, distance to travel, the future video
latent, future hand contact and masked-object position.
All video inputs are off in mode 1 (3D only, as on the robot) and for composite scenes. Controls: vid_ctl='occ' replaces
both grids by an occupancy raster of the current masks (location only), 'pix' by raw colour; app_ctl uses the position
code pooled in the same masks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cleantables.worldmodel.base_model import MIX, UWM, mlp, sel
from cleantables.worldmodel.data import fourier_uv

FUT_S = (0.5, 1.0, 2.0, 4.0)               # future horizons (s) of the world heads
REF_F = 8                                  # reference geometry features (ref_feat)
N50 = 0.012                                  # tracker rest noise: median 3D |displacement over 4 s| of static objects (m); offset in the distance head
S_MIN = 0.03                               # size floor of a reference (m), e.g. for the table token's zero height


def ref_frame(geo):
    """Object frames from token geometry (B, n, 8) = [x, y, top, L, W, H, sin 2yaw, cos 2yaw]: position p (B, n, 3),
    long-side yaw (B, n) (180 deg symmetric, as the input) and size s (B, n, 3) floored at S_MIN."""
    p = geo[..., :3]
    yaw = 0.5 * torch.atan2(geo[..., 6], geo[..., 7])
    sz = geo[..., 3:6].clamp(min=S_MIN)
    return p, yaw, sz


def to_frame(x, p, yaw, sz):
    """World points x (..., 3) -> reference frame (rotated by -yaw about z), in size units."""
    d = x - p
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[..., 0] + s_ * d[..., 1], -s_ * d[..., 0] + c * d[..., 1], d[..., 2]], -1) / sz


def from_frame(z, p, yaw, sz):
    """Reference-frame points (size units) -> world."""
    u = z * sz
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    return p + torch.stack([c * u[..., 0] - s_ * u[..., 1], s_ * u[..., 0] + c * u[..., 1], u[..., 2]], -1)


def sincos2d(h, w, d):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    q = d // 4
    f = torch.exp(-math.log(10000.0) * torch.arange(q, dtype=torch.float32) / q)
    ax, ay = xx.flatten()[:, None] * f, yy.flatten()[:, None] * f
    return torch.cat([ax.sin(), ax.cos(), ay.sin(), ay.cos()], -1)            # (h*w, d)


class GridRead(nn.Module):
    """Queries (B, n, d) read a frozen patch grid (B, h, w, C) with 2 pre-norm decoder layers (self-attention among the queries,
    cross-attention to the grid). Returns the residual update, zero where the grid is off."""

    def __init__(self, d, heads, hw, in_dim, layers=2, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d)) if in_dim > 4 else nn.Linear(in_dim, d)
        self.register_buffer("pos", sincos2d(hw[0], hw[1], d), persistent=False)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
                                     for _ in range(layers)])
        self.out = nn.LayerNorm(d)

    def forward(self, q, grid, on):
        mem = self.proj(grid.flatten(1, 2).float()) + self.pos
        x = q
        for l in self.layers:
            x = l(x, mem)
        return self.out(x - q) * on.float()[:, None, None]


class UWM2(UWM):
    def __init__(s, d=256, L=6, heads=8, flow_layers=3, drop=0.1, vid=("vj", "dn"), vid_ctl="", objg=True, app=True, app_ctl=False,
                 yaw=True, fut=True, rot6=False, stop=False, endref=False, world=False, cons=False):
        super().__init__(d=d, L=L, heads=heads, flow_layers=flow_layers, drop=drop, path_resid=True)
        s.rot6 = rot6                       # object rotation along the path, 6D representation relative to the start
        if rot6:
            s.f_in = mlp(12, d); s.f_out = nn.Linear(d, 12)
        del s.vj, s.vpos, s.seen, s.jepa                               # no pooled tokens; p(seen) and the old latent head are replaced
        s.vid, s.vid_ctl, s.use_objg, s.use_app, s.app_ctl, s.use_yaw, s.use_fut = tuple(vid), vid_ctl, objg, app, app_ctl, yaw, fut
        cin = {"": 1024, "occ": 1, "pix": 3}[vid_ctl]
        hw = {"vj": (16, 28), "dn": (18, 32)}
        if vid_ctl == "pix":
            hw = {"vj": (16, 28), "dn": (16, 28)}
        s.reads = nn.ModuleDict({g: GridRead(d, heads, hw[g], cin) for g in s.vid})
        s.uv_e = nn.Linear(32, d); s.no_uv = nn.Parameter(torch.zeros(d))
        if objg:
            s.objg_e = mlp(4, d); s.dgoal_e = mlp(4, d)
        if app:
            if app_ctl:
                s.app_c = nn.Linear(128, d)
            else:
                s.app_v = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d)); s.app_d = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        if yaw:
            s.yaw_h = mlp(3 * d, MIX * 4, 2 * d)
        if fut:
            s.fut_h = mlp(d, 1024, 2 * d)
        s.use_stop = stop                   # "no further object" option of the next-object head
        if stop:
            s.stop_h = mlp(d, 1, d)
        # endref: a single end-position output. A mixture over references j = every object token (the mover's own token stands for
        # its start, the table token for the table), each a Gaussian in j's frame scaled by j's size:
        #   p(end) = sum_j a_j N(end; p_j + R_j (s_j * o_j), R_j diag(s_j * sigma_j)^2 R_j^T)
        # trained only on the observed end position (mixture NLL). It replaces the destination pointer (TABLE / LEAVES), the relation
        # head and both end mixtures; the path's third context slot is the end (sampled at plan time).
        s.endref = endref
        if endref:
            del s.tgt_h, s.tgt_x, s.xemb, s.rel_h, s.mdn, s.disp
            s.ref_g = mlp(REF_F, d)                                    # the mover's start in j's frame (size units) + j's size
            s.ref_h = mlp(4 * d, 7, 2 * d)                             # [o_j, m, o_j * m, glob + geometry] -> logit a_j, offset o_j, log sigma_j
            s.end_e = mlp(3, d)                                        # the end, relative to the mover's start (PS units)
        # self-supervised world heads (every keyframe, including stretches without an instruction)
        s.use_world = world
        if world:
            s.fpos_h = mlp(2 * d, len(FUT_S) * 3, 2 * d)              # every object's displacement at +0.5 / 1 / 2 / 4 s (PS units)
            s.mpos_h = mlp(2 * d, 6, 2 * d)                            # a hidden object's position (PS units, table frame) + log sd
        # consequence head (PhysX interventions): is object j displaced or toppled when the mover is carried?
        s.use_cons = cons
        if cons:
            s.cons_h = mlp(3 * d + REF_F, 1, d)

    def encode(s, b, mode):
        B = b["obj_geo"].shape[0]
        om = b["obj_mask"].bool()
        vm = (mode != 2)[:, None, None].float()
        if "obj_hide" in b:                                                  # masked-object prediction: hidden tokens keep identity only
            vm = vm * (~b["obj_hide"].bool()).float()[..., None]
        g = s.geo(b["obj_geo"]) * vm + s.null_geo * (1 - vm)
        o = g + s.txt(b["obj_txt"]) + s.crop(b["obj_crop"]) * b["obj_crop_ok"].float()[..., None] + s.kind(b["obj_kind"])
        if s.use_app:
            ok = b["app_ok"].float()[..., None]
            o = o + (s.app_c(b["app_ctl"]) if s.app_ctl else s.app_v(b["app_vj"]) + s.app_d(b["app_dn"])) * ok
        if getattr(s, "ctc", None) is not None and "obj_contact" in b:     # HOIST-Former hand contact at k, as an input (0 = unknown or dropped)
            code = torch.where(b["obj_contact_ok"].bool(), 1 + b["obj_contact"].long(), torch.zeros_like(b["obj_contact"].long()))
            o = o + s.ctc(code)
            if getattr(s, "ctc2", None) is not None and "obj_contact2" in b:    # Hands23 contact as a second, separate channel
                code2 = torch.where(b["obj_contact2_ok"].bool(), 1 + b["obj_contact2"].long(), torch.zeros_like(b["obj_contact2"].long()))
                o = o + s.ctc2(code2)
        if getattr(s, "iv_e", None) is not None and "iv_obj" in b:           # sim only: the carried object and its commanded path
            iv = b["iv_obj"].long(); ar_ = torch.arange(B, device=o.device)
            add = s.iv_e(b["iv_path"].reshape(B, -1).to(o.dtype)) * (iv >= 0).to(o.dtype)[:, None]
            o = o.clone(); o[ar_, iv.clamp(min=0)] = o[ar_, iv.clamp(min=0)] + add
        hg = b["hand_geo"]
        h = s.hgeo(hg[..., 1:]) + s.hq(b["hand_q16"]) * b["hand_q_ok"].float()[..., None] + s.hid.weight[None]
        hmask = (hg[..., 0] > 0.5) & (mode != 2)[:, None]
        cls = s.cls[None].expand(B, -1, -1) + s.mod(mode)[:, None]
        t = s.task(b["task_emb"])[:, None]
        von = b["vid_ok"].bool() & (mode != 1)                               # any video input at all for this sample
        if s.vid and bool(von.any()):                                        # 3D-only batches (the robot) need no grids
            # the image-position code is added to the query only (so an object finds itself on the grid); the tokens keep only the
            # read's update, which is zero for samples without video, so objects without a mask (e.g. composite additions) stay unmarked
            uok = b["uv_ok"] & ~b["obj_hide"].bool() if "obj_hide" in b else b["uv_ok"]    # a hidden object's image position would leak
            uvq = torch.where(uok[..., None], s.uv_e(fourier_uv(b["uv"]).to(o.dtype)), s.no_uv.expand_as(o))
            pos = torch.cat([torch.zeros_like(cls), torch.zeros_like(t), uvq], 1)
            q = torch.cat([cls, t, o], 1)
            for gname, rd in s.reads.items():
                q = q + rd(q + pos, b[f"{gname}_grid"], von)
            cls, t, o = q[:, :4], q[:, 4:5], q[:, 5:]
        if s.use_objg:
            og = b["objg"]
            f = torch.cat([og, (og[..., :1] - og[..., 1:2]), torch.ones_like(og[..., :1])], -1)
            okg = (b["objg_ok"] & von[:, None]).float()[..., None]
            o = o + s.objg_e(f) * okg
            dg = b["dino_goal"]
            fg = torch.cat([dg, dg[:, :1] - dg[:, 1:2], torch.ones_like(dg[:, :1])], -1)
            cls = cls + (s.dgoal_e(fg) * (von & b["objg_ok"].any(-1)).float()[:, None])[:, None]
        x = torch.cat([cls + s.typ.weight[0], t + s.typ.weight[1], o + s.typ.weight[2], h + s.typ.weight[3]], 1)
        valid = torch.cat([torch.ones(B, 5, dtype=torch.bool, device=x.device), om, hmask], 1)
        y = s.norm(s.trunk(x, src_key_padding_mask=~valid))
        return dict(cls=y[:, :4], obj=y[:, 5:29], mem=y, pad=~valid, om=om, geo=b["obj_geo"], mode=mode, kindx=b["obj_kind"], von=von)

    # ---------------------------------------------------------------- flow sampling (positions scaled by PS, rotation channels not)
    @torch.no_grad()
    def sample_paths(s, E, c, K=8, steps=12):
        from cleantables.worldmodel.base_model import NP, PS
        B = c.shape[0]; nd = 12 if s.rot6 else 6
        Ek = dict(mem=E["mem"].repeat_interleave(K, 0), pad=E["pad"].repeat_interleave(K, 0))
        ck = c.repeat_interleave(K, 0)
        x = torch.randn(B * K, NP, nd, device=c.device)
        for i in range(steps):
            t = torch.full((B * K,), i / steps, device=c.device)
            x = x + s.velocity(Ek, ck, x, t) / steps
        x = x.view(B, K, NP, nd)
        s._last_rot6 = (x[..., 6:12] + x.new_tensor([1, 0, 0, 0, 1, 0])) if s.rot6 else None   # rotation path relative to the start
        return x[..., :6] * PS

    def pointers(s, E, mv_for_tgt):
        """endref models: next-object logits only, no destination pointer: (ml, None)."""
        if not s.endref:
            return super().pointers(E, mv_for_tgt)
        mov = E["om"] & (E["kind"] == 0)
        return s.mover_h(E["obj"]).squeeze(-1).masked_fill(~mov, -1e4), None

    def heads_from(s, E, mv, tg):
        if s.endref:
            raise RuntimeError("endref model: there is no destination label / pointer; use ctx_end(E, mv, end) / end_mode / plan_paths")
        return super().heads_from(E, mv, tg)

    # ---------------------------------------------------------------- endref: end-position mixture over references
    def ref_feat(s, E, mv):
        """(B, 24, REF_F): the mover's start in each reference's frame (size units, clipped) + the reference's size (m) + is-self."""
        B = E["geo"].shape[0]; ar = torch.arange(B, device=E["geo"].device)
        p, yaw, sz = ref_frame(E["geo"].float())
        st = E["geo"][ar, mv.clamp(min=0), :3].float()
        z = to_frame(st[:, None], p, yaw, sz).clamp(-5, 5)
        self_ = torch.zeros(B, p.shape[1], device=p.device); self_[ar, mv.clamp(min=0)] = 1
        return torch.cat([z, sz, (E["kindx"] == 1).float()[..., None], self_[..., None]], -1)

    def end_mix(s, E, mv):
        """Return dict(logit (B, 24) a_j logits (invalid references -1e4), off (B, 24, 3) o_j, logs (B, 24, 3), and the frames p / yaw / sz)."""
        o, om = E["obj"], E["om"]
        glob = E["cls"][:, 2]
        m = sel(o, mv)
        rg = s.ref_g(s.ref_feat(E, mv).to(o.dtype))
        z = s.ref_h(torch.cat([o, m[:, None].expand_as(o), o * m[:, None], glob[:, None] + rg], -1)).float()
        p, yaw, sz = ref_frame(E["geo"].float())
        logit = z[..., 0].masked_fill(~om, -1e4)
        return dict(logit=logit, off=z[..., 1:4], logs=z[..., 4:7].clamp(-4, 1.5), p=p, yaw=yaw, sz=sz)

    def end_logp(s, M, end):
        """log p(end) per reference (B, 24) incl. log a_j; end (B, 3) world metres. The Gaussian is in j's frame (size units): the
        Jacobian of the size scaling is included, so all references are compared in world units."""
        zz = to_frame(end[:, None].float(), M["p"], M["yaw"], M["sz"])
        sd = M["logs"].exp()
        lg = -0.5 * (((zz - M["off"]) / sd) ** 2).sum(-1) - M["logs"].sum(-1) - torch.log(M["sz"]).sum(-1) - 1.5 * math.log(2 * math.pi)
        return F.log_softmax(M["logit"], -1) + lg

    def end_nll(s, E, mv, end):
        M = s.end_mix(E, mv)
        lp = s.end_logp(M, end)
        a = F.softmax(M["logit"], -1)
        s._diag = dict(a_entropy=float(-(a * torch.log(a.clamp(min=1e-9))).sum(-1).mean().detach()), a_max=float(a.max(-1).values.mean().detach()))
        return -torch.logsumexp(lp, -1), lp

    @torch.no_grad()
    def end_sample(s, E, mv, K=8, sd_scale=0.5):
        """K ends per sample (B, K, 3) in world metres and their reference j (B, K); sampled at half the sd, as in the base model."""
        M = s.end_mix(E, mv); B = M["logit"].shape[0]
        j = torch.multinomial(F.softmax(M["logit"], -1), K, replacement=True)
        g3 = lambda t: torch.gather(t, 1, j[..., None].expand(B, K, 3))                         # noqa: E731
        zz = g3(M["off"]) + sd_scale * g3(M["logs"]).exp() * torch.randn(B, K, 3, device=j.device)
        return from_frame(zz, g3(M["p"]), torch.gather(M["yaw"], 1, j), g3(M["sz"])), j

    @torch.no_grad()
    def end_mode(s, E, mv):
        """Most likely reference (argmax a_j) and its component mean: (end (B, 3) world metres, j (B,), a (B, 24))."""
        M = s.end_mix(E, mv); a = F.softmax(M["logit"], -1); j = a.argmax(-1); ar = torch.arange(len(j), device=j.device)
        return from_frame(M["off"][ar, j], M["p"][ar, j], M["yaw"][ar, j], M["sz"][ar, j]), j, a

    def ctx_end(s, E, mv, end):
        """Path / yaw / contact context for a given end (B, 3) in world metres: [glob, mover token, end token]. The end token adds the
        reference tokens weighted by their (detached) responsibility for this end."""
        from cleantables.worldmodel.base_model import PS
        glob = E["cls"][:, 2]
        m = sel(E["obj"], mv)
        st = E["geo"][torch.arange(len(mv), device=mv.device), mv.clamp(min=0), :3].float()
        with torch.no_grad():
            r = F.softmax(s.end_logp(s.end_mix(E, mv), end), -1)
        e = s.end_e(((end.float() - st) / PS).to(m.dtype)) + (r[..., None].to(m.dtype) * E["obj"]).sum(1)
        return torch.cat([glob, m, e], -1)

    @torch.no_grad()
    def plan_paths(s, E, mv, K=8, steps=12, end=None):
        """K paths per sample, each anchored to its own end: ends are sampled from the end mixture (or the given end (B, 3) is shared by
        the K flow samples); path = flow residual + straight line start -> end. Returns (paths (B, K, NP, 6) in metres relative to the
        mover's start, ends (B, K, 3) in world metres)."""
        from cleantables.worldmodel.base_model import PS
        B = len(mv)
        ends = s.end_sample(E, mv, K)[0] if end is None else end[:, None].expand(B, K, 3)
        Ek = {k: (v.repeat_interleave(K, 0) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B else v) for k, v in E.items()}
        mk = mv.repeat_interleave(K, 0)
        ck = s.ctx_end(Ek, mk, ends.reshape(B * K, 3))
        r = s.sample_paths(dict(mem=Ek["mem"], pad=Ek["pad"]), ck, K=1, steps=steps)[:, 0] / PS           # (B*K, NP, 6)
        st = E["geo"][torch.arange(B, device=mv.device), mv.clamp(min=0), :3].float()
        dsp = (ends - st[:, None]).reshape(B * K, 3) / PS
        u = torch.linspace(0, 1, r.shape[1], device=r.device)[None, :, None]
        line = u * dsp[:, None, :]
        out = torch.cat([r[..., :3] + line, r[..., 3:6] + line], -1) * PS
        return out.view(B, K, -1, 6), ends

    # ---------------------------------------------------------------- world and consequence heads
    def fut_pos(s, E):
        """(B, 24, len(FUT_S), 3): every object's displacement at the future horizons (PS units)."""
        o = E["obj"]; c = E["cls"][:, 3][:, None].expand_as(o)
        return s.fpos_h(torch.cat([o, c], -1)).float().view(o.shape[0], o.shape[1], len(FUT_S), 3)

    def hidden_pos(s, E):
        """(B, 24, 3) position (PS units, table frame) and log sd (B, 24, 3) of each token, read from its output (used on hidden tokens)."""
        o = E["obj"]; c = E["cls"][:, 3][:, None].expand_as(o)
        z = s.mpos_h(torch.cat([o, c], -1)).float()
        return z[..., :3], z[..., 3:].clamp(-4, 2)

    def cons_logit(s, E, mv):
        """(B, 24): logit that object j is displaced / toppled when the mover is carried (PhysX interventions)."""
        o = E["obj"]; m = sel(o, mv)
        return s.cons_h(torch.cat([o, m[:, None].expand_as(o), o * m[:, None], s.ref_feat(E, mv).to(o.dtype)], -1)).squeeze(-1).float()

    # ---------------------------------------------------------------- stop (the instructed task is complete: no further object)
    def stop_logit(s, E):
        """(B,) stop logit, competing with the movable objects' mover logits in one softmax."""
        return s.stop_h(E["cls"][:, 0]).squeeze(-1)

    # ---------------------------------------------------------------- end yaw (2 x yaw, von Mises mixture)
    def yaw_mix(s, c):
        z = s.yaw_h(c).view(-1, MIX, 4)
        return dict(logit=z[..., 0], mu=torch.atan2(z[..., 2], z[..., 1]), logk=z[..., 3].clamp(-2, 5))

    def yaw_nll(s, c, yaw):
        """End yaw (B,) (rad, table frame) -> NLL (B,) of 2 x yaw under the mixture."""
        z = s.yaw_mix(c)
        x = 2 * yaw[:, None]
        k = z["logk"].exp()
        logp = k * torch.cos(x - z["mu"]) - (torch.log(torch.special.i0e(k)) + k) - math.log(2 * math.pi)
        return -torch.logsumexp(F.log_softmax(z["logit"], -1) + logp, -1)

    def yaw_mode(s, c):
        """Most likely end yaw (rad, in [-pi/2, pi/2), 180 deg symmetric)."""
        z = s.yaw_mix(c)
        i = z["logit"].argmax(-1)
        return 0.5 * z["mu"].gather(1, i[:, None]).squeeze(1)

    # ---------------------------------------------------------------- future global latent (move end)
    def fut_pred(s, E, b):
        base = F.normalize(b["glob_now"].float(), dim=-1) * E["von"].float()[:, None]
        return F.normalize(base + s.fut_h(E["cls"][:, 1]).float(), dim=-1)


class UWMW(UWM2):
    """World model that learns from watching: future-prediction outputs only, no move labels.

    Inputs are those of UWM2: object tokens, V-JEPA 2 / DINOv2 grids through cross-attention, instruction, appearance, WiLoR
    hand tokens and HOIST-Former hand contact at k (objg / dino_goal stay off unless objg=True). Heads:
      T1  each object's pose at the end of the task: per object i a mixture over references j (every object token, its own
          start, the table) with offsets in j's frame scaled by j's size; p(still present); end yaw (von Mises mixture)
      T2  every object's displacement over the next H steps as one joint flow-matching sample (one query per (object, step),
          self-attention across objects), plus its mean last-step displacement as a cheap read-out
      T3  the global V-JEPA latent at +4 s (fut_h)
      T5  each object's hand contact over the H steps
      and the hidden-object position from the UWM2 world heads.
    The mover, target, relation, stop, progress and time-to-move heads and the mover-only path decoder are not built."""

    def __init__(s, H=8, noise_floor=0.0, dist_head=False, t1_no_table=False, rem_head=False, **kw):
        kw.update(stop=False, endref=False, world=True, cons=False)
        super().__init__(**kw)
        from cleantables.worldmodel.base_model import NP
        d = s.d
        for nm in ("mover_h", "tgt_h", "tgt_x", "xemb", "rel_h", "mdn", "disp", "rel", "p_vid", "p_ttm", "p_in", "p_pos", "p_c", "p_dec",
                   "p_out", "f_in", "f_pos", "f_c", "f_t", "f_dec", "f_out", "yaw_h", "fpos_h"):
            if hasattr(s, nm):
                delattr(s, nm)
        s.watch, s.H = True, H
        s.use_yaw = True
        s.ctc = nn.Embedding(3, d)                                      # contact input: unknown / no / yes (HOIST-Former)
        s.ctc2 = nn.Embedding(3, d)                                     # the same from Hands23 (unknown where it did not run)
        s.iv_e = mlp(NP * 3, d)                                         # intervention: commanded path of the carried object (PS units)
        s.t1_g = mlp(REF_F, d); s.t1_h = mlp(4 * d, 7, 2 * d); s.pres_h = mlp(2 * d, 1, d); s.t1y_h = mlp(2 * d, MIX * 4, 2 * d)
        heads = s.trunk.layers[0].self_attn.num_heads
        s.g_in = mlp(3, d); s.g_step = nn.Parameter(torch.randn(H, d) * 0.02); s.g_obj = nn.Linear(d, d); s.g_t = mlp(d, d)
        s.g_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True, activation="gelu"), 3)
        s.g_out = nn.Linear(d, 3)
        s.d_h = mlp(2 * d, 3, d)
        s.t5_h = mlp(2 * d, H, d)
        # noise_floor: the tracker's measured rest noise as a floor on the T1 sd (world m per axis, added in quadrature to the model
        # variance). dist_head: a continuous per-object distance head on log(|T1 end - now| + n50) and log(|T2 displacement at +H| + n50)
        # (m), Huber loss; there is no moved / stayed cut-off in training
        s.sig0, s.dist_head = float(noise_floor), bool(dist_head)
        s.t1_no_table = bool(t1_no_table)                               # the table is not a T1 reference (ends relative to objects / own start)
        s.rem_head = bool(rem_head)                                     # remaining-motion fraction (scene token 0), used for stop
        if s.rem_head:
            s.rem_h = mlp(d, 1, d)
        if s.dist_head:
            s.e_h = mlp(2 * d, 2, d)

    def pointers(s, E, mv):
        raise RuntimeError("watch model: no next-object head; use next_scores (the T2 read-out)")

    def heads_from(s, E, mv, tg):
        raise RuntimeError("watch model: no destination / path context; use t1_* / sample_traj")

    def _oc(s, E):
        o = E["obj"]
        return torch.cat([o, E["cls"][:, 3][:, None].expand_as(o)], -1)

    # ---------------------------------------------------------------- T1
    def t1_mix(s, E):
        o, om = E["obj"], E["om"]
        B, n, d = o.shape
        p, yaw, sz = ref_frame(E["geo"].float())
        z = to_frame(p[:, :, None], p[:, None], yaw[:, None], sz[:, None]).clamp(-5, 5)             # (B, i, j, 3): i's start in j's frame
        eye = torch.eye(n, device=o.device)[None].expand(B, n, n)
        tab = (E["kindx"] == 1).float()[:, None, :].expand(B, n, n)
        f = torch.cat([z, sz[:, None].expand(B, n, n, 3), tab[..., None], eye[..., None]], -1)
        oi, oj = o[:, :, None].expand(B, n, n, d), o[:, None].expand(B, n, n, d)
        x = torch.cat([oi, oj, oi * oj, E["cls"][:, 2][:, None, None] + s.t1_g(f.to(o.dtype))], -1)
        y = s.t1_h(x).float()
        lg = y[..., 0].masked_fill(~om[:, None], -1e4)
        if getattr(s, "t1_no_table", False):                             # references = every other object + the object's own start
            lg = lg.masked_fill((E["kindx"] == 1)[:, None, :], -1e4)
        return dict(logit=lg, off=y[..., 1:4], logs=y[..., 4:7].clamp(-4, 1.5),
                    p=p[:, None].expand(B, n, n, 3), yaw=yaw[:, None].expand(B, n, n), sz=sz[:, None].expand(B, n, n, 3))

    def t1_sdf(s, M):
        """T1 component sd in j's frame (B, n, n, 3): exp(logs), with the noise floor sig0 (world m per axis) added in quadrature."""
        sd = M["logs"].exp()
        f = getattr(s, "sig0", 0.0)
        return torch.sqrt(sd ** 2 + (f / M["sz"]) ** 2) if f > 0 else sd

    def t1_logp(s, M, end):
        """End (B, n, 3) world -> log p (B, n, n) per (object i, reference j), including log a_ij."""
        zz = to_frame(end[:, :, None].float(), M["p"], M["yaw"], M["sz"])
        sd = s.t1_sdf(M)
        lg = -0.5 * (((zz - M["off"]) / sd) ** 2).sum(-1) - torch.log(sd).sum(-1) - torch.log(M["sz"]).sum(-1) - 1.5 * math.log(2 * math.pi)
        return F.log_softmax(M["logit"], -1) + lg

    def t1_nll(s, E, end):
        M = s.t1_mix(E)
        a = F.softmax(M["logit"], -1)
        s._diag = dict(a_entropy=float(-(a * torch.log(a.clamp(min=1e-9))).sum(-1)[E["om"]].mean().detach()))
        return -torch.logsumexp(s.t1_logp(M, end), -1)

    @torch.no_grad()
    def t1_mode(s, E):
        """Return (end (B, n, 3) world, the most likely reference's component mean; j (B, n); a (B, n, n); sd (B, n, 3), world-axis
        sd of that component (diagonal in j's frame, rotated magnitudes))."""
        M = s.t1_mix(E); a = F.softmax(M["logit"], -1); j = a.argmax(-1)
        end, sd = s.t1_comp(M, j)
        return end, j, a, sd * float(getattr(s, "sd_scale", 1.0))

    def t1_comp(s, M, j):
        """Component of reference j (B, n) for each object: (mean (B, n, 3) world, sd (B, n, 3) world-axis rotated magnitudes). The
        noise floor is included; the global sd_scale is not applied."""
        g3 = lambda t: torch.gather(t, 2, j[..., None, None].expand(*j.shape, 1, 3)).squeeze(2)        # noqa: E731
        off, sz = g3(M["off"]), g3(M["sz"]); yw = torch.gather(M["yaw"], 2, j[..., None]).squeeze(-1); pj = g3(M["p"])
        end = from_frame(off, pj, yw, sz)
        sdl = g3(s.t1_sdf(M)) * sz
        c, s_ = torch.cos(yw).abs(), torch.sin(yw).abs()
        return end, torch.stack([c * sdl[..., 0] + s_ * sdl[..., 1], s_ * sdl[..., 0] + c * sdl[..., 1], sdl[..., 2]], -1)

    def rem_logit(s, E):
        """(B,) logit of the fraction of the episode's total T1 motion still remaining (rem_head models)."""
        return s.rem_h(E["cls"][:, 0]).squeeze(-1).float()

    def present_logit(s, E):
        return s.pres_h(s._oc(E)).squeeze(-1).float()

    def t1_yaw_mix(s, E):
        z = s.t1y_h(s._oc(E)).view(*E["obj"].shape[:2], MIX, 4).float()
        return dict(logit=z[..., 0], mu=torch.atan2(z[..., 2], z[..., 1]), logk=z[..., 3].clamp(-2, 5))

    def t1_yaw_nll(s, E, yaw):
        z = s.t1_yaw_mix(E); k = z["logk"].exp()
        logp = k * torch.cos(2 * yaw[..., None] - z["mu"]) - (torch.log(torch.special.i0e(k)) + k) - math.log(2 * math.pi)
        return -torch.logsumexp(F.log_softmax(z["logit"], -1) + logp, -1)

    def t1_yaw_mode(s, E):
        z = s.t1_yaw_mix(E); i = z["logit"].argmax(-1)
        return 0.5 * z["mu"].gather(-1, i[..., None]).squeeze(-1)

    # ---------------------------------------------------------------- T2: joint flow over all objects
    def velocity_all(s, E, xt, t):
        """xt (B, n, H, 3) PS units at flow time t (B,) -> velocity (B, n, H, 3)."""
        from cleantables.worldmodel.base_model import temb
        B, n, H, _ = xt.shape
        q = s.g_in(xt.to(E["obj"].dtype)) + s.g_step[None, None] + s.g_obj(E["obj"])[:, :, None] + s.g_t(temb(t, s.d).to(E["obj"].dtype))[:, None, None]
        pad = (~E["om"])[:, :, None].expand(B, n, H).reshape(B, n * H)
        y = s.g_dec(q.reshape(B, n * H, -1), E["mem"], tgt_key_padding_mask=pad, memory_key_padding_mask=E["pad"])
        return s.g_out(y).view(B, n, H, 3)

    @torch.no_grad()
    def sample_traj(s, E, K=4, steps=12):
        """Return (B, K, n, H, 3): every object's displacement at the H steps (metres)."""
        from cleantables.worldmodel.base_model import PS
        B, n = E["obj"].shape[:2]
        Ek = {k: (v.repeat_interleave(K, 0) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B else v) for k, v in E.items()}
        x = torch.randn(B * K, n, s.H, 3, device=E["obj"].device)
        for i in range(steps):
            x = x + s.velocity_all(Ek, x, torch.full((B * K,), i / steps, device=x.device)).float() / steps
        return x.view(B, K, n, s.H, 3) * PS

    def mean_disp(s, E):
        """(B, n, 3) PS units: the mean last-step displacement (read-out for 'what moves next')."""
        return s.d_h(s._oc(E)).float()

    def dist_pred(s, E):
        """(B, n, 2): predicted log(|T1 end - now| + n50), log(|T2 displacement at +H| + n50) (metres; dist_head models)."""
        return s.e_h(s._oc(E)).float()

    def next_scores(s, E):
        """Next-object read-out: p_now (B, n) = each movable object's predicted displacement magnitude over the horizon, normalised over
        the movable objects (no threshold, no temperature). dist_head models use the 4 s distance head (exp(log d) - n50, floored at 0)."""
        from cleantables.worldmodel.base_model import PS
        m_ = (E["om"] & (E["kindx"] == 0)).float()
        if getattr(s, "dist_head", False):
            d = (s.dist_pred(E)[..., 1].exp() - N50).clamp(min=0) * m_
        else:
            d = s.mean_disp(E).norm(dim=-1) * PS * m_
        return d / d.sum(-1, keepdim=True).clamp(min=1e-9), d

    def t5_logit(s, E):
        return s.t5_h(s._oc(E)).float()

    @torch.no_grad()
    def stop_readout(s, E):
        """Stop read-out: every object's current position lies within one predicted sd of its most likely T1 end (per axis, using the
        component's own sd). Returns (stop (B,) bool, z (B,) the largest per-object normalised distance, p_stop = exp(-z^2 / 2))."""
        end, _, _, sd = s.t1_mode(E)
        cur = E["geo"][..., :3].float()
        zz = ((end - cur) / sd.clamp(min=1e-4)).abs().max(-1).values
        zz = zz.masked_fill(~(E["om"] & (E["kindx"] == 0)), 0.0).max(-1).values
        return zz < 1.0, zz, torch.exp(-0.5 * zz ** 2)

    # ---------------------------------------------------------------- physics fine-tuning interface (and T4)
    def phys_loss(s, b, mv, cmd_path, exec_end, exec_traj, traj_ok, parts=False):
        """Per-sample loss (B,) on PhysX replays of the model's own plans (bridge batch b, 3D only, no hands):
          (a) T1 NLL of the carried object's executed end (top centre, world m), encoded without the intervention (the plan's view);
          (b) T2 flow loss of every object's executed displacement at the H steps (exec_traj (B, n, H, 3) world m, relative to the
              current positions; traj_ok (B, n, H)), encoded with the intervention token (mv and its commanded path cmd_path
              (B, NP, 3) world m); this is T4, the effect of the carry on every object;
          (c) the same displacements without the intervention (T2 flow and the mean-displacement read-out d_h), so physics can
              change which object the next-object read-out picks and how it moves.
        parts=False returns (a) + (b) + (c) per sample (B,); parts=True returns dict(t1=(a), t4=(b), choice=(c)) so the caller can
        reward-weight (a) and (c) and train (b) on every replay unweighted."""
        from cleantables.worldmodel.base_model import PS
        B, n = b["obj_mask"].shape
        dev = b["obj_mask"].device; ar = torch.arange(B, device=dev)
        mode = torch.ones(B, dtype=torch.long, device=dev)
        E0 = s.encode(b, mode); E0["kind"] = b["obj_kind"]
        lp = s.t1_logp(s.t1_mix(E0), exec_end[:, None].expand(B, n, 3))[ar, mv]
        l1 = -torch.logsumexp(lp, -1)
        st = b["obj_geo"][ar, mv, :3].float()
        bi = dict(b); bi["iv_obj"] = mv; bi["iv_path"] = (cmd_path.float() - st[:, None]) / PS
        E1 = s.encode(bi, mode); E1["kind"] = b["obj_kind"]
        x1 = (exec_traj.float() / PS); w = (traj_ok.bool() & b["obj_mask"].bool()[..., None]).float()
        x0 = torch.randn_like(x1); t = torch.rand(B, device=dev)
        v = s.velocity_all(E1, (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1, t).float()
        l2 = (((v - (x1 - x0)) ** 2).sum(-1) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1.0)
        x0b = torch.randn_like(x1); tb = torch.rand(B, device=dev)
        vb = s.velocity_all(E0, (1 - tb)[:, None, None, None] * x0b + tb[:, None, None, None] * x1, tb).float()
        l3 = (((vb - (x1 - x0b)) ** 2).sum(-1) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1.0)
        wl = w[..., -1]
        l3 = l3 + 0.5 * (F.huber_loss(s.mean_disp(E0), x1[:, :, -1], reduction="none", delta=0.1).sum(-1) * wl).sum(1) / wl.sum(1).clamp(min=1.0)
        if getattr(s, "dist_head", False):                               # the distance head also learns the executed 4 s distance
            y2 = torch.log(exec_traj[:, :, -1].float().norm(dim=-1) + N50)
            l3 = l3 + 0.5 * (F.huber_loss(s.dist_pred(E0)[..., 1], y2, reduction="none", delta=0.5) * wl).sum(1) / wl.sum(1).clamp(min=1.0)
        if parts:
            return dict(t1=l1, t4=l2, choice=l3)
        return l1 + l2 + l3
