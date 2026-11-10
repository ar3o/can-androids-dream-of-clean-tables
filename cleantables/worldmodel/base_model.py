"""First version of the object-centric world model; cleantables.worldmodel.model extends it.

One transformer trunk over [4 CLS | task | 24 object tokens | 2 hand tokens | 32 V-JEPA 2 tokens].
Heads: mover/target pointers, relation and an end-offset mixture density; a flow-matching path decoder for the object
and hand path (K samples); progress (video scalar, time to move, per-step path progress); a future V-JEPA latent; p(seen).
Modality modes: 0 both, 1 3D only (video tokens masked), 2 video only (geometry replaced by a learned null token, hands
masked; name and crop features stay so the pointers can still address objects)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NP, PS, MIX = 32, 0.25, 5          # path steps, path scale (m), mixture components
HG = 13


def mlp(i, o, h=None):
    h = h or o
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def temb(t, d):
    f = torch.exp(torch.arange(d // 2, device=t.device) * (-math.log(1000.0) / (d // 2)))
    a = t[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


def sel(h, idx):
    """Gather h[b, idx[b]]; zero where idx < 0."""
    g = h[torch.arange(h.shape[0], device=h.device), idx.clamp(min=0)]
    return g * (idx >= 0).float()[:, None]


class UWM(nn.Module):
    def __init__(s, d=256, L=6, heads=8, flow_layers=3, drop=0.1, path_resid=False):
        super().__init__()
        s.d = d
        s.path_resid = path_resid
        s.geo = mlp(8, d); s.null_geo = nn.Parameter(torch.zeros(d)); s.txt = nn.Linear(384, d); s.crop = nn.Linear(768, d)
        s.kind = nn.Embedding(3, d)
        s.hgeo = mlp(HG - 1, d); s.hq = nn.Linear(16, d); s.hid = nn.Embedding(2, d)
        s.vj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d)); s.vpos = nn.Parameter(torch.randn(32, d) * 0.02)
        s.task = nn.Linear(384, d); s.cls = nn.Parameter(torch.randn(4, d) * 0.02); s.mod = nn.Embedding(3, d)
        s.typ = nn.Embedding(5, d)                                           # cls/task/obj/hand/video
        s.trunk = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu"), L)
        s.norm = nn.LayerNorm(d)
        # pointers
        s.rel = mlp(15, d)                                                    # explicit mover/target relative geometry
        s.mover_h = mlp(d, 1)
        s.tgt_h = mlp(3 * d, 1); s.tgt_x = mlp(2 * d, 2)                     # objects ; (TABLE, LEAVES)
        s.xemb = nn.Parameter(torch.randn(2, d) * 0.02)                      # learned target tokens for TABLE / LEAVES
        s.rel_h = mlp(3 * d, 6)
        s.mdn = mlp(3 * d, MIX * 7, 2 * d)
        if path_resid:      # end-anchored paths: mixture over the mover's end displacement (PS units, world axes); the flow
            s.disp = mlp(3 * d, MIX * 7, 2 * d)       # then models only the residual around the straight line start -> sampled end
        # flow matching decoder (object + hand path, 6 dims)
        s.f_in = mlp(6, d); s.f_pos = nn.Parameter(torch.randn(NP, d) * 0.02); s.f_c = nn.Linear(3 * d, d); s.f_t = mlp(d, d)
        s.f_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu"), flow_layers)
        s.f_out = nn.Linear(d, 6)
        # progress
        s.p_vid = mlp(d, 1); s.p_ttm = mlp(d, 1)
        s.p_in = mlp(6, d); s.p_pos = nn.Parameter(torch.randn(NP, d) * 0.02); s.p_c = nn.Linear(3 * d, d)
        s.p_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu"), 2)
        s.p_out = mlp(d, 1)
        # JEPA latent + p(seen)
        s.jepa = mlp(d, 1024, 2 * d); s.seen = mlp(d, 1)

    # ---------------------------------------------------------------- trunk
    def encode(s, b, mode):
        B = b["obj_geo"].shape[0]
        om = b["obj_mask"].bool()
        vm = (mode != 2)[:, None, None].float()
        g = s.geo(b["obj_geo"]) * vm + s.null_geo * (1 - vm)
        o = g + s.txt(b["obj_txt"]) + s.crop(b["obj_crop"]) * b["obj_crop_ok"].float()[..., None] + s.kind(b["obj_kind"])
        hg = b["hand_geo"]
        h = s.hgeo(hg[..., 1:]) + s.hq(b["hand_q16"]) * b["hand_q_ok"].float()[..., None] + s.hid.weight[None]
        hmask = (hg[..., 0] > 0.5) & (mode != 2)[:, None]
        v = s.vj(b["vj_tok"]) + s.vpos
        vmask = (b["vj_ok"].bool() & (mode != 1))[:, None].expand(B, 32)
        cls = s.cls[None].expand(B, -1, -1) + s.mod(mode)[:, None]
        t = s.task(b["task_emb"])[:, None]
        x = torch.cat([cls + s.typ.weight[0], t + s.typ.weight[1], o + s.typ.weight[2], h + s.typ.weight[3], v + s.typ.weight[4]], 1)
        valid = torch.cat([torch.ones(B, 5, dtype=torch.bool, device=x.device), om, hmask, vmask], 1)
        y = s.norm(s.trunk(x, src_key_padding_mask=~valid))
        return dict(cls=y[:, :4], obj=y[:, 5:29], mem=y, pad=~valid, om=om, geo=b["obj_geo"], mode=mode, kindx=b["obj_kind"])

    def ctx(s, E, mv, tg):
        """Return the global token, mover token and target token (TABLE/LEAVES use learned tokens, -1 gives zeros)."""
        glob = E["cls"][:, 2]
        m = sel(E["obj"], mv)
        tt = sel(E["obj"], tg.clamp(max=23).where(tg < 24, torch.full_like(tg, -1)))
        isx = tg >= 24
        tt = tt + isx.float()[:, None] * s.xemb[(tg - 24).clamp(0, 1)]
        geo = E["geo"]
        tabm = (E["kindx"] == 1) & E["om"]
        tab = torch.where(tabm.any(-1), tabm.float().argmax(-1), torch.full_like(tg, -1))
        tidx = torch.where(tg < 24, tg, tab)
        mp, ts = sel(geo, mv), sel(geo, tidx)
        f = torch.cat([(ts[:, :3] - mp[:, :3]) / PS, mp[:, 3:6] / PS, ts[:, 3:6] / PS, ts[:, 6:8], mp[:, 6:8],
                       (tg == 24).float()[:, None], (tg == 25).float()[:, None]], -1)
        f = f * ((E["mode"] != 2) & (mv >= 0)).float()[:, None]
        m = m + s.rel(f)
        return glob, m, tt

    def pointers(s, E, mv_for_tgt):
        o, om = E["obj"], E["om"]
        mov = om & (E["kind"] == 0)
        ml = s.mover_h(o).squeeze(-1).masked_fill(~mov, -1e4)
        glob = E["cls"][:, 2]
        m = sel(o, mv_for_tgt)
        tl_o = s.tgt_h(torch.cat([o, m[:, None].expand_as(o), o * m[:, None]], -1)).squeeze(-1)
        not_self = torch.ones_like(om)
        not_self[torch.arange(len(om)), mv_for_tgt.clamp(min=0)] = False
        tl_o = tl_o.masked_fill(~(om & not_self), -1e4)
        tl_x = s.tgt_x(torch.cat([glob, m], -1))
        pad = torch.full((len(om), 0), 0.0, device=o.device)
        tl = torch.cat([tl_o, tl_x], 1)                                      # 24 objects + TABLE(24) + LEAVES(25)
        return ml, tl

    def heads_from(s, E, mv, tg):
        glob, m, t = s.ctx(E, mv, tg)
        c = torch.cat([glob, m, t], -1)
        rel = s.rel_h(c)
        z = s.mdn(c).view(-1, MIX, 7)
        return c, rel, dict(logit=z[..., 0], mu=z[..., 1:4], logs=z[..., 4:7].clamp(-3, 1.5))

    # ---------------------------------------------------------------- flow matching
    def velocity(s, E, c, xt, t):
        q = s.f_in(xt) + s.f_pos + (s.f_c(c) + s.f_t(temb(t, s.d)))[:, None]
        return s.f_out(s.f_dec(q, E["mem"], memory_key_padding_mask=E["pad"]))

    @torch.no_grad()
    def sample_paths(s, E, c, K=8, steps=12):
        B = c.shape[0]
        Ek = dict(mem=E["mem"].repeat_interleave(K, 0), pad=E["pad"].repeat_interleave(K, 0))
        ck = c.repeat_interleave(K, 0)
        x = torch.randn(B * K, NP, 6, device=c.device)
        for i in range(steps):
            t = torch.full((B * K,), i / steps, device=c.device)
            x = x + s.velocity(Ek, ck, x, t) / steps
        return x.view(B, K, NP, 6) * PS

    def end_disp(s, c):
        z = s.disp(c).view(-1, MIX, 7)
        return dict(logit=z[..., 0], mu=z[..., 1:4], logs=z[..., 4:7].clamp(-3, 1.5))

    @torch.no_grad()
    def sample_paths_anchored(s, E, c, K=8, steps=12):
        """K paths = residual sample + straight line from the mover origin to an end sampled from the displacement mixture (metres)."""
        r = s.sample_paths(E, c, K, steps) / PS                                # (B,K,NP,6) residual, PS units
        z = s.end_disp(c); B = c.shape[0]
        comp = torch.multinomial(F.softmax(z["logit"].float(), -1), K, replacement=True)
        mu = torch.gather(z["mu"].float(), 1, comp[..., None].expand(B, K, 3)); sd = torch.gather(z["logs"].float(), 1, comp[..., None].expand(B, K, 3)).exp()
        dsp = mu + 0.5 * sd * torch.randn_like(mu)                                  # sample at half the sd so the spread stays near the mixture modes
        u = torch.linspace(0, 1, NP, device=c.device)[None, None, :, None]
        line = u * dsp[:, :, None, :]
        return torch.cat([r[..., :3] + line, r[..., 3:6] + line], -1) * PS

    # ---------------------------------------------------------------- progress over a path
    def path_prog(s, E, c, path6):
        q = s.p_in(path6 / PS) + s.p_pos + s.p_c(c)[:, None]
        return torch.sigmoid(s.p_out(s.p_dec(q, E["mem"], memory_key_padding_mask=E["pad"])).squeeze(-1))

    def jepa_pred(s, E, b, mode):
        """Future global latent: persistence (mean of the clip's V-JEPA tokens, only when video is visible) plus a learned residual; trained with a cosine target."""
        base = F.normalize(b["vj_tok"].float().mean(1), dim=-1) * (mode != 1).float()[:, None]
        return F.normalize(base, dim=-1) * (mode != 1).float()[:, None] + s.jepa(E["cls"][:, 1])
