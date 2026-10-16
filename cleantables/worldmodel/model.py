"""The world model: one transformer over [4 summary | task | 24 object | 2 hand | 32 V-JEPA] tokens.

Heads: next-object and destination pointers, relation, a mixture density over the end offset, a flow-matching
decoder for the object and hand path (anchored to a sampled end displacement), task progress, the future V-JEPA
latent and p(seen).
Modality modes: 0 = video and 3D, 1 = 3D only (video tokens masked), 2 = video only (box geometry replaced by a learned
null token, hands masked; object text and crop features stay so the pointers can still address objects).
Parameter names are short and stable so trained checkpoints load as they are.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PATH = 32  # points per path
PATH_SCALE = 0.25  # metres per path unit
N_MIX = 5  # mixture components of the end heads
HAND_DIM = 13  # valid flag, centroid, extent, two principal axes
MAX_OBJECTS = 24  # object token slots
TABLE, LEAVES = 24, 25  # destination indices after the object slots
END_OFFSET_SCALE = 0.1  # metres per unit of the end-offset mixture

MODE_BOTH, MODE_3D, MODE_VIDEO = 0, 1, 2


def mlp(n_in: int, n_out: int, hidden: int | None = None) -> nn.Sequential:
    hidden = hidden or n_out
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    freq = torch.exp(torch.arange(dim // 2, device=t.device) * (-math.log(1000.0) / (dim // 2)))
    angle = t[:, None] * 1000 * freq[None]
    return torch.cat([angle.sin(), angle.cos()], -1)


def gather_tokens(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """h[b, idx[b]], zero where idx < 0."""
    g = h[torch.arange(h.shape[0], device=h.device), idx.clamp(min=0)]
    return g * (idx >= 0).float()[:, None]


def _encoder(d: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def _decoder(d: int, heads: int, layers: int, dropout: float) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerDecoder(layer, layers)


def _mixture(z: torch.Tensor) -> dict:
    z = z.view(-1, N_MIX, 7)
    return dict(logit=z[..., 0], mu=z[..., 1:4], logs=z[..., 4:7].clamp(-3, 1.5))


class WorldModel(nn.Module):
    def __init__(self, d: int = 256, layers: int = 6, heads: int = 8, flow_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d = d
        # input tokens
        self.geo = mlp(8, d)
        self.null_geo = nn.Parameter(torch.zeros(d))
        self.txt = nn.Linear(384, d)
        self.crop = nn.Linear(768, d)
        self.kind = nn.Embedding(3, d)
        self.hgeo = mlp(HAND_DIM - 1, d)
        self.hq = nn.Linear(16, d)
        self.hid = nn.Embedding(2, d)
        self.vj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        self.vpos = nn.Parameter(torch.randn(32, d) * 0.02)
        self.task = nn.Linear(384, d)
        self.cls = nn.Parameter(torch.randn(4, d) * 0.02)
        self.mod = nn.Embedding(3, d)
        self.typ = nn.Embedding(5, d)  # summary / task / object / hand / video
        self.trunk = _encoder(d, heads, layers, dropout)
        self.norm = nn.LayerNorm(d)
        # pointers, relation and end heads
        self.rel = mlp(15, d)  # explicit mover-target geometry added to the mover token
        self.mover_h = mlp(d, 1)
        self.tgt_h = mlp(3 * d, 1)
        self.tgt_x = mlp(2 * d, 2)  # TABLE / LEAVES logits
        self.xemb = nn.Parameter(torch.randn(2, d) * 0.02)
        self.rel_h = mlp(3 * d, 6)
        self.mdn = mlp(3 * d, N_MIX * 7, 2 * d)
        self.disp = mlp(3 * d, N_MIX * 7, 2 * d)  # end displacement of the mover, path units, table axes
        # flow-matching path decoder (object + hand, 6 dims)
        self.f_in = mlp(6, d)
        self.f_pos = nn.Parameter(torch.randn(N_PATH, d) * 0.02)
        self.f_c = nn.Linear(3 * d, d)
        self.f_t = mlp(d, d)
        self.f_dec = _decoder(d, heads, flow_layers, dropout)
        self.f_out = nn.Linear(d, 6)
        # progress: whole clip, time to the next move, per path step
        self.p_vid = mlp(d, 1)
        self.p_ttm = mlp(d, 1)
        self.p_in = mlp(6, d)
        self.p_pos = nn.Parameter(torch.randn(N_PATH, d) * 0.02)
        self.p_c = nn.Linear(3 * d, d)
        self.p_dec = _decoder(d, heads, 2, dropout)
        self.p_out = mlp(d, 1)
        # future V-JEPA latent and p(seen)
        self.jepa = mlp(d, 1024, 2 * d)
        self.seen = mlp(d, 1)

    # ------------------------------------------------------------------ trunk
    def encode(self, b: dict, mode: torch.Tensor) -> dict:
        """Run the trunk. Returns the token outputs plus what the heads need (geometry, masks, mode)."""
        B = b["obj_geo"].shape[0]
        obj_mask = b["obj_mask"].bool()
        has_geo = (mode != MODE_VIDEO)[:, None, None].float()
        g = self.geo(b["obj_geo"]) * has_geo + self.null_geo * (1 - has_geo)
        o = (
            g
            + self.txt(b["obj_txt"])
            + self.crop(b["obj_crop"]) * b["obj_crop_ok"].float()[..., None]
            + self.kind(b["obj_kind"])
        )
        hg = b["hand_geo"]
        h = self.hgeo(hg[..., 1:]) + self.hq(b["hand_q16"]) * b["hand_q_ok"].float()[..., None] + self.hid.weight[None]
        hand_mask = (hg[..., 0] > 0.5) & (mode != MODE_VIDEO)[:, None]
        v = self.vj(b["vj_tok"]) + self.vpos
        video_mask = (b["vj_ok"].bool() & (mode != MODE_3D))[:, None].expand(B, 32)
        cls = self.cls[None].expand(B, -1, -1) + self.mod(mode)[:, None]
        t = self.task(b["task_emb"])[:, None]
        typ = self.typ.weight
        x = torch.cat([cls + typ[0], t + typ[1], o + typ[2], h + typ[3], v + typ[4]], 1)
        valid = torch.cat([torch.ones(B, 5, dtype=torch.bool, device=x.device), obj_mask, hand_mask, video_mask], 1)
        y = self.norm(self.trunk(x, src_key_padding_mask=~valid))
        return dict(
            cls=y[:, :4],
            obj=y[:, 5:29],
            mem=y,
            pad=~valid,
            om=obj_mask,
            geo=b["obj_geo"],
            mode=mode,
            kind=b["obj_kind"],
        )

    def context(self, E: dict, mover: torch.Tensor, target: torch.Tensor):
        """Summary token, mover token (plus explicit relative geometry) and target token for the given pair."""
        glob = E["cls"][:, 2]
        m = gather_tokens(E["obj"], mover)
        t = gather_tokens(
            E["obj"], target.clamp(max=MAX_OBJECTS - 1).where(target < MAX_OBJECTS, torch.full_like(target, -1))
        )
        is_extra = target >= MAX_OBJECTS
        t = t + is_extra.float()[:, None] * self.xemb[(target - MAX_OBJECTS).clamp(0, 1)]
        geo = E["geo"]
        table_mask = (E["kind"] == 1) & E["om"]
        table = torch.where(table_mask.any(-1), table_mask.float().argmax(-1), torch.full_like(target, -1))
        tidx = torch.where(target < MAX_OBJECTS, target, table)
        mp, ts = gather_tokens(geo, mover), gather_tokens(geo, tidx)
        f = torch.cat(
            [
                (ts[:, :3] - mp[:, :3]) / PATH_SCALE,
                mp[:, 3:6] / PATH_SCALE,
                ts[:, 3:6] / PATH_SCALE,
                ts[:, 6:8],
                mp[:, 6:8],
                (target == TABLE).float()[:, None],
                (target == LEAVES).float()[:, None],
            ],
            -1,
        )
        f = f * ((E["mode"] != MODE_VIDEO) & (mover >= 0)).float()[:, None]
        return glob, m + self.rel(f), t

    def pointers(self, E: dict, mover_for_target: torch.Tensor):
        """Logits over the next object (24) and the destination of `mover_for_target` (24 objects, TABLE, LEAVES)."""
        o, om = E["obj"], E["om"]
        movable = om & (E["kind"] == 0)
        mover_logits = self.mover_h(o).squeeze(-1).masked_fill(~movable, -1e4)
        glob = E["cls"][:, 2]
        m = gather_tokens(o, mover_for_target)
        obj_logits = self.tgt_h(torch.cat([o, m[:, None].expand_as(o), o * m[:, None]], -1)).squeeze(-1)
        not_self = torch.ones_like(om)
        not_self[torch.arange(len(om)), mover_for_target.clamp(min=0)] = False
        obj_logits = obj_logits.masked_fill(~(om & not_self), -1e4)
        extra_logits = self.tgt_x(torch.cat([glob, m], -1))
        return mover_logits, torch.cat([obj_logits, extra_logits], 1)

    def heads_from(self, E: dict, mover: torch.Tensor, target: torch.Tensor):
        """Conditioning vector c for the path heads, relation logits and the target-relative end-offset mixture."""
        glob, m, t = self.context(E, mover, target)
        c = torch.cat([glob, m, t], -1)
        return c, self.rel_h(c), _mixture(self.mdn(c))

    def end_disp(self, c: torch.Tensor) -> dict:
        """Mixture over the mover's end displacement (path units, table axes)."""
        return _mixture(self.disp(c))

    # ------------------------------------------------------------------ flow matching
    def velocity(self, E: dict, c: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        q = self.f_in(xt) + self.f_pos + (self.f_c(c) + self.f_t(time_embedding(t, self.d)))[:, None]
        return self.f_out(self.f_dec(q, E["mem"], memory_key_padding_mask=E["pad"]))

    @torch.no_grad()
    def sample_residuals(self, E: dict, c: torch.Tensor, K: int = 8, steps: int = 12) -> torch.Tensor:
        """K Euler-integrated flow samples per row, (B, K, N_PATH, 6) in path units."""
        B = c.shape[0]
        Ek = dict(mem=E["mem"].repeat_interleave(K, 0), pad=E["pad"].repeat_interleave(K, 0))
        ck = c.repeat_interleave(K, 0)
        x = torch.randn(B * K, N_PATH, 6, device=c.device)
        for i in range(steps):
            t = torch.full((B * K,), i / steps, device=c.device)
            x = x + self.velocity(Ek, ck, x, t) / steps
        return x.view(B, K, N_PATH, 6)

    @torch.no_grad()
    def sample_paths_anchored(self, E: dict, c: torch.Tensor, K: int = 8, steps: int = 12) -> torch.Tensor:
        """K paths (B, K, N_PATH, 6) in metres from the mover: residual sample + straight line to a sampled end."""
        r = self.sample_residuals(E, c, K, steps)
        z = self.end_disp(c)
        B = c.shape[0]
        comp = torch.multinomial(F.softmax(z["logit"].float(), -1), K, replacement=True)
        mu = torch.gather(z["mu"].float(), 1, comp[..., None].expand(B, K, 3))
        sd = torch.gather(z["logs"].float(), 1, comp[..., None].expand(B, K, 3)).exp()
        # half-sd draws: the spread of the paths follows the mixture modes rather than each mode's width
        end = mu + 0.5 * sd * torch.randn_like(mu)
        u = torch.linspace(0, 1, N_PATH, device=c.device)[None, None, :, None]
        line = u * end[:, :, None, :]
        return torch.cat([r[..., :3] + line, r[..., 3:6] + line], -1) * PATH_SCALE

    # ------------------------------------------------------------------ progress and latent heads
    def path_progress(self, E: dict, c: torch.Tensor, path6: torch.Tensor) -> torch.Tensor:
        q = self.p_in(path6 / PATH_SCALE) + self.p_pos + self.p_c(c)[:, None]
        return torch.sigmoid(self.p_out(self.p_dec(q, E["mem"], memory_key_padding_mask=E["pad"])).squeeze(-1))

    def clip_progress_logit(self, E: dict) -> torch.Tensor:
        return self.p_vid(E["cls"][:, 0]).squeeze(-1)

    def future_latent(self, E: dict, b: dict, mode: torch.Tensor) -> torch.Tensor:
        """Future global V-JEPA latent: persistence (mean clip token when video is visible) + a learned residual."""
        sees_video = (mode != MODE_3D).float()[:, None]
        base = F.normalize(b["vj_tok"].float().mean(1), dim=-1) * sees_video
        return F.normalize(base, dim=-1) * sees_video + self.jepa(E["cls"][:, 1])
