"""Scratch asynchronous history fusion + continuous-query matrix decoder.

Inspired by mTAN time representations, MulT cross-modal history attention and
Perceiver IO output queries. This is a CSI adaptation, not their reproduction.
NO pretrained CSI network. CNN features are the ONLY pretrained component.
CSI coefficient groups are computational groups, NOT physical antenna angles.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeCode(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer('freq', 2 ** torch.linspace(-2, 4, 16))
        self.net = nn.Sequential(nn.Linear(33, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, t):
        t = t.float() / .5
        a = t[..., None] * self.freq * math.pi
        return self.net(torch.cat([t[..., None], a.sin(), a.cos()], -1))


class Attention(nn.Module):
    def __init__(self, d, heads=4, dropout=.05):
        super().__init__()
        self.d, self.heads, self.dropout = d, heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2*d)
        self.out = nn.Linear(d, d)
        self.qnorm, self.knorm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.relative = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, heads))
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3*d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(3*d, d))

    def forward(self, q, kv, mask, qt=None, kt=None):
        B, N, D = q.shape
        present = mask.any(1)
        safe = mask.clone()
        safe[~present, 0] = True
        def split(x):
            return x.reshape(B, -1, self.heads, D//self.heads).transpose(1, 2)
        k, v = self.kv(self.knorm(kv)).chunk(2, -1)
        if qt is not None:
            dt = (qt[:, :, None] - kt[:, None, :]).float() / .5
            bias = self.relative(torch.stack([dt, dt.abs()], -1)).permute(0, 3, 1, 2)
            bias = bias.masked_fill(~safe[:, None, None], float('-inf'))
        else:
            bias = q.new_zeros(B, 1, 1, kv.shape[1]).masked_fill(~safe[:, None, None], float('-inf'))
        value = F.scaled_dot_product_attention(split(self.q(self.qnorm(q))), split(k), split(v),
            attn_mask=bias.to(k.dtype), dropout_p=self.dropout if self.training else 0.)
        value = self.out(value.transpose(1, 2).reshape(B, N, D)) * present[:, None, None]
        h = q + value
        return h + self.ff(h)


class AxialBlock(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.temporal = Attention(d, dropout=dropout)
        self.groups = Attention(d, dropout=dropout)

    def forward(self, x, mask, times):
        B, T, G, D = x.shape
        tm = mask[:, None].expand(B, G, T).reshape(B*G, T)
        tt = times[:, None].expand(B, G, T).reshape(B*G, T)
        h = x.transpose(1, 2).reshape(B*G, T, D)
        h = self.temporal(h, h, tm, tt, tt)
        h = h.reshape(B, G, T, D).transpose(1, 2).reshape(B*T, G, D)
        h = self.groups(h, h, torch.ones(B*T, G, dtype=torch.bool, device=x.device))
        return h.reshape(B, T, G, D) * mask[:, :, None, None]


class ScratchFusion(nn.Module):
    def __init__(self, features, modalities='WMI', revision=1, d=128, groups=8, layers=2):
        super().__init__()
        assert features % (2*groups) == 0
        self.features, self.modalities, self.revision = features, modalities, revision
        self.d, self.groups, self.width = d, groups, features//groups
        self.scale = math.sqrt(features / 2)
        self.time = TimeCode(d)
        self.group_id = nn.Parameter(torch.randn(groups, d) * .02)
        self.csi = nn.Sequential(nn.Linear(2*self.width, d), nn.GELU(), nn.LayerNorm(d))
        self.anchor = nn.Sequential(nn.Linear(self.width, d), nn.GELU(), nn.LayerNorm(d))
        self.global_csi = nn.Sequential(nn.Linear(features, d), nn.GELU(), nn.LayerNorm(d))
        self.motion = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.rgb = nn.Sequential(nn.Linear(2048, d), nn.GELU(), nn.LayerNorm(d))
        self.patch_id = nn.Parameter(torch.randn(64, d)*.02)
        self.camera_id = nn.Parameter(torch.randn(4, d)*.02)
        self.spatial_query = nn.Parameter(torch.randn(2, d)*.02)
        self.spatial = Attention(d, dropout=.0)
        self.motion_attention = Attention(d)
        self.rgb_attention = Attention(d)
        self.fusion = nn.Sequential(nn.LayerNorm(3*d), nn.Linear(3*d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.blocks = nn.ModuleList([AxialBlock(d, .05) for _ in range(layers)])
        self.decode = Attention(d)
        self.output_groups = Attention(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.width))
        self.multiplier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        # Small, nonzero random heads: all encoders receive gradients on step one.
        for head in [self.head[-1], self.multiplier[-1]]:
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)

    def visual_history(self, b, last):
        raw = b['rgb'].float()
        delta = torch.zeros_like(raw)
        delta[:, 1:] = raw[:, 1:] - raw[:, :-1]
        vk = self.rgb(torch.cat([raw, 3*delta], -1)) + self.patch_id
        B, T, V, P, D = vk.shape
        q = (self.global_csi(last*self.scale)[:, None, None, None]
             + self.camera_id[None, None, :, None] + self.spatial_query[None, None, None])
        q = q.expand(B, T, V, 2, D).reshape(B*T*V, 2, D)
        kv = vk.reshape(B*T*V, P, D)
        h = self.spatial(q, kv, torch.ones(B*T*V, P, device=raw.device, dtype=torch.bool))
        h = h.reshape(B, T, V, 2, D) + self.camera_id[None, None, :, None]
        rt = b['rgb_times'][:, :, None, None].expand(B, T, V, 2).reshape(B, -1)
        rm = b['rgb_mask'][:, :, None, None].expand(B, T, V, 2).reshape(B, -1)
        return h.reshape(B, -1, D) + self.time(rt), rt, rm

    def encode(self, b):
        x, mask, times = b['x'].float(), b['mask'], b['times']
        B, T, _ = x.shape
        last = x[torch.arange(B, device=x.device), mask.sum(1)-1]
        xx = x.reshape(B, T, self.groups, self.width)
        anchor = last.reshape(B, self.groups, self.width)
        h = self.csi(torch.cat([xx, xx-anchor[:, None]], -1)*self.scale)
        h = h + self.group_id + self.time(times)[:, :, None]
        query = h.mean(2)
        mc, vc = torch.zeros_like(query), torch.zeros_like(query)
        if 'M' in self.modalities:
            mk = self.motion(b['motion'].float()) + self.time(b['motion_times'])
            mc = self.motion_attention(query, mk, b['motion_mask'], times, b['motion_times']) - query
            mc = mc * b['motion_mask'].any(1)[:, None, None]
        if 'I' in self.modalities:
            vk, rt, rm = self.visual_history(b, last)
            vc = self.rgb_attention(query, vk, rm, times, rt) - query
            vc = vc * rm.any(1)[:, None, None]
        # Auxiliary history enters BEFORE the sequence prediction network.
        context = self.fusion(torch.cat([query, mc, vc], -1))
        h = (h + context[:, :, None]) * mask[:, :, None, None]
        for block in self.blocks:
            h = block(h, mask, times)
        return h, last

    def forward(self, b):
        h, last = self.encode(b)
        B, T, G, D = h.shape
        Q = b['query'].shape[1]
        a = self.anchor(last.reshape(B, G, self.width)*self.scale) + self.group_id
        q = a[:, None] + self.time(b['query'])[:, :, None]
        q = q.transpose(1, 2).reshape(B*G, Q, D)
        kv = h.transpose(1, 2).reshape(B*G, T, D)
        qt = b['query'][:, None].expand(B, G, Q).reshape(B*G, Q)
        kt = b['times'][:, None].expand(B, G, T).reshape(B*G, T)
        mask = b['mask'][:, None].expand(B, G, T).reshape(B*G, T)
        q = self.decode(q, kv, mask, qt, kt)
        q = q.reshape(B, G, Q, D).transpose(1, 2).reshape(B*Q, G, D)
        q = self.output_groups(q, q, torch.ones(B*Q, G, dtype=torch.bool, device=q.device))
        q = q.reshape(B, Q, G, D)
        raw = self.head(q).float().reshape(B, Q, self.features)/self.scale
        ab = self.multiplier(q.mean(2)).float()
        z = torch.view_as_complex(last.contiguous().reshape(B, 1, -1, 2))
        u = torch.view_as_complex(raw.contiguous().reshape(B, Q, -1, 2))
        parallel = (z.conj()*u).sum(-1, keepdim=True)/z.abs().square().sum(-1, keepdim=True).clamp_min(1e-12)
        alpha = torch.complex(ab[..., 0], ab[..., 1])[..., None]
        prediction = z + (b['query'].float()/.1)[..., None]*(alpha*z + u-parallel*z)
        return torch.view_as_real(prediction).flatten(-2)
