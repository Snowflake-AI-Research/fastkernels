"""CosyVoice3 Diffusion Transformer (DiT) backbone.

Adapted from vllm-omni CosyVoice3 cosyvoice3_dit.py.
Implements the DiT estimator with AdaLayerNorm modulation,
rotary embeddings, and classifier-free guidance support.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.dense_attention import DenseAttention


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0,
                         theta_rescale_factor: float = 1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0)
           * scale.unsqueeze(1)).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0, theta_rescale_factor=1.0,
                 freqs_for="lang", max_freq=10, num_freqs=1):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.theta_rescale_factor = theta_rescale_factor
        self.max_seq_len_cached = 0
        self.freqs_cis = None

    def _build_cache(self, seq_len: int, device: torch.device):
        if seq_len <= self.max_seq_len_cached and self.freqs_cis is not None:
            return
        self.max_seq_len_cached = max(seq_len, 2048)
        theta = self.theta * self.theta_rescale_factor ** (
            self.dim / (self.dim - 2))
        freqs = 1.0 / (theta ** (
            torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        t = torch.arange(self.max_seq_len_cached, device=device).float()
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.stack([freqs.cos(), freqs.sin()], dim=-1).to(device)

    def forward_from_seq_len(self, seq_len: int, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_cache(seq_len, device)
        return (self.freqs_cis[:seq_len].to(device), None)


def apply_rotary_pos_emb(t, freqs_cis_packed, scale=1.0):
    freqs_cos = freqs_cis_packed[..., 0]
    freqs_sin = freqs_cis_packed[..., 1]
    rot_dim = freqs_cos.shape[-1] * 2
    orig_dtype = t.dtype

    if t.dim() == 3:
        seq_len = t.shape[1]
        t_rot = t[..., :rot_dim]
        t_pass = t[..., rot_dim:]

        t_rot = t_rot.view(*t_rot.shape[:-1], -1, 2)
        freqs_cos = freqs_cos[:seq_len].unsqueeze(0)
        freqs_sin = freqs_sin[:seq_len].unsqueeze(0)

        if freqs_cos.dim() == 3:
            freqs_cos = freqs_cos.unsqueeze(-1).expand_as(t_rot[..., :1]).squeeze(-1)
            freqs_sin = freqs_sin.unsqueeze(-1).expand_as(t_rot[..., :1]).squeeze(-1)

        x0 = t_rot[..., 0]
        x1 = t_rot[..., 1]
        out0 = x0 * freqs_cos - x1 * freqs_sin
        out1 = x0 * freqs_sin + x1 * freqs_cos
        t_rot = torch.stack([out0, out1], dim=-1).flatten(-2)
        return torch.cat([t_rot * scale, t_pass], dim=-1).to(orig_dtype)
    return t


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0,
                 approximate="none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x_normed = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class DiTAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.scale = 1.0 / math.sqrt(dim_head)

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

        self.attn = DenseAttention()

    def forward(self, x, mask=None, rope=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale = xpos_scale if xpos_scale is not None else 1.0
            k_xpos_scale = xpos_scale ** -1.0 if xpos_scale is not None else 1.0
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        query = query.view(batch_size, seq_len, self.heads, self.dim_head)
        key = key.view(batch_size, seq_len, self.heads, self.dim_head)
        value = value.view(batch_size, seq_len, self.heads, self.dim_head)

        out = self.attn(query, key, value)
        out = out.view(batch_size, seq_len, self.inner_dim)
        out = out.to(query.dtype)
        out = self.to_out(out)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 4:
                mask = mask[:, 0, -1].unsqueeze(-1)
            out = out.masked_fill(~mask.bool(), 0.0)

        return out


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = DiTAttention(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv1(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv2(x)
        out = x.permute(0, 2, 1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, intermediate_dim, dilation=1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim,
            dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, spks):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = spks.unsqueeze(1).expand(-1, x.shape[1], -1)
            to_cat.append(spks)
        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    """Diffusion Transformer backbone for CosyVoice3 flow matching.

    Uses DenseAttention (L1 op) for the attention computation.
    """

    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(dim=dim, heads=heads, dim_head=dim_head,
                     ff_mult=ff_mult, dropout=dropout)
            for _ in range(depth)
        ])
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False)
            if long_skip_connection else None
        )
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        device = x.device
        rope = self.rotary_embed.forward_from_seq_len(seq_len, device=device)

        if self.long_skip_connection is not None:
            residual = x

        attn_mask = mask.bool().repeat(1, x.size(1), 1).unsqueeze(dim=1)

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output
