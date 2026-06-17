"""Rotary position embeddings with YaRN / YARN scaling.

Two variants:
  - ``YaRNRotaryEmbedding``: NeoX-style YaRN RoPE used by GPT-OSS. Applies
    magnitude correction via ``mscale``.
  - ``YarnRotaryEmbedding``: DeepSeek-style YARN RoPE (interleaved, NON-NeoX)
    used by DeepSeek V3.  Supports separate ``mscale`` / ``mscale_all_dim``
    knobs and exposes ``softmax_mscale`` as an attention scaling factor.

Both classes share the same L1 CUDA rotary kernel via
``torch.ops.kb_nano_rope.rotary_embedding``; the only differences are how
the ``cos_sin_cache`` is computed and whether NeoX layout is used.

References:
  - Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models"
  - vLLM: ``vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py``
  - vLLM: ``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Detect FlashInfer rotary op once at import time.  vLLM registers
# ``torch.ops.vllm.flashinfer_rotary_embedding`` only when the FlashInfer
# package is installed and the platform supports it.  We mirror that check
# without forcing a hard dependency.
def _detect_flashinfer_rope() -> bool:
    try:
        import vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope  # noqa: F401 — registers op
        return hasattr(torch.ops.vllm, "flashinfer_rotary_embedding")
    except Exception:
        return False

_USE_FLASHINFER_ROPE = _detect_flashinfer_rope()

from .rotary_emb import RotaryEmbedding


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float,
    max_position_embeddings: int, truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale: float) -> float:
    """GPT-OSS style mscale (no explicit mscale parameter)."""
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def yarn_get_mscale(scale: float, mscale: float) -> float:
    """DeepSeek-style mscale with explicit parameter (matches vLLM)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YaRNRotaryEmbedding(nn.Module):
    """YaRN RoPE with precomputed cos/sin cache.

    Uses the same L1 CUDA kernel as RotaryEmbedding for the rotation step
    (NeoX layout).  Used by GPT-OSS.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        rotary_dim = head_dim

        pos_freqs = rope_theta ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, rope_theta,
            original_max_position_embeddings, truncate,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
        )
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        mscale = _yarn_get_mscale(scaling_factor)

        max_t = int(max_position_embeddings * scaling_factor)
        t = torch.arange(max_t, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        if torch.compiler.is_compiling():
            return RotaryEmbedding.forward_native(
                positions, query, key, self.head_dim, cache,
            )
        torch.ops.kb_nano_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, True,
        )
        return query, key


class YarnRotaryEmbedding(nn.Module):
    """DeepSeek-style YARN (Yet Another RoPE extensioN) RoPE.

    Uses NON-NeoX (interleaved) layout, matching vLLM's
    ``DeepseekScalingRotaryEmbedding``.  The cos/sin cache is scaled by
    ``softmax_mscale`` which folds the attention magnitude correction into
    the rotary cache (so attention scores do not need to multiply by
    ``softmax_mscale`` separately).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        is_neox_style: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        rotary_dim = head_dim
        base = rope_theta

        softmax_mscale = (
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
            * attn_factor
        )
        self.softmax_mscale = softmax_mscale

        pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, base, max_position_embeddings,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
        ) * extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        t = torch.arange(max_position_embeddings * scaling_factor, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * softmax_mscale
        sin = freqs.sin() * softmax_mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        # vLLM's ``DeepseekScalingRotaryEmbedding.forward_cuda`` prefers the
        # FlashInfer fused kernel when available (see
        # ``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:181-198``).
        # FlashInfer keeps ``cos_sin_cache`` in float32; only the kb_nano
        # CUDA kernel needs the cache cast to query.dtype.
        if _USE_FLASHINFER_ROPE and query.dtype in (torch.float16, torch.bfloat16) \
                and self.head_dim in (64, 128, 256, 512):
            torch.ops.vllm.flashinfer_rotary_embedding(
                positions, query, key, self.head_dim, self.cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        torch.ops.kb_nano_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, self.is_neox_style,
        )
        return query, key
