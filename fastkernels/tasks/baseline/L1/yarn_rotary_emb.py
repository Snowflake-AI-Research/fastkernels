"""Rotary position embeddings with YaRN / YARN scaling.

Two variants:
  - ``YaRNRotaryEmbedding``: NeoX-style YaRN RoPE used by GPT-OSS. Applies
    magnitude correction via ``mscale``.
  - ``YarnRotaryEmbedding``: DeepSeek-style YARN RoPE (interleaved, NON-NeoX)
    used by DeepSeek V3.  Supports separate ``mscale`` / ``mscale_all_dim``
    knobs and exposes ``softmax_mscale`` as an attention scaling factor.

Both classes share the same L1 CUDA rotary kernel via
``torch.ops.fastkernels_rope.rotary_embedding``; the only differences are how
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

# Detect FlashInfer rotary op once at import time.  vLLM's
# ``torch.ops.vllm.flashinfer_rotary_embedding`` is a thin wrapper around
# ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace`` (see
# ``vllm/model_executor/layers/rotary_embedding/common.py``), so we call the
# FlashInfer package directly instead of importing vllm to register the op.
try:
    from flashinfer.rope import (
        apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope,
    )
    _USE_FLASHINFER_ROPE = True
except Exception:
    _flashinfer_apply_rope = None
    _USE_FLASHINFER_ROPE = False


# GLM-5.2's plain "default" rope is applied via the vendored vLLM rotary
# kernel (base RotaryEmbedding.forward_cuda), exposed as the
# ``torch.ops.fastkernels_rope.rotary_embedding`` custom op registered by
# ``rotary_emb``. Importing it here ensures the op is defined.
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
        torch.ops.fastkernels_rope.rotary_embedding(
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
        is_plain: bool = False,
        cache_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        # ``is_plain`` marks a degenerate (scaling_factor==1.0) instance that is
        # really standard RoPE — e.g. GLM-5.2's ``rope_type: "default"``. vLLM
        # maps a "default" rope to the base ``RotaryEmbedding``, which does NOT
        # use the FlashInfer kernel and casts the cos/sin cache to the model
        # dtype (bf16). DeepSeek-V3.2 YARN (scaling_factor>1) keeps FlashInfer +
        # fp32 cache. Threading this flag lets both match vLLM exactly.
        self.is_plain = is_plain
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
        # Plain "default" rope (GLM-5.2): vLLM's base ``RotaryEmbedding`` stores
        # the cos/sin cache in the model compute dtype (bf16) once at init, so
        # its forward never re-casts. Match that — computing in fp32 then
        # casting to bf16 here is bit-identical to casting per-forward, and
        # skips a full-cache dtype conversion on every rope call. YARN
        # (is_plain=False) keeps the fp32 cache for the FlashInfer path.
        if self.is_plain and cache_dtype is not None:
            cache = cache.to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        # vLLM's ``DeepseekScalingRotaryEmbedding.forward_cuda`` prefers the
        # FlashInfer fused kernel when available (see
        # ``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:181-198``).
        # FlashInfer keeps ``cos_sin_cache`` in float32; only the fastkernels
        # CUDA kernel needs the cache cast to query.dtype.
        if _USE_FLASHINFER_ROPE and not self.is_plain \
                and query.dtype in (torch.float16, torch.bfloat16) \
                and self.head_dim in (64, 128, 256, 512):
            # Mirrors vLLM's ``flashinfer_rotary_embedding`` custom op, which
            # just forwards to this FlashInfer entry point in-place.
            _flashinfer_apply_rope(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_dim,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
            )
            return query, key
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        # GLM-5.2 plain "default" rope and the scaled path both go through the
        # vendored vLLM rotary kernel. Call it via the registered
        # ``fastkernels_rope`` custom op (whose CUDA impl is exactly
        # ``_C.rotary_embedding``) rather than the raw pybind function, so
        # ``torch.compile`` / cudagraph capture can trace it. Numerically
        # identical to calling ``_C.rotary_embedding`` directly.
        torch.ops.fastkernels_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, self.is_neox_style,
        )
        return query, key
