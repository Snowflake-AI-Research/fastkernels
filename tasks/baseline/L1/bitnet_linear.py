"""BitLinear: native W1.58A8 linear layers for ``microsoft/bitnet-b1.58-2B-4T``.

This module provides two parameter-holding modules that wrap the
:func:`bitnet_int8xint2_linear` kernel from
``kb_nano/tasks/baseline/L1/bitnet_int8xint2_linear.py``:

* :class:`BitLinear` - single projection (one HF Linear in, one out).
* :class:`BitLinearMerged` - fused projection used to consolidate
  ``{q_proj, k_proj, v_proj}`` into one ``qkv_proj`` and
  ``{gate_proj, up_proj}`` into one ``gate_up_proj``.  This matches the
  SOTA reference (``vllm_repo/BitNet/gpu/model.py``'s ``wqkv`` / ``w13``)
  and lets a single GEMM kernel launch handle three (resp. two) projections.

Weight memory layout (for both classes)::

    weight       : uint8, (total_out, in_features // 4)   KN-packed ternary
    weight_scale : bf16,  (total_out,)                    per-output-row scale
    bias         : bf16,  (total_out,)  -- optional

The on-disk HuggingFace checkpoint stores ``weight`` as ``(out//4, in)``
uint8 packed along OUT (a 2-bit interleave per byte) and ``weight_scale``
as a single bf16 scalar.  The ``weight_loader`` callbacks below unpack the
HF format and re-pack into the simpler KN-major layout once at load time;
decode arithmetic happens in native int8 / int2 with bf16 scale folding.
For prefill, each module also materializes the SOTA-equivalent bf16 fake-
quant weight so large-M calls can use cuBLAS ``F.linear`` like Microsoft's
official split prefill/decode implementation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitnet_int8xint2_linear import (
    VALUES_PER_BYTE,
    bitnet_int8xint2_linear_official,
    bitnet_int8xint2_linear,
    bitnet_official_kernel_available,
    hf_packed_to_kn_packed,
    pack_ternary_sota_ladder,
    unpack_kn_to_ternary,
)


__all__ = ["BitLinear", "BitLinearMerged", "VALUES_PER_BYTE"]


_BITNET_FORCE_BF16 = bool(int(__import__("os").environ.get(
    "KB_BITNET_FORCE_BF16", "0")))


def _bitnet_use_bf16_path(M: int) -> bool:
    """True iff this call should use the bf16 fake-quant prefill path.

    Mirrors SOTA's hybrid model split (separate ``BitLinear(nn.Linear)``
    for prefill, ``BitLinearKernel`` for decode) using kb-nano's engine
    context flag.  Outside an engine context we fall back to an
    M-threshold heuristic (1024) so unit tests / micro-benchmarks pick
    a sensible default.

    ``KB_BITNET_FORCE_BF16=1`` short-circuits to the bf16 path always —
    used to bisect alignment vs throughput regressions: if avg-match
    stays the same with this on, the int8 kernel is correctly aligned;
    if it jumps, the int8 path has divergence (kernel/quantization).
    """
    if _BITNET_FORCE_BF16:
        return True
    try:
        from ....infra.context import get_context
        ctx = get_context()
        return bool(ctx.is_prefill)
    except Exception:
        return M > 1024


@torch.compile(dynamic=True)
def _fake_quant_act_bf16(x: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric int8 *fake* quantization.

    Bit-for-bit identical to SOTA ``BitLinear.quant_input``
    (vllm_repo/BitNet/gpu/model.py:79): all arithmetic in the input
    dtype (bf16), no fp32 promotion.  ``@torch.compile`` matches SOTA's
    wrapper and fuses absmax + scale + round + clamp + divide into one
    Triton kernel, avoiding 4-5 separate eager-mode kernel launches.
    """
    s = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * s).round().clamp_(-128, 127) / s


def _broadcast_scale_(slice_: torch.Tensor, loaded: torch.Tensor) -> None:
    """Fill ``slice_`` (1D) with the scalar value carried by ``loaded``.

    HF stores ``weight_scale`` as a single-element bf16 tensor.  We
    broadcast that scalar across every output row of the corresponding
    shard so the GEMM kernel can apply per-row dequantization in one pass
    (uniform for non-merged ``BitLinear``; per-shard for the merged form).
    """
    slice_.fill_(float(loaded.detach().to(torch.float32).flatten()[0]))


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value, persistent=False)


class BitLinear(nn.Module):
    """W1.58A8 linear layer (single projection)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        assert in_features % VALUES_PER_BYTE == 0, (
            f"BitLinear in_features={in_features} must be divisible by "
            f"{VALUES_PER_BYTE} for KN-packed ternary weights"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.scale_dtype = dtype

        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // VALUES_PER_BYTE,
                        dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self.weight_scale = nn.Parameter(
            torch.ones(out_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.weight_scale.weight_loader = self._scale_loader
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.bias = None

    # -- weight loading ----------------------------------------------------
    def _weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        # HF stores packed ternary as uint8 (out//4, in); occasionally it
        # arrives as a master-weight bf16 (out, in) tensor in {-1, 0, +1}.
        if loaded.dtype == torch.uint8 and loaded.shape == (
            self.out_features // VALUES_PER_BYTE, self.in_features
        ):
            kn = hf_packed_to_kn_packed(loaded.to(param.device))
        elif loaded.dtype == torch.uint8 and loaded.shape == (
            self.out_features, self.in_features // VALUES_PER_BYTE
        ):
            # Already KN-packed (e.g. another kb_nano checkpoint).
            kn = loaded.to(param.device)
        else:
            # Treat as bf16 master weights with values already in {-1, 0, +1}.
            from .bitnet_int8xint2_linear import repack_ternary_kn
            mw = loaded.to(param.device).round().clamp_(-1, 1).to(torch.int8)
            kn = repack_ternary_kn(mw)
        param.data.copy_(kn)

    def _scale_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if loaded.numel() == 1:
            _broadcast_scale_(param.data, loaded)
        elif loaded.numel() == param.data.numel():
            param.data.copy_(loaded.to(param.dtype))
        else:
            raise ValueError(
                f"BitLinear weight_scale shape mismatch: got {tuple(loaded.shape)}, "
                f"expected scalar or {tuple(param.shape)}"
            )

    # -- post-load: derive bf16 fake-quant weight (SOTA prefill path) -----
    def process_weights_after_loading(self) -> None:
        if getattr(self, "bf16_weight", None) is not None:
            return
        # Always materialize in the LIVE param dtype: load_model casts the
        # whole module to its target dtype (bf16 by default) AFTER __init__
        # captured ``scale_dtype``, so reading from ``weight_scale.dtype``
        # avoids producing a stray fp32 buffer.
        out_dtype = self.weight_scale.dtype
        ternary = unpack_kn_to_ternary(self.weight.data)
        bf16 = ternary.to(out_dtype) * self.weight_scale.data.unsqueeze(1)
        self.register_buffer("bf16_weight", bf16.contiguous(), persistent=False)
        self.set_official_decode_buffers(ternary=ternary)

    def set_official_decode_buffers(
        self,
        ternary: torch.Tensor | None = None,
        scale_values: Sequence[torch.Tensor] | None = None,
    ) -> None:
        if not bitnet_official_kernel_available():
            return
        if ternary is None:
            ternary = unpack_kn_to_ternary(self.weight.data)
        if scale_values is None:
            scale_values = [self.weight_scale.data.flatten()[0]]
        scale = torch.zeros(4, dtype=self.weight_scale.dtype,
                            device=self.weight_scale.device)
        scale[0] = scale_values[0]
        _set_buffer(self, "official_weight",
                    pack_ternary_sota_ladder(ternary).contiguous())
        _set_buffer(self, "official_weight_scale", scale.contiguous())

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = 1
        for d in x.shape[:-1]:
            M *= d
        if _bitnet_use_bf16_path(M) and getattr(self, "bf16_weight", None) is not None:
            return F.linear(_fake_quant_act_bf16(x), self.bf16_weight, self.bias)
        official_weight = getattr(self, "official_weight", None)
        official_scale = getattr(self, "official_weight_scale", None)
        if M == 1 and self.bias is None and official_weight is not None:
            return bitnet_int8xint2_linear_official(
                x, official_weight, official_scale,
            )
        return bitnet_int8xint2_linear(x, self.weight, self.weight_scale, self.bias)


class BitLinearMerged(nn.Module):
    """W1.58A8 linear with multiple shards fused along the output dim.

    Mirrors ``QKVParallelLinear`` / ``MergedColumnParallelLinear`` from
    kb_nano's L2 parallel-linear primitives, except specialized to BitNet's
    int8-x-int2 GEMM and *without* tensor-parallel sharding (BitNet 2B is TP=1
    by design - the model is small enough to fit on a single GPU).
    """

    def __init__(self, in_features: int, out_sizes: Sequence[int],
                 shard_id_map: Mapping | None = None,
                 bias: bool = False, device=None, dtype=None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        assert in_features % VALUES_PER_BYTE == 0
        assert all(s % VALUES_PER_BYTE == 0 for s in out_sizes), (
            f"BitLinearMerged shard sizes {out_sizes} must each be divisible "
            f"by {VALUES_PER_BYTE}"
        )
        self.in_features = in_features
        self.out_sizes = list(out_sizes)
        self.total_out = sum(self.out_sizes)
        self.shard_offsets = [0]
        for s in self.out_sizes:
            self.shard_offsets.append(self.shard_offsets[-1] + s)
        self.scale_dtype = dtype

        self.weight = nn.Parameter(
            torch.zeros(self.total_out, in_features // VALUES_PER_BYTE,
                        dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self.weight_scale = nn.Parameter(
            torch.ones(self.total_out, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.weight_scale.weight_loader = self._scale_loader
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.total_out, dtype=dtype, device=device),
                requires_grad=False,
            )
            self.bias.weight_loader = self._bias_loader
        else:
            self.bias = None

        # ``shard_id_map`` lets callers route HF shard names like ``"q"``,
        # ``"k"``, ``"v"`` to integer slot indices.  Unmapped integer ids
        # are passed through unchanged (used by gate_up_proj which natively
        # carries 0/1 from packed_modules_mapping).
        self._shard_map = dict(shard_id_map) if shard_id_map else {}

    # -- helpers -----------------------------------------------------------
    def _shard_index(self, shard_id) -> int:
        if shard_id in self._shard_map:
            return int(self._shard_map[shard_id])
        if isinstance(shard_id, int):
            return shard_id
        raise KeyError(
            f"BitLinearMerged: unknown shard_id {shard_id!r}; expected one of "
            f"{list(self._shard_map.keys()) or 'integer index'}"
        )

    def _shard_slice(self, idx: int) -> slice:
        start = self.shard_offsets[idx]
        end = self.shard_offsets[idx + 1]
        return slice(start, end)

    # -- weight loading ----------------------------------------------------
    def _weight_loader(self, param: nn.Parameter, loaded: torch.Tensor,
                       shard_id) -> None:
        idx = self._shard_index(shard_id)
        sz = self.out_sizes[idx]
        sl = self._shard_slice(idx)
        if loaded.dtype == torch.uint8 and loaded.shape == (
            sz // VALUES_PER_BYTE, self.in_features
        ):
            kn = hf_packed_to_kn_packed(loaded.to(param.device))
        elif loaded.dtype == torch.uint8 and loaded.shape == (
            sz, self.in_features // VALUES_PER_BYTE
        ):
            kn = loaded.to(param.device)
        else:
            from .bitnet_int8xint2_linear import repack_ternary_kn
            mw = loaded.to(param.device).round().clamp_(-1, 1).to(torch.int8)
            kn = repack_ternary_kn(mw)
        param.data[sl].copy_(kn)

    def _scale_loader(self, param: nn.Parameter, loaded: torch.Tensor,
                      shard_id) -> None:
        idx = self._shard_index(shard_id)
        sl = self._shard_slice(idx)
        if loaded.numel() == 1:
            _broadcast_scale_(param.data[sl], loaded)
        elif loaded.numel() == self.out_sizes[idx]:
            param.data[sl].copy_(loaded.to(param.dtype))
        else:
            raise ValueError(
                f"BitLinearMerged weight_scale shape mismatch for shard "
                f"{shard_id}: got {tuple(loaded.shape)}, expected scalar "
                f"or {(self.out_sizes[idx],)}"
            )

    def _bias_loader(self, param: nn.Parameter, loaded: torch.Tensor,
                     shard_id) -> None:
        idx = self._shard_index(shard_id)
        sl = self._shard_slice(idx)
        param.data[sl].copy_(loaded.to(param.dtype))

    # -- post-load: derive bf16 fake-quant weight (SOTA prefill path) -----
    def process_weights_after_loading(self) -> None:
        if getattr(self, "bf16_weight", None) is not None:
            return
        # Always materialize in the LIVE param dtype: load_model casts the
        # whole module to its target dtype (bf16 by default) AFTER __init__
        # captured ``scale_dtype``, so reading from ``weight_scale.dtype``
        # avoids producing a stray fp32 buffer.
        out_dtype = self.weight_scale.dtype
        ternary = unpack_kn_to_ternary(self.weight.data)
        bf16 = ternary.to(out_dtype) * self.weight_scale.data.unsqueeze(1)
        self.register_buffer("bf16_weight", bf16.contiguous(), persistent=False)
        scale_values = [
            self.weight_scale.data[self.shard_offsets[i]]
            for i in range(len(self.out_sizes))
        ]
        self.set_official_decode_buffers(
            ternary=ternary, scale_values=scale_values,
        )

    def set_official_decode_buffers(
        self,
        ternary: torch.Tensor | None = None,
        scale_values: Sequence[torch.Tensor] | None = None,
    ) -> None:
        if not bitnet_official_kernel_available():
            return
        if ternary is None:
            ternary = unpack_kn_to_ternary(self.weight.data)
        if scale_values is None:
            scale_values = [
                self.weight_scale.data[self.shard_offsets[i]]
                for i in range(len(self.out_sizes))
            ]
        scale = torch.zeros(4, dtype=self.weight_scale.dtype,
                            device=self.weight_scale.device)
        for i, value in enumerate(scale_values[:4]):
            scale[i] = value
        _set_buffer(self, "official_weight",
                    pack_ternary_sota_ladder(ternary).contiguous())
        _set_buffer(self, "official_weight_scale", scale.contiguous())

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = 1
        for d in x.shape[:-1]:
            M *= d
        if _bitnet_use_bf16_path(M) and getattr(self, "bf16_weight", None) is not None:
            return F.linear(_fake_quant_act_bf16(x), self.bf16_weight, self.bias)
        official_weight = getattr(self, "official_weight", None)
        official_scale = getattr(self, "official_weight_scale", None)
        if M == 1 and self.bias is None and official_weight is not None:
            return bitnet_int8xint2_linear_official(
                x, official_weight, official_scale,
            )
        return bitnet_int8xint2_linear(x, self.weight, self.weight_scale, self.bias)
