"""Llama SwiGLU MLP block: gate_up_proj -> SiluAndMul -> down_proj.

Also used by DeepSeek V3's "shared expert": pass ``reduce_results=False``
and override ``intermediate_size`` with ``moe_intermediate_size *
n_shared_experts``.  See ``L2/deepseek_moe.py``.
"""


from __future__ import annotations


# Inlined from infra/tp.py
import torch.distributed as dist


def _tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def _tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# Inlined from tasks/reference/L1/fp8_linear.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_GROUP_SIZE = 128


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _expand_weight_scale(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight_fp8.shape[-2], weight_fp8.shape[-1]
    row_blocks = _ceil_div(rows, _GROUP_SIZE)
    col_blocks = _ceil_div(cols, _GROUP_SIZE)
    scale_f = scale.float()
    if scale_f.shape[-2:] == (row_blocks, col_blocks):
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-2)
        expanded = expanded.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :rows, :cols]
    if scale_f.shape[-1] == col_blocks:
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :cols].unsqueeze(-2).expand_as(weight_fp8.float())
    return scale_f.expand_as(weight_fp8.float())


def _quantize_fp8_per_token_group(
    source: torch.Tensor,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
    *,
    use_ue8m0: bool = True,
    eps: float = 1e-10,
) -> None:
    info = torch.finfo(torch.float8_e4m3fn)
    flat = source.reshape(-1, source.shape[-1]).float()
    groups = _ceil_div(flat.shape[-1], _GROUP_SIZE)
    padded_cols = groups * _GROUP_SIZE
    if padded_cols != flat.shape[-1]:
        padded = flat.new_zeros(flat.shape[0], padded_cols)
        padded[:, :flat.shape[-1]] = flat
    else:
        padded = flat
    grouped = padded.view(flat.shape[0], groups, _GROUP_SIZE)
    scale = grouped.abs().amax(dim=-1).clamp_min(eps) / info.max
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    expanded = scale.repeat_interleave(_GROUP_SIZE, dim=-1)[:, :flat.shape[-1]]
    out_fp8.copy_(torch.clamp(flat / expanded, info.min, info.max).to(out_fp8.dtype).view_as(out_fp8))
    out_scale.copy_(scale.view_as(out_scale))


class _Fp8PrefillBufs:
    def __init__(self):
        self.input_fp8 = None
        self.input_scale = None
        self.output = None


class PerTokenGroupQuantFp8(nn.Module):
    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        _quantize_fp8_per_token_group(x, out_fp8, out_scale)


class Fp8Linear(nn.Module):
    BLOCK_SIZE = _GROUP_SIZE
    _FLASHINFER_M_THRESHOLD = 32

    def __init__(self):
        super().__init__()
        self._a_buf = None
        self._s_buf = None
        self._o_buf = None
        self._pf = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = torch.empty(max_tokens, math.ceil(K / _GROUP_SIZE), dtype=torch.float32, device=device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        n, k = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, k)
        weight = weight_fp8.float() * _expand_weight_scale(weight_fp8, weight_scale_inv)
        output = F.linear(input_2d.float(), weight.float(), bias.float() if bias is not None else None)
        return output.to(input_bf16.dtype).view(*input_bf16.shape[:-1], n)


def postprocess_fp8_weights(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return weight_fp8, scale_inv


# Inlined from tasks/reference/L1/allreduce.py
from contextlib import nullcontext
from typing import Optional

from torch.distributed import ProcessGroup


_CUSTOM_AR: Optional["CustomAllreduce"] = None


def set_custom_ar(ar):
    global _CUSTOM_AR
    _CUSTOM_AR = ar


def get_custom_ar():
    return _CUSTOM_AR


class AllReduce(nn.Module):
    def forward(self, tensor):
        dist.all_reduce(tensor)
        return tensor


class CustomAllreduce:
    """Compatibility shim for callers expecting the baseline custom AR API."""

    disabled = True

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
    ) -> None:
        del group, device, max_size

    def capture(self):
        return nullcontext()

    def custom_all_reduce(self, input: torch.Tensor) -> None:
        del input
        return None

    def close(self) -> None:
        pass

__all__ = ["AllReduce", "CustomAllreduce", "get_custom_ar", "set_custom_ar"]


# Inlined from tasks/reference/L2/parallel_linear.py


def _get_fp8_linear_cls():
    return Fp8Linear

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, input_size,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(self.output_size_per_partition, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        rows_per_shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * rows_per_shard, rows_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """gate_proj + up_proj merged into one linear, sharded across TP."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 quant_config: dict | None = None, disable_tp: bool = False):
        super().__init__()
        tp = _tp_size()
        self.disable_tp = disable_tp
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        if not disable_tp:
            assert all(s % tp == 0 for s in output_sizes)
        self.use_fp8 = quant_config is not None

        effective_tp = 1 if disable_tp else tp
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(total // effective_tp, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(total // effective_tp, input_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(total // effective_tp, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(total // tp))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: int | None = None):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id is None:
            # Fused weight: ``loaded_weight`` is the full ``[sum(output_sizes), in]``
            # tensor.  Recurse per-shard so each output block is sharded across
            # TP ranks independently (mirrors vLLM's ``MergedColumnParallelLinear``
            # weight loader when called without an explicit shard id).
            offset = 0
            for sid, sz in enumerate(self.output_sizes):
                self._weight_loader(
                    param, loaded_weight.narrow(0, offset, sz), sid,
                )
                offset += sz
            return
        effective_tp = 1 if self.disable_tp else tp
        shard_offset = sum(self.output_sizes[:shard_id]) // effective_tp
        shard_size = self.output_sizes[shard_id] // effective_tp
        dst = param.data.narrow(0, shard_offset, shard_size)
        if self.disable_tp:
            dst.copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        effective_tp = 1 if self.disable_tp else tp
        shard_size_out = self.output_sizes[shard_id] // effective_tp
        scale_rows = math.ceil(shard_size_out / _FP8_BLOCK)
        shard_offset_out = sum(self.output_sizes[:shard_id]) // effective_tp
        scale_offset = math.ceil(shard_offset_out / _FP8_BLOCK)
        if self.disable_tp:
            param.data.narrow(0, scale_offset, scale_rows).copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged and sharded across TP."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        # Replicate KV heads when not evenly divisible by TP
        if total_num_kv_heads % tp == 0:
            self.num_kv_heads = total_num_kv_heads // tp
            self._replicate_kv = False
        else:
            self.num_kv_heads = total_num_kv_heads
            self._replicate_kv = True
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, hidden_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            src = loaded_weight.chunk(tp, 0)[rank]
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        dst = param.data.narrow(0, shard_offset, shard_size)
        dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        scale_rows = math.ceil(shard_size / _FP8_BLOCK)
        scale_offset = math.ceil(shard_offset / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class ReplicatedLinear(nn.Module):
    """Full weight replicated on every TP rank (no sharding, no all-reduce)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks, all-reduces output."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.reduce_results = reduce_results
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, self.input_size_per_partition,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, self.input_size_per_partition),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        cols_per_shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * cols_per_shard, cols_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            y = self.linear_op(x, self.weight, self.weight_scale_inv,
                               self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_results and self.tp_size > 1:
            y = self.allreduce(y)
        return y


# Inlined from tasks/reference/L1/silu_and_mul.py


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        return SiluAndMul.forward_native(x)

    def forward(self, x):
        return self.forward_native(x)


class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config: dict | None = None,
                 hidden_size: int | None = None,
                 intermediate_size: int | None = None,
                 reduce_results: bool = True):
        super().__init__()
        h = hidden_size if hidden_size is not None else config.hidden_size
        i = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            h, [i] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            i, h,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)
