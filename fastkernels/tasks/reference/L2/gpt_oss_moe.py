"""Packed MXFP4 Triton reference for gpt_oss_moe.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

GPT-OSS production weights are packed MXFP4. This reference keeps that packed
format and delegates routing/fused expert matmuls to the L1 ``mxfp4_moe``
Triton reference instead of dequantizing expert weights in Python.
"""


from __future__ import annotations


# Inlined from infra/tp.py
import torch.distributed as dist


def _tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def _tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# Inlined from tasks/reference/L1/linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


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


# Inlined from tasks/reference/L1/mxfp4_moe.py
from dataclasses import dataclass
from typing import Any


@dataclass
class Mxfp4MoEQuantConfig:
    w1_precision: Any
    w2_precision: Any
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None


_FP4_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _dequant_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    lut = _FP4_E2M1_LUT.to(blocks.device)
    low = (blocks & 0x0F).long()
    high = ((blocks >> 4) & 0x0F).long()
    unpacked = torch.stack([low, high], dim=-1).reshape(*blocks.shape[:-1], 32)
    values = lut[unpacked]
    scale_float = torch.pow(2.0, scales.float() - 127.0)
    values = values * scale_float.unsqueeze(-1)
    return values.reshape(*values.shape[:-2], -1).to(dtype)


class Mxfp4MoE(nn.Module):
    """MXFP4-quantized MoE using dense PyTorch matmuls after dequantization."""

    @staticmethod
    def prepare_weight(
        quant_tensor: torch.Tensor,
        scale: torch.Tensor,
        num_warps: int = 8,
    ):
        del num_warps
        return _dequant_mxfp4(quant_tensor, scale, dtype=torch.bfloat16), None

    @staticmethod
    def make_quant_config(
        w1_precision: Any,
        w2_precision: Any,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> Mxfp4MoEQuantConfig:
        return Mxfp4MoEQuantConfig(
            w1_precision=w1_precision,
            w2_precision=w2_precision,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1,
        w2,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        quant_config: Mxfp4MoEQuantConfig,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        scores = torch.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)

        w1_dense = w1.float()
        w2_dense = w2.float()
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        x_all = hidden_states.float()

        for token in range(hidden_states.shape[0]):
            for slot in range(topk):
                expert = int(topk_ids[token, slot].item())
                weight = topk_weights[token, slot].float()
                x = x_all[token]
                if apply_router_weight_on_input:
                    x = x * weight

                bias1 = None
                if quant_config.w1_bias is not None:
                    bias1 = quant_config.w1_bias[expert].float()
                gate_up = F.linear(x, w1_dense[expert], bias1)
                gate = gate_up[0::2]
                up = gate_up[1::2]
                gate = gate.clamp(max=7.0)
                up = up.clamp(min=-7.0, max=7.0)
                hidden = (up + 1.0) * gate * torch.sigmoid(1.702 * gate)

                bias2 = None
                if quant_config.w2_bias is not None:
                    bias2 = quant_config.w2_bias[expert].float()
                y = F.linear(hidden, w2_dense[expert], bias2)
                if not apply_router_weight_on_input:
                    y = y * weight
                output[token] += y

        return output.to(hidden_states.dtype)


__all__ = ["Mxfp4MoE", "Mxfp4MoEQuantConfig"]


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class GptOssMoE(nn.Module):
    MXFP4_BLOCK = 32

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp
        self.router = Linear(config.hidden_size, config.num_local_experts, bias=True)
        e = config.num_local_experts
        i_pad = _round_up(self.intermediate_per_tp, 64)
        h = config.hidden_size
        blk = self.MXFP4_BLOCK
        self._I_pad = i_pad
        self.w13_weight = nn.Parameter(
            torch.zeros(e, 2 * i_pad, h // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(e, 2 * i_pad, h // blk, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_bias = nn.Parameter(
            torch.zeros(e, 2 * i_pad, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(e, h, i_pad // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(e, h, i_pad // blk, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_bias = nn.Parameter(
            torch.zeros(e, h, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w13_weight.weight_loader = self._w13_weight_loader
        self.w13_weight_scale.weight_loader = self._w13_scale_loader
        self.w13_bias.weight_loader = self._w13_bias_loader
        self.w2_weight.weight_loader = self._w2_weight_loader
        self.w2_weight_scale.weight_loader = self._w2_scale_loader
        self.w2_bias.weight_loader = self._w2_bias_loader
        self.allreduce = AllReduce()
        self.mxfp4_moe = Mxfp4MoE()
        self._quant_config = None
        self._processed = False
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight):
        if loaded_weight.ndim == 4:
            e, n, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(e, n, nb * bs)
        rank = _tp_rank()
        i = self.intermediate_per_tp
        start = 2 * rank * i
        param.data[:, :2 * i, :].copy_(loaded_weight[:, start:start + 2 * i, :])

    def _w13_scale_loader(self, param, loaded_weight):
        rank = _tp_rank()
        i = self.intermediate_per_tp
        start = 2 * rank * i
        param.data[:, :2 * i, :].copy_(loaded_weight[:, start:start + 2 * i, :])

    def _w13_bias_loader(self, param, loaded_weight):
        rank = _tp_rank()
        i = self.intermediate_per_tp
        start = 2 * rank * i
        param.data[:, :2 * i].copy_(loaded_weight[:, start:start + 2 * i])

    def _w2_weight_loader(self, param, loaded_weight):
        if loaded_weight.ndim == 4:
            e, h, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(e, h, nb * bs)
        rank = _tp_rank()
        i_half = self.intermediate_per_tp // 2
        param.data[:, :, :i_half].copy_(
            loaded_weight[:, :, rank * i_half:rank * i_half + i_half],
        )

    def _w2_scale_loader(self, param, loaded_weight):
        rank = _tp_rank()
        i_blk = self.intermediate_per_tp // self.MXFP4_BLOCK
        param.data[:, :, :i_blk].copy_(
            loaded_weight[:, :, rank * i_blk:rank * i_blk + i_blk],
        )

    def _w2_bias_loader(self, param, loaded_weight):
        if _tp_rank() == 0:
            param.data.copy_(loaded_weight)
        else:
            param.data.zero_()

    def process_weights_after_loading(self):
        if self._processed:
            return
        self.w13_bias.data = self.w13_bias.data.float()
        self.w2_bias.data = self.w2_bias.data.float()
        self._w13_swizzled, w13_precision = Mxfp4MoE.prepare_weight(
            self.w13_weight.data, self.w13_weight_scale.data,
        )
        self._w2_swizzled, w2_precision = Mxfp4MoE.prepare_weight(
            self.w2_weight.data, self.w2_weight_scale.data,
        )
        del self.w13_weight, self.w2_weight
        del self.w13_weight_scale, self.w2_weight_scale
        self._quant_config = Mxfp4MoE.make_quant_config(
            w1_precision=w13_precision,
            w2_precision=w2_precision,
            w1_bias=self.w13_bias.data,
            w2_bias=self.w2_bias.data,
        )
        self._processed = True

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            self.process_weights_after_loading()
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = self.router(hidden_states)
        output = self.mxfp4_moe(
            hidden_states=hidden_states,
            w1=self._w13_swizzled,
            w2=self._w2_swizzled,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=True,
            quant_config=self._quant_config,
            apply_router_weight_on_input=False,
        )
        if self.tp_size > 1:
            output = self.allreduce(output)
        return output.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(hidden_states)
