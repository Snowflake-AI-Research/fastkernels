"""Semantic PyTorch reference for MXFP4 MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
