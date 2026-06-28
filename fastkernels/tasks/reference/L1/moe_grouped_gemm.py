"""Semantic PyTorch reference for MoE grouped GEMM."""

from __future__ import annotations

import torch
import torch.nn as nn


def _get_default_config(M: int, E: int = 0, N: int = 0,
                        block_shape: list[int] | None = None) -> dict:
    del M, E, N, block_shape
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 5,
    }


def get_triton_config(M: int, w1_shape: tuple[int, ...], w2_shape: tuple[int, ...],
                      top_k: int, use_fp8: bool,
                      block_shape: list[int] | None = None) -> dict:
    del w1_shape, w2_shape, top_k, use_fp8
    return _get_default_config(M, block_shape=block_shape)


def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    del hidden_states, w1, w2
    return False


def m_grouped_fp8_gemm_nt_contiguous(a_and_scale, b_and_scale, output, expert_ids):
    raise RuntimeError("DeepGEMM is not available in the self-contained reference path")


def _expand_group_scale(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    block_shape: list[int] | None,
) -> torch.Tensor:
    if scale is None:
        return torch.ones_like(x, dtype=torch.float32)
    scale = scale.float()
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand_as(x.float())
    if scale.shape == x.shape:
        return scale
    if scale.ndim == 1 and scale.numel() == x.shape[-1]:
        return scale.view(1, -1).expand_as(x.float())
    if scale.ndim == 1 and scale.numel() == x.shape[0]:
        return scale.view(-1, 1).expand_as(x.float())
    if block_shape is not None and len(block_shape) == 2 and scale.ndim == 2:
        block_n, block_k = block_shape
        return scale.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)[
            : x.shape[0], : x.shape[1]
        ]
    if scale.ndim == 2 and scale.shape[0] == x.shape[0]:
        repeat = (x.shape[1] + scale.shape[1] - 1) // scale.shape[1]
        return scale.repeat_interleave(repeat, dim=1)[:, : x.shape[1]]
    return torch.ones_like(x, dtype=torch.float32) * scale.reshape(-1)[0]


def _dequant_a(
    A: torch.Tensor,
    a_scale: torch.Tensor | None,
    block_shape: list[int] | None,
) -> torch.Tensor:
    A_f = A.float()
    if a_scale is None:
        return A_f
    if block_shape is None and a_scale.ndim == 2 and a_scale.shape[0] == A.shape[0]:
        repeat = (A.shape[1] + a_scale.shape[1] - 1) // a_scale.shape[1]
        scale = a_scale.float().repeat_interleave(repeat, dim=1)[:, : A.shape[1]]
    else:
        scale = _expand_group_scale(A, a_scale, block_shape)
    return A_f * scale


def _dequant_b(
    B_e: torch.Tensor,
    b_scale_e: torch.Tensor | None,
    block_shape: list[int] | None,
) -> torch.Tensor:
    B_f = B_e.float()
    if b_scale_e is None:
        return B_f
    scale = _expand_group_scale(B_e, b_scale_e, block_shape)
    return B_f * scale


class MoeGroupedGemm(nn.Module):
    @staticmethod
    def get_config(M: int, N: int = 0, E: int = 0,
                   use_fp8: bool = False,
                   block_shape: list[int] | None = None) -> dict:
        del N, E, use_fp8
        return _get_default_config(M, block_shape=block_shape)

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        config: dict | None = None,
        a_scale: torch.Tensor | None = None,
        b_scale: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ):
        del num_tokens_post_padded, use_fp8_w8a8
        config = _get_default_config(A.size(0)) if config is None else config
        block_size = int(config.get("BLOCK_SIZE_M", 1))
        valid_tokens = A.size(0) * top_k
        A_deq = _dequant_a(A, a_scale, block_shape)
        flat_weights = topk_weights.reshape(-1).float() if topk_weights is not None else None
        C.zero_()

        if sorted_token_ids is None:
            for row, expert in enumerate(expert_ids.reshape(-1).tolist()):
                flat_id = row
                if flat_id >= valid_tokens or expert < 0:
                    continue
                token = flat_id // top_k
                B_e = _dequant_b(
                    B[int(expert)],
                    b_scale[int(expert)] if b_scale is not None and b_scale.ndim >= 1 else b_scale,
                    block_shape,
                )
                out = torch.matmul(A_deq[token], B_e.t())
                if mul_routed_weight and flat_weights is not None:
                    out = out * flat_weights[flat_id]
                C[flat_id].copy_(out.to(C.dtype))
            return C

        sorted_ids = sorted_token_ids.reshape(-1).to(torch.int64)
        for block, expert in enumerate(expert_ids.reshape(-1).tolist()):
            if expert < 0:
                continue
            start = block * block_size
            end = min(start + block_size, sorted_ids.numel())
            B_e = _dequant_b(
                B[int(expert)],
                b_scale[int(expert)] if b_scale is not None and b_scale.ndim >= 1 else b_scale,
                block_shape,
            )
            for flat_id_t in sorted_ids[start:end]:
                flat_id = int(flat_id_t.item())
                if flat_id >= valid_tokens:
                    continue
                token = flat_id // top_k
                out = torch.matmul(A_deq[token], B_e.t())
                if mul_routed_weight and flat_weights is not None:
                    out = out * flat_weights[flat_id]
                C[flat_id].copy_(out.to(C.dtype))
        return C
