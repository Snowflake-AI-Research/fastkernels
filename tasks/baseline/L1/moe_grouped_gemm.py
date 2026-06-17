"""Triton fused MoE grouped GEMM kernel with FP8 W8A8 block-scaled support.

Also supports DeepGEMM m_grouped_fp8_gemm_nt_contiguous for Hopper+ GPUs
when shapes are aligned, matching vLLM's TritonOrDeepGemmExperts behavior.

Triton kernel config selection uses per-device JSON tuning files (same format
as vLLM), falling back to hardcoded defaults when no file is found.
"""

from __future__ import annotations

import functools
import json
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import deep_gemm as _dg
    _HAS_DEEP_GEMM = True
except ImportError:
    _dg = None
    _HAS_DEEP_GEMM = False


_BLOCK_ALIGNMENT = 128


def _is_deep_gemm_supported() -> bool:
    if not _HAS_DEEP_GEMM:
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 9


@functools.cache
def _deep_gemm_alignment() -> int:
    if not _HAS_DEEP_GEMM:
        return _BLOCK_ALIGNMENT
    try:
        return _dg.get_mk_alignment_for_contiguous_layout()
    except AttributeError:
        return _BLOCK_ALIGNMENT


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align = _deep_gemm_alignment()
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    if not _is_deep_gemm_supported():
        return False
    M = hidden_states.size(0)
    _, K, N = w2.size()
    if not _valid_deep_gemm_shape(M, N, K):
        return False
    if N <= 512:
        return False
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return False
    if not (hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()):
        return False
    return True


def m_grouped_fp8_gemm_nt_contiguous(a_and_scale, b_and_scale, output, expert_ids):
    """Wrapper for deep_gemm.m_grouped_fp8_gemm_nt_contiguous.

    Args:
        a_and_scale: tuple of (a_fp8, a_scale)
        b_and_scale: tuple of (b_fp8, b_scale)
        output: output buffer
        expert_ids: per-row expert assignment (int32, -1 = skip)
    """
    _dg.m_grouped_fp8_gemm_nt_contiguous(
        a_and_scale, b_and_scale, output, expert_ids,
        disable_ue8m0_cast=True,
    )


@triton.jit
def _fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    NAIVE_BLOCK_ASSIGNMENT: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    if NAIVE_BLOCK_ASSIGNMENT:
        offs_token = tl.where(offs_m == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_expert * stride_bse + offs_bsn * stride_bsn
            )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k * BLOCK_SIZE_K + offs_k) < K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        if use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator = tl.dot(a.to(compute_type), b.to(compute_type), accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if use_fp8_w8a8 and not (group_k > 0 and group_n > 0):
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_expert)
        accumulator = accumulator * a_scale * b_scale

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _get_config_file_name(E: int, N: int, dtype: str | None,
                          block_shape: list[int] | None = None) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"


@functools.lru_cache
def _get_moe_configs(E: int, N: int, dtype: str | None,
                     block_n: int | None = None,
                     block_k: int | None = None) -> dict[int, dict] | None:
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = _get_config_file_name(E, N, dtype, block_shape)

    config_file_paths = []

    user_folder = os.environ.get("KB_NANO_TUNED_CONFIG_FOLDER")
    if user_folder is not None:
        config_file_paths.append(os.path.join(user_folder, json_file_name))

    vllm_configs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",
        "vllm_repo", "vllm", "vllm", "model_executor", "layers",
        "fused_moe", "configs",
    )
    if os.path.isdir(vllm_configs_dir):
        config_file_paths.append(os.path.join(vllm_configs_dir, json_file_name))

    local_configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_configs")
    if os.path.isdir(local_configs_dir):
        config_file_paths.append(os.path.join(local_configs_dir, json_file_name))

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                tuned_config = json.load(f)
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items()}

    return None


_DEFAULT_CONFIG_HEURISTIC = {
    "small": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5,
    },
    "medium": {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3,
    },
    "large": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4,
    },
}


def _get_vllm_default_config(M: int, E: int = 0, dtype: str | None = None) -> dict:
    """vLLM-style BF16/FP16 MoE defaults.

    Gemma4's BF16 top-8 experts are much closer to vLLM's generic MoE path
    than to the older kb-nano heuristic, especially in decode where tokens are
    spread thinly across 128 experts.
    """
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 1024:
        block_m = 64
    else:
        block_m = 128

    block_n = 64 if M <= 64 else 128
    block_k = 128 if dtype == "fp8_w8a8" or M <= 64 else 64
    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1
    num_warps = 4 if M <= 1024 else 8
    num_stages = 4 if M <= 32 else 3

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _get_default_config(M: int, E: int = 0, N: int = 0,
                        block_shape: list[int] | None = None) -> dict:
    if M <= 4:
        return dict(_DEFAULT_CONFIG_HEURISTIC["small"])
    if M <= 64:
        return dict(_DEFAULT_CONFIG_HEURISTIC["medium"])
    return dict(_DEFAULT_CONFIG_HEURISTIC["large"])


def get_triton_config(M: int, w1_shape: tuple[int, ...], w2_shape: tuple[int, ...],
                      top_k: int, use_fp8: bool,
                      block_shape: list[int] | None = None,
                      default_style: str = "legacy") -> dict:
    """Select best Triton kernel config, preferring JSON tuning files."""
    E, _, N = w2_shape
    dtype = "fp8_w8a8" if use_fp8 else None
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0

    configs = _get_moe_configs(E, N, dtype, block_n, block_k)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        return dict(config)

    if default_style == "vllm":
        return _get_vllm_default_config(M, E, dtype)
    if default_style != "legacy":
        raise ValueError(f"Unknown MoE config style: {default_style}")
    return _get_default_config(M, E, N, block_shape)


class MoeGroupedGemm(nn.Module):
    @staticmethod
    def get_config(M: int, N: int = 0, E: int = 0,
                   use_fp8: bool = False,
                   block_shape: list[int] | None = None) -> dict:
        """Select best kernel config based on batch size M and output dim N."""
        if E > 0:
            w2_shape = (E, 0, N // 2 if N > 0 else 0)
            w1_shape = (E, N, 0)
            return get_triton_config(M, w1_shape, w2_shape, 1, use_fp8, block_shape)
        return _get_default_config(M, E, N, block_shape)

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
        if config is None:
            config = _get_default_config(A.size(0), N=B.size(1))
        else:
            config = config.copy()

        naive = sorted_token_ids is None
        if naive:
            EM = expert_ids.numel() * config["BLOCK_SIZE_M"]
        else:
            EM = sorted_token_ids.size(0)
            if A.size(0) < config["BLOCK_SIZE_M"]:
                EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])

        grid = (
            triton.cdiv(EM, config["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), config["BLOCK_SIZE_N"]),
        )

        if use_fp8_w8a8:
            compute_type = tl.bfloat16
        elif A.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif A.dtype == torch.float16:
            compute_type = tl.float16
        else:
            compute_type = tl.float32

        if use_fp8_w8a8 and block_shape is not None:
            group_n, group_k = block_shape[0], block_shape[1]
            config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], min(group_n, group_k))
        else:
            group_n, group_k = 0, 0

        launch_kwargs = {}
        if "num_warps" in config:
            launch_kwargs["num_warps"] = config["num_warps"]
        if "num_stages" in config:
            launch_kwargs["num_stages"] = config["num_stages"]

        sorted_ids_ptr = sorted_token_ids if sorted_token_ids is not None else A
        a_scale_ptr = a_scale if a_scale is not None else A
        b_scale_ptr = b_scale if b_scale is not None else B

        _fused_moe_kernel[grid](
            A, B, C,
            a_scale_ptr, b_scale_ptr,
            topk_weights,
            sorted_ids_ptr,
            expert_ids,
            num_tokens_post_padded,
            B.size(1), B.size(2), EM,
            A.size(0) * top_k,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            C.stride(0), C.stride(1),
            a_scale.stride(0) if a_scale is not None and a_scale.ndim >= 2 else 0,
            a_scale.stride(1) if a_scale is not None and a_scale.ndim >= 2 else 0,
            b_scale.stride(0) if b_scale is not None and b_scale.ndim >= 2 else 0,
            b_scale.stride(2) if b_scale is not None and b_scale.ndim == 3 else 0,
            b_scale.stride(1) if b_scale is not None and b_scale.ndim >= 2 else 0,
            group_n=group_n,
            group_k=group_k,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NAIVE_BLOCK_ASSIGNMENT=naive,
            **launch_kwargs,
        )
