"""Shared CUDA extension loader for L1 baseline kernels.

Compiles and caches the extension once; all task modules import _C from here.
"""

import os

from torch.utils.cpp_extension import load as _load_ext

_DIR = os.path.dirname(os.path.abspath(__file__))

_C = _load_ext(
    name="kb_nano_L1_ops",
    sources=[os.path.join(_DIR, f) for f in [
        "binding.cpp", "rmsnorm.cu", "rmsnorm_quant.cu",
        "activation.cu", "pos_enc.cu",
        "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
        "eagle_utils.cu",
        # DeepSeek-V3 router ops (verbatim port of vLLM csrc/moe sources;
        # see binding.cpp for op-level descriptions).
        "dsv3_router_gemm_entry.cu",
        "dsv3_router_gemm_float_out.cu",
        "dsv3_router_gemm_bf16_out.cu",
        "router_gemm_bf16_fp32.cu",
        "grouped_topk_kernels.cu",
    ]],
    extra_cuda_cflags=["-O3",
                       "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16",
                       # vLLM's CMake unsets these so its noaux_tc grouped-topk
                       # kernel (ported verbatim into ``grouped_topk_kernels.cu``)
                       # can rely on implicit ``half``/``__nv_bfloat16``<->``float``
                       # constructors.  ``torch.utils.cpp_extension`` defines them
                       # by default; we undefine them here to match vLLM.
                       "-U__CUDA_NO_HALF_OPERATORS__",
                       "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)
