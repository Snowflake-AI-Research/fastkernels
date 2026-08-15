#!/usr/bin/env python3
"""Amalgamate the L1 ``csrc/`` tree into one self-contained ``<op>.cu`` per op.

Historically all L1 CUDA kernels compiled into a single ``fastkernels_L1_ops``
extension, which forced a shared tree of ~30 vendored vLLM headers. With per-op
JIT (see ``infra/cuda_ext.py``) each op is its own ``.so``, so we can inline the
headers each op needs directly into its sidecar and drop the shared tree.

This script is the generator: for each op it computes the transitive set of
local (``#include "..."``) headers its sources pull in, concatenates each header
exactly once in dependency order (local includes stripped, ``#pragma once``
removed, classic include guards kept), then appends the kernel source bodies and
a per-op ``PYBIND11_MODULE`` exporting only that op's symbols.

The generated code is byte-for-byte the vendored kernel code -- no hand edits --
so re-running this after re-vendoring vLLM regenerates the sidecars. System /
torch / cutlass includes (``#include <...>`` and unresolved quote includes such
as ``"cutlass/cutlass.h"``) are preserved verbatim.

Usage:
    python -m fastkernels.utils.amalgamate_csrc            # write all ops
    python -m fastkernels.utils.amalgamate_csrc moe_sum    # write one op
    python -m fastkernels.utils.amalgamate_csrc --check    # dry run (no write)
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_L1 = os.path.normpath(os.path.join(_HERE, "..", "tasks", "baseline", "L1"))
_CSRC = os.path.join(_L1, "csrc")

_QUOTE_INC = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"[ \t]*\r?$', re.M)
_PRAGMA_ONCE = re.compile(r'^[ \t]*#[ \t]*pragma[ \t]+once[ \t]*\r?$', re.M)
_HEADER_EXTS = (".h", ".cuh", ".hpp")


def _rel(path: str) -> str:
    return os.path.relpath(path, _CSRC)


def _resolve(cur_rel: str, inc: str) -> str | None:
    """Resolve a quote-include the way the old build did: dir-of-file first,
    then the ``-I vllm_port`` root. Returns a path relative to csrc, or None if
    it points outside the tree (system / torch / cutlass -> keep verbatim)."""
    cur_dir = os.path.dirname(os.path.join(_CSRC, cur_rel))
    for cand in (
        os.path.normpath(os.path.join(cur_dir, inc)),
        os.path.normpath(os.path.join(_CSRC, "vllm_port", inc)),
    ):
        if os.path.isfile(cand) and os.path.commonpath([cand, _CSRC]) == _CSRC:
            return os.path.relpath(cand, _CSRC)
    return None


_LINE_INC = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"[ \t]*$')
_LINE_PRAGMA_ONCE = re.compile(r'^[ \t]*#[ \t]*pragma[ \t]+once[ \t]*$')


# Macros that are *never* defined in fastkernels' classic ``_C`` JIT build, so
# their conditionals can be resolved at amalgamation time. USE_ROCM: we only
# target CUDA. TORCH_TARGET_VERSION: the stable-libtorch ABI target; the ``_C``
# path uses plain ``torch::Tensor``/ATen instead. Pruning these is required (not
# just cosmetic): e.g. ``common.cuh`` includes ``torch_utils.h`` only under
# ``#ifdef TORCH_TARGET_VERSION`` -- leaving it opaque would let include-once
# inline that header into a dead branch and skip the live include elsewhere.
_UNDEF_MACROS = ("USE_ROCM", "TORCH_TARGET_VERSION")


def _dead_cond(kind: str, rest: str) -> bool | None:
    """For a conditional controlled solely by an always-undefined macro, return
    whether its first branch is taken. None if we don't evaluate it (opaque)."""
    r = rest.replace(" ", "")
    for macro in _UNDEF_MACROS:
        if kind == "ifdef" and r == macro:
            return False
        if kind == "ifndef" and r == macro:
            return True
        if kind == "if":
            if r in (f"defined({macro})", macro, f"defined{macro}"):
                return False
            if r == f"!defined({macro})":
                return True
    return None


def _strip_rocm(text: str) -> str:
    """Drop branches controlled by always-undefined macros (see ``_UNDEF_MACROS``).
    Only pure single-macro conditionals are evaluated; every other ``#if`` is
    passed through untouched (its dead bodies stay dead at compile time). This
    keeps HIP/AMD and stable-ABI code paths out of the generated sources."""
    out: list[str] = []
    stack: list[dict] = []  # each: {rocm: bool, emit: bool, seen_true: bool}

    def emitting() -> bool:
        return all(f["emit"] for f in stack if f["rocm"])

    for line in text.split("\n"):
        s = line.strip()
        m = re.match(r"#\s*(ifdef|ifndef|if)\b(.*)$", s)
        if m:
            cond = _dead_cond(m.group(1), m.group(2).strip())
            if cond is not None:
                stack.append({"rocm": True, "emit": cond, "seen_true": cond})
            else:
                stack.append({"rocm": False, "emit": True, "seen_true": False})
                if emitting():
                    out.append(line)
            continue
        if re.match(r"#\s*elif\b", s) and stack:
            if stack[-1]["rocm"]:
                stack[-1]["emit"] = False
            elif emitting():
                out.append(line)
            continue
        if re.match(r"#\s*else\b", s) and stack:
            top = stack[-1]
            if top["rocm"]:
                top["emit"] = not top["seen_true"]
            elif emitting():
                out.append(line)
            continue
        if re.match(r"#\s*endif\b", s) and stack:
            top = stack.pop()
            if not top["rocm"] and emitting():
                out.append(line)
            continue
        if emitting():
            out.append(line)
    return "\n".join(out)


def _expand(text: str, cur_rel: str, expanded: set[str]) -> str:
    """Textual, preprocessor-faithful expansion of local quote-includes.

    Each ``#include "local"`` is replaced *in place* by the header's expanded
    contents (so surrounding ``#if``/``#else`` guards still gate it -- e.g. an
    AMD-only header behind ``#else``(USE_ROCM) stays dead on CUDA). A header is
    inlined at most once per op (include-once, like a guard). System / torch /
    cutlass includes (angle-bracket, or quote-includes that resolve outside the
    tree) are kept verbatim. ``#pragma once`` lines are dropped. ``USE_ROCM``
    branches are pruned first so no HIP/AMD code is inlined."""
    text = _strip_rocm(text)
    out: list[str] = []
    for line in text.split("\n"):
        m = _LINE_INC.match(line)
        if m:
            r = _resolve(cur_rel, m.group(1))
            if r is not None:
                if r in expanded:
                    continue
                expanded.add(r)
                htext = open(os.path.join(_CSRC, r), errors="ignore").read()
                out.append(f"// <<< inline {r}")
                out.append(_expand(htext, r, expanded))
                out.append(f"// >>> end {r}")
                continue
        if _LINE_PRAGMA_ONCE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def _norm_key(seg: str) -> str:
    """Whitespace/comment-insensitive key for a top-level chunk."""
    seg = re.sub(r'/\*.*?\*/', '', seg, flags=re.S)
    seg = re.sub(r'//[^\n]*', '', seg)
    return re.sub(r'\s+', ' ', seg).strip()


def _dedup_top_level_defs(text: str) -> str:
    """Remove byte-identical top-level braced definitions after the first.

    Concatenating sibling vendored ``.cu`` files (e.g. the dsv3 router GEMM
    float/bf16 variants) can duplicate identical file-local ``__device__``
    helpers. Two identical definitions at namespace scope are an ODR violation,
    so dropping later exact duplicates is always safe. Only depth-0 braced blocks
    are considered; strings/char-literals/comments/preprocessor lines are skipped
    while scanning so their braces never affect depth."""
    n = len(text)
    i = 0
    depth = 0
    stmt_start = 0
    seen: set[str] = set()
    remove: list[tuple[int, int]] = []

    def at_line_start(idx: int) -> bool:
        j = idx - 1
        while j >= 0 and text[j] in " \t":
            j -= 1
        return j < 0 or text[j] == "\n"

    while i < n:
        c = text[i]
        if c == "#" and at_line_start(i):
            # Consume the whole preprocessor logical line without scanning its
            # content for braces/terminators. Do NOT treat it as a statement
            # boundary: a directive such as ``#pragma unroll`` can sit *inside* a
            # function body, and resetting stmt_start there would split the
            # definition and corrupt dedup removal.
            j = i
            while j < n:
                if text[j] == "\n" and not (j > 0 and text[j - 1] == "\\"):
                    break
                j += 1
            i = j + 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "\"'":
            q = c
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == q:
                    break
                j += 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            i += 1
            if depth == 0:
                seg = text[stmt_start:i]
                key = _norm_key(seg)
                if key and "{" in seg:
                    if key in seen:
                        remove.append((stmt_start, i))
                    else:
                        seen.add(key)
                stmt_start = i
            continue
        if c == ";" and depth == 0:
            i += 1
            stmt_start = i
            continue
        i += 1

    for start, end in reversed(remove):
        text = text[:start] + text[end:]
    return text


def amalgamate(op: "Op") -> str:
    out: list[str] = []
    out.append("// GENERATED by fastkernels/utils/amalgamate_csrc.py -- do not edit by hand.")
    out.append(f"// Self-contained CUDA source for the {op.name!r} L1 op: vendored")
    out.append("// headers expanded in place (include-once), then the pybind module.")
    out.append("")

    expanded: set[str] = set()
    parts: list[str] = []
    for src in op.sources:
        body = open(os.path.join(_CSRC, src), errors="ignore").read()
        # NVFP4 kernels use e2m1x2, which ptxas rejects on sm_90a. Keep the
        # body out of Hopper compiles; dispatch is already ifdef'd the same way.
        wrap_nvfp4 = src.endswith("nvfp4_kv_cache_kernels.cu")
        if wrap_nvfp4:
            parts.append("#if defined(ENABLE_NVFP4_SM100) || defined(ENABLE_NVFP4_SM120)")
        parts.append(f"// ===== source: {src} =====")
        parts.append(_expand(body, src, expanded))
        if wrap_nvfp4:
            parts.append("#endif  // ENABLE_NVFP4_SM100 || ENABLE_NVFP4_SM120")
    merged = "\n".join(parts)
    if len(op.sources) > 1:
        merged = _dedup_top_level_defs(merged)
    out.append(merged.rstrip())
    out.append("")

    if op.pybind is not None:
        out.append("// ===== pybind module (per-op) =====")
        out.append("#include <torch/extension.h>")
        out.append(op.pybind.strip())
        out.append("")

    return "\n".join(out) + "\n"


class Op:
    def __init__(self, name: str, sources: list[str], pybind: str | None):
        self.name = name
        self.sources = sources
        self.pybind = pybind


def _mod(defs: str) -> str:
    return "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n" + defs + "\n}"


P = "vllm_port/libtorch_stable"

OPS: list[Op] = [
    Op("moe_sum.cu", ["moe_sum.cu"], _mod(
        '  m.def("moe_sum", &moe_sum, "MoE sum reduction (CUDA)");')),
    Op("moe_align.cu", ["moe_align.cu"], _mod(
        '  m.def("moe_align_block_size", &moe_align_block_size, "MoE align block size (CUDA)");')),
    Op("topk_softmax.cu", ["moe_topk_softmax.cu"], _mod(
        '  m.def("topk_softmax", &topk_softmax, "Top-K softmax for MoE (CUDA)");')),
    Op("eagle_tree_ops.cu", ["eagle_utils.cu"], _mod(
        '  m.def("build_tree_kernel_efficient", &build_tree_kernel_efficient, "EAGLE build tree kernel efficient (CUDA)");\n'
        '  m.def("build_tree_kernel_efficient_with_metadata", &build_tree_kernel_efficient_with_metadata, "EAGLE build tree and FA3 metadata kernel efficient (CUDA)");\n'
        '  m.def("verify_tree_greedy", &verify_tree_greedy, "EAGLE verify tree greedy (CUDA)");\n'
        '  m.def("build_tree_cascade_metadata", &build_tree_cascade_metadata, "EAGLE build FA3 cascade metadata (CUDA)");')),
    Op("grouped_topk.cu", ["grouped_topk_kernels.cu"], _mod(
        '  m.def("grouped_topk", &grouped_topk, "Fused noaux_tc grouped top-k for MoE routing (CUDA)");')),
    Op("gate_linear.cu", [
        "dsv3_router_gemm_entry.cu", "dsv3_router_gemm_float_out.cu",
        "dsv3_router_gemm_bf16_out.cu", "router_gemm_bf16_fp32.cu"], _mod(
        '  m.def("dsv3_router_gemm", &dsv3_router_gemm, "DeepSeek-V3 router GEMM (SM90+, BF16->{FP32,BF16}) (CUDA)");\n'
        '  m.def("router_gemm_bf16_fp32", &router_gemm_bf16_fp32, "cuBLAS BF16xBF16->FP32 router GEMM fallback (CUDA)");')),
    Op("rms_norm.cu", ["rmsnorm.cu", f"{P}/layernorm_kernels.cu"], _mod(
        '  m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA, local)");\n'
        '  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add + RMSNorm (CUDA, local)");\n'
        '  m.def("rms_norm", &rms_norm, "vLLM RMSNorm (CUDA)");\n'
        '  m.def("fused_add_rms_norm", &fused_add_rms_norm, "vLLM fused add+RMSNorm (CUDA)");')),
    Op("silu_and_mul.cu", [f"{P}/activation_kernels.cu"], _mod(
        '  m.def("silu_and_mul", &silu_and_mul, "vLLM silu_and_mul (CUDA)");')),
    Op("gelu_and_mul.cu", [f"{P}/activation_kernels.cu"], _mod(
        '  m.def("gelu_and_mul", &gelu_and_mul, "vLLM gelu_and_mul (CUDA)");\n'
        '  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "vLLM gelu_tanh_and_mul (CUDA)");')),
    Op("rotary_emb.cu", [f"{P}/pos_encoding_kernels.cu"],
        "static void rotary_embedding_py(\n"
        "    torch::Tensor& positions, torch::Tensor& query,\n"
        "    std::optional<torch::Tensor> key, int64_t head_size,\n"
        "    torch::Tensor& cos_sin_cache, bool is_neox,\n"
        "    int64_t rope_dim_offset = 0, bool inverse = false) {\n"
        "  rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox,\n"
        "                   rope_dim_offset, inverse);\n"
        "}\n\n"
        + _mod(
        '  m.def("rotary_embedding", &rotary_embedding_py, "vLLM rotary_embedding (CUDA)",\n'
        '        py::arg("positions"), py::arg("query"), py::arg("key"),\n'
        '        py::arg("head_size"), py::arg("cos_sin_cache"), py::arg("is_neox"),\n'
        '        py::arg("rope_dim_offset") = 0, py::arg("inverse") = false);')),
    Op("fp8_linear.cu", [
        f"{P}/quantization/w8a8/fp8/common.cu",
        f"{P}/quantization/w8a8/fp8/per_token_group_quant.cu"],
        "static void per_token_group_fp8_quant_py(\n"
        "    const torch::Tensor& input, torch::Tensor& output_q,\n"
        "    torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min,\n"
        "    double fp8_max, bool scale_ue8m0, bool is_scale_transposed = false,\n"
        "    bool is_tma_aligned = false) {\n"
        "  per_token_group_quant_fp8(input, output_q, output_s, group_size, eps, fp8_min,\n"
        "                            fp8_max, scale_ue8m0, is_scale_transposed,\n"
        "                            is_tma_aligned);\n"
        "}\n\n"
        "static void static_scaled_fp8_quant_py(\n"
        "    torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale,\n"
        "    std::optional<at::IntArrayRef> opt_group_shape = std::nullopt) {\n"
        "  static_scaled_fp8_quant(out, input, scale, opt_group_shape);\n"
        "}\n\n"
        + _mod(
        '  m.def("per_token_group_fp8_quant", &per_token_group_fp8_quant_py,\n'
        '        "vLLM per_token_group_fp8_quant (CUDA)",\n'
        '        py::arg("input"), py::arg("output_q"), py::arg("output_s"),\n'
        '        py::arg("group_size"), py::arg("eps"), py::arg("fp8_min"),\n'
        '        py::arg("fp8_max"), py::arg("scale_ue8m0"),\n'
        '        py::arg("is_scale_transposed") = false,\n'
        '        py::arg("is_tma_aligned") = false);\n'
        '  m.def("static_scaled_fp8_quant", &static_scaled_fp8_quant_py,\n'
        '        "vLLM static_scaled_fp8_quant (CUDA)",\n'
        '        py::arg("out"), py::arg("input"), py::arg("scale"),\n'
        '        py::arg("group_shape") = py::none());')),
    Op("trtllm_fp4_moe.cu", [
        f"{P}/quantization/fp4/nvfp4_quant_kernels.cu",
        f"{P}/quantization/fp4/nvfp4_quant_entry.cu",
        f"{P}/cutlass_extensions/common.cpp",
        f"{P}/cuda_utils_kernels.cu"], _mod(
        '  m.def("scaled_fp4_quant_out", &scaled_fp4_quant_out, "vLLM scaled_fp4_quant.out (CUDA)");')),
    Op("top_k_per_row.cu", [
        f"{P}/sampler.cu", f"{P}/topk.cu", f"{P}/cooperative_topk.cu"], _mod(
        '  m.def("top_k_per_row_prefill", &top_k_per_row_prefill, "vLLM top_k_per_row_prefill (CUDA)");\n'
        '  m.def("top_k_per_row_decode", &top_k_per_row_decode, "vLLM top_k_per_row_decode (CUDA)");\n'
        '  m.def("persistent_topk", &persistent_topk, "vLLM persistent_topk (CUDA)");\n'
        '  m.def("cooperative_topk", &cooperative_topk, "vLLM cooperative_topk (CUDA)");')),
    Op("store_kvcache_fp8_mla.cu", [
        f"{P}/cache_kernels.cu", f"{P}/nvfp4_kv_cache_kernels.cu"], _mod(
        '  m.def("concat_and_cache_mla", &concat_and_cache_mla, "vLLM concat_and_cache_mla (CUDA)");\n'
        '  m.def("gather_and_maybe_dequant_cache", &gather_and_maybe_dequant_cache, "vLLM gather_and_maybe_dequant_cache (CUDA)");\n'
        '  m.def("cp_gather_and_upconvert_fp8_kv_cache", &cp_gather_and_upconvert_fp8_kv_cache, "vLLM cp_gather_and_upconvert_fp8_kv_cache (CUDA)");')),
    Op("indexer_k_cache.cu", [
        f"{P}/cache_kernels.cu", f"{P}/nvfp4_kv_cache_kernels.cu"], _mod(
        '  m.def("indexer_k_quant_and_cache", &indexer_k_quant_and_cache, "vLLM indexer_k_quant_and_cache (CUDA)");\n'
        '  m.def("cp_gather_indexer_k_quant_cache", &cp_gather_indexer_k_quant_cache, "vLLM cp_gather_indexer_k_quant_cache (CUDA)");')),
    Op("merge_attn_states.cu", [f"{P}/attention/merge_attn_states.cu"], _mod(
        '  m.def("merge_attn_states", &merge_attn_states,\n'
        '        "vLLM merge_attn_states (CUDA)", py::arg("output"),\n'
        '        py::arg("output_lse"), py::arg("prefix_output"), py::arg("prefix_lse"),\n'
        '        py::arg("suffix_output"), py::arg("suffix_lse"),\n'
        '        py::arg("prefill_tokens_with_context"), py::arg("output_scale"));')),
    Op("mamba_ssm.cu", [f"{P}/mamba/selective_scan_fwd.cu"], _mod(
        '  m.def("selective_scan_fwd", &selective_scan_fwd, "vLLM Mamba selective_scan_fwd (CUDA)");')),
    # allreduce already ships a self-contained .cu with its own PYBIND11_MODULE
    # and zero local headers; just relocate it verbatim.
    Op("allreduce.cu", ["custom_allreduce_kernels.cu"], None),
]


def main(argv: list[str]) -> int:
    check = "--check" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    for op in OPS:
        stem = op.name[:-3]
        if wanted and stem not in wanted and op.name not in wanted:
            continue
        text = amalgamate(op)
        dest = os.path.join(_L1, op.name)
        n = len(text.encode())
        if check:
            print(f"[check] {op.name}: {n} bytes, {len(op.sources)} source(s)")
            continue
        with open(dest, "w") as f:
            f.write(text)
        print(f"[write] {dest} ({n} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
