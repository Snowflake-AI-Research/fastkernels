"""DeepSeek MoE with shared expert, grouped routing, and FP8 expert execution.

Uses :class:`GroupedTopK` for routing, :class:`VllmFusedExperts` for the
FP8 expert path (a fresh-allocation port of vLLM's ``fused_experts_impl``
that mirrors vLLM's Triton oracle for Hopper + block-FP8 + TP),
:class:`FusedExperts` for the BF16 / unquantized fallback, and a shared
expert (``LlamaMLP`` with ``reduce_results=False``) that runs on a
separate CUDA stream for overlap.

Matches vllm's DeepseekV2MoE: routed_scaling_factor is applied post-experts
(not folded into routing weights), and the shared expert uses
``moe_intermediate_size * n_shared_experts`` as its intermediate dimension.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ....infra.quant_scheme import (
    FP8_BLOCK, NVFP4, linear_quant_config, quant_scheme,
)
from ..L1.allreduce import AllReduce
from ..L1.grouped_topk import GroupedTopK
from ..L1.gate_linear import GateLinear
from .fused_experts import FusedExperts
from .llama_mlp import LlamaMLP
from .vllm_fused_experts import VllmFusedExperts
from ..L1.fp8_linear import PerTokenGroupQuantFp8
from .trtllm_fp8_moe import (
    TrtllmFp8MoE, prepare_trtllm_moe_weights, trtllm_fp8_moe_available,
)
from .trtllm_fp4_moe import (
    NvFp4Quantize, TrtllmFp4MoE, prepare_trtllm_fp4_moe_weights,
    trtllm_fp4_moe_available,
)

_FP8_BLOCK = 128
_NVFP4_GROUP = 16


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


class DeepSeekMoE(nn.Module):
    """DeepSeek Mixture-of-Experts with shared expert and grouped routing.

    Architecture:
    - Router: replicated gate + e_score_correction_bias
    - Shared expert: LlamaMLP (reduce_results=False)
    - Routed experts: FP8 weights (w13, w2) + per-block scales
    - Routing: GroupedTopK (sigmoid + grouped top-k with bias, matches vLLM)
    - routed_scaling_factor applied post-experts (not in routing weights)
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        # ``quant_scheme`` distinguishes DeepSeek block-FP8 from ModelOpt NVFP4;
        # ``use_fp8`` keeps its historical meaning (block-FP8 experts) so the
        # existing Triton / TRTLLM-fp8 branches are untouched.
        self.quant_scheme = quant_scheme(quant_config)
        self.use_fp8 = self.quant_scheme == FP8_BLOCK
        self.use_nvfp4 = self.quant_scheme == NVFP4
        if self.use_nvfp4 and not trtllm_fp4_moe_available():
            raise RuntimeError(
                "NVFP4 experts require a Blackwell (sm100) GPU with "
                "flashinfer's trtllm_fp4_block_scale_moe -- there is no "
                "fastkernels fallback kernel for NVFP4 W4A4."
            )

        n_group = getattr(config, 'n_group', 1)
        topk_group = getattr(config, 'topk_group', 1)
        self.n_group = n_group
        self.topk_group = topk_group
        self.scoring_func = getattr(config, 'scoring_func', 'softmax')
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.topk_method = getattr(config, 'topk_method', 'noaux_tc')

        self.gate_weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size),
        )
        self.gate_weight.weight_loader = lambda p, w: p.data.copy_(w)

        # ``e_score_correction_bias`` only exists for the ``noaux_tc`` topk
        # method (matches vLLM's ``DeepseekV2MoE`` which only allocates it
        # when ``config.topk_method == "noaux_tc"``). For other methods
        # (e.g., ``greedy``), the bias is simply absent.
        if self.topk_method == 'noaux_tc':
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(config.n_routed_experts, dtype=torch.float32),
            )
            self.e_score_correction_bias.weight_loader = (
                lambda p, w: p.data.copy_(w)
            )
        else:
            self.register_parameter('e_score_correction_bias', None)

        n_shared = getattr(config, 'n_shared_experts', 1)
        if n_shared is not None and n_shared > 0:
            shared_intermediate = config.moe_intermediate_size * n_shared
            # LlamaMLP with reduce_results=False — the final all-reduce
            # is deferred and runs after routed + shared expert are summed.
            #
            # ``linear_quant_config``: under NVFP4 the checkpoint's ``ignore``
            # list covers ``mlp.shared_experts*`` on every layer, so vLLM gives
            # the shared expert ``UnquantizedLinearMethod`` and it stays BF16.
            self.shared_expert = LlamaMLP(
                config,
                quant_config=linear_quant_config(quant_config),
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate,
                reduce_results=False,
            )
        else:
            self.shared_expert = None

        if self.use_nvfp4:
            # ModelOpt NVFP4 expert parameter set, matching
            # ``ModelOptNvFp4FusedMoE.create_weights`` shard-for-shard:
            #   w13                [E, 2N, K//2]   uint8   (2 fp4 per byte)
            #   w2                 [E, K,  N//2]   uint8
            #   w13_weight_scale   [E, 2N, K//16]  fp8e4m3 (per-16 block scale)
            #   w2_weight_scale    [E, K,  N//16]  fp8e4m3
            #   w13_weight_scale_2 [E, 2]          fp32    (per-expert global)
            #   w2_weight_scale_2  [E]             fp32
            #   w13_input_scale    [E, 2]          fp32    (static activation)
            #   w2_input_scale     [E]             fp32
            E, K, N = config.n_routed_experts, config.hidden_size, self.intermediate_per_tp
            assert K % _NVFP4_GROUP == 0 and N % _NVFP4_GROUP == 0, (
                f"NVFP4 needs hidden ({K}) and per-rank intermediate ({N}) "
                f"divisible by {_NVFP4_GROUP}"
            )
            self.w13 = nn.Parameter(
                torch.empty(E, 2 * N, K // 2, dtype=torch.uint8), requires_grad=False,
            )
            self.w2 = nn.Parameter(
                torch.empty(E, K, N // 2, dtype=torch.uint8), requires_grad=False,
            )
            self.w13_weight_scale = nn.Parameter(
                torch.empty(E, 2 * N, K // _NVFP4_GROUP, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.w2_weight_scale = nn.Parameter(
                torch.empty(E, K, N // _NVFP4_GROUP, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.w13_weight_scale_2 = nn.Parameter(
                torch.empty(E, 2, dtype=torch.float32), requires_grad=False,
            )
            self.w2_weight_scale_2 = nn.Parameter(
                torch.empty(E, dtype=torch.float32), requires_grad=False,
            )
            self.w13_input_scale = nn.Parameter(
                torch.empty(E, 2, dtype=torch.float32), requires_grad=False,
            )
            self.w2_input_scale = nn.Parameter(
                torch.empty(E, dtype=torch.float32), requires_grad=False,
            )
        elif self.use_fp8:
            self.w13 = nn.Parameter(torch.empty(
                config.n_routed_experts, 2 * self.intermediate_per_tp, config.hidden_size,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w2 = nn.Parameter(torch.empty(
                config.n_routed_experts, config.hidden_size, self.intermediate_per_tp,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w13_weight_scale_inv = nn.Parameter(torch.empty(
                config.n_routed_experts,
                *_scale_shape(2 * self.intermediate_per_tp, config.hidden_size),
                dtype=torch.float32,
            ), requires_grad=False)
            self.w2_weight_scale_inv = nn.Parameter(torch.empty(
                config.n_routed_experts,
                *_scale_shape(config.hidden_size, self.intermediate_per_tp),
                dtype=torch.float32,
            ), requires_grad=False)
        else:
            self.w13 = nn.Parameter(torch.empty(
                config.n_routed_experts, 2 * self.intermediate_per_tp, config.hidden_size,
            ))
            self.w2 = nn.Parameter(torch.empty(
                config.n_routed_experts, config.hidden_size, self.intermediate_per_tp,
            ))

        self.w13.weight_loader = self._w13_weight_loader
        self.w2.weight_loader = self._w2_weight_loader
        if self.use_fp8:
            self.w13_weight_scale_inv.weight_loader = self._w13_scale_loader
            self.w2_weight_scale_inv.weight_loader = self._w2_scale_loader
        if self.use_nvfp4:
            # Block scales shard exactly like the weights they belong to (row
            # shard for w13, packed-column shard for w2) -- vLLM reaches them
            # through the same ``_load_w13`` / ``_load_w2`` helpers. The
            # per-expert scalars are replicated.
            self.w13_weight_scale.weight_loader = self._w13_weight_loader
            self.w2_weight_scale.weight_loader = self._w2_weight_loader
            self.w13_weight_scale_2.weight_loader = self._w13_per_tensor_loader
            self.w2_weight_scale_2.weight_loader = self._w2_per_tensor_loader
            self.w13_input_scale.weight_loader = self._w13_per_tensor_loader
            self.w2_input_scale.weight_loader = self._w2_per_tensor_loader

        # Routing weights: vLLM always passes ``routed_scaling_factor=1.0``
        # to ``grouped_topk`` and applies the factor *post-experts* (see
        # ``vllm/model_executor/models/deepseek_v2.py:325`` and L378-379).
        # We mirror that by leaving the factor at 1.0 here.
        self.grouped_topk = GroupedTopK(
            scoring_func=self.scoring_func,
            renormalize=self.norm_topk_prob,
            routed_scaling_factor=1.0,
        )
        self.gate = GateLinear()
        # See the ``out_dtype`` note in ``forward_impl``.
        _router_dtype = getattr(config, "moe_router_dtype", None)
        self.router_dtype = (
            torch.float32 if _router_dtype == "float32" else None
        )
        # FP8 path uses a fresh-allocation, vLLM-mirrored op so it is
        # both bit-identical to vLLM's Triton MoE *and* safe to compose
        # with CUDA graph capture (no shared scratch buffers that an
        # eager prefill could reallocate underneath a captured graph).
        # The BF16 / unquantized path keeps the standard ``FusedExperts``.
        if self.use_fp8:
            self.fused_experts = VllmFusedExperts()
        else:
            self.fused_experts = FusedExperts()
        # On Blackwell (sm100) vLLM's fp8-MoE oracle selects the
        # FLASHINFER_TRTLLM kernel, not Triton (it only forces TRITON to the
        # front on Hopper/sm90). Match it: run the TRTLLM path for fp8 experts
        # on sm100, keeping the Triton ``VllmFusedExperts`` for Hopper/other
        # arches. Weights are transformed to the BlockMajorK layout at load
        # time by ``prepare_trtllm_weights`` (the weight postprocess hook calls
        # it and SKIPs the UE8M0 requant for these experts).
        self._use_trtllm = self.use_fp8 and trtllm_fp8_moe_available()
        self._trtllm_weights_ready = False
        if self._use_trtllm:
            self.trtllm_moe = TrtllmFp8MoE()
            self.act_quant = PerTokenGroupQuantFp8()
        if self.use_nvfp4:
            # NVFP4 has exactly one backend on this hardware (checked in
            # __init__), so there is no oracle branch to mirror here.
            # ``fi_moe_largest_bucket`` = max(max_num_tokens * dp_size, 8192);
            # dp_size is 1 under plain TP, and ``max_num_tokens`` is the
            # scheduler's max_num_batched_tokens -- the same value the engine
            # threads onto the config for the DSA top-k buffer.
            self.trtllm_fp4_moe = TrtllmFp4MoE(
                tune_max_num_tokens=max(
                    getattr(config, "max_num_batched_tokens", 8192), 8192,
                ),
            )
            self.act_quant_fp4 = NvFp4Quantize()
            self._fp4_weights_ready = False
        self.allreduce = AllReduce()
        # Mirrors vLLM's ``VLLM_DISABLE_SHARED_EXPERTS_STREAM`` env knob
        # (default: stream enabled). When disabled, the shared expert runs
        # serially on the main stream, which is helpful for debugging and
        # for arches where the secondary stream is harmful.
        import os as _os
        self._disable_shared_stream: bool = (
            _os.environ.get("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0") != "0"
        )
        self._shared_stream: torch.cuda.Stream | None = None

        # Custom-op dispatch scaffolding (matches the other MoE L2 modules).
        # ``_use_custom_op`` is flipped to True by ``enable_custom_ops`` so
        # ``torch.compile`` sees the MoE block as an opaque
        # ``fastkernels::moe_forward`` op (avoids tracing into CUDA stream and
        # DeepGEMM pybind boundaries).
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        """Row (output-dim) shard of one expert's gate_proj / up_proj tensor.

        Derives the per-rank row count from the ``param`` rather than from
        ``intermediate_per_tp`` so the same loader serves the plain weight and
        the NVFP4 block-scale tensor (whose row count is identical) regardless
        of how the trailing dim is packed. Mirrors vLLM's ``_load_w13``: shard
        ``dim 0`` of the loaded 2D tensor, then write into the gate half
        (offset 0) or up half (offset ``rows``) of the ``[2N, ...]`` param.
        """
        rank = _tp_rank()
        rows = param.data.shape[1] // 2
        shard = loaded_weight.narrow(0, rank * rows, rows)
        offset = 0 if is_w1 else rows
        param.data[expert_id, offset:offset + rows, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        """Column (input-dim) shard of one expert's down_proj tensor.

        The shard width comes from the ``param``'s own last dim, which is what
        makes this correct for the NVFP4 tensors: ``w2`` packs two fp4 values
        per byte (``N//2`` columns) and ``w2_weight_scale`` holds one scale per
        16 elements (``N//16`` columns), so a width computed from
        ``intermediate_per_tp`` would over-read both. vLLM shards the same way
        (``_load_w2``: ``loaded.shape[shard_dim] // tp_size``).
        """
        rank = _tp_rank()
        cols = param.data.shape[2]
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * cols, cols))

    def _w13_per_tensor_loader(self, param, loaded_weight, expert_id: int,
                               is_w1: bool):
        """Per-expert scalar for gate_proj / up_proj into a ``[E, 2]`` param.

        Matches vLLM's ``_load_per_tensor_weight_scale`` (column 0 for w1,
        column 1 for w3) and the ModelOpt ``input_scale`` special case in
        ``RoutedExperts.weight_loader``, which writes the two logical shards to
        the two columns instead of broadcasting over the row.
        """
        param.data[expert_id, 0 if is_w1 else 1] = loaded_weight.reshape(())

    def _w2_per_tensor_loader(self, param, loaded_weight, expert_id: int):
        """Per-expert scalar for down_proj into a ``[E]`` param."""
        param.data[expert_id] = loaded_weight.reshape(())

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        scale_rows = math.ceil(N / _FP8_BLOCK)
        offset = 0 if is_w1 else scale_rows
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data[expert_id, offset:offset + scale_rows, :].copy_(src)

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        scale_cols = math.ceil(N / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 1)[rank]
        param.data[expert_id].copy_(src)

    def prepare_trtllm_weights(self) -> None:
        """Transform the raw loaded fp8 expert weights into the FlashInfer
        BlockMajorK layout for the TRTLLM kernel and drop the originals.

        Called once from the weight postprocess hook for sm100 fp8 MoE modules
        (which then SKIP the UE8M0 requant — the TRTLLM path uses the
        un-requantized checkpoint weights). No-op off the TRTLLM path.
        """
        if not getattr(self, "_use_trtllm", False) or self._trtllm_weights_ready:
            return
        w13_fi, w2_fi, w13_scale_fi, w2_scale_fi = prepare_trtllm_moe_weights(
            self.w13.data, self.w2.data,
            self.w13_weight_scale_inv.data, self.w2_weight_scale_inv.data,
        )
        self.register_buffer("w13_fi", w13_fi, persistent=False)
        self.register_buffer("w2_fi", w2_fi, persistent=False)
        self.register_buffer("w13_scale_fi", w13_scale_fi, persistent=False)
        self.register_buffer("w2_scale_fi", w2_scale_fi, persistent=False)
        # Free the raw [E,2N,K]/[E,K,N] params — the FI buffers replace them.
        del self.w13, self.w2, self.w13_weight_scale_inv, self.w2_weight_scale_inv
        self._trtllm_weights_ready = True
        torch.cuda.empty_cache()

    def prepare_fp4_weights(self) -> None:
        """Transform the raw loaded NVFP4 expert weights into TRTLLM-gen layout.

        Called once from the weight postprocess hook. Mirrors
        ``ModelOptNvFp4FusedMoE.process_weights_after_loading`` plus
        ``TrtLlmNvFp4ExpertsBase.process_weights_after_loading``: permute /
        interleave the weights and block scales, collapse the activation scales
        to a global max, and fold them into the per-expert GEMM alphas. The raw
        parameters are dropped afterwards — vLLM likewise ``replace_parameter``s
        them, so nothing downstream reads the pre-transform tensors.
        """
        if not self.use_nvfp4 or self._fp4_weights_ready:
            return
        prepared = prepare_trtllm_fp4_moe_weights(
            self.w13.data, self.w13_weight_scale.data,
            self.w13_weight_scale_2.data, self.w13_input_scale.data,
            self.w2.data, self.w2_weight_scale.data,
            self.w2_weight_scale_2.data, self.w2_input_scale.data,
        )
        del (self.w13, self.w2, self.w13_weight_scale, self.w2_weight_scale,
             self.w13_weight_scale_2, self.w2_weight_scale_2,
             self.w13_input_scale, self.w2_input_scale)
        for name, tensor in prepared.items():
            self.register_buffer(f"fp4_{name}", tensor, persistent=False)
        self._fp4_weights_ready = True
        torch.cuda.empty_cache()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op: inside it Inductor
            # cannot see the collective, so ``AllReduceFusedAddRMSNormPass`` has
            # nothing to match at the MoE end of the layer -- half of every
            # layer's collectives. vLLM keeps its MoE reduction in traced Python
            # for the same reason (``moe_runner._maybe_reduce_final_output``).
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Launch shared expert on a separate CUDA stream for overlap.
        # Matches vLLM's monolithic DeepSeekV2MoE behaviour
        # (``DefaultMoeRunner`` in
        # ``vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py``).
        shared_out = None
        use_shared_stream = (
            self.shared_expert is not None and not self._disable_shared_stream
        )
        if use_shared_stream:
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream()
            self._shared_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._shared_stream):
                shared_out = self.shared_expert(hidden_states)
        elif self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)

        # Router gate with vLLM-parity dispatch. vLLM's GateLinear routes
        # through three tiers (DSV3 specialised kernel → cuBLAS BF16→FP32 →
        # PyTorch F.linear) and the accumulation order matters: a "promote
        # both to FP32 then matmul" path produces a different bit pattern
        # that flips near-tie group/expert selection in the noaux_tc
        # grouped-topk path.
        #
        # Router ``out_dtype`` mirrors vLLM ``DeepseekV2MoE``, which builds its
        # gate with ``out_dtype=_get_moe_router_dtype(config)``:
        #
        # * ``glm_moe_dsa`` (GLM-5.2) -> **FP32, unconditionally** ("Older
        #   GLM-5/5.2 configs require fp32 routing but do not expose
        #   moe_router_dtype yet"). Leaving it ``None`` would give FP32 in decode
        #   but BF16 in prefill (the ``F.linear`` fallback does not cast), so
        #   every prefill token would reach grouped-topk with a different bit
        #   pattern than vLLM and flip near-tie expert selection.
        # * every other model -> ``None`` unless the config says
        #   ``moe_router_dtype: "float32"``. With ``None`` vLLM's dispatch yields
        #   FP32 in decode and BF16 in prefill, and it does NOT upcast before the
        #   sigmoid, so that precision split feeds grouped-topk verbatim.
        router_logits = self.gate(
            hidden_states, self.gate_weight, out_dtype=self.router_dtype,
        )
        if self.use_nvfp4:
            # Blackwell FLASHINFER_TRTLLM NVFP4 path — vLLM's only viable NVFP4
            # MoE backend here, and it picks the MONOLITHIC variant
            # (``trtllm_fp4_block_scale_moe``) because all2all and EPLB are off:
            # routing (sigmoid + noaux_tc grouped top-k + renormalize) runs
            # inside the kernel from raw ``router_logits``, so ``GroupedTopK``
            # is not used at all on this path. Activations are quantized with
            # the checkpoint's STATIC global scale (``a1_gscale``), matching
            # ``MoEPrepareAndFinalizeNoDPEPMonolithic.prepare``.
            # ``routed_scaling_factor`` is applied post-experts below.
            a_fp4, a_scale = self.act_quant_fp4(
                hidden_states, self.fp4_a1_gscale,
            )
            out = self.trtllm_fp4_moe.forward_monolithic(
                a_fp4, a_scale,
                self.fp4_w13, self.fp4_w13_scale,
                self.fp4_w2, self.fp4_w2_scale,
                self.fp4_g1_alphas, self.fp4_g2_alphas, self.fp4_g1_scale_c,
                router_logits, self.e_score_correction_bias,
                self.num_experts, self.top_k, self.n_group, self.topk_group,
                self.intermediate_per_tp,
                norm_topk_prob=self.norm_topk_prob,
            )
        elif self._use_trtllm:
            # Blackwell FLASHINFER_TRTLLM path (vLLM's oracle choice on sm100).
            # Use the MONOLITHIC kernel (``trtllm_fp8_block_scale_moe``) with
            # routing (sigmoid + noaux_tc grouped top-k + norm) done INSIDE the
            # kernel from raw ``router_logits`` — exactly vLLM's DeepSeek/GLM CUDA
            # path (``routing_method_type=DeepSeekV3``). This is BIT-IDENTICAL to
            # vLLM; the pre-routed variant instead truncates the top-k combine
            # weights to bf16, a ~1-ULP error that scales with expert-output
            # magnitude. ``routed_scaling_factor`` is applied post-experts below.
            M, K = hidden_states.shape
            a_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn,
                                device=hidden_states.device)
            a_scale = torch.empty(M, K // _FP8_BLOCK, dtype=torch.float32,
                                  device=hidden_states.device)
            self.act_quant(hidden_states, a_fp8, a_scale)
            out = self.trtllm_moe.forward_monolithic(
                a_fp8, a_scale.t().contiguous(),
                self.w13_fi, self.w13_scale_fi, self.w2_fi, self.w2_scale_fi,
                router_logits, self.e_score_correction_bias,
                self.num_experts, self.top_k, self.n_group, self.topk_group,
                self.intermediate_per_tp,
            )
        elif self.use_fp8:
            topk_weights, topk_ids = self.grouped_topk(
                router_logits, self.e_score_correction_bias,
                self.n_group, self.topk_group, self.top_k,
            )
            # FP8 W8A8 block-quant on Hopper + TP: vLLM's oracle
            # (``select_fp8_moe_backend`` -> ``_get_priority_backends``
            # in ``vllm/.../fused_moe/oracle/fp8.py``) explicitly moves
            # ``TRITON`` to the front for this configuration.  The
            # DeepGEMM path drifts ~1 BF16 ULP which cascades through
            # near-tie expert selection in subsequent MoE layers.
            # ``VllmFusedExperts`` is a fresh-allocation port of vLLM's
            # ``fused_experts_impl`` Triton path and consumes
            # ``topk_weights`` in FP32 (vLLM's ``GroupedTopKRouter``
            # returns FP32 and the Triton kernel scales in FP32 before
            # the final ``.to(compute_type)`` cast).
            out = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
                w13_scale=self.w13_weight_scale_inv,
                w2_scale=self.w2_weight_scale_inv,
                block_shape=[_FP8_BLOCK, _FP8_BLOCK],
            )
        else:
            topk_weights, topk_ids = self.grouped_topk(
                router_logits, self.e_score_correction_bias,
                self.n_group, self.topk_group, self.top_k,
            )
            topk_weights_act = topk_weights.to(hidden_states.dtype)
            out = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights_act, topk_ids, self.num_experts,
            )

        # ``routed_scaling_factor`` is applied *post-experts*, matching
        # ``vllm/model_executor/models/deepseek_v2.py:378-379``.
        #
        # Scale and shared-expert add in ONE kernel. vLLM gets the same fusion
        # from Inductor (its profile shows a single ``triton_poi_fused_add_mul``
        # where fastkernels showed two separate elementwise kernels costing 2x
        # the time); fastkernels cannot rely on Inductor here because this whole
        # block is inside the opaque ``fastkernels::moe_forward`` custom op. One
        # kernel also rounds once instead of twice, which is what vLLM does.
        if shared_out is not None:
            if use_shared_stream:
                torch.cuda.current_stream().wait_stream(self._shared_stream)
            out = torch.add(shared_out, out, alpha=self.routed_scaling_factor)
        else:
            out = out * self.routed_scaling_factor

        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)

        return out.view(orig_shape)
