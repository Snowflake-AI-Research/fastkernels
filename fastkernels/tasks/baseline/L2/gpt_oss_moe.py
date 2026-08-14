"""GPT-OSS MoE: MXFP4-native fused MoE composed from FastKernels L1 ops.

128 experts (top-4, softmax routing), router bias, expert gate/up/down biases,
OAI SwiGLU activation fused inside the expert kernel.

Expert weights are kept in packed MXFP4 uint8 format (2× FP4 per byte) with
E8M0 block scales. No dequantization is performed.

Two expert kernels exist, and which one vLLM picks depends on the device
(``Mxfp4MoEMethod`` -> ``select_deepseek_v4_mxfp4_moe_backend``):

* **SM100** -> ``FLASHINFER_TRTLLM_MXFP4_BF16`` /
  ``TrtLlmMxfp4ExpertsMonolithic``, i.e. ``flashinfer.trtllm_fp4_block_scale_moe``
  (:mod:`..L1.trtllm_mxfp4_moe`). This needs hidden/intermediate rounded up to
  256 and a shuffled weight layout.
* **otherwise** -> the OAI Triton ``matmul_ogs`` kernel
  (:mod:`..L1.mxfp4_moe`).

Both are kept so the module matches vLLM on whichever device it runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.linear import Linear
from .mxfp4_moe import Mxfp4MoE
from .trtllm_mxfp4_moe import (
    TRTLLM_MXFP4_ALIGN,
    TrtLlmMxfp4MoE,
    prepare_trtllm_mxfp4_weights,
    trtllm_mxfp4_moe_supported,
)


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class GptOssMoE(nn.Module):
    """MXFP4-native MoE composed from FastKernels L1 ops.

    Weights stay in packed uint8 MXFP4 format. Routing, swizzling and the
    fused matmul_ogs forward are all delegated to ``L1.mxfp4_moe``.
    """

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

        E = config.num_local_experts
        BLK = self.MXFP4_BLOCK

        # Which expert kernel, and therefore which alignment. vLLM's
        # ``mxfp4_round_up_hidden_size_and_intermediate_size`` uses 256 for the
        # TRTLLM backends and 64 for the Triton one, and it pads *hidden* too
        # (2880 -> 3072 for gpt-oss), not just the intermediate.
        self.use_trtllm = trtllm_mxfp4_moe_supported()
        if self.use_trtllm:
            I_pad = _round_up(self.intermediate_per_tp, TRTLLM_MXFP4_ALIGN)
            H_pad = _round_up(self.hidden_size, TRTLLM_MXFP4_ALIGN)
        else:
            I_pad = _round_up(self.intermediate_per_tp, 64)
            H_pad = self.hidden_size
        H = H_pad

        self._I_pad = I_pad
        self._H_pad = H_pad

        # Expert weights in packed MXFP4 uint8 (2× FP4 per byte)
        self.w13_weight = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // BLK, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_bias = nn.Parameter(
            torch.zeros(E, 2 * I_pad, dtype=torch.bfloat16),
            requires_grad=False,
        )

        self.w2_weight = nn.Parameter(
            torch.zeros(E, H, I_pad // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(E, H, I_pad // BLK, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_bias = nn.Parameter(
            torch.zeros(E, H, dtype=torch.bfloat16),
            requires_grad=False,
        )

        # Set up weight loaders for checkpoint loading
        self.w13_weight.weight_loader = self._w13_weight_loader
        self.w13_weight_scale.weight_loader = self._w13_scale_loader
        self.w13_bias.weight_loader = self._w13_bias_loader
        self.w2_weight.weight_loader = self._w2_weight_loader
        self.w2_weight_scale.weight_loader = self._w2_scale_loader
        self.w2_bias.weight_loader = self._w2_bias_loader

        self.allreduce = AllReduce()
        self.mxfp4_moe = Mxfp4MoE()
        self.trtllm_moe = (
            TrtLlmMxfp4MoE(
                num_experts=E,
                top_k=self.top_k,
                intermediate_size=I_pad,
                hidden_size_unpadded=self.hidden_size,
            )
            if self.use_trtllm
            else None
        )

        # Populated after process_weights_after_loading
        self._quant_config = None
        self._processed = False

        # Custom-op dispatch for torch.compile (set by engine after model init)
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight):
        """Load w13 MXFP4 packed weight with TP sharding.

        Checkpoint shape: [E, 2*I_full, num_blocks, 16] (4D blocks) or
                          [E, 2*I_full, H//2] (pre-flattened).
        Gate/up rows are interleaved (gate_0, up_0, gate_1, up_1, ...);
        we keep them interleaved, matching the expert kernel's expectation.

        The destination may be padded in *both* dims (the TRTLLM backend rounds
        hidden and intermediate up to 256), so the copy is bounded by the
        checkpoint's own extents and the padding stays zero.
        """
        if loaded_weight.ndim == 4:
            E, N, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, N, nb * bs)
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        k = loaded_weight.shape[-1]
        param.data[:, :2*I, :k].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_scale_loader(self, param, loaded_weight):
        """Load w13 scales with TP shard, keeping interleaved layout."""
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        k = loaded_weight.shape[-1]
        param.data[:, :2*I, :k].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_bias_loader(self, param, loaded_weight):
        """Load w13 bias [E, 2*I] with TP shard, keeping interleaved layout."""
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        param.data[:, :2*I].copy_(loaded_weight[:, start : start + 2*I])

    def _w2_weight_loader(self, param, loaded_weight):
        """Load w2 MXFP4 packed weight with TP shard.

        Checkpoint shape: [E, H, num_blocks, 16] (4D blocks) or
                          [E, H, I//2] (pre-flattened).
        """
        if loaded_weight.ndim == 4:
            E, H, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, H, nb * bs)
        tp, rank = _tp_size(), _tp_rank()
        I_half = self.intermediate_per_tp // 2
        h = loaded_weight.shape[1]
        param.data[:, :h, :I_half].copy_(
            loaded_weight[:, :, rank * I_half : rank * I_half + I_half]
        )

    def _w2_scale_loader(self, param, loaded_weight):
        """Load w2 scales with TP shard."""
        tp, rank = _tp_size(), _tp_rank()
        I_blk = self.intermediate_per_tp // self.MXFP4_BLOCK
        h = loaded_weight.shape[1]
        param.data[:, :h, :I_blk].copy_(
            loaded_weight[:, :, rank * I_blk : rank * I_blk + I_blk]
        )

    def _w2_bias_loader(self, param, loaded_weight):
        """Load w2 bias [E, H]. Only rank 0 loads; others zero (reduced by allreduce)."""
        if _tp_rank() == 0:
            param.data[:, : loaded_weight.shape[1]].copy_(loaded_weight)
        else:
            param.data.zero_()

    def process_weights_after_loading(self):
        """Convert MXFP4 weights into the selected expert kernel's layout.

        Must be called after all weights are loaded and moved to GPU.
        """
        if self._processed:
            return

        if self.use_trtllm:
            # trtllm-gen wants float32 biases, a gate/up row swap, and the
            # shuffled/interleaved weight+scale layout for its transposed MMA
            # epilogue (vLLM's ``convert_gpt_oss_weight_to_mxfp4_moe_kernel_format``).
            (
                w13_weight, w13_scale, w13_bias,
                w2_weight, w2_scale, w2_bias,
            ) = prepare_trtllm_mxfp4_weights(
                self.w13_weight.data,
                self.w13_weight_scale.data,
                self.w13_bias.data,
                self.w2_weight.data,
                self.w2_weight_scale.data,
                self.w2_bias.data,
            )
            del self.w13_weight, self.w2_weight
            del self.w13_weight_scale, self.w2_weight_scale
            del self.w13_bias, self.w2_bias
            self._w13_shuffled = w13_weight
            self._w13_scale = w13_scale
            self._w13_bias_f32 = w13_bias
            self._w2_shuffled = w2_weight
            self._w2_scale = w2_scale
            self._w2_bias_f32 = w2_bias
            torch.cuda.empty_cache()
            self._processed = True
            return

        # Biases must be float32 for the Triton kernel
        self.w13_bias.data = self.w13_bias.data.float()
        self.w2_bias.data = self.w2_bias.data.float()

        w13_weight, w13_precision = Mxfp4MoE.prepare_weight(
            self.w13_weight.data, self.w13_weight_scale.data
        )
        w2_weight, w2_precision = Mxfp4MoE.prepare_weight(
            self.w2_weight.data, self.w2_weight_scale.data
        )

        # prepare_weight returns triton_kernels.Tensor objects, not
        # torch.Tensor; store as plain attributes (the original nn.Parameters
        # are no longer used)
        del self.w13_weight, self.w2_weight
        del self.w13_weight_scale, self.w2_weight_scale
        self._w13_swizzled = w13_weight
        self._w2_swizzled = w2_weight

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

        if self.use_trtllm:
            # Zero-pad the activation into the kernel's hidden width; the
            # kernel writes an unpadded output (vLLM: forward padding in
            # ``MoERunner`` + ``has_unpadded_output``).
            if self._H_pad != self.hidden_size:
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, self._H_pad - self.hidden_size),
                )
            output = self.trtllm_moe(
                hidden_states,
                router_logits,
                self._w13_shuffled,
                self._w13_scale,
                self._w13_bias_f32,
                self._w2_shuffled,
                self._w2_scale,
                self._w2_bias_f32,
            )
        else:
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

        if self.tp_size > 1 and not self._use_custom_op:
            output = self.allreduce(output)

        return output.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op on purpose. Inside it
            # Inductor cannot see the collective, so the AR+RMSNorm fusion has
            # nothing to match at the MoE end of the layer -- half of every
            # layer's collectives. vLLM keeps its MoE reduction in traced Python
            # for the same reason (``moe_runner._maybe_reduce_final_output``).
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)
