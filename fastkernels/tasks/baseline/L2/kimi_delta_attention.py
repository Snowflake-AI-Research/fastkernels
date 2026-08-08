from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import triton
from einops import rearrange

from ..L1.rms_norm_gated import FusedRMSNormGated
from ..L1.kda import (
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from ..L1.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)


def set_triton_allocator(device: torch.device):
    """Verbatim from vLLM's ``triton_utils.allocation.set_triton_allocator``."""

    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

from ....infra.context import get_context
from ....infra.tp import _tp_rank, _tp_size
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


class _Conv1DWeights(nn.Module):
    """Sharded depthwise-conv weight holder with HF-compatible parameter names."""

    def __init__(self, output_size: int, kernel_size: int):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                1,
                kernel_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self.bias = None

    def _weight_loader(self, param, loaded_weight):
        shard = param.data.size(0)
        rank = _tp_rank()
        param.data.copy_(
            loaded_weight.narrow(0, rank * shard, shard).to(torch.float32),
        )


@dataclass
class _KDAStateView:
    q_conv_state: torch.Tensor
    k_conv_state: torch.Tensor
    v_conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class KimiDeltaAttention(nn.Module):
    """Kimi Linear's KDA layer.

    Uses vLLM/FLA kernels for the gate and the gated delta attention core,
    while reading runtime state + metadata from fastkernels's global Context.
    """

    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.tp_size = _tp_size()
        self.hidden_size = config.hidden_size
        kda_config = config.linear_attn_config
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        self.layer_idx = layer_idx
        self.conv_size = kda_config["short_conv_kernel_size"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = self.num_heads // self.tp_size

        projection_size = self.head_dim * self.num_heads

        # q/k/v/b are four ColumnParallelLinear GEMMs over the same
        # hidden_states, so at decode width they are four launches of a GEMV-shaped
        # kernel. Profiled at tp=2 bs=1 they are the
        # ``nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`` at 99.2 calls/step x 7.0 us =
        # 694 us of a 4.06 ms step (16.8%) -- a 16-row tile is the compiler telling
        # us there is not enough work per launch. One [3*proj + num_heads] GEMM does
        # the same math in one launch with a tile that fits.
        #
        # Only the loader needs care: ColumnParallelLinear shards its output, and
        # rank r needs *its own shard of each sub-projection* laid out end to end --
        # not the r-th contiguous slice of the concatenation. ``_qkvb_weight_loader``
        # places each one at its local offset. Kept separate under quantization,
        # where the block scales would have to be concatenated too.
        self.qkvb_proj = ColumnParallelLinear(
            self.hidden_size,
            3 * projection_size + self.num_heads,
            bias=False,
            quant_config=None,
        ) if quant_config is None else None
        if self.qkvb_proj is not None:
            _ps_local = projection_size // self.tp_size
            _nh_local = self.num_heads // self.tp_size
            self._ps_local = _ps_local
            self._nh_local = _nh_local

            def _qkvb_weight_loader(param, loaded_weight, shard_id):
                # shard_id 0/1/2/3 = q/k/v/b (see packed_modules_mapping in
                # L4/kimi_linear.py). b_proj is num_heads wide, the others
                # projection_size.
                rank = _tp_rank()
                if shard_id == 3:
                    local, offset = _nh_local, 3 * _ps_local
                else:
                    local, offset = _ps_local, shard_id * _ps_local
                param.data[offset:offset + local].copy_(
                    loaded_weight.narrow(0, rank * local, local),
                )

            self.qkvb_proj.weight.weight_loader = _qkvb_weight_loader

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, projection_size, bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, projection_size, bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, projection_size, bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size // self.tp_size, dtype=torch.float32),
        )
        self.dt_bias.weight_loader = self._shard0_loader

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None

        self.q_conv1d = _Conv1DWeights(projection_size, self.conv_size)
        self.k_conv1d = _Conv1DWeights(projection_size, self.conv_size)
        self.v_conv1d = _Conv1DWeights(projection_size, self.conv_size)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32),
        )
        self.A_log.weight_loader = self._a_log_loader

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.rms_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self._triton_allocator_ready = False
        self._use_custom_op = False
        self._layer_name = ""

    @staticmethod
    def _shard0_loader(param, loaded_weight):
        shard = param.data.size(0)
        rank = _tp_rank()
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard).to(param.dtype))

    @staticmethod
    def _a_log_loader(param, loaded_weight):
        rank = _tp_rank()
        tp = _tp_size()
        shard = param.data.shape[2]
        param.data.copy_(
            loaded_weight.narrow(2, rank * shard, shard).to(param.dtype),
        )

    def _get_state(self) -> tuple[_KDAStateView | None, object | None]:
        ctx = get_context()
        kda_state = getattr(ctx, "kda_state", None)
        kda_meta = getattr(ctx, "kda_metadata", None)
        if kda_state is None or kda_meta is None:
            return None, None
        return _KDAStateView(
            q_conv_state=kda_state.q_conv_states[self.layer_idx],
            k_conv_state=kda_state.k_conv_states[self.layer_idx],
            v_conv_state=kda_state.v_conv_states[self.layer_idx],
            recurrent_state=kda_state.recurrent_states[self.layer_idx],
        ), kda_meta

    def _run_conv_prefill(self, x, state, conv_weight, meta):
        return causal_conv1d_fn(
            x.transpose(0, 1),
            conv_weight,
            None,
            activation="silu",
            conv_states=state.transpose(-1, -2),
            has_initial_state=meta.has_initial_state,
            cache_indices=meta.non_spec_state_indices_tensor,
            query_start_loc=meta.non_spec_query_start_loc.to(torch.int32),
            metadata=meta,
        ).transpose(0, 1)

    def _run_conv_decode(self, x, state, conv_weight, meta):
        return causal_conv1d_update(
            x,
            state.transpose(-1, -2),
            conv_weight,
            None,
            activation="silu",
            conv_state_indices=meta.non_spec_state_indices_tensor[:meta.num_actual_tokens],
            validate_data=True,
        )

    def _ensure_triton_allocator(self, device: torch.device) -> None:
        if torch.compiler.is_compiling():
            return
        if not self._triton_allocator_ready:
            set_triton_allocator(device)
            self._triton_allocator_ready = True

    def forward_impl(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        raw_g: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        state_view, meta = self._get_state()
        if state_view is None or meta is None:
            core_attn_out.zero_()
            return

        num_actual_tokens = meta.num_actual_tokens
        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        raw_g = raw_g[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0),
            self.q_conv1d.weight.size(2),
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0),
            self.k_conv1d.weight.size(2),
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0),
            self.v_conv1d.weight.size(2),
        )

        if meta.num_prefills > 0:
            q = self._run_conv_prefill(q_proj_states, state_view.q_conv_state, q_conv_weights, meta)
            k = self._run_conv_prefill(k_proj_states, state_view.k_conv_state, k_conv_weights, meta)
            v = self._run_conv_prefill(v_proj_states, state_view.v_conv_state, v_conv_weights, meta)
        else:
            q = self._run_conv_decode(q_proj_states, state_view.q_conv_state, q_conv_weights, meta)
            k = self._run_conv_decode(k_proj_states, state_view.k_conv_state, k_conv_weights, meta)
            v = self._run_conv_decode(v_proj_states, state_view.v_conv_state, v_conv_weights, meta)

        q, k, v = (
            rearrange(q, "n (h d) -> 1 n h d", d=self.head_dim),
            rearrange(k, "n (h d) -> 1 n h d", d=self.head_dim),
            rearrange(v, "n (h d) -> 1 n h d", d=self.head_dim),
        )

        num_prefill_tokens = meta.num_prefill_tokens
        num_decode_tokens = meta.num_decode_tokens

        if num_prefill_tokens > 0:
            pf_state_indices = meta.non_spec_state_indices_tensor[:meta.num_prefills]
            pf_has_initial = meta.has_initial_state[:meta.num_prefills]
            pf_cu_seqlens = (
                meta.non_spec_query_start_loc
                if meta.num_decodes == 0
                else meta.non_spec_query_start_loc[: meta.num_prefills + 1]
            )
            # int32, not int64: vLLM's GDN metadata builds
            # ``non_spec_query_start_loc`` as int32 and the FLA chunk kernels
            # index with that width. Handing them int64 offsets silently
            # changes the result (verified in isolation: identical inputs,
            # amax 0.042 with int32 vs 0.0005 with int64).
            pf_cu_seqlens = pf_cu_seqlens.to(torch.int32)
            if torch.cuda.is_current_stream_capturing() and meta.num_decodes == 0:
                pf_initial_state = state_view.recurrent_state[pf_state_indices].contiguous()
                pf_initial_state.zero_()
            else:
                zero_idx = pf_state_indices[~pf_has_initial]
                if zero_idx.numel() > 0:
                    state_view.recurrent_state[zero_idx] = 0
                pf_initial_state = state_view.recurrent_state[pf_state_indices].contiguous()
            # vLLM's Kimi prefill (``kimi_gdn_linear_attn._forward``) hands the
            # *raw* gate projection to ``chunk_kda_with_fused_gate``, which
            # applies ``A_log``/``dt_bias`` and the softplus in fp32 registers
            # inside the chunk kernel. Materializing the gate first with
            # ``fused_kda_gate`` and calling ``chunk_kda`` gives a bit-identical
            # output (verified: cos 1.000000, max|d| 0) but costs an extra pass
            # -- the fused form is 1.06-1.27x faster over 512..8192 tokens.
            pf_out, pf_last_state = chunk_kda_with_fused_gate(
                q=q[:, :num_prefill_tokens].contiguous(),
                k=k[:, :num_prefill_tokens].contiguous(),
                v=v[:, :num_prefill_tokens].contiguous(),
                raw_g=raw_g[:, :num_prefill_tokens].contiguous(),
                beta=beta[:, :num_prefill_tokens].contiguous(),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=pf_initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=pf_cu_seqlens,
            )
            state_view.recurrent_state[pf_state_indices] = pf_last_state
            core_attn_out[:, :num_prefill_tokens] = pf_out

        if num_decode_tokens > 0:
            dec_start = num_prefill_tokens
            dec_state_indices = meta.non_spec_state_indices_tensor
            if meta.num_prefills > 0:
                dec_state_indices = dec_state_indices[meta.num_prefills:]
            dec_q = q[:, dec_start:].contiguous()
            dec_k = k[:, dec_start:].contiguous()
            dec_v = v[:, dec_start:].contiguous()
            dec_beta = beta[:, dec_start:].contiguous()
            dec_cu = (
                meta.non_spec_query_start_loc
                if meta.num_prefills == 0
                else meta.non_spec_query_start_loc[: meta.num_decodes + 1]
            ).to(torch.int32)
            # vLLM's decode gates first with ``fused_kda_gate`` and then calls
            # ``fused_recurrent_kda`` against the full recurrent state, indexed
            # by ``ssm_state_indices``. This replaces a hand-written kernel whose
            # premise -- that ``chunk_kda`` and ``fused_recurrent_kda`` disagree
            # on the state layout -- could not be reproduced, and which was
            # slower at the batch sizes that matter (1.27x at 256 sequences).
            dec_g = fused_kda_gate(
                rearrange(raw_g[:, dec_start:], "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
            ).unsqueeze(0)
            # This kernel treats state index 0 as a null/skip slot, the same
            # convention as ``causal_conv1d``'s ``NULL_BLOCK_ID``: given slot 0
            # it returns NaN and writes no state (measured: slot 0 -> nan with
            # zero state delta; slots 1 and 3 -> clean, delta ~0.088). The state
            # allocator reserves slot 0 so no call site has to special-case it.
            dec_out, _ = fused_recurrent_kda(
                q=dec_q,
                k=dec_k,
                v=dec_v,
                g=dec_g,
                beta=dec_beta,
                initial_state=state_view.recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=dec_cu,
                ssm_state_indices=dec_state_indices,
            )
            core_attn_out[:, dec_start:] = dec_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_manager=None,
    ) -> torch.Tensor:
        del state_manager
        num_tokens = hidden_states.size(0)
        self._ensure_triton_allocator(hidden_states.device)

        if self.qkvb_proj is not None:
            _p = self._ps_local
            qkvb = self.qkvb_proj(hidden_states)
            q_proj_states = qkvb[..., :_p]
            k_proj_states = qkvb[..., _p:2 * _p]
            v_proj_states = qkvb[..., 2 * _p:3 * _p]
            raw_beta = qkvb[..., 3 * _p:]
        else:
            q_proj_states = self.q_proj(hidden_states)
            k_proj_states = self.k_proj(hidden_states)
            v_proj_states = self.v_proj(hidden_states)
            raw_beta = self.b_proj(hidden_states)

        # ``.float()`` already materializes a contiguous copy, so the beta slice
        # never reaches a kernel strided.
        beta = raw_beta.float().sigmoid().unsqueeze(0)
        # Raw gate projection, shaped [1, n, H, D] and left ungated: the prefill
        # chunk kernel applies A_log/dt_bias itself, and the decode path gates
        # with ``fused_kda_gate`` just before its call. Mirrors vLLM's
        # ``KimiDeltaAttention.forward``.
        raw_g = rearrange(
            self.f_b_proj(self.f_a_proj(hidden_states)),
            "n (h d) -> 1 n h d",
            d=self.head_dim,
        )

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states))
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if self._use_custom_op:
            torch.ops.fastkernels.kda_attention(
                q_proj_states,
                k_proj_states,
                v_proj_states,
                raw_g,
                beta,
                core_attn_out,
                self._layer_name,
            )
        else:
            self.forward_impl(
                q_proj_states=q_proj_states,
                k_proj_states=k_proj_states,
                v_proj_states=v_proj_states,
                raw_g=raw_g,
                beta=beta,
                core_attn_out=core_attn_out,
            )

        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)
