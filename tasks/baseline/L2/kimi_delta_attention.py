from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from vllm.model_executor.layers.fla.ops.kda import (
    FusedRMSNormGated,
    chunk_kda,
    fused_kda_gate,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.triton_utils.allocation import set_triton_allocator

from ....infra.context import get_context
from ....infra.tp import _tp_rank, _tp_size
from ..L1.kda_recurrence import FusedRecurrentKDAChunkOutput
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
    while reading runtime state + metadata from kb_nano's global Context.
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

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
        )

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
        )

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
        self.recurrent_decode = FusedRecurrentKDAChunkOutput()
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
            query_start_loc=meta.non_spec_query_start_loc,
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
        g1: torch.Tensor,
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
        g1 = g1[:, :num_actual_tokens]
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
            if torch.cuda.is_current_stream_capturing() and meta.num_decodes == 0:
                pf_initial_state = state_view.recurrent_state[pf_state_indices].contiguous()
                pf_initial_state.zero_()
            else:
                zero_idx = pf_state_indices[~pf_has_initial]
                if zero_idx.numel() > 0:
                    state_view.recurrent_state[zero_idx] = 0
                pf_initial_state = state_view.recurrent_state[pf_state_indices].contiguous()
            pf_out, pf_last_state = chunk_kda(
                q=q[:, :num_prefill_tokens].contiguous(),
                k=k[:, :num_prefill_tokens].contiguous(),
                v=v[:, :num_prefill_tokens].contiguous(),
                g=g1[:, :num_prefill_tokens].contiguous(),
                beta=beta[:, :num_prefill_tokens].contiguous(),
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
            dec_g = g1[:, dec_start:].contiguous()
            dec_beta = beta[:, dec_start:].contiguous()
            dec_cu = (
                meta.non_spec_query_start_loc
                if meta.num_prefills == 0
                else meta.non_spec_query_start_loc[: meta.num_decodes + 1]
            )
            dec_out, _ = self.recurrent_decode(
                q=dec_q,
                k=dec_k,
                v=dec_v,
                g=dec_g,
                beta=dec_beta,
                initial_state=state_view.recurrent_state,
                cu_seqlens=dec_cu,
                state_indices=dec_state_indices,
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

        q_proj_states = self.q_proj(hidden_states)
        k_proj_states = self.k_proj(hidden_states)
        v_proj_states = self.v_proj(hidden_states)

        beta = self.b_proj(hidden_states).float().sigmoid().unsqueeze(0)
        g1 = self.f_b_proj(self.f_a_proj(hidden_states))
        g1 = fused_kda_gate(g1, self.A_log, self.head_dim, g_bias=self.dt_bias).unsqueeze(0)

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states))
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if self._use_custom_op:
            torch.ops.kb_nano.kda_attention(
                q_proj_states,
                k_proj_states,
                v_proj_states,
                g1,
                beta,
                core_attn_out,
                self._layer_name,
            )
        else:
            self.forward_impl(
                q_proj_states=q_proj_states,
                k_proj_states=k_proj_states,
                v_proj_states=v_proj_states,
                g1=g1,
                beta=beta,
                core_attn_out=core_attn_out,
            )

        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)
