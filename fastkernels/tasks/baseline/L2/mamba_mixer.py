"""Mamba v1 mixer (selective state-space model).

Implementation mirrors vLLM's ``MambaMixer``
(``vllm/model_executor/layers/mamba/mamba_mixer.py``) so that:

  - kernel calls (causal_conv1d_fn / causal_conv1d_update /
    selective_scan_fn / selective_state_update) are bit-identical to vLLM
  - parameter layout / weight names match HF Mamba checkpoints
    (state-spaces/mamba-* family)
  - tensor parallelism uses ColumnParallelLinear (in_proj, conv1d,
    dt_proj) and RowParallelLinear (x_proj, out_proj), matching vLLM

State (conv_state, ssm_state) and per-batch metadata are read from
fastkernels's global ``Context`` (``infra/context.py``), analogous to
vLLM's ``ForwardContext``.

Weight names from HF Mamba checkpoint
-------------------------------------
    mixer.in_proj.weight        [2*intermediate, hidden]   (gate + x)
    mixer.conv1d.weight         [intermediate, 1, conv_kernel]
    mixer.conv1d.bias           [intermediate]
    mixer.x_proj.weight         [time_step_rank + 2*state_size, intermediate]
    mixer.dt_proj.weight        [intermediate, time_step_rank]
    mixer.dt_proj.bias          [intermediate]
    mixer.A_log                 [intermediate, state_size]
    mixer.D                     [intermediate]
    mixer.out_proj.weight       [hidden, intermediate]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)

from ....infra.context import get_context
from ....infra.tp import _tp_rank, _tp_size
from .parallel_linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


class MambaMixer(nn.Module):
    """Mamba v1 selective-scan mixer block."""

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        activation: str = "silu",
        layer_idx: int = 0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.tp_size = _tp_size()
        self.tp_rank = _tp_rank()

        assert intermediate_size % self.tp_size == 0, (
            "Mamba v1 requires intermediate_size divisible by tp_size."
        )

        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size
        self.time_step_rank = time_step_rank
        self.activation = activation
        self.layer_idx = layer_idx

        # conv1d as a column-parallel linear over the intermediate dim
        # (output_size == intermediate_size sharded across TP).
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=intermediate_size,
            bias=use_conv_bias,
            quant_config=None,
        )
        # Promote to depthwise-conv weight layout (D, 1, K) after load.
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # in_proj packs [x, gate], each of size intermediate_size.
        self.in_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=use_bias,
            quant_config=quant_config,
        )

        # x_proj: produces [dt, B, C] from x.  RowParallel because input
        # dim is intermediate (which is sharded), output is replicated.
        self.x_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=time_step_rank + 2 * ssm_state_size,
            bias=False,
            quant_config=None,
        )

        # dt_proj: time_step_rank -> intermediate (column-parallel).
        # Bias is added by the selective-scan kernel, so we keep it
        # separately and pass it through.
        self.dt_proj = ColumnParallelLinear(
            input_size=time_step_rank,
            output_size=intermediate_size,
            bias=True,
            quant_config=None,
        )

        tp_inter = intermediate_size // self.tp_size
        self.A = nn.Parameter(
            torch.empty(tp_inter, ssm_state_size, dtype=torch.float32),
        )
        self.D = nn.Parameter(torch.ones(tp_inter))

        # A_log is sharded along dim 0 with the -exp() transform applied
        # at load time so the kernel sees A directly.
        def _shard0_loader(param, loaded_weight):
            shard = param.data.size(0)
            param.data.copy_(
                loaded_weight.narrow(0, self.tp_rank * shard, shard).to(param.dtype),
            )

        def _A_loader(param, loaded_weight):
            shard = param.data.size(0)
            slice_ = loaded_weight.narrow(0, self.tp_rank * shard, shard).float()
            param.data.copy_(-torch.exp(slice_))

        self.A.weight_loader = _A_loader
        self.D.weight_loader = _shard0_loader

        self.out_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=use_bias, quant_config=quant_config,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _ssm_transform(self, x: torch.Tensor):
        """Compute (dt, B, C) from x via x_proj + dt_proj.

        x: [N, intermediate_per_rank]
        Returns:
          dt: [intermediate_per_rank, N]
          B:  [N, ssm_state_size]
          C:  [N, ssm_state_size]
        """
        ssm_params = self.x_proj(x)  # [N, dt_rank + 2*N_state]
        dt, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        # dt_proj (skip bias add - the kernel handles it).
        # ColumnParallelLinear adds bias inside; we want it raw, so we
        # call F.linear without the bias and pass the bias separately.
        import torch.nn.functional as F
        dt = F.linear(dt, self.dt_proj.weight, None).transpose(-2, -1)
        return dt, B, C

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Mamba v1 mixer forward.

        Reads cache state and per-batch metadata from the global Context.
        Mirrors vLLM ``MambaMixer.forward_impl`` and keeps the mixed
        batch in decode-first token order.

        ``hidden_states`` shape: [num_tokens, hidden_size]
        """
        ctx = get_context()
        mamba_state = getattr(ctx, "mamba_state", None)
        mamba_meta = getattr(ctx, "mamba_metadata", None)

        projected_states = self.in_proj(hidden_states).transpose(-2, -1)
        hidden_states_BC, gate = projected_states.chunk(2, dim=-2)

        if mamba_state is None or mamba_meta is None:
            # Profile / warmup path (no cache available).
            return self.out_proj(hidden_states_BC.transpose(-2, -1))

        # MambaStateManager allocates as ``[N, kernel-1, dim]`` so we
        # transpose to the kernel's expected ``[N, dim, kernel-1]`` view
        # which keeps ``stride(dim) == 1``.
        conv_state = mamba_state.conv_states[self.layer_idx].transpose(-1, -2)
        ssm_state = mamba_state.ssm_states[self.layer_idx]

        num_prefill_tokens = mamba_meta.num_prefill_tokens
        num_decode_tokens = mamba_meta.num_decode_tokens
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0
        num_actual = num_prefill_tokens + num_decode_tokens

        if has_prefill and has_decode:
            hidden_states_BC_d, hidden_states_BC_p = torch.split(
                hidden_states_BC[:, :num_actual],
                [num_decode_tokens, num_prefill_tokens],
                dim=-1,
            )
            gate_d, gate_p = torch.split(
                gate[:, :num_actual],
                [num_decode_tokens, num_prefill_tokens],
                dim=-1,
            )
        elif has_prefill:
            hidden_states_BC_p = hidden_states_BC[:, :num_prefill_tokens]
            gate_p = gate[:, :num_prefill_tokens]
            hidden_states_BC_d = None
            gate_d = None
        else:
            hidden_states_BC_d = hidden_states_BC[:, :num_decode_tokens]
            gate_d = gate[:, :num_decode_tokens]
            hidden_states_BC_p = None
            gate_p = None

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2),
        )
        ssm_outputs = []
        time_proj_bias = self.dt_proj.bias.float() if self.dt_proj.bias is not None else None

        if has_decode:
            conv_out_d = causal_conv1d_update(
                hidden_states_BC_d.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_meta.state_indices_d,
            ).transpose(0, 1)

            dt_d, B_d, C_d = self._ssm_transform(conv_out_d.transpose(-2, -1))
            out_d = torch.empty_like(hidden_states_BC_d.transpose(0, 1))
            selective_state_update(
                ssm_state,
                conv_out_d.transpose(0, 1),
                dt_d.transpose(0, 1),
                self.A,
                B_d,
                C_d,
                self.D,
                gate_d.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=mamba_meta.state_indices_d,
                out=out_d,
            )
            ssm_outputs.append(out_d.transpose(0, 1))

        if has_prefill:
            conv_out_p = causal_conv1d_fn(
                hidden_states_BC_p,
                conv_weights,
                self.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=mamba_meta.query_start_loc_p,
                cache_indices=mamba_meta.state_indices_p,
                has_initial_state=mamba_meta.has_initial_states_p,
                activation=self.activation,
                metadata=mamba_meta,
            )

            dt_p, B_p, C_p = self._ssm_transform(conv_out_p.transpose(-2, -1))
            scan_out_p = selective_scan_fn(
                conv_out_p,
                ssm_state,
                dt_p,
                self.A,
                B_p.transpose(-2, -1),
                C_p.transpose(-2, -1),
                self.D.float(),
                gate_p,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=mamba_meta.state_indices_p,
                has_initial_state=mamba_meta.has_initial_states_p,
                query_start_loc=mamba_meta.query_start_loc_p,
            )
            ssm_outputs.append(scan_out_p)

        scan_outputs = ssm_outputs[0] if len(ssm_outputs) == 1 else torch.cat(
            ssm_outputs,
            dim=-1,
        )
        return self.out_proj(scan_outputs.transpose(-2, -1))
