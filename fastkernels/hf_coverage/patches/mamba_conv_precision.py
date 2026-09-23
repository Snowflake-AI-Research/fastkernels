"""Preserve FP32 convolution output through SiLU before the BF16 store.

Parent: the existing MambaMixer and causal-convolution operations. Their fused
SiLU differs from HF at a few BF16 rounding boundaries. Reusing the convolution
without activation, followed by the existing FP32 SiLU, matches the measured HF
path. Projections, convolution reductions, scan, state layout and updates remain
the parent's operations. All conversions and state copies execute inside forward.
"""

import torch

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L2.mamba_mixer import (
    MambaMixer, _padded_token_cat, causal_conv1d_fn, causal_conv1d_update,
    selective_scan_fn, selective_state_update,
)


class MambaConvPrecision(MambaMixer):
    """The audit's separate prefill/decode workloads, with native rounding."""

    def __init__(self, config, layer_idx):
        super().__init__(
            hidden_size=config.hidden_size, ssm_state_size=config.state_size,
            conv_kernel_size=config.conv_kernel, intermediate_size=config.intermediate_size,
            time_step_rank=config.time_step_rank, use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias, activation=config.hidden_act, layer_idx=layer_idx,
        )
        if config.hidden_act != "silu":
            raise ValueError("This precision adaptation is for the selected Mamba SiLU path")
        self.silu = SiLU()

    def forward(self, hidden):
        context = get_context()
        state, metadata = context.mamba_state, context.mamba_metadata
        prefill = metadata.num_prefill_tokens > 0
        if prefill and metadata.num_decode_tokens:
            raise ValueError("The audit evaluates prefill and decode separately")
        x, gate = self._project_input(hidden, split=prefill)
        conv_state = state.conv_states[self.layer_idx].transpose(-1, -2)
        recurrent = state.ssm_states[self.layer_idx]
        # The existing convolution casts inputs to its state dtype. Both must
        # be FP32 to avoid rounding the pre-activation result to BF16.
        wide_state = conv_state.float()
        weight = self.conv1d.weight.view(self.conv1d.weight.shape[0], -1)
        bias = self.dt_proj.bias.float() if self.dt_proj.bias is not None else None
        if prefill:
            convolved = causal_conv1d_fn(
                x.float(), weight, self.conv1d.bias, conv_states=wide_state,
                query_start_loc=metadata.query_start_loc_p, cache_indices=metadata.state_indices_p,
                has_initial_state=metadata.has_initial_states_p, activation=None, metadata=metadata,
            )
            conv_state.copy_(wide_state)
            convolved = self.silu(convolved).to(x.dtype)
            delta, b, c = self._ssm_transform(convolved.transpose(-2, -1), dt_contiguous=True)
            output = selective_scan_fn(
                convolved, recurrent, delta, self.A, b.transpose(-2, -1), c.transpose(-2, -1),
                self.D.float(), gate, bias, delta_softplus=True,
                cache_indices=metadata.state_indices_p, has_initial_state=metadata.has_initial_states_p,
                query_start_loc=metadata.query_start_loc_p,
            )
        else:
            convolved = causal_conv1d_update(
                x.transpose(0, 1).float(), wide_state, weight, self.conv1d.bias,
                activation=None, conv_state_indices=metadata.state_indices_d, null_block_id=-1,
            )
            conv_state.copy_(wide_state)
            convolved = self.silu(convolved).to(x.dtype).transpose(0, 1)
            delta, b, c = self._ssm_transform(convolved.transpose(-2, -1))
            output = torch.empty_like(x.transpose(0, 1))
            selective_state_update(
                recurrent, convolved.transpose(0, 1), delta.transpose(0, 1), self.A,
                b, c, self.D, dt_bias=bias, z=gate.transpose(0, 1), dt_softplus=True,
                state_batch_indices=metadata.state_indices_d, null_block_id=-1, out=output,
            )
            output = output.transpose(0, 1)
        output, tokens = _padded_token_cat([output])
        return self.out_proj(output.transpose(-2, -1))[:tokens]
