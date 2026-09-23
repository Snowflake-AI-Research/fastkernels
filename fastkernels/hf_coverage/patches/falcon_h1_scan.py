"""Retain Falcon-H1's pre-scan time-step rounding in Mamba2Mixer.

Parent: Mamba2Mixer.conv_ssm_forward. HF prefill evaluates softplus(dt +
dt_bias) in model dtype before its scan; the parent scan normally fuses this
pointwise transform in FP32. Only that transform and its rounding boundary
move outside the unchanged varlen scan. Convolution, recurrence, state layout,
and the parent's cached single-token decode path are retained.
"""

import torch
from torch.nn import functional as F

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.causal_conv1d import causal_conv1d_fn
from fastkernels.tasks.baseline.L1.mamba_chunk_scan import mamba_chunk_scan_combined_varlen
from fastkernels.tasks.baseline.L2.mamba2_mixer import Mamba2Mixer


class FalconH1Mixer(Mamba2Mixer):
    def conv_ssm_forward(self, projected_states, output):
        context = get_context()
        metadata = context.mamba_metadata
        state = context.mamba_state
        if metadata is None or state is None or metadata.num_prefill_tokens == 0:
            return super().conv_ssm_forward(projected_states, output)
        if metadata.num_decode_tokens:
            raise ValueError('The selected Falcon-H1 workload separates prefill and decode')
        count = metadata.num_prefill_tokens
        hidden, dt = projected_states[:count, self.tped_intermediate_size:].split(
            [self.tped_conv_size, self.tped_dt_size], dim=-1,
        )
        recurrent = state.ssm_states[self.layer_idx]
        hidden = causal_conv1d_fn(
            hidden.transpose(0, 1), self.conv_weights, self.conv1d.bias,
            activation=self.activation,
            conv_states=state.conv_states[self.layer_idx].transpose(-1, -2),
            has_initial_state=metadata.has_initial_states_p,
            cache_indices=metadata.state_indices_p, query_start_loc=metadata.query_start_loc_p,
        ).transpose(0, 1)[:count]
        x, b, c = self._split_BC(hidden)
        initial = None
        if metadata.has_initial_states_p is not None and metadata.prep_initial_states:
            initial = torch.where(metadata.has_initial_states_p[:, None, None, None],
                                  recurrent[metadata.state_indices_p], 0)
        # Both the addition and softplus result retain HF's model-dtype boundary.
        time_step = F.softplus(dt + self.dt_bias)
        groups = self.n_groups // self.tp_size
        final = mamba_chunk_scan_combined_varlen(
            x.view(count, self.num_heads // self.tp_size, self.head_dim), time_step, self.A,
            b.view(count, groups, -1), c.view(count, groups, -1), chunk_size=self.chunk_size,
            D=self.D, z=None, dt_bias=None, dt_softplus=False,
            seq_idx=metadata.seq_idx_p, cu_seqlens=metadata.query_start_loc_p,
            cu_chunk_seqlens=metadata.cu_chunk_seqlen_p,
            last_chunk_indices=metadata.last_chunk_indices_p,
            initial_states=initial, return_intermediate_states=False,
            dt_limit=(0.0, float('inf')),
            out=output[:count].view(count, -1, self.head_dim), state_dtype=recurrent.dtype,
        )
        recurrent[metadata.state_indices_p] = final
