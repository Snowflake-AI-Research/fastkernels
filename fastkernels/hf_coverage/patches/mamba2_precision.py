"""Select HF-compatible rounding in the existing Mamba2 operations.

The selected native kernels use one head per prefix-sum tile and four rows per
decode state-update tile. FastKernels' other tile choices change FP32 reduction
order. This adaptation selects those launch settings without changing kernels,
scan stages or state layout. Convolution retains FP32 through existing SiLU;
gated normalization uses its existing unfused implementation. Parent calls are
hardcoded, so the small amount of scan/mixer wiring is local to this patch.
"""

import torch
import triton

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1 import mamba_chunk_scan as scan
from fastkernels.tasks.baseline.L1.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fastkernels.tasks.baseline.L1.mamba_ssm import (
    _get_default_ssm_launch_config, override_ssm_config, selective_state_update,
)
from fastkernels.tasks.baseline.L1.silu import SiLU

_silu = SiLU()


def _chunk_cumsum(dt, a, bias, metadata):
    """Launch the existing kernel with the native one-head scan tile."""
    heads = dt.shape[1]
    chunks = metadata.cu_chunk_seqlen_p.shape[0] - 1
    chunk_size = metadata.chunk_size
    delta = torch.empty(heads, chunks, chunk_size, device=dt.device, dtype=torch.float32)
    cumulative = torch.empty_like(delta)
    # Bypass only autotuning, whose candidate list omits BLOCK_SIZE_H=1.
    scan._chunk_cumsum_fwd_kernel.fn[(chunks, heads)](
        dt_ptr=dt, A_ptr=a, dt_bias_ptr=bias, dt_out_ptr=delta, dA_cumsum_ptr=cumulative,
        cu_chunk_seqlens_ptr=metadata.cu_chunk_seqlen_p,
        nheads=heads, chunk_size=chunk_size, dt_min=0.0, dt_max=float("inf"),
        stride_dt_seqlen=dt.stride(0), stride_dt_head=dt.stride(1),
        stride_A_head=a.stride(0), stride_dt_bias_head=bias.stride(0),
        stride_dt_out_head=delta.stride(0), stride_dt_out_chunk=delta.stride(1),
        stride_dt_out_csize=delta.stride(2), stride_dA_cs_head=cumulative.stride(0),
        stride_dA_cs_chunk=cumulative.stride(1), stride_dA_cs_csize=cumulative.stride(2),
        DT_SOFTPLUS=True, HAS_DT_BIAS=True, BLOCK_SIZE_H=1,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size), num_warps=4, num_stages=3,
    )
    return cumulative, delta


def _prefill_scan(mixer, x, b, c, dt, state, metadata):
    """Reuse the parent's complete variable-length scan stages."""
    cumulative, delta = _chunk_cumsum(dt, mixer.A, mixer.dt_bias, metadata)
    chunks = metadata.cu_chunk_seqlen_p
    states = scan._chunk_state_fwd(b, x, delta, cumulative, chunks, states_in_fp32=True)
    state_shape = states.shape
    states = scan._state_passing_fwd(
        states.flatten(-2), cumulative, metadata.last_chunk_indices_p, out_dtype=state.dtype,
    ).view(state_shape)
    cb = scan._bmm_chunk_fwd(c, b, mixer.chunk_size, chunks, output_dtype=torch.float32)
    output = torch.empty_like(x)
    scan._chunk_scan_fwd(
        cb, x, delta, cumulative, c, states, chunks, output, metadata.seq_idx_p, D=mixer.D,
    )
    state[metadata.state_indices_p] = states[metadata.last_chunk_indices_p]
    return output


def mamba2_forward(mixer, hidden):
    """The audit's separate prefill/decode calls with unchanged learned ops."""
    context = get_context()
    state, metadata = context.mamba_state, context.mamba_metadata
    prefill = metadata.num_prefill_tokens > 0
    if prefill and metadata.num_decode_tokens:
        raise ValueError("The audit evaluates prefill and decode separately")
    projected = mixer.in_proj(hidden)
    gate = projected[..., :mixer.tped_intermediate_size]
    xbc, dt = torch.split(projected[..., mixer.tped_intermediate_size:],
                          [mixer.tped_conv_size, mixer.tped_dt_size], dim=-1)
    conv_state = state.conv_states[mixer.layer_idx].transpose(-1, -2)
    recurrent = state.ssm_states[mixer.layer_idx]
    wide_state = conv_state.float()
    if prefill:
        convolved = causal_conv1d_fn(
            xbc.transpose(0, 1).float(), mixer.conv_weights, mixer.conv1d.bias,
            activation=None, conv_states=wide_state,
            has_initial_state=metadata.has_initial_states_p,
            cache_indices=metadata.state_indices_p, query_start_loc=metadata.query_start_loc_p,
        ).transpose(0, 1)
    else:
        convolved = causal_conv1d_update(
            xbc.float(), wide_state, mixer.conv_weights, mixer.conv1d.bias,
            activation=None, conv_state_indices=metadata.state_indices_d, null_block_id=-1,
        )
    conv_state.copy_(wide_state)
    convolved = _silu(convolved).to(xbc.dtype)
    x, b, c = mixer._split_BC(convolved)
    tokens = x.shape[0]
    heads = mixer.num_heads // mixer.tp_size
    groups = mixer.n_groups // mixer.tp_size
    x = x.view(tokens, heads, mixer.head_dim)
    b, c = b.view(tokens, groups, -1), c.view(tokens, groups, -1)
    if prefill:
        output = _prefill_scan(mixer, x, b, c, dt, recurrent, metadata)
    else:
        output = torch.empty_like(x)
        a = mixer.A[:, None, None].expand(-1, mixer.head_dim, mixer.ssm_state_size).float()
        delta = dt[:, :, None].expand(-1, -1, mixer.head_dim)
        bias = mixer.dt_bias[:, None].expand(-1, mixer.head_dim)
        d = mixer.D[:, None].expand(-1, mixer.head_dim)
        # Existing scoped configuration API restores the previous setting even
        # on error; no global kernel or autotuner replacement is installed.
        with override_ssm_config(_get_default_ssm_launch_config(mixer.ssm_state_size, False)):
            selective_state_update(
                recurrent, x, delta, a, b, c, d, dt_bias=bias, dt_softplus=True,
                state_batch_indices=metadata.state_indices_d, null_block_id=-1, out=output,
            )
    normalized = mixer.norm.forward_native(output.reshape(tokens, -1), gate)
    return mixer.out_proj(normalized)
