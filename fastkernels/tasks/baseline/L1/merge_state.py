"""Softmax-LSE state merge for cascade attention.

Combines two attention outputs ``(o_a, lse_a)`` and ``(o_b, lse_b)`` computed
over disjoint key sets into the exact result of attention over the union, using
the standard log-sum-exp merging trick:

    out = (exp(lse_a) * o_a + exp(lse_b) * o_b) / (exp(lse_a) + exp(lse_b))
    lse = log(exp(lse_a) + exp(lse_b))

Numerically stable via subtracting ``max(lse_a, lse_b)`` first. ``+inf`` lse
inputs (FA3's convention for "no valid keys") are mapped to ``-inf`` so the
corresponding side contributes zero.

Direct port of sglang's ``merge_state_triton`` (Apache-2.0).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_state_kernel(
    output,         # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,     # [NUM_TOKENS, NUM_HEADS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,     # [NUM_TOKENS, NUM_HEADS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,     # [NUM_TOKENS, NUM_HEADS]
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    LSE_HEAD_MAJOR: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    if LSE_HEAD_MAJOR:
        lse_offset = head_idx * num_tokens + token_idx
    else:
        lse_offset = token_idx * num_heads + head_idx
    p_lse = tl.load(prefix_lse + lse_offset)
    s_lse = tl.load(suffix_lse + lse_offset)
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    out_se = tl.exp(p_lse) + tl.exp(s_lse)

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + token_idx * num_heads + head_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )

    p_scale = tl.exp(p_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out,
        mask=head_mask,
    )


def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
    lse_head_major: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge ``(prefix_output, prefix_lse)`` and ``(suffix_output, suffix_lse)``.

    Shapes (token-major):
      - ``*_output``: ``[num_tokens, num_heads, head_size]``
      - ``*_lse``:    ``[num_tokens, num_heads]`` (NOT FA3's transposed layout)

    Returns ``(merged_output, merged_lse_or_None)``.
    """
    if output is None:
        output = torch.empty_like(prefix_output)
    write_lse = output_lse is not None
    if output_lse is None:
        # Dummy pointer: the Triton kernel does not touch it when
        # ``OUTPUT_LSE`` is false. Avoid allocating/writing LSE for tree
        # attention, which only consumes the merged output.
        output_lse = prefix_lse

    num_tokens = output.shape[0]
    num_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)

    _merge_state_kernel[(num_tokens, num_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        head_size,
        padded_head_size,
        write_lse,
        lse_head_major,
    )
    return output, output_lse if write_lse else None
