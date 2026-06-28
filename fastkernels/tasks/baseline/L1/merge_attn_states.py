"""Online softmax merge for two attention partitions.

Wraps vLLM's fused CUDA kernel ``merge_attn_states`` which combines two
partial attention results (prefix and suffix) using their log-sum-exps so
the result is numerically equivalent to a single attention over the full
KV span.  Falls back to a pure PyTorch implementation when the vLLM
kernel is unavailable.

Interface matches ``vllm.v1.attention.ops.merge_attn_states``:

    merge(output, output_lse, prefix_output, prefix_lse,
          suffix_output, suffix_lse)

where ``output`` and ``output_lse`` are written in-place.  ``output_lse``
may be ``None`` if the caller does not need the merged LSE (final
reduction step).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_merge_attn_states_impl = None
try:
    from vllm.v1.attention.ops.merge_attn_states import (
        merge_attn_states as _merge_attn_states_impl,
    )
except ImportError:
    _merge_attn_states_impl = None


def _merge_attn_states_pytorch(
    output: torch.Tensor,
    output_lse: torch.Tensor | None,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> None:
    """Reference PyTorch implementation of online softmax merge.

    Follows the standard log-sum-exp trick:

        m      = max(l_p, l_s)
        w_p    = exp(l_p - m);  w_s = exp(l_s - m)
        out    = (w_p * o_p + w_s * o_s) / (w_p + w_s)
        lse_m  = m + log(w_p + w_s)
    """
    p_lse = prefix_lse.to(torch.float32)
    s_lse = suffix_lse.to(torch.float32)
    m = torch.maximum(p_lse, s_lse)
    wp = torch.exp(p_lse - m)
    ws = torch.exp(s_lse - m)
    denom = wp + ws

    wp_b = wp.transpose(0, 1).unsqueeze(-1)
    ws_b = ws.transpose(0, 1).unsqueeze(-1)
    denom_b = denom.transpose(0, 1).unsqueeze(-1)

    merged = (wp_b * prefix_output.to(torch.float32)
              + ws_b * suffix_output.to(torch.float32)) / denom_b
    output.copy_(merged.to(output.dtype))

    if output_lse is not None:
        output_lse.copy_((m + torch.log(denom)).to(output_lse.dtype))


class MergeAttnStates(nn.Module):
    """Online softmax merge of two attention partitions."""

    def forward(
        self,
        output: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: torch.Tensor | None = None,
    ) -> None:
        if _merge_attn_states_impl is not None:
            kwargs = dict(
                output=output,
                prefix_output=prefix_output,
                prefix_lse=prefix_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
            if output_lse is not None:
                kwargs["output_lse"] = output_lse
            _merge_attn_states_impl(**kwargs)
            return
        _merge_attn_states_pytorch(
            output, output_lse,
            prefix_output, prefix_lse,
            suffix_output, suffix_lse,
        )
