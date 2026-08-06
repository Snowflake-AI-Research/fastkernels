"""Online softmax merge for two attention partitions.

Wraps vLLM's fused CUDA kernel ``merge_attn_states`` which combines two
partial attention results (prefix and suffix) using their log-sum-exps so
the result is numerically equivalent to a single attention over the full
KV span.

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
from vllm.v1.attention.ops.merge_attn_states import (
    merge_attn_states as _merge_attn_states_impl,
)


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
