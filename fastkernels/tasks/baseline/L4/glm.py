"""GLM-5.2 (``glm_moe_dsa``) model.

GLM-5.2 (``GlmMoeDsaForCausalLM``) is a pure config variant of DeepSeek-V3.2:
in vLLM ``GlmMoeDsaForCausalLM`` subclasses ``DeepseekV2ForCausalLM`` with an
empty body, sharing the entire MLA + DSA (sparse attention) + MoE stack. This
module mirrors that -- GLM's differences (plain RoPE, interleaved indexer RoPE,
fp32 MoE routing, index-topk sharing) are all encoded as config *values* parsed
by ``DeepSeekV3Config.from_pretrained``, so GLM only needs to name itself as a
distinct architecture / entry point and reuse the DeepSeek implementation rather
than fork it.
"""

from __future__ import annotations

from .deepseek import DeepSeekV3Config, DeepSeekV3ForCausalLM


__targets__ = ["GlmMoeDsaForCausalLM"]


class GlmMoeDsaConfig(DeepSeekV3Config):
    """GLM-5.2 config.

    Same schema as ``DeepSeekV3Config``; the GLM-specific fields
    (``indexer_rope_interleave``, ``index_topk_freq``, ``moe_router_dtype``,
    plain-RoPE ``rope_parameters``) are populated from the checkpoint by the
    inherited ``from_pretrained``.
    """


class GlmMoeDsaForCausalLM(DeepSeekV3ForCausalLM):
    """GLM-5.2 causal LM: DeepSeek-V3.2's MLA + DSA + MoE stack on a GLM config."""
