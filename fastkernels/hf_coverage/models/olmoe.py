"""OLMoE causal LM with full-projection Q/K norms and unrenormalized routing."""

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.jamba_moe import JambaMoE
from fastkernels.tasks.baseline.L4.mixtral import MixtralForCausalLM

from .llama import make_workloads
from .mixtral import library_config, load_moe_decoder_weights


class ProjectionRMSNorm(RMSNorm):
    """Normalize the complete projection through the carrier's per-head view."""

    def forward(self, hidden_states):
        shape = hidden_states.shape
        return super().forward(hidden_states.reshape(-1, self.hidden_size)).reshape(shape)


def build_from_config(config, device, dtype):
    if config.attention_bias or config.clip_qkv is not None or config.norm_topk_prob:
        raise ValueError("The OLMoE checkpoint uses bias-free attention, no clipping, and raw top-k probabilities")
    if config.num_attention_heads != config.num_key_value_heads:
        raise ValueError("The selected OLMoE checkpoint preserves multi-head attention")
    model = MixtralForCausalLM(library_config(config, dtype, config.num_experts))
    for layer in model.model.layers:
        layer.block_sparse_moe = JambaMoE(
            config.hidden_size, config.intermediate_size,
            config.num_experts, config.num_experts_per_tok,
        )
        layer.self_attn.q_norm = ProjectionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        layer.self_attn.k_norm = ProjectionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    load_moe_decoder_weights(model, state_dict, config, qk_norm=True)
