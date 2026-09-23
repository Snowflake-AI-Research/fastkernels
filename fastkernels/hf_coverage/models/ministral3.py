"""Ministral3 YaRN decoder with its enabled position-dependent query scaling."""

from fastkernels.hf_coverage.models.llama import load_state_dict_into, make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.ministral3_position import MinistralPositionAttention, MinistralYarn
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if config.hidden_act != "silu" or config.tie_word_embeddings or config.sliding_window is not None:
        raise ValueError("Selected Ministral3 uses full attention, SiLU and an untied head")
    if rope["rope_type"] != "yarn":
        raise ValueError("Selected Ministral3 uses YaRN")
    model = LlamaForCausalLM(decoder_config(config, dtype))
    rotary = MinistralYarn(model.config.head_dim, rope["original_max_position_embeddings"],
                           rope["rope_theta"], rope["factor"], beta_fast=rope["beta_fast"],
                           beta_slow=rope["beta_slow"], mscale=rope["mscale"],
                           mscale_all_dim=rope["mscale_all_dim"], is_neox_style=True)
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn = MinistralPositionAttention(
            config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
            model.config.head_dim, rotary_emb=rotary,
            floor_scale=rope["original_max_position_embeddings"], attn_scale=rope["llama_4_scaling_beta"],
        )
        layer.self_attn.attn_temperature_tuning = True
    return model.to(device=device, dtype=dtype).eval()
