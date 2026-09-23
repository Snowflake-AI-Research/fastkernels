"""Original Cohere: shared parallel blocks with interleaved RoPE on every layer."""

import torch

from .cohere2 import Cohere2ForCausalLM, load_state_dict_into, make_workloads
from .olmo2 import NativeFP32Rotary


def build_from_config(config, device, dtype):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    if (config.hidden_act != "silu" or config.attention_bias or config.use_qk_norm
            or not config.tie_word_embeddings or not config.use_cache
            or config.rope_parameters["rope_type"] != "default"
            or head_dim % 2 or config.hidden_size != config.num_attention_heads * head_dim
            or config.num_attention_heads % config.num_key_value_heads
            or getattr(config, "sliding_window", None) is not None
            or getattr(config, "layer_types", None) is not None):
        raise ValueError("Cohere coverage requires cached full attention, no Q/K norm, bias-free SiLU, "
                         "tied embeddings and default interleaved RoPE")
    model = Cohere2ForCausalLM(config, rotary_on_all_layers=True).to(device=device, dtype=dtype).eval()
    with torch.device("cpu"):
        model.rotary = NativeFP32Rotary(head_dim, config.max_position_embeddings,
                                      config.rope_parameters["rope_theta"], device)
    # Native coefficients round to model dtype before the FP32 Q/K rotation.
    model.rotary.cos_sin_cache = model.rotary.cos_sin_cache.to(dtype).float()
    return model
