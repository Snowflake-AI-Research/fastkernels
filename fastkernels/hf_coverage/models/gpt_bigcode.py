"""GPTBigCode's default cached multi-query decoder with learned positions."""

from .gpt2 import build_decoder, load_decoder_weights, make_workloads
from .qwen2_precision import DenseCachedAttention


def build_from_config(config, device, dtype):
    if (not config.multi_query or config.num_key_value_heads != 1
            or not config.attention_softmax_in_fp32 or not config.scale_attention_softmax_in_fp32):
        raise ValueError("This case preserves the documented GPTBigCode checkpoint's multi-query attention settings")
    model = build_decoder(config, device, dtype, kv_heads=1, activation="gelu_pytorch_tanh")
    for layer in model.model.layers:
        layer.self_attn.attn = DenseCachedAttention(config.n_head, 1, config.n_embd // config.n_head)
    return model


def load_state_dict_into(model, state_dict, config):
    del config
    load_decoder_weights(model, state_dict, transposed_projections=False)
