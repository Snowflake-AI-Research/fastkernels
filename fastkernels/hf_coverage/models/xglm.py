"""XGLM reuses the pre-normalized decoder with existing sinusoidal positions."""

from types import SimpleNamespace

import torch
from fastkernels.tasks.baseline.L1.dp3_sinusoidal_pos_emb import DP3SinusoidalPosEmb
from .biogpt import BioGptForCausalLM, load_decoder_weights
from .mvp import EagerAttention
from .qwen2_precision import DenseCachedAttention
from .llama import make_workloads


def build_from_config(config, device, dtype):
    if config.activation_function != 'gelu' or config.add_cross_attention or not config.use_cache:
        raise ValueError('XGLM case retains its cached GELU decoder without cross attention')
    adapted = SimpleNamespace(**(dict(config) | {
        'hidden_size': config.d_model, 'num_hidden_layers': config.num_layers,
        'num_attention_heads': config.attention_heads, 'intermediate_size': config.ffn_dim,
    }))
    model = BioGptForCausalLM(adapted)
    for layer in model.model.layers:
        layer.self_attn.attn = DenseCachedAttention(
            config.attention_heads, config.attention_heads,
            config.d_model // config.attention_heads,
        )
        layer.self_attn.attn.attention = EagerAttention()
    with torch.no_grad():
        # HF constructs this fixed sinusoidal table in FP32 on the CPU.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            table = DP3SinusoidalPosEmb(config.d_model)(
                torch.arange(config.max_position_embeddings + 2, device="cpu"),
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        if config.pad_token_id is not None:
            table[config.pad_token_id] = 0
        model.model.embed_positions.emb.weight.copy_(table)
        model.model.embed_positions.emb.weight.requires_grad_(False)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    load_decoder_weights(model, state_dict, config, prefix='model', head='lm_head', sinusoidal=True)
