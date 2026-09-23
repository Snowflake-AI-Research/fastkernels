"""MT5 conditional generation with existing gated T5 operations and unscaled head."""

import torch

from fastkernels.infra.tp import _tp_size

from .t5 import T5ForConditionalGeneration, load_state_dict_into, make_workloads


def _validate_config(config, dtype):
    if (_tp_size() != 1 or config.feed_forward_proj != "gated-gelu"
            or not config.is_gated_act or config.dense_act_fn != "gelu_new"
            or not config.tie_word_embeddings or not config.use_cache
            or not config.is_encoder_decoder):
        raise ValueError("Multilingual T5-small requires gated GELU, native tied embeddings and default seq2seq caching")
    if config.num_heads * config.d_kv * 4 != config.d_model * 3:
        raise ValueError("Multilingual T5-small preserves attention width equal to three quarters of model width")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Multilingual T5 coverage checks FP32 and BF16 loading semantics")


def build_from_config(config, device, dtype):
    _validate_config(config, dtype)
    return T5ForConditionalGeneration(config, output_scale=1.0).to(device=device, dtype=dtype).eval()
