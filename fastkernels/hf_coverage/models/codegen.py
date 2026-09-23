"""CodeGen's grouped Q/V/K layout and FP32 rotary/query computation."""

import torch

from .gptj import Float32ScoreAttention, GPTJRotary, build_from_config as build_gptj, make_workloads
from .gptj import load_state_dict_into as load_gptj


class CodeGenRotary(GPTJRotary):
    def forward(self, positions, query, key):
        dtype = key.dtype
        query, key = super().forward(positions, query.float(), key.float())
        # HF retains the FP32 query but rounds keys when updating its cache.
        return query, key.to(dtype)




def build_from_config(config, device, dtype):
    model = build_gptj(config, device, dtype, separate_qkv=False)
    head_dim = config.n_embd // config.n_head
    # Construct after the model conversion so this fixed angle table stays FP32.
    # Native HF prepares these fixed trigonometric values on CPU. CUDA's
    # transcendental rounding changes a few subsequent BF16 cached keys.
    with torch.device("cpu"):
        rotary = CodeGenRotary(head_dim, config.rotary_dim, config.n_positions, 10000.0)
    rotary = rotary.to(device=device)
    rotary.rotary.is_neox_style = False
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
        layer.self_attn.attn = Float32ScoreAttention(config.n_head, head_dim).to(device=device)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped = dict(state_dict)
    for index in range(config.n_layer):
        prefix = f"transformer.h.{index}.attn."
        packed = mapped.pop(prefix + "qkv_proj.weight")
        # HF hardcodes four logical shards, each storing Q|V|K.
        packed = packed.view(4, 3, config.n_embd // 4, config.n_embd)
        for slot, name in enumerate(("q", "v", "k")):
            mapped[prefix + name + "_proj.weight"] = packed[:, slot].reshape(config.n_embd, config.n_embd)
    load_gptj(model, mapped, config)
