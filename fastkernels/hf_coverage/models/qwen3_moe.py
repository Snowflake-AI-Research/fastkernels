"""Qwen3 MoE causal LM reusing its existing normalized-attention decoder."""

from types import SimpleNamespace

import torch

from fastkernels.tasks.baseline.L3.qwen3_moe_decoder import Qwen3MoEDecoderLayer
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM

from .llama import make_workloads
from .qwen2_moe import library_config, load_qwen_moe_weights


def build_from_config(config, device, dtype):
    if config.attention_bias:
        raise ValueError("Qwen3-30B-A3B uses bias-free attention projections")
    model = LlamaForCausalLM(library_config(config, dtype, qkv_bias=False))
    # Pinned HF serializes this configuration alias as num_local_experts.
    layer_config = SimpleNamespace(**(config.to_dict() | {"num_experts": config.num_local_experts}))
    model.model.layers = torch.nn.ModuleList(
        Qwen3MoEDecoderLayer(layer_config, rotary_emb=model.model.rotary_emb)
        for _ in range(config.num_hidden_layers)
    )
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    load_qwen_moe_weights(model, state_dict, config, shared_expert=False, qk_norm=True)
