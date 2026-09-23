"""Marian post-norm encoder-decoder with its frozen sinusoidal lookup tables."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.hf_coverage.runner import seq2seq_continuation_workloads
from .bart import BartForConditionalGeneration
from .plbart import make_workloads as forward_workloads


def build_from_config(config, device, dtype):
    if not config.share_encoder_decoder_embeddings:
        raise ValueError("The selected Marian checkpoint shares encoder/decoder embeddings")
    if config.activation_function != "swish" or not config.tie_word_embeddings or not config.use_cache:
        raise ValueError("Selected Marian checkpoint requires swish, tied output weights and caching")
    model = BartForConditionalGeneration(config)
    for stack in (model.encoder, model.decoder):
        stack.layernorm_embedding = nn.Identity()
        stack.position_offset = 0
        stack.embed_positions = Embedding(config.max_position_embeddings, config.d_model)
        stack.embed_positions.emb.weight.requires_grad_(False)
        for layer in (stack.layers if stack.is_decoder else stack.layers.layer):
            layer.intermediate.intermediate_act_fn = SiLU()
            layer.attention.self.attn = DenseAttention(backend="cudnn")
            if stack.is_decoder:
                layer.cross_attention.attention = DenseAttention(backend="cudnn")
    return model.to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs)
    return forward_workloads(model, inputs, config)


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    """Map post/pre-norm BART carriers without inventing missing HF parameters."""
    weights, consumed = {}, set()
    for name, parameter in model.state_dict().items():
        source = name.replace(".emb.weight", ".weight")
        if source == "final_logits_bias":
            pass
        elif source.startswith("lm_head."):
            pass
        else:
            source = "model." + source
        source = source.replace("encoder.layers.layer.", "encoder.layers.")
        source = source.replace(".attn.qkv.", ".self_attn.qkv.")
        source = source.replace(".attn.proj.", ".self_attn.out_proj.")
        source = source.replace(".norm1.", ".self_attn_layer_norm.")
        source = source.replace(".norm2.", ".final_layer_norm.")
        source = source.replace(".mlp.fc1.", ".fc1.")
        source = source.replace(".mlp.fc2.", ".fc2.")
        source = source.replace(".attention.self.qkv.", ".self_attn.qkv.")
        source = source.replace(".attention.output.dense.", ".self_attn.out_proj.")
        source = source.replace(".attention.output.LayerNorm.", ".self_attn_layer_norm.")
        source = source.replace(".intermediate.dense.", ".fc1.")
        source = source.replace(".output.dense.", ".fc2.")
        source = source.replace(".output.LayerNorm.", ".final_layer_norm.")
        source = source.replace(".cross_attention.norm.", ".encoder_attn_layer_norm.")
        source = source.replace(".cross_attention.", ".encoder_attn.")
        if ".embed_positions." in source and getattr(model, "generated_positions", False):
            weights[name] = parameter
            continue
        names = ([source.replace(".qkv.", f".{projection}_proj.") for projection in ("q", "k", "v")]
                 if ".qkv." in source else [source])
        value = torch.cat([state_dict[key] for key in names]) if len(names) == 3 else state_dict[source]
        if value.shape != parameter.shape:
            raise ValueError(f"Marian weight shape mismatch: {name}")
        weights[name] = value
        consumed.update(names)
    if consumed != set(state_dict):
        raise ValueError(f"Marian unmapped state: {sorted(set(state_dict) - consumed)}")
    model.load_state_dict(weights)
