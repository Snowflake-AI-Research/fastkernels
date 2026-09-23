"""Pre-normalized masked encoders composed from existing ViT encoder blocks."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention
from fastkernels.tasks.baseline.L2.encoder_embeddings import (
    XLMRobertaEmbeddings, create_roberta_position_ids_from_input_ids,
)
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from .bert import MaskedLMHead
from .roberta import make_workloads


class PreNormAttention(VitEncoderAttention):
    """Use the existing attention operation's explicit native cuDNN backend."""

    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_attention_heads, qkv_bias=True)
        self.attn = DenseAttention(backend="cudnn")

    def forward(self, hidden_states, attn_mask=None):
        batch, length, width = hidden_states.shape
        query, key, value = self.qkv(hidden_states).view(
            batch, length, 3, self.num_heads, self.head_dim,
        ).unbind(2)
        context = self.attn(query, key, value, attn_mask=attn_mask)
        return self.proj(context.reshape(batch, length, width))


class PreNormMaskedLM(nn.Module):
    def __init__(self, config, *, normalize_embeddings):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embeddings = XLMRobertaEmbeddings(config)
        if not normalize_embeddings:
            self.embeddings.LayerNorm = nn.Identity()
        self.layers = nn.ModuleList([
            VitEncoderBlock(
                dim=config.hidden_size, num_heads=config.num_attention_heads,
                mlp_ratio=config.intermediate_size / config.hidden_size,
                qkv_bias=True, proj_bias=True, act_approximate="none",
                attn_drop=config.attention_probs_dropout_prob,
                proj_drop=config.hidden_dropout_prob, norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])
        for layer in self.layers:
            layer.attn = PreNormAttention(config)
        self.final_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False,
        )
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        positions = create_roberta_position_ids_from_input_ids(input_ids, self.padding_idx)
        hidden_states = self.embeddings.forward_with_token_type_ids(
            input_ids, positions, token_type_ids=token_type_ids,
        )
        mask = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask=mask)
        return self.lm_head(self.final_norm(hidden_states))


def build_pre_norm(config, device, dtype, *, normalize_embeddings):
    if (config.hidden_act != "gelu" or config.is_decoder or config.add_cross_attention
            or getattr(config, "position_embedding_type", "absolute") != "absolute"):
        raise ValueError("Pre-normalized RoBERTa requires the documented bidirectional GELU encoder")
    return PreNormMaskedLM(config, normalize_embeddings=normalize_embeddings).to(device=device, dtype=dtype).eval()


def load_pre_norm_weights(model, state_dict, *, prefix, attention_norm, mlp_norm, final_norm):
    layer_names = {
        "norm1": attention_norm, "norm2": mlp_norm,
        "attn.proj": "attention.output.dense",
        "mlp.fc1": "intermediate.dense", "mlp.fc2": "output.dense",
    }
    weights = {}
    for name in model.state_dict():
        if name.startswith("layers."):
            _, index, rest = name.split(".", 2)
            module, field = rest.rsplit(".", 1)
            source = f"{prefix}.encoder.layer.{index}."
            if module == "attn.qkv":
                weights[name] = torch.cat([
                    state_dict[source + f"attention.self.{projection}.{field}"]
                    for projection in ("query", "key", "value")
                ])
                continue
            source += layer_names[module] + "." + field
        elif name.startswith("embeddings."):
            source = prefix + "." + name.replace(".emb.weight", ".weight")
        elif name.startswith("final_norm."):
            source = prefix + "." + final_norm + "." + name.split(".")[-1]
        else:
            source = name.replace("lm_head.LayerNorm.", "lm_head.layer_norm.")
        weights[name] = state_dict[source]
    model.load_state_dict(weights, strict=True)


def build_from_config(config, device, dtype):
    return build_pre_norm(config, device, dtype, normalize_embeddings=True)


def load_state_dict_into(model, state_dict, config):
    load_pre_norm_weights(
        model, state_dict, prefix="roberta_prelayernorm",
        attention_norm="attention.LayerNorm", mlp_norm="intermediate.LayerNorm",
        final_norm="LayerNorm",
    )
