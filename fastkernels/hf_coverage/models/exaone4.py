"""EXAONE-4 post-normalization with three local RoPE blocks per global NoPE block."""

from dataclasses import replace

import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM

from .llama import make_workloads as llama_workloads


class PostNormBackbone(nn.Module):
    """Each layer completes its residual additions before the final norm."""

    def __init__(self, backbone):
        super().__init__()
        self.embed_tokens, self.layers = backbone.embed_tokens, backbone.layers
        self.rotary_emb, self.norm = backbone.rotary_emb, backbone.norm

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden, _ = layer(positions, hidden)
        return self.norm(hidden)


class ExaoneLayer(nn.Module):
    def __init__(self, config, rotary, local):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, rotary_emb=rotary, qk_norm=True, rms_norm_eps=config.rms_norm_eps,
            nope=not local, sliding_window=config.sliding_window if local else None,
        )
        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden, residual=None):
        if residual is not None:
            hidden = hidden + residual
        hidden = hidden + self.post_attention_layernorm(self.self_attn(positions, hidden))
        hidden = hidden + self.post_feedforward_layernorm(self.mlp(hidden))
        return hidden, None


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or config.hidden_act != "silu" or config.tie_word_embeddings
            or not config.use_cache or rope["rope_type"] != "llama3"
            or config.sliding_window is None or set(config.layer_types) != {"sliding_attention", "full_attention"}):
        raise ValueError("The selected EXAONE-4 checkpoint uses hybrid local-RoPE/global-NoPE attention with Llama3 frequency scaling")
    adapted = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim, vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings, rms_norm_eps=config.rms_norm_eps,
        rope_theta=rope["rope_theta"], rope_scaling_factor=rope["factor"],
        rope_low_freq_factor=rope["low_freq_factor"], rope_high_freq_factor=rope["high_freq_factor"],
        rope_original_max_position_embeddings=rope["original_max_position_embeddings"], dtype=dtype,
    )
    # Construct only the shared embedding/rotary/head shell: every layer below
    # uses EXAONE's post-normalization and local/global attention layout.
    model = LlamaForCausalLM(replace(adapted, num_hidden_layers=0))
    model.config = adapted
    model.model.layers = nn.ModuleList([ExaoneLayer(config, model.model.rotary_emb, kind == "sliding_attention")
                                        for kind in config.layer_types])
    model.model = PostNormBackbone(model.model)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = dict(state_dict)
    mapped["model.embed_tokens.embedding_op.emb.weight"] = mapped.pop("model.embed_tokens.weight")
    mapped["lm_head.embedding_op.emb.weight"] = mapped.pop("lm_head.weight")
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        mapped[prefix + "self_attn.qkv_proj.weight"] = torch.cat([
            mapped.pop(prefix + f"self_attn.{part}_proj.weight") for part in ("q", "k", "v")])
        mapped[prefix + "mlp.gate_up_proj.weight"] = torch.cat([
            mapped.pop(prefix + f"mlp.{part}_proj.weight") for part in ("gate", "up")])
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    windows = [config.sliding_window if kind == "sliding_attention" else None
               for kind in config.layer_types]
    return llama_workloads(model, inputs, config, case=case, cache_windows=windows)
