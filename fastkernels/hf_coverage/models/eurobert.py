"""EuroBertForMaskedLM using existing bidirectional LLaDA transformer blocks."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L3.llada_block import LLaDABlock

from ..runner import Workload


class EuroBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # HF initializes fixed rotary frequencies on CPU. GPU construction
        # changes coefficients near BF16 rounding boundaries.
        with torch.device("cpu"):
            self.rotary_emb = RotaryEmbedding(
                config.head_dim, config.max_position_embeddings, config.rope_parameters["rope_theta"],
            )
        block_config = SimpleNamespace(
            d_model=config.hidden_size, n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
            mlp_hidden_size=config.intermediate_size, rms_norm_eps=config.rms_norm_eps,
            include_bias=False, include_qkv_bias=False, rope_full_precision=False,
        )
        self.layers = nn.ModuleList([
            LLaDABlock(block_config, self.rotary_emb) for _ in range(config.num_hidden_layers)
        ])
        for layer in self.layers:
            layer.attn_norm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
            layer.ff_norm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)
        return self.lm_head(self.norm(hidden_states))


def build_from_config(config, device, dtype):
    if (config.hidden_act != "silu" or config.attention_bias or config.mlp_bias
            or config.tie_word_embeddings or config.num_key_value_heads != config.num_attention_heads
            or config.head_dim != config.hidden_size // config.num_attention_heads
            or config.rope_parameters["rope_type"] != "default"):
        raise ValueError("EuroBERT coverage preserves the checkpoint's untied, bias-free masked-LM graph")
    return EuroBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    renames = {
        "attention.attn_out": "self_attn.o_proj", "attention.": "self_attn.",
        "attn_norm": "input_layernorm", "ff_norm": "post_attention_layernorm",
        "mlp.ff_proj": "mlp.gate_proj", "mlp.ff_out": "mlp.down_proj",
    }
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("layers."):
            for target, reference in renames.items():
                source = source.replace(target, reference)
        if not source.startswith("lm_head."):
            source = "model." + source
        parameter.copy_(state_dict[source])


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(**inputs)})}
