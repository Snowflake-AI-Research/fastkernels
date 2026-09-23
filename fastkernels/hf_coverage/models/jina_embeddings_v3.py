"""JinaEmbeddingsV3ForMaskedLM using BERT blocks and bidirectional rotary attention."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L2.llada_attention import LLaDAAttention
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from ..runner import Workload
from .bert import MaskedLMHead


class RotaryEncoderAttention(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.self_attn = LLaDAAttention(
            config.hidden_size, config.num_attention_heads, config.num_attention_heads,
            config.hidden_size // config.num_attention_heads, rotary_emb=rotary, bias=True,
        )
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, hidden_states):
        attended, _ = self.self_attn(hidden_states)
        return self.LayerNorm(hidden_states + attended)


class JinaEmbeddingsV3ForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = SimpleNamespace(**(dict(config) | {"position_embedding_type": "rotary"}))
        self.embeddings = BertEmbeddings(embedding_config)
        del self.embeddings.position_embeddings
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings, config.rope_parameters["rope_theta"],
        )
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            layer.attention = RotaryEncoderAttention(config, self.rotary_emb)
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        positions = self.embeddings.position_ids[:, :input_ids.shape[1]]
        hidden_states = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        return self.lm_head(self.encoder(hidden_states))


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.rope_parameters["rope_type"] != "default":
        raise ValueError("Jina Embeddings v3 coverage preserves the native masked-LM default graph")
    return JinaEmbeddingsV3ForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    renames = {
        "attention.self_attn.attn_out": "self_attn.o_proj",
        "attention.self_attn": "self_attn",
        "attention.LayerNorm": "post_attention_layernorm",
        "intermediate.dense": "mlp.fc1", "output.dense": "mlp.fc2",
        "output.LayerNorm": "post_mlp_layernorm",
    }
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("encoder.layer."):
            source = source.replace("encoder.layer.", "roberta.layers.", 1)
            for target, reference in renames.items():
                source = source.replace(target, reference)
        elif source.startswith("embeddings."):
            source = "roberta." + source
        else:
            source = source.replace("lm_head.LayerNorm.", "lm_head.layer_norm.")
        parameter.copy_(state_dict[source])


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(**inputs)})}
