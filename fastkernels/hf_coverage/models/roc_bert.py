"""RoCBertForMaskedLM with word, shape, and pronunciation inputs."""

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from ..runner import Workload
from .bert import MaskedLMHead, load_mlm_head


class RoCBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base = BertEmbeddings(config)
        self.shape_embed = Embedding(config.shape_vocab_size, config.shape_embed_dim, config.pad_token_id)
        self.pronunciation_embed = Embedding(
            config.pronunciation_vocab_size, config.pronunciation_embed_dim, config.pad_token_id,
        )
        width = config.hidden_size + config.shape_embed_dim + config.pronunciation_embed_dim
        self.map_inputs_layer = Linear(width, config.hidden_size)

    def forward(self, input_ids, input_shape_ids, input_pronunciation_ids):
        channels = [self.base.word_embeddings(input_ids), self.shape_embed(input_shape_ids),
                    self.pronunciation_embed(input_pronunciation_ids)]
        hidden_states = self.map_inputs_layer(torch.cat(channels, dim=-1))
        positions = self.base.position_ids[:, :input_ids.shape[1]]
        return self.base.forward_with_token_type_ids(input_ids, positions, inputs_embeds=hidden_states)


class RoCBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = RoCBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # Match the native HF SDPA backend; importing vLLM disables its default
        # cuDNN selection in the implementation worker.
        for layer in self.encoder.layer:
            layer.attention.self.attn = DenseAttention(backend="cudnn")
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.base.word_embeddings.emb.weight

    def forward(self, input_ids, input_shape_ids, input_pronunciation_ids):
        hidden_states = self.embeddings(input_ids, input_shape_ids, input_pronunciation_ids)
        return self.lm_head(self.encoder(hidden_states))


def build_from_config(config, device, dtype):
    if (not config.concat_input or not config.enable_shape or not config.enable_pronunciation
            or config.hidden_act != "gelu" or config.is_decoder or config.add_cross_attention):
        raise ValueError("RoCBERT coverage requires the checkpoint's three-channel bidirectional encoder")
    return RoCBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    for module_name in ("embeddings", "encoder"):
        module = getattr(model, module_name)
        weights = {}
        for name in module.state_dict():
            source = f"roc_bert.{module_name}." + name.removeprefix("base.").replace(".emb.weight", ".weight")
            if ".qkv." in source:
                weights[name] = torch.cat([
                    state_dict[source.replace(".qkv.", f".{projection}.")]
                    for projection in ("query", "key", "value")
                ])
            else:
                weights[name] = state_dict[source]
        module.load_state_dict(weights)
    load_mlm_head(model.lm_head, state_dict)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(**inputs)})}
