"""Documented BERT-to-BERT generation through existing encoder components."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer
from fastkernels.tasks.baseline.L3.bert_model import BertModel

from .bart import _cached_self_attention, _fresh_cache
from .bert import MaskedLMHead, load_mlm_head
from ..runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads


class BertDecoderLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.cross_query = Linear(config.hidden_size, config.hidden_size)
        self.cross_key = Linear(config.hidden_size, config.hidden_size)
        self.cross_value = Linear(config.hidden_size, config.hidden_size)
        self.cross_output = EncoderSelfOutput(config)

    def forward(self, hidden, memory, past_key_value=None):
        attention = self.attention.self
        batch, length = hidden.shape[:2]
        heads, width = attention.num_attention_heads, attention.attention_head_size
        context, self_cache = _cached_self_attention(
            attention, hidden, None if past_key_value is None else past_key_value[0],
        )
        hidden = self.attention.output(context, hidden)
        query = self.cross_query(hidden).view(batch, length, heads, width)
        if past_key_value is None:
            key = self.cross_key(memory).view(batch, memory.shape[1], heads, width).transpose(1, 2)
            value = self.cross_value(memory).view(batch, memory.shape[1], heads, width).transpose(1, 2)
            cross_cache = _fresh_cache(key, value)
        else:
            cross_cache = past_key_value[1]
        key, value = cross_cache
        context = attention.attn(query, key.transpose(1, 2), value.transpose(1, 2))
        hidden = self.cross_output(context.reshape(batch, length, -1), hidden)
        hidden = self.output(self.intermediate(hidden), hidden)
        return hidden, (self_cache, cross_cache)


class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel(config.encoder)
        self.encoder_pooler = Linear(config.encoder.hidden_size, config.encoder.hidden_size)
        self.pooler_activation = Tanh()
        self.decoder_embeddings = BertEmbeddings(config.decoder)
        self.decoder_layers = nn.ModuleList([BertDecoderLayer(config.decoder)
                                             for _ in range(config.decoder.num_hidden_layers)])
        # Match native HF's attention kernel despite dependencies that disable
        # global cuDNN selection. The same op serves self- and cross-attention.
        for layer in (*self.encoder.encoder.layer, *self.decoder_layers):
            layer.attention.self.attn = DenseAttention(backend="cudnn")
        self.lm_head = MaskedLMHead(config.decoder)
        if config.decoder.tie_word_embeddings:
            self.lm_head.decoder.weight = self.decoder_embeddings.word_embeddings.emb.weight

    def forward(self, input_ids, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("BERT-pair coverage evaluates unpadded token sequences")
        if encoder_hidden_states is None:
            memory = self.encoder.forward_with_attention_mask(input_ids)
            # HF's base encoder computes its pooler even though the wrapper discards it.
            self.pooler_activation(self.encoder_pooler(memory[:, 0]))
        else:
            memory = encoder_hidden_states
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)[None] + past_length
        hidden = self.decoder_embeddings(decoder_input_ids, positions)
        cache = []
        for index, layer in enumerate(self.decoder_layers):
            hidden, state = layer(hidden, memory, None if past_key_values is None else past_key_values[index])
            cache.append(state)
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory,
                "past_key_values": cache}


def build_from_config(config, device, dtype):
    enc, dec = config.encoder, config.decoder
    if (enc.model_type != "bert" or dec.model_type != "bert" or enc.is_decoder
            or not dec.is_decoder or not dec.add_cross_attention or not dec.use_cache
            or enc.hidden_act != "gelu" or dec.hidden_act != "gelu"
            or enc.hidden_size != dec.hidden_size or getattr(config, "tie_encoder_decoder", False)
            or getattr(enc, "position_embedding_type", "absolute") != "absolute"
            or getattr(dec, "position_embedding_type", "absolute") != "absolute"):
        raise ValueError("This case preserves the documented untied, same-width cached BERT pair")
    return EncoderDecoder(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {}
    for name in model.state_dict():
        if name.startswith("lm_head."):
            continue
        source = name.replace(".emb.weight", ".weight")
        source = source.replace("encoder_pooler.", "encoder.pooler.dense.")
        source = source.replace("decoder_embeddings.", "decoder.bert.embeddings.")
        source = source.replace("decoder_layers.", "decoder.bert.encoder.layer.")
        source = source.replace(".cross_query.", ".crossattention.self.query.")
        source = source.replace(".cross_key.", ".crossattention.self.key.")
        source = source.replace(".cross_value.", ".crossattention.self.value.")
        source = source.replace(".cross_output.", ".crossattention.output.")
        if ".qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".qkv.", f".{projection}."))
                                      for projection in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    head = {name.removeprefix("decoder."): remaining.pop(name)
            for name in list(remaining) if name.startswith("decoder.cls.")}
    load_mlm_head(model.lm_head, head)
    # The tied decoder head must agree with the embedding rather than overwrite it.
    if config.decoder.tie_word_embeddings and not torch.equal(
            mapped["decoder_embeddings.word_embeddings.emb.weight"], head["cls.predictions.decoder.weight"]):
        raise ValueError("BERT decoder tied word weights disagree")
    mapped.update({"lm_head." + name: tensor for name, tensor in model.lm_head.state_dict().items()})
    if remaining:
        raise KeyError(f"Unmapped encoder-decoder weights: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs)

    def run():
        output = model(**inputs)
        cache = output.pop("past_key_values")
        return dict(output, **seq2seq_cache_outputs(cache))

    return {"forward": Workload(run=run)}
