"""FSMT's default cached first translation step with separate source/target vocabularies."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from .bart import BartStack, _fresh_cache
from .marian import load_state_dict_into as load_seq2seq
from .mvp import EagerAttention


class FSMTForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BartStack(config, Embedding(config.src_vocab_size, config.d_model, config.pad_token_id), decoder=False)
        self.decoder = BartStack(config, Embedding(config.tgt_vocab_size, config.d_model, config.pad_token_id), decoder=True)
        self.pad = config.pad_token_id
        for stack in (self.encoder, self.decoder):
            stack.layernorm_embedding = nn.Identity()
            stack.embed_positions = Embedding(config.max_position_embeddings + self.pad + 1, config.d_model, self.pad)
            stack.embed_positions.emb.weight.requires_grad_(False)
            for layer in (stack.layers if stack.is_decoder else stack.layers.layer):
                layer.intermediate.intermediate_act_fn = ReLU()
                layer.attention.self.attn = EagerAttention()
                if stack.is_decoder:
                    layer.cross_attention.attention = EagerAttention()
        self.lm_head = Linear(config.d_model, config.tgt_vocab_size, bias=False)

    def positions(self, ids):
        valid = ids.ne(self.pad).int()
        return (valid.cumsum(1).to(valid.dtype) * valid).long() + self.pad

    def forward(self, input_ids, decoder_input_ids, memory=None, past=None, *,
                encoder_hidden_states=None, past_key_values=None, attention_mask=None,
                decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("FSMT coverage evaluates unpadded translation inputs")
        if encoder_hidden_states is not None:
            memory = encoder_hidden_states
        if past_key_values is not None:
            past = past_key_values
        if memory is None:
            memory = self.encoder(input_ids, self.positions(input_ids))
        # Positions use the full history before default cached slicing.
        positions = self.decoder.embed_positions(self.positions(decoder_input_ids))[:, -1:]
        hidden = self.decoder.embed_tokens(decoder_input_ids[:, -1:]) * self.decoder.embed_scale + positions
        cache = []
        for index, layer in enumerate(self.decoder.layers):
            attention = layer.attention.self
            batch, length = hidden.shape[:2]
            query, key, value = (
                tensor.view(batch, length, attention.num_attention_heads, attention.attention_head_size)
                for tensor in attention._project_qkv(hidden))
            key, value = key.transpose(1, 2), value.transpose(1, 2)
            if past is None:
                key, value = self_cache = _fresh_cache(key, value)
            else:
                key, value = self_cache = tuple(torch.cat((old, new), dim=2)
                                               for old, new in zip(past[index][0], (key, value)))
            context = attention.attn(query, key.transpose(1, 2), value.transpose(1, 2))
            hidden = layer.attention.output(context.reshape(batch, length, -1), hidden)
            cross = layer.cross_attention
            query = cross.q_proj(hidden).view(batch, length, cross.heads, cross.head_dim)
            if past is None:
                key = cross.k_proj(memory).view(batch, memory.shape[1], cross.heads, cross.head_dim).transpose(1, 2)
                value = cross.v_proj(memory).view(batch, memory.shape[1], cross.heads, cross.head_dim).transpose(1, 2)
                key, value = cross_cache = _fresh_cache(key, value)
            else:
                key, value = cross_cache = past[index][1]
            context = cross.attention(query, key.transpose(1, 2), value.transpose(1, 2))
            hidden = cross.norm(hidden + cross.out_proj(context.reshape(batch, length, -1)))
            hidden = layer.output(layer.intermediate(hidden), hidden)
            cache.append((self_cache, cross_cache))
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory, "past_key_values": tuple(cache)}


def build_from_config(config, device, dtype):
    if config.activation_function != "relu" or config.tie_word_embeddings or not config.use_cache:
        raise ValueError("Selected FSMT checkpoint requires untied ReLU translation with default caching")
    return FSMTForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    translated = {name.replace("model.decoder.output_projection.", "lm_head."): value
                  for name, value in state_dict.items()}
    load_seq2seq(model, translated, config)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs, full_decoder_history=True)
    ids, decoder_ids = inputs["input_ids"], inputs["decoder_input_ids"]
    state = {}

    def select(output):
        return {"logits": output["logits"], "encoder_last_hidden_state": output["encoder_last_hidden_state"],
                **seq2seq_cache_outputs(output["past_key_values"])}

    def prefill():
        return model(ids, decoder_ids[:, :1])

    def prepare_decode():
        output = prefill()
        state["memory"], state["past"] = output["encoder_last_hidden_state"], output["past_key_values"]

    return {"prefill": Workload(run=lambda: select(prefill())),
            "decode": Workload(run=lambda: select(model(ids, decoder_ids, **state)), prepare=prepare_decode)}
