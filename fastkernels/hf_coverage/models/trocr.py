"""TrOCR's public cross-attention decoder, using existing post-norm blocks."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from .bart import BartStack
from .marian import load_state_dict_into as load_seq2seq
from .mvp import EagerAttention


class TrOCRForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        shared = Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.decoder = BartStack(config, shared, decoder=True)
        for layer in self.decoder.layers:
            layer.attention.self.attn = EagerAttention()
            layer.cross_attention.attention = EagerAttention()
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = shared.emb.weight

    def forward(self, input_ids, encoder_hidden_states, past_key_values=None):
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + past_length + 2
        hidden, cache = self.decoder(input_ids, positions, encoder_hidden_states, past_key_values)
        return {"logits": self.lm_head(hidden), "past_key_values": cache}


def build_from_config(config, device, dtype):
    if (not config.use_learned_position_embeddings or not config.layernorm_embedding
            or config.cross_attention_hidden_size is not None or config.activation_function != "gelu"
            or not config.tie_word_embeddings or not config.use_cache):
        raise ValueError("TrOCR case preserves default learned positions, embedding norm, tied GELU decoder")
    return TrOCRForCausalLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    translated = {name.replace("output_projection.", "lm_head."): value for name, value in state_dict.items()}
    load_seq2seq(model, translated, config)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "causal_lm_continuation":
        # The public TrOCR class receives vision features directly; the shared
        # helper normally obtains those features from an encoder's first call.
        def decode(memory, ids, *, past_key_values=None, **kwargs):
            output = model(ids, memory, past_key_values)
            return {**output, "encoder_last_hidden_state": memory}

        return seq2seq_continuation_workloads(
            decode, {"decoder_input_ids": inputs["input_ids"],
                     "encoder_hidden_states": inputs["encoder_hidden_states"]},
            encoder_input_name="encoder_hidden_states", output_names=("logits",),
        )

    def collect(output):
        return {"logits": output["logits"], **seq2seq_cache_outputs(output["past_key_values"])}

    return {"forward": Workload(run=lambda: model(**inputs), collect=collect)}
