"""M2M-100 pre-norm composition with fixed sinusoidal position metadata."""

import math

import torch

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.embedding import Embedding
from .mbart import PreNormConditionalGeneration, select_native_attention
from .marian import load_state_dict_into


def sinusoidal_table(size, width, padding_idx):
    """Prepare fixed positions on CPU, matching native HF model construction."""
    half = width // 2
    # CUDA transcendental rounding changes some BF16 table entries. This table
    # is configuration-only metadata; reproduce HF's CPU constants before copy.
    frequencies = torch.exp(torch.arange(half, dtype=torch.int64, device="cpu").float()
                            * (-math.log(10000) / (half - 1)))
    angles = torch.arange(size, dtype=torch.int64, device="cpu").float().unsqueeze(1) * frequencies.unsqueeze(0)
    table = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if width % 2:
        table = torch.cat((table, torch.zeros(size, 1, device="cpu")), dim=1)
    table[padding_idx] = 0
    return table


class M2M100ForConditionalGeneration(PreNormConditionalGeneration):
    generated_positions = True

    def __init__(self, config):
        super().__init__(config, learned_positions=False)
        del self.final_logits_bias
        self.padding_idx = config.pad_token_id
        for stack in (self.encoder, self.decoder):
            stack.embed_positions = Embedding(config.max_position_embeddings + 2, config.d_model)
            stack.embed_positions.emb.weight.requires_grad_(False)
            stack.embed_positions.emb.weight.data.copy_(
                sinusoidal_table(config.max_position_embeddings + 2, config.d_model, config.pad_token_id))

    def forward(self, encoder_ids, decoder_ids, *, encoder_hidden_states=None, past_key_values=None,
                attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("M2M-100 coverage currently evaluates unpadded token sequences")

        def positions(ids, past_length=0):
            valid = ids.ne(self.padding_idx).to(torch.int32)
            return ((torch.cumsum(valid, dim=1).to(valid.dtype) + past_length) * valid).long() + self.padding_idx

        memory = encoder_hidden_states
        if memory is None:
            memory = self.encoder(encoder_ids, positions(encoder_ids))
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        hidden, cache = self.decoder(decoder_ids, positions(decoder_ids, past_length), memory, past_key_values)
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory,
                "past_key_values": cache}


def build_from_config(config, device, dtype):
    if (config.activation_function != "relu" or not config.scale_embedding
            or not config.tie_word_embeddings or not config.use_cache):
        raise ValueError("M2M-100 requires its default scaled, tied, pre-norm ReLU path")
    model = M2M100ForConditionalGeneration(config)
    select_native_attention(model)
    return model.to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs)

    def run():
        output = model(inputs["input_ids"], inputs["decoder_input_ids"])
        return {"logits": output["logits"], "encoder_last_hidden_state": output["encoder_last_hidden_state"],
                **seq2seq_cache_outputs(output["past_key_values"])}

    return {"forward": Workload(run=run)}
