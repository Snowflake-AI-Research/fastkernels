"""PLBART's default conditional-generation graph uses the BART composition."""

import torch

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from .bart import build_from_config, load_state_dict_into


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs)
    encoder_ids, decoder_ids = inputs["input_ids"], inputs["decoder_input_ids"]
    offset = getattr(model.encoder, "position_offset", 2)
    encoder_positions = torch.arange(encoder_ids.shape[1], device=encoder_ids.device) + offset
    decoder_positions = torch.arange(decoder_ids.shape[1], device=decoder_ids.device) + offset

    def run():
        output = model(encoder_ids, decoder_ids, encoder_positions, decoder_positions)
        return {"logits": output["logits"], "encoder_last_hidden_state": output["encoder_last_hidden_state"],
                **seq2seq_cache_outputs(output["past_key_values"])}

    return {"forward": Workload(run=run)}
