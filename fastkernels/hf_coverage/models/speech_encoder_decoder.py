"""Documented Wav2Vec2/mBART speech translation, including convolution adapters."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.qwen3_next_attention import _gate_mul_inplace

from . import wav2vec2
from .mbart import PreNormStack
from ..patches.product_gate import ProductGate
from ..runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads


class SpeechEncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc, dec = config.encoder, config.decoder
        self.encoder = wav2vec2.WaveformModel(enc, return_features=True)
        self.adapters = nn.ModuleList([Conv1dNative(enc.output_hidden_size, 2 * enc.output_hidden_size,
                                                    enc.adapter_kernel_size, stride=enc.adapter_stride, padding=1)
                                      for _ in range(enc.num_adapter_layers)])
        self.sigmoid = Sigmoid()
        self.product = ProductGate()
        self.embed_tokens = Embedding(dec.vocab_size, dec.d_model, padding_idx=dec.pad_token_id)
        self.decoder = PreNormStack(dec, self.embed_tokens, decoder=True, learned_positions=True)
        self.lm_head = Linear(dec.d_model, dec.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.emb.weight

    def encode(self, input_values):
        memory = self.encoder(input_values)["last_hidden_state"].transpose(1, 2)
        for adapter in self.adapters:
            values, gates = adapter(memory).chunk(2, dim=1)
            # The unchanged internal Qwen kernel computes sigmoid gating in FP32
            # before the output store, as required by fused convolution GLU.
            if values.is_cuda:
                memory = _gate_mul_inplace(values.contiguous(), gates.contiguous())
            else:
                # FP32 construction diagnostic; GPU runs the existing fused gate.
                packed = torch.cat((values.transpose(1, 2), self.sigmoid(gates).transpose(1, 2)), dim=-1)
                memory = self.product(packed).transpose(1, 2)
        return memory.transpose(1, 2)

    def forward(self, input_values, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("Speech translation coverage evaluates unpadded waveform and token sequences")
        memory = self.encode(input_values) if encoder_hidden_states is None else encoder_hidden_states
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)[None] + past_length + 2
        hidden, cache = self.decoder(decoder_input_ids, positions, memory, past_key_values)
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory,
                "past_key_values": cache}


def build_from_config(config, device, dtype):
    enc, dec = config.encoder, config.decoder
    if (enc.model_type != "wav2vec2" or dec.model_type != "mbart"
            or not enc.add_adapter or enc.hidden_size != enc.output_hidden_size
            or enc.output_hidden_size != dec.d_model or enc.feat_extract_norm != "layer"
            or not enc.do_stable_layer_norm or enc.hidden_act != "gelu"
            or enc.feat_extract_activation != "gelu" or dec.activation_function != "relu"
            or not dec.scale_embedding or not dec.use_cache or not dec.tie_word_embeddings):
        raise ValueError("The documented speech checkpoint uses normalized Wav2Vec2, three GLU adapters and a tied ReLU mBART decoder")
    return SpeechEncoderDecoder(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    encoder = {name.removeprefix("encoder."): remaining.pop(name)
               for name in list(remaining) if name.startswith("encoder.") and not name.startswith("encoder.adapter.")}
    wav2vec2.load_state_dict_into(model.encoder, encoder, config.encoder)
    for index, adapter in enumerate(model.adapters):
        adapter.load_state_dict({field: remaining.pop(f"encoder.adapter.layers.{index}.conv.{field}")
                                 for field in ("weight", "bias")})
    mapped = {}
    for name in model.state_dict():
        if name.startswith(("encoder.", "adapters.")):
            continue
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("embed_tokens."):
            source = "decoder.model.decoder." + source
        elif source.startswith("lm_head."):
            source = "decoder." + source
        else:
            source = source.replace("decoder.", "decoder.model.decoder.", 1)
        source = source.replace(".attention.output.dense.", ".self_attn.out_proj.")
        source = source.replace(".attention.output.LayerNorm.", ".self_attn_layer_norm.")
        source = source.replace(".intermediate.dense.", ".fc1.").replace(".output.dense.", ".fc2.")
        source = source.replace(".output.LayerNorm.", ".final_layer_norm.")
        source = source.replace(".cross_attention.norm.", ".encoder_attn_layer_norm.")
        source = source.replace(".cross_attention.", ".encoder_attn.")
        if ".attention.self.qkv." in source:
            names = [source.replace(".attention.self.qkv.", f".self_attn.{projection}_proj.")
                     for projection in ("q", "k", "v")]
            mapped[name] = torch.cat([remaining.pop(key) for key in names])
        elif source in remaining:
            mapped[name] = remaining.pop(source)
        elif name == "decoder.embed_tokens.emb.weight":
            mapped[name] = mapped["embed_tokens.emb.weight"]
        else:
            raise KeyError(source)
    if not torch.equal(mapped["embed_tokens.emb.weight"], mapped["lm_head.weight"]):
        raise ValueError("mBART tied word weights disagree")
    if remaining:
        raise KeyError(f"Unmapped speech translation weights: {sorted(remaining)}")
    mapped.update({name: value for name, value in model.state_dict().items()
                   if name.startswith(("encoder.", "adapters."))})
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs, encoder_input_name="input_values")

    def run():
        output = model(**inputs)
        cache = output.pop("past_key_values")
        return dict(output, **seq2seq_cache_outputs(cache))

    return {"forward": Workload(run=run)}
