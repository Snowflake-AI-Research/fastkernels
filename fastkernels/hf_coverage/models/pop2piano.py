"""Pop2Piano's composer-conditioned mel encoder and cached greedy MIDI decoding."""

from copy import copy
import re

import torch

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from .t5 import T5ForConditionalGeneration


class Pop2Piano(T5ForConditionalGeneration):
    def __init__(self, config):
        adapted = copy(config)
        adapted.scale_decoder_outputs = False
        super().__init__(adapted)
        self.config = config
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        # tie_word_embeddings=False leaves all three native tables independent.
        # The feature-input generation path only executes the decoder table.
        self.encoder.embed_tokens = Embedding(config.vocab_size, config.d_model)
        self.decoder_embedding = Embedding(config.vocab_size, config.d_model)
        self.composer = Embedding(config.composer_vocab_size, config.d_model)
        self.top1 = CodecTop1()

    def encode_features(self, features, mask, composer):
        composer_ids = torch.full((features.shape[0], 1), composer, dtype=torch.long, device=features.device)
        hidden = torch.cat((self.composer(composer_ids), features), dim=1)
        extended_mask = None
        if mask is not None:
            mask = torch.cat((mask[:, :1], mask), dim=1).bool()
            extended_mask = hidden.new_zeros(mask.shape[0], 1, 1, mask.shape[1])
            extended_mask.masked_fill_(~mask[:, None, None, :], torch.finfo(hidden.dtype).min)
        bias = None
        for block in self.encoder.block:
            hidden, bias = block(hidden, mask=extended_mask, position_bias=bias)
        return self.encoder.final_layer_norm(hidden), extended_mask

    def decode_step(self, ids, memory, mask, cache):
        hidden = self.decoder_embedding(ids)
        offset = 0 if cache is None else cache[0][0][0].shape[2]
        length, batch = ids.shape[1], ids.shape[0]
        query_positions = torch.arange(length, device=ids.device) + offset
        key_positions = torch.arange(length + offset, device=ids.device)
        buckets = T5SelfAttention._relative_position_bucket(
            key_positions[None, :] - query_positions[:, None], bidirectional=False,
            num_buckets=self.config.relative_attention_num_buckets,
            max_distance=self.config.relative_attention_max_distance)
        self_bias = self.decoder[0].self_attention.relative_attention_bias(buckets).permute(2, 0, 1)[None]
        self_bias = self_bias.masked_fill(key_positions[None, :] > query_positions[:, None],
                                         torch.finfo(hidden.dtype).min)
        next_cache = []
        for index, block in enumerate(self.decoder):
            attention = block.self_attention
            width = self.config.num_heads * self.config.d_kv
            query, key, value = (tensor.reshape(batch, length, self.config.num_heads, self.config.d_kv)
                                 .transpose(1, 2) for tensor in attention.qkv_proj(block.self_norm(hidden)).split(width, -1))
            if cache is not None:
                key, value = (torch.cat((old, new), dim=2) for old, new in zip(cache[index][0], (key, value)))
            else:
                key, value = key.contiguous(), value.contiguous()
            self_cache = (key, value)
            scores = attention.bmm(query, key.transpose(2, 3)) + self_bias
            probabilities = attention.softmax(scores.float()).to(scores.dtype)
            context = attention.bmm(probabilities, value).transpose(1, 2).reshape(batch, length, width)
            hidden = hidden + attention.o(context)
            cross = block.cross_attention
            query = cross.q(block.cross_norm(hidden)).reshape(batch, length, self.config.num_heads, self.config.d_kv).transpose(1, 2)
            if cache is None:
                key, value = (projection(memory).reshape(batch, memory.shape[1], self.config.num_heads, self.config.d_kv)
                              .transpose(1, 2).contiguous() for projection in (cross.k, cross.v))
            else:
                key, value = cache[index][1]
            cross_cache = (key, value)
            scores = cross.bmm(query, key.transpose(2, 3))
            if mask is not None:
                scores = scores + mask
            probabilities = cross.softmax(scores.float()).to(scores.dtype)
            context = cross.bmm(probabilities, value).transpose(1, 2).reshape(batch, length, width)
            hidden = block.ff(hidden + cross.o(context))
            next_cache.append((self_cache, cross_cache))
        return self.lm_head(self.decoder_norm(hidden)), tuple(next_cache)

    def generate(self, input_features, attention_mask, generation, steps):
        if input_features.shape[0] != 1:
            raise ValueError("This generation case measures one audio segment")
        composers = generation["composer_to_feature_token"]
        composer = composers["composer1"] - min(composers.values())
        memory, mask = self.encode_features(input_features, attention_mask, composer)
        ids = torch.full((1, 1), generation["decoder_start_token_id"], device=input_features.device, dtype=torch.long)
        sequences, logits_steps, cache = ids, [], None
        for _ in range(steps):
            logits, cache = self.decode_step(ids, memory, mask, cache)
            logits = logits[:, -1].float()
            logits_steps.append(logits)
            ids = self.top1(logits).reshape(1, 1)
            sequences = torch.cat((sequences, ids), dim=1)
            if ids.item() == generation["eos_token_id"]:
                break
        return {"sequences": sequences,
                **{f"logits.{index}": logits for index, logits in enumerate(logits_steps)},
                **seq2seq_cache_outputs(cache)}


def build_from_config(config, device, dtype):
    if config.tie_word_embeddings or not config.use_cache or not config.is_gated_act or config.dense_act_fn != "relu":
        raise ValueError("Published Pop2Piano checkpoint uses untied output weights and gated ReLU with caches")
    return Pop2Piano(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for key in model.state_dict():
        source = key.replace(".emb.", ".")
        source = source.replace("composer.", "mel_conditioner.embedding.")
        source = source.replace("decoder_embedding.", "decoder.embed_tokens.")
        source = source.replace("decoder_norm.", "decoder.final_layer_norm.")
        source = re.sub(r"^decoder\.(\d+)\.self_attention\.", r"decoder.block.\1.layer.0.SelfAttention.", source)
        source = re.sub(r"^decoder\.(\d+)\.self_norm\.", r"decoder.block.\1.layer.0.layer_norm.", source)
        source = re.sub(r"^decoder\.(\d+)\.cross_attention\.", r"decoder.block.\1.layer.1.EncDecAttention.", source)
        source = re.sub(r"^decoder\.(\d+)\.cross_norm\.", r"decoder.block.\1.layer.1.layer_norm.", source)
        source = re.sub(r"^decoder\.(\d+)\.ff\.", r"decoder.block.\1.layer.2.", source)
        if source.endswith("qkv_proj.weight"):
            mapped[key] = torch.cat([remaining.pop(source.replace("qkv_proj", projection))
                                     for projection in ("q", "k", "v")])
        elif source.endswith("DenseReluDense.wi.weight"):
            mapped[key] = torch.cat([remaining.pop(source.replace(".wi.", f".wi_{index}.")) for index in (0, 1)])
        else:
            mapped[key] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped Pop2Piano state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case):
    generation = case["reference"]["generation_config"]
    return {"generate": Workload(run=lambda: model.generate(
        inputs["input_features"], inputs.get("attention_mask"), generation,
        case["generation_kwargs"]["max_new_tokens"]))}
