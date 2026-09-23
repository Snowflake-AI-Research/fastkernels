"""Switch's default dense/sparse T5 layers with deterministic top-one routing."""

import re
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.top_k_per_row import _deterministic_topk_indices
from fastkernels.tasks.baseline.L2.t5_dense import T5DenseActDense
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from fastkernels.tasks.baseline.L4.t5_encoder import T5Stack
from .t5 import T5ForConditionalGeneration


class SparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Module()
        self.router.classifier = Linear(config.d_model, config.num_experts, bias=config.router_bias)
        self.experts = nn.ModuleDict({f"expert_{i}": T5DenseActDense(config)
                                     for i in range(config.num_experts)})
        self.softmax, self.product = Softmax(dim=-1), ProductGate()

    def forward(self, hidden):
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = self.router.classifier(flat.float())
        probabilities = self.softmax(logits).to(hidden.dtype)
        rows, experts = probabilities.shape
        starts = torch.zeros(rows, device=hidden.device, dtype=torch.int32)
        ends = torch.full_like(starts, experts)
        selected = _deterministic_topk_indices(probabilities.float(), starts, ends, 1).long()
        weights = probabilities.gather(1, selected)
        output = torch.zeros_like(flat)
        # The pinned router's priority axis has length one, so capacity>=1 drops
        # no token. Top-one dispatch gives disjoint destination rows.
        for index, expert in enumerate(self.experts.values()):
            tokens = (selected[:, 0] == index).nonzero().flatten()
            if tokens.numel():
                values = expert(flat[tokens])
                output[tokens] = self.product(torch.cat((values, weights[tokens].expand_as(values)), dim=-1))
        return output.reshape_as(hidden)


class IndependentEncoder(T5Stack):
    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for block in self.block:
            # Pinned Switch discards each block's returned bias. Only its first
            # block owns a learned table; later blocks construct zero bias.
            hidden, _ = block(hidden, position_bias=None)
        return self.final_layer_norm(hidden)


class SwitchForConditionalGeneration(T5ForConditionalGeneration):
    def _build_encoder(self, config):
        return IndependentEncoder(config, self.shared)

    def forward(self, input_ids, decoder_input_ids, buckets=None, causal_mask=None, *,
                encoder_hidden_states=None, past_key_values=None, attention_mask=None,
                decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("The selected Switch workload uses unpadded inputs")
        memory = self.encoder(input_ids) if encoder_hidden_states is None else encoder_hidden_states
        hidden = self.shared(decoder_input_ids)
        length = decoder_input_ids.shape[1]
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        if buckets is None or causal_mask is None:
            queries = torch.arange(length, device=hidden.device) + past_length
            keys = torch.arange(past_length + length, device=hidden.device)
        if buckets is None:
            buckets = T5SelfAttention._relative_position_bucket(
                keys[None, :] - queries[:, None], bidirectional=False,
                num_buckets=self.config.relative_attention_num_buckets,
                max_distance=self.config.relative_attention_max_distance,
            )
        if causal_mask is None:
            causal_mask = hidden.new_zeros(1, 1, length, past_length + length)
            causal_mask.masked_fill_(keys[None, :] > queries[:, None], torch.finfo(hidden.dtype).min)
        learned = self.decoder[0].self_attention.relative_attention_bias(buckets)
        self_bias = learned.permute(2, 0, 1).unsqueeze(0) + causal_mask
        cross_bias = hidden.new_zeros(1, learned.shape[-1], length, memory.shape[1])
        cache = []
        for index, block in enumerate(self.decoder):
            previous = None if past_key_values is None else past_key_values[index]
            hidden, layer_cache = block(
                hidden, memory, self_bias if index == 0 else causal_mask, cross_bias, previous,
            )
            cache.append(layer_cache)
        hidden = self.decoder_norm(hidden) * self.output_scale
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory, "past_key_values": tuple(cache)}


def build_from_config(config, device, dtype):
    if (config.dense_act_fn != "relu" or config.router_dtype != "float32"
            or not config.tie_word_embeddings or not config.use_cache or config.expert_capacity < 1):
        raise ValueError("Selected Switch checkpoint requires ReLU, FP32 routing, tied output and caching")
    carrier = SimpleNamespace(**{**dict(config), "scale_decoder_outputs": True})
    carrier.is_gated_act = False
    model = SwitchForConditionalGeneration(carrier)
    for index, block in enumerate(model.encoder.block):
        if (index + 1) % config.encoder_sparse_step == 0:
            block.layer[1].DenseReluDense = SparseMLP(carrier)
    for index, block in enumerate(model.decoder):
        if (index + 1) % config.decoder_sparse_step == 0:
            block.ff.DenseReluDense = SparseMLP(carrier)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    consumed, mapped = set(), {}
    for name, target in model.state_dict().items():
        source = name.replace(".emb.weight", ".weight").replace("DenseReluDense", "mlp")
        source = re.sub(r"^decoder\.(\d+)\.self_attention\.", r"decoder.block.\1.layer.0.SelfAttention.", source)
        source = re.sub(r"^decoder\.(\d+)\.self_norm\.", r"decoder.block.\1.layer.0.layer_norm.", source)
        source = re.sub(r"^decoder\.(\d+)\.cross_attention\.", r"decoder.block.\1.layer.1.EncDecAttention.", source)
        source = re.sub(r"^decoder\.(\d+)\.cross_norm\.", r"decoder.block.\1.layer.1.layer_norm.", source)
        source = re.sub(r"^decoder\.(\d+)\.ff\.", r"decoder.block.\1.layer.2.", source)
        source = source.replace("decoder_norm.", "decoder.final_layer_norm.")
        names = ([source.replace("qkv_proj", part) for part in ("q", "k", "v")]
                 if "qkv_proj" in source else [source])
        value = torch.cat([state_dict[key] for key in names]) if len(names) == 3 else state_dict[source]
        if value.shape != target.shape:
            raise ValueError(f"Switch state shape mismatch: {source}")
        mapped[name] = value
        consumed.update(names)
    for name in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"):
        if not torch.equal(state_dict[name], state_dict["shared.weight"]):
            raise ValueError(f"Switch embedding is not tied: {name}")
        consumed.add(name)
    if consumed != set(state_dict):
        raise ValueError(f"Switch unmapped state: {sorted(set(state_dict) - consumed)}")
    model.load_state_dict(mapped)
    # HF casts its classifier to FP32 on first forward. These are the same
    # already-rounded loaded values; only this inference-constant cast is moved.
    for module in model.modules():
        if isinstance(module, SparseMLP):
            module.router.classifier.float()


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs)
    ids, decoder_ids = inputs["input_ids"], inputs["decoder_input_ids"]
    positions = torch.arange(decoder_ids.shape[1], device=decoder_ids.device)
    buckets = T5SelfAttention._relative_position_bucket(
        positions[None, :] - positions[:, None], bidirectional=False,
        num_buckets=config.relative_attention_num_buckets, max_distance=config.relative_attention_max_distance)
    mask = torch.zeros((1, 1, len(positions), len(positions)), device=ids.device, dtype=next(model.parameters()).dtype)
    mask.masked_fill_(positions[None, :] > positions[:, None], torch.finfo(mask.dtype).min)

    def run():
        output = model(ids, decoder_ids, buckets, mask)
        cache = output.pop("past_key_values")
        return {**output, **seq2seq_cache_outputs(cache)}

    return {"forward": Workload(run=run)}
