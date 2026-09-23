"""Ovis2 soft visual tokens, learned visual table and Qwen2 decoder."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from . import llava, qwen2
from .olmo2 import decoder_config
from .qwen2_5_omni import TextMLP
from ..runner import Workload


class VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.dim = config.hidden_size // self.heads
        self.attention = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.attention, name, Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias))
        self.attn = DenseAttention(backend="sdpa")
        self.ffn = TextMLP(config)
        self.rms_norm1 = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.rms_norm2 = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden):
        normalized = self.rms_norm1(hidden)
        shape = (*hidden.shape[:-1], self.heads, self.dim)
        q, k, v = [getattr(self.attention, name)(normalized).reshape(shape)
                   for name in ("q_proj", "k_proj", "v_proj")]
        hidden = hidden + self.attention.out_proj(self.attn(q, k, v).reshape_as(hidden))
        return hidden + self.ffn(self.rms_norm2(hidden))


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, patch = config.hidden_size, config.patch_size
        self.stride = config.hidden_stride
        self.patch_embedding = Conv2d(config.num_channels, width, patch, stride=patch)
        count = (config.image_size // patch) ** 2
        self.position_embedding = Embedding(count, width)
        self.register_buffer("positions", torch.arange(count)[None], persistent=False)
        self.embedding_norm = RMSNormNative(width, config.rms_norm_eps)
        self.layers = nn.ModuleList(VisionLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNormNative(width, config.rms_norm_eps)
        self.head_linear = Linear(width * self.stride ** 2,
                                  config.vocab_size - config.num_visual_indicator_tokens, bias=False)
        self.head_norm = LayerNorm(config.vocab_size - config.num_visual_indicator_tokens)
        self.softmax = Softmax()

    def forward(self, pixels):
        hidden = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        hidden = self.embedding_norm(hidden) + self.position_embedding(self.positions)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        if self.stride > 1:
            batch, length, width = hidden.shape
            side, stride = int(length ** .5), self.stride
            if side * side != length or side % stride:
                raise ValueError("The Ovis2 image grid must divide its declared hidden stride")
            hidden = hidden.reshape(batch, side // stride, stride, side // stride, stride, width)
            hidden = hidden.permute(0, 1, 3, 2, 4, 5).reshape(batch, -1, stride ** 2 * width)
        return self.softmax(self.head_norm(self.head_linear(hidden)))


class Backbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.vision = Vision(config.vision_config)
        self.visual_embeddings_table = Embedding(config.vision_config.vocab_size, config.text_config.hidden_size)
        self.matmul = BMM()
        self.image_token_id = config.image_token_id
        self.indicators = config.visual_indicator_token_ids
        self.pixel_values = self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, ids, positions):
        hidden = self.text.embed_tokens(ids)
        if get_context().is_prefill:
            probabilities = self.vision(self.pixel_values)
            padding = probabilities.new_zeros(*probabilities.shape[:-1], len(self.indicators))
            self.image_hidden_states = self.matmul(torch.cat((probabilities, padding), -1),
                                                   self.visual_embeddings_table.emb.weight)
            hidden[ids == self.image_token_id] = self.image_hidden_states.flatten(0, 1)
            first = self.visual_embeddings_table.emb.weight.shape[0] - len(self.indicators)
            indicator_features = self.visual_embeddings_table(torch.arange(first, first + len(self.indicators), device=ids.device))
            for index, token in enumerate(self.indicators):
                hidden[ids == token] = indicator_features[index]
        return self.text(ids, positions, inputs_embeds=hidden)


class Model(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = Backbone(language.model, config)


def build_from_config(config, device, dtype):
    if (config.vision_config.tokenize_function != "softmax" or config.vision_config.mlp_bias
            or config.vision_config.hidden_act != "silu" or config.text_config.use_sliding_window):
        raise ValueError("The selected Ovis2 checkpoint uses soft visual tokens and full-attention SiLU blocks")
    local = decoder_config(config.text_config, dtype)
    local.qkv_bias = True
    language = LlamaForCausalLM(local)
    if config.tie_word_embeddings:
        language.lm_head.embedding_op.emb.weight = language.model.embed_tokens.embedding_op.emb.weight
    return Model(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    remaining = dict(state)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    qwen2.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config),
                              text, config.text_config)
    mapped = {}
    for name in model.model.vision.state_dict():
        source = name.replace("position_embedding.emb.", "position_embedding.")
        if source.startswith(("patch_embedding.", "position_embedding.", "embedding_norm.")):
            source = "transformer.embeddings." + source.replace("embedding_norm.", "rms_norm.")
        elif source.startswith("layers."):
            source = "transformer.encoder." + source
        elif source.startswith("norm."):
            source = source.replace("norm.", "transformer.rms_norm.")
        mapped[name] = remaining.pop("model.vision_tower." + source)
    model.model.vision.load_state_dict(mapped, strict=True)
    model.model.visual_embeddings_table.emb.weight.copy_(remaining.pop("model.visual_embeddings_table.weight"))
    if remaining:
        raise KeyError(f"Unmapped Ovis2 weights: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("The Ovis2 development case uses one image-text sequence")
    workloads = llava.make_workloads(model, inputs, config)
    for phase, workload in list(workloads.items()):
        def run(workload=workload, phase=phase):
            result = workload.run()
            length = inputs["input_ids"].shape[1] - (phase == "prefill")
            for index, layer in enumerate(model.model.layers):
                attention = layer.self_attn.attn
                for name, cache in (("key", attention.k_cache), ("value", attention.v_cache)):
                    if attention.kv_layout == "HND":
                        cache = cache.transpose(1, 2)
                    result[f"past_key_values.{index}.{name}"] = cache.flatten(0, 1)[:length].transpose(0, 1)[None]
            return result
        workloads[phase] = Workload(run=run, prepare=workload.prepare)
    return workloads
