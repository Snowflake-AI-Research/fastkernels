"""LightOnOCR's Pixtral vision, learned patch merge and tied Qwen3 decoder."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from . import llama, qwen3
from .qwen2_precision import NativeRotaryEmbedding, configure_language
from .pixtral import PixtralVisionModel, _Attention
from ..runner import Workload


class PixtralSDPA(_Attention):
    def __init__(self, config, backend="sdpa"):
        super().__init__(config)
        self.attention = DenseAttention(backend=backend)

    def forward(self, hidden, positions, table, mask):
        length = hidden.shape[1]
        query, key = RotaryEmbedding.forward_native(positions, self.q_proj(hidden).reshape(length, -1),
                                                     self.k_proj(hidden).reshape(length, -1), self.head_dim, table)
        query = query.view(1, length, self.heads, self.head_dim)
        key = key.view_as(query)
        value = self.v_proj(hidden).view_as(query)
        return self.o_proj(self.attention(query, key, value, attn_mask=mask).reshape_as(hidden))


class LightVision(PixtralVisionModel):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.transformer.layers:
            layer.attention = PixtralSDPA(config)

    def forward(self, pixels, sizes):
        patches = self.patch_conv(pixels)
        parts = [part[:, :h // self.patch_size, :w // self.patch_size].flatten(1).t()
                 for part, (h, w) in zip(patches, sizes)]
        hidden = self.ln_pre(torch.cat(parts)[None])
        positions = torch.cat([(torch.arange(h // self.patch_size, device=pixels.device)[:, None] * self.max_width +
                                torch.arange(w // self.patch_size, device=pixels.device)[None]).flatten()
                               for h, w in sizes])
        image_ids = torch.cat([torch.full((part.shape[0],), i, device=pixels.device) for i, part in enumerate(parts)])
        mask = hidden.new_zeros(hidden.shape[1], hidden.shape[1])
        mask.masked_fill_(image_ids[:, None] != image_ids[None], torch.finfo(hidden.dtype).min)
        for layer in self.transformer.layers:
            hidden = layer(hidden, positions, self.rotary_table, mask)
        return hidden[0]


class LightBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text, self.vision = text, LightVision(config.vision_config)
        width = config.vision_config.hidden_size
        self.norm = RMSNormNative(width, eps=config.text_config.rms_norm_eps)
        self.merger = Linear(width * config.spatial_merge_size ** 2, width, bias=False)
        self.linear_1 = Linear(width, config.text_config.hidden_size, bias=False)
        self.activation = GELU()
        self.linear_2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.merge, self.patch = config.spatial_merge_size, config.vision_config.patch_size
        self.image_token_id = config.image_token_id
        self.pixels = self.sizes = self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def features(self, pixels, sizes):
        hidden = self.norm(self.vision(pixels, sizes))
        grids = [(h // self.patch, w // self.patch) for h, w in sizes]
        merged = []
        for part, (h, w) in zip(hidden.split([h * w for h, w in grids]), grids):
            # The nonoverlapping unfold is a layout change: channel, row, column.
            grid = part.view(h, w, -1)[:h // self.merge * self.merge, :w // self.merge * self.merge]
            grid = grid.reshape(h // self.merge, self.merge, w // self.merge, self.merge, -1)
            merged.append(grid.permute(0, 2, 4, 1, 3).reshape(-1, hidden.shape[-1] * self.merge ** 2))
        hidden = self.merger(torch.cat(merged))
        return self.linear_2(self.activation(self.linear_1(hidden)))

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            self.image_hidden_states = self.features(self.pixels, self.sizes.tolist())
            embeddings = embeddings.masked_scatter((input_ids == self.image_token_id)[:, None].expand_as(embeddings),
                                                   self.image_hidden_states)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class LightOnModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config, self.lm_head = text.config, text.lm_head
        self.model = LightBackbone(text.model, config)


def build_from_config(config, device, dtype):
    text = config.text_config
    if (text.hidden_act != "silu" or text.attention_bias or not config.tie_word_embeddings
            or text.rope_parameters["rope_type"] != "default" or text.use_sliding_window):
        raise ValueError("The declared LightOnOCR uses a tied, bias-free full-attention Qwen3")
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "head_dim", "vocab_size", "max_position_embeddings", "rms_norm_eps")
    native = LlamaConfig(**{name: getattr(text, name) for name in fields}, dtype=dtype,
                         rope_theta=text.rope_parameters["rope_theta"], rope_scaling_factor=1.0)
    language = LlamaForCausalLM(native)
    configure_language(language.model, native)
    language.model.rotary_emb = NativeRotaryEmbedding(native.head_dim, native.max_position_embeddings,
                                                      native.rope_theta)
    language.lm_head.embedding_op.emb.weight = language.model.embed_tokens.embedding_op.emb.weight
    for layer in language.model.layers:
        layer.self_attn.q_norm = RMSNormNative(text.head_dim, eps=text.rms_norm_eps)
        layer.self_attn.k_norm = RMSNormNative(text.head_dim, eps=text.rms_norm_eps)
        layer.self_attn.rotary_emb = language.model.rotary_emb
    return LightOnModel(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    if not torch.equal(text["lm_head.weight"], text["model.embed_tokens.weight"]):
        raise ValueError("LightOnOCR requires matching tied head and embedding weights")
    qwen3.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    model.model.vision.load_state_dict({name: remaining.pop("model.vision_encoder." + name)
                                       for name in model.model.vision.state_dict()}, strict=True)
    names = {"norm": "norm", "merger": "patch_merger.merging_layer", "linear_1": "linear_1", "linear_2": "linear_2"}
    for name, source in names.items():
        module = getattr(model.model, name)
        module.load_state_dict({field: remaining.pop(f"model.vision_projection.{source}.{field}")
                                for field in module.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped LightOnOCR state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    model.model.pixels, model.model.sizes = inputs["pixel_values"], inputs["image_sizes"]
    workloads = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config)
    prefill = workloads["prefill"]
    workloads["prefill"] = Workload(run=lambda: {**prefill.run(), "image_hidden_states": model.model.image_hidden_states},
                                    prepare=prefill.prepare)
    return workloads
