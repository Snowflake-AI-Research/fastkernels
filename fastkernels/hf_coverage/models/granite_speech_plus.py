"""Granite Speech Plus's Conformer, query former, and scaled language decoder."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from ..patches.codec_top1 import CodecTop1
from . import granite, voxtral
from .blip_2 import QueryFormer
from .qwen2_precision import NativeRotaryEmbedding, configure_language
from .wav2vec2_conformer import GLU


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_norm = LayerNorm(config.hidden_dim, promote_fp32=False)
        self.up_proj = Linear(config.hidden_dim, config.hidden_dim * config.feedforward_mult)
        self.silu = SiLU()
        self.down_proj = Linear(config.hidden_dim * config.feedforward_mult, config.hidden_dim)

    def forward(self, hidden):
        return self.down_proj(self.silu(self.up_proj(self.pre_norm(hidden))))


class RelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        inner = config.dim_head * config.num_heads
        self.pre_norm = LayerNorm(config.hidden_dim, promote_fp32=False)
        self.to_q = Linear(config.hidden_dim, inner, bias=False)
        self.to_kv = Linear(config.hidden_dim, inner * 2, bias=False)
        self.to_out = Linear(inner, config.hidden_dim)
        self.rel_pos_emb = Embedding(2 * config.max_pos_emb + 1, config.dim_head)
        self.bmm, self.attention = BatchMatMul(), DenseAttention(backend="sdpa")

    def forward(self, hidden):
        config = self.config
        batch, length, _ = hidden.shape
        count, heads, width = config.context_size, config.num_heads, config.dim_head
        blocks = (length + count - 1) // count
        hidden = self.pre_norm(hidden)
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, blocks * count - length))
        query = self.to_q(hidden)
        key, value = self.to_kv(hidden).chunk(2, -1)
        query, key, value = (x.reshape(batch, blocks, count, heads, width).transpose(2, 3)
                             for x in (query, key, value))
        positions = torch.arange(count, device=hidden.device)
        distances = (positions[:, None] - positions[None]).clamp(-count, count) + config.max_pos_emb
        relative = self.rel_pos_emb(distances)
        relative_scores = self.bmm(query.permute(3, 0, 1, 2, 4).reshape(count, -1, width), relative.transpose(1, 2))
        relative_scores = relative_scores.reshape(count, batch, blocks, heads, count).permute(1, 2, 3, 0, 4) * width**-0.5
        remainder = length % count
        if remainder:
            mask = torch.ones(count, count, device=hidden.device, dtype=torch.bool)
            mask[:remainder, :remainder] = False
            relative_scores[:, -1].masked_fill_(mask, torch.finfo(hidden.dtype).min)
        query, key, value = (x.reshape(batch * blocks, heads, count, width).transpose(1, 2) for x in (query, key, value))
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = self.attention(query, key, value, softmax_scale=width**-0.5,
                                    attn_mask=relative_scores.reshape(batch * blocks, heads, count, count))
        return self.to_out(output.reshape(batch, blocks * count, -1)[:, :length])


class Convolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim * config.conv_expansion_factor
        self.norm = LayerNorm(config.hidden_dim, promote_fp32=False)
        self.up_conv = Conv1dNative(config.hidden_dim, 2 * width, 1)
        self.glu = GLU()
        self.depth_conv = Conv1dNative(width, width, config.conv_kernel_size, groups=width, bias=False)
        self.padding = (config.conv_kernel_size // 2, config.conv_kernel_size // 2 - (config.conv_kernel_size + 1) % 2)
        self.batch_norm = BatchNorm2d(width)
        self.silu = SiLU()
        self.down_conv = Conv1dNative(width, config.hidden_dim, 1)

    def forward(self, hidden):
        hidden = self.glu(self.up_conv(self.norm(hidden).transpose(1, 2)))
        hidden = self.depth_conv(torch.nn.functional.pad(hidden, self.padding))
        return self.down_conv(self.silu(self.batch_norm(hidden))).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ff1, self.ff2 = FeedForward(config), FeedForward(config)
        self.attn, self.conv = RelativeAttention(config), Convolution(config)
        self.post_norm = LayerNorm(config.hidden_dim, promote_fp32=False)

    def forward(self, hidden):
        hidden = self.ff1(hidden) * 0.5 + hidden
        hidden = self.attn(hidden) + hidden
        hidden = self.conv(hidden) + hidden
        hidden = self.ff2(hidden) * 0.5 + hidden
        return self.post_norm(hidden)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_linear = Linear(config.input_dim, config.hidden_dim)
        self.layers = nn.ModuleList(ConformerBlock(config) for _ in range(config.num_layers))
        self.out = Linear(config.hidden_dim, config.output_dim)
        self.out_mid = Linear(config.output_dim, config.hidden_dim)
        self.softmax = Softmax()
        self.cat_layers = set(getattr(config, "cat_hidden_layers", []) or [])

    def forward(self, features):
        hidden = self.input_linear(features)
        exported = [hidden] if 0 in self.cat_layers else []
        for index, layer in enumerate(self.layers, 1):
            hidden = layer(hidden)
            if index in self.cat_layers:
                exported.append(hidden)
            if index == len(self.layers) // 2:
                # Native updates this tensor in place, including exported aliases.
                hidden += self.out_mid(self.softmax(self.out(hidden.clone())))
        return torch.cat((*exported, hidden), -1) if exported else hidden


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window = config.window_size
        self.query = nn.Parameter(torch.empty(1, config.window_size // config.downsample_rate, config.projector_config.hidden_size))
        self.qformer = QueryFormer(config.projector_config)
        self.linear = Linear(config.projector_config.hidden_size, config.text_config.hidden_size)

    def forward(self, hidden):
        batch, length, width = hidden.shape
        blocks = (length + self.window - 1) // self.window
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, blocks * self.window - length))
        hidden = hidden.reshape(batch * blocks, self.window, width)
        query = self.qformer.layernorm(self.query)
        for layer in self.qformer.layers:
            if not hasattr(layer, "cross_query"):
                query = layer(query)
                continue
            query = layer.attention(query)
            query_batch, query_length, query_width = query.shape
            heads = layer.attention.self.num_attention_heads
            q = layer.cross_query(query).view(query_batch, query_length, heads, -1)
            k = layer.cross_key(hidden).view(batch * blocks, self.window, heads, -1)
            v = layer.cross_value(hidden).view_as(k)
            # HF broadcasts one initial learned-query batch across audio
            # windows only at the cross-attention product, after self attention.
            q = q.expand(batch * blocks, -1, -1, -1)
            context = layer.cross_attention(q, k, v).reshape(batch * blocks, query_length, query_width)
            query = layer.cross_output(context, query)
            query = layer.output(layer.intermediate(query), query)
        return self.linear(query.reshape(batch, -1, query.shape[-1]))


class Backbone(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.text, self.encoder, self.projector = language, Encoder(config.encoder_config), Projector(config)
        self.audio_token_id, self.inputs = config.audio_token_index, None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, ids, positions):
        audio_mask = ids == self.audio_token_id
        hidden = self.text.embed_tokens(ids.masked_fill(audio_mask, 0))
        if get_context().is_prefill:
            audio = self.projector(self.encoder(self.inputs["input_features"]))
            if "input_features_mask" in self.inputs:
                audio = audio[self.inputs["input_features_mask"].bool()]
            hidden = hidden.masked_scatter(audio_mask[:, None].expand_as(hidden), audio)
        return self.text(ids, positions, inputs_embeds=hidden)


class FixedAttentionScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, query, key, value, causal=False):
        batch, length, heads, width = query.shape
        key_length = key.shape[1]
        groups = heads // key.shape[2]
        key, value = key.repeat_interleave(groups, dim=2), value.repeat_interleave(groups, dim=2)
        query, key, value = (x.transpose(1, 2).reshape(batch * heads, -1, width) for x in (query, key, value))
        scores = self.bmm(query, key.transpose(1, 2)) * self.scale
        if causal:
            positions = torch.arange(length, device=query.device)
            mask = torch.zeros(length, key_length, device=query.device, dtype=query.dtype)
            mask.masked_fill_(torch.arange(key_length, device=query.device)[None] > positions[:, None], torch.finfo(query.dtype).min)
            scores = scores + mask
        probabilities = self.softmax(scores.float()).to(query.dtype)
        return self.bmm(probabilities, value).reshape(batch, heads, length, width).transpose(1, 2)


class LogitScale(nn.Module):
    def __init__(self, linear, scale):
        super().__init__()
        self.linear, self.scale = linear, scale

    def forward(self, *args):
        return self.linear(*args) / self.scale


def build_from_config(config, device, dtype):
    if config.has_lora_adapter or not config.text_config.use_cache:
        raise ValueError("The selected Granite Speech Plus checkpoint disables LoRA and enables caching")
    return build_speech_model(config, device, dtype)


def build_speech_model(config, device, dtype):
    language = granite.build_from_config(config.text_config, device, dtype)
    configure_language(language.model, language.config)
    language.model.rotary_emb = NativeRotaryEmbedding(language.config.head_dim, language.config.max_position_embeddings, language.config.rope_theta)
    for layer in language.model.layers:
        layer.self_attn.rotary_emb = language.model.rotary_emb
        layer.self_attn.attn.attention = FixedAttentionScale(config.text_config.attention_multiplier)
    language.lm_head.linear_op = LogitScale(language.lm_head.linear_op, config.text_config.logits_scaling)
    model = nn.Module()
    model.config, model.lm_head = language.config, language.lm_head
    model.model = Backbone(language.model, config)
    model.top1 = CodecTop1()
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.removeprefix("language_model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("language_model.")}
    granite.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    mapped = {}
    for name in model.model.encoder.state_dict():
        source = name.replace(".emb.weight", ".weight")
        source = source.replace(".depth_conv.", ".depth_conv.conv.")
        mapped[name] = remaining.pop("encoder." + source)
    model.model.encoder.load_state_dict(mapped, strict=True)
    mapped = {}
    for name in model.model.projector.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("qformer.layers."):
            source = source.replace("qformer.layers.", "qformer.encoder.layer.")
            for target, origin in ((".cross_query.", ".crossattention.attention.query."),
                                   (".cross_key.", ".crossattention.attention.key."),
                                   (".cross_value.", ".crossattention.attention.value."),
                                   (".cross_output.", ".crossattention.output."),
                                   (".intermediate.", ".intermediate_query.")):
                source = source.replace(target, origin)
            parts = source.split(".")
            if parts[4] == "output":
                parts[4] = "output_query"
            source = ".".join(parts).replace(".attention.self.", ".attention.attention.")
        source = "projector." + source
        mapped[name] = (torch.cat([remaining.pop(source.replace(".qkv.", f".{part}.")) for part in ("query", "key", "value")])
                        if ".qkv." in source else remaining.pop(source))
    model.model.projector.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f"Unmapped Granite Speech Plus state: {sorted(remaining)}")


make_workloads = voxtral.make_workloads
