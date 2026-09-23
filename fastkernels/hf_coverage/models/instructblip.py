"""InstructBLIP's instruction-aware query transformer and Vicuna decoder."""

import torch
from torch import nn
from types import SimpleNamespace

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderAttention, EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention
from . import llama
from .blip import BlipVision
from .mvp import EagerAttention
from ..runner import Workload


class InstructionLayer(nn.Module):
    def __init__(self, config, cross_attention):
        super().__init__()
        self.heads = config.num_attention_heads
        self.attention = EncoderAttention(config)
        self.attention.self.attn = EagerAttention(prescale_query=False, divide_scores=True)
        if cross_attention:
            self.cross_query = Linear(config.hidden_size, config.hidden_size)
            self.cross_key = Linear(config.encoder_hidden_size, config.hidden_size)
            self.cross_value = Linear(config.encoder_hidden_size, config.hidden_size)
            self.cross_attention = EagerAttention(prescale_query=False, divide_scores=True)
            self.cross_output = EncoderSelfOutput(config)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)
        self.intermediate_query = EncoderIntermediate(config)
        self.output_query = EncoderOutput(config)

    def forward(self, hidden, vision, query_length):
        attended = self.attention(hidden)
        query = attended[:, :query_length]
        if hasattr(self, "cross_attention"):
            batch, length, width = query.shape
            q = self.cross_query(query).view(batch, length, self.heads, width // self.heads)
            k = self.cross_key(vision).view(batch, -1, self.heads, width // self.heads)
            v = self.cross_value(vision).view_as(k)
            query = self.cross_output(self.cross_attention(q, k, v).reshape_as(query), query)
        query = self.output_query(self.intermediate_query(query), query)
        text = attended[:, query_length:]
        text = self.output(self.intermediate(text), text)
        return torch.cat((query, text), dim=1)


class InstructionFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList([InstructionLayer(config, index % config.cross_attention_frequency == 0)
                                     for index in range(config.num_hidden_layers)])

    def forward(self, queries, input_ids, vision):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        text = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        hidden = self.layernorm(torch.cat((queries, text), dim=1).to(self.layernorm.weight.dtype))
        for layer in self.layers:
            hidden = layer(hidden, vision, queries.shape[1])
        return hidden


class InstructBackbone(nn.Module):
    def __init__(self, text, config, *, video=False):
        super().__init__()
        self.text = text
        # InstructBLIP's visual convolution fixes the input to three channels.
        fields = ("hidden_size", "image_size", "patch_size", "num_hidden_layers",
                  "num_attention_heads", "intermediate_size", "layer_norm_eps")
        vision_config = SimpleNamespace(**{name: getattr(config.vision_config, name) for name in fields}, num_channels=3)
        self.vision = BlipVision(vision_config)
        for layer in self.vision.layers:
            layer.attn = VitEncoderAttention(config.vision_config.hidden_size, config.vision_config.num_attention_heads,
                                            qkv_bias=config.vision_config.qkv_bias)
        self.query_tokens = nn.Parameter(torch.empty(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructionFormer(config.qformer_config)
        self.language_projection = Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.image_token_id = config.video_token_index if video else config.image_token_index
        self.video = video
        self.inputs = self.extra_outputs = None

    @property
    def layers(self):
        return self.text.layers

    def features(self, pixels, instructions):
        batch = pixels.shape[0]
        if self.video:
            frames = pixels.shape[1]
            pixels = pixels.flatten(0, 1)
            instructions = instructions.repeat_interleave(frames, dim=0)
        vision, pooler = self.vision(pixels)
        queries = self.query_tokens.expand(vision.shape[0], -1, -1)
        hidden = self.qformer(queries, instructions, vision)
        projected = self.language_projection(hidden[:, :queries.shape[1]])
        self.extra_outputs = {"vision_outputs.last_hidden_state": vision, "vision_outputs.pooler_output": pooler,
                              "qformer_outputs.last_hidden_state": hidden, "qformer_outputs.pooler_output": hidden[:, 0]}
        return projected.reshape(batch, -1, projected.shape[-1])

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            features = self.features(self.inputs["pixel_values"], self.inputs["qformer_input_ids"])
            embeddings = embeddings.masked_scatter((input_ids == self.image_token_id)[:, None].expand_as(embeddings), features)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class InstructBlipModel(nn.Module):
    def __init__(self, text, config, *, video=False):
        super().__init__()
        self.config = text.config
        self.model = InstructBackbone(text.model, config, video=video)
        self.lm_head = text.lm_head


def build(config, device, dtype, *, video=False):
    if (not config.use_decoder_only_language_model or config.text_config.model_type != "llama"
            or config.qformer_config.hidden_act != "gelu" or config.vision_config.hidden_act != "gelu"):
        raise ValueError("The documented InstructBLIP forward example selects Vicuna and GELU vision/query layers")
    text = llama.build_from_config(config.text_config, device, dtype)
    model = InstructBlipModel(text, config, video=video).to(device=device, dtype=dtype).eval()
    if dtype == torch.float16:
        model.model.query_tokens.data = model.model.query_tokens.data.float()
    return model


def build_from_config(config, device, dtype):
    return build(config, device, dtype)


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.removeprefix("language_model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("language_model.")}
    from types import SimpleNamespace
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    load_frontend(model.model, remaining)


@torch.no_grad()
def load_frontend(model, remaining):
    """Load the visual and instruction modules without constructing a language model."""
    mapped = {}
    for name in model.vision.state_dict():
        source = name
        for target, origin in (("patch_embedding.proj.", "embeddings.patch_embedding."),
                               ("class_embedding", "embeddings.class_embedding"), ("position_embedding", "embeddings.position_embedding"),
                               ("layers.", "encoder.layers."), (".norm1.", ".layer_norm1."), (".norm2.", ".layer_norm2."),
                               (".attn.qkv.", ".self_attn.qkv."), (".attn.proj.", ".self_attn.projection.")):
            source = source.replace(target, origin)
        mapped[name] = remaining.pop("vision_model." + source)
    model.vision.load_state_dict(mapped, strict=True)
    mapped = {}
    for name in model.qformer.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("layers."):
            source = source.replace("layers.", "encoder.layer.", 1).replace(".attention.self.", ".attention.attention.")
            for target, origin in ((".cross_query.", ".crossattention.attention.query."),
                                   (".cross_key.", ".crossattention.attention.key."),
                                   (".cross_value.", ".crossattention.attention.value."),
                                   (".cross_output.", ".crossattention.output.")):
                source = source.replace(target, origin)
        else:
            source = "embeddings." + source
        source = "qformer." + source
        if ".qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".qkv.", f".{part}.")) for part in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    model.qformer.load_state_dict(mapped, strict=True)
    model.query_tokens.copy_(remaining.pop("query_tokens"))
    model.language_projection.load_state_dict({field: remaining.pop("language_projection." + field)
                                              for field in ("weight", "bias")})
    if remaining:
        raise KeyError(f"Unmapped InstructBLIP state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    model.model.inputs = inputs
    base = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config, cached_decode=False)["forward"]
    batch, length = inputs["input_ids"].shape

    def run():
        output = base.run()
        output.update(model.model.extra_outputs)
        output["language_model_outputs.logits"] = output["logits"]
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn.attn
            for kind, cache in (("key", attention.k_cache), ("value", attention.v_cache)):
                if attention.kv_layout == "HND":
                    cache = cache.permute(0, 2, 1, 3)
                value = cache.reshape(batch, -1, attention.num_kv_heads, attention.head_size)[:, :length]
                output[f"language_model_outputs.past_key_values.{index}.{kind}"] = value.permute(0, 2, 1, 3)
        return output

    return {"forward": Workload(run=run, prepare=base.prepare)}
