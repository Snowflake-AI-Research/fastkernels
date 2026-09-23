"""BLIP-2's vision encoder, query transformer and OPT conditional decoder."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer
from .blip import BlipVision
from .bridgetower import CrossLayer
from .mvp import EagerAttention
from ..runner import Workload


class QueryFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList()
        for index in range(config.num_hidden_layers):
            if index % config.cross_attention_frequency == 0:
                layer = CrossLayer(config)
                layer.cross_key = Linear(config.encoder_hidden_size, config.hidden_size)
                layer.cross_value = Linear(config.encoder_hidden_size, config.hidden_size)
                layer.cross_attention.divide_scores = True
            else:
                layer = BertLayer(config)
            layer.attention.self.attn = EagerAttention(prescale_query=False, divide_scores=True)
            self.layers.append(layer)

    def forward(self, queries, vision):
        hidden = self.layernorm(queries.to(self.layernorm.weight.dtype))
        vision = vision.to(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, vision)[0] if isinstance(layer, CrossLayer) else layer(hidden)
        return hidden


class OPTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.self_attn = nn.ModuleDict({name: Linear(config.hidden_size, config.hidden_size)
                                       for name in ("q_proj", "k_proj", "v_proj", "out_proj")})
        self.attention = DenseAttention(backend="sdpa")
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.fc1 = Linear(config.hidden_size, config.ffn_dim)
        self.fc2 = Linear(config.ffn_dim, config.hidden_size)
        self.activation = ReLU()

    def forward(self, hidden, previous=None):
        batch, length, width = hidden.shape
        normalized = self.self_attn_layer_norm(hidden)
        query = (self.self_attn["q_proj"](normalized) * (width // self.heads) ** -0.5).view(batch, length, self.heads, -1)
        key = self.self_attn["k_proj"](normalized).view(batch, length, self.heads, -1).transpose(1, 2)
        value = self.self_attn["v_proj"](normalized).view(batch, length, self.heads, -1).transpose(1, 2)
        if previous is not None:
            key, value = torch.cat((previous[0], key), dim=2), torch.cat((previous[1], value), dim=2)
        mask = None
        if previous is not None and length > 1:
            query_positions = torch.arange(length, device=hidden.device) + previous[0].shape[2]
            mask = torch.arange(key.shape[2], device=hidden.device)[None] <= query_positions[:, None]
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2),
                                 causal=previous is None, softmax_scale=1.0, attn_mask=mask)
        hidden = hidden + self.self_attn["out_proj"](context.reshape(batch, length, width))
        hidden = hidden + self.fc2(self.activation(self.fc1(self.final_layer_norm(hidden))))
        return hidden, (key, value)


class OPTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_positions = Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.layers = nn.ModuleList([OPTLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.emb.weight

    def forward(self, embeddings, previous=None):
        start = 0 if previous is None else previous[0][0].shape[2]
        positions = torch.arange(start, start + embeddings.shape[1], device=embeddings.device)
        hidden = embeddings + self.embed_positions(positions + 2)
        state = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, None if previous is None else previous[index])
            state.append(cache)
        return self.lm_head(self.final_layer_norm(hidden)), state


class Blip2ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_token_index = config.image_token_index
        self.vision_model = BlipVision(config.vision_config)
        for layer in self.vision_model.layers:
            layer.attn = VitEncoderAttention(config.vision_config.hidden_size, config.vision_config.num_attention_heads,
                                            qkv_bias=config.vision_config.qkv_bias)
        self.query_tokens = nn.Parameter(torch.empty(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = QueryFormer(config.qformer_config)
        self.language_projection = Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.language_model = OPTDecoder(config.text_config)

    def forward(self, input_ids, pixel_values):
        vision, vision_pooler = self.vision_model(pixel_values)
        queries = self.qformer(self.query_tokens.expand(vision.shape[0], -1, -1), vision)
        projected = self.language_projection(queries.to(vision.dtype))
        embeddings = self.language_model.embed_tokens(input_ids)
        embeddings = embeddings.masked_scatter((input_ids == self.image_token_index).unsqueeze(-1).expand_as(embeddings), projected)
        logits, cache = self.language_model(embeddings)
        outputs = {"logits": logits, "language_model_outputs.logits": logits,
                   "vision_outputs.last_hidden_state": vision, "vision_outputs.pooler_output": vision_pooler,
                   "qformer_outputs.last_hidden_state": queries, "qformer_outputs.pooler_output": queries[:, 0]}
        for index, (key, value) in enumerate(cache):
            outputs[f"language_model_outputs.past_key_values.{index}.key"] = key
            outputs[f"language_model_outputs.past_key_values.{index}.value"] = value
        return outputs


def build_from_config(config, device, dtype):
    text, query = config.text_config, config.qformer_config
    if (text.model_type != "opt" or not text.do_layer_norm_before or text.word_embed_proj_dim != text.hidden_size
            or text.activation_function != "relu" or not text.enable_bias or not text.tie_word_embeddings
            or not text.layer_norm_elementwise_affine or text._remove_final_layer_norm or not text.use_cache
            or query.use_qformer_text_input or query.hidden_act != "gelu" or config.vision_config.hidden_act != "gelu"):
        raise ValueError("This adapter preserves the documented BLIP2 OPT checkpoint and query-only former")
    model = Blip2ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()
    # Pinned HF's non-strict keep-in-FP32 rule applies only when loading FP16.
    # BF16 retains BF16 query tokens/former, despite the model's older comments.
    if dtype == torch.float16:
        model.qformer.float()
        model.query_tokens.data = model.query_tokens.data.float()
    return model


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("vision_model."):
            for target, origin in ((".patch_embedding.proj.", ".embeddings.patch_embedding."),
                                   (".class_embedding", ".embeddings.class_embedding"),
                                   (".position_embedding", ".embeddings.position_embedding"),
                                   (".layers.", ".encoder.layers."), (".norm1.", ".layer_norm1."),
                                   (".norm2.", ".layer_norm2."), (".attn.qkv.", ".self_attn.qkv."),
                                   (".attn.proj.", ".self_attn.projection.")):
                source = source.replace(target, origin)
        elif source.startswith("qformer.layers."):
            source = source.replace("qformer.layers.", "qformer.encoder.layer.")
            for target, origin in ((".cross_query.", ".crossattention.attention.query."),
                                   (".cross_key.", ".crossattention.attention.key."),
                                   (".cross_value.", ".crossattention.attention.value."),
                                   (".cross_output.", ".crossattention.output."),
                                   (".intermediate.", ".intermediate_query."), (".output.", ".output_query.")):
                # Attention output keeps its own name; only feed-forward output is renamed.
                if target == ".output.":
                    parts = source.split(".")
                    if parts[4] == "output": parts[4] = "output_query"
                    source = ".".join(parts)
                else:
                    source = source.replace(target, origin)
            source = source.replace(".attention.self.", ".attention.attention.")
        elif source.startswith("language_model.") and not source.startswith("language_model.lm_head."):
            source = source.replace("language_model.", "language_model.model.decoder.", 1)
        if source.startswith("qformer.") and ".qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".qkv.", f".{part}.")) for part in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped BLIP2 state: {sorted(remaining)}")
    if not torch.equal(mapped["language_model.embed_tokens.emb.weight"], mapped["language_model.lm_head.weight"]):
        raise ValueError("OPT tied embedding/head weights disagree")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
