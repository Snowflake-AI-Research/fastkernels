"""BridgeTower's interleaved unimodal encoders and bidirectional cross encoders."""

import torch
from torch import nn
from types import SimpleNamespace

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_embeddings import XLMRobertaEmbeddings
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock
from .clip import ClipVisionEmbeddings
from .mvp import EagerAttention
from .roberta_prelayernorm import PreNormAttention
from ..runner import Workload


class CrossLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.cross_query = Linear(config.hidden_size, config.hidden_size)
        self.cross_key = Linear(config.hidden_size, config.hidden_size)
        self.cross_value = Linear(config.hidden_size, config.hidden_size)
        self.cross_output = EncoderSelfOutput(config)
        self.attention.self.attn = EagerAttention(prescale_query=False)
        self.cross_attention = EagerAttention(prescale_query=False)

    def forward(self, hidden, memory):
        batch, length, width = hidden.shape
        heads = self.attention.self.num_attention_heads
        query, key, value = (
            tensor.view(batch, length, heads, width // heads)
            for tensor in self.attention.self._project_qkv(hidden)
        )
        attended, weights = self.attention.self.attn(query, key, value, return_weights=True)
        hidden = self.attention.output(attended.reshape(batch, length, width), hidden)
        query = self.cross_query(hidden).view(batch, length, heads, width // heads)
        key = self.cross_key(memory).view(batch, -1, heads, width // heads)
        value = self.cross_value(memory).view(batch, -1, heads, width // heads)
        hidden = self.cross_output(self.cross_attention(query, key, value).reshape(batch, length, width), hidden)
        return self.output(self.intermediate(hidden), hidden), weights


class BridgeTowerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_depth = config.num_hidden_layers
        text, vision = config.text_config, config.vision_config
        self.text_embeddings = XLMRobertaEmbeddings(text)
        self.text_encoder = BertEncoder(text)
        for layer in self.text_encoder.layer:
            layer.attention.self.attn = EagerAttention(prescale_query=False)
        self.vision_embeddings = ClipVisionEmbeddings(vision)
        self.vision_pre = LayerNorm(vision.hidden_size, eps=vision.layer_norm_eps, promote_fp32=False)
        self.vision_post = LayerNorm(vision.hidden_size, eps=vision.layer_norm_eps, promote_fp32=False)
        self.vision_encoder = nn.ModuleList()
        for _ in range(vision.num_hidden_layers):
            layer = VitEncoderBlock(vision.hidden_size, vision.hidden_size // 64, norm_eps=vision.layer_norm_eps)
            # Native vision MultiheadAttention selects cuDNN for this workload.
            layer.attn = PreNormAttention(SimpleNamespace(
                hidden_size=vision.hidden_size, num_attention_heads=vision.hidden_size // 64,
            ))
            layer.mlp.act = QuickGELU()
            self.vision_encoder.append(layer)
        width = config.hidden_size
        self.cross_modal_text_transform = Linear(text.hidden_size, width)
        self.cross_modal_image_transform = Linear(vision.hidden_size, width)
        self.token_type_embeddings = Embedding(2, width)
        for modality in ("text", "image"):
            setattr(self, f"cross_modal_{modality}_layers", nn.ModuleList([CrossLayer(text) for _ in range(self.cross_depth)]))
            setattr(self, f"cross_modal_{modality}_layernorm", LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False))
            setattr(self, f"cross_modal_{modality}_link_tower", nn.ModuleList([
                nn.ModuleDict({"LayerNorm": LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)})
                for _ in range(self.cross_depth - 1)]))
            setattr(self, f"cross_modal_{modality}_pooler", nn.ModuleDict({"dense": Linear(width, width), "activation": Tanh()}))

    def forward(self, input_ids, pixel_values):
        text = self.text_embeddings(input_ids, self.text_embeddings._resolve_position_ids(input_ids, None, 0))
        vision = self.vision_pre(self.vision_embeddings(pixel_values))
        text_history, image_history, cross_history, attentions = [text], [vision], [], []
        split = len(self.text_encoder.layer) - self.cross_depth + 1
        for index in range(split):
            text = self.text_encoder.layer[index](text)
            vision = self.vision_encoder[index](vision)
            text_history.append(text)
            image_history.append(vision)
        text_type = self.token_type_embeddings(torch.zeros(1, dtype=torch.long, device=text.device))
        image_type = self.token_type_embeddings(torch.ones(1, dtype=torch.long, device=text.device))
        cross_text = self.cross_modal_text_layernorm(self.cross_modal_text_transform(text) + text_type)
        cross_image = self.cross_modal_image_layernorm(self.cross_modal_image_transform(self.vision_post(vision)) + image_type)
        for index in range(self.cross_depth):
            if index:
                text = self.text_encoder.layer[split + index - 1](text)
                vision = self.vision_encoder[split + index - 1](vision)
                text_history.append(text)
                image_history.append(vision)
                cross_text = self.cross_modal_text_link_tower[index - 1]["LayerNorm"](
                    self.cross_modal_text_transform(text) + text_type + cross_text)
                cross_image = self.cross_modal_image_link_tower[index - 1]["LayerNorm"](
                    self.cross_modal_image_transform(self.vision_post(vision)) + image_type + cross_image)
            # Both directions use the same incoming pair, not one updated side.
            text_output, image_output = (self.cross_modal_text_layers[index](cross_text, cross_image),
                                         self.cross_modal_image_layers[index](cross_image, cross_text))
            cross_text, cross_image = text_output[0], image_output[0]
            cross_history.append((cross_text, cross_image))
            attentions.append((text_output[1], image_output[1]))
        text_pool = self.cross_modal_text_pooler["activation"](self.cross_modal_text_pooler["dense"](cross_text[:, 0]))
        image_pool = self.cross_modal_image_pooler["activation"](self.cross_modal_image_pooler["dense"](cross_image[:, 0]))
        outputs = {"text_features": cross_text, "image_features": cross_image,
                   "pooler_output": torch.cat((text_pool, image_pool), dim=-1)}
        # Pinned HF returns these histories unconditionally. Its vision tower
        # stores sequence-first tensors; the other histories are batch-first.
        for index, hidden in enumerate(text_history):
            outputs[f"hidden_states.0.{index}"] = hidden
        for index, hidden in enumerate(image_history):
            outputs[f"hidden_states.1.{index}"] = hidden.transpose(0, 1)
        for index, pair in enumerate(cross_history):
            for modality, hidden in enumerate(pair):
                outputs[f"hidden_states.2.{index}.{modality}"] = hidden
        for index, pair in enumerate(attentions):
            for modality, weights in enumerate(pair):
                outputs[f"attentions.{index}.{modality}"] = weights
        return outputs


def build_from_config(config, device, dtype):
    if (not config.share_cross_modal_transformer_layers or config.share_link_tower_layers
            or config.link_tower_type != "add" or not config.vision_config.share_layernorm
            or config.vision_config.vit_remove_last or config.text_config.hidden_act != "gelu"):
        raise ValueError("This adapter preserves the documented base checkpoint's shared transforms and additive links")
    return BridgeTowerModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        for target, origin in (("text_embeddings.", "text_model.embeddings."), ("text_encoder.", "text_model.encoder."),
                               ("vision_embeddings.", "vision_model.visual.embeddings."), ("vision_pre.", "vision_model.visual.ln_pre."),
                               ("vision_post.", "vision_model.visual.ln_post."), ("vision_encoder.", "vision_model.visual.transformer.resblocks."),
                               (".cross_query.", ".crossattention.self.query."), (".cross_key.", ".crossattention.self.key."),
                               (".cross_value.", ".crossattention.self.value."), (".cross_output.", ".crossattention.output."),
                               (".patch_embedding.proj.", ".patch_embedding.")):
            source = source.replace(target, origin)
        if name.startswith("vision_encoder."):
            for target, origin in ((".norm1.", ".ln_1."), (".norm2.", ".ln_2."),
                                   (".mlp.fc1.", ".mlp.c_fc."), (".mlp.fc2.", ".mlp.c_proj."),
                                   (".attn.proj.", ".attn.out_proj."), (".attn.qkv.", ".attn.in_proj_")):
                source = source.replace(target, origin)
            mapped[name] = remaining.pop(source)
        elif ".qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".qkv.", f".{part}.")) for part in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    # The base wrapper directly runs the text encoder and does not call its pooler.
    inactive = {"text_model.pooler.dense.weight", "text_model.pooler.dense.bias"}
    if set(remaining) != inactive:
        raise KeyError(f"Unexpected BridgeTower unused weights: {sorted(set(remaining) ^ inactive)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
