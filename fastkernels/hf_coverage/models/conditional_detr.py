"""Conditional DETR with its learned query positions and default detection heads."""

import torch
from torch import nn

from .detr import _Model, _MLP, _check_config, make_workloads
from ..patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sam3_position_encoding import Sam3PositionEncoding
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid


class _Attention(nn.Module):
    def __init__(self, width, heads, cross=False, first=True):
        super().__init__()
        self.heads, self.head_dim, self.cross = heads, width // heads, cross
        for name in ("q_content_proj", "q_pos_proj", "k_content_proj", "k_pos_proj", "v_proj", "o_proj"):
            self.add_module(name, Linear(width, width))
        if cross:
            self.q_pos_sine_proj = Linear(width, width)
            if not first:
                self.q_pos_proj = None
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, hidden, query_positions, memory=None, positions=None, sine=None, mask=None):
        shape = lambda value: value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim)
        if self.cross:
            query, key = self.q_content_proj(hidden), self.k_content_proj(memory)
            key_position = self.k_pos_proj(positions)
            if self.q_pos_proj is not None:
                query, key = query + self.q_pos_proj(query_positions), key + key_position
            query = torch.cat((shape(query), shape(self.q_pos_sine_proj(sine))), dim=-1)
            key = torch.cat((shape(key), shape(key_position)), dim=-1)
            value = shape(self.v_proj(memory))
        else:
            query = shape(self.q_content_proj(hidden) + self.q_pos_proj(query_positions))
            key = shape(self.k_content_proj(hidden) + self.k_pos_proj(query_positions))
            value = shape(self.v_proj(hidden))
        return self.o_proj(self.attention(query, key, value, attn_mask=mask).reshape(hidden.shape))


class _Layer(nn.Module):
    def __init__(self, config, first):
        super().__init__()
        width = config.d_model
        self.self_attn = _Attention(width, config.decoder_attention_heads)
        self.encoder_attn = _Attention(width, config.decoder_attention_heads, cross=True, first=first)
        self.self_attn_layer_norm, self.encoder_attn_layer_norm, self.final_layer_norm = (
            LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(3))
        self.mlp = _MLP(width, config.decoder_ffn_dim)

    def forward(self, hidden, query_positions, memory, positions, sine, mask):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, query_positions))
        hidden = self.encoder_attn_layer_norm(hidden + self.encoder_attn(
            hidden, query_positions, memory, positions, sine, mask))
        return self.final_layer_norm(hidden + self.mlp(hidden))


class _Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.layers = nn.ModuleList([_Layer(config, index == 0) for index in range(config.decoder_layers)])
        self.layernorm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.query_scale = RTDetrV2MLPPredictionHead(config, width, width, width, 2)
        self.ref_point_head = RTDetrV2MLPPredictionHead(config, width, width, 2, 2)
        self.sine = Sam3PositionEncoding(width)
        self.sigmoid, self.product = Sigmoid(), ProductGate()

    def forward(self, hidden, query_positions, memory, positions, mask):
        reference = self.sigmoid(self.ref_point_head(query_positions))
        x, y = self.sine._encode_xy(reference[..., 0].flatten(), reference[..., 1].flatten())
        base = torch.cat((y, x), dim=-1).reshape(query_positions.shape).to(hidden.dtype)
        for index, layer in enumerate(self.layers):
            sine = base if index == 0 else self.product(torch.cat((base, self.query_scale(hidden)), dim=-1))
            hidden = layer(hidden, query_positions, memory, positions, sine, mask)
        return self.layernorm(hidden), reference


class _Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _Model(config)
        self.model.decoder = _Decoder(config)
        self.class_labels_classifier = Linear(config.d_model, len(config.id2label))
        self.bbox_predictor = RTDetrV2MLPPredictionHead(config, config.d_model, config.d_model, 4, 3)
        self.sigmoid = Sigmoid()

    def forward(self, pixel_values, pixel_mask=None):
        memory, positions, query_positions, mask, _, _ = self.model.prepare(pixel_values, pixel_mask)
        hidden, reference = self.model.decoder(torch.zeros_like(query_positions), query_positions, memory, positions, mask)
        boxes = self.bbox_predictor(hidden)
        boxes = torch.cat((boxes[..., :2] + inverse_sigmoid(reference), boxes[..., 2:]), dim=-1)
        return {"logits": self.class_labels_classifier(hidden), "pred_boxes": self.sigmoid(boxes),
                "last_hidden_state": hidden, "encoder_last_hidden_state": memory}


def build_from_config(config, device, dtype):
    _check_config(config)
    model = _Detector(config)
    # Match native HF's SDPA backend without depending on global import state.
    for layer in model.modules():
        if isinstance(getattr(layer, "attention", None), DenseAttention):
            layer.attention = DenseAttention(backend="cudnn")
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for key, value in state_dict.items():
        key = key.replace("query_position_embeddings.weight", "query_position_embeddings.emb.weight")
        if key.startswith("model.encoder."):
            key = key.replace(".o_proj.", ".out_proj.")
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)
