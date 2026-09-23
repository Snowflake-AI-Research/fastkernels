"""DAB-DETR's default learned anchors, modulation and iterative box refinement."""

import math
import re
from copy import copy

import torch
from torch import nn

from .conditional_detr import _Layer
from .detr import _ConvEncoder, _EncoderLayer, _attention_mask, _positions, make_workloads
from ..patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.frozen_batch_norm2d import FrozenBatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sam3_position_encoding import Sam3PositionEncoding
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid


class _PReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.relu, self.product = ReLU(), ProductGate()

    def forward(self, hidden):
        negative = self.relu(-hidden)
        return self.relu(hidden) - self.product(torch.cat((negative, self.weight.expand_as(negative)), dim=-1))


class _AttentionCore(nn.Module):
    """DAB uses explicit score storage and FP32 softmax, even for BF16 inputs."""
    def __init__(self):
        super().__init__()
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, query, key, value, attn_mask=None):
        scores = self.bmm((query * query.shape[-1]**-0.5).transpose(1, 2), key.transpose(1, 2).transpose(-2, -1))
        if attn_mask is not None:
            scores = scores + attn_mask
        probabilities = self.softmax(scores.float()).to(query.dtype)
        return self.bmm(probabilities, value.transpose(1, 2)).transpose(1, 2)


class _EncoderAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (Linear(width, width) for _ in range(4))
        self.core = _AttentionCore()

    def forward(self, hidden, positions, mask=None):
        shape = lambda value: value.reshape(*value.shape[:2], self.heads, self.head_dim)
        return self.out_proj(self.core(shape(self.q_proj(hidden + positions)), shape(self.k_proj(hidden + positions)),
                                       shape(self.v_proj(hidden)), attn_mask=mask).reshape(hidden.shape))


class _Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_scale = RTDetrV2MLPPredictionHead(config, config.hidden_size, config.hidden_size, config.hidden_size, 2)
        self.layers = nn.ModuleList([_EncoderLayer(config) for _ in range(config.encoder_layers)])
        for layer in self.layers:
            layer.self_attn = _EncoderAttention(config.hidden_size, config.encoder_attention_heads)
            layer.mlp.activation = _PReLU()
        self.product = ProductGate()

    def forward(self, hidden, positions, mask):
        for layer in self.layers:
            scaled = self.product(torch.cat((positions, self.query_scale(hidden)), dim=-1))
            hidden = layer(hidden, scaled, mask)
        return hidden


class _PositiveRatio(nn.Module):
    """Positive anchor division through existing frozen normalization.

    FP64 holds squared finite positive FP32/BF16 widths without underflow or
    overflow. Zero widths are outside this composition's division domain.
    All conversions and arithmetic remain inside the measured forward.
    """
    def __init__(self):
        super().__init__()
        self.square = SquaredReLU()
        self.normalize = FrozenBatchNorm2d(1, eps=0.0)
        # Unit scale, zero bias/mean, and runtime variance are not model weights.
        self.normalize._non_persistent_buffers_set.update(self.normalize._buffers)

    def forward(self, numerator, denominator):
        self.normalize.running_var = self.square(denominator.double()).flatten()
        output = self.normalize(numerator.double().reshape(1, -1, 1, 1))
        return output.reshape_as(numerator).to(numerator.dtype)


class _Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.width = width
        self.layers = nn.ModuleList([_Layer(config, index == 0) for index in range(config.decoder_layers)])
        for layer in self.layers:
            layer.self_attn.attention, layer.encoder_attn.attention = _AttentionCore(), _AttentionCore()
            layer.mlp.activation = _PReLU()
        self.layernorm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.query_scale = RTDetrV2MLPPredictionHead(config, width, width, width, 2)
        self.ref_point_head = RTDetrV2MLPPredictionHead(config, 2 * width, width, width, 2)
        self.ref_anchor_head = RTDetrV2MLPPredictionHead(config, width, width, 2, 2)
        self.sine, self.sigmoid, self.product, self.ratio = Sam3PositionEncoding(width), Sigmoid(), ProductGate(), _PositiveRatio()

    def forward(self, hidden, initial_reference, memory, positions, mask):
        reference = self.sigmoid(initial_reference)
        references, intermediate = [reference], []
        for index, layer in enumerate(self.layers):
            x, y = self.sine._encode_xy(reference[..., 0].flatten(), reference[..., 1].flatten())
            width, height = self.sine._encode_xy(reference[..., 2].flatten(), reference[..., 3].flatten())
            full_sine = torch.cat((y, x, width, height), dim=-1).reshape(*hidden.shape[:2], -1).to(hidden.dtype)
            query_positions = self.ref_point_head(full_sine)
            sine = full_sine[..., :self.width]
            if index > 0:
                sine = self.product(torch.cat((sine, self.query_scale(hidden)), dim=-1))
            scale = self.ratio(self.sigmoid(self.ref_anchor_head(hidden)), reference[..., 2:])
            # The y sinusoid is scaled by the height ratio; x by the width ratio.
            scale = torch.cat((scale[..., 1:2].expand_as(sine[..., :self.width // 2]),
                               scale[..., :1].expand_as(sine[..., self.width // 2:])), dim=-1)
            sine = self.product(torch.cat((sine, scale), dim=-1))
            hidden = layer(hidden, query_positions, memory, positions, sine, mask)
            reference = self.sigmoid(self.bbox_embed(hidden) + inverse_sigmoid(reference)).detach()
            if index + 1 < len(self.layers):
                references.append(reference)
            intermediate.append(self.layernorm(hidden))
        return self.layernorm(hidden), torch.stack(intermediate), torch.stack(references)


class _Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.hidden_size
        self.temperatures = (config.temperature_width, config.temperature_height)
        self.scale = config.sine_position_embedding_scale or 2 * math.pi
        self.backbone = _ConvEncoder(config)
        self.input_projection = Conv2d(self.backbone.model.channels[-1], self.width, 1)
        self.query_refpoint_embeddings = Embedding(config.num_queries, config.query_dim)
        self.encoder, self.decoder = _Encoder(config), _Decoder(config)

    def forward(self, pixels, pixel_mask=None):
        if pixel_mask is None:
            pixel_mask = torch.ones(pixels.shape[0], *pixels.shape[-2:], device=pixels.device)
        features = self.backbone(pixels, pixel_mask)
        positions = [_positions(mask, self.width, torch.float32, self.temperatures, self.scale).to(feature.dtype)
                     for feature, mask in features][-1]
        projected = self.input_projection(features[-1][0]).flatten(2).transpose(1, 2)
        mask = _attention_mask(features[-1][1].flatten(1), pixels.dtype)
        memory = self.encoder(projected, positions, mask)
        reference = self.query_refpoint_embeddings.emb.weight.unsqueeze(0).expand(pixels.shape[0], -1, -1)
        hidden = torch.zeros(*reference.shape[:2], self.width, device=pixels.device, dtype=pixels.dtype)
        return self.decoder(hidden, reference, memory, positions, mask)


class _Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _Model(config)
        self.class_embed = Linear(config.hidden_size, len(config.id2label))
        self.bbox_predictor = RTDetrV2MLPPredictionHead(config, config.hidden_size, config.hidden_size, 4, 3)
        self.model.decoder.bbox_embed = self.bbox_predictor
        self.sigmoid = Sigmoid()

    def forward(self, pixel_values, pixel_mask=None):
        hidden, intermediate, references = self.model(pixel_values, pixel_mask)
        # HF evaluates refinement for every layer here, then returns the last.
        boxes = self.sigmoid(self.bbox_predictor(intermediate) + inverse_sigmoid(references))
        return {"logits": self.class_embed(intermediate[-1]), "pred_boxes": boxes[-1], "last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if (config.activation_function != "prelu" or config.normalize_before or config.num_patterns != 0
            or config.query_dim != 4 or config.keep_query_pos):
        raise ValueError("This case preserves the published DAB-DETR default PReLU and four-coordinate anchors")
    config = copy(config)
    config.d_model, config.dilation = config.hidden_size, False
    return _Detector(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for key, value in state_dict.items():
        key = key.replace(".backbone.conv_encoder.", ".backbone.")
        key = key.replace(".backbone.model._backbone.", ".backbone.model.")
        key = key.replace("query_refpoint_embeddings.weight", "query_refpoint_embeddings.emb.weight")
        if key.startswith("model.encoder.layers."):
            key = re.sub(r"(layers\.\d+)\.(fc[12])\.", r"\1.mlp.\2.", key)
            key = key.replace(".activation_fn.", ".mlp.activation.")
        elif key.startswith("model.decoder.layers."):
            for source, target in (("query_content", "q_content"), ("query_pos_sine", "q_pos_sine"),
                                   ("query_pos", "q_pos"), ("key_content", "k_content"),
                                   ("key_pos", "k_pos"), ("value", "v")):
                key = key.replace(f".self_attn.self_attn_{source}_proj.", f".self_attn.{target}_proj.")
                key = key.replace(f".cross_attn.cross_attn_{source}_proj.", f".encoder_attn.{target}_proj.")
            key = key.replace(".self_attn.self_attn.output_proj.", ".self_attn.o_proj.")
            key = key.replace(".cross_attn.cross_attn.output_proj.", ".encoder_attn.o_proj.")
            key = key.replace(".self_attn.self_attn_layer_norm.", ".self_attn_layer_norm.")
            key = key.replace(".cross_attn.cross_attn_layer_norm.", ".encoder_attn_layer_norm.")
            key = key.replace(".mlp.final_layer_norm.", ".final_layer_norm.")
            key = key.replace(".activation_fn.", ".activation.")
        mapped[key] = value
    # HF checkpoints omit the duplicate alias of the shared box predictor.
    for key in list(model.state_dict()):
        if key.startswith("model.decoder.bbox_embed.") and key not in mapped:
            mapped[key] = mapped[key.replace("model.decoder.bbox_embed.", "bbox_predictor.")]
    model.load_state_dict(mapped, strict=True)
