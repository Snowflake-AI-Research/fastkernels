"""Default DETR panoptic inference and shared detector construction components."""

import math

import torch
from torch import nn

from ..runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.frozen_batch_norm2d import FrozenBatchNorm2d
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead
from fastkernels.tasks.baseline.L2.rtdetrv2_multihead_attention import RTDetrV2MultiheadAttention


class _ResNetBlock(nn.Module):
    def __init__(self, source, planes, stride, bottleneck):
        super().__init__()
        target = planes * (4 if bottleneck else 1)
        self.conv1 = Conv2d(source, planes, 1 if bottleneck else 3,
                            stride=1 if bottleneck else stride, padding=0 if bottleneck else 1, bias=False)
        self.bn1, self.activation = FrozenBatchNorm2d(planes), ReLU()
        self.conv2 = Conv2d(planes, planes, 3, stride=stride if bottleneck else 1, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(planes)
        if bottleneck:
            self.conv3, self.bn3 = Conv2d(planes, target, 1, bias=False), FrozenBatchNorm2d(target)
        self.downsample = (nn.Sequential(Conv2d(source, target, 1, stride=stride, bias=False),
                                        FrozenBatchNorm2d(target)) if stride != 1 or source != target else None)

    def forward(self, hidden):
        residual = hidden if self.downsample is None else self.downsample(hidden)
        hidden = self.activation(self.bn1(self.conv1(hidden)))
        hidden = self.bn2(self.conv2(hidden))
        if hasattr(self, "conv3"):
            hidden = self.bn3(self.conv3(self.activation(hidden)))
        return self.activation(hidden + residual)


class _ResNetFeatures(nn.Module):
    def __init__(self, name):
        super().__init__()
        bottleneck = name == "resnet50"
        if name not in ("resnet18", "resnet50"):
            raise ValueError("These detector defaults use timm ResNet18 or ResNet50")
        depths = (3, 4, 6, 3) if bottleneck else (2, 2, 2, 2)
        self.channels = [width * (4 if bottleneck else 1) for width in (64, 128, 256, 512)]
        self.conv1, self.bn1 = Conv2d(3, 64, 7, stride=2, padding=3, bias=False), FrozenBatchNorm2d(64)
        self.activation, self.maxpool = ReLU(), MaxPool2d(3, stride=2, padding=1)
        source = 64
        for index, (planes, depth) in enumerate(zip((64, 128, 256, 512), depths), start=1):
            blocks = []
            for layer in range(depth):
                stride = 2 if index > 1 and layer == 0 else 1
                blocks.append(_ResNetBlock(source, planes, stride, bottleneck))
                source = planes * (4 if bottleneck else 1)
            self.add_module(f"layer{index}", nn.Sequential(*blocks))

    def forward(self, pixels):
        hidden, features = self.maxpool(self.activation(self.bn1(self.conv1(pixels)))), []
        for index in range(1, 5):
            hidden = self.get_submodule(f"layer{index}")(hidden)
            features.append(hidden)
        return features


class _ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = config.backbone_config
        if backbone.model_type != "timm_backbone" or backbone.out_indices != [1, 2, 3, 4] or config.dilation:
            raise ValueError("These detector cases retain the default four-stage, undilated timm backbone")
        self.model = _ResNetFeatures(backbone.backbone)
        self.interpolate = Interpolate()

    def forward(self, pixels, mask):
        return [(feature, self.interpolate(mask[None].float(), size=feature.shape[-2:]).bool()[0])
                for feature in self.model(pixels)]


def _positions(mask, width, dtype, temperatures=(10000, 10000), scale=2 * math.pi):
    """Position metadata only: no feature values enter these mask/index formulas."""
    y, x = mask.cumsum(1, dtype=dtype), mask.cumsum(2, dtype=dtype)
    y, x = y / (y[:, -1:, :] + 1e-6) * scale, x / (x[:, :, -1:] + 1e-6) * scale
    dim = torch.arange(width // 2, device=mask.device, dtype=torch.int64).to(dtype)
    exponent = 2 * torch.div(dim, 2, rounding_mode="floor") / (width // 2)
    x, y = x[..., None] / temperatures[0]**exponent, y[..., None] / temperatures[1]**exponent
    x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(3)
    y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=-1).flatten(3)
    return torch.cat((y, x), dim=-1).flatten(1, 2)


def _attention_mask(mask, dtype):
    # HF omits an all-valid padding mask. Keeping a zero additive mask can
    # select a different SDPA kernel; this metadata check remains timed.
    if bool(mask.all()):
        return None
    return torch.zeros(mask.shape[0], 1, 1, mask.shape[1], device=mask.device, dtype=dtype).masked_fill(
        ~mask[:, None, None, :], torch.finfo(dtype).min)


class _Attention(RTDetrV2MultiheadAttention):
    def __init__(self, width, heads, backend="sdpa"):
        super().__init__(width, heads)
        self.heads, self.head_dim, self.backend = heads, width // heads, backend
        self.attention, self.bmm, self.softmax = DenseAttention(backend="sdpa"), BMM(), Softmax(dim=-1)

    def forward(self, hidden, positions, memory=None, memory_positions=None, mask=None):
        if self.backend == "eager" and memory is None and (mask is None or mask.shape[0] == 1):
            # The existing self-attention accepts a shared two-dimensional mask.
            shared_mask = None if mask is None else mask[0, 0]
            return super().forward(hidden, attention_mask=shared_mask, position_embeddings=positions)[0]
        source = hidden if memory is None else memory
        key_positions = positions if memory is None else memory_positions
        shape = lambda value: value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim)
        query, key, value = shape(self.q_proj(hidden + positions)), shape(self.k_proj(source + key_positions)), shape(self.v_proj(source))
        if self.backend == "sdpa":
            output = self.attention(query, key, value, attn_mask=mask)
        else:
            query = (query * self.head_dim**-0.5).transpose(1, 2)
            key, value = key.transpose(1, 2), value.transpose(1, 2)
            scores = self.bmm(query, key.transpose(-2, -1))
            if mask is not None:
                scores = scores + mask
            output = self.bmm(self.softmax(scores), value).transpose(1, 2)
        return self.out_proj(output.reshape(hidden.shape))


class _MLP(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.fc1, self.fc2, self.activation = Linear(width, intermediate), Linear(intermediate, width), ReLU()

    def forward(self, hidden):
        return self.fc2(self.activation(self.fc1(hidden)))


class _EncoderLayer(nn.Module):
    def __init__(self, config, prenorm=False, backend="sdpa"):
        super().__init__()
        width = config.d_model
        self.prenorm = prenorm
        self.self_attn = _Attention(width, config.encoder_attention_heads, backend)
        self.self_attn_layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.final_layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.mlp = _MLP(width, config.encoder_ffn_dim)

    def forward(self, hidden, positions, mask):
        if self.prenorm:
            hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden), positions, mask=mask)
            return hidden + self.mlp(self.final_layer_norm(hidden))
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, positions, mask=mask))
        return self.final_layer_norm(hidden + self.mlp(hidden))


class _DecoderLayer(nn.Module):
    def __init__(self, config, prenorm=False, backend="sdpa"):
        super().__init__()
        width = config.d_model
        self.prenorm = prenorm
        self.self_attn, self.encoder_attn = (_Attention(width, config.decoder_attention_heads, backend) for _ in range(2))
        self.self_attn_layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.encoder_attn_layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.final_layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.mlp = _MLP(width, config.decoder_ffn_dim)

    def forward(self, hidden, query_positions, memory, positions, mask):
        if self.prenorm:
            hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden), query_positions)
            hidden = hidden + self.encoder_attn(self.encoder_attn_layer_norm(hidden), query_positions, memory, positions, mask)
            return hidden + self.mlp(self.final_layer_norm(hidden))
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, query_positions))
        hidden = self.encoder_attn_layer_norm(hidden + self.encoder_attn(hidden, query_positions, memory, positions, mask))
        return self.final_layer_norm(hidden + self.mlp(hidden))


class _Encoder(nn.Module):
    def __init__(self, config, prenorm=False, backend="sdpa"):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayer(config, prenorm, backend) for _ in range(config.encoder_layers)])
        if prenorm:
            self.layernorm = LayerNorm(config.d_model, eps=1e-5, promote_fp32=False)

    def forward(self, hidden, positions, mask):
        for layer in self.layers:
            hidden = layer(hidden, positions, mask)
        return self.layernorm(hidden) if hasattr(self, "layernorm") else hidden


class _Decoder(nn.Module):
    def __init__(self, config, prenorm=False, backend="sdpa"):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(config, prenorm, backend) for _ in range(config.decoder_layers)])
        self.layernorm = LayerNorm(config.d_model, eps=1e-5, promote_fp32=False)

    def forward(self, queries, query_positions, memory, positions, mask):
        for layer in self.layers:
            queries = layer(queries, query_positions, memory, positions, mask)
        return self.layernorm(queries)


class _Model(nn.Module):
    def __init__(self, config, prenorm=False, backend="sdpa"):
        super().__init__()
        self.width, self.all_positions = config.d_model, prenorm
        self.backbone = _ConvEncoder(config)
        self.input_projection = Conv2d(self.backbone.model.channels[-1], self.width, 1)
        self.query_position_embeddings = Embedding(config.num_queries, self.width)
        self.encoder, self.decoder = _Encoder(config, prenorm, backend), _Decoder(config, prenorm, backend)

    def prepare(self, pixels, pixel_mask=None):
        if pixel_mask is None:
            pixel_mask = torch.ones(pixels.shape[0], *pixels.shape[-2:], device=pixels.device)
        features = self.backbone(pixels, pixel_mask)
        projected = self.input_projection(features[-1][0])
        if self.all_positions:
            # Table Transformer computes position maps for every backbone stage.
            positions = [_positions(mask, self.width, torch.float32).to(feature.dtype) for feature, mask in features][-1]
        else:
            positions = _positions(features[-1][1], self.width, pixels.dtype)
        mask = _attention_mask(features[-1][1].flatten(1), projected.dtype)
        memory = self.encoder(projected.flatten(2).transpose(1, 2), positions, mask)
        query_positions = self.query_position_embeddings.emb.weight.unsqueeze(0).expand(pixels.shape[0], -1, -1)
        return memory, positions, query_positions, mask, projected, features

    def forward(self, pixels, pixel_mask=None):
        memory, positions, query_positions, mask, projected, features = self.prepare(pixels, pixel_mask)
        hidden = self.decoder(torch.zeros_like(query_positions), query_positions, memory, positions, mask)
        return hidden, memory, projected, features


class _ObjectDetection(nn.Module):
    def __init__(self, config, prenorm=False, backend="sdpa"):
        super().__init__()
        self.model = _Model(config, prenorm, backend)
        self.class_labels_classifier = Linear(config.d_model, len(config.id2label) + 1)
        self.bbox_predictor = RTDetrV2MLPPredictionHead(config, config.d_model, config.d_model, 4, 3)
        self.sigmoid = Sigmoid()

    def forward(self, pixel_values, pixel_mask=None):
        hidden, memory, _, _ = self.model(pixel_values, pixel_mask)
        return {"logits": self.class_labels_classifier(hidden), "pred_boxes": self.sigmoid(self.bbox_predictor(hidden)),
                "last_hidden_state": hidden, "encoder_last_hidden_state": memory}


class _ConvBlock(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.conv, self.norm, self.activation = Conv2d(source, target, 3, padding=1), GroupNorm(min(8, target), target, eps=1e-5), ReLU()

    def forward(self, hidden):
        return self.activation(self.norm(self.conv(hidden)))


class _FusionStage(nn.Module):
    def __init__(self, source, current, target):
        super().__init__()
        self.fpn_adapter, self.refine = Conv2d(source, current, 1), _ConvBlock(current, target)
        self.interpolate = Interpolate()

    def forward(self, hidden, feature):
        feature = self.fpn_adapter(feature)
        return self.refine(feature + self.interpolate(hidden, size=feature.shape[-2:], mode="nearest"))


class _MaskHead(nn.Module):
    def __init__(self, width, heads, channels):
        super().__init__()
        self.conv1, self.conv2 = _ConvBlock(width + heads, width + heads), _ConvBlock(width + heads, width // 2)
        self.fpn_stages = nn.ModuleList([_FusionStage(channels[i], width // 2**(i + 1), width // 2**(i + 2)) for i in range(3)])
        self.output_conv = Conv2d(width // 16, 1, 3, padding=1)

    def forward(self, features, attention, pyramid):
        queries = attention.shape[1]
        expand = lambda value: value.unsqueeze(1).expand(-1, queries, -1, -1, -1).flatten(0, 1)
        hidden = self.conv2(self.conv1(torch.cat((expand(features), attention.flatten(0, 1)), dim=1)))
        for layer, feature in zip(self.fpn_stages, pyramid):
            hidden = layer(hidden, expand(feature))
        return self.output_conv(hidden)


class _AttentionMap(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q_proj, self.k_proj = Linear(width, width), Conv2d(width, width, 1)
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, query, memory, mask):
        batch, count, width = query.shape
        height, spatial_width = memory.shape[-2:]
        query = self.q_proj(query).reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(memory).reshape(batch, self.heads, self.head_dim, height * spatial_width)
        attention = self.bmm(query * self.head_dim**-0.5, key).reshape(batch, self.heads, count, height, spatial_width).transpose(1, 2)
        attention = attention + mask[:, None, None]
        return self.softmax(attention.flatten(2)).reshape_as(attention)


class DetrForSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.detr = _ObjectDetection(config)
        self.mask_head = _MaskHead(config.d_model, config.encoder_attention_heads, [1024, 512, 256])
        self.bbox_attention = _AttentionMap(config.d_model, config.encoder_attention_heads)

    def forward(self, pixel_values, pixel_mask=None):
        hidden, memory, projected, features = self.detr.model(pixel_values, pixel_mask)
        batch, width, height, spatial_width = projected.shape
        mask = torch.zeros_like(features[-1][1], dtype=memory.dtype).masked_fill(~features[-1][1], torch.finfo(memory.dtype).min)
        attention = self.bbox_attention(hidden, memory.transpose(1, 2).reshape(batch, width, height, spatial_width), mask)
        masks = self.mask_head(projected, attention, [feature for feature, _ in features[:3]][::-1])
        return {"logits": self.detr.class_labels_classifier(hidden),
                "pred_boxes": self.detr.sigmoid(self.detr.bbox_predictor(hidden)),
                "pred_masks": masks.reshape(batch, hidden.shape[1], *masks.shape[-2:]),
                "last_hidden_state": hidden, "encoder_last_hidden_state": memory}


def _check_config(config):
    if config.activation_function != "relu" or config.position_embedding_type != "sine" or config.auxiliary_loss:
        raise ValueError("These detector cases use default ReLU/sine inference without auxiliary training outputs")


def build_from_config(config, device, dtype):
    _check_config(config)
    model = DetrForSegmentation(config)
    # Native HF SDPA selects cuDNN on the evaluated GPU. Reuse the explicit
    # backend because other library imports can disable its global selection.
    for layer in model.modules():
        if isinstance(layer, _Attention):
            layer.attention = DenseAttention(backend="cudnn")
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace(".o_proj.", ".out_proj.").replace(".query_position_embeddings.weight", ".query_position_embeddings.emb.weight")
        if name == "bbox_attention.k_proj.weight":
            value = value[:, :, None, None]
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
