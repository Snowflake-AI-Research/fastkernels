"""Default Mask2Former inference, including intermediate mask and class heads."""

import math
import re
import torch
from torch import nn

from .deformable_detr import _SamplingAttention, _encoder_references
from .detr import _Attention, _MLP, _positions, make_workloads
from .swin import SwinModel, _WindowBlock, load_state_dict_into as load_swin
from ..patches.codec_top1 import CodecTop1
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L2.sam3_mask_predictor import Sam3MaskPredictor
from fastkernels.tasks.baseline.L3.swinv2_block import window_partition, window_reverse


class _BackboneWindow(_WindowBlock):
    """HF backbones always partition, retaining shifts even at one-window size."""
    def __init__(self, config, stage, index, resolution):
        super().__init__(config, stage, index, resolution)
        self.shift = config.window_size // 2 if index % 2 else 0
        self.pad, self.bmm, self.softmax = Pad(), BMM(), Softmax(dim=-1)

    def forward(self, hidden):
        batch, height, width, channels = hidden.shape
        normalized = self.block.norm1(hidden)
        normalized = self.pad(normalized, (0, 0, 0, -width % self.window, 0, -height % self.window))
        padded_height, padded_width = normalized.shape[1:3]
        if self.shift:
            normalized = torch.roll(normalized, (-self.shift, -self.shift), (1, 2))
        windows = window_partition(normalized, (self.window, self.window)).reshape(-1, self.window**2, channels)
        bias = self.relative_position_bias_table[self.relative_position_index.flatten()]
        bias = bias.reshape(self.window**2, self.window**2, -1).permute(2, 0, 1).unsqueeze(0)
        mask = None
        if self.shift:
            # Region identifiers and mask depend only on shape/window metadata.
            regions = torch.zeros(1, padded_height, padded_width, 1, device=hidden.device)
            slices = (slice(0, -self.window), slice(-self.window, -self.shift), slice(-self.shift, None))
            for row, rows in enumerate(slices):
                for col, columns in enumerate(slices):
                    regions[:, rows, columns] = row * 3 + col
            regions = window_partition(regions, (self.window, self.window)).reshape(-1, self.window**2)
            mask = regions[:, None, :] - regions[:, :, None]
            mask = mask.masked_fill(mask != 0, -100).masked_fill(mask == 0, 0).to(hidden.dtype)
        # The native backbone stores BF16 scores, scales, then adds each bias.
        # A fused SDPA call skips those rounding boundaries (checked on shared
        # trained-model inputs); preserve them with existing BMM/Softmax ops.
        attention = self.block.attn
        count = self.window**2
        query, key, value = attention.qkv(windows).reshape(-1, count, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        scores = self.bmm(query, key.transpose(-1, -2)) / math.sqrt(attention.head_dim)
        scores = scores + bias
        if mask is not None:
            scores = (scores.reshape(batch, -1, attention.num_heads, count, count) + mask[None, :, None]).flatten(0, 1)
        windows = self.bmm(self.softmax(scores), value).transpose(1, 2).reshape(-1, count, channels)
        windows = attention.proj(windows).reshape(-1, self.window, self.window, channels)
        attended = window_reverse(windows, (self.window, self.window), (padded_height, padded_width))
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), (1, 2))
        hidden = hidden + attended[:, :height, :width]
        return hidden + self.block.mlp(self.block.norm2(hidden))


class _SwinBackbone(SwinModel):
    def __init__(self, config):
        if config.use_absolute_embeddings:
            raise ValueError("This Swin backbone requires the selected relative-position configuration")
        super().__init__(config)
        self.pad = Pad()
        self.norm = nn.Identity()
        del self.pool
        for stage_index, stage in enumerate(self.stages):
            resolution = config.image_size // config.patch_size // 2**stage_index
            stage.blocks = nn.ModuleList([_BackboneWindow(config, stage_index, index, resolution)
                                          for index in range(config.depths[stage_index])])
        self.hidden_states_norms = nn.ModuleDict({f"stage{i + 1}": LayerNorm(
            config.embed_dim * 2**i, eps=config.layer_norm_eps, promote_fp32=False) for i in range(4)})

    def embed_pixels(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1] != self.input_shape[0]:
            raise ValueError("Expected NCHW pixels with the configured channel count")
        patch_height, patch_width = self.patch_embed.patch_size
        pad_height, pad_width = -pixels.shape[-2] % patch_height, -pixels.shape[-1] % patch_width
        if pad_height or pad_width:
            pixels = self.pad(pixels, (0, pad_width, 0, pad_height))
        return self.patch_embed(pixels, random_sample=True)

    def downsample(self, hidden, stage):
        if not isinstance(stage.downsample, nn.Identity) and (hidden.shape[1] % 2 or hidden.shape[2] % 2):
            # Native Swin pads incomplete 2x2 groups before its unchanged
            # normalization and projection. Window padding is separate.
            hidden = self.pad(hidden, (0, 0, 0, hidden.shape[2] % 2,
                                      0, hidden.shape[1] % 2))
        return stage.downsample(hidden)

    def forward(self, pixels):
        hidden, features = self.embed_pixels(pixels), []
        for index, stage in enumerate(self.stages, start=1):
            for block in stage.blocks:
                hidden = block(hidden)
            features.append(self.hidden_states_norms[f"stage{index}"](hidden).permute(0, 3, 1, 2).contiguous())
            hidden = self.downsample(hidden, stage)
        return features


def _load_swin_backbone(model, state, config):
    state = dict(state)
    norms = {key: state.pop(key) for key in list(state) if key.startswith("hidden_states_norms.")}
    # The existing strict loader owns its stage and embedding mapping.
    saved = model.hidden_states_norms
    del model.hidden_states_norms
    load_swin(model, state, config)
    model.hidden_states_norms = saved
    saved.load_state_dict({key[len("hidden_states_norms."):]: value for key, value in norms.items()}, strict=True)


def _feature_positions(feature, width):
    mask = torch.ones(feature.shape[0], *feature.shape[-2:], device=feature.device, dtype=torch.bool)
    return _positions(mask, width, feature.dtype)


class _PixelLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.feature_size
        self.self_attn = _SamplingAttention(width, config.num_attention_heads, 3, 4)
        self.self_attn_layer_norm, self.final_layer_norm = (LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(2))
        self.mlp = _MLP(width, config.encoder_feedforward_dim)

    def forward(self, hidden, positions, references, shapes):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, hidden, positions, references, shapes))
        return self.final_layer_norm(hidden + self.mlp(hidden))


class _PixelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.feature_size
        channels = [config.backbone_config.embed_dim * 2**i for i in range(4)]
        self.width = width
        self.level_embed = nn.Parameter(torch.zeros(3, width))
        self.input_projections = nn.ModuleList([nn.Sequential(Conv2d(source, width, 1), GroupNorm(32, width, eps=1e-5))
                                               for source in channels[:0:-1]])
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_PixelLayer(config) for _ in range(config.encoder_layers)])
        self.mask_projection = Conv2d(width, config.mask_feature_size, 1)
        self.adapter_1 = nn.Sequential(Conv2d(channels[0], width, 1, bias=False), GroupNorm(32, width, eps=1e-5))
        self.layer_1 = nn.Sequential(Conv2d(width, width, 3, padding=1, bias=False), GroupNorm(32, width, eps=1e-5), ReLU())
        self.interpolate = Interpolate()

    def forward(self, features):
        features_reversed = features[:0:-1]
        inputs = [projection(feature) for projection, feature in zip(self.input_projections, features_reversed)]
        shapes = [tuple(feature.shape[-2:]) for feature in inputs]
        positions = torch.cat([_feature_positions(feature, self.width) + self.level_embed[index].view(1, 1, -1)
                               for index, feature in enumerate(features_reversed)], dim=1)
        hidden = torch.cat([feature.flatten(2).transpose(1, 2) for feature in inputs], dim=1)
        ratios = torch.ones(hidden.shape[0], 3, 2, device=hidden.device, dtype=hidden.dtype)
        references = _encoder_references(shapes, ratios)
        for layer in self.encoder.layers:
            hidden = layer(hidden, positions, references, shapes)
        outputs = [value.transpose(1, 2).reshape(hidden.shape[0], self.width, height, width)
                   for value, (height, width) in zip(hidden.split([h * w for h, w in shapes], dim=1), shapes)]
        lateral = self.adapter_1(features[0])
        fused = self.layer_1(lateral + self.interpolate(outputs[-1], size=lateral.shape[-2:], mode="bilinear", align_corners=False))
        return self.mask_projection(fused), outputs


class _CrossAttention(nn.Module):
    """Compose native MHA score storage and its unconditionally computed head mean."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (Linear(width, width) for _ in range(4))
        self.bmm, self.softmax, self.reduce = BMM(), Softmax(dim=-1), SegmentCSR()

    def forward(self, hidden, query_positions, memory, positions, mask):
        batch, queries = hidden.shape[:2]
        shape = lambda value: value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        query = shape(self.q_proj(hidden + query_positions)) * self.head_dim**-0.5
        key, value = shape(self.k_proj(memory + positions)), shape(self.v_proj(memory))
        scores = self.bmm(query, key.transpose(-2, -1)).masked_fill(mask[:, None], float("-inf"))
        probabilities = self.softmax(scores)
        output = self.out_proj(self.bmm(probabilities, value).transpose(1, 2).reshape(batch, queries, -1))
        rows = probabilities.permute(0, 2, 3, 1).contiguous().flatten()
        offsets = torch.arange(0, rows.numel() + 1, self.heads, device=rows.device)
        self.reduce(rows, offsets, reduce="mean")
        return output


class _DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.self_attn = _Attention(width, config.num_attention_heads, backend="eager")
        self.cross_attn = _CrossAttention(width, config.num_attention_heads)
        self.self_attn_layer_norm, self.cross_attn_layer_norm, self.final_layer_norm = (
            LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(3))
        self.mlp = _MLP(width, config.dim_feedforward)

    def forward(self, hidden, query_positions, memory, positions, mask):
        hidden = self.cross_attn_layer_norm(hidden + self.cross_attn(hidden, query_positions, memory, positions, mask))
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, query_positions))
        return self.final_layer_norm(hidden + self.mlp(hidden))


class _MaskPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_embedder = Sam3MaskPredictor(config.hidden_dim, config.mask_feature_size)
        self.interpolate, self.sigmoid, self.compare = Interpolate(), Sigmoid(), CodecTop1()

    def forward(self, hidden, pixels, size):
        masks = self.mask_embedder(hidden, pixels)
        probabilities = self.sigmoid(self.interpolate(masks, size=size, mode="bilinear", align_corners=False)).flatten(2)
        # CUB chooses index0 on equality, so index1 means probability < .5.
        masked = self.compare(torch.stack((probabilities, torch.full_like(probabilities, .5)), dim=-1)).bool()
        return masks, masked


class _Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.width = width
        self.queries_embedder, self.queries_features = Embedding(config.num_queries, width), Embedding(config.num_queries, width)
        self.level_embed = Embedding(3, width)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.decoder_layers - 1)])
        self.decoder.layernorm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.decoder.mask_predictor = _MaskPredictor(config)
        self.reduce = SegmentCSR()

    def forward(self, features, pixels):
        sizes = [feature.shape[-2:] for feature in features]
        positions = [_feature_positions(feature, self.width) for feature in features]
        memories = [feature.flatten(2).transpose(1, 2) + self.level_embed.emb.weight[index][None, None]
                    for index, feature in enumerate(features)]
        batch = pixels.shape[0]
        query_positions = self.queries_embedder.emb.weight.unsqueeze(0).expand(batch, -1, -1)
        hidden = self.queries_features.emb.weight.unsqueeze(0).expand(batch, -1, -1)
        intermediate = [self.decoder.layernorm(hidden)]
        masks, mask = self.decoder.mask_predictor(intermediate[-1], pixels, sizes[0])
        predictions = [masks]
        for index, layer in enumerate(self.decoder.layers):
            # A row with every location masked must become an unmasked row.
            offsets = torch.arange(0, mask.numel() + 1, mask.shape[-1], device=mask.device)
            all_masked = self.reduce(mask.float().flatten(), offsets, reduce="min").bool().reshape(mask.shape[:2])
            mask = mask.masked_fill(all_masked[..., None], False)
            hidden = layer(hidden, query_positions, memories[index % 3], positions[index % 3], mask)
            intermediate.append(self.decoder.layernorm(hidden))
            masks, mask = self.decoder.mask_predictor(intermediate[-1], pixels, sizes[(index + 1) % 3])
            predictions.append(masks)
        return hidden, intermediate, predictions


class _Segmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.pixel_level_module = nn.Module()
        self.model.pixel_level_module.encoder = _SwinBackbone(config.backbone_config)
        self.model.pixel_level_module.decoder = _PixelDecoder(config)
        self.model.transformer_module = _Transformer(config)
        self.class_predictor = Linear(config.hidden_dim, len(config.id2label) + 1)
        self.criterion = nn.Module()
        self.criterion.register_buffer("empty_weight", torch.ones(len(config.id2label) + 1))

    def forward(self, pixel_values):
        features = self.model.pixel_level_module.encoder(pixel_values)
        pixels, multi_scale = self.model.pixel_level_module.decoder(features)
        hidden, intermediate, masks = self.model.transformer_module(multi_scale, pixels)
        classes = [self.class_predictor(value) for value in intermediate]
        return {"class_queries_logits": classes[-1], "masks_queries_logits": masks[-1],
                "encoder_last_hidden_state": features[-1], "pixel_decoder_last_hidden_state": pixels,
                "transformer_decoder_last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if (config.backbone_config.model_type != "swin" or config.pre_norm or config.enforce_input_projection
            or config.feature_size != config.hidden_dim or config.common_stride != 4 or config.output_auxiliary_logits
            or config.activation_function != "relu"):
        raise ValueError("This case preserves the published Swin-small, post-norm, three-scale Mask2Former default")
    return _Segmentation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    state = dict(state_dict)
    prefix = "model.pixel_level_module.encoder."
    backbone = {key[len(prefix):]: state.pop(key) for key in list(state) if key.startswith(prefix)}
    _load_swin_backbone(model.model.pixel_level_module.encoder, backbone, config.backbone_config)
    mapped = {prefix + key: value for key, value in model.model.pixel_level_module.encoder.state_dict().items()}
    for key, value in state.items():
        if ".cross_attn.in_proj_" in key:
            field = key.rsplit("_", 1)[-1]
            for name, part in zip(("q", "k", "v"), value.chunk(3)):
                mapped[key.replace("in_proj_" + field, name + "_proj." + field)] = part
            continue
        key = re.sub(r"(layers\.\d+)\.(fc[12])\.", r"\1.mlp.\2.", key)
        key = re.sub(r"mask_predictor.mask_embedder\.(\d+)\.0\.", r"mask_predictor.mask_embedder.layers.\1.", key)
        for name in ("queries_embedder", "queries_features", "level_embed"):
            key = key.replace(f"transformer_module.{name}.weight", f"transformer_module.{name}.emb.weight")
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)
