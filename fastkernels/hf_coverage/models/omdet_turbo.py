"""Default OmDet-Turbo detection from Swin, CLIP and RT-DETR operations."""

import math
from collections import OrderedDict
from functools import lru_cache

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tensor_ops import Pad, Cat
from fastkernels.tasks.baseline.L2.rtdetrv2_deformable_attention import RTDetrV2MultiscaleDeformableAttention
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid
from fastkernels.tasks.baseline.L3.rtdetrv2_hybrid_encoder import RTDetrV2HybridEncoder
from fastkernels.tasks.baseline.L3.swinv2_block import window_partition as kb_window_partition, window_reverse as kb_window_reverse
from fastkernels.tasks.baseline.L4.clip_text_model import CLIPTextModel
from ..patches.detector_topk import DetectorTopK
from ..runner import Workload, config_values
from .clip import configure_encoder


def _window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    return kb_window_partition(x, (window, window))


def _window_reverse(windows: torch.Tensor, window: int, height: int, width: int) -> torch.Tensor:
    return kb_window_reverse(windows, (window, window), (height, width))


def _relative_position_index(window: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
    flat = coords.flatten(1)
    relative = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window - 1
    relative[:, :, 1] += window - 1
    relative[:, :, 0] *= 2 * window - 1
    return relative.sum(-1)


class TimmSwinPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Conv2d(3, 96, kernel_size=4, stride=4, bias=True)
        self.norm = LayerNorm(96, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class TimmSwinPatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm(4 * dim, eps=1e-5)
        self.reduction = Linear(4 * dim, out_dim, bias=False)
        self.pad = Pad()
        self.cat = Cat(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, _ = x.shape
        if height % 2 or width % 2:
            x = self.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        f0 = x[:, 0::2, 0::2, :]
        f1 = x[:, 1::2, 0::2, :]
        f2 = x[:, 0::2, 1::2, :]
        f3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(self.cat((f0, f1, f2, f3))))


class TimmSwinWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: int = 7):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Swin dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.qkv = Linear(dim, 3 * dim, bias=True)
        self.proj = Linear(dim, dim, bias=True)
        self.matmul = BMM()
        self.softmax = Softmax()
        self.relative_position_bias_table = nn.Parameter(torch.empty((2 * window - 1) ** 2, num_heads))
        self.register_buffer("relative_position_index", _relative_position_index(window), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch_windows, tokens, _ = x.shape
        qkv = self.qkv(x).view(batch_windows, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)
        # timm's default Swin path scales Q before the score matmul.
        scores = self.matmul(q.transpose(1, 2) * (self.head_dim ** -0.5), k.transpose(1, 2).transpose(-1, -2)) + bias
        if mask is not None:
            windows = mask.shape[0]
            scores = (scores.view(-1, windows, self.num_heads, tokens, tokens)
                      + mask[None, :, None]).view(batch_windows, self.num_heads, tokens, tokens)
        out = self.matmul(self.softmax(scores), v.transpose(1, 2)).transpose(1, 2)
        return self.proj(out.reshape(batch_windows, tokens, self.dim))



class TimmSwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, shift: int):
        super().__init__()
        self.window = 7
        self.shift = shift
        self.norm1 = LayerNorm(dim, eps=1e-5)
        self.attn = TimmSwinWindowAttention(dim, num_heads, self.window)
        self.norm2 = LayerNorm(dim, eps=1e-5)
        self.mlp = nn.Module()
        self.mlp.fc1 = Linear(dim, 4 * dim, bias=True)
        self.mlp.act = GELU(approximate="none")
        self.mlp.fc2 = Linear(4 * dim, dim, bias=True)
        self.pad = Pad()

    def _mask(self, height: int, width: int, dtype, device) -> torch.Tensor | None:
        if not self.shift:
            return None
        padded_h = math.ceil(height / self.window) * self.window
        padded_w = math.ceil(width / self.window) * self.window
        image_mask = torch.zeros((1, padded_h, padded_w, 1), dtype=dtype, device=device)
        count = 0
        h_slices = ((0, -self.window), (-self.window, -self.shift), (-self.shift, None))
        w_slices = ((0, -self.window), (-self.window, -self.shift), (-self.shift, None))
        for h_start, h_end in h_slices:
            for w_start, w_end in w_slices:
                image_mask[:, h_start:h_end, w_start:w_end, :] = count
                count += 1
        windows = _window_partition(image_mask, self.window).view(-1, self.window * self.window)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, channels = x.shape
        shifted = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2)) if self.shift else x
        pad_h = (self.window - height % self.window) % self.window
        pad_w = (self.window - width % self.window) % self.window
        if pad_h or pad_w:
            shifted = self.pad(shifted, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = shifted.shape[1:3]
        windows = _window_partition(shifted, self.window).view(-1, self.window * self.window, channels)
        windows = self.attn(windows, self._mask(height, width, x.dtype, x.device))
        shifted = _window_reverse(
            windows.view(-1, self.window, self.window, channels), self.window, padded_h, padded_w
        )[:, :height, :width, :].contiguous()
        return torch.roll(shifted, shifts=(self.shift, self.shift), dims=(1, 2)) if self.shift else shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        x = x + self._window_attention(self.norm1(x))
        flat = x.reshape(batch, -1, channels)
        flat = flat + self.mlp.fc2(self.mlp.act(self.mlp.fc1(self.norm2(flat))))
        return flat.view(batch, height, width, channels)


class TimmSwinStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int, num_heads: int, downsample: bool):
        super().__init__()
        self.downsample = TimmSwinPatchMerging(in_dim, out_dim) if downsample else nn.Identity()
        self.blocks = nn.ModuleList(
            [TimmSwinBlock(out_dim, num_heads, shift=0 if i % 2 == 0 else 3) for i in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class TimmSwinTinyBackbone(nn.Module):
    """Literal timm Swin-Tiny graph with OmDet's fixed feature indices 1/2/3."""

    def __init__(self):
        super().__init__()
        self.patch_embed = TimmSwinPatchEmbed()
        dims = (96, 192, 384, 768)
        depths = (2, 2, 6, 2)
        heads = (3, 6, 12, 24)
        self.stages = nn.ModuleList(
            [
                TimmSwinStage(dims[max(i - 1, 0)], dims[i], depths[i], heads[i], downsample=i > 0)
                for i in range(4)
            ]
        )

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(pixel_values)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (1, 2, 3):
                # Timm's feature extractor exposes NHWC Swin stage outputs;
                # OmDet's wrapper LayerNorm consumes that layout.
                outputs.append(x)
        return outputs


class OmDetTurboVisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.apply_layernorm_after_vision_backbone = bool(config.apply_layernorm_after_vision_backbone)
        self.vision_backbone = TimmSwinTinyBackbone()
        self.layer_norms = nn.ModuleList(
            [LayerNorm(int(channels), eps=float(config.layer_norm_eps)) for channels in config.encoder_in_channels]
        )

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.vision_backbone(pixel_values)
        if self.apply_layernorm_after_vision_backbone:
            outputs = [
                norm(output).permute(0, 3, 1, 2).contiguous()
                for norm, output in zip(self.layer_norms, outputs)
            ]
        return outputs


class LanguageBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CLIPTextModel(config.text_config)
        configure_encoder(self.model.text_model.encoder, config.text_config)
        self.text_projection = nn.Parameter(torch.empty(config.text_projection_in_dim, config.text_projection_out_dim))
        self.matmul = BMM()

    def forward(self, ids, mask=None):
        outputs = self.model(ids)
        if mask is None:
            return self.matmul(outputs.pooler_output, self.text_projection)
        length = int(mask.ne(0).sum(1).max())
        return outputs.last_hidden_state[:, :length].transpose(0, 1), mask[:, :length]


class HybridEncoder(RTDetrV2HybridEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.channel_projection_layers = nn.ModuleList([
            nn.Sequential(Conv2d(channels, config.encoder_hidden_dim, 1, bias=False),
                          BatchNorm2d(config.encoder_hidden_dim))
            for channels in config.encoder_in_channels
        ])
        for module in (*self.lateral_convs, *self.downsample_convs):
            module.activation = GELU()
        for encoder in self.encoder:
            for layer in encoder.layers:
                layer.self_attn = EncoderAttention(config.encoder_hidden_dim, config.encoder_attention_heads)

    def forward(self, features):
        features = [projection(value) for projection, value in zip(self.channel_projection_layers, features)]
        return super().forward(features, return_dict=True).last_hidden_state


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query, self.key, self.value = [Linear(width, width) for _ in range(3)]
        self.out_proj = Linear(width, width)
        self.matmul, self.softmax = BMM(), Softmax()

    def forward(self, query, key, value, mask):
        batch, length, width = query.shape
        def split(x):
            return x.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(key)), split(self.value(value))
        scores = self.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        output = self.matmul(self.softmax(scores), v).transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output)


class EncoderAttention(Attention):
    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, output_attentions=False):
        positioned = hidden_states if position_embeddings is None else hidden_states + position_embeddings
        return super().forward(positioned, positioned, hidden_states, attention_mask), None


class TaskEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.linear1 = Linear(config.class_embed_dim, config.task_encoder_hidden_dim)
        self.mlp.linear2 = Linear(config.task_encoder_hidden_dim, config.class_embed_dim)
        self.relu = ReLU()
        self.res1 = nn.Module()
        self.res1.norm1 = LayerNorm(config.class_embed_dim, eps=config.layer_norm_eps)

    def forward(self, x):
        return self.res1.norm1(x + self.mlp.linear2(self.relu(self.mlp.linear1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.decoder_hidden_dim
        self.self_attn = Attention(width, config.decoder_num_heads)
        self.cross_attn = RTDetrV2MultiscaleDeformableAttention(config)
        self.norm1, self.norm2, self.norm3 = [LayerNorm(width, eps=config.layer_norm_eps) for _ in range(3)]
        self.linear1 = Linear(width, config.decoder_dim_feedforward)
        self.linear2 = Linear(config.decoder_dim_feedforward, width)
        self.relu = ReLU()

    def forward(self, hidden, task, position, reference, vision, shapes, starts, mask):
        length = hidden.shape[1]
        task = task.transpose(0, 1)
        query = torch.cat((hidden + position, task), 1)
        combined = torch.cat((hidden, task), 1)
        combined = self.norm1(combined + self.self_attn(query, query, combined, mask))
        hidden, task = combined[:, :length], combined[:, length:].transpose(0, 1)
        attended, _ = self.cross_attn(
            hidden_states=hidden + position, encoder_hidden_states=vision,
            reference_points=reference.unsqueeze(2), spatial_shapes=shapes,
            spatial_shapes_list=shapes.tolist(), level_start_index=starts,
        )
        hidden = self.norm2(hidden + attended)
        hidden = self.norm3(hidden + self.linear2(self.relu(self.linear1(hidden))))
        return hidden, task


class ClassSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize_features, self.normalize_classes = L2Norm(dim=2), L2Norm(dim=1)
        self.matmul = BMM()
        # Retain HF's CPU FP32 scalar: a Python float changes BF16 multiplication rounding.
        self.scale = torch.tensor(1 / 0.07, dtype=torch.float32, device="cpu").log().exp()

    def forward(self, features, classes):
        return self.matmul(self.normalize_features(features), self.normalize_classes(classes)) * self.scale


class Decoder(nn.Module):
    @lru_cache(maxsize=32)
    def generate_anchors(self, shapes, device, dtype):
        # Shape-only metadata. HF exposes invalid anchors as +inf in its public output.
        anchors = []
        for level, (height, width) in enumerate(shapes):
            y, x = torch.meshgrid(torch.arange(height, dtype=dtype, device=device),
                                  torch.arange(width, dtype=dtype, device=device), indexing="ij")
            xy = (torch.stack((x, y), -1)[None] + 0.5) / torch.tensor((width, height), dtype=dtype, device=device)
            wh = torch.ones_like(xy) * 0.05 * (2.0 ** level)
            anchors.append(torch.cat((xy, wh), -1).reshape(1, -1, 4))
        anchors = torch.cat(anchors, 1)
        valid = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        return anchors.masked_fill(~valid, torch.inf), valid

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.decoder_hidden_dim
        self.channel_projection_layers = nn.ModuleList([
            nn.Sequential(Conv2d(channels, width, 1, bias=False), BatchNorm2d(width))
            for channels in config.vision_features_channels
        ])
        self.task_encoder = TaskEncoder(config)
        self.task_project = Linear(config.class_embed_dim, width)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_num_layers)])
        self.query_position_head = RTDetrV2MLPPredictionHead(None, 4, 2 * width, width, 2)
        self.encoder_vision_features = nn.Sequential(Linear(width, width), LayerNorm(width, eps=config.layer_norm_eps))
        self.encoder_class_head = Linear(config.class_embed_dim, width)
        self.encoder_bbox_head = RTDetrV2MLPPredictionHead(None, width, width, 4, 3)
        self.decoder_class_head = nn.ModuleList([Linear(config.class_embed_dim, width) for _ in self.layers])
        self.decoder_bbox_head = nn.ModuleList([RTDetrV2MLPPredictionHead(None, width, width, 4, 3) for _ in self.layers])
        self.similarity, self.sigmoid = ClassSimilarity(), Sigmoid()
        self.class_max, self.topk = SegmentCSR(), DetectorTopK()

    def forward(self, features, classes, task, task_mask):
        features = [projection(x) for projection, x in zip(self.channel_projection_layers, features)]
        shape_list = [tuple(x.shape[-2:]) for x in features]
        vision = torch.cat([x.flatten(2).transpose(1, 2) for x in features], 1)
        batch = vision.shape[0]
        shapes = torch.tensor(shape_list, device=vision.device, dtype=torch.long)
        starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
        anchors, valid = self.generate_anchors(tuple(shape_list), device=vision.device, dtype=vision.dtype)
        memory = self.encoder_vision_features(vision.masked_fill(~valid, 0))
        logits = self.similarity(memory, self.encoder_class_head(classes).permute(1, 2, 0))
        boxes = self.encoder_bbox_head(memory) + anchors
        flat = logits.reshape(-1)
        offsets = torch.arange(0, flat.numel() + 1, logits.shape[-1], device=flat.device)
        scores = self.class_max(flat, offsets, reduce="max").view(batch, -1)
        _, indices = self.topk(scores, self.config.num_queries)
        rows = torch.arange(batch, device=vision.device)[:, None]
        reference = self.sigmoid(boxes[rows, indices])
        encoder_boxes, encoder_logits = reference, logits[rows, indices]
        hidden = memory[rows, indices]
        task = self.task_project(self.task_encoder(task))
        padding = torch.cat((task_mask.new_ones((batch, self.config.num_queries)), task_mask), 1).eq(0)
        mask = vision.new_zeros((batch, 1, padding.shape[1], padding.shape[1]))
        mask = mask.masked_fill(padding[:, None, None], torch.finfo(vision.dtype).min)
        for index, layer in enumerate(self.layers):
            hidden, task = layer(hidden, task, self.query_position_head(reference), reference,
                                 vision, shapes, starts, mask)
            refined = self.sigmoid(self.decoder_bbox_head[index](hidden) + inverse_sigmoid(reference))
            projected = self.decoder_class_head[index](classes).permute(1, 2, 0)
            if index == len(self.layers) - 1:
                break
            reference = refined
        return dict(decoder_coord_logits=refined, decoder_class_logits=self.similarity(hidden, projected),
                    init_reference_points=anchors, intermediate_reference_points=reference,
                    encoder_coord_logits=encoder_boxes, encoder_class_logits=encoder_logits)


class OmDetTurbo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_backbone = OmDetTurboVisionBackbone(config)
        self.language_backbone = LanguageBackbone(config)
        self.encoder, self.decoder = HybridEncoder(config), Decoder(config)
        self.class_cache, self.task_cache = OrderedDict(), OrderedDict()
        self.pad = Pad()

    def cached_language(self, ids, mask, task=False):
        cache = self.task_cache if task else self.class_cache
        keys = [tuple(row[valid.ne(0)].tolist()) for row, valid in zip(ids, mask)]
        values, missing = [], []
        for index, key in enumerate(keys):
            value = cache.get(key)
            if value is None:
                missing.append(index)
            else:
                cache.move_to_end(key)
            values.append(value)
        if missing:
            encoded = self.language_backbone(ids[missing], mask[missing] if task else None)
            for local, index in enumerate(missing):
                value = (encoded[0][:, local:local + 1], encoded[1][local:local + 1]) if task else encoded[local]
                values[index] = value
                cache[keys[index]] = value
                cache.move_to_end(keys[index])
                if len(cache) > self.config.cache_size:
                    cache.popitem(last=False)
        if not task:
            return torch.stack(values)
        length = max(value[0].shape[0] for value in values)
        return (torch.cat([self.pad(value, (0, 0, 0, 0, 0, length - value.shape[0])) for value, _ in values], 1),
                torch.cat([self.pad(valid, (0, length - valid.shape[1])) for _, valid in values], 0))

    def forward(self, pixel_values, classes_input_ids, classes_attention_mask,
                tasks_input_ids, tasks_attention_mask, classes_structure):
        features = self.encoder(self.vision_backbone(pixel_values))
        classes = self.cached_language(classes_input_ids, classes_attention_mask)
        sizes = classes_structure.tolist()
        grouped, start = [], 0
        for size in sizes:
            grouped.append(self.pad(classes[start:start + size], (0, 0, 0, max(sizes) - size))[:, None])
            start += size
        classes = torch.cat(grouped, 1)
        task, mask = self.cached_language(tasks_input_ids, tasks_attention_mask, task=True)
        outputs = self.decoder(features, classes, task, mask)
        outputs.update(encoder_extracted_states=tuple(features), classes_structure=classes_structure)
        return outputs


def build_from_config(config, device, dtype):
    config = config_values(config.to_dict())
    if (config.backbone_config.backbone != "swin_tiny_patch4_window7_224"
            or config.class_distance_type != "cosine" or config.learn_initial_query
            or not config.apply_layernorm_after_vision_backbone
            or config.class_embed_dim == config.decoder_hidden_dim):
        raise ValueError("OmDet construction requires the checkpoint's default Swin/CLIP detector path")
    config.encoder_hidden_dim = config.d_model
    config.activation_function = config.csp_activation
    config.encode_proj_layers = config.encoder_projection_indices
    config.feat_strides = [8, 16, 32]
    config.decoder_attention_heads = config.decoder_num_heads
    config.decoder_n_points = config.decoder_num_points
    config.decoder_n_levels = config.num_feature_levels
    config.decoder_offset_scale, config.decoder_method = 0.5, "default"
    config.normalize_before = False
    config.dropout, config.activation_dropout = config.encoder_dropout, config.encoder_feedforward_dropout
    config.encoder_activation_function = config.encoder_feedforward_activation
    config.encoder_ffn_dim = config.encoder_dim_feedforward
    model = OmDetTurbo(config)
    for module in model.modules():
        if isinstance(module, LayerNorm):
            module.promote_fp32 = False
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, current in model.state_dict().items():
        if name.endswith(".n_points_scale"):
            mapped[name] = current
            continue
        source = name
        if source.startswith("vision_backbone.vision_backbone."):
            source = source.replace("vision_backbone.vision_backbone.", "vision_backbone.vision_backbone._backbone.")
            for index in range(4):
                source = source.replace(f".stages.{index}.", f".layers_{index}.")
        elif source.startswith("language_backbone.model."):
            source = source.replace("language_backbone.model.text_model.", "language_backbone.model.")
            source = source.replace(".emb.weight", ".weight")
            source = source.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
            source = source.replace(".mlp_fc1.", ".mlp.fc1.").replace(".mlp_fc2.", ".mlp.fc2.")
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                source = source.replace(f".{projection}.", f".self_attn.{projection}.")
        elif source.startswith("encoder."):
            for old, new in (("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")):
                source = source.replace(f".self_attn.{old}.", f".self_attn.{new}.")
        mapped[name] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f"Unmapped OmDet state: {sorted(set(state_dict) - used)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    def forward():
        outputs = model(**inputs)
        features = outputs.pop("encoder_extracted_states")
        outputs.update({f"encoder_extracted_states.{index}": value for index, value in enumerate(features)})
        return outputs
    return {"forward": Workload(run=forward)}
