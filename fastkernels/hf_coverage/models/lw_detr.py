"""LW-DETR's default ViT windows, proposal selection and deformable decoder."""

import math

import torch
from torch import nn

from .deformable_detr import _SamplingAttention, _valid_ratios
from .detr import _Attention, _MLP, make_workloads
from ..patches.detector_topk import DetectorTopK
from ..patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sam3_position_encoding import Sam3PositionEncoding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.tensor_ops import Exp
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead as PredictionHead


class _ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.query, self.key, self.value = Linear(width, width), Linear(width, width, bias=False), Linear(width, width)
        self.core = DenseAttention(backend="sdpa")

    def forward(self, hidden):
        shape = lambda value: value.reshape(*hidden.shape[:2], self.heads, -1)
        return self.core(shape(self.query(hidden)), shape(self.key(hidden)), shape(self.value(hidden))).reshape(hidden.shape)


class _ViTLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        width = config.hidden_size
        self.window, self.windows = index in config.window_block_indices, config.num_windows
        self.attention = nn.Module()
        self.attention.attention, self.attention.output = _ViTAttention(config), Linear(width, width)
        self.intermediate = _MLP(width, int(width * config.mlp_ratio))
        self.intermediate.activation = GELU()
        self.layernorm_before, self.layernorm_after = (LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False) for _ in range(2))
        self.gamma_1, self.gamma_2 = nn.Parameter(torch.ones(width)), nn.Parameter(torch.ones(width))
        self.product = ProductGate()

    def forward(self, hidden):
        normalized = self.layernorm_before(hidden)
        if not self.window:
            normalized = normalized.reshape(hidden.shape[0] // self.windows, -1, hidden.shape[-1])
        attention = self.attention.output(self.attention.attention(normalized))
        attention = self.product(torch.cat((attention, self.gamma_1.expand_as(attention)), dim=-1))
        hidden = hidden + attention.reshape(hidden.shape)
        output = self.intermediate(self.layernorm_after(hidden))
        return hidden + self.product(torch.cat((output, self.gamma_2.expand_as(output)), dim=-1))


class _ViTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Module()
        self.embeddings.projection = Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size)
        count = (config.pretrain_image_size // config.patch_size)**2 + 1
        self.embeddings.position_embeddings = nn.Parameter(torch.empty(1, count, config.hidden_size))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_ViTLayer(config, i) for i in range(config.num_hidden_layers)])
        self.interpolate = Interpolate()

    def forward(self, pixels):
        hidden = self.embeddings.projection(pixels)
        batch, channels, height, width = hidden.shape
        positions = self.embeddings.position_embeddings[:, 1:]
        side = math.isqrt(positions.shape[1])
        positions = positions.reshape(1, side, side, channels).permute(0, 3, 1, 2)
        if (side, side) != (height, width):
            positions = self.interpolate(positions, size=(height, width), mode="bicubic", align_corners=False)
        hidden = hidden + positions
        windows = self.config.num_windows_side
        wh, ww = height // windows, width // windows
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, windows, wh, windows, ww, channels).permute(0, 1, 3, 2, 4, 5)
        hidden = hidden.reshape(batch * windows**2, wh * ww, channels)
        features = []
        for index in range(len(self.encoder.layer) + 1):
            if index in self.config.out_indices:
                feature = hidden.reshape(batch, windows, windows, wh, ww, channels).permute(0, 5, 1, 3, 2, 4)
                features.append(feature.reshape(batch, channels, height, width))
            if index < len(self.encoder.layer):
                hidden = self.encoder.layer[index](hidden)
        return features


class _ConvNorm(nn.Module):
    def __init__(self, source, target, kernel, eps):
        super().__init__()
        self.conv, self.norm, self.activation = Conv2d(source, target, kernel, padding=kernel // 2, bias=False), BatchNorm2d(target, eps=eps), SiLU()

    def forward(self, hidden):
        return self.activation(self.norm(self.conv(hidden)))


class _C2F(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden = int(config.d_model * config.hidden_expansion)
        source = config.backbone_config.hidden_size * len(config.backbone_config.out_indices)
        self.conv1 = _ConvNorm(source, 2 * self.hidden, 1, config.batch_norm_eps)
        self.conv2 = _ConvNorm((2 + config.c2f_num_blocks) * self.hidden, config.d_model, 1, config.batch_norm_eps)
        self.bottlenecks = nn.ModuleList()
        for _ in range(config.c2f_num_blocks):
            block = nn.Module()
            block.conv1, block.conv2 = (_ConvNorm(self.hidden, self.hidden, 3, config.batch_norm_eps) for _ in range(2))
            self.bottlenecks.append(block)

    def forward(self, hidden):
        values = list(self.conv1(hidden).split(self.hidden, dim=1))
        for block in self.bottlenecks:
            values.append(block.conv2(block.conv1(values[-1])))
        return self.conv2(torch.cat(values, dim=1))


class _Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = _ViTBackbone(config.backbone_config)
        self.projector = nn.Module()
        scale = nn.Module()
        scale.projector_layer, scale.layer_norm = _C2F(config), LayerNorm(config.d_model, eps=1e-6, promote_fp32=False)
        self.projector.scale_layers = nn.ModuleList([scale])
        self.interpolate = Interpolate()

    def forward(self, pixels, mask):
        features = torch.cat(self.backbone(pixels), dim=1)
        scale = self.projector.scale_layers[0]
        features = scale.projector_layer(features)
        features = scale.layer_norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        mask = self.interpolate(mask[None].float(), size=features.shape[-2:]).bool()[0]
        return features, mask


class _DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.self_attn = _Attention(width, config.decoder_self_attention_heads)
        self.cross_attn = _SamplingAttention(width, config.decoder_cross_attention_heads, config.num_feature_levels, config.decoder_n_points)
        self.self_attn_layer_norm, self.cross_attn_layer_norm, self.layer_norm = (LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(3))
        self.mlp = _MLP(width, config.decoder_ffn_dim)

    def forward(self, hidden, positions, memory, refs, shapes, mask):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, positions))
        hidden = self.cross_attn_layer_norm(hidden + self.cross_attn(hidden, memory, positions, refs, shapes, mask))
        return self.layer_norm(hidden + self.mlp(hidden))


class _Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.ref_point_head = PredictionHead(config, width * 2, width, width, 2)
        self.sine, self.product = Sam3PositionEncoding(width), ProductGate()

    def forward(self, hidden, refs, ratios, memory, shapes, mask):
        sizes = torch.cat((ratios, ratios), dim=-1)[:, None].expand(*refs.shape[:2], ratios.shape[1], 4)
        refs_input = self.product(torch.cat((refs[:, :, None].expand_as(sizes), sizes), dim=-1))
        coordinates = refs_input[:, :, 0]
        x, y = self.sine._encode_xy(coordinates[..., 0].flatten(), coordinates[..., 1].flatten())
        w, h = self.sine._encode_xy(coordinates[..., 2].flatten(), coordinates[..., 3].flatten())
        positions = self.ref_point_head(torch.cat((y, x, w, h), dim=-1).reshape(*refs.shape[:2], -1).to(refs.dtype))
        states = []
        for layer in self.layers:
            hidden = layer(hidden, positions, memory, refs_input, shapes, mask)
            states.append(self.layernorm(hidden))
        return torch.stack(states)


class _Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.d_model
        self.backbone, self.decoder = _Backbone(config), _Decoder(config)
        self.reference_point_embed, self.query_feat = Embedding(config.num_queries * config.group_detr, 4), Embedding(config.num_queries * config.group_detr, width)
        self.enc_output = nn.ModuleList([Linear(width, width) for _ in range(config.group_detr)])
        self.enc_output_norm = nn.ModuleList([LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(config.group_detr)])
        self.enc_out_bbox_embed = nn.ModuleList([PredictionHead(config, width, width, 4, 3) for _ in range(config.group_detr)])
        self.enc_out_class_embed = nn.ModuleList([Linear(width, len(config.id2label)) for _ in range(config.group_detr)])
        self.product, self.exp, self.reduce, self.select = ProductGate(), Exp(), SegmentCSR(), DetectorTopK()

    def refine(self, refs, delta):
        center = self.product(torch.cat((delta[..., :2], refs[..., 2:]), dim=-1)) + refs[..., :2]
        size = self.product(torch.cat((self.exp(delta[..., 2:]), refs[..., 2:]), dim=-1))
        return torch.cat((center, size), dim=-1)

    def forward(self, pixels, pixel_mask=None):
        batch, _, height, width = pixels.shape
        if pixel_mask is None:
            pixel_mask = torch.ones(batch, height, width, dtype=torch.long, device=pixels.device)
        features, mask = self.backbone(pixels, pixel_mask)
        height, width = features.shape[-2:]
        memory = features.flatten(2).transpose(1, 2)
        valid_height, valid_width = mask[:, :, 0].sum(1), mask[:, 0, :].sum(1)
        y, x = torch.meshgrid(torch.linspace(0, height - 1, height, device=pixels.device, dtype=pixels.dtype),
                              torch.linspace(0, width - 1, width, device=pixels.device, dtype=pixels.dtype), indexing="ij")
        grid = (torch.stack((x, y), dim=-1)[None] + .5) / torch.stack((valid_width, valid_height), dim=-1)[:, None, None]
        proposals = torch.cat((grid, torch.ones_like(grid) * .05), dim=-1).flatten(1, 2)
        invalid = ~mask.flatten(1)[..., None] | ~((proposals > .01) & (proposals < .99)).all(-1, keepdim=True)
        proposals = proposals.masked_fill(invalid, 0)
        objects = self.enc_output_norm[0](self.enc_output[0](memory.masked_fill(invalid, 0)))
        classes = self.enc_out_class_embed[0](objects).masked_fill(invalid, -float("inf"))
        boxes = self.refine(proposals, self.enc_out_bbox_embed[0](objects))
        offsets = torch.arange(classes.numel() // classes.shape[-1] + 1, device=pixels.device) * classes.shape[-1]
        maxima = self.reduce(classes.flatten(), offsets, reduce="max").reshape(batch, -1)
        _, indices = self.select(maxima, self.config.num_queries)
        boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        objects = objects.gather(1, indices[..., None].expand(-1, -1, self.config.d_model))
        refs = self.refine(boxes, self.reference_point_embed.emb.weight[:self.config.num_queries][None].expand(batch, -1, -1))
        target = self.query_feat.emb.weight[:self.config.num_queries][None].expand(batch, -1, -1)
        states = self.decoder(target, refs, _valid_ratios(mask, pixels.dtype)[:, None], memory, [(height, width)], mask.flatten(1))
        return states, refs, objects, boxes


class _Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _Model(config)
        self.class_embed, self.bbox_embed = Linear(config.d_model, len(config.id2label)), PredictionHead(config, config.d_model, config.d_model, 4, 3)

    def forward(self, pixel_values, pixel_mask=None):
        states, refs, objects, boxes = self.model(pixel_values, pixel_mask)
        return {"logits": self.class_embed(states[-1]), "pred_boxes": self.model.refine(refs, self.bbox_embed(states[-1])),
                "last_hidden_state": states[-1], "intermediate_hidden_states": states,
                "intermediate_reference_points": refs[None], "init_reference_points": refs,
                "enc_outputs_class": self.model.enc_out_class_embed[0](objects), "enc_outputs_coord_logits": boxes}


def build_from_config(config, device, dtype):
    if (config.projector_scale_factors != [1.0] or config.activation_function != "silu"
            or config.decoder_activation_function != "relu" or config.backbone_config.hidden_act != "gelu"):
        raise ValueError("This case preserves the published single-scale LW-DETR default")
    return _Detector(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for key, value in state_dict.items():
        key = key.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
        for name in ("reference_point_embed", "query_feat"):
            key = key.replace(name + ".weight", name + ".emb.weight")
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)
