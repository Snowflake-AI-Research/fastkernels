"""Grounding DINO's text/vision fusion and iterative proposal refinement."""

import math
import re

import torch
from torch import nn

from .deformable_detr import _EncoderLayer, _SamplingAttention, _encoder_references, _valid_ratios
from .detr import _MLP, _positions
from .mask2former import _SwinBackbone, _load_swin_backbone
from ..runner import Workload
from ..patches.detector_topk import DetectorTopK
from ..patches.product_gate import ProductGate
from ..patches.grounding_logit import GroundingLogit
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.sam3_position_encoding import Sam3PositionEncoding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead as PredictionHead
from fastkernels.tasks.baseline.L3.bert_model import BertModel


def _text_metadata(ids):
    """Punctuation-delimited attention blocks use token IDs, not activations."""
    batch, length = ids.shape
    special = torch.isin(ids, torch.tensor([101, 102, 1012, 1029], device=ids.device))
    indices = torch.arange(length, device=ids.device)[None].expand(batch, -1)
    previous = torch.where(special, indices, -1).cummax(1).values
    following = torch.where(special, indices, length).flip([1]).cummin(1).values.flip([1])
    valid = (following != 0) & (following != length - 1) & (following != length)
    mask = (following[:, :, None] == following[:, None, :]) & valid[:, None]
    mask = mask | torch.eye(length, device=ids.device, dtype=torch.bool)[None]
    positions = torch.where(valid, indices - previous - 1, 0).clamp(min=0)
    return mask, positions


class _Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query, self.key, self.value, self.out_proj = (Linear(width, width) for _ in range(4))
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, queries, keys, values, mask=None):
        shape = lambda hidden: hidden.reshape(hidden.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)
        query, key, value = shape(self.query(queries)), shape(self.key(keys)), shape(self.value(values))
        scores = self.bmm(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        output = self.bmm(self.softmax(scores), value).transpose(1, 2).reshape(queries.shape)
        return self.out_proj(output)


class _BidirectionalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.encoder_attention_heads // 2
        self.width = config.encoder_ffn_dim // 2
        self.head_dim = self.width // self.heads
        for name in ("vision_proj", "text_proj", "values_vision_proj", "values_text_proj"):
            self.add_module(name, Linear(config.d_model, self.width))
        self.out_vision_proj, self.out_text_proj = Linear(self.width, config.d_model), Linear(self.width, config.d_model)
        self.bmm, self.softmax, self.reduce = BMM(), Softmax(dim=-1), SegmentCSR()

    def _clamp(self, scores):
        # Pairwise min/max preserves the native saturation without cancellation.
        offsets = torch.arange(scores.numel() + 1, device=scores.device) * 2
        pairs = torch.stack((scores, torch.full_like(scores, -50000)), dim=-1)
        scores = self.reduce(pairs.flatten(), offsets, reduce="max").reshape(scores.shape)
        pairs = torch.stack((scores, torch.full_like(scores, 50000)), dim=-1)
        return self.reduce(pairs.flatten(), offsets, reduce="min").reshape(scores.shape)

    def forward(self, vision, text, vision_mask, text_mask):
        batch = vision.shape[0]
        shape = lambda hidden: hidden.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2).reshape(batch * self.heads, -1, self.head_dim)
        query = shape(self.vision_proj(vision) * self.head_dim**-.5)
        key = shape(self.text_proj(text))
        vision_value, text_value = shape(self.values_vision_proj(vision)), shape(self.values_text_proj(text))
        scores = self.bmm(query, key.transpose(1, 2))
        maximum = self.reduce(scores.flatten(), torch.tensor([0, scores.numel()], device=scores.device), reduce="max")
        scores = self._clamp(scores - maximum)
        transposed = scores.transpose(1, 2).contiguous()
        offsets = torch.arange(transposed.numel() // transposed.shape[-1] + 1, device=scores.device) * transposed.shape[-1]
        row_max = self.reduce(transposed.flatten(), offsets, reduce="max").reshape(*transposed.shape[:-1], 1)
        transposed = self._clamp(transposed - row_max)
        transposed = transposed.masked_fill(vision_mask[:, None, None].expand(batch, self.heads, 1, -1).flatten(0, 1), -float("inf"))
        scores = scores.masked_fill(text_mask[:, None, None].expand(batch, self.heads, 1, -1).flatten(0, 1), -float("inf"))
        v = self.bmm(self.softmax(scores), text_value).reshape(batch, self.heads, -1, self.head_dim).transpose(1, 2).reshape(batch, -1, self.width)
        t = self.bmm(self.softmax(transposed), vision_value).reshape(batch, self.heads, -1, self.head_dim).transpose(1, 2).reshape(batch, -1, self.width)
        return self.out_vision_proj(v), self.out_text_proj(t)


class _Fusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.layer_norm_vision, self.layer_norm_text = (LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False) for _ in range(2))
        self.attn = _BidirectionalAttention(config)
        self.vision_param, self.text_param = nn.Parameter(torch.empty(width)), nn.Parameter(torch.empty(width))
        self.product = ProductGate()

    def forward(self, vision, text, vision_mask, text_mask):
        # The native residual is the normalized input, not the pre-normalized one.
        vision, text = self.layer_norm_vision(vision), self.layer_norm_text(text)
        v, t = self.attn(vision, text, vision_mask, text_mask)
        return (vision + self.product(torch.cat((v, self.vision_param.expand_as(v)), dim=-1)),
                text + self.product(torch.cat((t, self.text_param.expand_as(t)), dim=-1)))


class _TextEnhancer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.self_attn = _Attention(width, config.encoder_attention_heads // 2)
        self.layer_norm_before, self.layer_norm_after = (LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False) for _ in range(2))
        self.mlp = _MLP(width, config.encoder_ffn_dim // 2)

    def forward(self, text, positions, mask):
        text = self.layer_norm_before(text + self.self_attn(text + positions, text + positions, text, mask))
        return self.layer_norm_after(text + self.mlp(text))


class _EncoderLayerWithText(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fusion_layer, self.text_enhancer_layer, self.deformable_layer = _Fusion(config), _TextEnhancer(config), _EncoderLayer(config)

    def forward(self, vision, text, positions, text_positions, refs, shapes, mask, text_mask, self_mask):
        vision, text = self.fusion_layer(vision, text, ~mask, ~text_mask)
        text = self.text_enhancer_layer(text, text_positions, self_mask)
        return self.deformable_layer(vision, positions, refs, shapes, mask), text


class _DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.self_attn, self.encoder_attn_text = (_Attention(width, config.decoder_attention_heads) for _ in range(2))
        self.encoder_attn = _SamplingAttention(width, config.decoder_attention_heads, config.num_feature_levels, config.decoder_n_points)
        for name in ("self_attn_layer_norm", "encoder_attn_text_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"):
            self.add_module(name, LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False))
        self.mlp = _MLP(width, config.decoder_ffn_dim)

    def forward(self, hidden, positions, refs, vision, text, shapes, mask, text_mask):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden + positions, hidden + positions, hidden))
        hidden = self.encoder_attn_text_layer_norm(hidden + self.encoder_attn_text(hidden + positions, text, text, text_mask))
        hidden = self.encoder_attn_layer_norm(hidden + self.encoder_attn(hidden, vision, positions, refs, shapes, mask))
        return self.final_layer_norm(hidden + self.mlp(hidden))


class _Contrastive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_text_len, self.mm = config.max_text_len, config.model_type == "mm-grounding-dino"
        if self.mm:
            self.bias = nn.Parameter(torch.zeros(()))
        self.bmm = BMM()

    def forward(self, vision, text, mask):
        scores = self.bmm(vision, text.transpose(-1, -2))
        if self.mm:
            scores = scores / math.sqrt(vision.shape[-1]) + self.bias
        scores = scores.masked_fill(~mask[:, None], -float("inf"))
        # HF intentionally publishes FP32 padding/output, including BF16 scores.
        output = torch.full((*scores.shape[:-1], self.max_text_len), -float("inf"), device=scores.device)
        output[..., :scores.shape[-1]] = scores
        return output


class _Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.reference_points_head = PredictionHead(config, width * 2, width, width, 2)
        self.sine, self.product, self.sigmoid = Sam3PositionEncoding(width), ProductGate(), Sigmoid()
        self.logit = GroundingLogit()

    def forward(self, hidden, refs, ratios, vision, text, shapes, mask, text_mask):
        states, references = [], []
        ratio = torch.cat((ratios, ratios), dim=-1)[:, None]
        for index, layer in enumerate(self.layers):
            left, right = torch.broadcast_tensors(refs[:, :, None], ratio)
            points = self.product(torch.cat((left, right), dim=-1))
            coordinates = points[:, :, 0]
            x, y = self.sine._encode_xy(coordinates[..., 0].flatten(), coordinates[..., 1].flatten())
            w, h = self.sine._encode_xy(coordinates[..., 2].flatten(), coordinates[..., 3].flatten())
            sine = torch.cat((y, x, w, h), dim=-1).reshape(*refs.shape[:2], -1)
            positions = self.reference_points_head(sine)
            hidden = layer(hidden, positions, points, vision, text, shapes, mask, text_mask)
            refs = self.sigmoid(self.bbox_embed[index](hidden) + self.logit(refs))
            states.append(self.layer_norm(hidden)); references.append(refs)
        return self.layer_norm(hidden), torch.stack(states, dim=1), torch.stack(references, dim=1)


class _Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.d_model
        self.backbone = nn.Module(); self.backbone.conv_encoder = nn.Module()
        self.backbone.conv_encoder.model = _SwinBackbone(config.backbone_config)
        # Only returned Swin stages have trained normalization parameters.
        self.backbone.conv_encoder.model.hidden_states_norms.stage1 = nn.Identity()
        channels = [config.backbone_config.embed_dim * 2**i for i in (1, 2, 3)]
        self.input_proj_vision = nn.ModuleList([nn.Sequential(Conv2d(channel, width, 1), GroupNorm(32, width, eps=1e-5)) for channel in channels])
        self.input_proj_vision.append(nn.Sequential(Conv2d(channels[-1], width, 3, stride=2, padding=1), GroupNorm(32, width, eps=1e-5)))
        self.text_backbone, self.text_projection = BertModel(config.text_config), Linear(config.text_config.hidden_size, width)
        self.query_position_embeddings = Embedding(config.num_queries, width)
        self.encoder = nn.Module(); self.encoder.layers = nn.ModuleList([_EncoderLayerWithText(config) for _ in range(config.encoder_layers)])
        self.decoder = _Decoder(config)
        self.level_embed = nn.Parameter(torch.empty(config.num_feature_levels, width))
        self.enc_output, self.enc_output_norm = Linear(width, width), LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.encoder_output_bbox_embed = PredictionHead(config, width, width, 4, 3)
        self.encoder_output_class_embed = _Contrastive(config)
        self.interpolate, self.select, self.reduce, self.sigmoid = Interpolate(), DetectorTopK(), SegmentCSR(), Sigmoid()

    def forward(self, pixels, ids, token_types=None, attention_mask=None, pixel_mask=None):
        batch, _, height, width = pixels.shape
        self_mask, position_ids = _text_metadata(ids)
        text_mask = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        token_types = torch.zeros_like(ids) if token_types is None else token_types
        embedded = self.text_backbone.embeddings.forward_with_token_type_ids(input_ids=ids, position_ids=position_ids, token_type_ids=token_types)
        text = self.text_projection(self.text_backbone.encoder.forward_with_attention_mask(embedded, self_mask[:, None]))
        original_text = text
        if pixel_mask is None:
            pixel_mask = torch.ones(batch, height, width, device=pixels.device, dtype=torch.long)
        features = self.backbone.conv_encoder.model(pixels)[1:]
        sources, masks, positions, shapes = [], [], [], []
        for level in range(4):
            source = features[level] if level < 3 else features[-1]
            source = self.input_proj_vision[level](source)
            mask = self.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).bool()[0]
            position = _positions(mask, self.config.d_model, torch.float32,
                                  (self.config.positional_embedding_temperature,) * 2).to(source.dtype)
            sources.append(source.flatten(2).transpose(1, 2)); masks.append(mask.flatten(1))
            positions.append(position + self.level_embed[level]); shapes.append(source.shape[-2:])
        vision, mask, positions = torch.cat(sources, 1), torch.cat(masks, 1), torch.cat(positions, 1)
        mask_maps = [part.reshape(batch, *shape) for part, shape in zip(masks, shapes)]
        ratios = torch.stack([_valid_ratios(part, torch.float32) for part in mask_maps], dim=1)
        refs = _encoder_references(shapes, ratios)
        # Text positions are determined entirely by punctuation/token metadata.
        dim = torch.arange(self.config.d_model, device=ids.device, dtype=torch.float32)
        frequency = 10000 ** (2 * torch.div(dim, 2, rounding_mode="floor") / self.config.d_model)
        phase = position_ids[..., None] * (2 * math.pi) / frequency
        text_positions = torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=-1).flatten(2)
        additive_self_mask = torch.zeros_like(self_mask[:, None], dtype=text.dtype).masked_fill(~self_mask[:, None], torch.finfo(text.dtype).min)
        for layer in self.encoder.layers:
            vision, text = layer(vision, text, positions, text_positions, refs, shapes, mask, text_mask, additive_self_mask)
        proposals = []
        for level, ((height, width), valid) in enumerate(zip(shapes, mask_maps)):
            y, x = torch.meshgrid(torch.linspace(0, height - 1, height, device=pixels.device, dtype=torch.float32),
                                  torch.linspace(0, width - 1, width, device=pixels.device, dtype=torch.float32), indexing="ij")
            scale = torch.stack((valid[:, 0].sum(1), valid[:, :, 0].sum(1)), dim=-1)[:, None, None]
            grid = (torch.stack((x, y), dim=-1)[None] + .5) / scale
            proposals.append(torch.cat((grid, torch.ones_like(grid) * .05 * 2**level), dim=-1).flatten(1, 2))
        proposals = torch.cat(proposals, dim=1)
        valid = ((proposals > .01) & (proposals < .99)).all(-1, keepdim=True)
        logits = torch.log(proposals / (1 - proposals)).masked_fill(~mask[..., None] | ~valid, float("inf"))
        objects = self.enc_output_norm(self.enc_output(vision.masked_fill(~mask[..., None] | ~valid, 0)))
        classes = self.encoder_output_class_embed(objects, text, text_mask)
        boxes = self.encoder_output_bbox_embed(objects) + logits
        offsets = torch.arange(classes.numel() // classes.shape[-1] + 1, device=pixels.device) * classes.shape[-1]
        maxima = self.reduce(classes.flatten(), offsets, reduce="max").reshape(batch, -1)
        _, indices = self.select(maxima, self.config.num_queries)
        initial = self.sigmoid(boxes.gather(1, indices[..., None].expand(-1, -1, 4)))
        target = self.query_position_embeddings.emb.weight[None].expand(batch, -1, -1)
        encoder_logits = self.encoder_output_class_embed(target, original_text, text_mask)
        additive_text_mask = torch.zeros(batch, 1, 1, ids.shape[1], device=pixels.device, dtype=text.dtype).masked_fill(~text_mask[:, None, None], torch.finfo(text.dtype).min)
        hidden, states, references = self.decoder(target, initial, ratios, vision, text, shapes, mask, additive_text_mask)
        return {"last_hidden_state": hidden, "intermediate_hidden_states": states, "intermediate_reference_points": references,
                "init_reference_points": initial, "encoder_last_hidden_state_vision": vision, "encoder_last_hidden_state_text": text,
                "enc_outputs_class": classes, "enc_outputs_coord_logits": boxes, "encoder_logits": encoder_logits,
                "encoder_pred_boxes": initial, "input_ids": ids}


class _Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _Model(config)
        heads = [PredictionHead(config, config.d_model, config.d_model, 4, 3) for _ in range(config.decoder_layers)]
        if config.decoder_bbox_embed_share:
            heads = [heads[0]] * config.decoder_layers
        self.bbox_embed = nn.ModuleList(heads)
        classes = [_Contrastive(config) for _ in heads]
        if config.model_type == "mm-grounding-dino":
            classes = [classes[0]] * len(heads)
        self.class_embed = nn.ModuleList(classes)
        self.model.decoder.bbox_embed, self.model.decoder.class_embed = self.bbox_embed, self.class_embed
        self.sigmoid, self.logit = Sigmoid(), GroundingLogit()

    def forward(self, pixel_values, input_ids, token_type_ids=None, attention_mask=None, pixel_mask=None):
        output = self.model(pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask)
        states, references, initial = output["intermediate_hidden_states"], output["intermediate_reference_points"], output["init_reference_points"]
        mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        classes, boxes = [], []
        for level, head in enumerate(self.bbox_embed):
            reference = initial if level == 0 else references[:, level - 1]
            classes.append(self.class_embed[level](states[:, level], output["encoder_last_hidden_state_text"], mask))
            boxes.append(self.sigmoid(head(states[:, level]) + self.logit(reference)))
        output.update(logits=torch.stack(classes)[-1], pred_boxes=torch.stack(boxes)[-1])
        return output


def build_from_config(config, device, dtype):
    if (config.backbone_config.model_type != "swin" or config.text_config.model_type != "bert"
            or not config.two_stage or not config.embedding_init_target or config.query_dim != 4
            or config.num_feature_levels != 4 or config.activation_function != "relu"):
        raise ValueError("This case preserves the published Swin/BERT two-stage Grounding DINO path")
    return _Detector(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    prefix = "model.backbone.conv_encoder.model."
    _load_swin_backbone(model.model.backbone.conv_encoder.model, {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}, config.backbone_config)
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            continue
        key = key.replace("query_position_embeddings.weight", "query_position_embeddings.emb.weight")
        key = re.sub(r"((?:deformable_layer|text_enhancer_layer)|decoder.layers.\d+)\.(fc[12])\.", r"\1.mlp.\2.", key)
        if key.startswith("model.text_backbone."):
            key = key.replace(".weight", ".emb.weight") if ".embeddings." in key and "LayerNorm" not in key else key
            if any('.' + projection + '.' in key for projection in ("query", "key", "value")):
                if ".query." not in key:
                    continue
                value = torch.cat([state_dict[key.replace(".query.", '.' + projection + '.')] for projection in ("query", "key", "value")])
                key = key.replace(".query.", ".qkv.")
        mapped[key] = value
    for key, value in model.state_dict().items():
        if key.startswith(prefix):
            mapped[key] = value
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
