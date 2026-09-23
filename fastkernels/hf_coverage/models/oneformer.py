"""OneFormer inference composed from the existing Swin and segmentation operations."""

import copy
import re
import torch
from torch import nn

from .mask2former import _SwinBackbone, _load_swin_backbone, _PixelDecoder, _feature_positions, _CrossAttention, _MaskPredictor
from .detr import _MLP
from ..runner import Workload
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR


class _Attention(_CrossAttention):
    def forward(self, hidden, query_positions, memory, positions, mask=None, mean=True):
        batch, queries, width = hidden.shape
        shape = lambda x: x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        query = shape(self.q_proj(hidden + query_positions)) * self.head_dim**-0.5
        key, value = shape(self.k_proj(memory + positions)), shape(self.v_proj(memory))
        scores = self.bmm(query, key.transpose(-1, -2))
        if mask is not None:
            scores = scores.masked_fill(mask[:, None], float('-inf'))
        probs = self.softmax(scores)
        output = self.out_proj(self.bmm(probs, value).transpose(1, 2).reshape(batch, queries, width))
        if mean:
            rows = probs.permute(0, 2, 3, 1).contiguous().flatten()
            offsets = torch.arange(0, rows.numel() + 1, self.heads, device=rows.device)
            weights = self.reduce(rows, offsets, reduce='mean').reshape(batch, queries, -1)
        else:
            weights = probs
        return output, weights


class _QueryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = _Attention(config.hidden_dim, config.num_attention_heads)
        self.multihead_attn = _Attention(config.hidden_dim, config.num_attention_heads)
        self.mlp = _MLP(config.hidden_dim, config.dim_feedforward)
        self.norm1, self.norm2, self.norm3 = (LayerNorm(config.hidden_dim, eps=config.layer_norm_eps, promote_fp32=False) for _ in range(3))

    def forward(self, hidden, memory, positions, query_positions):
        hidden = self.norm1(hidden + self.self_attn(hidden, query_positions, hidden, query_positions)[0])
        hidden = self.norm2(hidden + self.multihead_attn(hidden, query_positions, memory, positions)[0])
        return self.norm3(hidden + self.mlp(hidden))


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attn, self.self_attn, self.ffn = nn.Module(), nn.Module(), nn.Module()
        self.cross_attn.multihead_attn = _Attention(config.hidden_dim, config.num_attention_heads)
        self.self_attn.self_attn = _Attention(config.hidden_dim, config.num_attention_heads)
        for module in (self.cross_attn, self.self_attn, self.ffn):
            module.norm = LayerNorm(config.hidden_dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.ffn.mlp = _MLP(config.hidden_dim, config.dim_feedforward)

    def forward(self, hidden, query_positions, memory, positions, mask):
        attended, cross_weights = self.cross_attn.multihead_attn(hidden, query_positions, memory, positions, mask)
        hidden = self.cross_attn.norm(hidden + attended)
        attended, self_weights = self.self_attn.self_attn(hidden, query_positions, hidden, query_positions, mean=False)
        hidden = self.self_attn.norm(hidden + attended)
        return self.ffn.norm(hidden + self.ffn.mlp(hidden)), (self_weights, cross_weights)


class _Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.width = width
        self.queries_embedder = Embedding(config.num_queries, width)
        self.level_embed = Embedding(3, width)
        self.decoder = nn.Module()
        self.decoder.query_transformer = nn.Module()
        self.decoder.query_transformer.decoder = nn.Module()
        self.decoder.query_transformer.decoder.layers = nn.ModuleList([_QueryLayer(config) for _ in range(config.query_dec_layers)])
        self.decoder.query_transformer.decoder.norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.decoder.decoder_norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.decoder.query_input_projection = Conv2d(config.conv_dim, width, 1)
        self.decoder.layers = nn.ModuleList([_Layer(config) for _ in range(config.decoder_layers - 1)])
        self.decoder.class_embed = Linear(width, len(config.id2label) + 1)
        mask_config = copy.copy(config)
        mask_config.mask_feature_size = config.mask_dim
        self.decoder.mask_predictor = _MaskPredictor(mask_config)
        self.reduce = SegmentCSR()

    def forward(self, scales, pixels, task):
        batch = pixels.shape[0]
        decoder = self.decoder
        task = decoder.decoder_norm(task)
        query_positions = self.queries_embedder.emb.weight[None].expand(batch, -1, -1)
        hidden = task[:, None].expand(-1, query_positions.shape[1] - 1, -1)
        memory = _feature_positions(pixels, self.width)
        positions = decoder.query_input_projection(pixels).flatten(2).transpose(1, 2)
        for layer in decoder.query_transformer.decoder.layers:
            hidden = layer(hidden, memory, positions, query_positions[:, :-1])
        hidden = decoder.query_transformer.decoder.norm(hidden)
        hidden = torch.cat((hidden, task[:, None]), dim=1)
        contrastive = hidden
        memories = [x.flatten(2).transpose(1, 2) + self.level_embed.emb.weight[i] for i, x in enumerate(scales)]
        positions = [_feature_positions(x, self.width) for x in scales]
        sizes = [x.shape[-2:] for x in scales]
        predictions, attentions = [], []
        normalized = decoder.decoder_norm(hidden)
        masks, mask = decoder.mask_predictor(normalized, pixels, sizes[0])
        predictions.append(dict(class_queries_logits=decoder.class_embed(normalized), masks_queries_logits=masks))
        for index, layer in enumerate(decoder.layers):
            offsets = torch.arange(0, mask.numel() + 1, mask.shape[-1], device=mask.device)
            all_masked = self.reduce(mask.float().flatten(), offsets, reduce='min').bool().reshape(mask.shape[:2])
            mask = mask.masked_fill(all_masked[..., None], False)
            hidden, weights = layer(hidden, query_positions, memories[index % 3], positions[index % 3], mask)
            attentions.append(weights)
            normalized = decoder.decoder_norm(hidden)
            masks, mask = decoder.mask_predictor(normalized, pixels, sizes[(index + 1) % 3])
            predictions.append(dict(class_queries_logits=decoder.class_embed(normalized), masks_queries_logits=masks))
        return hidden, contrastive, predictions, tuple(attentions)


class _Segmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.pixel_level_module = nn.Module()
        self.model.pixel_level_module.encoder = _SwinBackbone(config.backbone_config)
        pixel_config = copy.copy(config)
        pixel_config.feature_size, pixel_config.mask_feature_size = config.conv_dim, config.mask_dim
        self.model.pixel_level_module.decoder = _PixelDecoder(pixel_config)
        self.model.transformer_module = _Transformer(config)
        self.model.task_encoder = nn.Module()
        self.model.task_encoder.task_mlp = nn.Sequential(Linear(config.task_seq_len, config.hidden_dim), ReLU(), Linear(config.hidden_dim, config.hidden_dim))
        self.criterion = nn.Module()
        self.criterion.register_buffer('empty_weight', torch.ones(len(config.id2label) + 1))
        self.criterion.logit_scale = nn.Parameter(torch.zeros(()))

    def forward(self, pixel_values, task_inputs):
        features = self.model.pixel_level_module.encoder(pixel_values)
        pixels, scales = self.model.pixel_level_module.decoder(features)
        task = self.model.task_encoder.task_mlp(task_inputs.to(pixel_values.dtype))
        hidden, contrastive, predictions, attentions = self.model.transformer_module(scales, pixels, task)
        final, auxiliary = predictions[-1], tuple(predictions[:-1])
        outputs = dict(final)
        outputs.update(transformer_decoder_object_queries=hidden, transformer_decoder_contrastive_queries=contrastive,
                       transformer_decoder_mask_predictions=final['masks_queries_logits'],
                       transformer_decoder_class_predictions=final['class_queries_logits'], task_token=task)
        if self.config.use_auxiliary_loss:
            outputs['transformer_decoder_auxiliary_predictions'] = auxiliary
        if self.config.output_auxiliary_logits:
            outputs['auxiliary_predictions'] = auxiliary
        if self.config.output_hidden_states:
            outputs.update(encoder_hidden_states=tuple(features), pixel_decoder_hidden_states=(pixels, *scales),
                           transformer_decoder_hidden_states=auxiliary)
        if self.config.output_attentions:
            outputs['attentions'] = attentions
        return outputs


def build_from_config(config, device, dtype):
    if (config.backbone_config.model_type != 'swin' or config.pre_norm or config.enforce_input_proj
            or config.conv_dim != config.hidden_dim or config.common_stride != 4 or config.is_training
            or not config.use_task_norm):
        raise ValueError('This case preserves the published Swin-tiny inference configuration')
    return _Segmentation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    state = dict(state_dict)
    prefix = 'model.pixel_level_module.encoder.'
    backbone = {key[len(prefix):]: state.pop(key) for key in list(state) if key.startswith(prefix)}
    _load_swin_backbone(model.model.pixel_level_module.encoder, backbone, config.backbone_config)
    mapped = {prefix + key: value for key, value in model.model.pixel_level_module.encoder.state_dict().items()}
    for key, value in state.items():
        if '.in_proj_' in key:
            field = key.rsplit('_', 1)[-1]
            for name, part in zip(('q', 'k', 'v'), value.chunk(3)):
                mapped[key.replace('in_proj_' + field, name + '_proj.' + field)] = part
            continue
        key = re.sub(r'(layers\.\d+)\.(fc[12])\.', r'\1.mlp.\2.', key)
        key = re.sub(r'(query_transformer.decoder.layers\.\d+)\.linear([12])\.', r'\1.mlp.fc\2.', key)
        key = re.sub(r'\.ffn\.linear([12])\.', r'.ffn.mlp.fc\1.', key)
        key = re.sub(r'\.mask_embed\.layers\.(\d+)\.0\.', r'.mask_predictor.mask_embedder.layers.\1.', key)
        key = re.sub(r'\.task_mlp\.layers\.(\d+)\.0\.', lambda m: '.task_mlp.' + str(int(m[1]) * 2) + '.', key)
        for name in ('queries_embedder', 'level_embed'):
            key = key.replace(f'transformer_module.{name}.weight', f'transformer_module.{name}.emb.weight')
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    def run():
        result = {}

        def flatten(name, value):
            if isinstance(value, dict):
                for key, tensor in value.items():
                    flatten(f'{name}.{key}', tensor)
            elif isinstance(value, (tuple, list)):
                for index, tensor in enumerate(value):
                    flatten(f'{name}.{index}', tensor)
            else:
                result[name] = value

        for name, value in model(**inputs).items():
            flatten(name, value)
        return result

    return {'forward': Workload(run=run)}
