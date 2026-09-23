"""LightGlue image-pair matching with native SuperPoint, early stopping, and pruning."""
import math
import torch
from torch import nn
from .superpoint import SuperPointForKeypointDetection
from .superglue import _Matches
from ..patches.forecast_revin import ForecastNormalize
from ..runner import Workload
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.log_sigmoid import LogSigmoid
from fastkernels.tasks.baseline.L1.softmax import LogSoftmax
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.sam3_position_encoding import Sam3PositionEncoding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding


class _Position(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projector = Linear(2, config.hidden_size // config.num_attention_heads // 2, bias=False)
        self.encode = Sam3PositionEncoding(d_model=4, temperature=1., scale=1.)

    def forward(self, points):
        angles = self.projector(points)
        x, y = self.encode._encode_xy(angles[..., ::2].flatten(), angles[..., 1::2].flatten())
        both = torch.stack((x, y), dim=1).reshape(*angles.shape, 2).to(angles.dtype)
        return both[..., 1].repeat_interleave(2, -1), both[..., 0].repeat_interleave(2, -1)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        self.q_proj = Linear(config.hidden_size, self.heads * self.head_dim, bias=config.attention_bias)
        self.k_proj, self.v_proj = (Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=config.attention_bias) for _ in range(2))
        self.o_proj = Linear(self.heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.attention = DenseAttention(backend='sdpa')

    def rotary(self, q, k, position):
        rows = q.shape[0] * q.shape[1]
        # The learned table has one interleaved cosine/sine pair per keypoint.
        table = torch.cat((position[0][..., ::2], position[1][..., ::2]), dim=-1)
        query, key = RotaryEmbedding.forward_native_interleaved(
            torch.arange(rows, device=q.device), q.float().reshape(rows, -1),
            k.float().reshape(rows, -1), self.head_dim, table.reshape(rows, -1).float())
        return query.reshape_as(q).to(q.dtype), key.reshape_as(k).to(k.dtype)

    def forward(self, hidden, mask, position=None, memory=None):
        batch, length, _ = hidden.shape
        memory = hidden if memory is None else memory
        q = self.q_proj(hidden).reshape(batch, length, self.heads, self.head_dim)
        k, v = (projection(memory).reshape(batch, length, self.kv_heads, self.head_dim) for projection in (self.k_proj, self.v_proj))
        if position is not None:
            q, k = self.rotary(q, k, position)
        k, v = (x.repeat_interleave(self.heads // self.kv_heads, dim=2) for x in (k, v))
        return self.o_proj(self.attention(q, k, v, attn_mask=mask).reshape(batch, length, -1))


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1, self.fc2 = Linear(config.intermediate_size, config.intermediate_size), Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.intermediate_size, eps=1e-5, promote_fp32=False)
        self.activation = GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.layer_norm(self.fc1(x))))


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention, self.cross_attention = _Attention(config), _Attention(config)
        self.self_mlp, self.cross_mlp = _MLP(config), _MLP(config)

    def forward(self, hidden, position, mask):
        hidden = hidden + self.self_mlp(torch.cat((hidden, self.self_attention(hidden, mask, position)), dim=-1))
        attended = self.cross_attention(hidden, mask.flip(0), memory=hidden.flip(0))
        return hidden + self.cross_mlp(torch.cat((hidden, attended), dim=-1))


class _Assignment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_projection, self.matchability = Linear(config.descriptor_dim, config.descriptor_dim), Linear(config.descriptor_dim, 1)
        self.bmm, self.logsoftmax, self.logsigmoid = BMM(), LogSoftmax(dim=2), LogSigmoid()
        self.scale = config.descriptor_dim ** .25

    def forward(self, hidden, mask):
        projected = self.final_projection(hidden) / self.scale
        similarity = self.bmm(projected[None, 0], projected[None, 1].transpose(1, 2))
        similarity = similarity.masked_fill(~(mask[None, 0, :, None] & mask[None, 1, None, :]), torch.finfo(hidden.dtype).min)
        matchability = self.matchability(hidden)
        certainty = self.logsigmoid(matchability[None, 0]) + self.logsigmoid(matchability[None, 1]).transpose(1, 2)
        length = hidden.shape[1]
        scores = hidden.new_zeros(1, length + 1, length + 1)
        scores[:, :-1, :-1] = self.logsoftmax(similarity) + self.logsoftmax(similarity.transpose(1, 2).contiguous()).transpose(1, 2) + certainty
        scores[:, :-1, -1] = self.logsigmoid(-matchability[None, 0, :, 0])
        scores[:, -1, :-1] = self.logsigmoid(-matchability[None, 1, :, 0])
        return scores


class _Confidence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token, self.sigmoid = Linear(config.descriptor_dim, 1), Sigmoid()

    def forward(self, hidden):
        return self.sigmoid(self.token(hidden))[..., 0]


class _LightGlue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.keypoint_detector = SuperPointForKeypointDetection(config.keypoint_detector_config)
        self.input_projection = nn.Identity() if config.descriptor_dim == config.keypoint_detector_config.descriptor_decoder_dim else Linear(config.keypoint_detector_config.descriptor_decoder_dim, config.descriptor_dim)
        self.positional_encoder = _Position(config)
        self.transformer_layers = nn.ModuleList([_Layer(config) for _ in range(config.num_hidden_layers)])
        self.match_assignment_layers = nn.ModuleList([_Assignment(config) for _ in range(config.num_hidden_layers)])
        self.token_confidence = nn.ModuleList([_Confidence(config) for _ in range(config.num_hidden_layers - 1)])
        self.matches, self.reduce, self.sigmoid = _Matches(), SegmentCSR(), Sigmoid()
        self.normalize_count = ForecastNormalize(tolerance=0.)

    def count(self, mask):
        values = mask.flatten().float()
        return self.reduce(values, torch.tensor([0, values.numel()], device=values.device), 'sum')

    @staticmethod
    def pad_selected(tensor, keep, padding=0):
        selected = [row[valid] for row, valid in zip(tensor, keep)]
        length = max(x.shape[0] for x in selected)
        output = tensor.new_full((2, length, *tensor.shape[2:]), padding)
        for i, x in enumerate(selected):
            output[i, :x.shape[0]] = x
        return output

    def forward(self, pixel_values):
        if pixel_values.shape[:2] != (1, 2):
            raise ValueError('Declared workload is one image pair; adaptive stopping and pruning remain enabled')
        _, _, channels, height, width = pixel_values.shape
        detected = self.keypoint_detector(pixel_values.reshape(2, channels, height, width))
        points = detected['keypoints'].to(pixel_values.dtype)
        hidden = detected['descriptors'].to(pixel_values.dtype)
        original_mask = detected['mask'].reshape(1, 2, -1)
        mask = detected['mask'].bool()
        length = hidden.shape[1]
        if length == 0:
            return dict(matches=torch.full((1, 2, 0), -1, dtype=torch.int, device=points.device), matching_scores=points.new_zeros(1, 2, 0), keypoints=points[None], prune=points.new_zeros(1, 2, 0), mask=original_mask)
        # HF retains the original detected count even after pruning points.
        total = self.count(mask)
        size = points.new_tensor([width, height])
        position = self.positional_encoder((points * size - size / 2) / (max(width, height) / 2))
        hidden = self.input_projection(hidden.contiguous())
        indices = torch.arange(length, device=points.device).expand(2, -1)
        prune = torch.ones_like(indices)
        for i, layer in enumerate(self.transformer_layers):
            additive_mask = hidden.new_zeros(2, 1, 1, hidden.shape[1]).masked_fill(~mask[:, None, None], torch.finfo(hidden.dtype).min)
            hidden = layer(hidden, position, additive_mask)
            stop = i == len(self.transformer_layers) - 1
            threshold = .8 + .1 * math.exp(-4 * i / len(self.transformer_layers))
            if not stop:
                confidence = self.token_confidence[i](hidden)
                padded_confidence = confidence.masked_fill(~mask, 1)
                # Strict < uses reversed top-1 ordering so ties remain false.
                low = self.matches.top1(torch.stack((padded_confidence, torch.full_like(padded_confidence, threshold)), dim=-1)).bool()
                fraction = self.normalize_count(self.count(low), total.new_zeros(1), total)
                ratio = 1. - fraction.reshape(1)
                stop = bool(self.matches.greater(ratio, self.config.depth_confidence).item())
            if stop:
                matches, scores = self.matches(self.match_assignment_layers[i](hidden, mask), self.config.filter_threshold, False)
                break
            matchability = self.sigmoid(self.match_assignment_layers[i].matchability(hidden))[..., 0]
            keep = (self.matches.greater(matchability, 1 - self.config.width_confidence) | ~self.matches.greater(confidence, threshold)) & mask
            for image in range(2):
                prune[image, indices[image, keep[image]]] += 1
            hidden = self.pad_selected(hidden, keep)
            position = tuple(self.pad_selected(x, keep) for x in position)
            indices, mask = self.pad_selected(indices, keep, -1), self.pad_selected(mask, keep)
        final_matches = matches.new_full((1, 2, length), -1)
        final_scores = scores.new_zeros(1, 2, length)
        for image in range(2):
            target = indices[1-image].gather(0, matches[0, image].clamp(min=0)).masked_fill(matches[0, image] == -1, -1)
            final_matches[0, image, indices[image]] = target
            final_scores[0, image, indices[image]] = scores[0, image]
        return dict(matches=final_matches, matching_scores=final_scores, keypoints=points[None], prune=prune[None], mask=original_mask)


def build_from_config(config, device, dtype):
    if config.depth_confidence <= 0 or config.width_confidence <= 0 or config.keypoint_detector_config.max_keypoints != -1:
        raise ValueError('This workload preserves native adaptive depth/width and unlimited SuperPoint points')
    return _LightGlue(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
