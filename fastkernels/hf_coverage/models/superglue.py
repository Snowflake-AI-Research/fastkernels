"""SuperGlue image-pair matching, including SuperPoint and log-space transport."""

import math
import torch
from torch import nn
from .superpoint import SuperPointForKeypointDetection
from ..patches.codec_top1 import CodecTop1
from ..patches.ratio_log import RatioLog
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from ..runner import Workload
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax, LogSoftmax
from fastkernels.tasks.baseline.L1.tensor_ops import Exp


class _MLP(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.linear, self.batch_norm, self.activation = Linear(source, target), BatchNorm2d(target), ReLU()

    def forward(self, hidden):
        hidden = self.linear(hidden).transpose(-1, -2)
        return self.activation(self.batch_norm(hidden).transpose(-1, -2))


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.self = nn.Module()
        self.self.query, self.self.key, self.self.value = (Linear(width, width) for _ in range(3))
        self.output = nn.Module()
        self.output.dense = Linear(width, width)
        self.bmm, self.softmax = BMM(), Softmax()

    def forward(self, hidden, memory, mask):
        batch, length, width = hidden.shape
        shape = lambda x: x.reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = shape(self.self.query(hidden)), shape(self.self.key(memory)), shape(self.self.value(memory))
        scores = self.bmm(q, k.transpose(-1, -2)) / math.sqrt(width // self.heads)
        output = self.bmm(self.softmax(scores + mask), v).transpose(1, 2).reshape(batch, length, width)
        return self.output.dense(output)


class _Propagation(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.attention = _Attention(config)
        self.mlp = nn.ModuleList([_MLP(width * 2, width * 2), Linear(width * 2, width)])

    def forward(self, hidden, memory, mask):
        output = torch.cat((hidden, self.attention(hidden, memory, mask)), dim=-1)
        for layer in self.mlp:
            output = layer(output)
        return output


class _Matches(nn.Module):
    def __init__(self):
        super().__init__()
        self.top1, self.exp = CodecTop1(), Exp()

    def greater(self, x, threshold):
        return self.top1(torch.stack((torch.full_like(x, threshold), x), dim=-1)).bool()

    def forward(self, scores, threshold, zero_below_threshold):
        matrix = scores[:, :-1, :-1]
        indices0 = self.top1(matrix)
        indices1 = self.top1(matrix.transpose(1, 2).contiguous())
        maximum = matrix.gather(2, indices0[..., None])[..., 0]
        mutual0 = torch.arange(indices0.shape[1], device=scores.device)[None] == indices1.gather(1, indices0)
        mutual1 = torch.arange(indices1.shape[1], device=scores.device)[None] == indices0.gather(1, indices1)
        scores0 = self.exp(maximum).masked_fill(~mutual0, 0)
        valid0 = mutual0 & self.greater(scores0, threshold)
        if zero_below_threshold:
            scores0 = scores0.masked_fill(~self.greater(scores0, threshold), 0)
            valid0 = mutual0 & self.greater(scores0, 0)
        scores1 = scores0.gather(1, indices1).masked_fill(~mutual1, 0)
        valid1 = mutual1 & valid0.gather(1, indices1)
        matches = torch.stack((indices0.masked_fill(~valid0, -1), indices1.masked_fill(~valid1, -1)), dim=1)
        return matches, torch.stack((scores0, scores1), dim=1)


class _LogSumExp(nn.Module):
    """Stable transport reduction, preserving native exp/sum/log boundaries.

    Every transport row and column includes a finite dustbin entry, including
    padded keypoints, so its maximum remains finite during valid execution.
    """
    def __init__(self):
        super().__init__()
        self.reduce, self.exp, self.log = SegmentCSR(), Exp(), RatioLog()

    def forward(self, x, dim):
        rows = x.movedim(dim, -1).contiguous()
        shape, width = rows.shape[:-1], rows.shape[-1]
        offsets = torch.arange(0, rows.numel() + 1, width, device=x.device)
        maximum = self.reduce(rows.flatten(), offsets, 'max').reshape(*shape, 1)
        exponentials = self.exp(rows - maximum)
        # Native BF16 sum accumulates FP32 and rounds before the logarithm.
        summed = self.reduce(exponentials.float().flatten(), offsets, 'sum').to(x.dtype).reshape(shape)
        return self.log(summed, torch.ones_like(summed)) + maximum[..., 0]


class _Transport(nn.Module):
    def __init__(self, iterations):
        super().__init__()
        self.iterations = iterations
        self.logsumexp = _LogSumExp()

    def forward(self, scores, bin_score):
        batch, rows, columns = scores.shape
        row_bin, column_bin = bin_score.expand(batch, rows, 1), bin_score.expand(batch, 1, columns)
        costs = torch.cat((torch.cat((scores, row_bin), dim=2), torch.cat((column_bin, bin_score.expand(batch, 1, 1)), dim=2)), dim=1)
        # Counts are shape metadata. Preserve native dtype rounding here.
        count_r, count_c = scores.new_tensor(rows), scores.new_tensor(columns)
        normalizer = -(count_r + count_c).log()
        source = torch.cat((normalizer.expand(rows), count_c.log()[None] + normalizer))[None].expand(batch, -1)
        target = torch.cat((normalizer.expand(columns), count_r.log()[None] + normalizer))[None].expand(batch, -1)
        u, v = torch.zeros_like(source), torch.zeros_like(target)
        for _ in range(self.iterations):
            x = costs + v[:, None]
            u = source - self.logsumexp(x, 2)
            x = costs + u[:, :, None]
            v = target - self.logsumexp(x, 1)
        return costs + u[:, :, None] + v[:, None] - normalizer


class _SuperGlue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.keypoint_detector = SuperPointForKeypointDetection(config.keypoint_detector_config)
        widths = [3, *config.keypoint_encoder_sizes, config.hidden_size]
        self.keypoint_encoder = nn.Module()
        self.keypoint_encoder.encoder = nn.ModuleList([_MLP(a, b) for a, b in zip(widths[:-2], widths[1:-1])] + [Linear(widths[-2], widths[-1])])
        self.gnn = nn.Module()
        self.gnn.layers = nn.ModuleList([_Propagation(config) for _ in config.gnn_layers_types])
        self.final_projection = nn.Module()
        self.final_projection.final_proj = Linear(config.hidden_size, config.hidden_size)
        self.bin_score = nn.Parameter(torch.ones(()))
        self.bmm, self.transport, self.matches = BMM(), _Transport(config.sinkhorn_iterations), _Matches()

    def forward(self, pixel_values):
        batch, pair, channels, height, width = pixel_values.shape
        detected = self.keypoint_detector(pixel_values.reshape(batch * pair, channels, height, width))
        points = detected['keypoints'].reshape(batch, pair, -1, 2).to(pixel_values.dtype)
        scores = detected['scores'].to(pixel_values.dtype)
        hidden = detected['descriptors'].to(pixel_values.dtype)
        mask = detected['mask']
        count = hidden.shape[1]
        if count == 0:
            return dict(matches=torch.full((batch, pair, 0), -1, device=points.device, dtype=torch.int), matching_scores=points.new_zeros(batch, pair, 0), keypoints=points, mask=mask.reshape(batch, pair, 0))
        size = points.new_tensor([width, height])
        normalized = (points.reshape(batch * pair, count, 2) * size - size / 2) / (max(width, height) * .7)
        encoded = torch.cat((normalized, scores[..., None]), dim=-1)
        for layer in self.keypoint_encoder.encoder:
            encoded = layer(encoded)
        hidden = hidden + encoded
        attention_mask = torch.zeros(batch * pair, 1, 1, count, device=hidden.device, dtype=hidden.dtype).masked_fill(~mask[:, None, None].bool(), torch.finfo(hidden.dtype).min)
        for layer, kind in zip(self.gnn.layers, self.config.gnn_layers_types):
            memory = hidden.reshape(batch, pair, count, -1).flip(1).flatten(0, 1) if kind == 'cross' else hidden
            memory_mask = attention_mask.reshape(batch, pair, 1, 1, count).flip(1).flatten(0, 1) if kind == 'cross' else attention_mask
            hidden = hidden + layer(hidden, memory, memory_mask)
        hidden = self.final_projection.final_proj(hidden).reshape(batch, pair, count, -1)
        similarity = self.bmm(hidden[:, 0], hidden[:, 1].transpose(1, 2)) / math.sqrt(self.config.hidden_size)
        pair_mask = mask.reshape(batch, pair, count).bool()
        similarity = similarity.masked_fill(~(pair_mask[:, 0, :, None] & pair_mask[:, 1, None, :]), torch.finfo(similarity.dtype).min)
        matches, matching_scores = self.matches(self.transport(similarity, self.bin_score), self.config.matching_threshold, True)
        return dict(matches=matches, matching_scores=matching_scores, keypoints=points, mask=mask.reshape(batch, pair, count))


def build_from_config(config, device, dtype):
    if config.keypoint_detector_config.max_keypoints != -1 or config.keypoint_detector_config.descriptor_decoder_dim != config.hidden_size:
        raise ValueError('Preserve native unlimited SuperPoint descriptors and matching width')
    return _SuperGlue(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
