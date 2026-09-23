"""SuperPoint keypoints and descriptors with explicit existing-op suppression."""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.descriptor_grid_sample import DescriptorGridSample
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax


class PointSuppression(nn.Module):
    """Native two suppression rounds using max-pool and first-index selectors."""

    def __init__(self, radius):
        super().__init__()
        self.pool = MaxPool2d(2*radius+1, stride=1, padding=radius)
        self.top1 = CodecTop1()

    def equal_to_local_max(self, scores):
        # scores <= pooled scores, so first-index selection means equality.
        return self.top1(torch.stack((scores, self.pool(scores)), dim=-1)) == 0

    def forward(self, scores):
        maxima = self.equal_to_local_max(scores)
        for _ in range(2):
            occupied = self.pool(maxima.float())
            suppressed = self.top1(torch.stack((torch.zeros_like(occupied), occupied), dim=-1)).bool()
            remaining = scores.masked_fill(suppressed, 0)
            maxima = maxima | (self.equal_to_local_max(remaining) & ~suppressed)
        return scores.masked_fill(~maxima, 0)


class _EncoderBlock(nn.Module):
    def __init__(self, inputs, outputs, pooling):
        super().__init__()
        self.conv_a, self.conv_b = Conv2d(inputs, outputs, 3, padding=1), Conv2d(outputs, outputs, 3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2d(2) if pooling else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.conv_b(self.relu(self.conv_a(x)))))


class _Points(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_score_a = Conv2d(c.encoder_hidden_sizes[-1], c.decoder_hidden_size, 3, padding=1)
        self.conv_score_b = Conv2d(c.decoder_hidden_size, c.keypoint_decoder_dim, 1)
        self.relu, self.softmax = ReLU(), Softmax(dim=1)
        self.suppress = PointSuppression(c.nms_radius)
        self.threshold, self.border = c.keypoint_threshold, c.border_removal_distance
        self.top1 = CodecTop1()

    def forward(self, hidden):
        scores = self.softmax(self.conv_score_b(self.relu(self.conv_score_a(hidden))))[:, :-1]
        batch, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch, h, w, 8, 8)
        scores = self.suppress(scores.permute(0, 1, 3, 2, 4).reshape(batch, h*8, w*8))[0]
        selected = self.top1(torch.stack((torch.full_like(scores, self.threshold), scores), dim=-1)).bool()
        indices = selected.nonzero()
        # Keep the pinned reference's declared border bounds, including its *8.
        valid = ((indices[:, 0] >= self.border) & (indices[:, 0] < h*64-self.border)
                 & (indices[:, 1] >= self.border) & (indices[:, 1] < w*64-self.border))
        indices = indices[valid]
        return indices.flip(1).to(scores.dtype), scores[indices[:, 0], indices[:, 1]]


class _Descriptors(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_descriptor_a = Conv2d(c.encoder_hidden_sizes[-1], c.decoder_hidden_size, 3, padding=1)
        self.conv_descriptor_b = Conv2d(c.decoder_hidden_size, c.descriptor_decoder_dim, 1)
        self.relu, self.norm, self.sample = ReLU(), L2Norm(dim=1), DescriptorGridSample()

    def forward(self, hidden, keypoints):
        descriptors = self.norm(self.conv_descriptor_b(self.relu(self.conv_descriptor_a(hidden))))
        _, channels, h, w = descriptors.shape
        # Selected pixel indices are position metadata; preserve native dtype rounding.
        grid = keypoints[None] - 4 + 0.5
        grid = grid / grid.new_tensor([w*8-4.5, h*8-4.5])
        grid = (grid*2-1).reshape(1, 1, -1, 2)
        sampled = self.sample(descriptors, grid).reshape(1, channels, -1)
        return self.norm(sampled)[0].T


class SuperPointForKeypointDetection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.encoder = nn.Module()
        channels = [1, *c.encoder_hidden_sizes]
        self.encoder.conv_blocks = nn.ModuleList([_EncoderBlock(a, b, i<len(channels)-2)
                                                 for i, (a, b) in enumerate(zip(channels, channels[1:]))])
        self.keypoint_decoder, self.descriptor_decoder = _Points(c), _Descriptors(c)

    def forward(self, pixel_values):
        hidden = pixel_values[:, :1]
        for block in self.encoder.conv_blocks:
            hidden = block(hidden)
        points_scores = [self.keypoint_decoder(item[None]) for item in hidden]
        descriptors = [self.descriptor_decoder(item[None], points)
                       for item, (points, _) in zip(hidden, points_scores)]
        batch, _, height, width = pixel_values.shape
        count = max(points.shape[0] for points, _ in points_scores)
        outputs = {"keypoints": torch.zeros(batch, count, 2, device=hidden.device),
                   "scores": torch.zeros(batch, count, device=hidden.device),
                   "descriptors": torch.zeros(batch, count, self.config.descriptor_decoder_dim, device=hidden.device),
                   "mask": torch.zeros(batch, count, device=hidden.device, dtype=torch.int)}
        for i, ((points, scores), features) in enumerate(zip(points_scores, descriptors)):
            length = points.shape[0]
            outputs["keypoints"][i, :length] = points
            outputs["scores"][i, :length] = scores
            outputs["descriptors"][i, :length] = features
            outputs["mask"][i, :length] = 1
        outputs["keypoints"] = outputs["keypoints"] / hidden.new_tensor([width, height], dtype=torch.float32)
        return outputs


def build_from_config(config, device, dtype):
    if config.max_keypoints != -1 or config.keypoint_decoder_dim != 65:
        raise ValueError("This case preserves the documented unlimited-keypoint, eight-pixel decoder")
    return SuperPointForKeypointDetection(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
