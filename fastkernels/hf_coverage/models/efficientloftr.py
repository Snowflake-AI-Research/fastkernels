"""EfficientLoFTR default single-pair matching through existing operations."""
import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.patches.ernie4_5_rope import FP32RotaryEmbedding
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.rtdetrv2_conv_norm import RTDetrV2ConvNormLayer


class LeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(x) - .01 * self.relu(-x)


class Block(nn.Module):
    def __init__(self, c, stage, block):
        super().__init__()
        i, o, s = (v[stage][block] for v in
                   (c.stage_block_in_channels, c.stage_block_out_channels, c.stage_block_stride))
        self.conv1 = RTDetrV2ConvNormLayer(c, i, o, 3, s, 1)
        self.conv2 = RTDetrV2ConvNormLayer(c, i, o, 1, s, 0)
        self.identity = BatchNorm2d(i) if i == o and s == 1 else None
        self.activation = ReLU()

    def forward(self, x):
        return self.activation(self.conv1(x) + self.conv2(x) + (self.identity(x) if self.identity else 0))


class AggregatedAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c.hidden_size
        self.heads, self.head_dim = c.num_attention_heads, h // c.num_attention_heads
        self.stride = c.q_aggregation_kernel_size
        self.aggregation = nn.Module()
        self.aggregation.q_aggregation = Conv2d(h, h, c.q_aggregation_kernel_size,
            stride=c.q_aggregation_stride, groups=h, bias=False)
        self.aggregation.kv_aggregation = MaxPool2d(c.kv_aggregation_kernel_size, c.kv_aggregation_stride)
        self.aggregation.norm = LayerNorm(h, promote_fp32=False)
        self.attention = nn.Module()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self.attention, name, Linear(h, h, bias=c.attention_bias))
        self.mlp = nn.Module()
        self.mlp.fc1 = Linear(2*h, c.intermediate_size, bias=False)
        self.mlp.fc2 = Linear(c.intermediate_size, h, bias=False)
        self.mlp.activation, self.mlp.layer_norm = LeakyReLU(), LayerNorm(h, promote_fp32=False)
        # The selected native default is SDPA; eager BF16 stores change the
        # thresholded coarse matches on the full processor-sized image pair.
        self.attend, self.resize = DenseAttention(backend="sdpa"), Interpolate()

    def forward(self, x, context=None, rope=None):
        a = self.aggregation
        q = a.norm(a.q_aggregation(x).permute(0, 2, 3, 1))
        kv = a.norm(a.kv_aggregation(x if context is None else context).permute(0, 2, 3, 1))
        b, h, w, d = q.shape
        q, kv = q.reshape(b, h*w, d), kv.reshape(b, h*w, d)
        q, k, v = self.attention.q_proj(q), self.attention.k_proj(kv), self.attention.v_proj(kv)
        if rope is not None:
            positions = torch.arange(h*w, device=x.device).expand(b, -1).reshape(-1)
            q, k = rope(positions, q.reshape(-1, d).clone(), k.reshape(-1, d).clone())
        shape = (b, h*w, self.heads, self.head_dim)
        hidden = self.attend(q.reshape(shape), k.reshape(shape), v.reshape(shape))
        hidden = self.attention.o_proj(hidden.reshape(b, h*w, d)).transpose(1, 2).reshape(b, d, h, w)
        hidden = self.resize(hidden, scale_factor=self.stride, mode='bilinear', align_corners=False)
        hidden = torch.cat((x, hidden), 1).permute(0, 2, 3, 1)
        hidden = self.mlp.layer_norm(self.mlp.fc2(self.mlp.activation(self.mlp.fc1(hidden))))
        return x + hidden.permute(0, 3, 1, 2)


class OutConv(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.out_conv1 = Conv2d(outgoing, incoming, 1, bias=False)
        self.out_conv2 = Conv2d(incoming, incoming, 3, padding=1, bias=False)
        self.batch_norm, self.activation = BatchNorm2d(incoming), LeakyReLU()
        self.out_conv3 = Conv2d(incoming, outgoing, 3, padding=1, bias=False)
        self.resize = Interpolate()

    def forward(self, x, residual):
        x = self.out_conv2(self.out_conv1(residual) + x)
        x = self.out_conv3(self.activation(self.batch_norm(x)))
        return self.resize(x, scale_factor=2, mode='bilinear', align_corners=False)


class EfficientLoFTR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.efficientloftr = nn.Module()
        self.efficientloftr.backbone = nn.Module()
        self.efficientloftr.backbone.stages = nn.ModuleList()
        for i, count in enumerate(c.stage_num_blocks):
            stage = nn.Module()
            stage.blocks = nn.ModuleList([Block(c, i, j) for j in range(count)])
            self.efficientloftr.backbone.stages.append(stage)
        self.efficientloftr.local_feature_transformer = nn.Module()
        layers = self.efficientloftr.local_feature_transformer.layers = nn.ModuleList()
        for _ in range(c.num_attention_layers):
            layer = nn.Module()
            layer.self_attention, layer.cross_attention = AggregatedAttention(c), AggregatedAttention(c)
            layers.append(layer)
        self.refinement_layer = nn.Module()
        dims = c.fine_fusion_dims
        self.refinement_layer.out_conv = Conv2d(dims[0], dims[0], 1, bias=False)
        self.refinement_layer.out_conv_layers = nn.ModuleList([OutConv(i, o) for i, o in zip(dims, dims[1:])])
        self.bmm, self.softmax = BMM(), Softmax()
        self.product, self.top1, self.resize = ProductGate(), CodecTop1(), Interpolate()

    def matmul(self, a, b):
        shape = a.shape[:-2]
        return self.bmm(a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:])).reshape(*shape, a.shape[-2], b.shape[-1])

    def dual_softmax(self, x, axis1, axis2):
        a = self.softmax(x.movedim(axis1, -1)).movedim(-1, axis1)
        b = self.softmax(x.movedim(axis2, -1)).movedim(-1, axis2)
        return self.product(torch.cat((a, b), -1))

    def maximum(self, x):
        index = self.top1(x)
        return x.gather(-1, index[..., None]).squeeze(-1), index

    def greater(self, a, b):
        return self.top1(torch.stack((torch.broadcast_to(torch.as_tensor(b, device=a.device, dtype=a.dtype), a.shape), a), -1)) == 1

    def coarse(self, x):
        b, _, d, h, w = x.shape
        features = x.permute(0, 1, 3, 4, 2).reshape(b, 2, h*w, d) / d**.5
        scores = self.matmul(features[:, 0], features[:, 1].transpose(-1, -2)) / self.config.coarse_matching_temperature
        scores = self.dual_softmax(scores, 1, 2)
        row, _ = self.maximum(scores)
        col, _ = self.maximum(scores.transpose(-1, -2))
        # Finite first-index Top1 expresses comparisons; all masks are routing metadata.
        mask = self.greater(scores, self.config.coarse_matching_threshold)
        mask &= self.top1(torch.stack((scores, row[..., None].expand_as(scores)), -1)) == 0
        mask &= self.top1(torch.stack((scores, col[:, None, :].expand_as(scores)), -1)) == 0
        mask = mask.reshape(b, h, w, h, w)
        border = self.config.coarse_matching_border_removal
        for axis in range(1, 5):
            first, last = [slice(None)]*5, [slice(None)]*5
            first[axis], last[axis] = slice(0, border), slice(-border, None)
            mask[tuple(first)] = False
            mask[tuple(last)] = False
        scores = scores.masked_fill(~mask.reshape(b, h*w, h*w), 0)
        score0, index0 = self.maximum(scores.transpose(-1, -2))
        score1, index1 = self.maximum(scores)
        scores = torch.stack((score0, score1), 1)
        indices = torch.cat((index0, index1)).reshape(b, 2, -1)
        indices = indices.masked_fill(~self.greater(scores, 0), -1)
        points = torch.stack((indices % w, indices // w), -1) * 8.0
        return indices, scores, points

    @staticmethod
    def windows(x, kernel, padding):
        x = F.pad(x, (padding,)*4)
        x = x.unfold(2, kernel, 8).unfold(3, kernel, 8)
        return x.permute(0, 2, 3, 4, 5, 1).reshape(x.shape[0], -1, kernel*kernel, x.shape[1])

    def fine(self, a, b, points):
        batch, n, window, dim = a.shape
        sliced = self.config.fine_matching_slice_dim
        firstdim = dim-sliced
        confidence = self.matmul(a[..., :firstdim]/firstdim**.5, (b[..., :firstdim]/firstdim**.5).transpose(-1, -2))
        confidence = self.dual_softmax(confidence, 1, 2)
        confidence = confidence.reshape(batch, n, window, 10, 10)[..., 1:-1, 1:-1]
        indices = self.top1(confidence.reshape(batch, n, -1))
        i0, i1 = indices // window, indices % window
        y, x = torch.meshgrid(torch.arange(8, device=a.device), torch.arange(8, device=a.device), indexing='ij')
        grid = torch.stack((x, y), -1).to(a.dtype).reshape(1, 1, 64, 2) - 4 + .5
        grid = grid.expand(batch, n, -1, -1)
        # Preserve pinned HF's dimension-1 gather, including invalid coarse matches.
        delta0 = grid.gather(1, i0[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        delta1 = grid.gather(1, i1[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        points = points + torch.stack((delta0, delta1), 1)
        confidence = self.matmul(a[..., firstdim:], (b[..., firstdim:]/sliced**.5).transpose(-1, -2)).reshape(batch, n, window, 10, 10)
        dy, dx = torch.meshgrid(torch.arange(-1, 2, device=a.device), torch.arange(-1, 2, device=a.device), indexing='ij')
        confidence = confidence[torch.arange(batch, device=a.device)[:, None, None, None],
            torch.arange(n, device=a.device)[None, :, None, None], i0[..., None, None],
            i1[..., None, None]//8+dy, i1[..., None, None]%8+dx]
        probability = self.softmax(confidence.reshape(batch, n, 9)/self.config.fine_matching_regress_temperature)
        grid = torch.stack((dx, dy), -1).to(a.dtype).reshape(1, 9, 2).expand(batch, -1, -1)
        expectation = self.matmul(probability, grid)[0]
        return torch.stack((points[:, 0], points[:, 1]+expectation), 1)

    def forward(self, pixel_values):
        batch, _, channels, height, width = pixel_values.shape
        if batch != 1 or height % 32 or width % 32 or min(height, width) < 64:
            raise ValueError('The audited single-pair workload uses spatial dimensions >=64 divisible by32')
        x = pixel_values.reshape(batch*2, channels, height, width)[:, :1]
        residuals = []
        for stage in self.efficientloftr.backbone.stages:
            for block in stage.blocks:
                x = block(x)
            residuals.append(x)
        d, h, w = x.shape[-3:]
        for layer in self.efficientloftr.local_feature_transformer.layers:
            x = layer.self_attention(x, rope=self.rope).reshape(batch, 2, d, h, w)
            a = layer.cross_attention(x[:, 0], x[:, 1])
            b = layer.cross_attention(x[:, 1], a)
            x = torch.stack((a, b), 1).reshape(batch*2, d, h, w)
        indices, scores, points = self.coarse(x.reshape(batch, 2, d, h, w))
        x = self.refinement_layer.out_conv(x / d**.5)
        x = self.resize(x, scale_factor=2, mode='bilinear', align_corners=False)
        for layer, residual in zip(self.refinement_layer.out_conv_layers, reversed(residuals[1:-1])):
            x = layer(x, residual)
        x = x.reshape(batch, 2, *x.shape[1:])
        a, b = self.windows(x[:, 0], 8, 0), self.windows(x[:, 1], 10, 1)
        batches = torch.arange(batch, device=x.device)[:, None]
        points = self.fine(a[batches, indices[:, 0]], b[batches, indices[:, 1]], points)
        points[..., 0], points[..., 1] = points[..., 0]/width, points[..., 1]/height
        return {'matches': indices, 'matching_scores': scores, 'keypoints': points}


def build_from_config(config, device, dtype):
    if (config.activation_function != 'relu' or config.mlp_activation_function != 'leaky_relu'
        or config.fine_kernel_size != 8 or config.coarse_matching_skip_softmax
        or config.q_aggregation_stride != 4 or config.kv_aggregation_stride != 4):
        raise ValueError('Expected the documented EfficientLoFTR default computation')
    return EfficientLoFTR(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    pixel = inputs['pixel_values']
    h, w = (s//32 for s in pixel.shape[-2:])
    dim = int(config.hidden_size//config.num_attention_heads * config.rope_parameters['partial_rotary_factor'])
    inv = 1.0 / (config.rope_parameters['rope_theta'] ** (torch.arange(0, dim, 2, device=pixel.device).float()/dim))
    # Only position metadata is prepared; rounded coefficients match HF's BF16 table.
    y, x = torch.meshgrid(torch.arange(1, h+1, device=pixel.device), torch.arange(1, w+1, device=pixel.device), indexing='ij')
    angles = torch.stack((y[..., None]*inv, x[..., None]*inv), -1).flatten(-2).reshape(h*w, -1)
    rope = FP32RotaryEmbedding(config.hidden_size, h*w, config.rope_parameters['rope_theta'], is_neox_style=False).to(pixel.device)
    rope.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), -1).to(pixel.dtype).float()
    model.rope = rope
    return {'forward': Workload(run=lambda: model(**inputs))}
