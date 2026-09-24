"""DiNAT from existing conv/MLP ops and bounded gathered neighborhood attention.

There is no standalone optimized neighborhood-attention interface in the audit
inventory. Each query uses exactly its K*K real dilated neighbors, gathered in
chunks, with existing BatchMatMul/Softmax operations; no global dense attention
or NATTEN computation is used here.
"""
import torch
from torch import nn

from ..runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax


def _axis_neighbors(length, kernel, dilation, device):
    # Window shifting occurs within each dilation residue class. These are
    # spatial metadata only, including the uneven residue lengths at boundaries.
    position = torch.arange(length, device=device)
    residue, ordinal = position % dilation, position // dilation
    count = (length - 1 - residue) // dilation + 1
    start = torch.minimum(torch.clamp(ordinal - kernel // 2, min=0), count - kernel)
    offset = torch.arange(kernel, device=device)
    neighbors = (start[:, None] + offset) * dilation + residue[:, None]
    bias = kernel - 1 - (ordinal - start)[:, None] + offset
    return neighbors, bias


class NeighborhoodAttention(nn.Module):
    # Batch independent queries to limit dispatch overhead without gathering the whole image.
    query_chunk_size = 512

    def __init__(self, config, dim, heads, dilation):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.kernel, self.dilation = config.kernel_size, dilation
        self.query = Linear(dim, dim, bias=config.qkv_bias)
        self.key = Linear(dim, dim, bias=config.qkv_bias)
        self.value = Linear(dim, dim, bias=config.qkv_bias)
        self.rpb = nn.Parameter(torch.empty(heads, 2*self.kernel-1, 2*self.kernel-1))
        self.matmul = BatchMatMul()
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch, height, width, dim = x.shape
        count = height * width
        q, k, v = [op(x).reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
                   for op in (self.query, self.key, self.value)]
        q = q / self.head_dim**0.5
        rows, row_bias = _axis_neighbors(height, self.kernel, self.dilation, x.device)
        cols, col_bias = _axis_neighbors(width, self.kernel, self.dilation, x.device)
        result = []
        for start in range(0, count, self.query_chunk_size):
            positions = torch.arange(start, min(start+self.query_chunk_size, count), device=x.device)
            r, c = positions // width, positions % width
            indices = (rows[r, :, None] * width + cols[c, None, :]).flatten(1)
            # At most [B, heads, query_chunk_size, K*K, head_dim] for each
            # gathered K/V, O(B*heads*chunk*K*K*head_dim), independent of H*W.
            keys, values = k[:, :, indices], v[:, :, indices]
            n = positions.numel()
            scores = self.matmul(q[:, :, start:start+n].reshape(-1, 1, self.head_dim),
                                 keys.reshape(-1, self.kernel**2, self.head_dim).transpose(1, 2))
            scores = scores.reshape(batch, self.heads, n, self.kernel**2)
            bias = self.rpb[:, row_bias[r, :, None], col_bias[c, None, :]].flatten(-2)
            probabilities = self.softmax(scores + bias[None])
            context = self.matmul(probabilities.reshape(-1, 1, self.kernel**2),
                                  values.reshape(-1, self.kernel**2, self.head_dim))
            result.append(context.reshape(batch, self.heads, n, self.head_dim))
        return torch.cat(result, dim=2).transpose(1, 2).reshape(batch, height, width, dim)


class Layer(nn.Module):
    def __init__(self, config, dim, heads, dilation):
        super().__init__()
        self.window = config.kernel_size * dilation
        self.layernorm_before = LayerNorm(dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.attention = nn.ModuleDict({
            'self': NeighborhoodAttention(config, dim, heads, dilation),
            'output': nn.ModuleDict({'dense': Linear(dim, dim)})})
        self.layernorm_after = LayerNorm(dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.intermediate = nn.ModuleDict({'dense': Linear(dim, int(dim*config.mlp_ratio))})
        self.output = nn.ModuleDict({'dense': Linear(int(dim*config.mlp_ratio), dim)})
        self.activation = GELU()

    def forward(self, x):
        normalized = self.layernorm_before(x)
        batch, height, width, dim = x.shape
        if height < self.window or width < self.window:
            padded = normalized.new_zeros(batch, max(height, self.window), max(width, self.window), dim)
            padded[:, :height, :width] = normalized
            normalized = padded
        attended = self.attention['output']['dense'](self.attention['self'](normalized))
        x = x + attended[:, :height, :width]
        return x + self.output['dense'](self.activation(self.intermediate['dense'](self.layernorm_after(x))))


class Downsampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = Conv2d(dim, 2*dim, 3, stride=2, padding=1, bias=False)
        self.norm = LayerNorm(2*dim, eps=1e-5, promote_fp32=False)

    def forward(self, x):
        return self.norm(self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))


class Stage(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        dim = config.embed_dim * 2**index
        self.layers = nn.ModuleList([Layer(config, dim, config.num_heads[index], dilation)
                                    for dilation in config.dilations[index]])
        self.downsample = Downsampler(dim) if index+1 < len(config.depths) else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.downsample(x)


class DinatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embed_dim
        self.embeddings = nn.ModuleDict({
            'patch_embeddings': nn.ModuleDict({'projection': nn.Sequential(
                Conv2d(config.num_channels, dim//2, 3, stride=2, padding=1),
                Conv2d(dim//2, dim, 3, stride=2, padding=1))}),
            'norm': LayerNorm(dim, eps=1e-5, promote_fp32=False)})
        self.encoder = nn.ModuleDict({'levels': nn.ModuleList([Stage(config, i) for i in range(len(config.depths))])})
        self.layernorm = LayerNorm(dim*2**(len(config.depths)-1), eps=config.layer_norm_eps, promote_fp32=False)
        self.pool = GlobalAvgPool2d()

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError('DiNAT coverage supports inference only')
        x = self.embeddings['patch_embeddings']['projection'](pixel_values).permute(0, 2, 3, 1)
        x = self.embeddings['norm'](x)
        for stage in self.encoder['levels']:
            x = stage(x)
        x = self.layernorm(x)
        return {'last_hidden_state': x, 'pooler_output': self.pool(x.permute(0, 3, 1, 2))}


def build_from_config(config, device, dtype):
    if (config.patch_size != 4 or config.hidden_act != 'gelu' or config.layer_scale_init_value != 0
            or config.output_attentions or config.output_hidden_states or config.chunk_size_feed_forward):
        raise ValueError('DiNAT coverage preserves the default patch/GELU/no-layer-scale inference path')
    if len(config.depths) != 4 or any(len(ds) != n for ds, n in zip(config.dilations, config.depths)):
        raise ValueError('DiNAT requires four stages and one dilation per layer')
    return DinatModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    del config
    if set(inputs) != {'pixel_values'}:
        raise ValueError('DiNAT expects pixel_values')
    return {'forward': Workload(run=lambda: model(**inputs))}
