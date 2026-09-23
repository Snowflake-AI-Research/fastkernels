"""MRA's selected fine tiles and coarse attention using existing operations.

The native CUDA maximum truncates to thousandths. This composition uses an
unquantized maximum with the same finite floor; its numerical effect on the
stabilized coarse/fine normalization requires independent reference validation.
"""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.tensor_ops import Exp
from fastkernels.tasks.baseline.L3.yolov10_head import v10postprocess
from ..patches.codec_top1 import CodecTop1
from ..patches.product_gate import ProductGate
from ..runner import Workload
from .bert import MaskedLMHead


class SparseCoarseAttention(nn.Module):
    def __init__(self, num_selected):
        super().__init__()
        self.num_selected = num_selected
        self.bmm, self.segment = BMM(), SegmentCSR()
        self.product, self.select = ProductGate(), CodecTop1()
        self.exp, self.relu = Exp(), ReLU()
        self.normalizer = BatchNorm2d(1, eps=1e-30, affine=False).eval()
        self.normalizer._non_persistent_buffers_set.update(self.normalizer._buffers)

    def multiply(self, x, y):
        x, y = torch.broadcast_tensors(x, y)
        return self.product(torch.cat((x, y), -1))

    def reduce(self, x, axis, mode):
        rows = x.movedim(axis, -1).contiguous()
        width = rows.shape[-1]
        offsets = torch.arange(0, rows.numel() + 1, width, device=x.device)
        return self.segment(rows.reshape(-1, 1), offsets, mode).reshape(rows.shape[:-1])

    def grouped(self, x, order, offsets, batch, blocks):
        return self.segment(x.flatten(0, 1)[order], offsets, 'sum').reshape(batch, blocks, *x.shape[2:])

    def forward(self, query, key, value, mask):
        batch, length, dim = query.shape
        if length % 32:
            raise ValueError('MRA requires sequence lengths divisible by32')
        blocks = length // 32
        if self.num_selected > blocks * blocks or blocks * blocks >= 2**24:
            raise ValueError('Selected MRA block count or float metadata index range is unsupported')
        query = self.multiply(query, mask[..., None])
        key = self.multiply(key, mask[..., None])
        value = self.multiply(value, mask[..., None])
        qb, kb, vb = (x.reshape(batch, blocks, 32, dim) for x in (query, key, value))
        # Counts and reciprocal factors depend only on supplied padding metadata.
        counts = mask.reshape(batch, blocks, 32).sum(-1)
        inverse_counts = (counts + 1e-6).reciprocal()[..., None]
        qhat, khat, vhat = (self.multiply(self.reduce(x, -2, 'sum'), inverse_counts) for x in (qb, kb, vb))
        low = self.bmm(qhat, khat.transpose(-1, -2)) * (dim**-0.5)
        low_max = self.reduce(low, -1, 'max')[..., None]
        invalid = counts[:, :, None] * counts[:, None, :] < 0.5
        low = low + torch.where(invalid, -1e4, 0.)
        normalized = low - low_max
        # Actual unchanged internal selector, with four metadata columns. Its
        # second top-k retains the selected set. No standalone routing interface.
        metadata = torch.zeros(batch, blocks * blocks, 4, device=query.device)
        metadata[:, :, 0] = torch.arange(blocks * blocks, device=query.device)
        boxes, selected_values, _ = v10postprocess(
            torch.cat((metadata, normalized.reshape(batch, -1, 1)), -1), self.num_selected, nc=1)
        indices = boxes[:, :, 0].long()
        threshold = self.reduce(selected_values, -1, 'min')[:, None, None]
        threshold = threshold.expand_as(normalized)
        # Ties belong to the high-resolution exclusion mask even if unselected.
        below_threshold = self.select(torch.stack((normalized, threshold), -1)).bool()
        coarse_penalty = torch.where(below_threshold, 0., -1e4)
        query_ids, key_ids = indices // blocks, indices % blocks
        batch_ids = torch.arange(batch, device=query.device)[:, None]
        global_ids = (query_ids + batch_ids * blocks).flatten()
        order = torch.argsort(global_ids, stable=True)
        group_counts = torch.bincount(global_ids, minlength=batch * blocks)
        offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=query.device), group_counts.cumsum(0)))
        fine = self.bmm(qb[batch_ids, query_ids], kb[batch_ids, key_ids].transpose(-1, -2)) * (dim**-0.5)
        tile_max = self.reduce(fine, -1, 'max')
        row_max = self.segment(tile_max.flatten(0, 1)[order], offsets, 'max')
        floor = torch.full_like(row_max, -1e5)
        use_floor = self.select(torch.stack((row_max, floor), -1)).bool()
        row_max = torch.where(use_floor, floor, row_max).reshape(batch, blocks, 32)
        selected_max = row_max[batch_ids, query_ids]
        selected_mask = mask.reshape(batch, blocks, 32)[batch_ids, key_ids]
        fine = fine - selected_max[..., None] - 1e4 * (1 - selected_mask[:, :, None, :])
        fine_weights = self.exp(fine)
        fine_out = self.grouped(self.bmm(fine_weights, vb[batch_ids, key_ids]), order, offsets, batch, blocks)
        fine_norm = self.grouped(self.reduce(fine_weights, -1, 'sum'), order, offsets, batch, blocks)
        coarse_weights = self.multiply(self.exp(normalized + coarse_penalty), counts[:, None, :])
        coarse_out = self.bmm(coarse_weights, vhat)[:, :, None, :].expand(-1, -1, 32, -1)
        coarse_norm = self.reduce(coarse_weights, -1, 'sum')[:, :, None].expand(-1, -1, 32)
        correction = self.multiply(low_max.expand(-1, -1, 32) - row_max, mask.reshape(batch, blocks, 32))
        coarse_scale = self.exp(-self.relu(-correction))
        fine_scale = self.exp(-self.relu(correction))
        numerator = self.multiply(fine_out, fine_scale[..., None]) + self.multiply(coarse_out, coarse_scale[..., None])
        denominator = self.multiply(fine_norm, fine_scale) + self.multiply(coarse_norm, coarse_scale) + 1e-6
        # Finite ordinary attention has D>=1e-6. epsilon1e-30 rounds away in
        # FP32 D²>=1e-12; eps=0 is rejected by the actual BatchNorm backend.
        variance = self.product(torch.stack((denominator, denominator), -1)).flatten()
        self.normalizer.running_mean = torch.zeros_like(variance)
        self.normalizer.running_var = variance
        output = self.normalizer(numerator.reshape(1, -1, 1, dim)).reshape(batch, length, dim)
        return self.multiply(output, mask[..., None])


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        self.query = Linear(config.hidden_size, config.hidden_size)
        self.key = Linear(config.hidden_size, config.hidden_size)
        self.value = Linear(config.hidden_size, config.hidden_size)
        blocks = config.max_position_embeddings // 32
        self.core = SparseCoarseAttention(min(blocks * config.block_per_row, blocks**2))

    def forward(self, x, mask):
        batch, length, _ = x.shape
        q, k, v = (projection(x).reshape(batch, length, self.heads, self.dim).transpose(1, 2).float()
                   for projection in (self.query, self.key, self.value))
        if self.dim < 32:
            padding = torch.zeros(batch, self.heads, length, 32 - self.dim, device=x.device)
            q, k, v = (torch.cat((value, padding), -1) for value in (q, k, v))
        width = q.shape[-1]
        # This is the native supplied-mask conversion, including its behavior
        # for padded inputs. No activation is converted to an integer here.
        extended = (1.0 - mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
        native_mask = (1.0 + extended / 10000.0).squeeze().repeat(1, self.heads, 1).reshape(batch * self.heads, length).int().float()
        output = self.core(*(value.reshape(batch * self.heads, length, width) for value in (q, k, v)), native_mask)
        return output[..., :self.dim].reshape(batch, self.heads, length, self.dim).transpose(1, 2).contiguous().reshape(batch, length, -1)


class ResidualOutput(nn.Module):
    def __init__(self, in_features, config):
        super().__init__()
        self.dense = Linear(in_features, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, x, residual):
        return self.LayerNorm(self.dense(x) + residual)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = SelfAttention(config)
        self.attention.output = ResidualOutput(config.hidden_size, config)
        self.intermediate = nn.Module()
        self.intermediate.dense = Linear(config.hidden_size, config.intermediate_size)
        self.intermediate.activation = GELU()
        self.output = ResidualOutput(config.intermediate_size, config)

    def forward(self, x, mask):
        # Preserve native FP32 attention output; BF16 native/model Linear fails.
        x = self.attention.output(self.attention.self(x, mask), x)
        return self.output(self.intermediate.activation(self.intermediate.dense(x)), x)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings)[None] + 2)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            position_ids = self.position_ids[:, :input_ids.shape[1]]
        x = self.word_embeddings(input_ids) + self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(x + self.position_embeddings(position_ids))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mra = nn.Module()
        self.mra.embeddings = Embeddings(config)
        self.mra.encoder = nn.Module()
        self.mra.encoder.layer = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.mra.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.mra.embeddings(input_ids, token_type_ids, position_ids)
        for layer in self.mra.encoder.layer:
            hidden = layer(hidden, attention_mask)
        return self.lm_head(hidden)


def build_from_config(config, device, dtype):
    if (config.approx_mode != 'full' or config.initial_prior_first_n_blocks
            or config.initial_prior_diagonal_n_blocks or config.hidden_act != 'gelu'
            or getattr(config, 'is_decoder', False) or config.add_cross_attention):
        raise ValueError('Selected MRA checkpoint requires full approximation, no priors, and a GELU encoder')
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped, consumed = {}, set()
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if name.startswith('lm_head.'):
            suffix = name.removeprefix('lm_head.')
            source = 'cls.predictions.' + (suffix if suffix.startswith('decoder.') else 'transform.' + suffix)
        mapped[name] = state_dict[source]
        consumed.add(source)
    if not torch.equal(state_dict['cls.predictions.bias'], state_dict['cls.predictions.decoder.bias']):
        raise ValueError('Native MRA decoder bias aliases must agree')
    consumed.add('cls.predictions.bias')
    if config.tie_word_embeddings and not torch.equal(state_dict['cls.predictions.decoder.weight'], state_dict['mra.embeddings.word_embeddings.weight']):
        raise ValueError('Native MRA tied embedding and decoder weights must agree')
    if consumed != set(state_dict):
        raise KeyError(f'Unmapped MRA states: {sorted(set(state_dict)-consumed)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: {'logits': model(**inputs)})}
