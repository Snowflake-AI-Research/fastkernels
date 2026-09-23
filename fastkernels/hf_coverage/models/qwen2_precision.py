"""Existing Qwen operations with HF's BF16 rounding and explicit SDPA selection."""

import torch
from torch import nn

from fastkernels.infra.context import get_context, get_attn_backend_config
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.store_kvcache import StoreKVCache
from ..patches.grouped_dense_attention import GroupedDenseAttention


class SeparateQKV(nn.Module):
    """Keep packed weights but execute the three existing linear operations.

    Packing changes the GEMM shape and can change BF16 reduction rounding,
    even with identical inputs. Separate calls preserve HF's projection shapes.
    """

    def __init__(self, projection, sizes):
        super().__init__()
        self.weight, self.bias = projection.weight, projection.bias
        self.sizes, self.matmul = sizes, Matmul()

    def forward(self, hidden):
        biases = self.bias.split(self.sizes) if self.bias is not None else [None] * 3
        return torch.cat([self.matmul(hidden, weight, bias)
                          for weight, bias in zip(self.weight.split(self.sizes), biases)], dim=-1)


class ResidualRMSNorm(RMSNormNative):
    """Round the residual addition before the existing normalization operation."""

    def forward(self, hidden, residual=None):
        if residual is None:
            return super().forward(hidden)
        residual = hidden + residual
        return super().forward(residual), residual


class NativeRotaryEmbedding(RotaryEmbedding):
    def forward(self, positions, query, key):
        return self.forward_native(positions, query, key, self.head_dim,
                                   self.cos_sin_cache.to(query.dtype))


class DenseCachedAttention(nn.Module):
    """Compose cache storage and dense attention; cache gathering remains timed.

    Context lengths may stay on the CPU: they are used only for Python slicing,
    while block tables and slot mappings are consumed by GPU operations.
    """

    def __init__(self, heads, kv_heads, head_dim):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_size = heads, kv_heads, head_dim
        self._block_size = get_attn_backend_config().block_size
        self.kv_layout = "NHD"
        self.k_cache = self.v_cache = torch.tensor([])
        self.store_kvcache = StoreKVCache()
        self.attention = (DenseAttention(backend="cudnn") if heads == kv_heads else GroupedDenseAttention())

    def forward(self, query, key, value):
        context = get_context()
        query = query.reshape(-1, self.num_heads, self.head_size)
        key, value = (tensor.reshape(-1, self.num_kv_heads, self.head_size) for tensor in (key, value))
        self.store_kvcache(key, value, self.k_cache, self.v_cache, context.slot_mapping)
        outputs = []
        batch = context.block_tables.shape[0]
        for index in range(batch):
            if context.is_prefill:
                # A single prompt occupies the entire input. Avoid reading GPU
                # offsets back to the host in every attention layer.
                if batch == 1:
                    start, end = 0, query.shape[0]
                else:
                    start, end = int(context.cu_seqlens_q[index]), int(context.cu_seqlens_q[index + 1])
                q, k, v = query[start:end], key[start:end], value[start:end]
            else:
                length = int(context.context_lens[index])
                pages = context.block_tables[index, :(length + self._block_size - 1) // self._block_size].long()
                q = query[index:index + 1]
                k, v = (cache[pages].reshape(-1, self.num_kv_heads, self.head_size)[:length]
                        for cache in (self.k_cache, self.v_cache))
            outputs.append(self.attention(q[None], k[None], v[None], causal=context.is_prefill)[0])
        return torch.cat(outputs).reshape(-1, self.num_heads * self.head_size)


def configure_language(model, config):
    for layer in model.layers:
        layer.input_layernorm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
        layer.post_attention_layernorm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
        layer.self_attn.attn = DenseCachedAttention(config.num_attention_heads, config.num_key_value_heads,
                                                   config.head_dim)
    model.norm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
