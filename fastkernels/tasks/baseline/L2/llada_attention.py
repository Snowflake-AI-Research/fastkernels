"""LLaDA bidirectional self-attention with optional Fast-dLLM cache support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_linear import ReplicatedLinear


class LLaDAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rotary_emb: nn.Module | None = None,
        bias: bool = False,
        rope_full_precision: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.rope_full_precision = rope_full_precision

        self.q_proj = ReplicatedLinear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = ReplicatedLinear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = ReplicatedLinear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.attn_out = ReplicatedLinear(num_attention_heads * head_dim, hidden_size, bias=bias)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.view(batch_size, num_heads, seq_len, 2, head_dim // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope_to_states(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.rotary_emb is None:
            return x

        if self.rope_full_precision and hasattr(self.rotary_emb, "llada_pos_cos_cache"):
            index = positions[0] if positions.dim() == 2 else positions
            pos_cos = self.rotary_emb.llada_pos_cos_cache.index_select(2, index)
            pos_sin = self.rotary_emb.llada_pos_sin_cache.index_select(2, index)
            x_work = x.float()
            return ((x_work * pos_cos) + (self._rotate_half(x_work) * pos_sin)).to(x.dtype)

        compute_dtype = x.dtype
        cache = self.rotary_emb.cos_sin_cache
        if cache.dtype != compute_dtype:
            cache = cache.to(compute_dtype)
        flat = positions.reshape(-1)
        rope = cache.index_select(0, flat).view(*positions.shape, -1)
        half = self.head_dim // 2
        cos_half = rope[..., :half]
        sin_half = rope[..., half:]
        cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
        sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
        x_work = x.to(compute_dtype)
        return ((x_work * cos) + (self._rotate_half(x_work) * sin)).to(x.dtype)

    def _rotary_positions(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        replace_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_positions = torch.arange(key_len, device=device, dtype=torch.long)
        if replace_position is None:
            query_positions = torch.arange(key_len - query_len, key_len, device=device, dtype=torch.long)
        else:
            if replace_position.any():
                block_end_index = replace_position.nonzero(as_tuple=True)[1].max() + 1
            else:
                block_end_index = torch.tensor(key_len, device=device, dtype=torch.long)
            query_positions = torch.arange(
                int(block_end_index.item()) - query_len,
                int(block_end_index.item()),
                device=device,
                dtype=torch.long,
            )
        return query_positions.unsqueeze(0), key_positions.unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        replace_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        new_k = k
        new_v = v

        if layer_past is not None:
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=2)
                v = torch.cat((past_value, v), dim=2)
            else:
                k = past_key.clone()
                v = past_value.clone()
                for batch_idx in range(replace_position.shape[0]):
                    batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
                    if batch_replace_indices.numel() > 0:
                        count = batch_replace_indices.numel()
                        k[batch_idx, :, batch_replace_indices] = new_k[batch_idx, :, :count]
                        v[batch_idx, :, batch_replace_indices] = new_v[batch_idx, :, :count]

        present = (k, v) if use_cache else None

        query_positions, key_positions = self._rotary_positions(
            query_len=q.shape[-2],
            key_len=k.shape[-2],
            device=hidden_states.device,
            replace_position=replace_position,
        )
        q = self._apply_rope_to_states(q, query_positions)
        k = self._apply_rope_to_states(k, key_positions)

        if self.num_attention_heads != self.num_key_value_heads:
            assert self.num_attention_heads % self.num_key_value_heads == 0
            repeat = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if attention_bias is not None:
            key_len = k.shape[-2]
            query_len = q.shape[-2]
            attention_bias = attention_bias[:, :, key_len - query_len : key_len, :key_len]
            attention_bias = attention_bias.to(dtype=q.dtype)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.attn_out(attn), present
