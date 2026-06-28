from __future__ import annotations

import torch
import torch.nn.functional as F


def repeat_kv(k: torch.Tensor, target_heads: int) -> torch.Tensor:
    if k.shape[-2] == target_heads:
        return k
    if target_heads % k.shape[-2] != 0:
        raise ValueError(
            f"Cannot repeat {k.shape[-2]} KV heads to {target_heads} query heads"
        )
    return k.repeat_interleave(target_heads // k.shape[-2], dim=-2)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] | list[int] | None = (-1, -1),
    s_aux: torch.Tensor | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    window_size = (-1, -1) if window_size is None else tuple(window_size)
    q_in = q.transpose(-3, -2)
    k_in = repeat_kv(k, q.shape[-2]).transpose(-3, -2)
    v_in = repeat_kv(v, q.shape[-2]).transpose(-3, -2)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    has_backend_specific_mask = (
        window_size != (-1, -1)
        or s_aux is not None
        or softcap > 0.0
    )
    if q.is_cuda and not has_backend_specific_mask and q_in.shape[-2] == k_in.shape[-2]:
        out = torch.ops.aten._scaled_dot_product_flash_attention(
            q_in, k_in, v_in, 0.0, causal, scale=scale,
        )[0]
        return out.transpose(-3, -2)
    if (
        q.is_cuda
        and causal
        and not has_backend_specific_mask
        and q_in.shape[-2] == 1
    ):
        out = torch.ops.aten._scaled_dot_product_flash_attention(
            q_in, k_in, v_in, 0.0, False, scale=scale,
        )[0]
        return out.transpose(-3, -2)
    if causal or has_backend_specific_mask:
        q_len = q_in.shape[-2]
        k_len = k_in.shape[-2]
        left, right = window_size
        if causal:
            right = 0
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + (k_len - q_len)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        if left < 0:
            mask = k_pos <= q_pos + right
        else:
            mask = (k_pos <= torch.minimum(q_pos + right, torch.full_like(q_pos, k_len))) & (
                k_pos >= q_pos - left
            )
        scores = torch.matmul(q_in.float(), k_in.float().transpose(-2, -1)) * scale
        if softcap > 0.0:
            scores = torch.tanh(scores / softcap) * softcap
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        if s_aux is not None:
            sink = s_aux.to(device=scores.device, dtype=scores.dtype).view(1, -1, 1, 1)
            sink = sink.expand(scores.shape[0], -1, scores.shape[-2], -1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        probs = probs.masked_fill(torch.all(~mask, dim=-1, keepdim=True), 0.0)
        if s_aux is not None:
            out = torch.matmul(probs, v_in.float()).to(v_in.dtype)
        else:
            out = torch.matmul(probs.to(v_in.dtype), v_in)
        return out.transpose(-3, -2)
    out = F.scaled_dot_product_attention(
        q_in, k_in, v_in, is_causal=False, scale=scale,
    )
    return out.transpose(-3, -2)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] | list[int] | None = (-1, -1),
    s_aux: torch.Tensor | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    window_size = (-1, -1) if window_size is None else tuple(window_size)
    outputs = []
    batch = cu_seqlens_q.numel() - 1
    for i in range(batch):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
        out = dense_attention(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            s_aux=s_aux,
            softcap=softcap,
        ).squeeze(0)
        outputs.append(out)
    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0)


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor | None,
    seq_idx: int,
    seq_len: int,
    *,
    hnd: bool = False,
) -> torch.Tensor:
    if block_table is None:
        if cache.ndim == 4 and hnd:
            return cache.reshape(-1, cache.shape[1], cache.shape[-1])[:seq_len]
        if cache.ndim == 4:
            return cache.reshape(-1, cache.shape[-2], cache.shape[-1])[:seq_len]
        return cache[:seq_len]

    blocks = block_table[seq_idx]
    pieces = []
    remaining = seq_len
    for block in blocks:
        if remaining <= 0:
            break
        block_idx = int(block.item())
        if block_idx < 0:
            continue
        block_cache = cache[block_idx]
        if hnd:
            block_cache = block_cache.transpose(0, 1)
        take = min(remaining, block_cache.shape[0])
        pieces.append(block_cache[:take])
        remaining -= take
    if not pieces:
        shape = (0, cache.shape[1], cache.shape[-1]) if hnd else (0, cache.shape[-2], cache.shape[-1])
        return cache.new_empty(shape)
    return torch.cat(pieces, dim=0)
