"""TTT-E2E sliding-window attention (L2).

Mirrors the JAX/Equinox reference at
``github.com/test-time-training/e2e:ttt/model/attention.py``:

  - 4 dense projections wq, wk, wv, wo (no bias, hidden -> hidden)
  - per-head Q/K RMSNorm on the head_dim axis (qk_norm=True in the paper)
  - interleaved RoPE with LOCAL window-relative positions on the suffix
    path and GLOBAL token positions on the prefix path
  - sliding-window causal attention with window W (paper default: 8192)
  - prefix path: full sequence at once, no KV cache update
  - suffix path: chunked, attends each chunk over a (W,)-long rolling cache

This file is composed entirely of L1 ops (no torch.nn / torch.nn.functional
calls). It uses:
  - L1 :class:`Linear` for the four projections
  - L1 :class:`RMSNormNative` for q/k norm (autograd-friendly + correct on
    odd head_dim values where the L1 RMSNorm CUDA kernel is broken)
  - L1 :class:`TTTE2ERoPE` for interleaved RoPE
  - L1 :class:`DenseAttention` for the SDPA call (handles both prefix
    and suffix paths via the ``attn_mask`` argument; the kb-nano CUDA
    flash-attention L1 ops are paged-KV-cache-only and don't fit either
    of TTT-E2E's access patterns).
"""

from __future__ import annotations

import torch
from torch import nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.rms_norm_native import RMSNormNative
from ..L1.ttt_e2e_rope import TTTE2ERoPE


class TTTE2ESWA(nn.Module):
    """Sliding-window attention with prefix and suffix-chunked KV-cache paths.

    Args:
        hidden_size: model dim
        num_heads:   number of attention heads (head_dim = hidden_size // num_heads)
        window_size: sliding window size W (paper: 8192)
        chunk_size:  mini-batch tokens per inner-loop step (paper: 1024)
        max_position_embeddings: how many RoPE positions to precompute (must be
            >= max(seq_len, W + chunk_size))
        rope_theta: RoPE theta (paper: 500000)
        qk_norm: apply per-head RMSNorm to Q and K (paper: True)
        rms_norm_eps: eps for the q/k RMSNorms
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        chunk_size: int,
        max_position_embeddings: int,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        attention_backend: str = "cudnn",
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.qk_norm = qk_norm
        self.attention_backend = attention_backend

        self.wq = Linear(hidden_size, hidden_size, bias=False)
        self.wk = Linear(hidden_size, hidden_size, bias=False)
        self.wv = Linear(hidden_size, hidden_size, bias=False)
        self.wo = Linear(hidden_size, hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNormNative(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNormNative(self.head_dim, eps=rms_norm_eps)

        rope_len = max(max_position_embeddings, window_size + chunk_size)
        self.rope = TTTE2ERoPE(self.head_dim, rope_len, rope_theta=rope_theta)

        # Two attention dispatchers, one per path:
        # - prefix path: causal-no-mask. ``auto`` lets PyTorch pick the
        #   fastest path (cuDNN flash on Blackwell). cuDNN wins on long
        #   sequences (8K prefix), where FlexAttention's Triton kernel is
        #   slower than cuDNN's hand-tuned flash binary.
        # - suffix path: chunked sliding-window-causal with explicit mask.
        #   Two backend choices:
        #     "cudnn"  — pin cuDNN flash via sdpa_kernel. Works with our
        #                tensor-chunk_id graph-capture path. Default.
        #     "flex"   — FlexAttention + Triton-fused fwd+bwd autotuned
        #                for the (Q=1024, KV=9216, D=64) shape. ~1.37x
        #                faster than cuDNN at this exact chunked shape
        #                (microbenched). Requires int chunk_id (BlockMask
        #                lookup), so the engine threads ints in meta mode
        #                rather than the in-place tensor buffer trick.
        self.attn_prefix = DenseAttention(backend="auto")
        self.attn_suffix = DenseAttention(backend=attention_backend)

        # Cache for the chunk-suffix attention masks (dense bool) and
        # FlexAttention BlockMasks, plus RoPE position vectors. Keyed by
        # ``(chunk_id, device)`` for masks (chunk_id determines early-cache
        # padding via ``ki >= 0``) and by ``device`` for positions (chunk-
        # invariant). Lazily materialized on first call.
        self._suffix_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self._suffix_block_mask_cache: dict[tuple[int, torch.device], object] = {}
        self._suffix_q_pos: dict[torch.device, torch.Tensor] = {}
        self._suffix_k_pos: dict[torch.device, torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to (xq, xk, xv) of shape (B, T, H, D)."""
        b, t, _ = x.shape
        xq = self.wq(x).view(b, t, self.num_heads, self.head_dim)
        xk = self.wk(x).view(b, t, self.num_heads, self.head_dim)
        xv = self.wv(x).view(b, t, self.num_heads, self.head_dim)
        return xq, xk, xv

    def _qk_norm(self, xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.qk_norm:
            return xq, xk
        # RMSNormNative handles per-head_dim normalization on (B, T, H, D)
        # directly — the last-dim broadcast is correct and dtype-promoting.
        return self.q_norm(xq), self.k_norm(xk)

    # --------------------------------------------------------------- prefix path

    def forward_prefix(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Full-sequence sliding-window attention. No KV cache update.

        Args:
            x: (B, T, hidden)
            position_ids: (T,) int positions for RoPE. If None, uses 0..T-1.

        Two attention paths:
        - ``T <= self.window_size`` (no sliding-window pruning needed): we
          dispatch to the L1 ``DenseAttention`` with ``is_causal=True`` and
          no ``attn_mask``. SDPA picks the fast cuDNN/flash backend
          internally. Materializing a (T, T) bool mask just to encode pure
          causal would force the kernel into the ``mem_efficient`` fallback;
          on a 8192-token prefix that costs us ~10× over the flash path
          (verified empirically vs the JAX reference).
        - ``T  > self.window_size``: we DO need to mask out keys that are
          more than W-1 positions back. We build a windowed-causal mask
          and pay the slower path. Flash-attn-with-window would help here
          too, but flash isn't installed on this venv (Blackwell pre-FA3).
        """
        b, t, _ = x.shape
        xq, xk, xv = self._project_qkv(x)
        xq, xk = self._qk_norm(xq, xk)

        if position_ids is None:
            position_ids = torch.arange(t, device=x.device)
        xq = self.rope(xq, position_ids)
        xk = self.rope(xk, position_ids)

        if t <= self.window_size:
            # Pure causal — no mask materialization. SDPA picks flash.
            out = self.attn_prefix(xq, xk, xv, softmax_scale=self.head_dim ** -0.5,
                                   causal=True, attn_mask=None)
        else:
            # Sliding causal: position i attends to j iff j <= i and i - j < W.
            # Force cuDNN here since this path needs a mask.
            i = torch.arange(t, device=x.device).unsqueeze(1)
            j = torch.arange(t, device=x.device).unsqueeze(0)
            attn_mask = (j <= i) & (i - j < self.window_size)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            out = self.attn_suffix(xq, xk, xv, softmax_scale=self.head_dim ** -0.5,
                                   causal=False, attn_mask=attn_mask)
        out = out.reshape(b, t, self.hidden_size)
        return self.wo(out)

    # --------------------------------------------------------------- suffix path

    def init_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.zeros(batch_size, self.window_size, self.num_heads, self.head_dim, dtype=dtype, device=device)
        v = torch.zeros_like(k)
        return k, v

    def forward_suffix_chunk(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        chunk_id: int | torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Chunked sliding-window attention with rolling KV cache.

        Args:
            x: (B, C, hidden) where C == chunk_size
            kv_cache: (k_cache, v_cache) each of shape (B, W, H, D),
                holding POST-qk-norm, PRE-RoPE k/v from prior chunks
                (matches the JAX reference's storage layout).
            chunk_id: 0-indexed chunk number within the current sequence;
                used to mask out cache slots that haven't been written yet.
                May be a Python int OR a 0-dim int64 tensor — tensor form
                lets torch.compile produce a single graph that handles all
                chunk_ids dynamically (instead of one specialized trace per
                int value, which used to hit the dynamo recompile_limit).
        Returns:
            out:        (B, C, hidden)
            new_kv_cache: (k_cache', v_cache') — last W (post-qk-norm,
                pre-RoPE) k/v values from cat(cache, new_chunk).
        """
        b, c, _ = x.shape
        assert c == self.chunk_size, f"chunk size mismatch: got {c}, expected {self.chunk_size}"
        k_cache, v_cache = kv_cache
        W = self.window_size
        C = self.chunk_size
        device = x.device

        xq, xk, xv = self._project_qkv(x)
        xq, xk = self._qk_norm(xq, xk)

        # Cache stores POST-qk-norm but PRE-RoPE k/v. Concat with new chunk's
        # (post-qk-norm, pre-RoPE) k/v, then RoPE the cat'd tensor at LOCAL
        # window positions [0..W+C-1] and the new q at [W..W+C-1].
        full_k_pre = torch.cat([k_cache, xk], dim=1)          # (B, W+C, H, D)
        full_v = torch.cat([v_cache, xv], dim=1)

        q_pos = self._suffix_q_pos.get(device)
        if q_pos is None:
            q_pos = torch.arange(W, W + C, device=device)
            self._suffix_q_pos[device] = q_pos
        k_pos = self._suffix_k_pos.get(device)
        if k_pos is None:
            k_pos = torch.arange(W + C, device=device)
            self._suffix_k_pos[device] = k_pos
        xq = self.rope(xq, q_pos)
        full_k = self.rope(full_k_pre, k_pos)

        # JAX sw_causal_mask: query index in (chunk_id*C, chunk_id*C + C),
        # key index in (chunk_id*C - W, chunk_id*C + C - 1). Constraints:
        #   qi >= ki, qi < ki + W, ki >= 0.
        # In flex-coordinate frame (q_idx in [0,C), kv_idx in [0,W+C)):
        #   qi = q_idx + cid * C
        #   ki = kv_idx - W + cid * C
        # which simplify the constraints to depend only on (q_idx, kv_idx)
        # plus the cid-shift in ``ki >= 0``.
        if self.attention_backend == "flex":
            # FlexAttention path: build a BlockMask per chunk_id (cached).
            # Tensor chunk_id is not supported here — caller must pass int
            # so we can index the cache. Engine should use int in meta mode
            # when this backend is selected.
            if isinstance(chunk_id, torch.Tensor):
                chunk_id = int(chunk_id.item())
            from torch.nn.attention.flex_attention import create_block_mask
            cache_key = (chunk_id, device)
            attn_mask = self._suffix_block_mask_cache.get(cache_key)
            if attn_mask is None:
                cid = chunk_id
                def mask_mod(b, h, q_idx, kv_idx, _cid=cid):
                    qi = q_idx + _cid * C
                    ki = kv_idx - W + _cid * C
                    return (qi >= ki) & (qi < ki + W) & (ki >= 0)
                attn_mask = create_block_mask(
                    mask_mod, B=None, H=None, Q_LEN=C, KV_LEN=W + C,
                    device=str(device).split(":")[0],
                )
                self._suffix_block_mask_cache[cache_key] = attn_mask
        else:
            # Dense bool mask path (cuDNN / efficient / sdpa). Supports both
            # int chunk_id (cached) and tensor chunk_id (rebuilt every call,
            # one torch.compile graph for all chunk_ids).
            if isinstance(chunk_id, torch.Tensor):
                cid = chunk_id.to(device=device)
                qi = torch.arange(C, device=device).unsqueeze(1) + cid * C
                ki = torch.arange(-(W + C), 0, device=device).unsqueeze(0) + (cid + 1) * C
                mask = (qi >= ki) & (qi < ki + W) & (ki >= 0)
                attn_mask = mask.unsqueeze(0).unsqueeze(0)
            else:
                cache_key = (chunk_id, device)
                attn_mask = self._suffix_mask_cache.get(cache_key)
                if attn_mask is None:
                    starting_q = chunk_id * C
                    qi = (torch.arange(C, device=device) + starting_q).unsqueeze(1)
                    ki = (torch.arange(-(W + C), 0, device=device) + (starting_q + C)).unsqueeze(0)
                    mask = (qi >= ki) & (qi < ki + W) & (ki >= 0)
                    attn_mask = mask.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, C, W+C)
                    self._suffix_mask_cache[cache_key] = attn_mask

        out = self.attn_suffix(xq, full_k, full_v,
                               softmax_scale=self.head_dim ** -0.5,
                               causal=False, attn_mask=attn_mask)
        out = out.reshape(b, C, self.hidden_size)
        out = self.wo(out)

        new_k = full_k_pre[:, -W:].contiguous()
        new_v = full_v[:, -W:].contiguous()
        return out, (new_k, new_v)
