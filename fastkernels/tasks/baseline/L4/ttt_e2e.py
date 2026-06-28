"""TTT-E2E pipeline (L4).

Wires the L3 decoder + the chunk-by-chunk inner-loop SGD on the prime FFN
weights of the suffix blocks. Mirrors ``MetaModel.loss_for_sequence`` from
``ttt/model/transformer.py`` for ``train_mode in {"pretrain", "meta"}``:

  pretrain:
    1. embed input ids
    2. run prefix layers full-seq (sliding-window attention, no cache)
    3. for each chunk of ``mini_batch_size`` tokens:
         run suffix layers chunk-by-chunk with rolling KV cache;
         no parameter update.
    4. project to logits, return per-token NLL.

  meta (the actual TTT-E2E inference path):
    same as pretrain, but after each suffix-chunk forward, compute the CE
    loss on the chunk and take an SGD step that updates ONLY the prime FFN
    weights of each suffix block. The optimizer is
    ``optax.chain(clip_by_global_norm(1.0), sgd(lr=ilr_init * inner_lr))``,
    matching the reference exactly. Updated prime weights persist across
    chunks within a sequence; they reset at the start of the next sequence.

The inner-loop gradient is computed via ``torch.func.grad`` over a function
that takes a flat dict of prime parameters and returns the chunk loss. We
deliberately do NOT use autograd's ``backward()`` on Module parameters;
that would either pollute ``requires_grad`` state or require explicit zero-
grad / detach gymnastics. ``torch.func.grad`` is purely functional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ..L1.softmax import LogSoftmax
from ..L3.ttt_e2e_decoder import TTTE2EDecoder


def _token_nll(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token negative log-likelihood for ``targets`` given ``log_probs``.

    log_probs: (B, T, V) fp32 log-softmax of logits.
    targets:   (B, T) int64 token ids.
    Returns:   (B, T) fp32 NLL.
    """
    B, T, _ = log_probs.shape
    return -log_probs.gather(-1, targets.view(B, T, 1)).squeeze(-1)


@dataclass
class TTTE2EConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    chunk_size: int                     # mini_batch_size in the reference
    sliding_window_size: int
    suffix_len: int
    max_position_embeddings: int        # ref calls this seq_len
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_word_embeddings: bool = True
    has_prime: bool = True

    # Inner-loop SGD knobs (reference: optimizer_inner)
    inner_lr: float = 1.0               # peak inner LR
    inner_clip_grad_norm: float = 1.0
    ilr_init: float = 1.0               # post-warmup ilr multiplier
    # We always treat ourselves as "post-warmup" at inference (ilr=ilr_init).

    dtype: torch.dtype = torch.bfloat16  # compute dtype, like JAX compute_dtype="bf16"
    param_dtype: torch.dtype = torch.float32

    # Attention backend used by the chunked-suffix path (the prefix path
    # always uses cuDNN flash via DenseAttention's "auto" backend, where
    # cuDNN wins on the long-seq shape). Choices:
    #   "cudnn" (default) — F.scaled_dot_product_attention pinned to the
    #       cuDNN flash backend with explicit dense bool mask. Compatible
    #       with the in-place-tensor chunk_id graph-capture path.
    #   "flex"           — torch.nn.attention.flex_attention with a
    #       per-chunk_id BlockMask. Generates a fused Triton fwd+bwd
    #       autotuned for the (Q=chunk_size, KV=W+chunk_size, head_dim)
    #       shape. Microbenched ~1.37x faster than cuDNN on
    #       (1024, 9216, 64) on B200. Requires int chunk_id (BlockMask
    #       lookup), so the engine threads ints in meta mode rather than
    #       the in-place tensor buffer trick.
    attention_backend: str = "cudnn"

    @classmethod
    def from_jax_dict(cls, m: dict, t: dict) -> "TTTE2EConfig":
        return cls(
            vocab_size=m["vocab_size"],
            hidden_size=m["hidden_size"],
            intermediate_size=m["intermediate_size"],
            num_hidden_layers=m["num_hidden_layers"],
            num_attention_heads=m["num_attention_heads"],
            chunk_size=m["mini_batch_size"],
            sliding_window_size=m["sliding_window_size"],
            suffix_len=m["suffix_len"],
            max_position_embeddings=m["seq_len"],
            rope_theta=float(m["rope_theta"]),
            rms_norm_eps=float(m["rms_norm_eps"]),
            qk_norm=bool(m.get("qk_norm", True)),
            tie_word_embeddings=bool(m["tie_word_embeddings"]),
            has_prime=bool(m["prime"]),
            inner_lr=float(t.get("inner_lr", 1.0)),
            inner_clip_grad_norm=float(t.get("inner_clip", 1.0)),
            ilr_init=float(t.get("ilr_init", 1.0)),
        )


@dataclass
class TTTE2EOutput:
    logits: torch.Tensor | None           # (B, T, vocab_size) — None unless
                                           # explicitly materialized; the bench
                                           # path skips logits to save memory
                                           # and a wte^T projection per chunk.
    token_nll: torch.Tensor               # (B, T) per-token NLL of input_ids
    chunk_losses: list[torch.Tensor] = field(default_factory=list)


class TTTE2EPipeline(nn.Module):
    """End-to-end forward with optional inner-loop SGD on prime FFN params."""

    def __init__(self, config: TTTE2EConfig):
        super().__init__()
        self.config = config
        self.model = TTTE2EDecoder(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            window_size=config.sliding_window_size,
            chunk_size=config.chunk_size,
            suffix_len=config.suffix_len,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            attention_backend=config.attention_backend,
        )
        self.log_softmax = LogSoftmax(dim=-1)

    # ------------------------------------------------------------- helpers

    def _init_prime_state(self) -> list[dict[str, torch.Tensor]]:
        """Snapshot the meta-trained prime weights as the inner-loop init.

        Returns one dict per suffix layer: ``{"w1": (I, H), "w2": (H, I), "w3": (I, H)}``,
        where shapes follow PyTorch's ``Linear`` (``out, in``). Tensors are
        cloned in fp32 so the inner-loop SGD is numerically identical to the
        JAX reference (which keeps params in fp32 and computes in bf16).
        """
        prime_state = []
        for layer in self.model.suffix_layers:
            assert layer.has_prime
            w1 = layer.feed_forward_prime.w1.weight.detach().clone().to(self.config.param_dtype)
            w2 = layer.feed_forward_prime.w2.weight.detach().clone().to(self.config.param_dtype)
            w3 = layer.feed_forward_prime.w3.weight.detach().clone().to(self.config.param_dtype)
            prime_state.append({"w1": w1, "w2": w2, "w3": w3})
        return prime_state

    def _init_kv_caches(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """One (k,v) cache per suffix layer."""
        return [
            layer.seq_modeling_block.init_kv_cache(batch_size, dtype, device)
            for layer in self.model.suffix_layers
        ]

    # ----------------------------------------------------- chunk-level forward

    def _suffix_chunk_forward(
        self,
        prefix_chunk: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        chunk_id: int,
        prime_state: list[dict[str, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through suffix layers for a single chunk.

        Returns the final-norm hidden states for the chunk and updated KV
        caches. ``prime_state`` is used (instead of the stored prime params)
        when provided. When running for an inner-loop gradient, the final
        ln_f also dispatches to a gradient-friendly pure-PyTorch RMSNorm.
        """
        h = prefix_chunk
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.model.suffix_layers):
            override = None
            if prime_state is not None:
                override = prime_state[li]
            h, new_cache = layer.forward_suffix_chunk(h, kv_caches[li], chunk_id, prime_override=override)
            new_caches.append(new_cache)
        # ln_f is RMSNormNative (autograd-friendly), so the same call works for
        # both the loss-bearing forward inside torch.func.grad and the
        # ordinary no-grad forward.
        h = self.model.ln_f(h)
        return h, new_caches

    # ------------------------------------------------------ inner-loop helper

    def _inner_loop_step(
        self,
        prefix_chunk: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        chunk_id: int,
        target_chunk: torch.Tensor,
        prime_state: list[dict[str, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,                                            # per-token NLL for the chunk (B, C)
        list[tuple[torch.Tensor, torch.Tensor]],                  # new kv caches (post-chunk)
        list[dict[str, torch.Tensor]],                            # updated prime params
    ]:
        """One inner SGD step on prime params.

        Mirrors the JAX reference's ``inner_loop_step`` exactly:
        - ``torch.func.grad(loss_fn, has_aux=True)`` returns the grads PLUS
          the auxiliary outputs (per-token NLL and new caches) in a single
          forward+backward pass. The JAX side does the same with
          ``eqx.filter_value_and_grad(..., has_aux=True)``.
        - Grads are computed in fp32 (prime params live in fp32). The
          forward casts to compute_dtype (bf16) at each functional_call
          boundary via differentiable ``.to()``, matching JAX's
          ``promote_dtype(x, weight, dtype=compute_dtype)`` semantics.
        - Updated prime params persist into the next chunk; KV caches from
          the SAME loss-bearing forward are returned unchanged. The post-
          update prime params take effect on the NEXT chunk.
        """
        cfg = self.config

        # Build leaf tensors that require grad — fresh per chunk so we don't
        # accumulate grads across the inner loop. ``torch.autograd.grad``
        # gives us functional gradients (no .grad attribute pollution) and
        # plays nicely with CUDA Graph capture later.
        leaves: list[dict[str, torch.Tensor]] = [
            {k: v.detach().requires_grad_(True) for k, v in d.items()}
            for d in prime_state
        ]
        flat_leaves = [v for d in leaves for v in d.values()]      # in deterministic order

        h, new_caches = self._suffix_chunk_forward(
            prefix_chunk, kv_caches, chunk_id, prime_state=leaves,
        )
        logits = self.model.project_logits(h)
        log_probs = self.log_softmax(logits.float())
        chunk_nll = _token_nll(log_probs, target_chunk)            # (B, C)
        loss = chunk_nll.mean()

        flat_grads = torch.autograd.grad(loss, flat_leaves, create_graph=False, retain_graph=False)

        # Global-norm clip (matches optax.clip_by_global_norm).
        global_norm = torch.sqrt(sum(g.float().pow(2).sum() for g in flat_grads))
        scale = (cfg.inner_clip_grad_norm / (global_norm + 1e-9)).clamp(max=1.0)

        # SGD step: param -= lr * (scale * grad). Build new prime_state in
        # the same list-of-dicts shape, detached so the next chunk's grads
        # don't backprop through this chunk's update.
        lr = cfg.inner_lr * cfg.ilr_init
        new_prime_state: list[dict[str, torch.Tensor]] = []
        gi = 0
        for d in leaves:
            new_d: dict[str, torch.Tensor] = {}
            for k, p in d.items():
                new_d[k] = (p.detach() - lr * (scale * flat_grads[gi])).detach()
                gi += 1
            new_prime_state.append(new_d)

        # Detach caches and chunk_nll so downstream Python state is grad-free.
        new_caches = [(k.detach(), v.detach()) for k, v in new_caches]
        return chunk_nll.detach(), new_caches, new_prime_state

    # ------------------------------------------------------------------ entry

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T) int64
        target_tokens: torch.Tensor | None = None,   # (B, T) int64; if None, shift input_ids by 1 (rolled)
        train_mode: str = "meta",
    ) -> TTTE2EOutput:
        """End-to-end forward.

        train_mode:
            "pretrain"  — no inner loop (prime FFN frozen at meta-trained weights).
            "meta"      — inner loop active: SGD-update prime params per chunk.
        """
        cfg = self.config
        B, T = input_ids.shape
        device = input_ids.device
        assert T % cfg.chunk_size == 0, f"seqlen {T} must divide chunk_size {cfg.chunk_size}"
        n_chunks = T // cfg.chunk_size

        if target_tokens is None:
            target_tokens = torch.cat([input_ids[..., 1:], input_ids[..., :1]], dim=-1)

        x = self.model.embed(input_ids).to(cfg.dtype)

        # ---- Prefix: run all prefix layers full-seq with sliding-window attn.
        position_ids = torch.arange(T, device=device)
        for layer in self.model.prefix_layers:
            x = layer.forward_prefix(x, position_ids=position_ids)
        prefix_output = x  # (B, T, hidden), bf16

        # ---- Suffix: chunked. Active prime state evolves chunk-to-chunk in meta mode.
        prime_state = self._init_prime_state() if (cfg.has_prime and train_mode == "meta") else None
        kv_caches = self._init_kv_caches(B, cfg.dtype, device)

        chunk_nlls: list[torch.Tensor] = []
        chunk_losses: list[torch.Tensor] = []
        # Use a 0-dim tensor for chunk_id when meta-mode is active AND the
        # attention backend supports it. This lets ``torch.compile`` produce
        # a single graph that handles every chunk_id (otherwise the int form
        # trips dynamo specialization and we trace once per chunk_id, hitting
        # the recompile limit at long sequences).
        # The flex backend looks BlockMasks up by int chunk_id (a tensor key
        # would force a host-sync inside graph-capture territory), so for
        # that backend we accept per-chunk_id specialization. For our
        # workload (8K seq, chunk=1024 → 8 chunks), 8 specialized compiles
        # is well within recompile_limit=64.
        # pretrain mode keeps the int form so the cached mask path is hit
        # (no compile in pretrain).
        # NB: the persistent buffer ``self._chunk_id_buf`` is hoisted out of
        # the per-forward path so callers wrapping ``forward`` in a CUDA
        # Graph capture don't allocate during stream capture (which CUDA
        # forbids).
        meta_active = train_mode == "meta" and cfg.has_prime
        use_tensor_chunk_id = meta_active and cfg.attention_backend != "flex"
        if use_tensor_chunk_id:
            buf = getattr(self, "_chunk_id_buf", None)
            if buf is None or buf.device != device:
                buf = torch.zeros((), dtype=torch.int64, device=device)
                self._chunk_id_buf = buf
            chunk_id_t = buf
        else:
            chunk_id_t = None

        # Static contiguous chunk buffers, hoisted out of the per-forward path.
        # The naive ``prefix_chunk = prefix_output[:, s:e]`` returns a view
        # whose ``stride[0]`` is the parent's full-T stride (8192*H), but
        # most downstream ops produce contig outputs (stride[0] = chunk*H).
        # Mixing the two strides in the same compiled function trips
        # torch._dynamo's stride guards and causes recompile-storm in any
        # path that doesn't run inside a CUDA Graph capture (e.g. pretrain
        # mode, where capture is not used). Copying into a static contig
        # buffer makes ``prefix_chunk`` always have stride (C*H, H, 1),
        # matching the contig outputs of intermediate Linear layers, so all
        # 8 chunk specializations share a single dynamo cache entry.
        # The buffer is allocated once and reused — safe inside CUDA Graph
        # capture too.
        chunk_buf = getattr(self, "_chunk_in_buf", None)
        chunk_shape = (B, cfg.chunk_size, cfg.hidden_size)
        if chunk_buf is None or tuple(chunk_buf.shape) != chunk_shape or chunk_buf.dtype != cfg.dtype or chunk_buf.device != device:
            chunk_buf = torch.empty(chunk_shape, dtype=cfg.dtype, device=device)
            self._chunk_in_buf = chunk_buf
        target_buf = getattr(self, "_target_chunk_buf", None)
        target_shape = (B, cfg.chunk_size)
        if target_buf is None or tuple(target_buf.shape) != target_shape or target_buf.device != device:
            target_buf = torch.empty(target_shape, dtype=target_tokens.dtype, device=device)
            self._target_chunk_buf = target_buf

        for ci in range(n_chunks):
            s = ci * cfg.chunk_size
            e = s + cfg.chunk_size
            chunk_buf.copy_(prefix_output[:, s:e])
            target_buf.copy_(target_tokens[:, s:e])
            prefix_chunk = chunk_buf
            target_chunk = target_buf
            if meta_active:
                if use_tensor_chunk_id:
                    chunk_id_t.fill_(ci)
                    chunk_id_arg = chunk_id_t
                else:
                    chunk_id_arg = ci
                with torch.enable_grad():
                    chunk_nll, kv_caches, prime_state = self._inner_loop_step(
                        prefix_chunk, kv_caches, chunk_id_arg, target_chunk, prime_state,
                    )
            else:
                # Plain forward — use stored prime params directly. Matches
                # JAX ``train_mode="pretrain"`` where prime params are frozen.
                h, kv_caches = self._suffix_chunk_forward(
                    prefix_chunk, kv_caches, ci, prime_state=None,
                )
                logits_c = self.model.project_logits(h)
                log_probs = self.log_softmax(logits_c.float())
                chunk_nll = _token_nll(log_probs, target_chunk)
            chunk_nlls.append(chunk_nll)
            chunk_losses.append(chunk_nll.mean())

        token_nll = torch.cat(chunk_nlls, dim=1)               # (B, T)

        return TTTE2EOutput(logits=None, token_nll=token_nll, chunk_losses=chunk_losses)
