"""Per-layer recurrent state cache for FLA-family models.

Recurrent linear-attention models (GLA / RetNet / RWKV7) carry a
per-sequence state matrix instead of a paged KV cache. This thin
container is just a typed dict-of-dicts that lets each L2 attention
module key by ``id(self)`` and stash either:

  - ``states[layer_id]``   — the recurrent state tensor (KV running sum)
  - ``conv_states[layer_id]``  — RWKV7 token-shift carry (last hidden vec)

The L2 attention modules read/write these fields directly so we don't
have to thread a per-layer cache argument through every forward.

Mirrors the surface area of FLA's ``Cache`` (``recurrent_state`` /
``conv_state``) just enough for our use cases — full Cache semantics
(seen-tokens counters, layer-iteration helpers) live in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecurrentCache:
    """Holds per-layer recurrent / convolutional state.

    The keys are ``id(layer)`` so each L2 attention module can find its
    own slot without needing to know its layer index.

    ``seq_offsets`` (int or ``[B]`` int64 tensor) gives the global token
    position of the FIRST token in the current call, per batch row. This
    is consumed by RoPE-bearing layers (RetNet) so positional encoding
    spans cached prefill chunks and decode steps consistently. ``None``
    means start from position 0 (uncached single-shot forward).
    """

    states: dict[int, Any] = field(default_factory=dict)
    conv_states: dict[int, Any] = field(default_factory=dict)
    seq_offsets: Any = None

    def detach_(self) -> "RecurrentCache":
        """In-place detach all stored state tensors (post-step cleanup)."""
        for d in (self.states, self.conv_states):
            for k, v in list(d.items()):
                if v is not None and hasattr(v, "detach"):
                    d[k] = v.detach()
        return self


@dataclass
class CausalLMOutputWithPast:
    """Drop-in subset of HF's ``CausalLMOutputWithPast``.

    Only the fields we actually consume — keeps L4 free of an HF runtime
    dependency while remaining structurally compatible with FLA's SOTA
    forward return type.
    """

    logits: Any
    past_key_values: RecurrentCache | None = None
    loss: Any | None = None
    hidden_states: Any | None = None
