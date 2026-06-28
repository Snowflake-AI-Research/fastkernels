"""RWKV7 attention layer.

Implements the full RWKV7 attention block:
  token_shift -> addcmul -> projections + LoRA gates -> L2-norm + k_update
  -> DPLR recurrence -> GroupNorm -> gate output correction -> o_proj

Weight names match the FLA checkpoint format exactly:
  x_{r,w,k,v,a,g}, k_k, k_a, r_k
  {r,k,v,o}_proj.weight
  {w,a,g}_lora.lora.{0,2}.{weight,bias}
  v_lora.lora.{0,2}.{weight,bias}  (layers > 0)
  g_norm.{weight,bias}

The forward signature matches FLA's ``RWKV7Attention.forward`` for SOTA
parity (returns ``(output, attentions, past_key_values, v_first)``).

``nn.Sequential`` is used as a pure container to preserve the FLA
checkpoint key format ``<lora>.lora.{0,2}.{weight,bias}``; both children
are L1 ``Linear`` ops with an L1 activation between them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.chunk_rwkv7 import ChunkRWKV7
from ..L1.fused_recurrent_rwkv7 import FusedRecurrentRWKV7
from ..L1.group_norm import GroupNorm
from ..L1.l2_norm import L2Norm
from ..L1.linear import Linear
from ..L1.rwkv7_recurrence import NaiveRecurrentRWKV7
from ..L1.sigmoid import Sigmoid
from ..L1.tanh import Tanh

_CHUNK_THRESHOLD = 64


class LoRA(nn.Module):
    """Low-rank adapter: Linear -> activation -> Linear.

    Mirrors FLA's ``LoRA`` so the checkpoint keys
    ``<adapter>.lora.0.weight`` / ``<adapter>.lora.2.{weight,bias}`` map
    directly. Built from L1 ``Linear`` and L1 activation ops; the
    ``nn.Sequential`` is just a container.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        activation: str | None = "tanh",
        bias: bool = True,
    ):
        super().__init__()
        if activation is None:
            act = nn.Identity()
        elif activation == "tanh":
            act = Tanh()
        elif activation == "sigmoid":
            act = Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.lora = nn.Sequential(
            Linear(input_dim, low_rank_dim, bias=False),
            act,
            Linear(low_rank_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class RWKV7Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int = 96,
        gate_low_rank_dim: int = 320,
        a_low_rank_dim: int = 96,
        v_low_rank_dim: int = 64,
        norm_eps: float = 1e-5,
        layer_idx: int = 0,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads if num_heads is not None else hidden_size // head_dim
        self.key_dim = hidden_size
        self.value_dim = hidden_size
        self.head_v_dim = self.value_dim // self.num_heads
        self.layer_idx = layer_idx

        # Token-shift mixing parameters (one per channel, broadcast across [B, T])
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Per-channel scalars used during k-update / readout correction
        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, decay_low_rank_dim,
                           activation="tanh", bias=True)
        if layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, v_low_rank_dim,
                               activation=None, bias=True)
        self.a_lora = LoRA(hidden_size, self.key_dim, a_low_rank_dim,
                           activation=None, bias=True)
        self.g_lora = LoRA(hidden_size, self.value_dim, gate_low_rank_dim,
                           activation="sigmoid", bias=False)

        self.kk_norm = L2Norm(dim=-1)
        self.g_norm = GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            eps=self.head_dim * norm_eps,
            affine=True,
        )
        # Triton fast paths + naive fallback. Dispatch happens in forward
        # based on T and ``use_fast_kernels``.
        self.use_fast_kernels = use_fast_kernels
        self.naive_recurrence = NaiveRecurrentRWKV7()
        if use_fast_kernels:
            self.fused_recurrence = FusedRecurrentRWKV7()
            self.chunk = ChunkRWKV7()

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None, torch.Tensor]:
        B, T, _ = hidden_states.shape

        # Token shift: shifted[t] = previous token's hidden state.
        # For cached decode the previous token lives in past_key_values.conv_states[id(self)]
        # (a [B, hidden_size] tensor); for fresh prefill it's zero (left-pad).
        prev_shift = None
        if past_key_values is not None:
            cs = getattr(past_key_values, "conv_states", None)
            if cs is not None:
                prev_shift = cs.get(id(self))
        shifted = torch.empty_like(hidden_states)
        if prev_shift is not None:
            shifted[:, 0] = prev_shift
        else:
            shifted[:, 0].zero_()
        if T > 1:
            shifted[:, 1:] = hidden_states[:, :-1]
        delta = shifted - hidden_states
        # Save the last hidden vec as the conv_state for the next call.
        new_conv_state = hidden_states[:, -1].detach() if use_cache else None

        # Fused addcmul: xi = hidden_states + delta * x_i
        xr = torch.addcmul(hidden_states, delta, self.x_r)
        xw = torch.addcmul(hidden_states, delta, self.x_w)
        xk = torch.addcmul(hidden_states, delta, self.x_k)
        xv = torch.addcmul(hidden_states, delta, self.x_v)
        xa = torch.addcmul(hidden_states, delta, self.x_a)
        xg = torch.addcmul(hidden_states, delta, self.x_g)

        r = self.r_proj(xr)
        # FLA's exact magic constant: w = -softplus(-w_lora(xw)) - 0.5
        # is approximated in the reference checkpoint by a sigmoid scaled by
        # -0.5 / sigmoid(0) = -1.213... but the released models use the
        # exp(-0.5) = 0.6065... pre-bias. We keep the released-checkpoint
        # form to match fla-hub/rwkv7-2.9B-g1 exactly.
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        # Cross-layer v_first mixing: layer 0 sets it, others lerp toward it.
        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        # L2-normalized key (per-head)
        kk = self.kk_norm(
            (k * self.k_k).view(B, T, self.num_heads, self.head_dim)
        )

        # Key update: k = k * (1 + (a - 1) * k_a)
        k = k.addcmul(k * (a - 1), self.k_a)

        r_mh = r.view(B, T, self.num_heads, self.head_dim)
        w_mh = w.view(B, T, self.num_heads, self.head_dim)
        k_mh = k.view(B, T, self.num_heads, self.head_dim)
        a_mh = a.view(B, T, self.num_heads, self.head_dim)
        v_mh = v.view(B, T, self.num_heads, self.head_v_dim)

        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))

        # Dispatch:
        #   T >= 64 + fast kernels -> chunk_rwkv7 (prefill / training)
        #   T  < 64 + fast kernels -> fused_mul_recurrent_rwkv7 (decode)
        #   no fast kernels         -> naive PyTorch (CPU / debug / reference)
        if self.use_fast_kernels and r.is_cuda:
            if T >= _CHUNK_THRESHOLD:
                # The chunk kernel takes the DPLR decomposition (a=-kk, b=kk*gate_a).
                o, final_state = self.chunk(
                    r=r_mh, w=w_mh, k=k_mh, v=v_mh,
                    a=-kk, b=kk * a_mh,
                    scale=1.0,
                    initial_state=initial_state,
                    output_final_state=use_cache,
                )
            else:
                o, final_state = self.fused_recurrence(
                    r=r_mh, w=w_mh, k=k_mh, v=v_mh,
                    kk=kk, a=a_mh,
                    scale=1.0,
                    initial_state=initial_state,
                    output_final_state=use_cache,
                )
            # Fast-path output is already [B, T, H, V] — no transpose needed.
        else:
            # Naive path expects [B, H, T, D]
            o, final_state = self.naive_recurrence(
                r_mh.transpose(1, 2), w_mh.transpose(1, 2),
                k_mh.transpose(1, 2), v_mh.transpose(1, 2),
                kk.transpose(1, 2), a_mh.transpose(1, 2),
                scale=1.0,
                initial_state=initial_state,
                output_final_state=use_cache,
            )
            o = o.transpose(1, 2)  # [B, H, T, V] -> [B, T, H, V]

        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state
            if not hasattr(past_key_values, "conv_states"):
                past_key_values.conv_states = {}
            past_key_values.conv_states[id(self)] = new_conv_state

        # [B, T, H, V] -> [B*T, value_dim] for GroupNorm
        o = self.g_norm(o.reshape(B * T, -1)).view(B, T, -1)

        # Gate output correction: (o + sum_dim(r * k * r_k) * v) * g
        correction = (
            (r_mh * k_mh * self.r_k[None, None]).sum(-1, keepdim=True) * v_mh
        ).reshape(B, T, -1)
        o = (o + correction) * g

        return self.o_proj(o), None, past_key_values, v_first
