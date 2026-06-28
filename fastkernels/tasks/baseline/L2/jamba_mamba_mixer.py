"""Jamba's Mamba v1 selective-scan mixer (with per-layer dt/B/C RMSNorms).

Reference: ``transformers.models.jamba.modeling_jamba.JambaMambaMixer``
            and ``vllm.model_executor.layers.mamba.mamba_mixer.MambaMixer``.

Key differences from the plain Mamba v1 mixer
(``L2.mamba_mixer.MambaMixer``):

  * Three additional RMSNorms applied to (dt, B, C) after the x_proj
    split.  ``b_layernorm`` and ``c_layernorm`` use ``hidden_size = 16``
    (= ``mamba_d_state``), which falls outside the multiple-of-32
    range the L1 ``RMSNorm`` CUDA kernel handles correctly, so we use
    the autograd-friendly :class:`L1.rms_norm_native.RMSNormNative`
    everywhere here for safety.
  * Operates on flat varlen token layout (``[intermediate, total_tokens]``
    + ``query_start_loc`` + ``cache_indices``) so we can reuse vLLM's
    SOTA fused Mamba kernels (``causal_conv1d_fn``,
    ``selective_scan_fn``, ``causal_conv1d_update``,
    ``selective_state_update``).  The :class:`infra.jamba_engine.JambaEngine`
    flattens its left-padded ``[batch, seq_len]`` input to this layout
    on every forward.
  * Tensor parallel removed: targeted at single-GPU :class:`JambaEngine`.

Forward signature: ``forward(positions, hidden_states)`` -- matches the
project's ``(positions, hidden_states)`` mixer convention used by
Llama / Mamba / Mamba2.  Per-step Mamba state and metadata are read
from the global ``Context`` (populated by ``set_jamba_context``); the
mixer reaches into it for its per-layer slab using ``self.layer_idx``.

L1 ops used (no torch.nn.functional or external libs leaked into L2):

  - ``L1.linear.Linear``                   -- in_proj, x_proj, dt_proj, out_proj
  - ``L1.rms_norm_native.RMSNormNative``   -- dt/B/C layernorms
  - vLLM ``causal_conv1d_*`` / ``selective_*``  -- single-kernel L1 ops in our
                                                   taxonomy: each is one CUDA
                                                   kernel launch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)

from ....infra.context import get_context
from ..L1.linear import Linear
from ..L1.rms_norm_native import RMSNormNative


class JambaMambaMixer(nn.Module):
    """Mamba v1 mixer with Jamba's per-layer dt/B/C RMSNorms.

    Forward signature is ``(positions, hidden_states)`` matching the
    project's mixer convention.  The mixer reads ``conv_state`` /
    ``ssm_state`` (per-layer) and the per-batch metadata
    (``query_start_loc``, ``cache_indices``, ``has_initial_state``,
    ``is_decode``, ``mamba_pad_mask``) from the global ``Context``
    populated by the engine via ``set_jamba_context``.

    Input is ``[B, T, hidden]``; we reshape to flat ``[total_tokens,
    hidden]`` internally, run the vLLM Mamba kernels, then reshape back
    to ``[B, T, hidden]`` on output.

    The mixer mutates the per-layer ``conv_state`` / ``ssm_state`` slabs
    in place at the given slot indices (vLLM kernels do this internally).
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        rms_norm_eps: float,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size
        self.time_step_rank = time_step_rank
        self.use_conv_bias = use_conv_bias
        self.layer_idx = layer_idx

        # in_proj packs [hidden, gate], each of size intermediate_size.
        self.in_proj = Linear(
            hidden_size, intermediate_size * 2, bias=use_bias,
        )

        # Depthwise causal conv1d: parameter layout matches HF's
        # ``nn.Conv1d(intermediate, intermediate, K, groups=intermediate)``
        # which stores ``[intermediate, 1, K]``.  The kernels expect a
        # flat ``[intermediate, K]`` view, so we hold the weight as a
        # bare Parameter and ``.view`` to that shape inside forward.
        self.conv1d_weight = nn.Parameter(
            torch.empty(intermediate_size, 1, conv_kernel_size),
        )
        if use_conv_bias:
            self.conv1d_bias = nn.Parameter(torch.empty(intermediate_size))
        else:
            self.register_parameter("conv1d_bias", None)

        # x_proj: x -> (dt, B, C). dt has rank time_step_rank, B/C have
        # rank ssm_state_size each.
        self.x_proj = Linear(
            intermediate_size, time_step_rank + 2 * ssm_state_size,
            bias=False,
        )
        # dt_proj: time_step_rank -> intermediate.  Bias is added by
        # the selective-scan kernel; we keep it as the param's bias and
        # forward it to the kernel explicitly (using F.linear would
        # leak nn.functional into L2, so we go through Linear.matmul
        # without the bias instead).
        self.dt_proj = Linear(time_step_rank, intermediate_size, bias=True)

        # A_log -> A done at load time via the param's weight_loader.
        self.A = nn.Parameter(
            torch.empty(intermediate_size, ssm_state_size, dtype=torch.float32),
        )
        self.A.weight_loader = self._A_loader

        self.D = nn.Parameter(torch.ones(intermediate_size))

        self.out_proj = Linear(intermediate_size, hidden_size, bias=use_bias)

        # Per-layer norms on (dt, B, C). All dims are <= 32 here, which
        # falls outside the L1 RMSNorm CUDA kernel's safe range, so use
        # the native pure-PyTorch path everywhere.
        self.dt_layernorm = RMSNormNative(time_step_rank, eps=rms_norm_eps)
        self.b_layernorm = RMSNormNative(ssm_state_size, eps=rms_norm_eps)
        self.c_layernorm = RMSNormNative(ssm_state_size, eps=rms_norm_eps)

    @staticmethod
    def _A_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Convert HF's stored ``A_log`` to the kernel's expected ``A``.

        The HF checkpoint stores ``log(A)`` (positive); the SSM kernels
        consume ``A = -exp(A_log)`` (negative).  Matches vLLM's loader.
        """
        param.data.copy_(-torch.exp(loaded_weight.float()))

    def _ssm_transform(self, x: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Compute (dt, B, C) from x via x_proj + dt/B/C norms + dt_proj.

        ``x``: ``[total_tokens, intermediate]``
        Returns:
          dt: ``[intermediate, total_tokens]``
          B:  ``[total_tokens, ssm_state_size]``
          C:  ``[total_tokens, ssm_state_size]``
        """
        ssm_params = self.x_proj(x)
        dt, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        dt = self.dt_layernorm(dt)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
        # dt_proj WITHOUT its bias (the kernel adds it). ``F.linear``
        # would be cleaner but is forbidden in L2; go through the L1
        # Linear op's underlying ``Matmul`` directly.
        dt = self.dt_proj.matmul(dt, self.dt_proj.weight, None)
        # Kernels expect ``dt`` as ``[intermediate, total_tokens]``.
        return dt.transpose(-2, -1), B, C

    def forward(
        self,
        positions: torch.Tensor | None,    # unused (Mamba handles position
                                            # via recurrence); kept for
                                            # mixer-uniform signature.
        hidden_states: torch.Tensor,        # [N, hidden] flat varlen
    ) -> torch.Tensor:
        """Mamba v1 selective-scan forward, flat-varlen layout.

        Supports three batch shapes (selected by per-step metadata):

          * **Pure decode** (``is_decode=True``, no prefill):
            ``[num_decode, hidden]`` -- one new token per row.  Runs
            ``causal_conv1d_update`` + ``selective_state_update``.
          * **Pure prefill** (``is_decode=False``, no decode):
            ``[total_prefill_tokens, hidden]`` flat varlen.  Runs
            ``causal_conv1d_fn`` + ``selective_scan_fn``.
          * **Mixed prefill + decode**
            (``num_prefill_tokens > 0`` AND ``num_decode_tokens > 0``):
            input is ``[num_prefill_tokens + num_decode_tokens, hidden]``
            with prefill rows first.  Splits, runs prefill kernels on
            the prefill range and decode kernels on the decode range,
            concatenates outputs.  Same kernel pair as the homogeneous
            paths -- just two calls per layer instead of one.
            Mirrors ``infra.engine.run_mamba_mixed``'s convention but
            uses prefill-first ordering to match the project's
            ``Attention._forward_mixed`` ctx fields.

        Returns ``[N, hidden]`` (same shape as input).
        """
        ctx = get_context()
        meta = ctx.mamba_metadata
        assert meta is not None, (
            "JambaMambaMixer.forward called without a Mamba metadata "
            "installed on the global Context (use set_jamba_context)."
        )

        conv_state = meta.conv_states[self.layer_idx]
        ssm_state = meta.ssm_states[self.layer_idx]

        # Detect mixed batches via the project's standard ctx fields
        # (set by ``set_jamba_context`` when ``is_mixed=True``).  Falls
        # back to the binary ``is_decode`` flag for homogeneous batches.
        is_mixed = bool(getattr(ctx, "is_mixed", False))
        if is_mixed:
            num_prefill_tokens = ctx.num_prefill_tokens
            num_decode_tokens = ctx.num_decode_tokens
            # Per-phase Mamba metadata installed by ``set_jamba_context``
            # under dedicated mamba_metadata fields.
            state_indices_p = meta.state_indices_p
            state_indices_d = meta.state_indices_d
            query_start_loc_p = meta.query_start_loc
            has_initial_state_p = meta.has_initial_state
        else:
            cache_indices = meta.cache_indices
            query_start_loc = meta.query_start_loc
            has_initial_state = meta.has_initial_state
            is_decode = meta.is_decode

        # 1. Gated MLP-style input projection (one big matmul over the
        # full mixed batch -- same as vLLM's mamba mixer).
        projected = self.in_proj(hidden_states)
        # Mamba kernels expect [intermediate, total_tokens].
        projected = projected.transpose(-2, -1)
        x_states, gate = projected.chunk(2, dim=-2)

        conv_w = self.conv1d_weight.view(
            self.conv1d_weight.size(0), self.conv1d_weight.size(2),
        )
        conv_b = self.conv1d_bias
        time_proj_bias = (
            self.dt_proj.bias.float() if self.dt_proj.bias is not None else None
        )

        # ------------------------------------------------------------------
        # MIXED PATH: split into prefill rows [0:n_p] and decode rows
        # [n_p:n_p+n_d], run both kernel families, concat outputs.
        # ------------------------------------------------------------------
        if is_mixed:
            n_p = num_prefill_tokens
            n_d = num_decode_tokens

            x_p = x_states[:, :n_p]                                  # [I, n_p]
            x_d = x_states[:, n_p:].transpose(0, 1).contiguous()     # [n_d, I]
            gate_p = gate[:, :n_p]                                   # [I, n_p]
            gate_d = gate[:, n_p:]                                   # [I, n_d]

            # Conv: prefill (range update) + decode (single-step update).
            # Both update the SAME global conv_state slabs at their own
            # state indices.  Pad rows in either set use slot index -1
            # (the vendored kernels skip those rows).
            conv_out_p = causal_conv1d_fn(
                x_p, conv_w, conv_b,
                conv_states=conv_state,
                query_start_loc=query_start_loc_p,
                cache_indices=state_indices_p,
                has_initial_state=has_initial_state_p,
                activation="silu",
            )                                                        # [I, n_p]
            conv_out_d_t = causal_conv1d_update(
                x_d, conv_state, conv_w, conv_b, "silu",
                conv_state_indices=state_indices_d,
            )                                                        # [n_d, I]
            conv_out_d = conv_out_d_t.transpose(0, 1).contiguous()   # [I, n_d]
            conv_out = torch.cat([conv_out_p, conv_out_d], dim=-1)   # [I, n_p+n_d]

            # SSM transform on the combined post-conv output.
            dt, B_bts, C_bts = self._ssm_transform(conv_out.transpose(-2, -1))
            dt_p = dt[:, :n_p]                                       # [I, n_p]
            dt_d = dt[:, n_p:].transpose(0, 1).contiguous()          # [n_d, I]
            B_p = B_bts[:n_p]                                        # [n_p, S]
            B_d = B_bts[n_p:]                                        # [n_d, S]
            C_p = C_bts[:n_p]                                        # [n_p, S]
            C_d = C_bts[n_p:]                                        # [n_d, S]

            # Prefill scan.
            scan_out_p = selective_scan_fn(
                conv_out_p, ssm_state, dt_p,
                self.A,
                B_p.transpose(-2, -1),
                C_p.transpose(-2, -1),
                self.D.float(),
                gate_p, time_proj_bias,
                delta_softplus=True,
                cache_indices=state_indices_p,
                has_initial_state=has_initial_state_p,
                query_start_loc=query_start_loc_p,
            )                                                        # [I, n_p]

            # Decode scan (single-step state update).
            scan_out_d = torch.empty_like(conv_out_d.transpose(0, 1))  # [n_d, I]
            selective_state_update(
                ssm_state,
                conv_out_d.transpose(0, 1).contiguous(),
                dt_d,
                self.A,
                B_d, C_d, self.D,
                gate_d.transpose(0, 1).contiguous(),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_d,
                out=scan_out_d,
            )
            scan_out_d = scan_out_d.transpose(0, 1).contiguous()     # [I, n_d]

            scan_out = torch.cat([scan_out_p, scan_out_d], dim=-1)   # [I, n_p+n_d]
            return self.out_proj(scan_out.transpose(-2, -1))

        # ------------------------------------------------------------------
        # HOMOGENEOUS PATH: pure decode OR pure prefill.
        # ------------------------------------------------------------------
        if is_decode:
            x_t = x_states.transpose(0, 1).contiguous()
            conv_out_t = causal_conv1d_update(
                x_t, conv_state, conv_w, conv_b, "silu",
                conv_state_indices=cache_indices,
            )
            conv_out = conv_out_t.transpose(0, 1).contiguous()
        else:
            conv_out = causal_conv1d_fn(
                x_states, conv_w, conv_b,
                conv_states=conv_state,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                activation="silu",
            )

        dt, B_bts, C_bts = self._ssm_transform(conv_out.transpose(-2, -1))

        if is_decode:
            scan_out = torch.empty_like(conv_out.transpose(0, 1))
            selective_state_update(
                ssm_state,
                conv_out.transpose(0, 1).contiguous(),
                dt.transpose(0, 1).contiguous(),
                self.A, B_bts, C_bts, self.D,
                gate.transpose(0, 1).contiguous(),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=cache_indices,
                out=scan_out,
            )
            scan_out = scan_out.transpose(0, 1).contiguous()
        else:
            scan_out = selective_scan_fn(
                conv_out, ssm_state, dt,
                self.A,
                B_bts.transpose(-2, -1),
                C_bts.transpose(-2, -1),
                self.D.float(),
                gate, time_proj_bias,
                delta_softplus=True,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                query_start_loc=query_start_loc,
            )

        return self.out_proj(scan_out.transpose(-2, -1))
