"""Runtime Mamba state-slot cache management (non-benchmark infrastructure).

Mirrors vLLM's Mamba state plumbing (see
``vllm/v1/attention/backends/mamba_attn.py`` and
``vllm/v1/attention/backends/mamba2_attn.py``):

  - ``MambaStateManager`` owns the global conv/ssm state tensors, one
    pair per layer, allocated as ``[num_slots, ...]``. Free slots are
    managed via a deque so a ``Sequence`` claims one slot for its
    lifetime.
  - ``Mamba2Metadata`` / ``MambaMetadata`` carry per-batch tensors
    (state slot indices, prefill/decode split, chunk indices) consumed
    by the mixer in its forward pass via the global Context.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


def compute_causal_conv1d_metadata(
    query_start_loc_p: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Precompute the aux pointers used by vLLM's varlen causal-conv kernel.

    This is Mamba v1 prefill metadata, so it lives with the Mamba state
    structs rather than in the generic engine.
    """
    seqlens = query_start_loc_p.diff().to(device="cpu", dtype=torch.int32)
    nums_dict: dict = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    device = query_start_loc_p.device

    for block_m in [8]:
        nums = torch.div(
            seqlens + (block_m - 1),
            block_m,
            rounding_mode="floor",
        )
        nums_dict[block_m] = {}
        nums_dict[block_m]["nums"] = nums
        nums_dict[block_m]["tot"] = nums.sum().item()

        mlist = torch.repeat_interleave(
            torch.arange(len(nums), dtype=torch.int32, device="cpu"),
            nums.to(dtype=torch.int64),
        )
        nums_dict[block_m]["mlist"] = mlist
        mlist_len = len(mlist)
        nums_dict[block_m]["mlist_len"] = mlist_len
        max_num_programs = max(1024, mlist_len) * 2

        offsetlist: list[int] = []
        for num in nums.tolist():
            offsetlist.extend(range(num))
        offsetlist_t = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[block_m]["offsetlist"] = offsetlist_t

        if batch_ptr is None:
            batch_ptr = torch.full(
                (max_num_programs,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            token_chunk_offset_ptr = torch.full(
                (max_num_programs,),
                -1,
                dtype=torch.int32,
                device=device,
            )
        elif batch_ptr.numel() < max_num_programs:
            batch_ptr.resize_(max_num_programs).fill_(-1)
            token_chunk_offset_ptr.resize_(max_num_programs).fill_(-1)

        batch_ptr[:mlist_len].copy_(mlist.to(device=device))
        token_chunk_offset_ptr[:mlist_len].copy_(
            offsetlist_t.to(device=device),
        )
        nums_dict[block_m]["batch_ptr"] = batch_ptr
        nums_dict[block_m]["token_chunk_offset_ptr"] = token_chunk_offset_ptr

    return nums_dict, batch_ptr, token_chunk_offset_ptr


class MambaSlotCache:
    """Cache view for one sequence slot."""

    def __init__(
        self,
        conv_states: list[torch.Tensor],
        ssm_states: list[torch.Tensor],
        conv_kernel_size: int,
    ):
        self.conv_states = conv_states
        self.ssm_states = ssm_states
        self.conv_kernel_size = conv_kernel_size

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)
        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(
            device=conv_state.device,
            dtype=conv_state.dtype,
        )
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx].zero_()
        self.ssm_states[layer_idx] += new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]


class MambaStateManager:
    """Owns global Mamba recurrent state tensors and free-slot bookkeeping.

    All TP ranks maintain identical ``_free_slots`` deques: each rank
    deterministically pops slots in the same order, so the per-step
    state slot indices match across ranks without any cross-rank
    communication.
    """

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        conv_dim: int,
        ssm_state_shape: tuple[int, ...],
        conv_kernel: int,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_slots = num_slots
        self.conv_kernel = conv_kernel
        self._free_slots: deque[int] = deque(range(num_slots))
        self._in_use: set[int] = set()
        self._slot_views: dict[int, MambaSlotCache] = {}

        self.conv_states: list[torch.Tensor] = []
        self.ssm_states: list[torch.Tensor] = []
        for _ in range(num_hidden_layers):
            # Layout matches vLLM's mamba1/2 state cache:
            #   ``[num_slots, conv_kernel - 1, conv_dim]``.
            # Mixers transpose the last two dims when handing the cache
            # to ``causal_conv1d_fn`` / ``causal_conv1d_update`` so that
            # the kernel-required ``stride_istate_dim == 1`` holds.
            conv_state = torch.zeros(
                num_slots,
                max(conv_kernel - 1, 1),
                conv_dim,
                device=device,
                dtype=dtype,
            )
            ssm_state = torch.zeros(
                num_slots,
                *ssm_state_shape,
                device=device,
                dtype=dtype,
            )
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "mark_static_address"):
                torch._dynamo.mark_static_address(conv_state)
                torch._dynamo.mark_static_address(ssm_state)
            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def reset_slot(self, slot: int) -> None:
        for layer_idx in range(len(self.conv_states)):
            self.conv_states[layer_idx][slot].zero_()
            self.ssm_states[layer_idx][slot].zero_()

    def allocate(self, seq) -> int:
        """Claim a free slot for ``seq``.

        Each TP rank holds its own ``_free_slots`` deque.  Because ranks
        receive identical, in-order ``allocate`` / ``deallocate`` calls
        (broadcast via SHM in ``ModelRunner.call``), their free pools
        stay in lockstep so popping from each rank's deque produces the
        same slot index without explicit coordination.
        """
        if getattr(seq, "state_slot", None) is not None:
            return seq.state_slot
        if not self._free_slots:
            raise RuntimeError("No free Mamba state slots")
        slot = self._free_slots.popleft()
        self._in_use.add(slot)
        self.reset_slot(slot)
        seq.state_slot = slot
        return slot

    def deallocate(self, seq) -> None:
        slot = getattr(seq, "state_slot", None)
        if slot is None:
            return
        if slot in self._in_use:
            self._in_use.remove(slot)
            self.reset_slot(slot)
            self._free_slots.append(slot)
        seq.state_slot = None

    def get_slot_cache(self, slot: int) -> MambaSlotCache:
        cache = self._slot_views.get(slot)
        if cache is not None:
            return cache

        conv_views = [x[slot:slot + 1] for x in self.conv_states]
        ssm_views = [x[slot:slot + 1] for x in self.ssm_states]
        cache = MambaSlotCache(conv_views, ssm_views, self.conv_kernel)
        self._slot_views[slot] = cache
        return cache


class KimiLinearStateManager:
    """Flat recurrent state plus optional paged KV cache for hybrid models.

    Kimi-Linear uses separate q/k/v convolution states for KDA layers.
    Qwen3-Next uses one packed qkv convolution state for GDN layers plus
    regular paged full-attention KV for dense-attention layers.
    """

    def __init__(
        self,
        *,
        config,
        num_slots: int,
        block_size: int,
        num_mla_blocks: int,
        allocate_mla_kv_tensors: bool,
        tp_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_slots = num_slots
        self.block_size = block_size
        self.num_mla_blocks = num_mla_blocks
        self.tp_size = tp_size
        self.device = device
        self.dtype = dtype
        self.model_type = getattr(config, "model_type", "kimi_linear")

        self._free_slots: deque[int] = deque(range(num_slots))
        self._in_use: set[int] = set()

        self.conv_q: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_k: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_v: list[torch.Tensor | None] = [None] * self.num_layers
        self.gdn_conv: list[torch.Tensor | None] = [None] * self.num_layers
        self.recurrent: list[torch.Tensor | None] = [None] * self.num_layers

        self.k_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.v_cache: list[torch.Tensor | None] = [None] * self.num_layers

        self._free_blocks: deque[int] | None = None
        if num_mla_blocks > 0:
            self._free_blocks = deque(range(num_mla_blocks))

        if self.model_type == "qwen3_next":
            self._allocate_qwen3_next(allocate_mla_kv_tensors)
            return

        local_kda_heads = config.kda_num_heads // tp_size
        local_kda_proj = config.kda_num_heads * config.kda_head_dim // tp_size
        kernel = config.short_conv_kernel_size

        local_mla_heads = config.num_attention_heads // tp_size
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        for i in range(self.num_layers):
            if config.is_kda_layer(i):
                self.conv_q[i] = torch.zeros(
                    num_slots, kernel - 1, local_kda_proj,
                    device=device, dtype=dtype,
                )
                self.conv_k[i] = torch.zeros(
                    num_slots, kernel - 1, local_kda_proj,
                    device=device, dtype=dtype,
                )
                self.conv_v[i] = torch.zeros(
                    num_slots, kernel - 1, local_kda_proj,
                    device=device, dtype=dtype,
                )
                self.recurrent[i] = torch.zeros(
                    num_slots, local_kda_heads,
                    config.kda_head_dim, config.kda_head_dim,
                    device=device, dtype=torch.float32,
                )
            elif allocate_mla_kv_tensors:
                self.k_cache[i] = torch.zeros(
                    num_mla_blocks, block_size, local_mla_heads, qk_head_dim,
                    device=device, dtype=dtype,
                )
                self.v_cache[i] = torch.zeros(
                    num_mla_blocks, block_size, local_mla_heads, qk_head_dim,
                    device=device, dtype=dtype,
                )

    def _allocate_qwen3_next(self, allocate_mha_kv_tensors: bool) -> None:
        cfg = self.config
        local_k_heads = cfg.linear_num_key_heads // self.tp_size
        local_v_heads = cfg.linear_num_value_heads // self.tp_size
        head_k_dim = cfg.linear_key_head_dim
        head_v_dim = cfg.linear_value_head_dim
        conv_kernel = cfg.linear_conv_kernel_dim
        local_conv_dim = (
            2 * local_k_heads * head_k_dim
            + local_v_heads * head_v_dim
        )

        local_kv_heads = (
            cfg.num_key_value_heads // self.tp_size
            if cfg.num_key_value_heads % self.tp_size == 0
            else cfg.num_key_value_heads
        )
        head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads,
        )

        for i in range(self.num_layers):
            if cfg.is_linear_attn_layer(i):
                self.gdn_conv[i] = torch.zeros(
                    self.num_slots,
                    conv_kernel - 1,
                    local_conv_dim,
                    device=self.device,
                    dtype=self.dtype,
                ).transpose(-1, -2)
                self.recurrent[i] = torch.zeros(
                    self.num_slots,
                    local_v_heads,
                    head_k_dim,
                    head_v_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif allocate_mha_kv_tensors:
                self.k_cache[i] = torch.zeros(
                    self.num_mla_blocks,
                    self.block_size,
                    local_kv_heads,
                    head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.v_cache[i] = torch.zeros(
                    self.num_mla_blocks,
                    self.block_size,
                    local_kv_heads,
                    head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def reset_slot(self, slot: int) -> None:
        for layer_id in range(self.num_layers):
            if self.conv_q[layer_id] is not None:
                self.conv_q[layer_id][slot].zero_()
                self.conv_k[layer_id][slot].zero_()
                self.conv_v[layer_id][slot].zero_()
                self.recurrent[layer_id][slot].zero_()
            if self.gdn_conv[layer_id] is not None:
                self.gdn_conv[layer_id][slot].zero_()
                self.recurrent[layer_id][slot].zero_()

    def allocate(self, seq) -> int:
        if getattr(seq, "state_slot", None) is not None:
            return seq.state_slot
        if not self._free_slots:
            raise RuntimeError("No free recurrent state slots")
        slot = self._free_slots.popleft()
        self._in_use.add(slot)
        self.reset_slot(slot)
        seq.state_slot = slot
        seq.block_table = []
        return slot

    def ensure_blocks_for(self, seq, total_tokens: int) -> None:
        if self._free_blocks is None:
            return
        blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < blocks_needed:
            if not self._free_blocks:
                raise RuntimeError("No free MLA KV cache blocks")
            seq.block_table.append(self._free_blocks.popleft())

    def deallocate(self, seq) -> None:
        slot = getattr(seq, "state_slot", None)
        if slot is not None and slot in self._in_use:
            self._in_use.remove(slot)
            self.reset_slot(slot)
            self._free_slots.append(slot)
            seq.state_slot = None
        if self._free_blocks is not None and seq.block_table:
            self._free_blocks.extend(seq.block_table)
            seq.block_table = []

    @property
    def q_conv_states(self):
        return self.conv_q

    @property
    def k_conv_states(self):
        return self.conv_k

    @property
    def v_conv_states(self):
        return self.conv_v

    @property
    def recurrent_states(self):
        return self.recurrent


@dataclass
class MambaMetadata:
    """Per-batch metadata for a Mamba v1 forward pass.

    Mirrors vLLM ``Mamba1AttentionMetadata`` (a thin wrapper over
    ``BaseMambaAttentionMetadata`` -- see
    ``vllm/v1/attention/backends/mamba_attn.py``).

    All tensors live on the inference device.
    """
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # Prefill-only (None when num_prefills == 0)
    has_initial_states_p: torch.Tensor | None = None  # bool [num_prefills]
    query_start_loc_p: torch.Tensor | None = None     # int32 [num_prefills+1]
    state_indices_p: torch.Tensor | None = None       # int32 [num_prefills]
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # Decode-only (None when num_decodes == 0)
    state_indices_d: torch.Tensor | None = None       # int32 [num_decodes]


@dataclass
class Mamba2Metadata(MambaMetadata):
    """Per-batch metadata for a Mamba2 / SSD forward pass.

    Adds chunked-prefill support on top of ``MambaMetadata``.  Mirrors
    vLLM ``Mamba2AttentionMetadata``.
    """
    prep_initial_states: bool = False
    chunk_size: int = 256

    # Chunk metadata (prefill only) -- see vLLM
    # ``BaseMambaAttentionMetadataBuilder._compute_chunk_metadata``.
    seq_idx_p: torch.Tensor | None = None              # int32 [nchunks]
    cu_chunk_seqlen_p: torch.Tensor | None = None      # int32 [nchunks+1]
    last_chunk_indices_p: torch.Tensor | None = None   # int32 [num_prefills]


def build_chunk_metadata(
    query_start_loc_p: torch.Tensor,
    chunk_size: int,
    num_computed_tokens_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Mamba2 chunk-aligned varlen metadata.

    Direct port of vLLM ``BaseMambaAttentionMetadataBuilder._compute_chunk_metadata``.

    Args:
      query_start_loc_p: int32 tensor [num_prefills+1] cumulative token
        starts in the *prefill* sub-batch.
      chunk_size: physical chunk size (e.g. 256 for Codestral).
      num_computed_tokens_p: int32 tensor [num_prefills] of already-computed
        tokens per request (for chunked prefill resumption).  If ``None``,
        treated as all zeros (fresh prefill).

    Returns ``(cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p)`` on the
    same device as ``query_start_loc_p``.
    """
    device = query_start_loc_p.device
    qsl = query_start_loc_p.to("cpu").tolist()
    num_prefills = len(qsl) - 1
    if num_computed_tokens_p is None:
        nct = [0] * num_prefills
    else:
        nct = num_computed_tokens_p.to("cpu").tolist()

    cu_chunk_seqlen: list[int] = []
    seq_idx: list[int] = []
    last_chunk_indices: list[int] = []
    seqlen_pos = 0

    for req_idx in range(num_prefills):
        this_num_computed = nct[req_idx]
        this_new_tokens = qsl[req_idx + 1] - qsl[req_idx]

        # Finish off a partially-filled chunk if computed isn't chunk aligned.
        if this_num_computed % chunk_size != 0:
            seq_idx.append(req_idx)
            cu_chunk_seqlen.append(seqlen_pos)
            chunk_len = (
                ((this_num_computed + chunk_size - 1) // chunk_size) * chunk_size
                - this_num_computed
            )
            chunk_len = min(chunk_len, this_new_tokens)
            seqlen_pos += chunk_len
            this_new_tokens -= chunk_len

        n_chunks = (this_new_tokens + chunk_size - 1) // chunk_size
        for _ in range(n_chunks):
            seq_idx.append(req_idx)
            cu_chunk_seqlen.append(seqlen_pos)
            chunk_len = min(chunk_size, this_new_tokens)
            seqlen_pos += chunk_len
            this_new_tokens -= chunk_len

        assert this_new_tokens == 0
        last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

    cu_chunk_seqlen.append(seqlen_pos)

    cu_chunk_seqlen_p = torch.tensor(cu_chunk_seqlen, device=device, dtype=torch.int32)
    seq_idx_p = torch.tensor(seq_idx, device=device, dtype=torch.int32)
    last_chunk_indices_p = torch.tensor(last_chunk_indices, device=device, dtype=torch.int32)
    return cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p
