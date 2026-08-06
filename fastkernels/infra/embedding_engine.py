"""vLLM-style pooling engine for encoder-only embedding models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

from .embedder_loader import load_bge_m3_model, load_colbertv2_model


@dataclass
class PoolingOutput:
    data: torch.Tensor


@dataclass
class PoolingRequestOutput:
    request_id: str
    outputs: PoolingOutput
    prompt_token_ids: list[int]
    num_cached_tokens: int
    finished: bool


class EmbeddingEngine:
    """Minimal vLLM-compatible pooling interface for fastkernels embedders."""

    DEFAULT_MAX_NUM_BATCHED_TOKENS = 16384
    DEFAULT_MAX_NUM_SEQS = 1024

    def __init__(
        self,
        model_name: str,
        *,
        seed: int = 0,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda:0",
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        compile_model: bool | None = None,
    ):
        self.model_name = model_name
        self.seed = seed
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_num_batched_tokens = (
            max_num_batched_tokens or self.DEFAULT_MAX_NUM_BATCHED_TOKENS
        )
        self.max_num_seqs = max_num_seqs or self.DEFAULT_MAX_NUM_SEQS
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        lower = model_name.lower()
        if "bge-m3" in lower or "bge_m3" in lower:
            self.model_key = "bge_m3"
            self.model, self.config = load_bge_m3_model(model_name, self.device, dtype)
        elif "colbert" in lower:
            self.model_key = "colbertv2"
            self.model, self.config = load_colbertv2_model(model_name, self.device, dtype)
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        self._forward_varlen = self.model.forward_varlen
        if compile_model is None:
            # Default OFF. torch.compile cannot pay for itself on this model:
            # ``flash_attn_varlen_func`` is wrapped in ``torch._dynamo.disable``
            # (see tasks/baseline/L1/fa_utils.py -- FA4's CuTeDSL launcher
            # rebuilds a closure per call, which Dynamo guards on and
            # recompiles against), so the traced ``forward_varlen`` breaks at
            # every encoder layer. bge-m3 ends up as ~25 compiled regions, and
            # a profile of one small encode showed 86 Dynamo cache lookups and
            # 61 AOTDispatcher prologues per call. vLLM's BertModel compile
            # works because its piecewise CUDA graphs remove exactly that
            # per-piece dispatch cost; without them the fusion never covers the
            # overhead. Measured with compile on vs off:
            #   bge-m3   throughput  532k  vs  567k tok/s
            #   colbert  throughput  570k  vs  776k tok/s
            #   colbert  bs=1 latency 5.21 vs 3.36 ms
            # Off wins on every axis. Callers can still pass compile_model=True.
            compile_model = False
        if compile_model:
            # NOT mode="reduce-overhead". That turns on cudagraph trees, which
            # "support dynamic shapes by recording a new graph for each distinct
            # input size" -- and a varlen pooling batch is a new size almost
            # every time: bge-m3's 1000-document workload produces 318 distinct
            # total-token counts across 333 scheduler batches. The recorded
            # graphs accumulate instead of amortizing, so throughput *degrades*
            # as the run proceeds; a second pass over the very same 333 batches
            # measured 7.2x SLOWER than the first (207s vs 29s).
            #
            # dynamic=True is load-bearing for the same reason: with static
            # shapes every new token count is a fresh Dynamo compile.
            self._forward_varlen = torch.compile(
                self.model.forward_varlen,
                dynamic=True,
            )

        torch.manual_seed(seed)

    @staticmethod
    def _bucket_tokens(n: int) -> int:
        """Round a row count up to a coarse bucket.

        Used only to size the pinned staging buffers in the D2H ring, never to
        pad the model input: a varlen pooling batch is a different total-token
        count nearly every time (bge-m3's 1000-document workload produces 318
        distinct counts across 333 scheduler batches), so allocating each
        staging buffer at its exact size would re-page-lock memory constantly.
        Bucketing collapses that to a handful of sizes that get reused.

        Padding the actual input to these buckets was tried and dropped: it
        made no measurable difference to throughput (565,796 vs 567,008 tok/s
        on bge-m3), because the per-shape first-touch cost was never the
        bottleneck -- the unbounded pinned staging was, see _acquire_pin_slot.
        So it was pure wasted compute.
        """
        if n <= 1024:
            g = 64
        elif n <= 4096:
            g = 256
        else:
            g = 512
        return ((n + g - 1) // g) * g

    def _make_scheduler_batches(self, token_lengths: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for idx, length in enumerate(token_lengths):
            length = max(int(length), 1)
            would_exceed_tokens = (
                current
                and current_tokens + length > self.max_num_batched_tokens
            )
            would_exceed_seqs = current and len(current) >= self.max_num_seqs
            if would_exceed_tokens or would_exceed_seqs:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(idx)
            current_tokens += length
        if current:
            batches.append(current)
        return batches

    def token_embed(
        self,
        prompts: str | Sequence[str],
        *,
        use_tqdm: bool = True,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        outputs = self.encode(
            prompts,
            pooling_task="token_embed",
            use_tqdm=use_tqdm,
            tokenization_kwargs=tokenization_kwargs,
        )
        return [item.outputs.data for item in outputs]

    def encode(
        self,
        prompts: str | Sequence[str],
        pooling_params: Any = None,
        *,
        use_tqdm: bool = True,
        pooling_task: str | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        del pooling_params
        if pooling_task != "token_embed":
            raise ValueError('EmbeddingEngine.encode currently supports pooling_task="token_embed"')

        if isinstance(prompts, str):
            requests: list[str | dict[str, Any]] = [prompts]
        else:
            requests = list(prompts)

        tokenization_kwargs = dict(tokenization_kwargs or {})
        tokenization_kwargs.setdefault("truncation", True)
        if requests and isinstance(requests[0], dict):
            input_ids_list = [
                list(request["prompt_token_ids"])
                for request in requests
            ]
        else:
            tokenized_batch = self.tokenizer(
                requests,
                add_special_tokens=True,
                padding=False,
                verbose=False,
                **tokenization_kwargs,
            )
            input_ids_list = tokenized_batch["input_ids"]
        token_lengths = [len(input_ids) for input_ids in input_ids_list]
        scheduler_batches = self._make_scheduler_batches(token_lengths)

        results: list[PoolingRequestOutput | None] = [None] * len(requests)
        batch_iter = scheduler_batches
        if use_tqdm:
            import sys

            from tqdm.auto import tqdm

            batch_iter = tqdm(
                batch_iter,
                total=len(scheduler_batches),
                desc=(
                    "fastkernels scheduled batches "
                    f"(<= {self.max_num_batched_tokens} tok, <= {self.max_num_seqs} seq)"
                ),
                unit="batch",
                file=sys.stdout,
            )
        copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

        with torch.no_grad():
            for batch_indices in batch_iter:
                prompt_token_ids = [input_ids_list[idx] for idx in batch_indices]
                lengths = [len(ids) for ids in prompt_token_ids]
                # One flat host list per tensor, then a single H2D each. The
                # previous version built one ``torch.arange`` on the device per
                # sequence and concatenated them, which is a device allocation
                # and launch per request -- 3.36 ms vs 5.21 ms per call on a
                # 56-token colbertv2 batch once that was removed.
                flat_ids = torch.tensor(
                    [t for ids in prompt_token_ids for t in ids],
                    dtype=torch.long, device=self.device,
                )
                flat_positions = torch.tensor(
                    [p for length in lengths for p in range(length)],
                    dtype=torch.long, device=self.device,
                )
                cu_seqlens = torch.zeros(
                    len(lengths) + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
                cu_seqlens[1:] = torch.tensor(
                    lengths, dtype=torch.int32, device=self.device,
                ).cumsum(0)
                hidden_states = self._forward_varlen(
                    input_ids=flat_ids,
                    positions=flat_positions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max(lengths) if lengths else 0,
                )
                head_dtype = self.model.colbert_linear.weight.dtype
                hidden_states = hidden_states.to(head_dtype)
                if self.model_key == "bge_m3":
                    projected = self.model.norm(self.model.colbert_linear(hidden_states))
                    slice_offset = 1
                else:
                    projected = self.model.norm(self.model.colbert_linear(hidden_states))
                    slice_offset = 0

                # No explicit upcast here: ``head_dtype`` above is already fp32
                # (embedder_loader keeps colbert_linear in fp32 for both
                # models), so the old ``.float()`` was a no-op. Left out rather
                # than kept as a misleading hint that a conversion happens.

                if copy_stream is not None:
                    # Stage the D2H through a BOUNDED ring of pinned buffers,
                    # then copy each request's rows out into ordinary pageable
                    # memory. The previous version allocated a fresh pinned
                    # buffer per batch and held every one of them until the
                    # whole encode finished, so a 1000-document bge-m3 run
                    # page-locked ~18.7 GiB (4.57M tokens x 1024 dims x fp32)
                    # inside the timed region. That cost, and its variance --
                    # repeat runs of the same workload ranged 10.7-16.9 s --
                    # vanished when the buffers were reused: driving the same
                    # 333 batches one encode() call at a time, which frees each
                    # buffer immediately, measured 7.49 s on both the first and
                    # second pass, identically. A ring rather than a single
                    # buffer keeps the D2H of one batch overlapped with the next
                    # batch's compute.
                    slot = self._acquire_pin_slot(
                        projected.shape, projected.dtype, results,
                    )
                    copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(copy_stream):
                        slot["buf"][:projected.shape[0]].copy_(
                            projected, non_blocking=True,
                        )
                        slot["event"].record(copy_stream)
                    slot["pending"] = (
                        list(batch_indices),
                        [list(token_ids) for token_ids in prompt_token_ids],
                        list(lengths),
                        slice_offset,
                    )
                    continue

                for local_idx, (request_idx, token_ids) in enumerate(
                    zip(batch_indices, prompt_token_ids, strict=True),
                ):
                    start = int(cu_seqlens[local_idx].item()) + slice_offset
                    end = int(cu_seqlens[local_idx + 1].item())
                    results[request_idx] = PoolingRequestOutput(
                        request_id=str(request_idx),
                        outputs=PoolingOutput(data=projected[start:end].detach()),
                        prompt_token_ids=list(token_ids),
                        num_cached_tokens=0,
                        finished=True,
                    )

        if copy_stream is not None:
            for slot in self._pin_ring:
                self._drain_pin_slot(slot, results)

        return [item for item in results if item is not None]

    # --- bounded pinned-staging ring -------------------------------------
    _PIN_RING_SIZE = 3

    def _acquire_pin_slot(self, shape, dtype, results):
        """Return a free ring slot whose pinned buffer holds ``shape``.

        Drains the slot first if it still carries a batch, which is what keeps
        total page-locked memory at ``_PIN_RING_SIZE`` buffers instead of one
        per batch.
        """
        ring = getattr(self, "_pin_ring", None)
        if ring is None:
            ring = self._pin_ring = [
                {"buf": None, "event": torch.cuda.Event(), "pending": None}
                for _ in range(self._PIN_RING_SIZE)
            ]
            self._pin_cursor = 0
        slot = ring[self._pin_cursor]
        self._pin_cursor = (self._pin_cursor + 1) % len(ring)
        self._drain_pin_slot(slot, results)

        buf = slot["buf"]
        rows, dim = int(shape[0]), int(shape[1])
        if (buf is None or buf.shape[0] < rows or buf.shape[1] != dim
                or buf.dtype != dtype):
            # Allocated at a bucketed row count so the handful of distinct
            # sizes are page-locked once and then reused, rather than once per
            # batch's exact token count.
            slot["buf"] = torch.empty(
                (self._bucket_tokens(rows), dim),
                dtype=dtype, device="cpu", pin_memory=True,
            )
        return slot

    def _drain_pin_slot(self, slot, results):
        """Copy a staged batch out of pinned memory into pageable results."""
        pending = slot["pending"]
        if pending is None:
            return
        slot["pending"] = None
        batch_indices, prompt_token_ids, lengths, slice_offset = pending
        slot["event"].synchronize()
        buf = slot["buf"]
        start = 0
        for request_idx, token_ids, length in zip(
            batch_indices, prompt_token_ids, lengths, strict=True,
        ):
            end = start + length
            src = buf[start + slice_offset:end]
            # Pageable copy: the result must outlive the pinned buffer, which
            # is about to be handed to another batch.
            out = torch.empty(src.shape, dtype=src.dtype, device="cpu")
            out.copy_(src)
            results[request_idx] = PoolingRequestOutput(
                request_id=str(request_idx),
                outputs=PoolingOutput(data=out),
                prompt_token_ids=list(token_ids),
                num_cached_tokens=0,
                finished=True,
            )
            start = end
