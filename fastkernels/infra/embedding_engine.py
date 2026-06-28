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
            compile_model = self.device.type == "cuda"
        if compile_model:
            self._forward_varlen = torch.compile(
                self.model.forward_varlen,
                mode="reduce-overhead",
                dynamic=True,
            )

        torch.manual_seed(seed)

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
        pending_cpu_batches: list[
            tuple[torch.Tensor, list[int], list[list[int]], list[int], int]
        ] = []

        with torch.no_grad():
            for batch_indices in batch_iter:
                prompt_token_ids = [input_ids_list[idx] for idx in batch_indices]
                lengths = [len(ids) for ids in prompt_token_ids]
                flat_ids = torch.tensor(
                    [token_id for ids in prompt_token_ids for token_id in ids],
                    dtype=torch.long,
                    device=self.device,
                )
                flat_positions = torch.cat([
                    torch.arange(length, dtype=torch.long, device=self.device)
                    for length in lengths
                ])
                cu_seqlens = torch.zeros(
                    len(lengths) + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
                cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device=self.device).cumsum(0)
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

                projected = projected.float()

                if copy_stream is not None:
                    copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(copy_stream):
                        projected_cpu = torch.empty(
                            projected.shape,
                            dtype=projected.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        projected_cpu.copy_(projected, non_blocking=True)
                    pending_cpu_batches.append(
                        (
                            projected_cpu,
                            list(batch_indices),
                            [list(token_ids) for token_ids in prompt_token_ids],
                            list(lengths),
                            slice_offset,
                        ),
                    )
                    continue

                for local_idx, (request_idx, token_ids) in enumerate(
                    zip(batch_indices, prompt_token_ids, strict=True),
                ):
                    start = int(cu_seqlens[local_idx].item()) + slice_offset
                    end = int(cu_seqlens[local_idx + 1].item())
                    results[request_idx] = PoolingRequestOutput(
                        request_id=str(request_idx),
                        outputs=PoolingOutput(data=projected[start:end].detach().float()),
                        prompt_token_ids=list(token_ids),
                        num_cached_tokens=0,
                        finished=True,
                    )

        if copy_stream is not None:
            copy_stream.synchronize()
            for (
                projected_cpu,
                batch_indices,
                prompt_token_ids,
                lengths,
                slice_offset,
            ) in pending_cpu_batches:
                start = 0
                for request_idx, token_ids, length in zip(
                    batch_indices,
                    prompt_token_ids,
                    lengths,
                    strict=True,
                ):
                    end = start + length
                    results[request_idx] = PoolingRequestOutput(
                        request_id=str(request_idx),
                        outputs=PoolingOutput(data=projected_cpu[start + slice_offset:end]),
                        prompt_token_ids=list(token_ids),
                        num_cached_tokens=0,
                        finished=True,
                    )
                    start = end

        return [item for item in results if item is not None]
