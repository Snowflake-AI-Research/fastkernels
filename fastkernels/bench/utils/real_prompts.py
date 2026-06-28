"""Real chat-prompt workloads for LLM benchmarking.

The workload datasets store raw chat messages. Benchmark runners call this
module with the target model tokenizer so prompts are chat-templated and
tokenized exactly as that model expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


DEFAULT_WORKLOAD_DATASETS: dict[str, str] = {
    "prefill-heavy": "sfc-gh-goliaro/wildchat-fastkernels-prefill-heavy-1k",
    "balanced": "sfc-gh-goliaro/wildchat-fastkernels-balanced-1k",
    "decode-heavy": "sfc-gh-goliaro/wildchat-fastkernels-decode-heavy-1k",
}

LEGACY_WORKLOAD_DATASETS: dict[str, str] = {
    "prefill-heavy": "sfc-gh-goliaro/wildchat-prefill-heavy-1k",
    "balanced": "sfc-gh-goliaro/kb-nano-balanced",
    "decode-heavy": "sfc-gh-goliaro/wildchat-decode-heavy-1k",
}


@dataclass(frozen=True)
class RealPromptSample:
    prompt_token_ids: list[int]
    output_len: int


def _normalize_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    if "user" in row and "assistant" in row:
        messages: list[dict[str, str]] = []
        system = row.get("system") or ""
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": row["user"] or ""})
        return messages, row["assistant"] or ""

    if "messages" in row and row["messages"] is not None:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in row["messages"]
        ]
        assistant_text = row.get("assistant_text")
        if assistant_text is None:
            for message in reversed(messages):
                if message["role"] == "assistant":
                    assistant_text = message["content"]
                    break
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        return messages, assistant_text or ""

    conversations = row.get("conversations")
    if conversations is None:
        raise ValueError("Expected either 'messages' or 'conversations' in row")

    prompt_messages: list[dict[str, str]] = []
    assistant_text = ""
    for turn in conversations:
        role = turn.get("role") or turn.get("from")
        content = turn.get("content") or turn.get("value") or ""
        if role in ("human", "user"):
            prompt_messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            assistant_text = content

    return prompt_messages, assistant_text


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError):
        text = "\n\n".join(m["content"] for m in messages)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(
                "Expected one chat-templated prompt, got a batched input_ids result"
            )
        token_ids = token_ids[0]
    return list(token_ids)


def _tokenize_response(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def load_real_prompt_workload(
    scenario_name: str,
    tokenizer: Any,
    *,
    num_requests: int = 1000,
    decode_cap: int | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    seed: int | None = None,
) -> list[RealPromptSample]:
    """Load, chat-template, and tokenize a real LLM workload.

    ``decode_cap`` caps the per-request generation budget after tokenizing the
    source assistant response with the same tokenizer.
    """
    from datasets import load_dataset

    dataset_id = dataset_name or DEFAULT_WORKLOAD_DATASETS[scenario_name]
    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception:
        legacy_id = LEGACY_WORKLOAD_DATASETS.get(scenario_name)
        if legacy_id is None or legacy_id == dataset_id:
            raise
        ds = load_dataset(legacy_id, split=split)

    if seed is not None:
        ds = ds.shuffle(seed=seed)

    samples: list[RealPromptSample] = []
    for row in ds:
        messages, assistant_text = _normalize_messages(row)
        prompt_ids = _apply_chat_template(tokenizer, messages)

        response_ids = _tokenize_response(tokenizer, assistant_text)
        output_len = len(response_ids)
        if decode_cap is not None:
            output_len = min(output_len, decode_cap)
        output_len = max(1, output_len)

        samples.append(RealPromptSample(
            prompt_token_ids=prompt_ids,
            output_len=output_len,
        ))
        if len(samples) >= num_requests:
            break

    if len(samples) < num_requests:
        raise ValueError(
            f"{dataset_id} yielded only {len(samples)} requests; "
            f"needed {num_requests}"
        )
    return samples
