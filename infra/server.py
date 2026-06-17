"""
OpenAI-compatible HTTP server for kb_nano.

Serves /v1/chat/completions, /v1/completions, and /v1/models backed by
the LlamaEngine.  Supports chat templates (via HuggingFace tokenizer),
multimodal message content (forward-compatible with Qwen VL), tensor
parallelism, and SSE streaming.

Usage:
    python -m kb_nano.infra.server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8000 --tp 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic protocol models
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kb-nano"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# -- Multimodal content parts -----------------------------------------------

class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Union[ContentPartText, ContentPartImage]


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None


# -- Requests ----------------------------------------------------------------

class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    n: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    ignore_eos: bool = False
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: Union[str, List[str], List[int], List[List[int]]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 16
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    n: int = 1
    echo: bool = False
    ignore_eos: bool = False
    user: Optional[str] = None


# -- Response models ---------------------------------------------------------

class ChatMessageResponse(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class CompletionStreamChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionStreamChoice]


# ---------------------------------------------------------------------------
# Globals filled at startup
# ---------------------------------------------------------------------------

logger = logging.getLogger("kb_nano.server")

app = FastAPI(title="kb-nano OpenAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None          # LlamaEngine instance
_model_name: str = ""   # HF model identifier
_tokenizer = None       # HF tokenizer (same as engine.tokenizer)
_thread_pool: ThreadPoolExecutor | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _extract_text_from_content(
    content: Union[str, List[ContentPart], None],
) -> tuple[str, list[str]]:
    """Return (text, image_urls) from OpenAI-style message content.

    For string content, returns the string directly.
    For multimodal list content, concatenates text parts and collects image URLs.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    text_parts: list[str] = []
    image_urls: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    image_urls.append(url)
        elif hasattr(part, "type"):
            if part.type == "text":
                text_parts.append(part.text)
            elif part.type == "image_url":
                image_urls.append(part.image_url.url)
    return "\n".join(text_parts), image_urls


def _messages_to_prompt(messages: List[ChatMessage]) -> tuple[list[int], int]:
    """Apply the tokenizer's chat template and return (token_ids, num_prompt_tokens).

    Falls back to a simple concatenation if no chat template is available.
    """
    dicts: list[dict[str, Any]] = []
    all_image_urls: list[str] = []
    for msg in messages:
        text, img_urls = _extract_text_from_content(msg.content)
        dicts.append({"role": msg.role, "content": text})
        all_image_urls.extend(img_urls)

    try:
        token_ids = _tokenizer.apply_chat_template(
            dicts,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    except Exception:
        # Fallback: manual concatenation
        prompt_str = ""
        for d in dicts:
            prompt_str += f"<|{d['role']}|>\n{d['content']}\n"
        prompt_str += "<|assistant|>\n"
        token_ids = _tokenizer.encode(prompt_str)

    return token_ids, len(token_ids)


def _build_sampling_params(
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    seed: Optional[int],
    ignore_eos: bool = False,
):
    """Build a kb_nano SamplingParams from request fields."""
    from kb_nano.infra.engine import SamplingParams

    return SamplingParams(
        temperature=temperature if temperature is not None else 0.0,
        top_p=top_p if top_p is not None else 1.0,
        max_tokens=max_tokens if max_tokens is not None else 512,
        seed=seed,
        ignore_eos=ignore_eos,
    )


async def _generate_async(prompts, sampling_params):
    """Run engine.generate in a thread so we don't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _thread_pool,
        _engine.generate,
        prompts,
        sampling_params,
    )


def _error_json(message: str, status: int = 400) -> JSONResponse:
    body = ErrorResponse(message=message, code=str(status))
    return JSONResponse(content=body.model_dump(), status_code=status)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    card = ModelCard(id=_model_name)
    return ModelList(data=[card])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw: Request):
    if not request.messages:
        return _error_json("messages must not be empty")

    max_tokens = request.max_completion_tokens or request.max_tokens

    token_ids, num_prompt_tokens = _messages_to_prompt(request.messages)
    sp = _build_sampling_params(
        request.temperature, request.top_p, max_tokens, request.seed,
        request.ignore_eos,
    )

    request_id = _make_id("chatcmpl")
    created = int(time.time())

    include_usage = (
        request.stream_options is not None
        and request.stream_options.include_usage
    )

    if request.stream:
        return StreamingResponse(
            _stream_chat(request_id, created, request.model, token_ids,
                         num_prompt_tokens, sp, include_usage),
            media_type="text/event-stream",
        )

    outputs = await _generate_async([token_ids], sp)
    out = outputs[0]

    eos_id = _tokenizer.eos_token_id
    finish = "stop"
    if len(out.token_ids) >= sp.max_tokens:
        finish = "length"

    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessageResponse(content=out.generated_text),
        finish_reason=finish,
    )
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=len(out.token_ids),
        total_tokens=num_prompt_tokens + len(out.token_ids),
    )
    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=request.model or _model_name,
        choices=[choice],
        usage=usage,
    )


async def _stream_chat(
    request_id: str,
    created: int,
    model: str,
    token_ids: list[int],
    num_prompt_tokens: int,
    sp,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """SSE generator for streaming chat completions."""
    resolved_model = model or _model_name

    first = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=resolved_model,
        choices=[ChatCompletionStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=""),
            finish_reason=None,
        )],
    )
    yield f"data: {first.model_dump_json()}\n\n"

    outputs = await _generate_async([token_ids], sp)
    out = outputs[0]

    generated_ids = out.token_ids
    text_so_far = ""
    for i, tid in enumerate(generated_ids):
        partial = _tokenizer.decode(generated_ids[: i + 1], skip_special_tokens=True)
        delta_text = partial[len(text_so_far):]
        text_so_far = partial
        if not delta_text:
            continue
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=resolved_model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(content=delta_text),
                finish_reason=None,
            )],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    finish = "stop" if len(generated_ids) < sp.max_tokens else "length"
    final_data: Dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": resolved_model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish,
        }],
    }
    if include_usage:
        num_completion_tokens = len(generated_ids)
        final_data["usage"] = {
            "prompt_tokens": num_prompt_tokens,
            "completion_tokens": num_completion_tokens,
            "total_tokens": num_prompt_tokens + num_completion_tokens,
        }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest, raw: Request):
    prompts: list
    if isinstance(request.prompt, str):
        prompts = [request.prompt]
    elif isinstance(request.prompt, list):
        if request.prompt and isinstance(request.prompt[0], int):
            prompts = [request.prompt]  # single token-id list
        else:
            prompts = request.prompt  # list of strings or list of token-id lists
    else:
        return _error_json("invalid prompt type")

    sp = _build_sampling_params(
        request.temperature, request.top_p, request.max_tokens, request.seed,
        request.ignore_eos,
    )

    request_id = _make_id("cmpl")
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_completions(request_id, created, request.model, prompts, sp),
            media_type="text/event-stream",
        )

    outputs = await _generate_async(prompts, sp)

    choices = []
    total_prompt = 0
    total_completion = 0
    for i, out in enumerate(outputs):
        prompt_text = prompts[i] if isinstance(prompts[i], str) else ""
        prompt_toks = (
            len(prompts[i])
            if isinstance(prompts[i], list)
            else len(_tokenizer.encode(prompts[i]))
        )
        total_prompt += prompt_toks
        total_completion += len(out.token_ids)
        text = out.generated_text
        if request.echo and isinstance(prompts[i], str):
            text = prompts[i] + text
        finish = "stop" if len(out.token_ids) < sp.max_tokens else "length"
        choices.append(CompletionChoice(index=i, text=text, finish_reason=finish))

    return CompletionResponse(
        id=request_id,
        created=created,
        model=request.model or _model_name,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        ),
    )


async def _stream_completions(
    request_id: str,
    created: int,
    model: str,
    prompts: list,
    sp,
) -> AsyncIterator[str]:
    """SSE generator for streaming text completions."""
    outputs = await _generate_async(prompts, sp)

    for idx, out in enumerate(outputs):
        generated_ids = out.token_ids
        text_so_far = ""
        for i, tid in enumerate(generated_ids):
            partial = _tokenizer.decode(
                generated_ids[: i + 1], skip_special_tokens=True,
            )
            delta_text = partial[len(text_so_far):]
            text_so_far = partial
            if not delta_text:
                continue
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                model=model or _model_name,
                choices=[CompletionStreamChoice(
                    index=idx, text=delta_text, finish_reason=None,
                )],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        finish = "stop" if len(generated_ids) < sp.max_tokens else "length"
        final = CompletionStreamResponse(
            id=request_id,
            created=created,
            model=model or _model_name,
            choices=[CompletionStreamChoice(
                index=idx, text="", finish_reason=finish,
            )],
        )
        yield f"data: {final.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# CLI & startup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kb-nano OpenAI-compatible server")
    p.add_argument("--model", type=str, required=True,
                    help="HuggingFace model name (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graph capture")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-candidate-kernels", action="store_true", default=False,
                    help="Disable candidate kernel auto-detection; use only baseline kernels")
    return p.parse_args()


def main():
    global _engine, _model_name, _tokenizer, _thread_pool

    args = parse_args()
    _model_name = args.model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Add project root to path so kb_nano is importable
    from kb_nano import PROJECT_ROOT
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if not args.no_candidate_kernels:
        from kb_nano.infra.kernel_swapper import (
            apply_candidates,
            discover_candidates,
            print_candidate_summary,
        )
        candidates = discover_candidates()
        if candidates:
            print_candidate_summary(candidates)
            apply_candidates(candidates)

    from kb_nano.infra.engine import LlamaEngine

    logger.info("=" * 60)
    logger.info("  kb-nano OpenAI-compatible server")
    logger.info("=" * 60)
    logger.info(f"  Model:  {args.model}")
    logger.info(f"  TP:     {args.tp}")
    logger.info(f"  dtype:  {args.dtype}")
    logger.info(f"  Host:   {args.host}:{args.port}")
    logger.info("=" * 60)

    logger.info("Loading model …")
    _engine = LlamaEngine(
        model_name=args.model,
        dtype=dtype_map[args.dtype],
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
    )
    _tokenizer = _engine.tokenizer
    _thread_pool = ThreadPoolExecutor(max_workers=1)

    logger.info("Model loaded. Starting server …")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
