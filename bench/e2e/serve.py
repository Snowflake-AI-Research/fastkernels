"""Online serving benchmark for fastkernels.

Modeled after ``vllm bench serve``. Sends async HTTP requests to a running
OpenAI-compatible server (``fastkernels.infra.server``) and measures TTFT, TPOT,
ITL, E2E latency, and throughput.

Prerequisites:
    Start the server first:
        python -m fastkernels.infra.server --model meta-llama/Llama-3.1-8B-Instruct

Usage:
    python -m fastkernels.bench.e2e serve \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --base-url http://localhost:8000 \\
        --dataset-name random --random-input-len 512 --random-output-len 128 \\
        --num-prompts 100 --request-rate 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from fastkernels.bench.utils.datasets import (
    SampleRequest,
    add_dataset_parser,
    get_samples,
)
from fastkernels.infra.kernel_swapper import discover_candidates, print_candidate_summary


@dataclass
class RequestResult:
    success: bool = False
    prompt_len: int = 0
    output_len: int = 0
    ttft: float = 0.0
    latency: float = 0.0
    itl: list[float] = field(default_factory=list)
    generated_text: str = ""
    error: str = ""


@dataclass
class BenchmarkMetrics:
    completed: int = 0
    failed: int = 0
    total_input: int = 0
    total_output: int = 0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0


def _build_sampling_payload(args: argparse.Namespace) -> dict:
    """Build the sampling parameter portion of the request payload.

    Only includes parameters that are explicitly set (non-None), matching
    vLLM's approach of letting the server use its own defaults otherwise.
    """
    mapping = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": getattr(args, "top_k", None),
        "min_p": getattr(args, "min_p", None),
        "frequency_penalty": getattr(args, "frequency_penalty", None),
        "presence_penalty": getattr(args, "presence_penalty", None),
        "repetition_penalty": getattr(args, "repetition_penalty", None),
    }
    return {k: v for k, v in mapping.items() if v is not None}


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    request: SampleRequest,
    sampling_params: dict,
    ignore_eos: bool,
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a single streaming chat completion request and measure timing."""
    result = RequestResult(prompt_len=request.prompt_len)

    if isinstance(request.prompt, str):
        messages = [{"role": "user", "content": request.prompt}]
    elif isinstance(request.prompt, list):
        messages = request.prompt
    else:
        messages = [{"role": "user", "content": str(request.prompt)}]

    if request.multi_modal_data and isinstance(request.multi_modal_data, dict):
        images = request.multi_modal_data.get("image", [])
        if images:
            content_parts = []
            if isinstance(messages[-1].get("content"), str):
                content_parts.append({
                    "type": "text",
                    "text": messages[-1]["content"],
                })
            for img in (images if isinstance(images, list) else [images]):
                if hasattr(img, "tobytes"):
                    import base64
                    import io
                    from PIL import Image
                    buf = io.BytesIO()
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(np.array(img))
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
            if content_parts:
                messages[-1] = {**messages[-1], "content": content_parts}

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": request.expected_output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sampling_params,
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    try:
        start_time = time.perf_counter()
        first_token_time = None
        last_token_time = start_time
        generated_tokens = 0
        generated_text = ""
        usage_completion_tokens = None

        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                result.error = f"HTTP {response.status}: {error_text}"
                return result

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = data.get("usage")
                if usage:
                    usage_completion_tokens = usage.get("completion_tokens")

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        result.ttft = first_token_time - start_time
                    else:
                        result.itl.append(now - last_token_time)
                    last_token_time = now
                    generated_tokens += 1
                    generated_text += content

        end_time = time.perf_counter()
        result.latency = end_time - start_time
        if usage_completion_tokens is not None:
            result.output_len = usage_completion_tokens
        else:
            result.output_len = generated_tokens
        result.generated_text = generated_text
        result.success = True

    except Exception as e:
        result.error = str(e)

    if pbar is not None:
        pbar.update(1)

    return result


async def get_request_schedule(
    requests: list[SampleRequest],
    request_rate: float,
) -> list[float]:
    """Compute send times for requests following a Poisson process."""
    delays = []
    for i in range(len(requests)):
        if request_rate == float("inf"):
            delays.append(0.0)
        else:
            interval = np.random.exponential(1.0 / request_rate)
            delays.append(interval)
    cumulative = []
    total = 0.0
    for d in delays:
        total += d
        cumulative.append(total)
    return cumulative


def calculate_metrics(
    results: list[RequestResult],
    duration: float,
) -> BenchmarkMetrics:
    """Calculate benchmark metrics from request results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        return BenchmarkMetrics(failed=len(failed))

    ttfts = [r.ttft for r in successful if r.ttft > 0]
    e2els = [r.latency for r in successful]
    all_itls = []
    tpots = []
    for r in successful:
        all_itls.extend(r.itl)
        if r.output_len > 1:
            tpot = (r.latency - r.ttft) / (r.output_len - 1)
            tpots.append(tpot)

    total_input = sum(r.prompt_len for r in successful)
    total_output = sum(r.output_len for r in successful)

    def _stats(values: list[float], scale: float = 1000.0):
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(values) * scale
        return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr)), float(np.percentile(arr, 99))

    ttft_mean, ttft_med, ttft_std, ttft_p99 = _stats(ttfts)
    tpot_mean, tpot_med, tpot_std, tpot_p99 = _stats(tpots)
    itl_mean, itl_med, itl_std, itl_p99 = _stats(all_itls)
    e2el_mean, e2el_med, e2el_std, e2el_p99 = _stats(e2els)

    return BenchmarkMetrics(
        completed=len(successful),
        failed=len(failed),
        total_input=total_input,
        total_output=total_output,
        request_throughput=len(successful) / duration,
        output_throughput=total_output / duration,
        total_token_throughput=(total_input + total_output) / duration,
        mean_ttft_ms=ttft_mean,
        median_ttft_ms=ttft_med,
        std_ttft_ms=ttft_std,
        p99_ttft_ms=ttft_p99,
        mean_tpot_ms=tpot_mean,
        median_tpot_ms=tpot_med,
        std_tpot_ms=tpot_std,
        p99_tpot_ms=tpot_p99,
        mean_itl_ms=itl_mean,
        median_itl_ms=itl_med,
        std_itl_ms=itl_std,
        p99_itl_ms=itl_p99,
        mean_e2el_ms=e2el_mean,
        median_e2el_ms=e2el_med,
        std_e2el_ms=e2el_std,
        p99_e2el_ms=e2el_p99,
    )


async def run_benchmark(
    api_url: str,
    model: str,
    requests: list[SampleRequest],
    request_rate: float,
    sampling_params: dict,
    ignore_eos: bool,
) -> tuple[BenchmarkMetrics, list[RequestResult]]:
    """Run the online serving benchmark."""
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    try:
        print("Starting initial test request...")
        test_result = await send_request(
            session, api_url, model, requests[0],
            sampling_params, ignore_eos,
        )
        if not test_result.success:
            raise RuntimeError(
                f"Initial test request failed: {test_result.error}\n"
                f"Make sure the server is running at the correct URL."
            )
        print("Initial test request completed.")

        print(f"\nSending {len(requests)} requests at rate={request_rate} req/s...")
        schedule = await get_request_schedule(requests, request_rate)

        pbar = tqdm(total=len(requests), desc="Benchmark")
        tasks = []
        start_time = time.perf_counter()

        for i, req in enumerate(requests):
            wait_time = schedule[i] - (time.perf_counter() - start_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            task = asyncio.create_task(
                send_request(
                    session, api_url, model, req,
                    sampling_params, ignore_eos, pbar,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        pbar.close()

        duration = time.perf_counter() - start_time
        metrics = calculate_metrics(list(results), duration)
        return metrics, list(results)

    finally:
        await session.close()


def validate_args(args: argparse.Namespace):
    """Validate CLI arguments for the serve benchmark."""
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.top_p is not None and not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")
    top_k = getattr(args, "top_k", None)
    if top_k is not None and top_k < -1:
        raise ValueError("--top-k must be >= -1")
    min_p = getattr(args, "min_p", None)
    if min_p is not None and not (0 <= min_p <= 1.0):
        raise ValueError("--min-p must be in [0, 1]")

    if args.temperature is None:
        warnings.warn(
            "Temperature is not set. It will be server's default. "
            "Set --temperature explicitly for reproducible results.",
            stacklevel=2,
        )


def add_cli_args(parser: argparse.ArgumentParser):
    """Add serve-specific CLI arguments."""
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (must match the server's model). If not specified, "
             "fetches from the server.",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Tokenizer name or path (defaults to --model)",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--endpoint", type=str, default="/v1/chat/completions",
        help="API endpoint (default: /v1/chat/completions)",
    )
    parser.add_argument(
        "--request-rate", type=float, default=float("inf"),
        help="Requests per second (default: inf = send all at once)",
    )

    # Sampling parameters -- defaults are None (use server defaults),
    # matching vLLM serve bench behavior.
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: None = server default). "
             "Use 0.0 for greedy/deterministic",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Top-p (nucleus) sampling parameter (default: None = server default)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Top-k sampling parameter (default: None = server default)",
    )
    parser.add_argument(
        "--min-p", type=float, default=None,
        help="Min-p sampling parameter (default: None = server default)",
    )
    parser.add_argument(
        "--frequency-penalty", type=float, default=None,
        help="Frequency penalty (default: None = server default)",
    )
    parser.add_argument(
        "--presence-penalty", type=float, default=None,
        help="Presence penalty (default: None = server default)",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=None,
        help="Repetition penalty (default: None = server default)",
    )
    parser.add_argument(
        "--ignore-eos", action="store_true", default=False,
        help="Tell the server to ignore the EOS token and generate "
             "exactly max_tokens tokens",
    )

    parser.add_argument(
        "--input-len", type=int, default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len", type=int, default=None,
        help="Output length for each request. Overrides the "
             "output length from the dataset.",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save performance results in JSON format",
    )
    parser.add_argument(
        "--save-outputs", type=str, default=None,
        help="Path to save generated outputs alongside performance data",
    )
    parser.add_argument(
        "--no-candidate-kernels", action="store_true", default=False,
        help="Disable candidate kernel auto-detection; use only baseline kernels",
    )

    add_dataset_parser(parser)
    parser.set_defaults(seed=42)


async def _fetch_model_from_server(base_url: str) -> str:
    """Fetch the first available model from the server."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/v1/models") as resp:
            data = await resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
            raise RuntimeError("No models found on the server.")


def main(args: argparse.Namespace):
    asyncio.run(main_async(args))


async def main_async(args: argparse.Namespace):
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.no_candidate_kernels:
        candidates = discover_candidates()
        if candidates:
            names = ", ".join(
                f"{t.name} (L{t.level})" for t, _ in candidates
            )
            print(f"\nNOTE: Candidate kernels detected: {names}.")
            print("      These must be applied server-side. Ensure the server")
            print("      was started WITHOUT --no-candidate-kernels:")
            print("          python -m fastkernels.infra.server --model <model>\n")

    if args.model is None:
        print("Model not specified, fetching from server...")
        args.model = await _fetch_model_from_server(args.base_url)
        print(f"  Using model: {args.model}")

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    if args.input_len is not None:
        args.random_input_len = args.input_len
        args.sonnet_input_len = args.input_len
    if args.output_len is not None:
        args.random_output_len = args.output_len
        args.sonnet_output_len = args.output_len
        args.sharegpt_output_len = args.output_len
        args.custom_output_len = args.output_len
        args.hf_output_len = args.output_len
        args.spec_bench_output_len = args.output_len
        args.prefix_repetition_output_len = args.output_len

    if not hasattr(args, "backend"):
        args.backend = "openai-chat"
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    requests = get_samples(args, tokenizer)

    api_url = f"{args.base_url}{args.endpoint}"
    sampling_params = _build_sampling_payload(args)

    print("=" * 70)
    print("  fastkernels Serving Benchmark")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  Server         : {api_url}")
    print(f"  Requests       : {len(requests)}")
    print(f"  Request rate   : {args.request_rate}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Top-p          : {args.top_p}")
    print(f"  Ignore EOS     : {args.ignore_eos}")
    print(f"  Seed           : {args.seed}")
    print("=" * 70)

    metrics, results = await run_benchmark(
        api_url, args.model, requests, args.request_rate,
        sampling_params, args.ignore_eos,
    )

    print(f"\n{'=' * 50}")
    print(f"{'  Serving Benchmark Result':^50}")
    print(f"{'=' * 50}")
    print(f"  {'Successful requests:':<35} {metrics.completed}")
    print(f"  {'Failed requests:':<35} {metrics.failed}")
    print(f"  {'Total input tokens:':<35} {metrics.total_input}")
    print(f"  {'Total generated tokens:':<35} {metrics.total_output}")
    print(f"  {'Request throughput (req/s):':<35} {metrics.request_throughput:.2f}")
    print(f"  {'Output token throughput (tok/s):':<35} {metrics.output_throughput:.2f}")
    print(f"  {'Total token throughput (tok/s):':<35} {metrics.total_token_throughput:.2f}")

    print(f"\n{'-'*50}")
    print(f"{'  Time to First Token (TTFT)':^50}")
    print(f"{'-'*50}")
    print(f"  {'Mean TTFT (ms):':<35} {metrics.mean_ttft_ms:.2f}")
    print(f"  {'Median TTFT (ms):':<35} {metrics.median_ttft_ms:.2f}")
    print(f"  {'P99 TTFT (ms):':<35} {metrics.p99_ttft_ms:.2f}")

    print(f"\n{'-'*50}")
    print(f"{'  Time per Output Token (TPOT)':^50}")
    print(f"{'-'*50}")
    print(f"  {'Mean TPOT (ms):':<35} {metrics.mean_tpot_ms:.2f}")
    print(f"  {'Median TPOT (ms):':<35} {metrics.median_tpot_ms:.2f}")
    print(f"  {'P99 TPOT (ms):':<35} {metrics.p99_tpot_ms:.2f}")

    print(f"\n{'-'*50}")
    print(f"{'  Inter-token Latency (ITL)':^50}")
    print(f"{'-'*50}")
    print(f"  {'Mean ITL (ms):':<35} {metrics.mean_itl_ms:.2f}")
    print(f"  {'Median ITL (ms):':<35} {metrics.median_itl_ms:.2f}")
    print(f"  {'P99 ITL (ms):':<35} {metrics.p99_itl_ms:.2f}")

    print(f"\n{'-'*50}")
    print(f"{'  End-to-end Latency (E2EL)':^50}")
    print(f"{'-'*50}")
    print(f"  {'Mean E2EL (ms):':<35} {metrics.mean_e2el_ms:.2f}")
    print(f"  {'Median E2EL (ms):':<35} {metrics.median_e2el_ms:.2f}")
    print(f"  {'P99 E2EL (ms):':<35} {metrics.p99_e2el_ms:.2f}")
    print(f"{'=' * 50}")

    result_json = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ignore_eos": args.ignore_eos,
        "request_rate": args.request_rate if args.request_rate != float("inf") else "inf",
        "num_requests": len(requests),
        "completed": metrics.completed,
        "failed": metrics.failed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "mean_e2el_ms": metrics.mean_e2el_ms,
        "median_e2el_ms": metrics.median_e2el_ms,
        "std_e2el_ms": metrics.std_e2el_ms,
        "p99_e2el_ms": metrics.p99_e2el_ms,
    }

    # Log to MLflow
    from fastkernels.bench.tracking import tracker

    tracker.log_e2e(result_json, bench_type="serve")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")

    if args.save_outputs:
        output_data = {
            **result_json,
            "outputs": [
                {
                    "prompt_len": r.prompt_len,
                    "output_len": r.output_len,
                    "generated_text": r.generated_text,
                    "ttft": r.ttft,
                    "latency": r.latency,
                    "itl": r.itl,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
        }
        os.makedirs(os.path.dirname(args.save_outputs) or ".", exist_ok=True)
        with open(args.save_outputs, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Outputs saved to: {args.save_outputs}")
