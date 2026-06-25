#!/usr/bin/env python3
"""
Throughput, latency, and alignment benchmark: fastkernels JambaEngine vs vLLM (or HF) reference.

Mirrors the structure of ``tests/bench_fla.py``: each engine runs in its
own long-lived subprocess that processes all scenarios sequentially,
avoiding repeated model loading.

The reference engine is HuggingFace transformers' ``JambaForCausalLM``
driven via ``.generate()``.  By default we use HF's *fast* Mamba kernel
path (``use_mamba_kernels=True``), which dispatches to the same
``causal_conv1d_*`` / ``selective_*`` kernels that fastkernels consumes as
L1 ops.  This is the fair reference: both sides are pinned to the same
underlying CUDA primitives, and the only difference is the pipeline
wiring (fastkernels's flat-varlen JambaEngine vs HF's per-step
``.generate``).  Pass ``--ref-slow-mamba`` to fall back to HF's pure
PyTorch ``slow_forward`` (Python-loop) path -- only useful for
debugging / numerical comparison, not for honest perf claims.

vLLM Jamba would be a stronger reference but requires its own
serving setup; in this environment HF is the supported path.

Usage:
    python tests/bench_jamba.py --model ai21labs/Jamba-tiny-dev
    python tests/bench_jamba.py --model ai21labs/Jamba-v0.1
    python tests/bench_jamba.py --model ... --skip-ref   # fastkernels only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from fastkernels.bench.utils.worker import run_worker
from fastkernels.bench.utils.real_prompts import load_real_prompt_workload
from fastkernels.bench.utils.workloads import LATENCY_WORKLOADS, THROUGHPUT_WORKLOADS
from fastkernels.tests.bench_vllm import compute_alignment


SCENARIOS = [
    {
        "name": w.name,
        "dataset": w.dataset_name,
    }
    for w in THROUGHPUT_WORKLOADS
]

LATENCY_SCENARIOS = [
    {
        "name": w.name,
        "input_len": w.input_len,
        "output_len": w.output_len,
        "batch_size": w.batch_size,
    }
    for w in LATENCY_WORKLOADS
]


# ---------------------------------------------------------------------------
# HuggingFace transformers reference subprocess worker
# ---------------------------------------------------------------------------
HF_REF_WORKER = r'''
import json, os, sys, time

import torch
from transformers import AutoConfig, AutoTokenizer, JambaForCausalLM


def _load_jamba_ref(model_name, dtype, device, use_mamba_kernels=True):
    # By default HF uses its fused Mamba CUDA path (selective_scan_fn /
    # selective_state_update from mamba_ssm + causal_conv1d_fn / update
    # from causal_conv1d).  These are the same kernels fastkernels calls as
    # L1 ops, so the perf comparison is a fair pipeline-vs-pipeline
    # measurement.  ``use_mamba_kernels=False`` forces HF down its
    # Python-loop ``slow_forward`` -- not a fair perf reference, only
    # useful for debugging numerical drift.
    config = AutoConfig.from_pretrained(model_name)
    config.use_mamba_kernels = use_mamba_kernels
    model = JambaForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    return model


def _batched_generate(model, tokenizer, prompts, output_lens, eos, device,
                      ignore_eos=True):
    """Pad-batch prompts and run a single .generate() call."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        eos if eos is not None else 0
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_id

    max_len = max(len(p) for p in prompts)
    input_ids = torch.full(
        (len(prompts), max_len), pad_id, dtype=torch.long, device=device,
    )
    attn = torch.zeros(
        (len(prompts), max_len), dtype=torch.long, device=device,
    )
    for i, p in enumerate(prompts):
        input_ids[i, max_len - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        attn[i, max_len - len(p):] = 1

    max_new = max(output_lens)
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
        pad_token_id=pad_id,
    )
    if not ignore_eos and eos is not None:
        gen_kwargs["eos_token_id"] = eos

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
    gen = out[:, max_len:]
    rows = [row.tolist()[:ol] for row, ol in zip(gen, output_lens)]
    return rows


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    device = "cuda"
    dtype = torch.bfloat16
    model_name = cfg["model"]
    micro_batch = cfg.get("ref_max_num_seqs", 8)

    use_mamba_kernels = cfg.get("use_mamba_kernels", True)
    print(
        f"  [HF reference] loading {model_name} "
        f"(use_mamba_kernels={use_mamba_kernels})...",
        flush=True,
    )
    model = _load_jamba_ref(
        model_name, dtype, device, use_mamba_kernels=use_mamba_kernels,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos = tokenizer.eos_token_id

    # Warmup
    print(f"  [HF reference] warmup (mb={micro_batch})...", flush=True)
    _batched_generate(model, tokenizer, [[1, 2, 3, 4]], [16], eos, device, ignore_eos=True)
    torch.cuda.synchronize()

    scenarios = cfg["scenarios"]
    all_results = []
    for sc in scenarios:
        prompts = sc["prompt_token_ids"]
        out_lens = sc["output_lens"]
        max_out = max(out_lens) if out_lens else cfg.get("default_output_len", 128)
        print(
            f"  [HF reference] throughput {sc['name']}: "
            f"{len(prompts)} requests, avg_out={sum(out_lens) / max(len(out_lens), 1):.1f}",
            flush=True,
        )

        # Process in micro-batches.  HF's .generate has no continuous
        # batching; we cap concurrency at ``ref_max_num_seqs`` so the
        # reference does not OOM.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen_tokens = []
        for start in range(0, len(prompts), micro_batch):
            end = min(start + micro_batch, len(prompts))
            batch_prompts = prompts[start:end]
            batch_out_lens = out_lens[start:end]
            batch_gen = _batched_generate(
                model, tokenizer, batch_prompts, batch_out_lens,
                eos, device, ignore_eos=True,
            )
            gen_tokens.extend(batch_gen)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_in = sum(len(p) for p in prompts)
        total_out = sum(len(g) for g in gen_tokens)
        tps = total_out / elapsed if elapsed else 0.0
        print(
            f"  [HF reference] throughput {sc['name']} done: "
            f"{elapsed:.2f}s, {tps:,.0f} tok/s",
            flush=True,
        )
        all_results.append({
            "name": sc["name"],
            "elapsed": elapsed,
            "total_prompt_tokens": total_in,
            "total_output_tokens": total_out,
            "outputs": [
                {"text": tokenizer.decode(g, skip_special_tokens=True),
                 "token_ids": g}
                for g in gen_tokens
            ],
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompt_token_ids"]
        out_lens = ls["output_lens"]
        out_len = max(out_lens)
        print(
            f"  [HF reference] latency {ls['name']}: "
            f"bs={len(prompts)}, max_out={out_len}",
            flush=True,
        )
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 3)
        for _ in range(num_warmup):
            _batched_generate(model, tokenizer, prompts, out_lens, eos, device, ignore_eos=True)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _batched_generate(model, tokenizer, prompts, out_lens, eos, device, ignore_eos=True)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        print(
            f"  [HF reference] latency {ls['name']} done: "
            f"median={sorted(latencies)[len(latencies)//2]:.4f}s",
            flush=True,
        )
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": out_len,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del model
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# vLLM reference subprocess worker (preferred reference: continuous batching,
# paged-KV, CUDA-graph captured decode -- the apples-to-apples SOTA setup vs
# fastkernels's own continuous-batched JambaEngine)
# ---------------------------------------------------------------------------
VLLM_REF_WORKER = r'''
import json, os, sys, time


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model_name = cfg["model"]
    print(f"  [vLLM reference] loading {model_name} (dtype=bfloat16)...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.get("ref_gpu_memory_utilization", 0.6),
        max_model_len=cfg.get("ref_max_model_len") or 4096,
        max_num_seqs=cfg.get("ref_max_num_seqs", 32),
        seed=cfg["seed"],
        trust_remote_code=True,
        enforce_eager=False,  # vLLM uses CUDA-graph capture by default; keep it
    )
    load_s = time.time() - t0
    print(f"  [vLLM reference] loaded in {load_s:.1f}s", flush=True)

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    for sc in scenarios:
        prompts_ids = sc["prompt_token_ids"]
        out_lens = sc["output_lens"]
        max_out = max(out_lens) if out_lens else cfg.get("default_output_len", 128)

        # vLLM accepts pre-tokenized prompts via TokensPrompt; per-prompt
        # SamplingParams so each request honors its own ``max_tokens``.
        inputs = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts_ids]
        sp_list = [
            SamplingParams(temperature=cfg.get("temperature", 0.0),
                           top_p=1.0, max_tokens=ol, ignore_eos=True)
            for ol in out_lens
        ]

        print(
            f"  [vLLM reference] throughput {sc['name']}: "
            f"{len(prompts_ids)} requests, max_out={max_out}", flush=True,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(inputs, sp_list, use_tqdm=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_tokens = []
        gen_texts = []
        for o in outputs:
            tok_ids = list(o.outputs[0].token_ids)
            gen_tokens.append(tok_ids)
            gen_texts.append(o.outputs[0].text)

        total_in = sum(len(p) for p in prompts_ids)
        total_out = sum(len(g) for g in gen_tokens)
        tps = total_out / elapsed if elapsed else 0.0
        print(
            f"  [vLLM reference] throughput {sc['name']} done: "
            f"{elapsed:.2f}s, {tps:,.0f} tok/s", flush=True,
        )
        all_results.append({
            "name": sc["name"],
            "elapsed": elapsed,
            "total_prompt_tokens": total_in,
            "total_output_tokens": total_out,
            "outputs": [
                {"text": t, "token_ids": ids}
                for t, ids in zip(gen_texts, gen_tokens)
            ],
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts_ids = ls["prompt_token_ids"]
        out_lens = ls["output_lens"]
        out_len = max(out_lens)
        bs = len(prompts_ids)
        inputs = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts_ids]
        sp_list = [
            SamplingParams(temperature=cfg.get("temperature", 0.0),
                           top_p=1.0, max_tokens=ol, ignore_eos=True)
            for ol in out_lens
        ]
        print(
            f"  [vLLM reference] latency {ls['name']}: bs={bs}, max_out={out_len}",
            flush=True,
        )
        num_warmup = ls.get("num_warmup", 1)
        num_iters = ls.get("num_iters", 3)
        for _ in range(num_warmup):
            llm.generate(inputs, sp_list, use_tqdm=False)
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(inputs, sp_list, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        print(
            f"  [vLLM reference] latency {ls['name']} done: "
            f"median={sorted(latencies)[len(latencies)//2]:.4f}s",
            flush=True,
        )
        latency_results.append({
            "name": ls["name"],
            "batch_size": bs,
            "input_len": ls["input_len"],
            "output_len": out_len,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del llm
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# fastkernels JambaEngine subprocess worker
# ---------------------------------------------------------------------------
FASTKERNELS_JAMBA_WORKER = r'''
import json, os, sys, time


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    mod = __import__(
        f"{pkg}.infra.jamba_engine",
        fromlist=["JambaEngine", "SamplingParams"],
    )
    JambaEngine, SamplingParams = mod.JambaEngine, mod.SamplingParams

    engine = JambaEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        max_num_seqs=cfg.get("max_num_seqs", 32),
    )

    # Warmup
    engine.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    for sc in scenarios:
        prompts = sc["prompt_token_ids"]
        output_lens = sc["output_lens"]
        temperature = cfg.get("temperature", 0.0)

        sp_list = [
            SamplingParams(
                temperature=temperature,
                max_tokens=ol,
                ignore_eos=True,
            )
            for ol in output_lens
        ]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sp_list, use_tqdm=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_in = sum(len(p) for p in prompts)
        total_out = sum(len(o.token_ids) for o in outputs)
        all_results.append({
            "name": sc["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "outputs": [
                {"generated_text": o.generated_text, "token_ids": o.token_ids}
                for o in outputs
            ],
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompt_token_ids"]
        out_lens = ls["output_lens"]
        sp = [
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
            for ol in out_lens
        ]
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 3)
        for _ in range(num_warmup):
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine


if __name__ == "__main__":
    main()
'''


SUPPORTED_MODELS = {
    "ai21labs/Jamba-tiny-dev",
    "ai21labs/Jamba-v0.1",
    "ai21labs/AI21-Jamba-Mini-1.7",
    "ai21labs/AI21-Jamba-1.5-Mini",
}


def _model_max_position_embeddings(model_name: str) -> int | None:
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        return None
    value = config.get("max_position_embeddings")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fit_prompt_to_context(
    prompt_token_ids: list[int],
    output_len: int,
    max_model_len: int | None,
) -> tuple[list[int], int]:
    if max_model_len is None:
        return prompt_token_ids, output_len
    if max_model_len < 2:
        raise SystemExit(f"Model max context is too small: {max_model_len}")
    output_len = min(output_len, max_model_len - 1)
    prompt_budget = max_model_len - output_len
    if len(prompt_token_ids) > prompt_budget:
        prompt_token_ids = prompt_token_ids[-prompt_budget:]
    return prompt_token_ids, output_len


def main():
    parser = argparse.ArgumentParser(
        description="Throughput & alignment benchmark: fastkernels JambaEngine vs vLLM (or HF) reference",
    )
    parser.add_argument("--model", type=str, default="ai21labs/Jamba-tiny-dev")
    parser.add_argument("--num-seqs", type=int, default=1000,
                        help="Sequences per throughput scenario.  Default 1000 "
                             "matches bench_vllm.py and bench_fla.py.  Use "
                             "--num-seqs 200 with --ref hf if you want to "
                             "compare against HF's slow .generate() loop.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Max concurrent sequences in JambaEngine.  "
                             "Default 256 matches bench_vllm.py defaults and "
                             "the bench_mamba README.  Lower this if v0.1 "
                             "OOMs at full per-seq Mamba state.")
    parser.add_argument(
        "--ref", type=str, choices=["vllm", "hf"], default="vllm",
        help=(
            "Reference engine for the bench. 'vllm' (default) runs vLLM's "
            "production Jamba server (continuous batching, paged-KV, "
            "flashinfer kernels) — the apples-to-apples SOTA reference. "
            "'hf' runs HuggingFace transformers' .generate() loop (no "
            "continuous batching; useful for token-level alignment "
            "comparisons but not for fair throughput numbers)."
        ),
    )
    parser.add_argument("--ref-max-num-seqs", type=int, default=256,
                        help="Max concurrent sequences in the reference "
                             "engine (vLLM continuous-batched scheduler) "
                             "or max micro-batch (HF .generate). Default "
                             "256 matches fastkernels's max_num_seqs default "
                             "and bench_vllm.py.")
    parser.add_argument("--ref-gpu-memory-utilization", type=float, default=0.9,
                        help="vLLM-only: fraction of GPU memory vLLM may "
                             "use.  Default 0.9 matches bench_vllm.py.")
    parser.add_argument("--ref-max-model-len", type=int, default=None,
                        help="vLLM-only: max_model_len for vLLM. Defaults "
                             "to the model's max_position_embeddings.")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024,
                        help="Cap each prompt to at most this many tokens "
                             "(left-truncated).")
    parser.add_argument("--max-output-tokens", type=int, default=256,
                        help="Cap each scenario's output_len to this value.")
    parser.add_argument("--skip-ref", action="store_true",
                        help="Skip the HF reference (fastkernels only)")
    parser.add_argument(
        "--ref-slow-mamba", action="store_true",
        help=(
            "Force HF to use its Python-loop slow_forward Mamba path "
            "(use_mamba_kernels=False).  Useful for debugging numerical "
            "drift; NOT a fair perf reference, since slow_forward runs a "
            "Python loop per token instead of a fused CUDA kernel."
        ),
    )
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-iters", type=int, default=5,
                        help="Median over this many timed iterations after "
                             "warmup.  Default 5 matches bench_vllm.py and "
                             "bench_fla.py.")
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run only the throughput scenario with this name "
             "(e.g. 'balanced'). Default: all scenarios.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.model not in SUPPORTED_MODELS:
        print(f"WARNING: {args.model!r} not in known list: {sorted(SUPPORTED_MODELS)}",
              file=sys.stderr)

    gpu = _detect_gpu_name()
    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(_PACKAGE_DIR / "tests" / "results" / gpu / f"{short}_jamba_tp1")

    throughput_scenarios = list(SCENARIOS)
    latency_scenarios = list(LATENCY_SCENARIOS)
    if args.scenario is not None:
        throughput_scenarios = [
            s for s in throughput_scenarios if s["name"] == args.scenario
        ]
        if not throughput_scenarios:
            raise SystemExit(f"--scenario={args.scenario!r} did not match any scenario")

    scenario_data = []
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_model_len = _model_max_position_embeddings(args.model)
    if not args.skip_throughput:
        for i, sc in enumerate(throughput_scenarios):
            samples = load_real_prompt_workload(
                sc["name"],
                tokenizer,
                num_requests=args.num_seqs,
                decode_cap=args.max_output_tokens,
                dataset_name=sc["dataset"],
                seed=args.seed + i,
            )
            fitted = []
            for s in samples:
                p, ol = _fit_prompt_to_context(
                    s.prompt_token_ids, s.output_len, max_model_len,
                )
                # Cap prompt length so HF reference does not OOM on
                # extremely long prefills.
                if len(p) > args.max_prompt_tokens:
                    p = p[-args.max_prompt_tokens:]
                ol = min(ol, args.max_output_tokens)
                fitted.append((p, ol))
            prompt_token_ids = [p for p, _ in fitted]
            output_lens = [ol for _, ol in fitted]
            scenario_data.append({
                "name": sc["name"],
                "prompt_token_ids": prompt_token_ids,
                "output_lens": output_lens,
            })

    latency_data = []
    if not args.skip_latency:
        for j, ls in enumerate(latency_scenarios):
            bs = ls["batch_size"]
            samples = load_real_prompt_workload(
                "balanced",
                tokenizer,
                num_requests=bs,
                decode_cap=ls["output_len"],
                seed=args.seed + 100 + j,
            )
            fitted = []
            for s in samples:
                p, ol = _fit_prompt_to_context(
                    s.prompt_token_ids, ls["output_len"], max_model_len,
                )
                if len(p) > args.max_prompt_tokens:
                    p = p[-args.max_prompt_tokens:]
                fitted.append((p, ol))
            prompt_token_ids = [p for p, _ in fitted]
            output_lens = [ol for _, ol in fitted]
            latency_data.append({
                "name": ls["name"],
                "input_len": ls["input_len"],
                "output_len": ls["output_len"],
                "batch_size": bs,
                "prompt_token_ids": prompt_token_ids,
                "output_lens": output_lens,
                "num_warmup": 1,
                "num_iters": args.latency_iters,
            })

    print("=" * 70)
    ref_name = "vLLM" if args.ref == "vllm" else "HF"
    print(f"  fastkernels JambaEngine vs {ref_name} reference -- Multi-Scenario Benchmark")
    print("=" * 70)
    print(f"  Model            : {args.model}")
    print(f"  Seqs/scenario    : {args.num_seqs}")
    print(f"  max_num_seqs     : {args.max_num_seqs}")
    print(f"  ref micro-batch  : {args.ref_max_num_seqs}")
    print(f"  max_prompt_tokens: {args.max_prompt_tokens}")
    print(f"  max_output_tokens: {args.max_output_tokens}")
    if max_model_len is not None:
        print(f"  Max model len    : {max_model_len}")
    print(f"  Temperature      : {args.temperature}")
    print(f"  GPU              : {gpu}")
    print(f"  Seed             : {args.seed}")
    if not args.skip_ref:
        if args.ref == "vllm":
            ref_label = (
                f"vLLM {args.ref} (continuous batching, paged KV, "
                f"max_num_seqs={args.ref_max_num_seqs})"
            )
        else:
            ref_label = (
                "HF slow_forward (Python loop)" if args.ref_slow_mamba
                else "HF fast Mamba kernels (causal_conv1d + mamba_ssm)"
            )
        print(f"  Reference engine : {ref_label}")
    print(f"  Output dir       : {args.output_dir}")
    if scenario_data:
        print(f"  Throughput       : {', '.join(s['name'] for s in throughput_scenarios)}")
    if latency_data:
        print(f"  Latency          : "
              f"{', '.join(s['name'] for s in latency_scenarios)} "
              f"({args.latency_iters} iters)")
    print("=" * 70)

    short_name = args.model.split("/")[-1]
    base_cfg = {
        "model": args.model,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_model_len": max_model_len,
        "scenarios": scenario_data,
        "latency_scenarios": latency_data,
    }

    ref_raw = None
    if not args.skip_ref:
        ref_cfg = dict(base_cfg)
        ref_cfg["ref_max_num_seqs"] = args.ref_max_num_seqs
        if args.ref == "vllm":
            ref_cfg["ref_gpu_memory_utilization"] = args.ref_gpu_memory_utilization
            ref_cfg["ref_max_model_len"] = args.ref_max_model_len
            worker_src = VLLM_REF_WORKER
            ref_label_short = f"vLLM reference [{short_name}]"
        else:
            ref_cfg["use_mamba_kernels"] = not args.ref_slow_mamba
            worker_src = HF_REF_WORKER
            ref_label_short = f"HF reference [{short_name}]"
        ref_raw = run_worker(
            worker_src, ref_cfg,
            f"{ref_label_short} all scenarios",
            timeout=21600,
        )

    kb_cfg = dict(base_cfg)
    kb_cfg["project_root"] = str(_PROJECT_ROOT)
    kb_cfg["package_name"] = _PACKAGE_DIR.name
    kb_cfg["max_num_seqs"] = args.max_num_seqs
    kb_raw = run_worker(
        FASTKERNELS_JAMBA_WORKER, kb_cfg,
        f"fastkernels JambaEngine [{short_name}] all scenarios",
        timeout=21600,
    )
    if kb_raw is None:
        print("  ERROR: fastkernels subprocess failed.")
        sys.exit(1)

    kb_latency = kb_raw.get("latency", [])
    ref_latency = ref_raw.get("latency", []) if ref_raw else []

    # ------------------------------------------------------------------
    # Throughput summary
    # ------------------------------------------------------------------
    all_results = []
    if not args.skip_throughput:
        kb_thr = kb_raw["throughput"]
        ref_thr = ref_raw["throughput"] if ref_raw else None
        for i, sc in enumerate(throughput_scenarios):
            kb_d = kb_thr[i]
            kb_tps = kb_d["total_output_tokens"] / kb_d["elapsed"]
            r = {
                "scenario": sc["name"],
                "num_seqs": args.num_seqs,
                "fastkernels_elapsed": kb_d["elapsed"],
                "fastkernels_output_tokens": kb_d["total_output_tokens"],
                "fastkernels_tok_per_s": kb_tps,
            }
            if args.num_seqs:
                r["avg_output_len"] = kb_d["total_output_tokens"] / args.num_seqs
            if ref_thr is not None:
                f_d = ref_thr[i]
                f_tps = f_d["total_output_tokens"] / f_d["elapsed"]
                r["ref_elapsed"] = f_d["elapsed"]
                r["ref_output_tokens"] = f_d["total_output_tokens"]
                r["ref_tok_per_s"] = f_tps
                r["speedup"] = kb_tps / f_tps if f_tps else 0.0
                if args.temperature == 0.0:
                    r["alignment"] = compute_alignment(
                        kb_d["outputs"], f_d["outputs"]
                    )
            if args.output_dir:
                d = os.path.join(args.output_dir, sc["name"])
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "fastkernels_outputs.json"), "w") as f:
                    json.dump(kb_d, f, indent=2)
                if ref_thr is not None:
                    with open(os.path.join(d, "ref_outputs.json"), "w") as f:
                        json.dump(f_d, f, indent=2)
            all_results.append(r)

        print(f"\n\n{'=' * 100}")
        print(f"  THROUGHPUT SUMMARY (fastkernels JambaEngine vs {ref_name} reference)")
        print(f"{'=' * 100}")
        print(
            f"  {'SCENARIO':<16} {'OUT':>5} "
            f"{'FASTKERNELS tok/s':>15} {f'{ref_name} tok/s':>12} {'SPEEDUP':>9} "
            f"{'AVG MATCH TOKS':>18}"
        )
        print(f"  {'-' * 95}")
        for r in all_results:
            kb_str = f"{r['fastkernels_tok_per_s']:,.0f}"
            f_str = f"{r['ref_tok_per_s']:,.0f}" if "ref_tok_per_s" in r else "N/A"
            spd_str = f"{r['speedup']:.2f}x" if "speedup" in r else "N/A"
            align = r.get("alignment", {})
            avg = align.get("avg_matching_tokens_per_request", 0)
            tot = align.get("avg_output_len", 0)
            match_str = f"{avg:.1f}/{tot:.0f}" if tot > 0 else "N/A"
            out_str = f"{r.get('avg_output_len', 0):>5.0f}"
            print(
                f"  {r['scenario']:<16} {out_str} "
                f"{kb_str:>15} {f_str:>12} {spd_str:>9} {match_str:>18}"
            )
        print("=" * 100)

    # ------------------------------------------------------------------
    # Latency summary
    # ------------------------------------------------------------------
    latency_combined = []
    if kb_latency:
        print(f"\n{'=' * 110}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 110}")
        print(
            f"  {'SCENARIO':<18} {'BS':>4} {'OUT':>5} {'ITERS':>6}"
            f"  {'FASTKERNELS med':>12} {'HF med':>12}"
            f"  {'FASTKERNELS ms/tok':>15} {'HF ms/tok':>12} {'SPEEDUP':>8}"
        )
        print(f"  {'-' * 105}")
        for i, kb_lat in enumerate(kb_latency):
            kb_lats = np.array(kb_lat["latencies"])
            kb_med = float(np.median(kb_lats))
            bs = kb_lat["batch_size"]
            out_len = kb_lat["output_len"]
            total_out_tokens = bs * out_len
            kb_ms_per_tok = (kb_med / total_out_tokens) * 1000

            lat_result = {
                "scenario": kb_lat["name"],
                "batch_size": bs,
                "output_len": out_len,
                "num_iters": kb_lat["num_iters"],
                "fastkernels_median_s": kb_med,
                "fastkernels_ms_per_tok": kb_ms_per_tok,
                "fastkernels_latencies": kb_lat["latencies"],
            }
            f_med_str = "N/A"; spd_str = "N/A"; f_ms_str = "N/A"
            if i < len(ref_latency):
                f_lat = ref_latency[i]
                f_lats = np.array(f_lat["latencies"])
                f_med = float(np.median(f_lats))
                f_ms_per_tok = (f_med / total_out_tokens) * 1000
                spd = f_med / kb_med if kb_med else 0.0
                f_med_str = f"{f_med:.4f}s"
                spd_str = f"{spd:.2f}x"
                f_ms_str = f"{f_ms_per_tok:.2f}"
                lat_result["ref_median_s"] = f_med
                lat_result["ref_ms_per_tok"] = f_ms_per_tok
                lat_result["speedup"] = spd
                lat_result["ref_latencies"] = f_lat["latencies"]
            print(
                f"  {kb_lat['name']:<18} {bs:>4} {out_len:>5} {kb_lat['num_iters']:>6}"
                f"  {kb_med:.4f}s{'':<3} {f_med_str:>12}"
                f"  {kb_ms_per_tok:>13.2f}   {f_ms_str:>10} {spd_str:>8}"
            )
            latency_combined.append(lat_result)
        print("=" * 110)

    if args.output_dir and (all_results or latency_combined):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({
                "gpu": gpu,
                "model": args.model,
                "seed": args.seed,
                "temperature": args.temperature,
                "num_seqs": args.num_seqs,
                "max_num_seqs": args.max_num_seqs,
                "ref_max_num_seqs": args.ref_max_num_seqs,
                "scenarios": all_results,
                "latency_scenarios": latency_combined,
            }, f, indent=2)
        print(f"\n  Results saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
