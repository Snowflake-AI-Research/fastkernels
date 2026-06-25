#!/usr/bin/env python3
"""Benchmark repo-native LLaDA against official HF / Fast-dLLM baselines.

This benchmark now defaults to a Fast-dLLM official-style protocol:
- real task prompts, not synthetic profiling prompts
- variable-length requests batched together with left padding
- timing includes per-batch chat formatting, tokenization, padding, generation,
  and output post-processing
- throughput is counted from post-processed generated tokens, matching the
  official eval_llada.py convention more closely than the previous microbench
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PACKAGE_DIR))

from bench.utils.worker import run_worker


DEFAULT_TASK = "humaneval"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_SAMPLES: int | None = None
DEFAULT_GEN_LENGTH = 256
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_THRESHOLD = 0.9
DEFAULT_OURS_BACKEND = "dual"
DEFAULT_FASTDLLM_ROOT = "third_party/Fast-dLLM"
DEFAULT_FEWSHOT_SEED = 1234
MASK_TOKEN_ID = 126336
FASTDLLM_IGNORE_TOKEN_ID = 126081


TASK_SPECS: dict[str, dict[str, Any]] = {
    "humaneval": {
        "dataset": "openai_humaneval",
        "config": None,
        "split": "test",
        "count_mode": "humaneval_raw",
        "label": "HumanEval (0-shot)",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "count_mode": "retokenize_text",
        "label": "GSM8K",
    },
}


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-") or "value"


def _load_task_requests(task: str, max_samples: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")

    from lm_eval.tasks import TaskManager, get_task_dict

    if task not in TASK_SPECS:
        raise ValueError(f"Unsupported task: {task}")
    spec = TASK_SPECS[task]
    manager = TaskManager()
    task_obj = get_task_dict([task], task_manager=manager)[task]
    task_obj.set_fewshot_seed(DEFAULT_FEWSHOT_SEED)
    task_obj.build_all_requests(limit=max_samples)

    requests: list[dict[str, Any]] = []
    for inst in task_obj.instances:
        prompt = inst.args[0]
        gen_args = inst.args[1] if len(inst.args) > 1 and isinstance(inst.args[1], dict) else {}
        requests.append(
            {
                "prompt": prompt,
                "task_id": inst.doc.get("task_id", f"{task}-{inst.doc_id}"),
                "stop_tokens": list(gen_args.get("until", [])),
                "count_mode": spec["count_mode"],
            }
        )

    return requests, {
        "mode": "task",
        "task": task,
        "task_label": spec["label"],
        "dataset": spec["dataset"],
        "split": spec["split"],
        "num_fewshot": getattr(task_obj._config, "num_fewshot", None),
        "fewshot_seed": DEFAULT_FEWSHOT_SEED,
        "num_samples": len(requests),
        "examples": [req["prompt"] for req in requests[:2]],
    }


DLLM_WORKER = r'''
import json, os, sys, time
import importlib.util
from pathlib import Path
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

pkg_root = Path(cfg["project_root"])
spec = importlib.util.spec_from_file_location(
    "fastkernels", pkg_root / "__init__.py",
    submodule_search_locations=[str(pkg_root)],
)
fastkernels = importlib.util.module_from_spec(spec)
sys.modules["fastkernels"] = fastkernels
spec.loader.exec_module(fastkernels)

from transformers import AutoModelForCausalLM, AutoTokenizer

from fastkernels.infra.dllm_engine import (
    LLaDAEngine,
    DLLMSamplingParams,
    masked_diffusion_generate,
    masked_diffusion_generate_with_dual_cache,
    masked_diffusion_generate_with_prefix_cache,
)


def _chunked(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _prepare_batch_inputs(requests, tokenizer, is_instruct, device):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    encoded = []
    max_len = 0
    for req in requests:
        prompt = req["prompt"]
        if is_instruct:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        ids = tokenizer(prompt)["input_ids"]
        encoded.append(ids)
        max_len = max(max_len, len(ids))

    batch = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long, device=device)
    prompt_lens = []
    for row_idx, ids in enumerate(encoded):
        prompt_lens.append(len(ids))
        batch[row_idx, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)

    return batch, prompt_lens


def _finalize_batch_outputs(raw_generated_ids, requests, tokenizer, ignore_token_id):
    outputs = []
    token_count = 0
    for req, gen_ids in zip(requests, raw_generated_ids):
        if req["count_mode"] == "humaneval_raw":
            final_ids = [token for token in gen_ids if token != ignore_token_id]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        else:
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            for stop_seq in req.get("stop_tokens", []):
                if stop_seq in raw_text:
                    raw_text = raw_text.split(stop_seq)[0]
            final_ids = [
                token for token in tokenizer(raw_text)["input_ids"]
                if token != ignore_token_id
            ]
            text = tokenizer.decode(final_ids, skip_special_tokens=True)
        token_count += len(final_ids)
        outputs.append(
            {
                "task_id": req["task_id"],
                "token_ids": final_ids,
                "text": text,
            }
        )
    return outputs, token_count


def _run_backend(cfg):
    device = cfg.get("device", "cuda")
    dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float16
    model_name = cfg["model"]
    backend = cfg["backend"]
    requests = cfg["requests"]
    batches = _chunked(requests, cfg["batch_size"])

    if backend == "ours":
        engine = LLaDAEngine(model_name=model_name, dtype=dtype, device=device)
        tokenizer = engine.tokenizer
        model = engine.model
        run_generate = None
    elif backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()
        run_generate = masked_diffusion_generate
    else:
        fastdllm_root = cfg.get("fastdllm_root")
        if not fastdllm_root:
            raise ValueError("fastdllm_root is required for Fast-dLLM reference backends")
        fastdllm_llada = str(Path(fastdllm_root) / "llada")
        if fastdllm_llada not in sys.path:
            sys.path.insert(0, fastdllm_llada)
        from generate import generate_with_dual_cache, generate_with_prefix_cache
        from model.configuration_llada import LLaDAConfig as FastDLLMConfig
        from model.modeling_llada import LLaDAModelLM as FastDLLMLLaDAModelLM

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = FastDLLMConfig.from_pretrained(model_name)
        config.flash_attention = True
        model = FastDLLMLLaDAModelLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            config=config,
        ).to(device).eval()
        if backend == "fastdllm-prefix":
            run_generate = generate_with_prefix_cache
        elif backend == "fastdllm-dual":
            run_generate = generate_with_dual_cache
        else:
            raise ValueError(f"Unknown backend: {backend}")

    is_instruct = "instruct" in model_name.lower()
    model_device = _model_device(model)
    outputs = []
    total_tokens = 0
    total_nfe = 0
    prompt_lengths = []
    torch.cuda.synchronize()
    start = time.perf_counter()

    for batch_requests in batches:
        input_ids, batch_prompt_lens = _prepare_batch_inputs(
            batch_requests, tokenizer, is_instruct, model_device,
        )
        prompt_lengths.extend(batch_prompt_lens)

        if backend == "ours":
            sampling = DLLMSamplingParams(
                gen_length=cfg["gen_length"],
                steps=cfg["steps"],
                block_length=cfg["block_length"],
                temperature=cfg["temperature"],
                remasking=cfg["remasking"],
                threshold=cfg.get("threshold"),
                decode_mode=cfg.get("ours_backend", "dual"),
            )
            batch_outputs = engine.generate(input_ids.tolist(), sampling)
            raw_generated_ids = [out.token_ids for out in batch_outputs]
            batch_nfe = batch_outputs[0].nfe if batch_outputs else 0
        else:
            generated, batch_nfe = run_generate(
                model,
                input_ids,
                steps=cfg["steps"],
                gen_length=cfg["gen_length"],
                block_length=cfg["block_length"],
                temperature=cfg["temperature"],
                remasking=cfg["remasking"],
                mask_id=cfg["mask_token_id"],
                threshold=cfg.get("threshold"),
            )
            raw_generated_ids = [
                generated[row_idx][input_ids.shape[1]:].tolist()
                for row_idx in range(generated.shape[0])
            ]

        batch_outputs, batch_token_count = _finalize_batch_outputs(
            raw_generated_ids, batch_requests, tokenizer, cfg["ignore_token_id"],
        )
        total_tokens += batch_token_count
        total_nfe += batch_nfe
        outputs.extend(batch_outputs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits_sample = []
    for batch_requests in batches:
        input_ids, _ = _prepare_batch_inputs(
            batch_requests, tokenizer, is_instruct, model_device,
        )
        batch_logits = model(input_ids).logits[:, -1, :512].float().cpu().tolist()
        logits_sample.extend(batch_logits)

    result = {
        "backend": backend,
        "elapsed": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
        "total_generated_tokens": total_tokens,
        "total_nfe": total_nfe,
        "avg_nfe_per_batch": total_nfe / max(len(batches), 1),
        "num_batches": len(batches),
        "num_samples": len(requests),
        "batch_size": cfg["batch_size"],
        "prompt_lengths": prompt_lengths,
        "logits_shape": [len(logits_sample), 512],
        "logits_sample": logits_sample,
        "outputs": outputs,
    }
    return result


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    result = _run_backend(cfg)
    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())
    os._exit(0)


if __name__ == "__main__":
    main()
'''


def _cosine(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


def _mae(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def _compare_outputs(ours, ref):
    token_matches = []
    seq_exact = []
    for o, r in zip(ours["outputs"], ref["outputs"]):
        ot, rt = o["token_ids"], r["token_ids"]
        common_len = min(len(ot), len(rt))
        match = sum(int(a == b) for a, b in zip(ot[:common_len], rt[:common_len]))
        token_matches.append(match / max(common_len, 1))
        seq_exact.append(int(ot == rt))
    return {
        "throughput_ratio": ours["tokens_per_second"] / ref["tokens_per_second"],
        "logits_cosine": _cosine(ours["logits_sample"], ref["logits_sample"]),
        "logits_mae": _mae(ours["logits_sample"], ref["logits_sample"]),
        "token_match_rate": sum(token_matches) / len(token_matches),
        "sequence_exact_match_rate": sum(seq_exact) / len(seq_exact),
    }


def _result_filename(args, input_info: dict[str, Any], resolved_reference_backends: str) -> str:
    threshold_slug = "nothr" if args.no_threshold else f"thr{str(args.threshold).replace('.', 'p')}"
    ref_slug = "noref" if args.skip_reference else _slugify(resolved_reference_backends)
    return (
        f"{_slugify(input_info['task'])}_ns{input_info['num_samples']}_bs{args.batch_size}_"
        f"ours-{args.ours_backend}_ref-{ref_slug}_"
        f"gen{args.gen_length}_steps{args.steps}_block{args.block_length}_{threshold_slug}.json"
    )


def _default_reference_backends(ours_backend: str) -> str:
    return {
        "vanilla": "hf",
        "prefix": "fastdllm-prefix",
        "dual": "fastdllm-dual",
    }[ours_backend]


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark repo-native LLaDA vs official HF / Fast-dLLM baselines with an official-style task protocol",
    )
    ap.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--task", type=str, default=DEFAULT_TASK, choices=sorted(TASK_SPECS))
    ap.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--gen-length", type=int, default=DEFAULT_GEN_LENGTH)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--remasking", type=str, default="low_confidence")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--no-threshold", action="store_true")
    ap.add_argument(
        "--ours-backend",
        type=str,
        default=DEFAULT_OURS_BACKEND,
        choices=["vanilla", "prefix", "dual"],
        help="Decode backend for our implementation.",
    )
    ap.add_argument(
        "--reference-backends",
        type=str,
        default=None,
        help="Comma-separated baseline backends: hf, fastdllm-prefix, fastdllm-dual. Defaults to the matching official backend for --ours-backend.",
    )
    ap.add_argument(
        "--fastdllm-root",
        type=str,
        default=DEFAULT_FASTDLLM_ROOT,
        help="Path to the official Fast-dLLM repo when using fastdllm-* baselines.",
    )
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    threshold = None if args.no_threshold else args.threshold
    if args.steps is None:
        if threshold is None:
            args.steps = args.gen_length
        else:
            if args.gen_length % args.block_length != 0:
                raise ValueError(
                    "--gen-length must be divisible by --block-length when using threshold-based parallel decoding.",
                )
            args.steps = args.gen_length // args.block_length
    resolved_reference_backends = args.reference_backends or _default_reference_backends(args.ours_backend)

    gpu = _detect_gpu_name()
    out_dir = Path(args.output_dir or f"tests/results/{gpu}/llada-8b-instruct")
    out_dir.mkdir(parents=True, exist_ok=True)

    requests, input_info = _load_task_requests(args.task, args.max_samples)
    input_info["batch_size"] = args.batch_size
    input_info["protocol"] = "fastdllm-official-like"

    common = {
        "project_root": str(_PACKAGE_DIR),
        "model": args.model,
        "requests": requests,
        "batch_size": args.batch_size,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "remasking": args.remasking,
        "threshold": threshold,
        "mask_token_id": MASK_TOKEN_ID,
        "ignore_token_id": FASTDLLM_IGNORE_TOKEN_ID,
        "use_bf16": True,
        "device": "cuda",
        "ours_backend": args.ours_backend,
    }

    ours = run_worker(DLLM_WORKER, {**common, "backend": "ours"}, "fastkernels LLaDA")
    references = {}
    comparisons = {}
    if not args.skip_reference:
        backends = [b.strip() for b in resolved_reference_backends.split(",") if b.strip()]
        for backend in backends:
            if backend.startswith("fastdllm") and not Path(args.fastdllm_root).exists():
                raise FileNotFoundError(
                    f"Fast-dLLM baseline requested but repo not found: {args.fastdllm_root}"
                )
            label = {
                "hf": "official HF LLaDA",
                "fastdllm-prefix": "official Fast-dLLM (prefix cache)",
                "fastdllm-dual": "official Fast-dLLM (dual cache)",
            }.get(backend, backend)
            ref = run_worker(
                DLLM_WORKER,
                {**common, "backend": backend, "fastdllm_root": args.fastdllm_root},
                label,
            )
            references[backend] = ref
            if ours and ref:
                comparisons[backend] = _compare_outputs(ours, ref)

    results = {
        "model": args.model,
        "input": input_info,
        "config": {
            "protocol": "fastdllm-official-like",
            "gen_length": args.gen_length,
            "steps": args.steps,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "threshold": threshold,
            "batch_size": args.batch_size,
            "max_samples": len(requests),
            "ours_backend": args.ours_backend,
            "reference_backends": resolved_reference_backends if not args.skip_reference else "",
        },
        "ours": ours,
        "references": references,
        "comparisons": comparisons,
    }

    out_file = out_dir / _result_filename(args, input_info, resolved_reference_backends)
    out_file.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "model": results["model"],
                "input": results["input"],
                "config": results["config"],
                "ours_tokens_per_second": results["ours"]["tokens_per_second"],
                "comparisons": results["comparisons"],
                "output_file": str(out_file),
            },
            indent=2,
        )
    )
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
