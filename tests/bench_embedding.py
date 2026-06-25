#!/usr/bin/env python3
"""Token-level embedding throughput/latency benchmark: fastkernels vs vLLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from fastkernels.bench.utils.worker import run_worker
from fastkernels.bench.utils.workloads import (
    EMBEDDING_LATENCY_WORKLOADS,
    EMBEDDING_THROUGHPUT_WORKLOADS,
    EmbeddingThroughputWorkload,
)

from bench_vllm import (
    _detect_gpu_name,
    _install_flashinfer_sitecustomize,
    _make_run_id,
    _parse_port_env,
    _reserve_tcp_port,
)

_HELD_PORT_LOCKS: list[object] = []
CORRECTNESS_COSINE_THRESHOLD = 0.99


def _vllm_default_scheduler_limits(gpu: str) -> tuple[int, int]:
    high_mem_names = ("B200", "B100", "H200", "H100", "MI300")
    if any(name in gpu for name in high_mem_names) and "A100" not in gpu:
        return 16384, 1024
    if gpu == "unknown":
        return 4096, 256
    return 8192, 256


def _jsonl_path(workload: EmbeddingThroughputWorkload) -> Path:
    return _PACKAGE_DIR / "data" / "embedding_workloads" / workload.jsonl_name


def _iter_dataset_texts(workload: EmbeddingThroughputWorkload, seed: int):
    from datasets import load_dataset

    dataset = load_dataset(
        workload.dataset_name,
        workload.dataset_config,
        split=workload.dataset_split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    for idx, item in enumerate(dataset):
        text = item.get(workload.text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        if workload.id_column is not None and workload.id_column in item:
            record_id = str(item[workload.id_column])
        else:
            record_id = f"{workload.name}-{idx}"
        yield {"id": record_id, "text": text.strip()}


def _ensure_jsonl_workload(
    workload: EmbeddingThroughputWorkload,
    seed: int,
    force: bool = False,
) -> Path:
    path = _jsonl_path(workload)
    if path.exists() and not force:
        with path.open(encoding="utf-8") as f:
            num_cached = sum(1 for line in f if line.strip())
        if num_cached < workload.num_requests:
            print(
                f"  Cached JSONL workload {path} has {num_cached} records; "
                f"rebuilding for {workload.num_requests}",
                flush=True,
            )
        else:
            print(f"  Using existing JSONL workload: {path}", flush=True)
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"  Preparing JSONL workload {workload.name}: downloading/streaming "
        f"{workload.num_requests} records -> {path}",
        flush=True,
    )
    count = 0
    with path.open("w", encoding="utf-8") as f:
        progress = tqdm(
            _iter_dataset_texts(workload, seed),
            total=workload.num_requests,
            desc=f"build {workload.name}",
            unit="records",
            file=sys.stdout,
        )
        for record in progress:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= workload.num_requests:
                break
    if count < workload.num_requests:
        raise RuntimeError(
            f"Only wrote {count} records for {workload.name}; "
            f"expected {workload.num_requests}",
        )
    return path


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records.append({"id": str(item["id"]), "text": str(item["text"])})
    return records


def _tokenize_texts(tokenizer, texts: list[str], max_length: int) -> list[list[int]]:
    token_ids = []
    batch_starts = range(0, len(texts), 64)
    for start in tqdm(
        batch_starts,
        total=(len(texts) + 63) // 64,
        desc="tokenize workload",
        unit="batch",
        file=sys.stdout,
    ):
        encoded = tokenizer(
            texts[start:start + 64],
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=max_length,
            verbose=False,
        )
        token_ids.extend(list(ids) for ids in encoded["input_ids"])
    return token_ids


def _tokenizer_max_length(tokenizer) -> int:
    max_length = getattr(tokenizer, "model_max_length", None)
    try:
        max_length = int(max_length)
    except (TypeError, ValueError):
        max_length = 512
    if max_length <= 0 or max_length >= 100_000:
        max_length = 512
    return max_length


def _vllm_model_max_length(tokenizer) -> int:
    """Mirror vLLM's pooling-model default max length from tokenizer config."""
    return _tokenizer_max_length(tokenizer)


def _build_records(
    workload: EmbeddingThroughputWorkload,
    tokenizer,
    seed: int,
    force: bool,
) -> tuple[Path, list[dict]]:
    print(f"  Loading workload records for {workload.name}", flush=True)
    path = _ensure_jsonl_workload(workload, seed=seed, force=force)
    print(f"  Reading {path}", flush=True)
    records = _load_jsonl(path)[:workload.num_requests]
    if len(records) != workload.num_requests:
        raise RuntimeError(
            f"{path} contains {len(records)} records; expected {workload.num_requests}",
        )
    max_length = _vllm_model_max_length(tokenizer)
    print(
        f"  Tokenizing workload for {workload.name} "
        f"({len(records)} requests, max_length={max_length})",
        flush=True,
    )
    prompt_token_ids = _tokenize_texts(tokenizer, [r["text"] for r in records], max_length)
    return path, [
        {
            "id": record["id"],
            "text": record["text"],
            "prompt_token_ids": list(token_ids),
            "input_tokens": len(token_ids),
        }
        for record, token_ids in zip(records, prompt_token_ids, strict=True)
    ]


def _select_workloads(model: str) -> list[EmbeddingThroughputWorkload]:
    if model == "all":
        return list(EMBEDDING_THROUGHPUT_WORKLOADS)
    for workload in EMBEDDING_THROUGHPUT_WORKLOADS:
        names = {workload.model_key, workload.model_name, workload.model_name.split("/")[-1]}
        if model in names:
            return [workload]
    raise SystemExit(
        f"Unknown model {model!r}. Use one of: all, "
        + ", ".join(w.model_key for w in EMBEDDING_THROUGHPUT_WORKLOADS),
    )


WORKER_SHARED = r'''
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

with open(sys.argv[1]) as f:
    cfg = json.load(f)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _insert_prefix_token(tensor, prefix_id):
    prefix = torch.full(
        (tensor.size(0), 1),
        prefix_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor[:, :1], prefix, tensor[:, 1:]], dim=1)


def _save_outputs(path, outputs):
    print(f"  Materializing {len(outputs)} outputs on CPU", flush=True)
    arrays = [
        np.asarray(output, dtype=np.float32)
        for output in tqdm(outputs, desc="materialize outputs", unit="req", file=sys.stdout)
    ]
    dims = np.asarray([arr.shape[-1] if arr.ndim else 0 for arr in arrays], dtype=np.int64)
    lengths = np.asarray([arr.shape[0] if arr.ndim >= 2 else 1 for arr in arrays], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    if len(lengths):
        offsets[1:] = np.cumsum(lengths)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    prefix = path[:-4] if path.endswith(".npz") else path
    values_path = f"{prefix}_values.npy"
    meta_path = f"{prefix}_meta.json"
    total_rows = int(offsets[-1])
    dim = int(max(dims)) if len(dims) else 0
    gib = total_rows * dim * np.dtype(np.float16).itemsize / (1024 ** 3)
    print(
        f"  Saving output tensor values to {values_path} "
        f"({total_rows:,} x {dim}, {gib:.2f} GiB, uncompressed)",
        flush=True,
    )
    values = np.lib.format.open_memmap(
        values_path,
        mode="w+",
        dtype=np.float16,
        shape=(total_rows, dim),
    )
    cursor = 0
    for arr in tqdm(arrays, desc="write output values", unit="req", file=sys.stdout):
        if not arr.size:
            continue
        flat = arr.reshape(-1, arr.shape[-1]).astype(np.float16, copy=False)
        next_cursor = cursor + flat.shape[0]
        values[cursor:next_cursor, :flat.shape[1]] = flat
        cursor = next_cursor
    values.flush()
    del values

    metadata = {
        "format": "embedding-output-v2",
        "values_path": values_path,
        "offsets": offsets.tolist(),
        "lengths": lengths.tolist(),
        "dims": dims.tolist(),
    }
    print(f"  Saving output metadata to {meta_path}", flush=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    summaries = [
        {
            "shape": list(arr.shape),
            "sum": float(arr.sum(dtype=np.float64)),
            "l2": float(np.linalg.norm(arr.reshape(-1).astype(np.float64))),
        }
        for arr in arrays
    ]
    return summaries, meta_path


def _tokenize_bge(tokenizer, texts, max_length, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


def _tokenize_colbert_doc(tokenizer, texts, max_length, doc_marker_token_id, device):
    encoded = tokenizer(
        texts,
        padding="longest",
        truncation="longest_first",
        max_length=max_length - 1,
        return_tensors="pt",
    )
    input_ids = _insert_prefix_token(encoded["input_ids"], doc_marker_token_id)
    attention_mask = _insert_prefix_token(encoded["attention_mask"], 1)
    return input_ids.to(device), attention_mask.to(device)


def _latency_stats(latencies):
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "median_s": float(np.median(arr)),
        "p99_s": float(np.percentile(arr, 99)),
        "latencies": [float(x) for x in latencies],
    }
'''


KB_WORKER = WORKER_SHARED + r'''
def _load_kb_engine(cfg, device, dtype):
    sys.path.insert(0, cfg["project_root"])
    from fastkernels.infra.embedding_engine import EmbeddingEngine
    print(f"  Loading fastkernels model {cfg['model_name']} on {device}", flush=True)
    return EmbeddingEngine(
        cfg["model_name"],
        seed=cfg["seed"],
        dtype=dtype,
        device=device,
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        max_num_seqs=cfg["max_num_seqs"],
    )


def _encode(engine, prompts, cfg):
    print(f"  fastkernels encode: {len(prompts)} requests", flush=True)
    outputs = engine.encode(
        prompts,
        pooling_task="token_embed",
        use_tqdm=True,
    )
    return [
        item.outputs.data.detach().cpu().numpy()
        for item in outputs
    ]


def main():
    device = cfg["device"]
    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16
    torch.manual_seed(cfg["seed"])
    engine = _load_kb_engine(cfg, device, dtype)

    warm_prompts = [
        {"prompt_token_ids": r["prompt_token_ids"]}
        for r in cfg["records"][:min(4, len(cfg["records"]))]
    ]
    print(f"  fastkernels warmup: {len(warm_prompts)} requests", flush=True)
    _encode(engine, warm_prompts, cfg)

    prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in cfg["records"]]
    print(f"  fastkernels timed throughput encode starting: {len(prompts)} requests", flush=True)
    _cuda_sync()
    start = time.perf_counter()
    outputs = _encode(engine, prompts, cfg)
    _cuda_sync()
    elapsed = time.perf_counter() - start
    output_summaries, output_artifact = _save_outputs(cfg["output_npz"], outputs)

    latency = []
    for spec in cfg.get("latency_scenarios", []):
        lat_prompts = [
            {"prompt_token_ids": r["prompt_token_ids"]}
            for r in cfg["records"][:spec["batch_size"]]
        ]
        print(
            f"  fastkernels latency {spec['name']}: "
            f"{spec['num_warmup']} warmup, {spec['num_iters']} timed iterations",
            flush=True,
        )
        for _ in tqdm(
            range(spec["num_warmup"]),
            desc=f"kb warmup {spec['name']}",
            unit="iter",
            file=sys.stdout,
        ):
            _encode(engine, lat_prompts, cfg)
        latencies = []
        for _ in tqdm(
            range(spec["num_iters"]),
            desc=f"kb latency {spec['name']}",
            unit="iter",
            file=sys.stdout,
        ):
            _cuda_sync()
            start = time.perf_counter()
            _encode(engine, lat_prompts, cfg)
            _cuda_sync()
            latencies.append(time.perf_counter() - start)
        item = {
            "name": spec["name"],
            "batch_size": spec["batch_size"],
            "input_tokens": int(sum(r["input_tokens"] for r in cfg["records"][:spec["batch_size"]])),
            "num_iters": spec["num_iters"],
        }
        item.update(_latency_stats(latencies))
        latency.append(item)

    result = {
        "elapsed": elapsed,
        "num_requests": len(prompts),
        "total_input_tokens": int(sum(r["input_tokens"] for r in cfg["records"])),
        "outputs": output_summaries,
        "output_npz": output_artifact,
        "latency": latency,
    }
    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
'''


VLLM_WORKER = WORKER_SHARED + r'''
def _encode(llm, prompts, cfg):
    from vllm.pooling_params import PoolingParams

    print(f"  vLLM encode: {len(prompts)} requests", flush=True)
    params = PoolingParams(task="token_embed")
    outputs = llm.encode(
        prompts,
        pooling_params=params,
        pooling_task="token_embed",
        use_tqdm=True,
    )
    arrays = []
    for output in outputs:
        data = output.outputs.data
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)
        arrays.append(arr)
    return arrays


def main():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    from vllm import LLM

    torch.manual_seed(cfg["seed"])
    hf_overrides = None
    if cfg["model_key"] == "bge_m3":
        hf_overrides = {"architectures": ["BgeM3EmbeddingModel"]}

    print(f"  Loading vLLM model {cfg['model_name']}", flush=True)
    llm = LLM(
        model=cfg["model_name"],
        runner="pooling",
        tensor_parallel_size=cfg["tp"],
        dtype=cfg["dtype"],
        seed=cfg["seed"],
        enforce_eager=cfg["enforce_eager"],
        trust_remote_code=True,
        hf_overrides=hf_overrides,
        max_model_len=cfg["max_length"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        max_num_seqs=cfg["max_num_seqs"],
    )

    warm_prompts = [
        {"prompt_token_ids": r["prompt_token_ids"]}
        for r in cfg["records"][:min(4, len(cfg["records"]))]
    ]
    print(f"  vLLM warmup: {len(warm_prompts)} requests", flush=True)
    _encode(llm, warm_prompts, cfg)

    prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in cfg["records"]]
    print(f"  vLLM timed throughput encode starting: {len(prompts)} requests", flush=True)
    _cuda_sync()
    start = time.perf_counter()
    outputs = _encode(llm, prompts, cfg)
    _cuda_sync()
    elapsed = time.perf_counter() - start
    output_summaries, output_artifact = _save_outputs(cfg["output_npz"], outputs)

    latency = []
    for spec in cfg.get("latency_scenarios", []):
        lat_prompts = [
            {"prompt_token_ids": r["prompt_token_ids"]}
            for r in cfg["records"][:spec["batch_size"]]
        ]
        print(
            f"  vLLM latency {spec['name']}: "
            f"{spec['num_warmup']} warmup, {spec['num_iters']} timed iterations",
            flush=True,
        )
        for _ in tqdm(
            range(spec["num_warmup"]),
            desc=f"vLLM warmup {spec['name']}",
            unit="iter",
            file=sys.stdout,
        ):
            _encode(llm, lat_prompts, cfg)
        latencies = []
        for _ in tqdm(
            range(spec["num_iters"]),
            desc=f"vLLM latency {spec['name']}",
            unit="iter",
            file=sys.stdout,
        ):
            _cuda_sync()
            start = time.perf_counter()
            _encode(llm, lat_prompts, cfg)
            _cuda_sync()
            latencies.append(time.perf_counter() - start)
        item = {
            "name": spec["name"],
            "batch_size": spec["batch_size"],
            "input_tokens": int(sum(r["input_tokens"] for r in cfg["records"][:spec["batch_size"]])),
            "num_iters": spec["num_iters"],
        }
        item.update(_latency_stats(latencies))
        latency.append(item)

    result = {
        "elapsed": elapsed,
        "num_requests": len(prompts),
        "total_input_tokens": int(sum(r["input_tokens"] for r in cfg["records"])),
        "outputs": output_summaries,
        "output_npz": output_artifact,
        "latency": latency,
    }
    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
'''


def _load_output_artifact(path: str) -> dict:
    if path.endswith(".json"):
        with open(path) as f:
            metadata = json.load(f)
        values = np.load(metadata["values_path"], mmap_mode="r")
        return {
            "values": values,
            "offsets": np.asarray(metadata["offsets"], dtype=np.int64),
            "lengths": np.asarray(metadata["lengths"], dtype=np.int64),
            "dims": np.asarray(metadata["dims"], dtype=np.int64),
        }
    return np.load(path, mmap_mode="r")


def _delete_output_artifact(path: str) -> None:
    paths = [path]
    if path.endswith(".json") and os.path.exists(path):
        with open(path) as f:
            metadata = json.load(f)
        values_path = metadata.get("values_path")
        if isinstance(values_path, str):
            paths.insert(0, values_path)
    for item in paths:
        try:
            os.remove(item)
        except FileNotFoundError:
            pass


def _compare_outputs(kb_npz: str, vllm_npz: str, num_requests: int) -> dict:
    print(f"  Correctness: comparing {num_requests} request outputs", flush=True)
    kb_data = _load_output_artifact(kb_npz)
    vllm_data = _load_output_artifact(vllm_npz)
    cosines = []
    shape_matches = 0
    mean_abs_diffs = []
    max_abs_diff = 0.0

    for idx in tqdm(
        range(num_requests),
        desc="compare outputs",
        unit="req",
        file=sys.stdout,
    ):
        kb = kb_data["values"][
            int(kb_data["offsets"][idx]):int(kb_data["offsets"][idx + 1])
        ][:, :int(kb_data["dims"][idx])].astype(np.float32)
        vv = vllm_data["values"][
            int(vllm_data["offsets"][idx]):int(vllm_data["offsets"][idx + 1])
        ][:, :int(vllm_data["dims"][idx])].astype(np.float32)
        if kb.shape == vv.shape:
            shape_matches += 1
            a = kb.reshape(-1)
            b = vv.reshape(-1)
        else:
            rows = min(kb.shape[0], vv.shape[0])
            dim = min(kb.shape[1] if kb.ndim == 2 else 0, vv.shape[1] if vv.ndim == 2 else 0)
            a = kb[:rows, :dim].reshape(-1)
            b = vv[:rows, :dim].reshape(-1)
        if a.size == 0 or b.size == 0:
            cosine = 0.0
            diff = np.asarray([float("inf")], dtype=np.float32)
        else:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = float(np.dot(a, b) / denom) if denom else 0.0
            diff = np.abs(a - b)
        cosines.append(cosine)
        mean_abs_diffs.append(float(diff.mean()))
        max_abs_diff = max(max_abs_diff, float(diff.max()))

    min_cosine = float(min(cosines)) if cosines else 0.0
    return {
        "num_requests": num_requests,
        "shape_matches": shape_matches,
        "avg_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "min_cosine": min_cosine,
        "avg_mean_abs_diff": float(np.mean(mean_abs_diffs)) if mean_abs_diffs else 0.0,
        "max_abs_diff": max_abs_diff,
        "cosine_threshold": CORRECTNESS_COSINE_THRESHOLD,
        "pass": shape_matches == num_requests and min_cosine >= CORRECTNESS_COSINE_THRESHOLD,
    }


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _run_one_workload(
    workload: EmbeddingThroughputWorkload,
    args,
    gpu: str,
    vllm_port: int | None,
) -> tuple[dict, list[dict]]:
    print(f"\n{'-' * 70}", flush=True)
    print(f"  Workload {workload.name}: {workload.model_name}", flush=True)
    print(f"{'-' * 70}", flush=True)
    print("  Loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(workload.model_name, trust_remote_code=True)
    max_length = _vllm_model_max_length(tokenizer)
    jsonl_path, records = _build_records(
        workload,
        tokenizer,
        seed=args.seed,
        force=args.refresh_workloads,
    )
    total_input_tokens = int(sum(r["input_tokens"] for r in records))
    print(
        f"  Workload ready: {len(records)} requests, "
        f"{total_input_tokens:,} input tokens, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}, "
        f"max_num_seqs={args.max_num_seqs}",
        flush=True,
    )
    latency_scenarios = [
        {
            "name": item.name,
            "batch_size": item.batch_size,
            "num_warmup": item.num_warmup,
            "num_iters": args.latency_iters,
        }
        for item in EMBEDDING_LATENCY_WORKLOADS
        if item.batch_size <= len(records)
    ]

    scenario_dir = Path(args.artifact_dir) / workload.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    common_cfg = {
        "model_key": workload.model_key,
        "model_name": workload.model_name,
        "records": records,
        "max_length": max_length,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "cuda:0" if gpu != "unknown" else "cpu",
        "tp": args.tp,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "latency_scenarios": [] if args.skip_latency else latency_scenarios,
    }

    vllm_raw = None
    if not args.skip_vllm:
        print(f"  Launching vLLM worker for {workload.name}", flush=True)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(vllm_port)
        vllm_cfg = dict(common_cfg)
        vllm_cfg["output_npz"] = str(scenario_dir / "vllm_outputs.npz")
        vllm_raw = run_worker(
            VLLM_WORKER,
            vllm_cfg,
            f"vLLM [{workload.model_name}] token_embed",
            timeout=10800,
        )
        if vllm_raw is None:
            raise SystemExit("vLLM embedding worker failed")

    print(f"  Launching fastkernels worker for {workload.name}", flush=True)
    kb_cfg = dict(common_cfg)
    kb_cfg["output_npz"] = str(scenario_dir / "fastkernels_outputs.npz")
    kb_raw = run_worker(
        KB_WORKER,
        kb_cfg,
        f"fastkernels [{workload.model_name}] token_embed",
        timeout=10800,
    )
    if kb_raw is None:
        raise SystemExit("fastkernels embedding worker failed")

    print(f"  Aggregating throughput results for {workload.name}", flush=True)
    kb_tps = kb_raw["total_input_tokens"] / kb_raw["elapsed"]
    result = {
        "scenario": workload.name,
        "model": workload.model_name,
        "model_key": workload.model_key,
        "mode": "token_embed",
        "dataset": {
            "name": workload.dataset_name,
            "config": workload.dataset_config,
            "split": workload.dataset_split,
            "jsonl_path": str(jsonl_path),
        },
        "num_requests": len(records),
        "max_length": max_length,
        "total_input_tokens": total_input_tokens,
        "fastkernels_elapsed": kb_raw["elapsed"],
        "fastkernels_input_tok_per_s": kb_tps,
        "fastkernels_output_artifact": kb_raw["output_npz"],
    }
    artifacts_deleted = False
    if vllm_raw is not None:
        vllm_tps = vllm_raw["total_input_tokens"] / vllm_raw["elapsed"]
        correctness = _compare_outputs(
            kb_raw["output_npz"],
            vllm_raw["output_npz"],
            len(records),
        )
        _delete_output_artifact(kb_raw["output_npz"])
        _delete_output_artifact(vllm_raw["output_npz"])
        artifacts_deleted = True
        result.update({
            "vllm_elapsed": vllm_raw["elapsed"],
            "vllm_input_tok_per_s": vllm_tps,
            "vllm_output_artifact": vllm_raw["output_npz"],
            "speedup": _safe_div(kb_tps, vllm_tps),
            "correctness": correctness,
        })
    elif args.skip_vllm:
        _delete_output_artifact(kb_raw["output_npz"])
        artifacts_deleted = True
    result["output_artifacts_deleted"] = artifacts_deleted

    latency_results = []
    kb_latency = kb_raw.get("latency", [])
    vllm_latency = vllm_raw.get("latency", []) if vllm_raw else []
    for idx, kb_lat in enumerate(kb_latency):
        item = {
            "scenario": kb_lat["name"],
            "model": workload.model_name,
            "batch_size": kb_lat["batch_size"],
            "input_tokens": kb_lat["input_tokens"],
            "num_iters": kb_lat["num_iters"],
            "fastkernels_median_s": kb_lat["median_s"],
            "fastkernels_p99_s": kb_lat["p99_s"],
            "fastkernels_ms_per_tok": kb_lat["median_s"] / kb_lat["input_tokens"] * 1000,
            "fastkernels_latencies": kb_lat["latencies"],
        }
        if idx < len(vllm_latency):
            v_lat = vllm_latency[idx]
            item.update({
                "vllm_median_s": v_lat["median_s"],
                "vllm_p99_s": v_lat["p99_s"],
                "vllm_ms_per_tok": v_lat["median_s"] / v_lat["input_tokens"] * 1000,
                "vllm_latencies": v_lat["latencies"],
                "speedup": _safe_div(v_lat["median_s"], kb_lat["median_s"]),
            })
        latency_results.append(item)

    return result, latency_results


def _print_summary(results: list[dict], latency_results: list[dict]) -> None:
    print(f"\n\n{'=' * 104}")
    print("  EMBEDDING THROUGHPUT SUMMARY")
    print(f"{'=' * 104}")
    print(
        f"  {'SCENARIO':<28} {'REQS':>5} {'TOKENS':>10} "
        f"{'FASTKERNELS tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
        f"{'CORRECT':>8} {'MIN COS':>9}",
    )
    print(f"  {'-' * 96}")
    for row in results:
        vllm_tps = f"{row['vllm_input_tok_per_s']:,.0f}" if "vllm_input_tok_per_s" in row else "N/A"
        speedup = f"{row['speedup']:.2f}x" if "speedup" in row else "N/A"
        correctness = row.get("correctness")
        correct = "PASS" if correctness and correctness["pass"] else ("FAIL" if correctness else "N/A")
        min_cos = f"{correctness['min_cosine']:.6f}" if correctness else "N/A"
        print(
            f"  {row['scenario']:<28} {row['num_requests']:>5} "
            f"{row['total_input_tokens']:>10,} {row['fastkernels_input_tok_per_s']:>15,.0f} "
            f"{vllm_tps:>12} {speedup:>8} {correct:>8} {min_cos:>9}",
        )
    print(f"{'=' * 104}")

    if latency_results:
        print(f"\n{'=' * 112}")
        print("  EMBEDDING LATENCY SUMMARY")
        print(f"{'=' * 112}")
        print(
            f"  {'MODEL':<16} {'SCENARIO':<18} {'BS':>4} {'TOKENS':>8} {'ITERS':>6} "
            f"{'FASTKERNELS med':>12} {'vLLM med':>12} "
            f"{'KB ms/tok':>11} {'vLLM ms/tok':>12} {'SPEEDUP':>8}",
        )
        print(f"  {'-' * 104}")
        for row in latency_results:
            model = row["model"].split("/")[-1]
            v_med = f"{row['vllm_median_s']:.4f}s" if "vllm_median_s" in row else "N/A"
            v_ms = f"{row['vllm_ms_per_tok']:.2f}" if "vllm_ms_per_tok" in row else "N/A"
            speedup = f"{row['speedup']:.2f}x" if "speedup" in row else "N/A"
            print(
                f"  {model:<16} {row['scenario']:<18} {row['batch_size']:>4} "
                f"{row['input_tokens']:>8,} {row['num_iters']:>6} "
                f"{row['fastkernels_median_s']:.4f}s{'':<3} {v_med:>12} "
                f"{row['fastkernels_ms_per_tok']:>11.2f} {v_ms:>12} {speedup:>8}",
            )
        print(f"{'=' * 112}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-level embedding benchmark: fastkernels vs vLLM",
    )
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-iters", type=int, default=5)
    parser.add_argument("--refresh-workloads", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    gpu = _detect_gpu_name()
    args.max_num_batched_tokens, args.max_num_seqs = _vllm_default_scheduler_limits(gpu)
    if args.output_dir is None:
        run_id = _make_run_id(args.run_id)
        args.output_dir = str(
            _PACKAGE_DIR / "tests" / "results" / gpu / "embedding" / run_id
        )
    elif args.run_id is not None:
        print("  NOTE: --run-id is ignored because --output-dir was provided.")
    args.artifact_dir = str(
        Path(tempfile.gettempdir())
        / "fastkernels_embedding_outputs"
        / Path(args.output_dir).name
    )

    vllm_port = None
    flashinfer_namespace = None
    previous_flashinfer_namespace_env = os.environ.get(
        "FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE",
    )
    if not args.skip_vllm:
        vllm_port, vllm_port_lock = _reserve_tcp_port(
            preferred=_parse_port_env("VLLM_PORT"),
        )
        _HELD_PORT_LOCKS.append(vllm_port_lock)
        os.environ["VLLM_PORT"] = str(vllm_port)
        if args.tp > 1:
            flashinfer_namespace = (
                os.environ.get("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE")
                or f"bench-embedding-{os.getpid()}-{vllm_port}"
            )
            os.environ["FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE"] = flashinfer_namespace
            _install_flashinfer_sitecustomize()

    workloads = _select_workloads(args.model)
    print("=" * 78)
    print("  fastkernels Embedding Baseline vs vLLM -- token_embed Benchmark")
    print("=" * 78)
    print(f"  Workloads      : {', '.join(w.name for w in workloads)}")
    print("  Mode           : token_embed")
    print("  Metric         : end-to-end input tokens/sec")
    print("  Requests       : 1000 per workload")
    print(f"  TP             : {args.tp}")
    print(f"  DType          : {args.dtype}")
    print(f"  Seed           : {args.seed}")
    print(f"  Scheduler      : max_num_batched_tokens={args.max_num_batched_tokens}, max_num_seqs={args.max_num_seqs}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Artifact dir   : {args.artifact_dir}")
    if vllm_port is not None:
        print(f"  vLLM port      : {vllm_port}")
        if flashinfer_namespace is not None:
            print(f"  vLLM FI ns     : {flashinfer_namespace}")
    print("=" * 78)

    throughput_results = []
    latency_results = []
    for workload in tqdm(
        workloads,
        desc="embedding workloads",
        unit="workload",
        file=sys.stdout,
    ):
        result, latency = _run_one_workload(workload, args, gpu, vllm_port)
        throughput_results.append(result)
        latency_results.extend(latency)

    if previous_flashinfer_namespace_env is None:
        os.environ.pop("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE", None)
    else:
        os.environ["FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE"] = previous_flashinfer_namespace_env

    _print_summary(throughput_results, latency_results)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    combined = {
        "gpu": gpu,
        "mode": "token_embed",
        "metric": "end_to_end_input_tokens_per_second",
        "tp": args.tp,
        "seed": args.seed,
        "dtype": args.dtype,
        "throughput_scenarios": throughput_results,
        "latency_scenarios": latency_results,
    }
    if vllm_port is not None:
        combined["vllm_port"] = vllm_port
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
