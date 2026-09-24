"""Token-level retrieval embedders (BGE-M3, ColBERTv2) for ``fastkernels e2e``.

In-process, fastkernels-only port of the fastkernels side of
``fastkernels/validate/bench_embedding.py`` (``KB_WORKER``):

* **model** -- ``fastkernels.infra.embedding_engine.EmbeddingEngine`` (the production
  path: packed varlen batches, engine defaults for torch.compile / CUDA graphs), built
  lazily inside ``run`` so candidate kernels patched in by the runner are the ones used;
* **inputs** -- the harness's cached JSONL (MLDR documents for BGE-M3, MS MARCO passages
  for ColBERTv2; streamed from HF with the harness's shuffle seed on first use), tokenized
  exactly like the harness (special tokens, truncation at the tokenizer's max length);
* **scheduler** -- the harness's vLLM-default limits for the GPU (B200/H*: 16384 batched
  tokens, 1024 seqs);
* **throughput** -- warm up on the first 4 requests and ``spec.extra["warmup_passes"]``
  (default 1) untimed full passes, then time ``encode`` over the workload's requests
  (capped by ``spec.max_requests``), CUDA-synced, including the D2H of the token
  embeddings: input tokens / s, median of ``THROUGHPUT_PASSES`` timed passes (see the
  constants below for why not a single pass);
* **latency** -- the first ``batch_size`` requests, ``num_warmup`` warmups then
  ``LATENCY_ITERS`` timed ``encode`` calls: median seconds.

Correctness: an untimed pass over the first ``spec.correctness_samples`` requests saves, per
request, the BGE-M3 **dense** embedding (L2-normalized CLS hidden state, captured from the
same forward), the **mean-pooled** token (ColBERT) embedding, and a deterministic strided
subsample of token-embedding rows -- all fp16, a few MB in total.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import ClassVar

from .base import Adapter, RunSpec, Timing

DATA_SEED = 42          # bench_embedding --seed default: shuffle seed of the cached JSONL
WARMUP_REQUESTS = 4     # bench_embedding warms the engine on the first 4 records
TOKEN_ROWS = 32         # token-embedding rows stored per correctness sample
# Timed latency iterations (bench_embedding's --latency-iters, default 5). Same estimator
# (median after num_warmup warmups), more samples: at 5 iterations the ~13 ms single-request
# probe moved up to 6% between two runs of identical code. Override: spec.extra["latency_iters"].
LATENCY_ITERS = 20
# Throughput: bench_embedding times ONE pass right after a 4-request warmup. That pass
# carries one-time costs -- Triton autotuning of candidate kernels on every new token count
# (a 200-document first pass measured 76 s against 1.8 s steady state), pinned staging and
# allocator growth, and above all first-touch of the ~18.7 GB of host output memory (glibc
# only starts recycling it after a few passes): the baseline's full 1000-document pass went
# 11.1 s -> 8.2 s -> 7.8 s -> steady on some hosts. So: spec.extra["warmup_passes"] untimed
# full passes (orchestrator knob, default 1; 0 for cheap probes), then the MEDIAN of
# THROUGHPUT_PASSES timed passes, which lands on the steady state even when the first timed
# passes are still warming up. Override: spec.extra["throughput_passes"].
WARMUP_PASSES = 1
THROUGHPUT_PASSES = 5
_MODEL_KEYS = ("bge-m3", "bge_m3", "colbert")


def _data_workload(hf_name: str):
    """The throughput workload whose records feed this model (also for latency)."""
    from fastkernels.workloads import EMBEDDING_THROUGHPUT_WORKLOADS
    for w in EMBEDDING_THROUGHPUT_WORKLOADS:
        if w.model_name == hf_name:
            return w
    lower = hf_name.lower()
    for w in EMBEDDING_THROUGHPUT_WORKLOADS:
        if w.model_key.replace("_", "-") in lower.replace("_", "-"):
            return w
    raise LookupError(f"no embedding throughput workload for {hf_name}")


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def records_path(workload) -> Path:
    """Cached JSONL of the workload: the harness's repo-local file if it is complete,
    else ``~/.fastkernels/data/embedding_workloads/`` (built on first use with the
    harness's own streaming/shuffle code, so the records are identical)."""
    from fastkernels.validate import bench_embedding as be
    repo = be._jsonl_path(workload)
    if repo.is_file() and _count_lines(repo) >= workload.num_requests:
        return repo
    root = Path(os.environ.get("FASTKERNELS_EMBEDDING_DATA_DIR",
                               Path.home() / ".fastkernels" / "data" / "embedding_workloads"))
    path = root / workload.jsonl_name
    if path.is_file() and _count_lines(path) >= workload.num_requests:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    print(f"  building {path} ({workload.num_requests} records of {workload.dataset_name})", flush=True)
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for record in be._iter_dataset_texts(workload, DATA_SEED):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= workload.num_requests:
                break
    if count < workload.num_requests:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"only {count} records for {workload.name}; need {workload.num_requests}")
    tmp.replace(path)  # atomic: concurrent runs never read a partial file
    return path


def _select(scenario, spec: RunSpec) -> list:
    if spec.workloads is None:
        return list(scenario.workloads)
    wanted = set(spec.workloads)
    return [w for w in scenario.workloads if w.value in wanted or w.name in wanted]


def _token_row_index(n: int) -> list[int]:
    """Deterministic, evenly spaced row indices (first and last row included)."""
    if n <= TOKEN_ROWS:
        return list(range(n))
    return sorted({round(i * (n - 1) / (TOKEN_ROWS - 1)) for i in range(TOKEN_ROWS)})


class EmbeddingAdapter(Adapter):
    name = "embedding"
    metric: ClassVar[str] = (
        "d = clamp(1 - c, 0, 1) per request, c = min(cos(dense), cos(pooled), cos(token rows)): "
        "cosine similarity of the L2-normalized CLS (BGE-M3 dense) embedding, of the mean-pooled "
        "token (ColBERT) embedding, and of the flattened strided sample of <=32 token-embedding "
        "rows (unit-norm rows, so ~ the mean per-token cosine that bench_embedding reports); "
        "non-finite or mismatched outputs give d = 1")

    @classmethod
    def handles(cls, scenario) -> bool:
        from fastkernels.workloads import Embedding
        name = scenario.hf_name.lower()
        return (any(k in name for k in _MODEL_KEYS)
                and all(isinstance(w, Embedding) for w in scenario.workloads))

    # -- run -------------------------------------------------------------------------
    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from fastkernels.infra.embedding_engine import EmbeddingEngine
        from fastkernels.validate import bench_embedding as be
        from fastkernels.workloads import Purpose, purpose_of, spec_for

        workloads = _select(scenario, spec)
        data_wl = _data_workload(scenario.hf_name)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[scenario.dtype]
        eager = bool(spec.enforce_eager or scenario.enforce_eager)
        gpu = be._detect_gpu_name()
        max_tokens, max_seqs = be._vllm_default_scheduler_limits(gpu)
        max_seqs = scenario.max_num_seqs or max_seqs

        # How many records each phase needs (tokenize only that prefix).
        n_thr = {}
        n_need = min(WARMUP_REQUESTS, data_wl.num_requests)
        for w in workloads:
            p = spec_for(w).params
            if purpose_of(w) is Purpose.THROUGHPUT:
                n = p.num_requests if spec.max_requests is None else min(p.num_requests, spec.max_requests)
                n_thr[w] = max(1, n)
                n_need = max(n_need, n_thr[w])
            else:
                n_need = max(n_need, p.batch_size)
        n_corr = min(max(0, spec.correctness_samples), data_wl.num_requests)
        n_need = min(max(n_need, n_corr), data_wl.num_requests)

        torch.manual_seed(spec.seed)
        print(f"  loading fastkernels EmbeddingEngine {scenario.hf_name} ({scenario.dtype}, gpu={gpu}, "
              f"max_num_batched_tokens={max_tokens}, max_num_seqs={max_seqs}, eager={eager})", flush=True)
        engine = EmbeddingEngine(scenario.hf_name, seed=spec.seed, dtype=dtype, device="cuda:0",
                                 max_num_batched_tokens=max_tokens, max_num_seqs=max_seqs,
                                 compile_model=False if eager else None)
        if eager:
            engine._use_hopper_colbert_cudagraph = False

        path = records_path(data_wl)
        records = be._load_jsonl(path)[:n_need]
        max_length = be._vllm_model_max_length(engine.tokenizer)
        t0 = time.time()
        token_ids = be._tokenize_texts(engine.tokenizer, [r["text"] for r in records], max_length)
        print(f"  {len(records)} records from {path}, max_length={max_length}, "
              f"tokenized in {time.time() - t0:.1f}s", flush=True)
        prompts = [{"prompt_token_ids": list(ids)} for ids in token_ids]

        def encode(batch):
            out = engine.encode(batch, pooling_task="token_embed", use_tqdm=False)
            return [item.outputs.data.detach().cpu().numpy() for item in out]

        def sync():
            torch.cuda.synchronize()

        timings: dict[str, Timing] = {}
        for w in workloads:
            p = spec_for(w).params
            if purpose_of(w) is Purpose.THROUGHPUT:
                if data_wl.name != p.name:
                    raise ValueError(f"{w.value} is not a workload of {scenario.hf_name}")
                batch = prompts[:n_thr[w]]
                tokens = sum(len(x["prompt_token_ids"]) for x in batch)
                encode(prompts[:WARMUP_REQUESTS])
                for _ in range(int(spec.extra.get("warmup_passes", WARMUP_PASSES))):
                    encode(batch)  # untimed; outputs dropped right away
                passes = []
                for _ in range(int(spec.extra.get("throughput_passes", THROUGHPUT_PASSES))):
                    sync()
                    start = time.perf_counter()
                    outs = encode(batch)
                    sync()
                    passes.append(time.perf_counter() - start)
                    assert len(outs) == len(batch)
                    del outs
                elapsed = float(np.median(passes))
                timings[w.value] = {"kind": "throughput", "value": tokens / elapsed, "unit": "tok/s",
                                    "requests": len(batch), "input_tokens": tokens,
                                    "elapsed_s": round(elapsed, 4),
                                    "passes_s": [round(x, 4) for x in passes]}
            else:
                batch = prompts[:p.batch_size]
                iters = int(spec.extra.get("latency_iters", max(p.num_iters, LATENCY_ITERS)))
                for _ in range(p.num_warmup):
                    encode(batch)
                lat = []
                for _ in range(iters):
                    sync()
                    start = time.perf_counter()
                    encode(batch)
                    sync()
                    lat.append(time.perf_counter() - start)
                timings[w.value] = {"kind": "latency", "value": float(np.median(lat)), "unit": "s",
                                    "p99_s": float(np.percentile(lat, 99)), "batch_size": len(batch),
                                    "input_tokens": sum(len(x["prompt_token_ids"]) for x in batch),
                                    "iters": iters}
            print(f"  {w.value}: {timings[w.value]}", flush=True)

        # -- correctness (untimed): same engine.encode, CLS rows captured from its forward
        cls_rows: list = []
        forward = engine._forward_varlen

        def capture(*args, **kwargs):
            hidden = forward(*args, **kwargs)
            cu = kwargs["cu_seqlens"] if "cu_seqlens" in kwargs else args[2]
            cls_rows.append(hidden[cu[:-1].long()].float())
            return hidden

        engine._forward_varlen = capture
        try:
            with torch.no_grad():
                outs = engine.encode(prompts[:n_corr], pooling_task="token_embed", use_tqdm=False)
        finally:
            engine._forward_varlen = forward
        dense = None
        if cls_rows and sum(r.shape[0] for r in cls_rows) == n_corr:
            dense = F.normalize(torch.cat(cls_rows), dim=-1).half().cpu()
        pooled, rows, idxs = [], [], []
        for item in outs:
            data = item.outputs.data.detach().float().cpu()
            pooled.append(data.mean(0) if data.shape[0] else torch.zeros(data.shape[-1]))
            idx = _token_row_index(int(data.shape[0]))
            idxs.append(idx)
            rows.append(data[idx].half())
        torch.save({
            "kind": "embeddings",
            "model": scenario.hf_name,
            "dtype": scenario.dtype,
            "ids": [r["id"] for r in records[:n_corr]],
            "input_tokens": [len(x["prompt_token_ids"]) for x in prompts[:n_corr]],
            "lengths": [int(item.outputs.data.shape[0]) for item in outs],
            "dense": dense,                                   # [N, hidden] fp16 or None
            "pooled": torch.stack(pooled).half() if pooled else None,  # [N, dim] fp16
            "token_index": idxs,                              # per request: sampled row ids
            "token_rows": rows,                               # per request: [k, dim] fp16
            "metric": self.metric,
        }, f"{spec.out_dir}/outputs.pt")
        return timings

    # -- compare -----------------------------------------------------------------------
    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        import torch

        def cos(a, b) -> float:
            if a is None or b is None or a.shape != b.shape or a.numel() == 0:
                return 0.0
            a, b = a.double().reshape(-1), b.double().reshape(-1)
            if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                return 0.0
            den = float(a.norm() * b.norm())
            return float(a @ b) / den if den > 0 else (1.0 if float(a.norm()) == float(b.norm()) else 0.0)

        def pick(out, key, i):
            v = out.get(key)
            if v is None:
                return None
            return v[i] if i < len(v) else None

        n_ref = len(ref.get("token_rows") or [])
        per_sample, c_all, c_dense, c_pool, c_tok = [], [], [], [], []
        max_abs, mismatches, nonfinite = 0.0, 0, 0
        use_dense = ref.get("dense") is not None and cand.get("dense") is not None
        for i in range(n_ref):
            r_len, c_len = pick(ref, "lengths", i), pick(cand, "lengths", i)
            if c_len is None or r_len != c_len or pick(ref, "token_index", i) != pick(cand, "token_index", i):
                mismatches += 1
                per_sample.append(1.0)
                c_all.append(0.0)
                continue
            cs = []
            for key, acc in (("dense", c_dense), ("pooled", c_pool), ("token_rows", c_tok)):
                if key == "dense" and not use_dense:
                    continue
                a, b = pick(ref, key, i), pick(cand, key, i)
                c = cos(a, b)
                acc.append(c)
                cs.append(c)
                if b is not None and not bool(torch.isfinite(b.float()).all()):
                    nonfinite += 1
                elif a is not None and b is not None and a.shape == b.shape and a.numel():
                    max_abs = max(max_abs, float((a.float() - b.float()).abs().max()))
            c = min(cs)
            c_all.append(c)
            per_sample.append(min(1.0, max(0.0, 1.0 - c)))

        def mean(x):
            return float(sum(x) / len(x)) if x else float("nan")

        summary = {
            "n": n_ref,
            "mean_cosine": mean(c_all), "min_cosine": min(c_all) if c_all else float("nan"),
            "mean_cosine_dense": mean(c_dense), "min_cosine_dense": min(c_dense, default=float("nan")),
            "mean_cosine_pooled": mean(c_pool), "min_cosine_pooled": min(c_pool, default=float("nan")),
            "mean_cosine_tokens": mean(c_tok), "min_cosine_tokens": min(c_tok, default=float("nan")),
            "max_abs_diff": max_abs if nonfinite == 0 else float("inf"),
            "mean_d": mean(per_sample), "max_d": max(per_sample, default=float("nan")),
            "frac_cos_ge_0.99": mean([float(c >= 0.99) for c in c_all]),
            "length_mismatches": mismatches, "nonfinite_tensors": nonfinite,
        }
        return {"per_sample": per_sample, "summary": {k: (round(v, 8) if isinstance(v, float)
                                                           and math.isfinite(v) else v)
                                                       for k, v in summary.items()}}


if __name__ == "__main__":  # CPU-only data prep: python -m fastkernels.e2e.adapters.embedding [HF_NAME ...]
    import sys
    for hf in sys.argv[1:] or ["BAAI/bge-m3"]:
        print(hf, "->", records_path(_data_workload(hf)))
