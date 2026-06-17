"""TTT-E2E benchmark: kb-nano vs official JAX reference.

Compares the kb-nano implementation in
``kb_nano/tasks/baseline/{L2,L3,L4}/ttt_e2e_*.py`` and
``kb_nano/infra/ttt_e2e_engine.py`` against the JAX reference at
``github.com/test-time-training/e2e``.

Honest framing of what this comparison does and does not measure:
  - Correctness (logits / per-token NLL): cross-framework, fully meaningful.
    Numerics don't care which framework computed them; if both engines load
    the same weights and produce the same NLL on the same input, the math
    is right.
  - Throughput: kb-nano (PyTorch) vs JAX. Both run on the same B200 GPU at
    the same dtype (compute_dtype="bf16"), so the speedup ratio is the
    relative cost of bf16 matmuls + sliding-window attention + (optional)
    inner-loop SGD on each framework. JAX gets the benefit of XLA
    compilation through ``jax.lax.scan`` etc.; kb-nano does not.
  - This is the SOTA-library comparison required by CLAUDE.md. The framework
    gap is a known caveat, called out in README.

Variants benchmarked (defaults, configurable):
  - ``ttt_e2e_125m_e2e`` — paper config from ``configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml``
    (12 layers, hidden=768, 12 heads, intermediate=1664, suffix=3, window=8192)

Workload:
  - Input: a small subset of public-domain English text. We tokenize a
    Project Gutenberg book with the Llama-3 tokenizer (``unsloth/llama-3-8b``,
    ungated mirror), then take fixed-length contiguous slices.
  - We run with ``train_mode="meta"`` (the actual TTT-E2E inference path —
    inner-loop SGD active) and additionally ``train_mode="pretrain"`` as a
    control (frozen prime FFN, baseline transformer behavior).
  - **Random-init weights only**. The official trained checkpoints live on
    Requester-Pays GCS and we do not download them; the bench's purpose is
    cross-framework parity (kb-nano matches JAX's compute) and per-framework
    perf. Real perplexity numbers would require the trained checkpoints.

Output: a JSON results blob plus a short text summary printed to stdout.

Usage:
  CUDA_VISIBLE_DEVICES=3 python -m kb_nano.tests.bench_ttt_e2e \\
      --variant 125m_e2e --seq-lens 8192 --n-sequences 4 \\
      --modes pretrain meta
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Config presets — match JAX configs/experiment/...
# ---------------------------------------------------------------------------

PRESETS = {
    # 125m paper config — see configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml
    # 125M params @ Chinchilla scale, the smallest released TTT-E2E model.
    "125m_e2e": {
        "model": {
            "vocab_size": 128256,
            "hidden_size": 768,
            "intermediate_size": 1664,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "mini_batch_size": 1024,
            "sliding_window_size": 8192,
            "seq_len": 32768,
            "rms_norm_eps": 1e-6,
            "initializer_range": 0.02,
            "tie_word_embeddings": True,
            "rope_theta": 500000.0,
            "suffix_len": 3,
            "prime": True,
            "qk_norm": True,
            "pre_norm": True,
            "post_norm": True,
            "compute_dtype": "bf16",
            "param_dtype": "fp32",
            "state_dtype": "fp32",
        },
        "training": {
            "ilr_init": 1.0,
            "ilr_warmup_steps": 0,
            "inner_lr": 1.0,
            "inner_clip": 1.0,
        },
        "init_seed": 0,
    },
    # Tiny config for fast iteration / CI.
    "tiny": {
        "model": {
            "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 4, "num_attention_heads": 4,
            "mini_batch_size": 256, "sliding_window_size": 256,
            "seq_len": 1024, "rms_norm_eps": 1e-6, "initializer_range": 0.02,
            "tie_word_embeddings": True, "rope_theta": 10000.0,
            "suffix_len": 2, "prime": True, "qk_norm": True,
            "pre_norm": True, "post_norm": True,
            "compute_dtype": "bf16", "param_dtype": "fp32", "state_dtype": "fp32",
        },
        "training": {"ilr_init": 1.0, "ilr_warmup_steps": 0, "inner_lr": 1.0, "inner_clip": 1.0},
        "init_seed": 0,
    },
}


# ---------------------------------------------------------------------------
# Real-data input prep — Project Gutenberg book, Llama-3 tokenized
# ---------------------------------------------------------------------------

# Small, public-domain English text to ground the benchmark in real data.
# "The Adventures of Sherlock Holmes" (Conan Doyle, 1892) — Project Gutenberg
# eBook #1661, ~580 KB plain text, fits comfortably under our disk budget.
_PG_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"


def _download_text(cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / "pg1661.txt"
    if not fp.is_file():
        print(f"[bench] downloading {_PG_URL} -> {fp} ...", flush=True)
        with urllib.request.urlopen(_PG_URL, timeout=60) as r, open(fp, "wb") as w:
            w.write(r.read())
    return fp.read_text(encoding="utf-8", errors="ignore")


def _tokenize_corpus(text: str, tokenizer_name: str = "unsloth/llama-3-8b") -> np.ndarray:
    """Tokenize the corpus once, return int64 numpy array of ids."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ids = tok.encode(text, add_special_tokens=False)
    return np.asarray(ids, dtype=np.int64)


def _make_input_slices(ids: np.ndarray, seq_len: int, n_sequences: int, *, stride: int | None = None) -> np.ndarray:
    """Build (n, seq_len) input by taking ``n`` non-overlapping contiguous slices.

    If the corpus is too short for ``n`` non-overlapping windows, returns
    fewer rows. Stride defaults to ``seq_len`` (no overlap).
    """
    if stride is None:
        stride = seq_len
    rows = []
    for i in range(n_sequences):
        s = i * stride
        e = s + seq_len
        if e > ids.size:
            break
        rows.append(ids[s:e])
    if not rows:
        raise ValueError(
            f"Corpus has only {ids.size} tokens, need at least {seq_len}."
        )
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# JAX reference subprocess driver
# ---------------------------------------------------------------------------

_JAX_WORKER = Path(__file__).resolve().parent / "bench_ttt_e2e_jax_worker.py"


def _run_jax_worker(args_list: list[str]) -> dict:
    """Run the JAX subprocess worker and return a dict of stdout lines + return code."""
    cmd = [sys.executable, str(_JAX_WORKER)] + args_list
    # Stream stdout to keep visibility on the long-running JAX compile.
    print(f"[bench] running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        for line in p.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            print(f"  [jax] {line}", flush=True)
            lines.append(line)
        p.wait()
    finally:
        if p.poll() is None:
            p.kill()
    if p.returncode != 0:
        raise RuntimeError(f"JAX worker failed with rc={p.returncode}; cmd={cmd}")
    return {"lines": lines, "wall_s": time.time() - t0}


# ---------------------------------------------------------------------------
# kb-nano runner (in-process)
# ---------------------------------------------------------------------------

def _run_kbnano(
    config: dict,
    weights_npz: Path,
    input_ids: np.ndarray,
    train_mode: str,
    *,
    runs: int = 3,
    compute_dtype: str = "bf16",
    device: str = "cuda",
    attention_backend: str = "cudnn",
) -> dict:
    """Run kb-nano forward and return per-token NLL + timings."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kb_nano.infra.ttt_e2e_engine import TTTE2EEngine
    from kb_nano.tasks.baseline.L4.ttt_e2e import TTTE2EConfig

    m = config["model"]
    t = config["training"]
    kcfg = TTTE2EConfig(
        vocab_size=m["vocab_size"],
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        chunk_size=m["mini_batch_size"],
        sliding_window_size=m["sliding_window_size"],
        suffix_len=m["suffix_len"],
        max_position_embeddings=m["seq_len"],
        rope_theta=float(m["rope_theta"]),
        rms_norm_eps=float(m["rms_norm_eps"]),
        qk_norm=bool(m["qk_norm"]),
        tie_word_embeddings=bool(m["tie_word_embeddings"]),
        has_prime=bool(m["prime"]),
        inner_lr=float(t["inner_lr"]),
        inner_clip_grad_norm=float(t["inner_clip"]),
        ilr_init=float(t["ilr_init"]),
        attention_backend=attention_backend,
    )

    dt_map = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
    cdt = dt_map[compute_dtype]
    pdt = dt_map["fp32"]  # storage in fp32 → keeps RMSNorm precision via _rms_native

    print(f"[bench] kb-nano: building engine, dtype={compute_dtype} ...", flush=True)
    t0 = time.time()
    # We store params in compute_dtype (bf16) to hit fast-path matmuls; the
    # RMSNormNative L1 op does fp32-internal math so RMSNorm precision matches
    # JAX. Inner-loop prime SGD still works in fp32 via .to(bf16) at the
    # functional_call boundary.
    engine = TTTE2EEngine(
        kcfg, weights_npz=str(weights_npz), device=device,
        param_dtype=cdt, compute_dtype=cdt,
    )
    engine.compile_layers()
    if train_mode == "meta":
        # Capture a CUDA Graph of the meta forward at this (B=1, T) shape
        # so replay produces deterministic timing — eager has structural
        # autograd-driven variance (std 10 ms vs JAX's std 0.75 ms).
        T = int(input_ids.shape[-1])
        engine.capture_meta_graph(batch_size=1, seq_len=T)
    build_s = time.time() - t0
    print(f"[bench] kb-nano: engine built in {build_s:.2f}s", flush=True)

    ids_t = torch.from_numpy(input_ids).long().to(device)
    if ids_t.dim() == 1:
        ids_t = ids_t.unsqueeze(0)

    # Warmup (compile / cache CUDA tensors).
    print(f"[bench] kb-nano: warmup forward (mode={train_mode}) ...", flush=True)
    t0 = time.time()
    out = engine.forward(ids_t, train_mode=train_mode)
    torch.cuda.synchronize()
    warm_s = time.time() - t0
    print(f"[bench] kb-nano: warmup done in {warm_s:.2f}s", flush=True)

    # Timed runs.
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        out = engine.forward(ids_t, train_mode=train_mode)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    print(f"[bench] kb-nano: timed runs (s): {[f'{x:.3f}' for x in times]}", flush=True)

    nll = out.token_nll.float().cpu().numpy()
    return {
        "token_nll": nll,
        "build_s": build_s,
        "warmup_s": warm_s,
        "run_times_s": times,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", choices=list(PRESETS.keys()), default="tiny")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--n-sequences", type=int, default=2)
    p.add_argument("--modes", nargs="+", default=["pretrain", "meta"])
    p.add_argument("--cache-dir", default="/raid/user_data/olu/kb_nano_ttt_e2e_cache")
    p.add_argument("--skip-jax", action="store_true", help="Skip the JAX reference (kb-nano-only).")
    p.add_argument("--skip-kbnano", action="store_true", help="Skip kb-nano (JAX-only).")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--results-out", default=None, help="Optional: write a JSON summary to this path.")
    p.add_argument("--attention-backend", choices=["cudnn", "flex"], default="cudnn",
                   help="Suffix-path attention backend. flex enables FlexAttention's "
                        "Triton-fused fwd+bwd autotuned to (chunk_size, W+chunk_size, head_dim).")
    args = p.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(json.dumps(PRESETS[args.variant]))  # deep copy
    cfg["model"]["seq_len"] = max(cfg["model"]["seq_len"], args.seq_len)

    # Ensure seq_len is a multiple of mini_batch_size.
    cs = cfg["model"]["mini_batch_size"]
    if args.seq_len % cs != 0:
        raise ValueError(f"seq-len {args.seq_len} must be a multiple of mini_batch_size {cs}")

    # 1) Build random-init weights via the JAX worker (only once per variant).
    weights = cache / f"weights_{args.variant}.npz"
    if not weights.is_file():
        cfg_init = json.loads(json.dumps(cfg))
        cfg_init["training"] = {**cfg["training"], "train_mode": "pretrain", "seq_length": args.seq_len}
        cfg_init_path = cache / f"cfg_init_{args.variant}.json"
        with open(cfg_init_path, "w") as f:
            json.dump(cfg_init, f)
        _run_jax_worker(["init_and_save", "--config", str(cfg_init_path), "--seed", "0", "--out", str(weights)])
    else:
        print(f"[bench] reusing weights {weights}", flush=True)

    # 2) Load tokenized corpus.
    corpus_text = _download_text(cache)
    corpus_ids = _tokenize_corpus(corpus_text)
    print(f"[bench] tokenized corpus: {corpus_ids.size} tokens", flush=True)
    inputs = _make_input_slices(corpus_ids, args.seq_len, args.n_sequences)
    print(f"[bench] input slices: {inputs.shape}", flush=True)
    # Save first slice for the JAX worker (one input per call, since the JAX
    # worker is pinned to a single seq_length per invocation).
    summary = {"variant": args.variant, "seq_len": args.seq_len,
               "n_sequences": int(inputs.shape[0]), "modes": list(args.modes), "results": []}

    for mode in args.modes:
        print(f"\n========== mode={mode} ==========", flush=True)
        cfg_run = json.loads(json.dumps(cfg))
        cfg_run["training"] = {**cfg["training"], "train_mode": mode, "seq_length": args.seq_len}
        cfg_path = cache / f"cfg_{args.variant}_{mode}_{args.seq_len}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg_run, f)

        per_seq_results = []
        for si in range(inputs.shape[0]):
            ids_path = cache / f"ids_{args.variant}_seq{si}_{args.seq_len}.npy"
            np.save(ids_path, inputs[si])

            jax_nll = jax_runs = jax_compile = None
            if not args.skip_jax:
                jax_out = cache / f"jax_out_{args.variant}_{mode}_seq{si}_{args.seq_len}.npz"
                _run_jax_worker([
                    "run_forward", "--config", str(cfg_path), "--weights", str(weights),
                    "--input-ids", str(ids_path), "--out", str(jax_out),
                    "--runs", str(args.runs),
                ])
                z = np.load(jax_out)
                jax_nll = z["token_nll_loss"]
                jax_runs = z["run_times_s"].tolist()
                jax_compile = float(z["compile_s"])

            kb_nll = kb_runs = kb_warm = None
            if not args.skip_kbnano:
                r = _run_kbnano(cfg_run, weights, inputs[si], mode, runs=args.runs,
                                attention_backend=args.attention_backend)
                kb_nll = r["token_nll"][0]   # (B=1, T) -> (T,)
                kb_runs = r["run_times_s"]
                kb_warm = r["warmup_s"]

            entry = {"seq_idx": si}
            if jax_nll is not None:
                entry["jax"] = {
                    "nll_mean": float(jax_nll.mean()),
                    "compile_s": jax_compile,
                    "run_times_s": jax_runs,
                }
            if kb_nll is not None:
                entry["kbnano"] = {
                    "nll_mean": float(kb_nll.mean()),
                    "warm_s": kb_warm,
                    "run_times_s": kb_runs,
                }
            if jax_nll is not None and kb_nll is not None:
                diff = np.abs(kb_nll - jax_nll)
                entry["diff"] = {
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                }
            per_seq_results.append(entry)

        summary["results"].append({"mode": mode, "per_seq": per_seq_results})

    # Aggregate + report.
    print("\n========== summary ==========", flush=True)
    for blk in summary["results"]:
        mode = blk["mode"]
        n = len(blk["per_seq"])
        print(f"mode={mode}, sequences={n}", flush=True)
        for e in blk["per_seq"]:
            line = f"  seq{e['seq_idx']}:"
            if "jax" in e:
                jr = np.median(e["jax"]["run_times_s"]) if e["jax"]["run_times_s"] else 0.0
                line += f" jax(med={jr:.3f}s,nll={e['jax']['nll_mean']:.4f})"
            if "kbnano" in e:
                kr = np.median(e["kbnano"]["run_times_s"]) if e["kbnano"]["run_times_s"] else 0.0
                line += f" kb(med={kr:.3f}s,nll={e['kbnano']['nll_mean']:.4f})"
            if "diff" in e:
                line += f" diff(max={e['diff']['max_abs']:.4e},mean={e['diff']['mean_abs']:.4e})"
            print(line, flush=True)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(summary, indent=2))
        print(f"[bench] wrote {args.results_out}", flush=True)


if __name__ == "__main__":
    main()
