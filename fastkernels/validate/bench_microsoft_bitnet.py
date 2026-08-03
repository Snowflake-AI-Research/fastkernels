#!/usr/bin/env python3
"""Throughput + alignment benchmark for fastkernels BitNet b1.58 vs Microsoft
BitNet GPU lib.

The SOTA reference is the official Microsoft BitNet GPU implementation
(https://github.com/microsoft/BitNet/tree/main/gpu), which provides a
custom W2A8 (1.58-bit weight x int8 activation) CUDA kernel + CUDA-graph
batched generate path.  This is the only implementation that runs the
``microsoft/bitnet-b1.58-2B-4T`` model in its native quantized format on
GPU; the HuggingFace ``transformers`` integration falls back to bf16
matmul and is roughly 5-6x slower, so it is *not* a useful SOTA target.

Workloads come from the scenario file's declared ``workloads`` list, passed as
``--workloads``. Each is resolved through ``fastkernels.workloads``, so the
throughput/latency split and the datasets are whatever the scenario declares:

  * ``LLM.mixed``          throughput, 1000 reqs, wildchat-mixed-1k
  * ``LLM.long_context``   throughput,   64 reqs, longbench-longctx
  * ``LLM.single_request`` latency,  bs=1,  128 out
  * ``LLM.fixed_batch_32`` latency,  bs=32, 128 out

The official Microsoft decode GEMM only implements ``M == 1`` kernels, so the
SOTA worker must use ``--gen-bsz 1`` and loop requests one-by-one -- and a
``batch_size > 1`` latency probe has **no** like-for-like reference at all. Those
rows record ``speedup: null`` with ``reference_unsupported_reason`` rather than
comparing a batched fastkernels run against a serial reference loop, which would
read as a kernel win when it only reflects upstream having no batched kernel.

Because the reference pins ``(gen_bsz, prompt_length, gen_length)`` at
CUDA-graph build time and asserts uniform input *and* output lengths per
scenario, each workload becomes one fixed-shape regime. Prompt *content* is the
workload's real dataset; the fixed input length is the **mean** real prompt
length of the sampled set, applied by suffix-keep / left-pad. Mean rather than
median because these are throughput workloads and the mean preserves the trace's
total prefill token count -- wildchat-mixed is mostly short prompts (p50 60 vs
mean ~150), so a median-length ``mixed`` would have no prefill work at all.

The length is then clamped to what the model can represent: bitnet-b1.58-2B-4T
has ``max_position_embeddings=4096`` while longbench-longctx prompts are >= 8185
tokens, so ``long-context`` runs at the model's ceiling minus the output length
rather than at the real mean. Clamping is reported in the ``[data]`` line and in
``results.json``.

Both engines run greedy decoding (temperature 0, ignore_eos) on the **same**
prompts and return per-request output token ids, so speed and correctness are
measured in the *same* run, in the same execution mode. fastkernels runs
non-eager by default, matching bench_vllm.py and the reference's own CUDA-graph
timing path; pass ``--enforce-eager`` to disable graphs. Timing an eager
fastkernels against a graph-replaying reference understated this row by ~13x
(0.089x vs 1.17-1.30x like-for-like on B200).

Alignment is computed against the official models run with fresh per-step
attention metadata, which avoids comparing fastkernels against a known Microsoft
FastGen CUDA-graph metadata bug where generated tokens are not represented
correctly in the replayed attention bias.

Setup (one-time):
-----------------
Provisioned automatically -- the sweep does this for you, or run it directly::

    python -m fastkernels.validate.provision bitnet

That clones microsoft/BitNet under ``$FASTKERNELS_HOME/third_party``, builds
``libbitnet.so`` for the *local* compute capability (upstream's compile.sh
hardcodes ``compute_80`` PTX, which would leave the reference JIT-ing sm_80 code
on a newer GPU), installs the ``xformers`` the reference's ``model.py`` imports,
and runs the two-step checkpoint conversion. The equivalent by hand::

    cd /path/to/microsoft/BitNet/gpu
    bash bitnet_kernels/compile.sh
    hf download microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --local-dir checkpoints/bitnet-b1.58-2B-4T-bf16
    python convert_safetensors.py \\
        --safetensors_file checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors \\
        --output checkpoints/model_state.pt --model_name 2B
    python convert_checkpoint.py --input checkpoints/model_state.pt
    rm checkpoints/model_state.pt   # only the int2/fp16 splits are needed

Usage:
------
::

    # full benchmark (fastkernels + Microsoft BitNet GPU on every workload
    # the scenario declares) -- reference provisioned by `provision bitnet`.
    python tests/bench_microsoft_bitnet.py \\
        --workloads mixed,long-context,single-request,fixed-batch-32

    # smoke run using fastkernels's continuous scheduler
    python tests/bench_microsoft_bitnet.py --num-prompts 32 --kb-bsz 0 --skip-sota

    # one workload, throughput only
    python tests/bench_microsoft_bitnet.py --workloads mixed --skip-latency

    # fastkernels only
    python tests/bench_microsoft_bitnet.py --skip-sota
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from fastkernels.validate.comparison import (  # noqa: E402
    latency_entry,
    throughput_entry,
)
from fastkernels.validate.provision import BITNET_DIR  # noqa: E402


_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

# Default workloads when the caller names none. These are the LLM workloads
# full.yaml declares for this row; the sweep passes them explicitly via
# --workloads. LLM.fixed_batch_32 is excluded: the official int2 decode kernel
# dispatches only for M == 1, so the reference cannot batch and the workload has
# no baseline to compare against.
DEFAULT_WORKLOADS = ("mixed", "long-context", "single-request")

# Output length for throughput workloads whose spec leaves decode_cap unset
# (LLM.long_context). The reference pins gen_length at CUDA-graph build time, so
# some concrete value is required.
DEFAULT_LONG_CONTEXT_OUTPUT_LEN = 128


def _resolve_workloads(names: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """Split declared workload names into throughput and latency descriptors.

    The reference builds one CUDA graph per ``(gen_bsz, prompt_length,
    gen_length)`` and asserts uniform input *and* output lengths per scenario, so
    each workload becomes a single fixed-shape regime rather than a
    variable-length trace. Prompt *content* still comes from the workload's real
    dataset; only the lengths are normalized.
    """
    from fastkernels.workloads import WORKLOAD_SPECS, LLM, Purpose

    by_value = {w.value: w for w in LLM}
    throughput: list[dict] = []
    latency: list[dict] = []
    for name in names:
        workload = by_value.get(name)
        if workload is None:
            raise SystemExit(
                f"ERROR: unknown workload {name!r} for bench_microsoft_bitnet. "
                f"Known LLM workloads: {', '.join(sorted(by_value))}."
            )
        spec = WORKLOAD_SPECS[workload]
        params = spec.params
        if spec.purpose is Purpose.THROUGHPUT:
            throughput.append({
                "name": name,
                "num_requests": params.num_requests,
                "dataset_name": params.dataset_name,
                "output_len": (
                    params.decode_cap
                    if params.decode_cap is not None
                    else DEFAULT_LONG_CONTEXT_OUTPUT_LEN
                ),
            })
        else:
            latency.append({
                "name": name,
                "batch_size": params.batch_size,
                "output_len": params.output_len,
                "dataset_name": params.dataset_name,
                "num_warmup": params.num_warmup,
                "num_iters": params.num_iters,
            })
    return throughput, latency


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


def _build_random_token_prompts(num_prompts: int, input_len: int,
                                vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randint(2, vocab_size - 1) for _ in range(input_len)]
        for _ in range(num_prompts)
    ]


def _normalize_prompt_len(prompt_ids: list[int], input_len: int,
                          pad_token_id: int) -> list[int]:
    """Return exactly ``input_len`` tokens while keeping the generation edge.

    The Microsoft BitNet GPU runner captures one CUDA graph per fixed
    prompt length.  For real text prompts, keep the suffix when a prompt
    is long and left-pad short prompts so the last token remains real text.
    """
    if len(prompt_ids) >= input_len:
        return list(prompt_ids[-input_len:])
    return [pad_token_id] * (input_len - len(prompt_ids)) + list(prompt_ids)


def _build_real_token_prompts(
    tokenizer,
    scenario_name: str,
    num_prompts: int,
    input_len: int | None,
    output_len: int,
    seed: int,
    split: str,
    dataset_name: str | None = None,
    max_input_len: int | None = None,
    keep_prompts: int | None = None,
) -> tuple[list[list[int]], str, tuple[int, int, int], int, float]:
    try:
        from datasets import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        # datasets imports multiprocess, whose Python 3.12 ResourceTracker
        # destructor can emit an ignored shutdown exception after successful
        # runs.  Silence only that process-exit noise in this benchmark.
        from multiprocess import resource_tracker

        resource_tracker.ResourceTracker.__del__ = lambda self: None
    except Exception:
        pass

    from fastkernels.workloads import (
        DEFAULT_WORKLOAD_DATASETS,
        load_real_prompt_workload,
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 1

    # The Microsoft BitNet GPU kernel captures one CUDA graph per fixed prompt
    # length, so each workload becomes a single fixed (input_len, output_len)
    # regime. The tokens come from that workload's own dataset -- ``mixed`` ->
    # wildchat-mixed-1k, ``long-context`` -> longbench-longctx -- and are then
    # normalized to the regime length (suffix-keep / left-pad).
    dataset_id = dataset_name or DEFAULT_WORKLOAD_DATASETS["mixed"]
    samples = load_real_prompt_workload(
        scenario_name,
        tokenizer,
        num_requests=num_prompts,
        decode_cap=output_len,
        dataset_name=dataset_id,
        split=split,
        seed=seed,
    )
    raw_lens = sorted(len(sample.prompt_token_ids) for sample in samples)
    mean_len = sum(raw_lens) / len(raw_lens)
    length_stats = (raw_lens[0], raw_lens[len(raw_lens) // 2], raw_lens[-1])
    # With no explicit length, use the *mean* real prompt length, clamped to what
    # the model can represent. Mean rather than median because these are
    # throughput workloads: the mean preserves the total prefill token count of
    # the real trace, so the fixed-shape run does the same aggregate prefill work.
    # The median looked more principled but destroys that -- wildchat-mixed is
    # mostly short prompts (p50 60 vs mean ~150), so a median-length `mixed`
    # becomes a pure-decode workload with no prefill at all.
    if input_len is None:
        input_len = max(int(round(mean_len)), 1)
    if max_input_len is not None:
        input_len = max(min(input_len, max_input_len), 1)
    prompts = [
        _normalize_prompt_len(sample.prompt_token_ids, input_len, int(pad_id))
        for sample in samples
    ]
    # ``keep_prompts`` trims the *timed* request list only. ``input_len`` and the
    # length stats above stay derived from the workload's full declared trace, so
    # a capped run still measures the declared fixed shape instead of drifting to
    # whatever mean the subsample happens to have.
    if keep_prompts is not None:
        prompts = prompts[:keep_prompts]
    return prompts, dataset_id, length_stats, input_len, mean_len


# ---------------------------------------------------------------------------
# fastkernels subprocess worker.
# Returns per-request output token ids in ``outputs`` so the parent can
# do per-scenario alignment against the SOTA reference.
# ---------------------------------------------------------------------------
KB_WORKER = r'''
import json, os, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])
if cfg.get("bitnet_kernel_so"):
    os.environ.setdefault("KB_BITNET_KERNEL_LIB", cfg["bitnet_kernel_so"])

# fastsafetensors GDS path is unreliable on some hosts; force the
# threaded safetensors loader so the bench focuses on inference perf.
from fastkernels.infra import weight_loader as _wl
_wl._HAS_FASTSAFETENSORS = False

from fastkernels.infra.engine import LlamaEngine, SamplingParams

engine = LlamaEngine(
    model_name=cfg["model"],
    seed=cfg["seed"],
    enforce_eager=cfg.get("enforce_eager", True),
    tensor_parallel_size=cfg["tp"],
    max_model_len=cfg["max_model_len"],
    max_num_seqs=cfg.get("kb_bsz") if int(cfg.get("kb_bsz", 1)) > 0 else None,
)

# Warmup
engine.generate([[0] * 16], SamplingParams(temperature=0.0, max_tokens=16))

results = []
kb_bsz = int(cfg.get("kb_bsz", 1))
# Seconds of silence after which a serial loop prints progress; the sweep kills
# a task whose log has not moved for 900s (--stall-timeout).
HEARTBEAT_SEC = 30.0
for sc in cfg["scenarios"]:
    prompts = sc["prompt_token_ids"]
    out_lens = sc["output_lens"]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
        for ol in out_lens
    ]
    engine.block_manager.reset()
    torch.cuda.synchronize()
    # At kb_bsz > 0 the engine is built with max_num_seqs=kb_bsz, so it already
    # serves at most that many at once; issuing the prompts in kb_bsz-sized
    # calls is the same work and the same concurrency, but it gives the loop a
    # place to report progress. One 64-prompt call at kb_bsz=1 is ~2min of
    # silence, and the declared 1000 would be ~34min -- past the watchdog.
    # kb_bsz == 0 means "hand everything to the continuous scheduler", which
    # must stay a single call for the scheduler to batch across requests.
    chunk = kb_bsz if kb_bsz > 0 else len(prompts)
    outs = []
    elapsed = 0.0
    last_beat = time.perf_counter()
    for lo in range(0, len(prompts), max(1, chunk)):
        hi = min(lo + max(1, chunk), len(prompts))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs.extend(engine.generate(prompts[lo:hi], sps[lo:hi], use_tqdm=False))
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        if time.perf_counter() - last_beat >= HEARTBEAT_SEC:
            print(f"[kb] {sc['name']:>14}: {hi}/{len(prompts)} prompts, "
                  f"{elapsed:6.1f}s elapsed", flush=True)
            last_beat = time.perf_counter()
    n_in = sum(len(p) for p in prompts)
    n_out = sum(len(o.token_ids) for o in outs)
    out_records = [{"token_ids": list(o.token_ids)} for o in outs]
    results.append({
        "name": sc["name"], "elapsed": elapsed,
        "total_input_tokens": n_in, "total_output_tokens": n_out,
        "num_prompts": len(prompts),
        "kb_bsz": (kb_bsz if kb_bsz > 0 else len(prompts)),
        "outputs": out_records,
    })
    print(f"[kb] {sc['name']:>14}: {elapsed:7.2f}s  "
          f"in={n_in:>8d}  out={n_out:>8d}  "
          f"throughput={(n_in + n_out)/elapsed:>8.1f} tok/s",
          flush=True)

# Latency phase: one fixed batch, timed repeatedly, median reported. Separate
# from the throughput phase above because the metric is per-batch wall clock
# rather than aggregate tokens/s.
latency_results = []
for sc in cfg.get("latency_scenarios") or []:
    prompts = sc["prompt_token_ids"]
    out_lens = sc["output_lens"]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
        for ol in out_lens
    ]
    for _ in range(int(sc.get("num_warmup", 3))):
        engine.block_manager.reset()
        engine.generate(prompts, sps, use_tqdm=False)
    torch.cuda.synchronize()
    samples = []
    outs = None
    for _ in range(int(sc.get("num_iters", 5))):
        engine.block_manager.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = engine.generate(prompts, sps, use_tqdm=False)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples_sorted = sorted(samples)
    median = samples_sorted[len(samples_sorted) // 2]
    n_out = sum(len(o.token_ids) for o in outs) if outs else 0
    latency_results.append({
        "name": sc["name"],
        "batch_size": sc.get("batch_size", len(prompts)),
        "output_len": out_lens[0] if out_lens else 0,
        "num_iters": len(samples),
        "median": median,
        "mean": sum(samples) / len(samples),
        "p99": samples_sorted[-1],
        "latencies": samples,
        "ms_per_token": (median * 1000.0 / max(out_lens[0], 1)) if out_lens else None,
        "outputs": [{"token_ids": list(o.token_ids)} for o in (outs or [])],
    })
    print(f"[kb-lat] {sc['name']:>14}: median={median * 1000:8.2f}ms  "
          f"bs={sc.get('batch_size')}  out={out_lens[0] if out_lens else 0}  "
          f"n_out={n_out}", flush=True)

with open(cfg["output_file"], "w") as f:
    json.dump({
        "throughput": results,
        "latency": latency_results,
        "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }, f)
'''


# ---------------------------------------------------------------------------
# Microsoft BitNet GPU subprocess worker (official SOTA).
#
# The official lib hard-pins (gen_bsz, prompt_length, gen_length) at
# CUDA-graph capture time, so we re-build a FastGen instance per scenario.
# Its CUDA int2 decode kernels only dispatch for M == 1, so gen_bsz must
# remain 1. Build cost (compile_prefill + compile_generate) is excluded
# from the timed window; only ``generate_all`` is timed. Per-prompt outputs
# are captured into ``outputs`` for alignment scoring.
#
# Upstream ``generate.py`` has two bugs this worker fixes locally, leaving the
# provisioned third_party checkout untouched:
#
# 1. Batched prefill indexing: ``output[kv_seqlen - 1, :]`` ignores the
#    flattened per-request prompt offset.
# 2. The decode CUDA graph is captured with the AttnBias built at
#    ``kv_seqlen=[prompt_length]``, so the replayed decode attends over a
#    prompt-length window and never sees the tokens it generated. See
#    ``recapture_generate_full_window``.
# ---------------------------------------------------------------------------
SOTA_WORKER = r'''
import json, math, os, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

bitnet_repo = cfg["bitnet_repo"]
gpu_dir = os.path.join(bitnet_repo, "gpu")
sys.path.insert(0, gpu_dir)
# the official model.py loads ./bitnet_kernels/libbitnet.so via a
# relative path, so we have to chdir into ./gpu/ before importing it.
os.chdir(gpu_dir)

import generate as _bitnet_generate
import model as _bitnet_model

ckpt_dir = cfg["ckpt_dir"]
gen_bsz = int(cfg["gen_bsz"])
# Seconds of silence after which a serial loop prints progress. The sweep's
# watchdog kills a task whose log has not moved for 900s (--stall-timeout).
HEARTBEAT_SEC = 30.0
torch.cuda.set_device(0)
if gen_bsz != 1:
    raise ValueError(
        "Microsoft BitNet GPU decode kernels only implement M == 1; "
        f"got gen_bsz={gen_bsz}. Use --gen-bsz 1."
    )

@torch.inference_mode()
def generate_all_fixed(g, prompts, use_cuda_graphs, use_sampling):
    bs = len(prompts)
    prompt_lens = [len(p) for p in prompts]
    padded_prompt_lens = [g.gen_args.prompt_length] * bs
    max_prompt_length = max(prompt_lens)
    gen_length = g.gen_args.gen_length
    max_seq_length = max_prompt_length + gen_length

    bias = _bitnet_generate.AttnBias.from_seqlens(
        q_seqlen=padded_prompt_lens,
        kv_seqlen=prompt_lens,
        kv_padding=max_seq_length,
    )
    bias.q_seqinfo.to("cuda")
    bias.k_seqinfo.to("cuda")

    kv_seqlen = bias.k_seqinfo.seqlen
    padded = [
        prompt + [1] * (g.gen_args.prompt_length - len(prompt))
        for prompt in prompts
    ]
    tokens = torch.IntTensor(sum(padded, [])).cuda()
    out_tokens = torch.zeros(
        (max_seq_length, bs), dtype=torch.int, device=tokens.device,
    )

    stats = _bitnet_generate.Stats()
    torch.cuda.synchronize()
    stats.phase("prefill" if use_cuda_graphs else "total")

    output = g._prefill_compile_model(tokens, None)

    # Fixed upstream bug: prefill logits are flattened as
    # [request0 padded prompt][request1 padded prompt]..., so each
    # request needs its own row offset before selecting its last real
    # prompt token.
    row_offsets = (
        torch.arange(bs, device=kv_seqlen.device, dtype=kv_seqlen.dtype)
        * g.gen_args.prompt_length
    )
    logits = output[row_offsets + kv_seqlen - 1, :]
    logits = logits.view(bs, g.model_args.vocab_size)

    if use_sampling:
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_token = _bitnet_generate.sample_utils.top_p(probs, 0.95)
    else:
        next_token = torch.argmax(logits, dim=-1)

    next_token = next_token.reshape(bs)
    out_tokens[0, :] = next_token

    torch.cuda.synchronize()
    stats.phase("decode" if use_cuda_graphs else "total")

    eos_id = g.tokenizer.eot_id
    niter = 1
    for niter in range(1, gen_length):
        kv_seqlen.add_(kv_seqlen < max_seq_length)
        output = g._generate_compile_model(next_token, kv_seqlen)
        logits = output.view(bs, g.model_args.vocab_size)

        if use_sampling:
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_token = _bitnet_generate.sample_utils.top_p(probs, 0.95)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.reshape(bs)
        out_tokens[niter, :] = next_token

        if next_token.eq(eos_id).any():
            break

    torch.cuda.synchronize()
    stats.end_phase(tokens=niter * bs)

    def trim_answer(prompt_len, tokens):
        tokens = tokens[: max_seq_length - prompt_len]
        eos_id = g.tokenizer.eot_id
        if eos_id in tokens:
            return tokens[: tokens.index(eos_id) + 1]
        return tokens

    answers = [
        trim_answer(prompt_len, answer)
        for prompt_len, answer in zip(prompt_lens, out_tokens.t().tolist())
    ]
    return stats, answers

@torch.inference_mode()
def recapture_generate_full_window(g):
    """Re-record the decode CUDA graph over the *full* KV window.

    Upstream ``FastGen.compile_generate`` (generate.py:170) builds the decode
    graph's AttnBias with ``kv_seqlen=[prompt_length]``.  ``AttnBias.from_seqlens``
    precomputes Python-side ``max_seqlen``/``min_seqlen`` next to the device
    ``seqlen`` tensor, and those scalars are baked into the captured kernel's
    launch configuration.  ``replay()`` copies fresh seqlen *values* in, which
    only re-masks *within* that frozen window -- so the replayed decode attends
    over a ``prompt_length``-key window forever and never sees the tokens it just
    generated.  Nothing errors; the model simply computes the wrong thing.

    Capturing at ``max_seq_length`` makes the window cover the whole allocation,
    and the per-step seqlen copy then masks correctly.  Measured 2026-08-02 at
    prompt_len=298 / gen_len=1024 over 4 real wildchat prompts: graph vs eager
    decode went from diverging at 87/92/88/95 tokens to 1024/1024 exact on all
    four.  Both paths are individually deterministic, so the divergence was a
    real correctness bug, not nondeterminism.

    Cost: 2.476 -> 2.532 ms/token (2.3%), i.e. what the reference always owed for
    attending over the window it actually has.  In exchange the timed pass
    becomes usable for alignment, which removes a second, eager generation pass
    that cost ~37 ms/token.
    """
    gen_bsz = g.gen_args.gen_bsz
    max_seq = g.max_seq_length
    bias = _bitnet_generate.AttnBias.from_seqlens(
        q_seqlen=[1] * gen_bsz,
        kv_seqlen=[max_seq] * gen_bsz,
        kv_padding=max_seq,
    )
    bias.q_seqinfo.to("cuda")
    bias.k_seqinfo.to("cuda")
    tokens = torch.IntTensor([1] * gen_bsz).cuda()
    g._generate_inputs = (tokens, bias)

    # Warm on a side stream before capture, exactly as compile_generate does.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _ = g.decode_model.forward_with_attn_bias(
            token_values=tokens, attn_bias=bias, cache=g._cache,
        )
    torch.cuda.current_stream().wait_stream(s)

    g._generate_cuda_graph = torch.cuda.CUDAGraph()
    recording_kwargs = {}
    if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
        recording_kwargs["capture_error_mode"] = "thread_local"
    with torch.cuda.graph(g._generate_cuda_graph, **recording_kwargs):
        g._generate_logits = g.decode_model.forward_with_attn_bias(
            token_values=tokens, attn_bias=bias, cache=g._cache,
        )

    def replay(t, seq_lens):
        g._generate_inputs[0].copy_(t)
        g._generate_inputs[1].k_seqinfo.seqlen.copy_(seq_lens)
        g._generate_cuda_graph.replay()
        return g._generate_logits

    g._generate_compile_model = replay


@torch.inference_mode()
def generate_all_direct_fixed(g, prompts, use_sampling):
    """Correctness reference: official models with fresh decode metadata.

    Rebuilds the AttnBias every step (``Transformer.forward``, model.py:283) so
    no capture-time metadata can go stale.  Kept as a cross-check behind
    ``--alignment-reference direct``: since ``recapture_generate_full_window``
    fixes the capture, the timed graph path now produces byte-identical tokens to
    this loop, so the default no longer pays ~37 ms/token to regenerate them.
    """
    bs = len(prompts)
    prompt_lens = [len(p) for p in prompts]
    assert bs == 1, "direct reference follows official M == 1 decode kernels"
    max_prompt_length = max(prompt_lens)
    gen_length = g.gen_args.gen_length
    max_seq_length = max_prompt_length + gen_length

    bias = _bitnet_generate.AttnBias.from_seqlens(
        q_seqlen=prompt_lens,
        kv_seqlen=prompt_lens,
        kv_padding=max_seq_length,
    )
    bias.q_seqinfo.to("cuda")
    bias.k_seqinfo.to("cuda")

    tokens = torch.IntTensor(sum(prompts, [])).cuda()
    out_tokens = torch.zeros(
        (max_seq_length, bs), dtype=torch.int, device=tokens.device,
    )

    output = g.prefill_model.forward_with_attn_bias(
        token_values=tokens,
        attn_bias=bias,
        cache=g._cache,
    )
    logits = output[torch.tensor(prompt_lens, device=tokens.device) - 1, :]
    logits = logits.view(bs, g.model_args.vocab_size)

    if use_sampling:
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_token = _bitnet_generate.sample_utils.top_p(probs, 0.95)
    else:
        next_token = torch.argmax(logits, dim=-1)

    next_token = next_token.reshape(bs).to(torch.int32)
    out_tokens[0, :] = next_token

    token_lengths = torch.ones(bs, dtype=torch.int32, device=tokens.device)
    start_pos = torch.tensor(prompt_lens, dtype=torch.int32, device=tokens.device)
    eos_id = g.tokenizer.eot_id
    niter = 1
    for niter in range(1, gen_length):
        output = g.decode_model.forward(
            next_token,
            token_lengths,
            start_pos,
            g._cache,
            max_seq_length,
        )
        logits = output.view(bs, g.model_args.vocab_size)

        if use_sampling:
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_token = _bitnet_generate.sample_utils.top_p(probs, 0.95)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.reshape(bs).to(torch.int32)
        out_tokens[niter, :] = next_token
        start_pos.add_(start_pos < max_seq_length)

        if next_token.eq(eos_id).any():
            break

    def trim_answer(prompt_len, tokens):
        tokens = tokens[: max_seq_length - prompt_len]
        eos_id = g.tokenizer.eot_id
        if eos_id in tokens:
            return tokens[: tokens.index(eos_id) + 1]
        return tokens

    return [
        trim_answer(prompt_len, answer)
        for prompt_len, answer in zip(prompt_lens, out_tokens.t().tolist())
    ]

results = []
for sc in cfg["scenarios"]:
    prompts = sc["prompt_token_ids"]
    out_lens = sc["output_lens"]
    assert all(ol == out_lens[0] for ol in out_lens), \
        "Microsoft BitNet GPU worker requires uniform output length per scenario"
    in_len = len(prompts[0])
    assert all(len(p) == in_len for p in prompts), \
        "Microsoft BitNet GPU worker requires uniform input length per scenario"
    out_len = out_lens[0]

    print(f"[sota] building FastGen for "
          f"gen_bsz={gen_bsz}, prompt_len={in_len}, gen_len={out_len}...",
          flush=True)
    build_t0 = time.perf_counter()
    args = _bitnet_generate.GenArgs(
        prompt_length=in_len, gen_length=out_len, gen_bsz=gen_bsz,
    )
    g = _bitnet_generate.FastGen.build(ckpt_dir, args, "cuda:0")
    # Upstream generate.py expects ``tokenizer.eot_id`` for the
    # early-stop check, but the bundled Llama-3 tiktoken Tokenizer only
    # exposes ``eos_id``.  We force-disable early-stop by setting an
    # unreachable id so the bench runs ``gen_length`` decode iterations
    # for every prompt (ignore_eos=True semantics, matching fastkernels).
    g.tokenizer.eot_id = -1
    recapture_generate_full_window(g)
    torch.cuda.synchronize()
    print(f"[sota] build took {time.perf_counter() - build_t0:.1f}s",
          flush=True)

    # Warmup
    warm = [prompts[0][:in_len]] * gen_bsz
    generate_all_fixed(g, warm, use_cuda_graphs=True, use_sampling=False)
    torch.cuda.synchronize()

    n_batches = math.ceil(len(prompts) / gen_bsz)
    n_in = 0
    n_out = 0
    elapsed_total = 0.0
    last_beat = time.perf_counter()
    align_count = min(int(cfg.get("alignment_prompts", len(prompts))), len(prompts))
    align_ref = cfg.get("alignment_reference", "timed")
    # With the decode graph recaptured over the full window, the timed pass emits
    # the same tokens the eager path does, so alignment reuses them for free --
    # the way every other harness works.  ``direct`` re-runs the eager path
    # instead (~37 ms/token) as a cross-check.
    out_records = []
    for bi in range(n_batches):
        batch = prompts[bi * gen_bsz:(bi + 1) * gen_bsz]
        # Pad the last batch with copies of the first prompt so the
        # CUDA graph shapes still match.  These padded outputs are
        # NOT counted toward the throughput numerator/denominator and
        # are NOT written into out_records.
        real_count = len(batch)
        while len(batch) < gen_bsz:
            batch.append(batch[0])

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _stats, answers = generate_all_fixed(
            g, batch, use_cuda_graphs=True, use_sampling=False,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        elapsed_total += dt
        n_in += real_count * in_len
        n_out += real_count * out_len
        if align_ref == "timed":
            for answer in answers[:real_count]:
                if len(out_records) >= align_count:
                    break
                out_records.append({"token_ids": list(answer)[:out_len]})
        # gen_bsz is pinned to 1, so a scenario is minutes of otherwise silent
        # work and the sweep kills a task whose log has not moved for 900s.
        # Printed outside the timed window so it cannot affect the measurement.
        if time.perf_counter() - last_beat >= HEARTBEAT_SEC:
            print(f"[sota] {sc['name']:>14}: {bi + 1}/{n_batches} batches, "
                  f"{elapsed_total:6.1f}s elapsed", flush=True)
            last_beat = time.perf_counter()

    if align_ref == "direct" and align_count > 0:
        print(f"[sota] generating {align_count} direct-reference output(s) "
              f"for alignment...", flush=True)
        last_beat = time.perf_counter()
        for ai, prompt in enumerate(prompts[:align_count]):
            answers = generate_all_direct_fixed(
                g, [prompt], use_sampling=False,
            )
            out_records.append({"token_ids": list(answers[0])[:out_len]})
            # The direct path is uncompiled (~37 ms/token), so this loop also
            # needs to stay visible to the watchdog.
            if time.perf_counter() - last_beat >= HEARTBEAT_SEC:
                print(f"[sota] {sc['name']:>14}: alignment "
                      f"{ai + 1}/{align_count}", flush=True)
                last_beat = time.perf_counter()

    results.append({
        "name": sc["name"], "elapsed": elapsed_total,
        "total_input_tokens": n_in, "total_output_tokens": n_out,
        "num_prompts": len(prompts), "gen_bsz": gen_bsz,
        "alignment_reference": (
            "official_cuda_graph_full_window" if align_ref == "timed"
            else "official_direct_decode"
        ),
        "timed_reference": "official_cuda_graph_full_window",
        "decode_graph_capture": "max_seq_length",
        "outputs": out_records,
    })
    print(f"[sota] {sc['name']:>14}: {elapsed_total:7.2f}s  "
          f"in={n_in:>8d}  out={n_out:>8d}  "
          f"throughput={(n_in + n_out) / elapsed_total:>8.1f} tok/s",
          flush=True)

    # Free per-scenario state before the next FastGen build.
    del g
    torch.cuda.empty_cache()

# Latency phase. The official int2 decode kernels dispatch only for M == 1, so
# the reference can only serve one sequence at a time: a batch_size > 1 probe has
# no like-for-like reference here and is skipped with a machine-readable reason
# rather than silently compared against a serial loop.
latency_results = []
for sc in cfg.get("latency_scenarios") or []:
    bs = int(sc.get("batch_size", 1))
    prompts = sc["prompt_token_ids"]
    out_lens = sc["output_lens"]
    in_len = len(prompts[0])
    out_len = out_lens[0]
    if bs != 1:
        latency_results.append({
            "name": sc["name"], "batch_size": bs, "output_len": out_len,
            "unsupported": True,
            "reason": ("official int2 decode kernel dispatches only for M == 1; "
                       "the reference cannot batch"),
        })
        print(f"[sota-lat] {sc['name']:>14}: SKIPPED (bs={bs} > 1, "
              f"reference is M == 1 only)", flush=True)
        continue

    print(f"[sota-lat] building FastGen for gen_bsz=1, prompt_len={in_len}, "
          f"gen_len={out_len}...", flush=True)
    largs = _bitnet_generate.GenArgs(
        prompt_length=in_len, gen_length=out_len, gen_bsz=1,
    )
    g = _bitnet_generate.FastGen.build(ckpt_dir, largs, "cuda:0")
    g.tokenizer.eot_id = -1
    recapture_generate_full_window(g)
    for _ in range(int(sc.get("num_warmup", 3))):
        generate_all_fixed(g, [prompts[0]], use_cuda_graphs=True,
                           use_sampling=False)
    torch.cuda.synchronize()
    samples = []
    for _ in range(int(sc.get("num_iters", 5))):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_all_fixed(g, [prompts[0]], use_cuda_graphs=True,
                           use_sampling=False)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples_sorted = sorted(samples)
    median = samples_sorted[len(samples_sorted) // 2]
    latency_results.append({
        "name": sc["name"], "batch_size": bs, "output_len": out_len,
        "num_iters": len(samples), "median": median,
        "mean": sum(samples) / len(samples), "p99": samples_sorted[-1],
        "latencies": samples,
        "ms_per_token": median * 1000.0 / max(out_len, 1),
    })
    print(f"[sota-lat] {sc['name']:>14}: median={median * 1000:8.2f}ms",
          flush=True)
    del g
    torch.cuda.empty_cache()

with open(cfg["output_file"], "w") as f:
    json.dump({
        "throughput": results,
        "latency": latency_results,
        "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }, f)

# The official reference uses torch.compile internally. In this environment
# Inductor's atexit compile-worker shutdown can wait 300s after all benchmark
# work is complete. The worker has already written its JSON result, so exit
# directly to keep measured wall-clock time focused on the benchmark.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
'''


# ---------------------------------------------------------------------------
# Microsoft BitNet direct-decode top-k scoring worker.
#
# Exact free-running token prefixes are brittle on natural-language prompts:
# once two numerically close logits choose different but plausible tokens,
# the rest of the sequence diverges.  This worker scores both SOTA-generated
# and fastkernels-generated sequences under the official direct-decode path with
# teacher forcing, matching the "feed generated text back to the reference
# model and check top-k" alignment used by other LLM benchmarks.
# ---------------------------------------------------------------------------
SOTA_SCORE_WORKER = r'''
import json, os, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

bitnet_repo = cfg["bitnet_repo"]
gpu_dir = os.path.join(bitnet_repo, "gpu")
sys.path.insert(0, gpu_dir)
os.chdir(gpu_dir)

import generate as _bitnet_generate

ckpt_dir = cfg["ckpt_dir"]
topks = tuple(int(k) for k in cfg.get("topks", [1, 5, 20]))
max_topk = max(topks)
# Seconds of silence after which the scoring loop prints progress; the sweep
# kills a task whose log has not moved for 900s (--stall-timeout).
HEARTBEAT_SEC = 30.0
torch.cuda.set_device(0)

@torch.inference_mode()
def score_sequence_direct(g, prompt, answer):
    answer = list(answer)
    if not answer:
        return {str(k): 0 for k in topks}, 0

    prompt_len = len(prompt)
    max_seq_length = prompt_len + len(answer)
    counts = {str(k): 0 for k in topks}

    def score_logits(logits, target):
        top = torch.topk(logits.float(), k=max_topk, dim=-1).indices[0]
        eq = top.eq(int(target))
        for k in topks:
            if bool(eq[:k].any().item()):
                counts[str(k)] += 1

    bias = _bitnet_generate.AttnBias.from_seqlens(
        q_seqlen=[prompt_len],
        kv_seqlen=[prompt_len],
        kv_padding=max_seq_length,
    )
    bias.q_seqinfo.to("cuda")
    bias.k_seqinfo.to("cuda")

    tokens = torch.IntTensor(prompt).cuda()
    output = g.prefill_model.forward_with_attn_bias(
        token_values=tokens,
        attn_bias=bias,
        cache=g._cache,
    )
    logits = output[prompt_len - 1, :].view(1, g.model_args.vocab_size)
    score_logits(logits, answer[0])

    next_token = torch.tensor(
        [int(answer[0])], dtype=torch.int32, device="cuda",
    )
    token_lengths = torch.ones(1, dtype=torch.int32, device="cuda")
    start_pos = torch.tensor([prompt_len], dtype=torch.int32, device="cuda")
    for target in answer[1:]:
        output = g.decode_model.forward(
            next_token,
            token_lengths,
            start_pos,
            g._cache,
            max_seq_length,
        )
        logits = output.view(1, g.model_args.vocab_size)
        score_logits(logits, target)
        next_token = torch.tensor(
            [int(target)], dtype=torch.int32, device="cuda",
        )
        start_pos.add_(start_pos < max_seq_length)

    return counts, len(answer)

def empty_score():
    return {
        "total_tokens": 0,
        **{f"top{k}": 0.0 for k in topks},
        "elapsed": 0.0,
    }

results = {}
for sc in cfg["scenarios"]:
    prompts = sc["prompt_token_ids"]
    out_len = sc["output_len"]
    in_len = len(prompts[0])

    print(f"[topk] building FastGen for "
          f"prompt_len={in_len}, gen_len={out_len}...", flush=True)
    args = _bitnet_generate.GenArgs(
        prompt_length=in_len, gen_length=out_len, gen_bsz=1,
    )
    g = _bitnet_generate.FastGen.build(ckpt_dir, args, "cuda:0")
    g.tokenizer.eot_id = -1

    scenario_result = {}
    for group_name, records in sc["groups"].items():
        total = 0
        counts = {str(k): 0 for k in topks}
        t0 = time.perf_counter()
        last_beat = t0
        for si, (prompt, record) in enumerate(zip(prompts, records)):
            seq_counts, n = score_sequence_direct(
                g, prompt, record["token_ids"][:out_len],
            )
            total += n
            for k in topks:
                counts[str(k)] += seq_counts[str(k)]
            # ~40s per 1024-token sequence (opt-in path), so keep it visible.
            if time.perf_counter() - last_beat >= HEARTBEAT_SEC:
                print(f"[topk] {sc['name']:>14} {group_name}: "
                      f"{si + 1}/{len(records)} sequences", flush=True)
                last_beat = time.perf_counter()
        elapsed = time.perf_counter() - t0
        if total == 0:
            score = empty_score()
        else:
            score = {
                "total_tokens": total,
                **{f"top{k}": counts[str(k)] / total for k in topks},
                "elapsed": elapsed,
            }
        scenario_result[group_name] = score
        topk_text = " ".join(
            f"top{k}={score[f'top{k}']:.4f}" for k in topks
        )
        print(f"[topk] {sc['name']:>14} {group_name}: "
              f"{topk_text} tokens={total} elapsed={elapsed:.1f}s",
              flush=True)

    results[sc["name"]] = scenario_result
    del g
    torch.cuda.empty_cache()

with open(cfg["output_file"], "w") as f:
    json.dump({"topk_alignment": results}, f)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
'''


def run_worker(script: str, config: dict, label: str) -> dict | None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(script)
        wpath = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        cpath = f.name
    out_path = config["output_file"]
    try:
        print(f"\n{'-' * 70}\n  {label}\n{'-' * 70}", flush=True)
        r = subprocess.run([sys.executable, wpath, cpath], timeout=14400)
        if r.returncode != 0:
            print(f"  ERROR: {label} exit code {r.returncode}", flush=True)
            return None
        with open(out_path) as f:
            return json.load(f)
    finally:
        os.unlink(wpath)
        os.unlink(cpath)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ---------------------------------------------------------------------------
# Alignment check.
# ---------------------------------------------------------------------------
def compute_alignment(
    a_outputs: list[dict],
    b_outputs: list[dict],
) -> dict:
    """Compare per-request token_ids using consecutive-prefix alignment."""
    total_seqs = min(len(a_outputs), len(b_outputs))
    exact_matches = 0
    total_matching_tokens = 0
    total_position_matches = 0
    total_output_tokens = 0
    prefix_lengths = []

    for a, b in zip(a_outputs[:total_seqs], b_outputs[:total_seqs]):
        a_ids = a["token_ids"]
        b_ids = b["token_ids"]
        out_len = max(len(a_ids), len(b_ids))
        total_output_tokens += out_len

        min_len = min(len(a_ids), len(b_ids))
        prefix = 0
        for j in range(min_len):
            if a_ids[j] != b_ids[j]:
                break
            prefix += 1

        position_matches = sum(1 for j in range(min_len) if a_ids[j] == b_ids[j])
        total_position_matches += position_matches
        total_matching_tokens += prefix
        prefix_lengths.append(prefix)

        if a_ids == b_ids:
            exact_matches += 1

    avg_matching = total_matching_tokens / total_seqs if total_seqs else 0
    avg_position = total_position_matches / total_seqs if total_seqs else 0
    avg_output_len = total_output_tokens / total_seqs if total_seqs else 0
    prefix_lengths.sort()
    median_matching = (
        prefix_lengths[total_seqs // 2] if total_seqs else 0
    )

    return {
        "exact_matches": exact_matches,
        "total_seqs": total_seqs,
        "total_matching_tokens": total_matching_tokens,
        "total_position_matches": total_position_matches,
        "total_output_tokens": total_output_tokens,
        "avg_matching_tokens_per_request": avg_matching,
        "avg_position_matches_per_request": avg_position,
        "avg_output_len": avg_output_len,
        "median_matching_tokens_per_request": median_matching,
    }


def _print_throughput_table(label: str, data: dict):
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
    print(f"{'scenario':>14}  {'elapsed(s)':>10}  {'in':>10}  {'out':>10}  "
          f"{'tok/s':>10}")
    for r in data["throughput"]:
        tps = (r["total_input_tokens"] + r["total_output_tokens"]) / r["elapsed"]
        print(f"{r['name']:>14}  {r['elapsed']:>10.2f}  "
              f"{r['total_input_tokens']:>10d}  {r['total_output_tokens']:>10d}  "
              f"{tps:>10.1f}")
    if "memory_gb" in data:
        print(f"\n  Memory: {data['memory_gb']} GB")


def _print_summary_table(sota_data: dict, kb_data: dict,
                         alignments: dict[str, dict]):
    print(f"\n{'=' * 100}")
    print("  SUMMARY (fastkernels vs Microsoft BitNet GPU)")
    print(f"{'=' * 100}")
    header = (
        f"  {'SCENARIO':<16} {'IN':>5} {'OUT':>5} "
        f"{'FASTKERNELS tok/s':>15} {'SOTA tok/s':>12} {'SPEEDUP':>8} "
        f"{'AVG PREFIX':>15} {'POS MATCH':>12} {'EXACT':>10}"
    )
    print(header)
    print(f"  {'-' * 96}")
    for sr, kr in zip(sota_data["throughput"], kb_data["throughput"]):
        sota_tps = (sr["total_input_tokens"]
                    + sr["total_output_tokens"]) / sr["elapsed"]
        kb_tps = (kr["total_input_tokens"]
                  + kr["total_output_tokens"]) / kr["elapsed"]
        a = alignments.get(sr["name"], {})
        avg_match = a.get("avg_matching_tokens_per_request", 0)
        avg_pos = a.get("avg_position_matches_per_request", 0)
        avg_out = a.get("avg_output_len", 0)
        match_str = f"{avg_match:.1f}/{avg_out:.0f}" if avg_out else "N/A"
        pos_str = f"{avg_pos:.1f}/{avg_out:.0f}" if avg_out else "N/A"
        exact_str = (f"{a.get('exact_matches', 0)}/{a.get('total_seqs', 0)}"
                     if a else "N/A")
        # input length printed from the prompt records (not stored on
        # the throughput dict).
        in_len = sr["total_input_tokens"] // sr["num_prompts"] if sr["num_prompts"] else 0
        out_len = sr["total_output_tokens"] // sr["num_prompts"] if sr["num_prompts"] else 0
        print(
            f"  {sr['name']:<16} {in_len:>5} {out_len:>5} "
            f"{kb_tps:>15,.0f} {sota_tps:>12,.0f} "
            f"{kb_tps / sota_tps:>7.2f}x "
            f"{match_str:>15} {pos_str:>12} {exact_str:>10}"
        )
    print(f"{'=' * 100}")


def _print_topk_alignment_table(topk_alignments: dict[str, dict]):
    print(f"\n{'=' * 100}")
    print("  TEACHER-FORCED TOP-K ALIGNMENT (official direct-decode scorer)")
    print(f"{'=' * 100}")
    header = (
        f"  {'SCENARIO':<16} {'SOTA top1':>10} {'SOTA top20':>11} "
        f"{'KB top1':>10} {'KB top20':>9} {'TOKENS':>8}"
    )
    print(header)
    print(f"  {'-' * 82}")
    for name, scores in topk_alignments.items():
        self_score = scores.get("sota_self", {})
        kb_score = scores.get("kb_under_sota", {})
        print(
            f"  {name:<16} "
            f"{self_score.get('top1', 0):>10.4f} "
            f"{self_score.get('top20', 0):>11.4f} "
            f"{kb_score.get('top1', 0):>10.4f} "
            f"{kb_score.get('top20', 0):>9.4f} "
            f"{int(kb_score.get('total_tokens', 0)):>8d}"
        )
    print(f"{'=' * 100}")


def _persist_results(output_dir: str, model: str, num_prompts: int,
                     sota_data: dict | None, kb_data: dict | None,
                     alignments: dict[str, dict],
                     topk_alignments: dict[str, dict] | None,
                     scenarios: list[dict],
                     enforce_eager: bool = False) -> None:
    summary = {
        "model": model,
        "num_prompts_per_scenario": num_prompts,
        "workload": [
            {
                "name": sc["name"],
                "prompt_source": sc.get("prompt_source"),
                "dataset": sc.get("dataset"),
                "input_len": len(sc["prompt_token_ids"][0]),
                "output_len": sc["output_lens"][0],
                # Declared vs actually-timed request count: both engines are
                # pinned to one request at a time by the reference's M == 1
                # decode kernels, so the timed count is capped (see
                # --max-timed-prompts). Recorded per scenario so a sweep query
                # can tell a capped run from a full one.
                "num_requests_declared": sc.get("num_requests_declared"),
                "num_requests_timed": sc.get("num_requests_timed"),
            }
            for sc in scenarios
        ],
        "sota": sota_data,
        # One run per phase: these throughput numbers and the alignment block
        # below come from the same fastkernels run, in the same execution mode.
        "fastkernels": kb_data,
        "execution_mode": "eager" if enforce_eager else "cudagraph",
        "alignment": alignments,
        "topk_alignment": topk_alignments,
    }
    timed = kb_data

    # Standard comparison shape shared with the other harnesses: a per-scenario
    # `speedup` plus the existing prefix-alignment block. Previously only the
    # raw per-engine numbers were stored, so an aggregate query over a sweep
    # found no speedup for this row.
    scenarios: list[dict] = []
    latency_scenarios: list[dict] = []

    def _tps(row):
        el = row.get("elapsed") or 0
        if el <= 0:
            return None
        return (row.get("total_input_tokens", 0)
                + row.get("total_output_tokens", 0)) / el

    def _sec_per_request(row):
        el = row.get("elapsed") or 0
        n = row.get("num_prompts") or 0
        return (el / n) if el > 0 and n > 0 else None

    if sota_data and timed:
        for sr, kr in zip(sota_data.get("throughput") or [],
                          timed.get("throughput") or []):
            if sr.get("name") != kr.get("name"):
                continue
            scenarios.append(throughput_entry(
                sr["name"], _tps(kr), _tps(sr), metric="tok_per_s",
                alignment=alignments.get(sr["name"]),
                num_seqs=kr.get("num_prompts") or kr.get("num_seqs"),
            ))
            # The official BitNet GPU decode kernels only dispatch for M == 1,
            # so gen_bsz is pinned to 1 and the fastkernels side matches with
            # max_num_seqs=1. Both engines therefore serve these prompts one at
            # a time, which makes elapsed/num_prompts a real per-request
            # latency rather than a batched-throughput artifact.
    # Latency phase rows come from the dedicated latency workloads, not from
    # dividing a throughput run: the probe times one fixed batch repeatedly.
    kb_lat = {r["name"]: r for r in ((kb_data or {}).get("latency") or [])}
    sota_lat = {r["name"]: r for r in ((sota_data or {}).get("latency") or [])}
    for name, kr in kb_lat.items():
        peer = sota_lat.get(name) or {}
        if peer.get("unsupported"):
            # Report the gap explicitly instead of comparing our batched run
            # against a serial reference loop, which would read as a kernel win
            # when it only reflects the reference having no batched kernel.
            latency_scenarios.append(latency_entry(
                name, kr.get("median"), None, metric="median_s",
                batch_size=kr.get("batch_size"),
                output_len=kr.get("output_len"),
                fastkernels_ms_per_token=kr.get("ms_per_token"),
                reference_unsupported=True,
                reference_unsupported_reason=peer.get("reason"),
            ))
            continue
        latency_scenarios.append(latency_entry(
            name, kr.get("median"), peer.get("median"), metric="median_s",
            batch_size=kr.get("batch_size"),
            output_len=kr.get("output_len"),
            fastkernels_ms_per_token=kr.get("ms_per_token"),
            reference_ms_per_token=peer.get("ms_per_token"),
        ))
    summary["reference_name"] = "microsoft-bitnet-gpu"
    summary["scenarios"] = scenarios
    summary["latency_scenarios"] = latency_scenarios
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    by_name: dict[str, dict[str, dict]] = {}
    if sota_data:
        for row in sota_data["throughput"]:
            by_name.setdefault(row["name"], {})["sota_outputs"] = row
    if kb_data:
        for row in kb_data["throughput"]:
            by_name.setdefault(row["name"], {})["fastkernels_outputs"] = row

    for scenario_name, records in by_name.items():
        sdir = os.path.join(output_dir, scenario_name)
        os.makedirs(sdir, exist_ok=True)
        for stem, row in records.items():
            with open(os.path.join(sdir, f"{stem}.json"), "w") as f:
                json.dump(row, f)
        if scenario_name in alignments:
            with open(os.path.join(sdir, "alignment.json"), "w") as f:
                json.dump(alignments[scenario_name], f, indent=2)
        if topk_alignments and scenario_name in topk_alignments:
            with open(os.path.join(sdir, "topk_alignment.json"), "w") as f:
                json.dump(topk_alignments[scenario_name], f, indent=2)

    print(f"\n  Results saved under: {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="Override the request count for throughput "
                         "workloads. Default: each workload's declared "
                         "num_requests (mixed=1000, long-context=64). Latency "
                         "workloads always use their declared batch_size.")
    ap.add_argument("--max-timed-prompts", type=int, default=64,
                    help="Cap on how many requests each throughput workload "
                         "actually times, applied only when --num-prompts is "
                         "not given. The reference's int2 decode kernels only "
                         "dispatch for M == 1, so both engines serve requests "
                         "one at a time (~2.6s each at mixed's shape) and the "
                         "declared 1000 would take ~44min per side -- past the "
                         "sweep's watchdog. Each request is padded to the same "
                         "fixed shape with early stop disabled, so this trims "
                         "repetitions rather than coverage; the prompt length "
                         "is still derived from the full declared trace.")
    ap.add_argument("--workloads", type=str,
                    default=",".join(DEFAULT_WORKLOADS),
                    help="Comma-separated LLM workload names to run, as "
                         "declared in the scenario file (e.g. "
                         "'mixed,long-context,single-request'). Throughput and "
                         "latency workloads are split automatically by their "
                         "declared purpose.")
    ap.add_argument("--skip-throughput", action="store_true",
                    help="Skip the throughput phase (latency only)")
    ap.add_argument("--skip-latency", action="store_true",
                    help="Skip the latency phase (throughput only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-size", type=int, default=128256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--prompt-source", choices=("real", "random"),
                    default="real",
                    help="Prompt content source. 'real' uses the "
                         "WildChat-derived fastkernels workload datasets and "
                         "normalizes them to fixed BitNet SOTA graph shapes; "
                         "'random' keeps the old deterministic token-id "
                         "debug workload.")
    ap.add_argument("--dataset-split", default="train",
                    help="HF dataset split for --prompt-source real")
    ap.add_argument("--bitnet-repo",
                    default=os.environ.get(
                        "BITNET_REPO",
                        str(BITNET_DIR)),
                    help="Path to the Microsoft BitNet repo "
                         "(must contain gpu/checkpoints/model_state_int2.pt and "
                         "gpu/bitnet_kernels/libbitnet.so). Provisioned by "
                         "`python -m fastkernels.validate.provision bitnet`.")
    ap.add_argument("--gen-bsz", type=int, default=1,
                    help="CUDA-graph batch size for the Microsoft BitNet "
                         "GPU worker. Must be 1: the official int2 decode "
                         "kernels only implement M == 1.")
    ap.add_argument("--alignment-prompts", type=int, default=32,
                    help="Number of prompts per scenario compared against the "
                         "reference. These come from the timed run itself, so "
                         "raising this is free up to --max-timed-prompts.")
    ap.add_argument("--alignment-reference", choices=("timed", "direct"),
                    default="timed",
                    help="Which reference tokens alignment compares against. "
                         "'timed' (default) reuses the CUDA-graph run's own "
                         "output, which is byte-identical to eager decode once "
                         "the graph is captured over the full KV window. "
                         "'direct' re-generates through the eager path as a "
                         "cross-check, at ~37 ms/token.")
    ap.add_argument("--kb-bsz", type=int, default=1,
                    help="Number of requests per fastkernels generate() call. "
                         "Default 1 matches the Microsoft BitNet GPU "
                         "baseline's M==1 decode limit; use 0 to benchmark "
                         "fastkernels's continuous scheduler over all prompts.")
    ap.add_argument("--enforce-eager", action="store_true", default=False,
                    help="Disable fastkernels CUDA graphs. Default is "
                         "non-eager, matching bench_vllm.py and the "
                         "reference, which times its own CUDA-graph path. "
                         "Correctness and speed are measured in the same run.")
    ap.add_argument("--topk-alignment", dest="skip_topk_alignment",
                    action="store_false", default=True,
                    help="Also score teacher-forced top-k agreement under the "
                         "official direct-decode reference. Off by default: "
                         "the scorer decodes one token at a time with three "
                         "host syncs per token (~40s per 1024-token prompt per "
                         "group), which at the default --alignment-prompts "
                         "costs more than both timed throughput loops "
                         "combined. AVG PREFIX / EXACT already compare the "
                         "generated token ids directly.")
    ap.add_argument("--skip-topk-alignment", dest="skip_topk_alignment",
                    action="store_true", default=True,
                    help="Deprecated no-op: top-k scoring is already off by "
                         "default. Use --topk-alignment to turn it on.")
    ap.add_argument("--skip-sota", action="store_true",
                    help="Skip the Microsoft BitNet GPU SOTA reference run")
    ap.add_argument("--skip-kb", action="store_true",
                    help="Skip fastkernels (SOTA only)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory to save per-scenario outputs and "
                         "alignment json (default: tests/results/<gpu>/"
                         "<model>_bitnet)")
    args = ap.parse_args()

    gpu = _detect_gpu_name()
    print("=" * 70)
    print(f"  BitNet bench: {args.model}")
    print(f"  GPU: {gpu} | num_prompts/scenario: {args.num_prompts}")
    print(f"  Prompt source: {args.prompt_source}")
    print("=" * 70)

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(
            _PROJECT_ROOT / "tests" / "results" / gpu / f"{short}_bitnet"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_dir = os.path.join(args.bitnet_repo, "gpu", "checkpoints")
    kernel_so = os.path.join(args.bitnet_repo, "gpu",
                             "bitnet_kernels", "libbitnet.so")

    tokenizer = None
    if args.prompt_source == "real":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=False,
        )

    tp_workloads, lat_workloads = _resolve_workloads(
        [w.strip() for w in args.workloads.split(",") if w.strip()]
    )
    if args.skip_throughput:
        tp_workloads = []
    if args.skip_latency:
        lat_workloads = []
    if not tp_workloads and not lat_workloads:
        raise SystemExit(
            "ERROR: no workloads left to run. --workloads selected none, or "
            "both --skip-throughput and --skip-latency were passed."
        )

    # bitnet-b1.58-2B-4T has max_position_embeddings=4096, but longbench-longctx
    # prompts are >= 8185 tokens. Normalizing to the real mean would ask the model
    # to represent positions it was never trained for (and would exceed
    # --max-model-len), so every workload's fixed input length is clamped to what
    # the model can actually represent.
    model_ctx = None
    try:
        from transformers import AutoConfig

        cfg_obj = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
        model_ctx = int(getattr(cfg_obj, "max_position_embeddings", 0)) or None
    except Exception as exc:  # pragma: no cover - config always present in CI
        print(f"[warn] could not read max_position_embeddings: {exc}", flush=True)

    def _scenarios_for(descriptors: list[dict], kind: str) -> list[dict]:
        built = []
        for idx, wl in enumerate(descriptors):
            declared = wl.get("num_requests") or wl.get("batch_size") or 1
            if kind == "latency":
                # A latency probe times one fixed batch repeatedly, so it needs
                # exactly batch_size prompts regardless of --num-prompts.
                declared = n = wl["batch_size"]
            elif args.num_prompts is not None:
                # An explicit count sets the shape too, as it always has.
                declared = n = args.num_prompts
            else:
                # The reference's int2 decode kernels only dispatch for M == 1,
                # so both engines serve these one at a time (~2.6s per request at
                # mixed's shape). Every request is padded to the same fixed shape
                # with early stop disabled, so each is identical work and tok/s
                # is a rate over repetitions -- cap the timed count, keep the
                # declared shape.
                n = min(declared, max(1, args.max_timed_prompts))
            out_len = wl["output_len"]
            cap = None
            if model_ctx is not None:
                cap = max(model_ctx - out_len, 1)
            if args.prompt_source == "real":
                assert tokenizer is not None
                prompt_ids, dataset_id, length_stats, input_len, mean_len = (
                    _build_real_token_prompts(
                        tokenizer, wl["name"], declared, None, out_len,
                        args.seed, args.dataset_split,
                        dataset_name=wl.get("dataset_name"),
                        max_input_len=cap,
                        keep_prompts=n,
                    )
                )
                clamped = (
                    " (clamped to model ctx "
                    f"{model_ctx}-{out_len})"
                    if cap is not None and int(round(mean_len)) > cap
                    else ""
                )
                count_desc = (
                    f" n={n} of {declared} (timed-request cap)"
                    if n < declared else f" n={n}"
                )
                print(
                    f"[data] {wl['name']:>14}: {dataset_id} "
                    f"raw_prompt_len(min/p50/max/mean)="
                    f"{length_stats[0]}/{length_stats[1]}/{length_stats[2]}/"
                    f"{mean_len:.0f} "
                    f"normalized={input_len}{clamped} out={out_len}"
                    f"{count_desc}",
                    flush=True,
                )
            else:
                input_len = wl.get("input_len") or 512
                if cap is not None:
                    input_len = max(min(input_len, cap), 1)
                prompt_ids = _build_random_token_prompts(
                    n, input_len, args.vocab_size, args.seed + idx,
                )
                dataset_id = "deterministic-random-token-ids"
                length_stats = (input_len, input_len, input_len)
                mean_len = float(input_len)
            entry = {
                "name": wl["name"],
                "kind": kind,
                "prompt_token_ids": prompt_ids,
                "output_lens": [out_len] * n,
                "prompt_source": args.prompt_source,
                "dataset": dataset_id,
                "raw_prompt_len_min_p50_max": list(length_stats),
                "raw_prompt_len_mean": mean_len,
                "input_len": input_len,
                "model_max_position_embeddings": model_ctx,
                "num_requests_declared": declared,
                "num_requests_timed": n,
            }
            if kind == "latency":
                entry.update({
                    "batch_size": wl["batch_size"],
                    "num_warmup": wl["num_warmup"],
                    "num_iters": wl["num_iters"],
                })
            built.append(entry)
        return built

    throughput_scenarios = _scenarios_for(tp_workloads, "throughput")
    latency_scenarios_cfg = _scenarios_for(lat_workloads, "latency")
    scenarios = throughput_scenarios

    # --max-model-len must cover the longest (input + output) actually built. The
    # sweep never passes it, and the 2048 default cannot hold `long-context`
    # (3968 in + 128 out), so raise it to fit -- bounded by what the model can
    # represent, which is the same ceiling the input lengths were clamped to.
    needed = max(
        (sc["input_len"] + sc["output_lens"][0]
         for sc in (*throughput_scenarios, *latency_scenarios_cfg)
         if sc["output_lens"]),
        default=args.max_model_len,
    )
    if model_ctx is not None:
        needed = min(needed, model_ctx)
    if needed > args.max_model_len:
        print(
            f"[cfg] raising --max-model-len {args.max_model_len} -> {needed} "
            f"to fit the longest workload shape",
            flush=True,
        )
        args.max_model_len = needed

    sota_data = None
    if not args.skip_sota:
        int2_pt = os.path.join(ckpt_dir, "model_state_int2.pt")
        fp16_pt = os.path.join(ckpt_dir, "model_state_fp16.pt")
        missing = [p for p in (kernel_so, int2_pt, fp16_pt) if not os.path.isfile(p)]
        if missing:
            # Do not warn-and-continue: that produced a results.json with empty
            # `scenarios`/`latency_scenarios` and exit 0, i.e. a PASS with no
            # comparison at all. Skipping the reference has to be explicit.
            raise SystemExit(
                "ERROR: Microsoft BitNet GPU reference artifacts missing:\n"
                + "\n".join(f"  {p}" for p in missing)
                + "\n\nProvision them with:\n"
                "  python -m fastkernels.validate.provision bitnet\n"
                "or point --bitnet-repo / $BITNET_REPO at an existing "
                "checkout. Pass --skip-sota to intentionally run fastkernels "
                "alone (no speedup or alignment will be recorded)."
            )
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                sota_out = f.name
            sota_cfg = {
                "model": args.model, "seed": args.seed,
                "scenarios": scenarios, "output_file": sota_out,
                "bitnet_repo": args.bitnet_repo, "ckpt_dir": ckpt_dir,
                "gen_bsz": args.gen_bsz,
                "latency_scenarios": latency_scenarios_cfg,
                "alignment_prompts": args.alignment_prompts,
                "alignment_reference": args.alignment_reference,
            }
            sota_data = run_worker(
                SOTA_WORKER, sota_cfg,
                f"Microsoft BitNet GPU SOTA [{args.model}, "
                f"gen_bsz={args.gen_bsz}]")
            if sota_data:
                _print_throughput_table(
                    "Microsoft BitNet GPU (W1.58A8 official kernel)",
                    sota_data)

    kb_data = None
    if not args.skip_kb and (scenarios or latency_scenarios_cfg):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            kb_out = f.name
        kb_cfg = {
            "model": args.model, "seed": args.seed, "tp": args.tp,
            "scenarios": scenarios, "output_file": kb_out,
            "latency_scenarios": latency_scenarios_cfg,
            "max_model_len": args.max_model_len,
            "project_root": str(_PROJECT_ROOT),
            # Non-eager by default, matching bench_vllm.py and the reference,
            # which times its own CUDA-graph path. Timing our eager path against
            # a graph-replaying reference understated this row by ~13x (0.089x
            # vs 1.17-1.30x like-for-like on B200): eager decode costs ~29
            # ms/step against ~2 ms/step replayed.
            "enforce_eager": args.enforce_eager,
            "kb_bsz": args.kb_bsz,
            "bitnet_kernel_so": (
                kernel_so if os.path.isfile(kernel_so) else ""
            ),
        }
        kb_data = run_worker(
            KB_WORKER, kb_cfg, f"fastkernels [{args.model}] throughput",
        )
        if kb_data is None:
            # Do not fall through to "skip the comparison and exit 0": that is
            # how a failed engine got recorded as PASS with no results at all.
            raise SystemExit(
                f"ERROR: the fastkernels engine failed for {args.model}, so "
                f"there is nothing to benchmark or compare. See the traceback "
                f"above."
            )
        kb_kernel = (
            "official ladder decode"
            if os.path.isfile(kernel_so) else "Triton fallback"
        )
        mode = "eager" if args.enforce_eager else "cudagraph"
        _print_throughput_table(
            f"fastkernels (W1.58A8 {kb_kernel}, {mode})", kb_data,
        )

    alignments: dict[str, dict] = {}
    topk_alignments = None
    if sota_data and kb_data:
        for sr, kr in zip(sota_data["throughput"], kb_data["throughput"]):
            assert sr["name"] == kr["name"]
            sota_outs = sr.get("outputs") or []
            kb_outs = (kr.get("outputs") or [])[:len(sota_outs)]
            if not sota_outs or not kb_outs:
                continue
            alignments[sr["name"]] = compute_alignment(kb_outs, sota_outs)

        _print_summary_table(sota_data, kb_data, alignments)

        if not args.skip_topk_alignment:
            score_scenarios = []
            for sc, sr, kr in zip(
                scenarios, sota_data["throughput"], kb_data["throughput"],
            ):
                sota_outs = sr.get("outputs") or []
                kb_outs = (kr.get("outputs") or [])[:len(sota_outs)]
                if not sota_outs or not kb_outs:
                    continue
                align_count = min(len(sota_outs), len(kb_outs))
                score_scenarios.append({
                    "name": sc["name"],
                    "prompt_token_ids": sc["prompt_token_ids"][:align_count],
                    "output_len": sc["output_lens"][0],
                    "groups": {
                        "sota_self": sota_outs[:align_count],
                        "kb_under_sota": kb_outs[:align_count],
                    },
                })
            if score_scenarios:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False,
                ) as f:
                    topk_out = f.name
                score_cfg = {
                    "scenarios": score_scenarios,
                    "output_file": topk_out,
                    "bitnet_repo": args.bitnet_repo,
                    "ckpt_dir": ckpt_dir,
                    "topks": [1, 5, 20],
                }
                topk_data = run_worker(
                    SOTA_SCORE_WORKER, score_cfg,
                    "Microsoft BitNet direct-decode top-k alignment",
                )
                if topk_data:
                    topk_alignments = topk_data["topk_alignment"]
                    _print_topk_alignment_table(topk_alignments)

    if sota_data or kb_data:
        _persist_results(
            args.output_dir, args.model, args.num_prompts,
            sota_data, kb_data, alignments, topk_alignments, scenarios,
            enforce_eager=args.enforce_eager,
        )


if __name__ == "__main__":
    main()
