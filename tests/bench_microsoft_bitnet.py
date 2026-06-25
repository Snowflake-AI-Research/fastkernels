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

Both engines run the same 3-scenario LLM workload from
``adding-arch-instructions.md``:

  * prefill-heavy (1024 prefill / 512 decode)
  * balanced     ( 512 prefill / 512 decode)
  * decode-heavy ( 512 prefill / 1024 decode)

with 1000 requests per scenario.  fastkernels uses paged scheduling across
the full request set.  The official Microsoft decode GEMM only implements
``M == 1`` kernels, so the SOTA worker must use ``--gen-bsz 1`` and loop
requests one-by-one.

Both engines run greedy decoding (temperature 0, ignore_eos) on the
**same** real WildChat-derived natural-language prompts, tokenized with
the BitNet tokenizer and normalized to the fixed scenario lengths.  They
return per-request output token ids.  Throughput is measured against the
official CUDA-graph path, while alignment is computed against the same
official models run with fresh per-step attention metadata.  This avoids
comparing fastkernels against a known Microsoft FastGen CUDA-graph metadata
bug where generated tokens are not represented correctly in the replayed
attention bias.

Setup (one-time):
-----------------
The Microsoft GPU kernel + checkpoint conversion must be done first::

    cd /path/to/microsoft/BitNet/gpu
    bash bitnet_kernels/compile.sh
    huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --local-dir checkpoints/bitnet-b1.58-2B-4T-bf16
    python convert_safetensors.py \\
        --safetensors_file checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors \\
        --output checkpoints/model_state.pt --model_name 2B
    python convert_checkpoint.py --input checkpoints/model_state.pt
    rm checkpoints/model_state.pt   # only the int2/fp16 splits are needed

Usage:
------
::

    # full benchmark (fastkernels + Microsoft BitNet GPU on all 3 scenarios,
    # 1000 reqs each) -- requires BITNET_REPO env or --bitnet-repo flag.
    BITNET_REPO=/path/to/microsoft/BitNet \\
        python tests/bench_microsoft_bitnet.py

    # smoke run using fastkernels's continuous scheduler
    python tests/bench_microsoft_bitnet.py --num-prompts 32 --kb-bsz 0 --skip-sota

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
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

SCENARIOS = [
    {"name": "prefill-heavy", "input_len": 1024, "output_len": 512},
    {"name": "balanced",      "input_len": 512,  "output_len": 512},
    {"name": "decode-heavy",  "input_len": 512,  "output_len": 1024},
]


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
    input_len: int,
    output_len: int,
    seed: int,
    split: str,
) -> tuple[list[list[int]], str, tuple[int, int, int]]:
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

    from fastkernels.bench.utils.real_prompts import (
        DEFAULT_WORKLOAD_DATASETS,
        load_real_prompt_workload,
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 1

    samples = load_real_prompt_workload(
        scenario_name,
        tokenizer,
        num_requests=num_prompts,
        decode_cap=output_len,
        split=split,
        seed=seed,
    )
    raw_lens = sorted(len(sample.prompt_token_ids) for sample in samples)
    prompts = [
        _normalize_prompt_len(sample.prompt_token_ids, input_len, int(pad_id))
        for sample in samples
    ]
    length_stats = (raw_lens[0], raw_lens[len(raw_lens) // 2], raw_lens[-1])
    return prompts, DEFAULT_WORKLOAD_DATASETS[scenario_name], length_stats


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
for sc in cfg["scenarios"]:
    prompts = sc["prompt_token_ids"]
    out_lens = sc["output_lens"]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
        for ol in out_lens
    ]
    engine.block_manager.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sps, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
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

with open(cfg["output_file"], "w") as f:
    json.dump({
        "throughput": results,
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
# Upstream ``generate.py`` has a batched prefill indexing bug:
# ``output[kv_seqlen - 1, :]`` ignores the flattened per-request prompt
# offset. This worker uses a local fixed generation loop so the benchmark
# stays self-contained and does not require patching the Microsoft repo.
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
def generate_all_direct_fixed(g, prompts, use_sampling):
    """Correctness reference: official models with fresh decode metadata.

    ``FastGen.compile_generate`` captures an AttnBias at prompt length and
    only mutates ``k_seqinfo.seqlen`` during graph replay.  The resulting
    graph output diverges from the official model's direct autoregressive
    decode.  Use this direct path for alignment only; graph replay remains
    the timed SOTA throughput path.
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
        generate_all_fixed(
            g, batch, use_cuda_graphs=True, use_sampling=False,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        elapsed_total += dt
        n_in += real_count * in_len
        n_out += real_count * out_len

    align_count = min(int(cfg.get("alignment_prompts", len(prompts))), len(prompts))
    out_records = []
    if align_count > 0:
        print(f"[sota] generating {align_count} direct-reference output(s) "
              f"for alignment...", flush=True)
        for prompt in prompts[:align_count]:
            answers = generate_all_direct_fixed(
                g, [prompt], use_sampling=False,
            )
            out_records.append({"token_ids": list(answers[0])[:out_len]})

    results.append({
        "name": sc["name"], "elapsed": elapsed_total,
        "total_input_tokens": n_in, "total_output_tokens": n_out,
        "num_prompts": len(prompts), "gen_bsz": gen_bsz,
        "alignment_reference": "official_direct_decode",
        "timed_reference": "official_cuda_graph",
        "outputs": out_records,
    })
    print(f"[sota] {sc['name']:>14}: {elapsed_total:7.2f}s  "
          f"in={n_in:>8d}  out={n_out:>8d}  "
          f"throughput={(n_in + n_out) / elapsed_total:>8.1f} tok/s",
          flush=True)

    # Free per-scenario state before the next FastGen build.
    del g
    torch.cuda.empty_cache()

with open(cfg["output_file"], "w") as f:
    json.dump({
        "throughput": results,
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
        for prompt, record in zip(prompts, records):
            seq_counts, n = score_sequence_direct(
                g, prompt, record["token_ids"][:out_len],
            )
            total += n
            for k in topks:
                counts[str(k)] += seq_counts[str(k)]
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
                     scenarios: list[dict]) -> None:
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
            }
            for sc in scenarios
        ],
        "sota": sota_data,
        "fastkernels": kb_data,
        "alignment": alignments,
        "topk_alignment": topk_alignments,
    }
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
    ap.add_argument("--num-prompts", type=int, default=1000,
                    help="Requests per scenario (default 1000 per "
                         "adding-arch-instructions.md)")
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
                        "/home/yak/vllm_repo/BitNet"),
                    help="Path to the Microsoft BitNet repo "
                         "(must contain gpu/checkpoints/model_state_int2.pt and "
                         "gpu/bitnet_kernels/libbitnet.so)")
    ap.add_argument("--gen-bsz", type=int, default=1,
                    help="CUDA-graph batch size for the Microsoft BitNet "
                         "GPU worker. Must be 1: the official int2 decode "
                         "kernels only implement M == 1.")
    ap.add_argument("--alignment-prompts", type=int, default=32,
                    help="Number of prompts per scenario to score against "
                         "the official direct-decode reference. Throughput "
                         "still uses --num-prompts.")
    ap.add_argument("--kb-bsz", type=int, default=1,
                    help="Number of requests per fastkernels generate() call. "
                         "Default 1 matches the Microsoft BitNet GPU "
                         "baseline's M==1 decode limit; use 0 to benchmark "
                         "fastkernels's continuous scheduler over all prompts.")
    ap.add_argument("--use-kb-cudagraph", action="store_true",
                    help="Enable fastkernels CUDA graphs for debugging. The "
                         "default eager path is the alignment reference.")
    ap.add_argument("--skip-topk-alignment", action="store_true",
                    help="Skip teacher-forced top-k scoring under the "
                         "official direct-decode reference")
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
            _PACKAGE_DIR / "tests" / "results" / gpu / f"{short}_bitnet"
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

    scenarios = []
    for idx, sc in enumerate(SCENARIOS):
        if args.prompt_source == "real":
            assert tokenizer is not None
            prompt_ids, dataset_id, length_stats = _build_real_token_prompts(
                tokenizer,
                sc["name"],
                args.num_prompts,
                sc["input_len"],
                sc["output_len"],
                args.seed,
                args.dataset_split,
            )
            print(
                f"[data] {sc['name']:>14}: {dataset_id} "
                f"raw_prompt_len(min/p50/max)="
                f"{length_stats[0]}/{length_stats[1]}/{length_stats[2]} "
                f"normalized={sc['input_len']}",
                flush=True,
            )
        else:
            prompt_ids = _build_random_token_prompts(
                args.num_prompts,
                sc["input_len"],
                args.vocab_size,
                args.seed + idx,
            )
            dataset_id = "deterministic-random-token-ids"
        scenarios.append({
            "name": sc["name"],
            "prompt_token_ids": prompt_ids,
            "output_lens": [sc["output_len"]] * args.num_prompts,
            "prompt_source": args.prompt_source,
            "dataset": dataset_id,
        })

    sota_data = None
    if not args.skip_sota:
        int2_pt = os.path.join(ckpt_dir, "model_state_int2.pt")
        fp16_pt = os.path.join(ckpt_dir, "model_state_fp16.pt")
        missing = [p for p in (kernel_so, int2_pt, fp16_pt) if not os.path.isfile(p)]
        if missing:
            print(f"\n[skip-sota] Microsoft BitNet GPU artifacts missing: "
                  f"{missing}.  See module docstring for setup instructions.\n")
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                sota_out = f.name
            sota_cfg = {
                "model": args.model, "seed": args.seed,
                "scenarios": scenarios, "output_file": sota_out,
                "bitnet_repo": args.bitnet_repo, "ckpt_dir": ckpt_dir,
                "gen_bsz": args.gen_bsz,
                "alignment_prompts": args.alignment_prompts,
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
    if not args.skip_kb:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            kb_out = f.name
        kb_cfg = {
            "model": args.model, "seed": args.seed, "tp": args.tp,
            "scenarios": scenarios, "output_file": kb_out,
            "max_model_len": args.max_model_len,
            "project_root": str(_PROJECT_ROOT),
            "enforce_eager": not args.use_kb_cudagraph,
            "kb_bsz": args.kb_bsz,
            "bitnet_kernel_so": kernel_so if os.path.isfile(kernel_so) else "",
        }
        kb_data = run_worker(KB_WORKER, kb_cfg, f"fastkernels [{args.model}]")
        if kb_data:
            kb_kernel = (
                "official ladder decode"
                if os.path.isfile(kernel_so) else "Triton fallback"
            )
            _print_throughput_table(f"fastkernels (W1.58A8 {kb_kernel})",
                                    kb_data)

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
        )


if __name__ == "__main__":
    main()
