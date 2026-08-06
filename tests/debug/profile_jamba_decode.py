#!/usr/bin/env python3
"""Attribute a JambaEngine decode step: host prep vs GPU, and the kernel table.

Jamba's four bench rows sit at 0.93-0.97x of vLLM, so the question is
categorical before it is numerical: is the deficit kernel work, host work that
the engine fails to overlap, or GPU idle? This probe answers it directly for a
pure-decode step at a chosen batch size.

Three numbers per batch size:

  * ``prep``   -- wall time inside ``_run_decode_step`` BEFORE ``graph.replay()``
                  (the numpy metadata build + the six H2D copies). The engine
                  synchronizes on the sampled token every step, so this host
                  time does NOT overlap the GPU and lands on the critical path.
  * ``gpu``    -- CUDA-event time across the graph replay itself.
  * ``wall``   -- end-to-end per step, so ``wall - prep - gpu`` is slack.

With ``--profile`` it also prints the top kernels by self CUDA time and the
share held by ``copy``/``elementwise`` kernels -- the signature of a layer
materialising views the vendored kernels would have read strided.

Answered and not worth re-running: whether the decode graph is captured at
``__init__`` or re-recorded after a representative prefill makes no difference
(25.75 vs 25.86 ms at batch 256). The 24.2 ms that
``tune_jamba_moe_ingraph.py`` reports is an artifact of its own re-capture loop,
not of capture timing; only compare candidates within one of its runs.

Usage:
    python tests/debug/profile_jamba_decode.py --bs 1 --bs 32 --steps 30
    python tests/debug/profile_jamba_decode.py --bs 32 --profile
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from random import randint, seed as set_seed

import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from fastkernels.infra.jamba_engine import JambaEngine, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", "ai21labs/AI21-Jamba-Mini-1.7")


def _prefill_to_running(engine, prompts):
    """Drive the engine's own admit + chunked-prefill path, then hand back the
    running sequences so decode steps can be timed in isolation.

    Uses the engine's real ``Sequence`` / ``BlockManager`` / Mamba-slot
    bookkeeping rather than a hand-rolled stand-in, so the block tables and
    state slots a timed decode step reads are exactly what ``generate`` builds.
    """
    from fastkernels.infra.jamba_engine import Sequence

    engine.block_manager.reset()
    engine.mamba_pool.reset()

    seqs = [Sequence(list(p), max_tokens=4096, ignore_eos=True) for p in prompts]
    for seq in seqs:
        seq.state_slot = engine.mamba_pool.allocate()
        lifetime = len(seq.prompt_ids) + 256
        nblocks = (lifetime + engine._page_size - 1) // engine._page_size
        engine.block_manager.allocate_n(seq, nblocks)

    # Chunked prefill under the engine's own token budget.
    pending = list(seqs)
    while pending:
        chunks = []
        used = 0
        for seq in pending:
            remaining = len(seq.prompt_ids) - seq.num_computed_tokens
            if remaining <= 0:
                continue
            budget = engine.max_num_batched_tokens - used
            if budget <= 0:
                break
            take = min(remaining, budget)
            chunks.append((seq, take))
            used += take
        with torch.inference_mode():
            logits, done_mask = engine._run_prefill_chunks(chunks)
        for i, ((seq, take), is_done) in enumerate(zip(chunks, done_mask)):
            seq.num_computed_tokens += take
            if is_done:
                seq.append_token(int(logits[i].argmax().item()))
                seq.num_computed_tokens = len(seq)
        pending = [s for s in pending if s.num_computed_tokens < len(s.prompt_ids)]
    return seqs


def _time_decode(engine, running, steps):
    """Per-step prep / gpu / wall for ``steps`` decode iterations.

    Under ``inference_mode`` because ``JambaEngine.generate`` is -- the graphs'
    static buffers and the mixer's inference fast paths both key off it.
    """
    prep_ms, gpu_ms, wall_ms = [], [], []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # Time the host prep by re-running the same metadata build the engine does,
    # then let the engine do the real step. The duplicate build is what we
    # measure; the engine's own build is what actually feeds the graph, so the
    # timed step stays untouched.
    with torch.inference_mode():
        for _ in range(steps):
            torch.cuda.synchronize()
            t_wall = time.perf_counter()

            t0 = time.perf_counter()
            _rebuild_decode_metadata(engine, running)
            prep_ms.append((time.perf_counter() - t0) * 1000)

            start_ev.record()
            tokens = engine._run_decode_step(running)
            end_ev.record()
            torch.cuda.synchronize()
            gpu_ms.append(start_ev.elapsed_time(end_ev))
            wall_ms.append((time.perf_counter() - t_wall) * 1000)

            for seq, tok in zip(running, tokens):
                seq.append_token(tok)
                seq.num_computed_tokens = len(seq)

    return prep_ms, gpu_ms, wall_ms


def _rebuild_decode_metadata(engine, running):
    """Replica of ``_run_decode_step``'s host-side metadata build.

    This has to track the engine's build or ``prep`` becomes a number about code
    that no longer runs: it replicated the pre-optimization version for a while
    and kept reporting 0.58 ms after the engine's own prep had dropped to
    ~0.05 ms, which reads as "the optimization did nothing".

    Kept in the probe rather than factored out of the engine so the engine's hot
    loop is not restructured just to be measurable. What it mirrors is the
    steady-state ``reuse_invariants`` branch: the block tables and Mamba slots
    are per-sequence invariants, rebuilt only when the running set changes.
    """
    n = len(running)
    page_size = engine._page_size
    lens_np = np.fromiter((len(s) for s in running), dtype=np.int64, count=n)
    np.fromiter((s.last_token for s in running), dtype=np.int64, count=n)
    pos_np = lens_np - 1
    lens_np.astype(np.int32)
    bt_np = engine._decode_meta_bt
    if bt_np is None or bt_np.shape[0] != n:
        bps = engine._max_blocks_per_seq
        bt_np = np.full((n, bps), -1, dtype=np.int32)
        for i, s in enumerate(running):
            bt_np[i, : len(s.block_table)] = s.block_table
    return (
        bt_np[np.arange(n), pos_np // page_size].astype(np.int64) * page_size
        + pos_np % page_size
    )


def _profile(engine, running, steps):
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof, torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(steps):
            tokens = engine._run_decode_step(running)
            for seq, tok in zip(running, tokens):
                seq.append_token(tok)
                seq.num_computed_tokens = len(seq)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    # Leaf kernels only: a top-level ``aten::`` entry carries the device time of
    # the kernel it launched, so summing both double-counts and reports GPU busy
    # over 100%.
    events = [
        e for e in prof.key_averages()
        if e.self_device_time_total > 0
        and not e.key.startswith(("aten::", "_C::", "torch"))
    ]
    events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    total_gpu_us = sum(e.self_device_time_total for e in events)
    print(f"\n  wall {wall * 1000 / steps:.3f} ms/step, "
          f"kernel sum {total_gpu_us / 1000 / steps:.3f} ms/step, "
          f"GPU busy {total_gpu_us / 1e6 / wall * 100:.1f}%")
    print(f"  {'KERNEL':<62} {'CALLS/STEP':>10} {'ms/STEP':>9} {'%':>6}")
    for e in events[:22]:
        ms = e.self_device_time_total / 1000 / steps
        print(f"  {e.key[:62]:<62} {e.count / steps:>10.1f} {ms:>9.4f} "
              f"{e.self_device_time_total / total_gpu_us * 100:>5.1f}%")

    copyish = sum(
        e.self_device_time_total
        for e in events
        if any(t in e.key.lower() for t in ("copy", "elementwise", "transpose",
                                            "contiguous", "cat"))
    )
    print(f"\n  copy/elementwise/cat kernels: {copyish / 1000 / steps:.4f} ms/step "
          f"({copyish / total_gpu_us * 100:.1f}% of kernel time)")


def _profile_prefill(engine, widths, seq_len, steps, do_profile):
    """Per-call cost of a chunked-prefill forward at several batch widths.

    The ``mixed`` row spends half its wall time in prefill calls whose median
    width is 43 tokens, so what matters is the FIXED cost of a prefill pass --
    32 layers of eager dispatch plus a full sweep of the MoE weights -- against
    the marginal cost per token. Running several widths separates the two.
    """
    from random import randint

    from fastkernels.infra.jamba_engine import Sequence

    print(f"\n  {'TOKENS':>8} {'SEQS':>5} {'ms/CALL':>9} {'us/TOKEN':>9}")
    for width in widths:
        nseq = max(1, width // seq_len)
        per_seq = max(1, width // nseq)
        engine.block_manager.reset()
        engine.mamba_pool.reset()
        chunks = []
        for _ in range(min(nseq, engine.max_num_seqs)):
            seq = Sequence(
                [randint(5, 60000) for _ in range(per_seq)],
                max_tokens=1, ignore_eos=True,
            )
            seq.state_slot = engine.mamba_pool.allocate()
            nblocks = (per_seq + 1 + engine._page_size - 1) // engine._page_size
            engine.block_manager.allocate_n(seq, nblocks)
            chunks.append((seq, per_seq))

        with torch.inference_mode():
            for _ in range(3):
                engine._run_prefill_chunks(chunks)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                engine._run_prefill_chunks(chunks)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / steps * 1000
        tokens = len(chunks) * per_seq
        print(f"  {tokens:>8} {len(chunks):>5} {ms:>9.3f} {ms * 1000 / tokens:>9.2f}")

        if do_profile:
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof, torch.inference_mode():
                t0 = time.perf_counter()
                for _ in range(steps):
                    engine._run_prefill_chunks(chunks)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
            events = [
                e for e in prof.key_averages()
                if e.self_device_time_total > 0
                and not e.key.startswith(("aten::", "_C::", "torch"))
            ]
            events.sort(key=lambda e: e.self_device_time_total, reverse=True)
            total = sum(e.self_device_time_total for e in events)
            print(f"    kernel sum {total / 1000 / steps:.3f} ms/call, "
                  f"GPU busy {total / 1e6 / wall * 100:.1f}%, "
                  f"idle {wall * 1000 / steps - total / 1000 / steps:.3f} ms/call")
            for e in events[:10]:
                print(f"    {e.key[:58]:<58} {e.count / steps:>7.1f} "
                      f"{e.self_device_time_total / 1000 / steps:>8.4f} "
                      f"{e.self_device_time_total / total * 100:>5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, action="append", default=None)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument(
        "--prefill-width", type=int, action="append", default=None,
        help="Profile prefill calls at these total token widths instead of decode.",
    )
    ap.add_argument("--prefill-seq-len", type=int, default=206,
                    help="Per-sequence prompt length when splitting a width "
                         "across sequences (206 = the mixed workload's mean).")
    ap.add_argument(
        "--prefill-shape", action="append", default=None,
        help="'TOKENS:SEQLEN' pairs, to hold the token count fixed while "
             "varying how many sequences carry it. The Mamba-1 varlen scan's "
             "cost is O(num_seqs x total_tokens), so these are not equivalent.",
    )
    args = ap.parse_args()
    batch_sizes = args.bs or [1, 32]
    max_num_seqs = args.max_num_seqs or max(batch_sizes)

    if args.prefill_shape:
        engine = JambaEngine(model_name=MODEL, max_num_seqs=256)
        engine.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=4, ignore_eos=True))
        for spec in args.prefill_shape:
            width, seq_len = (int(v) for v in spec.split(":"))
            _profile_prefill(engine, [width], seq_len, args.steps, args.profile)
        del engine
        return

    if args.prefill_width:
        engine = JambaEngine(model_name=MODEL, max_num_seqs=max(max_num_seqs, 256))
        engine.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=4, ignore_eos=True))
        _profile_prefill(
            engine, args.prefill_width, args.prefill_seq_len, args.steps,
            args.profile,
        )
        del engine
        return

    engine = JambaEngine(model_name=MODEL, max_num_seqs=max_num_seqs)
    engine.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=4, ignore_eos=True))

    for bs in batch_sizes:
        set_seed(1234)
        prompts = [
            [randint(5, 60000) for _ in range(args.prompt_len)] for _ in range(bs)
        ]
        running = _prefill_to_running(engine, prompts)

        # Warm the bucket this batch size dispatches to.
        _time_decode(engine, running, 5)
        prep, gpu, wall = _time_decode(engine, running, args.steps)
        print(
            f"\n  bs={bs:<4d} wall {np.median(wall):7.3f} ms/step   "
            f"prep {np.median(prep):6.3f} ms   gpu {np.median(gpu):7.3f} ms   "
            f"slack {np.median(wall) - np.median(prep) - np.median(gpu):6.3f} ms"
        )
        if args.profile:
            _profile(engine, running, args.steps)

    del engine


if __name__ == "__main__":
    main()
