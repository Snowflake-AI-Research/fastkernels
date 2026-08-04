"""Forced-decode per-step top-1 agreement — the long-context correctness gate.

Greedy full-sequence token-match vs vLLM is an INVALID gate for DSA / sparse
models at ``seq > index_topk``: the sparse-attention path is order-sensitive and
both engines are batch-nondeterministic there, so a single near-tie flip
CASCADES (every later position is then decoded from a divergent prefix) and
collapses the metric to near-zero — measuring the reference's own reduction-order
noise, not fk's correctness.

This gate removes the cascade. It PINS fk to vLLM's generated token sequence and,
at each decode step, records whether fk's OWN argmax equals vLLM's token, then
FORCES vLLM's token so fk continues on vLLM's exact trajectory. Each step is thus
scored on the identical (reference) prefix, so one flip no longer poisons the
rest — the number is fk's per-step decode distribution agreement with vLLM. It
exercises the real decode path (growing KV cache, DSA top-k every step), unlike
prefill teacher-forcing which only tests the prefill kernels.

Interpreting the result: on GLM-5.2-FP8 (DSA) at 8k–60k context this lands ~88%
— which is at the batch-nondeterminism CEILING, not a defect. fk's own
solo(batch=1)-vs-batched self-consistency is ~83%, so NO reference-match can
exceed ~83% across batch compositions; 88% vs vLLM (same batch composition) is at
that ceiling. Pass ``--margins`` to confirm the disagreements are near-tie flips:
vLLM's token is typically fk's 2nd choice, separated by a few bf16 ULPs in the
final logit. A genuine op bug looks different — large logit margins and vLLM's
token ranked far down — and would not be dismissed here.

Usage
-----
1. Generate the reference (vLLM, deterministic top-k) once::

     python -m fastkernels.validate.forced_decode --gen-ref \
         --model zai-org/GLM-5.2-FP8 --max-layers 18 \
         --lens 8000,10000,12000,15000,18000,22000 --out ref.json

   (or pass ``--prompts-file prompts.json`` with an explicit ``[[token_id,...],
   ...]`` list instead of synthetic ``--lens``.)

2. Run the fk gate against that reference::

     python -m fastkernels.validate.forced_decode \
         --model zai-org/GLM-5.2-FP8 --max-layers 18 --ref ref.json --margins

Both steps set ``FASTKERNELS_DSA_DETERMINISTIC_TOPK=1`` so the top-k SET+ORDER is
bit-reproducible on both engines (isolating the reduction-order effect this gate
is meant to measure). Run with ``VLLM_ENABLE_V1_MULTIPROCESSING=0``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Reusable core (no vLLM / no torch import at module load — safe to import)
# ---------------------------------------------------------------------------
@contextmanager
def force_along_reference(Sequence, ref_by_idx, agree, fkpred):
    """Monkeypatch ``Sequence.append_token`` to record fk's argmax vs the
    reference token and then substitute the reference token.

    Keys on ``seq._req_idx`` (assigned unconditionally in ``LlamaEngine.
    generate``); ``pos`` is taken BEFORE the append so it indexes the step
    whose token is about to be produced. Restores the original on exit.
    """
    orig = Sequence.append_token

    def _forced(self, tid):
        ri = getattr(self, "_req_idx", None)
        if ri is not None and ri in ref_by_idx:
            r = ref_by_idx[ri]
            pos = len(self.generated_ids)
            if pos < len(r):
                agree.setdefault(ri, []).append(bool(tid == r[pos]))
                fkpred.setdefault(ri, []).append(int(tid))
                tid = r[pos]                       # force onto the reference
        return orig(self, tid)

    Sequence.append_token = _forced
    try:
        yield
    finally:
        Sequence.append_token = orig


def run_forced_decode(engine, Sequence, SamplingParams, prompts, ref_tokens,
                      collect_margins=False):
    """Run ``engine`` forced along ``ref_tokens`` (one list per prompt).

    Requires ``FASTKERNELS_FORCE_SYNC_DECODE=1`` so every decode step rebuilds
    its input from live seq state (the async fast path reuses the prior step's
    sampled ``token_ids`` for the next input and would ignore the substitution).
    ``collect_margins`` captures per-step logits (routes through the logit path,
    which also forces the fresh-rebuild path).

    Returns ``(agree, fkpred, logits_hist_or_None, forced_ok)``.
    """
    os.environ["FASTKERNELS_FORCE_SYNC_DECODE"] = "1"
    ref_by_idx = {i: list(ref_tokens[i]) for i in range(len(ref_tokens))}
    agree: dict[int, list[bool]] = {}
    fkpred: dict[int, list[int]] = {}
    sp = [SamplingParams(temperature=0.0, max_tokens=len(ref_tokens[i]),
                         ignore_eos=True) for i in range(len(prompts))]
    with force_along_reference(Sequence, ref_by_idx, agree, fkpred):
        out = engine.generate(prompts, sp, use_tqdm=False, decode_text=False,
                              collect_logits=collect_margins)
    forced_ok = all(list(out[i].token_ids) == list(ref_tokens[i])
                    for i in range(len(ref_tokens)))
    logits_hist = ([out[i].logits_history for i in range(len(prompts))]
                   if collect_margins else None)
    return agree, fkpred, logits_hist, forced_ok


def summarize(agree, fkpred, ref_tokens, logits_hist=None, threshold=0.80,
              lens=None):
    """Print the per-seq + overall agreement report (and margins if logits are
    supplied). Returns ``(summary_dict, passed_bool)``."""
    tot_a = tot_n = 0
    per_seq = []
    margins: list[float] = []
    ranks: list[int] = []
    buckets = {"<=0.01": 0, "<=0.1": 0, "<=0.5": 0, "<=1.0": 0, ">1.0": 0}
    print("[FORCED-DECODE: fk argmax vs reference token, scored on the "
          "reference's true prefix]")
    for i in sorted(agree):
        a = agree[i]
        n = len(a)
        m = sum(a)
        tot_a += m
        tot_n += n
        fd = next((k for k, x in enumerate(a) if not x), None)
        line = f"  seq{i:3d}"
        if lens is not None and i < len(lens):
            line += f" len={lens[i]:6d}"
        line += f": agree={m:3d}/{n:<3d} ({100*m/max(n,1):5.1f}%)  first_disagree@{fd}"
        if logits_hist is not None:
            dis = []
            lh = logits_hist[i]
            for k in range(n):
                if a[k]:
                    continue
                lg = lh[k].float().reshape(-1)
                fk_tok = fkpred[i][k]
                v_tok = ref_tokens[i][k]
                margin = float(lg[fk_tok] - lg[v_tok])
                rank = int((lg > lg[v_tok]).sum())
                margins.append(margin)
                ranks.append(rank)
                b = ("<=0.01" if margin <= 0.01 else "<=0.1" if margin <= 0.1
                     else "<=0.5" if margin <= 0.5 else "<=1.0" if margin <= 1.0
                     else ">1.0")
                buckets[b] += 1
                dis.append((k, round(margin, 4), rank))
            line += f"  disagree(step,Δlogit,rank)={dis}"
        per_seq.append((i, m, n))
        print(line)

    overall = tot_a / max(tot_n, 1)
    passed = overall >= threshold
    print(f"\nOVERALL per-step top-1 agreement = {tot_a}/{tot_n} = "
          f"{100*overall:.2f}%   (threshold {100*threshold:.0f}%)  "
          f"-> {'PASS' if passed else 'FAIL'}")
    summary = {"overall_agreement": overall, "matching": tot_a, "steps": tot_n,
               "threshold": threshold, "passed": passed, "per_seq": per_seq}
    if logits_hist is not None and margins:
        summary["margins"] = {
            "n": len(margins), "min": min(margins),
            "median": statistics.median(margins), "max": max(margins),
            "rank_median": int(statistics.median(ranks)), "rank_max": max(ranks),
            "buckets": buckets,
        }
        print(f"disagreements={len(margins)}  logit-margin: "
              f"min={min(margins):.4f} median={statistics.median(margins):.4f} "
              f"max={max(margins):.4f}")
        print(f"rank of reference token in fk (0 = fk's 2nd choice): "
              f"median={int(statistics.median(ranks))} max={max(ranks)}")
        print(f"margin buckets: {buckets}")
        near_tie = sum(v for k, v in buckets.items() if k != ">1.0")
        print(f"near-tie flips (<=1.0 logit) = {near_tie}/{len(margins)} "
              f"({100*near_tie/len(margins):.1f}%)  -> disagreements are "
              f"close-calls, not gross errors" if near_tie >= 0.9 * len(margins)
              else f"near-tie flips (<=1.0 logit) = {near_tie}/{len(margins)}")
    return summary, passed


# ---------------------------------------------------------------------------
# Reference generation (vLLM, deterministic top-k) — imports vLLM, run alone
# ---------------------------------------------------------------------------
def _install_vllm_deterministic_topk():
    """Override vLLM's nondeterministic DSA top-k with flashinfer.top_k
    (deterministic SET+ORDER, window-relative indices). Mirrors the
    bench_vllm sitecustomize path; applied in-process for offline TP=1."""
    import torch
    import flashinfer
    from flashinfer import TopKTieBreak
    import vllm._custom_ops  # noqa: F401 — registers torch.ops._C

    def _fi(logits, ks, ke, k):
        logits = logits.contiguous()
        R, N = logits.shape
        cols = torch.arange(N, device=logits.device)
        valid = ((cols.unsqueeze(0) >= ks.reshape(-1).unsqueeze(1).long())
                 & (cols.unsqueeze(0) < ke.reshape(-1).unsqueeze(1).long()))
        s = torch.finfo(logits.dtype).min
        lm = torch.where(valid, logits, torch.full_like(logits, s))
        vv, idx = flashinfer.top_k(lm, k, sorted=False, deterministic=True,
                                   tie_break=int(TopKTieBreak.SMALL))
        idx = idx.to(torch.int32)
        idx = torch.where(vv <= s, torch.full_like(idx, -1), idx)
        ksr = ks.reshape(-1, 1).to(idx.dtype)
        idx = torch.where(idx >= 0, idx - ksr, idx)          # window-relative
        IM = 2147483647
        t = torch.where(idx >= 0, idx, torch.full_like(idx, IM))
        t, _ = torch.sort(t, -1)
        return torch.where(t == IM, torch.full_like(t, -1), t)

    def _co(l, ln, o, w, k, m):
        o.copy_(_fi(l, torch.zeros(l.shape[0], dtype=torch.int32,
                                   device=l.device), ln.to(torch.int32), k))

    def _pf(l, rs, re, idx, nr, s0, s1, k):
        idx.copy_(_fi(l, rs.to(torch.int32), re.to(torch.int32), k))

    def _de(l, nn, sl, idx, nr, s0, s1, k):
        sl = sl.to(torch.int32)
        if nn == 1:
            rl = sl.reshape(-1)
        else:
            j = torch.arange(nn, device=sl.device, dtype=torch.int32).view(1, nn)
            rl = (sl.reshape(sl.numel() // nn, 1) - nn + 1 + j).clamp_min_(0).reshape(-1)
        idx.copy_(_fi(l, torch.zeros(nr, dtype=torch.int32, device=l.device), rl, k))

    lib = torch.library.Library("_C", "FRAGMENT")
    for name, fn in [("cooperative_topk", _co), ("persistent_topk", _co),
                     ("top_k_per_row_prefill", _pf), ("top_k_per_row_decode", _de)]:
        lib.impl(name, fn, "CUDA")


def _gen_ref(args, prompts):
    os.environ["FASTKERNELS_DSA_DETERMINISTIC_TOPK"] = "1"
    _install_vllm_deterministic_topk()
    from vllm import LLM, SamplingParams
    hf_overrides = {"num_hidden_layers": args.max_layers} if args.max_layers else {}
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enforce_eager=True,
              trust_remote_code=True, hf_overrides=hf_overrides,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
              enable_prefix_caching=False, max_num_batched_tokens=args.max_num_batched_tokens)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    out = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    ref_tokens = [list(o.outputs[0].token_ids) for o in out]
    ref = {"model": args.model, "max_layers": args.max_layers,
           "prompts": prompts, "ref_tokens": ref_tokens}
    with open(args.out, "w") as f:
        json.dump(ref, f)
    print(f"wrote reference: {args.out}  ({len(prompts)} seqs, "
          f"{args.max_tokens} tokens each)")


# ---------------------------------------------------------------------------
def _synthetic_prompts(lens):
    return [list(range(1000 * i + 10, 1000 * i + 10 + L)) for i, L in enumerate(lens)]


def _load_ref(path):
    d = json.load(open(path))
    if "prompts" in d and "ref_tokens" in d:            # explicit
        return d["prompts"], d["ref_tokens"], [len(p) for p in d["prompts"]]
    if "lens" in d and "toks" in d:                     # synthetic reproducer
        return _synthetic_prompts(d["lens"]), d["toks"], d["lens"]
    raise ValueError(f"{path}: expected keys {{prompts,ref_tokens}} or {{lens,toks}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=131280)
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="min per-step agreement to PASS (default 0.80; ~0.83 "
                         "is the batch-nondeterminism ceiling for GLM-5.2 DSA)")
    ap.add_argument("--margins", action="store_true",
                    help="capture per-step logits and report disagreement margins")
    # gate mode
    ap.add_argument("--ref", help="reference json to run the fk gate against")
    # gen-ref mode
    ap.add_argument("--gen-ref", action="store_true",
                    help="generate the reference with vLLM (deterministic top-k)")
    ap.add_argument("--lens", help="comma-separated synthetic prompt lengths (gen-ref)")
    ap.add_argument("--prompts-file", help="json [[token_id,...],...] (gen-ref)")
    ap.add_argument("--max-tokens", type=int, default=64, help="decode length (gen-ref)")
    ap.add_argument("--out", default="ref.json", help="reference output path (gen-ref)")
    args = ap.parse_args()

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    os.environ["FASTKERNELS_DSA_DETERMINISTIC_TOPK"] = "1"
    if args.max_layers:
        os.environ["FASTKERNELS_MAX_LAYERS"] = str(args.max_layers)

    if args.gen_ref:
        if args.prompts_file:
            prompts = json.load(open(args.prompts_file))
        elif args.lens:
            prompts = _synthetic_prompts([int(x) for x in args.lens.split(",")])
        else:
            ap.error("--gen-ref needs --lens or --prompts-file")
        _gen_ref(args, prompts)
        return

    if not args.ref:
        ap.error("gate mode needs --ref (or use --gen-ref)")
    prompts, ref_tokens, lens = _load_ref(args.ref)

    mod = __import__("fastkernels.infra.engine",
                     fromlist=["LlamaEngine", "SamplingParams", "Sequence"])
    engine = mod.LlamaEngine(model_name=args.model, enforce_eager=True,
                             tensor_parallel_size=args.tp, max_layers=args.max_layers,
                             max_model_len=args.max_model_len,
                             max_num_batched_tokens=args.max_num_batched_tokens)
    agree, fkpred, logits_hist, forced_ok = run_forced_decode(
        engine, mod.Sequence, mod.SamplingParams, prompts, ref_tokens,
        collect_margins=args.margins)
    if not forced_ok:
        print("WARNING: forced trajectory != reference — some tokens were "
              "appended off the monkeypatched path; agreement may be misaligned.",
              file=sys.stderr)
    _, passed = summarize(agree, fkpred, ref_tokens, logits_hist=logits_hist,
                          threshold=args.threshold, lens=lens)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
