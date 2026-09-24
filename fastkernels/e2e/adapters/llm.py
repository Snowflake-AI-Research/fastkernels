"""Text LLMs on ``LlamaEngine`` (Llama, Qwen3-Next, Kimi-Linear, GPT-OSS, ...; any TP).

Timing: the scenario's throughput and latency workloads exactly as ``fastkernels eval``
runs them (real WildChat / LongBench prompts, greedy, ``ignore_eos``), on the production
path (torch.compile + CUDA graphs).

Correctness: free-running greedy decoding diverges between two *correct* runs (a fresh
compile alone changes greedy outputs), so correctness is teacher-forced. Every run greedily
decodes the first ``correctness_samples`` requests of the first throughput workload as one
batch; the baseline saves those tokens, and every other run (noise, candidate) re-decodes
the same prompts forced along the baseline tokens, recording per step whether its own
argmax agreed.
"""

from __future__ import annotations

import atexit
import gc
import os
import time
import types

from .base import Adapter, RunSpec, Timing


def _is_special(hf_name: str) -> bool:
    n = hf_name.lower()
    # EAGLE-3 / FLA / Jamba run on dedicated engines (see capture._is_*).
    return "eagle3" in n or n.startswith("fla-hub/") or "jamba" in n


def _prefix_frac(a: list[int], b: list[int]) -> float:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(1, len(a))


class LLMAdapter(Adapter):
    name = "llm"
    metric = "1 - teacher-forced top-1 agreement with the baseline tokens (per request)"

    @classmethod
    def handles(cls, scenario) -> bool:
        wls = list(scenario.throughput_workloads) + list(scenario.latency_workloads)
        return (not _is_special(scenario.hf_name) and bool(wls)
                and all(type(w).__name__ == "LLM" for w in wls))

    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import numpy as np
        import torch
        from transformers import AutoTokenizer

        from fastkernels import eval as E
        from fastkernels.infra.engine import LlamaEngine, SamplingParams, Sequence

        args = types.SimpleNamespace(max_requests=spec.max_requests or 10**9, seed=spec.seed,
                                     temperature=0.0, max_layers=None,
                                     enforce_eager=spec.enforce_eager)
        tokenizer = AutoTokenizer.from_pretrained(scenario.hf_name, trust_remote_code=True)
        tput, lat, max_seq_len, _ = E._load_scenario_runs(scenario, args, tokenizer)
        if spec.workloads:
            tput = [r for r in tput if r["name"] in spec.workloads]
            lat = [r for r in lat if r["name"] in spec.workloads]
        if not tput:
            raise RuntimeError("llm adapter needs at least one throughput workload for correctness")

        kwargs = dict(model_name=scenario.hf_name, dtype=E._engine_dtype(scenario.dtype),
                      seed=spec.seed, tensor_parallel_size=scenario.tp,
                      enforce_eager=spec.enforce_eager or scenario.enforce_eager,
                      max_num_seqs=scenario.max_num_seqs, max_layers=None)
        if getattr(scenario, "kv_cache_dtype", None):
            kwargs["kv_cache_dtype"] = scenario.kv_cache_dtype
        if max_seq_len:
            kwargs["max_model_len"] = max_seq_len
        engine = LlamaEngine(**kwargs)

        timings: dict[str, Timing] = {}
        n = min(spec.correctness_samples, len(tput[0]["prompt_token_ids"]))
        prompts = tput[0]["prompt_token_ids"][:n]
        try:
            engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))
            for r in tput:
                sp = [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=ol, ignore_eos=True)
                      for ol in r["output_lens"]]
                engine.block_manager.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outs = engine.generate(r["prompt_token_ids"], sp, use_tqdm=False, decode_text=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                ntok = sum(len(o.token_ids) for o in outs)
                timings[r["name"]] = {"kind": "throughput", "value": ntok / elapsed, "unit": "tok/s"}
            for r in lat:
                sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=r["output_len"])
                for _ in range(r["num_warmup"]):
                    engine.block_manager.reset()
                    engine.generate(r["prompt_token_ids"], sp, use_tqdm=False)
                ts = []
                for _ in range(r["num_iters"]):
                    engine.block_manager.reset()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    engine.generate(r["prompt_token_ids"], sp, use_tqdm=False)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                timings[r["name"]] = {"kind": "latency", "value": float(np.median(ts)), "unit": "s",
                                      "samples": ts}

            # Correctness subset, decoded as its own batch in every run, so the reference
            # tokens and the forced re-decode share one batch regime (batch size changes
            # kernel configs, and near-tie argmaxes would flip systematically otherwise).
            out_lens = tput[0]["output_lens"][:n]
            engine.block_manager.reset()
            outs = engine.generate(prompts, [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=ol,
                                                            ignore_eos=True) for ol in out_lens],
                                   use_tqdm=False, decode_text=False)
            free_tokens = [list(o.token_ids) for o in outs]
            outputs: dict = {"kind": "tokens", "workload": tput[0]["name"], "tokens": free_tokens}
            if spec.reference:
                from fastkernels.validate.forced_decode import run_forced_decode
                ref = torch.load(spec.reference, weights_only=False)
                ref_tokens = [list(t) for t in ref["tokens"][:n]]
                engine.block_manager.reset()
                t0 = time.perf_counter()
                agree, _, _, forced_ok = run_forced_decode(engine, Sequence, SamplingParams,
                                                           prompts, ref_tokens)
                outputs.update(kind="forced", forced_ok=bool(forced_ok),
                               agree=[agree.get(i, []) for i in range(len(ref_tokens))],
                               t_forced_s=time.perf_counter() - t0)
            torch.save(outputs, os.path.join(spec.out_dir, "outputs.pt"))
        finally:
            engine._cleanup()
            atexit.unregister(engine._cleanup)
            del engine
            gc.collect()
        return timings

    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        rt, ct = ref.get("tokens") or [], cand.get("tokens") or []
        free = {}
        if rt and ct:
            pairs = list(zip(rt, ct))
            free = {"free_exact_match": sum(a == b for a, b in pairs) / len(pairs),
                    "free_prefix_frac": sum(_prefix_frac(a, b) for a, b in pairs) / len(pairs)}
        if cand.get("kind") != "forced":
            return {"per_sample": [], "summary": {"note": "candidate has no forced outputs", **free}}
        agree = cand["agree"]
        per = [1.0 - sum(a) / len(a) if a else 1.0 for a in agree]
        first = [bool(a[0]) for a in agree if a]
        return {"per_sample": per, "summary": {
            "n": len(per),
            "top1_agreement": 1.0 - sum(per) / max(1, len(per)),
            "min_agreement": 1.0 - max(per, default=1.0),
            "first_step_agreement": sum(first) / max(1, len(first)),
            "forced_ok": cand.get("forced_ok"), **free}}
