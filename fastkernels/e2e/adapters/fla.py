"""``fastkernels e2e`` adapter for FLA recurrent LLMs (``fla-hub/*``: GLA / RetNet / RWKV-7).

Builds the model with ``fastkernels.infra.fla_engine.FLAEngine`` IN PROCESS (so candidate
classes patched in by the runner are the ones instantiated) and mirrors the fastkernels
side of ``fastkernels/validate/bench_fla.py`` (``FASTKERNELS_FLA_WORKER``) -- same engine
config, workloads, prompts, seeds and timing methodology, without the FLA reference:

* throughput workloads (``mixed``, ``long-context``): real prompts from
  ``load_real_prompt_workload`` (seed ``seed + i``), prompts left-truncated to the model's
  ``max_position_embeddings`` exactly like bench_fla; one prefill+1-decode warmup at the
  workload's shapes, then ONE timed ``generate`` (greedy, ``ignore_eos``, per-request
  ``max_tokens``) -> output tok/s;
* latency workloads (``single-request``, ``fixed-batch-32``): real WildChat prompts
  (seed ``seed + 100 + j``), per-request decode budgets capped at ``output_len``; 2 warmup
  + ``num_iters`` timed ``generate`` calls -> median seconds.

Correctness (autoregressive => teacher forcing), same format as the LlamaEngine adapter
(``adapters/llm.py``). After the timing workloads, EVERY run greedily decodes the first
``correctness_samples`` requests of the first throughput workload (same decode budgets,
``ignore_eos``) as one batch of their own -- the reference and the forced re-decode then
share one batch regime, so batch-size-dependent near-tie flips do not pollute the signal:

* baseline (``spec.reference is None``): ``{"kind": "tokens", "tokens": [...]}`` (plus
  the prompts);
* with ``spec.reference``: additionally re-decode the same prompts as one batch with every
  sampled token replaced by the reference token (``FLAEngine._sample_batch`` is wrapped on
  the engine instance for the duration), recording per step whether the model's own argmax
  equals the reference token -> ``{"kind": "forced", "agree": [[bool]], "tokens": [...]}``.

``compare``: per-sample ``d = 1 - mean(agree)``; summary = mean / step-weighted / min
top-1 agreement, first-step agreement, disagreement logit margins, and (secondary)
free-running exact-match and matched-prefix fraction vs the reference tokens.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from .base import Adapter, RunSpec, Timing

# bench_fla.py defaults (argparse): max_num_seqs 256 for RetNet else 512,
# --chunked-prefill-size 1024, --max-prefill-tokens 196608, latency: 2 warmup + 5 iters.
_CHUNKED_PREFILL_SIZE = 1024
_MAX_PREFILL_TOKENS = 196608
_LATENCY_WARMUP = 2
_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def _norm_wl(name: str) -> str:
    name = name.strip().lower()
    if name.startswith("llm."):
        name = name[4:]
    return name.replace("_", "-")


def _fit_prompt_to_context(prompt: list[int], output_len: int,
                           max_model_len: int | None) -> tuple[list[int], int]:
    """bench_fla._fit_prompt_to_context: keep the prompt TAIL that fits with the decode."""
    if max_model_len is None:
        return prompt, output_len
    output_len = min(output_len, max_model_len - 1)
    budget = max_model_len - output_len
    if len(prompt) > budget:
        prompt = prompt[-budget:]
    return prompt, output_len


def _max_model_len(model_path: str) -> int | None:
    import json
    import os
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            v = json.load(f).get("max_position_embeddings")
        return int(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def _host_info() -> dict:
    """Where this run executed. FLAEngine is eager, so small-batch decode is CPU
    launch-bound and its timings depend on the host CPU as well as the GPU."""
    import os
    import socket

    import torch
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), None)
    except OSError:
        pass
    return {"hostname": socket.gethostname(), "cpu": cpu, "n_cpu": os.cpu_count(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def _prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@contextmanager
def _force_along_reference(engine, ref_by_idx: dict[int, list[int]], agree: dict,
                           margins: dict):
    """Teacher forcing for ``FLAEngine``: every token the engine samples -- the first token
    after prefill and each decode step -- goes through ``engine._sample_batch(logits,
    seqs)`` and is then appended to ``seq.generated_ids``, which feeds the next decode
    step. Wrapping it on the instance lets us record the engine's own choice vs the
    reference token at step ``len(seq.generated_ids)`` and return the reference token
    instead, so every step is scored on the reference prefix."""
    import torch

    orig = engine._sample_batch

    def _forced(logits, seqs):
        toks = list(orig(logits, seqs))
        for row, seq in enumerate(seqs):
            ref = ref_by_idx.get(seq.seq_id)
            if ref is None:
                continue
            pos = len(seq.generated_ids)
            if pos >= len(ref):
                continue
            ok = toks[row] == ref[pos]
            agree.setdefault(seq.seq_id, []).append(bool(ok))
            if not ok:  # logit margin of our pick over the reference token, and its rank
                lg = logits[row].float()
                if bool(torch.isfinite(lg).all()):
                    m = round(float(lg[toks[row]] - lg[ref[pos]]), 4)
                    rank = int((lg > lg[ref[pos]]).sum())
                else:  # NaN / inf logits (a broken kernel): no meaningful margin
                    m = rank = None
                margins.setdefault(seq.seq_id, []).append((pos, m, rank))
            toks[row] = ref[pos]
        return toks

    engine._sample_batch = _forced
    try:
        yield
    finally:
        del engine._sample_batch  # drop the instance attribute -> class method again


class FLAAdapter(Adapter):
    name = "fla"
    metric = ("1 - per-step top-1 agreement under teacher forcing on the baseline's greedy "
              "tokens (FLAEngine)")

    @classmethod
    def handles(cls, scenario) -> bool:
        return scenario.hf_name.startswith("fla-hub/")

    # ------------------------------------------------------------------ workloads
    @staticmethod
    def _selected(workloads, spec: RunSpec):
        if not spec.workloads:
            return list(workloads)
        want = {_norm_wl(w) for w in spec.workloads}
        return [w for w in workloads if _norm_wl(w.value) in want or _norm_wl(w.name) in want]

    def _build_runs(self, scenario, spec: RunSpec, tokenizer, max_model_len):
        from fastkernels.workloads import load_real_prompt_workload, spec_for

        cap = spec.max_requests
        thr_sel = set(self._selected(scenario.throughput_workloads, spec))
        lat_sel = set(self._selected(scenario.latency_workloads, spec))
        throughput, latency = [], []
        # Seeds index the scenario's FULL workload lists (as bench_fla indexes
        # THROUGHPUT_WORKLOADS / LATENCY_WORKLOADS), so subsetting never changes prompts.
        for i, wl in enumerate(scenario.throughput_workloads):
            if wl not in thr_sel:
                continue
            p = spec_for(wl).params
            n = p.num_requests if cap is None else min(cap, p.num_requests)
            samples = load_real_prompt_workload(
                wl.value, tokenizer, num_requests=n, decode_cap=p.decode_cap,
                dataset_name=p.dataset_name or None, seed=spec.seed + i)
            fitted = [_fit_prompt_to_context(list(s.prompt_token_ids), s.output_len,
                                             max_model_len) for s in samples]
            throughput.append({"name": wl.value, "prompts": [f[0] for f in fitted],
                               "output_lens": [f[1] for f in fitted]})
        for j, wl in enumerate(scenario.latency_workloads):
            if wl not in lat_sel:
                continue
            p = spec_for(wl).params
            bs = p.batch_size if cap is None else min(cap, p.batch_size)
            samples = load_real_prompt_workload(
                "mixed", tokenizer, num_requests=bs, decode_cap=p.output_len,
                dataset_name=p.dataset_name or None, seed=spec.seed + 100 + j)
            fitted = [_fit_prompt_to_context(list(s.prompt_token_ids), s.output_len,
                                             max_model_len) for s in samples]
            latency.append({"name": wl.value, "prompts": [f[0] for f in fitted],
                            "output_lens": [f[1] for f in fitted],
                            "num_warmup": _LATENCY_WARMUP, "num_iters": p.num_iters})
        return throughput, latency

    # ------------------------------------------------------------------ run
    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import torch
        from fastkernels.infra.fla_engine import FLAEngine, SamplingParams

        t_build = time.time()
        engine = FLAEngine(
            model_name=scenario.hf_name,
            dtype=getattr(torch, _DTYPES.get(scenario.dtype, "bfloat16")),
            seed=spec.seed,
            max_num_seqs=scenario.max_num_seqs
            or (256 if "retnet" in scenario.hf_name.lower() else 512),
            chunked_prefill_size=_CHUNKED_PREFILL_SIZE,
            max_prefill_tokens=_MAX_PREFILL_TOKENS,
        )
        t_build = time.time() - t_build
        # FLAEngine runs eagerly (no torch.compile / CUDA graphs): spec.enforce_eager is a no-op.
        max_model_len = _max_model_len(engine.model_path)
        throughput_runs, latency_runs = self._build_runs(scenario, spec, engine.tokenizer,
                                                         max_model_len)
        print(f"[fla] {scenario.hf_name}: engine built in {t_build:.1f}s, max_model_len="
              f"{max_model_len}, workloads="
              f"{[r['name'] for r in throughput_runs + latency_runs]}", flush=True)

        def _sp(n):
            return SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n, ignore_eos=True)

        engine.generate([[0] * 16], _sp(16))  # engine warmup (bench_fla)
        timings: dict[str, Timing] = {}
        meta: dict[str, Any] = {"t_build_s": round(t_build, 1), "max_model_len": max_model_len,
                                "host": _host_info(), "workloads": {}}
        print(f"[fla] host: {meta['host']}", flush=True)
        source = None  # (workload name, prompts, output lens) of the correctness subset

        for run in throughput_runs:
            prompts, out_lens = run["prompts"], run["output_lens"]
            # Prefill + one decode step at this workload's real shapes (Triton autotune), then
            # full untimed pass(es) so no lazily compiled kernel lands in the timed region.
            engine.generate(prompts, _sp(2))
            for _ in range(int(spec.extra.get("warmup_passes", 1))):
                engine.generate(prompts, [_sp(ol) for ol in out_lens])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = engine.generate(prompts, [_sp(ol) for ol in out_lens])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_out = sum(len(o.token_ids) for o in outs)
            timings[run["name"]] = {"kind": "throughput", "value": n_out / elapsed,
                                    "unit": "tok/s"}
            meta["workloads"][run["name"]] = {
                "requests": len(prompts), "elapsed_s": elapsed, "output_tokens": n_out,
                "prompt_tokens": sum(len(p) for p in prompts)}
            print(f"[fla] throughput {run['name']}: {len(prompts)} req, {n_out} tok, "
                  f"{elapsed:.2f}s -> {n_out / elapsed:,.1f} tok/s", flush=True)
            if source is None:
                source = (run["name"], prompts, out_lens)

        for run in latency_runs:
            prompts = run["prompts"]
            sp = [_sp(ol) for ol in run["output_lens"]]
            for _ in range(run["num_warmup"]):
                engine.generate(prompts, sp)
                torch.cuda.synchronize()
            lats = []
            for _ in range(run["num_iters"]):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                engine.generate(prompts, sp)
                torch.cuda.synchronize()
                lats.append(time.perf_counter() - t0)
            med = sorted(lats)[len(lats) // 2]
            timings[run["name"]] = {"kind": "latency", "value": med, "unit": "s",
                                    "samples": lats}
            meta["workloads"][run["name"]] = {
                "batch_size": len(prompts), "latencies_s": lats,
                "output_tokens": sum(run["output_lens"]),
                "input_len": max((len(p) for p in prompts), default=0)}
            print(f"[fla] latency {run['name']}: bs={len(prompts)} median={med:.4f}s "
                  f"({[round(x, 4) for x in lats]})", flush=True)
            if source is None:  # no throughput workload selected: use the first latency one
                source = (run["name"], prompts, run["output_lens"])

        # ---- correctness: the subset is decoded as ONE batch of its own in every run, so
        # the reference tokens and the forced re-decode share one batch regime (batch size
        # changes kernel configs; near-tie argmaxes would otherwise flip systematically).
        k = max(0, spec.correctness_samples)
        wl_name, prompts, out_lens = source if source is not None else (None, [], [])
        prompts, out_lens = prompts[:k], out_lens[:k]
        free = ([list(o.token_ids) for o in engine.generate(prompts, [_sp(ol) for ol in out_lens])]
                if prompts else [])
        out: dict[str, Any] = {"kind": "tokens", "workload": wl_name, "tokens": free,
                               "prompts": prompts, "prompts_idx": list(range(len(prompts))),
                               "meta": meta}
        if spec.reference is not None:
            ref = torch.load(spec.reference, weights_only=False)
            ref_tokens = [list(t) for t in ref.get("tokens", [])][:len(prompts)]
            n = len(ref_tokens)
            ref_prompts = ref.get("prompts")
            agree: dict[int, list[bool]] = {}
            margins: dict[int, list] = {}
            t0 = time.time()
            forced_ok = True
            if n:
                with _force_along_reference(engine, dict(enumerate(ref_tokens)), agree,
                                            margins):
                    forced = engine.generate(prompts[:n], [_sp(len(t)) for t in ref_tokens])
                forced_ok = all(list(forced[i].token_ids) == ref_tokens[i] for i in range(n))
            meta["t_forced_s"] = round(time.time() - t0, 1)
            agree_l = [agree.get(i, []) for i in range(n)]
            print(f"[fla] forced decode: {n} samples, "
                  f"{sum(map(sum, agree_l))}/{sum(map(len, agree_l))} steps agree, "
                  f"forced_ok={forced_ok}", flush=True)
            out.update(kind="forced", agree=agree_l, forced_ok=forced_ok,
                       margins=[margins.get(i, []) for i in range(n)],
                       prompts_match=None if ref_prompts is None
                       else [list(p) for p in ref_prompts[:n]] == [list(p) for p in prompts[:n]])
        torch.save(out, f"{spec.out_dir}/outputs.pt")
        return timings

    # ------------------------------------------------------------------ compare
    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        ref_tokens = [list(t) for t in ref.get("tokens") or []]
        cand_free = [list(t) for t in cand.get("tokens") or []]
        # Probe runs decode fewer samples than the baseline: compare the samples the candidate
        # produced (a candidate that produced none still scores as missing).
        n_cand = max(len(cand_free), len(cand.get("agree") or []))
        n = min(len(ref_tokens), n_cand) if n_cand else len(ref_tokens)
        ref_tokens = ref_tokens[:n]
        exact, prefix = [], []
        for i, r in enumerate(ref_tokens):
            c = cand_free[i] if i < len(cand_free) else []
            exact.append(c == r)
            prefix.append(_prefix_len(c, r) / len(r) if r else 1.0)
        free = {"free_exact_match": sum(exact) / n if n else None,
                "free_prefix_frac": sum(prefix) / n if n else None}
        if cand.get("kind") != "forced":
            return {"per_sample": [1.0 - p for p in prefix],
                    "summary": {"n": n, "note": "candidate has no forced outputs", **free}}
        agree = cand.get("agree") or []
        per_sample, fracs, first, steps, hits = [], [], [], 0, 0
        for i in range(n):
            a = agree[i] if i < len(agree) else []
            if not a:  # sample missing from the candidate -> maximal discrepancy
                per_sample.append(1.0)
                fracs.append(0.0)
                first.append(False)
                continue
            f = sum(a) / len(a)
            per_sample.append(1.0 - f)
            fracs.append(f)
            first.append(bool(a[0]))
            steps += len(a)
            hits += sum(a)
        dis = [(m, r) for ms in cand.get("margins") or [] for (_, m, r) in ms]
        finite = sorted(m for m, _ in dis if m is not None)
        summary: dict[str, Any] = {
            "n": n,
            "top1_agreement": sum(fracs) / n if n else None,       # mean over requests
            "top1_agreement_steps": hits / steps if steps else None,  # pooled over steps
            "min_agreement": min(fracs) if fracs else None,
            "first_step_agreement": sum(first) / n if n else None,
            "forced_steps": steps,
            "forced_ok": cand.get("forced_ok"),
            "n_disagree": len(dis),
            "n_disagree_nonfinite_logits": len(dis) - len(finite),
            **free,
        }
        if finite:
            summary["disagree_logit_margin_median"] = finite[len(finite) // 2]
            summary["disagree_logit_margin_max"] = finite[-1]
            summary["disagree_ref_rank_max"] = max(r for m, r in dis if m is not None)
        return {"per_sample": per_sample, "summary": summary}
