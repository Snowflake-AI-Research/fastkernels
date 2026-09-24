"""Vision-language (and omni-modal) models: Qwen3-VL, Qwen2-VL, Qwen2.5-Omni.

Runs the fastkernels side of the ``validate/bench_vllm.py`` VLM benchmark in-process on
one ``LlamaEngine`` (TP workers spawned by the engine):

* throughput workloads (``VLM.text_only`` / ``image`` / ``video``, Omni ``text`` /
  ``image`` / ``video`` / ``audio``): the bench's datasets, seeds and loaders
  (``_preload_mm_data`` + ``_filter_and_prepare`` are executed from the bench's own
  source, so filters and prompt construction are identical), one 16-token engine
  warmup, then per workload ``spec.extra["warmup_passes"]`` (default 1) full untimed
  passes over the same requests (so lazily JIT-compiled / autotuned kernels never compile
  inside the timed region; with 0, the bench's prefill-only warmup at the real shapes),
  then one timed ``generate`` of all requests (greedy, ``ignore_eos``) -> output tok/s;
* latency workloads (``single-image`` / ``single-video`` / ...): the first usable item,
  ``num_warmup`` untimed + ``num_iters`` timed batch-1 generates -> median seconds.

Correctness is autoregressive, so it is teacher-forced on the baseline's tokens (same
format as the text LLM adapter), with per-sample records for every throughput modality.
The correctness subset is the first ``spec.correctness_samples`` requests of each
throughput workload (same inputs and output lengths as in the timed run). After timing,
EVERY run decodes the subset greedily as its own batches, one batch per workload (= per
modality), so the reference tokens and the forced re-decode share one batch regime:

* baseline (``spec.reference is None``): ``{"kind": "tokens", "tokens", "modality",
  "keys", "workload"}`` with the free-running greedy tokens of the subset;
* with a reference: the same batches are decoded again forced along ``ref["tokens"]``
  (per step: record whether our argmax equals the reference token, then substitute the
  reference token; ``FASTKERNELS_FORCE_SYNC_DECODE=1``) -> ``{"kind": "forced", "agree",
  "forced_ok", ...}`` plus this run's own free-running ``tokens``.

``compare``: per-sample ``d = 1 - mean(agree)``; summary = top-1 agreement overall and
per modality, first-step agreement, and free-running exact-match / prefix fractions.

``spec.extra`` knobs (all optional): ``warmup_passes`` (default 1),
``max_correctness_tokens`` (cap on the decoded length per correctness sample; default:
the workload's own output length), ``latency_iters`` / ``latency_warmup`` (override the
workload spec), ``max_layers`` (debug only), ``gpu_memory_utilization``.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import Adapter, RunSpec, Timing

# bench_vllm sizes the context for media workloads as 16384 prompt tokens + decode.
_MEDIA_PROMPT_BUDGET = 16384

_MM_NS: dict | None = None


def _mm_helpers() -> dict:
    """``_preload_mm_data`` / ``_filter_and_prepare`` from bench_vllm's worker source.

    The bench keeps its multimodal loaders inside a worker-source string, so they are
    executed from that string here: same datasets, shuffles, filters and prompts.
    """
    global _MM_NS
    if _MM_NS is None:
        from fastkernels.validate import bench_vllm
        ns: dict = {"__name__": "fastkernels_e2e_vlm_mm"}
        exec(compile(bench_vllm._MM_PRELOAD_FN, "<bench_vllm._MM_PRELOAD_FN>", "exec"), ns)
        _MM_NS = ns
    return _MM_NS


def _wl_selected(wl, wanted: list[str] | None) -> bool:
    if not wanted:
        return True
    names = {wl.value, wl.name, f"{type(wl).__name__}.{wl.name}", wl.value.replace("-", "_")}
    return any(w in names for w in wanted)


def _model_max_ctx(model: str) -> int | None:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None
    for c in (cfg, getattr(cfg, "text_config", None)):
        v = getattr(c, "max_position_embeddings", None) if c is not None else None
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return None


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _media_args(items: list[dict]) -> tuple[list, list, list]:
    """Per-request ``images`` / ``videos`` / ``audio_features`` exactly as the bench's
    fastkernels VLM worker builds them (videos as ``(frames, metadata)`` pairs)."""
    images, videos, audios = [], [], []
    for it in items:
        images.append(it["images"] if it["images"] is not None else None)
        videos.append([(it["video_frames"], it["video_metadata"])]
                      if it["video_frames"] is not None else None)
        audios.append([it["audio"]] if it["audio"] is not None else None)
    return images, videos, audios


class _Forcer:
    """Patch ``Sequence.append_token``: record whether the engine's token equals the
    reference token at that step, then append the reference token instead.

    Same mechanism as ``validate.forced_decode.force_along_reference``, but agreement is
    stored by position, so a preempted-and-recomputed sequence overwrites (not
    duplicates) its steps.
    """

    def __init__(self, Sequence, ref_by_idx: dict[int, list[int]]):
        self.Sequence = Sequence
        self.ref = ref_by_idx
        self.agree: dict[int, dict[int, bool]] = {}

    def __enter__(self):
        self.orig = orig = self.Sequence.append_token
        ref, agree = self.ref, self.agree

        def _forced(seq, tid):
            ri = getattr(seq, "_req_idx", None)
            r = ref.get(ri) if ri is not None else None
            if r is not None:
                pos = len(seq.generated_ids)
                if pos < len(r):
                    agree.setdefault(ri, {})[pos] = bool(int(tid) == r[pos])
                    tid = r[pos]
            return orig(seq, tid)

        self.Sequence.append_token = _forced
        return self

    def __exit__(self, *exc):
        self.Sequence.append_token = self.orig
        return False


class VLMAdapter(Adapter):
    name = "vlm"
    metric = ("1 - per-step top-1 agreement of greedy decoding teacher-forced on the "
              "baseline's tokens (per sample; text/image/video[/audio] requests)")

    @classmethod
    def handles(cls, scenario) -> bool:
        from fastkernels.workloads import VLM, OmniModal
        return bool(scenario.workloads) and all(
            isinstance(w, (VLM, OmniModal)) for w in scenario.workloads)

    # ------------------------------------------------------------------ run
    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import torch
        from transformers import AutoProcessor

        from fastkernels.validate import bench_vllm
        from fastkernels.workloads import Purpose, load_real_prompt_workload, spec_for

        extra = spec.extra or {}
        model = scenario.hf_name
        max_corr_tokens = int(extra.get("max_correctness_tokens") or 0)  # 0 = no cap
        warmup_passes = int(extra.get("warmup_passes", 1))
        wls = [w for w in scenario.workloads if _wl_selected(w, spec.workloads)]
        if not wls:
            raise ValueError(f"workloads {spec.workloads} match none of "
                             f"{[w.value for w in scenario.workloads]}")
        thr = [w for w in wls if spec_for(w).purpose is Purpose.THROUGHPUT]
        lat = [w for w in wls if spec_for(w).purpose is Purpose.LATENCY]
        # Seeds are tied to the workload's position in the FULL scenario list (as in the
        # bench, where every workload runs), so a --workloads subset draws the same data.
        thr_index = {w: i for i, w in enumerate(scenario.throughput_workloads)}
        lat_index = {w: j for j, w in enumerate(scenario.latency_workloads)}

        # ---- per-model engine defaults (bench_vllm._PER_MODEL_DEFAULTS), NCCL port ----
        engine_kwargs: dict[str, Any] = {}
        lower = model.lower()
        for key, d in bench_vllm._PER_MODEL_DEFAULTS.items():
            if key in lower:
                for k, v in d.get("env", {}).items():
                    os.environ.setdefault(k, v)
                if d.get("gpu_memory_utilization") is not None:
                    engine_kwargs["gpu_memory_utilization"] = d["gpu_memory_utilization"]
        if extra.get("gpu_memory_utilization") is not None:
            engine_kwargs["gpu_memory_utilization"] = float(extra["gpu_memory_utilization"])
        if scenario.tp > 1 and not os.environ.get("FASTKERNELS_NCCL_PORT"):
            # A free port, so concurrent runs on one host do not collide. The engine reads
            # it at import (rank 0 may already have imported it via the candidates) and
            # spawned ranks re-read it from the environment.
            port = _free_port()
            os.environ["FASTKERNELS_NCCL_PORT"] = str(port)
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(port))
        os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

        from fastkernels.infra import engine as engine_mod
        if os.environ.get("FASTKERNELS_NCCL_PORT"):
            engine_mod.NCCL_PORT = int(os.environ["FASTKERNELS_NCCL_PORT"])
        LlamaEngine, SamplingParams, Sequence = (
            engine_mod.LlamaEngine, engine_mod.SamplingParams, engine_mod.Sequence)

        # ---- text prompts + context length (bench_vllm.main) ----
        tokenizer = bench_vllm._load_tokenizer(model)
        model_ctx = _model_max_ctx(model)

        def _fit(samples):
            ids_out = []
            for s in samples:
                ids = list(s.prompt_token_ids)
                if model_ctx is not None and s.messages is not None:
                    budget = model_ctx - s.output_len
                    if budget >= 1 and len(ids) > budget:
                        ids = bench_vllm._fit_messages_to_context(tokenizer, s.messages, budget)
                ids_out.append(ids)
            return ids_out

        def _n_req(params) -> int:
            n = getattr(params, "num_requests", 1000)
            return min(n, spec.max_requests) if spec.max_requests else n

        text_data: dict = {}
        max_seq_len = 0
        for w in thr:
            p = spec_for(w).params
            if p.modality == "text":
                samples = load_real_prompt_workload(
                    w.value, tokenizer, num_requests=_n_req(p), decode_cap=None,
                    dataset_name=p.dataset_name, seed=spec.seed + thr_index[w])
                prompts = _fit(samples)
                out_lens = [s.output_len for s in samples]
                text_data[w] = (prompts, out_lens)
                max_seq_len = max(max_seq_len, max(len(a) + b for a, b in zip(prompts, out_lens)))
            else:
                max_seq_len = max(max_seq_len, _MEDIA_PROMPT_BUDGET + p.output_len)
        for w in lat:
            p = spec_for(w).params
            if p.modality == "text":
                samples = load_real_prompt_workload(
                    "mixed", tokenizer, num_requests=p.batch_size, decode_cap=p.output_len,
                    dataset_name=p.dataset_name or None, seed=spec.seed + 100 + lat_index[w])
                prompts = _fit(samples)
                text_data[w] = (prompts, [s.output_len for s in samples])
                max_seq_len = max(max_seq_len, max(len(a) + p.output_len for a in prompts))
            else:
                max_seq_len = max(max_seq_len, _MEDIA_PROMPT_BUDGET + p.output_len)
        if model_ctx is not None:
            max_seq_len = min(max_seq_len, model_ctx)

        # ---- engine ----
        dt = getattr(torch, scenario.dtype, None)
        engine_kwargs.update(
            model_name=model,
            dtype=dt if isinstance(dt, torch.dtype) else None,  # fp8 etc.: from the checkpoint
            seed=spec.seed,
            tensor_parallel_size=scenario.tp,
            enforce_eager=bool(spec.enforce_eager or scenario.enforce_eager),
            max_model_len=max_seq_len,
        )
        if scenario.max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = scenario.max_num_seqs
        if scenario.kv_cache_dtype:
            engine_kwargs["kv_cache_dtype"] = scenario.kv_cache_dtype
        if extra.get("max_layers"):
            engine_kwargs["max_layers"] = int(extra["max_layers"])
        print(f"[vlm] building LlamaEngine: {engine_kwargs}", flush=True)
        t0 = time.perf_counter()
        engine = LlamaEngine(**engine_kwargs)
        print(f"[vlm] engine ready in {time.perf_counter() - t0:.1f}s", flush=True)
        processor = engine.processor or AutoProcessor.from_pretrained(model, trust_remote_code=True)
        mm = _mm_helpers()

        timings: dict[str, Timing] = {}
        details: dict[str, dict] = {}
        corr: list[dict] = []   # correctness subset: {key, workload, modality, max_tokens, inputs}
        try:
            engine.generate([[0] * 16], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))

            # ---- throughput ----
            for w in thr:
                p = spec_for(w).params
                if p.modality == "text":
                    prompts, out_lens = text_data[w]
                    images = videos = audios = None
                    sp_list = [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=ol,
                                              ignore_eos=True) for ol in out_lens]
                    items = None
                else:
                    t_load = time.perf_counter()
                    items = mm["_preload_mm_data"](p.dataset_name, p.dataset_split, _n_req(p), spec.seed)
                    items = mm["_filter_and_prepare"](items, processor, max_seq_len - p.output_len)
                    if not items:
                        raise RuntimeError(f"{w.value}: no usable {p.modality} requests")
                    print(f"[vlm] {w.value}: {len(items)} {p.modality} requests loaded in "
                          f"{time.perf_counter() - t_load:.1f}s", flush=True)
                    prompts = [it["prompt"] for it in items]
                    images, videos, audios = _media_args(items)
                    sp_list = [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=p.output_len,
                                              ignore_eos=True)] * len(items)
                if warmup_passes > 0:
                    # Full untimed pass(es) over the same inputs, so lazily JIT-compiled
                    # kernels (FlashInfer, DeepGEMM, Triton autotune, candidates) are not
                    # compiled inside the timed region.
                    for _ in range(warmup_passes):
                        engine.block_manager.reset()
                        engine.generate(prompts, sp_list, images=images, videos=videos,
                                        audio_features=audios, use_tqdm=False, decode_text=False)
                else:
                    # bench_vllm's prefill warmup at the real shapes (vision encoder included).
                    engine.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0,
                                                            max_tokens=1, ignore_eos=True),
                                    images=images, videos=videos, audio_features=audios,
                                    use_tqdm=False, decode_text=False)
                engine.block_manager.reset()
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = engine.generate(prompts, sp_list, images=images, videos=videos,
                                          audio_features=audios, use_tqdm=False, decode_text=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                n_out = sum(len(o.token_ids) for o in outputs)
                timings[w.value] = {"kind": "throughput", "value": n_out / elapsed, "unit": "tok/s"}
                details[w.value] = {"elapsed_s": elapsed, "num_requests": len(outputs),
                                    "output_tokens": n_out}
                print(f"[vlm] {w.value}: {len(outputs)} req, {n_out} tok in {elapsed:.2f}s "
                      f"-> {n_out / elapsed:.1f} tok/s", flush=True)
                # Inputs of the correctness subset: the first requests of this workload.
                for i in range(min(spec.correctness_samples, len(prompts))):
                    ol = sp_list[i].max_tokens
                    corr.append({
                        "key": f"{w.value}:{i}", "workload": w.value, "modality": p.modality,
                        "max_tokens": min(ol, max_corr_tokens) if max_corr_tokens else ol,
                        "prompt": prompts[i],
                        "images": images[i] if images else None,
                        "videos": videos[i] if videos else None,
                        "audios": audios[i] if audios else None,
                    })
                del outputs, items, prompts, images, videos, audios

            # ---- latency ----
            for w in lat:
                p = spec_for(w).params
                sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=p.output_len)
                if p.modality == "text":
                    prompts, out_lens = text_data[w]
                    sp = [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
                          for ol in out_lens]
                    kw: dict = {}
                else:
                    items = mm["_preload_mm_data"](p.dataset_name, p.dataset_split, 1, spec.seed)
                    items = mm["_filter_and_prepare"](items, processor, max_seq_len - p.output_len)
                    if not items:
                        raise RuntimeError(f"{w.value}: no usable {p.modality} request")
                    images, videos, audios = _media_args(items[:1])
                    prompts = [items[0]["prompt"]]
                    kw = {"images": images if images[0] is not None else None,
                          "videos": videos if videos[0] is not None else None,
                          "audio_features": audios if audios[0] is not None else None}

                def _once():
                    engine.block_manager.reset()
                    torch.cuda.synchronize()
                    engine.generate(prompts, sp, use_tqdm=False, decode_text=False, **kw)
                    torch.cuda.synchronize()

                for _ in range(int(extra.get("latency_warmup", p.num_warmup))):
                    _once()
                lats = []
                for _ in range(int(extra.get("latency_iters", p.num_iters))):
                    t1 = time.perf_counter()
                    _once()
                    lats.append(time.perf_counter() - t1)
                lats_sorted = sorted(lats)
                n = len(lats_sorted)
                med = (lats_sorted[n // 2] if n % 2 else
                       0.5 * (lats_sorted[n // 2 - 1] + lats_sorted[n // 2]))
                timings[w.value] = {"kind": "latency", "value": med, "unit": "s",
                                    "samples": lats}
                details[w.value] = {"latencies_s": lats, "batch_size": len(prompts),
                                    "output_len": p.output_len}
                print(f"[vlm] {w.value}: median {med * 1000:.1f} ms over {n} iters", flush=True)

            # ---- correctness: the subset decoded as its own batches, in every run ----
            # (the batch regime must match between the reference tokens and the forced
            # re-decode, or near-tie argmaxes flip systematically with the batch size)
            free = self._decode(engine, SamplingParams, Sequence, corr, None)
            out: dict[str, Any] = {
                "kind": "tokens",
                "keys": [c["key"] for c in corr],
                "workload": [c["workload"] for c in corr],
                "modality": [c["modality"] for c in corr],
                "tokens": free["tokens"],
                "details": details,
            }
            if spec.reference is not None:
                import torch as _t
                ref = _t.load(spec.reference, weights_only=False)
                forced = self._decode(engine, SamplingParams, Sequence, corr,
                                      dict(zip(ref.get("keys") or [], ref.get("tokens") or [])))
                out.update(kind="forced", agree=forced["agree"], forced_ok=forced["forced_ok"])
            torch.save(out, os.path.join(spec.out_dir, "outputs.pt"))
        finally:
            try:
                engine._cleanup()
            except Exception:
                pass
        return timings

    @staticmethod
    def _decode(engine, SamplingParams, Sequence, corr: list[dict],
                ref_by_key: dict | None) -> dict:
        """Greedy-decode the correctness samples, one batch per workload (so one modality
        per batch). With ``ref_by_key``, decode teacher-forced along the reference tokens
        (samples without a reference are skipped) and record per-step agreement."""
        forced = ref_by_key is not None
        tokens: list[list[int]] = [[] for _ in corr]
        agree: list[list[bool]] = [[] for _ in corr]
        forced_ok: list[bool] = [False for _ in corr]
        groups: dict[str, list[int]] = {}
        for k, c in enumerate(corr):
            if not forced or ref_by_key.get(c["key"]):
                groups.setdefault(c["workload"], []).append(k)
        prev = os.environ.get("FASTKERNELS_FORCE_SYNC_DECODE")
        if forced:
            # Every decode step rebuilds its input from the live sequences, so the
            # substituted token is what the next step consumes (the async fast path
            # would reuse the sampled token still on the device).
            os.environ["FASTKERNELS_FORCE_SYNC_DECODE"] = "1"
        try:
            for wl, idxs in groups.items():
                refs = ([list(map(int, ref_by_key[corr[k]["key"]])) for k in idxs]
                        if forced else None)
                prompts = [corr[k]["prompt"] for k in idxs]
                kw = {}
                for name, f in (("images", "images"), ("videos", "videos"),
                                ("audio_features", "audios")):
                    vals = [corr[k][f] for k in idxs]
                    kw[name] = vals if any(v is not None for v in vals) else None
                lens = [len(r) for r in refs] if forced else [corr[k]["max_tokens"] for k in idxs]
                sp = [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n, ignore_eos=True)
                      for n in lens]
                engine.block_manager.reset()
                t0 = time.perf_counter()
                if forced:
                    with _Forcer(Sequence, dict(enumerate(refs))) as f:
                        outs = engine.generate(prompts, sp, use_tqdm=False, decode_text=False, **kw)
                else:
                    outs = engine.generate(prompts, sp, use_tqdm=False, decode_text=False, **kw)
                for j, k in enumerate(idxs):
                    tokens[k] = [int(t) for t in outs[j].token_ids]
                    if forced:
                        steps = f.agree.get(j, {})
                        agree[k] = [steps.get(pos, False) for pos in range(len(refs[j]))]
                        forced_ok[k] = tokens[k] == refs[j]
                msg = ""
                if forced:
                    n_steps = sum(len(agree[k]) for k in idxs)
                    msg = f", agreement {sum(sum(agree[k]) for k in idxs)}/{n_steps}"
                print(f"[vlm] {'forced' if forced else 'free'} decode {wl}: {len(idxs)} samples"
                      f"{msg} in {time.perf_counter() - t0:.1f}s", flush=True)
        finally:
            if prev is None:
                os.environ.pop("FASTKERNELS_FORCE_SYNC_DECODE", None)
            else:
                os.environ["FASTKERNELS_FORCE_SYNC_DECODE"] = prev
        return {"tokens": tokens, "agree": agree, "forced_ok": forced_ok}

    # -------------------------------------------------------------- compare
    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        """Samples are aligned by ``keys`` (``<workload>:<i>``), else by position."""
        rt = ref.get("tokens") or []
        n = len(rt)
        ref_keys = ref.get("keys") or [str(i) for i in range(n)]
        ref_mod = ref.get("modality") or ["?"] * n
        ct = cand.get("tokens") or []
        cidx = {k: i for i, k in enumerate(cand.get("keys") or [str(i) for i in range(len(ct))])}
        c_agree = cand.get("agree") if cand.get("kind") == "forced" else None

        rows = []   # (modality, d, agreement or None, first-step or None, exact, prefix)
        for i in range(n):
            r = [int(t) for t in rt[i]]
            j = cidx.get(ref_keys[i])
            c = [int(t) for t in ct[j]] if j is not None and j < len(ct) else None
            pre = 0
            for a, b in zip(r, c or []):
                if a != b:
                    break
                pre += 1
            exact, prefix = c == r, (pre / max(1, len(r)) if c is not None else 0.0)
            ag = None
            if c_agree is not None and j is not None and j < len(c_agree) and c_agree[j]:
                ag = [bool(x) for x in c_agree[j]]
            if ag:
                d = 1.0 - sum(ag) / len(ag)
            elif c_agree is None and c is not None:
                d = 1.0 - prefix   # candidate ran without a reference (no teacher forcing)
            else:
                d = 1.0            # sample missing from the candidate
            rows.append((ref_mod[i], d, (1.0 - d) if ag else None, ag[0] if ag else None,
                         exact, prefix))

        def _agg(sel):
            forced_rows = [x for x in sel if x[2] is not None]
            out = {"n": len(sel),
                   "free_exact_match": sum(x[4] for x in sel) / len(sel) if sel else None,
                   "free_prefix_frac": sum(x[5] for x in sel) / len(sel) if sel else None}
            if forced_rows:
                out.update(
                    top1_agreement=sum(x[2] for x in forced_rows) / len(forced_rows),
                    min_agreement=min(x[2] for x in forced_rows),
                    first_step_agreement=sum(x[3] for x in forced_rows) / len(forced_rows))
            return out

        summary = _agg(rows)
        summary["mean_d"] = sum(x[1] for x in rows) / len(rows) if rows else None
        if c_agree is None:
            summary["note"] = "candidate has no forced outputs"
        if cand.get("forced_ok") is not None:
            fo = cand["forced_ok"]
            summary["forced_ok"] = bool(fo) and all(bool(x) for x in fo)
        summary["by_modality"] = {m: _agg([x for x in rows if x[0] == m])
                                  for m in dict.fromkeys(ref_mod)}
        return {"per_sample": [x[1] for x in rows], "summary": summary}
