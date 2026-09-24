"""Oasis 500M world model (``Etched/oasis-500m``) for ``fastkernels e2e``.

Runs the fastkernels side of ``fastkernels/validate/bench_oasis.py`` in-process (no
open-oasis reference): the same model (``OasisPipeline``, fp32 weights, fp16 autocast;
constructed directly on the GPU, see ``_build_pipeline``), the same real Minecraft prompt
frames + action streams (the harness's
TESS-Computer/minecraft-vla-stage1 cache), the same per-rollout seeding
(``Generator.manual_seed(seed)`` in ``OasisRollout``) and the same timing loop
(``bench_oasis._benchmark``: 2 warmup + 5 timed full rollouts per workload, each rollout =
prompt VAE encode + DiT DDIM rollout + VAE decode).

* throughput workloads -> ``videos/s`` (clips per timed rollout x iters / total time);
* latency workload     -> median (p50) seconds per rollout;
* ``spec.max_requests`` caps the warmup and timed rollouts per workload (shapes unchanged);
* ``spec.workloads`` filters by workload name (``short-bs4-16f-4ddim``,
  ``short_bs4_16f_4ddim`` or ``WorldModel.short_bs4_16f_4ddim``);
* ``spec.enforce_eager`` is a no-op: the Oasis path uses neither torch.compile nor graphs.

Correctness samples are clips generated with exactly the timed workload shapes (batch
size, frames, DDIM steps), taken round-robin over the scenario's workloads so that even a
few samples cover every batch size and rollout length; round ``r`` reuses the same clips
with seed ``spec.seed + r``. Each sample stores the generated latents (fp16, incl. the
prompt frame) and the decoded video (4x average-pooled to 90x160, uint8); 64 samples are
~90 MB. Calibration of ``d`` on B200 (5-10 clips): a fresh baseline process is bit-exact
(d = 0); TF32 on ~4e-6; fp16 weights (``OasisEngine``) <= 8e-4; SDPA math backend
<= 2e-3; 1% relative noise on every DiT output <= 3e-3. ``d`` grows with rollout length.

The model code is imported inside ``run`` -- after the runner has patched candidate
classes in -- so candidate kernels are what the pipeline instantiates.

Input data: the harness caches prompt/action tensors built from a streamed HF dataset.
Prepare them once (CPU) with ``python -m fastkernels.e2e.adapters.oasis prepare``; the
cache lives in ``$FASTKERNELS_OASIS_CACHE_DIR`` or ``~/.fastkernels/data/oasis_cache``
(the harness's own ``data/oasis_cache`` is used instead when it already holds the file).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .base import Adapter, RunSpec, Timing

LATENCY_ITERS = 5  # bench_oasis ``--latency-iters`` default
VIDEO_POOL = 4     # stored video: 360x640 -> 90x160 (average pool), uint8
CACHE_ENV = "FASTKERNELS_OASIS_CACHE_DIR"
_DTYPES = {"float16": "float16", "fp16": "float16", "half": "float16",
           "bfloat16": "bfloat16", "bf16": "bfloat16", "float32": "float32", "fp32": "float32"}


def _log(msg: str) -> None:
    print(f"[e2e-oasis] {msg}", flush=True)


def _oasis_workloads(scenario) -> list[tuple]:
    """``(workload enum, OasisWorkload)`` for the scenario's world-model workloads."""
    from fastkernels.workloads import OasisWorkload
    return [(ws.workload, ws.params) for ws in scenario.specs if isinstance(ws.params, OasisWorkload)]


def _workload_names(workload) -> set[str]:
    return {workload.value, workload.name, f"{type(workload).__name__}.{workload.name}"}


def _bench():
    """The validate harness module (imported lazily: it imports the Oasis model code)."""
    from fastkernels.validate import bench_oasis
    return bench_oasis


def _cache_dir(bench, params: list) -> Path:
    """Where the prompt/action cache lives: $FASTKERNELS_OASIS_CACHE_DIR, else the
    harness's ``data/oasis_cache`` if it already has the file, else ~/.fastkernels."""
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env)
    clips, frames, prompt_frames = bench._max_workload_requirements(params)
    harness_dir = bench._default_cache_dir()
    if bench._dataset_cache_path(params[0].dataset_name, params[0].dataset_split, cache_dir=harness_dir,
                                 num_clips=clips, num_frames=frames,
                                 n_prompt_frames=prompt_frames).exists():
        return harness_dir
    return Path.home() / ".fastkernels" / "data" / "oasis_cache"


def _load_inputs(bench, params: list, device):
    """Prompt frames + actions for the largest workload (same cache file as the harness)."""
    if any((p.dataset_name, p.dataset_split) != (params[0].dataset_name, params[0].dataset_split)
           for p in params):
        raise ValueError("oasis workloads must share one dataset")
    clips, frames, prompt_frames = bench._max_workload_requirements(params)
    return bench._load_real_dataset_inputs(
        dataset_name=params[0].dataset_name, dataset_split=params[0].dataset_split,
        cache_dir=_cache_dir(bench, params), num_clips=clips, num_frames=frames,
        n_prompt_frames=prompt_frames, device=device)


def _correctness_plan(params: list, n: int) -> list[tuple[int, int, int]]:
    """First ``n`` samples as ``(round, workload index, clip index)``: round-robin over
    workloads, clip by clip, so a few samples already cover every shape."""
    plan: list[tuple[int, int, int]] = []
    max_bs = max(p.batch_clips for p in params)
    rnd = 0
    while len(plan) < n:
        for clip in range(max_bs):
            for wi, p in enumerate(params):
                if clip < p.batch_clips and len(plan) < n:
                    plan.append((rnd, wi, clip))
        rnd += 1
    return plan


def _candidate_modules(root) -> dict[str, int]:
    """Instantiated submodules whose class comes from a candidate set, with counts."""
    counts: dict[str, int] = {}
    for m in root.modules():
        mod = type(m).__module__
        if ".tasks.candidate." in f".{mod}.":
            key = f"{mod.split('tasks.candidate.')[-1]}.{type(m).__name__}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _build_pipeline(bench, model_dir: str, device):
    """``bench_oasis._build_fastkernels`` (fp32 weights; autocast at run time), except that
    the modules are constructed directly on the GPU: the harness builds (and randomly
    initialises) a ~3 GB fp32 copy on the host first, which in a shared container pushes
    peak host RSS past 6 GB and got runs killed silently while loading weights. Every
    parameter is then overwritten by the checkpoint (the ones it lacks are deterministic
    rotary tables), so the resulting model is the same."""
    import torch
    with torch.device(device):
        pipeline = bench.OasisPipeline(bench.OasisConfig())
    pipeline.load_weights(model_dir)
    pipeline.model.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline.eval()


def _max_rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB on Linux


def _weight_coverage(model_dir: str, pipeline) -> dict:
    """``load_weights`` is non-strict: count model parameters the checkpoint does not
    fill (a candidate renaming parameters would silently run with random weights)."""
    from safetensors import safe_open
    out = {}
    for part, fname in (("dit", "oasis500m.safetensors"), ("vae", "vit-l-20.safetensors")):
        with safe_open(os.path.join(model_dir, fname), "pt") as f:
            keys = set(f.keys())
        params = [k for k, _ in getattr(pipeline, "model" if part == "dit" else "vae").named_parameters()]
        out[f"{part}_params_not_in_checkpoint"] = sum(k not in keys for k in params)
    return out


class OasisAdapter(Adapter):
    name = "oasis"
    metric = ("per clip d = clamp(1 - min(cos(latents), corr(video)), 0, 1); latents = generated "
              "latents incl. prompt frame (fp16), video = decoded frames 4x avg-pooled to 90x160 "
              "uint8, corr = mean-centred cosine (Pearson); clips use the timed workload shapes")

    @classmethod
    def handles(cls, scenario) -> bool:
        from fastkernels.workloads import WorldModel
        return "oasis" in scenario.hf_name.lower() or (
            bool(scenario.workloads) and all(isinstance(w, WorldModel) for w in scenario.workloads))

    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import torch
        import torch.nn.functional as F

        bench = _bench()  # imports tasks.baseline.L4.oasis -- after candidates were applied
        device = torch.device("cuda")
        dtype = getattr(torch, _DTYPES.get(str(scenario.dtype).lower(), "float16"))
        workloads = _oasis_workloads(scenario)
        if not workloads:
            raise ValueError(f"{scenario.hf_name}: no WorldModel workloads")
        params = [p for _, p in workloads]
        if spec.workloads:
            known = set().union(*(_workload_names(w) for w, _ in workloads))
            unknown = [w for w in spec.workloads if w not in known]
            if unknown:
                raise ValueError(f"unknown oasis workloads {unknown}; known: {sorted(known)}")
        timed = [(w, p) for w, p in workloads
                 if not spec.workloads or _workload_names(w) & set(spec.workloads)]
        if spec.enforce_eager:
            _log("enforce_eager: no-op (the Oasis pipeline uses no torch.compile / CUDA graphs)")

        torch.manual_seed(spec.seed)
        prompt, actions, input_info = _load_inputs(bench, params, device)
        model_dir = bench._download_model(scenario.hf_name)
        pipeline = _build_pipeline(bench, model_dir, device)
        cand_mods = _candidate_modules(pipeline)
        coverage = _weight_coverage(model_dir, pipeline)
        _log(f"model classes: dit={type(pipeline.model).__module__}.{type(pipeline.model).__name__}, "
             f"candidate submodules={cand_mods or 'none'}, {coverage}, "
             f"host max RSS {_max_rss_gb():.1f} GB")

        # Correctness: clips generated with the timed workload shapes (see module doc).
        plan = _correctness_plan(params, max(0, spec.correctness_samples))
        samples: list[dict] = [{} for _ in plan]
        latents: list = [None] * len(plan)
        videos: list = [None] * len(plan)
        t0 = time.time()
        for rnd, wi in sorted({(r, w) for r, w, _ in plan}):
            wl = params[wi]
            p, a = bench._slice_inputs(prompt, actions, wl)
            seed = spec.seed + rnd
            out = bench._run_kb_pipeline(pipeline, p, a, wl, dtype=dtype, seed=seed)
            for i, (r, w, clip) in enumerate(plan):
                if (r, w) != (rnd, wi):
                    continue
                vid = F.avg_pool2d(out["video"][clip].float(), VIDEO_POOL)
                latents[i] = out["latents"][clip].to(torch.float16).cpu()
                videos[i] = (vid.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                samples[i] = {"workload": wl.name, "clip": clip, "round": rnd, "seed": seed,
                              "num_frames": wl.num_frames, "ddim_steps": wl.ddim_steps,
                              "batch_clips": wl.batch_clips}
            del out
        torch.cuda.synchronize(device)
        _log(f"correctness: {len(plan)} clips in {time.time() - t0:.1f}s")
        torch.save({"kind": "videos", "samples": samples, "latents": latents, "video": videos,
                    "meta": {"model": scenario.hf_name, "dtype": str(dtype), "seed": spec.seed,
                             "video_pool": VIDEO_POOL, "input": {k: v for k, v in input_info.items()
                                                                 if k != "clips"},
                             "candidate_modules": cand_mods, **coverage}},
                   Path(spec.out_dir) / "outputs.pt")

        # Timing: bench_oasis._benchmark over full rollouts, per workload.
        timings: dict[str, Timing] = {}
        for w, wl in timed:
            p, a = bench._slice_inputs(prompt, actions, wl)
            iters = LATENCY_ITERS if wl.kind == "latency" else bench.THROUGHPUT_ITERS
            warmup = bench.WARMUP_ITERS
            if spec.max_requests is not None and spec.max_requests > 0:
                iters, warmup = min(iters, spec.max_requests), min(warmup, spec.max_requests)
            m = bench._benchmark(
                lambda wl=wl, p=p, a=a: bench._run_kb_pipeline(pipeline, p, a, wl, dtype=dtype,
                                                               seed=spec.seed),
                device=device, warmup=warmup, iters=iters, units_per_iter=int(p.shape[0]),
                desc=f"e2e-oasis {wl.name}")
            if wl.kind == "latency":
                timings[w.value] = {"kind": "latency", "value": m["latency_ms_p50"] / 1000.0, "unit": "s"}
            else:
                timings[w.value] = {"kind": "throughput", "value": m["videos_per_second"],
                                    "unit": "videos/s"}
            _log(f"{w.value}: {timings[w.value]['value']:.4f} {timings[w.value]['unit']} "
                 f"(warmup={warmup}, iters={iters}, p50={m['latency_ms_p50']:.1f}ms)")
        return timings

    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        import torch

        def cos(x, y) -> float:
            x, y = x.reshape(-1).double(), y.reshape(-1).double()
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                return 0.0
            nx, ny = x.norm(), y.norm()
            if nx == 0 and ny == 0:
                return 1.0
            if nx == 0 or ny == 0:
                return 0.0
            return float((x @ y) / (nx * ny))

        n = len(ref.get("samples") or [])
        per: list[float] = []
        lat_cos, vid_cos, vid_corr, lat_mse, vid_mse = [], [], [], [], []
        by_wl: dict[str, list[float]] = {}
        mismatched = nonfinite = 0
        for i in range(n):
            ok = (i < len(cand.get("samples") or []) and cand["samples"][i] == ref["samples"][i]
                  and cand["latents"][i] is not None
                  and cand["latents"][i].shape == ref["latents"][i].shape
                  and cand["video"][i].shape == ref["video"][i].shape)
            if not ok:
                mismatched += 1
                per.append(1.0)
                continue
            rl, cl = ref["latents"][i].float(), cand["latents"][i].float()
            rv, cv = ref["video"][i].float() / 255.0, cand["video"][i].float() / 255.0
            nonfinite += int(not torch.isfinite(cl).all())
            lc, vc = cos(rl, cl), cos(rv, cv)
            vr = cos(rv - rv.mean(), cv - cv.mean())
            d = min(1.0, max(0.0, 1.0 - min(lc, vr)))
            per.append(d)
            lat_cos.append(lc)
            vid_cos.append(vc)
            vid_corr.append(vr)
            diff = (rl - cl)
            lat_mse.append(float((diff * diff).mean()) if torch.isfinite(diff).all() else float("inf"))
            vid_mse.append(float(((rv - cv) ** 2).mean()))
            by_wl.setdefault(ref["samples"][i]["workload"], []).append(d)

        def mean(xs):
            return sum(xs) / len(xs) if xs else None

        vmse = mean(vid_mse)
        summary = {
            "n_samples": n, "mismatched_samples": mismatched, "nonfinite_samples": nonfinite,
            "latents_cos_mean": mean(lat_cos), "latents_cos_min": min(lat_cos, default=None),
            "video_cos_mean": mean(vid_cos), "video_cos_min": min(vid_cos, default=None),
            "video_corr_mean": mean(vid_corr), "video_corr_min": min(vid_corr, default=None),
            "latents_mse": mean(lat_mse), "video_mse": vmse,
            "video_psnr_db": (None if vmse is None else
                              float("inf") if vmse == 0 else float(-10.0 * torch.log10(torch.tensor(vmse)))),
            "d_mean": mean(per), "d_max": max(per, default=None),
            "d_mean_by_workload": {k: mean(v) for k, v in by_wl.items()},
        }
        return {"per_sample": per, "summary": summary}


def _main(argv: list[str]) -> int:
    """``python -m fastkernels.e2e.adapters.oasis prepare [--scenarios default]``: build the
    prompt/action cache (streams the HF dataset; CPU is enough) and fetch the weights."""
    import argparse

    import torch
    from fastkernels.workloads import resolve_benchmark

    ap = argparse.ArgumentParser(prog="python -m fastkernels.e2e.adapters.oasis")
    ap.add_argument("command", choices=["prepare"])
    ap.add_argument("--scenarios", default="default")
    args = ap.parse_args(argv)
    scenarios = [s for s in resolve_benchmark(args.scenarios) if OasisAdapter.handles(s)]
    bench = _bench()
    for s in scenarios:
        params = [p for _, p in _oasis_workloads(s)]
        prompt, actions, info = _load_inputs(bench, params, torch.device("cpu"))
        _log(f"{s.hf_name}: inputs prompt={tuple(prompt.shape)} actions={tuple(actions.shape)} "
             f"cache={info['cache_path']}")
        _log(f"{s.hf_name}: weights at {bench._download_model(s.hf_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
