"""FLUX text-to-image adapter for ``fastkernels e2e`` (e.g. black-forest-labs/FLUX.1-dev).

Mirrors the fastkernels side of ``fastkernels.validate.bench_vllm_omni``
(``FLUX_FASTKERNELS_WORKER``), but in-process (so swapped-in candidate kernels are used)
and without the vllm-omni reference:

* model: ``fastkernels.infra.diffusion_engine.DiffusionEngine`` -- the production FLUX path
  (eager transformer; the engine never torch.compiles FLUX, so ``enforce_eager`` is a no-op);
* prompts: nateraw/parti-prompts, shuffled with ``random.Random(seed)``;
* sampling: ``FLUX_CONFIG`` (28 steps, guidance 3.5), ``output_type="latent"`` for every
  timed call (VAE decode is not timed, as in the harness);
* warmup: one 2-step generation per distinct (height, width, batch) before timing;
* throughput workloads (``1024x1024``, ``512x512``): ``num_requests`` batches of
  ``batch_size`` prompts, a fresh ``manual_seed(seed)`` generator per image, each batch
  timed between ``cuda.synchronize`` calls; value = images / summed batch time (images/s);
* latency workloads (``single-*``): ``num_warmup`` untimed then ``num_iters`` timed
  single-prompt generations; value = median seconds.

``spec.max_requests`` caps the images per throughput workload (the batch size is kept
unless the cap is smaller, so no ragged batches) and the warmup/timed iterations per
latency workload. ``spec.workloads`` selects workloads by name (``1024x1024``,
``res_1024`` or ``Diffusion.res_1024``).

Correctness: ``spec.correctness_samples`` parti prompts (the first N of the shuffled
list), generated at 1024x1024 in batches of 4 with the workload's sampling config, image
``i`` seeded with ``manual_seed(seed + i)``. Decoded RGB is stored area-downsampled to
512x512 as uint8 (~0.75 MB/image), plus the final packed latents as bf16 (~0.5 MB/image).

Smoke-only knob: ``spec.extra["num_inference_steps"]`` or env ``FK_E2E_FLUX_STEPS``
overrides the step count of every generation (never set it for real runs).
"""

from __future__ import annotations

import math
import os
import statistics
import time

from .base import Adapter, RunSpec, Timing

_CORRECTNESS_RES = 1024     # correctness images are generated at FLUX's native resolution
_CORRECTNESS_BATCH = 4      # ... in batches of the 1024x1024 throughput workload's size
_STORE_RES = 512            # ... and stored area-downsampled to this size, as uint8
_PSNR_CAP = 100.0           # dB reported for bit-identical images (keeps JSON finite)


def _norm_wl(name: str) -> str:
    name = name.strip().lower()
    if name.startswith("diffusion."):
        name = name[len("diffusion."):]
    return name.replace("_", "-")


def _torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


class FluxAdapter(Adapter):
    name = "flux"
    metric = ("per image: d = clamp(1 - cos(ref, cand), 0, 1), cosine over the decoded RGB "
              "image (1024x1024 generation, stored area-downsampled to 512x512 uint8) mapped "
              "to [-1, 1]; non-finite or missing outputs score d = 1")

    @classmethod
    def handles(cls, scenario) -> bool:
        return "flux" in scenario.hf_name.lower()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _selected(workloads, spec: RunSpec):
        if spec.workloads is None:
            return list(workloads)
        want = {_norm_wl(w) for w in spec.workloads}
        return [w for w in workloads if _norm_wl(w.value) in want or _norm_wl(w.name) in want]

    @staticmethod
    def _steps(spec: RunSpec, default: int) -> int:
        override = spec.extra.get("num_inference_steps") or os.environ.get("FK_E2E_FLUX_STEPS")
        return int(override) if override else default

    @staticmethod
    def _decode(pipeline, latents, height: int, width: int):
        """Packed latents -> float RGB in [0, 1] (the harness' decode path), one image at a
        time to bound VAE activation memory."""
        import torch
        out = []
        for i in range(latents.shape[0]):
            x = pipeline._unpack_latents(latents[i:i + 1], height, width, pipeline.vae_scale_factor)
            x = (x / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            x = pipeline.vae.decode(x.to(dtype=pipeline.vae.dtype), return_dict=False)[0]
            out.append((x.float() / 2 + 0.5))
        return torch.cat(out, dim=0)

    # ---------------------------------------------------------------------- run

    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import torch
        import torch.nn.functional as F

        from fastkernels.infra.diffusion_engine import DiffusionEngine
        from fastkernels.tasks.baseline.L4.flux import DiffusionSamplingParams
        from fastkernels.validate.bench_vllm_omni import _load_parti_prompts
        from fastkernels.workloads import FLUX_CONFIG, spec_for

        seed = spec.seed
        steps = self._steps(spec, FLUX_CONFIG.num_inference_steps)
        guidance = FLUX_CONFIG.guidance_scale
        cap = spec.max_requests
        prompts = _load_parti_prompts(seed)

        # --- plan the workloads (mirrors _build_flux_{throughput,latency}_scenarios) ---
        thr_plan, lat_plan = [], []
        for w in self._selected(scenario.throughput_workloads, spec):
            p = spec_for(w).params
            n_img = p.batch_size * p.num_requests
            if cap is not None:
                n_img = max(1, min(n_img, cap))
            bs = min(p.batch_size, n_img)
            n_batches = max(1, n_img // bs)
            pool = (prompts * (bs * n_batches // len(prompts) + 1))[:bs * n_batches]
            thr_plan.append({"name": w.value, "height": p.height, "width": p.width, "bs": bs,
                             "batches": [pool[i * bs:(i + 1) * bs] for i in range(n_batches)]})
        for w in self._selected(scenario.latency_workloads, spec):
            p = spec_for(w).params
            warm, iters = p.num_warmup, p.num_iters
            if cap is not None:
                warm, iters = min(warm, max(1, cap)), max(1, min(iters, cap))
            lat_plan.append({"name": w.value, "height": p.height, "width": p.width,
                             "bs": p.batch_size, "prompts": prompts[:p.batch_size],
                             "num_warmup": warm, "num_iters": iters})

        # --- build the model (candidate classes, if any, are already patched in) ---
        t0 = time.time()
        engine = DiffusionEngine(model_name=scenario.hf_name, seed=seed,
                                 dtype=_torch_dtype(scenario.dtype),
                                 enforce_eager=spec.enforce_eager)
        pipeline = engine._get_pipeline()
        print(f"[flux] model loaded in {time.time() - t0:.1f}s; steps={steps} "
              f"guidance={guidance} seed={seed}", flush=True)

        def params(h, w, n_steps=steps):
            return DiffusionSamplingParams(height=h, width=w, num_inference_steps=n_steps,
                                           guidance_scale=guidance, seed=seed,
                                           output_type="latent")

        # --- warmup: one 2-step generation per distinct (h, w, batch), as the harness ---
        n_corr = max(0, int(spec.correctness_samples or 0))
        shapes = [(s["height"], s["width"], s["bs"]) for s in thr_plan + lat_plan]
        if n_corr:
            shapes.append((_CORRECTNESS_RES, _CORRECTNESS_RES, min(_CORRECTNESS_BATCH, n_corr)))
        for h, w, bs in dict.fromkeys(shapes):
            t0 = time.time()
            engine.generate([f"warmup {i}" for i in range(bs)], params(h, w, 2))
            torch.cuda.synchronize()
            print(f"[flux] warmup {h}x{w} bs={bs}: {time.time() - t0:.1f}s", flush=True)

        timings: dict[str, Timing] = {}

        # --- throughput ---
        for s in thr_plan:
            elapsed, n_img = 0.0, 0
            for batch in s["batches"]:
                gens = [torch.Generator(device="cuda").manual_seed(seed) for _ in batch]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                engine.generate(batch, params(s["height"], s["width"]), generator=gens)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - t0
                n_img += len(batch)
            timings[s["name"]] = {"kind": "throughput", "value": n_img / elapsed,
                                  "unit": "images/s"}
            print(f"[flux] {s['name']}: {n_img} images (bs={s['bs']}) in {elapsed:.2f}s -> "
                  f"{n_img / elapsed:.3f} images/s", flush=True)

        # --- latency ---
        for s in lat_plan:
            prm = params(s["height"], s["width"])
            for _ in range(s["num_warmup"]):
                torch.cuda.synchronize()
                engine.generate(s["prompts"], prm)
                torch.cuda.synchronize()
            lats = []
            for _ in range(s["num_iters"]):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                engine.generate(s["prompts"], prm)
                torch.cuda.synchronize()
                lats.append(time.perf_counter() - t0)
            med = statistics.median(lats)
            timings[s["name"]] = {"kind": "latency", "value": med, "unit": "s"}
            print(f"[flux] {s['name']}: median {med:.3f}s over {len(lats)} iters "
                  f"({[round(x, 3) for x in lats]})", flush=True)

        # --- correctness outputs ---
        corr_prompts = prompts[:n_corr]
        images, latents, finite = [], [], []
        t0 = time.time()
        prm = params(_CORRECTNESS_RES, _CORRECTNESS_RES)
        for b0 in range(0, n_corr, _CORRECTNESS_BATCH):
            batch = corr_prompts[b0:b0 + _CORRECTNESS_BATCH]
            gens = [torch.Generator(device="cuda").manual_seed(seed + b0 + i)
                    for i in range(len(batch))]
            out = engine.generate(batch, prm, generator=gens)
            with torch.inference_mode():
                img = self._decode(pipeline, out.latents, _CORRECTNESS_RES, _CORRECTNESS_RES)
            finite += torch.isfinite(img).flatten(1).all(dim=1).cpu().tolist()
            img = torch.nan_to_num(img, nan=0.0).clamp(0, 1)
            if _STORE_RES != _CORRECTNESS_RES:
                img = F.interpolate(img, size=(_STORE_RES, _STORE_RES), mode="area")
            images.append((img * 255).round().to(torch.uint8).cpu())
            latents.append(out.latents.to(torch.bfloat16).cpu())
        if n_corr:
            print(f"[flux] correctness: {n_corr} images in {time.time() - t0:.1f}s "
                  f"({sum(not f for f in finite)} non-finite)", flush=True)
        torch.save({
            "kind": "images",
            "model": scenario.hf_name,
            "prompts": corr_prompts,
            "seeds": [seed + i for i in range(n_corr)],
            "height": _CORRECTNESS_RES, "width": _CORRECTNESS_RES,
            "num_inference_steps": steps, "guidance_scale": guidance,
            "images": torch.cat(images) if images else torch.empty(0, 3, _STORE_RES, _STORE_RES,
                                                                    dtype=torch.uint8),
            "latents": torch.cat(latents) if latents else None,
            "finite": finite,
        }, os.path.join(spec.out_dir, "outputs.pt"))

        engine._cleanup()
        return timings

    # ------------------------------------------------------------------ compare

    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        import torch

        r_img, c_img = ref["images"], cand["images"]
        r_lat, c_lat = ref.get("latents"), cand.get("latents")
        r_ok = ref.get("finite") or [True] * len(r_img)
        c_ok = cand.get("finite") or [True] * len(c_img)
        r_pr, c_pr = ref.get("prompts") or [], cand.get("prompts") or []
        per_sample, cosines, mses, psnrs, lat_cos = [], [], [], [], []
        identical = bad = 0
        for i in range(len(r_img)):
            comparable = (i < len(c_img) and r_img[i].shape == c_img[i].shape and c_ok[i]
                          and r_ok[i] and (i >= len(r_pr) or i >= len(c_pr) or r_pr[i] == c_pr[i]))
            if not comparable:
                per_sample.append(1.0)
                bad += 1
                continue
            same = torch.equal(r_img[i], c_img[i])
            a = r_img[i].double().flatten() / 127.5 - 1.0
            b = c_img[i].double().flatten() / 127.5 - 1.0
            cos = 1.0 if same else float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))
            mse = float(((a - b) / 2).pow(2).mean())   # on the [0, 1] pixel scale
            per_sample.append(min(1.0, max(0.0, 1.0 - cos)))
            cosines.append(cos)
            mses.append(mse)
            psnrs.append(_PSNR_CAP if mse == 0 else min(_PSNR_CAP, -10 * math.log10(mse)))
            identical += int(same)
            if r_lat is not None and c_lat is not None and i < len(c_lat) \
                    and r_lat[i].shape == c_lat[i].shape:
                la, lb = r_lat[i].double().flatten(), c_lat[i].double().flatten()
                lc = float(torch.dot(la, lb) / (la.norm() * lb.norm()).clamp_min(1e-12))
                if math.isfinite(lc):
                    lat_cos.append(lc)

        def mean(xs):
            return sum(xs) / len(xs) if xs else None

        return {"per_sample": per_sample, "summary": {
            "n": len(per_sample),
            "mean_d": mean(per_sample),
            "max_d": max(per_sample) if per_sample else None,
            "mean_cosine": mean(cosines),
            "min_cosine": min(cosines) if cosines else None,
            "mean_mse": mean(mses),
            "mean_psnr_db": mean(psnrs),
            "min_psnr_db": min(psnrs) if psnrs else None,
            "frac_identical": identical / len(per_sample) if per_sample else None,
            "mean_latent_cosine": mean(lat_cos),
            "min_latent_cosine": min(lat_cos) if lat_cos else None,
            "n_invalid": bad,
        }}
