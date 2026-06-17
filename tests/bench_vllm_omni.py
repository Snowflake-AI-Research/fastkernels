#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano vs vllm-omni
for diffusion models (FLUX.1-dev, HunyuanVideo-1.5) and TTS models (CosyVoice3).

The benchmark mode is inferred from the model's category:
  - diffusion: FLUX.1-dev, HunyuanVideo-1.5
  - tts: CosyVoice3

Diffusion workloads compare throughput, latency, and correctness (latent/frame).
TTS workloads compare throughput (utterances/sec), latency, and mel spectrogram
cosine similarity.

Each engine runs in a subprocess to avoid import contamination.

Usage:
    # FLUX (default)
    python tests/bench_vllm_omni.py --model black-forest-labs/FLUX.1-dev
    python tests/bench_vllm_omni.py --skip-vllm-omni

    # HunyuanVideo
    python tests/bench_vllm_omni.py --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v

    # CosyVoice3
    python tests/bench_vllm_omni.py --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

from kb_nano.bench.eval.config import MODEL_CATEGORY
from kb_nano.bench.utils.worker import run_worker
from kb_nano.bench.utils.workloads import (
    COSYVOICE3_CONFIG,
    DIFFUSION_LATENCY_WORKLOADS,
    DIFFUSION_THROUGHPUT_WORKLOADS,
    FLUX_CONFIG,
    HUNYUAN_VIDEO_CONFIG,
    TTS_LATENCY_WORKLOADS,
    TTS_THROUGHPUT_WORKLOADS,
    VIDEO_DIFFUSION_LATENCY_WORKLOADS,
    VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS,
)

_SEED_TTS_EVAL_REPO = "zhaochenyang20/seed-tts-eval"


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


def _is_hunyuan_video(model: str) -> bool:
    lower = model.lower()
    return "hunyuanvideo" in lower or "hunyuan-video" in lower or "hunyuan_video" in lower


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PARTI_PROMPTS: list[str] | None = None
_MOVIE_GEN_PROMPTS: list[str] | None = None


def _load_parti_prompts(seed: int = 42) -> list[str]:
    """Load all prompts from nateraw/parti-prompts (P2), deterministically shuffled."""
    from datasets import load_dataset

    ds = load_dataset("nateraw/parti-prompts", split="train")
    prompts = [row["Prompt"] for row in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def _get_parti_prompts(seed: int = 42) -> list[str]:
    global _PARTI_PROMPTS
    if _PARTI_PROMPTS is None:
        _PARTI_PROMPTS = _load_parti_prompts(seed)
    return _PARTI_PROMPTS


def _load_movie_gen_prompts(seed: int = 42) -> list[str]:
    """Load all ~1003 prompts from Movie Gen Video Bench, deterministically shuffled."""
    from datasets import load_dataset

    ds = load_dataset(
        "meta-ai-for-media-research/movie_gen_video_bench_no_generations",
        split="test",
    )
    prompts = [row["prompt"] for row in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def _get_movie_gen_prompts(seed: int = 42) -> list[str]:
    global _MOVIE_GEN_PROMPTS
    if _MOVIE_GEN_PROMPTS is None:
        _MOVIE_GEN_PROMPTS = _load_movie_gen_prompts(seed)
    return _MOVIE_GEN_PROMPTS


def _load_seed_tts_eval(seed: int = 42, max_samples: int = 200) -> list[dict]:
    """Load text pairs from SEED-TTS-Eval for TTS benchmarking.

    Downloads only en/meta.lst (a few KB) via hf_hub_download rather than
    the full 1 GB+ repo, avoiding the many-small-files bottleneck.

    Each meta.lst line has the format:
        utterance_id | prompt_text | prompt_wav_path | target_text

    Returns list of dicts with keys: text, prompt_text, prompt_wav_rel,
    utterance_id.
    """
    from huggingface_hub import hf_hub_download

    meta_path = hf_hub_download(
        _SEED_TTS_EVAL_REPO, filename="en/meta.lst",
        repo_type="dataset",
    )

    samples = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            utt_id, prompt_text, prompt_wav_rel, target_text = (
                parts[0].strip(), parts[1].strip(),
                parts[2].strip(), parts[3].strip(),
            )
            samples.append({
                "text": target_text,
                "prompt_text": prompt_text,
                "prompt_wav_rel": prompt_wav_rel,
                "utterance_id": utt_id,
            })

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:max_samples]


_TTS_SAMPLES: list[dict] | None = None


def _get_tts_samples(seed: int = 42, max_samples: int = 200) -> list[dict]:
    """Return TTS samples (cached after first load)."""
    global _TTS_SAMPLES
    if _TTS_SAMPLES is None:
        _TTS_SAMPLES = _load_seed_tts_eval(seed, max_samples)
    return _TTS_SAMPLES


# ═══════════════════════════════════════════════════════════════════════════
# FLUX workers
# ═══════════════════════════════════════════════════════════════════════════

FLUX_VLLM_OMNI_WORKER = r'''
import asyncio, json, os, sys, time, torch
from tqdm import tqdm

def _patch_t5_load_weights_if_needed():
    """Patch vllm-omni <= 0.18.0 T5EncoderModel.load_weights bug."""
    try:
        import vllm_omni, inspect
        if getattr(vllm_omni, "__version__", "") > "0.18.0":
            return
        from vllm_omni.diffusion.models.t5_encoder.t5_encoder import T5EncoderModel
        src = inspect.getsource(T5EncoderModel.load_weights)
        if 'name.replace(f".{weight_name}."' in src:
            return
        from collections.abc import Iterable
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        def _patched_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            stacked_params_mapping = [
                ("qkv_proj", "q", "q"),
                ("qkv_proj", "k", "k"),
                ("qkv_proj", "v", "v"),
                ("wi", "wi_0", 0),
                ("wi", "wi_1", 1),
            ]
            params_dict = dict(self.named_parameters())
            loaded_params: set[str] = set()
            for name, loaded_weight in weights:
                original_name = name
                lookup_name = name
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if f".{weight_name}." not in name:
                        continue
                    lookup_name = name.replace(f".{weight_name}.", f".{param_name}.")
                    if lookup_name not in params_dict:
                        continue
                    param = params_dict[lookup_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                loaded_params.add(original_name)
                loaded_params.add(lookup_name)
            return loaded_params
        T5EncoderModel.load_weights = _patched_load_weights
        print("[bench] Patched vllm-omni T5EncoderModel.load_weights (v0.18.0 dotted-replace fix)", file=sys.stderr)
    except Exception as e:
        print(f"[bench] WARNING: failed to patch T5EncoderModel.load_weights: {e}", file=sys.stderr)

_patch_t5_load_weights_if_needed()

async def run_benchmark(cfg):
    from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    od_config = OmniDiffusionConfig(
        model=cfg["model"],
        dtype=torch.bfloat16,
        enforce_eager=True,
        output_type="latent",
    )
    engine = AsyncOmniDiffusion(model=cfg["model"], od_config=od_config)

    warmup_params = OmniDiffusionSamplingParams(
        height=256, width=256, num_inference_steps=2,
        guidance_scale=3.5,
    )
    warmup_params.seed = cfg["seed"]
    warmup_params.guidance_scale_provided = True
    await engine.generate_batch(["warmup"], warmup_params)

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    all_results = []
    for scenario in cfg["scenarios"]:
        batches = scenario.get("batches", [scenario.get("prompts", [])])
        if not isinstance(batches[0], list):
            batches = [batches]
        params = OmniDiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 3.5),
        )
        params.seed = cfg["seed"]
        params.guidance_scale_provided = True

        total_elapsed = 0.0
        total_images = 0
        desc = f"vllm-omni {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_idx, batch_prompts in enumerate(pbar):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = await engine.generate_batch(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)

            if latent_dir and output is not None:
                latent_tensor = None
                if hasattr(output, "latents") and output.latents is not None:
                    latent_tensor = output.latents
                elif hasattr(output, "images") and output.images:
                    for img in output.images:
                        if isinstance(img, torch.Tensor):
                            latent_tensor = img
                            break
                if latent_tensor is not None:
                    torch.save(
                        latent_tensor.cpu(),
                        os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                    )

            pbar.set_postfix(imgs=total_images, ips=f"{total_images / total_elapsed:.2f}")

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_images": total_images,
            "images_per_second": total_images / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        params = OmniDiffusionSamplingParams(
            height=ls["height"], width=ls["width"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 3.5),
        )
        params.seed = cfg["seed"]
        params.guidance_scale_provided = True
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"vllm-omni latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            await engine.generate_batch(prompts, params)
            torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"vllm-omni latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            await engine.generate_batch(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    engine.close()
    torch.cuda.empty_cache()
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    asyncio.run(run_benchmark(cfg))

if __name__ == "__main__":
    main()
'''


FLUX_KB_NANO_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    pkg = cfg["package_name"]
    sys.path.insert(0, cfg["project_root"])

    if cfg.get("pytorch_reference", False):
        from kb_nano.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    try:
        eng_mod = __import__(f"{pkg}.infra.diffusion_engine", fromlist=["DiffusionEngine"])
    except (ImportError, ModuleNotFoundError):
        sys.path.insert(0, cfg["project_root"])
        eng_mod = __import__(f"{pkg}.infra.diffusion_engine", fromlist=["DiffusionEngine"])
    flux_mod = __import__(f"{pkg}.tasks.baseline.L4.flux", fromlist=["DiffusionSamplingParams"])
    DiffusionEngine = eng_mod.DiffusionEngine
    DiffusionSamplingParams = flux_mod.DiffusionSamplingParams

    engine = DiffusionEngine(
        model_name=cfg["model"], seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
    )

    seen = set()
    for s in cfg["scenarios"] + cfg.get("latency_scenarios", []):
        bs = s.get("batch_size", len(s.get("prompts", ["w"])))
        key = (s["height"], s["width"], bs)
        if key not in seen:
            seen.add(key)
            wp = DiffusionSamplingParams(
                height=s["height"], width=s["width"],
                num_inference_steps=2, seed=cfg["seed"], output_type="latent",
            )
            warmup_prompts = [f"warmup {i}" for i in range(bs)]
            print(f"Warming up: {s['height']}x{s['width']} batch_size={bs}", file=sys.stderr, flush=True)
            engine.generate(warmup_prompts, wp)
            torch.cuda.synchronize()

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    pipeline = engine._get_pipeline()

    all_results = []
    for scenario in cfg["scenarios"]:
        batches = scenario.get("batches", [scenario.get("prompts", [])])
        if not isinstance(batches[0], list):
            batches = [batches]
        params = DiffusionSamplingParams(
            height=scenario["height"], width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 3.5),
            seed=cfg["seed"], output_type="latent",
        )
        total_elapsed = 0.0
        total_images = 0
        desc = f"kb-nano {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_idx, batch_prompts in enumerate(pbar):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = engine.generate(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)
            if latent_dir and output.latents is not None:
                packed = output.latents
                unpacked = pipeline._unpack_latents(
                    packed, scenario["height"], scenario["width"],
                    pipeline.vae_scale_factor,
                )
                decoded = (unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                decoded = decoded.to(dtype=pipeline.vae.dtype)
                decoded = pipeline.vae.decode(decoded, return_dict=False)[0]
                torch.save(
                    decoded.cpu(),
                    os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                )
            pbar.set_postfix(imgs=total_images, ips=f"{total_images / total_elapsed:.2f}")

        all_results.append({
            "name": scenario["name"], "elapsed": total_elapsed,
            "num_images": total_images, "images_per_second": total_images / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        params = DiffusionSamplingParams(
            height=ls["height"], width=ls["width"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 3.5),
            seed=cfg["seed"], output_type="latent",
        )
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"kb-nano latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize(); engine.generate(prompts, params); torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    engine._cleanup()
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════
# HunyuanVideo workers
# ═══════════════════════════════════════════════════════════════════════════

HUNYUAN_KB_NANO_WORKER = r'''
import json, os, sys, time, torch
import numpy as np
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])

    if cfg.get("pytorch_reference", False):
        from kb_nano.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    try:
        from kb_nano.infra.diffusion_engine import DiffusionEngine
        from kb_nano.tasks.baseline.L4.hunyuan_video import (
            HunyuanVideoDiffusionSamplingParams,
        )
    except (ImportError, ModuleNotFoundError):
        sys.path.insert(0, cfg["project_root"])
        from kb_nano.infra.diffusion_engine import DiffusionEngine
        from kb_nano.tasks.baseline.L4.hunyuan_video import (
            HunyuanVideoDiffusionSamplingParams,
        )

    engine = DiffusionEngine(
        model_name=cfg["model"], seed=cfg["seed"], enforce_eager=True,
    )

    frames_dir = cfg.get("frames_dir")
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    seen = set()
    for s in cfg["scenarios"] + cfg.get("latency_scenarios", []):
        key = (s["height"], s["width"], s["num_frames"])
        if key not in seen:
            seen.add(key)
            wp = HunyuanVideoDiffusionSamplingParams(
                height=s["height"], width=s["width"],
                num_frames=s["num_frames"], num_inference_steps=2,
                guidance_scale=s.get("guidance_scale", 6.0),
                seed=cfg["seed"], output_type="latent",
            )
            print(f"Warming up: {s['height']}x{s['width']} {s['num_frames']}f",
                  file=sys.stderr, flush=True)
            engine.generate(["warmup"], wp)
            torch.cuda.synchronize()

    save_frames = frames_dir is not None
    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        output_type = "pil" if save_frames else "latent"
        params = HunyuanVideoDiffusionSamplingParams(
            height=scenario["height"], width=scenario["width"],
            num_frames=scenario["num_frames"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 6.0),
            seed=cfg["seed"], output_type=output_type,
        )
        total_elapsed = 0.0
        total_videos = 0
        desc = f"kb-nano {scenario['name']}"
        pbar = tqdm(enumerate(prompts), total=len(prompts), desc=desc, unit="vid", file=sys.stderr)
        for pi, prompt in pbar:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = engine.generate([prompt], params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            total_videos += 1
            if save_frames and output.video is not None:
                frames_list = output.video
                if isinstance(frames_list, list) and frames_list:
                    arr = np.stack([np.array(f.convert("RGB")) for f in frames_list], axis=0)
                    np.save(os.path.join(frames_dir, f"{scenario['name']}_prompt{pi:04d}.npy"), arr)
            pbar.set_postfix(vids=total_videos, vps=f"{total_videos / total_elapsed:.3f}")
        all_results.append({
            "name": scenario["name"], "elapsed": total_elapsed,
            "num_videos": total_videos, "videos_per_second": total_videos / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompt = ls["prompt"]
        params = HunyuanVideoDiffusionSamplingParams(
            height=ls["height"], width=ls["width"],
            num_frames=ls["num_frames"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 6.0),
            seed=cfg["seed"], output_type="latent",
        )
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"kb-nano warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize(); engine.generate([prompt], params); torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate([prompt], params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_frames": ls["num_frames"], "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    engine._cleanup()
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


HUNYUAN_VLLM_OMNI_WORKER = r'''
import json, os, sys, time, torch
import numpy as np
from tqdm import tqdm

def _pil_images_to_array(images):
    return np.stack([np.array(img.convert("RGB")) for img in images], axis=0)

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    seed = cfg["seed"]
    print("[vllm-omni] Creating Omni engine...", file=sys.stderr, flush=True)
    omni = Omni(model=cfg["model"], enforce_eager=True)
    print("[vllm-omni] Engine ready", file=sys.stderr, flush=True)

    frames_dir = cfg.get("frames_dir")
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    def _make_params(s):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return OmniDiffusionSamplingParams(
            height=s["height"], width=s["width"],
            num_frames=s["num_frames"],
            num_inference_steps=s["num_inference_steps"],
            guidance_scale=s.get("guidance_scale", 6.0),
            generator=generator,
        )

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        total_elapsed = 0.0
        total_videos = 0
        desc = f"vllm-omni {scenario['name']}"
        pbar = tqdm(enumerate(prompts), total=len(prompts), desc=desc, unit="vid", file=sys.stderr)
        for pi, prompt in pbar:
            params = _make_params(scenario)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = omni.generate({"prompt": prompt}, params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            total_videos += 1
            if frames_dir:
                items = output if isinstance(output, list) else [output]
                for item in items:
                    if hasattr(item, "images") and item.images:
                        arr = _pil_images_to_array(item.images)
                        np.save(os.path.join(frames_dir, f"{scenario['name']}_prompt{pi:04d}.npy"), arr)
            pbar.set_postfix(vids=total_videos, vps=f"{total_videos / total_elapsed:.3f}")
        all_results.append({
            "name": scenario["name"], "elapsed": total_elapsed,
            "num_videos": total_videos, "videos_per_second": total_videos / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompt = ls["prompt"]
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"vllm-omni warmup {ls['name']}", file=sys.stderr):
            params = _make_params(ls)
            torch.cuda.synchronize(); omni.generate({"prompt": prompt}, params); torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"vllm-omni latency {ls['name']}", file=sys.stderr):
            params = _make_params(ls)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            omni.generate({"prompt": prompt}, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_frames": ls["num_frames"], "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════
# TTS workers (CosyVoice3)
# ═══════════════════════════════════════════════════════════════════════════

VLLM_OMNI_TTS_WORKER = r'''
import asyncio, json, os, sys, time, torch
import numpy as np
from tqdm import tqdm

def _find_stage_config():
    """Return a patched cosyvoice3.yaml with enforce_eager=true.

    CUDA graph capture for the multi-stage CosyVoice3 pipeline can
    take 10+ minutes on first run, easily exceeding the default
    stage_init_timeout.  We copy the shipped YAML and force eager
    mode so startup is fast and predictable.
    """
    import tempfile
    import vllm_omni
    import yaml

    pkg_dir = os.path.dirname(vllm_omni.__file__)
    src = os.path.join(pkg_dir, "model_executor", "stage_configs", "cosyvoice3.yaml")
    if not os.path.exists(src):
        raise FileNotFoundError(f"cosyvoice3.yaml not found at {src}")

    with open(src) as f:
        cfg = yaml.safe_load(f)

    for stage in cfg.get("stage_args", []):
        ea = stage.get("engine_args", {})
        ea["enforce_eager"] = True

    patched = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="cosyvoice3_eager_",
        delete=False,
    )
    yaml.dump(cfg, patched, default_flow_style=False)
    patched.close()
    print(f"  Using eager-mode stage config: {patched.name}", file=sys.stderr)
    return patched.name

def _download_and_get_tokenizer_path(model_name):
    """Download model and return path to CosyVoice-BlankEN tokenizer dir."""
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(model_name)
    tok_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    if os.path.isdir(tok_dir):
        return model_dir, tok_dir
    raise FileNotFoundError(f"Tokenizer dir not found: {tok_dir}")

def _ensure_config_has_model_type(model_dir, model_type="cosyvoice3"):
    """Patch config.json to include model_type if missing.

    The HF repo ships an empty config.json (``{}``), which causes
    ``AutoConfig.from_pretrained`` to fail with *Unrecognized model*.
    vllm-omni's ``OmniEngineArgs`` injects ``architectures`` via
    ``hf_overrides`` but does not inject ``model_type``.  Writing it
    into the cached config.json is the simplest workaround.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path) as f:
        data = json.load(f)
    if data.get("model_type") == model_type:
        return
    data["model_type"] = model_type
    with open(cfg_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Patched {cfg_path}: added model_type={model_type!r}", file=sys.stderr)

def _ensure_mel_filters_asset():
    """Download mel_filters.npz for vllm-omni's CosyVoice3 audio processor."""
    import urllib.request
    import vllm_omni
    pkg_dir = os.path.dirname(vllm_omni.__file__)
    assets_dir = os.path.join(pkg_dir, "model_executor", "models", "cosyvoice3", "assets")
    filters_path = os.path.join(assets_dir, "mel_filters.npz")
    if os.path.exists(filters_path):
        return
    os.makedirs(assets_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
    print(f"  Downloading mel_filters.npz from {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, filters_path)
    print(f"  Saved to {filters_path}", file=sys.stderr)

def _make_prompt(text, prompt_text, ref_audio, ref_sr):
    """Build an Omni prompt dict for CosyVoice3."""
    return {
        "prompt": text,
        "multi_modal_data": {
            "audio": (ref_audio, ref_sr),
        },
        "mm_processor_kwargs": {
            "prompt_text": prompt_text,
            "sample_rate": ref_sr,
        },
    }

def _build_sampling_params(config):
    from vllm import SamplingParams
    from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
    cv_cfg = CosyVoice3Config()
    greedy = config.get("greedy", False)
    temp = 0.0 if greedy else 1.0
    seed_val = config.get("seed", None) if greedy else None
    gpt_sampling = SamplingParams(
        temperature=temp,
        top_p=1.0 if greedy else 0.8,
        top_k=-1 if greedy else 25,
        repetition_penalty=2.0,
        min_tokens=10,
        max_tokens=2048,
        stop_token_ids=[6561 + 1],
        detokenize=False,
        seed=seed_val,
    )
    s2mel_sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        repetition_penalty=2.0, max_tokens=256, detokenize=False,
    )
    return [gpt_sampling, s2mel_sampling]

def _extract_audio(output, sample_rate):
    """Extract audio tensor and duration from an OmniRequestOutput."""
    ro = output.request_output if hasattr(output, "request_output") else output
    mm = getattr(ro, "multimodal_output", None)
    if not mm and hasattr(ro, "outputs") and ro.outputs:
        mm = getattr(ro.outputs[0], "multimodal_output", None)
    if mm and "audio" in mm:
        audio = mm["audio"]
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            return audio.shape[-1] / sample_rate, audio
    return 0.0, None

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    stage_config = _find_stage_config()
    model_dir, tok_dir = _download_and_get_tokenizer_path(cfg["model"])
    _ensure_config_has_model_type(model_dir)
    _ensure_mel_filters_asset()

    from vllm_omni.entrypoints.omni import Omni
    omni = Omni(
        model=model_dir,
        stage_configs_path=stage_config,
        trust_remote_code=True,
        tokenizer=tok_dir,
        stage_init_timeout=600,
    )
    sampling_params_list = _build_sampling_params(cfg)

    ref_audio = cfg.get("ref_audio")
    if ref_audio is None:
        ref_audio = np.random.randn(24000 * 3).astype(np.float32)
    else:
        ref_audio = np.array(ref_audio, dtype=np.float32)
    ref_sr = cfg.get("ref_sr", 24000)
    prompt_text = cfg.get("prompt_text", "Testing my voice.")
    sample_rate = cfg.get("sample_rate", 24000)

    audio_dir = cfg.get("audio_dir")
    if audio_dir:
        os.makedirs(audio_dir, exist_ok=True)

    # Warmup
    warmup_prompt = _make_prompt("Warmup.", prompt_text, ref_audio, ref_sr)
    list(omni.generate(warmup_prompt, sampling_params_list, use_tqdm=False))
    print("  vllm-omni warmup done", file=sys.stderr)

    all_results = []
    for scenario in cfg["scenarios"]:
        texts = scenario["texts"]
        total_elapsed = 0.0
        total_utterances = 0
        total_audio_duration = 0.0

        desc = f"vllm-omni TTS {scenario['name']}"
        pbar = tqdm(texts, desc=desc, unit="utt", file=sys.stderr)
        for utt_idx, text in enumerate(pbar):
            prompt = _make_prompt(text, prompt_text, ref_audio, ref_sr)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = list(omni.generate(prompt, sampling_params_list, use_tqdm=False))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            total_utterances += 1

            for out in outputs:
                dur, audio_tensor = _extract_audio(out, sample_rate)
                total_audio_duration += dur
                if audio_dir and audio_tensor is not None:
                    torch.save(
                        audio_tensor.cpu(),
                        os.path.join(audio_dir, f"{scenario['name']}_utt{utt_idx:04d}.pt"),
                    )

            pbar.set_postfix(
                utts=total_utterances,
                ups=f"{total_utterances / total_elapsed:.2f}",
            )

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_utterances": total_utterances,
            "utterances_per_second": total_utterances / total_elapsed if total_elapsed > 0 else 0,
            "total_audio_duration_s": total_audio_duration,
            "rtf": total_elapsed / total_audio_duration if total_audio_duration > 0 else float("inf"),
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        text = ls["texts"][0] if ls["texts"] else "Test utterance."
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        prompt = _make_prompt(text, prompt_text, ref_audio, ref_sr)

        for _ in tqdm(range(num_warmup), desc=f"vllm-omni latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            list(omni.generate(prompt, sampling_params_list, use_tqdm=False))
            torch.cuda.synchronize()

        latencies = []
        for _ in tqdm(range(num_iters), desc=f"vllm-omni latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            list(omni.generate(prompt, sampling_params_list, use_tqdm=False))
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    omni.close()
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


KB_NANO_TTS_WORKER = r'''
import json, os, sys, time, torch
import numpy as np
from tqdm import tqdm
from functools import partial

def _init_preprocessing(model_dir, config):
    """Initialise tokenizer, speech tokenizer, speaker embedder and mel extractor."""
    import onnxruntime
    from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
    from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    tokenizer = get_qwen_tokenizer(
        token_path=os.path.join(model_dir, config.qwen_pretrain_path),
        skip_special_tokens=config.skip_special_tokens,
        version=config.version,
    )
    speech_tokenizer = onnxruntime.InferenceSession(
        os.path.join(model_dir, config.speech_tokenizer_path),
        sess_options=option,
        providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"],
    )
    campplus = onnxruntime.InferenceSession(
        os.path.join(model_dir, config.campplus_onxx_path),
        sess_options=option,
        providers=["CPUExecutionProvider"],
    )
    feat_extractor = partial(mel_spectrogram, **getattr(config, "feat_extractor", {}))
    return tokenizer, speech_tokenizer, campplus, feat_extractor


def _preprocess(text, prompt_text, ref_audio, ref_sr, tokenizer, speech_tok, campplus, feat_ext, config, device):
    """Run the same preprocessing as vllm-omni to produce model inputs."""
    from vllm_omni.model_executor.models.cosyvoice3.utils import (
        extract_text_token, extract_speech_token, extract_speech_feat,
        extract_spk_embedding,
    )
    text_token, _ = extract_text_token(text, tokenizer, config.allowed_special)
    prompt_text_token, _ = extract_text_token(prompt_text, tokenizer, config.allowed_special)
    speech_token, _ = extract_speech_token((ref_audio, ref_sr), speech_tok, device)
    speech_feat, speech_feat_len = extract_speech_feat((ref_audio, ref_sr), feat_ext, device)

    if config.sample_rate == 24000:
        tok_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat = speech_feat[:, :2 * tok_len]
        speech_token = speech_token[:, :tok_len]

    spk_embedding = extract_spk_embedding((ref_audio, ref_sr), campplus, device)

    return {
        "text_token": text_token.to(device),
        "prompt_text_token": prompt_text_token.to(device),
        "speech_token": speech_token.to(device),
        "speech_feat": speech_feat.to(device),
        "spk_embedding": spk_embedding.to(device),
    }


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    pkg = cfg["package_name"]
    sys.path.insert(0, cfg["project_root"])

    if cfg.get("pytorch_reference", False):
        from kb_nano.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    try:
        cosyvoice_mod = __import__(
            f"{pkg}.tasks.baseline.L4.cosyvoice3",
            fromlist=["CosyVoice3Config", "CosyVoice3ForTTS"],
        )
    except ImportError:
        sys.path.insert(0, cfg["project_root"])
        cosyvoice_mod = __import__(
            f"{pkg}.tasks.baseline.L4.cosyvoice3",
            fromlist=["CosyVoice3Config", "CosyVoice3ForTTS"],
        )
    CosyVoice3Config = cosyvoice_mod.CosyVoice3Config
    CosyVoice3ForTTS = cosyvoice_mod.CosyVoice3ForTTS

    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(cfg["model"])

    config = CosyVoice3Config.from_pretrained(cfg["model"])
    device = torch.device("cuda")

    model = CosyVoice3ForTTS(config, model_stage="e2e")
    model.load_weights_e2e(model_dir, device)
    model.eval()
    print(f"  Loaded CosyVoice3 e2e model (talker=f32, code2wav=f32): {cfg['model']}", file=sys.stderr)

    tokenizer, speech_tok, campplus, feat_ext = _init_preprocessing(model_dir, config)

    ref_audio = cfg.get("ref_audio")
    if ref_audio is None:
        ref_audio = np.random.randn(24000 * 3).astype(np.float32)
    else:
        ref_audio = np.array(ref_audio, dtype=np.float32)
    ref_sr = cfg.get("ref_sr", 24000)
    prompt_text = cfg.get("prompt_text", "Testing my voice.")
    sample_rate = cfg.get("sample_rate", 24000)
    n_timesteps = cfg.get("n_timesteps", 10)
    greedy = cfg.get("greedy", False)
    gen_temperature = 0 if greedy else 1.0
    cfm_seed = 12345 if greedy else None

    audio_dir = cfg.get("audio_dir")
    if audio_dir:
        os.makedirs(audio_dir, exist_ok=True)

    all_results = []
    for scenario in cfg["scenarios"]:
        texts = scenario["texts"]
        total_elapsed = 0.0
        total_utterances = 0
        total_audio_duration = 0.0

        desc = f"kb-nano TTS {scenario['name']}"
        pbar = tqdm(texts, desc=desc, unit="utt", file=sys.stderr)
        for utt_idx, text in enumerate(pbar):
            inputs = _preprocess(
                text, prompt_text, ref_audio, ref_sr,
                tokenizer, speech_tok, campplus, feat_ext, config, "cpu",
            )

            torch.cuda.synchronize()
            start = time.perf_counter()
            audio, _ = model.generate(
                text_token=inputs["text_token"],
                prompt_text_token=inputs["prompt_text_token"],
                speech_token=inputs["speech_token"],
                speech_feat=inputs["speech_feat"],
                spk_embedding=inputs["spk_embedding"],
                n_timesteps=n_timesteps,
                temperature=gen_temperature,
                cfm_seed=cfm_seed,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            total_elapsed += elapsed
            total_utterances += 1

            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                audio_dur = audio.shape[-1] / sample_rate
                total_audio_duration += audio_dur
                if audio_dir:
                    torch.save(
                        audio.cpu(),
                        os.path.join(audio_dir, f"{scenario['name']}_utt{utt_idx:04d}.pt"),
                    )

            pbar.set_postfix(
                utts=total_utterances,
                ups=f"{total_utterances / total_elapsed:.2f}",
            )

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_utterances": total_utterances,
            "utterances_per_second": total_utterances / total_elapsed if total_elapsed > 0 else 0,
            "total_audio_duration_s": total_audio_duration,
            "rtf": total_elapsed / total_audio_duration if total_audio_duration > 0 else float("inf"),
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        text = ls["texts"][0] if ls["texts"] else "Test utterance."
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)

        inputs = _preprocess(
            text, prompt_text, ref_audio, ref_sr,
            tokenizer, speech_tok, campplus, feat_ext, config, "cpu",
        )

        for _ in tqdm(range(num_warmup), desc=f"kb-nano latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            model.generate(
                text_token=inputs["text_token"],
                prompt_text_token=inputs["prompt_text_token"],
                speech_token=inputs["speech_token"],
                speech_feat=inputs["speech_feat"],
                spk_embedding=inputs["spk_embedding"],
                n_timesteps=n_timesteps,
            )
            torch.cuda.synchronize()

        latencies = []
        for _ in tqdm(range(num_iters), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(
                text_token=inputs["text_token"],
                prompt_text_token=inputs["prompt_text_token"],
                speech_token=inputs["speech_token"],
                speech_feat=inputs["speech_feat"],
                spk_embedding=inputs["spk_embedding"],
                n_timesteps=n_timesteps,
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    # ── Code2Wav equivalence test ──────────────────────────────────────────
    c2w_check = {}
    try:
        import torch.nn.functional as F
        cosyvoice_c2w = cosyvoice_mod.CosyVoice3Code2Wav
        make_pad_mask = cosyvoice_mod.make_pad_mask

        from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import (
            CosyVoice3Code2Wav as VCode2Wav,
        )

        test_text = cfg["scenarios"][0]["texts"][0] if cfg["scenarios"] else "Hello."
        test_inputs = _preprocess(
            test_text, prompt_text, ref_audio, ref_sr,
            tokenizer, speech_tok, campplus, feat_ext, config, "cpu",
        )
        _, test_tokens = model.generate(
            text_token=test_inputs["text_token"],
            prompt_text_token=test_inputs["prompt_text_token"],
            speech_token=test_inputs["speech_token"],
            speech_feat=test_inputs["speech_feat"],
            spk_embedding=test_inputs["spk_embedding"],
            n_timesteps=n_timesteps, temperature=0, cfm_seed=12345,
        )

        del model
        torch.cuda.empty_cache()

        kb_c2w = cosyvoice_c2w(config)
        kb_c2w = kb_c2w.to(device=device, dtype=torch.bfloat16)
        kb_c2w.load_weights(model_dir, device)
        kb_c2w.eval()

        class _Cfg:
            pass
        vc = _Cfg()
        for k, v in vars(config).items():
            setattr(vc, k, v)
        vl_c2w = VCode2Wav(vc)
        vl_c2w.load_weights(model_dir, device)
        vl_c2w = vl_c2w.to(device=device, dtype=torch.bfloat16)
        vl_c2w.eval()

        gen_token = torch.tensor([test_tokens], device=device)
        prompt_token = test_inputs["speech_token"][:1].to(device)
        prompt_feat = test_inputs["speech_feat"][:1].to(device=device, dtype=torch.bfloat16)
        embedding = test_inputs["spk_embedding"][:1].to(device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            emb = F.normalize(embedding, dim=1)

            emb_kb = kb_c2w.flow_model.spk_embed_affine_layer(emb.clone())
            full_token = torch.cat([prompt_token, gen_token], dim=1)
            tl = torch.tensor([full_token.shape[1]], device=device, dtype=torch.int32)
            mask = (~make_pad_mask(tl)).unsqueeze(-1).to(emb_kb)
            tok_emb_kb = kb_c2w.flow_model.input_embedding(torch.clamp(full_token, min=0)) * mask
            h_kb = kb_c2w.flow_model.pre_lookahead_layer(tok_emb_kb).repeat_interleave(2, dim=1)

            emb_vl = vl_c2w.flow_model.spk_embed_affine_layer(emb.clone())
            tok_emb_vl = vl_c2w.flow_model.input_embedding(torch.clamp(full_token, min=0)) * mask
            h_vl = vl_c2w.flow_model.pre_lookahead_layer(tok_emb_vl).repeat_interleave(2, dim=1)

            mel_len1 = prompt_feat.shape[1]
            mel_len2 = h_kb.shape[1] - mel_len1
            conds = torch.zeros([1, mel_len1 + mel_len2, 80], device=device, dtype=torch.bfloat16)
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)
            mel_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h_kb)

            feat_kb, _ = kb_c2w.flow_model.decoder(
                mu=h_kb.transpose(1, 2).contiguous(), mask=mel_mask.unsqueeze(1),
                spks=emb_kb, cond=conds, n_timesteps=10, cfm_seed=12345,
            )

            feat_vl, _ = vl_c2w.flow_model.decoder(
                mu=h_vl.transpose(1, 2).contiguous(), mask=mel_mask.unsqueeze(1),
                spks=emb_vl, cond=conds, n_timesteps=10, cfm_seed=12345,
            )

            feat_kb_gen = feat_kb[:, :, mel_len1:].float()
            feat_vl_gen = feat_vl[:, :, mel_len1:].float()
            cos_mel = torch.nn.functional.cosine_similarity(
                feat_kb_gen.flatten().unsqueeze(0),
                feat_vl_gen.flatten().unsqueeze(0),
            ).item()
            mse_mel = ((feat_kb_gen - feat_vl_gen) ** 2).mean().item()

        c2w_check = {
            "mel_cosine_similarity": cos_mel,
            "mel_mse": mse_mel,
            "num_tokens": len(test_tokens),
            "pass": cos_mel > 0.99,
        }
        print(f"  Code2Wav equivalence: mel cos={cos_mel:.6f}, mse={mse_mel:.2e}, "
              f"{'PASS' if cos_mel > 0.99 else 'FAIL'}", file=sys.stderr)

        del kb_c2w, vl_c2w
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Code2Wav equivalence check failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        c2w_check = {"error": str(e)}
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "throughput": all_results,
            "latency": latency_results,
            "code2wav_equivalence": c2w_check,
        }, f)

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════
# Scenario builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_flux_throughput_scenarios(
    prompts: list[str], batch_size_override: int | None = None,
) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_THROUGHPUT_WORKLOADS:
        bs = batch_size_override or w.batch_size
        num_requests = w.num_requests
        total_needed = bs * num_requests
        pool = (prompts * ((total_needed // len(prompts)) + 1))[:total_needed]
        batches = [pool[i * bs : (i + 1) * bs] for i in range(num_requests)]
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": FLUX_CONFIG.num_inference_steps,
            "guidance_scale": FLUX_CONFIG.guidance_scale,
            "batches": batches,
            "batch_size": bs,
            "num_requests": num_requests,
        })
    return scenarios


def _build_flux_latency_scenarios(prompts: list[str]) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": FLUX_CONFIG.num_inference_steps,
            "guidance_scale": FLUX_CONFIG.guidance_scale,
            "prompts": prompts[:w.batch_size],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


def _build_hunyuan_throughput_scenarios(
    prompts: list[str],
) -> list[dict]:
    steps = HUNYUAN_VIDEO_CONFIG.num_inference_steps
    guidance = HUNYUAN_VIDEO_CONFIG.guidance_scale
    scenarios = []
    for w in VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS:
        n = min(w.num_prompts, len(prompts))
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_frames": w.num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompts": prompts[:n],
        })
    return scenarios


def _build_hunyuan_latency_scenarios(
    prompts: list[str],
) -> list[dict]:
    steps = HUNYUAN_VIDEO_CONFIG.num_inference_steps
    guidance = HUNYUAN_VIDEO_CONFIG.guidance_scale
    scenarios = []
    for w in VIDEO_DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_frames": w.num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompt": prompts[0],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


def _build_tts_throughput_scenarios(samples: list[dict]) -> list[dict]:
    scenarios = []
    for w in TTS_THROUGHPUT_WORKLOADS:
        texts = [s["text"] for s in samples if len(s["text"]) <= w.max_text_len]
        texts = texts[:w.num_requests]
        if not texts:
            texts = [s["text"][:w.max_text_len] for s in samples[:w.num_requests]]
        scenarios.append({
            "name": w.name,
            "texts": texts,
            "num_requests": len(texts),
            "max_text_len": w.max_text_len,
        })
    return scenarios


def _build_tts_latency_scenarios(samples: list[dict]) -> list[dict]:
    scenarios = []
    for w in TTS_LATENCY_WORKLOADS:
        texts = [s["text"] for s in samples if len(s["text"]) <= w.max_text_len]
        texts = texts[:w.batch_size]
        if not texts:
            texts = [samples[0]["text"][:w.max_text_len]] if samples else ["Test."]
        scenarios.append({
            "name": w.name,
            "texts": texts,
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Correctness comparison
# ═══════════════════════════════════════════════════════════════════════════

def _compare_latents(kb_latent_dir: str, vllm_latent_dir: str) -> dict:
    """Compare per-batch output tensors (FLUX latent space)."""
    import torch

    kb_files = sorted(
        f for f in os.listdir(kb_latent_dir) if f.endswith(".pt")
    ) if os.path.isdir(kb_latent_dir) else []
    vllm_files = sorted(
        f for f in os.listdir(vllm_latent_dir) if f.endswith(".pt")
    ) if os.path.isdir(vllm_latent_dir) else []

    common = sorted(set(kb_files) & set(vllm_files))
    if not common:
        return {}

    scenario_stats: dict[str, list[dict]] = defaultdict(list)
    for fname in common:
        kb_lat = torch.load(
            os.path.join(kb_latent_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()
        vllm_lat = torch.load(
            os.path.join(vllm_latent_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()
        if len(kb_lat) != len(vllm_lat):
            print(f"  WARNING: shape mismatch for {fname}, skipping", file=sys.stderr)
            continue
        kb_v, vllm_v = kb_lat.numpy(), vllm_lat.numpy()
        mse = float(np.mean((kb_v - vllm_v) ** 2))
        cos_sim = float(np.dot(kb_v, vllm_v) / (np.linalg.norm(kb_v) * np.linalg.norm(vllm_v) + 1e-12))
        scenario_name = fname.rsplit("_batch", 1)[0]
        scenario_stats[scenario_name].append({"file": fname, "mse": mse, "cosine_similarity": cos_sim})

    results = {}
    for scenario, batches in scenario_stats.items():
        mses = [b["mse"] for b in batches]
        cosines = [b["cosine_similarity"] for b in batches]
        results[scenario] = {
            "num_batches": len(batches),
            "mean_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
            "mean_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
        }
    return results


def _compare_frames(kb_frames_dir: str, vllm_frames_dir: str) -> dict:
    """Compare per-prompt decoded video frames (HunyuanVideo)."""
    kb_files = sorted(
        f for f in os.listdir(kb_frames_dir) if f.endswith(".npy")
    ) if os.path.isdir(kb_frames_dir) else []
    vllm_files = sorted(
        f for f in os.listdir(vllm_frames_dir) if f.endswith(".npy")
    ) if os.path.isdir(vllm_frames_dir) else []

    common = sorted(set(kb_files) & set(vllm_files))
    if not common:
        return {}

    scenario_stats: dict[str, list[dict]] = defaultdict(list)
    for fname in common:
        kb_arr = np.load(os.path.join(kb_frames_dir, fname)).astype(np.float32).flatten()
        vllm_arr = np.load(os.path.join(vllm_frames_dir, fname)).astype(np.float32).flatten()
        if len(kb_arr) != len(vllm_arr):
            print(f"  WARNING: shape mismatch for {fname}, skipping", file=sys.stderr)
            continue
        mse = float(np.mean((kb_arr - vllm_arr) ** 2))
        psnr = float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))
        cos_sim = float(np.dot(kb_arr, vllm_arr) / (np.linalg.norm(kb_arr) * np.linalg.norm(vllm_arr) + 1e-12))
        scenario_name = fname.rsplit("_prompt", 1)[0]
        scenario_stats[scenario_name].append({
            "file": fname, "mse": mse, "psnr": psnr, "cosine_similarity": cos_sim,
        })

    results = {}
    for scenario, items in scenario_stats.items():
        mses = [b["mse"] for b in items]
        psnrs = [b["psnr"] for b in items]
        cosines = [b["cosine_similarity"] for b in items]
        results[scenario] = {
            "num_prompts": len(items),
            "mean_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
            "mean_psnr": float(np.mean(psnrs)),
            "min_psnr": float(np.min(psnrs)),
            "mean_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Printing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _print_throughput_comparison(
    kb_results: list[dict], vllm_results: list[dict] | None, *, unit: str = "images",
):
    rate_key = "images_per_second" if unit == "images" else "videos_per_second"
    count_key = "num_images" if unit == "images" else "num_videos"
    rate_fmt = ".2f" if unit == "images" else ".4f"

    print("\n" + "=" * 90)
    print(f"  THROUGHPUT COMPARISON ({unit}/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {unit.title():>7} {'kb-nano':>12}"
    if vllm_results:
        header += f" {'vllm-omni':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb[count_key]:>7} {kb[rate_key]:>12{rate_fmt}}"
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                speedup = kb[rate_key] / vllm[rate_key]
                line += f" {vllm[rate_key]:>12{rate_fmt}} {speedup:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(
    kb_results: list[dict], vllm_results: list[dict] | None, *, is_video: bool = False,
):
    print("\n" + "=" * 90)
    unit_label = "seconds per video" if is_video else "seconds"
    print(f"  LATENCY COMPARISON ({unit_label})")
    print("=" * 90)
    header = f"  {'Scenario':<25}"
    if is_video:
        header += f" {'Res':>10} {'Frames':>7}"
    header += f" {'kb-nano p50':>12}"
    if vllm_results:
        header += f" {'vllm-omni p50':>14} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 80)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        line = f"  {kb['name']:<25}"
        if is_video:
            res = f"{kb['height']}x{kb['width']}"
            line += f" {res:>10} {kb.get('num_frames', ''):>7}"
        line += f" {kb_p50:>12.3f}"
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                vllm_lats = np.array(vllm["latencies"])
                vllm_p50 = np.percentile(vllm_lats, 50)
                speedup = vllm_p50 / kb_p50
                line += f" {vllm_p50:>14.3f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_correctness_flux(correctness: dict):
    print("\n" + "=" * 90)
    print("  CORRECTNESS COMPARISON (packed latent space, per-batch)")
    print("=" * 90)
    print(f"  {'Scenario':<25} {'Batches':>8} {'Mean CosSim':>12} {'Min CosSim':>11} {'Mean MSE':>12} {'Max MSE':>12} {'Result':>8}")
    print("  " + "-" * 88)

    all_pass = True
    for scenario, stats in correctness.items():
        mean_cos = stats["mean_cosine_sim"]
        verdict = "PASS" if mean_cos > 0.98 else ("WARN" if mean_cos > 0.95 else "FAIL")
        if verdict != "PASS":
            all_pass = False
        print(
            f"  {scenario:<25} {stats['num_batches']:>8} "
            f"{mean_cos:>12.6f} {stats['min_cosine_sim']:>11.6f} "
            f"{stats['mean_mse']:>12.2e} {stats['max_mse']:>12.2e} "
            f"{verdict:>8}"
        )
    print()
    if all_pass:
        print("  All scenarios PASS (mean cosine similarity > 0.98)")
    else:
        print("  WARNING: Some scenarios have divergent outputs")
    print()


def _print_correctness_hunyuan(correctness: dict):
    print("\n" + "=" * 110)
    print("  CORRECTNESS COMPARISON (decoded video frames, per-prompt)")
    print("=" * 110)
    print(
        f"  {'Scenario':<25} {'Prompts':>8} {'Mean CosSim':>12}"
        f" {'Min CosSim':>11} {'Mean PSNR':>10} {'Min PSNR':>9}"
        f" {'Mean MSE':>12} {'Result':>8}"
    )
    print("  " + "-" * 102)

    all_pass = True
    for scenario, stats in correctness.items():
        mean_cos = stats["mean_cosine_sim"]
        mean_psnr = stats.get("mean_psnr", 0)
        min_psnr = stats.get("min_psnr", 0)
        verdict = "PASS" if mean_cos > 0.95 else ("WARN" if mean_cos > 0.90 else "FAIL")
        if verdict != "PASS":
            all_pass = False
        print(
            f"  {scenario:<25} {stats['num_prompts']:>8}"
            f" {mean_cos:>12.6f} {stats['min_cosine_sim']:>11.6f}"
            f" {mean_psnr:>10.2f} {min_psnr:>9.2f}"
            f" {stats['mean_mse']:>12.2e}"
            f" {verdict:>8}"
        )
    print()
    if all_pass:
        print("  All scenarios PASS (mean cosine similarity > 0.95)")
    else:
        print("  WARNING: Some scenarios have divergent outputs")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# TTS result printing & comparison
# ═══════════════════════════════════════════════════════════════════════════

def _print_tts_throughput_comparison(kb_results: list[dict],
                                     vllm_results: list[dict] | None):
    print("\n" + "=" * 90)
    print("  TTS THROUGHPUT COMPARISON")
    print("=" * 90)
    header = f"  {'Scenario':<20} {'Utts':>6} {'kb-nano utt/s':>14} {'kb-nano RTF':>12}"
    if vllm_results:
        header += f" {'vllm-omni utt/s':>16} {'vllm-omni RTF':>14} {'Speedup':>8}"
    print(header)
    print("  " + "-" * 80)

    for kb in kb_results:
        line = (f"  {kb['name']:<20} {kb['num_utterances']:>6} "
                f"{kb['utterances_per_second']:>14.2f} "
                f"{kb['rtf']:>12.3f}")
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                speedup = kb["utterances_per_second"] / max(vllm["utterances_per_second"], 1e-9)
                line += (f" {vllm['utterances_per_second']:>16.2f} "
                         f"{vllm['rtf']:>14.3f} {speedup:>7.2f}x")
        print(line)
    print()


def _print_tts_latency_comparison(kb_results: list[dict],
                                   vllm_results: list[dict] | None):
    print("\n" + "=" * 80)
    print("  TTS LATENCY COMPARISON (seconds)")
    print("=" * 80)
    header = f"  {'Scenario':<20} {'kb-nano p50':>12}"
    if vllm_results:
        header += f" {'vllm-omni p50':>14} {'Speedup':>8}"
    print(header)
    print("  " + "-" * 60)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        line = f"  {kb['name']:<20} {kb_p50:>12.3f}"
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                vllm_lats = np.array(vllm["latencies"])
                vllm_p50 = np.percentile(vllm_lats, 50)
                speedup = vllm_p50 / kb_p50 if kb_p50 > 0 else float("inf")
                line += f" {vllm_p50:>14.3f} {speedup:>7.2f}x"
        print(line)
    print()


def _compute_mel_spectrogram(audio: np.ndarray, sr: int = 24000,
                              n_fft: int = 1024, hop_length: int = 256,
                              n_mels: int = 80) -> np.ndarray:
    """Compute log-mel spectrogram from audio waveform using numpy/scipy."""
    from scipy.signal import get_window

    win = get_window("hann", n_fft, fftbins=True)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames <= 0:
        return np.zeros((n_mels, 1), dtype=np.float32)

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    ).copy()
    frames *= win
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).T ** 2

    fmin, fmax = 0.0, sr / 2.0
    mel_lo = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_pts = 700.0 * (10.0 ** (np.linspace(mel_lo, mel_hi, n_mels + 2) / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * mel_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m, k] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m, k] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)

    mel_spec = fb @ spec
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    return log_mel.astype(np.float32)


def _compare_tts_audio(kb_audio_dir: str, vllm_audio_dir: str) -> dict:
    """Compare TTS audio between kb-nano and vllm-omni using mel spectrogram
    cosine similarity.
    """
    import torch

    kb_files = sorted(
        f for f in os.listdir(kb_audio_dir) if f.endswith(".pt")
    ) if os.path.isdir(kb_audio_dir) else []
    vllm_files = sorted(
        f for f in os.listdir(vllm_audio_dir) if f.endswith(".pt")
    ) if os.path.isdir(vllm_audio_dir) else []
    common = sorted(set(kb_files) & set(vllm_files))
    if not common:
        return {}

    scenario_utts: dict[str, list[dict]] = defaultdict(list)

    for fname in common:
        kb_audio = torch.load(
            os.path.join(kb_audio_dir, fname), map_location="cpu", weights_only=True
        ).detach().float().flatten().numpy()
        vllm_audio = torch.load(
            os.path.join(vllm_audio_dir, fname), map_location="cpu", weights_only=True
        ).detach().float().flatten().numpy()

        kb_len = len(kb_audio)
        vllm_len = len(vllm_audio)
        scenario_name = fname.rsplit("_utt", 1)[0]

        entry: dict = {
            "file": fname,
            "kb_samples": kb_len,
            "vllm_samples": vllm_len,
            "kb_nonempty": kb_len > 0,
            "vllm_nonempty": vllm_len > 0,
            "both_nonempty": kb_len > 0 and vllm_len > 0,
        }

        if kb_len > 0 and vllm_len > 0:
            entry["len_ratio"] = kb_len / vllm_len
            min_len = min(kb_len, vllm_len)
            kb_mel = _compute_mel_spectrogram(kb_audio[:min_len])
            vllm_mel = _compute_mel_spectrogram(vllm_audio[:min_len])
            kb_flat = kb_mel.flatten()
            vllm_flat = vllm_mel.flatten()
            norm_product = np.linalg.norm(kb_flat) * np.linalg.norm(vllm_flat)
            if norm_product > 1e-12:
                entry["mel_cosine_sim"] = float(
                    np.dot(kb_flat, vllm_flat) / norm_product)
            else:
                entry["mel_cosine_sim"] = 0.0
            entry["length_match"] = abs(kb_len - vllm_len) < 10

        scenario_utts[scenario_name].append(entry)

    results = {}
    for scenario, utts in scenario_utts.items():
        mel_sims = [u["mel_cosine_sim"] for u in utts if "mel_cosine_sim" in u]
        len_ratios = [u["len_ratio"] for u in utts if "len_ratio" in u]
        length_matches = sum(1 for u in utts if u.get("length_match", False))
        results[scenario] = {
            "num_utterances": len(utts),
            "both_nonempty": sum(1 for u in utts if u["both_nonempty"]),
            "kb_nonempty": sum(1 for u in utts if u["kb_nonempty"]),
            "vllm_nonempty": sum(1 for u in utts if u["vllm_nonempty"]),
            "mel_cosine_sims": mel_sims,
            "mean_mel_cosine_sim": float(np.mean(mel_sims)) if mel_sims else None,
            "median_mel_cosine_sim": float(np.median(mel_sims)) if mel_sims else None,
            "min_mel_cosine_sim": float(np.min(mel_sims)) if mel_sims else None,
            "p10_mel_cosine_sim": float(np.percentile(mel_sims, 10)) if mel_sims else None,
            "mean_len_ratio": float(np.mean(len_ratios)) if len_ratios else None,
            "length_match_count": length_matches,
            "length_match_total": len(utts),
            "per_utt": utts,
        }
    return results


def _print_tts_correctness(audio_check: dict):
    """Print e2e correctness table using mel spectrogram cosine similarity.

    PASS/FAIL is based on the **overall median** mel cosine similarity across
    all utterances (threshold >= 0.88).  Median is used instead of mean because
    attention-backend divergence (SDPA vs TritonAttention) causes a long tail
    of outliers that disproportionately pulls down the mean.
    """
    THRESHOLD = 0.88

    print("\n" + "=" * 100)
    print("  E2E CORRECTNESS: mel spectrogram cosine similarity (median for PASS/FAIL)")
    print("=" * 100)
    print(f"  {'Scenario':<16} {'Utts':>5} {'Median':>8} {'Mean':>8} "
          f"{'P10':>8} {'Min':>8} {'LenMatch':>10}")
    print("  " + "-" * 80)

    all_sims: list[float] = []
    all_medians = []
    total_utts = 0
    for scenario, s in sorted(audio_check.items()):
        n = s["num_utterances"]
        total_utts += n
        median_sim = s.get("median_mel_cosine_sim")
        mean_sim = s.get("mean_mel_cosine_sim")
        min_sim = s.get("min_mel_cosine_sim")
        p10_sim = s.get("p10_mel_cosine_sim")
        lm_count = s.get("length_match_count", 0)
        lm_total = s.get("length_match_total", n)

        if median_sim is not None:
            all_medians.append(median_sim)
        sims = s.get("mel_cosine_sims", [])
        all_sims.extend(sims)

        def _fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"

        lm_str = f"{lm_count}/{lm_total}"
        print(f"  {scenario:<16} {n:>5} {_fmt(median_sim):>8} {_fmt(mean_sim):>8} "
              f"{_fmt(p10_sim):>8} {_fmt(min_sim):>8} "
              f"{lm_str:>10}")

    overall_median = float(np.median(all_sims)) if all_sims else None
    overall_mean = float(np.mean(all_sims)) if all_sims else None

    print()
    print(f"  Overall median mel cosine similarity: "
          f"{overall_median:.3f}" if overall_median is not None else "  N/A")
    print(f"  Overall mean mel cosine similarity:   "
          f"{overall_mean:.3f}" if overall_mean is not None else "  N/A")
    print()

    passed = overall_median is not None and overall_median >= THRESHOLD
    if passed:
        print(f"  E2E RESULT: PASS (overall median {overall_median:.3f} >= {THRESHOLD})")
    else:
        print(f"  E2E RESULT: FAIL (overall median "
              f"{overall_median:.3f if overall_median is not None else 'N/A'} < {THRESHOLD})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _infer_mode(model: str) -> str:
    """Infer benchmark mode from the model's category."""
    category = MODEL_CATEGORY.get(model)
    if category == "tts":
        return "tts"
    if _is_hunyuan_video(model):
        return "video"
    if category == "diffusion":
        return "diffusion"
    raise ValueError(
        f"Cannot infer benchmark mode for model {model!r}. "
        f"Known categories: {dict(MODEL_CATEGORY)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: kb-nano vs vllm-omni (diffusion / TTS)",
    )
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Model name (mode is inferred from MODEL_CATEGORY)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-vllm-omni", action="store_true",
                        help="Skip vllm-omni (no correctness comparison)")
    parser.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Patch semantic PyTorch references from tasks/reference/L*/ into kb-nano.",
    )
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput phase (run latency only)")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip latency phase")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for FLUX scenarios")
    parser.add_argument("--max-tts-samples", type=int, default=None,
                        help="Limit TTS utterances per scenario (for quick tests)")
    parser.add_argument("--latency-iters", type=int, default=5,
                        help="Timed iterations per latency scenario")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    mode = _infer_mode(args.model)
    gpu_name = _detect_gpu_name()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(_PACKAGE_DIR / "tests" / "results" / gpu_name / short)

    os.makedirs(args.output_dir, exist_ok=True)

    if mode == "tts":
        _run_tts(args, gpu_name)
    elif mode == "video":
        _run_hunyuan(args, gpu_name)
    else:
        _run_flux(args, gpu_name)


def _run_flux(args, gpu_name: str):
    bench_prompts = _get_parti_prompts(args.seed)

    print(f"\n{'=' * 70}")
    print("  kb-nano vs vllm-omni -- FLUX Benchmark")
    print(f"{'=' * 70}")
    print(f"  Model          : {args.model}")
    print(f"  GPU            : {gpu_name}")
    print(f"  Seed           : {args.seed}")
    print(f"  Prompts source : parti-prompts ({len(bench_prompts)} prompts)")
    print(f"  Output dir     : {args.output_dir}")
    print(f"{'=' * 70}")

    run_vllm = not args.skip_vllm_omni
    save_latents = run_vllm

    kb_latent_dir = os.path.join(args.output_dir, "latents", "kb_nano") if save_latents else None
    vllm_latent_dir = os.path.join(args.output_dir, "latents", "vllm_omni") if save_latents else None

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    scenarios = _build_flux_throughput_scenarios(bench_prompts, args.batch_size) if not args.skip_throughput else []
    latency_scenarios = _build_flux_latency_scenarios(bench_prompts) if not args.skip_latency else []

    # --- kb-nano ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "pytorch_reference": args.pytorch_reference,
    }
    if kb_latent_dir:
        kb_config["latent_dir"] = kb_latent_dir
    kb_data = run_worker(FLUX_KB_NANO_WORKER, kb_config, "kb-nano FLUX benchmark", timeout=36000)

    # --- vllm-omni ---
    vllm_data = None
    if run_vllm:
        vllm_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
        if vllm_latent_dir:
            vllm_config["latent_dir"] = vllm_latent_dir
        vllm_data = run_worker(FLUX_VLLM_OMNI_WORKER, vllm_config, "vllm-omni FLUX benchmark", timeout=36000)

    # --- Print ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None
        if kb_tp:
            _print_throughput_comparison(kb_tp, vllm_tp, unit="images")
        if kb_lat:
            _print_latency_comparison(kb_lat, vllm_lat, is_video=False)
        correctness = None
        if save_latents and kb_latent_dir and vllm_latent_dir:
            correctness = _compare_latents(kb_latent_dir, vllm_latent_dir)
            if correctness:
                _print_correctness_flux(correctness)
            else:
                print("\n  WARNING: No matching latent files found for correctness comparison.")
        _save_results(args, gpu_name, bench_prompts, kb_data, vllm_data, correctness,
                      prompts_source="nateraw/parti-prompts")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)


def _run_hunyuan(args, gpu_name: str):
    bench_prompts = _get_movie_gen_prompts(args.seed)

    print(f"\n{'=' * 70}")
    print("  kb-nano vs vllm-omni -- HunyuanVideo-1.5 Benchmark")
    print(f"{'=' * 70}")
    print(f"  Model          : {args.model}")
    print(f"  GPU            : {gpu_name}")
    print(f"  Seed           : {args.seed}")
    print(f"  Prompts source : Movie Gen Video Bench ({len(bench_prompts)} prompts)")
    print(f"  Inference steps: {HUNYUAN_VIDEO_CONFIG.num_inference_steps}")
    print(f"  Guidance scale : {HUNYUAN_VIDEO_CONFIG.guidance_scale}")
    print(f"  Output dir     : {args.output_dir}")

    run_vllm = not args.skip_vllm_omni
    save_frames = run_vllm

    kb_frames_dir = os.path.join(args.output_dir, "frames", "kb_nano") if save_frames else None
    vllm_frames_dir = os.path.join(args.output_dir, "frames", "vllm_omni") if save_frames else None

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": True,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    scenarios = _build_hunyuan_throughput_scenarios(bench_prompts) if not args.skip_throughput else []
    latency_scenarios = _build_hunyuan_latency_scenarios(bench_prompts) if not args.skip_latency else []

    if not args.skip_throughput:
        tp_desc = ", ".join(f"{s['name']}({len(s['prompts'])} prompts)" for s in scenarios)
        print(f"  Throughput     : {tp_desc}")
    if latency_scenarios:
        lat_desc = ", ".join(s["name"] for s in latency_scenarios)
        print(f"  Latency        : {lat_desc} ({args.latency_iters} iters)")
    print(f"{'=' * 70}")

    # --- kb-nano ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "pytorch_reference": args.pytorch_reference,
    }
    if kb_frames_dir:
        kb_config["frames_dir"] = kb_frames_dir
    kb_data = run_worker(HUNYUAN_KB_NANO_WORKER, kb_config, "kb-nano HunyuanVideo benchmark", timeout=36000)

    # --- vllm-omni ---
    vllm_data = None
    if run_vllm:
        vllm_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
        if vllm_frames_dir:
            vllm_config["frames_dir"] = vllm_frames_dir
        vllm_data = run_worker(HUNYUAN_VLLM_OMNI_WORKER, vllm_config, "vllm-omni HunyuanVideo benchmark", timeout=36000)

    # --- Print ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None
        if kb_tp:
            _print_throughput_comparison(kb_tp, vllm_tp, unit="videos")
        if kb_lat:
            _print_latency_comparison(kb_lat, vllm_lat, is_video=True)
        correctness = None
        if save_frames and kb_frames_dir and vllm_frames_dir:
            correctness = _compare_frames(kb_frames_dir, vllm_frames_dir)
            if correctness:
                _print_correctness_hunyuan(correctness)
            else:
                print("\n  WARNING: No matching frame files found for correctness comparison.")
        _save_results(args, gpu_name, bench_prompts, kb_data, vllm_data, correctness,
                      prompts_source="meta-ai-for-media-research/movie_gen_video_bench")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)


def _run_tts(args, gpu_name: str):
    """Run TTS (CosyVoice3) benchmark."""
    print(f"\nBenchmark: CosyVoice3 TTS on {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {args.output_dir}")

    max_samples = max(w.num_requests for w in TTS_THROUGHPUT_WORKLOADS) + 10
    tts_samples = _get_tts_samples(args.seed, max_samples)
    print(f"Loaded {len(tts_samples)} TTS samples from SEED-TTS-Eval")

    run_vllm = not args.skip_vllm_omni
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    ref_audio = rng.randn(24000 * 3).astype(np.float32)
    ref_sr = 24000
    prompt_text = "Testing my voice."

    kb_audio_dir = os.path.join(args.output_dir, "audio", "kb_nano")
    vllm_audio_dir = os.path.join(args.output_dir, "audio", "vllm_omni")

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "n_timesteps": COSYVOICE3_CONFIG.n_timesteps,
        "sample_rate": COSYVOICE3_CONFIG.sample_rate,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
        "ref_audio": ref_audio.tolist(),
        "ref_sr": ref_sr,
        "prompt_text": prompt_text,
        "greedy": True,
    }

    scenarios = _build_tts_throughput_scenarios(tts_samples)
    latency_scenarios = _build_tts_latency_scenarios(tts_samples)

    if getattr(args, "max_tts_samples", None):
        cap = args.max_tts_samples
        for s in scenarios:
            s["texts"] = s["texts"][:cap]
            s["num_requests"] = len(s["texts"])
        print(f"  (capped to {cap} utterances per scenario for quick test)")

    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "audio_dir": kb_audio_dir,
        "pytorch_reference": args.pytorch_reference,
    }
    kb_data = run_worker(
        KB_NANO_TTS_WORKER, kb_config,
        "kb-nano TTS benchmark", timeout=36000,
    )

    vllm_data = None
    if run_vllm:
        vllm_config = {
            **base_config,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
            "audio_dir": vllm_audio_dir,
        }
        vllm_data = run_worker(
            VLLM_OMNI_TTS_WORKER, vllm_config,
            "vllm-omni TTS benchmark", timeout=36000,
        )

    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None

        _print_tts_throughput_comparison(kb_tp, vllm_tp)
        _print_tts_latency_comparison(kb_lat, vllm_lat)

        c2w_eq = kb_data.get("code2wav_equivalence", {})
        if c2w_eq:
            print("\n" + "=" * 70)
            print("  CODE2WAV EQUIVALENCE (same tokens, same seed)")
            print("=" * 70)
            if "error" in c2w_eq:
                print(f"  ERROR: {c2w_eq['error']}")
            else:
                print(f"  Mel cosine similarity: {c2w_eq.get('mel_cosine_similarity', 'N/A'):.6f}")
                print(f"  Mel MSE:               {c2w_eq.get('mel_mse', 'N/A'):.2e}")
                print(f"  Tokens tested:         {c2w_eq.get('num_tokens', 'N/A')}")
                print(f"  Result:                {'PASS' if c2w_eq.get('pass') else 'FAIL'}")
            print()

        audio_check = None
        if run_vllm:
            audio_check = _compare_tts_audio(kb_audio_dir, vllm_audio_dir)
            _print_tts_correctness(audio_check)

        results_path = os.path.join(args.output_dir, "results.json")
        results = {
            "model": args.model,
            "gpu": gpu_name,
            "seed": args.seed,
            "n_timesteps": COSYVOICE3_CONFIG.n_timesteps,
            "dataset": "SEED-TTS-Eval",
            "kb_nano": kb_data,
            "code2wav_equivalence": c2w_eq,
        }
        if vllm_data:
            results["vllm_omni"] = vllm_data
        if audio_check:
            for sc in audio_check.values():
                sc.pop("per_utt", None)
            results["audio_correctness"] = audio_check
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")
    else:
        print("ERROR: kb-nano TTS benchmark failed.")
        sys.exit(1)


def _save_results(
    args, gpu_name: str, bench_prompts: list[str],
    kb_data: dict, vllm_data: dict | None, correctness: dict | None,
    prompts_source: str,
):
    results_path = os.path.join(args.output_dir, "results.json")
    results = {
        "model": args.model,
        "gpu": gpu_name,
        "seed": args.seed,
        "prompts_source": prompts_source,
        "num_prompts": len(bench_prompts),
        "kb_nano": kb_data,
    }
    if vllm_data:
        results["vllm_omni"] = vllm_data
    if correctness:
        results["correctness"] = correctness
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
