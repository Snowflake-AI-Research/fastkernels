#!/usr/bin/env python3
"""Throughput, latency, and correctness benchmark: fastkernels vs open-oasis.

Runs the full Oasis 500M autoregressive diffusion pipeline for both engines:
prompt VAE encode, DiT denoising rollout, and VAE decode.

Reference baseline:
    Official open-oasis source from https://github.com/etched-ai/open-oasis

Weights:
    Downloaded from the Etched/oasis-500m Hugging Face checkpoint:
    oasis500m.safetensors and vit-l-20.safetensors.

Default workload:
    Real Minecraft clips from TESS-Computer/minecraft-vla-stage1. The script
    converts the dataset action strings to Oasis' 25-d action vector and caches
    prompt/action tensors under data/oasis_cache/.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_model
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm import tqdm


def _bootstrap_local_package() -> None:
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "fastkernels",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastkernels"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_bootstrap_local_package()

from fastkernels.tasks.baseline.L4.oasis import (  # noqa: E402
    OasisConfig,
    OasisPipeline,
    OasisSamplingParams,
)
from fastkernels.bench.utils.workloads import (  # noqa: E402
    OASIS_LATENCY_WORKLOADS,
    OASIS_THROUGHPUT_WORKLOADS,
    OasisWorkload,
)


OPEN_OASIS_REPO = "https://github.com/etched-ai/open-oasis.git"
OASIS_MODEL = "Etched/oasis-500m"
CACHE_FORMAT_VERSION = 3
WARMUP_ITERS = 2
THROUGHPUT_ITERS = 5
CORRECTNESS_COSINE_THRESHOLDS = {
    "prompt_latents": 0.999,
    "latents": 0.99,
    "video": 0.99,
}


def _log(message: str) -> None:
    print(f"[bench_oasis] {message}", flush=True)

OASIS_ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

ACTION_INDEX = {name: idx for idx, name in enumerate(OASIS_ACTION_KEYS)}

LUMINE_TOKEN_TO_ACTION = {
    "E": "inventory",
    "Esc": "ESC",
    "ESC": "ESC",
    "W": "forward",
    "S": "back",
    "A": "left",
    "D": "right",
    "Space": "jump",
    "Shift": "sneak",
    "Ctrl": "sprint",
    "F": "swapHands",
    "LMB": "attack",
    "RMB": "use",
    "MMB": "pickItem",
    "Q": "drop",
}


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


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _download_model(model_name: str) -> str:
    _log(f"checking/downloading Hugging Face checkpoint: {model_name}")
    model_dir = snapshot_download(model_name, allow_patterns=["*.safetensors", "README.md", "LICENSE"])
    _log(f"checkpoint ready: {model_dir}")
    return model_dir


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "oasis_cache"


def _default_open_oasis_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "open_oasis_src"


def _ensure_open_oasis_source(open_oasis_src: str | None, *, repo: str) -> Path:
    if open_oasis_src is not None:
        src = Path(open_oasis_src).expanduser().resolve()
        if not (src / "dit.py").exists() or not (src / "vae.py").exists() or not (src / "utils.py").exists():
            raise FileNotFoundError(f"{src} does not look like an open-oasis checkout")
        _log(f"using open-oasis source: {src}")
        return src

    src = _default_open_oasis_dir()
    if (src / "dit.py").exists() and (src / "vae.py").exists() and (src / "utils.py").exists():
        _log(f"using cached open-oasis source: {src}")
        return src

    src.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.rmtree(src)
    _log(f"cloning open-oasis into {src}")
    subprocess.run(["git", "clone", "--depth", "1", repo, str(src)], check=True)
    _log("open-oasis clone ready")
    return src


def _import_open_oasis_modules(open_oasis_src: Path):
    src = str(open_oasis_src)
    if src not in sys.path:
        sys.path.insert(0, src)
    for name in ("dit", "vae", "utils", "attention", "rotary_embedding_torch"):
        sys.modules.pop(name, None)
    dit = importlib.import_module("dit")
    vae = importlib.import_module("vae")
    utils = importlib.import_module("utils")
    return dit, vae, utils


def _checkpoint_paths(model_dir: str) -> tuple[str, str]:
    model_path = Path(model_dir) / "oasis500m.safetensors"
    vae_path = Path(model_dir) / "vit-l-20.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"missing Oasis DiT checkpoint: {model_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"missing Oasis VAE checkpoint: {vae_path}")
    return str(model_path), str(vae_path)


def _build_open_oasis(
    model_dir: str,
    open_oasis_src: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, Any]:
    _log("building open-oasis reference models")
    dit_mod, vae_mod, utils_mod = _import_open_oasis_modules(open_oasis_src)
    model_path, vae_path = _checkpoint_paths(model_dir)

    model = dit_mod.DiT_models["DiT-S/2"]()
    vae = vae_mod.VAE_models["vit-l-20-shallow-encoder"]()
    _log("loading open-oasis DiT weights")
    load_model(model, model_path)
    _log("loading open-oasis VAE weights")
    load_model(vae, vae_path)
    model = model.to(device=device).eval()
    vae = vae.to(device=device).eval()
    _log("open-oasis reference ready")
    return model, vae, utils_mod.sigmoid_beta_schedule


def _build_fastkernels(model_dir: str, *, device: torch.device, dtype: torch.dtype) -> OasisPipeline:
    _log("building fastkernels Oasis pipeline")
    pipeline = OasisPipeline(OasisConfig())
    _log("loading fastkernels Oasis weights")
    pipeline.load_weights(model_dir)
    del dtype
    pipeline.model.to(device=device)
    pipeline.vae.to(device=device)
    _log("fastkernels Oasis pipeline ready")
    return pipeline.eval()


def _normalize_mouse_delta(value: int) -> float:
    return max(-1.0, min(1.0, float(value) / 1000.0))


def _decode_dataset_image(image_value: Any) -> torch.Tensor:
    if isinstance(image_value, bytes):
        image = Image.open(io.BytesIO(image_value)).convert("RGB")
    elif hasattr(image_value, "convert"):
        image = image_value.convert("RGB")
    else:
        raise TypeError(f"unsupported dataset image type: {type(image_value)!r}")
    tensor = pil_to_tensor(image)
    tensor = resize(tensor, (360, 640))
    return tensor.float() / 255.0


def _parse_lumine_action(action_text: str) -> torch.Tensor:
    inner = action_text.replace("<|action_start|>", "").replace("<|action_end|>", "").strip()
    parts = [part.strip() for part in inner.split(";")]
    if len(parts) < 5:
        raise ValueError(f"malformed action string: {action_text!r}")

    mouse_tokens = parts[0].split()
    if len(mouse_tokens) < 3:
        raise ValueError(f"malformed mouse header: {action_text!r}")

    action = torch.zeros(len(OASIS_ACTION_KEYS), dtype=torch.float32)
    action[ACTION_INDEX["cameraX"]] = _normalize_mouse_delta(int(mouse_tokens[0]))
    action[ACTION_INDEX["cameraY"]] = _normalize_mouse_delta(int(mouse_tokens[1]))

    active_tokens: set[str] = set()
    for chunk in parts[1:5]:
        if chunk:
            active_tokens.update(token for token in chunk.split() if token)

    for token in active_tokens:
        mapped = LUMINE_TOKEN_TO_ACTION.get(token)
        if mapped is not None:
            action[ACTION_INDEX[mapped]] = 1.0
        elif token.isdigit() and 1 <= int(token) <= 9:
            action[ACTION_INDEX[f"hotbar.{token}"]] = 1.0
    return action


def _dataset_cache_path(
    dataset_name: str,
    dataset_split: str,
    *,
    cache_dir: Path,
    num_clips: int,
    num_frames: int,
    n_prompt_frames: int,
) -> Path:
    dataset_slug = dataset_name.replace("/", "__")
    split_slug = dataset_split.replace("/", "__")
    filename = (
        f"{dataset_slug}_{split_slug}_v{CACHE_FORMAT_VERSION}_clips{num_clips}_"
        f"frames{num_frames}_prompt{n_prompt_frames}.pt"
    )
    return cache_dir / filename


def _prepare_dataset_cache(
    dataset_name: str,
    dataset_split: str,
    *,
    cache_path: Path,
    num_clips: int,
    num_frames: int,
    n_prompt_frames: int,
) -> None:
    from datasets import load_dataset

    if n_prompt_frames > num_frames:
        raise ValueError("n_prompt_frames must be <= num_frames")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prompts: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    clip_metadata: list[dict[str, Any]] = []
    used_video_ids: set[str] = set()

    frame_buffer: deque[torch.Tensor] = deque()
    action_buffer: deque[torch.Tensor] = deque()
    current_video_id: str | None = None
    previous_frame_idx: int | None = None
    clip_start_frame_idx: int | None = None

    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    _log(
        f"building dataset cache: {dataset_name}:{dataset_split}, "
        f"clips={num_clips}, frames={num_frames}, prompt_frames={n_prompt_frames}"
    )
    pbar = tqdm(total=num_clips, desc="dataset clips", unit="clip", file=sys.stderr)
    for row in dataset:
        video_id = str(row["video_id"])
        frame_idx = int(row["frame_idx"])

        is_new_video = current_video_id != video_id
        is_non_consecutive = previous_frame_idx is not None and frame_idx != previous_frame_idx + 1
        if is_new_video or is_non_consecutive:
            frame_buffer.clear()
            action_buffer.clear()
            clip_start_frame_idx = None

        if video_id in used_video_ids:
            current_video_id = video_id
            previous_frame_idx = frame_idx
            frame_buffer.clear()
            action_buffer.clear()
            clip_start_frame_idx = None
            continue

        if clip_start_frame_idx is None:
            clip_start_frame_idx = frame_idx

        frame_buffer.append(_decode_dataset_image(row["image"]))
        action_buffer.append(_parse_lumine_action(row["action"]))
        current_video_id = video_id
        previous_frame_idx = frame_idx

        if len(frame_buffer) < num_frames:
            continue

        prompt = torch.stack(list(frame_buffer)[:n_prompt_frames], dim=0).half()
        if num_frames > 1:
            raw_actions = torch.stack(list(action_buffer)[: num_frames - 1], dim=0)
            action_tensor = torch.cat([torch.zeros_like(raw_actions[:1]), raw_actions], dim=0)
        else:
            action_tensor = torch.zeros(1, len(OASIS_ACTION_KEYS), dtype=torch.float32)

        if action_tensor.shape[0] > 1:
            control_activity = action_tensor[1:].abs().amax(dim=0)
            distinct_controls = int((control_activity > 0).sum().item())
            camera_activity = float(
                action_tensor[1:, ACTION_INDEX["cameraX"]].abs().mean().item()
                + action_tensor[1:, ACTION_INDEX["cameraY"]].abs().mean().item()
            )
            if distinct_controls < 2 and camera_activity < 0.02:
                frame_buffer.clear()
                action_buffer.clear()
                clip_start_frame_idx = None
                continue

        prompts.append(prompt)
        actions.append(action_tensor)
        used_video_ids.add(video_id)
        clip_metadata.append(
            {
                "video_id": video_id,
                "start_frame_idx": clip_start_frame_idx,
                "end_frame_idx": frame_idx,
            }
        )
        pbar.update(1)

        frame_buffer.clear()
        action_buffer.clear()
        clip_start_frame_idx = None
        if len(prompts) >= num_clips:
            break
    pbar.close()

    if len(prompts) < num_clips:
        raise RuntimeError(
            f"requested {num_clips} clips from {dataset_name}:{dataset_split}, collected {len(prompts)}"
        )

    torch.save(
        {
            "dataset": dataset_name,
            "split": dataset_split,
            "num_clips": num_clips,
            "num_frames": num_frames,
            "n_prompt_frames": n_prompt_frames,
            "prompt": torch.stack(prompts, dim=0),
            "actions": torch.stack(actions, dim=0),
            "clips": clip_metadata,
        },
        cache_path,
    )
    _log(f"saved dataset cache: {cache_path}")


def _load_real_dataset_inputs(
    *,
    dataset_name: str,
    dataset_split: str,
    cache_dir: Path,
    num_clips: int,
    num_frames: int,
    n_prompt_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    cache_path = _dataset_cache_path(
        dataset_name,
        dataset_split,
        cache_dir=cache_dir,
        num_clips=num_clips,
        num_frames=num_frames,
        n_prompt_frames=n_prompt_frames,
    )
    if not cache_path.exists():
        _prepare_dataset_cache(
            dataset_name,
            dataset_split,
            cache_path=cache_path,
            num_clips=num_clips,
            num_frames=num_frames,
            n_prompt_frames=n_prompt_frames,
        )
    else:
        _log(f"using dataset cache: {cache_path}")
    _log("loading dataset tensors")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    prompt = payload["prompt"].float().to(device)
    actions = payload["actions"].float().to(device)
    _log(f"dataset tensors ready: prompt={tuple(prompt.shape)}, actions={tuple(actions.shape)}")
    return (
        prompt,
        actions,
        {
            "mode": "dataset",
            "name": payload["dataset"],
            "split": payload["split"],
            "num_clips": int(payload["num_clips"]),
            "cache_path": str(cache_path),
            "clips": payload["clips"],
        },
    )


def _scenario_params(scenario: OasisWorkload, *, seed: int) -> OasisSamplingParams:
    return OasisSamplingParams(
        num_frames=scenario.num_frames,
        ddim_steps=scenario.ddim_steps,
        n_prompt_frames=scenario.n_prompt_frames,
        seed=seed,
    )


def _slice_inputs(
    prompt: torch.Tensor,
    actions: torch.Tensor,
    scenario: OasisWorkload,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = min(scenario.batch_clips, int(prompt.shape[0]))
    if batch < scenario.batch_clips:
        raise ValueError(f"scenario {scenario.name} needs {scenario.batch_clips} clips, only {batch} available")
    return prompt[:batch, : scenario.n_prompt_frames], actions[:batch, : scenario.num_frames]


def _run_kb_pipeline(
    pipeline: OasisPipeline,
    prompt: torch.Tensor,
    actions: torch.Tensor,
    scenario: OasisWorkload,
    *,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    output = pipeline.rollout(prompt, actions, _scenario_params(scenario, seed=seed), dtype=dtype)
    return {
        "prompt_latents": output.prompt_latents,
        "latents": output.latents,
        "video": output.video,
    }


def _run_open_oasis_pipeline(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    beta_schedule,
    prompt: torch.Tensor,
    actions: torch.Tensor,
    scenario: OasisWorkload,
    *,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    device = prompt.device
    params = _scenario_params(scenario, seed=seed)
    bsz, n_prompt_frames, _, height, width = prompt.shape
    scaling_factor = 0.07843137255
    max_noise_level = 1000
    noise_abs_max = 20
    stabilization_level = 15

    x = prompt.reshape(bsz * n_prompt_frames, 3, height, width)
    with torch.inference_mode(), _autocast(device, dtype):
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = x.reshape(bsz, n_prompt_frames, height // vae.patch_size, width // vae.patch_size, -1)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    prompt_latents = x.clone()

    noise_range = torch.linspace(-1, max_noise_level - 1, params.ddim_steps + 1, device=device)
    betas = beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).reshape(-1, 1, 1, 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    for index in range(params.n_prompt_frames, params.num_frames):
        chunk = torch.randn((bsz, 1, *x.shape[-3:]), generator=generator, device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, index + 1 - model.max_frames)

        for noise_idx in reversed(range(1, params.ddim_steps + 1)):
            t_ctx = torch.full((bsz, index), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((bsz, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((bsz, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x[:, start_frame:].clone()
            t_curr = t[:, start_frame:]
            t_next_curr = t_next[:, start_frame:]
            with torch.inference_mode(), _autocast(device, dtype):
                v = model(x_curr, t_curr, actions[:, start_frame : index + 1])

            x_start = alphas_cumprod[t_curr].sqrt() * x_curr - (1 - alphas_cumprod[t_curr]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                1 / alphas_cumprod[t_curr] - 1
            ).sqrt()
            alpha_next = alphas_cumprod[t_next_curr]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    latents = x
    z = latents.permute(0, 1, 3, 4, 2).reshape(bsz * params.num_frames, -1, latents.shape[2])
    with torch.inference_mode():
        video = (vae.decode(z / scaling_factor) + 1) / 2
    video = video.reshape(bsz, params.num_frames, 3, height, width)
    return {
        "prompt_latents": prompt_latents,
        "latents": latents,
        "video": video,
    }


def _benchmark(
    fn,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    units_per_iter: int,
    desc: str,
) -> dict[str, float]:
    for _ in tqdm(range(warmup), desc=f"{desc} warmup", unit="iter", file=sys.stderr):
        fn()
    _sync(device)
    times: list[float] = []
    for _ in tqdm(range(iters), desc=f"{desc} timed", unit="iter", file=sys.stderr):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    total = sum(times)
    sorted_times = sorted(times)
    return {
        "videos_per_second": units_per_iter * iters / total,
        "latency_ms_avg": total / iters * 1000.0,
        "latency_ms_p50": sorted_times[len(sorted_times) // 2] * 1000.0,
        "latency_ms_min": min(times) * 1000.0,
        "latency_ms_max": max(times) * 1000.0,
    }


def _safe_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.reshape(-1).float()
    rhs = rhs.reshape(-1).float()
    lhs_norm = lhs.norm()
    rhs_norm = rhs.norm()
    if lhs_norm.item() == 0.0 and rhs_norm.item() == 0.0:
        return 1.0
    if lhs_norm.item() == 0.0 or rhs_norm.item() == 0.0:
        return 0.0
    return float(F.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0)).item())


def _tensor_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    diff = (lhs - rhs).abs().float()
    return {
        "cosine": _safe_cosine(lhs, rhs),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "mse": float((diff * diff).mean().item()),
    }


def _correctness_status(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    overall_pass = True
    for name, values in metrics.items():
        threshold = CORRECTNESS_COSINE_THRESHOLDS[name]
        passed = values["cosine"] >= threshold
        checked[name] = {
            **values,
            "cosine_threshold": threshold,
            "pass": passed,
        }
        overall_pass = overall_pass and passed
    checked["overall_pass"] = overall_pass
    return checked


def _max_workload_requirements(scenarios: list[OasisWorkload]) -> tuple[int, int, int]:
    max_clips = max(s.batch_clips for s in scenarios)
    max_frames = max(s.num_frames for s in scenarios)
    max_prompt_frames = max(s.n_prompt_frames for s in scenarios)
    return max_clips, max_frames, max_prompt_frames


def _format_float(value: float | None, *, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _correctness_summary(correctness: dict[str, Any] | None) -> tuple[str, str]:
    if correctness is None:
        return "N/A", "N/A"
    status = "PASS" if correctness.get("overall_pass") else "FAIL"
    cosines = [
        values["cosine"]
        for key, values in correctness.items()
        if key != "overall_pass" and isinstance(values, dict) and "cosine" in values
    ]
    min_cosine = min(cosines) if cosines else None
    return status, _format_float(min_cosine, precision=6)


def _print_results_summary(results: dict[str, Any]) -> None:
    performance = results.get("performance", [])
    throughput = [item for item in performance if item["scenario"]["kind"] == "throughput"]
    latency = [item for item in performance if item["scenario"]["kind"] == "latency"]

    print(f"\n\n{'=' * 110}")
    print("  OASIS 500M BENCHMARK SUMMARY")
    print(f"{'=' * 110}")
    print(f"  Model      : {results['model']}")
    print(f"  Reference  : {results['reference']}")
    print(f"  GPU        : {results['gpu']}")
    print(f"  DType      : {results['dtype']}")
    print(f"  Correctness: {results['correctness_dtype']}")
    print(f"  Dataset    : {results['input']['name']} ({results['input']['split']})")
    print(f"  Scenarios  : {', '.join(s['name'] for s in results['scenarios'])}")
    print(f"{'=' * 110}")

    if throughput:
        print(f"\n{'=' * 110}")
        print("  THROUGHPUT SUMMARY")
        print(f"{'=' * 110}")
        print(
            f"  {'SCENARIO':<24} {'CLIPS':>5} {'FRAMES':>6} {'DDIM':>5}"
            f" {'FASTKERNELS vid/s':>15} {'open-oasis vid/s':>17} {'SPEEDUP':>8}"
            f" {'CORRECT':>9} {'MIN COS':>10}"
        )
        print(f"  {'-' * 104}")
        for item in throughput:
            scenario = item["scenario"]
            kb_vps = item["fastkernels"]["videos_per_second"]
            ref_vps = item.get("open_oasis", {}).get("videos_per_second")
            speedup = item.get("speedup")
            correctness_status, min_cosine = _correctness_summary(item.get("correctness"))
            print(
                f"  {scenario['name']:<24} {scenario['batch_clips']:>5}"
                f" {scenario['num_frames']:>6} {scenario['ddim_steps']:>5}"
                f" {kb_vps:>15.2f} {_format_float(ref_vps, precision=2):>17}"
                f" {_format_float(speedup, precision=2, suffix='x'):>8}"
                f" {correctness_status:>9} {min_cosine:>10}"
            )
        print(f"{'=' * 110}")

    if latency:
        print(f"\n{'=' * 105}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 105}")
        print(
            f"  {'SCENARIO':<24} {'CLIPS':>5} {'FRAMES':>6} {'DDIM':>5}"
            f" {'FASTKERNELS p50':>15} {'open-oasis p50':>17} {'SPEEDUP':>8}"
        )
        print(f"  {'-' * 91}")
        for item in latency:
            scenario = item["scenario"]
            kb_p50 = item["fastkernels"]["latency_ms_p50"]
            ref_p50 = item.get("open_oasis", {}).get("latency_ms_p50")
            speedup = item.get("latency_ratio_p50")
            print(
                f"  {scenario['name']:<24} {scenario['batch_clips']:>5}"
                f" {scenario['num_frames']:>6} {scenario['ddim_steps']:>5}"
                f" {kb_p50:>13.2f}ms {_format_float(ref_p50, precision=2, suffix='ms'):>17}"
                f" {_format_float(speedup, precision=2, suffix='x'):>8}"
            )
        print(f"{'=' * 105}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oasis 500M benchmark: fastkernels vs open-oasis")
    parser.add_argument("--model", default=OASIS_MODEL)
    parser.add_argument("--open-oasis-src", default=None, help="Path to an open-oasis checkout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-iters", type=int, default=5)
    parser.add_argument("--skip-open-oasis", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    _log(f"starting benchmark on {device} with dtype={dtype}")
    scenarios: list[OasisWorkload] = []
    if not args.skip_latency:
        scenarios.extend(OASIS_LATENCY_WORKLOADS)
    if not args.skip_throughput:
        scenarios.extend(OASIS_THROUGHPUT_WORKLOADS)
    if not scenarios:
        raise ValueError("Nothing to run: both latency and throughput were skipped.")
    _log("selected scenarios: " + ", ".join(s.name for s in scenarios))
    dataset_name = scenarios[0].dataset_name
    dataset_split = scenarios[0].dataset_split
    if any(s.dataset_name != dataset_name or s.dataset_split != dataset_split for s in scenarios):
        raise ValueError("Oasis benchmark expects all selected workloads to use the same dataset")
    max_clips, max_frames, max_prompt_frames = _max_workload_requirements(scenarios)

    model_dir = _download_model(args.model)
    open_oasis_src = None
    if not args.skip_open_oasis:
        open_oasis_src = _ensure_open_oasis_source(args.open_oasis_src, repo=OPEN_OASIS_REPO)

    prompt, actions, input_info = _load_real_dataset_inputs(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        cache_dir=_default_cache_dir(),
        num_clips=max_clips,
        num_frames=max_frames,
        n_prompt_frames=max_prompt_frames,
        device=device,
    )

    kb = _build_fastkernels(model_dir, device=device, dtype=dtype)
    ref_model = ref_vae = ref_beta_schedule = None
    if not args.skip_open_oasis:
        assert open_oasis_src is not None
        ref_model, ref_vae, ref_beta_schedule = _build_open_oasis(model_dir, open_oasis_src, device=device)

    results: dict[str, Any] = {
        "model": args.model,
        "reference": "open-oasis",
        "open_oasis_src": str(open_oasis_src) if open_oasis_src is not None else None,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "correctness_dtype": str(torch.float32),
        "input": input_info,
        "scenarios": [s.__dict__ for s in scenarios],
    }

    if scenarios:
        throughput_results: list[dict[str, Any]] = []
        for scenario in tqdm(scenarios, desc="scenarios", unit="scenario", file=sys.stderr):
            _log(
                f"scenario {scenario.name}: clips={scenario.batch_clips}, "
                f"frames={scenario.num_frames}, ddim_steps={scenario.ddim_steps}"
            )
            scenario_prompt, scenario_actions = _slice_inputs(prompt, actions, scenario)
            units_per_iter = int(scenario_prompt.shape[0])
            timed_iters = args.latency_iters if scenario.kind == "latency" else THROUGHPUT_ITERS

            correctness = None
            if (
                scenario.kind == "throughput"
                and ref_model is not None
                and ref_vae is not None
                and ref_beta_schedule is not None
            ):
                correctness_dtype = torch.float32
                _log(
                    f"{scenario.name}: running correctness pass for fastkernels and open-oasis "
                    f"with dtype={correctness_dtype}"
                )
                with torch.inference_mode():
                    _log(f"{scenario.name}: correctness fastkernels rollout")
                    kb_out = _run_kb_pipeline(
                        kb,
                        scenario_prompt,
                        scenario_actions,
                        scenario,
                        dtype=correctness_dtype,
                        seed=args.seed,
                    )
                    _log(f"{scenario.name}: correctness open-oasis rollout")
                    ref_out = _run_open_oasis_pipeline(
                        ref_model,
                        ref_vae,
                        ref_beta_schedule,
                        scenario_prompt,
                        scenario_actions,
                        scenario,
                        dtype=correctness_dtype,
                        seed=args.seed,
                    )
                correctness = _correctness_status(
                    {
                        "prompt_latents": _tensor_metrics(kb_out["prompt_latents"], ref_out["prompt_latents"]),
                        "latents": _tensor_metrics(kb_out["latents"], ref_out["latents"]),
                        "video": _tensor_metrics(kb_out["video"], ref_out["video"]),
                    }
                )
                _log(f"{scenario.name}: correctness overall_pass={correctness['overall_pass']}")

            _log(f"{scenario.name}: benchmarking fastkernels")
            kb_metrics = _benchmark(
                lambda s=scenario, p=scenario_prompt, a=scenario_actions: _run_kb_pipeline(
                    kb, p, a, s, dtype=dtype, seed=args.seed
                ),
                device=device,
                warmup=WARMUP_ITERS,
                iters=timed_iters,
                units_per_iter=units_per_iter,
                desc=f"fastkernels {scenario.name}",
            )
            item: dict[str, Any] = {
                "scenario": scenario.__dict__,
                "fastkernels": kb_metrics,
            }
            if correctness is not None:
                item["correctness"] = correctness
            if ref_model is not None and ref_vae is not None and ref_beta_schedule is not None:
                _log(f"{scenario.name}: benchmarking open-oasis")
                ref_metrics = _benchmark(
                    lambda s=scenario, p=scenario_prompt, a=scenario_actions: _run_open_oasis_pipeline(
                        ref_model, ref_vae, ref_beta_schedule, p, a, s, dtype=dtype, seed=args.seed
                    ),
                    device=device,
                    warmup=WARMUP_ITERS,
                    iters=timed_iters,
                    units_per_iter=units_per_iter,
                    desc=f"open-oasis {scenario.name}",
                )
                item["open_oasis"] = ref_metrics
                item["speedup"] = kb_metrics["videos_per_second"] / ref_metrics["videos_per_second"]
                if scenario.kind == "latency":
                    item["latency_ratio_p50"] = (
                        ref_metrics["latency_ms_p50"] / kb_metrics["latency_ms_p50"]
                    )
            throughput_results.append(item)
        results["performance"] = throughput_results

    output_dir = Path(args.output_dir) if args.output_dir else Path("tests/results") / _detect_gpu_name() / "oasis-500m"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    _print_results_summary(results)
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
