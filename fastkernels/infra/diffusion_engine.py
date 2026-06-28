"""
Diffusion inference engine for FLUX and HunyuanVideo DiT models.

Unlike the autoregressive LlamaEngine (paged KV, continuous batching),
this engine runs iterative denoising loops with no KV cache.

Mirrors vllm-omni's DiffusionEngine / DiffusionModelRunner.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from ..tasks.baseline.L4.flux import (
    DiffusionOutput,
    DiffusionSamplingParams,
    FluxConfig,
    FluxPipeline,
)
from ..tasks.baseline.L4.hunyuan_video import (
    HunyuanVideoConfig,
    HunyuanVideoDiffusionOutput,
    HunyuanVideoDiffusionSamplingParams,
    HunyuanVideoPipeline,
)

logger = logging.getLogger(__name__)


def _download_diffusion_model(model_name: str) -> str:
    """Download diffusion model weights from HuggingFace."""
    return snapshot_download(
        model_name,
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.model",
            "tokenizer*", "scheduler/*", "text_encoder/*",
            "text_encoder_2/*", "vae/*", "transformer/*",
        ],
    )


def _load_safetensors_from_dir(directory: str) -> list[tuple[str, torch.Tensor]]:
    """Load all safetensors from a directory."""
    safetensor_files = sorted(glob(os.path.join(directory, "*.safetensors")))
    weights = []
    for sf_file in safetensor_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for key in f.keys():
                weights.append((key, f.get_tensor(key)))
    return weights


def _detect_diffusion_type(model_path: str) -> str:
    """Detect which diffusion architecture a model uses."""
    index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = json.load(f)
        class_name = data.get("_class_name", "")
        if "HunyuanVideo" in class_name:
            return "hunyuan_video"
    return "flux"


def _load_flux_weights(pipeline: FluxPipeline, model_path: str) -> None:
    """Load transformer and T5 encoder weights for FLUX."""
    transformer_dir = os.path.join(model_path, "transformer")
    weights = _load_safetensors_from_dir(transformer_dir)
    if not weights:
        weights = _load_safetensors_from_dir(model_path)
    if not weights:
        raise FileNotFoundError(
            f"No .safetensors files found in {transformer_dir} or {model_path}"
        )
    loaded = pipeline.transformer.load_weights(weights)
    logger.info("Loaded %d transformer weight entries", len(loaded))

    t5_weights = _load_safetensors_from_dir(
        os.path.join(model_path, "text_encoder_2"),
    )
    if t5_weights:
        loaded_t5 = pipeline.text_encoder_2.load_weights(t5_weights)
        logger.info("Loaded %d T5 encoder weight entries", len(loaded_t5))


def _load_hunyuan_video_weights(pipeline: HunyuanVideoPipeline, model_path: str) -> None:
    """Load transformer and T5 encoder weights for HunyuanVideo."""
    transformer_dir = os.path.join(model_path, "transformer")
    weights = _load_safetensors_from_dir(transformer_dir)
    if not weights:
        weights = _load_safetensors_from_dir(model_path)
    if not weights:
        raise FileNotFoundError(
            f"No .safetensors files found in {transformer_dir} or {model_path}"
        )
    loaded = pipeline.transformer.load_weights(weights)
    logger.info("Loaded %d transformer weight entries", len(loaded))

    t5_weights = _load_safetensors_from_dir(
        os.path.join(model_path, "text_encoder_2"),
    )
    if t5_weights:
        loaded_t5 = pipeline.text_encoder_2.load_weights(t5_weights)
        logger.info("Loaded %d T5 encoder weight entries", len(loaded_t5))


class DiffusionEngine:
    """Engine for running diffusion inference (FLUX and HunyuanVideo).

    Handles model loading, device placement, optional torch.compile,
    and deterministic generation via seeded generators.
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enforce_eager: bool = False,
    ):
        self.model_name = model_name
        self.seed = seed
        self.dtype = dtype
        self.device_str = device
        self.device = torch.device(device)
        self.enforce_eager = enforce_eager
        self._pipeline: FluxPipeline | HunyuanVideoPipeline | None = None
        self._model_type: str | None = None

    def _get_pipeline(self) -> FluxPipeline | HunyuanVideoPipeline:
        if self._pipeline is not None:
            return self._pipeline

        model_path = _download_diffusion_model(self.model_name)
        self._model_type = _detect_diffusion_type(model_path)

        if self._model_type == "hunyuan_video":
            logger.info("Loading HunyuanVideo model: %s", self.model_name)
            config = HunyuanVideoConfig.from_pretrained(model_path)
            pipeline = HunyuanVideoPipeline(config, model_path)

            _load_hunyuan_video_weights(pipeline, model_path)

            pipeline.transformer.to(device=self.device, dtype=self.dtype)
            pipeline.text_encoder.to(device=self.device, dtype=self.dtype)
            pipeline.text_encoder_2.to(device=self.device, dtype=self.dtype)
            pipeline.vae.to(device=self.device)
        else:
            logger.info("Loading FLUX model: %s", self.model_name)
            config = FluxConfig.from_pretrained(model_path)
            pipeline = FluxPipeline(config, model_path)

            _load_flux_weights(pipeline, model_path)

            pipeline.transformer.to(device=self.device, dtype=self.dtype)
            pipeline.text_encoder.to(device=self.device)
            pipeline.text_encoder_2.to(device=self.device, dtype=self.dtype)
            pipeline.vae.to(device=self.device)

        is_diffusion_model = self._model_type in ("flux", "hunyuan_video")
        if not self.enforce_eager and not is_diffusion_model:
            try:
                pipeline.transformer = torch.compile(
                    pipeline.transformer, mode="default",
                )
                logger.info("torch.compile applied to transformer")
            except Exception as e:
                logger.warning("torch.compile failed, using eager: %s", e)
        else:
            logger.info("Running transformer in eager mode")

        pipeline.eval()
        self._pipeline = pipeline
        return pipeline

    def generate(
        self,
        prompts: str | list[str],
        params: DiffusionSamplingParams | HunyuanVideoDiffusionSamplingParams | None = None,
    ) -> DiffusionOutput | HunyuanVideoDiffusionOutput:
        """Generate images/video from text prompts."""
        pipeline = self._get_pipeline()

        if self._model_type == "hunyuan_video":
            params = params or HunyuanVideoDiffusionSamplingParams()
            seed = params.seed if params.seed is not None else self.seed
            generator = torch.Generator(device=self.device).manual_seed(seed)
            return pipeline.forward(prompts, params, generator=generator)
        else:
            params = params or DiffusionSamplingParams()
            seed = params.seed if params.seed is not None else self.seed
            generator = torch.Generator(device=self.device).manual_seed(seed)
            return pipeline.forward(prompts, params, generator=generator)

    def warmup(self, num_steps: int = 2) -> None:
        """Run a small warmup generation to prime CUDA graphs / compile."""
        if self._model_type == "hunyuan_video" or "hunyuan" in self.model_name.lower():
            params = HunyuanVideoDiffusionSamplingParams(
                height=256, width=256, num_frames=5,
                num_inference_steps=num_steps,
                output_type="latent",
            )
            self.generate("warmup", params)
        else:
            params = DiffusionSamplingParams(
                height=256, width=256,
                num_inference_steps=num_steps,
                output_type="latent",
            )
            self.generate("warmup", params)

    def _cleanup(self) -> None:
        """Release GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        torch.cuda.empty_cache()
