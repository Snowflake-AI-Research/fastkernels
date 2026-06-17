"""
SDXL inference engine for UNet-based diffusion models.

Handles model loading, device placement, optional torch.compile,
and deterministic generation via seeded generators.
"""

from __future__ import annotations

import logging
import os
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from ..tasks.baseline.L4.sdxl import (
    SDXLConfig,
    SDXLOutput,
    SDXLPipeline,
    SDXLSamplingParams,
)

logger = logging.getLogger(__name__)


def _download_sdxl_model(model_name: str) -> str:
    return snapshot_download(
        model_name,
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.model",
            "tokenizer*", "tokenizer_2/*", "scheduler/*",
            "text_encoder/*", "text_encoder_2/*", "vae/*", "unet/*",
        ],
    )


def _load_unet_weights(
    pipeline: SDXLPipeline, model_path: str, variant: str = "fp16",
) -> None:
    unet_dir = os.path.join(model_path, "unet")

    # Prefer the variant file (e.g. diffusion_pytorch_model.fp16.safetensors)
    # to match the diffusers loading path and avoid double-loading.
    variant_files = sorted(glob(os.path.join(unet_dir, f"*.{variant}.safetensors")))
    if not variant_files:
        variant_files = sorted(glob(os.path.join(unet_dir, "*.safetensors")))
    if not variant_files:
        variant_files = sorted(glob(os.path.join(model_path, f"*.{variant}.safetensors")))
    if not variant_files:
        variant_files = sorted(glob(os.path.join(model_path, "*.safetensors")))

    if not variant_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {unet_dir} or {model_path}"
        )

    weights = []
    for sf_file in variant_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for key in f.keys():
                weights.append((key, f.get_tensor(key)))

    loaded = pipeline.unet.load_weights(weights)
    logger.info("Loaded %d UNet weight entries from %s", len(loaded), [os.path.basename(f) for f in variant_files])


class SDXLEngine:
    """Engine for running SDXL diffusion inference."""

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
        self._pipeline: SDXLPipeline | None = None

    def _get_pipeline(self) -> SDXLPipeline:
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading SDXL model: %s", self.model_name)
        model_path = _download_sdxl_model(self.model_name)

        config = SDXLConfig.from_pretrained(model_path)
        pipeline = SDXLPipeline(
            config, model_path,
            torch_dtype=self.dtype, variant="fp16",
        )

        _load_unet_weights(pipeline, model_path, variant="fp16")

        pipeline.to(device=self.device, dtype=self.dtype)

        if not self.enforce_eager:
            try:
                pipeline.unet = torch.compile(
                    pipeline.unet, mode="max-autotune",
                )
                logger.info("torch.compile(mode='max-autotune') applied to UNet")
            except Exception as e:
                logger.warning("torch.compile failed, using eager: %s", e)

        pipeline.eval()
        self._pipeline = pipeline
        return pipeline

    def generate(
        self,
        prompts: str | list[str],
        params: SDXLSamplingParams | None = None,
    ) -> SDXLOutput:
        pipeline = self._get_pipeline()
        params = params or SDXLSamplingParams()

        seed = params.seed if params.seed is not None else self.seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return pipeline.forward(prompts, params, generator=generator)

    def warmup(self, num_steps: int = 2) -> None:
        params = SDXLSamplingParams(
            height=256, width=256,
            num_inference_steps=num_steps,
            output_type="latent",
        )
        self.generate("warmup", params)

    def _cleanup(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        torch.cuda.empty_cache()
