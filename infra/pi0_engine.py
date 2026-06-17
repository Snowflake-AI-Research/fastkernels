"""Pi0 inference engine for vision-language-action models.

Downloads the checkpoint, loads weights into the Pi0Pipeline, and provides
a generate() API for flow-matching action generation.

Mirrors the DiffusionEngine pattern but specialized for robotics VLA models.
"""

from __future__ import annotations

import logging
import os
import time
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from ..tasks.baseline.L4.pi0 import (
    Pi0Config,
    Pi0Output,
    Pi0Pipeline,
    Pi0SamplingParams,
)

logger = logging.getLogger(__name__)


def _download_pi0_model(model_name: str) -> str:
    """Download Pi0 model weights from HuggingFace, or use a local directory."""
    if os.path.isdir(model_name):
        return model_name
    return snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )


def _load_safetensors(model_path: str) -> list[tuple[str, torch.Tensor]]:
    """Load all safetensors files from a directory."""
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        safetensor_files = sorted(glob(os.path.join(model_path, "**", "*.safetensors"), recursive=True))
    weights = []
    for sf_file in safetensor_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for key in f.keys():
                weights.append((key, f.get_tensor(key)))
    return weights


class Pi0Engine:
    """Engine for running Pi0 inference.

    Handles model download, weight loading, device placement, and
    optional torch.compile.
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
        self._pipeline: Pi0Pipeline | None = None

    def _get_pipeline(self) -> Pi0Pipeline:
        if self._pipeline is not None:
            return self._pipeline

        model_path = _download_pi0_model(self.model_name)
        logger.info("Loading Pi0 model: %s", self.model_name)

        config = Pi0Config.from_pretrained(model_path)
        pipeline = Pi0Pipeline(config)

        weights = _load_safetensors(model_path)
        if not weights:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}"
            )

        loaded = pipeline.load_weights(weights)
        logger.info("Loaded %d weight entries", len(loaded))

        pipeline.to(device=self.device, dtype=self.dtype)
        pipeline.eval()

        if not self.enforce_eager:
            try:
                pipeline.model.dit = torch.compile(
                    pipeline.model.dit, mode="reduce-overhead",
                )
                logger.info("torch.compile applied to DiT action expert")
            except Exception as e:
                logger.warning("torch.compile failed, using eager: %s", e)

        self._pipeline = pipeline
        return pipeline

    def generate(
        self,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        params: Pi0SamplingParams | None = None,
        noise: torch.Tensor | None = None,
    ) -> Pi0Output:
        """Generate action chunk from observation.

        Args:
            state: (batch, max_state_dim) robot state.
            input_ids: (batch, seq_len) tokenized instruction.
            pixel_values: (batch, num_cameras, 3, H, W) images.
            pixel_attention_mask: (batch, num_cameras) bool mask.
            attention_mask: (batch, seq_len) text mask.
            params: Sampling parameters.
            noise: Optional pre-generated noise for reproducibility.

        Returns:
            Pi0Output with action chunk and timing info.
        """
        pipeline = self._get_pipeline()
        params = params or Pi0SamplingParams()

        if noise is None:
            seed = params.seed if params.seed is not None else self.seed
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(
                state.shape[0],
                pipeline.config.chunk_size,
                pipeline.config.max_action_dim,
                generator=generator,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            noise = noise.to(device=self.device, dtype=self.dtype)

        state = state.to(device=self.device, dtype=self.dtype)
        input_ids = input_ids.to(device=self.device)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        pixel_attention_mask = pixel_attention_mask.to(device=self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)

        return pipeline(
            state=state,
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            attention_mask=attention_mask,
            noise=noise,
            params=params,
        )

    def warmup(self, num_steps: int = 2) -> None:
        """Run a small warmup inference to prime CUDA / compile."""
        pipeline = self._get_pipeline()
        config = pipeline.config
        bsz = 1
        device = self.device
        dtype = self.dtype

        state = torch.zeros(bsz, config.max_state_dim, device=device, dtype=dtype)
        input_ids = torch.ones(bsz, 5, device=device, dtype=torch.long)
        num_cameras = 1
        pixel_values = torch.zeros(
            bsz, num_cameras, 3,
            config.vlm_vision_config.image_size,
            config.vlm_vision_config.image_size,
            device=device, dtype=dtype,
        )
        pixel_attention_mask = torch.ones(
            bsz, num_cameras, device=device, dtype=torch.bool,
        )

        params = Pi0SamplingParams(num_inference_steps=num_steps)
        self.generate(state, input_ids, pixel_values, pixel_attention_mask, params=params)

    def _cleanup(self) -> None:
        """Release GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        torch.cuda.empty_cache()
