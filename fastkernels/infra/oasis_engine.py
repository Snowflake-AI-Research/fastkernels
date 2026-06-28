"""Thin engine for Oasis autoregressive diffusion rollout."""

from __future__ import annotations

import logging

import torch
from huggingface_hub import snapshot_download

from ..tasks.baseline.L4.oasis import OasisConfig, OasisOutput, OasisPipeline, OasisSamplingParams

logger = logging.getLogger(__name__)


def _download_oasis_model(model_name: str) -> str:
    return snapshot_download(
        model_name,
        allow_patterns=[
            "*.safetensors",
            "README.md",
            "LICENSE",
        ],
    )


class OasisEngine:
    def __init__(
        self,
        model_name: str = "Etched/oasis-500m",
        *,
        seed: int = 0,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.seed = seed
        self.dtype = dtype
        self.device = torch.device(device)
        self._pipeline: OasisPipeline | None = None

    def _get_pipeline(self) -> OasisPipeline:
        if self._pipeline is not None:
            return self._pipeline
        model_dir = _download_oasis_model(self.model_name)
        pipeline = OasisPipeline(OasisConfig())
        pipeline.load_weights(model_dir)
        pipeline.model.to(device=self.device, dtype=self.dtype)
        pipeline.vae.to(device=self.device, dtype=self.dtype)
        pipeline.eval()
        self._pipeline = pipeline
        return pipeline

    def generate(
        self,
        prompt: torch.Tensor,
        actions: torch.Tensor,
        params: OasisSamplingParams | None = None,
    ) -> OasisOutput:
        params = params or OasisSamplingParams(seed=self.seed)
        if params.seed is None:
            params.seed = self.seed
        pipeline = self._get_pipeline()
        return pipeline.rollout(
            prompt.to(device=self.device),
            actions.to(device=self.device),
            params,
            dtype=self.dtype,
        )

    def warmup(self) -> None:
        prompt = torch.rand(1, 1, 3, 360, 640, device=self.device)
        actions = torch.zeros(1, 4, 25, device=self.device)
        self.generate(prompt, actions, OasisSamplingParams(num_frames=4, ddim_steps=2, seed=self.seed))
