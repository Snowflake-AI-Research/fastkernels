"""Eval configuration for Tier 3 evaluation sweep."""

from __future__ import annotations

from dataclasses import dataclass, field


MODEL_KEY_TO_DEFAULT_HF: dict[str, str] = {
    "gpt_oss": "openai/gpt-oss-20b",
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2_vl": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen3_vl": "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "flux": "black-forest-labs/FLUX.1-dev",
    "sam3": "facebook/sam3.1",
    "cosyvoice3": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "hunyuan_video": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "openfold3": "OpenFold/OpenFold3",
}

MODEL_CATEGORY: dict[str, str] = {
    "openai/gpt-oss-20b": "llm",
    "openai/gpt-oss-120b": "llm",
    "meta-llama/Llama-3.1-8B-Instruct": "llm",
    "meta-llama/Llama-3.1-70B-Instruct": "llm",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llm",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "llm",
    "Qwen/Qwen2-VL-7B-Instruct": "vlm",
    "Qwen/Qwen3-VL-8B-Instruct-FP8": "vlm",
    "black-forest-labs/FLUX.1-dev": "diffusion",
    "facebook/sam3.1": "segmentation",
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512": "tts",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "diffusion",
    "OpenFold/OpenFold3": "structure_prediction",
}


@dataclass
class EvalConfig:
    """Configuration for the eval sweep.

    The experiments use fixed standardized workloads (3 throughput + 2 latency).
    Filtering is only by model, TP degree, or category.
    """
    models: list[str] | None = None
    tp_degrees: list[int] = field(default_factory=lambda: [1, 4])
    categories: list[str] | None = None
    seed: int = 42
    temperature: float = 0.0
    enforce_eager: bool = False
    output_json: str = ""  # resolved at runtime by __main__.py
    num_prompts: int = 1000

    def get_model_category(self, model: str) -> str:
        """Return the category for a model (default: 'llm')."""
        return MODEL_CATEGORY.get(model, "llm")
