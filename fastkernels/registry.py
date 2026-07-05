"""Architecture registry aligned with table.tex and fastkernels baseline tasks."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .workloads import (
    LLM, VLM, Diffusion, Detection, Embedding, StructurePrediction, WorldModel,
    Workload,
)

@dataclass(frozen=True)
class Family:
    keyword: str
    display_name: str

@dataclass(frozen=True)
class Architecture:
    module: str
    family: str
    class_name: str
    model_type: str | None = None

# keyword -> full display name
_FAMILIES = (
    Family("llm", "Dense & MoE LLMs"),
    Family("linear_attn", "Linear Attention & New Archs"),
    Family("vision", "Vision / Video / Audio"),
    Family("multimodal", "Multimodal & Encoders"),
    Family("edge", "Edge & Detection"),
    Family("3d_robotics", "3D / Robotics / Science"),
    Family("recsys", "Recommendation & Specialized"),
    Family("world_models", "World Models"),
)
FAMILIES: dict[str, Family] = {f.keyword: f for f in _FAMILIES}

# L4 module stem -> Architecture
_ARCHITECTURES = (
    Architecture("llama", "llm", "Llama-3.1+", "llama"),
    Architecture("deepseek", "llm", "DeepSeek-V3.2", "deepseek_v32"),
    Architecture("mixtral", "llm", "Mixtral", "mixtral"),
    Architecture("bitnet", "llm", "BitNet 1.58b", "llama"),
    Architecture("gpt_oss", "llm", "GPT-OSS (MXFP4)", "gpt_oss"),
    Architecture("llama_eagle3", "llm", "EAGLE-3", "llama"),
    Architecture("gemma4", "llm", "Gemma-4", "gemma4"),
    Architecture("mamba", "linear_attn", "Mamba", "mamba"),
    Architecture("mamba2", "linear_attn", "Mamba2", "mamba2"),
    Architecture("rwkv7", "linear_attn", "RWKV-7", "rwkv7"),
    Architecture("gla", "linear_attn", "GLA", "gla"),
    Architecture("retnet", "linear_attn", "RetNet", "retnet"),
    Architecture("qwen3_next", "linear_attn", "Qwen-3-Next", "qwen3_next"),
    Architecture("kimi_linear", "linear_attn", "Kimi-Linear", "kimi_linear"),
    Architecture("ttt_e2e", "linear_attn", "TTT-E2E", None),
    Architecture("jamba", "linear_attn", "Jamba", "jamba"),
    Architecture("flux", "vision", "FLUX.1-Dev", None),
    Architecture("hunyuan_video", "vision", "HunyuanVideo-1.5", None),
    Architecture("sdxl", "vision", "SDXL", None),
    Architecture("sam3", "vision", "SAM3.1", "sam3_video"),
    Architecture("whisper", "vision", "Whisper", "whisper"),
    Architecture("cosyvoice3", "vision", "CosyVoice3", "cosyvoice3"),
    Architecture("qwen2_vl", "multimodal", "Qwen2-VL", "qwen2_vl"),
    Architecture("qwen3_vl", "multimodal", "Qwen3-VL", "qwen3_vl_moe"),
    Architecture("qwen2_5_omni", "multimodal", "Qwen-2.5-Omni", "qwen2_5_omni"),
    Architecture("siglip2", "multimodal", "SigLIP-2", "siglip2"),
    Architecture("dinov3", "multimodal", "DINOv3", "dinov3_vit"),
    Architecture("swinv2", "multimodal", "SwinV2", "swinv2"),
    Architecture("mobilenetv4", "edge", "MobileNetV4", None),
    Architecture("convnextv2", "edge", "ConvNeXtV2", "convnextv2"),
    Architecture("efficientnetv2", "edge", "EfficientNetV2", None),
    Architecture("yolov10", "edge", "YOLOv10", None),
    Architecture("rtdetrv2", "edge", "RTDetrV2", "rt_detr_v2"),
    Architecture("gaussian_splatting", "3d_robotics", "3DGS", None),
    Architecture("instant_ngp", "3d_robotics", "InstantNGP", None),
    Architecture("pointtransformerv3", "3d_robotics", "PointTransformerV3", None),
    Architecture("openfold3", "3d_robotics", "OpenFold3", None),
    Architecture("pi0", "3d_robotics", "Pi0", "pi0"),
    Architecture("dp3", "3d_robotics", "DP3", None),
    Architecture("dlrmv2", "recsys", "DLRMv2", None),
    Architecture("lightgcn", "recsys", "LightGCN", None),
    Architecture("bge_m3", "recsys", "BGE-M3", "xlm-roberta"),
    Architecture("colbertv2", "recsys", "ColBERTv2", "bert"),
    Architecture("llada", "recsys", "LLaDA", "llada"),
    Architecture("oasis", "world_models", "Oasis", None),
    Architecture("vjepa2", "world_models", "V-JEPA 2", "vjepa2"),
)
FASTKERNELS_ARCHITECTURES: dict[str, Architecture] = {a.module: a for a in _ARCHITECTURES}


def _normalize(text: str) -> str:
    """Lowercase, alphanumeric-only form for tolerant name matching."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


@functools.lru_cache(maxsize=None)
def _module_from_name(hf_name: str) -> str | None:
    """Infer the L4 module from the HF name alone -- no config, no network.

    Matches the normalized model name against each architecture's module stem
    and display name, preferring the most specific (longest) match, e.g.
    ``black-forest-labs/FLUX.1-dev`` -> ``flux`` and ``fla-hub/gla-2.7B-100B``
    -> ``gla``. Returns ``None`` when nothing matches.
    """
    norm = _normalize(hf_name)
    best_module: str | None = None
    best_len = 0
    for arch in _ARCHITECTURES:
        for token in (arch.module, arch.class_name):
            key = _normalize(token)
            if len(key) >= 3 and key in norm and len(key) > best_len:
                best_module, best_len = arch.module, len(key)
    return best_module


@functools.lru_cache(maxsize=None)
def module_for(hf_name: str) -> str | None:
    """Infer the fastkernels L4 module stem for a HuggingFace model.

    Prefers the authoritative ``model_type`` from the model's config, read via
    ``transformers`` (which, for uncached models, requires network access).
    Several architectures can share a ``model_type`` (e.g. Llama, BitNet and
    EAGLE-3 all report ``"llama"``), in which case the first registered one wins.

    Falls back to a name-based match against the registered architectures only
    when the config cannot be loaded (offline, gated, or a non-transformers
    pipeline such as a diffusers model) or its ``model_type`` is unknown. This
    keeps diffusers pipelines and custom repos (e.g. FLUX, YOLOv10, OpenFold3,
    Oasis) resolvable. Returns ``None`` when neither the config nor the name
    resolves.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", None)
    except Exception:
        model_type = None
    if model_type is not None:
        for arch in _ARCHITECTURES:
            if arch.model_type == model_type:
                return arch.module
    return _module_from_name(hf_name)


@dataclass(frozen=True)
class BenchmarkScenario:
    hf_name: str
    tp: int
    dtype: str
    workloads: list[Workload]
    num_requests: int | None = None
    enforce_eager: bool = False
    max_num_seqs: int | None = None

# Benchmarking workload registry derived from bench.yaml
DEFAULT_BENCHMARK: list[BenchmarkScenario] = [
    BenchmarkScenario("meta-llama/Llama-3.1-8B-Instruct", 1, "bfloat16", [LLM.prefill_heavy, LLM.decode_heavy], 100),
    BenchmarkScenario("openai/gpt-oss-120b", 2, "mxfp4", [LLM.prefill_heavy, LLM.decode_heavy], 100),
    BenchmarkScenario("fla-hub/gla-2.7B-100B", 1, "bfloat16", [LLM.prefill_heavy, LLM.decode_heavy], 100),
    BenchmarkScenario("black-forest-labs/FLUX.1-dev", 1, "bfloat16", [Diffusion.res_1024, Diffusion.res_512]),
    BenchmarkScenario("Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", 4, "fp8", [VLM.text_only, VLM.image, VLM.video], 100),
    BenchmarkScenario("jameslahm/yolov10n", 1, "bfloat16", [Detection.coco_val]),
    BenchmarkScenario("OpenFold/OpenFold3", 1, "bfloat16", [StructurePrediction.short, StructurePrediction.medium, StructurePrediction.long, StructurePrediction.extra_long]),
    BenchmarkScenario("BAAI/bge-m3", 1, "bfloat16", [Embedding.bge_m3_mldr_docs]),
    BenchmarkScenario("Etched/oasis-500m", 1, "float16", [WorldModel.short_bs4_16f_4ddim, WorldModel.medium_bs8_24f_4ddim, WorldModel.long_bs8_32f_4ddim, WorldModel.denoise_bs4_16f_8ddim]),
]