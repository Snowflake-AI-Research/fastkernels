"""Architecture registry aligned with table.tex and fastkernels baseline tasks."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .workloads import (
    ASR, LLM, TTS, VLM, Detection, Diffusion, Embedding, OmniModal,
    PointCloudPolicy, PointCloudSeg, Purpose, Recsys, Rendering, Robotics,
    Segmentation, StructurePrediction, VideoDiffusion, VideoRepresentation,
    VisionEncoder, Workload, WorkloadSpec, WorldModel, purpose_of, spec_for,
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
    """A model to benchmark, its per-scenario engine config and its workloads.

    ``enforce_eager`` and ``max_num_seqs`` are *engine* knobs that apply to the
    whole scenario (every one of its workloads). Per-*workload* parameters --
    request counts, sequence lengths, and the throughput/latency purpose -- live
    on the workload spec (``spec_for(workload)`` in ``workloads.py``), which is
    why there is no scenario-level ``num_requests``.

    ``workloads`` mixes throughput and latency workloads; use the
    ``throughput_workloads`` / ``latency_workloads`` splits to iterate one kind.
    """
    hf_name: str
    tp: int
    dtype: str
    workloads: list[Workload]
    enforce_eager: bool = False
    max_num_seqs: int | None = None

    @property
    def specs(self) -> list[WorkloadSpec]:
        """This scenario's workloads resolved to purpose + parameter specs."""
        return [spec_for(w) for w in self.workloads]

    @property
    def throughput_workloads(self) -> list[Workload]:
        return [w for w in self.workloads if purpose_of(w) is Purpose.THROUGHPUT]

    @property
    def latency_workloads(self) -> list[Workload]:
        return [w for w in self.workloads if purpose_of(w) is Purpose.LATENCY]

# Benchmarking workload registry derived from bench.yaml
DEFAULT_BENCHMARK: list[BenchmarkScenario] = [
    BenchmarkScenario("meta-llama/Llama-3.1-8B-Instruct", 1, "bfloat16", [LLM.mixed, LLM.long_context]),
    BenchmarkScenario("openai/gpt-oss-120b", 2, "mxfp4", [LLM.mixed, LLM.long_context]),
    BenchmarkScenario("fla-hub/gla-2.7B-100B", 1, "bfloat16", [LLM.mixed, LLM.long_context]),
    BenchmarkScenario("black-forest-labs/FLUX.1-dev", 1, "bfloat16", [Diffusion.res_1024, Diffusion.res_512]),
    BenchmarkScenario("Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", 4, "fp8", [VLM.text_only, VLM.image, VLM.video]),
    BenchmarkScenario("jameslahm/yolov10n", 1, "bfloat16", [Detection.coco_val]),
    BenchmarkScenario("OpenFold/OpenFold3", 1, "bfloat16", [StructurePrediction.short, StructurePrediction.medium, StructurePrediction.long, StructurePrediction.extra_long]),
    BenchmarkScenario("BAAI/bge-m3", 1, "bfloat16", [Embedding.bge_m3_mldr_docs]),
    BenchmarkScenario("Etched/oasis-500m", 1, "float16", [WorldModel.short_bs4_16f_4ddim, WorldModel.medium_bs8_24f_4ddim, WorldModel.long_bs8_32f_4ddim, WorldModel.denoise_bs4_16f_8ddim]),
]

# Full benchmark: every architecture in table.tex, each with the complete set
# of workloads for its modality. dtype follows the table's default-inference
# dtype (BF16->bfloat16, FP16->float16, FP32->float32, MXFP4->mxfp4, BitNet's
# W1.58 activations run in bfloat16). ``tp`` is sized to the checkpoint: dense
# <=8B and small vision/audio/science models run at tp=1; larger dense/MoE and
# 100B+ (MoE) models fan out to tp 2/4/8. Per-workload request counts and
# sequence lengths come from the workload specs in ``workloads.py`` (e.g. the
# LLM mixed=1000 / long-context=64 curated datasets), not from this table.
_LLM_ALL = [LLM.mixed, LLM.long_context, LLM.single_request, LLM.fixed_batch_32]
_VLM_ALL = [VLM.text_only, VLM.image, VLM.video, VLM.single_image, VLM.single_video]
_OMNI_ALL = [OmniModal.text, OmniModal.image, OmniModal.video, OmniModal.audio, OmniModal.single_text, OmniModal.single_image, OmniModal.single_video, OmniModal.single_audio]
_ASR_ALL = [ASR.librispeech, ASR.single_utterance, ASR.fixed_batch_32]
_TTS_ALL = [TTS.tts_short, TTS.tts_medium, TTS.tts_long, TTS.single_utterance]
_DIFFUSION_ALL = [Diffusion.res_1024, Diffusion.res_512, Diffusion.single_1024, Diffusion.single_512]
_VIDEO_DIFFUSION_ALL = [VideoDiffusion.p480_short, VideoDiffusion.p480_medium, VideoDiffusion.single_480p_short, VideoDiffusion.single_480p_medium]
_SEGMENTATION_ALL = [Segmentation.gold_metaclip_nps, Segmentation.gold_wiki_common, Segmentation.gold_crowded, Segmentation.veval_sav_val, Segmentation.veval_yt1b_val, Segmentation.sav_val_video, Segmentation.smartglasses_val_video, Segmentation.single_image_1008, Segmentation.batch_4_image_1008, Segmentation.single_video_frame_1008]
_DETECTION_ALL = [Detection.coco_val, Detection.single_image, Detection.batch_4]
_VISION_ENCODER_ALL = [VisionEncoder.default_res, VisionEncoder.high_res, VisionEncoder.single_image, VisionEncoder.batch_8]
_STRUCTURE_ALL = [StructurePrediction.short, StructurePrediction.medium, StructurePrediction.long, StructurePrediction.extra_long, StructurePrediction.single_short, StructurePrediction.single_medium, StructurePrediction.single_long, StructurePrediction.single_extra_long]
_ROBOTICS_ALL = [Robotics.libero_1cam, Robotics.libero_3cam, Robotics.single_3cam, Robotics.single_1cam]
_DP3_ALL = [PointCloudPolicy.dp3_1env, PointCloudPolicy.dp3_batch, PointCloudPolicy.single_step, PointCloudPolicy.batch_8]
_PTV3_ALL = [PointCloudSeg.scanobjectnn, PointCloudSeg.single_cloud, PointCloudSeg.batch_8]
_RENDERING_ALL = [Rendering.render, Rendering.single_render]
_WORLD_MODEL_ALL = [WorldModel.short_bs4_16f_4ddim, WorldModel.medium_bs8_24f_4ddim, WorldModel.long_bs8_32f_4ddim, WorldModel.denoise_bs4_16f_8ddim, WorldModel.latency_bs1_8f_4ddim]
_VJEPA_ALL = [VideoRepresentation.predictor, VideoRepresentation.encoder, VideoRepresentation.classification, VideoRepresentation.single_video]

FULL_BENCHMARK: list[BenchmarkScenario] = [
    # --- Dense & MoE LLMs ---
    BenchmarkScenario("meta-llama/Llama-3.1-8B-Instruct", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("deepseek-ai/DeepSeek-V3.2-Exp", 8, "fp8", _LLM_ALL),
    BenchmarkScenario("mistralai/Mixtral-8x7B-v0.1", 2, "bfloat16", _LLM_ALL),
    BenchmarkScenario("1bitLLM/bitnet_b1_58-3B", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("openai/gpt-oss-120b", 2, "mxfp4", _LLM_ALL),
    BenchmarkScenario("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("google/gemma-4-26B-A4B-it", 1, "bfloat16", _LLM_ALL),
    # --- Linear Attention & New Archs ---
    BenchmarkScenario("state-spaces/mamba-2.8b-hf", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("mistralai/Mamba-Codestral-7B-v0.1", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("fla-hub/rwkv7-1.5B-world", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("fla-hub/gla-2.7B-100B", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("fla-hub/retnet-2.7B-100B", 1, "bfloat16", _LLM_ALL),
    BenchmarkScenario("Qwen/Qwen3-Next-80B-A3B-Instruct", 2, "bfloat16", _LLM_ALL),
    BenchmarkScenario("moonshotai/Kimi-Linear-48B-A3B-Instruct", 2, "bfloat16", _LLM_ALL),
    BenchmarkScenario("ttt_e2e", 1, "bfloat16", _LLM_ALL),                      # TTT-E2E (Project Gutenberg contiguous text; random-init weights, no public ckpt)
    BenchmarkScenario("ai21labs/AI21-Jamba-Mini-1.7", 2, "bfloat16", _LLM_ALL),
    # --- Vision / Video / Audio ---
    BenchmarkScenario("black-forest-labs/FLUX.1-dev", 1, "bfloat16", _DIFFUSION_ALL),
    BenchmarkScenario("tencent/HunyuanVideo-1.5", 1, "bfloat16", _VIDEO_DIFFUSION_ALL),
    BenchmarkScenario("stabilityai/stable-diffusion-xl-base-1.0", 1, "float16", _DIFFUSION_ALL),
    BenchmarkScenario("facebook/sam3", 1, "bfloat16", _SEGMENTATION_ALL),
    BenchmarkScenario("openai/whisper-large-v3", 1, "float16", _ASR_ALL),
    BenchmarkScenario("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", 1, "bfloat16", _TTS_ALL),
    # --- Multimodal & Encoders ---
    BenchmarkScenario("Qwen/Qwen2-VL-7B-Instruct", 1, "bfloat16", _VLM_ALL),
    BenchmarkScenario("Qwen/Qwen3-VL-8B-Instruct-FP8", 1, "fp8", _VLM_ALL),
    BenchmarkScenario("Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", 4, "fp8", _VLM_ALL),
    BenchmarkScenario("Qwen/Qwen2.5-Omni-7B", 1, "bfloat16", _OMNI_ALL),
    BenchmarkScenario("google/siglip2-so400m-patch16-naflex", 1, "bfloat16", _VISION_ENCODER_ALL),
    BenchmarkScenario("facebook/dinov3-vit7b16-pretrain-lvd1689m", 1, "bfloat16", _VISION_ENCODER_ALL),
    BenchmarkScenario("microsoft/swinv2-large-patch4-window12-192-22k", 1, "float32", _VISION_ENCODER_ALL),
    # --- Edge & Detection ---
    BenchmarkScenario("timm/mobilenetv4_conv_medium.e500_r256_in1k", 1, "float32", _VISION_ENCODER_ALL),
    BenchmarkScenario("facebook/convnextv2-base-22k-384", 1, "float32", _VISION_ENCODER_ALL),
    BenchmarkScenario("timm/efficientnetv2_rw_m.agc_in1k", 1, "float32", _VISION_ENCODER_ALL),
    BenchmarkScenario("jameslahm/yolov10n", 1, "float16", _DETECTION_ALL),
    BenchmarkScenario("PekingU/rtdetr_v2_r101vd", 1, "float32", _DETECTION_ALL),
    # --- 3D / Robotics / Science ---
    BenchmarkScenario("gaussian_splatting", 1, "float32", _RENDERING_ALL),      # 3DGS (Voxel51/gaussian_splatting scene)
    BenchmarkScenario("instant_ngp", 1, "float16", _RENDERING_ALL),             # InstantNGP (Instant-NGP fox scene)
    BenchmarkScenario("Pointcept/PointTransformerV3", 1, "float32", _PTV3_ALL),
    BenchmarkScenario("OpenFold/OpenFold3", 1, "bfloat16", _STRUCTURE_ALL),
    BenchmarkScenario("lerobot/pi0_base", 1, "bfloat16", _ROBOTICS_ALL),
    BenchmarkScenario("dp3", 1, "float32", _DP3_ALL),                           # DP3 (rishabhrj11/gym-xarm-pointcloud)
    # --- Recommendation & Specialized ---
    BenchmarkScenario("dlrmv2", 1, "float32", [Recsys.ctr_batch, Recsys.single_request, Recsys.fixed_batch_32]),          # DLRMv2 (Criteo 1TB not public -> scikit-learn/adult-census-income substitute)
    BenchmarkScenario("lightgcn", 1, "float32", [Recsys.recommend_batch, Recsys.single_request, Recsys.fixed_batch_32]),  # LightGCN (GroupLens MovieLens-1M)
    BenchmarkScenario("BAAI/bge-m3", 1, "float16", [Embedding.bge_m3_mldr_docs, Embedding.single_request, Embedding.fixed_batch_32]),
    BenchmarkScenario("colbert-ir/colbertv2.0", 1, "float16", [Embedding.colbertv2_msmarco_passages, Embedding.single_request, Embedding.fixed_batch_32]),
    BenchmarkScenario("GSAI-ML/LLaDA-8B-Instruct", 1, "bfloat16", _LLM_ALL),
    # --- World Models ---
    BenchmarkScenario("Etched/oasis-500m", 1, "bfloat16", _WORLD_MODEL_ALL),
    BenchmarkScenario("facebook/vjepa2-vitl-fpc64-256", 1, "bfloat16", _VJEPA_ALL),
]