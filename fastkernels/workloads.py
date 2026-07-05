"""Benchmark workloads: identities, parameters, prompts and dataset adapter.

Single source of truth for everything workload-related. It holds:

* the per-family workload *identity* enums (e.g. ``LLM.prefill_heavy``) used in
  ``registry.py``;
* the per-workload throughput/latency *parameter* specs and model configs
  consumed by the eval pipeline and the ``tests/`` benchmarks;
* the real chat-prompt loader (``load_real_prompt_workload``) used to run the
  LLM prefill-heavy / balanced / decode-heavy workloads; and
* a thin, lazily-imported adapter over vLLM's dataset infrastructure.

Kept import-light on purpose: importing this module must not pull in ``vllm``
(the dataset re-exports below are resolved lazily via ``__getattr__``) or the
Hugging Face ``datasets`` package (imported inside the loader), so
``registry.py`` and ``capture.py`` can import it cheaply.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Workload(Enum):
    """Base for the per-family workload enums; value is the canonical name."""

    def __str__(self) -> str:  # so ", ".join(...) and logging read naturally
        return self.value


class LLM(Workload):
    """Text LLMs (also used by linear-attention / hybrid LLMs)."""
    prefill_heavy = "prefill-heavy"
    balanced = "balanced"
    decode_heavy = "decode-heavy"
    single_request = "single-request"
    fixed_batch_32 = "fixed-batch-32"


class VLM(Workload):
    """Vision-language models."""
    text_only = "text-only"
    image = "image"
    video = "video"
    single_image = "single-image"
    single_video = "single-video"


class OmniModal(Workload):
    """Any-to-any models (text + image + video + audio, e.g. Qwen2.5-Omni)."""
    text = "text"
    image = "image"
    video = "video"
    audio = "audio"
    single_text = "single-text"
    single_image = "single-image"
    single_video = "single-video"
    single_audio = "single-audio"


class ASR(Workload):
    """Speech recognition (e.g. Whisper)."""
    librispeech = "librispeech"
    single_utterance = "single-utterance"
    fixed_batch_32 = "fixed-batch-32"


class TTS(Workload):
    """Text-to-speech (e.g. CosyVoice3)."""
    tts_short = "tts-short"
    tts_medium = "tts-medium"
    tts_long = "tts-long"
    single_utterance = "single-utterance"


class Diffusion(Workload):
    """Image generation (e.g. FLUX, SDXL)."""
    res_1024 = "1024x1024"
    res_512 = "512x512"
    single_1024 = "single-1024x1024"
    single_512 = "single-512x512"


class VideoDiffusion(Workload):
    """Text-to-video generation (e.g. HunyuanVideo)."""
    p480_short = "480p-short"
    p480_medium = "480p-medium"
    single_480p_short = "single-480p-short"
    single_480p_medium = "single-480p-medium"


class WorldModel(Workload):
    """Autoregressive video world models (e.g. Oasis)."""
    short_bs4_16f_4ddim = "short-bs4-16f-4ddim"
    medium_bs8_24f_4ddim = "medium-bs8-24f-4ddim"
    long_bs8_32f_4ddim = "long-bs8-32f-4ddim"
    denoise_bs4_16f_8ddim = "denoise-bs4-16f-8ddim"
    latency_bs1_8f_4ddim = "latency-bs1-8f-4ddim"


class Segmentation(Workload):
    """Promptable concept segmentation (e.g. SAM3)."""
    gold_metaclip_nps = "gold-metaclip-nps"
    gold_wiki_common = "gold-wiki-common"
    gold_crowded = "gold-crowded"
    veval_sav_val = "veval-sav-val"
    veval_yt1b_val = "veval-yt1b-val"
    sav_val_video = "sav-val-video"
    smartglasses_val_video = "smartglasses-val-video"
    single_image_1008 = "single-image-1008"
    batch_4_image_1008 = "batch-4-image-1008"
    single_video_frame_1008 = "single-video-frame-1008"


class Detection(Workload):
    """Object detection (e.g. YOLOv10, RT-DETRv2)."""
    coco_val = "coco-val"
    single_image = "single-image"
    batch_4 = "batch-4"


class VisionEncoder(Workload):
    """Pure image feature extraction (e.g. SigLIP-2, DINOv3, SwinV2)."""
    default_res = "default-res"
    high_res = "high-res"
    single_image = "single-image"
    batch_8 = "batch-8"


class Embedding(Workload):
    """Token-level retrieval embeddings (e.g. BGE-M3, ColBERTv2)."""
    bge_m3_mldr_docs = "bge-m3-mldr-docs"
    colbertv2_msmarco_passages = "colbertv2-msmarco-passages"
    single_request = "single-request"
    fixed_batch_32 = "fixed-batch-32"


class StructurePrediction(Workload):
    """Protein structure prediction (e.g. OpenFold3)."""
    short = "short"
    medium = "medium"
    long = "long"
    extra_long = "extra-long"
    single_short = "single-short"
    single_medium = "single-medium"
    single_long = "single-long"
    single_extra_long = "single-extra-long"


class Robotics(Workload):
    """Vision-language-action policies (e.g. Pi0)."""
    libero_1cam = "libero-1cam"
    libero_3cam = "libero-3cam"
    single_3cam = "single-3cam"
    single_1cam = "single-1cam"


class PointCloudPolicy(Workload):
    """3-D point-cloud diffusion policies (e.g. DP3)."""
    dp3_1env = "dp3-1env"
    dp3_batch = "dp3-batch"
    single_step = "single-step"
    batch_8 = "batch-8"


# ===========================================================================
# Real chat-prompt workloads (LLM)
#
# The workload datasets store raw chat messages. Runners call the loader with
# the target model's tokenizer so prompts are chat-templated and tokenized
# exactly as that model expects.
# ===========================================================================

DEFAULT_WORKLOAD_DATASETS: dict[str, str] = {
    "prefill-heavy": "sfc-gh-goliaro/wildchat-fastkernels-prefill-heavy-1k",
    "balanced": "sfc-gh-goliaro/wildchat-fastkernels-balanced-1k",
    "decode-heavy": "sfc-gh-goliaro/wildchat-fastkernels-decode-heavy-1k",
}

LEGACY_WORKLOAD_DATASETS: dict[str, str] = {
    "prefill-heavy": "sfc-gh-goliaro/wildchat-prefill-heavy-1k",
    "balanced": "sfc-gh-goliaro/kb-nano-balanced",
    "decode-heavy": "sfc-gh-goliaro/wildchat-decode-heavy-1k",
}


@dataclass(frozen=True)
class RealPromptSample:
    prompt_token_ids: list[int]
    output_len: int


def _normalize_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    if "user" in row and "assistant" in row:
        messages: list[dict[str, str]] = []
        system = row.get("system") or ""
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": row["user"] or ""})
        return messages, row["assistant"] or ""

    if "messages" in row and row["messages"] is not None:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in row["messages"]
        ]
        assistant_text = row.get("assistant_text")
        if assistant_text is None:
            for message in reversed(messages):
                if message["role"] == "assistant":
                    assistant_text = message["content"]
                    break
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        return messages, assistant_text or ""

    conversations = row.get("conversations")
    if conversations is None:
        raise ValueError("Expected either 'messages' or 'conversations' in row")

    prompt_messages: list[dict[str, str]] = []
    assistant_text = ""
    for turn in conversations:
        role = turn.get("role") or turn.get("from")
        content = turn.get("content") or turn.get("value") or ""
        if role in ("human", "user"):
            prompt_messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            assistant_text = content

    return prompt_messages, assistant_text


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError):
        text = "\n\n".join(m["content"] for m in messages)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(
                "Expected one chat-templated prompt, got a batched input_ids result"
            )
        token_ids = token_ids[0]
    return list(token_ids)


def _tokenize_response(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def load_real_prompt_workload(
    scenario_name: str,
    tokenizer: Any,
    *,
    num_requests: int = 1000,
    decode_cap: int | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    seed: int | None = None,
) -> list[RealPromptSample]:
    """Load, chat-template, and tokenize a real LLM workload.

    ``decode_cap`` caps the per-request generation budget after tokenizing the
    source assistant response with the same tokenizer.
    """
    from datasets import load_dataset

    dataset_id = dataset_name or DEFAULT_WORKLOAD_DATASETS[scenario_name]
    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception:
        legacy_id = LEGACY_WORKLOAD_DATASETS.get(scenario_name)
        if legacy_id is None or legacy_id == dataset_id:
            raise
        ds = load_dataset(legacy_id, split=split)

    if seed is not None:
        ds = ds.shuffle(seed=seed)

    samples: list[RealPromptSample] = []
    for row in ds:
        messages, assistant_text = _normalize_messages(row)
        prompt_ids = _apply_chat_template(tokenizer, messages)

        response_ids = _tokenize_response(tokenizer, assistant_text)
        output_len = len(response_ids)
        if decode_cap is not None:
            output_len = min(output_len, decode_cap)
        output_len = max(1, output_len)

        samples.append(RealPromptSample(
            prompt_token_ids=prompt_ids,
            output_len=output_len,
        ))
        if len(samples) >= num_requests:
            break

    if len(samples) < num_requests:
        raise ValueError(
            f"{dataset_id} yielded only {len(samples)} requests; "
            f"needed {num_requests}"
        )
    return samples


# ===========================================================================
# Per-workload parameter specs (throughput / latency) and model configs
#
# These are constants by design so results are reproducible and comparable
# across runs and users. The enum members above carry the canonical *names*;
# the specs below carry the *parameters* keyed by those same names.
# ===========================================================================

# --- LLM (text-only, real WildChat-derived requests) -----------------------

@dataclass(frozen=True)
class ThroughputWorkload:
    name: str
    num_requests: int = 1000
    dataset_name: str = ""


@dataclass(frozen=True)
class LatencyWorkload:
    name: str
    batch_size: int
    input_len: int
    output_len: int
    num_warmup: int = 3
    num_iters: int = 5


THROUGHPUT_WORKLOADS: list[ThroughputWorkload] = [
    ThroughputWorkload("prefill-heavy", dataset_name=DEFAULT_WORKLOAD_DATASETS["prefill-heavy"]),
    ThroughputWorkload("balanced", dataset_name=DEFAULT_WORKLOAD_DATASETS["balanced"]),
    ThroughputWorkload("decode-heavy", dataset_name=DEFAULT_WORKLOAD_DATASETS["decode-heavy"]),
]

LATENCY_WORKLOADS: list[LatencyWorkload] = [
    LatencyWorkload(name="single-request", batch_size=1,  input_len=128, output_len=128),
    LatencyWorkload(name="fixed-batch-32", batch_size=32, input_len=128, output_len=128),
]


def get_max_seq_len() -> int:
    """Max static sequence length for standardized LLM latency workloads.

    Throughput decode lengths are data-dependent for real-prompt workloads, so
    eval computes their max sequence length after loading the dataset.
    """
    return max((w.input_len + w.output_len for w in LATENCY_WORKLOADS), default=0)


# --- ASR (audio transcription, e.g. Whisper) -------------------------------

@dataclass(frozen=True)
class ASRThroughputWorkload:
    name: str
    output_len: int
    dataset_name: str
    dataset_split: str
    use_full_dataset: bool = True


@dataclass(frozen=True)
class ASRLatencyWorkload:
    name: str
    output_len: int
    batch_size: int
    dataset_name: str
    dataset_split: str
    num_warmup: int = 3
    num_iters: int = 5


ASR_THROUGHPUT_WORKLOADS: list[ASRThroughputWorkload] = [
    ASRThroughputWorkload("librispeech", 448, "openslr/librispeech_asr", "test.clean"),
]

ASR_LATENCY_WORKLOADS: list[ASRLatencyWorkload] = [
    ASRLatencyWorkload("single-utterance", 448, 1, "openslr/librispeech_asr", "test.clean"),
    ASRLatencyWorkload("fixed-batch-32", 448, 32, "openslr/librispeech_asr", "test.clean"),
]


# --- VLM (multi-modal) -----------------------------------------------------

@dataclass(frozen=True)
class VLMThroughputWorkload:
    name: str
    modality: str  # "text", "image", "video", "audio"
    input_len: int | None  # fixed input token length (text only)
    output_len: int
    dataset_name: str | None = None  # HF dataset (image/video only)
    dataset_split: str | None = None
    num_requests: int = 1000


@dataclass(frozen=True)
class VLMLatencyWorkload:
    name: str
    modality: str  # "image", "video"
    output_len: int
    batch_size: int = 1
    dataset_name: str | None = None
    dataset_split: str | None = None
    num_warmup: int = 3
    num_iters: int = 5


VLM_THROUGHPUT_WORKLOADS: list[VLMThroughputWorkload] = [
    VLMThroughputWorkload("text-only", "text", input_len=512, output_len=1024),
    VLMThroughputWorkload("image", "image", input_len=None, output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train"),
    VLMThroughputWorkload("video", "video", input_len=None, output_len=512,
        dataset_name="yale-nlp/MMVU", dataset_split="validation"),
]

VLM_LATENCY_WORKLOADS: list[VLMLatencyWorkload] = [
    VLMLatencyWorkload("single-image", "image", output_len=128,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train"),
    VLMLatencyWorkload("single-video", "video", output_len=128,
        dataset_name="yale-nlp/MMVU", dataset_split="validation"),
]


# --- Qwen2.5-Omni (text + image + video + audio) ---------------------------

QWEN_OMNI_THROUGHPUT_WORKLOADS: list[VLMThroughputWorkload] = [
    VLMThroughputWorkload("text", "text", input_len=None, output_len=512,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["balanced"], dataset_split="train"),
    VLMThroughputWorkload("image", "image", input_len=None, output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train"),
    VLMThroughputWorkload("video", "video", input_len=None, output_len=512,
        dataset_name="yale-nlp/MMVU", dataset_split="validation"),
    VLMThroughputWorkload("audio", "audio", input_len=None, output_len=256,
        dataset_name="openslr/librispeech_asr", dataset_split="test.clean"),
]

QWEN_OMNI_LATENCY_WORKLOADS: list[VLMLatencyWorkload] = [
    VLMLatencyWorkload("single-text", "text", output_len=128,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["balanced"], dataset_split="train"),
    VLMLatencyWorkload("single-image", "image", output_len=128,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train"),
    VLMLatencyWorkload("single-video", "video", output_len=128,
        dataset_name="yale-nlp/MMVU", dataset_split="validation"),
    VLMLatencyWorkload("single-audio", "audio", output_len=128,
        dataset_name="openslr/librispeech_asr", dataset_split="test.clean"),
]


# --- Diffusion (image generation, e.g. FLUX, SDXL) -------------------------

@dataclass(frozen=True)
class DiffusionModelConfig:
    num_inference_steps: int
    guidance_scale: float


FLUX_CONFIG = DiffusionModelConfig(num_inference_steps=28, guidance_scale=3.5)
SDXL_CONFIG = DiffusionModelConfig(num_inference_steps=50, guidance_scale=5.0)


@dataclass(frozen=True)
class DiffusionThroughputWorkload:
    name: str
    height: int
    width: int
    batch_size: int
    num_requests: int = 10


@dataclass(frozen=True)
class DiffusionLatencyWorkload:
    name: str
    height: int
    width: int
    batch_size: int = 1
    num_warmup: int = 2
    num_iters: int = 5


DIFFUSION_THROUGHPUT_WORKLOADS: list[DiffusionThroughputWorkload] = [
    DiffusionThroughputWorkload("1024x1024", height=1024, width=1024, batch_size=4, num_requests=10),
    DiffusionThroughputWorkload("512x512", height=512, width=512, batch_size=8, num_requests=10),
]

DIFFUSION_LATENCY_WORKLOADS: list[DiffusionLatencyWorkload] = [
    DiffusionLatencyWorkload("single-1024x1024", height=1024, width=1024),
    DiffusionLatencyWorkload("single-512x512", height=512, width=512),
]


# --- Video diffusion (text-to-video, e.g. HunyuanVideo) --------------------

@dataclass(frozen=True)
class VideoDiffusionModelConfig:
    num_inference_steps: int
    guidance_scale: float


HUNYUAN_VIDEO_CONFIG = VideoDiffusionModelConfig(num_inference_steps=30, guidance_scale=6.0)


@dataclass(frozen=True)
class VideoDiffusionThroughputWorkload:
    name: str
    height: int
    width: int
    num_frames: int
    num_prompts: int


@dataclass(frozen=True)
class VideoDiffusionLatencyWorkload:
    name: str
    height: int
    width: int
    num_frames: int
    num_warmup: int = 2
    num_iters: int = 5


VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS: list[VideoDiffusionThroughputWorkload] = [
    VideoDiffusionThroughputWorkload("480p-short", height=480, width=832, num_frames=25, num_prompts=16),
    VideoDiffusionThroughputWorkload("480p-medium", height=480, width=832, num_frames=49, num_prompts=8),
]

VIDEO_DIFFUSION_LATENCY_WORKLOADS: list[VideoDiffusionLatencyWorkload] = [
    VideoDiffusionLatencyWorkload("single-480p-short", height=480, width=832, num_frames=25),
    VideoDiffusionLatencyWorkload("single-480p-medium", height=480, width=832, num_frames=49),
]


# --- Segmentation (promptable concept segmentation, e.g. SAM3) -------------

@dataclass(frozen=True)
class SegmentationThroughputWorkload:
    name: str
    resolution: int
    num_requests: int
    dataset_name: str
    dataset_subset: str = ""
    modality: str = "image"  # "image" or "video"


@dataclass(frozen=True)
class SegmentationLatencyWorkload:
    name: str
    resolution: int
    batch_size: int
    dataset_name: str
    dataset_subset: str = ""
    modality: str = "image"
    num_warmup: int = 3
    num_iters: int = 10


@dataclass(frozen=True)
class SegmentationVideoWorkload:
    name: str
    resolution: int
    num_clips: int
    frames_per_clip: int
    dataset_name: str
    dataset_subset: str = ""
    text_prompt: str = "objects"


SEGMENTATION_THROUGHPUT_WORKLOADS: list[SegmentationThroughputWorkload] = [
    SegmentationThroughputWorkload("gold-metaclip-nps", 1008, 500, "facebook/SACo-Gold", "metaclip_nps"),
    SegmentationThroughputWorkload("gold-wiki-common", 1008, 500, "facebook/SACo-Gold", "wiki_common"),
    SegmentationThroughputWorkload("gold-crowded", 1008, 500, "facebook/SACo-Gold", "crowded"),
    SegmentationThroughputWorkload("veval-sav-val", 1008, 100, "facebook/SACo-VEval", "sav_val", modality="video"),
    SegmentationThroughputWorkload("veval-yt1b-val", 1008, 100, "facebook/SACo-VEval", "yt1b_val", modality="video"),
]

SEGMENTATION_LATENCY_WORKLOADS: list[SegmentationLatencyWorkload] = [
    SegmentationLatencyWorkload("single-image-1008", 1008, 1, "facebook/SACo-Gold", "metaclip_nps"),
    SegmentationLatencyWorkload("batch-4-image-1008", 1008, 4, "facebook/SACo-Gold", "metaclip_nps"),
    SegmentationLatencyWorkload("single-video-frame-1008", 1008, 1, "facebook/SACo-VEval", "smartglasses_val", modality="video"),
]

SEGMENTATION_VIDEO_WORKLOADS: list[SegmentationVideoWorkload] = [
    SegmentationVideoWorkload("sav-val-video", 1008, 10, 16, "facebook/SACo-VEval", "sav_val"),
    SegmentationVideoWorkload("smartglasses-val-video", 1008, 10, 16, "facebook/SACo-VEval", "smartglasses_val"),
]


# --- TTS (text-to-speech, e.g. CosyVoice3) ---------------------------------

@dataclass(frozen=True)
class TTSModelConfig:
    sample_rate: int
    n_timesteps: int


COSYVOICE3_CONFIG = TTSModelConfig(sample_rate=24000, n_timesteps=10)


@dataclass(frozen=True)
class TTSThroughputWorkload:
    """SEED-TTS-Eval-derived request: a text prompt + reference audio."""
    name: str
    num_requests: int = 100
    max_text_len: int = 200
    dataset_name: str = "zhaochenyang20/seed-tts-eval"
    dataset_split: str = "train"


@dataclass(frozen=True)
class TTSLatencyWorkload:
    name: str
    batch_size: int = 1
    max_text_len: int = 200
    dataset_name: str = "zhaochenyang20/seed-tts-eval"
    dataset_split: str = "train"
    num_warmup: int = 2
    num_iters: int = 5


TTS_THROUGHPUT_WORKLOADS: list[TTSThroughputWorkload] = [
    TTSThroughputWorkload("tts-short", num_requests=100, max_text_len=50),
    TTSThroughputWorkload("tts-medium", num_requests=100, max_text_len=200),
    TTSThroughputWorkload("tts-long", num_requests=50, max_text_len=500),
]

TTS_LATENCY_WORKLOADS: list[TTSLatencyWorkload] = [
    TTSLatencyWorkload("single-utterance", batch_size=1, max_text_len=100),
]


# --- Robotics / VLA (action generation, e.g. Pi0) --------------------------

@dataclass(frozen=True)
class RoboticsModelConfig:
    num_inference_steps: int
    chunk_size: int
    max_action_dim: int
    max_state_dim: int
    image_resolution: tuple[int, int]


PI0_CONFIG = RoboticsModelConfig(
    num_inference_steps=10,
    chunk_size=50,
    max_action_dim=32,
    max_state_dim=32,
    image_resolution=(224, 224),
)


# --- Object detection (COCO val2017, e.g. YOLOv10) -------------------------

@dataclass(frozen=True)
class DetectionThroughputWorkload:
    name: str
    image_size: int
    num_images: int
    batch_size: int
    dataset_name: str = "detection-datasets/coco"
    dataset_split: str = "val"


@dataclass(frozen=True)
class DetectionLatencyWorkload:
    name: str
    image_size: int
    batch_size: int
    dataset_name: str = "detection-datasets/coco"
    dataset_split: str = "val"
    num_warmup: int = 3
    num_iters: int = 20


DETECTION_THROUGHPUT_WORKLOADS: list[DetectionThroughputWorkload] = [
    DetectionThroughputWorkload("coco-val", image_size=640, num_images=5000, batch_size=32),
]

DETECTION_LATENCY_WORKLOADS: list[DetectionLatencyWorkload] = [
    DetectionLatencyWorkload("single-image", image_size=640, batch_size=1),
    DetectionLatencyWorkload("batch-4", image_size=640, batch_size=4),
]


# --- Oasis (autoregressive video world model) ------------------------------

@dataclass(frozen=True)
class OasisWorkload:
    name: str
    batch_clips: int
    num_frames: int
    ddim_steps: int
    n_prompt_frames: int = 1
    kind: str = "throughput"
    dataset_name: str = "TESS-Computer/minecraft-vla-stage1"
    dataset_split: str = "train"


OASIS_THROUGHPUT_WORKLOADS: list[OasisWorkload] = [
    OasisWorkload("short-bs4-16f-4ddim", batch_clips=4, num_frames=16, ddim_steps=4),
    OasisWorkload("medium-bs8-24f-4ddim", batch_clips=8, num_frames=24, ddim_steps=4),
    OasisWorkload("long-bs8-32f-4ddim", batch_clips=8, num_frames=32, ddim_steps=4),
    OasisWorkload("denoise-bs4-16f-8ddim", batch_clips=4, num_frames=16, ddim_steps=8),
]

OASIS_LATENCY_WORKLOADS: list[OasisWorkload] = [
    OasisWorkload("latency-bs1-8f-4ddim", batch_clips=1, num_frames=8, ddim_steps=4, kind="latency"),
]


# --- Vision encoders (pure image feature extraction, e.g. SigLIP-2) --------

@dataclass(frozen=True)
class VisionEncoderThroughputWorkload:
    name: str
    resolution: int
    num_images: int
    batch_size: int
    dataset_name: str = "ILSVRC/imagenet-1k"
    dataset_split: str = "validation"


@dataclass(frozen=True)
class VisionEncoderLatencyWorkload:
    name: str
    resolution: int
    batch_size: int
    dataset_name: str = "ILSVRC/imagenet-1k"
    dataset_split: str = "validation"
    num_warmup: int = 3
    num_iters: int = 10


VISION_ENCODER_THROUGHPUT_WORKLOADS: list[VisionEncoderThroughputWorkload] = [
    VisionEncoderThroughputWorkload("default-res", resolution=0, num_images=5000, batch_size=32),
    VisionEncoderThroughputWorkload("high-res", resolution=512, num_images=2500, batch_size=16),
]

VISION_ENCODER_LATENCY_WORKLOADS: list[VisionEncoderLatencyWorkload] = [
    VisionEncoderLatencyWorkload("single-image", resolution=0, batch_size=1, num_warmup=5, num_iters=30),
    VisionEncoderLatencyWorkload("batch-8", resolution=0, batch_size=8, num_warmup=5, num_iters=30),
]


# --- Text embeddings (token-level retrieval, e.g. BGE-M3, ColBERTv2) -------

@dataclass(frozen=True)
class EmbeddingThroughputWorkload:
    name: str
    model_key: str
    model_name: str
    dataset_name: str
    dataset_config: str
    dataset_split: str
    id_column: str | None
    text_column: str
    jsonl_name: str
    num_requests: int = 1000


@dataclass(frozen=True)
class EmbeddingLatencyWorkload:
    name: str
    batch_size: int
    num_warmup: int = 3
    num_iters: int = 5


EMBEDDING_THROUGHPUT_WORKLOADS: list[EmbeddingThroughputWorkload] = [
    EmbeddingThroughputWorkload(
        name="bge-m3-mldr-docs",
        model_key="bge_m3",
        model_name="BAAI/bge-m3",
        dataset_name="sentence-transformers/mldr",
        dataset_config="en-triplet",
        dataset_split="train",
        id_column=None,
        text_column="positive",
        jsonl_name="bge_m3_mldr_documents.jsonl",
    ),
    EmbeddingThroughputWorkload(
        name="colbertv2-msmarco-passages",
        model_key="colbertv2",
        model_name="colbert-ir/colbertv2.0",
        dataset_name="sentence-transformers/msmarco",
        dataset_config="corpus",
        dataset_split="train",
        id_column="passage_id",
        text_column="passage",
        jsonl_name="colbertv2_msmarco_passages.jsonl",
        num_requests=60_000,
    ),
]

EMBEDDING_LATENCY_WORKLOADS: list[EmbeddingLatencyWorkload] = [
    EmbeddingLatencyWorkload(name="single-request", batch_size=1),
    EmbeddingLatencyWorkload(name="fixed-batch-32", batch_size=32),
]


# --- Structure prediction (OpenFold3 / AlphaFold3-style) -------------------

@dataclass(frozen=True)
class StructurePredictionThroughputWorkload:
    name: str
    num_queries: int
    description: str
    dataset_name: str = "OpenProteinSet"


@dataclass(frozen=True)
class StructurePredictionLatencyWorkload:
    name: str
    length_bucket: str
    num_warmup: int = 1
    num_iters: int = 3
    dataset_name: str = "OpenProteinSet"


STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS: list[StructurePredictionThroughputWorkload] = [
    StructurePredictionThroughputWorkload("short", 50, "short proteins (<=150 residues) x 50 queries"),
    StructurePredictionThroughputWorkload("medium", 20, "medium proteins (150-400 residues) x 20 queries"),
    StructurePredictionThroughputWorkload("long", 10, "long proteins (400-700 residues) x 10 queries"),
    StructurePredictionThroughputWorkload("extra-long", 5, "extra-long proteins (700+ residues) x 5 queries"),
]

STRUCTURE_PREDICTION_LATENCY_WORKLOADS: list[StructurePredictionLatencyWorkload] = [
    StructurePredictionLatencyWorkload("single-short", "short"),
    StructurePredictionLatencyWorkload("single-medium", "medium"),
    StructurePredictionLatencyWorkload("single-long", "long"),
    StructurePredictionLatencyWorkload("single-extra-long", "extra-long"),
]


# --- 3-D point-cloud robotics policies (DP3 / Simple-DP3) ------------------

@dataclass(frozen=True)
class DP3ModelConfig:
    """Hyperparameters that drive both fastkernels DP3 and the reference engine."""
    num_inference_steps: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    num_points: int
    state_dim: int
    action_dim: int


# Defaults match the xarm push-block dataset (the public point-cloud robotics
# benchmark closest to DP3's MetaWorld setup): 512 points x 6 channels
# (XYZRGB; sliced to XYZ for use_pc_color=False), 7-D joint-state
# proprioception, 4-D end-effector action (x, y, z, gripper).
DP3_CONFIG = DP3ModelConfig(
    num_inference_steps=10,
    horizon=16,
    n_obs_steps=2,
    n_action_steps=8,
    num_points=512,
    state_dim=7,
    action_dim=4,
)


@dataclass(frozen=True)
class DP3ThroughputWorkload:
    """3-D diffusion policy throughput workload (per-frame action chunk gen).

    Real point clouds + robot state + actions come from a public 3-D
    point-cloud robotics dataset (default ``rishabhrj11/gym-xarm-pointcloud``).
    """
    name: str
    num_requests: int
    batch_size: int = 1
    dataset_name: str = "rishabhrj11/gym-xarm-pointcloud"


@dataclass(frozen=True)
class DP3LatencyWorkload:
    name: str
    batch_size: int = 1
    num_warmup: int = 5
    num_iters: int = 20
    dataset_name: str = "rishabhrj11/gym-xarm-pointcloud"


DP3_THROUGHPUT_WORKLOADS: list[DP3ThroughputWorkload] = [
    DP3ThroughputWorkload("dp3-1env", num_requests=100, batch_size=1),
    DP3ThroughputWorkload("dp3-batch", num_requests=100, batch_size=8),
]

DP3_LATENCY_WORKLOADS: list[DP3LatencyWorkload] = [
    DP3LatencyWorkload("single-step", batch_size=1),
    DP3LatencyWorkload("batch-8", batch_size=8),
]


# ===========================================================================
# vLLM dataset adapter (lazy)
#
# Re-exports vLLM's dataset types/helpers so benchmarks reuse the exact same
# dataset CLI, loading and sampling logic. Resolved lazily via ``__getattr__``
# so that ``import fastkernels.workloads`` never imports ``vllm``; the cost is
# paid only by the e2e runners that actually access these names.
# ===========================================================================

_VLLM_DATASET_EXPORTS = ("SampleRequest", "add_dataset_parser", "get_samples")


def __getattr__(name: str) -> Any:  # PEP 562: supports `from ... import SampleRequest`
    if name in _VLLM_DATASET_EXPORTS:
        from vllm.benchmarks import datasets as _vllm_datasets

        for export in _VLLM_DATASET_EXPORTS:
            globals()[export] = getattr(_vllm_datasets, export)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
