"""Benchmark workloads, architecture registry, and scenario tables.

Single source of truth for everything workload- and benchmark-related. It holds:

* the per-family workload *identity* enums (e.g. ``LLM.mixed``);
* the per-workload throughput/latency *parameter* specs and model configs
  consumed by the eval pipeline and the ``tests/`` benchmarks;
* the real chat-prompt loader (``load_real_prompt_workload``) used to run the
  LLM mixed (WildChat) and long-context (LongBench-v2) workloads;
* the architecture registry (``FAMILIES``, ``FASTKERNELS_ARCHITECTURES``,
  ``module_for``) and the ``BenchmarkScenario`` dataclass; and
* the ``FULL_BENCHMARK`` / ``DEFAULT_BENCHMARK`` scenario tables, loaded from the
  user-editable YAML files in ``scenarios/`` via ``load_benchmark``.

Kept import-light on purpose: importing this module must not pull in ``vllm`` or
the Hugging Face ``datasets`` package (imported inside the loader), so
``capture.py`` and ``list.py`` can import it cheaply. ``module_for`` imports
``transformers`` lazily (inside the function), and the scenario loader imports
``yaml`` lazily.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Workload(Enum):
    """Base for the per-family workload enums; value is the canonical name."""

    def __str__(self) -> str:  # so ", ".join(...) and logging read naturally
        return self.value


class LLM(Workload):
    """Text LLMs (also used by linear-attention / hybrid LLMs).

    ``mixed`` and ``long_context`` are the two real-prompt *throughput* regimes
    (backed by the wildchat-mixed-1k and longbench-longctx datasets);
    ``single_request`` and ``fixed_batch_32`` are the standardized *latency*
    probes.
    """
    mixed = "mixed"
    long_context = "long-context"
    single_request = "single-request"
    fixed_batch_32 = "fixed-batch-32"


class DiffusionLM(Workload):
    """Masked-diffusion LMs (LLaDA).

    ``bench_dllm`` runs one protocol: real task prompts batched together and
    decoded with a fixed step/block schedule, scored on generated tokens/sec.
    There is no latency probe -- a single-request number would just be the same
    fixed step count with a batch of one.
    """
    humaneval = "humaneval"


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
    full_pipeline = "full-pipeline"
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
    """Vision-language-action policies (e.g. Pi0).

    Member names encode the dataset the workload runs on, because bench_openpi
    benchmarks one dataset per `--datasets` entry and each needs its own
    checkpoint: ALOHA is 3- or 1-camera, DROID and LIBERO are 2-camera.
    """
    aloha_3cam = "aloha-3cam"
    aloha_1cam = "aloha-1cam"
    droid_2cam = "droid-2cam"
    libero_2cam = "libero-2cam"
    aloha_single_3cam = "aloha-single-3cam"
    aloha_single_1cam = "aloha-single-1cam"
    droid_single = "droid-single"
    libero_single = "libero-single"


class PointCloudPolicy(Workload):
    """3-D point-cloud diffusion policies (e.g. DP3)."""
    dp3_1env = "dp3-1env"
    dp3_batch = "dp3-batch"
    single_step = "single-step"
    batch_8 = "batch-8"


class PointCloudSeg(Workload):
    """Point-cloud understanding / segmentation (e.g. PointTransformerV3)."""
    scanobjectnn = "scanobjectnn"
    single_cloud = "single-cloud"
    batch_8 = "batch-8"


class Rendering(Workload):
    """Neural scene rendering (e.g. 3DGS, InstantNGP)."""
    render = "render"
    single_render = "single-render"


class Recsys(Workload):
    """Recommendation / ranking (e.g. DLRMv2 CTR, LightGCN graph recommend)."""
    ctr_batch = "ctr-batch"
    recommend_batch = "recommend-batch"
    single_request = "single-request"
    fixed_batch_32 = "fixed-batch-32"


class VideoRepresentation(Workload):
    """Video world-model / representation forwards (e.g. V-JEPA 2)."""
    predictor = "predictor"
    encoder = "encoder"
    classification = "classification"
    single_video = "single-video"


# ===========================================================================
# Real chat-prompt workloads (LLM)
#
# The workload datasets store raw chat messages. Runners call the loader with
# the target model's tokenizer so prompts are chat-templated and tokenized
# exactly as that model expects.
# ===========================================================================

DEFAULT_WORKLOAD_DATASETS: dict[str, str] = {
    "mixed": "sfc-gh-goliaro/wildchat-mixed-1k",
    "long-context": "sfc-gh-goliaro/longbench-longctx",
}

# Canonical per-request generation budget applied when a caller does not pass an
# explicit ``decode_cap``. ``mixed`` caps WildChat responses at 1024 tokens (the
# calibrated Scenario A budget); ``long-context`` carries an authoritative
# per-row ``output_len``, so no cap is applied there.
DEFAULT_DECODE_CAPS: dict[str, int | None] = {
    "mixed": 1024,
    "long-context": None,
}


@dataclass(frozen=True)
class RealPromptSample:
    prompt_token_ids: list[int]
    output_len: int
    # Raw chat-templated prompt turns (role/content dicts). Populated for
    # consumers that need the un-tokenized prompt, e.g. the online serving
    # benchmark that posts chat messages to an OpenAI-compatible endpoint.
    messages: list[dict[str, str]] | None = None


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

    Prompts are stored as raw chat turns and are chat-templated + tokenized here
    with the *target model's* tokenizer, so every model sees the same requests
    tokenized as it expects. The per-request generation budget (``output_len``)
    is taken from an authoritative per-row ``output_len`` when present (the
    long-context set), otherwise derived from the stored assistant response.

    ``decode_cap`` bounds ``output_len``; when ``None`` it falls back to the
    scenario's canonical cap (``DEFAULT_DECODE_CAPS``). Curated fixed-size sets
    (e.g. long-context, 64 rows) may hold fewer than ``num_requests`` rows, in
    which case every available row is returned.
    """
    from datasets import load_dataset

    dataset_id = dataset_name or DEFAULT_WORKLOAD_DATASETS[scenario_name]
    if decode_cap is None:
        decode_cap = DEFAULT_DECODE_CAPS.get(scenario_name)

    ds = load_dataset(dataset_id, split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    samples: list[RealPromptSample] = []
    for row in ds:
        messages, assistant_text = _normalize_messages(row)
        prompt_ids = _apply_chat_template(tokenizer, messages)

        stored_output_len = row.get("output_len")
        if stored_output_len is not None:
            output_len = int(stored_output_len)
        else:
            output_len = len(_tokenize_response(tokenizer, assistant_text))
        if decode_cap is not None:
            output_len = min(output_len, decode_cap)
        output_len = max(1, output_len)

        samples.append(RealPromptSample(
            prompt_token_ids=prompt_ids,
            output_len=output_len,
            messages=messages,
        ))
        if len(samples) >= num_requests:
            break

    if not samples:
        raise ValueError(f"{dataset_id} yielded no requests")
    return samples


# ===========================================================================
# Per-workload parameter specs (throughput / latency) and model configs
#
# These are constants by design so results are reproducible and comparable
# across runs and users. The enum members above carry the canonical *names*;
# the specs below carry the *parameters* keyed by those same names.
# ===========================================================================

# --- LLM (text-only, real WildChat / LongBench-v2 requests) ----------------

@dataclass(frozen=True)
class ThroughputWorkload:
    name: str
    num_requests: int = 1000
    dataset_name: str = ""
    decode_cap: int | None = None


@dataclass(frozen=True)
class LatencyWorkload:
    name: str
    batch_size: int
    output_len: int = 128        # decode budget (ignore_eos); real prompt drives prefill
    dataset_name: str = ""       # real prompt source (WildChat mixed)
    num_warmup: int = 3
    num_iters: int = 5


# Two real-prompt throughput regimes:
#   * mixed        -- WildChat-1M, N=1000, natural short/long prompt+decode mix
#                     (decode capped at 1024); saturates continuous batching.
#   * long-context -- LongBench-v2, N=64, 8K..128K prefill buckets, fixed decode;
#                     exercises long-sequence attention + large-KV decode.
THROUGHPUT_WORKLOADS: list[ThroughputWorkload] = [
    ThroughputWorkload("mixed", num_requests=1000,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["mixed"], decode_cap=DEFAULT_DECODE_CAPS["mixed"]),
    ThroughputWorkload("long-context", num_requests=64,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["long-context"], decode_cap=DEFAULT_DECODE_CAPS["long-context"]),
]

# Latency probes draw REAL prompts from the same WildChat `mixed` set as
# throughput (batch 1 and batch 32); the prompt length is therefore
# data-dependent, not a fixed shape. ``output_len`` bounds the decode so
# per-token latency stays comparable across models.
LATENCY_WORKLOADS: list[LatencyWorkload] = [
    LatencyWorkload(name="single-request", batch_size=1,  output_len=128,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["mixed"]),
    LatencyWorkload(name="fixed-batch-32", batch_size=32, output_len=128,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["mixed"]),
]


def get_max_seq_len() -> int:
    """Decode-budget floor for latency workloads.

    Latency now draws real prompts, so the prompt (and full sequence) length is
    data-dependent and computed after the dataset is loaded. This returns only
    the max decode budget as a static floor.
    """
    return max((w.output_len for w in LATENCY_WORKLOADS), default=0)


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
    VLMThroughputWorkload("text-only", "text", input_len=None, output_len=512,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["mixed"], dataset_split="train"),
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
        dataset_name=DEFAULT_WORKLOAD_DATASETS["mixed"], dataset_split="train"),
    VLMThroughputWorkload("image", "image", input_len=None, output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train"),
    VLMThroughputWorkload("video", "video", input_len=None, output_len=512,
        dataset_name="yale-nlp/MMVU", dataset_split="validation"),
    VLMThroughputWorkload("audio", "audio", input_len=None, output_len=256,
        dataset_name="openslr/librispeech_asr", dataset_split="test.clean"),
]

QWEN_OMNI_LATENCY_WORKLOADS: list[VLMLatencyWorkload] = [
    VLMLatencyWorkload("single-text", "text", output_len=128,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["mixed"], dataset_split="train"),
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


# bench_sam makes exactly two throughput measurements, and these declare them.
# It pools SACo-Gold images with SACo-VEval frames and times one pass over the
# pooled set ("full-pipeline"), then tracks SACo-VEval clips and reports
# frames/sec. The five per-dataset image rows this table used to declare
# (gold-metaclip-nps, gold-wiki-common, gold-crowded, veval-sav-val,
# veval-yt1b-val) were never run -- bench_sam imported the list and ignored it,
# so the sweep advertised seven throughput workloads and produced none. Only
# sav-val-video is dropped: the clip pass runs one --veval-subset, the default.
SEGMENTATION_THROUGHPUT_WORKLOADS: list[SegmentationThroughputWorkload] = [
    SegmentationThroughputWorkload(
        "full-pipeline", 1008, 500, "facebook/SACo-Gold", "metaclip", modality="image"
    ),
]

SEGMENTATION_LATENCY_WORKLOADS: list[SegmentationLatencyWorkload] = [
    SegmentationLatencyWorkload("single-image-1008", 1008, 1, "facebook/SACo-Gold", "metaclip_nps"),
    SegmentationLatencyWorkload("batch-4-image-1008", 1008, 4, "facebook/SACo-Gold", "metaclip_nps"),
    SegmentationLatencyWorkload("single-video-frame-1008", 1008, 1, "facebook/SACo-VEval", "smartglasses_val", modality="video"),
]

# Clip counts match bench_sam's _save_video_clips_for_workers(max_clips=10,
# max_frames=16).
SEGMENTATION_VIDEO_WORKLOADS: list[SegmentationVideoWorkload] = [
    SegmentationVideoWorkload(
        "smartglasses-val-video", 1008, 10, 16, "facebook/SACo-VEval", "smartglasses_val"
    ),
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
    # 5000 unique COCO val2017 images cycled 6x (see bench_detection tiling):
    # shape-bound throughput so tiling only lengthens the timing window, giving
    # duration parity (~20-30s on fast detectors) with the other families while
    # the reported img/s rate is unchanged.
    DetectionThroughputWorkload("coco-val", image_size=640, num_images=30000, batch_size=32),
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
# Unified workload registry (identity + purpose + params)
#
# Single source of truth linking every workload *identity* enum member to its
# *purpose* (throughput vs latency) and its *parameter* spec. The per-family
# ``*_THROUGHPUT_WORKLOADS`` / ``*_LATENCY_WORKLOADS`` lists above remain the
# canonical parameter definitions (and stay importable for the bench scripts);
# this section joins them to their enum members exactly once so callers resolve
# a workload with ``spec_for(member)`` / ``spec_by_name(Family, name)`` instead
# of rebuilding ad-hoc ``{w.name: w}`` dicts.
# ===========================================================================

class Purpose(Enum):
    """What a workload is measured for."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"


@dataclass(frozen=True)
class WorkloadSpec:
    """A workload identity plus its purpose and (optional) parameter spec."""
    workload: Workload
    purpose: Purpose
    params: Any = None

    @property
    def name(self) -> str:
        return self.workload.value


# (family enum, throughput param specs, latency param specs). Segmentation's
# separate video-throughput list is folded into its throughput entry.
_SPEC_SOURCES: tuple[tuple[type[Workload], list[Any], list[Any]], ...] = (
    (LLM, THROUGHPUT_WORKLOADS, LATENCY_WORKLOADS),
    (VLM, VLM_THROUGHPUT_WORKLOADS, VLM_LATENCY_WORKLOADS),
    (OmniModal, QWEN_OMNI_THROUGHPUT_WORKLOADS, QWEN_OMNI_LATENCY_WORKLOADS),
    (ASR, ASR_THROUGHPUT_WORKLOADS, ASR_LATENCY_WORKLOADS),
    (TTS, TTS_THROUGHPUT_WORKLOADS, TTS_LATENCY_WORKLOADS),
    (Diffusion, DIFFUSION_THROUGHPUT_WORKLOADS, DIFFUSION_LATENCY_WORKLOADS),
    (VideoDiffusion, VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS, VIDEO_DIFFUSION_LATENCY_WORKLOADS),
    (WorldModel, OASIS_THROUGHPUT_WORKLOADS, OASIS_LATENCY_WORKLOADS),
    (Segmentation, [*SEGMENTATION_THROUGHPUT_WORKLOADS, *SEGMENTATION_VIDEO_WORKLOADS], SEGMENTATION_LATENCY_WORKLOADS),
    (Detection, DETECTION_THROUGHPUT_WORKLOADS, DETECTION_LATENCY_WORKLOADS),
    (VisionEncoder, VISION_ENCODER_THROUGHPUT_WORKLOADS, VISION_ENCODER_LATENCY_WORKLOADS),
    (Embedding, EMBEDDING_THROUGHPUT_WORKLOADS, EMBEDDING_LATENCY_WORKLOADS),
    (StructurePrediction, STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS, STRUCTURE_PREDICTION_LATENCY_WORKLOADS),
    (PointCloudPolicy, DP3_THROUGHPUT_WORKLOADS, DP3_LATENCY_WORKLOADS),
)

# Families benchmarked by bespoke scripts that carry no parameter specs yet;
# purpose is assigned by intent (batched runs -> throughput, single/latency
# probes -> latency).
_PARAMLESS_PURPOSES: dict[Workload, Purpose] = {
    DiffusionLM.humaneval: Purpose.THROUGHPUT,
    Robotics.aloha_3cam: Purpose.THROUGHPUT,
    Robotics.aloha_1cam: Purpose.THROUGHPUT,
    Robotics.droid_2cam: Purpose.THROUGHPUT,
    Robotics.libero_2cam: Purpose.THROUGHPUT,
    Robotics.aloha_single_3cam: Purpose.LATENCY,
    Robotics.aloha_single_1cam: Purpose.LATENCY,
    Robotics.droid_single: Purpose.LATENCY,
    Robotics.libero_single: Purpose.LATENCY,
    PointCloudSeg.scanobjectnn: Purpose.THROUGHPUT,
    PointCloudSeg.single_cloud: Purpose.LATENCY,
    PointCloudSeg.batch_8: Purpose.THROUGHPUT,
    Rendering.render: Purpose.THROUGHPUT,
    Rendering.single_render: Purpose.LATENCY,
    Recsys.ctr_batch: Purpose.THROUGHPUT,
    Recsys.recommend_batch: Purpose.THROUGHPUT,
    Recsys.single_request: Purpose.LATENCY,
    Recsys.fixed_batch_32: Purpose.LATENCY,
    VideoRepresentation.predictor: Purpose.THROUGHPUT,
    VideoRepresentation.encoder: Purpose.THROUGHPUT,
    VideoRepresentation.classification: Purpose.THROUGHPUT,
    VideoRepresentation.single_video: Purpose.LATENCY,
}

_ALL_FAMILIES: tuple[type[Workload], ...] = (
    LLM, VLM, OmniModal, ASR, TTS, Diffusion, VideoDiffusion, WorldModel,
    Segmentation, Detection, VisionEncoder, Embedding, StructurePrediction,
    Robotics, PointCloudPolicy, PointCloudSeg, Rendering, Recsys,
    VideoRepresentation, DiffusionLM,
)


def _build_workload_specs() -> dict[Workload, WorkloadSpec]:
    specs: dict[Workload, WorkloadSpec] = {}
    for family, throughput, latency in _SPEC_SOURCES:
        by_name = {member.value: member for member in family}
        for params, purpose in (
            *((p, Purpose.THROUGHPUT) for p in throughput),
            *((p, Purpose.LATENCY) for p in latency),
        ):
            member = by_name.get(params.name)
            if member is None:
                raise ValueError(
                    f"{family.__name__}: workload {params.name!r} has no matching enum member"
                )
            if member in specs:
                raise ValueError(f"{family.__name__}: duplicate spec for {member}")
            specs[member] = WorkloadSpec(member, purpose, params)
    for member, purpose in _PARAMLESS_PURPOSES.items():
        specs[member] = WorkloadSpec(member, purpose)
    # Every enum member must be classified, or resolution silently loses one.
    unclassified = [m for fam in _ALL_FAMILIES for m in fam if m not in specs]
    if unclassified:
        raise RuntimeError(f"Workloads missing a purpose/spec: {unclassified}")
    return specs


# member -> WorkloadSpec, the canonical resolution table.
WORKLOAD_SPECS: dict[Workload, WorkloadSpec] = _build_workload_specs()


def spec_for(workload: Workload) -> WorkloadSpec:
    """The WorkloadSpec (purpose + params) for a workload identity."""
    return WORKLOAD_SPECS[workload]


def purpose_of(workload: Workload) -> Purpose:
    """Whether *workload* is a throughput or latency measurement."""
    return WORKLOAD_SPECS[workload].purpose


def spec_by_name(family: type[Workload], name: str) -> WorkloadSpec | None:
    """Resolve a workload by its family enum and canonical name; ``None`` if unknown."""
    try:
        member = family(name)
    except ValueError:
        return None
    return WORKLOAD_SPECS.get(member)


def throughput_params(family: type[Workload], name: str) -> Any:
    """Parameter spec for a *throughput* workload named *name* in *family*.

    Returns ``None`` when the name is unknown to the family or is not a
    throughput workload -- the exact acceptance rule the tracing entry points
    previously encoded with per-call ``{w.name: w}`` dicts over the family's
    throughput list.
    """
    spec = spec_by_name(family, name)
    if spec is None or spec.purpose is not Purpose.THROUGHPUT:
        return None
    return spec.params


# ===========================================================================
# Architecture registry (aligned with table.tex) and benchmark scenarios
#
# Merged here from the former ``registry.py``: the family/architecture registry
# and name->module resolution, the ``BenchmarkScenario`` dataclass, and the
# loader that builds ``FULL_BENCHMARK`` / ``DEFAULT_BENCHMARK`` from the
# user-editable YAML tables in ``scenarios/``.
# ===========================================================================

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
    on the workload spec (``spec_for(workload)``), which is why there is no
    scenario-level ``num_requests``.

    ``workloads`` mixes throughput and latency workloads; use the
    ``throughput_workloads`` / ``latency_workloads`` splits to iterate one kind.

    ``hf_name`` is always a bare identifier -- an HF repo id, or a short module
    token for the rows that have no Hub checkpoint (``dlrmv2``, ``dp3``, ...).
    Anything else a harness needs goes in its own optional field rather than
    being packed into the name: ``draft_model`` (speculative decoding target +
    draft), ``variant`` (which net a multi-variant harness builds), ``scene``
    (which scene a renderer loads), ``reference_checkpoint`` (a reference
    implementation whose weights live somewhere other than ``hf_name``).
    """
    hf_name: str
    tp: int
    dtype: str
    workloads: list[Workload]
    enforce_eager: bool = False
    max_num_seqs: int | None = None
    draft_model: str | None = None
    variant: str | None = None
    scene: str | None = None
    reference_checkpoint: str | None = None

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


# --- Scenario YAML loader --------------------------------------------------
#
# The benchmark tables live in ``scenarios/full.yaml`` and ``scenarios/default.yaml``
# as user-editable YAML. Each scenario names its workloads with fully-qualified
# ``Family.member`` tokens (the enum member names above); there is no bare-family
# "all" shorthand -- every workload is spelled out.

_WORKLOAD_FAMILIES: dict[str, type[Workload]] = {c.__name__: c for c in _ALL_FAMILIES}
_ALLOWED_DTYPES = {"bfloat16", "float16", "float32", "fp8", "mxfp4"}


def _resolve_workload_token(token: str) -> Workload:
    """Resolve a ``Family.member`` token to its ``Workload`` enum member."""
    fam_name, sep, member = token.strip().partition(".")
    if not sep:  # bare family names are not allowed -- workloads must be explicit
        raise ValueError(f"workload token {token!r} must be 'Family.member'")
    fam = _WORKLOAD_FAMILIES.get(fam_name)
    if fam is None:
        raise ValueError(
            f"unknown workload family {fam_name!r} in {token!r}; "
            f"valid: {sorted(_WORKLOAD_FAMILIES)}"
        )
    try:
        return fam[member]  # enum lookup by MEMBER NAME
    except KeyError:
        raise ValueError(
            f"unknown workload {member!r} for {fam_name!r}; "
            f"valid: {[m.name for m in fam]}"
        )


_SCENARIO_KEYS = frozenset(
    {
        "model", "tp", "dtype", "workloads",
        "enforce_eager", "max_num_seqs",
        "draft_model", "variant", "scene", "reference_checkpoint",
    }
)


def _scenario_from_mapping(entry: Mapping[str, Any], *, source: str) -> BenchmarkScenario:
    """Build a ``BenchmarkScenario`` from one YAML mapping, validating as we go."""
    unknown = sorted(set(entry) - _SCENARIO_KEYS)
    if unknown:
        raise ValueError(
            f"{source}: {entry.get('model', entry)}: unknown key(s) {unknown}; "
            f"valid: {sorted(_SCENARIO_KEYS)}"
        )
    try:
        model, tp, dtype = entry["model"], int(entry["tp"]), entry["dtype"]
        tokens = entry["workloads"]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{source}: bad scenario entry {entry!r}: {e}")
    if dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"{source}: {model}: dtype {dtype!r} not in {sorted(_ALLOWED_DTYPES)}")
    for key in ("model", "draft_model", "variant", "scene"):
        value = entry.get(key)
        if value is not None and (not isinstance(value, str) or not value or " " in value):
            raise ValueError(
                f"{source}: {model}: {key} must be a non-empty identifier without "
                f"spaces, got {value!r}"
            )
    workloads: list[Workload] = []
    seen: set[Workload] = set()
    for tok in tokens:
        try:
            wl = _resolve_workload_token(tok)
        except ValueError as e:
            raise ValueError(f"{source}: {model}: {e}")
        if wl not in seen:
            seen.add(wl)
            workloads.append(wl)
    if not workloads:
        raise ValueError(f"{source}: {model}: no workloads")
    return BenchmarkScenario(
        model, tp, dtype, workloads,
        enforce_eager=bool(entry.get("enforce_eager", False)),
        max_num_seqs=entry.get("max_num_seqs"),
        draft_model=entry.get("draft_model"),
        variant=entry.get("variant"),
        scene=entry.get("scene"),
        reference_checkpoint=entry.get("reference_checkpoint"),
    )


def load_benchmark(path: Path) -> list[BenchmarkScenario]:
    """Load a benchmark scenario table from a YAML file (see ``scenarios/``)."""
    import yaml

    data = yaml.safe_load(path.read_text())
    return [_scenario_from_mapping(e, source=path.name) for e in data["scenarios"]]


_SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


def resolve_benchmark(name_or_path: str | Path) -> list[BenchmarkScenario]:
    """Load a scenario table from a filesystem path or a bare scenario name.

    An existing filesystem path is loaded directly. Otherwise the argument is
    treated as a name resolved against the packaged ``scenarios/`` directory
    (e.g. ``"minimal"`` or ``"minimal.yaml"`` -> ``scenarios/minimal.yaml``), so
    callers get the shipped ``full`` / ``default`` / ``minimal`` tables for free.
    """
    p = Path(name_or_path)
    if p.exists():
        return load_benchmark(p)
    stem = p.name[:-5] if p.name.endswith(".yaml") else p.name
    packaged = _SCENARIO_DIR / f"{stem}.yaml"
    if packaged.exists():
        return load_benchmark(packaged)
    raise FileNotFoundError(
        f"no scenarios file at {name_or_path!r}, and no packaged {packaged.name} "
        f"in {_SCENARIO_DIR}"
    )


FULL_BENCHMARK: list[BenchmarkScenario] = load_benchmark(_SCENARIO_DIR / "full.yaml")
DEFAULT_BENCHMARK: list[BenchmarkScenario] = load_benchmark(_SCENARIO_DIR / "default.yaml")


