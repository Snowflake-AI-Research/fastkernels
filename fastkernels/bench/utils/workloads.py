"""Standardized workload definitions for eval (Tier 3).

These workloads are constants that ensure reproducible, comparable results
across runs and users. They are not configurable by design.

LLM workloads (text-only, real WildChat-derived requests):
  Throughput: 3 scenarios (prefill-heavy, balanced, decode-heavy), 1000 reqs each.
  Latency: 2 scenarios (single-request, fixed-batch-32).

VLM workloads (multi-modal):
  Throughput: 3 scenarios (text-only, image, video), 1000 reqs each.
  Latency: 2 scenarios (single-image, single-video), batch_size=1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .real_prompts import DEFAULT_WORKLOAD_DATASETS


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
    ThroughputWorkload(
        name="prefill-heavy",
        dataset_name=DEFAULT_WORKLOAD_DATASETS["prefill-heavy"],
    ),
    ThroughputWorkload(
        name="balanced",
        dataset_name=DEFAULT_WORKLOAD_DATASETS["balanced"],
    ),
    ThroughputWorkload(
        name="decode-heavy",
        dataset_name=DEFAULT_WORKLOAD_DATASETS["decode-heavy"],
    ),
]

LATENCY_WORKLOADS: list[LatencyWorkload] = [
    LatencyWorkload(name="single-request",  batch_size=1,  input_len=128, output_len=128),
    LatencyWorkload(name="fixed-batch-32",  batch_size=32, input_len=128, output_len=128),
]

ALL_WORKLOADS = {
    "throughput": THROUGHPUT_WORKLOADS,
    "latency": LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Whisper / ASR workloads (audio transcription)
# ---------------------------------------------------------------------------

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
    ASRThroughputWorkload(
        name="librispeech",
        output_len=448,
        dataset_name="openslr/librispeech_asr",
        dataset_split="test.clean",
    ),
]

ASR_LATENCY_WORKLOADS: list[ASRLatencyWorkload] = [
    ASRLatencyWorkload(
        name="single-utterance",
        output_len=448,
        batch_size=1,
        dataset_name="openslr/librispeech_asr",
        dataset_split="test.clean",
    ),
    ASRLatencyWorkload(
        name="fixed-batch-32",
        output_len=448,
        batch_size=32,
        dataset_name="openslr/librispeech_asr",
        dataset_split="test.clean",
    ),
]

ALL_ASR_WORKLOADS = {
    "throughput": ASR_THROUGHPUT_WORKLOADS,
    "latency": ASR_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# VLM workloads (multi-modal)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VLMThroughputWorkload:
    name: str
    modality: str  # "text", "image", "video"
    input_len: int | None  # fixed input token length (text only)
    output_len: int
    dataset_name: str | None = None  # HF dataset (image/video only)
    dataset_split: str | None = None
    num_requests: int = 1000


VLM_THROUGHPUT_WORKLOADS: list[VLMThroughputWorkload] = [
    VLMThroughputWorkload(
        "text-only", "text", input_len=512, output_len=1024),
    VLMThroughputWorkload(
        "image", "image", input_len=None, output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train"),
    VLMThroughputWorkload(
        "video", "video", input_len=None, output_len=512,
        dataset_name="yale-nlp/MMVU", dataset_split="validation"),
]


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


VLM_LATENCY_WORKLOADS: list[VLMLatencyWorkload] = [
    VLMLatencyWorkload(
        "single-image", "image", output_len=128,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train"),
    VLMLatencyWorkload(
        "single-video", "video", output_len=128,
        dataset_name="yale-nlp/MMVU", dataset_split="validation"),
]

ALL_VLM_WORKLOADS = {
    "throughput": VLM_THROUGHPUT_WORKLOADS,
    "latency": VLM_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Qwen2.5-Omni workloads (text + image + video + audio)
# ---------------------------------------------------------------------------

QWEN_OMNI_THROUGHPUT_WORKLOADS: list[VLMThroughputWorkload] = [
    VLMThroughputWorkload(
        "text", "text", input_len=None, output_len=512,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["balanced"],
        dataset_split="train",
    ),
    VLMThroughputWorkload(
        "image", "image", input_len=None, output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train",
    ),
    VLMThroughputWorkload(
        "video", "video", input_len=None, output_len=512,
        dataset_name="yale-nlp/MMVU", dataset_split="validation",
    ),
    VLMThroughputWorkload(
        "audio", "audio", input_len=None, output_len=256,
        dataset_name="openslr/librispeech_asr", dataset_split="test.clean",
    ),
]

QWEN_OMNI_LATENCY_WORKLOADS: list[VLMLatencyWorkload] = [
    VLMLatencyWorkload(
        "single-text", "text", output_len=128,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["balanced"],
        dataset_split="train",
    ),
    VLMLatencyWorkload(
        "single-image", "image", output_len=128,
        dataset_name="lmarena-ai/VisionArena-Chat", dataset_split="train",
    ),
    VLMLatencyWorkload(
        "single-video", "video", output_len=128,
        dataset_name="yale-nlp/MMVU", dataset_split="validation",
    ),
    VLMLatencyWorkload(
        "single-audio", "audio", output_len=128,
        dataset_name="openslr/librispeech_asr", dataset_split="test.clean",
    ),
]


# ---------------------------------------------------------------------------
# Diffusion workloads (image generation)
# ---------------------------------------------------------------------------

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
    DiffusionThroughputWorkload("512x512",   height=512,  width=512,  batch_size=8, num_requests=10),
]

DIFFUSION_LATENCY_WORKLOADS: list[DiffusionLatencyWorkload] = [
    DiffusionLatencyWorkload("single-1024x1024", height=1024, width=1024),
    DiffusionLatencyWorkload("single-512x512",   height=512,  width=512),
]

ALL_DIFFUSION_WORKLOADS = {
    "throughput": DIFFUSION_THROUGHPUT_WORKLOADS,
    "latency": DIFFUSION_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Segmentation workloads (promptable concept segmentation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentationThroughputWorkload:
    """Workload for segmentation throughput measurement."""
    name: str
    resolution: int
    num_requests: int
    dataset_name: str
    dataset_subset: str = ""
    modality: str = "image"  # "image" or "video"

@dataclass(frozen=True)
class SegmentationLatencyWorkload:
    """Workload for segmentation latency measurement."""
    name: str
    resolution: int
    batch_size: int
    dataset_name: str
    dataset_subset: str = ""
    modality: str = "image"
    num_warmup: int = 3
    num_iters: int = 10

SEGMENTATION_THROUGHPUT_WORKLOADS: list[SegmentationThroughputWorkload] = [
    SegmentationThroughputWorkload(
        "gold-metaclip-nps", resolution=1008, num_requests=500,
        dataset_name="facebook/SACo-Gold", dataset_subset="metaclip_nps",
    ),
    SegmentationThroughputWorkload(
        "gold-wiki-common", resolution=1008, num_requests=500,
        dataset_name="facebook/SACo-Gold", dataset_subset="wiki_common",
    ),
    SegmentationThroughputWorkload(
        "gold-crowded", resolution=1008, num_requests=500,
        dataset_name="facebook/SACo-Gold", dataset_subset="crowded",
    ),
    SegmentationThroughputWorkload(
        "veval-sav-val", resolution=1008, num_requests=100,
        dataset_name="facebook/SACo-VEval", dataset_subset="sav_val",
        modality="video",
    ),
    SegmentationThroughputWorkload(
        "veval-yt1b-val", resolution=1008, num_requests=100,
        dataset_name="facebook/SACo-VEval", dataset_subset="yt1b_val",
        modality="video",
    ),
]

SEGMENTATION_LATENCY_WORKLOADS: list[SegmentationLatencyWorkload] = [
    SegmentationLatencyWorkload(
        "single-image-1008", resolution=1008, batch_size=1,
        dataset_name="facebook/SACo-Gold", dataset_subset="metaclip_nps",
    ),
    SegmentationLatencyWorkload(
        "batch-4-image-1008", resolution=1008, batch_size=4,
        dataset_name="facebook/SACo-Gold", dataset_subset="metaclip_nps",
    ),
    SegmentationLatencyWorkload(
        "single-video-frame-1008", resolution=1008, batch_size=1,
        dataset_name="facebook/SACo-VEval", dataset_subset="smartglasses_val",
        modality="video",
    ),
]

@dataclass(frozen=True)
class SegmentationVideoWorkload:
    """Workload for multi-frame video segmentation benchmark."""
    name: str
    resolution: int
    num_clips: int
    frames_per_clip: int
    dataset_name: str
    dataset_subset: str = ""
    text_prompt: str = "objects"

SEGMENTATION_VIDEO_WORKLOADS: list[SegmentationVideoWorkload] = [
    SegmentationVideoWorkload(
        "sav-val-video", resolution=1008, num_clips=10, frames_per_clip=16,
        dataset_name="facebook/SACo-VEval", dataset_subset="sav_val",
    ),
    SegmentationVideoWorkload(
        "smartglasses-val-video", resolution=1008, num_clips=10, frames_per_clip=16,
        dataset_name="facebook/SACo-VEval", dataset_subset="smartglasses_val",
    ),
]

ALL_SEGMENTATION_WORKLOADS = {
    "throughput": SEGMENTATION_THROUGHPUT_WORKLOADS,
    "latency": SEGMENTATION_LATENCY_WORKLOADS,
    "video": SEGMENTATION_VIDEO_WORKLOADS,
}


# ---------------------------------------------------------------------------
# TTS workloads (text-to-speech, e.g. CosyVoice3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTSModelConfig:
    sample_rate: int
    n_timesteps: int

COSYVOICE3_CONFIG = TTSModelConfig(sample_rate=24000, n_timesteps=10)


@dataclass(frozen=True)
class TTSThroughputWorkload:
    """TTS throughput workload definition.

    Uses the SEED-TTS-Eval dataset for realistic TTS benchmarking.
    Each request has a text prompt and a reference audio for voice cloning.
    """
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

ALL_TTS_WORKLOADS = {
    "throughput": TTS_THROUGHPUT_WORKLOADS,
    "latency": TTS_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Robotics / VLA workloads (Pi0 action generation)
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class RoboticsThroughputWorkload:
    """Robotics VLA throughput workload.

    Uses real robotics datasets (e.g. DROID, Libero) for representative
    image/instruction/state inputs.
    """
    name: str
    num_cameras: int
    num_requests: int
    dataset_name: str
    dataset_split: str = "train"

@dataclass(frozen=True)
class RoboticsLatencyWorkload:
    name: str
    num_cameras: int
    batch_size: int = 1
    dataset_name: str = ""
    num_warmup: int = 3
    num_iters: int = 10


ROBOTICS_THROUGHPUT_WORKLOADS: list[RoboticsThroughputWorkload] = [
    RoboticsThroughputWorkload(
        "libero-1cam", num_cameras=1, num_requests=100,
        dataset_name="lerobot/libero",
    ),
    RoboticsThroughputWorkload(
        "libero-3cam", num_cameras=3, num_requests=100,
        dataset_name="lerobot/libero",
    ),
]

ROBOTICS_LATENCY_WORKLOADS: list[RoboticsLatencyWorkload] = [
    RoboticsLatencyWorkload(
        "single-3cam", num_cameras=3, batch_size=1,
    ),
    RoboticsLatencyWorkload(
        "single-1cam", num_cameras=1, batch_size=1,
    ),
]

ALL_ROBOTICS_WORKLOADS = {
    "throughput": ROBOTICS_THROUGHPUT_WORKLOADS,
    "latency": ROBOTICS_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Object detection workloads (COCO val2017)
# ---------------------------------------------------------------------------

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

ALL_DETECTION_WORKLOADS = {
    "throughput": DETECTION_THROUGHPUT_WORKLOADS,
    "latency": DETECTION_LATENCY_WORKLOADS,
}


def get_max_seq_len() -> int:
    """Return the maximum static sequence length for standardized LLM workloads.

    Throughput decode lengths are data-dependent for real-prompt workloads, so
    eval computes their max sequence length after loading the dataset.
    """
    max_len = 0
    for w in LATENCY_WORKLOADS:
        max_len = max(max_len, w.input_len + w.output_len)
    return max_len


# ---------------------------------------------------------------------------
# Video diffusion workloads (text-to-video generation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoDiffusionModelConfig:
    num_inference_steps: int
    guidance_scale: float

HUNYUAN_VIDEO_CONFIG = VideoDiffusionModelConfig(
    num_inference_steps=30, guidance_scale=6.0,
)

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
    VideoDiffusionThroughputWorkload(
        "480p-short", height=480, width=832, num_frames=25, num_prompts=16,
    ),
    VideoDiffusionThroughputWorkload(
        "480p-medium", height=480, width=832, num_frames=49, num_prompts=8,
    ),
]

VIDEO_DIFFUSION_LATENCY_WORKLOADS: list[VideoDiffusionLatencyWorkload] = [
    VideoDiffusionLatencyWorkload(
        "single-480p-short", height=480, width=832, num_frames=25,
    ),
    VideoDiffusionLatencyWorkload(
        "single-480p-medium", height=480, width=832, num_frames=49,
    ),
]

ALL_VIDEO_DIFFUSION_WORKLOADS = {
    "throughput": VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS,
    "latency": VIDEO_DIFFUSION_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Oasis workloads (autoregressive video diffusion)
# ---------------------------------------------------------------------------

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
    OasisWorkload(
        name="short-bs4-16f-4ddim",
        batch_clips=4,
        num_frames=16,
        ddim_steps=4,
    ),
    OasisWorkload(
        name="medium-bs8-24f-4ddim",
        batch_clips=8,
        num_frames=24,
        ddim_steps=4,
    ),
    OasisWorkload(
        name="long-bs8-32f-4ddim",
        batch_clips=8,
        num_frames=32,
        ddim_steps=4,
    ),
    OasisWorkload(
        name="denoise-bs4-16f-8ddim",
        batch_clips=4,
        num_frames=16,
        ddim_steps=8,
    ),
]

OASIS_LATENCY_WORKLOADS: list[OasisWorkload] = [
    OasisWorkload(
        name="latency-bs1-8f-4ddim",
        batch_clips=1,
        num_frames=8,
        ddim_steps=4,
        kind="latency",
    ),
]

ALL_OASIS_WORKLOADS = {
    "throughput": OASIS_THROUGHPUT_WORKLOADS,
    "latency": OASIS_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Vision encoder workloads (pure image feature extraction, e.g. SigLIP-2, DINOv3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisionEncoderThroughputWorkload:
    """Throughput workload for vision encoders.

    Processes num_images real images from dataset_name at the given resolution
    in fixed batch_size batches, measuring images/sec.
    """
    name: str
    resolution: int
    num_images: int
    batch_size: int
    dataset_name: str = "ILSVRC/imagenet-1k"
    dataset_split: str = "validation"

@dataclass(frozen=True)
class VisionEncoderLatencyWorkload:
    """Latency workload for vision encoders.

    Repeated inference on real images from dataset_name at the model's default
    resolution, measuring median and P99 latency.
    """
    name: str
    resolution: int
    batch_size: int
    dataset_name: str = "ILSVRC/imagenet-1k"
    dataset_split: str = "validation"
    num_warmup: int = 3
    num_iters: int = 10


VISION_ENCODER_THROUGHPUT_WORKLOADS: list[VisionEncoderThroughputWorkload] = [
    VisionEncoderThroughputWorkload("default-res", resolution=0, num_images=5000, batch_size=32),
    VisionEncoderThroughputWorkload("high-res",    resolution=512, num_images=2500, batch_size=16),
]

VISION_ENCODER_LATENCY_WORKLOADS: list[VisionEncoderLatencyWorkload] = [
    VisionEncoderLatencyWorkload("single-image", resolution=0, batch_size=1, num_warmup=5, num_iters=30),
    VisionEncoderLatencyWorkload("batch-8",      resolution=0, batch_size=8, num_warmup=5, num_iters=30),
]

ALL_VISION_ENCODER_WORKLOADS = {
    "throughput": VISION_ENCODER_THROUGHPUT_WORKLOADS,
    "latency": VISION_ENCODER_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Text embedding workloads (token-level retrieval embeddings)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingThroughputWorkload:
    """Throughput workload for token-level text embedding models."""

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
    """Latency workload for token-level text embedding models."""

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

ALL_EMBEDDING_WORKLOADS = {
    "throughput": EMBEDDING_THROUGHPUT_WORKLOADS,
    "latency": EMBEDDING_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Structure prediction workloads (OpenFold3 / AlphaFold3-style models)
# ---------------------------------------------------------------------------

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


STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS: list[
    StructurePredictionThroughputWorkload
] = [
    StructurePredictionThroughputWorkload(
        name="short",
        num_queries=50,
        description="short proteins (<=150 residues) x 50 queries",
    ),
    StructurePredictionThroughputWorkload(
        name="medium",
        num_queries=20,
        description="medium proteins (150-400 residues) x 20 queries",
    ),
    StructurePredictionThroughputWorkload(
        name="long",
        num_queries=10,
        description="long proteins (400-700 residues) x 10 queries",
    ),
    StructurePredictionThroughputWorkload(
        name="extra-long",
        num_queries=5,
        description="extra-long proteins (700+ residues) x 5 queries",
    ),
]

STRUCTURE_PREDICTION_LATENCY_WORKLOADS: list[
    StructurePredictionLatencyWorkload
] = [
    StructurePredictionLatencyWorkload(
        name="single-short",
        length_bucket="short",
    ),
    StructurePredictionLatencyWorkload(
        name="single-medium",
        length_bucket="medium",
    ),
    StructurePredictionLatencyWorkload(
        name="single-long",
        length_bucket="long",
    ),
    StructurePredictionLatencyWorkload(
        name="single-extra-long",
        length_bucket="extra-long",
    ),
]

ALL_STRUCTURE_PREDICTION_WORKLOADS = {
    "throughput": STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS,
    "latency": STRUCTURE_PREDICTION_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# 3-D point-cloud robotics policy workloads (DP3 / Simple-DP3)
# ---------------------------------------------------------------------------

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
# benchmark closest to DP3's MetaWorld setup):
#   - 512 points x 6 channels (XYZRGB; we slice to XYZ for use_pc_color=False)
#   - 7-D joint-state proprioception
#   - 4-D end-effector action (x, y, z, gripper)
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
    point-cloud robotics dataset (default ``rishabhrj11/gym-xarm-pointcloud``
    — 18374 frames over 50 episodes, ``observation.environment_state`` is
    a 512x6 XYZRGB cloud, ``observation.state`` is 7-D, ``action`` is 4-D).
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
    DP3ThroughputWorkload("dp3-1env",  num_requests=100, batch_size=1),
    DP3ThroughputWorkload("dp3-batch", num_requests=100, batch_size=8),
]

DP3_LATENCY_WORKLOADS: list[DP3LatencyWorkload] = [
    DP3LatencyWorkload("single-step", batch_size=1),
    DP3LatencyWorkload("batch-8",     batch_size=8),
]

ALL_DP3_WORKLOADS = {
    "throughput": DP3_THROUGHPUT_WORKLOADS,
    "latency": DP3_LATENCY_WORKLOADS,
}
