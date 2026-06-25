"""Thin adapter over vLLM's dataset infrastructure.

Re-exports the key types and functions from vllm.benchmarks.datasets so that
fastkernels benchmarks can use the exact same dataset CLI, loading, and sampling
logic -- including all dataset types (random, random-mm, sharegpt, sonnet,
burstgpt, hf, custom, custom_mm, prefix_repetition, etc.) and full multimodal
support.
"""

from vllm.benchmarks.datasets import (
    BenchmarkDataset,
    RandomDataset,
    RandomMultiModalDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    add_dataset_parser,
    add_random_dataset_base_args,
    add_random_multimodal_dataset_args,
    get_samples,
    process_image,
    process_video,
)

__all__ = [
    "BenchmarkDataset",
    "RandomDataset",
    "RandomMultiModalDataset",
    "SampleRequest",
    "ShareGPTDataset",
    "SonnetDataset",
    "add_dataset_parser",
    "add_random_dataset_base_args",
    "add_random_multimodal_dataset_args",
    "get_samples",
    "process_image",
    "process_video",
]
