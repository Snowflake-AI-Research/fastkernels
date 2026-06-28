"""Engine abstraction and registry for E2E benchmarking.

Defines the BenchEngine protocol that all model engines must implement,
and an EngineRegistry that maps model names to engine classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fastkernels.bench.utils.datasets import SampleRequest


@dataclass
class ThroughputResult:
    """Result from a throughput benchmark run."""
    elapsed_time: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    output_tokens_per_second: float = 0.0
    outputs: list[dict] | None = None


@dataclass
class LatencyResult:
    """Result from a latency benchmark run."""
    avg_latency: float = 0.0
    latencies: list[float] = field(default_factory=list)
    percentiles: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class BenchEngine(Protocol):
    """Interface that all model engines must implement for E2E benchmarking."""

    def warmup(self) -> None:
        """Run warmup passes to prime CUDA graphs and caches."""
        ...

    def run_throughput(
        self,
        requests: list[SampleRequest],
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> ThroughputResult:
        """Run offline throughput benchmark."""
        ...

    def run_latency(
        self,
        input_len: int,
        output_len: int,
        batch_size: int,
        num_warmup: int = 10,
        num_iters: int = 30,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> LatencyResult:
        """Run fixed-batch latency benchmark."""
        ...

    def cleanup(self) -> None:
        """Release GPU memory and other resources."""
        ...


class LLMEngine:
    """BenchEngine wrapper around fastkernels's LlamaEngine.

    Handles all LLM architectures supported by LlamaEngine:
    Llama, Mixtral, DeepSeek, Mamba, etc.
    """

    def __init__(
        self,
        model_name: str,
        tp: int = 1,
        seed: int = 42,
        enforce_eager: bool = False,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ):
        self.model_name = model_name
        self.tp = tp
        self.seed = seed
        self.enforce_eager = enforce_eager
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from fastkernels.infra.engine import LlamaEngine
            kwargs: dict[str, Any] = {
                "model_name": self.model_name,
                "seed": self.seed,
                "enforce_eager": self.enforce_eager,
                "tensor_parallel_size": self.tp,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            }
            if self.max_model_len is not None:
                kwargs["max_model_len"] = self.max_model_len
            self._engine = LlamaEngine(**kwargs)
        return self._engine

    def warmup(self) -> None:
        from fastkernels.infra.engine import SamplingParams
        engine = self._get_engine()
        engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    def run_throughput(
        self,
        requests: list[SampleRequest],
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> ThroughputResult:
        import time
        import torch
        from fastkernels.infra.engine import SamplingParams

        engine = self._get_engine()
        prompts = [r.prompt for r in requests]
        sp_list = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=r.expected_output_len,
                ignore_eos=True,
                seed=seed,
            )
            for r in requests
        ]

        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts, sp_list)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_input = sum(
            len(engine.tokenizer.encode(p)) if isinstance(p, str) else len(p)
            for p in prompts
        )
        total_output = sum(len(o.token_ids) for o in outputs)
        total_tokens = total_input + total_output

        return ThroughputResult(
            elapsed_time=elapsed,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            requests_per_second=len(requests) / elapsed,
            tokens_per_second=total_tokens / elapsed,
            output_tokens_per_second=total_output / elapsed,
            outputs=[
                {"generated_text": o.generated_text, "token_ids": o.token_ids}
                for o in outputs
            ],
        )

    def run_latency(
        self,
        input_len: int,
        output_len: int,
        batch_size: int,
        num_warmup: int = 10,
        num_iters: int = 30,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> LatencyResult:
        import time
        import numpy as np
        import torch
        from fastkernels.infra.engine import SamplingParams

        engine = self._get_engine()
        dummy_prompts = np.random.randint(
            10000, size=(batch_size, input_len)
        ).tolist()

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=output_len,
            ignore_eos=True,
            seed=seed,
        )

        for _ in range(num_warmup):
            torch.cuda.synchronize()
            engine.generate(dummy_prompts, sp)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(dummy_prompts, sp)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        lat_arr = np.array(latencies)
        percentages = [10, 25, 50, 75, 90, 99]
        pcts = np.percentile(lat_arr, percentages)

        return LatencyResult(
            avg_latency=float(np.mean(lat_arr)),
            latencies=latencies,
            percentiles={str(p): float(v) for p, v in zip(percentages, pcts)},
        )

    def cleanup(self) -> None:
        if self._engine is not None:
            self._engine._cleanup()
            self._engine = None


class DiffusionBenchEngine:
    """BenchEngine wrapper for diffusion models (FLUX, etc.).

    Measures throughput by images/sec and latency by per-image timing.
    """

    def __init__(
        self,
        model_name: str,
        tp: int = 1,
        seed: int = 42,
        enforce_eager: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.seed = seed
        self.enforce_eager = enforce_eager
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from fastkernels.infra.diffusion_engine import DiffusionEngine
            self._engine = DiffusionEngine(
                model_name=self.model_name,
                seed=self.seed,
                enforce_eager=self.enforce_eager,
            )
        return self._engine

    def warmup(self) -> None:
        engine = self._get_engine()
        engine.warmup()

    def run_throughput(
        self,
        requests: list[SampleRequest],
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> ThroughputResult:
        import time
        import torch
        from fastkernels.tasks.baseline.L4.flux import DiffusionSamplingParams

        engine = self._get_engine()
        prompts = [r.prompt for r in requests]

        params = DiffusionSamplingParams(seed=seed)

        torch.cuda.synchronize()
        start = time.perf_counter()
        output = engine.generate(prompts, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        num_images = len(prompts)
        return ThroughputResult(
            elapsed_time=elapsed,
            total_input_tokens=0,
            total_output_tokens=0,
            requests_per_second=num_images / elapsed,
            tokens_per_second=0.0,
            output_tokens_per_second=0.0,
        )

    def run_latency(
        self,
        input_len: int,
        output_len: int,
        batch_size: int,
        num_warmup: int = 3,
        num_iters: int = 10,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> LatencyResult:
        import time
        import numpy as np
        import torch
        from fastkernels.tasks.baseline.L4.flux import DiffusionSamplingParams

        engine = self._get_engine()
        prompts = [f"A beautiful landscape photo, style {i}" for i in range(batch_size)]
        params = DiffusionSamplingParams(seed=seed, output_type="latent")

        for _ in range(num_warmup):
            torch.cuda.synchronize()
            engine.generate(prompts, params)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        lat_arr = np.array(latencies)
        percentages = [10, 25, 50, 75, 90, 99]
        pcts = np.percentile(lat_arr, percentages)

        return LatencyResult(
            avg_latency=float(np.mean(lat_arr)),
            latencies=latencies,
            percentiles={str(p): float(v) for p, v in zip(percentages, pcts)},
        )

    def cleanup(self) -> None:
        if self._engine is not None:
            self._engine._cleanup()
            self._engine = None


class SegmentationBenchEngine:
    """BenchEngine wrapper for segmentation models (SAM3, etc.).

    Measures throughput by images/sec and latency by per-image timing
    for promptable concept segmentation.
    """

    def __init__(
        self,
        model_name: str,
        tp: int = 1,
        seed: int = 42,
        enforce_eager: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.seed = seed
        self.enforce_eager = enforce_eager
        self._model = None

    def _get_model(self):
        if self._model is None:
            import torch
            from fastkernels.tasks.baseline.L4.sam3 import Sam3Config, Sam3Model

            config = Sam3Config.from_pretrained(self.model_name)
            self._model = Sam3Model(config)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                if not self.enforce_eager:
                    self._model = torch.compile(self._model)
        return self._model

    def warmup(self) -> None:
        import torch
        model = self._get_model()
        device = next(model.parameters()).device
        dummy_img = torch.randn(1, 3, 1008, 1008, device=device)
        dummy_text = torch.randint(0, 49408, (1, 32), device=device)
        with torch.no_grad():
            model(dummy_img, dummy_text)

    def run_throughput(
        self,
        requests: list[SampleRequest],
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> ThroughputResult:
        import time
        import torch

        model = self._get_model()
        device = next(model.parameters()).device

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for r in requests:
                dummy_img = torch.randn(1, 3, 1008, 1008, device=device)
                dummy_text = torch.randint(0, 49408, (1, 32), device=device)
                model(dummy_img, dummy_text)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        num_images = len(requests)
        return ThroughputResult(
            elapsed_time=elapsed,
            requests_per_second=num_images / elapsed,
        )

    def run_latency(
        self,
        input_len: int,
        output_len: int,
        batch_size: int,
        num_warmup: int = 3,
        num_iters: int = 10,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> LatencyResult:
        import time
        import numpy as np
        import torch

        model = self._get_model()
        device = next(model.parameters()).device

        dummy_img = torch.randn(batch_size, 3, 1008, 1008, device=device)
        dummy_text = torch.randint(0, 49408, (batch_size, 32), device=device)

        for _ in range(num_warmup):
            torch.cuda.synchronize()
            with torch.no_grad():
                model(dummy_img, dummy_text)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(dummy_img, dummy_text)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        lat_arr = np.array(latencies)
        percentages = [10, 25, 50, 75, 90, 99]
        pcts = np.percentile(lat_arr, percentages)

        return LatencyResult(
            avg_latency=float(np.mean(lat_arr)),
            latencies=latencies,
            percentiles={str(p): float(v) for p, v in zip(percentages, pcts)},
        )

    def cleanup(self) -> None:
        import torch
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class EngineRegistry:
    """Maps model names to engine classes for automatic dispatch."""

    _registry: dict[str, type] = {}
    _model_patterns: dict[str, type] = {
        "llama": LLMEngine,
        "mixtral": LLMEngine,
        "mistral": LLMEngine,
        "deepseek": LLMEngine,
        "qwen": LLMEngine,
        "mamba": LLMEngine,
        "gemma": LLMEngine,
        "phi": LLMEngine,
        "flux": DiffusionBenchEngine,
        "sam3": SegmentationBenchEngine,
        "hunyuan": DiffusionBenchEngine,
    }

    @classmethod
    def register(cls, name: str, engine_class: type) -> None:
        cls._registry[name] = engine_class

    @classmethod
    def get(cls, model_name: str) -> type:
        """Resolve model name to engine class.

        Checks exact matches in the registry first, then pattern matches
        against known model families.
        """
        if model_name in cls._registry:
            return cls._registry[model_name]

        name_lower = model_name.lower()
        for pattern, engine_cls in cls._model_patterns.items():
            if pattern in name_lower:
                return engine_cls

        return LLMEngine

    @classmethod
    def create(
        cls,
        model_name: str,
        tp: int = 1,
        seed: int = 42,
        enforce_eager: bool = False,
        **kwargs,
    ) -> Any:
        """Create an engine instance for the given model."""
        engine_cls = cls.get(model_name)
        return engine_cls(
            model_name=model_name,
            tp=tp,
            seed=seed,
            enforce_eager=enforce_eager,
            **kwargs,
        )
