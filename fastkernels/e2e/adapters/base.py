"""The adapter contract for ``fastkernels e2e``.

An adapter knows how to run one model family end to end. The runner calls it in a fresh
process; when a candidate set is being evaluated, the candidate classes have already been
patched in (``fastkernels.list.apply_candidates``) before ``run`` is called, so the
adapter simply builds and runs the model the normal fastkernels way.

Rules for adapter modules (``fastkernels/e2e/adapters/<name>.py``):

* define exactly one ``Adapter`` subclass; it is discovered automatically;
* import heavy dependencies (torch, engines, datasets) lazily inside methods -- the
  orchestrator imports every adapter module just to call ``handles``/``compare``;
* ``run`` must be deterministic given ``spec.seed`` (same inputs, same noise, same
  sampling for baseline and candidate), so outputs are comparable sample by sample;
* ``compare`` runs on CPU in the orchestrator and must not need a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

# A timing entry: {"kind": "throughput" | "latency", "value": float, "unit": str}.
# Throughput: higher is better (e.g. tok/s, images/s). Latency: lower is better (seconds).
Timing = dict[str, Any]


@dataclass
class RunSpec:
    """What the runner asks an adapter to do."""

    out_dir: str                      # write per-sample outputs to <out_dir>/outputs.pt
    seed: int = 42
    max_requests: int | None = None   # cap on requests/samples per timing workload (None = default)
    correctness_samples: int = 64     # samples used for correctness outputs
    enforce_eager: bool = False       # disable torch.compile / CUDA graphs (diagnostics only)
    reference: str | None = None      # baseline outputs.pt; autoregressive adapters teacher-force on it
    workloads: list[str] | None = None  # optional subset of the scenario's workloads (smoke tests)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class Adapter:
    """One model family. Subclasses override the four members below."""

    name: ClassVar[str] = ""
    # Human-readable description of the per-sample discrepancy ``compare`` returns.
    metric: ClassVar[str] = ""

    @classmethod
    def handles(cls, scenario) -> bool:
        """True if this adapter runs ``scenario`` (a ``workloads.BenchmarkScenario``)."""
        raise NotImplementedError

    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        """Build the model, run the scenario's workloads, return ``{workload: Timing}``.

        Also save the outputs needed for correctness with ``torch.save(obj,
        f"{spec.out_dir}/outputs.pt")`` -- a dict with at least ``{"kind": str}``. Keep it
        compact (well under ~200 MB): ``spec.correctness_samples`` samples, downcast or
        subsample large tensors if needed. If ``spec.reference`` is set, the adapter may use
        the reference outputs (e.g. teacher-forced decoding against reference tokens).
        """
        raise NotImplementedError

    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        """Score candidate outputs against reference outputs (both loaded from outputs.pt).

        Returns ``{"per_sample": [d_0, d_1, ...], "summary": {...}}`` where each ``d`` is a
        discrepancy in [0, 1] (0 = indistinguishable) and ``summary`` holds interpretable
        aggregates (e.g. mean cosine, top-1 agreement).
        """
        raise NotImplementedError
