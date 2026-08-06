"""Shared comparison-metric helpers for the validate harnesses.

Every harness compares a fastkernels baseline against a reference library, but
each grew its own result schema. That made the sweep unanalysable in aggregate:
`20260729-070206` had jobs recording no `speedup` at all (ttt_e2e, convnextv2,
efficientnetv2), jobs recording throughput but no latency (PointTransformerV3,
oasis-500m), and jobs writing their comparison to a differently-named file
(bench_openpi -> summary.json, bench_dllm -> a scenario-named json).

These helpers give every harness one shape:

    {
      "scenarios":         [ <throughput entry>, ... ],
      "latency_scenarios": [ <latency entry>, ... ],
    }

throughput entry (higher is better):
    {"scenario": str, "fastkernels_<metric>": float,
     "reference_<metric>": float, "speedup": ours/ref,
     "alignment": {...}}            # optional

latency entry (lower is better):
    {"scenario": str, "fastkernels_<metric>": float,
     "reference_<metric>": float, "speedup": ref/ours}

``speedup`` is always oriented so >1 means fastkernels is better, which is what
the summary tables and any aggregate analysis assume.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "throughput_entry",
    "latency_entry",
    "alignment_from_token_ids",
    "alignment_from_similarity",
    "standard_results",
]


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    try:
        if denominator == 0:
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ZeroDivisionError):
        return None


def throughput_entry(
    scenario: str,
    ours: float | None,
    reference: float | None,
    *,
    metric: str = "items_per_s",
    alignment: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One throughput comparison. ``speedup = ours / reference`` (higher better)."""
    entry: dict[str, Any] = {
        "scenario": scenario,
        f"fastkernels_{metric}": ours,
        f"reference_{metric}": reference,
        "speedup": _ratio(ours, reference),
    }
    if alignment is not None:
        entry["alignment"] = alignment
    entry.update(extra)
    return entry


def latency_entry(
    scenario: str,
    ours: float | None,
    reference: float | None,
    *,
    metric: str = "median_s",
    **extra: Any,
) -> dict[str, Any]:
    """One latency comparison. ``speedup = reference / ours`` (lower better)."""
    entry: dict[str, Any] = {
        "scenario": scenario,
        f"fastkernels_{metric}": ours,
        f"reference_{metric}": reference,
        "speedup": _ratio(reference, ours),
    }
    entry.update(extra)
    return entry


def alignment_from_token_ids(
    ours: list[list[int]],
    reference: list[list[int]],
) -> dict[str, Any]:
    """Consecutive-prefix token agreement, as the LLM harnesses report it.

    Mirrors the ``alignment`` block that bench_vllm writes, so a single
    aggregate query works across generative harnesses.
    """
    n = min(len(ours), len(reference))
    exact = 0
    total_match = 0
    total_out = 0
    for i in range(n):
        a, b = ours[i], reference[i]
        m = 0
        for x, y in zip(a, b):
            if x != y:
                break
            m += 1
        total_match += m
        total_out += len(b)
        if m == len(b) and len(a) == len(b):
            exact += 1
    return {
        "exact_matches": exact,
        "total_seqs": n,
        "total_matching_tokens": total_match,
        "total_output_tokens": total_out,
        "avg_matching_tokens_per_request": (total_match / n) if n else 0.0,
        "avg_output_len": (total_out / n) if n else 0.0,
    }


def alignment_from_similarity(
    metric_name: str,
    value: float,
    *,
    threshold: float | None = None,
    higher_is_better: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Alignment for non-generative tasks (cosine / match-rate / MAE).

    Non-generative harnesses have no token stream, so they report their own
    similarity metric plus a normalised ``score`` in [0, 1] and a ``passed``
    flag, which keeps them comparable in aggregate queries.

    Set ``higher_is_better=False`` for error metrics (MAE, NLL difference): the
    reported ``score`` is then ``1 - value`` so that 1.0 always means "perfectly
    aligned", and ``passed`` tests ``value <= threshold`` instead of ``>=``.
    """
    value = float(value)
    entry: dict[str, Any] = {
        "metric": metric_name,
        "value": value,
        "score": value if higher_is_better else max(0.0, 1.0 - value),
        "higher_is_better": higher_is_better,
    }
    if threshold is not None:
        entry["threshold"] = float(threshold)
        entry["passed"] = bool(
            value >= threshold if higher_is_better else value <= threshold
        )
    entry.update(extra)
    return entry


def standard_results(
    *,
    model: str,
    throughput: list[dict[str, Any]] | None = None,
    latency: list[dict[str, Any]] | None = None,
    reference_name: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the standard top-level results payload."""
    payload: dict[str, Any] = {
        "model": model,
        "scenarios": list(throughput or []),
        "latency_scenarios": list(latency or []),
    }
    if reference_name:
        payload["reference_name"] = reference_name
    payload.update(extra)
    return payload
