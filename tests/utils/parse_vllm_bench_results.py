#!/usr/bin/env python3
"""Parse, summarize and plot bench_vllm.py results.

Automatically discovers all results.json files under tests/results/,
groups them by GPU, prints a markdown summary table, and generates
bar-chart + speedup + alignment plots.

Usage:
    # Auto-discover all results and generate plots (no arguments needed)
    python tests/utils/parse_vllm_bench_results.py

    # Only print the markdown table, skip plots
    python tests/utils/parse_vllm_bench_results.py --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _REPO_ROOT / "tests" / "results"
_PLOTS_DIR = _REPO_ROOT / "tests" / "plots"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_results() -> dict[str, list[Path]]:
    """Walk tests/results/ and return {gpu: [results.json, ...]} sorted."""
    by_gpu: dict[str, list[Path]] = {}
    if not _RESULTS_DIR.is_dir():
        return by_gpu
    for p in sorted(_RESULTS_DIR.rglob("results.json")):
        with open(p) as f:
            data = json.load(f)
        gpu = data.get("gpu", p.parent.parent.name)
        by_gpu.setdefault(gpu, []).append(p)
    return by_gpu


def load_results(paths: list[Path]) -> list[dict]:
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_label(r: dict) -> str:
    """Short human-readable label like '8B TP=1' or 'Mixtral TP=4'."""
    name = r["model"].split("/")[-1]
    short = name
    for prefix in ("Llama-3.1-", "Llama-3-", "Mixtral-", "Meta-"):
        short = short.replace(prefix, "")
    short = short.replace("-Instruct", "").replace("-v0.1", "")
    return f"{short} TP={r['tp']}"


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def format_table(all_results: list[dict], gpu: str) -> str:
    lines: list[str] = []

    lines.append(f"# Benchmark Results — {gpu}")
    lines.append("")

    sorted_results = sorted(all_results, key=lambda r: (r["model"], r["tp"]))

    for r in sorted_results:
        lines.append(f"### {r['model']} (TP={r['tp']})")
        lines.append("")

        scenarios = r.get("scenarios", [])
        if scenarios:
            lines.append("#### Throughput")
            lines.append("")
            lines.append(
                "| Scenario | kb-nano tok/s | vLLM tok/s | Speedup "
                "| Avg Token Match |"
            )
            lines.append("|---|---|---|---|---|")

            for s in scenarios:
                kb = s["kb_nano_tok_per_s"]
                v = s.get("vllm_tok_per_s", 0)
                sp = s.get("speedup", 0)
                a = s.get("alignment", {})
                avg_match = a.get("avg_matching_tokens_per_request", 0)
                avg_out = a.get("avg_output_len", 0)
                match_str = (f"{avg_match:.1f}/{avg_out:.0f}"
                             if avg_out > 0 else "N/A")

                sp_str = f"**{sp:.2f}x**" if sp >= 1.0 else f"{sp:.2f}x"
                lines.append(
                    f"| {s['scenario']} | {kb:,.0f} | {v:,.0f} "
                    f"| {sp_str} | {match_str} |"
                )
            lines.append("")

        lat_scenarios = r.get("latency_scenarios", [])
        if lat_scenarios:
            lines.append("#### Latency")
            lines.append("")
            lines.append(
                "| Scenario | BS | IN | OUT | kb-nano median | vLLM median "
                "| kb-nano ms/tok | vLLM ms/tok | Speedup |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for ls in lat_scenarios:
                kb_med = ls["kb_nano_median_s"]
                kb_mpt = ls["kb_nano_ms_per_tok"]
                v_med = ls.get("vllm_median_s")
                v_mpt = ls.get("vllm_ms_per_tok")
                speedup = ls.get("speedup")
                if speedup is None and ls.get("ratio"):
                    speedup = 1.0 / ls["ratio"]
                sp_str = (f"**{speedup:.2f}x**" if speedup and speedup >= 1.0
                          else (f"{speedup:.2f}x" if speedup else "N/A"))
                v_med_str = f"{v_med:.4f}s" if v_med is not None else "N/A"
                v_mpt_str = f"{v_mpt:.2f}" if v_mpt is not None else "N/A"
                lines.append(
                    f"| {ls['scenario']} | {ls['batch_size']} "
                    f"| {ls['input_len']} | {ls['output_len']} "
                    f"| {kb_med:.4f}s | {v_med_str} "
                    f"| {kb_mpt:.2f} | {v_mpt_str} | {sp_str} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PALETTE = {
    "kb-nano": "#2563eb",
    "vLLM": "#9ca3af",
}


def _make_throughput_fig(all_results: list[dict], gpu: str) -> plt.Figure:
    groups = []
    for r in all_results:
        label = _model_label(r)
        for s in r["scenarios"]:
            groups.append(
                {
                    "label": f"{label}\n{s['scenario']}",
                    "kb": s["kb_nano_tok_per_s"],
                    "vllm": s.get("vllm_tok_per_s", 0),
                }
            )

    n = len(groups)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 6))
    bars_kb = ax.bar(x - w / 2, [g["kb"] for g in groups], w,
                     label="kb-nano", color=_PALETTE["kb-nano"])
    bars_vl = ax.bar(x + w / 2, [g["vllm"] for g in groups], w,
                     label="vLLM", color=_PALETTE["vLLM"])

    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"kb-nano vs vLLM Throughput — {gpu}")
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=8, ha="center")
    ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"
    ))

    for bar in bars_kb:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_vl:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return fig


def _make_speedup_fig(all_results: list[dict], gpu: str) -> plt.Figure:
    items = []
    for r in all_results:
        label = _model_label(r)
        for s in r["scenarios"]:
            sp = s.get("speedup", 0)
            if sp > 0:
                items.append({"label": f"{label} / {s['scenario']}", "speedup": sp})

    if not items:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No speedup data (--skip-vllm?)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    items.reverse()
    n = len(items)
    y = np.arange(n)

    colors = [_PALETTE["kb-nano"] if it["speedup"] >= 1.0 else "#ef4444"
              for it in items]

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.45)))
    bars = ax.barh(y, [it["speedup"] for it in items], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([it["label"] for it in items], fontsize=9)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Speedup (kb-nano / vLLM)")
    ax.set_title(f"Speedup vs vLLM — {gpu}")

    for bar, it in zip(bars, items):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{it['speedup']:.2f}x", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def _make_latency_fig(all_results: list[dict], gpu: str) -> plt.Figure | None:
    groups = []
    for r in all_results:
        label = _model_label(r)
        for ls in r.get("latency_scenarios", []):
            groups.append({
                "label": f"{label}\n{ls['scenario']} (bs={ls['batch_size']})",
                "kb": ls["kb_nano_median_s"] * 1000,
                "vllm": ls.get("vllm_median_s", 0) * 1000,
            })

    if not groups:
        return None

    n = len(groups)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 1.8), 6))
    bars_kb = ax.bar(x - w / 2, [g["kb"] for g in groups], w,
                     label="kb-nano", color=_PALETTE["kb-nano"])
    bars_vl = ax.bar(x + w / 2, [g["vllm"] for g in groups], w,
                     label="vLLM", color=_PALETTE["vLLM"])

    ax.set_ylabel("Median Latency (ms)")
    ax.set_title(f"kb-nano vs vLLM Latency — {gpu}")
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=8, ha="center")
    ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"
    ))

    for bar in bars_kb:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_vl:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return fig


def _make_alignment_fig(all_results: list[dict], gpu: str) -> plt.Figure:
    items = []
    for r in all_results:
        label = _model_label(r)
        for s in r["scenarios"]:
            a = s.get("alignment", {})
            avg_match = a.get("avg_matching_tokens_per_request", 0)
            avg_out = a.get("avg_output_len", 0)
            if avg_out > 0:
                items.append({
                    "label": f"{label}\n{s['scenario']}",
                    "match_rate": avg_match / avg_out,
                })

    if not items:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No alignment data",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    n = len(items)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))
    bars = ax.bar(x, [it["match_rate"] for it in items], color="#8b5cf6")
    ax.set_ylabel("Token Match Rate")
    ax.set_title(f"Token Alignment vs vLLM (greedy, temperature=0) — {gpu}")
    ax.set_xticks(x)
    ax.set_xticklabels([it["label"] for it in items], fontsize=8, ha="center")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    for bar, it in zip(bars, items):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{it['match_rate']:.1%}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse and visualize bench_vllm.py results",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation, only print the table",
    )
    args = parser.parse_args()

    by_gpu = discover_results()
    if not by_gpu:
        print(f"No results.json files found under {_RESULTS_DIR}")
        return

    for gpu, paths in sorted(by_gpu.items()):
        all_results = load_results(paths)
        table_md = format_table(all_results, gpu)
        print(table_md)

        if args.no_plot:
            continue

        plot_dir = _PLOTS_DIR / gpu
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig_tp = _make_throughput_fig(all_results, gpu)
        fig_sp = _make_speedup_fig(all_results, gpu)
        fig_al = _make_alignment_fig(all_results, gpu)
        fig_lat = _make_latency_fig(all_results, gpu)

        fig_tp.savefig(str(plot_dir / "throughput.png"), dpi=150)
        fig_sp.savefig(str(plot_dir / "speedup.png"), dpi=150)
        fig_al.savefig(str(plot_dir / "alignment.png"), dpi=150)
        if fig_lat is not None:
            fig_lat.savefig(str(plot_dir / "latency.png"), dpi=150)
        plt.close("all")

        md_path = plot_dir / "results.md"
        md_path.write_text(table_md + "\n")

        print(f"\nPlots and summary saved to {plot_dir}/\n")


if __name__ == "__main__":
    main()
