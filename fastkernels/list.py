"""List architectures and benchmarks via CLI."""

from __future__ import annotations

import argparse
import textwrap

from .registry import FAMILIES, FASTKERNELS_ARCHITECTURES, DEFAULT_BENCHMARK, BenchmarkScenario


def print_benchmark_set(benchmarks: list[BenchmarkScenario], title: str = "BENCHMARK SCENARIOS") -> None:
    """Print a benchmark set in a tabular format with line-wrapping for long columns."""
    header = f"{'Module':<12} | {'Family':<12} | {'HF Name':<38} | {'TP':<2} | {'dtype':<8} | {'#req':<4} | {'Workloads'}"
    width = 118  # Fixed width that easily fits in standard terminals
    
    print("\n")
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
    print(header)
    print("-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 38 + "-+-" + "-" * 2 + "-+-" + "-" * 8 + "-+-" + "-" * 4 + "-+-" + "-" * 23)
    
    for bs in benchmarks:
        arch = FASTKERNELS_ARCHITECTURES.get(bs.module)
        family = arch.family if arch else "Unknown"
        num_req = str(bs.num_requests) if bs.num_requests is not None else "-"
        wloads = ", ".join(bs.workloads)
        
        # Wrap the workloads column so it doesn't blow out the terminal width
        wrapped_wloads = textwrap.wrap(wloads, width=25, break_on_hyphens=False)
        if not wrapped_wloads:
            wrapped_wloads = [""]
            
        for i, line in enumerate(wrapped_wloads):
            if i == 0:
                print(f"{bs.module:<12} | {family:<12} | {bs.hf_name:<38} | {bs.tp:<2} | {bs.dtype:<8} | {num_req:<4} | {line}")
            else:
                print(f"{'':<12} | {'':<12} | {'':<38} | {'':<2} | {'':<8} | {'':<4} | {line}")
    print("=" * width)


def print_registry() -> None:
    """Print the families and FASTKERNELS_ARCHITECTURES in a tabular format."""
    fam_header = f"{'Family Name':<30} | {'Keyword'}"
    fam_rows = [f"{fam.display_name:<30} | {fam.keyword}" for fam in FAMILIES.values()]
    fam_width = max(len(fam_header), max((len(r) for r in fam_rows), default=0))
    
    print("=" * fam_width)
    print(f"{'FAMILIES':^{fam_width}}")
    print("=" * fam_width)
    print(fam_header)
    print("-" * 30 + "-+-" + "-" * (fam_width - 33))
    for row in fam_rows:
        print(row)
    print("\n")
    
    arch_header = f"{'Architecture':<20} | {'L4 Module':<20} | {'HuggingFace model_type':<24} | {'Family'}"
    arch_rows = []
    for arch in FASTKERNELS_ARCHITECTURES.values():
        m_type = str(arch.model_type) if arch.model_type is not None else "None"
        arch_rows.append(f"{arch.class_name:<20} | {arch.module:<20} | {m_type:<24} | {arch.family}")
        
    arch_width = max(len(arch_header), max((len(r) for r in arch_rows), default=0))
    
    print("=" * arch_width)
    print(f"{'FASTKERNELS ARCHITECTURES':^{arch_width}}")
    print("=" * arch_width)
    print(arch_header)
    print("-" * 20 + "-+-" + "-" * 20 + "-+-" + "-" * 24 + "-+-" + "-" * (arch_width - 73))
    for row in arch_rows:
        print(row)
    print("=" * arch_width)
    
    print_benchmark_set(DEFAULT_BENCHMARK, "DEFAULT BENCHMARK")


def main() -> None:
    print_registry()


if __name__ == "__main__":
    main()
