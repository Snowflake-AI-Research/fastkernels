"""List architectures, benchmarks, and the model<->operator map via CLI.

The model<->operator map is built with pure static analysis (``ast`` +
filesystem walking) so it has *no* external dependencies: it never imports
``torch`` or the baseline operator modules themselves. It walks
``tasks/baseline/L4/`` entry points, traces their internal relative imports,
and reports which operators each model uses (and vice versa).
"""

from __future__ import annotations

import argparse
import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

from fastkernels import KB_ROOT

from .registry import FAMILIES, FASTKERNELS_ARCHITECTURES, DEFAULT_BENCHMARK, BenchmarkScenario, module_for


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
        module = module_for(bs.hf_name)  # inferred from the HF model's model_type
        arch = FASTKERNELS_ARCHITECTURES.get(module)
        family = arch.family if arch else "Unknown"
        module_disp = module if module is not None else "?"
        num_req = str(bs.num_requests) if bs.num_requests is not None else "-"
        wloads = ", ".join(w.value for w in bs.workloads)
        
        # Wrap the workloads column so it doesn't blow out the terminal width
        wrapped_wloads = textwrap.wrap(wloads, width=25, break_on_hyphens=False)
        if not wrapped_wloads:
            wrapped_wloads = [""]
            
        for i, line in enumerate(wrapped_wloads):
            if i == 0:
                print(f"{module_disp:<12} | {family:<12} | {bs.hf_name:<38} | {bs.tp:<2} | {bs.dtype:<8} | {num_req:<4} | {line}")
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


# ---------------------------------------------------------------------------
# Model <-> operator map (pure static analysis, no external dependencies)
#
# For each registered architecture we take its L4 entry point
# (``tasks/baseline/L4/<module>.py``), trace its transitive *relative* imports,
# and attribute every operator file it reaches to that architecture. Model keys
# are the registry's L4 module stems, so this stays consistent with the "L4
# Module" column above. Everything is read from source with ``ast`` -- torch and
# the operator modules are never imported -- so it runs in a bare environment.
# ---------------------------------------------------------------------------

_BASELINE_DIR = KB_ROOT / "tasks" / "baseline"


@dataclass(frozen=True)
class OpTarget:
    """A baseline operator and the architecture keys that (transitively) use it."""

    name: str
    level: int
    class_name: str
    models: list[str]


def _internal_deps(filepath: Path, seen: set[Path] | None = None) -> set[Path]:
    """Recursively collect .py files reached from *filepath* via relative imports.

    Only follows ``from . / .. import`` targets resolving inside ``KB_ROOT``.
    Pure text analysis: nothing is imported or executed.
    """
    seen = set() if seen is None else seen
    if filepath in seen:
        return seen
    seen.add(filepath)
    try:
        tree = ast.parse(filepath.read_text(), filename=str(filepath))
    except (OSError, SyntaxError):
        return seen
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module and node.level):
            continue
        base = filepath.parent
        for _ in range(node.level - 1):
            base = base.parent
        dep = (base / node.module.replace(".", "/")).with_suffix(".py")
        if dep.is_file() and KB_ROOT in dep.parents:
            _internal_deps(dep, seen)
    return seen


def _last_class_name(filepath: Path) -> str | None:
    """The file's last top-level class (its primary operator), or None."""
    try:
        tree = ast.parse(filepath.read_text(), filename=str(filepath))
    except (OSError, SyntaxError):
        return None
    names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    return names[-1] if names else None


def discover_operator_targets() -> list[OpTarget]:
    """Statically map every baseline operator to the architectures that use it."""
    # operator module path (dotted) -> architecture keys that import it
    op_models: dict[str, set[str]] = {}
    for arch in FASTKERNELS_ARCHITECTURES.values():
        l4_file = _BASELINE_DIR / "L4" / f"{arch.module}.py"
        if not l4_file.is_file():
            continue
        for dep in _internal_deps(l4_file):
            dotted = str(dep.relative_to(KB_ROOT).with_suffix("")).replace("/", ".")
            op_models.setdefault(dotted, set()).add(arch.module)

    targets: list[OpTarget] = []
    for level in (1, 2, 3, 4):
        for path in sorted((_BASELINE_DIR / f"L{level}").glob("*.py")):
            if path.name.startswith("_"):
                continue
            models = sorted(op_models.get(f"tasks.baseline.L{level}.{path.stem}", ()))
            class_name = _last_class_name(path)
            # Skip operators no architecture reaches, and class-less files (e.g.
            # the pure-function Triton kernel ``merge_state``).
            if models and class_name:
                targets.append(OpTarget(path.stem, level, class_name, models))
    return targets


def print_model_operator_map() -> None:
    """Print which operators each model uses, and which models each operator belongs to."""
    targets = discover_operator_targets()

    by_model: dict[str, list[OpTarget]] = {}
    for t in targets:
        for m in t.models:
            by_model.setdefault(m, []).append(t)

    print(f"\n{'=' * 70}\n  OPERATORS BY MODEL\n{'=' * 70}")
    for model in sorted(by_model):
        print(f"\n  {model}:")
        for t in sorted(by_model[model], key=lambda t: (t.level, t.name)):
            print(f"    L{t.level}  {t.name:<25} {t.class_name}")

    print(f"\n{'=' * 70}\n  MODELS BY OPERATOR\n{'=' * 70}")
    for t in sorted(targets, key=lambda t: (t.level, t.name)):
        print(f"  L{t.level}  {t.name:<25} {','.join(t.models)}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fastkernels list",
        description="List families/architectures/benchmarks and the model<->operator map.",
    )
    parser.add_argument(
        "--map", action="store_true",
        help="Print operators-by-model and models-by-operator mappings instead "
             "of the family/architecture/benchmark registry.",
    )
    args = parser.parse_args(argv)

    if args.map:
        print_model_operator_map()
        return

    print_registry()


if __name__ == "__main__":
    main()
