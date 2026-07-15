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
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from fastkernels import CANDIDATE_DIR, KB_ROOT

from .workloads import FAMILIES, FASTKERNELS_ARCHITECTURES, DEFAULT_BENCHMARK, BenchmarkScenario, module_for


def print_benchmark_set(benchmarks: list[BenchmarkScenario], title: str = "BENCHMARK SCENARIOS") -> None:
    """Print a benchmark set in a tabular format with line-wrapping for long columns."""
    header = f"{'Module':<12} | {'Family':<12} | {'HF Name':<38} | {'TP':<2} | {'dtype':<8} | {'Workloads'}"
    width = 118  # Fixed width that easily fits in standard terminals
    
    print("\n")
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
    print(header)
    print("-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 38 + "-+-" + "-" * 2 + "-+-" + "-" * 8 + "-+-" + "-" * 23)
    
    for bs in benchmarks:
        module = module_for(bs.hf_name)  # inferred from the HF model's model_type
        arch = FASTKERNELS_ARCHITECTURES.get(module)
        family = arch.family if arch else "Unknown"
        module_disp = module if module is not None else "?"
        wloads = ", ".join(w.value for w in bs.workloads)
        
        # Wrap the workloads column so it doesn't blow out the terminal width
        wrapped_wloads = textwrap.wrap(wloads, width=25, break_on_hyphens=False)
        if not wrapped_wloads:
            wrapped_wloads = [""]
            
        for i, line in enumerate(wrapped_wloads):
            if i == 0:
                print(f"{module_disp:<12} | {family:<12} | {bs.hf_name:<38} | {bs.tp:<2} | {bs.dtype:<8} | {line}")
            else:
                print(f"{'':<12} | {'':<12} | {'':<38} | {'':<2} | {'':<8} | {line}")
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


# ---------------------------------------------------------------------------
# Candidate implementations: discovery + class swapping
#
# ``fastkernels eval`` (baseline vs candidate comparison) and the inference
# server run the baseline operators against user-provided *candidate* operators
# under ``tasks/candidate/L<level>/<name>.py``. Discovery reuses the static
# operator map above; swapping monkey-patches every reference to a baseline
# class with its candidate so a subsequently-built engine picks it up, applied
# bottom-up (L1 -> L4) so higher-level baseline code sees lower-level swaps.
# This replaces the retired ``infra.kernel_swapper``. torch is imported lazily
# inside the functions so ``fastkernels list`` itself stays dependency-free.
# ---------------------------------------------------------------------------
_CANDIDATE_BASELINE_PACKAGE = "fastkernels.tasks.baseline"
_candidates_applied = False


def _load_candidate_class(path: Path, class_name: str):
    """Import a candidate file and return its operator class (prefers the class
    named like the baseline op; falls back to the last ``nn.Module`` defined)."""
    import importlib.util

    import torch.nn as nn

    mod_name = f"_fk_candidate_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    cls = getattr(mod, class_name, None)
    if cls is None:
        for value in vars(mod).values():
            if (isinstance(value, type) and issubclass(value, nn.Module)
                    and value is not nn.Module):
                cls = value
    return cls


def discover_candidate_impls() -> list[tuple]:
    """Return ``(op_target, baseline_cls, candidate_cls)`` for every operator
    that has a candidate implementation under ``tasks/candidate``.

    Sorted L1 -> L4 so lower-level swaps land before the higher-level baseline
    code that imports them is patched.
    """
    import importlib

    pairs: list[tuple] = []
    for target in discover_operator_targets():
        cand_file = CANDIDATE_DIR / f"L{target.level}" / f"{target.name}.py"
        if not cand_file.is_file():
            continue
        try:
            base_mod = importlib.import_module(
                f"{_CANDIDATE_BASELINE_PACKAGE}.L{target.level}.{target.name}")
        except Exception as exc:  # noqa: BLE001 - a broken op module just skips
            print(f"  (skip candidate {target.name}: baseline import failed: "
                  f"{type(exc).__name__}: {exc})")
            continue
        base_cls = getattr(base_mod, target.class_name, None)
        if not isinstance(base_cls, type):
            continue
        try:
            cand_cls = _load_candidate_class(cand_file, target.class_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  (skip candidate {target.name}: {type(exc).__name__}: {exc})")
            continue
        if cand_cls is None:
            continue
        pairs.append((target, base_cls, cand_cls))
    pairs.sort(key=lambda p: p[0].level)
    return pairs


def apply_candidates(pairs: list[tuple]) -> list[tuple]:
    """Monkey-patch each baseline class with its candidate everywhere it is
    referenced across the loaded ``fastkernels`` modules. Returns undo info for
    :func:`restore_candidates`."""
    undo: list[tuple] = []
    for _target, base_cls, cand_cls in pairs:
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith("fastkernels"):
                continue
            try:
                members = list(vars(mod).items())
            except Exception:
                continue
            for attr, value in members:
                if value is base_cls:
                    undo.append((mod, attr, base_cls))
                    setattr(mod, attr, cand_cls)
    return undo


def restore_candidates(undo: list[tuple]) -> None:
    for mod, attr, base_cls in undo:
        setattr(mod, attr, base_cls)


def print_candidate_summary(pairs: list[tuple]) -> None:
    """Human-readable list of which candidate operators will be used."""
    if not pairs:
        return
    print(f"\n{'=' * 70}\n  CANDIDATE OPERATORS\n{'=' * 70}")
    for target, _base, cand in sorted(pairs, key=lambda p: p[0].level):
        print(f"    L{target.level}  {target.name:<25} -> {cand.__name__}")
    print(f"{'=' * 70}\n")


def _apply_candidates_from_env() -> None:
    """Swap the candidate operators in once, at interpreter startup, before the
    engine builds the model. Invoked from the eval sitecustomize in every
    spawned tensor-parallel worker. Idempotent."""
    global _candidates_applied
    if _candidates_applied:
        return
    pairs = discover_candidate_impls()
    if pairs:
        apply_candidates(pairs)
    _candidates_applied = True


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
