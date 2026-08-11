"""List architectures, benchmarks, workloads, and the model<->operator map via CLI.

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

from .workloads import (
    DEFAULT_BENCHMARK,
    FAMILIES,
    FASTKERNELS_ARCHITECTURES,
    WORKLOAD_SPECS,
    BenchmarkScenario,
    Purpose,
    Workload,
    _module_from_name,
    module_for,
    resolve_benchmark,
)


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


def print_workloads() -> None:
    """List every benchmark workload in a tabular format."""
    rows = []
    
    for member, spec in WORKLOAD_SPECS.items():
        family_name = type(member).__name__
        workload_name = f"{family_name}.{member.name}"
        
        ds_name = getattr(spec.params, "dataset_name", "") if spec.params else ""
        if not ds_name:
            ds_name = "-"
            
        purpose_str = "Throughput" if spec.purpose == Purpose.THROUGHPUT else "Latency"
        chars = [purpose_str]
        
        if spec.params:
            for field in ["batch_size", "num_requests", "output_len", "input_len", 
                          "max_text_len", "image_size", "resolution", "height", "width",
                          "batch_clips", "num_frames", "num_images", "num_queries"]:
                if hasattr(spec.params, field):
                    val = getattr(spec.params, field)
                    if val is not None and val != "" and val != 0:
                        chars.append(f"{field}={val}")
                        
        chars_str = ", ".join(chars)
        
        rows.append((family_name, workload_name, str(ds_name), chars_str))
        
    width_name = max(len("Workload Name"), max((len(r[1]) for r in rows), default=0))
    width_ds = max(len("Dataset"), max((len(r[2]) for r in rows), default=0))
    width_chars = max(len("Characteristic"), max((len(r[3]) for r in rows), default=0))
    
    total_width = width_name + width_ds + width_chars + 6
    
    print("\n" + "=" * total_width)
    print(f"{'AVAILABLE WORKLOADS':^{total_width}}")
    print("=" * total_width)
    print(f"{'Workload Name':<{width_name}} | {'Dataset':<{width_ds}} | {'Characteristic'}")
    print("-" * width_name + "-+-" + "-" * width_ds + "-+-" + "-" * width_chars)
    
    prev_family = None
    for r in rows:
        current_family = r[0]
        if prev_family is not None and current_family != prev_family:
            print("-" * total_width)
        print(f"{r[1]:<{width_name}} | {r[2]:<{width_ds}} | {r[3]}")
        prev_family = current_family
        
    print("=" * total_width + "\n")


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


def _declared_targets(tree: ast.Module) -> list[str] | None:
    """The file's explicit ``__targets__ = ["A", "B"]`` declaration, or None.

    A task file may pin exactly which of its ``nn.Module`` classes are operator
    targets (the rest being helpers) with a module-level ``__targets__`` list of
    class-name strings. Files without one fall back to auto-detection.
    """
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__targets__":
                    target = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__targets__":
                target = node.value
        if target is None:
            continue
        if isinstance(target, (ast.List, ast.Tuple)):
            return [
                e.value for e in target.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
        return []
    return None


def _nn_module_class_names(filepath: Path) -> list[str]:
    """Auto-detected non-helper ``nn.Module`` subclasses in *filepath*.

    A class qualifies when it subclasses ``nn.Module`` -- directly, via any
    ``nn.*`` / ``torch.nn.*`` base, or transitively through another class
    defined in the same file -- and its name does not start with an underscore.
    Pure ``ast`` (torch is never imported), preserving source order.
    """
    try:
        tree = ast.parse(filepath.read_text(), filename=str(filepath))
    except (OSError, SyntaxError):
        return []

    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    def _base_is_nn_module(base: ast.expr) -> bool:
        # ``nn.Module`` / ``torch.nn.Module`` and, more broadly, any ``nn.*``
        # (Linear, Conv2d, LayerNorm, ...) which are all nn.Module subclasses.
        if isinstance(base, ast.Attribute):
            if base.attr == "Module":
                return True
            value = base.value
            if isinstance(value, ast.Name) and value.id == "nn":
                return True
            if isinstance(value, ast.Attribute) and value.attr == "nn":
                return True
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
        return False

    cache: dict[str, bool] = {}

    def _is_module(name: str, seen: set[str] | None = None) -> bool:
        if name in cache:
            return cache[name]
        seen = seen or set()
        if name in seen:
            return False
        seen.add(name)
        node = classes.get(name)
        if node is None:
            return False
        result = False
        for base in node.bases:
            if _base_is_nn_module(base) or (
                isinstance(base, ast.Name) and _is_module(base.id, seen)
            ):
                result = True
                break
        cache[name] = result
        return result

    return [
        name
        for name, node in classes.items()
        if not name.startswith("_") and _is_module(name)
    ]


def _target_class_names(filepath: Path) -> list[str]:
    """The operator-target ``nn.Module`` classes for a task file.

    Honors an explicit ``__targets__`` declaration when present (returning only
    those classes, in declared order); otherwise auto-detects every non-helper
    ``nn.Module`` subclass in the file.
    """
    try:
        tree = ast.parse(filepath.read_text(), filename=str(filepath))
    except (OSError, SyntaxError):
        return []
    declared = _declared_targets(tree)
    if declared is not None:
        class_names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        return [name for name in declared if name in class_names]
    return _nn_module_class_names(filepath)


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
            if not models:
                continue
            # One target per non-helper nn.Module class in the file. Skip files
            # no architecture reaches, and class-less files (e.g. the pure
            # -function Triton kernel ``merge_state``).
            for class_name in _target_class_names(path):
                targets.append(OpTarget(path.stem, level, class_name, models))
    return targets


def print_model_operator_map(models: set[str] | None = None) -> None:
    """Print which operators each model uses, and which models each operator belongs to.

    When *models* is given, both sections are restricted to those architecture
    module keys (the scenario view); otherwise every registered architecture is
    covered. Restricting drops non-matching models from each operator's model
    list and hides operators no restricted model reaches.
    """
    targets = discover_operator_targets()
    if models is not None:
        targets = [
            OpTarget(t.name, t.level, t.class_name, kept)
            for t in targets
            if (kept := [m for m in t.models if m in models])
        ]

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


def print_operators_by_level(targets: list[OpTarget], title: str) -> None:
    """Print the unique operators in *targets*, grouped by level (L1 -> L4).

    Each ``OpTarget`` is already a unique ``(level, name, class)`` triple, so the
    union across several models is naturally free of duplicates.
    """
    by_level: dict[int, list[OpTarget]] = {}
    for t in targets:
        by_level.setdefault(t.level, []).append(t)

    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")
    total = 0
    for level in sorted(by_level):
        ops = sorted(by_level[level], key=lambda t: (t.name, t.class_name))
        print(f"\n  L{level} ({len(ops)}):")
        for t in ops:
            print(f"    {t.name:<25} {t.class_name}")
        total += len(ops)
    print(f"\n  Total: {total} operator(s) across {len(by_level)} level(s)")
    print()


def print_scenario_operator_map(scenario_ref: str) -> None:
    """Print the operator map restricted to a scenario's models, then the set of
    operators those models exercise (deduplicated), grouped by level.

    ``scenario_ref`` is a registered scenario name (``full`` / ``default`` /
    ``minimal``) or a path to a scenario YAML. Each model is resolved to its L4
    architecture module, preferring the offline name-based match and falling back
    to the config-driven ``module_for``.
    """
    scenarios = resolve_benchmark(scenario_ref)

    model_to_module: dict[str, str | None] = {}
    for bs in scenarios:
        if bs.hf_name not in model_to_module:
            model_to_module[bs.hf_name] = (
                _module_from_name(bs.hf_name) or module_for(bs.hf_name)
            )

    modules = {m for m in model_to_module.values() if m is not None}
    unresolved = sorted(n for n, m in model_to_module.items() if m is None)

    print(f"\n{'=' * 70}\n  SCENARIO: {scenario_ref}\n{'=' * 70}")
    print(f"  {len(model_to_module)} model(s) -> {len(modules)} architecture(s)\n")
    for hf_name in sorted(model_to_module):
        module = model_to_module[hf_name]
        print(f"    {hf_name:<45} {module if module else '(unresolved)'}")
    if unresolved:
        print(f"\n  Note: {len(unresolved)} model(s) had no architecture match "
              f"and are excluded from the operator map below.")

    print_model_operator_map(models=modules)

    used = [t for t in discover_operator_targets() if any(m in modules for m in t.models)]
    print_operators_by_level(used, "OPERATORS EXERCISED ACROSS SCENARIO (deduplicated)")


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


_MAP_ALL = object()  # ``--map`` given with no scenario argument


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fastkernels list",
        description="List families/architectures/benchmarks, the available "
                    "workloads (--workloads), or the model<->operator map (--map).",
    )
    view = parser.add_mutually_exclusive_group()
    view.add_argument(
        "--map", nargs="?", const=_MAP_ALL, default=None, metavar="SCENARIO",
        help="Print the model<->operator map. With no argument, covers every "
             "registered architecture. Given a SCENARIO (a registered name such "
             "as 'full'/'default'/'minimal', or a path to a scenario YAML), "
             "restrict the map to that scenario's models and also list the "
             "operators they exercise, deduplicated and grouped by level.",
    )
    view.add_argument(
        "--workloads", action="store_true",
        help="Print the available benchmark workloads, grouped by family and by "
             "throughput/latency purpose, instead of the registry.",
    )
    args = parser.parse_args(argv)

    if args.map is not None:
        if args.map is _MAP_ALL:
            print_model_operator_map()
        else:
            try:
                print_scenario_operator_map(args.map)
            except (FileNotFoundError, ValueError) as e:
                parser.error(str(e))
        return

    if args.workloads:
        print_workloads()
        return

    print_registry()


if __name__ == "__main__":
    main()
