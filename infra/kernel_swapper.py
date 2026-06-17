"""
Kernel target discovery, class monkey-patching, and candidate hot-swapping.

This module consolidates three concerns that were previously split across
bench/kernels/discovery.py, bench/kernels/replacement.py, and
infra/kernel_swapper.py:

1. **Target discovery** -- scans tasks/baseline/L{1-4}/ via static import
   analysis to build BenchTarget objects mapping operators to models.

2. **Class replacement** -- monkey-patches nn.Module classes across all loaded
   modules so that subsequent LlamaEngine construction picks up replacements.

3. **Candidate orchestration** -- scans tasks/candidate/L{level}/ for
   user-provided kernels, matches them against known targets, and applies
   them in bulk.

Used by:
  - bench/kernels/{runner,__main__}.py             (kernel-level benchmarking)
  - bench/e2e/{throughput,latency,serve}.py       (auto-detect all candidates)
  - infra/server.py                               (auto-detect all candidates)
  - agent/{agent,create_stubs}.py               (target introspection)
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn

from kb_nano import BASELINE_DIR, CANDIDATE_DIR, KB_ROOT, REFERENCE_DIR

_KB_ROOT = KB_ROOT
_CANDIDATE_DIR = CANDIDATE_DIR
_REFERENCE_DIR = REFERENCE_DIR

_L4_MODEL_KEYS: dict[str, str] = {
    "bitnet": "bitnet",
    "gla": "gla",
    "gpt_oss": "gpt_oss",
    "llama": "llama31",
    "llama4": "llama4",
    "mamba": "mamba",
    "mamba2": "mamba2",
    "mixtral": "mixtral",
    "convnextv2": "convnextv2",
    "efficientnetv2": "efficientnetv2",
    "qwen2_vl": "qwen2_vl",
    "qwen3_vl": "qwen3_vl",
    "retnet": "retnet",
    "rwkv7": "rwkv7",
    "flux": "flux",
    "gaussian_splatting": "3dgs",
    "instant_ngp": "instantngp",
    "vjepa2": "vjepa2",
    "sam3": "sam3",
    "cosyvoice3": "cosyvoice3",
    "hunyuan_video": "hunyuan_video",
    "bge_m3": "bge_m3",
    "colbertv2": "colbertv2",
    "oasis": "oasis",
    "pointtransformerv3": "pointtransformerv3",
    "dlrmv2": "dlrmv2",
    "lightgcn": "lightgcn",
    "yolov10": "yolov10",
    "rtdetrv2": "rtdetrv2",
    "openfold3": "openfold3",
    "siglip2": "siglip2",
    "dinov3": "dinov3",
}


# ---------------------------------------------------------------------------
# Target discovery
# ---------------------------------------------------------------------------

@dataclass
class BenchTarget:
    name: str
    level: int
    module_path: str
    models: list[str]
    target_cls: type
    # L1 targets sit inside the compiled graph; replacing them requires
    # re-triggering torch.compile.  L2+ targets are behind custom-op
    # boundaries and can be swapped at runtime without recompilation.
    requires_recompile: bool = False


_TARGETS: list[BenchTarget] | None = None


def _resolve_internal_imports(filepath: Path, visited: set[Path] | None = None) -> set[Path]:
    """Recursively collect all internal .py files imported by *filepath*.

    Only follows imports that resolve to files under _KB_ROOT and belong to
    tasks/baseline/L1/, tasks/baseline/L2/, tasks/baseline/L3/,
    tasks/baseline/L4/, or infra/ directories.
    """
    if visited is None:
        visited = set()
    if filepath in visited:
        return visited
    visited.add(filepath)

    try:
        source = filepath.read_text()
    except OSError:
        return visited

    tree = ast.parse(source, filename=str(filepath))
    file_dir = filepath.parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        level = node.level or 0
        if level == 0:
            continue
        base = file_dir
        for _ in range(level - 1):
            base = base.parent

        parts = node.module.split(".")
        candidate = base / "/".join(parts)
        candidate_file = candidate.with_suffix(".py")
        if candidate_file.is_file() and _KB_ROOT in candidate_file.parents:
            _resolve_internal_imports(candidate_file, visited)

    return visited


def _find_module_class(mod) -> type | None:
    """Find the primary nn.Module subclass defined in a target module.

    Returns the last nn.Module class defined in the file, since the
    top-level class is conventionally the final definition.
    """
    result = None
    for v in vars(mod).values():
        if (isinstance(v, type)
                and issubclass(v, nn.Module)
                and v is not nn.Module
                and v.__module__ == mod.__name__):
            result = v
    return result


def _build_model_map() -> dict[str, set[str]]:
    """Build a mapping from relative module path -> set of model keys.

    Walks tasks/baseline/L4/ entry points and traces their imports to build the reverse map.
    """
    l4_dir = _KB_ROOT / "tasks" / "baseline" / "L4"
    if not l4_dir.is_dir():
        return {}

    file_to_models: dict[Path, set[str]] = {}

    for fname in sorted(os.listdir(l4_dir)):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        stem = fname[:-3]
        model_key = _L4_MODEL_KEYS.get(stem)
        if model_key is None:
            continue
        l4_file = l4_dir / fname
        deps = _resolve_internal_imports(l4_file)
        for dep in deps:
            file_to_models.setdefault(dep, set()).add(model_key)

    path_to_models: dict[str, set[str]] = {}
    for fpath, keys in file_to_models.items():
        try:
            rel = fpath.relative_to(_KB_ROOT)
        except ValueError:
            continue
        mod_path = str(rel.with_suffix("")).replace("/", ".")
        path_to_models[mod_path] = keys

    return path_to_models


def discover_targets() -> list[BenchTarget]:
    global _TARGETS
    if _TARGETS is not None:
        return _TARGETS

    pkg_root = __package__.rsplit(".", 1)[0]
    model_map = _build_model_map()
    targets = []

    for level_num in (1, 2, 3, 4):
        level_dir = _KB_ROOT / "tasks" / "baseline" / f"L{level_num}"
        if not level_dir.is_dir():
            continue
        for fname in sorted(os.listdir(level_dir)):
            if fname.startswith("_") or not fname.endswith(".py"):
                continue
            name = fname[:-3]
            module_path = f"tasks.baseline.L{level_num}.{name}"
            models = sorted(model_map.get(module_path, []))
            if not models:
                continue

            mod = importlib.import_module(f"{pkg_root}.{module_path}")
            target_cls = _find_module_class(mod)
            if target_cls is None:
                continue

            targets.append(BenchTarget(
                name=name,
                level=level_num,
                module_path=module_path,
                models=models,
                target_cls=target_cls,
                requires_recompile=(level_num == 1),
            ))

    _TARGETS = targets
    return _TARGETS


def get(name: str) -> BenchTarget:
    targets = discover_targets()
    for t in targets:
        if t.name == name:
            return t
    available = ", ".join(sorted(t.name for t in targets))
    raise KeyError(f"Unknown bench target {name!r}. Available: {available}")


def list_targets(level: int | None = None) -> list[BenchTarget]:
    targets = discover_targets()
    if level is not None:
        targets = [t for t in targets if t.level == level]
    return sorted(targets, key=lambda t: (t.level, t.name))


def models_for_target(name: str) -> list[str]:
    """Return model keys that use a given target operator."""
    return get(name).models


def targets_for_model(model_key: str) -> list[BenchTarget]:
    """Return all targets used by a given model key (e.g. 'llama31', 'mixtral')."""
    return [t for t in discover_targets() if model_key in t.models]


def print_model_operator_map() -> None:
    """Print which operators each model uses, and which models each operator belongs to."""
    targets = discover_targets()

    model_to_targets: dict[str, list[BenchTarget]] = {}
    for t in targets:
        for m in t.models:
            model_to_targets.setdefault(m, []).append(t)

    print(f"\n{'=' * 70}")
    print("  OPERATORS BY MODEL")
    print(f"{'=' * 70}")
    for model_key in sorted(model_to_targets):
        ops = sorted(model_to_targets[model_key], key=lambda t: (t.level, t.name))
        print(f"\n  {model_key}:")
        for t in ops:
            print(f"    L{t.level}  {t.name:<25} {t.target_cls.__name__}")

    print(f"\n{'=' * 70}")
    print("  MODELS BY OPERATOR")
    print(f"{'=' * 70}")
    for t in sorted(targets, key=lambda t: (t.level, t.name)):
        print(f"  L{t.level}  {t.name:<25} {','.join(t.models)}")
    print()


# ---------------------------------------------------------------------------
# Class replacement (monkey-patching)
# ---------------------------------------------------------------------------

def _find_all_references(original_cls: type) -> list[tuple[object, str]]:
    """Find all (module, attr_name) pairs where attr is original_cls."""
    pkg_root = __package__.rsplit(".", 1)[0]
    refs = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith(pkg_root):
            continue
        for attr_name in list(vars(mod)):
            try:
                if getattr(mod, attr_name) is original_cls:
                    refs.append((mod, attr_name))
            except Exception:
                continue
    return refs


def patch_class(target: BenchTarget, user_cls: type) -> list[tuple]:
    """Monkey-patch the target class with user_cls everywhere it's referenced.

    Returns undo info: a list of (module, attr_name, original_cls) tuples.
    """
    original_cls = target.target_cls
    refs = _find_all_references(original_cls)
    undo_info = []
    for mod, attr_name in refs:
        undo_info.append((mod, attr_name, original_cls))
        setattr(mod, attr_name, user_cls)
    return undo_info


def restore(undo_info: list[tuple]) -> None:
    """Restore all original classes from undo info."""
    for mod, attr_name, original_cls in undo_info:
        setattr(mod, attr_name, original_cls)


@contextmanager
def replacement_context(target: BenchTarget, user_cls: type):
    """Context manager that patches the class and restores on exit."""
    undo = patch_class(target, user_cls)
    try:
        yield
    finally:
        restore(undo)


# ---------------------------------------------------------------------------
# Candidate kernel orchestration
# ---------------------------------------------------------------------------

def load_candidate(target_name: str) -> type | None:
    """Load a single candidate kernel for *target_name* from tasks/candidate/.

    Returns the nn.Module subclass if found, or None.
    """
    target = get(target_name)
    candidate_file = _CANDIDATE_DIR / f"L{target.level}" / f"{target_name}.py"
    if not candidate_file.is_file():
        return None
    class_name = target.target_cls.__name__
    module_name = f"_candidate_impl_L{target.level}_{target_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(candidate_file))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    cls = getattr(mod, class_name, None)
    if cls is None:
        for v in vars(mod).values():
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                cls = v
                break
    return cls


def load_reference(target_name: str) -> type | None:
    """Load a semantic PyTorch reference for *target_name* from tasks/reference/.

    References are specification implementations used for prompting and
    correctness validation. They intentionally share the baseline class name
    and public signatures, but are not production-speed baselines.
    """
    target = get(target_name)
    reference_file = _REFERENCE_DIR / f"L{target.level}" / f"{target_name}.py"
    if not reference_file.is_file():
        return None
    class_name = target.target_cls.__name__
    module_name = f"_reference_impl_L{target.level}_{target_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(reference_file))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    cls = getattr(mod, class_name, None)
    if cls is None:
        for v in vars(mod).values():
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                cls = v
                break
    return cls


def discover_candidates() -> list[tuple[BenchTarget, type]]:
    """Scan tasks/candidate/ and return all valid (target, candidate_class) pairs."""
    targets = discover_targets()
    results: list[tuple[BenchTarget, type]] = []

    for target in targets:
        level_dir = _CANDIDATE_DIR / f"L{target.level}"
        candidate_file = level_dir / f"{target.name}.py"
        if not candidate_file.is_file():
            continue
        cls = load_candidate(target.name)
        if cls is not None:
            results.append((target, cls))

    return results


def discover_references() -> list[tuple[BenchTarget, type]]:
    """Scan tasks/reference/ and return valid semantic reference classes."""
    targets = discover_targets()
    results: list[tuple[BenchTarget, type]] = []

    for target in targets:
        level_dir = _REFERENCE_DIR / f"L{target.level}"
        reference_file = level_dir / f"{target.name}.py"
        if not reference_file.is_file():
            continue
        cls = load_reference(target.name)
        if cls is not None:
            results.append((target, cls))

    return results


def _sort_by_level(
    candidates: list[tuple[BenchTarget, type]],
) -> list[tuple[BenchTarget, type]]:
    """Sort candidates by level (L1 first, then L2, L3, L4) for bottom-up patching."""
    return sorted(candidates, key=lambda pair: pair[0].level)


def _detect_subsumption(
    candidates: list[tuple[BenchTarget, type]],
) -> list[tuple[str, int, str, int]]:
    """Detect when higher-level candidates subsume lower-level ones.

    For each candidate at level > 1, walks its import tree and checks
    whether any lower-level candidate targets the same baseline classes.

    Returns list of (higher_name, higher_level, lower_name, lower_level) tuples.
    """
    warnings_list: list[tuple[str, int, str, int]] = []

    low_level_targets: dict[str, BenchTarget] = {}
    high_level_targets: list[tuple[BenchTarget, type]] = []

    for target, user_cls in candidates:
        if target.level == 1:
            low_level_targets[target.name] = target
        if target.level >= 2:
            high_level_targets.append((target, user_cls))

    if not low_level_targets or not high_level_targets:
        return warnings_list

    for higher_target, _ in high_level_targets:
        higher_file = (
            _KB_ROOT / "tasks" / "baseline"
            / f"L{higher_target.level}" / f"{higher_target.name}.py"
        )
        if not higher_file.is_file():
            continue
        deps = _resolve_internal_imports(higher_file)
        for dep_path in deps:
            try:
                dep_rel = dep_path.relative_to(_KB_ROOT)
            except ValueError:
                continue
            dep_stem = dep_rel.stem
            if dep_stem in low_level_targets:
                warnings_list.append((
                    higher_target.name,
                    higher_target.level,
                    dep_stem,
                    low_level_targets[dep_stem].level,
                ))

    return warnings_list


def apply_candidates(candidates: list[tuple[BenchTarget, type]]) -> list[tuple]:
    """Monkey-patch all candidate kernels into their baseline targets.

    Applies patches in bottom-up order (L1 -> L2 -> L3 -> L4) so that
    higher-level baseline code automatically picks up lower-level patches.
    Detects and warns about subsumption conflicts.

    Returns combined undo info that can be passed to ``restore()`` to revert
    all patches.
    """
    sorted_candidates = _sort_by_level(candidates)

    subsumptions = _detect_subsumption(sorted_candidates)
    for higher_name, higher_level, lower_name, lower_level in subsumptions:
        print(
            f"  WARNING: L{higher_level} {higher_name} subsumes "
            f"L{lower_level} {lower_name}\n"
            f"           L{lower_level} {lower_name} candidate will NOT be "
            f"active inside L{higher_level} {higher_name}\n"
            f"           (L{lower_level} {lower_name} candidate IS still "
            f"active for other models/contexts)"
        )

    all_undo: list[tuple] = []
    has_recompile_targets = False
    for target, user_cls in sorted_candidates:
        if target.requires_recompile:
            has_recompile_targets = True
            print(
                f"  NOTE: L{target.level} {target.name} sits inside the "
                f"compiled graph.\n"
                f"        Candidate must be torch.compile-compatible "
                f"(no graph breaks)."
            )
        undo = patch_class(target, user_cls)
        all_undo.extend(undo)
    if has_recompile_targets:
        print("  Candidates with requires_recompile=True need compilation "
              "to be re-triggered.")
    return all_undo


def print_candidate_summary(candidates: list[tuple[BenchTarget, type]]) -> None:
    """Print a human-readable summary of which candidate kernels will be used."""
    if not candidates:
        return
    print(f"\n{'=' * 70}")
    print("  CANDIDATE KERNELS")
    print(f"{'=' * 70}")
    sorted_candidates = _sort_by_level(candidates)
    for target, cls in sorted_candidates:
        print(f"    L{target.level}  {target.name:<25} -> {cls.__name__}")
    print(f"{'=' * 70}\n")


def print_reference_summary(references: list[tuple[BenchTarget, type]]) -> None:
    """Print a human-readable summary of semantic references being used."""
    if not references:
        return
    print(f"\n{'=' * 70}")
    print("  SEMANTIC PYTORCH REFERENCES")
    print(f"{'=' * 70}")
    sorted_references = _sort_by_level(references)
    for target, cls in sorted_references:
        print(f"    L{target.level}  {target.name:<25} -> {cls.__name__}")
    print(f"{'=' * 70}\n")
