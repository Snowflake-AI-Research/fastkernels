#!/usr/bin/env python3
"""Create skeleton replacement modules in tasks/candidate/.

Each stub mirrors the baseline operator's class with identical __init__ and
forward signatures but delegates to the baseline implementation, giving the
user a starting point for writing a custom kernel.

Usage:
    fastkernels create-stubs
    fastkernels create-stubs --level 1
    fastkernels create-stubs --architecture llama
    fastkernels create-stubs --level 1 --architecture mixtral
    fastkernels create-stubs --clear        # remove all stubs/candidates

    # equivalently, as a module:
    python -m fastkernels.utils.create_stubs
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import shutil
import sys
import time

from fastkernels import CANDIDATE_DIR, PREV_ATTEMPTS_DIR

_CANDIDATE_DIR = CANDIDATE_DIR
_PREV_ATTEMPTS_DIR = PREV_ATTEMPTS_DIR


def _resolve_operator_class(module_path: str, class_name: str):
    """Import a baseline operator module and return its primary nn.Module class.

    ``module_path`` is the dotted path relative to the ``fastkernels`` package
    (e.g. ``tasks.baseline.L1.rms_norm``). Prefers the class named by the static
    analyzer; falls back to the last nn.Module subclass defined in the module.
    Importing can fail when the operator pulls an uninstalled optional
    dependency (e.g. ``spconv``); the caller is expected to handle that.
    """
    import torch.nn as nn

    mod = importlib.import_module(f"fastkernels.{module_path}")
    cls = getattr(mod, class_name, None)
    if isinstance(cls, type) and issubclass(cls, nn.Module):
        return cls
    result = None
    for v in vars(mod).values():
        if (isinstance(v, type) and issubclass(v, nn.Module)
                and v is not nn.Module and v.__module__ == mod.__name__):
            result = v
    return result


def _candidate_has_kernels() -> bool:
    if not _CANDIDATE_DIR.exists():
        return False
    for item in _CANDIDATE_DIR.iterdir():
        if item.name in ("README.md", "prev-attempts"):
            continue
        return True
    return False


def _archive_existing_candidates() -> None:
    _PREV_ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = _PREV_ATTEMPTS_DIR / timestamp
    archive_dir.mkdir()
    for item in _CANDIDATE_DIR.iterdir():
        if item.name in ("README.md", "prev-attempts"):
            continue
        shutil.move(str(item), str(archive_dir / item.name))
    print(f"  Archived previous candidates to {archive_dir}")


def _format_parameter(name: str, param: inspect.Parameter) -> str:
    """Render a single parameter for a function signature."""
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
        return f"*{name}"
    if param.kind == inspect.Parameter.VAR_KEYWORD:
        return f"**{name}"

    annotation = param.annotation
    default = param.default

    ann_str = ""
    if annotation is not inspect.Parameter.empty:
        ann_str = _format_annotation(annotation)

    def_str = ""
    if default is not inspect.Parameter.empty:
        def_str = f" = {_format_default(default)}"

    if ann_str:
        return f"{name}: {ann_str}{def_str}"
    return f"{name}{def_str}"


def _format_annotation(ann) -> str:
    if ann is None:
        return "None"
    if isinstance(ann, str):
        return ann
    origin = getattr(ann, "__origin__", None)
    if origin is not None:
        args = getattr(ann, "__args__", ())
        if origin is type(None):
            return "None"
        origin_name = getattr(origin, "__name__", str(origin))
        if origin_name == "Union":
            parts = [_format_annotation(a) for a in args]
            none_parts = [p for p in parts if p == "None"]
            real_parts = [p for p in parts if p != "None"]
            if none_parts and len(real_parts) == 1:
                return f"{real_parts[0]} | None"
            return " | ".join(parts)
        if args:
            arg_strs = ", ".join(_format_annotation(a) for a in args)
            return f"{origin_name}[{arg_strs}]"
        return origin_name
    if hasattr(ann, "__name__"):
        module = getattr(ann, "__module__", "")
        name = ann.__name__
        if module == "torch" and name == "Tensor":
            return "torch.Tensor"
        if module == "torch":
            return f"torch.{name}"
        return name
    return str(ann)


def _format_default(default) -> str:
    if default is None:
        return "None"
    if isinstance(default, bool):
        return str(default)
    if isinstance(default, (int, float)):
        return repr(default)
    if isinstance(default, str):
        return repr(default)
    return repr(default)


def _format_return_annotation(sig: inspect.Signature) -> str:
    if sig.return_annotation is inspect.Signature.empty:
        return ""
    return f" -> {_format_annotation(sig.return_annotation)}"


def _build_signature_str(sig: inspect.Signature, skip_self: bool = True) -> str:
    """Render parameters as a comma-separated string for the def line."""
    parts = []
    for name, param in sig.parameters.items():
        if skip_self and name == "self":
            parts.append("self")
            continue
        parts.append(_format_parameter(name, param))
    return ", ".join(parts)


def _build_call_args(sig: inspect.Signature, skip_self: bool = True) -> str:
    """Render the forwarding call arguments (positional + keyword)."""
    parts = []
    for name, param in sig.parameters.items():
        if skip_self and name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{name}")
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            parts.append(f"{name}={name}")
        else:
            parts.append(name)
    return ", ".join(parts)


def _needs_init_stub(cls: type) -> bool:
    """True when the class defines its own __init__ (not inherited from nn.Module)."""
    import torch.nn as nn
    return "__init__" in cls.__dict__


def _stub_class_block(cls: type) -> str:
    """Render a single ``nn.Module`` stub class (init + forward signatures)."""
    class_name = cls.__name__
    init_sig = inspect.signature(cls.__init__) if _needs_init_stub(cls) else None
    forward_sig = inspect.signature(cls.forward)

    lines = [f"class {class_name}(nn.Module):"]

    if init_sig is not None:
        init_params = _build_signature_str(init_sig)
        lines.append(f"    def __init__({init_params}):")
        lines.append(f"        super().__init__()")
        lines.append(f"        # TODO: implement custom initialization here")
        lines.append(f"        pass")
    else:
        lines.append(f"    def __init__(self):")
        lines.append(f"        super().__init__()")
        lines.append(f"        # TODO: add custom state or buffers here if needed")
        lines.append(f"        pass")

    lines.append("")
    fwd_params = _build_signature_str(forward_sig)
    fwd_ret = _format_return_annotation(forward_sig)
    lines.append(f"    def forward({fwd_params}){fwd_ret}:")
    lines.append(f"        # TODO: implement custom forward kernel here")
    lines.append(f"        # To call a custom CUDA op: result = _custom_ops.my_op(tensor)")
    lines.append(f"        raise NotImplementedError")

    return "\n".join(lines)


def generate_stub(classes, level: int, name: str) -> str:
    """Generate a stub module string covering every target class in a file.

    ``classes`` may be a single class or a list of classes (files with more than
    one non-helper ``nn.Module`` target get all of them in one stub module).
    """
    if not isinstance(classes, (list, tuple)):
        classes = [classes]
    joined = ", ".join(c.__name__ for c in classes)

    lines = []

    lines.append(f'"""Stub replacement(s) for L{level}/{name}: {joined}."""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("")
    lines.append("")
    lines.append("# --- Example: inline CUDA custom op (optional) ---")
    lines.append("# To use a custom CUDA kernel, define it with torch.library and load_inline:")
    lines.append("#")
    lines.append("#   from torch.utils.cpp_extension import load_inline")
    lines.append("#")
    lines.append('#   _CUDA_SRC = r"""')
    lines.append("#   __global__ void my_kernel(const float* in, float* out, int n) {")
    lines.append("#       int i = blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append("#       if (i < n) out[i] = in[i];  // replace with your logic")
    lines.append("#   }")
    lines.append('#   """')
    lines.append("#")
    lines.append('#   _CPP_SRC = r"""')
    lines.append("#   torch::Tensor my_op(torch::Tensor input) {")
    lines.append("#       auto out = torch::empty_like(input);")
    lines.append("#       int n = input.numel();")
    lines.append("#       my_kernel<<<(n+255)/256, 256>>>(")
    lines.append("#           input.data_ptr<float>(), out.data_ptr<float>(), n);")
    lines.append("#       return out;")
    lines.append("#   }")
    lines.append('#   """')
    lines.append("#")
    lines.append("#   _custom_ops = load_inline(")
    lines.append('#       name="my_custom_op",')
    lines.append("#       cpp_sources=[_CPP_SRC],")
    lines.append("#       cuda_sources=[_CUDA_SRC],")
    lines.append('#       functions=["my_op"],')
    lines.append("#       verbose=False,")
    lines.append("#   )")
    lines.append("# -------------------------------------------------")

    for cls in classes:
        lines.append("")
        lines.append("")
        lines.append(_stub_class_block(cls))

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="fastkernels create-stubs",
        description="Create stub replacement modules in tasks/candidate/",
    )
    parser.add_argument(
        "--level", type=int, default=None, choices=[1, 2, 3, 4],
        help="Only create stubs for the given level (default: all)",
    )
    parser.add_argument(
        "--architecture", type=str, default=None,
        help="Only create stubs for operators used by this architecture, "
             "matched against the L4 module stem (e.g. 'llama', 'mixtral'); "
             "see 'fastkernels list --map'.",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Remove all kernels from tasks/candidate/ (moved to prev-attempts/ "
             "so they can be recovered) and exit without creating new stubs. "
             "Clears the whole folder; --level/--architecture are ignored.",
    )
    args = parser.parse_args(argv)

    if args.clear:
        if _candidate_has_kernels():
            _archive_existing_candidates()
            print(f"Cleared candidate kernels from {_CANDIDATE_DIR}")
        else:
            print(f"Nothing to clear: no candidate kernels in {_CANDIDATE_DIR}")
        return

    # Enumerate operators with list.py's pure-static analyzer (ast + filesystem,
    # never imports torch or the operator modules), so discovery never crashes
    # on an operator whose optional dependency is missing.
    from fastkernels.list import discover_operator_targets

    targets = discover_operator_targets()
    if args.level is not None:
        targets = [t for t in targets if t.level == args.level]
    if args.architecture is not None:
        arch = args.architecture.lower()
        matched = [t for t in targets if arch in t.models]
        if not matched:
            avail = sorted({m for t in targets for m in t.models})
            print(f"No operators for architecture {args.architecture!r}. "
                  f"Available: {', '.join(avail)}")
            sys.exit(1)
        targets = matched
    targets = sorted(targets, key=lambda t: (t.level, t.name))

    if not targets:
        print("No operators match the given filters.")
        sys.exit(1)

    if _candidate_has_kernels():
        print("tasks/candidate/ already contains kernels:")
        for item in sorted(_CANDIDATE_DIR.iterdir()):
            if item.name in ("README.md", "prev-attempts"):
                continue
            print(f"  {item.name}/")
        answer = input("Move existing contents to prev-attempts and continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)
        _archive_existing_candidates()

    _CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    # Group by file: a file with several non-helper nn.Module targets gets a
    # single stub module containing all of them.
    from collections import OrderedDict

    by_file: "OrderedDict[tuple[int, str], list]" = OrderedDict()
    for t in targets:
        by_file.setdefault((t.level, t.name), []).append(t)

    print(f"\nCreating stubs for up to {len(by_file)} operator file(s):\n")
    written = 0
    skipped: list[tuple[str, str]] = []
    for (level, name), file_targets in by_file.items():
        module_path = f"tasks.baseline.L{level}.{name}"
        classes = []
        for t in file_targets:
            try:
                cls = _resolve_operator_class(module_path, t.class_name)
            except Exception as exc:  # noqa: BLE001 - usually a missing optional dep
                skipped.append((f"L{level}/{name}:{t.class_name}",
                                f"{type(exc).__name__}: {exc}"))
                continue
            if cls is None:
                skipped.append((f"L{level}/{name}:{t.class_name}",
                                "no nn.Module class found"))
                continue
            classes.append(cls)
        if not classes:
            continue

        level_dir = _CANDIDATE_DIR / f"L{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        out_file = level_dir / f"{name}.py"
        out_file.write_text(generate_stub(classes, level, name))
        written += 1
        print(f"  L{level}/{name}.py  ({', '.join(c.__name__ for c in classes)})")

    print(f"\nDone. {written} stub(s) written to {_CANDIDATE_DIR}")
    if skipped:
        print(f"\nSkipped {len(skipped)} operator(s) whose module could not be "
              f"imported (likely a missing optional dependency):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
    print("\nEdit the forward() methods to add your custom implementations,")
    print("then benchmark with: fastkernels bench --target <name>")


if __name__ == "__main__":
    main()
