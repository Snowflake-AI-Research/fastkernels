#!/usr/bin/env python3
"""Create skeleton replacement modules in tasks/candidate/.

Each stub mirrors the baseline operator's class with identical __init__ and
forward signatures but delegates to the baseline implementation, giving the
user a starting point for writing a custom kernel.

Usage:
    python -m fastkernels.agent.create_stubs
    python -m fastkernels.agent.create_stubs --level 1
    python -m fastkernels.agent.create_stubs --architecture llama
    python -m fastkernels.agent.create_stubs --level 1 --architecture mixtral
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import sys
import time
from pathlib import Path

from fastkernels import CANDIDATE_DIR, PREV_ATTEMPTS_DIR, PROJECT_ROOT

_PROJECT_ROOT = str(PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_CANDIDATE_DIR = CANDIDATE_DIR
_PREV_ATTEMPTS_DIR = PREV_ATTEMPTS_DIR


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


def generate_stub(target) -> str:
    """Generate a stub module string for a BenchTarget."""
    cls = target.target_cls
    class_name = cls.__name__

    init_sig = inspect.signature(cls.__init__) if _needs_init_stub(cls) else None
    forward_sig = inspect.signature(cls.forward)

    lines = []

    lines.append(f'"""Stub replacement for {class_name} (L{target.level}/{target.name})."""')
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
    lines.append("")
    lines.append("")

    lines.append(f"class {class_name}(nn.Module):")

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
    fwd_call = _build_call_args(forward_sig)
    lines.append(f"    def forward({fwd_params}){fwd_ret}:")
    lines.append(f"        # TODO: implement custom forward kernel here")
    lines.append(f"        # To call a custom CUDA op: result = _custom_ops.my_op(tensor)")
    lines.append(f"        raise NotImplementedError")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Create stub replacement modules in tasks/candidate/",
    )
    parser.add_argument(
        "--level", type=int, default=None, choices=[1, 2, 3, 4],
        help="Only create stubs for the given level (default: all)",
    )
    parser.add_argument(
        "--architecture", type=str, default=None,
        help="Only create stubs for operators used by this architecture "
             "(e.g. 'llama', 'mixtral'). Matches against model keys.",
    )
    args = parser.parse_args()

    from fastkernels.infra.kernel_swapper import discover_targets, _L4_MODEL_KEYS

    arch_key = None
    if args.architecture:
        lower = args.architecture.lower()
        arch_key = _L4_MODEL_KEYS.get(lower, lower)

    targets = discover_targets()
    if args.level is not None:
        targets = [t for t in targets if t.level == args.level]
    if arch_key is not None:
        targets = [t for t in targets if arch_key in t.models]
    targets = sorted(targets, key=lambda t: (t.level, t.name))

    if not targets:
        print("No targets match the given filters.")
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

    print(f"\nCreating {len(targets)} stubs:\n")
    for t in targets:
        level_dir = _CANDIDATE_DIR / f"L{t.level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        out_file = level_dir / f"{t.name}.py"

        stub_code = generate_stub(t)
        out_file.write_text(stub_code)
        print(f"  L{t.level}/{t.name}.py  ({t.target_cls.__name__})")

    print(f"\nDone. Stubs written to {_CANDIDATE_DIR}")
    print("Edit the forward() methods to add your custom implementations,")
    print("then benchmark with: fastkernels kernels --target <name>")


if __name__ == "__main__":
    main()
