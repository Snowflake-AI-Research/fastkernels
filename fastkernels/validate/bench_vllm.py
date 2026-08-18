#!/usr/bin/env python3
"""
Throughput and alignment benchmark: fastkernels baseline vs vLLM.

For LLM models: runs three text-only scenarios (prefill-heavy, balanced,
decode-heavy) using WildChat-derived HuggingFace datasets, tokenized with
the target model's chat template.

For VLM models (Qwen2-VL, Qwen3-VL): runs three throughput scenarios
(text-only, image, video) and two latency scenarios (single-image,
single-video) using real multimodal datasets (VisionArena, MMVU). Qwen-Omni
extends this to text, image, video, and audio using real text/multimodal/audio
datasets.

Each engine (vLLM, fastkernels) is loaded once in a single long-lived subprocess
that processes all scenarios sequentially, avoiding repeated model loading.

Usage:
    # LLM benchmark
    python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct

    # VLM benchmark (auto-detected from model name)
    python tests/bench_vllm.py --model Qwen/Qwen2-VL-7B-Instruct

    python tests/bench_vllm.py --skip-vllm  # fastkernels only
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import subprocess

import numpy as np
from transformers import AutoTokenizer


def _detect_gpu_name() -> str:
    """Return short GPU name (e.g. 'H200', 'B200') via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


def _parse_port_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        port = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer TCP port, got {value!r}") from exc
    if not (1 <= port <= 65535):
        raise SystemExit(f"{name} must be between 1 and 65535, got {port}")
    return port


def _reserve_tcp_port(preferred: int | None = None) -> tuple[int, object]:
    """Reserve a local TCP port across concurrent benchmark processes.

    The lock avoids two copies of this script choosing the same port before
    their subprocesses initialize torch/vLLM distributed state.
    """
    min_port = int(os.environ.get("FASTKERNELS_BENCH_PORT_MIN", "20000"))
    max_port = int(os.environ.get("FASTKERNELS_BENCH_PORT_MAX", "60999"))
    if min_port > max_port:
        raise SystemExit("FASTKERNELS_BENCH_PORT_MIN must be <= FASTKERNELS_BENCH_PORT_MAX")

    lock_dir = Path(os.environ.get(
        "FASTKERNELS_BENCH_PORT_LOCK_DIR",
        "/tmp/fastkernels_bench_ports",
    ))
    lock_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[int] = []
    if preferred is not None:
        candidates.append(preferred)
    rng = random.Random((os.getpid() << 16) ^ time.time_ns())
    candidates.extend(rng.sample(range(min_port, max_port + 1),
                                 max_port - min_port + 1))

    for port in candidates:
        lock = open(lock_dir / f"{port}.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
                continue

        return port, lock

    raise SystemExit(
        f"Could not reserve a free local TCP port in {min_port}-{max_port}"
    )


def _make_run_id(requested: str | None) -> str:
    run_id = requested or f"{time.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in run_id)
    safe = safe.strip(".-_")
    if not safe:
        raise SystemExit("--run-id must contain at least one path-safe character")
    return safe


# --- phase-output caching (for --resume) -------------------------------------
# Each of the two heavy phases (vLLM reference, fastkernels engine) is a single
# subprocess whose full result dict (throughput + latency) is persisted as soon
# as it finishes.  On --resume we reload a phase's cache instead of rerunning it,
# but only when the config that determines its outputs is unchanged.

def _fingerprint(**parts) -> str:
    """Stable hash of the config that determines a phase's outputs."""
    import hashlib
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _save_raw(path: str, raw: dict, fingerprint: str) -> None:
    """Persist a phase's raw worker output alongside its config fingerprint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"_fingerprint": fingerprint, "raw": raw}, f)
    os.replace(tmp, path)  # atomic: a crash mid-write never leaves a partial cache


def _load_raw(path: str, fingerprint: str) -> dict | None:
    """Return the cached raw output iff it exists with a matching fingerprint."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    if blob.get("_fingerprint") != fingerprint:
        print(f"  NOTE: cache {os.path.basename(path)} fingerprint mismatch "
              f"(config changed) — ignoring it and rerunning this phase.")
        return None
    return blob.get("raw")


def _install_bench_sitecustomize() -> None:
    """Install a sitecustomize that patches vLLM/FlashInfer in every spawned
    Python process (the v1 EngineCore and TP worker ranks), driven by env vars.

    Two independent patches, each gated by its own env var so this is safe to
    install unconditionally:

    * ``FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE`` -- namespaces FlashInfer IPC
      socket IDs so concurrent TP>1 runs do not collide.
    * ``FASTKERNELS_MAX_LAYERS`` -- for ``--max-layers``: filters out the
      checkpoint weights of pruned decoder layers before they reach vLLM's
      per-model weight loaders. ``hf_overrides`` shrinks the model to the first
      N layers, but the loaders raise ``KeyError`` on the leftover
      ``layers.{i>=N}.*`` tensors from the full checkpoint, so they must be
      dropped from the weight iterator here. A monkeypatch in this process would
      not survive the spawn to EngineCore/TP ranks -- the sitecustomize does.
    """
    site_dir = Path(os.environ.get(
        "FASTKERNELS_FLASHINFER_SITECUSTOMIZE_DIR",
        "/tmp/fastkernels_flashinfer_sitecustomize",
    ))
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(r'''
import os

# vLLM 0.26 FA4 split-KV TYPE_UNSTABLE_JOIN: CuTeDSL JITs from
# inspect.getsourcelines / linecache. Prime the cache with a same-line-count
# hoist so both join paths are Int32. Does not write site-packages.
try:
    import linecache as _lc
    from pathlib import Path as _P
    import vllm as _vllm
    _fa4 = str(_P(_vllm.__file__).resolve().parent
               / "vllm_flash_attn" / "cute" / "flash_fwd_sm100.py")
    _src = _P(_fa4).read_text()
    _old = (
        "                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:\n"
        "                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0\n"
    )
    _new = (
        "                n_block_first = n_block_max - 1 if n_block_max > 0 else Int32(0)\n"
        "                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:\n"
    )
    if _old in _src:
        _st = os.stat(_fa4)
        _lines = _src.replace(_old, _new, 1).splitlines(True)
        _lc.cache[_fa4] = (_st.st_size, _st.st_mtime, _lines, _fa4)
        _real = os.path.realpath(_fa4)
        if _real != _fa4:
            _lc.cache[_real] = (_st.st_size, _st.st_mtime, _lines, _real)
except Exception:
    pass

namespace = os.environ.get("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE")
if namespace:
    try:
        import hashlib
        from flashinfer.comm import mnnvl
    except Exception:
        pass
    else:
        if not getattr(mnnvl.IpcSocket, "_fastkernels_namespaced", False):
            original_init = mnnvl.IpcSocket.__init__
            namespace_bits = int.from_bytes(
                hashlib.blake2b(namespace.encode(), digest_size=8).digest(),
                "little",
            )

            def namespaced_init(self, rank, op_id, use_abstract=True):
                if isinstance(op_id, int):
                    op_id = (op_id ^ namespace_bits) & ((1 << 64) - 1)
                original_init(self, rank, op_id, use_abstract)

            mnnvl.IpcSocket.__init__ = namespaced_init
            mnnvl.IpcSocket._fastkernels_namespaced = True

_max_layers_env = os.environ.get("FASTKERNELS_MAX_LAYERS")
if _max_layers_env:
    try:
        import re as _re
        _n = int(_max_layers_env)
        from vllm.model_executor.model_loader import default_loader as _dl
    except Exception:
        pass
    else:
        if _n >= 1 and not getattr(
            _dl.DefaultModelLoader, "_fastkernels_max_layers", False
        ):
            _orig_get_all = _dl.DefaultModelLoader.get_all_weights
            # Decoder blocks are named "...layers.<i>..." across the Llama /
            # Qwen / Mistral families; vision encoders use "...blocks.<i>...",
            # so they are never matched and stay intact.
            _layer_re = _re.compile(r"(?:^|\.)layers\.(\d+)\.")

            def _get_all_weights_capped(self, model_config, model,
                                        _orig=_orig_get_all, _n=_n,
                                        _pat=_layer_re):
                for name, tensor in _orig(self, model_config, model):
                    m = _pat.search(name)
                    if m is not None and int(m.group(1)) >= _n:
                        continue
                    yield name, tensor

            _dl.DefaultModelLoader.get_all_weights = _get_all_weights_capped
            _dl.DefaultModelLoader._fastkernels_max_layers = True

# FASTKERNELS_ALIGN_PROFILING_KV_BLOCKS -- work around an upstream vLLM 0.26
# bug that stops Kimi-Linear (and any hybrid MLA model whose attention page gets
# padded above 128) from starting on Blackwell.
#
# FlashInfer's trtllm-gen MLA decode kernel validates the *block-table width*:
#     block_num = page_table.shape[-1]; block_size = page_size
#     if block_num % (128 / block_size) != 0: raise
# (flashinfer/mla/_core.py:686-696), where ``page_size`` is the *kernel* page
# size (64 here).
#
# vLLM does align that width -- but against the wrong quantity:
#     max_num_blocks = [cdiv(n, 128 // bs) * (128 // bs) if bs <= 128 else n
#                       for n, bs in zip(max_num_blocks, block_sizes)]
# (v1/worker/block_table.py, upstream #39324). ``block_sizes`` holds the *spec*
# block size, and for Kimi-Linear the hybrid allocator pads attention up to the
# mamba page ("Setting attention block size to 960 tokens to ensure that
# attention page size is >= mamba page size"). 960 > 128, so the ``else n``
# branch skips alignment entirely -- while the kernel still demands
# ``width % 2 == 0`` against its 64-token page. cdiv(max_len, 960) then lands on
# an odd 135 and startup aborts:
#     ValueError: Expected block_num % (128 / block_size) == 0,
#                 got block_num=135 and block_size=64
#
# The fix is the one-line upstream correction: align against
# ``kernel_block_sizes``, which is what the kernel actually reads. Rounding the
# width *up* only adds a couple of unused (-1 padded) block-table columns, so it
# changes no kernel, no page size, no KV capacity and no scheduler setting --
# notably FLASHINFER_MLA is retained. Substituting a slower MLA backend would
# make each reported Kimi speedup a comparison against a handicapped reference.
#
# Drop this once vLLM aligns against kernel_block_sizes upstream.
if os.environ.get("FASTKERNELS_ALIGN_PROFILING_KV_BLOCKS") == "1":
    try:
        import inspect as _inspect

        from vllm.v1.worker.block_table import (
            MultiGroupBlockTable as _MGBT,
        )
    except Exception:
        pass
    else:
        if not getattr(_MGBT, "_fastkernels_kernel_aligned_width", False):
            _orig_mgbt_init = _MGBT.__init__
            _mgbt_sig = _inspect.signature(_orig_mgbt_init)

            def _mgbt_init_kernel_aligned(self, *args, _orig=_orig_mgbt_init,
                                          _sig=_mgbt_sig, **kwargs):
                try:
                    bound = _sig.bind(self, *args, **kwargs)
                    kbs = bound.arguments.get("kernel_block_sizes")
                    mnb = bound.arguments.get("max_num_blocks")
                    if kbs and mnb and len(kbs) == len(mnb):
                        aligned = []
                        for n, kbs_i in zip(mnb, kbs):
                            align = 128 // kbs_i if 0 < kbs_i <= 128 else 1
                            if align > 1 and n % align:
                                n += align - (n % align)
                            aligned.append(n)
                        if aligned != list(mnb):
                            bound.arguments["max_num_blocks"] = aligned
                            args = bound.args[1:]
                            kwargs = bound.kwargs
                except Exception:
                    pass
                return _orig(self, *args, **kwargs)

            _MGBT.__init__ = _mgbt_init_kernel_aligned
            _MGBT._fastkernels_kernel_aligned_width = True
# FASTKERNELS_DSA_DETERMINISTIC_TOPK -- override vLLM's nondeterministic DSA
# top-k (cooperative_topk / persistent_topk / top_k_per_row_{prefill,decode},
# whose atomic-append output ordering makes greedy decode nondeterministic at
# seq>index_topk) with flashinfer.top_k(sorted, deterministic, tie_break=SMALL).
# fastkernels honours the same env flag in its own TopKPerRow, so with this on
# BOTH engines use a bit-reproducible top-k and long-context greedy-match
# becomes a valid gate. Off by default (default = vLLM's native kernels).
if os.environ.get("FASTKERNELS_DSA_DETERMINISTIC_TOPK", "0") != "0":
    try:
        import torch as _t
        import flashinfer as _fi
        from flashinfer import TopKTieBreak as _TB
        import vllm._custom_ops  # noqa: F401  registers torch.ops._C
    except Exception:
        pass
    else:
        def _fk_fi_topk(logits, ks, ke, topk):
            logits = logits.contiguous()
            R, N = logits.shape
            cols = _t.arange(N, device=logits.device)
            valid = (cols.unsqueeze(0) >= ks.reshape(-1).unsqueeze(1).long()) & (
                cols.unsqueeze(0) < ke.reshape(-1).unsqueeze(1).long())
            sen = _t.finfo(logits.dtype).min
            lm = _t.where(valid, logits, _t.full_like(logits, sen))
            vals, idx = _fi.top_k(lm, topk, sorted=False, deterministic=True,
                                  tie_break=int(_TB.SMALL))
            idx = idx.to(_t.int32)
            idx = _t.where(vals <= sen, _t.full_like(idx, -1), idx)
            # Return WINDOW-RELATIVE indices, matching the native
            # top_k_per_row_prefill kernel: the DSA ``convert_indices`` kernel
            # maps ``block_id = idx // block_size`` against the PER-REQUEST
            # block table, so absolute columns of a sequence at a non-zero
            # packed offset overflow that request's block table and get
            # dropped (breaking every non-first sequence in a batch). Decode
            # passes ks=0 (no-op). Then emit index-ASCENDING (value-insensitive
            # so it survives tiny cross-engine logit ULP diffs).
            ksr = ks.reshape(-1, 1).to(idx.dtype)
            idx = _t.where(idx >= 0, idx - ksr, idx)
            _IM = 2147483647
            t = _t.where(idx >= 0, idx, _t.full_like(idx, _IM))
            t, _ = _t.sort(t, dim=-1)
            return _t.where(t == _IM, _t.full_like(t, -1), t)

        def _fk_coop(logits, lengths, output, workspace, k, msl):
            ks = _t.zeros(logits.shape[0], dtype=_t.int32, device=logits.device)
            output.copy_(_fk_fi_topk(logits, ks, lengths.to(_t.int32), k))

        def _fk_prefill(logits, rowStarts, rowEnds, indices, numRows, s0, s1, topK):
            indices.copy_(_fk_fi_topk(logits, rowStarts.to(_t.int32),
                                      rowEnds.to(_t.int32), topK))

        def _fk_decode(logits, next_n, seqLens, indices, numRows, s0, s1, topK):
            sl = seqLens.to(_t.int32)
            if next_n == 1:
                rl = sl.reshape(-1)
            else:
                B = sl.numel() // next_n
                j = _t.arange(next_n, device=sl.device, dtype=_t.int32)
                rl = (sl.reshape(B, 1) - next_n + 1 + j.view(1, next_n)).clamp_min_(0).reshape(-1)
            ks = _t.zeros(numRows, dtype=_t.int32, device=logits.device)
            indices.copy_(_fk_fi_topk(logits, ks, rl, topK))

        try:
            _lib = _t.library.Library("_C", "FRAGMENT")
            _lib.impl("cooperative_topk", _fk_coop, "CUDA")
            _lib.impl("persistent_topk", _fk_coop, "CUDA")
            _lib.impl("top_k_per_row_prefill", _fk_prefill, "CUDA")
            _lib.impl("top_k_per_row_decode", _fk_decode, "CUDA")
            import sys as _sys
            print("[bench sitecustomize] flashinfer deterministic DSA top-k "
                  "override installed", file=_sys.stderr, flush=True)
        except Exception:
            pass
''')

    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if str(site_dir) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(site_dir), *parts])

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
_PACKAGE_NAME = _PACKAGE_DIR.name

sys.path.insert(0, str(_PROJECT_ROOT))

from importlib import import_module

run_worker = import_module(f"{_PACKAGE_NAME}.validate.worker").run_worker
load_real_prompt_workload = import_module(
    f"{_PACKAGE_NAME}.workloads",
).load_real_prompt_workload
_workloads = import_module(f"{_PACKAGE_NAME}.workloads")
(
    ASR_LATENCY_WORKLOADS,
    ASR_THROUGHPUT_WORKLOADS,
    LATENCY_WORKLOADS,
    QWEN_OMNI_LATENCY_WORKLOADS,
    QWEN_OMNI_THROUGHPUT_WORKLOADS,
    THROUGHPUT_WORKLOADS,
    VLM_LATENCY_WORKLOADS,
    VLM_THROUGHPUT_WORKLOADS,
) = (
    _workloads.ASR_LATENCY_WORKLOADS,
    _workloads.ASR_THROUGHPUT_WORKLOADS,
    _workloads.LATENCY_WORKLOADS,
    _workloads.QWEN_OMNI_LATENCY_WORKLOADS,
    _workloads.QWEN_OMNI_THROUGHPUT_WORKLOADS,
    _workloads.THROUGHPUT_WORKLOADS,
    _workloads.VLM_LATENCY_WORKLOADS,
    _workloads.VLM_THROUGHPUT_WORKLOADS,
)

_HELD_PORT_LOCKS: list[object] = []


SCENARIOS = [
    {
        "name": w.name,
        "dataset": w.dataset_name,
    }
    for w in THROUGHPUT_WORKLOADS
]

LATENCY_SCENARIOS = [
    {
        "name": w.name,
        "dataset": w.dataset_name,
        "output_len": w.output_len,
        "batch_size": w.batch_size,
    }
    for w in LATENCY_WORKLOADS
]

VLM_SCENARIOS = [
    {
        "name": w.name,
        "modality": w.modality,
        "input_len": w.input_len,
        "output_len": w.output_len,
        "dataset": w.dataset_name,
        "dataset_split": w.dataset_split,
    }
    for w in VLM_THROUGHPUT_WORKLOADS
]

QWEN_OMNI_SCENARIOS = [
    {
        "name": w.name,
        "modality": w.modality,
        "input_len": w.input_len,
        "output_len": w.output_len,
        "dataset": w.dataset_name,
        "dataset_split": w.dataset_split,
    }
    for w in QWEN_OMNI_THROUGHPUT_WORKLOADS
]

WHISPER_SCENARIOS = [
    {
        "name": w.name,
        "output_len": w.output_len,
        "dataset": w.dataset_name,
        "dataset_split": w.dataset_split,
        "use_full_dataset": w.use_full_dataset,
    }
    for w in ASR_THROUGHPUT_WORKLOADS
]

WHISPER_LATENCY_SCENARIOS = [
    {
        "name": w.name,
        "output_len": w.output_len,
        "batch_size": w.batch_size,
        "dataset": w.dataset_name,
        "dataset_split": w.dataset_split,
    }
    for w in ASR_LATENCY_WORKLOADS
]

VLM_LATENCY_SCENARIOS = [
    {
        "name": w.name,
        "modality": w.modality,
        "output_len": w.output_len,
        "batch_size": w.batch_size,
        "dataset": w.dataset_name,
        "dataset_split": w.dataset_split,
    }
    for w in VLM_LATENCY_WORKLOADS
]

QWEN_OMNI_LATENCY_SCENARIOS = [
    {
        "name": w.name,
        "modality": w.modality,
        "output_len": w.output_len,
        "batch_size": w.batch_size,
        "dataset": w.dataset_name,
        "dataset_split": w.dataset_split,
        "input_len": 128,
    }
    for w in QWEN_OMNI_LATENCY_WORKLOADS
]


def _is_vlm_model(model_name: str) -> bool:
    lower = model_name.lower()
    return "qwen" in lower and "vl" in lower


def _is_whisper_model(model_name: str) -> bool:
    lower = model_name.lower()
    return "whisper" in lower


def _load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
    except AttributeError as exc:
        msg = str(exc)
        if "extra_special_tokens" not in msg and "keys" not in msg:
            raise
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(model_name, "tokenizer_config.json")
        with open(cfg_path) as f:
            tok_cfg = json.load(f)
        extra = tok_cfg.get("extra_special_tokens")
        if not isinstance(extra, list):
            raise
        extra_map = {
            f"extra_special_token_{i}": token
            for i, token in enumerate(extra)
        }
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            extra_special_tokens=extra_map,
        )


def _needs_trust_remote_code(model_name: str) -> bool:
    lower = model_name.lower()
    return "kimi" in lower or "qwen3-next" in lower


def _is_qwen_omni_model(model_name: str) -> bool:
    lower = model_name.lower()
    return "qwen" in lower and "omni" in lower


_PER_MODEL_DEFAULTS: dict[str, dict] = {
    "qwen3-vl-235b": {
        "env": {
            "FASTKERNELS_MAX_CUDAGRAPH_BS": "1024",
            "FASTKERNELS_MAX_ENCODER_TOKENS": "4096",
        },
    },
    "qwen2-vl": {
        # No FASTKERNELS_MAX_ENCODER_TOKENS override. vLLM sets
        # ``max_num_encoder_input_tokens = encoder_cache_size =
        # max_num_batched_tokens`` (16384 here, config/scheduler.py), charged in
        # post-merge multimodal embedding tokens -- the same unit fastkernels'
        # has_mm admission path uses. Capping ours at 4096 admitted 4x fewer
        # images per step (measured: 5.7 vs 22.5 on VisionArena, so 176 prefill
        # steps instead of ~45), which lengthened the ramp and cost ~15% of the
        # image scenario. It was also inconsistent with our own memory profiling:
        # _warmup_vision_encoder already sizes the multimodal reserve for a
        # *full* max_num_batched_tokens encoder batch.
        "gpu_memory_utilization": 0.80,
    },
    "mamba-codestral": {
        # H200 (141 GiB) cannot fit vLLM's default 1024 Mamba cache slots
        # at max_model_len=128000 / util=0.9; B200 can. Cap so CUDA-graph
        # capture can start. fastkernels already self-caps Mamba slots.
        "max_num_seqs": 512,
    },
}


# vLLM reference engines run with vLLM's *own* backend selection, page size
# and config. We deliberately do not override them: substituting a different
# attention backend (e.g. pinning Kimi-Linear's MLA decode to TRITON_MLA to
# dodge the FLASHINFER_MLA block-count assertion) makes the reference slower
# than vLLM actually is, so any reported fastkernels speedup would be measured
# against a handicapped baseline rather than like-for-like.
#
# Known consequence: Kimi-Linear's reference currently cannot start on
# Blackwell. vLLM 0.26 selects FLASHINFER_MLA, whose trtllm-gen decode kernel
# requires ``block_num % (128 // block_size) == 0``; vLLM aligns the
# per-request block-table width to that multiple
# (v1/worker/block_table.py, upstream #39324) but not the total
# ``num_gpu_blocks``, so CUDA-graph memory profiling aborts with
#     ValueError: Expected block_num % (128 / block_size) == 0,
#                 got block_num=2055 and block_size=64
# That is an upstream vLLM bug, and it is reported as a reference-side failure
# rather than worked around here. See docs/vllm-0.26-alignment-audit.md.


def _apply_per_model_defaults(model_name: str, args) -> dict[str, str]:
    """Apply H200-safe defaults without overriding explicit user settings."""
    lower = model_name.lower()
    applied: dict[str, str] = {}
    for key, spec in _PER_MODEL_DEFAULTS.items():
        if key not in lower:
            continue
        for name, value in spec.get("env", {}).items():
            if os.environ.get(name):
                continue
            os.environ[name] = value
            applied[name] = value
        utilization = spec.get("gpu_memory_utilization")
        if utilization is not None and args.gpu_memory_utilization is None:
            args.gpu_memory_utilization = utilization
            applied["gpu_memory_utilization"] = str(utilization)
        max_num_seqs = spec.get("max_num_seqs")
        if max_num_seqs is not None and getattr(args, "max_num_seqs", None) is None:
            # B200 (180 GiB) can fit vLLM's default 1024 Mamba slots; H200
            # cannot. Only apply the cap on Hopper-class cards.
            if _detect_gpu_name() in ("H200", "H100"):
                args.max_num_seqs = max_num_seqs
                applied["max_num_seqs"] = str(max_num_seqs)
    return applied


def _get_model_max_context_len(model_name: str) -> int | None:
    """Return the model's maximum context length (``max_position_embeddings``).

    vLLM (>=0.24.0) strictly rejects a user-specified ``max_model_len`` that
    exceeds the value it derives from the HF config, so the benchmark must cap
    ``global_max_seq_len`` at this length. Returns ``None`` when the config
    cannot be read or does not advertise a positional limit.
    """
    try:
        from vllm.transformers_utils.config import get_config

        # Use vLLM's parser so Mistral-format checkpoints derive their limit
        # from params.json using the same fallback rules as the reference.
        config = get_config(
            model_name,
            trust_remote_code=True,
            config_format="auto",
        )
    except Exception:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception:
            return None
    # Some multimodal configs nest the LM settings under ``text_config``.
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        max_pos = getattr(cfg, "max_position_embeddings", None)
        if isinstance(max_pos, (int, float)) and max_pos > 0:
            return int(max_pos)
    return None


def _chat_template_ids(tokenizer, messages) -> list[int]:
    """Tokenize chat ``messages`` (with generation prompt), normalizing the
    various return types HF tokenizers use (list / Encoding / dict / tensor).

    Base models (e.g. Mamba-Codestral) ship no chat template; rather than let
    ``apply_chat_template`` raise, fall back to a plain completion prompt --
    concatenate the message contents and tokenize with the tokenizer's default
    special tokens (so BOS is still prepended)."""
    if getattr(tokenizer, "chat_template", None) is None:
        text = "\n\n".join((m.get("content") or "") for m in messages).strip()
        return list(tokenizer.encode(text))
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def _fit_messages_to_context(tokenizer, messages, max_prompt_tokens: int) -> list[int]:
    """Chat-template ``messages`` into token ids that fit ``max_prompt_tokens``.

    When the templated prompt is too long, only the *tail of the prompt
    content* is trimmed — the end of the last message that carries real
    content — and the chat template is then re-applied. Because truncation
    happens on the raw content *before* templating, no special/template tokens
    are ever dropped: the BOS, the role headers before the content, and the
    trailing generation prompt (``<|end_header_id|>`` …) are all preserved.
    """
    ids = _chat_template_ids(tokenizer, messages)
    if max_prompt_tokens < 1 or len(ids) <= max_prompt_tokens:
        return ids
    # Trim the last message that actually has content (the tail of the prompt).
    target = None
    for i in range(len(messages) - 1, -1, -1):
        if (messages[i].get("content") or "").strip():
            target = i
            break
    if target is None:
        return ids  # nothing trimmable (all-empty content) -- leave as-is
    messages = [dict(m) for m in messages]  # don't mutate the caller's list
    content_ids = tokenizer.encode(
        messages[target]["content"], add_special_tokens=False,
    )
    keep = max(0, len(content_ids) - (len(ids) - max_prompt_tokens))
    # Re-template and shrink until it fits; a few iterations absorb the token
    # drift from BPE re-merging at the cut point and per-turn template overhead.
    for _ in range(8):
        messages[target]["content"] = tokenizer.decode(content_ids[:keep])
        ids = _chat_template_ids(tokenizer, messages)
        if len(ids) <= max_prompt_tokens or keep == 0:
            break
        keep = max(0, keep - (len(ids) - max_prompt_tokens) - 8)
    return ids


# ---------------------------------------------------------------------------
# Multi-scenario vLLM subprocess worker (LLM, text-only)
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def _configure_parallel_safe_flashinfer():
    namespace = os.environ.get("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE")
    if not namespace:
        return
    try:
        import hashlib
        from flashinfer.comm import mnnvl
    except Exception:
        return
    if getattr(mnnvl.IpcSocket, "_fastkernels_namespaced", False):
        return

    original_init = mnnvl.IpcSocket.__init__
    namespace_bits = int.from_bytes(
        hashlib.blake2b(namespace.encode(), digest_size=8).digest(),
        "little",
    )

    def namespaced_init(self, rank, op_id, use_abstract=True):
        if isinstance(op_id, int):
            op_id = (op_id ^ namespace_bits) & ((1 << 64) - 1)
        original_init(self, rank, op_id, use_abstract)

    mnnvl.IpcSocket.__init__ = namespaced_init
    mnnvl.IpcSocket._fastkernels_namespaced = True

_configure_parallel_safe_flashinfer()

def _set_cfg_attr(cfg, name, value):
    try:
        setattr(cfg, name, value)
        return
    except Exception:
        pass
    try:
        cfg.__dict__[name] = value
    except Exception:
        try:
            object.__setattr__(cfg, name, value)
        except Exception:
            pass


def _fastkernels_limit_layers(hf_config):
    # Gemma-4: transformers 5.x hides per-layer attrs. Enable global reads
    # and stash the full-attention head dim so vLLM does not build every
    # layer at sliding-window 256 (checkpoint weights for full attn are 512).
    # global_head_dim is consumed in Gemma4TextConfig.__post_init__ and is
    # not kept as an attribute, so vLLM's getattr(..., "global_head_dim",
    # config.head_dim) would otherwise fall back to 256.
    candidates = [hf_config, getattr(hf_config, "text_config", None)]
    try:
        candidates.append(hf_config.get_text_config())
    except Exception:
        pass
    max_head_dim = 0
    for cfg in candidates:
        if cfg is None:
            continue
        _set_cfg_attr(cfg, "allow_global_per_layer_attribute_access", True)
        try:
            for layer_cfg in cfg.per_layer_config:
                hd = getattr(layer_cfg, "head_dim", 0) or 0
                if hd > max_head_dim:
                    max_head_dim = int(hd)
        except Exception:
            pass
    if max_head_dim:
        for cfg in candidates:
            if cfg is None:
                continue
            _set_cfg_attr(cfg, "global_head_dim", max_head_dim)
    # --max-layers: build only the first N decoder layers. get_text_config()
    # returns the text sub-config for multimodal models and the config itself
    # for pure-text models, so this limits the transformer stack in both. N is
    # read from the env so this module-level function pickles cleanly into
    # vLLM's spawned EngineCore (a nested closure would not).
    n = os.environ.get("FASTKERNELS_MAX_LAYERS")
    if n:
        tc = hf_config.get_text_config()
        if getattr(tc, "num_hidden_layers", None):
            tc.num_hidden_layers = min(tc.num_hidden_layers, int(n))
    return hf_config

def _patch_gemma4_head_size():
    # convert() runs hasattr() on per-layer attrs. AmbiguousGlobalPerLayerAttributeError
    # subclasses RuntimeError, so hasattr does not swallow it. Also keep
    # get_head_size / get_total_num_kv_heads on the max per-layer values.
    try:
        from transformers.integrations.heterogeneity.configuration_utils import (
            AmbiguousGlobalPerLayerAttributeError,
        )
    except Exception:
        AmbiguousGlobalPerLayerAttributeError = ()
    try:
        from vllm.config import utils as vllm_config_utils
        orig_getattr_iter = vllm_config_utils.getattr_iter
        if not getattr(orig_getattr_iter, "_fk_patched", False):
            def getattr_iter(object, names, default=None, default_factory=None, warn=False):
                for name in names:
                    try:
                        has = hasattr(object, name)
                    except AmbiguousGlobalPerLayerAttributeError:
                        has = True
                    if not has:
                        continue
                    try:
                        return getattr(object, name)
                    except AmbiguousGlobalPerLayerAttributeError:
                        getter = getattr(
                            object, "_getattr_without_heterogeneous_validation", None
                        )
                        if getter is not None:
                            try:
                                return getter(name)
                            except Exception:
                                pass
                return default_factory() if default_factory is not None else default
            getattr_iter._fk_patched = True
            vllm_config_utils.getattr_iter = getattr_iter
            try:
                import vllm.transformers_utils.model_arch_config_convertor as conv
                conv.getattr_iter = getattr_iter
            except Exception:
                pass
    except Exception:
        pass
    try:
        from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
        from vllm.model_executor.models.utils import extract_layer_index
        orig_init = Gemma4DecoderLayer.__init__
        if not getattr(orig_init, "_fk_patched", False):
            def _gemma4_layer_init(
                self, config, cache_config=None, quant_config=None, prefix="",
            ):
                # Full-attn layers are 512-d / 2 KV heads; sliding are 256-d / 8.
                # Enabling global per-layer reads makes every layer see n_kv=8.
                try:
                    layer_idx = extract_layer_index(prefix)
                    class _LayerCfg:
                        def __init__(self, cfg, idx):
                            object.__setattr__(self, "_cfg", cfg)
                            object.__setattr__(self, "_idx", idx)
                        def __getattr__(self, name):
                            if name == "num_key_value_heads":
                                return self._cfg.per_layer_config[
                                    self._idx
                                ].num_key_value_heads
                            return getattr(self._cfg, name)
                    config = _LayerCfg(config, layer_idx)
                except Exception:
                    pass
                orig_init(
                    self, config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
            _gemma4_layer_init._fk_patched = True
            Gemma4DecoderLayer.__init__ = _gemma4_layer_init
    except Exception:
        pass
    try:
        from vllm.transformers_utils.model_arch_config_convertor import (
            Gemma4ModelArchConfigConvertor,
        )
    except Exception:
        return
    if getattr(Gemma4ModelArchConfigConvertor.get_head_size, "_fk_patched", False):
        return

    def _max_per_layer(tc, names, default):
        values = []
        try:
            for layer_cfg in tc.per_layer_config:
                for name in names:
                    val = getattr(layer_cfg, name, 0) or 0
                    if val:
                        values.append(int(val))
        except Exception:
            pass
        getter = getattr(tc, "_getattr_without_heterogeneous_validation", None)
        if getter is not None:
            for name in names:
                try:
                    val = getter(name)
                    if val:
                        values.append(int(val))
                except Exception:
                    pass
        return max(values) if values else default

    def get_head_size(self):
        return _max_per_layer(
            self.hf_text_config, ("head_dim", "global_head_dim"), 256
        )

    def get_total_num_kv_heads(self):
        return _max_per_layer(
            self.hf_text_config,
            ("num_key_value_heads", "num_global_key_value_heads"),
            1,
        )

    get_head_size._fk_patched = True
    Gemma4ModelArchConfigConvertor.get_head_size = get_head_size
    Gemma4ModelArchConfigConvertor.get_total_num_kv_heads = get_total_num_kv_heads

def main():
    from vllm import LLM, SamplingParams
    _patch_gemma4_head_size()

    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    llm_kwargs = dict(
        model=cfg["model"],
        seed=cfg["seed"],
        trust_remote_code=True,
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
    )
    if cfg.get("trust_remote_code"):
        llm_kwargs["trust_remote_code"] = True
    if cfg["tp"] > 1:
        # Pin multi-GPU execution to multiprocessing. vLLM's auto-selection
        # falls back to "ray" whenever Ray is initialized with a placement
        # group in the calling process, which would nest a second Ray cluster
        # inside the validate runner's. Only set for tp>1: at tp=1 vLLM picks
        # "uni" and runs in-process, and forcing "mp" there would spawn a
        # worker subprocess for no benefit.
        llm_kwargs["distributed_executor_backend"] = "mp"
    if cfg.get("is_qwen_omni", False):
        llm_kwargs["limit_mm_per_prompt"] = {
            "image": 0,
            "video": 0,
            "audio": 0,
        }
    if cfg.get("load_format"):
        llm_kwargs["load_format"] = cfg["load_format"]
    if cfg.get("kv_cache_dtype"):
        llm_kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    if cfg.get("max_num_seqs") is not None:
        llm_kwargs["max_num_seqs"] = cfg["max_num_seqs"]
    llm_kwargs["hf_overrides"] = _fastkernels_limit_layers
    # Reference-only backend overrides for models vLLM's default selection
    # cannot run on this hardware (see _REFERENCE_ENGINE_OVERRIDES).
    llm = LLM(**llm_kwargs)

    # Warmup -- ignore_eos so all 16 decode steps run (parity with the engines).
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
    )

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompt_token_ids = scenario["prompt_token_ids"]
        output_lens = scenario["output_lens"]
        temperature = cfg.get("temperature", 0.0)

        sp_list = [
            SamplingParams(
                temperature=temperature,
                ignore_eos=True,
                max_tokens=ol,
                detokenize=False,
            )
            for ol in output_lens
        ]

        vllm_prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
        # Prefill warmup at this scenario's real shapes. The engine-level
        # warmup above is a 16-token batch of 1, so without this the first
        # timed generate() absorbs the Triton/CuTeDSL JIT for the scenario's
        # prefill shapes. max_tokens=1 covers prefill without paying decode.
        llm.generate(
            vllm_prompts,
            SamplingParams(temperature=temperature, ignore_eos=True,
                           max_tokens=1, detokenize=False),
            use_tqdm=False,
        )
        start = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sp_list, use_tqdm=True)
        elapsed = time.perf_counter() - start

        total_prompt_tokens = sum(
            len(o.prompt_token_ids) if o.prompt_token_ids else 0
            for o in outputs
        )
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs if c)
            for o in outputs
        )

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {
                    "text": o.outputs[0].text,
                    "token_ids": list(o.outputs[0].token_ids),
                }
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = [dict(prompt_token_ids=p) for p in ls["prompt_token_ids"]]
        output_lens = ls.get("output_lens")
        if output_lens is None:
            sp = SamplingParams(temperature=0.0,
                                ignore_eos=True, max_tokens=ls["output_len"])
        else:
            sp = [
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
                for ol in output_lens
            ]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            llm.generate(prompts, sp, use_tqdm=False)
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del llm

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Multi-scenario fastkernels subprocess worker
# ---------------------------------------------------------------------------
FASTKERNELS_WORKER = r'''
import json, os, sys, time
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    if "max_layers" in cfg:
        engine_kwargs["max_layers"] = cfg["max_layers"]
    if cfg.get("kv_cache_dtype"):
        engine_kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    if cfg.get("max_num_seqs") is not None:
        engine_kwargs["max_num_seqs"] = cfg["max_num_seqs"]
    engine = LlamaEngine(**engine_kwargs)

    # Warmup -- same 16-token prompt as the vLLM worker, so both sides enter
    # the scenario loop having done identical work. ignore_eos so the 16 decode
    # steps run even if token 0 greedily decodes to EOS (parity with fla/jamba).
    engine.generate([[0] * 16],
                    SamplingParams(temperature=0.0, max_tokens=16,
                                   ignore_eos=True))

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompts = scenario["prompt_token_ids"]
        output_lens = scenario["output_lens"]
        temperature = cfg.get("temperature", 0.0)
        top_p = cfg.get("top_p", 1.0)

        sp_list = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=ol,
                ignore_eos=True,
            )
            for ol in output_lens
        ]

        # Prefill warmup at this scenario's real shapes -- see the matching
        # comment in the vLLM worker. capture_mamba_cudagraph() and friends
        # cover decode, but nothing runs prefill through the compiled model
        # before timing, so the first timed generate() would otherwise absorb
        # a Triton JIT/autotune spike. The reset() below frees what this
        # allocated (finished seqs release their Mamba state slots).
        engine.generate(
            prompts,
            SamplingParams(temperature=temperature, top_p=top_p,
                           max_tokens=1, ignore_eos=True),
            use_tqdm=False,
            decode_text=False,
        )

        engine.block_manager.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(
            prompts,
            sp_list,
            use_tqdm=True,
            decode_text=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_input_tokens = sum(len(p) for p in prompts)
        total_output_tokens = sum(len(o.token_ids) for o in outputs)

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {
                    "generated_text": o.generated_text,
                    "token_ids": o.token_ids,
                }
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompt_token_ids"]
        output_lens = ls.get("output_lens")
        if output_lens is None:
            sp = SamplingParams(temperature=0.0,
                                ignore_eos=True, max_tokens=ls["output_len"])
        else:
            sp = [
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
                for ol in output_lens
            ]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Shared multimodal data loading (no vLLM imports -- cv2, numpy, PIL only)
# Inlined into both VLM workers so each subprocess is self-contained.
# ---------------------------------------------------------------------------
_MM_PRELOAD_FN = r'''
import math
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm


def _decode_audio_array(audio):
    """Decode a HF Audio item to mono float32 samples without torchcodec."""
    if isinstance(audio, dict) and audio.get("array") is not None:
        samples = np.asarray(audio["array"], dtype=np.float32)
        return samples, int(audio["sampling_rate"])

    import av

    source = None
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            source = BytesIO(audio["bytes"])
        elif audio.get("path") is not None:
            source = audio["path"]
    if source is None:
        raise ValueError("Unsupported audio sample format")

    chunks = []
    sampling_rate = None
    with av.open(source) as container:
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            sampling_rate = frame.sample_rate
            chunks.append(arr)
    if not chunks or sampling_rate is None:
        raise ValueError("Audio sample has no decodable frames")

    samples = np.concatenate(chunks, axis=-1)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        samples = samples.astype(np.float32) / max(abs(info.min), info.max)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=0)
    return samples, int(sampling_rate)

def _load_video_opencv(video_path, num_frames=32):
    """Load video frames with OpenCV, matching vLLM's OpenCVVideoBackend."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames_num / original_fps if original_fps > 0 else 0

    num_frames_to_sample = total_frames_num
    if num_frames > 0:
        num_frames_to_sample = min(num_frames, total_frames_num)
    num_frames_to_sample = max(1, num_frames_to_sample)

    if num_frames_to_sample == total_frames_num:
        frame_idx = list(range(num_frames_to_sample))
    else:
        frame_idx = np.linspace(
            0, total_frames_num - 1, num_frames_to_sample, dtype=int
        ).tolist()

    frame_idx_set = set(frame_idx)
    max_idx = max(frame_idx)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.empty((num_frames_to_sample, height, width, 3), dtype=np.uint8)

    i = 0
    valid_frame_indices = []
    for idx in range(max_idx + 1):
        ok = cap.grab()
        if not ok:
            continue
        if idx in frame_idx_set:
            ret, frame = cap.retrieve()
            if ret:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                valid_frame_indices.append(idx)
                i += 1

    cap.release()
    valid_num_frames = len(valid_frame_indices)
    frames = frames[:valid_num_frames]

    metadata = {
        "total_num_frames": total_frames_num,
        "fps": original_fps,
        "duration": duration,
        "video_backend": "opencv",
        "frames_indices": valid_frame_indices,
        "do_sample_frames": valid_num_frames == total_frames_num,
    }
    return frames, metadata


def _preload_mm_data(dataset_name, dataset_split, num_seqs, seed,
                     num_video_frames=32):
    """Pre-download and load multimodal samples into memory.

    Returns list of dicts with keys:
      - prompt: str
      - images: list[PIL.Image] or None
      - video_frames: np.ndarray (T,H,W,3) or None
      - video_metadata: dict or None
      - audio: np.ndarray or None
      - audio_sampling_rate: int or None
    """
    from datasets import load_dataset
    use_streaming = "MMVU" not in dataset_name
    data = load_dataset(dataset_name, split=dataset_split,
                        streaming=use_streaming)
    if "librispeech_asr" in dataset_name:
        from datasets import Audio
        data = data.cast_column("audio", Audio(decode=False))
    data = data.shuffle(seed=seed)

    results = []
    if "VisionArena" in dataset_name:
        pbar = tqdm(data, total=num_seqs, desc="Loading images")
        for item in pbar:
            if len(results) >= num_seqs:
                break
            try:
                prompt = item["conversation"][0][0]["content"]
                if "base64" in prompt or len(prompt) > 4096:
                    continue
                img = item["images"][0]
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(BytesIO(img["bytes"]))
                if not isinstance(img, Image.Image):
                    continue
                img = img.convert("RGB")
                w, h = img.size
                if w * h > 2048 * 2048:
                    continue
            except Exception:
                continue
            results.append({
                "prompt": prompt,
                "images": [img],
                "video_frames": None,
                "video_metadata": None,
                "audio": None,
                "audio_sampling_rate": None,
            })
            pbar.update(0)
        pbar.close()
    elif "MMVU" in dataset_name:
        from huggingface_hub import snapshot_download
        import glob as _glob
        import os as _os
        local_root = snapshot_download(dataset_name, repo_type="dataset")
        n_clips = len(_glob.glob(
            _os.path.join(local_root, "**", "*.mp4"), recursive=True))
        if n_clips == 0:
            offline = _os.environ.get("HF_HUB_OFFLINE", "")
            reason = (
                f"HF_HUB_OFFLINE={offline} prevents clip downloads"
                if offline not in ("", "0")
                else "the snapshot contains no .mp4 files"
            )
            raise SystemExit(
                f"{dataset_name}: no video files under {local_root} ({reason})"
            )
        remote_root = (
            f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        )
        pbar = tqdm(data, total=num_seqs, desc="Loading videos")
        skipped = 0
        for item in pbar:
            if len(results) >= num_seqs:
                break
            try:
                prompt = item["question"] + " " + " ".join(
                    f"{k}.{v}" for k, v in item["choices"].items())
                video_path = item["video"].replace(remote_root, local_root)
                frames, metadata = _load_video_opencv(
                    video_path, num_frames=num_video_frames)
            except Exception:
                skipped += 1
                continue
            results.append({
                "prompt": prompt,
                "images": None,
                "video_frames": frames,
                "video_metadata": metadata,
                "audio": None,
                "audio_sampling_rate": None,
            })
            pbar.update(0)
        pbar.close()
        if skipped:
            print(
                f"  NOTE: skipped {skipped} unreadable MMVU video(s); "
                f"loaded {len(results)}/{num_seqs}",
                flush=True,
            )
        if not results:
            raise SystemExit(
                f"{dataset_name}: no readable clips among {skipped} attempted "
                f"({n_clips} .mp4 files under {local_root})"
            )
    elif "librispeech_asr" in dataset_name:
        pbar = tqdm(data, total=num_seqs, desc="Loading audio")
        for item in pbar:
            if len(results) >= num_seqs:
                break
            try:
                samples, sampling_rate = _decode_audio_array(item["audio"])
                if samples.ndim != 1 or samples.size == 0:
                    continue
            except Exception:
                continue
            results.append({
                "prompt": "Transcribe this audio and answer in text.",
                "images": None,
                "video_frames": None,
                "video_metadata": None,
                "audio": samples,
                "audio_sampling_rate": sampling_rate,
            })
            pbar.update(0)
        pbar.close()
    return results


def _filter_and_prepare(mm_data, processor, max_input_tokens):
    """Filter items by token count and pre-compute chat text in one pass."""
    prepared = []
    for item in tqdm(mm_data, desc="Filtering & preparing prompts"):
        try:
            messages = [{"role": "user", "content": []}]
            images_for_proc = None
            videos_for_proc = None
            audios_for_proc = None
            video_metadata = None
            do_sample_frames = None
            if item["audio"] is not None:
                messages[0]["content"].append(
                    {"type": "audio", "audio": item["audio"]})
                audios_for_proc = [item["audio"]]
            if item["images"] is not None:
                for img in item["images"]:
                    messages[0]["content"].append(
                        {"type": "image", "image": img})
                images_for_proc = item["images"]
            if item["video_frames"] is not None:
                # Count tokens the way both engines will actually process the
                # clip: frames + metadata. Counting a metadata-less PIL frame
                # list instead understates Qwen3-VL video by ~7x, because the
                # HF video processor then re-samples 32 frames down to 4.
                meta = item["video_metadata"] or {}
                messages[0]["content"].append(
                    {"type": "video", "video": item["video_frames"]})
                videos_for_proc = [item["video_frames"]]
                do_sample_frames = bool(meta.get("do_sample_frames", False))
                video_metadata = [
                    {k: v for k, v in meta.items() if k != "do_sample_frames"}
                ]
            messages[0]["content"].append(
                {"type": "text", "text": item["prompt"]})
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            processor_kwargs = dict(
                text=[text],
                images=images_for_proc,
                videos=videos_for_proc,
                return_tensors="pt",
                padding=True,
            )
            if video_metadata is not None:
                processor_kwargs["video_metadata"] = video_metadata
                processor_kwargs["do_sample_frames"] = do_sample_frames
            if audios_for_proc is not None:
                processor_kwargs["audio"] = audios_for_proc
            inputs = processor(**processor_kwargs)
            num_tokens = inputs["input_ids"].shape[1]
            if num_tokens <= max_input_tokens:
                item["chat_text"] = text
                prepared.append(item)
        except Exception:
            continue
    return prepared
'''

# ---------------------------------------------------------------------------
# Multi-scenario vLLM subprocess worker (VLM, multi-modal)
# ---------------------------------------------------------------------------
VLLM_VLM_WORKER = _MM_PRELOAD_FN + r'''
import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def _configure_parallel_safe_flashinfer():
    namespace = os.environ.get("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE")
    if not namespace:
        return
    try:
        import hashlib
        from flashinfer.comm import mnnvl
    except Exception:
        return
    if getattr(mnnvl.IpcSocket, "_fastkernels_namespaced", False):
        return

    original_init = mnnvl.IpcSocket.__init__
    namespace_bits = int.from_bytes(
        hashlib.blake2b(namespace.encode(), digest_size=8).digest(),
        "little",
    )

    def namespaced_init(self, rank, op_id, use_abstract=True):
        if isinstance(op_id, int):
            op_id = (op_id ^ namespace_bits) & ((1 << 64) - 1)
        original_init(self, rank, op_id, use_abstract)

    mnnvl.IpcSocket.__init__ = namespaced_init
    mnnvl.IpcSocket._fastkernels_namespaced = True

_configure_parallel_safe_flashinfer()


def _set_cfg_attr(cfg, name, value):
    try:
        setattr(cfg, name, value)
        return
    except Exception:
        pass
    try:
        cfg.__dict__[name] = value
    except Exception:
        try:
            object.__setattr__(cfg, name, value)
        except Exception:
            pass


def _fastkernels_limit_layers(hf_config):
    # Gemma-4: transformers 5.x hides per-layer attrs. Enable global reads
    # and stash the full-attention head dim so vLLM does not build every
    # layer at sliding-window 256 (checkpoint weights for full attn are 512).
    # global_head_dim is consumed in Gemma4TextConfig.__post_init__ and is
    # not kept as an attribute, so vLLM's getattr(..., "global_head_dim",
    # config.head_dim) would otherwise fall back to 256.
    candidates = [hf_config, getattr(hf_config, "text_config", None)]
    try:
        candidates.append(hf_config.get_text_config())
    except Exception:
        pass
    max_head_dim = 0
    for cfg in candidates:
        if cfg is None:
            continue
        _set_cfg_attr(cfg, "allow_global_per_layer_attribute_access", True)
        try:
            for layer_cfg in cfg.per_layer_config:
                hd = getattr(layer_cfg, "head_dim", 0) or 0
                if hd > max_head_dim:
                    max_head_dim = int(hd)
        except Exception:
            pass
    if max_head_dim:
        for cfg in candidates:
            if cfg is None:
                continue
            _set_cfg_attr(cfg, "global_head_dim", max_head_dim)
    # --max-layers: limit only the language-model decoder stack to the first N
    # layers (get_text_config() returns the LM sub-config for multimodal
    # models); vision / audio encoders and embeddings are left intact. N is
    # read from the env so this module-level function pickles cleanly into
    # vLLM's spawned EngineCore (a nested closure would not).
    n = os.environ.get("FASTKERNELS_MAX_LAYERS")
    if n:
        tc = hf_config.get_text_config()
        if getattr(tc, "num_hidden_layers", None):
            tc.num_hidden_layers = min(tc.num_hidden_layers, int(n))
    return hf_config


def _patch_gemma4_head_size():
    # convert() runs hasattr() on per-layer attrs. AmbiguousGlobalPerLayerAttributeError
    # subclasses RuntimeError, so hasattr does not swallow it. Also keep
    # get_head_size / get_total_num_kv_heads on the max per-layer values.
    try:
        from transformers.integrations.heterogeneity.configuration_utils import (
            AmbiguousGlobalPerLayerAttributeError,
        )
    except Exception:
        AmbiguousGlobalPerLayerAttributeError = ()
    try:
        from vllm.config import utils as vllm_config_utils
        orig_getattr_iter = vllm_config_utils.getattr_iter
        if not getattr(orig_getattr_iter, "_fk_patched", False):
            def getattr_iter(object, names, default=None, default_factory=None, warn=False):
                for name in names:
                    try:
                        has = hasattr(object, name)
                    except AmbiguousGlobalPerLayerAttributeError:
                        has = True
                    if not has:
                        continue
                    try:
                        return getattr(object, name)
                    except AmbiguousGlobalPerLayerAttributeError:
                        getter = getattr(
                            object, "_getattr_without_heterogeneous_validation", None
                        )
                        if getter is not None:
                            try:
                                return getter(name)
                            except Exception:
                                pass
                return default_factory() if default_factory is not None else default
            getattr_iter._fk_patched = True
            vllm_config_utils.getattr_iter = getattr_iter
            try:
                import vllm.transformers_utils.model_arch_config_convertor as conv
                conv.getattr_iter = getattr_iter
            except Exception:
                pass
    except Exception:
        pass
    try:
        from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
        from vllm.model_executor.models.utils import extract_layer_index
        orig_init = Gemma4DecoderLayer.__init__
        if not getattr(orig_init, "_fk_patched", False):
            def _gemma4_layer_init(
                self, config, cache_config=None, quant_config=None, prefix="",
            ):
                # Full-attn layers are 512-d / 2 KV heads; sliding are 256-d / 8.
                # Enabling global per-layer reads makes every layer see n_kv=8.
                try:
                    layer_idx = extract_layer_index(prefix)
                    class _LayerCfg:
                        def __init__(self, cfg, idx):
                            object.__setattr__(self, "_cfg", cfg)
                            object.__setattr__(self, "_idx", idx)
                        def __getattr__(self, name):
                            if name == "num_key_value_heads":
                                return self._cfg.per_layer_config[
                                    self._idx
                                ].num_key_value_heads
                            return getattr(self._cfg, name)
                    config = _LayerCfg(config, layer_idx)
                except Exception:
                    pass
                orig_init(
                    self, config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
            _gemma4_layer_init._fk_patched = True
            Gemma4DecoderLayer.__init__ = _gemma4_layer_init
    except Exception:
        pass
    try:
        from vllm.transformers_utils.model_arch_config_convertor import (
            Gemma4ModelArchConfigConvertor,
        )
    except Exception:
        return
    if getattr(Gemma4ModelArchConfigConvertor.get_head_size, "_fk_patched", False):
        return

    def _max_per_layer(tc, names, default):
        values = []
        try:
            for layer_cfg in tc.per_layer_config:
                for name in names:
                    val = getattr(layer_cfg, name, 0) or 0
                    if val:
                        values.append(int(val))
        except Exception:
            pass
        getter = getattr(tc, "_getattr_without_heterogeneous_validation", None)
        if getter is not None:
            for name in names:
                try:
                    val = getter(name)
                    if val:
                        values.append(int(val))
                except Exception:
                    pass
        return max(values) if values else default

    def get_head_size(self):
        return _max_per_layer(
            self.hf_text_config, ("head_dim", "global_head_dim"), 256
        )

    def get_total_num_kv_heads(self):
        return _max_per_layer(
            self.hf_text_config,
            ("num_key_value_heads", "num_global_key_value_heads"),
            1,
        )

    get_head_size._fk_patched = True
    Gemma4ModelArchConfigConvertor.get_head_size = get_head_size
    Gemma4ModelArchConfigConvertor.get_total_num_kv_heads = get_total_num_kv_heads


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    _patch_gemma4_head_size()

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    llm_kwargs = dict(
        model=model_name,
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
        # The per-scenario warmup replays the identical mm prompts right before
        # timing; with the cache on, the timed generate() would hit cached
        # pixel_values and skip the CPU preprocessing that the fastkernels
        # engine re-runs inside its own timed region.
        # vLLM 0.26 removed ``disable_mm_preprocessor_cache``; the equivalent
        # is sizing the multi-modal processor cache to 0 GiB, which
        # MultiModalConfig documents as disabling it completely.
        mm_processor_cache_gb=0,
        trust_remote_code=True,
    )
    if os.environ.get("FASTKERNELS_VLLM_LOG_STATS") == "1":
        # Diagnostic only: makes vLLM log "Running: N reqs / Waiting: M reqs /
        # GPU KV cache usage: X%" so its occupancy curve can be compared against
        # fastkernels' decode_batch histogram. Adds a little overhead to vLLM's
        # loop, so don't read speedups off a run with this enabled.
        llm_kwargs["disable_log_stats"] = False
        os.environ.setdefault("VLLM_LOG_STATS_INTERVAL", "1.0")
    if cfg.get("trust_remote_code"):
        llm_kwargs["trust_remote_code"] = True
    if cfg["tp"] > 1:
        # See the LLM worker: keep multi-GPU off vLLM's ray executor.
        llm_kwargs["distributed_executor_backend"] = "mp"
    if cfg.get("load_format"):
        llm_kwargs["load_format"] = cfg["load_format"]
    if cfg.get("kv_cache_dtype"):
        llm_kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    if cfg.get("limit_mm_per_prompt"):
        llm_kwargs["limit_mm_per_prompt"] = cfg["limit_mm_per_prompt"]
    if cfg.get("max_num_seqs") is not None:
        llm_kwargs["max_num_seqs"] = cfg["max_num_seqs"]
    llm_kwargs["hf_overrides"] = _fastkernels_limit_layers
    llm = LLM(**llm_kwargs)

    # Warmup -- ignore_eos so all 16 decode steps run (parity with the engines).
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
    )

    scenarios = cfg["scenarios"]
    all_results = []
    temperature = cfg.get("temperature", 0.0)

    for scenario in scenarios:
        modality = scenario.get("modality", "text")

        if modality == "text":
            prompt_token_ids = scenario["prompt_token_ids"]
            output_lens = scenario["output_lens"]
            sp_list = [
                SamplingParams(temperature=temperature,
                               ignore_eos=True, max_tokens=ol)
                for ol in output_lens
            ]
            vllm_prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
            # Prefill warmup at this scenario's real shapes (see LLM worker).
            llm.generate(
                vllm_prompts,
                SamplingParams(temperature=temperature, ignore_eos=True,
                               max_tokens=1),
                use_tqdm=False,
            )
            start = time.perf_counter()
            outputs = llm.generate(vllm_prompts, sp_list)
            elapsed = time.perf_counter() - start
        else:
            mm_data = _preload_mm_data(
                scenario["dataset"], scenario["dataset_split"],
                scenario["num_seqs"], cfg["seed"],
            )
            max_input_tokens = cfg["max_model_len"] - scenario["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_input_tokens)
            print(f"  vLLM: {len(mm_data)} items after token-count filter "
                  f"(max_input={max_input_tokens})")
            sp_list = []
            vllm_prompts = []
            for item in mm_data:
                sp_list.append(
                    SamplingParams(temperature=temperature,
                                   ignore_eos=True,
                                   max_tokens=scenario["output_len"]))
                mm_dict = {}
                if item["images"] is not None:
                    mm_dict["image"] = item["images"]
                if item["video_frames"] is not None:
                    mm_dict["video"] = [
                        (item["video_frames"], item["video_metadata"])
                    ]
                if item["audio"] is not None:
                    mm_dict["audio"] = (
                        item["audio"], item["audio_sampling_rate"]
                    )
                vllm_prompts.append(dict(
                    prompt=item["chat_text"],
                    multi_modal_data=mm_dict,
                ))

            # Prefill warmup at this scenario's real shapes. Covers the vision
            # encoder too, which is where the timed-region JIT showed up for
            # the VLMs (rotary_kernel, FlashAttentionForwardSm100,
            # _bilinear_pos_embed_kernel are all resolution-dependent and so
            # are never reached by a text-only engine warmup).
            llm.generate(
                vllm_prompts,
                SamplingParams(temperature=temperature, ignore_eos=True,
                               max_tokens=1),
                use_tqdm=False,
            )
            start = time.perf_counter()
            outputs = llm.generate(vllm_prompts, sp_list, use_tqdm=True)
            elapsed = time.perf_counter() - start

        total_prompt_tokens = sum(
            len(o.prompt_token_ids) if o.prompt_token_ids else 0
            for o in outputs
        )
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs if c)
            for o in outputs
        )
        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {"text": o.outputs[0].text,
                 "token_ids": list(o.outputs[0].token_ids)}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        modality = ls.get("modality", "text")

        if modality == "text":
            prompts = [dict(prompt_token_ids=p) for p in ls["prompt_token_ids"]]
            output_lens = ls.get("output_lens")
            if output_lens is None:
                sp = SamplingParams(temperature=0.0,
                                    ignore_eos=True, max_tokens=ls["output_len"])
            else:
                sp = [
                    SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
                    for ol in output_lens
                ]
            run_fn = lambda: llm.generate(prompts, sp, use_tqdm=False)
        else:
            mm_data = _preload_mm_data(
                ls["dataset"], ls["dataset_split"], 1, cfg["seed"],
            )
            max_lat_tokens = cfg["max_model_len"] - ls["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_lat_tokens)
            item = mm_data[0]
            sp = SamplingParams(temperature=0.0, ignore_eos=True,
                                max_tokens=ls["output_len"])
            mm_dict = {}
            if item["images"] is not None:
                mm_dict["image"] = item["images"]
            if item["video_frames"] is not None:
                mm_dict["video"] = [
                    (item["video_frames"], item["video_metadata"])
                ]
            if item["audio"] is not None:
                mm_dict["audio"] = (
                    item["audio"], item["audio_sampling_rate"]
                )
            lat_prompt = dict(prompt=item["chat_text"],
                              multi_modal_data=mm_dict)
            run_fn = lambda: llm.generate([lat_prompt], sp, use_tqdm=False)

        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            run_fn()
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            run_fn()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del llm

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)
        f.flush()
        os.fsync(f.fileno())

    os._exit(0)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Multi-scenario fastkernels subprocess worker (VLM, multi-modal)
# ---------------------------------------------------------------------------
FASTKERNELS_VLM_WORKER = _MM_PRELOAD_FN + r'''
import json, os, sys, time
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg["model"], trust_remote_code=True)

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    if "max_layers" in cfg:
        engine_kwargs["max_layers"] = cfg["max_layers"]
    engine = LlamaEngine(**engine_kwargs)

    # Warmup -- 16-token prompt, matching the LLM workers. ignore_eos so the 16
    # decode steps run even if token 0 greedily decodes to EOS (parity fix).
    engine.generate([[0] * 16],
                    SamplingParams(temperature=0.0, max_tokens=16,
                                   ignore_eos=True))

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    temperature = cfg.get("temperature", 0.0)
    top_p = cfg.get("top_p", 1.0)

    for scenario in scenarios:
        modality = scenario.get("modality", "text")

        if modality == "text":
            prompts = scenario["prompt_token_ids"]
            output_lens = scenario["output_lens"]
            sp_list = [
                SamplingParams(temperature=temperature, top_p=top_p,
                               max_tokens=ol, ignore_eos=True)
                for ol in output_lens
            ]
            # Prefill warmup at this scenario's real shapes (see LLM worker).
            engine.generate(
                prompts,
                SamplingParams(temperature=temperature, top_p=top_p,
                               max_tokens=1, ignore_eos=True),
                use_tqdm=False,
            )
            engine.block_manager.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(prompts, sp_list, use_tqdm=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_input_tokens = sum(len(p) for p in prompts)
        else:
            mm_data = _preload_mm_data(
                scenario["dataset"], scenario["dataset_split"],
                scenario["num_seqs"], cfg["seed"],
            )
            max_input_tokens = cfg["max_model_len"] - scenario["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_input_tokens)
            print(f"  fastkernels: {len(mm_data)} items after token-count filter "
                  f"(max_input={max_input_tokens})")

            prompts = [item["prompt"] for item in mm_data]
            batch_images = []
            batch_videos = []
            batch_audios = []
            for item in mm_data:
                if item["images"] is not None:
                    batch_images.append(item["images"])
                else:
                    batch_images.append(None)
                if item["video_frames"] is not None:
                    # Same (frames, metadata) pair the vLLM worker passes as
                    # multi_modal_data["video"]. Handing fastkernels bare PIL
                    # frames instead would drop the metadata the HF video
                    # processor needs, and for Qwen3-VL that silently reduces
                    # 32 frames to 4 (~490 video tokens vs vLLM's ~3520).
                    batch_videos.append([
                        (item["video_frames"], item["video_metadata"])
                    ])
                else:
                    batch_videos.append(None)
                if item["audio"] is not None:
                    batch_audios.append([item["audio"]])
                else:
                    batch_audios.append(None)

            sp_list = [
                SamplingParams(temperature=temperature, top_p=top_p,
                               max_tokens=scenario["output_len"],
                               ignore_eos=True)
            ] * len(mm_data)

            total_input_tokens = 0
            # Prefill warmup at this scenario's real shapes, vision encoder
            # included -- _warmup_vision_encoder() only covers the synthetic
            # max-resolution item, not this dataset's actual resolutions.
            engine.generate(
                prompts,
                SamplingParams(temperature=temperature, top_p=top_p,
                               max_tokens=1, ignore_eos=True),
                images=batch_images,
                videos=batch_videos,
                audio_features=batch_audios,
                use_tqdm=False,
            )
            engine.block_manager.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(prompts, sp_list,
                                      images=batch_images,
                                      videos=batch_videos,
                                      audio_features=batch_audios,
                                      use_tqdm=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        total_output_tokens = sum(len(o.token_ids) for o in outputs)

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_input_tokens if modality == "text" else 0,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {"generated_text": o.generated_text,
                 "token_ids": o.token_ids}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        modality = ls.get("modality", "text")

        if modality == "text":
            prompts = ls["prompt_token_ids"]
            output_lens = ls.get("output_lens")
            if output_lens is None:
                sp = SamplingParams(temperature=0.0, ignore_eos=True,
                                    max_tokens=ls["output_len"])
            else:
                sp = [
                    SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
                    for ol in output_lens
                ]
            def run_fn():
                engine.block_manager.reset()
                torch.cuda.synchronize()
                engine.generate(prompts, sp)
                torch.cuda.synchronize()
        else:
            mm_data = _preload_mm_data(
                ls["dataset"], ls["dataset_split"], 1, cfg["seed"],
            )
            max_lat_tokens = cfg["max_model_len"] - ls["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_lat_tokens)
            item = mm_data[0]
            sp = SamplingParams(temperature=0.0, ignore_eos=True,
                                max_tokens=ls["output_len"])
            lat_images = None
            lat_videos = None
            lat_audios = None
            if item["images"] is not None:
                lat_images = [item["images"]]
            if item["video_frames"] is not None:
                lat_videos = [[
                    (item["video_frames"], item["video_metadata"])
                ]]
            if item["audio"] is not None:
                lat_audios = [[item["audio"]]]
            def run_fn(p=item["prompt"], imgs=lat_images, vids=lat_videos,
                       auds=lat_audios):
                engine.block_manager.reset()
                torch.cuda.synchronize()
                engine.generate(
                    [p], sp,
                    images=imgs,
                    videos=vids,
                    audio_features=auds,
                )
                torch.cuda.synchronize()

        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            run_fn()
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            run_fn()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)
        f.flush()
        os.fsync(f.fileno())

    # Kill the engine's spawned TP rank workers before the hard exit. os._exit(0)
    # deliberately skips atexit (multiprocessing's no-timeout child join can hang
    # shutdown), but that also skips the engine's atexit cleanup -- so without
    # this the rank workers orphan and keep holding their GPUs, OOM-ing the next
    # scenario the scheduler launches on those GPUs.
    try:
        engine._cleanup()
    except Exception:
        pass

    os._exit(0)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Multi-scenario vLLM subprocess worker (Whisper, audio)
# ---------------------------------------------------------------------------
VLLM_WHISPER_WORKER = r'''
import json, os, sys, time
import numpy as np
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def _decode_audio_array(audio):
    from io import BytesIO
    if isinstance(audio, dict) and audio.get("array") is not None:
        return (
            np.asarray(audio["array"], dtype=np.float32),
            int(audio["sampling_rate"]),
        )
    import av
    source = None
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            source = BytesIO(audio["bytes"])
        elif audio.get("path") is not None:
            source = audio["path"]
    if source is None:
        raise ValueError("Unsupported audio sample format")
    chunks = []
    sampling_rate = None
    with av.open(source) as container:
        for frame in container.decode(audio=0):
            chunks.append(frame.to_ndarray())
            sampling_rate = frame.sample_rate
    if not chunks or sampling_rate is None:
        raise ValueError("Audio sample has no decodable frames")
    samples = np.concatenate(chunks, axis=-1)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        samples = samples.astype(np.float32) / max(abs(info.min), info.max)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=0)
    return samples, int(sampling_rate)

def _configure_parallel_safe_flashinfer():
    namespace = os.environ.get("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE")
    if not namespace:
        return
    try:
        import hashlib
        from flashinfer.comm import mnnvl
    except Exception:
        return
    if getattr(mnnvl.IpcSocket, "_fastkernels_namespaced", False):
        return

    original_init = mnnvl.IpcSocket.__init__
    namespace_bits = int.from_bytes(
        hashlib.blake2b(namespace.encode(), digest_size=8).digest(),
        "little",
    )

    def namespaced_init(self, rank, op_id, use_abstract=True):
        if isinstance(op_id, int):
            op_id = (op_id ^ namespace_bits) & ((1 << 64) - 1)
        original_init(self, rank, op_id, use_abstract)

    mnnvl.IpcSocket.__init__ = namespaced_init
    mnnvl.IpcSocket._fastkernels_namespaced = True

_configure_parallel_safe_flashinfer()

def _load_librispeech(dataset_name, dataset_split, num_seqs, seed):
    """Load audio samples from LibriSpeech and return as list of numpy arrays."""
    from datasets import Audio, load_dataset
    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=seed)
    samples = []
    seen = failed = 0
    first_error = None
    for item in ds:
        seen += 1
        try:
            arr, sr = _decode_audio_array(item["audio"])
        except Exception:
            failed += 1
            if first_error is None:
                import traceback
                first_error = traceback.format_exc()
            continue
        samples.append({"audio": arr, "sampling_rate": sr, "text": item["text"]})
        if len(samples) >= num_seqs:
            break
    if not samples:
        print(
            f"  [librispeech diag] seen={seen} fail={failed} "
            f"first_err=\n{first_error}",
            flush=True,
        )
    return samples

def main():
    from vllm import LLM, SamplingParams

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    llm_kwargs = dict(
        model=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
        # See the VLM worker: the per-scenario warmup replays the identical
        # audio prompts before timing, so leave the mm preprocessor cache off
        # to keep the timed region paying for audio preprocessing.
        # vLLM 0.26 removed ``disable_mm_preprocessor_cache``; 0 GiB disables.
        mm_processor_cache_gb=0,
    )
    if cfg.get("trust_remote_code"):
        llm_kwargs["trust_remote_code"] = True
    if cfg["tp"] > 1:
        # See the LLM worker: keep multi-GPU off vLLM's ray executor.
        llm_kwargs["distributed_executor_backend"] = "mp"
    if cfg.get("load_format"):
        llm_kwargs["load_format"] = cfg["load_format"]
    if cfg.get("kv_cache_dtype"):
        llm_kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    llm = LLM(**llm_kwargs)

    from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt

    # Warmup
    dummy_audio = np.zeros(16000, dtype=np.float32)
    warmup_prompt = ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt="",
            multi_modal_data={"audio": (dummy_audio, 16000)},
        ),
        decoder_prompt=TextPrompt(
            prompt="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        ),
    )
    llm.generate(
        [warmup_prompt],
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
    )

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        num_seqs = scenario["num_seqs"]
        output_len = scenario["output_len"]

        audio_samples = _load_librispeech(
            scenario["dataset"], scenario["dataset_split"],
            num_seqs, cfg["seed"],
        )
        print(f"  Loaded {len(audio_samples)} audio samples from "
              f"{scenario['dataset']} ({scenario['dataset_split']})")

        prompts = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            prompt = ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt="",
                    multi_modal_data={"audio": (audio, sr)},
                ),
                decoder_prompt=TextPrompt(
                    prompt="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                ),
            )
            prompts.append(prompt)

        sp = SamplingParams(
            temperature=0.0, ignore_eos=True, max_tokens=output_len,
        )

        # Prefill warmup at this scenario's real shapes. Covers the audio
        # encoder, whose kernels depend on mel length and so are not reached
        # by the fixed-shape engine warmup above.
        llm.generate(
            prompts,
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1),
            use_tqdm=False,
        )
        start = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        elapsed = time.perf_counter() - start

        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs if c)
            for o in outputs
        )
        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
            "num_seqs": len(audio_samples),
            "total_audio_duration_s": total_audio_s,
            "outputs": [
                {"text": o.outputs[0].text,
                 "token_ids": list(o.outputs[0].token_ids)}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        output_len = ls["output_len"]
        batch_size = ls.get("batch_size", 1)
        audio_samples = _load_librispeech(
            ls["dataset"], ls["dataset_split"], batch_size, cfg["seed"] + 200,
        )
        prompts = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            prompts.append(ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt="",
                    multi_modal_data={"audio": (audio, sr)},
                ),
                decoder_prompt=TextPrompt(
                    prompt="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                ),
            ))
        sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            llm.generate(prompts, sp, use_tqdm=False)
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": batch_size,
            "audio_duration_s": round(total_audio_s, 2),
            "output_len": output_len,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del llm
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Multi-scenario fastkernels subprocess worker (Whisper, audio)
# ---------------------------------------------------------------------------
FASTKERNELS_WHISPER_WORKER = r'''
import json, sys, time
import numpy as np

def _decode_audio_array(audio):
    from io import BytesIO
    if isinstance(audio, dict) and audio.get("array") is not None:
        return (
            np.asarray(audio["array"], dtype=np.float32),
            int(audio["sampling_rate"]),
        )
    import av
    source = None
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            source = BytesIO(audio["bytes"])
        elif audio.get("path") is not None:
            source = audio["path"]
    if source is None:
        raise ValueError("Unsupported audio sample format")
    chunks = []
    sampling_rate = None
    with av.open(source) as container:
        for frame in container.decode(audio=0):
            chunks.append(frame.to_ndarray())
            sampling_rate = frame.sample_rate
    if not chunks or sampling_rate is None:
        raise ValueError("Audio sample has no decodable frames")
    samples = np.concatenate(chunks, axis=-1)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        samples = samples.astype(np.float32) / max(abs(info.min), info.max)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=0)
    return samples, int(sampling_rate)

def _load_librispeech(dataset_name, dataset_split, num_seqs, seed):
    """Load audio samples from LibriSpeech and return as list of numpy arrays."""
    from datasets import Audio, load_dataset
    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=seed)
    samples = []
    seen = failed = 0
    first_error = None
    for item in ds:
        seen += 1
        try:
            arr, sr = _decode_audio_array(item["audio"])
        except Exception:
            failed += 1
            if first_error is None:
                import traceback
                first_error = traceback.format_exc()
            continue
        samples.append({"audio": arr, "sampling_rate": sr, "text": item["text"]})
        if len(samples) >= num_seqs:
            break
    if not samples:
        print(
            f"  [librispeech diag] seen={seen} fail={failed} "
            f"first_err=\n{first_error}",
            flush=True,
        )
    return samples

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", True),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    if "max_layers" in cfg:
        engine_kwargs["max_layers"] = cfg["max_layers"]
    engine = LlamaEngine(**engine_kwargs)

    import torch
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(cfg["model"])

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        num_seqs = scenario["num_seqs"]
        output_len = scenario["output_len"]

        audio_samples = _load_librispeech(
            scenario["dataset"], scenario["dataset_split"],
            num_seqs, cfg["seed"],
        )
        print(f"  Loaded {len(audio_samples)} audio samples from "
              f"{scenario['dataset']} ({scenario['dataset_split']})")

        audio_features_list = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            audio_features_list.append(inputs.input_features[0])

        decoder_prompt = processor.tokenizer.encode(
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            add_special_tokens=False,
        )
        decoder_prompts = [decoder_prompt] * len(audio_samples)

        sp = SamplingParams(
            temperature=0.0, ignore_eos=True, max_tokens=output_len,
        )

        # Prefill warmup at this scenario's real shapes (see vLLM worker).
        engine.generate(
            decoder_prompts,
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1),
            audio_features=audio_features_list, use_tqdm=False,
        )

        engine.block_manager.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(
            decoder_prompts, sp,
            audio_features=audio_features_list, use_tqdm=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_output_tokens = sum(len(o.token_ids) for o in outputs)
        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
            "num_seqs": len(audio_samples),
            "total_audio_duration_s": total_audio_s,
            "outputs": [
                {"generated_text": o.generated_text,
                 "token_ids": o.token_ids}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        output_len = ls["output_len"]
        batch_size = ls.get("batch_size", 1)
        audio_samples = _load_librispeech(
            ls["dataset"], ls["dataset_split"], batch_size, cfg["seed"] + 200,
        )
        audio_feats = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            inp = processor(audio, sampling_rate=sr, return_tensors="pt")
            audio_feats.append(inp.input_features[0])
        decoder_prompt = processor.tokenizer.encode(
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            add_special_tokens=False,
        )
        decoder_prompts = [decoder_prompt] * batch_size

        sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            engine.generate(
                decoder_prompts, sp, audio_features=audio_feats,
            )
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(
                decoder_prompts, sp, audio_features=audio_feats,
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": batch_size,
            "audio_duration_s": round(total_audio_s, 2),
            "output_len": output_len,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------
def compute_alignment(
    a_outputs: list[dict],
    b_outputs: list[dict],
) -> dict:
    """Compare per-request token_ids. Returns alignment statistics.

    ``avg_matching_tokens_per_request`` is the **matching prefix**: it stops at
    the first divergence, so a correct token *after* a wrong one never counts.
    That is the only reading that means anything for greedy decode -- once two
    near-tied logits pick differently the sequences fork permanently, and
    position-wise agreement past that point credits coincidental re-alignment.
    On Mamba-Codestral's 1000-request mixed run the position-wise count read
    222.9 tokens against a true prefix of 204.8, an 8% overstatement under a
    name that reads like a prefix.

    The position-wise count is still reported as
    ``avg_position_matches_per_request``: it separates "diverged and drifted"
    from "diverged and resynchronised", which is worth seeing.

    Same keys and semantics as ``bench_microsoft_bitnet.compute_alignment`` and
    ``comparison.alignment_from_token_ids``, so one aggregate query spans every
    generative harness. bench_jamba, bench_fla, and bench_sglang import this
    function rather than re-deriving it -- three copies are how the two
    definitions drifted apart in the first place.
    """
    total_seqs = len(a_outputs)
    exact_matches = 0
    total_matching_tokens = 0
    total_position_matches = 0
    total_output_tokens = 0

    for a, b in zip(a_outputs, b_outputs):
        a_ids = a["token_ids"]
        b_ids = b["token_ids"]
        total_output_tokens += max(len(a_ids), len(b_ids))

        prefix = 0
        for x, y in zip(a_ids, b_ids):
            if x != y:
                break
            prefix += 1
        total_matching_tokens += prefix
        total_position_matches += sum(
            1 for x, y in zip(a_ids, b_ids) if x == y
        )

        if a_ids == b_ids:
            exact_matches += 1

    avg_matching = total_matching_tokens / total_seqs if total_seqs else 0
    avg_position = total_position_matches / total_seqs if total_seqs else 0
    avg_output_len = total_output_tokens / total_seqs if total_seqs else 0

    return {
        "exact_matches": exact_matches,
        "total_seqs": total_seqs,
        "total_matching_tokens": total_matching_tokens,
        "total_position_matches": total_position_matches,
        "total_output_tokens": total_output_tokens,
        "avg_matching_tokens_per_request": avg_matching,
        "avg_position_matches_per_request": avg_position,
        "avg_output_len": avg_output_len,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Throughput & alignment benchmark: fastkernels baseline vs vLLM",
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Run only the first MAX_LAYERS transformer decoder layers of the "
             "model (both vLLM and fastkernels). Only those layers are built "
             "and their weights loaded; the embedding, final norm, and LM head "
             "are unaffected. Not supported for Whisper (encoder-decoder).",
    )
    parser.add_argument(
        "--kv-cache-dtype", default=None,
        help="Paged-KV cache dtype passed to BOTH engines (e.g. fp8_e4m3). "
             "Omit to leave each side on its own default ('auto'). This is "
             "independent of the weight quantization: nvidia/GLM-5.2-NVFP4 has "
             "NVFP4 weights and an fp8 KV cache.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the reference vLLM worker when required.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic alignment)",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument(
        "--vllm-python",
        type=str,
        default=None,
        help="Python interpreter for the vLLM reference worker. Defaults to "
        "the current interpreter.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="GPU memory fraction applied identically to both engines.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse cached phase outputs (vllm_raw.json / fastkernels_raw.json) "
             "under the output dir when their config fingerprint matches, "
             "instead of rerunning that phase. Lets an interrupted run continue "
             "without recomputing the completed (e.g. vLLM) side.",
    )
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip the throughput phase (run latency only)")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip the latency benchmark phase")
    parser.add_argument("--latency-iters", type=int, default=5,
                        help="Timed iterations per latency scenario (default: 5)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save per-scenario outputs, phase caches, and results "
             "JSON (default: ~/.fastkernels/validate/<run-id>)",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run id naming the output dir ~/.fastkernels/validate/<run-id>. "
             "Defaults to a timestamp+pid so concurrent runs do not overwrite "
             "each other. Ignored when --output-dir is provided.",
    )
    parser.add_argument(
        "--modality", type=str, default="all",
        choices=["all", "text", "image", "video", "audio"],
        help="Run only scenarios matching this modality (multimodal models only, default: all)",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run only the throughput scenario with this name (e.g. "
             "'mixed'). Default: run all scenarios for the model type.",
    )
    args = parser.parse_args()
    args.trust_remote_code = (
        args.trust_remote_code or _needs_trust_remote_code(args.model)
    )

    if args.num_seqs is None:
        args.num_seqs = 100 if _is_whisper_model(args.model) else 1000

    gpu = _detect_gpu_name()
    is_vlm = _is_vlm_model(args.model)
    is_qwen_omni = _is_qwen_omni_model(args.model)
    is_whisper = _is_whisper_model(args.model)
    engine_env = _apply_per_model_defaults(args.model, args)

    if args.max_layers is not None:
        if args.max_layers < 1:
            raise SystemExit("--max-layers must be >= 1")
        if is_whisper:
            print("  NOTE: --max-layers is ignored for Whisper "
                  "(encoder-decoder) models.")
            args.max_layers = None

    if args.output_dir is None:
        run_id = _make_run_id(args.run_id)
        args.output_dir = str(
            Path.home() / ".fastkernels" / "validate" / run_id
        )
    elif args.run_id is not None:
        print("  NOTE: --run-id is ignored because --output-dir was provided.")

    kb_nccl_port, kb_nccl_lock = _reserve_tcp_port(
        preferred=_parse_port_env("FASTKERNELS_NCCL_PORT"),
    )
    _HELD_PORT_LOCKS.append(kb_nccl_lock)
    os.environ["FASTKERNELS_NCCL_PORT"] = str(kb_nccl_port)

    vllm_port = None
    flashinfer_namespace = None
    previous_flashinfer_namespace_env = os.environ.get(
        "FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE",
    )
    previous_max_layers_env = os.environ.get("FASTKERNELS_MAX_LAYERS")
    if not args.skip_vllm:
        vllm_port, vllm_port_lock = _reserve_tcp_port(
            preferred=_parse_port_env("VLLM_PORT"),
        )
        _HELD_PORT_LOCKS.append(vllm_port_lock)
        os.environ["VLLM_PORT"] = str(vllm_port)
        if args.tp > 1:
            flashinfer_namespace = (
                os.environ.get("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE")
                or f"bench-vllm-{os.getpid()}-{vllm_port}"
            )
            os.environ["FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE"] = flashinfer_namespace
        if args.max_layers is not None:
            # Filter pruned-layer weights inside every spawned vLLM rank so its
            # per-model loaders don't KeyError on the full checkpoint's extra
            # ``layers.{i>=N}`` tensors (hf_overrides only shrinks the model).
            os.environ["FASTKERNELS_MAX_LAYERS"] = str(args.max_layers)
        # Align the throwaway CUDA-graph-profiling KV cache's block count so
        # vLLM's own default MLA decode backend can start (upstream bug; see
        # the sitecustomize body). Always on: it is a no-op for every model
        # whose page size already yields alignment == 1, and it never changes a
        # measured config -- so it does not need to be model-gated.
        os.environ["FASTKERNELS_ALIGN_PROFILING_KV_BLOCKS"] = "1"
        _install_bench_sitecustomize()

    if is_whisper:
        throughput_scenarios = WHISPER_SCENARIOS
        latency_scenarios = WHISPER_LATENCY_SCENARIOS
    elif is_qwen_omni:
        throughput_scenarios = QWEN_OMNI_SCENARIOS
        latency_scenarios = QWEN_OMNI_LATENCY_SCENARIOS
    elif is_vlm:
        throughput_scenarios = VLM_SCENARIOS
        latency_scenarios = VLM_LATENCY_SCENARIOS
    else:
        throughput_scenarios = SCENARIOS
        latency_scenarios = LATENCY_SCENARIOS

    if (is_vlm or is_qwen_omni) and not is_whisper and args.modality != "all":
        throughput_scenarios = [
            s for s in throughput_scenarios
            if s.get("modality", "text") == args.modality
        ]
        latency_scenarios = [
            s for s in latency_scenarios
            if s.get("modality", "text") == args.modality
        ]

    if args.scenario is not None:
        throughput_scenarios = [
            s for s in throughput_scenarios if s["name"] == args.scenario
        ]
        if not throughput_scenarios:
            raise SystemExit(
                f"--scenario={args.scenario!r} did not match any throughput "
                f"scenario for this model type."
            )

    # Pre-generate all scenario data
    scenario_data = []
    global_max_seq_len = 0
    tokenizer = None
    if not is_whisper:
        tokenizer = _load_tokenizer(args.model)
    # vLLM (>=0.24.0) rejects a max_model_len exceeding the model's derived
    # context window, and real long-context prompts can land a few tokens over
    # (e.g. LongBench rows whose prompt+decode slightly exceeds 128K). Trim the
    # tail of the raw prompt content of such rows (re-applying the chat template
    # so no special tokens are dropped) to keep every request valid.
    model_max_ctx = None if is_whisper else _get_model_max_context_len(args.model)

    def _fit_prompts_to_ctx(samples):
        """Prompt token ids for ``samples``, trimming the tail of the raw
        prompt *content* (never the chat template / special tokens) so each
        prompt plus its decode budget fits the model context window."""
        prompt_token_ids = []
        n_trunc = 0
        for s in samples:
            ids = list(s.prompt_token_ids)
            if model_max_ctx is not None and s.messages is not None:
                budget = model_max_ctx - s.output_len
                if budget >= 1 and len(ids) > budget:
                    ids = _fit_messages_to_context(
                        tokenizer, s.messages, budget)
                    n_trunc += 1
            prompt_token_ids.append(ids)
        if n_trunc:
            print(f"  NOTE: trimmed prompt content of {n_trunc} request(s) to "
                  f"fit {model_max_ctx}-token model context "
                  f"(chat template preserved)")
        return prompt_token_ids

    if not args.skip_throughput:
        for i, scenario in enumerate(throughput_scenarios):
            if is_whisper:
                output_len = scenario["output_len"]
                max_seq_len = output_len + 10  # decoder prompt + output
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                num_seqs = args.num_seqs
                if scenario.get("use_full_dataset"):
                    num_seqs = 999_999  # load all available samples
                scenario_data.append({
                    "name": scenario["name"],
                    "output_len": output_len,
                    "dataset": scenario["dataset"],
                    "dataset_split": scenario["dataset_split"],
                    "num_seqs": num_seqs,
                })
                continue

            modality = scenario.get("modality", "text") if (is_vlm or is_qwen_omni) else "text"
            if modality == "text":
                if scenario.get("dataset") is not None:
                    samples = load_real_prompt_workload(
                        scenario["name"],
                        tokenizer,
                        num_requests=args.num_seqs,
                        decode_cap=None,
                        dataset_name=scenario["dataset"],
                        seed=args.seed + i,
                    )
                    prompt_token_ids = _fit_prompts_to_ctx(samples)
                    output_lens = [s.output_len for s in samples]
                else:
                    raise ValueError(
                        f"text throughput scenario '{scenario['name']}' has no "
                        "dataset; all text workloads must use a real prompt "
                        "dataset (synthetic random-token prompts are not allowed)"
                    )
                max_seq_len = max(
                    len(p) + ol
                    for p, ol in zip(prompt_token_ids, output_lens)
                )
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                scenario_data.append({
                    "name": scenario["name"],
                    "modality": "text",
                    "prompt_token_ids": prompt_token_ids,
                    "output_lens": output_lens,
                })
            else:
                # Multimodal datasets are loaded inside the subprocess worker.
                # Large media inputs can produce many tokens; be generous.
                max_seq_len = 16384 + scenario["output_len"]
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                scenario_data.append({
                    "name": scenario["name"],
                    "modality": modality,
                    "output_len": scenario["output_len"],
                    "dataset": scenario["dataset"],
                    "dataset_split": scenario["dataset_split"],
                    "num_seqs": args.num_seqs,
                })

    # Pre-generate latency scenario data
    latency_data = []
    if not args.skip_latency:
        for j, ls in enumerate(latency_scenarios):
            if is_whisper:
                max_seq_len = ls["output_len"] + 10
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                latency_data.append({
                    "name": ls["name"],
                    "output_len": ls["output_len"],
                    "batch_size": ls["batch_size"],
                    "dataset": ls["dataset"],
                    "dataset_split": ls["dataset_split"],
                    "num_warmup": 3,
                    "num_iters": args.latency_iters,
                })
                continue

            modality = ls.get("modality", "text") if (is_vlm or is_qwen_omni) else "text"
            if modality == "text":
                bs = ls["batch_size"]
                samples = load_real_prompt_workload(
                    "mixed",
                    tokenizer,
                    num_requests=bs,
                    decode_cap=ls["output_len"],
                    dataset_name=ls.get("dataset") or None,
                    seed=args.seed + 100 + j,
                )
                prompt_token_ids = _fit_prompts_to_ctx(samples)
                output_lens = [s.output_len for s in samples]
                real_input_len = max((len(p) for p in prompt_token_ids), default=0)
                seq_len = max(
                    len(p) + ol
                    for p, ol in zip(prompt_token_ids, output_lens)
                )
                if seq_len > global_max_seq_len:
                    global_max_seq_len = seq_len
                latency_data.append({
                    "name": ls["name"],
                    "modality": "text",
                    "input_len": real_input_len,
                    "output_len": ls["output_len"],
                    "batch_size": bs,
                    "prompt_token_ids": prompt_token_ids,
                    "output_lens": output_lens,
                    "num_warmup": 3,
                    "num_iters": args.latency_iters,
                })
            else:
                max_seq_len = 16384 + ls["output_len"]
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                latency_data.append({
                    "name": ls["name"],
                    "modality": modality,
                    "output_len": ls["output_len"],
                    "batch_size": ls["batch_size"],
                    "dataset": ls["dataset"],
                    "dataset_split": ls["dataset_split"],
                    "num_warmup": 3,
                    "num_iters": args.latency_iters,
                })

    # Safety net: if any scenario path still produced a length beyond the
    # model's context window (e.g. the multimodal token estimate), cap it so
    # vLLM (>=0.24.0) does not reject the run.
    if model_max_ctx is not None and global_max_seq_len > model_max_ctx:
        print(
            f"  NOTE: capping max seq len {global_max_seq_len} -> "
            f"{model_max_ctx} (model context limit)"
        )
        global_max_seq_len = model_max_ctx

    print("=" * 70)
    print("  fastkernels Baseline vs vLLM -- Multi-Scenario Benchmark")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    model_type_str = (
        "Whisper" if is_whisper
        else ("Qwen-Omni" if is_qwen_omni else ("VLM" if is_vlm else "LLM"))
    )
    print(f"  Model type     : {model_type_str}")
    if (is_vlm or is_qwen_omni) and args.modality != "all":
        print(f"  Modality       : {args.modality}")
    print(f"  TP             : {args.tp}")
    if args.max_layers is not None:
        print(f"  Max layers     : {args.max_layers}")
    has_full = any(s.get("use_full_dataset") for s in throughput_scenarios) if is_whisper else False
    seqs_label = "full dataset" if has_full else str(args.num_seqs)
    print(f"  Seqs/scenario  : {seqs_label}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Enforce eager  : {args.enforce_eager}")
    print(f"  Seed           : {args.seed}")
    print(f"  Trust RC       : {args.trust_remote_code}")
    print(f"  Max seq len    : {global_max_seq_len}")
    if engine_env:
        print(
            "  Engine env     : "
            + ", ".join(f"{key}={value}" for key, value in sorted(engine_env.items()))
        )
    print(f"  fastkernels port   : {kb_nccl_port}")
    if vllm_port is not None:
        print(f"  vLLM port      : {vllm_port}")
        if flashinfer_namespace is not None:
            print(f"  vLLM FI ns     : {flashinfer_namespace}")
    print(f"  Output dir     : {args.output_dir}")
    if not args.skip_throughput:
        print(f"  Scenarios      : {', '.join(s['name'] for s in throughput_scenarios)}")
    else:
        print(f"  Scenarios      : (throughput skipped)")
    if latency_data:
        print(f"  Latency        : {', '.join(s['name'] for s in latency_scenarios)}"
              f" ({args.latency_iters} iters)")
    print("=" * 70)

    if is_whisper:
        vllm_worker = VLLM_WHISPER_WORKER
        kb_worker = FASTKERNELS_WHISPER_WORKER
    elif is_vlm or is_qwen_omni:
        vllm_worker = VLLM_VLM_WORKER
        kb_worker = FASTKERNELS_VLM_WORKER
    else:
        vllm_worker = VLLM_WORKER
        kb_worker = FASTKERNELS_WORKER

    if is_whisper:
        global_max_seq_len = 448  # Whisper max_target_positions

    # Config fingerprint shared by both phases: reuse a cached phase only when
    # the inputs that determine its outputs are unchanged.
    fingerprint = _fingerprint(
        model=args.model, tp=args.tp, seed=args.seed,
        temperature=args.temperature, enforce_eager=args.enforce_eager,
        max_layers=args.max_layers, max_model_len=global_max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        engine_env=engine_env,
        scenarios=scenario_data, latency=latency_data,
    )
    vllm_raw_path = (os.path.join(args.output_dir, "vllm_raw.json")
                     if args.output_dir else None)
    kb_raw_path = (os.path.join(args.output_dir, "fastkernels_raw.json")
                   if args.output_dir else None)

    # -- Run vLLM (one subprocess, all scenarios) --
    vllm_raw = None
    if not args.skip_vllm:
        if args.resume and vllm_raw_path:
            vllm_raw = _load_raw(vllm_raw_path, fingerprint)
        if vllm_raw is not None:
            print(f"  Resumed vLLM reference from cache: {vllm_raw_path}",
                  flush=True)
        else:
            short_name = args.model.split("/")[-1]
            vllm_config = {
                "model": args.model,
                "tp": args.tp,
                "seed": args.seed,
                "temperature": args.temperature,
                "enforce_eager": args.enforce_eager,
                "max_model_len": global_max_seq_len,
                **(
                    {"gpu_memory_utilization": args.gpu_memory_utilization}
                    if args.gpu_memory_utilization is not None
                    else {}
                ),
                "scenarios": scenario_data,
                "latency_scenarios": latency_data,
                "trust_remote_code": args.trust_remote_code,
                "load_format": "fastsafetensors",
                "is_qwen_omni": is_qwen_omni,
            }
            if args.max_layers is not None:
                vllm_config["max_layers"] = args.max_layers
            if args.kv_cache_dtype:
                vllm_config["kv_cache_dtype"] = args.kv_cache_dtype
            if getattr(args, "max_num_seqs", None) is not None:
                vllm_config["max_num_seqs"] = args.max_num_seqs
            if is_qwen_omni:
                vllm_config["limit_mm_per_prompt"] = {
                    "image": 1,
                    "video": 1,
                    "audio": 1,
                }
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(vllm_port)
            vllm_raw = run_worker(
                vllm_worker, vllm_config,
                f"vLLM [{short_name}] all scenarios (TP={args.tp})",
                timeout=10800,
                python_executable=args.vllm_python,
            )
            # Persist the vLLM reference immediately, BEFORE the fastkernels
            # phase runs, so a fastkernels crash never discards it.
            if vllm_raw is not None and vllm_raw_path:
                _save_raw(vllm_raw_path, vllm_raw, fingerprint)
        # Restore env set up for the vLLM subprocess (done above under
        # `not skip_vllm` regardless of whether we ran or resumed it).
        if previous_flashinfer_namespace_env is None:
            os.environ.pop("FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE", None)
        else:
            os.environ["FASTKERNELS_FLASHINFER_SOCKET_NAMESPACE"] = (
                previous_flashinfer_namespace_env
            )
        # The fastkernels worker takes max_layers via its JSON config, not the
        # env; clear it so the sitecustomize's vLLM weight filter stays inert
        # in that subprocess.
        if previous_max_layers_env is None:
            os.environ.pop("FASTKERNELS_MAX_LAYERS", None)
        else:
            os.environ["FASTKERNELS_MAX_LAYERS"] = previous_max_layers_env
        if vllm_raw is None:
            print("  ERROR: vLLM reference subprocess failed.")
            sys.exit(1)

    # -- Run fastkernels (one subprocess, all scenarios) --
    kb_raw = None
    if args.resume and kb_raw_path:
        kb_raw = _load_raw(kb_raw_path, fingerprint)
    if kb_raw is not None:
        print(f"  Resumed fastkernels outputs from cache: {kb_raw_path}",
              flush=True)
    else:
        kb_root = str(_PROJECT_ROOT)
        package_name = _PACKAGE_DIR.name
        kb_config = {
            "model": args.model,
            "tp": args.tp,
            "seed": args.seed,
            "temperature": args.temperature,
            "enforce_eager": args.enforce_eager,
            "max_model_len": global_max_seq_len,
            **(
                {"gpu_memory_utilization": args.gpu_memory_utilization}
                if args.gpu_memory_utilization is not None
                else {}
            ),
            "project_root": kb_root,
            "package_name": package_name,
            "scenarios": scenario_data,
            "latency_scenarios": latency_data,
        }
        if args.max_layers is not None:
            kb_config["max_layers"] = args.max_layers
        if args.kv_cache_dtype:
            kb_config["kv_cache_dtype"] = args.kv_cache_dtype
        if getattr(args, "max_num_seqs", None) is not None:
            kb_config["max_num_seqs"] = args.max_num_seqs
        short_name = args.model.split("/")[-1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(kb_nccl_port)
        kb_raw = run_worker(
            kb_worker, kb_config,
            f"fastkernels [{short_name}] all scenarios (TP={args.tp})",
            timeout=10800,
        )
        if kb_raw is not None and kb_raw_path:
            _save_raw(kb_raw_path, kb_raw, fingerprint)
    if kb_raw is None:
        print("  ERROR: fastkernels subprocess failed.")
        sys.exit(1)

    kb_latency = kb_raw.get("latency", [])
    vllm_latency = vllm_raw.get("latency", []) if vllm_raw else []

    # -- Compute throughput metrics per scenario --
    all_results = []
    if not args.skip_throughput:
        kb_results = kb_raw["throughput"]
        vllm_results = vllm_raw["throughput"] if vllm_raw else None

        for i, scenario in enumerate(throughput_scenarios):
            kb_data = kb_results[i]
            kb_tps = kb_data["total_output_tokens"] / kb_data["elapsed"]

            # Actual requests processed. Whisper workers report ``num_seqs``
            # explicitly; text/VLM workers don't, so fall back to the number of
            # returned outputs. Curated datasets (e.g. long-context, 64 rows)
            # run fewer than ``--num-seqs`` requests, so we must not assume
            # ``args.num_seqs`` here or the averages come out wrong.
            num_requests = kb_data.get("num_seqs") or len(
                kb_data.get("outputs", []))

            result = {
                "scenario": scenario["name"],
                "num_seqs": num_requests,
                "fastkernels_elapsed": kb_data["elapsed"],
                "fastkernels_output_tokens": kb_data["total_output_tokens"],
                "fastkernels_tok_per_s": kb_tps,
            }
            if "input_len" in scenario:
                result["input_len"] = scenario["input_len"]
            if "output_len" in scenario:
                result["output_len"] = scenario["output_len"]
            elif num_requests:
                result["avg_output_len"] = (
                    kb_data["total_output_tokens"] / num_requests
                )
            if is_whisper:
                result["total_audio_duration_s"] = kb_data.get(
                    "total_audio_duration_s", 0)

            if vllm_results is not None:
                v_data = vllm_results[i]
                v_tps = v_data["total_output_tokens"] / v_data["elapsed"]
                speedup = kb_tps / v_tps
                result["vllm_elapsed"] = v_data["elapsed"]
                result["vllm_output_tokens"] = v_data["total_output_tokens"]
                result["vllm_tok_per_s"] = v_tps
                result["speedup"] = speedup

                if args.temperature == 0.0:
                    alignment = compute_alignment(
                        kb_data["outputs"], v_data["outputs"]
                    )
                    result["alignment"] = alignment

            if args.output_dir:
                scenario_dir = os.path.join(args.output_dir, scenario["name"])
                os.makedirs(scenario_dir, exist_ok=True)

                kb_out_path = os.path.join(scenario_dir, "fastkernels_outputs.json")
                with open(kb_out_path, "w") as f:
                    json.dump(kb_data, f, indent=2)

                if vllm_results is not None:
                    vllm_out_path = os.path.join(scenario_dir, "vllm_outputs.json")
                    with open(vllm_out_path, "w") as f:
                        json.dump(vllm_results[i], f, indent=2)

            all_results.append(result)

        print(f"\n\n{'=' * 90}")
        print("  THROUGHPUT SUMMARY")
        print(f"{'=' * 90}")
        if is_whisper:
            header = (
                f"  {'SCENARIO':<16} {'SEQS':>5} {'AUDIO':>8} {'OUT':>5} "
                f"{'FASTKERNELS tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
                f"{'AVG PREFIX TOKS':>15}"
            )
        elif is_vlm or is_qwen_omni:
            header = (
                f"  {'SCENARIO':<16} {'SEQS':>5} {'OUT':>5} "
                f"{'FASTKERNELS tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
                f"{'AVG PREFIX TOKS':>15}"
            )
        else:
            header = (
                f"  {'SCENARIO':<16} {'SEQS':>5} {'IN':>5} {'OUT':>5} "
                f"{'FASTKERNELS tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
                f"{'AVG PREFIX TOKS':>15}"
            )
        print(header)
        print(f"  {'-' * 90}")

        for r in all_results:
            kb_tps_str = f"{r['fastkernels_tok_per_s']:,.0f}"
            v_tps_str = (
                f"{r['vllm_tok_per_s']:,.0f}" if "vllm_tok_per_s" in r else "N/A"
            )
            speedup_str = f"{r['speedup']:.2f}x" if "speedup" in r else "N/A"

            align = r.get("alignment", {})
            avg_match = align.get("avg_matching_tokens_per_request", 0)
            avg_out = align.get("avg_output_len", 0)
            if avg_out > 0:
                match_str = f"{avg_match:.1f}/{avg_out:.0f}"
            else:
                match_str = "N/A"

            if is_whisper:
                total_audio_s = r.get("total_audio_duration_s", 0)
                audio_min = total_audio_s / 60.0
                audio_str = f"{audio_min:.1f}m"
                out_str = f"{r.get('output_len', r.get('avg_output_len', 0)):>5.0f}"
                print(
                    f"  {r['scenario']:<16} {r['num_seqs']:>5} {audio_str:>8} "
                    f"{out_str} "
                    f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
                    f"{match_str:>15}"
                )
            elif is_vlm or is_qwen_omni:
                out_str = (
                    f"{r['output_len']:>5}"
                    if "output_len" in r
                    else f"{r.get('avg_output_len', 0):>5.0f}"
                )
                print(
                    f"  {r['scenario']:<16} {r['num_seqs']:>5} {out_str} "
                    f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
                    f"{match_str:>15}"
                )
            else:
                out_str = (
                    f"{r['output_len']:>5}"
                    if "output_len" in r
                    else f"{r.get('avg_output_len', 0):>5.0f}"
                )
                in_str = f"{r['input_len']:>5}" if "input_len" in r else f"{'var':>5}"
                print(
                    f"  {r['scenario']:<16} {r['num_seqs']:>5} {in_str} {out_str} "
                    f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
                    f"{match_str:>15}"
                )

        print(f"{'=' * 90}")

    # -- Latency summary table --
    latency_combined = []
    if kb_latency:
        print(f"\n{'=' * 110}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 110}")
        print(
            f"  {'SCENARIO':<18} {'BS':>4} {'OUT':>5} {'ITERS':>6}"
            f"  {'FASTKERNELS med':>12} {'vLLM med':>12}"
            f"  {'FASTKERNELS ms/tok':>15} {'vLLM ms/tok':>12} {'SPEEDUP':>8}"
        )
        print(f"  {'-' * 100}")

        for i, kb_lat in enumerate(kb_latency):
            kb_lats = np.array(kb_lat["latencies"])
            kb_med = float(np.median(kb_lats))
            kb_p99 = float(np.percentile(kb_lats, 99))
            bs = kb_lat["batch_size"]
            out_len = kb_lat["output_len"]
            total_out_tokens = bs * out_len
            kb_ms_per_tok = (kb_med / total_out_tokens) * 1000

            lat_result = {
                "scenario": kb_lat["name"],
                "batch_size": bs,
                "output_len": out_len,
                "num_iters": kb_lat["num_iters"],
                "fastkernels_median_s": kb_med,
                "fastkernels_p99_s": kb_p99,
                "fastkernels_ms_per_tok": kb_ms_per_tok,
                "fastkernels_latencies": kb_lat["latencies"],
            }
            if "input_len" in kb_lat:
                lat_result["input_len"] = kb_lat["input_len"]

            v_med_str = "N/A"
            speedup_str = "N/A"
            v_ms_str = "N/A"
            if i < len(vllm_latency):
                v_lat = vllm_latency[i]
                v_lats = np.array(v_lat["latencies"])
                v_med = float(np.median(v_lats))
                v_p99 = float(np.percentile(v_lats, 99))
                v_ms_per_tok = (v_med / total_out_tokens) * 1000
                speedup = v_med / kb_med
                v_med_str = f"{v_med:.4f}s"
                speedup_str = f"{speedup:.2f}x"
                v_ms_str = f"{v_ms_per_tok:.2f}"
                lat_result["vllm_median_s"] = v_med
                lat_result["vllm_p99_s"] = v_p99
                lat_result["vllm_ms_per_tok"] = v_ms_per_tok
                lat_result["speedup"] = speedup
                lat_result["vllm_latencies"] = v_lat["latencies"]

            print(
                f"  {kb_lat['name']:<18} {bs:>4}"
                f" {out_len:>5} {kb_lat['num_iters']:>6}"
                f"  {kb_med:.4f}s{'':<3} {v_med_str:>12}"
                f"  {kb_ms_per_tok:>13.2f}   {v_ms_str:>10} {speedup_str:>8}"
            )
            latency_combined.append(lat_result)

        print(f"{'=' * 110}")

    # -- Save combined results --
    if args.output_dir and (all_results or latency_combined):
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        combined = {
            "gpu": gpu,
            "model": args.model,
            "model_type": (
                "qwen_omni" if is_qwen_omni
                else ("vlm" if is_vlm else "llm")
            ),
            "tp": args.tp,
            "seed": args.seed,
            "temperature": args.temperature,
            "num_seqs": args.num_seqs,
            "enforce_eager": args.enforce_eager,
            "fastkernels_nccl_port": kb_nccl_port,
            "vllm_flashinfer_socket_namespace": flashinfer_namespace,
        }
        if args.max_layers is not None:
            combined["max_layers"] = args.max_layers
        if vllm_port is not None:
            combined["vllm_port"] = vllm_port
        if all_results:
            combined["scenarios"] = all_results
        if latency_combined:
            combined["latency_scenarios"] = latency_combined
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
