# FastKernels

A reproducible benchmarking suite for evaluating custom CUDA / Triton / PyTorch kernels across a broad zoo of modern model architectures: dense and MoE LLMs, linear-attention LLMs, diffusion / video / audio models, multimodal and vision encoders, detection / edge networks, 3D / robotics / science models, recommendation models, and world models.

Operators are organized into four levels of abstraction (L1 single-kernel, L2 fused/composite, L3 layer/block, L4 end-to-end pipeline). For each architecture the suite ships a reference implementation (`tasks/baseline/`) and a slot for candidate replacements (`tasks/candidate/`); a runner swaps candidates in, validates correctness against the baseline, and measures speedup.

## Layout

```
fastkernels/
├── tasks/
│   ├── baseline/                # Reference implementations
│   │   ├── L1/                  # Single-kernel ops
│   │   ├── L2/                  # Fused / composite blocks
│   │   ├── L3/                  # Decoder / encoder layers
│   │   └── L4/                  # Full-model pipelines
│   ├── candidate/               # Slot for replacement kernels (gitignored)
│   └── reference/               # Frozen reference snapshots
├── infra/                       # Engines, weight loaders, kernel swapper
├── bench/                       # Benchmark drivers (kernels / eval / e2e)
├── agent/                       # Optional LLM-driven kernel-generation agent
└── tests/                       # Comparison benchmarks vs upstream baselines
```

The full set of architectures, their HuggingFace references, default dtypes, and per-level operator counts is enumerated in the appendix of the accompanying paper.

## Install

Requires Python 3.10+, CUDA 12.x, and a recent NVIDIA GPU (Hopper / Blackwell tested; Ampere supported for a subset of kernels).

```bash
git clone <repo-url> fastkernels
cd fastkernels/fastkernels
pip install .
```

This installs the `fastkernels` CLI plus all benchmark dependencies (PyTorch, Triton, FlashAttention, DeepGEMM, fastsafetensors, plus the per-architecture reference packages — diffusers, timm, transformers, flash-linear-attention, ultralytics, sam3, openfold3, etc.). Some optional comparisons (vLLM, vllm-omni, JAX/Equinox for TTT-E2E, OpenPI for Pi0) are best installed in separate environments and pointed at via `--<framework>-python` flags on the relevant `bench_*.py` scripts.

## Run

```bash
# List all available kernel-level benchmark targets
fastkernels kernels --list

# Run a single L1/L2/L3 operator microbench
fastkernels kernels run --target rms_norm

# Run the multi-architecture L4 evaluation sweep
fastkernels eval --help

# Run end-to-end throughput / latency
fastkernels e2e throughput --help
fastkernels e2e latency    --help
```

Per-architecture comparison benchmarks live under `fastkernels/validate/`. Run one
directly, or let `fastkernels validate <scenario>` dispatch the right harness per model:

```bash
fastkernels validate minimal                              # dispatch by model in a scenario table
fastkernels validate full --max-requests 8 --max-layers 12

python fastkernels/validate/bench_vllm.py      --model <hf-id>   # LLMs vs vLLM
python fastkernels/validate/bench_fla.py       --model <hf-id>   # GLA / RetNet / RWKV-7 vs FLA
python fastkernels/validate/bench_vllm_omni.py --model <hf-id>   # Diffusion / video / TTS vs vllm-omni
python fastkernels/validate/bench_diffusers.py --model <hf-id>   # SDXL vs diffusers
python fastkernels/validate/bench_timm.py      --model <hf-id>   # SigLIP-2 / DINOv3 / SwinV2 vs timm
python fastkernels/validate/bench_embedding.py --model <hf-id>   # BGE-M3 / ColBERTv2
python fastkernels/validate/bench_recsys.py                      # DLRMv2 / LightGCN
python fastkernels/validate/bench_dp3.py                         # 3D-Diffusion-Policy
python fastkernels/validate/bench_openpi.py                      # Pi0 / OpenPI
python fastkernels/validate/bench_openfold3.py                   # OpenFold3
python fastkernels/validate/bench_ttt_e2e.py                     # TTT-E2E (vs JAX reference)
python fastkernels/validate/bench_sam.py                         # SAM3
# (see fastkernels/validate/ for the full list)
```


## Adding a candidate kernel

1. Drop a replacement implementation in `tasks/candidate/L<level>/<op_name>.py` exposing the same class name as the baseline.
2. Run `fastkernels kernels run --target <op_name>` (or any L4 benchmark) — the swapper auto-discovers the candidate, validates numerical agreement against the baseline, and reports speedup.

## Citation

If you use FastKernels, please cite the accompanying paper.

## License

See `LICENSE`.
