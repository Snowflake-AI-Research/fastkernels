# Developer Guide

## Internal Validation against Reference Libraries

The `validate` commands and scripts under `fastkernels/validate/` are used to test the FastKernels baseline implementations against reference SOTA production libraries (to ensure end-to-end token alignment and equal performance). **They are not intended for benchmarking user candidate kernels** (use `fastkernels bench` and `fastkernels eval` for that).

Note that running validations requires installing the per-architecture reference packages — diffusers, timm, transformers, flash-linear-attention, ultralytics, sam3, openfold3, etc. Some optional comparisons (vLLM, vllm-omni, JAX/Equinox for TTT-E2E, OpenPI for Pi0) are best installed in separate environments and pointed at via `--<framework>-python` flags on the relevant `bench_*.py` scripts.

Run a validation directly, or let `fastkernels validate <scenario>` dispatch the right harness per model:

```bash
fastkernels validate minimal                              # dispatch by model in a scenario table
fastkernels validate full --max-requests 8 --max-layers 12

python fastkernels/validate/bench_vllm.py      --model <hf-id>   # LLMs vs vLLM
python fastkernels/validate/bench_fla.py       --model <hf-id>   # GLA / RetNet / RWKV-7 vs FLA
python fastkernels/validate/bench_vllm_omni.py --model <hf-id>   # Diffusion / video / TTS vs vllm-omni
...
# (see fastkernels/validate/ for the full list)
```