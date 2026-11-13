# Benchmarking Agent-Generated Kernels

FastKernels evaluates AI-generated GPU kernels against state-of-the-art baselines via a two-stage pipeline: isolated kernel testing and end-to-end model evaluation.

### 1. Scaffold and Implement

Generate empty stubs with correct `__init__`/`forward` signatures, then have your agent fill them:

```bash
fastkernels create-stubs --architecture llama
# Agent implements: tasks/candidate/L<level>/<op_name>.py
```

*Note: The candidate `nn.Module` must exactly match the baseline's class name and signature. Lower-level replacements automatically propagate up the hierarchy.*

### 2. Kernel-Level Benchmark

Test isolated correctness (error ratio $\le$ 1.0) and performance (speedup > 1.0) using runtime-captured shapes:

```bash
fastkernels bench             # Benchmark all available candidates (compile_model + piecewise CUDA graphs)
fastkernels bench --level 1   # Benchmark all L1 candidates
fastkernels bench --target rms_norm  # Benchmark a single operator
fastkernels bench --enforce-eager    # Eager timing only (disable compile + CUDA graphs)
```

### 3. End-to-End Evaluation

Verify that the candidates preserve greedy token matching and improve true model throughput when integrated into the full architecture:

```bash
fastkernels eval fastkernels/scenarios/llama3.1.yaml
```