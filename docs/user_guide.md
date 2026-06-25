# fastkernels User Guide

A standalone LLM inference engine with a benchmarking suite for evaluating custom CUDA/Triton/PyTorch kernels. Supports **Llama 3.1** and **Mixtral-8x7B** with tensor parallelism.

---

## Quick Start

```bash
git clone https://github.com/Snowflake-AI-Research/fastkernels.git
cd fastkernels
pip install .
```

---

## Benchmarking

fastkernels provides three complementary benchmarking tools.

### E2E Benchmark CLI

`fastkernels e2e` mirrors vLLM's benchmarking CLI. The same flags work in both tools, making it easy to compare results:

```bash
# Throughput (mirrors `vllm bench throughput`)
fastkernels e2e throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 --tp 4

# Latency
fastkernels e2e latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 128 --output-len 128

# Evaluate candidate kernels from tasks/candidate/ against baseline
fastkernels eval \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Kernel Benchmark CLI

`fastkernels kernels` benchmarks a single operator replacement. It instantiates the baseline and candidate modules, compares their `forward()` outputs with a tolerance-normalized max error ratio, and measures wall-clock speedup. FP8 `(tensor, scale)` output pairs are compared after dequantization with scale-derived tolerances.

```bash
# List all benchmarkable targets
fastkernels kernels --list

# Show model-to-operator mapping
fastkernels kernels --map

# Benchmark a candidate from tasks/candidate/
fastkernels kernels --target rms_norm

# Filter by model and TP degree
fastkernels kernels \
    --target rms_norm \
    --model llama31 \
    --tp 1
```

A benchmark is **PASS** if the candidate output's max error ratio is <= 1.0 against the baseline. Speedup > 1.0 means your kernel is faster than the baseline.

### vLLM Alignment Test

`tests/bench_vllm.py` runs fastkernels and vLLM side-by-side across three workload scenarios (prefill-heavy, balanced, decode-heavy) plus latency benchmarks, comparing throughput and per-token alignment:

```bash
python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct
python tests/bench_vllm.py --model meta-llama/Llama-3.1-70B-Instruct --tp 4

# Latency only (skip throughput)
python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct --skip-throughput

# Parse and plot results
python tests/utils/parse_vllm_bench_results.py
```

Results are saved to `tests/results/<GPU>/<model>_tp<N>/results.json`. The parser auto-discovers these files and generates tables and plots in `tests/plots/<GPU>/`.

For a quick correctness check (no throughput measurement), use the `--skip-throughput` flag:

```bash
python tests/bench_vllm.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --skip-throughput --skip-latency
```

---

## LLM Kernel Agent

The agent uses Claude to generate replacement kernels, validate them, and benchmark them end-to-end:

```bash
# Generate all L1 kernels for Llama
fastkernels agent \
    --model meta-llama/Llama-3.1-8B-Instruct --level 1

# CUDA-only kernels (no Triton/PyTorch builtins)
fastkernels agent \
    --model meta-llama/Llama-3.1-8B-Instruct --level 1 --cuda-only

# Mixtral L2 operators with TP
fastkernels agent \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 --level 2 --tp 4
```

The agent discovers operators at the specified level, generates replacements in parallel, validates compilation and numerical correctness, patches all successful kernels into the model, and reports token match rate and speedup. Failed kernels are retried with error feedback.

Generated kernels are saved to `tasks/candidate/L{level}/{op_name}.py`.

---

## Experiment Tracking

Every `fastkernels agent`, `fastkernels kernels`, `fastkernels eval`, and `fastkernels e2e` run is automatically logged to [MLflow](https://mlflow.org). This provides:

- **Kernel lineage**: Every generated kernel is stored as an MLflow artifact, linked to the run parameters that produced it.
- **Benchmark history**: Speedup, correctness, and max error ratio for every operator across every benchmark run.
- **Comparison**: Use the MLflow UI to compare runs side-by-side and visualize speedup trends.

### What gets logged

| Command | Logged data |
|---------|-------------|
| `fastkernels agent` | Run params, per-op generation success/attempts, unit test results, e2e speedup, kernel source code |
| `fastkernels kernels` | Bench params, per-operator per-scenario speedup/correctness, kernel source code |
| `fastkernels eval` | Per-model throughput/latency speedup, alignment rate, MacroEval speedup/correctness/coverage/score, wall-clock time |
| `fastkernels e2e` | Throughput (tokens/s), latency (percentiles), serve metrics (TTFT, TPOT, ITL) |

### Querying from the CLI

```bash
fastkernels history                  # recent runs
fastkernels history --op rms_norm    # history for one operator
fastkernels history --best           # best speedup per operator
fastkernels history --limit 50       # show more results
```

#### Sample output: `fastkernels history`

```
======================================================================
  RECENT TRACKED RUNS
======================================================================
  TIMESTAMP          RUN NAME                            KEY METRICS
  ──────────────────────────────────────────────────────────────────
  2026-03-16 16:44   e2e_throughput_Llama-3.1-8B         tokens_per_second=15000.0
  2026-03-16 16:44   kernels_rms_norm                    avg_speedup=1.64x  total_passed=2  total_failed=0
  2026-03-16 16:44   agent_L1_Llama-3.1-8B               e2e_speedup=1.15x  e2e_token_match_rate=97.0%
======================================================================
```

#### Sample output: `fastkernels history --op rms_norm`

```
======================================================================
  TRACKING HISTORY: rms_norm
======================================================================
  TIMESTAMP          RUN NAME                     SPEEDUP   PASS  ERR_RATIO     RUN ID
  ──────────────────────────────────────────────────────────────────
  2026-03-16 16:44   kernels_rms_norm               1.64x    2/2   1.20e-04   a1b2c3d4
  2026-03-15 14:20   kernels_rms_norm               1.31x    2/2   2.10e-04   e5f6a7b8
  2026-03-14 09:55   agent_L1_Llama-3.1-8B          1.12x    2/2   3.40e-03   c9d0e1f2
======================================================================
```

#### Sample output: `fastkernels history --best`

```
======================================================================
  BEST SPEEDUP PER OPERATOR (from kernel benchmarks)
======================================================================
  OPERATOR                  BEST SPEEDUP DATE               RUN ID
  ──────────────────────────────────────────────────────────────────
  rms_norm                         1.64x 2026-03-16 16:44   a1b2c3d4
  rotary_emb                       1.22x 2026-03-15 11:30   d3e4f5a6
  silu_and_mul                     1.45x 2026-03-14 09:55   b7c8d9e0
======================================================================
```

### Tracking API for custom agents

Any kernel optimization script can use the tracking API directly:

```python
from fastkernels.bench.tracking import tracker

with tracker.start_run("my-run", params={"model": "llama", "level": 1}):
    # Log a generated kernel (stored as MLflow artifact)
    tracker.log_kernel("rms_norm", level=1, code=kernel_source)

    # Log kernel benchmark results (pass KernelBenchResult directly)
    tracker.log_kernel_bench(result)

    # Log eval results (pass EvalReport directly)
    tracker.log_eval(report)

    # Log e2e results
    tracker.log_e2e(results_dict, bench_type="throughput")

    # Log any custom metrics
    tracker.log_metrics({"my_score": 0.95, "compile_time": 12.3})
```

The full API surface is five logging functions plus one context manager:

| Function | Purpose |
|----------|---------|
| `tracker.start_run(name, params, tags)` | Context manager that opens an MLflow run |
| `tracker.log_kernel(op, level, code)` | Log kernel source code as artifact |
| `tracker.log_kernel_bench(result)` | Log `KernelBenchResult` metrics |
| `tracker.log_eval(report)` | Log `EvalReport` metrics |
| `tracker.log_e2e(results, bench_type)` | Log E2E benchmark metrics |
| `tracker.log_metrics(dict)` | Log arbitrary key-value metrics |

### MLflow Web UI

```bash
fastkernels mlflow-ui
# Open http://localhost:5000 in your browser
# Press Ctrl+C to stop
```

The UI launches a local MLflow server at http://localhost:5000 backed by the `mlruns/` directory. All runs are logged under the `fastkernels` experiment.

#### Navigating the UI

1. **Experiment list** (left sidebar): Select the `fastkernels` experiment to see all tracked runs.
2. **Runs table**: Each row is a tracked run (agent, kernel benchmark, eval, or e2e). Columns show run name, start time, duration, and logged metrics. Click column headers to sort -- e.g., sort by `e2e_speedup` to find your fastest runs.
3. **Filtering**: Use the search bar to filter runs by parameters or metrics. Examples:
   - `params.level = "1"` -- show only L1 runs
   - `params.cuda_only = "True"` -- show CUDA-only agent runs
   - `metrics.e2e_speedup > 1.0` -- show runs that beat the baseline
   - `tags.tier = "agent"` -- show only agent runs (vs `"kernel"`, `"eval"`, `"e2e"`)

#### Inspecting a run

Click any run to open its detail page:

- **Parameters**: Model, level, TP degree, LLM model, seed, and other run configuration.
- **Metrics**: Per-operator generation success (`gen_rms_norm_success`), unit test results (`utest_rms_norm_success`, `utest_rms_norm_max_diff`), and e2e results (`e2e_speedup`, `e2e_token_match_rate`).
- **Artifacts**: Browse the `kernels/` folder to view and download the exact source code of every generated kernel. Failed generations store error traces under `errors/`.

#### Comparing runs

1. Select two or more runs using the checkboxes in the runs table.
2. Click **Compare**. The comparison view shows:
   - **Parameter diff**: Which parameters changed between runs (e.g., `cuda_only: True` vs `False`).
   - **Metric comparison**: Side-by-side metric values -- useful for seeing how `e2e_speedup` or `e2e_token_match_rate` changed across iterations.
   - **Artifact diff**: Compare kernel source code between runs to see how the generated code evolved.

#### Downloading kernel artifacts

From any run's artifact browser, click a kernel file (e.g., `kernels/rms_norm.py`) to preview its contents. Click the download button to save it locally. This is useful for recovering a high-performing kernel from a previous run:

```bash
# Or query artifacts programmatically:
python -c "
import mlflow
mlflow.set_tracking_uri('file://$(pwd)/mlruns')
client = mlflow.tracking.MlflowClient()
client.download_artifacts('<run_id>', 'kernels/rms_norm.py', '/tmp/')
"
```

#### Tracking data structure

Each run is tagged with a `tier` that indicates its source:

| Tag | Source command | Key metrics |
|-----|---------------|-------------|
| `agent` | `fastkernels agent` | `gen_{op}_success`, `utest_{op}_success`, `e2e_speedup`, `e2e_token_match_rate` |
| `kernel` | `fastkernels kernels` | `{op}_avg_speedup`, `{op}_passed`, `{op}_failed`, `avg_speedup` |
| `eval` | `fastkernels eval` | `avg_throughput_speedup`, `avg_latency_speedup`, `alignment_rate`, `macro_speedup`, `macro_correctness`, `macro_coverage`, `macro_score` |
| `e2e` | `fastkernels e2e` | `tokens_per_second`, `avg_latency`, `mean_ttft_ms` (varies by bench type) |

### Disabling tracking

Tracking is always-on when `mlflow` is installed. To disable it, uninstall mlflow: `pip uninstall mlflow`. The system degrades gracefully -- a single warning is printed, and all tracking calls become no-ops.

---

## Design Philosophy

### The L1-L4 Hierarchy

All model operators live under `tasks/baseline/` at four abstraction levels:

- **L1** -- Single-kernel ops (e.g. `rms_norm`, `silu_and_mul`, `rotary_emb`, `linear`)
- **L2** -- Multi-op blocks (e.g. `LlamaAttention`, `LlamaMLP`, `MixtralMoE`, `QKVParallelLinear`)
- **L3** -- Decoder layers (e.g. `LlamaDecoderLayer`, `MixtralDecoderLayer`)
- **L4** -- Full models (e.g. `LlamaForCausalLM`, `MixtralForCausalLM`)

Higher levels compose lower ones via standard Python imports:

```
L4/llama.py (LlamaForCausalLM)
  L3/llama_decoder.py (LlamaDecoderLayer)
    L2/attention.py (LlamaAttention)
      L1/store_kvcache.py, L1/flash_attn_*.py
      L2/parallel_linear.py (QKVParallelLinear, RowParallelLinear)
        L1/linear.py, L1/allreduce.py
    L2/llama_mlp.py (LlamaMLP)
      L1/silu_and_mul.py
      L2/parallel_linear.py
    L1/rms_norm.py
```

When you replace an L1 operator, the change propagates upward through every level that uses it. The bench suite traces these dependencies automatically via import graph analysis.

### Interface Mirroring

Every module's `__init__` and `forward` signatures are designed to mirror the corresponding vLLM module. This makes it easy to port optimizations between the two codebases and ensures candidate kernels can be validated against a well-known reference.

Key conventions:

| fastkernels module | vLLM equivalent | `forward` signature |
|---|---|---|
| `LlamaAttention` | `LlamaAttention` | `(positions, hidden_states) -> Tensor` |
| `LlamaDecoderLayer` | `LlamaDecoderLayer` | `(positions, hidden_states, residual) -> (Tensor, Tensor)` |
| `QKVParallelLinear` | `QKVParallelLinear` | `(x) -> (output, bias)` |
| `RowParallelLinear` | `RowParallelLinear` | `(x) -> (output, bias)` |
| `MergedColumnParallelLinear` | `MergedColumnParallelLinear` | `(x) -> (output, bias)` |
| `VocabParallelEmbedding` | `VocabParallelEmbedding` | `(input_ids) -> Tensor` |
| `ParallelLMHead` | `ParallelLMHead` | `(hidden_states) -> Tensor` |
| `SiluAndMul` | `SiluAndMul` | `(x) -> Tensor` |
| `RMSNorm` | `RMSNorm` | `(x, residual=None) -> Tensor or (Tensor, Tensor)` |

All parallel linear layers return `(output, bias)` tuples to match vLLM. `LlamaAttention` stores `rotary_emb` as an `__init__` parameter (not a `forward` argument), matching vLLM's design where each `LlamaAttention` owns its rotary embedding.

---

## Adding a New Model or Kernel

### Adding a new model

Follow the L1 -> L4 pattern. For a hypothetical "NewModel":

1. **L1**: Identify which single-kernel ops are needed. Reuse existing L1 ops where possible (e.g. `RMSNorm`, `SiluAndMul`). Write new L1 modules only for ops that don't exist yet.
2. **L2**: Compose L1 ops into multi-op blocks (attention, MLP). Mirror the corresponding vLLM module's `__init__` and `forward` signatures.
3. **L3**: Write a decoder layer that combines L2 attention + L2 MLP + L1 normalization.
4. **L4**: Write the full model class (`NewModelForCausalLM`) with embedding, decoder stack, and LM head.

Each module should be a drop-in match for its vLLM counterpart. The bench suite auto-discovers new operators from the import graph.

### Writing a candidate kernel

To write a replacement kernel for benchmarking:

1. Read the baseline source to understand the `forward` signature.
2. Write an `nn.Module` with the **exact same class name** and **exact same `forward` signature**.
3. Place it at `tasks/candidate/L{level}/{op_name}.py` and run the benchmark.

Example -- replacing `RMSNorm`:

```python
# tasks/candidate/L1/rms_norm.py
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        result = (self.weight * x).to(x.dtype)
        if residual is not None:
            return result, residual
        return result
```

```bash
fastkernels kernels --target rms_norm
```

You can use pure PyTorch, Triton (`triton.jit`), inline CUDA (`torch.utils.cpp_extension.load_inline`), or external libraries. Avoid importing `vllm` or `sgl_kernel` in your replacement -- the point is to provide an alternative implementation.

---

## Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FASTKERNELS_NCCL_PORT` | `29501` | TCP port for NCCL process group initialization. |
| `FASTKERNELS_PROFILE` | `0` | Set to `1` to enable internal profiling timers. |
| `FASTKERNELS_DISABLE_CUSTOM_AR` | `0` | Set to `1` to disable custom IPC all-reduce (use NCCL). |

### Troubleshooting

**GPU memory**: The engine fills ~90% of free GPU memory with KV cache. Use a smaller model or increase TP if you run out of memory.

**NCCL port conflict**: Set `FASTKERNELS_NCCL_PORT=29502` if another instance is using the default port.

**Custom all-reduce hangs**: Disable with `FASTKERNELS_DISABLE_CUSTOM_AR=1`.

**HuggingFace auth**: Run `huggingface-cli login` for gated models like Llama.

**Bench target not found**: Run `fastkernels kernels --list` to see available targets. Names match file names under `tasks/baseline/` (without `.py`).

**Class name mismatch**: Your replacement class must have the exact same name as the baseline class (e.g. `RMSNorm`, not `MyRMSNorm`).
