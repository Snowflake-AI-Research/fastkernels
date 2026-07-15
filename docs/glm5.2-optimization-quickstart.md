# Optimizing GLM-5.2 with fastkernels → vLLM/SGLang

Quick-start for porting GLM-5.2 into fastkernels, optimizing its kernels against a
SOTA reference (vLLM or SGLang), and shipping the winners back.

**Install** (once, from the repo root — provides the `fastkernels` CLI used below):

```
pip install -e .
```

**Prereqs:** references at `/home/yak/reference_code/{vllm,sglang}/`; one 8-GPU node.
GLM-5.2 is huge (`tp=8`) — use `--max-layers` (build/run only the first N transformer
layers) and `--max-requests` (fewer prompts) to iterate fast on a single node.

---

## 1. Import GLM-5.2 into fastkernels (skills)

Run these `/`-skills in a Claude Code session (reference lib = whichever is SOTA):

```
/fastkernels-add-model     zai-org/GLM-5.2 sglang   # port arch + L1–L3 ops from reference
/fastkernels-check-parity  GLM-5.2 sglang           # adversarial mismatch/shortcut hunt → fix
/fastkernels-validate-e2e  zai-org/GLM-5.2          # tests/bench_vllm.py: align + speedup
```

**Output:** `fastkernels/tasks/baseline/L4/glm5_2.py` (+ its L1–L3 ops), matching the
reference's interface/dtypes; e2e token-alignment `100%` and speedup `≥1.0x`, eager & compiled.

## 2. (optional) List the operators GLM-5.2 uses

```
fastkernels list --map | grep -A40 'glm5_2:'
```

**Output:**
```
  glm5_2:
    L1  rms_norm                  RMSNorm
    L2  moe_grouped_gemm          GroupedGemmMoE
    L3  glm_decoder               GLMDecoderLayer
    ...
```

## 3. Capture shapes/dtypes for the workload

```
fastkernels capture fastkernels/scenarios/glm5.2.yaml --max-layers 4 --max-requests 64
```

**stdout** (per scenario·workload; parallel across GPUs, one scenario at a time at `tp=8`):
```
Discovering operators in fastkernels.tasks.baseline ...
  Instrumented 583 operator class(es).

########## Scenario: zai-org/GLM-5.2 (dtype=bfloat16, tp=8) ##########
  [1/6] Loading model weights...
  NOTE: --max-layers capping transformer layers 92 -> 4
  [1/6] Model loaded in 12.7s
  [4/6] Allocating KV cache...
  Engine ready in 41.2s total

=== Capturing workload: mixed ===
Loading 'mixed' workload prompts (64) ...
  Generation: 61/64 reached EOS; lengths=[228, 577, 43, ...]
  Verification 1 (hook cross-check)   [PASS]: 19/19 operator forwards match the capture.
  Verification 2 (mock batching replay) [PASS]: 5/5 checks passed (312 simulated steps).
  Captured 20 executed operator class(es) (of 583 instrumented).
  Report written to ~/.fastkernels/captures/zai-org__GLM-5.2_tp8_bfloat16_mixed_req64_seqsauto_L4_eager.json
```
One JSON per scenario·workload records every op's init/forward `shape`/`dtype`/`stride`.
Drop `--max-layers/--max-requests` for the full-fidelity capture.

## 4. Stub candidates, then write optimized kernels

```
fastkernels create-stubs --architecture glm5_2   # arch = L4 stem from step 2
```

**Output:** empty candidate modules in `fastkernels/tasks/candidate/L{1..4}/*.py`.
Then hand them to an agent to implement (fed the step-3 capture as the target shapes):

```
/fastkernels-bench-ops GLM-5.2 sglang     # agent writes + tunes each candidate vs reference
```

## 5. Benchmark at the kernel level (from the capture) — iterate

```
fastkernels bench --target moe_grouped_gemm --self-test   # harness sanity (~1.0x)
fastkernels bench --target moe_grouped_gemm               # baseline vs candidate
```

**Output:**
```
  OP                     SHAPES  CORRECT  BASELINE   CANDIDATE  SPEEDUP
  moe_grouped_gemm            7  OK       412 µs     287 µs     1.44x
```
Not faster / not correct → edit the candidate and re-run. Iterate to `SPEEDUP > 1.0x`.

## 6. Evaluate end-to-end (correctness + perf + cuda-graphs)

```
fastkernels eval fastkernels/scenarios/glm5.2.yaml --max-layers 4 --max-requests 64
```

**Output** — baseline-vs-candidate summary per scenario (CUDA graphs on; add
`--enforce-eager` to compare eager). `IN=var` since real prompts vary; `MATCH` is exact
token-id agreement:
```
==========================================================================================
  THROUGHPUT SUMMARY   zai-org/GLM-5.2 (tp=8)   correctness: PASS
==========================================================================================
  SCENARIO          SEQS    IN   OUT  BASELINE tok/s  CANDIDATE tok/s  SPEEDUP     MATCH
  ------------------------------------------------------------------------------------------
  mixed               64   var   390           9,102           11,540    1.27x     64/64
  long-context        64   var   256             341              402    1.18x     64/64
==========================================================================================

==============================================================================================
  LATENCY SUMMARY
==============================================================================================
  SCENARIO             BS   OUT  ITERS  BASELINE med  CANDIDATE med  SPEEDUP
  ----------------------------------------------------------------------------------------------
  single-request        1   128      5       0.0412s        0.0341s    1.21x
  fixed-batch-32       32   128      5       0.7620s        0.6903s    1.10x
==============================================================================================
```
Run `--self-test` first (candidate = baseline → `100%` match, `~1.0x`) to sanity-check the
harness. Correctness must stay `PASS`. (The analogous FASTKERNELS-vs-vLLM table is what
step 1's `/fastkernels-validate-e2e` prints.)

## 7. Ship the winners into vLLM/SGLang

The candidates were written to the reference op's exact interface (steps 1 & 5), so they're
drop-in — copy each validated kernel back:

```
cp fastkernels/tasks/candidate/L2/moe_grouped_gemm.py \
   /home/yak/reference_code/sglang/python/sglang/srt/layers/moe/<file>.py
```

Re-run the reference's own tests to confirm parity, then upstream.

---

**Loop:** steps 4→5→6 until the kernel and e2e speedups hold with `correctness: PASS`.
`--max-layers`/`--max-requests` keep every loop cheap; run the full config once before shipping.
