# Paper e2e run: handoff

End-to-end evaluation of the agents' kernel sets (Dr. Kernel, Claude Code, KDA, AKO;
independent and sequential winners) on the 11 models of `fastkernels/scenarios/default.yaml`.

## For the person running it (one 8xB200 node)

```bash
git clone -b paper-e2e git@github.com:Snowflake-AI-Research/fastkernels.git && cd fastkernels
pip install -e .                      # same environment you use for fastkernels
bash scripts/paper_e2e.sh             # re-run the same command to resume after any interruption
```

When it prints `DONE` (or `PREFLIGHT FAILED`), email back the `paper_e2e_results.tgz` it names
(a few MB). Needs a Hugging Face login with access to the default models (Llama-3.1, FLUX.1-dev,
...) and git access to `sfc-gh-goliaro/fastkernels-results` (candidate sets, cloned by the script).

What it does: verifies the candidate sets' checksums; a ~30-60 min preflight (every model, tiny
workloads, a do-nothing candidate set) that stops early with a clear message if anything is
broken; then the full run (~1-2 days): per model a baseline, a noise run (fresh-compile baseline,
the calibration floor) and, per candidate set, drop-and-retry until a deployable subset of the
agent's kernels runs.

## What `fastkernels e2e` measures (for the paper)

- **Kernels per model:** only the set's kernels that model uses (from the default capture).
- **Drop-and-retry:** a crash is attributed to the candidate file named innermost in the
  traceback (else found by bisection) and that kernel is dropped; grossly broken outputs (mean
  per-sample discrepancy > 0.5) are bisected the same way. Cheap probe runs (8 requests) search
  for the deployable subset; the real workloads run once on it. Every attempt, drop and reason
  (compile / runtime / timeout / import / incorrect) is recorded.
- **Timing:** the scenario's real workloads on the production path (torch.compile + CUDA graphs
  where fastkernels uses them), after an untimed warmup pass (lazy JIT compiles excluded).
- **Correctness (per sample, d in [0,1]):** LLMs / GLA / Qwen3-VL: 1 - teacher-forced top-1
  agreement with the baseline tokens (free-running greedy output diverges even between two
  correct runs); FLUX / Oasis: 1 - (centred) cosine of outputs; YOLOv10: IoU-matched detection
  F1 / IoU / score error; BGE-M3: 1 - embedding cosine; OpenFold3: structure agreement.
- **Scoring:** `fastkernels-iclr/figures/make_macroeval.py --results <out>/e2e/results`.

## Results layout

`<out>/e2e/results/<NN>_<model>/{baseline,noise,<set>}.json` (summaries; see
`fastkernels/e2e/core.py`), `<out>/e2e/runs/...` (per-run logs and outputs, stays on the node),
`<out>/e2e/progress.jsonl` (event log). `python -m fastkernels.e2e.report <out>/e2e` prints a
status table.

## Operating notes / known issues

- **Disk:** the 11 default models need ~1.1 TB of Hugging Face weights (Qwen3-VL-235B-FP8
  ~240 GB, Qwen3-Next-80B ~160 GB, Kimi-Linear-48B ~100 GB, GPT-OSS-120B ~65 GB, ...);
  downloaded on first use if not cached. Host RAM peaks ~20 GB per run (VLM video, BGE-M3).
- **OpenFold3 weights:** the harness checkpoint (`OpenFold/OpenFold3`, gated) is not
  required; the adapter downloads the public `openfold3_params/of3_ft3_v1.pt` (same
  architecture) plus MSA folders automatically. Set `FASTKERNELS_OF3_CHECKPOINT` to use
  another checkpoint.
- **Timing is host-sensitive** for small, launch-bound workloads (GLA, YOLOv10, OpenFold3
  short chains, Oasis/BGE-M3 latency): speedups are only meaningful between runs on the same
  node -- which is what the script does. Each throughput workload gets one untimed pass
  first (lazy JIT compiles excluded from timings).
- **Nondeterminism:** multi-GPU (tp>1) runs are not bit-deterministic (Qwen3-VL-235B tp=4:
  ~97% teacher-forced agreement between two baseline runs); the per-model noise run
  measures this floor and MacroEval calibrates correctness against it.
- **Verified on Modal B200s (2026-09-24), final code:** the preflight (baseline, noise run,
  do-nothing `selftest` set) passes on all 11 default models, including Qwen3-Next and
  Kimi-Linear (tp=2), GPT-OSS-120B (tp=2) and Qwen3-VL-235B (tp=4); drop-and-retry was
  exercised with a real agent set (kda-seq) on Llama, Qwen3-Next, Kimi-Linear, GPT-OSS,
  GLA, FLUX, YOLOv10 and Oasis.
- **Candidate JIT compile** can take 10-20 min per set on first import (`--prebuild` does it
  once per set before the timed runs).
