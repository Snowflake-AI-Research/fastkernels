# Hugging Face computation coverage

This audit tests which HF model computations can be built from existing
FastKernels operations and small, implemented adaptations. Correctness supports
the coverage result; latency helps diagnose inefficient constructions. We do not
claim production serving support or pretrained-model quality.

This document defines setup and methodology. [REVIEW.md](REVIEW.md) explains the
reviewer's work; [review.csv](review.csv) lists assignments and open issues.

## Setup and first run

Use a source checkout of this audit branch and follow the repository's
[installation instructions](../../README.md#quick-start). Activate its Python
environment. The tested environment uses Python 3.12, PyTorch 2.11.0, CUDA 13,
and a B200. [environment.txt](environment.txt) records tested versions for
troubleshooting; it is not an installation requirements file. A fresh dependency
installation has not been validated.

The installed Transformers package serves FastKernels. The independent HF
reference imports a separate, pinned checkout in its subprocess. Both workers
use the same Python environment in the example below.

```bash
# Run from the repository root, with the environment activated.
export HF_HOME=/path/to/your/storage/hf-cache
export HF_COVERAGE_REFERENCE_SOURCE=/path/to/your/storage/transformers-reference
export HF_COVERAGE_RUNS=/path/to/your/storage/hf-coverage-runs

# Once: obtain the reference source.
git clone https://github.com/huggingface/transformers.git "$HF_COVERAGE_REFERENCE_SOURCE"
git -C "$HF_COVERAGE_REFERENCE_SOURCE" checkout da6c53e431f7c9ef0691239d4ce89b0f711ecad7

python -m fastkernels.hf_coverage --list
# Choose an available GPU and a new output directory.
CUDA_VISIBLE_DEVICES=0 python -m fastkernels.hf_coverage bert \
  --hf-python "$VIRTUAL_ENV/bin/python" \
  --hf-source "$HF_COVERAGE_REFERENCE_SOURCE" \
  --output-dir "$HF_COVERAGE_RUNS/bert"
```

This run generates shared random weights and inputs; it downloads configuration
files if needed, not pretrained weights. Some HF models also fetch compiled
kernels. Bamba requires network access for that lookup even with cached
configuration; forcing `HF_HUB_OFFLINE=1` can prevent execution. If preparation
fails, inspect `prepare.log`; do not substitute another HF revision.

Use `--help` for options. `--variant` selects a declared workload when required.
`--reuse-from RUN_DIRECTORY` reuses common weights and inputs for the same case,
dtype, and seed. Every run needs a new or empty output directory.

## Configuration and input policy

**Select the computation from HF, then choose defensible test dimensions.**
Use the relevant public-task example in pinned HF's model/configuration
class documentation, forward-method documentation, or model documentation page:

- A named checkpoint selects its pinned `config.json`, with omitted settings
  supplied by the pinned configuration class.
- An example using `ViTConfig()` or another configuration constructor selects
  that constructor's defaults, usually defined in `configuration_<model>.py`.

Both sources can exist for the same architecture. Follow the relevant example,
not whichever file happens to exist. Record the source and rationale in
`cases.py`. Disclose conflicting or unusable examples; consult official
architecture sources when HF does not establish a usable choice.

Preserve the selected task's block types, activations, modalities, enabled
features, outputs, and state behavior. Features disabled in that default path
can be excluded. Model dimensions and input sizes may be reduced under these rules:

- Keep every distinct block and required dependency or state update.
- Preserve head dimensions, attention grouping, and feed-forward ratios where
  possible; justify changes to specialized dimensions.
- Keep enough tokens, patches, audio, or frames to exercise enabled windows,
  chunks, and state history. Do not shrink a test merely to obtain a pass.
- Record size changes in `dimension_overrides` and their reason in
  `dimension_purpose`. There is no universal scaling factor.

For example, BERT starts from `google-bert/bert-base-uncased` but uses four layers,
hidden size 256, four heads, feed-forward width 1024, and vocabulary size 1024.
It retains 64-wide heads, a 4:1 feed-forward ratio, and a 512-token input. The
checkpoint supplies configuration; the run uses shared random weights.

Input size is a separate choice. Use HF's documented input or processor settings
where they establish a choice; otherwise specify the workload explicitly. A
sampling rate, for example, does not determine audio duration.

Use valid shared random weights and synthetic inputs. Check that initialization
does not zero gates or collapse outputs and hide required computation. Such tests
need investigation before acceptance. A case may name zero-initialized gates in
`reference.randomize_zero_parameters`; preparation gives only those parameters
shared normal random values (standard deviation 0.2, seed plus two), so enabled
branches affect the outputs. Supplied weights are preserved. The case and saved
preparation record disclose this choice.

When native initialization collapses outputs or makes untrained generation
invalid, `reference.fan_in_normal_modules` names modules whose matrix weights
use random values scaled by their number of input terms (seed plus three).
Convolutions account for filter size and groups; transposed convolutions also
account for stride. Embeddings, biases, normalization weights, and tied output
heads are preserved unless an explicit exception is described below. Preparation
records the affected weight names and standard deviations. Each use needs a demonstrated reason.

PPDocLayoutV2 additionally initializes BatchNorm scales to one in the declared
feature-extractor modules. HF's small random scales repeatedly suppress image
features and leave proposal scores nearly tied. Both implementations receive
identical modified weights; biases and running statistics stay unchanged.
Preparation records every affected scale. This is a test initialization exception,
not a change to model computation or to supplied weights.

CSM's declared random preparation also follows the official converter's
identical-value copy between its two audio embedding tables and randomizes zero
codec centroids so generated codes affect the waveform. Supplied weights are
preserved, and the preparation record names every changed buffer.

CSM also requests deterministic cuDNN convolutions on both sides: repeated native
waveform decoding otherwise exceeded the numerical tolerance. The run records
this setting; it does not change the comparison rule.

These checks test numerical computation, not audio or tracking quality. Cases
requiring supplied inputs, packed weights, or adapter weights are
flagged in `review.csv`: `--input-dict` and `--state-dict` load common tensors but
do not generate missing formats. Implement valid synthetic preparation where needed.

Phi4 speech has a CPU preparer for its active random adapters and synthetic audio.
It reads the pinned configuration, downloads no pretrained weights, and refuses
an existing nonempty output directory. Its fixed seeds reproduce the reviewed
workload; preparing inputs does not establish correctness.

```bash
python -m fastkernels.hf_coverage.prepare_phi4 \
  --hf-source "$HF_COVERAGE_REFERENCE_SOURCE" \
  --output-dir "$HF_COVERAGE_RUNS/phi4-speech-inputs"
CUDA_VISIBLE_DEVICES=0 python -m fastkernels.hf_coverage phi4_multimodal \
  --variant speech --hf-python "$VIRTUAL_ENV/bin/python" \
  --hf-source "$HF_COVERAGE_REFERENCE_SOURCE" \
  --state-dict "$HF_COVERAGE_RUNS/phi4-speech-inputs/weights.pt" \
  --input-dict "$HF_COVERAGE_RUNS/phi4-speech-inputs/inputs.pt" \
  --output-dir "$HF_COVERAGE_RUNS/phi4-speech"
```

**Reuse adequate existing results, whether full-size or reduced.** They must
cover the reviewed implementation, required outputs, and meaningful computation.
Run again to address a specific evidence gap or evaluate a changed workload,
not solely to change model size. Preserve the configuration, weights, inputs,
and source version associated with each result.

DiNAT uses one explicit reference correction: its query/key/value axes are
reordered to match NATTEN. Saved loading information records `dinat_qkv_layout`;
the pinned HF checkout is unchanged. This is not a candidate patch.

## Construction rules

- **Reuse:** call existing operations when they provide the required behavior.
  Unchanged internal components count; disclose missing optimization interfaces.
- **Connecting code:** layout changes, indexing/copies, required additions, fixed
  scalar scaling, dtype conversion, and position/mask preparation are allowed.
  Loops may connect operations across layers or state steps.
- **Computation:** activation-dependent arithmetic, reductions, comparisons, and
  gating require an existing operation or explicit patch. Computing a mask from
  activations is computation, even if indexing with that mask is connecting code.
- **Patches:** put the implementation in `patches/`; identify its parent and exact
  change. Retain the main reduction, dependency, communication, and storage
  strategy. A short wrapper alone does not justify a patch.
- **Compositions:** do not introduce artificial work or memory expansion to
  simulate a missing operation, such as multiplication via a diagonal matrix.

Metadata-only counts/scales may use supplied padding or token masks. Fixed-weight
transformations may occur during loading. Activation computation and state
updates belong inside measured execution. Connecting code and patches count
toward runtime.

These rules constrain audit implementations, not future optimization candidates.
A missing-operation claim needs a concrete computational gap after checking
relevant operations, straightforward compositions, and permitted adaptations.

## Correctness and diagnostic timing

The runner calls FastKernels' tensor comparator, not the `fastkernels bench` CLI.
Each floating tensor needs at least 99% of elements within its dtype tolerances;
integer outputs must match exactly. The exact rule is saved in `result.json`.
BF16 is the default unless the case declares otherwise. OmDet's matching
intentional infinities have an explicit exception; further exceptions need review.
A known semantic bug must be fixed even if the numerical threshold passes.

Check required outputs and state. Selected continuation workloads use prefill
and two supplied-token continuations, not generated-text alignment. Investigate
failures on common intermediate inputs. Where supported, use
`--dtype float32 --upcast-from RUN_DIRECTORY` to diagnose the same rounded weights
and inputs; an FP32 pass does not waive a failure at the evaluated dtype.

Both workers use inference mode, a fixed seed, disabled TF32, 10 warmup calls,
and 50 synchronized wall-time samples. State preparation is outside timing;
execution, dispatch, and state updates are timed; output copying and saving
follow timing. Seeding does not guarantee GPU-kernel determinism. Investigate
unstable references and suspicious slowdowns before interpreting results.
Performance is diagnostic, with no acceptance cutoff.

## Results and code layout

The runner resolves the case, prepares common tensors, executes HF and our model
in separate processes, then compares outputs and saves results:

| Run artifact | Contents |
|---|---|
| `job.json`, `prepare.json` | Case, resolved configuration, seed, and preparation details |
| `result.json` | Per-output numerical results, timings, versions, and source hashes |
| `prepare.log`, `reference.log`, `implementation.log` | Preparation and execution messages |
| `prepared.pt` | Shared weights and inputs |
| `reference_outputs.pt`, `implementation_outputs.pt` | Compared tensors |

`passed_provisional` means that run passed the numerical rule, not the whole
review. `mismatch` means numerical disagreement; `error` means incomplete
execution. Both return exit status 1. `speedups` means **HF time / our time**;
values below one mean we are slower.

| Repository file | Responsibility |
|---|---|
| `cases.py` | HF source, overrides, input specification, execution sequence, and selected outputs |
| `models/` | Model assembly, weight mapping, and workload calls |
| `patches/` | Implemented numerical adaptations |
| `reference.py` | Pinned HF preparation and execution |
| `runner.py`, `__main__.py` | Shared measurement, comparison, and CLI |
| `corpus.json` | Pinned HF revision and all 447 candidate entries |
| `review.csv` | Ownership, recorded evidence, and next actions |

In `cases.py`, `reference` identifies the HF class and configuration source;
`config_overrides` and `dimension_overrides` declare changes; `input` specifies
common inputs; `workload` selects execution; and `outputs`/`decode_outputs`
select comparisons. Backend, dtype, generation, and cache options appear where
needed. `reference_backend=None` lets HF choose, while omission selects `eager`.
`reference_experts_backend` separately selects HF's mixture-of-experts backend
when its automatic choice cannot execute the declared quantization format.
Descriptions explain choices; each run saves its fully resolved configuration.

Keep large tensors, profiles, and investigation history outside Git. Record enough
information for another reviewer to reproduce an unresolved issue.
