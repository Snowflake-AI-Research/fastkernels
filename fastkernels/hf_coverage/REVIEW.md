# Review handoff

Start with the [README](README.md) for setup and methodology. Filter
[review.csv](review.csv) to `owner=reviewer` for your assignments. Each `model`
identifier is the CLI argument and filename in `models/`; `cases.py` defines its
workload. `next_action` identifies the open issue or next check.

## Assignment and priorities

You own **221 registered implementations**. Filter `review_priority` in
`review.csv` and work in this order:

| Priority | Models | Work |
|---|---:|---|
| `investigate` | 44 | Resolve numerical failures, large slowdowns, or specific evidence concerns. |
| `complete_execution` | 4 | Resolve CTRL's runtime errors; obtain paired GPU evidence for Gemma3, Jais2, and PPLCNetV3. |
| `optional` | 173 | Independently review previously passing GPU-tested workloads only if time permits. |

The first group contains 35 numerical investigations and four recorded slowdowns
of at least 3x HF time; two models appear in both, giving 37 distinct models.
Seven more have specific evidence concerns: zeroed branches in Idefics, the three
PE modality models, and PPFormulaNet; UDOP's coordinate units; and ModernBERT's
source mismatch. These are not routine optional reviews.

All 182 entries outside the numerical/execution groups have saved passing primary
GPU comparisons and timing measurements. Two are slow and seven have the concerns
above, leaving 173 optional reviews. Their timing records vary in quality; a
recorded time is not a cleared performance result. Reuse adequate evidence for
its tested workload. Do not rerun solely for model size or a second review.
Optional independent checks may be omitted under time pressure; known failures
and evidence gaps remain unresolved until addressed. The CSV's
`historical_evidence` records earlier milestones, not current acceptance policy.

You may complete missing behavior, repair bugs, and optimize your models within
the agreed rules. Coordinate shared-helper edits and discuss methodology changes.
We own the other 226 entries, including the entries without registered implementations,
and coordinate shared infrastructure and integration. `review_priority` is left
blank for our entries because this prioritization covers the reviewer assignment.

## Completing a model review

1. **Check construction and scope.** Compare the model and selected case with
   pinned HF. Verify required default computation, operation reuse, patches,
   outputs, and state. Justify any reduced dimensions.
2. **Check numerical evidence.** Inspect the tested configuration, source version,
   outputs, and whether random initialization hides a branch. Retain an adequate
   existing result; otherwise run the case and record what gap it addresses.
3. **Resolve problems.** Isolate numerical differences on common inputs; use FP32
   for diagnosis where appropriate. Profile suspicious latency, fix avoidable
   inefficiencies within the construction rules, and recheck changed execution.
   Separate demonstrated causes from hypotheses.
4. **Record the conclusion.** Give the case, source revision, command, accessible
   result location, checks performed, conclusion, and remaining issue in your
   change description. Update `next_action` with a short outcome or remaining task;
   retain `historical_evidence` as the original snapshot. Keep large artifacts
   outside Git.

Classify construction as **existing operations**, **implemented patches**, or
**missing operation**, with a specific justification; leave unfinished decisions
pending. Report correctness separately. Diagnosing a failure does not make it a
pass, and slow execution alone does not establish a missing operation.

Prior results and diagnostic tensors are not bundled with this source checkout.
Reuse them when the supporting evidence is available for inspection. Otherwise
obtain evidence for the reviewed case. Ordinary new runs need no private scripts.

## Known investigations

These are starting points from saved investigations. Check whether each issue
still applies to current code. Additional preparation, weak-signal, and execution
issues are listed per model in `review.csv`. Read existing findings before
repeating diagnostics; some component causes are isolated while full-model
acceptance remains unresolved.

### SmolVLM

The [vision path](models/smolvlm.py) now uses cuDNN attention and omits its mask
only when all patches are valid; common-input tests required both changes to
match HF. A first-text-layer difference remained on identical merged embeddings.
Compare that layer's normalization, projections, positions, attention, and
feed-forward outputs; the remaining text cause is not established.

### xLSTM

[Sequential recurrence](models/xlstm.py) differs from HF's chunked arithmetic.
FP32 log-forget calculations improved internal state agreement but barely changed
the hidden-output error; normalization also differed. Investigate both paths and
check that random initialization does not zero feed-forward gates. Saved prefill
time was **47.20x HF**; neither numerical nor performance closure is established.

### RWKV

The [current implementation](models/rwkv.py) retains HF's scaled numerator,
denominator, and running maximum, with a BF16 prefill readout cast. The saved
failure predates this change; recheck prefill, both continuations, and state.
Saved prefill time was **84.65x HF**. Profiling found many operation launches per
token, while HF uses a fused recurrence kernel. Resolve numerical behavior and
performance separately.

### EnCodec

The [recurrence](models/encodec.py) batches input projections but executes
recurrent linear operations, gates, and state updates per step, unlike HF's
fused recurrent backend. Review the
[recurrent-linear](patches/encodec_recurrent_linear.py) and
[product-gate](patches/product_gate.py) patches and remaining launch costs.
Saved forward time was up to **5.09x HF**; its passing numerical result used
pretrained common weights and is evidence for that exact workload.

### EfficientLoFTR

The [model](models/efficientloftr.py) passed after reusing SDPA attention, but
forward time was **5.43x HF**. A profile attributed 96.590 ms to nine `CodecTop1`
calls. Three calls express 4800-by-4800 predicates as two-value reductions,
launching 69,120,000 blocks; HF uses elementwise comparisons. Review admissibility
and improve this construction within the rules. The bottleneck is identified,
not repaired or proven unavoidable.

### UDOP and ModernBERT

[UDOP](models/udop.py) originally received boxes in 0–1000 units where the path
expects normalized 0–1 coordinates. A corrected comparison matched; verify the
reviewed inputs, visual-token selection, and corresponding results.
[ModernBERT](models/modernbert.py) has a saved passing result whose implementation
hash differs from current code. Establish whether the change affects that result
or rerun; this is an evidence gap, not a demonstrated numerical bug.

The ratios above are historical diagnostic timings, not acceptance cutoffs.
Bamba, Sam2Video, and GOT-OCR2 have numerical passes but withheld timing evidence
because of known GPU timing problems.
