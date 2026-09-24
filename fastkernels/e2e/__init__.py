"""End-to-end evaluation of agent-generated kernel sets (``fastkernels e2e``).

For every scenario (model) in a scenarios table, and every candidate set:

* a **baseline** run (production kernels) records timings and reference outputs;
* a **noise** run repeats the baseline in a fresh process (fresh compile) and is scored
  against it -- the run-to-run nondeterminism floor used to calibrate correctness;
* each **candidate set** is swapped in, restricted to the kernels that model uses, and
  run with **drop-and-retry**: a crash is attributed to the candidate kernel named in the
  traceback (or found by bisection), that kernel is dropped, and the model is re-run.

Every run is a separate process (``fastkernels.e2e.runner``) so compile caches, CUDA state
and imported candidate modules never leak between runs. Model-family specifics (how to
build the model, run workloads, save outputs and score them) live in
``fastkernels.e2e.adapters``.
"""
