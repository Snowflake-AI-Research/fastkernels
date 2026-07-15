"""Developer utilities for fastkernels.

Standalone helper tools that are not part of the core capture/bench runtime:

* ``create_stubs`` -- scaffold candidate-kernel skeletons under
  ``tasks/candidate/`` (CLI: ``fastkernels create-stubs``).
* ``build_datasets`` -- deterministically (re)build the frozen real-prompt
  E2E datasets (``python -m fastkernels.utils.build_datasets``).
"""
