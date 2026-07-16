## Instructions for adding a new architecture to KB-Nano

### Architecture & Task Hierarchy

- Base Class: Every task must be an `nn.Module`.
- L1 Tasks (Primitives): Should be single-kernels or indivisible operators (e.g., SDPA).
- L2+ Tasks (Composites): Modules composed of smaller ops (e.g., Attention = SDPA + KV cache + QKV projections). Strict Constraint: Do not use `torch.nn` modules, `torch.nn.functional` methods, or external libraries. We must exclusively use L1 ops. If a required L1 op does not exist, we build it.
- L4 Tasks (Pipelines): These should only serve as wiring/configuration for lower-level ops. Do not implement complex or lengthy logic here.

### Interface & Design Principles

- SOTA Parity: A task's interface (`__init__` and `forward` methods) must closely match the corresponding module in the reference SOTA library. It's ok if `__init__` ends up being a little different, but let's try to be very strict with the `forward` method's alignment. This ensures users can easily deploy our generated kernels into SOTA codebases without heavy modification. 
- Condense Variants: Consolidate very similar operators into a single class (e.g., use one unified Attention class for Llama, Qwen, and Mixtral) to prevent the agent from generating redundant kernels. Exception: Only split into multiple variants if condensing forces the interface to diverge noticeably from the SOTA reference.

### Benchmarking & Correctness

- Test Environment: Extend `tests/bench_<SOTA_LIBRARY>.py`. If adding the first model for a new library, create this file.
- Performance Goal: Compare the latency and throughput of our kb-nano baseline against the SOTA library. We must be faster or roughly on par.
- Standardized Workloads: For LLMs, use the reasonably sized workloads established in `tests/bench_vllm.py`. For LLMs, measure throughput using 1,000 requests across three splits: Prefill-heavy (1024 prefill / 512 decode), Balanced (512 prefill / 512 decode), and Decode-heavy (512 prefill / 1024 decode). For other model architectures (diffusion, recommendation, robotics, graphics, etc), create a similar standardized workload if one doesn't exist yet for that SOTA library / architecture type / modality.
- Correctness Check: Verify that our outputs match the SOTA library. For LLMs, check the number of consecutive matching tokens per request; aim for an average of at least 100 matching tokens per request.

### Configuration & PR Workflow

- TP Degree: Select the appropriate Tensor Parallelism degree based on model size (e.g., use TP=1 for models under 10B parameters).
- Iteration: Run the benchmark and continually iterate on your implementation until performance reaches SOTA parity.
- Merge Requirements: Once performance is on par, document the benchmark results in the README. The PR is then ready to merge.