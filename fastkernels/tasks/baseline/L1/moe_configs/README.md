# Tuned Triton MoE configs

`L1.moe_grouped_gemm._get_moe_configs` searches this directory (after
`FASTKERNELS_TUNED_CONFIG_FOLDER`, an adjacent vLLM source checkout, and the
installed vLLM wheel's own tables) for a JSON config keyed by expert count,
intermediate size and device:

    E=<num_experts>,N=<intermediate_size>,device_name=<gpu>.json

Each file maps a token count `M` to the `_fused_moe_kernel` launch config;
`get_triton_config` snaps the runtime `M` to the nearest key. Without a file the
`_get_default_config` heuristic applies, which keys `BLOCK_SIZE_M` on `M` alone.

## `E=16,N=14336,device_name=NVIDIA_B200.json` (AI21-Jamba-Mini-1.7)

vLLM ships no table for this shape, and the heuristic's small-`M` picks cost
Jamba real time: `_fused_moe_kernel` is 63-74% of its decode step. The tuned
entries here are worth 3.5% of the whole batch-1 and batch-32 decode step,
measured in-engine.

Only `M <= 64` is tuned. Above that the entries deliberately repeat the
heuristic's own "large" config, because the standalone sweep's picks for larger
`M` do not survive contact with the engine: its choice for `M=256`
(`BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64`) measured 1.07x faster in
isolation but made the real batch-256 decode step 6.7% SLOWER, and re-timing the
candidates inside the captured CUDA graph
(`tests/debug/tune_jamba_moe_ingraph.py`) put the heuristic's config back on
top by 1.06x. The engine replays this kernel inside a graph alongside 32 layers
of other work, and that context decides which config wins -- so entries belong
here only where an in-engine A/B confirmed the win.

Regenerate with:

    python tests/debug/tune_jamba_moe_config.py --write     # sweep + emit
    python tests/debug/tune_jamba_moe_ingraph.py --bs 256   # confirm in-graph

`--write` pins every key above `--max-tuned-m` (default 64) to the heuristic's
config for the reason above, so it reproduces this file rather than the sweep's
in-isolation winners. Raise the cutoff only after an in-graph A/B says to.
