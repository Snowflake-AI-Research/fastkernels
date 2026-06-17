"""Layer-by-layer hidden-state diff: kb-nano vs Microsoft BitNet GPU SOTA.

Runs the SAME single-prompt prefill through both models, captures the
output of every named op (norm, qkv, rope, attn, sub_norm, o, ffn ...)
via forward hooks, and prints a per-layer per-op diff so we can pinpoint
exactly which op first goes out-of-spec with SOTA.

This is for *correctness diagnosis only* — not used in production.

Usage::

    BITNET_REPO=/home/yak/vllm_repo/BitNet \
        python tests/diff_bitnet_layers.py [--prompt-len 16] [--layer N]

Setup expects the same converted SOTA checkpoints as bench_microsoft_bitnet.py
(``$BITNET_REPO/gpu/checkpoints/model_state_{int2,fp16}.pt``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

# Keep BitNet checkpoint loader from going through fastsafetensors GDS
# (we need plain CPU/GPU loading for repeatable, deterministic numerics).
os.environ.setdefault("KB_NANO_DISABLE_FASTSAFETENSORS", "1")
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Capture utilities
# ---------------------------------------------------------------------------

class Captured:
    """Per-prompt collection of named tensors captured during forward."""

    def __init__(self) -> None:
        self.named: dict[str, torch.Tensor] = {}

    def put(self, key: str, tensor: torch.Tensor) -> None:
        # Detach + clone + cpu so we can compare across two model invocations
        # that may overwrite their intermediate buffers.
        self.named[key] = tensor.detach().to(torch.float32).cpu().clone()


def _neox_to_gptj_per_head(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Permute the LAST dim of ``x`` from NeoX RoPE layout to GPT-J interleaved
    layout, applied independently per head along that dim.

    NeoX layout (kb-nano's qkv_proj output, HF format): per head of size D the
    values are ``[d_0, d_1, ..., d_{D/2-1}, d_{D/2}, ..., d_{D-1}]`` with the
    rotation pairs being ``(d_i, d_{D/2+i})``.

    GPT-J interleaved (SOTA's wqkv after ``invert_convert_q``): per head the
    values are ``[d_0, d_{D/2}, d_1, d_{D/2+1}, ...]`` with rotation pairs
    being ``(d_{2i}, d_{2i+1})``.

    Used by the diff harness so that kb-nano's per-head q/k vectors are in
    the same layout as SOTA's before per-element comparison.
    """
    *lead, last = x.shape
    assert last % head_dim == 0
    n_heads = last // head_dim
    half = head_dim // 2
    y = x.reshape(*lead, n_heads, 2, half)         # (..., H, 2, D/2)
    y = y.transpose(-2, -1).contiguous()           # (..., H, D/2, 2)
    return y.reshape(*lead, n_heads * head_dim)


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_abs, mean_abs, cos_sim) between two tensors of equal shape."""
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    n = a.numel()
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = (a.norm() * b.norm()).item()
    cos = float((a @ b).item() / denom) if denom > 0 else 0.0
    return max_abs, mean_abs, cos


# ---------------------------------------------------------------------------
# kb-nano forward with hooks
# ---------------------------------------------------------------------------

def run_kb_nano(prompt_ids: list[int]) -> tuple[Captured, torch.Tensor]:
    """Run kb-nano BitNet prefill once for ``prompt_ids`` (single seq) and
    return (captures, last_token_logits)."""
    from kb_nano.infra import weight_loader as _wl
    # GDS may not be available in all environments; force the safetensors
    # CPU+stream loader to avoid cuFile errors.
    _wl._HAS_FASTSAFETENSORS = False
    from kb_nano.infra.weight_loader import load_model
    from kb_nano.infra.context import set_context, reset_context
    from kb_nano.tasks.baseline.L1.bitnet_linear import BitLinear, BitLinearMerged
    from kb_nano.tasks.baseline.L1.rms_norm import RMSNorm
    from kb_nano.tasks.baseline.L2.bitnet_attention import BitNetAttention
    from kb_nano.tasks.baseline.L2.bitnet_mlp import BitNetMLP
    from kb_nano.tasks.baseline.L3.bitnet_decoder import BitNetDecoderLayer

    print("[kb] loading model...")
    model, config = load_model("microsoft/bitnet-b1.58-2B-4T", dtype=torch.bfloat16)
    model = model.eval().cuda()

    cap = Captured()

    # ----- 1) Attach forward hooks to capture all per-op outputs -----
    # We track per-layer, per-op outputs.
    def _hook_layer(layer_idx: int, dec: BitNetDecoderLayer) -> None:
        attn: BitNetAttention = dec.self_attn
        mlp: BitNetMLP = dec.mlp

        def _post_input_norm(_m, _i, out):
            # input_layernorm returns (hidden, residual) when residual is given,
            # else hidden alone.  We want the hidden (post-norm) portion.
            h = out[0] if isinstance(out, tuple) else out
            cap.put(f"L{layer_idx:02d}/01_input_norm", h)
        dec.input_layernorm.register_forward_hook(_post_input_norm)

        def _post_qkv(_m, _i, out):
            cap.put(f"L{layer_idx:02d}/02_qkv_proj", out)
        attn.qkv_proj.register_forward_hook(_post_qkv)

        # RoPE doesn't have a single nn.Module call site here; instead we
        # wrap the attention forward to capture the post-rope q,k and the
        # post-attn output before sub_norm.
        orig_attn_fwd = attn.forward
        def _attn_with_capture(positions, hidden_states):
            qkv = attn.qkv_proj(hidden_states)
            cap.put(f"L{layer_idx:02d}/02_qkv_proj", qkv)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            cap.put(f"L{layer_idx:02d}/03a_q_pre_rope", q)
            cap.put(f"L{layer_idx:02d}/03b_k_pre_rope", k)
            cap.put(f"L{layer_idx:02d}/03c_v",          v)
            q, k = attn.rotary_emb(positions, q, k)
            cap.put(f"L{layer_idx:02d}/04a_q_post_rope", q)
            cap.put(f"L{layer_idx:02d}/04b_k_post_rope", k)
            attn_output = attn.attn(q, k, v)
            cap.put(f"L{layer_idx:02d}/05_attn_out", attn_output)
            attn_output = attn.attn_sub_norm(attn_output)
            cap.put(f"L{layer_idx:02d}/06_attn_sub_norm", attn_output)
            o = attn.o_proj(attn_output)
            cap.put(f"L{layer_idx:02d}/07_o_proj", o)
            return o
        attn.forward = _attn_with_capture

        def _post_post_attn_norm(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cap.put(f"L{layer_idx:02d}/08_post_attn_norm", h)
        dec.post_attention_layernorm.register_forward_hook(_post_post_attn_norm)

        # MLP: capture gate_up, post act_fn, post sub_norm, post down.
        orig_mlp_fwd = mlp.forward
        def _mlp_with_capture(x):
            gate_up = mlp.gate_up_proj(x)
            cap.put(f"L{layer_idx:02d}/09_gate_up", gate_up)
            inner = mlp.act_fn(gate_up)
            cap.put(f"L{layer_idx:02d}/10_act_fn", inner)
            inner = mlp.ffn_sub_norm(inner)
            cap.put(f"L{layer_idx:02d}/11_ffn_sub_norm", inner)
            out = mlp.down_proj(inner)
            cap.put(f"L{layer_idx:02d}/12_down_proj", out)
            return out
        mlp.forward = _mlp_with_capture

    for i, layer in enumerate(model.model.layers):
        _hook_layer(i, layer)

    # Capture final norm output too.
    def _post_final_norm(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap.put("XX/final_norm", h)
    model.model.norm.register_forward_hook(_post_final_norm)

    # ----- 2) Set up engine context for a single-seq prefill -----
    n_tokens = len(prompt_ids)
    input_ids = torch.tensor(prompt_ids, dtype=torch.int64, device="cuda")
    positions = torch.arange(n_tokens, dtype=torch.int64, device="cuda")
    cu_seqlens_q = torch.tensor([0, n_tokens], dtype=torch.int32, device="cuda")
    cu_seqlens_k = cu_seqlens_q.clone()
    # Slot mapping: we don't need an actual KV cache because attention
    # only reads the K,V we just stored at [0..n_tokens-1].  But the
    # paged cache code requires valid slot indices, so allocate a single
    # block worth of dummy cache.
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    block_size = 256
    n_blocks = max(1, (n_tokens + block_size - 1) // block_size)
    slot_mapping = torch.arange(n_tokens, dtype=torch.int32, device="cuda")
    block_tables = torch.arange(
        n_blocks, dtype=torch.int32, device="cuda",
    ).reshape(1, n_blocks)
    req_id_per_token = torch.zeros(n_tokens, dtype=torch.int32, device="cuda")

    # Allocate KV cache for every BitNetAttention (engine normally does this).
    from kb_nano.tasks.baseline.L2.attention_impl import Attention
    for mod in model.modules():
        if isinstance(mod, Attention):
            kv_shape = (n_blocks, block_size, n_kv_heads, head_dim)
            mod.k_cache = torch.zeros(kv_shape, dtype=torch.bfloat16, device="cuda")
            mod.v_cache = torch.zeros(kv_shape, dtype=torch.bfloat16, device="cuda")

    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=n_tokens,
        max_seqlen_k=n_tokens,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        req_id_per_token=req_id_per_token,
    )

    print("[kb] running prefill...")
    with torch.no_grad():
        hidden = model(input_ids, positions)
        # ParallelLMHead.project consults context.cu_seqlens_q to pick the
        # per-seq last index inside the FULL hidden tensor.  Pass the
        # untouched hidden so the indexing is in-bounds.
        logits = model.compute_logits(hidden)

    reset_context()
    last_logits = logits[-1].detach().to(torch.float32).cpu()
    return cap, last_logits


# ---------------------------------------------------------------------------
# SOTA forward with hooks
# ---------------------------------------------------------------------------

def run_sota(prompt_ids: list[int], bitnet_repo: str) -> tuple[Captured, torch.Tensor]:
    """Run Microsoft BitNet GPU prefill model once and return (captures, last_logits)."""
    sota_gpu = Path(bitnet_repo) / "gpu"
    if not sota_gpu.is_dir():
        raise FileNotFoundError(f"SOTA gpu dir not found: {sota_gpu}")

    sys.path.insert(0, str(sota_gpu))
    cwd_save = os.getcwd()
    os.chdir(sota_gpu)
    try:
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)

        import model as fast  # type: ignore
        from xformers.ops.fmha.attn_bias import (
            BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
        )

        args = fast.ModelArgs(use_kernel=False)
        prefill_model = fast.Transformer(args)

        ck = sota_gpu / "checkpoints" / "model_state_fp16.pt"
        if not ck.exists():
            raise FileNotFoundError(
                f"Missing SOTA prefill checkpoint at {ck}.  Run convert_safetensors.py + "
                "convert_checkpoint.py first."
            )
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        prefill_model.load_state_dict(sd, strict=True)
        prefill_model.eval()

        cap = Captured()

        # Hook each TransformerBlock's sub-modules.
        for i, block in enumerate(prefill_model.layers):
            attn = block.attention
            ffn = block.feed_forward

            def _post_input_norm(_m, _i, out, _i_=i):
                cap.put(f"L{_i_:02d}/01_input_norm", out)
            block.attention_norm.register_forward_hook(_post_input_norm)

            # SOTA combines RoPE inside Attention.forward; capture by
            # monkey-patching forward to inject taps.
            def _attn_with_capture(x, cache, attn_bias, _attn=attn, _i=i):
                xqkv = _attn.wqkv(x)
                cap.put(f"L{_i:02d}/02_qkv_proj", xqkv)
                xq = xqkv[:, : (_attn.n_local_heads * _attn.head_dim)]
                xkv = xqkv[:, (_attn.n_local_heads * _attn.head_dim):]
                xk, xv = xkv.chunk(2, 1)
                cap.put(f"L{_i:02d}/03a_q_pre_rope", xq)
                cap.put(f"L{_i:02d}/03b_k_pre_rope", xk)
                cap.put(f"L{_i:02d}/03c_v",          xv)
                output_shape = xq.shape
                heads_per_group = _attn.n_local_heads // _attn.n_local_kv_heads
                xq_v = xq.view(1, xq.shape[0], _attn.n_local_kv_heads, heads_per_group, _attn.head_dim)
                xk_v = xk.view(1, xk.shape[0], _attn.n_local_kv_heads, 1, _attn.head_dim)
                xv_v = xv.view(1, xv.shape[0], _attn.n_local_kv_heads, 1, _attn.head_dim)
                cache_k, cache_v = cache
                from xformers.ops import rope_padded
                xq_v = rope_padded(
                    xq=xq_v, xk=xk_v, xv=xv_v,
                    cache_k=cache_k, cache_v=cache_v,
                    attn_bias=attn_bias, theta=_attn.rope_theta,
                )
                # Capture post-rope: cache_k / cache_v now hold rotated K/V at
                # positions 0..n_tokens-1; post-rope q is in xq_v.
                # Reshape post-rope q to flat (T, q_size) for direct compare.
                cap.put(f"L{_i:02d}/04a_q_post_rope",
                        xq_v.reshape(output_shape).contiguous())
                cap.put(f"L{_i:02d}/04b_k_post_rope",
                        cache_k[0, : xq.shape[0]].reshape(xq.shape[0], -1).contiguous())
                from xformers.ops import fmha
                output = fmha.memory_efficient_attention_forward(
                    xq_v, cache_k, cache_v, attn_bias, op=fmha.flash.FwOp,
                )
                output = output.reshape(output_shape)
                cap.put(f"L{_i:02d}/05_attn_out", output)
                output = _attn.attn_sub_norm(output)
                cap.put(f"L{_i:02d}/06_attn_sub_norm", output)
                output = _attn.wo(output)
                cap.put(f"L{_i:02d}/07_o_proj", output)
                return output
            attn.forward = _attn_with_capture

            def _post_ffn_norm(_m, _i, out, _i_=i):
                cap.put(f"L{_i_:02d}/08_post_attn_norm", out)
            block.ffn_norm.register_forward_hook(_post_ffn_norm)

            def _ffn_with_capture(x, _ffn=ffn, _i=i):
                x13 = _ffn.w13(x)
                cap.put(f"L{_i:02d}/09_gate_up", x13)
                from xformers.ops import RMSNorm  # noqa: F401
                # SOTA uses squared_relu(x1) * x3, then ffn_sub_norm.
                x1, x3 = x13.chunk(2, -1)
                inner = (torch.nn.functional.relu(x1) ** 2) * x3
                cap.put(f"L{_i:02d}/10_act_fn", inner)
                inner = _ffn.ffn_sub_norm(inner)
                cap.put(f"L{_i:02d}/11_ffn_sub_norm", inner)
                output = _ffn.w2(inner)
                cap.put(f"L{_i:02d}/12_down_proj", output)
                return output
            ffn.forward = _ffn_with_capture

        def _post_final_norm(_m, _i, out):
            cap.put("XX/final_norm", out)
        prefill_model.norm.register_forward_hook(_post_final_norm)

        # Set up single-batch attn_bias.
        n_tokens = len(prompt_ids)
        max_seq = max(n_tokens, 32)  # cache padding
        bias = AttnBias.from_seqlens(
            q_seqlen=[n_tokens],
            kv_seqlen=[n_tokens],
            kv_padding=max_seq,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        cache = fast.make_cache(args=args, length=max_seq)
        tokens = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")

        print("[sota] running prefill...")
        with torch.no_grad():
            logits = prefill_model.forward_with_attn_bias(
                token_values=tokens, attn_bias=bias, cache=cache,
            )
        last_logits = logits[-1].detach().to(torch.float32).cpu()
        return cap, last_logits
    finally:
        os.chdir(cwd_save)
        sys.path.remove(str(sota_gpu))


# ---------------------------------------------------------------------------
# Diff report
# ---------------------------------------------------------------------------

def report(kb: Captured, sota: Captured, layer_filter: int | None = None) -> None:
    keys = sorted(set(kb.named) & set(sota.named))
    only_kb = sorted(set(kb.named) - set(sota.named))
    only_sota = sorted(set(sota.named) - set(kb.named))
    if only_kb:
        print(f"\n[!] Only in kb-nano ({len(only_kb)}): {only_kb[:5]} ...")
    if only_sota:
        print(f"[!] Only in SOTA   ({len(only_sota)}): {only_sota[:5]} ...")

    print(f"\nComparing {len(keys)} captured tensors\n")
    print(f"{'op':46s}  {'shape':22s}  {'max|d|':>10s}  {'mean|d|':>10s}  {'cos':>8s}")
    print("-" * 110)

    # Per-head dim of BitNet b1.58-2B-4T (used to undo the NeoX/GPT-J
    # layout mismatch when comparing q/k tensors, so the diff reflects
    # actual numerical drift rather than the RoPE-convention permutation).
    HEAD_DIM = 128

    last_layer_printed = -1
    for k in keys:
        if k.startswith("L"):
            layer_idx = int(k.split("/")[0][1:])
            if layer_filter is not None and layer_idx != layer_filter:
                continue
            if layer_idx != last_layer_printed:
                print(f"--- Layer {layer_idx} ---")
                last_layer_printed = layer_idx
        a = kb.named[k]
        b = sota.named[k]
        # Re-layout kb-nano's q/k tensors from NeoX -> GPT-J before
        # comparing element-wise to SOTA, since SOTA's wqkv weight is
        # pre-permuted by ``invert_convert_q/k`` at convert time.
        op = k.split("/")[-1] if "/" in k else k
        if op in ("02_qkv_proj",):
            # Split (T, q+k+v) flat output, permute q and k slices, reassemble.
            q = a[:, :2560]
            k_ = a[:, 2560:2560 + 640]
            v = a[:, 2560 + 640:]
            a = torch.cat([
                _neox_to_gptj_per_head(q, HEAD_DIM),
                _neox_to_gptj_per_head(k_, HEAD_DIM),
                v,
            ], dim=-1)
        elif op in ("03a_q_pre_rope", "04a_q_post_rope",
                    "03b_k_pre_rope"):
            a = _neox_to_gptj_per_head(a, HEAD_DIM)
        if a.shape != b.shape:
            print(f"{k:46s}  shape mismatch kb={tuple(a.shape)} sota={tuple(b.shape)}")
            continue
        max_abs, mean_abs, cos = _diff(a, b)
        flag = ""
        if cos < 0.99:
            flag = "  <-- DIVERGED (cos<0.99)"
        elif cos < 0.999:
            flag = "  <-- drifting"
        print(f"{k:46s}  {str(tuple(a.shape)):22s}  {max_abs:10.5f}  {mean_abs:10.5f}  {cos:8.5f}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitnet-repo", default=os.environ.get("BITNET_REPO"),
                    required=os.environ.get("BITNET_REPO") is None,
                    help="Path to local clone of microsoft/BitNet")
    ap.add_argument("--prompt-len", type=int, default=16,
                    help="number of input tokens to feed (use small for quick diff)")
    ap.add_argument("--layer", type=int, default=None,
                    help="restrict diff report to this single layer index")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # Deterministic toy prompt: small ids, well within vocab.
    prompt_ids = [1] + [(i * 17 + 3) % 50000 for i in range(args.prompt_len - 1)]

    sota_cap, sota_last_logits = run_sota(prompt_ids, args.bitnet_repo)

    # Reset cuda dtype/device defaults that SOTA's setup mutated.
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)

    kb_cap, kb_last_logits = run_kb_nano(prompt_ids)

    report(kb_cap, sota_cap, layer_filter=args.layer)

    # End-to-end logits comparison: are they close enough that the same
    # argmax token is selected?
    print("\n--- Final last-token logits ---")
    max_abs, mean_abs, cos = _diff(kb_last_logits, sota_last_logits)
    print(f"max|d|={max_abs:.5f}  mean|d|={mean_abs:.5f}  cos={cos:.5f}")
    print(f"argmax kb-nano = {int(kb_last_logits.argmax())}, "
          f"argmax SOTA = {int(sota_last_logits.argmax())}")
    # Top-5 each side.
    top_kb = torch.topk(kb_last_logits, 5)
    top_so = torch.topk(sota_last_logits, 5)
    print(f"top-5 kb-nano: tokens={top_kb.indices.tolist()}, vals={[round(v, 3) for v in top_kb.values.tolist()]}")
    print(f"top-5 SOTA   : tokens={top_so.indices.tolist()}, vals={[round(v, 3) for v in top_so.values.tolist()]}")


if __name__ == "__main__":
    main()
