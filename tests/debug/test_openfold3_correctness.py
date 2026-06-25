#!/usr/bin/env python3
"""
Module-level correctness tests: fastkernels OpenFold3 vs reference openfold3.

For each key module (L1-L3), we:
1. Instantiate the reference openfold3 module
2. Randomize all weights (since default AF3 init zeros out output projections)
3. Instantiate the fastkernels module and load the same weights
4. Run identical inputs through both
5. Compare outputs via cosine similarity and max absolute difference

Usage:
    CUDA_VISIBLE_DEVICES=7 python tests/test_openfold3_correctness.py
    CUDA_VISIBLE_DEVICES=7 python tests/test_openfold3_correctness.py --module triangle_mul
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent

sys.path.insert(0, str(_PACKAGE_DIR))


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).double()
    b_flat = b.reshape(-1).double()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-12)).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def report(name: str, cos: float, mad: float, threshold: float = 0.999):
    status = "PASS" if cos >= threshold else "FAIL"
    print(f"  [{status}] {name}: cosine={cos:.8f}  max_abs_diff={mad:.6e}  (threshold={threshold})")
    return cos >= threshold


def randomize_weights(module: torch.nn.Module):
    """Fill all parameters with small random values to avoid zero-init dead outputs."""
    for p in module.parameters():
        p.data.normal_(0, 0.02)


# ---------------------------------------------------------------------------
# Test: TriangleMultiplicativeUpdate
# ---------------------------------------------------------------------------

def test_triangle_multiplication(device="cuda", dtype=torch.bfloat16):
    print("\n=== TriangleMultiplicativeUpdate ===")
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationOutgoing as RefTriMulOut,
    )
    from fastkernels.tasks.baseline.L2.alphafold3_triangle_multiplication import (
        TriangleMultiplicationOutgoing as KBTriMulOut,
    )

    c_z, c_hidden = 128, 128
    N = 48

    ref = RefTriMulOut(c_z, c_hidden).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBTriMulOut(c_z, c_hidden).to(device=device, dtype=dtype).eval()
    kb.load_state_dict(ref.state_dict())

    z = torch.randn(1, N, N, c_z, device=device, dtype=dtype)
    mask = torch.ones(1, N, N, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(z.clone(), mask=mask)
        kb_out = kb(z.clone(), mask=mask)

    cos = cosine_sim(ref_out, kb_out)
    mad = max_abs_diff(ref_out, kb_out)
    return report("TriMulOut", cos, mad)


# ---------------------------------------------------------------------------
# Test: TriangleAttention
# ---------------------------------------------------------------------------

def test_triangle_attention(device="cuda", dtype=torch.bfloat16):
    print("\n=== TriangleAttention ===")
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttention as RefTriAtt,
    )
    from fastkernels.tasks.baseline.L2.alphafold3_triangle_attention import (
        TriangleAttention as KBTriAtt,
    )

    c_in, c_hidden, no_heads = 128, 32, 4
    N = 32

    ref = RefTriAtt(c_in, c_hidden, no_heads, starting=True).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBTriAtt(c_in, c_hidden, no_heads, starting=True).to(device=device, dtype=dtype).eval()
    kb.load_state_dict(ref.state_dict())

    x = torch.randn(1, N, N, c_in, device=device, dtype=dtype)
    mask = torch.ones(1, N, N, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(x.clone(), mask=mask)
        kb_out = kb(x.clone(), mask=mask)

    cos = cosine_sim(ref_out, kb_out)
    mad = max_abs_diff(ref_out, kb_out)
    return report("TriAttStart", cos, mad)


# ---------------------------------------------------------------------------
# Test: OuterProductMean
# ---------------------------------------------------------------------------

def test_outer_product_mean(device="cuda", dtype=torch.bfloat16):
    print("\n=== OuterProductMean ===")
    from openfold3.core.model.layers.outer_product_mean import (
        OuterProductMean as RefOPM,
    )
    from fastkernels.tasks.baseline.L2.alphafold3_outer_product_mean import (
        OuterProductMean as KBOPM,
    )

    c_m, c_z, c_hidden = 64, 128, 32
    N_seq, N_res = 16, 32

    ref = RefOPM(c_m, c_z, c_hidden).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBOPM(c_m, c_z, c_hidden).to(device=device, dtype=dtype).eval()
    kb.load_state_dict(ref.state_dict())

    m = torch.randn(1, N_seq, N_res, c_m, device=device, dtype=dtype)
    mask = torch.ones(1, N_seq, N_res, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(m.clone(), mask=mask)
        kb_out = kb(m.clone(), mask=mask)

    cos = cosine_sim(ref_out, kb_out)
    mad = max_abs_diff(ref_out, kb_out)
    return report("OPM", cos, mad)


# ---------------------------------------------------------------------------
# Test: SwiGLU
# ---------------------------------------------------------------------------

def test_swiglu(device="cuda", dtype=torch.bfloat16):
    print("\n=== SwiGLU ===")
    from openfold3.core.model.primitives.activations import SwiGLU as RefSwiGLU
    from fastkernels.tasks.baseline.L2.alphafold3_swiglu import SwiGLU as KBSwiGLU

    c_in, c_out = 384, 1536

    ref = RefSwiGLU(c_in, c_out).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBSwiGLU(c_in, c_out).to(device=device, dtype=dtype).eval()
    kb.load_state_dict(ref.state_dict())

    x = torch.randn(1, 48, c_in, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(x.clone())
        kb_out = kb(x.clone())

    cos = cosine_sim(ref_out, kb_out)
    mad = max_abs_diff(ref_out, kb_out)
    return report("SwiGLU", cos, mad, threshold=0.9999)


# ---------------------------------------------------------------------------
# Test: SwiGLUTransition
# ---------------------------------------------------------------------------

def test_swiglu_transition(device="cuda", dtype=torch.bfloat16):
    print("\n=== SwiGLUTransition ===")
    from openfold3.core.model.layers.transition import SwiGLUTransition as RefTrans
    from fastkernels.tasks.baseline.L2.alphafold3_swiglu_transition import SwiGLUTransition as KBTrans

    c_in, n = 128, 4

    ref = RefTrans(c_in, n).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBTrans(c_in, n).to(device=device, dtype=dtype).eval()
    kb.load_state_dict(ref.state_dict())

    x = torch.randn(1, 48, c_in, device=device, dtype=dtype)
    mask = torch.ones(1, 48, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(x.clone(), mask=mask)
        kb_out = kb(x.clone(), mask=mask)

    cos = cosine_sim(ref_out, kb_out)
    mad = max_abs_diff(ref_out, kb_out)
    return report("SwiGLUTransition", cos, mad)


# ---------------------------------------------------------------------------
# Test: Attention
# ---------------------------------------------------------------------------

def test_attention(device="cuda", dtype=torch.bfloat16):
    print("\n=== Attention (MHA with biases) ===")
    from openfold3.core.model.primitives.attention import Attention as RefAttn
    from fastkernels.tasks.baseline.L2.alphafold3_of3_attention import OF3Attention as KBAttn

    c_q, c_hidden, no_heads = 384, 32, 16
    N = 48

    ref = RefAttn(c_q=c_q, c_k=c_q, c_v=c_q, c_hidden=c_hidden, no_heads=no_heads).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBAttn(c_q=c_q, c_k=c_q, c_v=c_q, c_hidden=c_hidden, no_heads=no_heads).to(device=device, dtype=dtype).eval()
    kb.load_state_dict(ref.state_dict())

    q_x = torch.randn(1, N, c_q, device=device, dtype=dtype)
    mask_bias = torch.zeros(1, 1, 1, N, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(q_x.clone(), q_x.clone(), biases=[mask_bias])
        kb_out = kb(q_x.clone(), q_x.clone(), biases=[mask_bias])

    cos = cosine_sim(ref_out, kb_out)
    mad = max_abs_diff(ref_out, kb_out)
    return report("Attention", cos, mad)


# ---------------------------------------------------------------------------
# Test: PairBlock (L2)
# ---------------------------------------------------------------------------

def test_pair_block(device="cuda", dtype=torch.bfloat16):
    print("\n=== PairBlock (L2) ===")
    from openfold3.core.model.latent.base_blocks import PairBlock as RefPairBlock
    from fastkernels.tasks.baseline.L2.alphafold3_pair_block import PairBlock as KBPairBlock

    c_z = 128
    N = 24

    ref = RefPairBlock(
        c_z=c_z, c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
        transition_type="swiglu", transition_n=4, pair_dropout=0.0,
        fuse_projection_weights=False, inf=1e9,
    ).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBPairBlock(
        c_z=c_z, c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
        transition_n=4, pair_dropout=0.0, inf=1e9,
    ).to(device=device, dtype=dtype).eval()

    # State dict key mapping between ref and kb
    ref_sd = ref.state_dict()
    kb_sd = kb.state_dict()

    # Check and report key differences
    ref_keys = set(ref_sd.keys())
    kb_keys = set(kb_sd.keys())
    unmatched_ref = ref_keys - kb_keys
    unmatched_kb = kb_keys - ref_keys
    if unmatched_ref:
        print(f"  WARNING: {len(unmatched_ref)} ref keys not in kb: {list(unmatched_ref)[:5]}...")
    if unmatched_kb:
        print(f"  WARNING: {len(unmatched_kb)} kb keys not in ref: {list(unmatched_kb)[:5]}...")

    common_keys = ref_keys & kb_keys
    for key in common_keys:
        if ref_sd[key].shape == kb_sd[key].shape:
            kb_sd[key] = ref_sd[key]
    kb.load_state_dict(kb_sd)

    z = torch.randn(1, N, N, c_z, device=device, dtype=dtype)
    pair_mask = torch.ones(1, N, N, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(z.clone(), pair_mask=pair_mask)
        kb_out = kb(z.clone(), pair_mask=pair_mask)

    cos = cosine_sim(ref_out, kb_out)
    mad = max_abs_diff(ref_out, kb_out)
    return report("PairBlock", cos, mad, threshold=0.99)


# ---------------------------------------------------------------------------
# Test: PairFormerBlock (L3)
# ---------------------------------------------------------------------------

def test_pairformer_block(device="cuda", dtype=torch.bfloat16):
    print("\n=== PairFormerBlock (L3) ===")
    from openfold3.core.model.latent.pairformer import PairFormerBlock as RefPFBlock
    from fastkernels.tasks.baseline.L3.alphafold3_pairformer import PairFormerBlock as KBPFBlock

    c_s, c_z = 384, 128
    N = 24

    ref = RefPFBlock(
        c_s=c_s, c_z=c_z, c_hidden_pair_bias=24, no_heads_pair_bias=16,
        c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
        transition_type="swiglu", transition_n=4, pair_dropout=0.0,
        fuse_projection_weights=False, inf=1e9,
    ).to(device=device, dtype=dtype).eval()
    randomize_weights(ref)

    kb = KBPFBlock(
        c_s=c_s, c_z=c_z, c_hidden_pair_bias=24, no_heads_pair_bias=16,
        c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
        transition_n=4, pair_dropout=0.0, inf=1e9,
    ).to(device=device, dtype=dtype).eval()

    ref_sd = ref.state_dict()
    kb_sd = kb.state_dict()
    ref_keys = set(ref_sd.keys())
    kb_keys = set(kb_sd.keys())
    unmatched_ref = ref_keys - kb_keys
    unmatched_kb = kb_keys - ref_keys
    if unmatched_ref:
        print(f"  WARNING: {len(unmatched_ref)} ref keys not in kb: {sorted(unmatched_ref)[:10]}...")
    if unmatched_kb:
        print(f"  WARNING: {len(unmatched_kb)} kb keys not in ref: {sorted(unmatched_kb)[:10]}...")

    for key in ref_keys & kb_keys:
        if ref_sd[key].shape == kb_sd[key].shape:
            kb_sd[key] = ref_sd[key]
    kb.load_state_dict(kb_sd)

    s = torch.randn(1, N, c_s, device=device, dtype=dtype)
    z = torch.randn(1, N, N, c_z, device=device, dtype=dtype)
    single_mask = torch.ones(1, N, device=device, dtype=dtype)
    pair_mask = torch.ones(1, N, N, device=device, dtype=dtype)

    with torch.no_grad():
        ref_s, ref_z = ref(s.clone(), z.clone(), single_mask=single_mask, pair_mask=pair_mask)
        kb_s, kb_z = kb(s.clone(), z.clone(), single_mask=single_mask, pair_mask=pair_mask)

    cos_s = cosine_sim(ref_s, kb_s)
    mad_s = max_abs_diff(ref_s, kb_s)
    cos_z = cosine_sim(ref_z, kb_z)
    mad_z = max_abs_diff(ref_z, kb_z)

    p1 = report("PairFormerBlock.s", cos_s, mad_s, threshold=0.99)
    p2 = report("PairFormerBlock.z", cos_z, mad_z, threshold=0.99)
    return p1 and p2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "triangle_mul": test_triangle_multiplication,
    "triangle_att": test_triangle_attention,
    "opm": test_outer_product_mean,
    "swiglu": test_swiglu,
    "swiglu_transition": test_swiglu_transition,
    "attention": test_attention,
    "pair_block": test_pair_block,
    "pairformer_block": test_pairformer_block,
}


def main():
    parser = argparse.ArgumentParser(description="OpenFold3 correctness tests")
    parser.add_argument("--module", type=str, default=None,
                        help="Run specific module test (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    torch.manual_seed(42)

    tests = ALL_TESTS if args.module is None else {args.module: ALL_TESTS[args.module]}
    results = {}

    for name, test_fn in tests.items():
        try:
            passed = test_fn(device=device, dtype=dtype)
            results[name] = passed
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {name}: {e}")
            results[name] = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    passed = sum(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n  {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
