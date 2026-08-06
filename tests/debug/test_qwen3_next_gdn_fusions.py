#!/usr/bin/env python3
"""Exactness checks for the Qwen3-Next launch-count fusions.

Each new kernel replaces a chain of tensor ops that the eager path used to run
separately, so the bar is bitwise equality (pure data movement) or, for the
fused QK-norm/RoPE/gate kernel, agreement with the unfused reference to bf16
rounding.

Usage:  python tests/debug/test_qwen3_next_gdn_fusions.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch

from fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention import (
    _split_conv_qkv,
    _unpack_qkvz_ba,
)

DEV = "cuda"
DT = torch.bfloat16

# tp=2 shard of Qwen3-Next-80B-A3B.
H, K, V, VP = 8, 128, 128, 2
HV = H * VP


def ref_unpack(qkvz, ba):
    """The tensor-op chain the fused kernel replaces."""
    n = qkvz.shape[0]
    per_group = 2 * K + 2 * VP * V
    x = qkvz.view(n, H, per_group)
    q = x[:, :, :K]
    k = x[:, :, K:2 * K]
    v = x[:, :, 2 * K:2 * K + VP * V].reshape(n, HV, V)
    z = x[:, :, 2 * K + VP * V:].reshape(n, HV, V)
    mixed = torch.cat(
        [q.reshape(n, H * K), k.reshape(n, H * K), v.reshape(n, HV * V)], dim=-1
    )
    y = ba.view(n, H, 2 * VP)
    b = y[:, :, :VP].reshape(n, HV)
    a = y[:, :, VP:].reshape(n, HV)
    return mixed, z, b, a


def ref_split(mixed):
    n = mixed.shape[0]
    q, k, v = mixed.split([H * K, H * K, HV * V], dim=-1)
    return (
        q.view(1, n, H, K).contiguous(),
        k.view(1, n, H, K).contiguous(),
        v.view(1, n, HV, V).contiguous(),
    )


def check_unpack():
    ok = True
    for n in (1, 2, 7, 16, 32, 33, 596, 4096, 16384):
        torch.manual_seed(n)
        qkvz = torch.randn(n, H * (2 * K + 2 * VP * V), device=DEV, dtype=DT)
        ba = torch.randn(n, H * 2 * VP, device=DEV, dtype=DT)
        got = _unpack_qkvz_ba(qkvz, ba, H, K, V, VP)
        exp = ref_unpack(qkvz, ba)
        for name, g, e in zip(("mixed_qkv", "z", "b", "a"), got, exp):
            same = torch.equal(g, e)
            ok &= same
            if not same:
                print(f"  FAIL unpack n={n} {name}: "
                      f"maxdiff={(g.float() - e.float()).abs().max().item()}")
        assert all(t.is_contiguous() for t in got), f"non-contiguous at n={n}"
    print(f"  unpack qkvz/ba: {'PASS' if ok else 'FAIL'} (bitwise)")
    return ok


def check_split():
    ok = True
    for n in (1, 2, 7, 16, 32, 33, 596, 4096, 16384):
        torch.manual_seed(1000 + n)
        mixed = torch.randn(n, 2 * H * K + HV * V, device=DEV, dtype=DT)
        got = _split_conv_qkv(mixed, H, K, HV, V)
        exp = ref_split(mixed)
        for name, g, e in zip(("q", "k", "v"), got, exp):
            same = torch.equal(g, e)
            ok &= same
            if not same:
                print(f"  FAIL split n={n} {name}: "
                      f"maxdiff={(g.float() - e.float()).abs().max().item()}")
    print(f"  split conv qkv : {'PASS' if ok else 'FAIL'} (bitwise)")
    return ok


def check_qk_norm_rope_gate():
    """Fused split+QK-norm+partial RoPE+gate vs the unfused module chain."""
    from vllm.model_executor.layers.fused_qk_norm_rope import (
        fused_qk_rmsnorm_rope_gate,
    )

    from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
    from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding

    nh, nkv, hd, rot = 8, 1, 256, 64
    eps = 1e-6
    rope = RotaryEmbedding(rot, 4096, 10_000_000.0).to(DEV)
    q_norm = GemmaRMSNorm(hd, eps=eps).to(DEV)
    k_norm = GemmaRMSNorm(hd, eps=eps).to(DEV)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn(hd, device=DEV) * 0.05)
        k_norm.weight.copy_(torch.randn(hd, device=DEV) * 0.05)

    ok = True
    for n in (1, 4, 32, 596):
        torch.manual_seed(n)
        q_gate = torch.randn(n, nh * 2 * hd, device=DEV, dtype=DT)
        k = torch.randn(n, nkv * hd, device=DEV, dtype=DT)
        pos = torch.randint(0, 4000, (n,), device=DEV, dtype=torch.int64)

        # reference: the previous eager chain
        qg = q_gate.view(n, nh, 2 * hd)
        rq = qg[:, :, :hd].contiguous()
        rgate = qg[:, :, hd:].contiguous()
        rk = k.view(n, nkv, hd)
        rq = q_norm(rq.reshape(-1, hd)).view(n, nh, hd)
        rk = k_norm(rk.reshape(-1, hd)).view(n, nkv, hd)
        q_rot, q_pass = rq[..., :rot].contiguous(), rq[..., rot:]
        k_rot, k_pass = rk[..., :rot].contiguous(), rk[..., rot:]
        q_rot, k_rot = rope(pos, q_rot, k_rot)
        rq = torch.cat([q_rot, q_pass], dim=-1)
        rk = torch.cat([k_rot, k_pass], dim=-1)

        fq, fk, fgate = fused_qk_rmsnorm_rope_gate(
            q_gate, k,
            q_norm.weight.float() + 1.0, k_norm.weight.float() + 1.0,
            rope.cos_sin_cache, pos, eps, nh, nkv, hd, rot,
        )
        for name, g, e in (
            ("q", fq.view(n, nh, hd), rq),
            ("k", fk.view(n, nkv, hd), rk),
            ("gate", fgate.view(n, nh, hd), rgate),
        ):
            d = (g.float() - e.float()).abs().max().item()
            scale = max(e.float().abs().max().item(), 1e-6)
            good = d <= 4e-2 * scale
            ok &= good
            print(f"    n={n:<5} {name:<4} maxdiff={d:.3e} rel={d / scale:.2e} "
                  f"{'ok' if good else 'FAIL'}")
    print(f"  qk norm+rope+gate: {'PASS' if ok else 'FAIL'} (bf16 tolerance)")
    return ok


def check_moe_shared_gate_add():
    """Fused MoE epilogue vs the three-kernel chain it replaces.

    Not bitwise: the fused form keeps ``shared * sigmoid(gate)`` in fp32 before
    the add instead of rounding it to bf16 in between, which is what Inductor
    does for vLLM. Difference should sit at one bf16 ulp. The second half also
    checks the variant that projects the gate inside the kernel instead of
    taking it from a gemv.
    """
    from fastkernels.tasks.baseline.L1.moe_shared_gate_add import (
        moe_shared_gate_add,
    )

    ok = True
    for n, h in ((1, 2048), (32, 2048), (596, 2048), (16384, 2048)):
        torch.manual_seed(n)
        routed = torch.randn(n, h, device=DEV, dtype=DT)
        shared = torch.randn(n, h, device=DEV, dtype=DT)
        gate = torch.randn(n, 1, device=DEV, dtype=DT)
        exp = routed + shared * torch.sigmoid(gate)
        got = moe_shared_gate_add(routed, shared, gate)
        d = (got.float() - exp.float()).abs().max().item()
        scale = max(exp.float().abs().max().item(), 1e-6)
        good = d <= 1e-2 * scale
        ok &= good
        print(f"    gate given   n={n:<6} maxdiff={d:.3e} rel={d / scale:.2e} "
              f"{'ok' if good else 'FAIL'}")

    # Fused projection: kernel computes gate = hidden . w itself.
    for n, h in ((1, 2048), (32, 2048), (256, 2048)):
        torch.manual_seed(5000 + n)
        routed = torch.randn(n, h, device=DEV, dtype=DT)
        shared = torch.randn(n, h, device=DEV, dtype=DT)
        hid = torch.randn(n, h, device=DEV, dtype=DT)
        w = torch.randn(1, h, device=DEV, dtype=DT) * 0.02
        ref_gate = torch.nn.functional.linear(hid, w)
        exp = routed + shared * torch.sigmoid(ref_gate)
        got = moe_shared_gate_add(routed, shared, hidden_states=hid,
                                  gate_weight=w)
        d = (got.float() - exp.float()).abs().max().item()
        scale = max(exp.float().abs().max().item(), 1e-6)
        good = d <= 2e-2 * scale
        ok &= good
        print(f"    gate fused   n={n:<6} maxdiff={d:.3e} rel={d / scale:.2e} "
              f"{'ok' if good else 'FAIL'}")

    # Strided gate column (what a merged projection would hand us).
    torch.manual_seed(77)
    n, h = 32, 2048
    routed = torch.randn(n, h, device=DEV, dtype=DT)
    shared = torch.randn(n, h, device=DEV, dtype=DT)
    wide = torch.randn(n, 5, device=DEV, dtype=DT)
    col = wide[:, 2:3]
    exp = routed + shared * torch.sigmoid(col)
    got = moe_shared_gate_add(routed, shared, col)
    d = (got.float() - exp.float()).abs().max().item()
    scale = max(exp.float().abs().max().item(), 1e-6)
    good = d <= 1e-2 * scale
    ok &= good
    print(f"    strided gate n={n:<6} maxdiff={d:.3e} rel={d / scale:.2e} "
          f"{'ok' if good else 'FAIL'}")
    print(f"  moe shared gate add: {'PASS' if ok else 'FAIL'} (bf16 tolerance)")
    return ok


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    print("\n  Qwen3-Next fusion exactness checks\n")
    results = [
        check_unpack(),
        check_split(),
        check_qk_norm_rope_gate(),
        check_moe_shared_gate_add(),
    ]
    print(f"\n  {'ALL PASS' if all(results) else 'FAILURES PRESENT'}\n")
    sys.exit(0 if all(results) else 1)
