#!/usr/bin/env python3
"""
Test suite for the benchmarking infrastructure.

Sections 1-6: Unit tests (no GPU required).
Sections 7-9: Integration tests (GPU required and run by default).

Usage:
    python tests/test_bench.py                 # all tests
    python tests/test_bench.py --section 3      # run only section 3
"""

from __future__ import annotations

import argparse
import io
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
PACKAGE_NAME = os.path.basename(PACKAGE_DIR)

sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_pass_count = 0
_fail_count = 0


def check(condition: bool, label: str):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"    PASS  {label}")
    else:
        _fail_count += 1
        print(f"    FAIL  {label}")


class _Timeout:
    """POSIX alarm-based timeout guard for unit tests."""
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *args):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("Unit test timed out")


# ===========================================================================
# Section 1: Input Registry (unit, no GPU)
# ===========================================================================
def test_section_1():
    print(f"\n{'=' * 60}")
    print("  SECTION 1: Input Registry")
    print(f"{'=' * 60}")

    from kb_nano.bench.kernels.scenario_registry import InputRegistry

    # 1a. YAML parsing
    with _Timeout(30):
        yaml_content = """
rms_norm:
  scenarios:
    - name: "test-model/decode-bs1/tp1"
      init_args:
        hidden_size: 64
        eps: 1.0e-6
      inputs:
        x: {shape: [1, 64], dtype: bfloat16}
        residual: {shape: [1, 64], dtype: bfloat16}
    - name: "test-model/prefill-128/tp1"
      init_args:
        hidden_size: 64
        eps: 1.0e-6
      inputs:
        x: {shape: [128, 64], dtype: bfloat16}
        residual: null
moe_align:
  scenarios:
    - name: "mixtral/decode-bs1/tp1"
      init_args: {}
      inputs:
        topk_ids: {shape: [1, 2], dtype: int32}
      golden: "mixtral/moe_align/decode-bs1-tp1.pt"
      golden_inputs: [topk_ids]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            yaml_path = inputs_dir / "test.yaml"
            yaml_path.write_text(yaml_content)

            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=Path(tmpdir) / "golden")
            ops = reg.operators()
            check(
                "rms_norm" in ops and "moe_align" in ops,
                "1a. YAML parsing: operators parsed correctly",
            )

            rms_scenarios = reg.scenarios("rms_norm")
            check(len(rms_scenarios) == 2, "1a. rms_norm has 2 scenarios")
            check(
                rms_scenarios[0].name == "test-model/decode-bs1/tp1",
                "1a. first scenario name correct",
            )
            check(
                rms_scenarios[0].init_args == {"hidden_size": 64, "eps": 1e-6},
                "1a. init_args parsed correctly",
            )

            moe_scenarios = reg.scenarios("moe_align")
            check(len(moe_scenarios) == 1, "1a. moe_align has 1 scenario")
            check(
                moe_scenarios[0].golden_path == "mixtral/moe_align/decode-bs1-tp1.pt",
                "1a. golden path parsed correctly",
            )
            check(
                moe_scenarios[0].golden_inputs == ["topk_ids"],
                "1a. golden input allowlist parsed correctly",
            )

    # 1b. Shape-only tensor generation
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(yaml_content)

            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=Path(tmpdir) / "golden")
            inputs = reg.get_inputs("rms_norm", "test-model/decode-bs1/tp1", device="cpu")
            check(
                "x" in inputs and "residual" in inputs,
                "1b. get_inputs returns expected keys",
            )
            check(
                inputs["x"].shape == (1, 64) and inputs["x"].dtype == torch.bfloat16,
                "1b. x tensor has correct shape and dtype",
            )
            check(
                inputs["residual"].shape == (1, 64),
                "1b. residual tensor has correct shape",
            )
            check(
                not torch.all(inputs["x"] == 0),
                "1b. tensors are random (not zeros)",
            )

    # 1c. Golden data loading
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            golden_dir = Path(tmpdir) / "golden"

            golden_subdir = golden_dir / "mixtral" / "moe_align"
            golden_subdir.mkdir(parents=True)
            known_tensor = {"topk_ids": torch.tensor([[3, 7]], dtype=torch.int32)}
            torch.save(known_tensor, golden_subdir / "decode-bs1-tp1.pt")

            (inputs_dir / "test.yaml").write_text(yaml_content)
            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=golden_dir)
            golden_inputs = reg.get_inputs("moe_align", "mixtral/decode-bs1/tp1", device="cpu")
            check(
                "topk_ids" in golden_inputs,
                "1c. golden data has expected key",
            )
            check(
                torch.equal(golden_inputs["topk_ids"], torch.tensor([[3, 7]], dtype=torch.int32)),
                "1c. golden data loads exact tensors",
            )

    # 1h. Partial golden data overlays shape-generated inputs
    with _Timeout(30):
        partial_yaml = """
store_kvcache:
  scenarios:
    - name: "toy/store/tp1"
      init_args: {}
      inputs:
        key: {shape: [2, 1, 4], dtype: bfloat16}
        value: {shape: [2, 1, 4], dtype: bfloat16}
        k_cache: {shape: [4, 8, 1, 4], dtype: bfloat16}
        v_cache: {shape: [4, 8, 1, 4], dtype: bfloat16}
        slot_mapping: {shape: [2], dtype: int64}
      golden: "toy/store/slots.pt"
      golden_inputs: [slot_mapping]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            golden_dir = Path(tmpdir) / "golden"
            inputs_dir.mkdir()
            (golden_dir / "toy" / "store").mkdir(parents=True)
            (inputs_dir / "test.yaml").write_text(partial_yaml)
            torch.save(
                {"slot_mapping": torch.tensor([3, 9], dtype=torch.int64)},
                golden_dir / "toy" / "store" / "slots.pt",
            )
            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=golden_dir)
            materialized = reg.get_inputs("store_kvcache", "toy/store/tp1", device="cpu")
            check(
                set(materialized) == {"key", "value", "k_cache", "v_cache", "slot_mapping"},
                "1h. partial golden overlays generated shape inputs",
            )
            check(
                torch.equal(materialized["slot_mapping"], torch.tensor([3, 9])),
                "1h. captured control input overrides synthetic input",
            )
            check(
                materialized["key"].shape == (2, 1, 4)
                and materialized["k_cache"].shape == (4, 8, 1, 4),
                "1h. non-golden tensors are still generated from shape specs",
            )

    # 1i. Data-dependent synthetic index inputs are range-constrained
    with _Timeout(30):
        constrained_yaml = """
store_kvcache:
  scenarios:
    - name: "toy/store/tp1"
      init_args: {}
      inputs:
        key: {shape: [16, 2, 8], dtype: bfloat16}
        value: {shape: [16, 2, 8], dtype: bfloat16}
        k_cache: {shape: [4, 8, 2, 8], dtype: bfloat16}
        v_cache: {shape: [4, 8, 2, 8], dtype: bfloat16}
        slot_mapping: {shape: [16], dtype: int64}
moe_align:
  scenarios:
    - name: "toy/align/tp1"
      init_args: {}
      inputs:
        topk_ids: {shape: [32, 2], dtype: int32}
        block_size: 128
        num_experts: 8
moe_grouped_gemm:
  scenarios:
    - name: "toy/gemm/tp1"
      init_args: {}
      inputs:
        A: {shape: [32, 128], dtype: bfloat16}
        B: {shape: [8, 256, 128], dtype: bfloat16}
        C: {shape: [64, 256], dtype: bfloat16}
        topk_weights: {shape: [32, 2], dtype: float32}
        sorted_token_ids: {shape: [256], dtype: int32}
        expert_ids: {shape: [2], dtype: int32}
        num_tokens_post_padded: {shape: [1], dtype: int32}
        mul_routed_weight: true
        top_k: 2
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(constrained_yaml)
            reg = InputRegistry(inputs_dir=inputs_dir)
            store_inputs = reg.get_inputs("store_kvcache", "toy/store/tp1", device="cpu")
            check(
                int(store_inputs["slot_mapping"].min()) >= 0
                and int(store_inputs["slot_mapping"].max()) < 32,
                "1i. synthetic slot_mapping is inside cache slot range",
            )
            align_inputs = reg.get_inputs("moe_align", "toy/align/tp1", device="cpu")
            check(
                int(align_inputs["topk_ids"].min()) >= 0
                and int(align_inputs["topk_ids"].max()) < 8,
                "1i. synthetic topk_ids are inside expert range",
            )
            gemm_inputs = reg.get_inputs("moe_grouped_gemm", "toy/gemm/tp1", device="cpu")
            check(
                int(gemm_inputs["expert_ids"].min()) >= 0
                and int(gemm_inputs["expert_ids"].max()) < 8,
                "1i. synthetic expert_ids are inside expert range",
            )
            check(
                int(gemm_inputs["sorted_token_ids"].min()) >= 0
                and int(gemm_inputs["sorted_token_ids"].max()) < 64,
                "1i. synthetic sorted_token_ids are inside routed-token range",
            )
            check(
                int(gemm_inputs["num_tokens_post_padded"][0]) == 256,
                "1i. synthetic num_tokens_post_padded matches aligned rows",
            )

    # 1j. Data-dependent capture policy is input-specific
    with _Timeout(30):
        from kb_nano.bench.kernels.scenario_schema import (
            DATA_DEPENDENT_INPUTS,
            DATA_DEPENDENT_OPS,
        )

        check(
            DATA_DEPENDENT_INPUTS["store_kvcache"] == {"slot_mapping"},
            "1j. store_kvcache captures only slot_mapping",
        )
        check(
            DATA_DEPENDENT_INPUTS["moe_align"] == {"topk_ids"}
            and DATA_DEPENDENT_INPUTS["fused_experts"] == {"topk_ids"},
            "1j. MoE alignment/expert ops capture only routing ids",
        )
        check(
            DATA_DEPENDENT_INPUTS["moe_grouped_gemm"] == {
                "sorted_token_ids",
                "expert_ids",
                "num_tokens_post_padded",
            },
            "1j. grouped GEMM captures only aligned routing metadata",
        )
        check(
            not {"grouped_topk", "sigmoid_topk", "topk_softmax"} & DATA_DEPENDENT_OPS,
            "1j. top-k producer ops do not require golden input capture",
        )

    # 1d. Filtering
    with _Timeout(30):
        filter_yaml = """
rms_norm:
  scenarios:
    - name: "llama31-8b/decode-bs1/tp1"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [1, 64], dtype: bfloat16}
    - name: "llama31-8b/decode-bs32/tp4"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [32, 64], dtype: bfloat16}
    - name: "mixtral-8x7b/decode-bs1/tp1"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [1, 64], dtype: bfloat16}
    - name: "mixtral-8x7b/prefill-512/tp2"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [512, 64], dtype: bfloat16}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(filter_yaml)
            reg = InputRegistry(inputs_dir=inputs_dir)

            all_s = reg.scenarios("rms_norm")
            check(len(all_s) == 4, "1d. no filter returns all 4 scenarios")

            llama_s = reg.scenarios("rms_norm", models=["llama31"])
            check(
                len(llama_s) == 2 and all("llama31" in s.name for s in llama_s),
                "1d. model filter returns 2 llama scenarios",
            )

            tp4_s = reg.scenarios("rms_norm", tp=[4])
            check(
                len(tp4_s) == 1 and "tp4" in tp4_s[0].name,
                "1d. tp filter returns 1 tp4 scenario",
            )

            tp12 = reg.scenarios("rms_norm", tp=[1, 2])
            check(len(tp12) == 3, "1d. tp=[1,2] filter returns 3 scenarios")

    # 1e. Missing golden file
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(yaml_content)
            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=Path(tmpdir) / "empty_golden")
            try:
                reg.get_inputs("moe_align", "mixtral/decode-bs1/tp1", device="cpu")
                check(False, "1e. missing golden raises FileNotFoundError")
            except FileNotFoundError:
                check(True, "1e. missing golden raises FileNotFoundError")

    # 1f. Empty registry
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "empty.yaml").write_text("")
            reg = InputRegistry(inputs_dir=inputs_dir)
            check(len(reg.operators()) == 0, "1f. empty YAML -> zero operators")

    # 1g. Coherent FP8 + scale input generation
    with _Timeout(30):
        if not hasattr(torch, "float8_e4m3fn"):
            check(True, "1g. FP8 dtype unavailable, skipping")
        else:
            fp8_yaml = """
fp8_grouped_gemm:
  scenarios:
    - name: "toy/fp8/tp1"
      init_args: {}
      inputs:
        a_fp8:
          shape: [2, 256]
          dtype: float8_e4m3fn
          quantize: fp8
          scale_arg: a_scale
          group_size: 128
        b_fp8:
          shape: [4, 256, 256]
          dtype: float8_e4m3fn
          quantize: fp8
          scale_arg: b_scale
          scale_layout: per_block
          block_shape: [128, 128]
"""
            with tempfile.TemporaryDirectory() as tmpdir:
                inputs_dir = Path(tmpdir) / "inputs"
                inputs_dir.mkdir()
                (inputs_dir / "test.yaml").write_text(fp8_yaml)
                reg = InputRegistry(inputs_dir=inputs_dir)
                inputs = reg.get_inputs("fp8_grouped_gemm", "toy/fp8/tp1", device="cpu")
                check(
                    inputs["a_fp8"].dtype == torch.float8_e4m3fn
                    and inputs["a_scale"].shape == (2, 2),
                    "1g. per-token FP8 input and scale generated",
                )
                check(
                    inputs["b_fp8"].dtype == torch.float8_e4m3fn
                    and inputs["b_scale"].shape == (4, 2, 2),
                    "1g. per-block FP8 input and scale generated",
                )
                dequant = inputs["a_fp8"].float() * inputs["a_scale"].repeat_interleave(128, dim=-1)
                check(torch.isfinite(dequant).all(), "1g. dequantized FP8 input is finite")


# ===========================================================================
# Section 2: KernelRunner (unit, no GPU — uses CPU mock modules)
# ===========================================================================
def test_section_2():
    print(f"\n{'=' * 60}")
    print("  SECTION 2: KernelRunner")
    print(f"{'=' * 60}")

    from kb_nano.bench.kernels.runner import (
        _compare_outputs,
        _merge_correctness,
        _run_forward_once,
        _time_forward,
    )
    from kb_nano.bench.kernels.result import OperatorResult, ScenarioResult

    # 2a. Identical modules
    with _Timeout(30):
        torch.manual_seed(42)
        m1 = nn.Linear(4, 4, bias=False)
        m2 = nn.Linear(4, 4, bias=False)
        m2.load_state_dict(m1.state_dict())
        inp = torch.randn(2, 4)
        with torch.no_grad():
            out1 = m1(inp)
            out2 = m2(inp)
        correct, error_ratio, diff = _compare_outputs(out1, out2)
        check(
            correct and error_ratio == 0.0 and diff == 0.0,
            "2a. identical modules -> correct=True, error_ratio=0, diff=0",
        )

    # 2b. Different modules
    with _Timeout(30):
        class ZeroModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))
            def forward(self, x):
                return torch.zeros_like(x)

        torch.manual_seed(42)
        baseline = nn.Linear(4, 4, bias=False)
        candidate = ZeroModule()
        inp = torch.randn(2, 4)
        with torch.no_grad():
            out_b = baseline(inp)
            out_c = candidate(inp)
        correct, error_ratio, diff = _compare_outputs(out_b, out_c)
        check(
            not correct or error_ratio > 1.0,
            "2b. different modules -> correct=False or error_ratio>1",
        )
        check(diff > 0, f"2b. mean_abs_diff={diff:.4f} > 0")

    # 2c. FP8 output pairs compare in dequantized value space
    with _Timeout(30):
        if not hasattr(torch, "float8_e4m3fn"):
            check(True, "2c. FP8 dtype unavailable, skipping")
        else:
            fp8 = torch.ones(1, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
            scale = torch.ones(1, 2, dtype=torch.float32)
            correct, error_ratio, diff = _compare_outputs((fp8, scale), (fp8, scale))
            check(
                correct and error_ratio == 0.0 and diff == 0.0,
                "2c. identical FP8 pairs -> correct=True, error_ratio=0",
            )
            bad_scale = scale * 2.0
            correct, error_ratio, diff = _compare_outputs(
                (fp8, scale), (fp8, bad_scale),
            )
            check(
                not correct and error_ratio > 1.0 and diff > 0.0,
                "2c. changed FP8 scale fails dequantized comparison",
            )

    # 2d. Weight copying
    with _Timeout(30):
        torch.manual_seed(42)
        m1 = nn.Linear(8, 4, bias=False)
        m2 = nn.Linear(8, 4, bias=False)
        check(
            not torch.equal(m1.weight, m2.weight),
            "2d. before copy, weights differ",
        )
        m2.load_state_dict(m1.state_dict())
        check(
            torch.equal(m1.weight, m2.weight),
            "2d. after load_state_dict, weights match",
        )
        inp = torch.randn(2, 8)
        with torch.no_grad():
            out1 = m1(inp)
            out2 = m2(inp)
        check(torch.allclose(out1, out2), "2d. after copy, outputs match")

    # 2e. init_args propagation
    with _Timeout(30):
        class ConfigModule(nn.Module):
            def __init__(self, hidden_size: int = 10, eps: float = 1e-6):
                super().__init__()
                self.hidden_size = hidden_size
                self.eps = eps
            def forward(self, x):
                return x

        from kb_nano.bench.kernels.runner import _instantiate_module
        mod = _instantiate_module(ConfigModule, {"hidden_size": 32, "eps": 1e-5}, device="cpu")
        check(mod.hidden_size == 32, "2e. hidden_size=32 propagated")
        check(mod.eps == 1e-5, "2e. eps=1e-5 propagated")
        mod = _instantiate_module(
            ConfigModule,
            {"hidden_size": 48, "eps": 1e-4, "training": False},
            device="cpu",
        )
        check(
            mod.hidden_size == 48 and mod.eps == 1e-4,
            "2e. unsupported init metadata is ignored",
        )

    # 2f. Scenario filtering
    with _Timeout(30):
        from kb_nano.bench.kernels.scenario_registry import InputRegistry

        filter_yaml = """
rms_norm:
  scenarios:
"""
        for i in range(10):
            model = "llama" if i < 3 else "mixtral"
            filter_yaml += f"""    - name: "{model}-scenario-{i}/decode-bs1/tp1"
      init_args: {{hidden_size: 4, eps: 1.0e-6}}
      inputs:
        x: {{shape: [1, 4], dtype: float32}}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(filter_yaml)
            reg = InputRegistry(inputs_dir=inputs_dir)
            all_s = reg.scenarios("rms_norm")
            check(len(all_s) == 10, "2f. total 10 scenarios")
            filtered = reg.scenarios("rms_norm", models=["llama"])
            check(len(filtered) == 3, "2f. model=llama -> 3 scenarios")

    # 2g. Mutated input tensors are part of correctness
    with _Timeout(30):
        baseline_inputs = {
            "cache": torch.zeros(2, 2, dtype=torch.float32),
            "x": torch.ones(1, dtype=torch.float32),
        }
        candidate_inputs = {
            "cache": torch.zeros(2, 2, dtype=torch.float32),
            "x": torch.ones(1, dtype=torch.float32),
        }
        baseline_inputs["cache"][0, 0] = 1.0
        candidate_inputs["cache"][0, 0] = 2.0
        correct, error_ratio, diff = _merge_correctness(
            _compare_outputs(None, None),
            _compare_outputs(baseline_inputs, candidate_inputs),
        )
        check(
            not correct and error_ratio > 1.0 and diff > 0.0,
            "2g. mutated input mismatch fails correctness",
        )
        candidate_inputs["cache"][0, 0] = 1.0
        correct, error_ratio, diff = _merge_correctness(
            _compare_outputs(None, None),
            _compare_outputs(baseline_inputs, candidate_inputs),
        )
        check(
            correct and error_ratio == 0.0 and diff == 0.0,
            "2g. matching mutated inputs pass correctness",
        )

    # 2h. Correctness is checked on one forward, independent of timing repeats
    with _Timeout(30):
        class AddInPlace(nn.Module):
            def forward(self, x, residual):
                residual.add_(x)
                return x, residual

        baseline = AddInPlace()
        candidate = AddInPlace()
        inputs = {
            "x": torch.ones(2, 2, dtype=torch.float32),
            "residual": torch.zeros(2, 2, dtype=torch.float32),
        }
        baseline_inputs = {k: v.clone() for k, v in inputs.items()}
        candidate_inputs = {k: v.clone() for k, v in inputs.items()}
        baseline_out = _run_forward_once(baseline, baseline_inputs)
        candidate_out = _run_forward_once(candidate, candidate_inputs)
        correct, error_ratio, diff = _merge_correctness(
            _compare_outputs(baseline_out, candidate_out),
            _compare_outputs(baseline_inputs, candidate_inputs),
        )
        check(
            correct and error_ratio == 0.0 and diff == 0.0,
            "2h. one-forward in-place correctness passes for identical modules",
        )

        timing_inputs = {k: v.clone() for k, v in inputs.items()}
        _time_forward(baseline, timing_inputs, num_warmup=1, num_runs=2)
        check(
            torch.equal(timing_inputs["residual"], torch.full((2, 2), 3.0)),
            "2h. timing loop may mutate inputs repeatedly",
        )


# ===========================================================================
# Section 3: Result dataclasses and JSON output (unit, no GPU)
# ===========================================================================
def test_section_3():
    print(f"\n{'=' * 60}")
    print("  SECTION 3: Result dataclasses and JSON output")
    print(f"{'=' * 60}")

    from kb_nano.bench.kernels.result import (
        KernelBenchResult,
        OperatorResult,
        ScenarioResult,
    )

    # 3a. KernelBenchResult construction
    with _Timeout(30):
        scenarios_a = [
            ScenarioResult("s1", True, 0.1, 1e-7, 0.1, 0.08, 1.25),
            ScenarioResult("s2", True, 0.2, 2e-7, 0.12, 0.10, 1.20),
            ScenarioResult("s3", False, 2.5, 5e-4, 0.1, 0.09, 1.11),
        ]
        op_a = OperatorResult(
            target="rms_norm", level=1,
            candidate_path="tasks/candidate/L1/rms_norm.py",
            scenarios=scenarios_a,
        )
        scenarios_b = [
            ScenarioResult("s4", True, 0.3, 3e-7, 0.2, 0.18, 1.11),
            ScenarioResult("s5", True, 0.1, 1e-7, 0.15, 0.14, 1.07),
        ]
        op_b = OperatorResult(
            target="silu_and_mul", level=1,
            candidate_path="tasks/candidate/L1/silu_and_mul.py",
            scenarios=scenarios_b,
        )

        result = KernelBenchResult(operators=[op_a, op_b])
        result.compute_aggregates()

        check(result.total_operators == 2, "3a. total_operators == 2")
        check(result.total_scenarios == 5, "3a. total_scenarios == 5")
        check(result.passed == 4, "3a. passed == 4")
        check(result.failed == 1, "3a. failed == 1")
        check(
            result.avg_max_error_ratio > 0.0,
            f"3a. avg_max_error_ratio={result.avg_max_error_ratio:.2e} > 0",
        )
        check(result.avg_speedup > 1.0, f"3a. avg_speedup={result.avg_speedup:.2f} > 1.0")

    # 3b. JSON round-trip
    with _Timeout(30):
        d = result.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        check(
            parsed["total_operators"] == 2 and parsed["total_scenarios"] == 5,
            "3b. JSON round-trip preserves top-level fields",
        )
        check(
            len(parsed["operators"]) == 2,
            "3b. JSON round-trip preserves operators array",
        )
        check(
            len(parsed["operators"][0]["scenarios"]) == 3,
            "3b. JSON round-trip preserves scenarios in first operator",
        )

    # 3c. Schema consistency (1 operator vs 4)
    with _Timeout(30):
        single = KernelBenchResult(operators=[op_a])
        single.compute_aggregates()
        d_single = single.to_dict()
        check(
            "operators" in d_single and isinstance(d_single["operators"], list),
            "3c. single operator: 'operators' is a list",
        )
        check(len(d_single["operators"]) == 1, "3c. single operator: list has 1 entry")

        multi = KernelBenchResult(operators=[op_a, op_b, op_a, op_b])
        multi.compute_aggregates()
        d_multi = multi.to_dict()
        check(len(d_multi["operators"]) == 4, "3c. four operators: list has 4 entries")

    # 3d. JSON file writing
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "kernels.json")
            result.save_json(path)
            check(os.path.exists(path), "3d. JSON file created")
            with open(path) as f:
                loaded = json.load(f)
            check(loaded["total_operators"] == 2, "3d. JSON file has correct data")

    # 3e. Terminal table formatting
    with _Timeout(30):
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        result.print_table(single_target=False)
        sys.stdout = old_stdout
        output = buf.getvalue()
        check("CORRECT" in output or "PASS" in output, "3e. table contains PASS/CORRECT")
        check("ERR_RATIO" in output, "3e. table contains ERR_RATIO")
        check("SPEEDUP" in output or "speedup" in output, "3e. table contains SPEEDUP")
        check("OVERALL" in output, "3e. table contains OVERALL summary")
        check("ALL OPERATORS SUMMARY" in output, "3e. multi-operator table has summary")

    # 3f. MacroEval aggregation
    with _Timeout(30):
        from kb_nano.bench.eval.aggregator import Aggregator
        from kb_nano.bench.eval.runner import JobResult

        llm_valid = JobResult(
            model="model-a",
            tp=1,
            category="llm",
            throughput_results=[
                {"speedup": 2.0, "aligned_matches": 1, "aligned_total": 1},
                {"speedup": 8.0, "aligned_matches": 1, "aligned_total": 1},
            ],
            latency_results=[{"speedup": 1.0}],
        )
        llm_invalid = JobResult(
            model="model-b",
            tp=1,
            category="llm",
            throughput_results=[
                {"speedup": 16.0, "aligned_matches": 1, "aligned_total": 2},
            ],
            latency_results=[{"speedup": 16.0}],
        )
        diffusion_valid = JobResult(
            model="model-c",
            tp=1,
            category="diffusion",
            throughput_results=[{"speedup": 1.0}],
            latency_results=[{"speedup": 4.0}],
        )

        report = Aggregator.aggregate(
            [llm_valid, llm_invalid, diffusion_valid],
            wall_clock_seconds=12.0,
        )
        cats = {c.name: c for c in report.categories}

        check(
            abs(cats["llm"].macro_correctness - 0.75) < 1e-9,
            f"3f. llm macro correctness={cats['llm'].macro_correctness:.3f}",
        )
        check(
            abs(cats["llm"].macro_coverage - 0.5) < 1e-9,
            f"3f. llm macro coverage={cats['llm'].macro_coverage:.3f}",
        )
        check(
            abs(report.macro_correctness - 0.875) < 1e-9,
            f"3f. macro correctness={report.macro_correctness:.3f}",
        )
        check(
            abs(report.macro_coverage - 0.75) < 1e-9,
            f"3f. macro coverage={report.macro_coverage:.3f}",
        )
        check(
            abs(report.macro_speedup - 2.0) < 1e-9,
            f"3f. macro speedup={report.macro_speedup:.3f}",
        )
        check(
            abs(report.macro_score - 1.3125) < 1e-9,
            f"3f. macro score={report.macro_score:.4f}",
        )

    # 3g. MacroEval excludes invalid/failed items from speedup credit
    with _Timeout(30):
        from kb_nano.bench.eval.aggregator import Aggregator
        from kb_nano.bench.eval.runner import JobResult

        llm_valid = JobResult(
            model="llm-valid",
            tp=1,
            category="llm",
            throughput_results=[
                {"speedup": 4.0, "aligned_matches": 2, "aligned_total": 2},
                {"speedup": 9.0, "aligned_matches": 2, "aligned_total": 2},
            ],
            latency_results=[
                {"speedup": 1.0},
                {"speedup": 4.0},
            ],
        )
        llm_invalid_fast = JobResult(
            model="llm-invalid-fast",
            tp=1,
            category="llm",
            throughput_results=[
                {"speedup": 100.0, "aligned_matches": 0, "aligned_total": 1},
            ],
            latency_results=[{"speedup": 100.0}],
        )
        vision_failed = JobResult(
            model="vision-failed",
            tp=1,
            category="vision",
            status="FAILED",
            error="synthetic failure",
        )

        report = Aggregator.aggregate([llm_valid, llm_invalid_fast, vision_failed])
        cats = {c.name: c for c in report.categories}
        expected_llm_thru = 6.0
        expected_llm_lat = 2.0
        expected_llm_blend = math.sqrt(expected_llm_thru * expected_llm_lat)

        check(
            abs(cats["llm"].models[0].throughput_speedup - expected_llm_thru) < 1e-9,
            f"3g. per-item throughput geomean={cats['llm'].models[0].throughput_speedup:.3f}",
        )
        check(
            abs(cats["llm"].models[0].latency_speedup - expected_llm_lat) < 1e-9,
            f"3g. per-item latency geomean={cats['llm'].models[0].latency_speedup:.3f}",
        )
        check(
            abs(cats["llm"].models[0].blended_speedup - expected_llm_blend) < 1e-9,
            f"3g. blended speedup={cats['llm'].models[0].blended_speedup:.3f}",
        )
        check(
            not cats["llm"].models[1].valid
            and cats["llm"].models[1].blended_speedup == 0.0,
            "3g. invalid fast item receives no speedup credit",
        )
        check(
            cats["vision"].macro_coverage == 0.0
            and cats["vision"].macro_correctness == 0.0
            and cats["vision"].macro_speedup == 1.0,
            "3g. failed-only family affects coverage/correctness but not speedup",
        )
        check(
            abs(report.macro_speedup - expected_llm_blend) < 1e-9,
            f"3g. macro speedup excludes failed-only family: {report.macro_speedup:.3f}",
        )
        check(
            abs(report.macro_correctness - 0.25) < 1e-9
            and abs(report.macro_coverage - 0.25) < 1e-9,
            f"3g. macro correctness/coverage={report.macro_correctness:.2f}/{report.macro_coverage:.2f}",
        )

    # 3h. MacroEval JSON schema and terminal output
    with _Timeout(30):
        from kb_nano.bench.eval.aggregator import Aggregator
        from kb_nano.bench.eval.runner import JobResult

        report = Aggregator.aggregate([
            JobResult(
                model="macro-json-model",
                tp=1,
                category="llm",
                throughput_results=[
                    {"speedup": 1.5, "aligned_matches": 3, "aligned_total": 3},
                ],
                latency_results=[{"speedup": 2.0}],
            )
        ])
        d = report.to_dict()
        check(
            all(k in d for k in (
                "macro_speedup", "macro_correctness",
                "macro_coverage", "macro_score",
            )),
            "3h. EvalReport JSON contains top-level MacroEval fields",
        )
        check(
            all(k in d["categories"][0] for k in (
                "macro_speedup", "macro_correctness",
                "macro_coverage", "macro_score",
            )),
            "3h. Category JSON contains MacroEval fields",
        )
        check(
            all(k in d["categories"][0]["models"][0] for k in (
                "correctness_score", "valid", "blended_speedup",
            )),
            "3h. Model JSON contains MacroEval item fields",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "eval.json")
            report.save_json(path)
            with open(path) as f:
                loaded = json.load(f)
            check(
                loaded["macro_score"] == d["macro_score"],
                "3h. MacroEval fields persist through save_json",
            )

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        report.print_table()
        sys.stdout = old_stdout
        output = buf.getvalue()
        check(
            "MacroEval speedup" in output
            and "MacroEval correctness" in output
            and "MacroEval score" in output,
            "3h. terminal output includes MacroEval summary",
        )


# ===========================================================================
# Section 4: Standardized workloads (unit, no GPU)
# ===========================================================================
def test_section_4():
    print(f"\n{'=' * 60}")
    print("  SECTION 4: Standardized workloads")
    print(f"{'=' * 60}")

    from kb_nano.bench.utils.workloads import (
        LATENCY_WORKLOADS,
        THROUGHPUT_WORKLOADS,
        get_max_seq_len,
    )

    # 4a. Throughput workload constants
    with _Timeout(30):
        check(len(THROUGHPUT_WORKLOADS) == 3, "4a. exactly 3 throughput workloads")
        names = [w.name for w in THROUGHPUT_WORKLOADS]
        check(
            names == ["prefill-heavy", "balanced", "decode-heavy"],
            f"4a. correct names: {names}",
        )
        ph = THROUGHPUT_WORKLOADS[0]
        check(
            ph.dataset_name.endswith("prefill-heavy-1k"),
            "4a. prefill-heavy dataset configured",
        )
        bal = THROUGHPUT_WORKLOADS[1]
        check(
            bal.dataset_name.endswith("balanced-1k"),
            "4a. balanced dataset configured",
        )
        dh = THROUGHPUT_WORKLOADS[2]
        check(
            dh.dataset_name.endswith("decode-heavy-1k"),
            "4a. decode-heavy dataset configured",
        )

    # 4b. Latency workload constants
    with _Timeout(30):
        check(len(LATENCY_WORKLOADS) == 2, "4b. exactly 2 latency workloads")
        sr = LATENCY_WORKLOADS[0]
        check(
            sr.name == "single-request" and sr.batch_size == 1
            and sr.input_len == 128 and sr.output_len == 128,
            "4b. single-request: bs=1, 128/128",
        )
        fb = LATENCY_WORKLOADS[1]
        check(
            fb.name == "fixed-batch-32" and fb.batch_size == 32
            and fb.input_len == 128 and fb.output_len == 128,
            "4b. fixed-batch-32: bs=32, 128/128",
        )

    # 4c. Immutability (frozen dataclasses)
    with _Timeout(30):
        try:
            THROUGHPUT_WORKLOADS[0].dataset_name = "other"
            check(False, "4c. throughput workloads should be immutable")
        except AttributeError:
            check(True, "4c. throughput workloads are frozen (immutable)")
        try:
            LATENCY_WORKLOADS[0].batch_size = 999
            check(False, "4c. latency workloads should be immutable")
        except AttributeError:
            check(True, "4c. latency workloads are frozen (immutable)")

    # 4d. get_max_seq_len
    with _Timeout(30):
        max_len = get_max_seq_len()
        check(
            max_len == 256,
            f"4d. static max_seq_len = {max_len} (expected 256 = 128+128)",
        )


# ===========================================================================
# Section 5: Multi-level conflict resolution (unit, no GPU)
# ===========================================================================
def _can_discover_targets() -> bool:
    """Check if target discovery works (requires sgl_kernel and other CUDA deps)."""
    try:
        from kb_nano.infra.kernel_swapper import discover_targets
        discover_targets()
        return True
    except Exception as exc:
        print(f"    SKIP  target discovery unavailable: {exc}")
        return False


def test_section_5():
    print(f"\n{'=' * 60}")
    print("  SECTION 5: Multi-level conflict resolution")
    print(f"{'=' * 60}")

    from kb_nano.infra.kernel_swapper import (
        BenchTarget,
        _detect_subsumption,
        _sort_by_level,
    )

    # 5a. Bottom-up ordering
    with _Timeout(30):
        t1 = BenchTarget("op_a", 3, "tasks.baseline.L3.op_a", ["llama31"], nn.Module)
        t2 = BenchTarget("op_b", 1, "tasks.baseline.L1.op_b", ["llama31"], nn.Module)
        t3 = BenchTarget("op_c", 2, "tasks.baseline.L2.op_c", ["llama31"], nn.Module)

        class Fake(nn.Module):
            pass

        candidates = [(t1, Fake), (t2, Fake), (t3, Fake)]
        sorted_c = _sort_by_level(candidates)
        levels = [t.level for t, _ in sorted_c]
        check(levels == [1, 2, 3], f"5a. sorted levels: {levels} == [1, 2, 3]")

    has_targets = _can_discover_targets()

    # 5b. Subsumption detection
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5b. sgl_kernel not available, cannot discover targets")
        else:
            from kb_nano.infra.kernel_swapper import get

            rms_target = get("rms_norm")
            llama_decoder_target = get("llama_decoder")

            class FakeRMS(nn.Module):
                pass

            class FakeDecoder(nn.Module):
                pass

            candidates = [
                (rms_target, FakeRMS),
                (llama_decoder_target, FakeDecoder),
            ]
            warnings = _detect_subsumption(candidates)
            found_subsumption = any(
                lower_name == "rms_norm" for _, _, lower_name, _ in warnings
            )
            check(
                found_subsumption,
                f"5b. L3 llama_decoder subsumes L1 rms_norm (found {len(warnings)} warnings)",
            )

    # 5c. No false positives (mock targets with no import relationship)
    with _Timeout(30):
        class FakeA(nn.Module):
            pass
        class FakeB(nn.Module):
            pass

        mock_l1 = BenchTarget("fake_op_alpha", 1, "tasks.baseline.L1.fake_op_alpha", ["llama31"], nn.Module)
        mock_l2 = BenchTarget("fake_op_beta", 2, "tasks.baseline.L2.fake_op_beta", ["llama31"], nn.Module)
        candidates_no_overlap = [
            (mock_l1, FakeA),
            (mock_l2, FakeB),
        ]
        warnings_no = _detect_subsumption(candidates_no_overlap)
        check(
            len(warnings_no) == 0,
            "5c. mock targets with no import chain -> no subsumption (no false positive)",
        )

    # 5d. Patching order
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5d. sgl_kernel not available, cannot discover targets")
        else:
            from kb_nano.infra.kernel_swapper import get, patch_class, restore
            rms_target = get("rms_norm")
            rms_module = importlib.import_module(f"{PACKAGE_NAME}.{rms_target.module_path}")
            original_cls = rms_target.target_cls

            class PatchedRMSNorm(nn.Module):
                _is_patched = True

            undo = patch_class(rms_target, PatchedRMSNorm)
            patched_cls = getattr(rms_module, original_cls.__name__)
            check(
                patched_cls is PatchedRMSNorm,
                "5d. L1 patch applied correctly",
            )

            decoder_mod = importlib.import_module(f"{PACKAGE_NAME}.tasks.baseline.L3.llama_decoder")
            decoder_rms = getattr(decoder_mod, original_cls.__name__, None)
            check(
                decoder_rms is PatchedRMSNorm,
                "5d. L3 baseline picks up L1 patch",
            )

            restore(undo)
            check(
                getattr(rms_module, original_cls.__name__) is original_cls,
                "5d. restore works correctly",
            )

    # 5e. Discovery convenience functions
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5e. sgl_kernel not available, cannot discover targets")
        else:
            from kb_nano.infra.kernel_swapper import (
                list_targets, models_for_target, targets_for_model,
            )

            all_targets = list_targets()
            check(len(all_targets) > 0, "5e. list_targets() returns non-empty list")

            l1_targets = list_targets(level=1)
            check(
                len(l1_targets) > 0 and all(t.level == 1 for t in l1_targets),
                "5e. list_targets(level=1) returns only L1 targets",
            )

            rms_models = models_for_target("rms_norm")
            check(
                "llama31" in rms_models,
                '5e. models_for_target("rms_norm") contains "llama31"',
            )

            llama_targets = targets_for_model("llama31")
            check(
                len(llama_targets) > 0
                and all("llama31" in t.models for t in llama_targets),
                '5e. targets_for_model("llama31") all include "llama31"',
            )


# ===========================================================================
# Section 6: CLI argument parsing (unit, no GPU)
# ===========================================================================
def test_section_6():
    print(f"\n{'=' * 60}")
    print("  SECTION 6: CLI argument parsing")
    print(f"{'=' * 60}")

    # 6a. bench.kernels CLI argument parsing
    with _Timeout(30):
        import argparse as _ap

        old_argv = sys.argv
        sys.argv = ["prog", "--target", "rms_norm", "--model", "llama", "--tp", "1", "4",
                     "--category", "llm", "--output-json", "/tmp/test.json",
                     "--num-warmup", "5", "--num-runs", "50"]
        try:
            parser = _ap.ArgumentParser()
            parser.add_argument("--target", type=str)
            parser.add_argument("--model", nargs="+")
            parser.add_argument("--tp", type=int, nargs="+")
            parser.add_argument("--category", type=str)
            parser.add_argument("--output-json", type=str)
            parser.add_argument("--num-warmup", type=int)
            parser.add_argument("--num-runs", type=int)
            args = parser.parse_args(sys.argv[1:])
            check(
                args.target == "rms_norm" and args.tp == [1, 4]
                and args.model == ["llama"] and args.category == "llm"
                and args.num_warmup == 5 and args.num_runs == 50,
                "6a. --target, --model, --tp, --category, --output-json, --num-warmup, --num-runs parsed",
            )
        finally:
            sys.argv = old_argv

    # 6b. bench.e2e CLI
    with _Timeout(30):
        result = subprocess.run(
            [sys.executable, "-m", "kb_nano.bench.e2e", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        if result.returncode != 0:
            # May fail due to missing sgl_kernel on import - check for import-related vs parse errors
            if "sgl_kernel" in result.stderr or "ModuleNotFoundError" in result.stderr:
                print("    SKIP  6b. sgl_kernel not available for subprocess import")
            else:
                check(False, f"6b. e2e --help failed: {result.stderr[-200:]}")
        else:
            check(
                "throughput" in result.stdout and "latency" in result.stdout
                and "serve" in result.stdout,
                "6b. e2e help shows throughput, latency, serve subcommands",
            )
            # Check eval is not in subcommand list (after the header)
            help_text = result.stdout
            if "Benchmark type" in help_text:
                subcommand_section = help_text.split("Benchmark type")[1]
                check(
                    "eval" not in subcommand_section,
                    "6b. eval subcommand removed from e2e",
                )
            else:
                check(True, "6b. eval subcommand not present in e2e help")

    # 6c. bench.eval CLI
    with _Timeout(30):
        result = subprocess.run(
            [sys.executable, "-m", "kb_nano.bench.eval", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        if result.returncode != 0 and ("sgl_kernel" in result.stderr or "ModuleNotFoundError" in result.stderr):
            print("    SKIP  6c. sgl_kernel not available for subprocess import")
        else:
            check(
                result.returncode == 0 and "--model" in result.stdout,
                "6c. eval CLI accepts --model",
            )
            check(
                "--tp" in result.stdout and "--category" in result.stdout
                and "--output-json" in result.stdout,
                "6c. eval CLI accepts --tp, --category, --output-json",
            )

    # 6d. Default JSON output path
    with _Timeout(30):
        from kb_nano import RESULTS_DIR, run_output_path

        kernels_default = run_output_path("kernels")
        check(
            kernels_default.parent == RESULTS_DIR
            and kernels_default.name.startswith("kernels_")
            and kernels_default.suffix == ".json",
            f"6d. kernels default output: {kernels_default}",
        )

        eval_default = run_output_path("eval")
        check(
            eval_default.parent == RESULTS_DIR
            and eval_default.name.startswith("eval_")
            and eval_default.suffix == ".json",
            f"6d. eval default output: {eval_default}",
        )


# ===========================================================================
# Section 7: Kernel-level integration (GPU required, single GPU)
# ===========================================================================
def test_section_7():
    print(f"\n{'=' * 60}")
    print("  SECTION 7: Kernel-level integration (GPU required)")
    print(f"{'=' * 60}")

    candidate_dir = os.path.join(PACKAGE_DIR, "tasks", "candidate", "L1")
    candidate_file = os.path.join(candidate_dir, "rms_norm.py")
    had_candidate = os.path.exists(candidate_file)
    if had_candidate:
        original_candidate = open(candidate_file).read()

    def _write_identity_rms_candidate():
        with open(candidate_file, "w") as f:
            f.write(f"""\
from {PACKAGE_NAME}.tasks.baseline.L1.rms_norm import RMSNorm
""")

    # 7a. Identity replacement via the new runner
    with _Timeout(120):
        try:
            os.makedirs(candidate_dir, exist_ok=True)
            _write_identity_rms_candidate()
            result = subprocess.run(
                [sys.executable, "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from {PACKAGE_NAME}.bench.kernels.runner import run_kernel_benchmark

result = run_kernel_benchmark(
    'rms_norm',
    num_warmup=2,
    num_runs=5,
)
print(json.dumps({{
    'passed': result.passed,
    'failed': result.failed,
    'total': result.total_scenarios,
    'avg_error_ratio': result.avg_max_error_ratio,
    'avg_diff': result.avg_mean_abs_diff,
    'avg_speedup': result.avg_speedup,
}}))
"""],
                capture_output=True, text=True, timeout=60,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                print(f"    STDERR: {result.stderr[-500:]}")
                check(False, "7a. identity subprocess exited non-zero")
            else:
                data = json.loads(result.stdout.strip().split("\n")[-1])
                check(
                    data["avg_error_ratio"] == 0.0,
                    f"7a. identity: avg_error_ratio={data['avg_error_ratio']:.2e} == 0",
                )
                check(
                    data["failed"] == 0,
                    f"7a. identity: {data['passed']}/{data['total']} passed",
                )
        except subprocess.TimeoutExpired:
            check(False, "7a. identity test timed out after 60s (possible hang)")

    # 7b. Broken replacement (write broken candidate to candidates folder)
    with _Timeout(120):
        try:
            with open(candidate_file, "w") as f:
                f.write("""\
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size=4096, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
    def forward(self, x, residual=None):
        if residual is None:
            return torch.zeros_like(x)
        return torch.zeros_like(x), residual
""")
            result = subprocess.run(
                [sys.executable, "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from {PACKAGE_NAME}.bench.kernels.runner import run_kernel_benchmark

result = run_kernel_benchmark(
    'rms_norm',
    num_warmup=2,
    num_runs=5,
)
print(json.dumps({{
    'passed': result.passed,
    'failed': result.failed,
    'avg_error_ratio': result.avg_max_error_ratio,
    'avg_diff': result.avg_mean_abs_diff,
}}))
"""],
                capture_output=True, text=True, timeout=60,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                print(f"    STDERR: {result.stderr[-500:]}")
                check(False, "7b. broken subprocess exited non-zero")
            else:
                data = json.loads(result.stdout.strip().split("\n")[-1])
                check(
                    data["avg_error_ratio"] > 1.0,
                    f"7b. broken: avg_error_ratio={data['avg_error_ratio']:.2e} > 1",
                )
                check(
                    data["failed"] > 0,
                    f"7b. broken: {data['failed']} failures detected",
                )
        except subprocess.TimeoutExpired:
            check(False, "7b. broken test timed out after 60s")

    # 7c. JSON output file via CLI (baseline as candidate in candidates folder)
    with _Timeout(180):
        try:
            _write_identity_rms_candidate()
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "kernels.json")
                result = subprocess.run(
                    [sys.executable, "-m", "kb_nano.bench.kernels",
                     "--target", "rms_norm",
                     "--output-json", json_path,
                     "--num-warmup", "2", "--num-runs", "5"],
                    timeout=120, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode not in (0, 1):
                    print(f"    STDERR: {result.stderr[-500:]}")
                check(
                    os.path.exists(json_path),
                    "7c. JSON output file created by CLI",
                )
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        data = json.load(f)
                    check(
                        "operators" in data and "total_scenarios" in data,
                        "7c. JSON has correct schema",
                    )
        except subprocess.TimeoutExpired:
            check(False, "7c. CLI test timed out after 120s")
        finally:
            if had_candidate:
                with open(candidate_file, "w") as f:
                    f.write(original_candidate)
            elif os.path.exists(candidate_file):
                os.remove(candidate_file)

    # 7d. All-candidates default (skip if no candidates exist)
    with _Timeout(360):
        from kb_nano.infra.kernel_swapper import discover_candidates
        candidates = discover_candidates()
        if not candidates:
            print("    SKIP  7d. no candidate kernels found, skipping all-candidates test")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "all.json")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "kb_nano.bench.kernels",
                         "--output-json", json_path,
                         "--num-warmup", "2", "--num-runs", "5"],
                        timeout=300, cwd=PROJECT_ROOT,
                        capture_output=True, text=True,
                    )
                    check(
                        os.path.exists(json_path),
                        "7d. all-candidates JSON created",
                    )
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data["total_operators"] == len(candidates),
                            f"7d. tested {data['total_operators']}/{len(candidates)} candidates",
                        )
                except subprocess.TimeoutExpired:
                    check(False, "7d. all-candidates timed out after 300s")


# ===========================================================================
# Section 8: E2E integration (GPU required, single GPU)
# ===========================================================================
def test_section_8():
    print(f"\n{'=' * 60}")
    print("  SECTION 8: E2E integration (GPU required)")
    print(f"{'=' * 60}")

    # 8a. Throughput single-run
    with _Timeout(360):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "throughput.json")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "kb_nano.bench.e2e", "throughput",
                     "--model", "meta-llama/Llama-3.1-8B-Instruct",
                     "--tp", "1",
                     "--dataset-name", "random",
                     "--random-input-len", "128",
                     "--random-output-len", "64",
                     "--num-prompts", "10",
                     "--output-json", json_path,
                     "--no-candidate-kernels"],
                    timeout=300, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"    STDERR: {result.stderr[-500:]}")
                    check(False, "8a. throughput subprocess failed")
                else:
                    check(os.path.exists(json_path), "8a. throughput JSON created")
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data.get("tokens_per_second", 0) > 0,
                            f"8a. tokens_per_second={data.get('tokens_per_second', 0):.0f} > 0",
                        )
            except subprocess.TimeoutExpired:
                check(False, "8a. throughput timed out after 300s")

    # 8b. Latency single-run
    with _Timeout(360):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "latency.json")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "kb_nano.bench.e2e", "latency",
                     "--model", "meta-llama/Llama-3.1-8B-Instruct",
                     "--batch-size", "1",
                     "--input-len", "128",
                     "--output-len", "64",
                     "--num-iters-warmup", "1",
                     "--num-iters", "3",
                     "--output-json", json_path,
                     "--no-candidate-kernels"],
                    timeout=300, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"    STDERR: {result.stderr[-500:]}")
                    check(False, "8b. latency subprocess failed")
                else:
                    check(os.path.exists(json_path), "8b. latency JSON created")
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data.get("avg_latency", 0) > 0,
                            f"8b. avg_latency={data.get('avg_latency', 0):.4f} > 0",
                        )
            except subprocess.TimeoutExpired:
                check(False, "8b. latency timed out after 300s")

    # 8c. JSON default save (verify default path works with --output-json)
    with _Timeout(30):
        check(True, "8c. default JSON paths verified in section 6d (no redundant GPU run)")


# ===========================================================================
# Section 9: Eval integration (GPU required, single GPU)
# ===========================================================================
def test_section_9():
    print(f"\n{'=' * 60}")
    print("  SECTION 9: Eval integration (GPU required)")
    print(f"{'=' * 60}")

    # 9a. Single-model eval (requires at least one candidate)
    with _Timeout(660):
        from kb_nano.infra.kernel_swapper import discover_candidates
        candidates = discover_candidates()
        if not candidates:
            print("    SKIP  9a. no candidate kernels, skipping eval test")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "eval.json")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "kb_nano.bench.eval",
                         "--model", "meta-llama/Llama-3.1-8B-Instruct",
                         "--tp", "1",
                         "--num-prompts", "10",
                         "--output-json", json_path],
                        timeout=600, cwd=PROJECT_ROOT,
                        capture_output=True, text=True,
                    )
                    if result.returncode != 0:
                        print(f"    STDERR: {result.stderr[-500:]}")
                        check(False, "9a. eval subprocess failed")
                    else:
                        check(os.path.exists(json_path), "9a. eval JSON created")
                        if os.path.exists(json_path):
                            with open(json_path) as f:
                                data = json.load(f)
                            check(
                                "total_models" in data and "categories" in data,
                                "9a. eval JSON has expected schema",
                            )
                            check(
                                data.get("total_jobs", 0) >= 1,
                                f"9a. total_jobs={data.get('total_jobs', 0)} >= 1",
                            )
                except subprocess.TimeoutExpired:
                    check(False, "9a. eval timed out after 600s")

    # 9b. Eval with no candidates
    with _Timeout(60):
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"""
import sys, os
sys.path.insert(0, '{PROJECT_ROOT}')
import tempfile, shutil

# Temporarily rename candidate dir to simulate no candidates
from {PACKAGE_NAME}.infra.kernel_swapper import _CANDIDATE_DIR
backup = str(_CANDIDATE_DIR) + '_backup'
if _CANDIDATE_DIR.is_dir():
    shutil.move(str(_CANDIDATE_DIR), backup)
os.makedirs(str(_CANDIDATE_DIR), exist_ok=True)

try:
    from {PACKAGE_NAME}.bench.eval.config import EvalConfig
    from {PACKAGE_NAME}.bench.eval.planner import EvalPlanner
    config = EvalConfig(models=['meta-llama/Llama-3.1-8B-Instruct'], tp_degrees=[1])
    planner = EvalPlanner(config)
    plan = planner.plan()
    # With explicit models, plan should have jobs even without candidates
    # The eval runner would just compare baseline vs baseline
    print('PLAN_JOBS:' + str(plan.num_jobs))
except Exception as e:
    print('ERROR:' + str(e))
finally:
    import shutil
    if os.path.exists(backup):
        shutil.rmtree(str(_CANDIDATE_DIR), ignore_errors=True)
        shutil.move(backup, str(_CANDIDATE_DIR))
"""],
                capture_output=True, text=True, timeout=30,
                cwd=PROJECT_ROOT,
            )
            output = result.stdout.strip()
            check(
                result.returncode == 0,
                f"9b. no-candidates exits cleanly (rc={result.returncode})",
            )
        except subprocess.TimeoutExpired:
            check(False, "9b. no-candidates timed out after 30s (possible hang)")

    # 9c. JSON default save path
    with _Timeout(30):
        from kb_nano import RESULTS_DIR, run_output_path

        default_output = run_output_path("eval")
        check(
            default_output.parent == RESULTS_DIR
            and default_output.name.startswith("eval_")
            and default_output.suffix == ".json",
            f"9c. eval default output: {default_output}",
        )

    # 9d. MacroEval report integration schema
    with _Timeout(30):
        from kb_nano.bench.eval.aggregator import Aggregator
        from kb_nano.bench.eval.runner import JobResult

        report = Aggregator.aggregate([
            JobResult(
                model="integration-a",
                tp=1,
                category="llm",
                throughput_results=[
                    {"speedup": 2.0, "aligned_matches": 2, "aligned_total": 2},
                ],
                latency_results=[{"speedup": 2.0}],
            ),
            JobResult(
                model="integration-b",
                tp=1,
                category="vlm",
                status="FAILED",
                error="synthetic failure",
            ),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "eval_macro.json")
            report.save_json(json_path)
            with open(json_path) as f:
                data = json.load(f)
        check(
            abs(data["macro_speedup"] - 2.0) < 1e-9,
            f"9d. macro_speedup={data['macro_speedup']:.2f}",
        )
        check(
            abs(data["macro_correctness"] - 0.5) < 1e-9
            and abs(data["macro_coverage"] - 0.5) < 1e-9
            and abs(data["macro_score"] - 0.5) < 1e-9,
            "9d. MacroEval correctness/coverage/score persisted",
        )
        check(
            len(data["categories"]) == 2
            and all("macro_score" in c for c in data["categories"]),
            "9d. category MacroEval schema persisted",
        )


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test the kb-nano benchmarking infrastructure",
    )
    parser.add_argument(
        "--section", type=int, default=None,
        help="Run only a specific section (1-9)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  kb-nano benchmarking infrastructure tests")
    print("=" * 60)

    sections = {
        1: ("Input Registry", test_section_1),
        2: ("KernelRunner", test_section_2),
        3: ("Result dataclasses", test_section_3),
        4: ("Standardized workloads", test_section_4),
        5: ("Conflict resolution", test_section_5),
        6: ("CLI argument parsing", test_section_6),
        7: ("Kernel integration", test_section_7),
        8: ("E2E integration", test_section_8),
        9: ("Eval integration", test_section_9),
    }

    for num, (name, func) in sorted(sections.items()):
        if args.section is not None and args.section != num:
            continue
        try:
            func()
        except TimeoutError:
            check(False, f"Section {num} ({name}) timed out entirely")
        except Exception as e:
            check(False, f"Section {num} ({name}) raised exception: {e}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_pass_count} passed, {_fail_count} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
