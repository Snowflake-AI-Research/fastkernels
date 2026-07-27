from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastkernels.validate import (
    ValidateScenario,
    _build_cmd,
    _resolve_validate_scenarios,
)
from fastkernels.validate.ray_runner import (
    _build_summary,
    _plan_jobs,
    _ray_job_id,
    _ray_resource_options,
    _visible_to_physical_gpu_ids,
)
from fastkernels.workloads import LLM


def _args(**overrides):
    values = {
        "max_layers": None,
        "max_requests": None,
        "resume": False,
        "vllm_python": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _scenario(model: str, *, tp: int = 1, workloads=("mixed",)):
    return ValidateScenario(
        hf_name=model,
        tp=tp,
        dtype="bfloat16",
        legacy_workloads=tuple(workloads),
    )


def test_build_vllm_command_forwards_supported_options(tmp_path):
    cmd = _build_cmd(
        _scenario("meta-llama/Llama-3.1-8B-Instruct", tp=2),
        "bench_vllm",
        _args(
            max_layers=3,
            max_requests=7,
            resume=True,
            vllm_python="/opt/vllm/bin/python",
        ),
        tmp_path,
    )

    assert cmd[0:2]
    assert cmd[cmd.index("--model") + 1] == "meta-llama/Llama-3.1-8B-Instruct"
    assert cmd[cmd.index("--tp") + 1] == "2"
    assert cmd[cmd.index("--max-layers") + 1] == "3"
    assert cmd[cmd.index("--num-seqs") + 1] == "7"
    assert cmd[cmd.index("--vllm-python") + 1] == "/opt/vllm/bin/python"
    assert cmd[cmd.index("--output-dir") + 1] == str(tmp_path)
    assert "--resume" in cmd


def test_build_eagle_command_splits_target_and_draft(tmp_path):
    scenario = _scenario(
        "meta-llama/Llama-3.1-8B-Instruct + "
        "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B (draft)"
    )
    cmd = _build_cmd(scenario, "bench_sglang", _args(), tmp_path)

    assert cmd[cmd.index("--model") + 1] == "meta-llama/Llama-3.1-8B-Instruct"
    assert (
        cmd[cmd.index("--draft-model") + 1]
        == "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
    )


def test_build_model_less_ttt_command_uses_results_file(tmp_path):
    cmd = _build_cmd(
        _scenario("ttt_e2e", workloads=("125m_e2e",)),
        "bench_ttt_e2e",
        _args(max_requests=4),
        tmp_path,
    )

    assert "--model" not in cmd
    assert cmd[cmd.index("--variant") + 1] == "125m_e2e"
    assert cmd[cmd.index("--n-sequences") + 1] == "4"
    assert cmd[cmd.index("--cache-dir") + 1] == str(tmp_path / "cache")
    assert cmd[cmd.index("--results-out") + 1] == str(tmp_path / "results.json")
    assert "--output-dir" not in cmd


def test_commands_launch_validation_harness_as_package_module(tmp_path):
    cmd = _build_cmd(
        _scenario("meta-llama/Llama-3.1-8B-Instruct"),
        "bench_vllm",
        _args(),
        tmp_path,
    )

    assert cmd[1:4] == ["-u", "-m", "fastkernels.validate.bench_vllm"]


def test_build_native_ttt_command_keeps_paper_variant(tmp_path):
    scenario = SimpleNamespace(
        hf_name="ttt_e2e",
        tp=1,
        dtype="bfloat16",
        workloads=[LLM.mixed, LLM.long_context],
        enforce_eager=False,
    )

    cmd = _build_cmd(scenario, "bench_ttt_e2e", _args(), tmp_path)

    assert cmd[cmd.index("--variant") + 1] == "125m_e2e"


def test_paper_scenarios_load_with_legacy_workloads():
    scenarios = _resolve_validate_scenarios("paper_scenarios")

    assert len(scenarios) == 47
    assert scenarios[0].legacy_workloads[0] == "prefill-heavy"
    assert scenarios[14].hf_name == "ttt_e2e"
    assert scenarios[14].legacy_workloads == ("125m_e2e",)


def test_ray_resources_scale_with_tensor_parallel_degree():
    resources = _ray_resource_options(
        2,
        8,
        {"CPU": 64, "memory": 256 * 1024**3},
    )

    assert resources["num_gpus"] == 2
    assert resources["num_cpus"] == 15
    assert 40 * 1024**3 < resources["memory"] < 50 * 1024**3


def test_visible_gpu_ids_map_back_to_parent_physical_ids():
    assert _visible_to_physical_gpu_ids(
        ["0", "2", "GPU-deadbeef"],
        ["1", "3", "7"],
    ) == ["1", "7", "GPU-deadbeef"]
    assert _visible_to_physical_gpu_ids(
        ["1", "3"],
        ["1", "3", "7"],
    ) == ["1", "3"]


def test_ray_task_name_contains_scenario_model_tp_and_workloads(tmp_path):
    scenario = _scenario("Qwen/Qwen3-Next-80B-A3B-Instruct", tp=2)
    jobs, results, cached = _plan_jobs(
        [scenario],
        _args(),
        4,
        tmp_path,
    )

    assert results == {}
    assert cached == []
    name = _ray_job_id("paper", jobs[0])
    assert name.startswith("validate_paper_000_Qwen_Qwen3-Next")
    assert "_tp2_mixed" in name


def test_resume_marks_existing_results_as_cached(tmp_path):
    scenario = _scenario("ttt_e2e", workloads=("125m_e2e",))
    initial_jobs, _, _ = _plan_jobs([scenario], _args(), 1, tmp_path)
    run_dir = Path(initial_jobs[0]["run_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text("{}")

    jobs, results, cached = _plan_jobs(
        [scenario],
        _args(resume=True),
        1,
        tmp_path,
    )

    assert jobs == []
    assert results == {0: "PASS(cached)"}
    assert cached[0]["status"] == "PASS(cached)"


def test_resume_uses_success_marker_without_results_json(tmp_path):
    scenario = _scenario("openpi-assets/checkpoints/pi0_aloha_pen_uncap")
    initial_jobs, _, _ = _plan_jobs([scenario], _args(), 1, tmp_path)
    job = initial_jobs[0]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "task_result.json").write_text(
        json.dumps(
            {
                **job,
                "returncode": 0,
                "status": "PASS",
            }
        )
    )

    jobs, results, cached = _plan_jobs(
        [scenario],
        _args(resume=True),
        1,
        tmp_path,
    )

    assert jobs == []
    assert results == {0: "PASS(cached)"}
    assert cached[0]["cached"] is True


def test_summary_uses_finished_task_paths(tmp_path):
    scenario = _scenario("ttt_e2e", workloads=("125m_e2e",))
    run_dir = tmp_path / "job"
    event = {
        "event": "task_finished",
        "result": {
            "index": 0,
            "name": scenario.hf_name,
            "harness": "bench_ttt_e2e",
            "tp": 1,
            "run_dir": str(run_dir),
            "status": "PASS",
        },
    }
    (tmp_path / "run.jsonl").write_text(json.dumps(event) + "\n")

    summary = _build_summary(tmp_path, [scenario], {0: "PASS"})

    assert summary["run"]["status"] == "PASS"
    assert summary["models"][0]["paths"]["results_json"] == str(
        run_dir / "results.json"
    )


def test_summary_aggregates_throughput_and_latency_rows(tmp_path):
    scenario = _scenario("meta-llama/Llama-3.1-8B-Instruct")
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "model": scenario.hf_name,
                "tp": 1,
                "scenarios": [
                    {
                        "scenario": "mixed",
                        "speedup": 1.5,
                        "alignment": {
                            "avg_matching_tokens_per_request": 8.0,
                            "exact_matches": 1,
                            "total_seqs": 2,
                        },
                    }
                ],
                "latency_scenarios": [
                    {"scenario": "single-request", "speedup": 1.25}
                ],
            }
        )
    )
    event = {
        "event": "task_finished",
        "result": {
            "index": 0,
            "name": scenario.hf_name,
            "harness": "bench_vllm",
            "tp": 1,
            "run_dir": str(run_dir),
            "status": "PASS",
        },
    }
    (tmp_path / "run.jsonl").write_text(json.dumps(event) + "\n")

    summary = _build_summary(tmp_path, [scenario], {0: "PASS"})

    assert summary["throughput"][0]["speedup"] == 1.5
    assert "exact match: 1/2" in summary["throughput"][0]["correctness"]
    assert summary["latency"][0]["speedup"] == 1.25
