from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastkernels.validate import (
    ValidateScenario,
    _build_cmd,
    _resolve_validate_scenarios,
)
from fastkernels.validate.ray_runner import (
    _build_summary,
    _is_fatal_worker_line,
    _latency_rows_for_result,
    _plan_jobs,
    _ray_job_id,
    _ray_resource_options,
    _throughput_rows_for_result,
    _visible_to_physical_gpu_ids,
    _write_summary,
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


def test_fatal_worker_line_ignores_vllm_warning_tracebacks():
    # Codestral: vLLM prints a config AttributeError as WARNING + traceback.
    warning = (
        "WARNING 08-21 09:44:11 [config.py:1242] "
        "Traceback (most recent call last):"
    )
    assert _is_fatal_worker_line(warning) is False
    assert _is_fatal_worker_line("Traceback (most recent call last):\n") is True
    assert _is_fatal_worker_line(
        '  File "engine.py", line 1, in generate\n'
    ) is False
    assert _is_fatal_worker_line(
        "terminate called after throwing an instance of 'std::runtime_error'\n"
    ) is True
    assert _is_fatal_worker_line(
        "torch.AcceleratorError: CUDA error: an illegal memory access "
        "was encountered\n"
    ) is True


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


def test_plan_jobs_skips_nvfp4_on_hopper(tmp_path):
    scenario = ValidateScenario(
        hf_name="nvidia/GLM-5.2-NVFP4",
        tp=8,
        dtype="nvfp4",
        legacy_workloads=("mixed",),
    )
    jobs, results, cached = _plan_jobs(
        [scenario], _args(), 8, tmp_path, hopper=True
    )
    assert jobs == []
    assert cached == []
    assert results == {0: "SKIP(nvfp4-hopper)"}


def test_plan_jobs_keeps_nvfp4_off_hopper(tmp_path):
    scenario = ValidateScenario(
        hf_name="nvidia/GLM-5.2-NVFP4",
        tp=8,
        dtype="nvfp4",
        legacy_workloads=("mixed",),
    )
    jobs, results, cached = _plan_jobs(
        [scenario], _args(), 8, tmp_path, hopper=False
    )
    assert len(jobs) == 1
    assert results == {}
    assert cached == []


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

    assert summary["models"][0]["paths"]["results_json"] == str(
        run_dir / "results.json"
    )
    # This job reported PASS but wrote no results.json, so it measured none of
    # its declared workloads. The coverage guard fails the run for it: a job
    # that exits 0 and produces nothing is the silent hole it exists to catch.
    assert summary["run"]["status"] == "FAIL"
    assert summary["coverage_gaps"][0]["missing"] == ["125m_e2e"]


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


# --- Summary-row extraction per harness result shape -----------------------
#
# Every harness compares against its reference and prints a speedup, but they
# disagree on where they record it: the rate field is named after the unit
# counted (images/videos/utterances/req), the median is variously "median_s",
# "median", "p50_ms" or only a raw "latencies" list, and bench_sglang names its
# ratio "speedup_vs_sglang". A shape the aggregator does not know produces a
# blank cell rather than an error, so each supported shape is pinned here.


def _rows(harness: str, data: dict) -> tuple[list[dict], list[dict]]:
    return (
        _throughput_rows_for_result("m", harness, data),
        _latency_rows_for_result("m", harness, data),
    )


def test_sglang_reference_named_speedup_is_read():
    throughput, latency = _rows(
        "bench_sglang",
        {
            "scenarios": [
                {"scenario": "eagle3", "speedup_vs_sglang": 0.98, "alignment": {}}
            ],
            "latency_scenarios": [
                {"scenario": "latency-bs1", "speedup_vs_sglang": 1.1}
            ],
        },
    )
    assert throughput[0]["speedup"] == 0.98
    assert latency[0]["speedup"] == 1.1


def test_two_sided_throughput_rate_key_follows_the_counted_unit():
    for rate_key, expected in (
        ("images_per_second", 2.0),
        ("videos_per_second", 2.0),
        ("utterances_per_second", 2.0),
        ("throughput_req_s", 2.0),
    ):
        throughput, _ = _rows(
            "bench_vllm_omni",
            {
                "fastkernels": {"throughput": [{"name": "w", rate_key: 4.0}]},
                "vllm_omni": {"throughput": [{"name": "w", rate_key: 2.0}]},
            },
        )
        assert throughput[0]["speedup"] == expected, rate_key


def test_two_sided_latency_falls_back_to_the_raw_latencies_list():
    # bench_vllm_omni / bench_diffusers / bench_timm record no median at all.
    _, latency = _rows(
        "bench_timm",
        {
            "fastkernels": {"latency": [{"name": "single-image", "latencies": [1.0, 2.0]}]},
            "timm": {"latency": [{"name": "single-image", "latencies": [2.0, 4.0]}]},
        },
    )
    assert latency[0]["speedup"] == 2.0


def test_detection_latency_uses_its_seconds_median_key():
    # bench_detection writes "median", not the "median_s" comparison.py uses.
    _, latency = _rows(
        "bench_detection",
        {
            "fastkernels": {"latency": [{"name": "single-image", "median": 0.004}]},
            "reference": {"latency": [{"name": "single-image", "median": 0.002}]},
        },
    )
    assert latency[0]["speedup"] == 0.5


def test_dp3_millisecond_percentile_is_converted_before_dividing():
    throughput, latency = _rows(
        "bench_dp3",
        {
            "fastkernels": {
                "throughput": [{"name": "dp3-1env", "throughput_req_s": 30.0}],
                "latency": [{"name": "single-step", "p50_ms": 34.0}],
            },
            "reference": {
                "throughput": [{"name": "dp3-1env", "throughput_req_s": 25.0}],
                "latency": [{"name": "single-step", "p50_ms": 41.0}],
            },
            "correctness": {"dp3-1env": {"mean_cos": 1.0}},
        },
    )
    assert throughput[0]["speedup"] == 30.0 / 25.0
    assert throughput[0]["correctness"] == "min_cos=1.0000"
    assert latency[0]["speedup"] == 41.0 / 34.0


def test_diffusers_rows_carry_the_scenario_cosine():
    throughput, latency = _rows(
        "bench_diffusers",
        {
            "fastkernels": {
                "throughput": [{"name": "512x512", "images_per_second": 4.0}],
                "latency": [{"name": "single-512x512", "latencies": [0.5]}],
            },
            "diffusers": {
                "throughput": [{"name": "512x512", "images_per_second": 2.0}],
                "latency": [{"name": "single-512x512", "latencies": [1.0]}],
            },
            "correctness": {"512x512": {"min_cosine_sim": 0.95, "mean_cosine_sim": 0.99}},
        },
    )
    assert throughput[0]["speedup"] == 2.0
    assert throughput[0]["correctness"] == "min_cos=0.9500"
    assert latency[0]["speedup"] == 2.0


def test_sam_flat_top_level_throughput_becomes_one_row():
    throughput, latency = _rows(
        "bench_sam",
        {
            "fastkernels_items_per_sec": 8.0,
            "ref_items_per_sec": 16.0,
            "correctness": {"boxes": {"min_cosine_similarity": 0.96}},
            "latency_scenarios": [{"scenario": "single-image-1008", "speedup": 1.05}],
        },
    )
    assert [(r["workload"], r["speedup"]) for r in throughput] == [
        ("full-pipeline", 0.5)
    ]
    assert throughput[0]["correctness"] == "min_cos=0.9600"
    assert latency[0]["speedup"] == 1.05


def test_recsys_rows_are_keyed_by_declared_workload_not_model():
    throughput, latency = _rows(
        "bench_recsys",
        {
            "models": {
                "dlrmv2": {
                    "alignment": {"logits": {"cosine": 1.0}},
                    "throughput": {
                        "reference": "torchrec.models.dlrm.DLRM",
                        "ours": {
                            "samples_per_second": 4.0,
                            "latency_ms_p50": 0.35,
                        },
                        "reference_metrics": {
                            "samples_per_second": 2.0,
                            "latency_ms_p50": 0.70,
                        },
                        "ratio_vs_reference": 2.0,
                    },
                }
            }
        },
    )
    assert throughput[0]["workload"] == "ctr-batch"
    assert throughput[0]["speedup"] == 2.0
    assert throughput[0]["reference"] == "torchrec.models.dlrm.DLRM"
    # The p50 belongs to the throughput batch, so it must not claim to be the
    # declared single-request scenario the harness never runs.
    assert latency[0]["workload"] == "ctr-batch-p50"
    assert latency[0]["speedup"] == 2.0


def test_vjepa2_task_is_the_throughput_workload_and_bs1_is_single_video():
    throughput, latency = _rows(
        "bench_vjepa2",
        {
            "task": "predictor",
            "throughput": {
                "ours": {"videos_per_second": 21.0},
                "reference": {"videos_per_second": 42.0},
            },
            "latency": {
                "ours": {"results": [{"batch_size": 1, "latencies": [0.05]}]},
                "reference": {"results": [{"batch_size": 1, "latencies": [0.10]}]},
            },
            "alignment": {"metrics": {"last_hidden_state": {"cosine": 1.0}}},
        },
    )
    assert throughput[0]["workload"] == "predictor"
    assert throughput[0]["speedup"] == 0.5
    assert throughput[0]["correctness"] == "min_cos=1.0000"
    assert [(r["workload"], r["speedup"]) for r in latency] == [("single-video", 2.0)]


def test_zero_and_missing_rates_leave_the_cell_blank_instead_of_dividing():
    throughput, latency = _rows(
        "bench_vllm_omni",
        {
            "fastkernels": {
                "throughput": [{"name": "w", "images_per_second": 0.0}],
                "latency": [{"name": "single-w", "latencies": []}],
            },
            "vllm_omni": {
                "throughput": [{"name": "w"}],
                "latency": [{"name": "single-w"}],
            },
        },
    )
    assert throughput[0]["speedup"] is None
    assert latency[0]["speedup"] is None


# --- Declared workloads actually reach the harness -------------------------
#
# For harnesses in _WORKLOADS_FLAG the scenario table drives the run, so a name
# the table declares but the harness does not accept would abort the job. The
# reverse drift is what this guards: bench_vjepa2 defaulted to --task predictor
# while full.yaml declared four workloads, and the sweep silently ran one.


def test_vjepa2_scenario_passes_every_declared_workload_in_one_call(tmp_path):
    from fastkernels.validate import _scenario_workloads

    scenario = next(
        s
        for s in _resolve_validate_scenarios("full")
        if "vjepa2" in s.hf_name
    )
    cmd = _build_cmd(scenario, "bench_vjepa2", _args(), tmp_path)

    declared = _scenario_workloads(scenario)
    assert cmd[cmd.index("--workloads") + 1] == ",".join(declared)
    assert cmd.count("--workloads") == 1  # one invocation covers them all


def test_bench_vjepa2_accepts_exactly_what_the_scenario_tables_declare():
    from fastkernels.validate import _scenario_workloads
    from fastkernels.validate.bench_vjepa2 import _resolve_workloads

    for table in ("full",):
        scenario = next(
            s for s in _resolve_validate_scenarios(table) if "vjepa2" in s.hf_name
        )
        declared = ",".join(_scenario_workloads(scenario))
        tasks, batches = _resolve_workloads(declared, "predictor", "1,2")
        assert tasks == ["predictor", "encoder"], table
        assert batches == [1], table


def test_bench_vjepa2_rejects_a_workload_it_cannot_run():
    from fastkernels.validate.bench_vjepa2 import _resolve_workloads

    for bad in ("mixed", "predictor,single-image"):
        try:
            _resolve_workloads(bad, "predictor", "1,2")
        except SystemExit as exc:
            assert "not a V-JEPA 2 workload" in str(exc)
        else:
            raise AssertionError(f"{bad!r} should not resolve")
    # A list naming only latency workloads has nothing to benchmark.
    try:
        _resolve_workloads("single-video", "predictor", "1,2")
    except SystemExit as exc:
        assert "no throughput workload" in str(exc)
    else:
        raise AssertionError("latency-only workload list should not resolve")


def test_bench_vjepa2_without_workloads_keeps_single_task_behaviour():
    from fastkernels.validate.bench_vjepa2 import _resolve_workloads

    assert _resolve_workloads("", "encoder", "1,2") == (["encoder"], [1, 2])


def test_vjepa2_standard_shape_is_read_and_legacy_shape_still_parses():
    standard = {
        "model": "facebook/vjepa2-vitl-fpc64-256",
        "reference_name": "transformers",
        "tasks": {"predictor": {}, "encoder": {}},
        "scenarios": [
            {"scenario": "predictor", "speedup": 0.96, "alignment": None},
            {"scenario": "encoder", "speedup": 1.02, "alignment": None},
        ],
        "latency_scenarios": [{"scenario": "single-video", "speedup": 1.0}],
    }
    throughput, latency = _rows("bench_vjepa2", standard)
    assert [r["workload"] for r in throughput] == ["predictor", "encoder"]
    assert [r["reference"] for r in throughput] == ["transformers", "transformers"]
    assert [r["workload"] for r in latency] == ["single-video"]

    legacy = {
        "task": "predictor",
        "throughput": {
            "ours": {"videos_per_second": 21.0},
            "reference": {"videos_per_second": 42.0},
        },
        "latency": {
            "ours": {"results": [{"batch_size": 1, "latencies": [0.05]}]},
            "reference": {"results": [{"batch_size": 1, "latencies": [0.10]}]},
        },
    }
    throughput, latency = _rows("bench_vjepa2", legacy)
    assert throughput[0]["speedup"] == 0.5
    assert latency[0]["speedup"] == 2.0


def test_sam_declared_throughput_workload_matches_the_row_the_runner_emits():
    from fastkernels.validate import _scenario_workloads

    scenario = next(
        s for s in _resolve_validate_scenarios("full") if "sam3" in s.hf_name
    )
    declared = _scenario_workloads(scenario)
    throughput, _ = _rows(
        "bench_sam",
        {"fastkernels_items_per_sec": 8.0, "ref_items_per_sec": 16.0},
    )
    assert throughput[0]["workload"] in declared


# --- SAM throughput and latency ---------------------------------------------
#
# bench_sam makes two throughput measurements (a pooled image pass and clip
# tracking) and three latency ones. The clip-tracking rate used to be printed
# and then dropped from results.json, losing a whole declared workload -- and it
# is SAM's weakest number, so the loss flattered the summary.


def _sam_standard_results():
    from fastkernels.validate.comparison import latency_entry, throughput_entry

    return {
        "model": "facebook/sam3.1",
        "reference_name": "sam3",
        "fastkernels_items_per_sec": 8.0,
        "ref_items_per_sec": 16.0,
        "fastkernels_video_frames_per_sec": 37.8,
        "ref_video_frames_per_sec": 47.7,
        "scenarios": [
            throughput_entry("full-pipeline", 8.0, 16.0, metric="items_per_s"),
            throughput_entry(
                "smartglasses-val-video", 37.8, 47.7, metric="frames_per_s"
            ),
        ],
        "latency_scenarios": [
            latency_entry("single-image-1008", 0.0611, 0.0640, batch_size=1),
            latency_entry("batch-4-image-1008", 0.0609, 0.0639, batch_size=4),
            latency_entry("single-video-frame-1008", 0.0609, 0.0640, batch_size=1),
        ],
    }


def test_sam_reports_both_image_and_clip_throughput():
    throughput, _ = _rows("bench_sam", _sam_standard_results())

    assert [r["workload"] for r in throughput] == [
        "full-pipeline",
        "smartglasses-val-video",
    ]
    # Throughput speedup is ours/reference: slower than the reference reads <1.
    assert throughput[0]["speedup"] == 0.5
    assert throughput[1]["speedup"] == pytest.approx(37.8 / 47.7)
    assert all(r["reference"] == "sam3" for r in throughput)


def test_sam_latency_speedup_is_reference_over_ours():
    _, latency = _rows("bench_sam", _sam_standard_results())

    assert [r["workload"] for r in latency] == [
        "single-image-1008",
        "batch-4-image-1008",
        "single-video-frame-1008",
    ]
    # Lower latency is better, so ours < reference must read >1.
    assert latency[0]["speedup"] == pytest.approx(0.0640 / 0.0611)
    assert all(r["speedup"] > 1.0 for r in latency)


def test_sam_rows_cover_every_declared_workload():
    from fastkernels.validate import _scenario_workloads

    scenario = next(
        s for s in _resolve_validate_scenarios("full") if "sam3" in s.hf_name
    )
    throughput, latency = _rows("bench_sam", _sam_standard_results())
    emitted = {r["workload"] for r in throughput} | {r["workload"] for r in latency}

    assert emitted == set(_scenario_workloads(scenario))


def test_sam_legacy_results_still_parse_without_the_video_row():
    # Runs recorded before bench_sam persisted the clip rate have only the flat
    # image-pass fields; they must still yield the full-pipeline row.
    legacy = {
        "fastkernels_items_per_sec": 8.0,
        "ref_items_per_sec": 16.0,
        "correctness": {"masks": {"min_cosine_similarity": 0.87}},
        "latency_scenarios": [
            {"scenario": "single-image-1008", "speedup": 1.047}
        ],
    }
    throughput, latency = _rows("bench_sam", legacy)

    assert [(r["workload"], r["speedup"]) for r in throughput] == [
        ("full-pipeline", 0.5)
    ]
    assert throughput[0]["correctness"] == "min_cos=0.8700"
    assert latency[0]["speedup"] == 1.047


def test_sam_alignment_block_uses_the_harness_pass_verdict():
    from fastkernels.validate.bench_sam import _alignment_block, _min_cosine

    correctness = {
        "boxes": {"min_cosine_similarity": 0.96, "avg_cosine_similarity": 0.98},
        "masks": {"min_cosine_similarity": 0.87, "avg_cosine_similarity": 0.95},
    }
    assert _min_cosine(correctness) == 0.87
    assert _alignment_block(correctness, True)["passed"] is True
    assert _alignment_block(correctness, False)["passed"] is False
    # Video correctness carries no pass verdict, so none is invented.
    assert "passed" not in _alignment_block(correctness, None)
    assert _alignment_block({}, True) is None


# --- Coverage guard ---------------------------------------------------------
#
# The 20260802 sweep reported PASS with 46/46 models passing while 17 of them
# had left a declared workload with no value in the summary table. Every job
# exited 0, so nothing complained. These pin the guard that now does.


def _guard_run(tmp_path, harness, workloads, results_json, status="PASS"):
    root = tmp_path / harness
    run_dir = root / "job"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(results_json))
    (root / "run.jsonl").write_text(
        json.dumps(
            {
                "event": "task_finished",
                "result": {
                    "index": 0,
                    "name": "m",
                    "harness": harness,
                    "tp": 1,
                    "run_dir": str(run_dir),
                    "status": status,
                },
            }
        )
        + "\n"
    )
    scenario = _scenario("m", workloads=workloads)
    return root, [scenario], _build_summary(root, [scenario], {0: status})


_COMPLETE_VLLM_RESULT = {
    "model": "m",
    "scenarios": [{"scenario": "mixed", "speedup": 1.1, "alignment": {}}],
    "latency_scenarios": [{"scenario": "single-request", "speedup": 1.2}],
}


def test_guard_passes_when_every_declared_workload_has_a_value(tmp_path):
    root, scenarios, summary = _guard_run(
        tmp_path, "bench_vllm", ["mixed", "single-request"], _COMPLETE_VLLM_RESULT
    )

    assert summary["run"]["status"] == "PASS"
    assert summary["coverage_gaps"] == []
    assert _write_summary(root, scenarios, {0: "PASS"}) == 0


def test_guard_fails_when_a_declared_workload_has_no_row(tmp_path):
    root, scenarios, summary = _guard_run(
        tmp_path,
        "bench_vllm",
        ["mixed", "single-request"],
        {**_COMPLETE_VLLM_RESULT, "latency_scenarios": []},
    )

    assert summary["run"]["status"] == "FAIL"
    assert summary["run"]["models_incomplete"] == 1
    assert summary["coverage_gaps"][0]["missing"] == ["single-request"]
    assert _write_summary(root, scenarios, {0: "PASS"}) == 1


def test_guard_fails_when_a_row_exists_but_carries_no_speedup(tmp_path):
    # A blank cell is the same failure as a missing row, and is how the
    # bench_detection and bench_vllm_omni latency gaps looked.
    root, scenarios, summary = _guard_run(
        tmp_path,
        "bench_vllm",
        ["mixed", "single-request"],
        {
            **_COMPLETE_VLLM_RESULT,
            "latency_scenarios": [{"scenario": "single-request", "speedup": None}],
        },
    )
    gap = summary["coverage_gaps"][0]

    assert summary["run"]["status"] == "FAIL"
    assert gap["blank"] == ["single-request"]
    # Diagnosed once, as a blank row rather than also as an absent one.
    assert gap["missing"] == []
    assert _write_summary(root, scenarios, {0: "PASS"}) == 1


def test_guard_stays_quiet_about_a_job_that_already_failed(tmp_path):
    root, scenarios, summary = _guard_run(
        tmp_path,
        "bench_vllm",
        ["mixed", "single-request"],
        {},
        status="FAIL(rc=1)",
    )

    assert summary["run"]["status"] == "FAIL"  # from the job, not the guard
    assert summary["run"]["models_incomplete"] == 0
    assert summary["coverage_gaps"] == []
    assert _write_summary(root, scenarios, {0: "FAIL(rc=1)"}) == 0


def test_guard_accepts_alias_harness_rows_under_their_own_names(tmp_path):
    _, _, summary = _guard_run(
        tmp_path,
        "bench_sglang",
        ["mixed", "single-request"],
        {
            "model": "m",
            "scenarios": [
                {
                    "scenario": "eagle3-16seqs-out256",
                    "speedup_vs_sglang": 0.98,
                    "alignment": {},
                }
            ],
            "latency_scenarios": [
                {"scenario": "latency-bs1-out256", "speedup_vs_sglang": 1.1}
            ],
        },
    )

    assert summary["run"]["status"] == "PASS"
    assert summary["coverage_gaps"] == []


def test_guard_still_fails_an_alias_harness_that_produced_nothing(tmp_path):
    _, _, summary = _guard_run(
        tmp_path,
        "bench_sglang",
        ["mixed"],
        {"model": "m", "scenarios": [], "latency_scenarios": []},
    )
    gap = summary["coverage_gaps"][0]

    assert summary["run"]["status"] == "FAIL"
    assert gap["missing"] == ["mixed"]
    assert "EAGLE-3" in gap["alias_reason"]


def test_guard_reports_known_unimplemented_workloads_without_failing(
    tmp_path, monkeypatch
):
    # The _UNIMPLEMENTED_WORKLOADS table is empty now that bench_recsys has real
    # batch-1 / batch-32 probes, so this exercises the mechanism with a synthetic
    # entry: a declared workload no harness implements is reported with its
    # reason but must not fail the run, because implementing the probe or
    # dropping the declaration is a judgement call, not the summary's to make.
    from fastkernels.validate import ray_runner

    monkeypatch.setitem(
        ray_runner._UNIMPLEMENTED_WORKLOADS,
        ("bench_vllm", "long-context"),
        "no long-context dataset wired up for this harness yet",
    )
    root, scenarios, summary = _guard_run(
        tmp_path,
        "bench_vllm",
        ["mixed", "long-context"],
        {
            "model": "m",
            "scenarios": [{"scenario": "mixed", "speedup": 1.1, "alignment": {}}],
            "latency_scenarios": [],
        },
    )
    gap = summary["coverage_gaps"][0]

    assert summary["run"]["status"] == "PASS"
    assert gap["missing"] == []
    assert [entry["workload"] for entry in gap["unimplemented"]] == ["long-context"]
    assert all(entry["reason"] for entry in gap["unimplemented"])
    assert _write_summary(root, scenarios, {0: "PASS"}) == 0


def test_recsys_latency_workloads_are_specified_not_name_only():
    # These two were the only single-request / fixed-batch-32 members in any
    # family with params=None, which is why bench_recsys never implemented them:
    # nothing said what they meant.
    from fastkernels.workloads import Recsys, purpose_of, spec_for

    for member, batch_size in ((Recsys.single_request, 1), (Recsys.fixed_batch_32, 32)):
        spec = spec_for(member)
        assert purpose_of(member).value == "latency"
        assert spec.params is not None, f"{member.value} is still name-only"
        assert spec.params.batch_size == batch_size


def test_recsys_emits_a_row_for_every_declared_workload():
    from fastkernels.validate.comparison import latency_entry, throughput_entry

    for model, throughput_name in (("dlrmv2", "ctr-batch"), ("lightgcn", "recommend-batch")):
        metric = "samples_per_s" if model == "dlrmv2" else "pairs_per_s"
        data = {
            "reference_name": "torchrec",
            "models": {model: {}},
            "scenarios": [throughput_entry(throughput_name, 4.0, 2.0, metric=metric)],
            "latency_scenarios": [
                latency_entry("single-request", 0.0004, 0.0005, batch_size=1),
                latency_entry("fixed-batch-32", 0.0006, 0.0007, batch_size=32),
            ],
        }
        throughput, latency = _rows("bench_recsys", data)
        assert [r["workload"] for r in throughput] == [throughput_name]
        assert [r["workload"] for r in latency] == [
            "single-request",
            "fixed-batch-32",
        ]
        assert all(r["speedup"] > 1.0 for r in throughput + latency)


def test_guard_accepts_a_row_the_reference_cannot_run(tmp_path):
    # bench_microsoft_bitnet's bs=32 latency probe: the official int2 decode GEMM
    # only implements M == 1, so there is no like-for-like reference. The harness
    # records speedup: null plus a reason rather than dividing by a serial
    # reference loop. That is a documented gap, not a parsing failure, so it must
    # be reported without failing the sweep.
    from fastkernels.validate.comparison import latency_entry, throughput_entry

    root, scenarios, summary = _guard_run(
        tmp_path,
        "bench_microsoft_bitnet",
        ["mixed", "single-request", "fixed-batch-32"],
        {
            "model": "m",
            "reference_name": "microsoft-bitnet-gpu",
            "scenarios": [throughput_entry("mixed", 900.0, 800.0, metric="tok_per_s")],
            "latency_scenarios": [
                latency_entry("single-request", 0.40, 0.42),
                latency_entry(
                    "fixed-batch-32",
                    0.50,
                    None,
                    reference_unsupported=True,
                    reference_unsupported_reason=(
                        "official int2 decode kernel dispatches only for M == 1"
                    ),
                ),
            ],
        },
    )
    gap = summary["coverage_gaps"][0]

    assert summary["run"]["status"] == "PASS"
    assert summary["run"]["models_incomplete"] == 0
    assert gap["missing"] == []
    assert "blank" not in gap  # diagnosed as unsupported, not as a blank cell
    assert gap["reference_unsupported"] == [
        {
            "workload": "fixed-batch-32",
            "reason": "official int2 decode kernel dispatches only for M == 1",
        }
    ]
    assert _write_summary(root, scenarios, {0: "PASS"}) == 0


def test_unsupported_marker_reaches_the_summary_row():
    from fastkernels.validate.comparison import latency_entry

    _, latency = _rows(
        "bench_microsoft_bitnet",
        {
            "latency_scenarios": [
                latency_entry(
                    "fixed-batch-32",
                    0.5,
                    None,
                    reference_unsupported=True,
                    reference_unsupported_reason="reference cannot batch",
                )
            ]
        },
    )

    assert latency[0]["speedup"] is None
    assert latency[0]["reference_unsupported_reason"] == "reference cannot batch"


def test_blank_row_without_a_reason_is_still_a_failure(tmp_path):
    # The escape hatch is the explicit marker, not "speedup happens to be null".
    from fastkernels.validate.comparison import latency_entry

    _, _, summary = _guard_run(
        tmp_path,
        "bench_microsoft_bitnet",
        ["single-request"],
        {"latency_scenarios": [latency_entry("single-request", 0.5, None)]},
    )

    assert summary["run"]["status"] == "FAIL"
    assert summary["coverage_gaps"][0]["blank"] == ["single-request"]


# --- SAM throughput measures the model, not the filesystem -----------------
#
# items_per_sec used to divide by a wall clock that enclosed ~68 MB/item of
# torch.save plus the .cpu() transfers: 110 ms/item measured against 61 ms of
# actual forward. That inflation, and its run-to-run variance, is what moved
# full-pipeline between 0.93x and 1.05x. The workers now divide by summed
# per-item forward time and report the wall clock separately.


def test_sam_worker_rate_uses_forward_time_not_wall_clock():
    import re

    source = Path("fastkernels/validate/bench_sam.py").read_text()

    # Every rate the workers report must be per forward_elapsed.
    assert 'len(images) / forward_elapsed' in source
    assert 'total_frames / forward_elapsed' in source
    # The wall clock is kept, but only under an explicitly-named field.
    assert '"wall_items_per_sec"' in source
    assert '"wall_frames_per_sec"' in source
    # No rate may still be computed from total_elapsed.
    for bad in ('"items_per_sec": len(images) / total_elapsed',
                '"frames_per_sec": total_frames / total_elapsed'):
        assert bad not in source, bad

    # One accumulator per throughput loop: two image workers, two video workers.
    assert len(re.findall(r"forward_elapsed = 0\.0", source)) == 4
    assert len(re.findall(r"forward_elapsed \+= elapsed_", source)) == 4


def test_sam_throughput_loops_warm_up_before_timing():
    source = Path("fastkernels/validate/bench_sam.py").read_text()

    # The latency probes always warmed up; the throughput passes did not, so
    # item 0 absorbed lazy init and autotune. At 10 clips that biased the video
    # row by more than the difference it was reporting.
    assert source.count('cfg.get("throughput_warmup"') == 4
    warmup_index = source.index("Warm up outside the timed region")
    timed_index = source.index("start = time.perf_counter()", warmup_index)
    assert warmup_index < timed_index, "warmup must precede the timed region"


def test_sam_saving_is_gated_to_the_correctness_subset():
    source = Path("fastkernels/validate/bench_sam.py").read_text()

    # Predictions cost ~68 MB per image and ~64 MB per clip per side, so the
    # throughput item count can only grow if saving is capped independently.
    assert source.count("if feats_dir and i < correctness_items:") == 2
    assert source.count("ci < correctness_clips") == 2
    assert source.count('cfg.get("correctness_items", num_items)') == 2
    assert source.count('cfg.get("correctness_clips", len(clips))') == 2


def test_sam_declared_throughput_size_matches_the_harness_default():
    import argparse
    import re

    from fastkernels.workloads import SEGMENTATION_THROUGHPUT_WORKLOADS

    source = Path("fastkernels/validate/bench_sam.py").read_text()
    match = re.search(r'"--num-items", type=int, default=(\d+)', source)
    assert match, "could not find the --num-items default"
    declared = next(
        w for w in SEGMENTATION_THROUGHPUT_WORKLOADS if w.name == "full-pipeline"
    )
    # A declared num_requests that disagrees with what the harness actually runs
    # is the same class of fiction as a workload that never runs at all.
    assert int(match.group(1)) == declared.num_requests


def test_sam_worker_sources_have_no_undefined_names():
    """Each bench_sam worker is a source string run in a fresh subprocess.

    Helpers defined in one worker are not in scope in its siblings, so a name
    copied between them fails only at runtime, on a GPU, minutes in. Two such
    bugs (model_dtype, run_detection) were introduced while adding warmup to the
    video workers; this catches that class statically.
    """
    import ast
    import builtins
    import re

    src = Path("fastkernels/validate/bench_sam.py").read_text()
    workers = re.findall(r"^(\w*WORKER\w*) = r?'''(.*?)'''", src, re.S | re.M)
    assert len(workers) == 4, f"expected 4 worker sources, found {len(workers)}"

    for name, body in workers:
        tree = ast.parse(body)  # raises on syntax errors
        defined = set(dir(builtins)) | {"__name__"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert not (used - defined), f"{name} uses undefined {sorted(used - defined)}"
