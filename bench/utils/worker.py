"""Subprocess worker runner for benchmark isolation.

Runs benchmark workers in clean subprocesses to avoid import contamination
and ensure CUDA graphs / torch.compile operate in a pristine environment.

Refactored from tests/bench_throughput.py.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile


def _terminate_process_group(pid: int) -> None:
    """Best-effort cleanup for worker grandchildren such as vLLM ranks."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        pass


def run_worker(
    script: str,
    config: dict,
    label: str,
    timeout: int = 3600,
    *,
    python_executable: str | None = None,
) -> dict | None:
    """Run a worker script in a subprocess and return parsed JSON output.

    Args:
        script: Python source code to execute.
        config: JSON-serializable configuration dict passed as argv[1].
                An ``output_file`` key is added automatically.
        label: Human-readable label printed before/after execution.
        timeout: Maximum wall-clock seconds before the subprocess is killed.
        python_executable: Path to the Python interpreter to invoke. Defaults
            to ``sys.executable``. Use this to run a worker in a different
            conda env (e.g. an isolated env where sglang/OpenPI is installed
            so its torch/CUDA versions do not contaminate the parent env).

    Returns:
        Parsed JSON dict written by the worker to ``output_file``, or None on
        failure.
    """
    py = python_executable or sys.executable
    if not os.path.exists(py):
        print(
            f"  ERROR: {label} -- python interpreter not found: {py}\n"
            f"         (set --sglang-python or create the env)"
        )
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp",
    ) as f:
        f.write(script)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp",
    ) as f:
        config["output_file"] = output_path
        json.dump(config, f)
        config_path = f.name

    try:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"  python: {py}")
        print(f"{'─' * 70}", flush=True)

        env = os.environ.copy()
        bindir = os.path.dirname(os.path.abspath(py))
        if bindir:
            env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        proc = subprocess.Popen(
            [py, "-u", script_path, config_path],
            start_new_session=True,
            env=env,
        )
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc.pid)
            print(f"  ERROR: {label} timed out after {timeout}s")
            return None

        if returncode != 0:
            _terminate_process_group(proc.pid)
            print(f"  ERROR: {label} failed with exit code {returncode}")
            return None

        with open(output_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(script_path)
        os.unlink(config_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


KB_NANO_WORKER = r'''
import json, sys, time
with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    pkg = cfg["package_name"]

    if not cfg.get("no_candidate_kernels", False):
        swapper = __import__(
            f"{pkg}.infra.kernel_swapper",
            fromlist=["discover_candidates", "apply_candidates", "print_candidate_summary"],
        )
        candidates = swapper.discover_candidates()
        if candidates:
            swapper.print_candidate_summary(candidates)
            swapper.apply_candidates(candidates)

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    engine = LlamaEngine(**engine_kwargs)

    prompts = cfg["prompts"]
    temperature = cfg.get("temperature", 1.0)
    top_p = cfg.get("top_p", 1.0)
    output_lens = cfg["output_lens"]
    ignore_eos = cfg.get("ignore_eos", True)

    sp_list = [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=ol,
            ignore_eos=ignore_eos,
        )
        for ol in output_lens
    ]

    # Warmup
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, sp_list, use_tqdm=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_input_tokens = sum(
        len(engine.tokenizer.encode(p)) if isinstance(p, str) else len(p)
        for p in prompts
    )
    total_output_tokens = sum(len(o.token_ids) for o in outputs)

    result = {
        "elapsed": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    if cfg.get("save_outputs", False):
        result["outputs"] = [
            {
                "prompt": o.prompt,
                "generated_text": o.generated_text,
                "token_ids": o.token_ids,
            }
            for o in outputs
        ]

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

    del engine

if __name__ == "__main__":
    main()
'''

KB_NANO_MULTI_SCENARIO_WORKER = r'''
import json, sys, time

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    if not cfg.get("no_candidate_kernels", False):
        swapper = __import__(
            f"{pkg}.infra.kernel_swapper",
            fromlist=["discover_candidates", "apply_candidates", "print_candidate_summary"],
        )
        candidates = swapper.discover_candidates()
        if candidates:
            swapper.print_candidate_summary(candidates)
            swapper.apply_candidates(candidates)

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    engine = LlamaEngine(**engine_kwargs)

    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompts = scenario["prompt_token_ids"]
        output_lens = scenario["output_lens"]
        temperature = cfg.get("temperature", 0.0)
        top_p = cfg.get("top_p", 1.0)

        sp_list = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=ol,
                ignore_eos=True,
            )
            for ol in output_lens
        ]

        engine.block_manager.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts, sp_list, use_tqdm=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_input_tokens = sum(len(p) for p in prompts)
        total_output_tokens = sum(len(o.token_ids) for o in outputs)

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {
                    "generated_text": o.generated_text,
                    "token_ids": o.token_ids,
                }
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompt_token_ids"]
        output_lens = ls.get("output_lens")
        if output_lens is None:
            sp = SamplingParams(temperature=0.0,
                                ignore_eos=True, max_tokens=ls["output_len"])
        else:
            sp = [
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
                for ol in output_lens
            ]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine

if __name__ == "__main__":
    main()
'''
