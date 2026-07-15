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

