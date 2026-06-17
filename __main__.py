"""CLI entry point for kb_nano.

Dispatches to the appropriate subcommand:

    kb_nano kernels [args...]
    kb_nano eval [args...]
    kb_nano e2e throughput|latency|serve [args...]
    kb_nano agent [args...]
    kb_nano generate-inputs [args...]
    kb_nano capture-golden [args...]
    kb_nano trace-inputs [args...]
    kb_nano build-input-registry [args...]
    kb_nano validate-input-registry [args...]
    kb_nano create-stubs [args...]
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand from argv so the downstream parsers see clean args
    sys.argv = [f"kb_nano {command}"] + sys.argv[2:]

    if command == "kernels":
        from kb_nano.bench.kernels.__main__ import main as kernels_main
        kernels_main()
    elif command == "eval":
        from kb_nano.bench.eval.__main__ import main as eval_main
        eval_main()
    elif command == "e2e":
        from kb_nano.bench.e2e.__main__ import main as e2e_main
        e2e_main()
    elif command == "agent":
        from kb_nano.agent.agent import main as agent_main
        agent_main()
    elif command == "generate-inputs":
        from kb_nano.bench.kernels.scenario_pipeline import generate_inputs_main as gen_main
        gen_main()
    elif command == "capture-golden":
        from kb_nano.bench.kernels.scenario_pipeline import capture_golden_main as cap_main
        cap_main()
    elif command == "trace-inputs":
        from kb_nano.bench.kernels.scenario_pipeline import trace_inputs_main as trace_main
        trace_main()
    elif command == "build-input-registry":
        from kb_nano.bench.kernels.scenario_pipeline import build_input_registry_main as build_main
        build_main()
    elif command == "validate-input-registry":
        from kb_nano.bench.kernels.scenario_pipeline import validate_input_registry_main as validate_main
        validate_main()
    elif command == "create-stubs":
        from kb_nano.agent.create_stubs import main as stubs_main
        stubs_main()
    elif command == "history":
        from kb_nano.bench.tracking.history import history_main
        history_main()
    elif command == "mlflow-ui":
        from kb_nano.bench.tracking.history import mlflow_ui_main
        mlflow_ui_main()
    elif command in ("-h", "--help", "help"):
        _print_usage()
    else:
        print(f"Unknown command: {command}")
        _print_usage()
        sys.exit(1)


def _print_usage():
    print("Usage: kb_nano <command> [args...]")
    print()
    print("Commands:")
    print("  kernels          Isolated kernel-level benchmarks")
    print("  eval             Multi-model evaluation sweep")
    print("  e2e              End-to-end benchmarks (throughput, latency, serve)")
    print("  agent            LLM-powered kernel generation agent")
    print("  create-stubs     Create skeleton replacement modules")
    print("  history          Query tracked experiment runs (MLflow)")
    print("  mlflow-ui        Launch the MLflow web UI")
    print("  generate-inputs  Generate YAML input manifests")
    print("  capture-golden   Capture golden tensor data for data-dependent ops")
    print("  trace-inputs     Trace real workloads into raw input metadata JSONL")
    print("  build-input-registry")
    print("                   Build generated YAML and representative workloads")
    print("  validate-input-registry")
    print("                   Validate registry loadability and golden coverage")
    print()
    print("Run 'kb_nano <command> --help' for command-specific options.")


if __name__ == "__main__":
    main()
