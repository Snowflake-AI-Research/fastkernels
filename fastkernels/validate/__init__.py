"""End-to-end validation harnesses (fastkernels vs SOTA reference libraries).

Each ``bench_<lib>.py`` compares fastkernels against a reference library (vLLM,
SGLang, FLA, diffusers, timm, …) for a family of models. ``fastkernels validate``
(see ``_cli.py``) resolves a scenario table and runs the proper harness per model.
"""
