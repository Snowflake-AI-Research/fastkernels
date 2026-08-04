"""Quantization-scheme classification for a checkpoint's quant config.

fastkernels carries the model's HF quantization config around as a plain dict
(``weight_loader._detect_quant_config``). Historically ``quant_config is not
None`` meant exactly one thing -- DeepSeek-style block-FP8 W8A8 on every linear
-- so modules tested that directly. ModelOpt NVFP4 breaks that equivalence in
two ways, and both matter for vLLM parity:

* The quantization is **W4A4 with group-16 block scales**, not block-FP8, so the
  expert kernels and the on-disk parameter set are completely different.
* The checkpoint quantizes only a **subset of modules**. vLLM honours the
  config's ``ignore`` / ``exclude_modules`` wildcard list per layer
  (``ModelOptQuantConfigBase.is_layer_excluded`` -> ``UnquantizedLinearMethod``);
  for nvidia/GLM-5.2-NVFP4 that list covers every ``self_attn*``, every
  ``mlp.shared_experts*``, the dense layers 0-2, the MTP layer 78, ``lm_head``
  and ``embed_tokens``, leaving only ``mlp.experts.*`` quantized. (The router
  ``mlp.gate`` is not in the list but is unquantized anyway: vLLM builds MoE
  gates with no ``quant_config`` at all -- ``deepseek_v2.py:308``.)

So under NVFP4 every fastkernels ``Linear`` must stay BF16 and only the routed
experts take the quantized path. :func:`linear_quant_config` is the one-line
expression of that split, used wherever a module forwards its config to a
child linear.
"""

from __future__ import annotations

from typing import Any, Mapping

# Schemes fastkernels understands. ``None`` means unquantized (BF16 weights).
FP8_BLOCK = "fp8_block"
NVFP4 = "nvfp4"


def quant_scheme(quant_config: Mapping[str, Any] | None) -> str | None:
    """Classify a checkpoint quant config into a fastkernels scheme tag.

    Returns ``NVFP4`` for a ModelOpt NVFP4 config, ``FP8_BLOCK`` for anything
    else non-None (the DeepSeek / Qwen block-FP8 checkpoints, the only other
    scheme this code path has ever seen), and ``None`` for an unquantized
    checkpoint.
    """
    if not quant_config:
        return None
    method = str(quant_config.get("quant_method", "")).lower()
    algo = str(quant_config.get("quant_algo", "")).upper()
    if method.startswith("modelopt") and algo == "NVFP4":
        return NVFP4
    return FP8_BLOCK


def is_nvfp4(quant_config: Mapping[str, Any] | None) -> bool:
    return quant_scheme(quant_config) == NVFP4


def linear_quant_config(
    quant_config: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """The config a plain ``Linear`` child should be built with.

    ``None`` under NVFP4 (every linear is in the checkpoint's exclude list, so
    vLLM gives them ``UnquantizedLinearMethod``); the config unchanged
    otherwise.
    """
    return None if is_nvfp4(quant_config) else quant_config
