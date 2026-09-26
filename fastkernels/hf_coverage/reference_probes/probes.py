"""Standalone probes for known HF Transformers reference bugs.

Each probe builds a small seeded random-init model (or the exact native method
under test) from the HF classes in ``--hf-source``, runs the triggering
computation and writes one JSON whose ``verdict`` is one of:

  execution_error              the HF reference raises in the probed setting
  wrong_computation            it runs but produces a verifiably wrong value
  reference_selection_failure  no runnable HF reference matches the published model
  ok                           the known bug did not reproduce
  probe_error                  the probe failed before reaching the known trigger

A reproduced HF bug is the expected outcome and exits 0; only probe_error exits 1.
Run as a script (not ``-m``) so ``--hf-source`` is imported before any other
Transformers installation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types

torch = None
BF16 = FP32 = None

DSV4_BASE = dict(
    architectures=["DeepseekV4ForCausalLM"], attention_bias=False, attention_dropout=0.0,
    bos_token_id=0, eos_token_id=1, expert_dtype="fp8", hc_eps=1e-6, hc_mult=4,
    hc_sinkhorn_iters=20, head_dim=512, hidden_act="silu", hidden_size=4096, index_head_dim=128,
    index_n_heads=64, index_topk=512, initializer_range=0.02, max_position_embeddings=1048576,
    moe_intermediate_size=2048, n_routed_experts=256, n_shared_experts=1, norm_topk_prob=True,
    num_attention_heads=64, num_experts_per_tok=6, num_hidden_layers=43, num_hash_layers=3,
    num_key_value_heads=1, num_nextn_predict_layers=1, o_groups=8, o_lora_rank=1024,
    q_lora_rank=1024, qk_rope_head_dim=64, rms_norm_eps=1e-6,
    rope_scaling=dict(beta_fast=32, beta_slow=1, factor=16,
                      original_max_position_embeddings=65536, type="yarn"),
    rope_theta=10000, routed_scaling_factor=1.5, scoring_func="sqrtsoftplus", sliding_window=128,
    swiglu_limit=10.0, tie_word_embeddings=False, topk_method="noaux_tc", dtype="bfloat16",
    use_cache=True, vocab_size=129280, compress_rope_theta=160000,
    compress_ratios=[0, 0] + [4, 128] * 20 + [4, 0],
)
DSV4_CAUSAL = dict(
    vocab_size=512, hidden_size=256, moe_intermediate_size=128, num_hidden_layers=5,
    num_attention_heads=8, head_dim=128, qk_rope_head_dim=16, partial_rotary_factor=0.125,
    q_lora_rank=128, n_routed_experts=8, o_groups=8, o_lora_rank=128, index_n_heads=8,
    index_head_dim=128, index_topk=16,
)
DSV4_WORKLOAD = dict(
    hidden_size=512, moe_intermediate_size=256, num_hidden_layers=5, num_attention_heads=8,
    o_groups=1, o_lora_rank=1024, q_lora_rank=128, index_n_heads=8, vocab_size=1024,
    compress_ratios=[0, 0, 4, 128, 4],
)
MRA = dict(
    approx_mode="full", block_per_row=4, hidden_act="gelu", hidden_size=128,
    initial_prior_diagonal_n_blocks=0, initial_prior_first_n_blocks=0, intermediate_size=256,
    layer_norm_eps=1e-5, max_position_embeddings=512, num_attention_heads=2, num_hidden_layers=1,
    pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type="absolute",
    type_vocab_size=1, vocab_size=1024,
)
REFORMER = dict(
    attention_head_size=64, attn_layers=["local"] * 4, axial_norm_std=1.0, axial_pos_embds=True,
    axial_pos_shape=[4, 25], axial_pos_embds_dim=[16, 16], chunk_size_lm_head=0, eos_token_id=2,
    feed_forward_size=32, hash_seed=0, hidden_act="gelu", hidden_size=32, is_decoder=False,
    layer_norm_eps=1e-12, local_num_chunks_before=1, local_num_chunks_after=0,
    local_attn_chunk_length=4, max_position_embeddings=512, num_attention_heads=2, num_hashes=1,
    vocab_size=1000, tie_word_embeddings=False, pad_token_id=0,
)
GROUNDING_IDS = [101, 1037, 4937, 1012, 1037, 6556, 2491, 1012, 102]  # "[CLS] a cat. a remote control. [SEP]"
CLVP_ENCODER = dict(hidden_size=128, intermediate_size=256, projection_dim=128,
                    num_hidden_layers=2, num_attention_heads=2)
CLVP = dict(
    text_config=CLVP_ENCODER | dict(vocab_size=256, bos_token_id=255, eos_token_id=0, pad_token_id=0),
    speech_config=CLVP_ENCODER | dict(vocab_size=8192, bos_token_id=0, eos_token_id=2, pad_token_id=1),
    decoder_config=dict(vocab_size=8194, max_position_embeddings=128, max_text_tokens=64,
                        hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                        num_mel_attn_blocks=2, bos_token_id=8192, eos_token_id=8193,
                        decoder_fixing_codes=[83, 45, 45, 248]),
    projection_dim=128, logit_scale_init_value=2.6592,
)
SAM_HQ = dict(
    vision_config=dict(
        attention_dropout=0.0, global_attn_indexes=[2], hidden_act="gelu", hidden_size=192,
        image_size=256, initializer_range=1e-10, layer_norm_eps=1e-6, mlp_dim=768, mlp_ratio=4.0,
        num_attention_heads=12, num_channels=3, num_hidden_layers=3, num_pos_feats=64,
        output_channels=128, patch_size=16, qkv_bias=True, use_abs_pos=True, use_rel_pos=True,
        window_size=14),
    prompt_encoder_config=dict(hidden_act="gelu", hidden_size=128, image_embedding_size=16,
                               image_size=256, layer_norm_eps=1e-6, mask_input_channels=16,
                               num_point_embeddings=4, patch_size=16),
    mask_decoder_config=dict(attention_downsample_rate=2, hidden_act="relu", hidden_size=128,
                             iou_head_depth=3, iou_head_hidden_dim=128, layer_norm_eps=1e-6,
                             mlp_dim=512, num_attention_heads=8, num_hidden_layers=2,
                             num_multimask_outputs=3, vit_dim=192),
    initializer_range=0.02,
)
PHI4_VISION = dict(hidden_size=144, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                   image_size=448, patch_size=14, hidden_act="gelu_pytorch_tanh",
                   layer_norm_eps=1e-6, crop_size=448, feature_layer=-2)
IBM_GRANITE_REVISION = "bf108f36960fb4df79bf035e506c592f4ee3c2d3"
IBM_GRANITE = dict(  # ibm-granite/granite-4.0-3b-vision config.json without image_grid_pinpoints/auto_map
    architectures=["Granite4VisionForConditionalGeneration"], spatial_target_layers=[12, 15, 18, 21],
    spatial_stride=2, spatial_vision_layer=-1, downsample_rate="4/8", dtype="bfloat16",
    image_seq_length=576, image_token_index=100352, initializer_range=0.02,
    projector_dropout=0.1, projector_hidden_act="gelu", tie_word_embeddings=True,
    use_spatial_sampling=True, use_image_newline_parameter=True,
    vision_feature_select_strategy="full", deepstack_layer_map=[[-19, 9], [-13, 6], [-7, 3], [-1, 0]],
    text_config=dict(
        architectures=["GraniteMoeHybridForCausalLM"], model_type="granitemoehybrid",
        attention_bias=False, attention_dropout=0.0, attention_multiplier=0.015625,
        bos_token_id=100257, eos_token_id=100257, pad_token_id=100256, embedding_multiplier=12,
        hidden_act="silu", hidden_size=2560, initializer_range=0.1, intermediate_size=8192,
        layer_types=["attention"] * 40, logits_scaling=10, mamba_chunk_size=256,
        mamba_conv_bias=True, mamba_d_conv=4, mamba_d_head=40, mamba_d_state=256, mamba_expand=2,
        mamba_n_groups=1, mamba_n_heads=128, mamba_proj_bias=False,
        max_position_embeddings=131072, normalization_function="rmsnorm", num_attention_heads=40,
        num_experts_per_tok=0, num_hidden_layers=40, num_key_value_heads=8, num_local_experts=0,
        output_router_logits=False, position_embedding_type="rope", residual_multiplier=0.22,
        rms_norm_eps=1e-5, rope_scaling=None, rope_theta=10000000, router_aux_loss_coef=0.01,
        shared_intermediate_size=8192, tie_word_embeddings=True, use_cache=False, vocab_size=100353),
    vision_config=dict(
        model_type="siglip_vision_model", attention_dropout=0.0, hidden_act="gelu_pytorch_tanh",
        hidden_size=1152, image_size=384, intermediate_size=4304, layer_norm_eps=1e-6,
        num_attention_heads=16, num_channels=3, num_hidden_layers=27, patch_size=16),
)


def failure(exc: BaseException) -> dict:
    """Exception summary plus the innermost Transformers (preferably model-file) frame that raised it."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "/transformers/" in f.filename]
    frames = [f for f in frames if "/transformers/models/" in f.filename] or frames
    site = None
    if frames:
        frame = frames[-1]
        site = f"{frame.filename.split('/transformers/', 1)[1]}:{frame.lineno}"
    return {"type": type(exc).__name__, "message": str(exc), "hf_site": site}


def attempt(fn):
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - the HF failure is the evidence
        return None, failure(exc)


def max_abs(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def load(model_class, config, weights, dtype, device_map, **kwargs):
    """Native ``from_pretrained`` of in-memory weights, so HF's FP32-module rules apply."""
    model, info = model_class.from_pretrained(
        None, config=config, state_dict=dict(weights), dtype=dtype, device_map=device_map,
        local_files_only=True, output_loading_info=True, **kwargs)
    info = {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()}
    return model.eval(), info


def param_dtypes(model, *patterns) -> dict:
    return {n: str(p.dtype) for n, p in model.named_parameters() if any(s in n for s in patterns)}


def deepseek_v4(device):
    import inspect

    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, FineGrainedFP8Config

    torch.manual_seed(913)
    config = DeepseekV4Config(**(DSV4_BASE | DSV4_CAUSAL))
    config._attn_implementation = "eager"
    model = DeepseekV4ForCausalLM(config).eval().to(device)
    torch.manual_seed(42)
    ids = torch.randint(2, 512, (1, 513), device=device)
    other = ids.clone()
    other[:, 256:] = torch.randint(2, 512, (1, 257), device=device)
    traces = {}
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda m, a, y, i=i: traces.setdefault(i, []).append(y[:, 0].clone()))
    with torch.inference_mode():
        a, b = model(ids, use_cache=True), model(other, use_cache=True)
    causal = {
        "dtype": "float32", "tokens": 513, "future_changed_from": 256,
        "first_logit_delta": max_abs(a.logits[:, 0], b.logits[:, 0]),
        "prefix_logit_delta": max_abs(a.logits[:, :256], b.logits[:, :256]),
        "finite": bool(torch.isfinite(a.logits).all()),
        "cache_returned": a.past_key_values is not None,
        "layer_first_delta": {f"{i}:{config.layer_types[i]}": max_abs(*traces[i]) for i in traces},
    }
    causal["verdict"] = ("wrong_computation" if causal["prefix_logit_delta"] > 1e-4
                         or not causal["cache_returned"] else "ok")
    del model, a, b

    torch.manual_seed(431)
    config = DeepseekV4Config(**(DSV4_BASE | DSV4_WORKLOAD))
    config._attn_implementation = "eager"
    weights = DeepseekV4ForCausalLM(config).state_dict()
    for name, value in weights.items():
        if name.endswith("tid2eid"):
            for row in value:
                row.copy_(torch.randperm(config.n_routed_experts)[:config.num_experts_per_tok])
    ids = torch.randint(2, config.vocab_size, (1, 2053)).to(device)
    fp8 = dict(activation_scheme="dynamic", weight_block_size=[128, 128])
    if "scale_fmt" in inspect.signature(FineGrainedFP8Config).parameters:
        fp8["scale_fmt"] = "ue8m0"
    model, info = load(DeepseekV4ForCausalLM, config, weights, BF16, device, attn_implementation="eager",
                       quantization_config=FineGrainedFP8Config(**fp8))
    del weights
    layer = model.model.layers[0]
    strict = sorted(getattr(model, "_keep_in_fp32_modules_strict", None) or [])
    with torch.inference_mode():
        normalized = layer.post_attention_layernorm(model.model.embed_tokens(ids[:, :2]))
        _, error = attempt(lambda: layer.mlp.gate(normalized, ids[:, :2]))
        boundary = {
            "keep_in_fp32_modules_strict": strict, "mlp_layer_types": list(config.mlp_layer_types),
            "post_attention_layernorm_weight": str(layer.post_attention_layernorm.weight.dtype),
            "norm_output": str(normalized.dtype), "hash_router_weight": str(layer.mlp.gate.weight.dtype),
            "error": error, "verdict": "execution_error" if error else "ok"}
        out, error = attempt(lambda: model(ids, use_cache=True))
    workload = {
        "dtype": "bfloat16", "tokens": 2053, "fp8_quantization": fp8, "loading": info,
        "q_b_proj_class": type(layer.self_attn.q_b_proj).__name__, "error": error,
        "note": "Weights are FP8-quantized on load from seeded BF16 values, not pre-quantized as in the "
                "original prepared workload. not_run means the FP8 kernel could not be obtained; "
                "inconclusive means the forward ran but on-the-fly quantization left weights unloaded.",
    }
    if out is not None:
        workload.update(finite=bool(torch.isfinite(out.logits).all()),
                        cache_returned=out.past_key_values is not None)
    if error is None:
        incomplete = any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        workload["verdict"] = "inconclusive" if incomplete else "ok"
    else:
        workload["verdict"] = "execution_error" if error["type"] == "RuntimeError" else "not_run"
    checks = {"causality_fp32": causal, "bf16_router_boundary": boundary, "bf16_fp8_workload_forward": workload}
    verdicts = [c["verdict"] for c in checks.values()]
    verdict = next((v for v in ("execution_error", "wrong_computation") if v in verdicts), "ok")
    return {"verdict": verdict, "checks": checks}


def mra(device):
    from transformers import MraConfig, MraForMaskedLM
    from transformers.models.mra import modeling_mra as ops

    torch.manual_seed(17)
    config = MraConfig(**MRA)
    model = MraForMaskedLM(config).eval().to(device=device, dtype=BF16)
    if ops.mra_cuda_kernel is None:
        raise RuntimeError("native MRA CUDA kernel did not load (needs CUDA, kernels, ninja on PATH); "
                           "refusing to report the zero-output fallback")
    r = {"kernel_file": getattr(ops.mra_cuda_kernel, "__file__", repr(ops.mra_cuda_kernel)), "runs": {}}
    ids = torch.randint(3, config.vocab_size, (2, 512), device=device)
    for dtype in (BF16, FP32):
        model = model.to(dtype)
        seen = []
        hook = model.mra.encoder.layer[0].attention.self.register_forward_hook(
            lambda m, a, y: seen.append({"dtype": str(y[0].dtype), "nonzero": int(y[0].count_nonzero()),
                                         "finite": bool(torch.isfinite(y[0]).all())}))
        with torch.inference_mode():
            out, error = attempt(lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids)))
        hook.remove()
        r["runs"][str(dtype)] = {"attention_output": seen, "error": error,
                                 "finite": None if out is None else bool(torch.isfinite(out.logits).all())}
    bf16, fp32 = r["runs"]["torch.bfloat16"], r["runs"]["torch.float32"]
    r["error"] = bf16["error"]
    r["verdict"] = "execution_error" if bf16["error"] else "ok"
    r["fp32_control_ok"] = fp32["error"] is None and bool(fp32["finite"])
    return r


def reformer(device):
    from transformers import ReformerConfig, ReformerForMaskedLM

    torch.manual_seed(17)
    config = ReformerConfig(**REFORMER)
    weights = ReformerForMaskedLM(config).state_dict()
    ids = torch.randint(3, config.vocab_size, (2, 96), device=device)
    r = {"runs": {}}
    for dtype in (BF16, FP32):
        model, info = load(ReformerForMaskedLM, config, weights, dtype, "cpu", attn_implementation="eager")
        model.to(device)
        with torch.inference_mode():
            out, error = attempt(lambda: model(input_ids=ids))
        r["runs"][str(dtype)] = {
            "loading": info, "error": error,
            "position_and_norm_dtypes": param_dtypes(model, "position_embeddings", "attention.layer_norm"),
            "finite": None if out is None else bool(torch.isfinite(out.logits).all())}
    bf16, fp32 = r["runs"]["torch.bfloat16"], r["runs"]["torch.float32"]
    r["error"] = bf16["error"]
    r["verdict"] = "execution_error" if bf16["error"] else "ok"
    r["fp32_control_ok"] = fp32["error"] is None and bool(fp32["finite"])
    return r


def _grounding(device, key, prefix):
    modeling = importlib.import_module(f"transformers.models.{key}.modeling_{key}")
    configuration = importlib.import_module(f"transformers.models.{key}.configuration_{key}")
    model_class = getattr(modeling, prefix + "ForObjectDetection")
    torch.manual_seed(0)
    config = getattr(configuration, prefix + "Config")()
    model, info = load(model_class, config, model_class(config).state_dict(), BF16, "cpu")
    model.to(device)
    ids = torch.tensor([GROUNDING_IDS], device=device)
    inputs = {"input_ids": ids, "token_type_ids": torch.zeros_like(ids), "attention_mask": torch.ones_like(ids),
              "pixel_values": torch.randn(1, 3, 224, 224).to(device, BF16)}
    _, positions = modeling.generate_masks_with_special_tokens_and_transfer_map(ids)
    helper, kwargs = getattr(modeling, "encode_sinusoidal_position_embedding", None), {}
    if helper is None:
        helper, kwargs = modeling.get_sine_pos_embed, {"exchange_xy": False}
    embedding = helper(positions[..., None], num_pos_feats=config.d_model, **kwargs)
    reference = helper(positions[..., None].float(), num_pos_feats=config.d_model, **kwargs)
    encoder_layer = getattr(modeling, prefix + "EncoderLayer")(config).to(device)
    consumed = encoder_layer.get_text_position_embeddings(
        torch.zeros((*ids.shape, config.d_model), dtype=BF16, device=device), None, positions)
    r = {
        "loading": info, "position_helper": helper.__name__,
        "position_ids": positions.tolist(), "position_ids_dtype": str(positions.dtype),
        "position_embedding_dtype": str(embedding.dtype),
        "position_embedding_sample": embedding[0, 2, :6].tolist(),
        "max_abs_vs_float_position_ids": max_abs(embedding, reference),
        "encoder_consumes_dtype": str(consumed.dtype),
        "encoder_consumes_helper_output": bool(torch.equal(consumed, embedding)),
    }
    queries = [m for n, m in model.named_modules()
               if "text_enhancer_layer" in n and n.endswith((".query", ".q_proj"))]
    for module in queries:
        module.register_forward_pre_hook(lambda m, a: r.update(text_query_input_dtype=str(a[0].dtype),
                                                               text_query_weight_dtype=str(m.weight.dtype)))
    with torch.inference_mode():
        out, r["error"] = attempt(lambda: model(**inputs))
    if out is not None:
        r["forward"] = {"logits_shape": list(out.logits.shape),
                        "pred_boxes_finite": bool(torch.isfinite(out.pred_boxes).all())}
    if r["error"]:
        r["verdict"] = "execution_error"
    else:
        r["verdict"] = "ok" if embedding.is_floating_point() else "wrong_computation"
    return r


def grounding_dino(device):
    return _grounding(device, "grounding_dino", "GroundingDino")


def mm_grounding_dino(device):
    return _grounding(device, "mm_grounding_dino", "MMGroundingDino")


def clvp(device):
    from transformers import ClvpConfig, ClvpModelForConditionalGeneration, GenerationConfig

    torch.manual_seed(17)
    config = ClvpConfig(**CLVP)
    config._attn_implementation = "eager"
    model = ClvpModelForConditionalGeneration(config).eval().to(device)
    model.generation_config = GenerationConfig(bos_token_id=8192, eos_token_id=8193, do_sample=True,
                                               max_new_tokens=256, return_dict_in_generate=True)
    calls, masks = [], []

    def decoder_call(m, a, kw):
        embeds, positions = kw.get("inputs_embeds"), kw.get("position_ids")
        calls.append({"positions": None if positions is None else positions.tolist(),
                      "inputs_embeds": None if embeds is None else list(embeds.shape)})

    def attention_call(m, a, kw):
        if kw.get("attention_mask") is not None:
            masks.append((kw["attention_mask"] == 0).sum(-1).flatten().tolist())

    model.speech_decoder_model.register_forward_pre_hook(decoder_call, with_kwargs=True)
    model.speech_decoder_model.model.decoder.layers[0].attn.register_forward_pre_hook(
        attention_call, with_kwargs=True)
    ids = torch.tensor([[140, 166, 213, 45, 25, 72, 200]], device=device)
    with torch.inference_mode():
        model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                       input_features=torch.randn(1, 80, 16, device=device), max_new_tokens=4)
    n = calls[0]["inputs_embeds"][1]
    r = {
        "conditioning_embeddings": n,
        "conditioning_position_ids": calls[0]["positions"],
        "expected_conditioning_position_ids": [list(range(n))],
        "decode_position_ids": [c["positions"] for c in calls[1:]],
        "prefill_admitted_keys_per_query": masks[0],
        "decode_admitted_keys": [m[0] for m in masks[1:]],
    }
    r["positions_wrong"] = r["conditioning_position_ids"] != r["expected_conditioning_position_ids"]
    r["prefill_mask_causal"] = masks[0] == list(range(1, n + 1))
    r["verdict"] = "wrong_computation" if r["positions_wrong"] or not r["prefill_mask_causal"] else "ok"
    return r


def nllb_moe(device):
    from transformers.models.nllb_moe.configuration_nllb_moe import NllbMoeConfig
    from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeSparseMLP

    torch.manual_seed(913)
    config = NllbMoeConfig(d_model=4, num_experts=4, encoder_ffn_dim=8, decoder_ffn_dim=8)
    model = NllbMoeSparseMLP(config, 8).eval().to(device)
    with torch.no_grad():
        model.router.classifier.weight.zero_()
        model.router.classifier.weight[2, 0] = 10
        model.router.classifier.weight[3, 1] = 10
    calls, routed = {}, {}
    for name, expert in model.experts.items():
        expert.register_forward_hook(lambda m, a, y, name=name: calls.update({name: len(a[0])}))
    model.router.register_forward_hook(lambda m, a, y: routed.update(mask=y[0].tolist()))
    with torch.inference_mode():
        model(torch.tensor([[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]], device=device))
    selected = sorted({row.index(1) for row in routed["mask"]})
    called = sorted(int(k.split("_")[-1]) for k in calls)
    return {"router_mask": routed["mask"], "selected_experts": selected, "called_experts": called,
            "expert_call_rows": calls, "verdict": "ok" if selected == called else "wrong_computation"}


def sam_hq(device):
    from transformers import SamHQConfig, SamHQModel

    torch.manual_seed(17)
    config = SamHQConfig(**SAM_HQ)
    config._attn_implementation = "eager"
    model = SamHQModel(config).eval().to(device)
    seen = {}
    decoder = model.mask_decoder
    decoder.transformer.register_forward_pre_hook(
        lambda m, a, kw: seen.update(before=kw["image_embeddings"].clone()), with_kwargs=True)
    decoder.transformer.register_forward_hook(lambda m, a, y: seen.update(after=y[1].clone()))
    decoder.upscale_conv1.register_forward_pre_hook(lambda m, a: seen.update(conv=a[0].clone()))
    with torch.inference_mode():
        out = model(pixel_values=torch.randn(1, 3, 256, 256, device=device),
                    input_points=torch.tensor([[[[100.0, 160.0]]]], device=device))
    before = seen["before"].transpose(2, 3).reshape_as(seen["conv"])
    after = seen["after"].transpose(2, 3).reshape_as(seen["conv"])
    r = {"upscaler_input_is_pre_transformer": torch.equal(before, seen["conv"]),
         "upscaler_input_is_transformer_output": torch.equal(after, seen["conv"]),
         "pre_vs_post_transformer_max_abs": max_abs(before, after),
         "pred_masks_shape": list(out.pred_masks.shape)}
    r["verdict"] = "ok" if r["upscaler_input_is_transformer_output"] else "wrong_computation"
    return r


def phi4_multimodal(device):
    from transformers.models.phi4_multimodal.configuration_phi4_multimodal import Phi4MultimodalVisionConfig
    from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalVisionEmbeddings

    torch.manual_seed(17)
    config = Phi4MultimodalVisionConfig(**PHI4_VISION)
    side = config.image_size // config.patch_size
    mask = torch.ones(7, side, side, dtype=torch.bool, device=device)
    mask[[3, 6], :, -1] = False
    pixels = torch.randn(7, 3, config.image_size, config.image_size, device=device)
    h, w = mask.shape[-2:]
    rows, cols = mask[:, :, 0].sum(1), mask[:, 0, :].sum(1)
    expected = ((side * torch.arange(h, device=device)[None] // rows[:, None])[:, :, None] * side
                + (side * torch.arange(w, device=device)[None] // cols[:, None])[:, None, :]).flatten(1)
    r = {"crops": 7, "valid_rows": rows.tolist(), "valid_cols": cols.tolist(), "incorrect_valid_positions": {}}
    for dtype in (FP32, BF16):
        module = Phi4MultimodalVisionEmbeddings(config).to(device=device, dtype=dtype).eval()
        seen = {}
        module.position_embedding.register_forward_pre_hook(lambda m, a: seen.update(ids=a[0].clone()))
        with torch.inference_mode():
            module(pixels.to(dtype), mask)
        r["incorrect_valid_positions"][str(dtype)] = int(((seen["ids"] != expected) & mask.flatten(1)).sum())
    wrong = any(r["incorrect_valid_positions"].values())
    r["verdict"] = "wrong_computation" if wrong else "ok"
    return r


def gemma4_assistant(device):
    from transformers.generation.candidate_generator import SinglePositionMultiTokenCandidateGenerator
    from transformers.generation.utils import _speculative_sampling

    q = torch.tensor([0.6, 0.4])

    class Assistant:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            return types.SimpleNamespace(logits=q.log().reshape(1, 1, 2), last_hidden_state=torch.zeros(1, 1, 1))

    generator = object.__new__(SinglePositionMultiTokenCandidateGenerator)
    generator.num_assistant_tokens = 1
    generator.is_main_model_prefill = True
    generator.main_model_max_length = 10
    generator.assistant_model = Assistant()
    generator.target_model_input_embeddings = lambda ids: torch.zeros(1, 1, 1)
    generator.eos_token_id = None
    ids, logits = generator.get_candidates(
        torch.tensor([[1, 1]]), {},
        types.SimpleNamespace(hidden_states=(torch.zeros(1, 2, 1),), shared_kv_states={}), False, 0)
    counts = [0, 0]
    for seed in range(100):
        torch.manual_seed(seed)
        tokens, _ = _speculative_sampling(ids, logits, 1, q.log().reshape(1, 1, 2).repeat(1, 2, 1), False)
        counts[tokens[0, 0].item()] += 1
    return {
        "scope": "Native candidate generation + speculative sampling; mocked assistant forward "
                 "prescribes draft distribution q. CPU; not a whole-model run.",
        "candidate_ids": ids.tolist(), "draft_probs": logits.softmax(-1).flatten().tolist(),
        "target_probs": q.tolist(), "first_token_counts_over_100_seeds": counts,
        "reason": "argmax drafts are verified as if sampled from q; with p=q the acceptance ratio is 1, "
                  "so token 0 is always emitted instead of following (0.6, 0.4).",
        "verdict": "wrong_computation" if counts[1] == 0 else "ok",
    }


def sam3_video(device):
    from torch import nn
    from transformers.models.sam3_video.configuration_sam3_video import Sam3VideoConfig
    from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoInferenceSession, Sam3VideoModel

    config = Sam3VideoConfig()
    model = Sam3VideoModel.__new__(Sam3VideoModel)
    nn.Module.__init__(model)
    model.config = config
    for name in ("low_res_mask_size", "score_threshold_detection", "det_nms_thresh", "assoc_iou_thresh",
                 "trk_assoc_iou_thresh", "new_det_thresh", "recondition_on_trk_masks", "hotstart_delay",
                 "hotstart_unmatch_thresh", "hotstart_dup_thresh", "suppress_unmatched_only_within_hotstart",
                 "init_trk_keep_alive", "max_trk_keep_alive", "min_trk_keep_alive",
                 "suppress_overlapping_based_on_recent_occlusion_threshold",
                 "decrease_trk_keep_alive_for_empty_masklets", "fill_hole_area", "max_num_objects",
                 "recondition_every_nth_frame", "high_conf_thresh", "high_iou_thresh"):
        setattr(model, name, getattr(config, name))

    class VisionProducer(nn.Module):
        def get_vision_features(self, **kwargs):
            return None

    model.detector_model = VisionProducer()
    session = Sam3VideoInferenceSession(video=torch.zeros(9, 3, 4, 4), dtype=torch.float32)
    session.add_prompt("object")
    for obj in (10, 20):
        session.obj_id_to_idx(obj)
        session.obj_id_to_prompt_id[obj] = 0
        session.obj_first_frame_idx[obj] = 0
        session.trk_keep_alive[obj] = 30
        session.obj_id_to_score[obj] = 0.9
    session.max_obj_id = 20
    session.unmatched_frame_inds[10] = []
    masks = torch.tensor([[[1.0, 1.0, -1.0, -1.0]] * 4, [[-2.0, -2.0, 2.0, 2.0]] * 4])
    scores = torch.tensor([1.0, 3.0])
    detection = {"bbox": torch.tensor([[2.0, 0.0, 4.0, 4.0]]), "mask": masks[1:].clone(),
                 "scores": torch.tensor([0.95])}
    model.get_vision_features_for_tracker = types.MethodType(lambda self, vision_embeds: ([], []), model)
    model.run_detection = types.MethodType(lambda self, **kw: {0: detection}, model)
    model.run_tracker_propagation = types.MethodType(lambda self, **kw: (masks.clone(), scores.clone()), model)
    model._tracker_update_memories = types.MethodType(lambda self, **kw: None, model)
    with torch.inference_mode():
        for frame in range(1, 9):
            result = model(inference_session=session, frame_idx=frame)
    survivor = result.obj_id_to_mask[20]
    r = {
        "scope": "Native forward/association/hotstart/removal/output assembly over frames 1-8; neural "
                 "detector/tracker producers stubbed with distinct per-object masks. CPU.",
        "hotstart_delay": model.hotstart_delay, "hotstart_unmatch_thresh": model.hotstart_unmatch_thresh,
        "removed_object_ids": sorted(result.removed_obj_ids), "output_object_ids": list(result.object_ids),
        "object20_mask_equals_removed_object10_mask": torch.equal(survivor, masks[0:1]),
        "object20_mask_equals_own_mask": torch.equal(survivor, masks[1:2]),
        "object20_tracker_score": float(result.obj_id_to_tracker_score[20]),
        "expected_object20_tracker_score": scores[1].sigmoid().item(),
        "removed_object10_tracker_score": scores[0].sigmoid().item(),
    }
    r["verdict"] = "ok" if r["object20_mask_equals_own_mask"] else "wrong_computation"
    return r


def granite4_vision(device):
    from transformers import Granite4VisionConfig, Granite4VisionForConditionalGeneration

    r = {"scope": "Reference-selection check on the meta device; no weights and no author model execution. "
                  f"IBM's published route is trust_remote_code at revision {IBM_GRANITE_REVISION}."}

    def ibm_import():
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (  # noqa: F401
            HybridMambaAttentionDynamicCache,
        )

    _, r["ibm_remote_code_import"] = attempt(ibm_import)

    def cached_remote_class():
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        return get_class_from_dynamic_module(
            "modeling.Granite4VisionForConditionalGeneration", "ibm-granite/granite-4.0-3b-vision",
            revision=IBM_GRANITE_REVISION, local_files_only=True)

    _, error = attempt(cached_remote_class)
    r["ibm_remote_code_from_hf_cache"] = error or "loaded"

    def builtin(config):
        with torch.device("meta"):
            Granite4VisionForConditionalGeneration(config)
        return "constructed"

    text = IBM_GRANITE["text_config"]
    granite_text = {k: text[k] for k in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "attention_multiplier", "embedding_multiplier", "residual_multiplier", "logits_scaling",
        "rms_norm_eps", "rope_theta", "max_position_embeddings", "tie_word_embeddings")} | {"model_type": "granite"}
    constructed, error = attempt(lambda: builtin(Granite4VisionConfig(**(IBM_GRANITE | {"text_config": granite_text}))))
    r["builtin_with_granite_text_config_control"] = constructed or error
    constructed, error = attempt(lambda: builtin(Granite4VisionConfig()))
    r["builtin_with_class_default_config"] = constructed or error
    config = Granite4VisionConfig(**IBM_GRANITE)
    r["ibm_text_config_class"] = type(config.text_config).__name__
    _, r["builtin_with_ibm_config"] = attempt(lambda: builtin(config))
    blocked = r["ibm_remote_code_import"] is not None and r["builtin_with_ibm_config"] is not None
    r["verdict"] = "reference_selection_failure" if blocked else "ok"
    return r


PROBES = {fn.__name__: fn for fn in (
    deepseek_v4, mra, reformer, grounding_dino, mm_grounding_dino, clvp, nllb_moe, sam_hq,
    phi4_multimodal, gemma4_assistant, sam3_video, granite4_vision)}
CPU_PROBES = {"gemma4_assistant", "sam3_video", "granite4_vision"}


def source_revision(source: Path):
    try:
        return subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(source: Path, extra: list[str]) -> dict:
    import importlib.metadata as metadata
    import transformers

    versions = {}
    for name in ("huggingface_hub", "tokenizers", "safetensors", "kernels"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "hf_source": str(source), "extra_path": extra, "transformers_file": transformers.__file__,
        "transformers_version": transformers.__version__, "transformers_git_revision": source_revision(source),
        "torch": torch.__version__, "python": platform.python_version(), "dependencies": versions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(PROBES))
    parser.add_argument("--hf-source", required=True, help="Transformers checkout or its src/ directory")
    parser.add_argument("--extra-path", action="append", default=[],
                        help="directory prepended to sys.path after --hf-source (repeatable, or os.pathsep-joined)")
    parser.add_argument("--out", required=True, help="output JSON file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = Path(args.hf_source).expanduser().resolve()
    source = source / "src" if (source / "src" / "transformers").is_dir() else source
    if not (source / "transformers" / "__init__.py").is_file():
        parser.error(f"--hf-source has no transformers package: {source}")
    extra = [str(Path(p).expanduser().resolve()) for item in args.extra_path for p in item.split(os.pathsep) if p]
    paths = [str(source), *extra]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    global torch, BF16, FP32
    import torch as _torch
    torch, BF16, FP32 = _torch, _torch.bfloat16, _torch.float32
    torch.set_num_threads(4)
    device = "cpu" if args.model in CPU_PROBES else args.device
    record = {"model": args.model, "device": device, **environment(source, extra)}
    if device.startswith("cuda") and torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name()
    start = time.perf_counter()
    try:
        record.update(PROBES[args.model](device))
    except Exception as exc:  # noqa: BLE001
        record.update(verdict="probe_error", error=failure(exc), traceback=traceback.format_exc())
    record["seconds"] = round(time.perf_counter() - start, 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps({k: record.get(k) for k in ("model", "verdict", "transformers_git_revision", "seconds")}))
    return 1 if record["verdict"] == "probe_error" else 0


if __name__ == "__main__":
    sys.exit(main())
