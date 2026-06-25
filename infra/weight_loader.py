"""
Weight loader for Llama 3.1, Llama 4, Mixtral, DeepSeek V3.2, Qwen2-VL,
Qwen3-VL, GPT-OSS, and Whisper with tensor parallelism.

Loads weights from HuggingFace safetensors and distributes them
across TP shards using the weight_loader callbacks on each parameter.

GPT-OSS uses a dedicated loader (_load_gpt_oss_weights) that keeps
expert weights in native MXFP4 packed uint8 format for Triton inference.
DeepSeek V3.2 FP8 weights are re-quantized in-place to UE8M0 scales
post-load and MLA absorbed weights are computed before inference.
"""

from __future__ import annotations

import os
import re
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

try:
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
    _HAS_FASTSAFETENSORS = True
except ImportError:
    _HAS_FASTSAFETENSORS = False

from concurrent.futures import ThreadPoolExecutor

from .tp import _tp_size
from ..tasks.baseline.L4.bitnet import BitNetConfig, BitNetForCausalLM
from ..tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from ..tasks.baseline.L4.llama_eagle3 import LlamaEagle3Config, LlamaForCausalLMEagle3
from ..tasks.baseline.L4.llama4 import Llama4Config, Llama4ForCausalLM
from ..tasks.baseline.L4.mixtral import MixtralConfig, MixtralForCausalLM
from ..tasks.baseline.L4.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from ..tasks.baseline.L4.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from ..tasks.baseline.L4.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from ..tasks.baseline.L4.deepseek import DeepSeekV3Config, DeepSeekV3ForCausalLM
from ..tasks.baseline.L4.flux import FluxConfig, FluxPipeline
from ..tasks.baseline.L4.gemma4 import Gemma4Config, Gemma4ForCausalLM
from ..tasks.baseline.L4.whisper import WhisperConfig, WhisperForConditionalGeneration
from ..tasks.baseline.L4.cosyvoice3 import CosyVoice3Config, CosyVoice3ForTTS


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def download_model(model_name: str) -> str:
    # Accept an already-staged local directory (used by the
    # ``diff_deepseek_layers`` diagnostic and other fixtures that feed a
    # truncated checkpoint to the engine).  Without this short-circuit,
    # ``snapshot_download`` rejects the absolute path as an invalid
    # ``org/repo`` id, which makes TP>1 worker processes fail on import
    # since they do not inherit any parent-side monkey-patches.
    if os.path.isdir(model_name):
        return model_name
    return snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )


_EXPERT_RE = re.compile(
    r"(.+\.block_sparse_moe)\.experts\.(\d+)\.(w[123])\.weight"
)

# DeepSeek-V3 per-expert weight and scale pattern
_DEEPSEEK_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight_scale_inv|weight)$"
)

_DEEPSEEK_SHARED_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.shared_experts\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)"
)

# Qwen3-MoE fused expert weight patterns: gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter]
_QWEN3_MOE_FUSED_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)$"
)
_QWEN3_MOE_FUSED_SCALE_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)_scale_inv$"
)

# Qwen3-MoE gate (router) weight
_QWEN3_MOE_GATE_RE = re.compile(r"(.+\.mlp)\.gate\.weight$")


# Qwen2-VL weight name remapping: checkpoint -> model parameter
# Checkpoint uses `model.` prefix for language, `visual.` for vision.
# Our model uses `model.` for language backbone and `visual.` for vision.
_QWEN2_VL_PREFIX_MAP = {
    "lm_head.": "lm_head.",
    "model.": "model.",
    "visual.": "visual.",
}

# Qwen3-VL weight name remapping: checkpoint uses `model.language_model.`
# and `model.visual.` prefixes.
_QWEN3_VL_PREFIX_MAP = {
    "model.language_model.": "model.",
    "model.visual.": "visual.",
    "lm_head.": "lm_head.",
}

# Vision encoder merged QKV pattern
_VISION_QKV_RE = re.compile(r"(.+\.attn)\.qkv\.(weight|bias)")

# Qwen3-VL vision MLP uses linear_fc1/linear_fc2 -> fc1/fc2
_QWEN3_VISION_MLP_RE = re.compile(r"(.+\.mlp)\.linear_fc([12])\.(weight|bias)")

# Qwen3-VL merger uses linear_fc1/linear_fc2 -> fc1/fc2
_QWEN3_MERGER_FC_RE = re.compile(r"(.+)\.(linear_fc1|linear_fc2)\.(weight|bias)")

# Qwen2-VL merger remapping: ln_q -> norm, mlp.0 -> fc1, mlp.2 -> fc2
_QWEN2_MERGER_RE = re.compile(r"(visual\.merger)\.(ln_q|mlp\.0|mlp\.2)\.(weight|bias)")

# Qwen3-VL learned pos embed: visual.pos_embed.weight -> visual.pos_embed_interp.pos_embed
_VISION_POS_EMBED_RE = re.compile(r"visual\.pos_embed\.weight$")

# L1 wrapper nesting: *.embed_tokens.weight / *.lm_head.weight -> *.embedding_op.emb.weight
_EMBED_WEIGHT_RE = re.compile(
    r"((?:model\.)?(?:embed_tokens|lm_head))\.weight$"
)

# L1 wrapper nesting: patch_embed.proj.X -> patch_embed.proj.conv.X
_VISION_PATCH_EMBED_RE = re.compile(r"(visual\.patch_embed\.proj)\.(weight|bias)")


# Llama4 fused expert weight patterns
_LLAMA4_FUSED_EXPERT_RE = re.compile(
    r"(.+\.feed_forward)\.experts\.(gate_up_proj|down_proj)"
)

_GEMMA4_FUSED_EXPERT_RE = re.compile(
    r"(.+\.moe)\.(gate_up_proj|down_proj)$"
)
_GEMMA4_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _permute_qk_for_rotary(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute Q/K weights from interleaved to contiguous layout for rotary."""
    f_out, f_in = weight.shape
    return (
        weight.view(n_heads, f_out // n_heads // 2, 2, f_in)
        .transpose(1, 2)
        .reshape(f_out, f_in)
    )


def _permute_bitnet_qk_to_sota(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute BitNet HF Q/K weights to Microsoft GPU's interleaved layout."""
    f_out, f_in = weight.shape
    return (
        weight.view(n_heads, 2, f_out // n_heads // 2, f_in)
        .transpose(1, 2)
        .reshape(f_out, f_in)
    )


def _remap_qwen2_vl_name(name: str) -> str:
    """Remap Qwen2-VL checkpoint weight names to our model parameter names."""
    for prefix, replacement in _QWEN2_VL_PREFIX_MAP.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name


def _remap_qwen3_vl_name(name: str) -> str:
    """Remap Qwen3-VL checkpoint weight names to our model parameter names."""
    for prefix, replacement in _QWEN3_VL_PREFIX_MAP.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name


def _remap_qwen2_5_omni_name(name: str) -> str | None:
    """Map Qwen2.5-Omni Thinker weights to fastkernels module names."""
    if name.startswith(("talker.", "token2wav.")):
        return None
    if name.startswith("thinker.lm_head."):
        return "lm_head." + name[len("thinker.lm_head."):]
    if name.startswith("thinker.model."):
        return "model." + name[len("thinker.model."):]
    if name.startswith("thinker.visual."):
        return "visual." + name[len("thinker.visual."):]
    if name.startswith("thinker.audio_tower."):
        return "audio_tower." + name[len("thinker.audio_tower."):]
    return None


def _load_vision_qkv(model, param_prefix: str, loaded_weight: torch.Tensor,
                      wb: str) -> int:
    """Load merged vision QKV weight/bias into our QKVParallelLinear.

    The checkpoint stores Q, K, V concatenated as a single tensor.
    We split and load as separate shards.
    """
    total_dim = loaded_weight.shape[0]
    per_head = total_dim // 3
    q, k, v = loaded_weight.narrow(0, 0, per_head), \
               loaded_weight.narrow(0, per_head, per_head), \
               loaded_weight.narrow(0, 2 * per_head, per_head)
    param_name = f"{param_prefix}.qkv.{wb}"
    try:
        param = model.get_parameter(param_name)
    except AttributeError:
        return 0
    wl = param.weight_loader
    wl(param, q, "q")
    wl(param, k, "k")
    wl(param, v, "v")
    return 1


_FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def _dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                     dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize MXFP4 weights to dense dtype (legacy fallback).

    For GPT-OSS, prefer keeping weights in MXFP4 format and using the
    native Triton MXFP4 MoE kernel. This function is only used by models
    that don't support native MXFP4 inference.
    """
    low = (blocks & 0x0F).long()
    high = ((blocks >> 4) & 0x0F).long()
    unpacked = torch.stack([low, high], dim=-1)
    unpacked = unpacked.reshape(*blocks.shape[:-1], 32)

    values = _FP4_E2M1_LUT[unpacked]

    scale_float = torch.pow(2.0, scales.float() - 127.0)

    values = values * scale_float.unsqueeze(-1)

    batch_shape = values.shape[:-2]
    values = values.reshape(*batch_shape, -1)
    return values.to(dtype)


_GPT_OSS_EXPERT_RE = re.compile(
    r"(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)_(blocks|scales|bias)"
)


def _load_gpt_oss_weights(model, model_path: str) -> None:
    """Load GPT-OSS weights keeping expert weights in native MXFP4 format.

    Expert blocks/scales are loaded directly as packed uint8 tensors into
    the model's MXFP4 parameters. No dequantization is performed.
    """
    import gc

    packed = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    print(f"  Loading GPT-OSS weights from {len(safetensor_files)} safetensors file(s)...")

    # Map checkpoint names to model parameter names:
    #   model.layers.{i}.mlp.experts.gate_up_proj_blocks -> ...mlp.w13_weight
    #   model.layers.{i}.mlp.experts.gate_up_proj_scales -> ...mlp.w13_weight_scale
    #   model.layers.{i}.mlp.experts.gate_up_proj_bias   -> ...mlp.w13_bias
    #   model.layers.{i}.mlp.experts.down_proj_blocks    -> ...mlp.w2_weight
    #   model.layers.{i}.mlp.experts.down_proj_scales    -> ...mlp.w2_weight_scale
    #   model.layers.{i}.mlp.experts.down_proj_bias      -> ...mlp.w2_bias
    _EXPERT_MAP = {
        ("gate_up_proj", "blocks"): "w13_weight",
        ("gate_up_proj", "scales"): "w13_weight_scale",
        ("gate_up_proj", "bias"): "w13_bias",
        ("down_proj", "blocks"): "w2_weight",
        ("down_proj", "scales"): "w2_weight_scale",
        ("down_proj", "bias"): "w2_bias",
    }

    loaded = 0

    for sf_file in safetensor_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for weight_name in f.keys():
                m = _GPT_OSS_EXPERT_RE.match(weight_name)
                if m:
                    prefix, proj, part = m.groups()
                    param_suffix = _EXPERT_MAP.get((proj, part))
                    if param_suffix is None:
                        continue
                    param_name = f"{prefix.replace('mlp.experts', 'mlp')}.{param_suffix}"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    tensor = f.get_tensor(weight_name)
                    param.weight_loader(param, tensor)
                    loaded += 1
                    continue

                # Non-expert weights
                tensor = f.get_tensor(weight_name)
                matched = False
                for orig_key, (packed_name, shard_id) in packed.items():
                    if orig_key in weight_name:
                        param_name = weight_name.replace(orig_key, packed_name)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            break
                        param.weight_loader(param, tensor, shard_id)
                        loaded += 1
                        matched = True
                        break
                if matched:
                    continue

                if "rotary_emb" in weight_name:
                    continue

                # Remap checkpoint names to model parameter names where
                # the module hierarchy differs (VocabParallelEmbedding
                # wraps the weight inside embedding_op.emb).
                mapped_name = weight_name
                _PARAM_REMAP = {
                    "model.embed_tokens.weight": "model.embed_tokens.embedding_op.emb.weight",
                    "lm_head.weight": "lm_head.embedding_op.emb.weight",
                }
                if weight_name in _PARAM_REMAP:
                    mapped_name = _PARAM_REMAP[weight_name]

                try:
                    param = model.get_parameter(mapped_name)
                except AttributeError:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
                loaded += 1

    gc.collect()
    print(f"  Loaded {loaded} weight shards.")


_WEIGHT_SCALE_INV_RE = re.compile(r"(.+)\.weight_scale_inv$")

# Whisper: remap checkpoint names
# Strip "model." prefix, fc1/fc2 -> mlp.fc1/mlp.fc2,
# conv -> conv.conv (L1 wrapper), layernorm -> layernorm.norm (L1 wrapper)
_WHISPER_FC_RE = re.compile(
    r"((?:encoder|decoder)\.layers\.\d+)\.fc([12])\.(weight|bias)"
)
_WHISPER_CONV_RE = re.compile(r"(encoder\.conv[12])\.(weight|bias)")
_WHISPER_LAYER_NORM_RE = re.compile(
    r"((?:encoder|decoder)(?:\.layers\.\d+)?\.(?:self_attn_layer_norm|"
    r"encoder_attn_layer_norm|final_layer_norm|layer_norm))\.(weight|bias)"
)
_WHISPER_EMBED_RE = re.compile(
    r"((?:encoder|decoder)\.embed_(?:positions|tokens))\.weight"
)
_WHISPER_OUT_PROJ_RE = re.compile(
    r"((?:encoder|decoder)\.layers\.\d+\.(?:self_attn|encoder_attn))\.out_proj\.(weight|bias)"
)


def _dequant_fp8_block(tensor: torch.Tensor, scale_inv: torch.Tensor,
                       block_size: int = 128) -> torch.Tensor:
    """Dequantize FP8 block-quantized tensor: out = fp8_val * scale_inv (per block).

    Each block of block_size elements along each non-batch dim shares one scale factor.
    Supports 2D [R, C] and 3D [E, R, C] tensors.
    """
    shape = tensor.shape
    ndim = len(shape)
    if ndim == 3:
        E, R, C = shape
        _, sR, sC = scale_inv.shape
        bR = (R + sR - 1) // sR
        bC = (C + sC - 1) // sC
        out = torch.zeros(E, sR * bR, sC * bC, dtype=torch.bfloat16, device=tensor.device)
        out[:, :R, :C] = tensor.to(torch.bfloat16)
        out = out.reshape(E, sR, bR, sC, bC) * scale_inv[:, :, None, :, None]
        return out.reshape(E, sR * bR, sC * bC)[:, :R, :C].contiguous()
    elif ndim == 2:
        R, C = shape
        sR, sC = scale_inv.shape
        bR = (R + sR - 1) // sR
        bC = (C + sC - 1) // sC
        out = torch.zeros(sR * bR, sC * bC, dtype=torch.bfloat16, device=tensor.device)
        out[:R, :C] = tensor.to(torch.bfloat16)
        out = out.reshape(sR, bR, sC, bC) * scale_inv[:, None, :, None]
        return out.reshape(sR * bR, sC * bC)[:R, :C].contiguous()
    else:
        raise ValueError(f"Unsupported tensor ndim={ndim} for FP8 dequantization")


def _threaded_safetensors_iterator(safetensor_files):
    """Yield (weight_name, tensor) with threaded pre-loading of safetensors files."""
    def _load_one(path):
        tensors = {}
        with safe_open(path, "pt", "cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors

    with ThreadPoolExecutor(max_workers=min(4, len(safetensor_files))) as pool:
        futures = [pool.submit(_load_one, sf) for sf in sorted(safetensor_files)]
        for fut in futures:
            tensors = fut.result()
            for k, v in tensors.items():
                yield k, v
            del tensors


def _fastsafetensors_iterator(safetensor_files):
    """Yield (weight_name, tensor) using fastsafetensors GPU-direct loading.

    Mirrors vLLM's ``fastsafetensors_weights_iterator``
    (vllm/model_executor/model_loader/weight_utils.py:942) which loads
    ``pg.size()`` files in parallel across the TP process group.  Each
    rank reads one file at a time, then broadcasts/redistributes via
    fastsafetensors' internal collectives.  With TP=8 and 21 shards this
    is 8x faster than the previous SingleGroup loop that made every rank
    sequentially read every file.
    """
    if torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    sorted_files = sorted(safetensor_files)
    world_size = pg.size()
    file_sub_lists = [
        sorted_files[i:i + world_size]
        for i in range(0, len(sorted_files), world_size)
    ]
    # vLLM disables GDS for TP>1 to avoid creating CUDA contexts on every
    # visible GPU (cuFileDriverOpen side-effect).  Match that.
    nogds = world_size > 1

    for f_list in file_sub_lists:
        loader = SafeTensorsFileLoader(pg, device, nogds=nogds)
        rank_file_map = {i: [f] for i, f in enumerate(f_list)}
        loader.add_filenames(rank_file_map)
        try:
            try:
                fb = loader.copy_files_to_device()
            except RuntimeError as e:
                msg = str(e).lower()
                if nogds or ("gds" not in msg and "cufile" not in msg):
                    raise
                loader.close()
                nogds = True
                loader = SafeTensorsFileLoader(pg, device, nogds=nogds)
                loader.add_filenames(rank_file_map)
                fb = loader.copy_files_to_device()
            try:
                for k in list(fb.key_to_rank_lidx.keys()):
                    yield k, fb.get_tensor(k)
            finally:
                fb.close()
        finally:
            loader.close()


def _assign_fused_expert(model, key, tensor, scale):
    """Assign a single fused expert weight+scale to the model, then free them.

    Handles FP8 TP-sharded assignment with scale transposition.
    Called as soon as both weight and scale are available for a given
    (mlp_prefix, proj) key, avoiding buffering all layers simultaneously.
    """
    from .tp import _tp_rank
    mlp_prefix, proj = key
    rank = _tp_rank() if _tp_size() > 1 else 0
    is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if proj == "gate_up_proj":
        param_name = f"{mlp_prefix}.w13"
    else:
        param_name = f"{mlp_prefix}.w2"
    try:
        param = model.get_parameter(param_name)
    except AttributeError:
        return

    model_is_fp8 = param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if is_fp8 and model_is_fp8 and scale is not None:
        weight = tensor.transpose(-1, -2)
        if proj == "gate_up_proj":
            full_inter_2 = weight.shape[1]
            half = full_inter_2 // 2
            tp = param.shape[1] // 2
            gate = weight[:, :half, :]
            up = weight[:, half:, :]
            param.data[:, :tp, :].copy_(gate[:, rank * tp:(rank + 1) * tp, :])
            param.data[:, tp:, :].copy_(up[:, rank * tp:(rank + 1) * tp, :])

            scale_param_name = f"{mlp_prefix}.w13_scale"
            try:
                scale_param = model.get_parameter(scale_param_name)
            except AttributeError:
                pass
            else:
                s = scale.transpose(1, 2)
                full_scale_rows = s.shape[1]
                gate_scale_rows = full_scale_rows // 2
                shard_scale_rows = scale_param.data.shape[1] // 2
                gate_s = s[:, rank * shard_scale_rows:(rank + 1) * shard_scale_rows, :]
                up_s = s[:, gate_scale_rows + rank * shard_scale_rows:gate_scale_rows + (rank + 1) * shard_scale_rows, :]
                scale_param.data[:, :shard_scale_rows, :].copy_(gate_s)
                scale_param.data[:, shard_scale_rows:2 * shard_scale_rows, :].copy_(up_s)
        else:
            full_inter = weight.shape[2]
            tp_inter = param.shape[2]
            param.data.copy_(weight[:, :, rank * tp_inter:(rank + 1) * tp_inter])

            scale_param_name = f"{mlp_prefix}.w2_scale"
            try:
                scale_param = model.get_parameter(scale_param_name)
            except AttributeError:
                pass
            else:
                s = scale.transpose(1, 2)
                shard_scale_cols = scale_param.data.shape[2]
                scale_param.data.copy_(s[:, :, rank * shard_scale_cols:(rank + 1) * shard_scale_cols])
    else:
        if is_fp8 and scale is not None:
            tensor = _dequant_fp8_block(tensor, scale, block_size=128)
        weight = tensor.transpose(-1, -2)
        if proj == "gate_up_proj":
            full_inter_2 = weight.shape[1]
            half = full_inter_2 // 2
            tp = param.shape[1] // 2
            gate = weight[:, :half, :]
            up = weight[:, half:, :]
            param.data[:, :tp, :].copy_(gate[:, rank * tp:(rank + 1) * tp, :])
            param.data[:, tp:, :].copy_(up[:, rank * tp:(rank + 1) * tp, :])
        else:
            full_inter = weight.shape[2]
            tp_inter = param.shape[2]
            param.data.copy_(weight[:, :, rank * tp_inter:(rank + 1) * tp_inter])


def _assign_gemma4_fused_expert(model, prefix: str, proj: str,
                                tensor: torch.Tensor) -> None:
    """Load Gemma4 3D expert tensors directly into fused expert parameters."""
    from .tp import _tp_rank

    rank = _tp_rank() if _tp_size() > 1 else 0
    if proj == "gate_up_proj":
        param = model.get_parameter(f"{prefix}.w13")
        weight = tensor
        if weight.shape[1] == model.config.hidden_size:
            weight = weight.transpose(-1, -2).contiguous()
        full_inter = weight.shape[1] // 2
        shard = param.shape[1] // 2
        gate = weight[:, :full_inter, :]
        up = weight[:, full_inter:, :]
        param.data[:, :shard, :].copy_(
            gate[:, rank * shard:(rank + 1) * shard, :],
        )
        param.data[:, shard:, :].copy_(
            up[:, rank * shard:(rank + 1) * shard, :],
        )
    else:
        param = model.get_parameter(f"{prefix}.w2")
        weight = tensor
        if weight.shape[1] != model.config.hidden_size:
            weight = weight.transpose(-1, -2).contiguous()
        shard = param.shape[2]
        param.data.copy_(weight[:, :, rank * shard:(rank + 1) * shard])


def _weights_iterator(safetensor_files, use_fastsafetensors=True):
    """Iterate over (weight_name, tensor) from safetensor files.

    Uses fastsafetensors (GPU Direct Storage) when available for fast
    GPU-direct loading. Falls back to safetensors CPU loading otherwise.
    Matches vllm's fastsafetensors_weights_iterator / safetensors_weights_iterator.
    """
    if use_fastsafetensors and _HAS_FASTSAFETENSORS:
        import torch.distributed as dist
        if dist.is_initialized():
            pg = dist.group.WORLD
        else:
            pg = SingleGroup()
        device = torch.device(f"cuda:{pg.rank()}")
        batch_size = pg.size()
        file_batches = [
            safetensor_files[i:i + batch_size]
            for i in range(0, len(safetensor_files), batch_size)
        ]
        nogds = False
        for f_list in file_batches:
            loader = SafeTensorsFileLoader(pg, device, nogds=nogds)
            rank_file_map = {i: [f] for i, f in enumerate(f_list)}
            loader.add_filenames(rank_file_map)
            try:
                try:
                    fb = loader.copy_files_to_device()
                except RuntimeError as e:
                    if "gds" not in str(e):
                        raise
                    loader.close()
                    nogds = True
                    loader = SafeTensorsFileLoader(pg, device, nogds=nogds)
                    loader.add_filenames(rank_file_map)
                    fb = loader.copy_files_to_device()
                try:
                    for k in list(fb.key_to_rank_lidx.keys()):
                        yield k, fb.get_tensor(k)
                finally:
                    fb.close()
            finally:
                loader.close()
    else:
        for sf_file in safetensor_files:
            with safe_open(sf_file, "pt", "cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)


def load_weights(model, model_path: str, model_type: str = "llama") -> None:
    """Load weights with support for packed modules, MoE experts, vision
    encoder QKV, FP8 weight_scale_inv, and TP sharding.

    Uses fastsafetensors (GPU Direct Storage) when available to load tensors
    directly to GPU, matching vllm's fastsafetensors_weights_iterator.
    """
    packed = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    is_whisper = model_type == "whisper"
    is_qwen2_vl = model_type == "qwen2_vl"
    is_qwen3_vl = model_type in ("qwen3_vl", "qwen3_vl_moe")
    is_qwen3_vl_moe = model_type == "qwen3_vl_moe"
    is_qwen2_5_omni = model_type == "qwen2_5_omni"
    is_qwen_vl = is_qwen2_vl or is_qwen3_vl or is_qwen2_5_omni
    is_llama4 = model_type == "llama4"
    is_gemma4 = model_type == "gemma4"
    is_mamba = model_type in ("mamba", "mamba2")
    if is_llama4:
        llama4_config = model.config
    if is_gemma4:
        gemma4_config = model.config
        gemma4_k_eq_v_layers = {
            i for i, layer_type in enumerate(gemma4_config.layer_types)
            if layer_type == "full_attention"
            and getattr(gemma4_config, "attention_k_eq_v", False)
        }

    import time as _time
    _t_load = _time.perf_counter()
    if _HAS_FASTSAFETENSORS:
        print(f"  Loading weights from {len(safetensor_files)} safetensors file(s) "
              f"[fastsafetensors GPU-direct]...", flush=True)
    elif len(safetensor_files) > 1:
        print(f"  Loading weights from {len(safetensor_files)} safetensors file(s) "
              f"[threaded]...", flush=True)
    else:
        print(f"  Loading weights from {len(safetensor_files)} safetensors file(s)...",
              flush=True)
    loaded = 0
    _last_report = _t_load
    _fused_expert_weights = {}
    _fused_expert_scales = {}

    if _HAS_FASTSAFETENSORS:
        _weight_iter = _fastsafetensors_iterator(safetensor_files)
    elif len(safetensor_files) > 1:
        _weight_iter = _threaded_safetensors_iterator(safetensor_files)
    else:
        def _std_iter():
            for sf_file in safetensor_files:
                with safe_open(sf_file, "pt", "cpu") as f:
                    for wn in f.keys():
                        yield wn, f.get_tensor(wn)
        _weight_iter = _std_iter()

    for weight_name, _loaded_tensor in _weight_iter:
        def _get_tensor(_t=_loaded_tensor):
            return _t
        now = _time.perf_counter()
        if now - _last_report > 5.0:
            print(f"    {loaded} shards loaded ({now - _t_load:.1f}s)", flush=True)
            _last_report = now
        # Remap checkpoint names for Qwen VL models
        if is_qwen2_vl:
            mapped_name = _remap_qwen2_vl_name(weight_name)
        elif is_qwen3_vl:
            mapped_name = _remap_qwen3_vl_name(weight_name)
        elif is_qwen2_5_omni:
            mapped_name = _remap_qwen2_5_omni_name(weight_name)
            if mapped_name is None:
                continue
        else:
            mapped_name = weight_name

        # Mamba / Mamba2 checkpoint remaps:
        #  - A_log -> A   (we negate-exp at load time via the param's
        #    weight_loader, matching vLLM's MambaMixer/MambaMixer2)
        #  - backbone.embeddings.weight -> backbone.embeddings.embedding_op.emb.weight
        #    (fastkernels's VocabParallelEmbedding wraps the weight inside
        #    embedding_op.emb)
        if is_mamba:
            if "A_log" in mapped_name:
                mapped_name = mapped_name.replace("A_log", "A")
            if mapped_name == "backbone.embeddings.weight":
                mapped_name = "backbone.embeddings.embedding_op.emb.weight"

        if model_type == "deepseek_v3":
            mapped_name = mapped_name.replace(
                ".shared_experts.", ".shared_expert.")
            mapped_name = mapped_name.replace(
                ".mlp.gate.weight", ".mlp.gate_weight")
            # The router's ``e_score_correction_bias`` lives directly on the
            # ``DeepSeekMoE`` module (not under a ``.gate`` submodule as in
            # the HuggingFace checkpoint), so strip the ``.gate`` segment.
            mapped_name = mapped_name.replace(
                ".mlp.gate.e_score_correction_bias",
                ".mlp.e_score_correction_bias")

        # Whisper: remap checkpoint names
        if is_whisper:
            if mapped_name.startswith("model."):
                mapped_name = mapped_name[len("model."):]
            if mapped_name.startswith("proj_out."):
                continue
            m_fc = _WHISPER_FC_RE.match(mapped_name)
            if m_fc:
                prefix, fc_num, wb = m_fc.groups()
                mapped_name = f"{prefix}.mlp.fc{fc_num}.{wb}"
            m_conv = _WHISPER_CONV_RE.match(mapped_name)
            if m_conv:
                prefix, wb = m_conv.groups()
                mapped_name = f"{prefix}.conv.{wb}"
            m_ln = _WHISPER_LAYER_NORM_RE.match(mapped_name)
            if m_ln:
                prefix, wb = m_ln.groups()
                mapped_name = f"{prefix}.norm.{wb}"
            m_emb = _WHISPER_EMBED_RE.match(mapped_name)
            if m_emb:
                prefix = m_emb.group(1)
                mapped_name = f"{prefix}.emb.weight"
            if mapped_name.endswith(".k_proj.weight"):
                tensor = _get_tensor()
                fake_bias_name = mapped_name.replace(".weight", ".bias")
                fake_bias = torch.zeros(tensor.size(0))
                for orig_key, (packed_name, shard_id) in packed.items():
                    if orig_key in fake_bias_name:
                        param_name = fake_bias_name.replace(orig_key, packed_name)
                        try:
                            param = model.get_parameter(param_name)
                            weight_loader_fn = getattr(param, "weight_loader")
                            weight_loader_fn(param, fake_bias, shard_id)
                        except AttributeError:
                            pass
                        break

        # Llama4: strip language_model. prefix, skip vision weights
        if is_llama4:
            if not mapped_name.startswith("language_model."):
                continue
            mapped_name = mapped_name[len("language_model."):]

            m_fused = _LLAMA4_FUSED_EXPERT_RE.match(mapped_name)
            if m_fused:
                prefix_part, proj = m_fused.groups()
                tensor = _get_tensor()
                if proj == "gate_up_proj":
                    param_name = f"{prefix_part}.w13"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    weight = tensor.transpose(-1, -2)
                    E = weight.shape[0]
                    full_inter = weight.shape[1] // 2
                    tp = param.shape[1] // 2
                    rank = 0
                    if full_inter != tp:
                        from .tp import _tp_rank
                        rank = _tp_rank()
                    gate = weight[:, :full_inter, :]
                    up = weight[:, full_inter:, :]
                    param.data[:, :tp, :].copy_(gate[:, rank * tp:(rank + 1) * tp, :])
                    param.data[:, tp:, :].copy_(up[:, rank * tp:(rank + 1) * tp, :])
                else:
                    param_name = f"{prefix_part}.w2"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    weight = tensor.transpose(-1, -2)
                    full_inter = weight.shape[2]
                    tp_inter = param.shape[2]
                    rank = 0
                    if full_inter != tp_inter:
                        from .tp import _tp_rank
                        rank = _tp_rank()
                    param.data.copy_(weight[:, :, rank * tp_inter:(rank + 1) * tp_inter])
                loaded += 1
                continue

        # Gemma4 checkpoints are multimodal wrappers.  The text stack lives
        # under model.language_model.*, while vision/audio weights are skipped.
        if is_gemma4:
            if not mapped_name.startswith("model.language_model."):
                continue
            mapped_name = "model." + mapped_name[len("model.language_model."):]
            mapped_name = mapped_name.replace(
                ".router.per_expert_scale",
                ".moe.per_expert_scale",
            )
            mapped_name = mapped_name.replace(
                ".experts.gate_up_proj",
                ".moe.gate_up_proj",
            )
            mapped_name = mapped_name.replace(
                ".experts.down_proj",
                ".moe.down_proj",
            )

            m_fused = _GEMMA4_FUSED_EXPERT_RE.match(mapped_name)
            if m_fused:
                prefix, proj = m_fused.groups()
                _assign_gemma4_fused_expert(model, prefix, proj, _get_tensor())
                loaded += 1
                continue

            if ".self_attn.k_proj." in mapped_name and gemma4_k_eq_v_layers:
                m_layer = _GEMMA4_LAYER_RE.search(mapped_name)
                if m_layer and int(m_layer.group(1)) in gemma4_k_eq_v_layers:
                    param_name = mapped_name.replace("k_proj", "qkv_proj")
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        pass
                    else:
                        weight_loader = getattr(param, "weight_loader")
                        tensor = _get_tensor()
                        weight_loader(param, tensor, "k")
                        weight_loader(param, tensor, "v")
                        loaded += 2
                    continue

        # Handle Qwen2-VL merger: ln_q -> norm, mlp.0 -> fc1, mlp.2 -> fc2
        if is_qwen2_vl or is_qwen2_5_omni:
            m_merger = _QWEN2_MERGER_RE.match(mapped_name)
            if m_merger:
                prefix, attr, wb = m_merger.groups()
                remap = {"ln_q": "norm", "mlp.0": "fc1", "mlp.2": "fc2"}
                mapped_name = f"{prefix}.{remap[attr]}.{wb}"

        # Handle Qwen3-VL vision MLP: linear_fc1 -> fc1, linear_fc2 -> fc2
        if is_qwen3_vl:
            m_mlp = _QWEN3_VISION_MLP_RE.match(mapped_name)
            if m_mlp:
                prefix, fc_num, wb = m_mlp.groups()
                mapped_name = f"{prefix}.fc{fc_num}.{wb}"
            m_merger = _QWEN3_MERGER_FC_RE.match(mapped_name)
            if m_merger and "merger" in mapped_name:
                prefix, fc_name, wb = m_merger.groups()
                fc = "fc1" if fc_name == "linear_fc1" else "fc2"
                mapped_name = f"{prefix}.{fc}.{wb}"

        # Remap learned pos embed nesting (Qwen3-VL)
        if is_qwen3_vl:
            if _VISION_POS_EMBED_RE.match(mapped_name):
                mapped_name = "visual.pos_embed_interp._embed.emb.weight"

        # Remap vision param names for L1 wrapper nesting
        if is_qwen_vl:
            m = _VISION_PATCH_EMBED_RE.match(mapped_name)
            if m:
                prefix, wb = m.groups()
                mapped_name = f"{prefix}.conv.{wb}"

        # Handle vision encoder merged QKV weights
        if is_qwen_vl:
            m_qkv = _VISION_QKV_RE.match(mapped_name)
            if m_qkv:
                prefix, wb = m_qkv.groups()
                loaded += _load_vision_qkv(model, prefix, _get_tensor(), wb)
                continue

        # DeepSeek MoE expert weights/scales — must be checked BEFORE
        # _WEIGHT_SCALE_INV_RE to avoid the generic scale handler consuming
        # expert weight_scale_inv names and silently skipping them.
        m_ds = _DEEPSEEK_EXPERT_RE.match(mapped_name)
        if m_ds:
            moe_prefix, expert_id_str, proj_name, attr = m_ds.groups()
            expert_id = int(expert_id_str)
            if proj_name in ("gate_proj", "up_proj"):
                is_w1 = (proj_name == "gate_proj")
                if attr == "weight":
                    param_name = f"{moe_prefix}.w13"
                else:
                    param_name = f"{moe_prefix}.w13_weight_scale_inv"
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    continue
                param.weight_loader(param, _get_tensor(), expert_id, is_w1=is_w1)
            else:
                if attr == "weight":
                    param_name = f"{moe_prefix}.w2"
                else:
                    param_name = f"{moe_prefix}.w2_weight_scale_inv"
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    continue
                param.weight_loader(param, _get_tensor(), expert_id)
            loaded += 1
            continue

        # Handle FP8 weight_scale_inv tensors
        m_scale = _WEIGHT_SCALE_INV_RE.match(mapped_name)
        if m_scale:
            layer_prefix = m_scale.group(1)
            matched_scale = False
            for orig_key, (packed_name, shard_id) in packed.items():
                if orig_key in layer_prefix:
                    scale_param_name = layer_prefix.replace(
                        orig_key, packed_name) + ".weight_scale_inv"
                    try:
                        param = model.get_parameter(scale_param_name)
                    except AttributeError:
                        break
                    scale_loader = getattr(param, "weight_loader", None)
                    if scale_loader:
                        scale_loader(param, _get_tensor(), shard_id)
                    else:
                        default_weight_loader(param, _get_tensor())
                    loaded += 1
                    matched_scale = True
                    break
            if matched_scale:
                continue
            scale_param_name = layer_prefix + ".weight_scale_inv"
            try:
                param = model.get_parameter(scale_param_name)
            except AttributeError:
                continue
            scale_loader = getattr(param, "weight_loader", default_weight_loader)
            scale_loader(param, _get_tensor())
            loaded += 1
            continue

        # Handle MoE expert weights
        m = _EXPERT_RE.match(mapped_name)
        if m:
            moe_prefix, expert_id_str, w_name = m.groups()
            expert_id = int(expert_id_str)
            if w_name in ("w1", "w3"):
                param_name = f"{moe_prefix}.w13"
                param = model.get_parameter(param_name)
                param.weight_loader(param, _get_tensor(), expert_id, is_w1=(w_name == "w1"))
            else:
                param_name = f"{moe_prefix}.w2"
                param = model.get_parameter(param_name)
                param.weight_loader(param, _get_tensor(), expert_id)
            loaded += 1
            continue

        # Handle Qwen3-VL-MoE fused 3D expert weights, scales, and gate
        if is_qwen3_vl_moe:
            m_gate = _QWEN3_MOE_GATE_RE.match(mapped_name)
            if m_gate:
                mlp_prefix = m_gate.group(1)
                param_name = f"{mlp_prefix}.gate.weight"
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    pass
                else:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, _get_tensor())
                    loaded += 1
                continue

            m_fused_scale = _QWEN3_MOE_FUSED_SCALE_RE.match(mapped_name)
            if m_fused_scale:
                mlp_prefix, proj = m_fused_scale.groups()
                _fused_expert_scales[(mlp_prefix, proj)] = _get_tensor()
                key = (mlp_prefix, proj)
                if key in _fused_expert_weights:
                    _assign_fused_expert(model, key, _fused_expert_weights.pop(key),
                                         _fused_expert_scales.pop(key))
                    loaded += 1
                continue

            m_fused = _QWEN3_MOE_FUSED_EXPERT_RE.match(mapped_name)
            if m_fused:
                mlp_prefix, proj = m_fused.groups()
                key = (mlp_prefix, proj)
                _fused_expert_weights[key] = _get_tensor()
                if key in _fused_expert_scales:
                    _assign_fused_expert(model, key, _fused_expert_weights.pop(key),
                                         _fused_expert_scales.pop(key))
                    loaded += 1
                else:
                    loaded += 1
                continue

        # Handle packed modules (qkv_proj, gate_up_proj)
        matched = False
        for orig_key, (packed_name, shard_id) in packed.items():
            if (is_llama4 or is_qwen3_vl_moe) and "experts." in mapped_name:
                continue
            if orig_key in mapped_name:
                param_name = mapped_name.replace(orig_key, packed_name)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    break
                weight_loader = getattr(param, "weight_loader")
                if is_llama4 and orig_key in ("q_proj", "k_proj"):
                    tensor = _get_tensor()
                    n_heads = (
                        llama4_config.num_key_value_heads
                        if orig_key == "k_proj"
                        else llama4_config.num_attention_heads
                    )
                    tensor = _permute_qk_for_rotary(tensor, n_heads)
                    weight_loader(param, tensor, shard_id)
                else:
                    weight_loader(param, _get_tensor(), shard_id)
                loaded += 1
                matched = True
                break
        if matched:
            continue
        if "rotary_emb" in mapped_name:
            continue
        m_emb_w = _EMBED_WEIGHT_RE.match(mapped_name)
        if m_emb_w:
            mapped_name = f"{m_emb_w.group(1)}.embedding_op.emb.weight"
        try:
            param = model.get_parameter(mapped_name)
        except AttributeError:
            continue
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, _get_tensor())
        loaded += 1

        # Whisper: duplicate decoder embed_tokens -> lm_head (tied weights)
        if is_whisper and mapped_name == "decoder.embed_tokens.emb.weight":
            lm_param = model.get_parameter("lm_head.embedding_op.emb.weight")
            lm_loader = getattr(lm_param, "weight_loader", default_weight_loader)
            lm_loader(lm_param, _get_tensor())
            loaded += 1

    # Assign any remaining buffered fused expert weights (weight arrived
    # in a different safetensors file than its scale, so the pair wasn't
    # complete during the main loop).
    if _fused_expert_weights:
        for key in list(_fused_expert_weights.keys()):
            scale = _fused_expert_scales.pop(key, None)
            _assign_fused_expert(model, key, _fused_expert_weights.pop(key), scale)
        del _fused_expert_weights, _fused_expert_scales

    print(f"  Loaded {loaded} weight shards ({_time.perf_counter()-_t_load:.1f}s).")


def _postprocess_moe_fp8_weights(module) -> int:
    """Post-process MoE expert FP8 weights for DeepGEMM scale layout.

    Keeps original FP8 weights and scales from the checkpoint (no UE8M0
    requantization) and creates DeepGEMM-layout transformed scale tensors
    stored in w13_scale_dg / w2_scale_dg.

    Only runs when DeepGEMM is available on Hopper+ GPUs.
    """
    import deep_gemm

    if not hasattr(module, 'w13') or not hasattr(module, 'w13_scale'):
        return 0
    if module.w13.dtype != torch.float8_e4m3fn:
        return 0

    from ..tasks.baseline.L1.moe_grouped_gemm import _is_deep_gemm_supported
    if not _is_deep_gemm_supported():
        return 0

    block_shape = getattr(module, 'block_shape', [128, 128])
    block_m, block_k = int(block_shape[0]), int(block_shape[1])

    count = 0
    for wname, sname in [('w13', 'w13_scale'), ('w2', 'w2_scale')]:
        wq = getattr(module, wname).data
        ws = getattr(module, sname).data

        E = wq.size(0)

        recipe = (1, block_m, block_k)
        dg_ws = deep_gemm.transform_sf_into_required_layout(
            sf=ws,
            mn=wq.size(1),
            k=wq.size(2),
            recipe=recipe,
            num_groups=E,
            is_sfa=False,
            disable_ue8m0_cast=True,
        )
        dg_sname = sname + "_dg"
        module.register_parameter(
            dg_sname, torch.nn.Parameter(dg_ws, requires_grad=False)
        )
        count += 1

    return count


def _postprocess_fp8_weights(model: torch.nn.Module) -> None:
    """Re-quantize FP8 weights to UE8M0 format and transform scale layout for DeepGEMM.

    Handles both regular Fp8Linear modules (e.g. DeepSeek MLA/MoE gating projections,
    Qwen3-MoE gate proj) and fused MoE expert stacks (DeepSeek V3 / Qwen3-MoE).
    """
    import time as _time
    from ..tasks.baseline.L1.fp8_linear import (
        Fp8Linear, postprocess_fp8_weights, postprocess_fp8_weights_batched,
    )

    _t_pp = _time.perf_counter()
    print("  Post-processing FP8 weights for DeepGEMM...", flush=True)

    # --- FP8 linear modules (DeepSeek MLA projections, Qwen3 gate proj, etc.) ---
    linear_modules = [
        m for m in model.modules()
        if isinstance(getattr(m, 'linear_op', None), Fp8Linear)
    ]
    for module in linear_modules:
        w = module.weight
        s = module.weight_scale_inv
        w_new, s_new = postprocess_fp8_weights(w.data, s.data)
        module.weight = torch.nn.Parameter(w_new, requires_grad=False)
        module.weight_scale_inv = torch.nn.Parameter(s_new, requires_grad=False)
    print(f"    {len(linear_modules)} FP8 linear layers done "
          f"({_time.perf_counter()-_t_pp:.1f}s)", flush=True)

    # --- MoE expert stacks: DeepSeek V3 uses postprocess_fp8_weights_batched
    # for in-place UE8M0 requantization + scale layout transform. Qwen3-MoE
    # keeps its original scales and uses a separate DG-layout buffer. ---
    moe_count = 0
    deepseek_moe_modules: list = []
    qwen3_moe_modules: list = []
    try:
        from ..tasks.baseline.L2.deepseek_moe import DeepSeekMoE
        deepseek_moe_modules = [
            m for m in model.modules()
            if isinstance(m, DeepSeekMoE) and getattr(m, 'use_fp8', False)
        ]
    except ImportError:
        pass
    try:
        from ..tasks.baseline.L2.qwen3_moe import Qwen3MoE
        qwen3_moe_modules = [
            m for m in model.modules()
            if isinstance(m, Qwen3MoE) and getattr(m, 'use_fp8', False)
        ]
    except ImportError:
        pass

    if deepseek_moe_modules:
        _t_moe = _time.perf_counter()
        total = len(deepseek_moe_modules)
        for j, module in enumerate(deepseek_moe_modules):
            for wname, sname in (("w13", "w13_weight_scale_inv"),
                                 ("w2", "w2_weight_scale_inv")):
                w = getattr(module, wname)
                s = getattr(module, sname)
                postprocess_fp8_weights_batched(w.data, s.data)
                moe_count += w.shape[0]
            if j % max(1, total // 5) == 0 or j == total - 1:
                print(f"    DeepSeek MoE postprocess {j+1}/{total} "
                      f"({(j+1)*100//total}%, "
                      f"{_time.perf_counter()-_t_moe:.1f}s)", flush=True)

    for module in qwen3_moe_modules:
        moe_count += _postprocess_moe_fp8_weights(module)

    if linear_modules or moe_count > 0:
        torch.cuda.empty_cache()
    print(f"  Post-processed {len(linear_modules)} FP8 linear layers, "
          f"{moe_count} MoE expert weight slices "
          f"({_time.perf_counter()-_t_pp:.1f}s total).", flush=True)


def _is_diffusion_model(model_name: str) -> bool:
    """Check if the model is a diffusion model (e.g., FLUX, HunyuanVideo) by looking for model_index.json."""
    import json as _json
    if os.path.isdir(model_name):
        index_path = os.path.join(model_name, "model_index.json")
    else:
        from huggingface_hub import hf_hub_download
        try:
            index_path = hf_hub_download(model_name, "model_index.json")
        except Exception:
            return False
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = _json.load(f)
        class_name = data.get("_class_name", "")
        if "Flux" in class_name or "Diffusion" in class_name or "HunyuanVideo" in class_name:
            return True
    return False


def _detect_diffusion_type(model_name: str) -> str:
    """Distinguish between diffusion model types (flux vs hunyuan_video)."""
    import json as _json
    if os.path.isdir(model_name):
        index_path = os.path.join(model_name, "model_index.json")
    else:
        from huggingface_hub import hf_hub_download
        index_path = hf_hub_download(model_name, "model_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = _json.load(f)
        class_name = data.get("_class_name", "")
        if "HunyuanVideo" in class_name:
            return "hunyuan_video"
    return "flux"


def _load_config_dict(model_name: str) -> dict:
    """Read ``config.json`` from either a local staging dir or the Hub."""
    import json
    if os.path.isdir(model_name):
        path = os.path.join(model_name, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_name, "config.json")
    with open(path) as f:
        return json.load(f)


def _detect_model_type(model_name: str) -> str:
    """Detect model architecture from HuggingFace config."""
    if _is_diffusion_model(model_name):
        return _detect_diffusion_type(model_name)
    try:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(hf_config, "model_type", "llama")
    except (ValueError, OSError):
        # DeepSeek-V3.2 ships a custom config class not registered with
        # transformers; BitNet b1.58 ships an ``auto_map`` pointing at
        # files that don't exist in the repo (model_type is registered
        # natively).  In both cases reading config.json directly works.
        model_type = _load_config_dict(model_name).get("model_type", "llama")
    if model_type == "deepseek_v32":
        model_type = "deepseek_v3"
    return model_type


def _detect_quant_config(model_name: str) -> dict | None:
    """Detect FP8 quantization config from HuggingFace config."""
    try:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except (ValueError, OSError):
        from types import SimpleNamespace
        hf_config = SimpleNamespace(**_load_config_dict(model_name))
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        return None
    if isinstance(qc, dict):
        quant_method = qc.get("quant_method", "")
    else:
        quant_method = getattr(qc, "quant_method", "")
    if quant_method != "fp8":
        return None
    if isinstance(qc, dict):
        return qc
    return qc.to_dict() if hasattr(qc, "to_dict") else {"quant_method": "fp8"}


def _restore_mamba_ssm_params(
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
) -> None:
    # vLLM keeps the SSM recurrence parameters in fp32 even when the
    # surrounding model weights run in reduced precision. The custom
    # selective-scan kernels rely on that contract.
    if model_type == "mamba":
        from ..tasks.baseline.L2.mamba_mixer import MambaMixer

        for module in model.modules():
            if isinstance(module, MambaMixer):
                module.A.data = module.A.data.to(
                    device=device, dtype=torch.float32,
                )
                module.D.data = module.D.data.to(
                    device=device, dtype=torch.float32,
                )
    elif model_type == "mamba2":
        from ..tasks.baseline.L2.mamba2_mixer import Mamba2Mixer

        for module in model.modules():
            if isinstance(module, Mamba2Mixer):
                module.A.data = module.A.data.to(
                    device=device, dtype=torch.float32,
                )
                module.D.data = module.D.data.to(
                    device=device, dtype=torch.float32,
                )
                module.dt_bias.data = module.dt_bias.data.to(
                    device=device, dtype=torch.float32,
                )


def load_model(
    model_name: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    model_path = download_model(model_name)
    model_type = _detect_model_type(model_name)
    quant_config = _detect_quant_config(model_name)

    if quant_config:
        print(f"  Detected FP8 quantization: {quant_config.get('quant_method')}, "
              f"block_size={quant_config.get('weight_block_size')}")

    if model_type in ("flux", "hunyuan_video"):
        raise ValueError(
            "Diffusion models should be loaded via "
            "fastkernels.infra.diffusion_engine.DiffusionEngine, "
            "not the LLM load_model() path."
        )
    if model_type == "cosyvoice3":
        raise ValueError(
            "CosyVoice3 TTS models should be loaded via "
            "fastkernels.infra.tts_engine.TTSEngine, "
            "not the LLM load_model() path."
        )
    if model_type == "jamba":
        # Jamba is a hybrid transformer+Mamba+MoE model that does not
        # plug into the paged-KV LlamaEngine cleanly: every batch needs
        # both a KV slab AND per-sequence Mamba selective-scan state.
        # ``infra.jamba_engine.JambaEngine`` owns that wiring (Pattern 2,
        # per-pipeline weight loader documented in CLAUDE.md).
        raise ValueError(
            "Jamba models should be loaded via "
            "fastkernels.infra.jamba_engine.JambaEngine, "
            "not the LLM load_model() path."
        )
    if model_type == "pi0":
        raise ValueError(
            "Pi0 robotics VLA models should be loaded via "
            "fastkernels.infra.pi0_engine.Pi0Engine, "
            "not the LLM load_model() path."
        )
    if model_type == "gpt_oss":
        from ..tasks.baseline.L4.gpt_oss import GptOssConfig, GptOssForCausalLM
        config = GptOssConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating GPT-OSS model ({config.num_local_experts} experts, "
              f"top-{config.num_experts_per_tok})...")
        model = GptOssForCausalLM(config)
    elif model_type == "gemma4":
        config = Gemma4Config.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Gemma4 model ({config.num_experts} experts, "
              f"top-{config.top_k_experts})...")
        model = Gemma4ForCausalLM(config)
    elif model_type == "whisper":
        from ..tasks.baseline.L4.whisper import (
            WhisperConfig, WhisperForConditionalGeneration,
        )
        config = WhisperConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Whisper model (enc={config.encoder_layers}L, "
              f"dec={config.decoder_layers}L, d={config.d_model})...")
        model = WhisperForConditionalGeneration(config)
    elif model_type == "llama4":
        from ..tasks.baseline.L4.llama4 import Llama4Config, Llama4ForCausalLM
        config = Llama4Config.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Llama4 model ({config.num_local_experts} experts, "
              f"top-{config.num_experts_per_tok})...")
        model = Llama4ForCausalLM(config)
    elif model_type == "mixtral":
        from ..tasks.baseline.L4.mixtral import MixtralConfig, MixtralForCausalLM
        config = MixtralConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Mixtral model ({config.num_local_experts} experts)...")
        model = MixtralForCausalLM(config)
    elif model_type == "qwen2_vl":
        from ..tasks.baseline.L4.qwen2_vl import (
            Qwen2VLConfig, Qwen2VLForConditionalGeneration,
        )
        config = Qwen2VLConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Qwen2-VL model...")
        model = Qwen2VLForConditionalGeneration(config)
    elif model_type in ("qwen3_vl", "qwen3_vl_moe"):
        from ..tasks.baseline.L4.qwen3_vl import (
            Qwen3VLConfig, Qwen3VLForConditionalGeneration,
        )
        config = Qwen3VLConfig.from_pretrained(model_name)
        config.dtype = dtype
        if config.is_moe:
            print(f"  Allocating Qwen3-VL-MoE model ({config.num_experts} experts, "
                  f"top-{config.num_experts_per_tok})...")
        else:
            print("  Allocating Qwen3-VL model...")
        model = Qwen3VLForConditionalGeneration(config, quant_config=quant_config)
    elif model_type == "qwen3_next":
        config = Qwen3NextConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(
            "  Allocating Qwen3-Next model "
            f"({config.num_experts} experts, "
            f"{sum(config.is_linear_attn_layer(i) for i in range(config.num_hidden_layers))} GDN + "
            f"{sum(not config.is_linear_attn_layer(i) for i in range(config.num_hidden_layers))} MHA layers)..."
        )
        model = Qwen3NextForCausalLM(config)
    elif model_type == "qwen2_5_omni":
        from ..tasks.baseline.L4.qwen2_5_omni import (
            Qwen2_5OmniConfig, Qwen2_5OmniThinkerForConditionalGeneration,
        )
        config = Qwen2_5OmniConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Qwen2.5-Omni Thinker model...")
        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
    elif model_type == "mamba2":
        from ..tasks.baseline.L4.mamba2 import Mamba2Config, Mamba2ForCausalLM
        config = Mamba2Config.from_pretrained(model_path)
        config.dtype = dtype
        print(f"  Allocating Mamba2 model "
              f"(L={config.num_hidden_layers}, hidden={config.hidden_size}, "
              f"heads={config.num_heads}, head_dim={config.head_dim}, "
              f"groups={config.n_groups}, state={config.state_size})...")
        model = Mamba2ForCausalLM(config)
    elif model_type == "mamba":
        from ..tasks.baseline.L4.mamba import MambaConfig, MambaForCausalLM
        config = MambaConfig.from_pretrained(model_path)
        config.dtype = dtype
        print(f"  Allocating Mamba model "
              f"(L={config.num_hidden_layers}, hidden={config.hidden_size}, "
              f"intermediate={config.intermediate_size}, state={config.state_size})...")
        model = MambaForCausalLM(config)
    elif model_type == "deepseek_v3":
        from ..tasks.baseline.L4.deepseek import (
            DeepSeekV3Config, DeepSeekV3ForCausalLM,
        )
        config = DeepSeekV3Config.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating DeepSeek V3.2 model ({config.n_routed_experts} experts, "
              f"top-{config.num_experts_per_tok}, DSA topk={config.index_topk})...")
        model = DeepSeekV3ForCausalLM(config, quant_config=quant_config)
    elif model_type == "bitnet":
        config = BitNetConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating BitNet b1.58 model "
              f"({config.num_hidden_layers}L, hidden={config.hidden_size}, "
              f"W1.58A8)...")
        model = BitNetForCausalLM(config)
    elif model_type == "kimi_linear":
        from ..tasks.baseline.L4.kimi_linear import (
            KimiLinearConfig,
            KimiLinearForCausalLM,
        )
        config = KimiLinearConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(
            "  Allocating Kimi-Linear model "
            f"({config.num_experts} experts, "
            f"{len(config.kda_layers)} KDA + {len(config.full_attn_layers)} MLA layers)..."
        )
        model = KimiLinearForCausalLM(config, quant_config=quant_config)
    else:
        config = LlamaConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Llama model...")
        model = LlamaForCausalLM(config)

    tp = _tp_size()
    if hasattr(config, "num_attention_heads") and config.num_attention_heads % tp != 0:
        raise ValueError(
            f"TP degree {tp} is incompatible with {config.num_attention_heads} Q heads "
            f"(num_attention_heads must be divisible by tensor_parallel_size)"
        )
    if (model_type != "deepseek_v3"
            and hasattr(config, "num_key_value_heads")
            and config.num_key_value_heads % tp != 0):
        raise ValueError(
            f"TP degree {tp} is incompatible with {config.num_key_value_heads} KV heads "
            f"(num_key_value_heads must be divisible by tensor_parallel_size)"
        )
    # Mamba2 num_heads must be divisible by TP world size
    if model_type == "mamba2":
        if config.num_heads % tp != 0:
            raise ValueError(
                f"TP degree {tp} is incompatible with {config.num_heads} Mamba2 heads "
                f"(num_heads must be divisible by tensor_parallel_size)"
            )
        if config.n_groups % tp != 0 and config.n_groups != 1:
            raise ValueError(
                f"TP degree {tp} requires n_groups ({config.n_groups}) to be divisible "
                f"by tp or equal to 1 (extra-groups replication only supported when n_groups==1)."
            )
    if model_type == "mamba" and config.intermediate_size % tp != 0:
        raise ValueError(
            f"TP degree {tp} is incompatible with Mamba intermediate_size "
            f"{config.intermediate_size}."
        )

    if model_type == "gpt_oss":
        _load_gpt_oss_weights(model, model_path)
        # GPT-OSS has mixed dtypes: uint8 MXFP4 expert weights must not be cast
        for name, param in model.named_parameters():
            if param.dtype == torch.uint8:
                if param.data.device != device:
                    param.data = param.data.to(device=device)
            elif param.data.device != device or param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
        for name, buf in model.named_buffers():
            if buf.device != device:
                buf.data = buf.data.to(device=device)
        # Swizzle MXFP4 weights for Triton kernels after GPU transfer
        from ..tasks.baseline.L2.gpt_oss_moe import GptOssMoE
        for mod in model.modules():
            if isinstance(mod, GptOssMoE):
                mod.process_weights_after_loading()
    else:
        load_weights(model, model_path, model_type)

    if quant_config:
        for name, param in model.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                if not param.is_cuda:
                    param.data = param.data.to(device=device)
            elif "weight_scale_inv" in name or "w13_scale" in name or "w2_scale" in name:
                param.data = param.data.to(device=device)
            elif param.dtype == torch.float32:
                # Preserve FP32 parameters (e.g. the DeepSeek router's
                # ``e_score_correction_bias``) that were intentionally allocated
                # in FP32 and must stay FP32 to match vLLM's router bias path.
                if param.data.device != device:
                    param.data = param.data.to(device=device)
            elif param.data.device != device or param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
        for name, buf in model.named_buffers():
            if buf.device != device:
                buf.data = buf.data.to(device=device)
        if model_type in ("deepseek_v3", "kimi_linear"):
            _compute_mla_absorbed_weights(model)
        _postprocess_fp8_weights(model)
    elif model_type != "gpt_oss":
        model = model.to(device=device, dtype=dtype)

    if model_type in ("mamba", "mamba2") and dtype != torch.float32:
        _restore_mamba_ssm_params(model, model_type, device)

    if model_type in ("deepseek_v3", "kimi_linear") and quant_config is None:
        _compute_mla_absorbed_weights(model)

    if model_type == "bitnet":
        # Materialize bf16 fake-quant weight per BitLinear (mirrors SOTA's
        # ``model_state_fp16.pt``).  ``ternary * scale`` derived once
        # on-device from the already-loaded packed-int2 weight + scale.
        from ..tasks.baseline.L1.bitnet_linear import BitLinear, BitLinearMerged
        n = 0
        for mod in model.modules():
            if isinstance(mod, (BitLinear, BitLinearMerged)):
                mod.process_weights_after_loading()
                n += 1
        print(f"  Materialized bf16 fake-quant weights for {n} BitLinear layers.",
              flush=True)
        if os.environ.get("KB_BITNET_BF16_ALIGN", "1") != "0":
            _override_bitnet_bf16_with_master(model, model_name, device)

    model.eval()
    return model, config



# Mapping from int2-release HF id -> matching BF16 master release id.
# The BF16 release contains the same model with full-precision master
# weights (used by SOTA's ``convert_safetensors`` + ``convert_checkpoint``
# pipeline to produce ``model_state_fp16.pt``).  Both releases come from
# the same QAT run but differ in their final-quantization scale, so the
# int2 weights don't round-trip back to the BF16 master values bit-exactly.
_BITNET_INT2_TO_BF16_MASTER: dict = {
    "microsoft/bitnet-b1.58-2B-4T": "microsoft/bitnet-b1.58-2B-4T-bf16",
}


def _override_bitnet_bf16_with_master(model, model_name: str,
                                      device: torch.device) -> None:
    """Re-derive ``bf16_weight`` buffers from the BF16 master release.

    SOTA's prefill weights (``model_state_fp16.pt``) are produced by
    applying ``quant_weight_fp16(w) = (w*s).round().clamp(-1,1) / s``
    (``s = 1/|w|.mean().clamp(min=1e-5)``) to the BF16 master weights of
    ``microsoft/bitnet-b1.58-2B-4T-bf16``.  The same formula on
    ``unpack(int2_release) * weight_scale`` does *not* round-trip to those
    bf16 values bit-exactly — the two HF releases were quantized with
    slightly different scales (~1-22% of ternary positions disagree per
    tensor between the two), which compounds across 30 layers and flips
    ~80% of greedy-decode argmaxes.

    By auto-fetching the BF16 master release alongside the int2 release
    and re-quantizing it via SOTA's exact formula, we make every
    ``BitLinear`` in the prefill path bit-identical to SOTA at the layer
    level.  The decode path still uses HF's int2 packed weights (unmodified).

    No-op for BitNet variants that don't have a published BF16 master
    release (the int2-derived bf16_weight from
    ``process_weights_after_loading`` is then the best we can do).
    """
    bf16_repo = _BITNET_INT2_TO_BF16_MASTER.get(model_name)
    if bf16_repo is None:
        print(f"  No BF16 master release registered for {model_name}; "
              f"keeping int2-derived bf16 fake-quant weights "
              f"(prefill alignment may drift).", flush=True)
        return

    from ..tasks.baseline.L1.bitnet_linear import BitLinear, BitLinearMerged
    from ..tasks.baseline.L1.bitnet_int8xint2_linear import (
        repack_ternary_kn, VALUES_PER_BYTE,
    )

    print(f"  Auto-fetching BF16 master release {bf16_repo} for "
          f"bit-exact alignment with SOTA (prefill + decode)...", flush=True)
    try:
        bf16_path = download_model(bf16_repo)
    except Exception as exc:
        print(f"  WARNING: failed to fetch {bf16_repo} ({exc}); keeping "
              f"int2-derived bf16 fake-quant weights.", flush=True)
        return

    # Stream tensors from disk one safetensors file at a time to avoid
    # holding the entire 5GB BF16 master in CPU memory.  We collect just
    # the names we need (every ``*_proj.weight``) into a name->path
    # index, then open each file once per layer it contributes to.
    sf_files = sorted(glob(os.path.join(bf16_path, "*.safetensors")))
    if not sf_files:
        print(f"  WARNING: no .safetensors files in {bf16_path}; "
              f"keeping int2-derived bf16 weights.", flush=True)
        return

    name_to_file: dict = {}
    for sf_file in sf_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for k in f.keys():
                name_to_file[k] = sf_file

    def quant_weight_int8_and_fp16(w: torch.Tensor):
        """Bit-for-bit identical to SOTA's ``quant_weight_int8`` and
        ``quant_weight_fp16`` (``vllm_repo/BitNet/gpu/convert_checkpoint.py``).
        Returns ``(ternary_int8, scale_bf16, fake_quant_bf16)`` for one shard,
        all on the same device as ``w`` (which is bf16).
        """
        s = 1.0 / w.abs().mean().clamp_(min=1e-5)
        ternary = (w * s).round().clamp(-1, 1)
        return (
            ternary.to(torch.int8),
            (1.0 / s).to(torch.bfloat16).reshape(1),  # SOTA returns shape (1,)
            ternary / s,                              # fake-quant bf16 weight
        )

    def _load_shard(shard_key: str) -> torch.Tensor:
        file_path = name_to_file.get(shard_key)
        if file_path is None:
            raise KeyError(
                f"BF16 master release {bf16_repo} is missing "
                f"{shard_key}; cannot align weights")
        with safe_open(file_path, "pt", "cpu") as f:
            return f.get_tensor(shard_key).to(device=device,
                                              dtype=torch.bfloat16)

    n_merged, n_single = 0, 0
    for mod_name, mod in model.named_modules():
        if isinstance(mod, BitLinearMerged):
            attn_match = re.match(r"(.+)\.qkv_proj$", mod_name)
            mlp_match = re.match(r"(.+)\.gate_up_proj$", mod_name)
            if attn_match:
                base = attn_match.group(1)
                shards = [f"{base}.q_proj.weight",
                          f"{base}.k_proj.weight",
                          f"{base}.v_proj.weight"]
            elif mlp_match:
                base = mlp_match.group(1)
                shards = [f"{base}.gate_proj.weight",
                          f"{base}.up_proj.weight"]
            else:
                continue
            ternary_parts, fake_parts, scale_parts = [], [], []
            for shard_key in shards:
                w = _load_shard(shard_key)
                if attn_match and (
                    shard_key.endswith(".q_proj.weight")
                    or shard_key.endswith(".k_proj.weight")
                ):
                    w = _permute_bitnet_qk_to_sota(
                        w, n_heads=w.shape[0] // model.config.head_dim,
                    )
                t_int8, s_bf16, fq_bf16 = quant_weight_int8_and_fp16(w)
                ternary_parts.append(t_int8)
                fake_parts.append(fq_bf16)
                # fastkernels stores per-row scale across the shard's rows;
                # SOTA stores a single scalar per shard.  Broadcast the
                # SOTA scalar across the matching number of rows so the
                # int8xint2 GEMM kernel sees the right per-row dequant.
                scale_parts.append(s_bf16.expand(t_int8.shape[0]).contiguous())
            ternary = torch.cat(ternary_parts, dim=0)
            mod.weight.data.copy_(repack_ternary_kn(ternary))
            if attn_match:
                # Microsoft BitNet's decode kernel indexes the 3 QKV scales
                # by equal output bands: out_idx / (3840 / 3).  That is not
                # the natural [q=2560, k=640, v=640] shard layout used by the
                # bf16 prefill path, so mirror the kernel's banded decode
                # semantics in our per-row scale tensor.
                q_scale = scale_parts[0][0]
                k_scale = scale_parts[1][0]
                v_scale = scale_parts[2][0]
                band = mod.total_out // 3
                mod.weight_scale.data[:band].fill_(float(q_scale))
                mod.weight_scale.data[band:2 * band].fill_(float(k_scale))
                mod.weight_scale.data[2 * band:].fill_(float(v_scale))
            else:
                mod.weight_scale.data.copy_(torch.cat(scale_parts, dim=0))
            mod.register_buffer(
                "bf16_weight",
                torch.cat(fake_parts, dim=0).contiguous(),
                persistent=False,
            )
            scale_values = [part[0] for part in scale_parts]
            mod.set_official_decode_buffers(
                ternary=ternary, scale_values=scale_values,
            )
            n_merged += 1
        elif isinstance(mod, BitLinear):
            shard_key = f"{mod_name}.weight"
            w = _load_shard(shard_key)
            t_int8, s_bf16, fq_bf16 = quant_weight_int8_and_fp16(w)
            mod.weight.data.copy_(repack_ternary_kn(t_int8))
            mod.weight_scale.data.fill_(float(s_bf16.item()))
            mod.register_buffer("bf16_weight", fq_bf16.contiguous(),
                                persistent=False)
            mod.set_official_decode_buffers(
                ternary=t_int8, scale_values=[s_bf16[0]],
            )
            n_single += 1
    print(f"  Bit-exact int8 + bf16 weights re-materialized from BF16 "
          f"master: {n_merged} merged + {n_single} single = "
          f"{n_merged + n_single} BitLinear layers.", flush=True)

def _iter_draft_weights(model_path: str):
    """Yield ``(name, tensor)`` from either safetensors or pytorch_model.bin.

    EAGLE-3 draft checkpoints are typically published as a single
    ``pytorch_model.bin`` file (HF legacy format). We also support
    safetensors for forward compatibility.
    """
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if safetensor_files:
        for sf_file in safetensor_files:
            with safe_open(sf_file, "pt", "cpu") as f:
                for k in f.keys():
                    yield k, f.get_tensor(k)
        return

    bin_files = sorted(glob(os.path.join(model_path, "pytorch_model*.bin")))
    if not bin_files:
        raise FileNotFoundError(
            f"No .safetensors or pytorch_model*.bin files found in {model_path}"
        )
    for bf in bin_files:
        state = torch.load(bf, map_location="cpu", weights_only=True)
        for k, v in state.items():
            yield k, v


def _load_eagle3_weights(model: LlamaForCausalLMEagle3, model_path: str) -> None:
    """Load weights for the EAGLE-3 draft model.

    Supports the published sglang draft checkpoints (e.g.
    ``jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B``). Handles the same
    packed_modules_mapping as the regular Llama loader, plus:
      - any ``*.d2t`` tensor is converted to ``hot_token_id = d2t + arange``
      - any ``*.t2d`` tensor is dropped
      - ``model.embed_tokens.weight`` and ``lm_head.weight`` are routed to the
        wrapped ``embedding_op.emb.weight`` parameter.
    """
    packed = getattr(model, "packed_modules_mapping", {})

    loaded = 0
    for weight_name, tensor in _iter_draft_weights(model_path):
        if "d2t" in weight_name:
            d2t = tensor.long()
            hot = d2t + torch.arange(d2t.shape[0], dtype=torch.long, device=d2t.device)
            model.hot_token_id.data.copy_(hot)
            model._has_hot_token_id = True
            loaded += 1
            continue

        if "t2d" in weight_name:
            continue

        mapped_name = weight_name
        if not mapped_name.startswith("model.") and not mapped_name.startswith("lm_head"):
            mapped_name = f"model.{mapped_name}"

        matched = False
        for orig_key, (packed_name, shard_id) in packed.items():
            if orig_key in mapped_name:
                param_name = mapped_name.replace(orig_key, packed_name)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    break
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id)
                loaded += 1
                matched = True
                break
        if matched:
            continue

        m_emb_w = _EMBED_WEIGHT_RE.match(mapped_name)
        if m_emb_w:
            mapped_name = f"{m_emb_w.group(1)}.embedding_op.emb.weight"

        try:
            param = model.get_parameter(mapped_name)
        except AttributeError:
            continue
        wl = getattr(param, "weight_loader", default_weight_loader)
        wl(param, tensor)
        loaded += 1

    print(f"  Loaded {loaded} EAGLE-3 weight shards.")


def load_eagle3_draft_model(
    draft_repo: str,
    target_config: LlamaConfig,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[LlamaForCausalLMEagle3, LlamaEagle3Config]:
    """Download and load an EAGLE-3 draft checkpoint.

    Supports both safetensors and pytorch_model.bin checkpoints. The returned
    model's ``embed_tokens`` is initialized from the checkpoint when present;
    the engine should still call
    ``model.set_embed_tokens(target.model.embed_tokens)`` to share memory with
    the target.
    """
    model_path = snapshot_download(
        draft_repo,
        allow_patterns=["*.safetensors", "*.json", "*.bin"],
    )
    config = LlamaEagle3Config.from_pretrained(draft_repo, target_config)
    config.dtype = dtype
    print(f"  Allocating EAGLE-3 draft model (hidden={config.hidden_size}, "
          f"draft_vocab={config.draft_vocab_size})...")
    model = LlamaForCausalLMEagle3(config)

    _load_eagle3_weights(model, model_path)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config



def _compute_mla_absorbed_weights(model: torch.nn.Module) -> None:
    """Compute absorbed W_UV weights for MLA decode after loading."""
    from ..tasks.baseline.L2.deepseek_mla_attention import DeepSeekMLAAttention
    from ..tasks.baseline.L2.kimi_mla_attention import KimiMLAAttention

    count = 0
    for module in model.modules():
        if isinstance(module, (DeepSeekMLAAttention, KimiMLAAttention)):
            module.compute_absorbed_weights()
            count += 1
    print(f"  Computed absorbed MLA weights for {count} attention layers.")
