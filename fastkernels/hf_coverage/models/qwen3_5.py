"""Qwen3.5 vision and hybrid decoder built from existing Qwen components."""

from dataclasses import fields

import torch
from torch import nn

from fastkernels.infra.context import (
    AttnBackendConfig,
    get_context,
    set_attn_backend_config,
)
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L1.mrope import MRotaryEmbedding
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP
from fastkernels.tasks.baseline.L2.parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from fastkernels.tasks.baseline.L2.qwen3_next_attention import Qwen3NextAttention
from fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention
from fastkernels.tasks.baseline.L4.qwen3_next import Qwen3NextConfig
from fastkernels.tasks.baseline.L4.qwen3_vl import (
    Qwen3VLVisionConfig,
    Qwen3VisionTransformer,
)
from . import qwen3_next
from .qwen2_vl import multimodal_positions
from ..patches.mamba_conv_weight import fp32_causal_conv_weight
from ..runner import Workload


class MultimodalRotary(nn.Module):
    """Keep the three position axes when the attention carrier flattens its input."""

    def __init__(self, config):
        super().__init__()
        rope = config.rope_parameters
        self.head_dim = int(config.head_dim * rope["partial_rotary_factor"])
        self.rotary = MRotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            rope["rope_theta"],
            rope["mrope_section"],
            mrope_interleaved=True,
        )
        self.positions = None

    def forward(self, unused_flat_positions, query, key):
        return self.rotary(self.positions, query, key)


class HybridLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                layer_idx=index,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
            self.conv_history = None
            self.linear_attn.in_proj_qkvz.register_forward_hook(self.record_conv_input)
        else:
            self.self_attn = Qwen3NextAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                index,
                rms_norm_eps=config.rms_norm_eps,
            )
        self.mlp = LlamaMLP(config)

    def record_conv_input(self, module, args, projected):
        """Retain HF's extra history column; the convolution itself needs three."""
        attention = self.linear_attn
        groups = projected.reshape(projected.shape[0], attention.num_k_heads, -1)
        width = attention.head_k_dim
        values = attention.value_dim // attention.num_k_heads
        q, k, v, unused_z = groups.split((width, width, values, values), dim=-1)
        packed = torch.cat([x.flatten(1) for x in (q, k, v)], dim=-1).T[None]
        kernel = attention.conv_kernel_size
        if get_context().is_prefill:
            self.conv_history = packed[..., -kernel:].clone()
        else:
            self.conv_history = torch.cat((self.conv_history[..., 1:], packed), dim=-1)

    def forward(self, hidden, positions, rotary, state):
        normalized = self.input_layernorm(hidden)
        if self.layer_type == "linear_attention":
            update = self.linear_attn(normalized, state_manager=state)
        else:
            update = self.self_attn(
                normalized, rotary_emb=rotary, positions=positions, state_manager=state
            )
        hidden = hidden + update
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class HybridBackbone(nn.Module):
    def __init__(self, config, text):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(text.vocab_size, text.hidden_size)
        self.layers = nn.ModuleList(
            [HybridLayer(text, i) for i in range(text.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.rotary = MultimodalRotary(config.text_config)


class Qwen35(nn.Module):
    def __init__(self, config, text):
        super().__init__()
        self.config, self.hf_config = text, config
        self.model = HybridBackbone(config, text)
        vc = Qwen3VLVisionConfig(
            **{
                field.name: getattr(config.vision_config, field.name, [])
                for field in fields(Qwen3VLVisionConfig)
            }
        )
        self.visual = Qwen3VisionTransformer(vc)
        self.lm_head = ParallelLMHead(text.vocab_size, text.hidden_size)
        self.inputs, self.rope_delta, self.last_state = None, None, None

    def forward(self, ids, positions, state_manager):
        hidden = self.model.embed_tokens(ids)
        if get_context().is_prefill:
            for token, pixels, grid in (
                (self.hf_config.image_token_id, "pixel_values", "image_grid_thw"),
                (
                    self.hf_config.video_token_id,
                    "pixel_values_videos",
                    "video_grid_thw",
                ),
            ):
                features = self.visual(
                    self.inputs[pixels], self.inputs[grid].detach().cpu()
                )
                hidden[ids == token] = features
            positions, self.rope_delta = multimodal_positions(
                ids,
                self.inputs["mm_token_type_ids"][0, : ids.numel()],
                self.inputs,
                self.hf_config,
            )
            self.rope_delta = self.rope_delta.reshape(1, 1)
        else:
            positions = positions[None].expand(3, -1) + self.rope_delta
        self.model.rotary.positions = positions
        for layer in self.model.layers:
            hidden = layer(hidden, positions, self.model.rotary, state_manager)
        self.last_state = state_manager
        return self.model.norm(hidden)


def build_from_config(config, device, dtype):
    text = config.text_config
    if (
        text.hidden_act != "silu"
        or text.attention_bias
        or config.vision_config.deepstack_visual_indexes
    ):
        raise ValueError(
            "The selected Qwen3.5 checkpoint uses bias-free SiLU and no DeepStack"
        )
    if text.rope_parameters["rope_type"] != "default":
        raise ValueError(
            "The selected Qwen3.5 checkpoint uses default partial multimodal RoPE"
        )
    set_attn_backend_config(AttnBackendConfig.auto_detect())
    local = Qwen3NextConfig._from_hf(text)
    local.dtype = dtype
    return Qwen35(config, local).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    remaining = dict(state)

    def copy(parameter, name):
        value = remaining.pop(name)
        if parameter.shape != value.shape:
            raise ValueError(f"Qwen3.5 weight shape mismatch at {name}")
        parameter.copy_(value)

    prefix = "model.language_model."
    copy(
        model.model.embed_tokens.embedding_op.emb.weight, prefix + "embed_tokens.weight"
    )
    copy(model.model.norm.weight, prefix + "norm.weight")
    copy(model.lm_head.embedding_op.emb.weight, "lm_head.weight")
    text = config.text_config
    for i, layer in enumerate(model.model.layers):
        base = prefix + f"layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            copy(getattr(layer, name).weight, base + name + ".weight")
        if layer.layer_type == "linear_attention":
            attention = layer.linear_attn
            ap = base + "linear_attn."
            nk, nv = text.linear_num_key_heads, text.linear_num_value_heads
            dk, dv = text.linear_key_head_dim, text.linear_value_head_dim
            q, k, v = remaining.pop(ap + "in_proj_qkv.weight").split(
                (nk * dk, nk * dk, nv * dv)
            )
            z = remaining.pop(ap + "in_proj_z.weight")
            packed = torch.cat(
                [x.reshape(nk, -1, text.hidden_size) for x in (q, k, v, z)], dim=1
            )
            attention.in_proj_qkvz.weight.copy_(packed.reshape(-1, text.hidden_size))
            ba = [
                remaining.pop(ap + "in_proj_" + name + ".weight").reshape(
                    nk, -1, text.hidden_size
                )
                for name in ("b", "a")
            ]
            attention.in_proj_ba.weight.copy_(
                torch.cat(ba, dim=1).reshape(-1, text.hidden_size)
            )
            remaining[ap + "conv1d.weight"] = remaining[ap + "conv1d.weight"].squeeze(1)
            for name in ("conv1d", "out_proj", "norm"):
                copy(getattr(attention, name).weight, ap + name + ".weight")
            for name in ("A_log", "dt_bias"):
                copy(getattr(attention, name), ap + name)
            attention.conv1d.weight.data = fp32_causal_conv_weight(
                attention.conv1d.weight
            )
            # Keep the existing separate-projection backend so its ordinary
            # module output exposes the raw convolution history for HF's cache.
        else:
            attention = layer.self_attn
            ap = base + "self_attn."
            for shard in ("q", "k", "v"):
                weight = attention.qkv_proj.weight
                weight.weight_loader(
                    weight, remaining.pop(ap + shard + "_proj.weight"), shard
                )
            copy(attention.o_proj.weight, ap + "o_proj.weight")
            for name in ("q_norm", "k_norm"):
                copy(getattr(attention, name).weight, ap + name + ".weight")
        for index, name in enumerate(("gate", "up")):
            weight = layer.mlp.gate_up_proj.weight
            weight.weight_loader(
                weight, remaining.pop(base + f"mlp.{name}_proj.weight"), index
            )
        copy(layer.mlp.down_proj.weight, base + "mlp.down_proj.weight")
    mapped = {}
    for name in model.visual.state_dict():
        source = name.replace("pos_embed_interp._embed.emb.weight", "pos_embed.weight")
        source = source.replace("patch_embed.proj.conv.", "patch_embed.proj.")
        source = source.replace(".mlp.fc1.", ".mlp.linear_fc1.").replace(
            ".mlp.fc2.", ".mlp.linear_fc2."
        )
        if "merger" in source:
            source = source.replace(".fc1.", ".linear_fc1.").replace(
                ".fc2.", ".linear_fc2."
            )
        mapped[name] = remaining.pop("model.visual." + source)
    model.visual.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f"Unmapped Qwen3.5 state: {sorted(remaining)}")


def make_workloads(model, inputs, config, *, case=None):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError(
            "The Qwen3.5 development workload uses one multimodal sequence"
        )
    model.inputs = inputs
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    decode_steps = 2 if continuation else 1
    prefix_length = inputs["input_ids"].shape[1] - decode_steps
    workloads = qwen3_next.make_workloads(model, inputs, config.text_config, case=case)
    for phase, workload in list(workloads.items()):
        length = prefix_length if phase == "prefill" else prefix_length + (
            int(phase.removeprefix("decode_")) if continuation else 1
        )

        def collect(result, length=length, workload=workload):
            result = workload.collect(result)
            state = model.last_state
            for i, layer in enumerate(model.model.layers):
                if layer.layer_type == "linear_attention":
                    result[f"past_key_values.{i}.conv_states"] = layer.conv_history
                    # The parent kernels store [value, key]; HF exposes [key, value].
                    if not continuation:
                        result[f"past_key_values.{i}.recurrent_states"] = state.recurrent[i][1:2].transpose(-1, -2)
                elif not continuation:
                    for name, cache in (
                        ("key", state.k_cache[i]),
                        ("value", state.v_cache[i]),
                    ):
                        if layer.self_attn.kv_layout == "HND":
                            cache = cache.transpose(1, 2)
                        result[f"past_key_values.{i}.{name}"] = (
                            cache.flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0)
                        )
            result["rope_deltas"] = model.rope_delta
            return result

        def prepare(workload=workload):
            # The model carries multimodal metadata; restore this workload's
            # inputs before replaying its prefix after any interleaved call.
            model.inputs = inputs
            model.rope_delta = None
            for layer in model.model.layers:
                if layer.layer_type == "linear_attention":
                    layer.conv_history = None
            workload.prepare()

        if continuation:
            workloads[phase] = Workload(run=workload.run, prepare=prepare, collect=collect)
        else:
            # Preserve the ordinary interface used by the existing one-decode
            # case, including state in the result of run().
            def run(workload=workload, collect=collect):
                return collect(workload.run())

            workloads[phase] = Workload(run=run, prepare=prepare)
    return workloads
