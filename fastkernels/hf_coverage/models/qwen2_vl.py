"""Qwen VL wrappers around existing vision towers and multimodal text decoders."""

from dataclasses import fields
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L4.qwen2_vl import (
    Qwen2VLConfig,
    Qwen2VLVisionConfig,
    Qwen2VLForConditionalGeneration,
)
from . import llama, qwen2, qwen3
from .qwen2_moe import load_qwen_moe_weights
from ..runner import Workload


def multimodal_positions(ids, types, inputs, config):
    """Build position metadata, including Qwen3's separately timestamped frames."""
    device = ids.device
    grids = {}
    for kind, key in ((1, "image_grid_thw"), (2, "video_grid_thw")):
        values = inputs[key].tolist()
        if kind == 2 and config.model_type.startswith("qwen3"):
            values = [[1, h, w] for t, h, w in values for _ in range(t)]
        grids[kind] = iter(values)
    seconds = iter(inputs.get("second_per_grid_ts", [1] * ids.numel()))
    merge = config.vision_config.spatial_merge_size
    parts, current, start = [], 0, 0
    values = types.tolist()
    for end in range(1, len(values) + 1):
        if end < len(values) and values[end] == values[start]:
            continue
        kind = values[start]
        if kind == 0:
            parts.append(
                torch.arange(end - start, device=device)[None].expand(3, -1) + current
            )
            current += end - start
        else:
            t, h, w = next(grids[kind])
            h //= merge
            w //= merge
            interval = 1
            if kind == 2 and config.model_type == "qwen2_5_vl":
                interval = config.vision_config.tokens_per_second * int(next(seconds))
            temporal = (
                torch.arange(t, device=device).repeat_interleave(h * w) * interval
            )
            height = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            width = torch.arange(w, device=device).repeat(h * t)
            position = torch.stack((temporal, height, width)) + current
            if position.shape[1] != end - start:
                raise ValueError("Visual grid and placeholder run disagree")
            parts.append(position)
            current += max(h, w)
        start = end
    positions = torch.cat(parts, dim=1)
    return positions, positions.max() + 1 - ids.numel()


class MultimodalBackbone(nn.Module):
    def __init__(self, native, config):
        super().__init__()
        self.text, self.vision = native.model, native.visual
        self.config = config
        self.inputs = None
        self.rope_delta = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        if not get_context().is_prefill:
            positions = positions[None].expand(3, -1) + self.rope_delta
            return self.text(input_ids, positions)
        embeddings = self.text.embed_tokens(input_ids)
        inputs = self.inputs
        deepstack = None
        for token, pixels, grid in (
            (self.config.image_token_id, "pixel_values", "image_grid_thw"),
            (self.config.video_token_id, "pixel_values_videos", "video_grid_thw"),
        ):
            features = self.vision(inputs[pixels], inputs[grid].detach().cpu())
            mask = input_ids == token
            width = embeddings.shape[-1]
            embeddings[mask] = features[:, :width]
            if features.shape[-1] > width:
                levels = features[:, width:].split(width, dim=-1)
                if deepstack is None:
                    deepstack = [torch.zeros_like(embeddings) for _ in levels]
                for target, source in zip(deepstack, levels):
                    target[mask] = source
        positions, self.rope_delta = multimodal_positions(
            input_ids,
            inputs["mm_token_type_ids"][0, : input_ids.numel()],
            inputs,
            self.config,
        )
        self.rope_delta = self.rope_delta.reshape(1, 1)
        kwargs = {"deepstack_embeds": deepstack} if deepstack is not None else {}
        return self.text(input_ids, positions, inputs_embeds=embeddings, **kwargs)


class QwenVL(nn.Module):
    def __init__(self, native, config):
        super().__init__()
        self.config = native.config
        self.model = MultimodalBackbone(native, config)
        self.lm_head = native.lm_head


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    rope = text.rope_parameters
    if (
        text.hidden_act != "silu"
        or rope["rope_type"] != "default"
        or getattr(text, "use_sliding_window", False)
    ):
        raise ValueError(
            "Selected Qwen VL checkpoints use SiLU and default full-attention M-RoPE"
        )
    common = {
        name: getattr(text, name)
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
        )
    }
    common.update(
        head_dim=getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads,
        rope_theta=rope["rope_theta"],
        mrope_section=rope["mrope_section"],
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        dtype=dtype,
    )
    if config.model_type.startswith("qwen3"):
        from fastkernels.tasks.baseline.L4.qwen3_vl import (
            Qwen3VLConfig,
            Qwen3VLVisionConfig,
            Qwen3VLForConditionalGeneration,
        )

        vc = Qwen3VLVisionConfig(
            **{f.name: getattr(vision, f.name) for f in fields(Qwen3VLVisionConfig)}
        )
        moe = config.model_type == "qwen3_vl_moe"
        extra = (
            {
                name: getattr(text, name)
                for name in (
                    "num_experts_per_tok",
                    "moe_intermediate_size",
                    "norm_topk_prob",
                )
            }
            if moe
            else {}
        )
        if moe:
            # The pinned HF configuration serializes the expert-count alias.
            extra["num_experts"] = text.num_local_experts
        native = Qwen3VLForConditionalGeneration(
            Qwen3VLConfig(**common, vision=vc, is_moe=moe, **extra)
        )
    else:
        vc = Qwen2VLVisionConfig(
            **{
                f.name: getattr(vision, f.name)
                for f in fields(Qwen2VLVisionConfig)
                if hasattr(vision, f.name)
            }
        )
        native = Qwen2VLForConditionalGeneration(Qwen2VLConfig(**common, vision=vc))
        if config.model_type == "qwen2_5_vl":
            from fastkernels.tasks.baseline.L4.qwen2_5_omni import (
                Qwen2_5OmniVisionConfig,
                Qwen2_5VisionTransformer,
            )

            vc = Qwen2_5OmniVisionConfig(
                **{
                    f.name: getattr(vision, f.name)
                    for f in fields(Qwen2_5OmniVisionConfig)
                    if hasattr(vision, f.name)
                }
            )
            native.visual = Qwen2_5VisionTransformer(vc)
    return QwenVL(native, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {
        name.replace("model.language_model.", "model."): remaining.pop(name)
        for name in list(remaining)
        if name.startswith("model.language_model.")
    }
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    carrier = SimpleNamespace(
        model=model.model.text, lm_head=model.lm_head, config=model.config
    )
    if config.model_type == "qwen3_vl_moe":
        load_qwen_moe_weights(
            carrier, text, config.text_config, shared_expert=False, qk_norm=True
        )
    elif config.model_type == "qwen3_vl":
        qwen3.load_state_dict_into(carrier, text, model.config)
    else:
        qwen2.load_state_dict_into(carrier, text, model.config)
    vision = model.model.vision
    mapped = {}
    for name, target in vision.state_dict().items():
        if name.endswith("inv_freq"):
            mapped[name] = target
            continue
        source = name.replace("pos_embed_interp._embed.emb.weight", "pos_embed.weight")
        source = source.replace("patch_embed.proj.conv.", "patch_embed.proj.")
        source = (
            source.replace(".mlp.fc1.", ".mlp.linear_fc1.").replace(
                ".mlp.fc2.", ".mlp.linear_fc2."
            )
            if config.model_type.startswith("qwen3")
            else source
        )
        if "merger" in source:
            source = source.replace(
                ".norm.",
                ".norm." if config.model_type.startswith("qwen3") else ".ln_q.",
            )
            source = source.replace(
                ".fc1.",
                ".linear_fc1." if config.model_type.startswith("qwen3") else ".mlp.0.",
            )
            source = source.replace(
                ".fc2.",
                ".linear_fc2." if config.model_type.startswith("qwen3") else ".mlp.2.",
            )
        if config.model_type == "qwen2_5_vl" and ".mlp.gate_up_proj." in source:
            value = torch.cat(
                [
                    remaining.pop(
                        "model.visual." + source.replace("gate_up_proj", part)
                    )
                    for part in ("gate_proj", "up_proj")
                ],
                dim=0,
            )
        else:
            value = remaining.pop("model.visual." + source)
        if value.shape != target.shape:
            raise ValueError(
                f"Visual weight shape mismatch: {source}: {value.shape} != {target.shape}"
            )
        mapped[name] = value
    vision.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f"Unmapped Qwen VL weights: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("The Qwen VL development case uses one multimodal sequence")
    model.model.inputs = inputs
    workloads = llama.make_workloads(model, inputs, model.config)
    for phase, workload in list(workloads.items()):

        def run(workload=workload, phase=phase):
            output = workload.run()
            length = inputs["input_ids"].shape[1] - (phase == "prefill")
            for index, layer in enumerate(model.model.layers):
                attention = layer.self_attn.attn
                for name, cache in (
                    ("key", attention.k_cache),
                    ("value", attention.v_cache),
                ):
                    if attention.kv_layout == "HND":
                        cache = cache.transpose(1, 2)
                    output[f"past_key_values.{index}.{name}"] = (
                        cache.flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0)
                    )
            output["rope_deltas"] = model.model.rope_delta
            return output

        workloads[phase] = Workload(run=run, prepare=workload.prepare)
    return workloads
