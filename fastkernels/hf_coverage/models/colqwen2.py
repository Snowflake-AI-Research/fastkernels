"""ColQwen2 image retrieval embeddings from the existing Qwen2-VL backbone."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import Linear
from . import llama, qwen2, qwen2_vl
from ..runner import Workload


class RetrievalBackbone(nn.Module):
    def __init__(self, native, config):
        super().__init__()
        self.text, self.vision = native.model.text, native.model.vision
        # This unchanged parent backend preserves HF's BF16 product rounding;
        # the fused backend narrowly fails the tested continuation key cache.
        self.text.rotary_emb.forward = self.text.rotary_emb.forward_native_2d
        self.image_token_id = config.vlm_config.image_token_id
        self.inputs = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, ids, positions):
        embeddings = self.text.embed_tokens(ids)
        if get_context().is_prefill and self.inputs.get("pixel_values") is not None:
            inputs = self.inputs
            grids = inputs["image_grid_thw"]
            valid = torch.arange(inputs["pixel_values"].shape[1], device=ids.device)[
                None, :
            ]
            valid = valid < (grids[:, 1] * grids[:, 2])[:, None]
            features = self.vision(inputs["pixel_values"][valid], grids.detach().cpu())
            embeddings[ids == self.image_token_id] = features
        # HF's retrieval wrapper supplies only inputs_embeds to Qwen2VL. Its
        # ordinary path consequently uses sequential positions on all axes.
        positions = positions[None].expand(3, -1)
        return self.text(ids, positions, inputs_embeds=embeddings)


class ColQwen2(nn.Module):
    def __init__(self, native, config):
        super().__init__()
        self.config = native.config
        self.model = RetrievalBackbone(native, config)
        self.projection = Linear(
            config.vlm_config.text_config.hidden_size, config.embedding_dim
        )
        self.normalize = L2Norm(eps=0.0)

    def project(self, hidden):
        return {"embeddings": self.normalize(self.projection(hidden))}


def build_from_config(config, device, dtype):
    native = qwen2_vl.build_from_config(config.vlm_config, device, dtype)
    return ColQwen2(native, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    remaining = dict(state)
    text = {
        name.replace("vlm.language_model.", "model."): remaining.pop(name)
        for name in list(remaining)
        if name.startswith("vlm.language_model.")
    }
    # The existing packed loader also loads an LM head. Alias that destination
    # to the same token embedding so the retrieval model gains no unused head.
    text["lm_head.weight"] = text["model.embed_tokens.weight"]
    fake_head = SimpleNamespace(embedding_op=model.model.text.embed_tokens.embedding_op)
    carrier = SimpleNamespace(
        model=model.model.text, lm_head=fake_head, config=model.config
    )
    qwen2.load_state_dict_into(carrier, text, model.config)
    mapped = {}
    for name in model.model.vision.state_dict():
        source = name.replace("patch_embed.proj.conv.", "patch_embed.proj.")
        if "merger" in source:
            source = (
                source.replace(".norm.", ".ln_q.")
                .replace(".fc1.", ".mlp.0.")
                .replace(".fc2.", ".mlp.2.")
            )
        mapped[name] = remaining.pop("vlm.visual." + source)
    model.model.vision.load_state_dict(mapped, strict=True)
    model.projection.load_state_dict(
        {
            name: remaining.pop("embedding_proj_layer." + name)
            for name in ("weight", "bias")
        },
        strict=True,
    )
    if remaining:
        raise KeyError(f"Unmapped ColQwen2 state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("The ColQwen2 development workload uses one document")
    model.model.inputs = inputs
    workloads = llama.make_workloads(
        model, inputs, model.config, output_projection=model.project
    )
    for phase, workload in list(workloads.items()):

        def run(workload=workload, phase=phase):
            result = workload.run()
            length = inputs["input_ids"].shape[1] - (phase == "prefill")
            for index, layer in enumerate(model.model.layers):
                attention = layer.self_attn.attn
                for name, cache in (
                    ("key", attention.k_cache),
                    ("value", attention.v_cache),
                ):
                    if attention.kv_layout == "HND":
                        cache = cache.transpose(1, 2)
                    result[f"past_key_values.{index}.{name}"] = (
                        cache.flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0)
                    )
            return result

        workloads[phase] = Workload(run=run, prepare=workload.prepare)
    return workloads
