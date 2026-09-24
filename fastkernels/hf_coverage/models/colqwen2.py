"""ColQwen2 image retrieval embeddings from the existing Qwen2-VL backbone."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
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
        self.retrieval_attention = DenseAttention(backend="sdpa")
        self.matmul = Matmul()

    def project(self, hidden):
        return {"embeddings": self.normalize(self.projection(hidden))}

    def retrieval_forward(self, input_ids, attention_mask=None, pixel_values=None,
                          image_grid_thw=None):
        """Document/query retrieval, including processor-supplied padding.

        Use the same loaded backbone weights and unchanged native norm/RoPE/
        SwiGLU callables. Dense SDPA accepts the supplied padding mask; the
        historical paged continuation workload has no such mask interface.
        """
        text = self.model.text
        batch, length = input_ids.shape
        hidden = text.embed_tokens(input_ids)
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("Image retrieval requires its patch grid")
            patches = torch.arange(pixel_values.shape[1], device=input_ids.device)[None]
            valid = patches < (image_grid_thw[:, 1] * image_grid_thw[:, 2])[:, None]
            features = self.model.vision(pixel_values[valid], image_grid_thw.detach().cpu())
            hidden[input_ids == self.model.image_token_id] = features.to(hidden.dtype)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Retrieval attention mask must match token IDs")
        # The native wrapper passes inputs_embeds, so each example uses ordinary
        # sequential positions on all three axes, including padded positions.
        sequence = torch.arange(length, device=input_ids.device)
        positions = sequence.repeat(batch)[None].expand(3, -1)
        allowed = (sequence[None, :] <= sequence[:, None])[None, None]
        allowed = allowed & attention_mask[:, None, None, :].bool()
        # Native SDPA uses a Boolean mask: fully masked leading padding rows
        # produce zero context rather than a uniform finite-min softmax.
        mask = allowed
        caches = {}

        def normalize(module, values):
            return module.forward_native(values, module.weight, module.eps, module.hidden_size)

        for index, layer in enumerate(text.layers):
            normalized = normalize(layer.input_layernorm, hidden)
            attention = layer.self_attn
            sizes = [attention.num_heads * attention.head_dim] + [attention.num_kv_heads * attention.head_dim] * 2
            weights = attention.qkv_proj.weight.split(sizes)
            biases = attention.qkv_proj.bias.split(sizes)
            query, key, value = (self.matmul(normalized, weight, bias)
                                 for weight, bias in zip(weights, biases))
            query, key = text.rotary_emb.forward_native_2d(
                positions, query.reshape(batch * length, -1), key.reshape(batch * length, -1))
            query = query.reshape(batch, length, attention.num_heads, attention.head_dim)
            key, value = (tensor.reshape(batch, length, attention.num_kv_heads, attention.head_dim)
                          for tensor in (key, value))
            caches[f"past_key_values.{index}.key"] = key.transpose(1, 2)
            caches[f"past_key_values.{index}.value"] = value.transpose(1, 2)
            groups = attention.num_heads // attention.num_kv_heads
            output = self.retrieval_attention(
                query, key.repeat_interleave(groups, 2), value.repeat_interleave(groups, 2),
                attn_mask=mask)
            hidden = hidden + attention.o_proj(output.reshape(batch, length, -1))
            normalized = normalize(layer.post_attention_layernorm, hidden)
            gate, up = (self.matmul(normalized, weight) for weight in layer.mlp.gate_up_proj.weight.chunk(2))
            activated = layer.mlp.act_fn.forward_native(torch.cat((gate, up), -1))
            hidden = hidden + layer.mlp.down_proj(activated)
        embeddings = self.project(normalize(text.norm, hidden))["embeddings"]
        embeddings = embeddings.masked_fill(~attention_mask[..., None].bool(), 0)
        return dict(embeddings=embeddings, **caches)


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
    if inputs.get("pixel_values") is None:
        return {"forward": Workload(run=lambda: model.retrieval_forward(**inputs))}
    if inputs.get("attention_mask") is not None:
        raise ValueError("Padded ColQwen2 retrieval requires the ordinary forward workload")
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
