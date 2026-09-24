"""Cohere2-Vision image/text assembly with native optional vision pooling."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul

from . import cohere2
from .modernvbert import VisionTower, base_source
from ..runner import Workload


class Cohere2VisionForConditionalGeneration(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.image_token_id = config.image_token_id
        self.factor = config.downsample_factor
        self.vision = VisionTower(config.vision_config).to(device=device, dtype=dtype)
        for layer in self.vision.encoder.layers:
            layer.self_attn.attn = DenseAttention(backend="sdpa")
        self.linear_1 = Linear(config.vision_config.hidden_size * self.factor**2,
                               config.alignment_intermediate_size).to(device=device, dtype=dtype)
        self.linear_2 = Linear(config.alignment_intermediate_size // 2,
                               config.text_config.hidden_size).to(device=device, dtype=dtype)
        self.language_model = cohere2.build_from_config(config.text_config, device, dtype)

    def forward(self, input_ids, pixel_values=None, past_key_values=None):
        if self.training:
            raise RuntimeError("Cohere2-Vision coverage supports inference only")
        text = self.language_model
        hidden = text.embed_tokens(input_ids)
        output = {}
        if pixel_values is not None:
            # VisionTower preserves the configured optional pool before image projection.
            image = self.vision(pixel_values)
            batch, length, width = image.shape
            side, factor = int(length**0.5), self.factor
            if side * side != length or side % factor:
                raise ValueError("Pixel shuffle requires a square grid divisible by its factor")
            image = image.reshape(batch, side, side // factor, width * factor)
            image = image.permute(0, 2, 1, 3).reshape(batch, side // factor, side // factor, -1)
            image = image.permute(0, 2, 1, 3)
            value, gate = self.linear_1(image).chunk(2, dim=-1)
            image = self.linear_2(SiluAndMul.forward_native(torch.cat((gate, value), dim=-1)))
            mask = (input_ids == self.image_token_id)[..., None].expand_as(hidden)
            if hidden[mask].numel() != image.numel():
                raise ValueError("Image placeholder count must equal projected image feature count")
            hidden = hidden.masked_scatter(mask, image.to(hidden.dtype))
            output["image_hidden_states"] = image
        start = 0 if past_key_values is None else past_key_values.seen_tokens
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + start
        states = []
        for index, layer in enumerate(text.layers):
            hidden, state = layer(hidden, positions, text.rotary.cos_sin_cache,
                                  None if past_key_values is None else past_key_values.layers[index])
            states.append(state)
        # HF's multimodal wrapper calls the text base model, then its tied head;
        # it does not use Cohere2ForCausalLM's additional logit_scale.
        output["logits"] = text.lm_head(text.norm(hidden))
        output["past_key_values"] = cohere2.Cache(tuple(states), start + input_ids.shape[1])
        return output


def build_from_config(config, device, dtype):
    vision = config.vision_config
    if (vision.model_type != "siglip_vision_model" or config.text_config.model_type != "cohere2"
            or not config.tie_word_embeddings
            or vision.hidden_act != "gelu_pytorch_tanh" or vision.num_channels != 3
            or config.downsample_factor != 2 or config.alignment_intermediate_size % 2
            or not 0 <= config.image_token_id < config.text_config.vocab_size):
        raise ValueError("Cohere2-Vision requires SigLIP, Cohere2 and factor-2 SwiGLU alignment")
    return Cohere2VisionForConditionalGeneration(config, device, dtype).eval()


def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict["lm_head.weight"], state_dict["model.language_model.embed_tokens.weight"]):
        raise ValueError("Cohere2-Vision tied input and output embeddings disagree")
    mapped, used = {}, set()
    for name, target in model.state_dict().items():
        if name.startswith(("vision.pool.q.", "vision.pool.kv.")):
            source = "model.vision_tower.head.attention.in_proj_" + name.rsplit(".", 1)[-1]
            width = config.vision_config.hidden_size
            mapped[name] = state_dict[source][:width] if ".q." in name else state_dict[source][width:]
        else:
            if name.startswith("vision."):
                suffix = base_source(name.replace("vision.", "vision_model.", 1)).removeprefix("vision_model.")
                source = "model.vision_tower." + suffix
            elif name.startswith(("linear_1.", "linear_2.")):
                source = "model.multi_modal_projector." + name
            elif name == "language_model.lm_head.weight":
                source = "lm_head.weight"
            else:
                source = "model." + name.replace(".emb.weight", ".weight")
            mapped[name] = state_dict[source].reshape(target.shape)
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f"Unmapped Cohere2-Vision weights: {sorted(set(state_dict) - used)}")
    model.load_state_dict(mapped, strict=True)
    for module in model.modules():
        if isinstance(module, LayerNorm):
            module._cast_done = False


def flatten(output):
    result = {name: value for name, value in output.items() if name != "past_key_values"}
    for index, (key, value) in enumerate(output["past_key_values"].layers):
        result[f"past_key_values.{index}.key"] = key
        result[f"past_key_values.{index}.value"] = value
    return result


def make_workloads(model, inputs, config, case=None):
    if set(inputs) != {"input_ids", "pixel_values"}:
        raise ValueError("Cohere2-Vision case selects unpadded token and image inputs")
    if case is None or case.get("workload") != "causal_lm_continuation":
        return {"forward": Workload(run=lambda: flatten(model(**inputs)))}
    ids = inputs["input_ids"]
    prefix = ids.shape[1] - 2
    if ids.ndim != 2 or prefix < 1:
        raise ValueError("Continuation requires a prefix and two supplied tokens")
    state = {}

    def initial():
        return model(ids[:, :prefix], pixel_values=inputs["pixel_values"])

    def advance(index, previous):
        return model(ids[:, prefix + index:prefix + index + 1], past_key_values=previous)

    def prepare(index):
        previous = initial()["past_key_values"]
        for step in range(index):
            previous = advance(step, previous)["past_key_values"]
        state["previous"] = previous

    return {
        "prefill": Workload(run=lambda: flatten(initial())),
        "decode_1": Workload(run=lambda: flatten(advance(0, state["previous"])), prepare=lambda: prepare(0)),
        "decode_2": Workload(run=lambda: flatten(advance(1, state["previous"])), prepare=lambda: prepare(1)),
    }
