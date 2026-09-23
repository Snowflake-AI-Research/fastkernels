"""UDOP document/image fusion with native spatial biases and T5 generation."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from .t5 import T5ForConditionalGeneration, load_state_dict_into as load_t5


class Udop(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config, output_scale=config.d_model ** -0.5)
        self.config = config
        self.patch_embed = Conv2d(config.num_channels, config.d_model, config.patch_size, stride=config.patch_size)
        self.cell_x = Embedding(config.max_2d_position_embeddings, config.d_model)
        self.cell_y = Embedding(config.max_2d_position_embeddings, config.d_model)
        self.relative_x = Embedding(32, config.num_heads)
        self.relative_y = Embedding(32, config.num_heads)

    def combine(self, input_ids, bbox, pixels, attention_mask):
        text = self.shared(input_ids)
        image = self.patch_embed(pixels).flatten(2).transpose(1, 2)
        side = self.config.image_size // self.config.patch_size
        # OCR boxes determine fixed spatial gathering and masks, not learned activation scores.
        x = ((bbox[..., 0] + bbox[..., 2]) / 2 * side).floor().long().clamp(0, side - 1)
        y = ((bbox[..., 1] + bbox[..., 3]) / 2 * side).floor().long().clamp(0, side - 1)
        indices = x + y * side
        bbox = bbox.double()
        endpoints = (bbox.mean(-1) == 0) | (bbox.mean(-1) == 1)
        selected = image.gather(1, indices.unsqueeze(-1).expand(-1, -1, image.shape[-1]))
        selected = selected.masked_fill(endpoints.unsqueeze(-1), 0)
        text = text + selected
        # Match HF's CPU preparation before spatial distances are bucketed.
        grid = (torch.arange(side + 1, dtype=torch.float32, device="cpu") / side).to(pixels.device)
        iy, ix = torch.meshgrid(torch.arange(side, device=pixels.device),
                                torch.arange(side, device=pixels.device), indexing="ij")
        image_boxes = torch.stack((grid[ix], grid[iy], grid[ix + 1], grid[iy + 1]), dim=-1).reshape(-1, 4)
        images, boxes, masks = [], [], []
        for row in range(input_ids.shape[0]):
            keep = torch.ones(image.shape[1], dtype=torch.bool, device=pixels.device)
            keep[indices[row]] = False
            retained = image[row, keep]
            count = retained.shape[0]
            images.append(torch.cat((retained, image.new_zeros(image.shape[1] - count, image.shape[-1]))))
            boxes.append(torch.cat((image_boxes[keep], bbox.new_zeros(image.shape[1] - count, 4))))
            masks.append(torch.cat((attention_mask.new_ones(count), attention_mask.new_zeros(image.shape[1] - count))))
        return (torch.cat((text, torch.stack(images)), dim=1),
                torch.cat((bbox, torch.stack(boxes)), dim=1),
                torch.cat((attention_mask, torch.stack(masks)), dim=1))

    def encode(self, input_ids, bbox, pixel_values, attention_mask):
        hidden, bbox, mask = self.combine(input_ids, bbox, pixel_values, attention_mask)
        cell_ids = (bbox.clamp(0, 1) * (self.config.max_2d_position_embeddings - 1)).long()
        cell = self.cell_x(cell_ids[..., 0]) + self.cell_y(cell_ids[..., 1])
        cell = cell + self.cell_x(cell_ids[..., 2])
        cell = cell + self.cell_y(cell_ids[..., 3])
        hidden = hidden + cell
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        buckets = T5SelfAttention._relative_position_bucket(positions[None, :] - positions[:, None])
        table = self.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
        bias = table(buckets).permute(2, 0, 1).unsqueeze(0)
        for axes, table in (((0, 2), self.relative_x), ((1, 3), self.relative_y)):
            positions = bbox[..., list(axes)].mean(-1)
            distance = ((positions[:, None, :] - positions[:, :, None]) * 100).long()
            buckets = T5SelfAttention._relative_position_bucket(distance, max_distance=100)
            bias = table(buckets).permute(0, 3, 1, 2) + bias
        mask_bias = (1 - mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        bias = bias + mask_bias
        for block in self.encoder.block:
            hidden, bias = block(hidden, position_bias=bias)
        return self.encoder.final_layer_norm(hidden), mask

    def forward(self, input_ids=None, bbox=None, pixel_values=None, attention_mask=None,
                decoder_input_ids=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_values=None):
        memory = encoder_hidden_states
        if memory is None:
            memory, encoder_attention_mask = self.encode(input_ids, bbox, pixel_values, attention_mask)
        output = super().forward(
            None, decoder_input_ids, encoder_hidden_states=memory,
            attention_mask=encoder_attention_mask, past_key_values=past_key_values,
        )
        # The expanded image/text mask belongs to the encoder result and must
        # accompany its states when they are reused for cached continuation.
        output["encoder_attention_mask"] = encoder_attention_mask
        return output


def build_from_config(config, device, dtype):
    if config.relative_bias_args != [{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}] or config.feed_forward_proj != "relu":
        raise ValueError("The selected UDOP checkpoint uses ungated ReLU and all three default relative biases")
    return Udop(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    for destination, source in (
        (model.patch_embed.weight, "patch_embed.proj.weight"),
        (model.patch_embed.bias, "patch_embed.proj.bias"),
        (model.cell_x.emb.weight, "encoder.cell_2d_embedding.x_position_embeddings.weight"),
        (model.cell_y.emb.weight, "encoder.cell_2d_embedding.y_position_embeddings.weight"),
        (model.relative_x.emb.weight, "encoder.relative_bias.biases.1.relative_attention_bias.weight"),
        (model.relative_y.emb.weight, "encoder.relative_bias.biases.2.relative_attention_bias.weight"),
    ):
        destination.copy_(remaining.pop(source))
    for field in ("weight", "bias"):
        value = remaining.pop(f"encoder.embed_patches.proj.{field}")
        if not torch.equal(value, state_dict[f"patch_embed.proj.{field}"]):
            raise ValueError("UDOP tied patch projections disagree")
        # Decoder image projection exists in HF but receives no pixels in the selected task.
        remaining.pop(f"decoder.embed_patches.proj.{field}")
    relative = state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    for stack in ("encoder", "decoder"):
        if not torch.equal(remaining.pop(f"{stack}.relative_bias.biases.0.relative_attention_bias.weight"), relative):
            raise ValueError("UDOP shared relative-bias aliases disagree")
    # The decoder explicitly bypasses its spatial-bias aggregator.
    for index in (1, 2):
        remaining.pop(f"decoder.relative_bias.biases.{index}.relative_attention_bias.weight")
    load_t5(model, remaining, config)


def make_workloads(model, inputs, config, case=None):
    ids = inputs["decoder_input_ids"]
    state = {}

    def call(start, end, previous=None):
        if previous is None:
            return model(**dict(inputs, decoder_input_ids=ids[:, start:end]))
        return model(
            decoder_input_ids=ids[:, start:end],
            encoder_hidden_states=previous["encoder_last_hidden_state"],
            encoder_attention_mask=previous["encoder_attention_mask"],
            past_key_values=previous["past_key_values"],
        )

    def retain(output):
        state["output"] = output
        return {"logits": output["logits"]}

    def collect(_):
        output = state.pop("output")
        return {"logits": output["logits"],
                "encoder_last_hidden_state": output["encoder_last_hidden_state"],
                **seq2seq_cache_outputs(output["past_key_values"])}

    if case is None or case["workload"] == "forward":
        return {"forward": Workload(run=lambda: retain(call(0, ids.shape[1])), collect=collect)}
    prefix_length = ids.shape[1] - 2
    if prefix_length < 1:
        raise ValueError("UDOP continuation needs a decoder prefix and two new tokens")

    def prepare(step):
        state["previous"] = call(0, prefix_length)
        if step == 1:
            state["previous"] = call(prefix_length, prefix_length + 1, state["previous"])

    workloads = {"prefill": Workload(run=lambda: retain(call(0, prefix_length)), collect=collect)}
    for step in range(2):
        workloads[f"decode_{step + 1}"] = Workload(
            run=lambda step=step: retain(call(prefix_length + step, prefix_length + step + 1, state["previous"])),
            prepare=lambda step=step: prepare(step), collect=collect,
        )
    return workloads
