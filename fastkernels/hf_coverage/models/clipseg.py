"""CLIPSeg text-conditioned segmentation, including its default returned features."""

import torch
from torch import nn

from fastkernels.hf_coverage.models import clip
from fastkernels.hf_coverage.models.depth_anything import NonoverlappingTransposeConv
from fastkernels.hf_coverage.models.mvp import EagerAttention
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU


class _Embeddings(clip.ClipVisionEmbeddings):
    def __init__(self, c):
        super().__init__(c)
        self.resize, self.grid = Interpolate(), c.image_size // c.patch_size

    def forward(self, pixels):
        projected = self.patch_embedding.proj(pixels)
        batch, width, h, w = projected.shape
        patches = projected.flatten(2).transpose(1, 2)
        positions = self.position_embedding(self.position_ids)
        if (h, w) != (self.grid, self.grid):
            spatial = positions[:, 1:].reshape(1, self.grid, self.grid, width).permute(0, 3, 1, 2)
            spatial = self.resize(spatial, size=(h, w), mode="bicubic", align_corners=False)
            positions = torch.cat((positions[:, :1], spatial.flatten(2).transpose(1, 2)), dim=1)
        return torch.cat((self.class_embedding.expand(batch, 1, -1), patches), dim=1) + positions


class _Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width//heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(width, width))
        self.attend = EagerAttention(prescale_query=False)

    def forward(self, hidden):
        batch, length, width = hidden.shape
        q, k, v = [getattr(self, name)(hidden).reshape(batch, length, self.heads, self.head_dim)
                   for name in ("q_proj", "k_proj", "v_proj")]
        return self.out_proj(self.attend(q, k, v).reshape(batch, length, width))


class _DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        width = c.reduce_dim
        self.self_attn = _Attention(width, c.decoder_num_attention_heads)
        self.layer_norm1 = LayerNorm(width, eps=c.vision_config.layer_norm_eps, promote_fp32=False)
        self.layer_norm2 = LayerNorm(width, eps=c.vision_config.layer_norm_eps, promote_fp32=False)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = Linear(width, c.decoder_intermediate_size), Linear(c.decoder_intermediate_size, width)
        self.activation = ReLU()

    def forward(self, hidden):
        hidden = self.layer_norm1(hidden + self.self_attn(hidden))
        return self.layer_norm2(hidden + self.mlp.fc2(self.activation(self.mlp.fc1(hidden))))


class CLIPSegForImageSegmentation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.clip = clip.ClipModel(c)
        self.clip.vision_model.embeddings = _Embeddings(c.vision_config)
        # Preserve HF's score-scaling/BF16 boundary; fused SDPA differs on the
        # same Q/K/V inputs and the default returned hidden states accumulate it.
        for encoder in (self.clip.vision_model.encoder, self.clip.text_model.text_model.encoder):
            for layer in encoder.layers:
                layer.attn = EagerAttention(prescale_query=False)
        self.decoder = nn.Module()
        self.decoder.film_mul, self.decoder.film_add = Linear(c.projection_dim, c.reduce_dim), Linear(c.projection_dim, c.reduce_dim)
        factor = c.vision_config.patch_size // 4
        self.decoder.transposed_convolution = nn.Sequential(
            Conv2d(c.reduce_dim, c.reduce_dim, 3, padding=1), ReLU(),
            NonoverlappingTransposeConv(c.reduce_dim, c.reduce_dim//2, factor), ReLU(),
            NonoverlappingTransposeConv(c.reduce_dim//2, 1, factor))
        self.decoder.reduces = nn.ModuleList([Linear(c.vision_config.hidden_size, c.reduce_dim) for _ in c.extract_layers])
        self.decoder.layers = nn.ModuleList([_DecoderLayer(c) for _ in c.extract_layers])
        self.product = ProductGate()

    def forward(self, pixel_values, input_ids, attention_mask=None):
        c = self.config
        vision = self.clip.vision_model
        hidden = vision.pre_layrnorm(vision.embeddings(pixel_values))
        states = [hidden]
        for layer in vision.encoder.layers:
            hidden = layer(hidden)
            states.append(hidden)
        pooled = self.clip.visual_projection(vision.post_layernorm(hidden[:, 0]))
        text = self.clip.text_model.text_model
        tokens = text.embeddings(input_ids)
        mask = text._make_causal_mask(input_ids.shape, tokens.dtype, tokens.device)
        if attention_mask is not None:
            padding = tokens.new_zeros(input_ids.shape[0], 1, 1, input_ids.shape[1])
            padding.masked_fill_(~attention_mask[:, None, None, :].bool(), torch.finfo(tokens.dtype).min)
            mask = mask + padding
        tokens = text.final_layer_norm(text.encoder(tokens, attention_mask=mask))
        text_pool = tokens[torch.arange(tokens.shape[0], device=tokens.device), input_ids.argmax(dim=-1)]
        condition = self.clip.text_projection(text_pool)
        decoded = None
        decoder_states = []
        for i, (index, reduce, layer) in enumerate(zip(reversed(c.extract_layers), self.decoder.reduces, self.decoder.layers)):
            reduced = reduce(states[index+1])
            decoded = reduced if decoded is None else reduced + decoded
            if i == c.conditional_layer:
                scale = self.decoder.film_mul(condition)[:, None].expand_as(decoded)
                decoded = self.product(torch.cat((scale, decoded), dim=-1)) + self.decoder.film_add(condition)[:, None]
            if not decoder_states:
                decoder_states.append(decoded)
            decoded = layer(decoded)
            decoder_states.append(decoded)
        side = int((decoded.shape[1]-1)**0.5)
        logits = self.decoder.transposed_convolution(decoded[:, 1:].transpose(1, 2).reshape(decoded.shape[0], -1, side, side)).squeeze(1)
        outputs = {"logits": logits, "conditional_embeddings": condition, "pooled_output": pooled,
                   "vision_model_output.last_hidden_state": hidden, "vision_model_output.pooler_output": pooled,
                   "decoder_output.logits": logits}
        outputs.update({f"vision_model_output.hidden_states.{i}": value for i, value in enumerate(states)})
        outputs.update({f"decoder_output.hidden_states.{i}": value for i, value in enumerate(decoder_states)})
        return outputs


def build_from_config(config, device, dtype):
    if (not config.use_complex_transposed_convolution or config.text_config.eos_token_id != 2
            or config.vision_config.patch_size != 16):
        raise ValueError("The documented refined checkpoint uses the complex patch16 decoder and legacy CLIP pooling")
    return CLIPSegForImageSegmentation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    clip_weights = {name.removeprefix("clip."): value for name, value in state_dict.items() if name.startswith("clip.")}
    clip.load_state_dict_into(model.clip, clip_weights, config)
    decoder = {name.removeprefix("decoder."): value for name, value in state_dict.items() if name.startswith("decoder.")}
    if len(clip_weights) + len(decoder) != len(state_dict):
        raise ValueError("Unmapped CLIPSeg state outside CLIP/decoder")
    for index in (2, 4):
        prefix = f"transposed_convolution.{index}."
        weight = decoder.pop(prefix+"weight")
        layer = model.decoder.transposed_convolution[index]
        decoder[prefix+"projection.weight"] = weight.permute(1, 2, 3, 0).reshape(-1, weight.shape[0]).contiguous()
        decoder[prefix+"projection.bias"] = decoder.pop(prefix+"bias").repeat_interleave(layer.factor**2)
    model.decoder.load_state_dict(decoder, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
