"""MiniCPM-V 4.6 packed vision, window merger and Qwen3.5 hybrid text."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import AttnBackendConfig, get_context, set_attn_backend_config
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead
from fastkernels.tasks.baseline.L4.qwen3_next import Qwen3NextConfig
from . import qwen3_5, qwen3_next
from ..runner import Workload, config_values


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.dim = config.hidden_size // self.heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, hidden, lengths):
        shape = (*hidden.shape[:-1], self.heads, self.dim)
        q, k, v = [getattr(self, name)(hidden).reshape(shape) for name in ("q_proj", "k_proj", "v_proj")]
        outputs, start = [], 0
        for length in lengths:
            end = start + length
            outputs.append(self.attention(q[:, start:end], k[:, start:end], v[:, start:end]))
            start = end
        return self.out_proj(torch.cat(outputs, 1).reshape_as(hidden))


class MLP(nn.Module):
    def __init__(self, hidden, intermediate, output, approximate="tanh"):
        super().__init__()
        self.fc1, self.fc2 = Linear(hidden, intermediate), Linear(intermediate, output)
        self.activation = GELU(approximate)

    def forward(self, hidden):
        return self.fc2(self.activation(self.fc1(hidden)))


class VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.layer_norm2 = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.hidden_size)

    def forward(self, hidden, lengths):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden), lengths)
        return hidden + self.mlp(self.layer_norm2(hidden))


def pack_windows(hidden, height, width, kernel):
    kh, kw = kernel
    return hidden.reshape(height // kh, kh, width // kw, kw, hidden.shape[-1]).permute(0, 2, 1, 3, 4)


class WindowMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kernel = tuple(config.window_kernel_size)
        count = self.kernel[0] * self.kernel[1]
        self.self_attn = Attention(config)
        self.layer_norm1 = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.pre_norm = LayerNorm(config.hidden_size * count, config.layer_norm_eps)
        self.linear_1 = Linear(config.hidden_size * count, config.intermediate_size * count)
        self.linear_2 = Linear(config.intermediate_size * count, config.hidden_size)
        self.activation = GELU("tanh")
        self.average = AvgPool2d((1, count), stride=(1, count))

    def forward(self, hidden, sizes):
        offsets, start, indices = [], 0, []
        kh, kw = self.kernel
        for height, width in sizes:
            index = torch.arange(height * width, device=hidden.device).reshape(height, width, 1)
            indices.append(pack_windows(index, height, width, self.kernel).flatten() + start)
            offsets.append((start, start + height * width))
            start += height * width
        index = torch.cat(indices)
        normalized = self.layer_norm1(hidden)[:, index]
        update = self.self_attn(normalized, [kh * kw] * (index.numel() // (kh * kw)))
        inverse = torch.empty_like(index)
        inverse[index] = torch.arange(index.numel(), device=index.device)
        hidden = hidden + update[:, inverse]
        parts = []
        for (height, width), (start, end) in zip(sizes, offsets):
            packed = pack_windows(hidden[0, start:end], height, width, self.kernel)
            windows = packed.reshape(-1, kh * kw, hidden.shape[-1])
            residual = self.average(windows.transpose(1, 2)[:, :, None])[:, :, 0, 0]
            merged = self.pre_norm(windows.flatten(1))
            parts.append(self.linear_2(self.activation(self.linear_1(merged))) + residual)
        return torch.cat(parts)[None]


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Module()
        self.embeddings.patch_embedding = Conv2d(config.num_channels, config.hidden_size,
                                                 config.patch_size, stride=config.patch_size)
        self.side = config.image_size // config.patch_size
        self.embeddings.position_embedding = Embedding(self.side ** 2, config.hidden_size)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(VisionLayer(config) for _ in range(config.num_hidden_layers))
        self.vit_merger = WindowMerger(config)
        self.post_layernorm = LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, pixels, target_sizes):
        sizes = target_sizes.tolist()
        hidden = self.embeddings.patch_embedding(pixels).flatten(2).transpose(1, 2)
        boundaries = torch.arange(1 / self.side, 1.0, 1 / self.side, device="cpu")
        positions = []
        for height, width in sizes:
            rows = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / height, device="cpu"), boundaries, right=True)
            columns = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / width, device="cpu"), boundaries, right=True)
            positions.append((rows[:, None] * self.side + columns).flatten())
        hidden = hidden + self.embeddings.position_embedding(torch.cat(positions).to(hidden.device))[None]
        for index, layer in enumerate(self.encoder.layers):
            hidden = layer(hidden, [height * width for height, width in sizes])
            if index == self.config.insert_layer_id:
                hidden = self.vit_merger(hidden, sizes)
                sizes = [(height // 2, width // 2) for height, width in sizes]
        return self.post_layernorm(hidden), sizes


class DownsampleMLP(nn.Module):
    def __init__(self, hidden, output):
        super().__init__()
        self.pre_norm = LayerNorm(4 * hidden, 1e-6)
        self.linear_1, self.linear_2 = Linear(4 * hidden, 4 * hidden), Linear(4 * hidden, output)
        self.activation = GELU()

    def forward(self, hidden):
        return self.linear_2(self.activation(self.linear_1(self.pre_norm(hidden))))


class Model(nn.Module):
    def __init__(self, config, text):
        super().__init__()
        self.config, self.hf_config = text, config
        local = config_values(config.to_dict())
        local.text_config.rope_parameters.setdefault("mrope_section", [11, 11, 10])
        self.model = qwen3_5.HybridBackbone(local, text)
        self.vision_tower = Vision(config.vision_config)
        self.merger = nn.Module()
        self.merger.mlp = nn.ModuleList([DownsampleMLP(config.vision_config.hidden_size, text.hidden_size)])
        self.lm_head = ParallelLMHead(text.vocab_size, text.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight
        self.inputs = self.last_state = None

    def features(self, pixels, sizes):
        hidden, sizes = self.vision_tower(pixels, sizes)
        parts, start = [], 0
        for height, width in sizes:
            end = start + height * width
            packed = pack_windows(hidden[0, start:end], height, width, self.hf_config.merge_kernel_size)
            parts.append(self.merger.mlp[0](packed.reshape(-1, 4 * hidden.shape[-1])))
            start = end
        return torch.cat(parts)

    def forward(self, ids, positions, state_manager):
        hidden = self.model.embed_tokens(ids)
        if get_context().is_prefill:
            for token, pixels, sizes in (
                (self.hf_config.image_token_id, "pixel_values", "target_sizes"),
                (self.hf_config.video_token_id, "pixel_values_videos", "target_sizes_videos"),
            ):
                if pixels in self.inputs:
                    hidden[ids == token] = self.features(self.inputs[pixels], self.inputs[sizes])
        self.model.rotary.positions = positions[None].expand(3, -1)
        for layer in self.model.layers:
            hidden = layer(hidden, positions, self.model.rotary, state_manager)
        self.last_state = state_manager
        return self.model.norm(hidden)


def build_from_config(config, device, dtype):
    if (config.downsample_mode != "16x" or config.merger_times != 1
            or list(config.merge_kernel_size) != [2, 2]
            or list(config.vision_config.window_kernel_size) != [2, 2]
            or config.vision_config.hidden_act != "gelu_pytorch_tanh"
            or not 0 <= config.insert_layer_id < config.vision_config.num_hidden_layers):
        raise ValueError("Preserve the selected MiniCPM full 16x vision merger pipeline")
    set_attn_backend_config(AttnBackendConfig.auto_detect())
    text = Qwen3NextConfig._from_hf(config.text_config)
    text.dtype = dtype
    return Model(config, text).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    remaining = dict(state)
    for module, prefix in ((model.vision_tower, "model.vision_tower."), (model.merger, "model.merger.")):
        mapped = {name: remaining.pop(prefix + name.replace(".position_embedding.emb.", ".position_embedding."))
                  for name in module.state_dict()}
        module.load_state_dict(mapped, strict=True)
    proxy = SimpleNamespace(model=model.model, lm_head=model.lm_head, visual=nn.Module())
    qwen3_5.load_state_dict_into(proxy, remaining, config)


def make_workloads(model, inputs, config):
    model.inputs = inputs
    workloads = qwen3_next.make_workloads(model, inputs, config.text_config)
    for phase, workload in list(workloads.items()):
        def run(workload=workload, phase=phase):
            result = workload.run()
            length = inputs["input_ids"].shape[1] - (phase == "prefill")
            state = model.last_state
            for index, layer in enumerate(model.model.layers):
                prefix = f"past_key_values.{index}."
                if layer.layer_type == "linear_attention":
                    result[prefix + "conv_states"] = layer.conv_history
                    result[prefix + "recurrent_states"] = state.recurrent[index][1:2].transpose(-1, -2)
                else:
                    for name, cache in (("key", state.k_cache[index]), ("value", state.v_cache[index])):
                        if layer.self_attn.kv_layout == "HND":
                            cache = cache.transpose(1, 2)
                        result[prefix + name] = cache.flatten(0, 1)[:length].transpose(0, 1)[None]
            return result
        workloads[phase] = Workload(run=run, prepare=workload.prepare)
    return workloads
