"""Florence-2 DaViT vision and BART with the checkpoint's three-beam decoding."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.softmax import LogSoftmax
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L3.yolov10_head import v10postprocess
from ..runner import Workload
from . import bart


class MLP(nn.Module):
    def __init__(self, width, ratio):
        super().__init__()
        self.fc1, self.fc2 = Linear(width, int(width * ratio)), Linear(int(width * ratio), width)
        self.act = GELU()

    def forward(self, hidden):
        return self.fc2(self.act(self.fc1(hidden)))


class ConvEmbed(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        source = config.embed_dim[index - 1] if index else config.in_channels
        target = config.embed_dim[index]
        self.pre = config.patch_prenorm[index]
        self.conv = Conv2d(source, target, config.patch_size[index], config.patch_stride[index], config.patch_padding[index])
        self.norm = LayerNorm(source if self.pre else target, eps=1e-5, promote_fp32=False)

    def forward(self, hidden):
        if self.pre:
            hidden = self.norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        hidden = self.conv(hidden)
        if not self.pre:
            hidden = self.norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return hidden


class VisionAttention(nn.Module):
    def __init__(self, config, index, channel):
        super().__init__()
        width = config.embed_dim[index]
        self.heads = config.num_groups[index] if channel else config.num_heads[index]
        self.window, self.channel = config.window_size, channel
        self.qkv, self.proj = Linear(width, 3 * width, bias=config.qkv_bias), Linear(width, width)
        self.attention, self.pad = DenseAttention(backend="sdpa"), Pad()

    def forward(self, hidden):
        if self.channel:
            batch, length, width = hidden.shape
            qkv = self.qkv(hidden).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 4, 3, 1)
            result = self.attention(*qkv.unbind(0), softmax_scale=length ** -0.5)
            return self.proj(result.permute(0, 3, 2, 1).reshape(batch, length, width))
        batch, height, width, channels = hidden.shape
        window = self.window
        ph, pw = (-height) % window, (-width) % window
        hidden = self.pad(hidden, (0, 0, 0, pw, 0, ph))
        rows, cols = (height + ph) // window, (width + pw) // window
        hidden = hidden.view(batch, rows, window, cols, window, channels).permute(0, 1, 3, 2, 4, 5)
        hidden = hidden.reshape(-1, window * window, channels)
        qkv = self.qkv(hidden).view(-1, window * window, 3, self.heads, channels // self.heads).permute(2, 0, 1, 3, 4)
        hidden = self.proj(self.attention(*qkv.unbind(0)).reshape(-1, window * window, channels))
        hidden = hidden.view(batch, rows, cols, window, window, channels).permute(0, 1, 3, 2, 4, 5)
        return hidden.reshape(batch, height + ph, width + pw, channels)[:, :height, :width].reshape(batch, height * width, channels)


class VisionBlock(nn.Module):
    def __init__(self, config, index, channel):
        super().__init__()
        width = config.embed_dim[index]
        self.channel = channel
        self.conv1, self.conv2 = (Conv2d(width, width, 3, padding=1, groups=width) for _ in range(2))
        self.norm1, self.norm2 = (LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(2))
        setattr(self, "channel_attn" if channel else "window_attn", VisionAttention(config, index, channel))
        self.ffn = MLP(width, config.mlp_ratio)

    def forward(self, hidden):
        batch, channels, height, width = hidden.shape
        hidden = (hidden + self.conv1(hidden)).flatten(2).transpose(1, 2)
        normalized = self.norm1(hidden)
        mixed = self.channel_attn(normalized) if self.channel else self.window_attn(normalized.view(batch, height, width, channels))
        hidden = (hidden + mixed).transpose(1, 2).reshape(batch, channels, height, width)
        hidden = (hidden + self.conv2(hidden)).flatten(2).transpose(1, 2)
        return (hidden + self.ffn(self.norm2(hidden))).transpose(1, 2).reshape(batch, channels, height, width)


class DualBlock(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.spatial_block, self.channel_block = VisionBlock(config, index, False), VisionBlock(config, index, True)

    def forward(self, hidden):
        return self.channel_block(self.spatial_block(hidden))


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convs = nn.ModuleList(ConvEmbed(config, i) for i in range(len(config.depths)))
        self.blocks = nn.ModuleList(nn.ModuleList(DualBlock(config, i) for _ in range(depth))
                                    for i, depth in enumerate(config.depths))

    def forward(self, hidden):
        for conv, blocks in zip(self.convs, self.blocks):
            hidden = conv(hidden)
            for block in blocks:
                hidden = block(hidden)
        return hidden


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.embed_dim[-1]
        self.image_projection = Linear(width, config.projection_dim, bias=False)
        self.image_proj_norm = LayerNorm(config.projection_dim, eps=1e-5, promote_fp32=False)
        self.image_position_embed = nn.Module()
        self.image_position_embed.row_embeddings = Embedding(config.max_position_embeddings, width // 2)
        self.image_position_embed.column_embeddings = Embedding(config.max_position_embeddings, width - width // 2)
        self.visual_temporal_embed = nn.Module()
        self.visual_temporal_embed.register_buffer("pos_idx_to_embed", torch.empty(config.max_temporal_embeddings, width))
        self.reduce = SegmentCSR()

    def forward(self, hidden):
        batch, channels, height, width = hidden.shape
        rows = self.image_position_embed.row_embeddings(torch.arange(height, device=hidden.device))
        cols = self.image_position_embed.column_embeddings(torch.arange(width, device=hidden.device))
        positions = torch.cat((cols[None].expand(height, -1, -1), rows[:, None].expand(-1, width, -1)), -1)
        hidden = (hidden + positions.permute(2, 0, 1)[None]).flatten(2).transpose(1, 2)
        hidden = hidden + self.visual_temporal_embed.pos_idx_to_embed[:1]
        pooled = self.reduce(hidden.transpose(0, 1).contiguous().float(),
                             torch.tensor([0, height * width], device=hidden.device), "mean").transpose(0, 1).to(hidden.dtype)
        return self.image_proj_norm(self.image_projection(torch.cat((pooled, hidden), 1)))


class Florence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower, self.multi_modal_projector = Vision(config.vision_config), Projector(config.vision_config)
        self.language_model = bart.BartForConditionalGeneration(config.text_config)
        self.language_model._non_persistent_buffers_set.add("final_logits_bias")
        self.log_softmax = LogSoftmax(-1)

    def topk(self, scores, count):
        # The unchanged detection selector carries candidate positions as its
        # four box fields. Its top-k accepts the forced-token -infinity scores.
        batch, length = scores.shape
        indices = torch.arange(length, device=scores.device, dtype=torch.float32)
        metadata = indices[None, :, None].expand(batch, -1, 4)
        boxes, values, _ = v10postprocess(torch.cat((metadata, scores.float()[..., None]), -1), count, nc=1)
        return values, boxes[..., 0].long()

    def encode(self, inputs):
        text = self.language_model
        hidden = text.shared(inputs["input_ids"])
        hidden[inputs["input_ids"] == self.config.image_token_id] = self.multi_modal_projector(self.vision_tower(inputs["pixel_values"])).reshape(-1, hidden.shape[-1])
        positions = torch.arange(hidden.shape[1], device=hidden.device) + 2
        hidden = text.encoder.layernorm_embedding(hidden * text.encoder.embed_scale + text.encoder.embed_positions(positions))
        return text.encoder.layers(hidden)

    def decode(self, ids, memory, cache, position):
        decoder = self.language_model.decoder
        hidden = decoder.layernorm_embedding(decoder.embed_tokens(ids) * decoder.embed_scale + decoder.embed_positions(position + 2))
        states = []
        for index, layer in enumerate(decoder.layers):
            attn = layer.attention.self
            batch, length = hidden.shape[:2]
            q, k, v = (x.view(batch, length, attn.num_attention_heads, attn.attention_head_size)
                       for x in attn._project_qkv(hidden))
            old = None if cache is None else cache[index]
            if old is not None:
                k, v = torch.cat((old[0], k), 1), torch.cat((old[1], v), 1)
            output = attn.attn(q, k, v, causal=old is None and length > 1)
            hidden = layer.attention.output(output.reshape(batch, length, -1), hidden)
            cross = layer.cross_attention
            cq = cross.q_proj(hidden).view(batch, length, cross.heads, cross.head_dim)
            if old is None:
                ck, cv = (getattr(cross, name)(memory).view(batch, memory.shape[1], cross.heads, cross.head_dim)
                          for name in ("k_proj", "v_proj"))
            else:
                ck, cv = old[2:]
            output = cross.attention(cq, ck, cv)
            hidden = cross.norm(hidden + cross.out_proj(output.reshape(batch, length, -1)))
            hidden = layer.output(layer.intermediate(hidden), hidden)
            states.append((k, v, ck, cv))
        return self.language_model.lm_head(hidden[:, -1]), states

    def generate(self, inputs, steps, generation):
        if inputs["input_ids"].shape[0] != 1:
            raise ValueError("Florence development beam search uses one image/prompt")
        beams, eos = generation["num_beams"], generation["eos_token_id"]
        memory = self.encode(inputs).expand(beams, -1, -1)
        ids = torch.full((beams, 1), generation["decoder_start_token_id"], device=memory.device, dtype=torch.long)
        scores = torch.full((beams,), -1e9, device=memory.device); scores[0] = 0
        cache, finished, finished_scores = None, [], []
        outputs = {}
        for step in range(steps):
            logits, cache = self.decode(ids[:, -1:], memory, cache, torch.tensor([step], device=memory.device))
            outputs[f"logits.{step}"] = logits.float()
            logits = self.log_softmax(logits.float())
            ngram = generation["no_repeat_ngram_size"]
            for row, tokens in enumerate(ids.tolist()):
                if len(tokens) >= ngram - 1:
                    prefix = tokens[-(ngram - 1):]
                    banned = [tokens[i + ngram - 1] for i in range(len(tokens) - ngram + 1)
                              if tokens[i:i + ngram - 1] == prefix]
                    logits[row, banned] = -float("inf")
            forced = generation["forced_bos_token_id"] if step == 0 else generation["forced_eos_token_id"] if step + 1 == steps else None
            if forced is not None:
                logits.fill_(-float("inf")); logits[:, forced] = 0
            values, indices = self.topk((logits + scores[:, None]).reshape(1, -1), 2 * beams)
            for rank, flat in enumerate(indices[0].tolist()):
                parent, token = divmod(flat, logits.shape[-1])
                if token == eos:
                    if rank < beams:
                        finished.append(torch.cat((ids[parent], ids.new_tensor([token]))))
                        finished_scores.append(values[0, rank] / (step + 1))
            # HF selects running beams again after marking completed candidates,
            # including the last step, before returning the reordered cache.
            token_ids = indices % logits.shape[-1]
            stopped = (token_ids == eos) | (step + 1 == steps)
            running_scores = values + stopped.float() * -1e9
            scores, running = self.topk(running_scores, beams)
            flat = indices[0, running[0]]
            parents, tokens = flat // logits.shape[-1], flat % logits.shape[-1]
            ids = torch.cat((ids[parents], tokens[:, None]), 1)
            scores = scores[0]
            cache = [tuple(value[parents] for value in layer) for layer in cache]
            if len(finished) >= beams or step + 1 == steps:
                break
        if not finished:
            raise RuntimeError("Florence's forced terminal EOS should finalize a beam")
        _, best = self.topk(torch.stack(finished_scores)[None], 1)
        outputs["sequences"] = finished[int(best[0, 0])][None]
        for index, layer in enumerate(cache):
            for name, value in zip(("self.key", "self.value", "cross.key", "cross.value"), layer):
                outputs[f"past_key_values.{index}.{name}"] = value.transpose(1, 2)
        return outputs


def build_from_config(config, device, dtype):
    model = Florence(config).to(device=device, dtype=dtype).eval()
    # Preserve HF's ordinary cuDNN selection despite candidate dependencies
    # changing PyTorch's process-wide SDPA backend preference.
    for operation in model.modules():
        if isinstance(operation, DenseAttention):
            operation.use_cudnn_kernel = True
    return model


def load_state_dict_into(model, weights, config):
    language = {name.replace("model.language_model.", "model."):value for name,value in weights.items()
                if name.startswith("model.language_model.")}
    language["lm_head.weight"] = weights["lm_head.weight"]
    language["final_logits_bias"] = model.language_model.final_logits_bias
    bart.load_state_dict_into(model.language_model, language, config.text_config)
    consumed = {name for name in weights if name.startswith("model.language_model.")} | {"lm_head.weight"}
    for prefix in ("vision_tower", "multi_modal_projector"):
        module = getattr(model, prefix); mapped = {}
        for name, target in module.state_dict().items():
            source = "model." + prefix + "." + name.replace(".emb.weight", ".weight")
            if weights[source].shape != target.shape:
                raise ValueError(f"Florence state shape mismatch: {source}")
            mapped[name] = weights[source]; consumed.add(source)
        module.load_state_dict(mapped, strict=True)
    if consumed != set(weights):
        raise ValueError(f"Florence unmapped weights: {sorted(set(weights) - consumed)}")


def make_workloads(model, inputs, config, case):
    generation = case["reference"]["generation_config"]
    if generation["num_beams"] != 3 or not generation["early_stopping"]:
        raise ValueError("Preserve Florence's native three beams and early stopping")
    return {"generate": Workload(run=lambda: model.generate(inputs, case["generation_kwargs"]["max_new_tokens"], generation))}
