"""Bark's sampled semantic/coarse/fine generation and EnCodec waveform decoder."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.engine import LlamaEngine
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
from fastkernels.tasks.baseline.L1.softmax import Softmax, LogSoftmax
from fastkernels.tasks.baseline.L1.top_k_per_row import TopKPerRow
from .encodec import Encodec, load_state_dict_into as load_codec


class Attention(nn.Module):
    def __init__(self, config, causal):
        super().__init__()
        self.heads, self.width, self.causal = config.num_heads, config.hidden_size, causal
        self.att_proj = Linear(self.width, 3 * self.width, bias=config.bias)
        self.out_proj = Linear(self.width, self.width, bias=config.bias)
        self.bmm, self.softmax = BatchMatMul(), Softmax(dim=-1)
        self.key = self.value = None

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        dim = self.width // self.heads
        query, key, value = [part.reshape(batch, length, self.heads, dim).transpose(1, 2)
                             for part in self.att_proj(hidden).chunk(3, dim=-1)]
        if self.causal:
            if self.key is not None:
                key, value = torch.cat((self.key, key), 2), torch.cat((self.value, value), 2)
            self.key, self.value = key, value
        scores = self.bmm(query.reshape(-1, length, dim), key.reshape(-1, key.shape[2], dim).transpose(1, 2))
        scores = scores * (dim ** -0.5)
        if self.causal:
            positions = torch.arange(length, device=hidden.device) + key.shape[2] - length
            scores = scores.masked_fill(torch.arange(key.shape[2], device=hidden.device)[None, :] > positions[:, None],
                                        torch.finfo(scores.dtype).min)
        output = self.bmm(self.softmax(scores), value.reshape(-1, value.shape[2], dim))
        return self.out_proj(output.reshape(batch, self.heads, length, dim).transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.out_proj = Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.gelu = GELU()

    def forward(self, hidden):
        return self.out_proj(self.gelu(self.in_proj(hidden)))


class Block(nn.Module):
    def __init__(self, config, causal):
        super().__init__()
        bias = config.bias if causal else True
        self.layernorm_1 = LayerNorm(config.hidden_size, create_offset=bias, promote_fp32=False)
        self.layernorm_2 = LayerNorm(config.hidden_size, create_offset=bias, promote_fp32=False)
        self.attn, self.mlp = Attention(config, causal), MLP(config)

    def forward(self, hidden):
        hidden = hidden + self.attn(self.layernorm_1(hidden))
        return hidden + self.mlp(self.layernorm_2(hidden))


class Transformer(nn.Module):
    def __init__(self, config, fine=False):
        super().__init__()
        self.fine, self.config = fine, config
        if fine:
            self.input_embeds_layers = nn.ModuleList(Embedding(config.input_vocab_size, config.hidden_size)
                                                    for _ in range(config.n_codes_total))
            self.lm_heads = nn.ModuleList(Linear(config.hidden_size, config.output_vocab_size, bias=False)
                                          for _ in range(config.n_codes_given, config.n_codes_total))
            for index, head in enumerate(self.lm_heads):
                head.weight = self.input_embeds_layers[index + config.n_codes_given].emb.weight
            self.sum = MoeSum()
        else:
            self.input_embeds_layer = Embedding(config.input_vocab_size, config.hidden_size)
            self.lm_head = Linear(config.hidden_size, config.output_vocab_size, bias=False)
        self.position_embeds_layer = Embedding(config.block_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config, not fine) for _ in range(config.num_layers))
        self.layernorm_final = LayerNorm(config.hidden_size, create_offset=True if fine else config.bias,
                                         promote_fp32=False)
        self.seen = 0

    def reset(self):
        self.seen = 0
        for layer in self.layers:
            layer.attn.key = layer.attn.value = None

    def forward(self, ids=None, embeddings=None, codebook=None):
        if self.fine:
            tables = [table(ids[..., index]) for index, table in enumerate(self.input_embeds_layers[:codebook + 1])]
            packed = torch.stack(tables, dim=-1)
            hidden = self.sum(packed.reshape(-1, 1), codebook + 1).reshape(tables[0].shape)
        else:
            hidden = self.input_embeds_layer(ids) if embeddings is None else embeddings
        positions = torch.arange(hidden.shape[1], device=hidden.device) + (0 if self.fine else self.seen)
        hidden = hidden + self.position_embeds_layer(positions)[None]
        if not self.fine:
            self.seen += hidden.shape[1]
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.layernorm_final(hidden)
        return self.lm_heads[codebook - self.config.n_codes_given](hidden) if self.fine else self.lm_head(hidden)


class Sampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk, self.compare = TopKPerRow(), CodecTop1()
        self.minimum = MaxPool2d((1, 50))
        self.log_softmax = LogSoftmax(dim=-1)

    def forward(self, logits, temperature, topk=False):
        if topk:
            # HF keeps all ties at the fiftieth score, not exactly fifty indices.
            logits = logits.float() * (1.0 / temperature)
            rows = logits.shape[0]
            indices = self.topk.forward_prefill(logits, torch.zeros(rows, dtype=torch.int32, device=logits.device),
                                                torch.full((rows,), logits.shape[-1], dtype=torch.int32, device=logits.device), 50)
            selected = logits.gather(1, indices.long())
            threshold = -self.minimum(-selected[:, None, None, :]).reshape(rows, 1)
            less = self.compare(torch.stack((logits, threshold.expand_as(logits)), dim=-1)).bool()
            logits = self.log_softmax(logits.masked_fill(less, -float("inf")))
            temperature = 1.0
        # Unchanged infrastructure sampler; its host transfer remains timed.
        sampled = LlamaEngine._sample(None, logits, SimpleNamespace(temperature=temperature, top_p=1.0))
        return torch.tensor(sampled, dtype=torch.long, device=logits.device)


class Bark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.semantic = Transformer(config.semantic_config)
        self.coarse_acoustics = Transformer(config.coarse_acoustics_config)
        self.fine_acoustics = Transformer(config.fine_acoustics_config, fine=True)
        self.codec_model, self.sample = Encodec(config.codec_config), Sampling()

    def generate(self, input_ids, attention_mask, generation, semantic_steps):
        semantic, coarse, fine = (generation[name] for name in
                                  ("semantic_config", "coarse_acoustics_config", "fine_acoustics_config"))
        if input_ids.shape[0] != 1:
            raise ValueError("This workload evaluates native unprompted generation for one text input")
        self.semantic.reset()
        ids = (input_ids + semantic["text_encoding_offset"]).masked_fill(~attention_mask.bool(), semantic["text_pad_token"])
        history = torch.full_like(ids, semantic["semantic_pad_token"])
        infer = ids.new_full((1, 1), semantic["semantic_infer_token"])
        embed = torch.cat((self.semantic.input_embeds_layer(ids) + self.semantic.input_embeds_layer(history),
                           self.semantic.input_embeds_layer(infer)), dim=1)
        tokens = []
        for step in range(semantic_steps):
            logits = self.semantic(embeddings=embed)[:, -1].float() if step == 0 else self.semantic(ids=token[:, None])[:, -1].float()
            logits[:, semantic["semantic_pad_token"] + 1:] = -float("inf")
            token = self.sample(logits, semantic["temperature"], topk=True)
            tokens.append(token)
            if token.item() == semantic["eos_token_id"]:
                break
        semantic_ids = torch.stack(tokens, 1)
        semantic_ids = semantic_ids.masked_fill(semantic_ids == semantic["semantic_pad_token"], coarse["coarse_semantic_pad_token"])
        # Generated integer token counts determine frame allocation, not activations.
        length = len(tokens) - int(tokens[-1].item() == semantic["eos_token_id"])
        ratio = coarse["coarse_rate_hz"] / semantic["semantic_rate_hz"] * coarse["n_coarse_codebooks"]
        total = int(math.floor(length * ratio / coarse["n_coarse_codebooks"]) * coarse["n_coarse_codebooks"])
        generated = ids.new_empty((1, 0))
        for window in range(0, total, coarse["sliding_window_len"]):
            start = max(0, round(window / ratio) - math.floor(coarse["max_coarse_history"] / ratio))
            prefix = semantic_ids[:, start:start + coarse["max_coarse_input_length"]]
            prefix = torch.cat((prefix, ids.new_full((1, coarse["max_coarse_input_length"] - prefix.shape[1]), coarse["coarse_semantic_pad_token"]),
                                ids.new_full((1, 1), coarse["coarse_infer_token"]), generated[:, -coarse["max_coarse_history"]:]), dim=1)
            self.coarse_acoustics.reset()
            for step in range(min(coarse["sliding_window_len"], total - window)):
                logits = self.coarse_acoustics(ids=prefix if step == 0 else token[:, None])[:, -1].float()
                low = semantic["semantic_vocab_size"] + ((window + step) % 2) * generation["codebook_size"]
                logits[:, :low] = -float("inf")
                # Native second-codebook selection keeps the allocated tail
                # vocabulary; fine generation subsequently applies modulo.
                if (window + step) % 2 == 0:
                    logits[:, low + generation["codebook_size"]:] = -float("inf")
                token = self.sample(logits, coarse["temperature"], topk=True)
                generated = torch.cat((generated, token[:, None]), dim=1)
        codes = (generated.reshape(1, -1, coarse["n_coarse_codebooks"]) - semantic["semantic_vocab_size"]) % generation["codebook_size"]
        frames = codes.shape[1]
        width, stride = fine["max_fine_input_length"], fine["max_fine_history_length"]
        fine_input = ids.new_full((1, max(frames, width), fine["n_fine_codebooks"]), generation["codebook_size"])
        fine_input[:, :frames, :coarse["n_coarse_codebooks"]] = codes
        for outer in range(max(0, math.ceil((frames - width) / stride)) + 1):
            start = min(outer * stride, fine_input.shape[1] - width)
            fill = min(outer * stride, fine_input.shape[1] - stride) - start
            buffer = fine_input[:, start:start + width].clone()
            for channel in range(coarse["n_coarse_codebooks"], fine["n_fine_codebooks"]):
                logits = self.fine_acoustics(ids=buffer, codebook=channel)[:, fill:, :generation["codebook_size"]]
                buffer[:, fill:, channel] = self.sample(logits.reshape(-1, logits.shape[-1]), fine["temperature"]).reshape(1, -1)
            fine_input[:, start + fill:start + width] = buffer[:, fill:]
        codes = fine_input[:, :frames].transpose(1, 2)
        hidden = torch.tensor(0.0, device=codes.device)
        for channel, table in enumerate(self.codec_model.codebooks[:fine["n_fine_codebooks"]]):
            hidden = hidden + table.decode(codes[:, channel])
        return self.codec_model.decoder(hidden).squeeze(1)


def build_from_config(config, device, dtype):
    return Bark(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    for name in list(remaining):
        if name.endswith(".attn.bias"):
            mask = remaining.pop(name)
            expected = torch.ones(mask.shape[-2:], dtype=torch.bool, device=mask.device).tril()
            if not torch.equal(mask, expected.reshape(mask.shape)):
                raise ValueError(f"Bark causal mask differs from the native lower triangle: {name}")
    codec = {name.removeprefix("codec_model."): remaining.pop(name) for name in list(remaining) if name.startswith("codec_model.")}
    load_codec(model.codec_model, codec, config.codec_config)
    mapped = {name: remaining.pop(name.replace(".emb.weight", ".weight"))
              for name in model.state_dict() if not name.startswith("codec_model.")}
    if remaining:
        raise KeyError(f"Unmapped Bark state: {sorted(remaining)}")
    mapped.update({"codec_model." + name: value for name, value in model.codec_model.state_dict().items()})
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case):
    return {"generate": Workload(run=lambda: {"waveform": model.generate(
        inputs["input_ids"], inputs["attention_mask"], case["reference"]["generation_config"],
        case["generation_kwargs"]["semantic_max_new_tokens"])})}
