"""GPT-J with adjacent-pair partial RoPE and a shared normalized parallel block."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import config_values
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .bloom import AlibiAttention
from .gpt_neox import DecoderLM
from .llama import make_workloads as decoder_workloads
from .phi import BiasedHeadProduct
from .stablelm import PartialRotary


class Float32ScoreAttention(AlibiAttention):
    """Existing BMM/softmax composition retaining the reference's dtype boundaries."""

    def _sdpa_one(self, query, key, value, key_offset=0):
        keys = torch.arange(key.shape[0], device=query.device)
        queries = key_offset + torch.arange(query.shape[0], device=query.device)
        excluded = keys[None, :] > queries[:, None]
        if self.sliding_window is not None:
            excluded |= keys[None, :] <= queries[:, None] - self.sliding_window
        mask = torch.zeros(query.shape[0], key.shape[0], device=query.device, dtype=torch.float32)
        mask = mask.masked_fill(excluded, torch.finfo(torch.float32).min)
        scores = self.qk(query.transpose(0, 1).float(), key.permute(1, 2, 0).float())
        probabilities = self.softmax((scores + mask) * self.scale).to(value.dtype)
        return self.pv(probabilities, value.transpose(0, 1)).transpose(0, 1)

    def forward(self, query, key, value):
        return super().forward(query, key, value).to(value.dtype)


class GPTJRotary(PartialRotary):
    """Reuse the existing interleaved operation with separately stored products."""

    def _rotate(self, positions, query, key):
        return self.rotary.forward_native_interleaved(
            positions, query, key, self.rotary_dim,
            self.rotary.cos_sin_cache.to(query.dtype),
        )


class SeparateQKV(nn.Module):
    """Keep packed weight storage, but execute HF's three linear projections."""

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(3 * width, width))
        self.matmul = Matmul()

    def forward(self, hidden):
        return torch.cat([self.matmul(hidden, weight) for weight in self.weight.chunk(3)], dim=-1)


class GPTJLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        width, heads = config.hidden_size, config.num_attention_heads
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.self_attn = LlamaAttention(width, heads, heads, width // heads, rotary_emb=rotary)
        self.self_attn.attn = Float32ScoreAttention(heads, width // heads)
        self.mlp = VitEncoderMlp(width, config.intermediate_size)
        self.mlp.act = NewGELUActivation()

    def forward(self, positions, hidden):
        normalized = self.input_layernorm(hidden)
        return self.self_attn(positions, normalized) + self.mlp(normalized) + hidden


def decoder_config(config):
    adapted = config_values(dict(config))
    adapted.update(hidden_size=config.n_embd, num_attention_heads=config.n_head,
                   num_hidden_layers=config.n_layer, intermediate_size=config.n_inner or 4 * config.n_embd,
                   layer_norm_eps=config.layer_norm_epsilon, max_position_embeddings=config.n_positions)
    return adapted


def build_from_config(config, device, dtype, *, separate_qkv=True):
    if (_tp_size() != 1 or config.activation_function != "gelu_new"
            or config.tie_word_embeddings or not config.use_cache):
        raise ValueError("The selected GPT-J checkpoint uses cached untied decoding with gelu_new")
    adapted = decoder_config(config)
    head_dim = adapted.hidden_size // adapted.num_attention_heads
    if not 0 < config.rotary_dim < head_dim:
        raise ValueError("Preserve GPT-J's partial rotary slice and unrotated tail")
    with torch.device("cpu"):
        rotary = GPTJRotary(head_dim, config.rotary_dim, config.n_positions, 10000.0)
    rotary.rotary.is_neox_style = False
    model = DecoderLM(adapted, [GPTJLayer(adapted, rotary) for _ in range(config.n_layer)])
    if separate_qkv:
        for layer in model.model.layers:
            # The larger packed projection selects different GEMV arithmetic
            # during single-token decoding; separate existing ops match HF.
            layer.self_attn.qkv_proj = SeparateQKV(adapted.hidden_size)
    model.lm_head.linear_op = BiasedHeadProduct(adapted.hidden_size, adapted.vocab_size)
    model.lm_head.linear_op.weight = model.lm_head.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {
        "model.embed_tokens.embedding_op.emb.weight": remaining.pop("transformer.wte.weight"),
        "lm_head.embedding_op.emb.weight": remaining["lm_head.weight"],
        "lm_head.linear_op.weight": remaining.pop("lm_head.weight"),
        "lm_head.linear_op.bias": remaining.pop("lm_head.bias"),
    }
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("transformer.ln_f." + field)
    for index in range(config.n_layer):
        src, dst = f"transformer.h.{index}.", f"model.layers.{index}."
        mapped[dst + "self_attn.qkv_proj.weight"] = torch.cat([
            remaining.pop(src + f"attn.{part}_proj.weight") for part in ("q", "k", "v")])
        mapped[dst + "self_attn.o_proj.weight"] = remaining.pop(src + "attn.out_proj.weight")
        for field in ("weight", "bias"):
            for target, source in (("input_layernorm", "ln_1"), ("mlp.fc1", "mlp.fc_in"), ("mlp.fc2", "mlp.fc_out")):
                mapped[dst + target + "." + field] = remaining.pop(src + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped GPT-J state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return decoder_workloads(model, inputs, model.config, case=case)
