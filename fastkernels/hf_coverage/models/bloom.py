"""BLOOM causal decoding with existing masked attention and its native GELU."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaModel

from ..patches.gelu_fast import FastGELU
from .llama import make_workloads as llama_workloads
from .olmo import ResidualLayerNorm


class AlibiAttention(Attention):
    """Keep HF's score rounding using existing matrix multiplies and softmax."""

    def __init__(self, num_heads, head_dim):
        super().__init__(num_heads, head_dim, head_dim ** -0.5, prefer_triton=True)
        self.qk = BatchMatMul()
        self.softmax = Softmax(dim=-1)
        self.pv = BatchMatMul()

    def _sdpa_one(self, query, key, value, key_offset=0):
        # Construct the same fixed slopes and dtype-rounded positional bias.
        closest = 2 ** math.floor(math.log2(self.num_heads))
        base = torch.tensor(2 ** (-(2 ** -(math.log2(closest) - 3))),
                            device=query.device, dtype=torch.float32)
        slopes = torch.pow(base, torch.arange(1, closest + 1, device=query.device, dtype=torch.int32))
        if closest != self.num_heads:
            extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest) - 3))),
                                      device=query.device, dtype=torch.float32)
            powers = torch.arange(1, 2 * (self.num_heads - closest) + 1, 2,
                                  device=query.device, dtype=torch.int32)
            slopes = torch.cat((slopes, torch.pow(extra_base, powers)))
        key_positions = torch.arange(key.shape[0], device=query.device)
        query_positions = key_offset + torch.arange(query.shape[0], device=query.device)
        bias = (slopes[:, None] * key_positions).to(query.dtype)
        mask = torch.zeros(query.shape[0], key.shape[0], device=query.device, dtype=query.dtype).masked_fill(
            key_positions[None, None, :] > query_positions[None, :, None], torch.finfo(query.dtype).min,
        )
        # HF baddbmm accumulates QK, scaling and ALiBi before storing scores in
        # the model dtype. A fused SDPA call skips that observed round boundary.
        scores = self.qk(query.transpose(0, 1).float(), key.permute(1, 2, 0).float())
        scores = (scores * self.scale + bias[:, None, :].float()).to(query.dtype)
        probabilities = self.softmax((scores + mask).float()).to(query.dtype)
        return self.pv(probabilities, value.transpose(0, 1)).transpose(0, 1)

    def forward(self, query, key, value):
        ctx = get_context()
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        self.store_kvcache(key, value, self.k_cache, self.v_cache, self._group_slot_mapping(ctx))
        return self._forward_pure_torch(query, key, value, self.k_cache, self.v_cache, ctx).reshape(
            query.shape[0], -1,
        )


class BloomBackbone(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = None
        self.embedding_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False)
        for layer in self.layers:
            layer.self_attn.rotary_emb = None
            layer.self_attn.nope = True
            layer.self_attn.attn = AlibiAttention(config.num_attention_heads, config.head_dim)
            layer.self_attn.o_proj = Linear(config.hidden_size, config.hidden_size)
            layer.input_layernorm = ResidualLayerNorm(config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False)
            layer.post_attention_layernorm = ResidualLayerNorm(config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False)
            layer.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size)
            layer.mlp.act = FastGELU()
        self.norm = ResidualLayerNorm(config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False)

    def forward(self, input_ids, positions):
        embeddings = self.embedding_norm(self.embed_tokens(input_ids))
        return super().forward(input_ids, positions, inputs_embeds=embeddings)


class BloomForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BloomBackbone(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight


def build_from_config(config, device, dtype):
    if (_tp_size() != 1 or config.apply_residual_connection_post_layernorm or config.slow_but_exact
            or not config.use_cache or not config.tie_word_embeddings
            or config.output_attentions or config.output_hidden_states):
        raise ValueError("This case preserves BLOOM's default cached decoder and residual order")
    if config.hidden_size % config.n_head:
        raise ValueError("BLOOM hidden width must divide into whole attention heads")
    adapted = LlamaConfig(hidden_size=config.hidden_size, intermediate_size=4 * config.hidden_size,
                          num_hidden_layers=config.n_layer, num_attention_heads=config.n_head,
                          num_key_value_heads=config.n_head, head_dim=config.hidden_size // config.n_head,
                          vocab_size=config.vocab_size, rms_norm_eps=config.layer_norm_epsilon,
                          qkv_bias=True, dtype=dtype)
    return BloomForCausalLM(adapted).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {}

    def copy(destination, source):
        mapped[destination] = remaining.pop(source)

    copy("model.embed_tokens.embedding_op.emb.weight", "transformer.word_embeddings.weight")
    copy("lm_head.embedding_op.emb.weight", "lm_head.weight")
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("BLOOM's tied source embedding and head disagree")
    for field in ("weight", "bias"):
        copy("model.embedding_norm." + field, "transformer.word_embeddings_layernorm." + field)
        copy("model.norm." + field, "transformer.ln_f." + field)
    for index in range(config.n_layer):
        source, target = f"transformer.h.{index}.", f"model.layers.{index}."
        for field in ("weight", "bias"):
            packed = remaining.pop(source + "self_attention.query_key_value." + field)
            shape = (config.n_head, 3, config.hidden_size // config.n_head, *packed.shape[1:])
            mapped[target + "self_attn.qkv_proj." + field] = packed.view(shape).transpose(0, 1).reshape(packed.shape)
            for destination, name in (("input_layernorm", "input_layernorm"),
                                      ("post_attention_layernorm", "post_attention_layernorm"),
                                      ("self_attn.o_proj", "self_attention.dense"),
                                      ("mlp.fc1", "mlp.dense_h_to_4h"),
                                      ("mlp.fc2", "mlp.dense_4h_to_h")):
                copy(target + destination + "." + field, source + name + "." + field)
    if remaining:
        raise KeyError(f"Unmapped BLOOM state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    # BLOOM has no learned-position table limiting context; the shared helper
    # needs only this run's capacity and vocabulary size.
    workload_config = SimpleNamespace(vocab_size=config.vocab_size,
                                      max_position_embeddings=inputs["input_ids"].shape[1])
    return llama_workloads(model, inputs, workload_config, case=case)
