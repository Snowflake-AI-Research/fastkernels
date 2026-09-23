"""Differential attention as existing BMM, softmax and weightless RMSNorm operations."""

import math
import torch

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM

from .bloom import AlibiAttention
from .llama import make_workloads


class DifferentialAttention(AlibiAttention):
    def __init__(self, config, index):
        super().__init__(config.num_attention_heads, config.head_dim)
        self.num_kv_heads = config.num_key_value_heads
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * index)
        self.register_buffer("coefficient", torch.tensor(self.lambda_init), persistent=False)
        self.groupnorm = RMSNorm(2 * config.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)

    def _sdpa_one(self, query, key, value, key_offset=0):
        key, value = self._repeat_kv_for_heads(key), self._repeat_kv_for_heads(value)
        # Both attention halves read concatenated value halves, as in pinned HF.
        value = torch.cat(value.chunk(2, dim=1), dim=-1).repeat(1, 2, 1)
        keys = torch.arange(key.shape[0], device=query.device)
        queries = key_offset + torch.arange(query.shape[0], device=query.device)
        mask = torch.zeros(query.shape[0], key.shape[0], device=query.device, dtype=query.dtype)
        mask = mask.masked_fill(keys[None, :] > queries[:, None], torch.finfo(query.dtype).min)
        scores = self.qk(query.transpose(0, 1), key.permute(1, 2, 0)) * self.scale
        probabilities = self.softmax((scores + mask).float()).to(query.dtype)
        first, second = self.pv(probabilities, value.transpose(0, 1)).chunk(2, dim=0)
        output = self.groupnorm(first - self.coefficient * second) * (1.0 - self.lambda_init)
        return output.transpose(0, 1).reshape(query.shape)


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or config.hidden_act != "silu" or config.attention_bias or config.mlp_bias
            or not config.use_cache or not config.tie_word_embeddings or rope["rope_type"] != "llama3"
            or config.num_attention_heads % 2 or config.num_key_value_heads % 2):
        raise ValueError("The documented DiffLlama checkpoint uses even differential head groups, tied embeddings and Llama3 RoPE")
    adapted = LlamaConfig(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                          num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
                          num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim,
                          vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
                          rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
                          rope_scaling_factor=rope["factor"], rope_low_freq_factor=rope["low_freq_factor"],
                          rope_high_freq_factor=rope["high_freq_factor"],
                          rope_original_max_position_embeddings=rope["original_max_position_embeddings"], dtype=dtype)
    model = LlamaForCausalLM(adapted)
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    for index, layer in enumerate(model.model.layers):
        layer.self_attn.attn = DifferentialAttention(config, index)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = dict(state_dict)
    mapped["model.embed_tokens.embedding_op.emb.weight"] = mapped.pop("model.embed_tokens.weight")
    mapped["lm_head.embedding_op.emb.weight"] = mapped.pop("lm_head.weight")
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("DiffLlama's tied head and embedding disagree")
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}."
        # Frozen learned vectors only affect inference through this scalar. Preserve
        # native multiplication and FP32 reduction, then the query-dtype rounding.
        products = []
        for number in (1, 2):
            q = mapped.pop(prefix + f"self_attn.lambda_q{number}")
            k = mapped.pop(prefix + f"self_attn.lambda_k{number}")
            products.append(torch.exp(torch.sum(q * k, dtype=torch.float32)).to(layer.self_attn.attn.coefficient.dtype))
        layer.self_attn.attn.coefficient.copy_(products[0] - products[1] + layer.self_attn.attn.lambda_init)
        mapped[prefix + "self_attn.qkv_proj.weight"] = torch.cat([
            mapped.pop(prefix + f"self_attn.{part}_proj.weight") for part in ("q", "k", "v")])
        mapped[prefix + "mlp.gate_up_proj.weight"] = torch.cat([
            mapped.pop(prefix + f"mlp.{part}_proj.weight") for part in ("gate", "up")])
    model.load_state_dict(mapped, strict=True)
