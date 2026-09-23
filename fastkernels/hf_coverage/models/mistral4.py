"""Mistral4 with explicit static FP8 boundaries and expanded MLA caches.

The existing static quantizer is reused before every quantized projection.
FP32 matrix multiplication consumes those FP8 values after conversion; fixed
weights are expanded once on load and scales applied after the dot product.
This is an executed composition whose
additional storage and runtime conversions must be included in reporting.
"""

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.flashinfer_mla_sparse import QuantFp8MLAQuery
from fastkernels.tasks.baseline.L1.grouped_topk import GroupedTopK
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YarnRotaryEmbedding
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class StaticFP8Linear(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.register_buffer("weight", torch.empty(output_width, input_width, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(1.0, dtype=torch.float32))
        self.quantize, self.matmul = QuantFp8MLAQuery(), Matmul()
        self.activation_scale = 1.0
        self.weight_scale = 1.0

    def forward(self, x):
        shape = x.shape[:-1]
        # HF divides in the input dtype before the FP8 cast. Quantizing with
        # a nonunit scale directly would omit this BF16 rounding boundary.
        scaled = x / self.input_scale
        quantized = self.quantize(scaled.reshape(-1, 1, x.shape[-1]), self.scale)
        output = self.matmul(quantized.float().reshape(-1, x.shape[-1]), self.weight)
        # Match HF's tensor-scale GEMM: scale the FP32 accumulator, then cast.
        output = output * self.activation_scale * self.weight_scale
        return output.to(x.dtype).view(*shape, -1)

    def load(self, weight, weight_scale, activation_scale):
        if weight.shape != self.weight.shape or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Static FP8 projection requires matching native FP8 weights")
        if weight_scale.numel() != 1 or activation_scale.numel() != 1:
            raise ValueError("Selected static FP8 projection uses per-tensor scales")
        self.weight = weight.float().to(self.weight.device)
        self.input_scale = activation_scale.float().reshape(()).to(self.weight.device)
        self.activation_scale = float(activation_scale)
        self.weight_scale = float(weight_scale)


class StaticMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = StaticFP8Linear(hidden, intermediate)
        self.up_proj = StaticFP8Linear(hidden, intermediate)
        self.down_proj = StaticFP8Linear(intermediate, hidden)
        self.activation = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.activation(torch.cat((self.gate_proj(x), self.up_proj(x)), -1)))


class StaticExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.register_buffer("router_weight", torch.empty(config.n_routed_experts, config.hidden_size,
                                                          dtype=torch.float32))
        self.gate_up = nn.ModuleList(StaticFP8Linear(config.hidden_size, 2 * config.moe_intermediate_size)
                                     for _ in range(config.n_routed_experts))
        self.down = nn.ModuleList(StaticFP8Linear(config.moe_intermediate_size, config.hidden_size)
                                  for _ in range(config.n_routed_experts))
        self.shared_experts = StaticMLP(config.hidden_size, config.n_shared_experts * config.moe_intermediate_size)
        self.router_matmul, self.activation = Matmul(), SiluAndMul()
        # With one selected group, its ranking has no effect. For the selected
        # finite softmax weights, HF's 1e-20 denominator addition rounds away.
        self.route = GroupedTopK(scoring_func="softmax", renormalize=True,
                                routed_scaling_factor=config.routed_scaling_factor)
        self.product, self.sum = ProductGate(), MoeSum()

    def forward(self, x):
        weights, indices = self.route(self.router_matmul(x, self.router_weight), None, 1, 1, self.top_k)
        outputs = torch.empty(x.shape[0], self.top_k, x.shape[-1], device=x.device, dtype=torch.float32)
        # This follows the model's existing expert dispatch, with all arithmetic
        # inside quantization, projection, activation, product and sum operations.
        for expert, (gate_up, down) in enumerate(zip(self.gate_up, self.down)):
            token, slot = torch.where(indices == expert)
            if token.numel() == 0:
                continue
            value = down(self.activation(gate_up(x[token])))
            routing = weights[token, slot, None].to(value.dtype).expand_as(value)
            outputs[token, slot] = self.product(torch.cat((value, routing), -1)).float()
        output = self.sum(outputs.reshape(-1, x.shape[-1]), self.top_k).to(x.dtype)
        return output + self.shared_experts(x)


class ExpandedStaticAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.nope, self.rope = config.qk_nope_head_dim, config.qk_rope_head_dim
        self.rank, self.value = config.kv_lora_rank, config.v_head_dim
        self.q_a_proj = StaticFP8Linear(config.hidden_size, config.q_lora_rank)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, 1e-6)
        self.q_b_proj = StaticFP8Linear(config.q_lora_rank, self.heads * config.qk_head_dim)
        self.kv_a_proj_with_mqa = StaticFP8Linear(config.hidden_size, self.rank + self.rope)
        self.kv_a_layernorm = RMSNorm(self.rank, 1e-6)
        self.kv_b_proj = StaticFP8Linear(self.rank, self.heads * (self.nope + self.value))
        self.o_proj = StaticFP8Linear(self.heads * self.value, config.hidden_size)
        self.attn = Attention(self.heads, config.qk_head_dim, config.qk_head_dim ** -0.5,
                              num_kv_heads=self.heads, prefer_triton=True)
        self.product = ProductGate()
        # These are fixed position values, not hidden-state computations.
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        rope = config.rope_parameters
        scale = 1 + rope["llama_4_scaling_beta"] * torch.log1p(torch.floor(positions / rope["original_max_position_embeddings"]))
        self.register_buffer("position_scale", scale, persistent=False)

    def forward(self, positions, hidden):
        count = hidden.shape[0]
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        q_plain, q_rot = q.view(count, self.heads, self.nope + self.rope).split((self.nope, self.rope), -1)
        latent, k_rot = self.kv_a_proj_with_mqa(hidden).split((self.rank, self.rope), -1)
        kv = self.kv_b_proj(self.kv_a_layernorm(latent.contiguous())).view(count, self.heads, self.nope + self.value)
        k_plain, value = kv.split((self.nope, self.value), -1)
        # HF changes interleaved input pairs into half-split output coordinates.
        q_rot = q_rot.reshape(count, self.heads, -1, 2).transpose(-1, -2).reshape(count, self.heads, self.rope)
        k_rot = k_rot.reshape(count, 1, -1, 2).transpose(-1, -2).reshape(count, 1, self.rope)
        # Reuse the library's native rotation to retain HF's intermediate rounding.
        q_rot, k_rot = RotaryEmbedding.forward_native(positions, q_rot, k_rot, self.rope,
                                                       self.rotary_emb.cos_sin_cache)
        query = torch.cat((q_plain, q_rot), -1)
        scaling = self.position_scale[positions].to(query.dtype).view(count, 1, 1).expand_as(query)
        query = self.product(torch.cat((query, scaling), -1))
        key = torch.cat((k_plain, k_rot.expand(-1, self.heads, -1)), -1)
        output = self.attn(query, key, value.contiguous())
        return self.o_proj(output.reshape(count, -1))


def build_from_config(config, device, dtype):
    quant = config.quantization_config
    if quant["activation_scheme"] != "static" or quant["weight_block_size"] is not None:
        raise ValueError("Selected Mistral4 uses per-tensor static FP8")
    if config.n_group != 1 or config.topk_group != 1 or config.first_k_dense_replace != 0:
        raise ValueError("Selected Mistral4 uses a single expert group and all MoE layers")
    if not config.rope_interleave or config.qk_head_dim != config.v_head_dim or config.q_lora_rank is None:
        raise ValueError("Selected Mistral4 uses interleaved RoPE and equal Q/K/V head widths")
    model = LlamaForCausalLM(decoder_config(config, dtype)).to(device=device, dtype=dtype)
    for layer in model.model.layers:
        layer.self_attn = ExpandedStaticAttention(config).to(device=device)
        layer.self_attn.q_a_layernorm.to(dtype=dtype)
        layer.self_attn.kv_a_layernorm.to(dtype=dtype)
        layer.mlp = StaticExperts(config).to(device=device)
        layer.mlp.router_weight = layer.mlp.router_weight.to(dtype=dtype)
    rope = config.rope_parameters
    yarn = YarnRotaryEmbedding(config.qk_rope_head_dim, rope["original_max_position_embeddings"],
                              rope["rope_theta"], rope["factor"], beta_fast=rope["beta_fast"],
                              beta_slow=rope["beta_slow"], mscale=rope["mscale"],
                              mscale_all_dim=rope["mscale_all_dim"], is_neox_style=True)
    # Keep the native frequencies while retaining only the evaluated context.
    yarn.cos_sin_cache = yarn.cos_sin_cache[:config.max_position_embeddings].to(device=device, dtype=dtype).clone()
    model.model.rotary_emb = yarn
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = yarn
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    consumed = set()

    def get(key):
        consumed.add(key)
        return state_dict[key]

    def copy(parameter, key):
        value = get(key)
        if value.shape != parameter.shape:
            raise ValueError(f"Mistral4 weight shape mismatch: {key}")
        parameter.copy_(value)

    def linear(module, prefix):
        module.load(get(prefix + ".weight"), get(prefix + ".weight_scale_inv"), get(prefix + ".activation_scale"))

    copy(model.model.embed_tokens.embedding_op.emb.weight, "model.embed_tokens.weight")
    copy(model.model.norm.weight, "model.norm.weight")
    copy(model.lm_head.embedding_op.emb.weight, "lm_head.weight")
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            copy(getattr(layer, name).weight, p + name + ".weight")
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            copy(getattr(layer.self_attn, name).weight, p + "self_attn." + name + ".weight")
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
            linear(getattr(layer.self_attn, name), p + "self_attn." + name)
        copy(layer.mlp.router_weight, p + "mlp.gate.weight")
        for source, target in (("gate_up_proj", layer.mlp.gate_up), ("down_proj", layer.mlp.down)):
            stem = p + "mlp.experts." + source
            weights, scales, activation = get(stem), get(stem + "_scale_inv"), get(stem + "_activation_scale")
            for e, module in enumerate(target):
                module.load(weights[e], scales[e], activation[e])
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear(getattr(layer.mlp.shared_experts, name), p + "mlp.shared_experts." + name)
    if consumed != state_dict.keys():
        raise KeyError(f"Unused Mistral4 state: {sorted(state_dict.keys() - consumed)}")
