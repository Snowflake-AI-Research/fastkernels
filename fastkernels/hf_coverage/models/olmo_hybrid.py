"""OLMo-Hybrid's NoPE attention and existing gated delta-rule recurrence."""

import torch
import triton
from torch import nn

from fastkernels.hf_coverage.patches.mamba_conv_weight import fp32_causal_conv_weight
from fastkernels.tasks.baseline.L1.bitnet_rms_norm import BitNetRMSNorm
from fastkernels.tasks.baseline.L1.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gated_delta_rule import chunk_gated_delta_rule, _fused_post_conv_kernel
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_gated import FusedRMSNormGated
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP

from .lfm2 import make_workloads


def prepare_gates(mixed, a, b, decay, bias, heads, key_dim, value_dim):
    """Reuse the native prep kernel with equal branch tiles for unequal K/V.

    Its public wrapper independently rounds K/V tile sizes, which fails Triton
    branch-type merging at native K96/V192. Equal tiles retain all native masks.
    """
    length = mixed.shape[0]
    q = mixed.new_empty(length, heads, key_dim)
    k = torch.empty_like(q)
    v = mixed.new_empty(length, heads, value_dim)
    g = torch.empty(length, heads, device=mixed.device, dtype=torch.float32)
    beta = torch.empty_like(g)
    tile = triton.next_power_of_2(max(key_dim, value_dim))
    _fused_post_conv_kernel[(triton.cdiv(length, 16), 2 * heads)](
        mixed_qkv_ptr=mixed, a_ptr=a, b_ptr=b, A_log_ptr=decay, dt_bias_ptr=bias,
        q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
        stride_x_tok=mixed.stride(0), stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
        stride_q_tok=q.stride(0), stride_k_tok=k.stride(0), stride_v_tok=v.stride(0),
        L=length, H=heads, HV=heads, K=key_dim, V=value_dim,
        APPLY_L2NORM=False, L2NORM_EPS=1e-6, OUTPUT_G_EXP=False,
        SOFTPLUS_THRESHOLD=20.0, BLOCK_T=16, BK=tile, BV=tile,
    )
    return q, k, v, g, beta


class ShortConvolution(Conv1dNative):
    def __init__(self, width, kernel):
        super().__init__(width, width, kernel, groups=width, bias=False)
        self.kernel_size = kernel
        self.activation = SiLU()
        self.state = None

    def forward(self, hidden):
        if hidden.is_cuda:
            batch, length, width = hidden.shape
            fresh = self.state is None
            if fresh:
                self.storage = hidden.new_zeros(batch + 1, self.kernel_size - 1, width).transpose(1, 2)
                self.state = self.storage[1:]
            slots = torch.arange(1, batch + 1, device=hidden.device, dtype=torch.int32)
            if length == 1 and not fresh:
                return causal_conv1d_update(hidden[:, 0], self.storage, self.fp32_weight, None,
                                             activation="silu", conv_state_indices=slots).unsqueeze(1)
            starts = torch.arange(batch + 1, device=hidden.device, dtype=torch.int32) * length
            output = causal_conv1d_fn(
                hidden.reshape(batch * length, width).T, self.fp32_weight, None,
                self.storage, starts, cache_indices=slots,
                has_initial_state=torch.full((batch,), not fresh, device=hidden.device, dtype=torch.bool),
                activation="silu",
            )
            return output.T.reshape(batch, length, width)
        hidden = hidden.transpose(1, 2)
        history = self.state
        if history is None:
            history = hidden.new_zeros(hidden.shape[0], hidden.shape[1], self.kernel_size - 1)
        joined = torch.cat((history, hidden), dim=-1)
        self.state = joined[..., -(self.kernel_size - 1):].contiguous()
        return self.activation(super().forward(joined)).transpose(1, 2)


class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.linear_num_key_heads
        self.key_dim, self.value_dim = config.linear_key_head_dim, config.linear_value_head_dim
        self.allow_negative = config.linear_allow_neg_eigval
        for name, width in (("q", self.heads * self.key_dim), ("k", self.heads * self.key_dim),
                            ("v", self.heads * self.value_dim), ("g", self.heads * self.value_dim),
                            ("a", self.heads), ("b", self.heads)):
            setattr(self, name + "_proj", Linear(config.hidden_size, width, bias=False))
        self.o_proj = Linear(self.heads * self.value_dim, config.hidden_size, bias=False)
        self.q_conv1d = ShortConvolution(self.heads * self.key_dim, config.linear_conv_kernel_dim)
        self.k_conv1d = ShortConvolution(self.heads * self.key_dim, config.linear_conv_kernel_dim)
        self.v_conv1d = ShortConvolution(self.heads * self.value_dim, config.linear_conv_kernel_dim)
        self.A_log = nn.Parameter(torch.empty(self.heads))
        self.dt_bias = nn.Parameter(torch.empty(self.heads))
        self.o_norm = FusedRMSNormGated(self.value_dim, eps=1e-5)
        self.state = None

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        mixed = torch.cat([getattr(self, name + "_conv1d")(getattr(self, name + "_proj")(hidden))
                           for name in ("q", "k", "v")], dim=-1)
        q, k, v, g, beta = prepare_gates(
            mixed.reshape(batch * length, -1).contiguous(),
            self.a_proj(hidden).reshape(batch * length, self.heads),
            self.b_proj(hidden).reshape(batch * length, self.heads),
            self.A_log, self.dt_bias, self.heads, self.key_dim, self.value_dim,
        )
        # HF rounds sigmoid to the projection dtype before the fixed factor2.
        beta = beta.to(hidden.dtype)
        if self.allow_negative:
            beta = beta * 2.0
        output, self.state = chunk_gated_delta_rule(
            q.reshape(batch, length, self.heads, self.key_dim),
            k.reshape(batch, length, self.heads, self.key_dim),
            v.reshape(batch, length, self.heads, self.value_dim),
            g.reshape(batch, length, self.heads), beta.reshape(batch, length, self.heads),
            initial_state=self.state, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        gate = self.g_proj(hidden).reshape(-1, self.value_dim)
        output = self.o_norm(output.reshape(-1, self.value_dim), gate)
        return self.o_proj(output.reshape(batch, length, -1))


class NoPEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        for name, heads in (("q", self.heads), ("k", self.kv_heads), ("v", self.kv_heads)):
            setattr(self, name + "_proj", Linear(config.hidden_size, heads * self.head_dim, bias=False))
        self.o_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = BitNetRMSNorm(self.heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = BitNetRMSNorm(self.kv_heads * self.head_dim, config.rms_norm_eps)
        self.attention = DenseAttention(backend="sdpa")
        self.keys = self.values = None

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        q = self.q_norm(self.q_proj(hidden)).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_norm(self.k_proj(hidden)).reshape(batch, length, self.kv_heads, self.head_dim)
        v = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.head_dim)
        if self.keys is not None:
            k, v = torch.cat((self.keys, k), dim=1), torch.cat((self.values, v), dim=1)
        self.keys, self.values = k, v
        output = self.attention(q, k, v, causal=length > 1)
        return self.o_proj(output.reshape(batch, length, -1))


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.is_attention = kind == "full_attention"
        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.is_attention:
            self.self_attn = NoPEAttention(config)
            self.post_feedforward_layernorm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.linear_attn = GatedDeltaNet(config)
            self.input_layernorm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden):
        if self.is_attention:
            hidden = hidden + self.post_attention_layernorm(self.self_attn(hidden))
            return hidden + self.post_feedforward_layernorm(self.mlp(hidden))
        hidden = hidden + self.linear_attn(self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class OlmoHybridForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.model.layers = nn.ModuleList([Layer(config, kind) for kind in config.layer_types])
        self.model.norm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, ids, positions):
        hidden = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return self.lm_head(self.model.norm(hidden))

    def reset(self):
        for layer in self.model.layers:
            if layer.is_attention:
                layer.self_attn.keys = layer.self_attn.values = None
            else:
                layer.linear_attn.state = None
                for name in ("q", "k", "v"):
                    getattr(layer.linear_attn, name + "_conv1d").state = None


def build_from_config(config, device, dtype):
    if (config.rope_parameters.get("rope_theta") is not None or config.attention_bias
            or config.linear_num_key_heads != config.linear_num_value_heads
            or config.num_attention_heads != config.num_key_value_heads
            or config.hidden_act != "silu" or config.tie_word_embeddings or not config.use_cache):
        raise ValueError("The selected OLMo-Hybrid checkpoint uses NoPE, equal head counts, untied cached SiLU layers")
    model = OlmoHybridForCausalLM(config).to(device=device, dtype=dtype).eval()
    # HF creates A_log in FP32; the shared reference weights retain that dtype.
    for layer in model.model.layers:
        if not layer.is_attention:
            layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
    return model


def load_state_dict_into(model, weights, config):
    mapped = dict(weights)
    mapped["model.embed_tokens.emb.weight"] = mapped.pop("model.embed_tokens.weight")
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}.mlp."
        mapped[prefix + "gate_up_proj.weight"] = torch.cat([
            mapped.pop(prefix + name + "_proj.weight") for name in ("gate", "up")])
    model.load_state_dict(mapped, strict=True)
    for layer in model.model.layers:
        if not layer.is_attention:
            for name in ("q", "k", "v"):
                convolution = getattr(layer.linear_attn, name + "_conv1d")
                convolution.fp32_weight = fp32_causal_conv_weight(convolution.weight)
