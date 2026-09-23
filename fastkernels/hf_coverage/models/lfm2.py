"""LFM2's gated causal convolution and normalized grouped-query attention."""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul


class ShortConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.kernel_size = config.conv_L_cache
        self.in_proj = Linear(width, 3 * width, bias=config.conv_bias)
        self.out_proj = Linear(width, width, bias=config.conv_bias)
        self.conv = Conv1dNative(width, width, self.kernel_size, groups=width, bias=config.conv_bias)
        self.product = ProductGate()
        self.state = None

    def forward(self, hidden):
        gate, output_gate, values = self.in_proj(hidden).chunk(3, dim=-1)
        values = self.product(torch.cat((gate, values), dim=-1)).transpose(1, 2)
        if self.state is None:
            history = values.new_zeros(values.shape[0], values.shape[1], self.kernel_size - 1)
        else:
            history = self.state[..., 1:]
        history = torch.cat((history, values), dim=-1)
        self.state = history[..., -self.kernel_size:].contiguous()
        values = self.conv(history).transpose(1, 2)
        return self.out_proj(self.product(torch.cat((output_gate, values), dim=-1)))


class Attention(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        self.q_proj = Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = RMSNormNative(self.head_dim, config.norm_eps, allow_cuda_kernel=True)
        self.k_layernorm = RMSNormNative(self.head_dim, config.norm_eps, allow_cuda_kernel=True)
        self.rotary, self.attention = rotary, DenseAttention(backend="sdpa")
        self.keys = self.values = None

    def forward(self, hidden, positions):
        batch, length, _ = hidden.shape
        q = self.q_layernorm(self.q_proj(hidden).reshape(batch, length, self.heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(hidden).reshape(batch, length, self.kv_heads, self.head_dim))
        v = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.head_dim)
        q, k = self.rotary(positions.repeat(batch), q.reshape(batch * length, -1), k.reshape(batch * length, -1))
        q, k = q.reshape(batch, length, self.heads, self.head_dim), k.reshape(batch, length, self.kv_heads, self.head_dim)
        if self.keys is not None:
            k, v = torch.cat((self.keys, k), dim=1), torch.cat((self.values, v), dim=1)
        self.keys, self.values = k, v
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
        # A cached one-token call attends to the whole accumulated prefix.
        output = self.attention(q, k, v, causal=length > 1)
        return self.out_proj(output.reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.w1 = Linear(hidden, intermediate, bias=False)
        self.w3 = Linear(hidden, intermediate, bias=False)
        self.w2 = Linear(intermediate, hidden, bias=False)
        self.activation = SiluAndMul()

    def forward(self, hidden):
        return self.w2(self.activation(torch.cat((self.w1(hidden), self.w3(hidden)), dim=-1)))


class Layer(nn.Module):
    def __init__(self, config, kind, rotary, intermediate):
        super().__init__()
        self.is_attention = kind == "full_attention"
        if self.is_attention:
            self.self_attn = Attention(config, rotary)
        else:
            self.conv = ShortConv(config)
        self.operator_norm = RMSNormNative(config.hidden_size, config.norm_eps, allow_cuda_kernel=True)
        self.ffn_norm = RMSNormNative(config.hidden_size, config.norm_eps, allow_cuda_kernel=True)
        self.feed_forward = MLP(config.hidden_size, intermediate)

    def forward(self, hidden, positions):
        normalized = self.operator_norm(hidden)
        hidden = hidden + (self.self_attn(normalized, positions) if self.is_attention else self.conv(normalized))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class Lfm2ForCausalLM(nn.Module):
    def __init__(self, config, intermediate):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads,
                                 config.max_position_embeddings, config.rope_parameters["rope_theta"])
        self.model.layers = nn.ModuleList([Layer(config, kind, rotary, intermediate) for kind in config.layer_types])
        self.model.embedding_norm = RMSNormNative(config.hidden_size, config.norm_eps, allow_cuda_kernel=True)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.emb.weight

    def forward(self, ids, positions):
        hidden = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.model.embedding_norm(hidden))

    def reset(self):
        for layer in self.model.layers:
            if layer.is_attention:
                layer.self_attn.keys = layer.self_attn.values = None
            else:
                layer.conv.state = None


def build_from_config(config, device, dtype):
    if not config.use_cache or config.rope_parameters["rope_type"] != "default":
        raise ValueError("The selected LFM2 case uses cached inference and default rotary frequencies")
    intermediate = config.intermediate_size
    if config.block_auto_adjust_ff_dim:
        intermediate = int(2 * intermediate / 3)
        if config.block_ffn_dim_multiplier is not None:
            intermediate = int(config.block_ffn_dim_multiplier * intermediate)
            multiple = config.block_multiple_of
            intermediate = multiple * ((intermediate + multiple - 1) // multiple)
    return Lfm2ForCausalLM(config, intermediate).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, weights, config):
    mapped = dict(weights)
    mapped["model.embed_tokens.emb.weight"] = mapped.pop("model.embed_tokens.weight")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs["input_ids"]
    positions = torch.arange(ids.shape[1], device=ids.device)
    continuation = case is not None and case["workload"] == "causal_lm_continuation"
    steps = 2 if continuation else 1
    prefix_length = ids.shape[1] - steps
    if prefix_length < 1:
        raise ValueError("LFM2 requires a prefix before its continuation tokens")

    def prefill():
        return {"logits": model(ids[:, :prefix_length], positions[:prefix_length])}

    def advance(index):
        start = prefix_length + index
        return {"logits": model(ids[:, start:start + 1], positions[start:start + 1])}

    def prepare_decode(index):
        model.reset()
        prefill()
        for previous in range(index):
            advance(previous)

    def collect(output):
        if continuation:
            for index, layer in enumerate(model.model.layers):
                prefix = f"past_key_values.{index}."
                if layer.is_attention:
                    output[prefix + "key"] = layer.self_attn.keys.transpose(1, 2)
                    output[prefix + "value"] = layer.self_attn.values.transpose(1, 2)
                else:
                    output[prefix + "conv_states"] = layer.conv.state
        return output

    workloads = {"prefill": Workload(run=prefill, prepare=model.reset, collect=collect)}
    for index in range(steps):
        name = f"decode_{index + 1}" if continuation else "decode"
        workloads[name] = Workload(
            run=lambda index=index: advance(index),
            prepare=lambda index=index: prepare_decode(index),
            collect=collect,
        )
    return workloads
