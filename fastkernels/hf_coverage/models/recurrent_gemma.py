"""RecurrentGemma using diagonal GLA and existing supplied-statistics normalization."""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.fused_recurrent_gla import FusedRecurrentGLA
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.tensor_ops import Exp

from .lfm2 import make_workloads as legacy_workloads
from .stablelm import PartialRotary


class RGLRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.lru_width // self.heads
        self.recurrent_param = nn.Parameter(torch.empty(config.lru_width))
        for name in ("input", "recurrent"):
            setattr(self, name + "_gate_weight", nn.Parameter(torch.empty(self.heads, self.width, self.width)))
            setattr(self, name + "_gate_bias", nn.Parameter(torch.empty(self.heads, self.width)))
        self.matmul, self.sigmoid, self.exp = Matmul(), Sigmoid(), Exp()
        self.product, self.scan = ProductGate(), FusedRecurrentGLA()
        self.gamma = BatchNorm2d(1, eps=1e-30, affine=False, track_running_stats=False)
        self.state = None

    def gate(self, hidden, name):
        shaped = hidden.reshape(*hidden.shape[:-1], self.heads, self.width)
        weight, bias = getattr(self, name + "_gate_weight"), getattr(self, name + "_gate_bias")
        blocks = [self.matmul(shaped[..., head, :], weight[head].T, bias[head]) for head in range(self.heads)]
        return self.sigmoid(torch.stack(blocks, dim=-2).reshape_as(hidden))

    def forward(self, hidden, positions):
        batch, length, width = hidden.shape
        input_gate = self.gate(hidden, "input")
        log_decay = self.product(torch.cat((self.gate(hidden, "recurrent") * -8.0,
                                           self.decay_parameter.expand_as(hidden)), dim=-1))
        variance = 1.0 - self.exp(2.0 * log_decay)
        # v / sqrt(v + eps) implements sqrt(v), including zero. All input-
        # dependent statistics and temporary tensors remain inside execution.
        self.gamma.running_mean = torch.zeros(variance.numel(), device=hidden.device, dtype=torch.float32)
        self.gamma.running_var = variance.float().reshape(-1)
        self.gamma.track_running_stats = True
        multiplier = self.gamma(variance.float().reshape(1, -1, 1, 1)).reshape_as(hidden).to(hidden.dtype)
        multiplier = multiplier.masked_fill((positions == 0)[None, :, None], 1.0)
        values = self.product(torch.cat((hidden, input_gate), dim=-1))
        values = self.product(torch.cat((values, multiplier), dim=-1))
        log_decay = log_decay.masked_fill((positions == 0)[None, :, None], float("-inf"))
        # One scalar GLA state per channel: no channel-by-channel matrix.
        ones = torch.ones(batch, length, width, 1, device=hidden.device, dtype=hidden.dtype)
        output, self.state = self.scan(ones, ones, values.unsqueeze(-1), log_decay.unsqueeze(-1),
                                       scale=1.0, initial_state=self.state, output_final_state=True)
        return output.squeeze(-1)


class RecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_x = Linear(config.hidden_size, config.lru_width)
        self.linear_y = Linear(config.hidden_size, config.lru_width)
        self.linear_out = Linear(config.lru_width, config.hidden_size)
        self.conv_1d = Conv1dNative(config.lru_width, config.lru_width, config.conv1d_width,
                                  groups=config.lru_width, bias=True)
        self.kernel = config.conv1d_width
        self.rg_lru = RGLRU(config)
        self.activation, self.product = GELU(approximate="tanh"), ProductGate()
        self.conv_state = None

    def forward(self, hidden, positions):
        y = self.activation(self.linear_y(hidden))
        x = self.linear_x(hidden).transpose(1, 2)
        history = self.conv_state
        if history is None:
            history = x.new_zeros(x.shape[0], x.shape[1], self.kernel - 1)
        joined = torch.cat((history, x), dim=-1)
        self.conv_state = joined[..., -(self.kernel - 1):].contiguous()
        x = self.rg_lru(self.conv_1d(joined).transpose(1, 2), positions)
        return self.linear_out(self.product(torch.cat((x, y), dim=-1)))


class WindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        self.window = config.attention_window_size
        self.q_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.rotary = PartialRotary(self.dim, self.dim // 2, 2 * self.window,
                                   config.rope_parameters["rope_theta"])
        self.attention = DenseAttention(backend="sdpa")
        self.keys = self.values = None

    def forward(self, hidden, positions):
        batch, length, _ = hidden.shape
        q, k = self.rotary(positions.repeat(batch), self.q_proj(hidden).reshape(batch * length, -1),
                           self.k_proj(hidden).reshape(batch * length, -1))
        q, k = q.reshape(batch, length, self.heads, self.dim), k.reshape(batch, length, self.heads, self.dim)
        v = self.v_proj(hidden).reshape(batch, length, self.heads, self.dim)
        if self.keys is not None:
            k, v = torch.cat((self.keys, k), dim=1), torch.cat((self.values, v), dim=1)
        self.keys, self.values = k, v
        keys = torch.arange(k.shape[1], device=hidden.device)
        mask = (keys[None, :] <= positions[:, None]) & (keys[None, :] > positions[:, None] - self.window)
        output = self.attention(q, k, v, attn_mask=mask[None, None])
        return self.o_proj(output.reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.intermediate_size // 2
        self.gate_proj, self.up_proj = Linear(config.hidden_size, width), Linear(config.hidden_size, width)
        self.down_proj = Linear(width, config.hidden_size)
        self.activation = GeluAndMul(approximate="tanh")

    def forward(self, hidden):
        return self.down_proj(self.activation(torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), dim=-1)))


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.temporal_pre_norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.channel_pre_norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.temporal_block = WindowAttention(config) if kind == "attention" else RecurrentBlock(config)
        self.mlp_block = MLP(config)

    def forward(self, hidden, positions):
        hidden = hidden + self.temporal_block(self.temporal_pre_norm(hidden), positions)
        return hidden + self.mlp_block(self.channel_pre_norm(hidden))


class RecurrentGemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        kinds = [config.block_types[index % len(config.block_types)] for index in range(config.num_hidden_layers)]
        self.model.layers = nn.ModuleList([Layer(config, kind) for kind in kinds])
        self.model.final_norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.emb.weight
        self.scale = torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16).item()
        self.cap, self.tanh = config.logits_soft_cap, Tanh()

    def forward(self, ids, positions, *, output_hidden_states=False):
        hidden = self.model.embed_tokens(ids) * self.scale
        states = []
        for layer in self.model.layers:
            if output_hidden_states:
                states.append(hidden)
            hidden = layer(hidden, positions)
        hidden = self.model.final_norm(hidden)
        logits = self.tanh(self.lm_head(hidden) / self.cap) * self.cap
        if output_hidden_states:
            # Pinned HF returns these intermediates unconditionally from its LM.
            states.append(hidden)
            return {"logits": logits, **{f"hidden_states.{i}": value for i, value in enumerate(states)}}
        return logits

    def reset(self):
        for layer in self.model.layers:
            block = layer.temporal_block
            if isinstance(block, WindowAttention):
                block.keys = block.values = None
            else:
                block.conv_state = block.rg_lru.state = None


def build_from_config(config, device, dtype):
    if (config.hidden_activation != "gelu_pytorch_tanh" or not config.use_cache or not config.tie_word_embeddings
            or config.num_attention_heads != config.num_key_value_heads
            or config.rope_parameters["rope_type"] != "default"):
        raise ValueError("The selected RecurrentGemma path requires tied cached tanh-GELU, equal heads and default half RoPE")
    return RecurrentGemmaForCausalLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, weights, config):
    mapped = dict(weights)
    mapped["model.embed_tokens.emb.weight"] = mapped.pop("model.embed_tokens.weight")
    model.load_state_dict(mapped, strict=True)
    for layer in model.model.layers:
        block = layer.temporal_block
        if isinstance(block, RecurrentBlock):
            # Inference-constant parameter transform; no activation work moved.
            block.rg_lru.decay_parameter = torch.nn.functional.softplus(block.rg_lru.recurrent_param.detach())


def make_workloads(model, inputs, config, *, case=None):
    if case is None or case.get("workload") != "causal_lm_continuation":
        return legacy_workloads(model, inputs, config)
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs["input_ids"]
    prefix_length = ids.shape[1] - 2
    if prefix_length < 1:
        raise ValueError("RecurrentGemma continuation requires a prefix and two tokens")
    positions = torch.arange(ids.shape[1], device=ids.device)

    def prefill():
        return model(ids[:, :prefix_length], positions[:prefix_length], output_hidden_states=True)

    def advance(index):
        position = prefix_length + index
        return model(ids[:, position:position + 1], positions[position:position + 1], output_hidden_states=True)

    def prepare_step(index):
        model.reset()
        prefill()
        for previous in range(index):
            advance(previous)

    def collect(output):
        output = dict(output)
        for index, layer in enumerate(model.model.layers):
            block = layer.temporal_block
            prefix = f"past_key_values.{index}."
            if isinstance(block, WindowAttention):
                # Native sliding caches retain window-1 entries after the call;
                # compare that history without changing timed attention storage.
                for name, values in (("key", block.keys), ("value", block.values)):
                    output[prefix + name] = values[:, 1 - block.window:].transpose(1, 2)
            else:
                output[prefix + "conv_states"] = block.conv_state
                output[prefix + "recurrent_states"] = block.rg_lru.state.squeeze(-1).squeeze(-1)
        return output

    workloads = {"prefill": Workload(run=prefill, prepare=model.reset, collect=collect)}
    for index in range(2):
        workloads[f"decode_{index + 1}"] = Workload(
            run=lambda index=index: advance(index),
            prepare=lambda index=index: prepare_step(index), collect=collect,
        )
    return workloads
