"""xLSTM causal inference with an explicit, stabilized mLSTM recurrence.

Each state step uses existing operations and ProductGate. The matrix, vector,
and stabilizer states retain their native meaning and FP32 internal storage. This
construction exposes the cost of separate operations instead of claiming the
performance of the reference's fused, chunked recurrence.
"""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.log_sigmoid import LogSigmoid
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.tensor_ops import Exp


class Recurrence(nn.Module):
    """Native C/N/M state updates, with linear work in sequence length."""

    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.bmm, self.logsigmoid = BatchMatMul(), LogSigmoid()
        self.maximum, self.exp = MaxPool2d((1, 2)), Exp()
        self.product, self.square = ProductGate(), SquaredReLU()
        self.normalize = BatchNorm2d(1, eps=1e-30, affine=False)
        # Runtime statistics are state-step intermediates, not model weights.
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            self.normalize.register_buffer(name, None, persistent=False)

    def multiply(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        return self.product(torch.cat((left, right), dim=-1))

    def pairwise_max(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        pairs = torch.stack((left, right), dim=-1).reshape(-1, 1, 1, 2)
        return self.maximum(pairs).reshape_as(left)

    def matmul(self, left, right):
        result = self.bmm(left.flatten(0, 1), right.flatten(0, 1))
        return result.reshape(*left.shape[:2], left.shape[-2], right.shape[-1])

    def step(self, scaled_query, key, value, input_gate, log_forget, state):
        cell, normalizer, maximum = state
        updated_max = self.pairwise_max(log_forget + maximum, input_gate)
        forget = self.exp(log_forget + maximum - updated_max)
        write = self.exp(input_gate - updated_max)

        outer = self.matmul(key.unsqueeze(-1), value.unsqueeze(-2))
        cell = self.multiply(forget.unsqueeze(-1), cell) + self.multiply(write.unsqueeze(-1), outer)
        normalizer = self.multiply(forget, normalizer) + self.multiply(write, key)
        numerator = self.matmul(scaled_query.unsqueeze(-2), cell.to(scaled_query.dtype)).squeeze(-2).float()
        projected = self.matmul(
            scaled_query.unsqueeze(-2), normalizer.to(scaled_query.dtype).unsqueeze(-1),
        ).squeeze(-2)
        absolute = self.pairwise_max(projected, -projected)
        denominator = (self.pairwise_max(absolute, self.exp(-updated_max)) + self.eps).float()

        # Existing BatchNorm divides by sqrt(d*d), as in the RWKV composition.
        # Gates are soft-capped and d is positive; statistics stay inside forward.
        self.normalize.running_var = self.square(denominator).flatten()
        output = self.normalize(numerator.reshape(1, -1, numerator.shape[-1], 1))
        return output.reshape_as(numerator).to(scaled_query.dtype), (cell, normalizer, updated_max)

    def forward(self, query, key, value, input_gate, forget_gate, state=None):
        batch, heads, length, key_width = query.shape
        if state is None:
            cell = torch.zeros(batch, heads, key_width, value.shape[-1], device=query.device, dtype=torch.float32)
            normalizer = torch.zeros(batch, heads, key_width, device=query.device, dtype=torch.float32)
            maximum = torch.zeros(batch, heads, 1, device=query.device, dtype=torch.float32)
            state = cell, normalizer, maximum
        else:
            state = tuple(tensor.float() for tensor in state)
        # These operations are independent of recurrent state, so one call
        # handles the complete sequence without adding matrix-state storage.
        scaled_query = query * (key_width ** -0.5)
        log_forget = self.logsigmoid(forget_gate)
        self.normalize.running_mean = torch.zeros(batch * heads, device=query.device, dtype=torch.float32)
        output = torch.empty_like(value)
        for index in range(length):
            output[:, :, index], state = self.step(
                scaled_query[:, :, index], key[:, :, index], value[:, :, index],
                input_gate[:, :, index:index + 1], log_forget[:, :, index:index + 1], state,
            )
        return output, state


class MultiHeadNorm(nn.Module):
    def __init__(self, heads, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads * width))
        self.norm = LayerNorm(width, eps, elementwise_affine=False)
        self.product = ProductGate()

    def forward(self, hidden):
        hidden = self.norm(hidden).flatten(-2)
        return self.product(torch.cat((hidden, self.weight.expand_as(hidden)), dim=-1))


class MemoryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, self.heads = config.embedding_dim, config.num_heads
        key_width = int(hidden * config.qk_dim_factor)
        value_width = int(hidden * config.v_dim_factor)
        self.q, self.k = Linear(hidden, key_width, bias=False), Linear(hidden, key_width, bias=False)
        self.v = Linear(hidden, value_width, bias=False)
        self.ogate_preact = Linear(hidden, value_width, bias=False)
        self.igate_preact = Linear(hidden, self.heads, bias=True)
        self.fgate_preact = Linear(hidden, self.heads, bias=True)
        self.out_proj = Linear(value_width, hidden, bias=False)
        self.multihead_norm = MultiHeadNorm(self.heads, value_width // self.heads, config.norm_eps)
        self.recurrence = Recurrence(config.eps)
        self.tanh, self.sigmoid, self.product = Tanh(), Sigmoid(), ProductGate()
        self.gate_cap = config.gate_soft_cap

    def forward(self, hidden, state):
        batch, length, _ = hidden.shape
        query, key, value = [projection(hidden).reshape(batch, length, self.heads, -1).transpose(1, 2)
                             for projection in (self.q, self.k, self.v)]
        input_gate = self.tanh(self.igate_preact(hidden) / self.gate_cap) * self.gate_cap
        forget_gate = self.tanh(self.fgate_preact(hidden) / self.gate_cap) * self.gate_cap
        output, state = self.recurrence(
            query, key, value, input_gate.transpose(1, 2), forget_gate.transpose(1, 2), state,
        )
        output = self.multihead_norm(output.transpose(1, 2))
        gate = self.sigmoid(self.ogate_preact(hidden))
        return self.out_proj(self.product(torch.cat((gate, output), dim=-1))), state


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        multiple = config.ffn_round_up_to_multiple_of
        width = int(((config.embedding_dim * config.ffn_proj_factor + multiple - 1) // multiple) * multiple)
        self.proj_up_gate = Linear(config.embedding_dim, width, bias=False)
        self.proj_up = Linear(config.embedding_dim, width, bias=False)
        self.proj_down = Linear(width, config.embedding_dim, bias=False)
        self.activation = SiluAndMul()

    def forward(self, hidden):
        projected = torch.cat((self.proj_up_gate(hidden), self.proj_up(hidden)), dim=-1)
        # Native xLSTM rounds SiLU to the input dtype before the multiplication.
        return self.proj_down(self.activation.forward_native(projected))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_mlstm = RMSNormNative(config.embedding_dim, config.norm_eps)
        self.norm_ffn = RMSNormNative(config.embedding_dim, config.norm_eps)
        self.mlstm_layer, self.ffn = MemoryLayer(config), FeedForward(config)

    def forward(self, hidden, state):
        update, state = self.mlstm_layer(self.norm_mlstm(hidden), state)
        hidden = hidden + update
        return hidden + self.ffn(self.norm_ffn(hidden)), state


class XLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.embeddings = Embedding(config.vocab_size, config.embedding_dim)
        self.backbone.blocks = nn.ModuleList([Block(config) for _ in range(config.num_blocks)])
        self.backbone.out_norm = RMSNormNative(config.embedding_dim, config.norm_eps)
        self.lm_head = Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.tanh, self.logit_cap = Tanh(), config.output_logit_soft_cap

    def forward(self, ids, previous=None):
        hidden = self.backbone.embeddings(ids)
        states = []
        for index, block in enumerate(self.backbone.blocks):
            hidden, state = block(hidden, None if previous is None else previous[index])
            # HF copies the FP32 recurrence result into its public cache, whose
            # dtype follows the embeddings. Continuation reads those rounded values.
            states.append(tuple(tensor.to(hidden.dtype) for tensor in state))
        logits = self.lm_head(self.backbone.out_norm(hidden)).float()
        logits = self.tanh(logits / self.logit_cap) * self.logit_cap
        return logits, states


def build_from_config(config, device, dtype):
    if (config.weight_mode != "single" or config.use_bias or not config.use_cache
            or config.inference_state_dtype != "float32" or not config.norm_reduction_force_float32):
        raise ValueError("xLSTM audit implements the selected checkpoint's bias-free single-weight cached FP32-state mode")
    return XLSTM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, weights, config):
    mapped = dict(weights)
    mapped["backbone.embeddings.emb.weight"] = mapped.pop("backbone.embeddings.weight")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs["input_ids"]
    prefix, tokens = ids[:, :-2], ids[:, -2:]
    state = {}

    def run(token_ids, previous=None):
        logits, state["output"] = model(token_ids, previous)
        return {"logits": logits}

    def collect(output):
        result = dict(output)
        for index, states in enumerate(state.pop("output")):
            for kind, tensor in enumerate(states):
                result[f"cache_params.rnn_state.{index}.{kind}"] = tensor
        return result

    def prepare_decode(index):
        _, state["cache"] = model(prefix)
        for previous in range(index):
            _, state["cache"] = model(tokens[:, previous:previous + 1], state["cache"])

    workloads = {"prefill": Workload(run=lambda: run(prefix), collect=collect)}
    for index in range(2):
        workloads[f"decode_{index + 1}"] = Workload(
            run=lambda index=index: run(tokens[:, index:index + 1], state["cache"]),
            prepare=lambda index=index: prepare_decode(index), collect=collect,
        )
    return workloads
