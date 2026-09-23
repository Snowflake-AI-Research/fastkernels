"""RWKV-v4 causal LM composed from existing operations and ProductGate.

The recurrence retains HF's scaled numerator, denominator, and running maximum.
Existing pointwise and normalization operations implement each state step;
the state meaning and BF16 conversion boundaries match HF. Separate operations
can still round differently from its fused core and incur many CUDA launches.
"""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.frozen_batch_norm2d import FrozenBatchNorm2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L1.tensor_ops import Exp


class TimeMix(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, 1, width))
        self.register_buffer("complement", torch.empty(1, 1, width))
        self.product = ProductGate()

    def forward(self, hidden, previous):
        current = self.product(torch.cat((hidden, self.weight.expand_as(hidden)), dim=-1))
        shifted = self.product(torch.cat((previous, self.complement.expand_as(previous)), dim=-1))
        return current + shifted


def previous_tokens(hidden, previous):
    first = torch.zeros_like(hidden[:, :1]) if previous is None else previous
    return torch.cat((first, hidden[:, :-1]), dim=1)


class WeightedKeyValue(nn.Module):
    """Compose HF's scaled recurrent state, including its BF16 readout rounding."""

    def __init__(self, width):
        super().__init__()
        self.time_decay = nn.Parameter(torch.empty(width))
        self.time_first = nn.Parameter(torch.empty(width))
        self.read_shift = FrozenBatchNorm2d(width, eps=0)
        self.state_shift = FrozenBatchNorm2d(width, eps=0)
        self.decode_shift = FrozenBatchNorm2d(width, eps=0)
        self.exp = Exp()
        self.maximum = MaxPool2d((1, 2))
        self.product = ProductGate()
        self.square = SquaredReLU()
        self.normalize = BatchNorm2d(1, eps=1e-30, affine=False)

    def prepare_weights(self):
        # HF CUDA prefill upcasts before exp; its native one-token path does not.
        self.read_shift.float().bias.copy_(self.time_first.float())
        self.state_shift.float().bias.copy_(-self.time_decay.float().exp())
        self.decode_shift.float().bias.copy_(-self.time_decay.exp())

    def multiply(self, left, right):
        return self.product(torch.cat((left, right), dim=-1))

    def pairwise_max(self, left, right):
        pairs = torch.stack((left, right), dim=-1).unsqueeze(-2)
        return self.maximum(pairs).squeeze(-1).squeeze(-1)

    def divide(self, numerator, denominator):
        # Stabilization makes the denominator >= 1 and at most the token count.
        # BatchNorm supplies division by sqrt(denominator**2 + eps), without
        # a reduction or expanded storage. All statistics are computed in forward.
        self.normalize.running_mean = torch.zeros_like(denominator).flatten()
        self.normalize.running_var = self.square(denominator).flatten()
        return self.normalize(numerator.reshape(1, -1, 1, 1)).reshape_as(numerator)

    def forward(self, key, value, state=None):
        batch, length, width = key.shape
        keys = key.float().permute(1, 0, 2).contiguous()
        values = value.float().permute(1, 0, 2).contiguous()
        read_keys = self.read_shift(keys.permute(1, 2, 0).unsqueeze(-1))
        read_keys = read_keys.squeeze(-1).permute(2, 0, 1).contiguous()
        if state is None:
            numerator = torch.zeros(batch, width, device=key.device, dtype=torch.float32)
            denominator = torch.zeros_like(numerator)
            maximum = torch.full_like(numerator, -1e30)
        else:
            numerator, denominator, maximum = state
        shift = self.decode_shift if length == 1 else self.state_shift
        output = torch.empty(length, batch, width, device=key.device, dtype=key.dtype)
        for index in range(length):
            read_max = self.pairwise_max(maximum, read_keys[index])
            previous_scale = self.exp(maximum - read_max)
            current_scale = self.exp(read_keys[index] - read_max)
            read_num = self.multiply(previous_scale, numerator) + self.multiply(current_scale, values[index])
            read_den = self.multiply(previous_scale, denominator) + current_scale
            # HF's cached CUDA prefill rounds the numerator before division.
            # Its native one-token path rounds only the final quotient.
            if length > 1 and key.dtype == torch.bfloat16:
                read_num = read_num.bfloat16().float()
            output[index] = self.divide(read_num, read_den).to(key.dtype)

            decayed_max = shift(maximum[:, :, None, None]).view(batch, width)
            updated_max = self.pairwise_max(decayed_max, keys[index])
            previous_scale = self.exp(decayed_max - updated_max)
            current_scale = self.exp(keys[index] - updated_max)
            numerator = self.multiply(previous_scale, numerator) + self.multiply(current_scale, values[index])
            denominator = self.multiply(previous_scale, denominator) + current_scale
            maximum = updated_max
        return output.permute(1, 0, 2).contiguous(), (numerator, denominator, maximum)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.time_mix_key = TimeMix(width)
        self.time_mix_value = TimeMix(width)
        self.time_mix_receptance = TimeMix(width)
        self.key = Linear(width, width, bias=False)
        self.value = Linear(width, width, bias=False)
        self.receptance = Linear(width, width, bias=False)
        self.output = Linear(width, width, bias=False)
        self.wkv = WeightedKeyValue(width)
        self.sigmoid = Sigmoid()
        self.product = ProductGate()

    def forward(self, hidden, state=None):
        previous, recurrent = (None, None) if state is None else state
        shifted = previous_tokens(hidden, previous)
        key = self.key(self.time_mix_key(hidden, shifted))
        value = self.value(self.time_mix_value(hidden, shifted))
        gate = self.sigmoid(self.receptance(self.time_mix_receptance(hidden, shifted)))
        weighted, recurrent = self.wkv(key, value, recurrent)
        output = self.output(self.product(torch.cat((gate, weighted), dim=-1)))
        return output, (hidden[:, -1:].clone(), recurrent)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.time_mix_key = TimeMix(width)
        self.time_mix_receptance = TimeMix(width)
        self.key = Linear(width, config.intermediate_size, bias=False)
        self.receptance = Linear(width, width, bias=False)
        self.value = Linear(config.intermediate_size, width, bias=False)
        self.activation = SquaredReLU()
        self.sigmoid = Sigmoid()
        self.product = ProductGate()

    def forward(self, hidden, previous=None):
        shifted = previous_tokens(hidden, previous)
        key = self.activation(self.key(self.time_mix_key(hidden, shifted)))
        gate = self.sigmoid(self.receptance(self.time_mix_receptance(hidden, shifted)))
        output = self.product(torch.cat((gate, self.value(key)), dim=-1))
        return output, hidden[:, -1:].clone()


class Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        if index == 0:
            self.pre_ln = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.ln1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.ln2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden, state=None):
        if hasattr(self, "pre_ln"):
            hidden = self.pre_ln(hidden)
        attention_state, previous_ffn = (None, None) if state is None else state
        attention, attention_state = self.attention(self.ln1(hidden), attention_state)
        hidden = hidden + attention
        feed_forward, previous_ffn = self.feed_forward(self.ln2(hidden), previous_ffn)
        return hidden + feed_forward, (attention_state, previous_ffn)


class RwkvForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Block(config, index) for index in range(config.num_hidden_layers))
        # Pinned HF uses LayerNorm's default epsilon for the final normalization.
        self.ln_out = LayerNorm(config.hidden_size, promote_fp32=False)
        self.head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, state=None):
        hidden = self.embeddings(input_ids)
        updated = []
        for index, block in enumerate(self.blocks):
            hidden, layer_state = block(hidden, None if state is None else state[index])
            updated.append(layer_state)
            if self.config.rescale_every > 0 and (index + 1) % self.config.rescale_every == 0:
                hidden = hidden / 2
        return self.head(self.ln_out(hidden)), updated


def build_from_config(config, device, dtype):
    if config.attention_hidden_size != config.hidden_size or config.tie_word_embeddings or not config.use_cache:
        raise ValueError("The selected RWKV checkpoint uses equal attention/model widths, an untied head, and caching")
    return RwkvForCausalLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"rwkv.embeddings.weight": model.embeddings.emb.weight, "head.weight": model.head.weight}
    for name in ("weight", "bias"):
        direct[f"rwkv.ln_out.{name}"] = getattr(model.ln_out, name)
    for index, block in enumerate(model.blocks):
        prefix = f"rwkv.blocks.{index}."
        for name in (("pre_ln",) if index == 0 else ()) + ("ln1", "ln2"):
            for kind in ("weight", "bias"):
                direct[prefix + f"{name}.{kind}"] = getattr(getattr(block, name), kind)
        for part in ("attention", "feed_forward"):
            module = getattr(block, part)
            for name in ("key", "value", "receptance") + (("output",) if part == "attention" else ()):
                direct[prefix + f"{part}.{name}.weight"] = getattr(module, name).weight
            for name in ("key", "receptance") + (("value",) if part == "attention" else ()):
                direct[prefix + f"{part}.time_mix_{name}"] = getattr(module, "time_mix_" + name).weight
        for name in ("time_decay", "time_first"):
            direct[prefix + "attention." + name] = getattr(block.attention.wkv, name)
    if direct.keys() != state_dict.keys():
        raise KeyError(f"RWKV weights differ: missing={sorted(direct.keys() - state_dict.keys())}, extra={sorted(state_dict.keys() - direct.keys())}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"RWKV weight shape differs: {name}")
        parameter.copy_(state_dict[name])
    for index, block in enumerate(model.blocks):
        block.attention.wkv.prepare_weights()
        for module in block.modules():
            if isinstance(module, TimeMix):
                module.complement.copy_(1 - module.weight)
        if config.rescale_every > 0:
            divisor = 2 ** (index // config.rescale_every)
            block.attention.output.weight.div_(divisor)
            block.feed_forward.value.weight.div_(divisor)


def make_workloads(model, inputs, config, *, case=None):
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs["input_ids"]
    if ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError("RWKV's selected workload is a single prompt followed by cached decoding")
    continuation = case is not None and case["workload"] == "causal_lm_continuation"
    steps = 2 if continuation else 1
    if ids.shape[1] <= steps:
        raise ValueError("RWKV needs a nonempty prompt before its continuation tokens")
    prompt = ids[:, :-steps]
    tokens = ids[:, -steps:]
    state = {}

    def run(token_ids, previous=None):
        logits, updated = model(token_ids, previous)
        state["output"] = updated
        return {"logits": logits}

    def collect(output):
        result = {"logits": output["logits"]}
        if continuation:
            layers = state.pop("output")
            # HF groups states by kind with shape [batch, width, layer].
            # This is only a layout conversion; recurrence work stays in run.
            result["state.0"] = torch.stack([ffn[:, 0] for _, ffn in layers], dim=-1)
            result["state.1"] = torch.stack([attention[0][:, 0] for attention, _ in layers], dim=-1)
            for kind in range(3):
                result[f"state.{kind + 2}"] = torch.stack(
                    [attention[1][kind] for attention, _ in layers], dim=-1,
                )
        return result

    def prepare_decode(step):
        _, state["cache"] = model(prompt)
        for previous in range(step):
            _, state["cache"] = model(tokens[:, previous:previous + 1], state["cache"])

    workloads = {"prefill": Workload(run=lambda: run(prompt), collect=collect)}
    for step in range(steps):
        name = f"decode_{step + 1}" if continuation else "decode"
        workloads[name] = Workload(
            run=lambda step=step: run(tokens[:, step:step + 1], state["cache"]),
            prepare=lambda step=step: prepare_decode(step),
            collect=collect,
        )
    return workloads
