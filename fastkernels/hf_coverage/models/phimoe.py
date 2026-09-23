"""PhiMoE's affine-normalized decoder and two-pass SparseMixer inference.

Routing is composed from existing reductions, comparisons and normalization;
it does not replace the native masks with ordinary top-k renormalization.
"""

import copy
import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.dinov3_rope import apply_rot_embed_cat
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.fused_experts import FusedExperts
from ..patches.codec_top1 import CodecTop1
from .gpt_neox import DecoderLM
from .llama import make_workloads as decoder_workloads
from .phi import BiasedHeadProduct
from ..runner import Workload


class SparseMixer(nn.Module):
    """Two native masked softmax domains, with linear intermediate storage.

    FP64 BatchNorm statistics avoid under/overflow when squaring any positive
    finite BF16/FP32 denominator. Numerators first round in the source dtype;
    quotients round back before the strict comparison. For zero denominators,
    native 0/0 -> NaN and this composition's zero both compare False against
    the documented positive threshold. No equality of that unused NaN is
    claimed. All casts, runtime statistics and extra launches remain timed.
    """
    def __init__(self, jitter):
        super().__init__()
        if jitter < 0:
            raise ValueError('SparseMixer composition requires nonnegative inference jitter')
        self.jitter = jitter
        self.select, self.pool, self.softmax = CodecTop1(), MaxPool2d((1, 3)), Softmax()
        self.square = SquaredReLU()
        self.normalize = BatchNorm2d(1, eps=1e-300, affine=False).eval()
        self.normalize._non_persistent_buffers_set.update(self.normalize._buffers)

    def forward(self, scores):
        weights, indices, selection_scores = [], [], scores
        for step in range(2):
            selected = self.select(selection_scores)[..., None]
            maximum = selection_scores.gather(-1, selected)
            packed = torch.stack((scores, -scores, maximum.expand_as(scores)), -1)
            factor = self.pool(packed.reshape(-1, 1, 1, 3)).reshape_as(scores)
            difference = maximum - scores
            self.normalize.running_mean = torch.zeros(factor.numel(), device=scores.device, dtype=torch.float64)
            self.normalize.running_var = self.square(factor.double()).flatten()
            quotient = self.normalize(difference.double().reshape(1, -1, 1, 1)).reshape_as(scores).to(scores.dtype)
            threshold = torch.full_like(quotient, 2 * self.jitter)
            mask = self.select(torch.stack((threshold, quotient), -1)).bool()
            probability = self.softmax(selection_scores.masked_fill(mask, -torch.inf))
            weights.append(probability.gather(-1, selected))
            indices.append(selected)
            if step == 0:
                selection_scores = scores.scatter(-1, selected, -torch.inf)
        return torch.cat(weights, -1), torch.cat(indices, -1)


class LongRotary(nn.Module):
    """PhiMoE's configuration-derived short/long factors and amplitude scales."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(self, positions, query, key):
        rope = self.config.rope_parameters
        # Supplied positions and fixed configuration are inference metadata.
        long = int(positions.max()) + 1 > rope['original_max_position_embeddings']
        factors = torch.tensor(rope['long_factor' if long else 'short_factor'], device=query.device, dtype=torch.float32)
        inverse = 1.0 / (factors * rope['rope_theta'] ** (torch.arange(0, self.head_dim, 2, device=query.device).float() / self.head_dim))
        phase = positions.float()[:, None] * inverse[None]
        scale = rope['long_mscale' if long else 'short_mscale']
        phase = torch.cat((phase, phase), -1)
        embedding = torch.cat((phase.sin() * scale, phase.cos() * scale), -1).to(query.dtype)[:, None]
        count = query.shape[0]
        query = apply_rot_embed_cat(query.reshape(count, -1, self.head_dim), embedding)
        key = apply_rot_embed_cat(key.reshape(count, -1, self.head_dim), embedding)
        return query.reshape(count, -1), key.reshape(count, -1)


class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.routing = SparseMixer(config.router_jitter_noise)
        self.gate_up_proj = nn.Parameter(torch.empty(config.num_local_experts, 2 * config.intermediate_size, config.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(config.num_local_experts, config.hidden_size, config.intermediate_size))
        self.experts = FusedExperts()
        self.num_experts = config.num_local_experts

    def forward(self, hidden):
        weights, selected = self.routing(self.router(hidden))
        return self.experts(hidden, self.gate_up_proj, self.down_proj, weights, selected, self.num_experts)


class Layer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        width = config.hidden_size
        self.input_layernorm = LayerNorm(width, config.rms_norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(width, config.rms_norm_eps, promote_fp32=False)
        self.self_attn = LlamaAttention(width, config.num_attention_heads, config.num_key_value_heads,
            width // config.num_attention_heads, rotary_emb=rotary, bias=config.attention_bias,
            o_proj_bias=config.attention_bias, sliding_window=config.sliding_window, layer_idx=0)
        self.mlp = Experts(config)

    def forward(self, positions, hidden):
        hidden = hidden + self.self_attn(positions, self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


def build_from_config(config, device, dtype):
    if (_tp_size() != 1 or config.hidden_act != 'silu' or config.num_experts_per_tok != 2
            or config.tie_word_embeddings or not config.use_cache or config.output_router_logits
            or config.rope_parameters['rope_type'] != 'longrope'):
        raise ValueError('PhiMoE requires the selected untied, cached SiLU/LongRoPE inference with two SparseMixer experts')
    if config.num_local_experts != 16 or config.router_jitter_noise != 0.01:
        raise ValueError('Keep the documented sixteen-expert SparseMixer threshold unchanged')
    decoder = copy.copy(config)
    decoder.layer_norm_eps = config.rms_norm_eps
    rotary = LongRotary(config)
    model = DecoderLM(decoder, [Layer(config, rotary) for _ in range(config.num_hidden_layers)])
    if config.lm_head_bias:
        model.lm_head.linear_op = BiasedHeadProduct(config.hidden_size, config.vocab_size)
        model.lm_head.linear_op.weight = model.lm_head.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {'model.embed_tokens.embedding_op.emb.weight': remaining.pop('model.embed_tokens.weight'),
              'lm_head.embedding_op.emb.weight': remaining.pop('lm_head.weight')}
    if config.lm_head_bias:
        mapped['lm_head.linear_op.weight'] = mapped['lm_head.embedding_op.emb.weight']
        mapped['lm_head.linear_op.bias'] = remaining.pop('lm_head.bias')
    for field in ('weight', 'bias'):
        mapped['model.norm.' + field] = remaining.pop('model.norm.' + field)
    for index in range(config.num_hidden_layers):
        prefix = f'model.layers.{index}.'
        for field in ('weight', 'bias'):
            for part in ('input_layernorm', 'post_attention_layernorm'):
                mapped[prefix + part + '.' + field] = remaining.pop(prefix + part + '.' + field)
        for field in ('weight', 'bias') if config.attention_bias else ('weight',):
            mapped[prefix + 'self_attn.qkv_proj.' + field] = torch.cat([
                remaining.pop(prefix + f'self_attn.{part}_proj.' + field) for part in ('q', 'k', 'v')])
            mapped[prefix + 'self_attn.o_proj.' + field] = remaining.pop(prefix + 'self_attn.o_proj.' + field)
        mapped[prefix + 'mlp.router.weight'] = remaining.pop(prefix + 'mlp.router.weight')
        for name in ('gate_up_proj', 'down_proj'):
            mapped[prefix + 'mlp.' + name] = remaining.pop(prefix + 'mlp.experts.' + name)
    if remaining:
        raise KeyError(f'Unmapped PhiMoE state: {sorted(remaining)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if inputs['input_ids'].shape[0] != 1:
        raise ValueError('The bounded PhiMoE cache comparison uses one sequence')
    if case is not None and case.get('workload') == 'causal_lm_continuation':
        return decoder_workloads(
            model, inputs, config, case=case,
            cache_windows=[config.sliding_window] * config.num_hidden_layers,
        )
    workloads = decoder_workloads(model, inputs, config)
    for phase, workload in list(workloads.items()):
        def run(workload=workload, phase=phase):
            output = workload.run()
            length = inputs['input_ids'].shape[1] - (phase == 'prefill')
            start = max(0, length - config.sliding_window + 1) if config.sliding_window else 0
            for index, layer in enumerate(model.model.layers):
                attention = layer.self_attn.attn
                for name, cache in (('key', attention.k_cache), ('value', attention.v_cache)):
                    if attention.kv_layout == 'HND':
                        cache = cache.transpose(1, 2)
                    output[f'past_key_values.{index}.{name}'] = cache.flatten(0, 1)[start:length].transpose(0, 1)[None]
            return output
        workloads[phase] = Workload(run=run, prepare=workload.prepare)
    return workloads
