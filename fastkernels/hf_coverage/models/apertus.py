"""Apertus xIELU decoder and whole-sequence greedy generation."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L2.cosyvoice3_hifigan import CausalConvRNNF0Predictor
from ..patches.codec_top1 import CodecTop1
from ..patches.product_gate import ProductGate
from ..runner import Workload


class XIELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_p = nn.Parameter(torch.empty(1))
        self.alpha_n = nn.Parameter(torch.empty(1))
        self.register_buffer('beta', torch.empty(()))
        self.register_buffer('eps', torch.empty(()))
        # Reuse the actual unchanged baseline child; no standalone ELU interface.
        self.elu = CausalConvRNNF0Predictor(num_class=1, in_channels=1, cond_channels=1).condnet[1]
        self.select, self.product = CodecTop1(), ProductGate()

    @torch.no_grad()
    def prepare_scalars(self):
        # Inference-constant learned-parameter preparation, excluded from timing.
        self.positive_scale = float(torch.nn.functional.softplus(self.alpha_p))
        self.negative_scale = float(self.beta + torch.nn.functional.softplus(self.alpha_n))
        self.linear_scale, self.threshold = float(self.beta), float(self.eps)
        if self.threshold >= 0:
            raise ValueError('The native xIELU clamp must be strictly negative')

    def forward(self, x):
        threshold = torch.full_like(x, self.threshold)
        clamp_index = self.select(torch.stack((threshold, x), -1))
        clamped = torch.where(clamp_index.bool(), threshold, x)
        positive = self.product(torch.cat((self.positive_scale * x, x), -1)) + self.linear_scale * x
        negative = (self.elu(clamped) - x) * self.negative_scale + self.linear_scale * x
        positive_index = self.select(torch.stack((torch.zeros_like(x), x), -1))
        return torch.where(positive_index.bool(), positive, negative)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = XIELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.dim = config.hidden_size // self.heads
        self.q_proj = Linear(config.hidden_size, self.heads * self.dim, bias=config.attention_bias)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.dim, bias=config.attention_bias)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.dim, bias=config.attention_bias)
        self.o_proj = Linear(self.heads * self.dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNormNative(self.dim, config.rms_norm_eps)
        self.k_norm = RMSNormNative(self.dim, config.rms_norm_eps)
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, x, positions, rotary, mask):
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.heads, self.dim))
        k = self.k_norm(self.k_proj(x).reshape(batch, length, self.kv_heads, self.dim))
        # Unchanged native callable preserves HF's BF16 intermediate stores.
        q, k = rotary.forward_native(positions.reshape(-1), q.reshape(batch * length, -1),
            k.reshape(batch * length, -1), self.dim, rotary.cos_sin_cache.to(q.dtype))
        q = q.reshape(batch, length, self.heads, self.dim)
        k = k.reshape(batch, length, self.kv_heads, self.dim)
        v = self.v_proj(x).reshape(batch, length, self.kv_heads, self.dim)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=2)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=2)
        output = self.attention(q, k, v, attn_mask=mask)
        return self.o_proj(output.reshape(batch, length, -1))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.mlp = Attention(config), MLP(config)
        self.attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.feedforward_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, positions, rotary, mask):
        x = x + self.self_attn(self.attention_layernorm(x), positions, rotary, mask)
        return x + self.mlp(self.feedforward_layernorm(x))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.model.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.model.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.select = CodecTop1()

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0):
        batch, length = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device)[None].expand(batch, -1)
        elif position_ids.shape[0] == 1:
            position_ids = position_ids.expand(batch, -1)
        index = torch.arange(length, device=input_ids.device)
        mask = (index[:, None] >= index[None, :])[None, None].expand(batch, 1, -1, -1)
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, position_ids, self.model.rotary_emb, mask)
        hidden = self.model.norm(hidden)
        return self.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=4, max_length=None,
                 eos_token_id=(2, 68, 72), pad_token_id=None, do_sample=False, use_cache=False):
        if input_ids.shape[0] != 1 or do_sample or use_cache:
            raise ValueError('Selected Apertus example uses one greedy prompt with cache disabled')
        if max_length is not None:
            max_new_tokens = max_length - input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        eos = () if eos_token_id is None else ((eos_token_id,) if isinstance(eos_token_id, int) else eos_token_id)
        sequences, logits = input_ids, []
        for _ in range(max_new_tokens):
            positions = attention_mask.long().cumsum(-1) - 1
            positions = positions.masked_fill(attention_mask == 0, 1)
            scores = self(sequences, attention_mask, positions, logits_to_keep=1)[:, -1].float()
            logits.append(scores)
            token = self.select(scores)[:, None]
            sequences = torch.cat((sequences, token), -1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(token)), -1)
            if int(token.item()) in eos:
                break
        return {'sequences': sequences, 'logits': tuple(logits)}


def build_from_config(config, device, dtype):
    if config.hidden_act != 'xielu' or config.tie_word_embeddings or config.use_cache:
        raise ValueError('Apertus checkpoint requires xIELU, an untied head, and disabled cache')
    rope = config.rope_parameters
    if rope['rope_type'] != 'llama3' or config.num_attention_heads != 4 * config.num_key_value_heads:
        raise ValueError('Apertus checkpoint requires Llama3 RoPE and 4:1 grouped-query attention')
    model = Model(config).to(device=device, dtype=dtype).eval()
    # Build after dtype conversion so fixed FP32 angles are rounded only at use.
    model.model.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_attention_heads,
        config.max_position_embeddings, rope['rope_theta'], rope['factor'],
        rope['low_freq_factor'], rope['high_freq_factor'], rope['original_max_position_embeddings']).to(device)
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped = {name: state_dict[name.replace('.emb.weight', '.weight')] for name in model.state_dict()}
    expected = {name.replace('.emb.weight', '.weight') for name in model.state_dict()}
    if set(state_dict) != expected:
        raise KeyError(f'Apertus state mismatch: missing={sorted(expected-set(state_dict))}, extra={sorted(set(state_dict)-expected)}')
    model.load_state_dict(mapped, strict=True)
    for layer in model.model.layers:
        layer.mlp.act_fn.prepare_scalars()


def make_workloads(model, inputs, config, case=None):
    options = {} if case is None else dict(case['generation_kwargs'])

    def run():
        output = model.generate(**inputs, **options)
        return {'sequences': output['sequences'],
                **{f'logits.{index}': value for index, value in enumerate(output['logits'])}}

    return {'generate': Workload(run=run)}
