"""Evolla protein/text components composed from existing operations."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh
from ..patches.gelu_python import PythonGELU
from ..runner import Workload
from .qwen_omni_sampling import OmniSampling


def norm(width, eps=1e-5):
    return LayerNorm(width, eps=eps, promote_fp32=False)


class CenteredAttention(nn.Module):
    """Native pre-softmax BF16 centering through MaxPool and unchanged BatchNorm.

    BatchNorm's supplied runtime mean is the row maximum; variance plus
    epsilon equals exactly one, making its inference transform a subtraction.
    All score and statistic tensors remain linear in the existing score size.
    """
    def __init__(self, fill):
        super().__init__()
        self.fill = fill
        self.matmul, self.softmax = BatchMatMul(), Softmax()
        self.maximum = MaxPool2d((1, 1))
        self.center = BatchNorm2d(1, eps=2. ** -20, affine=False)
        self.center._non_persistent_buffers_set.update(('running_mean', 'running_var', 'num_batches_tracked'))

    def forward(self, q, k, v, mask, scale):
        b, h, n, d = q.shape
        m = k.shape[-2]
        scores = self.matmul((q * scale).reshape(b * h, n, d), k.reshape(b * h, m, d).transpose(1, 2))
        self.maximum.kernel_size = self.maximum.stride = (1, m)
        values = scores.reshape(1, b * h * n, 1, m)
        self.center.running_mean = self.maximum(values).reshape(-1).float()
        self.center.running_var = torch.full_like(self.center.running_mean, 1. - self.center.eps)
        centered = self.center(values).reshape(b, h, n, m)
        fill = torch.finfo(centered.dtype).min if self.fill is None else self.fill
        probabilities = self.softmax(centered.masked_fill(~mask.bool(), fill))
        return self.matmul(probabilities.reshape(b * h, n, m), v.reshape(b * h, m, d)).reshape(b, h, n, d)


class FeedForward(nn.Module):
    def __init__(self, width, mult):
        super().__init__()
        self.norm = norm(width)
        self.fc1, self.fc2 = Linear(width, int(width * mult), bias=False), Linear(int(width * mult), width, bias=False)
        self.activation = GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(self.norm(x))))


class CompressorAttention(nn.Module):
    def __init__(self, width, heads, dim):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.norm_media, self.norm_latents = norm(width), norm(width)
        self.to_q, self.to_kv = Linear(width, heads * dim, bias=False), Linear(width, 2 * heads * dim, bias=False)
        self.to_out = Linear(heads * dim, width, bias=False)
        self.attend = CenteredAttention(-1e4)

    def forward(self, x, latents, mask):
        x, latents = self.norm_media(x), self.norm_latents(latents)
        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), 1)).chunk(2, -1)
        q, k, v = (t.reshape(t.shape[0], t.shape[1], self.heads, self.dim).transpose(1, 2) for t in (q, k, v))
        out = self.attend(q, k, v, mask[:, None, None], self.dim ** -.5)
        return self.to_out(out.transpose(1, 2).reshape(latents.shape[0], latents.shape[1], -1))


class Resampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.protein_encoder_config.hidden_size
        self.latents = nn.Parameter(torch.empty(config.resampler_num_latents, width))
        self.layers = nn.ModuleList([nn.ModuleList((CompressorAttention(width, config.resampler_heads, config.resampler_dim_head),
                                                  FeedForward(width, config.resampler_ff_mult)))
                                     for _ in range(config.resampler_depth)])
        self.protein_projector = Linear(width, config.hidden_size)
        self.norm = norm(config.hidden_size)

    def forward(self, x, mask):
        latent = self.latents[None].expand(x.shape[0], -1, -1)
        mask = torch.cat((mask, torch.ones(x.shape[0], latent.shape[1], device=mask.device, dtype=mask.dtype)), 1)
        for attention, ff in self.layers:
            latent = latent + attention(x, latent, mask)
            latent = latent + ff(latent)
        return self.norm(self.protein_projector(latent))


class ProteinLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config.hidden_size
        self.heads, self.dim = config.num_attention_heads, w // config.num_attention_heads
        self.attention = nn.Module()
        self.attention.LayerNorm = norm(w, config.layer_norm_eps)
        self.attention.self = nn.Module()
        for name in ('query', 'key', 'value'):
            setattr(self.attention.self, name, Linear(w, w))
        self.attention.output = nn.Module()
        self.attention.output.dense = Linear(w, w)
        self.LayerNorm = norm(w, config.layer_norm_eps)
        self.intermediate, self.output = nn.Module(), nn.Module()
        self.intermediate.dense = Linear(w, config.intermediate_size)
        self.output.dense = Linear(config.intermediate_size, w)
        self.activate, self.attend = PythonGELU(), DenseAttention(backend='sdpa')

    def forward(self, x, mask, table):
        b, n = x.shape[:2]
        z = self.attention.LayerNorm(x)
        q, k, v = (getattr(self.attention.self, name)(z) for name in ('query', 'key', 'value'))
        q = q * self.dim ** -.5
        positions = torch.arange(n, device=x.device).repeat(b)
        q, k = RotaryEmbedding.forward_native(positions, q.reshape(b*n, -1).float(), k.reshape(b*n, -1).float(), self.dim, table.to(x.dtype).float())
        q, k = q.to(x.dtype), k.to(x.dtype)
        q, k, v = (t.reshape(b, n, self.heads, self.dim) for t in (q, k, v))
        out = self.attend(q, k, v, softmax_scale=1., attn_mask=mask[:, None, None].bool())
        x = x + self.attention.output.dense(out.reshape(b, n, -1).contiguous())
        return x + self.output.dense(self.activate(self.intermediate.dense(self.LayerNorm(x))))


class ProteinModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.position_embedding_type != 'rotary' or config.is_decoder or config.add_cross_attention:
            raise ValueError('Evolla uses the bidirectional rotary SaProt encoder')
        self.config = config
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embeddings.layer_norm = norm(config.hidden_size, config.layer_norm_eps) if config.emb_layer_norm_before else None
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([ProteinLayer(config) for _ in range(config.num_hidden_layers)])
        self.encoder.emb_layer_norm_after = norm(config.hidden_size, config.layer_norm_eps)
        self.rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads, config.max_position_embeddings, config.rope_theta)
        self.rotary_embeddings = nn.Module()
        self.rotary_embeddings.register_buffer('inv_freq', torch.empty(config.hidden_size // config.num_attention_heads // 2))

    def forward(self, ids, mask):
        x = self.embeddings.word_embeddings(ids)
        if self.config.token_dropout:
            x = x.masked_fill((ids == self.config.mask_token_id)[..., None], 0.)
            # Scaling coefficients depend only on discrete token/padding metadata.
            ratio = (ids == self.config.mask_token_id).sum(-1).float() / mask.sum(-1)
            # Per-example metadata scalar division preserves native FP32
            # rounding. The .item() synchronization stays in measured forward;
            # this does not admit division by learned activation tensors.
            denominator = 1. - ratio
            x = torch.stack([(row * .88).float() / denominator[i].item()
                             for i, row in enumerate(x)]).to(x.dtype)
        if self.embeddings.layer_norm is not None:
            x = self.embeddings.layer_norm(x)
        x = x.masked_fill(~mask.bool()[..., None], 0.)
        for layer in self.encoder.layer:
            x = layer(x, mask, self.rotary.cos_sin_cache)
        return self.encoder.emb_layer_norm_after(x)


class ProteinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ProteinModel(config.protein_encoder_config)
        self.sequence_compressor_resampler = Resampler(config)

    def forward(self, ids, mask):
        hidden = self.model(ids, mask)
        return self.sequence_compressor_resampler(hidden, mask)


class Aligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config.hidden_size
        self.heads, self.dim = config.num_attention_heads, w // config.num_attention_heads
        self.query, self.key_protein, self.value_protein = (Linear(w, w) for _ in range(3))
        self.attention_norm = RMSNormNative(w, 1e-6)
        self.out_proj = Linear(w, w, bias=config.aligner_enable_bias)
        self.ff = FeedForward(w, config.aligner_ffn_mult)
        self.gate_attention, self.gate_ffw = (nn.Parameter(torch.empty(1)) for _ in range(2))
        self.attend, self.tanh = CenteredAttention(None), Tanh()

    def forward(self, x, protein, mask):
        b, n = x.shape[:2]
        q = self.query(self.attention_norm(x))
        k, v = self.key_protein(protein), self.value_protein(protein)
        q, k, v = (t.reshape(b, t.shape[1], self.heads, self.dim).transpose(1, 2) for t in (q, k, v))
        allowed = mask[:, None, :, None].bool().expand(b, 1, n, protein.shape[1])
        y = self.attend(q, k, v, allowed, self.heads ** -.5).transpose(1, 2).reshape(b, n, -1).contiguous()
        # Loaded scalar gates remain executed, including native zero initialization.
        x = x + self.out_proj(y) * self.tanh(self.gate_attention)
        return x + self.ff(x) * self.tanh(self.gate_ffw)


class TextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.dim = config.hidden_size // config.num_attention_heads
        w = config.hidden_size
        self.q_proj = Linear(w, self.heads * self.dim, bias=config.attention_bias)
        self.k_proj, self.v_proj = (Linear(w, self.kv_heads * self.dim, bias=config.attention_bias) for _ in range(2))
        self.o_proj = Linear(w, w, bias=config.attention_bias)
        self.attend = DenseAttention(backend='sdpa')

    def forward(self, x, mask, table):
        b, n = x.shape[:2]
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        positions = torch.arange(n, device=x.device).repeat(b)
        q, k = RotaryEmbedding.forward_native(positions, q.reshape(b*n, -1).float(), k.reshape(b*n, -1).float(), self.dim, table.to(x.dtype).float())
        q, k = q.to(x.dtype), k.to(x.dtype)
        q = q.reshape(b, n, self.heads, self.dim)
        k, v = (t.reshape(b, n, self.kv_heads, self.dim) for t in (k, v))
        groups = self.heads // self.kv_heads
        k, v = (t[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(b, n, self.heads, self.dim) for t in (k, v))
        out = self.attend(q, k, v, attn_mask=mask)
        return self.o_proj(out.reshape(b, n, -1).contiguous())


class TextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h = config.hidden_size, config.intermediate_size
        self.gate_proj, self.up_proj = (Linear(w, h, bias=config.mlp_bias) for _ in range(2))
        self.down_proj = Linear(h, w, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(SiluAndMul.forward_native(torch.cat((self.gate_proj(x), self.up_proj(x)), -1)))


class TextLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = TextAttention(config)
        self.mlp = TextMLP(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.adapter = Aligner(config) if (index + 1) % max(config.num_hidden_layers // config.aligner_num_add_layers, 1) == 0 else None

    def forward(self, x, table, causal_mask, protein, query_mask):
        x = x + self.self_attn(self.input_layernorm(x), causal_mask, table)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.adapter(x, protein, query_mask) if self.adapter is not None else x


class Evolla(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.use_cache or config.rope_parameters['rope_type'] != 'default':
            raise ValueError('The Evolla checkpoint selects uncached generation and default rotary frequencies')
        self.model = nn.Module()
        self.model.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.model.protein_encoder = ProteinEncoder(config)
        self.model.layers = nn.ModuleList([TextLayer(config, i) for i in range(config.num_hidden_layers)])
        self.model.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.model.rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads,
                                             config.max_position_embeddings, config.rope_parameters['rope_theta'])
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, protein_input_ids, protein_attention_mask, attention_mask=None, logits_to_keep=0):
        protein = self.model.protein_encoder(protein_input_ids, protein_attention_mask)
        x = self.model.embed_tokens(input_ids)
        b, n = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        positions = torch.arange(n, device=x.device)
        mask = (positions[:, None] >= positions[None, :])[None, None] & attention_mask[:, None, None].bool()
        for layer in self.model.layers:
            x = layer(x, self.model.rotary.cos_sin_cache, mask, protein, attention_mask)
        x = self.model.norm(x)
        return {'logits': self.lm_head(x[:, -logits_to_keep:] if logits_to_keep else x)}

    def generate(self, inputs, sampler, max_new_tokens, eos_token_id, pad_token_id):
        ids = inputs['input_ids'].clone()
        mask = inputs.get('attention_mask', torch.ones_like(ids)).clone()
        unfinished = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        step_logits = []
        for _ in range(max_new_tokens):
            logits = self(ids, inputs['protein_input_ids'], inputs['protein_attention_mask'], mask, logits_to_keep=1)['logits'][:, -1]
            # HF generation exposes FP32 logits before sampling processors.
            logits = logits.to(copy=True, dtype=torch.float32)
            step_logits.append(logits)
            token = sampler(logits, ids)
            token = torch.where(unfinished, token, pad_token_id)
            ids = torch.cat((ids, token[:, None]), -1)
            mask = torch.cat((mask, torch.ones_like(token[:, None])), -1)
            unfinished = unfinished & (token != eos_token_id)
            if not unfinished.any():
                break
        return {'sequences': ids, 'logits': tuple(step_logits)}


def build_from_config(config, device, dtype):
    return Evolla(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    protein = model.model.protein_encoder.model
    # HF keeps serialized non-parameter frequency buffers in their common
    # state dtype even when learned parameters are loaded as BF16.
    protein.rotary_embeddings.inv_freq = state_dict['model.protein_encoder.model.rotary_embeddings.inv_freq'].to(
        device=protein.rotary_embeddings.inv_freq.device)
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        mapped[name] = remaining.pop(source)
    model.load_state_dict(mapped, strict=True)
    # Native SaProt serializes its inverse frequencies; derive positional
    # metadata from that exact common buffer, including its load-time dtype.
    positions = torch.arange(protein.config.max_position_embeddings, device=protein.rotary_embeddings.inv_freq.device, dtype=torch.float32)
    angles = positions[:, None] * protein.rotary_embeddings.inv_freq.float()[None]
    protein.rotary.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), -1).to(protein.rotary_embeddings.inv_freq.dtype)
    if remaining:
        raise KeyError(f'Unmapped Evolla states: {sorted(remaining)}')


def make_workloads(model, inputs, config, case=None):
    if case is not None and case['workload'] == 'generate':
        generation = case['reference']['generation_config']
        if not generation['do_sample'] or generation['use_cache'] or generation.get('num_beams', 1) != 1:
            raise ValueError('The pinned Evolla task samples without KV caching or beam search')
        sampler = OmniSampling(generation.get('top_k', 50), generation['top_p'], generation['temperature'], generation.get('repetition_penalty', 1.))
        def generate():
            output = model.generate(inputs, sampler, case['generation_kwargs']['max_new_tokens'],
                                    generation['eos_token_id'], generation['pad_token_id'])
            return {'sequences': output['sequences'],
                    **{f'logits.{i}': logits for i, logits in enumerate(output['logits'])}}
        return {'generate': Workload(run=generate)}
    return {'forward': Workload(run=lambda: model(**inputs))}
