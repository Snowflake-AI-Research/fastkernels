"""BLT's ordinary byte-to-logit forward, including predicted entropy patches.

The selected public forward has no cache by default. This construction covers
one unpadded byte sequence; cached generation is a separate pinned-HF limitation.
"""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.pointtransformerv3_offsets import Offset2Batch
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.swiglu_mlp import SwiGLUMlp

from ..patches.categorical_log_softmax import CategoricalLogSoftmax
from ..patches.codec_top1 import CodecTop1
from ..patches.hashed_embedding import HashedEmbedding
from ..patches.integer_moe_sum import IntegerMoeSum
from ..patches.product_gate import ProductGate
from ..runner import Workload


def _norm(config):
    return RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)


class Attention(nn.Module):
    def __init__(self, config, cross=False):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        self.q_proj = Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.attention = DenseAttention(backend='cudnn')
        self.cross = cross
        if cross:
            self.q_norm, self.k_norm = _norm(config), _norm(config)

    def forward(self, hidden, rope=None, source=None, mask=None):
        query_input = self.q_norm(hidden) if self.cross else hidden
        source = self.k_norm(source) if self.cross else hidden
        q = self.q_proj(query_input)
        k, v = self.k_proj(source), self.v_proj(source)
        if rope is not None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            # Reuse the parent's native interleaved backend to preserve HF's
            # two rounded products, rather than a fused multiply-add rotation.
            q, k = rope.forward_native_interleaved(
                positions, q[0], k[0], self.head_dim,
                rope.cos_sin_cache.to(q.dtype),
            )
            q, k = q.unsqueeze(0), k.unsqueeze(0)
        q = q.view(1, -1, self.heads, self.head_dim)
        k = k.view(1, -1, self.kv_heads, self.head_dim)
        v = v.view(1, -1, self.kv_heads, self.head_dim)
        if self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=2)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=2)
        out = self.attention(q, k, v, causal=not self.cross, attn_mask=mask)
        out = self.o_proj(out.reshape(1, hidden.shape[1], -1))
        return out + hidden if self.cross else out


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = _norm(config)
        self.post_attention_layernorm = _norm(config)
        self.mlp = SwiGLUMlp(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, hidden, rope):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), rope=rope)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Stack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        # The selected HF loader constructs inverse frequencies on CPU in FP32.
        # CUDA's power calculation changes a few rounded position-table entries.
        with torch.device('cpu'):
            self.rotary_emb = RotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                config.max_position_embeddings, config.rope_parameters['rope_theta'],
                is_neox_style=False,
            )

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden, self.rotary_emb)
        return hidden


class EntropyPatcher(Stack):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.norm = _norm(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.log_normalize = CategoricalLogSoftmax()
        self.softmax, self.product = Softmax(), ProductGate()
        self.reduce, self.threshold = SegmentCSR(), CodecTop1()

    def forward(self, ids, threshold):
        logits = self.lm_head(self.norm(super().forward(self.embed_tokens(ids))))
        return self.patch_starts(logits, threshold)

    def patch_starts(self, logits, threshold):
        normalized, clamped = self.log_normalize(logits)
        terms = self.product(torch.cat((clamped, self.softmax(normalized)), dim=-1))
        offsets = torch.arange(0, terms.numel() + 1, terms.shape[-1], device=logits.device)
        # Native BF16 sum accumulates in FP32 and rounds once at the output.
        entropies = -self.reduce(terms.float().flatten(), offsets).to(terms.dtype).view(logits.shape[:-1])
        # First-index tie breaking implements native strict greater-than.
        scores = torch.stack((torch.full_like(entropies, threshold), entropies), -1)
        selected = self.threshold(scores)[0, 1:].bool()
        starts = torch.cat((torch.tensor([0, 1], device=logits.device), selected.nonzero().flatten() + 2))
        return starts


class ByteHashes(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.groups = tuple(config.encoder_hash_byte_group_size)
        self.coefficients = Embedding(config.vocab_size, max(self.groups))
        self.coefficients.emb.weight = nn.Parameter(
            torch.empty(config.vocab_size, max(self.groups), dtype=torch.int64),
            requires_grad=False,
        )
        self.lookups = nn.ModuleList(
            HashedEmbedding(config.encoder_hash_byte_group_vocab, config.encoder_config.hidden_size, prime=-1)
            for _ in self.groups
        )
        self.reduce = IntegerMoeSum()

    def forward(self, ids, hidden):
        terms = self.coefficients(ids)
        for group, lookup in zip(self.groups, self.lookups):
            padded = torch.nn.functional.pad(terms, (0, 0, group - 1, 0))
            columns = torch.arange(group, device=ids.device)
            windows = padded.unfold(1, group, 1)[:, :, columns, columns]
            hashes = self.reduce(windows.reshape(-1, 1), group).view_as(ids)
            hidden = hidden + lookup(hashes)
        return hidden


class LocalEncoder(Stack):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.patch_embedding_projection = Linear(
            config.hidden_size, config.hidden_size * config.cross_attn_k, bias=False,
        )
        count = config.num_hidden_layers if config.cross_attn_all_layers else 1
        self.cross_attn_layers = nn.ModuleList(Attention(config, cross=True) for _ in range(count))
        self.reduce = SegmentCSR()

    def forward(self, hidden, offsets, cross_mask):
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, self.rotary_emb)
            if index == len(self.layers) - 1 or self.config.cross_attn_all_layers:
                pooled = self.reduce(hidden[0], offsets, reduce='max')
                # Empty segments are known from routing offsets, not activations.
                # Native scatter_reduce leaves their zero initialization intact.
                pooled[offsets[1:] == offsets[:-1]] = 0
                patches = self.patch_embedding_projection(pooled).view(1, -1, self.config.hidden_size)
                cross = self.cross_attn_layers[index if self.config.cross_attn_all_layers else 0]
                # HF cross-attention adds a residual internally and again here.
                patches = patches + cross(patches, source=hidden, mask=cross_mask)
        return hidden, patches


class LocalDecoder(Stack):
    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding_projection = Linear(
            config.hidden_size_global, config.hidden_size * config.cross_attn_k, bias=False,
        )
        count = config.num_hidden_layers if config.cross_attn_all_layers else 1
        self.cross_attn_layers = nn.ModuleList(Attention(config, cross=True) for _ in range(count))
        self.norm = _norm(config)

    def forward(self, hidden, patches, cross_mask):
        patches = self.patch_embedding_projection(patches).view(1, -1, self.config.hidden_size)
        for index, layer in enumerate(self.layers):
            if index == 0 or self.config.cross_attn_all_layers:
                hidden = hidden + self.cross_attn_layers[index](hidden, source=patches, mask=cross_mask)
            hidden = layer(hidden, self.rotary_emb)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_encoder = LocalEncoder(config.encoder_config)
        self.global_transformer = Stack(config.global_config)
        self.local_decoder = LocalDecoder(config.decoder_config)
        self.patcher = EntropyPatcher(config.patcher_config)
        self.hashes = ByteHashes(config)
        self.lm_head = Linear(config.decoder_config.hidden_size, config.vocab_size, bias=False)
        self.offset_to_batch = Offset2Batch()

    def _cross_mask(self, patch_ids, count, dtype, encoder):
        permitted = patch_ids[:, None] == torch.arange(count, device=patch_ids.device)[None, :]
        permitted = permitted.repeat_interleave(self.config.cross_attn_k, dim=1)
        if encoder:
            permitted = permitted.T
        mask = torch.zeros(permitted.shape, device=patch_ids.device, dtype=dtype)
        return mask.masked_fill(~permitted, torch.finfo(dtype).min)[None, None]

    def forward(self, input_ids):
        if input_ids.shape[0] != 1:
            raise ValueError('BLT coverage uses one unpadded byte sequence')
        length = input_ids.shape[1]
        starts = self.patcher(input_ids, self.config.patching_threshold)
        offsets = torch.cat((starts, input_ids.new_tensor([length])))
        patch_ids = self.offset_to_batch(offsets[1:])
        hidden = self.hashes(input_ids, self.local_encoder.embed_tokens(input_ids))
        enc_mask = self._cross_mask(patch_ids, starts.numel(), hidden.dtype, encoder=True)
        hidden, patches = self.local_encoder(hidden, offsets, enc_mask)
        patches = self.global_transformer(patches.reshape(1, starts.numel(), -1))
        # Removing the first one-byte patch shifts the remaining starts by one.
        # Native decoder routing assigns the final extra position to its last patch.
        decoder_ends = torch.cat((starts[2:] - 1, input_ids.new_tensor([length])))
        decoder_ids = self.offset_to_batch(decoder_ends)
        dec_mask = self._cross_mask(decoder_ids, starts.numel(), hidden.dtype, encoder=False)
        return self.lm_head(self.local_decoder(hidden, patches, dec_mask)).float()


def build_from_config(config, device, dtype):
    if (not config.patch_in_forward or config.patching_mode != 'entropy'
            or config.max_patch_length is not None or config.encoder_hash_byte_group_nb_functions != 1
            or getattr(config, 'use_cache', None) or config.tie_word_embeddings
            or getattr(config.global_config, 'encoder_cross_output_size', None) is not None):
        raise ValueError('BLT construction requires the selected entropy-patching forward configuration')
    for part in (config.encoder_config, config.global_config, config.decoder_config, config.patcher_config):
        if part.hidden_act != 'silu' or part.rope_parameters['rope_type'] != 'default':
            raise ValueError('Selected BLT uses SiLU and ordinary interleaved RoPE')
    # HF from_pretrained keeps inverse frequencies FP32. The existing RoPE
    # constructor likewise computes its position table before dtype conversion.
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {}
    for name in model.state_dict():
        if name.startswith('hashes.'):
            continue
        source = name.replace('.emb.weight', '.weight')
        source = source.replace('.mlp.fc1_g.', '.mlp.gate_proj.').replace('.mlp.fc1_x.', '.mlp.up_proj.')
        source = source.replace('.mlp.fc2.', '.mlp.down_proj.')
        if not source.startswith('lm_head.'):
            source = 'model.' + source
        mapped[name] = remaining.pop(source)
    model.load_state_dict(mapped, strict=False)
    # Constant polynomial coefficients and a same-size hash-table permutation.
    # The prime=-1 gather maps h to (-h-1)%B exactly even at int64 endpoints.
    coefficients = [[((token * pow(1000000007, power)) + 2**63) % 2**64 - 2**63
                     for power in range(max(model.hashes.groups))] for token in range(config.vocab_size)]
    model.hashes.coefficients.emb.weight.copy_(torch.tensor(coefficients, dtype=torch.int64))
    weights = remaining.pop('model.encoder_hash_tok_embedding.weight')
    buckets = config.encoder_hash_byte_group_vocab
    for group, lookup in enumerate(model.hashes.lookups):
        for start in range(0, buckets, 4096):
            end = min(start + 4096, buckets)
            rows = (-torch.arange(start, end, device=weights.device) - 1) % buckets
            lookup.emb.weight[start:end].copy_(weights[group * buckets + rows])
    if remaining:
        raise KeyError(f'Unmapped BLT weights: {sorted(remaining)}')


def make_workloads(model, inputs, config, case=None):
    return {'forward': Workload(run=lambda: {'logits': model(**inputs)})}
