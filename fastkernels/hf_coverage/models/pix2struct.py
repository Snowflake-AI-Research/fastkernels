"""Pix2Struct patch-grid encoder and T5-style gated decoder compositions."""

from types import SimpleNamespace
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from fastkernels.tasks.baseline.L2.t5_dense import T5DenseGatedActDense
from .t5 import T5CrossAttention, T5DecoderBlock
from ..patches.finite_floor_softmax import FiniteFloorSoftmax
from ..runner import Workload


def t5_config(config, *, vision=False):
    return SimpleNamespace(d_model=config.hidden_size, d_ff=config.d_ff, d_kv=config.d_kv,
        num_heads=config.num_attention_heads if vision else config.num_heads,
        dense_act_fn=config.dense_act_fn, is_gated_act=True,
        layer_norm_epsilon=config.layer_norm_eps if vision else config.layer_norm_epsilon,
        relative_attention_num_buckets=config.relative_attention_num_buckets,
        relative_attention_max_distance=config.relative_attention_max_distance,
        is_decoder=not vision)


class VisionAttention(T5CrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.softmax = FiniteFloorSoftmax(dim=-1)

    def forward(self, hidden, mask):
        batch, length = hidden.shape[:2]
        query, key, value = (projection(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
                             for projection in (self.q, self.k, self.v))
        bias = (1 - mask[:, None, None].to(hidden.dtype)).expand(batch, self.heads, length, length)
        bias = bias.masked_fill(bias == 1, torch.finfo(hidden.dtype).min)
        probabilities = self.softmax(self.bmm(query, key.transpose(-1, -2)) + bias)
        return self.o(self.bmm(probabilities, value).transpose(1, 2).reshape(batch, length, -1))


class VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = VisionAttention(config)
        self.pre_attention_layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.pre_mlp_layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.mlp = T5DenseGatedActDense(config)

    def forward(self, hidden, mask):
        hidden = self.attention(self.pre_attention_layer_norm(hidden), mask) + hidden
        return self.mlp(self.pre_mlp_layer_norm(hidden)) + hidden


class Pix2StructForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision, text = config.vision_config, config.text_config
        self.patch_projection = Linear(vision.patch_embed_hidden_size, vision.hidden_size)
        self.row_embedder, self.column_embedder = Embedding(vision.seq_len, vision.hidden_size), Embedding(vision.seq_len, vision.hidden_size)
        self.vision_layers = nn.ModuleList([VisionLayer(t5_config(vision, vision=True)) for _ in range(vision.num_hidden_layers)])
        self.vision_norm = T5LayerNorm(vision.hidden_size, vision.layer_norm_eps)
        self.embed_tokens = Embedding(text.vocab_size, text.hidden_size)
        self.decoder = nn.ModuleList([T5DecoderBlock(t5_config(text), index == 0) for index in range(text.num_layers)])
        self.decoder_norm = T5LayerNorm(text.hidden_size, text.layer_norm_epsilon)
        self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=False)
        if text.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.emb.weight

    def forward(self, flattened_patches, attention_mask, decoder_input_ids, buckets, causal_mask):
        rows, columns = flattened_patches[..., 0].long(), flattened_patches[..., 1].long()
        memory = self.patch_projection(flattened_patches[..., 2:]) + self.row_embedder(rows) + self.column_embedder(columns)
        for layer in self.vision_layers:
            memory = layer(memory, attention_mask)
        memory = self.vision_norm(memory)
        hidden = self.embed_tokens(decoder_input_ids)
        values = self.decoder[0].self_attention.relative_attention_bias(buckets)
        self_bias = values.permute(2, 0, 1)[None] + causal_mask
        cross_bias = torch.zeros((attention_mask.shape[0], 1, 1, attention_mask.shape[1]), device=hidden.device, dtype=hidden.dtype)
        cross_bias = cross_bias.masked_fill(~attention_mask[:, None, None].bool(), torch.finfo(hidden.dtype).min)
        for layer in self.decoder:
            hidden, _ = layer(hidden, memory, self_bias, cross_bias)
        return {'logits': self.lm_head(self.decoder_norm(hidden)), 'encoder_last_hidden_state': memory}


def build_from_config(config, device, dtype):
    if (config.text_config.use_cache or config.text_config.dense_act_fn != 'gelu_new'
            or config.vision_config.dense_act_fn != 'gelu_new'
            or config.text_config.hidden_size != config.vision_config.hidden_size):
        raise ValueError('Pix2Struct case preserves gated GELU, matching tower widths and default disabled cache')
    return Pix2StructForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, parameter in model.state_dict().items():
        source = name.replace('.emb.weight', '.weight')
        if source.startswith(('patch_projection.', 'row_embedder.', 'column_embedder.')):
            source = 'encoder.embeddings.' + source
        elif source.startswith('vision_norm.'):
            source = source.replace('vision_norm.', 'encoder.layernorm.', 1)
        elif source.startswith('vision_layers.'):
            source = source.replace('vision_layers.', 'encoder.encoder.layer.', 1)
            for target, native in (('q', 'query'), ('k', 'key'), ('v', 'value'), ('o', 'output')):
                source = source.replace('.attention.' + target + '.', '.attention.' + native + '.')
        elif source.startswith(('embed_tokens.', 'lm_head.')):
            source = 'decoder.' + source
        elif source.startswith('decoder_norm.'):
            source = source.replace('decoder_norm.', 'decoder.final_layer_norm.', 1)
        elif source.startswith('decoder.'):
            source = source.replace('decoder.', 'decoder.layer.', 1)
            for target, native in (('self_attention', 'self_attention.attention'), ('self_norm', 'self_attention.layer_norm'),
                    ('cross_attention', 'encoder_decoder_attention.attention'), ('cross_norm', 'encoder_decoder_attention.layer_norm'), ('ff', 'mlp')):
                source = source.replace('.' + target + '.', '.' + native + '.')
            for target, native in (('q', 'query'), ('k', 'key'), ('v', 'value'), ('o', 'output')):
                source = source.replace('.attention.' + target + '.', '.attention.' + native + '.')
        if source.endswith('.qkv_proj.weight'):
            sources = [source.replace('qkv_proj.weight', native + '.weight') for native in ('query', 'key', 'value')]
        elif source.endswith('.wi.weight'):
            sources = [source.replace('wi.weight', native + '.weight') for native in ('wi_0', 'wi_1')]
        else:
            sources = [source]
        mapped[name] = torch.cat([state_dict[s] for s in sources]) if len(sources) > 1 else state_dict[source]
        used.update(sources)
        if mapped[name].shape != parameter.shape:
            raise ValueError(f'Pix2Struct state shape mismatch for {name}')
    if used != set(state_dict):
        raise KeyError(f'Unmapped Pix2Struct weights: {sorted(set(state_dict) - used)}')
    if config.text_config.tie_word_embeddings and not torch.equal(mapped['lm_head.weight'], mapped['embed_tokens.emb.weight']):
        raise ValueError('Pix2Struct tied decoder weights disagree')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    text = config.text_config
    length = inputs['decoder_input_ids'].shape[1]
    positions = torch.arange(length, device=inputs['decoder_input_ids'].device)
    buckets = T5SelfAttention._relative_position_bucket(positions[None] - positions[:, None], bidirectional=False,
        num_buckets=text.relative_attention_num_buckets, max_distance=text.relative_attention_max_distance)
    mask = torch.zeros((1, 1, length, length), device=positions.device, dtype=next(model.parameters()).dtype)
    mask.masked_fill_(positions[None] > positions[:, None], torch.finfo(mask.dtype).min)
    return {'forward': Workload(run=lambda: model(**inputs, buckets=buckets, causal_mask=mask))}
