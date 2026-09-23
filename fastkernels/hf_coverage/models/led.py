"""LED summarization with local/global encoder attention and the existing cached decoder."""

from types import SimpleNamespace
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from .bart import BartForConditionalGeneration
from .longformer import WindowAttention
from .mvp import EagerAttention
from .plbart import make_workloads as seq2seq_workloads


class EncoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        carrier = SimpleNamespace(hidden_size=config.d_model, intermediate_size=config.encoder_ffn_dim,
                                  layer_norm_eps=1e-5)
        self.attn = WindowAttention(config.d_model, config.encoder_attention_heads,
                                    config.attention_window[index], global_first=True)
        self.attn_output = EncoderSelfOutput(carrier)
        self.intermediate, self.output = EncoderIntermediate(carrier), EncoderOutput(carrier)

    def forward(self, hidden, valid_length):
        hidden = self.attn_output(self.attn(hidden, valid_length), hidden)
        return self.output(self.intermediate(hidden), hidden)


class Encoder(nn.Module):
    def __init__(self, config, shared):
        super().__init__()
        self.pad_id, self.window, self.position_offset = config.pad_token_id, max(config.attention_window), 0
        self.embed_tokens = shared
        self.embed_positions = Embedding(config.max_encoder_position_embeddings, config.d_model)
        self.layernorm_embedding = LayerNorm(config.d_model, eps=1e-5, promote_fp32=False)
        self.layers = nn.ModuleList([EncoderLayer(config, index) for index in range(config.encoder_layers)])

    def forward(self, ids, positions):
        length = ids.shape[1]
        padding = (-length) % self.window
        ids = torch.cat((ids, ids.new_full((ids.shape[0], padding), self.pad_id)), dim=1)
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.layernorm_embedding(self.embed_tokens(ids) + self.embed_positions(positions))
        for layer in self.layers:
            hidden = layer(hidden, length)
        return hidden[:, :length]


def build_from_config(config, device, dtype):
    if config.activation_function != 'gelu' or not config.use_cache or not config.tie_word_embeddings:
        raise ValueError('Selected LED summarization uses GELU, tied embeddings and decoder caching')
    adapted = SimpleNamespace(**(config.to_dict() if hasattr(config, 'to_dict') else dict(config)))
    adapted.max_position_embeddings = config.max_encoder_position_embeddings
    adapted.scale_embedding = False
    model = BartForConditionalGeneration(adapted)
    model.encoder = Encoder(config, model.shared)
    model.decoder.embed_positions = Embedding(config.max_decoder_position_embeddings, config.d_model)
    model.decoder.position_offset = 0
    for layer in model.decoder.layers:
        layer.attention.self.attn = EagerAttention(prescale_query=True)
        layer.cross_attention.attention = EagerAttention(prescale_query=True)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    for name in ('led.encoder.embed_tokens.weight', 'led.decoder.embed_tokens.weight', 'lm_head.weight'):
        if not torch.equal(remaining[name], remaining['led.shared.weight']):
            raise ValueError(f'LED tied embeddings disagree: {name}')
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if name.startswith(('shared.', 'encoder.', 'decoder.')):
            source = 'led.' + source
        if name.startswith('encoder.layers.'):
            source = source.replace('.attn.', '.self_attn.longformer_self_attn.')
            source = source.replace('.attn_output.dense.', '.self_attn.output.')
            source = source.replace('.attn_output.LayerNorm.', '.self_attn_layer_norm.')
        if name.startswith('decoder.layers.'):
            if '.attention.self.qkv.' in name:
                weights[name] = torch.cat([remaining.pop(source.replace('.attention.self.qkv.', f'.self_attn.{part}_proj.'))
                                           for part in ('q', 'k', 'v')])
                continue
            source = source.replace('.attention.output.dense.', '.self_attn.out_proj.')
            source = source.replace('.attention.output.LayerNorm.', '.self_attn_layer_norm.')
            source = source.replace('.cross_attention.norm.', '.encoder_attn_layer_norm.')
            source = source.replace('.cross_attention.', '.encoder_attn.')
        source = source.replace('.intermediate.dense.', '.fc1.').replace('.output.dense.', '.fc2.')
        source = source.replace('.output.LayerNorm.', '.final_layer_norm.')
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped LED parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    expected = torch.zeros_like(inputs['input_ids'])
    expected[:, 0] = 1
    if not torch.equal(inputs['global_attention_mask'], expected):
        raise ValueError('This selected summarization case uses exactly the first token as global')
    return seq2seq_workloads(model, inputs, config, case=case)
