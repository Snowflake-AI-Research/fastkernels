"""PE audio/text embedding composition, retaining default encoder outputs."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP
from .dac import CodecStack
from .modernbert import ModernLayer, normalization
from ..runner import Workload


class PaddedGroupNorm(GroupNorm):
    """Gather channel-shared valid positions for unchanged GroupNorm."""
    def forward(self, x, padding_mask=None):
        if padding_mask is None:
            return super().forward(x)
        valid = padding_mask[:, 0].bool()
        if not torch.equal(padding_mask.bool(), valid[:, None].expand_as(padding_mask)):
            raise ValueError("PE group normalization requires channel-shared padding metadata")
        if not valid.any(-1).all():
            raise ValueError("All-masked PE group normalization is unsupported; native returns NaNs")
        output = torch.zeros_like(x)
        for i in range(x.shape[0]):
            output[i:i+1, :, valid[i]] = super().forward(x[i:i+1, :, valid[i]])
        return output


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embedding_norm = normalization(config)
        self.layers = nn.ModuleList([ModernLayer(config, i, False) for i in range(config.num_hidden_layers)])
        self.final_norm = normalization(config)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding_norm(self.embeddings(input_ids))
        batch, length = input_ids.shape
        positions = torch.arange(length, device=hidden.device)
        states = [hidden]
        for layer in self.layers:
            q, k, v = layer.qkv(layer.attn_norm(hidden)).chunk(3, -1)
            index = positions.expand(batch, -1).reshape(-1)
            q, k = RotaryEmbedding.forward_native(index, q.reshape(batch * length, -1).float(),
                                                  k.reshape(batch * length, -1).float(), layer.width,
                                                  layer.rotary.cos_sin_cache.to(q.dtype).float())
            shape = (batch, length, layer.heads, layer.width)
            q, k, v = q.to(hidden.dtype).reshape(shape), k.to(hidden.dtype).reshape(shape), v.reshape(shape)
            mask = torch.ones(length, length, device=hidden.device, dtype=torch.bool)
            if layer.window is not None:
                mask &= (positions[:, None] - positions[None]).abs() <= layer.window
            mask = mask[None, None]
            if attention_mask is not None:
                mask = mask & attention_mask[:, None, None, :].bool()
            hidden = hidden + layer.out_proj(layer.attention(q, k, v, attn_mask=mask).reshape(batch, length, -1))
            mlp = layer.mlp
            hidden = hidden + mlp.down_proj(mlp.act_fn.forward_native(mlp.gate_up_proj(layer.mlp_norm(hidden))))
            states.append(hidden)
        hidden = self.final_norm(hidden)
        states[-1] = hidden
        return {"last_hidden_state": hidden, "hidden_states": tuple(states)}


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_norm = LayerNorm(in_dim, eps=1e-6, promote_fp32=False)
        self.proj = Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(self.layer_norm(x))


class ConvBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.groupnorm = PaddedGroupNorm(1, width, eps=1e-5)
        self.activation = SiLU()
        self.project = Conv1dNative(width, width, 3, padding=1)

    def forward(self, x, mask):
        return self.project(self.activation(self.groupnorm(x, mask)))


class PatchEmbedder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.empty(1, 1, width))
        self.resnet_block = nn.Module()
        self.resnet_block.block1, self.resnet_block.block2 = ConvBlock(width), ConvBlock(width)

    def forward(self, x, mask):
        x = torch.cat((self.class_embedding.expand(x.shape[0], -1, -1), x), 1).transpose(1, 2)
        if mask is not None:
            mask = torch.cat((mask[:, :1], mask), 1)
        conv_mask = None if mask is None else mask[:, None].expand_as(x)
        x = x + self.resnet_block.block2(self.resnet_block.block1(x, conv_mask), conv_mask)
        return x.transpose(1, 2), mask


class TemporalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = Linear(config.hidden_size, self.heads * self.dim, bias=config.attention_bias)
        self.k_proj, self.v_proj = (Linear(config.hidden_size, self.kv_heads * self.dim, bias=config.attention_bias) for _ in range(2))
        self.o_proj = Linear(self.heads * self.dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNormNative(self.dim, config.rms_norm_eps)
        self.k_norm = RMSNormNative(self.dim, config.rms_norm_eps)
        self.attend = DenseAttention(backend="sdpa")

    def forward(self, x, table, mask):
        batch, length = x.shape[:2]
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self.q_norm(q.reshape(batch, length, self.heads, self.dim))
        k = self.k_norm(k.reshape(batch, length, self.kv_heads, self.dim))
        positions = torch.arange(length, device=x.device).repeat(batch)
        q, k = RotaryEmbedding.forward_native_interleaved(positions, q.reshape(batch * length, -1),
                                                          k.reshape(batch * length, -1), self.dim, table.to(x.dtype))
        q = q.reshape(batch, length, self.heads, self.dim)
        k, v = (t.reshape(batch, length, self.kv_heads, self.dim) for t in (k, v))
        groups = self.heads // self.kv_heads
        k, v = (t[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(batch, length, self.heads, self.dim) for t in (k, v))
        mask = None if mask is None else mask[:, None, None, :].bool()
        return self.o_proj(self.attend(q, k, v, attn_mask=mask).reshape(batch, length, -1).contiguous())


class TemporalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = TemporalAttention(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x, table, mask):
        x = x + self.self_attn(self.input_layernorm(x), table, mask)
        packed = self.mlp.gate_up_proj(self.post_attention_layernorm(x))
        return x + self.mlp.down_proj(SiluAndMul.forward_native(packed))


class AudioEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dac_encoder = CodecStack(config.dac_config)
        self.bottleneck = Conv1dNative(config.dac_config.hidden_size, config.dac_config.codebook_dim, 1)
        self.data_proj = Linear(config.dac_config.codebook_dim, config.hidden_size)
        self.hop = config.dac_config.hop_length

    def forward(self, input_values, padding_mask=None):
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            hidden = self.bottleneck(self.dac_encoder(input_values))
        return self.data_proj(hidden.transpose(1, 2)), None if padding_mask is None else padding_mask[:, ::self.hop]


class TemporalEncoder(nn.Module):
    def __init__(self, config, embedder):
        super().__init__()
        self.embedder = embedder
        self.patch_embedder = PatchEmbedder(config.hidden_size)
        self.layers = nn.ModuleList([TemporalLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.output = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_parameters['rope_theta'])

    def forward(self, values, padding_mask=None):
        x, output_mask = self.embedder(values, padding_mask)
        x, mask = self.patch_embedder(x, output_mask)
        for layer in self.layers:
            x = layer(x, self.rotary.cos_sin_cache, mask)
        x = self.output(self.norm(x))
        return {"last_hidden_state": x[:, 1:], "pooler_output": x[:, 0], "output_mask": output_mask}


class PeAudio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = TextEncoder(config.text_config)
        self.audio_encoder = TemporalEncoder(config.audio_config, AudioEmbedder(config.audio_config))
        self.text_audio_head = ContrastiveHead(config.text_config.hidden_size, config.text_config.hidden_size)
        self.audio_head = ContrastiveHead(config.audio_config.hidden_size, config.text_config.hidden_size)
        self.text_audio_logit_scale = nn.Parameter(torch.empty(1))
        self.text_audio_logit_bias = nn.Parameter(torch.empty(1))
        self.matmul = Matmul()

    def forward(self, input_ids, input_values, attention_mask=None, padding_mask=None):
        audio = self.audio_encoder(input_values, padding_mask)
        text = self.text_model(input_ids, attention_mask)
        a = self.audio_head(audio['pooler_output'])
        t = self.text_audio_head(text['hidden_states'][-1][:, 0])
        logits = self.matmul(a, t) * self.text_audio_logit_scale + self.text_audio_logit_bias
        return {"logits_audio_text": logits, "text_audio_embeds": t, "audio_embeds": a,
                "text_outputs": text, "audio_outputs": audio}


def build_from_config(config, device, dtype):
    return PeAudio(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name, target in model.state_dict().items():
        source = name.replace('.conv.weight', '.weight').replace('.conv.bias', '.bias')
        if source.startswith('text_model.') or '.text_model.' in source:
            source = source.replace('embeddings.emb.', 'embeddings.tok_embeddings.').replace('embedding_norm.', 'embeddings.norm.')
            for a, b in [('qkv.', 'attn.Wqkv.'), ('out_proj.', 'attn.Wo.'), ('mlp.gate_up_proj.', 'mlp.Wi.'), ('mlp.down_proj.', 'mlp.Wo.')]:
                source = source.replace(a, b)
        if '.mlp.gate_up_proj.' in source:
            value = torch.cat([remaining.pop(source.replace('gate_up_proj', part)) for part in ('gate_proj', 'up_proj')])
        else:
            value = remaining.pop(source)
        mapped[name] = value.reshape_as(target) if source.endswith('.alpha') else value
    model.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f"Unmapped PE audio state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: flatten_output(model(**inputs)))}


def flatten_output(output):
    flattened = {}
    def visit(name, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{name}.{key}" if name else key, child)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(f"{name}.{index}", child)
        elif value is not None:
            flattened[name] = value
    visit("", output)
    return flattened
