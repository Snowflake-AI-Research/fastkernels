"""Moonshine waveform frontend, partial interleaved RoPE, and cached seq2seq outputs."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.conv1d import Conv1d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.whisper_mlp import WhisperMLP
from fastkernels.tasks.baseline.L3.siglip_encoder_layer import SigLIPEncoderLayer
from fastkernels.tasks.baseline.L3.whisper_decoder_layer import WhisperDecoderLayer
from fastkernels.tasks.baseline.L4.qwen2_5_omni import Qwen2_5VisionMLP


def norm(width):
    return LayerNorm(width, create_offset=False, promote_fp32=False)


class MoonshineAttention(nn.Module):
    def __init__(self, config, heads, *, rotary=None, causal=False, cache=False):
        super().__init__()
        width = config.hidden_size
        self.heads, self.head_dim = heads, width // heads
        self.rotary, self.causal, self.cache = rotary, causal, cache
        multiple = config.pad_head_dim_to_multiple_of
        self.padding = (-self.head_dim) % multiple if multiple else 0
        self.q_proj = Linear(width, width, bias=False)
        self.k_proj = Linear(width, width, bias=False)
        self.v_proj = Linear(width, width, bias=False)
        self.o_proj = Linear(width, width, bias=False)
        self.attention = DenseAttention(backend="cudnn")
        self.last_cache = None

    def forward(self, hidden, memory=None, past_key_value=None):
        source = hidden if memory is None else memory
        batch, length = hidden.shape[:2]
        query = self.q_proj(hidden).view(batch, length, self.heads, self.head_dim)
        if memory is not None and past_key_value is not None:
            key, value = (tensor.transpose(1, 2) for tensor in past_key_value)
        else:
            key, value = (
                projection(source).view(batch, -1, self.heads, self.head_dim)
                for projection in (self.k_proj, self.v_proj)
            )
        if self.rotary is not None:
            width = self.rotary.head_dim
            offset = 0 if past_key_value is None else past_key_value[0].shape[2]
            positions = (torch.arange(length, device=hidden.device) + offset).repeat(batch)
            # The existing native callable retains HF's separately rounded
            # BF16 products; the fused CUDA variant rounds only at the end.
            q_rot, k_rot = self.rotary.forward_native_interleaved(
                positions, query[..., :width].contiguous().view(-1, self.heads, width),
                key[..., :width].contiguous().view(-1, self.heads, width),
                width, self.rotary.cos_sin_cache.to(query.dtype),
            )
            query = torch.cat((q_rot.view(batch, length, self.heads, width), query[..., width:]), -1)
            key = torch.cat((k_rot.view(batch, length, self.heads, width), key[..., width:]), -1)
        if self.cache:
            key, value = (tensor.transpose(1, 2) for tensor in (key, value))
            if past_key_value is None:
                key, value = (tensor.contiguous().clone() for tensor in (key, value))
            elif memory is None:
                if length != 1:
                    raise ValueError("Moonshine continuation evaluates one new token per call")
                key, value = (torch.cat((old, new), dim=2)
                              for old, new in zip(past_key_value, (key, value)))
            self.last_cache = (key, value)
            key, value = key.transpose(1, 2), value.transpose(1, 2)
        if self.padding:
            query, key, value = (
                torch.cat((tensor, tensor.new_zeros(*tensor.shape[:-1], self.padding)), -1)
                for tensor in (query, key, value)
            )
        context = self.attention(query, key, value, causal=self.causal and past_key_value is None,
                                 softmax_scale=self.head_dim**-0.5)
        return self.o_proj(context[..., :self.head_dim].reshape(batch, length, -1))


class EncoderLayer(SigLIPEncoderLayer):
    def __init__(self, config, rotary):
        nn.Module.__init__(self)
        self.layer_norm1, self.layer_norm2 = norm(config.hidden_size), norm(config.hidden_size)
        self.self_attn = MoonshineAttention(config, config.encoder_num_attention_heads, rotary=rotary)
        self.mlp = WhisperMLP(config.hidden_size, config.intermediate_size)


class DecoderLayer(WhisperDecoderLayer):
    def __init__(self, config, rotary):
        nn.Module.__init__(self)
        self.self_attn_layer_norm = norm(config.hidden_size)
        self.encoder_attn_layer_norm = norm(config.hidden_size)
        self.final_layer_norm = norm(config.hidden_size)
        self.self_attn = MoonshineAttention(config, config.decoder_num_attention_heads,
                                           rotary=rotary, causal=True, cache=True)
        self.encoder_attn = MoonshineAttention(config, config.decoder_num_attention_heads, cache=True)
        self.mlp = Qwen2_5VisionMLP(config.hidden_size, config.intermediate_size)


class MoonshineForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.conv1 = Conv1d(1, width, 127, stride=64, bias=False)
        self.conv2 = Conv1d(width, 2 * width, 7, stride=3)
        self.conv3 = Conv1d(2 * width, width, 3, stride=2)
        self.groupnorm = GroupNorm(1, width, eps=1e-5)
        self.tanh, self.gelu = Tanh(), GELU()
        rope = config.rope_parameters
        rotary_dim = int((width // config.encoder_num_attention_heads) * rope['partial_rotary_factor'])
        self.rotary = RotaryEmbedding(rotary_dim, config.max_position_embeddings,
                                       rope['rope_theta'], is_neox_style=False)
        self.encoder = nn.ModuleList(EncoderLayer(config, self.rotary)
                                     for _ in range(config.encoder_num_hidden_layers))
        self.decoder = nn.ModuleList(DecoderLayer(config, self.rotary)
                                     for _ in range(config.decoder_num_hidden_layers))
        self.encoder_norm, self.decoder_norm = norm(width), norm(width)
        self.embed_tokens = Embedding(config.vocab_size, width, padding_idx=config.pad_token_id)
        self.proj_out = Linear(width, config.vocab_size, bias=False)
        self.proj_out.weight = self.embed_tokens.emb.weight

    def forward(self, input_values, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("Moonshine coverage evaluates unpadded waveform and token sequences")
        memory = encoder_hidden_states
        if memory is None:
            memory = self.groupnorm(self.tanh(self.conv1(input_values[:, None])))
            memory = self.gelu(self.conv3(self.gelu(self.conv2(memory)))).transpose(1, 2)
            for layer in self.encoder:
                memory = layer(memory)
            memory = self.encoder_norm(memory)
        hidden = self.embed_tokens(decoder_input_ids)
        cache = []
        for index, layer in enumerate(self.decoder):
            previous = (None, None) if past_key_values is None else past_key_values[index]
            hidden = hidden + layer.self_attn(layer.self_attn_layer_norm(hidden),
                                              past_key_value=previous[0])
            hidden = hidden + layer.encoder_attn(layer.encoder_attn_layer_norm(hidden), memory, previous[1])
            hidden = hidden + layer.mlp(layer.final_layer_norm(hidden))
            cache.append((layer.self_attn.last_cache, layer.encoder_attn.last_cache))
        return {"logits": self.proj_out(self.decoder_norm(hidden)),
                "encoder_last_hidden_state": memory, "past_key_values": tuple(cache)}


def build_from_config(config, device, dtype):
    if (config.attention_bias or config.encoder_hidden_act != "gelu"
            or config.decoder_hidden_act != "silu" or not config.use_cache
            or not config.tie_word_embeddings
            or config.encoder_num_attention_heads != config.encoder_num_key_value_heads
            or config.decoder_num_attention_heads != config.decoder_num_key_value_heads
            or config.encoder_num_attention_heads != config.decoder_num_attention_heads):
        raise ValueError("Moonshine case requires the documented equal-head, bias-free, tied cached model")
    return MoonshineForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    mapped, consumed = {}, set()
    for target, parameter in model.state_dict().items():
        if target == "proj_out.weight":
            source = target
        elif target == "embed_tokens.emb.weight":
            source = "model.decoder.embed_tokens.weight"
        elif target.startswith("encoder_norm."):
            source = target.replace("encoder_norm.", "model.encoder.layer_norm.")
        elif target.startswith("decoder_norm."):
            source = target.replace("decoder_norm.", "model.decoder.norm.")
        elif target.startswith(("conv", "groupnorm")):
            source = "model.encoder." + target.replace(".conv.", ".")
        else:
            stack, index, suffix = target.split(".", 2)
            if stack == "encoder":
                suffix = suffix.replace("layer_norm1.", "input_layernorm.").replace(
                    "layer_norm2.", "post_attention_layernorm.")
            else:
                suffix = suffix.replace("self_attn_layer_norm.", "input_layernorm.").replace(
                    "encoder_attn_layer_norm.", "post_attention_layernorm.").replace(
                    "final_layer_norm.", "final_layernorm.").replace(
                    "mlp.gate_up_proj.", "mlp.fc1.").replace("mlp.down_proj.", "mlp.fc2.")
            source = f"model.{stack}.layers.{index}.{suffix}"
        value = state[source]
        if ".mlp.gate_up_proj." in target:
            up, gate = value.chunk(2, dim=0)
            value = torch.cat((gate, up), dim=0)
        if value.shape != parameter.shape:
            raise ValueError(f"Moonshine weight shape mismatch at {source}")
        mapped[target] = value
        consumed.add(source)
    if consumed != set(state):
        raise ValueError(f"Moonshine unmapped state: {sorted(set(state) - consumed)}")
    if not torch.equal(state['proj_out.weight'], state['model.decoder.embed_tokens.weight']):
        raise ValueError("Moonshine projection and embedding must be tied")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs, encoder_input_name="input_values")

    def run():
        output = model(**inputs)
        cache = output.pop("past_key_values")
        return dict(output, **seq2seq_cache_outputs(cache))

    return {"forward": Workload(run=run)}
