"""The documented ViT/GPT-2 captioning wrapper with decoder cross-attention."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation

from . import vit
from .bart import _fresh_cache
from .mbart import PreNormEncoderAttention
from ..runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads


class CaptionDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.n_embd
        self.heads = config.n_head
        self.ln_1 = LayerNorm(width, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.ln_2 = LayerNorm(width, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.ln_cross_attn = LayerNorm(width, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.self_qkv = Linear(width, 3 * width)
        self.self_proj = Linear(width, width)
        self.cross_query = Linear(width, width)
        self.cross_kv = Linear(width, 2 * width)
        self.cross_proj = Linear(width, width)
        self.fc = Linear(width, config.n_inner or 4 * width)
        self.proj = Linear(config.n_inner or 4 * width, width)
        self.activation = NewGELUActivation()
        self.attention = DenseAttention(backend="cudnn")

    def forward(self, hidden, memory, past_key_value=None):
        batch, length, width = hidden.shape
        shape = (batch, length, self.heads, width // self.heads)
        query, key, value = [x.reshape(shape) for x in self.self_qkv(self.ln_1(hidden)).chunk(3, dim=-1)]
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        if past_key_value is None:
            self_cache = _fresh_cache(key, value)
            attention_kwargs = {"causal": True}
        else:
            past_self = past_key_value[0]
            self_cache = tuple(torch.cat((old, new), dim=2)
                               for old, new in zip(past_self, (key, value)))
            attention_kwargs = {}
            if length > 1:
                queries = torch.arange(length, device=hidden.device) + past_self[0].shape[2]
                keys = torch.arange(self_cache[0].shape[2], device=hidden.device)
                attention_kwargs["attn_mask"] = queries[:, None] >= keys[None, :]
        key, value = self_cache
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2), **attention_kwargs)
        hidden = hidden + self.self_proj(context.reshape(batch, length, width))
        query = self.cross_query(self.ln_cross_attn(hidden)).reshape(shape)
        if past_key_value is None:
            key, value = [x.reshape(batch, memory.shape[1], self.heads, width // self.heads).transpose(1, 2)
                          for x in self.cross_kv(memory).chunk(2, dim=-1)]
            cross_cache = _fresh_cache(key, value)
        else:
            cross_cache = past_key_value[1]
        key, value = cross_cache
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2))
        hidden = hidden + self.cross_proj(context.reshape(batch, length, width))
        hidden = hidden + self.proj(self.activation(self.fc(self.ln_2(hidden))))
        return hidden, (self_cache, cross_cache)


class VisionEncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = vit.ViTModel(config.encoder)
        # Reuse the existing projection-preserving wrapper to select native
        # HF's cuDNN attention even when imports disable global selection.
        for block in self.encoder.encoder:
            block.attn = PreNormEncoderAttention(block.attn)
        dec = config.decoder
        self.wte = Embedding(dec.vocab_size, dec.n_embd)
        self.wpe = Embedding(dec.n_positions, dec.n_embd)
        self.layers = nn.ModuleList([CaptionDecoderLayer(dec) for _ in range(dec.n_layer)])
        self.ln_f = LayerNorm(dec.n_embd, eps=dec.layer_norm_epsilon, promote_fp32=False)
        self.lm_head = Linear(dec.n_embd, dec.vocab_size, bias=False)
        self.lm_head.weight = self.wte.emb.weight

    def forward(self, pixel_values, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("The captioning workload uses unpadded images and decoder tokens")
        memory = encoder_hidden_states
        if memory is None:
            # ViT also computes its ordinary pooler; the composite discards it.
            memory = self.encoder(pixel_values)["last_hidden_state"]
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        positions = (torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
                     + past_length)[None]
        hidden = self.wte(decoder_input_ids) + self.wpe(positions)
        cache = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, memory, None if past_key_values is None else past_key_values[index])
            cache.append(state)
        return {"logits": self.lm_head(self.ln_f(hidden)), "encoder_last_hidden_state": memory,
                "past_key_values": cache}


def build_from_config(config, device, dtype):
    enc, dec = config.encoder, config.decoder
    if (enc.model_type != "vit" or dec.model_type != "gpt2" or not dec.add_cross_attention
            or not dec.use_cache or not dec.tie_word_embeddings or dec.activation_function != "gelu_new"
            or enc.hidden_size != dec.n_embd or dec.scale_attn_by_inverse_layer_idx
            or dec.reorder_and_upcast_attn or not dec.scale_attn_weights):
        raise ValueError("The selected captioning checkpoint requires same-width ViT and cached cross-attending GPT-2")
    return VisionEncoderDecoder(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    vision = {name.removeprefix("encoder."): remaining.pop(name)
              for name in list(remaining) if name.startswith("encoder.")}
    vit.load_state_dict_into(model.encoder, vision, config.encoder)
    mapped = {}
    for name in model.state_dict():
        if name.startswith("encoder."):
            continue
        source = name.replace(".emb.weight", ".weight")
        if name.startswith("lm_head."):
            source = "decoder." + source
        else:
            source = "decoder.transformer." + source.replace("layers.", "h.")
        for target, origin in (("self_qkv", "attn.c_attn"), ("self_proj", "attn.c_proj"),
                               ("cross_query", "crossattention.q_attn"), ("cross_kv", "crossattention.c_attn"),
                               ("cross_proj", "crossattention.c_proj"), ("fc", "mlp.c_fc"),
                               ("proj", "mlp.c_proj")):
            source = source.replace(f".{target}.", f".{origin}.")
        tensor = remaining.pop(source)
        if name.startswith("layers.") and name.endswith(".weight") and tensor.ndim == 2:
            tensor = tensor.t().contiguous()
        mapped[name] = tensor
    if not torch.equal(mapped["wte.emb.weight"], mapped["lm_head.weight"]):
        raise ValueError("GPT-2 tied word weights disagree")
    if remaining:
        raise KeyError(f"Unmapped captioning weights: {sorted(remaining)}")
    mapped.update({"encoder." + name: tensor for name, tensor in model.encoder.state_dict().items()})
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case.get("workload") == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs, encoder_input_name="pixel_values")

    def forward():
        output = model(**inputs)
        return {"logits": output["logits"], "encoder_last_hidden_state": output["encoder_last_hidden_state"],
                **seq2seq_cache_outputs(output["past_key_values"])}

    return {"forward": Workload(run=forward)}
