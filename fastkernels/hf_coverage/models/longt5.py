"""LongT5 transient-global attention using local blocks and segment sums."""

import re
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from .t5 import T5ForConditionalGeneration
from .switch_transformers import make_workloads


def _blocks(tensor, size):
    batch, length = tensor.shape[:2]
    padding = -length % size
    if padding:
        tensor = torch.cat((tensor, tensor.new_zeros((batch, padding) + tensor.shape[2:])), dim=1)
    return tensor.reshape(batch, -1, size, *tensor.shape[2:])


def _neighbors(blocks):
    padding = torch.zeros_like(blocks[:, :1])
    padded = torch.cat((padding, blocks, padding), dim=1)
    return torch.cat((padded[:, :-2], padded[:, 1:-1], padded[:, 2:]), dim=2)


class TransientGlobalAttention(nn.Module):
    def __init__(self, config, first):
        super().__init__()
        self.config = config
        self.heads, self.dim = config.num_heads, config.d_kv
        self.local, self.global_size = config.local_radius + 1, config.global_block_size
        for name in ("q", "k", "v"):
            setattr(self, name, Linear(config.d_model, self.heads * self.dim, bias=False))
        self.o = Linear(self.heads * self.dim, config.d_model, bias=False)
        self.global_input_layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        if first:
            self.relative_attention_bias = Embedding(config.relative_attention_num_buckets, self.heads)
            self.global_relative_attention_bias = Embedding(config.relative_attention_num_buckets, self.heads)
        self.bmm, self.softmax, self.aggregate = BMM(), Softmax(dim=-1), SegmentCSR()

    def buckets(self, relative):
        return T5SelfAttention._relative_position_bucket(relative, bidirectional=True,
            num_buckets=self.config.relative_attention_num_buckets, max_distance=self.config.relative_attention_max_distance)

    def forward(self, hidden, mask=None, position_bias=None):
        if mask is not None:
            raise ValueError("LongT5 case uses its ordinary unpadded input path")
        batch, length, width = hidden.shape
        globals_ = length // self.global_size
        if globals_ == 0:
            raise ValueError("LongT5 development input must exercise global aggregation")
        # Fixed contiguous segments; orphan tokens belong to the last complete
        # block. No token-by-global one-hot matrix or dense artificial GEMM.
        indices = torch.arange(batch * globals_, device=hidden.device)
        offsets = indices.div(globals_, rounding_mode="floor") * length + indices.remainder(globals_) * self.global_size
        offsets = torch.cat((offsets, offsets.new_tensor([batch * length])))
        sums = self.aggregate(hidden.reshape(-1, width).float(), offsets, "sum").to(hidden.dtype)
        global_hidden = self.global_input_layer_norm(sums.reshape(batch, globals_, width))

        def shaped(projection, values):
            return projection(values).reshape(batch, -1, self.heads, self.dim)

        query = _blocks(shaped(self.q, hidden), self.local)
        key, value = (_neighbors(_blocks(shaped(projection, hidden), self.local)) for projection in (self.k, self.v))
        blocks = query.shape[1]
        global_key, global_value = (shaped(projection, global_hidden)[:, None].expand(-1, blocks, -1, -1, -1)
                                    for projection in (self.k, self.v))
        key, value = torch.cat((key, global_key), dim=2), torch.cat((value, global_value), dim=2)
        scores = self.bmm(query.permute(0, 1, 3, 2, 4), key.permute(0, 1, 3, 4, 2))
        if position_bias is None:
            local_query = torch.arange(self.local, device=hidden.device) + self.local
            local_key = torch.arange(3 * self.local, device=hidden.device)
            relative = local_key[None] - local_query[:, None]
            learned = self.relative_attention_bias(self.buckets(relative)).permute(2, 0, 1)[None, None]
            starts = torch.arange(blocks, device=hidden.device)[:, None] * self.local
            query_ids = starts + torch.arange(self.local, device=hidden.device)
            key_ids = starts - self.local + local_key
            valid = ((query_ids[:, :, None] < length) & (key_ids[:, None] >= 0)
                     & (key_ids[:, None] < length) & (relative.abs()[None] < self.local))
            local_mask = torch.where(valid, 0.0, -1e10)[None, :, None]
            local_bias = (learned + local_mask).to(hidden.dtype).expand(batch, -1, -1, -1, -1)
            block_ids = (torch.arange(length, device=hidden.device) // self.global_size).clamp_max(globals_ - 1)
            side_relative = torch.arange(globals_, device=hidden.device)[None] - block_ids[:, None]
            side_bias = self.global_relative_attention_bias(self.buckets(side_relative)).permute(2, 0, 1)
            # Native side masking adds FP32 zeros before casting the learned bias.
            side_bias = side_bias.float().to(hidden.dtype).permute(1, 0, 2)[None].expand(batch, -1, -1, -1)
            side_bias = _blocks(side_bias, self.local).transpose(2, 3)
            position_bias = torch.cat((local_bias, side_bias), dim=-1)
        probabilities = self.softmax((scores + position_bias).float()).to(hidden.dtype)
        output = self.bmm(probabilities, value.permute(0, 1, 3, 2, 4))
        output = output.permute(0, 1, 3, 2, 4).reshape(batch, -1, self.heads * self.dim)[:, :length]
        return self.o(output), position_bias


def build_from_config(config, device, dtype):
    if (config.encoder_attention_type != "transient-global" or config.feed_forward_proj != "gated-gelu"
            or config.tie_word_embeddings or not config.use_cache):
        raise ValueError("Selected LongT5 checkpoint requires transient-global, gated GELU, untied output and caching")
    carrier = SimpleNamespace(**{**dict(config), "scale_decoder_outputs": False})
    model = T5ForConditionalGeneration(carrier)
    # With tie_word_embeddings=False, pinned HF leaves both stack input
    # embeddings independent; its separate top-level shared table is unused.
    model.encoder.embed_tokens = Embedding(config.vocab_size, config.d_model)
    model.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
    for index, block in enumerate(model.encoder.block):
        block.layer[0].SelfAttention = TransientGlobalAttention(carrier, first=index == 0)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    consumed, mapped = set(), {}
    for name, target in model.state_dict().items():
        source = name.replace(".emb.weight", ".weight")
        if source == "shared.weight":
            source = "decoder.embed_tokens.weight"
        if source.startswith("encoder.block."):
            source = source.replace(".SelfAttention.", ".TransientGlobalSelfAttention.")
        source = re.sub(r"^decoder\.(\d+)\.self_attention\.", r"decoder.block.\1.layer.0.SelfAttention.", source)
        source = re.sub(r"^decoder\.(\d+)\.self_norm\.", r"decoder.block.\1.layer.0.layer_norm.", source)
        source = re.sub(r"^decoder\.(\d+)\.cross_attention\.", r"decoder.block.\1.layer.1.EncDecAttention.", source)
        source = re.sub(r"^decoder\.(\d+)\.cross_norm\.", r"decoder.block.\1.layer.1.layer_norm.", source)
        source = re.sub(r"^decoder\.(\d+)\.ff\.", r"decoder.block.\1.layer.2.", source)
        source = source.replace("decoder_norm.", "decoder.final_layer_norm.")
        if "qkv_proj" in source:
            names = [source.replace("qkv_proj", part) for part in ("q", "k", "v")]
        elif ".DenseReluDense.wi.weight" in source:
            names = [source.replace(".wi.", f".wi_{index}.") for index in (0, 1)]
        else:
            names = [source]
        value = torch.cat([state_dict[key] for key in names]) if len(names) > 1 else state_dict[source]
        if value.shape != target.shape:
            raise ValueError(f"LongT5 state shape mismatch: {source}")
        mapped[name] = value
        consumed.update(names)
    consumed.add("shared.weight")  # Declared but unused in this untied HF call.
    if consumed != set(state_dict):
        raise ValueError(f"LongT5 unmapped state: {sorted(set(state_dict) - consumed)}")
    model.load_state_dict(mapped)
