"""OPT-350m's post-normalized decoder with both embedding-width projections."""

import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .llama import make_workloads as llama_workloads
from .qwen2_precision import DenseCachedAttention


class UnitScaleAttention(DenseAttention):
    """The query is already scaled, so request no additional attention scale."""

    def forward(self, query, key, value, **kwargs):
        return super().forward(query, key, value, softmax_scale=1.0, **kwargs)


class PrescaledAttention(LlamaAttention):
    """Keep OPT's fixed query scaling before the existing attention operation."""

    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_attention_heads,
                         config.num_attention_heads, config.hidden_size // config.num_attention_heads,
                         bias=True, o_proj_bias=True, nope=True)
        self.attn = DenseCachedAttention(
            config.num_attention_heads, config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )
        self.attn.attention = UnitScaleAttention(backend="cudnn")

    def forward(self, positions, hidden_states):
        del positions
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = query * self.head_dim ** -0.5
        return self.o_proj(self.attn(query, key, value))


class OPTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = PrescaledAttention(config)
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.output_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.mlp = VitEncoderMlp(config.hidden_size, config.ffn_dim)
        self.mlp.act = ReLU()

    def forward(self, hidden_states, positions):
        hidden_states = self.attention_norm(hidden_states + self.self_attn(positions, hidden_states))
        return self.output_norm(hidden_states + self.mlp(hidden_states))


class OPTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.word_embed_proj_dim)
        self.embed_positions = Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.project_in = Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        self.project_out = Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        self.layers = nn.ModuleList([OPTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, positions):
        hidden_states = self.project_in(self.embed_tokens(input_ids)) + self.embed_positions(positions + 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return self.project_out(hidden_states)


class OPTForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = OPTDecoder(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.word_embed_proj_dim)
        self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight


def build_from_config(config, device, dtype):
    if (_tp_size() != 1 or config.do_layer_norm_before or not config.enable_bias
            or not config.layer_norm_elementwise_affine or config.activation_function != "relu"
            or config.word_embed_proj_dim == config.hidden_size or not config.use_cache
            or not config.tie_word_embeddings or config.output_attentions or config.output_hidden_states):
        raise ValueError("This case preserves OPT-350m's post-normalized, projected, cached causal decoder")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("OPT hidden width must divide into whole attention heads")
    return OPTForCausalLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {}

    def copy(destination, source):
        mapped[destination] = remaining.pop(source)

    prefix = "model.decoder."
    copy("model.embed_tokens.embedding_op.emb.weight", prefix + "embed_tokens.weight")
    copy("model.embed_positions.emb.weight", prefix + "embed_positions.weight")
    copy("lm_head.embedding_op.emb.weight", "lm_head.weight")
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("OPT's tied source embedding and head disagree")
    for name in ("project_in", "project_out"):
        copy(f"model.{name}.weight", prefix + name + ".weight")
    for index in range(config.num_hidden_layers):
        source = prefix + f"layers.{index}."
        target = f"model.layers.{index}."
        for field in ("weight", "bias"):
            mapped[target + "self_attn.qkv_proj." + field] = torch.cat([
                remaining.pop(source + f"self_attn.{part}_proj.{field}") for part in ("q", "k", "v")
            ])
            for destination, name in (("self_attn.o_proj", "self_attn.out_proj"),
                                      ("attention_norm", "self_attn_layer_norm"),
                                      ("output_norm", "final_layer_norm"),
                                      ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2")):
                copy(target + destination + "." + field, source + name + "." + field)
    if remaining:
        raise KeyError(f"Unmapped OPT state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return llama_workloads(model, inputs, config, case=case)
