"""GPT-2 causal language modeling with learned positions and existing paged attention."""

import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaModel

from .llama import make_workloads as llama_workloads
from .olmo import ResidualLayerNorm
from .qwen2_precision import DenseCachedAttention


class LearnedPositionBackbone(LlamaModel):
    """Select affine normalization and plain MLP children in the existing decoder."""

    def __init__(self, config, activation):
        super().__init__(config)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.rotary_emb = None
        for layer in self.layers:
            layer.self_attn.rotary_emb = None
            layer.self_attn.nope = True
            layer.self_attn.o_proj = Linear(config.hidden_size, config.hidden_size)
            layer.input_layernorm = ResidualLayerNorm(
                config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False,
            )
            layer.post_attention_layernorm = ResidualLayerNorm(
                config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False,
            )
            layer.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size, act_approximate="tanh")
            if activation == "gelu_new":
                layer.mlp.act = NewGELUActivation()
        self.norm = ResidualLayerNorm(config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids) + self.position_embedding(positions)
        return super().forward(input_ids, positions, inputs_embeds=hidden_states)


class LearnedPositionCausalLM(nn.Module):
    def __init__(self, config, activation):
        super().__init__()
        self.config = config
        self.model = LearnedPositionBackbone(config, activation)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)


def build_decoder(config, device, dtype, *, kv_heads, activation):
    if _tp_size() != 1:
        raise ValueError("GPT coverage requires tensor parallel size 1")
    if (not config.use_cache or not config.tie_word_embeddings or config.add_cross_attention
            or not config.scale_attn_weights or config.activation_function != activation
            or config.output_attentions or config.output_hidden_states):
        raise ValueError("This case preserves the default cached, tied causal decoder and its activation")
    if config.n_embd % config.n_head:
        raise ValueError("GPT hidden width must divide into whole attention heads")
    adapted = LlamaConfig(
        hidden_size=config.n_embd, intermediate_size=config.n_inner or 4 * config.n_embd,
        num_hidden_layers=config.n_layer, num_attention_heads=config.n_head,
        num_key_value_heads=kv_heads, head_dim=config.n_embd // config.n_head,
        vocab_size=config.vocab_size, max_position_embeddings=config.n_positions,
        rms_norm_eps=config.layer_norm_epsilon, qkv_bias=True, dtype=dtype,
    )
    return LearnedPositionCausalLM(adapted, activation).to(device=device, dtype=dtype).eval()


def build_from_config(config, device, dtype):
    if config.scale_attn_by_inverse_layer_idx or config.reorder_and_upcast_attn:
        raise ValueError("The documented GPT-2 checkpoint disables layer-wise scaling and reordered attention")
    model = build_decoder(config, device, dtype, kv_heads=config.n_head, activation="gelu_new")
    # Reuse dense SDPA and cache storage to preserve HF's attention rounding.
    for layer in model.model.layers:
        layer.self_attn.attn = DenseCachedAttention(
            config.n_head, config.n_head, config.n_embd // config.n_head,
        )
    return model


def load_decoder_weights(model, state_dict, *, transposed_projections):
    remaining = dict(state_dict)
    mapped = {}

    def copy(destination, source, transpose=False):
        value = remaining.pop(source)
        mapped[destination] = value.t().contiguous() if transpose else value

    copy("model.embed_tokens.embedding_op.emb.weight", "transformer.wte.weight")
    copy("model.position_embedding.emb.weight", "transformer.wpe.weight")
    copy("lm_head.embedding_op.emb.weight", "lm_head.weight")
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("The source's tied word embedding and language-model head disagree")
    for field in ("weight", "bias"):
        copy(f"model.norm.{field}", f"transformer.ln_f.{field}")
    for index in range(len(model.model.layers)):
        for destination, source, is_projection in (
            ("input_layernorm", "ln_1", False),
            ("post_attention_layernorm", "ln_2", False),
            ("self_attn.qkv_proj", "attn.c_attn", True),
            ("self_attn.o_proj", "attn.c_proj", True),
            ("mlp.fc1", "mlp.c_fc", True),
            ("mlp.fc2", "mlp.c_proj", True),
        ):
            for field in ("weight", "bias"):
                copy(f"model.layers.{index}.{destination}.{field}",
                     f"transformer.h.{index}.{source}.{field}",
                     transpose=transposed_projections and is_projection and field == "weight")
    if remaining:
        raise KeyError(f"Unmapped GPT state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def load_state_dict_into(model, state_dict, config):
    del config
    load_decoder_weights(model, state_dict, transposed_projections=True)


def make_workloads(model, inputs, config, *, case=None):
    del config
    return llama_workloads(model, inputs, model.config, case=case)
