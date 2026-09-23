"""Bamba's attention/Mamba2 hybrid assembled from existing operations."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.patches.mamba_conv_weight import fp32_causal_conv_weight
from fastkernels.hf_coverage.patches.mamba2_precision import mamba2_forward
from fastkernels.infra.context import AttnBackendConfig, set_attn_backend_config, set_forward_context
from fastkernels.infra.mamba_state import Mamba2Metadata, MambaStateManager, build_chunk_metadata
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP
from fastkernels.tasks.baseline.L2.mamba2_mixer import Mamba2Mixer, Mixer2RMSNormGated
from fastkernels.tasks.baseline.L2.parallel_embedding import VocabParallelEmbedding
from .qwen2_precision import DenseCachedAttention, NativeRotaryEmbedding, SeparateQKV


def make_mixer(config, index):
    mixer = Mamba2Mixer(
        hidden_size=config.hidden_size,
        ssm_state_size=config.mamba_d_state,
        conv_kernel_size=config.mamba_d_conv,
        intermediate_size=config.mamba_expand * config.hidden_size,
        use_conv_bias=config.mamba_conv_bias,
        use_bias=config.mamba_proj_bias,
        n_groups=config.mamba_n_groups,
        num_heads=config.mamba_n_heads,
        head_dim=config.mamba_d_head,
        rms_norm_eps=config.rms_norm_eps,
        activation=config.hidden_act,
        chunk_size=config.mamba_chunk_size,
        layer_idx=index,
    )
    # Pinned Bamba normalizes all channels independently of SSM groups.
    mixer.norm = Mixer2RMSNormGated(mixer.intermediate_size, full_n_groups=1, eps=config.rms_norm_eps)
    return mixer


class BambaMLP(LlamaMLP):
    """Retain HF's separate projection shapes, including one-token decoding."""

    def __init__(self, config):
        super().__init__(config)
        self.matmul = Matmul()

    def forward(self, hidden):
        gate_weight, up_weight = self.gate_up_proj.weight.chunk(2, dim=0)
        gate = self.matmul(hidden, gate_weight)
        up = self.matmul(hidden, up_weight)
        return self.down_proj(self.act_fn(torch.cat((gate, up), dim=-1)))


class BambaLayer(nn.Module):
    def __init__(self, config, index, rotary):
        super().__init__()
        self.input_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = BambaMLP(config)
        self.is_attention = index in (config.attn_layer_indices or [])
        if self.is_attention:
            self.self_attn = LlamaAttention(
                config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                rotary_emb=rotary, bias=config.attention_bias,
                o_proj_bias=config.attention_bias, layer_idx=index,
            )
            self.self_attn.attn = DenseCachedAttention(
                config.num_attention_heads, config.num_key_value_heads, self.self_attn.head_dim,
            )
            head_dim = self.self_attn.head_dim
            self.self_attn.qkv_proj = SeparateQKV(
                self.self_attn.qkv_proj,
                [config.num_attention_heads * head_dim] + [config.num_key_value_heads * head_dim] * 2,
            )
        else:
            self.mamba = make_mixer(config, index)

    def forward(self, hidden, positions):
        normalized = self.input_layernorm(hidden)
        hidden = hidden + (self.self_attn(positions, normalized) if self.is_attention
                           else mamba2_forward(self.mamba, normalized))
        return hidden + self.feed_forward(self.pre_ff_layernorm(hidden))


class BambaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # Pinned HF default RoPE uses the full head despite partial_rotary_factor.
        self.rotary = NativeRotaryEmbedding(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            config.max_position_embeddings, config.rope_parameters["rope_theta"],
        )
        # HF prepares fixed inverse frequencies on CPU, then angles on GPU.
        # Reuse the native rotation callable to retain its BF16 rounding steps.
        head_dim = self.rotary.head_dim
        dimensions = torch.arange(0, head_dim, 2, device="cpu", dtype=torch.float32)
        frequencies = 1.0 / (config.rope_parameters["rope_theta"] ** (dimensions / head_dim))
        positions = torch.arange(config.max_position_embeddings,
                                 device=self.rotary.cos_sin_cache.device, dtype=torch.float32)
        angles = torch.outer(positions, frequencies.to(positions.device))
        self.rotary.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        self.layers = nn.ModuleList([BambaLayer(config, index, self.rotary) for index in range(config.num_hidden_layers)])
        self.final_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding_op.emb.weight

    def forward(self, ids, positions):
        hidden = self.embed_tokens(ids)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.final_layernorm(hidden))


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.mlp_bias or config.rope_parameters["rope_type"] != "default":
        raise ValueError("The selected Bamba case uses bias-free SwiGLU and default RoPE")
    if tuple(config.time_step_limit) != (0.0, float("inf")):
        raise ValueError("The selected Bamba case has unrestricted time steps")
    set_attn_backend_config(AttnBackendConfig(backend="flash_attn", block_size=256, kv_layout="NHD"))
    model = BambaForCausalLM(config).to(device=device, dtype=dtype).eval()
    for layer in model.layers:
        if not layer.is_attention:
            layer.mamba.A.data = layer.mamba.A.data.float()
    return model


def copy_parameter(parameter, value):
    loader = getattr(parameter, "weight_loader", None)
    if loader is None:
        parameter.data.copy_(value)
    else:
        loader(parameter, value)


def load_mixer(mixer, weights, prefix):
    for name in ("in_proj", "conv1d", "out_proj"):
        operation = getattr(mixer, name)
        copy_parameter(operation.weight, weights[prefix + name + ".weight"])
        if operation.bias is not None:
            copy_parameter(operation.bias, weights[prefix + name + ".bias"])
    copy_parameter(mixer.A, weights[prefix + "A_log"])
    for name in ("D", "dt_bias"):
        copy_parameter(getattr(mixer, name), weights[prefix + name])
    copy_parameter(mixer.norm.weight, weights[prefix + "norm.weight"])
    mixer.conv_weights = fp32_causal_conv_weight(mixer.conv1d.weight)


def load_state_dict_into(model, weights, config):
    copy_parameter(model.embed_tokens.embedding_op.emb.weight, weights["model.embed_tokens.weight"])
    copy_parameter(model.lm_head.weight, weights.get("lm_head.weight", weights["model.embed_tokens.weight"]))
    copy_parameter(model.final_layernorm.weight, weights["model.final_layernorm.weight"])
    for index, layer in enumerate(model.layers):
        prefix = f"model.layers.{index}."
        for name in ("input_layernorm", "pre_ff_layernorm"):
            copy_parameter(getattr(layer, name).weight, weights[prefix + name + ".weight"])
        for shard, name in enumerate(("gate", "up")):
            parameter = layer.feed_forward.gate_up_proj.weight
            parameter.weight_loader(parameter, weights[prefix + f"feed_forward.{name}_proj.weight"], shard)
        copy_parameter(layer.feed_forward.down_proj.weight, weights[prefix + "feed_forward.down_proj.weight"])
        if layer.is_attention:
            for shard in ("q", "k", "v"):
                parameter = layer.self_attn.qkv_proj.weight
                parameter.weight_loader(parameter, weights[prefix + f"self_attn.{shard}_proj.weight"], shard)
                if layer.self_attn.qkv_proj.bias is not None:
                    parameter = layer.self_attn.qkv_proj.bias
                    parameter.weight_loader(parameter, weights[prefix + f"self_attn.{shard}_proj.bias"], shard)
            copy_parameter(layer.self_attn.o_proj.weight, weights[prefix + "self_attn.o_proj.weight"])
            if layer.self_attn.o_proj.bias is not None:
                copy_parameter(layer.self_attn.o_proj.bias, weights[prefix + "self_attn.o_proj.bias"])
        else:
            load_mixer(layer.mamba, weights, prefix + "mamba.")


def hybrid_workloads(model, inputs, config, mixers, attentions, chunk_size,
                     *, case=None, attention_indices=None):
    """Shared cache layout for full-sequence attention plus Mamba2 state."""
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs["input_ids"]
    batch, length = ids.shape
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    steps = 2 if continuation else 1
    if length <= steps:
        raise ValueError("A hybrid workload needs a prompt and a decode token")
    if continuation and (attention_indices is None or len(attention_indices) != len(attentions)):
        raise ValueError("Full hybrid validation needs the native attention layer indices")
    prompt_length = length - steps
    device, dtype = ids.device, model.embed_tokens.embedding_op.emb.weight.dtype
    mixer = mixers[0]
    state = MambaStateManager(
        num_hidden_layers=config.num_hidden_layers, conv_dim=mixer.conv_dim,
        ssm_state_shape=(mixer.num_heads, mixer.head_dim, mixer.ssm_state_size),
        conv_kernel=mixer.conv_kernel_size, num_slots=batch + 1, dtype=dtype, device=device,
    )
    slots = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    starts = [index * prompt_length for index in range(batch + 1)]
    cu_prompt = torch.tensor(starts, device=device, dtype=torch.int32)
    prefill_meta = Mamba2Metadata(
        num_prefill_tokens=batch * prompt_length, num_prefills=batch,
        query_start_loc_p=cu_prompt, state_indices_p=slots,
        has_initial_states_p=torch.zeros(batch, device=device, dtype=torch.bool), chunk_size=chunk_size,
    )
    prefill_meta.cu_chunk_seqlen_p, prefill_meta.seq_idx_p, prefill_meta.last_chunk_indices_p = build_chunk_metadata(
        cu_prompt, chunk_size, host_qsl=starts,
    )
    decode_meta = Mamba2Metadata(num_decode_tokens=batch, num_decodes=batch, state_indices_d=slots)
    block_size = 256
    blocks_per_sequence = (length + block_size - 1) // block_size
    block_tables = torch.arange(batch * blocks_per_sequence, device=device, dtype=torch.int32).reshape(batch, -1)
    bases = torch.arange(batch, device=device, dtype=torch.int64) * blocks_per_sequence * block_size
    positions = torch.arange(prompt_length, device=device, dtype=torch.int64)
    prefill_slots = (bases[:, None] + positions).reshape(-1)
    prefill_positions = positions.repeat(batch)
    decode_positions = [torch.full((batch,), prompt_length + step, device=device, dtype=torch.int64)
                        for step in range(steps)]
    decode_lengths = [torch.full((batch,), prompt_length + step + 1, device=device, dtype=torch.int32)
                      for step in range(steps)]
    decode_slots = [bases + prompt_length + step for step in range(steps)]
    caches = []
    for attention in attentions:
        shape = (batch * blocks_per_sequence, block_size, attention.num_kv_heads, attention.head_size)
        attention.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        attention.v_cache = torch.zeros_like(attention.k_cache)
        caches.extend((attention.k_cache, attention.v_cache))

    def reset():
        for value in state.conv_states + state.ssm_states + caches:
            value.zero_()

    def forward(token_ids, is_prefill, step=0):
        context_length = prompt_length if is_prefill else prompt_length + step + 1
        with set_forward_context(
            is_prefill=is_prefill, mamba_state=state,
            mamba_metadata=prefill_meta if is_prefill else decode_meta,
            cu_seqlens_q=cu_prompt if is_prefill else None, cu_seqlens_k=cu_prompt if is_prefill else None,
            max_seqlen_q=prompt_length if is_prefill else 1, max_seqlen_k=context_length,
            slot_mapping=prefill_slots if is_prefill else decode_slots[step],
            block_tables=block_tables, context_lens=None if is_prefill else decode_lengths[step],
            max_context_len=context_length,
        ):
            output = model(token_ids.reshape(-1).contiguous(),
                           prefill_positions if is_prefill else decode_positions[step])
        return {"logits": output.reshape(batch, -1, config.vocab_size)}

    def prefill():
        return forward(ids[:, :prompt_length], True)

    def prepare_decode(step):
        reset()
        prefill()
        for previous in range(step):
            forward(ids[:, prompt_length + previous:prompt_length + previous + 1], False, previous)

    def collect(output, sequence_length):
        output = dict(output)
        for mixer in mixers:
            index = mixer.layer_idx
            # Exclude the reserved slot and expose the last kernel_size-1
            # inputs, which are the convolution history used on the next call.
            output[f"past_key_values.{index}.conv_states"] = state.conv_states[index][1:].transpose(-1, -2)
            output[f"past_key_values.{index}.recurrent_states"] = state.ssm_states[index][1:]
        for index, attention in zip(attention_indices, attentions):
            for name, cache in (("key", attention.k_cache), ("value", attention.v_cache)):
                logical = cache.reshape(batch, -1, attention.num_kv_heads, attention.head_size)
                output[f"past_key_values.{index}.{name}"] = logical[:, :sequence_length].transpose(1, 2)
        return output

    workloads = {"prefill": Workload(run=prefill, prepare=reset)}
    for step in range(steps):
        name = f"decode_{step + 1}" if continuation else "decode"
        workloads[name] = Workload(
            run=lambda step=step: forward(ids[:, prompt_length + step:prompt_length + step + 1], False, step),
            prepare=lambda step=step: prepare_decode(step),
        )
    if continuation:
        workloads["prefill"].collect = lambda output: collect(output, prompt_length)
        for step in range(steps):
            workloads[f"decode_{step + 1}"].collect = lambda output, step=step: collect(output, prompt_length + step + 1)
    return workloads


def make_workloads(model, inputs, config, *, case=None):
    return hybrid_workloads(
        model, inputs, config,
        [layer.mamba for layer in model.layers if not layer.is_attention],
        [layer.self_attn.attn for layer in model.layers if layer.is_attention], config.mamba_chunk_size,
        case=case, attention_indices=[i for i, layer in enumerate(model.layers) if layer.is_attention],
    )
