"""Pinned HF Zamba causal LM using existing attention and Mamba-v1 operations."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.patches.mamba_conv_weight import fp32_causal_conv_weight
from fastkernels.infra.context import (
    AttnBackendConfig,
    get_context,
    set_attn_backend_config,
    set_forward_context,
)
from fastkernels.infra.mamba_state import (
    MambaMetadata,
    MambaStateManager,
    compute_causal_conv1d_metadata,
)
from fastkernels.tasks.baseline.L1.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.mamba_ssm import selective_scan_fn, selective_state_update
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.parallel_embedding import VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.parallel_linear import MergedColumnParallelLinear, QKVParallelLinear


class ZambaMambaMixer(nn.Module):
    """Independent per-head projections and scans, with HF's interleaved gate."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.n_heads = config.n_mamba_heads
        self.head_dim = self.intermediate_size // self.n_heads
        self.state_size = config.mamba_d_state
        self.dt_rank = config.mamba_dt_rank
        self.activation = config.hidden_mamba_act
        self.in_proj = Linear(config.hidden_size, 2 * self.intermediate_size, bias=config.mamba_proj_bias)
        self.conv_weight = nn.Parameter(torch.empty(self.intermediate_size, 1, config.mamba_d_conv))
        self.register_buffer("conv_weights", torch.empty(0), persistent=False)
        self.conv_bias = nn.Parameter(torch.empty(self.intermediate_size)) if config.mamba_conv_bias else None
        self.x_proj_weight = nn.Parameter(torch.empty(self.n_heads, self.dt_rank + 2 * self.state_size, self.head_dim))
        self.dt_proj_weight = nn.Parameter(torch.empty(self.n_heads, self.head_dim, self.dt_rank))
        self.dt_proj_bias = nn.Parameter(torch.empty(self.n_heads, self.head_dim))
        self.A = nn.Parameter(torch.empty(self.n_heads, self.head_dim, self.state_size))
        self.D = nn.Parameter(torch.empty(self.n_heads, self.head_dim))
        self.out_proj = Linear(self.intermediate_size, config.hidden_size, bias=config.mamba_proj_bias)
        self.bmm = BMM()

    def forward(self, hidden):
        context = get_context()
        metadata = context.mamba_metadata
        conv_state = context.mamba_state.conv_states[self.layer_idx].transpose(-1, -2)
        ssm_state = context.mamba_state.ssm_states[self.layer_idx]
        tokens = hidden.shape[0]
        projected = self.in_proj(hidden).view(tokens, self.intermediate_size, 2)
        x = projected[..., 0].transpose(0, 1).contiguous()
        gate = projected[..., 1].transpose(0, 1).contiguous()
        conv_weight = self.conv_weights
        if context.is_prefill:
            x = causal_conv1d_fn(
                x, conv_weight, self.conv_bias,
                conv_states=conv_state,
                query_start_loc=metadata.query_start_loc_p,
                cache_indices=metadata.state_indices_p,
                has_initial_state=metadata.has_initial_states_p,
                activation=self.activation,
                metadata=metadata,
            )
        else:
            x = causal_conv1d_update(
                x.transpose(0, 1), conv_state, conv_weight, self.conv_bias,
                activation=self.activation, conv_state_indices=metadata.state_indices_d,
            ).transpose(0, 1)
        x = x.reshape(self.n_heads, self.head_dim, tokens)
        gate = gate.reshape(self.n_heads, self.head_dim, tokens)
        parameters = self.bmm(self.x_proj_weight, x).transpose(1, 2)
        dt, b, c = parameters.split((self.dt_rank, self.state_size, self.state_size), dim=-1)
        dt = self.bmm(self.dt_proj_weight, dt.transpose(1, 2))
        outputs = []
        # These heads are independent scans, never a Python recurrence over tokens.
        for head in range(self.n_heads):
            if context.is_prefill:
                output = selective_scan_fn(
                    x[head], ssm_state[:, head], dt[head], self.A[head],
                    b[head].transpose(0, 1), c[head].transpose(0, 1),
                    D=self.D[head].float(), z=gate[head],
                    delta_bias=self.dt_proj_bias[head].float(), delta_softplus=True,
                    query_start_loc=metadata.query_start_loc_p,
                    cache_indices=metadata.state_indices_p,
                    has_initial_state=metadata.has_initial_states_p,
                )
            else:
                output = torch.empty(tokens, self.head_dim, dtype=x.dtype, device=x.device)
                # Current library order is (..., D, dt_bias, z), unlike HF's
                # dependency signature (..., D, z, dt_bias). Use named arguments.
                selective_state_update(
                    ssm_state[:, head], x[head].transpose(0, 1), dt[head].transpose(0, 1),
                    self.A[head], b[head], c[head], D=self.D[head],
                    dt_bias=self.dt_proj_bias[head].float(), z=gate[head].transpose(0, 1),
                    dt_softplus=True, state_batch_indices=metadata.state_indices_d, out=output,
                )
                output = output.transpose(0, 1)
            outputs.append(output)
        return self.out_proj(torch.cat(outputs, dim=0).transpose(0, 1))


class ZambaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_size = config.num_attention_heads * config.attention_head_dim
        self.kv_size = config.num_key_value_heads * config.attention_head_dim
        self.qkv_proj = QKVParallelLinear(
            config.attention_hidden_size, config.attention_head_dim,
            config.num_attention_heads, config.num_key_value_heads, bias=False,
        )
        self.attention = Attention(
            config.num_attention_heads, config.attention_head_dim,
            scale=(config.attention_head_dim / 2) ** -0.5,
            num_kv_heads=config.num_key_value_heads,
        )
        self.o_proj = Linear(self.q_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        q, k, v = self.qkv_proj(hidden).split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        return self.o_proj(self.attention(q, k, v))


class ZambaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.attention_hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ZambaAttention(config)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size, config.intermediate_size], bias=False,
        )
        self.activation = GeluAndMul(approximate="none")
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden, original_embeddings):
        hidden = self.input_layernorm(torch.cat((hidden, original_embeddings), dim=-1))
        hidden = self.pre_ff_layernorm(self.self_attn(hidden))
        return self.down_proj(self.activation(self.gate_up_proj(hidden)))


class ZambaLayer(nn.Module):
    def __init__(self, config, index, block_type):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = ZambaMambaMixer(config, index)
        self.hybrid = block_type == "hybrid"
        if self.hybrid:
            self.shared_transf = ZambaTransformer(config)
            self.linear = Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden, original_embeddings):
        residual = hidden
        if self.hybrid:
            hidden = hidden + self.linear(self.shared_transf(hidden, original_embeddings))
        return residual + self.mamba(self.input_layernorm(hidden))


class ZambaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            ZambaLayer(config, index, kind) for index, kind in enumerate(config.layers_block_type)
        ])
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding_op.emb.weight
            shared = None
            for layer in self.layers:
                if not layer.hybrid:
                    continue
                if shared is None:
                    shared = layer.shared_transf
                else:
                    # Share weights while retaining the layer's independent KV cache.
                    for name, parameter in shared.named_parameters():
                        module_name, _, parameter_name = name.rpartition(".")
                        target = layer.shared_transf.get_submodule(module_name)
                        setattr(target, parameter_name, parameter)

    def forward(self, ids):
        hidden = self.embed_tokens(ids)
        original_embeddings = hidden
        for layer in self.layers:
            hidden = layer(hidden, original_embeddings)
        return self.lm_head(self.final_layernorm(hidden))


def build_from_config(config, device, dtype):
    set_attn_backend_config(AttnBackendConfig(backend="flash_attn", block_size=256, kv_layout="NHD"))
    model = ZambaForCausalLM(config).to(device=device, dtype=dtype).eval()
    for layer in model.layers:
        layer.mamba.A.data = layer.mamba.A.data.float()
    return model


def load_state_dict_into(model, state_dict, config):
    def copy(parameter, name):
        parameter.data.copy_(state_dict[name])

    copy(model.embed_tokens.embedding_op.emb.weight, "model.embed_tokens.weight")
    if "lm_head.weight" in state_dict:
        copy(model.lm_head.weight, "lm_head.weight")
    copy(model.final_layernorm.weight, "model.final_layernorm.weight")
    shared_prefix = None
    for index, layer in enumerate(model.layers):
        prefix = f"model.layers.{index}."
        if layer.hybrid:
            transformer = layer.shared_transf
            if shared_prefix is None or not config.tie_word_embeddings:
                shared_prefix = prefix + "shared_transf."
            copy(transformer.input_layernorm.weight, shared_prefix + "input_layernorm.weight")
            copy(transformer.pre_ff_layernorm.weight, shared_prefix + "pre_ff_layernorm.weight")
            qkv = transformer.self_attn.qkv_proj.weight
            for part in ("q", "k", "v"):
                qkv.weight_loader(qkv, state_dict[shared_prefix + f"self_attn.{part}_proj.weight"], part)
            copy(transformer.self_attn.o_proj.weight, shared_prefix + "self_attn.o_proj.weight")
            gate_up = transformer.gate_up_proj.weight
            for part, name in enumerate(("gate", "up")):
                gate_up.weight_loader(gate_up, state_dict[shared_prefix + f"feed_forward.{name}_proj.weight"], part)
            copy(transformer.down_proj.weight, shared_prefix + "feed_forward.down_proj.weight")
            copy(layer.linear.weight, prefix + "linear.weight")
            prefix += "mamba_decoder."
        copy(layer.input_layernorm.weight, prefix + "input_layernorm.weight")
        prefix += "mamba."
        mixer = layer.mamba
        for name in ("in_proj", "out_proj"):
            projection = getattr(mixer, name)
            copy(projection.weight, prefix + name + ".weight")
            if projection.bias is not None:
                copy(projection.bias, prefix + name + ".bias")
        copy(mixer.conv_weight, prefix + "conv1d.weight")
        mixer.conv_weights = fp32_causal_conv_weight(mixer.conv_weight)
        if mixer.conv_bias is not None:
            copy(mixer.conv_bias, prefix + "conv1d.bias")
        for name in ("x_proj_weight", "dt_proj_weight", "dt_proj_bias", "D"):
            copy(getattr(mixer, name), prefix + name)
        mixer.A.data.copy_(-torch.exp(state_dict[prefix + "A_log"].float()))


def make_workloads(model, inputs, config, *, case=None):
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs["input_ids"]
    batch, length = ids.shape
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    steps = 2 if continuation else 1
    if length <= steps:
        raise ValueError("A recurrent workload needs a prompt and a decode token")
    prompt_length = length - steps
    prompt_ids = ids[:, :prompt_length]
    dtype = next(model.parameters()).dtype
    device = ids.device
    intermediate = config.mamba_expand * config.hidden_size
    state = MambaStateManager(
        num_hidden_layers=config.num_hidden_layers, conv_dim=intermediate,
        ssm_state_shape=(config.n_mamba_heads, intermediate // config.n_mamba_heads, config.mamba_d_state),
        conv_kernel=config.mamba_d_conv, num_slots=batch + 1, dtype=dtype, device=device,
    )
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    cu_prompt = torch.arange(batch + 1, dtype=torch.int32, device=device) * prompt_length
    prefill_metadata = MambaMetadata(
        num_prefill_tokens=batch * prompt_length, num_prefills=batch,
        query_start_loc_p=cu_prompt, state_indices_p=slots,
        has_initial_states_p=torch.zeros(batch, dtype=torch.bool, device=device),
    )
    (
        prefill_metadata.nums_dict,
        prefill_metadata.batch_ptr,
        prefill_metadata.token_chunk_offset_ptr,
    ) = compute_causal_conv1d_metadata(cu_prompt, seqlens_cpu=[prompt_length] * batch)
    decode_metadata = MambaMetadata(num_decode_tokens=batch, num_decodes=batch, state_indices_d=slots)

    block_size = 256
    blocks_per_sequence = (length + block_size - 1) // block_size
    block_tables = torch.arange(batch * blocks_per_sequence, dtype=torch.int32, device=device).reshape(batch, -1)
    sequence_base = torch.arange(batch, device=device, dtype=torch.int64) * blocks_per_sequence * block_size
    prefill_mapping = (sequence_base[:, None] + torch.arange(prompt_length, device=device)).reshape(-1)
    decode_mapping = [sequence_base + prompt_length + step for step in range(steps)]
    caches = []
    for layer in model.layers:
        if layer.hybrid:
            attention = layer.shared_transf.self_attn.attention
            shape = (batch * blocks_per_sequence, block_size, config.num_key_value_heads, config.attention_head_dim)
            attention.k_cache = torch.zeros(shape, device=device, dtype=dtype)
            attention.v_cache = torch.zeros_like(attention.k_cache)
            caches.extend((attention.k_cache, attention.v_cache))
    lengths = [torch.full((batch,), prompt_length + step + 1, dtype=torch.int32, device=device)
               for step in range(steps)]

    def reset():
        for tensor in state.conv_states + state.ssm_states + caches:
            tensor.zero_()

    def forward(token_ids, is_prefill, step=0):
        context_length = prompt_length if is_prefill else prompt_length + step + 1
        with set_forward_context(
            is_prefill=is_prefill, mamba_state=state,
            mamba_metadata=prefill_metadata if is_prefill else decode_metadata,
            cu_seqlens_q=cu_prompt if is_prefill else None,
            cu_seqlens_k=cu_prompt if is_prefill else None,
            max_seqlen_q=prompt_length if is_prefill else 1,
            max_seqlen_k=context_length,
            slot_mapping=prefill_mapping if is_prefill else decode_mapping[step],
            block_tables=None if is_prefill else block_tables,
            context_lens=None if is_prefill else lengths[step],
            max_context_len=context_length if continuation else length,
        ):
            logits = model(token_ids.reshape(-1).contiguous())
        return {"logits": logits.reshape(batch, -1, logits.shape[-1])}

    def prepare_decode(step):
        reset()
        forward(prompt_ids, True)
        for previous in range(step):
            forward(ids[:, prompt_length + previous:prompt_length + previous + 1], False, previous)

    def collect(output, sequence_length):
        output = dict(output)
        for index, layer in enumerate(model.layers):
            output[f"past_key_values.{index}.conv_states"] = state.conv_states[index][1:].transpose(-1, -2)
            output[f"past_key_values.{index}.recurrent_states"] = state.ssm_states[index][1:]
            if layer.hybrid:
                attention = layer.shared_transf.self_attn.attention
                for name, cache in (("key", attention.k_cache), ("value", attention.v_cache)):
                    logical = cache.reshape(batch, -1, config.num_key_value_heads, config.attention_head_dim)
                    output[f"past_key_values.{index}.{name}"] = logical[:, :sequence_length].transpose(1, 2)
        return output

    workloads = {
        "prefill": Workload(run=lambda: forward(prompt_ids, True), prepare=reset),
    }
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
