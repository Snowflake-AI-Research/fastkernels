"""Qwen3-Next's existing full/Gated-DeltaNet attention and shared-expert MoE."""

import torch

from fastkernels.hf_coverage.models.bamba import copy_parameter
from fastkernels.hf_coverage.patches.mamba_conv_weight import fp32_causal_conv_weight
from fastkernels.infra.context import (AttnBackendConfig, KimiLinearMetadata, get_attn_backend_config,
                                      set_attn_backend_config, set_forward_context)
from fastkernels.infra.mamba_state import KimiLinearStateManager, compute_causal_conv1d_metadata
from fastkernels.tasks.baseline.L4.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM


def build_from_config(config, device, dtype):
    if config.hidden_act != 'silu' or config.attention_bias or config.tie_word_embeddings:
        raise ValueError('Selected Qwen3-Next example uses SiLU, bias-free attention and untied embeddings')
    if config.rope_parameters['rope_type'] != 'default':
        raise ValueError('Selected Qwen3-Next example uses default partial RoPE')
    if config.mlp_only_layers or config.decoder_sparse_step != 1 or config.output_router_logits:
        raise ValueError('Selected Qwen3-Next example uses an MoE at every layer without router outputs')
    if not {'full_attention', 'linear_attention'} <= set(config.layer_types):
        raise ValueError('The development case must retain both attention types')
    set_attn_backend_config(AttnBackendConfig(backend='flash_attn', block_size=256, kv_layout='NHD'))
    local_config = Qwen3NextConfig._from_hf(config)
    local_config.dtype = dtype
    return Qwen3NextForCausalLM(local_config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, weights, config):
    remaining = dict(weights)
    backbone = model.model
    copy_parameter(backbone.embed_tokens.embedding_op.emb.weight, remaining.pop('model.embed_tokens.weight'))
    copy_parameter(model.lm_head.embedding_op.emb.weight, remaining.pop('lm_head.weight'))
    copy_parameter(backbone.norm.weight, remaining.pop('model.norm.weight'))
    for index, layer in enumerate(backbone.layers):
        prefix = f'model.layers.{index}.'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            copy_parameter(getattr(layer, name).weight, remaining.pop(prefix + name + '.weight'))
        if layer.layer_type == 'full_attention':
            attention = layer.self_attn
            ap = prefix + 'self_attn.'
            for shard in ('q', 'k', 'v'):
                parameter = attention.qkv_proj.weight
                parameter.weight_loader(parameter, remaining.pop(ap + shard + '_proj.weight'), shard)
            copy_parameter(attention.o_proj.weight, remaining.pop(ap + 'o_proj.weight'))
            for name in ('q_norm', 'k_norm'):
                copy_parameter(getattr(attention, name).weight, remaining.pop(ap + name + '.weight'))
        else:
            attention = layer.linear_attn
            ap = prefix + 'linear_attn.'
            for name in ('in_proj_qkvz', 'in_proj_ba', 'conv1d', 'out_proj', 'norm'):
                copy_parameter(getattr(attention, name).weight, remaining.pop(ap + name + '.weight'))
            for name in ('A_log', 'dt_bias'):
                copy_parameter(getattr(attention, name), remaining.pop(ap + name))
            # This existing GDN component reads the live convolution parameter.
            attention.conv1d.weight.data = fp32_causal_conv_weight(attention.conv1d.weight)
            attention.process_weights_after_loading()
        mlp = layer.mlp
        mp = prefix + 'mlp.'
        copy_parameter(mlp.gate.weight, remaining.pop(mp + 'gate.weight'))
        mlp.w13.data.copy_(remaining.pop(mp + 'experts.gate_up_proj'))
        mlp.w2.data.copy_(remaining.pop(mp + 'experts.down_proj'))
        for shard, name in enumerate(('gate', 'up')):
            parameter = mlp.shared_expert.gate_up_proj.weight
            parameter.weight_loader(parameter, remaining.pop(mp + f'shared_expert.{name}_proj.weight'), shard)
        copy_parameter(mlp.shared_expert.down_proj.weight, remaining.pop(mp + 'shared_expert.down_proj.weight'))
        copy_parameter(mlp.shared_expert_gate.weight, remaining.pop(mp + 'shared_expert_gate.weight'))
        mlp.process_weights_after_loading()
    if remaining:
        raise KeyError(f'Unmapped Qwen3-Next state: {sorted(remaining)}')


def make_workloads(model, inputs, config, *, case=None):
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs['input_ids']
    if ids.ndim != 2 or ids.shape[0] < 1:
        raise ValueError('Qwen3-Next requires [batch, sequence] input_ids')
    batch, length = ids.shape
    continuation = case is not None and case.get('workload') == 'causal_lm_continuation'
    decode_steps = 2 if continuation else 1
    if length <= decode_steps:
        raise ValueError('Qwen3-Next requires a prompt and a decode token')
    prompt_length = length - decode_steps
    device, dtype = ids.device, model.model.embed_tokens.embedding_op.emb.weight.dtype
    block_size = get_attn_backend_config().block_size
    blocks_per_sequence = (length + block_size - 1) // block_size
    state = KimiLinearStateManager(
        config=model.config, num_slots=batch + 1, block_size=block_size,
        num_mla_blocks=batch * blocks_per_sequence, allocate_mla_kv_tensors=True,
        tp_size=1, device=device, dtype=dtype,
    )
    slots = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    block_tables = torch.arange(batch * blocks_per_sequence, device=device, dtype=torch.int32).reshape(batch, -1)
    bases = torch.arange(batch, device=device, dtype=torch.int64) * blocks_per_sequence * block_size
    prompt_positions = torch.arange(prompt_length, device=device, dtype=torch.int64)
    decode_phases = [f'decode_{step + 1}' for step in range(decode_steps)] if continuation else ['decode']
    phase_lengths = {'prefill': prompt_length,
                     **{phase: prompt_length + step + 1 for step, phase in enumerate(decode_phases)}}
    metadata, positions, selected_ids = {}, {}, {}
    for phase, context_length in phase_lengths.items():
        is_prefill = phase == 'prefill'
        tokens = prompt_length if is_prefill else 1
        position = context_length - 1
        starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * tokens
        nums, batch_ptr, offsets = compute_causal_conv1d_metadata(starts, seqlens_cpu=[tokens] * batch)
        metadata[phase] = KimiLinearMetadata(
            num_actual_tokens=batch * tokens, query_start_loc=starts, query_start_loc_int32=starts,
            max_query_len=tokens, seq_lens=torch.full((batch,), context_length, device=device, dtype=torch.int32),
            max_seq_len=context_length, state_indices=slots, state_indices_long=slots.long(),
            num_prefills=batch if is_prefill else 0, num_prefill_tokens=batch * tokens if is_prefill else 0,
            num_decodes=0 if is_prefill else batch, num_decode_tokens=0 if is_prefill else batch,
            has_initial_state=torch.zeros(batch, device=device, dtype=torch.bool) if is_prefill else None,
            all_have_initial_state=False, any_have_initial_state=False,
            slot_mapping=(bases[:, None] + prompt_positions).reshape(-1) if is_prefill else bases + position,
            block_tables=block_tables, nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets,
        )
        positions[phase] = (prompt_positions.repeat(batch) if is_prefill else
                            torch.full((batch,), position, device=device, dtype=torch.int64))
        selected_ids[phase] = ids[:, :prompt_length] if is_prefill else ids[:, position:position + 1]
    tensors = [t for collection in (state.gdn_conv, state.recurrent, state.k_cache, state.v_cache) for t in collection if t is not None]

    def reset():
        for tensor in tensors:
            tensor.zero_()

    def forward(phase):
        is_prefill = phase == 'prefill'
        with set_forward_context(is_prefill=is_prefill, kda_state=state, kda_metadata=metadata[phase]):
            hidden = model(selected_ids[phase].reshape(-1).contiguous(), positions[phase], state_manager=state)
            # ParallelLMHead's serving wrapper selects the last prompt token.
            # Its existing matrix operation retains all native-dtype HF logits.
            logits = model.lm_head.linear_op(hidden, model.lm_head.embedding_op.emb.weight)
        return {'logits': logits.reshape(batch, -1, config.vocab_size)}

    def collector(context_length):
        def collect(output):
            output = dict(output)
            for index, layer in enumerate(model.model.layers):
                prefix = f'past_key_values.{index}.'
                if layer.layer_type == 'linear_attention':
                    # HF retains four raw columns, but its oldest column is
                    # discarded before the next width-four convolution reads.
                    # These three columns are the complete future-used history.
                    output[prefix + 'conv_states'] = state.gdn_conv[index][1:batch + 1]
                    output[prefix + 'recurrent_states'] = state.recurrent[index][1:batch + 1].transpose(-1, -2)
                else:
                    for name, pages in (('key', state.k_cache[index]), ('value', state.v_cache[index])):
                        if layer.self_attn.kv_layout == 'HND':
                            pages = pages.transpose(1, 2)
                        logical = pages.reshape(batch, blocks_per_sequence * block_size, *pages.shape[-2:])
                        output[prefix + name] = logical[:, :context_length].transpose(1, 2)
            return output
        return collect

    workloads = {'prefill': Workload(run=lambda: forward('prefill'), prepare=reset)}
    for step, phase in enumerate(decode_phases):
        def prepare_decode(step=step):
            reset()
            forward('prefill')
            for prior in decode_phases[:step]:
                forward(prior)

        workloads[phase] = Workload(run=lambda phase=phase: forward(phase), prepare=prepare_decode)
    if continuation:
        for phase, workload in workloads.items():
            workload.collect = collector(phase_lengths[phase])
    return workloads
