"""The existing Jamba L4, with matching HF weights and bounded hybrid state."""

from dataclasses import fields

import torch

from fastkernels.hf_coverage.patches.mamba_conv_weight import fp32_causal_conv_weight
from fastkernels.infra.context import AttnBackendConfig, set_attn_backend_config, set_forward_context
from fastkernels.infra.jamba_engine import JambaMambaMetadata
from fastkernels.tasks.baseline.L4.jamba import JambaConfig, JambaForCausalLM


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.tie_word_embeddings or not config.use_cache or config.output_router_logits:
        raise ValueError("The selected Jamba case uses cached SiLU layers, an untied head, and no optional router output")
    values = {field.name: getattr(config, field.name) for field in fields(JambaConfig)
              if hasattr(config, field.name) and field.name != "dtype"}
    adapted = JambaConfig(**values, dtype=dtype)
    set_attn_backend_config(AttnBackendConfig(backend="flash_attn", block_size=256, kv_layout="NHD"))
    model = JambaForCausalLM(adapted).to(device=device, dtype=dtype).eval()
    for layer in model.model.layers:
        if hasattr(layer, "mamba"):
            layer.mamba.A.data = layer.mamba.A.data.float()
    return model


def load_state_dict_into(model, weights, config):
    mapped = {}
    for name, value in weights.items():
        if any(f".self_attn.{part}_proj." in name for part in ("q", "k", "v")):
            continue
        if ".feed_forward.gate_proj." in name or ".feed_forward.up_proj." in name:
            continue
        if ".feed_forward.experts.gate_up_proj" in name:
            target = name.replace("experts.gate_up_proj", "w13")
        elif ".feed_forward.experts.down_proj" in name:
            target = name.replace("experts.down_proj", "w2")
        else:
            target = model._remap_name(name)
        # Existing Mamba loader performs this inference-constant conversion.
        if target.endswith("mamba.A"):
            parameter = dict(model.named_parameters())[target]
            parameter.weight_loader(parameter, value)
            value = parameter.detach().clone()
        mapped[target] = value
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}."
        if hasattr(layer, "self_attn"):
            mapped[prefix + "self_attn.qkv_proj.weight"] = torch.cat([
                weights[prefix + f"self_attn.{part}_proj.weight"] for part in ("q", "k", "v")])
        if model.config.layers_num_experts[index] == 1:
            mapped[prefix + "feed_forward.gate_up_proj.weight"] = torch.cat([
                weights[prefix + f"feed_forward.{part}_proj.weight"] for part in ("gate", "up")])
    model.load_state_dict(mapped, strict=True)
    for layer in model.model.layers:
        if hasattr(layer, "mamba"):
            weight = layer.mamba.conv1d_weight
            weight.data = fp32_causal_conv_weight(weight).reshape_as(weight)
            layer.mamba.process_weights_after_loading()


def make_workloads(model, inputs, config, *, case=None):
    from fastkernels.hf_coverage.runner import Workload

    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] < 1:
        raise ValueError("Jamba inputs must have shape [batch >= 1, sequence]")
    batch, length = ids.shape
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    steps = 2 if continuation else 1
    if length <= steps:
        raise ValueError("A Jamba workload needs a prompt before its decode tokens")
    prompt = length - steps
    device, dtype = ids.device, model.lm_head.weight.dtype
    width = config.mamba_expand * config.hidden_size
    # L4 remaps each mixer to its compact per-kind slab index. Returned HF
    # state names instead use the original decoder layer indices.
    mixer_indices = [i for i, layer in enumerate(model.model.layers) if hasattr(layer, "mamba")]
    # The existing prefill kernels reserve state slot0 as the null block.
    conv = [torch.zeros(batch + 1, config.mamba_d_conv - 1, width, device=device, dtype=dtype).transpose(1, 2)
            for _ in mixer_indices]
    ssm = [torch.zeros(batch + 1, width, config.mamba_d_state, device=device, dtype=dtype)
           for _ in mixer_indices]
    slots = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * prompt
    metadata = {
        True: JambaMambaMetadata(conv, ssm, slots, is_decode=False, query_start_loc=starts,
                                 has_initial_state=torch.zeros(batch, device=device, dtype=torch.bool)),
        False: JambaMambaMetadata(conv, ssm, slots, is_decode=True),
    }
    block_size = 256
    blocks = (length + block_size - 1) // block_size
    tables = torch.arange(batch * blocks, device=device, dtype=torch.int32).reshape(batch, -1)
    bases = torch.arange(batch, device=device, dtype=torch.int64) * blocks * block_size
    positions = torch.arange(prompt, device=device, dtype=torch.int64)
    prefill_slots = (bases[:, None] + positions).reshape(-1)
    decode_lengths = [torch.full((batch,), prompt + step + 1, device=device, dtype=torch.int32)
                      for step in range(steps)]
    caches = []
    attentions = []
    for index, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            attention = layer.self_attn.attn
            shape = (batch * blocks, block_size, attention.num_kv_heads, attention.head_size)
            attention.k_cache = torch.zeros(shape, device=device, dtype=dtype)
            attention.v_cache = torch.zeros_like(attention.k_cache)
            caches.extend((attention.k_cache, attention.v_cache))
            attentions.append((index, attention, attention.k_cache, attention.v_cache))
    model.audit_states = (conv, ssm)

    def reset():
        for state in conv + ssm + caches:
            state.zero_()
        for _, attention, key, value in attentions:
            attention.k_cache, attention.v_cache = key, value

    def forward(tokens, prefill, step=0):
        context_length = prompt if prefill else prompt + step + 1
        with set_forward_context(
            is_prefill=prefill, mamba_metadata=metadata[prefill],
            cu_seqlens_q=starts if prefill else None, cu_seqlens_k=starts if prefill else None,
            max_seqlen_q=prompt if prefill else 1, max_seqlen_k=context_length,
            slot_mapping=prefill_slots if prefill else bases + prompt + step,
            block_tables=tables, context_lens=None if prefill else decode_lengths[step],
            max_context_len=context_length,
        ):
            hidden = model(tokens.reshape(-1).contiguous())
            logits = model.lm_head(hidden).reshape(batch, -1, config.vocab_size)
        return {"logits": logits}

    def prefill():
        return forward(ids[:, :prompt], True)

    def prepare_decode(step):
        reset()
        prefill()
        for previous in range(step):
            forward(ids[:, prompt + previous:prompt + previous + 1], False, previous)

    def collect(output, sequence_length):
        output = dict(output)
        for index, convolution, recurrent in zip(mixer_indices, conv, ssm):
            # Slot zero is reserved. Native HF keeps one extra oldest sample
            # that its next convolution update discards; compare its live
            # kernel_size-1 history using reference.conv_cache_history.
            output[f"past_key_values.{index}.conv_states"] = convolution[1:]
            output[f"past_key_values.{index}.recurrent_states"] = recurrent[1:]
        for index, attention, key, value in attentions:
            for name, cache in (("key", key), ("value", value)):
                logical = cache.reshape(batch, -1, attention.num_kv_heads, attention.head_size)
                output[f"past_key_values.{index}.{name}"] = logical[:, :sequence_length].transpose(1, 2)
        return output

    workloads = {"prefill": Workload(run=prefill, prepare=reset)}
    for step in range(steps):
        name = f"decode_{step + 1}" if continuation else "decode"
        workloads[name] = Workload(
            run=lambda step=step: forward(ids[:, prompt + step:prompt + step + 1], False, step),
            prepare=lambda step=step: prepare_decode(step),
        )
    if continuation:
        workloads["prefill"].collect = lambda output: collect(output, prompt)
        for step in range(steps):
            workloads[f"decode_{step + 1}"].collect = lambda output, step=step: collect(output, prompt + step + 1)
    return workloads
