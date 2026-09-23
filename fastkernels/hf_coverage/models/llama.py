"""Llama causal-LM adapter using the existing L4 model and paged attention."""

from __future__ import annotations

import torch

from fastkernels.hf_coverage.runner import Workload
from fastkernels.infra.context import set_forward_context
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L4.llama import (
    LlamaConfig,
    LlamaForCausalLM,
)


def build_from_config(config, device, dtype) -> LlamaForCausalLM:
    """Build the bias-free, untied Llama-2 graph selected by the HF example."""
    if _tp_size() != 1:
        raise ValueError("The Llama coverage workload requires tensor parallel size 1")
    if (
        config.hidden_act != "silu"
        or config.attention_bias
        or config.mlp_bias
        or config.tie_word_embeddings
    ):
        raise ValueError("The Llama pilot requires SiLU, bias-free layers, and an untied head")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The Llama-2 pilot requires default RoPE")
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError("The Llama-2 pilot preserves multi-head attention")
    if config.head_dim != config.hidden_size // config.num_attention_heads:
        raise ValueError("The Llama-2 pilot preserves the derived head dimension")

    fk_config = LlamaConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype,
        qkv_bias=False,
    )
    return LlamaForCausalLM(fk_config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config) -> None:
    """Map HF names to the existing packed QKV and gate/up weight loaders."""
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        expected.update(prefix + name for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ))
    if set(state_dict) != expected:
        missing = sorted(expected - set(state_dict))
        extra = sorted(set(state_dict) - expected)
        raise KeyError(f"Llama state mismatch: missing={missing}, extra={extra}")

    def copy_into(parameter, name):
        source = state_dict[name]
        if parameter.shape != source.shape:
            raise ValueError(f"Llama weight shape mismatch for {name}: {source.shape} != {parameter.shape}")
        parameter.copy_(source)

    backbone = model.model
    copy_into(backbone.embed_tokens.embedding_op.emb.weight, "model.embed_tokens.weight")
    copy_into(model.lm_head.embedding_op.emb.weight, "lm_head.weight")
    copy_into(backbone.norm.weight, "model.norm.weight")
    for index, layer in enumerate(backbone.layers):
        prefix = f"model.layers.{index}."
        copy_into(layer.input_layernorm.weight, prefix + "input_layernorm.weight")
        copy_into(layer.post_attention_layernorm.weight, prefix + "post_attention_layernorm.weight")
        attention = layer.self_attn
        qkv = attention.qkv_proj.weight
        for shard in ("q", "k", "v"):
            source = state_dict[prefix + f"self_attn.{shard}_proj.weight"]
            heads = config.num_attention_heads if shard == "q" else config.num_key_value_heads
            if source.shape != (heads * config.head_dim, config.hidden_size):
                raise ValueError(f"Llama {shard} projection has an incompatible shape")
            qkv.weight_loader(qkv, source, shard)
        copy_into(attention.o_proj.weight, prefix + "self_attn.o_proj.weight")
        gate_up = layer.mlp.gate_up_proj.weight
        for shard, name in enumerate(("gate", "up")):
            source = state_dict[prefix + f"mlp.{name}_proj.weight"]
            if source.shape != (config.intermediate_size, config.hidden_size):
                raise ValueError(f"Llama {name} projection has an incompatible shape")
            gate_up.weight_loader(gate_up, source, shard)
        copy_into(layer.mlp.down_proj.weight, prefix + "mlp.down_proj.weight")


def make_workloads(model, inputs, config, *, cached_decode=True, attentions=None,
                   output_projection=None, case=None, cache_windows=None) -> dict[str, Workload]:
    """Use bounded caches; the pilot adds two decode steps and cache comparison."""
    from fastkernels.tasks.baseline.L2.mla_attention_impl import MLAAttention
    from .qwen2_precision import DenseCachedAttention

    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 2:
        raise ValueError("Llama inputs must have shape [batch >= 1, sequence >= 2]")
    if input_ids.device.type != "cuda":
        raise ValueError("The Llama workloads require CUDA input_ids")
    batch, total_length = input_ids.shape
    if total_length > config.max_position_embeddings:
        raise ValueError("Llama sequence exceeds the configured RoPE cache")
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    decode_steps = 2 if continuation else int(cached_decode)
    if continuation and (not cached_decode or total_length < 3):
        raise ValueError("Two-step continuation requires a prompt and two supplied tokens")
    prompt_length = total_length - decode_steps
    device = input_ids.device
    parameter = next(model.parameters())
    if parameter.device != device:
        raise ValueError("Llama model and input_ids must be on the same device")
    if attentions is None:
        attentions = [layer.self_attn.attn for layer in model.model.layers]
    if not attentions:
        raise ValueError("The paged-cache workload requires at least one attention layer")
    if cache_windows is not None and len(cache_windows) != len(attentions):
        raise ValueError("Logical cache windows must specify each attention layer")
    # Dense attention uses lengths only for Python cache slicing. Keep that
    # metadata on the host instead of synchronizing a GPU scalar in every layer.
    # Paged attention kernels still receive device lengths.
    host_context_lengths = all(isinstance(attention, DenseCachedAttention) for attention in attentions)
    block_sizes = [64 if isinstance(attention, MLAAttention) else attention._block_size
                   for attention in attentions]
    block_size = block_sizes[0]
    if any(size != block_size for size in block_sizes):
        raise ValueError("Llama layers must use the same page size")
    blocks_per_request = (total_length + block_size - 1) // block_size
    num_blocks = batch * blocks_per_request
    caches = []
    for attention in attentions:
        if isinstance(attention, MLAAttention):
            if continuation:
                raise ValueError("Logical cache comparison requires ordinary key/value attention")
            if attention.kv_cache_dtype != "auto":
                raise ValueError("The MLA coverage workload uses the native unquantized latent cache")
            # MLA stores compressed content and rotary keys in one shared page.
            caches.append(torch.empty((num_blocks, block_size, attention._head_dim),
                                      device=device, dtype=parameter.dtype))
            continue
        if attention.kv_layout == "NHD":
            shape = (num_blocks, block_size, attention.num_kv_heads, attention.head_size)
        elif attention.kv_layout == "HND":
            shape = (num_blocks, attention.num_kv_heads, block_size, attention.head_size)
        else:
            raise ValueError(f"Unknown Llama KV layout: {attention.kv_layout}")
        caches.append(torch.empty((2, *shape), device=device, dtype=parameter.dtype))

    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(batch, -1)
    request_ids = torch.arange(batch, device=device, dtype=torch.int32)
    slot_bases = request_ids.to(torch.int64) * (blocks_per_request * block_size)
    prompt_positions = torch.arange(prompt_length, device=device, dtype=torch.int64)
    prompt_slots = (slot_bases[:, None] + prompt_positions[None, :]).reshape(-1)
    cu_prompt = torch.arange(batch + 1, device=device, dtype=torch.int32) * prompt_length
    prompt_metadata = dict(
        cu_seqlens_q=cu_prompt,
        cu_seqlens_k=cu_prompt,
        max_seqlen_q=prompt_length,
        max_seqlen_k=prompt_length,
        slot_mapping=prompt_slots,
        block_tables=block_tables,
        req_id_per_token=request_ids.repeat_interleave(prompt_length),
    )
    prompt_positions = prompt_positions.repeat(batch)
    prompt_ids = input_ids[:, :prompt_length]

    @torch.inference_mode()
    def prepare_prefill():
        # Reattach this workload's bounded buffers if another workload was used.
        for attention, cache in zip(attentions, caches):
            cache.zero_()
            if isinstance(attention, MLAAttention):
                attention.k_cache = attention.v_cache = cache
            else:
                attention.k_cache, attention.v_cache = cache[0], cache[1]

    def full_logits(hidden_states, phase_length):
        if output_projection is not None:
            return output_projection(hidden_states.view(batch, phase_length, -1))
        # ParallelLMHead.project selects only each prompt's last token. The
        # pinned HF task returns all logits in the model dtype; its head has
        # no additional postprocessing. Reuse the head's existing L1 Matmul.
        logits = model.lm_head.linear_op(
            hidden_states, model.lm_head.embedding_op.emb.weight,
        )
        return {"logits": logits.view(batch, phase_length, config.vocab_size)}

    @torch.inference_mode()
    def prefill():
        with set_forward_context(is_prefill=True, **prompt_metadata):
            hidden_states = model.model(prompt_ids.reshape(-1), prompt_positions)
            return full_logits(hidden_states, prompt_length)

    if continuation:
        def cache_collector(length):
            def collect(output):
                output = dict(output)
                for index, (attention, cache) in enumerate(zip(attentions, caches)):
                    for kind, pages in zip(("key", "value"), cache):
                        if attention.kv_layout == "HND":
                            pages = pages.permute(0, 2, 1, 3)
                        # Each request owns consecutive pages. Drop unused slots
                        # and expose HF's [batch, heads, sequence, head_dim] view.
                        logical = pages.reshape(
                            batch, blocks_per_request * block_size,
                            attention.num_kv_heads, attention.head_size,
                        )[:, :length].transpose(1, 2)
                        if cache_windows is not None and cache_windows[index] is not None:
                            # Match HF's retained sliding state after execution.
                            # Physical storage and timed attention are unchanged.
                            logical = logical[..., 1 - cache_windows[index]:, :]
                        output[f"past_key_values.{index}.{kind}"] = logical
                return output
            return collect

        decode_calls = []
        for step in range(decode_steps):
            position = prompt_length + step
            lengths = torch.full((batch,), position + 1, device=device, dtype=torch.int32)
            metadata = dict(
                slot_mapping=slot_bases + position,
                context_lens=(torch.full((batch,), position + 1, device="cpu", dtype=torch.int32)
                              if host_context_lengths else lengths),
                block_tables=block_tables,
                max_context_len=position + 1,
                decode_context_lens=lengths,
                decode_block_tables=block_tables,
                decode_max_context_len=position + 1,
                req_id_per_token=request_ids,
            )
            positions = torch.full((batch,), position, device=device, dtype=torch.int64)
            token_ids = input_ids[:, position]

            @torch.inference_mode()
            def run_decode(metadata=metadata, positions=positions, token_ids=token_ids):
                with set_forward_context(is_prefill=False, **metadata):
                    hidden = model.model(token_ids, positions)
                    return full_logits(hidden, 1)

            decode_calls.append(run_decode)

        workloads = {
            "prefill": Workload(run=prefill, prepare=prepare_prefill,
                                collect=cache_collector(prompt_length)),
        }
        for step, run_decode in enumerate(decode_calls):
            @torch.inference_mode()
            def prepare_step(step=step):
                prepare_prefill()
                with set_forward_context(is_prefill=True, **prompt_metadata):
                    model.model(prompt_ids.reshape(-1), prompt_positions)
                for prior in decode_calls[:step]:
                    prior()

            workloads[f"decode_{step + 1}"] = Workload(
                run=run_decode, prepare=prepare_step,
                collect=cache_collector(prompt_length + step + 1),
            )
        return workloads

    if not cached_decode:
        return {"forward": Workload(run=prefill, prepare=prepare_prefill)}

    context_lengths = torch.full((batch,), total_length, device=device, dtype=torch.int32)
    decode_metadata = dict(
        slot_mapping=slot_bases + prompt_length,
        context_lens=(torch.full((batch,), total_length, device="cpu", dtype=torch.int32)
                      if host_context_lengths else context_lengths),
        block_tables=block_tables,
        max_context_len=total_length,
        decode_context_lens=context_lengths,
        decode_block_tables=block_tables,
        decode_max_context_len=total_length,
        req_id_per_token=request_ids,
    )
    decode_positions = torch.full((batch,), prompt_length, device=device, dtype=torch.int64)
    decode_ids = input_ids[:, -1]

    @torch.inference_mode()
    def prepare_decode():
        prepare_prefill()
        # Rebuild the prompt state outside timing. Decode overwrites the same
        # final slot on every run and its context length never increments.
        with set_forward_context(is_prefill=True, **prompt_metadata):
            model.model(prompt_ids.reshape(-1), prompt_positions)

    @torch.inference_mode()
    def decode():
        with set_forward_context(is_prefill=False, **decode_metadata):
            hidden_states = model.model(decode_ids, decode_positions)
            return full_logits(hidden_states, 1)

    return {
        "prefill": Workload(run=prefill, prepare=prepare_prefill),
        "decode": Workload(run=decode, prepare=prepare_decode),
    }
