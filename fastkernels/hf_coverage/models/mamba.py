"""Pinned HF Mamba causal LM using the existing Mamba-v1 computation stack."""

from __future__ import annotations

import torch

from fastkernels.hf_coverage.patches.mamba_conv_weight import fp32_causal_conv_weight
from fastkernels.infra.context import set_forward_context
from fastkernels.infra.mamba_state import (
    MambaMetadata,
    MambaStateManager,
    compute_causal_conv1d_metadata,
)
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L4.mamba import MambaForCausalLM


class HFMambaForCausalLM(MambaForCausalLM):
    def forward(self, input_ids, positions=None):
        hidden = self.backbone.embeddings(input_ids)
        for layer in self.backbone.layers:
            residual = hidden.float() if self.config.residual_in_fp32 else hidden
            hidden = layer.norm(hidden.to(layer.norm.weight.dtype))
            hidden = residual + layer.mixer(hidden)
        hidden = self.backbone.norm_f(hidden)
        return self.compute_logits(hidden.to(self.lm_head.embedding_op.emb.weight.dtype))


def build_mamba(config, device, dtype, mixer_factory=None):
    model = HFMambaForCausalLM(config)
    for index, layer in enumerate(model.backbone.layers):
        if mixer_factory is not None:
            layer.mixer = mixer_factory(config, index)
        layer.mixer.activation = config.hidden_act
        # HF rounds normalized activations before multiplying by norm weights.
        layer.norm = RMSNormNative(config.hidden_size, eps=config.layer_norm_epsilon)
    model.backbone.norm_f = RMSNormNative(config.hidden_size, eps=config.layer_norm_epsilon)
    model.to(device=device, dtype=dtype).eval()
    for layer in model.backbone.layers:
        layer.mixer.A.data = layer.mixer.A.data.float()
    return model


def build_from_config(config, device, dtype):
    from fastkernels.hf_coverage.patches.mamba_conv_precision import MambaConvPrecision

    return build_mamba(config, device, dtype, mixer_factory=MambaConvPrecision)


def _copy(parameter, value):
    loader = getattr(parameter, "weight_loader", None)
    if loader is None:
        parameter.data.copy_(value)
    else:
        loader(parameter, value)


def load_state_dict_into(model, state_dict, config):
    _copy(model.backbone.embeddings.embedding_op.emb.weight, state_dict["backbone.embeddings.weight"])
    _copy(model.backbone.norm_f.weight, state_dict["backbone.norm_f.weight"])
    _copy(model.lm_head.embedding_op.emb.weight, state_dict.get("lm_head.weight", state_dict["backbone.embeddings.weight"]))
    for index, layer in enumerate(model.backbone.layers):
        prefix = f"backbone.layers.{index}."
        _copy(layer.norm.weight, state_dict[prefix + "norm.weight"])
        mixer = layer.mixer
        prefix += "mixer."
        for name in ("in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"):
            operation = getattr(mixer, name)
            _copy(operation.weight, state_dict[prefix + name + ".weight"])
            if operation.bias is not None:
                _copy(operation.bias, state_dict[prefix + name + ".bias"])
        _copy(mixer.A, state_dict[prefix + "A_log"])
        _copy(mixer.D, state_dict[prefix + "D"])
        # The existing mixer reads this parameter directly on each call. Keep
        # its loaded, rounded values in the verified FP32 weight representation.
        mixer.conv1d.weight.data = fp32_causal_conv_weight(mixer.conv1d.weight).unsqueeze(1)


def recurrent_workloads(ids, state, forward, *, continuation):
    """Exercise the recurrent state; expose only its live sequence slots."""
    from fastkernels.hf_coverage.runner import Workload

    steps = 2 if continuation else 1
    prompt_length = ids.shape[1] - steps

    def reset():
        for tensor in state.conv_states + state.ssm_states:
            tensor.zero_()

    def prefill():
        return forward(ids[:, :prompt_length], True)

    def collect(output):
        output = dict(output)
        for index, (conv, recurrent) in enumerate(zip(state.conv_states, state.ssm_states)):
            # Slot zero is reserved. HF stores channels before time; both
            # implementations retain the same last kernel_size-1 input values
            # needed for the next convolution, despite different cache layouts.
            output[f"cache_params.{index}.conv_states"] = conv[1:].transpose(-1, -2)
            output[f"cache_params.{index}.recurrent_states"] = recurrent[1:]
        return output

    def prepare_step(index):
        reset()
        prefill()
        for previous in range(index):
            forward(ids[:, prompt_length + previous:prompt_length + previous + 1], False)

    workloads = {"prefill": Workload(run=prefill, prepare=reset)}
    for index in range(steps):
        name = f"decode_{index + 1}" if continuation else "decode"
        workloads[name] = Workload(
            run=lambda index=index: forward(ids[:, prompt_length + index:prompt_length + index + 1], False),
            prepare=lambda index=index: prepare_step(index),
        )
    if continuation:
        for workload in workloads.values():
            workload.collect = collect
    return workloads


def make_workloads(model, inputs, config, *, case=None):
    ids = inputs["input_ids"]
    batch, length = ids.shape
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    prompt_length = length - (2 if continuation else 1)
    if prompt_length < 1:
        raise ValueError("A recurrent workload needs a prompt before its decode tokens")
    state = MambaStateManager(
        num_hidden_layers=config.num_hidden_layers,
        conv_dim=config.intermediate_size,
        ssm_state_shape=(config.intermediate_size, config.state_size),
        conv_kernel=config.conv_kernel,
        num_slots=batch + 1,
        dtype=model.backbone.embeddings.embedding_op.emb.weight.dtype,
        device=ids.device,
    )
    slots = torch.arange(1, batch + 1, device=ids.device, dtype=torch.int32)
    starts = [index * prompt_length for index in range(batch + 1)]
    metadata = MambaMetadata(
        num_prefill_tokens=batch * prompt_length,
        num_prefills=batch,
        query_start_loc_p=torch.tensor(starts, device=ids.device, dtype=torch.int32),
        state_indices_p=slots,
        has_initial_states_p=torch.zeros(batch, device=ids.device, dtype=torch.bool),
    )
    metadata.nums_dict, metadata.batch_ptr, metadata.token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        metadata.query_start_loc_p, seqlens_cpu=[prompt_length] * batch,
    )
    decode_metadata = MambaMetadata(num_decode_tokens=batch, num_decodes=batch, state_indices_d=slots)

    def forward(token_ids, is_prefill):
        selected_metadata = metadata if is_prefill else decode_metadata
        with set_forward_context(is_prefill=is_prefill, mamba_state=state, mamba_metadata=selected_metadata):
            logits = model(token_ids.reshape(-1).contiguous())
        return {"logits": logits.reshape(batch, -1, logits.shape[-1])}

    return recurrent_workloads(ids, state, forward, continuation=continuation)
