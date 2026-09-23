"""Pinned HF Mamba2 causal LM, assembled from the current Mamba2 stack."""

from __future__ import annotations

import copy

import torch

from fastkernels.infra.context import set_forward_context
from fastkernels.infra.mamba_state import Mamba2Metadata, MambaStateManager, build_chunk_metadata
from fastkernels.hf_coverage.patches.mamba_conv_weight import fp32_causal_conv_weight
from fastkernels.hf_coverage.patches.mamba2_precision import mamba2_forward
from fastkernels.hf_coverage.models.mamba import recurrent_workloads
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L2.mamba2_mixer import Mixer2RMSNormGated
from fastkernels.tasks.baseline.L4.mamba2 import Mamba2ForCausalLM


class HFMamba2ForCausalLM(Mamba2ForCausalLM):
    """Keep HF's FP32 residuals and its cast before each block's norm."""

    def forward(self, input_ids, positions=None):
        hidden = self.backbone.embeddings(input_ids)
        for layer in self.backbone.layers:
            residual = hidden.float() if self.config.residual_in_fp32 else hidden
            hidden = layer.norm(hidden.to(layer.norm.weight.dtype))
            hidden = residual + mamba2_forward(layer.mixer, hidden)
        hidden = self.backbone.norm_f(hidden)
        return self.compute_logits(hidden.to(self.lm_head.embedding_op.emb.weight.dtype))


def build_from_config(config, device, dtype):
    config = copy.copy(config)
    config.intermediate_size = int(config.expand) * int(config.hidden_size)
    if config.intermediate_size != int(config.num_heads) * int(config.head_dim):
        raise ValueError("Mamba2 requires expand * hidden_size == num_heads * head_dim")
    model = HFMamba2ForCausalLM(config)
    # HF's gated norm reduces across all channels, independently of SSM groups.
    for layer in model.backbone.layers:
        layer.norm = RMSNormNative(config.hidden_size, eps=config.layer_norm_epsilon)
        layer.mixer.norm = Mixer2RMSNormGated(
            full_hidden_size=config.intermediate_size,
            full_n_groups=1,
            use_rms_norm=config.rms_norm,
            eps=config.layer_norm_epsilon,
        )
    model.backbone.norm_f = RMSNormNative(config.hidden_size, eps=config.layer_norm_epsilon)
    model.to(device=device, dtype=dtype).eval()
    # The scan's A is -exp(A_log), computed from HF's rounded parameter in FP32.
    for layer in model.backbone.layers:
        layer.mixer.A.data = layer.mixer.A.data.float()
    return model


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
        for name in ("in_proj", "conv1d", "out_proj"):
            operation = getattr(mixer, name)
            _copy(operation.weight, state_dict[prefix + name + ".weight"])
            if operation.bias is not None:
                _copy(operation.bias, state_dict[prefix + name + ".bias"])
        _copy(mixer.A, state_dict[prefix + "A_log"])
        for name in ("D", "dt_bias"):
            _copy(getattr(mixer, name), state_dict[prefix + name])
        _copy(mixer.norm.weight, state_dict[prefix + "norm.weight"])
        mixer.conv_weights = fp32_causal_conv_weight(mixer.conv1d.weight)


def make_workloads(model, inputs, config, *, case=None):
    ids = inputs["input_ids"]
    batch, length = ids.shape
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    prompt_length = length - (2 if continuation else 1)
    if prompt_length < 1:
        raise ValueError("A recurrent workload needs a prompt before its decode tokens")
    state = MambaStateManager(
        num_hidden_layers=config.num_hidden_layers,
        conv_dim=config.expand * config.hidden_size + 2 * config.n_groups * config.state_size,
        ssm_state_shape=(config.num_heads, config.head_dim, config.state_size),
        conv_kernel=config.conv_kernel,
        num_slots=batch + 1,
        dtype=next(model.parameters()).dtype,
        device=ids.device,
    )
    # Slot zero is the kernels' null block, including during prefill.
    slots = torch.arange(1, batch + 1, device=ids.device, dtype=torch.int32)
    starts = [index * prompt_length for index in range(batch + 1)]
    prefill_meta = Mamba2Metadata(
        num_prefill_tokens=batch * prompt_length,
        num_prefills=batch,
        query_start_loc_p=torch.tensor(starts, device=ids.device, dtype=torch.int32),
        state_indices_p=slots,
        has_initial_states_p=torch.zeros(batch, device=ids.device, dtype=torch.bool),
        chunk_size=config.chunk_size,
    )
    (
        prefill_meta.cu_chunk_seqlen_p,
        prefill_meta.seq_idx_p,
        prefill_meta.last_chunk_indices_p,
    ) = build_chunk_metadata(prefill_meta.query_start_loc_p, config.chunk_size, host_qsl=starts)
    decode_meta = Mamba2Metadata(num_decode_tokens=batch, num_decodes=batch, state_indices_d=slots)

    def forward(token_ids, is_prefill):
        metadata = prefill_meta if is_prefill else decode_meta
        with set_forward_context(is_prefill=is_prefill, mamba_state=state, mamba_metadata=metadata):
            logits = model(token_ids.reshape(-1).contiguous())
        return {"logits": logits.reshape(batch, -1, logits.shape[-1])}

    return recurrent_workloads(ids, state, forward, continuation=continuation)
