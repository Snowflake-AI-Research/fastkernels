"""HYV3 routed decoder, including the checkpoint's enabled router-logit outputs."""

from copy import copy

from fastkernels.hf_coverage.models.dots1 import build_from_config as build_dots
from fastkernels.hf_coverage.models.dots1 import load_state_dict_into as load_dots
from fastkernels.hf_coverage.models.llama import make_workloads as llama_workloads
from fastkernels.hf_coverage.runner import Workload


def carrier(config):
    if config.enable_moe_fp32_combine or config.mlp_layer_types != ["dense"] + ["sparse"] * (config.num_hidden_layers - 1):
        raise ValueError("Selected HYV3 uses one dense layer and native-dtype shared expert addition")
    c = copy(config)
    c.layer_types = ["full_attention"] * c.num_hidden_layers
    c.first_k_dense_replace = 1
    c.n_routed_experts, c.n_shared_experts = c.num_experts, c.num_shared_experts
    c.n_group = c.topk_group = 1
    c.norm_topk_prob = True
    c.routed_scaling_factor = c.router_scaling_factor
    return c


def record_router(module, args, output):
    module.last_router_logits = output


def build_from_config(config, device, dtype):
    model = build_dots(carrier(config), device, dtype)
    for layer in model.model.layers[1:]:
        bias = layer.mlp.gate.e_score_correction_bias
        bias.data = bias.data.float()
        layer.mlp.gate_linear.register_forward_hook(record_router)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped = {key.replace(".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias"): value
              for key, value in state_dict.items()}
    load_dots(model, mapped, carrier(config))


def make_workloads(model, inputs, config):
    workloads = llama_workloads(model, inputs, config)
    if not config.output_router_logits:
        return workloads
    def with_routers(workload):
        outputs = workload.run()
        outputs.update({f"router_logits.{index}": layer.mlp.gate_linear.last_router_logits
                        for index, layer in enumerate(model.model.layers[1:])})
        return outputs
    return {name: Workload(run=lambda w=workload: with_routers(w), prepare=workload.prepare)
            for name, workload in workloads.items()}
