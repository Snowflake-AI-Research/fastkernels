"""Qwen3.5 multimodal hybrid attention with the existing shared-expert MoE."""

from types import SimpleNamespace

import torch

from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from . import qwen3_5


def build_from_config(config, device, dtype):
    model = qwen3_5.build_from_config(config, device, dtype)
    text = config.text_config
    for layer in model.model.layers:
        layer.mlp = SharedExpertMoE(
            hidden_size=text.hidden_size,
            num_experts=text.num_experts,
            top_k=text.num_experts_per_tok,
            moe_intermediate_size=text.moe_intermediate_size,
            routing="softmax",
            renormalize=True,
            shared_expert_intermediate_size=text.shared_expert_intermediate_size,
            shared_expert_gate=True,
        ).to(device=device, dtype=dtype)
        # Keep the existing separate router/expert path. The monolithic
        # Blackwell path fails during asynchronous execution for this case.
        layer.mlp.use_trtllm = False
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    # The dense sibling's loader already handles the identical vision, GDN,
    # full attention and normalization tensors. Present each shared expert as
    # its dense MLP solely for weight loading; no forward modules are replaced.
    remaining = dict(state)
    dense_weights, views = {}, []
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.language_model.layers.{index}.mlp."
        moe = layer.mlp
        for parameter, name in (
            (moe.gate.weight, "gate.weight"),
            (moe.w13, "experts.gate_up_proj"),
            (moe.w2, "experts.down_proj"),
            (moe.shared_expert_gate.weight, "shared_expert_gate.weight"),
        ):
            value = remaining.pop(prefix + name)
            if value.shape != parameter.shape:
                raise ValueError(f"Qwen3.5 MoE mapping shape mismatch: {prefix + name}")
            parameter.copy_(value)
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            dense_weights[prefix + name] = remaining.pop(prefix + "shared_expert." + name)
        view = SimpleNamespace(
            layer_type=layer.layer_type,
            input_layernorm=layer.input_layernorm,
            post_attention_layernorm=layer.post_attention_layernorm,
            mlp=moe.shared_expert,
        )
        if layer.layer_type == "linear_attention":
            view.linear_attn = layer.linear_attn
        else:
            view.self_attn = layer.self_attn
        views.append(view)
    proxy = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=model.model.embed_tokens,
                              norm=model.model.norm, layers=views),
        visual=model.visual,
        lm_head=model.lm_head,
    )
    qwen3_5.load_state_dict_into(proxy, {**remaining, **dense_weights}, config)
    for layer in model.model.layers:
        layer.mlp.process_weights_after_loading()


make_workloads = qwen3_5.make_workloads
