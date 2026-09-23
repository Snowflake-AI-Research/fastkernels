"""GPT-OSS uses the existing quantized L4 and its packed-weight loaders."""

import torch
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L4.gpt_oss import GptOssConfig, GptOssForCausalLM
from .llama import make_workloads


class HfGptOss(GptOssForCausalLM):
    def compute_logits(self, hidden_states):
        # HF exposes the projection dtype; the serving L4 upcasts these logits.
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (_tp_size() != 1 or dtype != torch.bfloat16
            or config.rope_parameters['rope_type'] != 'yarn'
            or config.layer_types != ['sliding_attention' if i % 2 == 0 else 'full_attention'
                                      for i in range(config.num_hidden_layers)]
            or config.quantization_config.get('quant_method') != 'mxfp4'):
        raise ValueError('Selected GPT-OSS path requires native MXFP4, BF16, YaRN and alternating attention')
    model = HfGptOss(GptOssConfig._from_hf(config)).to(device=device, dtype=dtype)
    for layer in model.model.layers:
        # Existing Triton MXFP4 matches the native HF expert on identical inputs;
        # the automatically selected TensorRT backend disagrees for this case.
        layer.mlp.use_trtllm = False
        layer.self_attn.sinks.data = layer.self_attn.sinks.data.float()
        layer.mlp.w13_bias.data = layer.mlp.w13_bias.data.float()
        layer.mlp.w2_bias.data = layer.mlp.w2_bias.data.float()
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    expert_names = {'gate_up_proj_blocks': 'w13_weight', 'gate_up_proj_scales': 'w13_weight_scale',
                    'gate_up_proj_bias': 'w13_bias', 'down_proj_blocks': 'w2_weight',
                    'down_proj_scales': 'w2_weight_scale', 'down_proj_bias': 'w2_bias'}
    loaded = set()
    for name, value in state_dict.items():
        if '.mlp.experts.' in name:
            prefix, field = name.split('.mlp.experts.')
            parameter = model.get_parameter(prefix + '.mlp.' + expert_names[field])
            parameter.weight_loader(parameter, value)
            loaded.add(prefix + '.mlp.' + expert_names[field])
            continue
        target = name.replace('model.embed_tokens.weight', 'model.embed_tokens.embedding_op.emb.weight')
        target = target.replace('lm_head.weight', 'lm_head.embedding_op.emb.weight')
        shard = None
        for part in ('q', 'k', 'v'):
            if f'.{part}_proj.' in target:
                target = target.replace(f'.{part}_proj.', '.qkv_proj.')
                shard = part
                break
        parameter = model.get_parameter(target)
        if shard is None:
            if parameter.shape != value.shape:
                raise ValueError(f'GPT-OSS weight shape mismatch: {name}')
            parameter.copy_(value)
        else:
            parameter.weight_loader(parameter, value, shard)
        loaded.add(target)
    expected = set(dict(model.named_parameters()))
    if loaded != expected:
        raise KeyError(f'GPT-OSS weight coverage mismatch: missing={sorted(expected-loaded)}, extra={sorted(loaded-expected)}')
