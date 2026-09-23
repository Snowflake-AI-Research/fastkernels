"""IBertForMaskedLM with the checkpoint's floating-point quant_mode=False path."""

import torch

from fastkernels.tasks.baseline.L2.sam3_prompt_encoder import LayerNorm2d

from .roberta import RobertaForMaskedLM, make_workloads


class IBertLayerNorm(LayerNorm2d):
    """Apply unchanged SAM3 input-dtype normalization to token features."""

    def forward(self, hidden_states):
        shape = hidden_states.shape
        return super().forward(hidden_states.reshape(-1, shape[-1], 1, 1)).reshape(shape)


def build_from_config(config, device, dtype):
    if config.quant_mode or config.hidden_act != "gelu":
        raise ValueError("IBERT coverage preserves the checkpoint's unquantized masked-LM path")
    model = RobertaForMaskedLM(config)
    model.roberta.embeddings.LayerNorm = IBertLayerNorm(config.hidden_size, config.layer_norm_eps)
    for layer in model.roberta.encoder.layer:
        layer.attention.output.LayerNorm = IBertLayerNorm(config.hidden_size, config.layer_norm_eps)
        layer.output.LayerNorm = IBertLayerNorm(config.hidden_size, config.layer_norm_eps)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("roberta."):
            source = source.replace("roberta.", "ibert.", 1)
        elif source.startswith("lm_head."):
            source = source.replace(".LayerNorm.", ".layer_norm.")
        if ".qkv." in source:
            weight = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weight = state_dict[source]
        parameter.copy_(weight)
