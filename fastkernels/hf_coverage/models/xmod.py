"""XmodForMaskedLM with the documented English adapter and complete adapter weights."""

from torch import nn

from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderOutput
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .roberta import RobertaForMaskedLM, load_state_dict_into as load_roberta_weights, make_workloads


class XmodOutput(EncoderOutput):
    def __init__(self, config):
        super().__init__(config)
        self.language = config.default_language
        self.adapter_modules = nn.ModuleDict({
            language: VitEncoderMlp(
                config.hidden_size, config.hidden_size // config.adapter_reduction_factor,
                act_approximate="none", bias=True,
            )
            for language in config.languages
        })

    def forward(self, hidden_states, input_tensor):
        normalized = super().forward(hidden_states, input_tensor)
        adapted = self.adapter_modules[self.language](normalized) + normalized
        return self.LayerNorm(adapted)


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or config.pre_norm or config.adapter_layer_norm
            or not config.adapter_reuse_layer_norm or not config.ln_before_adapter
            or config.default_language != "en_XX" or config.default_language not in config.languages
            or config.is_decoder or config.add_cross_attention):
        raise ValueError("X-MOD coverage preserves the documented English adapter with default normalization flags")
    model = RobertaForMaskedLM(config)
    for layer in model.roberta.encoder.layer:
        layer.output = XmodOutput(config)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    renamed = {name.replace(".dense1.", ".fc1.").replace(".dense2.", ".fc2."): value
               for name, value in state_dict.items()}
    load_roberta_weights(model, renamed, config)
