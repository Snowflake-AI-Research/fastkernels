"""UMT5 conditional generation with independent relative-position tables per layer."""

from torch import nn

from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L3.t5_block import T5Block
from fastkernels.tasks.baseline.L4.t5_encoder import T5Stack

from .mt5 import _validate_config
from .t5 import T5ForConditionalGeneration, load_state_dict_into, make_workloads


class IndependentBiasBlock(T5Block):
    def __init__(self, config):
        super().__init__(config, has_relative_attention_bias=True)

    def forward(self, hidden_states, mask=None, position_bias=None):
        # UMT5 computes this block's learned table instead of reusing its predecessor's.
        return super().forward(hidden_states, mask=mask, position_bias=None)


class UMT5Encoder(T5Stack):
    def __init__(self, config, shared):
        nn.Module.__init__(self)
        self.embed_tokens = shared
        self.block = nn.ModuleList([IndependentBiasBlock(config) for _ in range(config.num_layers)])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)


class UMT5ForConditionalGeneration(T5ForConditionalGeneration):
    def _build_encoder(self, config):
        return UMT5Encoder(config, self.shared)


def build_from_config(config, device, dtype):
    _validate_config(config, dtype)
    return UMT5ForConditionalGeneration(
        config, output_scale=config.d_model ** -0.5, decoder_bias_per_layer=True,
    ).to(device=device, dtype=dtype).eval()
