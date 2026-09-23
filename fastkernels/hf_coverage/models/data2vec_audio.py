"""Data2VecAudioModel with its full five-layer positional convolution stack."""

from torch import nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm

from .unispeech import PostNormWaveformModel, load_state_dict_into, make_workloads
from .wav2vec2 import FeatureConv


class PositionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel = config.conv_pos_kernel_size
        self.conv = Conv1dNative(
            config.hidden_size, config.hidden_size, kernel,
            padding=kernel // 2, groups=config.num_conv_pos_embedding_groups,
        )
        self.remove_last = kernel % 2 == 0
        self.layer_norm = LayerNorm(
            config.hidden_size, eps=1e-5, elementwise_affine=False, promote_fp32=False
        )
        self.activation = GELU()

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        if self.remove_last:
            hidden_states = hidden_states[:, :, :-1]
        hidden_states = self.layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        return self.activation(hidden_states)


class PositionStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(PositionLayer(config) for _ in range(config.num_conv_pos_embeddings))

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states.transpose(1, 2)


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.feat_extract_activation != "gelu" or config.add_adapter:
        raise ValueError("This case requires Data2VecAudio's default GELU encoder without an adapter")
    if config.output_attentions or config.output_hidden_states:
        raise ValueError("This case returns the ordinary final model outputs")
    model = PostNormWaveformModel(
        config,
        [FeatureConv(config, index) for index in range(len(config.conv_dim))],
        PositionStack(config),
    )
    # The encoder's optional varlen dependency disables global cuDNN SDPA.
    # Select the existing backend explicitly to preserve native HF rounding.
    for layer in model.layers:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
    return model.to(device=device, dtype=dtype).eval()
