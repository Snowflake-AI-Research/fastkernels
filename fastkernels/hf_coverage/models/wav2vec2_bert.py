"""W2V-BERT's feature-input Conformer, learned relative keys and causal convolution."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from .wav2vec2_conformer import ConformerLayer, GLU, make_workloads


class RelativeKeyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        self.left, self.right = config.left_max_position_embeddings, config.right_max_position_embeddings
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.distance_embedding = Embedding(self.left + self.right + 1, self.width)
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, positions=None, mask=None):
        batch, length, _ = hidden.shape
        q, k, v = (getattr(self, name)(hidden).reshape(batch, length, self.heads, self.width)
                   .transpose(1, 2) for name in ("linear_q", "linear_k", "linear_v"))
        scores = self.bmm(q.reshape(-1, length, self.width),
                          k.reshape(-1, length, self.width).transpose(1, 2)) * self.width**-0.5
        axis = torch.arange(length, device=hidden.device)
        distance = (axis[None, :] - axis[:, None]).clamp(-self.left, self.right) + self.left
        relative = self.distance_embedding(distance).to(q.dtype)
        # Batch the contractions by query position, sharing its distance table
        # across all heads and examples as in HF's einsum.
        query = q.permute(2, 0, 1, 3).reshape(length, batch * self.heads, self.width)
        relative_scores = self.bmm(query, relative.transpose(1, 2))
        relative_scores = relative_scores.reshape(length, batch, self.heads, length).permute(1, 2, 0, 3)
        scores = scores.reshape(batch, self.heads, length, length) + relative_scores * self.width**-0.5
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        output = self.bmm(self.softmax(scores).reshape(-1, length, length), v.reshape(-1, length, self.width))
        return self.linear_out(output.reshape(batch, self.heads, length, self.width).transpose(1, 2)
                               .reshape(batch, length, -1))


class CausalConvolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, self.kernel = config.hidden_size, config.conv_depthwise_kernel_size
        self.layer_norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.pointwise_conv1 = Conv1dNative(width, 2 * width, 1, bias=False)
        self.glu = GLU(precise_sigmoid=True)
        self.depthwise_conv = Conv1dNative(width, width, self.kernel, groups=width, bias=False)
        self.depthwise_layer_norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.activation = SiLU()
        self.pointwise_conv2 = Conv1dNative(width, width, 1, bias=False)

    def forward(self, hidden, mask=None):
        hidden = self.layer_norm(hidden)
        if mask is not None:
            hidden = hidden.masked_fill(~mask[:, :, None], 0)
        hidden = self.glu(self.pointwise_conv1(hidden.transpose(1, 2)))
        hidden = torch.cat((hidden.new_zeros(*hidden.shape[:-1], self.kernel - 1), hidden), dim=-1)
        hidden = self.depthwise_conv(hidden)
        hidden = self.depthwise_layer_norm(hidden.transpose(1, 2)).transpose(1, 2)
        return self.pointwise_conv2(self.activation(hidden)).transpose(1, 2)


class BertConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_projection = nn.Module()
        self.feature_projection.layer_norm = LayerNorm(config.feature_projection_input_dim,
                                                       eps=config.layer_norm_eps, promote_fp32=False)
        self.feature_projection.projection = Linear(config.feature_projection_input_dim, config.hidden_size)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(ConformerLayer(config, RelativeKeyAttention, CausalConvolution,
                                                          config.layer_norm_eps)
                                            for _ in range(config.num_hidden_layers))
        if config.mask_time_prob > 0 or config.mask_feature_prob > 0:
            self.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, input_features, attention_mask=None):
        features = self.feature_projection.layer_norm(input_features)
        hidden = self.feature_projection.projection(features)
        mask = attention_mask.bool() if attention_mask is not None else None
        if mask is not None:
            hidden = hidden.masked_fill(~mask[:, :, None], 0)
        for layer in self.encoder.layers:
            hidden = layer(hidden, None, mask)
        return {"last_hidden_state": hidden, "extract_features": features}


def build_from_config(config, device, dtype):
    if (config.position_embeddings_type != "relative_key" or config.hidden_act != "swish"
            or config.add_adapter or config.use_intermediate_ffn_before_adapter):
        raise ValueError("Selected W2V-BERT checkpoint uses relative keys, swish and no adapter")
    return BertConformer(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    state = {name.replace(".distance_embedding.weight", ".distance_embedding.emb.weight"): value
             for name, value in state_dict.items()}
    model.load_state_dict(state, strict=True)
