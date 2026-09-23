"""RT-DETR uses the existing detector with the v1 sampling configuration."""

from .rt_detr_v2 import build_from_config as build_detector
from .rt_detr_v2 import load_state_dict_into as load_detector
from .rt_detr_v2 import make_workloads
from ..runner import config_values
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L2.rtdetrv2_multihead_attention import RTDetrV2MultiheadAttention


class _Attention(RTDetrV2MultiheadAttention):
    """Reuse native SDPA attention rather than round an eager score tensor."""

    def __init__(self, width, heads):
        super().__init__(width, heads)
        self.attention = DenseAttention(backend="cudnn")

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                output_attentions=False):
        if output_attentions:
            raise ValueError("The ordinary SDPA workload does not return attention weights")
        positioned = hidden_states if position_embeddings is None else hidden_states + position_embeddings
        shape = (*hidden_states.shape[:2], self.num_heads, self.head_dim)
        output = self.attention(self.q_proj(positioned).view(shape),
                                self.k_proj(positioned).view(shape),
                                self.v_proj(hidden_states).view(shape),
                                attn_mask=attention_mask)
        return self.out_proj(output.reshape(hidden_states.shape)), None


def build_from_config(config, device, dtype):
    values = config.to_dict()
    # V1 has uniform sampling points at each level and the fixed half-box scale.
    values.update(decoder_n_levels=config.num_feature_levels,
                  decoder_offset_scale=0.5, decoder_method="default")
    model = build_detector(config_values(values), device, dtype)
    for layer in tuple(model.modules()):
        attention = getattr(layer, "self_attn", None)
        if isinstance(attention, RTDetrV2MultiheadAttention):
            layer.self_attn = _Attention(attention.embed_dim, attention.num_heads).to(device=device, dtype=dtype)
    return model.eval()


def load_state_dict_into(model, state_dict, config):
    weights = dict(state_dict)
    # V2 stores the fixed 1 / points scale; V1 computes it from the configuration.
    for name, value in model.state_dict().items():
        if name.endswith(".n_points_scale"):
            weights[name] = value
    load_detector(model, weights, config)
