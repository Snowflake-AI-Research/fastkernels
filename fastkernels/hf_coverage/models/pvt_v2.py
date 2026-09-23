"""PVTv2's default spatial-reduction encoder and final NCHW feature map."""

from torch import nn

from fastkernels.hf_coverage.models.segformer import (
    SpatialReductionEncoder, check_encoder_config, encoder_state_name,
)
from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.tasks.baseline.L2.vjepa2_attention import _eager_attention_forward


class _EagerAttentionCore(nn.Module):
    """Adapt layouts around the unchanged library's materialized eager core."""

    def forward(self, query, key, value, softmax_scale, causal=False):
        if causal:
            raise ValueError("PVTv2 attention is bidirectional")
        output, _ = _eager_attention_forward(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), softmax_scale,
        )
        return output.transpose(1, 2)


class PvtV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SpatialReductionEncoder(config, norm_eps=config.layer_norm_eps)
        for stage in self.encoder.stages:
            for block in stage.blocks:
                block.attention.attention.attn = _EagerAttentionCore()

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        return {"last_hidden_state": self.encoder(pixel_values)}


def build_from_config(config, device, dtype):
    check_encoder_config(config)
    if config.linear_attention or not config.qkv_bias:
        raise ValueError("This pilot preserves default spatial-reduction attention and biased QKV")
    return PvtV2Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace("encoder.layers.", "encoder.stages.")
        name = name.replace(".patch_embedding.proj.", ".projection.")
        name = name.replace(".patch_embedding.layer_norm.", ".embedding_norm.")
        name = name.replace(".attention.spatial_reduction.", ".attention.sr.")
        name = encoder_state_name(name)
        name = name.replace(".layer_norm.", ".norm.")
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
