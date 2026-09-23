"""Small position/casting adaptations of existing attention and YaRN operations."""

from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YarnRotaryEmbedding, YaRNRotaryEmbedding


class MinistralPositionAttention(LlamaAttention):
    def _get_attn_scale(self, positions):
        # Parent uses floor((position+1)/boundary); HF uses floor(position/boundary)
        # and rounds the scale before multiplication. The parent formula and
        # attention algorithm otherwise remain unchanged.
        return super()._get_attn_scale(positions - 1).to(self.qkv_proj.weight.dtype)


class MinistralYarn(YarnRotaryEmbedding):
    # Keep the parent's configurable mscale/mscale_all_dim initialization,
    # using the other existing YaRN variant's native-dtype NeoX CUDA dispatch.
    forward = YaRNRotaryEmbedding.forward
