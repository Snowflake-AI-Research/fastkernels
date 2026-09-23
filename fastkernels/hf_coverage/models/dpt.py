"""DPT's default non-hybrid base encoder, including positional interpolation."""

from fastkernels.hf_coverage.models.vit import (
    ViTModel, _Embeddings, load_state_dict_into, make_workloads,
)
from fastkernels.tasks.baseline.L1.interpolate import Interpolate


class _DPTEmbeddings(_Embeddings):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.interpolate = Interpolate()

    def forward(self, pixel_values):
        import torch

        patches = self.patch_embeddings(pixel_values)
        h, w = (size // self.patch_size for size in pixel_values.shape[-2:])
        grid_size = int((self.position_embeddings.shape[1] - 1) ** 0.5)
        grid = self.position_embeddings[:, 1:].reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
        grid = self.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)
        position = torch.cat((self.position_embeddings[:, :1], grid.flatten(2).transpose(1, 2)), dim=1)
        tokens = torch.cat((self.cls_token.expand(pixel_values.shape[0], -1, -1), patches), dim=1)
        return tokens + position


def build_from_config(config, device, dtype):
    if config.is_hybrid or config.hidden_act != "gelu" or config.pooler_act != "tanh":
        raise ValueError("This declared DPT base task uses the non-hybrid GELU encoder and tanh pooler")
    model = ViTModel(config)
    model.embeddings = _DPTEmbeddings(config)
    return model.to(device=device, dtype=dtype).eval()
