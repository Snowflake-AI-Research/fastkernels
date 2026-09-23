"""MaskFormer preserves shifted windows even in the final single-window stage."""

import torch

from .swin import SwinModel, _check_config, load_state_dict_into
from ..runner import Workload
from fastkernels.tasks.baseline.L3.swinv2_block import window_partition


class MaskFormerSwinModel(SwinModel):
    def __init__(self, config):
        super().__init__(config)
        grid = config.image_size // config.patch_size
        # The validated fixed input needs no padding. The last stage does not
        # merge patches, so its output repeats the final spatial dimensions.
        dimensions = [(grid, grid)]
        for stage in self.stages:
            if not isinstance(stage.downsample, torch.nn.Identity):
                grid //= 2
            dimensions.append((grid, grid))
        self.spatial_dimensions = tuple(dimensions)

    def forward(self, pixel_values):
        output = super().forward(pixel_values)
        output["hidden_states_spatial_dimensions"] = self.spatial_dimensions
        return output


def build_from_config(config, device, dtype):
    _check_config(config)
    model = MaskFormerSwinModel(config).to(device=device, dtype=dtype).eval()
    for stage_index, stage in enumerate(model.stages):
        resolution = config.image_size // config.patch_size // 2**stage_index
        for index, block in enumerate(stage.blocks):
            if index % 2 and block.shift == 0:
                block.shift = block.window // 2
                regions = torch.zeros(1, resolution, resolution, 1, device=device, dtype=dtype)
                intervals = (slice(0, -block.window), slice(-block.window, -block.shift), slice(-block.shift, None))
                for h, rows in enumerate(intervals):
                    for w, columns in enumerate(intervals):
                        regions[:, rows, columns, :] = 3 * h + w
                regions = window_partition(regions, (block.window, block.window)).reshape(-1, block.window**2)
                mask = regions.unsqueeze(1) - regions.unsqueeze(2)
                block.shift_mask = mask.masked_fill(mask != 0, -100).masked_fill(mask == 0, 0)
    return model


def make_workloads(model, inputs, config):
    if set(inputs) != {"pixel_values"}:
        raise ValueError("Default Swin inference expects only pixel_values")
    # Public forward retains the Python shape descriptors. The runner requires
    # tensor-only workload outputs even before its collection boundary.
    def forward():
        output = model(inputs["pixel_values"])
        return {name: output[name] for name in ("last_hidden_state", "pooler_output")}

    return {"forward": Workload(run=forward)}
