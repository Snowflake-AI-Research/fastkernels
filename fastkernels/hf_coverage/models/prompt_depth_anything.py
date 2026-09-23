"""Prompt Depth Anything, including prompt normalization and all four injections."""

import torch
from torch import nn

from .depth_anything import DepthAnythingForDepthEstimation, _Fusion, load_state_dict_into
from ..patches.product_gate import ProductGate
from ..runner import Workload
from fastkernels.tasks.baseline.L1.frozen_batch_norm2d import FrozenBatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU


class PromptRange(nn.Module):
    """Compose min/max reductions and normalization without an added epsilon."""

    def __init__(self):
        super().__init__()
        self.reduce, self.square = SegmentCSR(), SquaredReLU()
        self.normalize = FrozenBatchNorm2d(1, eps=0.0)
        # Unit scale, zero bias/mean, and runtime variance are not model weights.
        self.normalize._non_persistent_buffers_set.update(self.normalize._buffers)

    def forward(self, prompt):
        batch = prompt.shape[0]
        count = prompt[0].numel()
        offsets = torch.arange(0, prompt.numel() + 1, count, device=prompt.device)
        minimum = self.reduce(prompt.reshape(-1), offsets, reduce="min")
        maximum = self.reduce(prompt.reshape(-1), offsets, reduce="max")
        span = maximum - minimum
        centered = prompt - minimum.reshape(batch, 1, 1, 1)
        # FP64 preserves squared finite FP32/BF16 spans without under/overflow.
        # Each image is one channel; all statistics and conversions remain timed.
        self.normalize.running_var = self.square(span.double())
        normalized = self.normalize(centered.double().reshape(1, batch, *prompt.shape[-2:]))
        return normalized.reshape_as(prompt).to(prompt.dtype), minimum, span


class _PromptLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.convolution1 = Conv2d(1, width, 3, padding=1)
        self.convolution2 = Conv2d(width, width, 3, padding=1)
        self.convolution3 = Conv2d(width, width, 3, padding=1)
        self.activation1, self.activation2 = ReLU(), ReLU()

    def forward(self, prompt):
        return self.convolution3(self.activation2(self.convolution2(self.activation1(self.convolution1(prompt)))))


class _PromptFusion(_Fusion):
    def __init__(self, width):
        super().__init__(width)
        self.prompt_depth_layer = _PromptLayer(width)

    def forward(self, hidden, residual, size, prompt):
        if residual is not None:
            if residual.shape != hidden.shape:
                residual = self.interpolate(residual, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
            hidden = hidden + self.residual_layer1(residual)
        hidden = self.residual_layer2(hidden)
        prompt = self.interpolate(prompt, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
        hidden = hidden + self.prompt_depth_layer(prompt)
        return self.projection(self.interpolate(hidden, size=size, scale_factor=2 if size is None else None,
                                               mode="bilinear", align_corners=True))


class PromptDepthAnythingForDepthEstimation(DepthAnythingForDepthEstimation):
    def __init__(self, config):
        super().__init__(config)
        self.neck.fusion_stage.layers = nn.ModuleList([_PromptFusion(config.fusion_hidden_size)
                                                      for _ in config.neck_hidden_sizes])
        # This task always uses the final fusion output and does not scale by max_depth.
        self.head.index, self.head.max_depth = -1, 1
        self.prompt_range, self.product = PromptRange(), ProductGate()

    def forward(self, pixel_values, prompt_depth):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        hidden, features = self.backbone.embeddings(pixel_values), []
        for index, layer in enumerate(self.backbone.encoder, start=1):
            hidden = layer(hidden)
            if index in self.out_indices:
                features.append(self.backbone.layernorm(hidden) if self.apply_layernorm else hidden)
        height, width = (size // self.patch_size for size in pixel_values.shape[-2:])
        prompt, minimum, span = self.prompt_range(prompt_depth)
        features = self.neck.reassemble_stage(features, height, width)
        features = [conv(feature) for conv, feature in zip(self.neck.convs, features)][::-1]
        fused, outputs = None, []
        for index, (feature, layer) in enumerate(zip(features, self.neck.fusion_stage.layers)):
            size = features[index + 1].shape[-2:] if index + 1 < len(features) else None
            fused = layer(feature if fused is None else fused, None if fused is None else feature, size, prompt)
            outputs.append(fused)
        depth = self.head(outputs, height, width)
        depth = self.product(torch.cat((depth, span[:, None, None].expand_as(depth)), dim=-1))
        return {"predicted_depth": depth + minimum[:, None, None]}


def build_from_config(config, device, dtype):
    if config.backbone_config.model_type != "dinov2" or config.backbone_config.reshape_hidden_states:
        raise ValueError("The selected checkpoint uses unreshaped DINOv2 backbone features")
    return PromptDepthAnythingForDepthEstimation(config).to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config):
    del config
    return {"forward": Workload(run=lambda: model(**inputs))}
