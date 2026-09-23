"""DepthPro's default multiscale depth and field-of-view computation."""

import math

import torch
from torch import nn

from .depth_anything import NonoverlappingTransposeConv
from .dinov2 import Dinov2Model, load_state_dict_into as load_dinov2
from .vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU


class _TransposeConv(NonoverlappingTransposeConv):
    """Use the existing FP32 linear op, then restore the layer's output dtype.

    The deep fusion head amplifies BF16 GEMM/cuDNN accumulation differences.
    Weights retain their common rounded values; their fixed FP32 conversion is
    prepared at construction/loading, while activation casts remain in forward.
    """

    def forward(self, hidden):
        return super().forward(hidden.float()).to(hidden.dtype)


def _resize(x, *, size=None, scale_factor=None):
    return Interpolate()(x, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def _base_size(config, pixels):
    out_size = config.image_model_config.image_size // config.image_model_config.patch_size
    height, width = pixels.shape[-2:]
    exponent = int(math.log2(width / out_size))
    return height // 2**exponent, width // 2**exponent


def _reconstruct(hidden, batch, padding, size):
    grid = math.isqrt(hidden.shape[1])
    patches = hidden[:, -(grid * grid):].reshape(hidden.shape[0], grid, grid, hidden.shape[-1])
    patches = patches.permute(0, 3, 1, 2)
    count = patches.shape[0] // batch
    if count > 1:
        # HF retains the first complete square of patches, including for its
        # intermediate features collected across all three image scales.
        side = math.isqrt(count)
        padding = min(grid // 4, padding) if count >= 4 else 0
        rows = []
        for row in range(side):
            columns = []
            for col in range(side):
                index = row * side + col
                tile = patches[batch * index:batch * (index + 1)]
                top, bottom = (padding if row else 0), (grid - padding if row < side - 1 else grid)
                left, right = (padding if col else 0), (grid - padding if col < side - 1 else grid)
                columns.append(tile[:, :, top:bottom, left:right])
            rows.append(torch.cat(columns, dim=-1))
        patches = torch.cat(rows, dim=-2)
    return _resize(patches, size=size)


def _tower(model, pixels, hooks=()):
    hidden, intermediates = model.embeddings(pixels), {}
    for index, layer in enumerate(model.encoder):
        hidden = layer(hidden)
        if index in hooks:
            intermediates[index] = hidden
    return model.layernorm(hidden), intermediates


class _PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model = config, Dinov2Model(config.patch_model_config)

    def forward(self, pixels):
        config, batch = self.config, pixels.shape[0]
        patch_size = config.patch_size
        crops = []
        for scale, overlap in zip(config.scaled_images_ratios, config.scaled_images_overlap_ratios):
            scaled = _resize(pixels, scale_factor=scale)
            stride = int(patch_size * (1 - overlap))
            patches = scaled.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
            patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(-1, pixels.shape[1], patch_size, patch_size)
            crops.append(patches)
        hidden, intermediate = _tower(self.model, torch.cat(crops[::-1]), config.intermediate_hook_ids)
        split = hidden.split([part.shape[0] for part in crops[::-1]])[::-1]
        base = _base_size(config, pixels)
        features = [
            _reconstruct(part, batch, int(config.merge_padding_value / scale),
                         tuple(value * 2**index for value in base))
            for index, (part, scale) in enumerate(zip(split, config.scaled_images_ratios))
        ]
        size = tuple(value * 2 ** (len(crops) - 1) for value in base)
        features.extend(_reconstruct(intermediate[index], batch,
                                     int(config.merge_padding_value / config.scaled_images_ratios[-1]), size)
                        for index in config.intermediate_hook_ids)
        return features


class _ImageEncoder(nn.Module):
    def __init__(self, config, *, fov=False):
        super().__init__()
        self.config = config
        self.tower_config = config.fov_model_config if fov else config.image_model_config
        self.model = Dinov2Model(self.tower_config)
        if fov:
            self.neck = Linear(self.tower_config.hidden_size, config.fusion_hidden_size // 2)

    def forward(self, pixels):
        size = self.tower_config.image_size
        hidden, _ = _tower(self.model, _resize(pixels, size=(size, size)))
        if hasattr(self, "neck"):
            hidden = self.neck(hidden)
        return _reconstruct(hidden, pixels.shape[0], 0, _base_size(self.config, pixels))


class _Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_encoder, self.image_encoder = _PatchEncoder(config), _ImageEncoder(config)

    def forward(self, pixels):
        patches = self.patch_encoder(pixels)
        return [self.image_encoder(pixels), *patches]


class _Layers(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _upsample(input_width, intermediate, output_width, count, *, project=True, bias=False):
    layers = [Conv2d(input_width, intermediate, 1, bias=bias)] if project else []
    layers.extend(_TransposeConv(intermediate if i == 0 else output_width, output_width, 2, bias)
                  for i in range(count))
    return _Layers(layers)


class _Upsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        image, patch = config.image_model_config.hidden_size, config.patch_model_config.hidden_size
        self.image_block = _upsample(image, image, config.scaled_images_feature_dims[0], 1, project=False, bias=True)
        self.scaled_images = nn.ModuleList([_upsample(patch, width, width, 1)
                                           for width in config.scaled_images_feature_dims])
        self.intermediate = nn.ModuleList([
            _upsample(patch, config.fusion_hidden_size if i == 0 else width, width, 2 + i)
            for i, width in enumerate(config.intermediate_feature_dims)
        ])

    def forward(self, features):
        blocks = [self.image_block, *self.scaled_images, *self.intermediate]
        return [block(feature) for block, feature in zip(blocks, features)]


class _Projection(nn.Module):
    def __init__(self, config):
        super().__init__()
        widths = config.scaled_images_feature_dims + config.intermediate_feature_dims
        self.projections = nn.ModuleList([
            nn.Identity() if i == len(widths) - 1 and width == config.fusion_hidden_size else
            Conv2d(width, config.fusion_hidden_size, 3, padding=1, bias=False)
            for i, width in enumerate(widths)
        ])

    def forward(self, features):
        return [layer(feature) for layer, feature in zip(self.projections, features)]


class _Neck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_upsample, self.feature_projection = _Upsample(config), _Projection(config)
        width = config.scaled_images_feature_dims[0]
        self.fuse_image_with_low_res = Conv2d(width * 2, width, 1)

    def forward(self, features):
        features = self.feature_upsample(features)
        global_features = self.fuse_image_with_low_res(torch.cat((features[1], features[0]), dim=1))
        return self.feature_projection([global_features, *features[2:]])


class _DepthPro(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder, self.neck = _Encoder(config), _Neck(config)

    def forward(self, pixels):
        return self.neck(self.encoder(pixels))


class _Residual(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.fusion_hidden_size
        bias = config.use_bias_in_fusion_residual
        bias = True if bias is None else bias
        self.activation1, self.activation2 = ReLU(), ReLU()
        self.convolution1 = Conv2d(width, width, 3, padding=1, bias=bias)
        self.convolution2 = Conv2d(width, width, 3, padding=1, bias=bias)

    def forward(self, hidden):
        return hidden + self.convolution2(self.activation2(self.convolution1(self.activation1(hidden))))


class _Fusion(nn.Module):
    def __init__(self, config, deconv=True):
        super().__init__()
        width = config.fusion_hidden_size
        self.residual_layer1, self.residual_layer2 = _Residual(config), _Residual(config)
        if deconv:
            self.deconv = _TransposeConv(width, width, 2, bias=False)
        self.projection = Conv2d(width, width, 1)

    def forward(self, hidden, residual=None):
        if residual is not None:
            hidden = hidden + self.residual_layer1(residual)
        hidden = self.residual_layer2(hidden)
        if hasattr(self, "deconv"):
            hidden = self.deconv(hidden)
        return self.projection(hidden)


class _FusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        count = len(config.scaled_images_ratios) + len(config.intermediate_hook_ids)
        self.intermediate = nn.ModuleList([_Fusion(config) for _ in range(count - 1)])
        self.final = _Fusion(config, deconv=False)

    def forward(self, features):
        hidden = self.intermediate[0](features[0])
        for layer, feature in zip(self.intermediate[1:], features[1:-1]):
            hidden = layer(hidden, feature)
        return self.final(hidden, features[-1])


class _FovModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.fusion_hidden_size
        self.out_size = config.image_model_config.image_size // config.image_model_config.patch_size
        self.fov_encoder = _ImageEncoder(config, fov=True)
        self.conv, self.activation = Conv2d(width, width // 2, 3, stride=2, padding=1), ReLU()
        layers = []
        for i in range(config.num_fov_head_layers):
            layers.extend([Conv2d(math.ceil(width / 2**(i + 1)), math.ceil(width / 2**(i + 2)),
                                  3, stride=2, padding=1), ReLU()])
        layers.append(Conv2d(math.ceil(width / 2**(config.num_fov_head_layers + 1)), 1,
                             int((self.out_size - 1) / 2**config.num_fov_head_layers + 1)))
        self.head = _Layers(layers)

    def forward(self, pixels, features):
        hidden = self.fov_encoder(pixels) + self.activation(self.conv(features))
        return self.head(_resize(hidden, size=(self.out_size, self.out_size))).flatten()


class DepthProForDepthEstimation(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.fusion_hidden_size
        self.depth_pro, self.fusion_stage = _DepthPro(config), _FusionStage(config)
        self.head = _Layers([Conv2d(width, width // 2, 3, padding=1),
                             _TransposeConv(width // 2, width // 2, 2),
                             Conv2d(width // 2, 32, 3, padding=1), ReLU(), Conv2d(32, 1, 1), ReLU()])
        if config.use_fov_model:
            self.fov_model = _FovModel(config)

    def forward(self, pixel_values):
        features = self.depth_pro(pixel_values)
        outputs = {"predicted_depth": self.head(self.fusion_stage(features)).squeeze(1)}
        if hasattr(self, "fov_model"):
            outputs["field_of_view"] = self.fov_model(pixel_values, features[0].detach())
        return outputs


def build_from_config(config, device, dtype):
    if config.use_batch_norm_in_fusion_residual:
        raise ValueError("The default DepthPro checkpoint disables fusion batch normalization")
    for tower in (config.image_model_config, config.patch_model_config, config.fov_model_config):
        if tower.model_type != "dinov2" or tower.use_swiglu_ffn or tower.hidden_act != "gelu":
            raise ValueError("DepthPro's default towers use DINOv2 with exact GELU")
    model = DepthProForDepthEstimation(config).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        if isinstance(module, _TransposeConv):
            module.projection.float()
    return model


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    towers = [("depth_pro.encoder.patch_encoder.model", config.patch_model_config),
              ("depth_pro.encoder.image_encoder.model", config.image_model_config)]
    if config.use_fov_model:
        towers.append(("fov_model.fov_encoder.model", config.fov_model_config))
    for name, tower_config in towers:
        prefix, tower = name + ".", model.get_submodule(name)
        state = {key[len(prefix):]: remaining.pop(key) for key in list(remaining) if key.startswith(prefix)}
        load_dinov2(tower, state, tower_config)
        mapped.update({prefix + key: value for key, value in tower.state_dict().items()})
    for name, module in model.named_modules():
        if isinstance(module, NonoverlappingTransposeConv):
            weight = remaining.pop(name + ".weight")
            mapped[name + ".projection.weight"] = weight.permute(1, 2, 3, 0).reshape(-1, weight.shape[0])
            if module.projection.bias is not None:
                mapped[name + ".projection.bias"] = remaining.pop(name + ".bias").repeat_interleave(module.factor**2)
    mapped.update(remaining)
    model.load_state_dict(mapped, strict=True)
