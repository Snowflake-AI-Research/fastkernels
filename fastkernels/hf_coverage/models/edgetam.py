"""EdgeTAM image prediction with an explicit RepViT-M1 operation composition."""

from torch import nn

from fastkernels.hf_coverage.models import sam2
from fastkernels.hf_coverage.patches.sam_sine_dtype import SamSineDtype
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.efficientnetv2_squeeze_excite import SqueezeExcite


class ConvNorm(nn.Module):
    def __init__(self, source, target, kernel=1, stride=1, groups=1):
        super().__init__()
        self.c = Conv2d(source, target, kernel, stride=stride, padding=kernel // 2,
                        groups=groups, bias=False)
        self.bn = BatchNorm2d(target)

    def forward(self, x):
        return self.bn(self.c(x))


class ChannelMixer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1, self.conv2 = ConvNorm(width, width * 2), ConvNorm(width * 2, width)
        self.act = GELU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class TokenMixer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = ConvNorm(width, width, 3, groups=width)
        self.conv1 = ConvNorm(width, width, 1, groups=width)

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class Block(nn.Module):
    def __init__(self, width, excitation):
        super().__init__()
        self.token_mixer = TokenMixer(width)
        self.se = nn.Identity()
        if excitation:
            self.se = SqueezeExcite(width, max(8, int(width / 4 + 4) // 8 * 8))
            self.se.act1 = ReLU()
        self.channel_mixer = ChannelMixer(width)

    def forward(self, x):
        x = self.se(self.token_mixer(x))
        return x + self.channel_mixer(x)


class Downsample(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.pre_block = Block(source, False)
        self.spatial_downsample = ConvNorm(source, source, 3, stride=2, groups=source)
        self.channel_downsample = ConvNorm(source, target)
        self.ffn = ChannelMixer(target)

    def forward(self, x):
        x = self.channel_downsample(self.spatial_downsample(self.pre_block(x)))
        return self.ffn(x) + x


class Stage(nn.Module):
    def __init__(self, source, target, depth, downsample):
        super().__init__()
        self.downsample = Downsample(source, target) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[Block(target, index % 2 == 0) for index in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class RepVit(nn.Module):
    """Pinned timm repvit_m1 topology: legacy branch norms and all20 blocks."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Module()
        self.stem.conv1, self.stem.conv2 = ConvNorm(3, 24, 3, 2), ConvNorm(24, 48, 3, 2)
        self.stem.act1 = GELU()
        source = 48
        for index, (width, depth) in enumerate(zip((48, 96, 192, 384), (2, 2, 14, 2))):
            self.add_module(f"stages_{index}", Stage(source, width, depth, index > 0))
            source = width

    def forward(self, x):
        x = self.stem.conv2(self.stem.act1(self.stem.conv1(x)))
        outputs = []
        for index in range(4):
            x = getattr(self, f"stages_{index}")(x)
            outputs.append(x.permute(0, 2, 3, 1))
        return outputs


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.timm_model = RepVit()

    def forward(self, x):
        return self.timm_model(x)


class VisionEncoder(sam2.VisionEncoder):
    def __init__(self, c):
        nn.Module.__init__(self)
        if c.backbone_config.architecture != "repvit_m1":
            raise ValueError("The declared EdgeTAM checkpoint uses repvit_m1")
        self.backbone = Backbone()
        self.neck = nn.Module()
        self.neck.convs = nn.ModuleList([
            Conv2d(width, c.fpn_hidden_size, c.fpn_kernel_size,
                   stride=c.fpn_stride, padding=c.fpn_padding)
            for width in c.backbone_channel_list
        ])
        self.position, self.resize = SamSineDtype(c.fpn_hidden_size), Interpolate()
        self.top_down, self.levels = c.fpn_top_down_levels, c.num_feature_levels


def build_from_config(config, device, dtype):
    return sam2.Sam2Model(config, vision_encoder=VisionEncoder).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    renamed = {name.replace(".se.fc1.", ".se.conv_reduce.").replace(".se.fc2.", ".se.conv_expand."): value
               for name, value in state_dict.items()}
    sam2.load_state_dict_into(model, renamed, config)


make_workloads = sam2.make_workloads
