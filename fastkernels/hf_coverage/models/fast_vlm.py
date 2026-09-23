"""FastVLM's five-stage reparameterized FastViT and tied Qwen2 computation."""
from types import SimpleNamespace
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.mobilenetv4_uib import ConvNormAct
from fastkernels.tasks.baseline.L2.efficientnetv2_squeeze_excite import SqueezeExcite
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from ..patches.product_gate import ProductGate
from . import deepseek_vl, llava, qwen2


class _LayerScale2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, 1, 1))
        self.product = ProductGate()

    def forward(self, x):
        # The actual packing cost remains inside the measured forward.
        packed = torch.cat((x, self.gamma.expand_as(x)), dim=-1)
        return self.product(packed)


class _FastVitMobileOne(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, *, use_se=False, use_act=True):
        super().__init__()
        self.reparam_conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=kernel_size // 2, groups=groups, bias=True)
        self.se = SqueezeExcite(out_channels, max(1, out_channels // 16)) if use_se else nn.Identity()
        if use_se:
            self.se.act1 = ReLU()
        self.act = GELU() if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.se(self.reparam_conv(x)))


class _FastVitAttentionBlock(nn.Module):
    def __init__(self, channels, mlp_ratio):
        super().__init__()
        self.norm = LayerNorm2d(channels, eps=1e-5)
        self.token_mixer = _FastVitAttention(channels)
        self.layer_scale_1 = _LayerScale2d(channels)
        self.mlp = _FastVitConvMlp(channels, int(channels * mlp_ratio))
        self.layer_scale_2 = _LayerScale2d(channels)

    def forward(self, x):
        x = x + self.layer_scale_1(self.token_mixer(self.norm(x)))
        return x + self.layer_scale_2(self.mlp(x))


class _FastVitPositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reparam_conv = Conv2d(channels, channels, 7, padding=3, groups=channels)

    def forward(self, x):
        return self.reparam_conv(x)



class _FastVitConvMlp(VitEncoderMlp):
    """FastViT depthwise pre-conv plus exact checked-in GELU two-layer MLP."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__(channels, hidden_channels, channels, act_approximate="none", bias=True, drop=0.0)
        self.conv = ConvNormAct(
            channels,
            channels,
            kernel_size=7,
            groups=channels,
            apply_act=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)

class _FastVitAttention(VitEncoderAttention):
    """NCHW layout adapter around the unchanged checked-in ViT attention."""

    def __init__(self, channels: int):
        super().__init__(
            channels,
            num_heads=channels // 32,
            qkv_bias=False,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = super().forward(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

class _FastVitRepMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.reparam_conv = Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reparam_conv(x)

class _FastVitRepMixerBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float):
        super().__init__()
        self.token_mixer = _FastVitRepMixer(channels)
        self.mlp = _FastVitConvMlp(channels, int(channels * mlp_ratio))
        self.layer_scale = _LayerScale2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        return x + self.layer_scale(self.mlp(x))

class _FastVitPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            _FastVitMobileOne(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                groups=in_channels,
                use_act=True,
            ),
            _FastVitMobileOne(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                use_act=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class _FastVitStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        token_mixer: str,
        mlp_ratio: float,
        *,
        downsample: bool,
        positional_encoding: bool,
    ):
        super().__init__()
        self.downsample = _FastVitPatchEmbed(in_channels, out_channels) if downsample else nn.Identity()
        self.pos_emb = _FastVitPositionalEncoding(out_channels) if positional_encoding else nn.Identity()
        block_cls = _FastVitRepMixerBlock if token_mixer == "repmixer" else _FastVitAttentionBlock
        self.blocks = nn.Sequential(*(block_cls(out_channels, mlp_ratio) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.pos_emb(self.downsample(x)))

class _FastVit(nn.Module):
    def __init__(self, layers: list[int], dims: list[int], mlp_ratios: list[float]):
        super().__init__()
        self.stem = nn.Sequential(
            _FastVitMobileOne(3, dims[0], 3, 2, 1),
            _FastVitMobileOne(dims[0], dims[0], 3, 2, dims[0]),
            _FastVitMobileOne(dims[0], dims[0], 1, 1, 1),
        )
        token_mixers = ["repmixer", "repmixer", "repmixer", "attention", "attention"]
        previous = dims[0]
        stages = []
        for index, (depth, channels, ratio, mixer) in enumerate(zip(layers, dims, mlp_ratios, token_mixers)):
            stages.append(
                _FastVitStage(
                    previous,
                    channels,
                    depth,
                    mixer,
                    ratio,
                    downsample=index > 0,
                    positional_encoding=index >= 3,
                )
            )
            previous = channels
        self.stages = nn.Sequential(*stages)
        final_channels = dims[-1] * 2
        self.final_conv = _FastVitMobileOne(
            dims[-1],
            final_channels,
            kernel_size=3,
            stride=1,
            groups=dims[-1],
            use_se=True,
            use_act=True,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.final_conv(x)

class FastBackbone(deepseek_vl.DeepseekBackbone):
    def __init__(self, text, config):
        nn.Module.__init__(self)
        self.text = text
        args = config.vision_config.model_args
        self.vision = _FastVit(args.get("layers", [2, 12, 24, 4, 2]), args.get("embed_dims", [96, 192, 384, 768, 1536]), args.get("mlp_ratios", [4, 4, 4, 4, 4]))
        self.pool = GlobalAvgPool2d(keepdim=False)
        self.linear_1 = Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias)
        self.linear_2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias)
        self.activation = GELU()
        self.image_token_id = config.image_token_index
        self.pixel_values = self.image_hidden_states = None

    def features(self, pixels):
        hidden = self.vision.forward_features(pixels)
        self.pool(hidden)  # Native timm wrapper executes its default pooling head.
        images = self.linear_2(self.activation(self.linear_1(hidden.flatten(2).transpose(1, 2))))
        return images.flatten(0, 1)


class FastModel(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = FastBackbone(language.model, config)


def build_from_config(config, device, dtype):
    vision, text = config.vision_config, config.text_config
    if (vision.architecture != "fastvit_mci3" or not vision.model_args['inference_mode']
            or not vision.do_pooling or config.vision_feature_layer != -1
            or config.vision_feature_select_strategy != "full" or text.use_sliding_window
            or text.rope_parameters['rope_type'] != 'default' or not config.tie_word_embeddings):
        raise ValueError("Preserve the published FastViT inference graph and tied full-attention Qwen2")
    fields = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
              'vocab_size', 'max_position_embeddings', 'rms_norm_eps')
    native = LlamaConfig(**{name:getattr(text,name) for name in fields}, head_dim=text.hidden_size//text.num_attention_heads,
                         dtype=dtype, rope_theta=text.rope_parameters['rope_theta'], rope_scaling_factor=1., qkv_bias=True)
    language = LlamaForCausalLM(native)
    language.lm_head.embedding_op.emb.weight = language.model.embed_tokens.embedding_op.emb.weight
    return FastModel(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace('model.language_model.', 'model.'):remaining.pop(name)
            for name in list(remaining) if name.startswith('model.language_model.')}
    text['lm_head.weight'] = remaining.pop('lm_head.weight')
    if not torch.equal(text['lm_head.weight'], text['model.embed_tokens.weight']):
        raise ValueError('Tied FastVLM head and embeddings differ')
    qwen2.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    mapped = {}
    for name in model.model.vision.state_dict():
        source = name.replace('.se.conv_reduce.', '.se.fc1.').replace('.se.conv_expand.', '.se.fc2.')
        value = remaining.pop('model.vision_tower.timm_model.'+source)
        mapped[name] = value.reshape_as(model.model.vision.state_dict()[name])
    model.model.vision.load_state_dict(mapped, strict=True)
    for name in ('linear_1', 'linear_2'):
        module = getattr(model.model,name)
        module.load_state_dict({field:remaining.pop(f'model.multi_modal_projector.{name}.{field}') for field in module.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f'Unmapped FastVLM weights: {sorted(remaining)}')


make_workloads = llava.make_workloads
