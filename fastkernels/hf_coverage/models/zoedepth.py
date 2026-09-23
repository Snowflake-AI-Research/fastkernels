"""ZoeDepth's complete NYU/KITTI depth task, including native domain selection."""

import torch
from torch import nn

from .beit import BeitModel, _BeitEmbeddings, load_state_dict_into as load_beit
from .chmv2 import _Reassemble, _Upsample
from .depth_anything import NonoverlappingTransposeConv, _Fusion
from .mvp import EagerAttention
from .vit_msn import make_workloads
from ..patches.product_gate import ProductGate
from ..patches.zoe_probability_clip import ProbabilityClip
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.log_sigmoid import LogSigmoid
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L1.top_k_per_row import _deterministic_topk_indices
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid


class DepthEmbeddings(_BeitEmbeddings):
    """Use the existing patch projection at the processor's spatial size."""

    def forward(self, pixel_values):
        patches = self.patch_embeddings(pixel_values, random_sample=True)
        return torch.cat((self.cls_token.expand(patches.shape[0], -1, -1), patches), dim=1)


class _Arithmetic(nn.Module):
    """Linear-size compositions; supplied BatchNorm statistics remain timed work."""

    def __init__(self):
        super().__init__()
        self.product, self.segment = ProductGate(), SegmentCSR()
        self.square, self.logsigmoid = SquaredReLU(), LogSigmoid()
        self.normalization = BatchNorm2d(1, eps=1e-30, affine=False, track_running_stats=False)

    def multiply(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        return self.product(torch.cat((left, right), dim=-1))

    def reduce(self, value, axis, operation):
        value = value.movedim(axis, -1).contiguous()
        width = value.shape[-1]
        offsets = torch.arange(0, value.numel() + 1, width, device=value.device)
        return self.segment(value.reshape(-1), offsets, reduce=operation).reshape(value.shape[:-1])

    def divide(self, numerator, denominator):
        numerator, denominator = torch.broadcast_tensors(numerator, denominator)
        # Zoe denominators are positive: >=1 for attractors, >=2e-4 for
        # probability pairs, and >=min_temp for the distribution temperature.
        self.normalization.running_mean = torch.zeros(denominator.numel(), device=denominator.device)
        self.normalization.running_var = self.square(denominator.float()).reshape(-1)
        self.normalization.track_running_stats = True
        output = self.normalization(numerator.float().reshape(1, -1, 1, 1))
        return output.reshape_as(numerator).to(numerator.dtype)

    def softplus(self, value):
        return -self.logsigmoid(-value)

    def log_probability(self, probability):
        # The unchanged decoder helper exposes log odds. At p=1, +inf log
        # odds followed by log-sigmoid correctly gives log(1)=0.
        return self.logsigmoid(inverse_sigmoid(probability, eps=0.0))


class _RelativeBias(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window = config.image_size // config.patch_size
        self.table = nn.Parameter(torch.empty((2 * self.window - 1) ** 2 + 3, config.num_attention_heads))
        self.interpolate = Interpolate()

    def forward(self, height, width):
        side = 2 * self.window - 1
        table = self.table[:-3].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        table = self.interpolate(table, size=(2 * height - 1, 2 * width - 1), mode="bilinear", align_corners=False)
        table = torch.cat((table.permute(0, 2, 3, 1).reshape(-1, self.table.shape[1]), self.table[-3:]))
        rows, columns = torch.meshgrid(torch.arange(height, device=table.device),
                                      torch.arange(width, device=table.device), indexing="ij")
        rows, columns = rows.flatten(), columns.flatten()
        spatial = (rows[:, None] - rows[None, :] + height - 1) * (2 * width - 1)
        spatial = spatial + columns[:, None] - columns[None, :] + width - 1
        count = height * width
        index = torch.empty(count + 1, count + 1, dtype=torch.long, device=table.device)
        index[1:, 1:], index[0, :], index[:, 0], index[0, 0] = spatial, table.shape[0] - 3, table.shape[0] - 2, table.shape[0] - 1
        return table[index].permute(2, 0, 1).unsqueeze(0)


class _BeitAttention(nn.Module):
    """Preserve BEiT's score scaling and bias addition before softmax."""

    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.q_proj, self.v_proj, self.proj = (Linear(width, width) for _ in range(3))
        self.k_proj = Linear(width, width, bias=False)
        self.attention = EagerAttention(prescale_query=False, divide_scores=True)

    def forward(self, hidden, rope=None, attn_mask=None):
        batch, length, width = hidden.shape
        q, k, v = (projection(hidden).reshape(batch, length, self.heads, width // self.heads)
                   for projection in (self.q_proj, self.k_proj, self.v_proj))
        return self.proj(self.attention(q, k, v, attn_mask=attn_mask).reshape(batch, length, width))


class _Neck(nn.Module):
    def __init__(self, config):
        super().__init__()
        # CHMv2 and Zoe use the same projected CLS readout and resize graph.
        from types import SimpleNamespace
        readout_config = SimpleNamespace(backbone_config=config.backbone_config,
                                        post_process_channels=config.neck_hidden_sizes,
                                        reassemble_factors=config.reassemble_factors)
        self.reassemble_stage = _Reassemble(readout_config)
        self.convs = nn.ModuleList([Conv2d(width, config.fusion_hidden_size, 3, padding=1, bias=False)
                                   for width in config.neck_hidden_sizes])
        self.fusion_stage = nn.Module()
        self.fusion_stage.layers = nn.ModuleList([_Fusion(config.fusion_hidden_size) for _ in self.convs])

    def forward(self, features, height, width):
        pairs = [(value[:, 1:].transpose(1, 2).reshape(value.shape[0], -1, height, width), value[:, 0])
                 for value in features]
        features = [conv(value) for conv, value in zip(self.convs, self.reassemble_stage(pairs))]
        bottleneck, outputs, hidden = features[-1], [], None
        for value, layer in zip(features[::-1], self.fusion_stage.layers):
            hidden = layer(value) if hidden is None else layer(hidden, value)
            outputs.append(hidden)
        return outputs, bottleneck


class _RelativeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.fusion_hidden_size
        self.index = config.head_in_index
        self.conv1 = Conv2d(width, width // 2, 3, padding=1)
        self.upsample, self.activation = _Upsample(), ReLU()
        self.conv2 = Conv2d(width // 2, config.num_relative_features, 3, padding=1)
        self.conv3 = Conv2d(config.num_relative_features, 1, 1)

    def forward(self, features):
        hidden = self.activation(self.conv2(self.upsample(self.conv1(features[self.index]))))
        return self.activation(self.conv3(hidden)).squeeze(1), hidden


class _Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.width = heads, width // heads
        self.query, self.key, self.value = (Linear(width, width) for _ in range(3))
        self.out_proj = Linear(width, width)
        self.attention = EagerAttention(prescale_query=False, divide_scores=True)

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        q, k, v = (projection(hidden).reshape(batch, length, self.heads, self.width)
                   for projection in (self.query, self.key, self.value))
        return self.out_proj(self.attention(q, k, v).reshape(batch, length, -1))


class _TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.patch_transformer_hidden_size
        self.self_attn = _Attention(width, config.patch_transformer_num_attention_heads)
        self.linear1 = Linear(width, config.patch_transformer_intermediate_size)
        self.linear2 = Linear(config.patch_transformer_intermediate_size, width)
        self.norm1, self.norm2 = (LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(2))
        self.activation = ReLU()

    def forward(self, hidden):
        hidden = self.norm1(hidden + self.self_attn(hidden))
        return self.norm2(hidden + self.linear2(self.activation(self.linear1(hidden))))


class _PatchTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_convPxP = Conv2d(config.bottleneck_features, config.patch_transformer_hidden_size, 1)
        self.transformer_encoder = nn.ModuleList([_TransformerLayer(config) for _ in range(4)])

    def forward(self, hidden):
        hidden = self.embedding_convPxP(hidden).flatten(2).transpose(1, 2)
        hidden = torch.cat((hidden.new_zeros(hidden.shape[0], 1, hidden.shape[2]), hidden), dim=1)
        batch, length, width = hidden.shape
        positions = torch.arange(length, dtype=hidden.dtype, device=hidden.device)[:, None]
        indices = torch.arange(0, width, 2, dtype=hidden.dtype, device=hidden.device)[None]
        frequencies = torch.exp(indices * (-torch.log(torch.tensor(10000.0, device=hidden.device)) / width))
        angles = positions * frequencies
        hidden = hidden + torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)[None].expand(batch, -1, -1)
        for layer in self.transformer_encoder:
            hidden = layer(hidden)
        return hidden


class _Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1, self.linear2, self.activation = Linear(128, 128), Linear(128, 2), ReLU()

    def forward(self, hidden):
        return self.linear2(self.activation(self.linear1(hidden)))


class _Projector(nn.Module):
    def __init__(self, input_width, output_width, intermediate=128):
        super().__init__()
        self.conv1, self.act = Conv2d(input_width, intermediate, 1), ReLU()
        self.conv2 = Conv2d(intermediate, output_width, 1)

    def forward(self, hidden):
        return self.conv2(self.act(self.conv1(hidden)))


class _Seed(_Projector):
    def __init__(self, config, bins):
        super().__init__(config.bottleneck_features, bins, config.bin_embedding_dim // 2)
        self.math = _Arithmetic()

    def forward(self, hidden):
        return self.math.softplus(super().forward(hidden))


class _Attractor(_Projector):
    def __init__(self, config, count):
        super().__init__(config.bin_embedding_dim, count, config.bin_embedding_dim)
        self.math, self.interpolate = _Arithmetic(), Interpolate()

    def forward(self, embedding, previous, previous_embedding):
        previous_embedding = self.interpolate(previous_embedding, size=embedding.shape[-2:], mode="bilinear", align_corners=True)
        attractors = self.math.softplus(super().forward(embedding + previous_embedding))
        centers = self.interpolate(previous, size=attractors.shape[-2:], mode="bilinear", align_corners=True)
        difference = attractors.unsqueeze(2) - centers.unsqueeze(1)
        # Pinned HF invokes inv_attractor without arguments: alpha=300, gamma=2.
        denominator = 1 + 300 * self.math.multiply(difference, difference)
        correction = self.math.reduce(self.math.divide(difference, denominator), 1, "mean")
        return centers + correction


class _ConditionalDistribution(nn.Module):
    def __init__(self, config, bins):
        super().__init__()
        width = config.num_relative_features + config.bin_embedding_dim
        self.mlp = nn.Sequential(Conv2d(width, width // 4, 1), GELU(), Conv2d(width // 4, 4, 1))
        self.math, self.clip, self.softmax = _Arithmetic(), ProbabilityClip(), Softmax(dim=1)
        self.minimum, self.maximum, self.bins = config.min_temp, config.max_temp, bins

    def forward(self, features, embedding):
        pairs = self.math.softplus(self.mlp(torch.cat((features, embedding), dim=1))) + 1e-4
        probability = self.math.divide(pairs[:, :1], pairs[:, :1] + pairs[:, 1:2])
        temperature = self.math.divide(pairs[:, 2:3], pairs[:, 2:3] + pairs[:, 3:4])
        temperature = (self.maximum - self.minimum) * temperature + self.minimum
        log_p = self.math.log_probability(self.clip(probability))
        log_q = self.math.log_probability(self.clip(1 - probability))
        k = torch.arange(self.bins, device=features.device).reshape(1, -1, 1, 1)
        n = torch.tensor([self.bins - 1], device=features.device).reshape(1, -1, 1, 1)
        # The combinatorial coefficients depend only on bin-number metadata.
        nf, kf = n + 1e-7, k + 1e-7
        coefficients = nf * torch.log(nf) - kf * torch.log(kf) - (nf - kf) * torch.log(nf - kf + 1e-7)
        scores = coefficients + self.math.multiply(k.to(log_p.dtype), log_p)
        scores = scores + self.math.multiply((n - k).to(log_q.dtype), log_q)
        return self.softmax(self.math.divide(scores, temperature))


class _MetricHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.bin_embedding_dim
        self.names = [item.name for item in config.bin_configurations]
        self.conv2 = Conv2d(config.bottleneck_features, config.bottleneck_features, 1)
        self.patch_transformer, self.mlp_classifier = _PatchTransformer(config), _Classifier()
        self.seed_bin_regressors = nn.ModuleDict({item.name: _Seed(config, item.n_bins) for item in config.bin_configurations})
        self.seed_projector = _Projector(config.bottleneck_features, width, width // 2)
        self.projectors = nn.ModuleList([_Projector(config.fusion_hidden_size, width, width // 2) for _ in range(4)])
        # HF passes config.num_attractors into the unused n_bins argument;
        # the actual unnormalized attractor count remains its default sixteen.
        self.attractors = nn.ModuleDict({name: nn.ModuleList([_Attractor(config, 16) for _ in config.num_attractors])
                                        for name in self.names})
        self.conditional_log_binomial = nn.ModuleDict({item.name: _ConditionalDistribution(config, item.n_bins)
                                                       for item in config.bin_configurations})
        self.math, self.softmax, self.interpolate = _Arithmetic(), Softmax(dim=-1), Interpolate()

    def forward(self, features, bottleneck, blocks):
        hidden = self.conv2(bottleneck)
        logits = self.mlp_classifier(self.patch_transformer(hidden)[:, 0])
        vote = self.softmax(self.math.reduce(logits, 0, "sum").unsqueeze(0))
        starts = torch.zeros(1, dtype=torch.int32, device=vote.device)
        ends = torch.full_like(starts, len(self.names))
        name = self.names[_deterministic_topk_indices(vote.float(), starts, ends, 1).item()]
        centers = self.seed_bin_regressors[name](hidden)
        previous_embedding = self.seed_projector(hidden)
        for projector, attractor, feature in zip(self.projectors, self.attractors[name], blocks):
            embedding = projector(feature)
            centers = attractor(embedding, centers, previous_embedding)
            previous_embedding = embedding
        centers = self.interpolate(centers, size=features.shape[-2:], mode="bilinear", align_corners=True)
        embedding = self.interpolate(embedding, size=features.shape[-2:], mode="bilinear", align_corners=True)
        probabilities = self.conditional_log_binomial[name](features, embedding)
        return self.math.reduce(self.math.multiply(probabilities, centers), 1, "sum"), logits


class ZoeDepthForDepthEstimation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = BeitModel(config.backbone_config, add_pooling_layer=False)
        self.backbone.embeddings = DepthEmbeddings(config.backbone_config)
        for layer in self.backbone.encoder:
            layer.attn = _BeitAttention(config.backbone_config)
        self.position_biases = nn.ModuleList([_RelativeBias(config.backbone_config) for _ in self.backbone.encoder])
        self.out_indices, self.patch_size = config.backbone_config.out_indices, config.backbone_config.patch_size
        self.neck, self.relative_head, self.metric_head = _Neck(config), _RelativeHead(config), _MetricHead(config)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        hidden, features = self.backbone.embeddings(pixel_values), []
        height, width = (size // self.patch_size for size in pixel_values.shape[-2:])
        for index, (layer, bias) in enumerate(zip(self.backbone.encoder, self.position_biases), start=1):
            hidden = layer(hidden, attn_mask=bias(height, width))
            if index in self.out_indices:
                features.append(hidden)
        blocks, bottleneck = self.neck(features, height, width)
        # The public task computes this relative prediction even though only its
        # preceding feature map feeds the metric head; preserve that work.
        relative_depth, features = self.relative_head(blocks)
        depth, logits = self.metric_head(features, bottleneck, blocks)
        return {"predicted_depth": depth, "domain_logits": logits}


def build_from_config(config, device, dtype):
    if config.backbone_config.model_type != "beit" or config.bin_centers_type != "softplus" or config.readout_type != "project":
        raise ValueError("This case preserves the default BEiT, projected readout and softplus-bin head")
    if config.use_batch_norm_in_fusion_residual or config.add_projection or config.attractor_kind != "mean":
        raise ValueError("This case retains the checkpoint's default fusion and mean attractors")
    return ZoeDepthForDepthEstimation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    backbone, mapped = {}, {}
    for name, value in state_dict.items():
        if name.startswith("backbone."):
            name = name.removeprefix("backbone.")
            if ".relative_position_bias.relative_position_bias_table" in name:
                index = int(name.split(".")[2])
                mapped[f"position_biases.{index}.table"] = value
            else:
                backbone[name] = value
        else:
            mapped[name] = value
    load_beit(model.backbone, backbone, config.backbone_config)
    for index, layer in enumerate(model.neck.reassemble_stage.layers):
        if isinstance(layer.resize, NonoverlappingTransposeConv):
            prefix = f"neck.reassemble_stage.layers.{index}.resize."
            weight = mapped.pop(prefix + "weight")
            mapped[prefix + "projection.weight"] = weight.permute(1, 2, 3, 0).reshape(-1, weight.shape[0]).contiguous()
            mapped[prefix + "projection.bias"] = mapped.pop(prefix + "bias").repeat_interleave(layer.resize.factor ** 2)
    mapped.update({"backbone." + key: value for key, value in model.backbone.state_dict().items()})
    model.load_state_dict(mapped, strict=True)
