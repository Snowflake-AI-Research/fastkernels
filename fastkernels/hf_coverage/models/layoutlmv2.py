"""LayoutLMv2 document encoding, including its full ResNeXt-101/FPN visual path.

The candidate has no Detectron2 dependency. Its convolutions, grouped bottlenecks,
normalization, resize, pooling, attention and MLP use existing FK operations.
"""
import torch
from torch import nn

from ..runner import Workload
from .layoutlm import Pooler
from .layoutlmv3 import TextEmbeddings
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention


class ConvNorm(Conv2d):
    def __init__(self, incoming, outgoing, kernel, stride=1, padding=0, groups=1):
        super().__init__(incoming, outgoing, kernel, stride=stride, padding=padding, groups=groups, bias=False)
        # Detectron2 FrozenBN's inference path calls F.batch_norm. The existing
        # BatchNorm2d operation in eval mode gives that exact path, including
        # low-precision rounding. No batch-statistics update is performed.
        self.norm = BatchNorm2d(outgoing)
        self.norm.num_batches_tracked = None

    def forward(self, x):
        return self.norm(super().forward(x))


class Bottleneck(nn.Module):
    def __init__(self, incoming, outgoing, middle, stride):
        super().__init__()
        self.conv1 = ConvNorm(incoming, middle, 1)
        self.conv2 = ConvNorm(middle, middle, 3, stride=stride, padding=1, groups=32)
        self.conv3 = ConvNorm(middle, outgoing, 1)
        self.shortcut = ConvNorm(incoming, outgoing, 1, stride=stride) if incoming != outgoing else nn.Identity()
        self.relu = ReLU()

    def forward(self, x):
        y = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        return self.relu(y + self.shortcut(x))


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvNorm(3, 64, 7, stride=2, padding=3)
        self.relu, self.pool = ReLU(), MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.relu(self.conv1(x)))


class ResNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem()
        incoming = 64
        for stage, depth in enumerate((3, 4, 23, 3), start=2):
            outgoing = 256 * 2**(stage-2)
            blocks = [Bottleneck(incoming, outgoing, outgoing, 1 if stage == 2 else 2)]
            blocks += [Bottleneck(outgoing, outgoing, outgoing, 1) for _ in range(depth-1)]
            setattr(self, f'res{stage}', nn.Sequential(*blocks))
            incoming = outgoing

    def forward(self, x):
        x = self.stem(x)
        features = {}
        for stage in range(2, 6):
            x = getattr(self, f'res{stage}')(x)
            features[f'res{stage}'] = x
        return features


class FeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottom_up = ResNeXt()
        for stage in range(2, 6):
            setattr(self, f'fpn_lateral{stage}', Conv2d(256*2**(stage-2), 256, 1))
            setattr(self, f'fpn_output{stage}', Conv2d(256, 256, 3, padding=1))
        self.resize, self.top_block = Interpolate(), MaxPool2d(1, stride=2)

    def forward(self, x):
        bottom = self.bottom_up(x)
        previous = self.fpn_lateral5(bottom['res5'])
        features = {'p5': self.fpn_output5(previous)}
        for stage in (4, 3, 2):
            previous = getattr(self, f'fpn_lateral{stage}')(bottom[f'res{stage}']) + self.resize(previous, scale_factor=2.0)
            features[f'p{stage}'] = getattr(self, f'fpn_output{stage}')(previous)
        # Native computes every FPN output, including p6, before selecting p2.
        features['p6'] = self.top_block(features['p5'])
        return features


class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = config.detectron2_config_args
        self.register_buffer('pixel_mean', torch.tensor(args.get('MODEL.PIXEL_MEAN', [103.530, 116.280, 123.675])).view(3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(args['MODEL.PIXEL_STD']).view(3, 1, 1), persistent=False)
        self.backbone = FeaturePyramid()
        self.pool_shape = tuple(config.image_feature_pool_shape[:2])
        self.pool = GlobalAvgPool2d()

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError('LayoutLMv2 expects NCHW three-channel image with sides divisible by32')
        # Fixed processor constants; subtraction and division retain native
        # dtype stores rather than folding them into the learned convolution.
        x = self.backbone((image - self.pixel_mean) / self.pixel_std)['p2']
        height, width = x.shape[-2:]
        rows, cols = self.pool_shape
        # Adaptive averaging is exactly global averaging within each spatial
        # bin; overlapping boundary bins are retained for indivisible shapes.
        return torch.stack([
            self.pool(x[:, :, r*height//rows:((r+1)*height+rows-1)//rows,
                        c*width//cols:((c+1)*width+cols-1)//cols])
            for r in range(rows) for c in range(cols)], dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        self.qkv_linear = Linear(config.hidden_size, 3*config.hidden_size, bias=False)
        self.q_bias = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.v_bias = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.matmul, self.softmax = BatchMatMul(), Softmax(dim=-1)

    def forward(self, x, relative, spatial, mask):
        batch, length, dim = x.shape
        q, k, v = self.qkv_linear(x).chunk(3, dim=-1)
        q, v = q + self.q_bias, v + self.v_bias
        q, k, v = [a.reshape(batch, length, self.heads, self.width).transpose(1, 2)
                   .reshape(batch*self.heads, length, self.width) for a in (q, k, v)]
        scores = self.matmul(q / self.width**0.5, k.transpose(1, 2)).reshape(batch, self.heads, length, length)
        scores = scores + relative
        scores = scores + spatial
        scores = scores.float().masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = self.softmax(scores).to(v.dtype).reshape(batch*self.heads, length, length)
        return self.matmul(probs, v).reshape(batch, self.heads, length, self.width).transpose(1, 2).reshape(batch, length, dim)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.ModuleDict({'self': SpatialAttention(config), 'output': EncoderSelfOutput(config)})
        self.intermediate, self.output = EncoderIntermediate(config), EncoderOutput(config)

    def forward(self, x, relative, spatial, mask):
        x = self.attention['output'](self.attention['self'](x, relative, spatial, mask), x)
        return self.output(self.intermediate(x), x)


class LayoutLMv2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = TextEmbeddings(config)
        self.visual = VisualBackbone(config)
        self.visual_proj = Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        self.visual_LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.encoder = nn.ModuleDict({'layer': nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)]),
            'rel_pos_bias': Embedding(config.rel_pos_bins, config.num_attention_heads),
            'rel_pos_x_bias': Embedding(config.rel_2d_pos_bins, config.num_attention_heads),
            'rel_pos_y_bias': Embedding(config.rel_2d_pos_bins, config.num_attention_heads)})
        self.pooler = Pooler(config)

    def spatial_embeddings(self, bbox):
        e = self.embeddings
        return torch.cat((e.x_position_embeddings(bbox[..., 0]), e.y_position_embeddings(bbox[..., 1]),
            e.x_position_embeddings(bbox[..., 2]), e.y_position_embeddings(bbox[..., 3]),
            e.h_position_embeddings(bbox[..., 3]-bbox[..., 1]),
            e.w_position_embeddings(bbox[..., 2]-bbox[..., 0])), dim=-1)

    def relative_bias(self, positions, name, bins, distance):
        differences = positions.unsqueeze(-2)-positions.unsqueeze(-1)
        buckets = T5SelfAttention._relative_position_bucket(differences, num_buckets=bins, max_distance=distance)
        return self.encoder[name](buckets).permute(0, 3, 1, 2).contiguous()

    def forward(self, input_ids, bbox, image, attention_mask=None, token_type_ids=None, position_ids=None):
        if self.training:
            raise RuntimeError('LayoutLMv2 coverage supports inference only')
        batch, length = input_ids.shape
        e, config = self.embeddings, self.config
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device).expand(batch, -1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        text = e.word_embeddings(input_ids) + e.position_embeddings(position_ids)
        text = text + self.spatial_embeddings(bbox)
        text = e.LayerNorm(text + e.token_type_embeddings(token_type_ids))
        rows, cols = self.visual.pool_shape
        y, x = torch.meshgrid(torch.arange(rows, device=bbox.device), torch.arange(cols, device=bbox.device), indexing='ij')
        visual_boxes = torch.stack((x*1000//cols, y*1000//rows, (x+1)*1000//cols, (y+1)*1000//rows), dim=-1).reshape(1, -1, 4).expand(batch, -1, -1)
        visual_positions = torch.arange(rows*cols, device=input_ids.device).expand(batch, -1)
        visual = self.visual_proj(self.visual(image)) + e.position_embeddings(visual_positions)
        visual = self.visual_LayerNorm(visual + self.spatial_embeddings(visual_boxes))
        hidden = torch.cat((text, visual), dim=1)
        positions = torch.cat((position_ids, visual_positions), dim=1)
        boxes = torch.cat((bbox, visual_boxes), dim=1)
        relative = self.relative_bias(positions, 'rel_pos_bias', config.rel_pos_bins, config.max_rel_pos)
        spatial = self.relative_bias(boxes[..., 0], 'rel_pos_x_bias', config.rel_2d_pos_bins, config.max_rel_2d_pos)
        spatial = spatial + self.relative_bias(boxes[..., 3], 'rel_pos_y_bias', config.rel_2d_pos_bins, config.max_rel_2d_pos)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = torch.cat((attention_mask, torch.ones(batch, rows*cols, dtype=attention_mask.dtype, device=input_ids.device)), dim=1)
        mask = mask[:, None, None, :] == 0
        for layer in self.encoder['layer']:
            hidden = layer(hidden, relative, spatial, mask)
        return {'last_hidden_state': hidden, 'pooler_output': self.pooler(hidden)}


def build_from_config(config, device, dtype):
    if (config.hidden_act != 'gelu' or not config.fast_qkv or not config.has_relative_attention_bias
            or not config.has_spatial_attention_bias or config.has_visual_segment_embedding
            or config.output_attentions or config.output_hidden_states or config.chunk_size_feed_forward):
        raise ValueError('LayoutLMv2 candidate preserves selected fast-QKV/both-bias/no-segment default inference')
    if 4*config.coordinate_size+2*config.shape_size != config.hidden_size or config.image_feature_pool_shape[-1] != 256:
        raise ValueError('Spatial embeddings must match hidden width and FPN output must have256 channels')
    args = config.detectron2_config_args
    required = {'MODEL.BACKBONE.NAME':'build_resnet_fpn_backbone', 'MODEL.RESNETS.DEPTH':101,
        'MODEL.RESNETS.NUM_GROUPS':32, 'MODEL.RESNETS.WIDTH_PER_GROUP':8, 'MODEL.RESNETS.STRIDE_IN_1X1':False,
        'MODEL.RESNETS.OUT_FEATURES':['res2','res3','res4','res5'], 'MODEL.FPN.IN_FEATURES':['res2','res3','res4','res5']}
    defaults = {'MODEL.RESNETS.NORM':'FrozenBN','MODEL.RESNETS.RES2_OUT_CHANNELS':256,
        'MODEL.RESNETS.STEM_OUT_CHANNELS':64,'MODEL.RESNETS.RES5_DILATION':1,
        'MODEL.RESNETS.DEFORM_ON_PER_STAGE':[False]*4,'MODEL.FPN.OUT_CHANNELS':256,
        'MODEL.FPN.NORM':'','MODEL.FPN.FUSE_TYPE':'sum'}
    if any(args.get(k) != v for k,v in required.items()) or any(args.get(k,v) != v for k,v in defaults.items()):
        raise ValueError('Visual path requires selected full ResNeXt10132x8d/FrozenBN/sum-FPN configuration')
    return LayoutLMv2Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    remaining, mapped = dict(state_dict), {}
    for destination in model.state_dict():
        source = destination.replace('.emb.weight', '.weight')
        value = remaining.pop(source)
        if source.startswith('encoder.rel_pos'):
            value = value.T
        mapped[destination] = value
    if remaining:
        raise ValueError(f'Unmapped LayoutLMv2 state: {sorted(remaining)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    del config
    return {'forward': Workload(run=lambda: model(**inputs))}
