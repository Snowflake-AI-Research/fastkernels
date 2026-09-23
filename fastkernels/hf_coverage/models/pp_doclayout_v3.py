"""PP-DocLayoutV3 detection, mask refinement, and per-layer reading order."""
import math
import torch
from torch import nn
from .d_fine import AIFI, Attention, ConvNorm, MLP, RepBlock, activation, backbone, norm
from .deformable_detr import _SamplingAttention
from ..patches.codec_top1 import CodecTop1
from ..patches.detector_topk import DetectorTopK
from ..runner import Workload
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid


class CSP(nn.Module):
    def __init__(self, c):
        super().__init__()
        width = c.encoder_hidden_dim
        self.conv1, self.conv2 = [ConvNorm(c, 2*width, width, act=c.activation_function) for _ in range(2)]
        self.bottlenecks = nn.Sequential(*[RepBlock(c, width) for _ in range(3)])
        self.conv3 = nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class ConvLayer(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.convolution = Conv2d(source, target, 3, padding=1, bias=False)
        self.normalization = BatchNorm2d(target)
        self.activation = activation('silu')

    def forward(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolate = Interpolate()

    def forward(self, x):
        return self.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


class ScaleHead(nn.Module):
    def __init__(self, source, target, stride, base):
        super().__init__()
        layers = []
        for i in range(max(1, int(math.log2(stride)-math.log2(base)))):
            layers.append(ConvLayer(source if i == 0 else target, target))
            if stride != base:
                layers.append(Upsample())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MaskFPN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.order = sorted(range(len(c.feat_strides)), key=lambda i: c.feat_strides[i])
        base = min(c.feat_strides)
        self.scale_heads = nn.ModuleList([ScaleHead(c.encoder_hidden_dim, c.mask_feature_channels[0], c.feat_strides[i], base) for i in self.order])
        self.output_conv = ConvLayer(*c.mask_feature_channels)
        self.interpolate = Interpolate()

    def forward(self, inputs):
        outputs = [layer(inputs[i]) for layer, i in zip(self.scale_heads, self.order)]
        x = outputs[0]
        for value in outputs[1:]:
            x = x + self.interpolate(value, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.output_conv(x)


class MaskOutput(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.base_conv = ConvLayer(c.mask_feature_channels[1], c.mask_feature_channels[1])
        self.conv = Conv2d(c.mask_feature_channels[1], c.num_prototypes, 1)

    def forward(self, x):
        return self.conv(self.base_conv(x))


class Encoder(nn.Module):
    def __init__(self, c, with_masks=True):
        super().__init__()
        self.c, self.with_masks = c, with_masks
        width = c.encoder_hidden_dim
        count = len(c.encoder_in_channels)-1
        self.aifi = nn.ModuleList([AIFI(c) for _ in c.encode_proj_layers])
        self.lateral_convs = nn.ModuleList([ConvNorm(c, width, width, act=c.activation_function) for _ in range(count)])
        self.downsample_convs = nn.ModuleList([ConvNorm(c, width, width, 3, 2, act=c.activation_function) for _ in range(count)])
        self.fpn_blocks, self.pan_blocks = [nn.ModuleList([CSP(c) for _ in range(count)]) for _ in range(2)]
        self.interpolate = Interpolate()
        if with_masks:
            self.mask_feature_head = MaskFPN(c)
            self.encoder_mask_lateral = ConvLayer(c.x4_feat_dim, c.mask_feature_channels[1])
            self.encoder_mask_output = MaskOutput(c)

    def forward(self, features, low_feature=None):
        for layer, index in zip(self.aifi, self.c.encode_proj_layers):
            features[index] = layer(features[index])
        fpn = [features[-1]]
        for i, (lateral, block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            fpn[-1] = lateral(fpn[-1])
            up = self.interpolate(fpn[-1], scale_factor=2., mode='nearest')
            fpn.append(block(torch.cat((up, features[-i-2]), dim=1)))
        fpn.reverse()
        pan = [fpn[0]]
        for i, (down, block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            pan.append(block(torch.cat((down(pan[-1]), fpn[i+1]), dim=1)))
        if not self.with_masks:
            return pan, None
        masks = self.interpolate(self.mask_feature_head(pan), scale_factor=2, mode='bilinear', align_corners=False)
        masks = self.encoder_mask_output(masks + self.encoder_mask_lateral(low_feature))
        return pan, masks


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        width = c.d_model
        self.self_attn = Attention(width, c.decoder_attention_heads)
        self.encoder_attn = _SamplingAttention(width, c.decoder_attention_heads, c.num_feature_levels, c.decoder_n_points)
        self.self_attn_layer_norm, self.encoder_attn_layer_norm, self.final_layer_norm = [norm(width, c.layer_norm_eps) for _ in range(3)]
        self.mlp = MLP(width, c.decoder_ffn_dim, width, 2, c.decoder_activation_function)
        self.levels = c.num_feature_levels

    def forward(self, x, pos, memory, refs, shapes):
        x = self.self_attn_layer_norm(x + self.self_attn(x, position_embeddings=pos)[0])
        references = refs[:, :, None].expand(-1, -1, self.levels, -1)
        x = self.encoder_attn_layer_norm(x + self.encoder_attn(x, memory, pos, references, shapes))
        return self.final_layer_norm(x + self.mlp(x))


class GlobalPointer(nn.Module):
    def __init__(self, width, head_size):
        super().__init__()
        self.head_size = head_size
        self.dense, self.bmm = Linear(width, 2*head_size), BMM()

    def forward(self, x):
        query, key = self.dense(x).reshape(*x.shape[:2], 2, self.head_size).unbind(2)
        scores = self.bmm(query, key.transpose(-2, -1)) / self.head_size**0.5
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device).tril().bool()
        return scores.masked_fill(mask[None], -1e4)


class MaskBoxes(nn.Module):
    """Strict-positive mask selection and four existing segmented extrema."""
    def __init__(self):
        super().__init__()
        self.top1, self.reduce = CodecTop1(), SegmentCSR()

    def forward(self, logits, dtype):
        mask = self.top1(torch.stack((torch.zeros_like(logits), logits), dim=-1)).bool()
        height, width = mask.shape[-2:]
        shape = mask.shape[:-2]
        count = height*width
        offsets = torch.arange(0, mask.numel()+1, count, device=mask.device, dtype=torch.long)
        coords = torch.meshgrid(torch.arange(height, device=mask.device), torch.arange(width, device=mask.device), indexing='ij')
        bounds = []
        for grid, size in ((coords[1], width), (coords[0], height)):
            expanded = grid.to(dtype).expand_as(mask)
            maximum = self.reduce(expanded.masked_fill(~mask, 0).flatten(), offsets, reduce='max').reshape(shape)+1
            minimum = self.reduce(expanded.masked_fill(~mask, torch.finfo(dtype).max).flatten(), offsets, reduce='min').reshape(shape)
            bounds.append((minimum/size, maximum/size))
        nonempty = self.reduce(mask.float().flatten(), offsets, reduce='max').reshape(shape).to(torch.int64).bool()
        (left, right), (top, bottom) = bounds
        boxes = torch.stack(((left+right)/2, (top+bottom)/2, right-left, bottom-top), dim=-1)
        return boxes.masked_fill(~nonempty[..., None], 0)


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.backbone = nn.Module()
        self.backbone.model = backbone(c)
        self.encoder_input_proj = nn.ModuleList([nn.Sequential(Conv2d(w, c.encoder_hidden_dim, 1, bias=False), BatchNorm2d(c.encoder_hidden_dim)) for w in c.encoder_in_channels])
        self.encoder = Encoder(c)
        self.denoising_class_embed = nn.Embedding(c.num_labels, c.d_model)
        self.enc_output = nn.Sequential(Linear(c.d_model, c.d_model), norm(c.d_model, c.layer_norm_eps))
        self.enc_score_head = Linear(c.d_model, c.num_labels)
        self.enc_bbox_head = MLP(c.d_model, c.d_model, 4, 3)
        self.decoder_input_proj = nn.ModuleList([nn.Sequential(Conv2d(w, c.d_model, 1, bias=False), BatchNorm2d(c.d_model, eps=c.batch_norm_eps)) for w in c.decoder_in_channels])
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c.decoder_layers)])
        self.decoder.query_pos_head = MLP(4, 2*c.d_model, c.d_model, 2)
        self.decoder.class_embed = Linear(c.d_model, c.num_labels)
        self.decoder.bbox_embed = MLP(c.d_model, c.d_model, 4, 3)
        self.decoder_order_head = nn.ModuleList([Linear(c.d_model, c.d_model) for _ in range(c.decoder_layers)])
        self.decoder_global_pointer = GlobalPointer(c.d_model, c.global_pointer_head_size)
        self.decoder_norm = norm(c.d_model, c.layer_norm_eps)
        self.mask_query_head = MLP(c.d_model, c.d_model, c.num_prototypes, 3)
        self.bmm, self.sigmoid, self.reduce, self.topk = BMM(), Sigmoid(), SegmentCSR(), DetectorTopK()
        self.mask_boxes, self.interpolate = MaskBoxes(), Interpolate()

    def forward(self, pixel_values, pixel_mask=None):
        c = self.c
        if pixel_mask is None:
            pixel_mask = torch.ones((pixel_values.shape[0], *pixel_values.shape[-2:]), device=pixel_values.device)
        features = list(self.backbone.model(pixel_values).values())
        feature_masks = [self.interpolate(pixel_mask[None].float(), size=x.shape[-2:]).bool()[0] for x in features]
        encoded, mask_features = self.encoder([p(x) for p, x in zip(self.encoder_input_proj, features[1:])], features[0])
        sources = [p(x) for p, x in zip(self.decoder_input_proj, encoded)]
        shapes = [tuple(x.shape[-2:]) for x in sources]
        memory = torch.cat([x.flatten(2).transpose(1, 2) for x in sources], dim=1)
        anchors = []
        for level, (height, width) in enumerate(shapes):
            y, x = torch.meshgrid(torch.arange(height, device=memory.device).to(memory.dtype), torch.arange(width, device=memory.device).to(memory.dtype), indexing='ij')
            xy = torch.stack((x, y), dim=-1)[None]+0.5
            xy[..., 0] /= width
            xy[..., 1] /= height
            size = torch.ones_like(xy)*0.05*(2.**level)
            anchors.append(torch.cat((xy, size), dim=-1).reshape(1, -1, 4))
        anchors = torch.cat(anchors, dim=1)
        valid = ((anchors > .01)*(anchors < .99)).all(-1, keepdim=True)
        anchors = torch.log(anchors/(1-anchors)).masked_fill(~valid, torch.finfo(memory.dtype).max)
        projected = self.enc_output(memory.masked_fill(~valid, 0))
        classes, coordinates = self.enc_score_head(projected), self.enc_bbox_head(projected)+anchors
        flat = classes.flatten()
        offsets = torch.arange(0, flat.numel()+1, c.num_labels, device=flat.device, dtype=torch.long)
        scores = self.reduce(flat, offsets, reduce='max').reshape(classes.shape[:2])
        _, indices = self.topk(scores, c.num_queries)
        gather = lambda value: value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))
        references = gather(coordinates)
        proposal_boxes, proposal_logits = self.sigmoid(references), gather(classes)
        hidden = gather(projected)
        mask_query = self.mask_query_head(self.decoder_norm(hidden))
        if c.mask_enhanced:
            masks = self.bmm(mask_query, mask_features.flatten(2)).reshape(*hidden.shape[:2], *mask_features.shape[-2:])
            references = inverse_sigmoid(self.mask_boxes(masks, references.dtype))
        initial = references
        references = self.sigmoid(references)
        states, boxes, logits, masks, orders = [], [], [], [], []
        for i, layer in enumerate(self.decoder.layers):
            position = self.decoder.query_pos_head(references)
            hidden = layer(hidden, position, memory, references, shapes)
            references = self.sigmoid(self.decoder.bbox_embed(hidden)+inverse_sigmoid(references))
            query = self.decoder_norm(hidden)
            mask_query = self.mask_query_head(query)
            masks.append(self.bmm(mask_query, mask_features.flatten(2)).reshape(*hidden.shape[:2], *mask_features.shape[-2:]))
            orders.append(self.decoder_global_pointer(self.decoder_order_head[i](query)))
            states.append(hidden)
            boxes.append(references)
            logits.append(self.decoder.class_embed(query))
        output = dict(logits=logits[-1], pred_boxes=boxes[-1], order_logits=orders[-1], out_masks=masks[-1], last_hidden_state=hidden, intermediate_hidden_states=torch.stack(states, dim=1), intermediate_logits=torch.stack(logits, dim=1), intermediate_reference_points=torch.stack(boxes, dim=1), init_reference_points=initial, enc_topk_logits=proposal_logits, enc_topk_bboxes=proposal_boxes, enc_outputs_class=classes, enc_outputs_coord_logits=coordinates)
        # Every decoder mask/order head executes, including outputs the public
        # task discards after selecting its final decoder layer.
        torch.stack(masks, dim=1)
        torch.stack(orders, dim=1)
        for i, value in enumerate(encoded):
            output[f'encoder_last_hidden_state.{i}'] = value
        return output


class Detector(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.model = Model(c)

    def forward(self, **inputs):
        return self.model(**inputs)


def build_from_config(config, device, dtype):
    config.num_labels = len(config.id2label)
    if config.hidden_expansion != 1 or config.normalize_before or config.learn_initial_query or config.anchor_image_size is not None:
        raise ValueError('PP-DocLayout composition requires selected native checkpoint settings')
    return Detector(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for name in model.state_dict():
        source = name.replace('.self_attn.out_proj.', '.self_attn.o_proj.')
        source = source.replace('.mlp.layers.0.', '.mlp.fc1.').replace('.mlp.layers.1.', '.mlp.fc2.')
        mapped[name] = state_dict[source]
    model.load_state_dict(mapped, strict=True, assign=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
