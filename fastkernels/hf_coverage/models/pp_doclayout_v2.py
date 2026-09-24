"""PP-DocLayoutV2 detector and active reading-order graph from existing operations.

Predicted float-to-long coordinates are an admitted cast, not input metadata.
Boolean validity sorting reuses VJEPA2Predictor.unsort_tokens unchanged; this
internal capability has no standalone optimization interface. Duplicate validity
keys expand beyond its native inverse-permutation caller domain; this reuse is
provisional and does not promise future task optimizations preserve that domain.
"""
import math
import torch
from torch import nn
from .d_fine import Attention, MLP, backbone, norm
from .pp_doclayout_v3 import Encoder, DecoderLayer, GlobalPointer
from .vits import DurationArithmetic
from ..patches.dfine_clamp import DFineClamp
from ..patches.detector_topk import DetectorTopK
from ..patches.ratio_log import RatioLog
from ..runner import Workload
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.sinusoidal_embed import SinusoidalEmbed
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid
from fastkernels.tasks.baseline.L3.vjepa2_predictor import VJEPA2Predictor


def validity_order(mask):
    values = torch.arange(mask.shape[1], device=mask.device)[None, :, None].expand(mask.shape[0], -1, -1)
    return VJEPA2Predictor.unsort_tokens(None, values, -mask.to(torch.int8)).squeeze(-1)


class DetectorModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.backbone = nn.Module()
        self.backbone.model = backbone(c)
        self.encoder_input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(w, c.encoder_hidden_dim, 1, bias=False),
                    BatchNorm2d(c.encoder_hidden_dim)
                ) for w in c.encoder_in_channels
            ]
        )
        self.encoder = Encoder(c, with_masks=False)
        self.denoising_class_embed = Embedding(c.num_labels, c.d_model)
        self.enc_output = nn.Sequential(Linear(c.d_model, c.d_model), norm(c.d_model, c.layer_norm_eps))
        self.enc_score_head = Linear(c.d_model, c.num_labels)
        self.enc_bbox_head = MLP(c.d_model, c.d_model, 4, 3)
        self.decoder_input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(w, c.d_model, 1, bias=False),
                    BatchNorm2d(c.d_model, eps=c.batch_norm_eps)
                ) for w in c.decoder_in_channels
            ]
        )
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c.decoder_layers)])
        self.decoder.query_pos_head = MLP(4, 2*c.d_model, c.d_model, 2)
        self.decoder.class_embed = nn.ModuleList(
            [Linear(c.d_model, c.num_labels) for _ in range(c.decoder_layers)]
        )
        self.decoder.bbox_embed = nn.ModuleList(
            [MLP(c.d_model, c.d_model, 4, 3) for _ in range(c.decoder_layers)]
        )
        self.sigmoid, self.math, self.topk = Sigmoid(), DurationArithmetic(), DetectorTopK()

    def forward(self, pixel_values, pixel_mask=None):
        c = self.c
        features = list(self.backbone.model(pixel_values).values())
        encoded, _ = self.encoder([p(x) for p, x in zip(self.encoder_input_proj, features)])
        sources = [p(x) for p, x in zip(self.decoder_input_proj, encoded)]
        shapes = [tuple(x.shape[-2:]) for x in sources]
        memory = torch.cat([x.flatten(2).transpose(1, 2) for x in sources], 1)
        anchors = []
        for level, (height, width) in enumerate(shapes):
            y, x = torch.meshgrid(
                torch.arange(height, device=memory.device).to(memory.dtype),
                torch.arange(width, device=memory.device).to(memory.dtype),
                indexing='ij'
            )
            xy = torch.stack((x, y), -1)[None] + .5
            xy[..., 0] /= width
            xy[..., 1] /= height
            anchors.append(
                torch.cat((xy, torch.ones_like(xy)*.05*(2.**level)), -1).reshape(1, -1, 4)
            )
        anchors = torch.cat(anchors, 1)
        valid = ((anchors > .01) & (anchors < .99)).all(-1, keepdim=True)
        anchors = torch.log(anchors/(1-anchors)).masked_fill(~valid, torch.finfo(memory.dtype).max)
        projected = self.enc_output(self.math.mul(memory, valid.to(memory.dtype)))
        classes, coordinates = self.enc_score_head(projected), self.enc_bbox_head(projected) + anchors
        class_index = self.math.top1(classes)
        scores = classes.gather(-1, class_index[..., None]).squeeze(-1)
        _, indices = self.topk(scores, c.num_queries)
        gather = lambda value: value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))
        initial = gather(coordinates)
        references, hidden = self.sigmoid(initial), gather(projected)
        states, boxes, logits = [], [], []
        for i, layer in enumerate(self.decoder.layers):
            hidden = layer(
                hidden,
                self.decoder.query_pos_head(references),
                memory,
                references,
                shapes
            )
            references = self.sigmoid(self.decoder.bbox_embed[i](hidden) + inverse_sigmoid(references))
            states.append(hidden)
            boxes.append(references)
            logits.append(self.decoder.class_embed[i](hidden))
        output = dict(
            last_hidden_state=hidden,
            intermediate_hidden_states=torch.stack(states, 1),
            intermediate_logits=torch.stack(logits, 1),
            intermediate_reference_points=torch.stack(boxes, 1),
            init_reference_points=initial,
            enc_topk_logits=gather(classes),
            enc_topk_bboxes=self.sigmoid(initial),
            enc_outputs_class=classes,
            enc_outputs_coord_logits=coordinates
        )
        for i, value in enumerate(encoded):
            output[f'encoder_last_hidden_state.{i}'] = value
        return output


class SpatialRelation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale, self.width = c.relation_bias_scale, c.relation_bias_embed_dim
        self.pos_proj = Conv2d(4*self.width, c.num_attention_heads, 1)
        self.sine = SinusoidalEmbed(self.width)
        self.sine.sinusoid_freq = 1 / (
            c.relation_bias_theta ** (
                torch.arange(0, self.width, 2, dtype=torch.int64).float()/(self.width//2)
            )
        )
        self.math, self.log, self.relu, self.clamp = DurationArithmetic(), RatioLog(), ReLU(), DFineClamp()

    def forward(self, boxes):
        minimum, maximum = boxes[..., :2], boxes[..., 2:]
        sizes = self.clamp(maximum-minimum, 1e-3, float('inf'))
        center = (minimum+maximum)*.5
        delta = center[:, :, None]-center[:, None, :]
        distance = self.relu(delta)+self.relu(-delta)
        source, target = sizes[:, :, None]+1e-5, sizes[:, None, :]+1e-5
        relative = torch.cat(
            (
                self.log(self.math.div(distance, source)+1., torch.ones_like(distance)),
                self.log(source, target)
            ),
            -1
        )
        phase = (relative*self.scale).reshape(-1)
        position = self.sine(phase).reshape(*relative.shape, self.width).flatten(-2).to(relative.dtype)
        return self.pos_proj(position.permute(0, 3, 1, 2))


class ReadingEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        for name, count, width in [
            ('word', c.vocab_size, c.hidden_size),
            ('token_type', c.type_vocab_size, c.hidden_size),
            ('position', c.max_position_embeddings, c.hidden_size),
            ('x_position', c.max_2d_position_embeddings, c.coordinate_size),
            ('y_position', c.max_2d_position_embeddings, c.coordinate_size),
            ('h_position', c.max_2d_position_embeddings, c.shape_size),
            ('w_position', c.max_2d_position_embeddings, c.shape_size)
        ]:
            setattr(self, name+'_embeddings', Embedding(count, width))
        self.spatial_proj = Linear(4*c.coordinate_size+2*c.shape_size, c.hidden_size)
        self.norm, self.clamp = norm(c.hidden_size, c.layer_norm_eps), DFineClamp()
        self.math = DurationArithmetic()

    def forward(self, ids, boxes):
        # IDs depend on predicted valid counts; compare through the existing op.
        values = ids.float()
        padding = torch.full_like(values, self.c.pad_token_id)
        nonpad = (self.math.less(values, padding) | self.math.less(padding, values)).float()
        positions = self.math.mul(self.math.prefix(nonpad), nonpad).long()+self.c.pad_token_id
        hidden = self.word_embeddings(ids)+self.token_type_embeddings(torch.zeros_like(ids))
        hidden = hidden+self.position_embeddings(positions)
        # Current predicted boxes remain activation-dependent; this cast is admitted.
        boxes = boxes.long()
        spatial = torch.cat(
            (
                self.x_position_embeddings(boxes[..., 0]),
                self.y_position_embeddings(boxes[..., 1]),
                self.x_position_embeddings(boxes[..., 2]),
                self.y_position_embeddings(boxes[..., 3]),
                self.h_position_embeddings(self.clamp(boxes[..., 3]-boxes[..., 1], 0, 1023)),
                self.w_position_embeddings(self.clamp(boxes[..., 2]-boxes[..., 0], 0, 1023))
            ),
            -1
        )
        return hidden+self.spatial_proj(spatial)


class ReadingAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.width = c.num_attention_heads, c.hidden_size//c.num_attention_heads
        for name in ('query', 'key', 'value'):
            setattr(self, name, Linear(c.hidden_size, c.hidden_size))
        self.bmm, self.softmax, self.math = BMM(), Softmax(), DurationArithmetic()

    def forward(self, hidden, bias, mask):
        b, n, w = hidden.shape
        def split(x):
            return x.reshape(b, n, self.heads, self.width).transpose(1, 2)
        q, k, v = [split(getattr(self, name)(hidden)) for name in ('query', 'key', 'value')]
        scores = self.bmm(q/math.sqrt(self.width), k.transpose(-2, -1))+bias+mask
        scaled = scores/32
        maximum = scaled.gather(-1, self.math.top1(scaled)[..., None])
        probabilities = self.softmax((scaled-maximum)*32)
        return self.bmm(probabilities, v).transpose(1, 2).reshape(b, n, w)


class ReadingLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = ReadingAttention(c)
        self.attention.output = EncoderSelfOutput(c)
        self.intermediate, self.output = EncoderIntermediate(c), EncoderOutput(c)

    def forward(self, hidden, bias, mask):
        hidden = self.attention.output(self.attention.self(hidden, bias, mask), hidden)
        return self.output(self.intermediate(hidden), hidden)


class ReadingOrder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c, self.math = c, DurationArithmetic()
        self.embeddings = ReadingEmbeddings(c)
        self.label_embeddings = Embedding(c.num_classes, c.hidden_size)
        self.label_features_projection = Linear(c.hidden_size, c.hidden_size)
        self.encoder = nn.Module()
        self.encoder.rel_pos_x_bias = Linear(c.rel_2d_pos_bins, c.num_attention_heads, bias=False)
        self.encoder.rel_pos_y_bias = Linear(c.rel_2d_pos_bins, c.num_attention_heads, bias=False)
        self.encoder.rel_bias_module = SpatialRelation(c)
        self.encoder.layer = nn.ModuleList([ReadingLayer(c) for _ in range(c.num_hidden_layers)])
        self.relative_head = GlobalPointer(c.hidden_size, c.global_pointer_head_size)

    def forward(self, boxes, labels, valid):
        c = self.c
        b, n = valid.shape
        counts = self.math.sum_last(valid.float()).long()
        columns = torch.arange(n+2, device=boxes.device)[None].expand(b, -1)
        is_pred = self.math.less(columns.float(), (counts+1)[:, None].float()) & (columns >= 1)
        ids = torch.full((b, n+2), c.pad_token_id, device=boxes.device, dtype=torch.long)
        ids[:, 0] = c.start_token_id
        ids[is_pred] = c.pred_token_id
        ids[torch.arange(b, device=boxes.device), counts+1] = c.end_token_id
        padded = torch.cat((boxes.new_zeros(b, 1, 4), boxes, boxes.new_zeros(b, 1, 4)), 1)
        label = self.label_features_projection(self.label_embeddings(labels))
        label = torch.cat(
            (
                label.new_zeros(b, 1, label.shape[-1]),
                label,
                label.new_zeros(b, 1, label.shape[-1])
            ),
            1
        )
        hidden = self.embeddings.norm(self.embeddings(ids, padded)+label)
        permitted = self.math.less(columns.float(), (counts+2)[:, None].float())
        mask = hidden.new_zeros((b, 1, 1, n+2)).masked_fill(
            ~permitted[:, None, None], torch.finfo(hidden.dtype).min
        )
        bias = self.encoder.rel_bias_module(padded)
        for layer in self.encoder.layer:
            hidden = layer(hidden, bias, mask)
        return self.relative_head(hidden[:, 1:n+1])


class PPDocLayoutV2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c, self.model, self.reading_order = c, DetectorModel(c), ReadingOrder(c.reading_order_config)
        self.math, self.sigmoid, self.clamp = DurationArithmetic(), Sigmoid(), DFineClamp()

    def forward(self, pixel_values, pixel_mask=None):
        output = self.model(pixel_values, pixel_mask)
        raw = output['intermediate_reference_points'][:, -1]
        logits = output['intermediate_logits'][:, -1]
        center, size = raw.split(2, -1)
        boxes = self.clamp(torch.cat((center-.5*size, center+.5*size), -1)*1000, 0., 1000.)
        classes = self.math.top1(logits)
        probabilities = self.sigmoid(logits.gather(-1, classes[..., None]).squeeze(-1))
        thresholds = torch.tensor(
            self.c.class_thresholds,
            dtype=torch.float32,
            device=logits.device
        )[classes]
        valid = ~self.math.less(probabilities, thresholds)
        indices = validity_order(valid)
        gather = lambda value: value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))
        sorted_valid = valid.gather(1, indices)
        padded_boxes = gather(boxes).masked_fill(~sorted_valid[..., None], 0)
        labels = classes.gather(1, indices).masked_fill(~sorted_valid, 0)
        order = torch.tensor(self.c.class_order, dtype=torch.int32, device=logits.device)
        labels = order[labels]
        output.update(
            logits=gather(logits),
            pred_boxes=gather(raw),
            order_logits=self.reading_order(padded_boxes, labels, valid)
        )
        return output


def build_from_config(config, device, dtype):
    config.num_labels = len(config.id2label)
    r = config.reading_order_config
    if (
        config.hidden_expansion != 1
        or config.normalize_before
        or config.learn_initial_query
        or config.anchor_image_size is not None
        or r.has_relative_attention_bias
        or not r.has_spatial_attention_bias
        or r.hidden_act != 'gelu'
    ):
        raise ValueError(
            'PP-DocLayoutV2 requires selected native detector/spatial-reading settings'
        )
    model = PPDocLayoutV2(config).to(device=device, dtype=dtype).eval()
    if device.type == "cuda":
        # Keep the native attention route explicit: dependency imports disable
        # cuDNN globally, and its rounding affects the discrete proposal ranking.
        for module in model.modules():
            if isinstance(module, Attention):
                module.core = DenseAttention(backend="cudnn")
    # SinusoidalEmbed uses fixed FP32 frequencies; do not round those with weights.
    relation = model.reading_order.encoder.rel_bias_module
    relation.sine.sinusoid_freq = (
        1 / (
            r.relation_bias_theta ** (
                torch.arange(0, r.relation_bias_embed_dim, 2, dtype=torch.int64).float()
                / (r.relation_bias_embed_dim // 2)
            )
        )
    ).to(device)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, value in model.state_dict().items():
        source = (
            name.replace('.LayerNorm.', '.norm.')
            .replace('.emb.weight', '.weight')
            .replace('.self_attn.out_proj.', '.self_attn.o_proj.')
        )
        source = source.replace('.mlp.layers.0.', '.mlp.fc1.').replace('.mlp.layers.1.', '.mlp.fc2.')
        tensor = state_dict[source]
        if tensor.shape != value.shape:
            raise ValueError(
                f'PP-DocLayoutV2 shape mismatch {source}: {tensor.shape} != {value.shape}'
            )
        mapped[name] = tensor
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f'Unmapped PP-DocLayoutV2 weights: {sorted(set(state_dict)-used)}')
    model.load_state_dict(mapped, strict=True, assign=True)


def make_workloads(model, inputs, config, *, case=None):
    return {'forward': Workload(run=lambda: model(**inputs))}
