"""OWL-ViT detection from existing CLIP towers, heads and explicit adaptations."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.cosyvoice3_hifigan import CausalConvRNNF0Predictor

from .clip import ClipEncoderLayer, ClipModel, load_state_dict_into as load_clip
from ..patches.owl_l2_norm import AdditiveEpsilonL2Norm
from ..patches.product_gate import ProductGate
from ..runner import Workload


class DetectionMLP(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.dense0 = Linear(width, width)
        self.dense1 = Linear(width, width)
        self.dense2 = Linear(width, outputs)
        self.gelu = GELU()

    def forward(self, hidden):
        return self.dense2(self.gelu(self.dense1(self.gelu(self.dense0(hidden)))))


class MaskedTextLayer(ClipEncoderLayer):
    """Pass the combined causal/padding mask to the existing attention operation."""

    def __init__(self, config):
        super().__init__(config)
        self.ln_1.promote_fp32 = self.ln_2.promote_fp32 = False

    def _self_attention(self, hidden, attn_mask=None):
        batch, length, width = hidden.shape
        query, key, value = [projection(hidden).reshape(batch, length, self.n_head, self.head_dim)
                             for projection in (self.q_proj, self.k_proj, self.v_proj)]
        context = self.attn(query, key, value, attn_mask=attn_mask)
        return self.out_proj(context.reshape(batch, length, width))


def box_bias(grid):
    # Fixed position metadata, independent of image features.
    coords = torch.arange(1, grid + 1, dtype=torch.float32) / grid
    xx, yy = torch.meshgrid(coords, coords, indexing="xy")
    centers = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    sizes = torch.full_like(centers, 1.0 / grid)
    return torch.cat((torch.log(centers + 1e-4) - torch.log1p(-centers + 1e-4),
                      torch.log(sizes + 1e-4) - torch.log1p(-sizes + 1e-4)), dim=-1)


class OwlDetector(nn.Module):
    def __init__(self, config, *, objectness=False):
        super().__init__()
        width = config.vision_config.hidden_size
        self.backbone = ClipModel(config)
        self.backbone.text_model.text_model.encoder.layers = nn.ModuleList(
            [MaskedTextLayer(config.text_config) for _ in range(config.text_config.num_hidden_layers)])
        self.grid = config.vision_config.image_size // config.vision_config.patch_size
        self.layer_norm = LayerNorm(width, eps=config.vision_config.layer_norm_eps, promote_fp32=False)
        self.class_dense = Linear(width, config.text_config.hidden_size)
        self.logit_shift = Linear(width, 1)
        self.logit_scale = Linear(width, 1)
        # Reuse the unchanged existing ELU component; no standalone task exposes it.
        self.elu = CausalConvRNNF0Predictor(in_channels=1, cond_channels=1).condnet[1]
        self.class_norm = AdditiveEpsilonL2Norm(dim=-1, eps=1e-6)
        self.product = ProductGate()
        self.box_head = DetectionMLP(width, 4)
        self.objectness_head = DetectionMLP(width, 1) if objectness else None
        self.sigmoid = Sigmoid()
        self.register_buffer("box_bias", box_bias(self.grid), persistent=False)

    def multiply(self, a, b):
        return self.product(torch.cat((a, b.expand_as(a)), dim=-1))

    def forward(self, input_ids, pixel_values, attention_mask=None):
        vision_hidden, vision_pool = self.backbone.vision_model(pixel_values)
        if attention_mask is None:
            text = self.backbone.text_model(input_ids)
        else:
            tower = self.backbone.text_model.text_model
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            mask = (positions[:, None] >= positions[None, :])[None, None]
            mask = mask & attention_mask[:, None, None, :].bool()
            hidden = tower.final_layer_norm(tower.encoder(tower.embeddings(input_ids), attention_mask=mask))
            pooled = hidden[torch.arange(input_ids.shape[0], device=input_ids.device), input_ids.argmax(dim=-1)]
            text = SimpleNamespace(last_hidden_state=hidden, pooler_output=pooled)
        queries = self.backbone.normalize(self.backbone.text_projection(text.pooler_output))
        images = self.backbone.normalize(self.backbone.visual_projection(vision_pool))
        # HF computes the contrastive branch even though detection drops its logits.
        self.backbone.matmul(queries, images.t()) * self.backbone.scale
        features = self.backbone.vision_model.post_layernorm(vision_hidden)
        features = self.layer_norm(self.multiply(features[:, 1:], features[:, :1]))
        batch = features.shape[0]
        queries = queries.reshape(batch, -1, queries.shape[-1])
        classes = self.class_norm(self.class_dense(features))
        scores = self.backbone.matmul(classes, self.class_norm(queries).transpose(1, 2))
        scores = self.multiply(scores + self.logit_shift(features), self.elu(self.logit_scale(features)) + 1)
        mask = input_ids.reshape(batch, -1, input_ids.shape[-1])[..., 0] > 0
        scores = scores.masked_fill(~mask[:, None], torch.finfo(scores.dtype).min).float()
        boxes = self.box_head(features)
        boxes += self.box_bias
        outputs = {"logits": scores, "pred_boxes": self.sigmoid(boxes),
                   "image_embeds": features.reshape(batch, self.grid, self.grid, -1),
                   "text_embeds": queries, "class_embeds": classes,
                   "text_model_output.last_hidden_state": text.last_hidden_state,
                   "text_model_output.pooler_output": text.pooler_output,
                   "vision_model_output.last_hidden_state": vision_hidden,
                   "vision_model_output.pooler_output": vision_pool}
        if self.objectness_head is not None:
            outputs["objectness_logits"] = self.objectness_head(features)[..., 0]
        return outputs


def build_detector(config, device, dtype, *, objectness=False):
    if (config.text_config.hidden_act != "quick_gelu" or config.vision_config.hidden_act != "quick_gelu"
            or config.projection_dim != config.text_config.hidden_size):
        raise ValueError("The documented OWL detector uses QuickGELU towers and text-width projections")
    model = OwlDetector(config, objectness=objectness)
    position_bias = model.box_bias
    model.to(device=device, dtype=dtype).eval()
    # HF constructs this nonpersistent position buffer in FP32 during loading.
    model.box_bias = position_bias.to(device=device)
    return model


def build_from_config(config, device, dtype):
    return build_detector(config, device, dtype)


@torch.no_grad()
def load_detector(model, state_dict, config, *, prefix):
    remaining = dict(state_dict)
    tower = {name.removeprefix(prefix + ".").replace(".pre_layernorm.", ".pre_layrnorm."): remaining.pop(name)
             for name in list(remaining) if name.startswith(prefix + ".")}
    load_clip(model.backbone, tower, config)
    mapped = {}
    for name in model.state_dict():
        if name.startswith("backbone."):
            continue
        source = name.replace("class_dense.", "class_head.dense0.")
        if name.startswith(("logit_shift.", "logit_scale.")):
            source = "class_head." + name
        mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped OWL weights: {sorted(remaining)}")
    mapped.update({"backbone." + name: tensor for name, tensor in model.backbone.state_dict().items()})
    model.load_state_dict(mapped, strict=True)


def load_state_dict_into(model, state_dict, config):
    load_detector(model, state_dict, config, prefix="owlvit")


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
