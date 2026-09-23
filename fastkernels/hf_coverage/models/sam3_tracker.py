"""SAM3Tracker's constructor-default point-prompt image segmentation."""

import torch

from fastkernels.hf_coverage.patches.sam_position_dtype import SamPositionDtype

from .sam2 import Sam2Model, dense_position, load_state_dict_into, make_workloads
from .sam3_lite_text import VisionAttention, VisionEncoder


class Sam3TrackerModel(Sam2Model):
    def __init__(self, config):
        super().__init__(config, vision_encoder=VisionEncoder)
        # Native HF normalizes supplied FP32 points before casting for the
        # learned coordinate projection; keep that rounding boundary.
        self.prompt_encoder.pe_layer = SamPositionDtype(config.prompt_encoder_config.hidden_size // 2)

    def forward(self, pixel_values, input_points, input_labels=None):
        features, _ = self.vision_encoder(pixel_values)
        features[0] = self.mask_decoder.conv_s0(features[0])
        features[1] = self.mask_decoder.conv_s1(features[1])
        # Native constructor defaults compute four levels, add no-memory to
        # level four, then retain only the three declared backbone feature sizes.
        features[-1] = features[-1] + self.no_memory_embedding.reshape(1, -1, 1, 1)
        features = features[:len(self.config.vision_config.backbone_feature_sizes)]
        batch, point_batch, count, _ = input_points.shape
        labels = input_labels
        if labels is None:
            labels = torch.ones(input_points.shape[:-1], device=input_points.device, dtype=torch.long)
        sparse, dense = self.prompt_encoder(input_points.reshape(-1, count, 2), labels.reshape(-1, count))
        positions = dense_position(self.shared_image_embedding, self.prompt_encoder.image_embedding_size)
        masks, scores, _, object_scores = self.mask_decoder(
            features[-1].repeat_interleave(point_batch, 0), positions, sparse, dense,
            multimask_output=True, repeat_image=False,
            high_res_features=[value.repeat_interleave(point_batch, 0) for value in features[:-1]],
        )
        outputs = {
            "pred_masks": masks.reshape(batch, point_batch, *masks.shape[1:]),
            "iou_scores": scores.reshape(batch, point_batch, -1),
            "object_score_logits": object_scores.reshape(batch, point_batch, -1),
        }
        outputs.update({f"image_embeddings.{i}": value for i, value in enumerate(features)})
        return outputs


def build_from_config(config, device, dtype):
    model = Sam3TrackerModel(config).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        if isinstance(module, VisionAttention):
            module.rotary_table = module.position_table(device)
    return model
