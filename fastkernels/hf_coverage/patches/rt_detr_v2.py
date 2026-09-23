"""Expose default HF proposal outputs omitted by the existing RT-DETRv2 L3."""

import torch

from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L3.rtdetrv2_model import RTDetrV2Model
from .detector_topk import DetectorTopK


class RTDetrV2ProposalOutputs(RTDetrV2Model):
    """Retain the parent's stages and selection; add proposal sigmoid and gather.

    The parent has no interface exposing proposals before decoding, so its default
    orchestration is repeated here without recomputing any model stage.
    """

    def __init__(self, config):
        super().__init__(config)
        self.proposal_sigmoid = Sigmoid()
        self.class_max = SegmentCSR()
        self.proposal_topk = DetectorTopK()

    def forward(self, pixel_values, pixel_mask=None):
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        features = self.backbone(pixel_values, pixel_mask)
        projected = [
            self.encoder_input_proj[level](source)
            for level, (source, mask) in enumerate(features)
        ]
        encoded = self.encoder(projected, return_dict=True)
        sources = [
            self.decoder_input_proj[level](source)
            for level, source in enumerate(encoded.last_hidden_state)
        ]
        spatial_shapes_list = [tuple(source.shape[-2:]) for source in sources]
        spatial_shapes = torch.tensor(spatial_shapes_list, device=device, dtype=torch.long)
        source_flatten = torch.cat([
            source.flatten(2).transpose(1, 2) for source in sources
        ], dim=1)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1],
        ))

        anchors, valid_mask = self.generate_anchors(
            spatial_shapes_list, device=device, dtype=source_flatten.dtype,
        )
        memory = source_flatten.masked_fill(~valid_mask, 0)
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors
        scores = enc_outputs_class.reshape(-1)
        offsets = torch.arange(0, scores.numel() + 1, self.config.num_labels,
                               device=device, dtype=torch.long)
        scores = self.class_max(scores, offsets, reduce="max").view(batch_size, -1)
        _, topk_ind = self.proposal_topk(scores, self.config.num_queries)
        reference_points = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, 4),
        )
        enc_topk_bboxes = self.proposal_sigmoid(reference_points)
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, self.config.num_labels),
        )
        target = output_memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, self.config.d_model),
        ).detach()
        decoded = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=None,
            reference_points=reference_points.detach(),
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            return_dict=True,
        )
        return {
            "logits": decoded.intermediate_logits[:, -1],
            "pred_boxes": decoded.intermediate_reference_points[:, -1],
            "last_hidden_state": decoded.last_hidden_state,
            "intermediate_hidden_states": decoded.intermediate_hidden_states,
            "intermediate_logits": decoded.intermediate_logits,
            "intermediate_reference_points": decoded.intermediate_reference_points,
            **{f"encoder_last_hidden_state.{level}": source
               for level, source in enumerate(encoded.last_hidden_state)},
            "init_reference_points": reference_points.detach(),
            "enc_topk_bboxes": enc_topk_bboxes,
            "enc_topk_logits": enc_topk_logits,
            "enc_outputs_class": enc_outputs_class,
            "enc_outputs_coord_logits": enc_outputs_coord_logits,
        }
