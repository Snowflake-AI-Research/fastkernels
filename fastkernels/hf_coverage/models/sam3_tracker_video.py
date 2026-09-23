"""Constructor-default SAM3 point-initialized video tracking.

SAM2 supplies the identical forward-only one-object memory selection, pointer
history, memory attention and encoder compositions. SAM3 changes the visual
encoder, selected pyramid levels, point precision and mask-to-memory resize.
The workload is one positive point on frame zero followed by propagation;
interactive corrections, mask inputs and reverse propagation are not selected.
"""

import torch

from fastkernels.hf_coverage.models import sam2, sam2_video, sam3_tracker
from fastkernels.hf_coverage.models.sam import SamAttentionCore
from fastkernels.hf_coverage.models.sam3_lite_text import VisionAttention, VisionEncoder
from fastkernels.hf_coverage.patches.sam_position_dtype import SamPositionDtype
from fastkernels.hf_coverage.patches.siglip2_interpolate import PositionTableResize
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention


class Sam3TrackerVideoModel(sam2_video.Sam2VideoModel):
    def __init__(self, config):
        super().__init__(config, vision_encoder=VisionEncoder)
        self.prompt_encoder.pe_layer = SamPositionDtype(config.prompt_encoder_config.hidden_size // 2)
        self.memory_resize = PositionTableResize()

    def frame(self, pixels, points, labels, history, index, total_frames):
        c = self.config
        features, positions = self.vision_encoder(pixels)
        # Native get_image_features discards the fourth FPN level.
        features, positions = features[:-1], positions[:-1]
        features[0] = self.mask_decoder.conv_s0(features[0])
        features[1] = self.mask_decoder.conv_s1(features[1])
        conditioned = self.condition(features, positions, history, index, total_frames)
        if index:
            points = pixels.new_zeros(1, 1, 2)
            labels = torch.full((1, 1), -1, dtype=torch.long, device=pixels.device)
        else:
            points, labels = points.reshape(1, -1, 2), labels.reshape(1, -1)
        sparse, dense = self.prompt_encoder(points, labels)
        pe = sam2.dense_position(self.shared_image_embedding, self.prompt_encoder.image_embedding_size)
        # Native prompt batching materializes these maps in NCHW order. Preserve
        # that layout: Linear selects different BF16 accumulation paths otherwise.
        masks, scores, tokens, object_scores = self.mask_decoder(conditioned.contiguous(), pe.contiguous(), sparse, dense,
                                                                 multimask_output=True, repeat_image=False,
                                                                 high_res_features=features[:-1])
        appearing = self.positive(object_scores)
        masks = masks.masked_fill(~appearing[:, :, None, None], -1024)
        high = self.resize(masks.float(), size=(c.image_size, c.image_size), mode="bilinear", align_corners=False).to(masks.dtype)
        selected = self.top1(scores)
        batch = torch.arange(masks.shape[0], device=pixels.device)
        low, high = masks[batch, selected][:, None], high[batch, selected][:, None]
        pointer = self.object_pointer_proj(tokens[batch, selected])
        pointer = torch.where(appearing, pointer, self.no_object_pointer)
        # Patch14 visual grid and stride16 memory masks have distinct sizes.
        # Native resizes logits before the data-dependent threshold/sigmoid.
        memory_size = tuple(side * 16 for side in self.prompt_encoder.image_embedding_size)
        memory_masks = self.memory_resize(high.float(), memory_size).to(high.dtype)
        mask_for_memory = self.positive(memory_masks).to(high.dtype) if index == 0 else self.sigmoid(memory_masks)
        mask_for_memory = mask_for_memory*c.sigmoid_scale_for_mem_enc + c.sigmoid_bias_for_mem_enc
        # Match the native sequence-to-image view, including singleton batch
        # stride. cuDNN's layout choice otherwise changes BF16 depthwise results.
        memory_image = features[-1].flatten(2).permute(2, 0, 1)
        memory_image = memory_image.permute(1, 2, 0).view(*features[-1].shape)
        memory = self.memory_encoder(memory_image, mask_for_memory, skip_mask_sigmoid=True)
        memory_features, memory_positions = self.memory_state(memory, appearing, high)
        state = {"pred_masks": low, "high_res_masks": high,
                 "object_pointer": pointer[:, None], "object_score_logits": object_scores[:, None],
                 "maskmem_features": memory_features, "maskmem_pos_enc": memory_positions}
        history.append(state)
        return {"pred_masks": low, "object_score_logits": object_scores,
                **{f"state.{name}": value for name, value in state.items()}}


def build_from_config(config, device, dtype):
    model = Sam3TrackerVideoModel(config).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        if hasattr(module, "attend") and isinstance(module.attend, SamAttentionCore):
            module.attend = DenseAttention(backend="sdpa")
        if isinstance(module, VisionAttention):
            module.rotary_table = module.position_table(device)
        elif isinstance(module, sam2_video.MemoryAttention):
            module.rotary_table = module.position_table(config, device)
    return model


def load_state_dict_into(model, state_dict, config):
    sam2_video.load_state_dict_into(model, state_dict, config, image_module=sam3_tracker)


make_workloads = sam2_video.make_workloads
