"""SAM2 point-initialized video propagation with explicit persistent memories."""

import torch
from torch import nn

from fastkernels.hf_coverage.models import sam2
from fastkernels.hf_coverage.models.sam import SamAttentionCore
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.sam_sine_dtype import SamSineDtype
from fastkernels.hf_coverage.patches.sam_position_dtype import SamPositionDtype
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.sam3_memory_encoder import CXBlock, SimpleFuser, SimpleMaskDownSampler, Sam3MemoryEncoder
from fastkernels.tasks.baseline.L3.sam3_mask_decoder import MLP
from fastkernels.tasks.baseline.L3.sam3_memory_attention import Sam3MemoryAttentionLayer
from fastkernels.tasks.baseline.L4.sam3_tracker import get_1d_sine_pe


class MemoryAttention(nn.Module):
    """Existing interleaved rotary operation and eager attention composition."""

    def __init__(self, c, kv_width=None):
        super().__init__()
        width = c.memory_attention_hidden_size
        inner = width // c.memory_attention_downsample_rate
        self.heads = c.memory_attention_num_attention_heads
        self.head_dim = inner // self.heads
        self.q_proj = Linear(width, inner)
        self.k_proj = Linear(kv_width or width, inner)
        self.v_proj = Linear(kv_width or width, inner)
        self.out_proj = Linear(inner, width)
        self.attend = SamAttentionCore()
        self.register_buffer("rotary_table", self.position_table(c, "cpu"), persistent=False)

    def position_table(self, c, device):
        x, y = c.memory_attention_rope_feat_sizes
        frequency = 1.0 / (c.memory_attention_rope_theta ** (
            torch.arange(0, self.head_dim, 4, device="cpu").float() / self.head_dim
        ))
        indices = torch.arange(x * y, device="cpu")
        angles = torch.cat(((indices % x)[:, None] * frequency,
                            (indices // x)[:, None] * frequency), dim=-1)
        return torch.cat((angles.cos(), angles.sin()), dim=-1).to(device)

    def rotate(self, values, table=None):
        table = self.rotary_table if table is None else table
        batch, length, _ = values.shape
        positions = torch.arange(length, device=values.device).remainder(table.shape[0]).repeat(batch)
        flat = values.float().reshape(batch*length, -1)
        rotated, _ = RotaryEmbedding.forward_native_interleaved(positions, flat, flat, self.head_dim, table)
        return rotated.reshape_as(values).to(values.dtype)

    def forward(self, q, k, v, num_k_exclude_rope=0):
        query, key, value = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        query = self.rotate(query)
        length = key.shape[1] - num_k_exclude_rope
        key = torch.cat((self.rotate(key[:, :length]), key[:, length:]), dim=1)
        batch = query.shape[0]
        query = query.reshape(batch, -1, self.heads, self.head_dim)
        key = key.reshape(batch, -1, self.heads, self.head_dim)
        value = value.reshape(batch, -1, self.heads, self.head_dim)
        return self.out_proj(self.attend(query, key, value).reshape(batch, query.shape[1], -1))


class Sam2VideoModel(sam2.Sam2Model):
    def __init__(self, c, vision_encoder=sam2.VisionEncoder):
        super().__init__(c, vision_encoder=vision_encoder)
        self.memory_attention = nn.Module()
        self.memory_attention.layers = nn.ModuleList([
            Sam3MemoryAttentionLayer(activation=c.memory_attention_feed_forward_hidden_act,
                                     d_model=c.memory_attention_hidden_size,
                                     dim_feedforward=c.memory_attention_feed_forward_hidden_size,
                                     dropout=c.memory_attention_dropout,
                                     num_heads=c.memory_attention_num_attention_heads,
                                     pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True,
                                     pos_enc_at_cross_attn_queries=False,
                                     self_attention=MemoryAttention(c),
                                     cross_attention=MemoryAttention(c, c.memory_encoder_output_channels))
            for _ in range(c.memory_attention_num_layers)])
        self.memory_attention.norm = LayerNorm(c.memory_attention_hidden_size, eps=1e-5, promote_fp32=False)
        downsampler = SimpleMaskDownSampler(c.mask_downsampler_embed_dim,
                                            kernel_size=c.mask_downsampler_kernel_size,
                                            stride=c.mask_downsampler_stride,
                                            padding=c.mask_downsampler_padding,
                                            total_stride=c.mask_downsampler_total_stride)
        for index in range(1, len(downsampler.encoder)-1, 3):
            downsampler.encoder[index] = LayerNorm2d(downsampler.encoder[index].weight.numel())
        block = CXBlock(c.memory_fuser_embed_dim, kernel_size=c.memory_fuser_kernel_size,
                        padding=c.memory_fuser_padding, layer_scale_init_value=c.memory_fuser_layer_scale_init_value)
        fuser = SimpleFuser(block, c.memory_fuser_num_layers)
        for layer in fuser.layers:
            layer.norm = LayerNorm2d(c.memory_fuser_embed_dim)
        self.memory_encoder = Sam3MemoryEncoder(c.memory_encoder_output_channels, downsampler, fuser,
                                                SamSineDtype(c.memory_encoder_output_channels),
                                                in_dim=c.memory_encoder_hidden_size)
        width, memory_width = c.vision_config.fpn_hidden_size, c.memory_encoder_output_channels
        self.no_memory_positional_encoding = nn.Parameter(torch.empty(1, 1, width))
        self.memory_temporal_positional_encoding = nn.Parameter(torch.empty(c.num_maskmem, 1, 1, memory_width))
        self.no_object_pointer = nn.Parameter(torch.empty(1, width))
        self.occlusion_spatial_embedding_parameter = nn.Parameter(torch.empty(1, memory_width))
        self.mask_downsample = Conv2d(1, 1, 4, stride=4)
        self.object_pointer_proj = MLP(width, width, width, 3)
        self.temporal_positional_encoding_projection_layer = Linear(width, memory_width)
        self.resize, self.sigmoid, self.top1 = Interpolate(), Sigmoid(), CodecTop1()

    def positive(self, values):
        return self.top1(torch.stack((torch.zeros_like(values), values), dim=-1)).bool()

    def memory_state(self, memory, appearing, high):
        spatial = memory["vision_features"]
        missing = torch.zeros_like(spatial) + self.occlusion_spatial_embedding_parameter[:, :, None, None]
        spatial = spatial + missing.masked_fill(appearing[:, :, None, None], 0)
        return (spatial.to(torch.bfloat16).flatten(2).permute(2, 0, 1),
                memory["vision_pos_enc"][0].to(high.dtype).flatten(2).permute(2, 0, 1))

    def condition(self, features, positions, history, index, total_frames):
        c = self.config
        if index == 0:
            return features[-1] + self.no_memory_embedding.reshape(1, -1, 1, 1)
        memories, memory_positions = [], []
        selected = [(0, history[0])]
        selected.extend((offset, history[index-offset]) for offset in range(c.num_maskmem-1, 0, -1)
                        if 0 < index-offset < len(history))
        for offset, previous in selected:
            memories.append(previous["maskmem_features"])
            memory_positions.append(previous["maskmem_pos_enc"] + self.memory_temporal_positional_encoding[offset-1])
        max_pointers = min(total_frames, c.max_object_pointers_in_encoder)
        pointer_frames = [0] + [index-offset for offset in range(1, max_pointers) if index-offset > 0]
        pointers = torch.stack([history[t]["object_pointer"] for t in pointer_frames])
        temporal = torch.tensor([index-t for t in pointer_frames], device=pointers.device, dtype=torch.float32)/(max_pointers-1)
        temporal = get_1d_sine_pe(temporal, c.vision_config.fpn_hidden_size).to(pointers.dtype)
        temporal = self.temporal_positional_encoding_projection_layer(temporal)[:, None]
        splits = c.vision_config.fpn_hidden_size//c.memory_encoder_output_channels
        pointers = pointers.reshape(-1, 1, splits, c.memory_encoder_output_channels).permute(0, 2, 1, 3).flatten(0, 1)
        temporal = temporal.repeat_interleave(splits, dim=0)
        memories.append(pointers)
        memory_positions.append(temporal)
        memory = torch.cat(memories).to(features[-1].dtype).transpose(0, 1)
        memory_position = torch.cat(memory_positions).transpose(0, 1)
        query_position = positions[-1].flatten(2).transpose(1, 2)
        query = features[-1].flatten(2).transpose(1, 2) + query_position*0.1
        for layer in self.memory_attention.layers:
            query = layer(query, memory, pos=memory_position, query_pos=query_position,
                          num_k_exclude_rope=pointers.shape[0])
        return self.memory_attention.norm(query).transpose(1, 2).reshape_as(features[-1])

    def frame(self, pixels, points, labels, history, index, total_frames):
        c = self.config
        features, positions = self.vision_encoder(pixels)
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
        mask_for_memory = self.positive(high).to(high.dtype) if index == 0 else self.sigmoid(high)
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
    model = Sam2VideoModel(config)
    # Native accepts FP32 point metadata and casts after coordinate normalization.
    model.prompt_encoder.pe_layer = SamPositionDtype(config.prompt_encoder_config.hidden_size // 2)
    model = model.to(device=device, dtype=dtype).eval()
    # Match the ordinary native SDPA path with an existing attention operation.
    for module in model.modules():
        if hasattr(module, "attend") and isinstance(module.attend, SamAttentionCore):
            module.attend = DenseAttention(backend="sdpa")
    # HF's pretrained loader initializes these nonpersistent buffers in FP32,
    # independently of BF16 parameters. Reconstruct metadata after dtype casting.
    for module in model.modules():
        if isinstance(module, MemoryAttention):
            module.rotary_table = module.position_table(config, device)
    return model


def load_state_dict_into(model, state_dict, config, image_module=sam2):
    image_prefixes = ("vision_encoder.", "prompt_encoder.", "shared_image_embedding.", "mask_decoder.")
    image_state = {k: v for k, v in state_dict.items() if k.startswith(image_prefixes) or k == "no_memory_embedding"}
    image_model = image_module.build_from_config(config, next(model.parameters()).device, next(model.parameters()).dtype)
    image_module.load_state_dict_into(image_model, image_state, config)
    mapped = image_model.state_dict()
    used = set(image_state)
    for name in model.state_dict():
        if name in mapped:
            continue
        source = name
        if source.startswith("memory_attention."):
            source = source.replace(".out_proj.", ".o_proj.").replace(".norm.", ".layer_norm.")
            for i in range(1, 4):
                source = source.replace(f".norm{i}.", f".layer_norm{i}.")
        elif source.startswith("memory_encoder."):
            source = source.replace(".pix_feat_proj.", ".feature_projection.").replace(".out_proj.", ".projection.")
            source = source.replace(".fuser.", ".memory_fuser.")
            for a, b in (("dwconv", "depthwise_conv"), ("norm", "layer_norm"), ("pwconv1", "pointwise_conv1"), ("pwconv2", "pointwise_conv2")):
                source = source.replace(f".{a}.", f".{b}.")
            if source.endswith(".gamma"):
                source = source[:-5]+"scale"
            for i in range(4):
                source = source.replace(f"mask_downsampler.encoder.{i*3}.", f"mask_downsampler.layers.{i}.conv.")
                source = source.replace(f"mask_downsampler.encoder.{i*3+1}.", f"mask_downsampler.layers.{i}.layer_norm.")
            source = source.replace("mask_downsampler.encoder.12.", "mask_downsampler.final_conv.")
        elif source.startswith("object_pointer_proj."):
            for i, native in ((0, "proj_in"), (1, "layers.0"), (2, "proj_out")):
                source = source.replace(f".layers.{i}.", f".{native}.")
        mapped[name] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f"Unmapped SAM2 video state: {sorted(set(state_dict)-used)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    history = []

    def prepare():
        history.clear()

    def run():
        outputs = {}
        video = inputs["video"]
        for index in range(video.shape[0]):
            result = model.frame(video[index:index+1], inputs["input_points"], inputs["input_labels"],
                                 history, index, video.shape[0])
            outputs.update({f"frame{index}.{key}": value for key, value in result.items()})
        return outputs

    return {"forward": Workload(prepare=prepare, run=run)}
