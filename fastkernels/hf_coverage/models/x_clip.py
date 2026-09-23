"""X-CLIP's frame messages, temporal integration and video-conditioned text."""

from copy import copy
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU
from fastkernels.tasks.baseline.L2.siglip_attention import SigLIPAttention
from fastkernels.tasks.baseline.L4.clip_text_model import CLIPTextModel
from .clip import ClipEncoderLayer, ClipVisionEmbeddings, configure_encoder
from ..patches.product_gate import ProductGate
from ..runner import Workload


def norm(width, eps):
    return LayerNorm(width, eps=eps, promote_fp32=False)


class FrameLayer(ClipEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.frames = config.num_frames
        self.message_fc = Linear(config.hidden_size, config.hidden_size)
        self.message_ln = norm(config.hidden_size, config.layer_norm_eps)
        self.message_attn = SigLIPAttention(config.hidden_size, config.num_attention_heads)
        self.message_attn.attn = DenseAttention(backend='sdpa')

    def forward(self, hidden):
        frame_batch, length, width = hidden.shape
        message = self.message_fc(hidden[:, 0]).view(-1, self.frames, width)
        message = message + self.message_attn(self.message_ln(message))
        hidden = torch.cat((hidden, message.reshape(frame_batch, 1, width)), dim=1)
        hidden = (hidden + self._self_attention(self.ln_1(hidden)))[:, :length]
        return hidden + self.mlp_fc2(self.mlp_act(self.mlp_fc1(self.ln_2(hidden))))


class PromptLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.projection_dim
        self.heads = config.prompt_num_attention_heads
        self.width = width // self.heads
        self.norm1 = norm(width, config.text_config.layer_norm_eps)
        self.norm3 = norm(width, config.text_config.layer_norm_eps)
        self.q_proj, self.k_proj, self.v_proj = [Linear(width, width, bias=False) for _ in range(3)]
        self.proj = Linear(width, width)
        self.attn = DenseAttention(backend='sdpa')
        self.mlp = nn.Sequential(Linear(width, 4 * width), QuickGELU(), nn.Identity(), Linear(4 * width, width))

    def forward(self, text, visual):
        batch, length = text.shape[:2]
        q = self.q_proj(self.norm1(text)).view(batch, length, self.heads, self.width)
        k = self.k_proj(visual).view(batch, -1, self.heads, self.width)
        v = self.v_proj(visual).view(batch, -1, self.heads, self.width)
        text = text + self.proj(self.attn(q, k, v).reshape_as(text))
        return text + self.mlp(self.norm3(text))


class XClipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        text, vision = config.text_config, config.vision_config
        self.text_model = CLIPTextModel(text)
        configure_encoder(self.text_model.text_model.encoder, text)
        self.vision_embeddings = ClipVisionEmbeddings(vision)
        self.vision_pre = norm(vision.hidden_size, vision.layer_norm_eps)
        self.vision_post = norm(vision.hidden_size, vision.layer_norm_eps)
        self.vision_layers = nn.ModuleList([FrameLayer(vision) for _ in range(vision.num_hidden_layers)])
        self.visual_projection = Linear(vision.hidden_size, config.projection_dim, bias=False)
        self.text_projection = Linear(text.hidden_size, config.projection_dim, bias=False)
        mit = copy(vision)
        mit.hidden_size, mit.intermediate_size = vision.mit_hidden_size, vision.mit_intermediate_size
        mit.num_attention_heads = vision.mit_num_attention_heads
        self.mit_positions = nn.Parameter(torch.empty(1, vision.num_frames, mit.hidden_size))
        self.mit_layers = nn.ModuleList([ClipEncoderLayer(mit) for _ in range(vision.mit_num_hidden_layers)])
        self.frame_mean = AvgPool2d((vision.num_frames, 1))
        self.prompts_visual_layernorm = norm(vision.hidden_size, vision.layer_norm_eps)
        self.prompts_visual_projection = nn.Parameter(torch.empty(vision.hidden_size, config.projection_dim))
        self.prompt_norm = norm(config.projection_dim, vision.layer_norm_eps)
        self.prompt_layers = nn.ModuleList([PromptLayer(config) for _ in range(config.prompt_layers)])
        self.alpha = nn.Parameter(torch.empty(config.projection_dim))
        self.product = ProductGate()
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.register_buffer('scale', torch.empty(()), persistent=False)
        self.normalize, self.matmul = L2Norm(dim=-1, eps=0), BMM()
        for module in self.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False

    def forward(self, input_ids, pixel_values):
        batch, frames = pixel_values.shape[:2]
        vision = self.vision_pre(self.vision_embeddings(pixel_values.flatten(0, 1)))
        for layer in self.vision_layers:
            vision = layer(vision)
        vision_pool = self.vision_post(vision[:, 0])
        classes = self.visual_projection(vision_pool).view(batch, frames, -1)
        mit = classes + self.mit_positions
        for layer in self.mit_layers:
            mit = layer(mit)
        mit = mit + classes
        video_pool = self.frame_mean(mit.transpose(1, 2).unsqueeze(-1)).flatten(1)
        visual = self.matmul(self.prompts_visual_layernorm(vision[:, 1:]), self.prompts_visual_projection)
        visual = visual.view(batch, frames, *visual.shape[1:]).permute(0, 2, 3, 1)
        visual = self.frame_mean(visual.reshape(-1, visual.shape[2], frames, 1)).view(batch, vision.shape[1] - 1, -1)
        text_output = self.text_model(input_ids)
        text = self.text_projection(text_output.pooler_output)[None].expand(batch, -1, -1)
        prompted, visual = text, self.prompt_norm(visual)
        for layer in self.prompt_layers:
            prompted = layer(prompted, visual)
        text = self.normalize(text + self.product(torch.cat((self.alpha.expand_as(prompted), prompted), dim=-1)))
        video = self.normalize(video_pool)
        logits = self.matmul(video[:, None], (text * self.scale).transpose(-1, -2)).squeeze(1)
        return {'logits_per_video': logits, 'logits_per_text': logits.t(), 'text_embeds': text,
                'video_embeds': video, 'text_model_output.last_hidden_state': text_output.last_hidden_state,
                'text_model_output.pooler_output': text_output.pooler_output,
                'vision_model_output.last_hidden_state': vision, 'vision_model_output.pooler_output': vision_pool,
                'mit_output.last_hidden_state': mit, 'mit_output.pooler_output': video_pool}


def build_from_config(config, device, dtype):
    if (config.text_config.hidden_act != 'quick_gelu' or config.vision_config.hidden_act != 'quick_gelu'
            or config.prompt_hidden_act != 'quick_gelu' or config.text_config.eos_token_id != 2
            or config.vision_config.mit_hidden_size != config.projection_dim):
        raise ValueError('XCLIP case preserves checkpoint QuickGELU, legacy EOS pooling and temporal projection size')
    return XClipModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name in model.state_dict():
        source = name.replace('text_model.text_model.', 'text_model.').replace('.emb.weight', '.weight')
        source = source.replace('vision_embeddings.', 'vision_model.embeddings.')
        source = source.replace('patch_embedding.proj.', 'patch_embedding.')
        source = source.replace('vision_pre.', 'vision_model.pre_layernorm.').replace('vision_post.', 'vision_model.post_layernorm.')
        source = source.replace('vision_layers.', 'vision_model.encoder.layers.').replace('mit_layers.', 'mit.encoder.layers.')
        source = source.replace('mit_positions', 'mit.position_embedding').replace('prompt_norm.', 'prompts_generator.layernorm.')
        source = source.replace('prompt_layers.', 'prompts_generator.decoder.')
        if source == 'alpha':
            source = 'prompts_generator.alpha'
        source = source.replace('.ln_1.', '.layer_norm1.').replace('.ln_2.', '.layer_norm2.')
        source = source.replace('.mlp_fc1.', '.mlp.fc1.').replace('.mlp_fc2.', '.mlp.fc2.')
        module, field = source.rsplit('.', 1) if '.' in source else ('', source)
        if module.endswith(('.q_proj', '.k_proj', '.v_proj', '.out_proj')) and '.message_attn.' not in source:
            prefix, part = module.rsplit('.', 1)
            attention = 'cross_attn' if prefix.startswith('prompts_generator.decoder.') else 'self_attn'
            source = f'{prefix}.{attention}.{part}.{field}'
        elif module.endswith('.proj') and module.startswith('prompts_generator.decoder.'):
            source = source.replace('.proj.', '.cross_attn.proj.')
        mapped[name] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped XCLIP weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
    model.scale.copy_(model.logit_scale.exp())


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
