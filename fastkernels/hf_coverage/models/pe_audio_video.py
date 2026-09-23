"""PE full audio/video/text inference with shared encoders and all default outputs."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from ..runner import config_values
from . import pe_audio, pe_video


class JointEmbedder(nn.Module):
    def __init__(self, config, audio, video):
        super().__init__()
        self.audio_encoder, self.video_encoder = audio, video
        aw, vw = config.audio_config.hidden_size, config.video_config.hidden_size
        self.video_proj = Conv1dNative(vw, aw, 1)
        self.video_norm = LayerNorm(aw, eps=1e-5, promote_fp32=False)
        self.concat_modality_proj = Linear(aw + vw, config.hidden_size)
        self.data_proj = Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_values, pixel_values_videos, padding_mask, padding_mask_videos):
        audio = self.audio_encoder(input_values, padding_mask)
        video = self.video_encoder(pixel_values_videos, padding_mask_videos)
        a, v, mask = audio['last_hidden_state'], video['last_hidden_state'], audio['output_mask']
        v = self.video_proj(v.transpose(1, 2)).transpose(1, 2)
        if v.shape[1] != a.shape[1]:
            al = ([a.shape[1]] * a.shape[0]) if mask is None else mask.sum(-1).tolist()
            vl = ([v.shape[1]] * v.shape[0]) if padding_mask_videos is None else padding_mask_videos.sum(-1).tolist()
            if all(length == v.shape[1] for length in al) or all(length == a.shape[1] for length in vl):
                # Nearest interpolation is an index gather, with indices determined only by lengths.
                indices = torch.arange(a.shape[1], device=v.device) * v.shape[1] // a.shape[1]
                v = v[:, indices]
            else:
                aligned = a.new_zeros(a.shape[0], a.shape[1], v.shape[-1])
                for i, (na, nv) in enumerate(zip(al, vl)):
                    if na > 0 and nv > 0:
                        indices = torch.arange(na, device=v.device) * nv // na
                        aligned[i, :na] = v[i, indices]
                v = aligned
        x = self.data_proj(self.concat_modality_proj(torch.cat((a, self.video_norm(v)), -1)))
        return x, mask, audio, video


class JointEncoder(pe_audio.TemporalEncoder):
    def forward(self, input_values, pixel_values_videos, padding_mask, padding_mask_videos):
        x, mask, audio, video = self.embedder(input_values, pixel_values_videos, padding_mask, padding_mask_videos)
        x, mask = self.patch_embedder(x, mask)
        for layer in self.layers:
            x = layer(x, self.rotary.cos_sin_cache, mask)
        x = self.output(self.norm(x))
        return {'last_hidden_state': x[:, 1:], 'pooler_output': x[:, 0],
                'audio_model_output': audio, 'video_model_output': video}


class PeAudioVideo(nn.Module):
    def __init__(self, config):
        super().__init__()
        joint = config.audio_video_config
        ac = config_values({'text_config': config.text_config, 'audio_config': joint.audio_config})
        vc = config_values({'text_config': config.text_config, 'video_config': joint.video_config})
        self.text_model = pe_audio.TextEncoder(config.text_config)
        self.audio_model, self.video_model = pe_audio.PeAudio(ac), pe_video.PeVideo(vc)
        self.audio_model.text_model = self.text_model
        self.video_model.text_model = self.text_model
        embedder = JointEmbedder(joint, self.audio_model.audio_encoder, self.video_model.video_encoder)
        self.audio_video_encoder = JointEncoder(joint, embedder)
        tw, aw, vw = config.text_config.hidden_size, joint.audio_config.hidden_size, joint.video_config.hidden_size
        self.audio_video_head = pe_audio.ContrastiveHead(joint.hidden_size, tw)
        self.text_audio_video_head = pe_audio.ContrastiveHead(tw, tw)
        self.audio_plus_text_head = pe_audio.ContrastiveHead(tw + aw, tw)
        self.video_plus_text_head = pe_audio.ContrastiveHead(tw + vw, tw)
        for name in ('audio_video', 'text_audio_video', 'audio_plus_text', 'video_plus_text'):
            setattr(self, name + '_logit_scale', nn.Parameter(torch.empty(1)))
            setattr(self, name + '_logit_bias', nn.Parameter(torch.empty(1)))
        self.matmul = Matmul()

    def forward(self, input_ids, input_values, pixel_values_videos, attention_mask=None,
                padding_mask=None, padding_mask_videos=None):
        if input_ids is None or input_values is None or pixel_values_videos is None:
            raise ValueError('This workload composes the documented full three-modality PE forward')
        joint = self.audio_video_encoder(input_values, pixel_values_videos, padding_mask, padding_mask_videos)
        audio, video = joint['audio_model_output'], joint['video_model_output']
        text = self.text_model(input_ids, attention_mask)
        t = text['hidden_states'][-1][:, 0]
        a = self.audio_model.audio_head(audio['pooler_output'])
        v = self.video_model.video_head(video['pooler_output'])
        av = self.audio_video_head(joint['pooler_output'])
        ta, tv, tav = self.audio_model.text_audio_head(t), self.video_model.text_video_head(t), self.text_audio_video_head(t)
        at = self.audio_plus_text_head(torch.cat((audio['pooler_output'], t), -1))
        vt = self.video_plus_text_head(torch.cat((video['pooler_output'], t), -1))
        result = dict(audio_embeds=a, video_embeds=v, audio_video_embeds=av, text_audio_embeds=ta,
                      text_video_embeds=tv, text_audio_video_embeds=tav, audio_plus_text_embeds=at,
                      video_plus_text_embeds=vt, text_outputs=text, audio_outputs=audio,
                      video_outputs=video, audio_video_outputs=joint)
        for key, left, right, owner, name in (
            ('audio_text', a, ta, self.audio_model, 'text_audio'),
            ('video_text', v, tv, self.video_model, 'text_video'),
            ('audio_video', a, v, self, 'audio_video'),
            ('audio_video_text', av, tav, self, 'text_audio_video'),
            ('audio_plus_text_video', at, v, self, 'audio_plus_text'),
            ('video_plus_text_audio', vt, a, self, 'video_plus_text'),
        ):
            result['logits_' + key] = self.matmul(left, right) * getattr(owner, name + '_logit_scale') + getattr(owner, name + '_logit_bias')
        return result


def build_from_config(config, device, dtype):
    return PeAudioVideo(config).to(device=device, dtype=dtype).eval()


load_state_dict_into = pe_audio.load_state_dict_into
make_workloads = pe_audio.make_workloads
