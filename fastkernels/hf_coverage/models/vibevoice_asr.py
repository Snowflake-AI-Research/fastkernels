"""VibeVoice ASR with streaming dual encoders, native VAE sampling and Qwen2."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from ..patches.codec_top1 import CodecTop1
from ..patches.vibevoice_gaussian import sample_latents
from ..runner import Workload
from .qwen2_5_omni import TextModel
from .vibevoice_acoustic_tokenizer import CausalConv, Stack


class StreamingConv(nn.Module):
    """Retain causal-convolution history between the parent's declared chunks."""

    def __init__(self, original):
        super().__init__()
        self.conv, self.left_pad = original.conv, original.left_pad
        self.history = None

    def forward(self, hidden):
        if self.history is None:
            self.history = hidden.new_zeros((*hidden.shape[:-1], self.left_pad))
        padded = torch.cat((self.history, hidden), -1)
        self.history = padded[..., -self.left_pad:].clone() if self.left_pad else padded[..., :0]
        return self.conv(padded)


class Encoder(Stack):
    def __init__(self, config):
        super().__init__(config, decoder=False)
        for parent in list(self.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, CausalConv):
                    setattr(parent, name, StreamingConv(child))

    def reset(self):
        for module in self.modules():
            if isinstance(module, StreamingConv):
                module.history = None

    def forward(self, hidden):
        return super().forward(hidden).transpose(1, 2)


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        for kind in ("acoustic", "semantic"):
            source = getattr(config, kind + "_tokenizer_encoder_config").hidden_size
            target = config.text_config.hidden_size
            setattr(self, kind + "_linear_1", Linear(source, target))
            setattr(self, kind + "_norm", RMSNormNative(target, 1e-6))
            setattr(self, kind + "_linear_2", Linear(target, target))

    def forward(self, acoustic, semantic):
        branches = []
        for kind, value in (("acoustic", acoustic), ("semantic", semantic)):
            value = getattr(self, kind + "_linear_1")(value)
            value = getattr(self, kind + "_norm")(value)
            branches.append(getattr(self, kind + "_linear_2")(value))
        return branches[0] + branches[1]


class VibeVoiceASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.acoustic_tokenizer_encoder = Encoder(config.acoustic_tokenizer_encoder_config)
        self.semantic_tokenizer_encoder = Encoder(config.semantic_tokenizer_encoder_config)
        self.multi_modal_projector = Projector(config)
        values = config.text_config.to_dict()
        # Equal position axes specialize the existing native MRoPE callable to
        # ordinary one-dimensional Qwen2 RoPE without changing its arithmetic.
        dim = values["hidden_size"] // values["num_attention_heads"]
        values["rope_parameters"] = dict(values["rope_parameters"], mrope_section=[dim // 2, 0, 0])
        self.language_model = nn.Module()
        self.language_model.model = TextModel(SimpleNamespace(**values))
        self.language_model.lm_head = Linear(values["hidden_size"], values["vocab_size"], bias=False)
        self.argmax = CodecTop1()

    def audio_features(self, input_values, padding_mask):
        acoustic, semantic = self.acoustic_tokenizer_encoder, self.semantic_tokenizer_encoder
        acoustic.reset()
        semantic.reset()
        a, s = [], []
        for chunk in input_values.split(self.config.acoustic_tokenizer_chunk_size, -1):
            a.append(acoustic(chunk))
            s.append(semantic(chunk))
        a, s = torch.cat(a, 1), torch.cat(s, 1)
        a = sample_latents(a, self.config.acoustic_tokenizer_encoder_config.vae_std)
        combined = self.multi_modal_projector(a, s)
        hop = math.prod(self.config.acoustic_tokenizer_encoder_config.downsampling_ratios)
        lengths = (padding_mask.sum(-1) + hop - 1) // hop
        valid = torch.arange(combined.shape[1], device=combined.device)[None] < lengths[:, None]
        return combined[valid]

    def generate(self, inputs, steps, eos):
        ids = inputs["input_ids"]
        model = self.language_model.model
        model.reset()
        hidden = model.embed_tokens(ids)
        hidden[ids == self.config.audio_token_id] = self.audio_features(inputs["input_values"], inputs["padding_mask"])
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand(3, -1)
        outputs = {}
        for step in range(steps):
            logits = self.language_model.lm_head(model(hidden, positions)[:, -1]).float()
            outputs[f"logits.{step}"] = logits
            token = self.argmax(logits)[:, None]
            ids = torch.cat((ids, token), -1)
            if int(token[0, 0]) == eos or step + 1 == steps:
                break
            hidden = model.embed_tokens(token)
            positions = torch.full((3, 1), ids.shape[1] - 1, dtype=torch.long, device=ids.device)
        outputs["sequences"] = ids
        for index, layer in enumerate(model.layers):
            for name in ("key", "value"):
                outputs[f"past_key_values.{index}.{name}"] = getattr(layer.self_attn, name).transpose(1, 2)
        return outputs


def build_from_config(config, device, dtype):
    if config.text_config.tie_word_embeddings or config.text_config.use_sliding_window:
        raise ValueError("Preserve the native untied full-attention VibeVoice Qwen2 decoder")
    return VibeVoiceASR(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, weights, config):
    mapped, consumed = {}, set()
    for name, target in model.state_dict().items():
        source = name.replace(".emb.weight", ".weight")
        source = source.replace(".ffn.fc1.", ".ffn.linear1.").replace(".ffn.fc2.", ".ffn.linear2.")
        if weights[source].shape != target.shape:
            raise ValueError(f"VibeVoice ASR state shape mismatch: {source}")
        mapped[name] = weights[source]
        consumed.add(source)
    if consumed != set(weights):
        raise ValueError(f"VibeVoice ASR unmapped weights: {sorted(set(weights) - consumed)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, case):
    generation = case["generation_kwargs"]
    eos = case["reference"]["generation_config"]["eos_token_id"]
    return {"generate": Workload(run=lambda: model.generate(inputs, generation["max_new_tokens"], eos))}
