"""CosyVoice3 TTS model for text-to-speech synthesis.

Implements the full CosyVoice3 architecture with two stages:
  - Talker: Qwen2-based LM that generates speech tokens from text
  - Code2Wav: Flow matching (DiT) + HiFi-GAN vocoder for waveform synthesis

Matches vllm-omni's CosyVoice3Model interface.
Reference: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from ..L2.cosyvoice3_pre_lookahead import PreLookaheadLayer
from ..L2.cosyvoice3_cfm import CausalConditionalCFM
from ..L2.cosyvoice3_hifigan import CausalHiFTGenerator, CausalConvRNNF0Predictor
from ..L3.cosyvoice3_dit import DiT


@dataclass
class CosyVoice3Config:
    model_type: str = "cosyvoice3"
    sample_rate: int = 24000
    llm_input_size: int = 896
    llm_output_size: int = 896
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    spk_embed_dim: int = 192
    token_frame_rate: int = 25
    token_mel_ratio: int = 2
    vocab_size: int = 151923
    speech_token_size: int = 6561
    eos_token_id: int = 6562
    allowed_special: str = "all"
    skip_special_tokens: bool = True
    target_sr: int = 24000
    qwen_pretrain_path: str = "CosyVoice-BlankEN"
    campplus_onxx_path: str = "campplus.onnx"
    speech_tokenizer_path: str = "speech_tokenizer_v3.onnx"
    spk2info_path: str = "spk2info.pt"
    version: str = "cosyvoice3"
    dtype: torch.dtype = torch.bfloat16

    feat_extractor: dict = field(default_factory=lambda: {
        "n_fft": 1920,
        "num_mels": 80,
        "sampling_rate": 24000,
        "hop_size": 480,
        "win_size": 1920,
        "fmin": 0,
        "fmax": None,
        "center": False,
    })
    llm: dict = field(default_factory=lambda: {
        "llm_input_size": 896,
        "llm_output_size": 896,
        "speech_token_size": 6561,
        "eos_token_id": 6562,
        "length_normalized_loss": True,
        "lsm_weight": 0,
        "mix_ratio": [5, 15],
        "llm": {"pretrain_path": "CosyVoice-BlankEN"},
        "sampling": {"top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1},
        "spk_embed_dim": 192,
    })
    flow: dict = field(default_factory=lambda: {
        "input_size": 80,
        "output_size": 80,
        "spk_embed_dim": 192,
        "output_type": "mel",
        "vocab_size": 6561,
        "input_frame_rate": 25,
        "only_mask_loss": True,
        "token_mel_ratio": 2,
        "pre_lookahead_len": 3,
        "pre_lookahead_layer": {
            "in_channels": 80,
            "channels": 1024,
            "pre_lookahead_len": 3,
        },
        "decoder": {
            "in_channels": 240,
            "n_spks": 1,
            "spk_emb_dim": 80,
            "cfm_params": {
                "sigma_min": 1e-06,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
                "reg_loss_type": "l1",
            },
            "estimator": {
                "dim": 1024,
                "depth": 22,
                "heads": 16,
                "dim_head": 64,
                "ff_mult": 2,
                "mel_dim": 80,
                "mu_dim": 80,
                "spk_dim": 80,
                "out_channels": 80,
                "static_chunk_size": 50,
                "num_decoding_left_chunks": -1,
            },
        },
    })
    hift: dict = field(default_factory=lambda: {
        "in_channels": 80,
        "base_channels": 512,
        "nb_harmonics": 8,
        "sampling_rate": 24000,
        "nsf_alpha": 0.1,
        "nsf_sigma": 0.003,
        "nsf_voiced_threshold": 10,
        "upsample_rates": [8, 5, 3],
        "upsample_kernel_sizes": [16, 11, 7],
        "istft_params": {"n_fft": 16, "hop_len": 4},
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "source_resblock_kernel_sizes": [7, 7, 11],
        "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "lrelu_slope": 0.1,
        "audio_limit": 0.99,
        "conv_pre_look_right": 4,
        "f0_predictor": {
            "num_class": 1,
            "in_channels": 80,
            "cond_channels": 512,
        },
    })

    @classmethod
    def from_pretrained(cls, model_name: str) -> "CosyVoice3Config":
        try:
            hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            return cls(
                sample_rate=getattr(hf, "sample_rate", 24000),
                llm_input_size=getattr(hf, "llm_input_size", 896),
                llm_output_size=getattr(hf, "llm_output_size", 896),
                hidden_size=getattr(hf, "hidden_size", 896),
                num_attention_heads=getattr(hf, "num_attention_heads", 14),
                num_hidden_layers=getattr(hf, "num_hidden_layers", 24),
                spk_embed_dim=getattr(hf, "spk_embed_dim", 192),
                vocab_size=getattr(hf, "vocab_size", 151923),
                speech_token_size=getattr(hf, "llm", {}).get("speech_token_size", 6561),
                eos_token_id=getattr(hf, "llm", {}).get("eos_token_id", 6562),
                llm=getattr(hf, "llm", cls.llm),
                flow=getattr(hf, "flow", cls.flow),
                hift=getattr(hf, "hift", cls.hift),
                feat_extractor=getattr(hf, "feat_extractor", cls.feat_extractor),
            )
        except Exception:
            return cls()


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class CosyVoice3LM(nn.Module):
    """CosyVoice3 Language Model (talker stage).

    Wraps a Qwen2-based LLM with speech embedding and decoder layers
    for autoregressive speech token generation.
    """

    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: nn.Module,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: list[int] | None = None,
    ):
        super().__init__()
        if mix_ratio is None:
            mix_ratio = [5, 15]
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        self.llm = llm
        self.llm_decoder = nn.Linear(
            llm_output_size, speech_token_size + 200, bias=False)
        self.speech_embedding = nn.Embedding(
            speech_token_size + 200, llm_input_size)
        self.mix_ratio = mix_ratio
        self.stop_token_ids = [speech_token_size + i for i in range(200)]


class CosyVoice3Code2Wav(nn.Module):
    """CosyVoice3 Code2Wav stage for token-to-waveform conversion.

    Encapsulates flow matching decoder with DiT backbone and HiFi-GAN vocoder.
    """

    def __init__(self, config: CosyVoice3Config):
        super().__init__()
        self.config = config

        pre_lookahead_layer = PreLookaheadLayer(
            **config.flow["pre_lookahead_layer"])

        decoder_cfg = config.flow["decoder"]
        cfm_params = decoder_cfg["cfm_params"]

        estimator = DiT(**decoder_cfg["estimator"])

        decoder = CausalConditionalCFM(
            in_channels=decoder_cfg["in_channels"],
            estimator=estimator,
            cfm_params=cfm_params,
            n_spks=decoder_cfg["n_spks"],
            spk_emb_dim=decoder_cfg["spk_emb_dim"],
        )

        self.flow_model = CausalMaskedDiffWithDiT(
            input_size=config.flow["input_size"],
            output_size=config.flow["output_size"],
            spk_embed_dim=config.flow["spk_embed_dim"],
            output_type=config.flow["output_type"],
            vocab_size=config.flow["vocab_size"],
            input_frame_rate=config.flow["input_frame_rate"],
            only_mask_loss=config.flow["only_mask_loss"],
            token_mel_ratio=config.flow["token_mel_ratio"],
            pre_lookahead_len=config.flow["pre_lookahead_len"],
            pre_lookahead_layer=pre_lookahead_layer,
            decoder=decoder,
        )

        f0_predictor = CausalConvRNNF0Predictor(
            num_class=config.hift["f0_predictor"]["num_class"],
            in_channels=config.hift["f0_predictor"]["in_channels"],
            cond_channels=config.hift["f0_predictor"]["cond_channels"],
        )

        self.hift = CausalHiFTGenerator(
            in_channels=config.hift["in_channels"],
            base_channels=config.hift["base_channels"],
            nb_harmonics=config.hift["nb_harmonics"],
            sampling_rate=config.hift["sampling_rate"],
            nsf_alpha=config.hift["nsf_alpha"],
            nsf_sigma=config.hift["nsf_sigma"],
            nsf_voiced_threshold=config.hift["nsf_voiced_threshold"],
            upsample_rates=config.hift["upsample_rates"],
            upsample_kernel_sizes=config.hift["upsample_kernel_sizes"],
            istft_params=config.hift["istft_params"],
            resblock_kernel_sizes=config.hift["resblock_kernel_sizes"],
            resblock_dilation_sizes=config.hift["resblock_dilation_sizes"],
            source_resblock_kernel_sizes=config.hift["source_resblock_kernel_sizes"],
            source_resblock_dilation_sizes=config.hift["source_resblock_dilation_sizes"],
            lrelu_slope=config.hift["lrelu_slope"],
            audio_limit=config.hift["audio_limit"],
            conv_pre_look_right=config.hift["conv_pre_look_right"],
            f0_predictor=f0_predictor,
        )
        self.hift = self.hift.float()

        self.token_overlap_len = 20
        self.mel_overlap_len = int(
            self.token_overlap_len / self.flow_model.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        self.speech_window = np.hamming(2 * self.source_cache_len)

    @property
    def input_frame_rate(self):
        return self.flow_model.input_frame_rate

    @property
    def token_mel_ratio(self):
        return self.flow_model.token_mel_ratio

    @property
    def output_size(self):
        return self.flow_model.output_size

    @property
    def input_embedding(self):
        return self.flow_model.input_embedding

    @property
    def pre_lookahead_layer(self):
        return self.flow_model.pre_lookahead_layer

    @property
    def decoder(self):
        return self.flow_model.decoder

    @property
    def spk_embed_affine_layer(self):
        return self.flow_model.spk_embed_affine_layer

    @torch.inference_mode()
    def forward(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        cfm_seed: int | None = None,
    ) -> torch.Tensor:
        device = token.device
        dtype = next(self.flow_model.parameters()).dtype

        embedding = embedding.to(device=device, dtype=dtype)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        prompt_token = prompt_token.to(device=device)
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        prompt_token_len = torch.tensor([token_len1], device=device, dtype=torch.int32)
        token_len = torch.tensor([token_len2], device=device, dtype=torch.int32)

        full_token = torch.cat([prompt_token, token], dim=1)
        full_token_len = prompt_token_len + token_len

        mask = (~make_pad_mask(full_token_len)).unsqueeze(-1).to(embedding)
        token_emb = self.input_embedding(torch.clamp(full_token, min=0)) * mask
        h = self.pre_lookahead_layer(token_emb)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1

        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size],
            device=device, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mel_mask = (~make_pad_mask(
            torch.tensor([mel_len1 + mel_len2]))).to(h)

        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            cfm_seed=cfm_seed,
        )
        feat = feat[:, :, mel_len1:]

        hift_weight = self.hift.m_source.l_linear.weight
        tts_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

        if tts_mel.shape[-1] == 0:
            tts_speech = torch.zeros(
                (tts_mel.shape[0], 1, 0),
                device=tts_mel.device, dtype=tts_mel.dtype)
        else:
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel)

        return tts_speech

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        flow_path = os.path.join(model_dir, "flow.pt")
        self.flow_model.load_state_dict(
            torch.load(flow_path, map_location=device), strict=False)
        self.flow_model.to(device).eval()
        print(f"  Loaded flow weights from {flow_path}")

        hift_path = os.path.join(model_dir, "hift.pt")
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(hift_path, map_location=device).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(device).eval()
        print(f"  Loaded hift weights from {hift_path}")


class CausalMaskedDiffWithDiT(nn.Module):
    def __init__(
        self,
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=4096,
        input_frame_rate=50,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        pre_lookahead_layer=None,
        decoder=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio


class CosyVoice3ForTTS(nn.Module):
    """Top-level CosyVoice3 model matching vllm-omni's CosyVoice3Model.

    Supports three modes via *model_stage*:
      - ``"talker"``: text → speech tokens (autoregressive LLM)
      - ``"code2wav"``: speech tokens → audio waveform
      - ``"e2e"``: full end-to-end pipeline (talker + code2wav)
    """

    packed_modules_mapping = {}

    def __init__(self, config: CosyVoice3Config, model_stage: str = "code2wav"):
        super().__init__()
        self.config = config
        self.model_stage = model_stage
        self.model = None

        init_talker = model_stage in ("talker", "e2e")
        init_code2wav = model_stage in ("code2wav", "e2e")

        if init_talker:
            from .llama import LlamaConfig, LlamaModel
            llm_cfg = config.llm.get("llm", {})
            qwen_pretrain = llm_cfg.get("pretrain_path", config.qwen_pretrain_path)
            self._qwen_pretrain_path = qwen_pretrain
            qwen_config = LlamaConfig(
                hidden_size=config.llm_input_size,
                intermediate_size=config.llm_input_size * 4,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_attention_heads,
                head_dim=config.llm_input_size // config.num_attention_heads,
                vocab_size=config.vocab_size,
            )
            llm = LlamaModel(qwen_config)
            self.talker = CosyVoice3LM(
                llm_input_size=config.llm_input_size,
                llm_output_size=config.llm_output_size,
                speech_token_size=config.speech_token_size,
                llm=llm,
            )
            if model_stage == "talker":
                self.model = self.talker

        if init_code2wav:
            self.code2wav = CosyVoice3Code2Wav(config)
            if model_stage == "code2wav":
                self.model = self.code2wav.flow_model
                self.hift = self.code2wav.hift

        if not init_talker and not init_code2wav:
            raise ValueError(f"Model stage not supported: {model_stage}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token: torch.Tensor | None = None,
        prompt_token: torch.Tensor | None = None,
        prompt_feat: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        n_timesteps: int = 10,
        **kwargs,
    ):
        if self.model_stage == "talker":
            if inputs_embeds is None:
                inputs_embeds = self.talker.speech_embedding(input_ids)
            hidden_states = self.talker.llm(inputs_embeds, positions)
            return hidden_states
        elif self.model_stage == "code2wav":
            if token is not None:
                return self.code2wav(
                    token=token,
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=embedding,
                    n_timesteps=n_timesteps,
                )
            return torch.zeros(1)
        elif self.model_stage == "e2e":
            raise RuntimeError("Use generate() for end-to-end inference")

    def compute_logits(self, hidden_states):
        logits = self.talker.llm_decoder(hidden_states)
        speech_token_size = self.config.speech_token_size
        eos_id = self.config.eos_token_id
        eos_val = logits[..., eos_id].clone()
        logits[..., speech_token_size:] = float("-inf")
        logits[..., eos_id] = eos_val
        return logits

    @torch.inference_mode()
    def generate(
        self,
        text_token: torch.Tensor,
        prompt_text_token: torch.Tensor,
        speech_token: torch.Tensor,
        speech_feat: torch.Tensor,
        spk_embedding: torch.Tensor,
        n_timesteps: int = 10,
        top_p: float = 0.8,
        top_k: int = 25,
        temperature: float = 1.0,
        repetition_penalty: float = 2.0,
        max_tokens: int = 2048,
        cfm_seed: int | None = None,
    ) -> torch.Tensor:
        """Full end-to-end TTS: text + reference audio → waveform.

        Uses standard PyTorch SDPA for the talker LLM (bypassing the vLLM
        paged-attention infrastructure which requires engine context).

        Args:
            text_token: [1, text_len] int – tokenized target text
            prompt_text_token: [1, prompt_len] int – tokenized prompt text
            speech_token: [1, speech_tok_len] int – reference speech tokens
            speech_feat: [1, feat_len, 80] float – reference mel features
            spk_embedding: [1, spk_embed_dim] float – speaker embedding
            n_timesteps: flow-matching ODE steps
            cfm_seed: if set, seeds the RNG before CFM noise init for
                      deterministic Code2Wav output
        Returns:
            audio waveform tensor [1, 1, samples]
        """
        device = next(self.parameters()).device
        dtype = next(self.talker.llm.parameters()).dtype

        input_ids = torch.cat([prompt_text_token, text_token], dim=1).to(device)

        embed_weight = self.talker.llm.embed_tokens.embedding_op.emb.weight
        text_emb = F.embedding(input_ids[0], embed_weight).to(dtype)

        sos_emb = self.talker.speech_embedding.weight[self.talker.sos].unsqueeze(0).to(dtype)
        task_emb = self.talker.speech_embedding.weight[self.talker.task_id].unsqueeze(0).to(dtype)
        prompt_speech_emb = self.talker.speech_embedding(
            speech_token[0].to(device)).to(dtype)

        prefix_emb = torch.cat([
            sos_emb, text_emb, task_emb, prompt_speech_emb
        ], dim=0).unsqueeze(0)

        eos_id = self.config.eos_token_id
        speech_token_size = self.config.speech_token_size
        generated_tokens = []

        prompt_token_set: set[int] = set()
        if repetition_penalty != 1.0:
            logits_size = speech_token_size + 200
            placeholder_ids = [1] * (2 + speech_token.shape[1])
            all_prompt_ids = placeholder_ids + input_ids[0].cpu().tolist()
            prompt_token_set = {t for t in all_prompt_ids if t < logits_size}

        cur_emb = prefix_emb
        kv_cache = None
        for step in range(max_tokens):
            if kv_cache is not None:
                inp_emb = cur_emb[:, -1:]
            else:
                inp_emb = cur_emb
            hidden, kv_cache = self._llm_forward_sdpa(inp_emb, kv_cache=kv_cache)
            last_hidden = hidden[:, -1:]
            logits = self.compute_logits(last_hidden).squeeze(0).squeeze(0).float()

            if repetition_penalty != 1.0:
                seen_set = prompt_token_set | set(generated_tokens)
                if seen_set:
                    seen = torch.tensor(sorted(seen_set), device=device)
                    penalty_logits = logits[seen]
                    logits[seen] = torch.where(
                        penalty_logits > 0,
                        penalty_logits / repetition_penalty,
                        penalty_logits * repetition_penalty,
                    )

            if temperature > 0:
                logits = logits / temperature

            if temperature > 0 and top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[-1]] = float("-inf")

            if temperature > 0 and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumprobs > top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

            if temperature <= 0:
                next_token = logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id or next_token >= speech_token_size:
                break
            generated_tokens.append(next_token)

            next_emb = self.talker.speech_embedding(
                torch.tensor([[next_token]], device=device)).to(dtype)
            cur_emb = torch.cat([cur_emb, next_emb], dim=1)

        if not generated_tokens:
            return torch.zeros(1, 1, 0, device=device), []

        gen_token = torch.tensor([generated_tokens], device=device)

        audio = self.code2wav(
            token=gen_token,
            prompt_token=speech_token[:1].to(device),
            prompt_feat=speech_feat[:1].to(device=device),
            embedding=spk_embedding[:1].to(device=device),
            n_timesteps=n_timesteps,
            cfm_seed=cfm_seed,
        )
        return audio, generated_tokens

    def _llm_forward_sdpa(
        self,
        inputs_embeds: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run the talker LLM using standard SDPA with optional KV cache.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            kv_cache: list of (k, v) per layer, or None for prefill
        Returns:
            (hidden_states [batch, seq_len, hidden_size], new_kv_cache)
        """
        llm = self.talker.llm
        B, S, H = inputs_embeds.shape
        device = inputs_embeds.device

        if kv_cache is not None:
            past_len = kv_cache[0][0].shape[2]
        else:
            past_len = 0
        positions = torch.arange(past_len, past_len + S, device=device)

        new_kv_cache = []
        hidden = inputs_embeds
        for layer_idx, layer in enumerate(llm.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            residual = hidden
            ln_out = layer.input_layernorm(hidden.view(B * S, H)).view(B, S, H)

            qkv = attn.qkv_proj(ln_out.view(B * S, H))
            q_size = attn.num_heads * attn.head_dim
            kv_size = attn.num_kv_heads * attn.head_dim
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

            if attn.rotary_emb is not None:
                q, k = attn.rotary_emb(positions, q, k)

            q = q.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(B, S, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = v.view(B, S, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

            if kv_cache is not None:
                past_k, past_v = kv_cache[layer_idx]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_kv_cache.append((k, v))

            k_for_attn, v_for_attn = k, v
            if attn.num_kv_heads < attn.num_heads:
                n_rep = attn.num_heads // attn.num_kv_heads
                k_for_attn = k_for_attn.repeat_interleave(n_rep, dim=1)
                v_for_attn = v_for_attn.repeat_interleave(n_rep, dim=1)

            attn_out = F.scaled_dot_product_attention(
                q, k_for_attn, v_for_attn, is_causal=(kv_cache is None),
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B * S, -1)
            attn_out = attn.o_proj(attn_out).view(B, S, H)

            hidden = residual + attn_out

            residual = hidden
            ln_out = layer.post_attention_layernorm(hidden.view(B * S, H)).view(B, S, H)
            mlp_out = mlp(ln_out.view(B * S, -1)).view(B, S, H)
            hidden = residual + mlp_out

        hidden_flat = hidden.view(B * S, H)
        hidden_flat = llm.norm(hidden_flat)
        return hidden_flat.view(B, S, H), new_kv_cache

    def load_weights_e2e(self, model_dir: str, device: torch.device) -> None:
        """Load weights for both talker and code2wav stages.

        Rebuilds the inner LlamaModel from the actual Qwen2 config on disk
        (which may differ from the default LlamaConfig used in __init__).
        """
        from .llama import LlamaConfig, LlamaModel

        qwen_dir = os.path.join(model_dir, self._qwen_pretrain_path)
        qwen_config = LlamaConfig.from_pretrained(qwen_dir)
        new_llm = LlamaModel(qwen_config).to(device=device, dtype=torch.float32)
        self.talker.llm = new_llm

        llm_path = os.path.join(model_dir, "llm.pt")
        checkpoint = torch.load(llm_path, map_location=device)

        sd = {}
        qkv_parts: dict[str, dict[str, torch.Tensor]] = {}
        gate_up_parts: dict[str, dict[str, torch.Tensor]] = {}

        for name, weight in checkpoint.items():
            if not name.startswith("llm.model.model."):
                continue
            key = name.replace("llm.model.model.", "")

            if key == "embed_tokens.weight":
                sd["embed_tokens.embedding_op.emb.weight"] = weight
            elif key == "norm.weight":
                sd["norm.weight"] = weight
            elif ".self_attn.q_proj." in key or ".self_attn.k_proj." in key or ".self_attn.v_proj." in key:
                parts = key.split(".")
                layer_prefix = ".".join(parts[:3])
                suffix = parts[-1]
                proj = parts[3]
                full_key = f"{layer_prefix}.{proj}.{suffix}"
                qkv_key = f"{layer_prefix}.{suffix}"
                if qkv_key not in qkv_parts:
                    qkv_parts[qkv_key] = {}
                qkv_parts[qkv_key][proj] = weight
            elif ".mlp.gate_proj." in key or ".mlp.up_proj." in key:
                parts = key.split(".")
                layer_prefix = ".".join(parts[:3])
                suffix = parts[-1]
                proj = parts[3]
                gu_key = f"{layer_prefix}.{suffix}"
                if gu_key not in gate_up_parts:
                    gate_up_parts[gu_key] = {}
                gate_up_parts[gu_key][proj] = weight
            else:
                sd[key] = weight

        for qkv_key, parts in qkv_parts.items():
            layer_prefix = qkv_key.rsplit(".", 1)[0]
            suffix = qkv_key.rsplit(".", 1)[1]
            q, k, v = parts["q_proj"], parts["k_proj"], parts["v_proj"]
            sd[f"{layer_prefix}.qkv_proj.{suffix}"] = torch.cat([q, k, v], dim=0)

        for gu_key, parts in gate_up_parts.items():
            layer_prefix = gu_key.rsplit(".", 1)[0]
            suffix = gu_key.rsplit(".", 1)[1]
            gate, up = parts["gate_proj"], parts["up_proj"]
            sd[f"{layer_prefix}.gate_up_proj.{suffix}"] = torch.cat([gate, up], dim=0)

        self.talker.llm.load_state_dict(sd, strict=True)

        speech_emb_state = {
            k.replace("speech_embedding.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("speech_embedding.")
        }
        self.talker.speech_embedding.load_state_dict(speech_emb_state)

        llm_decoder_state = {
            k.replace("llm_decoder.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("llm_decoder.")
        }
        self.talker.llm_decoder.load_state_dict(llm_decoder_state)
        self.talker.to(device=device, dtype=torch.float32).eval()

        self.code2wav.load_weights(model_dir, device)
        self.code2wav.flow_model.to(dtype=torch.float32)
