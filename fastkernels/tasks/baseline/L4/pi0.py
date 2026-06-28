"""Pi0 vision-language-action model (L4 pipeline).

Contains:
- SigLIPVisionEncoder: SigLIP So400m/14 vision tower.
- GemmaModel: Shared Gemma transformer (used for both VLM and action expert).
- Pi0Model: Wires VLM + DiT with KV-cache sharing.
- Pi0Pipeline: Full inference pipeline with flow-matching action generation.

L4 wiring/configuration; computation lives in L1-L3 tasks.

Reference SOTA: HuggingFace Transformers ``PI0ForConditionalGeneration``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import nn

from ..L1.conv2d import Conv2d
from ..L1.embedding import Embedding
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.rms_norm import RMSNorm
from ..L2.gemma_dense_attention import GemmaRotaryEmbedding, build_pi0_dit_attn_bias
from ..L2.pi0_action_embed import Pi0ActionTimeEmbedding
from ..L3.gemma_dense_decoder_layer import GemmaDenseDecoderLayer
from ..L3.siglip_encoder_layer import SigLIPEncoderLayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SigLIPVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6


@dataclass
class GemmaConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 18
    intermediate_size: int = 16384
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 257152
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0


@dataclass
class Pi0Config:
    """Full Pi0 configuration."""

    vlm_text_config: GemmaConfig = field(default_factory=GemmaConfig)
    vlm_vision_config: SigLIPVisionConfig = field(default_factory=SigLIPVisionConfig)
    dit_config: GemmaConfig = field(default_factory=lambda: GemmaConfig(
        hidden_size=1024,
        intermediate_size=4096,
        head_dim=256,
    ))
    projection_dim: int = 2048
    image_token_id: int = 257152

    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10

    min_period: float = 0.004
    max_period: float = 4.0

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Pi0Config":
        """Load config from a HuggingFace PI0 checkpoint."""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json at {config_path}")
        with open(config_path) as f:
            data = json.load(f)

        vlm_cfg = data.get("vlm_config", {})
        text_cfg = vlm_cfg.get("text_config", {})
        vision_cfg = vlm_cfg.get("vision_config", {})
        dit_cfg = data.get("dit_config", {})

        return cls(
            vlm_text_config=GemmaConfig(
                hidden_size=text_cfg.get("hidden_size", 2048),
                num_hidden_layers=text_cfg.get("num_hidden_layers", 18),
                intermediate_size=text_cfg.get("intermediate_size", 16384),
                num_attention_heads=text_cfg.get("num_attention_heads", 8),
                num_key_value_heads=text_cfg.get("num_key_value_heads", 1),
                head_dim=text_cfg.get("head_dim", 256),
                vocab_size=text_cfg.get("vocab_size", 257152),
                rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
                max_position_embeddings=text_cfg.get("max_position_embeddings", 8192),
                rope_theta=text_cfg.get("rope_theta", 10000.0),
            ),
            vlm_vision_config=SigLIPVisionConfig(
                hidden_size=vision_cfg.get("hidden_size", 1152),
                intermediate_size=vision_cfg.get("intermediate_size", 4304),
                num_hidden_layers=vision_cfg.get("num_hidden_layers", 27),
                num_attention_heads=vision_cfg.get("num_attention_heads", 16),
                image_size=vision_cfg.get("image_size", 224),
                patch_size=vision_cfg.get("patch_size", 14),
            ),
            dit_config=GemmaConfig(
                hidden_size=dit_cfg.get("hidden_size", 1024),
                num_hidden_layers=dit_cfg.get("num_hidden_layers", 18),
                intermediate_size=dit_cfg.get("intermediate_size", 4096),
                num_attention_heads=dit_cfg.get("num_attention_heads", 8),
                num_key_value_heads=dit_cfg.get("num_key_value_heads", 1),
                head_dim=dit_cfg.get("head_dim", 256),
                vocab_size=text_cfg.get("vocab_size", 257152),
                rms_norm_eps=dit_cfg.get("rms_norm_eps", 1e-6),
                max_position_embeddings=dit_cfg.get("max_position_embeddings", 8192),
                rope_theta=dit_cfg.get("rope_theta", 10000.0),
            ),
            projection_dim=vlm_cfg.get("projection_dim", 2048),
            image_token_id=vlm_cfg.get("image_token_id", 257152),
            chunk_size=data.get("chunk_size", data.get("action_horizon", 50)),
            max_state_dim=data.get("max_state_dim", 32),
            max_action_dim=data.get("max_action_dim", 32),
            num_inference_steps=data.get("num_inference_steps", 10),
            min_period=data.get("min_period", 0.004),
            max_period=data.get("max_period", 4.0),
        )


# ---------------------------------------------------------------------------
# SigLIP Vision Encoder
# ---------------------------------------------------------------------------

class SigLIPVisionEncoder(nn.Module):
    """SigLIP So400m/14 vision tower.

    Conv2d patch embedding -> learned position embeddings -> N transformer
    encoder layers -> LayerNorm.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patch_embedding = Conv2d(
            3, config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size),
        )
        self.layers = nn.ModuleList([
            SigLIPEncoderLayer(
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.layer_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, image_size, image_size)
        Returns:
            (batch, num_patches, hidden_size)
        """
        patch_embeds = self.patch_embedding(pixel_values)
        bsz, hidden, h, w = patch_embeds.shape
        patch_embeds = patch_embeds.reshape(bsz, hidden, h * w).transpose(1, 2)
        hidden_states = patch_embeds + self.position_embedding

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.post_layernorm(hidden_states)


# ---------------------------------------------------------------------------
# Gemma Transformer
# ---------------------------------------------------------------------------

class GemmaModel(nn.Module):
    """Gemma decoder-only transformer (shared by VLM and DiT).

    When used as the VLM backbone, includes token embedding.
    When used as the DiT (action expert), receives pre-embedded inputs.
    """

    def __init__(self, config: GemmaConfig, embed_tokens: bool = True):
        super().__init__()
        self.config = config
        if embed_tokens:
            self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None
        self.layers = nn.ModuleList([
            GemmaDenseDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GemmaRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Returns:
            hidden_states: (batch, seq, hidden_size)
            new_kv_caches: list of (key, value) per layer, or None.
        """
        if inputs_embeds is None:
            assert input_ids is not None and self.embed_tokens is not None
            hidden_states = self.embed_tokens(input_ids)
            hidden_states = hidden_states * (self.config.hidden_size ** 0.5)
        else:
            hidden_states = inputs_embeds

        bsz, seq_len = hidden_states.shape[:2]
        is_dit_with_prefix = kv_caches is not None and not use_cache

        if position_ids is None:
            if is_dit_with_prefix:
                prefix_len = kv_caches[0][0].shape[1]
                if attention_mask is not None:
                    ones_suf = torch.ones(
                        bsz, seq_len,
                        device=hidden_states.device,
                        dtype=attention_mask.dtype,
                    )
                    dit_am = torch.cat([attention_mask, ones_suf], dim=1)
                else:
                    am = torch.ones(
                        bsz, prefix_len,
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
                    ones_suf = torch.ones(
                        bsz, seq_len,
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
                    dit_am = torch.cat([am, ones_suf], dim=1)
                position_ids = (dit_am.cumsum(dim=1) - 1)[:, -seq_len:]
            else:
                cache_len = kv_caches[0][0].shape[1] if kv_caches else 0
                position_ids = torch.arange(
                    cache_len, cache_len + seq_len,
                    device=hidden_states.device,
                ).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        pi0_attn_bias = None
        if is_dit_with_prefix:
            prefix_len = kv_caches[0][0].shape[1]
            pi0_attn_bias = build_pi0_dit_attn_bias(
                prefix_len, seq_len, bsz, hidden_states.device,
            )
            if attention_mask is not None:
                prefix_pad = (attention_mask == 0)[:, None, None, :]
                neg_inf = torch.finfo(pi0_attn_bias.dtype).min
                pi0_attn_bias = pi0_attn_bias.clone()
                pi0_attn_bias[..., :prefix_len].masked_fill_(prefix_pad, neg_inf)
        elif attention_mask is not None:
            am = attention_mask.to(torch.bool)
            pad_2d = (am[:, :, None] & am[:, None, :])[:, None, :, :]
            neg_inf = torch.finfo(hidden_states.dtype).min
            pi0_attn_bias = torch.zeros(
                bsz, 1, seq_len, seq_len,
                dtype=hidden_states.dtype, device=hidden_states.device,
            ).masked_fill_(~pad_2d, neg_inf)

        new_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches else None
            hidden_states, new_kv = layer(
                hidden_states,
                cos,
                sin,
                attention_mask=pi0_attn_bias,
                kv_cache=layer_kv,
            )
            if use_cache:
                new_kv_caches.append(new_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_caches


# ---------------------------------------------------------------------------
# Pi0 Model (VLM + DiT with KV sharing)
# ---------------------------------------------------------------------------

class Pi0Model(nn.Module):
    """Wires VLM backbone and DiT action expert with KV-cache sharing."""

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        self.vision_tower = SigLIPVisionEncoder(config.vlm_vision_config)
        self.multi_modal_projector = Linear(
            config.vlm_vision_config.hidden_size, config.projection_dim,
        )
        self.vlm = GemmaModel(config.vlm_text_config, embed_tokens=True)
        self.dit = GemmaModel(config.dit_config, embed_tokens=False)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through SigLIP + projector.

        Args:
            pixel_values: (batch * num_cameras, 3, H, W)
        Returns:
            (batch * num_cameras, num_patches, projection_dim)
        """
        vision_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(vision_outputs)

    def embed_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed image+text tokens for the VLM prefix.

        Scatters image features into the positions indicated by image_token_id.
        """
        max_num_cameras = pixel_attention_mask.shape[1]
        flat_pixels = pixel_values.flatten(0, 1)
        image_features = self.encode_images(flat_pixels)
        image_features = image_features.reshape(
            -1, max_num_cameras, image_features.shape[1], image_features.shape[2],
        )

        total_image_features = []
        for batch_idx, mask in enumerate(pixel_attention_mask):
            unpadded = image_features[batch_idx][mask]
            total_image_features.append(unpadded)
        total_image_features = torch.cat(total_image_features, dim=0)

        llm_input_ids = input_ids.clone()
        llm_input_ids[input_ids == self.config.image_token_id] = 0
        inputs_embeds = self.vlm.embed_tokens(llm_input_ids)
        inputs_embeds = inputs_embeds * (self.config.vlm_text_config.hidden_size ** 0.5)

        special_image_mask = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask, total_image_features,
        )
        return inputs_embeds

    def forward_vlm_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Forward the VLM prefix and return cached KV.

        Returns:
            kv_caches: list of (key, value) per layer.
        """
        inputs_embeds = self.embed_prefix(input_ids, pixel_values, pixel_attention_mask)

        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(-1) - 1

        _, kv_caches = self.vlm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return kv_caches

    def forward_dit(
        self,
        action_embeds: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward action tokens through the DiT using VLM KV cache.

        Args:
            action_embeds: (batch, 1 + chunk_size, dit_hidden_size)
            kv_caches: VLM-prefix KV caches.
            attention_mask: (batch, prefix_seq_len) mask for the VLM prefix (HF-aligned).

        Returns:
            (batch, 1 + chunk_size, dit_hidden_size)
        """
        hidden_states, _ = self.dit(
            inputs_embeds=action_embeds,
            position_ids=None,
            attention_mask=attention_mask,
            kv_caches=kv_caches,
            use_cache=False,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Sampling outputs
# ---------------------------------------------------------------------------

@dataclass
class Pi0SamplingParams:
    """Parameters for Pi0 action generation."""
    num_inference_steps: int = 10
    seed: int | None = None


@dataclass
class Pi0Output:
    """Output of the Pi0 pipeline."""
    actions: torch.Tensor | None = None
    infer_ms: float = 0.0


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class Pi0Pipeline(nn.Module):
    """Full Pi0 inference pipeline.

    Composes: SigLIP encode -> VLM prefix -> flow-matching action generation.
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        self.model = Pi0Model(config)
        self.embed_action_time = Pi0ActionTimeEmbedding(
            expert_hidden_size=config.dit_config.hidden_size,
            max_action_dim=config.max_action_dim,
            max_state_dim=config.max_state_dim,
            min_period=config.min_period,
            max_period=config.max_period,
        )
        self.action_out_proj = Linear(
            config.dit_config.hidden_size, config.max_action_dim,
        )

    @torch.inference_mode()
    def forward(
        self,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        params: Pi0SamplingParams | None = None,
    ) -> Pi0Output:
        """Run flow-matching inference to generate actions.

        Args:
            state: (batch, max_state_dim) robot state.
            input_ids: (batch, seq_len) tokenized instruction.
            pixel_values: (batch, num_cameras, 3, H, W) camera images.
            pixel_attention_mask: (batch, num_cameras) bool mask.
            attention_mask: (batch, seq_len) text attention mask.
            noise: Optional (batch, chunk_size, max_action_dim) initial noise.
            params: Sampling parameters.

        Returns:
            Pi0Output with predicted action chunk.
        """
        import time as _time

        params = params or Pi0SamplingParams()
        num_steps = params.num_inference_steps or self.config.num_inference_steps
        batch_size = state.shape[0]
        device = state.device

        if noise is None:
            noise = torch.randn(
                batch_size, self.config.chunk_size, self.config.max_action_dim,
                dtype=pixel_values.dtype, device=device,
            )

        t0 = _time.perf_counter()

        vlm_kv = self.model.forward_vlm_prefix(
            input_ids, pixel_values, pixel_attention_mask, attention_mask,
        )
        prefix_len = vlm_kv[0][0].shape[1]

        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 + step * dt
            timestep = torch.tensor(
                t, dtype=torch.float32, device=device,
            ).expand(batch_size)

            action_embeds = self.embed_action_time(state, noise, timestep)

            dit_out = self.model.forward_dit(
                action_embeds, vlm_kv, attention_mask=attention_mask,
            )

            velocity = self.action_out_proj(
                dit_out[:, -self.config.chunk_size:],
            )
            noise = noise + dt * velocity

            vlm_kv = [
                (kv[0][:, :prefix_len], kv[1][:, :prefix_len])
                for kv in vlm_kv
            ]

        elapsed_ms = (_time.perf_counter() - t0) * 1000
        return Pi0Output(actions=noise, infer_ms=elapsed_ms)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights from LeRobot-format checkpoint (lerobot/pi0_base).

        Handles name remapping between LeRobot's ``paligemma_with_expert``
        weight keys and our module hierarchy.
        """
        _PWE_PAL = "paligemma_with_expert.paligemma."
        _PWE_EXP = "paligemma_with_expert.gemma_expert."

        remap = {
            # Vision tower
            f"{_PWE_PAL}model.vision_tower.vision_model.embeddings.patch_embedding.": "model.vision_tower.patch_embedding.",
            f"{_PWE_PAL}model.vision_tower.vision_model.encoder.layers.": "model.vision_tower.layers.",
            f"{_PWE_PAL}model.vision_tower.vision_model.post_layernorm.": "model.vision_tower.post_layernorm.",
            # Multi-modal projector
            f"{_PWE_PAL}model.multi_modal_projector.linear.": "model.multi_modal_projector.",
            # VLM language model
            f"{_PWE_PAL}model.language_model.layers.": "model.vlm.layers.",
            f"{_PWE_PAL}model.language_model.norm.": "model.vlm.norm.",
            # DiT action expert
            f"{_PWE_EXP}model.layers.": "model.dit.layers.",
            f"{_PWE_EXP}model.norm.": "model.dit.norm.",
            # Action embeddings (top-level in checkpoint)
            "action_in_proj.": "embed_action_time.action_in_proj.",
            "action_time_mlp_in.": "embed_action_time.action_time_mlp_in.",
            "action_time_mlp_out.": "embed_action_time.action_time_mlp_out.",
            "state_proj.": "embed_action_time.state_proj.",
            # Action output projection
            "action_out_proj.": "action_out_proj.",
        }

        _POS_EMB_KEY = (
            f"{_PWE_PAL}model.vision_tower.vision_model."
            "embeddings.position_embedding.weight"
        )
        _VLM_LM_HEAD = f"{_PWE_PAL}lm_head.weight"

        params_dict = dict(self.named_parameters())
        for name, buf in self.named_buffers():
            params_dict[name] = buf

        loaded = set()

        for name, tensor in weights:
            original_name = name

            if name == _POS_EMB_KEY:
                self.model.vision_tower.position_embedding.data.copy_(
                    tensor.unsqueeze(0) if tensor.ndim == 2 else tensor,
                )
                loaded.add(original_name)
                continue

            if name == _VLM_LM_HEAD:
                self.model.vlm.embed_tokens.emb.weight.data.copy_(tensor)
                loaded.add(original_name)
                continue

            if "lm_head" in name:
                loaded.add(original_name)
                continue

            mapped = name
            for src, dst in remap.items():
                if name.startswith(src):
                    mapped = name.replace(src, dst, 1)
                    break

            is_gate = ".mlp.gate_proj." in name
            is_up = ".mlp.up_proj." in name

            if is_gate or is_up:
                gate_up_name = mapped.replace(
                    ".mlp.gate_proj.", ".mlp.gate_up_proj.",
                ).replace(
                    ".mlp.up_proj.", ".mlp.gate_up_proj.",
                )
                if gate_up_name in params_dict:
                    param = params_dict[gate_up_name]
                    mid = param.shape[0] // 2
                    if is_gate:
                        param.data[:mid].copy_(tensor)
                    else:
                        param.data[mid:].copy_(tensor)
                    loaded.add(original_name)
                    continue

            if mapped in params_dict:
                # Gemma RMSNorm stores weight as offset from 1 (initialized
                # to zeros, applied as ``x * (1 + w)``).  Our RMSNorm uses
                # ``x * w``, so we add 1 to compensate.
                is_gemma_norm = (
                    mapped.startswith("model.vlm.") or mapped.startswith("model.dit.")
                ) and mapped.endswith(("layernorm.weight", "norm.weight"))
                if is_gemma_norm:
                    params_dict[mapped].data.copy_(tensor + 1.0)
                else:
                    params_dict[mapped].data.copy_(tensor)
                loaded.add(original_name)
            else:
                logger.debug("Skipping unrecognized weight: %s -> %s", name, mapped)

        return loaded
