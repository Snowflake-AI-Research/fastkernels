"""V-JEPA 2 base model and video classification head."""

from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import VJEPA2Config as HFVJEPA2Config

from ..L2.vjepa2_embeddings import VJEPA2Embeddings, VJEPA2PatchEmbeddings3D
from ..L3.vjepa2_layer import VJEPA2Layer
from ..L3.vjepa2_pooler import VJEPA2AttentivePooler
from ..L3.vjepa2_predictor import VJEPA2Predictor, apply_masks


@dataclass
class VJEPA2WithMaskedInputPredictorOutput:
    last_hidden_state: torch.Tensor
    masked_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    target_hidden_state: torch.Tensor | None = None


@dataclass
class VJEPA2WithMaskedInputModelOutput:
    last_hidden_state: torch.Tensor
    masked_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    predictor_output: VJEPA2WithMaskedInputPredictorOutput | None = None


@dataclass
class VJEPA2VideoClassifierOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class VJEPA2Encoder(nn.Module):
    def __init__(self, config: HFVJEPA2Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size)
        drop_path_rates = [
            (config.drop_path_rate * i / (config.num_hidden_layers - 1) if config.num_hidden_layers > 1 else 0.0)
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList([
            VJEPA2Layer(
                config,
                drop_path_rate=drop_path_rates[i],
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = self.embeddings(pixel_values_videos)
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                position_mask=None,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return hidden_states, all_hidden_states, all_self_attentions


def _convert_head_mask_to_5d(head_mask: torch.Tensor | None, num_hidden_layers: int):
    if head_mask is not None:
        head_mask = head_mask.unsqueeze(1).unsqueeze(0)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def _snapshot_dir(model_name: str, local_files_only: bool = False) -> str:
    if os.path.isdir(model_name):
        return model_name
    return snapshot_download(
        model_name,
        local_files_only=local_files_only,
        allow_patterns=["*.safetensors", "*.json"],
    )


def _load_vjepa2_state_dict(weight_dir: str) -> dict[str, torch.Tensor]:
    safetensor_files = sorted(glob(os.path.join(weight_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {weight_dir}")
    state_dict: dict[str, torch.Tensor] = {}
    for fpath in safetensor_files:
        state_dict.update(load_file(fpath))

    remapped: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        new_name = name
        if ".patch_embeddings.proj.weight" in name:
            new_name = name.replace(".patch_embeddings.proj.weight", ".patch_embeddings.proj.conv.weight")
        elif ".patch_embeddings.proj.bias" in name:
            new_name = name.replace(".patch_embeddings.proj.bias", ".patch_embeddings.proj.conv.bias")
        remapped[new_name] = tensor
    return remapped


def _load_weights_checked(model: nn.Module, state_dict: dict[str, torch.Tensor], model_name: str) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected V-JEPA 2 checkpoint mapping for {model_name}: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )


class VJEPA2Model(nn.Module):
    def __init__(self, config: HFVJEPA2Config):
        super().__init__()
        self.config = config
        self.encoder = VJEPA2Encoder(config)
        self.predictor = VJEPA2Predictor(config)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> VJEPA2PatchEmbeddings3D:
        return self.encoder.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        context_head_mask: torch.Tensor | None = None,
        context_mask: list[torch.Tensor] | None = None,
        target_head_mask: torch.Tensor | None = None,
        target_mask: list[torch.Tensor] | None = None,
        skip_predictor: bool = False,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> VJEPA2WithMaskedInputModelOutput:
        output_attentions = bool(output_attentions) if output_attentions is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        context_head_mask = _convert_head_mask_to_5d(context_head_mask, self.config.num_hidden_layers)
        target_head_mask = _convert_head_mask_to_5d(target_head_mask, self.config.pred_num_hidden_layers)

        sequence_output, hidden_states, attentions = self.encoder(
            pixel_values_videos=pixel_values_videos,
            head_mask=context_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if context_mask is None and target_mask is None:
            batch_size = pixel_values_videos.size(0)
            num_patches = sequence_output.size(1)
            default_mask = torch.arange(num_patches, device=pixel_values_videos.device).unsqueeze(0).repeat(batch_size, 1)
            context_mask = [default_mask]
            target_mask = [default_mask]

        predictor_output = None
        if not skip_predictor:
            predictor_hidden, predictor_hidden_states, predictor_attentions = self.predictor(
                encoder_hidden_states=sequence_output,
                context_mask=context_mask,
                target_mask=target_mask,
                head_mask=target_head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            predictor_output = VJEPA2WithMaskedInputPredictorOutput(
                last_hidden_state=predictor_hidden,
                target_hidden_state=apply_masks(sequence_output, target_mask),
                hidden_states=predictor_hidden_states,
                attentions=predictor_attentions,
            )

        return VJEPA2WithMaskedInputModelOutput(
            last_hidden_state=sequence_output,
            masked_hidden_state=apply_masks(sequence_output, context_mask),
            hidden_states=hidden_states,
            attentions=attentions,
            predictor_output=predictor_output,
        )

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.forward(pixel_values_videos, skip_predictor=True).last_hidden_state

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        local_files_only: bool = False,
        **kwargs,
    ) -> "VJEPA2Model":
        config = HFVJEPA2Config.from_pretrained(model_name, local_files_only=local_files_only)
        model = cls(config)
        weight_dir = _snapshot_dir(model_name, local_files_only=local_files_only)
        state_dict = _load_vjepa2_state_dict(weight_dir)
        _load_weights_checked(model, state_dict, model_name)
        return model


class VJEPA2ForVideoClassification(nn.Module):
    def __init__(self, config: HFVJEPA2Config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.vjepa2 = VJEPA2Model(config)
        self.pooler = VJEPA2AttentivePooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> VJEPA2VideoClassifierOutput:
        outputs = self.vjepa2(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooler_output = self.pooler(outputs.last_hidden_state)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.to(logits.dtype))
            else:
                loss = F.cross_entropy(logits, labels)

        return VJEPA2VideoClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        local_files_only: bool = False,
        **kwargs,
    ) -> "VJEPA2ForVideoClassification":
        config = HFVJEPA2Config.from_pretrained(model_name, local_files_only=local_files_only)
        model = cls(config)
        weight_dir = _snapshot_dir(model_name, local_files_only=local_files_only)
        state_dict = _load_vjepa2_state_dict(weight_dir)
        _load_weights_checked(model, state_dict, model_name)
        return model
