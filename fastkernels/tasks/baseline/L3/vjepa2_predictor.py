"""V-JEPA 2 predictor stack."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L3.vjepa2_layer import VJEPA2Layer


def apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather tokens from ``tensor`` according to each mask in ``masks``."""
    masked_tensors = []
    for mask in masks:
        mask = mask.to(tensor.device)
        mask_keep = mask.unsqueeze(-1).repeat(1, 1, tensor.size(-1))
        masked_tensors.append(torch.gather(tensor, dim=1, index=mask_keep))
    return torch.cat(masked_tensors, dim=0)


class VJEPA2PredictorEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictor_embeddings = Linear(config.hidden_size, config.pred_hidden_size, bias=True)
        self.num_mask_tokens = config.pred_num_mask_tokens
        self.zero_init_mask_tokens = config.pred_zero_init_mask_tokens
        self.mask_tokens = nn.Parameter(
            torch.zeros(self.num_mask_tokens, 1, 1, config.pred_hidden_size)
        )
        self.patch_size = config.patch_size
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mask_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)
        context = self.predictor_embeddings(hidden_states)

        mask_index = mask_index % self.num_mask_tokens
        target = self.mask_tokens[mask_index]
        max_patch_num = int(target_mask[0].max().item()) + 1
        target = target.repeat(batch_size, max_patch_num, 1)
        target = apply_masks(target, target_mask)

        context = context.repeat(len(context_mask), 1, 1)
        embeddings = torch.cat([context, target], dim=1)
        masks = torch.cat([torch.cat(context_mask, dim=0), torch.cat(target_mask, dim=0)], dim=1)
        return embeddings, masks


class VJEPA2Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA2PredictorEmbeddings(config)
        drop_path_rates = [
            (config.drop_path_rate * i / (config.pred_num_hidden_layers - 1)
             if config.pred_num_hidden_layers > 1 else 0.0)
            for i in range(config.pred_num_hidden_layers)
        ]
        self.layer = nn.ModuleList([
            VJEPA2Layer(
                config,
                drop_path_rate=drop_path_rates[i],
                hidden_size=config.pred_hidden_size,
                num_attention_heads=config.pred_num_attention_heads,
                mlp_ratio=config.pred_mlp_ratio,
            )
            for i in range(config.pred_num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(config.pred_hidden_size, eps=config.layer_norm_eps)
        self.proj = Linear(config.pred_hidden_size, config.hidden_size, bias=True)

    def sort_tokens(
        self,
        hidden_states: torch.Tensor,
        position_masks: torch.Tensor,
        argsort: torch.Tensor,
        head_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        argsort = argsort.to(position_masks.device)
        position_masks = torch.gather(position_masks, dim=1, index=argsort)

        argsort_hidden = argsort.to(hidden_states.device).unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, dim=1, index=argsort_hidden)

        if head_mask is not None and head_mask[0] is not None:
            hm = head_mask.permute(1, 0, 2, 3, 4)
            argsort_4d = (
                argsort.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, hm.size(1), hm.size(2), -1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, -1, hm.size(-1))
            )
            hm = torch.gather(hm, dim=3, index=argsort_4d)
            argsort_5d = (
                argsort.unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, hm.size(1), hm.size(2), hm.size(3), -1)
            )
            hm = torch.gather(hm, dim=4, index=argsort_5d)
            head_mask = hm.permute(1, 0, 2, 3, 4)

        return hidden_states, position_masks, head_mask

    def unsort_tokens(self, hidden_states: torch.Tensor, argsort: torch.Tensor) -> torch.Tensor:
        reverse_argsort = torch.argsort(argsort.to(hidden_states.device), dim=1)
        reverse_argsort = reverse_argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        return torch.gather(hidden_states, dim=1, index=reverse_argsort)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        num_context_tokens = encoder_hidden_states.shape[1]
        hidden_states, position_masks = self.embeddings(
            encoder_hidden_states, context_mask, target_mask,
        )

        argsort = torch.argsort(position_masks, dim=1)
        hidden_states, position_masks, head_mask = self.sort_tokens(
            hidden_states, position_masks, argsort, head_mask=head_mask,
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                position_mask=position_masks,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, num_context_tokens:]
        hidden_states = self.proj(hidden_states)

        return hidden_states, all_hidden_states, all_self_attentions
