"""HunyuanVideo-1.5 multi-stream conditioning merge (L2 composite).

Takes three conditioning streams (MLLM text, ByT5 text, image) with
their attention masks, adds per-stream type embeddings, and reorders
tokens into ``[valid_image, valid_byt5, valid_mllm, padding]`` order.

Mirrors the conditioning logic in vllm-omni's
``HunyuanVideo15Transformer3DModel.forward`` (lines 640-721 of
``hunyuan_video_15_transformer.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.embedding import Embedding


class HunyuanVideo15ConditioningMerge(nn.Module):
    """Merge and reorder three conditioning streams for HunyuanVideo.

    Adds per-stream type embeddings (IDs 0, 1, 2) and reorders tokens
    so valid tokens come first and padding tokens last, separately for
    each batch element.
    """

    def __init__(self, inner_dim: int):
        super().__init__()
        self.cond_type_embed = Embedding(3, inner_dim)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        encoder_attention_mask_2: torch.Tensor,
        image_hidden_states: torch.Tensor,
        image_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge three streams and reorder by validity.

        Args:
            encoder_hidden_states: MLLM text embeddings ``(B, S1, D)``
            encoder_attention_mask: MLLM mask ``(B, S1)``
            encoder_hidden_states_2: ByT5 text embeddings ``(B, S2, D)``
            encoder_attention_mask_2: ByT5 mask ``(B, S2)``
            image_hidden_states: Image embeddings ``(B, S3, D)``
            image_attention_mask: Image mask ``(B, S3)``

        Returns:
            Reordered ``(encoder_hidden_states, encoder_attention_mask)``
            with layout ``[valid_image, valid_byt5, valid_mllm, padding]``.
        """
        encoder_hidden_states = encoder_hidden_states + self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )

        encoder_hidden_states_2 = encoder_hidden_states_2 + self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )

        image_hidden_states = image_hidden_states + self.cond_type_embed(
            2 * torch.ones_like(image_hidden_states[:, :, 0], dtype=torch.long)
        )

        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        image_attention_mask = image_attention_mask.bool()

        new_hidden = []
        new_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            image_hidden_states,
            image_attention_mask,
        ):
            new_hidden.append(torch.cat([
                image[image_mask],
                text_2[text_mask_2],
                text[text_mask],
                image[~image_mask],
                torch.zeros_like(text_2[~text_mask_2]),
                torch.zeros_like(text[~text_mask]),
            ], dim=0))
            new_mask.append(torch.cat([
                image_mask[image_mask],
                text_mask_2[text_mask_2],
                text_mask[text_mask],
                image_mask[~image_mask],
                text_mask_2[~text_mask_2],
                text_mask[~text_mask],
            ], dim=0))

        return torch.stack(new_hidden), torch.stack(new_mask)
