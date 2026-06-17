"""CLIP text encoder for diffusion pipelines (L4).

Self-contained implementation of CLIPTextModel matching the HuggingFace
transformers checkpoint layout.  Used by FLUX.1-dev as ``text_encoder``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPTextConfig

from ..L1.layer_norm import LayerNorm
from ..L2.clip_mlp import CLIPTextEmbeddings
from ..L3.clip_encoder_layer import CLIPEncoderLayer


@dataclass
class CLIPTextModelOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class CLIPTextModel(nn.Module):
    """CLIP text encoder with pooled output.

    Weight names match ``transformers.CLIPTextModel`` under the ``text_model.``
    prefix so checkpoints are directly loadable.
    """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.text_model = _CLIPTextTransformer(config)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> CLIPTextModelOutput:
        return self.text_model(input_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        subfolder: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> "CLIPTextModel":
        import os
        from glob import glob
        from safetensors.torch import load_file

        local_path = model_name
        if subfolder:
            local_path = os.path.join(model_name, subfolder)

        if os.path.isdir(local_path):
            config = CLIPTextConfig.from_pretrained(
                local_path, local_files_only=True,
            )
            weight_dir = local_path
        else:
            config = CLIPTextConfig.from_pretrained(
                model_name, subfolder=subfolder,
                local_files_only=local_files_only,
            )
            from huggingface_hub import snapshot_download
            repo_dir = snapshot_download(
                model_name, local_files_only=local_files_only,
            )
            weight_dir = os.path.join(repo_dir, subfolder) if subfolder else repo_dir

        model = cls(config)

        safetensor_files = sorted(glob(os.path.join(weight_dir, "*.safetensors")))
        state_dict: dict[str, torch.Tensor] = {}
        for fpath in safetensor_files:
            state_dict.update(load_file(fpath))

        remapped: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = name
            if "token_embedding.weight" in name:
                new_name = name.replace("token_embedding.weight", "token_embedding.emb.weight")
            elif "position_embedding.weight" in name:
                new_name = name.replace("position_embedding.weight", "position_embedding.emb.weight")
            remapped[new_name] = tensor

        model.load_state_dict(remapped, strict=False)
        return model


class _CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.eos_token_id = config.eos_token_id

    def forward(self, input_ids: torch.Tensor) -> CLIPTextModelOutput:
        hidden_states = self.embeddings(input_ids)

        causal_mask = self._make_causal_mask(input_ids.shape, hidden_states.dtype, hidden_states.device)

        hidden_states = self.encoder(hidden_states, attention_mask=causal_mask)
        last_hidden_state = self.final_layer_norm(hidden_states)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        return CLIPTextModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )

    @staticmethod
    def _make_causal_mask(
        input_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size, seq_length = input_shape
        mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
