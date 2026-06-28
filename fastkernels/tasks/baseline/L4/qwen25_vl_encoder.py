"""Qwen2.5-VL text encoder for HunyuanVideo-1.5 (L4).

Reuses the same L1-L3 stack as ``qwen2_vl.py``:
``MRotaryEmbedding`` (L1), ``LlamaAttention`` (L2, with paged attention
in prefill-only mode), ``LlamaMLP`` (L2, fused SiluAndMul), and
``LlamaDecoderLayer`` (L3, fused add+RMSNorm).

The only L4-level additions are the ``output_hidden_states`` collection
loop, prefill ``Context`` setup, and ``from_pretrained`` weight loading.

Used by HunyuanVideoPipeline as ``text_encoder`` (MLLM path).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob

import torch
import torch.nn as nn
from transformers import AutoConfig

from ....infra.context import set_context
from ..L1.embedding import Embedding
from ..L1.mrope import MRotaryEmbedding
from ..L1.rms_norm import RMSNorm
from ..L3.llama_decoder import LlamaDecoderLayer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Qwen25VLTextConfig:
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 128000
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    pad_token_id: int | None = None

    @classmethod
    def from_pretrained(cls, model_name: str, subfolder: str | None = None,
                        local_files_only: bool = False) -> "Qwen25VLTextConfig":
        hf = AutoConfig.from_pretrained(
            model_name, subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        text = getattr(hf, "text_config", hf)
        rope = getattr(text, "rope_scaling", None) or {}
        return cls(
            vocab_size=text.vocab_size,
            hidden_size=text.hidden_size,
            intermediate_size=text.intermediate_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=getattr(text, "num_key_value_heads", text.num_attention_heads),
            head_dim=getattr(text, "head_dim", text.hidden_size // text.num_attention_heads),
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=getattr(text, "rope_theta", 1000000.0),
            max_position_embeddings=text.max_position_embeddings,
            mrope_section=rope.get("mrope_section", [16, 24, 24]),
            pad_token_id=getattr(text, "pad_token_id", None),
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Qwen25VLTextEncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen25VLTextEncoder(nn.Module):
    """Qwen2.5-VL text encoder for HunyuanVideo-1.5.

    L4 pipeline wiring the same L1-L3 ops used by ``Qwen2Model`` in
    ``qwen2_vl.py``: ``MRotaryEmbedding`` -> ``LlamaDecoderLayer``
    (with ``LlamaAttention`` + paged attention prefill + ``LlamaMLP``)
    -> ``RMSNorm``.

    The paged ``Attention`` backend is run in prefill-only mode (no KV
    cache pages, no block tables) by setting up a minimal ``Context``
    before each forward pass.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen25VLTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = MRotaryEmbedding(
            config.head_dim, config.max_position_embeddings,
            config.rope_theta, config.mrope_section,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, rotary_emb=self.rotary_emb, bias=True)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.emb.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.emb.weight.device

    def _setup_prefill_context(self, seq_len: int, device: torch.device) -> None:
        """Configure the global Context for a single-sequence prefill pass."""
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=torch.empty(0, dtype=torch.long, device=device),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Qwen25VLTextEncoderOutput:
        batch_size, seq_len = input_ids.shape

        if batch_size != 1:
            return self._forward_batched(input_ids, attention_mask, output_hidden_states)

        self._setup_prefill_context(seq_len, input_ids.device)

        hidden_states = self.embed_tokens(input_ids.view(-1))
        positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)

        all_hidden_states = () if output_hidden_states else None

        residual = None
        for layer in self.layers:
            if output_hidden_states:
                if residual is not None:
                    all_hidden_states += (hidden_states + residual,)
                else:
                    all_hidden_states += (hidden_states,)
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = hidden_states.unsqueeze(0)

        if output_hidden_states:
            all_hidden_states = tuple(h.unsqueeze(0) for h in all_hidden_states)
            all_hidden_states += (hidden_states,)

        return Qwen25VLTextEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def _forward_batched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
    ) -> Qwen25VLTextEncoderOutput:
        """Handle batch_size > 1 by iterating over each sequence.

        The paged attention backend uses global Context which is
        single-batch, so we process one sequence at a time and stack.
        Each sequence is processed at full length (including padding)
        so output shapes match the input — callers rely on attention
        masks to ignore padding positions.
        """
        results = []
        for i in range(input_ids.shape[0]):
            ids_i = input_ids[i:i+1]
            results.append(self.forward(ids_i, output_hidden_states=output_hidden_states))

        stacked_last = torch.cat([r.last_hidden_state for r in results], dim=0)

        stacked_hidden = None
        if output_hidden_states:
            num_layers_plus_one = len(results[0].hidden_states)
            stacked_hidden = tuple(
                torch.cat([r.hidden_states[li] for r in results], dim=0)
                for li in range(num_layers_plus_one)
            )

        return Qwen25VLTextEncoderOutput(
            last_hidden_state=stacked_last,
            hidden_states=stacked_hidden,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        subfolder: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> "Qwen25VLTextEncoder":
        config = Qwen25VLTextConfig.from_pretrained(
            model_name, subfolder=subfolder,
            local_files_only=local_files_only,
        )
        model = cls(config)

        from safetensors.torch import load_file

        local_path = model_name
        if subfolder:
            local_path = os.path.join(model_name, subfolder)

        if not os.path.isdir(local_path):
            from huggingface_hub import snapshot_download
            repo_dir = snapshot_download(
                model_name, local_files_only=local_files_only,
                allow_patterns=[f"{subfolder}/*"] if subfolder else None,
            )
            local_path = os.path.join(repo_dir, subfolder) if subfolder else repo_dir

        safetensor_files = sorted(glob(os.path.join(local_path, "*.safetensors")))
        state_dict: dict[str, torch.Tensor] = {}
        for fpath in safetensor_files:
            state_dict.update(load_file(fpath))

        _load_weights(model, state_dict)
        return model


def _default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)


def _load_weights(model: Qwen25VLTextEncoder, state_dict: dict[str, torch.Tensor]) -> None:
    """Load HF checkpoint weights with packed-module remapping.

    The HF checkpoint stores separate ``q_proj``/``k_proj``/``v_proj``
    and ``gate_proj``/``up_proj``/``down_proj``, while our L2/L3 layers
    use merged ``qkv_proj`` and ``gate_up_proj``.  This function handles
    the remapping, mirroring ``infra/weight_loader.py``.
    """
    packed = model.packed_modules_mapping
    params_dict = dict(model.named_parameters())

    for name, tensor in state_dict.items():
        mapped_name = name
        if mapped_name.startswith("model."):
            mapped_name = mapped_name[len("model."):]

        if name == "embed_tokens.weight" or mapped_name == "embed_tokens.weight":
            mapped_name = "embed_tokens.emb.weight"

        matched = False
        for orig_key, (packed_name, shard_id) in packed.items():
            if orig_key in mapped_name:
                param_name = mapped_name.replace(orig_key, packed_name)
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader:
                        weight_loader(param, tensor, shard_id)
                        matched = True
                break

        if matched:
            continue

        if "rotary_emb" in mapped_name:
            continue

        if mapped_name in params_dict:
            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader", _default_weight_loader)
            weight_loader(param, tensor)
