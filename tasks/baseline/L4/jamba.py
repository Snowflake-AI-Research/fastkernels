"""Standalone Jamba model implementation (L4 pipeline wiring).

Jamba (AI21Labs) is a triple-hybrid:

  * **Transformer attention** in a small fraction of layers (4 out of
    every 32 in v0.1, 4 out of every 16 in tiny-dev -- the schedule
    derives from ``attn_layer_period``/``attn_layer_offset`` in the HF
    config).
  * **Mamba v1** selective state-space mixers in the remaining majority
    of layers.
  * **Sparse Mixture of Experts** (top-2 of 16 in v0.1, top-2 of 8 in
    tiny-dev) on every other layer's FFN.  The non-MoE layers use a
    plain SwiGLU MLP.

The mix gives Jamba large effective context with subquadratic compute:
the few attention layers carry global token-token mixing, the Mamba
layers carry the bulk of the work in linear time, and MoE adds
parameter capacity without proportional FLOPs.

This file is the L4 pipeline wiring: layer construction, embedding,
final layer norm, LM head, and a checkpoint-aligned ``forward``.  All
heavy lifting lives in the L2/L3 modules.

Reference for the layer schedule and weight names:
``transformers.models.jamba.modeling_jamba`` (HuggingFace).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.rms_norm import RMSNorm
from ..L3.jamba_decoder import (
    JambaAttentionDecoderLayer,
    JambaMambaDecoderLayer,
)


# ---------------------------------------------------------------------------
# Config (mirrors transformers.models.jamba.configuration_jamba.JambaConfig
# field-for-field for the bits the pipeline actually consumes).
# ---------------------------------------------------------------------------
@dataclass
class JambaConfig:
    # ---- Common LLM fields ----
    vocab_size: int = 65536
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False

    # ---- Mamba mixer fields ----
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int = 256
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    # ---- MoE fields ----
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # ---- Hybrid recipe ----
    # Attention layers at indices ``i % attn_layer_period == attn_layer_offset``.
    # MoE layers   at indices ``i % expert_layer_period == expert_layer_offset``.
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    expert_layer_period: int = 2
    expert_layer_offset: int = 1

    # Derived: per-layer block type and per-layer expert count.
    # Filled by ``__post_init__`` if not provided explicitly.
    layers_block_type: list[str] = field(default_factory=list)
    layers_num_experts: list[int] = field(default_factory=list)

    # Inference dtype.
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if not self.layers_block_type:
            self.layers_block_type = [
                "attention" if (i % self.attn_layer_period == self.attn_layer_offset)
                else "mamba"
                for i in range(self.num_hidden_layers)
            ]
        if not self.layers_num_experts:
            self.layers_num_experts = [
                self.num_experts
                if (i % self.expert_layer_period == self.expert_layer_offset)
                else 1
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layers_block_type) == self.num_hidden_layers
        assert len(self.layers_num_experts) == self.num_hidden_layers

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "JambaConfig":
        path = Path(model_path)
        if path.is_dir():
            config_path = path / "config.json"
        else:
            config_path = path
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        kwargs = {
            "vocab_size": data["vocab_size"],
            "hidden_size": data["hidden_size"],
            "intermediate_size": data["intermediate_size"],
            "num_hidden_layers": data["num_hidden_layers"],
            "num_attention_heads": data["num_attention_heads"],
            "num_key_value_heads": data["num_key_value_heads"],
            "rms_norm_eps": data.get("rms_norm_eps", 1e-6),
            "pad_token_id": data.get("pad_token_id", 0),
            "bos_token_id": data.get("bos_token_id", 1),
            "eos_token_id": data.get("eos_token_id", 2),
            "tie_word_embeddings": data.get("tie_word_embeddings", False),
            "mamba_d_state": data.get("mamba_d_state", 16),
            "mamba_d_conv": data.get("mamba_d_conv", 4),
            "mamba_expand": data.get("mamba_expand", 2),
            "mamba_dt_rank": data.get("mamba_dt_rank", 256),
            "mamba_conv_bias": data.get("mamba_conv_bias", True),
            "mamba_proj_bias": data.get("mamba_proj_bias", False),
            "num_experts": data.get("num_experts", 16),
            "num_experts_per_tok": data.get("num_experts_per_tok", 2),
            "attn_layer_period": data.get("attn_layer_period", 8),
            "attn_layer_offset": data.get("attn_layer_offset", 4),
            "expert_layer_period": data.get("expert_layer_period", 2),
            "expert_layer_offset": data.get("expert_layer_offset", 1),
        }
        # Some checkpoints (e.g. v0.1) ship the explicit per-layer
        # arrays; honour those when present so we never disagree with
        # the checkpoint about which layer is which.
        if "layers_block_type" in data:
            kwargs["layers_block_type"] = list(data["layers_block_type"])
        if "layers_num_experts" in data:
            kwargs["layers_num_experts"] = list(data["layers_num_experts"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class JambaModel(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id,
        )

        layers: list[nn.Module] = []
        for i in range(config.num_hidden_layers):
            block_type = config.layers_block_type[i]
            if block_type == "attention":
                layers.append(JambaAttentionDecoderLayer(config, layer_idx=i))
            elif block_type == "mamba":
                layers.append(JambaMambaDecoderLayer(config, layer_idx=i))
            else:  # pragma: no cover -- defensive
                raise ValueError(f"Unknown layers_block_type[{i}]={block_type!r}")
        self.layers = nn.ModuleList(layers)

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Convenience handle: indices of the attention vs mamba layers.
        # Used by the engine to size KV cache slabs and to build the
        # layered cache views.
        self.attention_layer_indices = [
            i for i, t in enumerate(config.layers_block_type) if t == "attention"
        ]
        self.mamba_layer_indices = [
            i for i, t in enumerate(config.layers_block_type) if t == "mamba"
        ]

        # Each L2 mixer (``JambaAttention`` / ``JambaMambaMixer``) reads
        # its per-layer slab from the per-step metadata installed on the
        # global ``Context`` using ``self.layer_idx``.  The slab list is
        # sized to the *count* of attention (or Mamba) layers, not the
        # full layer count, so we remap the physical layer index to its
        # position in the per-kind slab list here once at init time.
        attn_layer_to_slab = {li: si for si, li in enumerate(self.attention_layer_indices)}
        mamba_layer_to_slab = {li: si for si, li in enumerate(self.mamba_layer_indices)}
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = attn_layer_to_slab[layer.layer_idx]
            elif hasattr(layer, "mamba"):
                layer.mamba.layer_idx = mamba_layer_to_slab[layer.layer_idx]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward following the project convention.

        Per-step KV / Mamba state is read from the global ``Context``
        (populated by ``set_jamba_context`` in the engine) so that this
        ``forward`` matches the ``(input_ids, positions, inputs_embeds=None)``
        signature used by every other LLM model in the project (Llama,
        Mamba, Mamba2, Mixtral, ...).

        Returns the post-final-layernorm hidden states; the LM-head
        projection is performed separately in ``compute_logits`` to keep
        graph capture / sampling logic out of the model body.
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        # Llama-style fused residual: ``hidden_states`` carries the
        # per-block delta and ``residual`` is the running residual stream;
        # the L3 layer's input_layernorm / pre_ff_layernorm both call
        # ``norm(hidden_states, residual)`` which fuses the add + norm
        # into one CUDA kernel (vLLM ``fused_add_rms_norm``) for
        # bf16-identical numerics with vLLM's Jamba forward.
        # ``positions`` is unused by Jamba's mixers (no RoPE; Mamba
        # carries position via its recurrence) but kept for signature
        # parity with Llama / Mamba / Mixtral.
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Fold the trailing residual into the final norm (same pattern
        # as LlamaModel.forward).
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class JambaForCausalLM(nn.Module):
    """Jamba LM head wrapper.

    The HF checkpoint stores the LM head separately
    (``tie_word_embeddings=False``) -- we load both ``embed_tokens`` and
    ``lm_head`` from the safetensors.

    Weight name layout (matches HF transformers' Jamba checkpoint):

        model.embed_tokens.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.pre_ff_layernorm.weight

        # Attention layers:
        model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight

        # Mamba layers:
        model.layers.{i}.mamba.in_proj.weight
        model.layers.{i}.mamba.conv1d.weight        [intermediate, 1, K]
        model.layers.{i}.mamba.conv1d.bias          [intermediate]
        model.layers.{i}.mamba.x_proj.weight
        model.layers.{i}.mamba.dt_proj.weight
        model.layers.{i}.mamba.dt_proj.bias
        model.layers.{i}.mamba.A_log
        model.layers.{i}.mamba.D
        model.layers.{i}.mamba.out_proj.weight
        model.layers.{i}.mamba.dt_layernorm.weight
        model.layers.{i}.mamba.b_layernorm.weight
        model.layers.{i}.mamba.c_layernorm.weight

        # FFN (MLP layers):
        model.layers.{i}.feed_forward.{gate_proj,up_proj,down_proj}.weight

        # FFN (MoE layers):
        model.layers.{i}.feed_forward.router.weight
        model.layers.{i}.feed_forward.experts.{e}.{gate_proj,up_proj,down_proj}.weight

        model.final_layernorm.weight
        lm_head.weight
    """

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config
        self.model = JambaModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits.float()

    # ------------------------------------------------------------------
    # Weight loading: read the HF safetensors and dispatch to per-param
    # weight_loader callbacks (used for the MoE expert pack-into-3D-tensor
    # and the Mamba A_log -> A negative-exp transform).  Mirrors the
    # ``Pattern 2`` per-pipeline loader documented in CLAUDE.md, since
    # Jamba's hybrid wiring needs custom routing.
    # ------------------------------------------------------------------
    # vLLM-style fused-projection mapping: HF stores Q, K, V and gate, up
    # as separate weights but our L2 modules use fused projections
    # (``QKVParallelLinear`` for attention, ``MergedColumnParallelLinear``
    # for the dense MLP).  The loader translates each HF ``.q_proj`` /
    # ``.k_proj`` / ``.v_proj`` / ``.gate_proj`` / ``.up_proj`` tensor
    # into the appropriate shard of the fused parameter via the
    # corresponding ``_weight_loader(param, tensor, shard_id)``.
    # Mirrors :class:`LlamaForCausalLM.packed_modules_mapping`.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def load_weights(self, model_path: str) -> int:
        """Load Jamba weights from a local snapshot directory.

        Returns the number of tensors copied into model parameters.
        """
        import re
        from glob import glob
        from safetensors import safe_open

        config = self.config

        # Build a name -> Parameter map for direct keys.  MoE expert
        # weights need pattern matching, so handle them in the loop.
        params = dict(self.named_parameters())

        moe_re = re.compile(
            r"^model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.weight$"
        )
        # Pattern for HF self_attn projections that need to be packed
        # into the fused qkv_proj.  Captured groups: (layer_idx, proj).
        qkv_re = re.compile(
            r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.weight$"
        )
        # Pattern for HF dense-MLP gate/up projections (only present in
        # layers where layers_num_experts == 1 -- the MoE layers use the
        # ``feed_forward.experts.*`` path which moe_re catches above).
        # Packs into the fused ``gate_up_proj`` via shard_id 0/1.
        mlp_re = re.compile(
            r"^model\.layers\.(\d+)\.feed_forward\.(gate_proj|up_proj)\.weight$"
        )

        sf_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors in {model_path}")

        loaded = 0
        unmatched: list[str] = []
        for sf in sf_files:
            with safe_open(sf, "pt", "cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)

                    # MoE expert: pack into 3D tensor at (expert_idx, ...).
                    m = moe_re.match(name)
                    if m is not None:
                        layer_idx = int(m.group(1))
                        expert_idx = int(m.group(2))
                        proj = m.group(3)
                        ff_path = f"model.layers.{layer_idx}.feed_forward"
                        if proj == "gate_proj":
                            param = params[f"{ff_path}.w13"]
                            param.weight_loader(param, tensor, expert_idx, True)
                        elif proj == "up_proj":
                            param = params[f"{ff_path}.w13"]
                            param.weight_loader(param, tensor, expert_idx, False)
                        else:
                            param = params[f"{ff_path}.w2"]
                            param.weight_loader(param, tensor, expert_idx)
                        loaded += 1
                        continue

                    # Fused QKV: pack q_proj/k_proj/v_proj into qkv_proj
                    # via QKVParallelLinear._weight_loader(param, tensor, shard_id).
                    qm = qkv_re.match(name)
                    if qm is not None:
                        layer_idx = int(qm.group(1))
                        proj = qm.group(2)  # q_proj / k_proj / v_proj
                        shard_id = proj[0]  # "q" / "k" / "v"
                        param = params[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"]
                        param.weight_loader(param, tensor, shard_id)
                        loaded += 1
                        continue

                    # Fused gate_up: pack gate_proj/up_proj into gate_up_proj
                    # for dense-MLP layers (layers_num_experts == 1).
                    mm = mlp_re.match(name)
                    if mm is not None:
                        layer_idx = int(mm.group(1))
                        proj = mm.group(2)  # gate_proj / up_proj
                        shard_id = 0 if proj == "gate_proj" else 1
                        param = params[
                            f"model.layers.{layer_idx}.feed_forward.gate_up_proj.weight"
                        ]
                        param.weight_loader(param, tensor, shard_id)
                        loaded += 1
                        continue

                    # Generic path: name remap + direct copy / weight_loader.
                    mapped = self._remap_name(name)
                    if mapped is None or mapped not in params:
                        unmatched.append(name)
                        continue
                    param = params[mapped]
                    if hasattr(param, "weight_loader"):
                        param.weight_loader(param, tensor)
                    else:
                        if tensor.shape != param.data.shape:
                            raise RuntimeError(
                                f"shape mismatch loading {name} "
                                f"(checkpoint {tuple(tensor.shape)} vs "
                                f"model {tuple(param.data.shape)})"
                            )
                        param.data.copy_(tensor)
                    loaded += 1

        if unmatched:
            # Be loud about anything we silently dropped -- helps catch
            # a checkpoint vs config mismatch (e.g. expert count).
            print(f"  [JambaForCausalLM] WARNING: {len(unmatched)} unmatched "
                  f"weight names; first few: {unmatched[:5]}")
        return loaded

    @staticmethod
    def _remap_name(name: str) -> str | None:
        """HF checkpoint -> our param path.  Returns None if MoE expert
        / fused-QKV (handled separately) or a name we should silently skip."""
        # Embedding wrapper: HF stores at model.embed_tokens.weight,
        # our L1 ``Embedding`` nests an nn.Embedding as ``self.emb``.
        if name == "model.embed_tokens.weight":
            return "model.embed_tokens.emb.weight"
        if name == "lm_head.weight":
            return "lm_head.weight"
        if name == "model.final_layernorm.weight":
            return "model.final_layernorm.weight"

        # Mamba-mixer: A_log -> A (param has its own weight_loader to
        # apply -exp() at load time), conv1d.weight -> conv1d_weight,
        # conv1d.bias -> conv1d_bias.
        if "mamba.A_log" in name:
            return name.replace("mamba.A_log", "mamba.A")
        if "mamba.conv1d.weight" in name:
            return name.replace("mamba.conv1d.weight", "mamba.conv1d_weight")
        if "mamba.conv1d.bias" in name:
            return name.replace("mamba.conv1d.bias", "mamba.conv1d_bias")

        # Everything else is a 1-to-1 mapping (input_layernorm,
        # pre_ff_layernorm, o_proj, layernorms, etc.).
        return name
