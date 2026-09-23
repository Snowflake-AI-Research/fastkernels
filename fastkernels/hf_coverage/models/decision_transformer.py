"""Decision Transformer trajectory embedding, causal blocks, and all regression heads."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock
from .bart import _fresh_cache
from .mvp import EagerAttention


class TrajectoryBlock(VitEncoderBlock):
    def __init__(self, config):
        super().__init__(config.hidden_size, config.n_head,
                         mlp_ratio=(config.n_inner or 4 * config.hidden_size) / config.hidden_size,
                         norm_eps=config.layer_norm_epsilon)
        self.mlp.act = ReLU()
        self.attention = EagerAttention(prescale_query=False)

    def forward(self, hidden):
        batch, length, width = hidden.shape
        packed = self.attn.qkv(self.norm1(hidden)).view(batch, length, 3, self.attn.num_heads, self.attn.head_dim)
        query, key, value = packed.unbind(2)
        key, value = cache = _fresh_cache(key.transpose(1, 2), value.transpose(1, 2))
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2), causal=True)
        hidden = hidden + self.attn.proj(context.reshape(batch, length, width))
        return hidden + self.mlp(self.norm2(hidden)), cache


class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.embed_timestep = Embedding(config.max_ep_len, width)
        self.embed_return, self.embed_state, self.embed_action = Linear(1, width), Linear(config.state_dim, width), Linear(config.act_dim, width)
        self.embed_ln = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.position_embedding = Embedding(config.n_positions, width)
        self.layers = nn.ModuleList([TrajectoryBlock(config) for _ in range(config.n_layer)])
        self.final_norm = LayerNorm(width, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.predict_state, self.predict_return = Linear(width, config.state_dim), Linear(width, 1)
        self.predict_action = nn.Sequential(Linear(width, config.act_dim), Tanh())

    def forward(self, states, actions, returns_to_go, timesteps):
        time = self.embed_timestep(timesteps)
        values = (self.embed_return(returns_to_go) + time, self.embed_state(states) + time,
                  self.embed_action(actions) + time)
        batch, length, width = values[0].shape
        hidden = self.embed_ln(torch.stack(values, dim=2).reshape(batch, 3 * length, width))
        hidden = hidden + self.position_embedding(torch.zeros(batch, 3 * length, device=states.device, dtype=torch.long))
        cache = []
        for layer in self.layers:
            hidden, state = layer(hidden)
            cache.append(state)
        hidden = self.final_norm(hidden)
        slots = hidden.view(batch, length, 3, width)
        return {"last_hidden_state": hidden, "state_preds": self.predict_state(slots[:, :, 2]),
                "return_preds": self.predict_return(slots[:, :, 2]), "action_preds": self.predict_action(slots[:, :, 1])}


def build_from_config(config, device, dtype):
    if (config.activation_function != "relu" or not config.action_tanh or not config.scale_attn_weights
            or config.scale_attn_by_inverse_layer_idx or config.reorder_and_upcast_attn or config.add_cross_attention):
        raise ValueError("Selected Decision Transformer requires ReLU, tanh actions and ordinary causal attention")
    return DecisionTransformer(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights, consumed = {}, set()
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        transpose = False
        if source.startswith("layers."):
            source = source.replace("layers.", "encoder.h.", 1)
            for old, new in (("norm1", "ln_1"), ("norm2", "ln_2"), ("attn.qkv", "attn.c_attn"),
                             ("attn.proj", "attn.c_proj"), ("mlp.fc1", "mlp.c_fc"), ("mlp.fc2", "mlp.c_proj")):
                source = source.replace(old, new)
            transpose = source.endswith("weight") and (".attn.c_" in source or ".mlp.c_" in source)
        source = source.replace("position_embedding.", "encoder.wpe.").replace("final_norm.", "encoder.ln_f.")
        weights[name] = state_dict[source].t().contiguous() if transpose else state_dict[source]
        consumed.add(source)
    # The native GPT2 token table is unused because the public trajectory model supplies inputs_embeds.
    if set(state_dict) - consumed != {"encoder.wte.weight"}:
        raise ValueError(f"Unexpected Decision Transformer unmapped state: {sorted(set(state_dict)-consumed)}")
    model.load_state_dict(weights)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
