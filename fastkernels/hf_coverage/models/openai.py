"""Original GPT: post-normalization blocks, learned positions, and no KV cache."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation

from ..runner import Workload


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_head
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = Linear(config.n_embd, config.n_embd)
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden):
        batch, length, width = hidden.shape
        q, k, v = (x.reshape(batch, length, self.heads, width // self.heads)
                   .transpose(1, 2).reshape(batch * self.heads, length, -1)
                   for x in self.c_attn(hidden).chunk(3, dim=-1))
        scores = self.bmm(q, k.transpose(1, 2)) / (width // self.heads) ** 0.5
        # HF replaces future scores with finite -1e4, rather than adding -inf.
        # The mask depends only on positions, not on activation values.
        positions = torch.arange(length, device=hidden.device)
        scores = scores.masked_fill(positions[None, :] > positions[:, None], -1e4)
        context = self.bmm(self.softmax(scores), v)
        context = context.reshape(batch, self.heads, length, -1).transpose(1, 2)
        return self.c_proj(context.reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.ln_1 = LayerNorm(config.n_embd, config.layer_norm_epsilon, promote_fp32=False)
        self.ln_2 = LayerNorm(config.n_embd, config.layer_norm_epsilon, promote_fp32=False)
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd)
        self.act = NewGELUActivation()

    def forward(self, hidden):
        hidden = self.ln_1(hidden + self.attn(hidden))
        return self.ln_2(hidden + self.c_proj(self.act(self.c_fc(hidden))))


class OpenAIGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokens_embed = Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tokens_embed.emb.weight

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
        hidden = self.tokens_embed(input_ids) + self.positions_embed(positions)
        for layer in self.h:
            hidden = layer(hidden)
        return self.lm_head(hidden)


def build_from_config(config, device, dtype):
    if config.afn != "gelu" or not config.tie_word_embeddings:
        raise ValueError("The documented GPT checkpoint uses GELU and tied output weights")
    return OpenAIGPT(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict["transformer.tokens_embed.weight"], state_dict["lm_head.weight"]):
        raise ValueError("GPT's tied embedding and head weights disagree")
    mapped = {}
    for name, value in state_dict.items():
        target = name.removeprefix("transformer.").replace(".mlp.", ".")
        if target in ("tokens_embed.weight", "positions_embed.weight"):
            target = target.replace(".weight", ".emb.weight")
        if name.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            value = value.T.contiguous()
        mapped[target] = value
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(inputs["input_ids"])})}
