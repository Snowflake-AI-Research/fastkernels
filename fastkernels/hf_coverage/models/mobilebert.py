"""MobileBertForMaskedLM with trigram embeddings and four bottleneck FFNs."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.frozen_batch_norm2d import FrozenBatchNorm2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput

from ..patches.mobilebert_projection import PostBiasMatmul
from ..runner import Workload


class NoNorm(nn.Module):
    """Use frozen unit moments for HF's learned elementwise scale and bias."""

    def __init__(self, width):
        super().__init__()
        self.affine = FrozenBatchNorm2d(width, eps=0.0)
        self.affine.weight = nn.Parameter(self.affine.weight)
        self.affine.bias = nn.Parameter(self.affine.bias)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        return self.affine(hidden_states.reshape(-1, shape[-1], 1, 1)).reshape(shape)


def intermediate(config, input_size):
    block_config = SimpleNamespace(**(dict(config) | {"hidden_size": input_size}))
    block = EncoderIntermediate(block_config)
    block.intermediate_act_fn = ReLU()
    return block


def output(config, input_size, output_size):
    block_config = SimpleNamespace(**(dict(config) | {
        "intermediate_size": input_size, "hidden_size": output_size,
    }))
    block = EncoderOutput(block_config)
    block.LayerNorm = NoNorm(output_size)
    return block


class BottleneckLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NoNorm(config.intra_bottleneck_size)

    def forward(self, hidden_states):
        return self.LayerNorm(self.dense(hidden_states))


class MobileBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.embedding_size, config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_transformation = Linear(3 * config.embedding_size, config.hidden_size)
        self.LayerNorm = NoNorm(config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings)[None], persistent=False)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        padding = torch.zeros_like(embeddings[:, :1])
        trigrams = torch.cat([
            torch.cat([embeddings[:, 1:], padding], dim=1), embeddings,
            torch.cat([padding, embeddings[:, :-1]], dim=1),
        ], dim=-1)
        embeddings = self.embedding_transformation(trigrams)
        positions = self.position_embeddings(self.position_ids[:, :input_ids.shape[1]])
        token_types = self.token_type_embeddings(torch.zeros_like(input_ids))
        return self.LayerNorm(embeddings + positions + token_types)


class MobileBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.true_hidden_size // self.num_heads
        self.query = Linear(config.true_hidden_size, config.true_hidden_size)
        self.key = Linear(config.true_hidden_size, config.true_hidden_size)
        self.value = Linear(config.hidden_size, config.true_hidden_size)
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, shared_input, hidden_states):
        batch, length = hidden_states.shape[:2]
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.query(shared_input).view(shape)
        key = self.key(shared_input).view(shape)
        value = self.value(hidden_states).view(shape)
        return self.attention(query, key, value).reshape(batch, length, -1)


class MobileBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.true_hidden_size
        self.bottleneck = nn.ModuleDict({"input": BottleneckLayer(config), "attention": BottleneckLayer(config)})
        self.attention = nn.ModuleDict({
            "self": MobileBertSelfAttention(config), "output": output(config, width, width),
        })
        self.ffn = nn.ModuleList([
            nn.ModuleDict({"intermediate": intermediate(config, width),
                           "output": output(config, config.intermediate_size, width)})
            for _ in range(config.num_feedforward_networks - 1)
        ])
        self.intermediate = intermediate(config, width)
        self.output = output(config, config.intermediate_size, width)
        self.output.bottleneck = output(config, width, config.hidden_size)

    def forward(self, hidden_states):
        layer_input = self.bottleneck["input"](hidden_states)
        shared_input = self.bottleneck["attention"](hidden_states)
        attention_output = self.attention["self"](shared_input, hidden_states)
        attention_output = self.attention["output"](attention_output, layer_input)
        for block in self.ffn:
            attention_output = block["output"](block["intermediate"](attention_output), attention_output)
        layer_output = self.output(self.intermediate(attention_output), attention_output)
        return self.output.bottleneck(layer_output, hidden_states)


class MobileBertPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = nn.ModuleDict({
            "dense": Linear(config.hidden_size, config.hidden_size),
            "activation": ReLU(),
            "LayerNorm": LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False),
        })
        self.dense = Linear(config.vocab_size, config.hidden_size - config.embedding_size, bias=False)
        self.decoder = Linear(config.embedding_size, config.vocab_size)
        self.projection = PostBiasMatmul()

    def forward(self, hidden_states):
        hidden_states = self.transform["dense"](hidden_states)
        hidden_states = self.transform["LayerNorm"](self.transform["activation"](hidden_states))
        weight = torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0).t()
        return self.projection(hidden_states, weight, self.decoder.bias)


class MobileBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = MobileBertEmbeddings(config)
        self.layers = nn.ModuleList([MobileBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = MobileBertPredictionHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (not config.trigram_input or not config.use_bottleneck or config.use_bottleneck_attention
            or not config.key_query_shared_bottleneck or config.normalization_type != "no_norm"
            or config.hidden_act != "relu" or config.num_feedforward_networks != 4
            or config.embedding_size >= config.hidden_size
            or config.true_hidden_size != config.intra_bottleneck_size):
        raise ValueError("MobileBERT coverage preserves the mobilebert-uncased bottleneck computation")
    return MobileBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight").replace(".affine.", ".")
        if source.startswith("layers."):
            source = source.replace("layers.", "mobilebert.encoder.layer.", 1)
        elif source.startswith("lm_head."):
            source = source.replace("lm_head.", "cls.predictions.", 1)
        else:
            source = "mobilebert." + source
        parameter.copy_(state_dict[source])


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(**inputs)})}
