"""FNet masked language modeling using the adapted existing STFT capability."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput

from ..patches.fourier import RealFourier2D
from ..runner import Workload
from .bert import MaskedLMHead


class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fourier = RealFourier2D(config.max_position_embeddings, config.hidden_size)
        self.fourier_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.intermediate = EncoderIntermediate(config)
        self.intermediate.intermediate_act_fn = GELU(approximate="tanh")
        self.output = EncoderOutput(config)

    def forward(self, hidden_states):
        hidden_states = self.fourier_norm(hidden_states + self.fourier(hidden_states))
        return self.output(self.intermediate(hidden_states), hidden_states)


class FNetForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.embeddings.word_embeddings = Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.projection = Linear(config.hidden_size, config.hidden_size)
        self.layers = nn.ModuleList(FNetLayer(config) for _ in range(config.num_hidden_layers))
        self.pooler = nn.Sequential(Linear(config.hidden_size, config.hidden_size), Tanh())
        self.lm_head = MaskedLMHead(config)
        self.lm_head.activation = GELU(approximate="tanh")
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def backbone(self, input_ids):
        positions = self.embeddings.position_ids[:, :input_ids.shape[1]]
        hidden_states = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        hidden_states = self.projection(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # HF's masked-LM model executes this pooler even though its final
        # MaskedLMOutput discards the pooled tensor. Keep that execution.
        return hidden_states, self.pooler(hidden_states[:, 0])

    def forward(self, input_ids):
        if self.training:
            raise RuntimeError("FNet coverage supports inference only")
        hidden_states, _ = self.backbone(input_ids)
        return {"logits": self.lm_head(hidden_states)}


def build_from_config(config, device, dtype):
    if dtype != torch.float32:
        raise ValueError("This case uses FNet's native FP32 FFT execution; lower precision is not silently promoted")
    if (config.use_tpu_fourier_optimizations or config.hidden_act != "gelu_new"
            or config.chunk_size_feed_forward or config.output_hidden_states):
        raise ValueError("This case requires the documented non-TPU Fourier masked-LM computation")
    return FNetForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {}

    def copy(destination, source):
        mapped[destination] = remaining.pop(source)

    for name in model.embeddings.state_dict():
        copy("embeddings." + name, "fnet.embeddings." + name.replace(".emb.weight", ".weight"))
    for field in ("weight", "bias"):
        copy(f"projection.{field}", f"fnet.embeddings.projection.{field}")
        copy(f"pooler.0.{field}", f"fnet.pooler.dense.{field}")
        copy(f"lm_head.dense.{field}", f"cls.predictions.transform.dense.{field}")
        copy(f"lm_head.LayerNorm.{field}", f"cls.predictions.transform.LayerNorm.{field}")
        copy(f"lm_head.decoder.{field}", f"cls.predictions.decoder.{field}")
    bias_alias = remaining.pop("cls.predictions.bias")
    if not torch.equal(bias_alias, mapped["lm_head.decoder.bias"]):
        raise ValueError("HF's tied prediction-bias values disagree")
    if config.tie_word_embeddings and not torch.equal(
        mapped["lm_head.decoder.weight"], mapped["embeddings.word_embeddings.emb.weight"]
    ):
        raise ValueError("HF's tied word-embedding values disagree")
    for index in range(len(model.layers)):
        for target, source in (
            ("fourier_norm", "fourier.output.LayerNorm"),
            ("intermediate.dense", "intermediate.dense"),
            ("output.dense", "output.dense"),
            ("output.LayerNorm", "output.LayerNorm"),
        ):
            for field in ("weight", "bias"):
                copy(f"layers.{index}.{target}.{field}", f"fnet.encoder.layer.{index}.{source}.{field}")
    if remaining:
        raise KeyError(f"Unmapped FNet state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    if set(inputs) != {"input_ids"} or inputs["input_ids"].shape[1] != config.max_position_embeddings:
        raise ValueError("This case uses ordinary input_ids at the documented full sequence length")
    return {"forward": Workload(run=lambda: model(inputs["input_ids"]))}
