"""ViLT image/text retrieval with position interpolation and native patch sampling."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from .blip import BlipAttention
from .vit import _Embeddings, _encoder_block
from ..runner import Workload


class ViltRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_embeddings = _Embeddings(config)
        self.position_grid = config.image_size // config.patch_size
        self.interpolate = Interpolate()
        self.text_embeddings = BertEmbeddings(config)
        self.token_type_embeddings = Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = _encoder_block(config)
            layer.attn = BlipAttention(config.hidden_size, config.num_attention_heads)
            layer.attn.attention.divide_scores = True
            self.layers.append(layer)
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.pooler = nn.ModuleDict({"dense": Linear(config.hidden_size, config.hidden_size), "activation": Tanh()})
        self.rank_output = Linear(config.hidden_size, 1)

    def forward(self, input_ids, pixel_values):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        text = self.text_embeddings(input_ids, positions)
        visual = self.visual_embeddings
        patches = visual.patch_embeddings.proj(pixel_values)
        spatial_positions = visual.position_embeddings[:, 1:].transpose(1, 2).reshape(
            1, patches.shape[1], self.position_grid, self.position_grid)
        spatial_positions = self.interpolate(spatial_positions, size=patches.shape[-2:],
                                             mode="bilinear", align_corners=True).flatten(2).transpose(1, 2)
        patches = patches.flatten(2).transpose(1, 2)
        # An omitted pixel mask marks the complete rectangular image valid.
        # Native sampling still permutes ALL patches, even during evaluation.
        patch_count = patches.shape[1]
        selections = torch.stack([torch.multinomial(torch.ones(patch_count, device="cpu").float(), patch_count)
                                  for _ in range(patches.shape[0])]).to(patches.device)
        batches = torch.arange(patches.shape[0], device=patches.device)[:, None]
        patches = patches[batches, selections]
        position = spatial_positions.expand(patches.shape[0], -1, -1)[batches, selections]
        images = torch.cat((visual.cls_token.expand(patches.shape[0], -1, -1), patches), dim=1)
        position = torch.cat((visual.position_embeddings[:, :1].expand(patches.shape[0], -1, -1), position), dim=1)
        images = images + position
        text = text + self.token_type_embeddings(torch.zeros(text.shape[:2], dtype=torch.long, device=text.device))
        images = images + self.token_type_embeddings(torch.ones(images.shape[:2], dtype=torch.long, device=images.device))
        hidden = torch.cat((text, images), dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.layernorm(hidden)
        pooler = self.pooler["activation"](self.pooler["dense"](hidden[:, 0]))
        return {"logits": self.rank_output(pooler)}


def build_from_config(config, device, dtype):
    if config.max_image_length != -1 or config.hidden_act != "gelu" or not config.qkv_bias:
        raise ValueError("The documented retrieval checkpoint retains all image patches and uses biased GELU blocks")
    return ViltRetrieval(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if not source.startswith("rank_output."):
            if source.startswith("token_type_embeddings."):
                source = "embeddings." + source
            for target, origin in (("visual_embeddings.", "embeddings."), ("text_embeddings.", "embeddings.text_embeddings."),
                                   ("layers.", "encoder.layer."), (".patch_embeddings.proj.", ".patch_embeddings.projection."),
                                   (".norm1.", ".layernorm_before."), (".norm2.", ".layernorm_after."),
                                   (".attn.proj.", ".attention.output.dense."), (".mlp.fc1.", ".intermediate.dense."),
                                   (".mlp.fc2.", ".output.dense.")):
                source = source.replace(target, origin)
            source = "vilt." + source
        if ".attn.qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".attn.qkv.", f".attention.attention.{part}."))
                                      for part in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped ViLT weights: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
