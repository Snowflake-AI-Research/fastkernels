"""MiniMax's normalized residual blocks and constant-decay linear attention."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads as decoder_workloads
from fastkernels.hf_coverage.models.olmo2 import PostNormModel, decoder_config
from fastkernels.hf_coverage.patches.hunyuan_moe_precision import HunyuanFusedExperts
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L1.tensor_ops import Exp
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class LightningAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.num_attention_heads, config.head_dim
        width = self.heads * self.head_dim
        self.qkv_proj = Linear(config.hidden_size, 3 * width, False)
        self.out_proj = Linear(width, config.hidden_size, False)
        self.output_gate = Linear(config.hidden_size, width, False)
        self.norm = T5LayerNorm(width, eps=1e-6)
        self.activation, self.sigmoid, self.product = SiLU(), Sigmoid(), ProductGate()
        self.bmm, self.exp = BMM(), Exp()
        self.block_size = config.block_size
        self.state = None
        # Keep native rounded tables and block/state stores: GLA
        # evaluates log decay with different chunking and FP32 accumulation.
        block = config.block_size
        for name, shape in (("slope_rate", (self.heads, 1, 1)),
                            ("query_decay", (self.heads, block, 1)),
                            ("key_decay", (self.heads, block, 1)),
                            ("diagonal_decay", (1, self.heads, block, block))):
            self.register_buffer(name, torch.empty(shape))

    def _multiply(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        return self.product(torch.cat((left, right), dim=-1))

    def _attention(self, query, key, value):
        batch, length = query.shape[:2]
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        outputs = []
        state = self.state
        if state is None:
            state = value.new_zeros((batch, self.heads, self.head_dim, self.head_dim))
            for start in range(0, length, self.block_size):
                end = min(start + self.block_size, length)
                count = end - start
                q, k, v = query[:, :, start:end], key[:, :, start:end], value[:, :, start:end]
                query_decay = self.query_decay[:, :count]
                key_decay = self.key_decay[:, -count:]
                diagonal = self.diagonal_decay[:, :, :count, :count]
                block_decay = self.exp(-self.slope_rate * count)

                scores = self.bmm(q, k.transpose(-1, -2))
                intra = self.bmm(self._multiply(scores, diagonal), v)
                inter = self.bmm(self._multiply(q, query_decay), state)
                outputs.append(inter + intra)
                update = self.bmm(self._multiply(k, key_decay).transpose(-1, -2), v)
                state = self._multiply(state, block_decay) + update
        else:
            ratio = self.exp(-self.slope_rate)
            for index in range(length):
                q = query[:, :, index:index + 1]
                k = key[:, :, index:index + 1]
                v = value[:, :, index:index + 1]
                update = self.bmm(k.transpose(-1, -2), v)
                state = self._multiply(ratio, state) + update
                outputs.append(self.bmm(q, state))
        self.state = state
        return torch.cat(outputs, dim=-2).transpose(1, 2).reshape(batch * length, -1)

    def forward(self, positions, hidden_states):
        batch = self.batch_size
        length = hidden_states.shape[0] // batch
        qkv = self.activation(self.qkv_proj(hidden_states))
        q, k, v = qkv.view(batch, length, self.heads, 3 * self.head_dim).split(self.head_dim, -1)
        output = self.norm(self._attention(q, k, v))
        gate = self.sigmoid(self.output_gate(hidden_states))
        return self.out_proj(self.product(torch.cat((gate, output), -1)))


class MiniMaxLayer(nn.Module):
    def __init__(self, layer, config, kind):
        super().__init__()
        self.self_attn = LightningAttention(config) if kind == "linear_attention" else layer.self_attn
        self.input_layernorm = T5LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = T5LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SharedExpertMoE(hidden_size=config.hidden_size, num_experts=config.num_local_experts,
                                  top_k=config.num_experts_per_tok, moe_intermediate_size=config.intermediate_size,
                                  keep_router_weights_fp32=True)
        # Native grouped_mm rounds down projections before FP32 route weighting
        # and reduction. Select this carrier before loading unshuffled weights.
        self.mlp.use_trtllm = False
        self.mlp.fused_experts = HunyuanFusedExperts()
        stem = "linear_attn" if kind == "linear_attention" else "full_attn"
        self.attention_alpha, self.attention_beta = getattr(config, stem + "_alpha_factor"), getattr(config, stem + "_beta_factor")
        self.mlp_alpha, self.mlp_beta = config.mlp_alpha_factor, config.mlp_beta_factor

    def forward(self, positions, hidden_states, residual=None):
        hidden = self.input_layernorm(hidden_states)
        hidden = hidden * self.attention_alpha + self.self_attn(positions, hidden) * self.attention_beta
        hidden = self.post_attention_layernorm(hidden)
        return hidden * self.mlp_alpha + self.mlp(hidden) * self.mlp_beta, None


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.output_router_logits or config.tie_word_embeddings:
        raise ValueError("Selected MiniMax inference uses SiLU, no router outputs and an untied head")
    if set(config.layer_types) != {"linear_attention", "full_attention"}:
        raise ValueError("MiniMax case must preserve both attention types")
    model = LlamaForCausalLM(decoder_config(config, dtype))
    model.model.__class__ = PostNormModel
    model.model.layers = nn.ModuleList(MiniMaxLayer(layer, config, kind)
                                      for layer, kind in zip(model.model.layers, config.layer_types))
    model.model.norm = T5LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            direct[prefix + name + ".weight"] = getattr(layer, name).weight
        attn = layer.self_attn
        if isinstance(attn, LightningAttention):
            for name in ("qkv_proj", "out_proj", "output_gate", "norm"):
                direct[prefix + "self_attn." + name + ".weight"] = getattr(attn, name).weight
            for name in ("slope_rate", "query_decay", "key_decay", "diagonal_decay"):
                direct[prefix + "self_attn." + name] = getattr(attn, name)
        else:
            direct[prefix + "self_attn.o_proj.weight"] = attn.o_proj.weight
            for shard in ("q", "k", "v"):
                packed[prefix + "self_attn." + shard + "_proj.weight"] = (attn.qkv_proj.weight, shard)
        direct[prefix + "mlp.gate.weight"] = layer.mlp.gate.weight
        direct[prefix + "mlp.experts.gate_up_proj"] = layer.mlp.w13
        direct[prefix + "mlp.experts.down_proj"] = layer.mlp.w2
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"MiniMax state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"MiniMax weight mismatch: {name}: {parameter.shape}/{parameter.dtype}, {state_dict[name].shape}/{state_dict[name].dtype}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
    for layer in model.model.layers:
        layer.mlp.process_weights_after_loading()


def make_workloads(model, inputs, config, *, case=None):
    linear = [(index, layer.self_attn) for index, layer in enumerate(model.model.layers)
              if isinstance(layer.self_attn, LightningAttention)]
    full = [(index, layer.self_attn.attn) for index, layer in enumerate(model.model.layers)
            if not isinstance(layer.self_attn, LightningAttention)]
    for _, layer in linear:
        layer.batch_size = inputs["input_ids"].shape[0]
    workloads = decoder_workloads(model, inputs, config,
                                  attentions=[attention for _, attention in full], case=case)
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"

    def prepare(original):
        for _, layer in linear:
            layer.state = None
        if original is not None:
            original()

    def collect(output, original):
        output = original(output) if original is not None else output
        # The shared helper numbers only full-attention layers. Restore native
        # layer indices before adding each linear layer's carried K-transpose-V.
        remapped = {}
        for key, value in output.items():
            if key.startswith("past_key_values."):
                _, index, field = key.split(".", 2)
                key = f"past_key_values.{full[int(index)][0]}.{field}"
            remapped[key] = value
        for index, layer in linear:
            remapped[f"past_key_values.{index}.recurrent_states"] = layer.state
        return remapped

    return {name: Workload(run=work.run,
                           prepare=lambda work=work: prepare(work.prepare),
                           collect=(lambda output, work=work: collect(output, work.collect))
                                   if continuation else work.collect)
            for name, work in workloads.items()}
