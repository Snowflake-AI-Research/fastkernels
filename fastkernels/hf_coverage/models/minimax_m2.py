"""MiniMax M2 with native FP8 projections/experts and joint Q/K normalization."""

from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastkernels.hf_coverage.models.deepseek_v3 import NativeFP8Experts, load_mapped_state, prepare_linears
from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.models.olmo2 import JointNorm, decoder_config
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from .stablelm import NativePartialRotary
from .qwen2_precision import configure_language
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.hf_coverage.patches.grouped_dense_attention import GroupedDenseAttention


def build_from_config(config, device, dtype):
    if config.quantization_config["quant_method"] != "fp8" or config.output_router_logits:
        raise ValueError("Selected MiniMax M2 uses FP8 and does not return router outputs")
    model = LlamaForCausalLM(decoder_config(config, dtype))
    rotary_dim = config.get("rotary_dim", int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)))
    with torch.device("cpu"):
        model.model.rotary_emb = NativePartialRotary(config.head_dim, rotary_dim, config.max_position_embeddings,
                                                     config.rope_parameters["rope_theta"])
    expert_config = SimpleNamespace(
        hidden_size=config.hidden_size, n_routed_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok, moe_intermediate_size=config.intermediate_size,
        n_shared_experts=0, n_group=1, topk_group=1, scoring_func="sigmoid", topk_method="noaux_tc",
        norm_topk_prob=True, routed_scaling_factor=1.0,
    )
    for layer in model.model.layers:
        layer.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim,
            rotary_emb=model.model.rotary_emb, quant_config=config.quantization_config,
        )
        layer.self_attn.q_norm = JointNorm(config.num_attention_heads * config.head_dim, config.rms_norm_eps)
        layer.self_attn.k_norm = JointNorm(config.num_key_value_heads * config.head_dim, config.rms_norm_eps)
        layer.mlp = NativeFP8Experts(expert_config, config.quantization_config, epsilon=0.0, gate_fp32=False)
    configure_language(model.model, config)
    for layer in model.model.layers:
        layer.self_attn.q_norm.norm = RMSNormNative(config.num_attention_heads * config.head_dim, config.rms_norm_eps)
        layer.self_attn.k_norm.norm = RMSNormNative(config.num_key_value_heads * config.head_dim, config.rms_norm_eps)
        layer.self_attn = BatchedAttention(layer.self_attn)
    # Position-only constants: native computes frequencies on CPU and angles
    # on the execution device. Preserve those rounding boundaries.
    dims=torch.arange(0,rotary_dim,2,device="cpu",dtype=torch.float32)
    frequency=1.0/(config.rope_parameters["rope_theta"]**(dims/rotary_dim))
    angles=torch.outer(torch.arange(config.max_position_embeddings,device=device,dtype=torch.float32),frequency.to(device))
    model.model.rotary_emb.rotary.cos_sin_cache=torch.cat((angles.cos(),angles.sin()),-1)
    return model.to(device=device).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            direct[p + name + ".weight"] = getattr(layer, name).weight
        a, e = layer.self_attn, layer.mlp
        for name in ("q_norm", "k_norm"):
            direct[p + "self_attn." + name + ".weight"] = getattr(a, name).norm.weight
        for field in ("weight", "weight_scale_inv"):
            direct[p + "self_attn.o_proj." + field] = getattr(a.o_proj, field)
            for shard in ("q", "k", "v"):
                packed[p + "self_attn." + shard + "_proj." + field] = (getattr(a.qkv_proj, field), shard)
        direct[p + "mlp.gate.weight"] = e.gate_weight
        direct[p + "mlp.e_score_correction_bias"] = e.e_score_correction_bias
        for source, dest in (("gate_up_proj", "w13"), ("down_proj", "w2")):
            direct[p + "mlp.experts." + source] = getattr(e, dest)
            direct[p + "mlp.experts." + source + "_scale_inv"] = getattr(e, dest + "_weight_scale_inv")
    load_mapped_state(state_dict, direct, packed)
    prepare_linears(model)


class BatchedAttention(torch.nn.Module):
    """Retain the native batch and GQA storage around existing attention ops."""
    def __init__(self, source):
        super().__init__()
        for name in ("qkv_proj", "o_proj", "q_norm", "k_norm", "rotary_emb"):
            setattr(self,name,getattr(source,name))
        for name in ("num_heads", "num_kv_heads", "head_dim"):
            setattr(self,name,getattr(source,name))
        self.attention=GroupedDenseAttention(backend="sdpa");self.key=self.value=None;self.batch=1
    def forward(self,positions,hidden):
        total=hidden.shape[0];b=self.batch;seq=total//b;h=self.head_dim
        q,k,v=self.qkv_proj(hidden).split([self.num_heads*h,self.num_kv_heads*h,self.num_kv_heads*h],-1)
        q=self.q_norm(q);k=self.k_norm(k)
        q,k=self.rotary_emb(positions,q,k)
        q=q.reshape(b,seq,self.num_heads,h);k=k.reshape(b,seq,self.num_kv_heads,h);v=v.reshape(b,seq,self.num_kv_heads,h).contiguous()
        self.key=k if self.key is None else torch.cat((self.key,k),1)
        self.value=v if self.value is None else torch.cat((self.value,v),1)
        # Native partial-RoPE concatenation produces contiguous B,H,S,D.
        # Retain that layout through SDPA; FP8 output quantization can amplify
        # even one BF16 rounding difference from a different attention kernel.
        q=q.transpose(1,2).contiguous().transpose(1,2)
        # Baseline imports may disable cuDNN globally. Restore native SDPA
        # availability only inside this call; keep all native backends eligible.
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out=self.attention(q,self.key,self.value,causal=seq>1)
        return self.o_proj(out.reshape(total,self.num_heads*h))


def make_workloads(model,inputs,config,*,case=None):
    ids=inputs["input_ids"];b,length=ids.shape
    continuation=case is not None and case.get("workload")=="causal_lm_continuation"
    steps=2 if continuation else 1;prefix=length-steps
    def reset():
        for l in model.model.layers:l.self_attn.key=l.self_attn.value=None;l.self_attn.batch=b
    def call(start,end):
        positions=torch.arange(start,end,device=ids.device).repeat(b)
        logits=model.lm_head(model(ids[:,start:end].reshape(-1),positions)).reshape(b,end-start,-1)
        result={"logits":logits}
        if continuation:
            for i,l in enumerate(model.model.layers):
                result[f"past_key_values.{i}.key"]=l.self_attn.key.transpose(1,2)
                result[f"past_key_values.{i}.value"]=l.self_attn.value.transpose(1,2)
        return result
    def initial():reset();return call(0,prefix)
    def prepare(i):
        initial()
        for j in range(i):call(prefix+j,prefix+j+1)
    return {"prefill":Workload(run=initial),**{(f"decode_{i+1}" if continuation else "decode"):Workload(run=lambda i=i:call(prefix+i,prefix+i+1),prepare=lambda i=i:prepare(i)) for i in range(steps)}}
