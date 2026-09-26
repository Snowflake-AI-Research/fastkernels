"""Native block-FP8 DeepSeek V3 with compressed MLA and explicit HF routing."""

from dataclasses import fields

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.models.qwen2_precision import ResidualRMSNorm
from fastkernels.hf_coverage.models.exaone4_5 import NativeGate
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.hf_coverage.patches.plain_scale_fp8 import PlainScaleFP8Linear, PlainScaleQuant, PlainScaleFP8Experts
from fastkernels.hf_coverage.patches.grouped_topk_normalization import GroupedTopKNormalization
from fastkernels.tasks.baseline.L1.fp8_linear import Fp8Linear
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L2.deepseek_moe import DeepSeekMoE
from fastkernels.tasks.baseline.L4.deepseek import DeepSeekV3Config, DeepSeekV3ForCausalLM


class NativeFP8Experts(DeepSeekMoE):
    """Compose the existing FP8 experts with HF's FP32 routing and weight scale."""
    def __init__(self, config, quant_config, *, epsilon=1e-20, gate_fp32=True):
        super().__init__(config, quant_config)
        self.gate_fp32 = gate_fp32
        self.gate_matmul = Matmul()
        self.grouped_topk = GroupedTopKNormalization(scoring_func="sigmoid", epsilon=epsilon,
                                                    scale=self.routed_scaling_factor)
        self.fused_experts = PlainScaleFP8Experts()

    def forward(self, hidden):
        if self.gate_fp32:
            scores = self.gate_matmul(hidden.float(), self.gate_weight.float())
        else:
            scores = self.gate_matmul(hidden.to(self.gate_weight.dtype), self.gate_weight).float()
        weights, indices = self.grouped_topk(scores, self.e_score_correction_bias,
                                            self.n_group, self.topk_group, self.top_k)
        output = self.fused_experts(hidden, self.w13, self.w2, weights, indices, self.num_experts,
                                   w13_scale=self.w13_weight_scale_inv, w2_scale=self.w2_weight_scale_inv,
                                   block_shape=[128, 128])
        return output + self.shared_expert(hidden) if self.shared_expert is not None else output


def prepare_linears(model):
    """Retain original FP8 weights/FP32 scales and select the existing GEMM core."""
    for module in model.modules():
        if isinstance(getattr(module, "linear_op", None), Fp8Linear):
            module.linear_op = PlainScaleFP8Linear()


def build_from_config(config, device, dtype):
    if config.quantization_config["quant_method"] != "fp8" or config.q_lora_rank is None:
        raise ValueError("Selected DeepSeek V3 uses block FP8 and low-rank queries")
    values = {f.name: getattr(config, f.name) for f in fields(DeepSeekV3Config) if hasattr(config, f.name)}
    values.update(dtype=dtype, rope_theta=config.rope_parameters["rope_theta"], scoring_func="sigmoid",
                  moe_router_dtype="float32", kv_cache_dtype="auto")
    carrier = DeepSeekV3Config(**values)
    model = DeepSeekV3ForCausalLM(carrier, quant_config=config.quantization_config)
    for i, layer in enumerate(model.model.layers):
        if i >= config.first_k_dense_replace:
            layer.mlp = NativeFP8Experts(carrier, config.quantization_config)
    for layer in model.model.layers:
        layer.self_attn = UnabsorbedMLA(layer.self_attn, config)
        layer.input_layernorm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
        layer.post_attention_layernorm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
        mlp = layer.mlp.shared_expert if isinstance(layer.mlp, NativeFP8Experts) else layer.mlp
        if mlp is not None:
            mlp.act_fn = NativeGate()
    model.model.norm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
    # FP8 weights and FP32 block scales must retain their native dtypes.
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
        a = layer.self_attn
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            direct[p + "self_attn." + name + ".weight"] = getattr(a, name).weight
        for name in ("q_b_proj", "kv_b_proj", "o_proj"):
            for field in ("weight", "weight_scale_inv"):
                direct[p + "self_attn." + name + "." + field] = getattr(getattr(a, name), field)
        for shard, name in enumerate(("q_a_proj", "kv_a_proj_with_mqa")):
            for field in ("weight", "weight_scale_inv"):
                packed[p + "self_attn." + name + "." + field] = (getattr(a.fused_qkv_a_proj, field), shard)
        if i < config.first_k_dense_replace:
            target, stem = layer.mlp, p + "mlp."
        else:
            e = layer.mlp
            direct[p + "mlp.gate.weight"] = e.gate_weight
            direct[p + "mlp.gate.e_score_correction_bias"] = e.e_score_correction_bias
            for source, dest in (("gate_up_proj", "w13"), ("down_proj", "w2")):
                direct[p + "mlp.experts." + source] = getattr(e, dest)
                direct[p + "mlp.experts." + source + "_scale_inv"] = getattr(e, dest + "_weight_scale_inv")
            target, stem = e.shared_expert, p + "mlp.shared_experts."
        for field in ("weight", "weight_scale_inv"):
            direct[stem + "down_proj." + field] = getattr(target.down_proj, field)
            for shard, name in enumerate(("gate_proj", "up_proj")):
                packed[stem + name + "." + field] = (getattr(target.gate_up_proj, field), shard)
    load_mapped_state(state_dict, direct, packed)
    prepare_linears(model)


def load_mapped_state(state_dict, direct, packed):
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"FP8 state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"FP8 weight shape mismatch: {name}")
        parameter.data = state_dict[name].to(parameter.device).clone()
    for name, (parameter, shard) in packed.items():
        if parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"Packed FP8 dtype mismatch: {name}")
        parameter.weight_loader(parameter, state_dict[name], shard)


class UnabsorbedMLA(torch.nn.Module):
    """Reuse projection/norm/rotary/attention ops with native compressed storage.

    The checkpoint's FP8 kv_b projection executes on each compressed prefix.
    Absorbing this matrix into the attention changes its dynamic quantization
    points. Keeping the original contraction order preserves that computation.
    """
    def __init__(self, source, config):
        super().__init__()
        for name in ("fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj", "rotary_emb"):
            setattr(self,name,getattr(source,name))
        for name in ("q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "qk_head_dim", "v_head_dim", "num_heads", "scaling"):
            setattr(self,name,getattr(source,name))
        self.q_a_layernorm=RMSNormNative(self.q_lora_rank,config.rms_norm_eps)
        self.kv_a_layernorm=RMSNormNative(self.kv_lora_rank,config.rms_norm_eps)
        self.attention=DenseAttention(backend="sdpa");self.interleave=config.rope_interleave
        self.latent=self.rotated=None;self.batch=1

    def forward(self,positions,hidden):
        total=hidden.shape[0];b=self.batch;seq=total//b
        packed=self.fused_qkv_a_proj(hidden)
        qc,kvc=packed.split([self.q_lora_rank,self.kv_lora_rank+self.qk_rope_head_dim],-1)
        q=self.q_b_proj(self.q_a_layernorm(qc)).view(total,self.num_heads,self.qk_head_dim)
        qpass,qrot=q.split([self.qk_nope_head_dim,self.qk_rope_head_dim],-1)
        latent,krot=kvc.split([self.kv_lora_rank,self.qk_rope_head_dim],-1)
        latent=self.kv_a_layernorm(latent).reshape(b,seq,self.kv_lora_rank)
        krot=krot[:,None,:]
        if self.interleave:
            qrot=torch.cat((qrot[...,0::2],qrot[...,1::2]),-1)
            krot=torch.cat((krot[...,0::2],krot[...,1::2]),-1)
        qrot,krot=RotaryEmbedding.forward_native(positions,qrot,krot,self.qk_rope_head_dim,self.rotary_emb.cos_sin_cache.to(hidden.dtype))
        krot=krot.reshape(b,seq,1,self.qk_rope_head_dim)
        self.latent=latent if self.latent is None else torch.cat((self.latent,latent),1)
        self.rotated=krot if self.rotated is None else torch.cat((self.rotated,krot),1)
        n=self.latent.shape[1]
        kv=self.kv_b_proj(self.latent.reshape(b*n,self.kv_lora_rank)).view(b,n,self.num_heads,self.qk_nope_head_dim+self.v_head_dim)
        key,value=kv.split([self.qk_nope_head_dim,self.v_head_dim],-1)
        key=torch.cat((key,self.rotated.expand(b,n,self.num_heads,self.qk_rope_head_dim)),-1)
        query=torch.cat((qpass,qrot),-1).reshape(b,seq,self.num_heads,self.qk_head_dim)
        # Native concatenation materializes contiguous B,H,S,D Q/K. Preserve
        # that attention layout before the following dynamic FP8 quantizer.
        query=query.transpose(1,2).contiguous().transpose(1,2)
        key=key.transpose(1,2).contiguous().transpose(1,2)
        # Baseline imports may disable cuDNN globally. Restore native SDPA
        # availability only inside this call; keep all native backends eligible.
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            result=self.attention(query,key,value,softmax_scale=self.scaling,causal=seq>1)
        return self.o_proj(result.reshape(total,self.num_heads*self.v_head_dim))


def make_workloads(model,inputs,config,*,case=None):
    ids=inputs["input_ids"];b,length=ids.shape
    continuation=case is not None and case.get("workload")=="causal_lm_continuation"
    steps=2 if continuation else 1;prefix=length-steps
    def reset():
        for l in model.model.layers:l.self_attn.latent=l.self_attn.rotated=None;l.self_attn.batch=b
    def call(start,end):
        pos=torch.arange(start,end,device=ids.device).repeat(b)
        logits=model.lm_head(model(ids[:,start:end].reshape(-1),pos)).reshape(b,end-start,-1)
        result={"logits":logits}
        if continuation:
            for i,l in enumerate(model.model.layers):
                result[f"past_key_values.{i}.key"]=l.self_attn.latent[:,None]
                result[f"past_key_values.{i}.value"]=l.self_attn.rotated.transpose(1,2)
        return result
    def initial():reset();return call(0,prefix)
    def prepare(i):
        initial()
        for j in range(i):call(prefix+j,prefix+j+1)
    return {"prefill":Workload(run=initial),**{(f"decode_{i+1}" if continuation else "decode"):Workload(run=lambda i=i:call(prefix+i,prefix+i+1),prepare=lambda i=i:prepare(i)) for i in range(steps)}}
