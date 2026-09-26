"""Doge dense default path with its value-dependent attention mask.

Exp and LogSigmoid provide exp(A*softplus(dt)); ProductGate provides learned
products. Native top-k selection is extracted from the existing grouped router
because BF16 score ties make the selection backend part of the result.
"""
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.log_sigmoid import LogSigmoid
from fastkernels.tasks.baseline.L1.tensor_ops import Exp
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from ..patches.product_gate import ProductGate
from ..patches.doge_mask_topk import DogeMaskTopK
from .dbrx import make_workloads


def multiply(op,a,b):return op(torch.cat((a,b.expand_as(a)),-1))


class Attention(nn.Module):
    def __init__(self,c):
        super().__init__();self.heads=c.num_attention_heads;self.kv=c.num_key_value_heads;self.dim=c.hidden_size//c.num_attention_heads;self.keep=c.keep_window_size
        for n,h in [('q_proj',self.heads),('k_proj',self.kv),('v_proj',self.kv)]:setattr(self,n,Linear(c.hidden_size,h*self.dim,bias=c.attention_bias))
        self.o_proj=Linear(self.heads*self.dim,c.hidden_size,bias=c.attention_bias)
        self.dt_proj=Linear(self.kv*self.dim,self.kv,bias=c.attention_bias)
        self.A=nn.Parameter(torch.zeros(self.kv));self.q_norm=RMSNormNative(self.dim,c.rms_norm_eps);self.k_norm=RMSNormNative(self.dim,c.rms_norm_eps)
        self.attention=DenseAttention(backend='sdpa');self.product=ProductGate();self.exp=Exp();self.logsigmoid=LogSigmoid();self.topk=DogeMaskTopK();self.key=self.value=None

    def forward(self,x,positions,rotary):
        b,s,_=x.shape
        q=self.q_norm(self.q_proj(x).reshape(b,s,self.heads,self.dim));k=self.k_norm(self.k_proj(x).reshape(b,s,self.kv,self.dim));v=self.v_proj(x).reshape(b,s,self.kv,self.dim)
        q,k=rotary.forward_native(positions.repeat(b),q.reshape(b*s,-1),k.reshape(b*s,-1),self.dim,rotary.cos_sin_cache.to(x.dtype))
        q=q.reshape(b,s,self.heads,self.dim);k=k.reshape(b,s,self.kv,self.dim)
        self.key=k if self.key is None else torch.cat((self.key,k),1);self.value=v if self.value is None else torch.cat((self.value,v),1)
        dt=self.dt_proj(self.value.flatten(2));softplus=-self.logsigmoid(-dt)
        dynamic=self.exp(multiply(self.product,softplus,self.A)).transpose(1,2)
        n=self.key.shape[1];mask=dynamic[:,:,None,:].expand(b,self.kv,s,n).clone()
        causal=torch.arange(n,device=x.device)[None,:]>positions[:,None]
        mask=mask.masked_fill(causal[None,None],torch.finfo(x.dtype).min)
        if n>self.keep:
            _,idx=self.topk(mask,self.keep)
            selected=torch.zeros_like(mask,dtype=torch.bool).scatter(-1,idx,True)
            mask=mask.masked_fill(~selected,torch.finfo(x.dtype).min)
        group=self.heads//self.kv;k=self.key.repeat_interleave(group,2);v=self.value.repeat_interleave(group,2)
        output=self.attention(q,k,v,attn_mask=mask.repeat_interleave(group,1))
        return self.o_proj(output.reshape(b,s,-1))


class MLP(nn.Module):
    def __init__(self,c):
        super().__init__();self.gate_proj=Linear(c.hidden_size,c.intermediate_size,bias=c.mlp_bias);self.up_proj=Linear(c.hidden_size,c.intermediate_size,bias=c.mlp_bias);self.down_proj=Linear(c.intermediate_size,c.hidden_size,bias=c.mlp_bias)
    def forward(self,x):return self.down_proj(SiluAndMul.forward_native(torch.cat((self.gate_proj(x),self.up_proj(x)),-1)))


class Layer(nn.Module):
    def __init__(self,c):
        super().__init__();self.input_layernorm=RMSNormNative(c.hidden_size,c.rms_norm_eps);self.post_attention_layernorm=RMSNormNative(c.hidden_size,c.rms_norm_eps);self.self_attn=Attention(c);self.mlp=MLP(c);self.input_residual=nn.Parameter(torch.ones(c.hidden_size));self.post_attention_residual=nn.Parameter(torch.ones(c.hidden_size));self.product=ProductGate()
    def forward(self,x,p,r):
        x=multiply(self.product,x,self.input_residual)+self.self_attn(self.input_layernorm(x),p,r)
        return multiply(self.product,x,self.post_attention_residual)+self.mlp(self.post_attention_layernorm(x))


class Model(nn.Module):
    def __init__(self,c):
        super().__init__()
        if c.is_moe or c.hidden_act!='silu':raise ValueError('Selected Doge constructor path is dense SiLU')
        self.model=nn.Module();m=self.model;m.embed_tokens=Embedding(c.vocab_size,c.hidden_size);m.layers=nn.ModuleList(Layer(c) for _ in range(c.num_hidden_layers));m.norm=RMSNormNative(c.hidden_size,c.rms_norm_eps)
        self.theta=c.rope_parameters["rope_theta"];self.lm_head=Linear(c.hidden_size,c.vocab_size,bias=False);self.rotary=RotaryEmbedding(c.hidden_size//c.num_attention_heads,c.max_position_embeddings,c.rope_parameters['rope_theta'])
    def reset(self):
        for l in self.model.layers:l.self_attn.key=l.self_attn.value=None
    def forward(self,ids,positions):
        h=self.model.embed_tokens(ids)
        # Default RoPE is valid beyond its initial cache length. Position-only
        # constants extend to the actual prefix, as native dynamic angles do.
        cached=self.model.layers[0].self_attn.key
        length=ids.shape[1]+(0 if cached is None else cached.shape[1])
        if length>self.rotary.cos_sin_cache.shape[0]:
            dim=self.rotary.head_dim
            freq=1.0/(self.theta**(torch.arange(0,dim,2,device="cpu",dtype=torch.float32)/dim))
            angles=torch.outer(torch.arange(length,device=ids.device,dtype=torch.float32),freq.to(ids.device))
            self.rotary.cos_sin_cache=torch.cat((angles.cos(),angles.sin()),-1).to(h.dtype)
        for l in self.model.layers:h=l(h,positions,self.rotary)
        return self.lm_head(self.model.norm(h))
    def outputs(self,logits):
        out={'logits':logits}
        for i,l in enumerate(self.model.layers):
            out[f'past_key_values.{i}.key']=l.self_attn.key.transpose(1,2);out[f'past_key_values.{i}.value']=l.self_attn.value.transpose(1,2)
        return out


def build_from_config(config,device,dtype):return Model(config).to(device=device,dtype=dtype).eval()
def load_state_dict_into(model,state_dict,config):model.load_state_dict({k.replace('model.embed_tokens.weight','model.embed_tokens.emb.weight'):v for k,v in state_dict.items()},strict=True)
