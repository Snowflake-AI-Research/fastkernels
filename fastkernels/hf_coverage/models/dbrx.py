"""DBRX full decoder from existing normalization, GQA, router and GLU ops.

Routing uses GroupedTopK's unchanged no-bias one-group path, including its
low-precision softmax and normalization boundaries. Expert gather/scatter and
contiguous cached K/V are connecting code. The expert projection orientation
matches the repaired upstream implementation, with no native model imports.
"""
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.grouped_topk import GroupedTopK
from fastkernels.tasks.baseline.L1.silu import SiLU
from ..patches.product_gate import ProductGate
from ..patches.dbrx_clip import DbrxClip
from ..runner import Workload


class Attention(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.heads,self.kv,self.dim=c.n_heads,c.attn_config.kv_n_heads,c.d_model//c.n_heads
        self.Wqkv=Linear(c.d_model,c.d_model+2*self.kv*self.dim,bias=False)
        self.out_proj=Linear(c.d_model,c.d_model,bias=False)
        self.clip=DbrxClip(c.attn_config.clip_qkv)
        self.attention=DenseAttention(backend='sdpa')
        self.key=self.value=None

    def forward(self,x,positions,rotary):
        b,s,_=x.shape
        q,k,v=self.clip(self.Wqkv(x)).split([self.heads*self.dim,self.kv*self.dim,self.kv*self.dim],-1)
        q,k=rotary.forward_native(positions.repeat(b),q.reshape(b*s,-1),k.reshape(b*s,-1),self.dim,rotary.cos_sin_cache.to(x.dtype))
        q=q.reshape(b,s,self.heads,self.dim);k=k.reshape(b,s,self.kv,self.dim);v=v.reshape(b,s,self.kv,self.dim)
        fresh=self.key is None
        self.key=k if fresh else torch.cat((self.key,k),1)
        self.value=v if fresh else torch.cat((self.value,v),1)
        k=self.key.repeat_interleave(self.heads//self.kv,2);v=self.value.repeat_interleave(self.heads//self.kv,2)
        # Restore native SDPA availability locally: transitive vLLM imports
        # disable cuDNN process-wide in candidate workers.
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION,SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
            state=self.attention(q,k,v,causal=fresh and s>1)
        return self.out_proj(state.reshape_as(x))


class Experts(nn.Module):
    def __init__(self,c):
        super().__init__();f=c.ffn_config
        if f.moe_normalize_expert_weights!=1 or f.ffn_act_fn.get('name','silu')!='silu':
            raise ValueError('Selected DBRX uses SiLU experts and L1 route normalization')
        self.hidden,self.inner,self.count,self.topk=c.d_model,f.ffn_hidden_size,f.moe_num_experts,f.moe_top_k
        self.router=nn.Module();self.router.layer=Linear(c.d_model,self.count,bias=False)
        self.experts=nn.Module();self.experts.mlp=nn.Module()
        for name in ('w1','v1','w2'):setattr(self.experts.mlp,name,nn.Parameter(torch.empty(self.count*self.inner,self.hidden)))
        self.route=GroupedTopK(scoring_func='softmax',renormalize=True,force_sorted=True)
        self.linear,self.activation,self.product=Matmul(),SiLU(),ProductGate()
        self.last_routes=None

    def forward(self,x):
        shape=x.shape;x=x.reshape(-1,self.hidden)
        weights,ids=self.route(self.router.layer(x),None,1,1,self.topk)
        weights=weights.to(x.dtype);self.last_routes=ids
        out=torch.zeros_like(x)
        for expert in range(self.count):
            tokens,slots=torch.where(ids==expert)
            if tokens.numel()==0:continue
            w1,v1,w2=(getattr(self.experts.mlp,n).view(self.count,self.inner,self.hidden)[expert] for n in ('w1','v1','w2'))
            gate=self.activation(self.linear(x[tokens],w1));up=self.linear(x[tokens],v1)
            state=self.linear(self.product(torch.cat((gate,up),-1)),w2.T)
            scales=weights[tokens,slots,None].expand_as(state)
            state=self.product(torch.cat((state,scales),-1))
            out.index_add_(0,tokens,state)
        return out.reshape(shape)


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.norm_attn_norm=nn.Module();n=self.norm_attn_norm
        n.norm_1=LayerNorm(c.d_model,create_offset=False,promote_fp32=False)
        n.norm_2=LayerNorm(c.d_model,create_offset=False,promote_fp32=False)
        n.attn=Attention(c);self.ffn=Experts(c)

    def forward(self,x,positions,rotary):
        n=self.norm_attn_norm;x=x+n.attn(n.norm_1(x),positions,rotary)
        return x+self.ffn(n.norm_2(x))


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.transformer=nn.Module();m=self.transformer
        m.wte=Embedding(c.vocab_size,c.d_model);m.blocks=nn.ModuleList(Block(c) for _ in range(c.n_layers))
        m.norm_f=LayerNorm(c.d_model,create_offset=False,promote_fp32=False)
        self.lm_head=Linear(c.d_model,c.vocab_size,bias=False)
        self.rotary=RotaryEmbedding(c.d_model//c.n_heads,c.max_seq_len,c.rope_parameters['rope_theta'])

    def reset(self):
        for l in self.transformer.blocks:l.norm_attn_norm.attn.key=l.norm_attn_norm.attn.value=None

    def forward(self,ids,positions):
        h=self.transformer.wte(ids)
        for l in self.transformer.blocks:h=l(h,positions,self.rotary)
        return self.lm_head(self.transformer.norm_f(h))

    def outputs(self,logits):
        out={'logits':logits}
        for i,l in enumerate(self.transformer.blocks):
            out[f'past_key_values.{i}.key']=l.norm_attn_norm.attn.key.transpose(1,2)
            out[f'past_key_values.{i}.value']=l.norm_attn_norm.attn.value.transpose(1,2)
        return out


def build_from_config(config,device,dtype):
    model=Model(config).to(device=device,dtype=dtype).eval()
    # Position-only constants: native HF builds frequencies on CPU and evaluates
    # angles/cos/sin on the execution device. CPU trig rounding can change a
    # later BF16 MoE route when two expert scores are close.
    dim=config.d_model//config.n_heads
    inv=(1.0/(config.rope_parameters['rope_theta']**(torch.arange(0,dim,2,device="cpu",dtype=torch.float32)/dim))).to(device)
    angles=torch.arange(config.max_seq_len,device=device,dtype=torch.float32)[:,None]*inv[None,:]
    model.rotary.cos_sin_cache=torch.cat((angles.cos(),angles.sin()),-1).to(dtype)
    return model


def load_state_dict_into(model,state_dict,config):
    mapped={k.replace('transformer.wte.weight','transformer.wte.emb.weight'):v for k,v in state_dict.items()}
    model.load_state_dict(mapped,strict=True)


def make_workloads(model,inputs,config,*,case=None):
    ids=inputs['input_ids'];prefix=ids.shape[1]-2;positions=torch.arange(ids.shape[1],device=ids.device)
    def initial():model.reset();return model.outputs(model(ids[:,:prefix],positions[:prefix]))
    def prepare(i):
        initial()
        for j in range(i):model(ids[:,prefix+j:prefix+j+1],positions[prefix+j:prefix+j+1])
    def advance(i):return model.outputs(model(ids[:,prefix+i:prefix+i+1],positions[prefix+i:prefix+i+1]))
    return {'prefill':Workload(run=initial),**{f'decode_{i+1}':Workload(run=lambda i=i:advance(i),prepare=lambda i=i:prepare(i)) for i in range(2)}}
