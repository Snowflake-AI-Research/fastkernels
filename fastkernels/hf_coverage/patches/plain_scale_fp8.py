"""Reuse existing FP8 quantization and grouped-GEMM core with ordinary scales.

No weight requantization: HF FP8 checkpoints carry FP32 block scales whose
represented values must be preserved. Fp8Linear's existing group quantizer is adapted only at its scalar scale
formula to preserve HF's division and zero-row guard. MoeGroupedGemm's unchanged block-scaled GEMM
is invoked for one ordinary matrix: one weight view, one output per input row,
identity row indices and expert zero. There is no activation replication or
artificial diagonal/matrix expansion. Only bounded GEMM tile-padding metadata
is allocated. All quantization/GEMM/index setup belongs to execution.
"""
import torch
from torch import nn
import triton
import triton.language as tl
from fastkernels.tasks.baseline.L1.moe_grouped_gemm import MoeGroupedGemm


@triton.jit
def _ordinary_group_quant(x, out, scales, K:tl.constexpr, GROUP:tl.constexpr=128):
    # Parent: L1.fp8_linear._fp8_group_quant_kernel. Keep one program per
    # row/group, the same 128-value maximum reduction and output storage.
    # Native HF uses division by448 and floors the divisor *after* the max,
    # whereas the parent intentionally multiplies a reciprocal for vLLM.
    # That one-ULP scale difference can change subsequent FP8 bins.
    pid=tl.program_id(0);offsets=pid*GROUP+tl.arange(0,GROUP)
    values=tl.load(x+offsets).to(tl.float32)
    scale=tl.max(tl.abs(values),0)/448.0
    quantized=(values/tl.maximum(scale,1e-12)).to(out.dtype.element_ty)
    tl.store(out+offsets,quantized);tl.store(scales+pid,scale)


class PlainScaleQuant(nn.Module):
    def forward(self,x,output,scale):
        x=x.contiguous()
        if x.shape[-1]%128:raise ValueError("Ordinary FP8 requires full128-wide activation groups")
        _ordinary_group_quant[(x.numel()//128,)](x,output,scale,x.shape[-1])


class PlainScaleFP8Linear(nn.Module):
    def __init__(self):
        super().__init__();self.quant=PlainScaleQuant();self.gemm=MoeGroupedGemm()

    def forward(self,x,weight,weight_scale_inv,bias=None):
        n,k=weight.shape;a=x.reshape(-1,k).contiguous();m=a.shape[0]
        q=torch.empty_like(a,dtype=torch.float8_e4m3fn)
        scale=torch.empty(m,(k+127)//128,device=a.device,dtype=torch.float32)
        self.quant(a,q,scale)
        block=16;padded=((m+block-1)//block)*block
        rows=torch.arange(padded,device=a.device,dtype=torch.int32)
        expert=torch.zeros(padded//block,device=a.device,dtype=torch.int32)
        count=torch.full((1,),padded,device=a.device,dtype=torch.int32)
        out=torch.empty(m,n,device=a.device,dtype=x.dtype)
        self.gemm(q,weight.unsqueeze(0),out,None,rows,expert,count,False,1,
                  config={'BLOCK_SIZE_M':block,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':1,'num_warps':4,'num_stages':3},
                  a_scale=scale,b_scale=weight_scale_inv.unsqueeze(0),use_fp8_w8a8=True,block_shape=[128,128])
        if bias is not None:out=out+bias
        return out.reshape(*x.shape[:-1],n)


class PlainScaleFP8Experts(nn.Module):
    """Unchanged quantizer/GEMM/SiLU/product/reduction ops with explicit stores.

    Parent composition is VllmFusedExperts. Its fused activation and weighted
    down epilogue are replaced by the same existing operations separated at
    HF's BF16 store boundaries; routing weights narrow before multiplication.
    Routing/alignment and the two block-scaled GEMMs retain their algorithms.
    """
    def __init__(self):
        super().__init__()
        from fastkernels.tasks.baseline.L1.moe_align import MoeAlign
        from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
        from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
        from .product_gate import ProductGate
        self.quant=PlainScaleQuant();self.gemm=MoeGroupedGemm();self.align=MoeAlign()
        self.sum=MoeSum();self.gate=SiluAndMul.forward_native;self.product=ProductGate()

    def forward(self,x,w13,w2,weights,ids,num_experts,*,w13_scale,w2_scale,block_shape):
        from fastkernels.tasks.baseline.L1.moe_grouped_gemm import get_triton_config
        m,h=x.shape;k=ids.shape[-1];inner=w13.shape[1]//2
        config=get_triton_config(m,w13.shape,w2.shape,k,use_fp8=True,block_shape=block_shape)
        # Native grouped projections reduce one complete 128-wide scale block
        # before applying its FP32 scales, as this parent kernel supports.
        config['BLOCK_SIZE_K']=128
        alignment=self.align(ids,config['BLOCK_SIZE_M'],num_experts)
        q=torch.empty_like(x,dtype=torch.float8_e4m3fn);scale=torch.empty(m,(h+127)//128,device=x.device,dtype=torch.float32)
        self.quant(x,q,scale);up=torch.empty(m*k,inner*2,device=x.device,dtype=x.dtype)
        self.gemm(q,w13,up,None,*alignment,False,k,config=config,a_scale=scale,b_scale=w13_scale,use_fp8_w8a8=True,block_shape=block_shape)
        hidden=self.gate(up)
        q2=torch.empty_like(hidden,dtype=torch.float8_e4m3fn);scale2=torch.empty(m*k,(inner+127)//128,device=x.device,dtype=torch.float32)
        self.quant(hidden,q2,scale2);down=torch.empty(m*k,h,device=x.device,dtype=x.dtype)
        self.gemm(q2,w2,down,None,*alignment,False,1,config=config,a_scale=scale2,b_scale=w2_scale,use_fp8_w8a8=True,block_shape=block_shape)
        weighted=self.product(torch.cat((down,weights.to(x.dtype).reshape(-1,1).expand_as(down)),-1))
        return self.sum(weighted,k)
