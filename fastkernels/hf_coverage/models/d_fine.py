"""D-FINE and DEIMv2 distribution-refinement detectors using library operations."""
import torch
from torch import nn

from .hgnet_v2 import Backbone, Conv
from .rt_detr_v2 import ScoreScaledAttention
from ..runner import Workload
from ..patches.dfine_clamp import DFineClamp
from ..patches.detector_topk import DetectorTopK
from ..patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.frozen_batch_norm2d import FrozenBatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rtdetrv2_deformable_attention import MultiScaleDeformableAttentionV2
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid


def activation(name):
    return {None: nn.Identity, 'relu': ReLU, 'gelu': GELU, 'silu': SiLU}[name]()


def norm(width, eps=1e-5):
    return LayerNorm(width, eps=eps, promote_fp32=False)


class Attention(ScoreScaledAttention):
    def __init__(self, width, heads):
        super().__init__(width, heads)
        self.core = DenseAttention(backend='sdpa')

    def forward(self, hidden_states, position_embeddings=None):
        x = hidden_states
        positioned = x if position_embeddings is None else x+position_embeddings
        shape = lambda value: value.reshape(*x.shape[:2], self.num_heads, self.head_dim)
        result = self.core(shape(self.q_proj(positioned)), shape(self.k_proj(positioned)), shape(self.v_proj(x)))
        return self.out_proj(result.reshape(x.shape)), None


class MLP(nn.Module):
    def __init__(self, source, middle, target, count, act='relu'):
        super().__init__()
        dims = [source] + [middle] * (count - 1) + [target]
        self.layers = nn.ModuleList([Linear(a, b) for a, b in zip(dims, dims[1:])])
        self.act = activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 != len(self.layers):
                x = self.act(x)
        return x


class ScalarAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale, self.bias = nn.Parameter(torch.ones(1)), nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scale * x + self.bias


class AffineConv(Conv):
    def forward(self, x):
        return self.lab(super().forward(x))


def backbone(config):
    model = Backbone(config.backbone_config)
    for child in tuple(model.modules()):
        if isinstance(child, Conv):
            child.normalization = FrozenBatchNorm2d(child.normalization.running_mean.numel())
            if config.backbone_config.use_learnable_affine_block and isinstance(child.activation, ReLU):
                child.__class__ = AffineConv
                child.lab = ScalarAffine()
    return model


class ConvNorm(nn.Module):
    def __init__(self, c, source, target, kernel=1, stride=1, groups=1, act=None):
        super().__init__()
        self.conv = Conv2d(source, target, kernel, stride=stride, padding=(kernel-1)//2, groups=groups, bias=False)
        self.norm = BatchNorm2d(target, eps=c.batch_norm_eps)
        self.activation = activation(act)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class RepBlock(nn.Module):
    def __init__(self, c, width):
        super().__init__()
        self.conv1, self.conv2 = ConvNorm(c, width, width, 3), ConvNorm(c, width, width)
        self.activation = activation(c.activation_function)

    def forward(self, x):
        return self.activation(self.conv1(x) + self.conv2(x))


class CSP(nn.Module):
    def __init__(self, c, source, target):
        super().__init__()
        self.conv1, self.conv2 = [ConvNorm(c, source, target, act=c.activation_function) for _ in range(2)]
        self.bottlenecks = nn.ModuleList([RepBlock(c, target) for _ in range(round(3*c.depth_mult))])
        self.conv3 = nn.Identity()

    def forward(self, x):
        a = self.conv1(x)
        for layer in self.bottlenecks:
            a = layer(a)
        return self.conv3(a + self.conv2(x))


class DeimCSP(nn.Module):
    def __init__(self, c, source, target):
        super().__init__()
        self.conv1 = ConvNorm(c, source, 2*target, act=c.activation_function)
        self.bottlenecks = nn.ModuleList([RepBlock(c, target) for _ in range(round(3*c.depth_mult))])
        self.conv2 = ConvNorm(c, target, target, 3, act=c.activation_function)

    def forward(self, x):
        residual, x = self.conv1(x).chunk(2, dim=1)
        for layer in self.bottlenecks:
            x = layer(x)
        return self.conv2(residual+x)


class ELAN(nn.Module):
    def __init__(self, c, deim):
        super().__init__()
        w, h = c.encoder_hidden_dim, round(c.hidden_expansion*c.encoder_hidden_dim//2)
        self.deim = deim
        self.conv1 = ConvNorm(c, w if deim else 2*w, 2*w, act=c.activation_function)
        csp = DeimCSP if deim else CSP
        self.csp_rep1, self.csp_rep2 = csp(c, w, h), csp(c, h, h)
        if deim:
            self.conv2 = ConvNorm(c, 2*w+2*h, w, act=c.activation_function)
        else:
            self.conv2, self.conv3 = [ConvNorm(c, h, h, 3, act=c.activation_function) for _ in range(2)]
            self.conv4 = ConvNorm(c, 2*w+2*h, w, act=c.activation_function)

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, dim=1)
        d = self.csp_rep1(b)
        if not self.deim:
            d = self.conv2(d)
        e = self.csp_rep2(d)
        if not self.deim:
            e = self.conv3(e)
        return (self.conv2 if self.deim else self.conv4)(torch.cat((a,b,d,e), dim=1))


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c.encoder_hidden_dim
        self.conv1, self.conv2 = ConvNorm(c,w,w), ConvNorm(c,w,w,3,2,w)

    def forward(self,x):
        return self.conv2(self.conv1(x))


class EncoderLayer(nn.Module):
    def __init__(self,c):
        super().__init__()
        w = c.encoder_hidden_dim
        self.self_attn = Attention(w,c.encoder_attention_heads)
        self.self_attn_layer_norm, self.final_layer_norm = norm(w,c.layer_norm_eps), norm(w,c.layer_norm_eps)
        self.mlp = MLP(w,c.encoder_ffn_dim,w,2,c.encoder_activation_function)

    def forward(self,x,pos):
        x = self.self_attn_layer_norm(x+self.self_attn(x,position_embeddings=pos)[0])
        return self.final_layer_norm(x+self.mlp(x))


class AIFI(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c = c
        self.layers = nn.ModuleList([EncoderLayer(c) for _ in range(c.encoder_layers)])

    def forward(self,x):
        b,w,h,k = x.shape
        x = x.flatten(2).transpose(1,2)
        pos = None
        if self.c.eval_size is None:
            # Shape-only positional metadata, preserving reference dtype/axis order.
            xx,yy = torch.meshgrid(torch.arange(k,device=x.device).to(x.dtype),torch.arange(h,device=x.device).to(x.dtype),indexing='xy')
            omega = 1 / self.c.positional_encoding_temperature ** (torch.arange(w//4,device=x.device).to(x.dtype)/(w//4))
            xx,yy = xx.flatten()[:,None]*omega[None], yy.flatten()[:,None]*omega[None]
            pos = torch.cat((yy.sin(),yy.cos(),xx.sin(),xx.cos()),dim=1)[None]
        for layer in self.layers:
            x = layer(x,pos)
        return x.transpose(1,2).reshape(b,w,h,k).contiguous()


class Encoder(nn.Module):
    def __init__(self,c,deim):
        super().__init__()
        self.c,self.deim = c,deim
        w,n = c.encoder_hidden_dim,len(c.encoder_in_channels)-1
        self.aifi = nn.ModuleList([AIFI(c) for _ in c.encode_proj_layers])
        self.lateral_convs = nn.ModuleList([ConvNorm(c,w,w) for _ in range(n)])
        self.fpn_blocks,self.pan_blocks = [nn.ModuleList([ELAN(c,deim) for _ in range(n)]) for _ in range(2)]
        self.downsample_convs = nn.ModuleList([Down(c) for _ in range(n)])
        self.interpolate = Interpolate()

    def fuse(self,a,b):
        return a+b if self.deim else torch.cat((a,b),dim=1)

    def forward(self,features):
        for layer,index in zip(self.aifi,self.c.encode_proj_layers):
            features[index] = layer(features[index])
        fpn=[features[-1]]
        for i,(lateral,block) in enumerate(zip(self.lateral_convs,self.fpn_blocks)):
            fpn[-1] = lateral(fpn[-1])
            up = self.interpolate(fpn[-1],scale_factor=2.,mode='nearest')
            fpn.append(block(self.fuse(up,features[-i-2])))
        fpn.reverse()
        pan=[fpn[0]]
        for i,(down,block) in enumerate(zip(self.downsample_convs,self.pan_blocks)):
            pan.append(block(self.fuse(down(pan[-1]),fpn[i+1])))
        return pan


class Sampling(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        self.points=c.decoder_n_points
        total=sum(self.points)*c.decoder_attention_heads
        self.sampling_offsets,self.attention_weights = Linear(c.d_model,total*2),Linear(c.d_model,total)
        self.register_buffer('num_points_scale',torch.tensor([1/n for n in self.points for _ in range(n)]))
        self.softmax,self.product,self.sampling = Softmax(dim=-1),ProductGate(),MultiScaleDeformableAttentionV2()

    def forward(self,x,memory,refs,shapes):
        b,q,w=x.shape
        heads=self.c.decoder_attention_heads
        offsets=self.sampling_offsets(x).reshape(b,q,heads,sum(self.points),2)
        scale=self.num_points_scale.to(x.dtype)[None,None,None,:,None].expand_as(offsets)
        offsets=self.product(torch.cat((offsets,scale),dim=-1))
        size=refs[:,:,None,None,2:].expand_as(offsets)
        offsets=self.product(torch.cat((offsets,size),dim=-1))*self.c.decoder_offset_scale
        locations=refs[:,:,None,None,:2]+offsets
        weights=self.softmax(self.attention_weights(x).reshape(b,q,heads,-1))
        return self.sampling(memory.reshape(b,-1,heads,w//heads),shapes,locations,weights,self.points)


class Gate(nn.Module):
    def __init__(self,c,deim):
        super().__init__()
        self.gate=Linear(2*c.d_model,2*c.d_model)
        self.norm=RMSNormNative(c.d_model) if deim else norm(c.d_model)
        self.sigmoid,self.product=Sigmoid(),ProductGate()

    def forward(self,a,b):
        ga,gb=self.sigmoid(self.gate(torch.cat((a,b),dim=-1))).chunk(2,dim=-1)
        return self.norm(self.product(torch.cat((ga,a),dim=-1))+self.product(torch.cat((gb,b),dim=-1)))


class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate_proj,self.up_proj=[Linear(c.d_model,c.decoder_ffn_dim//2) for _ in range(2)]
        self.down_proj=Linear(c.decoder_ffn_dim//2,c.d_model)
        self.activation = SiluAndMul.forward_native

    def forward(self,x):
        return self.down_proj(self.activation(torch.cat((self.gate_proj(x),self.up_proj(x)),dim=-1)))


class DecoderLayer(nn.Module):
    def __init__(self,c,deim):
        super().__init__()
        self.deim=deim
        self.self_attn=Attention(c.d_model,c.decoder_attention_heads)
        self.self_attn_layer_norm,self.final_layer_norm=[RMSNormNative(c.d_model) if deim else norm(c.d_model,c.layer_norm_eps) for _ in range(2)]
        self.encoder_attn,self.gateway=Sampling(c),Gate(c,deim)
        self.mlp=SwiGLU(c) if deim else MLP(c.d_model,c.decoder_ffn_dim,c.d_model,2,c.decoder_activation_function)
        self.clamp=DFineClamp()

    def forward(self,x,pos,memory,refs,shapes):
        x=self.self_attn_layer_norm(x+self.self_attn(x,position_embeddings=pos)[0])
        x=self.gateway(x,self.encoder_attn(x+pos,memory,refs,shapes))
        x=x+self.mlp(x)
        return self.final_layer_norm(x if self.deim else self.clamp(x,-65504,65504))


class LQE(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        self.reg_conf=MLP(4*(c.top_prob_values+1),c.lqe_hidden_dim,1,c.lqe_layers)
        self.softmax,self.topk,self.reduce=Softmax(dim=-1),DetectorTopK(),SegmentCSR()

    def forward(self,scores,corners):
        b,q=corners.shape[:2]
        prob=self.softmax(corners.reshape(b,q,4,self.c.max_num_bins+1))
        top,_=self.topk(prob,self.c.top_prob_values)
        top=top.to(prob.dtype)
        flat=top.reshape(-1)
        offsets=torch.arange(0,flat.numel()+1,self.c.top_prob_values,device=flat.device,dtype=torch.long)
        # Native mean accumulates low-precision inputs in FP32. SegmentCSR
        # otherwise rounds each intermediate sum in the input dtype.
        mean=self.reduce(flat.float(),offsets,reduce='mean').to(top.dtype).reshape(b,q,4,1)
        return scores+self.reg_conf(torch.cat((top,mean),dim=-1).reshape(b,q,-1))


class Decoder(nn.Module):
    def __init__(self,c,deim):
        super().__init__()
        self.c=c
        self.layers=nn.ModuleList([DecoderLayer(c,deim) for _ in range(c.decoder_layers)])
        self.query_pos_head=MLP(4,c.d_model if deim else 2*c.d_model,c.d_model,3 if deim else 2,c.decoder_activation_function if deim else 'relu')
        self.pre_bbox_head=MLP(c.d_model,c.d_model,4,3)
        self.reg_scale,self.up=nn.Parameter(torch.tensor([c.reg_scale]),requires_grad=False),nn.Parameter(torch.tensor([c.up]),requires_grad=False)
        self.lqe_layers=nn.ModuleList([LQE(c) for _ in range(c.decoder_layers)])
        self.clamp,self.sigmoid,self.softmax,self.integral,self.product=DFineClamp(),Sigmoid(),Softmax(dim=-1),Matmul(),ProductGate()

    def prepare_constants(self):
        # Inference-constant projection; evaluated once after weights and dtypes load.
        up,scale=self.up.abs()[0],self.reg_scale.abs()
        bound=up*scale
        step=(bound+1)**(2/(self.c.max_num_bins-2))
        values=[-bound*2]+[-step**i+1 for i in range(self.c.max_num_bins//2-1,0,-1)]+[torch.zeros_like(up[None])]+[step**i-1 for i in range(1,self.c.max_num_bins//2)]+[bound*2]
        self.register_buffer('project',torch.cat(values),persistent=False)
        self.scale_value=float(scale.item())

    def boxes(self,refs,distance):
        scale=self.scale_value
        size=(refs[...,2:]/scale).repeat(1,1,2)
        offsets=distance+0.5*scale
        offsets=self.product(torch.cat((offsets,size),dim=-1))
        lo,hi=refs[...,:2]-offsets[...,:2],refs[...,:2]+offsets[...,2:]
        return torch.cat(((lo+hi)/2,hi-lo),dim=-1)

    def forward(self,x,memory,references,shapes):
        refs=self.sigmoid(references)
        previous=0
        corners_previous=0
        intermediate=[]
        for i,layer in enumerate(self.layers):
            pos=self.clamp(self.query_pos_head(refs),-10,10)
            x=layer(x,pos,memory,refs,shapes)
            if i==0:
                initial=self.sigmoid(self.pre_bbox_head(x)+inverse_sigmoid(refs))
            corners=self.bbox_embed[i](x+previous)+corners_previous
            distances=self.integral(self.softmax(corners.reshape(-1,self.c.max_num_bins+1)),self.project).reshape(*corners.shape[:2],4)
            refs=self.boxes(initial,distances)
            previous,corners_previous=x,corners
            intermediate.append(x)
        scores=self.lqe_layers[-1](self.class_embed[-1](x),corners)
        return dict(last_hidden_state=x,intermediate_hidden_states=torch.stack(intermediate),intermediate_logits=scores[:,None],intermediate_reference_points=refs[:,None],intermediate_predicted_corners=corners[:,None],initial_reference_points=initial[:,None])


class Detector(nn.Module):
    def __init__(self,c,deim=False):
        super().__init__()
        self.c,self.deim=c,deim
        self.model=nn.Module()
        m=self.model
        carrier=nn.Module()
        carrier.model=backbone(c)
        if deim:
            m.conv_encoder=carrier
            carrier.encoder_input_proj=nn.ModuleList([ConvNorm(c,w,c.encoder_hidden_dim) for w in c.encoder_in_channels])
        else:
            m.backbone=carrier
            m.encoder_input_proj=nn.ModuleList([nn.Sequential(Conv2d(w,c.encoder_hidden_dim,1,bias=False),BatchNorm2d(c.encoder_hidden_dim)) for w in c.encoder_in_channels])
        m.encoder=Encoder(c,deim)
        m.denoising_class_embed=nn.Embedding(c.num_labels+1,c.d_model,padding_idx=c.num_labels)
        m.enc_output=nn.Sequential(Linear(c.d_model,c.d_model),norm(c.d_model,c.layer_norm_eps))
        m.enc_score_head,m.enc_bbox_head=Linear(c.d_model,c.num_labels),MLP(c.d_model,c.d_model,4,3)
        m.decoder_input_proj=nn.ModuleList([nn.Identity() if w==c.d_model else nn.Sequential(Conv2d(w,c.d_model,1,bias=False),BatchNorm2d(c.d_model,eps=c.batch_norm_eps)) for w in c.decoder_in_channels])
        m.decoder=Decoder(c,deim)
        self.class_embed=nn.ModuleList([Linear(c.d_model,c.num_labels) for _ in range(c.decoder_layers)])
        self.bbox_embed=nn.ModuleList([MLP(c.d_model,c.d_model,4*(c.max_num_bins+1),3) for _ in range(c.decoder_layers)])
        m.decoder.class_embed,m.decoder.bbox_embed=self.class_embed,self.bbox_embed
        self.reduce,self.topk,self.sigmoid=SegmentCSR(),DetectorTopK(),Sigmoid()
        self.interpolate = Interpolate()

    def forward(self,pixel_values,pixel_mask=None):
        m,c=self.model,self.c
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (pixel_values.shape[0], *pixel_values.shape[-2:]),
                device=pixel_values.device,
            )
        carrier=m.conv_encoder if self.deim else m.backbone
        features=list(carrier.model(pixel_values).values())
        if not self.deim:
            # The native backbone computes these metadata masks even though
            # the hybrid encoder subsequently consumes only feature tensors.
            feature_masks = [
                self.interpolate(pixel_mask[None].float(), size=x.shape[-2:]).bool()[0]
                for x in features
            ]
        projs=carrier.encoder_input_proj if self.deim else m.encoder_input_proj
        encoded=m.encoder([p(x) for p,x in zip(projs,features)])
        sources=[p(x) for p,x in zip(m.decoder_input_proj,encoded)]
        shapes=[tuple(x.shape[-2:]) for x in sources]
        memory=torch.cat([x.flatten(2).transpose(1,2) for x in sources],dim=1)
        anchors=[]
        for level,(h,w) in enumerate(shapes):
            y,x=torch.meshgrid(torch.arange(h,device=memory.device).to(memory.dtype),torch.arange(w,device=memory.device).to(memory.dtype),indexing='ij')
            xy=torch.stack((x,y),dim=-1)[None]+0.5
            xy[...,0]/=w
            xy[...,1]/=h
            wh=torch.ones_like(xy)*0.05*(2.**level)
            anchors.append(torch.cat((xy,wh),dim=-1).reshape(1,-1,4))
        anchors=torch.cat(anchors,dim=1)
        valid=((anchors>.01)*(anchors<.99)).all(-1,keepdim=True)
        anchors=torch.log(anchors/(1-anchors)).masked_fill(~valid,torch.finfo(memory.dtype).max)
        projected=m.enc_output(memory.masked_fill(~valid,0))
        classes=m.enc_score_head(projected)
        coordinates=m.enc_bbox_head(projected)+anchors
        flat=classes.reshape(-1)
        offsets=torch.arange(0,flat.numel()+1,c.num_labels,device=flat.device,dtype=torch.long)
        scores=self.reduce(flat,offsets,reduce='max').reshape(classes.shape[:2])
        _,indices=self.topk(scores,c.num_queries)
        def gather(value):
            return value.gather(1,indices[...,None].expand(-1,-1,value.shape[-1]))
        references=gather(coordinates)
        output=m.decoder(gather(projected),memory,references,shapes)
        output.update(logits=output['intermediate_logits'][:,-1],pred_boxes=output['intermediate_reference_points'][:,-1],init_reference_points=references,enc_topk_logits=gather(classes),enc_topk_bboxes=self.sigmoid(references),enc_outputs_class=classes,enc_outputs_coord_logits=coordinates)
        for i,x in enumerate(encoded):
            output[f'encoder_last_hidden_state.{i}']=x
        return output


def build_from_config(config,device,dtype):
    config.num_labels = len(config.id2label)
    deim = config.model_type == 'deimv2'
    if (config.normalize_before or config.eval_idx != -1
            or config.learn_initial_query or config.anchor_image_size is not None
            or config.decoder_method != 'default' or config.layer_scale != 1
            or config.decoder_layers < 2 or config.num_denoising <= 0
            or not config.with_box_refine or not config.freeze_backbone_batch_norms
            or config.backbone_config.model_type != 'hgnet_v2'
            or config.num_feature_levels != len(config.encoder_in_channels)
            or config.num_feature_levels != len(config.decoder_in_channels)
            or (deim and (config.encoder_type != 'hybrid'
                          or config.encoder_fuse_op != 'sum'
                          or not config.encoder_has_trailing_conv
                          or not config.use_gateway or config.share_bbox_head))):
        raise ValueError('Detector composition requires selected checkpoint inference settings')
    return Detector(config,deim).to(device=device,dtype=dtype).eval()


def load_state_dict_into(model,state_dict,config):
    mapped={}
    for name in model.state_dict():
        source=name.replace('.self_attn.out_proj.','.self_attn.o_proj.')
        mapped[name]=state_dict[source]
    model.load_state_dict(mapped,strict=True,assign=True)
    model.model.decoder.prepare_constants()


def make_workloads(model,inputs,config):
    return {'forward':Workload(run=lambda:model(**inputs))}
