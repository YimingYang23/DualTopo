"""DualTopo RATopo inference components."""
import ast,copy,math,warnings


import numpy as np


import torch


from torch import nn


from mmcv.cnn import xavier_init,constant_init


from mmcv.runner.base_module import BaseModule


from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch


def centerline_sine_position(normalized_xy):
    """Same y,x 256-dimensional sinusoid as actual DualTopo donor."""
    if normalized_xy.shape[-1]!=2:
        raise ValueError('Normalized XY coordinates required')
    dim_t=torch.arange(128,dtype=torch.float32,device=normalized_xy.device)
    dim_t=10000**(2*(dim_t//2)/128)
    xy=normalized_xy*(2*math.pi)
    x=xy[...,0,None]/dim_t;y=xy[...,1,None]/dim_t
    x=torch.stack((x[...,0::2].sin(),x[...,1::2].cos()),dim=-1).flatten(-2)
    y=torch.stack((y[...,0::2].sin(),y[...,1::2].cos()),dim=-1).flatten(-2)
    return torch.cat((y,x),dim=-1)


class CenterlineDynamicPointPE(nn.Module):
    def __init__(self,num_points=11,num_layers=6):
        super().__init__()
        self.num_points=num_points
        template=nn.Linear(256,256)
        self.projections=nn.ModuleList([copy.deepcopy(template) for _ in range(num_layers)])
        for m in self.projections:xavier_init(m,distribution='uniform',bias=0.)

    def forward(self,layer_index,normalized_centerline):
        if normalized_centerline.ndim!=4 or normalized_centerline.shape[-2]!=self.num_points or normalized_centerline.shape[-1] not in (2,3):
            raise ValueError('B,Q,11,XYZ centerline points required')
        if not 0<=layer_index<len(self.projections):raise ValueError('Original decoder layer index required')
        # Native layer-to-layer geometry detach is retained; current query
        # and every projection still receive the actual attention gradients.
        xy=normalized_centerline.detach()[...,:2]
        position=self.projections[layer_index](centerline_sine_position(xy))
        references=xy.reshape(xy.shape[0],-1,1,2)
        return position,references

    def main_lane_region(self,layer_index,normalized_centerlines,start,count):
        if start<0 or count<=0 or start+count>normalized_centerlines.shape[1]:
            raise ValueError('Explicit main-lane region required; exclude DN and auxiliary queries')
        return self(layer_index,normalized_centerlines[:,start:start+count])


def build_point_attention_class(donor_path,cuda_function=None):
    """Actual donor attention AST, only edge-point count becomes a parameter.

    No registry mutation. Native CUDA implementation must be explicitly
    supplied before GPU integration; CPU contracts use real PyTorch MSDA.
    """
    tree=ast.parse(donor_path.read_text())
    cls=copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='StreamLaneAttention'))
    cls.decorator_list=[];cls.name='PrivateCenterlinePointAttention'
    init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    init.args.args.append(ast.arg(arg='num_centerline_points'));init.args.defaults.append(ast.Constant(value=11))
    node=next(x for x in ast.walk(init) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='num_pts_per_vec' for t in x.targets))
    assert isinstance(node.value,ast.Constant) and node.value.value==8
    node.value=ast.Name(id='num_centerline_points',ctx=ast.Load())
    namespace=dict(torch=torch,nn=nn,np=np,math=math,warnings=warnings,BaseModule=BaseModule,
        xavier_init=xavier_init,constant_init=constant_init,
        multi_scale_deformable_attn_pytorch=multi_scale_deformable_attn_pytorch,
        MultiScaleDeformableAttnFunction_fp32=cuda_function)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(donor_path),'exec'),namespace)
    return namespace[cls.name]

