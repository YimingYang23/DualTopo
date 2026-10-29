"""DualTopo RATopo inference components."""
import ast


import copy


import hashlib


import importlib.util


from pathlib import Path


import types


import warnings


import math


import numpy as np


import torch


from torch import nn


import mmcv


from mmcv import Config


from mmcv.cnn import xavier_init, constant_init


from mmcv.cnn.utils import kaiming_init


from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE


from mmcv.cnn.bricks.transformer import TransformerLayerSequence, build_transformer_layer_sequence, build_positional_encoding


from mmcv.runner import force_fp32, auto_fp16


from mmcv.runner.base_module import BaseModule


from mmcv.utils import TORCH_VERSION, digit_version, deprecated_api_warning


from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch


def _class_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def load_components(donor_root, registry_prefix='DualTopoTransfer'):
    root = Path(donor_root)
    sources = {
        'base':root/'projects/bevformer/modules/custom_base_transformer_layer.py',
        'attention':root/'projects/bevformer/modules/temporal_self_attention.py',
        'ops':root/'projects/bevformer/modules/multi_scale_deformable_attn_function.py',
        'encoder':root/'projects/bevformer/modules/encoder.py',
        'constructor':root/'projects/lanesegnet/models/modules/wm_bevformer_constructer.py',
        'gru':root/'projects/plugin/models/necks/gru.py',
    }
    raw = {key:path.read_text() for key,path in sources.items()}
    ns = dict(globals())
    # Import its external dependencies, then execute only the undecorated class.
    # The donor also registers its base under a native name: do not overwrite it.
    base_tree = ast.parse(raw['base'])
    imports = [node for node in base_tree.body
               if isinstance(node,(ast.Import,ast.ImportFrom))
               and not (isinstance(node,ast.ImportFrom) and node.level)]
    exec(compile(ast.Module(body=imports,type_ignores=[]),str(sources['base']),'exec'),ns)
    exec(compile(_class_source(raw['base'],'MyCustomBaseTransformerLayer'),str(sources['base']),'exec'),ns)
    spec = importlib.util.spec_from_file_location(registry_prefix+'_ops',sources['ops'])
    ops = importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
    ns['MultiScaleDeformableAttnFunction_fp32'] = ops.MultiScaleDeformableAttnFunction_fp32
    ns['normal_'] = torch.nn.init.normal_
    # Unused spatial-attention types are kept for constructor init_weights' type checks.
    ns['MSDeformableAttention3D'] = type('_UnusedSpatialAttention',(nn.Module,),{})
    ns['CustomMSDeformableAttention'] = type('_UnusedCustomAttention',(nn.Module,),{})
    classes=[('attention','TemporalSelfAttention',ATTENTION),
             ('encoder','WMBEVFormerLayer',TRANSFORMER_LAYER),
             ('encoder','WMBEVFormerEncoder',TRANSFORMER_LAYER_SEQUENCE),
             ('constructor','WMBEVFormerConstructer',None),('gru','ConvGRU',None)]
    for key,name,registry in classes:
        exec(compile(_class_source(raw[key],name),str(sources[key]),'exec'),ns)
        if registry is not None:
            registry.register_module(name=registry_prefix+name,module=ns[name])
    config=Config.fromfile(str(root/'configs_geometry/standard_stage2.py'),import_custom_modules=False)
    cfg=copy.deepcopy(config.model.lane_head.wm_bev_constructor)
    cfg.pop('type')
    cfg.encoder.type=registry_prefix+'WMBEVFormerEncoder'
    cfg.encoder.transformerlayers.type=registry_prefix+'WMBEVFormerLayer'
    for attention in cfg.encoder.transformerlayers.attn_cfgs:
        assert attention.type=='TemporalSelfAttention'
        attention.type=registry_prefix+'TemporalSelfAttention'
    bwm=ns['WMBEVFormerConstructer'](**cfg)
    gru=ns['ConvGRU'](out_channels=256)
    provenance={key:{'path':str(sources[key]),'sha256':hashlib.sha256(sources[key].read_bytes()).hexdigest()} for key in sources}
    return bwm,gru,provenance


class BEVWorldModel(nn.Module):
    """Exact donor spatial path, using its per-sample BWM inference convention."""
    def __init__(self, donor_root, registry_prefix='DualTopoTransfer'):
        super().__init__()
        self.wm_bev_constructor,self.stream_fusion_neck,self.provenance=load_components(donor_root,registry_prefix)
        self.action_aware_bev_encoder=nn.Sequential(
            nn.Linear(256+12,256),nn.ReLU(inplace=True),
            nn.Linear(256,256),nn.ReLU(inplace=True),nn.Linear(256,256))
        self.down_sample=nn.AvgPool2d(kernel_size=2,stride=2)

    def predict_one(self, previous_bev, previous_meta, current_meta):
        from temporal_state import previous_to_current
        if previous_bev.shape!=(256,100,200):
            raise ValueError('Expected the donor [256,100,200] historical BEV')
        down=self.down_sample(previous_bev.unsqueeze(0))
        tokens=down.flatten(2).transpose(1,2)
        action=previous_to_current(previous_meta,current_meta,tokens).float()[:3].reshape(1,1,12)
        latent=self.action_aware_bev_encoder(torch.cat((tokens,action.expand(1,tokens.shape[1],12)),dim=-1))
        prediction=self.wm_bev_constructor(latent.permute(1,0,2),[previous_meta],prev_bev=None)
        planar=prediction.transpose(1,2).reshape(1,256,50,100)
        return torch.nn.functional.interpolate(planar,size=(100,200),mode='bilinear',align_corners=False)[0]

