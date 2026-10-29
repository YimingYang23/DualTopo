"""Strict, inference-only DualTopo-RATopo model loading and state management."""
import copy
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'vendor' / 'native', ROOT / 'vendor' / 'openlane', ROOT / 'dualtopo_ratopo' / 'runtime'):
    sys.path.insert(0, str(directory))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config():
    from mmcv import Config
    return Config.fromfile(str(ROOT.parents[1] / 'configs' / 'subset_b.py'))


def load_state(path):
    """Read data-only safetensors after checking the published checksum."""
    from safetensors.numpy import load_file
    manifest = json.loads((ROOT / 'weights.json').read_text())
    path = Path(path)
    if sha256(path) != manifest['sha256']:
        raise ValueError('Checkpoint SHA256 does not match weights.json')
    arrays = load_file(str(path))
    state = {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}
    if len(state) != manifest['tensor_count']:
        raise ValueError('Checkpoint tensor count changed')
    if any(torch.is_floating_point(value) and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint contains non-finite tensors')
    return state


def build_model(weights, device='cuda:0'):
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.models import build_model as build_native_model
    from donor_bwm import BEVWorldModel
    from wm_bev_state import BEVWorldModelState
    from query_world_model import QueryWorldModel
    from ratopo_shared_head import attach_shared_head
    from ratopo_centerline_decoder import attach_centerline_decoder
    from ratopo_owned_inference import attach_owned_centerline_inference
    from .runtime.geometry import NativeBootstrapRouter, GEOMETRY_MARKER

    config = load_config()
    import_modules_from_strings(**config.custom_imports)
    model = build_native_model(copy.deepcopy(config.model))
    native = ROOT / 'vendor' / 'native'
    donor = ROOT / 'vendor' / 'donor'
    head = model.pts_bbox_head
    decoder = head.transformer.decoder
    module = importlib.util.spec_from_file_location('DualTopoRATopoAttentionKernel', donor / 'projects/bevformer/modules/multi_scale_deformable_attn_function.py')
    kernel = importlib.util.module_from_spec(module)
    module.loader.exec_module(kernel)
    attach_shared_head(head, native / 'projects/ratopo/models/dense_heads/ratopo_head.py',
                       native / 'projects/ratopo/models/modules/ratopo_transformer.py',
                       native / 'projects/ratopo/models/modules/ratopo_decoder.py')
    attach_centerline_decoder(decoder, donor / 'projects/lanesegnet/models/modules/lane_attention.py',
                              (native / 'projects/ratopo/models/modules/ratopo_decoder.py').read_text(),
                              vars(sys.modules[type(decoder).__module__]), kernel.MultiScaleDeformableAttnFunction_fp32)
    model.add_module('dualtopo_bev_bridge', BEVWorldModelState(BEVWorldModel(donor, registry_prefix='DualTopoRATopoInference')))
    model.add_module('dualtopo_query_world_model', QueryWorldModel())
    state = load_state(weights)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('Checkpoint is not an exact match for this project')
    if any(not torch.equal(value.cpu(), state[name]) for name, value in model.state_dict().items()):
        raise ValueError('Loaded model tensor values differ from checkpoint')
    del state
    model.video_test_mode = False
    model._consistent_stage_policy = bool(config.release.get('consistent_stage_policy', False))
    entry = attach_owned_centerline_inference(model, native / 'projects/ratopo/models/detectors/ratopo.py',
                                              geometry_router_class=NativeBootstrapRouter, require_isolated_mode=True)
    model.to(device).eval()
    model._release_entry = model.simple_test_owned_centerline_query
    model._release_geometry_marker = GEOMETRY_MARKER
    return model


@contextmanager
def inference_session(model, branch='temporal'):
    """Keep chronological frames together; clear history at session boundaries."""
    from migration_temporal_evaluation import isolated_native_evaluation_mode
    if branch not in ('singleframe', 'temporal'):
        raise ValueError('Choose singleframe or temporal')
    model._consistent_stage_policy = bool(load_config().release.get('consistent_stage_policy', False)) and branch == 'temporal'
    with isolated_native_evaluation_mode(model, branch), torch.no_grad():
        yield branch


def predict(model, data, branch='temporal'):
    inputs = {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in data.items()}
    metadata = inputs.pop('img_metas')
    bev, results = model._release_entry(metadata, **inputs, evaluation_branch=branch,
                                        initial_centerline_points=model._release_geometry_marker, rescale=True)
    return bev, results, model._owned_eval_last_trace
