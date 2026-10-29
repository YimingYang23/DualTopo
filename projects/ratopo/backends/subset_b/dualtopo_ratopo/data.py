"""Native image geometry and chronological metadata; no dataset files included."""
import copy
import json
from pathlib import Path

import torch

from .model import ROOT, load_config, sha256


def build_dataset(data_root, annotations, allow_annotation_mismatch=False):
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets import DATASETS, build_dataset as build_native_dataset
    from native_temporal_dataset import register_native_temporal_dataset, temporal_dataset_config

    config = load_config()
    import_modules_from_strings(**config.custom_imports)
    expected = json.loads((ROOT / 'results.json').read_text())
    if not allow_annotation_mismatch and sha256(annotations) != expected['annotation_sha256']:
        raise ValueError('Annotation checksum differs from the reported evaluation; use --allow-annotation-mismatch only for another dataset version')
    name = 'DualTopoRATopoInferenceDataset'
    if DATASETS.get(name) is None:
        register_native_temporal_dataset(DATASETS.get(config.data_val.type), name)
    dataset_config = temporal_dataset_config(config.data_val, name)
    dataset_config.update(ann_file=str(Path(annotations).absolute()), data_root=str(Path(data_root).absolute()))
    dataset = build_native_dataset(dataset_config)
    if len(dataset) != expected['samples']:
        raise ValueError('Validation frame count differs from the selected checkpoint protocol')
    return dataset


def prepare(dataset, index, device='cuda:0'):
    from mmcv.parallel import DataContainer, collate

    def place(value):
        if isinstance(value, DataContainer):
            return value.data[0] if value.cpu_only else place(value.data[0])
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: place(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(place(item) for item in value)
        return value

    data = place(collate([dataset[index]], samples_per_gpu=1))
    images, metadata = data['img'], data['img_metas']
    if isinstance(images, list):
        images, metadata = images[0], metadata[0]
    if images.ndim == 6:
        images = images[:, 0]
    if isinstance(metadata, dict):
        metadata = [metadata]
    if 0 in metadata[0]:
        metadata = [item[0] for item in metadata]
    metadata = copy.deepcopy(metadata)
    if load_config().release.get('zero_can_bus', False):
        for item in metadata:
            item['can_bus'][:3] = 0
            item['can_bus'][-1] = 0
    return dict(img=images, img_metas=metadata)
