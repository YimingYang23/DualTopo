"""DualTopo RATopo inference components."""
import copy


import inspect


import random


import numpy as np


from functools import partial


import torch


from mmdet.datasets import DATASETS


from mmdet.datasets.builder import PIPELINES


from mmcv.parallel import collate


from temporal_sampling import sequence_flags


STREAM_META_KEYS = ('stream_frame_index', 'stream_prev_index', 'stream_next_index')


def register_native_temporal_dataset(native_class, name):
    class NativeTemporalDataset(native_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.flag = sequence_flags(self.data_infos)

        def get_data_info(self, index):
            value = super().get_data_info(index)
            if value is None:
                return None
            previous = index - 1 if index and self.flag[index] == self.flag[index-1] else -1
            following = index + 1 if index + 1 < len(self) and self.flag[index] == self.flag[index+1] else -1
            value.update(stream_frame_index=index, stream_prev_index=previous,
                         stream_next_index=following)
            return value
    NativeTemporalDataset.__name__ = name
    DATASETS.register_module(name=name, module=NativeTemporalDataset)
    return NativeTemporalDataset


def temporal_dataset_config(native_config, registered_name):
    config = copy.deepcopy(native_config)
    config['type'] = registered_name

    def extend(sequence):
        for transform in sequence:
            if transform['type'] == 'CustomCollect3D':
                original = transform.get('meta_keys')
                if original is None:
                    original = inspect.signature(PIPELINES.get('CustomCollect3D').__init__).parameters['meta_keys'].default
                transform['meta_keys'] = tuple(original) + tuple(k for k in STREAM_META_KEYS if k not in original)
            if 'transforms' in transform:
                extend(transform['transforms'])
    extend(config['pipeline'])
    return config

