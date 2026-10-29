"""DualTopo RATopo inference components."""
import numpy as np


def sequence_flags(data_infos):
    flags = np.empty(len(data_infos), dtype=np.int64)
    closed = set()
    previous_scene = None
    previous_timestamp = None
    group = -1
    for index, info in enumerate(data_infos):
        scene = str(info['segment_id'])
        timestamp = int(info['timestamp'])
        if scene != previous_scene:
            if scene in closed:
                raise ValueError('Interleaved scenes require explicit ordering audit')
            if previous_scene is not None:
                closed.add(previous_scene)
            group += 1
            previous_timestamp = None
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError('Native scene frames must be strictly chronological')
        flags[index] = group
        previous_scene, previous_timestamp = scene, timestamp
    return flags

