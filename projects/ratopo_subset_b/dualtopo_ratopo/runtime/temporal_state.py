"""DualTopo RATopo inference components."""
import copy


import torch


def tokens_to_bev(tokens, height, width):
    """Convert explicit [B,H*W,C] tokens to [B,C,H,W] without mixing cells."""
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError('Expected [B,H*W,C] tokens')
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)


def bev_to_tokens(bev):
    if bev.ndim != 4:
        raise ValueError('Expected [B,C,H,W] BEV')
    return bev.flatten(2).transpose(1, 2)


def previous_to_current(previous, current, reference):
    """The donor action is inverse(current lidar->global) @ previous pose."""
    def matrix(meta):
        rotation = reference.new_tensor(meta['lidar2global_rotation'], dtype=torch.float64)
        translation = reference.new_tensor(meta['lidar2global_translation'], dtype=torch.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError('Expected a 3x3 rotation and 3-vector translation')
        m = torch.eye(4, dtype=torch.float64, device=reference.device)
        m[:3, :3], m[:3, 3] = rotation, translation
        return m
    prev, cur = matrix(previous), matrix(current)
    inverse = torch.eye(4, dtype=torch.float64, device=reference.device)
    inverse[:3, :3] = cur[:3, :3].T
    inverse[:3, 3] = -(cur[:3, :3].T @ cur[:3, 3])
    return inverse @ prev


class TemporalState:
    """Per-batch-slot detached history, reset on scene or explicit no-history.

    Payloads can contain BEV, queries and points. Nested tensors are detached
    and copied. Switching mode never exposes or clears the other mode's state.
    A batch-size change clears the selected buffer rather than mixing slots.
    """
    def __init__(self):
        self.training = True
        self._buffers = {True: [], False: []}

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def clear(self, all_modes=False):
        for mode in ([True, False] if all_modes else [self.training]):
            self._buffers[mode] = []

    def _sized(self, size):
        if len(self._buffers[self.training]) != size:
            self._buffers[self.training] = [None] * size
        return self._buffers[self.training]

    def get(self, metas):
        buffer = self._sized(len(metas))
        result = []
        for index, meta in enumerate(metas):
            old = buffer[index]
            if (old is None or old['meta']['scene_token'] != meta['scene_token']
                    or meta.get('prev_bev_exists') is False):
                buffer[index] = None
            result.append(buffer[index])
        return result

    @staticmethod
    def _snapshot(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: TemporalState._snapshot(item) for key, item in value.items()}
        if isinstance(value, list):
            return [TemporalState._snapshot(item) for item in value]
        if isinstance(value, tuple):
            return tuple(TemporalState._snapshot(item) for item in value)
        return copy.deepcopy(value)

    def update(self, payloads, metas):
        if len(payloads) != len(metas):
            raise ValueError('Payload/meta batch mismatch')
        buffer = self._sized(len(metas))
        for index, (payload, meta) in enumerate(zip(payloads, metas)):
            buffer[index] = {'payload': self._snapshot(payload), 'meta': copy.deepcopy(meta)}

