# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""Detached scene-aware query, adjacency and BEV latent memory buffers."""

import copy


class StreamTensorMemory:
    """Stores per-batch BEV tensors across frames for temporal fusion."""

    def __init__(self, batch_size):
        self.train_bs = batch_size
        self.training = True
        self.bs = self.train_bs
        self.train_memory_list = [None for i in range(self.bs)]
        self.train_img_metas_memory = [None for i in range(self.bs)]
        self.train_topoidx_memory_list = [None for i in range(self.bs)]
        self.test_memory_list = []
        self.test_img_metas_memory = []

    @property
    def memory_list(self):
        if self.training:
            return self.train_memory_list
        else:
            return self.test_memory_list

    @property
    def img_metas_memory(self):
        if self.training:
            return self.train_img_metas_memory
        else:
            return self.test_img_metas_memory

    def _ensure_capacity(self, size):
        while len(self.test_memory_list) < size:
            self.test_memory_list.append(None)
            self.test_img_metas_memory.append(None)

    def update(self, memory, img_metas, topo_idx=None):
        for i in range(self.bs):
            self.memory_list[i] = memory[i].clone().detach()
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])
            if topo_idx is not None and i < len(self.train_topoidx_memory_list):
                self.train_topoidx_memory_list[i] = topo_idx[i].clone().detach()

    def reset_single(self, idx):
        self.memory_list[idx] = None
        self.img_metas_memory[idx] = None
        if idx < len(self.train_topoidx_memory_list):
            self.train_topoidx_memory_list[idx] = None

    def get(self, img_metas):
        actual_bs = len(img_metas)
        if not self.training:
            self._ensure_capacity(actual_bs)
            self.bs = actual_bs
        tensor_list = []
        img_metas_list = []
        topo_idx_list = []
        is_first_frame_list = []
        for i in range(self.bs):
            if not self.img_metas_memory[i]:
                is_first_frame = True
            else:
                is_first_frame = (
                    img_metas[i]["scene_token"] != self.img_metas_memory[i]["scene_token"]
                )
            if is_first_frame:
                self.reset_single(i)
            tensor_list.append(self.memory_list[i])
            img_metas_list.append(self.img_metas_memory[i])
            is_first_frame_list.append(is_first_frame)
            topo_idx_list.append(
                self.train_topoidx_memory_list[i]
                if i < len(self.train_topoidx_memory_list)
                else None
            )
        return {
            "tensor": tensor_list,
            "img_metas": img_metas_list,
            "is_first_frame": is_first_frame_list,
            "topo_idx": topo_idx_list,
        }

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.bs = self.train_bs

    def eval(self):
        self.training = False

    def reset(self):
        self.train_memory_list = [None for i in range(self.train_bs)]
        self.train_img_metas_memory = [None for i in range(self.train_bs)]
        self.train_topoidx_memory_list = [None for i in range(self.train_bs)]
        self.test_memory_list = []
        self.test_img_metas_memory = []


class StreamListMemory:
    """List-based temporal memory that stores references (no clone on update)."""

    def __init__(self, batch_size):
        self.train_bs = batch_size
        self.training = True
        self.bs = self.train_bs
        self.train_memory_list = [None for i in range(self.bs)]
        self.train_img_metas_memory = [None for i in range(self.bs)]
        self.train_topoidx_memory_list = [None for i in range(self.bs)]
        self.test_memory_list = []
        self.test_img_metas_memory = []

    @property
    def memory_list(self):
        if self.training:
            return self.train_memory_list
        else:
            return self.test_memory_list

    @property
    def img_metas_memory(self):
        if self.training:
            return self.train_img_metas_memory
        else:
            return self.test_img_metas_memory

    def _ensure_capacity(self, size):
        while len(self.test_memory_list) < size:
            self.test_memory_list.append(None)
            self.test_img_metas_memory.append(None)

    def update(self, memory, img_metas, topo_idx=None):
        for i in range(self.bs):
            self.memory_list[i] = memory[i]
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])
            if topo_idx is not None and i < len(self.train_topoidx_memory_list):
                self.train_topoidx_memory_list[i] = topo_idx[i].clone().detach()

    def reset_single(self, idx):
        self.memory_list[idx] = None
        self.img_metas_memory[idx] = None
        if idx < len(self.train_topoidx_memory_list):
            self.train_topoidx_memory_list[idx] = None

    def get(self, img_metas):
        actual_bs = len(img_metas)
        if not self.training:
            self._ensure_capacity(actual_bs)
            self.bs = actual_bs
        tensor_list = []
        img_metas_list = []
        topo_idx_list = []
        is_first_frame_list = []
        for i in range(self.bs):
            if not self.img_metas_memory[i]:
                is_first_frame = True
            else:
                is_first_frame = (
                    img_metas[i]["scene_token"] != self.img_metas_memory[i]["scene_token"]
                )
            if is_first_frame:
                self.reset_single(i)
            tensor_list.append(self.memory_list[i])
            img_metas_list.append(self.img_metas_memory[i])
            is_first_frame_list.append(is_first_frame)
            topo_idx_list.append(
                self.train_topoidx_memory_list[i]
                if i < len(self.train_topoidx_memory_list)
                else None
            )
        return {
            "tensor": tensor_list,
            "img_metas": img_metas_list,
            "is_first_frame": is_first_frame_list,
            "topo_idx": topo_idx_list,
        }

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.bs = self.train_bs

    def eval(self):
        self.training = False

    def reset(self):
        self.train_memory_list = [None for i in range(self.train_bs)]
        self.train_img_metas_memory = [None for i in range(self.train_bs)]
        self.train_topoidx_memory_list = [None for i in range(self.train_bs)]
        self.test_memory_list = []
        self.test_img_metas_memory = []


class StreamWMLatentMemory:
    """Memory buffer for world-model BEV latent storage."""

    def __init__(self, batch_size):
        self.train_bs = batch_size
        self.training = True
        self.bs = self.train_bs
        self.train_memory_list = [None for i in range(self.bs)]
        self.train_img_metas_memory = [None for i in range(self.bs)]
        self.train_topoidx_memory_list = [None for i in range(self.bs)]
        self.test_memory_list = []
        self.test_img_metas_memory = []

    @property
    def memory_list(self):
        if self.training:
            return self.train_memory_list
        else:
            return self.test_memory_list

    @property
    def img_metas_memory(self):
        if self.training:
            return self.train_img_metas_memory
        else:
            return self.test_img_metas_memory

    def _ensure_capacity(self, size):
        while len(self.test_memory_list) < size:
            self.test_memory_list.append(None)
            self.test_img_metas_memory.append(None)

    def update(self, memory, img_metas, topo_idx=None):
        for i in range(self.bs):
            self.memory_list[i] = memory[i].clone().detach()
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])
            if topo_idx is not None and i < len(self.train_topoidx_memory_list):
                self.train_topoidx_memory_list[i] = topo_idx[i].clone().detach()

    def reset_single(self, idx):
        self.memory_list[idx] = None
        self.img_metas_memory[idx] = None
        if idx < len(self.train_topoidx_memory_list):
            self.train_topoidx_memory_list[idx] = None

    def get(self, img_metas):
        actual_bs = len(img_metas)
        if not self.training:
            self._ensure_capacity(actual_bs)
            self.bs = actual_bs
        tensor_list = []
        img_metas_list = []
        topo_idx_list = []
        is_first_frame_list = []
        for i in range(self.bs):
            if not self.img_metas_memory[i]:
                is_first_frame = True
            else:
                is_first_frame = (
                    img_metas[i]["scene_token"] != self.img_metas_memory[i]["scene_token"]
                )
            if is_first_frame:
                self.reset_single(i)
            tensor_list.append(self.memory_list[i])
            img_metas_list.append(self.img_metas_memory[i])
            is_first_frame_list.append(is_first_frame)
            topo_idx_list.append(
                self.train_topoidx_memory_list[i]
                if i < len(self.train_topoidx_memory_list)
                else None
            )
        return {
            "tensor": tensor_list,
            "img_metas": img_metas_list,
            "is_first_frame": is_first_frame_list,
            "topo_idx": topo_idx_list,
        }

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.bs = self.train_bs

    def eval(self):
        self.training = False

    def reset(self):
        self.train_memory_list = [None for i in range(self.train_bs)]
        self.train_img_metas_memory = [None for i in range(self.train_bs)]
        self.train_topoidx_memory_list = [None for i in range(self.train_bs)]
        self.test_memory_list = []
        self.test_img_metas_memory = []
