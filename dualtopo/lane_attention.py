# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DualTopo inference layers: lane attention. No MMCV dependency."""

import warnings
import math
import torch
import torch.nn as nn
from .deformable_attention import multi_scale_deformable_attention


def _xavier_init(module, bias=0.0):
    """Xavier uniform initialization for an nn.Linear (or nn.Sequential of Linears)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, nn.Sequential):
        for m in module:
            _xavier_init(m, bias=bias)


def _constant_init(module, val=0.0, bias=0.0):
    """Constant initialization for an nn.Linear."""
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, val)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, nn.Sequential):
        for m in module:
            _constant_init(m, val=val, bias=bias)


class LaneAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=32,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        def _is_power_of_2(n):
            if not isinstance(n, int) or n < 0:
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return n & n - 1 == 0 and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in MultiScaleDeformAttention to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation."
            )
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        _constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(1, 1, 8, 1, 2)
            .repeat(self.num_heads, self.num_levels, 1, self.num_points // 8, 1)
        )
        for i in range(self.num_points // 8):
            grid_init[:, :, :, i, :] *= i + 1
        grid_init = grid_init.view(self.num_heads, self.num_levels, self.num_points, 2)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        _constant_init(self.attention_weights, val=0.0, bias=0.0)
        _xavier_init(self.value_proj, bias=0.0)
        _xavier_init(self.output_proj, bias=0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            reference_points = reference_points.permute(0, 1, 3, 2, 4)
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, :, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2, but get {reference_points.shape[-1]} instead."
            )
        output = multi_scale_deformable_attention(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


class StreamLaneAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=32,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        def _is_power_of_2(n):
            if not isinstance(n, int) or n < 0:
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return n & n - 1 == 0 and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in MultiScaleDeformAttention to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation."
            )
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        num_pts_per_vec = 8
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims * num_pts_per_vec, embed_dims * num_pts_per_vec // 2),
            nn.ReLU(),
            nn.Linear(embed_dims * num_pts_per_vec // 2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        _constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(1, 1, 8, 1, 2)
            .repeat(self.num_heads, self.num_levels, 1, self.num_points // 8, 1)
        )
        for i in range(self.num_points // 8):
            grid_init[:, :, :, i, :] *= i + 1
        grid_init = grid_init.view(self.num_heads, self.num_levels, self.num_points, 2)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        _constant_init(self.attention_weights, val=0.0, bias=0.0)
        _xavier_init(self.value_proj, bias=0.0)
        _xavier_init(self.output_proj, bias=0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        pt_query_pos=None,
        flag="decoder",
        **kwargs,
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        if pt_query_pos is not None:
            bs, num_iq, num_pt, _ = pt_query_pos.shape
            query = query.unsqueeze(2) + pt_query_pos
            query = query.reshape(bs, num_iq * num_pt, -1)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2, but get {reference_points.shape[-1]} instead."
            )
        output = multi_scale_deformable_attention(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = output.view(bs, num_iq, num_pt, -1)
        output = output.flatten(2)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
