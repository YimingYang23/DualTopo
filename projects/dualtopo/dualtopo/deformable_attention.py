# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DualTopo inference layers: deformable attention. No MMCV dependency."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.mask2former.modeling_mask2former import (
    multi_scale_deformable_attention as transformers_msda,
)


@torch.compiler.disable
def _tensor_to_spatial_shapes_list(spatial_shapes: Tensor):
    """Convert spatial_shapes tensor to list-of-tuples outside the compiled graph.

    spatial_shapes is a fixed architectural constant (same values every step),
    so dynamo will specialize on its values after the first call. Isolating this
    conversion prevents the .tolist() graph break from fragmenting the main
    attention computation.
    """
    return spatial_shapes.tolist()


def multi_scale_deformable_attention(
    value: Tensor, spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    """
    Multi-scale deformable attention forward function.
    Wrapper around transformers implementation.

    The @_dynamo_disable decorator has been removed so torch.compile can trace
    the full attention computation (split + grid_sample + stack).
    spatial_shapes.tolist() is handled via a @torch.compiler.disable helper so
    it doesn't fragment the compiled attention graph.

    Args:
        value: (batch_size, num_keys, num_heads, head_dim)
        spatial_shapes: (num_levels, 2) or list of (height, width) tuples
        sampling_locations: (batch_size, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights: (batch_size, num_queries, num_heads, num_levels, num_points)

    Returns:
        output: (batch_size, num_queries, num_heads * head_dim)
    """
    if isinstance(spatial_shapes, Tensor):
        spatial_shapes_list = _tensor_to_spatial_shapes_list(spatial_shapes)
    else:
        spatial_shapes_list = spatial_shapes
    if sampling_locations.dtype != value.dtype:
        sampling_locations = sampling_locations.to(dtype=value.dtype)
    if attention_weights.dtype != value.dtype:
        attention_weights = attention_weights.to(dtype=value.dtype)
    return transformers_msda(value, spatial_shapes_list, sampling_locations, attention_weights)


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module.

    This wraps the transformers library implementation of multi-scale deformable attention.
    Used for BEVFormer and deformable DETR architectures.

    Args:
        embed_dims: Embedding dimension
        num_heads: Number of attention heads
        num_levels: Number of feature levels
        num_points: Number of sampling points per level
        dropout: Dropout rate
        batch_first: Whether batch dimension comes first
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper defaults for deformable attention."""
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor (batch_size, num_queries, embed_dims) or (num_queries, batch_size, embed_dims)
            key: Key tensor (not used in deformable attention)
            value: Value tensor (batch_size, num_keys, embed_dims)
            identity: Residual connection tensor
            query_pos: Positional encoding for query
            key_padding_mask: Mask for padding keys
            reference_points: Reference points for sampling (batch_size, num_queries, num_levels, 2)
            spatial_shapes: Shape of each level (num_levels, 2)
            level_start_index: Start index of each level

        Returns:
            output: (batch_size, num_queries, embed_dims)
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if not self.batch_first:
            query = query.transpose(0, 1)
            value = value.transpose(0, 1)
            identity = identity.transpose(0, 1)
        batch_size, num_queries, _ = query.shape
        if query_pos is not None:
            query = query + query_pos
        value_proj = self.value_proj(value)
        if key_padding_mask is not None:
            value_proj = value_proj.masked_fill(key_padding_mask[..., None], float(0))
        value_proj = value_proj.view(batch_size, -1, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points is None:
            raise ValueError("reference_points is required for deformable attention")
        if spatial_shapes is not None:
            offset_normalizer = (
                torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
                .to(sampling_offsets.device)
                .to(sampling_offsets.dtype)
            )
            sampling_locations = (
                reference_points[:, :, None, None, :, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            sampling_locations = reference_points[:, :, None, None, :, :] + sampling_offsets
        output = multi_scale_deformable_attention(
            value_proj, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        output = self.dropout(output) + identity
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output


class MSDeformableAttention3D(nn.Module):
    """
    Multi-scale deformable attention for 3D BEV queries.

    This extends MultiScaleDeformableAttention to handle 3D reference points
    with Z anchors, used in BEVFormer spatial cross-attention.

    The main difference is handling reference_points with shape:
    (batch_size, num_queries, num_Z_anchors, num_levels, 2)
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = None
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper defaults for deformable attention."""
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass for 3D deformable attention.

        Args:
            query: (batch_size, num_query, embed_dims)
            value: (batch_size, num_value, embed_dims)
            reference_points: usually (batch_size, num_query, num_Z_anchors, 2)
                from BEVFormer spatial cross-attention. A legacy 5D layout
                (batch_size, num_query, num_Z_anchors, num_levels, 2) is also supported.
            spatial_shapes: (num_levels, 2)

        Returns:
            output: (batch_size, num_query, embed_dims)
        """
        if value is None:
            value = query
        if not self.batch_first:
            query = query.transpose(0, 1)
            value = value.transpose(0, 1)
        batch_size, num_queries, _ = query.shape
        if query_pos is not None:
            query = query + query_pos
        value_proj = self.value_proj(value)
        if key_padding_mask is not None:
            value_proj = value_proj.masked_fill(key_padding_mask[..., None], float(0))
        value_proj = value_proj.view(batch_size, -1, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points is not None:
            if reference_points.dim() == 4:
                num_Z_anchors = reference_points.shape[2]
                if self.num_points % num_Z_anchors != 0:
                    raise ValueError(
                        f"num_points ({self.num_points}) must be divisible by num_Z_anchors ({num_Z_anchors})"
                    )
                offset_normalizer = (
                    torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
                    .to(sampling_offsets.device)
                    .to(sampling_offsets.dtype)
                )
                sampling_offsets = (
                    sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                )
                num_points_per_anchor = self.num_points // num_Z_anchors
                sampling_offsets = sampling_offsets.view(
                    batch_size,
                    num_queries,
                    self.num_heads,
                    self.num_levels,
                    num_points_per_anchor,
                    num_Z_anchors,
                    2,
                )
                reference_points_expand = reference_points[:, :, None, None, None, :, :]
                sampling_locations = reference_points_expand + sampling_offsets
                sampling_locations = sampling_locations.view(
                    batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
                )
            elif reference_points.dim() == 5:
                num_Z_anchors = reference_points.shape[2]
                reference_points_expand = reference_points.permute(0, 1, 3, 2, 4)
                reference_points_expand = reference_points_expand[:, :, None, :, :, :]
                offset_normalizer = (
                    torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
                    .to(sampling_offsets.device)
                    .to(sampling_offsets.dtype)
                )
                sampling_offsets = (
                    sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                )
                if self.num_points % num_Z_anchors != 0:
                    raise ValueError(
                        f"num_points ({self.num_points}) must be divisible by num_Z_anchors ({num_Z_anchors})"
                    )
                num_points_per_anchor = self.num_points // num_Z_anchors
                sampling_offsets = sampling_offsets.view(
                    batch_size,
                    num_queries,
                    self.num_heads,
                    self.num_levels,
                    num_points_per_anchor,
                    num_Z_anchors,
                    2,
                )
                sampling_locations = (
                    reference_points_expand[:, :, :, :, None, :, :] + sampling_offsets
                )
                sampling_locations = sampling_locations.view(
                    batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
                )
            else:
                raise ValueError(
                    f"reference_points must be 4D or 5D, got shape {tuple(reference_points.shape)}"
                )
        else:
            raise ValueError("reference_points is required for 3D deformable attention")
        output = multi_scale_deformable_attention(
            value_proj, spatial_shapes, sampling_locations, attention_weights
        )
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output
