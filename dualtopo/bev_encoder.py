# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DualTopo inference layers: bev encoder. No MMCV dependency."""

import copy
import math
import warnings
import torch
import torch.nn as nn
from .deformable_attention import multi_scale_deformable_attention, MSDeformableAttention3D


def inverse_sigmoid(x, eps=1e-05):
    """Inverse of the sigmoid function."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class FFN(nn.Module):
    """Feed-Forward Network."""

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        act_cfg=None,
        **kwargs,
    ):
        super().__init__()
        layers = []
        in_dim = embed_dims
        for i in range(num_fcs - 1):
            layers.append(nn.Linear(in_dim, feedforward_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(ffn_drop))
            in_dim = feedforward_channels
        layers.append(nn.Linear(in_dim, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return out + identity


class TemporalSelfAttention(nn.Module):
    """BEVFormer temporal self-attention using deformable attention.

    Fuses current BEV query with previous BEV features through learned
    sampling offsets.  Uses ``num_bev_queue=2`` (current + 1 history).
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        num_bev_queue=2,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        **kwargs,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

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
        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()

    def _init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

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
        **kwargs,
    ):
        """Forward Function of TemporalSelfAttention.

        Args:
            query: (bs, num_query, embed_dims)
            value: if None, defaults to stack of [query, query] (self-attention
                   on current bev).  Otherwise (bs*2, num_query, embed_dims)
                   = [prev_bev; cur_bev]
            reference_points: (bs*2, num_query, num_levels, 2) or
                (bs, num_query, num_levels, 2)
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
        """
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points
        )
        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(
                bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points
            )
            .contiguous()
        )
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        output = multi_scale_deformable_attention(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


class SpatialCrossAttention(nn.Module):
    """BEVFormer Spatial Cross-Attention.

    Rebatches BEV queries by camera, runs deformable attention per camera,
    then averages contributions.
    """

    def __init__(
        self,
        embed_dims=256,
        num_cams=7,
        pc_range=None,
        dropout=0.1,
        deformable_attention=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        if deformable_attention is not None:
            da_cfg = dict(deformable_attention)
            da_cfg.pop("type", None)
            self.deformable_attention = MSDeformableAttention3D(**da_cfg)
        else:
            self.deformable_attention = MSDeformableAttention3D(embed_dims=embed_dims)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward Function of SpatialCrossAttention.

        Args:
            query: (bs, num_query, embed_dims)
            key: (num_cam, num_value, bs, embed_dims) -- multi-camera features
            value: same as key
            reference_points_cam: (num_cam, bs, num_query, D, 2)
            bev_mask: (num_cam, bs, num_query, D)
            spatial_shapes: (num_levels, 2)
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        if max_len == 0:
            max_len = 1
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2]
        )
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                if len(index_query_per_img) > 0:
                    queries_rebatch[j, i, : len(index_query_per_img)] = query[
                        j, index_query_per_img
                    ]
                    reference_points_rebatch[j, i, : len(index_query_per_img)] = (
                        reference_points_per_img[j, index_query_per_img]
                    )
        num_cams, l, bs_kv, embed_dims_kv = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                if len(index_query_per_img) > 0:
                    slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1).to(dtype=slots.dtype)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


class BEVFormerLayer(nn.Module):
    """Single BEVFormer encoder layer.

    operation_order: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    """

    def __init__(
        self, attn_cfgs, ffn_cfgs, operation_order, embed_dims=256, norm_cfg=None, **kwargs
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.operation_order = operation_order
        self.attentions = nn.ModuleList()
        attn_index = 0
        for op in operation_order:
            if op in ("self_attn", "cross_attn"):
                cfg = attn_cfgs[attn_index] if isinstance(attn_cfgs, (list, tuple)) else attn_cfgs
                cfg = dict(cfg) if isinstance(cfg, dict) else cfg
                attn_type = cfg.pop("type", None)
                if attn_type == "TemporalSelfAttention":
                    self.attentions.append(TemporalSelfAttention(**cfg))
                elif attn_type == "SpatialCrossAttention":
                    self.attentions.append(SpatialCrossAttention(**cfg))
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                attn_index += 1
        self.ffns = nn.ModuleList()
        for op in operation_order:
            if op == "ffn":
                if isinstance(ffn_cfgs, (list, tuple)):
                    fcfg = ffn_cfgs[0] if isinstance(ffn_cfgs[0], dict) else ffn_cfgs[0]
                else:
                    fcfg = ffn_cfgs
                fcfg = dict(fcfg) if isinstance(fcfg, dict) else fcfg
                fcfg.pop("type", None)
                self.ffns.append(FFN(**fcfg))
        num_norms = operation_order.count("norm")
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_norms)])
        self.num_attn = len(self.attentions)
        self.pre_norm = operation_order[0] == "norm"

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        """Forward function for BEVFormerLayer.

        Args:
            query: (bs, num_queries, embed_dims) when batch_first else
                   (num_queries, bs, embed_dims)
            key: (num_cam, num_value, bs, embed_dims) -- multi-camera features
            value: same as key
            bev_pos: BEV positional encoding
            ref_2d: 2D reference points for temporal self-attention
            ref_3d: 3D reference points for spatial cross-attention
            reference_points_cam: projected 2D reference points per camera
            bev_mask: visibility mask per camera
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
            prev_bev: previous BEV features (if available)
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in {self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of attn_masks ({len(attn_masks)}) must be equal to the number of attention in operation_order ({self.num_attn})"
            )
        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


class BEVWorldModelLayer(nn.Module):
    """World-model BEVFormer layer.

    Only self-attention + FFN, no cross-attention.
    """

    def __init__(
        self, attn_cfgs, ffn_cfgs, operation_order, embed_dims=256, norm_cfg=None, **kwargs
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.operation_order = operation_order
        self.attentions = nn.ModuleList()
        attn_index = 0
        for op in operation_order:
            if op == "self_attn":
                cfg = attn_cfgs[attn_index] if isinstance(attn_cfgs, (list, tuple)) else attn_cfgs
                cfg = dict(cfg) if isinstance(cfg, dict) else cfg
                attn_type = cfg.pop("type", None)
                if attn_type == "TemporalSelfAttention":
                    self.attentions.append(TemporalSelfAttention(**cfg))
                else:
                    raise ValueError(f"Unknown attention type for WM layer: {attn_type}")
                attn_index += 1
        self.ffns = nn.ModuleList()
        for op in operation_order:
            if op == "ffn":
                if isinstance(ffn_cfgs, (list, tuple)):
                    fcfg = ffn_cfgs[0] if isinstance(ffn_cfgs[0], dict) else ffn_cfgs[0]
                else:
                    fcfg = ffn_cfgs
                fcfg = dict(fcfg) if isinstance(fcfg, dict) else fcfg
                fcfg.pop("type", None)
                self.ffns.append(FFN(**fcfg))
        num_norms = operation_order.count("norm")
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_norms)])
        self.num_attn = len(self.attentions)
        self.pre_norm = operation_order[0] == "norm"

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        spatial_shapes=None,
        level_start_index=None,
        reference_points_cam=None,
        bev_mask=None,
        prev_bev=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


class BEVFormerEncoder(nn.Module):
    """BEVFormer encoder: generates 3D/2D reference points, does point
    sampling, and runs through multiple BEVFormerLayer layers."""

    def __init__(
        self,
        num_layers=3,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=None,
        **kwargs,
    ):
        super().__init__()
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList()
        layer_cfg = dict(transformerlayers)
        layer_type = layer_cfg.pop("type", "BEVFormerLayer")
        embed_dims = 256
        if "attn_cfgs" in layer_cfg:
            for acfg in layer_cfg["attn_cfgs"]:
                if isinstance(acfg, dict) and "embed_dims" in acfg:
                    embed_dims = acfg["embed_dims"]
                    break
        layer_cfg["embed_dims"] = embed_dims
        for _ in range(num_layers):
            cfg_copy = copy.deepcopy(layer_cfg)
            if layer_type == "BEVFormerLayer":
                self.layers.append(BEVFormerLayer(**cfg_copy))
            elif layer_type == "BEVWorldModelLayer":
                self.layers.append(BEVWorldModelLayer(**cfg_copy))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

    @staticmethod
    def get_reference_points(
        H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, device="cuda", dtype=torch.float
    ):
        """Get the reference points used in SCA and TSA.

        Args:
            H, W: spatial shape of BEV.
            Z: height of pillar.
            num_points_in_pillar: sample this many points uniformly from each
                pillar.
            device: device for the tensors.
            dtype: dtype for the tensors.

        Returns:
            Tensor: reference points used in decoder, has shape
                (bs, num_keys, num_levels, 2) for '2d', or
                (bs, num_query, num_points_in_pillar, 3) for '3d'.
        """
        if dim == "3d":
            zs = (
                torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @staticmethod
    @torch.no_grad()
    def point_sampling(reference_points, pc_range, lidar2img_tensor, img_hw):
        """Project 3D reference points to 2D camera coordinates.

        Args:
            reference_points: (B, num_query, num_levels, 3)
            pc_range: list of 6 floats [x_min, y_min, z_min, x_max, y_max, z_max]
            lidar2img_tensor: (B, N_cam, 4, 4) float32 -- projection matrices
            img_hw: (B, N_cam, 4) -- valid H,W and padded H,W
        """
        lidar2img = lidar2img_tensor.to(
            device=reference_points.device, dtype=reference_points.dtype
        )
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)
        reference_points = (
            reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        )
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-05
        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )
        sizes = img_hw.to(device=reference_points_cam.device, dtype=reference_points_cam.dtype)
        assert sizes.shape == (B, num_cam, 4), sizes.shape
        assert torch.all(sizes[..., :2] > 0)
        assert torch.all(sizes[..., 2:] >= sizes[..., :2])
        valid_wh = sizes[..., [1, 0]].view(1, B, num_cam, 1, 2)
        padded_wh = sizes[..., [3, 2]].view(1, B, num_cam, 1, 2)
        bev_mask = (
            bev_mask
            & (reference_points_cam > 0).all(-1, keepdim=True)
            & (reference_points_cam < valid_wh).all(-1, keepdim=True)
        )
        reference_points_cam = reference_points_cam / padded_wh
        bev_mask = torch.nan_to_num(bev_mask)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return (reference_points_cam, bev_mask)

    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):
        """Forward function for BEVFormerEncoder.

        Args:
            bev_query: (num_query, bs, embed_dims)
            key & value: (num_cam, num_value, bs, embed_dims)
            bev_pos: (num_query, bs, embed_dims)
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
            prev_bev: (num_query, bs, embed_dims) or None
            shift: (bs, 2) ego motion shift

        Returns:
            Tensor: Results with shape [num_layers, bs, num_query, embed_dims]
                when return_intermediate is True, otherwise
                [bs, num_query, embed_dims].
        """
        output = bev_query
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs["lidar2img_tensor"], kwargs["img_hw"]
        )
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class BEVWorldModelEncoder(nn.Module):
    """World-model BEVFormer encoder.

    Only uses temporal self-attention + FFN.
    """

    def __init__(
        self,
        num_layers=2,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=None,
        **kwargs,
    ):
        super().__init__()
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList()
        layer_cfg = dict(transformerlayers)
        layer_cfg.pop("type", "BEVWorldModelLayer")
        embed_dims = 256
        if "attn_cfgs" in layer_cfg:
            attn_cfgs_list = (
                layer_cfg["attn_cfgs"]
                if isinstance(layer_cfg["attn_cfgs"], list)
                else [layer_cfg["attn_cfgs"]]
            )
            for acfg in attn_cfgs_list:
                if isinstance(acfg, dict) and "embed_dims" in acfg:
                    embed_dims = acfg["embed_dims"]
                    break
        layer_cfg["embed_dims"] = embed_dims
        for _ in range(num_layers):
            cfg_copy = copy.deepcopy(layer_cfg)
            self.layers.append(BEVWorldModelLayer(**cfg_copy))

    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):
        output = bev_query
        ref_2d = BEVFormerEncoder.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        shift_ref_2d = ref_2d.clone()
        if isinstance(shift, torch.Tensor) and shift.dim() > 0:
            shift_ref_2d += shift[:, None, None, :]
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        intermediate = []
        for layer in self.layers:
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                **kwargs,
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class CustomMSDeformableAttention(nn.Module):
    """Multi-scale deformable attention used in the Deformable DETR decoder.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query in each
            head. Default: 4.
        im2col_step (int): The step used in image_to_column. Default: 64.
        dropout (float): A Dropout layer on `inp_identity`. Default: 0.1.
        batch_first (bool): Whether batch dimension comes first. Default: False.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        **kwargs,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.im2col_step = im2col_step
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
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()

    def _init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
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
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

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
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query: (num_query, bs, embed_dims) if not batch_first,
                else (bs, num_query, embed_dims).
            key: (num_key, bs, embed_dims) or (bs, num_key, embed_dims).
            value: same shape as key.
            identity: residual tensor. Default None -> query is used.
            query_pos: positional encoding for query.
            reference_points: (bs, num_query, num_levels, 2) or
                (bs, num_query, num_levels, 4).
            spatial_shapes: (num_levels, 2).
            level_start_index: (num_levels,).

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims]
                (or [bs, num_query, embed_dims] if batch_first).
        """
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
        _, num_value, _ = value.shape
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
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        output = multi_scale_deformable_attention(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


class DetectionTransformerDecoderLayer(nn.Module):
    """A single transformer decoder layer used inside DetectionTransformerDecoder.

    Consists of self-attention + norm + cross-attention (MS deformable) + norm
    + FFN + norm.

    Args:
        embed_dims: embedding dimension.
        num_heads: number of attention heads.
        num_levels: number of feature levels for deformable attention.
        num_points: number of sampling points per level per head.
        feedforward_channels: FFN inner dimension.
        num_fcs: number of FC layers in FFN.
        ffn_drop: dropout rate in FFN.
        dropout: dropout rate for attention.
        batch_first: whether batch dim comes first.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        dropout=0.1,
        batch_first=False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=batch_first
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.cross_attn = CustomMSDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=ffn_drop,
        )
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            query: (num_query, bs, embed_dims) if not batch_first
            key/value: BEV features for cross-attention
            query_pos: positional encoding
            reference_points: (bs, num_query, num_levels, 2)
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
        """
        identity = query
        q = k = query + query_pos if query_pos is not None else query
        query_attn = self.self_attn(
            q,
            k,
            query if not self.batch_first else query,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        query = self.norm1(identity + self.dropout_layer(query_attn))
        identity = query
        query_cross = self.cross_attn(
            query,
            key,
            value,
            identity=identity,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs,
        )
        query = self.norm2(query_cross)
        identity = query
        query = self.norm3(self.ffn(query, identity))
        return query


class DetectionTransformerDecoder(nn.Module):
    """Transformer decoder for detection heads.

    Iteratively refines object queries through multiple decoder layers with
    iterative bounding box refinement.

    Args:
        num_layers (int): Number of decoder layers.
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature levels for deformable attention.
        num_points (int): Number of sampling points.
        feedforward_channels (int): FFN inner dimension.
        num_fcs (int): Number of FC layers in FFN.
        ffn_drop (float): Dropout in FFN.
        dropout (float): Dropout for attention.
        batch_first (bool): Whether batch dim is first.
        return_intermediate (bool): Whether to return intermediate outputs.
    """

    def __init__(
        self,
        num_layers=6,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        dropout=0.1,
        batch_first=False,
        return_intermediate=False,
        **kwargs,
    ):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.batch_first = batch_first
        self.layers = nn.ModuleList(
            [
                DetectionTransformerDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                    feedforward_channels=feedforward_channels,
                    num_fcs=num_fcs,
                    ffn_drop=ffn_drop,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = (
            nn.LayerNorm(embed_dims) if kwargs.get("norm_cfg", None) is not None or True else None
        )

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for DetectionTransformerDecoder.

        Args:
            query: (num_query, bs, embed_dims) -- Input query.
            reference_points: (bs, num_query, 3) -- The reference points.
                Contains (x, y, z) for each query.
            reg_branches: (obj:`nn.ModuleList`) -- Used for refining the
                regression results. Each element is an nn.Module that maps
                (bs, num_query, embed_dims) -> (bs, num_query, 6+)
                where outputs[..., :2] are xy offsets and outputs[..., 4:5]
                is the z offset.
            key_padding_mask: ByteTensor with shape [bs, num_key].

        Returns:
            tuple[Tensor]:
                - output or intermediate stack: decoder output(s)
                - reference_points or intermediate_reference_points stack:
                  reference points (original or refined)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 3
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(
                    reference_points[..., 2:3]
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return (torch.stack(intermediate), torch.stack(intermediate_reference_points))
        return (output, reference_points)
