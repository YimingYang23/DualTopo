# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""Shared Dual-Pathway Decoder for init and temporal inference."""

import copy
import math
import torch
import torch.nn as nn
from .bev_encoder import FFN, inverse_sigmoid
from .lane_attention import LaneAttention, StreamLaneAttention


def _xavier_init(module, bias=0.0):
    """Xavier uniform initialization for an nn.Linear (or nn.Sequential)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, nn.Sequential):
        for m in module:
            _xavier_init(m, bias=bias)


def gen_sineembed_for_position(pos_tensor):
    """Generate sinusoidal positional embeddings from 2-D or 4-D coords."""
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class CustomDetrTransformerDecoderLayer(nn.Module):
    """A single transformer decoder layer for lane-segment decoding.

    Operation order (6 items):
      self_attn -> norm -> cross_attn -> norm -> ffn -> norm

    self_attn uses standard nn.MultiheadAttention.
    cross_attn uses LaneAttention (deformable attention on BEV features).
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=32,
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
        self.cross_attn = LaneAttention(
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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
        identity = query
        q = k = query + query_pos if query_pos is not None else query
        attn_mask = kwargs.get("attn_masks", None)
        if isinstance(attn_mask, (list, tuple)):
            attn_mask = attn_mask[0] if len(attn_mask) > 0 else None
        query_attn = self.self_attn(
            q, k, query, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        query = self.norm1(identity + self.dropout1(query_attn))
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


class StreamCustomDetrTransformerDecoderLayer(nn.Module):
    """Streaming variant of the decoder layer.

    Uses StreamLaneAttention instead of LaneAttention for cross-attention.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=32,
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
        self.cross_attn = StreamLaneAttention(
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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
        pt_query_pos=None,
        query_key_padding_mask=None,
        **kwargs,
    ):
        identity = query
        q = k = query + query_pos if query_pos is not None else query
        attn_mask = kwargs.get("attn_masks", None)
        if isinstance(attn_mask, (list, tuple)):
            attn_mask = attn_mask[0] if len(attn_mask) > 0 else None
        query_attn = self.self_attn(
            q,
            k,
            query,
            attn_mask=attn_mask,
            key_padding_mask=query_key_padding_mask,
            need_weights=False,
        )[0]
        query = self.norm1(identity + self.dropout1(query_attn))
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
            pt_query_pos=pt_query_pos,
            **kwargs,
        )
        query = self.norm2(query_cross)
        identity = query
        query = self.norm3(self.ffn(query, identity))
        return query


class DualPathwayDecoder(nn.Module):
    """Dual-Pathway Decoder. Shared layers produce init and temporal predictions; history enters after the first layer."""

    def __init__(
        self,
        num_layers=6,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=32,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        dropout=0.1,
        batch_first=False,
        return_intermediate=False,
        prop_add_stage=0,
        pc_range=None,
        sample_idx=None,
        pts_dim=3,
        **kwargs,
    ):
        super().__init__()
        if sample_idx is None:
            sample_idx = [0, 3, 6, 9]
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.sample_idx = sample_idx
        self.pts_dim = pts_dim
        self.prop_add_stage = prop_add_stage
        self.embed_dims = embed_dims
        assert prop_add_stage >= 0 and prop_add_stage < 6
        self.layers = nn.ModuleList(
            [
                StreamCustomDetrTransformerDecoderLayer(
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

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        pt_pos_query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.pt_pos_query_projs = _get_clones(pt_pos_query_proj, num_layers)
        self.init_weights()

    def init_weights(self):
        for m in self.pt_pos_query_projs:
            _xavier_init(m, bias=0.0)

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        dn_centerline_reference_points=None,
        dn_lane_reference_points=None,
        reg_branches=None,
        cls_branches=None,
        key_padding_mask=None,
        prop_query=None,
        prop_reference_points=None,
        prop_centerline_reference_points=None,
        is_first_frame_list=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_lane_ref_points = []
        init_inter_reference_points = None
        init_inter_lane_reference_points = None
        num_queries, bs, embed_dims = query.shape
        if "value" in kwargs and isinstance(kwargs["value"], torch.Tensor):
            value = kwargs["value"]
            kwargs = dict(kwargs)
        else:
            value = None
        value_bev = value[:, :bs, :] if value is not None else None
        lane_ref_points = reference_points[:, :, self.sample_idx * 2, :]
        init_reference_out = reference_points
        for lid, layer in enumerate(self.layers):
            if (
                lid == self.prop_add_stage
                and prop_query is not None
                and (prop_reference_points is not None)
            ):
                bs, topk, embed_dims = prop_query.shape
                output = output.permute(1, 0, 2)
                with torch.no_grad():
                    tmp_scores, _ = cls_branches[lid](output).max(-1)
                new_query = []
                new_refpts = []
                new_centerline_refpts = []
                for i in range(bs):
                    if is_first_frame_list[i]:
                        new_query.append(output[i])
                        new_refpts.append(lane_ref_points[i])
                        new_centerline_refpts.append(reference_points[i])
                    else:
                        _, valid_idx = torch.topk(tmp_scores[i], k=num_queries - topk, dim=-1)
                        new_query.append(torch.cat([prop_query[i], output[i][valid_idx]], dim=0))
                        new_refpts.append(
                            torch.cat(
                                [prop_reference_points[i], lane_ref_points[i][valid_idx]], dim=0
                            )
                        )
                        new_centerline_refpts.append(
                            torch.cat(
                                [
                                    prop_centerline_reference_points[i],
                                    reference_points[i][valid_idx],
                                ],
                                dim=0,
                            )
                        )
                lane_ref_points_stream_part = torch.stack(new_refpts)
                output_stream_part = torch.stack(new_query).permute(1, 0, 2)
                reference_points_stream_part = torch.stack(new_centerline_refpts)
            if (
                lid == self.prop_add_stage
                and prop_query is not None
                and (prop_reference_points is not None)
            ):
                lane_ref_points = torch.cat((lane_ref_points, lane_ref_points_stream_part), dim=0)
                output = torch.cat((output.permute(1, 0, 2), output_stream_part), dim=1)
                reference_points = torch.cat(
                    (reference_points, reference_points_stream_part), dim=0
                )
                init_inter_reference_points = reference_points
                init_inter_lane_reference_points = lane_ref_points
                value_bev = value
            lane_ref_points_reshape = lane_ref_points[..., :2]
            bs = lane_ref_points_reshape.shape[0]
            lane_ref_points_reshape = lane_ref_points_reshape.view(bs, -1, 2)
            query_sine_embed = gen_sineembed_for_position(lane_ref_points_reshape[..., :2])
            query_sine_embed = query_sine_embed.view(bs, -1, 8, 256)
            point_query_pos = self.pt_pos_query_projs[lid](query_sine_embed)
            reference_points_input = lane_ref_points_reshape[..., :2].unsqueeze(2)
            _layer_kw = {k: v for k, v in kwargs.items() if k not in ("key", "value")}
            output = layer(
                output,
                None,
                value_bev,
                *args,
                pt_query_pos=point_query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                query_key_padding_mask=None,
                **_layer_kw,
            )
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                reg_center = reg_branches[0]
                reg_offset = reg_branches[1]
                tmp = reg_center[lid](output)
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)
                assert reference_points.shape[-1] == self.pts_dim
                tmp = tmp + inverse_sigmoid(reference_points)
                tmp = tmp.sigmoid()
                reference_points = tmp.detach()
                coord = tmp.clone()
                coord[..., 0] = (
                    coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                )
                coord[..., 1] = (
                    coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                )
                if self.pts_dim == 3:
                    coord[..., 2] = (
                        coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                    )
                centerline = coord.view(bs, num_query, -1).contiguous()
                offset = reg_offset[lid](output)
                left_laneline = centerline + offset
                right_laneline = centerline - offset
                left_laneline = left_laneline.view(bs, num_query, -1, self.pts_dim)[
                    :, :, self.sample_idx, :
                ]
                right_laneline = right_laneline.view(bs, num_query, -1, self.pts_dim)[
                    :, :, self.sample_idx, :
                ]
                lane_ref_points = (
                    torch.cat([left_laneline, right_laneline], axis=-2).contiguous().detach()
                )
                lane_ref_points[..., 0] = (lane_ref_points[..., 0] - self.pc_range[0]) / (
                    self.pc_range[3] - self.pc_range[0]
                )
                lane_ref_points[..., 1] = (lane_ref_points[..., 1] - self.pc_range[1]) / (
                    self.pc_range[4] - self.pc_range[1]
                )
                if self.pts_dim == 3:
                    lane_ref_points[..., 2] = (lane_ref_points[..., 2] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2]
                    )
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points.clone().detach())
                intermediate_lane_ref_points.append(lane_ref_points.clone().detach())
        if self.return_intermediate:
            return (
                intermediate[0].unsqueeze(0),
                torch.stack(intermediate[1:]),
                init_reference_out,
                init_inter_reference_points,
                torch.stack(intermediate_reference_points[1:]),
                init_inter_lane_reference_points,
                torch.stack(intermediate_lane_ref_points[1:]),
            )
        return (output, reference_points)


class DualPathwayTransformer(nn.Module):
    """Full transformer wrapper for lane-segment decoding with streaming.

    Receives reference points externally (no internal reference_points layer).
    Supports DN queries, propagated queries, and per-sample first-frame flags.

    Args:
        decoder: DualPathwayDecoder instance.
        embed_dims: embedding dimension.
        points_num: number of repeated reference points per query.
        pts_dim: dimensionality of points (2 or 3).
    """

    def __init__(self, decoder=None, embed_dims=256, points_num=1, pts_dim=3, **kwargs):
        super().__init__()
        if decoder is not None:
            if isinstance(decoder, dict):
                dec_cfg = decoder.copy()
                dec_type = dec_cfg.pop("type", "DualPathwayDecoder")
                if dec_type == "DualPathwayDecoder":
                    self.decoder = DualPathwayDecoder(**dec_cfg)
                elif dec_type == "LaneSegNetDecoder":
                    raise ValueError("This release supports only DualPathwayDecoder")
                else:
                    raise ValueError(f"Unknown decoder type: {dec_type}")
            else:
                self.decoder = decoder
        else:
            self.decoder = DualPathwayDecoder(embed_dims=embed_dims)
        self.embed_dims = embed_dims
        self.points_num = points_num
        self.pts_dim = pts_dim

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, StreamLaneAttention):
                m.init_weights()
            elif isinstance(m, LaneAttention):
                m.init_weights()

    def forward(
        self,
        mlvl_feats,
        bev_embed,
        object_query_embed,
        reference_points,
        bev_h,
        bev_w,
        reg_branches=None,
        cls_branches=None,
        prop_query=None,
        prop_reference_points=None,
        prop_centerline_reference_points=None,
        is_first_frame_list=None,
        attn_masks=None,
        **kwargs,
    ):
        mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=2)
        query_pos = None
        query = query.permute(1, 0, 2)
        if query_pos is not None:
            query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        (
            init_hs,
            inter_hs,
            init_reference_out,
            init_inter_reference_points,
            inter_references,
            init_inter_lane_reference_points,
            inter_lane_reference,
        ) = self.decoder(
            query=query,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            is_first_frame_list=is_first_frame_list,
            prop_query=prop_query,
            prop_reference_points=prop_reference_points,
            prop_centerline_reference_points=prop_centerline_reference_points,
            attn_masks=attn_masks,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )
        return (
            init_hs,
            inter_hs,
            init_reference_out,
            init_inter_reference_points,
            inter_references,
            init_inter_lane_reference_points,
            inter_lane_reference,
        )
