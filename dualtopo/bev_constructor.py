# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DualTopo inference layers: bev constructor. No MMCV dependency."""

import copy
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from .bev_encoder import BEVFormerEncoder, BEVWorldModelEncoder


class LearnedPositionalEncoding(nn.Module):
    """2D Learned positional encoding for BEV plane.

    Args:
        num_feats: Half of the embedding dimension (row/col each contribute num_feats).
        row_num_embed: Number of row embeddings.
        col_num_embed: Number of column embeddings.
    """

    def __init__(self, num_feats=128, row_num_embed=100, col_num_embed=200, **kwargs):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask):
        """
        Args:
            mask: (bs, H, W) -- used only to determine spatial size and device.
        Returns:
            (bs, num_feats*2, H, W)
        """
        h, w = mask.shape[-2:]
        x_embed = self.col_embed(torch.arange(w, device=mask.device))
        y_embed = self.row_embed(torch.arange(h, device=mask.device))
        pos = torch.cat(
            [x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)], dim=-1
        )
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        pos = pos.repeat(mask.shape[0], 1, 1, 1)
        return pos


def _build_encoder(encoder_cfg):
    """Build a BEVFormerEncoder or BEVWorldModelEncoder from config dict."""
    cfg = copy.deepcopy(encoder_cfg)
    enc_type = cfg.pop("type", "BEVFormerEncoder")
    if enc_type == "BEVFormerEncoder":
        return BEVFormerEncoder(**cfg)
    elif enc_type == "BEVWorldModelEncoder":
        return BEVWorldModelEncoder(**cfg)
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")


def _build_positional_encoding(pe_cfg):
    """Build positional encoding from config dict."""
    cfg = copy.deepcopy(pe_cfg)
    pe_type = cfg.pop("type", "LearnedPositionalEncoding")
    if pe_type == "LearnedPositionalEncoding":
        return LearnedPositionalEncoding(**cfg)
    else:
        raise ValueError(f"Unknown positional encoding type: {pe_type}")


class BEVFormerConstructer(nn.Module):
    """Builds BEV features from multi-level multi-camera image features.

    Pipeline:
    1. Flatten multi-level features, add level/camera embeddings
    2. Generate BEV queries from learned embedding
    3. Add can_bus (ego motion) signal to BEV queries
    4. Rotate previous BEV if available
    5. Run through BEVFormerEncoder
    """

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=7,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        pc_range=None,
        bev_h=200,
        bev_w=200,
        rotate_center=None,
        encoder=None,
        positional_encoding=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.rotate_center = rotate_center if rotate_center else [bev_h // 2, bev_w // 2]
        self.encoder = _build_encoder(encoder)
        self.positional_encoding = _build_positional_encoding(positional_encoding)
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
        )
        if can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(embed_dims))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        for p in self.can_bus_mlp.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)
        for m in self.encoder.modules():
            if hasattr(m, "_init_weights"):
                m._init_weights()

    def forward(
        self,
        mlvl_feats,
        can_bus_tensor,
        lidar2global_rot,
        lidar2img_tensor,
        img_hw,
        prev_bev=None,
        **kwargs,
    ):
        """
        Args:
            mlvl_feats: list of (B, N_cam, C, H, W)
            can_bus_tensor: (B, 18) float32 -- ego-motion signal
            lidar2global_rot: (B, 3, 3) float32 -- lidar-to-global rotation
            lidar2img_tensor: (B, N_cam, 4, 4) float32 -- projection matrices
            img_hw: (2,) float32 -- [H, W] image size (constant)
        """
        import numpy as np

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        delta_global = can_bus_tensor[:, :3].cpu().numpy().astype(np.float64)
        lidar2global_rot_np = lidar2global_rot.cpu().numpy().astype(np.float64)
        delta_lidar = np.array(
            [np.linalg.inv(lidar2global_rot_np[i]) @ delta_global[i] for i in range(bs)]
        )
        shift_x = delta_lidar[:, 0] / self.real_w * self.use_shift
        shift_y = delta_lidar[:, 1] / self.real_h * self.use_shift
        shift = bev_queries.new_tensor(np.stack([shift_x, shift_y], axis=1)).to(dtype)
        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = can_bus_tensor[i, -1].item()
                    tmp_prev_bev = (
                        prev_bev[:, i].reshape(self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                    )
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        self.bev_h * self.bev_w, 1, -1
                    )
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
        can_bus_enc = self.can_bus_mlp(can_bus_tensor.to(device=device, dtype=dtype))[None, :, :]
        bev_queries = bev_queries + can_bus_enc * self.use_can_bus
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs_f, num_cam_f, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            lidar2img_tensor=lidar2img_tensor,
            img_hw=img_hw,
            **kwargs,
        )
        return bev_embed


class BEVWorldModel(nn.Module):
    """World-model BEV constructor. Takes pre-computed BEV queries (action-aware latent)
    instead of camera features. Only uses temporal self-attention."""

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=7,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        pc_range=None,
        bev_h=50,
        bev_w=100,
        rotate_center=None,
        encoder=None,
        positional_encoding=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.rotate_center = rotate_center if rotate_center else [bev_h // 2, bev_w // 2]
        self.encoder = _build_encoder(encoder)
        self.positional_encoding = _build_positional_encoding(positional_encoding)
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
        )
        if can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(embed_dims))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        for p in self.can_bus_mlp.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)
        for m in self.encoder.modules():
            if hasattr(m, "_init_weights"):
                m._init_weights()

    def forward(self, bev_queries, img_metas, prev_bev=None, **kwargs):
        """
        Args:
            bev_queries: (num_query, bs, embed_dims) -- action-aware BEV latent
            img_metas: list of dicts with 'can_bus' and 'lidar2global_rotation'
        """
        import numpy as np

        bs = bev_queries.shape[1]
        dtype = bev_queries.dtype
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        delta_global = np.array([each["can_bus"][:3] for each in img_metas])
        lidar2global_rotation = np.array([each["lidar2global_rotation"] for each in img_metas])
        delta_lidar = []
        for i in range(bs):
            delta_lidar.append(np.linalg.inv(lidar2global_rotation[i]) @ delta_global[i])
        delta_lidar = np.array(delta_lidar)
        shift_x = delta_lidar[:, 0] / self.real_w
        shift_y = delta_lidar[:, 1] / self.real_h
        shift_x = shift_x * self.use_shift
        shift_y = shift_y * self.use_shift
        shift = bev_queries.new_tensor(np.stack([shift_x, shift_y], axis=1)).to(dtype)
        can_bus = bev_queries.new_tensor(np.array([each["can_bus"] for each in img_metas]))
        can_bus_enc = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus_enc * self.use_can_bus
        bev_embed = self.encoder(
            bev_queries,
            None,
            None,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=None,
            level_start_index=None,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )
        return bev_embed
