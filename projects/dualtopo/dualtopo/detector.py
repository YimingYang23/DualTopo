# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DualTopo inference entry. Always use forward_test so ego-motion deltas and scene resets are applied."""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNetBackbone, FPN
from .bev_constructor import BEVFormerConstructer
from .heads import DualPathwayLaneHead, RelationshipHead
from .memory_buffer import StreamWMLatentMemory
from .gru import ConvGRU


def image_geometry_tensor(img_metas, device):
    """Per sample/camera: valid H,W and actual padded H,W."""
    return torch.tensor(
        [
            [
                [float(v[0]), float(v[1]), float(p[0]), float(p[1])]
                for v, p in zip(meta["img_shape"], meta["pad_shape"])
            ]
            for meta in img_metas
        ],
        dtype=torch.float32,
        device=device,
    )


class DualTopo(nn.Module):
    """DualTopo: multi-view lane segment graph inference with QueryWorldModel, BEVWorldModel and a shared Dual-Pathway Decoder."""

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"] if "model" in cfg else cfg
        self.bev_h = model_cfg["bev_h"]
        self.bev_w = model_cfg["bev_w"]
        self.roi_size = model_cfg.get("roi_size", (102.4, 51.2, 4.0))
        self.embed_dims = model_cfg.get("embed_dims", 256)
        bb_cfg = model_cfg["img_backbone"]
        self.img_backbone = ResNetBackbone(
            out_indices=tuple(bb_cfg.get("out_indices", (1, 2, 3))),
            frozen_stages=bb_cfg.get("frozen_stages", 1),
            pretrained=bb_cfg.get("pretrained", True),
        )
        neck_cfg = model_cfg["img_neck"]
        self.img_neck = FPN(
            in_channels=neck_cfg["in_channels"],
            out_channels=neck_cfg["out_channels"],
            num_outs=neck_cfg["num_outs"],
            start_level=neck_cfg.get("start_level", 0),
            add_extra_convs=neck_cfg.get("add_extra_convs", "on_output"),
            relu_before_extra_convs=neck_cfg.get("relu_before_extra_convs", True),
        )
        bev_cfg = model_cfg.get("bev_constructor", {}).copy()
        bev_cfg.pop("type", None)
        self.bev_constructor = BEVFormerConstructer(**bev_cfg)
        if model_cfg.get("lane_head"):
            lane_cfg = model_cfg["lane_head"].copy()
            lane_cfg.pop("type", None)
            self.lane_head = DualPathwayLaneHead(**lane_cfg)
        else:
            self.lane_head = None
        if model_cfg.get("lclc_head"):
            lclc_cfg = model_cfg["lclc_head"].copy()
            lclc_cfg.pop("type", None)
            self.lclc_head = RelationshipHead(**lclc_cfg)
        else:
            self.lclc_head = None
        streaming_cfg = model_cfg.get("streaming_cfg", {})
        self.streaming_bev = streaming_cfg.get("streaming_bev", False)
        self.stream = streaming_cfg.get("stream", False)
        if self.streaming_bev:
            self.stream_fusion_neck = ConvGRU(
                out_channels=streaming_cfg.get("fusion_cfg", {}).get(
                    "out_channels", self.embed_dims
                )
            )
            self.stream_fusion_neck.init_weights()
            batch_size = streaming_cfg.get("batch_size", 1)
            self.bev_memory = StreamWMLatentMemory(batch_size)
            xmin, xmax = (-self.roi_size[0] / 2, self.roi_size[0] / 2)
            ymin, ymax = (-self.roi_size[1] / 2, self.roi_size[1] / 2)
            x = torch.linspace(xmin, xmax, self.bev_w)
            y = torch.linspace(ymax, ymin, self.bev_h)
            y, x = torch.meshgrid(y, x, indexing="ij")
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)
            self.register_buffer("plane", plane.double())
        self.video_test_mode = model_cfg.get("video_test_mode", False)
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def init_weights(self):
        """Initialize weights for all sub-modules that define ``init_weights``."""
        for module in self.modules():
            if module is not self and hasattr(module, "init_weights"):
                module.init_weights()

    def wm_prediction(self, view_query_feat, pose_memory, img_metas):
        """Predict future BEV features using the World Model.

        Args:
            view_query_feat: (C, H, W) BEV feature from memory buffer.
            pose_memory: dict with pose info of the previous frame.
            img_metas: dict with pose info of the current frame.

        Returns:
            wm_next_latent: (C, H, W) predicted BEV feature.
        """
        view_query_feat = view_query_feat.permute(1, 2, 0).unsqueeze(0)
        view_query_feat = view_query_feat.permute(0, 3, 1, 2)
        view_query_feat = self.lane_head.down_sample(view_query_feat)
        view_query_feat = view_query_feat.flatten(2).transpose(1, 2)
        batch_size, num_tokens, num_channel = view_query_feat.shape
        prev_e2g_trans = view_query_feat.new_tensor(
            pose_memory["lidar2global_translation"], dtype=torch.float64
        )
        prev_e2g_rot = view_query_feat.new_tensor(
            pose_memory["lidar2global_rotation"], dtype=torch.float64
        )
        curr_e2g_trans = view_query_feat.new_tensor(
            img_metas["lidar2global_translation"], dtype=torch.float64
        )
        curr_e2g_rot = view_query_feat.new_tensor(
            img_metas["lidar2global_rotation"], dtype=torch.float64
        )
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(view_query_feat.device)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans
        curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(view_query_feat.device)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
        pos_encoding = prev2curr_matrix.float()[:3].view(-1)
        pos_encoding = pos_encoding.view(1, -1).repeat(1, num_tokens, 1)
        cur_view_query_feat_with_ego = torch.cat([view_query_feat, pos_encoding], dim=-1)
        action_aware_latent = self.lane_head.action_aware_bev_encoder(cur_view_query_feat_with_ego)
        action_aware_latent = action_aware_latent.permute(1, 0, 2)
        pose_memory_list = [pose_memory]
        wm_next_latent_downsample = self.lane_head.bev_world_model(
            action_aware_latent, pose_memory_list, prev_bev=None
        )
        wm_next_latent_downsample = wm_next_latent_downsample.transpose(1, 2).reshape(
            batch_size, num_channel, int(self.bev_h // 2), int(self.bev_w // 2)
        )
        wm_next_latent = F.interpolate(
            wm_next_latent_downsample,
            size=(self.bev_h, self.bev_w),
            mode="bilinear",
            align_corners=False,
        )
        return wm_next_latent[0]

    def update_bev_feature(self, curr_bev_feats, img_metas):
        """Fuse current BEV features with buffered previous-frame features.

        For DualTopo: uses World Model prediction instead of coordinate warping.
        In streaming mode, the returned tensor is doubled (single-frame + multi-frame)
        for the lane head's prop_add_stage logic.

        Args:
            curr_bev_feats: (B, C, H, W) current-frame BEV features.
            img_metas: list[dict] -- per-sample metadata with pose info.

        Returns:
            fused_feats: (B or 2B, C, H, W) fused BEV features.
            wm_next_latent: stacked WM predictions or None.
            is_first_frame_list: list[bool]
        """
        bs = curr_bev_feats.size(0)
        fused_feats_list = []
        wm_next_latent_list = []
        single_frame_feats_list = []
        memory = self.bev_memory.get(img_metas)
        bev_memory, pose_memory = (memory["tensor"], memory["img_metas"])
        is_first_frame_list = memory["is_first_frame"]
        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if self.stream:
                if is_first_frame:
                    new_feat = self.stream_fusion_neck(
                        curr_bev_feats[i].clone().detach(), curr_bev_feats[i]
                    )
                    fused_feats_list.append(new_feat)
                    wm_next_latent_list.append(new_feat)
                else:
                    wm_next_latent = self.wm_prediction(bev_memory[i], pose_memory[i], img_metas[i])
                    new_feat = self.stream_fusion_neck(wm_next_latent, curr_bev_feats[i])
                    fused_feats_list.append(new_feat)
                    wm_next_latent_list.append(wm_next_latent)
                wm_next_latent_stack = torch.stack(wm_next_latent_list)
                single_feat = self.stream_fusion_neck(
                    curr_bev_feats[i].clone().detach(), curr_bev_feats[i]
                )
                single_frame_feats_list.append(single_feat)
            else:
                new_feat = self.stream_fusion_neck(
                    curr_bev_feats[i].clone().detach(), curr_bev_feats[i]
                )
                fused_feats_list.append(new_feat)
                wm_next_latent_stack = None
        fused_feats = torch.stack(fused_feats_list, dim=0)
        self.bev_memory.update(fused_feats, img_metas)
        if self.stream:
            single_frame_feats = torch.stack(single_frame_feats_list, dim=0)
            fused_feats = torch.cat((single_frame_feats, fused_feats), dim=0)
        return (fused_feats, wm_next_latent_stack, is_first_frame_list)

    def extract_img_feat(self, img, len_queue=None):
        """Extract multi-scale image features from multi-view images.

        Args:
            img: (B, N, C, H, W) or (B*N, C, H, W) multi-view images.
            len_queue: If given, reshape output as
                       (B/len_queue, len_queue, N, C, H, W) for temporal
                       queue processing.

        Returns:
            List of feature maps.  When len_queue is None each entry has
            shape (B, N, C, H, W); otherwise
            (B/len_queue, len_queue, N, C, H, W).
        """
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract image features (convenience wrapper)."""
        return self.extract_img_feat(img, len_queue=len_queue)

    def forward(self, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, **kwargs):
        """Inference forward pass.

        Manages temporal state (prev_bev, ego pose deltas) and delegates
        to simple_test.

        Args:
            img_metas: list[dict] -- per-sample metadata.
            img: image tensor.
            **kwargs: seq_info, etc.

        Returns:
            list[dict] of per-sample results.
        """
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img
        if img_metas[0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return results_list

    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None, seq_info=None, rescale=False):
        """Core test logic: BEV -> lane head -> topology head.

        Args:
            x: list of multi-scale image features.
            img_metas: list[dict] metadata.
            img: image tensor (unused, kept for API compat).
            prev_bev: previous-frame BEV features.
            seq_info: sequence information for temporal propagation.
            rescale: whether to rescale outputs.

        Returns:
            Tuple of (bev_feats, lane_results, lsls_results,
                      pos_results, meta_results).
        """
        batchsize = len(img_metas)
        if seq_info is not None:
            scene_name, local_idx, seq_length = seq_info[0]
            img_metas[0]["local_idx"] = local_idx
            first_frame = local_idx == 0
        else:
            first_frame = True
        can_bus_tensor = torch.stack(
            [torch.from_numpy(meta["can_bus"]).float() for meta in img_metas]
        ).to(x[0].device)
        lidar2global_rot = torch.stack(
            [torch.from_numpy(meta["lidar2global_rotation"]).float() for meta in img_metas]
        ).to(x[0].device)
        lidar2img_list = []
        for meta in img_metas:
            lidar2img_list.append(torch.from_numpy(np.stack(meta["lidar2img"])).float())
        lidar2img_tensor = torch.stack(lidar2img_list).to(x[0].device)
        img_hw_tensor = image_geometry_tensor(img_metas, x[0].device)
        bev_feats = self.bev_constructor(
            x, can_bus_tensor, lidar2global_rot, lidar2img_tensor, img_hw_tensor, prev_bev
        )
        bev_feats = bev_feats.view(bev_feats.shape[0], self.bev_h, self.bev_w, self.embed_dims)
        bev_feats = bev_feats.permute(0, 3, 1, 2)
        if self.streaming_bev:
            self.bev_memory.eval()
            bev_feats, wm_next_latent, is_first_frame_list = self.update_bev_feature(
                bev_feats, img_metas
            )
        bev_feats = bev_feats.permute(0, 2, 3, 1)
        bev_feats = bev_feats.flatten(start_dim=1, end_dim=2)
        outs = self.lane_head(x, bev_feats, img_metas)
        lane_results = self.lane_head.get_lanes(outs, img_metas, rescale=rescale)
        if seq_info is not None:
            thr_det = 0.3 if first_frame else 0.8
            pos_results = self.lane_head.prepare_temporal_propagation(
                outs, scene_name, local_idx, thr_track=0.3, thr_det=thr_det
            )
        else:
            pos_results = None
        meta_results = img_metas
        if self.lclc_head is not None:
            lane_feats = outs["history_states"]
            lsls_results = self.lclc_head.get_relationship(lane_feats, lane_feats)
            if getattr(self.lane_head, "topology_memory_enabled", False):
                temporal_feats = self.lane_head._temporal_history_states
                raw_memory = torch.stack(
                    self.lclc_head.get_relationship(temporal_feats, temporal_feats)
                )
                self.lane_head.store_topology_memory(raw_memory, img_metas)
            lsls_results = [result.detach().cpu().numpy() for result in lsls_results]
        else:
            lsls_results = [None for _ in range(batchsize)]
        return (bev_feats, lane_results, lsls_results, pos_results, meta_results)

    def simple_test(self, img_metas, img=None, prev_bev=None, seq_info=None, rescale=False):
        """Test function without augmentation.

        Args:
            img_metas: list[dict] metadata.
            img: image tensor.
            prev_bev: previous-frame BEV features.
            seq_info: sequence information.
            rescale: whether to rescale outputs.

        Returns:
            Tuple of (new_prev_bev, results_list).
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        results_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, lane_results, lsls_results, pos_results, meta_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale, seq_info=seq_info
        )
        for result_dict, lane, lsls, pos, meta in zip(
            results_list, lane_results, lsls_results, pos_results, meta_results
        ):
            result_dict["lane_results"] = lane
            result_dict["bbox_results"] = None
            result_dict["lsls_results"] = lsls
            result_dict["lste_results"] = None
            result_dict["pos_results"] = pos
            result_dict["meta_results"] = meta
            result_dict["sample_idx"] = img_metas[0].get("sample_idx", None)
        return (new_prev_bev, results_list)

    def train(self, mode=True):
        changed = self.training != mode
        if changed and self.streaming_bev:
            self.bev_memory.reset()
        super().train(mode)
        self.img_backbone.train(mode)
        if self.streaming_bev:
            if mode:
                self.bev_memory.train()
            else:
                self.bev_memory.eval()
        return self

    def eval(self):
        self.train(False)
        return self
