# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""Inference lane and relationship heads. QWM/BWM terminology follows the DualTopo manuscript."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bev_encoder import inverse_sigmoid
from .decoder import DualPathwayTransformer
from .memory_buffer import StreamTensorMemory, StreamListMemory


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    """Initialize module weights with Xavier initialization."""
    nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class LaneSegMotionMLP(nn.Module):
    """Motion MLP for query propagation across frames.

    Args:
        c_dim (int): dimension of condition (pose encoding).
        f_dim (int): feature dimension.
        identity (bool): if True, use residual (zero-init) connection.
    """

    def __init__(self, c_dim, f_dim=512, identity=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.identity = identity
        self.fc = nn.Sequential(
            nn.Linear(c_dim + f_dim, 2 * f_dim),
            nn.LayerNorm(2 * f_dim),
            nn.ReLU(),
            nn.Linear(2 * f_dim, f_dim),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.fc:
            for param in m.parameters():
                if param.dim() > 1:
                    if self.identity:
                        nn.init.zeros_(param)
                    else:
                        nn.init.xavier_uniform_(param)

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=-1)
        out = self.fc(xc)
        if self.identity:
            out = out + x
        return out


from .topology import DirectedGraphPropagation


class DualPathwayLaneHead(nn.Module):
    """Dual-Pathway lane segment head with QWM, BWM and directed graph propagation. This release supports inference only."""

    def __init__(
        self,
        num_classes,
        in_channels,
        dn_cls_num=3,
        num_points=10,
        roi_size=(60, 30),
        num_query=200,
        with_box_refine=False,
        with_shared_param=None,
        transformer=None,
        bbox_coder=None,
        num_reg_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        pc_range=None,
        pts_dim=3,
        num_lane_type_classes=3,
        streaming_cfg=dict(),
        dn_cfg=dict(),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        pred_mask=False,
        bev_world_model=None,
        num_traj_modal=3,
        use_sigmoid=True,
        lane_type_use_sigmoid=True,
        **kwargs,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.test_cfg = test_cfg
        self.pred_mask = pred_mask
        if self.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        if lane_type_use_sigmoid:
            self.cls_lane_type_out_channels = num_lane_type_classes
        else:
            self.cls_lane_type_out_channels = num_lane_type_classes + 1
        if transformer is not None:
            tf_cfg = transformer.copy() if isinstance(transformer, dict) else dict(transformer)
            tf_cfg.pop("type", None)
            self.transformer = DualPathwayTransformer(**tf_cfg)
            self.embed_dims = self.transformer.embed_dims
        else:
            self.transformer = None
            self.embed_dims = kwargs.get("embed_dims", 256)
        self.bev_h = bev_h
        self.bev_w = bev_w
        assert pts_dim in (2, 3)
        self.pts_dim = pts_dim
        self.with_box_refine = with_box_refine
        if with_shared_param is not None:
            self.with_shared_param = with_shared_param
        else:
            self.with_shared_param = not self.with_box_refine
        self.as_two_stage = False
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = pts_dim * 30
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0] * self.code_size
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        self.gt_c_save = self.code_size
        if bbox_coder is not None:
            bc_cfg = bbox_coder.copy()
            bc_cfg.pop("type", None)
            self.bbox_coder = LaneSegmentPseudoCoder(**bc_cfg)
        else:
            self.bbox_coder = LaneSegmentPseudoCoder()
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self.num_lane_type_classes = num_lane_type_classes
        self.dn_cfg = dn_cfg
        self.num_queries = num_query
        self.dn_cls_num = dn_cls_num
        self.num_points = num_points
        if streaming_cfg:
            self.streaming_query = streaming_cfg.get("streaming", False)
        else:
            self.streaming_query = False
        if self.streaming_query:
            self.batch_size = streaming_cfg["batch_size"]
            self.topk_query = streaming_cfg["topk"]
            self.topk_select_method = streaming_cfg.get("topk_select_method", "score")
            self.query_memory = StreamTensorMemory(self.batch_size)
            self.reference_points_memory = StreamTensorMemory(self.batch_size)
            self.lane_ref_points_memory = StreamTensorMemory(self.batch_size)
            self.lane_id_memory = StreamListMemory(self.batch_size)
            self.topology_memory_enabled = streaming_cfg.get("topology_memory", False)
            self.topology_aggregation_enabled = streaming_cfg.get(
                "directed_graph_propagation", False
            )
            if streaming_cfg.get("stream_topology_loss", False):
                raise ValueError("This run explicitly disables extra topology supervision")
            if self.topology_memory_enabled:
                self.topology_memory = StreamTensorMemory(self.batch_size)
                self.topology_valid_memory = StreamTensorMemory(self.batch_size)
            if self.topology_aggregation_enabled:
                assert self.topology_memory_enabled
                self.directed_graph_propagation = DirectedGraphPropagation(self.embed_dims)
            self.capture_trace = False
            c_dim = 12
            self.query_update = LaneSegMotionMLP(c_dim=c_dim, f_dim=self.embed_dims, identity=True)
        self.num_traj_modal = num_traj_modal
        if self.streaming_query and bev_world_model is not None:
            from .bev_constructor import BEVWorldModel

            wm_cfg = bev_world_model.copy()
            wm_cfg.pop("type", None)
            self.bev_world_model = BEVWorldModel(**wm_cfg)
        else:
            self.bev_world_model = None
        wm_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dims, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.query_world_model = QueryWorldModel(wm_decoder_layer, 2)
        self.action_aware_bev_encoder = nn.Sequential(
            nn.Linear(self.embed_dims + 12, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.action_aware_query_encoder = nn.Sequential(
            nn.Linear(self.embed_dims + 12, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.register_buffer("roi_size", torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0] / 2, -roi_size[1] / 2, self.pc_range[2])
        self.register_buffer("origin", torch.tensor(origin, dtype=torch.float32))
        self.map_size = [-50, -25, 50, 25]
        self._init_layers()

    def _select_topk_indices(self, cls_scores):
        assert cls_scores.dim() == 1
        num_queries = cls_scores.size(0)
        k = min(self.topk_query, num_queries)
        if self.topk_select_method == "random":
            return torch.randperm(num_queries, device=cls_scores.device)[:k]
        _, topk_idx = cls_scores.topk(k=k, dim=-1)
        return topk_idx

    def _init_layers(self):

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size // 3))
        reg_branch = nn.Sequential(*reg_branch)
        reg_branch_offset = []
        for _ in range(self.num_reg_fcs):
            reg_branch_offset.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch_offset.append(nn.ReLU())
        reg_branch_offset.append(nn.Linear(self.embed_dims, self.code_size // 3))
        reg_branch_offset = nn.Sequential(*reg_branch_offset)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        cls_left_type_branch = []
        for _ in range(self.num_reg_fcs):
            cls_left_type_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_left_type_branch.append(nn.LayerNorm(self.embed_dims))
            cls_left_type_branch.append(nn.ReLU(inplace=True))
        cls_left_type_branch.append(nn.Linear(self.embed_dims, self.cls_lane_type_out_channels))
        fc_cls_left_type = nn.Sequential(*cls_left_type_branch)
        cls_right_type_branch = []
        for _ in range(self.num_reg_fcs):
            cls_right_type_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_right_type_branch.append(nn.LayerNorm(self.embed_dims))
            cls_right_type_branch.append(nn.ReLU(inplace=True))
        cls_right_type_branch.append(nn.Linear(self.embed_dims, self.cls_lane_type_out_channels))
        fc_cls_right_type = nn.Sequential(*cls_right_type_branch)
        mask_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        num_pred = (
            self.transformer.decoder.num_layers + 1
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )
        if not self.with_shared_param:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.reg_branches_offset = _get_clones(reg_branch_offset, num_pred)
            self.cls_left_type_branches = _get_clones(fc_cls_left_type, num_pred)
            self.cls_right_type_branches = _get_clones(fc_cls_right_type, num_pred)
            self.mask_embed = _get_clones(mask_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
            self.reg_branches_offset = nn.ModuleList([reg_branch_offset for _ in range(num_pred)])
            self.cls_left_type_branches = nn.ModuleList([fc_cls_left_type for _ in range(num_pred)])
            self.cls_right_type_branches = nn.ModuleList(
                [fc_cls_right_type for _ in range(num_pred)]
            )
            self.mask_embed = nn.ModuleList([mask_branch for _ in range(num_pred)])
        if self.dn_cfg is not None:
            self.dn_pts = nn.Linear(self.embed_dims * 3 // 4, self.embed_dims // 2)
            self.label_embedding = nn.Embedding(self.dn_cls_num, self.embed_dims // 2)
            self.left_type_label_embedding = nn.Embedding(self.dn_cls_num + 1, self.embed_dims // 2)
            self.right_type_label_embedding = nn.Embedding(
                self.dn_cls_num + 1, self.embed_dims // 2
            )
            self.query_convert = nn.Linear(self.embed_dims, self.embed_dims)
            self.instance_convert = nn.Linear(
                self.embed_dims * self.num_points // 2, self.embed_dims // 2
            )
            self.content_fusion = nn.Linear(self.embed_dims // 2 * 3, self.embed_dims // 2)
            xavier_init(self.dn_pts, distribution="uniform", bias=0.0)
            xavier_init(self.query_convert, distribution="uniform", bias=0.0)
            xavier_init(self.instance_convert, distribution="uniform", bias=0.0)
            xavier_init(self.content_fusion, distribution="uniform", bias=0.0)
        self.reference_points_embed = nn.Linear(self.embed_dims, self.pts_dim)
        xavier_init(self.reference_points_embed, distribution="uniform", bias=0.0)
        if self.streaming_query:
            if isinstance(self.query_update, LaneSegMotionMLP):
                self.query_update.init_weights()

    def _forward_mask_head(self, output, bev_feats, lvl):
        bs_out = output.shape[0]
        if bev_feats.shape[0] != bs_out:
            bev_feats = bev_feats[:bs_out]
        bev_feats = bev_feats.view([bev_feats.shape[0], self.bev_h, self.bev_w, self.embed_dims])
        bev_feats = bev_feats.permute(0, 3, 1, 2).contiguous()
        mask_embed = self.mask_embed[lvl](output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, bev_feats)
        return outputs_mask

    def propagate_worldmodel_pre(self, query_embedding, bev_feats, img_metas):
        bs = query_embedding.shape[0]
        propagated_query_list = []
        prop_reference_points_list = []
        prop_centerline_reference_points_list = []
        prev2curr_matrix_list = []
        tmp = self.query_memory.get(img_metas)
        query_memory, pose_memory = (tmp["tensor"], tmp["img_metas"])
        tmp = self.reference_points_memory.get(img_metas)
        ref_pts_memory, pose_memory = (tmp["tensor"], tmp["img_metas"])
        tmp = self.lane_ref_points_memory.get(img_metas)
        lane_ref_pts_memory, pose_memory = (tmp["tensor"], tmp["img_metas"])
        is_first_frame_list = tmp["is_first_frame"]
        if getattr(self, "topology_memory_enabled", False):
            topology_memory = self.topology_memory.get(img_metas)["tensor"]
            topology_valid = self.topology_valid_memory.get(img_metas)["tensor"]
            self._topology_read = list(zip(is_first_frame_list, topology_memory, topology_valid))
        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                padding = query_embedding.new_zeros((self.topk_query, self.embed_dims))
                propagated_query_list.append(padding)
                padding = query_embedding.new_zeros((self.topk_query, 8, self.pts_dim))
                prop_reference_points_list.append(padding)
                padding = query_embedding.new_zeros(
                    (self.topk_query, self.num_points, self.pts_dim)
                )
                prop_centerline_reference_points_list.append(padding)
                prev2curr_matrix_list.append(
                    torch.eye(4, dtype=torch.float64, device=query_embedding.device)
                )
            else:
                prev_e2g_trans = self.roi_size.new_tensor(
                    pose_memory[i]["lidar2global_translation"], dtype=torch.float64
                )
                prev_e2g_rot = self.roi_size.new_tensor(
                    pose_memory[i]["lidar2global_rotation"], dtype=torch.float64
                )
                curr_e2g_trans = self.roi_size.new_tensor(
                    img_metas[i]["lidar2global_translation"], dtype=torch.float64
                )
                curr_e2g_rot = self.roi_size.new_tensor(
                    img_metas[i]["lidar2global_rotation"], dtype=torch.float64
                )
                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans
                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                pos_encoding = prev2curr_matrix.float()[:3].view(-1)
                prev2curr_matrix_list.append(prev2curr_matrix)
                prop_q = query_memory[i].unsqueeze(0)
                batch_size_q, num_tokens, num_channel = prop_q.shape
                pos_encoding_expanded = pos_encoding.view(1, -1).repeat(1, num_tokens, 1)
                pre_query_feat_with_pos = torch.cat([prop_q, pos_encoding_expanded], dim=-1)
                action_aware_query_latent = self.action_aware_query_encoder(pre_query_feat_with_pos)
                query_memory_updated = self.query_world_model(
                    action_aware_query_latent, action_aware_query_latent
                )
                query_memory_updated = query_memory_updated.squeeze(0)
                if getattr(self, "topology_aggregation_enabled", False):
                    adjacency, valid = (topology_memory[i], topology_valid[i])
                    if adjacency is None or valid is None:
                        raise RuntimeError("Query history has no aligned topology history")
                    query_memory_updated = self.directed_graph_propagation(
                        query_memory_updated, adjacency, valid.bool()
                    )
                propagated_query_list.append(query_memory_updated.clone())
                pred = self.reg_branches[-1](query_memory_updated)
                pred_offset = self.reg_branches_offset[-1](query_memory_updated)
                pred_offset = pred_offset.view(self.topk_query, self.num_points, self.pts_dim)
                assert list(pred.shape) == [self.topk_query, self.pts_dim * self.num_points]
                prev_ref_pts = ref_pts_memory[i][-1, :, :, :]
                assert list(prev_ref_pts.shape) == [self.topk_query, self.num_points, self.pts_dim]
                denormed_last_layer_prev_ref_pts = (
                    prev_ref_pts * self.roi_size[: self.pts_dim] + self.origin[: self.pts_dim]
                )
                _pad_z = 4 - self.pts_dim - 1
                _pads = []
                if _pad_z > 0:
                    _pads.append(
                        denormed_last_layer_prev_ref_pts.new_zeros(
                            (self.topk_query, self.num_points, _pad_z)
                        )
                    )
                _pads.append(
                    denormed_last_layer_prev_ref_pts.new_ones((self.topk_query, self.num_points, 1))
                )
                denormed_last_layer_prev_ref_pts = torch.cat(
                    [denormed_last_layer_prev_ref_pts] + _pads, dim=-1
                )
                normed_last_layer_prev2curr_ref_pts = torch.einsum(
                    "lk,ijk->ijl", prev2curr_matrix, denormed_last_layer_prev_ref_pts.double()
                ).float()
                normed_last_layer_prev2curr_ref_pts = (
                    normed_last_layer_prev2curr_ref_pts[..., : self.pts_dim]
                    - self.origin[: self.pts_dim]
                ) / self.roi_size[: self.pts_dim]
                normed_last_layer_prev2curr_ref_pts = torch.clip(
                    normed_last_layer_prev2curr_ref_pts, min=0.0, max=1.0
                )
                prop_centerline_reference_points_list.append(normed_last_layer_prev2curr_ref_pts)
                pred = pred.view(self.topk_query, self.num_points, self.pts_dim)
                pred = pred + inverse_sigmoid(normed_last_layer_prev2curr_ref_pts)
                pred = pred.sigmoid().clone()
                denormed_pred = pred * self.roi_size[: self.pts_dim] + self.origin[: self.pts_dim]
                denormed_pred = denormed_pred.contiguous()
                denormed_pred_left = denormed_pred + pred_offset
                denormed_pred_right = denormed_pred - pred_offset
                normed_pred = torch.cat(
                    [denormed_pred, denormed_pred_left, denormed_pred_right], dim=1
                )
                assert list(normed_pred.shape) == [
                    self.topk_query,
                    self.num_points * 3,
                    self.pts_dim,
                ]
                denormed_lane_ref_pts = (
                    lane_ref_pts_memory[i][-1, :, :, :] * self.roi_size[: self.pts_dim]
                    + self.origin[: self.pts_dim]
                )
                assert list(denormed_lane_ref_pts.shape) == [self.topk_query, 8, self.pts_dim]
                _pad_z = 4 - self.pts_dim - 1
                _pads = []
                if _pad_z > 0:
                    _pads.append(denormed_lane_ref_pts.new_zeros((self.topk_query, 8, _pad_z)))
                _pads.append(denormed_lane_ref_pts.new_ones((self.topk_query, 8, 1)))
                denormed_lane_ref_pts = torch.cat([denormed_lane_ref_pts] + _pads, dim=-1)
                assert list(denormed_lane_ref_pts.shape) == [self.topk_query, 8, 4]
                curr_lane_ref_pts = torch.einsum(
                    "lk,ijk->ijl", prev2curr_matrix, denormed_lane_ref_pts.double()
                ).float()
                normed_lane_ref_pts = (
                    curr_lane_ref_pts[..., : self.pts_dim] - self.origin[: self.pts_dim]
                ) / self.roi_size[: self.pts_dim]
                normed_lane_ref_pts = torch.clip(normed_lane_ref_pts, min=0.0, max=1.0)
                prop_reference_points_list.append(normed_lane_ref_pts)
        prop_query_embedding = torch.stack(propagated_query_list)
        prop_ref_pts = torch.stack(prop_reference_points_list)
        prop_centerline_ref_pts = torch.stack(prop_centerline_reference_points_list)
        assert list(prop_query_embedding.shape) == [bs, self.topk_query, self.embed_dims]
        assert list(prop_ref_pts.shape) == [bs, self.topk_query, 8, self.pts_dim]
        assert list(prop_centerline_ref_pts.shape) == [
            bs,
            self.topk_query,
            self.num_points,
            self.pts_dim,
        ]
        query_pos, _ = torch.split(query_embedding, self.embed_dims, dim=2)
        init_reference_points = self.reference_points_embed(query_pos)
        init_reference_points = init_reference_points.repeat(1, 1, self.num_points)
        init_reference_points = init_reference_points.sigmoid()
        bs, num_qeury, _ = init_reference_points.shape
        init_reference_points = init_reference_points.view(
            bs, num_qeury, self.num_points, self.pts_dim
        )
        memory_query_embedding = None
        return (
            query_embedding,
            prop_query_embedding,
            init_reference_points,
            prop_ref_pts,
            prop_centerline_ref_pts,
            memory_query_embedding,
            is_first_frame_list,
        )

    def forward_test(self, mlvl_feats, bev_feats, img_metas):
        bs = len(img_metas)
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)[None, ...].repeat(bs, 1, 1)
        if self.streaming_query:
            (
                query_embedding,
                prop_query_embedding,
                init_reference_points,
                prop_ref_pts,
                prop_centerline_ref_pts,
                memory_query,
                is_first_frame_list,
            ) = self.propagate_worldmodel_pre(object_query_embeds, bev_feats, img_metas)
        else:
            query_pos, _ = torch.split(object_query_embeds, self.embed_dims, dim=1)
            init_reference_points = self.reference_points_embed(query_pos)
            init_reference_points = init_reference_points.repeat(1, 1, self.num_points)
            init_reference_points = init_reference_points.sigmoid()
            bs, num_qeury, _ = init_reference_points.shape
            init_reference_points = init_reference_points.view(
                bs, num_qeury, self.num_points, self.pts_dim
            )
            prop_query_embedding = None
            prop_ref_pts = None
            prop_centerline_ref_pts = None
            is_first_frame_list = [True for _ in range(bs)]
        assert list(init_reference_points.shape) == [
            bs,
            self.num_queries,
            self.num_points,
            self.pts_dim,
        ]
        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            object_query_embeds,
            reference_points=init_reference_points,
            prop_query=prop_query_embedding,
            prop_reference_points=prop_ref_pts,
            prop_centerline_reference_points=prop_centerline_ref_pts,
            is_first_frame_list=is_first_frame_list,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            reg_branches=(self.reg_branches, self.reg_branches_offset)
            if self.with_box_refine
            else None,
            cls_branches=self.cls_branches,
            img_metas=img_metas,
        )
        (
            init_hs,
            inter_hs,
            init_reference,
            init_inter_reference_points,
            inter_references,
            init_inter_lane_reference_points,
            inter_lanepts_references,
        ) = outputs
        init_hs = init_hs.permute(0, 2, 1, 3)
        inter_hs = inter_hs.permute(0, 2, 1, 3)
        reference = inter_references[-1]
        reference = inverse_sigmoid(reference)
        assert reference.shape[-1] == self.pts_dim
        outputs_class = self.cls_branches[-1](inter_hs[-1])
        output_left_type = self.cls_left_type_branches[-1](inter_hs[-1])
        output_right_type = self.cls_right_type_branches[-1](inter_hs[-1])
        tmp = self.reg_branches[-1](inter_hs[-1])
        bs_, num_query, _ = tmp.shape
        tmp = tmp.view(bs_, num_query, -1, self.pts_dim)
        tmp = tmp + reference
        tmp = tmp.sigmoid()
        coord = tmp.clone()
        coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        if self.pts_dim == 3:
            coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        centerline = coord.view(bs_, num_query, -1).contiguous()
        offset = self.reg_branches_offset[-1](inter_hs[-1])
        left_laneline = centerline + offset
        right_laneline = centerline - offset
        if self.pred_mask:
            outputs_mask = self._forward_mask_head(inter_hs[-1], bev_feats, -1)
        stream_start = bs_ // 2
        output_slice = (
            slice(0, stream_start)
            if getattr(self, "eval_output", "temporal") == "singleframe" and self.streaming_query
            else slice(stream_start, None)
        )
        outputs_classes = torch.stack([outputs_class[output_slice]])
        if self.pts_dim == 2:
            outputs_coords = centerline[output_slice].unsqueeze(0)
        else:
            outputs_coords = torch.stack(
                [
                    torch.cat(
                        [
                            centerline[output_slice],
                            left_laneline[output_slice],
                            right_laneline[output_slice],
                        ],
                        axis=-1,
                    )
                ]
            )
        output_left_types = torch.stack([output_left_type[output_slice]])
        output_right_types = torch.stack([output_right_type[output_slice]])
        hs_merge = inter_hs[:, output_slice, :, :]
        self._temporal_history_states = inter_hs[:, bs:, :, :]
        outs = {
            "all_cls_scores": outputs_classes,
            "all_lanes_preds": outputs_coords,
            "all_mask_preds": torch.stack([outputs_mask]) if self.pred_mask else None,
            "all_lanes_left_type": output_left_types,
            "all_lanes_right_type": output_right_types,
            "history_states": hs_merge,
            "sample_idx": img_metas[0]["sample_idx"],
        }
        if self.streaming_query:
            query_list = []
            self._topk_indices = []
            self._topk_valid = []
            ref_pts_list = []
            lane_ref_pts_list = []
            lines = outputs_class[bs_ // 2 :]
            scores = outputs_class[bs_ // 2 :]
            for i in range(bs):
                si = bs_ // 2 + i
                _lines = lines[i]
                _scores = scores[i]
                _queries = inter_hs[-1][si]
                _inter_reference = inter_references[:, si, :, :]
                _inter_lane_reference = inter_lanepts_references[:, si, :, :]
                assert len(_lines) == len(_queries)
                _class_logits = _scores
                _scores, _ = _scores.max(-1)
                topk_idx = self._select_topk_indices(_scores)
                self._topk_indices.append(topk_idx)
                self._topk_valid.append(_class_logits.argmax(-1)[topk_idx] == 0)
                _scores[topk_idx]
                _queries = _queries[topk_idx]
                _lines = _lines[topk_idx]
                _inter_reference = _inter_reference[:, topk_idx, :, :]
                _inter_lane_reference = _inter_lane_reference[:, topk_idx, :, :]
                _all_reference = _inter_reference
                query_list.append(_queries)
                ref_pts_list.append(_all_reference)
                lane_ref_pts_list.append(_inter_lane_reference)
            self.query_memory.update(query_list, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.lane_ref_points_memory.update(lane_ref_pts_list, img_metas)
        return outs

    def get_lanes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            lanes = preds["lane3d"]
            scores = preds["scores"]
            labels = preds["labels"]
            result = [lanes, scores, labels]
            if "left_type_scores" in preds:
                left_type_scores = preds["left_type_scores"]
                left_type_labels = preds["left_type_labels"]
                right_type_scores = preds["right_type_scores"]
                right_type_labels = preds["right_type_labels"]
                result.extend(
                    [left_type_scores, left_type_labels, right_type_scores, right_type_labels]
                )
            ret_list.append(result)
        return ret_list

    def store_topology_memory(self, relationship_probability, img_metas):
        if not getattr(self, "topology_memory_enabled", False):
            return
        assert relationship_probability.ndim == 3
        assert len(self._topk_indices) == len(relationship_probability)
        self._raw_relationship_for_memory = relationship_probability.detach()
        selected = []
        for adjacency, indices in zip(relationship_probability, self._topk_indices):
            assert adjacency.shape == (self.num_query, self.num_query)
            selected.append(adjacency.index_select(0, indices).index_select(1, indices))
        self.topology_memory.update(selected, img_metas)
        self.topology_valid_memory.update(self._topk_valid, img_metas)

    def train(self, mode=True):
        changed = self.training != mode
        super().train(mode)
        for v in self.__dict__.values():
            if isinstance(v, (StreamTensorMemory, StreamListMemory)):
                if changed:
                    v.reset()
                v.train(mode)
        return self

    def eval(self):
        super().eval()
        for v in self.__dict__.values():
            if isinstance(v, StreamTensorMemory):
                v.eval()

    def forward(self, *args, **kwargs):
        return self.forward_test(*args, **kwargs)

    def prepare_temporal_propagation(
        self, preds_dict, scene_name, local_idx, memory_bank=None, thr_track=0.1, thr_det=0.5
    ):
        lines = preds_dict["all_lanes_preds"][0]
        queries = preds_dict["history_states"][0]
        left_types = preds_dict["all_lanes_left_type"][0]
        right_types = preds_dict["all_lanes_right_type"][0]
        bs = len(lines)
        assert bs == 1
        scores = preds_dict["all_cls_scores"][0]
        first_frame = (
            local_idx == 0
            or not hasattr(self, "prop_info")
            or self.prop_info.get("scene_name") != scene_name
        )
        tmp_vectors = lines[0]
        tmp_queries = queries[0]
        tmp_left_types = left_types[0]
        tmp_right_types = right_types[0]
        if self.use_sigmoid:
            tmp_scores, tmp_labels = scores[0].max(-1)
            tmp_scores = tmp_scores.sigmoid()
            if not first_frame:
                last_track_num = min(self.prop_info["prop_frame_num_instance"], self.topk_query)
                pos_track = tmp_scores[:last_track_num] > thr_track
                pos_det = tmp_scores[last_track_num:] > thr_det
                pos = torch.cat([pos_track, pos_det], dim=0)
            else:
                pos_det = tmp_scores > thr_det
                pos = pos_det
        else:
            raise RuntimeError("The experiment uses sigmoid for cls outputs")
        if torch.sum(pos) > self.topk_query and (not first_frame):
            _, valid_idx = torch.topk(
                tmp_scores[last_track_num:], k=self.topk_query - torch.sum(pos_track), dim=-1
            )
            pad_pos_det = torch.zeros_like(pos_det, dtype=torch.bool)
            pad_pos_det[valid_idx] = True
            pos_det = pad_pos_det
            pos = torch.cat([pos_track, pos_det], dim=0)
        elif torch.sum(pos) > self.topk_query and first_frame:
            _, valid_idx = torch.topk(tmp_scores, k=self.topk_query, dim=-1)
            pad_pos_det = torch.zeros_like(pos_det, dtype=torch.bool)
            pad_pos_det[valid_idx] = True
            pos_det = pad_pos_det
            pos = pos_det
        pos_vectors = tmp_vectors[pos]
        pos_labels = tmp_labels[pos]
        pos_queries = tmp_queries[pos]
        pos_scores = tmp_scores[pos]
        _, pos_left_types = tmp_left_types[pos].sigmoid().max(-1)
        _, pos_right_types = tmp_right_types[pos].sigmoid().max(-1)
        if first_frame:
            global_ids = torch.arange(len(pos_vectors), device=pos_vectors.device)
            num_instance = len(pos_vectors)
        else:
            prop_ids = self.prop_info["global_ids"]
            prop_num_instance = self.prop_info["num_instance"]
            self.prop_info["prop_frame_num_instance"]
            global_ids_track = prop_ids[pos_track[: len(prop_ids)]]
            num_newborn = int(pos_det.sum())
            global_ids_newborn = (
                torch.arange(num_newborn, device=pos_vectors.device) + prop_num_instance
            )
            global_ids = torch.cat([global_ids_track, global_ids_newborn])
            num_instance = prop_num_instance + num_newborn
        if first_frame:
            self.prop_info = {
                "vectors": pos_vectors,
                "queries": pos_queries,
                "scores": pos_scores,
                "labels": pos_labels,
                "left_types": pos_left_types,
                "right_types": pos_right_types,
                "scene_name": scene_name,
                "local_idx": local_idx,
                "global_ids": global_ids,
                "num_instance": num_instance,
                "prop_frame_num_instance": num_instance,
            }
        else:
            self.prop_info = {
                "vectors": pos_vectors,
                "queries": pos_queries,
                "scores": pos_scores,
                "labels": pos_labels,
                "left_types": pos_left_types,
                "right_types": pos_right_types,
                "scene_name": scene_name,
                "local_idx": local_idx,
                "global_ids": global_ids,
                "num_instance": num_instance,
                "prop_frame_num_instance": len(pos_vectors),
            }
        if memory_bank is not None:
            if first_frame:
                num_tracks = 0
            else:
                num_tracks = self.prop_active_tracks
            pos_out_inds = torch.where(pos)[0]
            prev_out = {"history_states": queries, "all_cls_scores": scores}
            memory_bank.update_memory(
                0, first_frame, pos_out_inds, prev_out, num_tracks, local_idx, memory_bank.curr_t
            )
            self.prop_active_tracks = len(pos_out_inds)
        save_pos_results = {
            "vectors": pos_vectors.cpu().numpy(),
            "scores": pos_scores.cpu().numpy(),
            "labels": pos_labels.cpu().numpy(),
            "left_types": pos_left_types.cpu().numpy(),
            "right_types": pos_right_types.cpu().numpy(),
            "global_ids": global_ids.cpu().numpy(),
            "scene_name": scene_name,
            "local_idx": local_idx,
            "num_instance": num_instance,
        }
        return [save_pos_results]


class _RelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            (nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RelationshipHead(nn.Module):
    """Predict directed pairwise lane connectivity probabilities."""

    def __init__(self, in_channels_o1, in_channels_o2=None, shared_param=True, init_cfg=None):
        super().__init__()
        self.MLP_o1 = _RelMLP(in_channels_o1, in_channels_o1, 128, 3)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = _RelMLP(in_channels_o2, in_channels_o2, 128, 3)
        self.classifier = _RelMLP(256, 256, 1, 3)

    def get_relationship(self, o1_feats, o2_feats):
        rel_pred = self._forward_rel(o1_feats, o2_feats)
        rel_results = rel_pred.squeeze(-1).sigmoid()
        rel_results = [_ for _ in rel_results]
        return rel_results

    def _forward_rel(self, o1_feats, o2_feats):
        o1_embeds = self.MLP_o1(o1_feats[-1])
        o2_embeds = self.MLP_o2(o2_feats[-1])
        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)
        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)
        relationship_pred = self.classifier(relationship_tensor)
        return relationship_pred


from .coder import LaneSegmentPseudoCoder


class QueryWorldModel(nn.TransformerDecoder):
    """QWM: action-aware query latent prediction with two Transformer layers."""
