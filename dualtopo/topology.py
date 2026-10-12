# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DirectedTopologyMemory: ordered raw Top-K adjacency and confidence-gated graph propagation."""

import torch
from torch import nn


class DirectedGraphPropagation(nn.Module):
    """Directed graph propagation (paper Eq. 6). A[i,j] is an edge i -> j; lambda=0.25. Cache raw adjacency, never post-processed topology."""

    def __init__(self, embed_dims, residual_scale=0.25):
        super().__init__()
        self.residual_scale = residual_scale
        self.predecessor_projection = nn.Linear(embed_dims, embed_dims)
        self.successor_projection = nn.Linear(embed_dims, embed_dims)
        self.graph_fusion = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        nn.init.zeros_(self.graph_fusion[-1].weight)
        nn.init.zeros_(self.graph_fusion[-1].bias)

    def forward(self, query, topology, valid):
        if topology.shape != (len(query), len(query)) or valid.shape != (len(query),):
            raise ValueError("Topology, valid mask and query order disagree")
        edges = topology.to(query.dtype) * (valid[:, None] & valid[None, :]).to(query.dtype)
        edges = edges.clone()
        edges.fill_diagonal_(0)
        succ = (
            edges @ self.successor_projection(query) / edges.sum(-1, keepdim=True).clamp_min(1e-06)
        )
        pred = (
            edges.T
            @ self.predecessor_projection(query)
            / edges.sum(0, keepdim=True).T.clamp_min(1e-06)
        )
        delta = self.graph_fusion(torch.cat([query, pred, succ], -1))
        confidence = torch.maximum(edges.max(1).values, edges.max(0).values).clamp(0, 1)
        return query + self.residual_scale * delta * (valid.to(query.dtype) * confidence)[:, None]
