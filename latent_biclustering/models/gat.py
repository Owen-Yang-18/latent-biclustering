"""Graph-attentive encoder for object and attribute embeddings."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_biclustering.config import ModelConfig

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None  # type: ignore


class GATEmbeddingEncoder(nn.Module):
    def __init__(self, n_total: int, config: ModelConfig, node_feat: str = "fixed", seed: int = 0):
        super().__init__()
        if GATConv is None:
            raise ImportError("torch-geometric is required for GATEmbeddingEncoder")
        if node_feat not in ("fixed", "learned"):
            raise ValueError("node_feat must be 'fixed' or 'learned'")
        self.config = config
        self.node_feat = node_feat

        g = torch.Generator()
        g.manual_seed(int(seed))
        if node_feat == "learned":
            self.node_emb = nn.Embedding(int(n_total), int(config.in_dim))
            with torch.no_grad():
                self.node_emb.weight.copy_(torch.randn((int(n_total), int(config.in_dim)), generator=g) * 0.02)
        else:
            self.node_emb = None

        self.conv1 = GATConv(
            int(config.in_dim),
            int(config.hidden_dim),
            heads=int(config.heads),
            concat=True,
            dropout=float(config.dropout),
            add_self_loops=False,
        )
        self.conv2 = GATConv(
            int(config.hidden_dim) * int(config.heads),
            int(config.hidden_dim),
            heads=1,
            concat=False,
            dropout=float(config.dropout),
            add_self_loops=False,
        )
        self.ln1 = nn.LayerNorm(int(config.hidden_dim) * int(config.heads))
        self.ln2 = nn.LayerNorm(int(config.hidden_dim))
        self.res1 = nn.Identity()
        if int(config.in_dim) != int(config.hidden_dim) * int(config.heads):
            self.res1 = nn.Linear(int(config.in_dim), int(config.hidden_dim) * int(config.heads), bias=False)
        self.res2 = nn.Identity()
        if int(config.hidden_dim) * int(config.heads) != int(config.hidden_dim):
            self.res2 = nn.Linear(int(config.hidden_dim) * int(config.heads), int(config.hidden_dim), bias=False)

        self.obj_proj1 = nn.Linear(int(config.hidden_dim), int(config.proj_hidden_dim))
        self.obj_proj2 = nn.Linear(int(config.proj_hidden_dim), int(config.d_obj))
        self.attr_proj1 = nn.Linear(int(config.hidden_dim), int(config.proj_hidden_dim))
        self.attr_proj2 = nn.Linear(int(config.proj_hidden_dim), int(config.d_attr))

    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        obj_mask: torch.Tensor,
        node_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.node_feat == "learned":
            if node_ids is None:
                raise ValueError("node_ids is required when node_feat='learned'")
            h = self.node_emb(node_ids)  # type: ignore[operator]
        else:
            if x is None:
                raise ValueError("x is required when node_feat='fixed'")
            h = x

        h = F.dropout(h, p=float(self.config.dropout), training=self.training)
        h1 = self.conv1(h, edge_index)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=float(self.config.dropout), training=self.training)
        h1 = self.ln1(h1 + self.res1(h))

        h2 = self.conv2(h1, edge_index)
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=float(self.config.dropout), training=self.training)
        h2 = self.ln2(h2 + self.res2(h1))

        h_obj = h2[obj_mask]
        h_attr = h2[~obj_mask]
        z_raw = self.obj_proj2(F.relu(self.obj_proj1(h_obj)))
        y_raw = self.attr_proj2(F.relu(self.attr_proj1(h_attr)))
        z_raw = z_raw - z_raw.mean(dim=0, keepdim=True)
        y_raw = y_raw - y_raw.mean(dim=0, keepdim=True)
        z = F.normalize(z_raw, p=2, dim=1, eps=1e-12)
        y = F.normalize(y_raw, p=2, dim=1, eps=1e-12)
        return z, y
