"""Joint model combining the GAT encoder with the structured interaction core."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from latent_biclustering.config import ModelConfig
from latent_biclustering.models.gat import GATEmbeddingEncoder
from latent_biclustering.models.structured_core import BernoulliStructuredCore


class JointEmbeddingBiclusteringModel(nn.Module):
    def __init__(self, n_total: int, config: ModelConfig, node_feat: str = "fixed", seed: int = 0):
        super().__init__()
        self.encoder = GATEmbeddingEncoder(n_total=n_total, config=config, node_feat=node_feat, seed=seed)
        self.core = BernoulliStructuredCore(
            d_obj=int(config.d_obj),
            d_attr=int(config.d_attr),
            kmax=int(config.kmax),
            prior_a=float(config.prior_a),
            prior_b=float(config.prior_b),
            prior_gamma=float(config.prior_gamma),
            seed=int(seed) + 17,
        )

    def encode(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        obj_mask: torch.Tensor,
        node_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x=x, edge_index=edge_index, obj_mask=obj_mask, node_ids=node_ids)

    def core_matrix(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        return self.core.matrix(temperature=temperature, hard=hard)

    @staticmethod
    def score_oo(z: torch.Tensor, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        return (z[anchor] * z[positive]).sum(dim=1)

    @staticmethod
    def score_aa(y: torch.Tensor, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        return (y[anchor] * y[positive]).sum(dim=1)

    @staticmethod
    def score_oa(z: torch.Tensor, y: torch.Tensor, m: torch.Tensor, obj: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        zy = z[obj] @ m
        return (zy * y[attr]).sum(dim=1)
