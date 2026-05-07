"""Three-phase optimization loop for the released latent biclustering model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from latent_biclustering.config import TrainConfig
from latent_biclustering.models.joint_model import JointEmbeddingBiclusteringModel
from latent_biclustering.objectives.losses import embedding_sparsity_loss, oa_bce_loss, same_type_sgns_loss
from latent_biclustering.training.phases import phase_for_step, temperature_for_step
from latent_biclustering.types import PairCollections


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(bool(value))


def train_model(
    model: JointEmbeddingBiclusteringModel,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    obj_mask: torch.Tensor,
    pairs: PairCollections,
    config: TrainConfig,
    node_ids: Optional[torch.Tensor] = None,
    log_every: int = 10,
) -> List[Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    history: List[Dict[str, float]] = []

    for step in range(int(config.steps)):
        phase = phase_for_step(step, config)
        _set_requires_grad(model.encoder, phase in ("encoder", "joint"))
        _set_requires_grad(model.core, phase in ("core", "joint"))

        model.train()
        if phase == "core":
            model.encoder.eval()
        opt.zero_grad(set_to_none=True)
        z, y = model.encode(x=features, edge_index=edge_index, obj_mask=obj_mask, node_ids=node_ids)
        temp = temperature_for_step(step, config)
        m = model.core_matrix(temperature=temp, hard=bool(config.hard_gumbel))

        loss_oo = same_type_sgns_loss(z, pairs.oo)
        loss_aa = same_type_sgns_loss(y, pairs.aa)
        loss_oa = oa_bce_loss(z, y, m, pairs.oa)
        loss_kl = model.core.kl_loss(reduction="mean")
        loss_sparse = embedding_sparsity_loss(z, y, config.lambda_z, config.lambda_y)

        if phase == "encoder":
            loss = float(config.lambda_oo) * loss_oo + float(config.lambda_aa) * loss_aa + loss_sparse
        elif phase == "core":
            loss = loss_oa + float(config.beta_kl) * loss_kl
        else:
            loss = (
                loss_oa
                + float(config.beta_kl) * loss_kl
                + float(config.lambda_oo) * loss_oo
                + float(config.lambda_aa) * loss_aa
                + loss_sparse
            )

        loss.backward()
        opt.step()

        if step % max(1, int(log_every)) == 0 or step == int(config.steps) - 1:
            history.append(
                {
                    "step": float(step),
                    "phase": phase,
                    "temperature": float(temp),
                    "loss": float(loss.detach().cpu()),
                    "loss_oo": float(loss_oo.detach().cpu()),
                    "loss_aa": float(loss_aa.detach().cpu()),
                    "loss_oa": float(loss_oa.detach().cpu()),
                    "loss_kl": float(loss_kl.detach().cpu()),
                    "loss_sparse": float(loss_sparse.detach().cpu()),
                }
            )
    _set_requires_grad(model.encoder, True)
    _set_requires_grad(model.core, True)
    return history
