"""Loss functions for graph context, interaction learning, and regularization."""

from latent_biclustering.objectives.losses import (
    oa_bce_loss,
    same_type_sgns_loss,
    embedding_sparsity_loss,
)

__all__ = ["oa_bce_loss", "same_type_sgns_loss", "embedding_sparsity_loss"]
