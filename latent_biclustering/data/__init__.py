"""Data loading, synthetic generation, graph construction, and node features."""

from latent_biclustering.data.sparse_io import (
    make_bipartite,
    load_x_coo_npz,
    save_x_coo_npz,
)
from latent_biclustering.data.synthetic import generate_synthetic
from latent_biclustering.data.realworld import preprocess_realworld_dataset

__all__ = [
    "make_bipartite",
    "load_x_coo_npz",
    "save_x_coo_npz",
    "generate_synthetic",
    "preprocess_realworld_dataset",
]
