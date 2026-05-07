"""Export and latent-to-data mapping utilities."""

from latent_biclustering.inference.export_core import export_core_npz
from latent_biclustering.inference.mapping import map_latent_bicluster

__all__ = ["export_core_npz", "map_latent_bicluster"]
