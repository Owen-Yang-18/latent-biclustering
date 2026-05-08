"""Export and latent-to-data mapping utilities."""

from importlib import import_module

__all__ = [
    "LatentBicluster",
    "run_biclustlib_on_matrix",
    "save_biclusters_npz",
    "export_core_npz",
    "map_latent_bicluster",
]


def __getattr__(name):
    if name in {"LatentBicluster", "run_biclustlib_on_matrix", "save_biclusters_npz"}:
        module = import_module("latent_biclustering.inference.biclustlib_adapter")
        return getattr(module, name)
    if name == "export_core_npz":
        from latent_biclustering.inference.export_core import export_core_npz

        return export_core_npz
    if name == "map_latent_bicluster":
        from latent_biclustering.inference.mapping import map_latent_bicluster

        return map_latent_bicluster
    raise AttributeError(name)
