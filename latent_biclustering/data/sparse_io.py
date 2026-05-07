"""Sparse matrix I/O and degree utilities for object-attribute data."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import scipy.sparse as sp

from latent_biclustering.types import SparseBipartite


def make_bipartite(x: sp.spmatrix) -> SparseBipartite:
    x_csr = x.tocsr().astype(np.float32)
    x_csr.sum_duplicates()
    x_csr.eliminate_zeros()
    x_csc = x_csr.tocsc()
    x_csc.sum_duplicates()
    return SparseBipartite(x_csr=x_csr, x_csc=x_csc)


def load_x_coo_npz(path: Union[str, Path]) -> sp.coo_matrix:
    data = np.load(str(path))
    rows = data["rows"].astype(np.int64, copy=False)
    cols = data["cols"].astype(np.int64, copy=False)
    shape = tuple(int(x) for x in data["shape"])
    values = data["values"].astype(np.float32, copy=False) if "values" in data else np.ones(rows.shape[0], dtype=np.float32)
    return sp.coo_matrix((values, (rows, cols)), shape=shape, dtype=np.float32)


def save_x_coo_npz(path: Union[str, Path], x: sp.spmatrix) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_coo = x.tocoo()
    np.savez_compressed(
        str(path),
        rows=x_coo.row.astype(np.int64, copy=False),
        cols=x_coo.col.astype(np.int64, copy=False),
        values=x_coo.data.astype(np.float32, copy=False),
        shape=np.asarray(x_coo.shape, dtype=np.int64),
    )


def degree_distribution(degrees: np.ndarray, power: float = 0.75, eps: float = 1e-12) -> np.ndarray:
    weights = np.power(np.maximum(degrees.astype(np.float64, copy=False), 0.0), float(power)) + float(eps)
    weights /= weights.sum()
    return weights.astype(np.float64, copy=False)
