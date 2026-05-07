"""Sparse random projection features for object and attribute nodes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from latent_biclustering.types import SparseBipartite


def _sparse_sign_projection(
    n_input: int,
    dim: int,
    density: float,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    n_input = int(n_input)
    dim = int(dim)
    density = float(density)
    nnz = max(dim, int(round(n_input * dim * max(density, 1.0 / max(n_input, 1)))))
    row = rng.integers(0, n_input, size=nnz, dtype=np.int64)
    col = rng.integers(0, dim, size=nnz, dtype=np.int64)
    val = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=nnz)
    scale = np.sqrt(max(1.0, n_input * density))
    mat = sp.coo_matrix((val / scale, (row, col)), shape=(n_input, dim), dtype=np.float32)
    mat.sum_duplicates()
    return mat.tocsr()


def sparse_random_projection_features(
    graph: SparseBipartite,
    dim: int = 32,
    density: float = 0.05,
    seed: int = 0,
    normalize: bool = True,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Project object and attribute interaction profiles into a shared feature space."""
    rng = np.random.default_rng(int(seed))
    dtype = np.float32 if dtype is None else dtype

    r_attr = _sparse_sign_projection(graph.n_attr, int(dim), float(density), rng)
    r_obj = _sparse_sign_projection(graph.n_obj, int(dim), float(density), rng)

    obj_feat = graph.x_csr @ r_attr
    attr_feat = graph.x_csc.T @ r_obj
    features = sp.vstack([obj_feat, attr_feat], format="csr").astype(dtype)
    out = np.asarray(features.toarray(), dtype=dtype)

    if normalize:
        mean = out.mean(axis=0, keepdims=True)
        std = out.std(axis=0, keepdims=True)
        out = (out - mean) / np.maximum(std, 1e-6)
    return out.astype(dtype, copy=False)
