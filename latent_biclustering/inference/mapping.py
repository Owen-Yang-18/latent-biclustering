"""Map selected latent factor biclusters back to data rows and columns."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def map_latent_bicluster(
    z: np.ndarray,
    y: np.ndarray,
    obj_dims: Iterable[int],
    attr_dims: Iterable[int],
    rho: float = 0.7,
    tau: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map a user-provided latent factor bicluster (I, J) into data rows and columns."""
    obj_idx = np.asarray(list(obj_dims), dtype=np.int64)
    attr_idx = np.asarray(list(attr_dims), dtype=np.int64)
    if obj_idx.size == 0 or attr_idx.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    z_sel = z[:, obj_idx]
    y_sel = y[:, attr_idx]
    row_frac = (z_sel >= float(rho)).mean(axis=1)
    col_frac = (y_sel >= float(rho)).mean(axis=1)
    rows = np.flatnonzero(row_frac >= float(tau)).astype(np.int64, copy=False)
    cols = np.flatnonzero(col_frac >= float(tau)).astype(np.int64, copy=False)
    return rows, cols
