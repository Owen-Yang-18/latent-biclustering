"""Optional adapter for running biclustlib algorithms on learned M."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union
import warnings

import numpy as np


@dataclass(frozen=True)
class LatentBicluster:
    rows: np.ndarray
    cols: np.ndarray


def _ensure_numpy_legacy_aliases() -> None:
    """Provide NumPy aliases expected by older biclustlib releases."""
    aliases = {
        "int": int,
        "bool": bool,
        "float": float,
        "str": str,
        "object": object,
    }
    for name, value in aliases.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                getattr(np, name)
            except AttributeError:
                setattr(np, name, value)


def _build_biclustlib_algorithm(
    method: str,
    n_biclusters: int,
    qubic_binary_path: Optional[str] = None,
):
    _ensure_numpy_legacy_aliases()
    method = method.lower()
    if method == "las":
        from biclustlib.algorithms.las import LargeAverageSubmatrices  # type: ignore

        return LargeAverageSubmatrices(num_biclusters=int(n_biclusters))
    if method == "plaid":
        from biclustlib.algorithms.plaid import Plaid  # type: ignore

        return Plaid(num_biclusters=int(n_biclusters))
    if method == "cca":
        from biclustlib.algorithms.cca import ChengChurchAlgorithm  # type: ignore

        return ChengChurchAlgorithm(num_biclusters=int(n_biclusters))
    if method == "qubic":
        from biclustlib.algorithms.wrappers.qubic import QualitativeBiclustering  # type: ignore

        return QualitativeBiclustering(
            num_biclusters=int(n_biclusters),
            binary_path=qubic_binary_path,
        )
    raise ValueError("unknown biclustlib method: {}".format(method))


def run_biclustlib_on_matrix(
    matrix: np.ndarray,
    method: str = "las",
    n_biclusters: int = 10,
    qubic_binary_path: Optional[str] = None,
) -> List[LatentBicluster]:
    """Run a biclustlib algorithm on M_hat or M_prob."""
    try:
        algo = _build_biclustlib_algorithm(
            method=method,
            n_biclusters=int(n_biclusters),
            qubic_binary_path=qubic_binary_path,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "biclustlib extraction requires optional dependencies. Install them with "
            "`pip install -r requirements-extraction.txt` from the release directory."
        ) from exc

    result = algo.run(np.asarray(matrix, dtype=np.float64))
    out: List[LatentBicluster] = []
    for bic in getattr(result, "biclusters", []):
        rows = np.asarray(getattr(bic, "rows", []), dtype=np.int64)
        cols = np.asarray(getattr(bic, "cols", []), dtype=np.int64)
        rows = np.unique(rows)
        cols = np.unique(cols)
        if rows.size > 0 and cols.size > 0:
            out.append(LatentBicluster(rows=rows, cols=cols))
    return out


def save_biclusters_npz(path: Union[str, Path], biclusters: Sequence[LatentBicluster]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_parts = []
    col_parts = []
    row_offsets = [0]
    col_offsets = [0]
    for bic in biclusters:
        rows = np.asarray(bic.rows, dtype=np.int64)
        cols = np.asarray(bic.cols, dtype=np.int64)
        row_parts.append(rows)
        col_parts.append(cols)
        row_offsets.append(row_offsets[-1] + int(rows.size))
        col_offsets.append(col_offsets[-1] + int(cols.size))
    rows_all = np.concatenate(row_parts) if row_parts else np.empty((0,), dtype=np.int64)
    cols_all = np.concatenate(col_parts) if col_parts else np.empty((0,), dtype=np.int64)
    np.savez_compressed(
        str(path),
        rows=rows_all,
        cols=cols_all,
        row_offsets=np.asarray(row_offsets, dtype=np.int64),
        col_offsets=np.asarray(col_offsets, dtype=np.int64),
    )
