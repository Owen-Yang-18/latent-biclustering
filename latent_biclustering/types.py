"""Shared data containers for sparse bipartite data and sampled pair sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class SparseBipartite:
    x_csr: sp.csr_matrix
    x_csc: sp.csc_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.x_csr.shape[0]), int(self.x_csr.shape[1])

    @property
    def n_obj(self) -> int:
        return int(self.x_csr.shape[0])

    @property
    def n_attr(self) -> int:
        return int(self.x_csr.shape[1])

    def obj_degrees(self) -> np.ndarray:
        return np.asarray(self.x_csr.getnnz(axis=1), dtype=np.int64)

    def attr_degrees(self) -> np.ndarray:
        return np.asarray(self.x_csr.getnnz(axis=0), dtype=np.int64)

    def obj_neighbors(self, obj: int) -> np.ndarray:
        start = int(self.x_csr.indptr[obj])
        end = int(self.x_csr.indptr[obj + 1])
        return self.x_csr.indices[start:end].astype(np.int64, copy=False)

    def attr_neighbors(self, attr: int) -> np.ndarray:
        start = int(self.x_csc.indptr[attr])
        end = int(self.x_csc.indptr[attr + 1])
        return self.x_csc.indices[start:end].astype(np.int64, copy=False)


@dataclass(frozen=True)
class SameTypePairs:
    anchor: np.ndarray
    positive: np.ndarray
    weight: np.ndarray
    negative: np.ndarray


@dataclass(frozen=True)
class ObjectAttributePairs:
    obj: np.ndarray
    attr: np.ndarray
    target: np.ndarray
    negative_attr: np.ndarray


@dataclass(frozen=True)
class PairCollections:
    oo: SameTypePairs
    aa: SameTypePairs
    oa: ObjectAttributePairs


@dataclass(frozen=True)
class SyntheticData:
    x: sp.csr_matrix
    object_membership: np.ndarray
    attribute_membership: np.ndarray
    relations: List[Tuple[int, int]]
    metadata: Dict[str, float]
