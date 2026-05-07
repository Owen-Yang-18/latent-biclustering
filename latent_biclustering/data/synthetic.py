"""Synthetic sparse object-attribute data with planted group relations."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from latent_biclustering.types import SyntheticData


def _memberships(
    n: int,
    groups: int,
    conc: float,
    overlap_decay: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probs = rng.dirichlet(np.full((groups,), float(conc), dtype=np.float64))
    primary = rng.choice(groups, size=int(n), replace=True, p=probs)
    mem = np.zeros((int(n), int(groups)), dtype=bool)
    mem[np.arange(int(n)), primary] = True
    if groups > 1 and overlap_decay > 0:
        extra_prob = min(0.95, float(overlap_decay) / max(1, groups - 1))
        extra = rng.random((int(n), int(groups))) < extra_prob
        mem |= extra
        mem[np.arange(int(n)), primary] = True
    return mem


def _restricted_probs(ids: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    if ids.size == 0:
        return None
    w = np.asarray(weights[ids], dtype=np.float64)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return w / total


def _sample_unique_pairs(
    rows: Sequence[int],
    cols: Sequence[int],
    n_edges: int,
    rng: np.random.Generator,
    row_weights: Optional[np.ndarray] = None,
    col_weights: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    rows_arr = np.asarray(rows, dtype=np.int64)
    cols_arr = np.asarray(cols, dtype=np.int64)
    n_edges = int(min(max(0, n_edges), rows_arr.size * cols_arr.size))
    if n_edges <= 0 or rows_arr.size == 0 or cols_arr.size == 0:
        return []

    rp = _restricted_probs(rows_arr, row_weights) if row_weights is not None else None
    cp = _restricted_probs(cols_arr, col_weights) if col_weights is not None else None
    out = set()
    rounds = 0
    while len(out) < n_edges and rounds < 200:
        rounds += 1
        draw = min(max(4 * (n_edges - len(out)), 256), 50000)
        rr = rng.choice(rows_arr, size=draw, replace=True, p=rp)
        cc = rng.choice(cols_arr, size=draw, replace=True, p=cp)
        for r, c in zip(rr, cc):
            out.add((int(r), int(c)))
            if len(out) >= n_edges:
                break

    if len(out) < n_edges and rows_arr.size * cols_arr.size <= 2000000:
        all_pairs = [(int(r), int(c)) for r in rows_arr for c in cols_arr]
        rng.shuffle(all_pairs)
        for pair in all_pairs:
            out.add(pair)
            if len(out) >= n_edges:
                break
    return list(out)


def generate_synthetic(
    n_obj: int = 5000,
    n_attr: int = 5000,
    go: int = 10,
    ga: int = 10,
    nrel: int = 10,
    conc: float = 1.0,
    overlap_decay: float = 0.2,
    target_sparsity: float = 0.999,
    background_rate: float = 1e-4,
    heterogeneity: float = 0.5,
    seed: int = 0,
) -> SyntheticData:
    rng = np.random.default_rng(int(seed))
    n_obj = int(n_obj)
    n_attr = int(n_attr)
    go = int(go)
    ga = int(ga)
    nrel = int(min(nrel, go * ga))

    obj_mem = _memberships(n_obj, go, conc, overlap_decay, rng)
    attr_mem = _memberships(n_attr, ga, conc, overlap_decay, rng)

    all_rel = [(i, j) for i in range(go) for j in range(ga)]
    chosen = rng.choice(len(all_rel), size=nrel, replace=False)
    relations = [all_rel[int(i)] for i in chosen]

    total_cells = int(n_obj * n_attr)
    target_nnz = int(round((1.0 - float(target_sparsity)) * total_cells))
    bg_nnz = int(round(float(background_rate) * total_cells))
    bg_nnz = min(bg_nnz, target_nnz)
    signal_nnz = max(0, target_nnz - bg_nnz)

    row_w = rng.lognormal(mean=0.0, sigma=max(0.0, float(heterogeneity)), size=n_obj)
    col_w = rng.lognormal(mean=0.0, sigma=max(0.0, float(heterogeneity)), size=n_attr)

    edges = set(_sample_unique_pairs(np.arange(n_obj), np.arange(n_attr), bg_nnz, rng, row_w, col_w))
    if signal_nnz > 0 and nrel > 0:
        rel_weight = rng.dirichlet(np.full((nrel,), max(float(conc), 1e-3), dtype=np.float64))
        rel_counts = rng.multinomial(signal_nnz, rel_weight)
        for (og, ag), count in zip(relations, rel_counts):
            rows = np.flatnonzero(obj_mem[:, int(og)])
            cols = np.flatnonzero(attr_mem[:, int(ag)])
            for edge in _sample_unique_pairs(rows, cols, int(count), rng, row_w, col_w):
                edges.add(edge)

    if edges:
        rows_np = np.fromiter((e[0] for e in edges), dtype=np.int64)
        cols_np = np.fromiter((e[1] for e in edges), dtype=np.int64)
        data = np.ones((rows_np.size,), dtype=np.float32)
    else:
        rows_np = np.empty((0,), dtype=np.int64)
        cols_np = np.empty((0,), dtype=np.int64)
        data = np.empty((0,), dtype=np.float32)
    x = sp.coo_matrix((data, (rows_np, cols_np)), shape=(n_obj, n_attr), dtype=np.float32).tocsr()
    x.sum_duplicates()
    x.eliminate_zeros()

    metadata = {
        "target_sparsity": float(target_sparsity),
        "actual_sparsity": float(1.0 - x.nnz / max(1, total_cells)),
        "background_rate": float(background_rate),
        "heterogeneity": float(heterogeneity),
        "nnz": float(x.nnz),
    }
    return SyntheticData(
        x=x,
        object_membership=obj_mem,
        attribute_membership=attr_mem,
        relations=relations,
        metadata=metadata,
    )
