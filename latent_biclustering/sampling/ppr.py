"""Personalized PageRank routines used to build graph-proximity pairs."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import scipy.sparse as sp


def row_stochastic(adjacency: sp.spmatrix) -> sp.csr_matrix:
    adj = adjacency.tocsr().astype(np.float32)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    inv[mask] = 1.0 / deg[mask]
    trans = sp.diags(inv, dtype=np.float32) @ adj
    return trans.tocsr()


def personalized_pagerank(
    transition: sp.csr_matrix,
    anchor: int,
    alpha: float = 0.15,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    n = int(transition.shape[0])
    anchor = int(anchor)
    restart = np.zeros((n,), dtype=np.float32)
    restart[anchor] = 1.0
    score = restart.copy()
    trans_t = transition.T.tocsr()
    for _ in range(int(max_iter)):
        new_score = float(alpha) * restart + (1.0 - float(alpha)) * trans_t.dot(score)
        if np.abs(new_score - score).sum() <= float(tol):
            score = new_score.astype(np.float32, copy=False)
            break
        score = new_score.astype(np.float32, copy=False)
    return score


def ppr_for_anchors(
    transition: sp.csr_matrix,
    anchors: Iterable[int],
    alpha: float = 0.15,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for anchor in anchors:
        out[int(anchor)] = personalized_pagerank(
            transition,
            int(anchor),
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
    return out


def topk_from_scores(
    scores: np.ndarray,
    candidates: np.ndarray,
    k: int,
    epsilon: float,
    exclude: Tuple[int, ...] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0 or int(k) <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    vals = scores[candidates]
    keep = vals > float(epsilon)
    if exclude:
        ex = np.asarray(exclude, dtype=np.int64)
        keep &= ~np.isin(candidates, ex)
    kept = candidates[keep]
    kept_vals = vals[keep]
    if kept.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    order = np.argsort(kept_vals)[::-1][: int(k)]
    return kept[order].astype(np.int64, copy=False), kept_vals[order].astype(np.float32, copy=False)
