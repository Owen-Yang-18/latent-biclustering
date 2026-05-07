"""Construction of object-object, attribute-attribute, and object-attribute pairs."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from latent_biclustering.config import PairConfig
from latent_biclustering.data.graph import bipartite_adjacency
from latent_biclustering.sampling.ppr import row_stochastic, personalized_pagerank, topk_from_scores
from latent_biclustering.types import ObjectAttributePairs, PairCollections, SameTypePairs, SparseBipartite


def _empty_same(n_neg: int) -> SameTypePairs:
    return SameTypePairs(
        anchor=np.empty((0,), dtype=np.int64),
        positive=np.empty((0,), dtype=np.int64),
        weight=np.empty((0,), dtype=np.float32),
        negative=np.empty((0, int(n_neg)), dtype=np.int64),
    )


def _empty_oa(n_neg: int) -> ObjectAttributePairs:
    return ObjectAttributePairs(
        obj=np.empty((0,), dtype=np.int64),
        attr=np.empty((0,), dtype=np.int64),
        target=np.empty((0,), dtype=np.float32),
        negative_attr=np.empty((0, int(n_neg)), dtype=np.int64),
    )


def _draw_negatives(
    n_items: int,
    excluded: Set[int],
    low_score: np.ndarray,
    n_neg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_neg = int(n_neg)
    if n_neg <= 0:
        return np.empty((0,), dtype=np.int64)
    low_score = np.asarray(low_score, dtype=np.int64)
    if low_score.size > 0:
        pool = low_score[~np.isin(low_score, np.fromiter(excluded, dtype=np.int64, count=len(excluded)))]
    else:
        pool = np.empty((0,), dtype=np.int64)
    if pool.size == 0:
        all_ids = np.arange(int(n_items), dtype=np.int64)
        if excluded:
            ex = np.fromiter(excluded, dtype=np.int64, count=len(excluded))
            pool = all_ids[~np.isin(all_ids, ex)]
        else:
            pool = all_ids
    if pool.size == 0:
        pool = np.zeros((1,), dtype=np.int64)
    return rng.choice(pool, size=n_neg, replace=True).astype(np.int64, copy=False)


def _same_type_pairs(
    *,
    ppr_rows: Dict[int, np.ndarray],
    anchors_local: np.ndarray,
    candidates_global: np.ndarray,
    n_items: int,
    top_t: int,
    n_neg: int,
    epsilon: float,
    rng: np.random.Generator,
    global_offset: int,
) -> SameTypePairs:
    a_out: List[int] = []
    p_out: List[int] = []
    w_out: List[float] = []
    n_out: List[np.ndarray] = []

    for local_anchor in anchors_local:
        anchor_global = int(local_anchor + global_offset)
        scores = ppr_rows[anchor_global]
        pos_global, pos_score = topk_from_scores(
            scores,
            candidates_global,
            int(top_t),
            float(epsilon),
            exclude=(anchor_global,),
        )
        if pos_global.size == 0:
            continue
        max_score = max(float(pos_score.max()), 1e-12)
        pos_local = pos_global - int(global_offset)
        low_global = candidates_global[scores[candidates_global] < float(epsilon)]
        low_local = low_global - int(global_offset)
        for p, s in zip(pos_local, pos_score):
            excluded = {int(local_anchor), int(p)}
            a_out.append(int(local_anchor))
            p_out.append(int(p))
            w_out.append(float(s) / max_score)
            n_out.append(_draw_negatives(n_items, excluded, low_local, int(n_neg), rng))

    if not a_out:
        return _empty_same(n_neg)
    return SameTypePairs(
        anchor=np.asarray(a_out, dtype=np.int64),
        positive=np.asarray(p_out, dtype=np.int64),
        weight=np.asarray(w_out, dtype=np.float32),
        negative=np.vstack(n_out).astype(np.int64, copy=False),
    )


def _oa_pairs(
    *,
    graph: SparseBipartite,
    ppr_rows: Dict[int, np.ndarray],
    top_t: int,
    n_neg: int,
    epsilon: float,
    rng: np.random.Generator,
) -> ObjectAttributePairs:
    obj_out: List[int] = []
    attr_out: List[int] = []
    target_out: List[float] = []
    neg_out: List[np.ndarray] = []
    attr_global = np.arange(graph.n_obj, graph.n_obj + graph.n_attr, dtype=np.int64)

    for obj in range(graph.n_obj):
        scores = ppr_rows[int(obj)]
        observed = set(int(v) for v in graph.obj_neighbors(obj))
        ppr_global, ppr_score = topk_from_scores(scores, attr_global, int(top_t), float(epsilon))
        max_score = max(float(ppr_score.max()), 1e-12) if ppr_score.size else 1.0
        targets: Dict[int, float] = {int(v): 1.0 for v in observed}
        for global_attr, score in zip(ppr_global, ppr_score):
            local_attr = int(global_attr - graph.n_obj)
            soft_target = min(1.0, max(0.0, float(score) / max_score))
            targets[local_attr] = max(targets.get(local_attr, 0.0), soft_target)
        if not targets:
            continue
        low_global = attr_global[scores[attr_global] < float(epsilon)]
        low_local = low_global - graph.n_obj
        positive_set = set(int(k) for k in targets.keys())
        for attr, target in targets.items():
            excluded = set(positive_set)
            excluded.add(int(attr))
            obj_out.append(int(obj))
            attr_out.append(int(attr))
            target_out.append(float(target))
            neg_out.append(_draw_negatives(graph.n_attr, excluded, low_local, int(n_neg), rng))

    if not obj_out:
        return _empty_oa(n_neg)
    return ObjectAttributePairs(
        obj=np.asarray(obj_out, dtype=np.int64),
        attr=np.asarray(attr_out, dtype=np.int64),
        target=np.asarray(target_out, dtype=np.float32),
        negative_attr=np.vstack(neg_out).astype(np.int64, copy=False),
    )


def build_ppr_pair_sets(
    graph: SparseBipartite,
    config: PairConfig,
    seed: int = 0,
) -> PairCollections:
    rng = np.random.default_rng(int(seed))
    adjacency = bipartite_adjacency(graph)
    transition = row_stochastic(adjacency)

    obj_global = np.arange(0, graph.n_obj, dtype=np.int64)
    attr_global = np.arange(graph.n_obj, graph.n_obj + graph.n_attr, dtype=np.int64)
    ppr_rows: Dict[int, np.ndarray] = {}
    for anchor in np.concatenate([obj_global, attr_global]):
        ppr_rows[int(anchor)] = personalized_pagerank(
            transition,
            int(anchor),
            alpha=float(config.alpha),
            max_iter=int(config.max_iter),
            tol=float(config.tol),
        )

    oo = _same_type_pairs(
        ppr_rows=ppr_rows,
        anchors_local=np.arange(graph.n_obj, dtype=np.int64),
        candidates_global=obj_global,
        n_items=graph.n_obj,
        top_t=int(config.top_t),
        n_neg=int(config.negatives_per_positive),
        epsilon=float(config.epsilon),
        rng=rng,
        global_offset=0,
    )
    aa = _same_type_pairs(
        ppr_rows=ppr_rows,
        anchors_local=np.arange(graph.n_attr, dtype=np.int64),
        candidates_global=attr_global,
        n_items=graph.n_attr,
        top_t=int(config.top_t),
        n_neg=int(config.negatives_per_positive),
        epsilon=float(config.epsilon),
        rng=rng,
        global_offset=graph.n_obj,
    )
    oa = _oa_pairs(
        graph=graph,
        ppr_rows=ppr_rows,
        top_t=int(config.top_t),
        n_neg=int(config.negatives_per_positive),
        epsilon=float(config.epsilon),
        rng=rng,
    )
    return PairCollections(oo=oo, aa=aa, oa=oa)
