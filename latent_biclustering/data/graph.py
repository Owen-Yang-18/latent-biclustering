"""Bipartite graph construction for sparse object-attribute matrices."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch

from latent_biclustering.types import SparseBipartite


def bipartite_adjacency(graph: SparseBipartite) -> sp.csr_matrix:
    n_obj, n_attr = graph.shape
    zero_obj = sp.csr_matrix((n_obj, n_obj), dtype=np.float32)
    zero_attr = sp.csr_matrix((n_attr, n_attr), dtype=np.float32)
    adj = sp.bmat(
        [[zero_obj, graph.x_csr], [graph.x_csc, zero_attr]],
        format="csr",
        dtype=np.float32,
    )
    adj.sum_duplicates()
    adj.eliminate_zeros()
    return adj


def torch_edge_index(graph: SparseBipartite, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x_coo = graph.x_csr.tocoo()
    row = torch.from_numpy(x_coo.row.astype(np.int64, copy=False))
    col = torch.from_numpy((x_coo.col + graph.n_obj).astype(np.int64, copy=False))
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0).to(device)
    n_total = graph.n_obj + graph.n_attr
    obj_mask = torch.zeros((n_total,), dtype=torch.bool, device=device)
    obj_mask[: graph.n_obj] = True
    return edge_index, obj_mask


def global_node_ids(graph: SparseBipartite, device: torch.device) -> torch.Tensor:
    return torch.arange(0, graph.n_obj + graph.n_attr, dtype=torch.long, device=device)
