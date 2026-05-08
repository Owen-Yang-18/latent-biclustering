#!/usr/bin/env python
"""Run latent biclustering on a preprocessed sparse matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import numpy as np
    import scipy.sparse as sp
    import torch
    import torch_geometric  # noqa: F401
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        "Missing required dependency '{}'. Install the release dependencies with "
        "`pip install -r requirements.txt` or activate the Python environment "
        "used for this project before running the pipeline.".format(missing)
    ) from exc

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from latent_biclustering.config import ModelConfig, PairConfig, TrainConfig, load_json_config  # noqa: E402
from latent_biclustering.data.features import sparse_random_projection_features  # noqa: E402
from latent_biclustering.data.graph import global_node_ids, torch_edge_index  # noqa: E402
from latent_biclustering.data.sparse_io import load_x_coo_npz, make_bipartite  # noqa: E402
from latent_biclustering.inference.export_core import export_core_npz  # noqa: E402
from latent_biclustering.models.joint_model import JointEmbeddingBiclusteringModel  # noqa: E402
from latent_biclustering.sampling.pairs import build_ppr_pair_sets  # noqa: E402
from latent_biclustering.training.trainer import train_model  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    default_config = RELEASE_ROOT / "configs" / "realworld_pipeline.json"
    p = argparse.ArgumentParser(description="Run latent biclustering on a sparse X matrix.")
    p.add_argument("--matrix", required=True, help="Path to X.npz from preprocessing.")
    p.add_argument("--config", default=str(default_config))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--feature-dim", type=int, default=None)
    p.add_argument("--d-obj", type=int, default=None)
    p.add_argument("--d-attr", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--kmax", type=int, default=None)
    p.add_argument("--ppr-top-t", type=int, default=None)
    p.add_argument("--negatives-per-positive", type=int, default=None)
    return p


def _get(args: argparse.Namespace, cfg: Dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return cfg.get(name, default)


def _load_matrix(path: Path) -> sp.csr_matrix:
    try:
        return load_x_coo_npz(path).tocsr()
    except Exception:
        return sp.load_npz(str(path)).tocsr().astype(np.float32)


def main() -> None:
    args = _arg_parser().parse_args()
    cfg = load_json_config(args.config)
    seed = int(_get(args, cfg, "seed", 7))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)

    x_csr = _load_matrix(Path(args.matrix))
    graph = make_bipartite(x_csr)
    feature_dim = int(_get(args, cfg, "feature_dim", 32))
    features_np = sparse_random_projection_features(graph, dim=feature_dim, density=0.08, seed=seed)
    features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
    edge_index, obj_mask = torch_edge_index(graph, device=device)
    node_ids = global_node_ids(graph, device=device)

    pair_cfg = PairConfig(
        top_t=int(_get(args, cfg, "ppr_top_t", 8)),
        negatives_per_positive=int(_get(args, cfg, "negatives_per_positive", 4)),
        epsilon=float(_get(args, cfg, "ppr_epsilon", 1e-5)),
        alpha=float(_get(args, cfg, "ppr_alpha", 0.15)),
        max_iter=int(_get(args, cfg, "ppr_max_iter", 50)),
        tol=float(_get(args, cfg, "ppr_tol", 1e-8)),
    )
    pairs = build_ppr_pair_sets(graph, pair_cfg, seed=seed)

    model_cfg = ModelConfig(
        in_dim=feature_dim,
        hidden_dim=int(_get(args, cfg, "hidden_dim", 64)),
        proj_hidden_dim=int(_get(args, cfg, "hidden_dim", 64)),
        d_obj=int(_get(args, cfg, "d_obj", 32)),
        d_attr=int(_get(args, cfg, "d_attr", 32)),
        heads=int(_get(args, cfg, "heads", 2)),
        dropout=float(_get(args, cfg, "dropout", 0.05)),
        kmax=int(_get(args, cfg, "kmax", 8)),
        prior_a=float(_get(args, cfg, "prior_a", 0.1)),
        prior_b=float(_get(args, cfg, "prior_b", 0.1)),
        prior_gamma=float(_get(args, cfg, "prior_gamma", 0.1)),
    )
    model = JointEmbeddingBiclusteringModel(
        n_total=graph.n_obj + graph.n_attr,
        config=model_cfg,
        node_feat="fixed",
        seed=seed,
    ).to(device)
    train_cfg = TrainConfig(
        steps=int(_get(args, cfg, "steps", 18)),
        lr=float(_get(args, cfg, "lr", 1e-3)),
        weight_decay=float(_get(args, cfg, "weight_decay", 1e-4)),
        lambda_oo=float(_get(args, cfg, "lambda_oo", 1.0)),
        lambda_aa=float(_get(args, cfg, "lambda_aa", 1.0)),
        beta_kl=float(_get(args, cfg, "beta_kl", 1.0)),
        lambda_z=float(_get(args, cfg, "lambda_z", 0.01)),
        lambda_y=float(_get(args, cfg, "lambda_y", 0.01)),
        phase1_frac=float(_get(args, cfg, "phase1_frac", 0.40)),
        phase2_frac=float(_get(args, cfg, "phase2_frac", 0.35)),
        temp_start=float(_get(args, cfg, "temp_start", 1.0)),
        temp_end=float(_get(args, cfg, "temp_end", 0.1)),
    )

    print("matrix: shape={} nnz={}".format(graph.shape, graph.x_csr.nnz), flush=True)
    print("pairs: OO={} AA={} OA={}".format(pairs.oo.anchor.size, pairs.aa.anchor.size, pairs.oa.obj.size), flush=True)
    history = train_model(
        model=model,
        features=features,
        edge_index=edge_index,
        obj_mask=obj_mask,
        pairs=pairs,
        config=train_cfg,
        node_ids=node_ids,
        log_every=max(1, int(train_cfg.steps) // 6),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        z, y = model.encode(x=features, edge_index=edge_index, obj_mask=obj_mask, node_ids=node_ids)
        np.save(str(out_dir / "Z.npy"), z.detach().cpu().numpy().astype(np.float32))
        np.save(str(out_dir / "Y.npy"), y.detach().cpu().numpy().astype(np.float32))
    export_core_npz(model.core, out_dir / "structured_core.npz")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("final: loss={:.6f} output={}".format(history[-1]["loss"], out_dir), flush=True)


if __name__ == "__main__":
    main()
