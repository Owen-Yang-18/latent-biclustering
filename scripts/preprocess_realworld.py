#!/usr/bin/env python
"""Preprocess supported real-world datasets into the release sparse format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from latent_biclustering.data.realworld import preprocess_realworld_dataset  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Preprocess a real-world interaction dataset into X.npz plus row/column "
            "mapping files for the latent biclustering pipeline."
        )
    )
    p.add_argument("--dataset", choices=["retail", "lastfm", "movielens", "amazon"], required=True)
    p.add_argument("--input", required=True, help="Path to the raw dataset file.")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--min-user", type=int, default=5)
    p.add_argument("--min-item", type=int, default=5)
    p.add_argument("--recursive", action="store_true", help="Use recursive k-core filtering.")
    p.add_argument("--amazon-item-key", choices=["asin", "parent_asin"], default="asin")
    p.add_argument("--amazon-min-rating", type=float, default=None)
    p.add_argument("--amazon-max-records", type=int, default=None)
    p.add_argument("--movielens-min-rating", type=float, default=None)
    return p


def _fmt_optional(value: Optional[float]) -> str:
    return "None" if value is None else str(value)


def main() -> None:
    args = _arg_parser().parse_args()
    try:
        result = preprocess_realworld_dataset(
            dataset=args.dataset,
            input_path=args.input,
            output_dir=args.output_dir,
            min_user=int(args.min_user),
            min_item=int(args.min_item),
            recursive=bool(args.recursive),
            amazon_item_key=str(args.amazon_item_key),
            amazon_min_rating=args.amazon_min_rating,
            amazon_max_records=args.amazon_max_records,
            movielens_min_rating=args.movielens_min_rating,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    stats = result.stats
    print(
        "preprocessed dataset={} matrix={} shape=({},{}) nnz={} density={:.6g}".format(
            args.dataset,
            result.matrix_path,
            int(stats["n_rows"]),
            int(stats["n_cols"]),
            int(stats["filtered_interactions"]),
            float(stats["density"]),
        ),
        flush=True,
    )
    print("row mapping: {}".format(result.row_mapping_path), flush=True)
    print("col mapping: {}".format(result.col_mapping_path), flush=True)
    print("stats: {}".format(result.stats_path), flush=True)
    if args.dataset == "amazon":
        print(
            "amazon options: item_key={} min_rating={} max_records={}".format(
                args.amazon_item_key,
                _fmt_optional(args.amazon_min_rating),
                _fmt_optional(args.amazon_max_records),
            ),
            flush=True,
        )
    if args.dataset == "movielens":
        print(
            "movielens options: min_rating={}".format(_fmt_optional(args.movielens_min_rating)),
            flush=True,
        )


if __name__ == "__main__":
    main()
