"""Preprocessing utilities for real-world object-attribute datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gzip
import json

import numpy as np
import scipy.sparse as sp

from latent_biclustering.data.sparse_io import save_x_coo_npz


@dataclass(frozen=True)
class RealworldPreprocessResult:
    matrix_path: Path
    row_mapping_path: Path
    col_mapping_path: Path
    stats_path: Path
    stats: Dict[str, float]


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Real-world preprocessing requires pandas. Install optional dependencies "
            "with `pip install -r requirements-realworld.txt` from the release directory."
        ) from exc
    return pd


def _read_table(path: Union[str, Path], sep: str = "\t"):
    pd = _require_pandas()
    return pd.read_csv(path, sep=sep)


def _drop_missing_and_dupes(df, user_col: str, item_col: str):
    df = df[[user_col, item_col]].dropna()
    df = df.drop_duplicates()
    df = df.rename(columns={user_col: "user_id", item_col: "item_id"})
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    return df


def recursive_k_core(df, min_user: int = 5, min_item: int = 5, max_iters: int = 1):
    min_user = int(min_user)
    min_item = int(min_item)
    max_iters = int(max_iters)
    cur = df.copy()
    for _ in range(max_iters):
        n0 = int(len(cur))
        if min_user > 1:
            user_counts = cur["user_id"].value_counts()
            cur = cur[cur["user_id"].isin(user_counts[user_counts >= min_user].index)]
        if min_item > 1:
            item_counts = cur["item_id"].value_counts()
            cur = cur[cur["item_id"].isin(item_counts[item_counts >= min_item].index)]
        if int(len(cur)) == n0:
            break
    return cur


def load_online_retail(path: Union[str, Path]):
    pd = _require_pandas()
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Online Retail .xlsx files requires openpyxl. Install "
                "`pip install -r requirements-realworld.txt` from the release directory."
            ) from exc
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"customerid", "stockcode", "quantity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Online Retail file is missing columns: {}".format(sorted(missing)))
    df = df.dropna(subset=["customerid", "stockcode"])
    df = df[df["quantity"] > 0]
    invalid_codes = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK"}
    df["stockcode"] = df["stockcode"].astype(str)
    df = df[~df["stockcode"].isin(invalid_codes)]
    return _drop_missing_and_dupes(df, "customerid", "stockcode")


def load_lastfm(path: Union[str, Path]):
    df = _read_table(path, sep="\t")
    required = {"userID", "artistID"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Last.fm file is missing columns: {}".format(sorted(missing)))
    if "weight" in df.columns:
        df = df[df["weight"] > 0]
    return _drop_missing_and_dupes(df, "userID", "artistID")


def load_movielens(path: Union[str, Path], min_rating: Optional[float] = None):
    pd = _require_pandas()
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["userID", "movieID", "rating", "timestamp"],
        engine="python",
        encoding="ISO-8859-1",
    )
    required = {"userID", "movieID", "rating"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("MovieLens file is missing columns: {}".format(sorted(missing)))
    if min_rating is not None:
        df = df[df["rating"] >= float(min_rating)]
    return _drop_missing_and_dupes(df, "userID", "movieID")


def load_amazon_reviews(
    path: Union[str, Path],
    item_key: str = "asin",
    min_rating: Optional[float] = None,
    max_records: Optional[int] = None,
):
    pd = _require_pandas()
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if max_records is not None and line_no >= int(max_records):
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rating = obj.get("rating")
            if min_rating is not None and rating is not None and float(rating) < float(min_rating):
                continue
            user_id = obj.get("user_id") or obj.get("reviewerID")
            item_id = obj.get(item_key) or obj.get("asin") or obj.get("parent_asin")
            if user_id is not None and item_id is not None:
                rows.append({"user_id": str(user_id), "item_id": str(item_id)})
    if not rows:
        raise ValueError("No valid Amazon interactions found in {}".format(path))
    return pd.DataFrame(rows).drop_duplicates()


def interactions_to_sparse(df) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[str, int]]:
    users = sorted(df["user_id"].astype(str).unique())
    items = sorted(df["item_id"].astype(str).unique())
    row_map = {raw_id: i for i, raw_id in enumerate(users)}
    col_map = {raw_id: i for i, raw_id in enumerate(items)}
    row = df["user_id"].astype(str).map(row_map).to_numpy(dtype=np.int64)
    col = df["item_id"].astype(str).map(col_map).to_numpy(dtype=np.int64)
    data = np.ones((row.shape[0],), dtype=np.float32)
    x = sp.coo_matrix((data, (row, col)), shape=(len(users), len(items)), dtype=np.float32).tocsr()
    x.sum_duplicates()
    x.eliminate_zeros()
    return x, row_map, col_map


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def preprocess_realworld_dataset(
    dataset: str,
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    min_user: int = 5,
    min_item: int = 5,
    recursive: bool = False,
    amazon_item_key: str = "asin",
    amazon_min_rating: Optional[float] = None,
    amazon_max_records: Optional[int] = None,
    movielens_min_rating: Optional[float] = None,
) -> RealworldPreprocessResult:
    dataset = str(dataset).lower()
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if dataset == "retail":
        df = load_online_retail(input_path)
    elif dataset == "lastfm":
        df = load_lastfm(input_path)
    elif dataset == "movielens":
        df = load_movielens(input_path, min_rating=movielens_min_rating)
    elif dataset == "amazon":
        df = load_amazon_reviews(
            input_path,
            item_key=amazon_item_key,
            min_rating=amazon_min_rating,
            max_records=amazon_max_records,
        )
    else:
        raise ValueError("unknown dataset: {}".format(dataset))

    raw_interactions = int(len(df))
    max_iters = 999 if bool(recursive) else 1
    df = recursive_k_core(df, min_user=min_user, min_item=min_item, max_iters=max_iters)
    if df.empty:
        raise ValueError("No interactions remain after filtering.")

    x, row_map, col_map = interactions_to_sparse(df)
    ds_dir = output_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = ds_dir / "X.npz"
    row_mapping_path = ds_dir / "row_mapping.json"
    col_mapping_path = ds_dir / "col_mapping.json"
    stats_path = ds_dir / "stats.json"

    save_x_coo_npz(matrix_path, x)
    _write_json(row_mapping_path, row_map)
    _write_json(col_mapping_path, col_map)
    density = float(x.nnz) / float(max(1, x.shape[0] * x.shape[1]))
    stats = {
        "raw_interactions": float(raw_interactions),
        "filtered_interactions": float(x.nnz),
        "n_rows": float(x.shape[0]),
        "n_cols": float(x.shape[1]),
        "density": density,
        "sparsity": 1.0 - density,
        "min_user": float(min_user),
        "min_item": float(min_item),
        "recursive": float(1 if recursive else 0),
    }
    _write_json(stats_path, stats)
    return RealworldPreprocessResult(
        matrix_path=matrix_path,
        row_mapping_path=row_mapping_path,
        col_mapping_path=col_mapping_path,
        stats_path=stats_path,
        stats=stats,
    )
