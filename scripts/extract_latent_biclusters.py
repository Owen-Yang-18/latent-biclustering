#!/usr/bin/env python
"""Run optional biclustlib extraction on exported latent interaction matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from latent_biclustering.inference.biclustlib_adapter import (  # noqa: E402
    LatentBicluster,
    run_biclustlib_on_matrix,
    save_biclusters_npz,
)
from latent_biclustering.inference.mapping import map_latent_bicluster  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract latent biclusters from structured_core.npz by running an "
            "optional biclustlib algorithm on M_hat or M_prob."
        )
    )
    p.add_argument("--run-dir", type=str, required=True, help="Pipeline output directory.")
    p.add_argument("--core", type=str, default=None, help="Path to structured_core.npz.")
    p.add_argument("--matrix-key", type=str, choices=["M_hat", "M_prob"], default="M_hat")
    p.add_argument("--method", type=str, choices=["las", "plaid", "cca", "qubic"], default="las")
    p.add_argument("--n-biclusters", type=int, default=10)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--map-to-data", action="store_true")
    p.add_argument("--z", type=str, default=None, help="Path to Z.npy for latent-to-data mapping.")
    p.add_argument("--y", type=str, default=None, help="Path to Y.npy for latent-to-data mapping.")
    p.add_argument("--mapped-output", type=str, default=None)
    p.add_argument("--rho", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--qubic-binary-path", type=str, default=None)
    return p


def _default_output(run_dir: Path, method: str, matrix_key: str) -> Path:
    return run_dir / "latent_biclusters_{}_{}.npz".format(method, matrix_key)


def _default_mapped_output(run_dir: Path, method: str, matrix_key: str) -> Path:
    return run_dir / "mapped_biclusters_{}_{}.npz".format(method, matrix_key)


def main() -> None:
    args = _arg_parser().parse_args()
    run_dir = Path(args.run_dir)
    core_path = Path(args.core) if args.core is not None else run_dir / "structured_core.npz"
    output_path = Path(args.output) if args.output is not None else _default_output(run_dir, args.method, args.matrix_key)

    core = np.load(str(core_path))
    if args.matrix_key not in core:
        raise SystemExit("matrix key '{}' not found in {}".format(args.matrix_key, core_path))
    matrix = np.asarray(core[args.matrix_key], dtype=np.float64)

    biclusters = run_biclustlib_on_matrix(
        matrix,
        method=args.method,
        n_biclusters=int(args.n_biclusters),
        qubic_binary_path=args.qubic_binary_path,
    )
    save_biclusters_npz(output_path, biclusters)
    print(
        "latent extraction: method={} matrix={} shape={} biclusters={} output={}".format(
            args.method,
            args.matrix_key,
            matrix.shape,
            len(biclusters),
            output_path,
        ),
        flush=True,
    )

    if args.map_to_data:
        z_path = Path(args.z) if args.z is not None else run_dir / "Z.npy"
        y_path = Path(args.y) if args.y is not None else run_dir / "Y.npy"
        mapped_output = (
            Path(args.mapped_output)
            if args.mapped_output is not None
            else _default_mapped_output(run_dir, args.method, args.matrix_key)
        )
        z = np.load(str(z_path))
        y = np.load(str(y_path))
        mapped: List[LatentBicluster] = []
        for bic in biclusters:
            rows, cols = map_latent_bicluster(
                z,
                y,
                bic.rows,
                bic.cols,
                rho=float(args.rho),
                tau=float(args.tau),
            )
            mapped.append(LatentBicluster(rows=rows, cols=cols))
        save_biclusters_npz(mapped_output, mapped)
        print("mapped biclusters: output={}".format(mapped_output), flush=True)


if __name__ == "__main__":
    main()

