# Latent Biclustering via Structured Interaction Learning

This directory contains the source release for training and applying the latent
biclustering model on sparse object-attribute matrices.

## Quick Start

Run from the repository root:

```bash
cd release
pip install -r requirements.txt
python scripts/run_synthetic_pipeline.py \
  --config configs/synthetic_pipeline.json \
  --device cpu
```

For Slurm:

```bash
sbatch release/scripts/synthetic_pipeline.sbatch
```

The Slurm script changes into `release/` and writes model artifacts under
`release/outputs/`.

## Directory Map

```text
release/
  configs/                  JSON configuration files
  docs/                     concise data command references
  latent_biclustering/      importable implementation package
  scripts/                  command-line entry points

  requirements.txt          core training dependencies
  requirements-realworld.txt optional preprocessing dependencies
  requirements-extraction.txt optional biclustlib extraction dependencies
```

## Package Map

```text
latent_biclustering/
  data/          sparse matrix IO, synthetic data, real-world preprocessing
  sampling/      PPR-based pair construction
  models/        encoder and structured interaction core
  objectives/    training losses
  training/      optimization schedule and trainer
  inference/     export, optional extraction, and mapping utilities
```

Start with `scripts/run_synthetic_pipeline.py` to trace the complete code path.
For a preprocessed sparse matrix, use `scripts/run_matrix_pipeline.py`.

## Entry Points

```text
scripts/run_synthetic_pipeline.py     generate a matrix, train, and export artifacts
scripts/run_matrix_pipeline.py        train on an existing X.npz matrix
scripts/preprocess_realworld.py       convert downloaded datasets into X.npz
scripts/extract_latent_biclusters.py  optional extraction from structured_core.npz
scripts/synthetic_pipeline.sbatch     Slurm wrapper for the synthetic pipeline
```

Real-world download commands are listed in `docs/real_world_data.md`.

## Configurations

```text
configs/synthetic_pipeline.json   default synthetic run
configs/realworld_pipeline.json   default run on a preprocessed matrix
configs/appendix_defaults.json    recorded appendix-style hyperparameters
```

## Outputs

A training run writes to the selected `--output-dir`:

```text
X.npz
Z.npy
Y.npy
structured_core.npz
history.json
config_used.json
```

`structured_core.npz` contains exported latent factors and learned interaction
matrices used by the optional extraction script.
