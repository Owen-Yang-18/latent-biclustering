# Scripts

This directory contains command-line entry points for running the source
release.

## `run_synthetic_pipeline.py`

Runs the full model pipeline on a generated sparse object-attribute matrix:

1. generate `X`
2. build the bipartite graph
3. compute sparse random projection node features
4. construct PPR-based training pairs
5. train the GAT encoder and structured Bernoulli interaction core
6. export embeddings and interaction-core artifacts

Usage from `release/`:

```bash
python scripts/run_synthetic_pipeline.py \
  --config configs/synthetic_pipeline.json \
  --device cpu
```

## `synthetic_pipeline.sbatch`

Slurm entry point for the same pipeline. Submit it from the repository root:

```bash
sbatch release/scripts/synthetic_pipeline.sbatch
```

Edit the environment activation block in the script if dependencies are
provided through conda or a virtual environment.

Slurm stdout and stderr are written in the submission directory as
`synthetic_pipeline-<jobid>.out` and `synthetic_pipeline-<jobid>.err`. Model
artifacts are written under `release/outputs/`.
