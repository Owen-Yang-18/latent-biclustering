#!/bin/bash
#SBATCH --job-name=lb-synthetic-pipeline
#SBATCH --output=synthetic_pipeline-%j.out
#SBATCH --error=synthetic_pipeline-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

REPO_ROOT="/hpc/dctrl/yy398/Sparse-Biclustering"
cd "${REPO_ROOT}/release"
mkdir -p outputs

# Activate the Python environment used for this source release if these
# dependencies are not already available in the batch environment:
# numpy, scipy, torch, torch-geometric.
# source /path/to/conda.sh
# conda activate your-env

python -u scripts/run_synthetic_pipeline.py \
  --config configs/synthetic_pipeline.json \
  --device cpu \
  --output-dir "outputs/synthetic_pipeline_${SLURM_JOB_ID}"
