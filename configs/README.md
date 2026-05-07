# Configurations

Configuration files are JSON dictionaries consumed by
`scripts/run_synthetic_pipeline.py`. Values in a config file can be overridden
from the command line for common fields such as matrix size, embedding
dimension, PPR neighborhood size, and training steps.

## `synthetic_pipeline.json`

Default configuration for running the complete synthetic pipeline. It specifies
the generated sparse matrix, PPR pair construction, model dimensions, structured
core rank, and training length.

Main groups:

- synthetic data: `n_obj`, `n_attr`, `go`, `ga`, `nrel`, `target_sparsity`,
  `background_rate`
- features and embeddings: `feature_dim`, `d_obj`, `d_attr`
- structured core: `kmax`
- pair construction: `ppr_top_t`, `negatives_per_positive`
- optimization: `steps`

## `appendix_defaults.json`

Appendix-style hyperparameter setting for the method. This file records the
larger embedding and training configuration used by the paper-level setup,
including PPR `T=64`, `Nneg=8`, `Kmax=20`, `d1=d2=256`, the 40/35/25 phase
split, and the stated loss weights.

