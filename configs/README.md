# Configurations

JSON files in this directory are consumed by the scripts in `../scripts/`.

```text
synthetic_pipeline.json   default synthetic pipeline configuration
realworld_pipeline.json   default configuration for a preprocessed X.npz matrix
appendix_defaults.json    recorded appendix-style hyperparameter setting
```

Common command-line overrides are exposed by the scripts for run length,
matrix size, embedding dimension, PPR neighborhood size, output directory, and
device.
