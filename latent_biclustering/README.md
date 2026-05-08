# Package Map

`latent_biclustering` is the importable implementation package.

```text
data/          sparse matrix IO, synthetic data, preprocessing, features
sampling/      PPR routines and pair construction
models/        encoder, structured core, and combined model
objectives/    loss functions
training/      phase schedule and trainer
inference/     exports, optional extraction adapter, mapping utilities
```

The shortest implementation path starts in:

```text
../scripts/run_synthetic_pipeline.py
../scripts/run_matrix_pipeline.py
```
