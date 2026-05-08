# Scripts

Run Python scripts from `release/` unless noted.

## `run_synthetic_pipeline.py`

```bash
python scripts/run_synthetic_pipeline.py \
  --config configs/synthetic_pipeline.json \
  --device cpu
```

## `run_matrix_pipeline.py`

```bash
python scripts/run_matrix_pipeline.py \
  --matrix data/processed/lastfm/X.npz \
  --config configs/realworld_pipeline.json \
  --output-dir outputs/lastfm
```

## `preprocess_realworld.py`

Supported datasets: `retail`, `lastfm`, `movielens`, `amazon`.

```bash
python scripts/preprocess_realworld.py \
  --dataset lastfm \
  --input data/raw/hetrec2011-lastfm-2k/user_artists.dat \
  --output-dir data/processed
```

Dataset download commands are in `docs/real_world_data.md`.

## `extract_latent_biclusters.py`

Requires `requirements-extraction.txt`.

```bash
python scripts/extract_latent_biclusters.py \
  --run-dir outputs/synthetic_pipeline_<jobid> \
  --matrix-key M_hat \
  --method las \
  --n-biclusters 10 \
  --map-to-data
```

## `synthetic_pipeline.sbatch`

Submit from the repository root:

```bash
sbatch release/scripts/synthetic_pipeline.sbatch
```
