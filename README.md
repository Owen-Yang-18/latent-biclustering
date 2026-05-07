# Latent Biclustering via Structured Interaction Learning

This source release implements latent biclustering for sparse object-attribute
matrices. Given a sparse matrix `X` with objects as rows and attributes as
columns, the method learns:

- object embeddings `Z`
- attribute embeddings `Y`
- a structured latent interaction matrix `M`

The central modeling assumption is that bicluster structure is expressed in a
joint embedding space. Objects and attributes are first embedded with a
graph-attentive encoder on the bipartite graph induced by `X`. The interaction
matrix is then learned as a sparse structured Bernoulli factorization:

```text
M = A diag(gamma) B^T
```

where `A`, `B`, and `gamma` identify active latent object factors, attribute
factors, and relations between them.

## Pipeline

The implemented pipeline follows the organization of Section 3.

1. Build a weighted bipartite graph from the sparse matrix `X`.
2. Initialize object and attribute node features using sparse random
   projections of their interaction profiles.
3. Construct object-object, attribute-attribute, and object-attribute training
   pairs using Personalized PageRank on the bipartite graph.
4. Train GAT encoders to produce object embeddings `Z` and attribute embeddings
   `Y`.
5. Learn the structured Bernoulli interaction core `M = A diag(gamma) B^T`
   with Gumbel-Sigmoid relaxation and Bernoulli KL regularization.
6. Export learned embeddings, posterior factor probabilities, hard factors, and
   the resulting interaction matrices.

The release focuses on the proposed model. Baseline biclustering methods,
public datasets, plotting utilities, and large experiment launchers are not
bundled with this source tree.

## Repository Layout

```text
release/
  configs/
    synthetic_pipeline.json      default configuration for the synthetic pipeline
    appendix_defaults.json       Appendix E-style hyperparameter setting
    README.md                    configuration guide

  latent_biclustering/
    README.md                    package-level implementation guide

    data/
      sparse_io.py               sparse COO/CSR loading and saving
      synthetic.py               synthetic sparse matrix generator
      graph.py                   bipartite adjacency and PyG edge construction
      features.py                sparse random projection node features

    sampling/
      ppr.py                     Personalized PageRank routines
      pairs.py                   OO, AA, and OA pair construction

    models/
      gat.py                     graph-attentive object/attribute encoder
      structured_core.py         Bernoulli factorization of M
      joint_model.py             encoder plus structured interaction model

    objectives/
      losses.py                  SGNS, OA BCE, KL, and sparsity objectives

    training/
      phases.py                  phase schedule and temperature annealing
      trainer.py                 three-phase optimization loop

    inference/
      export_core.py             export A, B, gamma, and M
      mapping.py                 map latent factors to rows and columns

  scripts/
    run_synthetic_pipeline.py    command-line synthetic pipeline
    synthetic_pipeline.sbatch    Slurm entry point
    README.md                    script usage notes
```

## Installation

The release uses NumPy, SciPy, PyTorch, PyTorch Geometric, and tqdm.

```bash
cd release
pip install -r requirements.txt
```

If these dependencies are already available in the active environment, no
additional installation step is required. On a cluster, make sure the same
environment is active inside the Slurm job.

## Running The Pipeline

From the repository root:

```bash
cd release
python scripts/run_synthetic_pipeline.py \
  --config configs/synthetic_pipeline.json \
  --device cpu
```

From the repository root with Slurm:

```bash
sbatch release/scripts/synthetic_pipeline.sbatch
```

The Slurm script changes into `release/` and writes outputs under:

```text
release/outputs/synthetic_pipeline_${SLURM_JOB_ID}/
```

Slurm stdout and stderr are written in the submission directory as
`synthetic_pipeline-<jobid>.out` and `synthetic_pipeline-<jobid>.err`.

If your Python dependencies are provided by conda or a virtual environment,
edit the activation block in `scripts/synthetic_pipeline.sbatch`.

## Configurations

`configs/synthetic_pipeline.json` controls the synthetic matrix size, planted
relation structure, embedding dimensions, PPR pair construction, structured
core rank `Kmax`, and number of training steps.

`configs/appendix_defaults.json` records the Appendix E-style setting:

- `d1=d2=256`
- `Kmax=20`
- PPR `T=64`
- `Nneg=8`
- `epsilon=1e-5`
- temperature annealing from `1.0` to `0.1`
- three-phase split `40/35/25`
- `lambda_OO=lambda_AA=1.0`
- `beta=1.0`
- `lambda_Z=lambda_Y=0.01`

To run with those settings, pass the appendix config to the same entry point:

```bash
cd release
python scripts/run_synthetic_pipeline.py \
  --config configs/appendix_defaults.json \
  --device cuda
```

## Output Files

Each run writes the following artifacts.

```text
X.npz
synthetic_metadata.npz
Z.npy
Y.npy
structured_core.npz
history.json
config_used.json
```

`X.npz` stores the generated sparse object-attribute matrix in COO form.

`synthetic_metadata.npz` stores planted object memberships, attribute
memberships, and planted object-group/attribute-group relations.

`Z.npy` and `Y.npy` store the learned object and attribute embeddings.

`structured_core.npz` stores:

- `A_prob`, `B_prob`, `gamma_prob`: posterior Bernoulli probabilities
- `A_hat`, `B_hat`, `gamma_hat`: thresholded hard factors
- `M_prob`: posterior mean interaction matrix
- `M_hat`: hard structured interaction matrix

`history.json` stores training losses, phase labels, and temperature values.

## Code Path

The synthetic pipeline is intentionally explicit. The entry point
`scripts/run_synthetic_pipeline.py` performs the following calls:

1. `generate_synthetic(...)`
2. `make_bipartite(...)`
3. `sparse_random_projection_features(...)`
4. `build_ppr_pair_sets(...)`
5. `JointEmbeddingBiclusteringModel(...)`
6. `train_model(...)`
7. `export_core_npz(...)`

This is the shortest path through the implementation and is the best starting
point for tracing how the paper method is realized in code.
