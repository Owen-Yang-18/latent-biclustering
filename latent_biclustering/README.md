# Package Guide

`latent_biclustering` contains the implementation of the proposed model. The
package is organized by method stage rather than by experiment.

## Data

`data/` turns sparse object-attribute matrices into the graph representation
used by the encoder. It also contains the synthetic generator and sparse random
projection features.

## Sampling

`sampling/` implements Personalized PageRank on the bipartite graph and uses
the resulting scores to form the three training pair families:

- object-object pairs for object embedding structure
- attribute-attribute pairs for attribute embedding structure
- object-attribute pairs for interaction learning

## Models

`models/` contains the graph-attentive encoder and the structured interaction
core. The core is parameterized by Bernoulli logits for `A`, `B`, and `gamma`,
with relaxed samples used during optimization.

## Objectives

`objectives/` contains the losses used by the training curriculum: SGNS for
same-type pairs, BCE for object-attribute pairs, KL regularization for the
Bernoulli factors, and sparsity penalties on the embeddings.

## Training

`training/` implements the curriculum:

1. encoder learning from same-type graph context
2. structured-core learning with fixed embeddings
3. joint optimization of encoder and core

## Inference

`inference/` exports posterior probabilities, hard factors, and learned
interaction matrices. It also provides the latent-to-data mapping utility used
after a latent factor bicluster has been selected.

