"""Configuration dataclasses shared by the pipeline, model, and trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json


@dataclass(frozen=True)
class PairConfig:
    top_t: int = 64
    negatives_per_positive: int = 8
    epsilon: float = 1e-5
    alpha: float = 0.15
    max_iter: int = 50
    tol: float = 1e-8


@dataclass(frozen=True)
class ModelConfig:
    in_dim: int = 32
    hidden_dim: int = 128
    proj_hidden_dim: int = 128
    d_obj: int = 256
    d_attr: int = 256
    heads: int = 4
    dropout: float = 0.1
    kmax: int = 20
    prior_a: float = 0.1
    prior_b: float = 0.1
    prior_gamma: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1600
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_oo: float = 1.0
    lambda_aa: float = 1.0
    beta_kl: float = 1.0
    lambda_z: float = 0.01
    lambda_y: float = 0.01
    phase1_frac: float = 0.40
    phase2_frac: float = 0.35
    temp_start: float = 1.0
    temp_end: float = 0.1
    hard_gumbel: bool = False


def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
