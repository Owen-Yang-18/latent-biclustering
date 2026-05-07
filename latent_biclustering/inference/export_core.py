"""Export learned Bernoulli factors and interaction matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from latent_biclustering.models.structured_core import BernoulliStructuredCore


def export_core_npz(
    core: BernoulliStructuredCore,
    path: Union[str, Path],
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    threshold_gamma: float = 0.5,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = core.training
    core.eval()
    with torch.no_grad():
        pa, pb, pg = core.probabilities()
        a_hat, b_hat, gamma_hat = core.hard_factors(threshold_a, threshold_b, threshold_gamma)
        m_prob = (pa * pg.unsqueeze(0)) @ pb.t()
        m_hat = (a_hat * gamma_hat.unsqueeze(0)) @ b_hat.t()
        np.savez_compressed(
            str(path),
            A_prob=pa.detach().cpu().numpy().astype(np.float32),
            B_prob=pb.detach().cpu().numpy().astype(np.float32),
            gamma_prob=pg.detach().cpu().numpy().astype(np.float32),
            A_hat=a_hat.detach().cpu().numpy().astype(np.float32),
            B_hat=b_hat.detach().cpu().numpy().astype(np.float32),
            gamma_hat=gamma_hat.detach().cpu().numpy().astype(np.float32),
            M_prob=m_prob.detach().cpu().numpy().astype(np.float32),
            M_hat=m_hat.detach().cpu().numpy().astype(np.float32),
        )
    core.train(was_training)
