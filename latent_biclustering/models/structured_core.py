"""Structured Bernoulli interaction core for the latent matrix M."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(p)).item())


class BernoulliStructuredCore(nn.Module):
    """Bernoulli factorization M = A diag(gamma) B^T with relaxed sampling."""

    def __init__(
        self,
        d_obj: int,
        d_attr: int,
        kmax: int,
        prior_a: float = 0.1,
        prior_b: float = 0.1,
        prior_gamma: float = 0.1,
        init_prob: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(int(seed))
        init = _logit(float(init_prob))
        self.logit_a = nn.Parameter(torch.randn((int(d_obj), int(kmax)), generator=g) * 0.01 + init)
        self.logit_b = nn.Parameter(torch.randn((int(d_attr), int(kmax)), generator=g) * 0.01 + init)
        self.logit_gamma = nn.Parameter(torch.randn((int(kmax),), generator=g) * 0.01 + _logit(float(prior_gamma)))
        self.prior_a = float(prior_a)
        self.prior_b = float(prior_b)
        self.prior_gamma = float(prior_gamma)

    @staticmethod
    def _gumbel_sigmoid(logits: torch.Tensor, temperature: float, hard: bool) -> torch.Tensor:
        u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        g = torch.log(u) - torch.log1p(-u)
        y = torch.sigmoid((logits + g) / max(float(temperature), 1e-6))
        if not hard:
            return y
        y_hard = (y >= 0.5).to(y.dtype)
        return y_hard.detach() - y.detach() + y

    def probabilities(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.logit_a), torch.sigmoid(self.logit_b), torch.sigmoid(self.logit_gamma)

    def relaxed_factors(self, temperature: float = 1.0, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            return (
                self._gumbel_sigmoid(self.logit_a, temperature, hard),
                self._gumbel_sigmoid(self.logit_b, temperature, hard),
                self._gumbel_sigmoid(self.logit_gamma, temperature, hard),
            )
        return self.probabilities()

    def matrix(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        a, b, gamma = self.relaxed_factors(temperature=temperature, hard=hard)
        return (a * gamma.unsqueeze(0)) @ b.t()

    def hard_factors(
        self,
        threshold_a: float = 0.5,
        threshold_b: float = 0.5,
        threshold_gamma: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pa, pb, pg = self.probabilities()
        return (
            (pa >= float(threshold_a)).to(pa.dtype),
            (pb >= float(threshold_b)).to(pb.dtype),
            (pg >= float(threshold_gamma)).to(pg.dtype),
        )

    def hard_matrix(
        self,
        threshold_a: float = 0.5,
        threshold_b: float = 0.5,
        threshold_gamma: float = 0.5,
    ) -> torch.Tensor:
        a, b, gamma = self.hard_factors(threshold_a, threshold_b, threshold_gamma)
        return (a * gamma.unsqueeze(0)) @ b.t()

    @staticmethod
    def _kl_bernoulli(q: torch.Tensor, prior: float) -> torch.Tensor:
        p = min(max(float(prior), 1e-6), 1.0 - 1e-6)
        q = q.clamp(1e-6, 1.0 - 1e-6)
        return q * (torch.log(q) - torch.log(torch.tensor(p, device=q.device, dtype=q.dtype))) + (
            1.0 - q
        ) * (torch.log1p(-q) - torch.log(torch.tensor(1.0 - p, device=q.device, dtype=q.dtype)))

    def kl_loss(self, reduction: str = "mean") -> torch.Tensor:
        pa, pb, pg = self.probabilities()
        terms = [
            self._kl_bernoulli(pa, self.prior_a).reshape(-1),
            self._kl_bernoulli(pb, self.prior_b).reshape(-1),
            self._kl_bernoulli(pg, self.prior_gamma).reshape(-1),
        ]
        kl = torch.cat(terms)
        if reduction == "sum":
            return kl.sum()
        if reduction == "none":
            return kl
        return kl.mean()
