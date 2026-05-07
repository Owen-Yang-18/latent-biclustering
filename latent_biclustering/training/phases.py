"""Phase schedule and Gumbel-Sigmoid temperature annealing."""

from __future__ import annotations

from latent_biclustering.config import TrainConfig


def phase_for_step(step: int, config: TrainConfig) -> str:
    steps = max(1, int(config.steps))
    p1 = int(round(steps * float(config.phase1_frac)))
    p2 = p1 + int(round(steps * float(config.phase2_frac)))
    if int(step) < p1:
        return "encoder"
    if int(step) < p2:
        return "core"
    return "joint"


def temperature_for_step(step: int, config: TrainConfig) -> float:
    steps = max(1, int(config.steps) - 1)
    t = min(1.0, max(0.0, float(step) / float(steps)))
    return float(config.temp_start) + t * (float(config.temp_end) - float(config.temp_start))
