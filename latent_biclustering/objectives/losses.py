"""Objective terms used by the three-phase training curriculum."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latent_biclustering.types import ObjectAttributePairs, SameTypePairs


def _device_tensor(array, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def same_type_sgns_loss(emb: torch.Tensor, pairs: SameTypePairs) -> torch.Tensor:
    if pairs.anchor.size == 0:
        return emb.sum() * 0.0
    device = emb.device
    anchor = _device_tensor(pairs.anchor, device, torch.long)
    positive = _device_tensor(pairs.positive, device, torch.long)
    negative = _device_tensor(pairs.negative, device, torch.long)
    weight = _device_tensor(pairs.weight, device, emb.dtype)

    s_pos = (emb[anchor] * emb[positive]).sum(dim=1)
    if negative.numel() == 0:
        s_neg = torch.empty((anchor.numel(), 0), dtype=emb.dtype, device=device)
    else:
        s_neg = (emb[anchor].unsqueeze(1) * emb[negative]).sum(dim=2)
    pos = F.logsigmoid(s_pos)
    neg = F.logsigmoid(-s_neg).sum(dim=1)
    return -(weight * (pos + neg)).mean()


def oa_bce_loss(z: torch.Tensor, y: torch.Tensor, m: torch.Tensor, pairs: ObjectAttributePairs) -> torch.Tensor:
    if pairs.obj.size == 0:
        return (z.sum() + y.sum() + m.sum()) * 0.0
    device = z.device
    obj = _device_tensor(pairs.obj, device, torch.long)
    attr = _device_tensor(pairs.attr, device, torch.long)
    target = _device_tensor(pairs.target, device, z.dtype)
    neg_attr = _device_tensor(pairs.negative_attr, device, torch.long)

    zy = z[obj] @ m
    s_pos = (zy * y[attr]).sum(dim=1)
    loss_pos = F.binary_cross_entropy_with_logits(s_pos, target, reduction="mean")
    if neg_attr.numel() == 0:
        return loss_pos
    s_neg = (zy.unsqueeze(1) * y[neg_attr]).sum(dim=2)
    target_neg = torch.zeros_like(s_neg)
    loss_neg = F.binary_cross_entropy_with_logits(s_neg, target_neg, reduction="mean")
    return loss_pos + loss_neg


def embedding_sparsity_loss(z: torch.Tensor, y: torch.Tensor, lambda_z: float, lambda_y: float) -> torch.Tensor:
    return float(lambda_z) * z.abs().mean() + float(lambda_y) * y.abs().mean()
