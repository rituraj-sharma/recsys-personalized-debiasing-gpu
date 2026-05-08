"""
Uniform Inverse Propensity Scoring (IPS) loss.

    weight(i) = 1 / e_pop(i)
    L = weight(i) * (-log P(target_item))
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformIPSLoss(nn.Module):
    def __init__(
        self,
        e_pop: Dict[int, float],
        num_items: int,
        max_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_weight = max_weight

        weights = torch.zeros(num_items + 1)
        for item_idx, ep in e_pop.items():
            if 0 < item_idx <= num_items and ep > 0:
                weights[item_idx] = min(1.0 / ep, max_weight)
        self.register_buffer("item_weights", weights)

    def forward(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target = batch["target_item"]

        logits = logits.clone()
        logits[:, 0] = float("-inf")

        ce_loss = F.cross_entropy(logits, target, reduction="none")
        weights = self.item_weights[target]
        weights = weights / (weights.mean() + 1e-8)

        return (weights * ce_loss).mean()
